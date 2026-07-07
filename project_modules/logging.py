"""Experiment logging, run-directory authority, and config serialization.

This logger owns ONE timestamped run root per study (the whole 2x2) and hands
out the subpaths every phase writes to. The directory contract it enforces:

    results/run_{ts}/
        config.json          <- the receipt (write_receipt): frame + CONFIGS + env + data meta
        experiment_log.txt    <- this text log
        data/{dataset}/splits.pt          <- Phase 0  (save_splits)
        models/{dataset}/...              <- Phase 1  (train_config writes; output_dir = model_base_dir)
        models/{dataset}/{cfg}/fold_metrics.json   (save_fold_metrics)
        models/{dataset}/selection.json            (save_selection)
        test/grid.json + test/confusion/  <- Phase 3  (save_grid / save_confusion)
        xai/{dataset}/                    <- Phase 3  (Explainer save_dir = xai_dir_for)

Receipt = inputs only (config). Results (fold metrics, selection, grid) live in
their own files, never in the receipt -- that clean split is the reproducibility
discipline the paper argues for.
"""
import time
import json
import platform
import datetime
import importlib.metadata
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import GPUtil
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def _to_jsonable(value: Any) -> Any:
    """Coerce numpy scalars/arrays to native Python types.

    value[np.generic|np.ndarray|Any] -> output[python scalar|list|Any]

    Metrics from sklearn/torch are often np.float32 / np.int64, which json.dump
    can't serialize AND which fail isinstance(v, float), silently dropping out of
    mean/std aggregation. Coercing at ingestion fixes both at once.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _capture_env() -> Dict[str, Any]:
    """Pinned-version snapshot for the receipt. Telemetry-grade: never raises."""
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("torch", "torchvision", "transformers", "grad-cam", "scikit-learn", "numpy"):
        try:
            env[pkg] = importlib.metadata.version(pkg)
        except Exception:  # noqa: BLE001
            env[pkg] = "unknown"
    try:
        env["cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        pass
    return env


class ExperimentLogger:
    """One run root per study; the directory authority every phase writes through.

    Construct once in run.py with base_path=results/. The phases then ask it for
    paths (model_base_dir, xai_dir_for, ...) and hand it artifacts to persist
    (save_splits, save_fold_metrics, save_selection, save_grid, write_receipt).
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = Path(base_path)
        # microsecond stamp -> unique by construction, so exist_ok=False only ever
        # fires on a genuine logic error (a reused logger), never an unlucky restart.
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = self.base_path / f"run_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)

        # scope subdirs (created lazily by the helpers below as each phase needs them)
        self.data_dir = self.run_dir / "data"
        self.models_dir = self.run_dir / "models"
        self.test_dir = self.run_dir / "test"
        self.xai_dir = self.run_dir / "xai"

        self.config_file = self.run_dir / "config.json"
        self.main_log_file = self.run_dir / "experiment_log.txt"
        self.metrics_file = self.run_dir / "metrics.json"

        self._data_meta: Dict[str, Any] = {}             # filled by save_splits, read by write_receipt
        self.fold_metrics: List[Dict[str, Any]] = []     # end_fold only (human-log rollup)
        self.logged_metrics: List[Dict[str, Any]] = []   # log_metrics only
        self.training_start_time: Optional[float] = None
        self.current_fold: Optional[int] = None

    # --- directory authority: phases ask here, never hardcode paths ----------
    def data_dir_for(self, dataset: str) -> Path:
        d = self.data_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_base_dir(self, dataset: str) -> Path:
        """The output_dir run.py hands train_config; it writes {cfg}/fold_k/model under it."""
        d = self.models_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d

    def xai_dir_for(self, dataset: str) -> Path:
        d = self.xai_dir / dataset
        d.mkdir(parents=True, exist_ok=True)
        return d

    # --- Phase 0: split persistence -----------------------------------------
    def save_splits(self, dataset: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        """Persist the bulk index arrays and record the split metadata for the receipt.

        payload -- raw indices, e.g.
                   {"test_indices": [...], "folds": [{"train_indices":[...], "val_indices":[...]}, ...]}
                   kept out of the JSON receipt precisely because it's bulk.
        meta    -- summary the receipt absorbs (sizes, lesion counts, class_names, n_splits).
        """
        torch.save(payload, self.data_dir_for(dataset) / "splits.pt")
        self._data_meta[dataset] = {k: _to_jsonable(v) for k, v in meta.items()}
        self.log_message(f"[phase 0] saved splits for {dataset}: {meta}")

    def load_splits(self, dataset: str) -> Dict[str, Any]:
        return torch.load(self.data_dir_for(dataset) / "splits.pt")

    # --- the receipt: inputs only -------------------------------------------
    def write_receipt(self, *, frame: Dict[str, Any], configs, training_args_template: Dict[str, Any],
                      model_repository: str) -> None:
        """Assemble and write config.json: the study's reproduction inputs.

        frame                  -- N_SPLITS / TEST_SIZE / RANDOM_STATE etc.
        configs                -- the CONFIGS list (the 'lab notebook'); asdict'd here.
        training_args_template -- args.to_dict() of the shared HF TrainingArguments
                                  (per-config deltas live in configs[].num_train_epochs).
        """
        receipt = {
            "frame": frame,
            "model_repository": model_repository,
            "configs": [asdict(c) for c in configs],
            "training_args_template": training_args_template,
            "data": self._data_meta,
            "env": _capture_env(),
        }
        with open(self.config_file, "w") as f:
            json.dump(receipt, f, indent=4, default=_to_jsonable)
        self.log_message("[receipt] wrote config.json")

    # --- Phase 1/2: per-config metrics + selection --------------------------
    def save_fold_metrics(self, dataset: str, cfg_name: str, fold_records: List[Dict[str, Any]]) -> None:
        out = self.model_base_dir(dataset) / cfg_name
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "fold_metrics.json", "w") as f:
            json.dump(fold_records, f, indent=4, default=_to_jsonable)

    def save_selection(self, dataset: str, selection: Dict[str, Any]) -> None:
        with open(self.model_base_dir(dataset) / "selection.json", "w") as f:
            json.dump(selection, f, indent=4, default=_to_jsonable)
        self.log_message(f"[phase 2] {dataset} selected: {selection.get('selected')}")

    # --- Phase 3: the 2x2 + confusion ---------------------------------------
    def save_grid(self, grid: Dict[str, Any]) -> None:
        self.test_dir.mkdir(parents=True, exist_ok=True)
        with open(self.test_dir / "grid.json", "w") as f:
            json.dump(grid, f, indent=4, default=_to_jsonable)
        self.log_message("[phase 3] wrote test/grid.json")

    def save_confusion(self, arm_name: str, cm, class_names) -> None:
        cdir = self.test_dir / "confusion"
        cdir.mkdir(parents=True, exist_ok=True)
        self._plot_confusion_matrix(cm, class_names, cdir / f"{arm_name}.png")

    # --- human-readable logging (unchanged behavior, routed to run_dir) ------
    def log_message(self, message: str) -> None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.main_log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")

    def log_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> None:
        metrics = {k: _to_jsonable(v) for k, v in metrics.items()}
        self.log_message("\nMetrics:")
        for key, value in metrics.items():
            self.log_message(f"{prefix}{key}: {value:.4f}" if isinstance(value, float)
                             else f"{prefix}{key}: {value}")
        self.logged_metrics.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {f"{prefix}{k}": v for k, v in metrics.items()},
        })
        self._save_metrics()

    def log_dataset_stats(self, dataset, split_name: str) -> None:
        distribution = self._get_class_distribution(dataset)
        self.log_message(f"\n{split_name} Dataset Statistics:")
        self.log_message(f"Total samples: {len(dataset)}")
        for class_name, count in distribution.items():
            self.log_message(f"  {class_name}: {count}")
        self._plot_class_distribution(distribution, split_name)

    def start_fold(self, fold_num: int) -> None:
        self.training_start_time = time.time()
        self.current_fold = fold_num
        self.log_message(f"\nStarting Fold {fold_num}")
        self._log_system_state()

    def end_fold(self, metrics: Dict[str, float],
                 additional_info: Optional[Dict[str, Any]] = None) -> None:
        if self.training_start_time is None:
            raise RuntimeError("end_fold called without a matching start_fold.")
        training_time = time.time() - self.training_start_time
        fold_results = {"fold": self.current_fold, "training_time": training_time, **metrics}
        if additional_info:
            fold_results.update(additional_info)
        fold_results = {k: _to_jsonable(v) for k, v in fold_results.items()}
        self.fold_metrics.append(fold_results)
        self.log_message(f"\nFold {self.current_fold} Results:")
        for key, value in fold_results.items():
            self.log_message(f"{key}: {value:.4f}" if isinstance(value, float)
                             else f"{key}: {value}")
        self._save_metrics()

    def log_final_results(self) -> None:
        if not self.fold_metrics:
            return
        numeric_keys = [k for k, v in self.fold_metrics[0].items()
                        if isinstance(v, (int, float)) and k not in ("fold", "training_time")]
        self.log_message("\nFinal Results:")
        for metric in numeric_keys:
            values = [fold[metric] for fold in self.fold_metrics if metric in fold]
            self.log_message(f"Mean {metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # --- internals -----------------------------------------------------------
    def _save_metrics(self) -> None:
        with open(self.metrics_file, "w") as f:
            json.dump({"folds": self.fold_metrics, "logged": self.logged_metrics},
                      f, indent=4, default=_to_jsonable)

    def _log_system_state(self) -> None:
        state: Dict[str, Any] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_memory_used": "N/A",
            "gpu_utilization": "N/A",
        }
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                state["gpu_memory_used"] = f"{gpu.memoryUsed}MB"
                state["gpu_utilization"] = f"{gpu.load * 100:.0f}%"
        except Exception as e:  # noqa: BLE001
            state["gpu_memory_used"] = f"unavailable ({e})"
        self.log_message("\nSystem State:")
        for key, value in state.items():
            self.log_message(f"{key}: {value}")

    def _extract_targets(self, dataset) -> Optional[np.ndarray]:
        if hasattr(dataset, "targets"):
            return np.asarray(dataset.targets)
        base = getattr(dataset, "dataset", None)
        idx = getattr(dataset, "indices", None)
        if base is not None and idx is not None and hasattr(base, "targets"):
            return np.asarray(base.targets)[idx]
        return None

    def _class_names(self, dataset) -> Optional[List[str]]:
        # canonical CLASS_TO_IDX replaced the LabelEncoder; read the dataset property.
        base = getattr(dataset, "dataset", dataset)
        names = getattr(base, "class_names", None)
        return list(names) if names is not None else None

    def _get_class_distribution(self, dataset) -> Dict[str, int]:
        targets = self._extract_targets(dataset)
        if targets is not None:
            unique, counts = np.unique(targets, return_counts=True)
            names = self._class_names(dataset)
            keys = [names[u] if names is not None and u < len(names) else str(u) for u in unique]
            return dict(zip(keys, counts.tolist()))
        labels = [dataset[i]["label"].item() for i in range(len(dataset))]
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(map(str, unique), counts.tolist()))

    def _plot_class_distribution(self, distribution: Dict[str, int], split_name: str) -> None:
        plt.figure(figsize=(12, 6))
        plt.bar(list(distribution.keys()), list(distribution.values()))
        plt.title(f"Class Distribution - {split_name}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.data_dir / f"class_distribution_{split_name}.png")
        plt.close()

    def _plot_confusion_matrix(self, cm, class_names, path: Path) -> None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


def log_model_info(model, logger: ExperimentLogger) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message("\nModel Information:")
    logger.log_message(f"Total parameters: {total:,}")
    logger.log_message(f"Trainable parameters: {trainable:,}")
