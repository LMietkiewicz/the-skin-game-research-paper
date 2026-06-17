"""Experiment entry point: orchestrate the 2x2 cross-dataset study.

Phase 0  split each dataset ONCE (lesion-grouped holdout + CV), persist.
Phase 1  train every CONFIG on each dataset's folds (train_config).
Phase 2  select one config per dataset on mean validation f1.
Phase 3  score the 2x2 once (per-arm distribution over fold checkpoints +
         akiec/bkl sensitivity cut), then Grad-CAM/ROAD on the selected models.

The firewall is structural: the test holdouts are built in Phase 0 and not
touched until Phase 3, after selection. train_config is never handed a test set.

Run from project root:  python run_experiment.py
"""
import gc

import numpy as np
import torch
from transformers import TrainingArguments, AutoImageProcessor, set_seed

from project_modules.config import (
    N_SPLITS, TEST_SIZE, RANDOM_STATE, PATH_BASE, MODEL_REPOSITORY, DEVICE,
    ExperimentConfig,
)
from project_modules.dataloader import (
    build_dataset, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES,
)
from project_modules.splits import (
    create_test_split, setup_cross_validation, get_dataset_indices,
)
from project_modules.training import train_config
from project_modules.metrics import compute_metrics_from_preds
from project_modules.evaluation import evaluate_model, plot_confusion_matrix
from project_modules.model_init import load_model
from project_modules.logging import ExperimentLogger
from project_modules.xai import Explainer


# ===========================================================================
# Run-time decisions (the §4 open calls, surfaced as switches)
# ===========================================================================
RESULTS_ROOT = "results"
DATASETS = ["HAM10000", "BCN20000"]

# The lab notebook. Edit freely; each instance is a frozen, self-documenting run.
CONFIGS = [
    ExperimentConfig(name="no_aug", use_augmentation=False),
    ExperimentConfig(name="aug",    use_augmentation=True, num_augments=5),
]

PRIMARY_METRIC = "f1"          # config selection key (macro-F1). HF logs it as eval_f1.

# Off-diagonal (cross) test target. "held_out" -> the foreign dataset's own test
# holdout (clean column-comparability: every column is the same samples for both
# rows). "full_foreign" -> the entire foreign set (more power for the cross gap).
# Either is defensible and disclosed; recorded per-arm in grid.json.
OFF_DIAGONAL_TARGET = "held_out"

# XAI explains ONE model per train dataset. We reject best-fold selection, so use
# the MEDIAN-val fold (a representative, non-optimistic pick), on its own
# in-distribution test holdout.
XAI_N_CORRECT, XAI_N_INCORRECT, XAI_SEED = 3, 3, 42

# Compute knobs (tune for your 5090 / Blackwell; bf16 over fp16 on sm_120).
BATCH_SIZE, LEARNING_RATE = 16, 5e-5
USE_BF16 = True


def build_augment():
    """uint8 [3,H,W] -> uint8 [3,H,W], applied BEFORE the processor (which then
    resizes + normalizes), so augmented and clean samples normalize identically.
    This policy is a DISCLOSED config axis -- tune it; it's not load-bearing here.
    NOTE: num_augments is unused under on-the-fly augmentation (fresh per epoch);
    it's carried in the receipt for record-keeping, not consumed.
    """
    from torchvision.transforms import v2
    return v2.Compose([
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.5),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    ])


def _validate_config():
    if PATH_BASE is ... or PATH_BASE is None:
        raise SystemExit("Set PATH_BASE in project_modules/config.py before running.")


def _val_metric(metrics, key):
    """train_config returns trainer.evaluate() dicts (HF prefixes keys 'eval_')."""
    return metrics.get(f"eval_{key}", metrics.get(key))


def _median_fold(records):
    """Index of the median-val-f1 fold -- the representative model for XAI/CM."""
    vals = [_val_metric(r["val_metrics"], PRIMARY_METRIC) for r in records]
    return int(np.argsort(vals)[len(vals) // 2])


def _arm_test_set(test_name, train_name, holdouts, clean):
    """The dataset an arm scores on, plus a self-describing tag for grid.json."""
    if train_name == test_name:
        return holdouts[test_name], f"{test_name}:test_holdout (in-distribution)"
    if OFF_DIAGONAL_TARGET == "full_foreign":
        return clean[test_name], f"{test_name}:full (cross, full foreign)"
    return holdouts[test_name], f"{test_name}:test_holdout (cross, held-out foreign)"


def main():
    _validate_config()
    set_seed(RANDOM_STATE)
    logger = ExperimentLogger(base_path=RESULTS_ROOT)
    logger.log_message(f"=== study start: {DATASETS}, configs={[c.name for c in CONFIGS]} ===")

    clean = {name: build_dataset(name) for name in DATASETS}          # clean full views
    needs_aug = any(c.use_augmentation for c in CONFIGS)
    aug_views = ({name: build_dataset(name, augment=build_augment()) for name in DATASETS}
                 if needs_aug else {})

    # ---- Phase 0: split once per dataset, persist (shared by every config) ----
    holdouts, train_fulls, cv_by_dataset = {}, {}, {}
    for name in DATASETS:
        ds = clean[name]
        train_full, test = create_test_split(ds)                     # Subsets of ds
        cv_splits = setup_cross_validation(train_full)               # (tr, va) over train_full
        holdouts[name], train_fulls[name], cv_by_dataset[name] = test, train_full, cv_splits

        groups = np.asarray(ds.groups)
        folds_payload, fold_sizes = [], []
        for tr, va in cv_splits:
            # resolve fold positions (into train_full) back to BASE dataset indices
            tr_base = [train_full.indices[i] for i in tr.indices]
            va_base = [train_full.indices[i] for i in va.indices]
            folds_payload.append({"train_indices": tr_base, "val_indices": va_base})
            fold_sizes.append({"train": len(tr), "val": len(va)})

        logger.save_splits(name,
            payload={"test_indices": get_dataset_indices(test),
                     "train_full_indices": get_dataset_indices(train_full),
                     "folds": folds_payload},
            meta={"train_full_size": len(train_full), "test_size": len(test),
                  "n_splits": N_SPLITS,
                  "n_lesions_train_full": int(len(set(groups[train_full.indices]))),
                  "n_lesions_test": int(len(set(groups[test.indices]))),
                  "class_names": list(ds.class_names), "fold_sizes": fold_sizes})
        logger.log_dataset_stats(test, f"{name}_test")

    # ---- shared TrainingArguments template (per-fold output_dir + epochs set in train_config) ----
    base_training_args = TrainingArguments(
        output_dir=str(logger.models_dir),
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=PRIMARY_METRIC, greater_is_better=True,
        num_train_epochs=1,                          # overridden per cfg inside train_config
        per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_total_limit=1, logging_steps=50, report_to="none",
        bf16=USE_BF16, seed=RANDOM_STATE,
        remove_unused_columns=False,                 # keep pixel_values/label (default collator maps label->labels)
    )

    # receipt now: data metadata is populated, template is pre-mutation
    logger.write_receipt(
        frame={"N_SPLITS": N_SPLITS, "TEST_SIZE": TEST_SIZE, "RANDOM_STATE": RANDOM_STATE,
               "datasets": DATASETS, "num_classes": NUM_CLASSES,
               "class_to_idx": CLASS_TO_IDX, "off_diagonal_target": OFF_DIAGONAL_TARGET},
        configs=CONFIGS, training_args_template=base_training_args.to_dict(),
        model_repository=MODEL_REPOSITORY)

    # ---- Phase 1: train every config on each dataset's folds ----
    records_by = {name: {} for name in DATASETS}
    for name in DATASETS:
        base_training_args.output_dir = str(logger.model_base_dir(name))   # models/{name}
        for cfg in CONFIGS:
            aug = aug_views.get(name) if cfg.use_augmentation else None
            records = train_config(cfg, NUM_CLASSES, base_training_args,
                                   train_fulls[name], cv_by_dataset[name],
                                   aug_dataset=aug, logger=logger)
            logger.save_fold_metrics(name, cfg.name, records)
            records_by[name][cfg.name] = records

    # ---- Phase 2: select one config per dataset on mean val f1 ----
    selected = {}
    for name in DATASETS:
        means = {cfg.name: float(np.mean([_val_metric(r["val_metrics"], PRIMARY_METRIC)
                                          for r in records_by[name][cfg.name]]))
                 for cfg in CONFIGS}
        winner = max(means, key=means.get)
        selected[name] = winner
        logger.save_selection(name, {"selected": winner, "metric": PRIMARY_METRIC,
                                     "per_config_mean_val": means})

    # ---- Phase 3: score the 2x2 once + the sensitivity cut ----
    clean4 = [i for i in range(NUM_CLASSES)
              if i not in (CLASS_TO_IDX["akiec"], CLASS_TO_IDX["bkl"])]   # bcc, df, mel, nv
    grid = {"off_diagonal_target": OFF_DIAGONAL_TARGET, "clean4_class_indices": clean4, "arms": []}
    median_ck = {}   # per train dataset, reused for confusion + XAI

    for train_name in DATASETS:
        recs = records_by[train_name][selected[train_name]]
        ckpts = [r["checkpoint_dir"] for r in recs]
        med = _median_fold(recs)
        median_ck[train_name] = ckpts[med]

        for test_name in DATASETS:
            test_set, test_desc = _arm_test_set(test_name, train_name, holdouts, clean)
            scores, cm_source = [], None
            for i, ck in enumerate(ckpts):
                model = load_model(ck)
                res = evaluate_model(model, test_set, num_classes=NUM_CLASSES,
                                     class_names=IDX_TO_CLASS)
                cut = compute_metrics_from_preds(res["y_true"], res["y_pred"],
                                                 NUM_CLASSES, class_indices=clean4)
                m = res["metrics"]
                scores.append({"fold": recs[i]["fold"], "checkpoint": ck,
                               "f1": m["f1"], "recall": m["recall"],
                               "precision": m["precision"], "specificity": m["specificity"],
                               "n_classes_averaged": m["n_classes_averaged"],
                               "f1_clean4": cut["f1"], "recall_clean4": cut["recall"]})
                if i == med:
                    cm_source = res          # representative CM from the median fold
                del model
                gc.collect(); torch.cuda.empty_cache()

            arm = f"train-{train_name}_test-{test_name}"
            plot_confusion_matrix(cm_source["y_true"], cm_source["y_pred"], IDX_TO_CLASS,
                                  logger.test_dir / "confusion" / f"{arm}.png",
                                  show_percentages=True)
            grid["arms"].append({
                "train": train_name, "test": test_name,
                "kind": "in_distribution" if train_name == test_name else "cross",
                "config": selected[train_name], "test_set": test_desc,
                "checkpoints": ckpts, "scores": scores})

    logger.test_dir.mkdir(parents=True, exist_ok=True)
    (logger.test_dir / "confusion").mkdir(parents=True, exist_ok=True)
    logger.save_grid(grid)

    # ---- Phase 3 (XAI): median model per dataset, on its in-distribution holdout ----
    processor = AutoImageProcessor.from_pretrained(MODEL_REPOSITORY)
    for train_name in DATASETS:
        model = load_model(median_ck[train_name])
        Explainer(model, processor, device=DEVICE).analyze(
            holdouts[train_name], IDX_TO_CLASS,
            n_correct=XAI_N_CORRECT, n_incorrect=XAI_N_INCORRECT, seed=XAI_SEED,
            save_dir=str(logger.xai_dir_for(train_name)))
        del model
        gc.collect(); torch.cuda.empty_cache()

    logger.log_message(f"=== study complete: {logger.run_dir} ===")
    print(f"Done. Results in {logger.run_dir}")


if __name__ == "__main__":
    main()