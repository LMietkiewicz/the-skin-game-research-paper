"""Training engine: cross-validate ONE ExperimentConfig.

Deliberately narrow. `train_config` trains a single config across the pre-made CV
folds and returns each fold's validation metrics plus its saved checkpoint. It does
NOT carve the test split (Phase 0, run_experiment.py, once per dataset), select a
'best' fold (rejected -- we report the per-arm distribution), touch the test set
(Phase 3, after selection), or own the config receipt (run_experiment.py logs it
with the split metadata). The firewall is structural: this function is never handed
the test set, so it cannot leak it.
"""
import os
from copy import deepcopy
from typing import Any, Dict, List

from torch.utils.data import Subset
from transformers import Trainer, EarlyStoppingCallback, set_seed

from .config import RANDOM_STATE
from .model_init import create_model_for_classification          # confirm module name
from .metrics import make_compute_metrics


def train_config(cfg, num_labels: int, base_training_args, train_full, cv_splits,
                 *, aug_dataset=None, logger=None) -> List[Dict[str, Any]]:
    """Cross-validate `cfg`; return one record per fold.

    cfg                : ExperimentConfig (name, use_augmentation, num_augments,
                         num_train_epochs, early_stopping_patience).
    num_labels         : fixed class count from the shared label space (config-independent).
    base_training_args : TrainingArguments TEMPLATE. Per fold it's deep-copied and given
                         its own output_dir + cfg.num_train_epochs. The template must set
                         eval_strategy="epoch", load_best_model_at_end=True,
                         metric_for_best_model="f1", greater_is_better=True so per-fold
                         early stopping + best-checkpoint restore work on validation.
    train_full         : CLEAN Subset; its .indices map into BOTH the clean and aug
                         datasets (identical sorted order -> a fold index is the same
                         image in each view).
    cv_splits          : list of (train_sub, val_sub) over train_full.
    aug_dataset        : augmenting twin, passed iff cfg.use_augmentation. Mirrored onto
                         the train_full frame ONCE, then Subset per fold (no rebuild).
    logger             : ExperimentLogger built by the caller (so directory topology
                         lives in one place). Optional.

    Returns: [{"fold": k, "val_metrics": {...}, "checkpoint_dir": str}, ...]
    """
    if cfg.use_augmentation and aug_dataset is None:
        raise ValueError(f"cfg '{cfg.name}' has use_augmentation=True but no aug_dataset.")

    compute_metrics = make_compute_metrics(num_labels)
    # Aug twin mirrored once; same indices select the same images (shared sorted order).
    aug_train_full = Subset(aug_dataset, train_full.indices) if aug_dataset is not None else None

    fold_records: List[Dict[str, Any]] = []
    for fold, (train_sub, val_sub) in enumerate(cv_splits, start=1):
        if logger is not None:
            logger.start_fold(fold)

        # Paired heads: seed BEFORE creation so the random classifier head is identical
        # across configs -- aug vs no_aug then differ only by data, not by init.
        set_seed(RANDOM_STATE)
        model = create_model_for_classification(num_labels)

        # Two-view: train from the augmenting twin (same fold indices), validate clean.
        train_view = (Subset(aug_train_full, train_sub.indices)
                      if aug_train_full is not None else train_sub)
        val_view = val_sub

        fold_args = deepcopy(base_training_args)
        fold_args.output_dir = os.path.join(base_training_args.output_dir, cfg.name, f"fold_{fold}")
        fold_args.num_train_epochs = cfg.num_train_epochs

        if logger is not None:
            logger.log_dataset_stats(train_view, f"{cfg.name} fold {fold} train")
            logger.log_dataset_stats(val_view, f"{cfg.name} fold {fold} val")

        callbacks = []
        if cfg.early_stopping_patience:        # 0/None disables early stopping for this config
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

        trainer = Trainer(
            model=model, args=fold_args,
            train_dataset=train_view, eval_dataset=val_view,
            compute_metrics=compute_metrics, callbacks=callbacks,
        )
        trainer.train()
        val_metrics = trainer.evaluate()       # on val_view; best checkpoint already restored

        checkpoint_dir = os.path.join(fold_args.output_dir, "model")
        trainer.save_model(checkpoint_dir)     # persist so Phase 3 scores the 2x2 without retraining

        if logger is not None:
            logger.end_fold(val_metrics, {"checkpoint_dir": checkpoint_dir})
        fold_records.append({"fold": fold, "val_metrics": val_metrics,
                             "checkpoint_dir": checkpoint_dir})

    return fold_records