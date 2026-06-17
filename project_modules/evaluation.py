"""Single source of truth for evaluation.

The canonical numbers come from metrics.compute_metrics_from_preds -- the SAME
core the Trainer uses for validation -- so test and val are computed identically.
classification_report and the confusion-matrix plot are supplementary,
human-readable views, never the source of the reported figures.
"""
# loading standard modules
from textwrap import wrap
from typing import Optional, Sequence
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# loading project-specific modules
from .config import DEVICE
from .metrics import compute_metrics_from_preds


def _unwrap(dataset):
    """Follow .dataset through nested Subsets to the base dataset (for class_names)."""
    while not hasattr(dataset, "class_names") and hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


@torch.no_grad()
def collect_predictions(model, dataset, device=DEVICE, batch_size=32):
    """Run the model once over `dataset`; return (y_true, y_pred) as int arrays."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    for batch in tqdm(loader, desc="predicting"):
        logits = model(batch["pixel_values"].to(device)).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        labels.extend(batch["label"].cpu().numpy())
    return np.array(labels), np.array(preds)


def evaluate_model(model, dataset, num_classes: Optional[int] = None,
                   class_names: Optional[Sequence[str]] = None,
                   class_indices: Optional[Sequence[int]] = None,
                   device=DEVICE, batch_size=32):
    """Full evaluation: predictions + canonical metrics + a text report.

    num_classes / class_names default from the dataset (resolved through Subsets)
    but can be passed explicitly. The returned `metrics` dict is computed by the
    shared core, so it matches validation exactly and is aggregatable across fold
    checkpoints (the per-arm distribution) and re-cuttable to a class subset (the
    akiec/bkl sensitivity analysis) via `class_indices`.
    """
    base = _unwrap(dataset)
    if num_classes is None:
        num_classes = base.num_classes
    if class_names is None:
        class_names = base.class_names
    class_names = [str(c) for c in class_names]

    y_true, y_pred = collect_predictions(model, dataset, device, batch_size)

    # canonical numbers -- same core as validation
    metrics = compute_metrics_from_preds(y_true, y_pred, num_classes, class_indices)

    # supplementary, human-readable only; labels pinned so a missing class can't crash it
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)),
        target_names=class_names, zero_division=0,
    )
    return {
        "y_true": y_true, "y_pred": y_pred,
        "num_classes": num_classes, "class_names": class_names,
        "metrics": metrics, "report": report,
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path,
                          show_percentages=False, figsize=(8, 7), dpi=200):
    """Confusion-matrix heatmap. Labels pinned to the full class set so the matrix
    is always K x K and comparable across folds / arms (even if a class is absent).
    `show_percentages=True` annotates count + row %."""
    class_names = [str(c) for c in class_names]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    wrapped = ["\n".join(wrap(c, 20)) for c in class_names]

    plt.figure(figsize=figsize)
    if show_percentages:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                p = cm_pct[i, j]
                annot[i, j] = f"{cm[i, j]}\n({0.0 if np.isnan(p) else p:.1f}%)"
        sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues",
                    xticklabels=wrapped, yticklabels=wrapped)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=wrapped, yticklabels=wrapped)

    plt.xticks(rotation=90, ha="center")
    plt.yticks(rotation=0, va="center")
    plt.title("Confusion Matrix", pad=20)
    plt.xlabel("Predicted", labelpad=20)
    plt.ylabel("True", labelpad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()
    return cm