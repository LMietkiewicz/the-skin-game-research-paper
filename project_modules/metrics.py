"""Classification metrics, derived locally from a single confusion matrix.

No `evaluate`/network dependency: precision / recall / f1 / specificity all come
from one sklearn confusion matrix, so metrics are deterministic and version-stable.

ONE definition, used in two places:
  - compute_metrics_from_preds(y_true, y_pred, num_classes)  -- the core.
  - make_compute_metrics(num_classes)                        -- HF Trainer adapter
    (EvalPrediction -> argmax -> core).
So validation (via Trainer) and test (via evaluation.evaluate_model) compute
byte-identical numbers; selection and reporting speak the same language.

Two disciplines encoded here, both deliberate:
  1. Fixed label set. The confusion matrix always spans 0..num_classes-1, so
     matrices are the same shape across folds and across the 2x2 arms (a class
     absent from one test set can't reshape or crash it), and per-class
     specificity correctly sees every other class as a negative.
  2. Support-based exclusion (NOT zero-substitution). A class with zero support
     (no true instances in this eval) is dropped from each macro mean rather than
     scored 0 -- the model had no chance to be evaluated on it. Substituting 0
     would deflate recall AND inflate specificity (an absent class scores a free
     TNR of 1.0). A class that IS present but never predicted still counts,
     scoring precision/f1 = 0: that's a real failure, not an absence.
"""
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def _per_class_counts(cm: np.ndarray):
    """(K, K) confusion matrix [rows=true, cols=pred] -> per-class TP/FP/FN/TN.

    cm[int64] -> (tp, fp, fn, tn), each int64 array of shape (K,).
    """
    tp = np.diag(cm).astype(np.int64)
    fp = cm.sum(axis=0) - tp          # predicted c, truly other
    fn = cm.sum(axis=1) - tp          # truly c, predicted other
    tn = cm.sum() - (tp + fp + fn)
    return tp, fp, fn, tn


def compute_metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray,
                               num_classes: int,
                               class_indices: Optional[Sequence[int]] = None
                               ) -> Dict[str, object]:
    """Macro precision / recall / f1 / specificity from hard predictions.

    y_true[int64, (N,)], y_pred[int64, (N,)] -> dict of macro scalars + per-class.
    recall == sensitivity (TPR); specificity == TNR.

    class_indices: if given, the macro is restricted to those class indices
    (intersected with the classes actually present). This is the sensitivity-
    analysis hook -- e.g. pass the 4 'clean' classes to get the cross-dataset
    gap with akiec/bkl excluded, from the SAME predictions, no re-evaluation.
    The per-class block is always reported over the full K classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    tp, fp, fn, tn = _per_class_counts(cm)

    support = tp + fn                       # true instances per class (row sums)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision   = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        recall      = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)   # sensitivity
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)   # TNR
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    # Macro membership: present (support > 0), optionally narrowed to a subset.
    present = support > 0
    if class_indices is not None:
        keep = np.zeros(num_classes, dtype=bool)
        keep[np.asarray(class_indices, dtype=int)] = True
        present = present & keep

    def _macro(values: np.ndarray) -> float:
        return float(values[present].mean()) if present.any() else 0.0

    return {
        "precision":   _macro(precision),
        "recall":      _macro(recall),         # macro sensitivity
        "f1":          _macro(f1),
        "specificity": _macro(specificity),    # macro TNR
        "n_classes_averaged": int(present.sum()),   # transparency: how many entered the macro
        "per_class": {                              # full K, for reporting + the sensitivity cut
            "precision":   precision.tolist(),
            "recall":      recall.tolist(),
            "f1":          f1.tolist(),
            "specificity": specificity.tolist(),
            "support":     support.tolist(),
        },
    }


# Scalar keys the HF Trainer should log (arrays/nested dicts would break logging).
_SCALAR_KEYS = ("precision", "recall", "f1", "specificity", "n_classes_averaged")


def make_compute_metrics(num_classes: int):
    """HF Trainer adapter: EvalPrediction -> argmax -> compute_metrics_from_preds.

    Returns a callable so num_classes is captured once (every fold's macro spans
    the same K-class set). Strips the per-class block so the Trainer logs only
    flat scalars; the test path calls compute_metrics_from_preds directly to get
    the per-class detail. Same core either way -> identical numbers.
    """
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):       # some models return (logits, ...)
            logits = logits[0]
        y_pred = np.asarray(logits).argmax(axis=-1)
        m = compute_metrics_from_preds(np.asarray(labels), y_pred, num_classes)
        return {k: m[k] for k in _SCALAR_KEYS}
    return compute_metrics
