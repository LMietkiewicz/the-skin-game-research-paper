"""Dataset splitting and split persistence.

All splitting in this pipeline is LESION-GROUPED: images that share a
``lesion_id`` are never separated across a train / val / test boundary. This is
the central methodological safeguard. 

Every datatset passed in is expected to expose (see dataloader.py):
    - ``.targets``     : Sequence[int]  per-sample class index
    - ``.groups``      : Sequence       per-sample lesion id (group key)
    - ``.num_classes`` : int            canonical class count (on the base dataset)
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedGroupKFold

from project_modules.config import N_SPLITS, RANDOM_STATE, TEST_SIZE

PathLike = Union[str, Path]


# --- per-sample attribute extraction (through nested Subsets) --------------
def _resolve(dataset: Dataset, attr: str) -> List[Any]:
    """Return the per-sample sequence ``dataset.<attr>`` (e.g. 'targets',
    'groups'), transparently remapping indices through any nesting of Subsets.

    A Subset stores its parent indices in ``.indices``; we recurse to the parent's
    sequence and gather by those indices, so the result aligns 1:1 with ``dataset``
    in its OWN index space. This is what lets us run CV on the train_full Subset
    that ``create_test_split`` returns.
    """
    if isinstance(dataset, Subset):
        parent = _resolve(dataset.dataset, attr)
        return [parent[i] for i in dataset.indices]
    seq = getattr(dataset, attr, None)
    if seq is not None:
        return list(seq)
    raise AttributeError(
        f"dataset {type(dataset).__name__} exposes no '{attr}'; grouped "
        f"splitting requires both 'targets' and 'groups'."
    )


def _labels(dataset: Dataset) -> List[int]:
    """Per-sample integer class labels (int), Subset-aware."""
    return _resolve(dataset, "targets")


def _groups(dataset: Dataset) -> List[Any]:
    """Per-sample lesion ids (group keys for grouped splitting), Subset-aware."""
    return _resolve(dataset, "groups")


def _base(dataset: Dataset) -> Dataset:
    """Unwrap nested Subsets down to the underlying concrete dataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def _expected_classes(dataset: Dataset, labels: np.ndarray) -> Set[int]:
    """Canonical label set used for fold-coverage checks.

    Prefer the dataset's declared ``num_classes`` (so we also notice a class that
    is absent from the *entire* training pool, not just one fold); fall back to the
    labels actually observed if the attribute is missing.
    """
    base = _base(dataset)
    n = getattr(base, "num_classes", None)
    return set(range(n)) if n is not None else {int(x) for x in labels}


def _warn_missing(fold: int, part: str, present: Set[int], expected: Set[int]) -> None:
    missing = expected - present
    if missing:
        print(f"  [coverage] fold {fold} {part}: classes absent -> {sorted(missing)} "
              f"(metrics nan-exclude them for this fold)")


# --- splitting -------------------------------------------------------------
def create_test_split(
    dataset: Dataset,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[Subset, Subset]:
    """Carve a single lesion-disjoint, stratified held-out test set.

    Implemented as the first fold of a ``StratifiedGroupKFold`` with
    ``n_splits = round(1 / test_size)``.
    The realised test fraction is only *approximately* ``test_size``,
    because whole lesions are kept intact.

    Returns (train_full, test) as Subsets of ``dataset``.
    """
    labels = np.asarray(_labels(dataset))
    groups = np.asarray(_groups(dataset))

    n_splits = round(1.0 / test_size)
    if n_splits < 2:
        raise ValueError(f"test_size={test_size!r} too large; need >= 2 grouped folds.")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(sgkf.split(np.zeros(len(labels)), labels, groups))

    # The splitter guarantees this by construction, but the entire paper is about
    # leakage, so we assert the invariant rather than trust it silently.
    assert set(groups[train_idx]).isdisjoint(groups[test_idx]), \
        "lesion leak: a lesion appears in both train and test."

    return Subset(dataset, train_idx), Subset(dataset, test_idx)


def setup_cross_validation(
    dataset: Dataset,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
    check_coverage: bool = True,
) -> List[Tuple[Subset, Subset]]:
    """Lesion-grouped, stratified k-fold over ``dataset``.

    Returns a list of (train_subset, val_subset), one per fold. Every fold is
    lesion-disjoint between train and val (asserted). With ``check_coverage`` we
    also report any class missing from a fold's train or val side — unlikely at our
    class sizes (smallest is df) but worth surfacing, since a missing class changes
    that fold's macro support.

    NB: we deliberately use a single FIXED seed and do NOT search seeds for nicer
    coverage. Seed-shopping to make folds look balanced is exactly the hidden
    researcher degree-of-freedom this paper criticises; the honest move is one seed
    + disclosure, with absent classes handled downstream by support-based exclusion.
    """
    labels = np.asarray(_labels(dataset))
    groups = np.asarray(_groups(dataset))
    expected = _expected_classes(dataset, labels)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: List[Tuple[Subset, Subset]] = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        if check_coverage:
            _warn_missing(k, "train", {int(x) for x in labels[tr]}, expected)
            _warn_missing(k, "val",   {int(x) for x in labels[va]}, expected)
            assert set(groups[tr]).isdisjoint(groups[va]), \
                f"fold {k}: lesion leak between train and val."
        folds.append((Subset(dataset, tr), Subset(dataset, va)))
    return folds


# --- persistence of the chosen split indices ------------------------------
def get_dataset_indices(dataset: Dataset) -> List[int]:
    """Indices of ``dataset`` within its parent (identity range if not a Subset)."""
    return list(dataset.indices) if isinstance(dataset, Subset) else list(range(len(dataset)))


def _n_groups(dataset: Dataset) -> int:
    """Distinct lesion count in a split (descriptive provenance for the methods section)."""
    try:
        return len(set(_groups(dataset)))
    except AttributeError:
        return -1


def save_dataset_configuration(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    model_path: PathLike,
    n_splits: int = N_SPLITS,
) -> None:
    """Persist split sizes, lesion counts, class names, and exact index assignments."""
    save_dir = Path(model_path)
    base = _base(train_dataset)

    config: Dict[str, Any] = {
        "n_splits": n_splits,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_lesions": _n_groups(train_dataset),
        "val_lesions": _n_groups(val_dataset),
        "test_lesions": _n_groups(test_dataset),
    }
    if hasattr(base, "class_names"):
        config["classes"] = list(base.class_names)

    splits_dict = {
        "train_indices": get_dataset_indices(train_dataset),
        "val_indices": get_dataset_indices(val_dataset),
        "test_indices": get_dataset_indices(test_dataset),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=4)
    torch.save(splits_dict, save_dir / "dataset_splits.pt")
    print(f"Dataset configuration saved to: {save_dir / 'dataset_config.json'}")


def save_final_splits(
    model_path: PathLike,
    train_data: Dataset,
    val_data: Dataset,
    final_test_set: Dataset,
    n_splits: int = N_SPLITS,
) -> None:
    try:
        save_dataset_configuration(train_data, val_data, final_test_set, model_path, n_splits)
    except Exception as e:  # noqa: BLE001
        print(f"Warning: failed to save dataset configuration: {e}")


def load_dataset_configuration(
    model_path: PathLike,
    full_dataset: Dataset,
) -> Tuple[Subset, Subset, Subset, Dict[str, Any]]:
    save_dir = Path(model_path)
    with open(save_dir / "dataset_config.json") as f:
        config = json.load(f)
    splits_dict = torch.load(save_dir / "dataset_splits.pt")
    return (
        Subset(full_dataset, splits_dict["train_indices"]),
        Subset(full_dataset, splits_dict["val_indices"]),
        Subset(full_dataset, splits_dict["test_indices"]),
        config,
    )
