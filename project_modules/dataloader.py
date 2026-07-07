"""Dataset classes and loader for the dermoscopic skin-lesion datasets.

Two datasets (HAM10000, BCN20000) over ONE shared 6-class label space -- their
intersection {akiec, bcc, bkl, df, mel, nv} -- so a model trained on one can be
evaluated on the other (the 2x2 cross-dataset design).

Encoding is a FIXED canonical map (CLASS_TO_IDX), not a data-fit LabelEncoder,
so 'mel' is index 4 in EVERY dataset regardless of load order. That's what makes
cross-dataset labels line up; a per-dataset encoder could silently disagree.
"""
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from .config import PATH_BASE, MODEL_REPOSITORY


# --- shared canonical label space (the HAM10000 / BCN20000 intersection) ---
# Fixed so class indices are identical across datasets. Any diagnosis not in
# this map (HAM's vasc; BCN's metastasis/SCC/scar/unlabeled) is dropped at load.
CLASS_TO_IDX: Dict[str, int] = {
    "akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5,
}
IDX_TO_CLASS: List[str] = sorted(CLASS_TO_IDX, key=CLASS_TO_IDX.get)   # index order
NUM_CLASSES: int = len(CLASS_TO_IDX)


class BaseSkinDataset(Dataset):
    """Shared mechanics for the CSV-driven dermoscopic datasets.

    A subclass builds and passes up a normalized frame with three columns:
        image_id[str] -- file stem; image is <images_dir>/<image_id>.jpg
        label[int]    -- canonical class index (CLASS_TO_IDX)
        group[str]    -- lesion_id, for lesion-grouped splitting
    Rows whose diagnosis isn't in CLASS_TO_IDX must already be filtered out.
    """

    def __init__(self, frame: pd.DataFrame, images_dir: str,
                model_repository: str, augment: Optional[Callable] = None) -> None:
        # Final chokepoint: guarantee no NaN in ANY used column (image_id / label /
        # group). Subclasses already drop unmapped labels (needed before the int
        # cast); this re-checks all three so a missing image_id or lesion_id can
        # never reach grouped splitting as a silent singleton group. 
        used = ["image_id", "label", "group"]
        n_before = len(frame)
        frame = frame.dropna(subset=used)
        if n_before - len(frame):
            print(f"[{type(self).__name__}] dropped {n_before - len(frame)} row(s) "
                f"with NaN in a used column ({n_before} -> {len(frame)})")
        # Canonical sort so row i is deterministic across instances and datasets. 
        self.frame = frame.sort_values("image_id").reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = AutoImageProcessor.from_pretrained(model_repository)
        self.augment = augment

    @property
    def targets(self) -> np.ndarray:
        """Encoded labels, no image I/O. -> output[int64, (N,)]"""
        return self.frame["label"].to_numpy()

    @property
    def groups(self) -> np.ndarray:
        """Per-sample lesion ids for grouped splitting. -> output[object, (N,)]"""
        return self.frame["group"].to_numpy()

    @property
    def class_names(self) -> List[str]:
        """Class names in index order (for confusion matrices / logging)."""
        return IDX_TO_CLASS

    @property
    def num_classes(self) -> int:
        return NUM_CLASSES

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        img_path = os.path.join(self.images_dir, f"{row['image_id']}.jpg")
        image = read_image(img_path, mode=ImageReadMode.RGB)   # -> uint8 [3, H, W], forced 3-channel
        if self.augment is not None:
            image = self.augment(image)                        # uint8 [3,H,W] -> uint8 [3,H,W]
        # processor: uint8 [3,H,W] -> float32 [3,H,W] (resized + normalized)
        image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        label = torch.tensor(int(row["label"]), dtype=torch.long)   # -> int64 scalar
        return {"pixel_values": image, "label": label}


class HAM10000(BaseSkinDataset):
    """HAM10000, restricted to the shared 6-class space (vasc dropped).

    Reads `metadata_path` (columns image_id, dx, lesion_id); images at
    <images_dir>/<image_id>.jpg.
    """

    def __init__(self, metadata_path: str, images_dir: str,
                 model_repository: str, augment: Optional[Callable] = None) -> None:
        meta = pd.read_csv(metadata_path, usecols=["image_id", "dx", "lesion_id"])
        # dx -> canonical index; vasc isn't in the map -> NaN -> dropped.
        meta["label"] = meta["dx"].map(CLASS_TO_IDX)
        meta = meta.dropna(subset=["label"]).copy()
        meta["label"] = meta["label"].astype(np.int64)
        frame = (meta.rename(columns={"lesion_id": "group"})
                     [["image_id", "label", "group"]])
        super().__init__(frame, images_dir, model_repository, augment)


# BCN diagnosis_3 -> shared-class abbreviation (derived from HAM10000). Diagnoses NOT listed here
# (Melanoma metastasis, Squamous cell carcinoma NOS, Scar, NaN) are intentionally
# absent and therefore dropped at load.
BCN_DIAGNOSIS_TO_CLASS: Dict[str, str] = {
    "Nevus": "nv",
    "Melanoma, NOS": "mel",
    "Basal cell carcinoma": "bcc",
    "Seborrheic keratosis": "bkl",
    "Solar lentigo": "bkl",
    "Solar or actinic keratosis": "akiec",
    "Dermatofibroma": "df",
}


class BCN20000(BaseSkinDataset):
    """BCN20000, mapped to the shared 6-class space.

    Reads `metadata_path` (columns isic_id, diagnosis_3, lesion_id); images at
    <images_dir>/<isic_id>.jpg.
    """

    def __init__(self, metadata_path: str, images_dir: str,
                 model_repository: str, augment: Optional[Callable] = None) -> None:
        meta = pd.read_csv(metadata_path, usecols=["isic_id", "diagnosis_3", "lesion_id"])
        # diagnosis_3 -> abbreviation -> canonical index; anything unmapped -> dropped.
        abbrev = meta["diagnosis_3"].map(BCN_DIAGNOSIS_TO_CLASS)
        meta["label"] = abbrev.map(CLASS_TO_IDX)
        meta = meta.dropna(subset=["label"]).copy()
        meta["label"] = meta["label"].astype(np.int64)
        frame = (meta.rename(columns={"isic_id": "image_id", "lesion_id": "group"})
                     [["image_id", "label", "group"]])
        super().__init__(frame, images_dir, model_repository, augment)


# ---------------------------------------------------------------------------
# Construction. Path specs live here so both the clean loader and the training
# code build views the same way. Pass augment=<transform> for the per-fold
# training view; leave it None for clean (val/test, and split-deriving) views.
# ---------------------------------------------------------------------------
_DATASET_BUILDERS = {
    "HAM10000": lambda augment: HAM10000(
        metadata_path=os.path.join(PATH_BASE, "HAM10000", "HAM10000_metadata.csv"),
        images_dir=os.path.join(PATH_BASE, "HAM10000"),  # images sit straight in this dir
        model_repository=MODEL_REPOSITORY, augment=augment,
    ),
    "BCN20000": lambda augment: BCN20000(
        metadata_path=os.path.join(PATH_BASE, "BCN20000", "bcn20000_metadata_2026-06-14.csv"),
        images_dir=os.path.join(PATH_BASE, "BCN20000"),   # images sit straight in this dir
        model_repository=MODEL_REPOSITORY, augment=augment,
    ),
}


def build_dataset(name: str, augment: Optional[Callable] = None) -> BaseSkinDataset:
    """Build a single dataset view. augment=None -> clean; augment=<fn> -> training view."""
    return _DATASET_BUILDERS[name](augment)


def load_datasets() -> Tuple[Dict[str, BaseSkinDataset], Dict[str, int]]:
    """Clean (un-augmented) views, for deriving splits and for val/test.
    The augmenting train-view is built per fold in training via build_dataset()."""
    datasets: Dict[str, BaseSkinDataset] = {name: build_dataset(name) for name in _DATASET_BUILDERS}
    n_labels: Dict[str, int] = {name: ds.num_classes for name, ds in datasets.items()}
    return datasets, n_labels
