"""
Project configuration.

Two layers: the evaluation *protocol* is fixed so
cross-config remains static, while the
*model/training knobs* are free to vary during the validation-phase.

  - Frame (module-level constants below): splits, seed, paths, device. Held
    constant across every config. RANDOM_STATE in particular is shared so
    StratifiedKFold produces identical folds for every config.

  - ExperimentConfig: every knob that defines a run. Each instance is frozen
    only so its serialized record can't drift from what actually ran.

No tensors or GPU allocation happen at import; this module is side-effect free.

Note: dataloader.py and training.py will receive an ExperimentConfig and read
knobs from it (e.g. cfg.use_augmentation) instead of importing them from here.
That cfg-threading lands when we revisit those modules.
"""
from dataclasses import dataclass

import torch

# --- frame: the evaluation protocol (fixed across all configs) ---
N_SPLITS: int = 5                # StratifiedKFold folds over the non-test pool
TEST_SIZE: float = 0.1           # stratified holdout fraction (create_test_split)
RANDOM_STATE: int = 42           # shared → identical splits across configs (paired comparison)

PATH_BASE = ...                  # REQUIRED: set to the dir containing /datasets.
MODEL_REPOSITORY: str = "facebook/dinov2-large"

# A device *descriptor* — creating this touches no GPU memory, so it's a safe
# top-level declaration rather than an allocation.
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- model/training knobs: free to vary during validation exploration ---
@dataclass(frozen=True)
class ExperimentConfig:
    """
    One run's worth of model/training settings.

    Frozen so a serialized instance is a faithful, immutable record of what
    produced a given run. 

    name[str]                    -> stable id; used for output dirs / log labels.
    use_augmentation[bool]       -> selects the augmenting "twin" dataset view.
    num_augments[int]            -> augmented copies per image; ignored when
                                    use_augmentation is False.
    num_train_epochs[int]        -> fixed training budget for the run.
    early_stopping_patience[int] -> epochs without val improvement before stop.
    """
    name: str
    use_augmentation: bool
    num_augments: int = 5
    num_train_epochs: int = 15
    early_stopping_patience: int = 3
