"""Model construction and loading."""
# loading standard modules
import torch
from transformers import Dinov2ForImageClassification

# loading project-specific modules
from .config import MODEL_REPOSITORY, DEVICE


def create_model_for_classification(num_labels: int,
                                    model_repository: str = MODEL_REPOSITORY,
                                    device: torch.device = DEVICE
                                    ) -> Dinov2ForImageClassification:
    """DINOv2-Large classifier with a `num_labels` head, on `device`."""
    model = Dinov2ForImageClassification.from_pretrained(
        model_repository,
        num_labels=num_labels,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,   # reinit head to num_labels (also masks any other mismatch)
    ).float()                           # force fp32 in case a checkpoint loads in reduced precision;
                                        # compatible with fp16 training (Trainer autocasts the forward)
    return model.to(device)


def load_model(model_path: str,
               device: torch.device = DEVICE) -> Dinov2ForImageClassification:
    """Load a fine-tuned checkpoint and set eval mode for inference."""
    model = Dinov2ForImageClassification.from_pretrained(model_path).to(device)
    return model.eval()
