"""Explainability for the DINOv2 classifier, built on pytorch-grad-cam.

Each Grad-CAM map is scored with a ROAD-Combined faithfulness metric (average of
most-relevant-first and least-relevant-first removal curves, swept over
percentiles). ROADCombined cancels two confounds the single-percentile metric
carries: generic image-degradation (via the LeRF-MoRF difference) and the
arbitrary cutoff (via the percentile average).
Polarity: higher (more positive) == more faithful. 
"""

import os
import random
import warnings
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget, ClassifierOutputSoftmaxTarget)
from pytorch_grad_cam.utils.image import show_cam_on_image


# --- HF <-> library glue ---------------------------------------------------
class _LogitWrapper(nn.Module):
    """Adapt a HF image classifier to the (tensor -> logits tensor) interface
    grad-cam expects (it doesn't want a ModelOutput object)."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.hf_model(pixel_values=pixel_values).logits


def get_target_layer(hf_model: nn.Module) -> nn.Module:
    """Default Grad-CAM target: the last transformer block's first LayerNorm.
    Verified against current transformers (Dinov2Layer.norm1 on the `.dinov2`
    backbone). If a future build renames it, fetch the layer yourself and pass
    it as Explainer(target_layer=...)."""
    return hf_model.dinov2.encoder.layer[-1].norm1


def make_reshape_transform(num_prefix_tokens: int) -> Callable:
    """Turn DINOv2's [B, seq, hidden] tokens into a [B, hidden, g, g] feature map.

    Strips ALL prefix tokens (CLS + any register tokens) and infers the square
    grid from the remaining patch count at runtime -- so it is correct for
    patch-14, register variants, and any input resolution, nothing hardcoded.
    (This is where the old code's patch_size=16 and [1:] assumptions went wrong.)
    """
    def reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
        n_patches = tensor.shape[1] - num_prefix_tokens
        grid = int(round(n_patches ** 0.5))
        if grid * grid != n_patches:
            raise ValueError(
                f"Patch tokens ({n_patches}) are not a square grid; "
                f"check num_prefix_tokens (got {num_prefix_tokens})."
            )
        patches = tensor[:, num_prefix_tokens:, :]
        patches = patches.reshape(tensor.size(0), grid, grid, tensor.size(2))
        return patches.permute(0, 3, 1, 2).contiguous()
    return reshape_transform


def denormalize(image: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Undo the processor's normalization so the displayed image is the TRUE input.
    Returns HWC float32 in [0, 1] (the old code min-max-stretched the normalized
    tensor, which is not the real image)."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    img = (image.detach().cpu() * std_t + mean_t).clamp(0, 1)
    return img.permute(1, 2, 0).numpy().astype(np.float32)


# --- Explainer -------------------------------------------------------------
class Explainer:
    """Grad-CAM explanations for a trained DINOv2 classifier.

    explainer = Explainer(model, processor, device)
    explainer.analyze(test_set, class_names, save_dir="runs/xai")
    """

    def __init__(self, hf_model, processor, device: Union[str, torch.device] = "cuda",
                 target_layer: Optional[nn.Module] = None) -> None:
        self.hf_model = hf_model.to(device).eval()
        self.device = device
        self.image_mean = processor.image_mean
        self.image_std = processor.image_std

        self.wrapped = _LogitWrapper(self.hf_model).to(device).eval()

        num_prefix = 1 + getattr(hf_model.config, "num_register_tokens", 0)
        reshape = make_reshape_transform(num_prefix)
        layer = target_layer if target_layer is not None else get_target_layer(hf_model)
        self.cam = GradCAM(model=self.wrapped, target_layers=[layer],
                           reshape_transform=reshape)
        
    # -- heatmap; returns a [H, W] map in [0, 1]. Extension seam: add methods here. --
    def _gradcam(self, x: torch.Tensor, target: int) -> np.ndarray:
        # CAM generation differentiates the raw class logit (standard, sharper grads).
        cams = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])
        return cams[0]

    # -- faithfulness (ROAD-Combined), defensive so a version mismatch can't kill the run --
    def faithfulness(self, x: torch.Tensor, cam_map: np.ndarray, target: int) -> Optional[float]:
        """ROAD-Combined faithfulness, scored on the class-PROBABILITY change
        (ClassifierOutputSoftmaxTarget -- the target grad-cam's own docs use for
        metrics, since faithfulness should track confidence, not raw logits).
        Higher (more positive) == more faithful.
        """
        try:
            from pytorch_grad_cam.metrics.road import ROADCombined
            metric = ROADCombined(percentiles=[10, 20, 30, 40, 50, 60, 70, 80, 90])
            score = metric(x, cam_map[None, :], [ClassifierOutputSoftmaxTarget(target)], self.wrapped)
            return float(np.asarray(score).reshape(-1)[0])
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"ROAD faithfulness skipped ({type(e).__name__}: {e}). "
                f"Verify the metric signature for your grad-cam version."
            )
            return None

    # -- one figure: input + Grad-CAM overlay(s) --
    def visualize_sample(self, image: torch.Tensor, true_class: int, pred_class: int,
                         class_names: List[str]) -> plt.Figure:
        rgb = denormalize(image, self.image_mean, self.image_std)
        misclassified = true_class != pred_class

        # Which class to explain. A correct call is fully told by the predicted-class
        # map. On a miss we ALSO explain the true class: the predicted map shows what
        # the model latched onto, the true map shows the evidence it should have used
        # -- the artifact-vs-lesion contrast that matters most for the paper.
        roles = [("pred", pred_class)]
        if misclassified:
            roles.append(("true", true_class))

        panels = [("Input", rgb)]
        for role, target in roles:
            x = image.unsqueeze(0).to(self.device).clone().requires_grad_(True)
            cam_map = self._gradcam(x, target)
            overlay = show_cam_on_image(rgb, cam_map, use_rgb=True) / 255.0
            label = "Grad-CAM"
            if misclassified:
                label += f"\n{role}: {class_names[target]}"
            score = self.faithfulness(image.unsqueeze(0).to(self.device), cam_map, target)
            if score is not None:
                label += f"\nROAD: {score:+.3f}"   # higher == more faithful
            panels.append((label, overlay))

        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
        if len(panels) == 1:
            axes = [axes]
        verdict = "correct" if true_class == pred_class else "INCORRECT"
        axes[0].set_title(
            f"True: {class_names[true_class]}\nPred ({verdict}): {class_names[pred_class]}",
            fontsize=10,
        )
        for ax, (label, img) in zip(axes, panels):
            ax.imshow(img)
            if ax is not axes[0]:
                ax.set_title(label, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        return fig

    # -- reproducible analysis over a dataset --
    def analyze(self, dataset, class_names: List[str], n_correct: int = 3,
                n_incorrect: int = 3, seed: int = 42, batch_size: int = 32,
                save_dir: Optional[str] = None) -> None:
        """Visualize a reproducible sample of correct + incorrect predictions.

        `dataset` MUST be the settled held-out test set carved in run.py -- the
        same set Phase 3 scores -- never a training or split-deriving view. This
        method has no way to verify disjointness; the firewall lives at the call
        site. Predictions and overlays both run on this one passed-in dataset, so
        they can't diverge internally.

        Sample selection (n_correct + n_incorrect, seeded) is fixed and disclosed
        on purpose: it's the honest alternative to hand-picking flattering maps.
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        preds, labels = self._collect_predictions(dataset, batch_size)
        correct = [i for i, (p, l) in enumerate(zip(preds, labels)) if p == l]
        incorrect = [i for i, (p, l) in enumerate(zip(preds, labels)) if p != l]
        print(f"{len(correct)} correct, {len(incorrect)} incorrect")

        rng = random.Random(seed)  # reproducible choice of which samples to show
        chosen = (
            [("correct", i) for i in rng.sample(correct, min(n_correct, len(correct)))]
            + [("incorrect", i) for i in rng.sample(incorrect, min(n_incorrect, len(incorrect)))]
        )

        for kind, idx in chosen:
            sample = dataset[idx]
            fig = self.visualize_sample(
                sample["pixel_values"], int(labels[idx]), int(preds[idx]), class_names,
            )
            if save_dir:
                fig.savefig(f"{save_dir}/{kind}_{idx}.png", bbox_inches="tight", dpi=200)
            plt.close(fig)

    def _collect_predictions(self, dataset, batch_size: int):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():  # predictions only -- the explanation passes need gradients
            for batch in loader:
                logits = self.hf_model(pixel_values=batch["pixel_values"].to(self.device)).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                labels.extend(batch["label"].cpu().tolist())
        return preds, labels
