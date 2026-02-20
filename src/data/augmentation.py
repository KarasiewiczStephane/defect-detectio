"""Advanced data augmentation strategies for defect detection.

Provides MixUp augmentation and a configurable augmentation pipeline
builder for training-time data diversity.
"""

import logging
import random

import torch

logger = logging.getLogger(__name__)


class MixUp:
    """MixUp augmentation for batch-level regularization.

    Linearly interpolates between pairs of images and their labels
    to create virtual training samples, reducing overfitting.

    Args:
        alpha: Beta distribution parameter controlling interpolation
            strength. Higher values produce more mixing.

    Reference:
        Zhang et al., "mixup: Beyond Empirical Risk Minimization", 2018.
    """

    def __init__(self, alpha: float = 0.4) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha

    def __call__(
        self, batch: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp to a batch of images and labels.

        Args:
            batch: Image tensor of shape (B, C, H, W).
            labels: Label tensor of shape (B,).

        Returns:
            Tuple of (mixed_batch, labels_a, labels_b, lambda) where
            lambda is the interpolation coefficient.
        """
        lam = random.betavariate(self.alpha, self.alpha)
        indices = torch.randperm(batch.size(0))
        mixed_batch = lam * batch + (1 - lam) * batch[indices]
        return mixed_batch, labels, labels[indices], lam


class CutOut:
    """CutOut augmentation that randomly masks square regions.

    Erases a random square patch from each image in a batch,
    encouraging the model to use distributed features.

    Args:
        mask_size: Side length of the square mask in pixels.
        fill_value: Pixel value to fill the masked region.
    """

    def __init__(self, mask_size: int = 32, fill_value: float = 0.0) -> None:
        self.mask_size = mask_size
        self.fill_value = fill_value

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """Apply CutOut to a batch of images.

        Args:
            batch: Image tensor of shape (B, C, H, W).

        Returns:
            Batch with random square regions masked out.
        """
        result = batch.clone()
        _, _, h, w = result.shape
        half = self.mask_size // 2

        for i in range(result.size(0)):
            cy = random.randint(half, h - half)
            cx = random.randint(half, w - half)
            y1, y2 = max(0, cy - half), min(h, cy + half)
            x1, x2 = max(0, cx - half), min(w, cx + half)
            result[i, :, y1:y2, x1:x2] = self.fill_value

        return result
