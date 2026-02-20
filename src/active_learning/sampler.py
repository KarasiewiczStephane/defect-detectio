"""Uncertainty sampling for active learning.

Identifies the most uncertain predictions from a model, enabling
targeted human review of ambiguous samples to maximize labeling
efficiency.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class UncertaintySample:
    """A single sample flagged for human review.

    Attributes:
        index: Index in the original dataset.
        image_path: File path to the image.
        uncertainty: Normalized entropy score in [0, 1].
        predicted_class: Model's predicted class index.
        probabilities: Full softmax probability distribution.
    """

    index: int
    image_path: str
    uncertainty: float
    predicted_class: int
    probabilities: np.ndarray


class UncertaintySampler:
    """Selects the most uncertain samples for human labeling.

    Uses entropy-based uncertainty to identify predictions where the
    model is least confident, prioritizing them for review.

    Args:
        model: Trained classification model.
        threshold: Minimum normalized entropy to consider a sample uncertain.
        device: Device for model inference.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.6,
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.model.eval()
        self.threshold = threshold
        self.device = device

    def compute_uncertainty(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute normalized entropy as an uncertainty measure.

        Args:
            probs: Softmax probability tensor of shape (B, C).

        Returns:
            Uncertainty scores in [0, 1] of shape (B,).
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        max_entropy = np.log(probs.shape[1])
        return entropy / max_entropy

    def select_uncertain_samples(
        self,
        dataloader: DataLoader,
        n_samples: int,
        image_paths: list[str] | None = None,
    ) -> list[UncertaintySample]:
        """Select the N most uncertain samples from a dataset.

        Args:
            dataloader: DataLoader to scan for uncertain predictions.
            n_samples: Maximum number of samples to select.
            image_paths: Optional list of file paths corresponding to
                dataset indices. Used to populate UncertaintySample.image_path.

        Returns:
            List of UncertaintySample objects sorted by descending uncertainty.
        """
        uncertainties: list[float] = []
        indices: list[int] = []
        all_probs: list[np.ndarray] = []
        all_preds: list[int] = []

        with torch.no_grad():
            idx = 0
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                uncertainty = self.compute_uncertainty(probs)

                for i in range(len(images)):
                    uncertainties.append(uncertainty[i].item())
                    indices.append(idx + i)
                    all_probs.append(probs[i].cpu().numpy())
                    all_preds.append(outputs[i].argmax().item())
                idx += len(images)

        # Filter by threshold and sort by uncertainty descending
        candidates = [
            (u, i, p, pred)
            for u, i, p, pred in zip(uncertainties, indices, all_probs, all_preds, strict=True)
            if u >= self.threshold
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:n_samples]

        samples = []
        for u, i, p, pred in selected:
            path = image_paths[i] if image_paths and i < len(image_paths) else f"sample_{i}"
            samples.append(
                UncertaintySample(
                    index=i,
                    image_path=str(path),
                    uncertainty=u,
                    predicted_class=pred,
                    probabilities=p,
                )
            )

        logger.info(
            "Selected %d uncertain samples (threshold=%.2f, scanned=%d)",
            len(samples),
            self.threshold,
            len(uncertainties),
        )
        return samples
