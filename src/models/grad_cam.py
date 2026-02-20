"""Grad-CAM implementation for visual explanation of defect predictions.

Generates class activation heatmaps highlighting which image regions
contributed most to the model's defect classification decision.
Supports overlay visualization and batch processing.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from PIL import Image

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients, then computes a weighted
    combination to produce a spatial heatmap.

    Args:
        model: The classification model.
        target_layer: The convolutional layer to visualize.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._handles: list = []

        self._handles.append(target_layer.register_forward_hook(self._save_activation))
        self._handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Hook to capture forward activations."""
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Hook to capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, H, W).
            target_class: Class index to explain. If None, uses the
                predicted class.

        Returns:
            Normalized heatmap array of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured")
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = functional.relu(cam)

        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        cam = functional.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        return cam.squeeze().cpu().numpy()

    def generate_batch(
        self,
        input_batch: torch.Tensor,
        target_classes: list[int | None] | None = None,
    ) -> list[np.ndarray]:
        """Generate Grad-CAM heatmaps for a batch of images.

        Args:
            input_batch: Batch tensor of shape (B, 3, H, W).
            target_classes: Per-image target classes. Defaults to predicted.

        Returns:
            List of heatmap arrays, one per image.
        """
        if target_classes is None:
            target_classes = [None] * input_batch.size(0)

        heatmaps: list[np.ndarray] = []
        for i in range(input_batch.size(0)):
            single = input_batch[i : i + 1]
            heatmap = self.generate(single, target_classes[i])
            heatmaps.append(heatmap)
        return heatmaps

    def remove_hooks(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original RGB image as a numpy array (H, W, 3).
        heatmap: Normalized heatmap array (H, W) with values in [0, 1].
        alpha: Blending factor for the overlay (0 = image only, 1 = heatmap only).
        colormap: OpenCV colormap constant for heatmap coloring.

    Returns:
        Blended RGB image as uint8 array.
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = alpha * heatmap_colored.astype(np.float32) + (1 - alpha) * image.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_gradcam_visualization(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    output_path: str | Path,
    alpha: float = 0.5,
) -> Path:
    """Save a Grad-CAM overlay visualization to disk.

    Args:
        original_image: Original RGB image array.
        heatmap: Normalized heatmap from GradCAM.generate().
        output_path: File path to save the visualization.
        alpha: Overlay blending factor.

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = overlay_heatmap(original_image, heatmap, alpha=alpha)
    Image.fromarray(overlay).save(output_path)

    logger.info("Saved Grad-CAM visualization to %s", output_path)
    return output_path
