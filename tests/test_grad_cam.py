"""Tests for Grad-CAM defect localization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.grad_cam import (
    GradCAM,
    overlay_heatmap,
    save_gradcam_visualization,
)
from src.models.resnet_classifier import BaselineCNN, DefectResNet


@pytest.fixture()
def baseline_model() -> BaselineCNN:
    """Create a baseline CNN for testing."""
    return BaselineCNN(num_classes=2)


@pytest.fixture()
def resnet_model() -> DefectResNet:
    """Create a ResNet model (no pretrained weights for speed)."""
    return DefectResNet(num_classes=2, pretrained=False)


@pytest.fixture()
def dummy_input() -> torch.Tensor:
    """Single image input tensor."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture()
def dummy_image() -> np.ndarray:
    """Random RGB image as numpy array."""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


class TestGradCAM:
    """Tests for the GradCAM class."""

    def test_heatmap_shape_baseline(
        self, baseline_model: BaselineCNN, dummy_input: torch.Tensor
    ) -> None:
        """Heatmap should match input spatial dimensions."""
        target_layer = baseline_model.get_target_layer()
        grad_cam = GradCAM(baseline_model, target_layer)
        heatmap = grad_cam.generate(dummy_input)
        assert heatmap.shape == (64, 64)
        grad_cam.remove_hooks()

    def test_heatmap_shape_resnet(self, resnet_model: DefectResNet) -> None:
        """Heatmap should match input spatial dimensions for ResNet."""
        target_layer = resnet_model.get_target_layer()
        grad_cam = GradCAM(resnet_model, target_layer)
        x = torch.randn(1, 3, 224, 224)
        heatmap = grad_cam.generate(x)
        assert heatmap.shape == (224, 224)
        grad_cam.remove_hooks()

    def test_heatmap_normalized(
        self, baseline_model: BaselineCNN, dummy_input: torch.Tensor
    ) -> None:
        """Heatmap values should be in [0, 1]."""
        target_layer = baseline_model.get_target_layer()
        grad_cam = GradCAM(baseline_model, target_layer)
        heatmap = grad_cam.generate(dummy_input)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        grad_cam.remove_hooks()

    def test_specific_target_class(
        self, baseline_model: BaselineCNN, dummy_input: torch.Tensor
    ) -> None:
        """Should accept explicit target class."""
        target_layer = baseline_model.get_target_layer()
        grad_cam = GradCAM(baseline_model, target_layer)
        heatmap_0 = grad_cam.generate(dummy_input, target_class=0)
        heatmap_1 = grad_cam.generate(dummy_input, target_class=1)
        assert heatmap_0.shape == (64, 64)
        assert heatmap_1.shape == (64, 64)
        grad_cam.remove_hooks()

    def test_forward_pass_still_works(
        self, baseline_model: BaselineCNN, dummy_input: torch.Tensor
    ) -> None:
        """Hooks should not break normal forward pass."""
        target_layer = baseline_model.get_target_layer()
        grad_cam = GradCAM(baseline_model, target_layer)
        output = baseline_model(dummy_input)
        assert output.shape == (1, 2)
        grad_cam.remove_hooks()

    def test_generate_batch(self, baseline_model: BaselineCNN) -> None:
        """Batch generation should produce one heatmap per image."""
        target_layer = baseline_model.get_target_layer()
        grad_cam = GradCAM(baseline_model, target_layer)
        batch = torch.randn(3, 3, 64, 64)
        heatmaps = grad_cam.generate_batch(batch)
        assert len(heatmaps) == 3
        for hm in heatmaps:
            assert hm.shape == (64, 64)
        grad_cam.remove_hooks()

    def test_remove_hooks(self, baseline_model: BaselineCNN) -> None:
        """remove_hooks should clear all registered hooks."""
        target_layer = baseline_model.get_target_layer()
        grad_cam = GradCAM(baseline_model, target_layer)
        assert len(grad_cam._handles) == 2
        grad_cam.remove_hooks()
        assert len(grad_cam._handles) == 0


class TestOverlayHeatmap:
    """Tests for the overlay_heatmap function."""

    def test_output_shape(self, dummy_image: np.ndarray) -> None:
        """Overlay should match input image shape."""
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = overlay_heatmap(dummy_image, heatmap)
        assert overlay.shape == dummy_image.shape

    def test_output_dtype(self, dummy_image: np.ndarray) -> None:
        """Overlay should be uint8."""
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = overlay_heatmap(dummy_image, heatmap)
        assert overlay.dtype == np.uint8

    def test_alpha_zero(self, dummy_image: np.ndarray) -> None:
        """Alpha 0 should produce the original image."""
        heatmap = np.ones((64, 64), dtype=np.float32)
        overlay = overlay_heatmap(dummy_image, heatmap, alpha=0.0)
        np.testing.assert_array_equal(overlay, dummy_image)

    def test_float_image_handling(self) -> None:
        """Should handle float images with values in [0, 1]."""
        image = np.random.rand(64, 64, 3).astype(np.float32)
        heatmap = np.random.rand(64, 64).astype(np.float32)
        overlay = overlay_heatmap(image, heatmap)
        assert overlay.dtype == np.uint8

    def test_different_heatmap_size(self, dummy_image: np.ndarray) -> None:
        """Should handle heatmaps with different spatial dimensions."""
        heatmap = np.random.rand(7, 7).astype(np.float32)
        overlay = overlay_heatmap(dummy_image, heatmap)
        assert overlay.shape == dummy_image.shape


class TestSaveVisualization:
    """Tests for save_gradcam_visualization."""

    def test_saves_file(self, tmp_path: Path, dummy_image: np.ndarray) -> None:
        """Should create an image file at the specified path."""
        heatmap = np.random.rand(64, 64).astype(np.float32)
        output_path = tmp_path / "vis" / "gradcam.png"
        result = save_gradcam_visualization(dummy_image, heatmap, output_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path: Path, dummy_image: np.ndarray) -> None:
        """Should create parent directories if they don't exist."""
        heatmap = np.random.rand(64, 64).astype(np.float32)
        output_path = tmp_path / "deep" / "nested" / "gradcam.png"
        save_gradcam_visualization(dummy_image, heatmap, output_path)
        assert output_path.exists()
