"""Tests for ResNet-50 transfer learning model and baseline CNN."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.models.resnet_classifier import (
    BaselineCNN,
    DefectResNet,
    create_model,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture()
def dummy_input() -> torch.Tensor:
    """Create a random input tensor matching expected image dimensions."""
    return torch.randn(2, 3, 224, 224)


class TestDefectResNet:
    """Tests for the DefectResNet model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        """Output shape should be (batch_size, num_classes)."""
        model = DefectResNet(num_classes=2, pretrained=False)
        output = model(dummy_input)
        assert output.shape == (2, 2)

    def test_multiclass_output(self, dummy_input: torch.Tensor) -> None:
        """Model should support arbitrary num_classes."""
        model = DefectResNet(num_classes=5, pretrained=False)
        output = model(dummy_input)
        assert output.shape == (2, 5)

    def test_pretrained_weights_differ(self) -> None:
        """Pretrained model should have different weights than random init."""
        pretrained = DefectResNet(num_classes=2, pretrained=True)
        random_init = DefectResNet(num_classes=2, pretrained=False)
        # Compare first conv layer weights
        w1 = pretrained.backbone.conv1.weight.data
        w2 = random_init.backbone.conv1.weight.data
        assert not torch.allclose(w1, w2)

    def test_freeze_backbone(self) -> None:
        """Frozen backbone should have no trainable params except classifier."""
        model = DefectResNet(num_classes=2, pretrained=False, freeze_backbone=True)
        # Backbone params should not require grad
        for name, param in model.backbone.named_parameters():
            if "fc" not in name:
                assert not param.requires_grad, f"{name} should be frozen"
        # FC params should require grad
        for param in model.backbone.fc.parameters():
            assert param.requires_grad

    def test_get_target_layer(self) -> None:
        """get_target_layer should return a valid module."""
        model = DefectResNet(num_classes=2, pretrained=False)
        layer = model.get_target_layer()
        assert isinstance(layer, nn.Module)

    def test_single_image(self) -> None:
        """Model should handle batch size of 1."""
        model = DefectResNet(num_classes=2, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 2)


class TestBaselineCNN:
    """Tests for the BaselineCNN model."""

    def test_forward_shape(self, dummy_input: torch.Tensor) -> None:
        """Output shape should be (batch_size, num_classes)."""
        model = BaselineCNN(num_classes=2)
        output = model(dummy_input)
        assert output.shape == (2, 2)

    def test_multiclass(self) -> None:
        """Baseline should support multi-class classification."""
        model = BaselineCNN(num_classes=10)
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        assert output.shape == (4, 10)

    def test_different_input_sizes(self) -> None:
        """Baseline with adaptive pooling should handle various input sizes."""
        model = BaselineCNN(num_classes=2)
        for size in [64, 128, 224]:
            x = torch.randn(1, 3, size, size)
            output = model(x)
            assert output.shape == (1, 2)

    def test_get_target_layer(self) -> None:
        """get_target_layer should return a Conv2d module."""
        model = BaselineCNN(num_classes=2)
        layer = model.get_target_layer()
        assert isinstance(layer, nn.Conv2d)


class TestCreateModel:
    """Tests for the model factory function."""

    def test_create_resnet(self) -> None:
        """Factory should create a DefectResNet."""
        model = create_model("resnet50", num_classes=2, pretrained=False)
        assert isinstance(model, DefectResNet)

    def test_create_baseline(self) -> None:
        """Factory should create a BaselineCNN."""
        model = create_model("baseline", num_classes=2)
        assert isinstance(model, BaselineCNN)

    def test_unknown_architecture(self) -> None:
        """Factory should raise ValueError for unknown architecture."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_model("vgg16", num_classes=2)


class TestCheckpointing:
    """Tests for checkpoint save/load utilities."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Saved checkpoint should restore model state correctly."""
        model = BaselineCNN(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters())
        path = str(tmp_path / "checkpoint.pt")

        save_checkpoint(model, optimizer, epoch=5, val_loss=0.25, path=path)

        model2 = BaselineCNN(num_classes=2)
        optimizer2 = torch.optim.Adam(model2.parameters())
        ckpt = load_checkpoint(path, model2, optimizer2)

        assert ckpt["epoch"] == 5
        assert ckpt["val_loss"] == pytest.approx(0.25)

        # Verify model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
            assert torch.allclose(p1, p2)

    def test_save_without_optimizer(self, tmp_path: Path) -> None:
        """Checkpoint should work without optimizer state."""
        model = BaselineCNN(num_classes=2)
        path = str(tmp_path / "model_only.pt")

        save_checkpoint(model, None, epoch=1, val_loss=0.5, path=path)

        model2 = BaselineCNN(num_classes=2)
        ckpt = load_checkpoint(path, model2)
        assert ckpt["epoch"] == 1

    def test_save_extra_metadata(self, tmp_path: Path) -> None:
        """Extra kwargs should be stored in checkpoint."""
        model = BaselineCNN(num_classes=2)
        path = str(tmp_path / "meta.pt")

        save_checkpoint(
            model,
            None,
            epoch=3,
            val_loss=0.1,
            path=path,
            history={"loss": [0.5, 0.3, 0.1]},
        )

        ckpt = load_checkpoint(path, model)
        assert ckpt["history"] == {"loss": [0.5, 0.3, 0.1]}
