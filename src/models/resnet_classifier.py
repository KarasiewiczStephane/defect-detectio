"""ResNet-50 transfer learning model and baseline CNN for defect classification.

Provides a fine-tunable ResNet-50 with a custom classification head,
a lightweight baseline CNN for comparison, and a factory function
to instantiate models by architecture name.
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

from src.utils.config import config

logger = logging.getLogger(__name__)


class DefectResNet(nn.Module):
    """ResNet-50 with transfer learning for defect classification.

    Replaces the final fully-connected layer with a custom classification
    head featuring dropout regularization. Supports freezing the backbone
    for feature-extraction mode.

    Args:
        num_classes: Number of output classes (2 for binary defect/normal).
        pretrained: Whether to load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all backbone parameters.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)

    def get_target_layer(self) -> nn.Module:
        """Return the last convolutional layer for Grad-CAM.

        Returns:
            The final bottleneck block in layer4.
        """
        return self.backbone.layer4[-1]


class BaselineCNN(nn.Module):
    """Simple CNN baseline for comparison against ResNet-50.

    Three convolutional blocks with batch normalization and max pooling,
    followed by adaptive average pooling and a linear classifier.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)
        return self.classifier(x)

    def get_target_layer(self) -> nn.Module:
        """Return the last convolutional layer for Grad-CAM.

        Returns:
            The last Conv2d block in the features sequential.
        """
        return self.features[8]  # Third Conv2d


def create_model(
    architecture: Literal["resnet50", "baseline"] | None = None,
    num_classes: int | None = None,
    **kwargs: bool,
) -> nn.Module:
    """Factory function to create a defect classification model.

    Args:
        architecture: Model architecture name. Defaults to config value.
        num_classes: Number of output classes. Defaults to config value.
        **kwargs: Additional arguments passed to the model constructor
            (e.g., ``pretrained``, ``freeze_backbone``).

    Returns:
        Instantiated model.

    Raises:
        ValueError: If the architecture name is unknown.
    """
    architecture = architecture or config.get("model.architecture", "resnet50")
    num_classes = num_classes or config.get("model.num_classes", 2)

    if architecture == "resnet50":
        model = DefectResNet(num_classes=num_classes, **kwargs)
        logger.info("Created DefectResNet with %d classes", num_classes)
        return model

    if architecture == "baseline":
        model = BaselineCNN(num_classes=num_classes)
        logger.info("Created BaselineCNN with %d classes", num_classes)
        return model

    raise ValueError(f"Unknown architecture: {architecture}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    val_loss: float,
    path: str,
    **extra: object,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        model: The model to save.
        optimizer: Optimizer state to save (optional).
        epoch: Current epoch number.
        val_loss: Validation loss at this checkpoint.
        path: File path for the checkpoint.
        **extra: Additional metadata to store.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
        **extra,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    logger.info("Saved checkpoint to %s (epoch %d, val_loss=%.4f)", path, epoch, val_loss)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load a training checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore state.

    Returns:
        The full checkpoint dictionary with metadata.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint.get("epoch", -1))
    return checkpoint
