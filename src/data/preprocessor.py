"""Image preprocessing pipeline with PyTorch Dataset and stratified splitting.

Provides transform pipelines for train/val/test with ImageNet normalization,
a PyTorch Dataset class for defect images, and utilities for stratified
data splitting and class balance analysis.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.config import config

logger = logging.getLogger(__name__)


class DefectDataset(Dataset):
    """PyTorch Dataset for defect detection images.

    Loads images from file paths with corresponding labels and applies
    optional transforms for data augmentation or normalization.

    Args:
        image_paths: List of paths to image files.
        labels: List of integer labels corresponding to each image.
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        image_paths: list[Path],
        labels: list[int],
        transform: transforms.Compose | None = None,
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError(f"Mismatch: {len(image_paths)} images vs {len(labels)} labels")
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return a single image-label pair.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple of (image_tensor, label).

        Raises:
            OSError: If the image file cannot be read.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(
    mode: Literal["train", "val", "test"],
    image_size: int | None = None,
    imagenet_mean: tuple[float, ...] | None = None,
    imagenet_std: tuple[float, ...] | None = None,
) -> transforms.Compose:
    """Build a torchvision transform pipeline for the given data split.

    Training transforms include random augmentations (flips, rotation,
    color jitter, random crop). Validation and test transforms only
    resize and normalize.

    Args:
        mode: One of "train", "val", or "test".
        image_size: Target image size. Defaults to config value.
        imagenet_mean: Channel means for normalization. Defaults to config.
        imagenet_std: Channel standard deviations. Defaults to config.

    Returns:
        Composed transform pipeline.
    """
    image_size = image_size or config.get("data.image_size", 224)
    imagenet_mean = imagenet_mean or tuple(config.get("data.imagenet_mean", [0.485, 0.456, 0.406]))
    imagenet_std = imagenet_std or tuple(config.get("data.imagenet_std", [0.229, 0.224, 0.225]))

    if mode == "train":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomCrop(image_size, padding=16),
                transforms.ToTensor(),
                transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def collect_image_paths_and_labels(
    data_dir: str | Path,
    categories: list[str] | None = None,
) -> tuple[list[Path], list[int]]:
    """Scan a data directory and collect image paths with binary labels.

    Images under ``*/good/*`` are labeled 0 (normal), all others are
    labeled 1 (defect).

    Args:
        data_dir: Root directory containing category subdirectories.
        categories: Specific categories to include. If None, scans all.

    Returns:
        Tuple of (image_paths, labels).
    """
    data_dir = Path(data_dir)
    image_paths: list[Path] = []
    labels: list[int] = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if categories:
        scan_dirs = [data_dir / cat for cat in categories if (data_dir / cat).exists()]
    else:
        scan_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    for cat_dir in scan_dirs:
        for img_path in cat_dir.rglob("*"):
            if img_path.suffix.lower() in extensions and img_path.is_file():
                label = 0 if "good" in img_path.parts else 1
                image_paths.append(img_path)
                labels.append(label)

    logger.info(
        "Collected %d images (%d normal, %d defect) from %s",
        len(image_paths),
        labels.count(0),
        labels.count(1),
        data_dir,
    )
    return image_paths, labels


def create_stratified_split(
    image_paths: list[Path],
    labels: list[int],
    train_ratio: float | None = None,
    val_ratio: float | None = None,
) -> tuple[list[Path], list[int], list[Path], list[int], list[Path], list[int]]:
    """Split image paths and labels into stratified train/val/test sets.

    Args:
        image_paths: Full list of image file paths.
        labels: Corresponding integer labels.
        train_ratio: Proportion for training. Defaults to config value.
        val_ratio: Proportion for validation. Defaults to config value.

    Returns:
        Six-element tuple of (train_paths, train_labels, val_paths,
        val_labels, test_paths, test_labels).
    """
    train_ratio = train_ratio or config.get("data.train_split", 0.7)
    val_ratio = val_ratio or config.get("data.val_split", 0.15)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=42,
    )

    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        train_size=val_ratio_adjusted,
        stratify=temp_labels,
        random_state=42,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def analyze_class_balance(labels: list[int]) -> dict[int, int]:
    """Count occurrences of each label.

    Args:
        labels: List of integer class labels.

    Returns:
        Dictionary mapping each label to its count.
    """
    return dict(Counter(labels))


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        labels: List of integer class labels.

    Returns:
        Float tensor of per-class weights suitable for CrossEntropyLoss.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / (len(counts) * counts[i]) for i in sorted(counts.keys())]
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(labels: list[int]) -> torch.utils.data.WeightedRandomSampler:
    """Create a weighted random sampler for class-balanced batches.

    Args:
        labels: List of integer class labels.

    Returns:
        WeightedRandomSampler for use with a DataLoader.
    """
    class_weights = compute_class_weights(labels)
    sample_weights = np.array([class_weights[label].item() for label in labels])
    sample_weights_tensor = torch.from_numpy(sample_weights).double()
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(labels),
        replacement=True,
    )
