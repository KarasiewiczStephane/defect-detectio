"""Tests for data preprocessing, dataset classes, and splitting utilities."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.preprocessor import (
    DefectDataset,
    analyze_class_balance,
    collect_image_paths_and_labels,
    compute_class_weights,
    create_stratified_split,
    create_weighted_sampler,
    get_transforms,
)


@pytest.fixture()
def sample_images(tmp_path: Path) -> tuple[list[Path], list[int]]:
    """Create sample images with known labels."""
    paths: list[Path] = []
    labels: list[int] = []

    for split_name, label in [("good", 0), ("defect", 1)]:
        split_dir = tmp_path / "bottle" / "train" / split_name
        split_dir.mkdir(parents=True)
        for i in range(20):
            img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            img_path = split_dir / f"img_{i:03d}.png"
            img.save(img_path)
            paths.append(img_path)
            labels.append(label)

    return paths, labels


class TestDefectDataset:
    """Tests for the DefectDataset class."""

    def test_length(self, sample_images: tuple[list[Path], list[int]]) -> None:
        """Dataset length should match input size."""
        paths, labels = sample_images
        ds = DefectDataset(paths, labels)
        assert len(ds) == 40

    def test_getitem_returns_tuple(self, sample_images: tuple[list[Path], list[int]]) -> None:
        """__getitem__ should return (image, label) tuple."""
        paths, labels = sample_images
        transform = get_transforms("val", image_size=64)
        ds = DefectDataset(paths, labels, transform=transform)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_getitem_shape(self, sample_images: tuple[list[Path], list[int]]) -> None:
        """Transformed images should have shape (3, H, W)."""
        paths, labels = sample_images
        transform = get_transforms("val", image_size=224)
        ds = DefectDataset(paths, labels, transform=transform)
        image, _ = ds[0]
        assert image.shape == (3, 224, 224)

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched paths and labels should raise ValueError."""
        with pytest.raises(ValueError, match="Mismatch"):
            DefectDataset([Path("a.png")], [0, 1])

    def test_no_transform(self, sample_images: tuple[list[Path], list[int]]) -> None:
        """Without transform, __getitem__ should return PIL Image."""
        paths, labels = sample_images
        ds = DefectDataset(paths, labels, transform=None)
        image, label = ds[0]
        assert isinstance(image, Image.Image)
        assert label in (0, 1)


class TestTransforms:
    """Tests for get_transforms factory function."""

    def test_train_transforms_output_shape(self) -> None:
        """Train transforms should produce (3, size, size) tensors."""
        transform = get_transforms("train", image_size=224)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = transform(img)
        assert result.shape == (3, 224, 224)

    def test_val_transforms_deterministic(self) -> None:
        """Validation transforms should be deterministic."""
        transform = get_transforms("val", image_size=128)
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        r1 = transform(img)
        r2 = transform(img)
        assert torch.allclose(r1, r2)

    def test_test_transforms_same_as_val(self) -> None:
        """Test transforms should match validation transforms."""
        val_t = get_transforms("val", image_size=64)
        test_t = get_transforms("test", image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        assert torch.allclose(val_t(img), test_t(img))

    def test_train_augments_differ(self) -> None:
        """Train transforms should produce different results across calls."""
        transform = get_transforms("train", image_size=64)
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        results = [transform(img) for _ in range(10)]
        # At least some should differ
        differs = any(not torch.allclose(results[0], r) for r in results[1:])
        assert differs


class TestSplitting:
    """Tests for stratified splitting and class analysis."""

    def test_stratified_split_sizes(self, sample_images: tuple[list[Path], list[int]]) -> None:
        """Split should produce correct proportions."""
        paths, labels = sample_images
        tp, tl, vp, vl, tep, tel = create_stratified_split(
            paths, labels, train_ratio=0.7, val_ratio=0.15
        )
        total = len(tp) + len(vp) + len(tep)
        assert total == len(paths)
        assert len(tp) == len(tl)
        assert len(vp) == len(vl)
        assert len(tep) == len(tel)

    def test_stratified_split_preserves_ratio(
        self, sample_images: tuple[list[Path], list[int]]
    ) -> None:
        """Each split should approximately maintain class proportions."""
        paths, labels = sample_images
        tp, tl, vp, vl, tep, tel = create_stratified_split(
            paths, labels, train_ratio=0.7, val_ratio=0.15
        )
        original_ratio = sum(labels) / len(labels)
        train_ratio = sum(tl) / len(tl) if tl else 0
        assert abs(original_ratio - train_ratio) < 0.15

    def test_analyze_class_balance(self) -> None:
        """analyze_class_balance should return correct counts."""
        labels = [0, 0, 0, 1, 1]
        balance = analyze_class_balance(labels)
        assert balance == {0: 3, 1: 2}

    def test_compute_class_weights_balanced(self) -> None:
        """Equal class counts should produce equal weights."""
        labels = [0, 0, 1, 1]
        weights = compute_class_weights(labels)
        assert torch.allclose(weights[0], weights[1])

    def test_compute_class_weights_imbalanced(self) -> None:
        """Minority class should get higher weight."""
        labels = [0, 0, 0, 0, 1]
        weights = compute_class_weights(labels)
        assert weights[1] > weights[0]

    def test_weighted_sampler(self) -> None:
        """create_weighted_sampler should return a valid sampler."""
        labels = [0, 0, 0, 1, 1]
        sampler = create_weighted_sampler(labels)
        assert sampler.num_samples == len(labels)


class TestCollectPaths:
    """Tests for collect_image_paths_and_labels."""

    def test_collect_from_directory(self, tmp_path: Path) -> None:
        """Should find images and assign correct labels."""
        for _name, label_dir in [("good", "good"), ("scratch", "scratch")]:
            d = tmp_path / "bottle" / label_dir
            d.mkdir(parents=True)
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
                img.save(d / f"img_{i}.png")

        paths, labels = collect_image_paths_and_labels(tmp_path, categories=["bottle"])
        assert len(paths) == 10
        assert labels.count(0) == 5
        assert labels.count(1) == 5

    def test_collect_empty_directory(self, tmp_path: Path) -> None:
        """Should return empty lists for empty directories."""
        paths, labels = collect_image_paths_and_labels(tmp_path)
        assert paths == []
        assert labels == []
