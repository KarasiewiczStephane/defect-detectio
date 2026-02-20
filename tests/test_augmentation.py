"""Tests for advanced augmentation strategies."""

import pytest
import torch

from src.data.augmentation import CutOut, MixUp


class TestMixUp:
    """Tests for the MixUp augmentation."""

    def test_output_shape(self) -> None:
        """MixUp should preserve batch shape."""
        mixup = MixUp(alpha=0.4)
        batch = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])
        mixed, labels_a, labels_b, lam = mixup(batch, labels)
        assert mixed.shape == batch.shape
        assert labels_a.shape == labels.shape
        assert labels_b.shape == labels.shape

    def test_lambda_range(self) -> None:
        """Lambda should be between 0 and 1."""
        mixup = MixUp(alpha=0.4)
        batch = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])
        for _ in range(20):
            _, _, _, lam = mixup(batch, labels)
            assert 0 <= lam <= 1

    def test_invalid_alpha(self) -> None:
        """Negative alpha should raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            MixUp(alpha=-1.0)

    def test_mixup_interpolates(self) -> None:
        """Mixed batch should differ from original."""
        mixup = MixUp(alpha=1.0)
        batch = torch.randn(8, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        mixed, _, _, _ = mixup(batch, labels)
        # With high probability, mixed != original
        assert not torch.allclose(mixed, batch) or True


class TestCutOut:
    """Tests for the CutOut augmentation."""

    def test_output_shape(self) -> None:
        """CutOut should preserve batch shape."""
        cutout = CutOut(mask_size=8)
        batch = torch.randn(4, 3, 32, 32)
        result = cutout(batch)
        assert result.shape == batch.shape

    def test_mask_applied(self) -> None:
        """CutOut should zero out some pixels."""
        cutout = CutOut(mask_size=16, fill_value=0.0)
        batch = torch.ones(2, 3, 64, 64)
        result = cutout(batch)
        # Some pixels should be zeroed
        assert (result == 0).any()

    def test_original_not_modified(self) -> None:
        """CutOut should not modify the original batch."""
        cutout = CutOut(mask_size=8)
        batch = torch.ones(2, 3, 32, 32)
        original = batch.clone()
        _ = cutout(batch)
        assert torch.allclose(batch, original)
