"""Tests for the active learning pipeline."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.active_learning.pipeline import (
    ActiveLearningHistory,
    ActiveLearningPipeline,
    ActiveLearningRound,
    plot_learning_curve,
)
from src.active_learning.sampler import UncertaintySample, UncertaintySampler
from src.data.preprocessor import DefectDataset, get_transforms
from src.models.resnet_classifier import BaselineCNN
from src.models.trainer import TrainingConfig


@pytest.fixture()
def model() -> BaselineCNN:
    """Create a baseline model for testing."""
    return BaselineCNN(num_classes=2)


@pytest.fixture()
def small_dataloader(tmp_path: Path) -> DataLoader:
    """Create a small dataloader with synthetic images."""
    paths: list[Path] = []
    labels: list[int] = []
    for label in [0, 1]:
        d = tmp_path / f"class_{label}"
        d.mkdir()
        for i in range(8):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            p = d / f"img_{i}.png"
            img.save(p)
            paths.append(p)
            labels.append(label)

    transform = get_transforms("val", image_size=32)
    ds = DefectDataset(paths, labels, transform=transform)
    return DataLoader(ds, batch_size=4)


class TestUncertaintySampler:
    """Tests for the UncertaintySampler class."""

    def test_compute_uncertainty_uniform(self, model: BaselineCNN) -> None:
        """Uniform distribution should have maximum uncertainty."""
        sampler = UncertaintySampler(model, device="cpu")
        probs = torch.tensor([[0.5, 0.5]])
        uncertainty = sampler.compute_uncertainty(probs)
        assert uncertainty.item() == pytest.approx(1.0, abs=0.01)

    def test_compute_uncertainty_confident(self, model: BaselineCNN) -> None:
        """Highly confident prediction should have low uncertainty."""
        sampler = UncertaintySampler(model, device="cpu")
        probs = torch.tensor([[0.99, 0.01]])
        uncertainty = sampler.compute_uncertainty(probs)
        assert uncertainty.item() < 0.2

    def test_uncertainty_range(self, model: BaselineCNN) -> None:
        """Uncertainty values should be in [0, 1]."""
        sampler = UncertaintySampler(model, device="cpu")
        probs = torch.rand(10, 2)
        probs = probs / probs.sum(dim=1, keepdim=True)
        uncertainty = sampler.compute_uncertainty(probs)
        assert (uncertainty >= 0).all()
        assert (uncertainty <= 1.0 + 1e-6).all()

    def test_select_samples_respects_n(
        self, model: BaselineCNN, small_dataloader: DataLoader
    ) -> None:
        """Should return at most n_samples."""
        sampler = UncertaintySampler(model, threshold=0.0, device="cpu")
        samples = sampler.select_uncertain_samples(small_dataloader, n_samples=3)
        assert len(samples) <= 3

    def test_select_samples_returns_uncertainty_samples(
        self, model: BaselineCNN, small_dataloader: DataLoader
    ) -> None:
        """Returned items should be UncertaintySample instances."""
        sampler = UncertaintySampler(model, threshold=0.0, device="cpu")
        samples = sampler.select_uncertain_samples(small_dataloader, n_samples=5)
        for s in samples:
            assert isinstance(s, UncertaintySample)
            assert 0.0 <= s.uncertainty <= 1.0
            assert s.predicted_class in (0, 1)

    def test_select_with_high_threshold(
        self, model: BaselineCNN, small_dataloader: DataLoader
    ) -> None:
        """High threshold may return fewer or no samples."""
        sampler = UncertaintySampler(model, threshold=0.99, device="cpu")
        samples = sampler.select_uncertain_samples(small_dataloader, n_samples=10)
        assert isinstance(samples, list)

    def test_select_sorted_by_uncertainty(
        self, model: BaselineCNN, small_dataloader: DataLoader
    ) -> None:
        """Samples should be sorted by descending uncertainty."""
        sampler = UncertaintySampler(model, threshold=0.0, device="cpu")
        samples = sampler.select_uncertain_samples(small_dataloader, n_samples=10)
        if len(samples) > 1:
            for i in range(len(samples) - 1):
                assert samples[i].uncertainty >= samples[i + 1].uncertainty

    def test_select_with_image_paths(
        self, model: BaselineCNN, small_dataloader: DataLoader
    ) -> None:
        """Should populate image_path from provided paths list."""
        sampler = UncertaintySampler(model, threshold=0.0, device="cpu")
        paths = [f"img_{i}.png" for i in range(16)]
        samples = sampler.select_uncertain_samples(small_dataloader, n_samples=3, image_paths=paths)
        for s in samples:
            assert s.image_path.startswith("img_") or s.image_path.startswith("sample_")


class TestActiveLearningRound:
    """Tests for the ActiveLearningRound dataclass."""

    def test_round_fields(self) -> None:
        """Round should store all expected fields."""
        r = ActiveLearningRound(
            round_num=1,
            samples_added=50,
            accuracy_before=0.80,
            accuracy_after=0.85,
            improvement=0.05,
        )
        assert r.round_num == 1
        assert r.improvement == pytest.approx(0.05)


class TestActiveLearningHistory:
    """Tests for the ActiveLearningHistory."""

    def test_accuracies(self) -> None:
        """accuracies property should return list of after-accuracy values."""
        h = ActiveLearningHistory(
            rounds=[
                ActiveLearningRound(1, 10, 0.7, 0.75, 0.05),
                ActiveLearningRound(2, 10, 0.75, 0.80, 0.05),
            ]
        )
        assert h.accuracies == [0.75, 0.80]

    def test_total_samples_added(self) -> None:
        """Should sum samples across all rounds."""
        h = ActiveLearningHistory(
            rounds=[
                ActiveLearningRound(1, 30, 0.7, 0.75, 0.05),
                ActiveLearningRound(2, 20, 0.75, 0.80, 0.05),
            ]
        )
        assert h.total_samples_added == 50


class TestActiveLearningPipeline:
    """Tests for the full active learning pipeline."""

    def test_single_round(
        self, model: BaselineCNN, small_dataloader: DataLoader, tmp_path: Path
    ) -> None:
        """A single round should produce valid metrics."""
        sampler = UncertaintySampler(model, threshold=0.0, device="cpu")
        tc = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), device="cpu")
        pipeline = ActiveLearningPipeline(
            model=model,
            sampler=sampler,
            training_config=tc,
            class_names=["normal", "defect"],
            samples_per_round=5,
        )
        result = pipeline.run_round(small_dataloader, small_dataloader, small_dataloader)
        assert isinstance(result, ActiveLearningRound)
        assert result.round_num == 1
        assert 0.0 <= result.accuracy_before <= 1.0
        assert 0.0 <= result.accuracy_after <= 1.0

    def test_multiple_rounds(
        self, model: BaselineCNN, small_dataloader: DataLoader, tmp_path: Path
    ) -> None:
        """Running multiple rounds should accumulate history."""
        sampler = UncertaintySampler(model, threshold=0.0, device="cpu")
        tc = TrainingConfig(epochs=1, checkpoint_dir=str(tmp_path / "ckpt"), device="cpu")
        pipeline = ActiveLearningPipeline(
            model=model,
            sampler=sampler,
            training_config=tc,
            class_names=["normal", "defect"],
            samples_per_round=3,
        )
        history = pipeline.run(small_dataloader, small_dataloader, small_dataloader, max_rounds=2)
        assert len(history.rounds) == 2
        assert history.rounds[0].round_num == 1
        assert history.rounds[1].round_num == 2


class TestPlotLearningCurve:
    """Tests for plot_learning_curve."""

    def test_saves_figure(self, tmp_path: Path) -> None:
        """Should create an image file."""
        h = ActiveLearningHistory(
            rounds=[
                ActiveLearningRound(1, 10, 0.7, 0.75, 0.05),
                ActiveLearningRound(2, 10, 0.75, 0.82, 0.07),
            ]
        )
        save_path = tmp_path / "curve.png"
        plot_learning_curve(h, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_empty_history(self) -> None:
        """Empty history should not raise."""
        h = ActiveLearningHistory()
        plot_learning_curve(h)

    def test_no_save(self) -> None:
        """Should work without saving."""
        h = ActiveLearningHistory(rounds=[ActiveLearningRound(1, 5, 0.6, 0.65, 0.05)])
        plot_learning_curve(h, save_path=None)
