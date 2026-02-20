"""Tests for the training pipeline with early stopping and checkpointing."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.preprocessor import DefectDataset, get_transforms
from src.models.resnet_classifier import BaselineCNN
from src.models.trainer import (
    EarlyStopping,
    Trainer,
    TrainingConfig,
    TrainingHistory,
    plot_training_history,
)


@pytest.fixture()
def small_dataset(tmp_path: Path) -> tuple[DataLoader, DataLoader]:
    """Create small train and val dataloaders with synthetic images."""
    paths: list[Path] = []
    labels: list[int] = []
    for label in [0, 1]:
        label_dir = tmp_path / f"class_{label}"
        label_dir.mkdir()
        for i in range(16):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            p = label_dir / f"img_{i}.png"
            img.save(p)
            paths.append(p)
            labels.append(label)

    transform = get_transforms("val", image_size=32)
    ds = DefectDataset(paths, labels, transform=transform)

    train_ds = torch.utils.data.Subset(ds, list(range(24)))
    val_ds = torch.utils.data.Subset(ds, list(range(24, 32)))

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)
    return train_loader, val_loader


@pytest.fixture()
def training_config(tmp_path: Path) -> TrainingConfig:
    """Create a training config for tests."""
    return TrainingConfig(
        epochs=3,
        learning_rate=0.01,
        early_stopping_patience=5,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )


class TestEarlyStopping:
    """Tests for the EarlyStopping monitor."""

    def test_no_stop_on_improvement(self) -> None:
        """Should not stop when loss decreases."""
        es = EarlyStopping(patience=3)
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)

    def test_stops_after_patience(self) -> None:
        """Should stop after patience epochs without improvement."""
        es = EarlyStopping(patience=3)
        es(0.5)  # best
        es(0.6)  # worse, counter=1
        es(0.7)  # worse, counter=2
        result = es(0.8)  # worse, counter=3 => stop
        assert result is True
        assert es.should_stop is True

    def test_counter_resets_on_improvement(self) -> None:
        """Counter should reset when loss improves."""
        es = EarlyStopping(patience=3)
        es(1.0)
        es(1.1)  # counter=1
        es(0.8)  # improvement, counter=0
        es(0.9)  # counter=1
        assert not es.should_stop
        assert es.counter == 1

    def test_min_delta(self) -> None:
        """Improvement must exceed min_delta to count."""
        es = EarlyStopping(patience=2, min_delta=0.1)
        es(1.0)
        es(0.95)  # not enough improvement (delta=0.05 < 0.1)
        result = es(0.92)  # still not enough
        assert result is True

    def test_reset(self) -> None:
        """Reset should restore initial state."""
        es = EarlyStopping(patience=2)
        es(0.5)
        es(0.6)
        es(0.7)
        es.reset()
        assert es.counter == 0
        assert es.best_loss == float("inf")
        assert es.should_stop is False


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_device(self) -> None:
        """Default device should be set automatically."""
        tc = TrainingConfig()
        assert tc.device in ("cuda", "cpu")

    def test_explicit_device(self) -> None:
        """Explicit device should be preserved."""
        tc = TrainingConfig(device="cpu")
        assert tc.device == "cpu"


class TestTrainer:
    """Tests for the Trainer class."""

    def test_train_one_epoch(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Training one epoch should return a finite loss."""
        train_loader, _ = small_dataset
        model = BaselineCNN(num_classes=2)
        trainer = Trainer(model, training_config)
        loss = trainer.train_epoch(train_loader)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_validate(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Validation should return loss and accuracy in [0, 1]."""
        _, val_loader = small_dataset
        model = BaselineCNN(num_classes=2)
        trainer = Trainer(model, training_config)
        val_loss, val_acc = trainer.validate(val_loader)
        assert np.isfinite(val_loss)
        assert 0.0 <= val_acc <= 1.0

    def test_fit_runs(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Fit should run for the configured number of epochs."""
        train_loader, val_loader = small_dataset
        model = BaselineCNN(num_classes=2)
        trainer = Trainer(model, training_config)
        history = trainer.fit(train_loader, val_loader)
        assert len(history.train_loss) == training_config.epochs
        assert len(history.val_loss) == training_config.epochs
        assert len(history.val_acc) == training_config.epochs

    def test_checkpoint_saved(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Fit should create checkpoint files."""
        train_loader, val_loader = small_dataset
        model = BaselineCNN(num_classes=2)
        trainer = Trainer(model, training_config)
        trainer.fit(train_loader, val_loader)

        ckpt_dir = Path(training_config.checkpoint_dir)
        assert (ckpt_dir / "best_model.pt").exists()
        assert any(ckpt_dir.glob("checkpoint_epoch_*.pt"))

    def test_class_weights(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Trainer should accept class weights for weighted loss."""
        train_loader, val_loader = small_dataset
        model = BaselineCNN(num_classes=2)
        weights = torch.tensor([1.0, 2.0])
        trainer = Trainer(model, training_config, class_weights=weights)
        loss = trainer.train_epoch(train_loader)
        assert np.isfinite(loss)

    def test_lr_decreases(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Cosine annealing should decrease the learning rate."""
        train_loader, val_loader = small_dataset
        model = BaselineCNN(num_classes=2)
        trainer = Trainer(model, training_config)
        history = trainer.fit(train_loader, val_loader)
        lrs = history.learning_rates
        assert lrs[-1] <= lrs[0]

    def test_load_checkpoint(
        self,
        small_dataset: tuple[DataLoader, DataLoader],
        training_config: TrainingConfig,
    ) -> None:
        """Loading a checkpoint should restore model and resume epoch."""
        train_loader, val_loader = small_dataset
        model = BaselineCNN(num_classes=2)
        trainer = Trainer(model, training_config)
        trainer.fit(train_loader, val_loader)

        # Load checkpoint into a fresh trainer
        model2 = BaselineCNN(num_classes=2)
        trainer2 = Trainer(model2, training_config)
        ckpt_path = Path(training_config.checkpoint_dir) / "best_model.pt"
        start_epoch = trainer2.load_checkpoint(ckpt_path)
        assert start_epoch >= 1


class TestTrainingHistory:
    """Tests for TrainingHistory."""

    def test_as_dict(self) -> None:
        """as_dict should return all tracked metrics."""
        h = TrainingHistory(
            train_loss=[0.5, 0.3],
            val_loss=[0.6, 0.4],
            val_acc=[0.7, 0.8],
            learning_rates=[0.001, 0.0005],
        )
        d = h.as_dict()
        assert d["train_loss"] == [0.5, 0.3]
        assert len(d) == 4


class TestPlotTrainingHistory:
    """Tests for plot_training_history."""

    def test_plot_saves_figure(self, tmp_path: Path) -> None:
        """plot_training_history should create a valid image file."""
        h = TrainingHistory(
            train_loss=[0.5, 0.3, 0.2],
            val_loss=[0.6, 0.4, 0.3],
            val_acc=[0.7, 0.8, 0.85],
            learning_rates=[0.001, 0.0008, 0.0005],
        )
        save_path = tmp_path / "plots" / "training.png"
        plot_training_history(h, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_no_save(self) -> None:
        """plot_training_history should work without saving."""
        h = TrainingHistory(
            train_loss=[0.5],
            val_loss=[0.6],
            val_acc=[0.7],
            learning_rates=[0.001],
        )
        plot_training_history(h, save_path=None)
