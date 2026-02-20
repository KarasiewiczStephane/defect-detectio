"""Training pipeline with early stopping, checkpointing, and LR scheduling.

Implements a complete training loop with validation, cosine annealing
learning rate schedule, early stopping, model checkpointing, and
training history tracking.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import config

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for the training pipeline.

    Attributes:
        epochs: Maximum number of training epochs.
        learning_rate: Initial learning rate for the optimizer.
        early_stopping_patience: Epochs without improvement before stopping.
        checkpoint_dir: Directory to save model checkpoints.
        device: Device to train on ('cuda' or 'cpu').
    """

    epochs: int = 50
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    device: str = ""

    def __post_init__(self) -> None:
        """Set device to CUDA if available when not specified."""
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_config(cls) -> "TrainingConfig":
        """Create TrainingConfig from the global config file.

        Returns:
            TrainingConfig populated from config.yaml values.
        """
        return cls(
            epochs=config.get("training.epochs", 50),
            learning_rate=config.get("training.learning_rate", 0.001),
            early_stopping_patience=config.get("training.early_stopping_patience", 10),
            checkpoint_dir=config.get("training.checkpoint_dir", "checkpoints"),
        )


class EarlyStopping:
    """Monitors validation loss and signals when to stop training.

    Tracks the best validation loss and counts consecutive epochs
    without improvement. Signals early stopping after exceeding
    the patience threshold.

    Args:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum decrease in loss to count as improvement.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """Check whether training should stop.

        Args:
            val_loss: Current epoch's validation loss.

        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False


@dataclass
class TrainingHistory:
    """Tracks metrics across training epochs.

    Attributes:
        train_loss: Per-epoch training loss.
        val_loss: Per-epoch validation loss.
        val_acc: Per-epoch validation accuracy.
        learning_rates: Per-epoch learning rate.
    """

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        """Convert history to a plain dictionary.

        Returns:
            Dictionary with metric names as keys and value lists.
        """
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
        }


class Trainer:
    """Orchestrates model training with validation and checkpointing.

    Args:
        model: The neural network to train.
        training_config: Training hyperparameters and settings.
        class_weights: Optional per-class weights for the loss function.
    """

    def __init__(
        self,
        model: nn.Module,
        training_config: TrainingConfig,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model.to(training_config.device)
        self.config = training_config
        self.device = training_config.device

        weight = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = Adam(model.parameters(), lr=training_config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=training_config.epochs)
        self.early_stopping = EarlyStopping(patience=training_config.early_stopping_patience)

        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = TrainingHistory()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, dataloader: DataLoader) -> tuple[float, float]:
        """Evaluate the model on a validation set.

        Args:
            dataloader: Validation data loader.

        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss at this epoch.
            is_best: If True, also save as ``best_model.pt``.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "history": self.history.as_dict(),
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best model (epoch %d, val_loss=%.4f)", epoch, val_loss)

    def load_checkpoint(self, path: str | Path) -> int:
        """Resume training from a checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            h = checkpoint["history"]
            self.history = TrainingHistory(**h)
        start_epoch = checkpoint["epoch"] + 1
        logger.info("Resumed from checkpoint at epoch %d", start_epoch)
        return start_epoch

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0,
    ) -> TrainingHistory:
        """Run the full training loop with validation and early stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            start_epoch: Epoch to start from (for resuming).

        Returns:
            TrainingHistory with per-epoch metrics.
        """
        best_val_loss = float("inf")

        for epoch in range(start_epoch, self.config.epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.learning_rates.append(current_lr)

            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.val_acc.append(val_acc)

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_acc=%.4f, lr=%.6f",
                epoch + 1,
                self.config.epochs,
                train_loss,
                val_loss,
                val_acc,
                current_lr,
            )

            if self.early_stopping(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        return self.history


def plot_training_history(
    history: TrainingHistory,
    save_path: str | Path | None = None,
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: TrainingHistory containing per-epoch metrics.
        save_path: Optional path to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history.train_loss) + 1)

    axes[0].plot(epochs, history.train_loss, label="Train Loss")
    axes[0].plot(epochs, history.val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, history.val_acc, label="Val Accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    if history.learning_rates:
        axes[2].plot(epochs, history.learning_rates[: len(epochs)], label="LR", color="orange")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_title("Learning Rate Schedule")
        axes[2].legend()

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved training plot to %s", save_path)

    plt.close(fig)
