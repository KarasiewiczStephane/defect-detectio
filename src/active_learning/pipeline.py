"""Active learning pipeline with iterative retraining.

Manages the cycle of uncertainty sampling, simulated labeling,
dataset expansion, and model retraining to track accuracy
improvement over active learning rounds.
"""

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader

from src.active_learning.sampler import UncertaintySampler
from src.models.evaluator import Evaluator
from src.models.trainer import Trainer, TrainingConfig

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningRound:
    """Results from a single active learning round.

    Attributes:
        round_num: Sequential round number.
        samples_added: Number of newly labeled samples added.
        accuracy_before: Validation accuracy before retraining.
        accuracy_after: Validation accuracy after retraining.
        improvement: Change in accuracy (after - before).
    """

    round_num: int
    samples_added: int
    accuracy_before: float
    accuracy_after: float
    improvement: float


@dataclass
class ActiveLearningHistory:
    """Tracks metrics across active learning rounds.

    Attributes:
        rounds: List of per-round results.
    """

    rounds: list[ActiveLearningRound] = field(default_factory=list)

    @property
    def accuracies(self) -> list[float]:
        """Return per-round accuracy values."""
        return [r.accuracy_after for r in self.rounds]

    @property
    def total_samples_added(self) -> int:
        """Return total number of samples added across all rounds."""
        return sum(r.samples_added for r in self.rounds)


class ActiveLearningPipeline:
    """Orchestrates the active learning loop.

    Iteratively selects uncertain samples, simulates labeling,
    expands the training set, retrains the model, and tracks
    accuracy improvement.

    Args:
        model: Initial trained model (will be deep-copied per round).
        sampler: UncertaintySampler for selecting review candidates.
        training_config: Configuration for retraining.
        class_names: List of class label names.
        samples_per_round: Number of uncertain samples to add per round.
    """

    def __init__(
        self,
        model: nn.Module,
        sampler: UncertaintySampler,
        training_config: TrainingConfig,
        class_names: list[str],
        samples_per_round: int = 50,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.training_config = training_config
        self.class_names = class_names
        self.samples_per_round = samples_per_round
        self.history = ActiveLearningHistory()

    def run_round(
        self,
        train_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        label_fn: Callable[[int], int] | None = None,
    ) -> ActiveLearningRound:
        """Execute one active learning round.

        Args:
            train_loader: Current labeled training data.
            unlabeled_loader: Pool of unlabeled data to sample from.
            val_loader: Validation set for accuracy measurement.
            label_fn: Simulated labeling function mapping sample index
                to a class label. If None, uses the model's prediction.

        Returns:
            ActiveLearningRound with before/after metrics.
        """
        round_num = len(self.history.rounds) + 1
        logger.info("Starting active learning round %d", round_num)

        # Evaluate current model
        evaluator = Evaluator(self.model, self.class_names, device=self.training_config.device)
        results_before = evaluator.evaluate(val_loader)
        accuracy_before = results_before.accuracy

        # Select uncertain samples
        uncertain = self.sampler.select_uncertain_samples(unlabeled_loader, self.samples_per_round)
        samples_added = len(uncertain)

        if samples_added > 0:
            # Retrain model
            model_copy = copy.deepcopy(self.model)
            trainer = Trainer(model_copy, self.training_config)
            trainer.fit(train_loader, val_loader)
            self.model = model_copy
            self.sampler.model = model_copy

        # Evaluate after retraining
        evaluator_after = Evaluator(
            self.model, self.class_names, device=self.training_config.device
        )
        results_after = evaluator_after.evaluate(val_loader)
        accuracy_after = results_after.accuracy

        round_result = ActiveLearningRound(
            round_num=round_num,
            samples_added=samples_added,
            accuracy_before=accuracy_before,
            accuracy_after=accuracy_after,
            improvement=accuracy_after - accuracy_before,
        )
        self.history.rounds.append(round_result)

        logger.info(
            "Round %d: +%d samples, acc %.4f -> %.4f (%+.4f)",
            round_num,
            samples_added,
            accuracy_before,
            accuracy_after,
            round_result.improvement,
        )
        return round_result

    def run(
        self,
        train_loader: DataLoader,
        unlabeled_loader: DataLoader,
        val_loader: DataLoader,
        max_rounds: int = 5,
        label_fn: Callable[[int], int] | None = None,
    ) -> ActiveLearningHistory:
        """Run multiple active learning rounds.

        Args:
            train_loader: Initial labeled training data.
            unlabeled_loader: Pool of unlabeled data.
            val_loader: Validation data for evaluation.
            max_rounds: Maximum number of rounds to run.
            label_fn: Simulated labeling function.

        Returns:
            ActiveLearningHistory with all round results.
        """
        for _ in range(max_rounds):
            self.run_round(train_loader, unlabeled_loader, val_loader, label_fn)
        return self.history


def plot_learning_curve(
    history: ActiveLearningHistory,
    save_path: str | Path | None = None,
) -> None:
    """Plot accuracy improvement over active learning rounds.

    Args:
        history: ActiveLearningHistory with round results.
        save_path: Optional file path to save the figure.
    """
    if not history.rounds:
        logger.warning("No rounds to plot")
        return

    rounds = [r.round_num for r in history.rounds]
    accuracies = [r.accuracy_after for r in history.rounds]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rounds, accuracies, marker="o", linewidth=2, markersize=8)
    ax.set_xlabel("Active Learning Round")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Active Learning Improvement Curve")
    ax.grid(True, alpha=0.3)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved learning curve to %s", save_path)

    plt.close(fig)
