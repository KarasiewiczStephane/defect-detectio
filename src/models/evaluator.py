"""Comprehensive evaluation suite for defect classification models.

Provides metrics computation (accuracy, precision, recall, F1),
confusion matrix visualization, per-category performance breakdown,
ROC/PR curves, and configurable sensitivity thresholds.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics.

    Attributes:
        accuracy: Overall classification accuracy.
        precision: Per-class precision scores.
        recall: Per-class recall scores.
        f1: Per-class F1 scores.
        confusion_mat: Confusion matrix as numpy array.
        per_category_metrics: Full classification report as dict.
        predictions: Raw predicted class indices.
        labels: Ground truth labels.
        probabilities: Softmax probability distributions.
    """

    accuracy: float
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    confusion_mat: np.ndarray
    per_category_metrics: dict[str, dict]
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray


class Evaluator:
    """Evaluates a classification model on a test dataset.

    Computes standard classification metrics, generates visualizations,
    and supports configurable decision thresholds for sensitivity tuning.

    Args:
        model: Trained classification model.
        class_names: Ordered list of class label names.
        device: Device to run inference on.
        sensitivity_threshold: Decision threshold for binary classification.
            Predictions with defect probability >= threshold are classified
            as defect.
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        device: str | None = None,
        sensitivity_threshold: float = 0.5,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.device = device
        self.threshold = sensitivity_threshold

    def evaluate(self, dataloader: DataLoader) -> EvaluationResults:
        """Run evaluation on the provided dataloader.

        Args:
            dataloader: DataLoader yielding (images, labels) batches.

        Returns:
            EvaluationResults with all computed metrics.
        """
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probs: list[np.ndarray] = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)

                if len(self.class_names) == 2:
                    preds = (probs[:, 1] >= self.threshold).long()
                else:
                    preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
                all_probs.extend(probs.cpu().numpy())

        pred_arr = np.array(all_preds)
        label_arr = np.array(all_labels)
        prob_arr = np.array(all_probs)

        return EvaluationResults(
            accuracy=accuracy_score(label_arr, pred_arr),
            precision=self._per_class_metric(precision_score, label_arr, pred_arr),
            recall=self._per_class_metric(recall_score, label_arr, pred_arr),
            f1=self._per_class_metric(f1_score, label_arr, pred_arr),
            confusion_mat=confusion_matrix(label_arr, pred_arr),
            per_category_metrics=classification_report(
                label_arr,
                pred_arr,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0,
            ),
            predictions=pred_arr,
            labels=label_arr,
            probabilities=prob_arr,
        )

    def _per_class_metric(
        self,
        metric_fn: object,
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> dict[str, float]:
        """Compute a metric per class.

        Args:
            metric_fn: Sklearn metric function accepting (y_true, y_pred).
            labels: Ground truth array.
            preds: Predictions array.

        Returns:
            Dict mapping class names to metric values.
        """
        if len(self.class_names) == 2:
            score = metric_fn(labels, preds, pos_label=1, zero_division=0)
            return {self.class_names[1]: float(score)}

        scores = metric_fn(labels, preds, average=None, zero_division=0)
        return {name: float(score) for name, score in zip(self.class_names, scores, strict=True)}

    def plot_confusion_matrix(
        self,
        results: EvaluationResults,
        save_path: Path | str | None = None,
    ) -> None:
        """Plot and optionally save a confusion matrix heatmap.

        Args:
            results: Evaluation results containing the confusion matrix.
            save_path: Optional file path to save the figure.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            results.confusion_mat,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved confusion matrix to %s", save_path)
        plt.close(fig)

    def plot_roc_curve(
        self,
        results: EvaluationResults,
        save_path: Path | str | None = None,
    ) -> float:
        """Plot ROC curve for binary classification.

        Args:
            results: Evaluation results with probabilities and labels.
            save_path: Optional file path to save the figure.

        Returns:
            Area under the ROC curve.
        """
        if len(self.class_names) != 2:
            logger.warning("ROC curve only supported for binary classification")
            return 0.0

        fpr, tpr, _ = roc_curve(results.labels, results.probabilities[:, 1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return float(roc_auc)

    def plot_precision_recall_curve(
        self,
        results: EvaluationResults,
        save_path: Path | str | None = None,
    ) -> float:
        """Plot precision-recall curve for binary classification.

        Args:
            results: Evaluation results with probabilities and labels.
            save_path: Optional file path to save the figure.

        Returns:
            Area under the precision-recall curve.
        """
        if len(self.class_names) != 2:
            logger.warning("PR curve only supported for binary classification")
            return 0.0

        prec, rec, _ = precision_recall_curve(results.labels, results.probabilities[:, 1])
        pr_auc = auc(rec, prec)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(rec, prec, label=f"PR (AUC = {pr_auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return float(pr_auc)

    def generate_report(self, results: EvaluationResults) -> str:
        """Generate a human-readable evaluation report.

        Args:
            results: Evaluation results to summarize.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 50,
            "EVALUATION REPORT",
            "=" * 50,
            f"Accuracy: {results.accuracy:.4f}",
            f"Threshold: {self.threshold}",
            "",
            "Per-Class Metrics:",
            "-" * 30,
        ]
        for class_name in self.class_names:
            p = results.precision.get(class_name, 0)
            r = results.recall.get(class_name, 0)
            f = results.f1.get(class_name, 0)
            lines.append(f"  {class_name}: P={p:.4f}, R={r:.4f}, F1={f:.4f}")

        lines.extend(["", "=" * 50])
        return "\n".join(lines)

    def find_optimal_threshold(
        self,
        results: EvaluationResults,
        metric: str = "f1",
        thresholds: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """Find the threshold that maximizes a given metric.

        Args:
            results: Evaluation results with probabilities and labels.
            metric: Metric to optimize ("f1", "precision", or "recall").
            thresholds: Array of thresholds to test. Defaults to
                0.1 to 0.9 in steps of 0.05.

        Returns:
            Tuple of (best_threshold, best_metric_value).
        """
        if len(self.class_names) != 2:
            logger.warning("Threshold optimization only for binary classification")
            return 0.5, 0.0

        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)

        metric_fns = {
            "f1": lambda y, p: f1_score(y, p, zero_division=0),
            "precision": lambda y, p: precision_score(y, p, zero_division=0),
            "recall": lambda y, p: recall_score(y, p, zero_division=0),
        }
        fn = metric_fns.get(metric, metric_fns["f1"])

        best_threshold = 0.5
        best_score = 0.0

        for t in thresholds:
            preds = (results.probabilities[:, 1] >= t).astype(int)
            score = fn(results.labels, preds)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        logger.info(
            "Optimal threshold for %s: %.2f (score=%.4f)",
            metric,
            best_threshold,
            best_score,
        )
        return best_threshold, float(best_score)
