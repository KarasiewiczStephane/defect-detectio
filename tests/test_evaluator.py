"""Tests for the comprehensive evaluation suite."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.data.preprocessor import DefectDataset, get_transforms
from src.models.evaluator import EvaluationResults, Evaluator
from src.models.resnet_classifier import BaselineCNN


@pytest.fixture()
def eval_dataset(tmp_path: Path) -> DataLoader:
    """Create a small evaluation dataloader."""
    paths: list[Path] = []
    labels: list[int] = []
    for label in [0, 1]:
        d = tmp_path / f"class_{label}"
        d.mkdir()
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            p = d / f"img_{i}.png"
            img.save(p)
            paths.append(p)
            labels.append(label)

    transform = get_transforms("val", image_size=32)
    ds = DefectDataset(paths, labels, transform=transform)
    return DataLoader(ds, batch_size=8)


@pytest.fixture()
def evaluator() -> Evaluator:
    """Create an evaluator with a baseline model."""
    model = BaselineCNN(num_classes=2)
    return Evaluator(model, class_names=["normal", "defect"], device="cpu")


@pytest.fixture()
def eval_results(evaluator: Evaluator, eval_dataset: DataLoader) -> EvaluationResults:
    """Run evaluation and return results."""
    return evaluator.evaluate(eval_dataset)


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_evaluate_returns_results(self, eval_results: EvaluationResults) -> None:
        """evaluate should return an EvaluationResults object."""
        assert isinstance(eval_results, EvaluationResults)

    def test_accuracy_range(self, eval_results: EvaluationResults) -> None:
        """Accuracy should be between 0 and 1."""
        assert 0.0 <= eval_results.accuracy <= 1.0

    def test_confusion_matrix_shape(self, eval_results: EvaluationResults) -> None:
        """Confusion matrix should be (num_classes, num_classes)."""
        assert eval_results.confusion_mat.shape == (2, 2)

    def test_confusion_matrix_sums(self, eval_results: EvaluationResults) -> None:
        """Confusion matrix entries should sum to total samples."""
        assert eval_results.confusion_mat.sum() == 20

    def test_predictions_shape(self, eval_results: EvaluationResults) -> None:
        """Predictions array length should match dataset size."""
        assert len(eval_results.predictions) == 20

    def test_probabilities_shape(self, eval_results: EvaluationResults) -> None:
        """Probabilities should have shape (N, num_classes)."""
        assert eval_results.probabilities.shape == (20, 2)

    def test_probabilities_sum_to_one(self, eval_results: EvaluationResults) -> None:
        """Softmax probabilities should sum to ~1 per sample."""
        sums = eval_results.probabilities.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_precision_dict(self, eval_results: EvaluationResults) -> None:
        """Precision should be a dict with class names."""
        assert "defect" in eval_results.precision

    def test_recall_dict(self, eval_results: EvaluationResults) -> None:
        """Recall should be a dict with class names."""
        assert "defect" in eval_results.recall

    def test_f1_dict(self, eval_results: EvaluationResults) -> None:
        """F1 should be a dict with class names."""
        assert "defect" in eval_results.f1

    def test_per_category_metrics(self, eval_results: EvaluationResults) -> None:
        """Per-category metrics should contain class-level data."""
        assert "normal" in eval_results.per_category_metrics
        assert "defect" in eval_results.per_category_metrics

    def test_threshold_affects_predictions(self, eval_dataset: DataLoader) -> None:
        """Different thresholds should produce different prediction counts."""
        model = BaselineCNN(num_classes=2)
        ev_low = Evaluator(model, ["normal", "defect"], device="cpu", sensitivity_threshold=0.1)
        ev_high = Evaluator(model, ["normal", "defect"], device="cpu", sensitivity_threshold=0.9)
        results_low = ev_low.evaluate(eval_dataset)
        results_high = ev_high.evaluate(eval_dataset)
        # Low threshold => more predicted defects
        defects_low = (results_low.predictions == 1).sum()
        defects_high = (results_high.predictions == 1).sum()
        assert defects_low >= defects_high


class TestVisualization:
    """Tests for evaluation visualizations."""

    def test_plot_confusion_matrix(
        self, evaluator: Evaluator, eval_results: EvaluationResults, tmp_path: Path
    ) -> None:
        """Should create a confusion matrix image."""
        save_path = tmp_path / "cm.png"
        evaluator.plot_confusion_matrix(eval_results, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_roc_curve(
        self, evaluator: Evaluator, eval_results: EvaluationResults, tmp_path: Path
    ) -> None:
        """Should create a ROC curve and return AUC."""
        save_path = tmp_path / "roc.png"
        auc_val = evaluator.plot_roc_curve(eval_results, save_path=save_path)
        assert save_path.exists()
        assert 0.0 <= auc_val <= 1.0

    def test_plot_pr_curve(
        self, evaluator: Evaluator, eval_results: EvaluationResults, tmp_path: Path
    ) -> None:
        """Should create a PR curve and return AUC."""
        save_path = tmp_path / "pr.png"
        auc_val = evaluator.plot_precision_recall_curve(eval_results, save_path=save_path)
        assert save_path.exists()
        assert 0.0 <= auc_val <= 1.0


class TestReport:
    """Tests for the text report generation."""

    def test_generate_report(self, evaluator: Evaluator, eval_results: EvaluationResults) -> None:
        """Report should contain key metrics."""
        report = evaluator.generate_report(eval_results)
        assert "Accuracy" in report
        assert "Threshold" in report
        assert "defect" in report


class TestThresholdOptimization:
    """Tests for optimal threshold search."""

    def test_find_optimal_threshold(
        self, evaluator: Evaluator, eval_results: EvaluationResults
    ) -> None:
        """Should return a threshold between 0 and 1."""
        threshold, score = evaluator.find_optimal_threshold(eval_results, metric="f1")
        assert 0.0 <= threshold <= 1.0
        assert 0.0 <= score <= 1.0

    def test_optimal_threshold_precision(
        self, evaluator: Evaluator, eval_results: EvaluationResults
    ) -> None:
        """Should work for precision metric."""
        threshold, score = evaluator.find_optimal_threshold(eval_results, metric="precision")
        assert 0.0 <= threshold <= 1.0

    def test_optimal_threshold_recall(
        self, evaluator: Evaluator, eval_results: EvaluationResults
    ) -> None:
        """Should work for recall metric."""
        threshold, score = evaluator.find_optimal_threshold(eval_results, metric="recall")
        assert 0.0 <= threshold <= 1.0


class TestAllCorrectPredictions:
    """Edge case: all predictions correct."""

    def test_perfect_accuracy(self) -> None:
        """Perfect predictions should yield accuracy 1.0."""
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]])
        results = EvaluationResults(
            accuracy=1.0,
            precision={"defect": 1.0},
            recall={"defect": 1.0},
            f1={"defect": 1.0},
            confusion_mat=np.array([[2, 0], [0, 2]]),
            per_category_metrics={},
            predictions=preds,
            labels=labels,
            probabilities=probs,
        )
        assert results.accuracy == 1.0
        assert results.confusion_mat.diagonal().sum() == 4
