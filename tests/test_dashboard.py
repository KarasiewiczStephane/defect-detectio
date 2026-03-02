"""Tests for the defect detection dashboard data generators."""

import numpy as np
import pandas as pd

from src.dashboard.app import (
    CATEGORIES,
    generate_category_metrics,
    generate_confusion_matrix,
    generate_latency_comparison,
    generate_training_history,
)


class TestCategoryMetrics:
    def test_returns_dataframe(self) -> None:
        df = generate_category_metrics()
        assert isinstance(df, pd.DataFrame)

    def test_has_all_categories(self) -> None:
        df = generate_category_metrics()
        assert len(df) == len(CATEGORIES)

    def test_has_required_columns(self) -> None:
        df = generate_category_metrics()
        for col in ["category", "precision", "recall", "f1", "support"]:
            assert col in df.columns

    def test_scores_bounded(self) -> None:
        df = generate_category_metrics()
        for col in ["precision", "recall", "f1"]:
            assert (df[col] >= 0).all()
            assert (df[col] <= 1).all()

    def test_support_positive(self) -> None:
        df = generate_category_metrics()
        assert (df["support"] > 0).all()

    def test_reproducible(self) -> None:
        df1 = generate_category_metrics(seed=99)
        df2 = generate_category_metrics(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestConfusionMatrix:
    def test_returns_ndarray(self) -> None:
        cm = generate_confusion_matrix()
        assert isinstance(cm, np.ndarray)

    def test_correct_shape(self) -> None:
        cm = generate_confusion_matrix()
        n = len(CATEGORIES)
        assert cm.shape == (n, n)

    def test_diagonal_dominant(self) -> None:
        cm = generate_confusion_matrix()
        for i in range(cm.shape[0]):
            assert cm[i, i] >= cm[i].sum() * 0.5

    def test_non_negative(self) -> None:
        cm = generate_confusion_matrix()
        assert (cm >= 0).all()


class TestTrainingHistory:
    def test_returns_dataframe(self) -> None:
        df = generate_training_history()
        assert isinstance(df, pd.DataFrame)

    def test_has_30_epochs(self) -> None:
        df = generate_training_history()
        assert len(df) == 30

    def test_loss_positive(self) -> None:
        df = generate_training_history()
        assert (df["train_loss"] > 0).all()
        assert (df["val_loss"] > 0).all()

    def test_accuracy_bounded(self) -> None:
        df = generate_training_history()
        assert (df["train_accuracy"] >= 0).all()
        assert (df["train_accuracy"] <= 1).all()


class TestLatencyComparison:
    def test_returns_dataframe(self) -> None:
        df = generate_latency_comparison()
        assert isinstance(df, pd.DataFrame)

    def test_has_three_runtimes(self) -> None:
        df = generate_latency_comparison()
        assert len(df) == 3

    def test_latencies_positive(self) -> None:
        df = generate_latency_comparison()
        assert (df["avg_latency_ms"] > 0).all()
        assert (df["p99_latency_ms"] > 0).all()

    def test_p99_gte_avg(self) -> None:
        df = generate_latency_comparison()
        assert (df["p99_latency_ms"] >= df["avg_latency_ms"]).all()
