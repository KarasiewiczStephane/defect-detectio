"""Streamlit dashboard for defect detection visualization.

Displays model performance, per-category accuracy, confusion matrix,
inference latency comparison, and training history using synthetic data.

Run with: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CATEGORIES = ["Bottle", "Carpet", "Metal Nut", "Pill", "Tile", "Wood"]


def generate_category_metrics(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic per-category performance metrics."""
    rng = np.random.default_rng(seed)
    rows = []
    for cat in CATEGORIES:
        precision = round(rng.uniform(0.78, 0.98), 4)
        recall = round(rng.uniform(0.72, 0.96), 4)
        f1 = round(2 * precision * recall / (precision + recall), 4)
        rows.append(
            {
                "category": cat,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(rng.integers(50, 200)),
            }
        )
    return pd.DataFrame(rows)


def generate_confusion_matrix(seed: int = 42) -> np.ndarray:
    """Generate synthetic confusion matrix."""
    rng = np.random.default_rng(seed)
    n = len(CATEGORIES)
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        total = int(rng.integers(80, 200))
        correct = int(total * rng.uniform(0.75, 0.95))
        matrix[i, i] = correct
        remaining = total - correct
        for j in range(n):
            if j != i and remaining > 0:
                misclass = int(rng.integers(0, max(remaining // (n - 1) + 1, 1)))
                matrix[i, j] = misclass
                remaining -= misclass
    return matrix


def generate_training_history(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic training history."""
    rng = np.random.default_rng(seed)
    epochs = list(range(1, 31))
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    for epoch in epochs:
        tl = 1.5 * np.exp(-0.08 * epoch) + rng.uniform(0, 0.05)
        vl = 1.5 * np.exp(-0.06 * epoch) + rng.uniform(0, 0.08)
        ta = 1.0 - 0.5 * np.exp(-0.1 * epoch) + rng.uniform(-0.02, 0.02)
        va = 1.0 - 0.5 * np.exp(-0.08 * epoch) + rng.uniform(-0.03, 0.03)
        train_loss.append(round(max(tl, 0.01), 4))
        val_loss.append(round(max(vl, 0.01), 4))
        train_acc.append(round(min(max(ta, 0), 1), 4))
        val_acc.append(round(min(max(va, 0), 1), 4))
    return pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
        }
    )


def generate_latency_comparison(seed: int = 42) -> pd.DataFrame:
    """Generate synthetic inference latency comparison."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "runtime": ["PyTorch (GPU)", "PyTorch (CPU)", "ONNX Runtime"],
            "avg_latency_ms": [
                round(rng.uniform(8, 15), 2),
                round(rng.uniform(45, 80), 2),
                round(rng.uniform(5, 12), 2),
            ],
            "p99_latency_ms": [
                round(rng.uniform(20, 35), 2),
                round(rng.uniform(90, 150), 2),
                round(rng.uniform(15, 25), 2),
            ],
        }
    )


def render_header() -> None:
    """Render the dashboard header."""
    st.title("Manufacturing Defect Detection Dashboard")
    st.caption(
        "ResNet-50 with Grad-CAM interpretability, active learning, "
        "and ONNX deployment for real-time defect classification"
    )


def render_summary_metrics(cat_df: pd.DataFrame, latency_df: pd.DataFrame) -> None:
    """Render top-level summary metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Categories", len(cat_df))
    col2.metric("Avg F1 Score", f"{cat_df['f1'].mean():.4f}")
    col3.metric("Total Samples", f"{cat_df['support'].sum():,}")
    onnx_latency = latency_df[latency_df["runtime"] == "ONNX Runtime"]["avg_latency_ms"].iloc[0]
    col4.metric("ONNX Latency", f"{onnx_latency:.1f} ms")


def render_category_performance(cat_df: pd.DataFrame) -> None:
    """Render per-category performance bar chart."""
    st.subheader("Per-Category Performance")
    fig = go.Figure()
    for metric in ["precision", "recall", "f1"]:
        fig.add_trace(
            go.Bar(
                name=metric.capitalize(),
                x=cat_df["category"],
                y=cat_df[metric],
                text=cat_df[metric].apply(lambda x: f"{x:.3f}"),
                textposition="auto",
            )
        )
    fig.update_layout(
        barmode="group",
        yaxis={"range": [0.6, 1.0]},
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(cm: np.ndarray) -> None:
    """Render confusion matrix heatmap."""
    st.subheader("Confusion Matrix")
    fig = px.imshow(
        cm,
        x=CATEGORIES,
        y=CATEGORIES,
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_training_history(hist_df: pd.DataFrame) -> None:
    """Render training loss and accuracy curves."""
    st.subheader("Training History")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist_df["epoch"],
                y=hist_df["train_loss"],
                name="Train Loss",
                line={"color": "#2196F3"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df["epoch"],
                y=hist_df["val_loss"],
                name="Val Loss",
                line={"color": "#FF9800"},
            )
        )
        fig.update_layout(
            yaxis_title="Loss",
            height=300,
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=hist_df["epoch"],
                y=hist_df["train_accuracy"],
                name="Train Acc",
                line={"color": "#4CAF50"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hist_df["epoch"],
                y=hist_df["val_accuracy"],
                name="Val Acc",
                line={"color": "#9C27B0"},
            )
        )
        fig.update_layout(
            yaxis_title="Accuracy",
            height=300,
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)


def render_latency_comparison(latency_df: pd.DataFrame) -> None:
    """Render inference latency comparison."""
    st.subheader("Inference Latency Comparison")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Avg Latency",
            x=latency_df["runtime"],
            y=latency_df["avg_latency_ms"],
            text=latency_df["avg_latency_ms"].apply(lambda x: f"{x:.1f} ms"),
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="P99 Latency",
            x=latency_df["runtime"],
            y=latency_df["p99_latency_ms"],
            text=latency_df["p99_latency_ms"].apply(lambda x: f"{x:.1f} ms"),
            textposition="auto",
        )
    )
    fig.update_layout(
        barmode="group",
        yaxis_title="Latency (ms)",
        height=350,
        margin={"l": 40, "r": 20, "t": 30, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Main dashboard entry point."""
    render_header()

    cat_df = generate_category_metrics()
    cm = generate_confusion_matrix()
    hist_df = generate_training_history()
    latency_df = generate_latency_comparison()

    render_summary_metrics(cat_df, latency_df)
    st.markdown("---")

    render_category_performance(cat_df)

    col_left, col_right = st.columns(2)
    with col_left:
        render_confusion_matrix(cm)
    with col_right:
        render_latency_comparison(latency_df)

    st.markdown("---")
    render_training_history(hist_df)


if __name__ == "__main__":
    main()
