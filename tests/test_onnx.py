"""Tests for ONNX export, validation, and benchmarking."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.deployment.benchmark import BenchmarkResults, benchmark_inference
from src.deployment.onnx_exporter import ONNXExporter, ONNXInference
from src.models.resnet_classifier import BaselineCNN


@pytest.fixture()
def model() -> BaselineCNN:
    """Create a baseline model for ONNX tests."""
    return BaselineCNN(num_classes=2)


@pytest.fixture()
def onnx_path(model: BaselineCNN, tmp_path: Path) -> Path:
    """Export a model and return the ONNX file path."""
    exporter = ONNXExporter(model, input_shape=(1, 3, 64, 64))
    return exporter.export(tmp_path / "model.onnx", opset_version=14)


class TestONNXExporter:
    """Tests for ONNX model export."""

    def test_export_creates_file(self, model: BaselineCNN, tmp_path: Path) -> None:
        """Export should create a valid .onnx file."""
        exporter = ONNXExporter(model, input_shape=(1, 3, 64, 64))
        path = exporter.export(tmp_path / "test.onnx", opset_version=14)
        assert path.exists()
        assert path.suffix == ".onnx"
        assert path.stat().st_size > 0

    def test_export_creates_parent_dirs(self, model: BaselineCNN, tmp_path: Path) -> None:
        """Export should create parent directories if needed."""
        exporter = ONNXExporter(model, input_shape=(1, 3, 64, 64))
        path = exporter.export(tmp_path / "nested" / "dir" / "model.onnx", opset_version=14)
        assert path.exists()

    def test_onnx_model_valid(self, onnx_path: Path) -> None:
        """Exported model should pass onnx.checker validation."""
        import onnx

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_validate_outputs_match(self, model: BaselineCNN, onnx_path: Path) -> None:
        """PyTorch and ONNX outputs should match within tolerance."""
        exporter = ONNXExporter(model, input_shape=(1, 3, 64, 64))
        assert exporter.validate_outputs(onnx_path, rtol=1e-3, atol=1e-4)

    def test_validate_with_specific_input(self, model: BaselineCNN, onnx_path: Path) -> None:
        """Validation should work with a provided test input."""
        exporter = ONNXExporter(model, input_shape=(1, 3, 64, 64))
        test_input = torch.randn(1, 3, 64, 64)
        assert exporter.validate_outputs(onnx_path, test_input=test_input)

    def test_dynamic_batch_size(self, model: BaselineCNN, tmp_path: Path) -> None:
        """Dynamic batch should allow different batch sizes at inference."""
        exporter = ONNXExporter(model, input_shape=(1, 3, 64, 64))
        path = exporter.export(tmp_path / "dynamic.onnx", opset_version=14, dynamic_batch=True)

        inference = ONNXInference(path)
        for batch_size in [1, 4, 8]:
            input_arr = np.random.randn(batch_size, 3, 64, 64).astype(np.float32)
            output = inference.predict(input_arr)
            assert output.shape == (batch_size, 2)


class TestONNXInference:
    """Tests for the ONNX Runtime inference wrapper."""

    def test_predict_shape(self, onnx_path: Path) -> None:
        """Predict should return correct output shape."""
        inference = ONNXInference(onnx_path)
        input_arr = np.random.randn(1, 3, 64, 64).astype(np.float32)
        output = inference.predict(input_arr)
        assert output.shape == (1, 2)

    def test_predict_batch(self, onnx_path: Path) -> None:
        """Predict should handle batch inputs."""
        inference = ONNXInference(onnx_path)
        input_arr = np.random.randn(4, 3, 64, 64).astype(np.float32)
        output = inference.predict(input_arr)
        assert output.shape == (4, 2)

    def test_predict_with_probabilities(self, onnx_path: Path) -> None:
        """predict_with_probabilities should return logits and valid probs."""
        inference = ONNXInference(onnx_path)
        input_arr = np.random.randn(2, 3, 64, 64).astype(np.float32)
        logits, probs = inference.predict_with_probabilities(input_arr)
        assert logits.shape == (2, 2)
        assert probs.shape == (2, 2)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_range(self, onnx_path: Path) -> None:
        """Probabilities should be in [0, 1]."""
        inference = ONNXInference(onnx_path)
        input_arr = np.random.randn(3, 3, 64, 64).astype(np.float32)
        _, probs = inference.predict_with_probabilities(input_arr)
        assert (probs >= 0).all()
        assert (probs <= 1).all()


class TestBenchmark:
    """Tests for inference benchmarking."""

    def test_benchmark_runs(self, model: BaselineCNN, onnx_path: Path) -> None:
        """Benchmark should complete and return valid results."""
        inference = ONNXInference(onnx_path)
        results = benchmark_inference(
            model,
            inference,
            input_shape=(1, 3, 64, 64),
            num_iterations=5,
            warmup=2,
        )
        assert isinstance(results, BenchmarkResults)
        assert results.pytorch_cpu_ms > 0
        assert results.onnx_cpu_ms > 0
        assert results.speedup_cpu > 0

    def test_benchmark_summary(self, model: BaselineCNN, onnx_path: Path) -> None:
        """Summary should contain timing information."""
        inference = ONNXInference(onnx_path)
        results = benchmark_inference(
            model,
            inference,
            input_shape=(1, 3, 64, 64),
            num_iterations=3,
            warmup=1,
        )
        summary = results.summary()
        assert "PyTorch CPU" in summary
        assert "ONNX CPU" in summary
        assert "Speedup" in summary

    def test_benchmark_results_summary_with_gpu(self) -> None:
        """Summary should include GPU results when available."""
        results = BenchmarkResults(
            pytorch_cpu_ms=50.0,
            onnx_cpu_ms=25.0,
            speedup_cpu=2.0,
            pytorch_gpu_ms=10.0,
            onnx_gpu_ms=5.0,
            speedup_gpu=2.0,
        )
        summary = results.summary()
        assert "GPU" in summary
