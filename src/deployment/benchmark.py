"""Inference benchmarking for PyTorch vs ONNX Runtime.

Measures and compares latency between PyTorch and ONNX Runtime
inference on CPU (and optionally GPU) to quantify speedup.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.deployment.onnx_exporter import ONNXInference

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    """Container for benchmark timing results.

    Attributes:
        pytorch_cpu_ms: Average PyTorch CPU inference time in milliseconds.
        onnx_cpu_ms: Average ONNX Runtime CPU inference time in milliseconds.
        speedup_cpu: Ratio of PyTorch to ONNX CPU time (>1 means ONNX is faster).
        pytorch_gpu_ms: Average PyTorch GPU inference time (None if unavailable).
        onnx_gpu_ms: Average ONNX GPU inference time (None if unavailable).
        speedup_gpu: GPU speedup ratio (None if unavailable).
        num_iterations: Number of benchmark iterations performed.
    """

    pytorch_cpu_ms: float
    onnx_cpu_ms: float
    speedup_cpu: float
    pytorch_gpu_ms: float | None = None
    onnx_gpu_ms: float | None = None
    speedup_gpu: float | None = None
    num_iterations: int = 100

    def summary(self) -> str:
        """Generate a human-readable benchmark summary.

        Returns:
            Formatted string with timing comparisons.
        """
        lines = [
            "=" * 50,
            "INFERENCE BENCHMARK RESULTS",
            "=" * 50,
            f"Iterations: {self.num_iterations}",
            "",
            f"PyTorch CPU: {self.pytorch_cpu_ms:.2f} ms/image",
            f"ONNX CPU:    {self.onnx_cpu_ms:.2f} ms/image",
            f"CPU Speedup: {self.speedup_cpu:.2f}x",
        ]
        if self.pytorch_gpu_ms is not None and self.onnx_gpu_ms is not None:
            lines.extend(
                [
                    "",
                    f"PyTorch GPU: {self.pytorch_gpu_ms:.2f} ms/image",
                    f"ONNX GPU:    {self.onnx_gpu_ms:.2f} ms/image",
                    f"GPU Speedup: {self.speedup_gpu:.2f}x",
                ]
            )
        lines.append("=" * 50)
        return "\n".join(lines)


def benchmark_inference(
    pytorch_model: nn.Module,
    onnx_inference: ONNXInference,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    num_iterations: int = 100,
    warmup: int = 10,
) -> BenchmarkResults:
    """Benchmark PyTorch vs ONNX Runtime inference latency.

    Args:
        pytorch_model: PyTorch model for comparison.
        onnx_inference: ONNXInference wrapper to benchmark.
        input_shape: Input tensor shape (including batch dimension).
        num_iterations: Number of timed iterations.
        warmup: Number of warmup iterations before timing.

    Returns:
        BenchmarkResults with timing comparisons.
    """
    dummy_np = np.random.randn(*input_shape).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np)

    pytorch_model.cpu().eval()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            pytorch_model(dummy_torch)
        onnx_inference.predict(dummy_np)

    # PyTorch CPU benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            pytorch_model(dummy_torch.cpu())
    pytorch_cpu_ms = (time.perf_counter() - start) * 1000 / num_iterations

    # ONNX CPU benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        onnx_inference.predict(dummy_np)
    onnx_cpu_ms = (time.perf_counter() - start) * 1000 / num_iterations

    speedup_cpu = pytorch_cpu_ms / max(onnx_cpu_ms, 1e-6)

    results = BenchmarkResults(
        pytorch_cpu_ms=pytorch_cpu_ms,
        onnx_cpu_ms=onnx_cpu_ms,
        speedup_cpu=speedup_cpu,
        num_iterations=num_iterations,
    )

    logger.info(
        "Benchmark: PyTorch=%.2fms, ONNX=%.2fms, speedup=%.2fx",
        pytorch_cpu_ms,
        onnx_cpu_ms,
        speedup_cpu,
    )
    return results
