"""ONNX model export and inference runtime.

Handles exporting PyTorch models to ONNX format, validating output
consistency between runtimes, and providing an ONNX Runtime inference
wrapper for production use.
"""

import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from src.utils.config import config

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Exports a PyTorch model to ONNX format with validation.

    Args:
        model: PyTorch model to export.
        input_shape: Expected input tensor shape including batch dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
    ) -> None:
        self.model = model
        self.input_shape = input_shape

    def export(
        self,
        output_path: str | Path,
        opset_version: int | None = None,
        dynamic_batch: bool = True,
    ) -> Path:
        """Export the model to ONNX format.

        Args:
            output_path: File path for the exported ONNX model.
            opset_version: ONNX opset version. Defaults to config value.
            dynamic_batch: If True, allow variable batch sizes at inference.

        Returns:
            Path to the exported ONNX model file.

        Raises:
            onnx.checker.ValidationError: If the exported model is invalid.
        """
        opset_version = opset_version or config.get("deployment.onnx_opset_version", 14)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        dummy_input = torch.randn(self.input_shape)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        logger.info("Exported ONNX model to %s (opset=%d)", output_path, opset_version)
        return output_path

    def validate_outputs(
        self,
        onnx_path: str | Path,
        test_input: torch.Tensor | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ) -> bool:
        """Verify ONNX model produces outputs matching PyTorch.

        Args:
            onnx_path: Path to the exported ONNX model.
            test_input: Optional test tensor. Defaults to random input.
            rtol: Relative tolerance for comparison.
            atol: Absolute tolerance for comparison.

        Returns:
            True if outputs match within tolerance.
        """
        if test_input is None:
            test_input = torch.randn(self.input_shape)

        self.model.eval()
        with torch.no_grad():
            pytorch_output = self.model(test_input).numpy()

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_output = session.run(None, {"input": test_input.numpy()})[0]

        matches = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
        if matches:
            logger.info("ONNX validation passed (rtol=%s, atol=%s)", rtol, atol)
        else:
            max_diff = np.max(np.abs(pytorch_output - onnx_output))
            logger.warning("ONNX validation failed: max_diff=%.6f", max_diff)

        return bool(matches)


class ONNXInference:
    """ONNX Runtime inference wrapper for production use.

    Args:
        onnx_path: Path to the ONNX model file.
        device: Device for inference ('cpu' or 'cuda').
    """

    def __init__(self, onnx_path: str | Path, device: str = "cpu") -> None:
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("Loaded ONNX model from %s (device=%s)", onnx_path, device)

    def predict(self, input_array: np.ndarray) -> np.ndarray:
        """Run inference on a single input or batch.

        Args:
            input_array: Input array of shape (B, C, H, W) as float32.

        Returns:
            Output logits array of shape (B, num_classes).
        """
        return self.session.run(None, {self.input_name: input_array})[0]

    def predict_with_probabilities(self, input_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run inference and return both logits and softmax probabilities.

        Args:
            input_array: Input array of shape (B, C, H, W) as float32.

        Returns:
            Tuple of (logits, probabilities) arrays.
        """
        logits = self.predict(input_array)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return logits, probabilities
