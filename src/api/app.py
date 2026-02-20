"""FastAPI application for defect detection inference.

Provides REST endpoints for single-image detection, batch detection,
and health checks. Uses ONNX Runtime for fast inference and supports
optional Grad-CAM overlay generation.
"""

import base64
import io
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from src.api.schemas import BatchDetectionResponse, DetectionResult, HealthResponse
from src.data.preprocessor import get_transforms
from src.deployment.onnx_exporter import ONNXInference
from src.models.grad_cam import GradCAM, overlay_heatmap
from src.models.resnet_classifier import create_model, load_checkpoint
from src.utils.config import config

logger = logging.getLogger(__name__)


class ModelState:
    """Holds loaded model and inference state.

    Attributes:
        pytorch_model: PyTorch model for Grad-CAM generation.
        onnx_inference: ONNX Runtime session for fast inference.
        grad_cam: GradCAM instance for heatmap generation.
        transform: Preprocessing transform pipeline.
        loaded: Whether models have been loaded successfully.
    """

    def __init__(self) -> None:
        self.pytorch_model: torch.nn.Module | None = None
        self.onnx_inference: ONNXInference | None = None
        self.grad_cam: GradCAM | None = None
        self.transform = get_transforms("val")
        self.loaded: bool = False

    def load(
        self,
        checkpoint_path: str | None = None,
        onnx_path: str | None = None,
    ) -> None:
        """Load models from disk.

        Args:
            checkpoint_path: Path to PyTorch checkpoint for Grad-CAM.
            onnx_path: Path to ONNX model for inference.
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            self.pytorch_model = create_model(pretrained=False)
            load_checkpoint(checkpoint_path, self.pytorch_model)
            self.pytorch_model.eval()
            target_layer = self.pytorch_model.get_target_layer()
            self.grad_cam = GradCAM(self.pytorch_model, target_layer)
            logger.info("Loaded PyTorch model from %s", checkpoint_path)

        if onnx_path and Path(onnx_path).exists():
            self.onnx_inference = ONNXInference(onnx_path)
            logger.info("Loaded ONNX model from %s", onnx_path)

        self.loaded = self.onnx_inference is not None or self.pytorch_model is not None


state = ModelState()


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits.

    Args:
        x: Logits array of shape (C,) or (B, C).

    Returns:
        Probability array with same shape as input.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def _preprocess_image(image_bytes: bytes) -> tuple[np.ndarray, torch.Tensor]:
    """Decode and preprocess an image for inference.

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        Tuple of (original_rgb_array, preprocessed_tensor).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}") from e

    original = np.array(image)
    tensor = state.transform(image).unsqueeze(0)
    return original, tensor


def _detect_single(
    image_bytes: bytes,
    include_gradcam: bool = True,
    threshold: float = 0.5,
) -> DetectionResult:
    """Run detection on a single image.

    Args:
        image_bytes: Raw image file bytes.
        include_gradcam: Whether to generate Grad-CAM overlay.
        threshold: Decision threshold for defect classification.

    Returns:
        DetectionResult with prediction and optional Grad-CAM.
    """
    original, tensor = _preprocess_image(image_bytes)

    if state.onnx_inference is not None:
        output = state.onnx_inference.predict(tensor.numpy())
        probs = _softmax(output[0])
    elif state.pytorch_model is not None:
        state.pytorch_model.eval()
        with torch.no_grad():
            output = state.pytorch_model(tensor)
            probs = torch.softmax(output, dim=1)[0].numpy()
    else:
        raise RuntimeError("No model loaded")

    is_defect = bool(probs[1] >= threshold)
    confidence = float(probs[1] if is_defect else probs[0])

    grad_cam_base64 = None
    if include_gradcam and is_defect and state.grad_cam is not None:
        heatmap = state.grad_cam.generate(tensor, target_class=1)
        overlay = overlay_heatmap(original, heatmap)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        grad_cam_base64 = base64.b64encode(buf.getvalue()).decode()

    return DetectionResult(
        is_defect=is_defect,
        confidence=confidence,
        defect_type="defect" if is_defect else "normal",
        grad_cam_base64=grad_cam_base64,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models on startup, clean up on shutdown."""
    checkpoint = config.get("api.checkpoint_path")
    onnx_path = config.get("deployment.model_path")
    state.load(checkpoint_path=checkpoint, onnx_path=onnx_path)
    logger.info("API startup complete, model_loaded=%s", state.loaded)
    yield
    if state.grad_cam is not None:
        state.grad_cam.remove_hooks()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Defect Detection API",
    version="1.0.0",
    description="Manufacturing defect detection with Grad-CAM visualization",
    lifespan=lifespan,
)


@app.post("/detect", response_model=DetectionResult)
async def detect_defect(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(default=True),
    threshold: float = Query(default=0.5, ge=0, le=1),
) -> DetectionResult:
    """Detect defects in a single uploaded image.

    Args:
        file: Uploaded image file (JPEG, PNG, BMP).
        include_gradcam: Whether to include Grad-CAM overlay in response.
        threshold: Decision threshold for defect classification.

    Returns:
        DetectionResult with prediction and optional Grad-CAM base64 image.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        return _detect_single(contents, include_gradcam, threshold)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error in /detect")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: list[UploadFile] = File(...),
    include_gradcam: bool = Query(default=False),
    threshold: float = Query(default=0.5, ge=0, le=1),
) -> BatchDetectionResponse:
    """Detect defects in multiple uploaded images.

    Args:
        files: List of uploaded image files.
        include_gradcam: Whether to include Grad-CAM overlays.
        threshold: Decision threshold for defect classification.

    Returns:
        BatchDetectionResponse with per-image results and timing.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.perf_counter()
    results: list[DetectionResult] = []

    for f in files:
        try:
            contents = await f.read()
            result = _detect_single(contents, include_gradcam, threshold)
            results.append(result)
        except Exception as e:
            logger.error("Error processing %s: %s", f.filename, e)
            results.append(
                DetectionResult(
                    is_defect=False,
                    confidence=0.0,
                    defect_type=f"error: {e}",
                )
            )

    processing_time = (time.perf_counter() - start_time) * 1000

    return BatchDetectionResponse(
        results=results,
        processing_time_ms=processing_time,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check API health and model status.

    Returns:
        HealthResponse with service status information.
    """
    return HealthResponse(
        status="healthy" if state.loaded else "degraded",
        model_loaded=state.loaded,
        model_type=config.get("model.architecture", "resnet50"),
        onnx_enabled=state.onnx_inference is not None,
        version="1.0.0",
    )
