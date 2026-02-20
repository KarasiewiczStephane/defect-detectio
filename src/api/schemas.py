"""Pydantic schemas for the defect detection API request/response models."""

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """Response schema for a single defect detection prediction.

    Attributes:
        is_defect: Whether the image is classified as defective.
        confidence: Model confidence in the prediction.
        defect_type: Descriptive label for the prediction.
        grad_cam_base64: Optional base64-encoded Grad-CAM overlay PNG.
    """

    is_defect: bool
    confidence: float = Field(ge=0, le=1)
    defect_type: str | None = None
    grad_cam_base64: str | None = None


class BatchDetectionResponse(BaseModel):
    """Response schema for batch defect detection.

    Attributes:
        results: List of per-image detection results.
        processing_time_ms: Total processing time in milliseconds.
    """

    results: list[DetectionResult]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Response schema for the health check endpoint.

    Attributes:
        status: Service health status string.
        model_loaded: Whether the inference model is loaded.
        model_type: Architecture name of the loaded model.
        onnx_enabled: Whether ONNX Runtime inference is available.
        version: API version string.
    """

    status: str
    model_loaded: bool
    model_type: str
    onnx_enabled: bool
    version: str
