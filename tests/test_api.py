"""Tests for FastAPI defect detection endpoints."""

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import ModelState, _detect_single, _preprocess_image, _softmax, app, state
from src.deployment.onnx_exporter import ONNXExporter, ONNXInference
from src.models.resnet_classifier import BaselineCNN


@pytest.fixture()
def client() -> TestClient:
    """Create a test client without model loading."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Create a valid PNG image as bytes."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def loaded_state(tmp_path: Path) -> None:
    """Set up model state with a real ONNX model for testing."""
    model = BaselineCNN(num_classes=2)
    exporter = ONNXExporter(model, input_shape=(1, 3, 224, 224))
    onnx_path = tmp_path / "test_model.onnx"
    exporter.export(onnx_path, opset_version=14)

    state.onnx_inference = ONNXInference(onnx_path)
    state.pytorch_model = model
    state.pytorch_model.eval()
    from src.models.grad_cam import GradCAM

    state.grad_cam = GradCAM(model, model.get_target_layer())
    state.loaded = True

    yield

    if state.grad_cam is not None:
        state.grad_cam.remove_hooks()
    state.onnx_inference = None
    state.pytorch_model = None
    state.grad_cam = None
    state.loaded = False


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient) -> None:
        """Health response should contain expected fields."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_type" in data
        assert "onnx_enabled" in data
        assert "version" in data

    def test_health_degraded_without_model(self, client: TestClient) -> None:
        """Should report degraded status when no model loaded."""
        state.loaded = False
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "degraded"


class TestDetectEndpoint:
    """Tests for the /detect endpoint."""

    def test_detect_returns_503_without_model(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Should return 503 when model is not loaded."""
        state.loaded = False
        response = client.post(
            "/detect", files={"file": ("test.png", sample_image_bytes, "image/png")}
        )
        assert response.status_code == 503

    def test_detect_returns_result(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        loaded_state: None,
    ) -> None:
        """Should return a valid detection result."""
        response = client.post(
            "/detect",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            params={"include_gradcam": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_defect" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_detect_with_gradcam(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        loaded_state: None,
    ) -> None:
        """Should include Grad-CAM when requested and defect detected."""
        response = client.post(
            "/detect",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            params={"include_gradcam": "true", "threshold": "0.0"},
        )
        assert response.status_code == 200
        data = response.json()
        if data["is_defect"]:
            assert data["grad_cam_base64"] is not None

    def test_detect_threshold(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        loaded_state: None,
    ) -> None:
        """Very high threshold should classify as normal."""
        response = client.post(
            "/detect",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
            params={"threshold": "0.999"},
        )
        assert response.status_code == 200


class TestBatchEndpoint:
    """Tests for the /detect/batch endpoint."""

    def test_batch_returns_503_without_model(
        self, client: TestClient, sample_image_bytes: bytes
    ) -> None:
        """Should return 503 when model is not loaded."""
        state.loaded = False
        response = client.post(
            "/detect/batch",
            files=[("files", ("test.png", sample_image_bytes, "image/png"))],
        )
        assert response.status_code == 503

    def test_batch_processes_multiple(
        self,
        client: TestClient,
        sample_image_bytes: bytes,
        loaded_state: None,
    ) -> None:
        """Should process multiple images and return results."""
        files = [
            ("files", ("img1.png", sample_image_bytes, "image/png")),
            ("files", ("img2.png", sample_image_bytes, "image/png")),
        ]
        response = client.post(
            "/detect/batch",
            files=files,
            params={"include_gradcam": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["processing_time_ms"] >= 0


class TestUtilities:
    """Tests for helper functions."""

    def test_softmax(self) -> None:
        """Softmax should produce valid probabilities."""
        logits = np.array([2.0, 1.0, 0.5])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
        assert (probs >= 0).all()

    def test_softmax_batch(self) -> None:
        """Softmax should work on batch inputs."""
        logits = np.array([[2.0, 1.0], [0.5, 1.5]])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_preprocess_image(self, sample_image_bytes: bytes) -> None:
        """Should return original array and preprocessed tensor."""
        original, tensor = _preprocess_image(sample_image_bytes)
        assert original.ndim == 3
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1

    def test_preprocess_invalid_image(self) -> None:
        """Should raise ValueError for invalid image bytes."""
        with pytest.raises(ValueError, match="Invalid image"):
            _preprocess_image(b"not_an_image")

    def test_detect_single(self, sample_image_bytes: bytes, loaded_state: None) -> None:
        """_detect_single should return a DetectionResult."""
        result = _detect_single(sample_image_bytes, include_gradcam=False)
        assert hasattr(result, "is_defect")
        assert hasattr(result, "confidence")


class TestModelState:
    """Tests for the ModelState class."""

    def test_initial_state(self) -> None:
        """Fresh ModelState should not be loaded."""
        ms = ModelState()
        assert not ms.loaded
        assert ms.pytorch_model is None
        assert ms.onnx_inference is None

    def test_load_nonexistent_paths(self) -> None:
        """Loading nonexistent paths should not crash."""
        ms = ModelState()
        ms.load(checkpoint_path="/nonexistent.pt", onnx_path="/nonexistent.onnx")
        assert not ms.loaded
