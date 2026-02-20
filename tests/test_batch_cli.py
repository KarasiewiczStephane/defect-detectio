"""Tests for batch processing CLI and report generation."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.deployment.onnx_exporter import ONNXExporter
from src.main import _softmax, _write_report, main, parse_args, run_batch
from src.models.resnet_classifier import BaselineCNN


@pytest.fixture()
def sample_images_dir(tmp_path: Path) -> Path:
    """Create a directory with sample images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / f"sample_{i:03d}.png")
    return img_dir


@pytest.fixture()
def onnx_model_path(tmp_path: Path) -> Path:
    """Export a test ONNX model."""
    model = BaselineCNN(num_classes=2)
    exporter = ONNXExporter(model, input_shape=(1, 3, 224, 224))
    path = tmp_path / "model.onnx"
    exporter.export(path, opset_version=14)
    return path


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_batch_args(self) -> None:
        """Batch command should parse all required args."""
        args = parse_args(["batch", "-i", "in/", "-o", "out/", "-m", "model.onnx"])
        assert args.command == "batch"
        assert args.input == "in/"
        assert args.output == "out/"
        assert args.model == "model.onnx"

    def test_batch_threshold(self) -> None:
        """Batch should accept threshold."""
        args = parse_args(["batch", "-i", "in/", "-o", "out/", "-m", "m.onnx", "-t", "0.7"])
        assert args.threshold == 0.7

    def test_batch_gradcam_flag(self) -> None:
        """Batch should accept --gradcam flag."""
        args = parse_args(["batch", "-i", "in/", "-o", "out/", "-m", "m.onnx", "--gradcam"])
        assert args.gradcam is True

    def test_batch_format_csv(self) -> None:
        """Batch should accept --format csv."""
        args = parse_args(["batch", "-i", "in/", "-o", "out/", "-m", "m.onnx", "--format", "csv"])
        assert args.format == "csv"


class TestRunBatch:
    """Tests for the batch processing function."""

    def test_batch_json_output(
        self,
        sample_images_dir: Path,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Batch should produce a valid JSON report."""
        output_dir = tmp_path / "output"
        summary = run_batch(
            input_dir=str(sample_images_dir),
            output_dir=str(output_dir),
            model_path=str(onnx_model_path),
            threshold=0.5,
            output_format="json",
        )
        assert summary["total_images"] == 5
        assert "defects_found" in summary
        assert "defect_rate" in summary

        report_path = output_dir / "defect_report.json"
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "summary" in data
        assert len(data["results"]) == 5

    def test_batch_csv_output(
        self,
        sample_images_dir: Path,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Batch should produce a valid CSV report."""
        output_dir = tmp_path / "output_csv"
        run_batch(
            input_dir=str(sample_images_dir),
            output_dir=str(output_dir),
            model_path=str(onnx_model_path),
            output_format="csv",
        )
        report_path = output_dir / "defect_report.csv"
        assert report_path.exists()
        content = report_path.read_text()
        assert "filename" in content
        assert "is_defect" in content

    def test_batch_empty_directory(
        self,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Batch should handle empty input directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        summary = run_batch(
            input_dir=str(empty_dir),
            output_dir=str(tmp_path / "out"),
            model_path=str(onnx_model_path),
        )
        assert summary["total_images"] == 0
        assert summary["defects_found"] == 0

    def test_batch_with_gradcam(
        self,
        sample_images_dir: Path,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Batch should save Grad-CAM overlays when flag is set."""
        output_dir = tmp_path / "gradcam_output"
        run_batch(
            input_dir=str(sample_images_dir),
            output_dir=str(output_dir),
            model_path=str(onnx_model_path),
            threshold=0.0,
            save_gradcam=True,
        )
        assert output_dir.exists()

    def test_batch_threshold_affects_count(
        self,
        sample_images_dir: Path,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """High threshold should result in fewer defects."""
        out_low = tmp_path / "low"
        out_high = tmp_path / "high"
        summary_low = run_batch(
            input_dir=str(sample_images_dir),
            output_dir=str(out_low),
            model_path=str(onnx_model_path),
            threshold=0.01,
        )
        summary_high = run_batch(
            input_dir=str(sample_images_dir),
            output_dir=str(out_high),
            model_path=str(onnx_model_path),
            threshold=0.99,
        )
        assert summary_low["defects_found"] >= summary_high["defects_found"]


class TestWriteReport:
    """Tests for report writing."""

    def test_write_json(self, tmp_path: Path) -> None:
        """Should write a valid JSON report."""
        results = [{"filename": "a.png", "is_defect": True}]
        summary = {"total": 1}
        _write_report(tmp_path, summary, results, "json")
        path = tmp_path / "defect_report.json"
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["total"] == 1

    def test_write_csv(self, tmp_path: Path) -> None:
        """Should write a valid CSV report."""
        results = [{"filename": "a.png", "is_defect": True, "confidence": 0.9}]
        _write_report(tmp_path, {}, results, "csv")
        path = tmp_path / "defect_report.csv"
        assert path.exists()
        assert "filename" in path.read_text()


class TestMainCLI:
    """Tests for the main CLI entry point."""

    def test_main_batch(
        self,
        sample_images_dir: Path,
        onnx_model_path: Path,
        tmp_path: Path,
    ) -> None:
        """Main should run batch command successfully."""
        output_dir = tmp_path / "cli_out"
        result = main(
            [
                "--config",
                "configs/config.yaml",
                "batch",
                "-i",
                str(sample_images_dir),
                "-o",
                str(output_dir),
                "-m",
                str(onnx_model_path),
            ]
        )
        assert result == 0
        assert (output_dir / "defect_report.json").exists()

    def test_main_no_command(self) -> None:
        """No command should return 0."""
        result = main(["--config", "configs/config.yaml"])
        assert result == 0


class TestSoftmax:
    """Tests for the softmax utility."""

    def test_softmax_sums_to_one(self) -> None:
        """Softmax output should sum to 1."""
        logits = np.array([2.0, 1.0, 0.5])
        probs = _softmax(logits)
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)

    def test_softmax_positive(self) -> None:
        """All softmax values should be positive."""
        logits = np.array([-1.0, 0.0, 1.0])
        probs = _softmax(logits)
        assert (probs > 0).all()
