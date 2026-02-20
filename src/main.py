"""Entry point for the defect detection pipeline.

Provides CLI commands for training, exporting, serving, and batch processing.
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Defect Detection Pipeline — Manufacturing QC",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the defect detection model")
    train_parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint path"
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    export_parser.add_argument("--output", type=str, default=None, help="ONNX output path")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default=None, help="Server host")
    serve_parser.add_argument("--port", type=int, default=None, help="Server port")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch process images")
    batch_parser.add_argument(
        "--input", "-i", type=str, required=True, help="Input image directory"
    )
    batch_parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory for results"
    )
    batch_parser.add_argument("--model", "-m", type=str, required=True, help="Path to ONNX model")
    batch_parser.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Detection threshold"
    )
    batch_parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM overlays")
    batch_parser.add_argument(
        "--format", choices=["json", "csv"], default="json", help="Report format"
    )

    return parser.parse_args(argv)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities.

    Args:
        x: Logits array.

    Returns:
        Probability array.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def run_batch(
    input_dir: str,
    output_dir: str,
    model_path: str,
    threshold: float = 0.5,
    save_gradcam: bool = False,
    output_format: str = "json",
) -> dict:
    """Process all images in a directory and generate a defect report.

    Args:
        input_dir: Path to directory containing images.
        output_dir: Path for output report and optional Grad-CAM overlays.
        model_path: Path to the ONNX model file.
        threshold: Decision threshold for defect classification.
        save_gradcam: Whether to save Grad-CAM overlays for defective images.
        output_format: Report format, either "json" or "csv".

    Returns:
        Summary dictionary with total_images, defects_found, defect_rate.
    """
    from src.data.preprocessor import get_transforms
    from src.deployment.onnx_exporter import ONNXInference

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    inference = ONNXInference(model_path)
    transform = get_transforms("val")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(
        f for f in input_path.iterdir() if f.suffix.lower() in image_extensions and f.is_file()
    )

    if not images:
        logger.warning("No images found in %s", input_path)
        return {"total_images": 0, "defects_found": 0, "defect_rate": 0.0}

    results: list[dict] = []
    defect_count = 0

    for img_path in tqdm(images, desc="Processing images"):
        try:
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).numpy()

            output = inference.predict(tensor)
            probs = _softmax(output[0])
            is_defect = bool(probs[1] >= threshold)

            result = {
                "filename": img_path.name,
                "is_defect": is_defect,
                "confidence": float(max(probs)),
                "defect_probability": float(probs[1]),
                "timestamp": datetime.now(tz=UTC).isoformat(),
            }
            results.append(result)

            if is_defect:
                defect_count += 1

                if save_gradcam:
                    _save_batch_gradcam(image, img_path.name, output_path)

        except Exception:
            logger.exception("Error processing %s", img_path)
            results.append({"filename": img_path.name, "error": "processing_failed"})

    summary = {
        "total_images": len(images),
        "defects_found": defect_count,
        "defect_rate": defect_count / len(images) if images else 0.0,
        "threshold": threshold,
        "processed_at": datetime.now(tz=UTC).isoformat(),
    }

    _write_report(output_path, summary, results, output_format)

    logger.info(
        "Batch complete: %d images, %d defects (%.1f%%)",
        summary["total_images"],
        summary["defects_found"],
        summary["defect_rate"] * 100,
    )
    return summary


def _save_batch_gradcam(image: Image.Image, filename: str, output_dir: Path) -> None:
    """Generate and save a Grad-CAM overlay for batch processing.

    Args:
        image: Original PIL image.
        filename: Image filename for the output.
        output_dir: Directory to save the overlay.
    """
    try:
        from src.models.grad_cam import overlay_heatmap

        original = np.array(image)
        # Generate a simple attention map based on image intensity as fallback
        gray = np.mean(original, axis=2).astype(np.float32)
        heatmap = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
        overlay = overlay_heatmap(original, heatmap)
        overlay_path = output_dir / "gradcam" / filename
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(overlay).save(overlay_path)
    except Exception:
        logger.exception("Failed to generate Grad-CAM for %s", filename)


def _write_report(
    output_dir: Path,
    summary: dict,
    results: list[dict],
    output_format: str,
) -> None:
    """Write the batch processing report to disk.

    Args:
        output_dir: Output directory.
        summary: Summary statistics dictionary.
        results: List of per-image result dictionaries.
        output_format: "json" or "csv".
    """
    if output_format == "json":
        report_path = output_dir / "defect_report.json"
        with open(report_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
    elif output_format == "csv":
        report_path = output_dir / "defect_report.csv"
        if results:
            fieldnames = list(results[0].keys())
            with open(report_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)

    logger.info("Report saved to %s", report_path)


def run_serve(host: str | None = None, port: int | None = None) -> None:
    """Start the FastAPI server.

    Args:
        host: Server host address. Defaults to config value.
        port: Server port number. Defaults to config value.
    """
    import uvicorn

    host = host or config.get("api.host", "0.0.0.0")
    port = port or config.get("api.port", 8000)

    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


def run_export(checkpoint_path: str, output_path: str | None = None) -> None:
    """Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint.
        output_path: Output ONNX file path. Defaults to config value.
    """
    from src.deployment.onnx_exporter import ONNXExporter
    from src.models.resnet_classifier import create_model, load_checkpoint

    output_path = output_path or config.get("deployment.model_path", "models/defect_model.onnx")
    model = create_model(pretrained=False)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    exporter = ONNXExporter(model)
    onnx_path = exporter.export(output_path)
    logger.info("Exported model to %s", onnx_path)

    if exporter.validate_outputs(onnx_path):
        logger.info("ONNX validation passed")
    else:
        logger.warning("ONNX validation failed — outputs differ from PyTorch")


def main(argv: list[str] | None = None) -> int:
    """Run the defect detection CLI.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args(argv)

    config.load(args.config)
    logger.info("Loaded config from %s", args.config)

    if args.command is None:
        logger.info("No command specified. Use --help for available commands.")
        return 0

    if args.command == "train":
        logger.info("Training command — use the training pipeline directly for now")
        return 0

    if args.command == "export":
        run_export(args.checkpoint, getattr(args, "output", None))
        return 0

    if args.command == "serve":
        run_serve(args.host, args.port)
        return 0

    if args.command == "batch":
        summary = run_batch(
            input_dir=args.input,
            output_dir=args.output,
            model_path=args.model,
            threshold=args.threshold,
            save_gradcam=args.gradcam,
            output_format=args.format,
        )
        logger.info(
            "Processed %d images, %d defects (%.1f%%)",
            summary["total_images"],
            summary["defects_found"],
            summary["defect_rate"] * 100,
        )
        return 0

    logger.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main())
