"""Entry point for the defect detection pipeline.

Provides CLI commands for training, exporting, serving, and batch processing.
"""

import argparse
import sys

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
        logger.info("Training command selected")
        # Training logic is implemented in task 5
        return 0

    if args.command == "export":
        logger.info("Export command selected")
        # Export logic is implemented in task 9
        return 0

    if args.command == "serve":
        logger.info("Serve command selected")
        # Serve logic is implemented in task 10
        return 0

    if args.command == "batch":
        logger.info("Batch command selected")
        # Batch logic is implemented in task 11
        return 0

    logger.error("Unknown command: %s", args.command)
    return 1


if __name__ == "__main__":
    sys.exit(main())
