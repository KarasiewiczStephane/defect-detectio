"""Tests for the CLI entry point."""

from src.main import main, parse_args


def test_parse_args_no_command() -> None:
    """Parsing with no arguments should return command=None."""
    args = parse_args([])
    assert args.command is None


def test_parse_args_train() -> None:
    """Train command should parse correctly."""
    args = parse_args(["train"])
    assert args.command == "train"


def test_parse_args_train_with_epochs() -> None:
    """Train command should accept --epochs flag."""
    args = parse_args(["train", "--epochs", "10"])
    assert args.epochs == 10


def test_parse_args_export() -> None:
    """Export command should require --checkpoint."""
    args = parse_args(["export", "--checkpoint", "model.pt"])
    assert args.command == "export"
    assert args.checkpoint == "model.pt"


def test_parse_args_serve() -> None:
    """Serve command should parse host and port."""
    args = parse_args(["serve", "--host", "127.0.0.1", "--port", "9000"])
    assert args.host == "127.0.0.1"
    assert args.port == 9000


def test_parse_args_batch() -> None:
    """Batch command should parse all required arguments."""
    args = parse_args(["batch", "-i", "images/", "-o", "out/", "-m", "model.onnx"])
    assert args.command == "batch"
    assert args.input == "images/"
    assert args.output == "out/"
    assert args.model == "model.onnx"


def test_main_no_command() -> None:
    """Running with no command should return 0."""
    result = main(["--config", "configs/config.yaml"])
    assert result == 0


def test_main_train() -> None:
    """Train command should run and return 0."""
    result = main(["--config", "configs/config.yaml", "train"])
    assert result == 0
