"""Tests for logging setup."""

import logging
from pathlib import Path

from src.utils.logger import setup_logger


def test_logger_returns_logger() -> None:
    """setup_logger should return a logging.Logger instance."""
    logger = setup_logger("test_basic")
    assert isinstance(logger, logging.Logger)


def test_logger_has_console_handler() -> None:
    """Logger should have at least one StreamHandler."""
    name = "test_console_handler"
    logger = setup_logger(name)
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types


def test_logger_level() -> None:
    """Logger should respect the specified level."""
    logger = setup_logger("test_level", level=logging.DEBUG)
    assert logger.level == logging.DEBUG


def test_logger_with_file(tmp_path: Path) -> None:
    """Logger should create a file handler when log_file is specified."""
    log_file = tmp_path / "logs" / "test.log"
    logger = setup_logger("test_file_output", log_file=str(log_file))

    logger.info("test message")

    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content


def test_logger_no_duplicate_handlers() -> None:
    """Calling setup_logger twice should not duplicate handlers."""
    name = "test_no_duplicates"
    logger1 = setup_logger(name)
    handler_count = len(logger1.handlers)
    logger2 = setup_logger(name)
    assert len(logger2.handlers) == handler_count
    assert logger1 is logger2


def test_logger_creates_parent_dirs(tmp_path: Path) -> None:
    """Logger should create parent directories for log file."""
    log_file = tmp_path / "deep" / "nested" / "dir" / "test.log"
    logger = setup_logger("test_parent_dirs", log_file=str(log_file))
    logger.info("deep test")
    assert log_file.parent.exists()
