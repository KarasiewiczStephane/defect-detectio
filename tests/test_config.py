"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import Config


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    """Reset singleton between tests."""
    Config.reset()


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    data = {
        "data": {
            "root_dir": "data",
            "image_size": 224,
            "categories": ["bottle", "carpet"],
        },
        "model": {
            "architecture": "resnet50",
            "num_classes": 2,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    }
    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def test_config_singleton() -> None:
    """Config should always return the same instance."""
    c1 = Config()
    c2 = Config()
    assert c1 is c2


def test_config_load(config_file: Path) -> None:
    """Config should load YAML file successfully."""
    cfg = Config()
    cfg.load(config_file)
    assert cfg.get("data.image_size") == 224


def test_config_nested_access(config_file: Path) -> None:
    """Dot-notation access should resolve nested keys."""
    cfg = Config()
    cfg.load(config_file)
    assert cfg.get("model.architecture") == "resnet50"
    assert cfg.get("training.batch_size") == 32


def test_config_default_value(config_file: Path) -> None:
    """Missing keys should return the default value."""
    cfg = Config()
    cfg.load(config_file)
    assert cfg.get("nonexistent.key", "fallback") == "fallback"


def test_config_missing_file() -> None:
    """Loading a nonexistent file should raise FileNotFoundError."""
    cfg = Config()
    with pytest.raises(FileNotFoundError):
        cfg.load("/nonexistent/config.yaml")


def test_config_as_dict(config_file: Path) -> None:
    """as_dict should return the full config dictionary."""
    cfg = Config()
    cfg.load(config_file)
    d = cfg.as_dict()
    assert "data" in d
    assert "model" in d


def test_config_list_access(config_file: Path) -> None:
    """Config should handle list values."""
    cfg = Config()
    cfg.load(config_file)
    categories = cfg.get("data.categories")
    assert isinstance(categories, list)
    assert "bottle" in categories


def test_config_loads_real_config() -> None:
    """Real config.yaml should load without errors."""
    real_config = Path("configs/config.yaml")
    if real_config.exists():
        cfg = Config()
        cfg.load(real_config)
        assert cfg.get("data.image_size") == 224
