"""Configuration management with singleton pattern and dot-notation access."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Singleton configuration manager that loads YAML config files.

    Supports dot-notation key access for nested configuration values.
    Thread-safe singleton ensures consistent config across the application.

    Example:
        >>> config = Config()
        >>> config.load("configs/config.yaml")
        >>> config.get("data.image_size")
        224
    """

    _instance: "Config | None" = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str | Path = "configs/config.yaml") -> None:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            yaml.YAMLError: If the file contains invalid YAML.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self._config = yaml.safe_load(f) or {}

        logger.info("Loaded configuration from %s", config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-notation.

        Args:
            key: Dot-separated key path (e.g., "data.image_size").
            default: Value to return if the key is not found.

        Returns:
            The configuration value, or the default if not found.
        """
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def as_dict(self) -> dict[str, Any]:
        """Return the full configuration as a dictionary.

        Returns:
            A copy of the configuration dictionary.
        """
        return dict(self._config)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None
        cls._config = {}


config = Config()
