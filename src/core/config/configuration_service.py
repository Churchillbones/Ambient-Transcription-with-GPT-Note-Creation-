from __future__ import annotations

from pathlib import Path
from typing import Any

from ..exceptions import ConfigurationError
from ..interfaces.config_service import IConfigurationService
from .settings import ApplicationSettings

__all__ = ["ConfigurationService"]


class ConfigurationService(IConfigurationService):
    """Runtime wrapper around :class:`ApplicationSettings`."""

    def __init__(self, *, settings: ApplicationSettings) -> None:
        self._settings = settings
        if not self.validate_configuration():
            raise ConfigurationError("Invalid application configuration detected")

    # ------------------------------------------------------------------
    # IConfigurationService implementation
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:  # noqa: D401
        parts = key.split(".")
        current: Any = self._settings
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        return current

    def validate_configuration(self) -> bool:  # noqa: D401
        # Minimal sanity checks – extend as needed.
        try:
            # Ensure base directory exists and writable
            base_dir: Path = getattr(self._settings, "base_dir", Path("./app_data"))
            base_dir.mkdir(parents=True, exist_ok=True)
            test_path = base_dir / ".write_test"
            test_path.touch(exist_ok=True)
            test_path.unlink(missing_ok=True)

            # Azure endpoint validation already handled by pydantic validator.
            return True
        except Exception as exc:  # pragma: no cover – unexpected env/fs issues
            raise ConfigurationError(f"Configuration validation failed: {exc}") from exc 