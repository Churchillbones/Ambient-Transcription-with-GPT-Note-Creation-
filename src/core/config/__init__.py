from __future__ import annotations

"""Core configuration package.

Exposes a singleton :pydata:`config_service` that loads `ApplicationSettings`
from environment variables and registers itself with the application's global
DI container.
"""

from .settings import ApplicationSettings
from .configuration_service import ConfigurationService
from ..container import global_container
from ..interfaces.config_service import IConfigurationService

# ---------------------------------------------------------------------------
# Instantiate & register the configuration service exactly once at import time
# ---------------------------------------------------------------------------

_config_settings = ApplicationSettings()
config_service = ConfigurationService(settings=_config_settings)

global_container.register_instance(IConfigurationService, config_service)

__all__ = ["config_service", "ApplicationSettings"] 