"""Filesystem locations for LocalRedactor.

Uses platformdirs so paths are correct on Windows (%APPDATA%\\LocalRedactor),
and still sane on Linux/macOS for development. Nothing here ever stores
document text — only settings, profiles, status logs, and cached models.
"""
from __future__ import annotations

from pathlib import Path

from platformdirs import PlatformDirs

APP_NAME = "LocalRedactor"
APP_AUTHOR = "LocalRedactor"

_dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR, roaming=True)


def config_dir() -> Path:
    """%APPDATA%\\LocalRedactor on Windows."""
    p = Path(_dirs.user_config_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def settings_path() -> Path:
    return config_dir() / "settings.json"


def profiles_db_path() -> Path:
    return config_dir() / "profiles.sqlite"


def logs_dir() -> Path:
    p = config_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_models_dir() -> Path:
    """Fallback model cache when models are not bundled under app/models."""
    p = config_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def bundled_models_dir() -> Path:
    """Models shipped with the app (preferred for offline use)."""
    # app/config/paths.py -> app/ -> project root -> models/
    return Path(__file__).resolve().parents[2] / "models"
