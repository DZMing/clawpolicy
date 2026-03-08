#!/usr/bin/env python3
"""Cross-platform application paths for ClawPolicy."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union


APP_NAME = "clawpolicy"


def _home() -> Path:
    return Path.home()


def get_config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", _home() / "AppData/Roaming"))
        return base / APP_NAME
    base = Path(os.environ.get("XDG_CONFIG_HOME", _home() / ".config"))
    return base / APP_NAME


def get_cache_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", _home() / "AppData/Local"))
        return base / APP_NAME / "cache"
    base = Path(os.environ.get("XDG_CACHE_HOME", _home() / ".cache"))
    return base / APP_NAME


def get_state_dir() -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", _home() / "AppData/Local"))
        return base / APP_NAME / "state"
    base = Path(os.environ.get("XDG_STATE_HOME", _home() / ".local/state"))
    return base / APP_NAME


def get_default_config_path() -> Path:
    return get_config_dir() / "config.json"


def get_default_model_dir() -> Path:
    return get_cache_dir() / "models" / "rl"


def resolve_config_path(config_path: Optional[Union[str, Path]] = None) -> Path:
    if config_path:
        return Path(config_path).expanduser()
    return get_default_config_path()


def resolve_model_dir(model_path: Optional[Union[str, Path]] = None) -> Path:
    if model_path:
        return Path(model_path).expanduser()
    return get_default_model_dir()


def get_local_config_path(cwd: Optional[Union[str, Path]] = None) -> Path:
    """Return project-local .clawpolicy/config.json path (3.0.0 default)."""
    base = Path(cwd or Path.cwd())
    return base / ".clawpolicy" / "config.json"


def resolve_local_config_path(
    config_path: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve config path with project-local default.

    Args:
        config_path: Explicit config path (takes precedence)
        cwd: Working directory for project-local default

    Returns:
        Resolved config path. Uses .clawpolicy/config.json in cwd by default.
    """
    if config_path:
        return Path(config_path).expanduser()
    return get_local_config_path(cwd)
