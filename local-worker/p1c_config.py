"""Side-effect-free configuration for the canonical Translator worker."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class P1CConfigurationError(ValueError):
    """Raised when canonical Translator configuration is incomplete."""


@dataclass(frozen=True)
class P1CSettings:
    translator_job_dir: Path
    media_dir: Path
    engine_status_dir: Path
    output_dir: Path
    control_plane_root: Path | None = None


def load_p1c_settings(
    environment: Mapping[str, str] | None = None,
    *,
    require_control_plane: bool = False,
) -> P1CSettings:
    """Parse paths only; never create directories or load legacy settings."""

    values = os.environ if environment is None else environment
    translator_job_dir = _required_path(values, "DUBVI_TRANSLATOR_JOB_DIR")
    media_dir = _required_path(values, "DUBVI_MEDIA_DIR")
    engine_status_dir = _required_path(values, "DUBVI_ENGINE_STATUS_DIR")
    output_dir = _required_path(values, "DUBVI_OUTPUT_DIR")
    control_plane_value = values.get("DUBVI_CONTROL_PLANE_ROOT")
    if require_control_plane and not control_plane_value:
        raise P1CConfigurationError("DUBVI_CONTROL_PLANE_ROOT is required for canonical execution")
    control_plane_root = _path(control_plane_value) if control_plane_value else None
    return P1CSettings(
        translator_job_dir=translator_job_dir,
        media_dir=media_dir,
        engine_status_dir=engine_status_dir,
        output_dir=output_dir,
        control_plane_root=control_plane_root,
    )


def _required_path(values: Mapping[str, str], name: str) -> Path:
    raw = values.get(name)
    if raw is None or not raw.strip():
        raise P1CConfigurationError(f"{name} must be configured")
    return _path(raw)


def _path(raw: str) -> Path:
    if "\x00" in raw:
        raise P1CConfigurationError("configured paths must not contain NUL")
    return Path(raw).expanduser().resolve(strict=False)
