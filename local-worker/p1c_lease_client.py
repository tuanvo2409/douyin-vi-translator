"""Offline argv-only bridge client for Translator HEAVY_MEDIA leases."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


BRIDGE_TIMEOUT_SECONDS = 10
MAX_BRIDGE_STDOUT = 64 * 1024
_CANONICAL_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


class LeaseBridgeUnavailable(RuntimeError):
    """The Control Plane lease subprocess was unavailable or unsafe."""


@dataclass(frozen=True)
class LeaseBridgeResponse:
    action: str
    granted: bool | None
    heartbeat_seconds: int
    ttl_seconds: int
    lease: Mapping[str, object] | None


class TranslatorLeaseBridgeClient:
    """Call the local Control Plane bridge with a bounded child process."""

    def __init__(
        self,
        control_plane_root: Path,
        *,
        executable: str | None = None,
        timeout: int = BRIDGE_TIMEOUT_SECONDS,
    ) -> None:
        root = Path(control_plane_root)
        if (
            root.is_symlink()
            or not root.is_dir()
            or not (root / "src" / "dubvi_control_plane").is_dir()
        ):
            raise LeaseBridgeUnavailable("DUBVI_CONTROL_PLANE_ROOT is not a valid Control Plane root")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise LeaseBridgeUnavailable("lease bridge timeout must be a positive integer")
        self._root = root.resolve()
        self._executable = executable or sys.executable
        self._timeout = timeout

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        executable: str | None = None,
    ) -> "TranslatorLeaseBridgeClient":
        values = os.environ if environment is None else environment
        raw = values.get("DUBVI_CONTROL_PLANE_ROOT")
        if raw is None or not raw.strip():
            raise LeaseBridgeUnavailable("DUBVI_CONTROL_PLANE_ROOT is required")
        return cls(Path(raw), executable=executable)

    def acquire(self, job: Mapping[str, object]) -> LeaseBridgeResponse:
        return self._call("acquire", job)

    def heartbeat(self, job: Mapping[str, object], lease_id: str) -> LeaseBridgeResponse:
        return self._call("heartbeat", job, lease_id)

    def release(self, job: Mapping[str, object], lease_id: str) -> LeaseBridgeResponse:
        return self._call("release", job, lease_id)

    def _call(
        self,
        action: str,
        job: Mapping[str, object],
        lease_id: str | None = None,
    ) -> LeaseBridgeResponse:
        if action not in {"acquire", "heartbeat", "release"}:
            raise LeaseBridgeUnavailable("unsupported lease bridge action")
        job_id = _canonical_uuid(job.get("translator_job_id"), "translator_job_id")
        correlation = _canonical_uuid(job.get("dispatch_id"), "dispatch_id")
        argv = [
            self._executable,
            "-m",
            "dubvi_control_plane.engine_lease_bridge",
            action,
            "--engine-kind",
            "translator",
            "--job-id",
            job_id,
            "--correlation-id",
            correlation,
        ]
        if lease_id is not None:
            argv.extend(["--lease-id", _canonical_uuid(lease_id, "lease_id")])
        elif action != "acquire":
            raise LeaseBridgeUnavailable("lease_id is required")

        child_environment = os.environ.copy()
        source_root = str(self._root / "src")
        child_environment["PYTHONPATH"] = source_root + (
            os.pathsep + child_environment["PYTHONPATH"]
            if child_environment.get("PYTHONPATH")
            else ""
        )
        try:
            result = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env=child_environment,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LeaseBridgeUnavailable("lease bridge is unavailable") from error
        if result.returncode != 0:
            raise LeaseBridgeUnavailable("lease bridge returned a nonzero result")
        if len(result.stdout.encode("utf-8")) > MAX_BRIDGE_STDOUT:
            raise LeaseBridgeUnavailable("lease bridge response is oversized")
        try:
            value = json.loads(result.stdout)
        except (TypeError, ValueError) as error:
            raise LeaseBridgeUnavailable("lease bridge returned malformed JSON") from error
        return _validate_response(value, action, job_id, lease_id)


def _validate_response(
    value: object,
    action: str,
    job_id: str,
    expected_lease_id: str | None = None,
) -> LeaseBridgeResponse:
    if not isinstance(value, dict):
        raise LeaseBridgeUnavailable("lease bridge response must be an object")
    allowed = {"bridge_version", "action", "ok", "engine_kind", "timing", "lease", "granted"}
    if set(value) - allowed:
        raise LeaseBridgeUnavailable("lease bridge response shape is unsafe")
    if (
        value.get("bridge_version") != 1
        or value.get("action") != action
        or value.get("ok") is not True
        or value.get("engine_kind") != "translator"
    ):
        raise LeaseBridgeUnavailable("lease bridge response identity is unsafe")
    timing = value.get("timing")
    if not isinstance(timing, dict) or set(timing) != {"heartbeat_seconds", "ttl_seconds"}:
        raise LeaseBridgeUnavailable("lease bridge timing is invalid")
    heartbeat = timing.get("heartbeat_seconds")
    ttl = timing.get("ttl_seconds")
    if (
        isinstance(heartbeat, bool)
        or isinstance(ttl, bool)
        or not isinstance(heartbeat, int)
        or not isinstance(ttl, int)
        or heartbeat <= 0
        or ttl < 4 * heartbeat
    ):
        raise LeaseBridgeUnavailable("lease bridge timing is invalid")
    if action == "acquire" and not isinstance(value.get("granted"), bool):
        raise LeaseBridgeUnavailable("lease bridge acquire response is invalid")
    lease = value.get("lease")
    if lease is not None:
        if not isinstance(lease, dict):
            raise LeaseBridgeUnavailable("lease bridge lease is invalid")
        expected_keys = {
            "lease_id", "resource_class", "owner", "job_id", "state",
            "acquired_at", "heartbeat_at", "expires_at", "released_at",
        }
        if set(lease) != expected_keys:
            raise LeaseBridgeUnavailable("lease bridge lease shape is invalid")
        if (
            lease.get("resource_class") != "HEAVY_MEDIA"
            or lease.get("owner") != f"translator:{job_id}"
            or lease.get("job_id") != job_id
        ):
            raise LeaseBridgeUnavailable("lease bridge lease identity is invalid")
        _canonical_uuid(lease.get("lease_id"), "lease_id")
        if expected_lease_id is not None and lease["lease_id"] != expected_lease_id:
            raise LeaseBridgeUnavailable("lease bridge returned an unrelated lease")
        expected_state = "active" if action != "release" else "released"
        if lease.get("state") != expected_state:
            raise LeaseBridgeUnavailable("lease bridge lease state is invalid")
        acquired = _timestamp(lease.get("acquired_at"), "acquired_at")
        heartbeat_at = _timestamp(lease.get("heartbeat_at"), "heartbeat_at")
        expires = _timestamp(lease.get("expires_at"), "expires_at")
        if not acquired <= heartbeat_at < expires:
            raise LeaseBridgeUnavailable("lease bridge lease timestamps are not ordered")
        if expected_state == "active":
            if lease.get("released_at") is not None:
                raise LeaseBridgeUnavailable("active bridge lease must not have released_at")
        else:
            released_at = _timestamp(lease.get("released_at"), "released_at")
            if released_at < heartbeat_at:
                raise LeaseBridgeUnavailable("released_at precedes lease heartbeat")
    elif action != "acquire" or value.get("granted") is not None:
        raise LeaseBridgeUnavailable("lease bridge response is missing its lease")
    if action == "acquire" and value["granted"] != (lease is not None):
        raise LeaseBridgeUnavailable("lease bridge acquire decision is invalid")
    return LeaseBridgeResponse(action, value.get("granted"), heartbeat, ttl, lease)


def _canonical_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise LeaseBridgeUnavailable(f"{field} is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as error:
        raise LeaseBridgeUnavailable(f"{field} is invalid") from error
    if value != str(parsed):
        raise LeaseBridgeUnavailable(f"{field} is invalid")
    return value


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or _CANONICAL_TIMESTAMP.fullmatch(value) is None:
        raise LeaseBridgeUnavailable(f"{field} is not canonical UTC milliseconds")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise LeaseBridgeUnavailable(f"{field} is not a real calendar timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
        raise LeaseBridgeUnavailable(f"{field} is not UTC")
    return parsed
