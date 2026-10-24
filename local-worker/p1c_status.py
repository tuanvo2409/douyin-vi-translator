"""Canonical Translator immutable status-event writer and reader."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Mapping

from dubvi_engine_contract import (
    ContractValidationError,
    canonical_json_bytes,
    parse_json_document,
    validate_engine_status_event,
)

from p1c_config import P1CSettings


MAX_STATUS_BYTES = 64 * 1024


class TranslatorStatusError(RuntimeError):
    """Base canonical status error."""


class TranslatorStatusConflict(TranslatorStatusError):
    """An immutable sequence already contains different evidence."""


def publish_status_event(settings: P1CSettings, document: Mapping[str, object]) -> Path:
    """Validate and publish one immutable non-replacing status event."""

    try:
        validated = validate_engine_status_event(document)
        payload = canonical_json_bytes(validated)
    except ContractValidationError as error:
        raise TranslatorStatusError(f"invalid Translator status event: {error}") from error
    if len(payload) > MAX_STATUS_BYTES:
        raise TranslatorStatusError("Translator status event exceeds 64KiB")
    if validated["engine_kind"] != "translator":
        raise TranslatorStatusError("Translator status writer accepts translator events only")
    root = _require_root(settings.engine_status_dir)
    directory = root / "translator" / str(validated["engine_job_id"])
    _ensure_child_directory(root / "translator", directory)
    sequence = int(validated["sequence"])
    final = directory / f"event-{sequence:06d}.json"
    if _real_file(final):
        if _matches(final, payload):
            return final
        raise TranslatorStatusConflict("existing Translator status sequence differs")
    if final.is_symlink():
        raise TranslatorStatusConflict("Translator status final is an unsafe symlink")
    fingerprint = _fingerprint(payload)
    stage = directory / f".event-{sequence:06d}.{fingerprint[7:19]}.part"
    _stage_exact(stage, payload)
    try:
        os.link(stage, final)
    except FileExistsError:
        if _matches(final, payload):
            return final
        raise TranslatorStatusConflict("racing Translator status sequence differs")
    except OSError as error:
        raise TranslatorStatusError(f"Translator status publication failed: {error}") from error
    if not _matches(final, payload):
        raise TranslatorStatusError("published Translator status failed verification")
    stage.unlink(missing_ok=True)
    return final


def load_status_event(
    settings: P1CSettings, engine_job_id: str, sequence: int
) -> dict[str, object]:
    """Load one exact canonical Translator status event."""

    root = _require_root(settings.engine_status_dir)
    path = root / "translator" / engine_job_id / f"event-{sequence:06d}.json"
    _assert_under(root, path)
    if path.is_symlink() or not path.is_file():
        raise TranslatorStatusError("Translator status event is absent or unsafe")
    size = path.stat().st_size
    if size <= 0 or size > MAX_STATUS_BYTES:
        raise TranslatorStatusError("Translator status event has an unsafe size")
    payload = path.read_bytes()
    if len(payload) != size:
        raise TranslatorStatusError("Translator status changed while read")
    try:
        document = validate_engine_status_event(parse_json_document(payload))
    except ContractValidationError as error:
        raise TranslatorStatusError(f"invalid Translator status event: {error}") from error
    if canonical_json_bytes(document) != payload:
        raise TranslatorStatusError("Translator status is not canonical bytes")
    if document["engine_kind"] != "translator" or document["engine_job_id"] != engine_job_id or document["sequence"] != sequence:
        raise TranslatorStatusError("Translator status identity does not match its path")
    expected_pairs = {
        1: {("accepted", "accepted")},
        2: {("started", "running")},
        3: {
            ("succeeded", "succeeded"),
            ("failed", "failed"),
            ("rejected", "rejected"),
            ("review_required", "review_required"),
            ("lease_lost", "running"),
        },
    }
    if sequence not in expected_pairs or (document["event_kind"], document["state"]) not in expected_pairs[sequence]:
        raise TranslatorStatusError("Translator status sequence has an illegal event kind")
    return document


def _require_root(root: Path) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise TranslatorStatusError("DUBVI_ENGINE_STATUS_DIR must be an existing non-symlink directory")
    return root.resolve()


def _ensure_child_directory(parent: Path, directory: Path) -> None:
    if parent.is_symlink():
        raise TranslatorStatusError("Translator status directory is unsafe")
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir() or directory.is_symlink():
        raise TranslatorStatusError("Translator status directory is unsafe")
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise TranslatorStatusError("Translator status directory is unsafe")


def _stage_exact(path: Path, payload: bytes) -> None:
    if _real_file(path):
        if _matches(path, payload):
            return
        raise TranslatorStatusConflict("existing Translator status stage differs")
    if path.is_symlink():
        raise TranslatorStatusConflict("Translator status stage is unsafe")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if _matches(path, payload):
            return
        raise TranslatorStatusConflict("racing Translator status stage differs")


def _real_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _fingerprint(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _matches(path: Path, payload: bytes) -> bool:
    return _real_file(path) and path.stat().st_size == len(payload) and _hash(path) == _fingerprint(payload)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _assert_under(root: Path, path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise TranslatorStatusError("Translator status path escapes its root") from error
