"""Canonical Translator job-envelope intake for CP-managed work."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping

from dubvi_engine_contract import (
    ContractValidationError,
    canonical_json_bytes,
    parse_json_document,
    validate_control_plane_to_translator_job,
    validate_reup_to_translator_handoff,
)

from p1c_config import P1CSettings
from p1c_status import (
    TranslatorStatusError,
    load_status_event,
    publish_status_event,
)


MAX_ENVELOPE_BYTES = 64 * 1024
_ENVELOPE_RE = re.compile(r"^translator-job-([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})\.job\.json$")


class TranslatorIntakeError(RuntimeError):
    """Canonical intake failed closed."""


@dataclass(frozen=True)
class AcceptedTranslatorJob:
    envelope_path: Path
    handoff_media_path: Path
    handoff_sidecar_path: Path
    status_path: Path
    document: Mapping[str, object]
    handoff_document: Mapping[str, object]


def iter_translator_envelopes(settings: P1CSettings) -> Iterator[Path]:
    """Yield candidate envelope paths in deterministic lexical order."""

    root = _require_directory(settings.translator_job_dir, "DUBVI_TRANSLATOR_JOB_DIR")
    paths = [
        path for path in root.iterdir()
        if path.name.startswith("translator-job-") and path.name.endswith(".job.json")
    ]
    yield from sorted(paths, key=lambda item: item.name)


def load_translator_envelope(
    path: Path,
    settings: P1CSettings,
) -> AcceptedTranslatorJob:
    """Validate one envelope and its exact immutable v2 handoff pair."""

    root = _require_directory(settings.translator_job_dir, "DUBVI_TRANSLATOR_JOB_DIR")
    if path.is_symlink() or not path.is_file():
        raise TranslatorIntakeError("Translator envelope must be a regular non-symlink file")
    _assert_under(root, path)
    payload = _bounded_read(path, "Translator envelope")
    try:
        parsed = parse_json_document(payload)
        document = validate_control_plane_to_translator_job(parsed)
    except ContractValidationError as error:
        raise TranslatorIntakeError(f"invalid Translator envelope: {error}") from error
    if canonical_json_bytes(document) != payload:
        raise TranslatorIntakeError("Translator envelope is not canonical bytes")
    match = _ENVELOPE_RE.fullmatch(path.name)
    if match is None or document["translator_job_id"] != match.group(1):
        raise TranslatorIntakeError("Translator envelope filename does not match translator_job_id")
    if document.get("attempt_number") != 1 or "parent_translator_job_id" in document:
        raise TranslatorIntakeError("CP6 first Translator attempt has invalid retry metadata")

    media_root = _require_directory(settings.media_dir, "DUBVI_MEDIA_DIR")
    media_path = _resolve_ref(media_root, str(document["handoff_media_ref"]), "handoff_media_ref")
    sidecar_path = _resolve_ref(media_root, str(document["handoff_sidecar_ref"]), "handoff_sidecar_ref")
    sidecar_payload = _bounded_read(sidecar_path, "Reup handoff sidecar")
    try:
        handoff = validate_reup_to_translator_handoff(parse_json_document(sidecar_payload))
    except ContractValidationError as error:
        raise TranslatorIntakeError(f"invalid Reup v2 handoff: {error}") from error
    if canonical_json_bytes(handoff) != sidecar_payload:
        raise TranslatorIntakeError("Reup handoff sidecar is not canonical bytes")
    if media_path.is_symlink() or not media_path.is_file() or media_path.stat().st_size <= 0:
        raise TranslatorIntakeError("Reup handoff media is absent, empty, or unsafe")
    media_fingerprint = _hash(media_path)
    if media_fingerprint != document["reup_output_fingerprint"]:
        raise TranslatorIntakeError("Reup handoff media fingerprint differs")

    expected_profile = str(handoff["reup_profile"])
    expected_job = str(handoff["reup_job_id"])
    expected_sidecar_ref = f"{expected_profile}/reup-{expected_job}.meta.json"
    expected_media_ref = f"{expected_profile}/reup-{expected_job}.mp4"
    if (
        document["handoff_sidecar_ref"] != expected_sidecar_ref
        or document["handoff_media_ref"] != expected_media_ref
        or sidecar_path.name != f"reup-{expected_job}.meta.json"
        or media_path.name != f"reup-{expected_job}.mp4"
    ):
        raise TranslatorIntakeError("Reup handoff physical naming is not canonical")
    identity_fields = (
        "reup_job_id", "dispatch_id", "correlation_id", "candidate_id", "schedule_id",
        "channel_id", "channel_slug", "target_platform", "localization_profile",
        "policy_profile", "handoff_id", "reup_output_fingerprint",
    )
    for field in identity_fields:
        if document.get(field) != handoff.get(field):
            raise TranslatorIntakeError(f"Translator envelope/handoff mismatch: {field}")

    status_path = _status_path(settings, str(document["translator_job_id"]), 1)
    return AcceptedTranslatorJob(path, media_path, sidecar_path, status_path, document, handoff)


def accept_translator_envelope(
    path: Path,
    settings: P1CSettings,
    *,
    occurred_at_utc: str | None = None,
) -> AcceptedTranslatorJob:
    """Validate intake, then publish/reuse only sequence-one accepted."""

    accepted = load_translator_envelope(path, settings)
    job_id = str(accepted.document["translator_job_id"])
    if accepted.status_path.exists() or accepted.status_path.is_symlink():
        try:
            existing = load_status_event(settings, job_id, 1)
        except TranslatorStatusError as error:
            raise TranslatorIntakeError(f"existing accepted status is invalid: {error}") from error
        if not _accepted_identity_matches(existing, accepted.document):
            raise TranslatorIntakeError("existing accepted status conflicts with the envelope")
        return accepted
    event = {
        "contract_version": 1,
        "message_kind": "engine_status_event",
        "event_id": str(uuid.uuid4()),
        "engine_kind": "translator",
        "engine_job_id": job_id,
        "dispatch_id": accepted.document["dispatch_id"],
        "correlation_id": accepted.document["dispatch_id"],
        "sequence": 1,
        "attempt_number": accepted.document["attempt_number"],
        "event_kind": "accepted",
        "state": "accepted",
        "occurred_at_utc": _normalize_timestamp(occurred_at_utc or _now_iso()),
        "handoff_id": accepted.document["handoff_id"],
        "handoff_ref": accepted.document["handoff_sidecar_ref"],
    }
    try:
        publish_status_event(settings, event)
    except TranslatorStatusError as error:
        raise TranslatorIntakeError(str(error)) from error
    return accepted


def accept_next_translator_job(
    settings: P1CSettings,
    *,
    occurred_at_utc: str | None = None,
) -> AcceptedTranslatorJob | None:
    """Accept the first admission-ready envelope in lexical order.

    Immutable status evidence, rather than output files or envelope metadata,
    determines whether an envelope is runnable.  A started attempt without a
    terminal event is skipped for automatic queue progress and must be handled
    by explicit reconciliation; a terminal sequence-three event is never
    rerun.
    """

    for path in iter_translator_envelopes(settings):
        accepted = accept_translator_envelope(path, settings, occurred_at_utc=occurred_at_utc)
        job_id = str(accepted.document["translator_job_id"])
        started = _optional_status_event(settings, job_id, 2)
        terminal = _optional_status_event(settings, job_id, 3)
        if started is not None and terminal is None:
            continue
        if terminal is not None:
            continue
        return accepted
    return None


def _optional_status_event(
    settings: P1CSettings,
    job_id: str,
    sequence: int,
) -> dict[str, object] | None:
    """Read one immutable event, distinguishing absence from corruption."""

    path = _status_path(settings, job_id, sequence)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        return load_status_event(settings, job_id, sequence)
    except TranslatorStatusError as error:
        raise TranslatorIntakeError(f"existing Translator status is invalid: {error}") from error


def _accepted_identity_matches(event: Mapping[str, object], document: Mapping[str, object]) -> bool:
    return (
        event.get("engine_kind") == "translator"
        and event.get("engine_job_id") == document.get("translator_job_id")
        and event.get("dispatch_id") == document.get("dispatch_id")
        and event.get("correlation_id") == document.get("dispatch_id")
        and event.get("attempt_number") == document.get("attempt_number")
        and event.get("event_kind") == "accepted"
        and event.get("state") == "accepted"
        and event.get("handoff_id") == document.get("handoff_id")
        and event.get("handoff_ref") == document.get("handoff_sidecar_ref")
    )


def _status_path(settings: P1CSettings, job_id: str, sequence: int) -> Path:
    root = _require_directory(settings.engine_status_dir, "DUBVI_ENGINE_STATUS_DIR")
    directory = root / "translator" / job_id
    return directory / f"event-{sequence:06d}.json"


def _require_directory(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise TranslatorIntakeError(f"{name} must be an existing non-symlink directory")
    return path.resolve()


def _resolve_ref(root: Path, reference: str, field: str) -> Path:
    if not reference or reference != reference.strip() or reference.startswith("/") or reference.endswith("/") or ":" in reference or "\\" in reference:
        raise TranslatorIntakeError(f"{field} must be a portable relative reference")
    if any(part in {"", ".", ".."} for part in reference.split("/")):
        raise TranslatorIntakeError(f"{field} contains an unsafe path component")
    path = root.joinpath(*reference.split("/"))
    _assert_under(root, path)
    current = root
    for part in reference.split("/"):
        current = current / part
        if current.is_symlink():
            raise TranslatorIntakeError(f"{field} crosses a symlink")
    return path


def _assert_under(root: Path, path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise TranslatorIntakeError("canonical Translator path escapes its root") from error


def _bounded_read(path: Path, field: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise TranslatorIntakeError(f"{field} must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_ENVELOPE_BYTES:
        raise TranslatorIntakeError(f"{field} has an unsafe size")
    payload = path.read_bytes()
    if len(payload) != size:
        raise TranslatorIntakeError(f"{field} changed while read")
    return payload


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranslatorIntakeError("occurred_at_utc must be an aware ISO timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise TranslatorIntakeError("occurred_at_utc must be an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TranslatorIntakeError("occurred_at_utc must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    milliseconds = utc.microsecond // 1000
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{milliseconds:03d}Z"
