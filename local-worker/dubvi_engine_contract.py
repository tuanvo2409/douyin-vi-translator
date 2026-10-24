"""Pure, versioned JSON contracts shared by the three DUBVI repositories."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
import json
import math
import re
import uuid
from typing import Any


CONTRACT_IMPLEMENTATION_VERSION = "p1c-cp1-v1"
CONTRACT_VERSION = 1
HANDOFF_SCHEMA_VERSION = 2


class ContractValidationError(ValueError):
    """Raised when a document does not satisfy a known contract schema."""


_FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
_ERROR_CLASSIFICATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")

_DECISIONS = {"pass", "reject", "review", "error", "unavailable"}
_POLICY_MODES = {"off", "review", "enforce"}


def _error(message: str) -> None:
    raise ContractValidationError(message)


def _utf8_size(value: str, field: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ContractValidationError(f"{field} must be valid UTF-8 text") from exc


def _text(value: object, field: str, *, maximum: int = 256, nonblank: bool = True) -> str:
    if not isinstance(value, str):
        _error(f"{field} must be a string")
    if "\x00" in value:
        _error(f"{field} must not contain NUL")
    if nonblank and not value:
        _error(f"{field} must be non-empty")
    if value != value.strip():
        _error(f"{field} must not have leading or trailing whitespace")
    if _utf8_size(value, field) > maximum:
        _error(f"{field} exceeds {maximum} UTF-8 bytes")
    return value


def _profile(value: object, field: str) -> str:
    return _text(value, field, maximum=128)


def _optional_profile(value: object, field: str) -> None | str:
    if value is None:
        return None
    return _profile(value, field)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        _error(f"{field} must be a positive integer")
    return value


def _number(value: object, field: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(f"{field} must be a number")
    if not math.isfinite(float(value)):
        _error(f"{field} must be finite")
    if positive and value <= 0:
        _error(f"{field} must be greater than zero")
    return value


def _uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        _error(f"{field} must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ContractValidationError(f"{field} must be a valid UUID") from exc
    if value != str(parsed):
        _error(f"{field} must use lowercase canonical UUID text")
    return value


def _fingerprint(value: object, field: str) -> str:
    if not isinstance(value, str) or not _FINGERPRINT_RE.fullmatch(value):
        _error(f"{field} must be a lowercase sha256:<hex> fingerprint")
    return value


def _timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not _TIMESTAMP_RE.fullmatch(value):
        _error(f"{field} must be canonical UTC milliseconds text")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ContractValidationError(f"{field} is not a valid UTC timestamp") from exc
    return value


def _basename(value: object, field: str, *, canonical_media: bool = False) -> str:
    text = _text(value, field, maximum=255)
    if text in {".", ".."} or any(token in text for token in ("/", "\\", ":")):
        _error(f"{field} must be a safe basename")
    if canonical_media and not text.startswith("dubvi-"):
        _error(f"{field} must use the reserved dubvi- namespace")
    return text


def _relative_ref(value: object, field: str) -> str:
    text = _text(value, field, maximum=1024)
    if text.startswith("/") or text.endswith("/") or ":" in text or "\\" in text:
        _error(f"{field} must be a portable relative reference")
    components = text.split("/")
    if any(component in {"", ".", ".."} for component in components):
        _error(f"{field} contains an unsafe path component")
    return text


def _mapping(value: object, field: str = "document") -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _error(f"{field} must be a JSON object")
    return value


def _keys(document: Mapping[str, object], required: set[str], optional: set[str]) -> None:
    actual = set(document)
    missing = required - actual
    if missing:
        _error(f"missing required field(s): {', '.join(sorted(missing))}")
    unknown = actual - required - optional
    if unknown:
        _error(f"unknown field(s): {', '.join(sorted(unknown))}")


def _exact(document: Mapping[str, object], field: str, expected: object) -> None:
    if document.get(field) != expected:
        _error(f"{field} must equal {expected!r}")


def _detach(document: Mapping[str, object]) -> dict[str, object]:
    return deepcopy(dict(document))


def _validate_common_ids(document: Mapping[str, object], fields: tuple[str, ...]) -> None:
    for field in fields:
        _uuid(document[field], field)
    if "dispatch_id" in document and "correlation_id" in document:
        if document["correlation_id"] != document["dispatch_id"]:
            _error("correlation_id must equal dispatch_id")


def _validate_profile_fields(document: Mapping[str, object], fields: tuple[str, ...]) -> None:
    for field in fields:
        _profile(document[field], field)


def _validate_optional_metadata(document: Mapping[str, object], fields: set[str]) -> None:
    if "parent_reup_job_id" in fields and "parent_reup_job_id" in document:
        _uuid(document["parent_reup_job_id"], "parent_reup_job_id")
    if "parent_translator_job_id" in fields and "parent_translator_job_id" in document:
        _uuid(document["parent_translator_job_id"], "parent_translator_job_id")
    if "attempt_number" in fields and "attempt_number" in document:
        _positive_int(document["attempt_number"], "attempt_number")
    if "localization_profile" in fields and "localization_profile" in document:
        _optional_profile(document["localization_profile"], "localization_profile")
    if "policy_profile" in fields and "policy_profile" in document:
        _optional_profile(document["policy_profile"], "policy_profile")
    if "producer_version" in fields and "producer_version" in document:
        _text(document["producer_version"], "producer_version")
    if "display_label" in fields and "display_label" in document:
        _text(document["display_label"], "display_label")


def validate_control_plane_to_reup_job(document: Mapping[str, object]) -> dict[str, object]:
    """Validate and detach a complete Control Plane to Reup job envelope."""
    document = _mapping(document)
    required = {
        "contract_version", "message_kind", "envelope_status", "dispatch_id",
        "reup_job_id", "correlation_id", "candidate_id", "schedule_id",
        "channel_id", "channel_slug", "reup_profile", "target_platform",
        "media_name", "media_size", "source_fingerprint", "source_provenance_ref",
        "created_at_utc",
    }
    optional = {
        "parent_reup_job_id", "attempt_number", "localization_profile",
        "policy_profile", "original_filename", "producer_version", "display_label",
    }
    _keys(document, required, optional)
    _exact(document, "contract_version", CONTRACT_VERSION)
    _exact(document, "message_kind", "control_plane_to_reup_job")
    _exact(document, "envelope_status", "complete")
    _validate_common_ids(document, ("dispatch_id", "reup_job_id", "correlation_id", "candidate_id", "schedule_id", "channel_id"))
    _validate_profile_fields(document, ("channel_slug", "reup_profile", "target_platform"))
    _basename(document["media_name"], "media_name", canonical_media=True)
    _positive_int(document["media_size"], "media_size")
    _fingerprint(document["source_fingerprint"], "source_fingerprint")
    expected_provenance = f"candidate:{document['candidate_id']}"
    if document["source_provenance_ref"] != expected_provenance:
        _error("source_provenance_ref must identify candidate_id exactly")
    _timestamp(document["created_at_utc"], "created_at_utc")
    _validate_optional_metadata(document, optional)
    if "original_filename" in document:
        _text(document["original_filename"], "original_filename")
    return _detach(document)


def validate_reup_to_translator_handoff(document: Mapping[str, object]) -> dict[str, object]:
    """Validate the immutable Reup to Translator handoff v2."""
    document = _mapping(document)
    required = {
        "handoff_schema_version", "handoff_status", "handoff_id", "candidate_id",
        "schedule_id", "dispatch_id", "reup_job_id", "correlation_id", "channel_id",
        "channel_slug", "reup_profile", "target_platform", "localization_profile",
        "policy_profile", "source_fingerprint", "reup_output_fingerprint", "created_at_utc",
    }
    optional = {
        "original_filename", "duration_seconds", "encoder", "vpdq_score",
        "processed_at_utc", "producer_version",
    }
    _keys(document, required, optional)
    _exact(document, "handoff_schema_version", HANDOFF_SCHEMA_VERSION)
    _exact(document, "handoff_status", "complete")
    _validate_common_ids(document, ("handoff_id", "candidate_id", "schedule_id", "dispatch_id", "reup_job_id", "correlation_id", "channel_id"))
    _validate_profile_fields(document, ("channel_slug", "reup_profile", "target_platform"))
    _optional_profile(document["localization_profile"], "localization_profile")
    _optional_profile(document["policy_profile"], "policy_profile")
    _fingerprint(document["source_fingerprint"], "source_fingerprint")
    _fingerprint(document["reup_output_fingerprint"], "reup_output_fingerprint")
    _timestamp(document["created_at_utc"], "created_at_utc")
    if "original_filename" in document:
        _text(document["original_filename"], "original_filename")
    if "duration_seconds" in document:
        _number(document["duration_seconds"], "duration_seconds", positive=True)
    if "encoder" in document:
        _profile(document["encoder"], "encoder")
    if "vpdq_score" in document:
        _number(document["vpdq_score"], "vpdq_score")
    if "processed_at_utc" in document:
        _timestamp(document["processed_at_utc"], "processed_at_utc")
    if "producer_version" in document:
        _text(document["producer_version"], "producer_version")
    return _detach(document)


def validate_control_plane_to_translator_job(document: Mapping[str, object]) -> dict[str, object]:
    """Validate and detach a complete Control Plane to Translator envelope."""
    document = _mapping(document)
    required = {
        "contract_version", "message_kind", "envelope_status", "translator_job_id",
        "dispatch_id", "reup_job_id", "handoff_id", "correlation_id", "candidate_id",
        "schedule_id", "channel_id", "channel_slug", "target_platform",
        "localization_profile", "policy_profile", "handoff_media_ref",
        "handoff_sidecar_ref", "reup_output_fingerprint", "created_at_utc",
    }
    optional = {"parent_translator_job_id", "attempt_number", "producer_version", "display_label"}
    _keys(document, required, optional)
    _exact(document, "contract_version", CONTRACT_VERSION)
    _exact(document, "message_kind", "control_plane_to_translator_job")
    _exact(document, "envelope_status", "complete")
    _validate_common_ids(document, ("translator_job_id", "dispatch_id", "reup_job_id", "handoff_id", "correlation_id", "candidate_id", "schedule_id", "channel_id"))
    _validate_profile_fields(document, ("channel_slug", "target_platform"))
    _optional_profile(document["localization_profile"], "localization_profile")
    _optional_profile(document["policy_profile"], "policy_profile")
    _relative_ref(document["handoff_media_ref"], "handoff_media_ref")
    _relative_ref(document["handoff_sidecar_ref"], "handoff_sidecar_ref")
    _fingerprint(document["reup_output_fingerprint"], "reup_output_fingerprint")
    _timestamp(document["created_at_utc"], "created_at_utc")
    _validate_optional_metadata(document, optional)
    return _detach(document)


def _validate_reason_codes(value: object, field: str = "reason_codes") -> list[object]:
    if not isinstance(value, list) or len(value) > 32:
        _error(f"{field} must be a list of at most 32 codes")
    normalized: list[str] = []
    for index, code in enumerate(value):
        normalized.append(_text(code, f"{field}[{index}]", maximum=128))
    if len(set(normalized)) != len(normalized):
        _error(f"{field} must not contain duplicates")
    return value


def _validate_deep_policy(value: object) -> dict[str, object]:
    policy = _mapping(value, "deep_policy")
    required = {"policy_schema_version", "mode", "overall_decision", "reason_codes", "needs_review", "evaluated_at"}
    optional = {"report_fingerprint", "report_ref"}
    _keys(policy, required, optional)
    _exact(policy, "policy_schema_version", 1)
    if not isinstance(policy["mode"], str) or policy["mode"] not in _POLICY_MODES:
        _error("deep_policy.mode is unsupported")
    if not isinstance(policy["overall_decision"], str) or policy["overall_decision"] not in _DECISIONS:
        _error("deep_policy.overall_decision is unsupported")
    _validate_reason_codes(policy["reason_codes"], "deep_policy.reason_codes")
    if not isinstance(policy["needs_review"], bool):
        _error("deep_policy.needs_review must be boolean")
    _timestamp(policy["evaluated_at"], "deep_policy.evaluated_at")
    if "report_fingerprint" in policy:
        _fingerprint(policy["report_fingerprint"], "deep_policy.report_fingerprint")
    if "report_ref" in policy:
        _relative_ref(policy["report_ref"], "deep_policy.report_ref")
    return _detach(policy)


_EVENT_PAIRS = {
    "reup": {
        "accepted": "accepted",
        "started": "running",
        "output_published": "running",
        "handoff_published": "running",
        "succeeded": "succeeded",
        "failed": "failed",
        "lease_lost": "running",
    },
    "translator": {
        "accepted": "accepted",
        "started": "running",
        "succeeded": "succeeded",
        "failed": "failed",
        "rejected": "rejected",
        "review_required": "review_required",
        "lease_lost": "running",
    },
}


def validate_engine_status_event(document: Mapping[str, object]) -> dict[str, object]:
    """Validate one immutable Reup or Translator status event."""
    document = _mapping(document)
    required = {
        "contract_version", "message_kind", "event_id", "engine_kind", "engine_job_id",
        "dispatch_id", "correlation_id", "sequence", "attempt_number", "event_kind",
        "state", "occurred_at_utc",
    }
    optional = {
        "parent_engine_job_id", "lease_id", "output_fingerprint", "handoff_id",
        "handoff_ref", "error_classification", "diagnostic_summary", "deep_policy",
        "producer_version",
    }
    _keys(document, required, optional)
    _exact(document, "contract_version", CONTRACT_VERSION)
    _exact(document, "message_kind", "engine_status_event")
    if not isinstance(document["engine_kind"], str) or document["engine_kind"] not in _EVENT_PAIRS:
        _error("engine_kind is unsupported")
    _validate_common_ids(document, ("event_id", "engine_job_id", "dispatch_id", "correlation_id"))
    _positive_int(document["sequence"], "sequence")
    _positive_int(document["attempt_number"], "attempt_number")
    event_kind = document["event_kind"]
    state = document["state"]
    if not isinstance(event_kind, str) or _EVENT_PAIRS[document["engine_kind"]].get(event_kind) != state:
        _error("event_kind/state pair is not legal for engine_kind")
    _timestamp(document["occurred_at_utc"], "occurred_at_utc")
    if "parent_engine_job_id" in document:
        _uuid(document["parent_engine_job_id"], "parent_engine_job_id")
    if "lease_id" in document:
        _uuid(document["lease_id"], "lease_id")
    if "output_fingerprint" in document:
        _fingerprint(document["output_fingerprint"], "output_fingerprint")
    if "handoff_id" in document:
        _uuid(document["handoff_id"], "handoff_id")
    if "handoff_ref" in document:
        _relative_ref(document["handoff_ref"], "handoff_ref")
    if "error_classification" in document:
        if not isinstance(document["error_classification"], str) or not _ERROR_CLASSIFICATION_RE.fullmatch(document["error_classification"]):
            _error("error_classification must be uppercase classification text")
    if "diagnostic_summary" in document:
        _text(document["diagnostic_summary"], "diagnostic_summary", maximum=4096)
    if "deep_policy" in document:
        _validate_deep_policy(document["deep_policy"])
    if "producer_version" in document:
        _text(document["producer_version"], "producer_version")

    if event_kind == "output_published" and "output_fingerprint" not in document:
        _error("Reup output_published requires output_fingerprint")
    if event_kind == "handoff_published":
        for field in ("handoff_id", "handoff_ref", "output_fingerprint"):
            if field not in document:
                _error(f"Reup handoff_published requires {field}")
    if document["engine_kind"] == "reup" and event_kind == "succeeded":
        for field in ("output_fingerprint", "handoff_id", "handoff_ref"):
            if field not in document:
                _error(f"Reup succeeded requires {field}")
    if document["engine_kind"] == "translator" and event_kind == "succeeded" and "output_fingerprint" not in document:
        _error("Translator succeeded requires output_fingerprint")
    if event_kind == "failed" and "error_classification" not in document:
        _error("failed requires error_classification")
    if document["engine_kind"] == "translator" and event_kind == "rejected":
        if document.get("error_classification") != "DEEP_POLICY_REJECT" or "deep_policy" not in document:
            _error("Translator rejected requires DEEP_POLICY_REJECT and deep_policy")
    if document["engine_kind"] == "translator" and event_kind == "review_required":
        if document.get("error_classification") != "DEEP_POLICY_REVIEW" or "deep_policy" not in document:
            _error("Translator review_required requires DEEP_POLICY_REVIEW and deep_policy")
    return _detach(document)


def _reject_constant(value: str) -> None:
    raise ContractValidationError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _error(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def parse_json_document(data: bytes | str) -> dict[str, object]:
    """Parse strict JSON bytes/text into a detached top-level object."""
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractValidationError("JSON input must be valid UTF-8") from exc
    elif isinstance(data, str):
        text = data
    else:
        _error("JSON input must be bytes or string text")
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ContractValidationError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ContractValidationError("invalid JSON document") from exc
    if not isinstance(parsed, dict):
        _error("top-level JSON value must be an object")
    return deepcopy(parsed)


def canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    """Serialize a mapping as compact, deterministic UTF-8 JSON plus one LF."""
    if not isinstance(document, Mapping):
        _error("canonical JSON input must be an object")
    try:
        payload = json.dumps(
            dict(document),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ContractValidationError("document cannot be canonically serialized") from exc
    return payload + b"\n"


def validate_document(document: Mapping[str, object]) -> dict[str, object]:
    """Dispatch a document to its versioned family validator."""
    document = _mapping(document)
    if "message_kind" in document:
        kind = document.get("message_kind")
        if kind == "control_plane_to_reup_job":
            return validate_control_plane_to_reup_job(document)
        if kind == "control_plane_to_translator_job":
            return validate_control_plane_to_translator_job(document)
        if kind == "engine_status_event":
            return validate_engine_status_event(document)
        _error("message_kind is unsupported")
    if "handoff_schema_version" in document or "handoff_status" in document:
        return validate_reup_to_translator_handoff(document)
    _error("document has no recognized contract discriminator")
