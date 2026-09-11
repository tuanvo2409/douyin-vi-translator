"""Lease-aware Translator worker shell for the CP7 lifecycle boundary."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol

from dubvi_engine_contract import ContractValidationError, validate_control_plane_to_translator_job

from p1c_config import P1CSettings
from p1c_config import load_p1c_settings
from p1c_intake import (
    TranslatorIntakeError,
    accept_next_translator_job,
    accept_translator_envelope,
)
from p1c_lease_client import (
    LeaseBridgeUnavailable,
    LeaseBridgeResponse,
    TranslatorLeaseBridgeClient,
)
from p1c_status import TranslatorStatusError, load_status_event, publish_status_event


class HeavyWorkHandle(Protocol):
    def poll(self) -> object: ...
    def terminate(self) -> None: ...
    def wait(self, timeout: float | None = None) -> object: ...


class TranslatorWorkerError(RuntimeError):
    """The supplied canonical Translator job is invalid."""


@dataclass(frozen=True)
class WorkerResult:
    outcome: str
    lease_id: str | None = None


def run_lease_aware_work(
    job: Mapping[str, object],
    settings: P1CSettings,
    lease_client,
    start_work: Callable[[Mapping[str, object]], HeavyWorkHandle],
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    occurred_at_utc: str | None = None,
    utc_now: Callable[[], str] | None = None,
    completion_callback: Callable[[Mapping[str, object], HeavyWorkHandle], None] | None = None,
) -> WorkerResult:
    """Admit one accepted job, start only after a valid lease, then renew it.

    CP7 deliberately has no terminal-success publication.  Normal completion
    releases the exact lease and leaves terminal engine processing to CP8.
    """

    document = _validate_job(job)
    job_id = str(document["translator_job_id"])
    try:
        accepted = load_status_event(settings, job_id, 1)
    except TranslatorStatusError:
        return WorkerResult("accepted_status_unavailable")
    if not _accepted_matches(accepted, document):
        return WorkerResult("accepted_status_unavailable")
    try:
        admission: LeaseBridgeResponse = lease_client.acquire(document)
    except LeaseBridgeUnavailable:
        return WorkerResult("lease_unavailable")
    if not admission.granted or admission.lease is None:
        return WorkerResult("capacity_unavailable")
    lease_id = str(admission.lease["lease_id"])
    try:
        publish_status_event(settings, _event(document, 2, "started", "running", lease_id, occurred_at_utc))
    except Exception:
        return _cleanup_before_work(lease_client, document, lease_id, "started_status_unavailable")
    try:
        handle = start_work(document)
    except Exception:
        return _cleanup_before_work(lease_client, document, lease_id, "work_start_failed")

    now = utc_now or _utc_now
    confirmed_expiry = str(admission.lease["expires_at"])
    next_heartbeat = monotonic() + admission.heartbeat_seconds
    while handle.poll() is None:
        if _normalize_timestamp(now()) >= confirmed_expiry:
            return _lease_lost(handle, document, lease_id, settings, occurred_at_utc)
        current_monotonic = monotonic()
        if current_monotonic < next_heartbeat:
            sleep(max(0.0, next_heartbeat - current_monotonic))
        try:
            renewed = lease_client.heartbeat(document, lease_id)
            if renewed.lease is None:
                raise LeaseBridgeUnavailable("missing renewed lease")
        except LeaseBridgeUnavailable:
            return _lease_lost(handle, document, lease_id, settings, occurred_at_utc)
        confirmed_expiry = str(renewed.lease["expires_at"])
        next_heartbeat = monotonic() + renewed.heartbeat_seconds
    completion_outcome = "work_completed"
    if completion_callback is not None:
        try:
            completion_callback(document, handle)
        except Exception:
            completion_outcome = "completion_reconciliation_required"
    try:
        released = lease_client.release(document, lease_id)
    except LeaseBridgeUnavailable:
        return WorkerResult(completion_outcome + "_release_unconfirmed", lease_id)
    return WorkerResult(
        completion_outcome + ("_lease_released" if released.lease is not None else "_release_unconfirmed"),
        lease_id,
    )


def run_translator_worker(*args, **kwargs) -> WorkerResult:
    """Named alias for callers that prefer the engine-specific API."""

    return run_lease_aware_work(*args, **kwargs)


def _cleanup_before_work(lease_client, job: Mapping[str, object], lease_id: str, outcome: str) -> WorkerResult:
    try:
        lease_client.release(job, lease_id)
    except LeaseBridgeUnavailable:
        return WorkerResult(outcome + "_release_unconfirmed", lease_id)
    return WorkerResult(outcome, lease_id)


def _lease_lost(
    handle: HeavyWorkHandle,
    job: Mapping[str, object],
    lease_id: str,
    settings: P1CSettings,
    occurred_at_utc: str | None,
) -> WorkerResult:
    handle.terminate()
    try:
        publish_status_event(settings, _event(job, 3, "lease_lost", "running", lease_id, occurred_at_utc))
    except TranslatorStatusError:
        return WorkerResult("lease_lost_status_unavailable", lease_id)
    return WorkerResult("lease_lost", lease_id)


def _validate_job(job: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(job, Mapping):
        raise TranslatorWorkerError("Translator job must be a mapping")
    source = job.get("document", job)
    try:
        return validate_control_plane_to_translator_job(source)
    except ContractValidationError as error:
        raise TranslatorWorkerError(f"invalid Translator job: {error}") from error


def _accepted_matches(event: Mapping[str, object], job: Mapping[str, object]) -> bool:
    return (
        event.get("engine_kind") == "translator"
        and event.get("engine_job_id") == job.get("translator_job_id")
        and event.get("dispatch_id") == job.get("dispatch_id")
        and event.get("correlation_id") == job.get("dispatch_id")
        and event.get("attempt_number") == job.get("attempt_number", 1)
        and event.get("sequence") == 1
        and event.get("event_kind") == "accepted"
        and event.get("state") == "accepted"
        and event.get("handoff_id") == job.get("handoff_id")
        and event.get("handoff_ref") == job.get("handoff_sidecar_ref")
    )


def _event(
    job: Mapping[str, object],
    sequence: int,
    event_kind: str,
    state: str,
    lease_id: str,
    occurred_at_utc: str | None,
) -> dict[str, object]:
    return {
        "contract_version": 1,
        "message_kind": "engine_status_event",
        "event_id": str(uuid.uuid4()),
        "engine_kind": "translator",
        "engine_job_id": job["translator_job_id"],
        "dispatch_id": job["dispatch_id"],
        "correlation_id": job["dispatch_id"],
        "sequence": sequence,
        "attempt_number": job.get("attempt_number", 1),
        "event_kind": event_kind,
        "state": state,
        "occurred_at_utc": _normalize_timestamp(occurred_at_utc or _utc_now()),
        "lease_id": lease_id,
        "handoff_id": job["handoff_id"],
        "handoff_ref": job["handoff_sidecar_ref"],
    }


def _normalize_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranslatorWorkerError("worker timestamp must be an aware ISO timestamp")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise TranslatorWorkerError("worker timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TranslatorWorkerError("worker timestamp must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{utc.microsecond // 1000:03d}Z"


def _utc_now() -> str:
    return _normalize_timestamp(datetime.now(timezone.utc).isoformat())


def run_canonical_once(
    settings: P1CSettings,
    *,
    translator_job_id: str | None = None,
) -> dict[str, object]:
    """Run one canonical Translator admission without duplicating CP8 logic."""

    _require_canonical_roots(settings)
    if translator_job_id is not None:
        job_id = _canonical_cli_uuid(translator_job_id)
        envelope_path = settings.translator_job_dir / f"translator-job-{job_id}.job.json"
        if not envelope_path.is_file() or envelope_path.is_symlink():
            raise TranslatorIntakeError("requested Translator job envelope was not found")
        accepted = accept_translator_envelope(envelope_path, settings)
    else:
        accepted = accept_next_translator_job(settings)
    if accepted is None:
        return {"outcome": "no_work"}

    job_id = str(accepted.document["translator_job_id"])
    started = _optional_worker_status(settings, job_id, 2)
    terminal = _optional_worker_status(settings, job_id, 3)
    if terminal is not None:
        return _translator_report(
            settings,
            job_id,
            outcome=_terminal_outcome(terminal),
            worker_outcome="terminal_existing",
            lease_id=terminal.get("lease_id"),
            terminal=terminal,
        )
    if started is not None:
        return _translator_report(
            settings,
            job_id,
            outcome="reconciliation_required",
            worker_outcome="started_without_terminal",
            lease_id=started.get("lease_id"),
        )

    from p1c_processing import run_canonical_processing

    bridge = TranslatorLeaseBridgeClient.from_settings(settings)
    worker_result = run_canonical_processing(
        settings,
        accepted.envelope_path,
        bridge,
    )
    terminal = _optional_worker_status(settings, job_id, 3)
    outcome = _terminal_outcome(terminal) if terminal is not None else _worker_outcome(worker_result.outcome)
    return _translator_report(
        settings,
        job_id,
        outcome=outcome,
        worker_outcome=worker_result.outcome,
        lease_id=worker_result.lease_id,
        terminal=terminal,
    )


def _require_canonical_roots(settings: P1CSettings) -> None:
    roots = (
        ("DUBVI_TRANSLATOR_JOB_DIR", settings.translator_job_dir),
        ("DUBVI_MEDIA_DIR", settings.media_dir),
        ("DUBVI_ENGINE_STATUS_DIR", settings.engine_status_dir),
        ("DUBVI_OUTPUT_DIR", settings.output_dir),
        ("DUBVI_CONTROL_PLANE_ROOT", settings.control_plane_root),
    )
    for name, path in roots:
        if path is None or path.is_symlink() or not path.is_dir():
            raise TranslatorWorkerError(f"{name} must be an existing non-symlink directory")


def _optional_worker_status(
    settings: P1CSettings,
    job_id: str,
    sequence: int,
) -> dict[str, object] | None:
    path = settings.engine_status_dir / "translator" / job_id / f"event-{sequence:06d}.json"
    if not path.exists() and not path.is_symlink():
        return None
    try:
        return load_status_event(settings, job_id, sequence)
    except TranslatorStatusError as error:
        raise TranslatorWorkerError(f"existing Translator status is invalid: {error}") from error


def _canonical_cli_uuid(value: str) -> str:
    if not isinstance(value, str):
        raise TranslatorWorkerError("translator job id is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as error:
        raise TranslatorWorkerError("translator job id is invalid") from error
    if value != str(parsed):
        raise TranslatorWorkerError("translator job id is invalid")
    return value


def _terminal_outcome(event: Mapping[str, object] | None) -> str:
    if event is None:
        return "reconciliation_required"
    event_kind = event.get("event_kind")
    if event_kind in {"succeeded", "failed", "rejected", "review_required"}:
        return str(event_kind)
    return "reconciliation_required"


def _worker_outcome(value: str) -> str:
    if value.startswith("completion_reconciliation") or value in {"lease_lost", "lease_lost_status_unavailable"}:
        return "reconciliation_required"
    return value


def _translator_report(
    settings: P1CSettings,
    job_id: str,
    *,
    outcome: str,
    worker_outcome: str,
    lease_id: object | None = None,
    terminal: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "translator_job_id": job_id,
        "outcome": outcome,
        "worker_outcome": worker_outcome,
    }
    if isinstance(lease_id, str):
        report["lease_id"] = lease_id
    if terminal is not None:
        report["terminal_state"] = terminal.get("state")
        report["terminal_event"] = terminal.get("event_kind")
        if terminal.get("event_kind") == "succeeded":
            report["canonical_output_path"] = str(
                settings.output_dir / "p1c" / job_id / "output.mp4"
            )
    return report


def _exit_code(report: Mapping[str, object]) -> int:
    if report.get("outcome") in {"no_work", "succeeded"}:
        return 0
    if report.get("outcome") in {"failed", "rejected", "review_required"}:
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canonical Translator P1C runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    once = subparsers.add_parser("once", help="admit and run at most one canonical job")
    once.add_argument("--translator-job-id")
    args = parser.parse_args(argv)
    try:
        settings = load_p1c_settings(require_control_plane=True)
        report = run_canonical_once(settings, translator_job_id=args.translator_job_id)
    except Exception as error:
        report = {"outcome": "error", "diagnostic_summary": str(error)[:2048]}
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
