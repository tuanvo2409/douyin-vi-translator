"""Canonical CP8 Translator processing boundary.

The parent owns immutable status publication and lease cleanup.  The child
receives only bounded path arguments, reloads canonical evidence, runs the
legacy local processing graph behind an in-memory adapter, and returns a
bounded receipt.  No child code writes engine status events.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from dubvi_engine_contract import (
    ContractValidationError,
    canonical_json_bytes,
    parse_json_document,
    validate_engine_status_event,
)

from p1c_config import P1CSettings
from p1c_intake import load_translator_envelope
from p1c_status import load_status_event, publish_status_event
from p1c_worker import HeavyWorkHandle, WorkerResult, run_lease_aware_work


MAX_RECEIPT_BYTES = 64 * 1024
MAX_DIAGNOSTIC_BYTES = 2048
_CHUNK_SIZE = 1024 * 1024
_OUTCOMES = {"succeeded", "failed", "rejected", "review_required", "incomplete"}


class CanonicalProcessingError(RuntimeError):
    """Canonical processing failed closed."""


class CanonicalOutputConflict(CanonicalProcessingError):
    """Canonical output already exists with unrelated bytes."""


class CanonicalPolicyReject(CanonicalProcessingError):
    """Typed canonical policy rejection; legacy mode never raises this type."""

    def __init__(self, deep_policy: Mapping[str, object]) -> None:
        super().__init__("canonical policy rejected the Translator job")
        self.deep_policy = dict(deep_policy)


class CanonicalPolicyReviewRequired(CanonicalProcessingError):
    """Typed canonical policy review result; legacy mode never raises this type."""

    def __init__(self, deep_policy: Mapping[str, object]) -> None:
        super().__init__("canonical policy requires Translator review")
        self.deep_policy = dict(deep_policy)


@dataclass(frozen=True)
class CanonicalOutput:
    path: Path
    fingerprint: str
    reused_existing_final: bool


@dataclass(frozen=True)
class CanonicalChildHandle:
    process: subprocess.Popen[bytes]

    def poll(self) -> object:
        return self.process.poll()

    def terminate(self) -> None:
        self.process.terminate()

    def wait(self, timeout: float | None = None) -> object:
        return self.process.wait(timeout=timeout)

    def read_receipt(self) -> dict[str, object]:
        try:
            stdout, _stderr = self.process.communicate(timeout=2)
        except subprocess.TimeoutExpired as error:
            raise CanonicalProcessingError("Translator child did not close its bounded receipt") from error
        if len(stdout) > MAX_RECEIPT_BYTES:
            raise CanonicalProcessingError("Translator child receipt exceeds 64KiB")
        try:
            parsed = parse_json_document(stdout)
            receipt = _validate_receipt(parsed)
            if canonical_json_bytes(receipt) != stdout:
                raise CanonicalProcessingError("Translator child receipt is not canonical bytes")
            return receipt
        except ContractValidationError as error:
            raise CanonicalProcessingError(f"Translator child receipt is invalid: {error}") from error


class CanonicalRpcAdapter:
    """In-memory replacement for legacy RPC calls in canonical processing."""

    def __init__(self) -> None:
        self.reports: list[dict[str, object]] = []
        self.segments: list[dict[str, object]] | None = None
        self.completed_output_path: Path | None = None
        self.last_status: str | None = None

    def report(
        self,
        job_id: str,
        stage: str,
        progress: int,
        status: str | None = None,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        record: dict[str, object] = {
            "job_id": job_id,
            "stage": stage,
            "progress": progress,
        }
        if status is not None:
            record["status"] = status
            self.last_status = status
        if output_path is not None:
            record["output_path"] = output_path
            if stage == "complete" or status == "complete":
                self.completed_output_path = Path(output_path)
        if error is not None:
            record["error"] = error[:MAX_DIAGNOSTIC_BYTES]
        self.reports.append(record)

    def replace_segments(self, job_id: str, segments: list[dict[str, object]]) -> None:
        self.segments = [dict(segment) for segment in segments]

    def correct_ocr(self, payload: dict[str, object]) -> dict[str, object]:
        raise CanonicalProcessingError("canonical processing forbids remote OCR correction")


def build_canonical_runtime_config(
    document: Mapping[str, object],
    *,
    capcut_voice: str | None = None,
) -> dict[str, object]:
    """Build the bounded CP8 runtime profile without importing engine clients."""

    voice = capcut_voice or os.environ.get("DUBVI_CAPCUT_VOICE", "BV421_vivn_streaming")
    if not isinstance(voice, str) or not voice.strip():
        raise CanonicalProcessingError("canonical CapCut voice must be nonblank")
    return {
        "voice": {"name": voice.strip(), "maxTempo": 1.35},
        "ocr": {
            "enabled": True,
            "sampleFrames": 3,
            "minConfidence": 65,
            "llmCorrection": False,
        },
        "roi": {
            "xPercent": 2,
            "yPercent": 66,
            "widthPercent": 96,
            "heightPercent": 9.8,
            "blurPx": 24,
        },
        "audioMode": "duck",
        "channelProfile": document["channel_slug"],
    }


def normalize_deep_policy(policy_result: object, mode: str) -> dict[str, object] | None:
    """Reduce the legacy result to the CP1 bounded deep-policy shape."""

    if policy_result is None:
        return None
    if mode not in {"off", "review", "enforce"}:
        raise CanonicalProcessingError("canonical policy mode is invalid")
    decision = getattr(getattr(policy_result, "overall_decision", None), "value", None)
    reason_codes = getattr(policy_result, "reason_codes", None)
    needs_review = getattr(policy_result, "needs_review", None)
    evaluated_at = getattr(policy_result, "evaluated_at", None)
    if (
        decision not in {"pass", "reject", "review", "error", "unavailable"}
        or not isinstance(reason_codes, (tuple, list))
        or not isinstance(needs_review, bool)
        or not isinstance(evaluated_at, str)
    ):
        raise CanonicalProcessingError("legacy policy result has an unsafe shape")
    bounded_reasons: list[str] = []
    for code in reason_codes[:32]:
        if not isinstance(code, str) or not code or len(code.encode("utf-8")) > 128:
            raise CanonicalProcessingError("legacy policy reason code is unbounded")
        bounded_reasons.append(code)
    return {
        "policy_schema_version": 1,
        "mode": mode,
        "overall_decision": decision,
        "reason_codes": bounded_reasons,
        "needs_review": needs_review,
        "evaluated_at": _normalize_timestamp(evaluated_at),
    }


def publish_canonical_output(
    settings: P1CSettings,
    translator_job_id: str,
    candidate_path: Path,
) -> CanonicalOutput:
    """Verify a work-root candidate and publish a non-replacing canonical final."""

    output_root = _require_directory(settings.output_dir, "DUBVI_OUTPUT_DIR")
    work_root = output_root / ".p1c-work" / translator_job_id
    _assert_contained(work_root, candidate_path)
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise CanonicalProcessingError("canonical output candidate must be a regular file")
    if candidate_path.stat().st_size <= 0:
        raise CanonicalProcessingError("canonical output candidate must be nonempty")
    fingerprint = _hash(candidate_path)
    final_dir = output_root / "p1c" / translator_job_id
    _make_directory_chain(output_root / "p1c", final_dir)
    final = final_dir / "output.mp4"
    if _real_file(final):
        if _matches(final, fingerprint):
            return CanonicalOutput(final, fingerprint, True)
        raise CanonicalOutputConflict("canonical output final differs; refusing overwrite")
    if final.is_symlink():
        raise CanonicalOutputConflict("canonical output final is an unsafe symlink")
    stage = final_dir / f".output.{fingerprint[7:19]}.part"
    _stage_copy(stage, candidate_path, fingerprint)
    try:
        os.link(stage, final)
    except FileExistsError:
        if not _matches(final, fingerprint):
            raise CanonicalOutputConflict("racing canonical output final differs")
        stage.unlink(missing_ok=True)
        return CanonicalOutput(final, fingerprint, True)
    except OSError as error:
        raise CanonicalProcessingError(f"canonical output publication failed: {error}") from error
    if not _matches(final, fingerprint):
        raise CanonicalProcessingError("canonical output final failed verification")
    stage.unlink(missing_ok=True)
    return CanonicalOutput(final, fingerprint, False)


def spawn_processing_child(settings: P1CSettings, document: Mapping[str, object]) -> CanonicalChildHandle:
    """Start a child with roots and identity only; no mutable job blob in argv."""

    _require_directory(settings.translator_job_dir, "DUBVI_TRANSLATOR_JOB_DIR")
    _require_directory(settings.media_dir, "DUBVI_MEDIA_DIR")
    _require_directory(settings.engine_status_dir, "DUBVI_ENGINE_STATUS_DIR")
    _require_directory(settings.output_dir, "DUBVI_OUTPUT_DIR")
    job_id = _canonical_uuid(document.get("translator_job_id"), "translator_job_id")
    script = Path(__file__).resolve()
    argv = [
        sys.executable,
        str(script),
        "--child",
        "--translator-job-id",
        job_id,
        "--translator-job-root",
        _bounded_path(settings.translator_job_dir),
        "--media-root",
        _bounded_path(settings.media_dir),
        "--status-root",
        _bounded_path(settings.engine_status_dir),
        "--output-root",
        _bounded_path(settings.output_dir),
    ]
    environment = os.environ.copy()
    worker_root = str(script.parent)
    environment["PYTHONPATH"] = worker_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    try:
        process = subprocess.Popen(
            argv,
            cwd=worker_root,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise CanonicalProcessingError("Translator processing child could not start") from error
    return CanonicalChildHandle(process)


def run_canonical_processing(
    settings: P1CSettings,
    envelope_path: Path,
    lease_client,
    *,
    monotonic: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    occurred_at_utc: str | None = None,
    utc_now: Callable[[], str] | None = None,
) -> WorkerResult:
    """Run the parent-owned CP8 lifecycle for one exact Translator envelope."""

    accepted = load_translator_envelope(envelope_path, settings)
    accepted_event = load_status_event(settings, str(accepted.document["translator_job_id"]), 1)
    if not _accepted_matches(accepted_event, accepted.document):
        raise CanonicalProcessingError("Translator accepted evidence does not match its envelope")

    def start_work(document: Mapping[str, object]) -> HeavyWorkHandle:
        return spawn_processing_child(settings, document)

    def complete(document: Mapping[str, object], handle: HeavyWorkHandle) -> None:
        if not hasattr(handle, "read_receipt"):
            raise CanonicalProcessingError("canonical completion requires a receipt-capable child handle")
        receipt = _validate_receipt(handle.read_receipt())
        started = load_status_event(settings, str(document["translator_job_id"]), 2)
        if not _started_matches(started, document):
            raise CanonicalProcessingError("started evidence changed before terminal publication")
        event_document = dict(document)
        event_document["lease_id"] = started["lease_id"]
        event = _receipt_to_terminal_event(event_document, receipt)
        if event is None:
            raise CanonicalProcessingError("Translator child returned no terminal receipt")
        publish_status_event(settings, event)

    kwargs: dict[str, object] = {
        "completion_callback": complete,
        "occurred_at_utc": occurred_at_utc,
        "utc_now": utc_now,
    }
    if monotonic is not None:
        kwargs["monotonic"] = monotonic
    if sleep is not None:
        kwargs["sleep"] = sleep
    return run_lease_aware_work(
        accepted.document,
        settings,
        lease_client,
        start_work,
        **kwargs,
    )


def run_child_job(settings: P1CSettings, translator_job_id: str) -> dict[str, object]:
    """Reload all canonical evidence and execute only the legacy local graph."""

    job_id = _canonical_uuid(translator_job_id, "translator_job_id")
    envelope = settings.translator_job_dir / f"translator-job-{job_id}.job.json"
    accepted = load_translator_envelope(envelope, settings)
    if accepted.document["translator_job_id"] != job_id:
        raise CanonicalProcessingError("Translator envelope identity differs from child argument")
    accepted_event = load_status_event(settings, job_id, 1)
    started_event = load_status_event(settings, job_id, 2)
    if not _accepted_matches(accepted_event, accepted.document) or not _started_matches(started_event, accepted.document):
        raise CanonicalProcessingError("Translator accepted/started evidence is inconsistent")

    work_base = settings.output_dir / ".p1c-work"
    _make_directory_chain(settings.output_dir, work_base)
    work_root = work_base / job_id
    if work_root.exists() and work_root.is_symlink():
        raise CanonicalProcessingError("Translator attempt work root is an unsafe symlink")
    work_root.mkdir(exist_ok=True)
    if work_root.is_symlink() or not work_root.is_dir():
        raise CanonicalProcessingError("Translator attempt work root is not a real directory")
    source_ref = str(accepted.document["handoff_media_ref"])
    source = accepted.handoff_media_path
    _assert_contained(settings.media_dir, source)
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise CanonicalProcessingError("Translator source handoff is not a usable regular file")

    policy_mode = os.environ.get("DUBVI_POLICY_MODE", "off").strip().lower()
    if policy_mode not in {"off", "review", "enforce"}:
        raise CanonicalProcessingError("DUBVI_POLICY_MODE must be off, review, or enforce")
    runtime_config = build_canonical_runtime_config(accepted.document)
    legacy_job = {
        "id": job_id,
        "sourceKey": source_ref,
        "sourceName": source.name,
        "configJson": runtime_config,
        "attempt_number": 1,
    }
    adapter = CanonicalRpcAdapter()
    policy_holder: dict[str, object] = {"deep_policy": None}

    def observe_policy(policy_result: object) -> None:
        deep_policy = normalize_deep_policy(policy_result, policy_mode)
        policy_holder["deep_policy"] = deep_policy
        if deep_policy is None:
            return
        decision = deep_policy["overall_decision"]
        if policy_mode == "enforce" and decision == "reject":
            raise CanonicalPolicyReject(deep_policy)
        if policy_mode == "enforce" and decision in {"review", "error", "unavailable"}:
            raise CanonicalPolicyReviewRequired(deep_policy)

    legacy_settings = _legacy_settings(settings, work_base, policy_mode)
    try:
        _legacy_process_job(
            legacy_settings,
            adapter,
            legacy_job,
            policy_observer=observe_policy,
            canonical_policy_outcomes=True,
        )
    except CanonicalPolicyReject as error:
        return _receipt(job_id, "rejected", deep_policy=error.deep_policy, error_classification="DEEP_POLICY_REJECT")
    except CanonicalPolicyReviewRequired as error:
        return _receipt(job_id, "review_required", deep_policy=error.deep_policy, error_classification="DEEP_POLICY_REVIEW")
    except Exception as error:
        return _receipt(
            job_id,
            "failed",
            deep_policy=policy_holder["deep_policy"],
            error_classification="PROCESSING_ERROR",
            diagnostic_summary=str(error),
        )
    if adapter.last_status != "complete" or adapter.completed_output_path is None:
        return _receipt(job_id, "incomplete", deep_policy=policy_holder["deep_policy"])
    try:
        output = publish_canonical_output(settings, job_id, adapter.completed_output_path)
    except Exception as error:
        return _receipt(
            job_id,
            "failed",
            deep_policy=policy_holder["deep_policy"],
            error_classification="OUTPUT_PUBLICATION_ERROR",
            diagnostic_summary=str(error),
        )
    return _receipt(
        job_id,
        "succeeded",
        deep_policy=policy_holder["deep_policy"],
        output_ref=output.path.relative_to(settings.output_dir).as_posix(),
        output_fingerprint=output.fingerprint,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--translator-job-id", required=True)
    parser.add_argument("--translator-job-root", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--status-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if not args.child:
        return 2
    try:
        settings = P1CSettings(
            Path(args.translator_job_root).resolve(strict=False),
            Path(args.media_root).resolve(strict=False),
            Path(args.status_root).resolve(strict=False),
            Path(args.output_root).resolve(strict=False),
        )
        receipt = run_child_job(settings, args.translator_job_id)
        sys.stdout.buffer.write(canonical_json_bytes(receipt))
        return 0
    except Exception as error:
        receipt = _receipt(
            args.translator_job_id,
            "failed",
            error_classification="CHILD_BOUNDARY_ERROR",
            diagnostic_summary=str(error),
        )
        sys.stdout.buffer.write(canonical_json_bytes(receipt))
        return 0


def _legacy_process_job(settings, adapter: CanonicalRpcAdapter, job: dict[str, object], **kwargs: object) -> None:
    from dubvi_worker import process_job

    process_job(settings, adapter, job, **kwargs)


def _legacy_settings(settings: P1CSettings, work_base: Path, policy_mode: str):
    from dubvi_worker import Settings as LegacySettings

    return LegacySettings(
        api_base="",
        token="",
        media_dir=settings.media_dir,
        output_dir=work_base,
        model_dir=work_base / ".models",
        log_dir=work_base,
        device=os.environ.get("DUBVI_DEVICE", "cpu"),
        whisper_model=os.environ.get("DUBVI_WHISPER_MODEL", "small"),
        poll_seconds=10,
        tts_provider=os.environ.get("DUBVI_TTS_PROVIDER", "capcut").lower(),
        capcut_voice=os.environ.get("DUBVI_CAPCUT_VOICE", "BV421_vivn_streaming"),
        translation_engine=os.environ.get("DUBVI_TRANSLATION_ENGINE", "llm").lower(),
        llm_provider=os.environ.get("DUBVI_LLM_PROVIDER", "gemini").lower(),
        gemini_api_key=None,
        deepseek_api_key=None,
        openai_api_key=None,
        policy_mode=policy_mode,
        policy_enable_ai_judge=False,
        policy_min_duration_seconds=35.0,
        policy_min_speech_density=0.60,
        policy_judge_threshold=8.0,
    )


def _receipt(
    job_id: str,
    outcome: str,
    *,
    deep_policy: object | None = None,
    output_ref: str | None = None,
    output_fingerprint: str | None = None,
    error_classification: str | None = None,
    diagnostic_summary: str | None = None,
) -> dict[str, object]:
    if outcome not in _OUTCOMES:
        raise CanonicalProcessingError("unsupported child receipt outcome")
    result: dict[str, object] = {
        "receipt_version": 1,
        "message_kind": "translator_processing_receipt",
        "translator_job_id": _canonical_uuid(job_id, "translator_job_id"),
        "attempt_number": 1,
        "outcome": outcome,
    }
    if deep_policy is not None:
        result["deep_policy"] = deep_policy
    if output_ref is not None:
        result["output_ref"] = output_ref
    if output_fingerprint is not None:
        result["output_fingerprint"] = output_fingerprint
    if error_classification is not None:
        result["error_classification"] = error_classification[:128]
    if diagnostic_summary is not None:
        result["diagnostic_summary"] = diagnostic_summary[:MAX_DIAGNOSTIC_BYTES]
    return _validate_receipt(result)


def _validate_receipt(document: Mapping[str, object]) -> dict[str, object]:
    allowed = {
        "receipt_version", "message_kind", "translator_job_id", "attempt_number",
        "outcome", "deep_policy", "output_ref", "output_fingerprint",
        "error_classification", "diagnostic_summary",
    }
    if not isinstance(document, Mapping) or set(document) - allowed:
        raise ContractValidationError("Translator receipt has unsupported fields")
    if document.get("receipt_version") != 1 or document.get("message_kind") != "translator_processing_receipt":
        raise ContractValidationError("Translator receipt identity is invalid")
    _canonical_uuid(document.get("translator_job_id"), "translator_job_id")
    if document.get("attempt_number") != 1 or document.get("outcome") not in _OUTCOMES:
        raise ContractValidationError("Translator receipt attempt or outcome is invalid")
    if "output_fingerprint" in document:
        value = document["output_fingerprint"]
        if (
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
            or any(char not in "0123456789abcdef" for char in value[7:])
        ):
            raise ContractValidationError("Translator receipt output fingerprint is invalid")
    if "output_ref" in document:
        output_ref = document["output_ref"]
        if (
            not isinstance(output_ref, str)
            or not output_ref
            or output_ref != output_ref.strip()
            or output_ref.startswith("/")
            or "\\" in output_ref
            or ":" in output_ref
            or any(part in {"", ".", ".."} for part in output_ref.split("/"))
        ):
            raise ContractValidationError("Translator receipt output reference is unsafe")
    outcome = document.get("outcome")
    if outcome == "succeeded" and (
        "output_ref" not in document or "output_fingerprint" not in document
    ):
        raise ContractValidationError("succeeded receipt requires canonical output evidence")
    if outcome in {"rejected", "review_required"} and "deep_policy" not in document:
        raise ContractValidationError("policy terminal receipt requires deep_policy evidence")
    if outcome == "failed" and "error_classification" not in document:
        raise ContractValidationError("failed receipt requires error classification")
    if "diagnostic_summary" in document and (
        not isinstance(document["diagnostic_summary"], str)
        or len(document["diagnostic_summary"].encode("utf-8")) > MAX_DIAGNOSTIC_BYTES
    ):
        raise ContractValidationError("Translator receipt diagnostic is oversized")
    payload = canonical_json_bytes(document)
    if len(payload) > MAX_RECEIPT_BYTES:
        raise ContractValidationError("Translator receipt exceeds 64KiB")
    return dict(document)


def _receipt_to_terminal_event(document: Mapping[str, object], receipt: Mapping[str, object]) -> dict[str, object] | None:
    outcome = receipt.get("outcome")
    if outcome == "incomplete":
        return None
    if outcome not in {"succeeded", "failed", "rejected", "review_required"}:
        raise CanonicalProcessingError("Translator receipt does not identify a terminal outcome")
    event_kind = str(outcome)
    event: dict[str, object] = {
        "contract_version": 1,
        "message_kind": "engine_status_event",
        "event_id": str(uuid.uuid4()),
        "engine_kind": "translator",
        "engine_job_id": document["translator_job_id"],
        "dispatch_id": document["dispatch_id"],
        "correlation_id": document["dispatch_id"],
        "sequence": 3,
        "attempt_number": document.get("attempt_number", 1),
        "event_kind": event_kind,
        "state": event_kind,
        "occurred_at_utc": _utc_now(),
        "lease_id": document.get("lease_id"),
        "handoff_id": document["handoff_id"],
        "handoff_ref": document["handoff_sidecar_ref"],
    }
    if event_kind == "succeeded":
        event["output_fingerprint"] = receipt.get("output_fingerprint")
    elif event_kind == "rejected":
        event["error_classification"] = "DEEP_POLICY_REJECT"
    elif event_kind == "review_required":
        event["error_classification"] = "DEEP_POLICY_REVIEW"
    else:
        event["error_classification"] = receipt.get("error_classification", "PROCESSING_ERROR")
    if receipt.get("diagnostic_summary") is not None:
        event["diagnostic_summary"] = receipt["diagnostic_summary"]
    if receipt.get("deep_policy") is not None:
        event["deep_policy"] = receipt["deep_policy"]
    try:
        return validate_engine_status_event(event)
    except ContractValidationError as error:
        raise CanonicalProcessingError(f"terminal Translator event is invalid: {error}") from error


def _accepted_matches(event: Mapping[str, object], document: Mapping[str, object]) -> bool:
    return (
        event.get("engine_kind") == "translator"
        and event.get("engine_job_id") == document.get("translator_job_id")
        and event.get("dispatch_id") == document.get("dispatch_id")
        and event.get("correlation_id") == document.get("dispatch_id")
        and event.get("attempt_number") == document.get("attempt_number", 1)
        and event.get("sequence") == 1
        and event.get("event_kind") == "accepted"
        and event.get("state") == "accepted"
        and event.get("handoff_id") == document.get("handoff_id")
        and event.get("handoff_ref") == document.get("handoff_sidecar_ref")
    )


def _started_matches(event: Mapping[str, object], document: Mapping[str, object]) -> bool:
    return (
        _accepted_identity(event, document)
        and event.get("sequence") == 2
        and event.get("event_kind") == "started"
        and event.get("state") == "running"
        and isinstance(event.get("lease_id"), str)
    )


def _accepted_identity(event: Mapping[str, object], document: Mapping[str, object]) -> bool:
    return (
        event.get("engine_kind") == "translator"
        and event.get("engine_job_id") == document.get("translator_job_id")
        and event.get("dispatch_id") == document.get("dispatch_id")
        and event.get("correlation_id") == document.get("dispatch_id")
        and event.get("attempt_number") == document.get("attempt_number", 1)
        and event.get("handoff_id") == document.get("handoff_id")
        and event.get("handoff_ref") == document.get("handoff_sidecar_ref")
    )


def _require_directory(path: Path, name: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise CanonicalProcessingError(f"{name} must be an existing real directory")
    return path.resolve()


def _make_directory_chain(parent: Path, child: Path) -> None:
    if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
        raise CanonicalProcessingError("canonical output directory is unsafe")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise CanonicalProcessingError("canonical output directory is an unsafe symlink")
    if child.is_symlink() or (child.exists() and not child.is_dir()):
        raise CanonicalProcessingError("canonical output directory is unsafe")
    child.mkdir(parents=True, exist_ok=True)


def _assert_contained(root: Path, path: Path) -> None:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as error:
        raise CanonicalProcessingError("canonical path escapes its root") from error


def _stage_copy(stage: Path, source: Path, fingerprint: str) -> None:
    if _real_file(stage):
        if _matches(stage, fingerprint):
            return
        raise CanonicalOutputConflict("canonical output stage differs")
    if stage.is_symlink():
        raise CanonicalOutputConflict("canonical output stage is an unsafe symlink")
    try:
        with source.open("rb") as src, stage.open("xb") as dst:
            while chunk := src.read(_CHUNK_SIZE):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    except FileExistsError:
        if _matches(stage, fingerprint):
            return
        raise CanonicalOutputConflict("racing canonical output stage differs")
    if not _matches(stage, fingerprint):
        raise CanonicalProcessingError("canonical output stage failed verification")


def _real_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _matches(path: Path, fingerprint: str) -> bool:
    return _real_file(path) and _hash(path) == fingerprint


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _bounded_path(path: Path) -> str:
    value = str(path.resolve(strict=False))
    if "\x00" in value or len(value) > 1024:
        raise CanonicalProcessingError("canonical child path is unsafe")
    return value


def _canonical_uuid(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise CanonicalProcessingError(f"{field} is invalid")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as error:
        raise CanonicalProcessingError(f"{field} is invalid") from error
    if value != str(parsed):
        raise CanonicalProcessingError(f"{field} is invalid")
    return value


def _normalize_timestamp(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalProcessingError("policy timestamp is invalid")
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise CanonicalProcessingError("policy timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CanonicalProcessingError("policy timestamp must be timezone-aware")
    utc = parsed.astimezone(timezone.utc)
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{utc.microsecond // 1000:03d}Z"


def _utc_now() -> str:
    return _normalize_timestamp(datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    raise SystemExit(main())
