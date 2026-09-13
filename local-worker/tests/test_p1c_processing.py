from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import p1c_processing
from dubvi_engine_contract import canonical_json_bytes
from p1c_config import P1CSettings
from p1c_intake import accept_translator_envelope
from p1c_lease_client import LeaseBridgeResponse
from p1c_processing import (
    CanonicalOutputConflict,
    CanonicalPolicyReject,
    CanonicalPolicyReviewRequired,
    CanonicalRpcAdapter,
    build_canonical_runtime_config,
    normalize_deep_policy,
    publish_canonical_output,
    run_canonical_processing,
    run_child_job,
)
from p1c_status import load_status_event, publish_status_event
from p1c_worker import WorkerResult


def _fingerprint(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class Handle:
    def __init__(self, receipt: dict[str, object] | None = None, error: Exception | None = None):
        self.receipt = receipt
        self.error = error
        self.terminated = False

    def poll(self):
        return 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def read_receipt(self):
        if self.error is not None:
            raise self.error
        assert self.receipt is not None
        return self.receipt


class Client:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.released: list[str] = []

    def acquire(self, job):
        lease = {
            "lease_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "resource_class": "HEAVY_MEDIA",
            "owner": f"translator:{self.job_id}",
            "job_id": self.job_id,
            "state": "active",
            "acquired_at": "2026-09-11T00:00:00.000Z",
            "heartbeat_at": "2026-09-11T00:00:00.000Z",
            "expires_at": "2026-09-11T00:01:00.000Z",
            "released_at": None,
        }
        return LeaseBridgeResponse("acquire", True, 15, 60, lease)

    def release(self, job, lease_id):
        self.released.append(lease_id)
        lease = {
            "lease_id": lease_id,
            "resource_class": "HEAVY_MEDIA",
            "owner": f"translator:{self.job_id}",
            "job_id": self.job_id,
            "state": "released",
            "acquired_at": "2026-09-11T00:00:00.000Z",
            "heartbeat_at": "2026-09-11T00:00:00.000Z",
            "expires_at": "2026-09-11T00:01:00.000Z",
            "released_at": "2026-09-11T00:00:30.000Z",
        }
        return LeaseBridgeResponse("release", None, 15, 60, lease)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = P1CSettings(root / "jobs", root / "media", root / "status", root / "output")
        for path in (self.settings.translator_job_dir, self.settings.media_dir, self.settings.engine_status_dir, self.settings.output_dir):
            path.mkdir()
        self.ids = {
            "translator": "11111111-1111-4111-8111-111111111111",
            "dispatch": "22222222-2222-4222-8222-222222222222",
            "reup": "33333333-3333-4333-8333-333333333333",
            "handoff": "44444444-4444-4444-8444-444444444444",
            "candidate": "55555555-5555-4555-8555-555555555555",
            "schedule": "66666666-6666-4666-8666-666666666666",
            "channel": "77777777-7777-4777-8777-777777777777",
        }
        self.media_bytes = b"tiny canonical handoff"
        profile = self.settings.media_dir / "home"
        profile.mkdir()
        (profile / f"reup-{self.ids['reup']}.mp4").write_bytes(self.media_bytes)
        handoff = {
            "handoff_schema_version": 2,
            "handoff_status": "complete",
            "handoff_id": self.ids["handoff"],
            "candidate_id": self.ids["candidate"],
            "schedule_id": self.ids["schedule"],
            "dispatch_id": self.ids["dispatch"],
            "reup_job_id": self.ids["reup"],
            "correlation_id": self.ids["dispatch"],
            "channel_id": self.ids["channel"],
            "channel_slug": "demo-channel",
            "reup_profile": "home",
            "target_platform": "tiktok",
            "localization_profile": "loc-v1",
            "policy_profile": "policy-v1",
            "source_fingerprint": "sha256:" + "a" * 64,
            "reup_output_fingerprint": _fingerprint(self.media_bytes),
            "created_at_utc": "2026-09-11T00:00:00.000Z",
        }
        (profile / f"reup-{self.ids['reup']}.meta.json").write_bytes(canonical_json_bytes(handoff))
        self.document = {
            "contract_version": 1,
            "message_kind": "control_plane_to_translator_job",
            "envelope_status": "complete",
            "translator_job_id": self.ids["translator"],
            "dispatch_id": self.ids["dispatch"],
            "reup_job_id": self.ids["reup"],
            "handoff_id": self.ids["handoff"],
            "correlation_id": self.ids["dispatch"],
            "candidate_id": self.ids["candidate"],
            "schedule_id": self.ids["schedule"],
            "channel_id": self.ids["channel"],
            "channel_slug": "demo-channel",
            "target_platform": "tiktok",
            "localization_profile": "loc-v1",
            "policy_profile": "policy-v1",
            "handoff_media_ref": f"home/reup-{self.ids['reup']}.mp4",
            "handoff_sidecar_ref": f"home/reup-{self.ids['reup']}.meta.json",
            "reup_output_fingerprint": _fingerprint(self.media_bytes),
            "created_at_utc": "2026-09-11T00:01:00.000Z",
            "attempt_number": 1,
        }
        self.envelope_path = self.settings.translator_job_dir / f"translator-job-{self.ids['translator']}.job.json"
        self.envelope_path.write_bytes(canonical_json_bytes(self.document))
        accept_translator_envelope(self.envelope_path, self.settings, occurred_at_utc="2026-09-11T00:02:00.000Z")
        self.started = {
            "contract_version": 1,
            "message_kind": "engine_status_event",
            "event_id": "88888888-8888-4888-8888-888888888888",
            "engine_kind": "translator",
            "engine_job_id": self.ids["translator"],
            "dispatch_id": self.ids["dispatch"],
            "correlation_id": self.ids["dispatch"],
            "sequence": 2,
            "attempt_number": 1,
            "event_kind": "started",
            "state": "running",
            "occurred_at_utc": "2026-09-11T00:03:00.000Z",
            "lease_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "handoff_id": self.ids["handoff"],
            "handoff_ref": self.document["handoff_sidecar_ref"],
        }

    def test_runtime_profile_and_policy_normalization_are_exact_and_bounded(self):
        profile = build_canonical_runtime_config(self.document, capcut_voice="BV007_streaming")
        self.assertEqual({"name": "BV007_streaming", "maxTempo": 1.12}, profile["voice"])
        self.assertEqual({"enabled": True, "sampleFrames": 3, "discoveryFrames": 8, "minConfidence": 65, "llmCorrection": False}, profile["ocr"])
        self.assertEqual({"xPercent": 2, "yPercent": 66, "widthPercent": 96, "heightPercent": 9.8, "blurPx": 24}, profile["roi"])
        self.assertEqual("duck", profile["audioMode"])
        result = SimpleNamespace(
            overall_decision=SimpleNamespace(value="review"),
            reason_codes=("A", "B"),
            needs_review=True,
            evaluated_at="2026-09-11T07:03:04.123456+07:00",
        )
        self.assertEqual(
            {
                "policy_schema_version": 1,
                "mode": "review",
                "overall_decision": "review",
                "reason_codes": ["A", "B"],
                "needs_review": True,
                "evaluated_at": "2026-09-11T00:03:04.123Z",
            },
            normalize_deep_policy(result, "review"),
        )

    def test_typed_canonical_policy_outcomes_are_distinct(self):
        deep = {"policy_schema_version": 1, "mode": "enforce", "overall_decision": "reject", "reason_codes": ["R"], "needs_review": False, "evaluated_at": "2026-09-11T00:00:00.000Z"}
        self.assertEqual(deep, CanonicalPolicyReject(deep).deep_policy)
        self.assertEqual(deep, CanonicalPolicyReviewRequired(deep).deep_policy)

    def test_adapter_is_in_memory_and_correct_ocr_is_forbidden(self):
        adapter = CanonicalRpcAdapter()
        adapter.report(self.ids["translator"], "complete", 100, status="complete", output_path="x.mp4")
        self.assertEqual(Path("x.mp4"), adapter.completed_output_path)
        with self.assertRaises(p1c_processing.CanonicalProcessingError):
            adapter.correct_ocr({"ocrText": "x"})

    def test_output_publication_is_source_preserving_and_non_replacing(self):
        work = self.settings.output_dir / ".p1c-work" / self.ids["translator"]
        work.mkdir(parents=True)
        candidate = work / "render.mp4"
        candidate.write_bytes(b"rendered")
        published = publish_canonical_output(self.settings, self.ids["translator"], candidate)
        self.assertEqual(b"rendered", candidate.read_bytes())
        self.assertFalse(published.reused_existing_final)
        self.assertTrue(publish_canonical_output(self.settings, self.ids["translator"], candidate).reused_existing_final)
        published.path.write_bytes(b"different")
        with self.assertRaises(CanonicalOutputConflict):
            publish_canonical_output(self.settings, self.ids["translator"], candidate)

    def test_child_reload_and_fake_processing_publishes_canonical_output_without_status_write(self):
        publish_status_event(self.settings, self.started)

        def fake_process(legacy_settings, adapter, job, **kwargs):
            target = legacy_settings.output_dir / job["id"] / "legacy.mp4"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake rendered output")
            adapter.report(job["id"], "complete", 100, status="complete", output_path=str(target))

        with patch.dict(os.environ, {"DUBVI_POLICY_MODE": "off"}, clear=False), patch.object(p1c_processing, "_legacy_process_job", side_effect=fake_process):
            receipt = run_child_job(self.settings, self.ids["translator"])
        self.assertEqual("succeeded", receipt["outcome"])
        self.assertTrue((self.settings.output_dir / "p1c" / self.ids["translator"] / "output.mp4").is_file())
        self.assertFalse((self.settings.engine_status_dir / "translator" / self.ids["translator"] / "event-000003.json").exists())

    def test_parent_maps_receipt_before_release_and_crash_keeps_running(self):
        success_receipt = {
            "receipt_version": 1,
            "message_kind": "translator_processing_receipt",
            "translator_job_id": self.ids["translator"],
            "attempt_number": 1,
            "outcome": "succeeded",
            "output_ref": "p1c/11111111-1111-4111-8111-111111111111/output.mp4",
            "output_fingerprint": "sha256:" + "b" * 64,
        }
        client = Client(self.ids["translator"])
        with patch.object(p1c_processing, "spawn_processing_child", return_value=Handle(success_receipt)):
            result = run_canonical_processing(self.settings, self.envelope_path, client, occurred_at_utc="2026-09-11T00:04:00Z")
        self.assertEqual("work_completed_lease_released", result.outcome)
        self.assertEqual("succeeded", load_status_event(self.settings, self.ids["translator"], 3)["state"])
        self.assertEqual(["aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"], client.released)

        crash_settings = P1CSettings(self.settings.translator_job_dir, self.settings.media_dir, Path(self.temp.name) / "crash-status", self.settings.output_dir)
        crash_settings.engine_status_dir.mkdir()
        accept_translator_envelope(self.envelope_path, crash_settings, occurred_at_utc="2026-09-11T00:02:00Z")
        crash_client = Client(self.ids["translator"])
        with patch.object(p1c_processing, "spawn_processing_child", return_value=Handle(error=RuntimeError("child died"))):
            result = run_canonical_processing(crash_settings, self.envelope_path, crash_client, occurred_at_utc="2026-09-11T00:04:00Z")
        self.assertEqual("completion_reconciliation_required_lease_released", result.outcome)
        self.assertFalse((crash_settings.engine_status_dir / "translator" / self.ids["translator"] / "event-000003.json").exists())


if __name__ == "__main__":
    unittest.main()
