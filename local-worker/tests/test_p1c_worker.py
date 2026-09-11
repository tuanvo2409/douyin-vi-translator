from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from p1c_config import P1CSettings
from p1c_lease_client import LeaseBridgeResponse, LeaseBridgeUnavailable
from p1c_status import TranslatorStatusError, publish_status_event
import p1c_worker
from p1c_worker import run_lease_aware_work


JOB = "11111111-1111-4111-8111-111111111111"
DISPATCH = "22222222-2222-4222-8222-222222222222"
LEASE = "33333333-3333-4333-8333-333333333333"


def job() -> dict[str, object]:
    return {
        "contract_version": 1,
        "message_kind": "control_plane_to_translator_job",
        "envelope_status": "complete",
        "translator_job_id": JOB,
        "dispatch_id": DISPATCH,
        "reup_job_id": "44444444-4444-4444-8444-444444444444",
        "handoff_id": "55555555-5555-4555-8555-555555555555",
        "correlation_id": DISPATCH,
        "candidate_id": "66666666-6666-4666-8666-666666666666",
        "schedule_id": "77777777-7777-4777-8777-777777777777",
        "channel_id": "88888888-8888-4888-8888-888888888888",
        "channel_slug": "channel",
        "target_platform": "tiktok",
        "localization_profile": "loc-v1",
        "policy_profile": "policy-v1",
        "handoff_media_ref": "home/reup-44444444-4444-4444-8444-444444444444.mp4",
        "handoff_sidecar_ref": "home/reup-44444444-4444-4444-8444-444444444444.meta.json",
        "reup_output_fingerprint": "sha256:" + "a" * 64,
        "created_at_utc": "2026-09-11T00:00:00.000Z",
        "attempt_number": 1,
    }


def lease_response(action: str, *, granted: bool | None = None) -> LeaseBridgeResponse:
    lease = {
        "lease_id": LEASE,
        "resource_class": "HEAVY_MEDIA",
        "owner": f"translator:{JOB}",
        "job_id": JOB,
        "state": "released" if action == "release" else "active",
        "acquired_at": "2026-09-11T00:00:00.000Z",
        "heartbeat_at": "2026-09-11T00:00:00.000Z",
        "expires_at": "2026-09-11T00:01:00.000Z",
        "released_at": "2026-09-11T00:00:30.000Z" if action == "release" else None,
    }
    return LeaseBridgeResponse(action, granted, 15, 60, lease if granted is not False else None)


class Handle:
    def __init__(self, running: bool = False):
        self.running = running
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = P1CSettings(root / "jobs", root / "media", root / "status", root / "output")
        for path in (self.settings.translator_job_dir, self.settings.media_dir, self.settings.engine_status_dir, self.settings.output_dir):
            path.mkdir()
        self.job = job()
        publish_status_event(self.settings, {
            "contract_version": 1,
            "message_kind": "engine_status_event",
            "event_id": "99999999-9999-4999-8999-999999999999",
            "engine_kind": "translator",
            "engine_job_id": JOB,
            "dispatch_id": DISPATCH,
            "correlation_id": DISPATCH,
            "sequence": 1,
            "attempt_number": 1,
            "event_kind": "accepted",
            "state": "accepted",
            "occurred_at_utc": "2026-09-11T00:00:00.000Z",
            "handoff_id": self.job["handoff_id"],
            "handoff_ref": self.job["handoff_sidecar_ref"],
        })

    def test_capacity_denial_never_starts_work_or_started_event(self):
        class Client:
            def acquire(_, job): return lease_response("acquire", granted=False)
        called = []
        result = run_lease_aware_work(self.job, self.settings, Client(), lambda value: called.append(value))
        self.assertEqual("capacity_unavailable", result.outcome)
        self.assertEqual([], called)
        self.assertFalse((self.settings.engine_status_dir / "translator" / JOB / "event-000002.json").exists())

    def test_start_and_normal_completion_release_without_terminal_status(self):
        class Client:
            def acquire(_, job): return lease_response("acquire", granted=True)
            def release(_, job, lease_id): return lease_response("release", granted=True)
        result = run_lease_aware_work(self.job, self.settings, Client(), lambda value: Handle())
        self.assertEqual("work_completed_lease_released", result.outcome)
        self.assertTrue((self.settings.engine_status_dir / "translator" / JOB / "event-000002.json").is_file())
        self.assertFalse((self.settings.engine_status_dir / "translator" / JOB / "event-000003.json").exists())

    def test_started_publication_failure_releases_exact_lease(self):
        released = []
        class Client:
            def acquire(_, job): return lease_response("acquire", granted=True)
            def release(_, job, lease_id): released.append(lease_id); return lease_response("release", granted=True)
        original = p1c_worker.publish_status_event
        try:
            p1c_worker.publish_status_event = lambda *args, **kwargs: (_ for _ in ()).throw(TranslatorStatusError("conflict"))
            result = run_lease_aware_work(self.job, self.settings, Client(), lambda value: (_ for _ in ()).throw(AssertionError()))
        finally:
            p1c_worker.publish_status_event = original
        self.assertEqual("started_status_unavailable", result.outcome)
        self.assertEqual([LEASE], released)

    def test_heartbeat_loss_terminates_only_owned_work_and_publishes_lease_lost(self):
        class Client:
            def acquire(_, job): return lease_response("acquire", granted=True)
            def heartbeat(_, job, lease_id): raise LeaseBridgeUnavailable("bridge down")
        handle = Handle(running=True)
        ticks = iter((0.0, 15.0, 15.0))
        result = run_lease_aware_work(
            self.job,
            self.settings,
            Client(),
            lambda value: handle,
            monotonic=lambda: next(ticks),
            sleep=lambda _: None,
        )
        self.assertEqual("lease_lost", result.outcome)
        self.assertTrue(handle.terminated)
        self.assertTrue((self.settings.engine_status_dir / "translator" / JOB / "event-000003.json").is_file())

    def test_missing_accepted_evidence_never_acquires(self):
        settings = P1CSettings(
            self.settings.translator_job_dir,
            self.settings.media_dir,
            Path(self.temp.name) / "missing-status",
            self.settings.output_dir,
        )
        class Client:
            def acquire(_, job): raise AssertionError("acquire must not run")
        result = run_lease_aware_work(self.job, settings, Client(), lambda value: (_ for _ in ()).throw(AssertionError()))
        self.assertEqual("accepted_status_unavailable", result.outcome)


if __name__ == "__main__":
    unittest.main()
