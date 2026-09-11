from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import p1c_processing
import p1c_worker
from dubvi_engine_contract import canonical_json_bytes
from p1c_config import P1CSettings
from p1c_intake import accept_next_translator_job, accept_translator_envelope
from p1c_processing import (
    CanonicalChildHandle,
    CanonicalOutputConflict,
    CanonicalProcessingError,
    run_child_job,
    spawn_processing_child,
)
from p1c_status import publish_status_event
from p1c_worker import WorkerResult


class TranslatorDemoRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.settings = P1CSettings(
            root / "jobs", root / "media", root / "status", root / "output", root / "control-plane"
        )
        for path in (
            self.settings.translator_job_dir,
            self.settings.media_dir,
            self.settings.engine_status_dir,
            self.settings.output_dir,
        ):
            path.mkdir()
        (self.settings.control_plane_root / "src" / "dubvi_control_plane").mkdir(parents=True)

    def _write_job(self, number: int) -> tuple[Path, dict[str, object]]:
        if number == 1:
            ids = {
                "translator_job_id": "11111111-1111-4111-8111-111111111111",
                "dispatch_id": "22222222-2222-4222-8222-222222222222",
                "reup_job_id": "33333333-3333-4333-8333-333333333333",
                "handoff_id": "44444444-4444-4444-8444-444444444444",
                "candidate_id": "55555555-5555-4555-8555-555555555555",
                "schedule_id": "66666666-6666-4666-8666-666666666666",
                "channel_id": "77777777-7777-4777-8777-777777777777",
            }
        else:
            ids = {
                "translator_job_id": "88888888-8888-4888-8888-888888888888",
                "dispatch_id": "99999999-9999-4999-8999-999999999999",
                "reup_job_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "handoff_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "candidate_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
                "schedule_id": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
                "channel_id": "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            }
        profile = f"home{number}"
        media = self.settings.media_dir / profile
        media.mkdir()
        media_path = media / f"reup-{ids['reup_job_id']}.mp4"
        media_bytes = f"demo-media-{number}".encode()
        media_path.write_bytes(media_bytes)
        output_fingerprint = "sha256:" + hashlib.sha256(media_bytes).hexdigest()
        handoff = {
            "handoff_schema_version": 2,
            "handoff_status": "complete",
            "handoff_id": ids["handoff_id"],
            "candidate_id": ids["candidate_id"],
            "schedule_id": ids["schedule_id"],
            "dispatch_id": ids["dispatch_id"],
            "reup_job_id": ids["reup_job_id"],
            "correlation_id": ids["dispatch_id"],
            "channel_id": ids["channel_id"],
            "channel_slug": "demo-channel",
            "reup_profile": profile,
            "target_platform": "tiktok",
            "localization_profile": "loc-v1",
            "policy_profile": "policy-v1",
            "source_fingerprint": "sha256:" + "a" * 64,
            "reup_output_fingerprint": output_fingerprint,
            "created_at_utc": "2026-09-11T10:00:00.000Z",
        }
        (media / f"reup-{ids['reup_job_id']}.meta.json").write_bytes(canonical_json_bytes(handoff))
        document: dict[str, object] = {
            "contract_version": 1,
            "message_kind": "control_plane_to_translator_job",
            "envelope_status": "complete",
            **ids,
            "correlation_id": ids["dispatch_id"],
            "channel_slug": "demo-channel",
            "target_platform": "tiktok",
            "localization_profile": "loc-v1",
            "policy_profile": "policy-v1",
            "handoff_media_ref": f"{profile}/{media_path.name}",
            "handoff_sidecar_ref": f"{profile}/{media_path.stem}.meta.json",
            "reup_output_fingerprint": output_fingerprint,
            "created_at_utc": "2026-09-11T10:01:00.000Z",
            "attempt_number": 1,
        }
        path = self.settings.translator_job_dir / f"translator-job-{ids['translator_job_id']}.job.json"
        path.write_bytes(canonical_json_bytes(document))
        return path, document

    def _publish_started(self, document: dict[str, object]) -> None:
        publish_status_event(self.settings, {
            "contract_version": 1,
            "message_kind": "engine_status_event",
            "event_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "engine_kind": "translator",
            "engine_job_id": document["translator_job_id"],
            "dispatch_id": document["dispatch_id"],
            "correlation_id": document["dispatch_id"],
            "sequence": 2,
            "attempt_number": 1,
            "event_kind": "started",
            "state": "running",
            "occurred_at_utc": "2026-09-11T10:02:00.000Z",
            "lease_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "handoff_id": document["handoff_id"],
            "handoff_ref": document["handoff_sidecar_ref"],
        })

    def _publish_terminal(self, document: dict[str, object], kind: str = "succeeded") -> None:
        event: dict[str, object] = {
            "contract_version": 1,
            "message_kind": "engine_status_event",
            "event_id": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            "engine_kind": "translator",
            "engine_job_id": document["translator_job_id"],
            "dispatch_id": document["dispatch_id"],
            "correlation_id": document["dispatch_id"],
            "sequence": 3,
            "attempt_number": 1,
            "event_kind": kind,
            "state": kind,
            "occurred_at_utc": "2026-09-11T10:03:00.000Z",
            "handoff_id": document["handoff_id"],
            "handoff_ref": document["handoff_sidecar_ref"],
        }
        if kind == "succeeded":
            event["output_fingerprint"] = document["reup_output_fingerprint"]
        else:
            event["error_classification"] = "PROCESSING_FAILED"
            event["diagnostic_summary"] = "failed"
        publish_status_event(self.settings, event)

    def test_once_accepts_new_job_and_invokes_canonical_processing_once(self) -> None:
        path, document = self._write_job(1)
        bridge = Mock()
        with patch.object(p1c_worker.TranslatorLeaseBridgeClient, "from_settings", return_value=bridge), \
             patch("p1c_processing.run_canonical_processing", return_value=WorkerResult("capacity_unavailable")) as run:
            report = p1c_worker.run_canonical_once(self.settings)
        self.assertEqual(document["translator_job_id"], report["translator_job_id"])
        self.assertEqual("capacity_unavailable", report["outcome"])
        run.assert_called_once_with(self.settings, path, bridge)
        self.assertTrue((self.settings.engine_status_dir / "translator" / document["translator_job_id"] / "event-000001.json").is_file())

    def test_automatic_queue_skips_terminal_old_job_and_selects_new_job(self) -> None:
        first_path, first = self._write_job(1)
        second_path, second = self._write_job(2)
        accept_translator_envelope(first_path, self.settings)
        self._publish_started(first)
        self._publish_terminal(first)
        accepted = accept_next_translator_job(self.settings)
        self.assertIsNotNone(accepted)
        self.assertEqual(second["translator_job_id"], accepted.document["translator_job_id"])
        self.assertEqual(first_path, first_path)

    def test_started_without_terminal_is_not_rerun_and_later_job_is_selected(self) -> None:
        first_path, first = self._write_job(1)
        _, second = self._write_job(2)
        accept_translator_envelope(first_path, self.settings)
        self._publish_started(first)
        accepted = accept_next_translator_job(self.settings)
        self.assertIsNotNone(accepted)
        self.assertEqual(second["translator_job_id"], accepted.document["translator_job_id"])
        self.assertTrue((self.settings.engine_status_dir / "translator" / first["translator_job_id"] / "event-000002.json").is_file())

    def test_explicit_terminal_job_is_reported_without_rerun(self) -> None:
        path, document = self._write_job(1)
        accept_translator_envelope(path, self.settings)
        self._publish_started(document)
        self._publish_terminal(document)
        with patch.object(p1c_worker.TranslatorLeaseBridgeClient, "from_settings") as bridge, \
             patch("p1c_processing.run_canonical_processing") as run:
            report = p1c_worker.run_canonical_once(
                self.settings, translator_job_id=str(document["translator_job_id"])
            )
        self.assertEqual("succeeded", report["outcome"])
        self.assertEqual("succeeded", report["terminal_state"])
        bridge.assert_not_called()
        run.assert_not_called()

    def test_windows_termination_uses_exact_owned_pid_tree(self) -> None:
        process = Mock()
        process.pid = 4242
        process.poll.return_value = None
        handle = CanonicalChildHandle(process)
        with patch.object(p1c_processing.os, "name", "nt"), patch.object(p1c_processing.subprocess, "run") as run:
            handle.terminate()
        run.assert_called_once_with(
            ["taskkill", "/PID", "4242", "/T", "/F"],
            shell=False,
            check=False,
            stdout=p1c_processing.subprocess.DEVNULL,
            stderr=p1c_processing.subprocess.DEVNULL,
        )
        process.terminate.assert_not_called()

    def test_processing_and_output_error_classifications_are_approved(self) -> None:
        path, document = self._write_job(1)
        accept_translator_envelope(path, self.settings)
        self._publish_started(document)
        with patch.object(p1c_processing, "_legacy_process_job", side_effect=RuntimeError("boom")):
            processing = run_child_job(self.settings, str(document["translator_job_id"]))
        self.assertEqual("PROCESSING_FAILED", processing["error_classification"])

        def fake_process(legacy_settings, adapter, job, **kwargs):
            candidate = legacy_settings.output_dir / job["id"] / "render.mp4"
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_bytes(b"rendered")
            adapter.report(job["id"], "complete", 100, status="complete", output_path=str(candidate))

        with patch.object(p1c_processing, "_legacy_process_job", side_effect=fake_process), \
             patch.object(p1c_processing, "publish_canonical_output", side_effect=CanonicalOutputConflict("different")):
            conflict = run_child_job(self.settings, str(document["translator_job_id"]))
        self.assertEqual("FINAL_PATH_CONFLICT", conflict["error_classification"])

        path2, document2 = self._write_job(2)
        accept_translator_envelope(path2, self.settings)
        self._publish_started(document2)
        with patch.object(p1c_processing, "_legacy_process_job", side_effect=fake_process), \
             patch.object(p1c_processing, "publish_canonical_output", side_effect=CanonicalProcessingError("verify")):
            verify = run_child_job(self.settings, str(document2["translator_job_id"]))
        self.assertEqual("OUTPUT_VERIFY_FAILED", verify["error_classification"])

    def test_provider_environment_is_inherited_by_canonical_child(self) -> None:
        _, document = self._write_job(1)
        process = Mock()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "do-not-print"}, clear=False), \
             patch.object(p1c_processing.subprocess, "Popen", return_value=process) as popen:
            spawn_processing_child(self.settings, document)
        self.assertEqual("do-not-print", popen.call_args.kwargs["env"]["GEMINI_API_KEY"])

    def test_main_once_prints_bounded_machine_result(self) -> None:
        with patch.object(p1c_worker, "load_p1c_settings", return_value=self.settings), \
             patch.object(p1c_worker, "run_canonical_once", return_value={"outcome": "no_work"}):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = p1c_worker.main(["once"])
        self.assertEqual(0, code)
        self.assertEqual({"outcome": "no_work"}, json.loads(output.getvalue()))


if __name__ == "__main__":
    unittest.main()
