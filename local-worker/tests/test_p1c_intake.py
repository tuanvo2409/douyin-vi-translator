from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from dubvi_engine_contract import canonical_json_bytes
from p1c_config import P1CSettings
from p1c_intake import (
    TranslatorIntakeError,
    accept_next_translator_job,
    accept_translator_envelope,
)
from p1c_status import load_status_event


def _fp(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class P1CIntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.jobs = self.root / "jobs"
        self.media = self.root / "media"
        self.status = self.root / "status"
        self.output = self.root / "output"
        for path in (self.jobs, self.media, self.status, self.output):
            path.mkdir()
        self.settings = P1CSettings(self.jobs, self.media, self.status, self.output)
        self.ids = {
            "translator": "66666666-6666-4666-8666-666666666666",
            "dispatch": "11111111-1111-4111-8111-111111111111",
            "reup": "22222222-2222-4222-8222-222222222222",
            "handoff": "33333333-3333-4333-8333-333333333333",
            "candidate": "44444444-4444-4444-8444-444444444444",
            "schedule": "55555555-5555-4555-8555-555555555555",
            "channel": "77777777-7777-4777-8777-777777777777",
        }
        self.media_bytes = b"tiny reup handoff media"
        self.profile = self.media / "home"
        self.profile.mkdir()
        self.media_path = self.profile / f"reup-{self.ids['reup']}.mp4"
        self.media_path.write_bytes(self.media_bytes)
        self.handoff = {
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
            "reup_output_fingerprint": _fp(self.media_bytes),
            "created_at_utc": "2026-09-11T10:00:00.000Z",
        }
        self.sidecar = self.profile / f"reup-{self.ids['reup']}.meta.json"
        self.sidecar.write_bytes(canonical_json_bytes(self.handoff))
        self.envelope = {
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
            "reup_output_fingerprint": _fp(self.media_bytes),
            "created_at_utc": "2026-09-11T10:01:00.000Z",
            "attempt_number": 1,
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    @property
    def envelope_path(self) -> Path:
        return self.jobs / f"translator-job-{self.ids['translator']}.job.json"

    def write_envelope(self, document: dict | None = None) -> None:
        self.envelope_path.write_bytes(canonical_json_bytes(document or self.envelope))

    def test_valid_pair_accepts_and_duplicate_reuses_exact_event(self) -> None:
        self.write_envelope()
        accepted = accept_translator_envelope(
            self.envelope_path, self.settings, occurred_at_utc="2026-09-11T10:02:00+07:00"
        )
        event_path = self.status / "translator" / self.ids["translator"] / "event-000001.json"
        before = event_path.read_bytes()
        repeated = accept_translator_envelope(self.envelope_path, self.settings)
        self.assertEqual(accepted.document, repeated.document)
        self.assertEqual(before, event_path.read_bytes())
        self.assertEqual("accepted", load_status_event(self.settings, self.ids["translator"], 1)["state"])

    def test_handoff_alone_does_not_ack(self) -> None:
        self.assertIsNone(accept_next_translator_job(self.settings))
        self.assertFalse((self.status / "translator").exists())

    def test_envelope_without_handoff_does_not_ack(self) -> None:
        self.sidecar.unlink()
        self.write_envelope()
        with self.assertRaises(TranslatorIntakeError):
            accept_next_translator_job(self.settings)
        self.assertFalse((self.status / "translator").exists())

    def test_malformed_or_noncanonical_handoff_does_not_ack(self) -> None:
        self.sidecar.write_text("{\"handoff_status\": \"complete\"}\n", encoding="utf-8")
        self.write_envelope()
        with self.assertRaises(TranslatorIntakeError):
            accept_next_translator_job(self.settings)
        self.sidecar.write_text(json.dumps(self.handoff, indent=2), encoding="utf-8")
        with self.assertRaises(TranslatorIntakeError):
            accept_next_translator_job(self.settings)

    def test_identity_and_fingerprint_mismatches_do_not_ack(self) -> None:
        for field, value in (
            ("localization_profile", "other-loc"),
            ("policy_profile", "other-policy"),
            ("channel_slug", "other-channel"),
            ("target_platform", "shorts"),
            ("dispatch_id", self.ids["reup"]),
            ("reup_output_fingerprint", "sha256:" + "f" * 64),
        ):
            with self.subTest(field=field):
                document = dict(self.envelope)
                document[field] = value
                self.write_envelope(document)
                with self.assertRaises(TranslatorIntakeError):
                    accept_next_translator_job(self.settings)
                self.envelope_path.unlink()

    def test_media_source_is_verified_by_sha256_and_not_modified(self) -> None:
        self.write_envelope()
        before = self.media_path.read_bytes()
        accepted = accept_next_translator_job(self.settings)
        self.assertIsNotNone(accepted)
        self.assertEqual(before, self.media_path.read_bytes())
        self.media_path.write_bytes(b"different")
        self.write_envelope()
        with self.assertRaises(TranslatorIntakeError):
            accept_next_translator_job(self.settings)


if __name__ == "__main__":
    unittest.main()
