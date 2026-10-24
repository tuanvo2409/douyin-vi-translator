from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from p1c_config import P1CSettings
from p1c_status import TranslatorStatusConflict, publish_status_event


class P1CStatusTests(unittest.TestCase):
    def test_conflicting_immutable_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = P1CSettings(root / "jobs", root / "media", root / "status", root / "output")
            for path in (settings.translator_job_dir, settings.media_dir, settings.engine_status_dir, settings.output_dir):
                path.mkdir()
            base = {
                "contract_version": 1,
                "message_kind": "engine_status_event",
                "event_id": "11111111-1111-4111-8111-111111111111",
                "engine_kind": "translator",
                "engine_job_id": "22222222-2222-4222-8222-222222222222",
                "dispatch_id": "33333333-3333-4333-8333-333333333333",
                "correlation_id": "33333333-3333-4333-8333-333333333333",
                "sequence": 1,
                "attempt_number": 1,
                "event_kind": "accepted",
                "state": "accepted",
                "occurred_at_utc": "2026-09-11T10:00:00.000Z",
                "handoff_id": "44444444-4444-4444-8444-444444444444",
                "handoff_ref": "home/reup-55555555-5555-4555-8555-555555555555.meta.json",
            }
            publish_status_event(settings, base)
            changed = dict(base)
            changed["event_id"] = "66666666-6666-4666-8666-666666666666"
            with self.assertRaises(TranslatorStatusConflict):
                publish_status_event(settings, changed)


if __name__ == "__main__":
    unittest.main()
