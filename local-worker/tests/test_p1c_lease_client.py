from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from p1c_lease_client import (
    LeaseBridgeUnavailable,
    TranslatorLeaseBridgeClient,
    _validate_response,
)


JOB = "11111111-1111-4111-8111-111111111111"
DISPATCH = "22222222-2222-4222-8222-222222222222"
LEASE = "33333333-3333-4333-8333-333333333333"


def payload(*, action="acquire", granted=True, state="active", released_at=None):
    lease = {
        "lease_id": LEASE,
        "resource_class": "HEAVY_MEDIA",
        "owner": f"translator:{JOB}",
        "job_id": JOB,
        "state": state,
        "acquired_at": "2026-09-11T00:00:00.000Z",
        "heartbeat_at": "2026-09-11T00:00:15.000Z",
        "expires_at": "2026-09-11T00:01:00.000Z",
        "released_at": released_at,
    }
    return {
        "bridge_version": 1,
        "action": action,
        "ok": True,
        "engine_kind": "translator",
        "timing": {"heartbeat_seconds": 15, "ttl_seconds": 60},
        "lease": lease if granted else None,
        "granted": granted,
    }


class LeaseClientTests(unittest.TestCase):
    def test_response_validation_requires_translator_identity_and_safe_timestamps(self):
        self.assertEqual(15, _validate_response(payload(), "acquire", JOB).heartbeat_seconds)
        bad = payload()
        bad["engine_kind"] = "reup"
        with self.assertRaises(LeaseBridgeUnavailable):
            _validate_response(bad, "acquire", JOB)
        bad = payload()
        bad["lease"] = {**bad["lease"], "expires_at": "2026-09-11T00:00:15.000Z"}
        with self.assertRaises(LeaseBridgeUnavailable):
            _validate_response(bad, "acquire", JOB)

    def test_released_response_requires_terminal_release_evidence(self):
        valid = _validate_response(
            payload(action="release", state="released", released_at="2026-09-11T00:00:30.000Z"),
            "release",
            JOB,
            LEASE,
        )
        self.assertEqual("released", valid.lease["state"])
        with self.assertRaises(LeaseBridgeUnavailable):
            _validate_response(payload(action="release", state="released"), "release", JOB, LEASE)

    def test_client_uses_argv_bridge_shell_false_and_control_plane_pythonpath(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src" / "dubvi_control_plane").mkdir(parents=True)
            client = TranslatorLeaseBridgeClient(root, executable="python-test")
            completed = type("Completed", (), {"returncode": 0, "stdout": json.dumps(payload()), "stderr": ""})()
            job = {"translator_job_id": JOB, "dispatch_id": DISPATCH}
            with patch("p1c_lease_client.subprocess.run", return_value=completed) as run:
                result = client.acquire(job)
            self.assertTrue(result.granted)
            args, kwargs = run.call_args
            self.assertEqual("python-test", args[0][0])
            self.assertIn("--engine-kind", args[0])
            self.assertIn("translator", args[0])
            self.assertFalse(kwargs["shell"])
            self.assertEqual(10, kwargs["timeout"])
            self.assertIn(str(root / "src"), kwargs["env"]["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
