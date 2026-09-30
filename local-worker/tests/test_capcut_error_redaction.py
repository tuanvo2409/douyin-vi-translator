from __future__ import annotations

import io
import json
import logging
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import patch

import dubvi_worker
from capcut_tts_api.exceptions import CapCutAPIError, CapCutTaskError, CapCutUploadError
from capcut_tts_api.client import CapCutClient, _checked_json_response as checked_client_response
from capcut_tts_api.models import DeviceConfig, UploadResult
from capcut_tts_api.uploader import VODUploader, _checked_json_response as checked_upload_response
from tts_timing import run_tts_timing


class CapCutErrorRedactionTests(unittest.TestCase):
    def _alignment(self) -> dict[str, object]:
        return {
            "dialogue": [{
                "utteranceId": "u1",
                "resolvedZh": "租房",
                "nativeVi": "Thuê nhà",
                "sourcePositions": [1],
                "startMs": 0,
                "endMs": 1000,
                "reviewRequired": False,
                "reviewFlags": [],
                "ttsEligible": True,
                "alignedCues": [{
                    "position": 1,
                    "startMs": 0,
                    "endMs": 1000,
                    "translatedTextVi": "Thuê nhà",
                    "sourcePositions": [1],
                }],
            }],
            "excluded": [],
            "coverage": {},
        }

    def test_capcut_4xx_secret_and_signed_url_are_absent_from_exception_manifest_and_log(self) -> None:
        secret = "TEST_SECRET"
        signed_url = "https://cdn.example.test/audio.mp3?X-Amz-Signature=SIGNED_SECRET"

        class FakeClient:
            def generate_speech(self, **_kwargs):
                raise CapCutAPIError(
                    f"HTTP 403 body token={secret} signed={signed_url}",
                    status_code=403,
                    response_data={"token": secret, "url": signed_url},
                )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            logger = logging.getLogger("dubvi-worker")
            logger.addHandler(handler)
            try:
                with patch("capcut_tts_api.client.CapCutClient", return_value=FakeClient()), patch(
                    "dubvi_worker.time.sleep"
                ):
                    with self.assertRaises(RuntimeError) as raised:
                        dubvi_worker.capcut_synthesize("Thuê nhà", "BV075_streaming", output / "direct.mp3")
                    self.assertNotIn(secret, str(raised.exception))
                    self.assertNotIn(signed_url, str(raised.exception))
                    formatted_error = "".join(traceback.format_exception(raised.exception))
                    self.assertNotIn(secret, formatted_error)
                    self.assertNotIn(signed_url, formatted_error)
                    manifest = run_tts_timing(
                        self._alignment(),
                        output,
                        voice="BV075_streaming",
                        tts_provider="capcut",
                        synthesizer=dubvi_worker.capcut_synthesize,
                        measure_duration=lambda _path: 500,
                        fit_audio=lambda source, target, _slot, _tempo: (500, False),
                    )
            finally:
                logger.removeHandler(handler)

        serialized_manifest = json.dumps(manifest, ensure_ascii=False)
        serialized_log = log_capture.getvalue()
        for leaked in (secret, signed_url):
            self.assertNotIn(leaked, serialized_manifest)
            self.assertNotIn(leaked, serialized_log)

    def test_interrupted_capcut_download_does_not_replace_existing_final_audio(self) -> None:
        class FakeClient:
            calls = 0

            def generate_speech(self, **_kwargs):
                self.calls += 1
                return {"data": {"tasks": [{"payload": json.dumps({"audio_url": "https://example.test/audio.mp3"})}]}}

        class FakeResponse:
            content = b"valid audio" * 512

            def raise_for_status(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "voice.mp3"
            target.write_bytes(b"previous-valid-audio")
            attempted_temps: list[Path] = []

            def interrupted_write(path: Path, data: bytes) -> int:
                if (
                    path.parent == target.parent
                    and path.name.startswith(f".{target.name}.")
                    and path.name.endswith(".part")
                ):
                    attempted_temps.append(path)
                    with path.open("wb") as handle:
                        handle.write(b"partial-corrupt-audio")
                    raise OSError("interrupted write")
                with path.open("wb") as handle:
                    return handle.write(data)

            client = FakeClient()
            with patch("capcut_tts_api.client.CapCutClient", return_value=client), patch(
                "requests.get", return_value=FakeResponse()
            ) as download, patch("dubvi_worker.time.sleep"), patch.object(
                Path, "write_bytes", autospec=True, side_effect=interrupted_write
            ) as write_file:
                with self.assertRaisesRegex(RuntimeError, "CapCut TTS thất bại"):
                    dubvi_worker.capcut_synthesize("Thuê nhà", "BV075_streaming", target)

            self.assertEqual(3, client.calls)
            self.assertEqual(3, download.call_count)
            self.assertEqual(3, write_file.call_count)
            self.assertEqual(3, len(attempted_temps))
            self.assertEqual(b"previous-valid-audio", target.read_bytes())
            self.assertFalse(list(target.parent.glob(f".{target.name}.*.part")))

    def test_short_capcut_download_reaches_http_response_before_rejecting(self) -> None:
        class FakeClient:
            calls = 0

            def generate_speech(self, **_kwargs):
                self.calls += 1
                return {"data": {"tasks": [{"payload": json.dumps({"audio_url": "https://example.test/audio.mp3"})}]}}

        class ShortResponse:
            content = b"too short"

            def raise_for_status(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "voice.mp3"
            target.write_bytes(b"previous-valid-audio")
            client = FakeClient()
            with patch("capcut_tts_api.client.CapCutClient", return_value=client), patch(
                "requests.get", return_value=ShortResponse()
            ) as download, patch("dubvi_worker.time.sleep"):
                with self.assertRaisesRegex(RuntimeError, "CapCut TTS thất bại"):
                    dubvi_worker.capcut_synthesize("Thuê nhà", "BV075_streaming", target)
            self.assertEqual(3, client.calls)
            self.assertEqual(3, download.call_count)
            self.assertEqual(b"previous-valid-audio", target.read_bytes())

    def test_client_and_uploader_http_errors_do_not_expose_raw_secret_body(self) -> None:
        secret = "TEST_SECRET"
        signed_url = "https://cdn.example.test/audio.mp3?X-Amz-Signature=SIGNED_SECRET"

        class FakeResponse:
            status_code = 403
            text = f"token={secret} signed={signed_url}"

            def json(self):
                return {"token": secret, "signed_url": signed_url}

        for checker in (checked_client_response, checked_upload_response):
            with self.subTest(checker=checker.__module__):
                with self.assertRaises(CapCutAPIError) as raised:
                    checker(FakeResponse(), "fake_request")
                self.assertNotIn(secret, str(raised.exception))
                self.assertNotIn(signed_url, str(raised.exception))

    def test_missing_stt_task_error_does_not_expose_token_or_signed_url(self) -> None:
        secret = "STT_TASK_TOKEN"
        signed_url = "https://cdn.example.test/stt.json?X-Amz-Signature=STT_SIGNATURE"
        client = CapCutClient.__new__(CapCutClient)
        client.upload_audio = lambda _path: UploadResult("vid", "md5", "local-md5", 1000)
        client.create_stt_task = lambda **_kwargs: {
            "data": {"tasks": [], "token": secret, "url": signed_url}
        }

        with self.assertRaises(CapCutTaskError) as raised:
            client.transcribe_file("audio.mp3")

        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(signed_url, str(raised.exception))

    def test_failed_stt_poll_error_does_not_expose_token_or_signed_url(self) -> None:
        secret = "STT_POLL_TOKEN"
        signed_url = "https://cdn.example.test/stt.json?X-Amz-Signature=POLL_SIGNATURE"
        client = CapCutClient.__new__(CapCutClient)
        client.upload_audio = lambda _path: UploadResult("vid", "md5", "local-md5", 1000)
        client.create_stt_task = lambda **_kwargs: {"data": {"tasks": [{"id": "task-1", "token": secret}]}}
        client.query_stt_task = lambda _task_id, _token: {
            "data": {
                "tasks": [{"status": "failed"}],
                "token": secret,
                "signed_url": signed_url,
            }
        }

        with patch("capcut_tts_api.client.time.sleep"):
            with self.assertRaises(CapCutTaskError) as raised:
                client.transcribe_file("audio.mp3", poll_interval=0)

        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(signed_url, str(raised.exception))

    def test_upload_sign_missing_field_error_does_not_expose_raw_payload(self) -> None:
        secret = "UPLOAD_SESSION_SECRET"
        signed_url = "https://upload.example.test/sign?X-Amz-Signature=UPLOAD_SIGNATURE"

        class FakeResponse:
            status_code = 200

            def json(self):
                return {
                    "data": {
                        "access_key_id": "access-key",
                        "secret_access_key": secret,
                        "session_token": "session-token",
                        "space_name": "space",
                        "url": signed_url,
                    }
                }

        class FakeSession:
            def post(self, *_args, **_kwargs):
                return FakeResponse()

        uploader = VODUploader(DeviceConfig(), session=FakeSession())
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.mp3"
            media.write_bytes(b"audio")
            with self.assertRaises(CapCutUploadError) as raised:
                uploader.upload_file(media)

        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(signed_url, str(raised.exception))

if __name__ == "__main__":
    unittest.main()
