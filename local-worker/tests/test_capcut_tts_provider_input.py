import json
import unicodedata
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from capcut_tts_api.client import CapCutClient
import dubvi_worker


class CapCutTtsProviderInputTests(unittest.TestCase):
    def _normalize(self, text: str) -> str:
        normalizer = getattr(dubvi_worker, "normalize_tts_provider_text", None)
        self.assertIsNotNone(normalizer)
        return normalizer(text)  # type: ignore[union-attr]

    def test_failed_page_a_text_uses_nfd_only_for_provider_input(self) -> None:
        canonical = "Thuê nhà"

        provider_text = self._normalize(canonical)

        self.assertEqual("Thuê nhà", provider_text)
        self.assertEqual(canonical, unicodedata.normalize("NFC", provider_text))
        self.assertNotEqual(canonical, provider_text)

    def test_provider_payload_contains_normalized_text(self) -> None:
        canonical = "Thuê nhà"
        provider_text = self._normalize(canonical)

        _, _, body = CapCutClient().build_tts_new_request(
            texts=provider_text,
            voice="BV007_streaming",
        )
        task = json.loads(body)["tasks"][0]
        payload = json.loads(task["payload"])

        self.assertIn(provider_text, payload["ssml"])
        self.assertNotIn(canonical, payload["ssml"])

    def test_canonical_display_text_is_not_mutated(self) -> None:
        canonical = "Thuê nhà"

        provider_text = self._normalize(canonical)

        self.assertEqual("Thuê nhà", canonical)
        self.assertEqual(canonical, unicodedata.normalize("NFC", provider_text))

    def test_ascii_successful_input_is_unchanged(self) -> None:
        text = "Phong tro"

        self.assertEqual(text, self._normalize(text))

    def test_empty_and_nonlexical_inputs_are_not_fabricated(self) -> None:
        self.assertEqual("", self._normalize(""))
        self.assertEqual("...", self._normalize("..."))

    def test_existing_combining_text_is_stable(self) -> None:
        text = "Thuê nhà"

        self.assertEqual(text, self._normalize(text))

    def test_capcut_adapter_sends_normalized_provider_text(self) -> None:
        class FakeClient:
            calls: list[str] = []

            def generate_speech(self, *, texts: str, voice: str, timeout: float) -> dict[str, object]:
                self.calls.append(texts)
                return {"data": {"tasks": [{"payload": json.dumps({"audio_url": "https://example.test/a.mp3"})}]}}

        class FakeResponse:
            content = b"audio" * 512

            def raise_for_status(self) -> None:
                return None

        canonical = "Thuê nhà"
        with TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "voice.mp3"
            fake_client = FakeClient()
            with patch("capcut_tts_api.client.CapCutClient", return_value=fake_client), patch(
                "requests.get", return_value=FakeResponse()
            ):
                dubvi_worker.capcut_synthesize(canonical, "BV007_streaming", target)

        self.assertEqual(["Thuê nhà"], fake_client.calls)


if __name__ == "__main__":
    unittest.main()
