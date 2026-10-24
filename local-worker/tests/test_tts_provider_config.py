from __future__ import annotations

import os
import json
import io
import contextlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dubvi_worker
from dubvi_worker import Settings, synthesize
from modern_p1c_adapter import build_modern_stage_bundle
from p1c_processing import build_canonical_runtime_config


class TtsProviderConfigTests(unittest.TestCase):
    def test_verify_fails_for_missing_selected_model_without_printing_readiness(self) -> None:
        settings = SimpleNamespace(
            media_dir=Path("media"),
            output_dir=Path("output"),
            model_dir=Path("models"),
            log_dir=Path("logs"),
            device="cpu",
            whisper_model="small",
        )
        secret = "VERIFY_SECRET_MUST_NOT_LEAK"
        stdout = io.StringIO()
        with patch("dubvi_worker.require_binary", return_value=None), patch(
            "dubvi_worker.ensure_whisper_model",
            side_effect=RuntimeError(f"small is missing; token={secret}"),
        ), contextlib.redirect_stdout(stdout):
            with self.assertRaisesRegex(RuntimeError, "small") as raised:
                dubvi_worker.check_environment(settings)

        self.assertNotIn("Whisper device/model", stdout.getvalue())
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn(secret, stdout.getvalue())

    def test_all_canonical_profiles_default_to_declared_tts_provider(self) -> None:
        expected = {
            "loc-v1": "capcut",
            "page_goc_tro_bat_on": "capcut",
            "page_giai_cuu_chuong_lon": "capcut",
        }
        for profile_id, provider in expected.items():
            document = {
                "channel_slug": "demo-channel",
                "target_platform": "tiktok",
                "localization_profile": profile_id,
            }
            with self.subTest(profile_id=profile_id), patch.dict(os.environ, {}, clear=True):
                runtime = build_canonical_runtime_config(document)
            self.assertEqual(provider, runtime["ttsProvider"])

    def test_page1_defaults_to_selected_resolvable_capcut_voice(self) -> None:
        from capcut_tts_api.client import CapCutClient

        document = {
            "channel_slug": "demo-channel",
            "target_platform": "tiktok",
            "localization_profile": "page_goc_tro_bat_on",
        }
        with patch.dict(os.environ, {}, clear=True):
            runtime = build_canonical_runtime_config(document)

        self.assertEqual("capcut", runtime["ttsProvider"])
        self.assertEqual("BV075_streaming", runtime["voice"]["name"])
        self.assertEqual(
            ("BV075_streaming", "7102355803792740865"),
            CapCutClient().resolve_voice(voice=runtime["voice"]["name"]),
        )

    def _environment(self, root: Path, **overrides: str) -> dict[str, str]:
        values = {
            "USERPROFILE": str(root),
            "HOME": str(root),
            "DUBVI_MEDIA_DIR": str(root / "media"),
            "DUBVI_OUTPUT_DIR": str(root / "output"),
            "DUBVI_MODEL_DIR": str(root / "models"),
            "DUBVI_LOG_DIR": str(root / "logs"),
        }
        values.update(overrides)
        return values

    def test_settings_default_to_cpu_piper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ, self._environment(root), clear=True
            ):
                settings = Settings.from_env()

        self.assertEqual("piper", settings.tts_provider)
        self.assertEqual("fast", settings.piper_speed)
        self.assertTrue(str(settings.piper_model).endswith("piper-vietnamese\\pretrained_vi.onnx"))

    def test_explicit_capcut_provider_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ,
                self._environment(root, DUBVI_TTS_PROVIDER="capcut"),
                clear=True,
            ):
                self.assertEqual("capcut", Settings.from_env().tts_provider)

    def test_invalid_provider_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ,
                self._environment(root, DUBVI_TTS_PROVIDER="remote"),
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_invalid_piper_speed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("dubvi_worker.load_dotenv", return_value=False), patch.dict(
                os.environ,
                self._environment(root, DUBVI_PIPER_SPEED="turbo"),
                clear=True,
            ):
                with self.assertRaises(ValueError):
                    Settings.from_env()

    def test_runtime_config_uses_explicit_provider_without_changing_persona(self) -> None:
        document = {
            "channel_slug": "demo-channel",
            "target_platform": "tiktok",
            "localization_profile": "page_goc_tro_bat_on",
        }
        with patch.dict(os.environ, {"DUBVI_TTS_PROVIDER": "piper"}, clear=False):
            runtime = build_canonical_runtime_config(document)

        self.assertEqual("piper", runtime["ttsProvider"])
        self.assertEqual(
            "sarcastic_dramatic_roommate_landlord_student_housing",
            runtime["persona"],
        )

    def test_provider_is_authoritative_even_when_voice_name_looks_like_another_provider(self) -> None:
        settings = SimpleNamespace(tts_provider="piper", capcut_voice="BV421_vivn_streaming")
        with patch("dubvi_worker.piper_synthesize") as piper, patch("dubvi_worker.capcut_synthesize") as capcut, patch("dubvi_worker.edge_synthesize", new=Mock()) as edge:
            synthesize("Xin chào", "BV421_vivn_streaming", Path("out.mp3"), settings)
        piper.assert_called_once()
        edge.assert_not_called()
        capcut.assert_not_called()

    def test_capcut_failure_does_not_fallback_to_edge(self) -> None:
        settings = SimpleNamespace(tts_provider="capcut", capcut_voice="BV421_vivn_streaming")
        with patch("dubvi_worker.validate_capcut_voice"), patch(
            "dubvi_worker.capcut_synthesize", side_effect=RuntimeError("capcut failed")
        ), patch("dubvi_worker.asyncio.run") as run:
            with self.assertRaisesRegex(RuntimeError, "capcut failed"):
                synthesize("Xin chào", "BV421_vivn_streaming", Path("out.mp3"), settings)
        run.assert_not_called()

    def test_unresolved_capcut_voice_fails_before_first_synthesis_call(self) -> None:
        settings = SimpleNamespace(tts_provider="capcut", capcut_voice="missing-capcut-voice")
        with patch("dubvi_worker.validate_capcut_voice", side_effect=ValueError("CapCut voice is not resolvable")), patch(
            "dubvi_worker.capcut_synthesize"
        ) as capcut:
            with self.assertRaisesRegex(ValueError, "not resolvable"):
                synthesize("Xin chào", "missing-capcut-voice", Path("out.mp3"), settings)
        capcut.assert_not_called()

    def test_modern_tts_rejects_unresolved_capcut_voice_before_timing_or_synthesis(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "loc-v1",
            "ttsProvider": "capcut",
            "voice": {"name": "missing-capcut-voice"},
        })
        settings = SimpleNamespace(tts_provider="capcut", capcut_voice="missing-capcut-voice")
        with tempfile.TemporaryDirectory() as directory:
            with patch("dubvi_worker.Settings.from_env", return_value=settings), patch(
                "dubvi_worker.validate_capcut_voice", side_effect=ValueError("CapCut voice is not resolvable")
            ), patch("tts_timing.run_tts_timing") as timing, patch("dubvi_worker.capcut_synthesize") as capcut:
                with self.assertRaisesRegex(ValueError, "not resolvable"):
                    stages["tts"]([], Path(directory))
        timing.assert_not_called()
        capcut.assert_not_called()

    def test_piper_manifest_records_actual_provider_and_model_identity(self) -> None:
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "loc-v1",
            "ttsProvider": "piper",
            "voice": {"name": "BV421_vivn_streaming"},
        })
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            settings = SimpleNamespace(
                tts_provider="piper",
                capcut_voice="BV421_vivn_streaming",
                piper_model=output / "models" / "pretrained_vi.onnx",
                piper_speed="fast",
            )
            with patch("dubvi_worker.Settings.from_env", return_value=settings):
                stages["tts"]([], output)
            manifest = json.loads((output / "tts_manifest_v4.json").read_text(encoding="utf-8"))

        self.assertEqual("piper", manifest["provider"])
        self.assertEqual("piper", manifest["resourceIdentity"]["provider"])
        self.assertEqual(str(settings.piper_model), manifest["resourceIdentity"]["model"])
        self.assertNotEqual("BV421_vivn_streaming", manifest["voice"])

    def test_edge_rejects_a_capcut_voice_instead_of_substituting_one(self) -> None:
        settings = SimpleNamespace(tts_provider="edge", capcut_voice="BV421_vivn_streaming")
        with patch("dubvi_worker.asyncio.run") as run:
            with self.assertRaises(ValueError):
                synthesize("Xin chào", "BV421_vivn_streaming", Path("out.mp3"), settings)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
