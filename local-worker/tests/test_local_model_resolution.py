from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import dubvi_worker
from modern_p1c_adapter import build_modern_stage_bundle


class _DownloadTrap:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.calls.append(args)
        raise AssertionError("download/network helper must not be called")


def _offline_helpers(trap: _DownloadTrap) -> types.ModuleType:
    helpers = types.ModuleType("huggingface_hub")
    helpers.snapshot_download = trap
    helpers.hf_hub_download = trap
    return helpers


class LocalModelResolutionTests(unittest.TestCase):
    def test_page1_missing_selected_small_model_fails_before_audio_extraction(self) -> None:
        trap = _DownloadTrap()

        class WhisperModel:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise OSError("small is not present in the local cache")

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        stages = build_modern_stage_bundle(runtime_config={
            "localizationProfile": "page_goc_tro_bat_on",
            "sourceEvidenceMode": "local_asr_text_only",
            "page1MaskMode": "none",
            "whisperModel": "small",
        })
        with patch.dict(
            sys.modules,
            {"faster_whisper": faster_whisper, "huggingface_hub": _offline_helpers(trap)},
        ), patch.object(dubvi_worker, "_whisper_model", None), patch(
            "dubvi_worker.ffprobe_duration", return_value=1.0
        ) as probe, patch("dubvi_worker.extract_audio") as extract:
            with self.assertRaisesRegex(RuntimeError, r"small.*local cache"):
                with tempfile.TemporaryDirectory() as directory:
                    stages["source"](Path("source.mp4"), Path(directory))

        probe.assert_called_once()
        extract.assert_not_called()
        self.assertEqual([], trap.calls)

    def test_selected_small_model_missing_locally_fails_without_network(self) -> None:
        trap = _DownloadTrap()

        class WhisperModel:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise OSError("small is not present in the local cache")

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(
            sys.modules,
            {"faster_whisper": faster_whisper, "huggingface_hub": _offline_helpers(trap)},
        ), patch.object(dubvi_worker, "_whisper_model", None):
            with self.assertRaisesRegex(RuntimeError, r"small.*local cache"):
                dubvi_worker.transcribe(Path("audio.wav"), "small", "cpu")

        self.assertEqual([], trap.calls)

    def test_available_selected_small_model_proceeds_offline(self) -> None:
        class Part:
            start = 0.25
            end = 1.0
            text = "  原文  "
            avg_logprob = -0.1

        class WhisperModel:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                self.model_name = model_name
                self.kwargs = kwargs

            def transcribe(self, *_args: object, **_kwargs: object):
                return iter([Part()]), None

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}), patch.object(
            dubvi_worker, "_whisper_model", None
        ):
            rows = dubvi_worker.transcribe(Path("audio.wav"), "small", "cpu")

        self.assertEqual("原文", rows[0]["asrTextZh"])
        self.assertEqual(250, rows[0]["startMs"])

    def test_verify_loads_selected_model_from_effective_hf_home_offline(self) -> None:
        loaded: list[dict[str, object]] = []

        class WhisperModel:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                loaded.append({"model_name": model_name, **kwargs})

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "USERPROFILE": str(root),
                "HOME": str(root),
                "DUBVI_MEDIA_DIR": str(root / "media"),
                "DUBVI_OUTPUT_DIR": str(root / "output"),
                "DUBVI_MODEL_DIR": str(root / "models"),
                "DUBVI_LOG_DIR": str(root / "logs"),
                "DUBVI_WHISPER_MODEL": "small",
                "DUBVI_WORKER_TOKEN": "VERIFY_SECRET_MUST_NOT_LEAK",
            }
            with patch.dict(os.environ, environment, clear=True), patch(
                "dubvi_worker.load_dotenv", return_value=False
            ), patch.dict(sys.modules, {"faster_whisper": faster_whisper}), patch.object(
                dubvi_worker, "_whisper_model", None
            ), patch.object(dubvi_worker, "_whisper_model_key", None), patch(
                "dubvi_worker.require_binary", return_value=None
            ):
                settings = dubvi_worker.Settings.from_env()
                with patch("builtins.print") as printed:
                    dubvi_worker.check_environment(settings)
                    effective_hf_home = os.environ["HF_HOME"]
                    diagnostic = " ".join(str(call.args[0]) for call in printed.call_args_list)

            self.assertEqual("small", loaded[0]["model_name"])
            self.assertTrue(loaded[0]["local_files_only"])
            self.assertEqual(str(root / "models" / "huggingface"), effective_hf_home)
            self.assertIn(str(root / "models" / "huggingface"), diagnostic)
            self.assertNotIn("VERIFY_SECRET_MUST_NOT_LEAK", diagnostic)

    def test_changing_hf_home_reloads_selected_model(self) -> None:
        loaded: list[str | None] = []

        class WhisperModel:
            def __init__(self, _model_name: str, **_kwargs: object) -> None:
                loaded.append(os.environ.get("HF_HOME"))

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}), patch.object(
            dubvi_worker, "_whisper_model", None
        ), patch.object(dubvi_worker, "_whisper_model_key", None):
            with patch.dict(os.environ, {"HF_HOME": "cache-a"}):
                dubvi_worker.ensure_whisper_model("small", "cpu")
            with patch.dict(os.environ, {"HF_HOME": "cache-b"}):
                dubvi_worker.ensure_whisper_model("small", "cpu")
        self.assertEqual(["cache-a", "cache-b"], loaded)

    def test_changing_selected_model_does_not_reuse_previous_model(self) -> None:
        loaded: list[str] = []

        class WhisperModel:
            def __init__(self, model_name: str, **_kwargs: object) -> None:
                loaded.append(model_name)

            def transcribe(self, *_args: object, **_kwargs: object):
                return iter(()), None

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}), patch.object(
            dubvi_worker, "_whisper_model", None
        ):
            dubvi_worker.transcribe(Path("audio.wav"), "small", "cpu")
            dubvi_worker.transcribe(Path("audio.wav"), "tiny", "cpu")

        self.assertEqual(["small", "tiny"], loaded)

    def test_repository_ignores_machine_local_models_directory(self) -> None:
        gitignore = Path(__file__).resolve().parents[2] / ".gitignore"
        rules = gitignore.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(rule.strip().rstrip("/") == "models" for rule in rules))

    def test_whisper_constructor_is_local_only(self) -> None:
        calls: list[dict[str, object]] = []

        class WhisperModel:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                calls.append({"model_name": model_name, **kwargs})

            def transcribe(self, *_args: object, **_kwargs: object):
                return ([], None)

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}), patch.object(
            dubvi_worker, "_whisper_model", None
        ):
            dubvi_worker.transcribe(Path("audio.wav"), "small", "cpu")

        self.assertEqual("small", calls[0]["model_name"])
        self.assertTrue(calls[0]["local_files_only"])

    def test_legacy_translation_constructors_are_local_only(self) -> None:
        tokenizer_calls: list[dict[str, object]] = []
        model_calls: list[dict[str, object]] = []

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, model_name: str, **kwargs: object) -> "Tokenizer":
                tokenizer_calls.append({"model_name": model_name, **kwargs})
                return cls()

            def __call__(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                return {}

            def batch_decode(self, *_args: object, **_kwargs: object) -> list[str]:
                return ["Xin chào"]

        class Seq2SeqModel:
            @classmethod
            def from_pretrained(cls, model_name: str, **kwargs: object) -> "Seq2SeqModel":
                model_calls.append({"model_name": model_name, **kwargs})
                return cls()

            def generate(self, **_kwargs: object) -> list[object]:
                return [object()]

        argos = types.ModuleType("argostranslate")
        argos.__path__ = []
        argos_translate = types.ModuleType("argostranslate.translate")
        argos_translate.translate = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no Argos model"))
        argos.translate = argos_translate
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Tokenizer
        transformers.AutoModelForSeq2SeqLM = Seq2SeqModel
        with patch.dict(
            sys.modules,
            {
                "argostranslate": argos,
                "argostranslate.translate": argos_translate,
                "transformers": transformers,
            },
        ), patch.object(dubvi_worker, "_nllb_components", None):
            translated, _ = dubvi_worker.translate_to_vietnamese("你好", 2000)

        self.assertEqual("Xin chào", translated)
        self.assertTrue(tokenizer_calls[0]["local_files_only"])
        self.assertTrue(model_calls[0]["local_files_only"])

    def test_missing_whisper_cache_fails_without_download_and_without_secret(self) -> None:
        model_name = "local-whisper-missing"
        secret = "SUPER_SECRET_TOKEN"
        trap = _DownloadTrap()

        class WhisperModel:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                raise OSError(f"cache missing for {model_name}; token={secret}")

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = WhisperModel
        with patch.dict(
            sys.modules,
            {"faster_whisper": faster_whisper, "huggingface_hub": _offline_helpers(trap)},
        ), patch.object(dubvi_worker, "_whisper_model", None):
            with self.assertRaises(RuntimeError) as raised:
                dubvi_worker.transcribe(Path("audio.wav"), model_name, "cpu")

        message = str(raised.exception)
        self.assertIn(model_name, message)
        self.assertRegex(message.lower(), r"local|cache|offline")
        self.assertNotIn(secret, message)
        self.assertEqual([], trap.calls)

    def test_missing_translation_cache_fails_without_download_and_without_secret(self) -> None:
        model_name = "local-translation-missing"
        secret = "SUPER_SECRET_TOKEN"
        trap = _DownloadTrap()

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Tokenizer":
                raise OSError(f"cache missing for {model_name}; token={secret}")

        class Seq2SeqModel:
            @classmethod
            def from_pretrained(cls, *_args: object, **_kwargs: object) -> "Seq2SeqModel":
                raise AssertionError("model constructor should not run after tokenizer cache failure")

        argos = types.ModuleType("argostranslate")
        argos.__path__ = []
        argos_translate = types.ModuleType("argostranslate.translate")
        argos_translate.translate = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no Argos model"))
        argos.translate = argos_translate
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Tokenizer
        transformers.AutoModelForSeq2SeqLM = Seq2SeqModel
        with patch.dict(
            sys.modules,
            {
                "argostranslate": argos,
                "argostranslate.translate": argos_translate,
                "transformers": transformers,
                "huggingface_hub": _offline_helpers(trap),
            },
        ), patch.object(dubvi_worker, "_nllb_components", None), patch.dict(
            __import__("os").environ, {"DUBVI_MT_MODEL": model_name}
        ):
            with self.assertRaises(RuntimeError) as raised:
                dubvi_worker.translate_to_vietnamese("你好", 2000)

        message = str(raised.exception)
        self.assertIn(model_name, message)
        self.assertRegex(message.lower(), r"local|cache|offline")
        self.assertNotIn(secret, message)
        self.assertEqual([], trap.calls)


if __name__ == "__main__":
    unittest.main()
