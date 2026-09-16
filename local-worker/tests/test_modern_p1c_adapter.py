from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from canonical_profile import resolve_canonical_profile
from modern_p1c_adapter import _localization_stage_profile, run_modern_p1c


class ModernP1CAdapterTests(unittest.TestCase):
    def test_canonical_page_profiles_choose_explicit_localization_stage_profiles(self) -> None:
        self.assertEqual("vn_native_light_comedy", _localization_stage_profile({"localizationProfile": "page_goc_tro_bat_on"}))
        self.assertEqual("vn_native_light_comedy", _localization_stage_profile({"localizationProfile": "page_giai_cuu_chuong_lon"}))
        self.assertEqual("neutral_faithful", _localization_stage_profile({"localizationProfile": "loc-v1"}))

    def test_p1c_fallback_keeps_legacy_graph_reachable_without_modern_stages(self) -> None:
        import p1c_processing

        with patch.object(p1c_processing, "_legacy_process_job") as legacy:
            p1c_processing._modern_process_job(object(), object(), {"id": "job"})
        legacy.assert_called_once()

    def test_injected_stage_bundle_selects_modern_adapter(self) -> None:
        import p1c_processing

        profile = resolve_canonical_profile({"localization_profile": "page_giai_cuu_chuong_lon", "target_platform": "tiktok"})
        stages = {name: (lambda *args, **kwargs: None) for name in ("source", "localization", "tts", "delivery", "timeline", "subtitle")}
        with patch("modern_p1c_adapter.run_modern_p1c", return_value={}) as modern, patch.object(p1c_processing, "_legacy_process_job") as legacy:
            p1c_processing._modern_process_job(
                object(), object(), {
                    "id": "job", "handoffId": "handoff", "sourcePath": "source.mp4",
                    "outputDir": "out", "modernProfile": profile, "modernStages": stages,
                }
            )
        modern.assert_called_once()
        legacy.assert_not_called()

    def test_adapter_composes_identity_safe_modern_stages_without_rendering(self) -> None:
        calls: list[str] = []
        profile = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        identity = {"translator_job_id": "job-1", "handoff_id": "handoff-1"}

        def source_stage(source: Path, output: Path, **kwargs):
            calls.append("source")
            return [{"utteranceId": "u1", "resolvedZh": "你好", "startMs": 0, "endMs": 1000, "visualOnly": False}]

        def localization_stage(records, **kwargs):
            calls.append("localization")
            return [{**records[0], "nativeVi": "Xin chào", "ttsEligible": True, "semanticStartMs": 0, "semanticEndMs": 1000, "sourcePositions": [0]}]

        def tts_stage(records, output, **kwargs):
            calls.append("tts")
            return {"voiceEvents": [{"utteranceId": "u1", "audioPath": "audio/u1.mp3", "baseStartMs": 0, "baseEndMs": 1000, "effectiveStartMs": 0, "effectiveEndMs": 1000, "fittedDurationMs": 500, "nativeViApproved": "Xin chào", "deliveryNativeVi": "Xin chào"}]}

        def delivery_stage(records, **kwargs):
            calls.append("delivery")
            return [{"utteranceId": "u1", "sourceStartMs": 0, "sourceEndMs": 1000, "action": "KEEP", "tempo": 1.0, "borrowedGapMs": 0}]

        def timeline_stage(voices, cues, plans, **kwargs):
            calls.append("timeline")
            return {"voiceEvents": voices, "subtitleCues": cues, "plans": plans}

        def subtitle_stage(cues, **kwargs):
            calls.append("subtitle")
            return {"cues": cues, "maxVisibleLines": 2}

        result = run_modern_p1c(
            Path("source.mp4"), Path("out"), identity, profile,
            source_stage=source_stage, localization_stage=localization_stage,
            tts_stage=tts_stage, delivery_stage=delivery_stage,
            timeline_stage=timeline_stage, subtitle_stage=subtitle_stage,
        )
        self.assertEqual(["source", "localization", "tts", "delivery", "timeline", "subtitle"], calls)
        self.assertEqual(identity, result["identity"])
        self.assertEqual("u1", result["timeline"]["voiceEvents"][0]["utteranceId"])
        self.assertEqual(2, result["subtitles"]["maxVisibleLines"])


if __name__ == "__main__":
    unittest.main()
