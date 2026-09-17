"""Small P1C boundary adapter for the existing modern Translator chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from canonical_profile import CanonicalLocalizationProfile


class ModernP1CAdapterError(RuntimeError):
    """Modern canonical processing cannot safely continue."""


def _canonical_av_analysis(result: object) -> Mapping[str, object]:
    """Extract the producer's canonical AV document without reshaping it."""
    if isinstance(result, Mapping):
        inner = result.get("av_analysis")
    else:
        inner = getattr(result, "av_analysis", None)
    if not isinstance(inner, Mapping):
        raise ModernP1CAdapterError("AV analysis result has no valid av_analysis document")
    return inner


def _presentation_render_config(profile: CanonicalLocalizationProfile) -> dict[str, object]:
    """Project canonical presentation metadata into the renderer boundary."""

    presentation = dict(profile.subtitle_presentation)
    color = presentation.get("color")
    return {
        "presentation": presentation,
        "presentation_color": color if isinstance(color, str) and color.strip() else None,
    }


_LOCALIZATION_STAGE_PROFILES = {
    "loc-v1": "neutral_faithful",
    "page_goc_tro_bat_on": "vn_native_light_comedy",
    "page_giai_cuu_chuong_lon": "vn_native_light_comedy",
}


def _localization_stage_profile(runtime_config: Mapping[str, object]) -> str:
    profile_id = runtime_config.get("localizationProfile")
    if not isinstance(profile_id, str) or profile_id not in _LOCALIZATION_STAGE_PROFILES:
        raise ModernP1CAdapterError("canonical localization profile is unsupported")
    return _LOCALIZATION_STAGE_PROFILES[profile_id]


def build_modern_stage_bundle(*, runtime_config: Mapping[str, object]) -> dict[str, Callable[..., object]]:
    """Construct the child-owned media graph; providers are resolved at runtime."""
    if not isinstance(runtime_config, Mapping):
        raise ModernP1CAdapterError("canonical runtime configuration is required")
    state: dict[str, object] = {}
    runtime_paths: dict[str, Path] = {}

    def source_stage(source: Path, output_dir: Path, **_: object) -> Sequence[Mapping[str, object]]:
        from av_analyzer import GeminiAVAnalyzer, analyze_media_windows_with_subdivision
        from auto_roi import sparse_whole_frame_discovery
        from dubvi_worker import ffprobe_dimensions, ffprobe_duration
        from localized_masking import apply_localized_masks, build_mask_plan
        from source_reconciler import reconcile_source
        from utterance_builder import build_utterances
        duration = round(ffprobe_duration(source) * 1000)
        state["sourceDurationMs"] = duration
        runtime_paths.update(source=source, output_dir=output_dir)
        analysis = analyze_media_windows_with_subdivision(
            GeminiAVAnalyzer(model=str(runtime_config.get("avModel", "gemini-2.5-flash"))),
            source, duration, diagnostics_path=output_dir / "av_subdivision_diagnostics.json",
        )
        detections = sparse_whole_frame_discovery(source, int(runtime_config.get("ocrDiscoveryFrames", 8)))
        mask_plan = build_mask_plan(detections, ffprobe_dimensions(source))
        runtime_paths["render_source"] = source
        if mask_plan:
            masked_source = output_dir / "masked_source.mkv"
            apply_localized_masks(source, masked_source, mask_plan)
            runtime_paths["render_source"] = masked_source
        reconstructed = reconcile_source([], _canonical_av_analysis(analysis))
        return build_utterances(reconstructed["dialogue"], [])

    def localization_stage(records: Sequence[Mapping[str, object]], **_: object) -> Sequence[Mapping[str, object]]:
        from cp2_gemini_adapter import live_cp2_gemini_adapter
        from scene_memory import build_scene_memory, build_terminology_evidence
        from vn_localizer import localize
        memory = build_scene_memory({"terminologyEvidence": build_terminology_evidence(records), "summary": "", "tone": "", "register": ""})
        adapter = live_cp2_gemini_adapter(model=str(runtime_config.get("localizationModel", "gemini-2.5-flash")))
        profile_name = _localization_stage_profile(runtime_config)
        localized = localize(records, scene_memory=memory, request=adapter, profile=profile_name)
        return [{**item, "semanticStartMs": int(item["startMs"]), "semanticEndMs": int(item["endMs"]), "approvedNativeVi": item["nativeVi"], "deliveryNativeVi": item["nativeVi"], "sourcePositions": list(item.get("positions") or [])} for item in localized["accepted"]]

    def tts_stage(records: Sequence[Mapping[str, object]], output_dir: Path, **_: object) -> Mapping[str, object]:
        from dubvi_worker import capcut_synthesize, fit_voice, ffprobe_duration
        from tts_timing import run_tts_timing
        alignment = {"dialogue": [dict(item, ttsEligible=True) for item in records], "excluded": []}
        return run_tts_timing(alignment, output_dir, voice=str(runtime_config["voice"]["name"]), synthesizer=capcut_synthesize, measure_duration=lambda p: round(ffprobe_duration(p) * 1000), fit_audio=fit_voice)

    def delivery_stage(records: Sequence[Mapping[str, object]], *, voice_events: Sequence[Mapping[str, object]], **_: object) -> Sequence[Mapping[str, object]]:
        from delivery_planner import plan_delivery
        measured = {str(item["utteranceId"]): item for item in voice_events}
        enriched = [{**item, "ttsMeasuredMs": int(measured[str(item["utteranceId"])].get("fittedDurationMs", 0))} for item in records]
        return list(plan_delivery(enriched)["plans"])

    def timeline_stage(voices, cues, plans, **_: object):
        from adaptive_renderer import build_adaptive_timeline
        timeline = build_adaptive_timeline(voices, cues, plans, source_duration_ms=int(state["sourceDurationMs"]))
        state["timeline"] = timeline
        return timeline

    def subtitle_stage(cues, *, profile: CanonicalLocalizationProfile, **_: object):
        from adaptive_renderer import build_adaptive_ffmpeg_command, render_adaptive
        from subtitle_layout import write_ass_subtitles
        timeline = state.get("timeline")
        if not isinstance(timeline, Mapping):
            raise ModernP1CAdapterError("adaptive timeline is missing at render seam")
        ass_path = runtime_paths["output_dir"] / "subtitles.ass"
        presentation_config = _presentation_render_config(profile)
        result = write_ass_subtitles(
            cues,
            ass_path,
            presentation_color=presentation_config["presentation_color"],
        )
        result["presentation"] = presentation_config["presentation"]
        voice_inputs = [
            (runtime_paths["output_dir"] / Path(str(event["audioPath"])), int(event["scheduledStartMs"]))
            for event in timeline["voiceEvents"]
        ]
        output = runtime_paths["output_dir"] / "translated.mp4"
        command = build_adaptive_ffmpeg_command(runtime_paths["render_source"], output, ass_path, voice_inputs, timeline=timeline, width=720, height=1280)
        render_adaptive(command)
        result["outputPath"] = str(output)
        return result

    return {"source": source_stage, "localization": localization_stage, "tts": tts_stage, "delivery": delivery_stage, "timeline": timeline_stage, "subtitle": subtitle_stage}


def run_modern_p1c(
    source: Path,
    output_dir: Path,
    identity: Mapping[str, object],
    profile: CanonicalLocalizationProfile,
    *,
    source_stage: Callable[..., Sequence[Mapping[str, object]]],
    localization_stage: Callable[..., Sequence[Mapping[str, object]]],
    tts_stage: Callable[..., Mapping[str, object]],
    delivery_stage: Callable[..., Sequence[Mapping[str, object]]],
    timeline_stage: Callable[..., Mapping[str, object]],
    subtitle_stage: Callable[..., Mapping[str, object]],
) -> dict[str, object]:
    """Compose modern stages while carrying identity and canonical utterance IDs."""

    if not isinstance(identity, Mapping) or not identity.get("translator_job_id") or not identity.get("handoff_id"):
        raise ModernP1CAdapterError("modern P1C identity is incomplete")
    records = list(source_stage(source, output_dir, profile=profile, identity=dict(identity)))
    localized = list(localization_stage(records, profile=profile, identity=dict(identity)))
    tts = dict(tts_stage(localized, output_dir, profile=profile, identity=dict(identity)))
    voices = tts.get("voiceEvents")
    if not isinstance(voices, list):
        raise ModernP1CAdapterError("modern TTS stage returned no voiceEvents")
    plans = list(delivery_stage(localized, voice_events=voices, profile=profile, identity=dict(identity)))
    cues = _subtitle_cues(voices)
    timeline = dict(timeline_stage(voices, cues, plans, profile=profile, identity=dict(identity)))
    subtitles = dict(subtitle_stage(timeline.get("subtitleCues", cues), profile=profile, identity=dict(identity)))
    return {
        "identity": dict(identity),
        "profile": profile.profile_id,
        "records": localized,
        "tts": tts,
        "delivery": plans,
        "timeline": timeline,
        "subtitles": subtitles,
        "outputPath": subtitles.get("outputPath"),
    }


def _subtitle_cues(voice_events: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    cues: list[dict[str, object]] = []
    for event in voice_events:
        identifier = event.get("utteranceId")
        if not identifier or event.get("reviewRequired"):
            continue
        cues.append({
            "voiceEventId": str(identifier),
            "utteranceId": str(identifier),
            "startMs": int(event.get("baseStartMs", event.get("effectiveStartMs", 0))),
            "endMs": int(event.get("baseEndMs", event.get("effectiveEndMs", 0))),
            "translatedTextVi": event.get("deliveryNativeVi", event.get("nativeViApproved")),
            "sourcePositions": list(event.get("sourcePositions") or []),
        })
    return cues
