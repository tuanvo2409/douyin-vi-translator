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


def _local_asr_source(
    source: Path,
    output_dir: Path,
    runtime_config: Mapping[str, object],
    state: dict[str, object],
    runtime_paths: dict[str, Path],
) -> Sequence[Mapping[str, object]]:
    """Build canonical source rows from local ASR without entering the AV boundary."""

    import json
    import os

    from dubvi_worker import extract_audio, ffprobe_duration, transcribe
    from source_reconciler import assign_canonical_utterance_ids

    duration = round(ffprobe_duration(source) * 1000)
    state["sourceDurationMs"] = duration
    runtime_paths.update(source=source, output_dir=output_dir, render_source=source)
    audio = output_dir / "source.wav"
    extract_audio(source, audio)
    model = str(runtime_config.get("whisperModel") or os.environ.get("DUBVI_WHISPER_MODEL") or "small")
    device = str(os.environ.get("DUBVI_DEVICE") or "cpu")
    rows = list(transcribe(audio, model, device))
    if not rows:
        raise ModernP1CAdapterError("local ASR produced no source utterances")
    min_confidence = int((runtime_config.get("ocr") or {}).get("minConfidence", 65))
    cues: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        position = row.get("position", index)
        start_ms = row.get("startMs")
        end_ms = row.get("endMs")
        text = str(row.get("asrTextZh") or "").strip()
        confidence = int(row.get("confidence", 0))
        if not isinstance(position, int) or isinstance(position, bool):
            raise ModernP1CAdapterError("local ASR source position is invalid")
        if not isinstance(start_ms, int) or isinstance(start_ms, bool) or not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms <= start_ms:
            raise ModernP1CAdapterError("local ASR source timing is invalid")
        if not text:
            continue
        cue_id = f"asr-{position}"
        high_confidence = confidence >= min_confidence
        flags = [] if high_confidence else ["SOURCE_UNCERTAIN"]
        cues.append({
            "cue_id": cue_id,
            "position": position,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "asr_text_zh": text,
            "ocr_text_zh": "",
            "needs_review": not high_confidence,
        })
        classification = "RESOLVED_DIALOGUE" if high_confidence else "UNRESOLVED_DIALOGUE"
        records.append({
            "startMs": start_ms,
            "endMs": end_ms,
            "spokenZh": text,
            "subtitleZh": "",
            "resolvedZh": text,
            "confidence": "HIGH" if high_confidence else "LOW",
            "source_confidence": "high" if high_confidence else "low",
            "positions": [position],
            "cue_ids": [cue_id],
            "evidence": {"asr": text, "ocr": "", "avSpoken": "", "avSubtitle": ""},
            "decision": "LOCAL_ASR",
            "flags": flags,
            "source_flags": flags,
            "kind": "dialogue",
            "type": "DIALOGUE" if high_confidence else classification,
            "classification": classification,
        })
    if not records:
        raise ModernP1CAdapterError("local ASR produced no usable source utterances")
    canonical = assign_canonical_utterance_ids(records)
    state["sourceCues"] = cues
    state["sourceEvidenceMode"] = "local_asr_text_only"
    (output_dir / "source_evidence.json").write_text(
        json.dumps({"schemaVersion": 1, "mode": "local_asr_text_only", "durationMs": duration, "cues": cues, "records": canonical}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # OCR remains local and is used only for visual masking. Its absence never
    # invents dialogue or blocks the speech demo.
    try:
        from auto_roi import sparse_whole_frame_discovery
        from dubvi_worker import ffprobe_dimensions
        from localized_masking import apply_localized_masks, build_mask_plan

        detections = sparse_whole_frame_discovery(source, int((runtime_config.get("ocr") or {}).get("discoveryFrames", 8)))
        mask_plan = build_mask_plan(detections, ffprobe_dimensions(source)) if detections else []
        if mask_plan:
            masked_source = output_dir / "masked_source.mkv"
            apply_localized_masks(source, masked_source, mask_plan)
            runtime_paths["render_source"] = masked_source
        state["localOcrDetections"] = len(detections)
    except Exception:
        state["localOcrDetections"] = 0
    return canonical


def build_modern_stage_bundle(*, runtime_config: Mapping[str, object]) -> dict[str, Callable[..., object]]:
    """Construct the child-owned media graph; providers are resolved at runtime."""
    if not isinstance(runtime_config, Mapping):
        raise ModernP1CAdapterError("canonical runtime configuration is required")
    state: dict[str, object] = {}
    runtime_paths: dict[str, Path] = {}

    def source_stage(source: Path, output_dir: Path, **_: object) -> Sequence[Mapping[str, object]]:
        from dubvi_worker import ffprobe_dimensions, ffprobe_duration
        from localized_masking import apply_localized_masks, build_mask_plan
        from source_reconciler import reconcile_source
        from utterance_builder import build_utterances
        if runtime_config.get("sourceEvidenceMode") == "local_asr_text_only":
            return _local_asr_source(source, output_dir, runtime_config, state, runtime_paths)
        from av_analyzer import GeminiAVAnalyzer, analyze_media_windows_with_subdivision
        from auto_roi import sparse_whole_frame_discovery
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
        from cue_aligner import build_alignment, write_alignment_artifacts
        from cp2_gemini_adapter import live_cp2_gemini_adapter
        from scene_memory import build_scene_memory, build_terminology_evidence
        from text_translation_fallback import GeminiPrimaryWithGroqTextFallback, GroqTextTranslationAdapter
        from vn_localizer import localize
        memory = build_scene_memory({"terminologyEvidence": build_terminology_evidence(records), "summary": "", "tone": "", "register": ""})
        gemini = live_cp2_gemini_adapter(model=str(runtime_config.get("localizationModel", "gemini-2.5-flash")))
        adapter = GeminiPrimaryWithGroqTextFallback(gemini, GroqTextTranslationAdapter())
        profile_name = _localization_stage_profile(runtime_config)
        text_only_records = [
            {
                "utteranceId": str(item["utteranceId"]),
                "resolvedZh": str(item.get("resolvedZh") or ""),
                "source_confidence": str(item.get("source_confidence") or ""),
                "kind": str(item.get("kind") or "dialogue"),
            }
            for item in records
        ]
        localized = localize(text_only_records, scene_memory=memory, request=adapter, profile=profile_name)
        state["translationProviderPath"] = adapter.last_provider
        alignment = build_alignment(
            records,
            accepted=localized["accepted"],
            review_required=localized["review_required"],
            source_cues=state.get("sourceCues"),
        )
        write_alignment_artifacts(alignment, runtime_paths["output_dir"])
        state["alignment"] = alignment
        return [
            {
                **item,
                "semanticStartMs": int(item["startMs"]),
                "semanticEndMs": int(item["endMs"]),
                "approvedNativeVi": item["nativeVi"],
                "deliveryNativeVi": item["nativeVi"],
                "sourcePositions": list(item.get("sourcePositions") or []),
            }
            for item in alignment["dialogue"]
        ]

    def tts_stage(records: Sequence[Mapping[str, object]], output_dir: Path, **_: object) -> Mapping[str, object]:
        from dubvi_worker import Settings, fit_voice, ffprobe_duration, synthesize
        from tts_timing import run_tts_timing
        settings = Settings.from_env()
        provider = str(runtime_config.get("ttsProvider") or settings.tts_provider).strip().lower()
        settings.tts_provider = provider

        def configured_synthesizer(text: str, voice: str, target: Path) -> None:
            synthesize(text, voice, target, settings)

        alignment = {
            "dialogue": [dict(item) for item in records],
            "excluded": list((state.get("alignment") or {}).get("excluded", [])) if isinstance(state.get("alignment"), Mapping) else [],
        }
        return run_tts_timing(
            alignment,
            output_dir,
            voice=str((runtime_config.get("voice") or {}).get("name") or settings.capcut_voice),
            tts_provider=provider,
            synthesizer=configured_synthesizer,
            measure_duration=lambda p: round(ffprobe_duration(p) * 1000),
            fit_audio=fit_voice,
        )

    def delivery_stage(records: Sequence[Mapping[str, object]], *, voice_events: Sequence[Mapping[str, object]], **_: object) -> Sequence[Mapping[str, object]]:
        from delivery_planner import DeliveryPolicy, plan_delivery
        measured = {str(item["utteranceId"]): item for item in voice_events}
        enriched: list[dict[str, object]] = []
        for item in records:
            identifier = str(item["utteranceId"])
            event = measured.get(identifier)
            if not isinstance(event, Mapping):
                raise ModernP1CAdapterError(f"modern TTS event is missing for {identifier}")
            if not _is_renderable_tts_event(event):
                raise ModernP1CAdapterError(f"modern TTS event is not renderable for {identifier}")
            enriched.append({
                **item,
                "ttsMeasuredMs": int(event["fittedDurationMs"]),
                "audioPath": str(event["audioPath"]),
                "nextSpeechStartMs": int(event["effectiveEndMs"]),
            })
        return list(plan_delivery(enriched, policy=DeliveryPolicy(max_tempo=1.0))["plans"])

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
    _validate_tts_events(localized, voices)
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


def _is_renderable_tts_event(event: Mapping[str, object]) -> bool:
    """Return whether an event has success-only audio metadata."""

    duration = event.get("fittedDurationMs")
    audio_path = event.get("audioPath")
    return (
        event.get("reviewReason") != "TTS_FAILED"
        and isinstance(audio_path, str)
        and bool(audio_path.strip())
        and isinstance(duration, int)
        and not isinstance(duration, bool)
        and duration > 0
    )


def _validate_tts_events(
    records: Sequence[Mapping[str, object]], voice_events: Sequence[object]
) -> None:
    """Reject failed or malformed TTS events before delivery/timeline planning."""

    by_id: dict[str, Mapping[str, object]] = {}
    duplicate_ids: set[str] = set()
    for event in voice_events:
        if not isinstance(event, Mapping):
            raise ModernP1CAdapterError("modern TTS event is not an object")
        identifier = str(event.get("utteranceId") or "")
        if not identifier or identifier in by_id:
            duplicate_ids.add(identifier or "<missing>")
        by_id[identifier] = event
    if duplicate_ids:
        raise ModernP1CAdapterError(
            "modern TTS event identities are duplicated or missing: "
            + ", ".join(sorted(duplicate_ids))
        )

    invalid: list[str] = []
    for record in records:
        identifier = str(record.get("utteranceId") or "")
        event = by_id.get(identifier)
        if event is None:
            invalid.append(f"MISSING_EVENT for {identifier or '<missing>'}")
        elif not _is_renderable_tts_event(event):
            reason = str(event.get("reviewReason") or "INVALID_AUDIO_METADATA")
            invalid.append(f"{reason} for {identifier or '<missing>'}")
    if invalid:
        raise ModernP1CAdapterError(
            "modern TTS stage produced non-renderable events: " + "; ".join(invalid)
        )


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
