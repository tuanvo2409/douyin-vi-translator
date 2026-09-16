"""Small P1C boundary adapter for the existing modern Translator chain."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from canonical_profile import CanonicalLocalizationProfile


class ModernP1CAdapterError(RuntimeError):
    """Modern canonical processing cannot safely continue."""


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
