"""CP3A semantic voice-event timing and subtitle alignment."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from cue_aligner import alignment_from_cp2_artifact, write_alignment_artifacts, _split_for_slots


class TtsTimingError(RuntimeError):
    """The bounded TTS timing stage cannot continue safely."""


MAX_TEMPO = 1.12
_TIMING_POLICY_VERSION = 4
_REPAIR_REMOVABLE_TOKENS = {
    "tớ", "mình", "cậu", "ngươi", "ta", "đi", "nhé", "nha", "nhỉ", "à", "ạ",
    "cái", "đang", "rất", "với", "là", "đó", "một", "đâu", "gì",
}
_ROLEPLAY_TOKENS = {"bổn", "cung", "công", "chúa", "lão", "nô", "tiểu", "chủ"}
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_TOKEN = re.compile(r"[A-Za-zÀ-ỹĐđ]+", re.IGNORECASE)


def run_cp3a(
    cp2_artifact_path: Path,
    output_dir: Path,
    *,
    voice: str = "BV421_vivn_streaming",
    tts_provider: str = "capcut",
    max_tempo: float = MAX_TEMPO,
    synthesizer: Callable[[str, str, Path], None] | None = None,
    measure_duration: Callable[[Path], int] | None = None,
    fit_audio: Callable[[Path, Path, int, float], tuple[int, bool]] | None = None,
    timing_repair: Callable[[str, dict[str, object]], str | None] | None = None,
) -> dict[str, object]:
    """Run only CP3A from the persisted CP2 artifact; never renders video."""

    artifact = json.loads(cp2_artifact_path.read_text(encoding="utf-8"))
    alignment = alignment_from_cp2_artifact(artifact)
    write_alignment_artifacts(alignment, output_dir)
    return run_tts_timing(
        alignment,
        output_dir,
        voice=voice,
        tts_provider=tts_provider,
        max_tempo=max_tempo,
        synthesizer=synthesizer,
        measure_duration=measure_duration,
        fit_audio=fit_audio,
        timing_repair=timing_repair if timing_repair is not None else _default_timing_repair,
    )


def run_tts_timing(
    alignment: Mapping[str, object],
    output_dir: Path,
    *,
    voice: str = "BV421_vivn_streaming",
    tts_provider: str = "capcut",
    max_tempo: float = MAX_TEMPO,
    synthesizer: Callable[[str, str, Path], None] | None = None,
    measure_duration: Callable[[Path], int] | None = None,
    fit_audio: Callable[[Path, Path, int, float], tuple[int, bool]] | None = None,
    timing_repair: Callable[[str, dict[str, object]], str | None] | None = None,
) -> dict[str, object]:
    """Create one measured voice event per eligible semantic utterance."""

    _validate_options(tts_provider, voice, max_tempo)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer = synthesizer or _default_synthesizer
    measure_duration = measure_duration or _default_measure_duration
    fit_audio = fit_audio or _default_fit_audio
    windows = _delivery_windows(alignment)
    manifest_path = output_dir / "tts_manifest_v4.json"
    old_manifest = _load_manifest(manifest_path)
    old_by_key = {
        str(event.get("cacheKey")): event
        for event in (old_manifest or {}).get("voiceEvents", [])
        if isinstance(event, Mapping) and event.get("cacheKey")
    }
    voice_events: list[dict[str, object]] = []
    event_sources: list[tuple[Mapping[str, object], dict[str, object]]] = []
    provider_calls = 0
    cache_hits = 0
    review_by_id: dict[str, str] = {}

    for utterance, window in windows:
        identifier = str(utterance.get("utteranceId") or "")
        if not identifier:
            raise TtsTimingError("alignment dialogue row has no utterance identity")
        source_positions = list(utterance.get("sourcePositions") or [])
        approved_text = str(utterance.get("nativeVi") or "").strip()
        key = _cache_key(identifier, source_positions, approved_text, voice, tts_provider, window)
        event = _base_event(utterance, window, approved_text, key)
        if bool(utterance.get("reviewRequired")) or not bool(utterance.get("ttsEligible")):
            reason = _review_reason(utterance)
            event.update({"reviewRequired": True, "reviewReason": reason})
            review_by_id.setdefault(identifier, reason)
            voice_events.append(event)
            event_sources.append((utterance, event))
            continue
        cached = old_by_key.get(key)
        if cached is not None and _cached_event_is_valid(output_dir, cached, event):
            event = dict(cached)
            event["cacheHit"] = True
            voice_events.append(event)
            event_sources.append((utterance, event))
            cache_hits += 1
            if bool(event.get("reviewRequired")):
                review_by_id.setdefault(identifier, str(event.get("reviewReason") or "REVIEW_REQUIRED"))
            continue
        if not approved_text:
            event.update({"reviewRequired": True, "reviewReason": "MISSING_TTS_TEXT"})
            review_by_id.setdefault(identifier, "MISSING_TTS_TEXT")
            voice_events.append(event)
            event_sources.append((utterance, event))
            continue
        raw_path = output_dir / "audio" / f"voice-utt-{_safe(identifier)}-raw.mp3"
        fitted_path = output_dir / "audio" / f"voice-utt-{_safe(identifier)}-fitted.mp3"
        try:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            synthesizer(approved_text, voice.strip(), raw_path)
            provider_calls += 1
            _require_audio(raw_path)
            measured_before = int(measure_duration(raw_path))
            if measured_before <= 0:
                raise TtsTimingError("measured TTS duration is not positive")
            available_ms = int(window["availableMs"])
            tempo_required = max(1.0, measured_before / available_ms)
            fitted_duration, needs_review = _fit(fit_audio, raw_path, fitted_path, available_ms, float(max_tempo))
            event.update({
                "measuredMsBefore": measured_before,
                "tempoRequired": round(tempo_required, 6),
                "tempoUsed": round(tempo_required, 6) if not needs_review else None,
                "fittedDurationMs": int(fitted_duration),
                "finalTtsMs": measured_before,
                "audioPath": fitted_path.relative_to(output_dir).as_posix(),
            })
            if needs_review and timing_repair is not None:
                event["timingRepairAttempted"] = True
                context = _repair_context(utterance, window, approved_text, measured_before)
                replacement = timing_repair(approved_text, context)
                accepted, rejection_reason = _assess_repair(approved_text, replacement)
                if accepted:
                    repaired_text = str(replacement).strip()
                    repair_raw = output_dir / "audio" / f"voice-utt-{_safe(identifier)}-repair1-raw.mp3"
                    repair_fitted = output_dir / "audio" / f"voice-utt-{_safe(identifier)}-repair1-fitted.mp3"
                    synthesizer(repaired_text, voice.strip(), repair_raw)
                    provider_calls += 1
                    _require_audio(repair_raw)
                    measured_after = int(measure_duration(repair_raw))
                    if measured_after <= 0:
                        raise TtsTimingError("measured repaired TTS duration is not positive")
                    repaired_duration, repaired_review = _fit(fit_audio, repair_raw, repair_fitted, available_ms, float(max_tempo))
                    repaired_tempo = max(1.0, measured_after / available_ms)
                    event.update({
                        "deliveryNativeVi": repaired_text,
                        "nativeViAfterTiming": repaired_text,
                        "timingRepairAccepted": True,
                        "measuredMsAfter": measured_after,
                        "tempoRequiredAfter": round(repaired_tempo, 6),
                        "tempoUsed": round(repaired_tempo, 6) if not repaired_review else None,
                        "fittedDurationMs": int(repaired_duration),
                        "finalTtsMs": measured_after,
                        "audioPath": repair_fitted.relative_to(output_dir).as_posix(),
                    })
                    needs_review = repaired_review
                else:
                    event["timingRepairRejectionReason"] = rejection_reason
            if needs_review:
                event.update({"reviewRequired": True, "reviewReason": "TTS_TIMING_OVERFLOW"})
                review_by_id.setdefault(identifier, "TTS_TIMING_OVERFLOW")
        except Exception as error:
            event.update({
                "reviewRequired": True,
                "reviewReason": "TTS_FAILED",
                "audioPath": None,
                "providerError": f"{type(error).__name__}: {error}",
            })
            review_by_id.setdefault(identifier, "TTS_FAILED")
        voice_events.append(event)
        event_sources.append((utterance, event))

    subtitle_cues = [cue for utterance, event in event_sources for cue in _subtitle_cues(utterance, str(event.get("deliveryNativeVi") or ""), event)]
    coverage = dict(alignment.get("coverage") or {})
    manifest = {
        "schemaVersion": 1,
        "artifact": "tts_manifest_v4",
        "provider": str(tts_provider).lower(),
        "voice": voice.strip(),
        "timingPolicyVersion": _TIMING_POLICY_VERSION,
        "tempoCeiling": float(max_tempo),
        "alignmentCoverage": coverage,
        "providerCalls": provider_calls,
        "cacheHits": cache_hits,
        "voiceEvents": voice_events,
        "subtitleCues": subtitle_cues,
        "rows": voice_events,
    }
    qc = {
        "schemaVersion": 1,
        "artifact": "tts_qc_v4",
        "provider": str(tts_provider).lower(),
        "voice": voice.strip(),
        "tempoCeiling": float(max_tempo),
        "alignedDialogueRows": len(alignment.get("dialogue", [])),
        "voiceEventRows": len(voice_events),
        "subtitleCueRows": len(subtitle_cues),
        "providerCalls": provider_calls,
        "cacheHits": cache_hits,
        "reviewCount": len(review_by_id),
        "reviewRequired": [{"utteranceId": identifier, "reason": reason} for identifier, reason in review_by_id.items()],
        "rows": voice_events,
    }
    voice_document = {
        "schemaVersion": 1,
        "artifact": "voice_events_v4",
        "provider": str(tts_provider).lower(),
        "voice": voice.strip(),
        "tempoCeiling": float(max_tempo),
        "alignmentCoverage": coverage,
        "voiceEvents": voice_events,
    }
    subtitle_document = {
        "schemaVersion": 1,
        "artifact": "subtitle_alignment_v4",
        "cues": subtitle_cues,
        "excluded": alignment.get("excluded", []),
        "alignmentCoverage": coverage,
    }
    _atomic_json(output_dir / "voice_events_v4.json", voice_document)
    _atomic_json(output_dir / "subtitle_alignment_v4.json", subtitle_document)
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_dir / "tts_qc_v4.json", qc)
    _atomic_text(output_dir / "tts_qc_v4.md", _markdown(qc))
    return manifest


def _validate_options(tts_provider: str, voice: str, max_tempo: float) -> None:
    if str(tts_provider).lower() not in {"piper", "capcut", "edge"}:
        raise TtsTimingError("CP3A provider must be piper, capcut, or edge")
    if not isinstance(voice, str) or not voice.strip():
        raise TtsTimingError("CP3A voice must be nonblank")
    if not isinstance(max_tempo, (int, float)) or max_tempo > MAX_TEMPO or max_tempo <= 0:
        raise TtsTimingError("CP3A tempo ceiling is invalid")


def _delivery_windows(alignment: Mapping[str, object]) -> list[tuple[Mapping[str, object], dict[str, int]]]:
    dialogue = [row for row in alignment.get("dialogue", []) if isinstance(row, Mapping)]
    eligible = [row for row in dialogue if bool(row.get("ttsEligible")) and not bool(row.get("reviewRequired"))]
    eligible.sort(key=lambda row: _time_range(row)[0])
    excluded = [row for row in alignment.get("excluded", []) if isinstance(row, Mapping)]
    result: list[tuple[Mapping[str, object], dict[str, int]]] = []
    for row in dialogue:
        start_ms, end_ms = _time_range(row)
        effective_end = end_ms
        if row in eligible:
            later = [candidate for candidate in eligible if _time_range(candidate)[0] > start_ms]
            if later:
                next_start = _time_range(later[0])[0]
                blockers = [candidate for candidate in [*dialogue, *excluded] if candidate is not row and _blocks_gap(candidate, end_ms, next_start)]
                if next_start > end_ms and not blockers:
                    effective_end = next_start
        window = {
            "baseStartMs": start_ms,
            "baseEndMs": end_ms,
            "effectiveStartMs": start_ms,
            "effectiveEndMs": effective_end,
            "borrowedGapMs": max(0, effective_end - end_ms),
            "availableMs": effective_end - start_ms,
        }
        if window["availableMs"] <= 0:
            raise TtsTimingError("voice delivery window is invalid")
        result.append((row, window))
    _assert_non_overlapping(result)
    return result


def _blocks_gap(row: Mapping[str, object], start_ms: int, end_ms: int) -> bool:
    if str(row.get("type") or "").upper() == "VISUAL_ONLY":
        return False
    try:
        row_start, row_end = _time_range(row)
    except TtsTimingError:
        return False
    return row_start < end_ms and row_end > start_ms and not bool(row.get("ttsEligible"))


def _assert_non_overlapping(windows: Sequence[tuple[Mapping[str, object], Mapping[str, int]]]) -> None:
    ordered = sorted((window for row, window in windows if bool(row.get("ttsEligible")) and not bool(row.get("reviewRequired"))), key=lambda item: item["effectiveStartMs"])
    for previous, current in zip(ordered, ordered[1:]):
        if previous["effectiveEndMs"] > current["effectiveStartMs"]:
            raise TtsTimingError("voice delivery windows overlap")


def _time_range(row: Mapping[str, object]) -> tuple[int, int]:
    start = row.get("startMs", row.get("start_ms"))
    end = row.get("endMs", row.get("end_ms"))
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise TtsTimingError("utterance time range is invalid")
    return start, end


def _base_event(utterance: Mapping[str, object], window: Mapping[str, int], approved_text: str, key: str) -> dict[str, object]:
    source_positions = list(utterance.get("sourcePositions") or [])
    return {
        "timingPolicyVersion": _TIMING_POLICY_VERSION,
        "utteranceId": str(utterance.get("utteranceId")),
        "position": source_positions[0] if source_positions else None,
        "sourcePositions": source_positions,
        "alignedPositions": source_positions,
        "resolvedZh": str(utterance.get("resolvedZh") or ""),
        "intent": utterance.get("intent"),
        "roleplay": utterance.get("roleplay"),
        "register": utterance.get("register"),
        "nativeViApproved": approved_text,
        "nativeViBeforeTiming": approved_text,
        "deliveryNativeVi": approved_text,
        "nativeViAfterTiming": approved_text,
        "baseStartMs": window["baseStartMs"],
        "baseEndMs": window["baseEndMs"],
        "effectiveStartMs": window["effectiveStartMs"],
        "effectiveEndMs": window["effectiveEndMs"],
        "borrowedGapMs": window["borrowedGapMs"],
        "availableMs": window["availableMs"],
        "slotMs": window["availableMs"],
        "measuredMsBefore": None,
        "tempoRequired": None,
        "tempoUsed": None,
        "timingRepairAttempted": False,
        "timingRepairAccepted": False,
        "timingRepairRejectionReason": None,
        "measuredMsAfter": None,
        "tempoRequiredAfter": None,
        "finalTtsMs": None,
        "fittedDurationMs": None,
        "providerError": None,
        "reviewRequired": False,
        "reviewReason": None,
        "audioPath": None,
        "cacheKey": key,
        "cacheHit": False,
    }


def _subtitle_cues(utterance: Mapping[str, object], delivery_text: str, event: Mapping[str, object]) -> list[dict[str, object]]:
    source_cues = utterance.get("alignedCues")
    if not isinstance(source_cues, list):
        raise TtsTimingError("alignment dialogue row has no subtitle cues")
    approved_text = str(utterance.get("nativeVi") or "").strip()
    original_cues = [cue for cue in source_cues if isinstance(cue, Mapping)]
    if not original_cues:
        return []
    original_texts = [str(cue.get("translatedTextVi") or "") for cue in original_cues]
    texts = original_texts if delivery_text == approved_text else _split_to_count(delivery_text, len(original_texts))
    result: list[dict[str, object]] = []
    for cue, text in zip(original_cues, texts):
        result.append({
            "position": cue.get("position"),
            "startMs": cue.get("startMs"),
            "endMs": cue.get("endMs"),
            "translatedTextVi": text,
            "approvedNativeVi": approved_text,
            "deliveryNativeVi": delivery_text,
            "utteranceId": event["utteranceId"],
            "voiceEventId": event["utteranceId"],
            "sourcePositions": list(cue.get("sourcePositions") or []),
        })
    return result


def _split_to_count(text: str, count: int) -> list[str]:
    chunks = _split_for_slots(text, count)
    while len(chunks) < count:
        index = max(range(len(chunks)), key=lambda item: len(chunks[item].split()))
        words = chunks[index].split()
        if len(words) < 2:
            chunks.insert(index + 1, "")
            continue
        cut = max(1, len(words) // 2)
        chunks[index:index + 1] = [" ".join(words[:cut]), " ".join(words[cut:])]
    return chunks[:count]


def _repair_context(utterance: Mapping[str, object], window: Mapping[str, int], text: str, measured_ms: int) -> dict[str, object]:
    return {
        "utteranceId": str(utterance.get("utteranceId")),
        "position": (list(utterance.get("sourcePositions") or []) or [None])[0],
        "sourcePositions": list(utterance.get("sourcePositions") or []),
        "resolvedZh": str(utterance.get("resolvedZh") or ""),
        "approvedNativeVi": text,
        "startMs": window["effectiveStartMs"],
        "endMs": window["effectiveEndMs"],
        "baseStartMs": window["baseStartMs"],
        "baseEndMs": window["baseEndMs"],
        "effectiveEndMs": window["effectiveEndMs"],
        "measuredDurationMs": measured_ms,
        "availableDurationMs": window["availableMs"],
        "requiredReductionMs": max(0, measured_ms - window["availableMs"]),
        "intent": utterance.get("intent"),
        "roleplay": utterance.get("roleplay"),
        "register": utterance.get("register"),
        "style": utterance.get("style"),
        "reviewFlags": list(utterance.get("reviewFlags") or []),
        "neighboringVietnamese": [],
    }


def _assess_repair(original: str, replacement: object) -> tuple[bool, str]:
    if not isinstance(replacement, str) or not replacement.strip():
        return False, "REPAIR_NO_CANDIDATE"
    candidate = replacement.strip()
    if candidate == original:
        return False, "REPAIR_NOT_SHORTER"
    if len(candidate) >= len(original):
        return False, "REPAIR_NOT_SHORTER"
    if re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", candidate):
        return False, "REPAIR_CONTAINS_CJK"
    original_tokens = set(_TOKEN.findall(original.lower()))
    candidate_tokens = set(_TOKEN.findall(candidate.lower()))
    if _ROLEPLAY_TOKENS.intersection(original_tokens - candidate_tokens):
        return False, "REPAIR_DROPS_ROLEPLAY"
    missing_tokens = original_tokens - candidate_tokens
    if not original_tokens or not missing_tokens.issubset(_REPAIR_REMOVABLE_TOKENS):
        return False, "REPAIR_DROPS_SEMANTIC_TOKEN"
    return True, "ACCEPTED"


def _default_timing_repair(text: str, context: dict[str, object]) -> str | None:
    from llm_translator import refine_for_tts_overflow

    segment = {
        "position": context["position"],
        "startMs": context["startMs"],
        "endMs": context["endMs"],
        "sourceTextZh": context["resolvedZh"],
        "translatedTextVi": text,
    }
    return refine_for_tts_overflow(
        segment,
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        measured_ms=int(context["measuredDurationMs"]),
        previous=[],
        following=[],
        context_card={
            "intent": context.get("intent"),
            "roleplay": context.get("roleplay"),
            "register": context.get("register"),
        },
        glossary=(),
    )


def _default_synthesizer(text: str, voice: str, target: Path) -> None:
    from dubvi_worker import capcut_synthesize

    capcut_synthesize(text, voice, target)


def _default_measure_duration(path: Path) -> int:
    from dubvi_worker import ffprobe_duration

    return int(round(ffprobe_duration(path) * 1000))


def _default_fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
    from dubvi_worker import fit_voice

    return fit_voice(raw, fitted, slot_ms, max_tempo)


def _review_reason(utterance: Mapping[str, object]) -> str:
    flags = utterance.get("reviewFlags")
    if isinstance(flags, list) and flags:
        for preferred in ("SOURCE_UNCERTAIN", "TTS_FAILED", "TTS_TIMING_OVERFLOW", "MISSING_TTS_TEXT"):
            if preferred in flags:
                return preferred
        return str(flags[0])
    return "REVIEW_REQUIRED"


def _cache_key(identifier: str, source_positions: Sequence[object], text: str, voice: str, provider: str, window: Mapping[str, int]) -> str:
    payload = json.dumps({
        "utteranceId": identifier,
        "sourcePositions": list(source_positions),
        "deliveryText": text,
        "voice": voice,
        "provider": provider,
        "effectiveStartMs": window["effectiveStartMs"],
        "effectiveEndMs": window["effectiveEndMs"],
        "timingPolicyVersion": _TIMING_POLICY_VERSION,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _cached_event_is_valid(output_dir: Path, cached: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    if (
        cached.get("timingPolicyVersion") != _TIMING_POLICY_VERSION
        or str(cached.get("utteranceId")) != str(expected.get("utteranceId"))
        or list(cached.get("sourcePositions") or []) != list(expected.get("sourcePositions") or [])
        or cached.get("nativeViApproved") != expected.get("nativeViApproved")
        or cached.get("effectiveStartMs") != expected.get("effectiveStartMs")
        or cached.get("effectiveEndMs") != expected.get("effectiveEndMs")
    ):
        return False
    value = cached.get("audioPath")
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        return False
    path = output_dir.joinpath(*value.split("/"))
    try:
        path.resolve(strict=False).relative_to(output_dir.resolve())
    except ValueError:
        return False
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _fit(fit_audio: Callable[[Path, Path, int, float], tuple[int, bool]], raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
    result = fit_audio(raw, fitted, slot_ms, max_tempo)
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], (int, float)) or not isinstance(result[1], bool):
        raise TtsTimingError("TTS fit result is invalid")
    return int(result[0]), result[1]


def _require_audio(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise TtsTimingError("TTS audio is missing or empty")


def _safe(value: object) -> str:
    text = _SAFE_ID.sub("-", str(value)).strip("-.")
    return text or "unknown"


def _load_manifest(path: Path) -> Mapping[str, object] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, Mapping) and document.get("schemaVersion") == 1 and isinstance(document.get("voiceEvents"), list) else None


def _markdown(qc: Mapping[str, object]) -> str:
    lines = [
        "# TTS QC v4",
        "",
        f"Provider: `{qc.get('provider')}`  ",
        f"Voice: `{qc.get('voice')}`  ",
        f"Tempo ceiling: `{qc.get('tempoCeiling')}`  ",
        f"Review utterances: `{qc.get('reviewCount')}`  ",
        "",
        "| ID | Source time | Effective window | Borrowed gap | Approved VI | Delivery VI | Raw TTS ms | Tempo | Repair | Final TTS ms | Review |",
        "|---|---|---|---:|---|---|---:|---:|---|---:|---|",
    ]
    for row in qc.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        source_time = f"{row.get('baseStartMs')}–{row.get('baseEndMs')}"
        effective_window = f"{row.get('effectiveStartMs')}–{row.get('effectiveEndMs')}"
        repair = "accepted" if row.get("timingRepairAccepted") else ("attempted/rejected" if row.get("timingRepairAttempted") else "not attempted")
        review = str(row.get("reviewReason") or "")
        lines.append(
            f"| {_cell(row.get('utteranceId'))} | {source_time} | {effective_window} | {row.get('borrowedGapMs')} | "
            f"{_cell(row.get('nativeViApproved'))} | {_cell(row.get('deliveryNativeVi'))} | {row.get('measuredMsBefore') or '—'} | "
            f"{row.get('tempoRequired') or '—'} | {repair} | {row.get('finalTtsMs') or '—'} | {_cell(review)} |"
        )
    lines.extend(["", "## Review reasons", "", "```json", json.dumps(qc.get("reviewRequired", []), ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def _cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _atomic_json(path: Path, document: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
