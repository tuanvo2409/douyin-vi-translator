"""Bounded CP3A CapCut synthesis, measured fitting, and resume artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Mapping, Sequence

from cue_aligner import alignment_from_cp2_artifact, write_alignment_artifacts


class TtsTimingError(RuntimeError):
    """The bounded TTS timing stage cannot continue safely."""


MAX_TEMPO = 1.12
_TIMING_POLICY_VERSION = 3
_REMOVABLE_REPAIR_TOKENS = {"nhé", "nha", "nhỉ", "à", "ạ"}
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


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
    """Synthesize eligible aligned cues with one bounded repair opportunity."""

    if str(tts_provider).lower() != "capcut":
        raise TtsTimingError("CP3A requires the CapCut provider")
    if not isinstance(voice, str) or not voice.strip():
        raise TtsTimingError("CP3A voice must be nonblank")
    if not isinstance(max_tempo, (int, float)) or max_tempo > MAX_TEMPO or max_tempo <= 0:
        raise TtsTimingError("CP3A tempo ceiling is invalid")
    output_dir.mkdir(parents=True, exist_ok=True)
    synthesizer = synthesizer or _default_synthesizer
    measure_duration = measure_duration or _default_measure_duration
    fit_audio = fit_audio or _default_fit_audio
    manifest_path = output_dir / "tts_manifest_v3.json"
    old_manifest = _load_manifest(manifest_path)
    old_by_key = {
        str(row.get("cacheKey")): row
        for row in (old_manifest or {}).get("rows", [])
        if isinstance(row, Mapping) and row.get("cacheKey")
    }
    rows: list[dict[str, object]] = []
    provider_calls = 0
    cache_hits = 0
    review_by_id: dict[str, str] = {}

    for utterance in alignment.get("dialogue", []):
        if not isinstance(utterance, Mapping):
            raise TtsTimingError("alignment dialogue row is invalid")
        identifier = str(utterance.get("utteranceId") or "")
        cues = utterance.get("alignedCues")
        if not identifier or not isinstance(cues, list) or not cues:
            raise TtsTimingError("alignment dialogue row has no cues")
        accepted_repairs: set[str] = set()
        for cue_index, cue in enumerate(cues):
            if not isinstance(cue, Mapping):
                raise TtsTimingError("alignment cue is invalid")
            text = str(cue.get("translatedTextVi") or "").strip()
            slot_ms = _slot_ms(cue)
            source_positions = list(cue.get("sourcePositions") or [])
            key = _cache_key(identifier, cue.get("position"), source_positions, text, voice, tts_provider, slot_ms, float(max_tempo))
            base = {
                "timingPolicyVersion": _TIMING_POLICY_VERSION,
                "utteranceId": identifier,
                "position": cue.get("position"),
                "alignedPositions": source_positions,
                "resolvedZh": str(utterance.get("resolvedZh") or ""),
                "nativeViBeforeTiming": text,
                "nativeViAfterTiming": text,
                "slotMs": slot_ms,
                "measuredMsBefore": None,
                "tempoRequired": None,
                "tempoUsed": None,
                "timingRepairAttempted": False,
                "measuredMsAfter": None,
                "reviewRequired": False,
                "reviewReason": None,
                "audioPath": None,
                "cacheKey": key,
                "cacheHit": False,
            }
            if bool(utterance.get("reviewRequired")):
                reason = _review_reason(utterance)
                base.update({"reviewRequired": True, "reviewReason": reason})
                review_by_id.setdefault(identifier, reason)
                rows.append(base)
                continue
            cached = old_by_key.get(key)
            if cached is not None and _cached_audio_is_valid(
                output_dir,
                cached,
                utterance_id=identifier,
                position=cue.get("position"),
                source_positions=source_positions,
                text=text,
                slot_ms=slot_ms,
            ):
                reused = dict(cached)
                reused["cacheHit"] = True
                rows.append(reused)
                cache_hits += 1
                if bool(reused.get("reviewRequired")):
                    review_by_id.setdefault(identifier, str(reused.get("reviewReason") or "REVIEW_REQUIRED"))
                continue
            if not text:
                base.update({"reviewRequired": True, "reviewReason": "MISSING_TTS_TEXT"})
                review_by_id.setdefault(identifier, "MISSING_TTS_TEXT")
                rows.append(base)
                continue
            raw_path = output_dir / "audio" / f"utt-{_safe(identifier)}-{cue_index:02d}-raw.mp3"
            fitted_path = output_dir / "audio" / f"utt-{_safe(identifier)}-{cue_index:02d}-fitted.mp3"
            try:
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                synthesizer(text, voice.strip(), raw_path)
                provider_calls += 1
                _require_audio(raw_path)
                measured_before = int(measure_duration(raw_path))
                if measured_before <= 0:
                    raise TtsTimingError("measured TTS duration is not positive")
                tempo_required = max(1.0, measured_before / slot_ms)
                fitted_duration, needs_review = _fit(fit_audio, raw_path, fitted_path, slot_ms, float(max_tempo))
                base.update({
                    "measuredMsBefore": measured_before,
                    "tempoRequired": round(tempo_required, 6),
                    "tempoUsed": round(tempo_required, 6) if not needs_review else None,
                    "audioPath": fitted_path.relative_to(output_dir).as_posix(),
                    "fittedDurationMs": int(fitted_duration),
                })
                if needs_review and timing_repair is not None:
                    base["timingRepairAttempted"] = True
                    context = _repair_context(utterance, cue, text, measured_before, slot_ms)
                    replacement = timing_repair(text, context)
                    if _repair_is_safe(text, replacement, accepted_repairs):
                        repaired_text = replacement.strip()
                        repair_raw = output_dir / "audio" / f"utt-{_safe(identifier)}-{cue_index:02d}-repair1-raw.mp3"
                        repair_fitted = output_dir / "audio" / f"utt-{_safe(identifier)}-{cue_index:02d}-repair1-fitted.mp3"
                        synthesizer(repaired_text, voice.strip(), repair_raw)
                        provider_calls += 1
                        _require_audio(repair_raw)
                        measured_after = int(measure_duration(repair_raw))
                        repaired_duration, repaired_review = _fit(fit_audio, repair_raw, repair_fitted, slot_ms, float(max_tempo))
                        base.update({
                            "nativeViAfterTiming": repaired_text,
                            "measuredMsAfter": measured_after,
                            "tempoUsed": round(max(1.0, measured_after / slot_ms), 6) if not repaired_review else None,
                            "audioPath": repair_fitted.relative_to(output_dir).as_posix(),
                            "fittedDurationMs": int(repaired_duration),
                        })
                        needs_review = repaired_review
                        accepted_repairs.add(repaired_text)
                    else:
                        needs_review = True
                if needs_review:
                    base.update({"reviewRequired": True, "reviewReason": "TTS_TIMING_OVERFLOW"})
                    review_by_id.setdefault(identifier, "TTS_TIMING_OVERFLOW")
                rows.append(base)
            except Exception:
                base.update({"reviewRequired": True, "reviewReason": "TTS_FAILED", "audioPath": None})
                review_by_id.setdefault(identifier, "TTS_FAILED")
                rows.append(base)

    manifest = {
        "schemaVersion": 1,
        "artifact": "tts_manifest_v3",
        "provider": "capcut",
        "voice": voice.strip(),
        "timingPolicyVersion": _TIMING_POLICY_VERSION,
        "tempoCeiling": float(max_tempo),
        "alignmentCoverage": alignment.get("coverage", {}),
        "providerCalls": provider_calls,
        "cacheHits": cache_hits,
        "rows": rows,
    }
    qc = {
        "schemaVersion": 1,
        "artifact": "tts_qc_v3",
        "provider": "capcut",
        "voice": voice.strip(),
        "tempoCeiling": float(max_tempo),
        "alignedDialogueRows": len(alignment.get("dialogue", [])),
        "ttsRows": len(rows),
        "providerCalls": provider_calls,
        "cacheHits": cache_hits,
        "reviewCount": len(review_by_id),
        "reviewRequired": [{"utteranceId": identifier, "reason": reason} for identifier, reason in review_by_id.items()],
        "rows": rows,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_dir / "tts_qc_v3.json", qc)
    return manifest


def _default_synthesizer(text: str, voice: str, target: Path) -> None:
    from dubvi_worker import capcut_synthesize

    capcut_synthesize(text, voice, target)


def _default_measure_duration(path: Path) -> int:
    from dubvi_worker import ffprobe_duration

    return int(round(ffprobe_duration(path) * 1000))


def _default_fit_audio(raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
    from dubvi_worker import fit_voice

    return fit_voice(raw, fitted, slot_ms, max_tempo)


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
        context_card={},
        glossary=(),
    )


def _repair_context(utterance: Mapping[str, object], cue: Mapping[str, object], text: str, measured_ms: int, slot_ms: int) -> dict[str, object]:
    return {
        "utteranceId": str(utterance.get("utteranceId")),
        "position": cue.get("position"),
        "sourcePositions": list(cue.get("sourcePositions") or []),
        "resolvedZh": str(utterance.get("resolvedZh") or ""),
        "approvedNativeVi": text,
        "startMs": cue.get("startMs"),
        "endMs": cue.get("endMs"),
        "measuredDurationMs": measured_ms,
        "availableDurationMs": slot_ms,
        "reviewFlags": list(utterance.get("reviewFlags") or []),
        "neighboringVietnamese": [],
    }


def _fit(fit_audio: Callable[[Path, Path, int, float], tuple[int, bool]], raw: Path, fitted: Path, slot_ms: int, max_tempo: float) -> tuple[int, bool]:
    result = fit_audio(raw, fitted, slot_ms, max_tempo)
    if not isinstance(result, tuple) or len(result) != 2 or not isinstance(result[0], (int, float)) or not isinstance(result[1], bool):
        raise TtsTimingError("TTS fit result is invalid")
    return int(result[0]), result[1]


def _slot_ms(cue: Mapping[str, object]) -> int:
    start, end = cue.get("startMs"), cue.get("endMs")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or end <= start:
        raise TtsTimingError("aligned cue duration is invalid")
    return end - start


def _review_reason(utterance: Mapping[str, object]) -> str:
    flags = utterance.get("reviewFlags")
    if isinstance(flags, list) and flags:
        for preferred in ("SOURCE_UNCERTAIN", "TTS_FAILED", "TTS_TIMING_OVERFLOW", "MISSING_TTS_TEXT"):
            if preferred in flags:
                return preferred
        return str(flags[0])
    return "REVIEW_REQUIRED"


def _cache_key(identifier: str, position: object, source_positions: Sequence[object], text: str, voice: str, provider: str, slot_ms: int, max_tempo: float) -> str:
    payload = json.dumps({"utteranceId": identifier, "position": position, "sourcePositions": list(source_positions), "text": text, "voice": voice, "provider": provider, "slotMs": slot_ms, "maxTempo": max_tempo, "timingPolicyVersion": _TIMING_POLICY_VERSION}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _cached_audio_is_valid(
    output_dir: Path,
    row: Mapping[str, object],
    *,
    utterance_id: str,
    position: object,
    source_positions: Sequence[object],
    text: str,
    slot_ms: int,
) -> bool:
    if (
        row.get("timingPolicyVersion") != _TIMING_POLICY_VERSION
        or
        str(row.get("utteranceId")) != utterance_id
        or row.get("position") != position
        or list(row.get("alignedPositions") or []) != list(source_positions)
        or row.get("nativeViBeforeTiming") != text
        or row.get("slotMs") != slot_ms
    ):
        return False
    value = row.get("audioPath")
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value or ":" in value or any(part in {"", ".", ".."} for part in value.split("/")):
        return False
    path = output_dir.joinpath(*value.split("/"))
    try:
        path.resolve(strict=False).relative_to(output_dir.resolve())
    except ValueError:
        return False
    return path.is_file() and not path.is_symlink() and path.stat().st_size > 0


def _require_audio(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise TtsTimingError("TTS audio is missing or empty")


def _repair_is_safe(original: str, replacement: object, accepted_repairs: set[str]) -> bool:
    if not isinstance(replacement, str) or not replacement.strip() or replacement.strip() == original:
        return False
    candidate = replacement.strip()
    if len(candidate) >= len(original) or re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", candidate):
        return False
    original_tokens = set(re.findall(r"[A-Za-zÀ-ỹĐđ]+", original.lower()))
    candidate_tokens = set(re.findall(r"[A-Za-zÀ-ỹĐđ]+", candidate.lower()))
    missing_tokens = original_tokens - candidate_tokens
    if not original_tokens or not missing_tokens.issubset(_REMOVABLE_REPAIR_TOKENS):
        return False
    return candidate not in accepted_repairs


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
    return document if isinstance(document, Mapping) and document.get("schemaVersion") == 1 and isinstance(document.get("rows"), list) else None


def _atomic_json(path: Path, document: object) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
