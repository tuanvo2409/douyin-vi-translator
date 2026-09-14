"""Evidence-first reconciliation of canonical multimodal source evidence."""
from __future__ import annotations

import string
import unicodedata
from typing import Mapping, Sequence

from av_analyzer import validate_av_analysis


_CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
_PUNCTUATION = set(string.punctuation) | set("，。！？、；：‘’“”【】（）《》…—·")


def _overlaps(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return int(left["start_ms"]) < int(right["end_ms"]) and int(right["start_ms"]) < int(left["end_ms"])


def _normalise_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", value)
        if not character.isspace() and character not in _PUNCTUATION
    )


def _texts_agree(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_values = {_normalise_text(left.get(key)) for key in ("spokenZh", "subtitleZh") if left.get(key)}
    right_values = {_normalise_text(right.get(key)) for key in ("spokenZh", "subtitleZh") if right.get(key)}
    left_values.discard("")
    right_values.discard("")
    return bool(left_values & right_values) or any(
        len(short) >= 2 and (short in long or long in short)
        for short in left_values
        for long in right_values
    )


def _near(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return min(abs(int(left["end_ms"]) - int(right["start_ms"])), abs(int(right["end_ms"]) - int(left["start_ms"]))) <= 100


def _continuation(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_text = str(left.get("spokenZh") or left.get("subtitleZh") or "").strip()
    right_text = str(right.get("spokenZh") or right.get("subtitleZh") or "").strip()
    if not left_text or not right_text or len(_normalise_text(right_text)) >= len(_normalise_text(left_text)):
        return False
    return _normalise_text(right_text) in _normalise_text(left_text)


def _fragment_continuation(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    left_text = str(left.get("spokenZh") or left.get("subtitleZh") or "").strip()
    right_text = str(right.get("spokenZh") or right.get("subtitleZh") or "").strip()
    return bool(left_text and right_text and right_text.startswith(("这", "那")) and not right_text.endswith(("。", "！", "？")))


def _as_internal(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "start_ms": int(item["startMs"]),
        "end_ms": int(item["endMs"]),
        "spokenZh": item["spokenZh"],
        "subtitleZh": item["subtitleZh"],
        "confidence": item["confidence"],
        "speakerRef": item["speakerRef"],
        "intent": item["intent"],
        "emotion": item["emotion"],
    }


def _merge_groups(items: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for item in sorted(items, key=lambda value: (int(value["start_ms"]), int(value["end_ms"]))):
        target = None
        for group in groups:
            if any(_overlaps(item, member) for member in group["items"]):
                target = group
                break
            if _near(item, group["items"][-1]) and (
                (
                    _texts_agree(item, group["items"][-1])
                    and (
                        _continuation(group["items"][-1], item)
                        or _normalise_text(str(item.get("spokenZh") or item.get("subtitleZh") or ""))
                        == _normalise_text(str(group["items"][-1].get("spokenZh") or group["items"][-1].get("subtitleZh") or ""))
                    )
                )
                or _fragment_continuation(group["items"][-1], item)
            ):
                target = group
                break
        if target is None:
            groups.append({"items": [dict(item)], "start_ms": int(item["start_ms"]), "end_ms": int(item["end_ms"])})
        else:
            target["items"].append(dict(item))
            target["start_ms"] = min(int(target["start_ms"]), int(item["start_ms"]))
            target["end_ms"] = max(int(target["end_ms"]), int(item["end_ms"]))
    return groups


def _best_text(items: Sequence[Mapping[str, object]], key: str) -> str | None:
    values = [str(item[key]).strip() for item in items if isinstance(item.get(key), str) and str(item[key]).strip()]
    if not values:
        return None
    result = values[0]
    for value in values[1:]:
        previous = {"spokenZh": result, "subtitleZh": result}
        current = {"spokenZh": value, "subtitleZh": value}
        if _normalise_text(value) == _normalise_text(result) or _continuation(previous, current):
            continue
        if _fragment_continuation(previous, current):
            result += "，" + value
            continue
        if len(_normalise_text(value)) > len(_normalise_text(result)):
            result = value
    return result.replace(",", "，").replace("?", "？")


def _join_values(values: Sequence[object]) -> str:
    unique = []
    for value in values:
        text = str(value).strip() if value is not None else ""
        if text and text not in unique:
            unique.append(text)
    return "; ".join(unique)


def _conflict(items: Sequence[Mapping[str, object]]) -> bool:
    return any(_overlaps(left, right) and not _texts_agree(left, right) for index, left in enumerate(items) for right in items[index + 1:])


def _linked_cues(group: Mapping[str, object], cues: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    window = {"start_ms": group["start_ms"], "end_ms": group["end_ms"]}
    return [cue for cue in cues if _overlaps(window, cue)]


def _is_unresolved(group: Mapping[str, object], linked: Sequence[Mapping[str, object]], chosen: str | None) -> bool:
    if not chosen or len(linked) < 2:
        return False
    reviewed = [cue for cue in linked if bool(cue.get("needs_review"))]
    if len(reviewed) < 2:
        return False
    chosen_text = _normalise_text(chosen)
    legacy_text = {
        _normalise_text(cue.get(key))
        for cue in reviewed
        for key in ("asr_text_zh", "ocr_text_zh")
        if cue.get(key)
    }
    beyond_legacy = int(group["end_ms"]) > max(int(cue["end_ms"]) for cue in linked)
    return beyond_legacy and (not any(value and (value in chosen_text or chosen_text in value) for value in legacy_text) if legacy_text else True)


def _dialogue_record(group: Mapping[str, object], linked: Sequence[Mapping[str, object]]) -> dict[str, object]:
    items = group["items"]
    conflict = _conflict(items)
    spoken = _best_text(items, "spokenZh")
    subtitle = _best_text(items, "subtitleZh")
    same_text = bool(spoken and subtitle and _normalise_text(spoken) == _normalise_text(subtitle))
    subtitle_is_question = bool(subtitle and subtitle.endswith(("?", "？")))
    chosen = None if conflict else (subtitle if same_text and subtitle_is_question else spoken or subtitle)
    positions = sorted({int(cue["position"]) for cue in linked if isinstance(cue.get("position"), int) and not isinstance(cue.get("position"), bool)})
    flags: list[str] = []
    if conflict:
        flags.append("AV_CONFLICT")
    if linked and any(int(group["start_ms"]) > int(cue["start_ms"]) or int(group["end_ms"]) < int(cue["end_ms"]) for cue in linked):
        flags.append("PARTIAL_CUE")
    if _is_unresolved(group, linked, chosen):
        chosen = None
        flags.append("UNRESOLVED")
    if not chosen and "UNRESOLVED" not in flags:
        flags.append("UNRESOLVED")
    evidence = {
        "asr": _join_values([cue.get("asr_text_zh") for cue in linked]),
        "ocr": _join_values([cue.get("ocr_text_zh") for cue in linked]),
        "avSpoken": spoken or "",
        "avSubtitle": subtitle or "",
    }
    alternatives = []
    for item in items:
        alternative = {"spokenZh": item.get("spokenZh"), "subtitleZh": item.get("subtitleZh")}
        if alternative not in alternatives:
            alternatives.append(alternative)
    strongest = max((str(item["confidence"]) for item in items), key=lambda value: _CONFIDENCE_RANK[value], default="LOW")
    if conflict or "UNRESOLVED" in flags:
        strongest = "LOW"
    if spoken and subtitle and _normalise_text(spoken) == _normalise_text(subtitle):
        decision = "AV_SPOKEN_SUBTITLE_AGREE"
    elif spoken:
        decision = "AV_SPOKEN"
    elif subtitle:
        decision = "AV_SUBTITLE"
    else:
        decision = "UNRESOLVED"
    return {
        "startMs": int(group["start_ms"]),
        "endMs": int(group["end_ms"]),
        "resolvedZh": chosen,
        "confidence": strongest,
        "positions": positions,
        "evidence": evidence,
        "decision": decision,
        "flags": flags,
        "cue_ids": [str(cue["cue_id"]) for cue in linked],
        "alternatives": alternatives[:2] if conflict else [],
    }


def _visual_records(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in sorted(values, key=lambda value: (int(value["startMs"]), int(value["endMs"]), str(value["textZh"]))):
        text = str(item["textZh"]).strip()
        normalized = _normalise_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append({
            "startMs": int(item["startMs"]),
            "endMs": int(item["endMs"]),
            "textZh": text,
            "confidence": item["confidence"],
            "type": "VISUAL_ONLY",
            "positions": [],
            "evidence": {"avSpoken": "", "avSubtitle": text},
            "flags": ["VISUAL_ONLY"],
        })
    return result


def reconcile_source(
    cues: Sequence[Mapping[str, object]],
    av_analysis: Mapping[str, object],
    unresolved: Mapping[str, object] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Reconcile legacy cue evidence against one canonical AV analysis document."""
    validate_av_analysis(av_analysis)
    internal = [_as_internal(item) for item in av_analysis["utterances"]]
    dialogue = [_dialogue_record(group, _linked_cues(group, cues)) for group in _merge_groups(internal)]
    if unresolved:
        unresolved_start = int(unresolved.get("start_ms", unresolved.get("startMs", 0)))
        unresolved_end = int(unresolved.get("end_ms", unresolved.get("endMs", 0)))
        for item in dialogue:
            if int(item["startMs"]) < unresolved_end and unresolved_start < int(item["endMs"]):
                item["resolvedZh"] = None
                if "UNRESOLVED" not in item["flags"]:
                    item["flags"].append("UNRESOLVED")
                item["confidence"] = "LOW"
                break
    dialogue.sort(key=lambda item: (int(item["startMs"]), int(item["endMs"])))
    unresolved_items = [item for item in dialogue if "UNRESOLVED" in item["flags"]]
    visual_only = _visual_records(av_analysis["visualOnly"])
    return {"dialogue": dialogue, "visual_only": visual_only, "unresolved": unresolved_items}
