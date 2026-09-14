"""Deterministic CP2 quality checks and one bounded identity-safe repair."""
from __future__ import annotations

import re
from typing import Mapping, Sequence

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_QA_CRITERIA = ["SEMANTIC_DRIFT", "STIFF_VI", "TRANSLATIONESE", "ROLEPLAY_INCONSISTENCY", "FORCED_HUMOR", "FORCED_SLANG"]


def deterministic_issues(records: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    result = []
    seen: set[str] = set()
    for record in records:
        identifier = str(record.get("utteranceId", ""))
        if not identifier or identifier in seen:
            result.append({"utteranceId": identifier, "code": "DUPLICATE_UTTERANCE"})
        seen.add(identifier)
        semantic, native = record.get("semanticVi"), record.get("nativeVi")
        if not isinstance(semantic, str) or not semantic.strip() or not isinstance(native, str) or not native.strip():
            result.append({"utteranceId": identifier, "code": "MISSING_TRANSLATION"})
        if _CJK.search(str(semantic or "")) or _CJK.search(str(native or "")):
            result.append({"utteranceId": identifier, "code": "CJK_IN_VI"})
        if record.get("source_confidence") in {"low", "unresolved"}:
            result.append({"utteranceId": identifier, "code": "UNRESOLVED_SOURCE"})
        text = str(native or "").lower()
        if any(token in text for token in ("bro", "flex", "cháy luôn", "viral")):
            result.append({"utteranceId": identifier, "code": "FORCED_SLANG"})
        if any(token in text for token in ("haha", "hề hề", "cười xỉu")):
            result.append({"utteranceId": identifier, "code": "FORCED_HUMOR"})
        if "đối với" in text:
            result.append({"utteranceId": identifier, "code": "TRANSLATIONESE"})
    return result


def apply_one_repair(original: Mapping[str, object], repair: Mapping[str, object]) -> dict[str, object]:
    if repair.get("utteranceId") != original.get("utteranceId") or not isinstance(repair.get("nativeVi"), str) or not repair["nativeVi"].strip():
        raise ValueError("repair must preserve utterance identity")
    return {**original, "nativeVi": repair["nativeVi"].strip()}


def llm_issues(records: Sequence[Mapping[str, object]], request) -> list[dict[str, str]]:
    """Ask one injected evaluator, with all three texts needed to judge drift."""
    utterances = [{key: record.get(key, "") for key in ("utteranceId", "resolvedZh", "semanticVi", "nativeVi", "source_confidence")} for record in records]
    document = request({"mode": "qa", "utterances": utterances, "criteria": _QA_CRITERIA})
    if not isinstance(document, dict) or set(document) != {"issues"} or not isinstance(document["issues"], list):
        raise ValueError("quality response schema is invalid")
    result = []
    valid_ids = {str(item["utteranceId"]) for item in records}
    for issue in document["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"utteranceId", "code"} or str(issue.get("utteranceId")) not in valid_ids or not isinstance(issue.get("code"), str):
            raise ValueError("quality response issue is invalid")
        result.append({"utteranceId": issue["utteranceId"], "code": issue["code"]})
    return result
