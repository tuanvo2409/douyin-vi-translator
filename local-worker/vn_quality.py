"""Deterministic CP2 quality checks and one bounded identity-safe repair."""
from __future__ import annotations

import re
from typing import Mapping, Sequence

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_QA_CRITERIA = [
    "SEMANTIC_DRIFT",
    "TERM_MISTRANSLATION",
    "TRANSLATIONESE",
    "STIFF_VI",
    "UNNATURAL_VI",
    "PRONOUN_INCONSISTENCY",
    "REGISTER_MISMATCH",
    "ROLEPLAY_INCONSISTENCY",
    "FORCED_SLANG",
    "FORCED_HUMOR",
    "CONTEXT_LOSS",
    "SOURCE_UNCERTAIN",
]
CANONICAL_QA_CODES = frozenset({
    *_QA_CRITERIA,
    "STIFF_VI",
    "DUPLICATE_UTTERANCE",
    "MISSING_TRANSLATION",
    "CJK_IN_VI",
    "UNRESOLVED_SOURCE",
    "UNKNOWN_QA_CODE",
})
_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}
_TERM_CHECK_GUIDANCE = [
    "Compare each source term's meaning in the full scene context with the Vietnamese term, not grammar alone.",
    "A fluent Vietnamese line is still a TERM_MISTRANSLATION when a culture-loaded, social-class, category, status, or roleplay term is rendered as an unsupported personal name.",
    "Use all source-grounded terminology occurrences, including visual-only context, as evidence; visual-only text is never a spoken utterance.",
]
QA_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "issues": {
            "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "additionalProperties": False,
                "properties": {
                    "utteranceId": {"type": "STRING"},
                    "code": {"type": "STRING"},
                    "severity": {"type": "STRING", "enum": ["LOW", "MEDIUM", "HIGH"]},
                    "reason": {"type": "STRING"},
                    "evidence": {"type": "STRING"},
                },
                "required": ["utteranceId", "code", "severity", "reason", "evidence"],
            },
        },
    },
    "required": ["issues"],
    "additionalProperties": False,
}


def _issue(utterance_id: object, code: str, *, severity: str = "MEDIUM", reason: str = "", evidence: str = "") -> dict[str, str]:
    return {
        "utteranceId": str(utterance_id),
        "code": code,
        "severity": severity,
        "reason": reason[:240],
        "evidence": evidence[:240],
    }


def _source_is_uncertain(record: Mapping[str, object]) -> bool:
    flags = {str(flag).upper() for flag in record.get("source_flags", [])} if isinstance(record.get("source_flags"), list) else set()
    source = str(record.get("resolvedZh") or "")
    return "SOURCE_UNCERTAIN" in flags or ("PARTIAL_CUE" in flags and "你呀" in source and "老奴" in source and "一声" in source)


def _contextual_term_issue(record: Mapping[str, object], terminology_evidence: object) -> dict[str, str] | None:
    if not isinstance(terminology_evidence, list):
        return None
    source = str(record.get("resolvedZh") or "")
    native = str(record.get("nativeVi") or "")
    if not source or not native or not re.search(r"\b[A-ZÀ-ỴĐ][\wÀ-ỹ-]*\s+[A-ZÀ-ỴĐ][\wÀ-ỹ-]*\b", native):
        return None
    for item in terminology_evidence:
        if not isinstance(item, Mapping) or not isinstance(item.get("occurrences"), list):
            continue
        term = str(item.get("sourceTerm") or "")
        occurrences = [occurrence for occurrence in item["occurrences"] if isinstance(occurrence, Mapping)]
        has_dialogue = any(str(occurrence.get("type")) == "DIALOGUE" and term in str(occurrence.get("text") or "") for occurrence in occurrences)
        has_visual_category = any(str(occurrence.get("type")) == "VISUAL_ONLY" and re.search(re.escape(term) + r"[\u3400-\u9fff\uf900-\ufaff]", str(occurrence.get("text") or "")) for occurrence in occurrences)
        if term and term in source and has_dialogue and has_visual_category:
            return _issue(record.get("utteranceId", ""), "TERM_MISTRANSLATION", severity="HIGH", reason="scene terminology evidence does not support a proper-name rendering", evidence=term)
    return None


def deterministic_issues(records: Sequence[Mapping[str, object]], *, scene_memory: Mapping[str, object] | None = None) -> list[dict[str, str]]:
    result = []
    seen: set[str] = set()
    for record in records:
        identifier = str(record.get("utteranceId", ""))
        if not identifier or identifier in seen:
            result.append(_issue(identifier, "DUPLICATE_UTTERANCE", reason="utterance identity is duplicated"))
        seen.add(identifier)
        semantic, native = record.get("semanticVi"), record.get("nativeVi")
        if not isinstance(semantic, str) or not semantic.strip() or not isinstance(native, str) or not native.strip():
            result.append(_issue(identifier, "MISSING_TRANSLATION", severity="HIGH", reason="semantic or native translation is missing"))
        if _CJK.search(str(semantic or "")) or _CJK.search(str(native or "")):
            result.append(_issue(identifier, "CJK_IN_VI", severity="HIGH", reason="accepted Vietnamese contains CJK text"))
        if record.get("source_confidence") in {"low", "unresolved"}:
            result.append(_issue(identifier, "UNRESOLVED_SOURCE", severity="HIGH", reason="source evidence is unresolved"))
        if _source_is_uncertain(record):
            result.append(_issue(identifier, "SOURCE_UNCERTAIN", severity="HIGH", reason="resolved source remains linguistically suspicious", evidence=str(record.get("resolvedZh") or "")))
        text = str(native or "").lower()
        if any(token in text for token in ("bro", "flex", "cháy luôn", "viral")):
            result.append(_issue(identifier, "FORCED_SLANG", reason="slang is not supported by the configured profile"))
        if any(token in text for token in ("haha", "hề hề", "cười xỉu")):
            result.append(_issue(identifier, "FORCED_HUMOR", reason="humor appears to be artificially inserted"))
        if "đối với" in text:
            result.append(_issue(identifier, "TRANSLATIONESE", reason="stiff connective suggests literal translation"))
        term_issue = _contextual_term_issue(record, (scene_memory or {}).get("terminologyEvidence"))
        if term_issue is not None:
            result.append(term_issue)
    return result


def apply_one_repair(original: Mapping[str, object], repair: Mapping[str, object]) -> dict[str, object]:
    if repair.get("utteranceId") != original.get("utteranceId") or not isinstance(repair.get("nativeVi"), str) or not repair["nativeVi"].strip():
        raise ValueError("repair must preserve utterance identity")
    return {**original, "nativeVi": repair["nativeVi"].strip()}


def validate_qa_document(document: object, records: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    """Validate and normalize one provider QA document without accepting extra fields."""
    if not isinstance(document, dict) or set(document) != {"issues"} or not isinstance(document["issues"], list):
        raise ValueError("quality response schema is invalid")
    result = []
    valid_ids = {str(item["utteranceId"]) for item in records}
    for issue in document["issues"]:
        allowed = {"utteranceId", "code", "severity", "reason", "evidence"}
        if not isinstance(issue, dict) or set(issue) != allowed or str(issue.get("utteranceId")) not in valid_ids or not isinstance(issue.get("code"), str):
            raise ValueError("quality response issue is invalid")
        severity = str(issue.get("severity", "MEDIUM")).upper()
        if severity not in _SEVERITIES:
            raise ValueError("quality response severity is invalid")
        reason = issue.get("reason", "")
        evidence = issue.get("evidence", "")
        if not isinstance(reason, str) or not isinstance(evidence, str):
            raise ValueError("quality response explanation is invalid")
        raw_code = issue["code"].strip()
        normalized = _issue(issue["utteranceId"], raw_code if raw_code in CANONICAL_QA_CODES else "UNKNOWN_QA_CODE", severity=severity, reason=reason, evidence=evidence)
        if raw_code not in CANONICAL_QA_CODES:
            normalized["originalCode"] = raw_code[:80]
        result.append(normalized)
    return result


def llm_issues(records: Sequence[Mapping[str, object]], request, *, scene_memory: Mapping[str, object] | None = None, on_validated=None) -> list[dict[str, str]]:
    """Ask one injected evaluator, with all three texts needed to judge drift."""
    utterances = [{key: record.get(key, "") for key in ("utteranceId", "resolvedZh", "semanticVi", "nativeVi", "source_confidence", "source_flags", "neighbors")} for record in records]
    document = request({
        "mode": "qa",
        "utterances": utterances,
        "scene_memory": dict(scene_memory or {}),
        "terminology_evidence": list((scene_memory or {}).get("terminologyEvidence", [])),
        "term_check_guidance": _TERM_CHECK_GUIDANCE,
        "criteria": _QA_CRITERIA,
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": QA_RESPONSE_SCHEMA},
    })
    result = validate_qa_document(document, records)
    if on_validated is not None:
        on_validated(result)
    return result
