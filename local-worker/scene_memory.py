"""Bounded, evidence-only scene context for a single localization run."""
from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

MAX_SCENE_MEMORY_BYTES = 8 * 1024
MAX_TERMINOLOGY_BYTES = 3500
_CJK_RUN = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]+")


def _record_type(record: Mapping[str, object]) -> str:
    classification = str(record.get("classification") or record.get("type") or "").upper()
    if record.get("kind") == "visual_only" or classification == "VISUAL_ONLY":
        return "VISUAL_ONLY"
    return "DIALOGUE"


def build_terminology_evidence(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Extract bounded repeated source terms without assigning translations."""
    terms: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for record in records:
        identifier = str(record.get("utteranceId", ""))
        text = str(record.get("resolvedZh") or record.get("textZh") or "").strip()
        if not identifier or not text:
            continue
        occurrence = {"utteranceId": identifier, "type": _record_type(record), "text": text[:160]}
        seen_in_record: set[str] = set()
        for run in _CJK_RUN.findall(text):
            for length in range(2, min(4, len(run)) + 1):
                for start in range(len(run) - length + 1):
                    term = run[start:start + length]
                    if term in seen_in_record:
                        continue
                    seen_in_record.add(term)
                    if term not in terms:
                        terms[term] = []
                        order.append(term)
                    if len(terms[term]) < 4:
                        terms[term].append(occurrence)
    result: list[dict[str, object]] = []
    for term in order:
        occurrences = terms[term]
        if len({item["utteranceId"] for item in occurrences}) < 2:
            continue
        candidate = {"sourceTerm": term, "occurrences": occurrences}
        encoded = json.dumps(result + [candidate], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_TERMINOLOGY_BYTES:
            break
        result.append(candidate)
    return result


def build_scene_memory(evidence: Mapping[str, object]) -> dict[str, object]:
    relationships = evidence.get("relationships", {})
    if not isinstance(relationships, dict):
        relationships = {}
    clean_relationships = {
        str(key)[:80]: {"value": str(value.get("value", "unknown"))[:80], "confidence": str(value.get("confidence", "low"))[:16]}
        for key, value in list(relationships.items())[:32] if isinstance(value, dict)
    }
    result = {
        "summary": str(evidence.get("summary", ""))[:500],
        "tone": str(evidence.get("tone", ""))[:80],
        "register": str(evidence.get("register", ""))[:80],
        "speakers": evidence.get("speakers", {}) if isinstance(evidence.get("speakers"), dict) else {},
        "relationships": clean_relationships,
        "pronounPolicy": evidence.get("pronoun_policy", {}) if isinstance(evidence.get("pronoun_policy"), dict) else {},
        "roleplayTerms": evidence.get("roleplay_terms", {}) if isinstance(evidence.get("roleplay_terms"), dict) else {},
        "runningJokes": [str(value)[:160] for value in evidence.get("running_jokes", [])[:16]] if isinstance(evidence.get("running_jokes"), list) else [],
        "glossary": evidence.get("glossary", {}) if isinstance(evidence.get("glossary"), dict) else {},
    }
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_SCENE_MEMORY_BYTES:
        raise ValueError("scene memory exceeds bounded size")
    return result
