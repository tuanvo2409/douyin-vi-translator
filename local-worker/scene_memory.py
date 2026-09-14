"""Bounded, evidence-only scene context for a single localization run."""
from __future__ import annotations

import json
from typing import Mapping

MAX_SCENE_MEMORY_BYTES = 8 * 1024


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
