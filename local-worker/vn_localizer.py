"""Two-pass, provider-injectable Vietnamese text localization."""
from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence


PROFILES = {
    "neutral_faithful": {"locale": "vi-VN", "humor": 0, "slang": "none"},
    "vn_native": {"locale": "vi-VN", "humor": 1, "slang": "low"},
    "vn_native_light_comedy": {"locale": "vi-VN", "humor": 2, "slang": "low"},
}
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")

_NATIVE_EDITORIAL_CONTRACT = {
    "prefer": [
        "spoken Vietnamese",
        "discourse particles",
        "omit pronouns / subjects when natural",
        "shorter colloquial phrasing",
        "attitude and roleplay consistency from scene memory",
        "contextual wit without invented jokes",
    ],
    "avoid": [
        "các bạn ... sao? constructions unless context requires them",
        "repeated nouns when Vietnamese can omit them",
        "literal source word order",
        "overly formal Tôi/Bạn defaults",
        "mechanically preserving the source subject",
        "flat roleplay declarations",
        "blindly forcing tao/mày",
    ],
}

_TERMINOLOGY_GUIDANCE = [
    "Use repeated source-grounded scene occurrences before treating an ambiguous token as a proper name.",
    "Distinguish proper names from culture-loaded common nouns, internet/social-class terms, category or status terms, and roleplay terminology.",
    "A token in a class or category construction may describe a group, style, or status rather than a person.",
    "Visual-only evidence may clarify scene meaning but must never become spoken dialogue or TTS text.",
]


def _valid_vi(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not _CJK.search(value)


def validate_translation_document(document: object, records: Sequence[Mapping[str, object]], key: str) -> dict[str, str]:
    if not isinstance(document, dict) or set(document) != {"translations"} or not isinstance(document["translations"], list):
        raise ValueError("localization response schema is invalid")
    expected = {str(item["utteranceId"]) for item in records}
    result: dict[str, str] = {}
    for item in document["translations"]:
        if not isinstance(item, dict) or set(item) != {"utteranceId", key} or not isinstance(item.get("utteranceId"), str) or not _valid_vi(item.get(key)):
            raise ValueError("localization response item is invalid")
        identifier = item["utteranceId"]
        if identifier not in expected or identifier in result:
            raise ValueError("localization response identity is invalid")
        result[identifier] = item[key].strip()
    if set(result) != expected:
        raise ValueError("localization response is incomplete")
    return result


def _with_neighbors(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Attach bounded read-only context without changing utterance identity."""
    base = [dict(record) for record in records]
    return [
        {
            **record,
            "neighbors": [dict(base[index]) for index in range(max(0, current - 2), min(len(base), current + 3)) if index != current],
        }
        for current, record in enumerate(base)
    ]


def localize(records: Sequence[Mapping[str, object]], *, scene_memory: Mapping[str, object], request: Callable[[dict[str, object]], object], profile: str, on_stage=None) -> dict[str, list[dict[str, object]]]:
    if profile not in PROFILES:
        raise ValueError("unsupported localization profile")
    unresolved = lambda item: item.get("classification") == "UNRESOLVED_DIALOGUE" or item.get("type") == "UNRESOLVED_DIALOGUE" or item.get("source_confidence") in {"low", "unresolved"}
    resolved = [dict(item) for item in records if item.get("kind") == "dialogue" and not unresolved(item)]
    review_required = [{"utteranceId": item["utteranceId"], "reason": "UNRESOLVED_SOURCE"} for item in records if item.get("kind") == "dialogue" and unresolved(item)]
    if not resolved:
        return {"accepted": [], "review_required": review_required}
    semantic_input = _with_neighbors(resolved)
    semantic = validate_translation_document(request({"mode": "semantic", "utterances": semantic_input, "scene_memory": dict(scene_memory), "terminology_evidence": list(scene_memory.get("terminologyEvidence", [])), "terminology_guidance": _TERMINOLOGY_GUIDANCE, "localization_profile": profile}), semantic_input, "semanticVi")
    if on_stage is not None:
        on_stage("semantic", [{"utteranceId": item["utteranceId"], "semanticVi": semantic[str(item["utteranceId"])]} for item in resolved])
    semantic_records = [{**item, "semanticVi": semantic[str(item["utteranceId"])]} for item in resolved]
    native_input = _with_neighbors(semantic_records)
    native = validate_translation_document(request({"mode": "native", "utterances": native_input, "scene_memory": dict(scene_memory), "terminology_evidence": list(scene_memory.get("terminologyEvidence", [])), "terminology_guidance": _TERMINOLOGY_GUIDANCE, "localization_profile": profile, "profile": PROFILES[profile], "editorial_contract": _NATIVE_EDITORIAL_CONTRACT}), native_input, "nativeVi")
    if on_stage is not None:
        on_stage("native", [{"utteranceId": item["utteranceId"], "nativeVi": native[str(item["utteranceId"])]} for item in resolved])
    accepted = [{**item, "nativeVi": native[str(item["utteranceId"])], "style": {"register": str(scene_memory.get("register", "neutral")), "humorLevel": PROFILES[profile]["humor"]}} for item in semantic_records]
    return {"accepted": accepted, "review_required": review_required}
