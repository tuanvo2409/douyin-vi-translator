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


def _valid_vi(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not _CJK.search(value)


def _validate_response(document: object, records: Sequence[Mapping[str, object]], key: str) -> dict[str, str]:
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


def localize(records: Sequence[Mapping[str, object]], *, scene_memory: Mapping[str, object], request: Callable[[dict[str, object]], object], profile: str) -> dict[str, list[dict[str, object]]]:
    if profile not in PROFILES:
        raise ValueError("unsupported localization profile")
    unresolved = lambda item: item.get("classification") == "UNRESOLVED_DIALOGUE" or item.get("type") == "UNRESOLVED_DIALOGUE" or item.get("source_confidence") in {"low", "unresolved"}
    resolved = [dict(item) for item in records if item.get("kind") == "dialogue" and not unresolved(item)]
    review_required = [{"utteranceId": item["utteranceId"], "reason": "UNRESOLVED_SOURCE"} for item in records if item.get("kind") == "dialogue" and unresolved(item)]
    if not resolved:
        return {"accepted": [], "review_required": review_required}
    semantic = _validate_response(request({"mode": "semantic", "utterances": resolved, "scene_memory": dict(scene_memory)}), resolved, "semanticVi")
    semantic_records = [{**item, "semanticVi": semantic[str(item["utteranceId"])]} for item in resolved]
    native = _validate_response(request({"mode": "native", "utterances": semantic_records, "scene_memory": dict(scene_memory), "profile": PROFILES[profile], "editorial_contract": _NATIVE_EDITORIAL_CONTRACT}), semantic_records, "nativeVi")
    accepted = [{**item, "nativeVi": native[str(item["utteranceId"])], "style": {"register": str(scene_memory.get("register", "neutral")), "humorLevel": PROFILES[profile]["humor"]}} for item in semantic_records]
    return {"accepted": accepted, "review_required": review_required}
