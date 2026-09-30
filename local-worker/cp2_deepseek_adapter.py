"""Text-only DeepSeek boundary for the canonical two-pass localizer."""
from __future__ import annotations

import json
import os
from typing import Mapping

import requests

from vn_localizer import validate_translation_document


class CP2DeepSeekAdapter:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is not configured")
        self.last_provider = "deepseek"

    def __call__(self, payload: Mapping[str, object]) -> dict[str, object]:
        mode = payload.get("mode")
        if mode not in {"semantic", "native"}:
            raise ValueError("DeepSeek CP2 supports semantic and native only")
        records = payload.get("utterances")
        if not isinstance(records, list) or not records:
            raise ValueError("DeepSeek CP2 utterances are invalid")
        identifiers = [item.get("utteranceId") for item in records if isinstance(item, Mapping)]
        if len(identifiers) != len(records) or any(not isinstance(value, str) or not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("DeepSeek CP2 utterance identity is invalid")
        key = "semanticVi" if mode == "semantic" else "nativeVi"
        instruction = (
            "Dịch sát ý từng câu Trung sang tiếng Việt nói tự nhiên. Không thêm dữ kiện, đại từ, "
            "quan hệ hay hài hước không có trong nguồn."
            if mode == "semantic" else
            "Biên tập semanticVi thành lời nói tiếng Việt tự nhiên. Đối chiếu resolvedZh và câu lân cận "
            "để giữ đúng ý, chủ thể, quan hệ, số, thuật ngữ. Không ép xưng tao/tui/mày. "
            "Nếu nguồn không đủ rõ, giữ semanticVi; không đoán."
        )
        output: list[dict[str, str]] = []
        for start in range(0, len(records), 10):
            batch = records[start:start + 10]
            pending = {item["utteranceId"]: item for item in batch}
            values: dict[str, str] = {}
            for attempt in range(2):
                if not pending:
                    break
                rows = []
                for item in pending.values():
                    source = item.get("resolvedZh")
                    semantic = item.get("semanticVi")
                    if not isinstance(source, str) or not source.strip() or (mode == "native" and (not isinstance(semantic, str) or not semantic.strip())):
                        raise ValueError("DeepSeek CP2 source is invalid")
                    row = {"utteranceId": item["utteranceId"], "resolvedZh": source.strip()}
                    if mode == "native":
                        row["semanticVi"] = semantic.strip()
                    neighbors = item.get("neighbors")
                    if isinstance(neighbors, list):
                        row["neighbors"] = [
                            {"resolvedZh": str(neighbor.get("resolvedZh") or "")[:160]}
                            for neighbor in neighbors[:4] if isinstance(neighbor, Mapping)
                        ]
                    rows.append(row)
                prompt = (
                    f"{instruction} Giữ nguyên từng utteranceId, trả đúng một {key} không rỗng cho mỗi ID. "
                    f"Chỉ trả JSON {{\"translations\":[{{\"utteranceId\":\"...\",\"{key}\":\"...\"}}]}}."
                    " Không để sót chữ Hán, kể cả tên ứng dụng; dùng phiên âm Latin khi cần."
                )
                if attempt:
                    prompt += " Bản trước có mục thiếu hoặc còn chữ Hán; chỉ dịch lại các mục được gửi lần này."
                response = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": "deepseek-flash", "thinking": {"type": "disabled"},
                        "stream": False, "response_format": {"type": "json_object"},
                        "messages": [{"role": "system", "content": prompt},
                                     {"role": "user", "content": json.dumps(rows, ensure_ascii=False)}],
                    },
                    timeout=45,
                )
                response.raise_for_status()
                document = json.loads(response.json()["choices"][0]["message"]["content"])
                items = document.get("translations") if isinstance(document, dict) and set(document) == {"translations"} else None
                if not isinstance(items, list):
                    raise ValueError("DeepSeek CP2 response schema is invalid")
                seen: set[str] = set()
                for item in items:
                    identifier = item.get("utteranceId") if isinstance(item, dict) else None
                    if not isinstance(identifier, str) or identifier not in pending or identifier in seen:
                        continue
                    seen.add(identifier)
                    try:
                        values.update(validate_translation_document({"translations": [item]}, [pending[identifier]], key))
                    except ValueError:
                        continue
                pending = {identifier: item for identifier, item in pending.items() if identifier not in values}
            if pending:
                raise ValueError("DeepSeek CP2 response identity or translation remains invalid after retry")
            output.extend({"utteranceId": item["utteranceId"], key: values[item["utteranceId"]]} for item in batch)
        return {"translations": output}
