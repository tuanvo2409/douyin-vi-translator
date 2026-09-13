"""LLM Contextual Transcreation Engine (VideoLingo Architecture) for Douyin -> Native Vietnamese TikTok.

Features:
- Whole-Script Contextual Understanding
- TikTok Reviewer/Vlogger Persona & Douyin Slang Dictionary
- Syllable & Cadence Budgeting (matches CapCut TTS duration)
- Multi-provider support (Gemini 2.0/2.5 Flash, DeepSeek, OpenAI, Free Google Translate Fallback)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("dubvi_worker.llm_translator")

PAGE_PERSONAS: Dict[str, Dict[str, Any]] = {
    "page_giai_cuu_chuong_lon": {
        "name": "Giải Cứu Chuồng Lợn (Before-After & Review Gia Dụng)",
        "tone": "Hài hước, tự trào bừa bộn, mê dọn phòng kiểu lười, review đồ gia dụng rác vs chân ái, phòng nano.",
        "pronouns": "Xưng: tui/mình, gọi người xem: mấy bà/các bác/cả nhà.",
        "slang": "cái chuồng lợn, khai hoang, đồ gia dụng rác, chân ái, hack diện tích, phòng nano, bài lười kinh điển, đại oan chủng.",
        "style_prompt": """
🎯 PERSONA KÊNH: GIẢI CỨU CHUỒNG LỢN (Review Đồ Gia Dụng / Before-After)
- Tính cách: Tự trào phúng về độ bừa bộn của bản thân, đam mê khai hoang "cái chuồng lợn", thích đồ gia dụng thông minh tiết kiệm diện tích.
- Trục nội dung: So sánh đồ gia dụng rác (phí tiền) vs đồ chân ái (cứu tinh); mẹo người lười; tối ưu phòng nano 10m2.
- Xưng hô: "tui / mấy bà / các bác", xéo xắt nhưng dí dỏm, tấu hài.
"""
    },
    "page_goc_tro_bat_on": {
        "name": "Góc Trọ Bất Ổn (Drama KTX & Ở Chung)",
        "tone": "Xéo xắt, kịch tính, bóc phốt bạn cùng phòng trời đày, chủ trọ hắc ám, chuyện xóm trọ dở khóc dở cười.",
        "pronouns": "Xưng: tao/tôi/mình, gọi người xem: mấy bà/các bác/chúng mày.",
        "slang": "bạn cùng phòng trời đày, chủ trọ hắc ám, drama KTX, bóc phốt, cay đắng, xéo xắt, trầm cảm ngang, đúng nhận sai cãi hộ.",
        "style_prompt": """
🎯 PERSONA KÊNH: GÓC TRỌ BẤT ỔN (Drama KTX / Sống Chung / Bóc Phốt)
- Tính cách: Người từng trải qua 1001 kiếp nạn ở trọ, chuyên bóc phốt thói quen bừa bãi của bạn cùng phòng/người yêu và sự tích chủ trọ.
- Trục nội dung: Drama sinh viên, cãi nhau vì dọn vệ sinh, bóc phốt đồ dùng chung bị phá, trải nghiệm dở khóc dở cười.
- Xưng hô: "tao/tui", gọi người xem "mấy bà/chúng mày/các bác", giọng kể chuyện cuốn hút, hồi hộp, gay cấn.
"""
    }
}


def build_system_prompt(channel_profile: Optional[str] = None) -> str:
    persona_key = "page_giai_cuu_chuong_lon"
    if channel_profile:
        cleaned = channel_profile.lower().replace(" ", "_").replace("-", "_")
        if "goc_tro" in cleaned or "bat_on" in cleaned:
            persona_key = "page_goc_tro_bat_on"
        elif "chuong_lon" in cleaned or "giai_cuu" in cleaned:
            persona_key = "page_giai_cuu_chuong_lon"
            
    persona_info = PAGE_PERSONAS[persona_key]

    return f"""Bạn là biên dịch viên tiếng Việt cho video ngắn.
Ưu tiên theo thứ tự: (1) trung thành với lời thoại và bằng chứng nguồn,
(2) tiếng Việt nói tự nhiên, (3) chỉ dùng sắc thái của persona khi không làm đổi
sự kiện, quan hệ, chủ thể, mức độ chắc chắn hoặc ý định của câu gốc.

{persona_info['style_prompt']}

Không được bịa, thêm joke, hook, drama, lời kêu gọi, con số, quan hệ hay sự kiện
không có trong nguồn. Không tự đổi câu đầu thành hook. Giữ từng `position` nguyên
vẹn và trả về đúng một câu không rỗng cho mỗi position được yêu cầu. Tôn trọng
`max_words` để TTS đọc tự nhiên; không dùng mẹo nén thời gian.

Chỉ trả JSON thuần túy:
{{"translations": [{{"position": 0, "translatedTextVi": "..."}}]}}
"""

VIDEOLINGO_TIKTOK_SYSTEM_PROMPT = build_system_prompt()


def estimate_max_words(slot_ms: int) -> int:
    """Ước tính số lượng từ tiếng Việt tối đa cho một khoảng thời lượng."""
    slot_s = max(0.8, slot_ms / 1000.0)
    return max(4, int(slot_s * 3.4))


def _source_text(segment: Dict[str, Any]) -> str:
    return str(segment.get("sourceTextZh") or segment.get("ocrTextZh") or segment.get("asrTextZh") or "").strip()


def normalize_glossary(entries: Any) -> tuple[Dict[str, Any], ...]:
    """Keep a small, deterministic glossary; first valid source term wins."""
    if not isinstance(entries, list):
        return ()
    normalized: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries[:64]:
        if not isinstance(entry, dict):
            continue
        source, target = entry.get("source"), entry.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        source, target = source.strip(), target.strip()
        if not source or not target or source in seen or len(source) > 128 or len(target) > 128:
            continue
        seen.add(source)
        normalized.append({
            "source": source,
            "target": target,
            "category": entry.get("category") if isinstance(entry.get("category"), str) else "term",
            "confidence": entry.get("confidence") if isinstance(entry.get("confidence"), (int, float)) else None,
        })
    return tuple(normalized)


def normalize_context_card(card: Any) -> Dict[str, Any]:
    """Whitelist only identity-safe rolling context; never retain free-form story claims."""
    if not isinstance(card, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key in ("pronouns", "names", "relationships", "locations"):
        value = card.get(key)
        if isinstance(value, dict):
            clean = {str(k)[:80]: str(v)[:120] for k, v in value.items() if str(k).strip() and str(v).strip()}
            if clean:
                normalized[key] = dict(list(clean.items())[:32])
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return normalized if len(encoded.split()) <= 350 else {}


def build_contextual_batches(
    segments: List[Dict[str, Any]], max_current: int = 14, max_duration_ms: int = 60_000
) -> List[Dict[str, List[Dict[str, Any]]]]:
    """Partition immutable cues while showing only bounded neighboring source context."""
    if max_current < 1 or max_duration_ms < 1:
        raise ValueError("invalid contextual batch limits")
    batches: List[Dict[str, List[Dict[str, Any]]]] = []
    start = 0
    while start < len(segments):
        first_start = int(segments[start].get("startMs", 0))
        end = start
        while end < len(segments) and end - start < max_current:
            end_ms = int(segments[end].get("endMs", first_start))
            if end > start and end_ms - first_start > max_duration_ms:
                break
            end += 1
        current = segments[start:end]
        batches.append({
            "current": current,
            "previous": segments[max(0, start - 2):start],
            "following": segments[end:min(len(segments), end + 2)],
        })
        start = end
    return batches


def _prompt_cues(cues: List[Dict[str, Any]], include_translation: bool = False) -> List[Dict[str, Any]]:
    result = []
    for fallback_position, cue in enumerate(cues):
        position = cue.get("position", fallback_position)
        record: Dict[str, Any] = {"position": position, "chinese_text": _source_text(cue)}
        if include_translation and isinstance(cue.get("translatedTextVi"), str):
            record["translatedTextVi"] = cue["translatedTextVi"].strip()
        result.append(record)
    return result


def build_contextual_prompt(
    cues: List[Dict[str, Any]], previous: List[Dict[str, Any]], following: List[Dict[str, Any]],
    context_card: Dict[str, Any], glossary: tuple[Dict[str, Any], ...], channel_profile: Optional[str] = None,
) -> str:
    current = []
    for fallback_position, cue in enumerate(cues):
        slot_ms = max(0, int(cue.get("endMs", 0)) - int(cue.get("startMs", 0)))
        current.append({
            "position": cue.get("position", fallback_position), "chinese_text": _source_text(cue),
            "max_words": estimate_max_words(slot_ms),
        })
    return (
        f"{build_system_prompt(channel_profile)}\n\n"
        "Ngữ cảnh trước là bản dịch đã chốt; ngữ cảnh sau chỉ để hiểu nghĩa, không được dịch thay. "
        "Không được bịa hoặc suy diễn ngoài nguồn.\n"
        f"PREVIOUS={json.dumps(_prompt_cues(previous, True), ensure_ascii=False)}\n"
        f"FOLLOWING_SOURCE_ONLY={json.dumps(_prompt_cues(following), ensure_ascii=False)}\n"
        f"CONTEXT_CARD={json.dumps(normalize_context_card(context_card), ensure_ascii=False)}\n"
        f"GLOSSARY={json.dumps(glossary, ensure_ascii=False)}\n"
        f"CUES={json.dumps(current, ensure_ascii=False)}"
    )


def validate_translation_response(cues: List[Dict[str, Any]], document: Any) -> Dict[int, str]:
    """Fail closed unless the provider returns exactly one nonblank result per cue."""
    if not isinstance(document, dict) or not isinstance(document.get("translations"), list):
        raise ValueError("malformed translation response")
    expected = {cue.get("position", index) for index, cue in enumerate(cues)}
    translated: Dict[int, str] = {}
    for item in document["translations"]:
        if not isinstance(item, dict) or not isinstance(item.get("position"), int):
            raise ValueError("malformed translation item")
        position, text = item["position"], item.get("translatedTextVi")
        if position not in expected or position in translated or not isinstance(text, str) or not text.strip():
            raise ValueError("invalid translation positions")
        translated[position] = text.strip()
    if set(translated) != expected:
        raise ValueError("incomplete translation response")
    return translated


def refine_for_tts_overflow(
    segment: Dict[str, Any], *, api_key: Optional[str], measured_ms: int,
    model: str = "gemini-flash-lite-latest", channel_profile: Optional[str] = None,
) -> str | None:
    """One bounded semantic retry for a measured TTS overflow; never truncate text locally."""
    active_key = gemini_pool.get_key() or api_key
    if not active_key:
        return None
    slot_ms = max(1, int(segment.get("endMs", 0)) - int(segment.get("startMs", 0)))
    position = int(segment.get("position", 0))
    prompt = (
        f"{build_system_prompt(channel_profile)}\n\n"
        "Đây là Pass B duy nhất vì TTS đo được dài hơn slot. Giữ nguyên nghĩa, sự kiện, chủ thể và quan hệ; "
        "chỉ viết tự nhiên ngắn hơn. Không được bịa hoặc bỏ ý quan trọng.\n"
        f"SOURCE={json.dumps(_source_text(segment), ensure_ascii=False)}\n"
        f"CURRENT_VI={json.dumps(str(segment.get('translatedTextVi') or ''), ensure_ascii=False)}\n"
        f"MEASURED_MS={int(measured_ms)} SLOT_MS={slot_ms} MAX_WORDS={estimate_max_words(slot_ms)}\n"
        f"Return only {{\"translations\":[{{\"position\":{position},\"translatedTextVi\":\"...\"}}]}}"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={active_key}"
    try:
        response = requests.post(url, json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
        }, timeout=40)
        response.raise_for_status()
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return validate_translation_response([segment], json.loads(text.strip())).get(position)
    except requests.exceptions.HTTPError as error:
        status = error.response.status_code if error.response is not None else 500
        gemini_pool.report_error(active_key, status)
        logger.warning("Gemini timing pass failed with HTTP %s", status)
    except Exception as error:
        logger.warning("Gemini timing pass failed: %s", type(error).__name__)
    return None


def translate_with_google_free(text: str) -> str:
    """Fallback dịch tiếng Trung sang tiếng Việt qua Google Translate API miễn phí không cần key."""
    if not text or not text.strip():
        return ""
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "zh-CN", "tl": "vi", "dt": "t", "q": text}
        res = requests.get(url, params=params, timeout=6)
        if res.status_code == 200:
            data = res.json()
            return "".join([part[0] for part in data[0] if part and part[0]]).strip()
    except Exception:
        pass
    return text


def translate_with_gemini_single_chunk(
    chunk_segs: List[Dict[str, Any]],
    chunk_offset: int,
    api_key: str,
    model: str = "gemini-flash-lite-latest",
    channel_profile: Optional[str] = None,
    previous: Optional[List[Dict[str, Any]]] = None,
    following: Optional[List[Dict[str, Any]]] = None,
    context_card: Optional[Dict[str, Any]] = None,
    glossary: tuple[Dict[str, Any], ...] = (),
) -> Dict[int, str]:
    """Dịch 1 nhóm câu thoại qua Google Gemini REST API v1beta."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    prompt = build_contextual_prompt(
        chunk_segs, previous or [], following or [], context_card or {}, glossary, channel_profile,
    )
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.4
        }
    }
    
    resp = requests.post(url, json=payload, timeout=40)
    resp.raise_for_status()
    data = resp.json()
    
    text_resp = data["candidates"][0]["content"]["parts"][0]["text"].strip()
    if text_resp.startswith("```json"):
        text_resp = text_resp[7:]
    if text_resp.startswith("```"):
        text_resp = text_resp[3:]
    if text_resp.endswith("```"):
        text_resp = text_resp[:-3]
    return validate_translation_response(chunk_segs, json.loads(text_resp.strip()))


from gemini_pool import gemini_pool


def translate_with_gemini(
    segments: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "gemini-flash-lite-latest",
    channel_profile: Optional[str] = None,
    context_card: Optional[Dict[str, Any]] = None,
    glossary: Any = None,
) -> List[Dict[str, Any]]:
    """Dịch các đoạn thoại bằng Google Gemini với khả năng xoay tua key thông minh."""
    models_to_try = [model, "gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash"]
    unique_models = list(dict.fromkeys(models_to_try))
    bounded_context = normalize_context_card(context_card)
    bounded_glossary = normalize_glossary(glossary)
    
    chunk_start = 0
    for batch in build_contextual_batches(segments, max_current=14, max_duration_ms=60_000):
        chunk_segs = batch["current"]
        trans_map = {}
        
        # Thử xoay tua các key trong pool
        max_attempts = max(3, len(gemini_pool.keys) * 2)
        for attempt in range(max_attempts):
            active_key = gemini_pool.get_key() or api_key
            if not active_key:
                break
                
            success = False
            for m in unique_models:
                try:
                    trans_map = translate_with_gemini_single_chunk(
                        chunk_segs, chunk_start, active_key,
                        model=m, channel_profile=channel_profile,
                        previous=batch["previous"], following=batch["following"],
                        context_card=bounded_context, glossary=bounded_glossary,
                    )
                    if trans_map:
                        success = True
                        break
                except requests.exceptions.HTTPError as he:
                    status = he.response.status_code if he.response is not None else 500
                    logger.warning(f"⚠️ Gemini model {m} gặp HTTP {status}. Đang đổi key khác...")
                    gemini_pool.report_error(active_key, status)
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Gemini model {m} lỗi kiểu {type(e).__name__}")
                    continue
                    
            if success and trans_map:
                break
                
        for idx, seg in enumerate(chunk_segs, start=chunk_start):
            pos = seg.get("position", idx)
            if pos in trans_map and trans_map[pos]:
                seg["translatedTextVi"] = trans_map[pos]
        chunk_start += len(chunk_segs)

    return segments


def translate_with_openai_compatible(
    segments: List[Dict[str, Any]],
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini"
) -> List[Dict[str, Any]]:
    """Dịch toàn bộ kịch bản qua OpenAI / DeepSeek API."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    input_data = []
    for idx, seg in enumerate(segments):
        slot_ms = seg.get("endMs", 0) - seg.get("startMs", 0)
        source_text = seg.get("sourceTextZh") or seg.get("ocrTextZh") or seg.get("asrTextZh", "")
        input_data.append({
            "position": seg.get("position", idx),
            "slot_s": round(slot_ms / 1000, 2),
            "max_words": estimate_max_words(slot_ms),
            "chinese_text": source_text
        })
        
    prompt = (
        f"Đây là toàn bộ kịch bản video Douyin:\n"
        f"{json.dumps(input_data, ensure_ascii=False, indent=2)}\n\n"
        f"Hãy chuyển thể từng câu sang tiếng Việt bản xứ tự nhiên theo đúng `max_words`."
    )
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VIDEOLINGO_TIKTOK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4
    }
    
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=35)
    resp.raise_for_status()
    data = resp.json()
    
    text_resp = data["choices"][0]["message"]["content"]
    result = json.loads(text_resp)
    
    trans_map = {item["position"]: item["translatedTextVi"] for item in result.get("translations", [])}
    for idx, seg in enumerate(segments):
        pos = seg.get("position", idx)
        if pos in trans_map and trans_map[pos]:
            seg["translatedTextVi"] = trans_map[pos]
    return segments


def clean_vietnamese_text(text: str) -> str:
    """Loại bỏ triệt để mọi ký tự tiếng Trung hoặc token hán tự còn sót lại trong bản dịch."""
    if not text:
        return ""
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def translate_segments_native(
    segments: List[Dict[str, Any]],
    provider: str = "gemini",
    gemini_key: Optional[str] = None,
    deepseek_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    channel_profile: Optional[str] = None,
    context_card: Optional[Dict[str, Any]] = None,
    glossary: Any = None,
) -> List[Dict[str, Any]]:
    """Điểm vào chính: Chuyển ngữ bản xứ bằng LLM với cơ chế xoay tua Gemini Key Pool 100% khi gặp 429."""
    if not segments:
        return segments

    # 1. Gọi Gemini API với Key Pool xoay tua tự động khi 429
    if (provider == "gemini" or not deepseek_key) and (gemini_key or gemini_pool.keys):
        try:
            logger.info("Chuyển ngữ kịch bản bằng Gemini AI (Key Pool Rotation)...")
            segments = translate_with_gemini(
                segments, gemini_key,
                model="gemini-flash-lite-latest",
                channel_profile=channel_profile, context_card=context_card, glossary=glossary,
            )
        except Exception as exc:
            logger.warning(f"Lỗi khi gọi Gemini API ({type(exc).__name__})...")

    # 2. Thử DeepSeek API nếu được cấu hình
    if not all(s.get("translatedTextVi") for s in segments) and deepseek_key:
        try:
            logger.info("Chuyển ngữ kịch bản bằng DeepSeek Chat API...")
            segments = translate_with_openai_compatible(
                segments, deepseek_key,
                base_url="https://api.deepseek.com/v1",
                model="deepseek-chat"
            )
        except Exception as exc:
            logger.warning(f"Lỗi khi gọi DeepSeek API ({type(exc).__name__})...")

    # 3. Thử OpenAI API nếu được cấu hình
    if not all(s.get("translatedTextVi") for s in segments) and openai_key:
        try:
            logger.info("Chuyển ngữ kịch bản bằng OpenAI gpt-4o-mini...")
            segments = translate_with_openai_compatible(
                segments, openai_key,
                base_url="https://api.openai.com/v1",
                model="gpt-4o-mini"
            )
        except Exception as exc:
            logger.warning(f"Lỗi khi gọi OpenAI API ({type(exc).__name__})...")

    # 4. Emergency Fallback: Tự động dịch các câu còn lại qua Google Translate nếu toàn bộ LLM key bị giới hạn 429
    untranslated = [s for s in segments if not s.get("translatedTextVi")]
    if untranslated:
        logger.info(f"Đang dùng Fallback Google Translate cho {len(untranslated)}/{len(segments)} câu chưa dịch...")
        for s in untranslated:
            src = s.get("sourceTextZh") or s.get("asrTextZh") or ""
            if src:
                s["translatedTextVi"] = translate_with_google_free(src)

    # 5. Sanitize sạch 100% ký tự tiếng Trung còn sót lại
    for s in segments:
        if s.get("translatedTextVi"):
            s["translatedTextVi"] = clean_vietnamese_text(s["translatedTextVi"])

    return segments


def generate_viral_hooks(
    segments: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    channel_profile: Optional[str] = None
) -> List[Dict[str, str]]:
    """Tự động phân tích bối cảnh toàn bộ video và sinh ra 8 biến thể Hook 3s đầu siêu bén theo Persona kênh."""
    if not segments:
        return []
    
    first_slot_ms = segments[0].get("endMs", 3000) - segments[0].get("startMs", 0)
    max_w = estimate_max_words(first_slot_ms)
    
    # Tóm tắt bối cảnh các câu đầu trong video
    context_lines = []
    for s in segments[:10]:
        t = s.get("translatedTextVi") or s.get("sourceTextZh") or s.get("asrTextZh") or ""
        if t:
            context_lines.append(t)
    context_text = " | ".join(context_lines)
    
    persona_key = "page_giai_cuu_chuong_lon"
    if channel_profile:
        cleaned = channel_profile.lower().replace(" ", "_").replace("-", "_")
        if "goc_tro" in cleaned or "bat_on" in cleaned:
            persona_key = "page_goc_tro_bat_on"
            
    persona_desc = PAGE_PERSONAS[persona_key]["name"]
    
    prompt = f"""Bạn là bậc thầy sáng tạo Hook triệu view trên TikTok/Reels Việt Nam cho kênh: {persona_desc}.
Dựa vào bối cảnh toàn bộ video sau đây:
\"\"\"{context_text}\"\"\"

Hãy sáng tạo ra ĐÚNG 8 biến thể Hook mở đầu (Câu #1) cực bén theo Ma trận 8 công thức Hook triệu view kinh điển:
YÊU CẦU BẮT BUỘC:
1. Mỗi câu Hook KHÔNG ĐƯỢC VƯỢT QUÁ {max_w} từ (để đọc vừa vặn trong {round(first_slot_ms/1000, 1)}s).
2. Viết tự nhiên như lời nói, nhịp dứt khoát, dùng xưng hô gần gũi (tui/mình/mấy bà/các bác) và POWER WORDS (chân ái, cứu tinh, đỉnh chóp, nghiện luôn, hack diện tích, tiếc hùi hụi, 3 nốt nhạc).
3. TUYỆT ĐỐI CẤM từ sáo rỗng AI (hành trình, chìa khóa, bí quyết, bạn sẽ không tin, game changer).
4. Phân loại theo đúng 8 góc nhìn tâm lý:
   - "contradiction": 🎭 Mâu Thuẫn (Phá vỡ niềm tin sai lầm)
   - "shocking_number": 🔢 Con Số Sốc (Kết quả/chi phí/thời gian cụ thể)
   - "insider_secret": 🤫 Bí Mật Nghề (Điều dân trong ngành/shop giấu kín)
   - "result_first": ⚡ Kết Quả Trước (Thành quả bất ngờ trước quy trình)
   - "personal_question": 🎯 Gọi Tên (Đánh trúng người xem & nỗi đau cụ thể)
   - "in_medias_res": 🎬 Giữa Drama (Bắt đầu giữa tình huống căng thẳng)
   - "pattern_interrupt": 🤯 Phá Chuẩn (So sánh ngược đời/cắt đứt thói quen lướt)
   - "warning": 🚨 Cảnh Báo (Cảnh báo thẳng thừng/ngược tâm lý)

ĐẦU RA BẮT BUỘC (JSON thuần túy):
{{
  "hooks": [
    {{ "type": "contradiction", "label": "🎭 Mâu Thuẫn", "text": "..." }},
    {{ "type": "shocking_number", "label": "🔢 Con Số Sốc", "text": "..." }},
    {{ "type": "insider_secret", "label": "🤫 Bí Mật Nghề", "text": "..." }},
    {{ "type": "result_first", "label": "⚡ Kết Quả Trước", "text": "..." }},
    {{ "type": "personal_question", "label": "🎯 Gọi Tên", "text": "..." }},
    {{ "type": "in_medias_res", "label": "🎬 Giữa Drama", "text": "..." }},
    {{ "type": "pattern_interrupt", "label": "🤯 Phá Chuẩn", "text": "..." }},
    {{ "type": "warning", "label": "🚨 Cảnh Báo", "text": "..." }}
  ]
}}
"""

    key = gemini_pool.get_key() or api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return [
            {"type": "contradiction", "label": "🎭 Mâu Thuẫn", "text": "Phòng chật không phải do đồ nhiều đâu!"},
            {"type": "shocking_number", "label": "🔢 Con Số Sốc", "text": "Góc 10m2 rộng gấp đôi sau 3 ngày!"},
            {"type": "insider_secret", "label": "🤫 Bí Mật Nghề", "text": "Món đồ dân decor giấu kín bấy lâu nay!"},
            {"type": "result_first", "label": "⚡ Kết Quả Trước", "text": "Hack phòng trọ ngăn nắp trong 3 nốt nhạc!"},
            {"type": "personal_question", "label": "🎯 Gọi Tên", "text": "Bác nào phòng bừa cứu tinh đây rồi!"},
            {"type": "in_medias_res", "label": "🎬 Giữa Drama", "text": "Đang dọn phòng mà muốn khóc thét nè!"},
            {"type": "pattern_interrupt", "label": "🤯 Phá Chuẩn", "text": "Tủ sắt mini mà đựng được cả thế giới!"},
            {"type": "warning", "label": "🚨 Cảnh Báo", "text": "Đừng mua tủ này nếu sợ quá nghiện nha!"}
        ]

    for model_name in ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.0-flash"]:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.7}
            }
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                parsed = json.loads(raw_text)
                hooks = parsed.get("hooks", [])
                if hooks and len(hooks) >= 6:
                    for h in hooks:
                        h["text"] = clean_vietnamese_text(h.get("text", ""))
                    return hooks
        except Exception as e:
            logger.warning(f"Lỗi khi sinh Viral Hook bằng {model_name}: {type(e).__name__}")
            continue

    fallback_hooks = [
        {"type": "contradiction", "label": "🎭 Mâu Thuẫn", "text": "Phòng chật không phải do đồ nhiều đâu!"},
        {"type": "shocking_number", "label": "🔢 Con Số Sốc", "text": "Góc 10m2 rộng gấp đôi sau 3 ngày!"},
        {"type": "insider_secret", "label": "🤫 Bí Mật Nghề", "text": "Món đồ dân decor giấu kín bấy lâu nay!"},
        {"type": "result_first", "label": "⚡ Kết Quả Trước", "text": "Hack phòng trọ ngăn nắp trong 3 nốt nhạc!"},
        {"type": "personal_question", "label": "🎯 Gọi Tên", "text": "Bác nào phòng bừa cứu tinh đây rồi!"},
        {"type": "in_medias_res", "label": "🎬 Giữa Drama", "text": "Đang dọn phòng mà muốn khóc thét nè!"},
        {"type": "pattern_interrupt", "label": "🤯 Phá Chuẩn", "text": "Tủ sắt mini mà đựng được cả thế giới!"},
        {"type": "warning", "label": "🚨 Cảnh Báo", "text": "Đừng mua tủ này nếu sợ quá nghiện nha!"}
    ]
    for h in fallback_hooks:
        h["text"] = clean_vietnamese_text(h["text"])
    return fallback_hooks


def generate_social_post_caption(
    segments: List[Dict[str, Any]],
    channel_profile: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Sinh tiêu đề giật tít, nội dung caption và 5-7 hashtags chuẩn SEO đăng TikTok/Reels/Shorts."""
    context_lines = []
    for s in segments[:8]:
        t = s.get("translatedTextVi") or s.get("sourceTextZh") or ""
        if t:
            context_lines.append(t)
    context_text = " ".join(context_lines)
    
    persona_key = "page_giai_cuu_chuong_lon"
    default_tags = ["#giaicuuchuonglon", "#reviewgiadung", "#decorphongtro", "#meodondep", "#giadungthongminh", "#fyp", "#xuhuong"]
    if channel_profile:
        cleaned = channel_profile.lower().replace(" ", "_").replace("-", "_")
        if "goc_tro" in cleaned or "bat_on" in cleaned:
            persona_key = "page_goc_tro_bat_on"
            default_tags = ["#goctrobaton", "#dramasinhtro", "#dramaktx", "#bancungphong", "#chutro", "#sinhvien", "#xuhuong"]

    prompt = f"""Bạn là chuyên gia sáng tạo Caption & Hashtag triệu view trên TikTok/Reels cho kênh: {PAGE_PERSONAS[persona_key]['name']}.
Dựa trên nội dung kịch bản video sau:
\"\"\"{context_text}\"\"\"

Hãy tạo ra:
1. "title": Tiêu đề giật tít thu hút (dưới 15 từ, có icon sinh động, đánh trúng tò mò).
2. "body": Đoạn caption ngắn 1-2 câu kể lể/kêu gọi thảo luận bình luận (VD: "Ai cùng cảnh ngộ điểm danh coi?").
3. "hashtags": Danh sách đúng 6-8 hashtags chuẩn SEO theo chủ đề kênh.
4. "full_post": Ghép hoàn chỉnh title + body + hashtags thành 1 đoạn văn bản sẵn sàng copy đăng bài.

ĐẦU RA BẮT BUỘC (JSON thuần túy):
{{
  "title": "...",
  "body": "...",
  "hashtags": ["#tag1", "#tag2", ...],
  "full_post": "..."
}}
"""
    key = gemini_pool.get_key() or api_key or os.getenv("GEMINI_API_KEY")
    if key:
        for model_name in ["gemini-flash-lite-latest", "gemini-2.5-flash", "gemini-2.0-flash"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json", "temperature": 0.6}
                }
                resp = requests.post(url, json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    parsed = json.loads(raw_text)
                    if parsed.get("title") and parsed.get("full_post"):
                        return parsed
            except Exception:
                continue

    # Fallback
    first_sentence = context_lines[0] if context_lines else "Cải tạo góc nhỏ siêu mê!"
    tags_str = " ".join(default_tags)
    full_fallback = f"🔥 {first_sentence}\n\nMấy bà thấy món này thế nào? Cùng chia sẻ ở dưới nha!\n\n{tags_str}"
    return {
        "title": f"🔥 {first_sentence}",
        "body": "Mấy bà thấy món này thế nào? Cùng chia sẻ ở dưới nha!",
        "hashtags": default_tags,
        "full_post": full_fallback
    }
