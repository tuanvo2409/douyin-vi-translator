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

VIDEOLINGO_TIKTOK_SYSTEM_PROMPT = """Bạn là chuyên gia chuyển ngữ (Transcreation Specialist) và Top Content Creator trên TikTok/Reels Việt Nam với hơn 5 triệu followers.
Nhiệm vụ của bạn là chuyển thể toàn bộ kịch bản video Douyin (Trung Quốc) sang tiếng Việt TỰ NHIÊN, BẮT TREND và ĐÚNG NHỊP ĐIỆU người Việt nói chuyện trên video ngắn.

QUY TẮC BẮT BUỘC:
1. TUYỆT ĐỐI KHÔNG DỊCH WORD-BY-WORD (dịch máy thô). Hãy đọc TOÀN BỘ kịch bản video để hiểu rõ ngữ cảnh câu chuyện, sau đó diễn đạt lại bằng khẩu ngữ tự nhiên, mượt mà của người Việt.
2. PHONG CÁCH VĂN PHONG TIKTOK VIỆT NAM (DÍ DỎM, GẦN GŨI, HÀO HỨNG):
   - Xưng hô tự nhiên: "mình/tôi/em", xưng với người xem: "mọi người/các bác/cả nhà".
   - Dùng các từ đệm/từ cảm thán tự nhiên ở cuối câu: "nè, nha, luôn á, trộm vía, bao mê, cực kỳ, đỉnh chóp...".
3. DỊCH CHUẨN TIẾNG LÓNG & TỪ VỰNG DOUYIN:
   - 心心念念 -> Hằng mong ước / ao ước bấy lâu
   - 搬回家 / 终于拥有 -> Rước về nhà / tậu về / chốt đơn liền tay
   - 幸福感好物 / 绝绝子 -> Món đồ siêu mê / đồ tiện ích nâng tầm cuộc sống / chân ái
   - 铁皮文件柜 -> Chiếc tủ sắt vintage / tủ tài liệu mini
   - 量身定做 -> Đúng chuẩn sinh ra để dành cho...
   - 散落在各处的小零碎 -> Đống đồ lặt vặt / đồ linh tinh
   - 种草 / 拔草 -> Gợi ý món này / mua về dùng thử
4. KIỂM SOÁT NHỊP ĐIỆU & ÂM TIẾT (CADENCE & SYLLABLES):
   - Tốc độ đọc tiếng Việt tự nhiên của người Việt là 3.0 - 3.5 từ / giây.
   - Mỗi câu có kèm tham số `max_words`. Bạn PHẢI viết câu tiếng Việt có số lượng từ KHÔNG VƯỢT QUÁ `max_words` để giọng đọc không bị dồn chữ hoặc đọc hụt hơi.

ĐẦU RA BẮT BUỘC:
Chỉ trả về định dạng JSON thuần túy (không kèm giải thích markdown ngoài JSON):
{
  "translations": [
    { "position": 0, "translatedTextVi": "..." },
    { "position": 1, "translatedTextVi": "..." }
  ]
}
"""


def estimate_max_words(slot_ms: int) -> int:
    """Ước tính số lượng từ tiếng Việt tối đa cho một khoảng thời lượng."""
    slot_s = max(0.8, slot_ms / 1000.0)
    return max(4, int(slot_s * 3.4))


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
    chunk_start: int,
    api_key: str,
    model: str = "gemini-2.0-flash"
) -> Dict[int, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    input_data = []
    for idx, seg in enumerate(chunk_segs, start=chunk_start):
        slot_ms = seg.get("endMs", 0) - seg.get("startMs", 0)
        source_text = seg.get("sourceTextZh") or seg.get("ocrTextZh") or seg.get("asrTextZh", "")
        input_data.append({
            "position": seg.get("position", idx),
            "slot_s": round(slot_ms / 1000, 2),
            "max_words": estimate_max_words(slot_ms),
            "chinese_text": source_text
        })
        
    prompt = (
        f"{VIDEOLINGO_TIKTOK_SYSTEM_PROMPT}\n\n"
        f"Dịch kịch bản các câu sau sang tiếng Việt chuẩn TikTok:\n"
        f"{json.dumps(input_data, ensure_ascii=False, indent=2)}\n\n"
        f"Xuất JSON: {{\"translations\": [{{\"position\": 0, \"translatedTextVi\": \"...\"}}]}}"
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
    result = json.loads(text_resp.strip())
    return {item["position"]: item["translatedTextVi"] for item in result.get("translations", []) if item.get("translatedTextVi")}


from gemini_pool import gemini_pool


def translate_with_gemini(
    segments: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "gemini-flash-lite-latest"
) -> List[Dict[str, Any]]:
    """Dịch các đoạn thoại bằng Google Gemini với khả năng xoay tua key thông minh."""
    models_to_try = [model, "gemini-flash-lite-latest", "gemini-3.6-flash", "gemini-flash-latest", "gemini-2.5-flash"]
    unique_models = list(dict.fromkeys(models_to_try))
    
    chunk_size = 12
    for chunk_start in range(0, len(segments), chunk_size):
        chunk_segs = segments[chunk_start:chunk_start + chunk_size]
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
                    trans_map = translate_with_gemini_single_chunk(chunk_segs, chunk_start, active_key, model=m)
                    if trans_map:
                        success = True
                        break
                except requests.exceptions.HTTPError as he:
                    status = he.response.status_code if he.response is not None else 500
                    logger.warning(f"⚠️ Gemini Key '...{active_key[-6:]}' bị lỗi HTTP {status}. Đang đổi key khác...")
                    gemini_pool.report_error(active_key, status)
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Gemini {m} với key '...{active_key[-6:]}' lỗi: {e}")
                    continue
                    
            if success and trans_map:
                break
                
        for idx, seg in enumerate(chunk_segs, start=chunk_start):
            pos = seg.get("position", idx)
            if pos in trans_map and trans_map[pos]:
                seg["translatedTextVi"] = trans_map[pos]

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


def translate_segments_native(
    segments: List[Dict[str, Any]],
    provider: str = "gemini",
    gemini_key: Optional[str] = None,
    deepseek_key: Optional[str] = None,
    openai_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Điểm vào chính: Chuyển ngữ bản xứ bằng LLM với cơ chế xoay tua Gemini Key Pool 100% khi gặp 429."""
    if not segments:
        return segments

    # 1. Gọi Gemini API với Key Pool xoay tua tự động khi 429
    if (provider == "gemini" or not deepseek_key) and (gemini_key or gemini_pool.keys):
        try:
            logger.info("Chuyển ngữ kịch bản bằng Gemini AI (Key Pool Rotation)...")
            segments = translate_with_gemini(segments, gemini_key, model="gemini-flash-lite-latest")
        except Exception as exc:
            logger.warning(f"Lỗi khi gọi Gemini API ({exc})...")

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
            logger.warning(f"Lỗi khi gọi DeepSeek API ({exc})...")

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
            logger.warning(f"Lỗi khi gọi OpenAI API ({exc})...")

    # Báo cáo các câu chưa hoàn tất nếu tất cả key đều cạn
    untranslated = [s for s in segments if not s.get("translatedTextVi")]
    if untranslated:
        logger.warning(f"⚠️ Có {len(untranslated)}/{len(segments)} câu chưa dịch xong. Vui lòng bổ sung thêm Gemini API Key vào .env.")

    return segments
