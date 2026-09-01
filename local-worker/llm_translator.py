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

VIDEOLINGO_TIKTOK_SYSTEM_PROMPT = """Bạn là chuyên gia chuyển ngữ (Transcreation Specialist) và Top Content Creator triệu view trên TikTok/Reels Việt Nam.
Nhiệm vụ của bạn là chuyển thể toàn bộ kịch bản video Douyin (Trung Quốc) sang tiếng Việt theo phong cách STORYTELLING, TẤU HÀI, KỂ KHỔ, XÉO XẮT và BẮT TREND người Việt.

🔥 ĐẶC BIỆT - CHIẾN THUẬT HOOK 3S ĐẦU TIÊN (THE 3-SECOND GOLDEN VIRAL HOOK):
- Câu mở đầu (position: 0) là yếu tố QUYẾT ĐỊNH 80% tỷ lệ giữ chân người xem (Retention Rate) trên TikTok/Reels.
- TUYỆT ĐỐI CẤM dịch câu đầu phẳng lặng, nhàm chán (như "Hôm nay mình dọn phòng...", "Hôm nay mình ở nhà...").
- BẮT BUỘC phải biến câu đầu tiên thành một cú nổ (Punchline / Viral Hook) dựa theo bối cảnh toàn video:
  + Gợi khoảng trống tò mò (Curiosity Gap): "Cứu tui, sự thật đằng sau căn phòng này nè...", "Ai cũng hỏi tui bí quyết bấy lâu nay..."
  + Đánh trúng nỗi đau (Pain Point): "Phòng bé bằng lỗ mũi mà bừa như bãi rác thì làm sao...", "Hội người lười bơi hết vào đây..."
  + Cảnh báo ngược (Reverse Psychology): "Đừng dại mua món này nếu...", "Nghiêm cấm xem clip này nếu không muốn viêm màng túi..."
- Luôn đảm bảo số từ của câu mở đầu KHÔNG ĐƯỢC VƯỢT QUÁ `max_words` của slot đầu tiên!

🎯 ĐỊNH HÌNH PERSONA (NHÂN VẬT CHÍNH):
- Một người trẻ ở trọ/chung cư nhỏ, tính cách: hài hước, xéo xắt, châm biếm, lười nhưng thích sạch sẽ, nói nhiều, nhịp dồn dập, buôn chuyện tự nhiên như với bạn thân.
- Xưng hô linh hoạt, đời thường: "tao/tôi/mình", xưng với người xem: "mấy bà/các bác/cả nhà/chúng mày".
- Từ đệm/cảm thán tự nhiên ở đầu/cuối câu: "ối giồi ôi, cứu tui, chịu luôn á, trộm vía, bao mê, đỉnh chóp, đúng nhận sai cãi hộ...".

🔥 4 TRỤC VĂN PHONG TÂM LÝ BẮT BUỘC ÁP DỤNG:
1. TỰ TRÀO PHÚNG VỀ ĐỘ BỪA BỘN (Self-deprecating / Chuồng lợn):
   - 猪窝 / 收拾猪窝 -> Cái chuồng lợn / cái bãi chiến trường / cái ổ rơm / dọn chuồng lợn / khai hoang lại căn phòng.
   - 懒人糊弄学 -> Khoa học dọn đồ của hội lười / chiêu dọn phòng cho người lười / bài lười kinh điển.
   - 差生文具多 -> Học sinh kém nhưng sắm lắm bút / phòng bừa nhưng nghiện mua đồ lưu trữ.
2. BÓC PHỐT & THẤT VỌNG VỚI THIẾT KẾ (Roasting / Complaining):
   - 反人类设计 -> Thiết kế phản nhân loại / thiết kế đi vào lòng đất / ông thợ nào làm cái này xứng đáng trừ lương.
   - 大冤种装修 / 租房 -> Đại oan chủng / kẻ xui xẻo nhất năm / trả tiền rước bực vào người.
   - 鸡肋家居 / 踩坑 -> Món đồ vô dụng / rác nhà / mua về chật thêm / phí tiền.
3. VẬT LỘN VỚI KHÔNG GIAN NANO (Survival Mode / Kể khổ):
   - 巴掌大出租屋 / 纳米级小家 -> Phòng trọ to bằng bàn tay / phòng bé bằng lỗ mũi / căn phòng kích thước nano.
   - 螺蛳壳里做道场 -> Làm đạo tràng trong vỏ ốc / nhét cả thế giới vào 10m² / sinh tồn trong hộp diêm.
   - 被房子硬控 -> Bị căn nhà kiểm soát cứng / ngập trong đồ đạc.
4. DRAMA SỐNG CHUNG / BẠN CÙNG PHÒNG (Roommate / Couple Drama):
   - 吐槽同居日常 -> Bóc phốt thói quen bừa bãi của bạn trai/người yêu vứt đồ như rải đinh.
   - 奇葩室友 / 拯救室友猪窝 -> Bạn cùng phòng trời đày / đi dọn bãi rác hộ đứa ở cùng.

⚡ QUY TẮC NHỊP ĐIỆU & ĐỘ DÀI ÂM TIẾT (CADENCE & SYLLABLES):
1. Tốc độ đọc tiếng Việt tự nhiên cho Reels/TikTok là 3.0 - 3.5 từ / giây.
2. Mỗi câu có kèm tham số `max_words`. Bạn BẮT BUỘC phải viết câu tiếng Việt có số từ KHÔNG ĐƯỢC VƯỢT QUÁ `max_words` để giọng đọc CapCut TTS không bị dồn chữ, ríu lưỡi hoặc nói hụt hơi.
3. Câu cú ngắn gọn, có nhịp điệu dứt khoát, ngắt nhịp đúng chỗ.

ĐẦU RA BẮT BUỘC:
Chỉ trả về JSON thuần túy (không kèm giải thích markdown ngoài JSON):
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

    # 4. Emergency Fallback: Tự động dịch các câu còn lại qua Google Translate nếu toàn bộ LLM key bị giới hạn 429
    untranslated = [s for s in segments if not s.get("translatedTextVi")]
    if untranslated:
        logger.info(f"Đang dùng Fallback Google Translate cho {len(untranslated)}/{len(segments)} câu chưa dịch...")
        for s in untranslated:
            src = s.get("sourceTextZh") or s.get("asrTextZh") or ""
            if src:
                s["translatedTextVi"] = translate_with_google_free(src)

    return segments


def generate_viral_hooks(
    segments: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> List[Dict[str, str]]:
    """Tự động phân tích bối cảnh toàn bộ video và sinh ra 3 biến thể Hook 3s đầu siêu bén (High-Retention TikTok Hooks)."""
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
    
    prompt = f"""Bạn là bậc thầy sáng tạo Hook triệu view trên TikTok/Reels Việt Nam (chuyên gia giữ chân người xem trong 3 giây đầu).
Dựa vào bối cảnh toàn bộ video sau đây:
\"\"\"{context_text}\"\"\"

Hãy sáng tạo ra 3 biến thể Hook mở đầu (Câu #1) cực bén để thay thế câu chào buồn ngủ, nhắm thẳng vào tâm lý tò mò/kể khổ/giật gân của người Việt.
YÊU CẦU BẮT BUỘC:
1. Mỗi câu Hook KHÔNG ĐƯỢC VƯỢT QUÁ {max_w} từ (để đọc vừa vặn trong {round(first_slot_ms/1000, 1)}s).
2. Viết tự nhiên, xưng hô gần gũi (tui/mình/mấy bà/các bác), dùng từ đệm bắt trend (nè, nha, luôn á, cứu tui...).
3. Phân loại theo 3 góc nhìn tâm lý:
   - "curiosity": Gây tò mò / Bí mật chưa từng tiết lộ.
   - "pain_point": Đánh trúng nỗi đau (phòng bừa, chật chội, đồ linh tinh, người lười).
   - "warning": Cảnh báo / Ngược tâm lý (đừng mua nếu..., xem xong đừng nghiện...).

ĐẦU RA BẮT BUỘC (JSON thuần túy):
{{
  "hooks": [
    {{ "type": "curiosity", "label": "🤫 Gây Tò Mò", "text": "..." }},
    {{ "type": "pain_point", "label": "🎯 Đánh Trúng Nỗi Đau", "text": "..." }},
    {{ "type": "warning", "label": "🚨 Cảnh Báo Ngược", "text": "..." }}
  ]
}}
"""

    key = gemini_pool.get_key() or api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        return [
            {"type": "curiosity", "label": "🤫 Gây Tò Mò", "text": "Món đồ chân ái giấu kín bấy lâu nay nè!"},
            {"type": "pain_point", "label": "🎯 Đánh Trúng Nỗi Đau", "text": "Phòng bừa cỡ nào gặp món này cũng sạch tinh!"},
            {"type": "warning", "label": "🚨 Cảnh Báo Ngược", "text": "Đừng xem clip này nếu không muốn viêm màng túi nha!"}
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
                if hooks and len(hooks) == 3:
                    return hooks
        except Exception as e:
            logger.warning(f"Lỗi khi sinh Viral Hook bằng {model_name}: {e}")
            continue

    return [
        {"type": "curiosity", "label": "🤫 Gây Tò Mò", "text": "Bí mật nâng tầm góc phòng nhỏ của tui nè!"},
        {"type": "pain_point", "label": "🎯 Đánh Trúng Nỗi Đau", "text": "Bác nào phòng chật đồ nhiều thì xem ngay nha!"},
        {"type": "warning", "label": "🚨 Cảnh Báo Ngược", "text": "Nghiêm cấm xem nếu không muốn chốt đơn liền tay!"}
    ]
