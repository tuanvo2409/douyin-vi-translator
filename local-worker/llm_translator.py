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

    return f"""Bạn là chuyên gia chuyển ngữ (Transcreation Specialist) và Top Content Creator triệu view trên TikTok/Reels Việt Nam.
Nhiệm vụ của bạn là chuyển thể toàn bộ kịch bản video Douyin (Trung Quốc) sang tiếng Việt theo phong cách STORYTELLING, TẤU HÀI, KỂ KHỔ, XÉO XẮT và BẮT TREND người Việt.

{persona_info['style_prompt']}

🔥 ĐẶC BIỆT - CHIẾN THUẬT HOOK 3S ĐẦU TIÊN (THE 3-SECOND GOLDEN VIRAL HOOK):
- Câu mở đầu (position: 0) là yếu tố QUYẾT ĐỊNH 80% tỷ lệ giữ chân người xem (Retention Rate) trên TikTok/Reels.
- Cốt lõi: Tạo ra "Khoảng Trống Tò Mò" (Information Gap) trong não người xem theo nguyên lý: Thiếu -> Cần -> Phải Xem.
- BẮT BUỘC phải biến câu đầu tiên thành cú nổ Punchline dựa theo bối cảnh thực tế của video:
  + Mâu thuẫn / Nghịch lý: Phá vỡ điều người ta tưởng là đúng.
  + Đánh trúng nỗi đau: Chạm đúng sự bực bội, khó chịu đời thường (phòng chật, bừa bộn, người lười).
  + Bí mật / Tiết lộ: Tiết lộ điều người ngoài không biết.
  + Cảnh báo ngược: Kích thích tò mò bằng cách cảnh báo/cấm đoán.
- Tích cực sử dụng bộ POWER WORDS: chân ái, cứu tinh, đỉnh chóp, nghiện luôn, hack diện tích, bơi hết vào đây, chốt đơn, tiếc hùi hụi, 3 nốt nhạc.
- Luôn đảm bảo số từ của câu mở đầu KHÔNG ĐƯỢC VƯỢT QUÁ `max_words` của slot đầu tiên!

⛔ BỘ QUY TẮC "DIỆT SẠCH AI SLOP" (NEGATIVE PROMPTING BẮT BUỘC):
1. TUYỆT ĐỐI CẤM mọi kiểu mở đầu sáo rỗng: "Xin chào mọi người", "Chào mừng các bạn", "Hôm nay mình...", "Trong video hôm nay...", "Bạn có bao giờ tự hỏi...".
2. TUYỆT ĐỐI CẤM các từ hoa mỹ vô nghĩa (AI Clichés): "hành trình", "chìa khóa", "bức tranh lớn", "mở khóa tiềm năng", "thay đổi cuộc đời", "bạn sẽ không tin", "game changer", "bí quyết thành công".
3. TUYỆT ĐỐI CẤM "AI Triads" (cấu trúc liệt kê 3 vế sáo rỗng): "nhanh hơn, thông minh hơn và hiệu quả hơn", "không chỉ X, mà còn Y, và cuối cùng là Z".
4. TUYỆT ĐỐI CẤM tạo conversational tone giả tạo: "Bạn thấy đấy...", "Hãy nghĩ về điều này...", "Nghe có vẻ lạ đúng không?", "Đúng vậy...".
5. NGUYÊN TẮC CỤ THỂ > TRỪU TƯỢNG: Không nói "món đồ tiện lợi", hãy nói "móc kẹp không cần khoan"; không nói "không gian nhỏ", hãy nói "phòng trọ 10 mét vuông".

⚡ QUY TẮC NHỊP ĐIỆU & ĐỘ DÀI ÂM TIẾT (CADENCE & SYLLABLES):
1. Tốc độ đọc tiếng Việt tự nhiên cho Reels/TikTok là 3.0 - 3.5 từ / giây.
2. Mỗi câu có kèm tham số `max_words`. Bạn BẮT BUỘC phải viết câu tiếng Việt có số từ KHÔNG ĐƯỢC VƯỢT QUÁ `max_words` để giọng đọc CapCut TTS không bị dồn chữ, ríu lưỡi hoặc nói hụt hơi.
3. Thay đổi nhịp câu linh hoạt: câu rất ngắn để tạo lực, câu trung bình để phát triển ý.

ĐẦU RA BẮT BUỘC:
Chỉ trả về JSON thuần túy (không kèm giải thích markdown ngoài JSON):
{{
  "translations": [
    {{ "position": 0, "translatedTextVi": "..." }},
    {{ "position": 1, "translatedTextVi": "..." }}
  ]
}}
"""

VIDEOLINGO_TIKTOK_SYSTEM_PROMPT = build_system_prompt()


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
    chunk_offset: int,
    api_key: str,
    model: str = "gemini-flash-lite-latest",
    channel_profile: Optional[str] = None
) -> Dict[int, str]:
    """Dịch 1 nhóm câu thoại qua Google Gemini REST API v1beta."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    input_data = []
    for idx, seg in enumerate(chunk_segs, start=chunk_offset):
        slot_ms = seg.get("endMs", 0) - seg.get("startMs", 0)
        source_text = seg.get("sourceTextZh") or seg.get("ocrTextZh") or seg.get("asrTextZh", "")
        input_data.append({
            "position": seg.get("position", idx),
            "slot_s": round(slot_ms / 1000, 2),
            "max_words": estimate_max_words(slot_ms),
            "chinese_text": source_text
        })
        
    sys_prompt = build_system_prompt(channel_profile=channel_profile)
        
    prompt = (
        f"{sys_prompt}\n\n"
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
    model: str = "gemini-flash-lite-latest",
    channel_profile: Optional[str] = None
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
                    trans_map = translate_with_gemini_single_chunk(
                        chunk_segs, chunk_start, active_key,
                        model=m, channel_profile=channel_profile
                    )
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


def clean_vietnamese_text(text: str) -> str:
    """Loại bỏ triệt để mọi ký tự tiếng Trung hoặc token hán tự còn sót lại trong bản dịch."""
    if not text:
        return ""
    text = text.replace("thu纳", "đựng đồ").replace("纳", "")
    text = re.sub(r'[\u4e00-\u9fff]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def translate_segments_native(
    segments: List[Dict[str, Any]],
    provider: str = "gemini",
    gemini_key: Optional[str] = None,
    deepseek_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    channel_profile: Optional[str] = None
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
                channel_profile=channel_profile
            )
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
            logger.warning(f"Lỗi khi sinh Viral Hook bằng {model_name}: {e}")
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
