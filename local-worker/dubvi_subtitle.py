#!/usr/bin/env python3
"""DUBVI Subtitle, OCR & Computer Vision Module."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_rapidocr_engine: Any | None = None        # Cache RapidOCR engine giá»¯a cĂ¡c segment


def round_rect_ass(w: int, h: int, r: int = 20) -> str:
    """Táº¡o ASS Vector path cho hĂ¬nh chá»¯ nháº­t bo trĂ²n tá»« (0, 0) Ä‘áº¿n (w, h)."""
    r = min(r, w // 2, h // 2)
    k = int(r * 0.55228)
    return (
        f"m {r} 0 "
        f"l {w - r} 0 "
        f"b {w - r + k} 0 {w} {r - k} {w} {r} "
        f"l {w} {h - r} "
        f"b {w} {h - r + k} {w - r + k} {h} {w - r} {h} "
        f"l {r} {h} "
        f"b {r - k} {h} 0 {h - r + k} 0 {h - r} "
        f"l 0 {r} "
        f"b 0 {r - k} {r - k} 0 {r} 0"
    )


def auto_detect_subtitle_roi(video_path: Path, sample_count: int = 8) -> dict[str, Any]:
    """
    DYNAMIC AI OCR SUBTITLE SCANNER:
    Tá»± Ä‘á»™ng quĂ©t vá»‹ trĂ­ Y, chiá»u cao font chá»¯ vĂ  dáº£i che cháº¯n cá»§a phá»¥ Ä‘á» tiáº¿ng Trung gá»‘c trong video.
    KhĂ´ng hardcode - thĂ­ch á»©ng 100% vá»›i tá»«ng video Douyin riĂªng biá»‡t.
    """
    try:
        import cv2
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return {
            "has_hardsub": False,
            "yPercent": 75.0,
            "yCenterPx": 1440,
            "avgHeightPx": 60,
            "boxBlurY": 1380,
            "boxBlurH": 120
        }

    global _rapidocr_engine
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()
    engine = _rapidocr_engine

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1080
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920

    if total_frames <= 0:
        cap.release()
        return {"has_hardsub": False, "yPercent": 75.0, "yCenterPx": int(height * 0.75), "avgHeightPx": 60, "boxBlurY": int(height * 0.75 - 60), "boxBlurH": 120}

    step = max(1, total_frames // (sample_count + 1))
    detected_boxes = []

    for i in range(1, sample_count + 1):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Chá»‰ quĂ©t 50% ná»­a dÆ°á»›i mĂ n hĂ¬nh (nÆ¡i phá»¥ Ä‘á» Douyin xuáº¥t hiá»‡n)
        crop_y_start = int(height * 0.50)
        crop_bottom = frame[crop_y_start:, :]

        result, _ = engine(crop_bottom)
        if not result:
            continue

        for item in result:
            box = item[0]
            text = item[1]
            score = float(item[2])

            if score < 0.48 or len(text) < 2:
                continue

            xs = [p[0] for p in box]
            ys = [p[1] + crop_y_start for p in box]

            box_x = min(xs)
            box_y = min(ys)
            box_w = max(xs) - min(xs)
            box_h = max(ys) - min(ys)

            # Lá»c box phá»¥ Ä‘á» há»£p lá»‡: chiá»u cao 20px - 160px, náº±m á»Ÿ ná»­a dÆ°á»›i
            if 20 <= box_h <= 160 and box_w >= 100 and (box_y / height) >= 0.52:
                detected_boxes.append({
                    "text": text,
                    "x": box_x,
                    "y": box_y,
                    "w": box_w,
                    "h": box_h,
                    "y_center": box_y + box_h / 2.0,
                    "y_percent": (box_y + box_h / 2.0) / height * 100.0,
                    "score": score
                })

    cap.release()

    if not detected_boxes:
        return {
            "has_hardsub": False,
            "yPercent": 75.0,
            "yCenterPx": int(height * 0.75),
            "avgHeightPx": 60,
            "boxBlurY": int(height * 0.75 - 60),
            "boxBlurH": 120
        }

    avg_y_center = sum(b["y_center"] for b in detected_boxes) / len(detected_boxes)
    avg_h = sum(b["h"] for b in detected_boxes) / len(detected_boxes)
    avg_y_percent = (avg_y_center / height) * 100.0

    blur_h = int(avg_h + 40)
    blur_y = max(0, int(avg_y_center - blur_h / 2.0))

    return {
        "has_hardsub": True,
        "sampleCount": len(detected_boxes),
        "yPercent": round(avg_y_percent, 2),
        "yCenterPx": int(avg_y_center),
        "avgHeightPx": int(avg_h),
        "boxBlurY": blur_y,
        "boxBlurH": blur_h
    }


DOUYIN_OUTRO_CTA_KEYWORDS = [
    "ä½å°å¸®", "è®¾è®¡æˆ‘å®¶", "å·¦ä¸‹è§’", "æœå°åŒº", "å°åŒºå", 
    "å…è´¹ç”Ÿæˆ", "ä¸“å±ä½ å®¶", "ç‚¹èµå…³æ³¨", "å…³æ³¨æˆ‘", "ä¸»é¡µ", 
    "ç§ä¿¡", "ç‚¹å‡»é“¾æ¥", "ç‚¹å‡»ä¸‹æ–¹", "ä¸‹è½½", "ä¸€é”®åŒ¹é…",
    "æˆ·å‹å›¾", "å…è´¹é¢†å–", "åŒæ¬¾è®¾è®¡", "ç«‹å³é¢„çº¦", "å°æˆ¿å­"
]


def detect_outro_cta(
    segments: list[dict[str, Any]],
    video_duration_s: float,
    tail_window_s: float = 20.0
) -> tuple[int | None, str | None, int | None]:
    """
    SMART OUTRO CTA TRIMMER (Tá»± Ä‘á»™ng phĂ¡t hiá»‡n & Cáº¯t gá»t CTA quáº£ng cĂ¡o Ä‘uĂ´i video Douyin):
    QuĂ©t ngÆ°á»£c trong tail_window_s cuá»‘i cĂ¹ng cá»§a video.
    Tráº£ vá»: (cutoff_ms, matched_keyword, drop_from_index) hoáº·c (None, None, None) náº¿u video sáº¡ch.
    """
    if not segments or video_duration_s <= 0:
        return None, None, None

    threshold_ms = int(max(0.0, video_duration_s - tail_window_s) * 1000)
    
    first_cta_idx = None
    matched_keyword = None
    
    for i in range(len(segments) - 1, -1, -1):
        seg = segments[i]
        st_ms = seg.get("startMs", 0)
        if st_ms < threshold_ms:
            break
            
        text = (seg.get("asrTextZh") or seg.get("chinese_text") or seg.get("text") or "").strip()
        for kw in DOUYIN_OUTRO_CTA_KEYWORDS:
            if kw in text:
                first_cta_idx = i
                matched_keyword = kw
                break
                
    if first_cta_idx is not None:
        cutoff_ms = segments[first_cta_idx]["startMs"]
        return cutoff_ms, matched_keyword, first_cta_idx

    return None, None, None


def attach_dynamic_subtitle_coordinates(
    video_path: Path,
    segments: list[dict[str, Any]],
    roi: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    DYNAMIC SUBTITLE TRACKER:
    Soi tá»«ng frame tÆ°Æ¡ng á»©ng vá»›i tá»«ng cĂ¢u thoáº¡i trĂªn video.
    Báº¯t trá»n tá»a Ä‘á»™ Y thá»±c táº¿ cá»§a phá»¥ Ä‘á» gá»‘c tiáº¿ng Trung táº¡i thá»i Ä‘iá»ƒm Ä‘Ă³,
    giĂºp tháº» Ä‘en bo trĂ²n tá»± Ä‘á»™ng bay Ä‘áº¿n Ä‘Ăºng vá»‹ trĂ­ cá»§a cĂ¢u Ä‘Ă³ (khĂ´ng cá»‘ Ä‘á»‹nh).
    """
    try:
        import cv2
        import re
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return segments

    global _rapidocr_engine
    if _rapidocr_engine is None:
        _rapidocr_engine = RapidOCR()
    ocr = _rapidocr_engine

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = total_frames / fps if fps > 0 else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1920
    default_y = int(roi.get("yCenterPx", int(height * 0.75)))
    crop_y_start = int(height * 0.40)

    for seg in segments:
        st_s = seg.get("startMs", 0) / 1000.0
        et_s = seg.get("endMs", 0) / 1000.0
        t_mid = (st_s + et_s) / 2.0
        if t_mid >= dur:
            t_mid = max(0.0, dur - 0.2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_mid * fps))
        ret, frame = cap.read()
        if not ret or frame is None:
            seg["yCenterPx"] = default_y
            continue

        crop = frame[crop_y_start:, :]
        res, _ = ocr(crop)
        
        # Náº¿u chÆ°a tĂ¬m tháº¥y chá»¯ HĂ¡n (vĂ­ dá»¥ video bá»‹ láº­t gÆ°Æ¡ng hflip Ä‘á»ƒ nĂ© báº£n quyá»n), quĂ©t trĂªn báº£n láº­t ngÆ°á»£c láº¡i:
        has_zh = res and any(re.search(r'[\u4e00-\u9fff]', item[1]) for item in res)
        if not has_zh:
            crop_flipped = cv2.flip(crop, 1)
            res_flipped, _ = ocr(crop_flipped)
            if res_flipped and any(re.search(r'[\u4e00-\u9fff]', item[1]) for item in res_flipped):
                res = res_flipped

        if not res:
            seg["yCenterPx"] = None
            continue

        zh_boxes = []
        for item in res:
            text = item[1].strip()
            score = float(item[2])
            if score < 0.40 or len(text) < 1:
                continue
            if not re.search(r'[\u4e00-\u9fff]', text):
                continue
            ys = [p[1] + crop_y_start for p in item[0]]
            min_y = min(ys)
            max_y = max(ys)
            h = max_y - min_y
            if 20 <= h <= 180:
                center_y = int((min_y + max_y) / 2.0)
                zh_boxes.append((score, center_y, h))

        if zh_boxes:
            zh_boxes.sort(key=lambda b: -b[0])
            best_score, best_cy, best_h = zh_boxes[0]
            seg["yCenterPx"] = best_cy
            seg["boxHeightPx"] = best_h
        else:
            seg["yCenterPx"] = None

    cap.release()

    # TĂ­nh tá»a Ä‘á»™ Y trung vá»‹ (median) cá»§a cĂ¡c cĂ¢u Ä‘Ă£ báº¯t Ä‘Æ°á»£c phá»¥ Ä‘á» thĂ nh cĂ´ng
    valid_ys = [s["yCenterPx"] for s in segments if s.get("yCenterPx") is not None]
    if valid_ys:
        median_y = int(sorted(valid_ys)[len(valid_ys) // 2])
        # BĂ¹ láº¡i cho cĂ¡c cĂ¢u bá»‹ trÆ°á»£t OCR Ä‘á»ƒ luĂ´n bĂ¡m cháº·t dáº£i phá»¥ Ä‘á» gá»‘c (khĂ´ng bá»‹ rÆ¡i xuá»‘ng Ä‘Ă¡y 1440px)
        for s in segments:
            if s.get("yCenterPx") is None:
                s["yCenterPx"] = median_y
    else:
        # Fallback an toĂ n: Douyin drama luĂ´n Ä‘áº·t phá»¥ Ä‘á» á»Ÿ khoáº£ng 50% - 52% (Y â‰ˆ 960px)
        fallback_y = int(roi.get("yCenterPx", int(height * 0.50)))
        if fallback_y > int(height * 0.65):
            fallback_y = int(height * 0.50)
        for s in segments:
            if s.get("yCenterPx") is None:
                s["yCenterPx"] = fallback_y

    return segments


def generate_channel_logo_badge(
    avatar_path: Path,
    output_path: Path,
    size_px: int = 200,
    opacity: float = 0.70,
    border_w: int = 5
) -> Path | None:
    """Tá»± Ä‘á»™ng cáº¯t trĂ²n vĂ  lĂ m má» Avatar kĂªnh theo chuáº©n nháº­n diá»‡n thÆ°Æ¡ng hiá»‡u."""
    if not avatar_path or not Path(avatar_path).exists():
        return None
    try:
        from PIL import Image, ImageDraw, ImageOps
        img = Image.open(avatar_path).convert("RGBA")
        size = (size_px, size_px)
        img = img.resize(size, Image.Resampling.LANCZOS)

        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)

        circular_img = ImageOps.fit(img, size, centering=(0.5, 0.5))
        circular_img.putalpha(mask)

        border_img = Image.new("RGBA", (size[0] + border_w * 2, size[1] + border_w * 2), (0, 0, 0, 0))
        draw_b = ImageDraw.Draw(border_img)
        draw_b.ellipse((0, 0, border_img.width - 1, border_img.height - 1), fill=(255, 255, 255, 255))
        border_img.paste(circular_img, (border_w, border_w), circular_img)

        for x in range(border_img.width):
            for y in range(border_img.height):
                r, g, b, a = border_img.getpixel((x, y))
                if a > 0:
                    border_img.putpixel((x, y), (r, g, b, int(a * opacity)))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        border_img.save(output_path, "PNG")
        return output_path
    except Exception as exc:
        print(f"â ï¸ Lá»—i sinh avatar badge: {exc}")
        return None


def draw_ass(
    segments: list[dict[str, Any]],
    video_width: int,
    video_height: int,
    roi: dict[str, Any],
    path: Path,
    font_name: str = "Arial",
    font_size: int = 44,
    font_color: str = "#111111",
    box_color: str = "white",
    video_duration_sec: float | None = None
) -> None:
    """Váº½ phá»¥ Ä‘á» ASS Tháº» Bo GĂ³c ÄÆ¡n Tinh Gá»n (Sleek Compact Pill) tá»± Ä‘á»™ng khá»›p theo tá»a Ä‘á»™ OCR vĂ  nháº­n diá»‡n kĂªnh."""
    if video_height >= video_width:
        target_w, target_h = 1080, 1920
    else:
        target_w, target_h = 1920, 1080

    center_x = target_w // 2
    actual_font_size = font_size if font_size else 44
    
    # TĂ­nh tĂ¢m Y theo káº¿t quáº£ OCR tá»± Ä‘á»™ng quĂ©t Ä‘Æ°á»£c
    if roi and "yPercent" in roi:
        center_y = int(target_h * roi.get("yPercent", 75.0) / 100.0)
    elif roi and "yCenterPx" in roi:
        center_y = int(roi["yCenterPx"])
    else:
        center_y = int(target_h * 0.75)

    # Cáº¥u hĂ¬nh mĂ u sáº¯c theo Persona KĂªnh (Black Pill + Yellow Text vs White Pill + Dark Text)
    if box_color.lower() in ("black", "#111111", "#000000", "den", "Ä‘en"):
        box_ass_color = "&H00111111"
    else:
        box_ass_color = "&H00FFFFFF"

    if font_color.lower() in ("yellow", "#ffe600", "#ffea00", "#ffff00", "vang", "vĂ ng"):
        text_ass_color = "&H0000E6FF"  # VĂ ng rá»±c trong chuáº©n ASS (&H00BBGGRR&)
    elif font_color.startswith("&H"):
        text_ass_color = font_color
    else:
        text_ass_color = "&H00111111"

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {target_w}", f"PlayResY: {target_h}",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: CleanBox,Arial,10,{box_ass_color},&H00000000,{box_ass_color},{box_ass_color},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        f"Style: CleanText,{font_name},{actual_font_size},{text_ass_color},&H00000000,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    ]

    def ass_time(value_ms: int) -> str:
        hours, remainder = divmod(value_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1000)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds // 10:02d}"

    valid_segments = [s for s in segments if s.get("translatedTextVi", "").strip()]
    if not valid_segments:
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    # 1. TĂ­nh toĂ¡n tá»a Ä‘á»™ Y chuáº©n vĂ  kĂ­ch thÆ°á»›c thanh mask cá»‘ Ä‘á»‹nh rá»™ng rĂ£i (KhĂ´ng bao giá» trĂ n sub)
    detected_ys = [int(s.get("yCenterPx")) for s in valid_segments if s.get("yCenterPx")]
    common_cy = int(sorted(detected_ys)[len(detected_ys) // 2]) if detected_ys else center_y
    common_h = 180 # TÄƒng lĂªn 180px táº¡o padding trĂªn dÆ°á»›i thoĂ¡ng Ä‘Ă£ng
    common_w = 980 # Má»Ÿ rá»™ng tá»« 880 lĂªn 980px, chá»«a lá» an toĂ n 50px má»—i bĂªn viá»n 1080px
    box_left = center_x - (common_w // 2)
    box_top = common_cy - (common_h // 2)
    pill_vector = round_rect_ass(common_w, common_h, r=24)

    # 2. Gom nhĂ³m cĂ¡c khoáº£ng thoáº¡i liĂªn tá»¥c (náº¿u khoáº£ng nghá»‰ giá»¯a 2 cĂ¢u thoáº¡i <= 2.2s thĂ¬ giá»¯ nguyĂªn thanh mask)
    intervals: list[tuple[int, int]] = []
    curr_start = int(valid_segments[0].get("startMs", 0))
    curr_end = int(valid_segments[0].get("endMs", curr_start + 1500))

    for s in valid_segments[1:]:
        s_st = int(s.get("startMs", 0))
        s_et = int(s.get("endMs", s_st + 1500))
        if s_st - curr_end <= 2800: # Khoáº£ng cĂ¡ch nghá»‰ <= 2.8s -> Ná»‘i liá»n máº¡ch khĂ´ng chá»›p nhĂ¡y che hardsub
            curr_end = max(curr_end, s_et)
        else:
            intervals.append((curr_start, curr_end))
            curr_start = s_st
            curr_end = s_et
    intervals.append((curr_start, curr_end))

    # Layer 0: Váº½ cĂ¡c thanh mask cá»‘ Ä‘á»‹nh Ăªm Ă¡i vá»›i hiá»‡u á»©ng fade-in / fade-out (Zero Flickering)
    for i_st, i_et in intervals:
        if video_duration_sec is not None:
            max_limit_ms = int(video_duration_sec * 1000)
            if i_et > max_limit_ms:
                i_et = max_limit_ms
        st_str = ass_time(i_st)
        et_str = ass_time(i_et)
        lines.append(f"Dialogue: 0,{st_str},{et_str},CleanBox,,0,0,0,,{{\\fad(150,150)\\an7\\pos({box_left},{box_top})\\p1}}{pill_vector}{{\\p0}}")

    # Layer 1: Váº½ phá»¥ Ä‘á» tiáº¿ng Viá»‡t nháº£y cĂ¢u mÆ°á»£t mĂ  trĂªn ná»n thanh mask cá»‘ Ä‘á»‹nh
    for seg in valid_segments:
        st_ms = int(seg.get("startMs", 0))
        et_ms = int(seg.get("endMs", st_ms + 1500))
        vi_txt = seg.get("translatedTextVi", "").strip()
        if not vi_txt:
            continue

        words = vi_txt.split()
        if len(vi_txt) > 20 and len(words) > 3:
            mid = len(words) // 2
            formatted_vi = " ".join(words[:mid]) + r"\N" + " ".join(words[mid:])
        else:
            formatted_vi = vi_txt

        st_str = ass_time(st_ms)
        et_str = ass_time(et_ms)
        clean_text = formatted_vi.replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        lines.append(f"Dialogue: 1,{st_str},{et_str},CleanText,,0,0,0,,{{\\an5\\pos({center_x},{common_cy})}}{clean_text}")

    path.write_text("\n".join(lines), encoding="utf-8")
