#!/usr/bin/env python3
"""DUBVI Automated Pipeline Watchdog (Giai đoạn 2).

Chạy ngầm giám sát thư mục video sạch từ Giai đoạn 1 (video reup raw/...).
Ngay khi có video mới kèm file .meta.json, tự động bốc chạy trọn gói:
Bóc tách -> Dịch Gemini 2.5 Flash -> 8 Hook -> Caption SEO -> CapCut TTS -> Subtitle Safe Zone 22% -> LUFS -14dB Master -> Thumbnail Cover.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dubvi_worker import (
    Settings,
    scan_multi_channel_raw,
    extract_audio,
    transcribe,
    draw_ass,
    draw_srt,
    render_video,
    synthesize,
    fit_voice,
    clamp_and_bridge_audio_gaps,
    ffprobe_dimensions,
    generate_video_thumbnail,
)
from auto_roi import auto_detect_subtitle_roi, scan_silent_subtitles, merge_asr_and_ocr_segments
from llm_translator import translate_segments_native, generate_viral_hooks, generate_social_post_caption

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("dubvi-watchdog")


def process_single_video(item: dict, settings: Settings) -> bool:
    source: Path = item["video_path"]
    job_dir = settings.output_dir / f"{source.stem}-full"
    job_dir.mkdir(parents=True, exist_ok=True)
    master_mp4 = job_dir / f"{source.stem}_1080p_master_vi.mp4"
    lock_file = job_dir / ".processing.lock"

    if master_mp4.is_file() and master_mp4.stat().st_size > 1024 * 1024:
        return False  # Đã hoàn thành trước đó

    if lock_file.is_file():
        if time.time() - lock_file.stat().st_mtime < 1800:
            return False

    lock_file.write_text(str(time.time()), encoding="utf-8")
    channel_profile = item.get("channel_profile")
    channel_name = item.get("channel_name", "Đa Kênh")
    default_voice = item.get("default_voice", "BV421_vivn_streaming")

    logger.info(f"🚀 [BẮT ĐẦU AUTO-PIPELINE] {source.name} | Kênh: {channel_name} | Platform: {item.get('target_platform')}")

    try:
        # 1. Trích xuất audio WAV
        audio_wav = job_dir / "source_full.wav"
        if not audio_wav.is_file():
            extract_audio(source, audio_wav)

        # 2. Whisper ASR + RapidOCR
        logger.info("  1/6 🗣️ Đang nhận diện tiếng Trung (Whisper ASR)...")
        asr_segs = transcribe(audio_wav, "tiny", "cpu")
        for s in asr_segs:
            s["sourceTextZh"] = s.get("asrTextZh", "")
            s["subType"] = "🎙️ Thoại"

        roi = auto_detect_subtitle_roi(source)
        try:
            silent_segs, mask_only = scan_silent_subtitles(source, roi, asr_segs, 1.4, 3)
            for s in silent_segs:
                s["subType"] = "📺 Sub Câm"
            segs = merge_asr_and_ocr_segments(asr_segs, silent_segs)
        except Exception:
            segs = asr_segs
            mask_only = []

        # 3. Tách BGM sạch bằng Demucs AI
        clean_bgm_wav = job_dir / "clean_bgm.wav"
        if not clean_bgm_wav.is_file():
            logger.info("  2/6 🎵 Đang tách sạch tiếng Trung & giữ nhạc nền (Demucs AI)...")
            demucs_dir = job_dir / "demucs_out"
            try:
                import subprocess
                subprocess.run([
                    sys.executable, "-m", "demucs.separate",
                    "--two-stems=vocals", "-n", "htdemucs",
                    "-o", str(demucs_dir), str(audio_wav)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                demucs_out = demucs_dir / "htdemucs" / audio_wav.stem / "no_vocals.wav"
                if demucs_out.is_file():
                    import shutil
                    shutil.copy2(demucs_out, clean_bgm_wav)
            except Exception as e:
                logger.warning(f"  ⚠️ Demucs fallback ({e})")

        # 4. Dịch Gemini 2.5 Flash + Ma trận 8 Hook + Caption SEO
        logger.info(f"  3/6 🧠 Gemini 2.5 Flash chuyển ngữ Persona '{channel_name}'...")
        segs = translate_segments_native(segs, "gemini", channel_profile=channel_profile)
        
        # Tạo 8 Hook và gán Hook đỉnh nhất vào câu #1
        hooks = generate_viral_hooks(segs, channel_profile=channel_profile)
        if hooks and len(segs) > 0:
            segs[0]["translatedTextVi"] = hooks[0]["text"]
            logger.info(f"  ✓ Auto-Hook 3s: \"{hooks[0]['text']}\"")

        # Lưu segments.json
        seg_file = job_dir / "segments.json"
        seg_file.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")

        # Tạo Caption SEO
        caption_data = generate_social_post_caption(segs, channel_profile=channel_profile)
        if caption_data and caption_data.get("full_post"):
            caption_file = job_dir / f"{source.stem}_caption.txt"
            caption_file.write_text(caption_data["full_post"], encoding="utf-8")
            logger.info(f"  ✓ Đã xuất Caption & 7 Hashtags: {caption_file.name}")

        # 5. CapCut TTS Voiceover
        logger.info(f"  4/6 🎙️ Tổng hợp {len(segs)} câu thoại CapCut TTS ({default_voice})...")
        raw_clips = []
        for idx, seg in enumerate(segs):
            raw_v = job_dir / f"voice_{idx:03d}_raw.mp3"
            fit_v = job_dir / f"voice_{idx:03d}.mp3"
            slot_ms = max(800, seg.get("endMs", 0) - seg.get("startMs", 0))
            vi_text = (seg.get("translatedTextVi") or "...").strip()
            synthesize(vi_text, default_voice, raw_v, settings)
            fit_ms, _ = fit_voice(raw_v, fit_v, slot_ms, max_tempo=1.35)
            raw_clips.append({
                "segment": seg,
                "fitted_voice": fit_v,
                "startMs": seg["startMs"],
                "endMs": seg["endMs"],
                "durationSec": fit_ms / 1000.0,
            })

        # Áp dụng thuật toán Smart Gap Clamping để khống chế khoảng câm <= 600ms
        clamped_clips = clamp_and_bridge_audio_gaps(raw_clips, max_gap_ms=600, min_pause_ms=250)
        voice_inputs = []
        for c in clamped_clips:
            seg = c["segment"]
            seg["startMs"] = c["startMs"]
            seg["endMs"] = c["endMs"]
            voice_inputs.append((c["fitted_voice"], c["startMs"]))

        # 6. Render Master 1080p Single White Pill Card + LUFS -14dB + Thumbnail Cover
        logger.info("  5/6 🎞️ Render FFmpeg 1080p Master (Single White Pill Card & LUFS -14dB)...")
        ass_path = job_dir / f"{source.stem}_1080p.ass"
        srt_path = job_dir / f"{source.stem}_1080p.srt"
        draw_ass(segs, 1080, 1920, roi, ass_path, font_size=48)
        draw_srt(segs, srt_path)

        render_video(
            source, master_mp4, ass_path, voice_inputs, roi,
            audio_mode="duck", clean_bgm_path=clean_bgm_wav if clean_bgm_wav.is_file() else None,
            segments=segs, mask_only_intervals=mask_only
        )

        # 7. Thumbnail Cover
        cover_path = job_dir / f"{source.stem}_cover.jpg"
        best_hook = segs[0].get("translatedTextVi", "Review Đỉnh Chóp!")
        generate_video_thumbnail(master_mp4, best_hook, cover_path, 2.0)
        logger.info(f"  6/6 🖼️ Đã tạo ảnh bìa Thumbnail: {cover_path.name}")

        lock_file.unlink(missing_ok=True)
        logger.info(f"🎉 [XUẤT BẢN THÀNH CÔNG] {master_mp4.name} ({master_mp4.stat().st_size/1024/1024:.2f} MB)\n")
        return True
    except Exception as exc:
        logger.error(f"❌ Lỗi xử lý {source.name}: {exc}", exc_info=True)
        lock_file.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="DUBVI Background Watchdog Pipeline")
    parser.add_argument("--watch", action="store_true", help="Chạy chế độ giám sát liên tục (Polling)")
    parser.add_argument("--interval", type=int, default=5, help="Chu kỳ quét (giây)")
    args = parser.parse_args()

    settings = Settings.from_env()
    logger.info("🟢 DUBVI Pipeline Watchdog đã khởi động.")
    logger.info(f"📁 Thư mục xuất bản: {settings.output_dir}")

    while True:
        items = scan_multi_channel_raw()
        pending = []
        for item in items:
            source = item["video_path"]
            job_dir = settings.output_dir / f"{source.stem}-full"
            master_mp4 = job_dir / f"{source.stem}_1080p_master_vi.mp4"
            if not (master_mp4.is_file() and master_mp4.stat().st_size > 1024 * 1024):
                pending.append(item)

        if pending:
            logger.info(f"🔍 Phát hiện {len(pending)} video mới cần xử lý tự động.")
            for item in pending:
                process_single_video(item, settings)

        if not args.watch:
            break

        time.sleep(max(3, args.interval))


if __name__ == "__main__":
    main()
