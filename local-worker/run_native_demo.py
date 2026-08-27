"""Complete End-to-End Master Demo with VideoLingo Native Transcreation:
1. Demucs AI Clean BGM (Tách sạch 100% tiếng Trung, giữ nguyên nhạc nền Douyin).
2. Auto Subtitle ROI Detection (Tự động che phủ kín 100% phụ đề Trung Quốc).
3. VideoLingo-Style Native TikTok Transcreation (Chuyển ngữ bản xứ tự nhiên, bắt trend).
4. CapCut TikTok Voice (Mai BV421).
5. Subtitle & Voice Perfect Timing Sync.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from auto_roi import auto_detect_subtitle_roi
from llm_translator import translate_segments_native
from dubvi_worker import (
    Settings,
    draw_ass,
    draw_srt,
    extract_audio,
    ffprobe_dimensions,
    fit_voice,
    ocr_consensus_details,
    synthesize,
    transcribe,
    translate_to_vietnamese,
)


def run_master_native_pipeline(source_video: Path, output_dir: Path, max_segments: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings.from_env()
    print("=" * 70)
    print("🎬 KHỞI ĐỘNG PIPELINE MASTER DUBVI: CHUYỂN NGỮ BẢN XỨ (VIDEOLINGO ENGINE)")
    print(f"📁 Video gốc: {source_video.name}")
    print(f"🎙️ Giọng đọc: CapCut TikTok ({settings.capcut_voice})")
    print(f"🧠 Động cơ dịch: {settings.translation_engine.upper()} ({settings.llm_provider.upper()})")
    print("=" * 70)

    # 1. Trích xuất audio để Whisper ASR bóc tách câu thoại
    temp_full_wav = output_dir / "temp_full.wav"
    extract_audio(source_video, temp_full_wav)
    
    print("\n[1/6] 🧠 Whisper ASR bóc tách câu thoại...")
    all_segments = transcribe(temp_full_wav, settings.whisper_model, settings.device)
    segments = all_segments[:max_segments]
    
    last_seg_end = segments[-1]["endMs"] / 1000.0
    clip_duration = last_seg_end + 1.5
    print(f"✓ Chọn {len(segments)} câu thoại đầu tiên. Trích xuất clip video: {clip_duration:.1f}s")

    # 2. Cắt video mẫu & Tách BGM AI
    demo_source = output_dir / "demo_source_clip.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-ss", "0", "-i", str(source_video),
        "-t", str(clip_duration), "-c:v", "libx264", "-c:a", "aac", str(demo_source)
    ], check=True, capture_output=True)

    demo_wav = output_dir / "demo_audio.wav"
    extract_audio(demo_source, demo_wav)

    print("\n[2/6] 🎵 Tách sạch tiếng Trung & Giữ lại nhạc nền Douyin bằng Demucs AI...")
    demucs_dir = output_dir / "demucs_out"
    subprocess.run([
        sys.executable, "-m", "demucs.separate",
        "--two-stems=vocals", "-n", "htdemucs",
        "-o", str(demucs_dir), str(demo_wav)
    ], check=True, capture_output=True)
    clean_bgm_wav = demucs_dir / "htdemucs" / "demo_audio" / "no_vocals.wav"
    print(f"✓ Đã tách BGM sạch 100%: {clean_bgm_wav.name}")

    # 3. Auto ROI Detection
    print("\n[3/6] 🔍 Tự động quét Bounding Box phụ đề...")
    roi = auto_detect_subtitle_roi(demo_source)
    print(f"✓ Vùng che phụ đề tự động tính toán: Y=[{roi['yPercent']}% -> {roi['yPercent']+roi['heightPercent']}%], Width={roi['widthPercent']}%")

    # 4. RapidOCR & LLM Native Transcreation
    print("\n[4/6] 🔍 RapidOCR & VideoLingo LLM Transcreation...")
    for seg in segments:
        details = ocr_consensus_details(demo_source, seg, roi, samples=2)
        ocr_text = details.get("text")
        source_text = ocr_text if (ocr_text and details["confidence"] >= 60) else seg.get("asrTextZh", "")
        seg["ocrTextZh"] = ocr_text
        seg["sourceTextZh"] = source_text

    # Gọi LLM chuyển ngữ bản xứ cả kịch bản
    try:
        segments = translate_segments_native(
            segments,
            provider=settings.llm_provider,
            gemini_key=settings.gemini_api_key,
            deepseek_key=settings.deepseek_api_key,
            openai_key=settings.openai_api_key,
        )
    except Exception as exc:
        print(f"  ⚠️ LLM lỗi: {exc}, fallback sang dịch cơ bản...")

    for seg in segments:
        if not seg.get("translatedTextVi"):
            vi_text, _ = translate_to_vietnamese(seg["sourceTextZh"], seg["endMs"] - seg["startMs"])
            seg["translatedTextVi"] = vi_text
        print(f"  • [{seg['startMs']/1000:.1f}s -> {seg['endMs']/1000:.1f}s] Gốc: '{seg['sourceTextZh']}'\n    -> 🇻🇳 Bản xứ: '{seg['translatedTextVi']}'")

    # 5. Sinh Voice CapCut & Đồng bộ Subtitle
    print(f"\n[5/6] 🎙️ Sinh giọng đọc Mai CapCut ({settings.capcut_voice}) & Đồng bộ thời lượng...")
    voice_inputs: list[tuple[Path, int]] = []
    for idx, seg in enumerate(segments):
        raw_voice = output_dir / f"voice_{idx:02d}_raw.mp3"
        fitted_voice = output_dir / f"voice_{idx:02d}.mp3"
        slot_ms = seg["endMs"] - seg["startMs"]
        
        synthesize(seg["translatedTextVi"], settings.capcut_voice, raw_voice, settings)
        fit_ms, _ = fit_voice(raw_voice, fitted_voice, slot_ms, max_tempo=1.35)
        seg["voicePath"] = str(fitted_voice)
        seg["voiceDurationMs"] = fit_ms
        
        # ĐỒNG BỘ: Subtitle kết thúc đúng lúc giọng đọc vừa dứt
        seg["endMs"] = seg["startMs"] + fit_ms
        voice_inputs.append((fitted_voice, seg["startMs"]))
        print(f"  ✓ Voice {idx+1}: '{seg['translatedTextVi']}' (Đọc trong {fit_ms/1000:.2f}s)")

    # 6. Render Video MP4 với Clean BGM, Fixed Blur Box và Ass Subtitle
    print("\n[6/6] 🎞️ Render video chất lượng cao...")
    ass_path = output_dir / "subtitles_vi_master.ass"
    srt_path = output_dir / "subtitles_vi_master.srt"
    width, height = ffprobe_dimensions(demo_source)
    draw_ass(segments, width, height, roi, ass_path)
    draw_srt(segments, srt_path)

    w_px = int(width * roi["widthPercent"] / 100)
    h_px = int(height * roi["heightPercent"] / 100)
    x_px = int(width * roi["xPercent"] / 100)
    y_px = int(height * roi["yPercent"] / 100)

    ass_posix = ass_path.as_posix()
    if len(ass_posix) >= 2 and ass_posix[1] == ":":
        ass_posix = ass_posix[0] + "\\:" + ass_posix[2:]

    video_filter = (
        f"[0:v]split=2[base][ref];"
        f"[ref]crop={w_px}:{h_px}:{x_px}:{y_px},boxblur=26:26,drawbox=x=0:y=0:w={w_px}:h={h_px}:color=black@0.45:t=fill[blur];"
        f"[base][blur]overlay={x_px}:{y_px},subtitles='{ass_posix}'[video]"
    )

    cmd = ["ffmpeg", "-y", "-i", str(demo_source), "-i", str(clean_bgm_wav)]
    audio_labels = []
    for idx, (voice_path, offset_ms) in enumerate(voice_inputs, start=1):
        cmd.extend(["-i", str(voice_path)])
        input_idx = idx + 1
        audio_labels.append(f"[{input_idx}:a]adelay={offset_ms}|{offset_ms}[v{idx}]")

    all_v_tags = "".join(f"[v{i}]" for i in range(1, len(voice_inputs) + 1))
    if len(voice_inputs) == 1:
        mix_voice = f"{audio_labels[0]};[v1]anull[allvoice]"
    else:
        mix_voice = f"{';'.join(audio_labels)};{all_v_tags}amix=inputs={len(voice_inputs)}:duration=longest:normalize=0[allvoice]"

    audio_filter = f"{mix_voice};[1:a]volume=0.85[bgm];[bgm][allvoice]amix=inputs=2:duration=first:normalize=0[finalaudio]"
    final_output = output_dir / "douyin_master_native_complete.mp4"

    full_cmd = cmd + [
        "-filter_complex", f"{video_filter};{audio_filter}",
        "-map", "[video]", "-map", "[finalaudio]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(final_output)
    ]
    subprocess.run(full_cmd, check=True)

    print("\n" + "=" * 70)
    print("🎉 ĐÃ XUẤT BẢN MASTER HOÀN THIỆN ĐẦY ĐỦ!")
    print(f"📹 Video: {final_output}")
    print(f"📊 Dung lượng: {final_output.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    sample_vid = Path(r"C:\Users\vmath\Videos\douyin\7530236719954775306_video.mp4")
    out = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\demo-test")
    run_master_native_pipeline(sample_vid, out, max_segments=3)
