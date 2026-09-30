#!/usr/bin/env python3
"""DUBVI Video Rendering & CTA Compositing Module."""
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from dubvi_config import Settings, get_project_root
from dubvi_transcribe import ffprobe_duration, run
from dubvi_tts import synthesize


def ffprobe_dimensions(source: Path) -> tuple[int, int]:
    result = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(source)])
    width, height = result.stdout.strip().split("x")
    return int(width), int(height)


def ffprobe_has_audio(source: Path) -> bool:
    """Kiá»ƒm tra video gá»‘c cĂ³ chá»©a luá»“ng Ă¢m thanh (audio stream) hay khĂ´ng."""
    try:
        res = run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(source)])
        return bool(res.stdout.strip())
    except Exception:
        return False


def render_video(
    source: Path,
    output: Path,
    ass_path: Path,
    audio_segments: list[tuple[Path, int]],
    roi: dict[str, Any],
    audio_mode: str,
    clean_bgm_path: Path | None = None,
    logo_badge_path: Path | None = None,
    logo_position: str = "left",
    orig_audio_source: Path | None = None,
    max_duration_s: float | None = None,
    **kwargs: Any
) -> None:
    """Render video Master 1080p vá»›i Tháº» Bo GĂ³c ÄÆ¡n Tinh Gá»n, Logo KĂªnh Ä‘á»™ng vĂ  hĂ²a Ă¢m LUFS -14dB."""
    orig_w, orig_h = ffprobe_dimensions(source)
    
    # Chuáº©n hĂ³a Ä‘á»™ phĂ¢n giáº£i xuáº¥t ra Full HD 1080p (Chuáº©n TikTok/Reels/Shorts)
    if orig_h >= orig_w:
        scale_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    else:
        scale_filter = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"

    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:")
    video_filter = f"[0:v]{scale_filter},ass='{ass_escaped}'[v_sub]"
    
    use_clean_bgm = clean_bgm_path and clean_bgm_path.is_file()
    bg_audio_file = orig_audio_source if (orig_audio_source and orig_audio_source.is_file()) else source
    has_orig_audio = ffprobe_has_audio(bg_audio_file)
    
    command = ["ffmpeg", "-y", "-i", str(source)]
    curr_offset = 1

    if bg_audio_file != source and has_orig_audio:
        command.extend(["-i", str(bg_audio_file)])
        bg_audio_idx = curr_offset
        curr_offset += 1
    else:
        bg_audio_idx = 0

    clean_bgm_idx = None
    if use_clean_bgm:
        clean_bgm_idx = curr_offset
        command.extend(["-stream_loop", "-1", "-i", str(clean_bgm_path)])
        curr_offset += 1
        
    audio_labels: list[str] = []
    for index, (audio_path, offset_ms) in enumerate(audio_segments, start=curr_offset):
        command.extend(["-i", str(audio_path)])
        idx_name = index - curr_offset + 1
        audio_labels.append(f"[{index}:a]adelay={offset_ms}|{offset_ms}[v{idx_name}]")

    # Xá»­ lĂ½ chĂ¨n Logo Avatar KĂªnh Ä‘á»™ng
    has_logo = logo_badge_path and Path(logo_badge_path).exists()
    if has_logo:
        logo_idx = curr_offset + len(audio_segments)
        command.extend(["-i", str(logo_badge_path)])
        if logo_position == "right":
            video_filter += f";[v_sub][{logo_idx}:v]overlay=x=main_w-overlay_w-45:y=95:format=auto[video]"
        else:
            video_filter += f";[v_sub][{logo_idx}:v]overlay=x=45:y=95:format=auto[video]"
    else:
        video_filter = video_filter.replace("[v_sub]", "[video]")

    filters = [video_filter]
    if audio_labels:
        inputs = "".join(f"[v{i}]" for i in range(1, len(audio_segments) + 1))
        if len(audio_segments) == 1:
            mix_voice = f"{audio_labels[0]};[v1]anull[voice]"
        else:
            mix_voice = f"{';'.join(audio_labels)};{inputs}amix=inputs={len(audio_segments)}:duration=longest:normalize=0[voice]"

        if use_clean_bgm:
            # DĂ¹ng nháº¡c ná»n BGM Viral (volume 0.18) + fade-in 0.5s + chuáº©n hĂ³a Ă¢m lÆ°á»£ng Mobile LUFS -14dB
            filters.append(f"{mix_voice};[{clean_bgm_idx}:a]volume=0.18,afade=t=in:st=0:d=0.5[bgm];[voice][bgm]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio_pre]")
        elif audio_mode in ("duck", "ducking") and has_orig_audio:
            # Triá»‡t tiĂªu giá»ng Trung 99% (volume 0.01 + lowpass 2000Hz) + giá»ng CapCut Ä‘Ä©nh Ä‘áº¡c
            filters.append(f"{mix_voice};[{bg_audio_idx}:a]volume=0.01,lowpass=f=2000[bg];[bg][voice]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio_pre]")
        elif audio_mode == "keep" and has_orig_audio:
            filters.append(f"[{bg_audio_idx}:a]loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio_pre]")
        else:
            filters.append(f"{mix_voice};[voice]apad,loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio_pre]")
    else:
        if has_orig_audio:
            filters.append("[0:a]loudnorm=I=-14:TP=-1.5:LRA=11[finalaudio_pre]")
        else:
            filters.append("aevalsrc=0:d=1[finalaudio_pre]")

    # Ăp dá»¥ng audio fade-out 0.5s á»Ÿ Ä‘uĂ´i náº¿u cĂ³ giá»›i háº¡n thá»i lÆ°á»£ng (cáº¯t sáº¡ch Outro CTA Douyin)
    if max_duration_s is not None and max_duration_s > 0:
        fade_start = max(0.0, max_duration_s - 0.5)
        fade_dur = min(0.5, max_duration_s)
        filters.append(f"[finalaudio_pre]afade=t=out:st={fade_start:.3f}:d={fade_dur:.3f}[finalaudio]")
    else:
        filters.append("[finalaudio_pre]anull[finalaudio]")

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[video]", "-map", "[finalaudio]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-threads", "2", "-crf", "20",
        "-c:a", "aac"
    ])
    if max_duration_s is not None and max_duration_s > 0:
        command.extend(["-t", f"{max_duration_s:.3f}"])
    command.extend(["-movflags", "+faststart", str(output)])
    run(command)


CTA_VARIATIONS = {
    "page_goc_tro_bat_on": [
        "Xem cuá»‘n tháº¿ nĂ y mĂ  tiáº¿c cĂ¡i nĂºt follow háº£?!",
        "KhĂ´ng follow láº¹ lĂ  trá» báº¥t á»•n giá»‘ng váº­y Ä‘Ă³ nha!",
        "Xem chĂ¹a khĂ´ng follow lĂ  tá»‘i trá» báº¥t á»•n nha!"
    ],
    "page_giai_cuu_chuong_lon": [
        "KhĂ´ng follow coi chá»«ng phĂ²ng thĂ nh chuá»“ng lá»£n nha!",
        "Xem cuá»‘n tháº¿ nĂ y mĂ  tiáº¿c nĂºt follow háº£ trá»i?!",
        "Báº¥m follow láº¹ Ä‘i, tiáº¿c gĂ¬ má»™t cĂ¡i click háº£?!"
    ]
}


def get_raw_cta_video_path(channel_profile: str) -> Path | None:
    """Láº¥y Ä‘Æ°á»ng dáº«n video CTA gá»‘c 3s cá»§a kĂªnh."""
    root = get_project_root()
    is_goc_tro = "goc_tro" in channel_profile.lower()
    fname = "CTA-goc tro bat on.mp4" if is_goc_tro else "CTA-giai cuu chuong lon.mp4"
    candidates = [
        root / "assets" / "cta" / fname,
        root / "assets" / fname
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def build_dynamic_cta_clip(
    channel_profile: str,
    voice_name: str,
    settings: Settings,
    output_path: Path,
    variation_idx: int | None = None
) -> tuple[Path | None, float]:
    """Táº¡o video CTA Ä‘á»™ng: DĂ¹ng chung Voice cá»§a video chĂ­nh vĂ  tá»± Ä‘á»™ng cáº¯t video 3s khá»›p Ä‘Ăºng thá»i lÆ°á»£ng Voice (2.0s - 2.5s)."""
    raw_cta = get_raw_cta_video_path(channel_profile)
    if not raw_cta:
        return None, 0.0

    key = "page_goc_tro_bat_on" if "goc_tro" in channel_profile.lower() else "page_giai_cuu_chuong_lon"
    var_list = CTA_VARIATIONS.get(key, CTA_VARIATIONS["page_giai_cuu_chuong_lon"])
    if variation_idx is not None:
        cta_text = var_list[variation_idx % len(var_list)]
    else:
        cta_text = random.choice(var_list)

    temp_audio = output_path.parent / f"temp_cta_voice_{output_path.stem}.mp3"
    synthesize(cta_text, voice_name, temp_audio, settings)
    if not temp_audio.is_file() or temp_audio.stat().st_size == 0:
        return None, 0.0

    dur_voice = ffprobe_duration(temp_audio)
    # Giá»¯ cháº¯c cháº¯n trong khoáº£ng 1.8s - 2.2s theo chuáº©n Founder yĂªu cáº§u
    target_dur = max(1.8, min(dur_voice + 0.15, 2.2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-t", f"{target_dur:.3f}", "-i", str(raw_cta),
        "-i", str(temp_audio),
        "-filter_complex",
        f"[0:v]scale=1080:1920,fps=30[v];[1:a]apad,atrim=0:{target_dur:.3f},loudnorm=I=-14:TP=-1.5:LRA=11[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-movflags", "+faststart",
        str(output_path)
    ]
    run(cmd)
    temp_audio.unlink(missing_ok=True)
    
    actual_dur = ffprobe_duration(output_path) if output_path.is_file() else target_dur
    return output_path, actual_dur


def splice_cta_into_video(
    main_video: Path,
    cta_video: Path,
    output_path: Path,
    insert_ratio: float = 0.70,
    insert_sec: float | None = None
) -> Path:
    """ChĂ¨n clip CTA vĂ o video chĂ­nh táº¡i thá»i Ä‘iá»ƒm ~70% thá»i lÆ°á»£ng (Pre-Reveal Golden Point)."""
    if not cta_video or not cta_video.is_file():
        shutil.copy2(main_video, output_path)
        return output_path

    total_dur = ffprobe_duration(main_video)
    if insert_sec is not None:
        actual_insert = insert_sec
    else:
        actual_insert = round(total_dur * insert_ratio, 2)

    # Äáº£m báº£o khĂ´ng chĂ¨n lá»‘ vĂ o sĂ¡t mĂ©p Ä‘uĂ´i video (chá»«a Ă­t nháº¥t 2 giĂ¢y)
    if actual_insert >= total_dur - 2.0:
        actual_insert = max(0.0, total_dur - 2.0)

    if actual_insert <= 0.5 or actual_insert >= total_dur:
        # Ná»‘i Outro: main + cta
        filter_str = (
            "[0:v]scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v0];"
            "[0:a]aresample=44100,asetpts=PTS-STARTPTS[a0];"
            "[1:v]scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v1];"
            "[1:a]aresample=44100,asetpts=PTS-STARTPTS[a1];"
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(main_video),
            "-i", str(cta_video),
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-threads", "2",
            "-c:a", "aac", "-movflags", "+faststart",
            str(output_path)
        ]
    else:
        # ChĂ¨n Mid-roll táº¡i 70% thá»i lÆ°á»£ng (Pre-Reveal):
        filter_str = (
            f"[0:v]trim=0:{actual_insert},scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v1];"
            f"[0:a]atrim=0:{actual_insert},aresample=44100,asetpts=PTS-STARTPTS[a1];"
            f"[1:v]scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v2];"
            f"[1:a]aresample=44100,asetpts=PTS-STARTPTS[a2];"
            f"[0:v]trim=start={actual_insert},scale=1080:1920,setsar=1,fps=30,setpts=PTS-STARTPTS[v3];"
            f"[0:a]atrim=start={actual_insert},aresample=44100,asetpts=PTS-STARTPTS[a3];"
            f"[v1][a1][v2][a2][v3][a3]concat=n=3:v=1:a=1[v][a]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(main_video),
            "-i", str(cta_video),
            "-filter_complex", filter_str,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20", "-threads", "2",
            "-c:a", "aac", "-movflags", "+faststart",
            str(output_path)
        ]

    run(cmd)
    return output_path
