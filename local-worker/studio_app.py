"""🎬 DUBVI STUDIO - Local Web & Desktop Studio App (NiceGUI)
Tích hợp toàn bộ Golden Standard Pipeline:
- Meta Demucs AI Clean BGM
- OpenAI Whisper ASR & RapidOCR
- Google Gemini 2.5 Flash Native Transcreation
- CapCut TTS Voiceover (Mai BV421, Ban Mai, Minh Quang...)
- Frosted Glassmorphism Subtitle Bar (1080p Full HD)
- AG Grid Interactive Bilingual Script Editor
- HTML5 Video Player with Live WebSockets
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List
from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from nicegui import app, ui

from file_picker import open_windows_file_dialog

# Import local worker modules
from dubvi_worker import (
    Settings,
    compute_mask_intervals,
    draw_ass,
    draw_srt,
    extract_audio,
    ffprobe_dimensions,
    fit_voice,
    synthesize,
    transcribe,
)
from auto_roi import auto_detect_subtitle_roi
from llm_translator import translate_segments_native
from gemini_pool import gemini_pool

# Cấu hình môi trường
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
settings = Settings.from_env()

# Thư mục làm việc
DEFAULT_RAW_DIR = Path(r"C:\Users\vmath\Downloads\video douyin raw")
DEFAULT_OUT_DIR = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output")
DEFAULT_RAW_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mount các thư mục video trên máy để phát trực tiếp qua HTTP Range streaming
app.add_static_files("/output", str(DEFAULT_OUT_DIR))
app.add_static_files("/raw", str(DEFAULT_RAW_DIR))
app.add_static_files("/downloads", r"C:\Users\vmath\Downloads")
app.add_static_files("/videos", r"C:\Users\vmath\Videos")

@app.post("/api/upload_video")
async def api_upload_video(file: UploadFile = File(...)):
    try:
        target = DEFAULT_RAW_DIR / file.filename
        content = await file.read()
        target.write_bytes(content)
        state["selected_video"] = target
        return {"status": "ok", "path": str(target), "name": file.filename}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Trạng thái ứng dụng
state = {
    "selected_video": None,
    "segments": [],
    "clean_bgm_path": None,
    "current_master_mp4": None,
    "is_processing": False,
    "progress_val": 0.0,
    "status_text": "Sẵn sàng",
    "selected_voice": settings.capcut_voice,
    "font_name": "Arial",
    "font_size": 56,
    "font_color": "#FFFFFF",
    "roi": {"xPercent": 2.0, "yPercent": 66.0, "widthPercent": 96.0, "heightPercent": 9.8, "blurPx": 24},
    "logs": ["🟢 Hệ thống Dubvi Studio v4.5 sẵn sàng."],
}

VOICE_OPTIONS = {
    "BV421_vivn_streaming": "Mai (CapCut - Nữ truyền cảm, review Douyin)",
    "BV007_streaming": "Minh Quang (CapCut - Nam trầm ấm, tự nhiên)",
    "BV001_streaming": "Ngọc Mai (CapCut - Nữ ngọt ngào, nhẹ nhàng)",
    "BV004_streaming": "Hải Đăng (CapCut - Nam review, kể chuyện)",
    "vi-VN-HoaiMyNeural": "Hoài My (Edge-TTS - Nữ chuẩn phát thanh)",
    "vi-VN-NamMinhNeural": "Nam Minh (Edge-TTS - Nam chuyên nghiệp)",
}

COLOR_OPTIONS = {
    "#FFFFFF": "⚪ Trắng Tinh Khôi (Chuẩn)",
    "#FFE600": "🟡 Vàng Nổi Bật (Trend Douyin/TikTok)",
    "#00FFFF": "🌐 Xanh Cyan Neon",
    "#00FF7F": "🟢 Xanh Lá Sáng",
    "#FF69B4": "🌸 Hồng Pastel",
    "#FFA500": "🟠 Cam Năng Động",
}


def scan_raw_videos() -> List[Dict[str, Any]]:
    """Quét toàn bộ video mp4 trong thư mục raw (kể cả thư mục con) và Downloads, sắp xếp mới nhất lên đầu."""
    raw_files = []
    seen = set()
    
    # 1. Quét sâu trong DEFAULT_RAW_DIR
    if DEFAULT_RAW_DIR.is_dir():
        for p in list(DEFAULT_RAW_DIR.rglob("*.mp4")) + list(DEFAULT_RAW_DIR.rglob("*.mov")) + list(DEFAULT_RAW_DIR.rglob("*.mkv")):
            if p.is_file() and not p.name.startswith("_") and str(p.resolve()) not in seen:
                seen.add(str(p.resolve()))
                raw_files.append(p)
                
    # 2. Quét Downloads & Videos ngoài
    for d in [Path(r"C:\Users\vmath\Downloads"), Path(r"C:\Users\vmath\Videos")]:
        if d.is_dir():
            for p in list(d.glob("*.mp4")) + list(d.glob("*.mov")) + list(d.glob("*.mkv")):
                if p.is_file() and not p.name.startswith("_") and str(p.resolve()) not in seen:
                    seen.add(str(p.resolve()))
                    raw_files.append(p)
                    
    # Sắp xếp video mới nhất lên đầu tiên theo thời gian sửa đổi (mtime)
    raw_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    videos = []
    for p in raw_files:
        size_mb = p.stat().st_size / (1024 * 1024)
        parent_title = p.parent.name if p.parent.name != DEFAULT_RAW_DIR.name else ""
        label_name = f"{parent_title}/{p.name}" if parent_title else p.name
        
        # Nhận diện trạng thái đã render hay là video mới
        job_dir = DEFAULT_OUT_DIR / f"{p.stem}-full"
        master_mp4 = job_dir / f"{p.stem}_1080p_master_vi.mp4"
        is_rendered = master_mp4.is_file() and master_mp4.stat().st_size > 1024 * 1024
        status_tag = "✅ [ĐÃ RENDER]" if is_rendered else "✨ [MỚI]"

        videos.append({
            "name": f"{status_tag} {label_name}",
            "path": str(p),
            "size": f"{size_mb:.1f} MB",
            "is_rendered": is_rendered
        })
    return videos


@ui.page("/")
def main_page():
    ui.dark_mode(True)
    ui.colors(primary="#6366f1", secondary="#4f46e5", accent="#ec4899", dark="#0f172a")

    # Header Studio
    with ui.header().classes("bg-slate-900 border-b border-slate-800 px-6 py-3 items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("movie", size="2rem").classes("text-indigo-400")
            ui.label("DUBVI STUDIO").classes("text-xl font-black tracking-wider text-white")
            ui.badge("v4.5 AI Studio", color="indigo").classes("text-xs font-bold")
        with ui.row().classes("items-center gap-4"):
            status_label = ui.label("🟢 Hệ thống sẵn sàng").classes("text-sm text-slate-400 font-medium")
            ui.button("Mở thư mục Video", icon="folder", on_click=lambda: os.startfile(str(DEFAULT_OUT_DIR))).props("flat color=white size=sm")

    # Layout 3 Cột Studio
    with ui.row().classes("w-full h-[calc(100vh-64px)] p-4 gap-4 no-wrap bg-slate-950 text-slate-100"):

        # ----------------------------------------------------
        # CỘT 1: Quản lý Video & Cài đặt (Width: 24%)
        # ----------------------------------------------------
        with ui.column().classes("w-1/4 h-full bg-slate-900/80 rounded-2xl p-3 border border-slate-800 flex flex-col justify-between shadow-xl backdrop-blur-md overflow-hidden"):
            # Vùng nội dung cài đặt (Xếp thẻ liền mạch, đẹp mắt)
            with ui.column().classes("w-full gap-2.5"):
                # CARD 1: VIDEO NGUỒN (TỐI GIẢN - 1 CÁCH DUY NHẤT)
                with ui.column().classes("w-full bg-slate-800/40 p-2.5 rounded-xl border border-slate-800/80 gap-2"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("📁 VIDEO NGUỒN").classes("text-[11px] font-extrabold tracking-wider text-indigo-400 uppercase")
                        ui.button(icon="refresh", on_click=lambda: refresh_video_list()).props("round flat size=xs color=indigo").tooltip("Quét lại video")

                    video_select = ui.select(
                        options={},
                        label="Chọn video Douyin",
                        on_change=lambda e: select_video(e.value) if e.value else None
                    ).classes("w-full bg-slate-900/60 rounded-lg text-xs").props("dense options-dense")

                    ui.upload(
                        label="📂 BẤM ĐỂ CHỌN VIDEO TỪ MÁY...",
                        auto_upload=True,
                        max_files=1,
                        on_upload=lambda e: handle_browser_upload(e)
                    ).props("dense flat bordered accept='video/*,.mp4,.mov,.mkv,.avi' color=indigo").classes("w-full text-xs font-bold bg-slate-900/40 rounded-lg border border-indigo-500/40 shadow-sm")

                    selected_file_label = ui.label("🎬 Chưa chọn video").classes("text-[11px] text-indigo-300 font-bold truncate w-full px-1")

                # CARD 2: GIỌNG ĐỌC
                with ui.column().classes("w-full bg-slate-800/40 p-2.5 rounded-xl border border-slate-800/80 gap-1"):
                    ui.label("🎙️ GIỌNG ĐỌC CAPCUT").classes("text-[11px] font-extrabold tracking-wider text-indigo-400 uppercase")
                    voice_select = ui.select(
                        options=VOICE_OPTIONS,
                        value=state["selected_voice"],
                        label="Chọn giọng đọc",
                        on_change=lambda e: update_voice(e.value)
                    ).classes("w-full bg-slate-900/60 rounded-lg text-xs").props("dense options-dense")
                    sample_audio_player = ui.audio("").classes("hidden")

                # CARD 3: CÀI ĐẶT & XEM TRƯỚC PHỤ ĐỀ (GOM CHUNG 1 NƠI)
                with ui.column().classes("w-full bg-slate-800/40 p-2.5 rounded-xl border border-slate-800/80 gap-2"):
                    with ui.row().classes("w-full items-center justify-between"):
                        ui.label("🔤 PHỤ ĐỀ & KÍNH MỜ").classes("text-[11px] font-extrabold tracking-wider text-indigo-400 uppercase")
                        roi_badge = ui.badge("AI Mask: Y=68%", color="purple").classes("text-[10px] font-bold")

                    with ui.row().classes("w-full gap-1.5 no-wrap"):
                        font_select = ui.select(
                            options={
                                "Arial": "Arial",
                                "Segoe UI": "Segoe UI",
                                "Tahoma": "Tahoma",
                                "Trebuchet MS": "Trebuchet MS",
                                "Verdana": "Verdana",
                                "Impact": "Impact",
                                "Calibri": "Calibri",
                            },
                            value=state["font_name"],
                            label="Font chữ",
                            on_change=lambda e: update_font_name(e.value)
                        ).classes("w-1/2 bg-slate-900/60 rounded-lg text-xs").props("dense options-dense")

                        color_select = ui.select(
                            options=COLOR_OPTIONS,
                            value=state["font_color"],
                            label="Màu chữ",
                            on_change=lambda e: update_font_color(e.value)
                        ).classes("w-1/2 bg-slate-900/60 rounded-lg text-xs").props("dense options-dense")

                    with ui.row().classes("w-full items-center justify-between gap-1.5 px-2 py-0.5 bg-slate-900/40 rounded-lg border border-slate-800/60"):
                        ui.label("Cỡ chữ:").classes("text-[11px] font-semibold text-slate-300")
                        ui.slider(
                            min=36, max=76, step=2, value=state["font_size"],
                            on_change=lambda e: update_font_size(e.value, font_size_badge)
                        ).props("dense color=indigo").classes("flex-1")
                        font_size_badge = ui.badge(f"{state['font_size']} px", color="indigo").classes("text-[10px] font-bold")

                    # Khung Live Preview ngay trong Card này
                    with ui.element('div').classes("w-full rounded-lg border border-slate-700/80 bg-slate-950 p-2 shadow-inner"):
                        with ui.element('div').classes("w-full rounded flex items-center justify-center p-1.5 text-center bg-white/15 backdrop-blur-md border border-white/20"):
                            preview_text_label = ui.label("Món đồ siêu mê góc nhỏ nè!").style(
                                f"font-family: '{state['font_name']}', sans-serif; font-size: {int(state['font_size'] * 0.28)}px; font-weight: 700; color: {state['font_color']}; text-shadow: -1.2px -1.2px 0 #000, 1.2px -1.2px 0 #000, -1.2px 1.2px 0 #000, 1.2px 1.2px 0 #000, 0 2px 4px rgba(0,0,0,0.8);"
                            )

                    btn_preview_3s = ui.button(
                        "⚡ Xem Thử Mask & Cỡ Chữ (3s)",
                        icon="visibility",
                        on_click=lambda: preview_3s_mask()
                    ).classes("w-full bg-indigo-600/60 hover:bg-indigo-600 text-white font-bold py-1.5 rounded-lg shadow text-xs transition-all")

            # NÚT ACTION 1-CLICK Ở CUỐI CÙNG
            with ui.column().classes("w-full pt-2"):
                btn_auto_all = ui.button(
                    "🚀 1-CLICK TỰ ĐỘNG TOÀN BỘ",
                    icon="bolt",
                    on_click=lambda: run_full_pipeline()
                ).classes("w-full bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 hover:opacity-95 text-white font-black py-3 rounded-xl shadow-xl text-xs tracking-wider uppercase transition-all")

        # ----------------------------------------------------
        # CỘT 2: Bảng Kịch Bản Song Ngữ AG Grid (Width: 46%)
        # ----------------------------------------------------
        with ui.column().classes("w-1/2 h-full bg-slate-900/80 rounded-2xl p-4 border border-slate-800 flex flex-col shadow-xl backdrop-blur-md"):
            with ui.row().classes("w-full items-center justify-between mb-2"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("translate", size="1.2rem").classes("text-indigo-400")
                    ui.label("📝 BẢNG KỊCH BẢN SONG NGỮ (AG GRID)").classes("text-xs font-extrabold tracking-wider text-indigo-400 uppercase")
                with ui.row().classes("items-center gap-2"):
                    ui.button("▶️ Nghe thử câu chọn", icon="play_arrow", on_click=lambda: preview_selected_sentence()).props("flat size=xs color=emerald").classes("text-xs font-bold bg-emerald-950/40 border border-emerald-800/40 px-2 py-1 rounded")
                    segment_count_badge = ui.badge("0 câu", color="slate").classes("text-xs font-bold")

            ui.label("💡 Click đúp vào cột 'Tiếng Việt' để chỉnh sửa trực tiếp câu từ/tiếng lóng trước khi Render.").classes("text-xs text-slate-400 italic mb-2")

            grid = ui.aggrid({
                "columnDefs": [
                    {"headerName": "#", "field": "position", "width": 55, "sortable": True},
                    {"headerName": "Thời gian", "field": "timeline", "width": 110},
                    {"headerName": "🇨🇳 Tiếng Trung (ASR/OCR)", "field": "sourceTextZh", "width": 210, "wrapText": True, "autoHeight": True},
                    {"headerName": "🇻🇳 Tiếng Việt (Click sửa)", "field": "translatedTextVi", "editable": True, "flex": 1, "wrapText": True, "autoHeight": True, "cellClass": "text-emerald-400 font-semibold"},
                ],
                "rowData": [],
                "rowSelection": "single",
                "stopEditingWhenCellsLoseFocus": True,
            }).classes("w-full flex-grow rounded-xl overflow-hidden border border-slate-800 bg-slate-950")

            grid.on("cellValueChanged", lambda e: on_cell_edited(e.args))

        # ----------------------------------------------------
        # CỘT 3: Video Player & Tiến Trình Pipeline (Width: 30%)
        # ----------------------------------------------------
        with ui.column().classes("w-1/3 h-full bg-slate-900/80 rounded-2xl p-4 border border-slate-800 flex flex-col justify-between shadow-xl backdrop-blur-md overflow-hidden"):
            with ui.column().classes("w-full flex-grow overflow-y-auto pr-1 gap-2.5"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-1.5 truncate flex-1"):
                        video_status_badge = ui.badge("✨ MỚI", color="blue").classes("text-[10px] font-extrabold shrink-0")
                        current_video_title = ui.label("Chưa có video").classes("text-xs font-extrabold text-white truncate")
                    res_badge = ui.badge("1080p Full HD", color="emerald").classes("text-xs font-bold shrink-0")

                # Video Player HTML5
                video_player = ui.video("").classes("w-full aspect-[9/16] max-h-[300px] rounded-xl bg-black object-contain border border-slate-800 shadow-inner")

                ui.separator().classes("bg-slate-800 my-0.5")

                ui.label("⚡ TIẾN TRÌNH XỬ LÝ (LIVE PIPELINE)").classes("text-xs font-extrabold tracking-wider text-indigo-400 uppercase")

                progress_bar = ui.linear_progress(value=0.0).props("stripe rounded size=10px color=indigo")
                current_step_label = ui.label("Đang chờ lệnh...").classes("text-xs text-slate-300 font-medium")

                # Log Box (Terminal Style sắc nét)
                log_box = ui.log(max_lines=50).classes("w-full h-28 text-[11px] font-mono bg-slate-950/90 border border-slate-700/60 rounded-xl p-2.5 text-emerald-400 shadow-inner overflow-y-auto")

            # Nút Tải Video / Xem Video Master
            with ui.row().classes("w-full gap-2 pt-2 mt-auto border-t border-slate-800 bg-slate-900"):
                btn_play_master = ui.button("Xem Video Hoàn Thiện", icon="play_circle", on_click=lambda: play_master_video()).classes("w-full bg-indigo-600 hover:bg-indigo-500 text-white font-bold py-2 rounded-xl text-xs shadow-md")

    # --------------------------------------------------------
    # LOGIC XỬ LÝ SỰ KIỆN & PERSISTENCE
    # --------------------------------------------------------
    def push_log(msg: str):
        state["logs"].append(msg)
        if len(state["logs"]) > 100:
            state["logs"] = state["logs"][-100:]
        try:
            log_box.push(msg)
        except Exception:
            pass

    # Tự động nạp lại toàn bộ Log khi người dùng tải lại trang
    for prev_msg in state.get("logs", []):
        try:
            log_box.push(prev_msg)
        except Exception:
            pass

    def refresh_video_list():
        vids = scan_raw_videos()
        opt = {v["path"]: f"{v['name']} ({v['size']})" for v in vids}
        video_select.set_options(opt)

    def get_http_video_url(p: Path) -> str:
        if not p or not p.is_file():
            return ""
        p_res = p.resolve()
        p_str = str(p_res).lower()
        out_str = str(DEFAULT_OUT_DIR.resolve()).lower()
        raw_str = str(DEFAULT_RAW_DIR.resolve()).lower()
        dl_str = str(Path(r"C:\Users\vmath\Downloads").resolve()).lower()
        vid_str = str(Path(r"C:\Users\vmath\Videos").resolve()).lower()

        if p_str.startswith(out_str):
            rel = p_res.relative_to(DEFAULT_OUT_DIR.resolve()).as_posix()
            return f"/output/{rel}?t={time.time()}"
        elif p_str.startswith(raw_str):
            rel = p_res.relative_to(DEFAULT_RAW_DIR.resolve()).as_posix()
            return f"/raw/{rel}?t={time.time()}"
        elif p_str.startswith(dl_str):
            rel = p_res.relative_to(Path(r"C:\Users\vmath\Downloads").resolve()).as_posix()
            return f"/downloads/{rel}?t={time.time()}"
        elif p_str.startswith(vid_str):
            rel = p_res.relative_to(Path(r"C:\Users\vmath\Videos").resolve()).as_posix()
            return f"/videos/{rel}?t={time.time()}"
        else:
            target = DEFAULT_RAW_DIR / p.name
            if not target.is_file() or target.stat().st_size != p.stat().st_size:
                shutil.copy2(p, target)
            return f"/raw/{target.name}?t={time.time()}"

    async def select_video(path_str: str):
        if not path_str:
            return
        p = Path(path_str.strip().strip('"').strip("'"))
        if not p.is_file():
            ui.notify(f"File video không tồn tại: {p.name}", type="warning")
            return
        state["selected_video"] = p
        
        # Đảm bảo dropdown và các nhãn hiển thị đúng
        job_dir = DEFAULT_OUT_DIR / f"{p.stem}-full"
        master_mp4 = job_dir / f"{p.stem}_1080p_master_vi.mp4"
        is_rendered = master_mp4.is_file() and master_mp4.stat().st_size > 1024 * 1024

        cur_opts = dict(video_select.options or {})
        if str(p) not in cur_opts:
            status_tag = "✅ [ĐÃ RENDER]" if is_rendered else "✨ [MỚI]"
            cur_opts[str(p)] = f"{status_tag} {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)"
            video_select.set_options(cur_opts)
        if video_select.value != str(p):
            video_select.value = str(p)

        # 1. Cập nhật nhãn và trạng thái rõ ràng trên toàn bộ giao diện
        if is_rendered:
            video_status_badge.set_text("✅ ĐÃ RENDER")
            video_status_badge.props("color=emerald")
            status_label.set_text("🎉 Video này đã được xử lý & xuất bản 1080p Master")
            btn_auto_all.set_text("🔄 RENDER LẠI (NẾU SỬA KỊCH BẢN)")
            selected_file_label.set_text(f"✅ {p.name} (Đã Render - {p.stat().st_size/1024/1024:.1f} MB)")
            push_log(f"📌 [VIDEO ĐÃ RENDER] {p.name} -> Đã nạp ngay bản Master 1080p!")
        else:
            video_status_badge.set_text("✨ MỚI")
            video_status_badge.props("color=blue")
            status_label.set_text("🟢 Video mới sẵn sàng — Bấm 1-Click để bắt đầu!")
            btn_auto_all.set_text("🚀 1-CLICK TỰ ĐỘNG TOÀN BỘ")
            selected_file_label.set_text(f"✨ {p.name} (Mới - {p.stat().st_size/1024/1024:.1f} MB)")
            push_log(f"📌 [VIDEO MỚI TINH] {p.name} -> Đã phát video gốc, sẵn sàng chạy quy trình AI.")

        current_video_title.set_text(p.name)

        # 2. Xử lý kịch bản:
        seg_file = job_dir / "segments.json"
        if seg_file.is_file():
            try:
                cached_segs = json.loads(seg_file.read_text(encoding="utf-8"))
                if cached_segs:
                    state["segments"] = cached_segs
                    row_data = []
                    for s in cached_segs:
                        st = s.get("startMs", 0) / 1000
                        et = s.get("endMs", 0) / 1000
                        row_data.append({
                            "position": s.get("position", 0),
                            "timeline": f"{st:04.1f}s → {et:04.1f}s",
                            "sourceTextZh": s.get("sourceTextZh", ""),
                            "translatedTextVi": s.get("translatedTextVi", ""),
                        })
                    grid.options["rowData"] = row_data
                    grid.update()
                    segment_count_badge.set_text(f"{len(cached_segs)} câu")
                    progress_bar.set_value(1.0 if is_rendered else 0.5)
                    current_step_label.set_text("Đã có sẵn kịch bản và bản render Master!" if is_rendered else "Đã nạp kịch bản từ Cache!")
                    push_log(f"⚡ Đã nạp {len(cached_segs)} câu kịch bản từ Cache (không cần dịch lại).")
            except Exception as ex:
                push_log(f"⚠️ Đọc cache segments: {ex}")
        else:
            state["segments"] = []
            grid.options["rowData"] = []
            grid.update()
            segment_count_badge.set_text("0 câu")
            progress_bar.set_value(0.0)
            current_step_label.set_text("Video mới chưa xử lý. Bấm [1-Click Tự Động Toàn Bộ] để bắt đầu...")

        # 3. Xử lý Video Player:
        if is_rendered:
            state["current_master_mp4"] = master_mp4
            video_player.set_source(get_http_video_url(master_mp4))
            res_badge.set_text("1080p Master")
            res_badge.props("color=emerald")
        else:
            state["current_master_mp4"] = None
            video_player.set_source(get_http_video_url(p))
            res_badge.set_text("Video Gốc Preview")
            res_badge.props("color=blue")

        # Tự động AI nhận diện vị trí phụ đề gốc (AI Auto-Detect ROI)
        try:
            loop = asyncio.get_event_loop()
            detected_roi = await loop.run_in_executor(None, auto_detect_subtitle_roi, p)
            state["roi"] = detected_roi
            roi_badge.set_text(f"AI Mask: Y={detected_roi['yPercent']}%, H={detected_roi['heightPercent']}%")
            push_log(f"🎯 AI Auto-ROI: Đã dò ra vị trí phụ đề Y={detected_roi['yPercent']}%, Cao={detected_roi['heightPercent']}%")
        except Exception as ex:
            push_log(f"⚠️ Auto-ROI fallback: {ex}")
            state["roi"] = {"xPercent": 2.0, "yPercent": 66.0, "widthPercent": 96.0, "heightPercent": 9.8, "blurPx": 24}

    async def pick_local_file_native():
        ui.notify("Đang mở File Explorer...", type="info")
        push_log("📂 Đang mở Windows File Explorer để chọn video...")
        loop = asyncio.get_event_loop()
        file_path = await loop.run_in_executor(None, open_windows_file_dialog)
        if file_path:
            p = Path(file_path.strip().strip('"').strip("'"))
            if p.is_file():
                cur_opts = dict(video_select.options or {})
                job_dir = DEFAULT_OUT_DIR / f"{p.stem}-full"
                master_mp4 = job_dir / f"{p.stem}_1080p_master_vi.mp4"
                is_rendered = master_mp4.is_file() and master_mp4.stat().st_size > 1024 * 1024
                status_tag = "✅ [ĐÃ RENDER]" if is_rendered else "✨ [MỚI]"
                label_text = f"{status_tag} {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)"
                cur_opts[str(p)] = label_text
                video_select.set_options(cur_opts)
                video_select.set_value(str(p))
                await select_video(str(p))
                ui.notify(f"Đã nạp video: {p.name}", type="positive")
                push_log(f"✓ Đã chọn file từ máy: {p.name}")

    async def load_custom_path(path_str: str):
        if not path_str:
            return
        cleaned = path_str.strip().strip('"').strip("'")
        p = Path(cleaned)
        if p.is_file():
            cur_opts = dict(video_select.options or {})
            cur_opts[str(p)] = f"💻 {p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)"
            video_select.set_options(cur_opts)
            video_select.set_value(str(p))
            await select_video(str(p))
            ui.notify(f"Đã nạp file: {p.name}", type="positive")
            push_log(f"✓ Đã nạp đường dẫn: {p.name}")
        elif p.is_dir():
            vids = list(p.glob("*.mp4")) + list(p.glob("*.mov")) + list(p.glob("*.mkv"))
            if vids:
                cur_opts = dict(video_select.options or {})
                for v in vids:
                    cur_opts[str(v)] = f"📁 {v.name} ({v.stat().st_size / 1024 / 1024:.1f} MB)"
                video_select.set_options(cur_opts)
                video_select.set_value(str(vids[0]))
                await select_video(str(vids[0]))
                ui.notify(f"Đã nạp {len(vids)} video từ thư mục!", type="positive")
                push_log(f"✓ Đã quét thư mục: {p.name} ({len(vids)} video)")
            else:
                ui.notify("Thư mục không chứa file video nào!", type="warning")
        else:
            ui.notify("Đường dẫn file hoặc thư mục không tồn tại!", type="warning")

    async def handle_browser_upload(e):
        try:
            # Lấy tên file an toàn trên mọi phiên bản NiceGUI
            filename = getattr(e, "name", None)
            if not filename and hasattr(e, "file"):
                filename = getattr(e.file, "filename", None) or getattr(e.file, "name", None)
            if not filename:
                filename = f"upload_{int(time.time())}.mp4"
                
            target = DEFAULT_RAW_DIR / filename
            
            # Đọc nội dung file an toàn
            content = b""
            if hasattr(e, "file") and hasattr(e.file, "read"):
                read_res = e.file.read()
                if inspect.iscoroutine(read_res):
                    content = await read_res
                else:
                    content = read_res
            elif hasattr(e, "content"):
                if hasattr(e.content, "seek"):
                    e.content.seek(0)
                content = e.content.read()
                
            target.write_bytes(content)
            
            refresh_video_list()
            video_select.set_value(str(target))
            await select_video(str(target))
            ui.notify(f"Đã mở video: {filename}", type="positive")
            push_log(f"✓ Đã nạp video từ máy: {filename}")
        except Exception as ex:
            push_log(f"❌ Lỗi mở file: {ex}")
            ui.notify(f"Lỗi mở file: {ex}", type="warning")

    async def preview_3s_mask():
        if not state["selected_video"]:
            ui.notify("Vui lòng chọn video trước!", type="warning")
            return
        source = state["selected_video"]
        job_dir = DEFAULT_OUT_DIR / f"{source.stem}-full"
        job_dir.mkdir(parents=True, exist_ok=True)
        
        roi = state.get("roi") or {"xPercent": 2.0, "yPercent": 66.0, "widthPercent": 96.0, "heightPercent": 9.8}
        
        ui.notify("⚡ Đang tạo video xem thử 3 giây (Kính mờ + Cỡ chữ)...", type="info")
        push_log(f"⚡ Đang render 3s test clip cho '{source.name}': Mask Y={roi['yPercent']}%, Cỡ {state['font_size']}px...")
        
        # Lấy câu thoại đầu tiên của video này (nếu có) hoặc câu test trực quan
        sample_text = f"Xem thử Kính Mờ & Cỡ Chữ ({state['font_size']}px)"
        if state.get("segments") and len(state["segments"]) > 0:
            sample_text = state["segments"][0].get("translatedTextVi", sample_text)
            
        test_segs = [
            {"position": 0, "startMs": 0, "endMs": 3000, "translatedTextVi": sample_text}
        ]
        
        ass_path = job_dir / "preview_test_3s.ass"
        draw_ass(
            test_segs, 1080, 1920, roi, ass_path,
            font_name=state["font_name"], font_size=state["font_size"],
            font_color=state["font_color"]
        )
        
        preview_out = job_dir / "preview_3s_test.mp4"
        
        w_px = int(1080 * (roi["widthPercent"] / 100.0))
        h_px = int(1920 * (roi["heightPercent"] / 100.0))
        x_px = int(1080 * (roi["xPercent"] / 100.0))
        y_px = int(1920 * (roi["yPercent"] / 100.0))
        ass_posix = ass_path.as_posix()
        if len(ass_posix) >= 2 and ass_posix[1] == ":":
            ass_posix = ass_posix[0] + "\\:" + ass_posix[2:]

        video_filter = (
            f"[0:v]scale=1080:1920,split=2[base][ref];"
            f"[ref]crop={w_px}:{h_px}:{x_px}:{y_px},boxblur=24:3:24:3,drawbox=x=0:y=0:w={w_px}:h={h_px}:color=white@0.14:t=fill[blur];"
            f"[base][blur]overlay={x_px}:{y_px},subtitles='{ass_posix}'[video]"
        )
        
        cmd = [
            "ffmpeg", "-y", "-ss", "2.0", "-t", "3.0", "-i", str(source),
            "-filter_complex", video_filter, "-map", "[video]",
            "-c:v", "libx264", "-preset", "ultrafast", str(preview_out)
        ]
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: subprocess.run(cmd, check=True))
        
        preview_url = get_http_video_url(preview_out)
        video_player.set_source(preview_url)
        current_video_title.set_text(f"Xem thử: {source.name}")
        res_badge.set_text(f"3s Test ({state['font_size']}px)")
        res_badge.props("color=purple")
        push_log(f"✓ Đã phát 3s Test Clip ({state['font_size']}px) lên Video Player!")
        ui.notify("Đã tạo video xem thử 3 giây thành công!", type="positive")

    def update_voice(voice_val: str):
        state["selected_voice"] = voice_val
        push_log(f"🎙️ Đã đổi giọng đọc: {VOICE_OPTIONS.get(voice_val, voice_val)}")

    def update_font_name(font_val: str):
        state["font_name"] = font_val
        css_size = int(state['font_size'] * 0.28)
        preview_text_label.style(f"font-family: '{state['font_name']}', sans-serif; font-size: {css_size}px; font-weight: 700; color: {state['font_color']}; text-shadow: -1.2px -1.2px 0 #000, 1.2px -1.2px 0 #000, -1.2px 1.2px 0 #000, 1.2px 1.2px 0 #000, 0 2px 4px rgba(0,0,0,0.8);")
        push_log(f"🔤 Đã đổi Font chữ: {font_val}")

    def update_font_color(color_val: str):
        state["font_color"] = color_val
        css_size = int(state['font_size'] * 0.28)
        preview_text_label.style(f"font-family: '{state['font_name']}', sans-serif; font-size: {css_size}px; font-weight: 700; color: {state['font_color']}; text-shadow: -1.2px -1.2px 0 #000, 1.2px -1.2px 0 #000, -1.2px 1.2px 0 #000, 1.2px 1.2px 0 #000, 0 2px 4px rgba(0,0,0,0.8);")
        push_log(f"🎨 Đã đổi màu chữ: {COLOR_OPTIONS.get(color_val, color_val)}")

    def update_font_size(size_val: int, badge_elem):
        state["font_size"] = int(size_val)
        badge_elem.set_text(f"{state['font_size']} px")
        css_size = int(state['font_size'] * 0.28)
        preview_text_label.style(f"font-family: '{state['font_name']}', sans-serif; font-size: {css_size}px; font-weight: 700; color: {state['font_color']}; text-shadow: -1.2px -1.2px 0 #000, 1.2px -1.2px 0 #000, -1.2px 1.2px 0 #000, 1.2px 1.2px 0 #000, 0 2px 4px rgba(0,0,0,0.8);")
        push_log(f"📏 Đã chỉnh cỡ chữ: {state['font_size']} px")

    async def preview_selected_sentence():
        selected_rows = await grid.get_selected_rows()
        if not selected_rows:
            ui.notify("Vui lòng click chọn 1 hàng trong bảng kịch bản để nghe thử!", type="warning")
            return
        row = selected_rows[0]
        text = row.get("translatedTextVi", "")
        if not text:
            return
        ui.notify(f"Đang tạo giọng đọc cho câu: '{text[:28]}...' ", type="info")
        push_log(f"🎙️ Đang nghe thử câu #{row.get('position', 0)+1}: '{text}'")
        temp_voice = DEFAULT_OUT_DIR / "temp_preview.mp3"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, synthesize, text, state["selected_voice"], temp_voice, settings)
        sample_audio_player.set_source(f"/output/temp_preview.mp3?t={time.time()}")
        sample_audio_player.play()

    def on_cell_edited(args: dict):
        data = args.get("data", {})
        pos = data.get("position")
        new_val = data.get("translatedTextVi")
        for s in state["segments"]:
            if s.get("position") == pos:
                s["translatedTextVi"] = new_val
                push_log(f"✏️ Đã sửa câu #{pos+1}: '{new_val}'")
                break
        if state["selected_video"]:
            job_dir = DEFAULT_OUT_DIR / f"{state['selected_video'].stem}-full"
            seg_file = job_dir / "segments.json"
            try:
                seg_file.write_text(json.dumps(state["segments"], ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    async def run_transcribe_and_translate():
        if not state["selected_video"]:
            ui.notify("Vui lòng chọn video trước!", type="warning")
            return
        source = state["selected_video"]
        job_dir = DEFAULT_OUT_DIR / f"{source.stem}-full"
        job_dir.mkdir(parents=True, exist_ok=True)
        seg_file = job_dir / "segments.json"

        # 1. Kiểm tra Cache Kịch bản đã dịch trước đó
        if seg_file.is_file():
            try:
                cached_segs = json.loads(seg_file.read_text(encoding="utf-8"))
                if cached_segs and all(s.get("translatedTextVi") for s in cached_segs):
                    push_log(f"⚡ TÌM THẤY CACHE: Đã có sẵn kịch bản {len(cached_segs)} câu đã dịch, bỏ qua Whisper & Gemini!")
                    state["segments"] = cached_segs
                    row_data = []
                    for s in cached_segs:
                        st = s.get("startMs", 0) / 1000
                        et = s.get("endMs", 0) / 1000
                        row_data.append({
                            "position": s.get("position", 0),
                            "timeline": f"{st:04.1f}s → {et:04.1f}s",
                            "sourceTextZh": s.get("sourceTextZh", ""),
                            "translatedTextVi": s.get("translatedTextVi", ""),
                        })
                    grid.options["rowData"] = row_data
                    grid.update()
                    segment_count_badge.set_text(f"{len(cached_segs)} câu")
                    progress_bar.set_value(0.5)
                    current_step_label.set_text(f"⚡ Đã nạp {len(cached_segs)} câu từ Cache (0.01s)!")
                    ui.notify(f"Đã nạp {len(cached_segs)} câu từ Cache!", type="positive")
                    return
            except Exception:
                pass

        progress_bar.set_value(0.1)
        current_step_label.set_text("1/4: Đang trích audio WAV & Whisper ASR...")
        push_log("🎵 Đang trích xuất âm thanh từ video...")
        
        audio_wav = job_dir / "source_full.wav"
        if not audio_wav.is_file():
            extract_audio(source, audio_wav)
        
        push_log("🗣️ Đang chạy Whisper ASR nhận diện tiếng Trung...")
        loop = asyncio.get_event_loop()
        segs = await loop.run_in_executor(None, transcribe, audio_wav, "tiny", "cpu")
        push_log(f"✓ Whisper phát hiện {len(segs)} câu thoại.")

        # Tự động tách BGM bằng Demucs AI nếu chưa có
        clean_bgm_wav = job_dir / "clean_bgm.wav"
        if not clean_bgm_wav.is_file():
            push_log("🧠 Đang tách sạch tiếng Trung & giữ nhạc nền bằng Demucs AI...")
            demucs_dir = job_dir / "demucs_out"
            try:
                proc = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "demucs.separate",
                    "--two-stems=vocals", "-n", "htdemucs",
                    "-o", str(demucs_dir), str(audio_wav),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()
                demucs_out = demucs_dir / "htdemucs" / audio_wav.stem / "no_vocals.wav"
                if demucs_out.is_file():
                    shutil.copy2(demucs_out, clean_bgm_wav)
                    push_log("✓ Đã tách BGM sạch 100% bằng Demucs AI.")
            except Exception as exc:
                push_log(f"⚠️ Demucs BGM không khả dụng ({exc}), dùng audio gốc.")

        progress_bar.set_value(0.4)
        current_step_label.set_text("2/4: Gemini 2.5 Flash chuyển ngữ bản xứ...")
        push_log("🧠 Đang gọi Google Gemini 2.5 Flash chuyển ngữ kịch bản...")
        for s in segs:
            s["sourceTextZh"] = s.get("asrTextZh", "")

        segs = await loop.run_in_executor(
            None, translate_segments_native, segs, "gemini"
        )
        state["segments"] = segs
        try:
            seg_file.write_text(json.dumps(segs, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        push_log("✓ Gemini chuyển ngữ hoàn tất chuẩn TikTok bản xứ!")

        # Cập nhật AG Grid
        row_data = []
        for s in segs:
            st = s["startMs"] / 1000
            et = s["endMs"] / 1000
            row_data.append({
                "position": s["position"],
                "timeline": f"{st:04.1f}s → {et:04.1f}s",
                "sourceTextZh": s.get("sourceTextZh", ""),
                "translatedTextVi": s.get("translatedTextVi", ""),
            })
        grid.options["rowData"] = row_data
        grid.update()
        segment_count_badge.set_text(f"{len(segs)} câu")
        progress_bar.set_value(0.5)
        current_step_label.set_text("Đã nạp kịch bản vào AG Grid! Bạn có thể sửa câu từ.")
        ui.notify("Đã dịch xong kịch bản!", type="positive")

    async def run_tts_and_render():
        if not state["segments"]:
            ui.notify("Chưa có kịch bản! Hãy bấm '1. Bóc Tách & Dịch AI' trước.", type="warning")
            return
        source = state["selected_video"]
        job_dir = DEFAULT_OUT_DIR / f"{source.stem}-full"
        clean_bgm_wav = job_dir / "clean_bgm.wav"

        progress_bar.set_value(0.6)
        current_step_label.set_text(f"3/4: Đang sinh {len(state['segments'])} giọng đọc CapCut...")
        push_log(f"🎙️ Đang tổng hợp {len(state['segments'])} giọng đọc CapCut TikTok...")

        loop = asyncio.get_event_loop()
        
        def gen_voice(item):
            idx, seg = item
            raw_v = job_dir / f"voice_{idx:03d}_raw.mp3"
            fit_v = job_dir / f"voice_{idx:03d}.mp3"
            slot_ms = seg["endMs"] - seg["startMs"]
            vi_text = seg.get("translatedTextVi") or seg.get("sourceTextZh") or "..."
            
            # Kiểm tra xem file voice đã có sẵn chưa
            if not fit_v.is_file() or fit_v.stat().st_size < 100:
                synthesize(vi_text, state["selected_voice"], raw_v, settings)
                fit_ms, _ = fit_voice(raw_v, fit_v, slot_ms, max_tempo=1.35)
            else:
                fit_ms = slot_ms

            seg["voicePath"] = str(fit_v)
            seg["voiceDurationMs"] = fit_ms
            seg["endMs"] = seg["startMs"] + fit_ms
            return idx, fit_v, seg["startMs"]

        with ThreadPoolExecutor(max_workers=5) as ex:
            results = await loop.run_in_executor(None, lambda: list(ex.map(gen_voice, enumerate(state["segments"]))))

        results.sort(key=lambda x: x[0])
        voice_inputs = [(r[1], r[2]) for r in results]
        push_log(f"✓ Đã sinh xong {len(voice_inputs)} file voice CapCut đồng bộ.")

        progress_bar.set_value(0.8)
        current_step_label.set_text("4/4: Đang render video Full HD 1080p Kính Mờ...")
        push_log("🎞️ Đang render FFmpeg 1080p Full HD với hiệu ứng Kính Mờ...")

        # Render 1080p MP4
        roi = state.get("roi") or {"xPercent": 2.0, "yPercent": 66.0, "widthPercent": 96.0, "heightPercent": 9.8}
        ass_path = job_dir / f"{source.stem}_1080p.ass"
        srt_path = job_dir / f"{source.stem}_1080p.srt"
        draw_ass(
            state["segments"], 1080, 1920, roi, ass_path,
            font_name=state["font_name"], font_size=state["font_size"],
            font_color=state["font_color"]
        )
        draw_srt(state["segments"], srt_path)

        out_mp4 = job_dir / f"{source.stem}_1080p_master_vi.mp4"
        
        # FFmpeg call with dynamic AI-detected ROI
        w_px = int(1080 * (roi["widthPercent"] / 100.0))
        h_px = int(1920 * (roi["heightPercent"] / 100.0))
        x_px = int(1080 * (roi["xPercent"] / 100.0))
        y_px = int(1920 * (roi["yPercent"] / 100.0))
        ass_posix = ass_path.as_posix()
        if len(ass_posix) >= 2 and ass_posix[1] == ":":
            ass_posix = ass_posix[0] + "\\:" + ass_posix[2:]

        # Gom các câu thoại liên tục thành các khối thời gian cố định (loại bỏ nhấp nháy, chỉ tắt khi nghỉ >= 1.8s)
        if state["segments"]:
            blocks = compute_mask_intervals(state["segments"], max_gap_s=1.8, padding_s=0.15)
            active_intervals = [f"between(t,{st},{et})" for st, et in blocks]
        else:
            active_intervals = []

        if active_intervals:
            enable_filter = f":enable='{'+'.join(active_intervals)}'"
        else:
            enable_filter = ":enable='0'"

        video_filter = (
            f"[0:v]scale=1080:1920,split=2[base][ref];"
            f"[ref]crop={w_px}:{h_px}:{x_px}:{y_px},boxblur=24:3:24:3,drawbox=x=0:y=0:w={w_px}:h={h_px}:color=white@0.14:t=fill[blur];"
            f"[base][blur]overlay={x_px}:{y_px}{enable_filter},subtitles='{ass_posix}'[video]"
        )

        has_clean_bgm = clean_bgm_wav.is_file()
        if has_clean_bgm:
            cmd = ["ffmpeg", "-y", "-i", str(source), "-i", str(clean_bgm_wav)]
            audio_offset = 2
        else:
            cmd = ["ffmpeg", "-y", "-i", str(source)]
            audio_offset = 1

        audio_labels = []
        for idx, (voice_path, offset_ms) in enumerate(voice_inputs, start=1):
            cmd.extend(["-i", str(voice_path)])
            input_idx = idx + audio_offset - 1
            audio_labels.append(f"[{input_idx}:a]adelay={offset_ms}|{offset_ms}[v{idx}]")

        all_v_tags = "".join(f"[v{i}]" for i in range(1, len(voice_inputs) + 1))
        mix_voice = f"{';'.join(audio_labels)};{all_v_tags}amix=inputs={len(voice_inputs)}:duration=longest:normalize=0[allvoice]"
        
        if has_clean_bgm:
            audio_filter = f"{mix_voice};[1:a]volume=0.85[bgm];[bgm][allvoice]amix=inputs=2:duration=first:normalize=0[finalaudio]"
        else:
            audio_filter = f"{mix_voice};[0:a]volume=0.22[bgm];[bgm][allvoice]amix=inputs=2:duration=first:normalize=0[finalaudio]"

        full_cmd = cmd + [
            "-filter_complex", f"{video_filter};{audio_filter}",
            "-map", "[video]", "-map", "[finalaudio]",
            "-threads", "4",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(out_mp4)
        ]
        
        # Chạy FFmpeg Native Async Subprocess - Hoàn toàn không chặn WebSocket của NiceGUI
        proc = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        
        state["current_master_mp4"] = out_mp4
        master_url = get_http_video_url(out_mp4)
        push_log(f"🎉 RENDER THÀNH CÔNG: {out_mp4.name} ({out_mp4.stat().st_size/1024/1024:.2f} MB)")
        push_log("🎬 Đã nạp Video Master 1080p lên Player!")
        try:
            progress_bar.set_value(1.0)
            current_step_label.set_text("🎉 HOÀN TẤT XUẤT VIDEO 1080P MASTER!")
            video_player.set_source(master_url)
            current_video_title.set_text(source.name)
            video_status_badge.set_text("✅ ĐÃ RENDER")
            video_status_badge.props("color=emerald")
            res_badge.set_text("1080p Master")
            res_badge.props("color=emerald")
            btn_auto_all.set_text("🔄 RENDER LẠI (NẾU SỬA KỊCH BẢN)")
            refresh_video_list()
            ui.notify("Đã render xong Video 1080p Master!", type="positive")
        except Exception:
            pass

    async def run_full_pipeline():
        await run_transcribe_and_translate()
        await run_tts_and_render()

    def play_master_video():
        if state["current_master_mp4"] and state["current_master_mp4"].is_file():
            master_mp4 = state["current_master_mp4"]
            master_url = get_http_video_url(master_mp4)
            video_player.set_source(master_url)
            res_badge.set_text("1080p Master")
            res_badge.props("color=emerald")
            video_status_badge.set_text("✅ ĐÃ RENDER")
            video_status_badge.props("color=emerald")
            if state.get("selected_video"):
                current_video_title.set_text(state["selected_video"].name)
            progress_bar.set_value(1.0)
            current_step_label.set_text(f"🎉 Đang phát: {master_mp4.name}")
            ui.notify("Đang phát video Master 1080p!", type="info")
        else:
            ui.notify("Chưa có file video Master!", type="warning")

    # Tự động nạp hoặc khôi phục trạng thái video hiện tại
    if state.get("selected_video"):
        p = state["selected_video"]
        job_dir = DEFAULT_OUT_DIR / f"{p.stem}-full"
        master_mp4 = job_dir / f"{p.stem}_1080p_master_vi.mp4"
        is_rendered = master_mp4.is_file() and master_mp4.stat().st_size > 1024 * 1024
        
        current_video_title.set_text(p.name)
        selected_file_label.set_text(f"{'✅ [ĐÃ RENDER]' if is_rendered else '✨ [MỚI]'} {p.name}")
        video_status_badge.set_text("✅ ĐÃ RENDER" if is_rendered else "✨ MỚI")
        video_status_badge.props(f"color={'emerald' if is_rendered else 'blue'}")
        
        if is_rendered:
            state["current_master_mp4"] = master_mp4
            video_player.set_source(get_http_video_url(master_mp4))
            res_badge.set_text("1080p Master")
            res_badge.props("color=emerald")
            progress_bar.set_value(1.0)
            current_step_label.set_text("🎉 Đã render hoàn tất 1080p Master!")
            btn_auto_all.set_text("🔄 RENDER LẠI (NẾU SỬA KỊCH BẢN)")
        else:
            video_player.set_source(get_http_video_url(p))
            res_badge.set_text("Video Gốc Preview")
            res_badge.props("color=blue")

        # Nạp kịch bản vào AG Grid
        if state.get("segments"):
            row_data = []
            for s in state["segments"]:
                st = s.get("startMs", 0) / 1000
                et = s.get("endMs", 0) / 1000
                row_data.append({
                    "position": s.get("position", 0),
                    "timeline": f"{st:04.1f}s → {et:04.1f}s",
                    "sourceTextZh": s.get("sourceTextZh", ""),
                    "translatedTextVi": s.get("translatedTextVi", ""),
                })
            grid.options["rowData"] = row_data
            grid.update()
            segment_count_badge.set_text(f"{len(state['segments'])} câu")

    # Nạp danh sách video ban đầu
    refresh_video_list()


# Khởi chạy NiceGUI server
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title="DUBVI STUDIO - Douyin Video Dubbing AI",
        port=8080,
        reload=False,
        show=True,
        reconnect_timeout=120.0
    )
