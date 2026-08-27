"""Render snapshot with 56px bigger font size:
"""
import subprocess
from pathlib import Path
from dubvi_worker import draw_ass

raw_video = Path(r"C:\Users\vmath\Downloads\video douyin raw\MS4wLjABAAAAwAhJTV1V81xlr9MWGYk30jeWfDSy4CkGGfxv0Aj7IJmvmHxDGwU221itoQ6tsTLY\7615935754560097570\7615935754560097570_video.mp4")
out_dir = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\mask_audit\font_test")
out_dir.mkdir(parents=True, exist_ok=True)

segments = [
    {"position": 0, "startMs": 460, "endMs": 2640, "translatedTextVi": "Món đồ siêu mê cho góc nhỏ của mình nè!"},
    {"position": 1, "startMs": 3420, "endMs": 5650, "translatedTextVi": "Lại dời tủ nhỏ ra cuối giường rồi nè."}
]

roi = {"xPercent": 2.0, "yPercent": 66.0, "widthPercent": 96.0, "heightPercent": 9.8}
ass_path = out_dir / "test_56px.ass"
draw_ass(segments, 1080, 1920, roi, ass_path, font_name="Arial")

w_px, h_px = int(1080 * 0.96), int(1920 * 0.098)
x_px, y_px = int(1080 * 0.02), int(1920 * 0.660)

ass_posix = ass_path.as_posix()
if len(ass_posix) >= 2 and ass_posix[1] == ":":
    ass_posix = ass_posix[0] + "\\:" + ass_posix[2:]

video_filter = (
    f"[0:v]scale=1080:1920,split=2[base][ref];"
    f"[ref]crop={w_px}:{h_px}:{x_px}:{y_px},boxblur=24:3:24:3,drawbox=x=0:y=0:w={w_px}:h={h_px}:color=white@0.14:t=fill[blur];"
    f"[base][blur]overlay={x_px}:{y_px},subtitles='{ass_posix}'[video]"
)

test_mp4 = out_dir / "test_big_font.mp4"
subprocess.run([
    "ffmpeg", "-y", "-ss", "0", "-t", "6", "-i", str(raw_video),
    "-filter_complex", video_filter, "-map", "[video]",
    "-c:v", "libx264", "-preset", "ultrafast", str(test_mp4)
], check=True)

# Trích frame 4.5s
frame_45 = out_dir / "preview_big_font_45s.jpg"
subprocess.run([
    "ffmpeg", "-y", "-ss", "4.5", "-i", str(test_mp4),
    "-frames:v", "1", "-q:v", "2", str(frame_45)
], check=True)

print("✓ Đã render snapshot font to 56px:", frame_45)
