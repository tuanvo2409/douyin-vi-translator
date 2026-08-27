"""Test adjusted box ROI to cover 100% of the Chinese text:
"""
import cv2
import subprocess
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
raw_video = Path(r"C:\Users\vmath\Downloads\video douyin raw\MS4wLjABAAAAwAhJTV1V81xlr9MWGYk30jeWfDSy4CkGGfxv0Aj7IJmvmHxDGwU221itoQ6tsTLY\7615935754560097570\7615935754560097570_video.mp4")
test_dir = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\mask_audit\test_fix")
test_dir.mkdir(parents=True, exist_ok=True)

# Vùng ROI mới: Y = 66.0% -> 76.0% (width = 96%, height = 10.0%)
# Tương ứng 1080x1920: y=1267, h=192, x=21, w=1036
w_px, h_px = 1036, 192
x_px, y_px = 21, 1267

# Test render 5s đầu tiên (chứa frame 4.5s)
test_out = test_dir / "test_fix_5s.mp4"
video_filter = (
    f"[0:v]scale=1080:1920,split=2[base][ref];"
    f"[ref]crop={w_px}:{h_px}:{x_px}:{y_px},boxblur=24:3:24:3,drawbox=x=0:y=0:w={w_px}:h={h_px}:color=white@0.14:t=fill[blur];"
    f"[base][blur]overlay={x_px}:{y_px}[video]"
)

subprocess.run([
    "ffmpeg", "-y", "-ss", "0", "-t", "6", "-i", str(raw_video),
    "-filter_complex", video_filter, "-map", "[video]",
    "-c:v", "libx264", "-preset", "ultrafast", str(test_out)
], check=True)

# Extract frame 4.5s từ video test mới
frame_45 = test_dir / "frame_45s_fixed.jpg"
subprocess.run([
    "ffmpeg", "-y", "-ss", "4.5", "-i", str(test_out),
    "-frames:v", "1", "-q:v", "2", str(frame_45)
], check=True)

img = cv2.imread(str(frame_45))
res, _ = ocr(img)
print("\n🔍 KẾT QUẢ TEST MASK MỚI TẠI FRAME 4.5s:")
has_chinese = False
if res:
    for box, text, score in res:
        print(f"  • Nhận diện: '{text}' (score: {score:.2f})")
        for ch in text:
            if '\u4e00' <= ch <= '\u9fff':
                has_chinese = True
print(f"👉 KẾT LUẬN: {'❌ VẪN LỘ CHỮ TRUNG' if has_chinese else '✅ ĐÃ CHE SẠCH 100% KHÔNG CÒN 1 KÝ TỰ!'}")
