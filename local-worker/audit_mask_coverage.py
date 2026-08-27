"""Audit Subtitle Mask Coverage on Master Video:
- Samples 8 timestamps across the full video
- Crops the Frosted Glass Subtitle Box
- Runs RapidOCR to detect if any Chinese character is visible or leaked
- Saves inspection snapshot images
"""
from __future__ import annotations

import subprocess
from pathlib import Path
import cv2
from rapidocr_onnxruntime import RapidOCR

video_path = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\7615935754560097570_1080p_master_vi.mp4")
out_dir = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\mask_audit")
out_dir.mkdir(parents=True, exist_ok=True)

timestamps = [
    (1.5, "01. Món đồ siêu mê..."),
    (4.5, "02. Lại dời tủ nhỏ..."),
    (12.0, "05. Hai tầng mở..."),
    (22.5, "07. Mặt trước hút nam châm..."),
    (34.5, "12. Tủ đựng đồ cuối giường..."),
    (44.0, "17. Đi chân trần hơi cứng..."),
    (55.0, "22. Tủ sắt bao mê..."),
    (67.0, "27. Sắp xếp gọn gàng..."),
    (72.0, "28. Ngăn nắp xịn xò..."),
]

ocr = RapidOCR()
audit_results = []

print("=" * 75)
print(f"🔍 BẮT ĐẦU QUÉT KIỂM ĐỊNH MASK KÍNH MỜ TRÊN VIDEO HOÀN THIỆN")
print(f"📹 Video: {video_path.name}")
print("=" * 75)

for sec, label in timestamps:
    full_frame = out_dir / f"audit_frame_{int(sec*10):03d}s.jpg"
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(sec), "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(full_frame)
    ], capture_output=True)
    
    img = cv2.imread(str(full_frame))
    if img is None:
        continue
        
    h, w = img.shape[:2]
    # Vùng box: y=65.5% -> 72.3%
    y1, y2 = int(h * 0.64), int(h * 0.74)
    x1, x2 = int(w * 0.01), int(w * 0.99)
    box_crop = img[y1:y2, x1:x2]
    
    crop_file = out_dir / f"audit_crop_{int(sec*10):03d}s.jpg"
    cv2.imwrite(str(crop_file), box_crop)
    
    # OCR trên box
    res, _ = ocr(box_crop)
    detected_texts = [r[1] for r in res] if res else []
    
    # Kiểm tra xem có chữ Hán nào lọt qua không (chỉ nhận diện được chữ Việt)
    has_chinese_leak = False
    chinese_chars = []
    for t in detected_texts:
        for ch in t:
            if '\u4e00' <= ch <= '\u9fff':
                has_chinese_leak = True
                chinese_chars.append(ch)
                
    status = "❌ CÓ LỘ CHỮ TRUNG" if has_chinese_leak else "✅ CHE SẠCH 100%"
    audit_results.append({
        "time": f"{sec:.1f}s",
        "label": label,
        "status": status,
        "detected": detected_texts,
        "leak_chars": "".join(chinese_chars),
        "img_path": str(full_frame)
    })
    
    print(f"\n⏱️ [{sec:04.1f}s] {label}:")
    print(f"  • Trạng thái: {status}")
    print(f"  • Chữ nhận diện trên box: {detected_texts}")
    if has_chinese_leak:
        print(f"  ⚠️ Ký tự Trung bị lộ: {''.join(chinese_chars)}")
    print(f"  🖼️ Ảnh kiểm tra: {full_frame.name}")

print("\n" + "=" * 75)
print("🎯 TỔNG KẾT KIỂM ĐỊNH:")
clean_count = sum(1 for r in audit_results if "CHE SẠCH" in r["status"])
print(f"✓ Tỷ lệ che phủ hoàn hảo: {clean_count}/{len(audit_results)} ({clean_count/len(audit_results)*100:.1f}%)")
print("=" * 75)
