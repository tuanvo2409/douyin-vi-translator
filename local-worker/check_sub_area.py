"""Verify subtitle area specifically:
"""
import cv2
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
frame_45 = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\mask_audit\test_fix\frame_45s_fixed.jpg")
img = cv2.imread(str(frame_45))
h, w = img.shape[:2]

# Crop vùng phụ đề đáy video (Y từ 60% đến 85%)
sub_area = img[int(h * 0.60):int(h * 0.85), :]
res, _ = ocr(sub_area)

print("\n🔍 QUÉT VÙNG PHỤ ĐỀ ĐÁY VIDEO (Y: 60% -> 85%):")
if res:
    for box, text, score in res:
        print(f"  • Nhận diện: '{text}' (score: {score:.2f})")
else:
    print("  ✅ HOÀN TOÀN TRỐNG / SẠCH 100% (Chữ Trung gốc đã bị xoá mờ triệt để)!")
