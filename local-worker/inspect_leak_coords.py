"""Inspect where the Chinese text is located vs the box mask position:
"""
import cv2
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()
audit_dir = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\mask_audit")

# Kiểm tra frame 4.5s và 12.0s
for sec in [45, 120, 225]:
    f = audit_dir / f"audit_frame_{sec:03d}s.jpg"
    img = cv2.imread(str(f))
    h, w = img.shape[:2]
    
    res, _ = ocr(img)
    print(f"\n--- Frame {sec/10}s ({w}x{h}) ---")
    if res:
        for box, text, score in res:
            # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            ymin = min(p[1] for p in box)
            ymax = max(p[1] for p in box)
            xmin = min(p[0] for p in box)
            xmax = max(p[0] for p in box)
            print(f"Text: '{text}' (score: {score:.2f}) | Y: {ymin/h*100:.1f}% -> {ymax/h*100:.1f}% (px: {ymin}->{ymax}) | X: {xmin/w*100:.1f}% -> {xmax/w*100:.1f}%")
