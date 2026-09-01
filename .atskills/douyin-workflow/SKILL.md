---
name: douyin-workflow
description: Chu trình dịch thuật, lồng tiếng và render video Douyin sang tiếng Việt chuẩn 1080p
---

# Douyin Workflow Playbook

## Các bước vận hành chuẩn:
1. Nạp video gốc vào `C:\Users\vmath\Videos\douyin`
2. Khởi chạy Web Studio: `python studio_app.py` tại cổng `8080`
3. Quét Sub câm và dò Kính Mờ (Auto ROI 14%)
4. Sinh Ma trận 8 Hook (Túi Kịch Bản) và chọn Hook sắc bén nhất cho câu #1
5. Lồng tiếng CapCut TTS (Giọng Mai BV421)
6. Tách sạch BGM bằng Demucs AI (`clean_bgm.wav`)
7. Render FFmpeg 1080p Master với phụ đề căn giữa hoàn hảo trong Kính Mờ.
