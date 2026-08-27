# DUBVI Local Worker

Worker này chạy trên **máy của bạn**, nhận job từ giao diện DUBVI và không upload video nguồn. Nó quét `DUBVI_MEDIA_DIR`, claim job được xếp hàng, xử lý bằng FFmpeg, rồi chỉ gửi metadata/tiến trình/lối ra local về control plane.

## Điều kiện chạy

Bạn cần Python 3.11+, FFmpeg/FFprobe trong `PATH`, đủ dung lượng trống cho file WAV và output, cùng kết nối mạng nếu dùng Edge TTS. `faster-whisper` và RapidOCR có thể chạy CPU, nhưng OCR/ASR sẽ nhanh hơn đáng kể trên máy có GPU NVIDIA tương thích.

## Cài đặt

```bash
cd local-worker
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp worker.env.template .env
python install_argos_models.py
python dubvi_worker.py verify
```

Sau đó tạo worker pairing token trong giao diện DUBVI, chép token vào `DUBVI_WORKER_TOKEN` trong `.env`, rồi chạy:

```bash
python dubvi_worker.py watch
```

## Pipeline thật

1. FFmpeg tách audio WAV 16 kHz.
2. faster-whisper tạo segment tiếng Trung có timestamp.
3. RapidOCR lấy nhiều frame trong fixed ROI rồi vote text/confidence.
4. Argos Translate được ưu tiên nếu có model cục bộ; khi Argos chưa có cặp Trung–Việt, worker fallback sang **MarianMT `Helsinki-NLP/opus-mt-zh-vi`** và lưu model trong `DUBVI_MODEL_DIR`. Lần tải đầu tiên khoảng 300 MB; sau đó text không cần đi qua dịch vụ dịch bên ngoài. Bản dịch được đánh giá độ dài theo slot voice và chỉ rút tại ranh giới clause tự nhiên; câu còn dài sẽ được gắn cờ duyệt.
5. Với OCR đa khung hình có confidence thấp, disagreement giữa frame hoặc mâu thuẫn với ASR, worker gọi endpoint correction LLM. Endpoint dùng schema JSON, chỉ chấp nhận sửa khi confidence model đạt ngưỡng; mọi candidate, bản gốc, bản sửa và lý do đều được lưu trong `ocrAuditJson` để người dùng duyệt.
5. Edge TTS tạo voice từng segment; FFmpeg chỉ tăng nhịp trong giới hạn `maxTempo`.
6. Nếu voice vẫn dài, worker chuyển job sang `awaiting_review` thay vì cắt lời hoặc làm giọng méo.
7. FFmpeg blur fixed ROI theo từng frame, render ASS/SRT tiếng Việt, mix audio và xuất MP4.

## Privacy & giới hạn

Video, WAV, frame OCR, ASS/SRT, voice file, MP4 output, model cache và job log đều nằm trên máy local. Chỉ token, inventory file (tên/kích thước/thời gian), config, segment text, tiến trình và đường dẫn output local được trao đổi với control plane. Correction LLM chạy server-side qua model tích hợp và chỉ nhận metadata/text, không nhận file video hoặc frame; mỗi lần correction có thể phát sinh usage model. Edge TTS không phải offline: text tiếng Việt được gửi đến Microsoft để tổng hợp giọng. Để offline hoàn toàn, cần thay Edge TTS bằng provider Piper/Coqui.
