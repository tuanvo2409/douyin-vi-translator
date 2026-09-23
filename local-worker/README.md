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

### Piper TTS local (mặc định)

Worker mặc định dùng Piper/ONNX local với `DUBVI_TTS_PROVIDER=piper` và tốc độ
`fast`. TTS không gửi text ra ngoài và không dùng API token; chỉ phần dịch LLM
(nếu được bật) vẫn có thể dùng provider từ xa.

`piper-tts==1.8.0` được cài cùng `requirements.txt`. Model không nằm trong Git;
tải model Vietnamese VITS từ [phatjkk/vits-tts-vietnamese](https://github.com/phatjkk/vits-tts-vietnamese)
vào thư mục model cache:

```powershell
$modelDir = if ($env:DUBVI_PIPER_MODEL) {
    Split-Path -Parent $env:DUBVI_PIPER_MODEL
} elseif ($env:DUBVI_MODEL_DIR) {
    Join-Path $env:DUBVI_MODEL_DIR 'piper-vietnamese'
} elseif ($env:DUBVI_MEDIA_DIR) {
    Join-Path $env:DUBVI_MEDIA_DIR 'dubvi-model-cache\piper-vietnamese'
} else {
    Join-Path $env:USERPROFILE 'Videos\douyin\dubvi-model-cache\piper-vietnamese'
}
New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
Invoke-WebRequest 'https://raw.githubusercontent.com/phatjkk/vits-tts-vietnamese/main/pretrained_vi.onnx' -OutFile (Join-Path $modelDir 'pretrained_vi.onnx')
Invoke-WebRequest 'https://raw.githubusercontent.com/phatjkk/vits-tts-vietnamese/main/pretrained_vi.onnx.json' -OutFile (Join-Path $modelDir 'pretrained_vi.onnx.json')
```

Có thể chọn provider/tốc độ trong `.env`:

```dotenv
DUBVI_TTS_PROVIDER=piper
DUBVI_PIPER_SPEED=fast
# Chỉ cần đặt nếu model không ở vị trí mặc định:
# DUBVI_PIPER_MODEL=C:\\path\\to\\pretrained_vi.onnx
```

`capcut` và `edge` vẫn dùng được khi đặt rõ `DUBVI_TTS_PROVIDER`. Piper load
model một lần mỗi process rồi tái sử dụng cho các segment tiếp theo.

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
5. Piper TTS local tạo voice từng segment; FFmpeg chỉ tăng nhịp trong giới hạn `maxTempo`.
6. Nếu voice vẫn dài, worker chuyển job sang `awaiting_review` thay vì cắt lời hoặc làm giọng méo.
7. FFmpeg blur fixed ROI theo từng frame, render ASS/SRT tiếng Việt, mix audio và xuất MP4.

## Privacy & giới hạn

Video, WAV, frame OCR, ASS/SRT, voice file, MP4 output, model cache và job log đều nằm trên máy local. Chỉ token, inventory file (tên/kích thước/thời gian), config, segment text, tiến trình và đường dẫn output local được trao đổi với control plane. Correction LLM chạy server-side qua model tích hợp và chỉ nhận metadata/text, không nhận file video hoặc frame; mỗi lần correction có thể phát sinh usage model. Piper TTS mặc định là offline; Edge TTS chỉ gửi text tiếng Việt tới Microsoft khi được chọn rõ qua `DUBVI_TTS_PROVIDER=edge`. Kiểm tra license của package `piper-tts` và model upstream trước khi phân phối lại sản phẩm.
