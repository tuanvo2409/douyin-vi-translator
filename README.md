# 🎬 DUBVI STUDIO — AI Douyin Video Translator & Dubbing

> **Hệ thống AI tự động chuyển ngữ, lồng tiếng và làm phụ đề video Douyin / TikTok sang tiếng Việt chất lượng cao Full HD 1080p.**

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python)
![Framework](https://img.shields.io/badge/UI-NiceGUI%20%2B%20AG%20Grid-indigo)
![AI](https://img.shields.io/badge/AI-Demucs%20%7C%20Whisper%20%7C%20Gemini%202.5-emerald)
![TTS](https://img.shields.io/badge/TTS-CapCut%20TikTok%20Official-purple)
![Output](https://img.shields.io/badge/Render-FFmpeg%201080p%20Master-orange)

---

## 🌟 Tính Năng Nổi Bật

- 🧠 **Tách Nhạc Nền Bằng AI (Demucs v4)**: Bóc tách sạch \%$ giọng đọc tiếng Trung bản xứ, bảo toàn trọn vẹn nhạc nền (BGM) và các hiệu ứng âm thanh gốc của video.
- 🗣️ **Nhận Diện Giọng Nói Siêu Tốc (Faster-Whisper)**: Trích xuất lời thoại tiếng Trung khớp theo từng mili-giây (timestamp chính xác).
- 🌐 **Dịch Thuật Bản Xứ Bằng Gemini 2.5 Flash**: Chuyển ngữ mượt mà theo văn phong review, lifestyle, trend TikTok Việt Nam; tích hợp cơ chế bộ nhớ đệm (Cache) tái sử dụng tức thì (.01	ext{s}$).
- 🎙️ **Lồng Tiếng AI CapCut TikTok**: Tổng hợp giọng đọc tiếng Việt truyền cảm, tự nhiên (Giọng Mai review, Minh Quang nam trầm, Ngọc Mai, Hải Đăng...).
- 🛡️ **Kính Mờ Thông Minh (Smart Frosted Glass Mask)**:
  - Tự động dò đúng dải phụ đề của giọng nói ở /3$ dưới màn hình ( \approx 66.5\%$).
  - Bỏ qua các sticker, caption tò mò ở giữa màn hình không có giọng nói.
  - **Khử nhấp nháy (Smart Speech Clustering)**: Chỉ bật kính mờ khi có tiếng nói, tự động tắt trả lại khung hình gốc khi người nói nghỉ $\ge 1.8\text{s}$.
- 🎛️ **Studio Web Trực Quan (http://localhost:8080)**:
  - Bấm nút native mở File Explorer chọn video trực tiếp từ máy tính.
  - Bảng kịch bản **AG Grid** cho phép nghe thử từng câu và chỉnh sửa câu chữ trực tiếp.
  - Chức năng **Xem Thử 3 Giây (3s Test Clip)** kiểm tra kính mờ và cỡ chữ trước khi xuất file.
  - Trình phát video Master 1080p tích hợp sẵn trên giao diện.

---

## 📋 Yêu Cầu Hệ Thống

1. **Hệ điều hành**: Windows 10/11, macOS hoặc Linux.
2. **Python**: Phiên bản 3.10 hoặc 3.11+ (Khuyên dùng Python 3.11).
3. **FFmpeg**: Đã cài đặt và thêm vào biến môi trường PATH.
   - Kiểm tra bằng lệnh: fmpeg -version
4. **Phần cứng**:
   - RAM: Tối thiểu 8GB (khuyên dùng 16GB).
   - CPU: 4 cores trở lên (hỗ trợ tăng tốc GPU NVIDIA nếu có CUDA).

---

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Clone Repository
\\ash
git clone https://github.com/tuanvo2409/douyin-vi-translator.git
cd douyin-vi-translator/local-worker
\
### Bước 2: Tạo môi trường ảo và cài đặt thư viện
\\ash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt môi trường ảo:
# Trên Windows (PowerShell / CMD):
.\.venv\Scripts\activate
# Trên macOS / Linux:
source .venv/bin/activate

# Cài đặt các dependencies cần thiết
pip install -r requirements.txt
\
### Bước 3: Cấu hình biến môi trường (\.env\)
Sao chép file mẫu và điền thông tin API:
\\ash
cp worker.env.template .env
\
Mở file \.env\ và cập nhật các thông số:
\\ini
# API Key của Google Gemini (dùng để dịch thuật kịch bản)
GEMINI_API_KEY=AIzaSy...

# Thư mục chứa video Douyin tải về máy
DUBVI_MEDIA_DIR=C:/Users/vmath/Videos/douyin

# Giọng đọc mặc định (BV421_vivn_streaming: Mai CapCut, BV007_streaming: Minh Quang)
DUBVI_TTS_PROVIDER=capcut
DUBVI_CAPCUT_VOICE=BV421_vivn_streaming
\
---

## 🖥️ Hướng Dẫn Sử Dụng Giao Diện DUBVI Studio

### 1. Khởi động Web Studio:
Tại thư mục \local-worker\, chạy lệnh:
\\ash
python studio_app.py
\Ứng dụng sẽ tự động mở tại trình duyệt: **[http://localhost:8080](http://localhost:8080)**

---

### 2. Quy trình làm việc 4 bước đơn giản:

1. **Chọn Video**: Bấm nút **\[ 📂 BẤM ĐỂ CHỌN VIDEO TỪ MÁY... ]\** $\rightarrow$ File Explorer mở lên để bạn chọn video \.mp4\.
2. **Xem Thử Kính Mờ**: Bấm **\⚡ Xem Thử Mask & Cỡ Chữ (3s)\** để kiểm tra vị trí kính mờ và kiểu chữ trước khi làm toàn bộ.
3. **Bóc Tách & Dịch AI**: Bấm **. Bóc Tách & Dịch AI\** $\rightarrow$ AI sẽ tự động tách BGM, nhận diện lời thoại và dịch sang tiếng Việt.
4. **Kiểm Tra & Tinh Chỉnh**: Đọc các câu thoại trên bảng **AG Grid**, chỉnh sửa lại từ ngữ theo ý thích (bấm đúp vào ô để sửa), bấm nghe thử từng câu.
5. **Xuất Video Hoàn Thiện**: Bấm **. Sinh Voice & Render 1080p\** $\rightarrow$ Hệ thống tự động lồng tiếng CapCut, hòa trộn nhạc nền và xuất video Full HD 1080p.
6. **Thưởng Thức**: Bấm **\▶ XEM VIDEO HOÀN THIỆN\** để phát trực tiếp thành phẩm trên giao diện!

---

## 📁 Cấu Trúc Thư Mục Tối Giản (Lean Architecture)

```plaintext
douyin-vi-translator/
├── local-worker/                 # Toàn bộ mã nguồn hệ thống
│   ├── studio_app.py             # Giao diện Web NiceGUI Studio chính (Port 8080)
│   ├── auto_roi.py               # AI dò dải phụ đề & Quét Sub Câm (RapidOCR)
│   ├── dubvi_worker.py           # Pipeline xử lý chính (Whisper, CapCut TTS, FFmpeg 1080p)
│   ├── llm_translator.py         # Chuyển ngữ Gemini 2.5 Flash & Ma trận 8 Hook Viral
│   ├── gemini_pool.py            # Quản lý tự động xoay tua API Key Gemini
│   ├── capcut_tts_api/           # Bộ kết nối API CapCut TikTok TTS
│   ├── Voice.json                # Danh mục mã giọng đọc CapCut
│   └── requirements.txt          # Danh sách thư viện Python
├── .gitignore                    # Bộ lọc bảo mật
└── README.md                     # Tài liệu hướng dẫn sử dụng
```
---

## ⚙️ Các Lựa Chọn Giọng Đọc & Tùy Biến

| Mã Giọng | Tên Giọng | Thể Loại Phù Hợp |
|---|---|---|
| \BV421_vivn_streaming\ | **Mai (CapCut)** | Nữ truyền cảm, review sản phẩm, vlog đời sống Douyin |
| \BV007_streaming\ | **Minh Quang (CapCut)** | Nam trầm ấm, kể chuyện, video công nghệ |
| \BV001_streaming\ | **Ngọc Mai (CapCut)** | Nữ nhẹ nhàng, tâm sự, video thẩm mỹ |
| \BV004_streaming\ | **Hải Đăng (CapCut)** | Nam review sôi nổi, video đồ gia dụng |
| \i-VN-HoaiMyNeural\ | **Hoài My (Edge-TTS)** | Nữ phát thanh viên chuẩn truyền hình |
| \i-VN-NamMinhNeural\ | **Nam Minh (Edge-TTS)**| Nam chuyên nghiệp, video tài liệu |

---

## 🔒 Bảo Mật & Quyền Riêng Tư

- **100% Video & Audio lưu trữ cục bộ**: Mọi file video gốc, file nhạc nền WAV, giọng đọc MP3 và video Master 1080p đều được lưu trực tiếp trên ổ cứng máy bạn (\C:\Users\<user>\Videos\douyin\dubvi-output\\).
- **Không bao giờ tải video lên Cloud**: Chỉ có các đoạn văn bản ngắn được gửi tới Gemini API để dịch thuật, bảo đảm tốc độ tối đa và tuyệt đối an toàn dữ liệu.

---

## 🤝 Đóng Góp & Phát Triển
Mọi đóng góp (Pull Request / Issue) nhằm tối ưu hóa hiệu năng, thêm giọng đọc hoặc nâng cấp bộ lọc phụ đề đều được hoan nghênh!

⭐ *Nếu thấy dự án hữu ích, đừng quên thả 1 Star trên GitHub nhé!*