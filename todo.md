# Việc cần hoàn thành cho pipeline video thực tế

## Đã chốt kiến trúc

- [x] Thiết lập web làm control plane; worker Python local không nhận video từ máy chủ công khai.
- [x] Quy ước thư mục local cho nguồn video, model cache, job manifest, artefact và log.
- [x] Tạo token cặp nối giữa giao diện web và worker local.
- [x] Ghi log worker/job vào thư mục local và quy ước cache model tách riêng.

- [x] Chốt đường chạy: app web quản lý job, worker Python xử lý FFmpeg/OCR/ASR/TTS ở môi trường có đủ tài nguyên.
- [x] Nâng project lên full-stack để có API server, database, storage và trạng thái job.
- [x] Thiết kế bảng `video_jobs`, `video_segments` và `worker_clients`.
- [x] Tạo API tạo pairing token, heartbeat worker, claim job, báo tiến trình và lưu segment.
- [x] Xây worker Python nhận job, probe video, tạo fixed ROI blur và render MP4 bằng FFmpeg.
- [x] Tích hợp mã nguồn ASR tiếng Trung, OCR đa khung hình trong fixed ROI và cơ chế fusion/confidence.
- [x] Tích hợp mã nguồn dịch local theo giới hạn thời lượng và TTS tiếng Việt theo từng segment.
- [x] Tạo duration fitter cho audio, mix voice/background và sinh SRT/ASS.
- [x] Kết nối UI hiện tại với API, polling trạng thái job và pairing local worker.
- [x] Smoke test FFmpeg fixed blur + ASS trên đoạn video mẫu 7 giây.
- [x] Xác thực artefact SRT cùng ASS và MP4 render trong smoke test.
- [x] Cài thư viện/model local và chạy full smoke pipeline có OCR, dịch Marian, TTS, duration fit và FFmpeg render trên video mẫu.
- [x] Bổ sung editor lưu sửa segment sau khi worker trả trạng thái `awaiting_review`, tạo lại TTS/duration fit rồi xếp lại worker render local.
- [x] Kiểm thử luồng review rerender với lời Việt đã sửa, xác thực voice mới, SRT/ASS và MP4 output được tạo.

## Tối ưu OCR theo ngữ cảnh

- [x] Xác định ngưỡng chỉ gọi LLM cho OCR confidence thấp hoặc có mâu thuẫn đa khung hình/ASR.
- [x] Mở rộng segment với audit trail: OCR gốc, OCR đã sửa, confidence, lý do và trạng thái cần duyệt.
- [x] Tạo endpoint server-side gọi LLM theo JSON schema để sửa lỗi chính tả tiếng Trung có kiểm soát.
- [x] Nâng worker gửi context OCR/ASR đến endpoint, nhưng giữ nguyên OCR khi LLM trả confidence thấp.
- [x] Hiển thị bản gốc và bản LLM sửa trong review queue để người dùng quyết định.
- [x] Viết unit test policy correction và smoke test với OCR video mẫu.
- [x] Hiển thị song song OCR gốc, ASR và đề xuất LLM; cho phép áp dụng từng nguồn vào text Trung trước khi lưu review.
