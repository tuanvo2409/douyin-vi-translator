# Định hướng thiết kế — Douyin → Việt Translator

## Ba hướng thẩm mỹ

### 1. Bàn dựng phim biên tập
**Giới thiệu ngắn:** Một không gian dựng video mang cảm giác studio hậu kỳ, nhấn vào preview dọc, timeline và các quyết định biên tập rõ ràng. Giao diện tạo cảm giác công cụ nghiêm túc nhưng vẫn dễ tiếp cận.
**Xác suất:** 0.07

### 2. Sổ tay ngôn ngữ chuyển động
**Giới thiệu ngắn:** Giao diện sáng, mềm, giống một cuốn sổ ghi chú song ngữ với nhịp đọc và thời lượng là trung tâm. Hướng này thân thiện, giàu tính biên tập nhưng ít cảm giác điều khiển video chuyên sâu.
**Xác suất:** 0.04

### 3. Phòng điều phối âm thanh
**Giới thiệu ngắn:** Một workspace tối với chỉ dấu thời gian sắc nét, meter âm thanh và tương tác dạng console. Hướng này ưu tiên cảm giác kỹ thuật và tốc độ.
**Xác suất:** 0.09

---

## Hướng đã chọn: Bàn dựng phim biên tập

### Design Movement
**Swiss editorial design** kết hợp ngôn ngữ giao diện của bàn dựng hậu kỳ. Trật tự typographic chặt chẽ, các đường dẫn nhịp thời gian và khối preview tạo cảm giác biên tập có chủ ý thay vì dashboard chung chung.

### Core Principles
1. **Video là tâm điểm:** Khung preview dọc luôn là yếu tố thị giác lớn nhất; mọi công cụ đều phục vụ quyết định trên preview.
2. **Timeline là ngôn ngữ chính:** Thời gian, segment và lớp xử lý được nhìn như những vật thể có thể kiểm soát.
3. **Kỹ thuật dễ đọc:** Thông tin OCR, voice và SRT dùng nhãn ngắn, màu có ý nghĩa và hierarchy rõ ràng; không phô diễn jargon.
4. **Can thiệp tối thiểu:** Mọi lựa chọn nhấn vào fixed blur box — một giải pháp đơn giản, ổn định, dễ preview.

### Color Philosophy
Nền **ink navy** sâu tạo không gian tập trung cho hình ảnh video. **Giấy ngà ấm** dành cho các panel biên tập để giảm mỏi mắt khi đọc song ngữ. **Cam hổ phách** là màu thao tác/chuyển đổi, gợi một “frame đang được chọn”; **xanh bạc hà** chỉ trạng thái hợp lệ/đã đồng bộ. Màu được dùng như tín hiệu, không phải trang trí.

### Layout Paradigm
Một **editorial split-workspace**: cột trái hẹp là thông tin project và các mode xử lý; vùng trung tâm là khung preview 9:16 có overlay blur box; cột phải là inspector cho segment được chọn. Bên dưới, timeline kéo ngang toàn màn hình như một dải phim. Tránh dashboard dạng thẻ xếp lưới ở trung tâm.

### Signature Elements
1. **Frame ruler:** Thước thời gian mảnh với tick 0.0–5.0s và playhead cam hổ phách.
2. **Blur region overlay:** Khung subtitle dạng glass/blur với viền chấm và điều khiển góc, nhắc rõ cách xử lý mà app chọn.
3. **Transcript strips:** Mỗi segment là một dải ba tầng gồm Trung văn, Việt ngữ và nhịp voice.

### Interaction Philosophy
Tương tác phải có cảm giác chỉnh sửa trực tiếp: click segment để đổi overlay preview, kéo thanh blur để thấy cường độ, chọn `Auto-fit voice` để cập nhật chỉ dấu timing. Mọi hành động thử nghiệm đều có phản hồi tại chỗ, không điều hướng rời workspace.

### Animation
Chỉ dùng chuyển động ngắn 140–220ms với `cubic-bezier(0.23, 1, 0.32, 1)`. Playhead và waveform chuyển nhẹ khi đổi segment; inspector trượt 8px cùng opacity khi segment thay đổi. Không dùng animation lặp hoặc glow neon. Tôn trọng `prefers-reduced-motion`.

### Typography System
**Space Grotesk** cho title, số thời gian và control label nhằm tạo cảm giác kỹ thuật, có cấu trúc. **Manrope** cho nội dung song ngữ và trợ giúp dài hơn để dễ đọc. Hệ thống cấp bậc: headline 30–36px/600; module title 13px/700; label uppercase 10px với tracking rộng; transcript 14–16px/500.

### Brand Essence
**Một bàn dựng local-first cho người muốn chuyển video Hoa ngữ thành trải nghiệm xem tiếng Việt có nhịp, có kiểm soát.**

Tính cách thương hiệu: **chính xác, điềm tĩnh, có nghề**.

### Brand Voice
Ngắn, kỹ thuật nhưng không khô; nói theo động từ biên tập thay vì khẩu hiệu chung chung.

Ví dụ: “Giữ nhịp gốc. Đặt lời Việt vào đúng chỗ.”

Ví dụ: “Blur vùng chữ — không phá chuyển động phía sau.”

### Wordmark & Logo
Logo là **hai khung subtitle chồng lệch**, khung trước có một đường playhead thẳng đứng ở giữa; tạo thành monogram trừu tượng gợi chữ `D` và `V` nhưng không dùng ký tự. Wordmark dùng Space Grotesk bold với khoảng cách chữ hẹp.

### Signature Brand Color
**Signal Amber — #F5A524**. Màu chỉ xuất hiện tại playhead, nút render và các trạng thái cần hành động để tạo nhận diện rõ ràng.
