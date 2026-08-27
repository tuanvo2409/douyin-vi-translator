"""Re-render 1080p Master with 100% full coverage golden standard box:
"""
import subprocess
import time
from pathlib import Path
from dubvi_worker import draw_ass, draw_srt

source_video = Path(r"C:\Users\vmath\Downloads\video douyin raw\MS4wLjABAAAAwAhJTV1V81xlr9MWGYk30jeWfDSy4CkGGfxv0Aj7IJmvmHxDGwU221itoQ6tsTLY\7615935754560097570\7615935754560097570_video.mp4")
out_dir = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full")
clean_bgm_wav = out_dir / "clean_bgm.wav"

width, height = 1080, 1920

script_data = [
    (460, 2180, "Món đồ siêu mê cho góc nhỏ của mình nè!"),
    (3420, 2230, "Lại dời tủ nhỏ ra cuối giường rồi nè."),
    (5880, 1560, "Thiết kế tủ siêu mỏng gọn."),
    (7520, 2160, "Thế là vừa khít giữa giường với tủ luôn."),
    (10920, 2420, "Hai tầng mở, lấy đồ tiện lợi cực kỳ."),
    (13840, 1970, "Chiều rộng để sách cũng vừa vặn luôn."),
    (21320, 2280, "Mặt trước còn hút nam châm để trang trí nữa."),
    (26240, 2450, "Đồ dùng hàng ngày, với tay là có liền."),
    (28800, 1970, "Có bánh xe di chuyển siêu mượt luôn."),
    (31000, 1150, "Dọn dẹp cũng dễ dàng."),
    (32159, 1340, "Vừa làm bàn đầu giường"),
    (33840, 2040, "Lại còn thành tủ đựng đồ cuối giường"),
    (36460, 1490, "Chiếc thảm đỏ hằng mong ước"),
    (38140, 1180, "Mình cũng chốt đơn luôn!"),
    (40320, 1760, "Chất liệu đay đỏ cực kỳ có gu"),
    (42100, 1250, "Lại còn dày dặn nữa"),
    (43380, 2210, "Mỗi tội đi chân trần hơi cứng xíu nha"),
    (45860, 1780, "Thảm đay đỏ nhìn hoài vẫn mê"),
    (48520, 1100, "Nói thật lòng nha"),
    (49640, 1510, "Trải thảm này vô phòng là..."),
    (51300, 2720, "Nhìn bàn gọn gàng, sạch sẽ là thấy có gu liền nè."),
    (54040, 1450, "Tủ sắt này, bao mê luôn!"),
    (55520, 1200, "Khen cả vạn lần!"),
    (56740, 1200, "Cùng mẫu trong phòng."),
    (58120, 1730, "Tổng cộng có ba kiểu ngăn kéo."),
    (60360, 2350, "Bên trong còn chia ngăn, cực kỳ tiện lợi."),
    (65550, 2620, "Giờ đống đồ lặt vặt được sắp xếp gọn gàng, hợp lý."),
    (70790, 2880, "Mình mê cái cảm giác mọi thứ được sắp xếp ngăn nắp này lắm."),
    (74610, 1390, "Nhìn xịn xò ghê chưa!")
]

segments = []
voice_inputs = []
for idx, (start_ms, dur_ms, text) in enumerate(script_data):
    voice_file = out_dir / f"voice_{idx:03d}.mp3"
    end_ms = start_ms + dur_ms
    segments.append({
        "position": idx,
        "startMs": start_ms,
        "endMs": end_ms,
        "translatedTextVi": text
    })
    voice_inputs.append((voice_file, start_ms))

roi = {"xPercent": 2.0, "yPercent": 66.0, "widthPercent": 96.0, "heightPercent": 9.8}

ass_path = out_dir / "7615935754560097570_1080p.ass"
srt_path = out_dir / "7615935754560097570_1080p.srt"
draw_ass(segments, width, height, roi, ass_path)
draw_srt(segments, srt_path)

w_px, h_px = int(width * 0.96), int(height * 0.098)
x_px, y_px = int(width * 0.02), int(height * 0.660)

ass_posix = ass_path.as_posix()
if len(ass_posix) >= 2 and ass_posix[1] == ":":
    ass_posix = ass_posix[0] + "\\:" + ass_posix[2:]

video_filter = (
    f"[0:v]scale=1080:1920,split=2[base][ref];"
    f"[ref]crop={w_px}:{h_px}:{x_px}:{y_px},boxblur=24:3:24:3,drawbox=x=0:y=0:w={w_px}:h={h_px}:color=white@0.14:t=fill[blur];"
    f"[base][blur]overlay={x_px}:{y_px},subtitles='{ass_posix}'[video]"
)

cmd = ["ffmpeg", "-y", "-i", str(source_video), "-i", str(clean_bgm_wav)]
audio_labels = []
for idx, (voice_path, offset_ms) in enumerate(voice_inputs, start=1):
    cmd.extend(["-i", str(voice_path)])
    input_idx = idx + 1
    audio_labels.append(f"[{input_idx}:a]adelay={offset_ms}|{offset_ms}[v{idx}]")

all_v_tags = "".join(f"[v{i}]" for i in range(1, len(voice_inputs) + 1))
mix_voice = f"{';'.join(audio_labels)};{all_v_tags}amix=inputs={len(voice_inputs)}:duration=longest:normalize=0[allvoice]"
audio_filter = f"{mix_voice};[1:a]volume=0.85[bgm];[bgm][allvoice]amix=inputs=2:duration=first:normalize=0[finalaudio]"

final_output = out_dir / "7615935754560097570_1080p_master_vi.mp4"

full_cmd = cmd + [
    "-filter_complex", f"{video_filter};{audio_filter}",
    "-map", "[video]", "-map", "[finalaudio]",
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
    "-c:a", "aac", "-shortest", "-movflags", "+faststart", str(final_output)
]

print("🎞️ Đang render lại Video 1080p với Box Chuẩn Vàng 100% che sạch chữ Trung...")
t0 = time.time()
subprocess.run(full_cmd, check=True)
print(f"🎉 RENDER THÀNH CÔNG trong {time.time() - t0:.1f}s!")
