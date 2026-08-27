import json
import time
from pathlib import Path
from dubvi_worker import Settings, transcribe, synthesize, fit_voice, draw_ass, draw_srt, ffprobe_dimensions
from llm_translator import translate_segments_native

settings = Settings.from_env()
audio_wav = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\source_full.wav")
out_dir = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full")

print("1. Transcribing with Whisper...")
segments = transcribe(audio_wav, "tiny", "cpu")
print(f"✓ Whisper found {len(segments)} segments.")

print("\n2. Translating with Gemini 2.5 Flash...")
t0 = time.time()
for s in segments:
    s["sourceTextZh"] = s.get("asrTextZh", "")

segments = translate_segments_native(segments, "gemini", gemini_key=settings.gemini_api_key)
print(f"✓ Gemini translated {len(segments)} segments in {time.time() - t0:.2f}s:")
for i, s in enumerate(segments):
    print(f"  [{i:02d}] {s.get('translatedTextVi')}")

print("\n3. Generating CapCut voices for all segments...")
voice_inputs = []
for idx, seg in enumerate(segments):
    raw_voice = out_dir / f"voice_{idx:03d}_raw.mp3"
    fitted_voice = out_dir / f"voice_{idx:03d}.mp3"
    slot_ms = seg["endMs"] - seg["startMs"]
    
    t_v0 = time.time()
    synthesize(seg["translatedTextVi"], settings.capcut_voice, raw_voice, settings)
    fit_ms, _ = fit_voice(raw_voice, fitted_voice, slot_ms, max_tempo=1.35)
    seg["voicePath"] = str(fitted_voice)
    seg["voiceDurationMs"] = fit_ms
    seg["endMs"] = seg["startMs"] + fit_ms
    voice_inputs.append((fitted_voice, seg["startMs"]))
    print(f"  ✓ Voice {idx:02d}: '{seg['translatedTextVi']}' ({fit_ms/1000:.2f}s in {time.time()-t_v0:.2f}s)")

print("\n🎉 TOÀN BỘ 29 VOICE CAPCUT ĐÃ SINH XONG!")
