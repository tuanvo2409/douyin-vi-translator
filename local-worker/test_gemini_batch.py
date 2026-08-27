import json
from dubvi_worker import Settings, transcribe
from llm_translator import translate_with_gemini
from pathlib import Path

settings = Settings.from_env()
audio_wav = Path(r"C:\Users\vmath\Videos\douyin\dubvi-output\7615935754560097570-full\source_full.wav")

segs = transcribe(audio_wav, "tiny", "cpu")
print(f"Transcribed {len(segs)} segments.")
for s in segs:
    s["sourceTextZh"] = s.get("asrTextZh", "")

res = translate_with_gemini(segs, settings.gemini_api_key)
print("SUCCESS GEMINI! Count:", len(res))
for i, s in enumerate(res):
    print(f"{i:02d}: {s.get('translatedTextVi')}")
