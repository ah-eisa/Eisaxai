#!/usr/bin/env python3
"""
Voice cloning with Coqui XTTS v2.
Usage: python3 voice_clone.py "النص هنا" output.wav
"""
import sys
import os

SAMPLE_PATH = "/home/ubuntu/investwise/data/voice_samples/sample_933252341.ogg"
MODEL_NAME  = "tts_models/multilingual/multi-dataset/xtts_v2"

def clone(text: str, out_path: str, lang: str = "ar") -> None:
    from TTS.api import TTS
    tts = TTS(MODEL_NAME, progress_bar=False)
    tts.tts_to_file(
        text=text,
        speaker_wav=SAMPLE_PATH,
        language=lang,
        file_path=out_path,
    )
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    text     = sys.argv[1] if len(sys.argv) > 1 else "مرحباً، أنا صوت مستنسخ بالذكاء الاصطناعي"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/cloned_output.wav"
    lang     = sys.argv[3] if len(sys.argv) > 3 else "ar"
    clone(text, out_path, lang)
