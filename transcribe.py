"""CLI: transcribe a local file without running the API.

    python transcribe.py samples/sample.mp3 --format srt
"""
import argparse
import json
import tempfile

from app import pipeline
from app.transcriber import get_transcriber


def main():
    p = argparse.ArgumentParser(description="Transcribe an audio file with timestamps")
    p.add_argument("audio")
    p.add_argument("--language", default=None)
    p.add_argument("--words", action="store_true", help="include word-level timestamps")
    p.add_argument("--format", choices=["json", "txt", "srt", "vtt"], default="json")
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as work:
        result = pipeline.run(a.audio, work, get_transcriber(),
                              language=a.language, word_timestamps=a.words)
    if a.format == "txt":
        print(result["text"])
    elif a.format == "srt":
        print(pipeline.to_srt(result))
    elif a.format == "vtt":
        print(pipeline.to_vtt(result))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
