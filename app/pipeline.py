"""The transcription pipeline: normalise -> (chunk) -> transcribe -> merge -> post-process."""
from __future__ import annotations

import re
from pathlib import Path

from . import audio, config
from .transcriber import Segment, Transcriber


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def merge_chunk_segments(chunk_results: list[tuple[float, list[Segment]]],
                         dedupe_tolerance: float = 0.2) -> list[Segment]:
    """Shift each chunk's timestamps by its offset and drop segments that
    were already emitted from the overlap with the previous chunk."""
    merged: list[Segment] = []
    last_end = 0.0
    for offset, segments in chunk_results:
        for s in segments:
            start, end = s.start + offset, s.end + offset
            if merged and start < last_end - dedupe_tolerance:
                continue
            merged.append(Segment(start=start, end=end, text=s.text, confidence=s.confidence,
                                  words=[{**w, "start": w["start"] + offset, "end": w["end"] + offset}
                                         for w in s.words]))
            last_end = end
    return merged


def run(src_path: str | Path, work_dir: str | Path, engine: Transcriber,
        language: str | None = None, word_timestamps: bool = False,
        progress=lambda pct: None) -> dict:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    duration = audio.probe(src_path)
    if duration > config.MAX_DURATION_SEC:
        raise audio.AudioError(f"Audio is {duration:.0f}s; the limit is {config.MAX_DURATION_SEC}s.")

    wav = audio.to_wav(src_path, work_dir / "normalised_16k.wav")
    progress(5)

    chunks = audio.cut_chunks(wav, duration, work_dir / "chunks",
                              target=config.CHUNK_SECONDS, overlap=config.CHUNK_OVERLAP_SECONDS)

    results, detected_lang, lang_prob, prev_tail = [], language, None, None
    for i, chunk in enumerate(chunks):
        r = engine.transcribe(str(chunk.path), language=detected_lang,
                              initial_prompt=prev_tail, word_timestamps=word_timestamps)
        if detected_lang is None:            # detect once, then pin it for later chunks
            detected_lang, lang_prob = r.language, r.language_probability
        results.append((chunk.offset, r.segments))
        if r.segments:
            prev_tail = r.segments[-1].text
        progress(5 + int(90 * (i + 1) / len(chunks)))

    segments = merge_chunk_segments(results)
    out_segments = []
    for i, s in enumerate(segments):
        item = {"id": i, "start": round(s.start, 2), "end": round(s.end, 2),
                "text": _clean(s.text), "confidence": s.confidence}
        if word_timestamps:
            item["words"] = [{**w, "start": round(w["start"], 2), "end": round(w["end"], 2)} for w in s.words]
        out_segments.append(item)

    progress(100)
    return {
        "duration_sec": round(duration, 2),
        "language": detected_lang,
        "language_probability": lang_prob,
        "chunks": len(chunks),
        "text": _clean(" ".join(s["text"] for s in out_segments)),
        "segments": out_segments,
    }


# ---------- export formats ----------

def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(result: dict) -> str:
    blocks = [f"{i + 1}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text']}\n"
              for i, s in enumerate(result["segments"])]
    return "\n".join(blocks)


def to_vtt(result: dict) -> str:
    body = "\n".join(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text']}\n"
                     for s in result["segments"])
    return "WEBVTT\n\n" + body
