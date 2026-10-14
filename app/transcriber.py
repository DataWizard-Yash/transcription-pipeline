"""Speech-to-text engines behind one small interface.

The pipeline only depends on `Transcriber.transcribe()`, so the open-source
model can be swapped for a hosted API (OpenAI, Deepgram, AssemblyAI) or a
fake one in tests without touching anything else.
"""
from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field
from typing import Protocol

from . import config


@dataclass
class Segment:
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[dict] = field(default_factory=list)


@dataclass
class TranscriptionResult:
    language: str | None
    language_probability: float | None
    segments: list[Segment]


class Transcriber(Protocol):
    def transcribe(self, wav_path: str, language: str | None = None,
                   initial_prompt: str | None = None,
                   word_timestamps: bool = False) -> TranscriptionResult: ...


def load_wav(path: str):
    """Read our normalised 16 kHz mono 16-bit WAV into float32 samples.

    The pipeline has already converted the audio with FFmpeg, so there is no
    need to decode it a second time inside faster-whisper (which goes through
    PyAV and breaks with some PyAV versions).
    """
    import numpy as np
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path} is not 16 kHz mono 16-bit PCM")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class FasterWhisperTranscriber:
    """faster-whisper (CTranslate2 Whisper): open source, CPU or GPU,
    ~4x faster and lighter than the reference implementation."""

    def __init__(self, model_size: str = config.WHISPER_MODEL,
                 device: str = config.WHISPER_DEVICE,
                 compute_type: str = config.WHISPER_COMPUTE_TYPE):
        from faster_whisper import WhisperModel   # imported lazily: heavy dependency
        self.model_name = f"faster-whisper/{model_size}"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_path, language=None, initial_prompt=None, word_timestamps=False):
        segments, info = self.model.transcribe(
            load_wav(wav_path),             # already-normalised samples, no second decode
            language=language,              # None -> auto-detect
            initial_prompt=initial_prompt,  # carries context across chunk boundaries
            vad_filter=True,                # skip silence; reduces hallucinated text
            beam_size=5,
            word_timestamps=word_timestamps,
        )
        out = []
        for s in segments:                  # generator: memory stays flat on long audio
            out.append(Segment(
                start=s.start, end=s.end, text=s.text.strip(),
                confidence=round(math.exp(s.avg_logprob), 3),
                words=[{"start": w.start, "end": w.end, "word": w.word.strip(),
                        "probability": round(w.probability, 3)} for w in (s.words or [])],
            ))
        return TranscriptionResult(info.language, round(info.language_probability, 3), out)


_engine: Transcriber | None = None


def get_transcriber() -> Transcriber:
    """Load the model once per process, not once per request."""
    global _engine
    if _engine is None:
        _engine = FasterWhisperTranscriber()
    return _engine


def set_transcriber(engine: Transcriber) -> None:
    """Dependency injection hook (tests, or switching to a hosted API)."""
    global _engine
    _engine = engine
