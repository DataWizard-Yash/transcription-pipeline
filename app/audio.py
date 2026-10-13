"""Audio validation, normalisation and chunking (all via FFmpeg / ffprobe).

Design choice: every input is converted to ONE canonical format - 16 kHz,
mono, 16-bit PCM WAV - which is what Whisper expects. The rest of the
pipeline therefore never has to care about mp3 vs m4a vs video files.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AudioError(Exception):
    """Permanent problem with the input file (corrupt, no audio, ...). Not retried."""


@dataclass
class Chunk:
    path: Path
    offset: float      # start of this chunk within the original audio, in seconds
    duration: float


def probe(path: str | Path) -> float:
    """Return duration in seconds; raise AudioError if there is no readable audio stream."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AudioError("File could not be read as audio (corrupt or unsupported format).")
    info = json.loads(proc.stdout or "{}")
    if not any(s.get("codec_type") == "audio" for s in info.get("streams", [])):
        raise AudioError("File contains no audio stream.")
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError):
        raise AudioError("Could not determine audio duration.")


def to_wav(src: str | Path, dst: str | Path, loudnorm: bool = False) -> Path:
    """Convert any FFmpeg-readable input (incl. video) to 16 kHz mono PCM WAV."""
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-map", "0:a:0",          # first audio track only
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    if loudnorm:
        cmd[cmd.index("-c:a"):cmd.index("-c:a")] = ["-af", "loudnorm"]
    cmd.append(str(dst))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"Audio conversion failed: {proc.stderr.strip()[:300]}")
    return Path(dst)


def silence_points(wav: str | Path, noise_db: int = -35, min_silence: float = 0.4) -> list[float]:
    """Midpoints of silent stretches, used as natural places to cut long audio."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(wav), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    points, start = [], None
    for line in proc.stderr.splitlines():
        if "silence_start:" in line:
            start = float(line.split("silence_start:")[1].split()[0])
        elif "silence_end:" in line and start is not None:
            end = float(line.split("silence_end:")[1].split()[0])
            points.append((start + end) / 2)
            start = None
    return points


def plan_chunks(duration: float, target: float, silences: list[float]) -> list[tuple[float, float]]:
    """Split [0, duration] into ~target-second pieces, preferring to cut at silence.

    A cut is moved to the nearest silence within +/-15% of the target length,
    so words are not cut in half. Returns (start, end) pairs without overlap;
    overlap is added when the chunk files are cut.
    """
    if duration <= target:
        return [(0.0, duration)]
    cuts, pos = [], 0.0
    window = target * 0.15
    while duration - pos > target + window:   # avoid leaving a tiny tail chunk
        ideal = pos + target
        near = [s for s in silences if ideal - window <= s <= ideal + window]
        cut = min(near, key=lambda s: abs(s - ideal)) if near else ideal
        cuts.append(cut)
        pos = cut
    bounds = [0.0, *cuts, duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def cut_chunks(wav: str | Path, duration: float, out_dir: str | Path,
               target: float, overlap: float) -> list[Chunk]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_chunks(duration, target, silence_points(wav))
    chunks = []
    for i, (start, end) in enumerate(plan):
        s = max(0.0, start - (overlap if i else 0.0))
        e = min(duration, end + overlap)
        path = out_dir / f"chunk_{i:04d}.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                        "-i", str(wav), "-c", "copy", str(path)], check=True)
        chunks.append(Chunk(path=path, offset=s, duration=e - s))
    return chunks
