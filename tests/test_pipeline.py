"""Tests run without downloading a model: a fake transcriber stands in for Whisper,
while FFmpeg handling, chunking, merging, exports and the API are exercised for real."""
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import audio, config, jobs, pipeline
from app.transcriber import Segment, TranscriptionResult, set_transcriber


class FakeTranscriber:
    """Emits one segment per 2 s of audio so timestamps are predictable."""
    model_name = "fake"

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0

    def transcribe(self, wav_path, language=None, initial_prompt=None, word_timestamps=False):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated GPU out of memory")
        dur = audio.probe(wav_path)
        segs, t = [], 0.0
        while t + 2 <= dur + 1e-6:
            segs.append(Segment(t, t + 2, f"words at {t:.0f}", 0.9,
                                [{"start": t, "end": t + 1, "word": "words", "probability": 0.9}]))
            t += 2
        return TranscriptionResult("en", 0.99, segs)


def make_audio(path: Path, seconds: int, fmt_args=()):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-ac", "2", "-ar", "44100",
                    *fmt_args, str(path)], check=True)
    return path


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    set_transcriber(FakeTranscriber())


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".m4a", ".flac", ".ogg"])
def test_any_format_is_normalised_to_16k_mono_wav(tmp_path, ext):
    src = make_audio(tmp_path / f"in{ext}", 4)
    wav = audio.to_wav(src, tmp_path / "out.wav")
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                           "stream=sample_rate,channels,codec_name", "-of", "csv=p=0", str(wav)],
                          capture_output=True, text=True).stdout.strip()
    assert info == "pcm_s16le,16000,1"


def test_corrupt_file_is_rejected(tmp_path):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not really audio")
    with pytest.raises(audio.AudioError):
        audio.probe(bad)


def test_plan_chunks_prefers_silence():
    plan = plan = audio.plan_chunks(25, 10, silences=[9.2, 19.5])
    assert plan == [(0.0, 9.2), (9.2, 19.5), (19.5, 25)]


def test_merge_shifts_offsets_and_drops_overlap_duplicates():
    a = [Segment(0, 4, "a"), Segment(4, 9, "b")]
    b = [Segment(0, 1, "b-dup"), Segment(1, 5, "c")]      # chunk 2 starts at 8 s (1 s overlap)
    merged = pipeline.merge_chunk_segments([(0.0, a), (8.0, b)])
    assert [(s.start, s.end, s.text) for s in merged] == [(0, 4, "a"), (4, 9, "b"), (9.0, 13.0, "c")]


def test_long_audio_is_chunked_and_timestamps_are_absolute(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHUNK_SECONDS", 10)
    src = make_audio(tmp_path / "long.mp3", 30)
    result = pipeline.run(src, tmp_path / "work", FakeTranscriber())
    assert result["chunks"] == 3
    starts = [s["start"] for s in result["segments"]]
    assert starts == sorted(starts) and starts[-1] >= 20
    assert result["language"] == "en"


def test_srt_export():
    result = {"segments": [{"start": 0, "end": 2.5, "text": "Hello"}]}
    assert pipeline.to_srt(result) == "1\n00:00:00,000 --> 00:00:02,500\nHello\n"


def test_transient_errors_are_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    engine = FakeTranscriber(fail_times=1)
    set_transcriber(engine)
    src = make_audio(tmp_path / "a.wav", 4)
    job = jobs.create(src, "a.wav", {})
    job = jobs.run(job["id"])
    assert job["status"] == "completed" and job["attempts"] == 2


def test_api_sync_transcription_with_timestamps(tmp_path):
    src = make_audio(tmp_path / "speech.mp3", 6)
    with TestClient(__import__("app.main", fromlist=["app"]).app) as client:
        r = client.post("/v1/transcriptions", files={"file": ("speech.mp3", src.read_bytes(), "audio/mpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "completed"
        assert [s["start"] for s in body["segments"]] == [0, 2, 4]
        srt = client.get(f"/v1/transcriptions/{body['job_id']}?format=srt")
        assert "00:00:02,000 --> 00:00:04,000" in srt.text


def test_api_rejects_unsupported_extension():
    from app.main import app
    with TestClient(app) as client:
        r = client.post("/v1/transcriptions", files={"file": ("notes.txt", b"hi", "text/plain")})
        assert r.status_code == 400


def test_missing_cuda_libs_fall_back_to_cpu(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from app.transcriber import FasterWhisperTranscriber

    class Model:
        def __init__(self, fail):
            self.fail = fail

        def transcribe(self, samples, **kw):
            if self.fail:
                raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
            seg = SimpleNamespace(start=0.0, end=1.0, text=" hi", avg_logprob=-0.1, words=None)
            return iter([seg]), SimpleNamespace(language="en", language_probability=0.99)

    loaded = []
    monkeypatch.setattr(FasterWhisperTranscriber, "_load",
                        lambda self, device, ct: loaded.append(device) or Model(fail=device != "cpu"))
    engine = FasterWhisperTranscriber(model_size="tiny", device="auto")
    wav = audio.to_wav(make_audio(tmp_path / "a.mp3", 2), tmp_path / "a.wav")
    result = engine.transcribe(str(wav))
    assert loaded == ["auto", "cpu"] and engine.device == "cpu"
    assert result.segments[0].text == "hi"
