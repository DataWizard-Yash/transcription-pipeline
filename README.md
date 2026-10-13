# Transcription Pipeline

A small, production-minded service that turns an audio file into text with
per-segment timestamps, ready for downstream use (search, summarisation,
subtitles, analytics).

- **Speech-to-text:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (open-source Whisper on CTranslate2)
- **API:** FastAPI (`POST /v1/transcriptions`, `GET /v1/transcriptions/{id}`)
- **Audio handling:** FFmpeg / ffprobe
- **Outputs:** JSON (segments + timestamps + confidence), plain text, SRT, VTT

---

## Quick start

Requires Python 3.10+ and FFmpeg on the PATH.

```bash
pip install -r requirements.txt

# CLI
python transcribe.py samples/your_audio.mp3              # JSON
python transcribe.py samples/your_audio.mp3 --format srt # subtitles
python transcribe.py samples/your_audio.mp3 --words      # word-level timestamps

# API
uvicorn app.main:app --reload
curl -F "file=@samples/your_audio.mp3" http://localhost:8000/v1/transcriptions
```

Interactive API docs: http://localhost:8000/docs

With Docker:

```bash
docker build -t transcription .
docker run -p 8000:8000 -e WHISPER_MODEL=small transcription
```

Tests (no model download needed - a fake transcriber stands in for Whisper,
while FFmpeg, chunking, merging, retries and the API run for real):

```bash
pytest -q
```

---

## Example response

```json
{
  "job_id": "3f2c9a...",
  "status": "completed",
  "duration_sec": 42.7,
  "language": "en",
  "language_probability": 0.99,
  "chunks": 1,
  "text": "Hello and welcome to the call. Today we will review the release plan.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 3.2, "text": "Hello and welcome to the call.", "confidence": 0.94},
    {"id": 1, "start": 3.2, "end": 7.8, "text": "Today we will review the release plan.", "confidence": 0.91}
  ],
  "model": "faster-whisper/small"
}
```

---

## How it works

```
upload ─► validate (extension + ffprobe) ─► normalise to 16 kHz mono WAV (FFmpeg)
       ─► split long audio at silences (~10 min chunks, 1 s overlap)
       ─► transcribe each chunk (faster-whisper, VAD on, context carried over)
       ─► merge: shift timestamps by chunk offset, drop overlap duplicates
       ─► post-process: clean text, language, confidence ─► JSON / TXT / SRT / VTT
```

| File | Responsibility |
|---|---|
| `app/main.py` | HTTP API, upload streaming, validation, sync vs async decision |
| `app/audio.py` | ffprobe validation, FFmpeg normalisation, silence detection, chunking |
| `app/transcriber.py` | STT engine behind a `Transcriber` interface (faster-whisper implementation) |
| `app/pipeline.py` | Orchestration, chunk merging, export formats |
| `app/jobs.py` | Job state, background execution, retries, crash recovery |
| `transcribe.py` | CLI entry point |
| `tests/` | Unit and API tests |

---

## Design decisions

### 1. Why faster-whisper
- Strong accuracy and multilingual support with automatic language detection.
- Runs on CPU or GPU; roughly 4x faster and lighter than the reference Whisper
  implementation, with int8 quantisation on CPU.
- Model size is a config value (`WHISPER_MODEL=tiny|small|large-v3`), so the
  same code runs cheaply on a laptop or accurately on a GPU.
- `vad_filter=True` skips silence, which speeds things up and reduces Whisper's
  known tendency to hallucinate text on silent sections.
- The engine sits behind a small `Transcriber` protocol. Swapping to a hosted
  API (OpenAI, Deepgram, AssemblyAI) or a fake for tests is one class, with no
  changes to the pipeline.
- The model is loaded once per process, not per request.

### 2. Accepting files
- Extension allow-list, **plus** ffprobe to confirm the file really has an
  audio stream - the extension alone is not trusted.
- Uploads are streamed to disk in 1 MB chunks, so large files never sit fully in memory.
- Size and duration limits return clear 400/413 errors instead of failing deep in the model.

### 3. Different audio formats
Everything is converted once to the canonical format Whisper expects -
**16 kHz, mono, 16-bit PCM WAV**. Anything FFmpeg can decode works (mp3, m4a/aac,
ogg/opus, flac, webm, and video files - only the first audio track is used).
Multi-channel audio is downmixed. Optional `loudnorm` helps with very quiet recordings.

### 4. Timestamps
Each Whisper segment carries start/end times; the API returns them per segment,
plus a confidence score (from the segment's average log-probability) so
downstream systems can flag low-confidence parts for human review. Word-level
timestamps are available on request. SRT/VTT exports cover subtitle use cases.

### 5. Long audio
- **Sync vs async:** audio up to 2 minutes is answered in the same request;
  longer audio returns `202 Accepted` with a job id and is processed in the
  background. Clients poll `GET /v1/transcriptions/{id}` (progress is reported).
- **Chunking at silences:** audio is cut into ~10-minute chunks, moving each cut
  to the nearest silence (FFmpeg `silencedetect`) so words are not cut in half,
  with 1 s overlap between chunks.
- **Merging:** each chunk's timestamps are shifted by its offset, and segments
  that repeat inside the overlap are dropped.
- **Context continuity:** the last sentence of the previous chunk is passed as
  Whisper's `initial_prompt`, keeping names and terminology consistent across
  boundaries. Language is detected on the first chunk and pinned for the rest.
- faster-whisper yields segments lazily, so memory stays flat.

### 6. Failures and retries
- **Permanent errors** (corrupt file, no audio stream, too long) fail
  immediately with a clear message - retrying them only wastes compute.
- **Transient errors** (worker crash, GPU out-of-memory, timeouts) are retried
  up to 3 times with exponential backoff.
- Job state is written atomically to disk; on startup, jobs left `queued` or
  `processing` by a crashed process are re-queued.
- Temporary audio is deleted after a successful job; the transcript is kept.

---

## Scaling to production

This repo deliberately runs with **zero infrastructure** (thread pool + JSON
job files) so it is easy to review and run. The interfaces map directly onto a
production setup:

| Concern | This repo | Production |
|---|---|---|
| Concurrent uploads | Stateless FastAPI, background thread | API pods behind a load balancer; pre-signed S3 uploads so large files bypass the API; per-user rate limits and idempotency keys (+ file SHA-256 dedupe) |
| Job queue / workers | `ThreadPoolExecutor` | Redis + Celery or SQS; GPU workers autoscaled on queue depth (KEDA); fixed concurrency per GPU |
| Long files | Sequential chunks | Chunks fanned out in parallel, or GPU batching (`BatchedInferencePipeline`); per-chunk checkpoints so a retry re-runs only the failed chunk |
| Audio storage | Local `data/` folder | S3/GCS, encrypted, lifecycle rules to delete raw audio after N days |
| Transcript storage | `job.json` | Full JSON in object storage + PostgreSQL `jobs` / `segments` tables; Postgres full-text search or pgvector for search/RAG |
| Retries | In-process loop | Queue visibility timeout / `acks_late`, sweeper for stuck jobs, dead-letter queue + alerting, fallback to a smaller model or hosted API |
| Notifications | Polling | Signed (HMAC) webhooks on completion/failure; WebSocket for real-time streaming |
| Security | - | API keys / OAuth2, PII redaction, per-tenant retention, `DELETE` endpoint for data removal |
| Observability | Error + traceback stored on the job | Structured logs per job id, metrics (latency, failure rate, real-time factor), tracing |

---

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/transcriptions` | Multipart upload (`file`, optional `language`, `word_timestamps`). Returns `200` with the result for short audio, or `202` + `job_id` for long audio. |
| `GET` | `/v1/transcriptions/{job_id}` | Status, progress and result. `?format=json|txt|srt|vtt` |
| `GET` | `/health` | Liveness check |

Errors: `400` unsupported/corrupt file, `413` too large/long, `422` transcription failed, `404` unknown job.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | `cpu`, `cuda`, `auto` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8`, `float16`, ... |
| `CHUNK_SECONDS` | `600` | Target chunk length for long audio |
| `SYNC_MAX_SECONDS` | `120` | Longer audio is processed in the background |
| `MAX_UPLOAD_MB` / `MAX_DURATION_SEC` | `500` / `10800` | Input limits |
| `MAX_ATTEMPTS` | `3` | Retries for transient failures |

## Trade-offs and next steps
- Speaker diarisation (who spoke when) via pyannote, merged onto segments by timestamp.
- Streaming / real-time transcription over WebSocket.
- Evaluation: word error rate (WER) on a small labelled set to choose model size per use case.
- Downstream hooks: PII redaction, LLM summary / action items, embeddings for search.
