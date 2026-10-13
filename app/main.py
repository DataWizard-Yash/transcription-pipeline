"""HTTP API.

POST /v1/transcriptions         upload audio -> 200 with result (short files) or 202 + job id (long files)
GET  /v1/transcriptions/{id}    status / progress / result; ?format=json|txt|srt|vtt
GET  /health                    liveness
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import audio, config, jobs, pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    jobs.recover_stuck_jobs()
    yield


app = FastAPI(title="Transcription Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    language: str | None = Form(None, description="ISO code, e.g. 'en'. Empty = auto-detect"),
    word_timestamps: bool = Form(False),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")

    # Stream to disk in 1 MB chunks - never hold the whole upload in memory.
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    tmp = Path(tempfile.mkstemp(suffix=ext)[1])
    size = 0
    with tmp.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {config.MAX_UPLOAD_MB} MB")
            out.write(chunk)

    try:
        duration = audio.probe(tmp)          # validate the content, not just the extension
    except audio.AudioError as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    if duration > config.MAX_DURATION_SEC:
        tmp.unlink(missing_ok=True)
        raise HTTPException(413, f"Audio longer than {config.MAX_DURATION_SEC} seconds")

    job = jobs.create(tmp, file.filename, {"language": language, "word_timestamps": word_timestamps})

    if duration <= config.SYNC_MAX_SECONDS:  # short audio: answer in the same request
        job = jobs.run(job["id"])
        if job["status"] == jobs.FAILED:
            raise HTTPException(422, job["error"])
        return {"job_id": job["id"], "status": job["status"], **job["result"]}

    jobs.submit(job["id"])                   # long audio: background job
    return JSONResponse(status_code=202, content={
        "job_id": job["id"], "status": job["status"], "duration_sec": round(duration, 2),
        "status_url": f"/v1/transcriptions/{job['id']}"})


@app.get("/v1/transcriptions/{job_id}")
def get_transcription(job_id: str, format: str = Query("json", pattern="^(json|txt|srt|vtt)$")):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != jobs.COMPLETED:
        return {"job_id": job_id, "status": job["status"], "progress": job["progress"],
                "attempts": job["attempts"], "error": job["error"]}
    result = job["result"]
    if format == "txt":
        return PlainTextResponse(result["text"])
    if format == "srt":
        return PlainTextResponse(pipeline.to_srt(result), media_type="application/x-subrip")
    if format == "vtt":
        return PlainTextResponse(pipeline.to_vtt(result), media_type="text/vtt")
    return {"job_id": job_id, "status": job["status"], **result}
