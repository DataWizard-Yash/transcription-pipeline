"""Job tracking, background execution and retries.

For this assignment jobs run in an in-process thread pool and state lives in
JSON files, so the project runs with zero infrastructure. In production the
same interface maps onto PostgreSQL (job table) + Redis/SQS + Celery workers
(see README, "Scaling to production").
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config, pipeline
from .audio import AudioError
from .transcriber import get_transcriber

QUEUED, PROCESSING, COMPLETED, FAILED = "queued", "processing", "completed", "failed"

_executor = ThreadPoolExecutor(max_workers=1)   # one model instance -> one job at a time per process
_lock = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return Path(config.DATA_DIR) / "jobs" / job_id


def _save(job: dict) -> None:
    job["updated_at"] = time.time()
    path = _job_dir(job["id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(json.dumps(job, indent=2))
        tmp.replace(path)                         # atomic write


def get(job_id: str) -> dict | None:
    path = _job_dir(job_id) / "job.json"
    return json.loads(path.read_text()) if path.exists() else None


def create(src_file: Path, filename: str, options: dict) -> dict:
    job_id = str(uuid.uuid4())
    d = _job_dir(job_id)
    d.mkdir(parents=True)
    stored = d / ("input" + Path(filename).suffix.lower())
    shutil.move(str(src_file), stored)
    job = {"id": job_id, "status": QUEUED, "filename": filename, "input": str(stored),
           "options": options, "attempts": 0, "progress": 0, "error": None,
           "result": None, "created_at": time.time()}
    _save(job)
    return job


def run(job_id: str) -> dict:
    """Execute a job with retries. Permanent errors (bad audio) fail fast;
    transient errors are retried with exponential backoff."""
    job = get(job_id)
    engine = get_transcriber()
    while job["attempts"] < config.MAX_ATTEMPTS:
        job["attempts"] += 1
        job["status"], job["error"] = PROCESSING, None
        _save(job)

        def progress(pct, job=job):
            job["progress"] = pct
            _save(job)

        try:
            result = pipeline.run(job["input"], _job_dir(job_id) / "work", engine,
                                  language=job["options"].get("language"),
                                  word_timestamps=job["options"].get("word_timestamps", False),
                                  progress=progress)
            result["model"] = getattr(engine, "model_name", type(engine).__name__)
            job.update(status=COMPLETED, result=result, progress=100)
            _save(job)
            shutil.rmtree(_job_dir(job_id) / "work", ignore_errors=True)   # keep transcript, drop temp audio
            return job
        except AudioError as e:                       # permanent: retrying will not help
            job.update(status=FAILED, error=str(e))
            _save(job)
            return job
        except Exception as e:                        # transient: retry with backoff
            job["error"] = f"{type(e).__name__}: {e}"
            job["last_traceback"] = traceback.format_exc(limit=3)
            _save(job)
            if job["attempts"] < config.MAX_ATTEMPTS:
                time.sleep(min(2 ** job["attempts"], 30))
    job["status"] = FAILED                             # would go to a dead-letter queue in production
    _save(job)
    return job


def submit(job_id: str) -> None:
    _executor.submit(run, job_id)


def recover_stuck_jobs() -> int:
    """On startup, re-queue jobs left 'queued'/'processing' by a crashed process."""
    root = Path(config.DATA_DIR) / "jobs"
    n = 0
    for p in root.glob("*/job.json") if root.exists() else []:
        job = json.loads(p.read_text())
        if job["status"] in (QUEUED, PROCESSING):
            submit(job["id"])
            n += 1
    return n
