"""Runtime configuration, read from environment variables so the same image
can run small models on a laptop and large models on a GPU box."""
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")          # tiny | base | small | medium | large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")         # auto | cpu | cuda
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

DATA_DIR = os.getenv("DATA_DIR", "./data")                   # uploads, normalised audio, transcripts
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_DURATION_SEC = int(os.getenv("MAX_DURATION_SEC", str(3 * 60 * 60)))

# Files longer than this are split into chunks and processed in the background.
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "600"))       # ~10 min per chunk
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "1.0"))
SYNC_MAX_SECONDS = int(os.getenv("SYNC_MAX_SECONDS", "120")) # <= 2 min -> answer inline

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4"}
