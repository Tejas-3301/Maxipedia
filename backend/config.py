import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FORENSICS_DATA_DIR", ROOT / "data")).resolve()
UPLOADS_DIR = DATA_DIR / "uploads"
PROXIES_DIR = DATA_DIR / "proxies"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
DB_PATH = DATA_DIR / "forensics.db"

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_ANALYSIS_SECONDS = 60.0  # derived artifacts are capped; original is never truncated
FRAME_SAMPLE_FPS = 2.0
AUDIO_SAMPLE_RATE = 16000
PROXY_MAX_HEIGHT = 360
JOB_WORKERS = 2
CHECK_WORKERS = 4
TOOL_TIMEOUT_S = 300

# Adaptive analysis
DEEP_BUDGET_S = 20.0  # hard deadline for deep units (visual region/temporal, lip sync) per job
DEEP_FPS = 4.0  # full-res frame rate for deep region checks inside intervals
