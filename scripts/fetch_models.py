"""Download open model files used locally by checks (run once). Apache-2.0 MediaPipe Face Landmarker."""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def main() -> None:
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        dest = out / name
        if dest.exists():
            print(f"{name}: present")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"{name}: {dest.stat().st_size} bytes sha256 {hashlib.sha256(dest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
