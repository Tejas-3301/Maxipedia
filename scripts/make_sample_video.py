"""Generate a short synthetic sample clip (test pattern + tone) for pipeline smoke tests."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def make_sample(out: Path, seconds: float = 4.0) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size=640x480:rate=25:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples/sample.mp4")
    print(make_sample(target))
