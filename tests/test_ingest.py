import os
import shutil
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.ingest import sha256_file
from backend.main import app
from scripts.make_sample_video import make_sample

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _wait_done(client: TestClient, job_id: str, timeout: float = 60) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise TimeoutError(job)


def test_video_ingest_end_to_end(tmp_path: Path):
    sample = make_sample(tmp_path / "sample.mp4", seconds=3)
    with TestClient(app) as client:
        with open(sample, "rb") as f:
            r = client.post("/jobs", files={"file": ("sample.mp4", f, "video/mp4")})
        assert r.status_code == 202
        body = r.json()
        assert body["sha256"] == sha256_file(sample)

        job = _wait_done(client, body["job_id"])
        assert job["status"] == "done", job.get("error")
        assert job["media_type"] == "video"
        arts = job["artifacts"]
        assert arts["errors"] == {}

        original = Path(job["media_path"])
        assert not os.access(original, os.W_OK)
        assert sha256_file(original) == body["sha256"]

        assert arts["frame_count"] == 6  # 3 s @ 2 fps
        assert all(p.suffix == ".png" for p in Path(arts["frames_dir"]).iterdir())
        with wave.open(arts["audio_path"]) as w:
            assert (w.getframerate(), w.getnchannels()) == (16000, 1)
        assert Path(arts["proxy_path"]).stat().st_size > 0
        assert Path(arts["metadata_path"]).read_text().count('"ffprobe"') == 1

        findings = client.get(f"/jobs/{body['job_id']}/findings").json()
        by_check = {f["check"]: f for f in findings}
        assert by_check["ingest_inventory"]["stance"] == "neutral"
        assert by_check["ingest_inventory"]["evidence"]["frame_count"] == 6
        assert "metadata.timestamps" in by_check


def test_unknown_job_404():
    with TestClient(app) as client:
        assert client.get("/jobs/nope").status_code == 404
        assert client.get("/jobs/nope/findings").status_code == 404


def test_preprocess_from_script_creates_proxy_dir(tmp_path, monkeypatch):
    """Regression: proxies/ used to exist only when the FastAPI lifespan had run, so script-driven ingest
    silently produced no proxy (and screening lost its frame-difference and audio signals)."""
    import backend.ingest as ingest

    fresh = tmp_path / "never_created"
    monkeypatch.setattr(ingest, "PROXIES_DIR", fresh / "proxies")
    monkeypatch.setattr(ingest, "ARTIFACTS_DIR", fresh / "artifacts")
    assert not fresh.exists()
    sample = make_sample(tmp_path / "s.mp4", seconds=2)
    arts = ingest.preprocess("proxytest", sample)
    assert "proxy" not in arts["errors"], arts["errors"]
    assert Path(arts["proxy_path"]).exists() and Path(arts["proxy_path"]).stat().st_size > 0
