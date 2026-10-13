import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import db, runner
from backend.checks.capture import capture_check
from backend.checks.registry import JobContext
from backend.fusion import fuse
from backend.fusion.fuse import STATE_AUTH, STATE_INC
from backend.models import Finding
from backend.provenance import device
from scripts.make_sample_video import make_sample

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

CLEAN = [Finding(check=c, modality=m, stance="contradicts", score=0.5, reliability=0.5)
         for c, m in (("m.a", "metadata"), ("m.b", "metadata"), ("v.a", "visual"), ("a.a", "audio"), ("s.a", "sync"))]


def quality(degraded: bool) -> Finding:
    return Finding(check="quality.media", modality="metadata", stance="neutral", score=0.0, reliability=0.5,
                   evidence={"degraded": degraded, "reasons": ["heavy compression (x264 crf=35 >= 30)"] if degraded else []})


def test_quality_gate_blocks_authentic_on_degraded_media():
    assert fuse(CLEAN + [quality(False)], media_type="video")["state"] == STATE_AUTH
    out = fuse(CLEAN + [quality(True)], media_type="video")
    assert out["state"] == STATE_INC and any("degraded" in r for r in out["reasons"])
    # neutral quality finding moves no probability mass
    assert out["probability_manipulated"] == fuse(CLEAN, media_type="video")["probability_manipulated"]
    strong = Finding(check="provenance.capture_signature", modality="provenance", stance="contradicts", score=0.9, reliability=0.9)
    assert fuse(CLEAN + [quality(True), strong], media_type="video")["state"] == STATE_AUTH


def _ctx(media: Path, tmp: Path, manifest: dict | None) -> JobContext:
    folder = tmp / uuid.uuid4().hex
    folder.mkdir()
    dst = folder / "original.mp4"
    shutil.copy(media, dst)
    if manifest is not None:
        (folder / "capture_manifest.json").write_text(json.dumps(manifest))
    return JobContext("cap", dst, "video", folder)


@needs_ffmpeg
def test_capture_signature_chain_states(tmp_path):
    db.init_db()
    clip = make_sample(tmp_path / "cam.mp4", seconds=3)
    priv, pub = device.new_device_key()
    dev = f"cam-{uuid.uuid4().hex[:6]}"
    device.register(dev, pub, "test cam")
    m = device.sign_capture(clip, dev, priv)

    assert capture_check(_ctx(clip, tmp_path, None))[0].stance == "neutral"
    intact = capture_check(_ctx(clip, tmp_path, m))[0]
    assert intact.stance == "contradicts" and intact.evidence["chain"] == "intact"

    remux = tmp_path / "remux.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-c", "copy", "-metadata", "title=x", str(remux)], check=True)
    assert capture_check(_ctx(remux, tmp_path, m))[0].evidence["chain"] == "content_intact"

    swapped = tmp_path / "swapped.mp4"  # same video packets, new audio
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-f", "lavfi", "-i", "sine=frequency=880:duration=3",
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-shortest", str(swapped)], check=True)
    broken = capture_check(_ctx(swapped, tmp_path, m))[0]
    assert broken.stance == "supports_manipulation" and broken.evidence["chain"] == "broken"
    assert all(k == "audio" for k, _, _ in broken.evidence["changed_segments"])

    forged = dict(m, sha256="0" * 64)
    assert capture_check(_ctx(clip, tmp_path, forged))[0].stance == "supports_manipulation"
    unknown = dict(m, device_id="nobody")
    assert capture_check(_ctx(clip, tmp_path, unknown))[0].stance == "not_assessable"


@needs_ffmpeg
def test_interrupted_job_is_resumed_without_duplicates(tmp_path):
    from backend.ingest import store_original

    db.init_db()
    clip = make_sample(tmp_path / "r.mp4", seconds=2)
    jid = uuid.uuid4().hex
    with open(clip, "rb") as f:
        p, sha, size = store_original(jid, "r.mp4", f)
    db.create_job(jid, "r.mp4", str(p), sha, size)
    db.save_findings(jid, [Finding(check="stale", modality="visual", stance="neutral")])  # half-finished run
    db.update_job(jid, status="analyzing")
    assert jid in runner.recover()
    deadline = time.time() + 120
    while db.get_job(jid)["status"] != "done" and time.time() < deadline:
        time.sleep(0.2)
    checks = [f.check for f in db.get_findings(jid)]
    assert db.get_job(jid)["status"] == "done" and "stale" not in checks
    assert len(checks) == len(set((f.check, f.segment.start if f.segment else None, f.stance)
                                  for f in db.get_findings(jid)))


def test_single_process_server_serves_api_and_ui():
    from backend.serve import app

    with TestClient(app) as c:
        assert c.get("/api/health").json()["db"] is True
        r = c.get("/")
        assert r.status_code == 200 and "html" in r.headers["content-type"]
