import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from scripts.make_sample_video import make_sample

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def done_job(tmp_path_factory):
    clip = make_sample(tmp_path_factory.mktemp("ws") / "s.mp4", seconds=3)
    with TestClient(app) as c:
        with open(clip, "rb") as f:
            jid = c.post("/jobs", files={"file": ("s.mp4", f, "video/mp4")}).json()["job_id"]
        statuses = []
        deadline = time.time() + 120
        while time.time() < deadline:
            s = c.get(f"/jobs/{jid}").json()["status"]
            if not statuses or statuses[-1] != s:
                statuses.append(s)
            if s in ("done", "failed"):
                break
            time.sleep(0.1)
        yield c, jid, statuses


def test_progressive_status_and_screening(done_job):
    c, jid, statuses = done_job
    assert statuses[-1] == "done"
    assert all(s in ("queued", "preprocessing", "screening", "analyzing", "deep", "done") for s in statuses)
    sc = c.get(f"/jobs/{jid}/screening").json()
    assert sc["screening"]["curves"] and sc["schedule"]["analysed"]


def test_media_range_and_info(done_job):
    c, jid, _ = done_job
    r = c.get(f"/jobs/{jid}/media", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100
    info = c.get(f"/jobs/{jid}/info").json()
    assert info["width"] == 640 and info["height"] == 480 and info["has_proxy"]


def test_artifact_confined_to_job(done_job, tmp_path):
    c, jid, _ = done_job
    paths = [v for f in c.get(f"/jobs/{jid}/findings").json() for k, v in f["evidence"].items()
             if k in ("heatmap", "spectrogram", "overview", "plot") and isinstance(v, str)]
    assert paths and c.get(f"/jobs/{jid}/artifact", params={"path": paths[0]}).status_code == 200
    outside = tmp_path / "secret.txt"
    outside.write_text("x")
    assert c.get(f"/jobs/{jid}/artifact", params={"path": str(outside)}).status_code == 404
    assert c.get(f"/jobs/{jid}/artifact", params={"path": paths[0] + "/../../../../forensics.db"}).status_code == 404


def test_reviews_round_trip_through_custody(done_job):
    c, jid, _ = done_job
    fid = c.get(f"/jobs/{jid}/findings").json()[0]["id"]
    assert c.post(f"/jobs/{jid}/findings/{fid}/review", json={"decision": "reject", "actor": "alice"}).status_code == 200
    c.post(f"/jobs/{jid}/findings/{fid}/review", json={"decision": "note", "note": "compression artefact", "actor": "bob"})
    r = c.get(f"/jobs/{jid}/reviews").json()["reviews"][fid]
    assert r["decision"] == "reject" and r["notes"][0]["note"] == "compression artefact"
    assert [h["actor"] for h in r["history"]] == ["alice", "bob"]
    c.post(f"/jobs/{jid}/findings/{fid}/review", json={"decision": "clear", "actor": "alice"})
    assert c.get(f"/jobs/{jid}/reviews").json()["reviews"][fid]["decision"] is None
    cus = c.get(f"/jobs/{jid}/custody").json()
    assert cus["verification"]["intact"] and sum(e["action"] == "review" for e in cus["entries"]) == 3
    assert c.post(f"/jobs/{jid}/findings/{fid}/review", json={"decision": "maybe"}).status_code == 400
    assert c.post(f"/jobs/{jid}/findings/nope/review", json={"decision": "accept"}).status_code == 404
