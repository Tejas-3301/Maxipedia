import hashlib
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from backend.attack import ATTACKS, compare
from backend.main import app
from backend.models import Finding
from scripts.make_sample_video import make_sample

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _wait(c, jid, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get(f"/jobs/{jid}").json()["status"]
        if s in ("done", "failed"):
            return s
        time.sleep(0.2)
    raise TimeoutError(jid)


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    clip = make_sample(tmp_path_factory.mktemp("rep") / "s.mp4", seconds=3)
    with TestClient(app) as c:
        with open(clip, "rb") as f:
            jid = c.post("/jobs", files={"file": ("s.mp4", f, "video/mp4")}).json()["job_id"]
        assert _wait(c, jid) == "done"
        yield c, jid


def test_pdf_report_hash_logged_in_custody(job):
    c, jid = job
    r = c.get(f"/jobs/{jid}/report.pdf", params={"actor": "alice"})
    assert r.status_code == 200 and r.content[:5] == b"%PDF-"
    digest = hashlib.sha256(r.content).hexdigest()
    assert r.headers["x-report-sha256"] == digest
    cus = c.get(f"/jobs/{jid}/custody").json()
    last = cus["entries"][-1]
    assert last["action"] == "report_exported" and last["actor"] == "alice" and digest in last["detail"]
    assert cus["verification"]["intact"]


def test_attack_creates_linked_job_and_comparison(job):
    c, jid = job
    r = c.post(f"/jobs/{jid}/attack", json={"kind": "resize", "level": "light", "actor": "bob"})
    assert r.status_code == 202
    child = r.json()["job_id"]
    assert _wait(c, child) == "done"
    meta = c.get(f"/jobs/{child}").json()
    assert meta["derived_from"] == jid and meta["attack"]["kind"] == "resize"
    a = c.get(f"/jobs/{jid}/attacks").json()
    row = next(x for x in a["attacks"] if x["job_id"] == child)
    assert row["state"] in ("likely manipulated", "likely authentic", "inconclusive") and row["checks"]
    assert any(e["action"] == "attack_mode" for e in c.get(f"/jobs/{jid}/custody").json()["entries"])
    assert c.post(f"/jobs/{jid}/attack", json={"kind": "blur"}).status_code == 400


def test_compare_marks_lost_and_kept():
    sup = lambda check, s: Finding(check=check, modality="visual", stance="supports_manipulation", score=s, reliability=0.5)
    rows = {r["check"]: r["status"] for r in compare([sup("a", 0.8), sup("b", 0.6)], [sup("b", 0.6), sup("c", 0.5)])}
    assert rows == {"a": "lost", "b": "kept", "c": "new"}
    assert set(ATTACKS) == {"recompress", "resize", "noise"}
