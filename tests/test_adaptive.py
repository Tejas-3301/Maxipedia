import shutil
import time
from pathlib import Path

import pytest

from backend.adaptive import deep
from backend.adaptive.scheduler import CONTROL_COUNT, schedule, union_length
from backend.checks.registry import JobContext
from backend.models import Finding

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "testset"


def _scr(curve, bin_s=0.25):
    return {"bin_s": bin_s, "flag_level": 0.35, "combined": curve, "curves": {"video": curve}}


def _curve():
    curve = [0.0] * 80  # 20 s
    curve[20] = curve[21] = 0.9  # 5.0-5.5 s
    curve[60] = 0.5  # 15 s
    return curve


def test_order_manual_then_top_flagged_then_controls_then_rest():
    s = schedule(_scr(_curve()), 20.0, "abc", manual=[{"start": 10.0, "end": 11.0}])
    kinds = [c["kind"] for c in s["candidates"]]
    assert kinds[0] == "manual" and kinds[1] == "flagged"
    assert kinds[2:2 + CONTROL_COUNT] == ["control"] * CONTROL_COUNT
    assert kinds[2 + CONTROL_COUNT] == "flagged"
    flagged = [c for c in s["candidates"] if c["kind"] == "flagged"]
    assert flagged[0]["score"] >= flagged[1]["score"]
    taken = [c for c in s["candidates"] if c["kind"] != "control"]
    controls = [c for c in s["candidates"] if c["kind"] == "control"]
    assert all(c["end"] <= t["start"] or c["start"] >= t["end"] for c in controls for t in taken)
    assert union_length(s["candidates"]) == pytest.approx(20.0, abs=0.01)  # candidates tile the clip


def test_schedule_is_deterministic_per_file():
    assert schedule(_scr(_curve()), 20.0, "f1") == schedule(_scr(_curve()), 20.0, "f1")
    assert schedule(_scr(_curve()), 20.0, "f1") != schedule(_scr(_curve()), 20.0, "f2")


def _fake_units(monkeypatch, seconds):
    def analyse(ctx, iv):
        time.sleep(seconds)
        return {"iv": iv}

    unit = deep.DeepUnit("fake", ("fake.check",), "visual", analyse, lambda ctx, res: [])
    monkeypatch.setattr(deep, "UNITS", (unit,))


def test_hard_deadline_stops_and_reports_unanalysed(monkeypatch, tmp_path):
    _fake_units(monkeypatch, 0.15)
    sched = schedule(_scr(_curve()), 20.0, "abc")
    ctx = JobContext("j", tmp_path / "x.mp4", "video", tmp_path, duration_s=20.0)
    _, rep = deep.run_deep(ctx, sched, budget_s=0.4)
    assert 1 <= len(rep["analysed"]) < len(sched["candidates"])
    assert len(rep["analysed"]) + len(rep["unanalysed"]) == len(sched["candidates"])
    assert rep["elapsed_s"] < 0.4 + 0.15 + 0.1  # overshoot bounded by one interval
    assert all(a["started_s"] < 0.4 for a in rep["analysed"])
    assert rep["analysed"][0]["kind"] == "flagged"
    assert 0 < rep["coverage"] < 1


def test_unlimited_budget_covers_everything(monkeypatch, tmp_path):
    _fake_units(monkeypatch, 0.0)
    sched = schedule(_scr(_curve()), 20.0, "abc")
    ctx = JobContext("j", tmp_path / "x.mp4", "video", tmp_path, duration_s=20.0)
    _, rep = deep.run_deep(ctx, sched, budget_s=float("inf"))
    assert rep["unanalysed"] == [] and rep["coverage"] == pytest.approx(1.0, abs=0.01)
    assert rep["miss_rate_est"] == pytest.approx(1 / (rep["controls_analysed"] + 2))


@pytest.mark.skipif(shutil.which("ffmpeg") is None or not (TESTSET / "03_splice.mp4").exists(),
                    reason="needs ffmpeg and the generated test set")
def test_splice_found_inside_deadline_run():
    import uuid

    from backend import db
    from backend.ingest import preprocess, store_original
    from backend.runner import build_context, plan

    db.init_db()
    jid = uuid.uuid4().hex
    with open(TESTSET / "03_splice.mp4", "rb") as f:
        path, sha, size = store_original(jid, "03_splice.mp4", f)
    db.create_job(jid, "03_splice.mp4", str(path), sha, size)
    arts = preprocess(jid, path)
    ctx = build_context(db.get_job(jid), arts)
    scr, sched = plan(ctx, arts)
    flagged = [i for i in sched["candidates"] if i["kind"] == "flagged"]
    assert any(i["start"] <= 3.8 <= i["end"] for i in flagged) and any(i["start"] <= 6.0 <= i["end"] for i in flagged)
    out, rep = deep.run_deep(ctx, sched)
    assert rep["elapsed_s"] <= rep["budget_s"] + max(a["took_s"] for a in rep["analysed"])
    hits = {f.check for f in out if f.stance == "supports_manipulation"}
    assert {"visual.noise_residual", "visual.temporal"} <= hits
    assert all(f.depth == "deep" for f in out if f.check in deep.DEEP_CHECKS and f.stance != "not_assessable")
    assert all(isinstance(Finding.model_validate(f.model_dump()), Finding) for f in out)
