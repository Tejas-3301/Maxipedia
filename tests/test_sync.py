import shutil
from pathlib import Path

import numpy as np
import pytest

from backend.checks.sync import MIN_FACE_FRAC, MODEL, SYNC_FPS, WIN_S, analyse_span, sync_analyse, sync_finalize
from backend.models import Finding

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "testset"


def _speechy(n, seed=0):
    rng = np.random.default_rng(seed)
    env = np.convolve(rng.random(n) ** 3, np.ones(4) / 4, mode="same")
    return env, np.ones(n, bool)


def test_aligned_signals_are_ok_at_zero_lag():
    env, sp = _speechy(100)
    ok = [x for x in analyse_span(env.copy(), np.log(env + 1e-3), sp, 0.0) if x["status"] == "ok"]
    assert ok and all(x["best_lag_s"] == 0 and x["r_max"] > 0.8 for x in ok)


def test_audio_delay_is_measured():
    env, sp = _speechy(150, seed=1)
    audio = np.r_[np.full(3, env.min()), env[:-3]]  # audio 3 frames late
    ok = [x for x in analyse_span(env, audio, sp, 0.0) if x["status"] == "ok"]
    assert np.median([x["best_lag_s"] for x in ok]) == pytest.approx(3 / SYNC_FPS)


def test_no_face_and_no_speech_windows_not_assessable():
    env, sp = _speechy(100)
    mouth = env.copy()
    mouth[:60] = np.nan
    sp = sp.copy()
    sp[60:] = False
    assert {x["status"] for x in analyse_span(mouth, env, sp, 0.0)} >= {"no_face", "no_speech"}


def test_not_assessable_windows_are_neither_clean_nor_covered(tmp_path):
    """Finalize on synthetic windows: only usable windows produce clean evidence and coverage."""
    from backend.checks.registry import JobContext

    env, sp = _speechy(250, seed=3)  # 10 s
    mouth = env.copy()
    mouth[125:] = np.nan  # second half: no face
    windows = analyse_span(mouth, np.log(env + 1e-3), sp, 0.0)
    ctx = JobContext("s", tmp_path / "x.mp4", "video", tmp_path, duration_s=10.0)
    out = sync_finalize(ctx, [({"start": 0.0, "end": 10.0, "kind": "flagged"}, {"windows": windows})])
    for f in out:
        Finding.model_validate(f.model_dump())
        assert f.depth == "deep" and f.modality == "sync" and f.check == "sync.lipsync"
    clean = [f for f in out if f.stance == "contradicts"]
    na = [f for f in out if f.stance == "not_assessable"]
    assert clean and na
    assert all(f.coverage is None for f in na)
    # clean only where the face was visible (a window may be up to 30% face-less and still usable)
    slack = WIN_S * (1 - MIN_FACE_FRAC) + 0.05
    assert all(f.segment.end <= 5.0 + slack for f in clean)
    assert sum(f.coverage for f in clean) <= (5.0 + slack) / 10.0


needs = pytest.mark.skipif(shutil.which("ffmpeg") is None or not MODEL.exists(), reason="needs ffmpeg + model")


@needs
def test_clip_without_face_is_not_assessable(tmp_path):
    from backend.ingest import preprocess
    from backend.runner import build_context
    from scripts.make_sample_video import make_sample

    clip = make_sample(tmp_path / "s.mp4", seconds=4)
    arts = preprocess("synctest", clip)
    ctx = build_context({"id": "synctest", "media_path": str(clip), "sha256": "x"}, arts)
    iv = {"start": 0.0, "end": 4.0, "kind": "flagged"}
    out = sync_finalize(ctx, [(iv, sync_analyse(ctx, iv))])
    assert out and all(f.stance == "not_assessable" and f.depth == "deep" for f in out)


@needs
@pytest.mark.skipif(not (TESTSET / "02_voice_swap.mp4").exists(), reason="run scripts/make_testset.py first")
def test_voice_swap_flagged_clean_consistent_schema_valid():
    from backend.ingest import preprocess
    from backend.runner import build_context

    res = {}
    for name in ("01_clean", "02_voice_swap"):
        arts = preprocess(f"sync_{name}", TESTSET / f"{name}.mp4")
        ctx = build_context({"id": name, "media_path": str(TESTSET / f"{name}.mp4"), "sha256": "x"}, arts)
        ivs = [{"start": 0.0, "end": 4.0, "kind": "flagged"}, {"start": 4.0, "end": 8.0, "kind": "control"}]
        res[name] = sync_finalize(ctx, [(iv, sync_analyse(ctx, iv)) for iv in ivs])
    for out in res.values():  # step-5 sanity: valid schema, depth=deep
        for f in out:
            assert Finding.model_validate(f.model_dump()).depth == "deep"
    assert any(f.stance == "contradicts" for f in res["01_clean"])
    hits = [f for f in res["02_voice_swap"] if f.stance == "supports_manipulation"]
    assert hits and hits[0].segment.start <= 3.0 and hits[0].segment.end >= 5.0
    assert Path(hits[0].evidence["plot"]).exists() and Path(hits[0].evidence["curve_path"]).exists()
