import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.checks.registry import JobContext
from backend.checks.visual import _region_events, _zscores, noise_map, visual_check


def _noisy_frames(n=12, seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8), (0, 0), 4)
    return [np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8) for _ in range(n)]


def test_region_events_find_denoised_patch():
    frames = _noisy_frames()
    for t in (5, 6, 7):
        frames[t][32:128, 160:288] = cv2.GaussianBlur(frames[t][32:128, 160:288], (0, 0), 3)
    z, dev = _zscores(np.stack([noise_map(f) for f in frames]), temporal=True)
    events = _region_events(z, dev, min_frames=2)
    assert len(events) == 1
    ev = events[0]
    assert ev.frames == [5, 6, 7] and ev.mean_log_ratio < 0
    x, y, w, h = ev.box
    assert abs(x - 160) <= 16 and abs(y - 32) <= 16 and abs(w - 128) <= 32 and abs(h - 96) <= 32


def test_region_events_quiet_on_clean_frames():
    z, dev = _zscores(np.stack([noise_map(f) for f in _noisy_frames(seed=1)]), temporal=True)
    assert _region_events(z, dev, min_frames=2) == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_visual_check_runs_end_to_end(tmp_path: Path):
    from backend.ingest import preprocess
    from backend.runner import build_context
    from scripts.make_sample_video import make_sample

    clip = make_sample(tmp_path / "s.mp4", seconds=3)
    arts = preprocess("vtest", clip)
    ctx = build_context({"id": "vtest", "media_path": str(clip), "sha256": "x"}, arts)
    out = visual_check(ctx)  # whole-file part only; frame-level units run via the scheduler
    assert {f.check for f in out} == {"visual.pts_gaps", "visual.gop"}
    assert all(f.stance != "not_assessable" for f in out)

    from backend.checks.visual import region_analyse, region_finalize, temporal_analyse, temporal_finalize
    iv = {"start": 0.0, "end": 3.0, "kind": "flagged", "order": 0}
    deep = region_finalize(ctx, [(iv, region_analyse(ctx, iv))]) + temporal_finalize(ctx, [(iv, temporal_analyse(ctx, iv))])
    assert {f.check for f in deep} == {"visual.noise_residual", "visual.ela", "visual.temporal"}
    assert all(f.depth == "deep" and f.coverage is not None for f in deep)
