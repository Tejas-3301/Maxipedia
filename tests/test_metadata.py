import subprocess
from pathlib import Path

import pytest

from backend.checks import _mp4
from backend.checks.metadata import check_edit_lists, metadata_check
from backend.checks.registry import JobContext
from scripts.make_sample_video import make_sample
from scripts.make_testset import patch_track_times

pytestmark = pytest.mark.skipif(subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
                                reason="ffmpeg not installed")


def _ctx(path: Path, tmp: Path) -> JobContext:
    return JobContext("t", path, "video", tmp)


def _by_check(findings):
    return {f.check: f for f in findings}


def test_clean_sample_has_no_metadata_alarms(tmp_path: Path):
    clip = make_sample(tmp_path / "a.mp4", seconds=3)
    out = _by_check(metadata_check(_ctx(clip, tmp_path)))
    assert set(out) == {"metadata.timestamps", "metadata.durations", "metadata.edit_list",
                        "metadata.encoder", "metadata.reencode_generations", "provenance.c2pa"}
    assert all(f.stance != "supports_manipulation" for f in out.values())
    assert any("Lavf" in fld["value"] for fld in out["metadata.encoder"].evidence["fields"])


def test_forged_track_times_and_camera_claim(tmp_path: Path):
    src = make_sample(tmp_path / "a.mp4", seconds=3)
    dst = tmp_path / "b.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c", "copy",
                    "-metadata", "creation_time=2019-01-01T00:00:00Z",
                    "-metadata", "com.apple.quicktime.make=Apple", "-movflags", "use_metadata_tags", str(dst)],
                   check=True)
    patch_track_times(dst, {"vide": "2024-05-05T00:00:00Z"})
    movie = _mp4.parse(dst)
    assert movie and movie.tracks[0].handler == "vide"
    assert movie.tracks[0].tkhd_creation.year == 2024 and movie.mvhd_creation.year == 2019

    out = _by_check(metadata_check(_ctx(dst, tmp_path)))
    ts = out["metadata.timestamps"]
    assert ts.stance == "supports_manipulation"
    assert "moov/trak[0](vide)/tkhd.creation_time = '2024-05-05T00:00:00Z'" in ts.rationale
    assert "moov/mvhd.creation_time = '2019-01-01T00:00:00Z'" in ts.rationale
    assert out["metadata.encoder"].stance == "supports_manipulation"


def test_multi_segment_edit_list_flagged(tmp_path: Path):
    movie = _mp4.Movie(mvhd_timescale=1000, tracks=[_mp4.Track(
        index=0, handler="vide", mdhd_timescale=12800,
        edits=[_mp4.Edit(2000, 0, 1.0), _mp4.Edit(3000, 64000, 1.0)])])
    f = check_edit_lists(_ctx(tmp_path / "x.mp4", tmp_path), {}, movie)
    assert f.stance == "supports_manipulation" and "2 non-empty edit segments" in f.rationale
