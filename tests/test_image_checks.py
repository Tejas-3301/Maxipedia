import io
import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from backend.checks import _jpeg

ROOT = Path(__file__).resolve().parent.parent
ITS = ROOT / "data" / "image_testset"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_set = pytest.mark.skipif(shutil.which("exiftool") is None or not (ITS / "spliced.jpg").exists(),
                               reason="needs exiftool and scripts/make_image_testset.py")


def _photo(h=480, w=640, seed=0, flat_sky=False) -> np.ndarray:
    """Photo-like test image: multi-scale texture + sensor noise; optionally a smooth upper half (sky)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for s in (2, 6, 18):
        img += cv2.GaussianBlur(rng.normal(0, 40, (h, w)).astype(np.float32), (0, 0), s)
    img = 128 + img
    if flat_sky:
        img[: h // 2] = 170 + np.linspace(-10, 10, w)[None, :]
    img += rng.normal(0, 2.5, (h, w))
    g = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _jpeg_file(tmp: Path, rgb: np.ndarray, name: str, **kw) -> Path:
    p = tmp / name
    Image.fromarray(rgb).save(p, "JPEG", **kw)
    return p


def test_parse_quality_and_standard_tables(tmp_path):
    rgb = _photo()
    p = _jpeg_file(tmp_path, rgb, "q75.jpg", quality=75)
    info = _jpeg.parse(p)
    assert info is not None and info.sof == "baseline"
    assert np.array_equal(info.tables[0], _jpeg.ijg_table(75))
    assert _jpeg.ijg_quality(info.tables[0]) == (75, True)
    custom = (_jpeg.ijg_table(90) + np.tile([0, 1], 32)).clip(1, 255).tolist()
    pc = _jpeg_file(tmp_path, rgb, "custom.jpg", qtables=[custom, custom])
    assert _jpeg.ijg_quality(_jpeg.parse(pc).tables[0])[1] is False
    cv2.imwrite(str(tmp_path / "x.png"), rgb)
    assert _jpeg.parse(tmp_path / "x.png") is None


def test_double_jpeg_gap_separates_single_from_double(tmp_path):
    rgb = _photo(720, 960, seed=3)
    single = _jpeg_file(tmp_path, rgb, "s.jpg", quality=85)
    coarse = np.array(Image.open(_jpeg_file(tmp_path, rgb, "c.jpg", quality=50)).convert("RGB"))
    double = _jpeg_file(tmp_path, coarse, "d.jpg", quality=85)
    s, _ = _jpeg.double_jpeg_gap(cv2.imread(str(single)), _jpeg.parse(single).tables[0])
    d, _ = _jpeg.double_jpeg_gap(cv2.imread(str(double)), _jpeg.parse(double).tables[0])
    assert s is not None and d is not None and s < 0.2 < 0.4 < d


def _ctx_for(path: Path):
    from backend.ingest import preprocess
    from backend.runner import build_context

    jid = uuid.uuid4().hex
    arts = preprocess(jid, path)
    return build_context({"id": jid, "media_path": str(path), "sha256": "x"}, arts)


@needs_ffmpeg
def test_flat_sky_is_not_flagged_and_png_ela_not_assessable(tmp_path):
    from backend.checks.visual import check_ela, check_noise

    p = _jpeg_file(tmp_path, _photo(900, 1200, seed=5, flat_sky=True), "sky.jpg", quality=90)
    ctx = _ctx_for(p)
    for f in check_noise(ctx) + check_ela(ctx):
        assert f.stance != "supports_manipulation", f.rationale
        assert f.depth == "screen" and (f.coverage or 0) < 1.0  # flat sky skipped from coverage
    clean = [f for f in check_noise(ctx) if f.stance == "contradicts"]
    assert clean and clean[0].reliability <= 0.3
    png = tmp_path / "shot.png"
    cv2.imwrite(str(png), _photo(seed=6))
    assert check_ela(_ctx_for(png))[0].stance == "not_assessable"


@needs_ffmpeg
def test_still_image_thumbnails_for_lineage(tmp_path):
    from backend.provenance.fingerprint import fingerprint

    fp = fingerprint(_jpeg_file(tmp_path, _photo(), "t.jpg", quality=90))
    assert fp.thumbs is not None and fp.thumbs.shape[0] == 1 and len(fp.phashes) == 1


@needs_ffmpeg
@needs_set
def test_image_testset_exif_and_quant_outcomes():
    from backend.checks.image import image_check

    def by(name):
        return {f.check: f for f in image_check(_ctx_for(ITS / name))}

    clean = by("clean.jpg")
    assert clean["image.exif_consistency"].stance == "contradicts"
    assert clean["image.jpeg_quant"].stance == "contradicts" and clean["image.double_jpeg"].stance == "contradicts"
    edited = by("exif_edited.jpg")["image.exif_consistency"]
    assert edited.stance == "supports_manipulation" and "Photoshop" in edited.rationale and "ModifyDate" in edited.rationale
    spliced = by("spliced.jpg")["image.exif_consistency"]
    assert spliced.stance == "supports_manipulation" and spliced.region is not None
    x, y, w, h = spliced.region
    assert x < 1343 and x + w > 1043 and y < 511 and y + h > 211  # overlaps the pasted region
    resaved = by("resaved_3gen.jpg")
    assert resaved["image.double_jpeg"].stance == "supports_manipulation"
    assert resaved["image.exif_consistency"].stance == "not_assessable"
    shot = by("screenshot.png")
    assert all(shot[k].stance == "not_assessable" for k in ("image.jpeg_quant", "image.double_jpeg", "image.exif_consistency"))
