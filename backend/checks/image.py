"""Image-native checks (stills only): JPEG quantisation tables, double-compression indicator, EXIF consistency.

All findings quote exact field names/values. A non-JPEG file, or a JPEG without EXIF, gives "not assessable"
for the checks that need them instead of guessing.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

from backend.checks import _jpeg
from backend.checks.registry import JobContext, register
from backend.models import Depth, Finding, Modality, Stance

DJPEG_STRONG, DJPEG_WEAK, DJPEG_SINGLE = 0.40, 0.20, 0.15
TIME_TOL_S = 60
THUMB_MATCH = 0.85
THUMB_LOCAL_DIFF = 0.6  # standardised units; a clean camera thumbnail stays near 0.1
THUMB_LOCAL_RATIO = 6.0
EDITORS = ["photoshop", "lightroom", "gimp", "snapseed", "picsart", "canva", "affinity", "pixelmator", "facetune",
           "luminar", "capture one", "paint.net", "photopea", "remini", "faceapp", "fotor", "polarr"]


def _f(check: str, modality: Modality, stance: Stance, score: Optional[float], rel: float, text: str, lim: str,
       **evidence: Any) -> Finding:
    return Finding(modality=modality, check=check, depth=Depth.screen, stance=stance, score=score, reliability=rel,
                   rationale=text, limitations=lim, coverage=1.0, evidence=evidence)


def _na(check: str, modality: Modality, why: str) -> Finding:
    return _f(check, modality, Stance.not_assessable, None, 0.0, why, why)


def _exif(ctx: JobContext) -> dict[str, Any]:
    if not ctx.metadata_path or not ctx.metadata_path.exists():
        return {}
    ex = json.loads(ctx.metadata_path.read_text()).get("exiftool") or {}
    return ex.get("tags") or {} if ex.get("available") else {}


def _tag(tags: dict, *names: str) -> tuple[Optional[str], Any]:
    for n in names:
        for k, v in tags.items():
            if k.split(":")[-1] == n and not k.startswith(("System:", "File:", "Composite:")):
                return k, v
    return None, None


# ---------------------------------------------------------------- quantisation tables

def check_quant(ctx: JobContext, info: Optional[_jpeg.JpegInfo], tags: dict) -> Finding:
    name = "image.jpeg_quant"
    lim = ("Some cameras and phones also use the standard libjpeg tables, and editors can write custom ones; the table "
           "says which encoder wrote the file last, not what happened to the content.")
    if info is None:
        return _na(name, Modality.metadata, "Not a JPEG: no quantisation tables to analyse.")
    if 0 not in info.tables:
        return _na(name, Modality.metadata, "JPEG without a luminance quantisation table (DQT 0).")
    luma = info.tables[0]
    q, exact = _jpeg.ijg_quality(luma)
    mk, make = _tag(tags, "Make")
    mo, model = _tag(tags, "Model")
    fields = [{"field": "DQT[0] (natural order, first row)", "value": str(list(map(int, luma[:8])))},
              {"field": "DQT tables", "value": str(sorted(info.tables))},
              {"field": "SOF", "value": info.sof},
              {"field": "estimated IJG quality", "value": str(q)},
              {"field": "standard IJG table", "value": "exact match" if exact else "no (custom table)"}]
    if mk:
        fields.append({"field": mk, "value": str(make)})
    if mo:
        fields.append({"field": mo, "value": str(model)})
    ev = {"fields": fields, "ijg_quality": q, "standard_ijg": exact, "sof": info.sof, "app_markers": info.app_markers}
    claim = f"{mk} = {make!r}" + (f", {mo} = {model!r}" if mo else "") if mk else None
    if claim and exact:
        return _f(name, Modality.metadata, Stance.supports_manipulation, 0.45, 0.4,
                  f"{claim} claims a camera original, but DQT[0] is exactly the standard libjpeg table at quality {q}: "
                  f"the file was last written by generic software (re-saved or exported).", lim, **ev)
    if claim and not exact:
        return _f(name, Modality.metadata, Stance.contradicts, 0.3, 0.3,
                  f"{claim} with a custom quantisation table (nearest IJG quality {q}, not an exact match): consistent "
                  f"with the camera's own encoder.", lim, **ev)
    kind = f"the standard libjpeg table at quality {q}" if exact else f"a custom table (nearest IJG quality {q})"
    return _f(name, Modality.metadata, Stance.neutral, 0.0, 0.3,
              f"DQT[0] is {kind}; no camera make/model to compare it with.", lim, **ev)


# ---------------------------------------------------------------- double compression

def check_double(ctx: JobContext, info: Optional[_jpeg.JpegInfo], bgr: Optional[np.ndarray]) -> Finding:
    name = "image.double_jpeg"
    lim = ("Detects an earlier compression that was coarser than the last one (aligned 8x8 grid); a re-save at equal or "
           "finer quality, cropping or resizing between saves hides it. Re-saving alone is not manipulation.")
    if info is None:
        return _na(name, Modality.visual, "Not a JPEG: no compression history to analyse.")
    if bgr is None or 0 not in info.tables:
        return _na(name, Modality.visual, "Pixels or luminance quantisation table unavailable.")
    score, per = _jpeg.double_jpeg_gap(bgr, info.tables[0])
    ev = {"gap_score": score, "per_position": per,
          "fields": [{"field": "DCT histogram gap score (median of low-frequency positions)", "value": str(score)}]}
    if score is None:
        return _na(name, Modality.visual, "Too few non-zero DCT coefficients (small or very flat image) for a histogram test.")
    if score >= DJPEG_STRONG:
        return _f(name, Modality.visual, Stance.supports_manipulation, 0.6, 0.4,
                  f"DCT coefficient histograms have periodic empty bins (gap score {score:.2f} >= {DJPEG_STRONG}): the "
                  f"image was JPEG-compressed at least twice, the first time more coarsely (re-saved after editing or "
                  f"exporting).", lim, **ev)
    if score >= DJPEG_WEAK:
        return _f(name, Modality.visual, Stance.supports_manipulation, 0.35, 0.3,
                  f"Weak double-compression trace in the DCT histograms (gap score {score:.2f}, {DJPEG_WEAK}-"
                  f"{DJPEG_STRONG}): likely re-saved at least once.", lim, **ev)
    if score < DJPEG_SINGLE:
        return _f(name, Modality.visual, Stance.contradicts, 0.3, 0.3,
                  f"DCT histograms are smooth (gap score {score:.2f} < {DJPEG_SINGLE}): consistent with a single JPEG "
                  f"compression.", lim, **ev)
    return _f(name, Modality.visual, Stance.neutral, 0.0, 0.3,
              f"Gap score {score:.2f} is between single ({DJPEG_SINGLE}) and re-saved ({DJPEG_WEAK}) levels.", lim, **ev)


# ---------------------------------------------------------------- EXIF consistency

def _dt(v: Any) -> Optional[datetime]:
    m = re.match(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})", str(v))
    return datetime(*map(int, m.groups())) if m else None


def _thumbnail(path) -> Optional[np.ndarray]:
    if shutil.which("exiftool") is None:
        return None
    raw = subprocess.run(["exiftool", "-b", "-ThumbnailImage", str(path)], capture_output=True, timeout=60).stdout
    if not raw:
        return None
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    return img if img is not None and img.size else None


def _thumb_compare(thumb: np.ndarray, main_bgr: np.ndarray) -> dict[str, Any]:
    """Global correlation (best of 4 rotations: thumbnails ignore EXIF orientation) and a local difference map:
    the largest local mean |difference| (standardised units) and its ratio to the image's typical difference."""
    main = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2GRAY)
    th, tw = thumb.shape
    best = None
    for k in range(4):
        m = np.ascontiguousarray(np.rot90(main, k))
        a = cv2.resize(thumb, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(m, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
        a, b = (a - a.mean()) / (a.std() + 1e-6), (b - b.mean()) / (b.std() + 1e-6)
        corr = float((a * b).mean())
        if best is None or corr > best[0]:
            best = (corr, k, a, b)
    corr, k, a, b = best
    d = cv2.GaussianBlur(np.abs(a - b), (0, 0), 1.0)
    win = max(5, int(round(min(th, tw) * 0.15)))
    loc = cv2.boxFilter(d, -1, (win, win))
    typical = max(float(np.median(loc)), 1e-3)
    y, x = np.unravel_index(int(np.argmax(loc)), loc.shape)
    out = {"corr": round(corr, 3), "rotation_k": k, "local_max": round(float(loc.max()), 3),
           "local_ratio": round(float(loc.max()) / typical, 1), "box": None}
    if k == 0:  # box only when thumbnail and image share orientation
        H, W = main.shape
        sx, sy = W / tw, H / th
        out["box"] = (int(max(0, x - win / 2) * sx), int(max(0, y - win / 2) * sy), int(win * sx), int(win * sy))
    return out


def check_exif(ctx: JobContext, tags: dict, bgr: Optional[np.ndarray]) -> Finding:
    name = "image.exif_consistency"
    lim = ("EXIF is easy to edit or strip; consistent EXIF is weak evidence. Phones and messaging apps legitimately "
           "rewrite dates or drop thumbnails.")
    dto_k, dto = _tag(tags, "DateTimeOriginal")
    mod_k, mod = _tag(tags, "ModifyDate")
    sw_k, sw = _tag(tags, "Software")
    mk_k, make = _tag(tags, "Make")
    has_thumb_tag = any(k.split(":")[-1] in ("ThumbnailImage", "ThumbnailLength", "ThumbnailOffset") for k in tags)
    if not any((dto_k, mod_k, sw_k, mk_k, has_thumb_tag)):
        return _na(name, Modality.metadata, "No EXIF capture data (DateTimeOriginal / ModifyDate / Software / Make / "
                                            "thumbnail): stripped or never written - common for web, screenshots and "
                                            "messaging apps.")
    fields = [{"field": k, "value": str(v)} for k, v in ((dto_k, dto), (mod_k, mod), (sw_k, sw), (mk_k, make)) if k]
    issues: list[tuple[float, str]] = []
    t_orig, t_mod = _dt(dto), _dt(mod)
    if t_orig and t_mod:
        delta = (t_mod - t_orig).total_seconds()
        if delta > TIME_TOL_S:
            issues.append((0.4, f"{mod_k} = {mod!r} is {delta / 86400:.1f} days after {dto_k} = {dto!r}: "
                                f"the file was modified after capture"))
        elif delta < -TIME_TOL_S:
            issues.append((0.5, f"{mod_k} = {mod!r} is before {dto_k} = {dto!r}: impossible order"))
    if sw and any(e in str(sw).lower() for e in EDITORS):
        issues.append((0.45, f"{sw_k} = {sw!r} is editing software"))
    thumb = _thumbnail(ctx.media_path) if has_thumb_tag else None
    sim, region = None, None
    if thumb is not None and bgr is not None:
        tc = _thumb_compare(thumb, bgr)
        sim = tc["corr"]
        fields.append({"field": "IFD1:ThumbnailImage vs main image (correlation)", "value": str(sim)})
        fields.append({"field": "IFD1:ThumbnailImage vs main image (largest local difference, x typical)",
                       "value": f"{tc['local_max']} ({tc['local_ratio']}x)"})
        if sim < THUMB_MATCH:
            issues.append((0.65, f"the embedded EXIF thumbnail does not match the main image (correlation {sim:.2f} < "
                                 f"{THUMB_MATCH}): the picture changed after the thumbnail was written"))
        elif tc["local_max"] >= THUMB_LOCAL_DIFF and tc["local_ratio"] >= THUMB_LOCAL_RATIO:
            region = tc["box"]
            where = f" around {list(region)}" if region else ""
            issues.append((0.6, f"the embedded EXIF thumbnail matches overall (correlation {sim:.2f}) but differs "
                                f"locally{where} ({tc['local_ratio']:.0f}x the typical difference): that area was "
                                f"changed after the thumbnail was written"))
    ev = {"fields": fields, "thumbnail_similarity": sim, "issues": [t for _, t in issues]}
    if issues:
        p = 1.0
        for w, _ in issues:
            p *= 1 - w
        f = _f(name, Modality.metadata, Stance.supports_manipulation, round(min(0.95, 1 - p), 3), 0.5,
               "EXIF inconsistencies: " + "; ".join(t for _, t in issues) + ".", lim, **ev)
        if region:
            f.region = region
        return f
    checked = []
    if t_orig and t_mod:
        checked.append("capture and modify times agree")
    if sim is not None:
        checked.append(f"thumbnail matches the image (correlation {sim:.2f})")
    if not checked:
        return _f(name, Modality.metadata, Stance.neutral, 0.0, 0.3,
                  "EXIF present but without the fields needed to cross-check (no date pair, no thumbnail).", lim, **ev)
    return _f(name, Modality.metadata, Stance.contradicts, 0.3, 0.35, "EXIF consistent: " + "; ".join(checked) + ".",
              lim, **ev)


@register("image", Modality.metadata, ("image",))
def image_check(ctx: JobContext) -> list[Finding]:
    info = _jpeg.parse(ctx.media_path)
    tags = _exif(ctx)
    paths = ctx.frame_paths()
    bgr = cv2.imread(str(paths[0])) if paths else None
    out: list[Finding] = []
    for nm, mod, fn in (("image.jpeg_quant", Modality.metadata, lambda: check_quant(ctx, info, tags)),
                        ("image.double_jpeg", Modality.visual, lambda: check_double(ctx, info, bgr)),
                        ("image.exif_consistency", Modality.metadata, lambda: check_exif(ctx, tags, bgr))):
        try:
            out.append(fn())
        except Exception as exc:
            out.append(_na(nm, mod, f"Sub-check failed: {type(exc).__name__}: {exc}"))
    return out
