"""Visual forensics (thin): noise-residual & ELA region anomalies, temporal discontinuities, GOP structure.

Region checks compare every 16x16 block with *its own* statistics over the sampled frames (after removing
each frame's global level), so flat-but-genuine areas do not fire; inserted/replaced regions change
character over time and do. Single images fall back to a spatial-only comparison.
Temporal checks decode the original at native rate, low resolution.
Heatmaps go to <work_dir>/heatmaps/. score = strength for stated stance; reliability = trust in the method.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from backend.checks.registry import JobContext, register
from backend.config import DEEP_FPS, MAX_ANALYSIS_SECONDS
from backend.models import Depth, Finding, Modality, Segment, Stance

BLOCK = 16
Z_THRESH = 3.5
RATIO_THRESH = np.log(1.6)
MIN_EVENT_FRAC = 0.02  # component must cover >= 2% of the frame's blocks
MIN_TEMPORAL_FRAMES = 5
TEMP_W, TEMP_H = 160, 120
DROP_ISOLATION = 3.0
HARD_CUT_RATIO = 10.0
PTS_GAP_FACTOR = 1.8
MIN_EVENT_S = 0.75  # a region anomaly must persist this long in deep frames
ABS_FLOOR = 0.8  # grey levels RMS: blocks below this in both frame and baseline are ignored (flat/dark areas)


# ---------------------------------------------------------------- shared helpers

def _py(v):
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _finding(check: str, stance: Stance, score: Optional[float], reliability: float, rationale: str,
             limitations: str, segment: Optional[Segment] = None,
             region: Optional[tuple[int, int, int, int]] = None, depth: Depth = Depth.screen,
             coverage: Optional[float] = None, **evidence) -> Finding:
    return Finding(modality=Modality.visual, check=check, depth=depth, segment=segment, coverage=coverage,
                   region=tuple(int(v) for v in region) if region else None, stance=stance,
                   score=None if score is None else float(score), reliability=float(reliability),
                   rationale=rationale, limitations=limitations, evidence=_py(evidence))


def _na(check: str, why: str) -> Finding:
    return _finding(check, Stance.not_assessable, None, 0.0, why, why)


@lru_cache(maxsize=64)
def _probe_video(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", str(path)],
                         capture_output=True, text=True, timeout=60, check=True).stdout
    s = (json.loads(out).get("streams") or [{}])[0]
    fps = 0.0
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = str(s.get(key, "0/1")).partition("/")
        if float(den or 1) and float(num) > 0:
            fps = float(num) / float(den or 1)
            break
    return {"width": s.get("width"), "height": s.get("height"), "fps": fps or 25.0}


def _heat_overlay(frame: np.ndarray, zmap: np.ndarray, box: Optional[tuple[int, int, int, int]], out: Path,
                  label: str) -> str:
    h, w = frame.shape[:2]
    z = np.clip(np.abs(zmap) / 8.0, 0, 1)
    heat = cv2.applyColorMap((cv2.resize(z, (w, h), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    img = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
    if box:
        x, y, bw, bh = box
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
    cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return str(out)


# ---------------------------------------------------------------- region maps (noise residual, ELA)

def _blocks(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    return m[: h // BLOCK * BLOCK, : w // BLOCK * BLOCK].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3))


def noise_map(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r = g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32)
    return np.sqrt(_blocks(r * r))


def ela_map(img: np.ndarray, quality: int = 90) -> np.ndarray:
    re = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    return _blocks(np.abs(img.astype(np.float32) - re.astype(np.float32)).mean(2))


@dataclass
class RegionEvent:
    frames: list[int]  # sampled-frame indices
    box: tuple[int, int, int, int]  # pixels, union over frames
    peak: int
    blocks: int
    mean_abs_z: float
    mean_log_ratio: float


def _zscores(maps: np.ndarray, temporal: bool) -> tuple[np.ndarray, np.ndarray]:
    L = np.log(maps + 1e-3)
    L = L - np.median(L, axis=(1, 2), keepdims=True)  # remove per-frame level (I/P-frame quality swings)
    if temporal:
        med = np.median(L, 0)
        spread = np.median(np.abs(L - med), 0) * 1.4826
    else:
        med = np.median(L)
        spread = np.median(np.abs(L - med)) * 1.4826
    dev = L - med
    return dev / np.maximum(spread, 0.15), dev


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter or 1)


def _region_events(z: np.ndarray, dev: np.ndarray, min_frames: int) -> list[RegionEvent]:
    n_blocks = z.shape[1] * z.shape[2]
    min_blocks = max(6, int(MIN_EVENT_FRAC * n_blocks))
    mask = ((np.abs(z) > Z_THRESH) & (np.abs(dev) > RATIO_THRESH)).astype(np.uint8)
    comps: list[list[tuple[int, tuple[int, int, int, int], np.ndarray]]] = []
    for t in range(len(mask)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(mask[t], connectivity=8)
        comps.append([(t, tuple(int(v) for v in st[k, :4]), lab == k) for k in range(1, n) if st[k, 4] >= min_blocks])

    tracks: list[list[tuple[int, tuple[int, int, int, int], np.ndarray]]] = []
    for t, frame_comps in enumerate(comps):
        for c in frame_comps:
            for tr in tracks:
                if tr[-1][0] == t - 1 and _iou(tr[-1][1], c[1]) >= 0.3:
                    tr.append(c)
                    break
            else:
                tracks.append([c])

    events = []
    for tr in tracks:
        if len(tr) < min_frames:
            continue
        xs = [b[0] for _, b, _ in tr]
        ys = [b[1] for _, b, _ in tr]
        x2 = [b[0] + b[2] for _, b, _ in tr]
        y2 = [b[1] + b[3] for _, b, _ in tr]
        sizes = [int(m.sum()) for _, _, m in tr]
        peak = tr[int(np.argmax(sizes))][0]
        absz = np.concatenate([np.abs(z[t][m]) for t, _, m in tr])
        lr = np.concatenate([dev[t][m] for t, _, m in tr])
        box = (min(xs) * BLOCK, min(ys) * BLOCK, (max(x2) - min(xs)) * BLOCK, (max(y2) - min(ys)) * BLOCK)
        events.append(RegionEvent([t for t, _, _ in tr], box, peak, max(sizes),
                                  float(absz.mean()), float(lr.mean())))
    return events


def _duration(ctx: JobContext) -> float:
    return min(float(ctx.duration_s or 0.0) or MAX_ANALYSIS_SECONDS, MAX_ANALYSIS_SECONDS)


def _iv_cov(ctx: JobContext, iv: dict) -> float:
    dur = _duration(ctx)
    return round(min(1.0, (iv["end"] - iv["start"]) / dur), 4) if dur else 1.0


def _decode_bgr(path: Path, start: float, end: float, fps: float, w: int, h: int) -> list[np.ndarray]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}",
                          "-map", "0:v:0", "-vf", f"fps={fps}", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                         capture_output=True, timeout=600, check=True).stdout
    n = len(raw) // (w * h * 3)
    return list(np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3))


REGION_CHECKS = (("visual.noise_residual", "noise residual", noise_map, 0.55,
                  "Camera motion, focus changes and heavy compression alter local noise."),
                 ("visual.ela", "ELA error", ela_map, 0.35,
                  "ELA on decoded video frames is weak: inter-frame codec already equalizes error levels; "
                  "saturated colors and sharp edges inflate ELA."))


def _baseline(ctx: JobContext, fn: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Clip-wide per-block median/spread from the ingest frames (computed once per job and map)."""
    cache = ctx.extras.setdefault("_region_baseline", {})
    if fn.__name__ not in cache:
        paths = ctx.frame_paths()
        frames = [cv2.imread(str(p)) for p in paths[:: max(1, len(paths) // 40)]]
        raw = np.stack([fn(f) for f in frames])
        base = np.log(raw + 1e-3)
        base -= np.median(base, axis=(1, 2), keepdims=True)
        med = np.median(base, 0)
        cache[fn.__name__] = (med, np.maximum(np.median(np.abs(base - med), 0) * 1.4826, 0.15), frames[0].shape[:2],
                              np.median(raw, 0))
    return cache[fn.__name__]


def region_analyse(ctx: JobContext, iv: dict) -> dict[str, Any]:
    """Deep unit, one interval: full-res frames at DEEP_FPS, z-scored against the clip-wide baseline."""
    if len(ctx.frame_paths()) < MIN_TEMPORAL_FRAMES:
        return {"error": "too few ingest frames for a clip-wide baseline"}
    h, w = _baseline(ctx, noise_map)[2]
    frames = _decode_bgr(ctx.media_path, iv["start"], iv["end"], DEEP_FPS, w, h)
    if not frames:
        return {"error": "no frames decoded"}
    native_fps = _probe_video(ctx.media_path)["fps"]
    heat_dir = ctx.work_dir / "heatmaps"
    cov = _iv_cov(ctx, iv)
    out: dict[str, Any] = {"n_frames": len(frames)}
    for name, label, fn, reliability, lim in REGION_CHECKS:
        med, spread, _, base_abs = _baseline(ctx, fn)
        raw = np.stack([fn(f) for f in frames])
        L = np.log(raw + 1e-3)
        L -= np.median(L, axis=(1, 2), keepdims=True)
        dev = L - med
        z = dev / spread
        flat = (raw < ABS_FLOOR) & (base_abs[None] < ABS_FLOOR)  # ratio of two near-zero residuals is noise
        z[flat], dev[flat] = 0.0, 0.0
        found = []
        for k, ev in enumerate(_region_events(z, dev, min_frames=max(2, int(round(MIN_EVENT_S * DEEP_FPS))))):
            t0 = iv["start"] + ev.frames[0] / DEEP_FPS
            t1 = min(iv["end"], iv["start"] + (ev.frames[-1] + 1) / DEEP_FPS)
            heat = _heat_overlay(frames[ev.peak], z[ev.peak], ev.box, heat_dir / f"{name}_i{iv.get('order', 0)}_e{k}.png",
                                 f"{label} t={iv['start'] + ev.peak / DEEP_FPS:.2f}s")
            direction = "lower" if ev.mean_log_ratio < 0 else "higher"
            found.append(_finding(
                name, Stance.supports_manipulation, round(min(0.85, 0.3 + 0.04 * ev.mean_abs_z + 0.025 * len(ev.frames)), 3),
                reliability,
                f"Region {list(ev.box)} has {direction} {label} than its clip-wide baseline "
                f"(x{float(np.exp(abs(ev.mean_log_ratio))):.1f}, mean |z| {ev.mean_abs_z:.1f}, {ev.blocks} blocks) in "
                f"{len(ev.frames)} full-res frames at {DEEP_FPS:g} fps, {t0:.2f}-{t1:.2f}s ({iv['kind']} interval): "
                f"consistent with an inserted or replaced region.",
                lim, segment=Segment(start=round(t0, 3), end=round(t1, 3), unit="s"), region=ev.box, depth=Depth.deep,
                coverage=cov, frame_range=[int(t0 * native_fps), int(t1 * native_fps)], interval=iv, heatmap=heat,
                mean_abs_z=round(ev.mean_abs_z, 2), mean_log_ratio=round(ev.mean_log_ratio, 3)))
        out[name] = found
    return out


def _clean_in_interval(ctx: JobContext, name: str, iv: dict, reliability: float, score: float, text: str,
                       lim: str) -> Finding:
    """Clean evidence scoped to the interval actually analysed; coverage = that interval's share of the clip."""
    return _finding(name, Stance.contradicts, score, reliability, text, lim + " Clean evidence covers only this "
                    "interval; absence of anomalies is weak evidence.",
                    segment=Segment(start=round(iv["start"], 3), end=round(iv["end"], 3), unit="s"),
                    depth=Depth.deep, coverage=_iv_cov(ctx, iv), interval=iv)


def region_finalize(ctx: JobContext, results: list[tuple[dict, dict]]) -> list[Finding]:
    out: list[Finding] = []
    for name, label, _, reliability, lim in REGION_CHECKS:
        for iv, r in results:
            if "error" in r:
                out.append(_na(name, f"{iv['start']:.2f}-{iv['end']:.2f}s: {r['error']}"))
            elif r[name]:
                out.extend(r[name])
            else:
                out.append(_clean_in_interval(ctx, name, iv, reliability * 0.6, 0.15,
                                              f"No persistent {label} region anomaly in {r['n_frames']} full-res frames "
                                              f"at {iv['start']:.2f}-{iv['end']:.2f}s ({iv['kind']} interval).", lim))
    return out


def check_noise(ctx: JobContext) -> list[Finding]:
    """Still images only (texture/brightness-stratified spatial baseline); video uses the deep units."""
    return _image_region_check(ctx, "noise")


def check_ela(ctx: JobContext) -> list[Finding]:
    """Still images only; ELA is defined only for JPEGs (re-saved at the file's own estimated quality)."""
    return _image_region_check(ctx, "ela")


# ---------------------------------------------------------------- still images: stratified spatial check

IMG_Z = 4.0
IMG_RATIO = np.log(1.8)
IMG_EVENT_Z = 5.0  # an event's mean |z| must be well above the per-block threshold
IMG_MIN_EVENT_FRAC = 0.015  # of assessable blocks
IMG_MIN_EVENT_BLOCKS = 16
IMG_MIN_FILL = 0.35  # flagged blocks / bounding-box blocks: compact regions, not scattered texture
IMG_MIN_COVERAGE = 0.2
IMG_STRATUM_MIN = 25
IMG_FLAT_STD = 2.0  # grey levels: flatter blocks carry no measurable noise after JPEG
_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
IMG_CHECKS = {
    "noise": ("visual.noise_residual", "noise residual", 0.45, 0.30,
              "Strong lens blur, heavy denoising or beauty filters also lower local noise; very small edits fall below "
              "the region-size threshold."),
    "ela": ("visual.ela", "ELA error", 0.30, 0.20,
            "ELA reflects local compression history; repeated saves, sharpening and text overlays also change it, so "
            "it is weak evidence on its own."),
}


def _bmean(m: np.ndarray, b: int) -> np.ndarray:
    h, w = m.shape
    return m[: h // b * b, : w // b * b].reshape(h // b, b, w // b, b).mean((1, 3))


def _strata_z(v: np.ndarray, bright: np.ndarray, tex: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust z of log(value) against blocks of similar brightness and texture (3 x 5 quantile strata; strata too
    small for a baseline fall back to texture-only bins)."""
    L = np.log(v + 1e-3)
    z = np.zeros_like(L)
    dev = np.zeros_like(L)
    bq = np.quantile(bright[ok], [1 / 3, 2 / 3])
    tq = np.quantile(tex[ok], [0.2, 0.4, 0.6, 0.8])
    bb, tb = np.digitize(bright, bq), np.digitize(tex, tq)
    for t in range(5):
        tex_mask = ok & (tb == t)
        for b in range(3):
            m = tex_mask & (bb == b)
            ref = m if m.sum() >= IMG_STRATUM_MIN else tex_mask
            if ref.sum() < IMG_STRATUM_MIN:
                continue
            med = np.median(L[ref])
            spread = max(float(np.median(np.abs(L[ref] - med))) * 1.4826, 0.15)
            dev[m] = L[m] - med
            z[m] = dev[m] / spread
    return z, dev


def _image_region_check(ctx: JobContext, kind: str) -> list[Finding]:
    from backend.checks import _jpeg

    name, label, rel_event, rel_clean_cap, lim = IMG_CHECKS[kind]
    paths = ctx.frame_paths()
    if not paths:
        return [_na(name, "No decoded image available.")]
    bgr = cv2.imread(str(paths[0]))
    if kind == "ela":
        info = _jpeg.parse(ctx.media_path)
        if info is None or 0 not in info.tables:
            return [_na(name, "Not a JPEG (no JPEG compression history): ELA is undefined for this file.")]
        quality, _ = _jpeg.ijg_quality(info.tables[0])
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    blk = 16 if max(h, w) <= 2048 else 32
    bright = _bmean(g, blk)
    raw_std = np.sqrt(np.maximum(_bmean(g * g, blk) - bright ** 2, 0))
    sm = cv2.GaussianBlur(g, (0, 0), 2.0)
    tex = np.sqrt(np.maximum(_bmean(sm * sm, blk) - _bmean(sm, blk) ** 2, 0))  # structure, not noise
    clipped = _bmean(((g >= 250) | (g <= 5)).astype(np.float32), blk)
    ok = (bright >= 16) & (bright <= 239) & (clipped < 0.05) & (raw_std >= IMG_FLAT_STD)
    coverage = round(float(ok.mean()), 4)
    if kind == "noise":
        val = _bmean(np.abs(cv2.filter2D(g, -1, _IMMERKAER)), blk) * np.sqrt(np.pi / 2) / 6
    else:
        val = _bmean(np.abs(bgr.astype(np.float32) - _jpeg.resave(bgr, quality).astype(np.float32)).mean(2), blk)
    ev_base = {"coverage": coverage, "block_px": blk, "mode": "spatial, brightness x texture strata",
               **({"ela_quality": quality} if kind == "ela" else {})}
    if coverage < IMG_MIN_COVERAGE or ok.sum() < 3 * IMG_STRATUM_MIN:
        return [_finding(name, Stance.neutral, 0.0, 0.2,
                         f"Only {coverage:.0%} of the image is assessable for {label} (flat, saturated or near-black "
                         f"blocks are skipped); no conclusion.", lim, **ev_base)]

    z, dev = _strata_z(val, bright, tex, ok)
    mask = (ok & (np.abs(z) > IMG_Z) & (np.abs(dev) > IMG_RATIO)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_blocks = max(IMG_MIN_EVENT_BLOCKS, int(IMG_MIN_EVENT_FRAC * ok.sum()))
    heat_dir = ctx.work_dir / "heatmaps"
    zshow = np.where(ok, z, 0.0)
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[k])
        comp = lab == k
        mz = float(np.abs(z[comp]).mean())
        if area < min_blocks or mz < IMG_EVENT_Z or area / float(bw * bh) < IMG_MIN_FILL:
            continue
        box = (x * blk, y * blk, bw * blk, bh * blk)
        lr = float(dev[comp].mean())
        heat = _heat_overlay(bgr, zshow, box, heat_dir / f"{name}_event{len(out)}.png", label)
        out.append(_finding(
            name, Stance.supports_manipulation, round(min(0.8, 0.3 + 0.04 * mz), 3), rel_event,
            f"Region {list(box)} has {'lower' if lr < 0 else 'higher'} {label} than blocks of similar brightness and "
            f"texture elsewhere in the image (x{float(np.exp(abs(lr))):.1f}, mean |z| {mz:.1f}, {area} blocks): "
            f"consistent with a pasted or retouched region.", lim, region=box, heatmap=heat,
            mean_abs_z=round(mz, 2), mean_log_ratio=round(lr, 3), blocks=area, **ev_base))
    if out:
        return out
    overview = _heat_overlay(bgr, zshow, None, heat_dir / f"{name}_overview.png", f"{label} |z|")
    return [_finding(name, Stance.contradicts, 0.3, round(rel_clean_cap * coverage, 3),
                     f"No region of {label} stands out from blocks of similar brightness and texture ({coverage:.0%} of "
                     f"the image assessable).", lim + " Clean evidence covers only the assessable blocks and is weak.",
                     coverage=coverage, heatmap=overview,
                     **{k: v for k, v in ev_base.items() if k != "coverage"})]


# ---------------------------------------------------------------- temporal (dup / drop / flow jumps / pts gaps)

def _decode_small(path: Path, start: float = 0.0, end: float = MAX_ANALYSIS_SECONDS) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}", "-map", "0:v:0",
         "-vf", f"scale={TEMP_W}:{TEMP_H},format=gray", "-f", "rawvideo", "-"],
        capture_output=True, timeout=600, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, TEMP_H, TEMP_W)


@lru_cache(maxsize=64)
def _packets(path: Path) -> list[dict]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals",
                          f"%+{MAX_ANALYSIS_SECONDS}", "-show_entries", "packet=pts_time,flags", "-of", "json",
                          str(path)], capture_output=True, text=True, timeout=300, check=True).stdout
    pk = [p for p in json.loads(out).get("packets", []) if p.get("pts_time") not in (None, "N/A")]
    return sorted(({"t": float(p["pts_time"]), "key": "K" in p.get("flags", "")} for p in pk), key=lambda p: p["t"])


def _gaps(path: Path) -> tuple[list[tuple[float, float]], float]:
    pk = _packets(path)
    if len(pk) <= 2:
        return [], 0.0
    deltas = np.diff([p["t"] for p in pk])
    nominal = float(np.median(deltas))
    return [(pk[i]["t"], float(deltas[i])) for i in range(len(deltas))
            if nominal and deltas[i] > PTS_GAP_FACTOR * nominal], nominal


def _local_median(a: np.ndarray, i: int, r: int = 8) -> float:
    return float(np.median(np.r_[a[max(0, i - r):i], a[i + 1:i + 1 + r]]))


def _flow_p998(a: np.ndarray, b: np.ndarray) -> float:
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.percentile(np.linalg.norm(fl, axis=2), 99.8))


TEMPORAL_MARGIN_S = 0.5
TEMPORAL_LIM = ("Fast genuine motion, flashes or blinks can look like discontinuities; frame-rate conversion produces "
                "periodic duplicates; duplicates in low-motion stretches are indistinguishable from encoder skips and "
                "are ignored; scene cuts are normal in edited footage.")


def check_pts_gaps(ctx: JobContext) -> list[Finding]:
    """Whole-file, cheap: container timestamp gaps."""
    name = "visual.pts_gaps"
    if ctx.media_type != "video":
        return [_na(name, "Timestamp gaps need video.")]
    gaps, nominal = _gaps(ctx.media_path)
    out = [_finding(name, Stance.supports_manipulation, 0.5, 0.5,
                    f"Packet pts_time jumps from {t:.3f}s by {dt:.3f}s (nominal {nominal:.3f}s): "
                    f"~{dt / nominal - 1:.0f} frame(s) missing from the timeline.",
                    "Variable-frame-rate capture (phones, screen recorders) also produces gaps.",
                    segment=Segment(start=round(t, 3), end=round(t + dt, 3), unit="s"), coverage=1.0,
                    gap_s=round(dt, 4), nominal_s=round(nominal, 4)) for t, dt in gaps[:10]]
    return out or [_finding(name, Stance.neutral, 0.0, 0.4, f"No container timestamp gaps (nominal frame duration "
                            f"{nominal:.4f}s).", "Gaps are only one way frames go missing.", coverage=1.0)]


def temporal_analyse(ctx: JobContext, iv: dict) -> dict[str, Any]:
    """Deep unit, one interval (+margin): duplicates, isolated jumps, hard cuts at native rate."""
    fps = _probe_video(ctx.media_path)["fps"]
    start = max(0.0, iv["start"] - TEMPORAL_MARGIN_S)
    G = _decode_small(ctx.media_path, start, iv["end"] + TEMPORAL_MARGIN_S)
    if len(G) < 10:
        return {"error": f"too few frames ({len(G)})"}
    found = _temporal_events(G, int(round(start * fps)), fps, _gaps(ctx.media_path)[0], iv, _iv_cov(ctx, iv),
                             "visual.temporal", TEMPORAL_LIM, ctx.work_dir / "heatmaps", 1000 * iv.get("order", 0))
    return {"findings": found, "n_frames": len(G)}


def temporal_finalize(ctx: JobContext, results: list[tuple[dict, dict]]) -> list[Finding]:
    out: list[Finding] = []
    for iv, r in results:
        if "error" in r:
            out.append(_na("visual.temporal", f"{iv['start']:.2f}-{iv['end']:.2f}s: {r['error']}"))
            continue
        out.extend(r["findings"])
        if not any(f.stance == Stance.supports_manipulation for f in r["findings"]):
            out.append(_clean_in_interval(ctx, "visual.temporal", iv, 0.35, 0.2,
                                          f"No duplicated frames or isolated jumps in {r['n_frames']} frames at "
                                          f"{iv['start']:.2f}-{iv['end']:.2f}s ({iv['kind']} interval).", TEMPORAL_LIM))
    return out


def _temporal_events(G: np.ndarray, i0: int, fps: float, gaps: list[tuple[float, float]], iv: dict, cov: float,
                     name: str, lim: str, heat_dir: Path, k0: int) -> list[Finding]:
    """i0 = absolute index of G[0]; returned frame numbers and times are absolute."""
    blur = np.stack([cv2.GaussianBlur(f, (0, 0), 1.5) for f in G])
    S = np.stack([cv2.resize(cv2.GaussianBlur(f.astype(np.float32), (0, 0), 1.5), (80, 60),
                             interpolation=cv2.INTER_AREA) for f in G])
    S -= S.mean((1, 2), keepdims=True)  # remove global brightness pumping
    d = np.abs(np.diff(S, axis=0)).mean((1, 2))  # d[i] = change from frame i to i+1
    med_all = float(np.median(d))
    out: list[Finding] = []
    deep = {"depth": Depth.deep, "coverage": cov, "interval": iv}

    def seg_at(i: int, span: int = 1) -> Segment:
        return Segment(start=round((i0 + i) / fps, 3), end=round((i0 + i + span) / fps, 3), unit="s")

    def pair_image(i: int, label: str) -> str:
        a, b = G[i], G[i + 1]
        diff = cv2.applyColorMap(np.clip(np.abs(a.astype(np.int16) - b) * 4, 0, 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        img = np.hstack([cv2.cvtColor(a, cv2.COLOR_GRAY2BGR), cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), diff])
        img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        p = heat_dir / f"{name}_event{k0 + len(out)}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        return str(p)

    if med_all > 0.05:
        # A duplicate sits between normally-moving neighbours. In low-motion stretches encoders skip
        # frames (P-skip) and produce identical output, so those are not assessable as duplicates.
        dup = []
        for i in range(edge_dup := 8, len(d) - edge_dup):
            if not iv["start"] - 1 / fps <= (i0 + i + 1) / fps < iv["end"]:
                continue
            lm = _local_median(d, i)
            if d[i] < max(0.01, 0.05 * lm) and min(d[i - 1], d[i + 1]) > 0.5 * lm and lm >= med_all:
                dup.append(i)
        runs: list[list[int]] = []
        for i in dup:
            if runs and i == runs[-1][-1] + 1:
                runs[-1].append(i)
            else:
                runs.append([i])
        spacing = np.diff(dup) if len(dup) > 2 else np.array([])
        if len(spacing) >= 3 and float(np.std(spacing)) < 0.5:
            out.append(_finding(name, Stance.neutral, 0.0, 0.4,
                                f"{len(dup)} duplicated frames repeating every {int(np.median(spacing))} frames: "
                                f"frame-rate conversion pattern, not an edit.", lim, kind="periodic_duplicates",
                                frames=[i0 + i + 1 for i in dup[:50]], **deep))
        else:
            for r in runs[:10]:
                out.append(_finding(name, Stance.supports_manipulation, round(min(0.6, 0.3 + 0.05 * len(r)), 3), 0.45,
                                    f"Frame(s) {i0 + r[0] + 1}-{i0 + r[-1] + 1} ({(i0 + r[0] + 1) / fps:.2f}s) are "
                                    f"duplicates of the previous frame (mean diff {d[r[0]]:.3f} vs typical "
                                    f"{med_all:.3f}) in an otherwise moving sequence.", lim,
                                    segment=seg_at(r[0] + 1, len(r)), kind="duplicate",
                                    frame_range=[i0 + r[0] + 1, i0 + r[-1] + 1],
                                    heatmap=pair_image(r[0], f"dup {i0 + r[0] + 1}"), **deep))

    edge = 8  # local-median radius: frames nearer the decode edges have one-sided statistics (and a seek
    # may decode the first frames slightly differently), so they are left to the neighbouring interval
    lo_t, hi_t = iv["start"] - 1 / fps, iv["end"]
    for i in range(len(d)):
        if i < edge or i >= len(d) - edge or not lo_t <= (i0 + i + 1) / fps < hi_t:
            continue
        nb = [d[j] for j in (i - 1, i + 1) if 0 <= j < len(d)]
        lm = max(_local_median(d, i), 1e-3)
        ratio = d[i] / lm
        isolation = min(d[i] / max(max(nb), 1e-3), ratio)
        a_i = i0 + i
        if ratio >= HARD_CUT_RATIO and d[i] > 3.0:
            out.append(_finding(name, Stance.supports_manipulation, 0.3, 0.3,
                                f"Hard cut between frames {a_i} and {a_i + 1} ({(a_i + 1) / fps:.2f}s): content "
                                f"change x{ratio:.1f} the local level. The sequence is edited (shot change).",
                                lim + " Shot changes alone do not imply deceptive manipulation.",
                                segment=seg_at(i + 1), kind="hard_cut", frame_range=[a_i, a_i + 1],
                                heatmap=pair_image(i, f"cut {a_i}->{a_i + 1}"), **deep))
        elif isolation >= DROP_ISOLATION:
            nbr = [_flow_p998(blur[j], blur[j + 1]) for j in range(max(0, i - 3), min(len(d), i + 4)) if j != i]
            flow_jump = _flow_p998(blur[i], blur[i + 1]) / max(float(np.median(nbr)) if nbr else 0.0, 0.3)
            gap_hit = any(abs(t - (a_i + 1) / fps) < 1.5 / fps for t, _ in gaps)
            score = min(0.8, 0.35 + 0.1 * (isolation - DROP_ISOLATION) + (0.1 if flow_jump > 2 else 0)
                        + (0.15 if gap_hit else 0))
            out.append(_finding(
                name, Stance.supports_manipulation, round(score, 3), 0.45,
                f"Isolated temporal discontinuity between frames {a_i} and {a_i + 1} ({(a_i + 1) / fps:.2f}s): "
                f"change is x{ratio:.1f} the local level and x{isolation:.1f} both neighbours; optical-flow p99.8 "
                f"jump x{flow_jump:.1f}. Consistent with dropped/removed frames or a splice in continuous footage.",
                lim, segment=seg_at(i + 1), kind="drop_or_splice", frame_range=[a_i, a_i + 1],
                isolation=round(isolation, 2), local_ratio=round(ratio, 2), flow_jump=round(flow_jump, 2),
                pts_gap=gap_hit, heatmap=pair_image(i, f"jump {a_i}->{a_i + 1}"), **deep))
    return out


# ---------------------------------------------------------------- GOP / I-frame spacing

def check_gop(ctx: JobContext) -> list[Finding]:
    name = "visual.gop"
    if ctx.media_type != "video":
        return [_na(name, "GOP analysis needs video.")]
    pk = _packets(ctx.media_path)
    keys = [p["t"] for p in pk if p["key"]]
    n = len(pk)
    lim = ("Encoders place I-frames adaptively at scene changes; irregular spacing alone is normal. "
           "Double-compression traces are assessed in metadata.reencode_generations.")
    ev = {"keyframe_times_s": [round(t, 3) for t in keys[:100]], "packets": n}
    if n and len(keys) >= 0.9 * n:
        return [_finding(name, Stance.neutral, 0.0, 0.3, f"All-intra stream ({len(keys)}/{n} key packets).", lim, **ev)]
    if len(keys) < 2:
        return [_finding(name, Stance.neutral, 0.0, 0.3,
                         f"Single I-frame at {keys[0]:.2f}s in {n} packets; no GOP spacing to compare." if keys
                         else "No key packets reported.", lim, **ev)]
    iv = np.diff(keys)
    ev["intervals_s"] = [round(float(x), 3) for x in iv]
    best = None
    for s in range(3, len(iv) - 2):
        a, b = iv[:s], iv[s:]
        ma, mb = float(np.median(a)), float(np.median(b))
        cva, cvb = float(np.std(a) / ma), float(np.std(b) / mb)
        jump = abs(ma - mb) / min(ma, mb)
        if jump > 0.3 and cva < 0.15 and cvb < 0.15 and (best is None or jump > best[0]):
            best = (jump, s, ma, mb)
    if best:
        jump, s, ma, mb = best
        t = keys[s]
        return [_finding(name, Stance.supports_manipulation, round(min(0.6, 0.3 + 0.3 * jump), 3), 0.4,
                         f"I-frame spacing changes regime at {t:.2f}s: regular {ma:.2f}s intervals before, "
                         f"{mb:.2f}s after. Consistent with segments encoded separately and joined.",
                         lim, segment=Segment(start=round(keys[s - 1], 3), end=round(keys[s + 1], 3), unit="s"),
                         **ev)]
    return [_finding(name, Stance.neutral, 0.0, 0.35,
                     f"{len(keys)} I-frames, median spacing {float(np.median(iv)):.2f}s "
                     f"(min {float(iv.min()):.2f}, max {float(iv.max()):.2f}); no regime change.", lim, **ev)]


# ---------------------------------------------------------------- entry point

SUBCHECKS: list[tuple[str, Callable[[JobContext], list[Finding]]]] = [
    ("visual.pts_gaps", check_pts_gaps),
    ("visual.gop", check_gop),
]
IMAGE_SUBCHECKS = [("visual.noise_residual", check_noise), ("visual.ela", check_ela)]


@register("visual", Modality.visual, ("video", "image"))
def visual_check(ctx: JobContext) -> list[Finding]:
    """Whole-file visual checks. Frame-level deep checks for video run only through the scheduler's deadline
    loop (backend/adaptive/deep.py) via region_analyse/temporal_analyse."""
    out: list[Finding] = []
    for name, fn in (IMAGE_SUBCHECKS if ctx.media_type == "image" else SUBCHECKS):
        try:
            out.extend(fn(ctx))
        except Exception as exc:
            out.append(_na(name, f"Sub-check failed: {type(exc).__name__}: {exc}"))
    return out
