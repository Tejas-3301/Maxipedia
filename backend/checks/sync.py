"""Audio-video synchronisation: mouth aperture (MediaPipe Face Landmarker) vs speech-band audio envelope.

Per frame at SYNC_FPS: inner-lip gap / inter-ocular distance; speech-band (300-3400 Hz) log energy from the
16 kHz analysis wav. Sliding windows (2 s, hop 0.5 s): normalised cross-correlation over +-0.4 s lags.
The clip's own offset is the median best lag of confident windows; flagged are windows whose correlation
drops (mouth no longer matches the sound) or whose best lag drifts from the clip offset. Windows without a
face or without speech are not assessable and count neither as clean evidence nor as coverage.
Runs only as a deep unit inside the scheduler's deadline loop (or an investigator's "analyze deeper"); never
over the whole clip on its own. Output: offset curve (JSON + PNG) and per-interval findings.
"""
from __future__ import annotations

import json
import os
import subprocess
import wave
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from backend.checks.registry import JobContext
from backend.config import MAX_ANALYSIS_SECONDS, ROOT
from backend.models import Depth, Finding, Modality, Segment, Stance

os.environ.setdefault("GLOG_minloglevel", "2")  # silence MediaPipe/TFLite info logs
MODEL = ROOT / "models" / "face_landmarker.task"
SYNC_FPS = 25.0
WIN_S, HOP_S = 2.0, 0.5
MAX_LAG_S = 0.4
MIN_FACE_FRAC = 0.7
MIN_SPEECH_FRAC = 0.3
CONFIDENT_R = 0.3
DROP_R = 0.1  # correlation at the clip offset below this = mouth and sound no longer match
DRIFT_S = 0.08  # best lag further than this from the clip offset (with a clear peak) = drift
DRIFT_MIN_R = 0.4
GLOBAL_OFFSET_S = 0.2
MIN_SPAN_S = 3.0  # an interval is widened to this so that 2 s windows fit
UPPER_LIP, LOWER_LIP, EYE_L, EYE_R = 13, 14, 33, 263


def _f(stance: Stance, score: Optional[float], rel: float, rationale: str, lim: str,
       segment: Optional[Segment] = None, coverage: Optional[float] = None, **evidence) -> Finding:
    return Finding(modality=Modality.sync, check="sync.lipsync", depth=Depth.deep, stance=stance, score=score,
                   reliability=rel, rationale=rationale, limitations=lim, segment=segment, coverage=coverage,
                   evidence=evidence)


def _na(why: str, **ev) -> Finding:
    return _f(Stance.not_assessable, None, 0.0, why, why, **ev)


# ---------------------------------------------------------------- signals

def mouth_aperture(path: Path, start: float, end: float) -> np.ndarray:
    """Normalised inner-lip opening per frame at SYNC_FPS; NaN where no face is found."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if w > 640:
        w, h = 640, int(round(h * 640 / w / 2) * 2)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}",
                          "-map", "0:v:0", "-vf", f"fps={SYNC_FPS},scale={w}:{h}", "-pix_fmt", "rgb24", "-f", "rawvideo",
                          "-"], capture_output=True, timeout=600, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODEL)),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1,
                                        min_face_detection_confidence=0.4, min_face_presence_confidence=0.4)
    out = np.full(len(frames), np.nan)
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        for i, fr in enumerate(frames):
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)),
                                    int(round(i * 1000 / SYNC_FPS)))
            if r.face_landmarks:
                p = r.face_landmarks[0]
                gap = np.hypot((p[LOWER_LIP].x - p[UPPER_LIP].x) * w, (p[LOWER_LIP].y - p[UPPER_LIP].y) * h)
                eyes = np.hypot((p[EYE_R].x - p[EYE_L].x) * w, (p[EYE_R].y - p[EYE_L].y) * h)
                out[i] = gap / eyes if eyes > 1 else np.nan
    return out


def audio_envelope(wav_path: Path, start: float, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Speech-band log energy per video frame and a speech-present mask."""
    with wave.open(str(wav_path)) as wf:
        sr = wf.getframerate()
        x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    hop = sr / SYNC_FPS
    win = int(2 * hop)
    f = np.fft.rfftfreq(win, 1 / sr)
    band = (f >= 300) & (f <= 3400)
    hann = np.hanning(win).astype(np.float32)
    env = np.full(n_frames, -120.0)
    for i in range(n_frames):
        c = int((start + i / SYNC_FPS) * sr)
        seg = x[max(0, c - win // 2): max(0, c - win // 2) + win]
        if len(seg) == win:
            env[i] = 10 * np.log10(float((np.abs(np.fft.rfft(seg * hann))[band] ** 2).sum()) + 1e-12)
    live = env > -110
    floor = float(np.percentile(env[live], 10)) if live.any() else -120.0
    return env, env >= floor + 15


def _ncc(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Correlation of a[t] with b[t + lag] over frames valid in both."""
    if lag >= 0:
        x, y = a[: len(a) - lag], b[lag:]
    else:
        x, y = a[-lag:], b[: len(b) + lag]
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 10:
        return np.nan
    x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


# ---------------------------------------------------------------- windows

def analyse_span(mouth: np.ndarray, env: np.ndarray, speech: np.ndarray, t0: float) -> list[dict[str, Any]]:
    n_win, n_hop, max_lag = int(WIN_S * SYNC_FPS), int(HOP_S * SYNC_FPS), int(MAX_LAG_S * SYNC_FPS)
    lags = range(-max_lag, max_lag + 1)
    out = []
    for s in range(0, max(1, len(mouth) - n_win + 1), n_hop):
        m, e, sp = mouth[s:s + n_win], env[s:s + n_win], speech[s:s + n_win]
        w = {"start": round(t0 + s / SYNC_FPS, 3), "end": round(t0 + (s + len(m)) / SYNC_FPS, 3)}
        face = float(np.mean(~np.isnan(m)))
        talk = float(np.mean(sp))
        w.update(face_frac=round(face, 2), speech_frac=round(talk, 2))
        if face < MIN_FACE_FRAC:
            out.append({**w, "status": "no_face"})
            continue
        if talk < MIN_SPEECH_FRAC:
            out.append({**w, "status": "no_speech"})
            continue
        # positive lag = audio later than video
        rs = {lag: _ncc(m, e, lag) for lag in lags}
        rs = {k: v for k, v in rs.items() if not np.isnan(v)}
        if not rs:
            out.append({**w, "status": "no_face"})
            continue
        best = max(rs, key=rs.get)
        out.append({**w, "status": "ok", "best_lag_s": round(best / SYNC_FPS, 3), "r_max": round(rs[best], 3),
                    "r_by_lag": {str(k): round(v, 3) for k, v in rs.items()}})
    return out


def _merge(windows: list[dict], key: str) -> list[tuple[float, float, list[dict]]]:
    spans: list[tuple[float, float, list[dict]]] = []
    for w in windows:
        if not w.get(key):
            continue
        if spans and w["start"] <= spans[-1][1]:
            s, e, ws = spans[-1]
            spans[-1] = (s, max(e, w["end"]), ws + [w])
        else:
            spans.append((w["start"], w["end"], [w]))
    return spans


def plot_curve(windows: list[dict], offset: float, duration: float, out: Path) -> str:
    W, H, pad = 900, 240, 30
    img = np.full((H, W, 3), 255, np.uint8)
    tx = lambda t: int(pad + (W - 2 * pad) * t / max(duration, 1e-6))
    ly = lambda lag: int(H / 2 - lag / MAX_LAG_S * (H / 2 - pad))
    ry = lambda r: int(H - pad - max(-1, min(1, r)) * (H / 2 - pad))
    cv2.line(img, (pad, ly(0)), (W - pad, ly(0)), (200, 200, 200), 1)
    cv2.line(img, (pad, ly(offset)), (W - pad, ly(offset)), (180, 220, 180), 1)
    for w in windows:
        x0, x1 = tx(w["start"]), tx(w["end"])
        if w["status"] != "ok":
            cv2.rectangle(img, (x0, pad), (x1, H - pad), (235, 235, 235), -1)
        elif w.get("flag"):
            cv2.rectangle(img, (x0, pad), (x1, H - pad), (210, 210, 255), -1)
    pts_lag = [(tx((w["start"] + w["end"]) / 2), ly(w["best_lag_s"])) for w in windows if w["status"] == "ok"]
    pts_r = [(tx((w["start"] + w["end"]) / 2), ry(w["r_at_offset"])) for w in windows if w["status"] == "ok"]
    for a, b in zip(pts_lag, pts_lag[1:]):
        cv2.line(img, a, b, (200, 80, 0), 2)
    for a, b in zip(pts_r, pts_r[1:]):
        cv2.line(img, a, b, (0, 140, 0), 1)
    cv2.putText(img, f"offset (blue, +-{MAX_LAG_S:.1f}s)  r at clip offset (green)  red=flagged grey=not assessable",
                (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return str(out)


# ---------------------------------------------------------------- deep unit (scheduler-driven)

LIM = ("Lip-audio correlation is weak for off-screen speakers, voice-over, music, multiple faces or heavy "
       "occlusion; a genuine but sloppy mux shows a constant offset, not a local drop.")


def _dur(ctx: JobContext) -> float:
    return min(ctx.duration_s or 0.0, MAX_ANALYSIS_SECONDS)


def sync_analyse(ctx: JobContext, iv: dict) -> dict[str, Any]:
    """One interval, widened to >= MIN_SPAN_S so at least a couple of 2 s windows fit."""
    if not MODEL.exists():
        return {"error": "Face Landmarker model missing (run scripts/fetch_models.py)"}
    if not ctx.audio_path or not ctx.audio_path.exists():
        return {"error": "no audio track to compare with lip movement"}
    dur = _dur(ctx)
    s, e = iv["start"], iv["end"]
    if e - s < MIN_SPAN_S:
        c = (s + e) / 2
        s, e = c - MIN_SPAN_S / 2, c + MIN_SPAN_S / 2
    # Widen by half a window so windows straddling interval boundaries are evaluated, and snap to an absolute
    # hop grid so windows from neighbouring intervals coincide (finalize de-duplicates them by start time).
    hop = int(HOP_S * SYNC_FPS) / SYNC_FPS
    s = max(0.0, np.floor((s - WIN_S / 2) / hop) * hop)
    e = min(dur, e + WIN_S / 2)
    mouth = mouth_aperture(ctx.media_path, s, e)
    if len(mouth) == 0:
        return {"error": "no frames decoded"}
    env, speech = audio_envelope(ctx.audio_path, s, len(mouth))
    return {"windows": analyse_span(mouth, env, speech, s), "span": [round(s, 3), round(e, 3)]}


def _cov(spans: list[tuple[float, float, list]], dur: float) -> float:
    return round(min(1.0, sum(e - s for s, e, _ in spans) / dur), 4) if dur else 0.0


def sync_finalize(ctx: JobContext, results: list[tuple[dict, dict]]) -> list[Finding]:
    dur = _dur(ctx)
    errors = sorted({r["error"] for _, r in results if "error" in r})
    if not results or (errors and all("error" in r for _, r in results)):
        return [_na("; ".join(errors) or "No interval was scheduled for lip-sync analysis.")]
    seen: dict[float, dict] = {}
    for _, r in results:
        for w in r.get("windows", []):
            seen.setdefault(round(w["start"], 2), w)  # overlapping spans: keep one window per start time
    windows = [seen[k] for k in sorted(seen)]
    ok = [w for w in windows if w["status"] == "ok"]

    out: list[Finding] = []
    for s, e, ws in _merge([{**w, "na": w["status"] != "ok"} for w in windows], "na"):
        reasons = sorted({w["status"] for w in ws})
        # not assessable: no clean evidence and no coverage
        out.append(_na(f"{s:.2f}-{e:.2f}s not assessable for lip sync ({', '.join(reasons)}).",
                       segment=Segment(start=s, end=e, unit="s"), reasons=reasons))
    if len(ok) < 2:
        return out or [_na("Too few windows with both a visible face and speech.")]

    confident = [w for w in ok if w["r_max"] >= CONFIDENT_R]
    offset = float(np.median([w["best_lag_s"] for w in (confident or ok)]))
    off_key = str(int(round(offset * SYNC_FPS)))
    for w in ok:
        w["r_at_offset"] = w["r_by_lag"].get(off_key, np.nan)
        w["drop"] = bool(w["r_at_offset"] < DROP_R)
        w["drift"] = bool(abs(w["best_lag_s"] - offset) > DRIFT_S and w["r_max"] >= DRIFT_MIN_R)
        w["flag"] = w["drop"] or w["drift"]
        w["clean"] = not w["flag"]
    clip_r = float(np.median([w["r_at_offset"] for w in ok]))
    usable = _merge(ok, "status")  # windows with a usable offset estimate = covered time
    usable_cov = _cov(usable, dur)

    curve = [{k: v for k, v in w.items() if k not in ("r_by_lag", "na")} for w in windows]
    curve_path = ctx.work_dir / "sync_curve.json"
    curve_path.write_text(json.dumps({"fps": SYNC_FPS, "window_s": WIN_S, "hop_s": HOP_S, "clip_offset_s": offset,
                                      "clip_median_r": clip_r, "usable_coverage": usable_cov,
                                      "windows": curve}, indent=1, default=float))
    plot = plot_curve(windows, offset, dur, ctx.work_dir / "heatmaps" / "sync_offset_curve.png")
    ev = {"clip_offset_s": round(offset, 3), "clip_median_r": round(clip_r, 3), "curve_path": str(curve_path),
          "plot": plot, "assessable_windows": len(ok), "windows": len(windows), "usable_coverage": usable_cov}

    if clip_r < CONFIDENT_R:
        out.append(_f(Stance.neutral, 0.0, 0.3,
                      f"Mouth movement and speech energy correlate only weakly in the analysed windows (median r "
                      f"{clip_r:.2f} at offset {offset * 1000:+.0f} ms); lip sync cannot be judged reliably.", LIM, **ev))
        return out

    for s, e, ws in _merge(ok, "flag"):
        r_min = min(w["r_at_offset"] for w in ws)
        drift = [w for w in ws if w["drift"]]
        lags = sorted({w["best_lag_s"] for w in drift})
        what = []
        if any(w["drop"] for w in ws):
            what.append(f"correlation falls to r={r_min:.2f} (clip median {clip_r:.2f})")
        if drift:
            what.append(f"best offset moves to {', '.join(f'{l * 1000:+.0f}' for l in lags)} ms "
                        f"(clip {offset * 1000:+.0f} ms)")
        out.append(_f(Stance.supports_manipulation, round(min(0.85, 0.35 + 0.5 * max(0.0, clip_r - r_min) + 0.05 * len(ws)), 3),
                      0.5, f"{s:.2f}-{e:.2f}s: lips and audio stop matching - {'; '.join(what)}. Consistent with "
                      f"audio replaced or re-timed in this interval.", LIM, Segment(start=s, end=e, unit="s"),
                      coverage=round(min(1.0, (e - s) / dur), 4) if dur else None,
                      flagged_windows=[{k: w[k] for k in ("start", "end", "best_lag_s", "r_max", "r_at_offset")}
                                       for w in ws], **ev))
    # clean evidence only for the windows that actually showed a stable offset
    for s, e, ws in _merge(ok, "clean"):
        r_med = float(np.median([w["r_at_offset"] for w in ws]))
        out.append(_f(Stance.contradicts, round(min(0.6, r_med), 3), 0.45,
                      f"{s:.2f}-{e:.2f}s: lip movement matches speech in {len(ws)} window(s) (median r {r_med:.2f} at "
                      f"clip offset {offset * 1000:+.0f} ms, no drift > {DRIFT_S * 1000:.0f} ms).",
                      LIM + " Clean evidence covers only these windows.", Segment(start=s, end=e, unit="s"),
                      coverage=round(min(1.0, (e - s) / dur), 4) if dur else None, **ev))
    if abs(offset) > GLOBAL_OFFSET_S:
        out.append(_f(Stance.supports_manipulation, 0.35, 0.35,
                      f"Constant audio-video offset of {offset * 1000:+.0f} ms across the analysed windows (r "
                      f"{clip_r:.2f}): audio {'late' if offset > 0 else 'early'} relative to lips.",
                      LIM + " Constant offsets are common in careless re-muxing and are weak evidence.",
                      coverage=usable_cov, **ev))
    return out
