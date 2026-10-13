"""Screening pass: cheap per-modality suspicion curves over the whole clip.

Video (proxy + ingest frames): frame-difference isolation spikes (proxy, native rate, 160x120), noise-residual
region anomalies (2 fps ingest frames capped at 640 px: the proxy's compression erases sensor noise),
face-present flag when a detector is available.
Audio (proxy audio): spectral flux spikes, high-band energy drops below room tone, noise-floor jumps.
Each signal -> robust z-score per 0.25 s bin -> suspicion s = clip((z - Z0) / Z_SPAN, 0, 1); a modality
curve is the max over its signals. Screening never decides anything; it only tells the scheduler where to
look, and its findings are neutral (timeline only).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from backend.config import MAX_ANALYSIS_SECONDS
from backend.models import Depth, Finding, Modality, Segment, Stance

BIN_S = 0.25
Z0, Z_SPAN = 2.0, 6.0
FLAG_LEVEL = 0.35
SCREEN_MAX_W = 640
BLOCK = 16
AUDIO_SR = 32000
# Shown on the timeline but not used to schedule: speech onsets make spectral flux fire on ~40% of any
# talking clip (measured on the test set), so it carries no localisation value on its own.
DISPLAY_ONLY = {"face_present", "spectral_flux"}


def _robust_z(x: np.ndarray) -> np.ndarray:
    m = float(np.nanmedian(x))
    s = float(np.nanmedian(np.abs(x - m))) * 1.4826
    return (x - m) / max(s, 1e-6)


def _to_bins(times: np.ndarray, values: np.ndarray, n_bins: int, width_s: float = 0.0) -> np.ndarray:
    """Max-pool samples into bins; a sample covers [t, t + width_s) (its sampling period) when width_s > 0."""
    out = np.zeros(n_bins)
    span = max(1, int(round(width_s / BIN_S)))
    for k in range(span):
        idx = np.clip((times / BIN_S).astype(int) + k, 0, n_bins - 1)
        np.maximum.at(out, idx, values)
    return out


def _suspicion(z: np.ndarray) -> np.ndarray:
    return np.clip((z - Z0) / Z_SPAN, 0, 1)


# ---------------------------------------------------------------- video signals

def _frame_diff_signal(proxy: Path) -> tuple[np.ndarray, np.ndarray]:
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                       "stream=avg_frame_rate", "-of", "json", str(proxy)],
                                      capture_output=True, text=True, check=True).stdout)
    num, _, den = probe["streams"][0]["avg_frame_rate"].partition("/")
    fps = float(num) / float(den or 1) if float(num) else 25.0
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(proxy), "-t", str(MAX_ANALYSIS_SECONDS), "-map", "0:v:0",
                          "-vf", "scale=160:120,format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, timeout=300, check=True).stdout
    G = np.frombuffer(raw, np.uint8).reshape(-1, 120, 160).astype(np.float32)
    if len(G) < 4:
        return np.array([]), np.array([])
    S = np.stack([cv2.resize(cv2.GaussianBlur(g, (0, 0), 1.5), (80, 60), interpolation=cv2.INTER_AREA) for g in G])
    S -= S.mean((1, 2), keepdims=True)
    d = np.abs(np.diff(S, axis=0)).mean((1, 2))
    iso = np.empty(len(d))
    for i in range(len(d)):
        nb = max(d[j] for j in (i - 1, i + 1) if 0 <= j < len(d))
        lm = float(np.median(np.r_[d[max(0, i - 8):i], d[i + 1:i + 9]]))
        iso[i] = min(d[i] / max(nb, 1e-3), d[i] / max(lm, 1e-3))
    # isolation ~1 is normal; map ratio onto a z-like scale: 3x isolation -> z ~ 6
    return (np.arange(1, len(d) + 1) / fps), (iso - 1) * 3


def _noise_region_signal(frame_paths: list[Path], sample_fps: float) -> tuple[np.ndarray, np.ndarray]:
    maps = []
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[1] > SCREEN_MAX_W:
            img = cv2.resize(img, (SCREEN_MAX_W, int(img.shape[0] * SCREEN_MAX_W / img.shape[1])),
                             interpolation=cv2.INTER_AREA)
        r = img.astype(np.float32) - cv2.medianBlur(img, 3).astype(np.float32)
        h, w = r.shape
        m = np.sqrt((r * r)[: h // BLOCK * BLOCK, : w // BLOCK * BLOCK]
                    .reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK).mean((1, 3)))
        L = np.log(m + 1e-3)
        maps.append(L - np.median(L))
    if len(maps) < 5:
        return np.array([]), np.array([])
    M = np.stack(maps)
    med = np.median(M, 0)
    mad = np.median(np.abs(M - med), 0) * 1.4826
    Z = (M - med) / np.maximum(mad, 0.15)
    A = ((np.abs(Z) > 3.5) & (np.abs(M - med) > np.log(1.6))).astype(np.uint8)
    largest = []
    for a in A:
        n, _, st, _ = cv2.connectedComponentsWithStats(a, connectivity=8)
        largest.append(max((st[k, 4] for k in range(1, n)), default=0) / a.size)
    largest = np.array(largest)
    # >= 2% of the frame in one connected component maps to z ~ 6
    return np.arange(len(largest)) / sample_fps, largest / 0.02 * 6


def _face_signal(frame_paths: list[Path], sample_fps: float) -> Optional[tuple[np.ndarray, np.ndarray]]:
    cascade = os.path.join(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "",
                           "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade):
        return None
    det = cv2.CascadeClassifier(cascade)
    flags = []
    for p in frame_paths[::2]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        flags.append(1.0 if img is not None and len(det.detectMultiScale(img, 1.2, 5)) else 0.0)
    return np.arange(len(flags)) * 2 / sample_fps, np.array(flags)


# ---------------------------------------------------------------- audio signals

def _audio_signals(proxy: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(proxy), "-t", str(MAX_ANALYSIS_SECONDS), "-vn",
                          "-ac", "1", "-ar", str(AUDIO_SR), "-f", "f32le", "-"], capture_output=True, timeout=300).stdout
    x = np.frombuffer(raw, np.float32)
    N, hop = 1024, 256
    if len(x) < N * 8:
        return {}
    fr = np.lib.stride_tricks.sliding_window_view(x, N)[::hop] * np.hanning(N).astype(np.float32)
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-20
    f = np.fft.rfftfreq(N, 1 / AUDIO_SR)
    t = np.arange(len(P)) * hop / AUDIO_SR
    e = 10 * np.log10(P.sum(1))
    live = e > -170
    floor = float(np.percentile(e[live], 10)) if live.any() else -200.0
    active = e >= floor + 15
    pauses = (~active) & live
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    logP = np.log(P)
    flux = np.r_[0.0, np.maximum(np.diff(logP, axis=0), 0).mean(1)]
    out["spectral_flux"] = (t, _robust_z(flux))

    hb = (f > 6000) & (f < 12000)
    if pauses.sum() >= 10:
        hf = 10 * np.log10(P[:, hb].mean(1))
        drop = np.median(hf[pauses]) - hf
        out["hf_drop"] = (t, np.where(active, drop / 20 * 6, 0.0))  # 20 dB below room tone -> z ~ 6

    F = int(0.02 * AUDIO_SR)
    en = 10 * np.log10((x[: len(x) // F * F].reshape(-1, F).astype(np.float64) ** 2).mean(1) + 1e-20)
    per, step = 50, 12  # 1 s windows, 0.24 s hop
    if len(en) > per * 2:
        starts = np.arange(0, len(en) - per + 1, step)
        fl = np.array([np.percentile(en[s:s + per], 5) for s in starts])
        base = float(np.median(fl))
        jump = np.abs(fl - base)
        out["noise_floor"] = ((starts + per / 2) * 0.02, jump / 8 * 6)  # 8 dB step -> z ~ 6
    return out


# ---------------------------------------------------------------- curves + findings

def screen(proxy: Optional[Path], frame_paths: list[Path], sample_fps: Optional[float], duration: float,
           has_audio: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    n_bins = max(1, int(np.ceil(min(duration, MAX_ANALYSIS_SECONDS) / BIN_S)))
    signals: dict[str, dict[str, list[float]]] = {"video": {}, "audio": {}}
    notes: list[str] = []

    def add(mod: str, name: str, sig: tuple[np.ndarray, np.ndarray], width_s: float = 0.0) -> None:
        times, z = sig
        if len(times):
            signals[mod][name] = _suspicion(_to_bins(times, z, n_bins, width_s)).round(3).tolist()

    if proxy and proxy.suffix == ".mp4" and proxy.exists():
        add("video", "frame_diff", _frame_diff_signal(proxy))
        if has_audio:
            for name, sig in _audio_signals(proxy).items():
                add("audio", name, sig)
    if frame_paths and sample_fps:
        add("video", "noise_region", _noise_region_signal(frame_paths, sample_fps), width_s=1 / sample_fps)
        face = _face_signal(frame_paths, sample_fps)
        if face is None:
            notes.append("face-present flag unavailable (no face detector model in this OpenCV build)")
        else:
            signals["video"]["face_present"] = _to_bins(face[0], face[1], n_bins).tolist()

    curves = {}
    for mod, sigs in signals.items():
        scored = [np.array(v) for k, v in sigs.items() if k not in DISPLAY_ONLY]
        if scored:
            curves[mod] = np.max(np.stack(scored), 0).round(3).tolist()
    combined = np.max(np.stack([np.array(c) for c in curves.values()]), 0).round(3).tolist() if curves else []
    return {"bin_s": BIN_S, "n_bins": n_bins, "signals": signals, "curves": curves, "combined": combined,
            "flag_level": FLAG_LEVEL, "notes": notes, "elapsed_s": round(time.perf_counter() - t0, 3)}


def screening_findings(scr: dict[str, Any]) -> list[Finding]:
    """Neutral, depth=screen findings for flagged stretches (timeline only; never used as evidence)."""
    out = []
    for mod, curve in scr["curves"].items():
        c = np.array(curve)
        on = c >= FLAG_LEVEL
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(on) and on[j + 1]:
                j += 1
            sigs = [k for k, v in scr["signals"][mod].items() if k not in DISPLAY_ONLY and max(v[i:j + 1]) >= FLAG_LEVEL]
            out.append(Finding(
                modality=Modality.visual if mod == "video" else Modality.audio, check=f"screen.{mod}",
                depth=Depth.screen, stance=Stance.neutral, score=float(c[i:j + 1].max()), reliability=0.2,
                segment=Segment(start=round(i * BIN_S, 3), end=round((j + 1) * BIN_S, 3), unit="s"),
                coverage=1.0,
                rationale=f"Screening flagged {mod} at {i * BIN_S:.2f}-{(j + 1) * BIN_S:.2f}s "
                          f"(peak suspicion {c[i:j + 1].max():.2f}; signals: {', '.join(sigs)}). Sent to deep analysis.",
                limitations="Screening signals are cheap and noisy; they direct deep checks and carry no weight in the verdict.",
                evidence={"signals": sigs, "peak": float(c[i:j + 1].max())}))
            i = j + 1
    return out
