"""Audio forensics (thin): spectrogram, high-frequency cutoff, spectral flatness, noise-floor / room-tone splices.

Works on the original's first audio stream at its native sample rate (the 16 kHz analysis wav cannot show
cutoffs above 8 kHz), peak-normalised so levels are dB re peak. Everything is relative to the clip's own
baseline: a segment is suspicious when it differs from the rest of the same recording.
Spectrogram overview + per-finding crops go to <work_dir>/spectrograms/.
score = strength for stated stance; reliability = trust in the method.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from backend.checks.registry import JobContext, register
from backend.config import MAX_ANALYSIS_SECONDS
from backend.models import Depth, Finding, Modality, Segment, Stance

WIN_S, HOP_S = 0.5, 0.25  # spectral windows
FLOOR_WIN_S = 1.0  # noise-floor windows (long enough to contain pauses in continuous speech)
FLOOR_FRAME_S = 0.02
HF_DROP_DB = 20.0
FLOOR_STEP_DB = 8.0
DIGITAL_SILENCE_DB = -100.0
FLAT_RATIO = 2.5
FLAT_Z = 3.5
ACTIVE_DB = 15.0  # speech-active frame: >= floor + 15 dB


# ---------------------------------------------------------------- helpers

def _py(v):
    if isinstance(v, dict):
        return {k: _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    return v.item() if isinstance(v, np.generic) else v


def _finding(check: str, stance: Stance, score: Optional[float], reliability: float, rationale: str,
             limitations: str, segment: Optional[Segment] = None, **evidence) -> Finding:
    return Finding(modality=Modality.audio, check=check, depth=Depth.screen, segment=segment, stance=stance,
                   score=None if score is None else float(score), reliability=float(reliability),
                   rationale=rationale, limitations=limitations, evidence=_py(evidence))


def _na(check: str, why: str) -> Finding:
    return _finding(check, Stance.not_assessable, None, 0.0, why, why)


def _db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(x, 1e-20))


def _segments(mask: np.ndarray, starts: np.ndarray, win: float, min_len: float,
              merge_gap: float = 0.0) -> list[tuple[int, int, float, float]]:
    """Merge consecutive True windows (and runs closer than merge_gap seconds) -> (first_idx, last_idx, t0, t1)."""
    runs: list[list[int]] = []
    for i in np.nonzero(mask)[0]:
        if runs and (i == runs[-1][1] + 1 or starts[i] - (starts[runs[-1][1]] + win) <= merge_gap):
            runs[-1][1] = int(i)
        else:
            runs.append([int(i), int(i)])
    out = []
    for i, j in runs:
        t0, t1 = float(starts[i]), float(starts[j] + win)
        if t1 - t0 >= min_len:
            out.append((i, j, t0, t1))
    return out


# ---------------------------------------------------------------- audio + spectrogram

@dataclass
class Audio:
    x: np.ndarray
    sr: int
    P: np.ndarray  # power spectrogram (frames, bins)
    f: np.ndarray
    hop: int
    path: Path

    @property
    def duration(self) -> float:
        return len(self.x) / self.sr

    def frame_t(self, i: int) -> float:
        return i * self.hop / self.sr


def load_audio(path: Path) -> Optional[Audio]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, timeout=60, check=True).stdout
    streams = json.loads(out).get("streams") or []
    if not streams:
        return None
    sr = int(streams[0]["sample_rate"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-t", str(MAX_ANALYSIS_SECONDS), "-map", "0:a:0",
                          "-ac", "1", "-f", "f32le", "-"], capture_output=True, timeout=300, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(np.float32)
    n_fft = 1 << int(np.ceil(np.log2(0.046 * sr)))
    if len(x) < n_fft * 4:
        return None
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak
    hop = n_fft // 4
    frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft).astype(np.float32)
    P = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).astype(np.float32)
    return Audio(x, sr, P, np.fft.rfftfreq(n_fft, 1 / sr), hop, path)


def render_spectrogram(a: Audio, out: Path, t0: float = 0.0, t1: Optional[float] = None,
                       marks: tuple[float, ...] = (), label: str = "", px_per_s: int = 120, height: int = 256) -> str:
    t1 = a.duration if t1 is None else t1
    i0, i1 = int(t0 * a.sr / a.hop), max(int(t0 * a.sr / a.hop) + 2, int(t1 * a.sr / a.hop))
    S = _db(a.P[i0:i1].T)
    top = float(np.percentile(_db(a.P), 99.9))
    img = np.clip((S - (top - 100)) / 100, 0, 1)
    img = (img[::-1] * 255).astype(np.uint8)  # low frequencies at the bottom
    w = max(64, int((t1 - t0) * px_per_s))
    img = cv2.applyColorMap(cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA), cv2.COLORMAP_INFERNO)
    nyq = a.sr / 2
    for khz in range(5, int(nyq / 1000) + 1, 5):
        y = int(height * (1 - khz * 1000 / nyq))
        cv2.line(img, (0, y), (w, y), (90, 90, 90), 1)
        cv2.putText(img, f"{khz}k", (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    for m in marks:
        x = int((m - t0) * px_per_s)
        if 0 <= x < w:
            cv2.line(img, (x, 0), (x, height), (255, 255, 255), 1)
    if label:
        cv2.putText(img, label, (30, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    return str(out)


def _crop(a: Audio, ctx: JobContext, name: str, k: int, t0: float, t1: float, label: str) -> str:
    return render_spectrogram(a, ctx.work_dir / "spectrograms" / f"{name}_event{k}.png",
                              max(0.0, t0 - 0.75), min(a.duration, t1 + 0.75), (t0, t1), label)


@dataclass
class Windows:
    starts: np.ndarray
    spectra: np.ndarray  # mean power per window (windows, bins), active frames only (falls back to all)
    active: np.ndarray  # bool per window: contains speech-like activity
    flatness: np.ndarray  # median log10 spectral flatness of active frames (nan if none)


def _frame_floor(a: Audio) -> float:
    """10th percentile of frame energy, ignoring digital silence (gaps would otherwise drag it to -200 dB)."""
    e = _db(a.P.sum(1))
    e = e[e > DIGITAL_SILENCE_DB + 30]
    return float(np.percentile(e, 10)) if len(e) else DIGITAL_SILENCE_DB


def windows(a: Audio) -> Windows:
    per_win = max(2, int(WIN_S * a.sr / a.hop))
    step = max(1, int(HOP_S * a.sr / a.hop))
    e = _db(a.P.sum(1))
    act = e >= _frame_floor(a) + ACTIVE_DB
    band = (a.f > 100) & (a.f < min(8000, a.sr / 2 * 0.95))
    logP = np.log(np.maximum(a.P[:, band], 1e-20))
    flat_frames = np.exp(logP.mean(1)) / np.maximum(a.P[:, band].mean(1), 1e-20)
    starts, spectra, active, flat = [], [], [], []
    for s in range(0, max(1, len(a.P) - per_win + 1), step):
        sl = slice(s, s + per_win)
        m = act[sl]
        starts.append(a.frame_t(s))
        spectra.append(a.P[sl][m].mean(0) if m.sum() >= 2 else a.P[sl].mean(0))
        active.append(m.mean() >= 0.3)
        flat.append(float(np.median(np.log10(np.maximum(flat_frames[sl][m], 1e-12)))) if m.sum() >= 2 else np.nan)
    return Windows(np.array(starts), np.array(spectra), np.array(active), np.array(flat))


# ---------------------------------------------------------------- high-frequency cutoff

def _cutoff(level_db: np.ndarray, f: np.ndarray, ref_db: float, margin: float = 40.0) -> float:
    """Highest frequency whose (smoothed) level is within `margin` dB of the 1-4 kHz reference."""
    k = max(1, len(level_db) // 100)
    sm = np.convolve(level_db, np.ones(k) / k, mode="same")
    idx = np.nonzero(sm > ref_db - margin)[0]
    return float(f[idx.max()]) if len(idx) else 0.0


def _band_edge(ref_db: np.ndarray, seg_db: np.ndarray, f: np.ndarray, thresh: float = HF_DROP_DB) -> float:
    """Lowest frequency above which the segment stays >= thresh dB below the reference for >= 80% of the bins
    that the reference actually occupies (bins above the codec's own low-pass are ignored)."""
    k = max(1, len(f) // 100)
    deficit = np.convolve(ref_db - seg_db, np.ones(k) / k, mode="same")
    valid = (ref_db > np.percentile(ref_db, 5) + 10) & (f > 500)
    idx = np.nonzero(valid)[0]
    for n, i in enumerate(idx):
        if np.mean(deficit[idx[n:]] > thresh) >= 0.8:
            return float(f[i])
    return float(f[idx.max()]) if len(idx) else 0.0


def check_hf_cutoff(a: Audio, w: Windows, ctx: JobContext, overview: str) -> list[Finding]:
    name = "audio.hf_cutoff"
    lim = ("Low-bitrate codecs, telephone/VoIP paths and some microphones also band-limit audio; "
           "a cutoff that changes mid-clip is the stronger cue.")
    ltas = _db(a.P.mean(0))
    mid = (a.f > 1000) & (a.f < 4000)
    clip_cut = _cutoff(ltas, a.f, float(np.median(ltas[mid])))
    nyq = a.sr / 2
    hi_band = (a.f > 0.55 * clip_cut) & (a.f < 0.95 * clip_cut)
    if hi_band.sum() < 4 or not w.active.any():
        return [_na(name, "No high band or no active audio to compare.")]

    # Room tone is additive, so genuine audio never dips below the room-tone level in the high band, even in
    # vowels. Inserted band-limited speech (TTS/vocoder) carries no room tone there and falls far below it.
    frame_hf = _db(a.P[:, hi_band].mean(1))
    frame_e = _db(a.P.sum(1))
    pauses = (frame_e < _frame_floor(a) + ACTIVE_DB) & (frame_e > DIGITAL_SILENCE_DB)
    hf_floor = float(np.median(frame_hf[pauses])) if pauses.sum() >= 10 else float(np.percentile(frame_hf, 10))
    per_win = max(2, int(WIN_S * a.sr / a.hop))
    hf = np.array([float(np.median(frame_hf[int(t * a.sr / a.hop):int(t * a.sr / a.hop) + per_win]))
                   for t in w.starts])
    flat_base = float(np.nanmedian(w.flatness)) if np.isfinite(w.flatness).any() else float("nan")
    no_room_tone = pauses.sum() < 10  # pauses are digital silence: no room-tone reference to compare against
    drop = (hf_floor - hf > HF_DROP_DB) & w.active if not no_room_tone else np.zeros(len(hf), bool)
    out: list[Finding] = []
    pause_spec = _db(a.P[pauses].mean(0)) if pauses.sum() >= 10 else ltas
    for k, (i, j, t0, t1) in enumerate(_segments(drop, w.starts, WIN_S, WIN_S, merge_gap=0.75)):
        seg_frames = slice(int(t0 * a.sr / a.hop), int(t1 * a.sr / a.hop))
        act = frame_e[seg_frames] >= _frame_floor(a) + ACTIVE_DB
        seg_spec = _db(a.P[seg_frames][act].mean(0)) if act.sum() >= 2 else _db(a.P[seg_frames].mean(0))
        seg_cut = _band_edge(pause_spec, seg_spec, a.f)
        seg_flat = float(np.nanmedian(w.flatness[i:j + 1])) if np.isfinite(w.flatness[i:j + 1]).any() else None
        deficit = float(hf_floor - np.median(hf[i:j + 1]))
        crop = _crop(a, ctx, name, k, t0, t1, f"band edge ~{seg_cut / 1000:.1f}k vs clip {clip_cut / 1000:.1f}k")
        score = round(min(0.85, 0.4 + 0.01 * (deficit - HF_DROP_DB) + 0.05 * min(4.0, t1 - t0)), 3)
        out.append(_finding(
            name, Stance.supports_manipulation, score, 0.55,
            f"{t0:.2f}-{t1:.2f}s: from ~{seg_cut / 1000:.1f} kHz upward the spectrum stays >= {HF_DROP_DB:.0f} dB "
            f"below the clip's room-tone spectrum (high band "
            f"{0.55 * clip_cut / 1000:.1f}-{0.95 * clip_cut / 1000:.1f} kHz sits {deficit:.0f} dB below the clip's "
            f"room-tone level in that band), while the rest of the clip extends to ~{clip_cut / 1000:.1f} kHz. A band-limited "
            f"segment inside full-band audio is consistent with inserted TTS/vocoder or low-rate speech."
            + (f" Spectral flatness {10 ** seg_flat:.4f} vs clip {10 ** flat_base:.4f}." if seg_flat is not None
               and np.isfinite(flat_base) else ""),
            lim, Segment(start=round(t0, 3), end=round(t1, 3), unit="s"),
            segment_cutoff_hz=round(seg_cut), clip_cutoff_hz=round(clip_cut), sample_rate=a.sr,
            hf_band_deficit_db=round(deficit, 1), segment_flatness=None if seg_flat is None else round(10 ** seg_flat, 5),
            clip_flatness=None if not np.isfinite(flat_base) else round(10 ** flat_base, 5),
            spectrogram=crop, overview=overview))

    if not out and clip_cut < 0.45 * nyq and nyq > 11000:
        out.append(_finding(
            name, Stance.supports_manipulation, 0.3, 0.35,
            f"Whole clip is band-limited to ~{clip_cut / 1000:.1f} kHz although stored at {a.sr} Hz: audio was "
            f"produced at a lower rate and upsampled (TTS, telephone, or low-bitrate source).", lim,
            clip_cutoff_hz=round(clip_cut), sample_rate=a.sr, overview=overview))
    if not out:
        out.append(_finding(name, Stance.contradicts, 0.2, 0.35,
                            f"Bandwidth is consistent across the clip (~{clip_cut / 1000:.1f} kHz at {a.sr} Hz; "
                            f"no active segment drops more than {HF_DROP_DB:.0f} dB below the high-band room-tone level).",
                            lim, clip_cutoff_hz=round(clip_cut), sample_rate=a.sr, overview=overview))
    return out


# ---------------------------------------------------------------- spectral flatness

def check_flatness(a: Audio, w: Windows, ctx: JobContext, overview: str) -> list[Finding]:
    name = "audio.spectral_flatness"
    lim = ("Flatness depends heavily on what is said (fricatives are noise-like, vowels tonal); only long, "
           "strong departures from the speaker's own baseline are reported.")
    ok = np.isfinite(w.flatness)
    if ok.sum() < 6:
        return [_na(name, "Too little active audio for a flatness baseline.")]
    med = float(np.median(w.flatness[ok]))
    mad = float(np.median(np.abs(w.flatness[ok] - med))) * 1.4826
    z = np.where(ok, (w.flatness - med) / max(mad, 0.05), 0.0)
    dev = ok & (np.abs(z) > FLAT_Z) & (np.abs(w.flatness - med) > np.log10(FLAT_RATIO))
    out: list[Finding] = []
    for k, (i, j, t0, t1) in enumerate(_segments(dev, w.starts, WIN_S, 1.0)):
        seg = float(np.median(w.flatness[i:j + 1][ok[i:j + 1]]))
        crop = _crop(a, ctx, name, k, t0, t1, f"flatness {10 ** seg:.4f} vs {10 ** med:.4f}")
        out.append(_finding(
            name, Stance.supports_manipulation, round(min(0.6, 0.3 + 0.03 * float(np.abs(z[i:j + 1]).max())), 3), 0.25,
            f"{t0:.2f}-{t1:.2f}s: spectral flatness of active audio is {10 ** seg:.4f} vs the clip's {10 ** med:.4f} "
            f"(x{10 ** abs(seg - med):.1f}, max |z| {float(np.abs(z[i:j + 1]).max()):.1f}); the timbre/noise "
            f"structure differs from the rest of the recording.", lim,
            Segment(start=round(t0, 3), end=round(t1, 3), unit="s"),
            segment_flatness=round(10 ** seg, 5), clip_flatness=round(10 ** med, 5), spectrogram=crop,
            overview=overview))
    if not out:
        out.append(_finding(name, Stance.neutral, 0.0, 0.25,
                            f"Spectral flatness of active audio stays near the clip baseline {10 ** med:.4f} "
                            f"(robust spread x{10 ** mad:.2f}).", lim, clip_flatness=round(10 ** med, 5),
                            overview=overview))
    return out


# ---------------------------------------------------------------- noise floor / room tone

def check_noise_floor(a: Audio, w: Windows, ctx: JobContext, overview: str) -> list[Finding]:
    name = "audio.noise_floor"
    lim = ("Noise gates, automatic gain control, mic changes or moving between rooms also shift the floor; "
           "joins between two takes of the same room tone are not detectable this way.")
    F = int(FLOOR_FRAME_S * a.sr)
    e = _db((a.x[: len(a.x) // F * F].reshape(-1, F).astype(np.float64) ** 2).mean(1))
    per, step = int(FLOOR_WIN_S / FLOOR_FRAME_S), int(HOP_S / FLOOR_FRAME_S)
    if len(e) < per * 2:
        return [_na(name, "Clip too short for noise-floor tracking.")]
    starts = np.arange(0, len(e) - per + 1, step)
    floor = np.array([np.percentile(e[s:s + per], 5) for s in starts])
    t_starts = starts * FLOOR_FRAME_S
    base = float(np.median(floor))
    if base < DIGITAL_SILENCE_DB:
        return [_finding(name, Stance.neutral, 0.0, 0.3,
                         f"No room tone: the noise floor is digital silence ({base:.0f} dB re peak) for most of the "
                         f"clip (studio, noise-gated or synthetic audio), so floor continuity cannot be assessed.", lim,
                         floor_db_median=round(base, 1), overview=overview)]
    spread = float(np.median(np.abs(floor - base)))
    silent = floor < DIGITAL_SILENCE_DB
    step_mask = (np.abs(floor - base) > max(FLOOR_STEP_DB, 4 * spread)) | silent
    out: list[Finding] = []
    for k, (i, j, _, _) in enumerate(_segments(step_mask, t_starts, FLOOR_WIN_S, FLOOR_WIN_S)):
        # Window-resolution bounds are +-0.5 s; refine with a short running floor.
        fine = np.array([np.percentile(e[max(0, n - 7):n + 8], 5) for n in range(len(e))])
        lo, hi = int(t_starts[i] / FLOOR_FRAME_S), min(len(e) - 1, int((t_starts[j] + FLOOR_WIN_S) / FLOOR_FRAME_S))
        bad = np.nonzero(np.abs(fine[lo:hi + 1] - base) > FLOOR_STEP_DB)[0]
        t0 = (lo + bad[0]) * FLOOR_FRAME_S if len(bad) else float(t_starts[i])
        t1 = (lo + bad[-1] + 1) * FLOOR_FRAME_S if len(bad) else float(t_starts[j] + FLOOR_WIN_S)
        seg_floor = float(np.median(floor[i:j + 1]))
        has_silence = bool(silent[i:j + 1].any())
        delta = seg_floor - base
        score = 0.75 if has_silence else min(0.8, 0.4 + 0.02 * abs(delta))
        what = ("drops to digital silence" if has_silence
                else f"{'rises' if delta > 0 else 'falls'} by {abs(delta):.0f} dB")
        crop = _crop(a, ctx, name, k, t0, t1, f"floor {what}")
        out.append(_finding(
            name, Stance.supports_manipulation, round(score, 3), 0.55,
            f"{t0:.2f}-{t1:.2f}s: background noise floor {what} (segment median {seg_floor:.0f} dB vs clip room "
            f"tone {base:.0f} dB re peak, normal spread +-{spread:.1f} dB). A room-tone discontinuity with splice "
            f"points near {t0:.2f}s and {t1:.2f}s is consistent with inserted or replaced audio.", lim,
            Segment(start=round(t0, 3), end=round(t1, 3), unit="s"),
            splice_points_s=[round(t0, 3), round(t1, 3)], floor_db_segment=round(seg_floor, 1),
            floor_db_clip=round(base, 1), digital_silence=has_silence, spectrogram=crop, overview=overview))
    if not out:
        out.append(_finding(name, Stance.contradicts, 0.25, 0.4,
                            f"Room tone is continuous at {base:.0f} dB re peak (+-{spread:.1f} dB) across "
                            f"{a.duration:.1f}s; no floor steps or dropouts.", lim,
                            floor_db_clip=round(base, 1), overview=overview))
    return out


# ---------------------------------------------------------------- entry point

SUBCHECKS: list[tuple[str, Callable[[Audio, Windows, JobContext, str], list[Finding]]]] = [
    ("audio.hf_cutoff", check_hf_cutoff),
    ("audio.spectral_flatness", check_flatness),
    ("audio.noise_floor", check_noise_floor),
]


@register("audio", Modality.audio, ("video", "audio"))
def audio_check(ctx: JobContext) -> list[Finding]:
    a = load_audio(ctx.media_path)
    if a is None:
        return [_na(n, "No usable audio stream.") for n, _ in SUBCHECKS]
    w = windows(a)
    overview = render_spectrogram(a, ctx.work_dir / "spectrograms" / "overview.png", label="overview",
                                  px_per_s=max(20, min(120, int(2400 / max(a.duration, 1)))))
    out: list[Finding] = []
    for name, fn in SUBCHECKS:
        try:
            out.extend(fn(a, w, ctx, overview))
        except Exception as exc:
            out.append(_na(name, f"Sub-check failed: {type(exc).__name__}: {exc}"))
    return out
