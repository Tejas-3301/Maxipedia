"""Media quality: is the file degraded enough that the absence of forensic traces means little?

Neutral finding (never evidence for or against manipulation); fusion uses `evidence.degraded` to refuse a
"likely authentic" verdict on heavily compressed / noisy / tiny media, because recompression and noise erase the
traces the other checks look for (measured in attack mode).
Signals: x264 crf from the bitstream SEI (>= 30 = heavy compression; standard x264 quality scale), Immerkaer
noise sigma (>= 5 grey levels = heavy noise; typical sensor noise 1-4), bits per pixel only when no crf is known
(< 0.01), frame height < 240 px.
"""
from __future__ import annotations

import json
import re
import subprocess

import cv2
import numpy as np

from backend.checks.registry import JobContext, register
from backend.models import Finding, Modality, Stance

CRF_HEAVY = 30.0
NOISE_HEAVY = 5.0
BPP_MIN = 0.01
MIN_HEIGHT = 240
JPEG_Q_HEAVY = 60  # stills only
_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer (1996) fast noise variance estimate."""
    h, w = gray.shape
    conv = np.abs(cv2.filter2D(gray.astype(np.float32), -1, _K)[1:-1, 1:-1]).sum()
    return float(np.sqrt(np.pi / 2) * conv / (6 * (w - 2) * (h - 2)))


@register("quality", Modality.metadata, ("video", "image"))
def quality_check(ctx: JobContext) -> list[Finding]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate,bit_rate:format=bit_rate", "-of", "json",
                          str(ctx.media_path)], capture_output=True, text=True, timeout=60, check=True).stdout
    probe = json.loads(out)
    s = (probe.get("streams") or [{}])[0]
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    num, _, den = str(s.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) and float(num or 0) else 0.0
    rate = s.get("bit_rate") or (probe.get("format") or {}).get("bit_rate")
    bpp = float(rate) / (w * h * fps) if rate and w and h and fps else None
    with open(ctx.media_path, "rb") as f:
        m = re.search(rb"x264 - core .{0,400}?crf=([\d.]+)", f.read(4_000_000), re.S)
    crf = float(m.group(1)) if m else None
    frames = ctx.frame_paths()[:: max(1, len(ctx.frame_paths()) // 12)]
    sigmas = [noise_sigma(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in frames]
    sigma = float(np.median(sigmas)) if sigmas else None

    reasons = []
    if crf is not None and crf >= CRF_HEAVY:
        reasons.append(f"heavy compression (x264 crf={crf:g} >= {CRF_HEAVY:g})")
    if crf is None and bpp is not None and bpp < BPP_MIN:
        reasons.append(f"very low bitrate ({bpp:.4f} bits/pixel < {BPP_MIN})")
    if sigma is not None and sigma >= NOISE_HEAVY:
        reasons.append(f"heavy noise (sigma {sigma:.1f} >= {NOISE_HEAVY:g} grey levels)")
    if h and h < MIN_HEIGHT:
        reasons.append(f"low resolution ({w}x{h})")
    jpeg_q = None
    if ctx.media_type == "image":  # stills: JPEG quality from the file's own quantisation table
        from backend.checks import _jpeg

        info = _jpeg.parse(ctx.media_path)
        if info is not None and 0 in info.tables:
            jpeg_q = _jpeg.ijg_quality(info.tables[0])[0]
            if jpeg_q < JPEG_Q_HEAVY:
                reasons.append(f"heavy JPEG compression (estimated quality {jpeg_q} < {JPEG_Q_HEAVY})")
    ev = {"width": w, "height": h, "bits_per_pixel": None if bpp is None else round(bpp, 4), "x264_crf": crf,
          "jpeg_quality": jpeg_q,
          "noise_sigma": None if sigma is None else round(sigma, 2), "degraded": bool(reasons), "reasons": reasons}
    text = ("Media quality is degraded: " + "; ".join(reasons) + ". Recompression, noise and downscaling erase the "
            "traces the forensic checks rely on, so missing evidence cannot vouch for authenticity."
            if reasons else
            f"Media quality adequate for forensic checks ({w}x{h}, crf {crf if crf is not None else 'n/a'}, "
            f"noise sigma {ev['noise_sigma']}).")
    return [Finding(modality=Modality.metadata, check="quality.media", stance=Stance.neutral, score=0.0,
                    reliability=0.5, rationale=text, coverage=1.0,
                    limitations="Quality is not evidence of manipulation either way; thresholds follow common x264 "
                                "quality levels and typical sensor noise.", evidence=ev)]
