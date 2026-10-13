"""Minimal JPEG reader for image forensics: quantisation tables (natural order), SOF type, IJG-quality estimate,
and a double-compression indicator from DCT coefficient histograms. No third-party JPEG library needed."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ZIGZAG = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
                   13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
                   45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99])
SOF_NAMES = {0xC0: "baseline", 0xC1: "extended sequential", 0xC2: "progressive", 0xC3: "lossless"}
GAP_POSITIONS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (0, 3), (3, 0), (2, 2)]


@dataclass
class JpegInfo:
    tables: dict[int, np.ndarray] = field(default_factory=dict)  # table id -> 64 values, natural (row-major) order
    sof: Optional[str] = None
    app_markers: list[str] = field(default_factory=list)


def parse(path: Path) -> Optional[JpegInfo]:
    """None if the file is not a JPEG."""
    data = path.read_bytes()[: 4 * 1024 * 1024]
    if data[:2] != b"\xff\xd8":
        return None
    info = JpegInfo()
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of scan: headers are over
            break
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        seg = data[i + 4:i + 2 + length]
        if marker == 0xDB:
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 15
                n = 128 if pq else 64
                vals = (np.frombuffer(seg[j + 1:j + 1 + n], ">u2") if pq else np.frombuffer(seg[j + 1:j + 1 + n], np.uint8))
                nat = np.zeros(64, int)
                nat[ZIGZAG] = vals.astype(int)
                info.tables[tq] = nat
                j += 1 + n
        elif marker in SOF_NAMES:
            info.sof = SOF_NAMES[marker]
        elif 0xE0 <= marker <= 0xEF:
            tag = seg[:5].split(b"\x00")[0].decode("latin-1", "replace")
            info.app_markers.append(f"APP{marker - 0xE0}:{tag}")
        i += 2 + length
    return info


def ijg_table(q: int) -> np.ndarray:
    scale = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((STD_LUMA * scale + 50) / 100), 1, 255).astype(int)


def ijg_quality(luma: np.ndarray) -> tuple[int, bool]:
    """(closest IJG/libjpeg quality, True if the table is exactly the standard table at that quality)."""
    errs = [(float(np.abs(ijg_table(q) - luma).mean()), q) for q in range(1, 101)]
    err, q = min(errs)
    return q, err == 0.0


def double_jpeg_gap(bgr: np.ndarray, luma: np.ndarray) -> tuple[Optional[float], list[float]]:
    """Median 'gap' score over low-frequency DCT positions: ~0.1 for one compression, high when an earlier, coarser
    compression left periodic empty bins in the coefficient histograms. None when there are too few coefficients."""
    rgb = bgr[..., ::-1].astype(np.float32)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2] - 128
    h, w = y.shape[0] // 8 * 8, y.shape[1] // 8 * 8
    blocks = y[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    if len(blocks) > 60000:  # plenty for stable histograms
        blocks = blocks[np.linspace(0, len(blocks) - 1, 60000).astype(int)]
    dct = np.stack([cv2.dct(b) for b in blocks])
    q = luma.reshape(8, 8)
    scores = []
    for u, v in GAP_POSITIONS:
        c = np.abs(np.round(dct[:, u, v] / q[u, v])).astype(int)
        c = c[(c >= 1) & (c <= 40)]
        if c.size < 2000:
            continue
        hist = np.bincount(c, minlength=42)[1:41].astype(float)
        sc = [max(0.0, 1 - hist[k] / ((hist[k - 1] + hist[k + 1]) / 2)) for k in range(1, len(hist) - 1)
              if (hist[k - 1] + hist[k + 1]) / 2 >= 30]
        if len(sc) >= 4:
            scores.append(float(np.mean(sc)))
    return (float(np.median(scores)) if len(scores) >= 3 else None), [round(s, 3) for s in scores]


def resave(bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
