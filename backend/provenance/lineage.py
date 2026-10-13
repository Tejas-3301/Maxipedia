"""Perceptual lineage: near-duplicate detection, time alignment (trims, cuts), crop handling, graph.

Identity uses per-frame 64-bit pHash (robust to re-encode/scale/brightness). Alignment uses normalised
10 fps thumbnails and a dynamic program over time offsets that keeps the offset constant unless the content
clearly jumps (a cut or insertion), searching a few crop factors so cropped copies still align.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from backend.provenance.fingerprint import THUMB_FPS, hamming

NEAR_DUP_BITS = 16  # unrelated frames sit at ~28-32 of 64 bits
MIN_MATCH_FRAC = 0.6
CROPS = (1.0, 0.9, 0.8)
UNMATCHED_COST = 0.6


@dataclass
class Relation:
    other: str  # job_id
    median_bits: float
    match_frac: float
    crop: float  # >1: this shows a sub-area of other; <1: other is a crop of this
    start_offset_s: float  # this t=0 corresponds to other t=start_offset_s
    offset_changes: list[dict] = field(default_factory=list)  # [{at_s, removed_s|inserted_s}]
    coverage: tuple[float, float] = (0.0, 0.0)  # span of `other` that this copy covers (s)
    align_cost: float = 0.0


def phash_similarity(a: list[int], b: list[int]) -> tuple[float, float]:
    """(median best-match Hamming distance a->b, fraction of a's frames with a near-duplicate in b)."""
    if not a or not b:
        return 64.0, 0.0
    step = max(1, len(a) // 60)
    best = [min(hamming(x, y) for y in b) for x in a[::step]]
    return float(np.median(best)), float(np.mean([d <= NEAR_DUP_BITS for d in best]))


def _norm(T: np.ndarray, frac: float) -> np.ndarray:
    h, w = T.shape[1:]
    mh, mw = int(h * (1 - frac) / 2), int(w * (1 - frac) / 2)
    C = T[:, mh:h - mh, mw:w - mw].astype(np.float32)
    C = np.stack([cv2.GaussianBlur(cv2.resize(c, (64, 48), interpolation=cv2.INTER_AREA), (0, 0), 1) for c in C])
    C = (C - C.mean((1, 2), keepdims=True)) / (C.std((1, 2), keepdims=True) + 1e-6)
    return C.reshape(len(C), -1)


def _dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.abs(A[:, None, :] - B[None, :, :]).mean(2)


def _dp(D: np.ndarray) -> tuple[np.ndarray, float]:
    """Monotone piecewise-constant offset path: off[i] = i - j(i); returns offsets and mean matched cost."""
    nA, nB = D.shape
    offs = np.arange(-(nB - 1), nA)
    lam = 3 * float(np.median(np.abs(np.diff(D.min(1))))) + 0.05
    cost = np.empty((nA, len(offs)))
    back = np.zeros((nA, len(offs)), int)
    for i in range(nA):
        j = i - offs
        valid = (j >= 0) & (j < nB)
        local = np.where(valid, D[i, np.clip(j, 0, nB - 1)], UNMATCHED_COST)
        if i == 0:
            cost[0] = local
            continue
        prev = cost[i - 1]
        best = int(prev.argmin())
        stay_better = prev <= prev[best] + lam
        cost[i] = np.where(stay_better, prev, prev[best] + lam) + local
        back[i] = np.where(stay_better, np.arange(len(offs)), best)
    k = int(cost[-1].argmin())
    path = [k]
    for i in range(nA - 1, 0, -1):
        k = back[i, k]
        path.append(k)
    off = offs[np.array(path[::-1])]
    j = np.arange(nA) - off
    ok = (j >= 0) & (j < nB)
    return off, float(D[np.arange(nA)[ok], j[ok]].mean()) if ok.any() else UNMATCHED_COST


def align(this: np.ndarray, other: np.ndarray, max_frames: int = 600) -> tuple[np.ndarray, float, float]:
    """Returns (offset per frame of `this`, mean cost, crop factor)."""
    this, other = this[:max_frames], other[:max_frames]
    best = None
    for c in CROPS:
        for a_frac, b_frac, crop in ((0.8, 0.8 * c, 1 / c), (0.8 * c, 0.8, c)):
            if c == 1.0 and crop != 1.0:
                continue
            D = _dist(_norm(this, a_frac), _norm(other, b_frac))
            off, cost = _dp(D)
            if best is None or cost < best[1]:
                best = (off, cost, crop)
    return best


def relate(this_id: str, this_ph: list[int], this_thumbs: Optional[np.ndarray], other_id: str,
           other_ph: list[int], other_thumbs: Optional[np.ndarray]) -> Optional[Relation]:
    med, frac = phash_similarity(this_ph, other_ph)
    if frac < MIN_MATCH_FRAC:
        return None
    rel = Relation(other_id, med, frac, 1.0, 0.0)
    if this_thumbs is None or other_thumbs is None:
        return rel
    off, cost, crop = align(this_thumbs, other_thumbs)
    rel.crop, rel.align_cost = round(crop, 3), round(cost, 4)
    rel.start_offset_s = round(-int(off[0]) / THUMB_FPS, 2)
    for i in range(1, len(off)):
        if off[i] != off[i - 1]:
            delta = int(off[i - 1] - off[i]) / THUMB_FPS  # >0: parent content skipped (removed)
            key = "removed_s" if delta > 0 else "inserted_s"
            rel.offset_changes.append({"at_s": round(i / THUMB_FPS, 2), key: round(abs(delta), 2)})
    j = np.arange(len(off)) - off
    j = j[(j >= 0) & (j < len(other_thumbs))]
    if len(j):
        rel.coverage = (round(float(j.min()) / THUMB_FPS, 2), round(float(j.max() + 1) / THUMB_FPS, 2))
    return rel


def parent_score(child: dict, parent: dict, rel: Relation) -> float:
    """Higher = more plausible that `parent` is the source of `child` (resolution, duration, trust)."""
    s = -rel.median_bits
    cw, pw = child.get("width") or 0, parent.get("width") or 0
    cd, pd = child.get("duration_s") or 0, parent.get("duration_s") or 0
    if pw > cw:
        s += 4
    if pd > cd + 0.2:
        s += 4
    if rel.crop > 1.0:
        s += 4  # child shows a sub-area of parent
    if parent.get("trusted"):
        s += 2
    if pw and cw and abs(pw - cw) < 1 and abs(pd - cd) < 0.2:
        s -= 2  # same geometry/length: direction unknown
    return s


def is_ancestor_like(child: dict, parent: dict, rel: Relation) -> bool:
    cw, pw = child.get("width") or 0, parent.get("width") or 0
    cd, pd = child.get("duration_s") or 0, parent.get("duration_s") or 0
    return bool(parent.get("trusted")) or pw > cw or pd > cd + 0.2 or rel.crop > 1.0
