"""Content fingerprints of one media file.

- Packet leaves: per stream, per 1 s segment, SHA-256 over the compressed packet hashes (ffmpeg framehash,
  stream copy, no decode). Independent of container metadata, so a re-mux keeps every leaf; any re-encode
  or edit of a segment changes that leaf.
- Merkle tree over the leaves (sorted by stream, then segment) -> root + inclusion proofs.
- Visual signature: 10 fps grayscale thumbnails (for alignment) and 64-bit DCT pHash per frame (identity).
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.config import MAX_ANALYSIS_SECONDS

SEGMENT_S = 1.0
THUMB_FPS = 10
THUMB_W, THUMB_H = 96, 72


@dataclass
class Leaf:
    stream: int
    kind: str  # video | audio | other
    segment: int
    start: float
    end: float
    packets: int
    hash: str


def packet_hashes(path: Path) -> tuple[dict[int, str], list[tuple[int, float, float, str]]]:
    """Returns ({stream: media_type}, [(stream, start_s, dur_s, sha256), ...]) for every packet."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0", "-c", "copy",
                          "-f", "framehash", "-hash", "sha256", "-"],
                         capture_output=True, text=True, timeout=600, check=True).stdout
    tb: dict[int, float] = {}
    kinds: dict[int, str] = {}
    packets = []
    for line in out.splitlines():
        if line.startswith("#tb "):
            s, frac = line[4:].split(":")
            num, den = frac.strip().split("/")
            tb[int(s)] = int(num) / int(den)
        elif line.startswith("#media_type "):
            s, kind = line[12:].split(":")
            kinds[int(s)] = kind.strip()
        elif line and not line.startswith("#"):
            parts = [p.strip() for p in line.split(",")]
            s = int(parts[0])
            pts, dur = int(parts[2]), int(parts[3])
            t = tb.get(s, 1.0)
            packets.append((s, pts * t, dur * t, parts[5].split("=")[-1]))
    return kinds, packets


def leaves_from_packets(kinds: dict[int, str], packets: list[tuple[int, float, float, str]]) -> list[Leaf]:
    buckets: dict[tuple[int, int], list[str]] = {}
    for s, t, _, h in packets:
        if t < 0 or t > MAX_ANALYSIS_SECONDS * 10:
            continue
        buckets.setdefault((s, int(t // SEGMENT_S)), []).append(h)
    return [Leaf(s, kinds.get(s, "other"), seg, seg * SEGMENT_S, (seg + 1) * SEGMENT_S, len(hs),
                 hashlib.sha256("".join(hs).encode()).hexdigest())
            for (s, seg), hs in sorted(buckets.items())]


def _h(a: str, b: str) -> str:
    return hashlib.sha256(bytes.fromhex(a) + bytes.fromhex(b)).hexdigest()


def merkle_levels(leaf_hashes: list[str]) -> list[list[str]]:
    if not leaf_hashes:
        return [[hashlib.sha256(b"").hexdigest()]]
    levels = [list(leaf_hashes)]
    while len(levels[-1]) > 1:
        cur = levels[-1]
        if len(cur) % 2:
            cur = cur + [cur[-1]]
        levels.append([_h(cur[i], cur[i + 1]) for i in range(0, len(cur), 2)])
    return levels


def merkle_root(leaf_hashes: list[str]) -> str:
    return merkle_levels(leaf_hashes)[-1][0]


def merkle_proof(leaf_hashes: list[str], index: int) -> list[tuple[str, str]]:
    """Sibling path [(side, hash)] proving leaf `index` is under the root."""
    proof = []
    for level in merkle_levels(leaf_hashes)[:-1]:
        lvl = level + [level[-1]] if len(level) % 2 else level
        sib = index ^ 1
        proof.append(("left" if sib < index else "right", lvl[sib]))
        index //= 2
    return proof


def verify_proof(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    h = leaf
    for side, sib in proof:
        h = _h(sib, h) if side == "left" else _h(h, sib)
    return h == root


def thumbnails(path: Path, still: bool = False) -> Optional[np.ndarray]:
    """10 fps grey thumbnails; a still image (one video packet) is decoded as a single frame, because the fps
    filter emits nothing for a picture without a timeline."""
    if still:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf",
               f"scale={THUMB_W}:{THUMB_H},format=gray", "-frames:v", "1", "-f", "rawvideo", "-"]
    else:
        cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-t", str(MAX_ANALYSIS_SECONDS),
               "-map", "0:v:0", "-vf", f"fps={THUMB_FPS},scale={THUMB_W}:{THUMB_H},format=gray",
               "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    if not raw:
        return None
    return np.frombuffer(raw, np.uint8).reshape(-1, THUMB_H, THUMB_W).copy()


def phash(gray: np.ndarray) -> int:
    d = cv2.dct(cv2.resize(gray.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA))[:8, :8].flatten()[1:]
    bits = d > np.median(d)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class Fingerprint:
    kinds: dict[int, str]
    leaves: list[Leaf]
    merkle_root: str
    packet_count: int
    packet_set: set[str]
    thumbs: Optional[np.ndarray]
    phashes: list[int]
    width: Optional[int] = None
    height: Optional[int] = None
    duration_s: Optional[float] = None

    def leaves_json(self) -> list[dict]:
        return [asdict(l) for l in self.leaves]


def fingerprint(path: Path) -> Fingerprint:
    kinds, packets = packet_hashes(path)
    leaves = leaves_from_packets(kinds, packets)
    still = len(packets) == 1 and set(kinds.values()) == {"video"}
    thumbs = thumbnails(path, still=still) if "video" in kinds.values() else None
    ph = [phash(t) for t in thumbs] if thumbs is not None else []
    dur = max((t + d for _, t, d, _ in packets), default=None)
    w = h = None
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
    cap.release()
    return Fingerprint(kinds, leaves, merkle_root([l.hash for l in leaves]), len(packets),
                       {p[3] for p in packets}, thumbs, ph, w, h, dur)
