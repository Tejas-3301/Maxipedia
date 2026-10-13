"""Scheduler: turn the suspicion curve into an ordered list of deep-analysis intervals.

There is no cost estimate. The deep runner (backend/adaptive/deep.py) walks this list in order, checks the
elapsed time before starting each interval and stops at the deadline, so the budget sets K and a longer clip
gets less coverage instead of a timeout. Order:
  manual (investigator) -> top flagged -> CONTROL_COUNT random controls -> remaining flagged (desc. suspicion)
  -> remaining unflagged chunks (random, seeded per file).
Every unflagged chunk that gets analysed acts as a control for the screening miss-rate estimate.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Optional

import numpy as np

PAD_S = 0.75
MIN_LEN_S = 1.0
MERGE_GAP_S = 0.5
CONTROL_COUNT = 2
CHUNK_S = 2.0


def union_length(iv: list[dict[str, Any]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for x in sorted(iv, key=lambda i: i["start"]):
        if cur_e is None or x["start"] > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = x["start"], x["end"]
        else:
            cur_e = max(cur_e, x["end"])
    return total + ((cur_e - cur_s) if cur_e is not None else 0.0)


def _merge(iv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for x in sorted(iv, key=lambda i: i["start"]):
        if out and x["start"] <= out[-1]["end"] + MERGE_GAP_S and x["kind"] == out[-1]["kind"]:
            out[-1]["end"] = max(out[-1]["end"], x["end"])
            out[-1]["score"] = max(out[-1]["score"], x["score"])
            out[-1]["sources"] = sorted(set(out[-1]["sources"]) | set(x["sources"]))
        else:
            out.append(dict(x))
    return out


def _peaks(curve: np.ndarray, level: float) -> list[int]:
    return sorted((i for i in range(len(curve)) if curve[i] >= level
                   and curve[i] >= (curve[i - 1] if i else 0) and curve[i] >= (curve[i + 1] if i + 1 < len(curve) else 0)),
                  key=lambda i: -curve[i])


def _subtract(dur: float, taken: list[dict[str, Any]]) -> list[tuple[float, float]]:
    free, cur = [], 0.0
    for x in sorted(taken, key=lambda i: i["start"]):
        if x["start"] > cur:
            free.append((cur, x["start"]))
        cur = max(cur, x["end"])
    if cur < dur:
        free.append((cur, dur))
    return [(s, e) for s, e in free if e - s >= 0.25]


def _chunks(free: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for s, e in free:
        n = max(1, int(round((e - s) / CHUNK_S)))
        step = (e - s) / n
        out += [(round(s + k * step, 3), round(s + (k + 1) * step, 3)) for k in range(n)]
    return out


def schedule(scr: dict[str, Any], duration: float, seed: str,
             manual: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    bin_s, level = scr["bin_s"], scr["flag_level"]
    curve = np.array(scr["combined"]) if scr.get("combined") else np.zeros(1)
    dur = min(duration, len(curve) * bin_s) if scr.get("combined") else duration

    flagged: list[dict[str, Any]] = []
    for p in _peaks(curve, level):
        if any(x["start"] <= p * bin_s <= x["end"] for x in flagged):
            continue
        lo = hi = p
        while lo > 0 and curve[lo - 1] >= curve[p] / 2:
            lo -= 1
        while hi + 1 < len(curve) and curve[hi + 1] >= curve[p] / 2:
            hi += 1
        s, e = max(0.0, lo * bin_s - PAD_S), min(dur, (hi + 1) * bin_s + PAD_S)
        if e - s < MIN_LEN_S:
            c = (s + e) / 2
            s, e = max(0.0, c - MIN_LEN_S / 2), min(dur, c + MIN_LEN_S / 2)
        sources = sorted(m for m, c in scr["curves"].items() if max(c[lo:hi + 1]) >= level)
        flagged = _merge(flagged + [{"start": round(s, 3), "end": round(e, 3), "kind": "flagged",
                                     "score": round(float(curve[p]), 3), "sources": sources}])
    flagged.sort(key=lambda x: -x["score"])

    manual_iv = _merge([{"start": round(m["start"], 3), "end": round(m["end"], 3), "kind": "manual",
                         "score": 1.0, "sources": ["investigator"]} for m in manual or []])

    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:12], 16))  # deterministic per file
    free = _chunks(_subtract(dur, flagged + manual_iv))
    rng.shuffle(free)
    controls = [{"start": s, "end": e, "kind": "control", "score": 0.0, "sources": []} for s, e in free]

    ordered = manual_iv + flagged[:1] + controls[:CONTROL_COUNT] + flagged[1:] + controls[CONTROL_COUNT:]
    for k, iv in enumerate(ordered):
        iv["order"] = k
    return {"candidates": ordered, "duration_s": round(dur, 3), "flag_level": level,
            "flagged": len(flagged), "manual": manual or []}
