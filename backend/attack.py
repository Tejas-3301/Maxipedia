"""Attack mode: derive a degraded copy of a job's original (recompress / resize / add noise) to show honestly how
evidence and confidence degrade. The copy is analysed as a new job linked to its parent.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from backend.config import TOOL_TIMEOUT_S

LEVELS = ("light", "medium", "strong")
ATTACKS: dict[str, dict[str, dict[str, Any]]] = {
    "recompress": {"light": {"crf": 28, "abr": "96k"}, "medium": {"crf": 35, "abr": "64k"},
                   "strong": {"crf": 42, "abr": "32k"}},
    "resize": {"light": {"scale": 0.75}, "medium": {"scale": 0.5}, "strong": {"scale": 0.33}},
    "noise": {"light": {"v": 6, "a": 0.003}, "medium": {"v": 12, "a": 0.01}, "strong": {"v": 24, "a": 0.03}},
}


def describe(kind: str, level: str) -> str:
    p = ATTACKS[kind][level]
    if kind == "recompress":
        return f"re-encode H.264 crf {p['crf']}, AAC {p['abr']}"
    if kind == "resize":
        return f"downscale to {p['scale']:.0%} then re-encode"
    return f"temporal video noise strength {p['v']}, white audio noise amplitude {p['a']}"


def make_attack(src: Path, kind: str, level: str, out: Path) -> dict[str, Any]:
    if kind not in ATTACKS or level not in LEVELS:
        raise ValueError(f"unknown attack {kind}/{level}")
    p = ATTACKS[kind][level]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if kind == "recompress":
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", str(p["crf"]), "-preset", "veryfast",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", p["abr"]]
    elif kind == "resize":
        cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-vf", f"scale=trunc(iw*{p['scale']}/2)*2:-2",
                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-f", "lavfi", "-i", f"anoisesrc=color=white:amplitude={p['a']}:sample_rate=44100",
                "-filter_complex", f"[0:v:0]noise=alls={p['v']}:allf=t[v];[0:a:0][1:a]amix=inputs=2:duration=first:"
                                   f"normalize=0[a]",
                "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k"]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd + ["-movflags", "+faststart", str(out)], capture_output=True, text=True,
                   timeout=TOOL_TIMEOUT_S, check=True)
    return {"kind": kind, "level": level, "params": p, "description": describe(kind, level)}


def compare(parent: list, child: list) -> list[dict[str, Any]]:
    """Per check: strongest supporting score and clean score, parent vs attacked copy."""
    def summary(fs: list) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for f in fs:
            if f.check.startswith("screen.") or f.check == "ingest_inventory":
                continue
            d = out.setdefault(f.check, {"supports": 0.0, "contradicts": 0.0, "not_assessable": 0})
            if f.stance == "supports_manipulation":
                d["supports"] = max(d["supports"], f.score or 0.0)
            elif f.stance == "contradicts":
                d["contradicts"] = max(d["contradicts"], f.score or 0.0)
            elif f.stance == "not_assessable":
                d["not_assessable"] += 1
        return out

    a, b = summary(parent), summary(child)
    rows = []
    for check in sorted(set(a) | set(b)):
        pa, pb = a.get(check, {}), b.get(check, {})
        sa, sb = pa.get("supports", 0.0), pb.get("supports", 0.0)
        status = ("lost" if sa > 0 and sb == 0 else "weaker" if sb < sa - 0.05 else "new" if sa == 0 and sb > 0
                  else "stronger" if sb > sa + 0.05 else "kept" if sa > 0 else "")
        rows.append({"check": check, "parent_supports": round(sa, 3), "attacked_supports": round(sb, 3),
                     "parent_clean": round(pa.get("contradicts", 0.0), 3),
                     "attacked_clean": round(pb.get("contradicts", 0.0), 3),
                     "attacked_not_assessable": pb.get("not_assessable", 0), "status": status})
    return rows
