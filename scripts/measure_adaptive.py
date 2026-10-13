"""Measure deadline-driven deep analysis vs an unlimited run (real timings, not estimates).

For every file: ingest once, screen, schedule, then run the deep units (visual region, visual temporal, lip sync)
(a) with no deadline (every interval) and (b) with the deadline. Reports elapsed vs budget, coverage, the
intervals left unanalysed, whether ground-truth segments (expected.json, evaluation only) were analysed,
control hits / miss-rate estimate, and both verdicts.
Usage: python scripts/measure_adaptive.py [--testset DIR] [--budget 20] [--skip-full]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def truth_segments(entry: dict) -> list[tuple[float, float]]:
    return [(t["segment"]["start"], t["segment"]["end"]) for t in entry.get("should_trigger", [])
            if t.get("modality") in ("visual", "audio", "sync") and t.get("segment")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--skip-full", action="store_true", help="only run the deadline mode")
    args = ap.parse_args()
    os.environ["FORENSICS_DATA_DIR"] = tempfile.mkdtemp(prefix="forensics_adaptive_")
    sys.path.insert(0, str(ROOT))

    from backend import db
    from backend.adaptive.deep import run_deep
    from backend.adaptive.screening import screening_findings
    from backend.checks import run_checks
    from backend.config import DEEP_BUDGET_S
    from backend.fusion import fuse
    from backend.ingest import preprocess, store_original
    from backend.runner import build_context, plan

    budget = args.budget or DEEP_BUDGET_S
    db.init_db()
    exp_path = args.testset / "expected.json"
    expected = json.loads(exp_path.read_text())["files"] if exp_path.exists() else {}
    rows = []
    for media in sorted(args.testset.glob("*.mp4")):
        jid = uuid.uuid4().hex
        with open(media, "rb") as f:
            path, sha, size = store_original(jid, media.name, f)
        db.create_job(jid, media.name, str(path), sha, size)
        arts = preprocess(jid, path)
        ctx = build_context(db.get_job(jid), arts)
        t = time.perf_counter()
        scr, sched = plan(ctx, arts)
        t_screen = time.perf_counter() - t
        base = run_checks(ctx)

        full = None
        if not args.skip_full:
            deep_full, full = run_deep(ctx, sched, budget_s=float("inf"))
            v_full = fuse(base + deep_full + screening_findings(scr), schedule=full, media_type="video")["state"]
        deep, rep = run_deep(ctx, sched, budget_s=budget)
        a = fuse(base + deep + screening_findings(scr), schedule=rep, media_type="video")

        truth = truth_segments(expected.get(media.name, {}))
        covered = [any(i["start"] <= (s + e) / 2 <= i["end"] for i in rep["analysed"]) for s, e in truth]
        unan = [i for i in rep["unanalysed"]]
        rows.append({
            "file": media.name, "screen_s": t_screen, "full_s": full["elapsed_s"] if full else None,
            "elapsed_s": rep["elapsed_s"], "budget_s": budget, "coverage": rep["coverage"],
            "analysed": len(rep["analysed"]), "unanalysed": unan,
            "truth": f"{sum(covered)}/{len(covered)}" if truth else "n/a",
            "controls": f"{rep['control_hits']}/{rep['controls_analysed']}", "miss": rep["miss_rate_est"],
            "verdict_full": v_full if full else "-", "verdict": a["state"], "p": a["probability_manipulated"]})

    print("| file | screen s | full deep s | deadline run s | budget s | coverage | analysed | unanalysed (flagged) "
          "| truth analysed | control hits | miss est | verdict (no deadline) | verdict (deadline) | P |")
    print("|---" * 14 + "|")
    for r in rows:
        unf = sum(i["kind"] in ("flagged", "manual") for i in r["unanalysed"])
        full = "-" if r["full_s"] is None else f"{r['full_s']:.2f}"
        print(f"| {r['file']} | {r['screen_s']:.2f} | {full} | {r['elapsed_s']:.2f} | {r['budget_s']:.0f} | "
              f"{r['coverage']:.0%} | {r['analysed']} | {len(r['unanalysed'])} ({unf}) | {r['truth']} | {r['controls']} "
              f"| {r['miss']:.0%} | {r['verdict_full']} | {r['verdict']} | {r['p']:.3f} |")
    for r in rows:
        if r["unanalysed"]:
            print(f"\n{r['file']} unanalysed: " + ", ".join(f"{i['start']:.1f}-{i['end']:.1f}s ({i['kind']})"
                                                             for i in r["unanalysed"]))


if __name__ == "__main__":
    main()
