"""Results table over the test set: verdict + the system's own explanation per file, measured screening time
saving, screening recall against ground truth, control-interval miss-rate, and (optionally) attack-mode
degradation and long clips. Ground truth comes from expected.json and is used only for scoring here.

Usage: python scripts/results_table.py [--testset data/testset] [--trust 01_clean.mp4] [--attacks] [--long DIR]
Writes data/results/results.md and data/results/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def truth_segments(entry: dict) -> list[tuple[float, float]]:
    return [(t["segment"]["start"], t["segment"]["end"]) for t in entry.get("should_trigger", [])
            if t.get("modality") in ("visual", "audio", "sync") and t.get("segment")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--trust", default="01_clean.mp4", help="file registered as trusted reference ('' for none)")
    ap.add_argument("--attacks", action="store_true", help="attack every manipulated file (recompress/resize/noise)")
    ap.add_argument("--level", default="medium")
    ap.add_argument("--long", type=Path, help="extra directory of long clips for timing only")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "results")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    os.environ["FORENSICS_DATA_DIR"] = tempfile.mkdtemp(prefix="forensics_results_")
    sys.path.insert(0, str(ROOT))

    from backend import db, runner
    from backend.adaptive.deep import run_deep
    from backend.attack import make_attack
    from backend.fusion import fuse
    from backend.ingest import store_original
    from backend.provenance import ledger

    db.init_db()
    expected = json.loads((args.testset / "expected.json").read_text())["files"]

    def ingest(path: Path, name: str, parent: str | None = None, attack: dict | None = None) -> str:
        jid = uuid.uuid4().hex
        with open(path, "rb") as f:
            p, sha, size = store_original(jid, name, f)
        db.create_job(jid, name, str(p), sha, size)
        if parent:
            db.update_job(jid, derived_from=parent, attack=attack)
        runner.process_job(jid)
        return jid

    def assess(jid: str) -> dict:
        job = db.get_job(jid)
        return fuse(db.get_findings(jid), schedule=(job["artifacts"] or {}).get("schedule"), media_type=job["media_type"])

    def timing(jid: str) -> dict:
        job = db.get_job(jid)
        arts = job["artifacts"] or {}
        sched, scr = arts.get("schedule"), arts.get("screening")
        if not sched:
            return {}
        ctx = runner.build_context(job, arts)
        cand = {k: v for k, v in sched.items() if k in ("candidates", "duration_s", "flag_level", "flagged", "manual")}
        _, full = run_deep(ctx, cand, budget_s=float("inf"))
        return {"screen_s": scr.get("elapsed_s", 0.0), "deadline_s": sched["elapsed_s"], "budget_s": sched["budget_s"],
                "unlimited_s": full["elapsed_s"], "coverage": sched["coverage"],
                "controls": sched["controls_analysed"], "control_hits": sched["control_hits"],
                "unanalysed": [{k: i[k] for k in ("start", "end", "kind")} for i in sched["unanalysed"]],
                "analysed": [{k: i[k] for k in ("start", "end", "kind")} for i in sched["analysed"]]}

    files = sorted(args.testset.glob("*.mp4"), key=lambda p: (p.name != args.trust, p.name))
    rows, jobs = [], {}
    for media in files:
        t0 = time.perf_counter()
        jid = ingest(media, media.name)
        wall = time.perf_counter() - t0
        if media.name == args.trust:
            ledger.set_trusted(jid, True, "trusted reference")
        jobs[media.name] = jid
        a = assess(jid)
        exp = expected.get(media.name, {})
        tm = timing(jid)
        truth = truth_segments(exp)
        flagged = [i for i in (db.get_job(jid)["artifacts"]["schedule"] or {}).get("analysed", []) if i["kind"] == "flagged"]
        hit = [any(i["start"] <= (s + e) / 2 <= i["end"] for i in flagged) for s, e in truth]
        rows.append({"file": media.name, "truth": exp.get("ground_truth"), "expected": exp.get("expected_verdict"),
                     "acceptable": exp.get("acceptable_verdicts", []), "state": a["state"],
                     "p": a["probability_manipulated"], "confidence": a["confidence"],
                     "ok": a["state"] in exp.get("acceptable_verdicts", []), "exact": a["state"] == exp.get("expected_verdict"),
                     "explanation": a["summary"], "reasons": a["reasons"],
                     "top": [f"{s['check']} ({s['segment']['start']:.1f}-{s['segment']['end']:.1f}s)" if s["segment"]
                             else s["check"] for s in a["supporting"][:3]],
                     "truth_segments": len(truth), "truth_flagged": sum(hit), "wall_s": wall, **tm})

    long_rows = []
    if args.long:
        for media in sorted(args.long.glob("*.mp4")):
            jid = ingest(media, media.name)
            a = assess(jid)
            long_rows.append({"file": media.name, "state": a["state"], "p": a["probability_manipulated"], **timing(jid)})

    attack_rows = []
    if args.attacks:
        for r in rows:
            if not str(r["truth"]).startswith("manipulated"):
                continue
            parent = jobs[r["file"]]
            for kind in ("recompress", "resize", "noise"):
                with tempfile.TemporaryDirectory() as tmp:
                    out = Path(tmp) / f"{Path(r['file']).stem}__{kind}-{args.level}.mp4"
                    info = make_attack(Path(db.get_job(parent)["media_path"]), kind, args.level, out)
                    cid = ingest(out, out.name, parent, info)
                a = assess(cid)
                attack_rows.append({"file": r["file"], "attack": info["description"], "kind": kind,
                                    "p_before": r["p"], "state_before": r["state"],
                                    "p_after": a["probability_manipulated"], "state_after": a["state"],
                                    "top_after": [s["check"] for s in a["supporting"][:3]]})

    write(args.out, rows, long_rows, attack_rows, args)


def write(out: Path, rows: list, long_rows: list, attack_rows: list, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# Results on the ground-truth test set", "",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}; trusted reference: {args.trust or 'none'}; "
         f"verdicts by evidence fusion only.", "", "## Verdicts", "",
         "| file | ground truth | expected | verdict | P(manip) | confidence | acceptable | system's explanation |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        conf = "abstained" if r["confidence"] is None else f"{r['confidence']:.2f}"
        expl = "; ".join(r["top"]) or "no evidence for manipulation"
        if r["reasons"]:
            expl += " — held back: " + "; ".join(r["reasons"])
        L.append(f"| {r['file']} | {r['truth']} | {r['expected']} | **{r['state']}** | {r['p']:.3f} | {conf} | "
                 f"{'yes' if r['ok'] else 'NO'}{' (exact)' if r['exact'] else ''} | {expl} |")
    n, ok, exact = len(rows), sum(r["ok"] for r in rows), sum(r["exact"] for r in rows)
    L += ["", f"Acceptable verdict on **{ok}/{n}** files; exactly the expected verdict on {exact}/{n}.", ""]

    def timing_table(rs: list, title: str) -> list[str]:
        out_ = [f"## {title}", "",
                "| file | screening s | deep with deadline s | budget s | deep unlimited s | saving | deep coverage | not analysed |",
                "|---|---|---|---|---|---|---|---|"]
        tot_a = tot_u = 0.0
        for r in rs:
            if "unlimited_s" not in r:
                continue
            spent = r["screen_s"] + r["deadline_s"]
            tot_a, tot_u = tot_a + spent, tot_u + r["unlimited_s"]
            sav = 1 - spent / r["unlimited_s"] if r["unlimited_s"] else 0.0
            unf = sum(i["kind"] == "flagged" for i in r["unanalysed"])
            out_.append(f"| {r['file']} | {r['screen_s']:.2f} | {r['deadline_s']:.2f} | {r['budget_s']:.0f} | "
                        f"{r['unlimited_s']:.2f} | {sav:+.0%} | {r['coverage']:.0%} | {len(r['unanalysed'])} "
                        f"({unf} flagged) |")
        if tot_u:
            out_ += ["", f"Total: screening + deadline-limited deep analysis **{tot_a:.1f} s** vs unlimited deep analysis "
                         f"**{tot_u:.1f} s** → measured saving **{1 - tot_a / tot_u:+.0%}** (negative = adaptive costs more).", ""]
        return out_

    L += timing_table(rows, "Screening time saving (measured, test set)")
    if long_rows:
        L += timing_table(long_rows, "Screening time saving (measured, long clips)")

    seg_total = sum(r["truth_segments"] for r in rows)
    seg_hit = sum(r["truth_flagged"] for r in rows)
    ctl = sum(r.get("controls", 0) for r in rows + long_rows)
    hits = sum(r.get("control_hits", 0) for r in rows + long_rows)
    L += ["## Screening recall and miss-rate", "",
          f"- Ground-truth manipulated segments inside a *flagged* interval: **{seg_hit}/{seg_total}** "
          f"(recall {seg_hit / seg_total:.0%})." if seg_total else "- No ground-truth segments.",
          f"- Control intervals deep-checked (unflagged time): **{ctl}**, with deep findings: **{hits}** → raw miss-rate "
          f"{hits / ctl:.0%}, Laplace-smoothed {(hits + 1) / (ctl + 2):.0%} (the value fusion uses to cap clean evidence)."
          if ctl else "- No control intervals analysed.", ""]

    if attack_rows:
        L += [f"## Attack mode ({args.level}) — evidence degrading", "",
              "| file | attack | verdict before | P before | verdict after | P after | remaining top evidence |",
              "|---|---|---|---|---|---|---|"]
        for a in attack_rows:
            L.append(f"| {a['file']} | {a['attack']} | {a['state_before']} | {a['p_before']:.3f} | **{a['state_after']}** | "
                     f"{a['p_after']:.3f} | {', '.join(a['top_after']) or '-'} |")
        to_auth = sum(a["state_after"] == "likely authentic" for a in attack_rows)
        L += ["", f"{len(attack_rows)} attacked copies of manipulated files: {to_auth} flipped to 'likely authentic', "
                  f"{sum(a['state_after'] == 'inconclusive' for a in attack_rows)} became inconclusive, "
                  f"{sum(a['state_after'] == 'likely manipulated' for a in attack_rows)} stayed 'likely manipulated'.", ""]

    (out / "results.md").write_text("\n".join(L), encoding="utf-8")
    (out / "results.json").write_text(json.dumps({"files": rows, "long": long_rows, "attacks": attack_rows}, indent=2,
                                                 default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
