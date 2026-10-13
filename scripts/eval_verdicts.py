"""Verdict table for the test set through the full in-process pipeline (fresh temp data dir).

--trust NAME processes that file first and registers it as the trusted reference before the others.
--out FILE saves {file: {state, p, confidence, coverage}} for before/after comparisons; --compare FILE prints both.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--trust")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args()
    os.environ["FORENSICS_DATA_DIR"] = tempfile.mkdtemp(prefix="forensics_eval_")
    sys.path.insert(0, str(ROOT))

    from backend import db, runner
    from backend.fusion import fuse
    from backend.ingest import store_original
    from backend.provenance import ledger

    db.init_db()
    expected = json.loads((args.testset / "expected.json").read_text())["files"]
    files = sorted(args.testset.glob("*.mp4"), key=lambda p: (p.name != args.trust, p.name))
    res = {}
    for media in files:
        jid = uuid.uuid4().hex
        with open(media, "rb") as f:
            path, sha, size = store_original(jid, media.name, f)
        db.create_job(jid, media.name, str(path), sha, size)
        runner.process_job(jid)
        if media.name == args.trust:
            ledger.set_trusted(jid, True, "trusted reference")
        job = db.get_job(jid)
        if job["status"] != "done":
            res[media.name] = {"state": "FAILED", "p": None, "confidence": None, "coverage": None, "error": job["error"]}
            continue
        a = fuse(db.get_findings(jid), schedule=(job["artifacts"] or {}).get("schedule"))
        res[media.name] = {"state": a["state"], "p": a["probability_manipulated"], "confidence": a["confidence"],
                           "coverage": a.get("coverage", {}).get("deep_coverage"), "reasons": a["reasons"]}

    before = json.loads(args.compare.read_text()) if args.compare else None
    hdr = "| file | expected |" + (" before | P before |" if before else "") + " verdict | P | deep coverage | ok |"
    print(hdr)
    print("|---" * (hdr.count("|") - 1) + "|")
    for name, r in res.items():
        exp = expected.get(name, {})
        ok = "OK" if r["state"] in exp.get("acceptable_verdicts", []) else "MISS"
        cov = "-" if r["coverage"] is None else f"{r['coverage']:.0%}"
        p = "-" if r["p"] is None else f"{r['p']:.3f}"
        pre = ""
        if before:
            b = before.get(name, {})
            pre = f" {b.get('state', '?')} | {b.get('p', 0) or 0:.3f} |"
        print(f"| {name} | {exp.get('expected_verdict', '?')} |{pre} {r['state']} | {p} | {cov} | {ok} |")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
