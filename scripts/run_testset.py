"""Run every test-set file through the full pipeline (ingest -> all checks) and print findings + summary table.

Uses a throwaway data dir unless --keep is given (then artifacts/heatmaps land in data/artifacts/).
Usage: python scripts/run_testset.py [--testset data/testset] [--check visual] [--out results.json] [--keep] [--quiet]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def print_table(results: dict, expected_path: Path) -> None:
    """Markdown table: cell = max score of supports_manipulation findings ('-' none, 'n/a' not assessable)."""
    truth = {}
    if expected_path.exists():
        truth = {k: v["ground_truth"] for k, v in json.loads(expected_path.read_text())["files"].items()}
    checks = sorted({f["check"] for r in results.values() for f in r["findings"]} - {"ingest_inventory"})
    print("\n| file | truth | " + " | ".join(checks) + " |")
    print("|---|---|" + "---|" * len(checks))
    for name, r in results.items():
        cells = []
        for c in checks:
            fs = [f for f in r["findings"] if f["check"] == c]
            sup = [f["score"] or 0 for f in fs if f["stance"] == "supports_manipulation"]
            if sup:
                cells.append(f"**{max(sup):.2f}**" + (f" (x{len(sup)})" if len(sup) > 1 else ""))
            elif fs and all(f["stance"] == "not_assessable" for f in fs):
                cells.append("n/a")
            else:
                cells.append("-")
        print(f"| {name} | {truth.get(name, '?')} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--check", help="only include findings whose check name starts with this")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--keep", action="store_true", help="use the real data/ dir instead of a temp dir")
    ap.add_argument("--quiet", action="store_true", help="only print the summary table")
    args = ap.parse_args()
    if not args.keep:
        os.environ["FORENSICS_DATA_DIR"] = tempfile.mkdtemp(prefix="forensics_run_")
    sys.path.insert(0, str(ROOT))

    from backend import db, runner
    from backend.ingest import store_original

    db.init_db()
    results = {}
    for media in sorted(args.testset.glob("*.mp4")):
        job_id = uuid.uuid4().hex
        with open(media, "rb") as f:
            path, sha, size = store_original(job_id, media.name, f)
        db.create_job(job_id, media.name, str(path), sha, size)
        runner.process_job(job_id)
        job = db.get_job(job_id)
        findings = [f for f in db.get_findings(job_id) if not args.check or f.check.startswith(args.check)]
        results[media.name] = {"job_id": job_id, "status": job["status"], "error": job["error"],
                               "findings": [f.model_dump() for f in findings]}
        if args.quiet:
            continue
        print(f"\n=== {media.name}  [{job['status']}]  job={job_id}")
        for f in findings:
            score = "  -  " if f.score is None else f"{f.score:.2f}"
            seg = f" seg={f.segment.start}-{f.segment.end}{f.segment.unit}" if f.segment else ""
            reg = f" region={list(f.region)}" if f.region else ""
            print(f"  {f.check:30s} {f.stance:22s} score={score} rel={f.reliability:.2f}{seg}{reg}")
            print(textwrap.indent(textwrap.fill(f.rationale, 110), " " * 6))
    print_table(results, args.testset / "expected.json")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
