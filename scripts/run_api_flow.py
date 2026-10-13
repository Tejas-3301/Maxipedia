"""End-to-end flow over HTTP: upload each test file, poll the job, fetch findings + assessment, compare with
expected.json (evaluation only). Point --base at the backend (http://127.0.0.1:8000) or the Vite proxy
(http://127.0.0.1:5173/api) to exercise the same path the browser uses.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--trust", help="upload this file first and register it as the trusted reference")
    ap.add_argument("--provenance", action="store_true", help="print provenance findings, lineage graph, custody")
    args = ap.parse_args()
    expected = json.loads((args.testset / "expected.json").read_text())["files"]

    rows = []
    files = sorted(args.testset.glob("*.mp4"))
    if args.trust:
        files.sort(key=lambda p: p.name != args.trust)
    jobs: dict[str, str] = {}
    with httpx.Client(base_url=args.base, timeout=120) as c:
        print("health:", c.get("/health").json())
        for media in files:
            t0 = time.time()
            with open(media, "rb") as f:
                r = c.post("/jobs", files={"file": (media.name, f, "video/mp4")})
            r.raise_for_status()
            job_id = r.json()["job_id"]
            while True:
                job = c.get(f"/jobs/{job_id}").json()
                if job["status"] in ("done", "failed") or time.time() - t0 > args.timeout:
                    break
                time.sleep(0.5)
            jobs[media.name] = job_id
            if media.name == args.trust and job["status"] == "done":
                print("trusted reference:", c.post(f"/ledger/{job_id}/trust",
                                                   json={"label": "original from source"}).json()["filename"])
            if job["status"] != "done":
                rows.append((media.name, job["status"], None, None, "-", job.get("error"), time.time() - t0))
                continue
            findings = c.get(f"/jobs/{job_id}/findings").json()
            a = c.get(f"/jobs/{job_id}/assessment").json()
            exp = expected.get(media.name, {})
            ok = "OK" if a["state"] in exp.get("acceptable_verdicts", []) else "MISS"
            top = ", ".join(f"{s['check']}" for s in a["supporting"][:3]) or "-"
            if args.provenance:
                print(f"\n--- {media.name}")
                for f in findings:
                    if f["modality"] == "provenance" and f["check"] != "ingest_inventory":
                        print(f"  {f['check']:26s} {f['stance']:22s} score={f['score']}  {f['rationale']}")
            rows.append((media.name, a["state"], a["probability_manipulated"], a["confidence"],
                         f"{ok} (exp. {exp.get('expected_verdict', '?')})", f"{len(findings)} findings; top: {top}",
                         time.time() - t0))

    if args.provenance and jobs:
        with httpx.Client(base_url=args.base, timeout=60) as c:
            g = c.get("/provenance/graph").json()
            names = {n["job_id"]: n["filename"] + (" [trusted]" if n["trusted"] else "") for n in g["nodes"]}
            print("\nLineage graph (child -> parent):")
            for e in g["edges"]:
                r = e["relation"]
                print(f"  {names[e['child']]} -> {names[e['parent']]}  bits={r['median_bits']} crop={r['crop']} "
                      f"start={r['start_offset_s']}s coverage={r['coverage']} edits={r['offset_changes']}")
            for e in g["near_duplicates"]:
                print(f"  {names[e['a']]} ~ {names[e['b']]} (near-duplicate, direction unknown)")
            print("\nCustody verification:")
            for name, jid in jobs.items():
                v = c.get(f"/jobs/{jid}/custody").json()["verification"]
                print(f"  {name:24s} entries={v['entries']} intact={v['intact']} {v['problems'] or ''}")

    print("\n| file | verdict | P(manip) | confidence | vs expected | evidence | time |")
    print("|---|---|---|---|---|---|---|")
    for name, state, p, conf, ok, note, dt in rows:
        pf = "-" if p is None else f"{p:.2f}"
        cf = "-" if conf is None else f"{conf:.2f}"
        print(f"| {name} | {state} | {pf} | {cf} | {ok} | {note} | {dt:.1f}s |")


if __name__ == "__main__":
    main()
