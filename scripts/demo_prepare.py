"""Prepare a prerecorded demo: run the whole pipeline over the test set into a self-contained data folder that
`scripts/demo_serve.py` can serve offline (no processing, no Node) if the live demo fails.

Contents: 01 captured & signed by a simulated camera (cam01) and registered as trusted; 02 submitted with 01's
camera manifest (claims to be the camera original -> origin chain broken, localized to the audio); 03-07 analysed;
a recompressed attack copy of 03; PDF reports for every file; DEMO.md with deep links and talking points.

    python scripts/demo_prepare.py [--data demo_data] [--force]
Paths inside the folder are absolute: build it on the demo machine and do not move it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = ".forensics_demo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "demo_data")
    ap.add_argument("--testset", type=Path, default=ROOT / "data" / "testset")
    ap.add_argument("--force", action="store_true", help="rebuild (only deletes a folder created by this script)")
    args = ap.parse_args()
    data = args.data.resolve()
    if data.exists() and any(data.iterdir()):
        if not args.force:
            sys.exit(f"{data} already exists; use --force to rebuild it")
        if not (data / MARKER).exists():
            sys.exit(f"{data} was not created by demo_prepare.py; refusing to delete it")
        shutil.rmtree(data)
    data.mkdir(parents=True)
    (data / MARKER).write_text("created by scripts/demo_prepare.py\n")
    os.environ["FORENSICS_DATA_DIR"] = str(data)
    sys.path.insert(0, str(ROOT))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from backend import db, runner
    from backend.attack import make_attack
    from backend.fusion import fuse
    from backend.ingest import store_original
    from backend.main import export_report, startup
    from backend.provenance import custody, device, ledger

    startup()
    expected = json.loads((args.testset / "expected.json").read_text())["files"]

    # simulated capture device
    priv, pub = device.new_device_key()
    (data / "devices").mkdir()
    (data / "devices" / "cam01.key.pem").write_bytes(priv)
    device.register("cam01", pub, "simulated body camera")
    cam_manifest = device.sign_capture(args.testset / "01_clean.mp4", "cam01", priv)
    (data / "devices" / "01_clean.mp4.capture.json").write_text(json.dumps(cam_manifest, indent=1))

    def ingest(path: Path, name: str, manifest: dict | None = None, parent: str | None = None,
               attack: dict | None = None) -> str:
        jid = uuid.uuid4().hex
        with open(path, "rb") as f:
            p, sha, size = store_original(jid, name, f)
        detail = {"filename": name, "sha256": sha, "size_bytes": size}
        if manifest:
            (Path(p).parent / "capture_manifest.json").write_text(json.dumps(manifest))
            detail["capture_manifest"] = manifest["device_id"]
        db.create_job(jid, name, str(p), sha, size)
        if parent:
            db.update_job(jid, derived_from=parent, attack=attack)
            detail["attack_of"] = parent
        custody.append(jid, "ingested", detail, actor="demo")
        runner.process_job(jid)
        print(f"  {name:34s} {db.get_job(jid)['status']}")
        return jid

    print("analysing test set ...")
    jobs: dict[str, str] = {}
    for media in sorted(args.testset.glob("*.mp4")):
        manifest = cam_manifest if media.name in ("01_clean.mp4", "02_voice_swap.mp4") else None
        jobs[media.name] = ingest(media, media.name, manifest)
        if media.name == "01_clean.mp4":
            ledger.set_trusted(jobs[media.name], True, "camera original (cam01)")
            custody.append(jobs[media.name], "trust_set", {"label": "camera original (cam01)"}, actor="demo")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "03_splice__recompress-medium.mp4"
        info = make_attack(args.testset / "03_splice.mp4", "recompress", "medium", out)
        jobs[out.name] = ingest(out, out.name, parent=jobs["03_splice.mp4"], attack=info)
        custody.append(jobs["03_splice.mp4"], "attack_mode", {"child": jobs[out.name], "attack": info}, actor="demo")

    print("exporting reports ...")
    (data / "reports").mkdir()
    rows = []
    for name, jid in jobs.items():
        job = db.get_job(jid)
        a = fuse(db.get_findings(jid), schedule=(job["artifacts"] or {}).get("schedule"), media_type=job["media_type"])
        pdf = export_report(jid, actor="demo")
        (data / "reports" / f"{Path(name).stem}.pdf").write_bytes(pdf.body)
        top = [s["check"] for s in a["supporting"][:3]]
        rows.append({"file": name, "job_id": jid, "state": a["state"], "p": a["probability_manipulated"],
                     "expected": expected.get(name, {}).get("expected_verdict", "(attack copy)"), "top": top})

    (data / "MANIFEST.json").write_text(json.dumps({"jobs": jobs, "rows": rows}, indent=1))
    write_demo_md(data, rows)
    print(f"\ndemo data ready in {data}\nserve it with:  python scripts/demo_serve.py --data {data}")


TALK = {
    "01_clean.mp4": "Clean camera original, signed at capture by cam01: origin chain intact, every check clean.",
    "02_voice_swap.mp4": "TTS voice swapped in at 2.5-5.5 s. Audio lane: high band missing + noise floor drops to "
                         "digital silence; sync lane: lips stop matching; camera manifest of 01 -> origin chain broken, "
                         "localized to the audio.",
    "03_splice.mp4": "Region replaced 3.0-4.6 s (click the red block: box over the vanished ball) and 0.4 s cut at "
                     "6.0 s (temporal jump + lineage vs the trusted original says '0.4 s removed at 6.0 s').",
    "04_reencoded_3x.mp4": "Content untouched, re-encoded three times: re-encode hint fires, lineage to the trusted "
                           "original is clean.",
    "05_metadata_edited.mp4": "Streams bit-identical to 01 (Merkle root equal) but forged timestamps / device tags: "
                              "manipulated metadata, untouched content.",
    "06_derived_a.mp4": "Trimmed + downscaled copy: lineage finds 1.0-7.0 s of the trusted original, no cuts.",
    "07_derived_b.mp4": "Copy of a copy (crop + brighten): lineage via 06, judged against the trusted original.",
    "03_splice__recompress-medium.mp4": "Attack mode: recompression erases the pixel evidence; the system drops to "
                                        "inconclusive (quality gate) instead of claiming authenticity.",
}


def write_demo_md(data: Path, rows: list[dict]) -> None:
    L = ["# Prerecorded demo", "", "Serve: `python scripts/demo_serve.py --data " + str(data) + "` then open the links.",
         "", "| file | verdict | P | expected | top evidence | open | talking point |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['file']} | **{r['state']}** | {r['p']:.2f} | {r['expected']} | {', '.join(r['top']) or '-'} | "
                 f"http://127.0.0.1:8000/#{r['job_id']} | {TALK.get(r['file'], '')} |")
    L += ["", "PDF reports (with chain of custody): `reports/`.", ""]
    (data / "DEMO.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
