"""Image verdict table: run every image in the given folders through the full pipeline (fresh temp data dir) and
print file, checks fired, assessable count, P, verdict, confidence and reasons.

    python scripts/eval_images.py data/images_test [more dirs or files] [--out table.md]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--details", action="store_true", help="print every finding's rationale")
    args = ap.parse_args()
    os.environ["FORENSICS_DATA_DIR"] = tempfile.mkdtemp(prefix="forensics_images_")
    sys.path.insert(0, str(ROOT))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from backend import db, runner
    from backend.fusion import fuse
    from backend.ingest import store_original

    db.init_db()
    files = []
    for p in args.paths:
        files += sorted(x for x in p.iterdir() if x.suffix.lower() in EXTS) if p.is_dir() else [p]

    lines = ["| file | checks fired (supports) | clean (contradicts) | assessable | P | verdict | confidence | why |",
             "|---|---|---|---|---|---|---|---|"]
    for media in files:
        jid = uuid.uuid4().hex
        with open(media, "rb") as f:
            path, sha, size = store_original(jid, media.name, f)
        db.create_job(jid, media.name, str(path), sha, size)
        runner.process_job(jid)
        job = db.get_job(jid)
        fs = db.get_findings(jid)
        a = fuse(fs, schedule=None, media_type=job["media_type"])
        sup = sorted({f.check for f in fs if f.stance == "supports_manipulation"})
        con = sorted({f.check for f in fs if f.stance == "contradicts"})
        conf = "abstained" if a["confidence"] is None else f"{a['confidence']:.2f}"
        why = "; ".join(a["reasons"]) or "-"
        lines.append(f"| {media.name} | {', '.join(sup) or '-'} | {', '.join(con) or '-'} | "
                     f"{a['counts']['assessable']} | {a['probability_manipulated']:.3f} | {a['state']} | {conf} | {why} |")
        if args.details:
            print(f"\n=== {media.name} ({job['media_type']})")
            for f in fs:
                if f.check != "ingest_inventory":
                    print(f"  {f.check:30s} {f.stance:22s} score={f.score} rel={f.reliability} cov={f.coverage}\n"
                          f"      {f.rationale[:220]}")
    print("\n".join(lines))
    if args.out:
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
