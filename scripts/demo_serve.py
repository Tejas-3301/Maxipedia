"""Serve the workspace (API + built UI) from one process on one port - the fallback if the live demo fails.

    python scripts/demo_serve.py                 # prerecorded demo_data/ (run scripts/demo_prepare.py first)
    python scripts/demo_serve.py --data data     # the normal data folder
Builds frontend/dist with npm if it is missing and npm is available.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "demo_data")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args()
    data = args.data.resolve()
    if not (data / "forensics.db").exists():
        sys.exit(f"no forensics.db in {data}; run: python scripts/demo_prepare.py --data {data}")

    dist = ROOT / "frontend" / "dist" / "index.html"
    if not dist.exists():
        npm = shutil.which("npm")
        if npm is None:
            print("WARNING: frontend/dist missing and npm not found - only the API will be served (/api/docs)")
        else:
            subprocess.run([npm, "install", "--no-audit", "--no-fund"], cwd=ROOT / "frontend", check=True)
            subprocess.run([npm, "run", "build"], cwd=ROOT / "frontend", check=True)

    manifest = data / "MANIFEST.json"
    url = f"http://127.0.0.1:{args.port}/"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        print("Prerecorded jobs:")
        for r in m["rows"]:
            print(f"  {r['file']:36s} {r['state']:20s} {url}#{r['job_id']}")
        first = next(iter(m["jobs"].values()))
        os.environ["FORENSICS_DATA_DIR"] = str(data)
        sys.path.insert(0, str(ROOT))
        from backend import db

        job = db.get_job(first)
        if job and not Path(job["media_path"]).exists():
            print("WARNING: media paths do not resolve (folder moved?) - re-run demo_prepare.py on this machine")
    os.environ["FORENSICS_DATA_DIR"] = str(data)
    sys.path.insert(0, str(ROOT))
    if args.open:
        webbrowser.open(url)
    import uvicorn

    print(f"\nServing {data} at {url}")
    uvicorn.run("backend.serve:app", host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
