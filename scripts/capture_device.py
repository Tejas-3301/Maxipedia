"""Simulated capture device (IoT camera) that signs the content hash at capture.

    python scripts/capture_device.py keygen cam01                  # key in data/devices/, prints public key
    python scripts/capture_device.py register cam01 [--api URL]    # register the public key with the server
    python scripts/capture_device.py sign cam01 clip.mp4           # writes clip.mp4.capture.json
    curl -F file=@clip.mp4 -F manifest=@clip.mp4.capture.json http://localhost:8000/jobs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KEYS = ROOT / "data" / "devices"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("device_id")
    r = sub.add_parser("register")
    r.add_argument("device_id")
    r.add_argument("--api", default="http://127.0.0.1:8000")
    r.add_argument("--label")
    s = sub.add_parser("sign")
    s.add_argument("device_id")
    s.add_argument("media", type=Path)
    s.add_argument("--out", type=Path)
    args = ap.parse_args()

    from backend.provenance.device import new_device_key, sign_capture

    priv, pub = KEYS / f"{args.device_id}.key.pem", KEYS / f"{args.device_id}.pub.pem"
    if args.cmd == "keygen":
        if priv.exists():
            sys.exit(f"{priv} exists; refusing to overwrite a device key")
        KEYS.mkdir(parents=True, exist_ok=True)
        pem, pub_pem = new_device_key()
        priv.write_bytes(pem)
        pub.write_text(pub_pem)
        print(pub_pem)
    elif args.cmd == "register":
        import httpx

        r = httpx.post(f"{args.api}/devices", json={"device_id": args.device_id, "public_key_pem": pub.read_text(),
                                                     "label": args.label}, timeout=30)
        r.raise_for_status()
        print(r.json())
    else:
        manifest = sign_capture(args.media, args.device_id, priv.read_bytes())
        out = args.out or args.media.with_name(args.media.name + ".capture.json")
        out.write_text(json.dumps(manifest, indent=1))
        print(f"signed {args.media.name}: sha256 {manifest['sha256'][:16]}... root {manifest['merkle_root'][:16]}... -> {out}")


if __name__ == "__main__":
    main()
