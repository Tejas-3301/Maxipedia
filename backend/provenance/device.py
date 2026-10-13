"""Capture-device signing (simulation of an IoT camera that signs the content hash at capture).

A device holds an Ed25519 key. At capture it signs a manifest: device id, capture time, SHA-256 of the file, the
Merkle root over per-segment packet hashes, and the leaves themselves (so a verifier can localize changes).
The verifier trusts only devices registered with their public key.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from backend.db import session
from backend.provenance.fingerprint import fingerprint

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id      TEXT PRIMARY KEY,
    public_key_pem TEXT NOT NULL,
    label          TEXT,
    created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""
SIGNED_FIELDS = ("version", "device_id", "captured_at", "filename", "sha256", "merkle_root", "leaves")


def _canonical(manifest: dict[str, Any]) -> bytes:
    return json.dumps({k: manifest[k] for k in SIGNED_FIELDS}, sort_keys=True, separators=(",", ":")).encode()


def new_device_key() -> tuple[bytes, str]:
    key = Ed25519PrivateKey.generate()
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    pub = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return pem, pub


def sign_capture(media: Path, device_id: str, private_pem: bytes) -> dict[str, Any]:
    """What the device does at capture time."""
    import hashlib

    key = serialization.load_pem_private_key(private_pem, password=None)
    fp = fingerprint(media)
    manifest = {
        "version": 1,
        "device_id": device_id,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filename": media.name,
        "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
        "merkle_root": fp.merkle_root,
        "leaves": [{"kind": l.kind, "segment": l.segment, "start": l.start, "end": l.end, "hash": l.hash} for l in fp.leaves],
    }
    manifest["signature"] = base64.b64encode(key.sign(_canonical(manifest))).decode()
    return manifest


def register(device_id: str, public_key_pem: str, label: Optional[str] = None) -> None:
    serialization.load_pem_public_key(public_key_pem.encode())  # validate
    with session() as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT OR REPLACE INTO devices (device_id, public_key_pem, label) VALUES (?,?,?)",
                     (device_id, public_key_pem, label))


def devices() -> list[dict[str, Any]]:
    with session() as conn:
        conn.executescript(SCHEMA)
        return [dict(r) for r in conn.execute("SELECT device_id, label, created_at FROM devices ORDER BY created_at")]


def public_key(device_id: str) -> Optional[Ed25519PublicKey]:
    with session() as conn:
        conn.executescript(SCHEMA)
        r = conn.execute("SELECT public_key_pem FROM devices WHERE device_id = ?", (device_id,)).fetchone()
    return serialization.load_pem_public_key(r["public_key_pem"].encode()) if r else None


def verify_signature(manifest: dict[str, Any]) -> tuple[Optional[bool], str]:
    """(True, '') valid; (False, why) invalid; (None, why) cannot verify (unknown device / malformed)."""
    try:
        missing = [k for k in SIGNED_FIELDS + ("signature",) if k not in manifest]
        if missing:
            return None, f"manifest missing fields {missing}"
        pub = public_key(manifest["device_id"])
        if pub is None:
            return None, f"device {manifest['device_id']!r} is not registered"
        pub.verify(base64.b64decode(manifest["signature"]), _canonical(manifest))
        return True, ""
    except InvalidSignature:
        return False, "signature does not match the manifest (manifest altered or signed by another key)"
    except (ValueError, TypeError) as exc:
        return None, f"malformed manifest: {exc}"
