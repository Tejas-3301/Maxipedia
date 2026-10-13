"""Hash-chained, Ed25519-signed chain of custody.

entry_hash = SHA-256(canonical JSON [job_id, action, actor, detail, ts, prev_hash]); signature = Ed25519
over entry_hash. Each job has its own chain. Key pair lives in data/keys/ (generated on first use).
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from backend.config import DATA_DIR
from backend.db import session

KEY_DIR = DATA_DIR / "keys"
SCHEME = 2
_lock = threading.RLock()
_key: Optional[Ed25519PrivateKey] = None


def _private_key() -> Ed25519PrivateKey:
    global _key
    with _lock:
        if _key is None:
            KEY_DIR.mkdir(parents=True, exist_ok=True)
            p = KEY_DIR / "custody_ed25519.pem"
            if p.exists():
                _key = serialization.load_pem_private_key(p.read_bytes(), password=None)
            else:
                _key = Ed25519PrivateKey.generate()
                p.write_bytes(_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()))
                (KEY_DIR / "custody_ed25519.pub.pem").write_bytes(public_pem(_key.public_key()).encode())
        return _key


def public_pem(pub: Optional[Ed25519PublicKey] = None) -> str:
    pub = pub or _private_key().public_key()
    return pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()


def key_id() -> str:
    raw = _private_key().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).hexdigest()[:16]


def _entry_hash(job_id: str, action: str, actor: str, detail_json: str, ts: str, prev: Optional[str]) -> str:
    return hashlib.sha256(json.dumps([job_id, action, actor, detail_json, ts, prev],
                                     separators=(",", ":")).encode()).hexdigest()


def append(job_id: str, action: str, detail: dict[str, Any], actor: str = "system") -> dict[str, Any]:
    detail_json = json.dumps(detail, sort_keys=True, separators=(",", ":"), default=str)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    key, kid = _private_key(), key_id()
    with _lock, session() as conn:
        prev = conn.execute("SELECT entry_hash FROM custody_log WHERE job_id = ? ORDER BY seq DESC LIMIT 1",
                            (job_id,)).fetchone()
        prev_hash = prev["entry_hash"] if prev else None
        h = _entry_hash(job_id, action, actor, detail_json, ts, prev_hash)
        sig = base64.b64encode(key.sign(bytes.fromhex(h))).decode()
        conn.execute("INSERT INTO custody_log (job_id, action, actor, detail, prev_hash, entry_hash, signature, ts, "
                     "scheme, key_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (job_id, action, actor, detail_json, prev_hash, h, sig, ts, SCHEME, kid))
    return {"action": action, "entry_hash": h, "prev_hash": prev_hash, "ts": ts}


def entries(job_id: str) -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute("SELECT * FROM custody_log WHERE job_id = ? ORDER BY seq", (job_id,)).fetchall()
    return [dict(r) for r in rows]


def verify(job_id: str, pub: Optional[Ed25519PublicKey] = None) -> dict[str, Any]:
    """Recompute every hash, check links and signatures. Any edit, deletion or reordering breaks it."""
    pub = pub or _private_key().public_key()
    rows = entries(job_id)
    problems: list[str] = []
    prev = None
    for r in rows:
        tag = f"seq {r['seq']} ({r['action']})"
        if r.get("scheme") != SCHEME:
            problems.append(f"{tag}: legacy unsigned entry")
            prev = r["entry_hash"]
            continue
        if r["prev_hash"] != prev:
            problems.append(f"{tag}: prev_hash does not link to the previous entry")
        h = _entry_hash(r["job_id"], r["action"], r["actor"], r["detail"], r["ts"], r["prev_hash"])
        if h != r["entry_hash"]:
            problems.append(f"{tag}: content does not match entry_hash (entry altered)")
        try:
            pub.verify(base64.b64decode(r["signature"] or ""), bytes.fromhex(r["entry_hash"]))
        except (InvalidSignature, ValueError):
            problems.append(f"{tag}: invalid signature")
        prev = r["entry_hash"]
    return {"job_id": job_id, "entries": len(rows), "intact": not problems, "problems": problems,
            "key_id": key_id(), "head": prev}
