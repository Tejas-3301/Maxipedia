"""Ledger of known versions: whole-file SHA-256, Merkle root + leaves, packet hashes, visual signature.

Every analysed file is recorded. A version can be marked `trusted` (a registered reference/original,
e.g. received from the source); only trusted references produce supports/contradicts provenance findings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.db import session
from backend.provenance.fingerprint import Fingerprint, Leaf


@dataclass
class Version:
    job_id: str
    sha256: str
    filename: str
    trusted: bool
    label: Optional[str]
    merkle_root: str
    leaves: list[Leaf]
    packet_set: set[str]
    phashes: list[int]
    thumbs_path: Optional[str]
    width: Optional[int]
    height: Optional[int]
    duration_s: Optional[float]
    created_at: str

    def thumbs(self) -> Optional[np.ndarray]:
        if self.thumbs_path and Path(self.thumbs_path).exists():
            return np.load(self.thumbs_path)["t"]
        return None

    def brief(self) -> dict[str, Any]:
        return {"job_id": self.job_id, "filename": self.filename, "sha256": self.sha256, "trusted": self.trusted,
                "label": self.label, "merkle_root": self.merkle_root, "width": self.width, "height": self.height,
                "duration_s": self.duration_s, "created_at": self.created_at}


def _row(r) -> Version:
    return Version(r["job_id"], r["sha256"], r["filename"], bool(r["trusted"]), r["label"], r["merkle_root"],
                   [Leaf(**l) for l in json.loads(r["leaves"])], set(json.loads(r["packets"])),
                   json.loads(r["phashes"]), r["thumbs_path"], r["width"], r["height"], r["duration_s"],
                   r["created_at"])


def record(job_id: str, sha256: str, filename: str, fp: Fingerprint, work_dir: Path) -> None:
    thumbs_path = None
    if fp.thumbs is not None:
        thumbs_path = str(work_dir / "provenance_thumbs.npz")
        np.savez_compressed(thumbs_path, t=fp.thumbs)
    with session() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO versions (job_id, sha256, filename, merkle_root, leaves, packets, phashes, "
            "thumbs_path, width, height, duration_s, trusted, label) VALUES (?,?,?,?,?,?,?,?,?,?,?,"
            "COALESCE((SELECT trusted FROM versions WHERE job_id = ?), 0), "
            "(SELECT label FROM versions WHERE job_id = ?))",
            (job_id, sha256, filename, fp.merkle_root, json.dumps(fp.leaves_json()), json.dumps(sorted(fp.packet_set)),
             json.dumps(fp.phashes), thumbs_path, fp.width, fp.height, fp.duration_s, job_id, job_id),
        )


def set_trusted(job_id: str, trusted: bool, label: Optional[str] = None) -> bool:
    with session() as conn:
        cur = conn.execute("UPDATE versions SET trusted = ?, label = COALESCE(?, label) WHERE job_id = ?",
                           (int(trusted), label, job_id))
        return cur.rowcount > 0


def get(job_id: str) -> Optional[Version]:
    with session() as conn:
        r = conn.execute("SELECT * FROM versions WHERE job_id = ?", (job_id,)).fetchone()
    return _row(r) if r else None


def by_sha(sha256: str, exclude_job: Optional[str] = None) -> list[Version]:
    with session() as conn:
        rows = conn.execute("SELECT * FROM versions WHERE sha256 = ? AND job_id != ? ORDER BY created_at",
                            (sha256, exclude_job or "")).fetchall()
    return [_row(r) for r in rows]


def all_versions(exclude_job: Optional[str] = None) -> list[Version]:
    with session() as conn:
        rows = conn.execute("SELECT * FROM versions WHERE job_id != ? ORDER BY created_at",
                            (exclude_job or "",)).fetchall()
    return [_row(r) for r in rows]
