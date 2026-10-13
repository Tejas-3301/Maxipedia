from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from backend.config import DB_PATH
from backend.models import Finding

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    media_path    TEXT NOT NULL,
    media_type    TEXT,
    sha256        TEXT,
    status        TEXT NOT NULL DEFAULT 'queued',
    verdict       TEXT,
    confidence    REAL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS findings (
    id            TEXT PRIMARY KEY,
    job_id        TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    modality      TEXT NOT NULL,
    "check"       TEXT NOT NULL,
    depth         TEXT NOT NULL,
    stance        TEXT NOT NULL,
    score         REAL,
    reliability   REAL NOT NULL,
    data          TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_findings_job ON findings(job_id);

CREATE TABLE IF NOT EXISTS custody_log (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id        TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    action        TEXT NOT NULL,
    actor         TEXT NOT NULL DEFAULT 'system',
    detail        TEXT,
    prev_hash     TEXT,
    entry_hash    TEXT NOT NULL,
    signature     TEXT,
    ts            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_custody_job ON custody_log(job_id);

CREATE TABLE IF NOT EXISTS versions (
    job_id        TEXT PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
    sha256        TEXT NOT NULL,
    filename      TEXT,
    merkle_root   TEXT NOT NULL,
    leaves        TEXT NOT NULL,
    packets       TEXT NOT NULL,
    phashes       TEXT NOT NULL,
    thumbs_path   TEXT,
    width         INTEGER,
    height        INTEGER,
    duration_s    REAL,
    trusted       INTEGER NOT NULL DEFAULT 0,
    label         TEXT,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_versions_sha ON versions(sha256);

CREATE TABLE IF NOT EXISTS lineage_edges (
    a             TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    b             TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    relation      TEXT NOT NULL,
    PRIMARY KEY (a, b)
);
"""

# Additive column migrations: (table, column, type)
MIGRATIONS = [
    ("jobs", "size_bytes", "INTEGER"),
    ("jobs", "duration_s", "REAL"),
    ("jobs", "artifacts", "TEXT"),
    ("jobs", "error", "TEXT"),
    ("custody_log", "scheme", "INTEGER"),
    ("custody_log", "key_id", "TEXT"),
    ("jobs", "derived_from", "TEXT"),
    ("jobs", "attack", "TEXT"),
]


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def session(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with session(path) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        for table, col, typ in MIGRATIONS:
            cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def create_job(job_id: str, filename: str, media_path: str, sha256: str, size_bytes: int) -> None:
    with session() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, media_path, sha256, size_bytes) VALUES (?,?,?,?,?)",
            (job_id, filename, media_path, sha256, size_bytes),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    for key in ("artifacts", "attack"):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key])
    cols = ", ".join(f"{k} = ?" for k in fields)
    with session() as conn:
        conn.execute(
            f"UPDATE jobs SET {cols}, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
            (*fields.values(), job_id),
        )


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with session() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        job = dict(row)
        job["findings_count"] = conn.execute(
            "SELECT COUNT(*) FROM findings WHERE job_id = ?", (job_id,)
        ).fetchone()[0]
    job["artifacts"] = json.loads(job["artifacts"]) if job.get("artifacts") else None
    job["attack"] = json.loads(job["attack"]) if job.get("attack") else None
    return job


def list_jobs(limit: int = 50) -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute(
            "SELECT id, filename, media_type, status, verdict, confidence, created_at, derived_from FROM jobs "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def save_findings(job_id: str, findings: list[Finding]) -> None:
    with session() as conn:
        conn.executemany(
            'INSERT INTO findings (id, job_id, modality, "check", depth, stance, score, reliability, data) '
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (f.id, job_id, f.modality, f.check, f.depth, f.stance, f.score, f.reliability, f.model_dump_json())
                for f in findings
            ],
        )


def replace_findings(job_id: str, checks: list[str], findings: list[Finding]) -> None:
    """Atomically swap all findings of the named checks (used by re-runs such as 'analyze deeper')."""
    with session() as conn:
        conn.executemany('DELETE FROM findings WHERE job_id = ? AND "check" = ?', [(job_id, c) for c in checks])
        conn.executemany(
            'INSERT INTO findings (id, job_id, modality, "check", depth, stance, score, reliability, data) '
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(f.id, job_id, f.modality, f.check, f.depth, f.stance, f.score, f.reliability, f.model_dump_json())
             for f in findings],
        )


def get_findings(job_id: str) -> list[Finding]:
    with session() as conn:
        rows = conn.execute(
            "SELECT data FROM findings WHERE job_id = ? ORDER BY created_at, rowid", (job_id,)
        ).fetchall()
    return [Finding.model_validate_json(r["data"]) for r in rows]


def children(job_id: str) -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute("SELECT id FROM jobs WHERE derived_from = ? ORDER BY created_at", (job_id,)).fetchall()
    return [get_job(r["id"]) for r in rows]


def delete_findings(job_id: str) -> None:
    with session() as conn:
        conn.execute("DELETE FROM findings WHERE job_id = ?", (job_id,))


def jobs_with_status(statuses: tuple[str, ...]) -> list[dict[str, Any]]:
    marks = ",".join("?" * len(statuses))
    with session() as conn:
        rows = conn.execute(f"SELECT id, status FROM jobs WHERE status IN ({marks})", statuses).fetchall()
    return [dict(r) for r in rows]
