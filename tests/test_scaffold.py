from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.checks.registry import JobContext, _REGISTRY, register, run_checks
from backend.db import connect, init_db
from backend.main import app
from backend.models import Finding, Modality


def test_finding_defaults_and_schema_keys():
    f = Finding(modality="visual", check="ela")
    assert set(f.model_dump()) >= {
        "id", "modality", "check", "depth", "segment", "region", "score",
        "reliability", "stance", "rationale", "limitations", "evidence",
    }
    assert f.stance == "not_assessable" and f.depth == "screen"


def test_finding_rejects_bad_enum():
    with pytest.raises(ValidationError):
        Finding(modality="smell", check="x")


def test_db_tables(tmp_path: Path):
    db = tmp_path / "t.db"
    init_db(db)
    with connect(db) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"jobs", "findings", "custody_log"} <= names


def test_registry_fallback_on_crash(tmp_path: Path):
    @register("_boom", Modality.audio, ("audio",))
    def boom(ctx):
        raise RuntimeError("nope")

    try:
        out = run_checks(JobContext("j1", tmp_path / "a.wav", "audio", tmp_path))
        boom_out = [f for f in out if f.check == "_boom"]
        assert len(boom_out) == 1 and boom_out[0].stance == "not_assessable"
    finally:
        _REGISTRY.pop("_boom", None)


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200 and r.json()["db"] is True
