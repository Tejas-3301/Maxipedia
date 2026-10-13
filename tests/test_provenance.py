import shutil
import uuid
from pathlib import Path

import pytest

from backend import db
from backend.provenance import custody
from backend.provenance.fingerprint import fingerprint, merkle_proof, merkle_root, verify_proof

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "testset"
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
needs_testset = pytest.mark.skipif(not (TESTSET / "06_derived_a.mp4").exists(),
                                   reason="run scripts/make_testset.py first")


def test_merkle_proofs_verify_and_detect_tamper():
    leaves = [f"{i:064x}" for i in range(7)]
    root = merkle_root(leaves)
    for i in range(len(leaves)):
        assert verify_proof(leaves[i], merkle_proof(leaves, i), root)
    assert not verify_proof(f"{99:064x}", merkle_proof(leaves, 3), root)


def _job() -> str:
    db.init_db()
    jid = uuid.uuid4().hex
    db.create_job(jid, "x.mp4", "/nowhere", "0" * 64, 1)
    return jid


def test_custody_chain_signed_and_tamper_evident():
    jid = _job()
    custody.append(jid, "ingested", {"sha256": "0" * 64})
    custody.append(jid, "annotation", {"note": "checked by A"}, actor="alice")
    custody.append(jid, "analyzed", {"verdict": "inconclusive"})
    assert custody.verify(jid)["intact"]

    with db.session() as conn:  # edit a past entry
        conn.execute("UPDATE custody_log SET detail = ? WHERE job_id = ? AND action = 'annotation'",
                     ('{"note":"checked by B"}', jid))
    v = custody.verify(jid)
    assert not v["intact"] and any("altered" in p for p in v["problems"])


def test_custody_detects_deleted_entry():
    jid = _job()
    for a in ("ingested", "preprocessed", "analyzed"):
        custody.append(jid, a, {})
    with db.session() as conn:
        conn.execute("DELETE FROM custody_log WHERE job_id = ? AND action = 'preprocessed'", (jid,))
    v = custody.verify(jid)
    assert not v["intact"] and any("prev_hash" in p for p in v["problems"])


@needs_ffmpeg
@needs_testset
def test_metadata_only_copy_has_identical_merkle_root():
    a, b = fingerprint(TESTSET / "01_clean.mp4"), fingerprint(TESTSET / "05_metadata_edited.mp4")
    assert a.merkle_root == b.merkle_root


@needs_ffmpeg
@needs_testset
def test_derived_copies_lineage():
    from backend.provenance.lineage import relate

    fps = {n: fingerprint(TESTSET / f"{n}.mp4") for n in ("01_clean", "06_derived_a", "07_derived_b")}
    r06 = relate("06", fps["06_derived_a"].phashes, fps["06_derived_a"].thumbs,
                 "01", fps["01_clean"].phashes, fps["01_clean"].thumbs)
    assert r06 and r06.offset_changes == []
    assert abs(r06.start_offset_s - 1.0) <= 0.1 and abs(r06.coverage[1] - 7.0) <= 0.1  # trim 1-7 s

    r07 = relate("07", fps["07_derived_b"].phashes, fps["07_derived_b"].thumbs,
                 "06", fps["06_derived_a"].phashes, fps["06_derived_a"].thumbs)
    assert r07 and r07.offset_changes == [] and r07.crop > 1.05  # 07 is a crop of 06


@needs_ffmpeg
@needs_testset
def test_cut_is_found_by_alignment():
    from backend.provenance.lineage import relate

    a, b = fingerprint(TESTSET / "03_splice.mp4"), fingerprint(TESTSET / "01_clean.mp4")
    r = relate("03", a.phashes, a.thumbs, "01", b.phashes, b.thumbs)
    assert r.offset_changes == [{"at_s": 6.0, "removed_s": 0.4}]
