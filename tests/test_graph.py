import uuid

from fastapi.testclient import TestClient

from backend import db
from backend.fusion import counterfactual, fuse, impacts
from backend.main import app
from backend.models import Finding, Segment


def S(a, b):
    return Segment(start=a, end=b, unit="s")


def findings():
    return [
        Finding(check="audio.hf_cutoff", modality="audio", stance="supports_manipulation", score=0.8, reliability=0.55, segment=S(2.4, 5.6)),
        Finding(check="audio.noise_floor", modality="audio", stance="supports_manipulation", score=0.75, reliability=0.55, segment=S(3.7, 5.2)),
        Finding(check="sync.lipsync", modality="sync", stance="supports_manipulation", score=0.8, reliability=0.5, segment=S(2.4, 5.8), depth="deep"),
        Finding(check="visual.temporal", modality="visual", stance="contradicts", score=0.2, reliability=0.35, segment=S(2, 4), depth="deep", coverage=0.25),
        Finding(check="metadata.timestamps", modality="metadata", stance="contradicts", score=0.5, reliability=0.5),
        Finding(check="metadata.encoder", modality="metadata", stance="supports_manipulation", score=0.3, reliability=0.5),
        Finding(check="provenance.c2pa", modality="provenance", stance="neutral", score=0.0, reliability=0.5),
        Finding(check="audio.spectral_flatness", modality="audio", stance="not_assessable", reliability=0.0),
    ]


def test_graph_edges_mirror_fusion():
    fs = findings()
    g = fuse(fs, media_type="video")["graph"]
    by = {f.id: f.check for f in fs}
    pair = {(by.get(e["source"]), e["relation"], by.get(e["target"])) for e in g["edges"]}
    assert ("audio.hf_cutoff", "redundant_with", "audio.noise_floor") in pair
    assert ("audio.hf_cutoff", "corroborates", "sync.lipsync") in pair
    assert ("metadata.timestamps", "conflicts", "metadata.encoder") in pair
    # opposite stances in different modalities are independent channels: no edge
    assert not any({a, b} == {"audio.hf_cutoff", "visual.temporal"} for a, _, b in pair)
    # every weighted finding points at its modality, every assessed modality at the verdict
    assert any(e["source"] == fs[0].id and e["target"] == "mod:audio" and e["relation"] == "supports" for e in g["edges"])
    assert any(e["source"] == "mod:audio" and e["target"] == "verdict" and e["relation"] == "supports" for e in g["edges"])
    ctx = {n["id"]: n["role"] for n in g["nodes"] if n["type"] == "finding"}
    assert ctx[fs[6].id] == "context" and ctx[fs[7].id] == "context"
    assert not any(e["source"] in (fs[6].id, fs[7].id) for e in g["edges"])


def test_counterfactual_removing_support_lowers_p_and_can_flip():
    fs = findings()
    sup = {f.id for f in fs if f.stance == "supports_manipulation"}
    cf = counterfactual(fs, sup, None, "video")
    assert cf["before"]["state"] == "likely manipulated"
    assert cf["after"]["probability_manipulated"] < cf["before"]["probability_manipulated"]
    assert cf["state_changed"] and len(cf["excluded"]) == len(sup)


def test_impacts_signs():
    fs = findings()
    imp = impacts(fs, None, "video")
    assert imp[fs[0].id]["delta_p"] < 0  # removing support lowers P
    assert imp[fs[4].id]["delta_p"] > 0  # removing clean evidence raises P
    assert fs[6].id not in imp and fs[7].id not in imp  # context nodes have no impact


def _seed_job() -> tuple[str, list[Finding]]:
    db.init_db()
    jid = uuid.uuid4().hex
    db.create_job(jid, "seed.mp4", "/nowhere", "0" * 64, 1)
    fs = findings()
    db.save_findings(jid, fs)
    db.update_job(jid, status="done", media_type="video", artifacts={"work_dir": "/nowhere"})
    return jid, fs


def test_api_graph_and_counterfactual():
    jid, fs = _seed_job()
    with TestClient(app) as c:
        g = c.get(f"/jobs/{jid}/graph").json()
        assert g["state"] == "likely manipulated"
        assert any("impact" in n for n in g["nodes"] if n["type"] == "finding")
        r = c.post(f"/jobs/{jid}/counterfactual", json={"exclude_modalities": ["audio", "sync"], "actor": "alice"})
        assert r.status_code == 200 and r.json()["state_changed"]
        assert c.get(f"/jobs/{jid}").json()["verdict"] is None  # stored verdict untouched (seeded without one)
        assert c.post(f"/jobs/{jid}/counterfactual", json={"exclude": ["nope"]}).status_code == 400
        assert c.post(f"/jobs/{jid}/counterfactual", json={}).status_code == 400
        actions = [e["action"] for e in c.get(f"/jobs/{jid}/custody").json()["entries"]]
        assert actions == ["counterfactual"]
