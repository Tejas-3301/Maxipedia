from backend.fusion import fuse
from backend.fusion.fuse import PROV_MAX_DROP, STATE_AUTH, STATE_INC, STATE_MANIP, effective_llrs, group_findings
from backend.models import Finding, Segment


def F(check, modality, stance, score, rel, seg=None, region=None, depth="screen", coverage=None):
    return Finding(check=check, modality=modality, stance=stance, score=score, reliability=rel, depth=depth,
                   coverage=coverage, segment=Segment(start=seg[0], end=seg[1], unit="s") if seg else None,
                   region=region)


CLEAN = [F("m.a", "metadata", "contradicts", 0.5, 0.5), F("m.b", "metadata", "contradicts", 0.4, 0.5),
         F("v.a", "visual", "contradicts", 0.5, 0.5), F("a.a", "audio", "contradicts", 0.5, 0.5),
         F("s.a", "sync", "contradicts", 0.5, 0.5)]


def P(fs, **kw):
    return fuse(fs, media_type="video", **kw)["probability_manipulated"]


def test_no_findings_is_insufficient():
    out = fuse([F("x", "visual", "neutral", 0.0, 0.3), F("y", "audio", "not_assessable", None, 0.0)])
    assert out["state"] == STATE_INC and "insufficient" in out["reasons"][0]


def test_consistent_clean_checks_give_authentic():
    out = fuse(CLEAN, media_type="video")
    assert out["state"] == STATE_AUTH and out["confidence"] > 0.8


def test_monotone_contradicts_only_lower_supports_only_raise():
    base = P(CLEAN)
    assert P(CLEAN + [F("p.x", "provenance", "contradicts", 0.3, 0.4)]) < base  # new modality, clean -> lower
    assert P(CLEAN + [F("v.b", "visual", "contradicts", 0.2, 0.3)]) < base
    assert P(CLEAN + [F("v.c", "visual", "supports_manipulation", 0.2, 0.3)]) > base
    assert P(CLEAN + [F("p.y", "provenance", "supports_manipulation", 0.2, 0.3)]) > base


def test_neutral_and_not_assessable_move_nothing():
    base = P(CLEAN)
    assert P(CLEAN + [F("p.n", "provenance", "neutral", 0.9, 0.9), F("p.na", "provenance", "not_assessable", None, 0.0),
                      F("s.na", "sync", "not_assessable", None, 0.0)]) == base


def test_clean_deep_evidence_scoped_by_coverage_and_miss_rate():
    full = F("v.t", "visual", "contradicts", 0.5, 0.5, (0, 10), depth="deep", coverage=1.0)
    part = F("v.t", "visual", "contradicts", 0.5, 0.5, (0, 3), depth="deep", coverage=0.3)
    e_full = effective_llrs([full], miss_rate=0.0)[full.id]
    assert effective_llrs([part], miss_rate=0.0)[part.id] == pytest_approx(0.3 * e_full)
    assert effective_llrs([full], miss_rate=0.25)[full.id] == pytest_approx(0.75 * e_full)  # capped by 1 - miss


def test_supports_not_discounted_by_coverage():
    s = F("v.n", "visual", "supports_manipulation", 0.8, 0.5, (3, 4), depth="deep", coverage=0.1)
    assert effective_llrs([s], miss_rate=0.5)[s.id] == pytest_approx(5 * 0.8 * 0.5)


def test_low_coverage_without_strong_contradiction_is_inconclusive():
    sched = {"coverage": 0.3, "unanalysed": [], "miss_rate_est": 0.2, "analysed": [], "elapsed_s": 20, "budget_s": 20}
    out = fuse(CLEAN, schedule=sched, media_type="video")
    assert out["state"] == STATE_INC and any("analysed in depth" in r for r in out["reasons"])
    strong = F("p.lineage", "provenance", "contradicts", 0.8, 0.8)  # e.g. Merkle root equals a trusted original
    assert fuse(CLEAN + [strong], schedule=sched, media_type="video")["state"] == STATE_AUTH


def test_unanalysed_flagged_interval_blocks_authentic():
    sched = {"coverage": 0.9, "unanalysed": [{"start": 5, "end": 7, "kind": "flagged"}], "miss_rate_est": 0.2,
             "analysed": [], "elapsed_s": 20, "budget_s": 20}
    out = fuse(CLEAN, schedule=sched, media_type="video")
    assert out["state"] == STATE_INC and any("flagged" in r for r in out["reasons"])


def test_provenance_single_finding_cap():
    huge = F("p.sha", "provenance", "contradicts", 1.0, 1.0)
    assert effective_llrs([huge], 0.0)[huge.id] == -PROV_MAX_DROP


def test_other_modalities_do_not_cancel_a_strong_detection():
    strong = [F("a.hf", "audio", "supports_manipulation", 0.8, 0.55, (2, 5)),
              F("a.floor", "audio", "supports_manipulation", 0.75, 0.55, (3, 5)),
              F("s.sync", "sync", "supports_manipulation", 0.8, 0.5, (2, 5))]
    others = [f for f in CLEAN if f.modality not in ("audio", "sync")] + [F("v.b", "visual", "contradicts", 0.3, 0.5)]
    out = fuse(others + strong, media_type="video")
    assert out["state"] == STATE_MANIP


def test_single_modality_cannot_be_called_authentic():
    out = fuse([F("m.a", "metadata", "contradicts", 0.9, 0.9), F("m.b", "metadata", "contradicts", 0.9, 0.9),
                F("m.c", "metadata", "contradicts", 0.9, 0.9)], media_type="video")
    # unassessed modalities stay at their prior, so one modality alone cannot get P under T_AUTH
    assert out["state"] == STATE_INC and out["probability_manipulated"] > 0.2


def test_conflict_within_modality_forces_inconclusive():
    out = fuse(CLEAN + [F("v.x", "visual", "supports_manipulation", 0.9, 0.9, (1, 2)),
                        F("v.y", "visual", "contradicts", 0.9, 0.9, (1, 2))], media_type="video")
    assert out["modalities"]["visual"]["conflict"] and out["state"] != STATE_MANIP


def test_same_modality_overlap_is_discounted():
    a = F("v.noise", "visual", "supports_manipulation", 0.8, 0.5, (3, 4.5), [384, 48, 176, 144])
    b = F("v.ela", "visual", "supports_manipulation", 0.7, 0.4, (3, 4.5), [384, 48, 176, 144])
    c = F("v.other", "visual", "supports_manipulation", 0.7, 0.4, (3, 4.5), [0, 300, 50, 50])
    assert len(group_findings([a, b, c])) == 2
    out = fuse([a, b])
    assert abs(out["modalities"]["visual"]["llr"] - (5 * 0.8 * 0.5 + 0.5 * 5 * 0.7 * 0.4)) < 1e-6


def test_exclude_removes_evidence():
    strong = F("a.hf", "audio", "supports_manipulation", 0.95, 0.9, (2, 5))
    s2 = F("s.x", "sync", "supports_manipulation", 0.9, 0.9, (2, 5))
    assert fuse(CLEAN + [strong, s2], media_type="video")["state"] == STATE_MANIP
    assert fuse(CLEAN + [strong, s2], exclude={strong.id, s2.id}, media_type="video")["state"] == STATE_AUTH


def pytest_approx(x):
    import pytest
    return pytest.approx(x, rel=1e-6)
