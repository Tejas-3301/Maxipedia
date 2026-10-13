"""Evidence graph and counterfactuals.

Graph (built from the same intermediate values fusion used, so it explains the verdict exactly):
  finding --supports|contradicts (weight = its effective llr contribution)--> modality
  modality --supports|contradicts (weight = modality log-odds vs prior)--> verdict
  finding <-> finding inside an overlap group: redundant_with (same modality, same stance: discounted),
  conflicts (same modality, opposite stances), corroborates (same stance, different modalities, localized
  in time/region). Opposite stances in different modalities get no edge (independent channels).
  neutral / not_assessable findings are context nodes without weighted edges.
Counterfactual: recompute the verdict with chosen findings removed ("remove this evidence").
"""
from __future__ import annotations

from typing import Any, Optional

from backend.models import Finding

VERDICT = "verdict"


def _sign_rel(llr: float) -> str:
    return "supports" if llr > 0 else "contradicts"


def build_graph(used: list[Finding], groups: list, contrib: dict[str, float], per_mod: dict[str, dict],
                p_manip: float, state: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [{"id": VERDICT, "type": "verdict", "label": state, "p_manipulated": round(p_manip, 4)}]
    edges: list[dict[str, Any]] = []
    for mod, v in per_mod.items():
        nodes.append({"id": f"mod:{mod}", "type": "modality", "label": mod, "posterior": v["posterior"],
                      "llr": v["llr"], "assessed": v["assessed"], "conflict": v["conflict"]})
        if v["assessed"] and v["llr"] != 0:
            edges.append({"source": f"mod:{mod}", "target": VERDICT, "relation": _sign_rel(v["llr"]),
                          "weight": v["llr"]})

    group_of = {f.id: g.id for g in groups for f in g.findings}
    for f in used:
        if f.check == "ingest_inventory":
            continue
        weight = round(contrib.get(f.id, 0.0), 3)
        nodes.append({"id": f.id, "type": "finding", "label": f.check, "modality": f.modality, "stance": f.stance,
                      "depth": f.depth, "score": f.score, "reliability": f.reliability, "coverage": f.coverage,
                      "segment": f.segment.model_dump() if f.segment else None,
                      "region": list(f.region) if f.region else None, "llr": weight,
                      "group": group_of.get(f.id), "role": "evidence" if f.id in contrib else "context"})
        if f.id in contrib and weight != 0:
            edges.append({"source": f.id, "target": f"mod:{f.modality}", "relation": _sign_rel(weight),
                          "weight": weight})

    # Pairwise edges mirror the fusion model: modalities are independent channels, so opposite stances in
    # different modalities are not a conflict (no edge); cross-modal agreement only counts when localized.
    for g in groups:
        fs = [f for f in g.findings if f.id in contrib]
        for i, a in enumerate(fs):
            for b in fs[i + 1:]:
                if a.modality == b.modality:
                    rel = "redundant_with" if a.stance == b.stance else "conflicts"
                elif a.stance == b.stance and g.start is not None:
                    rel = "corroborates"
                else:
                    continue
                edges.append({"source": a.id, "target": b.id, "relation": rel, "group": g.id,
                              "segment": None if g.start is None else {"start": g.start, "end": g.end, "unit": "s"}})

    for g in groups:
        nodes.append({"id": f"group:{g.id}", "type": "group", "label": "file-level" if g.start is None
                      else f"{g.start:.2f}-{g.end:.2f}s", "segment": None if g.start is None else
                      {"start": g.start, "end": g.end, "unit": "s"}, "region": g.region,
                      "members": [f.id for f in g.findings]})
    counts: dict[str, int] = {}
    for e in edges:
        counts[e["relation"]] = counts.get(e["relation"], 0) + 1
    return {"nodes": nodes, "edges": edges, "edge_counts": counts}


def _brief(a: dict[str, Any]) -> dict[str, Any]:
    return {"state": a["state"], "probability_manipulated": a["probability_manipulated"],
            "confidence": a["confidence"], "reasons": a["reasons"]}


def counterfactual(findings: list[Finding], exclude: set[str], schedule: Optional[dict[str, Any]],
                   media_type: Optional[str]) -> dict[str, Any]:
    from backend.fusion.fuse import fuse

    base = fuse(findings, schedule=schedule, media_type=media_type)
    alt = fuse(findings, exclude=exclude, schedule=schedule, media_type=media_type)
    removed = [f for f in findings if f.id in exclude]
    return {
        "excluded": [{"id": f.id, "check": f.check, "modality": f.modality, "stance": f.stance,
                      "segment": f.segment.model_dump() if f.segment else None} for f in removed],
        "before": _brief(base),
        "after": _brief(alt),
        "delta_p": round(alt["probability_manipulated"] - base["probability_manipulated"], 4),
        "state_changed": alt["state"] != base["state"],
        "assessment": alt,
    }


def impacts(findings: list[Finding], schedule: Optional[dict[str, Any]], media_type: Optional[str]) -> dict[str, dict]:
    """For every weighted finding: P and verdict if that single finding were removed."""
    from backend.fusion.fuse import fuse

    base = fuse(findings, schedule=schedule, media_type=media_type)
    weighted = {n["id"] for n in base["graph"]["nodes"] if n["type"] == "finding" and n["role"] == "evidence"}
    out = {}
    for fid in weighted:
        alt = fuse(findings, exclude={fid}, schedule=schedule, media_type=media_type)
        out[fid] = {"p_without": alt["probability_manipulated"], "state_without": alt["state"],
                    "delta_p": round(alt["probability_manipulated"] - base["probability_manipulated"], 4),
                    "flips_verdict": alt["state"] != base["state"]}
    return out
