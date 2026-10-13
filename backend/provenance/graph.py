"""Lineage graph over the ledger: stored pairwise relations -> nodes + directed parent edges."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Optional

from backend.db import session
from backend.provenance import ledger
from backend.provenance.lineage import Relation


def save_relation(a: str, b: str, rel: Relation) -> None:
    with session() as conn:
        conn.execute("INSERT OR REPLACE INTO lineage_edges (a, b, relation) VALUES (?,?,?)",
                     (a, b, json.dumps(asdict(rel))))


def relations() -> dict[tuple[str, str], Relation]:
    with session() as conn:
        rows = conn.execute("SELECT a, b, relation FROM lineage_edges").fetchall()
    out = {}
    for r in rows:
        d = json.loads(r["relation"])
        d["coverage"] = tuple(d["coverage"])
        out[(r["a"], r["b"])] = Relation(**d)
    return out


def _ancestor_like(child: ledger.Version, parent: ledger.Version, rel: Relation) -> bool:
    cw, pw = child.width or 0, parent.width or 0
    cd, pd = child.duration_s or 0, parent.duration_s or 0
    return parent.trusted or pw > cw or pd > cd + 0.2 or rel.crop > 1.0


def _closeness(child: ledger.Version, parent: ledger.Version, rel: Relation) -> tuple:
    """Smaller = more plausible direct parent: a consistent timeline first (no cuts/insertions relative to
    it), then visual similarity, then the nearest generation step (length, size), trust as tie-break."""
    ratio = (parent.width or 1) / max(child.width or 1, 1)
    return (len(rel.offset_changes), round(rel.median_bits / 4),
            round(abs((parent.duration_s or 0) - (child.duration_s or 0)), 1), round(abs(ratio - 1), 2),
            0 if parent.trusted else 1)


def parent_of(job_id: str, rels: Optional[dict] = None,
              versions: Optional[dict[str, ledger.Version]] = None) -> Optional[tuple[str, Relation]]:
    rels = rels if rels is not None else relations()
    versions = versions or {v.job_id: v for v in ledger.all_versions()}
    me = versions.get(job_id)
    if me is None or me.trusted:
        return None
    cands = []
    for (a, b), rel in rels.items():
        if a == job_id and b in versions and _ancestor_like(me, versions[b], rel):
            cands.append((_closeness(me, versions[b], rel), b, rel))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][2]


def trusted_relative(job_id: str) -> Optional[tuple[str, Relation]]:
    """Nearest *trusted* version this file is a near-duplicate of (if any). Findings are judged against it so trust
    is not lost when a closer, untrusted relative exists (e.g. an attacked copy of an already-seen file)."""
    versions = {v.job_id: v for v in ledger.all_versions()}
    me = versions.get(job_id)
    if me is None or me.trusted:
        return None
    cands = [(_closeness(me, versions[b], rel), b, rel) for (a, b), rel in relations().items()
             if a == job_id and b in versions and versions[b].trusted]
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][2]


def build(job_id: Optional[str] = None) -> dict[str, Any]:
    """Whole lineage graph, or the connected component containing job_id."""
    versions = {v.job_id: v for v in ledger.all_versions()}
    rels = relations()
    edges, similar = [], []
    for jid in versions:
        p = parent_of(jid, rels, versions)
        if p:
            parent, rel = p
            edges.append({"child": jid, "parent": parent, "kind": "derived_from", "relation": asdict(rel)})
    directed = {(e["child"], e["parent"]) for e in edges} | {(e["parent"], e["child"]) for e in edges}
    for (a, b), rel in rels.items():
        if a < b and (a, b) not in directed:
            similar.append({"a": a, "b": b, "kind": "near_duplicate", "median_bits": rel.median_bits})

    nodes = set(versions)
    if job_id:
        adj: dict[str, set[str]] = {}
        for e in edges:
            adj.setdefault(e["child"], set()).add(e["parent"])
            adj.setdefault(e["parent"], set()).add(e["child"])
        for e in similar:
            adj.setdefault(e["a"], set()).add(e["b"])
            adj.setdefault(e["b"], set()).add(e["a"])
        seen, stack = set(), [job_id]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(adj.get(n, ()))
        nodes = seen & set(versions)
        edges = [e for e in edges if e["child"] in nodes]
        similar = [e for e in similar if e["a"] in nodes]
    return {"nodes": [versions[n].brief() for n in sorted(nodes, key=lambda n: versions[n].created_at)],
            "edges": edges, "near_duplicates": similar}
