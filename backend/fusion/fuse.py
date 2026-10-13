"""Evidence fusion: grouping by overlap, per-modality Bayesian log-odds, noisy-OR across modalities.

Model
- Each assessable finding contributes a log-likelihood ratio llr = ±K · reliability · score
  (+ supports_manipulation, − contradicts). neutral / not_assessable contribute nothing.
- Partial coverage: a deep "contradicts" finding is clean evidence only for the interval it analysed, so its llr
  is multiplied by its coverage (interval share of the clip). Per check, the summed coverage of its clean
  findings is capped at 1 − (estimated screening miss rate from control intervals). supports_manipulation
  findings are never discounted.
- Provenance: a single provenance finding can lower the log-odds by at most PROV_MAX_DROP.
- Findings overlapping in time (and region, when both have one) form a group. Inside a group, findings of
  the same modality are partly redundant: the strongest counts fully, the rest at REDUNDANCY.
- Every modality that applies to the media type starts at prior P0 (P0 chosen so the file-level prior is
  FILE_PRIOR); its posterior is σ(logit(P0) + Σ llr). P(manipulated) = 1 − Π(1 − P_m) over applicable modalities.
  Unassessed modalities stay at their prior, so contradicting evidence can only lower P and supporting evidence
  can only raise it (monotone); absence of visual edits cannot cancel an audio splice.
- States: likely manipulated (P >= T_MANIP); likely authentic (P <= T_AUTH, >= MIN_MODALITIES assessed, and
  enough deep coverage with no flagged interval left unanalysed — unless a single strong contradicting finding
  exists, and the media is not too degraded: heavy compression / noise / downscaling erase traces); otherwise
  inconclusive. Insufficient evidence and within-modality conflict force inconclusive.
Thresholds are fixed constants, not fitted per file. The fusion step is the only place a verdict is made.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.fusion.graph import build_graph
from backend.models import Finding

K = 5.0
FILE_PRIOR = 0.25
REDUNDANCY = 0.5
T_MANIP = 0.70
T_AUTH = 0.20
MIN_MODALITIES = 2
MIN_ASSESSABLE = 3  # fewer assessable findings than this = insufficient evidence
CONFLICT_LLR = 1.5
TIME_TOL_S = 0.25
MIN_AUTH_COVERAGE = 0.6  # deep coverage needed to call a video authentic without strong contradicting evidence
STRONG_CONTRA_LLR = 2.0
PROV_MAX_DROP = 3.0
MODALITIES = ["visual", "audio", "sync", "metadata", "provenance"]
APPLICABLE = {"video": MODALITIES, "audio": ["audio", "metadata", "provenance"],
              "image": ["visual", "metadata", "provenance"], "other": ["metadata", "provenance"]}

STATE_MANIP = "likely manipulated"
STATE_AUTH = "likely authentic"
STATE_INC = "inconclusive"


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def finding_llr(f: Finding) -> float:
    """Raw llr before coverage/caps: ±K · reliability · score."""
    s, r = f.score or 0.0, f.reliability or 0.0
    if f.stance == "supports_manipulation":
        return K * r * s
    if f.stance == "contradicts":
        return -K * r * s
    return 0.0


def effective_llrs(findings: list[Finding], miss_rate: float) -> dict[str, float]:
    """Raw llr with partial-coverage scoping of clean evidence and the provenance cap applied."""
    cap = max(0.0, 1.0 - miss_rate)
    cov_sum: dict[str, float] = {}
    for f in findings:
        if f.stance == "contradicts" and f.depth == "deep" and f.coverage is not None:
            cov_sum[f.check] = cov_sum.get(f.check, 0.0) + f.coverage
    out = {}
    for f in findings:
        llr = finding_llr(f)
        if llr < 0 and f.depth == "deep" and f.coverage is not None:
            scale = min(1.0, cap / cov_sum[f.check]) if cov_sum.get(f.check) else 1.0
            llr *= f.coverage * scale
        if llr < 0 and f.modality == "provenance":
            llr = max(llr, -PROV_MAX_DROP)
        out[f.id] = llr
    return out


def _overlaps(a: Finding, b: Finding) -> bool:
    if a.segment is None or b.segment is None:
        return a.segment is None and b.segment is None  # all file-level findings share one group
    if a.segment.start > b.segment.end + TIME_TOL_S or b.segment.start > a.segment.end + TIME_TOL_S:
        return False
    if a.region and b.region:
        ax, ay, aw, ah = a.region
        bx, by, bw, bh = b.region
        return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
    return True


@dataclass
class Group:
    id: int
    findings: list[Finding]
    start: Optional[float] = None
    end: Optional[float] = None
    region: Optional[list[int]] = None
    llr_by_modality: dict[str, float] = field(default_factory=dict)

    def to_dict(self, contrib: dict[str, float]) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": "file" if self.start is None else "segment",
            "segment": None if self.start is None else {"start": self.start, "end": self.end, "unit": "s"},
            "region": self.region,
            "modalities": sorted({f.modality for f in self.findings}),
            "llr_by_modality": {k: round(v, 3) for k, v in self.llr_by_modality.items()},
            "findings": [{"id": f.id, "check": f.check, "stance": f.stance, "llr": round(contrib[f.id], 3)}
                         for f in self.findings],
        }


def group_findings(findings: list[Finding]) -> list[Group]:
    parent = list(range(len(findings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(findings)):
        for j in range(i + 1, len(findings)):
            if _overlaps(findings[i], findings[j]):
                parent[find(i)] = find(j)
    buckets: dict[int, list[Finding]] = {}
    for i, f in enumerate(findings):
        buckets.setdefault(find(i), []).append(f)

    groups = []
    for gid, members in enumerate(sorted(buckets.values(), key=lambda m: (m[0].segment is not None,
                                                                         m[0].segment.start if m[0].segment else 0))):
        segs = [f.segment for f in members if f.segment]
        regs = [f.region for f in members if f.region]
        g = Group(gid, members)
        if segs:
            g.start, g.end = round(min(s.start for s in segs), 3), round(max(s.end for s in segs), 3)
        if regs:
            x0, y0 = min(r[0] for r in regs), min(r[1] for r in regs)
            x1, y1 = max(r[0] + r[2] for r in regs), max(r[1] + r[3] for r in regs)
            g.region = [x0, y0, x1 - x0, y1 - y0]
        groups.append(g)
    return groups


def _combine(values: list[float]) -> float:
    """Strongest counts fully, the rest at REDUNDANCY (same modality, same group = correlated)."""
    vals = sorted(values, key=abs, reverse=True)
    return vals[0] + REDUNDANCY * sum(vals[1:]) if vals else 0.0


def coverage_report(findings: list[Finding], schedule: Optional[dict[str, Any]]) -> dict[str, Any]:
    sched = schedule or {}
    sync_cov = max((f.evidence.get("usable_coverage") or 0.0 for f in findings if f.check == "sync.lipsync"),
                   default=None)
    unanalysed = sched.get("unanalysed", [])
    return {
        "deep_coverage": sched.get("coverage"),
        "sync_usable_coverage": sync_cov,
        "deep_elapsed_s": sched.get("elapsed_s"),
        "budget_s": sched.get("budget_s"),
        "analysed_intervals": [{k: i[k] for k in ("start", "end", "kind")} for i in sched.get("analysed", [])],
        "unanalysed_intervals": [{k: i[k] for k in ("start", "end", "kind")} for i in unanalysed],
        "unanalysed_flagged": sum(i["kind"] in ("flagged", "manual") for i in unanalysed),
        "controls": sched.get("controls_analysed", 0),
        "control_hits": sched.get("control_hits", 0),
        "miss_rate_est": sched.get("miss_rate_est"),
        "screen_findings": sum(f.check.startswith("screen.") for f in findings),
        "deep_findings": sum(f.depth == "deep" for f in findings),
    }


def fuse(findings: list[Finding], exclude: Optional[set[str]] = None, schedule: Optional[dict[str, Any]] = None,
         media_type: Optional[str] = None) -> dict[str, Any]:
    exclude = exclude or set()
    used = [f for f in findings if f.id not in exclude]
    miss_rate = float((schedule or {}).get("miss_rate_est") or 0.0)
    eff = effective_llrs(used, miss_rate)
    assessable = [f for f in used if f.stance in ("supports_manipulation", "contradicts") and eff[f.id] != 0]
    not_assessable = [f for f in used if f.stance == "not_assessable" and f.check != "ingest_inventory"]

    groups = group_findings(assessable)
    contrib: dict[str, float] = {}
    llr_mod: dict[str, float] = {}
    for g in groups:
        for mod in {f.modality for f in g.findings}:
            sup = [eff[f.id] for f in g.findings if f.modality == mod and eff[f.id] > 0]
            con = [eff[f.id] for f in g.findings if f.modality == mod and eff[f.id] < 0]
            g.llr_by_modality[mod] = _combine(sup) + _combine(con)
            llr_mod[mod] = llr_mod.get(mod, 0.0) + g.llr_by_modality[mod]
            for sign in (1, -1):
                ordered = sorted((f for f in g.findings if f.modality == mod and sign * eff[f.id] > 0),
                                 key=lambda f: abs(eff[f.id]), reverse=True)
                for k, f in enumerate(ordered):
                    contrib[f.id] = eff[f.id] * (1.0 if k == 0 else REDUNDANCY)

    applicable = list(APPLICABLE.get(media_type or "", MODALITIES))
    applicable += [m for m in llr_mod if m not in applicable]
    p0 = 1 - (1 - FILE_PRIOR) ** (1 / len(applicable))
    per_mod = {}
    for mod in applicable:
        llr = llr_mod.get(mod, 0.0)
        sup = sum(v for f in assessable if f.modality == mod and (v := contrib[f.id]) > 0)
        con = -sum(v for f in assessable if f.modality == mod and (v := contrib[f.id]) < 0)
        per_mod[mod] = {"llr": round(llr, 3), "posterior": round(_sigmoid(_logit(p0) + llr), 4),
                        "support_llr": round(sup, 3), "contradict_llr": round(con, 3),
                        "assessed": mod in llr_mod,
                        "conflict": sup >= CONFLICT_LLR and con >= CONFLICT_LLR}

    p_manip = 1 - math.prod(1 - m["posterior"] for m in per_mod.values())
    weight = sum((f.reliability or 0) * (f.score or 0) for f in assessable)
    assessed = sorted(m for m, v in per_mod.items() if v["assessed"])
    missing = [m for m in applicable if m not in assessed]
    conflicts = [m for m, v in per_mod.items() if v["conflict"]]
    cov = coverage_report(used, schedule)
    strong_contra = [f for f in assessable if eff[f.id] <= -STRONG_CONTRA_LLR]

    reasons: list[str] = []
    if len(assessable) < MIN_ASSESSABLE:
        state, reasons = STATE_INC, ["insufficient evidence: too few assessable findings"]
    elif conflicts and p_manip >= T_MANIP and all(per_mod[m]["posterior"] >= 0.5 for m in conflicts):
        state, reasons = STATE_INC, [f"conflicting evidence within {', '.join(conflicts)}"]
    elif p_manip >= T_MANIP:
        state = STATE_MANIP
    elif p_manip <= T_AUTH and len(assessed) >= MIN_MODALITIES:
        state = STATE_AUTH
        if schedule is not None and not strong_contra:
            if cov["deep_coverage"] is not None and cov["deep_coverage"] < MIN_AUTH_COVERAGE:
                state = STATE_INC
                reasons.append(f"only {cov['deep_coverage']:.0%} of the clip was analysed in depth "
                               f"(< {MIN_AUTH_COVERAGE:.0%}) and no strong contradicting evidence exists")
            if cov["unanalysed_flagged"]:
                state = STATE_INC
                reasons.append(f"{cov['unanalysed_flagged']} flagged interval(s) were not analysed before the deadline")
        degraded = [r for f in used if f.check == "quality.media" and f.evidence.get("degraded")
                    for r in f.evidence.get("reasons", [])]
        if degraded and not strong_contra:
            state = STATE_INC
            reasons.append("media too degraded for absent traces to vouch for authenticity: " + "; ".join(degraded))
    else:
        state = STATE_INC
        if p_manip <= T_AUTH:
            reasons.append(f"only {len(assessed)} modality assessed; at least {MIN_MODALITIES} needed to call authentic")
        else:
            reasons.append(f"P(manipulated)={p_manip:.2f} lies between {T_AUTH} and {T_MANIP}")

    confidence = (p_manip if state == STATE_MANIP else 1 - p_manip if state == STATE_AUTH else None)
    by_strength = sorted(assessable, key=lambda f: contrib.get(f.id, 0.0))
    supporting = [f for f in reversed(by_strength) if contrib[f.id] > 0][:5]
    contradicting = [f for f in by_strength if contrib[f.id] < 0][:5]

    limitations = []
    if missing:
        limitations.append(f"Not assessed: {', '.join(missing)} (no assessable findings; held at prior).")
    if not_assessable:
        limitations.append("Not assessable: " + "; ".join(
            sorted({f"{f.check} ({f.rationale[:90]})" for f in not_assessable})[:8]))
    if cov["deep_coverage"] is not None:
        line = (f"Deep analysis covered {cov['deep_coverage']:.0%} of the clip in {cov['deep_elapsed_s']}s "
                f"(deadline {cov['budget_s']}s); clean deep findings count only for their own intervals.")
        if cov["unanalysed_intervals"]:
            line += " Not analysed: " + ", ".join(f"{i['start']:.1f}-{i['end']:.1f}s ({i['kind']})"
                                                  for i in cov["unanalysed_intervals"][:10]) + "."
        limitations.append(line)
        limitations.append(f"Control intervals: {cov['controls']} analysed, {cov['control_hits']} with deep findings; "
                           f"estimated screening miss rate {cov['miss_rate_est']:.0%} caps clean evidence.")
    if cov["sync_usable_coverage"] is not None:
        limitations.append(f"Lip sync usable (face + speech) on {cov['sync_usable_coverage']:.0%} of the clip; "
                           f"not-assessable windows count neither as clean evidence nor as coverage.")
    for r in reasons:
        limitations.append(f"Verdict held back: {r}.")
    limitations.extend(sorted({f.limitations for f in supporting if f.limitations})[:4])
    limitations.append("Thresholds, prior and reliability weights are hand-set, not calibrated on a large corpus.")

    def brief(f: Finding) -> dict[str, Any]:
        return {"id": f.id, "check": f.check, "modality": f.modality, "stance": f.stance,
                "score": f.score, "reliability": f.reliability, "coverage": f.coverage,
                "llr": round(contrib.get(f.id, 0.0), 3), "segment": f.segment.model_dump() if f.segment else None,
                "region": list(f.region) if f.region else None, "rationale": f.rationale}

    return {
        "state": state,
        "probability_manipulated": round(p_manip, 4),
        "confidence": None if confidence is None else round(confidence, 4),
        "reasons": reasons,
        "summary": _summary(state, p_manip, supporting, contradicting, reasons, assessed),
        "modalities": per_mod,
        "assessed_modalities": assessed,
        "supporting": [brief(f) for f in supporting],
        "contradicting": [brief(f) for f in contradicting],
        "groups": [g.to_dict(contrib) for g in groups],
        "graph": build_graph(used, groups, contrib, per_mod, p_manip, state),
        "limitations": limitations,
        "evidence_weight": round(weight, 3),
        "coverage": cov,
        "counts": {"findings": len(used), "assessable": len(assessable), "not_assessable": len(not_assessable),
                   "excluded": len(findings) - len(used)},
        "model": {"K": K, "file_prior": FILE_PRIOR, "prior_per_modality": round(p0, 4), "redundancy": REDUNDANCY,
                  "t_manipulated": T_MANIP, "t_authentic": T_AUTH, "min_modalities": MIN_MODALITIES,
                  "min_assessable": MIN_ASSESSABLE, "min_auth_coverage": MIN_AUTH_COVERAGE,
                  "strong_contra_llr": STRONG_CONTRA_LLR, "prov_max_drop": PROV_MAX_DROP, "miss_rate_used": miss_rate},
    }


def _where(f: Finding) -> str:
    if f.segment:
        return f" at {f.segment.start:.2f}-{f.segment.end:.2f}s"
    return ""


def _summary(state: str, p: Optional[float], sup: list[Finding], con: list[Finding], reasons: list[str],
             assessed: list[str]) -> str:
    """Template wording from structured evidence only."""
    head = state.capitalize() + (f" (P(manipulated) = {p:.2f})" if p is not None else "") + "."
    parts = [head]
    if sup:
        parts.append("Main evidence for manipulation: " + "; ".join(f"{f.check}{_where(f)}" for f in sup[:3]) + ".")
    if con and state != STATE_MANIP:
        parts.append("Checks finding nothing anomalous: " + ", ".join(sorted({f.check for f in con})) + ".")
    if reasons:
        parts.append("Why not decided: " + "; ".join(reasons) + ".")
    parts.append(f"Modalities assessed: {', '.join(assessed) if assessed else 'none'}.")
    return " ".join(parts)
