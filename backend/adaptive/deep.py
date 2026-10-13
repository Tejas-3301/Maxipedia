"""Deadline-driven deep analysis.

Walks the scheduler's ordered intervals; before starting each one it checks the elapsed time and stops once the
budget is reached (an interval already running finishes, so the overshoot is at most one interval). For each
interval every deep unit runs (in parallel threads): visual region (noise + ELA), visual temporal, lip sync.
Units then finalize over all analysed intervals. Reports actual elapsed time and the intervals left unanalysed.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from backend.adaptive.scheduler import union_length
from backend.checks import JobContext
from backend.checks import sync as sync_mod
from backend.checks import visual
from backend.config import DEEP_BUDGET_S
from backend.models import Finding, Modality, Stance


@dataclass(frozen=True)
class DeepUnit:
    name: str
    checks: tuple[str, ...]
    modality: Modality
    analyse: Callable[[JobContext, dict], dict]
    finalize: Callable[[JobContext, list[tuple[dict, dict]]], list[Finding]]


UNITS = (
    DeepUnit("visual.region", ("visual.noise_residual", "visual.ela"), Modality.visual,
             visual.region_analyse, visual.region_finalize),
    DeepUnit("visual.temporal", ("visual.temporal",), Modality.visual, visual.temporal_analyse, visual.temporal_finalize),
    DeepUnit("sync", ("sync.lipsync",), Modality.sync, sync_mod.sync_analyse, sync_mod.sync_finalize),
)
DEEP_CHECKS = tuple(c for u in UNITS for c in u.checks)


def _safe(fn: Callable, *args) -> dict:
    try:
        return fn(*args)
    except Exception as exc:  # one unit failing must not stop the others
        return {"error": f"{type(exc).__name__}: {exc}"}


def _fail(unit: DeepUnit, exc: Exception) -> list[Finding]:
    return [Finding(modality=unit.modality, check=c, stance=Stance.not_assessable, reliability=0.0,
                    rationale="Deep unit failed to finalize.", limitations=f"{type(exc).__name__}: {exc}")
            for c in unit.checks]


def _control_hit(f: Finding, iv: dict) -> bool:
    """A deep supports finding whose segment midpoint lies inside the control interval proper."""
    if f.stance != Stance.supports_manipulation or f.segment is None:
        return False
    mid = (f.segment.start + f.segment.end) / 2
    return iv["start"] <= mid <= iv["end"]


def run_deep(ctx: JobContext, sched: dict[str, Any], budget_s: Optional[float] = None) -> tuple[list[Finding], dict]:
    budget = DEEP_BUDGET_S if budget_s is None else budget_s
    t0 = time.perf_counter()
    analysed: list[dict] = []
    unanalysed: list[dict] = []
    results: dict[str, list[tuple[dict, dict]]] = {u.name: [] for u in UNITS}
    with ThreadPoolExecutor(max_workers=len(UNITS), thread_name_prefix="deep") as pool:
        for iv in sched["candidates"]:
            started = time.perf_counter() - t0
            if started >= budget:
                unanalysed.append(iv)
                continue
            raws = list(pool.map(lambda u: _safe(u.analyse, ctx, iv), UNITS))
            for u, r in zip(UNITS, raws):
                results[u.name].append((iv, r))
            analysed.append({**iv, "started_s": round(started, 3), "took_s": round(time.perf_counter() - t0 - started, 3)})

    findings: list[Finding] = []
    for u in UNITS:
        try:
            findings.extend(u.finalize(ctx, results[u.name]))
        except Exception as exc:
            findings.extend(_fail(u, exc))
    elapsed = time.perf_counter() - t0

    dur = sched["duration_s"] or 0.0
    controls = [iv for iv in analysed if iv["kind"] == "control"]
    hits = [iv for iv in controls if any(_control_hit(f, iv) for f in findings)]
    report = {
        **sched,
        "budget_s": budget,
        "elapsed_s": round(elapsed, 3),
        "analysed": analysed,
        "unanalysed": unanalysed,
        "coverage": round(min(1.0, union_length(analysed) / dur), 4) if dur else 0.0,
        "controls_analysed": len(controls),
        "control_hits": len(hits),
        "control_hit_intervals": hits,
        # Laplace-smoothed screening miss rate: a few clean controls must not claim a perfect screener
        "miss_rate_est": round((len(hits) + 1) / (len(controls) + 2), 4),
    }
    for f in findings:
        f.evidence.setdefault("schedule_coverage", report["coverage"])
    return findings, report
