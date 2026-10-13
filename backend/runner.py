from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from backend import db
from backend.adaptive.deep import DEEP_CHECKS, run_deep
from backend.adaptive.scheduler import schedule
from backend.adaptive.screening import screen, screening_findings
from backend.checks import JobContext, run_checks
from backend.config import CHECK_WORKERS, DEEP_BUDGET_S, JOB_WORKERS
from backend.fusion import fuse
from backend.ingest import preprocess, sha256_file
from backend.models import Finding
from backend.provenance import custody

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=JOB_WORKERS, thread_name_prefix="job")


def _opt_path(v: Any) -> Optional[Path]:
    return Path(v) if v else None


def build_context(job: dict[str, Any], arts: dict[str, Any]) -> JobContext:
    sched = arts.get("schedule")
    return JobContext(
        job_id=job["id"],
        media_path=Path(job["media_path"]),
        media_type=arts.get("media_type", "other"),
        work_dir=Path(arts["work_dir"]),
        sha256=job["sha256"],
        duration_s=arts.get("duration_s"),
        frames_dir=_opt_path(arts.get("frames_dir")),
        frame_sample_fps=arts.get("frame_sample_fps"),
        audio_path=_opt_path(arts.get("audio_path")),
        proxy_path=_opt_path(arts.get("proxy_path")),
        metadata_path=_opt_path(arts.get("metadata_path")),
        deep_intervals=sched["analysed"] if sched and "analysed" in sched else None,
    )


def plan(ctx: JobContext, arts: dict[str, Any], manual: Optional[list[dict]] = None) -> tuple[Optional[dict], Optional[dict]]:
    """Screening + ordered interval list for video; (None, None) for other media."""
    if ctx.media_type != "video":
        return None, None
    scr = arts.get("screening") or screen(ctx.proxy_path, ctx.frame_paths(), ctx.frame_sample_fps,
                                          ctx.duration_s or 0.0, bool(arts.get("has_audio")))
    return scr, schedule(scr, ctx.duration_s or 0.0, ctx.sha256 or "0", manual=manual)


def analyse(ctx: JobContext, arts: dict[str, Any], manual: Optional[list[dict]] = None,
            budget_s: float = DEEP_BUDGET_S) -> tuple[list[Finding], Optional[dict], Optional[dict]]:
    """Whole-file checks (parallel) + deadline-driven deep units on scheduled intervals."""
    findings = run_checks(ctx, max_workers=CHECK_WORKERS)
    scr, sched = plan(ctx, arts, manual)
    report = None
    if sched is not None:
        deep, report = run_deep(ctx, sched, budget_s)
        findings += deep + screening_findings(scr)
    return findings, scr, report


def process_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if job is None:
        return
    try:
        db.delete_findings(job_id)  # a resumed job starts clean (findings are saved in stages)
        db.update_job(job_id, status="preprocessing", error=None)
        original = Path(job["media_path"])
        if sha256_file(original) != job["sha256"]:
            raise RuntimeError("original file hash changed after ingest")
        arts = preprocess(job_id, original)
        db.update_job(job_id, status="screening", media_type=arts["media_type"],
                      duration_s=arts.get("duration_s"), artifacts=arts)
        custody.append(job_id, "preprocessed", {"media_type": arts["media_type"], "errors": arts["errors"]})

        # Staged so the workspace fills in progressively: screening curve -> whole-file findings -> deep findings.
        ctx = build_context(job, arts)
        scr, sched = plan(ctx, arts)
        arts["screening"], arts["schedule"] = scr, sched
        db.update_job(job_id, status="analyzing", artifacts=arts)
        findings = run_checks(ctx, max_workers=CHECK_WORKERS) + (screening_findings(scr) if scr else [])
        db.save_findings(job_id, findings)
        report = None
        if sched is not None:
            db.update_job(job_id, status="deep")
            deep, report = run_deep(ctx, sched)
            db.save_findings(job_id, deep)
            findings += deep
            arts["schedule"] = report
        assessment = fuse(findings, schedule=report, media_type=arts["media_type"])
        db.update_job(job_id, status="done", verdict=assessment["state"], confidence=assessment["confidence"],
                      artifacts=arts)
        custody.append(job_id, "analyzed", {
            "findings": len(findings), "verdict": assessment["state"],
            "p_manipulated": assessment["probability_manipulated"],
            "deep_coverage": report["coverage"] if report else None,
            "deep_elapsed_s": report["elapsed_s"] if report else None})
    except Exception as exc:
        log.exception("job %s failed", job_id)
        db.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")


def deepen(job_id: str, start: float, end: float, actor: str) -> None:
    """'Analyze deeper here': the manual interval goes first in the deadline loop; deep findings are replaced."""
    job = db.get_job(job_id)
    arts = job["artifacts"] or {}
    try:
        ctx = build_context(job, arts)
        manual = list((arts.get("schedule") or {}).get("manual", [])) + [{"start": start, "end": end}]
        scr, sched = plan(ctx, arts, manual=manual)
        deep, report = run_deep(ctx, sched)
        db.replace_findings(job_id, list(DEEP_CHECKS), deep)
        arts["schedule"] = report
        assessment = fuse(db.get_findings(job_id), schedule=report, media_type=arts.get("media_type"))
        db.update_job(job_id, status="done", verdict=assessment["state"], confidence=assessment["confidence"],
                      artifacts=arts)
        custody.append(job_id, "deep_analysis", {"interval": [start, end], "coverage": report["coverage"],
                                                 "elapsed_s": report["elapsed_s"], "verdict": assessment["state"]},
                       actor=actor)
    except Exception as exc:
        log.exception("deepen %s failed", job_id)
        db.update_job(job_id, status="done", error=f"deep analysis failed: {type(exc).__name__}: {exc}")


RUNNING = ("queued", "preprocessing", "screening", "analyzing", "deep")


def recover() -> list[str]:
    """Called at startup: re-run jobs interrupted by a restart; an interrupted 'analyze deeper' keeps its
    previous (complete) results."""
    resumed = []
    for j in db.jobs_with_status(RUNNING):
        custody.append(j["id"], "resumed_after_restart", {"interrupted_status": j["status"]})
        submit(j["id"])
        resumed.append(j["id"])
    for j in db.jobs_with_status(("deepening",)):
        db.update_job(j["id"], status="done", error="deep analysis interrupted by a restart; previous results kept")
        custody.append(j["id"], "deep_analysis_interrupted", {})
    return resumed


def submit(job_id: str) -> Future:
    return _executor.submit(process_job, job_id)


def submit_deep(job_id: str, start: float, end: float, actor: str) -> Future:
    db.update_job(job_id, status="deepening")
    return _executor.submit(deepen, job_id, start, end, actor)

