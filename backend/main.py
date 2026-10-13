from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

from backend import db, runner
from backend.provenance import custody, graph, ledger
from backend.checks import get_checks
from backend.config import ARTIFACTS_DIR, PROXIES_DIR, UPLOADS_DIR
from backend.fusion import counterfactual, fuse, impacts
from backend.ingest import UploadTooLarge, store_original
from backend.models import Finding


def startup() -> None:
    for d in (UPLOADS_DIR, PROXIES_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    db.init_db()
    get_checks()  # import & register all checks at startup
    runner.recover()  # resume jobs interrupted by a restart


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup()
    yield


app = FastAPI(title="Media Forensics & Provenance", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    try:
        with db.session() as conn:
            conn.execute("SELECT 1")
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "checks_registered": [s.name for s in get_checks()],
    }


@app.post("/jobs", status_code=202)
async def create_job(file: UploadFile, manifest: UploadFile | None = None) -> dict:
    """Upload media; optionally a capture-device manifest (`*.capture.json`) signed at capture."""
    filename = file.filename or "upload.bin"
    job_id = uuid.uuid4().hex
    try:
        path, sha, size = await run_in_threadpool(store_original, job_id, filename, file.file)
    except UploadTooLarge as exc:
        raise HTTPException(413, str(exc))
    if size == 0:
        raise HTTPException(400, "empty upload")
    detail = {"filename": filename, "sha256": sha, "size_bytes": size}
    if manifest is not None:
        raw = await manifest.read()
        if len(raw) > 5_000_000:
            raise HTTPException(413, "manifest too large")
        (Path(path).parent / "capture_manifest.json").write_bytes(raw)
        detail["capture_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    db.create_job(job_id, filename, str(path), sha, size)
    custody.append(job_id, "ingested", detail)
    runner.submit(job_id)
    return {"job_id": job_id, "sha256": sha, "size_bytes": size, "status": "queued"}


@app.get("/jobs")
def list_jobs(limit: int = 50) -> list[dict]:
    return db.list_jobs(min(max(limit, 1), 200))


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/jobs/{job_id}/findings")
def get_findings(job_id: str) -> list[Finding]:
    if db.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    return db.get_findings(job_id)


@app.get("/jobs/{job_id}/assessment")
def get_assessment(job_id: str) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    sched = (job.get("artifacts") or {}).get("schedule")
    return {"job_id": job_id, **fuse(db.get_findings(job_id), schedule=sched, media_type=job.get("media_type"))}


@app.get("/jobs/{job_id}/screening")
def get_screening(job_id: str) -> dict:
    """Suspicion curves (per modality + combined) and the deep-analysis schedule."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    arts = job.get("artifacts") or {}
    return {"job_id": job_id, "status": job["status"], "screening": arts.get("screening"),
            "schedule": arts.get("schedule")}


class DeepRequest(BaseModel):
    start: float
    end: float
    actor: str = "investigator"


@app.post("/jobs/{job_id}/deep", status_code=202)
def analyze_deeper(job_id: str, req: DeepRequest) -> dict:
    """'Analyze deeper here': deep-check [start, end] in addition to the scheduled intervals."""
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    if job.get("media_type") != "video":
        raise HTTPException(400, "deep interval analysis applies to video")
    dur = job.get("duration_s") or 0.0
    start, end = max(0.0, req.start), min(dur, req.end) if dur else req.end
    if end - start < 0.25:
        raise HTTPException(400, "interval must lie within the clip and be at least 0.25 s long")
    runner.submit_deep(job_id, start, end, req.actor)
    return {"job_id": job_id, "status": "deepening", "interval": {"start": start, "end": end}}


class TrustRequest(BaseModel):
    trusted: bool = True
    label: str | None = None
    actor: str = "investigator"


class Annotation(BaseModel):
    actor: str
    note: str


@app.post("/ledger/{job_id}/trust")
def set_trust(job_id: str, req: TrustRequest) -> dict:
    """Register (or unregister) an analysed version as a trusted reference original."""
    if not ledger.set_trusted(job_id, req.trusted, req.label):
        raise HTTPException(404, "version not in ledger (job not analysed yet?)")
    custody.append(job_id, "trust_set" if req.trusted else "trust_revoked", {"label": req.label}, actor=req.actor)
    return ledger.get(job_id).brief()


@app.get("/provenance/graph")
def lineage_graph(job_id: str | None = None) -> dict:
    return graph.build(job_id)


@app.get("/jobs/{job_id}/custody")
def get_custody(job_id: str) -> dict:
    if db.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    return {"verification": custody.verify(job_id), "public_key_pem": custody.public_pem(),
            "entries": custody.entries(job_id)}


@app.post("/jobs/{job_id}/custody")
def annotate(job_id: str, a: Annotation) -> dict:
    if db.get_job(job_id) is None:
        raise HTTPException(404, "job not found")
    return custody.append(job_id, "annotation", {"note": a.note}, actor=a.actor)


def _done_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return job


@app.get("/jobs/{job_id}/graph")
def evidence_graph(job_id: str) -> dict:
    """Evidence graph (finding -> modality -> verdict, plus corroborates/conflicts/redundant_with edges) with the
    effect of removing each weighted finding."""
    job = _done_job(job_id)
    findings = db.get_findings(job_id)
    sched = (job.get("artifacts") or {}).get("schedule")
    a = fuse(findings, schedule=sched, media_type=job.get("media_type"))
    imp = impacts(findings, sched, job.get("media_type"))
    for n in a["graph"]["nodes"]:
        if n["id"] in imp:
            n["impact"] = imp[n["id"]]
    return {"job_id": job_id, "state": a["state"], "probability_manipulated": a["probability_manipulated"],
            **a["graph"]}


class CounterfactualRequest(BaseModel):
    exclude: list[str] = []
    exclude_checks: list[str] = []
    exclude_modalities: list[str] = []
    actor: str | None = None


@app.post("/jobs/{job_id}/counterfactual")
def run_counterfactual(job_id: str, req: CounterfactualRequest) -> dict:
    """'Remove this evidence': recompute the verdict without the given findings / checks / modalities.
    The stored verdict is not changed; an investigator's request is logged to the custody chain."""
    job = _done_job(job_id)
    findings = db.get_findings(job_id)
    ids = {f.id for f in findings}
    unknown = [i for i in req.exclude if i not in ids]
    if unknown:
        raise HTTPException(400, f"unknown finding id(s): {unknown[:5]}")
    exclude = set(req.exclude) | {f.id for f in findings
                                  if f.check in req.exclude_checks or f.modality in req.exclude_modalities}
    if not exclude:
        raise HTTPException(400, "nothing to exclude")
    out = counterfactual(findings, exclude, (job.get("artifacts") or {}).get("schedule"), job.get("media_type"))
    if req.actor:
        custody.append(job_id, "counterfactual", {"excluded": sorted(exclude), "before": out["before"]["state"],
                                                  "after": out["after"]["state"], "delta_p": out["delta_p"]},
                       actor=req.actor)
    return {"job_id": job_id, **out}


# ---------------------------------------------------------------- workspace support

def _job_or_404(job_id: str) -> dict:
    job = db.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/jobs/{job_id}/media")
def media(job_id: str, kind: str = "proxy") -> FileResponse:
    """Playable media for the workspace (proxy: H.264/AAC, faststart). Range requests are supported for seeking."""
    job = _job_or_404(job_id)
    arts = job.get("artifacts") or {}
    path = arts.get("proxy_path") if kind == "proxy" else job["media_path"] if kind == "original" else None
    if not path or not Path(path).exists():
        raise HTTPException(404, f"no {kind} media")
    return FileResponse(path, media_type="video/mp4" if path.endswith(".mp4") else None)


@app.get("/jobs/{job_id}/info")
def media_info(job_id: str) -> dict:
    """Original frame size (finding regions are in these pixels), fps and duration."""
    import cv2

    job = _job_or_404(job_id)
    cap = cv2.VideoCapture(job["media_path"])
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None, "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": cap.get(cv2.CAP_PROP_FPS) or None}
    cap.release()
    arts = job.get("artifacts") or {}
    return {"job_id": job_id, "media_type": job.get("media_type"), "duration_s": job.get("duration_s"),
            "has_proxy": bool(arts.get("proxy_path")), **info}


@app.get("/jobs/{job_id}/artifact")
def artifact(job_id: str, path: str) -> FileResponse:
    """Serve a file referenced in a finding's evidence (heatmap, spectrogram, sync curve) - only from this job's
    artifact folder."""
    _job_or_404(job_id)
    root = (ARTIFACTS_DIR / job_id).resolve()
    p = Path(path).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "artifact not found for this job")
    return FileResponse(p)


class ReviewRequest(BaseModel):
    decision: str  # accept | reject | note | clear
    note: str | None = None
    actor: str = "investigator"


@app.post("/jobs/{job_id}/findings/{finding_id}/review")
def review_finding(job_id: str, finding_id: str, req: ReviewRequest) -> dict:
    """Accept / reject / annotate one finding. Logged (signed) to the custody chain, which is the source of truth;
    the stored verdict is not changed (rejections feed the counterfactual instead)."""
    _job_or_404(job_id)
    if req.decision not in ("accept", "reject", "note", "clear"):
        raise HTTPException(400, "decision must be accept, reject, note or clear")
    if req.decision == "note" and not (req.note or "").strip():
        raise HTTPException(400, "a note is required")
    f = next((x for x in db.get_findings(job_id) if x.id == finding_id), None)
    if f is None:
        raise HTTPException(404, "finding not found")
    entry = custody.append(job_id, "review", {"finding_id": finding_id, "check": f.check, "decision": req.decision,
                                              "note": req.note}, actor=req.actor)
    return {"finding_id": finding_id, **entry, "reviews": reviews(job_id)["reviews"].get(finding_id)}


@app.get("/jobs/{job_id}/reviews")
def reviews(job_id: str) -> dict:
    """Current review state per finding, rebuilt from the custody chain (latest accept/reject wins; notes accumulate)."""
    _job_or_404(job_id)
    import json as _json

    state: dict[str, dict] = {}
    for e in custody.entries(job_id):
        if e["action"] != "review":
            continue
        d = _json.loads(e["detail"])
        r = state.setdefault(d["finding_id"], {"decision": None, "notes": [], "history": []})
        if d["decision"] in ("accept", "reject"):
            r["decision"] = d["decision"]
        elif d["decision"] == "clear":
            r["decision"] = None
        if d.get("note"):
            r["notes"].append({"actor": e["actor"], "note": d["note"], "ts": e["ts"]})
        r["history"].append({"actor": e["actor"], "decision": d["decision"], "ts": e["ts"], "entry_hash": e["entry_hash"]})
    return {"job_id": job_id, "reviews": state}


# ---------------------------------------------------------------- report export

@app.get("/jobs/{job_id}/report.pdf")
def export_report(job_id: str, actor: str = "investigator") -> Response:
    """PDF report with assessment, evidence, reviews and the full chain of custody. The PDF's SHA-256 is then
    appended to the chain as a signed 'report_exported' entry (returned in X-Report-SHA256)."""
    import hashlib

    from backend.report import build_report

    job = _done_job(job_id)
    findings = db.get_findings(job_id)
    sched = (job.get("artifacts") or {}).get("schedule")
    assessment = fuse(findings, schedule=sched, media_type=job.get("media_type"))
    rv = reviews(job_id)["reviews"]
    rejected = {fid for fid, r in rv.items() if r["decision"] == "reject"}
    reviewed = counterfactual(findings, rejected, sched, job.get("media_type")) if rejected else None
    pdf = build_report(job, findings, assessment, rv, reviewed,
                       {"verification": custody.verify(job_id), "entries": custody.entries(job_id)},
                       custody.public_pem())
    digest = hashlib.sha256(pdf).hexdigest()
    entry = custody.append(job_id, "report_exported", {"sha256": digest, "bytes": len(pdf),
                                                       "verdict": assessment["state"]}, actor=actor)
    name = Path(job["filename"]).stem
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="report_{name}_{job_id[:8]}.pdf"',
        "X-Report-SHA256": digest, "X-Custody-Entry": entry["entry_hash"]})


# ---------------------------------------------------------------- attack mode

class AttackRequest(BaseModel):
    kind: str  # recompress | resize | noise
    level: str = "medium"
    actor: str = "investigator"


@app.post("/jobs/{job_id}/attack", status_code=202)
def attack(job_id: str, req: AttackRequest) -> dict:
    """Derive a degraded copy of the original and analyse it as a linked job, to show evidence degrading."""
    import tempfile

    from backend.attack import make_attack

    job = _done_job(job_id)
    if job.get("media_type") != "video":
        raise HTTPException(400, "attack mode applies to video")
    child_id = uuid.uuid4().hex
    stem = Path(job["filename"]).stem
    name = f"{stem}__{req.kind}-{req.level}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / name
        try:
            info = make_attack(Path(job["media_path"]), req.kind, req.level, out)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        with open(out, "rb") as f:
            path, sha, size = store_original(child_id, name, f)
    db.create_job(child_id, name, str(path), sha, size)
    db.update_job(child_id, derived_from=job_id, attack=info)
    custody.append(child_id, "ingested", {"filename": name, "sha256": sha, "size_bytes": size,
                                          "attack_of": job_id, "attack": info}, actor=req.actor)
    custody.append(job_id, "attack_mode", {"child": child_id, "attack": info}, actor=req.actor)
    runner.submit(child_id)
    return {"job_id": child_id, "parent": job_id, "attack": info, "status": "queued"}


@app.get("/jobs/{job_id}/attacks")
def attacks(job_id: str) -> dict:
    """Attacked copies of this job with verdict / P / confidence next to the parent and per-check degradation."""
    from backend.attack import compare

    job = _done_job(job_id)
    p_findings = db.get_findings(job_id)
    base = fuse(p_findings, schedule=(job.get("artifacts") or {}).get("schedule"), media_type=job.get("media_type"))
    out = []
    for ch in db.children(job_id):
        row = {"job_id": ch["id"], "filename": ch["filename"], "status": ch["status"], "attack": ch["attack"]}
        if ch["status"] == "done":
            c_findings = db.get_findings(ch["id"])
            a = fuse(c_findings, schedule=(ch.get("artifacts") or {}).get("schedule"), media_type=ch.get("media_type"))
            row.update(state=a["state"], probability_manipulated=a["probability_manipulated"],
                       confidence=a["confidence"], reasons=a["reasons"],
                       checks=compare(p_findings, c_findings))
        out.append(row)
    return {"job_id": job_id, "parent": {"state": base["state"], "probability_manipulated": base["probability_manipulated"],
                                         "confidence": base["confidence"]}, "attacks": out}


# ---------------------------------------------------------------- capture devices (IoT signing simulation)

class DeviceRequest(BaseModel):
    device_id: str
    public_key_pem: str
    label: str | None = None


@app.post("/devices")
def register_device(req: DeviceRequest) -> dict:
    """Register a capture device's public key as a trust anchor."""
    from backend.provenance import device

    try:
        device.register(req.device_id, req.public_key_pem, req.label)
    except ValueError as exc:
        raise HTTPException(400, f"invalid public key: {exc}")
    return {"device_id": req.device_id, "registered": True}


@app.get("/devices")
def list_devices() -> list[dict]:
    from backend.provenance import device

    return device.devices()
