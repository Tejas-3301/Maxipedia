"""PDF forensic report: assessment, coverage, reviews, key evidence (with heatmaps / spectrograms), all findings,
provenance, and the full chain of custody with its verification and public key.

Built-in PDF fonts are Latin-1 only, so text is normalised (dashes, arrows, >=) instead of depending on a
platform TTF. The caller logs the PDF's SHA-256 as a signed custody entry after generation.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fpdf import FPDF

from backend.config import ARTIFACTS_DIR
from backend.models import Finding

_REPL = {"–": "-", "—": "-", "≥": ">=", "≤": "<=", "→": "->", "±": "+/-", "×": "x", "…": "...", "·": "-",
         "✱": "*", "‘": "'", "’": "'", "“": '"', "”": '"', "Δ": "d", "σ": "sigma", "π": "pi", "ε": "eps",
         "|z|": "|z|"}
STATE_RGB = {"likely manipulated": (185, 28, 28), "likely authentic": (21, 128, 61), "inconclusive": (180, 120, 0)}
IMAGE_KEYS = ("heatmap", "spectrogram", "plot")


def _t(s: Any) -> str:
    s = "" if s is None else str(s)
    for k, v in _REPL.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _fmt(x: Any, d: int = 2) -> str:
    return "-" if x is None else f"{float(x):.{d}f}"


def _seg(f: Finding) -> str:
    return f"{f.segment.start:.2f}-{f.segment.end:.2f}s" if f.segment else "file"


class Report(FPDF):
    def __init__(self, job_id: str):
        super().__init__(format="A4")
        self.job_id = job_id
        self.set_auto_page_break(True, margin=14)
        self.set_margins(14, 14, 14)

    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, _t(f"Forensic report - job {self.job_id}"), align="L")
        self.cell(0, 5, _t(f"page {self.page_no()}"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def h1(self, text: str):
        self.set_font("Helvetica", "B", 15)
        self.multi_cell(0, 8, _t(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def h2(self, text: str):
        self.ln(2)
        self.set_font("Helvetica", "B", 11.5)
        self.set_fill_color(235, 238, 243)
        self.cell(0, 7, _t(text), fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str, size: float = 9.5, style: str = ""):
        self.set_font("Helvetica", style, size)
        self.multi_cell(0, 4.6, _t(text), new_x="LMARGIN", new_y="NEXT")

    def kv(self, rows: list[tuple[str, Any]], key_w: float = 42):
        for k, v in rows:
            self.set_font("Helvetica", "B", 9)
            self.cell(key_w, 5, _t(k))
            self.set_font("Helvetica", "", 9)
            self.multi_cell(0, 5, _t(v), new_x="LMARGIN", new_y="NEXT")

    def table(self, head: list[str], rows: list[list[Any]], widths: list[float], size: float = 7.5):
        self.set_font("Helvetica", "B", size)
        self.set_fill_color(225, 229, 236)
        for h, w in zip(head, widths):
            self.cell(w, 5, _t(h), border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", size)
        for r in rows:
            cells = [_t(c) for c in r]
            lines = max(len(self.multi_cell(w, 3.6, c, dry_run=True, output="LINES")) for c, w in zip(cells, widths))
            h = 3.6 * max(1, lines) + 0.8
            if self.get_y() + h > self.h - 16:
                self.add_page()
            x0, y0 = self.get_x(), self.get_y()
            for c, w in zip(cells, widths):
                x = self.get_x()
                self.rect(x, y0, w, h)
                self.multi_cell(w, 3.6, c, new_x="RIGHT", new_y="TOP", max_line_height=3.6)
                self.set_xy(x + w, y0)
            self.set_xy(x0, y0 + h)


def _image(pdf: Report, job_id: str, path: Optional[str], w: float = 85) -> bool:
    if not path:
        return False
    p = Path(path).resolve()
    if (ARTIFACTS_DIR / job_id).resolve() not in p.parents or not p.is_file():
        return False
    if pdf.get_y() + w * 0.75 > pdf.h - 16:
        pdf.add_page()
    pdf.image(str(p), w=w)
    return True


def build_report(job: dict[str, Any], findings: list[Finding], assessment: dict[str, Any],
                 reviews: dict[str, dict], reviewed: Optional[dict[str, Any]], custody: dict[str, Any],
                 public_key_pem: str) -> bytes:
    pdf = Report(job["id"])
    pdf.add_page()
    pdf.h1("Synthetic-media forensic report")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pdf.kv([("File", job["filename"]), ("SHA-256", job["sha256"]), ("Size", f"{job.get('size_bytes')} bytes"),
            ("Media type", job.get("media_type")), ("Duration", f"{_fmt(job.get('duration_s'))} s"),
            ("Job", job["id"]), ("Ingested", job.get("created_at")), ("Report generated", now)])

    # ---------------- assessment
    pdf.h2("Final assessment (made by evidence fusion, not by any single detector)")
    r, g, b = STATE_RGB.get(assessment["state"], (0, 0, 0))
    pdf.set_text_color(r, g, b)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, _t(assessment["state"].upper()), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.kv([("P(manipulated)", _fmt(assessment["probability_manipulated"], 3)),
            ("Confidence", "abstained" if assessment["confidence"] is None else _fmt(assessment["confidence"], 3)),
            ("Why not decided", "; ".join(assessment["reasons"]) or "-")])
    pdf.para(assessment["summary"])
    cov = assessment.get("coverage") or {}
    if cov.get("deep_coverage") is not None:
        pdf.para(f"Deep analysis covered {cov['deep_coverage'] * 100:.0f}% of the clip in {cov['deep_elapsed_s']} s "
                 f"(deadline {cov['budget_s']} s); {len(cov.get('unanalysed_intervals') or [])} interval(s) not analysed; "
                 f"control intervals {cov['controls']} ({cov['control_hits']} with findings), estimated screening miss "
                 f"rate {(cov.get('miss_rate_est') or 0) * 100:.0f}%.", size=9)
    pdf.ln(1)
    pdf.table(["Modality", "assessed", "log-odds", "posterior", "conflict"],
              [[m, "yes" if v["assessed"] else "prior", _fmt(v["llr"]), _fmt(v["posterior"], 3), "yes" if v["conflict"] else ""]
               for m, v in assessment["modalities"].items()], [40, 25, 30, 30, 25], size=8.5)
    if reviewed:
        pdf.para(f"Reviewed verdict (investigator-rejected evidence removed, {len(reviewed['excluded'])} finding(s)): "
                 f"{reviewed['after']['state']}, P={_fmt(reviewed['after']['probability_manipulated'], 3)}. The automatic "
                 f"verdict above is unchanged.", size=9, style="B")
    pdf.h2("Limitations")
    for line in assessment["limitations"]:
        pdf.para(f"- {line}", size=8.5)

    # ---------------- key evidence
    by_id = {f.id: f for f in findings}
    key = [by_id[x["id"]] for x in assessment["supporting"] + assessment["contradicting"][:3] if x["id"] in by_id]
    pdf.h2("Key evidence")
    for f in key:
        rv = reviews.get(f.id, {})
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.multi_cell(0, 5, _t(f"{f.check}  [{f.stance}]  score {_fmt(f.score)}  reliability {_fmt(f.reliability)}  "
                                f"{f.depth}  {_seg(f)}" + (f"  region {list(f.region)}" if f.region else "")
                                + (f"  review: {rv['decision']}" if rv.get("decision") else "")),
                       new_x="LMARGIN", new_y="NEXT")
        pdf.para(f.rationale, size=8.5)
        pdf.para(f"Limitations: {f.limitations}", size=8, style="I")
        for n in rv.get("notes", []):
            pdf.para(f"Annotation ({n['actor']}, {n['ts'][:19]}): {n['note']}", size=8.5)
        for k in IMAGE_KEYS:
            if _image(pdf, job["id"], f.evidence.get(k)):
                break
        fields = f.evidence.get("fields")
        if isinstance(fields, list) and fields:
            pdf.table(["Field", "Value"], [[x.get("field"), str(x.get("value"))[:160]] for x in fields[:12]], [70, 112], size=7)
        pdf.ln(2)

    # ---------------- all findings
    pdf.add_page()
    pdf.h2(f"All findings ({len(findings)})")
    rows = []
    for f in sorted(findings, key=lambda f: (f.modality, f.check, f.segment.start if f.segment else -1)):
        if f.check == "ingest_inventory":
            continue
        rows.append([f.check, f.modality, f.stance, f.depth, _fmt(f.score), _fmt(f.reliability), _seg(f),
                     (reviews.get(f.id) or {}).get("decision") or ""])
    pdf.table(["Check", "Modality", "Stance", "Depth", "Score", "Rel.", "Segment", "Review"], rows,
              [42, 20, 34, 13, 13, 13, 30, 17])

    prov = [f for f in findings if f.modality == "provenance" and f.stance != "not_assessable"]
    if prov:
        pdf.h2("Provenance")
        for f in prov:
            pdf.para(f"{f.check}: {f.rationale}", size=8.5)

    # ---------------- custody
    pdf.add_page()
    v = custody["verification"]
    pdf.h2("Chain of custody")
    pdf.kv([("Verification", "INTACT - every entry hash recomputes and every Ed25519 signature is valid"
              if v["intact"] else "BROKEN: " + "; ".join(v["problems"])),
            ("Entries", v["entries"]), ("Signing key id", v["key_id"]), ("Chain head", v["head"])])
    pdf.para("Each entry hash = SHA-256 over (job, action, actor, detail, timestamp, previous hash); the signature is "
             "Ed25519 over that hash. Editing, removing or reordering any entry breaks verification. The SHA-256 of this "
             "PDF is appended to the chain as a signed 'report_exported' entry after generation.", size=8)
    pdf.ln(1)
    pdf.table(["#", "Time (UTC)", "Actor", "Action", "Detail", "Entry hash"],
              [[e["seq"], e["ts"].replace("T", " ")[:19], e["actor"], e["action"],
                json.dumps(json.loads(e["detail"]) if e["detail"] else {}, separators=(",", ":"))[:220],
                e["entry_hash"][:16] + "..."] for e in custody["entries"]],
              [8, 30, 20, 24, 70, 30], size=6.8)
    pdf.ln(2)
    pdf.set_font("Courier", "", 7)
    pdf.multi_cell(0, 3.4, _t("Public key (Ed25519, PEM):\n" + public_key_pem), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
