"""Provenance check: SHA-256 ledger match, per-segment Merkle comparison, perceptual lineage.

Records this file in the ledger after comparing. Only a *trusted* reference (registered original) yields
supports/contradicts; matches against ordinary earlier uploads are reported as neutral context, because a
pairwise difference cannot say which of the two was altered.
"""
from __future__ import annotations

from typing import Optional

from backend import db
from backend.checks.registry import JobContext, register
from backend.models import Depth, Finding, Modality, Segment, Stance
from backend.provenance import graph, ledger
from backend.provenance.fingerprint import Fingerprint, fingerprint, merkle_proof
from backend.provenance.lineage import relate


MIN_SHARED = 0.05  # fraction of this file's packets that must also occur in the other version


def _f(check: str, stance: Stance, score: Optional[float], rel: float, rationale: str, lim: str,
       segment: Optional[Segment] = None, **evidence) -> Finding:
    return Finding(modality=Modality.provenance, check=check, depth=Depth.screen, stance=stance, score=score,
                   reliability=rel, rationale=rationale, limitations=lim, segment=segment, evidence=evidence)


def _ref_name(v: ledger.Version) -> str:
    return f"{'trusted reference' if v.trusted else 'earlier upload'} {v.filename!r} (job {v.job_id[:8]})"


# ---------------------------------------------------------------- SHA-256 ledger

def check_sha(ctx: JobContext) -> Finding:
    name = "provenance.sha256_ledger"
    lim = "A hash match proves identity with the recorded bytes, not that the recorded version was itself genuine."
    hits = ledger.by_sha(ctx.sha256 or "", exclude_job=ctx.job_id)
    trusted = [v for v in hits if v.trusted]
    if trusted:
        v = trusted[0]
        return _f(name, Stance.contradicts, 0.9, 0.9,
                  f"SHA-256 {ctx.sha256} matches {_ref_name(v)} exactly: the file is bit-identical to the "
                  f"registered version.", lim, match=v.brief(), sha256=ctx.sha256)
    if hits:  # identical to something seen before, but nobody vouched for it: weak clean evidence only
        return _f(name, Stance.contradicts, 0.3, 0.4,
                  f"SHA-256 matches {len(hits)} earlier upload(s), first {_ref_name(hits[0])}; none is a trusted "
                  f"reference, so the match only weakly supports authenticity.", lim,
                  matches=[h.brief() for h in hits], sha256=ctx.sha256)
    return _f(name, Stance.neutral, 0.0, 0.5, f"SHA-256 {ctx.sha256} is not in the ledger (first sighting).",
              lim, sha256=ctx.sha256)


# ---------------------------------------------------------------- Merkle / packet comparison

def _spans(segs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def check_merkle(ctx: JobContext, fp: Fingerprint, others: list[ledger.Version]) -> Finding:
    name = "provenance.merkle"
    lim = ("Compares compressed packets: any re-encode changes every leaf even if the picture is the same, so "
           "'changed' means re-encoded or edited, not necessarily altered in content.")
    base = {"merkle_root": fp.merkle_root, "leaves": len(fp.leaves)}
    if not fp.leaves:
        return _f(name, Stance.not_assessable, None, 0.0, "No packets could be hashed.", lim, **base)
    scored = [(len(fp.packet_set & v.packet_set) / max(1, len(fp.packet_set)), v) for v in others]
    scored = [s for s in scored if s[0] >= MIN_SHARED]
    if not scored:
        return _f(name, Stance.neutral, 0.0, 0.5,
                  f"Merkle root {fp.merkle_root[:16]}... over {len(fp.leaves)} one-second stream segments; no "
                  f"recorded version shares any compressed packet.", lim, **base)
    scored.sort(key=lambda s: (-s[1].trusted, -s[0]))
    shared, ref = scored[0]
    ref_leaves = {(l.kind, l.segment): l.hash for l in ref.leaves}
    rows, changed = [], []
    hashes = [l.hash for l in fp.leaves]
    for idx, l in enumerate(fp.leaves):
        status = ("identical" if ref_leaves.get((l.kind, l.segment)) == l.hash
                  else "changed" if (l.kind, l.segment) in ref_leaves else "added")
        rows.append({"stream": l.stream, "kind": l.kind, "segment": l.segment, "start": l.start, "end": l.end,
                     "status": status, "hash": l.hash})
        if status != "identical":
            changed.append((l.kind, l.start, l.end, idx))
    kinds = sorted({r["kind"] for r in rows})
    per_kind = {k: {"identical": sum(r["kind"] == k and r["status"] == "identical" for r in rows),
                    "total": sum(r["kind"] == k for r in rows)} for k in kinds}
    summary = "; ".join(f"{k}: {v['identical']}/{v['total']} segments identical" for k, v in per_kind.items())
    ev = {**base, "reference": ref.brief(), "reference_root": ref.merkle_root, "shared_packet_fraction": round(shared, 4),
          "segments": rows, "per_stream": per_kind,
          "proofs": {str(i): merkle_proof(hashes, i) for *_, i in changed[:5]}}

    if fp.merkle_root == ref.merkle_root:
        stance, score, rel = (Stance.contradicts, 0.8, 0.8) if ref.trusted else (Stance.contradicts, 0.3, 0.4)
        return _f(name, stance, score, rel,
                  f"Merkle root {fp.merkle_root[:16]}... equals that of {_ref_name(ref)}: every audio/video packet is "
                  f"bit-identical, so the content is unaltered; only the container/metadata bytes differ"
                  f"{' (SHA-256 differs)' if ctx.sha256 != ref.sha256 else ''}.", lim, **ev)

    spans = {k: _spans([(s, e) for kk, s, e, _ in changed if kk == k]) for k in kinds}
    span_txt = "; ".join(f"{k} changed at " + ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in v)
                         for k, v in spans.items() if v)
    untouched = [k for k, v in per_kind.items() if v["identical"] == v["total"]]
    all_changed = [s for v in spans.values() for s in v]
    seg = Segment(start=min(a for a, _ in all_changed), end=max(b for _, b in all_changed), unit="s")
    text = (f"Relative to {_ref_name(ref)} (shares {shared:.0%} of packets): {summary}. {span_txt}."
            + (f" Unchanged stream(s): {', '.join(untouched)} - consistent with a selective edit/replacement of the "
               f"other stream(s)." if untouched else ""))
    localized = any(r["status"] == "identical" for r in rows)  # some segments survive bit-exact
    if not localized:
        return _f(name, Stance.neutral, 0.0, 0.5,
                  text + " Every segment differs (re-encoded throughout), so the tree cannot localize an edit; "
                  "a hash mismatch alone is not evidence of manipulation.", lim, **ev)
    if ref.trusted:
        score = 0.6 if untouched else 0.45
        return _f(name, Stance.supports_manipulation, score, 0.7, text, lim, segment=seg, **ev)
    return _f(name, Stance.neutral, 0.0, 0.5,
              text + " The other version is not a trusted reference, so the direction of the change is unknown.",
              lim, segment=seg, **ev)


# ---------------------------------------------------------------- perceptual lineage

def check_lineage(ctx: JobContext, fp: Fingerprint, others: list[ledger.Version]) -> Finding:
    name = "provenance.lineage"
    lim = ("Perceptual hashes match visually similar content, including legitimate re-uploads; alignment works "
           "at 0.1 s resolution and assumes mostly stable framing.")
    if fp.thumbs is None:
        return _f(name, Stance.not_assessable, None, 0.0, "No video stream for perceptual lineage.", lim)
    found = 0
    for v in others:
        vt = v.thumbs()
        fwd = relate(ctx.job_id, fp.phashes, fp.thumbs, v.job_id, v.phashes, vt)
        if fwd is None:
            continue
        found += 1
        graph.save_relation(ctx.job_id, v.job_id, fwd)
        back = relate(v.job_id, v.phashes, vt, ctx.job_id, fp.phashes, fp.thumbs)
        if back:
            graph.save_relation(v.job_id, ctx.job_id, back)
    if not found:
        return _f(name, Stance.neutral, 0.0, 0.5, "No perceptual near-duplicates in the ledger.", lim,
                  near_duplicates=0)

    p = graph.trusted_relative(ctx.job_id) or graph.parent_of(ctx.job_id)
    component = graph.build(ctx.job_id)
    if p is None:
        return _f(name, Stance.neutral, 0.0, 0.5,
                  f"{found} perceptual near-duplicate(s) in the ledger, none identifiable as this file's source "
                  f"(same size and length, no trusted reference).", lim, near_duplicates=found, graph=component)
    parent_id, rel = p
    parent = ledger.get(parent_id)
    ops = []
    if rel.coverage and (rel.coverage[0] > 0.15 or (parent.duration_s and parent.duration_s - rel.coverage[1] > 0.15)):
        ops.append(f"covers {rel.coverage[0]:.1f}-{rel.coverage[1]:.1f}s of it")
    if rel.crop > 1.01:
        ops.append(f"shows the central {1 / rel.crop:.0%} of its frame (cropped)")
    if parent.width and fp.width and parent.width != fp.width:
        ops.append(f"rescaled {parent.width}->{fp.width}px wide")
    edits = rel.offset_changes
    edit_txt = "; ".join(f"{e.get('removed_s', e.get('inserted_s'))}s {'removed' if 'removed_s' in e else 'inserted'} "
                         f"at {e['at_s']:.1f}s" for e in edits)
    desc = (f"Perceptual lineage: derived from {_ref_name(parent)} (pHash median {rel.median_bits:.0f}/64 bits, "
            f"{rel.match_frac:.0%} of frames matched)" + (f", {', '.join(ops)}" if ops else "") + ".")
    ev = {"parent": parent.brief(), "relation": {**rel.__dict__}, "graph": component, "near_duplicates": found}
    if edits:
        seg = Segment(start=max(0.0, edits[0]["at_s"] - 0.1), end=edits[-1]["at_s"] + 0.1, unit="s")
        text = desc + f" Timeline differs from the parent: {edit_txt} (content cut out or inserted)."
        if parent.trusted:
            return _f(name, Stance.supports_manipulation, 0.7, 0.6, text, lim, segment=seg, **ev)
        return _f(name, Stance.neutral, 0.0, 0.5, text + " Parent is not a trusted reference.", lim, segment=seg, **ev)
    text = desc + " Same timeline as the parent apart from the listed trims; no cuts or insertions."
    if parent.trusted:
        return _f(name, Stance.contradicts, 0.6, 0.7, text, lim + " Pixel-level edits inside frames are not "
                  "ruled out by lineage alone.", **ev)
    return _f(name, Stance.neutral, 0.0, 0.5, text, lim, **ev)


# ---------------------------------------------------------------- entry point

@register("provenance", Modality.provenance, ("video", "audio", "image", "other"))
def provenance_check(ctx: JobContext) -> list[Finding]:
    fp = fingerprint(ctx.media_path)
    others = ledger.all_versions(exclude_job=ctx.job_id)
    job = db.get_job(ctx.job_id)
    ledger.record(ctx.job_id, ctx.sha256 or "", job["filename"] if job else ctx.media_path.name, fp, ctx.work_dir)
    out = [check_sha(ctx)]
    for nm, fn in (("provenance.merkle", check_merkle), ("provenance.lineage", check_lineage)):
        try:
            out.append(fn(ctx, fp, others))
        except Exception as exc:
            out.append(_f(nm, Stance.not_assessable, None, 0.0, f"Sub-check failed: {type(exc).__name__}: {exc}",
                          "Sub-check failed."))
    return out
