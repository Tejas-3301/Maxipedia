"""Capture-signature check: is this file what a registered device signed at capture?

Reads uploads/<job>/capture_manifest.json (attached at upload). Origin chain:
  intact          - signature valid and SHA-256 equal (strong clean evidence; device-signed = trusted)
  content intact  - signature valid, file bytes differ but the Merkle root (packet content) is equal
  broken          - signature valid, some segments still match the signed leaves and others changed (localized)
  re-encoded      - signature valid, no segment matches (cannot localize; neutral, a re-encode is not proof)
  forged          - signature invalid
"""
from __future__ import annotations

import hashlib
import json

from backend.checks.registry import JobContext, register
from backend.models import Depth, Finding, Modality, Segment, Stance
from backend.provenance.device import verify_signature
from backend.provenance.fingerprint import fingerprint

NAME = "provenance.capture_signature"
LIM = ("Proves only that this content left a registered device unchanged; a compromised device or key, or "
       "staging in front of the camera, is outside what a signature can show.")


def _f(stance: Stance, score, rel: float, text: str, segment=None, **ev) -> Finding:
    return Finding(modality=Modality.provenance, check=NAME, depth=Depth.screen, stance=stance, score=score,
                   reliability=rel, rationale=text, limitations=LIM, segment=segment, coverage=1.0, evidence=ev)


@register("capture", Modality.provenance, ("video", "audio", "image", "other"))
def capture_check(ctx: JobContext) -> list[Finding]:
    path = ctx.media_path.parent / "capture_manifest.json"
    if not path.exists():
        return [_f(Stance.neutral, 0.0, 0.3, "No capture-device manifest supplied; origin chain not available.")]
    try:
        m = json.loads(path.read_text())
    except ValueError as exc:
        return [_f(Stance.not_assessable, None, 0.0, f"Capture manifest unreadable: {exc}")]
    ok, why = verify_signature(m)
    base = {"device_id": m.get("device_id"), "captured_at": m.get("captured_at"), "signed_sha256": m.get("sha256"),
            "signed_merkle_root": m.get("merkle_root")}
    if ok is None:
        return [_f(Stance.not_assessable, None, 0.0, f"Capture manifest cannot be verified: {why}.", **base)]
    if ok is False:
        return [_f(Stance.supports_manipulation, 0.6, 0.6,
                   f"Capture manifest claims device {m.get('device_id')!r} but {why}: the origin claim is forged.", **base)]
    sha = hashlib.sha256(ctx.media_path.read_bytes()).hexdigest()
    if sha == m["sha256"]:
        return [_f(Stance.contradicts, 0.9, 0.9,
                   f"Origin chain intact: device {m['device_id']!r} signed this exact file (SHA-256 {sha[:16]}...) at "
                   f"{m['captured_at']}.", chain="intact", sha256=sha, **base)]
    fp = fingerprint(ctx.media_path)
    if fp.merkle_root == m["merkle_root"]:
        return [_f(Stance.contradicts, 0.5, 0.6,
                   f"Origin content intact: file bytes differ from what device {m['device_id']!r} signed, but every "
                   f"audio/video packet matches its signed Merkle root (container / metadata changed after capture).",
                   chain="content_intact", sha256=sha, merkle_root=fp.merkle_root, **base)]
    signed = {(l["kind"], l["segment"]): l["hash"] for l in m["leaves"]}
    rows = [(l.kind, l.start, l.end, signed.get((l.kind, l.segment)) == l.hash) for l in fp.leaves]
    same = sum(r[3] for r in rows)
    changed = [r for r in rows if not r[3]]
    if same == 0:
        return [_f(Stance.neutral, 0.0, 0.5,
                   f"Origin chain broken: no segment matches what device {m['device_id']!r} signed (re-encoded "
                   f"throughout), so the change cannot be localized; a re-encode alone is not proof of manipulation.",
                   chain="reencoded", sha256=sha, **base)]
    spans = {}
    for kind, s, e, _ in changed:
        spans.setdefault(kind, []).append((s, e))
    text = "; ".join(f"{k} {min(s for s, _ in v):.0f}-{max(e for _, e in v):.0f}s" for k, v in spans.items())
    seg = Segment(start=min(r[1] for r in changed), end=max(r[2] for r in changed), unit="s")
    return [_f(Stance.supports_manipulation, 0.7, 0.7,
               f"Origin chain broken: {same}/{len(rows)} segments still match what device {m['device_id']!r} signed at "
               f"{m['captured_at']}, but {len(changed)} changed ({text}) - content was altered after capture.",
               segment=seg, chain="broken", sha256=sha, changed_segments=[list(r[:3]) for r in changed], **base)]
