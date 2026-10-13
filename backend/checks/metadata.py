"""Metadata & encoding inspection.

Sub-checks (one finding each): timestamps, stream durations, edit lists, encoder strings,
re-encoding generation hints, C2PA manifest. All read the original bytes, never the proxy.
Every finding quotes exact field names and values in `evidence.fields` and in the rationale.

score = strength of the evidence for the stated stance (0-1); reliability = how far this kind of
evidence can be trusted in general (metadata is easy to forge or strip, so it stays moderate).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from backend.checks import _mp4
from backend.checks.registry import JobContext, register
from backend.models import Depth, Finding, Modality, Segment, Stance

TIME_TOL = timedelta(seconds=60)
DUR_TOL_S = 0.2
START_TOL_S = 0.1
SPIKE_RATIO = 1.6
STRONG_SPIKE_RATIO = 2.0
HEAD_BYTES = 4 * 1024 * 1024

EDITORS = [
    "premiere", "after effects", "final cut", "davinci", "resolve", "imovie", "capcut", "vegas",
    "filmora", "camtasia", "shotcut", "kdenlive", "openshot", "movavi", "clipchamp", "inshot",
    "vsdc", "avid", "media composer", "lightworks", "photoshop", "lightroom", "gimp", "canva",
]
TRANSCODERS = ["lavf", "lavc", "ffmpeg", "handbrake", "x264", "x265", "libx264", "libx265", "mencoder"]
ORIGINALITY = re.compile(r"\b(original|unedited|unmodified|untouched|raw footage|camera original)\b", re.I)


# ---------------------------------------------------------------- helpers

def _fmt(v: Any) -> str:
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(v)


def _f(field: str, value: Any) -> dict[str, str]:
    return {"field": field, "value": _fmt(value)}


def _q(field: str, value: Any) -> str:
    return f"{field} = {_fmt(value)!r}"


def _combine(weights: list[float]) -> float:
    p = 1.0
    for w in weights:
        p *= 1 - w
    return round(min(0.95, 1 - p), 3)


def _parse_dt(s: str) -> Optional[datetime]:
    s = s.strip()
    for cand in (s.replace("Z", "+00:00"), re.sub(r"^(\d{4}):(\d{2}):(\d{2})", r"\1-\2-\3", s)):
        try:
            d = datetime.fromisoformat(re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", cand))
        except ValueError:
            continue
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return None


def _finding(check: str, stance: Stance, score: Optional[float], reliability: float, rationale: str,
             limitations: str, fields: list[dict], modality: Modality = Modality.metadata,
             segment: Optional[Segment] = None, **extra: Any) -> Finding:
    return Finding(modality=modality, check=check, depth=Depth.screen, stance=stance, score=score,
                   reliability=reliability, rationale=rationale, limitations=limitations,
                   segment=segment, evidence={"fields": fields, **extra})


def _not_assessable(check: str, why: str, modality: Modality = Modality.metadata) -> Finding:
    return _finding(check, Stance.not_assessable, None, 0.0, why, why, [], modality)


def _load_meta(ctx: JobContext) -> dict[str, Any]:
    if ctx.metadata_path and ctx.metadata_path.exists():
        return json.loads(ctx.metadata_path.read_text())
    from backend.ingest import exiftool, ffprobe
    return {"ffprobe": ffprobe(ctx.media_path), "exiftool": exiftool(ctx.media_path)}


def _streams(probe: dict, kind: str) -> list[dict]:
    return [s for s in probe.get("streams", []) if s.get("codec_type") == kind
            and not s.get("disposition", {}).get("attached_pic")]


def _float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _exif_tags(meta: dict) -> dict[str, Any]:
    """exiftool tags minus System:/File: groups, which describe our stored copy, not the media."""
    ex = meta.get("exiftool") or {}
    tags = ex.get("tags") or {} if ex.get("available") else {}
    return {k: v for k, v in tags.items() if not k.startswith(("System:", "File:", "ExifTool:", "SourceFile"))}


# ---------------------------------------------------------------- timestamps

def check_timestamps(ctx: JobContext, meta: dict, movie: Optional[_mp4.Movie]) -> Finding:
    name = "metadata.timestamps"
    probe = meta.get("ffprobe", {})
    stamps: list[tuple[str, datetime]] = []
    pairs: list[tuple[str, datetime, str, datetime]] = []  # (creation field, value, modification field, value)

    if movie:
        if movie.mvhd_creation:
            stamps.append(("moov/mvhd.creation_time", movie.mvhd_creation))
        if movie.mvhd_creation and movie.mvhd_modification:
            pairs.append(("moov/mvhd.creation_time", movie.mvhd_creation,
                          "moov/mvhd.modification_time", movie.mvhd_modification))
        for t in movie.tracks:
            for box, c, m in (("tkhd", t.tkhd_creation, t.tkhd_modification),
                              ("mdhd", t.mdhd_creation, t.mdhd_modification)):
                path = f"{t.path}/{box}" if box == "tkhd" else f"{t.path}/mdia/{box}"
                if c:
                    stamps.append((f"{path}.creation_time", c))
                if c and m:
                    pairs.append((f"{path}.creation_time", c, f"{path}.modification_time", m))
    else:
        for s in probe.get("streams", []):
            v = (s.get("tags") or {}).get("creation_time")
            if v and (d := _parse_dt(v)):
                stamps.append((f"ffprobe:streams[{s.get('index')}].tags.creation_time", d))

    tag_stamps: list[tuple[str, datetime]] = []
    for k, v in (probe.get("format", {}).get("tags") or {}).items():
        if "creation" in k.lower() or k.lower() in ("date", "date_recorded"):
            for part in str(v).split(";"):
                if d := _parse_dt(part):
                    tag_stamps.append((f"ffprobe:format.tags.{k}", d))
    for k, v in _exif_tags(meta).items():
        if re.search(r"(Create|Modify|DateTimeOriginal|CreationDate)", k) and isinstance(v, str):
            if (d := _parse_dt(v)) and d.year > 1904:
                tag_stamps.append((f"exiftool:{k}", d))

    all_stamps = stamps + tag_stamps
    fields = [_f(k, v) for k, v in all_stamps]
    if not all_stamps:
        return _finding(name, Stance.neutral, 0.0, 0.3,
                        "No embedded creation/modification times (unset or stripped).",
                        "Many exporters and platforms zero or strip timestamps; absence is not evidence either way.",
                        fields)

    issues: list[tuple[float, str]] = []
    for cf, c, mf, m in pairs:
        if m < c - TIME_TOL:
            issues.append((0.6, f"{_q(mf, m)} is earlier than {_q(cf, c)}"))

    ref = next(((k, v) for k, v in stamps if k == "moov/mvhd.creation_time"), None)
    if ref:
        for k, v in stamps:
            if k != ref[0] and abs(v - ref[1]) > TIME_TOL:
                later = "after" if v > ref[1] else "before"
                w = 0.65 if abs(v - ref[1]) > timedelta(days=1) else 0.45
                issues.append((w, f"{_q(k, v)} is {abs(v - ref[1])} {later} {_q(*ref)}"))
        for k, v in tag_stamps:
            delta = abs(v - ref[1])
            if delta > TIME_TOL:
                tz_like = delta <= timedelta(hours=14) and delta.total_seconds() % 900 < 61
                w = 0.25 if tz_like else 0.5
                note = " (could be a timezone offset)" if tz_like else ""
                issues.append((w, f"{_q(k, v)} differs from {_q(*ref)} by {delta}{note}"))

    now = datetime.now(timezone.utc)
    for k, v in all_stamps:
        if v > now + timedelta(days=1):
            issues.append((0.6, f"{_q(k, v)} is in the future"))
        elif v.year < 1995:
            issues.append((0.3, f"{_q(k, v)} predates consumer digital video"))

    lim = ("Container times are trivially editable and are rewritten by remuxing; "
           "filesystem times are not available after upload.")
    if issues:
        issues.sort(key=lambda x: -x[0])
        return _finding(name, Stance.supports_manipulation, _combine([w for w, _ in issues]), 0.6,
                        "Timestamp inconsistencies: " + "; ".join(t for _, t in issues) + ".", lim, fields,
                        issues=[t for _, t in issues])
    if len(all_stamps) >= 2:
        return _finding(name, Stance.contradicts, 0.2, 0.3,
                        f"{len(all_stamps)} embedded timestamps agree within {TIME_TOL.seconds}s "
                        f"(e.g. {_q(*all_stamps[0])}).", lim, fields)
    return _finding(name, Stance.neutral, 0.0, 0.3,
                    f"Only one embedded timestamp: {_q(*all_stamps[0])}; nothing to cross-check.", lim, fields)


# ---------------------------------------------------------------- durations

def check_durations(ctx: JobContext, meta: dict, movie: Optional[_mp4.Movie]) -> Finding:
    name = "metadata.durations"
    probe = meta.get("ffprobe", {})
    v, a = _streams(probe, "video"), _streams(probe, "audio")
    if not v or not a or ctx.media_type != "video":
        return _not_assessable(name, "Needs both an audio and a video stream.")
    vs, as_ = v[0], a[0]
    vf, af = f"ffprobe:streams[{vs['index']}]", f"ffprobe:streams[{as_['index']}]"
    vd, ad = _float(vs.get("duration")), _float(as_.get("duration"))
    v0, a0 = _float(vs.get("start_time")) or 0.0, _float(as_.get("start_time")) or 0.0
    fd = _float(probe.get("format", {}).get("duration"))
    fields = [_f(f"{vf}.duration", vd), _f(f"{af}.duration", ad), _f(f"{vf}.start_time", v0),
              _f(f"{af}.start_time", a0), _f("ffprobe:format.duration", fd)]
    if vd is None or ad is None:
        return _not_assessable(name, "Stream durations not reported by the container.")

    issues: list[tuple[float, str]] = []
    diff = abs(vd - ad)
    seg = None
    if diff > DUR_TOL_S:
        w = min(0.75, 0.3 + 0.45 * min(1.0, (diff - DUR_TOL_S) / 1.0))
        longer = "audio" if ad > vd else "video"
        issues.append((w, f"{_q(vf + '.duration', vd)} vs {_q(af + '.duration', ad)}: "
                          f"{longer} is {diff:.3f}s longer"))
        seg = Segment(start=min(v0 + vd, a0 + ad), end=max(v0 + vd, a0 + ad), unit="s")
    if abs(v0 - a0) > START_TOL_S:
        issues.append((0.4, f"{_q(vf + '.start_time', v0)} vs {_q(af + '.start_time', a0)}: "
                            f"streams start {abs(v0 - a0):.3f}s apart"))
    if fd and fd - max(v0 + vd, a0 + ad) > 0.5:
        issues.append((0.3, f"{_q('ffprobe:format.duration', fd)} exceeds the longest stream by "
                            f"{fd - max(v0 + vd, a0 + ad):.3f}s"))

    lim = "Small offsets are normal (AAC priming, B-frame delay); some cameras record audio slightly longer."
    if issues:
        return _finding(name, Stance.supports_manipulation, _combine([w for w, _ in issues]), 0.5,
                        "Stream timing mismatch: " + "; ".join(t for _, t in issues) + ".", lim, fields,
                        segment=seg, issues=[t for _, t in issues])
    return _finding(name, Stance.contradicts, 0.15, 0.3,
                    f"Audio and video durations agree ({vd:.3f}s vs {ad:.3f}s, start offset {abs(v0 - a0):.3f}s).",
                    lim + " Agreement is expected after any clean re-export, so it is weak evidence.", fields)


# ---------------------------------------------------------------- edit lists

def check_edit_lists(ctx: JobContext, meta: dict, movie: Optional[_mp4.Movie]) -> Finding:
    name = "metadata.edit_list"
    if movie is None:
        return _not_assessable(name, "Edit lists only exist in MP4/MOV containers.")
    fields: list[dict] = []
    issues: list[tuple[float, str]] = []
    mts = movie.mvhd_timescale or 1
    for t in movie.tracks:
        if not t.edits:
            continue
        ts = t.mdhd_timescale or 1
        base = f"{t.path}/edts/elst"
        for i, e in enumerate(t.edits):
            fields.append(_f(f"{base}.entries[{i}]",
                             f"segment_duration={e.segment_duration}/{mts}s media_time={e.media_time}/{ts}s "
                             f"rate={e.rate:g}"))
        real = [e for e in t.edits if e.media_time != -1]
        empty = [e for e in t.edits if e.media_time == -1]
        if len(real) > 1:
            issues.append((0.6, f"{base} has {len(real)} non-empty edit segments "
                                f"(media_time values {[e.media_time for e in real]}): the track is "
                                f"cut/rearranged at container level"))
        lead = sum(e.segment_duration for e in empty) / mts
        if lead > 0.1:
            issues.append((0.4, f"{base} empty edit delays the track by {lead:.3f}s"))
        if real and real[0].media_time / ts > 0.5:
            issues.append((0.35, f"{base}.entries media_time = {real[0].media_time} "
                                 f"(={real[0].media_time / ts:.3f}s) skips the start of the media"))
        if any(abs(e.rate - 1) > 1e-6 for e in t.edits):
            issues.append((0.5, f"{base} has playback rate != 1"))

    lim = "Single short edits are normal (encoder delay / AAC priming). Re-encoding removes edit lists entirely."
    if issues:
        return _finding(name, Stance.supports_manipulation, _combine([w for w, _ in issues]), 0.5,
                        "Edit-list anomalies: " + "; ".join(t for _, t in issues) + ".", lim, fields,
                        issues=[t for _, t in issues])
    msg = ("Edit lists contain only standard priming/delay offsets." if fields
           else "No edit lists present.")
    return _finding(name, Stance.neutral, 0.0, 0.4, msg, lim, fields)


# ---------------------------------------------------------------- encoder strings

def _bitstream_signatures(path: Path) -> dict[str, str]:
    with open(path, "rb") as f:
        head = f.read(HEAD_BYTES)
    sigs: dict[str, str] = {}
    if m := re.search(rb"x264 - core \d+[^\x00]{0,2000}", head):
        sigs["bitstream:h264_sei.x264"] = m.group().decode("latin-1", "replace")
    if m := re.search(rb"x265 \(build \d+\)[^\x00]{0,1500}", head):
        sigs["bitstream:hevc_sei.x265"] = m.group().decode("latin-1", "replace")
    if m := re.search(rb"Lavc\d+\.\d+\.\d+", head):
        sigs["bitstream:Lavc"] = m.group().decode()
    return sigs


def _x264_opts(sei: str) -> dict[str, str]:
    opts = sei.split("options:", 1)[1] if "options:" in sei else ""
    return dict(kv.split("=", 1) for kv in opts.split() if "=" in kv)


def check_encoders(ctx: JobContext, meta: dict, movie: Optional[_mp4.Movie]) -> Finding:
    name = "metadata.encoder"
    probe = meta.get("ffprobe", {})
    strings: dict[str, str] = {}
    for k, v in (probe.get("format", {}).get("tags") or {}).items():
        strings[f"ffprobe:format.tags.{k}"] = str(v)
    for s in probe.get("streams", []):
        for k, v in (s.get("tags") or {}).items():
            if k.lower() in ("encoder", "handler_name", "vendor_id", "comment"):
                strings[f"ffprobe:streams[{s.get('index')}].tags.{k}"] = str(v)
    for k, v in _exif_tags(meta).items():
        if re.search(r"(Software|Encoder|Make|Model|CreatorTool|Compressor|Comment|Description|History)", k):
            strings[f"exiftool:{k}"] = str(v)
    strings.update(_bitstream_signatures(ctx.media_path))

    def hits(words: list[str], keys: Callable[[str], bool] = lambda k: True) -> list[tuple[str, str]]:
        return [(k, v) for k, v in strings.items() if keys(k) and any(w in v.lower() for w in words)]

    make_keys = lambda k: bool(re.search(r"(make|model)$", k.lower().split(".")[-1].split(":")[-1]))
    camera = [(k, v) for k, v in strings.items() if make_keys(k)]
    editors = hits(EDITORS, lambda k: not make_keys(k))
    transcoders = hits(TRANSCODERS, lambda k: not make_keys(k) and "handler_name" not in k)
    claims = [(k, v) for k, v in strings.items() if ORIGINALITY.search(v)
              and re.search(r"(comment|description|title|caption)", k.lower())]

    interesting = {k for k, _ in camera + editors + transcoders + claims}
    fields = [_f(k, v if len(v) < 300 else v[:300] + "…") for k, v in strings.items() if k in interesting]
    sei = strings.get("bitstream:h264_sei.x264")
    extra = {"x264_options": {k: v for k, v in _x264_opts(sei).items()
                              if k in ("crf", "keyint", "keyint_min", "bframes", "rc", "bitrate", "preset")}} if sei else {}

    issues: list[tuple[float, str]] = []
    sw = transcoders + editors
    if camera and sw:
        shown = " and ".join(_q(k, v[:60]) for k, v in sw[:2])
        issues.append((0.7, f"{' / '.join(_q(k, v) for k, v in camera)} claims a camera origin, but "
                            f"{shown} show software encoding/muxing; camera originals use hardware encoders"))
    if editors:
        issues.append((0.4, "exported from editing software: " + "; ".join(_q(k, v) for k, v in editors)))
    if claims and sw:
        issues.append((0.55, f"{_q(*claims[0])} asserts the file is original, contradicted by {_q(*sw[0])}"))

    lim = ("Encoder tags can be stripped or forged; an export/transcode is not by itself a content edit "
           "(platform uploads are re-encoded too).")
    if issues:
        return _finding(name, Stance.supports_manipulation, _combine([w for w, _ in issues]), 0.55,
                        "Encoder/software inconsistencies: " + "; ".join(t for _, t in issues) + ".", lim,
                        fields, issues=[t for _, t in issues], **extra)
    if transcoders:
        return _finding(name, Stance.neutral, 0.0, 0.5,
                        "Last written by a software transcoder (" + "; ".join(_q(k, v[:80]) for k, v in transcoders[:3])
                        + "): consistent with an export or re-encode, not a camera original.", lim, fields, **extra)
    if camera:
        return _finding(name, Stance.contradicts, 0.25, 0.35,
                        "Camera make/model present with no software encoder traces: "
                        + "; ".join(_q(k, v) for k, v in camera) + ".", lim, fields, **extra)
    return _finding(name, Stance.neutral, 0.0, 0.3, "No encoder, software or device strings found.", lim, fields)


# ---------------------------------------------------------------- re-encoding generations

def _frame_stats(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+60",
         "-show_entries", "frame=pict_type,pkt_size,key_frame", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=300, check=True,
    ).stdout
    return json.loads(out).get("frames", [])


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2


def check_generations(ctx: JobContext, meta: dict, movie: Optional[_mp4.Movie]) -> Finding:
    """Double-compression hint: P-frames that were I-frames in an earlier encode come out unusually large."""
    name = "metadata.reencode_generations"
    if ctx.media_type != "video":
        return _not_assessable(name, "GOP analysis applies to video only.")
    frames = _frame_stats(ctx.media_path)
    if len(frames) < 30:
        return _not_assessable(name, f"Too few frames ({len(frames)}) for GOP analysis.")
    keys = [i for i, f in enumerate(frames) if f.get("key_frame")]
    p_idx = [i for i, f in enumerate(frames) if f.get("pict_type") == "P"]
    sizes = [float(frames[i]["pkt_size"]) for i in p_idx]
    if len(p_idx) < 20:
        return _not_assessable(name, "Too few P-frames (intra-only or B-heavy stream).")

    spikes: list[tuple[int, float]] = []
    for k, i in enumerate(p_idx):
        local = _median(sizes[max(0, k - 4):k + 5])
        ratio = sizes[k] / local if local else 0.0
        if ratio > SPIKE_RATIO and all(abs(i - j) > 1 for j in keys):
            spikes.append((i, round(ratio, 2)))
    strong = [s for s in spikes if s[1] >= STRONG_SPIKE_RATIO]

    gop_now = round(_median([b - a for a, b in zip(keys, keys[1:])])) if len(keys) > 1 else None
    gop_txt_now = f"current GOP {gop_now}" if gop_now else "single I-frame in analysed span"
    # A prior GOP p should put spikes near most of its multiples, not just a few (small p matches anything).
    periods = []
    n = len(frames)
    for p in range(12, n // 2 + 1):
        if gop_now and abs(p - gop_now) <= 2:
            continue
        multiples = list(range(p, n - 3, p))
        hit = [i for i, _ in spikes if any(abs(i - m) <= 3 for m in multiples)]
        frac = len({m for m in multiples if any(abs(i - m) <= 3 for i, _ in spikes)}) / len(multiples)
        if len(hit) >= 2 and frac >= 0.5:
            periods.append((frac, len(hit), p, hit))
    periods.sort(key=lambda x: (-x[0], -x[1], x[2]))
    distinct: list[tuple[int, int, list[int]]] = []
    for _, h, p, hit in periods:
        if all(abs(p - q) > 3 for _, q, _ in distinct):
            distinct.append((h, p, hit))
    periods = distinct
    best = periods[:2]

    fps = ctx.frame_sample_fps  # sampled fps, not native; native comes from ffprobe
    v = _streams(meta.get("ffprobe", {}), "video")
    rate = v[0].get("avg_frame_rate", "0/1") if v else "0/1"
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else fps
    fields = [_f("ffprobe:frames[key_frame=1].index", keys[:20]),
              _f("ffprobe:frames[pict_type=P].pkt_size spikes (frame, size/local median)", spikes[:20])]
    lim = ("Scene cuts, flashes or fast motion also inflate P-frames; an earlier encode with GOP aligned to the "
           "current one leaves no trace; low-quality re-encodes can erase the pattern.")
    extra = {"current_gop_frames": gop_now, "spikes": spikes, "candidate_prior_gops": [(p, h) for _, p, h in best]}

    if len(strong) >= 2:
        segs = ", ".join(f"frame {i} ({i / fps:.2f}s, x{r})" if fps else f"frame {i} (x{r})" for i, r in strong[:6])
        gop_txt = "; spacing fits a prior GOP of " + " or ".join(f"{p} frames" for _, p, _ in best) if best else ""
        score = round(min(0.7, 0.25 + 0.08 * len(strong) + (0.1 if best else 0)), 3)
        return _finding(name, Stance.supports_manipulation, score, 0.4,
                        f"{len(strong)} P-frames are >= {STRONG_SPIKE_RATIO}x their local median size away from "
                        f"current I-frames ({gop_txt_now}): {segs}{gop_txt}. Pattern indicates at least one "
                        f"earlier compression generation (re-encoding, not necessarily content editing).",
                        lim, fields, **extra)
    return _finding(name, Stance.neutral, 0.0, 0.4,
                    f"No periodic P-frame size spikes ({gop_txt_now}); no double-compression trace found.",
                    lim + " Absence does not prove a single generation.", fields, **extra)


# ---------------------------------------------------------------- C2PA

def check_c2pa(ctx: JobContext, meta: dict, movie: Optional[_mp4.Movie]) -> Finding:
    name = "provenance.c2pa"
    tool = shutil.which("c2patool")
    if tool:
        r = subprocess.run([tool, str(ctx.media_path)], capture_output=True, text=True, timeout=120)
        if r.returncode != 0 or not r.stdout.strip().startswith("{"):
            msg = (r.stderr or r.stdout).strip()[:300]
            return _finding(name, Stance.neutral, 0.0, 0.7, f"No C2PA manifest found (c2patool: {msg!r}).",
                            "Most media carry no C2PA manifest; absence is not evidence.", [],
                            Modality.provenance)
        store = json.loads(r.stdout)
        active = store.get("active_manifest")
        man = (store.get("manifests") or {}).get(active, {})
        status = store.get("validation_status") or []
        actions = [a.get("action") for asr in man.get("assertions", []) if asr.get("label", "").startswith("c2pa.actions")
                   for a in asr.get("data", {}).get("actions", [])]
        fields = [_f("c2pa.active_manifest", active), _f("c2pa.claim_generator", man.get("claim_generator")),
                  _f("c2pa.signature_info.issuer", (man.get("signature_info") or {}).get("issuer")),
                  _f("c2pa.signature_info.time", (man.get("signature_info") or {}).get("time")),
                  _f("c2pa.actions", actions), _f("c2pa.validation_status", [s.get("code") for s in status])]
        if status:
            return _finding(name, Stance.supports_manipulation, 0.8, 0.8,
                            f"C2PA manifest fails validation: {_q('c2pa.validation_status', [s.get('code') for s in status])}.",
                            "Validation failures can also come from unsupported or expired certificates.",
                            fields, Modality.provenance, manifest=store)
        return _finding(name, Stance.contradicts, 0.6, 0.7,
                        f"Valid C2PA manifest signed by {(man.get('signature_info') or {}).get('issuer')!r}, "
                        f"generator {man.get('claim_generator')!r}, declared actions {actions}.",
                        "A valid manifest proves who signed the claims, not that the depicted content is true.",
                        fields, Modality.provenance, manifest=store)

    with open(ctx.media_path, "rb") as f:
        head = f.read(HEAD_BYTES)
        f.seek(max(0, ctx.media_path.stat().st_size - HEAD_BYTES))
        tail = f.read()
    present = bool(movie and movie.has_jumbf) or any(b"c2pa" in blob and b"jumb" in blob for blob in (head, tail))
    if present:
        return _finding(name, Stance.not_assessable, None, 0.0,
                        "C2PA/JUMBF manifest bytes present but c2patool is not installed, so it cannot be validated.",
                        "Install c2patool to verify signature and actions.", [_f("bytes", "jumb+c2pa")],
                        Modality.provenance)
    return _finding(name, Stance.neutral, 0.0, 0.5, "No C2PA manifest found (byte scan; c2patool not installed).",
                    "Most media carry no C2PA manifest; absence is not evidence.", [], Modality.provenance)


# ---------------------------------------------------------------- entry point

SUBCHECKS = [
    ("metadata.timestamps", check_timestamps),
    ("metadata.durations", check_durations),
    ("metadata.edit_list", check_edit_lists),
    ("metadata.encoder", check_encoders),
    ("metadata.reencode_generations", check_generations),
    ("provenance.c2pa", check_c2pa),
]


@register("metadata", Modality.metadata, ("video", "audio", "image", "other"))
def metadata_check(ctx: JobContext) -> list[Finding]:
    meta = _load_meta(ctx)
    try:
        movie = _mp4.parse(ctx.media_path)
    except Exception:
        movie = None
    out: list[Finding] = []
    for name, fn in SUBCHECKS:
        try:
            out.append(fn(ctx, meta, movie))
        except Exception as exc:
            out.append(_not_assessable(name, f"Sub-check failed: {type(exc).__name__}: {exc}",
                                       Modality.provenance if name.startswith("provenance") else Modality.metadata))
    return out
