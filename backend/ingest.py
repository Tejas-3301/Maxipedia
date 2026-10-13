from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, BinaryIO, Optional

from backend.config import (
    ARTIFACTS_DIR,
    AUDIO_SAMPLE_RATE,
    FRAME_SAMPLE_FPS,
    MAX_ANALYSIS_SECONDS,
    MAX_UPLOAD_BYTES,
    PROXIES_DIR,
    PROXY_MAX_HEIGHT,
    TOOL_TIMEOUT_S,
    UPLOADS_DIR,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".gif"}
CHUNK = 1024 * 1024


class UploadTooLarge(Exception):
    pass


def _safe_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,8}", suffix) else ".bin"


def store_original(job_id: str, filename: str, src: BinaryIO) -> tuple[Path, str, int]:
    """Stream upload to disk, hash on the fly, then mark read-only. Returns (path, sha256, size)."""
    dest_dir = UPLOADS_DIR / job_id
    dest_dir.mkdir(parents=True, exist_ok=False)
    dest = dest_dir / f"original{_safe_suffix(filename)}"
    h = hashlib.sha256()
    size = 0
    with open(dest, "wb") as out:
        while chunk := src.read(CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise UploadTooLarge(f"upload exceeds {MAX_UPLOAD_BYTES} bytes")
            h.update(chunk)
            out.write(chunk)
    os.chmod(dest, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return dest, h.hexdigest(), size


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=TOOL_TIMEOUT_S, check=True)


def _err(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        return (exc.stderr or str(exc)).strip()[-500:]
    return f"{type(exc).__name__}: {exc}"


def ffprobe(path: Path) -> dict[str, Any]:
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-show_chapters", "-show_programs", str(path),
    ])
    return json.loads(out.stdout or "{}")


def exiftool(path: Path) -> dict[str, Any]:
    if shutil.which("exiftool") is None:
        return {"available": False, "reason": "exiftool not installed"}
    out = _run(["exiftool", "-j", "-G1", "-a", "-u", "-n", str(path)])
    data = json.loads(out.stdout or "[]")
    return {"available": True, "tags": data[0] if data else {}}


def classify_media(path: Path, probe: dict[str, Any]) -> str:
    streams = probe.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fmt = probe.get("format", {}).get("format_name", "")
    if path.suffix.lower() in IMAGE_EXTS or "image2" in fmt or fmt.endswith("_pipe"):
        return "image" if has_video else "other"
    if has_video:
        # Cover-art-only containers (e.g. mp3 with picture) count as audio.
        real_video = [s for s in streams if s.get("codec_type") == "video"
                      and not s.get("disposition", {}).get("attached_pic")]
        if real_video:
            return "video"
    if has_audio:
        return "audio"
    return "other"


def _duration(probe: dict[str, Any]) -> Optional[float]:
    try:
        return float(probe.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        return None


def extract_frames(src: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    _run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(MAX_ANALYSIS_SECONDS),
        "-map", "0:v:0", "-vf", f"fps={FRAME_SAMPLE_FPS}", "-fps_mode", "passthrough",
        str(out_dir / "frame_%06d.png"),
    ])
    return len(list(out_dir.glob("frame_*.png")))


def extract_audio(src: Path, out: Path) -> None:
    _run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(MAX_ANALYSIS_SECONDS),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-c:a", "pcm_s16le", str(out),
    ])


def make_proxy(src: Path, out: Path, media_type: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale=-2:'min({PROXY_MAX_HEIGHT},ih)'"
    if media_type == "image":
        _run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", scale, "-frames:v", "1", str(out)])
        return
    _run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", str(MAX_ANALYSIS_SECONDS),
        "-map", "0:v:0", "-map", "0:a:0?", "-vf", scale,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out),
    ])


def preprocess(job_id: str, original: Path) -> dict[str, Any]:
    """Derive analysis artifacts. Each step fails independently; failures are recorded, not raised."""
    work = ARTIFACTS_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)
    errors: dict[str, str] = {}
    arts: dict[str, Any] = {"work_dir": str(work), "errors": errors}

    try:
        probe = ffprobe(original)
    except Exception as exc:
        probe, errors["ffprobe"] = {}, _err(exc)
    try:
        exif = exiftool(original)
    except Exception as exc:
        exif = {"available": True, "error": _err(exc)}
    meta_path = work / "metadata.json"
    meta_path.write_text(json.dumps({"ffprobe": probe, "exiftool": exif}, indent=2, default=str))
    arts["metadata_path"] = str(meta_path)

    media_type = classify_media(original, probe)
    streams = probe.get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    arts.update(media_type=media_type, duration_s=_duration(probe), has_audio=has_audio)

    if media_type in ("video", "image"):
        frames_dir = work / "frames"
        try:
            if media_type == "image":
                frames_dir.mkdir(exist_ok=True)
                _run(["ffmpeg", "-v", "error", "-y", "-i", str(original), "-frames:v", "1",
                      str(frames_dir / "frame_000001.png")])
                n = 1
            else:
                n = extract_frames(original, frames_dir)
            arts.update(frames_dir=str(frames_dir), frame_count=n,
                        frame_sample_fps=None if media_type == "image" else FRAME_SAMPLE_FPS)
        except Exception as exc:
            errors["frames"] = _err(exc)

        proxy = PROXIES_DIR / f"{job_id}{'.png' if media_type == 'image' else '.mp4'}"
        try:
            make_proxy(original, proxy, media_type)
            arts["proxy_path"] = str(proxy)
        except Exception as exc:
            errors["proxy"] = _err(exc)

    if has_audio:
        wav = work / "audio_16k_mono.wav"
        try:
            extract_audio(original, wav)
            arts["audio_path"] = str(wav)
        except Exception as exc:
            errors["audio"] = _err(exc)

    arts["truncated_for_analysis"] = bool(arts["duration_s"] and arts["duration_s"] > MAX_ANALYSIS_SECONDS)
    return arts
