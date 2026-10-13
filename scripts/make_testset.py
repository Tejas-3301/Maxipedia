"""Generate the ground-truth test set into data/testset/ (+ expected.json).

Files (all derived from one controlled base clip):
  01_clean.mp4            single-encode baseline
  02_voice_swap.mp4       TTS voice replaces speech in [2.5, 5.5] s; video stream copied untouched
  03_splice.mp4           region replaced in [3.0, 4.6] s + 0.4 s of frames/audio cut at 6.0 s
  04_reencoded_3x.mp4     clean re-encoded 3 generations with different quality/GOP
  05_metadata_edited.mp4  stream-copied clean with forged/inconsistent tags and timestamps
  06_derived_a.mp4        trimmed + downscaled copy of clean
  07_derived_b.mp4        cropped + brightened copy of 06 (copy of a copy)

Base clip is synthetic by default (OpenCV-drawn scene + speech-like audio). Pass --source to use a
real talking-head clip instead (recommended for face/lip-sync checks).
expected.json is evaluation ground truth only; checks and fusion must never read it.

Usage: python scripts/make_testset.py [--source clip.mp4] [--out data/testset] [--piper-voice model.onnx]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

SR = 44100
FPS = 25
DURATION = 8.0
W, H = 640, 480
SEED = 1337

SWAP_WINDOW = (2.5, 5.5)
REGION_WINDOW = (3.0, 4.6)
REGION_FRAC = (0.6, 0.1, 0.28, 0.29)  # x, y, w, h as fractions of frame size
DROP_WINDOW = (6.0, 6.4)
TTS_TEXT = "I never authorized that transfer. Please cancel it right now."

CLEAN_CREATION = "2025-06-01T10:00:00.000000Z"
FORGED_CREATION = "2019-03-14T09:26:53.000000Z"
FORGED_VIDEO_TRACK = "2024-11-02T17:40:12Z"
FORGED_AUDIO_TRACK = "2023-01-07T08:00:00Z"
FORGED_SOFTWARE = "Adobe Premiere Pro 2023.0 (Windows)"
MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------- utils

def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-v", "error", "-y", *args])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "expected 16-bit mono"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def to_mono_44k(src: Path, dst: Path, extra_af: Optional[str] = None) -> None:
    af = ["-af", extra_af] if extra_af else []
    ffmpeg("-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), *af, "-c:a", "pcm_s16le", str(dst))


# ---------------------------------------------------------------- synthetic audio

def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    x = np.fft.irfft(spec / np.sqrt(f), n)
    return x / np.std(x)


def synth_speech(duration: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Additive-harmonic 'speech' with formants, fricatives and constant room tone. Returns (audio, envelope)."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    env = np.zeros(n)
    formants = np.zeros((3, n))
    vowels = [(730, 1090, 2440), (270, 2290, 3010), (530, 1840, 2480), (300, 870, 2240), (660, 1720, 2410)]
    fric = np.zeros(n)

    pos, count = 0.25, 0
    while pos < duration - 0.3:
        d = rng.uniform(0.14, 0.28)
        a, b = int(pos * SR), min(n, int((pos + d) * SR))
        env[a:b] = np.maximum(env[a:b], np.hanning(b - a) ** 0.7 * rng.uniform(0.6, 1.0))
        formants[:, a:b] = np.array(vowels[rng.integers(len(vowels))])[:, None]
        if rng.random() < 0.45:
            fb = min(n, a + int(0.06 * SR))
            fric[a:fb] = np.diff(rng.standard_normal(fb - a + 1)) * np.hanning(fb - a) * 0.25
        count += 1
        pos += d + rng.uniform(0.04, 0.12) + (rng.uniform(0.3, 0.5) if count % 6 == 0 else 0)

    # Smooth formant tracks so vowels glide.
    k = np.hanning(int(0.03 * SR))
    k /= k.sum()
    filled = np.where(formants == 0, np.array([500, 1500, 2500])[:, None], formants)
    formants = np.stack([np.convolve(row, k, mode="same") for row in filled])

    jitter = np.convolve(rng.standard_normal(n), np.ones(2000) / 2000, mode="same") * 40
    f0 = 118 + 12 * np.sin(2 * np.pi * 0.3 * t) + jitter
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = np.zeros(n)
    for h in range(1, 70):
        fh = h * f0
        amp = sum(np.exp(-0.5 * ((fh - formants[i]) / (80 + 40 * i)) ** 2) / (i + 1) for i in range(3))
        amp *= (fh < 9000) / np.sqrt(h)
        voice += amp * np.sin(h * phase)
    voice = voice / (np.max(np.abs(voice)) + 1e-9) * env + fric * (env > 0.05)
    voice = voice / (np.max(np.abs(voice)) + 1e-9) * 0.5
    room = _pink_noise(n, rng) * 0.004
    return voice + room, env


# ---------------------------------------------------------------- TTS backends

def tts_piper(text: str, out: Path, voice: Optional[str]) -> bool:
    voice = voice or os.environ.get("PIPER_VOICE")
    if not voice or not Path(voice).exists():
        return False
    exe = shutil.which("piper")
    cmd = [exe] if exe else [sys.executable, "-m", "piper"]
    try:
        subprocess.run([*cmd, "--model", voice, "--output_file", str(out)],
                       input=text, text=True, check=True, capture_output=True, timeout=120)
        return out.exists()
    except Exception:
        return False


def tts_sapi(text: str, out: Path) -> bool:
    if sys.platform != "win32":
        return False
    safe = text.replace("'", "''")
    path = str(out).replace("'", "''")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SetOutputToWaveFile('{path}');$s.Speak('{safe}');$s.Dispose()"
    )
    try:
        run(["powershell", "-NoProfile", "-Command", ps], timeout=60)
        return out.exists() and out.stat().st_size > 1000
    except Exception:
        return False


def tts_espeak(text: str, out: Path) -> bool:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return False
    try:
        run([exe, "-w", str(out), text], timeout=60)
        return out.exists()
    except Exception:
        return False


def tts_fallback(duration: float, out: Path, rng: np.random.Generator) -> None:
    """Vocoder-like stand-in: different pitch/timbre, hard 4 kHz band limit, digital-silence floor."""
    voice, _ = synth_speech(duration + 0.5, rng)
    spec = np.fft.rfft(voice)
    spec[np.fft.rfftfreq(len(voice), 1 / SR) > 4000] = 0
    x = np.fft.irfft(spec, len(voice))
    x = np.where(np.abs(x) < 0.01, 0.0, x)  # gate away room tone -> dead-silent floor
    write_wav(out, x)


def make_tts(text: str, window_s: float, work: Path, piper_voice: Optional[str],
             rng: np.random.Generator) -> tuple[np.ndarray, str]:
    raw = work / "tts_raw.wav"
    engine = next((name for name, fn in [
        ("piper", lambda: tts_piper(text, raw, piper_voice)),
        ("windows_sapi", lambda: tts_sapi(text, raw)),
        ("espeak", lambda: tts_espeak(text, raw)),
    ] if fn()), None)
    if engine is None:
        engine = "numpy_fallback"
        tts_fallback(window_s, raw, rng)

    conv = work / "tts_44k.wav"
    to_mono_44k(raw, conv, "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                           "silenceremove=start_periods=1:start_threshold=-45dB,areverse")
    x = read_wav(conv)
    speed = len(x) / (window_s * SR)
    if speed > 1.02:
        to_mono_44k(raw, conv, f"silenceremove=start_periods=1:start_threshold=-45dB,atempo={min(speed, 2.0):.4f}")
        x = read_wav(conv)
    target = int(window_s * SR)
    x = np.pad(x[:target], (0, max(0, target - len(x))))
    return x, engine


# ---------------------------------------------------------------- video sources

@dataclass
class Base:
    frame: Callable[[int], np.ndarray]
    n_frames: int
    w: int
    h: int
    audio: np.ndarray
    kind: str


def synthetic_base(rng: np.random.Generator) -> Base:
    n = int(DURATION * FPS)
    audio, env = synth_speech(DURATION, rng)
    hop = SR // FPS
    mouth = np.array([env[i * hop:(i + 1) * hop].mean() for i in range(n)])
    mouth = np.convolve(mouth, [0.25, 0.5, 0.25], mode="same")

    tex = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 6)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = np.stack([90 + 60 * yy / H, 110 + 40 * xx / W, 140 - 50 * yy / H], -1)
    bg = np.clip(grad + (tex.astype(np.float32) - 128) * 0.8, 0, 255).astype(np.uint8)
    cv2.rectangle(bg, (20, 330), (130, 470), (60, 80, 110), -1)  # "shelf"
    cv2.rectangle(bg, (470, 300), (620, 460), (90, 70, 60), -1)  # "cabinet"
    cv2.line(bg, (0, 320), (W, 300), (70, 90, 100), 3)

    def frame(i: int) -> np.ndarray:
        t = i / FPS
        f = bg.copy()
        bx = int(60 + 110 * t)
        cv2.circle(f, (bx, 110), 18, (40, 60, 220), -1)
        cv2.circle(f, (bx - 5, 104), 5, (160, 170, 255), -1)
        cx, cy = 200 + int(6 * np.sin(2 * np.pi * 0.4 * t)), 300 + int(3 * np.sin(2 * np.pi * 0.7 * t))
        cv2.ellipse(f, (cx, cy), (95, 125), 0, 0, 360, (120, 160, 210), -1)
        cv2.ellipse(f, (cx, cy - 135), (100, 45), 0, 180, 360, (40, 50, 70), -1)
        blink = (t % 2.7) < 0.12
        for ex in (cx - 35, cx + 35):
            cv2.ellipse(f, (ex, cy - 30), (16, 2 if blink else 9), 0, 0, 360, (250, 250, 250), -1)
            if not blink:
                cv2.circle(f, (ex, cy - 30), 5, (50, 40, 30), -1)
        cv2.line(f, (cx, cy - 15), (cx - 8, cy + 20), (90, 120, 170), 2)
        cv2.ellipse(f, (cx, cy + 55), (32, int(3 + 26 * mouth[i])), 0, 0, 360, (40, 30, 120), -1)
        noise = np.random.default_rng(SEED * 1000 + i).normal(0, 3.0, f.shape)
        return np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return Base(frame, n, W, H, audio, "synthetic")


def source_base(path: Path, work: Path) -> Base:
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    frames: list[np.ndarray] = []
    want = int(DURATION * FPS)
    i = 0
    while len(frames) < want:
        ok, f = cap.read()
        if not ok:
            break
        if int(i * FPS / src_fps) >= len(frames):  # resample to FPS by index
            if f.shape[0] > 720:
                f = cv2.resize(f, (int(f.shape[1] * 720 / f.shape[0]) // 2 * 2, 720), interpolation=cv2.INTER_AREA)
            frames.append(f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2])
        i += 1
    cap.release()
    if len(frames) < want:
        raise SystemExit(f"--source must be at least {DURATION:.0f}s long")
    wav = work / "source_audio.wav"
    ffmpeg("-i", str(path), "-t", str(DURATION), "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav))
    audio = read_wav(wav)
    audio = np.pad(audio, (0, max(0, int(DURATION * SR) - len(audio))))
    h, w = frames[0].shape[:2]
    return Base(lambda k: frames[k].copy(), want, w, h, audio, f"source:{path.name}")


# ---------------------------------------------------------------- mp4 timestamp forgery

def _boxes(buf: bytearray, start: int, end: int):
    i = start
    while i + 8 <= end:
        size = int.from_bytes(buf[i:i + 4], "big")
        typ = bytes(buf[i + 4:i + 8]).decode("latin-1")
        hdr = 8
        if size == 1:
            size, hdr = int.from_bytes(buf[i + 8:i + 16], "big"), 16
        elif size == 0:
            size = end - i
        yield typ, i, i + hdr, i + size
        i += size


def _set_box_times(buf: bytearray, body: int, when: datetime) -> None:
    """tkhd/mdhd: version(1) flags(3) creation modification (32-bit for v0, 64-bit for v1), secs since 1904."""
    secs = int((when - MP4_EPOCH).total_seconds())
    width = 8 if buf[body] == 1 else 4
    for off in (body + 4, body + 4 + width):
        buf[off:off + width] = secs.to_bytes(width, "big")


def patch_track_times(path: Path, per_handler: dict[str, str]) -> None:
    """Hex-edit per-track creation/modification times, as a forger would (ffmpeg only sets them globally)."""
    buf = bytearray(path.read_bytes())
    for typ, _, moov_b, moov_e in _boxes(buf, 0, len(buf)):
        if typ != "moov":
            continue
        for ttyp, _, trak_b, trak_e in _boxes(buf, moov_b, moov_e):
            if ttyp != "trak":
                continue
            kids = {t: (b, e) for t, _, b, e in _boxes(buf, trak_b, trak_e)}
            mdia = {t: (b, e) for t, _, b, e in _boxes(buf, *kids["mdia"])}
            handler = bytes(buf[mdia["hdlr"][0] + 8:mdia["hdlr"][0] + 12]).decode("latin-1")
            if handler in per_handler:
                when = datetime.fromisoformat(per_handler[handler].replace("Z", "+00:00"))
                _set_box_times(buf, kids["tkhd"][0], when)
                _set_box_times(buf, mdia["mdhd"][0], when)
    path.write_bytes(bytes(buf))


# ---------------------------------------------------------------- encoding

def encode(frames: list[int], get: Callable[[int], np.ndarray], w: int, h: int, audio: np.ndarray,
           out: Path, work: Path, extra: list[str]) -> None:
    wav = work / f"{out.stem}_audio.wav"
    write_wav(wav, audio)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-",
        "-i", str(wav), "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "50", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-shortest", *extra, str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for i in frames:
        proc.stdin.write(np.ascontiguousarray(get(i)).tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out.name}: {err}")


def splice_region(w: int, h: int) -> tuple[int, int, int, int]:
    fx, fy, fw, fh = REGION_FRAC
    return int(fx * w), int(fy * h), int(fw * w), int(fh * h)


# ---------------------------------------------------------------- builders

def build(out_dir: Path, source: Optional[Path], piper_voice: Optional[str]) -> dict:
    rng = np.random.default_rng(SEED)
    work = out_dir / "_work"
    work.mkdir(parents=True, exist_ok=True)
    base = source_base(source, work) if source else synthetic_base(rng)
    all_idx = list(range(base.n_frames))
    tagged = ["-metadata", f"creation_time={CLEAN_CREATION}"]
    p = {k: out_dir / v for k, v in {
        "clean": "01_clean.mp4", "voice": "02_voice_swap.mp4", "splice": "03_splice.mp4",
        "reenc": "04_reencoded_3x.mp4", "meta": "05_metadata_edited.mp4",
        "der_a": "06_derived_a.mp4", "der_b": "07_derived_b.mp4",
    }.items()}

    # 01 clean
    encode(all_idx, base.frame, base.w, base.h, base.audio, p["clean"], work, tagged)

    # 02 voice swap: replace audio window with TTS (no room tone), copy the clean video stream bit-exact
    a, b = (int(s * SR) for s in SWAP_WINDOW)
    tts, engine = make_tts(TTS_TEXT, SWAP_WINDOW[1] - SWAP_WINDOW[0], work, piper_voice, rng)
    orig_rms = np.sqrt(np.mean(base.audio[a:b] ** 2))
    tts_rms = np.sqrt(np.mean(tts[np.abs(tts) > 1e-3] ** 2)) if np.any(np.abs(tts) > 1e-3) else 1.0
    tts = tts * (orig_rms / (tts_rms + 1e-9))
    swapped = base.audio.copy()
    xf = int(0.01 * SR)
    ramp = np.linspace(0, 1, xf)
    patch_audio = tts.copy()
    patch_audio[:xf] = patch_audio[:xf] * ramp + swapped[a:a + xf] * (1 - ramp)
    patch_audio[-xf:] = patch_audio[-xf:] * ramp[::-1] + swapped[b - xf:b] * ramp
    swapped[a:b] = patch_audio
    swap_wav = work / "voice_swap.wav"
    write_wav(swap_wav, swapped)
    ffmpeg("-i", str(p["clean"]), "-i", str(swap_wav), "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", *tagged, str(p["voice"]))

    # 03 splice: static denoised+JPEG'd patch from frame 0 over a region (hides the ball) + cut frames/audio
    rx, ry, rw, rh = splice_region(base.w, base.h)
    patch = cv2.GaussianBlur(base.frame(0)[ry:ry + rh, rx:rx + rw], (5, 5), 0)
    patch = cv2.imdecode(cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 35])[1], cv2.IMREAD_COLOR)
    r0, r1 = (int(s * FPS) for s in REGION_WINDOW)
    d0, d1 = (int(s * FPS) for s in DROP_WINDOW)

    def spliced(i: int) -> np.ndarray:
        f = base.frame(i)
        if r0 <= i < r1:
            f[ry:ry + rh, rx:rx + rw] = patch
        return f

    kept = [i for i in all_idx if not d0 <= i < d1]
    cut_audio = np.concatenate([base.audio[: d0 * SR // FPS], base.audio[d1 * SR // FPS:]])
    encode(kept, spliced, base.w, base.h, cut_audio, p["splice"], work, tagged)

    # 04 three re-encode generations with differing quality/GOP
    gens = [("23", "30", "128k"), ("27", "45", "96k"), ("24", "60", "128k")]
    prev = p["clean"]
    for k, (crf, gop, abr) in enumerate(gens, 1):
        nxt = p["reenc"] if k == len(gens) else work / f"reenc_gen{k}.mp4"
        ffmpeg("-i", str(prev), "-c:v", "libx264", "-crf", crf, "-g", gop, "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", abr, *tagged, str(nxt))
        prev = nxt

    # 05 metadata/timestamps forged; streams copied bit-exact
    ffmpeg("-i", str(p["clean"]), "-map", "0", "-c", "copy", "-map_metadata", "-1",
           "-metadata", f"creation_time={FORGED_CREATION}",
           "-metadata", "comment=Original camera footage, unedited",
           "-metadata", "com.apple.quicktime.make=Apple",
           "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
           "-metadata", f"com.apple.quicktime.software={FORGED_SOFTWARE}",
           "-metadata", f"com.apple.quicktime.creationdate={FORGED_CREATION[:19]}+0000",
           "-movflags", "use_metadata_tags", str(p["meta"]))
    patch_track_times(p["meta"], {"vide": FORGED_VIDEO_TRACK, "soun": FORGED_AUDIO_TRACK})
    forged_mtime = datetime(2018, 7, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    os.utime(p["meta"], (forged_mtime, forged_mtime))  # filesystem mtime predates claimed creation

    # 06/07 derived copies: clean -> A (trim+downscale) -> B (crop+brighten)
    ffmpeg("-ss", "1.0", "-i", str(p["clean"]), "-t", "6.0", "-vf", "scale=480:-2",
           "-c:v", "libx264", "-crf", "24", "-c:a", "aac", "-b:a", "128k", str(p["der_a"]))
    ffmpeg("-i", str(p["der_a"]), "-vf", "crop=iw*0.9:ih*0.9,scale=432:-2,eq=brightness=0.04",
           "-c:v", "libx264", "-crf", "28", "-c:a", "aac", "-b:a", "96k", str(p["der_b"]))

    shutil.rmtree(work, ignore_errors=True)
    return expected(p, base, engine, (rx, ry, rw, rh))


# ---------------------------------------------------------------- ground truth

def seg(start: float, end: float) -> dict:
    return {"start": start, "end": end, "unit": "s"}


def expected(p: dict[str, Path], base: Base, engine: str, region: tuple[int, int, int, int]) -> dict:
    face_note = ("synthetic cartoon face; MediaPipe may not detect it, so sync checks may legitimately be "
                 "not_assessable" if base.kind == "synthetic" else "real face from source clip")
    drop_s = DROP_WINDOW[1] - DROP_WINDOW[0]
    files = {
        "01_clean.mp4": {
            "ground_truth": "authentic", "expected_verdict": "likely authentic",
            "acceptable_verdicts": ["likely authentic", "inconclusive"], "parent": None,
            "manipulations": [],
            "should_trigger": [],
            "should_not_trigger": ["visual", "audio", "sync", "metadata"],
            "notes": "Single encode (x264 crf18 g50, AAC 192k). Constant sensor noise and room tone throughout.",
        },
        "02_voice_swap.mp4": {
            "ground_truth": "manipulated", "expected_verdict": "likely manipulated",
            "acceptable_verdicts": ["likely manipulated", "inconclusive"], "parent": "01_clean.mp4",
            "manipulations": [{"type": "tts_voice_swap", "engine": engine, "text": TTS_TEXT,
                               "segment": seg(*SWAP_WINDOW)}],
            "should_trigger": [
                {"modality": "audio", "signal": "vocoder/TTS high-frequency cutoff or spectral mismatch",
                 "segment": seg(*SWAP_WINDOW)},
                {"modality": "audio", "signal": "noise-floor / room-tone discontinuity at splice points",
                 "segment": seg(*SWAP_WINDOW), "points_s": list(SWAP_WINDOW)},
                {"modality": "sync", "signal": "mouth aperture uncorrelated with audio envelope",
                 "segment": seg(*SWAP_WINDOW), "requires_face": True},
            ],
            "should_not_trigger": ["visual"],
            "notes": f"Video stream is a bit-exact copy of 01. 10 ms crossfades at edges. {face_note}.",
        },
        "03_splice.mp4": {
            "ground_truth": "manipulated", "expected_verdict": "likely manipulated",
            "acceptable_verdicts": ["likely manipulated", "inconclusive"], "parent": "01_clean.mp4",
            "manipulations": [
                {"type": "region_replacement", "segment": seg(*REGION_WINDOW), "region": list(region),
                 "detail": "static patch from frame 0, blurred + JPEG q35, hides moving ball (object removal)"},
                {"type": "temporal_cut", "removed_source_segment": seg(*DROP_WINDOW),
                 "cut_point_s": DROP_WINDOW[0], "detail": f"{drop_s:.1f}s of frames and matching audio removed"},
            ],
            "should_trigger": [
                {"modality": "visual", "signal": "ELA / noise-residual inconsistency in region",
                 "segment": seg(*REGION_WINDOW), "region": list(region)},
                {"modality": "visual", "signal": "frozen region while surroundings carry fresh noise",
                 "segment": seg(*REGION_WINDOW), "region": list(region)},
                {"modality": "visual", "signal": "optical-flow jump / frame drop at cut",
                 "segment": seg(DROP_WINDOW[0] - 0.04, DROP_WINDOW[0] + 0.04)},
                {"modality": "audio", "signal": "splice discontinuity at cut",
                 "segment": seg(DROP_WINDOW[0] - 0.05, DROP_WINDOW[0] + 0.05)},
            ],
            "should_not_trigger": ["metadata"],
            "notes": f"Rendered from raw frames and encoded once, so no double-compression confound. "
                     f"Output is {DURATION - drop_s:.1f}s long; region times are on the output timeline.",
        },
        "04_reencoded_3x.mp4": {
            "ground_truth": "authentic_content_recompressed", "expected_verdict": "inconclusive",
            "acceptable_verdicts": ["inconclusive", "likely authentic"], "parent": "01_clean.mp4",
            "manipulations": [{"type": "recompression", "generations": 3,
                               "params": [{"crf": 23, "gop": 30}, {"crf": 27, "gop": 45}, {"crf": 24, "gop": 60}]}],
            "should_trigger": [
                {"modality": "visual", "signal": "double/multiple compression traces"},
                {"modality": "metadata", "signal": "GOP / I-frame spacing differs from typical camera output"},
            ],
            "should_not_trigger": ["sync"],
            "notes": "Content unaltered; tests that compression alone lowers confidence rather than flipping to manipulated.",
        },
        "05_metadata_edited.mp4": {
            "ground_truth": "manipulated_metadata", "expected_verdict": "likely manipulated",
            "acceptable_verdicts": ["likely manipulated", "inconclusive"], "parent": "01_clean.mp4",
            "manipulations": [{"type": "metadata_forgery", "fields": {
                "format.creation_time (mvhd)": FORGED_CREATION,
                "video track tkhd/mdhd times": FORGED_VIDEO_TRACK,
                "audio track tkhd/mdhd times": FORGED_AUDIO_TRACK,
                "com.apple.quicktime.software": FORGED_SOFTWARE,
                "make/model": "Apple iPhone 15 Pro", "comment": "Original camera footage, unedited",
                "filesystem_mtime": "2018-07-01T12:00:00Z"}}],
            "should_trigger": [
                {"modality": "metadata", "signal": "mvhd vs per-track tkhd/mdhd creation times disagree (and tracks post-date container)"},
                {"modality": "metadata", "signal": "claimed iPhone make/model but encoder=Lavf and x264 SEI in stream"},
                {"modality": "metadata", "signal": "editing-software tag contradicts 'unedited camera footage' comment"},
                {"modality": "metadata", "signal": "filesystem mtime predates claimed creation",
                 "caveat": "mtime is lost on HTTP upload; only visible if read from disk"},
                {"modality": "provenance", "signal": "sha256 differs from 01 but perceptual hash identical"},
            ],
            "should_not_trigger": ["visual", "audio", "sync"],
            "notes": "Audio/video streams are bit-exact copies of 01; only container metadata changed (track times hex-edited).",
        },
        "06_derived_a.mp4": {
            "ground_truth": "authentic_derivative", "expected_verdict": "likely authentic",
            "acceptable_verdicts": ["likely authentic", "inconclusive"], "parent": "01_clean.mp4",
            "manipulations": [{"type": "derivation", "ops": ["trim 1.0-7.0s", "scale 480w", "re-encode crf24"]}],
            "should_trigger": [
                {"modality": "provenance", "signal": "perceptual-hash match to 01 (child); lineage edge 01 -> 06"},
            ],
            "should_not_trigger": ["sync", "audio"],
            "notes": "Benign derivative; lineage only.",
        },
        "07_derived_b.mp4": {
            "ground_truth": "authentic_derivative", "expected_verdict": "likely authentic",
            "acceptable_verdicts": ["likely authentic", "inconclusive"], "parent": "06_derived_a.mp4",
            "manipulations": [{"type": "derivation", "ops": ["crop 90%", "scale 432w", "brightness +0.04",
                                                             "re-encode crf28"]}],
            "should_trigger": [
                {"modality": "provenance", "signal": "perceptual-hash match to 06 and 01; lineage 01 -> 06 -> 07"},
            ],
            "should_not_trigger": ["sync", "audio"],
            "notes": "Copy of a copy; lineage graph should order it after 06.",
        },
    }
    for name, entry in files.items():
        entry["sha256"] = sha256(p["clean"].parent / name)
    return {
        "_note": "Evaluation ground truth only. Checks and fusion must never read this file.",
        "generator": {
            "script": "scripts/make_testset.py", "seed": SEED, "base": base.kind, "tts_engine": engine,
            "fps": FPS, "duration_s": DURATION, "resolution": [base.w, base.h], "audio_sr": SR,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "files": files,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "testset")
    ap.add_argument("--source", type=Path, help="real talking-head clip (>= 8 s) to use instead of synthetic base")
    ap.add_argument("--piper-voice", help="path to Piper .onnx voice (or set PIPER_VOICE)")
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} not found on PATH")
    args.out.mkdir(parents=True, exist_ok=True)
    exp = build(args.out, args.source, args.piper_voice)
    (args.out / "expected.json").write_text(json.dumps(exp, indent=2))
    print(f"base={exp['generator']['base']} tts={exp['generator']['tts_engine']}")
    for name, e in exp["files"].items():
        print(f"  {name:26s} {e['ground_truth']:32s} {e['sha256'][:12]}")


if __name__ == "__main__":
    main()
