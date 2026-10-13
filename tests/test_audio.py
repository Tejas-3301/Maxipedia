import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from backend.checks.audio import audio_check
from backend.checks.registry import JobContext
from scripts.make_testset import synth_speech

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
SR = 44100


def _speech_with_room(seconds: float, seed: int) -> np.ndarray:
    return synth_speech(seconds, np.random.default_rng(seed))[0]


def _write(path: Path, x: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _lowpass(x: np.ndarray, hz: float) -> np.ndarray:
    s = np.fft.rfft(x)
    s[np.fft.rfftfreq(len(x), 1 / SR) > hz] = 0
    return np.fft.irfft(s, len(x))


def _run(path: Path, tmp: Path):
    return audio_check(JobContext("a", path, "audio", tmp))


def test_clean_speech_with_room_tone_does_not_fire(tmp_path: Path):
    out = _run(_write(tmp_path / "clean.wav", _speech_with_room(8, 0)), tmp_path)
    assert {f.check for f in out} == {"audio.hf_cutoff", "audio.spectral_flatness", "audio.noise_floor"}
    assert all(f.stance != "supports_manipulation" for f in out)
    assert (tmp_path / "spectrograms" / "overview.png").exists()


def test_inserted_bandlimited_segment_without_room_tone_fires(tmp_path: Path):
    x = _speech_with_room(8, 1)
    a, b = int(3 * SR), int(5.5 * SR)
    other = _speech_with_room(3, 2)[: b - a]
    x[a:b] = _lowpass(np.where(np.abs(other) < 0.01, 0.0, other), 5000)  # band-limited, no room tone
    x[int(4 * SR):int(4.6 * SR)] = 0.0
    out = {f.check: f for f in _run(_write(tmp_path / "swap.wav", x), tmp_path) if f.stance == "supports_manipulation"}
    assert "audio.hf_cutoff" in out and "audio.noise_floor" in out
    seg = out["audio.hf_cutoff"].segment
    assert seg.start <= 3.3 and seg.end >= 4.9  # end depends on where the inserted speech is active
    assert Path(out["audio.hf_cutoff"].evidence["spectrogram"]).exists()
