from __future__ import annotations

import importlib
import pkgutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from backend.models import Finding, Modality, Stance


@dataclass
class JobContext:
    job_id: str
    media_path: Path  # read-only original; integrity/metadata checks must use this
    media_type: str  # video | audio | image | other
    work_dir: Path
    sha256: Optional[str] = None
    duration_s: Optional[float] = None
    frames_dir: Optional[Path] = None
    frame_sample_fps: Optional[float] = None
    audio_path: Optional[Path] = None  # 16 kHz mono wav
    proxy_path: Optional[Path] = None
    metadata_path: Optional[Path] = None  # {"ffprobe": ..., "exiftool": ...}
    deep_intervals: Optional[list[dict[str, Any]]] = None  # scheduler output; None = whole clip
    extras: dict[str, Any] = field(default_factory=dict)

    def frame_paths(self) -> list[Path]:
        return sorted(self.frames_dir.glob("frame_*.png")) if self.frames_dir else []


CheckFn = Callable[[JobContext], list[Finding]]


@dataclass(frozen=True)
class CheckSpec:
    name: str
    modality: Modality
    fn: CheckFn
    media_types: frozenset[str]


_REGISTRY: dict[str, CheckSpec] = {}
_DISCOVERED = False


def register(
    name: str,
    modality: Modality,
    media_types: tuple[str, ...] = ("video", "audio", "image"),
) -> Callable[[CheckFn], CheckFn]:
    def deco(fn: CheckFn) -> CheckFn:
        if name in _REGISTRY:
            raise ValueError(f"check already registered: {name}")
        _REGISTRY[name] = CheckSpec(name, modality, fn, frozenset(media_types))
        return fn

    return deco


def discover() -> None:
    """Import every module in backend.checks so their @register decorators run."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    import backend.checks as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name != "registry" and not mod.name.startswith("_"):
            importlib.import_module(f"backend.checks.{mod.name}")
    _DISCOVERED = True


def get_checks(media_type: Optional[str] = None) -> list[CheckSpec]:
    discover()
    specs = list(_REGISTRY.values())
    if media_type is None:
        return specs
    return [s for s in specs if media_type in s.media_types]


def _failure(spec: CheckSpec, exc: Exception) -> Finding:
    return Finding(
        modality=spec.modality,
        check=spec.name,
        stance=Stance.not_assessable,
        reliability=0.0,
        rationale="Check failed to execute.",
        limitations=f"{type(exc).__name__}: {exc}",
    )


def _safe_run(spec: CheckSpec, ctx: JobContext) -> list[Finding]:
    try:
        findings = list(spec.fn(ctx))
    except Exception as exc:
        return [_failure(spec, exc)]
    out = []
    for f in findings:
        try:
            f.model_dump_json()  # a non-serialisable finding must not sink the whole job
            out.append(f)
        except Exception as exc:
            out.append(_failure(spec, exc))
    return out


def run_checks(ctx: JobContext, max_workers: int = 4) -> list[Finding]:
    """Run every applicable check in parallel; a crashing check yields a not_assessable finding."""
    specs = get_checks(ctx.media_type)
    if not specs:
        return []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="check") as pool:
        results = pool.map(lambda s: _safe_run(s, ctx), specs)
        return [f for batch in results for f in batch]
