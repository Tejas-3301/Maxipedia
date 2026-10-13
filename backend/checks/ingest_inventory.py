"""Dummy check proving the pipeline: reports what ingestion produced. Always neutral, zero reliability."""
from __future__ import annotations

from backend.checks.registry import JobContext, register
from backend.models import Depth, Finding, Modality, Stance


@register("ingest_inventory", Modality.provenance, ("video", "audio", "image", "other"))
def ingest_inventory(ctx: JobContext) -> list[Finding]:
    return [
        Finding(
            modality=Modality.provenance,
            check="ingest_inventory",
            depth=Depth.screen,
            stance=Stance.neutral,
            score=None,
            reliability=0.0,
            rationale="Pipeline placeholder: lists derived artifacts; carries no forensic weight.",
            limitations="Not a forensic check. Must be ignored by fusion.",
            evidence={
                "media_type": ctx.media_type,
                "sha256": ctx.sha256,
                "duration_s": ctx.duration_s,
                "frame_count": len(ctx.frame_paths()),
                "has_audio_wav": bool(ctx.audio_path and ctx.audio_path.exists()),
                "has_proxy": bool(ctx.proxy_path and ctx.proxy_path.exists()),
                "has_metadata": bool(ctx.metadata_path and ctx.metadata_path.exists()),
            },
        )
    ]
