from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class Modality(str, Enum):
    visual = "visual"
    audio = "audio"
    sync = "sync"
    metadata = "metadata"
    provenance = "provenance"


class Depth(str, Enum):
    screen = "screen"
    deep = "deep"


class Stance(str, Enum):
    supports_manipulation = "supports_manipulation"
    contradicts = "contradicts"
    neutral = "neutral"
    not_assessable = "not_assessable"


class Segment(BaseModel):
    start: float
    end: float
    unit: str  # e.g. "s", "frame", "sample"


class Finding(BaseModel):
    """Frozen schema: fields may be added, never renamed or removed."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    modality: Modality
    check: str
    depth: Depth = Depth.screen
    segment: Optional[Segment] = None
    region: Optional[tuple[int, int, int, int]] = None  # [x, y, w, h]
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    stance: Stance = Stance.not_assessable
    rationale: str = ""
    limitations: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    coverage: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # share of the clip this check examined at `depth`
