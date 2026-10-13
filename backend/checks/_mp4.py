"""Minimal ISO-BMFF (MP4/MOV) box reader: movie/track times, handlers, edit lists, top-level layout."""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Iterator, Optional

MP4_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)
CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl", "edts", "udta"}
MAX_BOX_READ = 64 * 1024 * 1024


@dataclass
class Edit:
    segment_duration: int  # movie timescale units
    media_time: int  # media timescale units; -1 = empty edit
    rate: float


@dataclass
class Track:
    index: int
    handler: Optional[str] = None
    tkhd_creation: Optional[datetime] = None
    tkhd_modification: Optional[datetime] = None
    tkhd_duration: Optional[int] = None
    mdhd_creation: Optional[datetime] = None
    mdhd_modification: Optional[datetime] = None
    mdhd_timescale: Optional[int] = None
    mdhd_duration: Optional[int] = None
    edits: list[Edit] = field(default_factory=list)

    @property
    def path(self) -> str:
        return f"moov/trak[{self.index}]({self.handler or '?'})"


@dataclass
class Movie:
    top_level: list[str] = field(default_factory=list)
    mvhd_creation: Optional[datetime] = None
    mvhd_modification: Optional[datetime] = None
    mvhd_timescale: Optional[int] = None
    mvhd_duration: Optional[int] = None
    tracks: list[Track] = field(default_factory=list)
    has_jumbf: bool = False  # C2PA manifests live in JUMBF ('jumb') boxes


def _mp4_time(secs: int) -> Optional[datetime]:
    if secs <= 0:
        return None
    try:
        return MP4_EPOCH + timedelta(seconds=secs)
    except OverflowError:
        return None


def _iter(buf: bytes, start: int, end: int) -> Iterator[tuple[str, int, int]]:
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack(">Q", buf[i + 8:i + 16])[0], 16
        elif size == 0:
            size = end - i
        if size < hdr:
            return
        yield typ.decode("latin-1"), i + hdr, min(i + size, end)
        i += size


def _times(buf: bytes, body: int) -> tuple[int, int, int]:
    """Returns (version, creation, modification) for mvhd/tkhd/mdhd."""
    v = buf[body]
    if v == 1:
        c, m = struct.unpack(">QQ", buf[body + 4:body + 20])
    else:
        c, m = struct.unpack(">II", buf[body + 4:body + 12])
    return v, c, m


def _parse_moov(buf: bytes, movie: Movie) -> None:
    for typ, b, e in _iter(buf, 0, len(buf)):
        if typ == "mvhd":
            v, c, m = _times(buf, b)
            o = b + (20 if v == 1 else 12)
            movie.mvhd_creation, movie.mvhd_modification = _mp4_time(c), _mp4_time(m)
            movie.mvhd_timescale = struct.unpack(">I", buf[o:o + 4])[0]
            movie.mvhd_duration = (struct.unpack(">Q", buf[o + 4:o + 12]) if v == 1
                                   else struct.unpack(">I", buf[o + 4:o + 8]))[0]
        elif typ == "trak":
            movie.tracks.append(_parse_trak(buf, b, e, len(movie.tracks)))
        elif typ == "jumb":
            movie.has_jumbf = True


def _parse_trak(buf: bytes, start: int, end: int, idx: int) -> Track:
    t = Track(index=idx)

    def walk(s: int, e: int) -> None:
        for typ, b, be in _iter(buf, s, e):
            if typ == "tkhd":
                v, c, m = _times(buf, b)
                t.tkhd_creation, t.tkhd_modification = _mp4_time(c), _mp4_time(m)
                o = b + (20 if v == 1 else 12) + 8
                t.tkhd_duration = (struct.unpack(">Q", buf[o:o + 8]) if v == 1 else struct.unpack(">I", buf[o:o + 4]))[0]
            elif typ == "mdhd":
                v, c, m = _times(buf, b)
                t.mdhd_creation, t.mdhd_modification = _mp4_time(c), _mp4_time(m)
                o = b + (20 if v == 1 else 12)
                t.mdhd_timescale = struct.unpack(">I", buf[o:o + 4])[0]
                t.mdhd_duration = (struct.unpack(">Q", buf[o + 4:o + 12]) if v == 1
                                   else struct.unpack(">I", buf[o + 4:o + 8]))[0]
            elif typ == "hdlr":
                t.handler = buf[b + 8:b + 12].decode("latin-1")
            elif typ == "elst":
                v = buf[b]
                (n,) = struct.unpack(">I", buf[b + 4:b + 8])
                o = b + 8
                for _ in range(min(n, 10000)):
                    if v == 1:
                        sd, mt = struct.unpack(">Qq", buf[o:o + 16])
                        o += 16
                    else:
                        sd, mt = struct.unpack(">Ii", buf[o:o + 8])
                        o += 8
                    ri, rf = struct.unpack(">hH", buf[o:o + 4])
                    o += 4
                    t.edits.append(Edit(sd, mt, ri + rf / 65536))
            elif typ in CONTAINERS:
                walk(b, be)

    walk(start, end)
    return t


def _read_boxes(f: BinaryIO, size: int) -> Iterator[tuple[str, int, int]]:
    """Top-level boxes as (type, body_offset, box_end) without reading mdat."""
    pos = 0
    while pos + 8 <= size:
        f.seek(pos)
        hdr = f.read(16)
        if len(hdr) < 8:
            return
        bsize, typ = struct.unpack(">I4s", hdr[:8])
        h = 8
        if bsize == 1:
            bsize, h = struct.unpack(">Q", hdr[8:16])[0], 16
        elif bsize == 0:
            bsize = size - pos
        if bsize < h:
            return
        yield typ.decode("latin-1"), pos + h, pos + bsize
        pos += bsize


def parse(path: Path) -> Optional[Movie]:
    """Parse an MP4/MOV; returns None if the file does not look like ISO-BMFF."""
    size = path.stat().st_size
    movie = Movie()
    with open(path, "rb") as f:
        for typ, b, e in _read_boxes(f, size):
            if not typ.isprintable():
                return None
            movie.top_level.append(typ)
            if typ == "moov" and e - b <= MAX_BOX_READ:
                f.seek(b)
                _parse_moov(f.read(e - b), movie)
            elif typ in ("jumb", "uuid") and e - b <= MAX_BOX_READ:
                f.seek(b)
                body = f.read(e - b)
                if typ == "jumb" or b"jumb" in body[:64] or b"c2pa" in body[:256]:
                    movie.has_jumbf = True
    if "moov" not in movie.top_level:
        return None
    return movie
