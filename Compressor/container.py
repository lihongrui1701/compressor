from __future__ import annotations

import json
import struct
from dataclasses import dataclass

from . import FIXED_HEADER_SIZE, FORMAT_MAGIC, FORMAT_VERSION


_HEADER_STRUCT = struct.Struct("<4sBBHIIIQQ28s")


@dataclass(frozen=True)
class CpssHeader:
    version: int
    selector_mode: int
    pipeline_id: int
    metadata_size: int
    tail_size: int
    payload_size: int
    original_size: int


def build_header(
    *,
    selector_mode: int,
    pipeline_id: int,
    metadata_size: int,
    tail_size: int,
    payload_size: int,
    original_size: int,
) -> bytes:
    if metadata_size < 0 or tail_size < 0 or payload_size < 0 or original_size < 0:
        raise ValueError("header sizes must be non-negative")
    packed = _HEADER_STRUCT.pack(
        FORMAT_MAGIC,
        FORMAT_VERSION,
        int(selector_mode),
        FIXED_HEADER_SIZE,
        int(pipeline_id),
        int(metadata_size),
        int(tail_size),
        int(payload_size),
        int(original_size),
        b"\x00" * 28,
    )
    if len(packed) != FIXED_HEADER_SIZE:
        raise AssertionError(f"unexpected header size: {len(packed)}")
    return packed


def parse_header(blob: bytes) -> CpssHeader:
    if len(blob) < FIXED_HEADER_SIZE:
        raise ValueError("file is smaller than the fixed CPSS header")
    magic, version, selector_mode, header_size, pipeline_id, metadata_size, tail_size, payload_size, original_size, _ = (
        _HEADER_STRUCT.unpack(blob[:FIXED_HEADER_SIZE])
    )
    if magic != FORMAT_MAGIC:
        raise ValueError("invalid CPSS magic")
    if header_size != FIXED_HEADER_SIZE:
        raise ValueError(f"unsupported CPSS header size: {header_size}")
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported CPSS version: {version}")
    return CpssHeader(
        version=version,
        selector_mode=selector_mode,
        pipeline_id=pipeline_id,
        metadata_size=metadata_size,
        tail_size=tail_size,
        payload_size=payload_size,
        original_size=original_size,
    )


def encode_metadata(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_metadata(blob: bytes) -> dict:
    if not blob:
        return {}
    return json.loads(blob.decode("utf-8"))
