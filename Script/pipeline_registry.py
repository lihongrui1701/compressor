"""
功能：管线注册中心——filter/压缩器组合、`execute_pipeline`、dtype/shape 推断、preset 解析（含 curated、single_filter 等）。
改版：
  - 2026-03-28：补充模块头注释（功能 / 改版 / 改版特点）。
改版特点：
  - `single_filter_x_compressor` 排除无 Python 运行时的压缩器及 float 不适用的整型 filter；`ndcell` 支持 1D 切片提升为 2D；
  - 与 `BENCHMARK_CURATED_EXCLUDED_COMPRESSORS`、`SINGLE_FILTER_FLOAT_UNSUPPORTED_FILTERS` 等常量协同。
"""

from __future__ import annotations

import bz2
import copy
import gzip
import json
import lzma
import math
import os
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import imagecodecs
except ImportError:  # pragma: no cover
    imagecodecs = None

try:
    import zstandard as zstd_mod
except ImportError:  # pragma: no cover
    zstd_mod = None

try:
    import lz4.frame as lz4_frame
except ImportError:  # pragma: no cover
    lz4_frame = None

try:
    import brotli
except ImportError:  # pragma: no cover
    brotli = None

try:
    import snappy
except ImportError:  # pragma: no cover
    snappy = None

try:
    import liblzfse
except ImportError:  # pragma: no cover
    liblzfse = None

try:
    import fastlz
except ImportError:  # pragma: no cover
    fastlz = None

try:
    import numcodecs
except ImportError:  # pragma: no cover
    numcodecs = None

try:
    import blosc2
except ImportError:  # pragma: no cover
    blosc2 = None


ROOT_DIR = Path(__file__).resolve().parent.parent
COMPRESS_ROOT = ROOT_DIR / "Compress"
FILTER_ROOT = ROOT_DIR / "Filter"

RECOMMENDED_FILTERS = [
    "shuffle",
    "bitshuffle",
    "delta",
    "higher_order_delta",
    "byte_stream_split",
    "dictionary_categorize",
    "ndcell",
    "tdt",
]

RECOMMENDED_COMPRESSORS = [
    "zstd",
    "libdeflate",
    "zlib_ng",
    "lz4",
    "lz4hc",
    "brotli",
    "lzfse",
    "fse",
]

# 默认 benchmark 不纳入的压缩器：环境常缺绑定/需单独编译/本仓库未接稳定 Python 运行时（易 unavailable 或性价比低）
BENCHMARK_CURATED_EXCLUDED_COMPRESSORS: tuple[str, ...] = (
    "zlib_ng",
    "lzo",
    "ucl_nrv",
    "fastlz",
    "libbsc",
    "huff0",
    "fse",
)

# single_filter_x_compressor 预设下：与 float / 典型科学切片不兼容的 filter（整型专用；nbit 内部走 bit_packing）
SINGLE_FILTER_FLOAT_UNSUPPORTED_FILTERS: frozenset[str] = frozenset(
    {
        "zigzag_uleb128_varint",
        "bit_packing",
        "nbit",
    }
)

# 精选约 30 条「单 filter -> compressor」：面向科学浮点/网格数据，偏 zstd/libdeflate 与通用预处理（shuffle/delta/byte_stream_split 等）
BENCHMARK_CURATED_PIPELINES: list[str] = [
    "shuffle->zstd",
    "delta->zstd",
    "byte_stream_split->zstd",
    "higher_order_delta->zstd",
    "bitshuffle->zstd",
    "shuffle->libdeflate",
    "delta->libdeflate",
    "byte_stream_split->libdeflate",
    "shuffle->lz4hc",
    "delta->lz4hc",
    "shuffle->brotli",
    "delta->brotli",
    "shuffle->xz_lzma2",
    "delta->xz_lzma2",
    "shuffle->bzip2",
    "ndcell->zstd",
    "dictionary_categorize->zstd",
    "tdt->zstd",
    "xor_residual->zstd",
    "fcm_residual->zstd",
    "shuffle->zlib",
    "delta->zlib",
    "shuffle->lz4",
    "shuffle->lzfse",
    "delta->lzfse",
    "shuffle->snappy",
    "shuffle->lzf",
    "shuffle->szip_libaec",
    "shuffle->blosclz",
    "lorenzo_residual->zstd",
]

ALL_FILTERS = [
    "shuffle",
    "bitshuffle",
    "delta",
    "higher_order_delta",
    "zigzag_uleb128_varint",
    "bit_packing",
    "rle",
    "rle_bitpack_hybrid",
    "dictionary_categorize",
    "byte_stream_split",
    "nbit",
    "packbits",
    "delta_length_byte_array",
    "delta_byte_array",
    "ndcell",
    "tdt",
    "fcm_residual",
    "dfcm_residual",
    "xor_residual",
    "lorenzo_residual",
]

ALL_COMPRESSORS = [
    "zlib",
    "zlib_ng",
    "libdeflate",
    "zstd",
    "lz4",
    "lz4hc",
    "lzo",
    "snappy",
    "brotli",
    "bzip2",
    "xz_lzma2",
    "lzf",
    "szip_libaec",
    "ucl_nrv",
    "blosclz",
    "fastlz",
    "lzfse",
    "libbsc",
    "huff0",
    "fse",
]

ALIASES = {
    "zlib-ng": "zlib_ng",
    "xz/lzma2": "xz_lzma2",
    "szip/libaec": "szip_libaec",
    "ucl/nrv": "ucl_nrv",
    "byte stream split": "byte_stream_split",
    "higher-order delta": "higher_order_delta",
    "zigzag+uleb128/varint": "zigzag_uleb128_varint",
    "zigzag + uleb128 / varint": "zigzag_uleb128_varint",
    "bit-packing": "bit_packing",
    "rle / bit-pack hybrid": "rle_bitpack_hybrid",
    "dictionary / categorize": "dictionary_categorize",
    "delta length byte array": "delta_length_byte_array",
    "delta byte array": "delta_byte_array",
    "fcm residual": "fcm_residual",
    "dfcm residual": "dfcm_residual",
    "xor residual coding": "xor_residual",
    "lorenzo residual": "lorenzo_residual",
}

if np is not None:
    DTYPE_MAP = {
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "float32": np.float32,
        "float64": np.float64,
    }
    UNSIGNED_BY_SIZE = {
        1: np.uint8,
        2: np.uint16,
        4: np.uint32,
        8: np.uint64,
    }
else:  # pragma: no cover
    DTYPE_MAP = {}
    UNSIGNED_BY_SIZE = {}

KNOWN_SHAPES = {
    "cesm-atm": [(1800, 3600), (26, 1800, 3600)],
    "hurricane-isabel": [(100, 500, 500)],
    "hurricane": [(100, 500, 500)],
    "miranda": [(256, 384, 384), (3072, 3072, 3072)],
}

UNSTRUCTURED_DATASETS = {"xgc"}


class ComponentUnavailable(RuntimeError):
    pass


@dataclass
class PipelineContext:
    dtype_name: str | None = None
    shape: tuple[int, ...] | None = None
    typesize: int | None = None
    endian: str = "little"
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "PipelineContext":
        return PipelineContext(
            dtype_name=self.dtype_name,
            shape=tuple(self.shape) if self.shape is not None else None,
            typesize=self.typesize,
            endian=self.endian,
            metadata=copy.deepcopy(self.metadata),
        )


def canonical_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("+", "_")
    s = s.replace("/", "_")
    s = s.replace("-", "_")
    s = s.replace(" ", "_")
    s = s.replace("__", "_")
    return ALIASES.get(s, s)


def norm_dataset_name(name: str) -> str:
    return (name or "").lower().replace("_", "-").replace(" ", "-")


def guess_dtype(name: str | None) -> str | None:
    s = (name or "").lower()
    for dtype_name in ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"):
        if dtype_name in s:
            return dtype_name
    if "double" in s or "d64" in s or "f64" in s or "float64" in s:
        return "float64"
    if "f32" in s or "float32" in s or "single" in s:
        return "float32"
    return None


def element_size(dtype_name: str | None) -> int | None:
    if np is not None and dtype_name in DTYPE_MAP:
        return int(np.dtype(DTYPE_MAP[dtype_name]).itemsize)
    return None


def shape_product(shape: tuple[int, ...] | list[int] | None) -> int | None:
    if shape is None:
        return None
    value = 1
    for item in shape:
        value *= int(item)
    return value


def guess_shape(file_path: Path, dtype_name: str | None, dataset_name: str, used_size: int) -> tuple[tuple[int, ...] | None, str]:
    item_size = element_size(dtype_name)
    if item_size is None or used_size == 0:
        return None, "unknown"

    count = used_size // item_size
    ds_key = norm_dataset_name(dataset_name)
    if ds_key in UNSTRUCTURED_DATASETS:
        return None, "unstructured_dataset"

    candidates: list[tuple[tuple[int, ...], str]] = []
    for shape in KNOWN_SHAPES.get(ds_key, []):
        if shape_product(shape) == count:
            candidates.append((tuple(shape), "dataset_table"))

    import re

    stem = file_path.name.rsplit(".", 1)[0]
    numbers = [int(x) for x in re.findall(r"\d+", stem)]
    for ndim in (4, 3, 2, 1):
        if len(numbers) >= ndim:
            shape = tuple(numbers[-ndim:])
            while len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
            if shape_product(shape) == count:
                candidates.append((shape, "filename_numbers"))

    if candidates:
        return candidates[0]
    return (count,), "1d_fallback"


def prepare_bytes(file_path: Path, dtype_name: str | None) -> tuple[bytes, int, int]:
    raw = file_path.read_bytes()
    raw_size = len(raw)
    item_size = element_size(dtype_name)
    if item_size is None or item_size <= 0:
        return raw, raw_size, raw_size
    used_size = raw_size - (raw_size % item_size)
    return raw[:used_size], raw_size, used_size


def ok_result(
    pipeline_name: str,
    filters: list[str],
    compressor: str,
    input_size: int,
    transformed_size: int,
    compressed_size: int,
    seconds: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline": pipeline_name,
        "filters": filters,
        "compressor": compressor,
        "input_size_bytes": input_size,
        "transformed_size_bytes": transformed_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": round(input_size / compressed_size, 6) if compressed_size else None,
        "compressed_rate": round(compressed_size / input_size, 6) if input_size else None,
        "time_sec": round(seconds, 6),
        "speed_mb_s": round((input_size / 1024 / 1024) / seconds, 6) if seconds > 0 else None,
        "metadata": metadata or {},
    }


def skip_result(
    pipeline_name: str,
    filters: list[str],
    compressor: str,
    input_size: int,
    reason: str,
    *,
    time_sec: float | None = None,
) -> dict[str, Any]:
    """失败/跳过：time_sec 为从进入管线到抛出异常为止的 wall 时间（可选）。"""
    ts = round(time_sec, 6) if time_sec is not None else None
    return {
        "ok": False,
        "pipeline": pipeline_name,
        "filters": filters,
        "compressor": compressor,
        "input_size_bytes": input_size,
        "transformed_size_bytes": None,
        "compressed_size_bytes": None,
        "compression_ratio": None,
        "compressed_rate": None,
        "time_sec": ts,
        "speed_mb_s": round((input_size / 1024 / 1024) / time_sec, 6) if time_sec and time_sec > 0 and input_size else None,
        "metadata": {},
        "reason": reason,
    }


def require_numpy() -> None:
    if np is None:
        raise ComponentUnavailable("numpy is required for typed filters")


def normalize_context(context: PipelineContext | None, raw: bytes) -> PipelineContext:
    ctx = context.clone() if context is not None else PipelineContext()
    if ctx.dtype_name and ctx.typesize is None:
        ctx.typesize = element_size(ctx.dtype_name)
    if ctx.typesize is None and ctx.dtype_name is None:
        ctx.typesize = 1
    if ctx.shape is not None:
        ctx.shape = tuple(int(x) for x in ctx.shape)
    if ctx.endian not in ("little", "big"):
        ctx.endian = "little"
    if ctx.typesize is None and len(raw) > 0:
        ctx.typesize = 1
    return ctx


def as_array(raw: bytes, context: PipelineContext) -> Any:
    require_numpy()
    dtype_name = context.dtype_name
    item_size = context.typesize or element_size(dtype_name)
    if dtype_name in DTYPE_MAP:
        np_dtype = DTYPE_MAP[dtype_name]
    elif item_size in UNSIGNED_BY_SIZE:
        np_dtype = UNSIGNED_BY_SIZE[item_size]
    else:
        raise ComponentUnavailable("typed transform requires a known dtype or typesize")

    arr = np.frombuffer(raw, dtype=np_dtype).copy()
    if context.shape and shape_product(context.shape) == int(arr.size):
        arr = arr.reshape(context.shape)
    return arr


def to_unsigned_view(arr: Any) -> Any:
    require_numpy()
    item_size = int(arr.dtype.itemsize)
    if item_size not in UNSIGNED_BY_SIZE:
        raise ComponentUnavailable(f"unsupported item size: {item_size}")
    return np.ascontiguousarray(arr).view(UNSIGNED_BY_SIZE[item_size])


def typed_or_raise(raw: bytes, context: PipelineContext, min_typesize: int = 2) -> Any:
    arr = as_array(raw, context)
    if int(arr.dtype.itemsize) < min_typesize:
        raise ComponentUnavailable(f"typesize must be >= {min_typesize}")
    return arr


def encode_uleb128(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError("ULEB128 only supports non-negative integers")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def zigzag_encode(value: int) -> int:
    value = int(value)
    return (value << 1) ^ (value >> 63)


def common_prefix_len(a: bytes, b: bytes) -> int:
    upto = min(len(a), len(b))
    idx = 0
    while idx < upto and a[idx] == b[idx]:
        idx += 1
    return idx


def pack_nonnegative(values: list[int], bit_width: int) -> bytes:
    if bit_width <= 0:
        return b""
    mask = (1 << bit_width) - 1
    acc = 0
    acc_bits = 0
    out = bytearray()
    for value in values:
        acc |= (int(value) & mask) << acc_bits
        acc_bits += bit_width
        while acc_bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            acc_bits -= 8
    if acc_bits:
        out.append(acc & 0xFF)
    return bytes(out)


def infer_records(raw: bytes, context: PipelineContext) -> tuple[list[bytes], dict[str, Any]]:
    if b"\n" in raw and raw.count(b"\n") >= 2:
        return raw.splitlines(keepends=True), {"mode": "newline"}
    record_size = context.typesize or 16
    if record_size <= 0:
        record_size = 16
    records = [raw[i : i + record_size] for i in range(0, len(raw), record_size)]
    return records, {"mode": "fixed", "record_size": int(record_size)}


def filter_shuffle(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if imagecodecs is not None:
        encoded = imagecodecs.byteshuffle_encode(arr)
        return np.ascontiguousarray(encoded).tobytes(), {"engine": "imagecodecs.byteshuffle"}
    item_size = int(arr.dtype.itemsize)
    matrix = arr.view(np.uint8).reshape(-1, item_size)
    return matrix.T.copy().reshape(-1).tobytes(), {"engine": "numpy", "typesize": item_size}


def filter_bitshuffle(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if imagecodecs is None:
        raise ComponentUnavailable("imagecodecs.bitshuffle is not available")
    encoded = imagecodecs.bitshuffle_encode(arr)
    return np.ascontiguousarray(encoded).tobytes(), {"engine": "imagecodecs.bitshuffle"}


def _manual_delta(arr: Any) -> Any:
    flat = np.ascontiguousarray(arr).reshape(-1)
    out = flat.copy()
    if flat.size > 1:
        out[1:] = flat[1:] - flat[:-1]
    return out.reshape(arr.shape)


def filter_delta(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context, min_typesize=1)
    if imagecodecs is not None:
        encoded = imagecodecs.delta_encode(arr)
        return np.ascontiguousarray(encoded).tobytes(), {"engine": "imagecodecs.delta"}
    encoded = _manual_delta(arr)
    return np.ascontiguousarray(encoded).tobytes(), {"engine": "numpy"}


def filter_higher_order_delta(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    order = max(2, int(params.get("order", 2)))
    arr = typed_or_raise(raw, context, min_typesize=1)
    work = np.ascontiguousarray(arr)
    for _ in range(order):
        if imagecodecs is not None:
            work = imagecodecs.delta_encode(work)
        else:
            work = _manual_delta(work)
    return np.ascontiguousarray(work).tobytes(), {
        "engine": "imagecodecs.delta" if imagecodecs is not None else "numpy",
        "order": order,
    }


def filter_zigzag_varint(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if arr.dtype.kind not in ("i", "u"):
        raise ComponentUnavailable("zigzag/varint requires integer data")
    flat = np.ascontiguousarray(arr).reshape(-1)
    signed = arr.dtype.kind == "i"
    out = bytearray()
    for value in flat.tolist():
        if signed:
            out.extend(encode_uleb128(zigzag_encode(int(value))))
        else:
            out.extend(encode_uleb128(int(value)))
    return bytes(out), {"count": int(flat.size), "signed": signed, "engine": "python"}


def filter_bit_packing(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if arr.dtype.kind not in ("i", "u"):
        raise ComponentUnavailable("bit-packing requires integer data")
    flat = np.ascontiguousarray(arr).reshape(-1)
    signed = arr.dtype.kind == "i"
    values = [zigzag_encode(int(v)) if signed else int(v) for v in flat.tolist()]
    max_value = max(values) if values else 0
    bit_width = max(1, max_value.bit_length())
    packed = pack_nonnegative(values, bit_width)
    header = bytearray()
    header.extend(encode_uleb128(len(values)))
    header.extend(encode_uleb128(bit_width))
    return bytes(header) + packed, {"count": len(values), "bit_width": bit_width, "signed": signed, "engine": "python"}


def filter_rle(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if len(raw) == 0:
        return b"", {"run_count": 0, "engine": "numpy" if np is not None else "python"}
    if np is not None:
        arr = np.frombuffer(raw, dtype=np.uint8)
        cuts = np.flatnonzero(arr[1:] != arr[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [len(arr)]))
        values = arr[bounds[:-1]]
        lengths = np.diff(bounds)
        out = bytearray()
        for run_len, value in zip(lengths.tolist(), values.tolist()):
            out.extend(encode_uleb128(run_len))
            out.append(int(value))
        return bytes(out), {"run_count": int(len(lengths)), "engine": "numpy"}
    out = bytearray()
    run_count = 0
    idx = 0
    while idx < len(raw):
        value = raw[idx]
        j = idx + 1
        while j < len(raw) and raw[j] == value:
            j += 1
        out.extend(encode_uleb128(j - idx))
        out.append(value)
        run_count += 1
        idx = j
    return bytes(out), {"run_count": run_count, "engine": "python"}


def filter_rle_bitpack_hybrid(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context, min_typesize=1)
    if arr.dtype.kind not in ("i", "u", "f"):
        raise ComponentUnavailable("rle/bit-pack hybrid needs typed numeric data")
    if arr.dtype.kind == "f":
        values = to_unsigned_view(arr).reshape(-1).tolist()
        signed = False
    else:
        flat = np.ascontiguousarray(arr).reshape(-1)
        signed = flat.dtype.kind == "i"
        values = [zigzag_encode(int(v)) if signed else int(v) for v in flat.tolist()]
    max_value = max(values) if values else 0
    bit_width = max(1, max_value.bit_length())
    out = bytearray()
    i = 0
    literal_blocks = 0
    run_blocks = 0
    while i < len(values):
        run_len = 1
        while i + run_len < len(values) and values[i + run_len] == values[i]:
            run_len += 1
        if run_len >= 4:
            out.append(1)
            out.extend(encode_uleb128(run_len))
            out.extend(encode_uleb128(values[i]))
            run_blocks += 1
            i += run_len
            continue
        literals = [values[i]]
        i += 1
        while i < len(values):
            next_run = 1
            while i + next_run < len(values) and values[i + next_run] == values[i]:
                next_run += 1
            if next_run >= 4 or len(literals) >= 128:
                break
            literals.append(values[i])
            i += 1
        out.append(0)
        out.extend(encode_uleb128(len(literals)))
        out.extend(pack_nonnegative(literals, bit_width))
        literal_blocks += 1
    return bytes(out), {
        "bit_width": bit_width,
        "literal_blocks": literal_blocks,
        "run_blocks": run_blocks,
        "engine": "python",
    }


def filter_dictionary(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    token_size = int(params.get("token_size") or (context.typesize or 1))
    if token_size <= 0 or len(raw) % token_size != 0:
        token_size = 1
    tokens = [raw[i : i + token_size] for i in range(0, len(raw), token_size)]
    dictionary: list[bytes] = []
    index_of: dict[bytes, int] = {}
    indices: list[int] = []
    for token in tokens:
        idx = index_of.get(token)
        if idx is None:
            idx = len(dictionary)
            dictionary.append(token)
            index_of[token] = idx
        indices.append(idx)
    bit_width = max(1, max(indices).bit_length()) if indices else 1
    packed = pack_nonnegative(indices, bit_width)
    out = bytearray()
    out.extend(encode_uleb128(token_size))
    out.extend(encode_uleb128(len(tokens)))
    out.extend(encode_uleb128(len(dictionary)))
    out.extend(encode_uleb128(bit_width))
    out.extend(b"".join(dictionary))
    out.extend(packed)
    return bytes(out), {
        "token_size": token_size,
        "dictionary_size": len(dictionary),
        "bit_width": bit_width,
        "engine": "python",
    }


def filter_byte_stream_split(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    item_size = int(arr.dtype.itemsize)
    matrix = arr.view(np.uint8).reshape(-1, item_size)
    return matrix.T.copy().reshape(-1).tobytes(), {"engine": "numpy", "typesize": item_size}


def filter_nbit(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    encoded, meta = filter_bit_packing(raw, context, params)
    meta = dict(meta)
    meta["nbit"] = meta.get("bit_width")
    return encoded, meta


def filter_packbits(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if imagecodecs is None:
        raise ComponentUnavailable("imagecodecs.packbits is not available")
    return imagecodecs.packbits_encode(raw), {"engine": "imagecodecs.packbits"}


def filter_delta_length_byte_array(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    records, record_meta = infer_records(raw, context)
    lengths = [len(rec) for rec in records]
    deltas: list[int] = []
    prev = 0
    for length in lengths:
        deltas.append(int(length) - prev)
        prev = int(length)
    delta_stream = bytearray()
    for delta in deltas:
        delta_stream.extend(encode_uleb128(zigzag_encode(delta)))
    payload = b"".join(records)
    out = bytearray()
    out.extend(encode_uleb128(len(records)))
    out.extend(encode_uleb128(len(delta_stream)))
    out.extend(delta_stream)
    out.extend(payload)
    meta = dict(record_meta)
    meta.update({"record_count": len(records), "engine": "python"})
    return bytes(out), meta


def filter_delta_byte_array(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    records, record_meta = infer_records(raw, context)
    out = bytearray()
    out.extend(encode_uleb128(len(records)))
    previous = b""
    for record in records:
        prefix_len = common_prefix_len(previous, record)
        suffix = record[prefix_len:]
        out.extend(encode_uleb128(prefix_len))
        out.extend(encode_uleb128(len(suffix)))
        out.extend(suffix)
        previous = record
    meta = dict(record_meta)
    meta.update({"record_count": len(records), "engine": "python"})
    return bytes(out), meta


def filter_ndcell(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if not context.shape or len(context.shape) < 1:
        raise ComponentUnavailable("ndcell requires shape context")
    promote_1d = False
    shape: tuple[int, ...]
    if len(context.shape) == 1:
        n = int(context.shape[0])
        arr = np.ascontiguousarray(arr).reshape(n, 1)
        shape = (n, 1)
        promote_1d = True
    elif len(context.shape) < 2:
        raise ComponentUnavailable("ndcell requires a known multi-dimensional shape")
    else:
        shape = tuple(int(x) for x in context.shape)
    cell_shape = tuple(min(int(params.get(f"c{i}", 4)), dim) for i, dim in enumerate(shape))
    pieces: list[bytes] = []
    import itertools

    ranges = [range(0, dim, cell) for dim, cell in zip(shape, cell_shape)]
    for origin in itertools.product(*ranges):
        block = tuple(slice(start, min(start + cell, dim)) for start, cell, dim in zip(origin, cell_shape, shape))
        pieces.append(np.ascontiguousarray(arr[block]).tobytes())
    meta: dict[str, Any] = {"cell_shape": list(cell_shape), "engine": "numpy"}
    if promote_1d:
        meta["promoted_1d_to_2d"] = True
    return b"".join(pieces), meta


def filter_tdt(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    item_size = int(arr.dtype.itemsize)
    matrix = arr.view(np.uint8).reshape(-1, item_size)
    if context.endian == "little":
        order = list(range(item_size - 1, -1, -1))
    else:
        order = list(range(item_size))
    transformed = matrix[:, order].T.copy().reshape(-1)
    return transformed.tobytes(), {"byte_order": order, "engine": "numpy"}


def _fcm_like_residual(raw: bytes, context: PipelineContext, params: dict[str, Any], differential: bool) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    unsigned = to_unsigned_view(arr).reshape(-1)
    bits = int(params.get("hash_bits", 16))
    table_size = 1 << bits
    mask = table_size - 1
    table = [0] * table_size
    residual = np.empty_like(unsigned)
    hash_value = 0
    last_value = 0
    item_bits = 8 * unsigned.dtype.itemsize
    item_mask = (1 << item_bits) - 1
    for idx, value in enumerate(unsigned.tolist()):
        predicted = table[hash_value]
        if differential:
            predicted = (last_value + predicted) & item_mask
        residual[idx] = value ^ predicted
        if differential:
            stride = (value - last_value) & item_mask
            table[hash_value] = stride
            hash_value = ((hash_value << 2) ^ (stride >> max(0, item_bits - 24))) & mask
            last_value = value
        else:
            table[hash_value] = value
            hash_value = ((hash_value << 5) ^ (value >> max(0, item_bits - 16))) & mask
    return residual.tobytes(), {"hash_bits": bits, "engine": "python"}


def filter_fcm(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    out, meta = _fcm_like_residual(raw, context, params, differential=False)
    meta["predictor"] = "fcm"
    return out, meta


def filter_dfcm(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    out, meta = _fcm_like_residual(raw, context, params, differential=True)
    meta["predictor"] = "dfcm"
    return out, meta


def filter_xor(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if imagecodecs is not None:
        encoded = imagecodecs.xor_encode(arr)
        return np.ascontiguousarray(encoded).tobytes(), {"engine": "imagecodecs.xor"}
    unsigned = to_unsigned_view(arr).reshape(-1)
    out = unsigned.copy()
    if out.size > 1:
        out[1:] = unsigned[1:] ^ unsigned[:-1]
    return out.tobytes(), {"engine": "numpy"}


def filter_lorenzo(raw: bytes, context: PipelineContext, params: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    arr = typed_or_raise(raw, context)
    if not context.shape:
        raise ComponentUnavailable("lorenzo residual requires shape information")
    unsigned = to_unsigned_view(arr).reshape(context.shape)
    ndim = unsigned.ndim
    if ndim == 1:
        residual = unsigned.copy()
        residual[1:] = unsigned[1:] - unsigned[:-1]
    elif ndim == 2:
        residual = unsigned.copy()
        residual[1:, :] = residual[1:, :] - unsigned[:-1, :]
        residual[:, 1:] = residual[:, 1:] - unsigned[:, :-1]
        residual[1:, 1:] = residual[1:, 1:] + unsigned[:-1, :-1]
    elif ndim == 3:
        residual = unsigned.copy()
        residual[1:, :, :] = residual[1:, :, :] - unsigned[:-1, :, :]
        residual[:, 1:, :] = residual[:, 1:, :] - unsigned[:, :-1, :]
        residual[:, :, 1:] = residual[:, :, 1:] - unsigned[:, :, :-1]
        residual[1:, 1:, :] = residual[1:, 1:, :] + unsigned[:-1, :-1, :]
        residual[1:, :, 1:] = residual[1:, :, 1:] + unsigned[:-1, :, :-1]
        residual[:, 1:, 1:] = residual[:, 1:, 1:] + unsigned[:, :-1, :-1]
        residual[1:, 1:, 1:] = residual[1:, 1:, 1:] - unsigned[:-1, :-1, :-1]
    else:
        raise ComponentUnavailable("lorenzo residual is implemented for 1D/2D/3D only")
    return np.ascontiguousarray(residual).tobytes(), {"engine": "numpy", "ndim": ndim}


FILTER_FUNCS: dict[str, Callable[[bytes, PipelineContext, dict[str, Any]], tuple[bytes, dict[str, Any]]]] = {
    "shuffle": filter_shuffle,
    "bitshuffle": filter_bitshuffle,
    "delta": filter_delta,
    "higher_order_delta": filter_higher_order_delta,
    "zigzag_uleb128_varint": filter_zigzag_varint,
    "bit_packing": filter_bit_packing,
    "rle": filter_rle,
    "rle_bitpack_hybrid": filter_rle_bitpack_hybrid,
    "dictionary_categorize": filter_dictionary,
    "byte_stream_split": filter_byte_stream_split,
    "nbit": filter_nbit,
    "packbits": filter_packbits,
    "delta_length_byte_array": filter_delta_length_byte_array,
    "delta_byte_array": filter_delta_byte_array,
    "ndcell": filter_ndcell,
    "tdt": filter_tdt,
    "fcm_residual": filter_fcm,
    "dfcm_residual": filter_dfcm,
    "xor_residual": filter_xor,
    "lorenzo_residual": filter_lorenzo,
}


def _codec_call(name: str, *args: Any, **kwargs: Any) -> Any:
    if imagecodecs is None:
        raise ComponentUnavailable(f"{name} requires imagecodecs")
    func = getattr(imagecodecs, name)
    return func(*args, **kwargs)


def compress_zlib(data: bytes, params: dict[str, Any]) -> bytes:
    return zlib.compress(data, level=int(params.get("level", 6)))


def compress_zlib_ng(data: bytes, params: dict[str, Any]) -> bytes:
    try:
        return _codec_call("zlibng_encode", data)
    except Exception as exc:  # pragma: no cover
        raise ComponentUnavailable(f"zlib_ng runtime unavailable: {exc}") from exc


def compress_libdeflate(data: bytes, params: dict[str, Any]) -> bytes:
    try:
        return _codec_call("deflate_encode", data)
    except Exception as exc:  # pragma: no cover
        raise ComponentUnavailable(f"libdeflate-style runtime unavailable: {exc}") from exc


def compress_zstd(data: bytes, params: dict[str, Any]) -> bytes:
    level = int(params.get("level", 3))
    if zstd_mod is not None:
        return zstd_mod.ZstdCompressor(level=level).compress(data)
    return _codec_call("zstd_encode", data)


def compress_lz4(data: bytes, params: dict[str, Any]) -> bytes:
    if lz4_frame is not None:
        return lz4_frame.compress(data, compression_level=int(params.get("level", 0)))
    return _codec_call("lz4_encode", data)


def compress_lz4hc(data: bytes, params: dict[str, Any]) -> bytes:
    if lz4_frame is None:
        raise ComponentUnavailable("lz4.frame is not available for lz4hc")
    return lz4_frame.compress(data, compression_level=int(params.get("level", 16)))


def compress_lzo(data: bytes, params: dict[str, Any]) -> bytes:
    if imagecodecs is not None and hasattr(imagecodecs, "lzo_encode"):
        try:
            return _codec_call("lzo_encode", data)
        except Exception:
            pass
    raise ComponentUnavailable("lzo runtime is not available in the current environment")


def compress_snappy(data: bytes, params: dict[str, Any]) -> bytes:
    if imagecodecs is not None:
        try:
            return _codec_call("snappy_encode", data)
        except Exception:
            pass
    if snappy is not None:
        return snappy.compress(data)
    raise ComponentUnavailable("snappy runtime is not available")


def compress_brotli(data: bytes, params: dict[str, Any]) -> bytes:
    level = int(params.get("level", 5))
    if brotli is not None:
        return brotli.compress(data, quality=level)
    return _codec_call("brotli_encode", data)


def compress_bzip2(data: bytes, params: dict[str, Any]) -> bytes:
    return bz2.compress(data, compresslevel=int(params.get("level", 9)))


def compress_xz(data: bytes, params: dict[str, Any]) -> bytes:
    return lzma.compress(data, preset=int(params.get("preset", 6)), format=lzma.FORMAT_XZ)


def compress_lzf(data: bytes, params: dict[str, Any]) -> bytes:
    return _codec_call("lzf_encode", data)


def compress_szip_libaec(data: bytes, params: dict[str, Any]) -> bytes:
    return _codec_call("aec_encode", data)


def compress_ucl_nrv(data: bytes, params: dict[str, Any]) -> bytes:
    raise ComponentUnavailable("ucl/nrv runtime is not available in Python yet")


def compress_blosclz(data: bytes, params: dict[str, Any]) -> bytes:
    if imagecodecs is not None:
        return _codec_call("blosc_encode", data, compressor="blosclz", shuffle="noshuffle")
    if blosc2 is not None:
        return blosc2.compress(data, codec=blosc2.Codec.BLOSCLZ)
    raise ComponentUnavailable("blosclz runtime is not available")


def compress_fastlz(data: bytes, params: dict[str, Any]) -> bytes:
    if fastlz is None:
        raise ComponentUnavailable("fastlz package is not installed")
    return fastlz.compress(data)


def compress_lzfse(data: bytes, params: dict[str, Any]) -> bytes:
    if liblzfse is not None:
        return liblzfse.compress(data)
    try:
        return _codec_call("lzfse_encode", data)
    except Exception as exc:  # pragma: no cover
        raise ComponentUnavailable(f"lzfse runtime unavailable: {exc}") from exc


def compress_libbsc(data: bytes, params: dict[str, Any]) -> bytes:
    raise ComponentUnavailable("libbsc runtime is not available in Python yet")


def compress_huff0(data: bytes, params: dict[str, Any]) -> bytes:
    raise ComponentUnavailable("Huff0 runtime is not available in Python yet")


def compress_fse(data: bytes, params: dict[str, Any]) -> bytes:
    raise ComponentUnavailable("FSE runtime is not available in Python yet")


COMPRESSOR_FUNCS: dict[str, Callable[[bytes, dict[str, Any]], bytes]] = {
    "zlib": compress_zlib,
    "zlib_ng": compress_zlib_ng,
    "libdeflate": compress_libdeflate,
    "zstd": compress_zstd,
    "lz4": compress_lz4,
    "lz4hc": compress_lz4hc,
    "lzo": compress_lzo,
    "snappy": compress_snappy,
    "brotli": compress_brotli,
    "bzip2": compress_bzip2,
    "xz_lzma2": compress_xz,
    "lzf": compress_lzf,
    "szip_libaec": compress_szip_libaec,
    "ucl_nrv": compress_ucl_nrv,
    "blosclz": compress_blosclz,
    "fastlz": compress_fastlz,
    "lzfse": compress_lzfse,
    "libbsc": compress_libbsc,
    "huff0": compress_huff0,
    "fse": compress_fse,
}


FILTER_REGISTRY = {name: {"name": name, "source_dir": FILTER_ROOT / name} for name in ALL_FILTERS}
COMPRESSOR_REGISTRY = {name: {"name": name, "source_dir": COMPRESS_ROOT / name} for name in ALL_COMPRESSORS}


def make_pipeline_name(filters: list[str], compressor: str) -> str:
    return "->".join(filters + [compressor]) if filters else compressor


def parse_pipeline_name(name: str) -> tuple[list[str], str]:
    parts = [canonical_name(part) for part in name.split("->") if part.strip()]
    if not parts:
        raise ValueError("empty pipeline")
    compressor = parts[-1]
    filters = parts[:-1]
    if compressor not in COMPRESSOR_FUNCS:
        raise ValueError(f"unknown compressor: {compressor}")
    for item in filters:
        if item not in FILTER_FUNCS:
            raise ValueError(f"unknown filter: {item}")
    return filters, compressor


def execute_pipeline(
    raw: bytes,
    pipeline_name: str,
    context: PipelineContext | None = None,
    filter_params: dict[str, dict[str, Any]] | None = None,
    compressor_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filter_names, compressor_name = parse_pipeline_name(pipeline_name)
    filter_params = filter_params or {}
    compressor_params = compressor_params or {}
    ctx = normalize_context(context, raw)
    transformed = raw
    trace: list[dict[str, Any]] = []
    start = time.perf_counter()
    try:
        for filter_name in filter_names:
            transformed, meta = FILTER_FUNCS[filter_name](transformed, ctx, filter_params.get(filter_name, {}))
            trace.append({"stage": filter_name, "size_bytes": len(transformed), "metadata": meta})
        compressed = COMPRESSOR_FUNCS[compressor_name](transformed, compressor_params)
        seconds = time.perf_counter() - start
        return ok_result(
            pipeline_name=pipeline_name,
            filters=filter_names,
            compressor=compressor_name,
            input_size=len(raw),
            transformed_size=len(transformed),
            compressed_size=len(compressed),
            seconds=seconds,
            metadata={"stages": trace},
        )
    except Exception as exc:
        fail_sec = time.perf_counter() - start
        return skip_result(
            pipeline_name,
            filter_names,
            compressor_name,
            len(raw),
            str(exc),
            time_sec=fail_sec,
        )


def resolve_pipeline_names(
    preset: str = "all_compressors",
    explicit: list[str] | None = None,
    filters: list[str] | None = None,
    compressors: list[str] | None = None,
) -> list[str]:
    if explicit:
        output = []
        for name in explicit:
            flts, comp = parse_pipeline_name(name)
            output.append(make_pipeline_name(flts, comp))
        return output

    filters = [canonical_name(x) for x in (filters or ALL_FILTERS)]
    compressors = [canonical_name(x) for x in (compressors or ALL_COMPRESSORS)]
    preset = canonical_name(preset)

    if preset in ("all_compressors", "compressors", "compressors_only"):
        return compressors
    if preset in ("recommended_8x8", "recommended"):
        return [make_pipeline_name([flt], comp) for flt in RECOMMENDED_FILTERS for comp in RECOMMENDED_COMPRESSORS]
    if preset in ("single_filter_x_compressor", "single_filter", "all_single_filter_combos"):
        compressors = [c for c in compressors if c not in BENCHMARK_CURATED_EXCLUDED_COMPRESSORS]
        filters_use = [f for f in filters if f not in SINGLE_FILTER_FLOAT_UNSUPPORTED_FILTERS]
        return [make_pipeline_name([flt], comp) for flt in filters_use for comp in compressors]
    if preset in ("compressors_plus_single_filter_x_compressor", "all_registered"):
        names = list(compressors)
        names.extend(make_pipeline_name([flt], comp) for flt in filters for comp in compressors)
        return names
    if preset in ("benchmark_curated", "curated_30", "curated_benchmark"):
        return [make_pipeline_name(*parse_pipeline_name(name)) for name in BENCHMARK_CURATED_PIPELINES]
    raise ValueError(f"unknown pipeline preset: {preset}")


def check_component_availability(sample_context: PipelineContext | None = None) -> dict[str, dict[str, Any]]:
    filters: dict[str, Any] = {}
    compressors: dict[str, Any] = {}

    float_context = sample_context or PipelineContext(dtype_name="float32", shape=(256,), typesize=4)
    int_context = PipelineContext(dtype_name="int32", shape=(256,), typesize=4)
    nd_context = PipelineContext(dtype_name="float32", shape=(16, 16), typesize=4)

    if np is not None:
        float_raw = np.linspace(0, 1, 256, dtype=np.float32).tobytes()
        int_raw = np.arange(256, dtype=np.int32).tobytes()
        nd_raw = np.arange(256, dtype=np.float32).reshape(16, 16).tobytes()
    else:  # pragma: no cover
        float_raw = b"\x00" * 1024
        int_raw = float_raw
        nd_raw = float_raw

    integer_filters = {"zigzag_uleb128_varint", "bit_packing", "nbit"}
    shape_filters = {"ndcell"}

    for name in ALL_FILTERS:
        try:
            if name in integer_filters:
                raw = int_raw
                ctx = int_context
            elif name in shape_filters:
                raw = nd_raw
                ctx = nd_context
            else:
                raw = float_raw
                ctx = float_context
            out, meta = FILTER_FUNCS[name](raw, ctx, {})
            filters[name] = {"available": True, "output_size_bytes": len(out), "metadata": meta}
        except Exception as exc:
            filters[name] = {"available": False, "reason": str(exc)}

    for name in ALL_COMPRESSORS:
        try:
            out = COMPRESSOR_FUNCS[name](float_raw, {})
            compressors[name] = {"available": True, "output_size_bytes": len(out)}
        except Exception as exc:
            compressors[name] = {"available": False, "reason": str(exc)}

    return {"filters": filters, "compressors": compressors}


def choose_best_pipeline(results: dict[str, dict[str, Any]]) -> tuple[str | None, float | None]:
    best_name = None
    best_ratio = -1.0
    for name, info in results.items():
        ratio = info.get("compression_ratio")
        if info.get("ok") and ratio is not None and float(ratio) > best_ratio:
            best_name = name
            best_ratio = float(ratio)
    return best_name, (round(best_ratio, 6) if best_name is not None else None)


def flatten_result(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    slug = canonical_name(prefix)
    ok = result.get("ok")
    out = {
        f"{slug}_compressed_size_bytes": result.get("compressed_size_bytes") if ok else None,
        f"{slug}_ratio": result.get("compression_ratio") if ok else None,
        f"{slug}_rate": result.get("compressed_rate") if ok else None,
        # 失败时也可能有 time_sec / speed_mb_s（execute_pipeline 在异常前耗时）
        f"{slug}_time_sec": result.get("time_sec"),
        f"{slug}_speed_mb_s": result.get("speed_mb_s"),
    }
    if not ok:
        out[f"{slug}_reason"] = result.get("reason")
    return out


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def pipeline_source_summary() -> dict[str, Any]:
    return {
        "compressors": {name: str(COMPRESSOR_REGISTRY[name]["source_dir"]) for name in ALL_COMPRESSORS},
        "filters": {name: str(FILTER_REGISTRY[name]["source_dir"]) for name in ALL_FILTERS},
    }


def pipeline_config_from_env(default_preset: str = "all_compressors") -> dict[str, Any]:
    explicit = os.environ.get("PIPELINES")
    if explicit:
        return {
            "preset": "explicit",
            "pipeline_names": [part.strip() for part in explicit.split(",") if part.strip()],
        }
    preset = os.environ.get("PIPELINE_PRESET", default_preset)
    return {"preset": preset, "pipeline_names": resolve_pipeline_names(preset=preset)}
