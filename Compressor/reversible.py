from __future__ import annotations

import bz2
import gzip
import io
import itertools
import lzma
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import brotli
except ImportError:  # pragma: no cover
    brotli = None

try:
    import imagecodecs
except ImportError:  # pragma: no cover
    imagecodecs = None

try:
    import lz4.frame as lz4_frame
except ImportError:  # pragma: no cover
    lz4_frame = None

try:
    import snappy
except ImportError:  # pragma: no cover
    snappy = None

try:
    import zstandard as zstd_mod
except ImportError:  # pragma: no cover
    zstd_mod = None

from .deploy_registry import PIPELINE_ID_TO_NAME, PIPELINE_NAME_TO_ID
from .runtime import (
    COMPRESSOR_FUNCS,
    FILTER_FUNCS,
    ComponentUnavailable,
    PipelineContext,
    canonical_name,
    canonical_pipeline_name,
    normalize_context,
    parse_pipeline_name,
)


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
SIGNED_BY_SIZE = {
    1: np.int8,
    2: np.int16,
    4: np.int32,
    8: np.int64,
}


@dataclass
class CompressionArtifact:
    pipeline_name: str
    pipeline_id: int
    payload: bytes
    tail: bytes
    context_metadata: dict[str, Any]
    aligned_size: int
    filter_trace: list[dict[str, Any]]
    selector_metadata: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "pipeline_id": self.pipeline_id,
            "context": self.context_metadata,
            "aligned_size": self.aligned_size,
            "tail_size": len(self.tail),
            "filter_trace": self.filter_trace,
            "selector": self.selector_metadata,
        }


def shape_product(shape: tuple[int, ...] | list[int] | None) -> int | None:
    if shape is None:
        return None
    out = 1
    for item in shape:
        out *= int(item)
    return out


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


def decode_uleb128(blob: bytes, offset: int = 0) -> tuple[int, int]:
    shift = 0
    value = 0
    idx = offset
    while True:
        if idx >= len(blob):
            raise ValueError("unexpected EOF while decoding ULEB128")
        byte = blob[idx]
        idx += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, idx
        shift += 7
        if shift > 63:
            raise ValueError("ULEB128 value is too large")


def zigzag_encode(value: int) -> int:
    value = int(value)
    return (value << 1) ^ (value >> 63)


def zigzag_decode(value: int) -> int:
    value = int(value)
    return (value >> 1) ^ -(value & 1)


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


def unpack_nonnegative(blob: bytes, count: int, bit_width: int) -> list[int]:
    if count <= 0:
        return []
    if bit_width <= 0:
        return [0] * count
    mask = (1 << bit_width) - 1
    acc = 0
    acc_bits = 0
    out: list[int] = []
    idx = 0
    while len(out) < count:
        while acc_bits < bit_width:
            if idx >= len(blob):
                raise ValueError("unexpected EOF while unpacking bit-packed values")
            acc |= blob[idx] << acc_bits
            acc_bits += 8
            idx += 1
        out.append(acc & mask)
        acc >>= bit_width
        acc_bits -= bit_width
    return out


def typed_dtype(context: PipelineContext) -> np.dtype:
    if context.dtype_name in DTYPE_MAP:
        return np.dtype(DTYPE_MAP[context.dtype_name])
    typesize = int(context.typesize or 1)
    if typesize in UNSIGNED_BY_SIZE:
        return np.dtype(UNSIGNED_BY_SIZE[typesize])
    raise ComponentUnavailable(f"unsupported typesize: {typesize}")


def typed_array(raw: bytes, context: PipelineContext) -> np.ndarray:
    dtype = typed_dtype(context)
    if len(raw) % dtype.itemsize != 0:
        raise ComponentUnavailable("typed transform requires aligned input bytes")
    arr = np.frombuffer(raw, dtype=dtype).copy()
    if context.shape and shape_product(context.shape) == int(arr.size):
        arr = arr.reshape(context.shape)
    return arr


def unsigned_dtype_for_context(context: PipelineContext) -> np.dtype:
    item_size = int(context.typesize or typed_dtype(context).itemsize)
    if item_size not in UNSIGNED_BY_SIZE:
        raise ComponentUnavailable(f"unsupported unsigned item size: {item_size}")
    return np.dtype(UNSIGNED_BY_SIZE[item_size])


def signed_dtype_for_context(context: PipelineContext) -> np.dtype:
    item_size = int(context.typesize or typed_dtype(context).itemsize)
    if item_size not in SIGNED_BY_SIZE:
        raise ComponentUnavailable(f"unsupported signed item size: {item_size}")
    return np.dtype(SIGNED_BY_SIZE[item_size])


def context_from_metadata(metadata: dict[str, Any]) -> PipelineContext:
    context_info = metadata.get("context") or {}
    ctx = PipelineContext(
        dtype_name=context_info.get("dtype_name"),
        shape=tuple(context_info["shape"]) if context_info.get("shape") is not None else None,
        typesize=context_info.get("typesize"),
        endian=context_info.get("endian", "little"),
    )
    return normalize_context(ctx, b"")


def split_aligned_prefix(raw: bytes, context: PipelineContext) -> tuple[bytes, bytes]:
    item_size = int(context.typesize or 1)
    if item_size <= 1:
        return raw, b""
    aligned_size = len(raw) - (len(raw) % item_size)
    return raw[:aligned_size], raw[aligned_size:]


def _packbits_encode_manual(raw: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(raw)
    while i < n:
        run_len = 1
        while i + run_len < n and raw[i + run_len] == raw[i] and run_len < 128:
            run_len += 1
        if run_len >= 3:
            out.append((257 - run_len) & 0xFF)
            out.append(raw[i])
            i += run_len
            continue
        literal_start = i
        literal_len = 0
        while i < n and literal_len < 128:
            run_len = 1
            while i + run_len < n and raw[i + run_len] == raw[i] and run_len < 128:
                run_len += 1
            if run_len >= 3:
                break
            i += 1
            literal_len += 1
        out.append((literal_len - 1) & 0xFF)
        out.extend(raw[literal_start:literal_start + literal_len])
    return bytes(out)


def _packbits_decode_manual(raw: bytes) -> bytes:
    out = bytearray()
    idx = 0
    while idx < len(raw):
        header = struct.unpack("b", bytes([raw[idx]]))[0]
        idx += 1
        if 0 <= header <= 127:
            count = header + 1
            out.extend(raw[idx:idx + count])
            idx += count
        elif -127 <= header <= -1:
            count = 1 - header
            if idx >= len(raw):
                raise ValueError("unexpected EOF in PackBits run")
            out.extend(raw[idx:idx + 1] * count)
            idx += 1
        else:
            continue
    return bytes(out)


def _compress_libdeflate_fallback(data: bytes) -> bytes:
    comp = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-15)
    return comp.compress(data) + comp.flush()


def _decompress_libdeflate_fallback(data: bytes) -> bytes:
    return zlib.decompress(data, wbits=-15)


def apply_forward_filters(
    raw: bytes,
    pipeline_name: str,
    context: PipelineContext,
) -> tuple[bytes, list[dict[str, Any]], str]:
    filter_names, compressor_name = parse_pipeline_name(pipeline_name)
    transformed = raw
    trace: list[dict[str, Any]] = []
    for filter_name in filter_names:
        if imagecodecs is None and filter_name in {"delta", "higher_order_delta"}:
            dtype_name = context.dtype_name or ""
            if dtype_name.startswith("float"):
                raise ComponentUnavailable(
                    f"filter '{filter_name}' needs imagecodecs for bit-exact float reversibility"
                )
        try:
            transformed, meta = FILTER_FUNCS[filter_name](transformed, context, {})
        except Exception as exc:
            if filter_name == "packbits":
                transformed = _packbits_encode_manual(transformed)
                meta = {"engine": "python.packbits"}
            else:
                raise ComponentUnavailable(f"filter '{filter_name}' failed: {exc}") from exc
        trace.append({"name": filter_name, "meta": meta})
    return transformed, trace, compressor_name


def compress_payload(data: bytes, compressor_name: str) -> bytes:
    compressor_name = canonical_name(compressor_name)
    try:
        return COMPRESSOR_FUNCS[compressor_name](data, {})
    except Exception as exc:
        if compressor_name == "libdeflate":
            return _compress_libdeflate_fallback(data)
        raise ComponentUnavailable(f"compressor '{compressor_name}' failed: {exc}") from exc


def decompress_payload(data: bytes, compressor_name: str) -> bytes:
    compressor_name = canonical_name(compressor_name)
    if compressor_name == "zlib":
        return zlib.decompress(data)
    if compressor_name == "libdeflate":
        if imagecodecs is not None and hasattr(imagecodecs, "deflate_decode"):
            return imagecodecs.deflate_decode(data)
        return _decompress_libdeflate_fallback(data)
    if compressor_name == "zstd":
        if zstd_mod is not None:
            return zstd_mod.ZstdDecompressor().decompress(data)
        if imagecodecs is not None and hasattr(imagecodecs, "zstd_decode"):
            return imagecodecs.zstd_decode(data)
        raise ComponentUnavailable("zstd runtime is not available")
    if compressor_name == "lz4":
        if lz4_frame is None:
            raise ComponentUnavailable("lz4 runtime is not available")
        return lz4_frame.decompress(data)
    if compressor_name == "lz4hc":
        if lz4_frame is None:
            raise ComponentUnavailable("lz4 runtime is not available")
        return lz4_frame.decompress(data)
    if compressor_name == "snappy":
        if imagecodecs is not None and hasattr(imagecodecs, "snappy_decode"):
            return imagecodecs.snappy_decode(data)
        if snappy is not None:
            return snappy.decompress(data)
        raise ComponentUnavailable("snappy runtime is not available")
    if compressor_name == "brotli":
        if brotli is None:
            raise ComponentUnavailable("brotli runtime is not available")
        return brotli.decompress(data)
    if compressor_name == "bzip2":
        return bz2.decompress(data)
    if compressor_name == "xz_lzma2":
        return lzma.decompress(data, format=lzma.FORMAT_XZ)
    if compressor_name == "lzf":
        if imagecodecs is not None and hasattr(imagecodecs, "lzf_decode"):
            return imagecodecs.lzf_decode(data)
        raise ComponentUnavailable("lzf runtime is not available")
    if compressor_name == "lzfse":
        if imagecodecs is not None and hasattr(imagecodecs, "lzfse_decode"):
            return imagecodecs.lzfse_decode(data)
        raise ComponentUnavailable("lzfse runtime is not available")
    raise ComponentUnavailable(f"unsupported decompressor: {compressor_name}")


def inverse_shuffle(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    item_size = int(context.typesize or 1)
    if item_size <= 1:
        return raw
    if len(raw) % item_size != 0:
        raise ValueError("shuffle payload is not aligned")
    matrix = np.frombuffer(raw, dtype=np.uint8).reshape(item_size, -1).T
    return matrix.copy().reshape(-1).tobytes()


def inverse_delta(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    if imagecodecs is not None and hasattr(imagecodecs, "delta_decode"):
        arr = typed_array(raw, context)
        decoded = imagecodecs.delta_decode(arr)
        return np.ascontiguousarray(decoded).tobytes()
    dtype_name = context.dtype_name or ""
    if dtype_name.startswith("float"):
        raise ComponentUnavailable("float delta inverse requires imagecodecs for strict losslessness")
    arr = typed_array(raw, context)
    flat = np.ascontiguousarray(arr).reshape(-1).copy()
    for idx in range(1, flat.size):
        flat[idx] = flat[idx] + flat[idx - 1]
    return flat.reshape(arr.shape).tobytes()


def inverse_higher_order_delta(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    order = int(meta.get("order", 2))
    out = raw
    for _ in range(order):
        out = inverse_delta(out, context, meta, aligned_size)
    return out


def inverse_rle(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    out = bytearray()
    idx = 0
    while idx < len(raw):
        run_len, idx = decode_uleb128(raw, idx)
        if idx >= len(raw):
            raise ValueError("unexpected EOF in RLE stream")
        value = raw[idx]
        idx += 1
        out.extend(bytes([value]) * run_len)
    return bytes(out)


def inverse_rle_bitpack_hybrid(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    bit_width = int(meta["bit_width"])
    count = aligned_size // int(context.typesize or 1)
    values: list[int] = []
    idx = 0
    while len(values) < count:
        if idx >= len(raw):
            raise ValueError("unexpected EOF in rle_bitpack_hybrid stream")
        tag = raw[idx]
        idx += 1
        if tag == 1:
            run_len, idx = decode_uleb128(raw, idx)
            value, idx = decode_uleb128(raw, idx)
            values.extend([value] * run_len)
            continue
        if tag != 0:
            raise ValueError(f"unknown rle_bitpack_hybrid block tag: {tag}")
        literal_count, idx = decode_uleb128(raw, idx)
        packed_size = math.ceil(literal_count * bit_width / 8)
        literals = unpack_nonnegative(raw[idx:idx + packed_size], literal_count, bit_width)
        idx += packed_size
        values.extend(literals)
    values = values[:count]

    dtype_name = context.dtype_name or ""
    if dtype_name.startswith("float"):
        unsigned_dtype = unsigned_dtype_for_context(context)
        arr_uint = np.array(values, dtype=unsigned_dtype)
        return arr_uint.view(typed_dtype(context)).tobytes()
    if dtype_name.startswith("int"):
        signed_dtype = signed_dtype_for_context(context)
        signed_values = [zigzag_decode(item) for item in values]
        return np.array(signed_values, dtype=signed_dtype).tobytes()
    return np.array(values, dtype=typed_dtype(context)).tobytes()


def inverse_dictionary(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    idx = 0
    token_size, idx = decode_uleb128(raw, idx)
    token_count, idx = decode_uleb128(raw, idx)
    dictionary_size, idx = decode_uleb128(raw, idx)
    bit_width, idx = decode_uleb128(raw, idx)
    dict_blob_size = token_size * dictionary_size
    dict_blob = raw[idx:idx + dict_blob_size]
    idx += dict_blob_size
    dictionary = [dict_blob[i:i + token_size] for i in range(0, len(dict_blob), token_size)]
    packed = raw[idx:]
    indices = unpack_nonnegative(packed, token_count, bit_width)
    return b"".join(dictionary[item] for item in indices)


def inverse_packbits(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    if imagecodecs is not None and hasattr(imagecodecs, "packbits_decode"):
        return imagecodecs.packbits_decode(raw)
    return _packbits_decode_manual(raw)


def inverse_delta_length_byte_array(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    idx = 0
    record_count, idx = decode_uleb128(raw, idx)
    delta_len, idx = decode_uleb128(raw, idx)
    delta_blob = raw[idx:idx + delta_len]
    idx += delta_len
    payload = raw[idx:]

    deltas: list[int] = []
    didx = 0
    while didx < len(delta_blob):
        value, didx = decode_uleb128(delta_blob, didx)
        deltas.append(zigzag_decode(value))
    if len(deltas) != record_count:
        raise ValueError("record count does not match delta-length stream")

    lengths: list[int] = []
    prev = 0
    for delta in deltas:
        prev = prev + int(delta)
        lengths.append(prev)

    out = bytearray()
    payload_idx = 0
    for length in lengths:
        out.extend(payload[payload_idx:payload_idx + length])
        payload_idx += length
    return bytes(out)


def inverse_delta_byte_array(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    idx = 0
    record_count, idx = decode_uleb128(raw, idx)
    previous = b""
    out = bytearray()
    for _ in range(record_count):
        prefix_len, idx = decode_uleb128(raw, idx)
        suffix_len, idx = decode_uleb128(raw, idx)
        suffix = raw[idx:idx + suffix_len]
        idx += suffix_len
        record = previous[:prefix_len] + suffix
        out.extend(record)
        previous = record
    return bytes(out)


def inverse_tdt(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    item_size = int(context.typesize or 1)
    if item_size <= 1:
        return raw
    matrix = np.frombuffer(raw, dtype=np.uint8).reshape(item_size, -1).T.copy()
    if context.endian == "little":
        order = list(range(item_size - 1, -1, -1))
    else:
        order = list(range(item_size))
    restored = np.empty_like(matrix)
    restored[:, order] = matrix
    return restored.reshape(-1).tobytes()


def inverse_lorenzo(raw: bytes, context: PipelineContext, meta: dict[str, Any], aligned_size: int) -> bytes:
    if not context.shape:
        raise ComponentUnavailable("lorenzo inverse needs shape metadata")
    unsigned_dtype = unsigned_dtype_for_context(context)
    residual = np.frombuffer(raw, dtype=unsigned_dtype).reshape(context.shape)
    out = np.zeros_like(residual)
    mask = (1 << (8 * unsigned_dtype.itemsize)) - 1
    ndim = residual.ndim

    if ndim == 1:
        for i in range(residual.shape[0]):
            left = int(out[i - 1]) if i > 0 else 0
            out[i] = (int(residual[i]) + left) & mask
    elif ndim == 2:
        for i in range(residual.shape[0]):
            for j in range(residual.shape[1]):
                up = int(out[i - 1, j]) if i > 0 else 0
                left = int(out[i, j - 1]) if j > 0 else 0
                diag = int(out[i - 1, j - 1]) if i > 0 and j > 0 else 0
                out[i, j] = (int(residual[i, j]) + up + left - diag) & mask
    elif ndim == 3:
        for i, j, k in itertools.product(
            range(residual.shape[0]),
            range(residual.shape[1]),
            range(residual.shape[2]),
        ):
            xm = int(out[i - 1, j, k]) if i > 0 else 0
            ym = int(out[i, j - 1, k]) if j > 0 else 0
            zm = int(out[i, j, k - 1]) if k > 0 else 0
            xym = int(out[i - 1, j - 1, k]) if i > 0 and j > 0 else 0
            xzm = int(out[i - 1, j, k - 1]) if i > 0 and k > 0 else 0
            yzm = int(out[i, j - 1, k - 1]) if j > 0 and k > 0 else 0
            xyzm = int(out[i - 1, j - 1, k - 1]) if i > 0 and j > 0 and k > 0 else 0
            out[i, j, k] = (int(residual[i, j, k]) + xm + ym + zm - xym - xzm - yzm + xyzm) & mask
    else:
        raise ComponentUnavailable("lorenzo inverse is only implemented for 1D/2D/3D")
    return out.view(typed_dtype(context)).tobytes()


INVERSE_FILTERS = {
    "shuffle": inverse_shuffle,
    "byte_stream_split": inverse_shuffle,
    "delta": inverse_delta,
    "higher_order_delta": inverse_higher_order_delta,
    "rle": inverse_rle,
    "rle_bitpack_hybrid": inverse_rle_bitpack_hybrid,
    "dictionary_categorize": inverse_dictionary,
    "packbits": inverse_packbits,
    "delta_length_byte_array": inverse_delta_length_byte_array,
    "delta_byte_array": inverse_delta_byte_array,
    "tdt": inverse_tdt,
    "lorenzo_residual": inverse_lorenzo,
}


def compress_with_pipeline(
    raw: bytes,
    pipeline_name: str,
    context: PipelineContext,
    *,
    context_metadata: dict[str, Any],
    selector_metadata: dict[str, Any] | None = None,
) -> CompressionArtifact:
    pipeline_name = canonical_pipeline_name(pipeline_name)
    if pipeline_name not in PIPELINE_NAME_TO_ID:
        raise ValueError(f"pipeline is not deployable yet: {pipeline_name}")
    context = normalize_context(context, raw)
    aligned, tail = split_aligned_prefix(raw, context)
    transformed, filter_trace, compressor_name = apply_forward_filters(aligned, pipeline_name, context)
    payload = compress_payload(transformed, compressor_name)
    return CompressionArtifact(
        pipeline_name=pipeline_name,
        pipeline_id=PIPELINE_NAME_TO_ID[pipeline_name],
        payload=payload,
        tail=tail,
        context_metadata=context_metadata,
        aligned_size=len(aligned),
        filter_trace=filter_trace,
        selector_metadata=selector_metadata or {},
    )


def decompress_with_metadata(
    payload: bytes,
    metadata: dict[str, Any],
    tail: bytes,
) -> bytes:
    pipeline_name = canonical_pipeline_name(metadata["pipeline_name"])
    filter_names, compressor_name = parse_pipeline_name(pipeline_name)
    context = context_from_metadata(metadata)
    aligned_size = int(metadata.get("aligned_size", 0))
    transformed = decompress_payload(payload, compressor_name)
    for stage in reversed(metadata.get("filter_trace", [])):
        filter_name = canonical_name(stage["name"])
        if filter_name not in INVERSE_FILTERS:
            raise ComponentUnavailable(f"inverse filter is not implemented: {filter_name}")
        transformed = INVERSE_FILTERS[filter_name](transformed, context, stage.get("meta") or {}, aligned_size)
    if aligned_size and len(transformed) != aligned_size:
        raise ValueError(
            f"decoded aligned payload size mismatch: expected {aligned_size}, got {len(transformed)}"
        )
    return transformed + tail


def decompress_from_pipeline_id(
    payload: bytes,
    pipeline_id: int,
    metadata: dict[str, Any],
    tail: bytes,
) -> bytes:
    pipeline_name = PIPELINE_ID_TO_NAME.get(int(pipeline_id))
    if pipeline_name is None:
        raise ValueError(f"unknown pipeline id: {pipeline_id}")
    metadata = dict(metadata)
    metadata.setdefault("pipeline_name", pipeline_name)
    metadata.setdefault("pipeline_id", pipeline_id)
    return decompress_with_metadata(payload, metadata, tail)


def evaluate_pipeline_size(
    raw: bytes,
    pipeline_name: str,
    context: PipelineContext,
    *,
    context_metadata: dict[str, Any],
    selector_metadata: dict[str, Any] | None = None,
) -> tuple[int, CompressionArtifact]:
    artifact = compress_with_pipeline(
        raw,
        pipeline_name,
        context,
        context_metadata=context_metadata,
        selector_metadata=selector_metadata,
    )
    metadata_size = len(io.BytesIO())  # cheap placeholder to keep the function side-effect free
    del metadata_size
    full_size = len(artifact.payload) + len(artifact.tail)
    for stage in artifact.filter_trace:
        if stage.get("meta"):
            full_size += len(str(stage["meta"]).encode("utf-8"))
    return full_size, artifact
