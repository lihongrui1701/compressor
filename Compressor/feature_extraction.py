from __future__ import annotations

import bz2
import gzip
import math
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_MODULE_NAME = "embedded_train_selector_model_science_fixed"

NUM_WINDOWS = 5
WINDOW_BYTES = 128 * 1024
PILOT_BYTES = 128 * 1024
TYPED_MAX_VALUES = 16384
FULL_READ_BYTES = 8 * 1024 * 1024

USE_PILOT_FEATURES = True
USE_TYPED_FLOAT_FEATURES = True
PILOT_GZIP_LEVEL = 1
PILOT_BZ2_LEVEL = 1

BASE_FEATURE_NAMES = [
    "log_file_size",
    "sample_ratio",
    "mean_byte",
    "std_byte",
    "entropy",
    "zero_frac",
    "ff_frac",
    "top1_frac",
    "top4_frac",
    "unique_byte_ratio",
    "same_adjacent_ratio",
    "mean_abs_delta",
    "std_abs_delta",
    "delta_entropy",
    "xor_entropy",
    "window_entropy_mean",
    "window_entropy_std",
    "window_entropy_min",
    "window_entropy_max",
    "is_f32",
    "is_f64",
    "is_unknown_dtype",
    "typed_finite_ratio",
    "typed_best_offset",
    "typed_log_num_values",
    "typed_zero_frac",
    "typed_near_zero_frac",
    "typed_sign_frac",
    "typed_log_range",
    "typed_std_over_meanabs",
    "typed_mean_abs_diff1_norm",
    "typed_mean_abs_diff2_norm",
    "typed_diff_std_ratio",
    "typed_repeat_frac",
    "typed_sign_change_frac",
    "typed_lag1_corr",
    "typed_lag2_corr",
    "typed_exp_entropy",
    "typed_exp_top1",
    "typed_mantissa_zero_score",
]
BASE_FEATURE_NAMES += [f"byte_hist_{idx}" for idx in range(256)]
BASE_FEATURE_NAMES += [f"bit_density_{idx}" for idx in range(8)]
BASE_FEATURE_NAMES += [
    "pilot_gzip_raw",
    "pilot_bz2_raw",
    "pilot_gzip_delta",
    "pilot_gzip_xor",
]

META_FEATURE_NAMES = [
    "meta_ndim",
    "meta_log_element_count",
    "meta_log_shape_dim0",
    "meta_log_shape_dim1",
    "meta_log_shape_dim2",
    "meta_log_shape_dim3",
    "meta_shape_source_dataset_table",
    "meta_shape_source_filename_numbers",
    "meta_shape_source_1d_fallback",
    "meta_shape_source_byte_fallback",
    "meta_shape_source_unknown",
    "meta_dtype_float32",
    "meta_dtype_float64",
    "meta_dtype_unknown_or_other",
]

FULL_FEATURE_NAMES = BASE_FEATURE_NAMES + META_FEATURE_NAMES


def build_feature_names(include_meta: bool = True) -> list[str]:
    return list(FULL_FEATURE_NAMES if include_meta else BASE_FEATURE_NAMES)


def guess_dtype_name(original_suffix: str = "", dtype_hint: str | None = None) -> str | None:
    if dtype_hint:
        s = str(dtype_hint).lower()
        if "float32" in s or s == "f32":
            return "float32"
        if "float64" in s or "double" in s or s in ("f64", "d64"):
            return "float64"

    s = (original_suffix or "").lower()
    if "f32" in s or "float32" in s:
        return "float32"
    if "d64" in s or "f64" in s or "double" in s or "float64" in s:
        return "float64"
    return None


def guess_dtype_flags(original_suffix: str = "", dtype_hint: str | None = None) -> np.ndarray:
    dtype = guess_dtype_name(original_suffix, dtype_hint=dtype_hint)
    return np.array(
        [
            int(dtype == "float32"),
            int(dtype == "float64"),
            int(dtype is None),
        ],
        dtype=np.float32,
    )


def read_sample_windows(path: Path, num_windows: int = NUM_WINDOWS, window_bytes: int = WINDOW_BYTES) -> tuple[bytes, int]:
    file_size = path.stat().st_size
    if file_size <= FULL_READ_BYTES:
        return path.read_bytes(), file_size

    offsets = np.linspace(0, file_size - window_bytes, num_windows, dtype=np.int64)
    chunks: list[bytes] = []
    with path.open("rb") as f:
        for off in offsets:
            f.seek(int(off))
            chunks.append(f.read(window_bytes))
    return b"".join(chunks), file_size


def sample_raw_bytes(raw: bytes, num_windows: int = NUM_WINDOWS, window_bytes: int = WINDOW_BYTES) -> tuple[bytes, int]:
    file_size = len(raw)
    if file_size <= FULL_READ_BYTES:
        return raw, file_size
    offsets = np.linspace(0, file_size - window_bytes, num_windows, dtype=np.int64)
    return b"".join(raw[int(off): int(off) + window_bytes] for off in offsets), file_size


def entropy_from_hist(hist: np.ndarray) -> float:
    p = hist.astype(np.float64)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def safe_ratio(raw_size: int, compressed_size: int) -> float:
    return float(raw_size) / float(max(1, compressed_size))


def byte_delta(raw: bytes) -> bytes:
    if len(raw) <= 1:
        return raw
    x = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
    d = (x[1:] - x[:-1]) & 0xFF
    return d.astype(np.uint8).tobytes()


def byte_xor(raw: bytes) -> bytes:
    if len(raw) <= 1:
        return raw
    x = np.frombuffer(raw, dtype=np.uint8)
    y = np.bitwise_xor(x[1:], x[:-1])
    return y.tobytes()


def pilot_compression_features(raw: bytes) -> np.ndarray:
    probe = raw[: min(PILOT_BYTES, len(raw))]
    if len(probe) == 0:
        return np.zeros(4, dtype=np.float32)

    size = len(probe)
    feats = [
        safe_ratio(size, len(gzip.compress(probe, compresslevel=PILOT_GZIP_LEVEL, mtime=0))),
        safe_ratio(size, len(bz2.compress(probe, compresslevel=PILOT_BZ2_LEVEL))),
    ]

    delta_probe = byte_delta(probe)
    feats.append(safe_ratio(len(delta_probe), len(gzip.compress(delta_probe, compresslevel=PILOT_GZIP_LEVEL, mtime=0))))

    xor_probe = byte_xor(probe)
    feats.append(safe_ratio(len(xor_probe), len(gzip.compress(xor_probe, compresslevel=PILOT_GZIP_LEVEL, mtime=0))))

    return np.array(feats, dtype=np.float32)


def lag_corr(arr: np.ndarray, lag: int) -> float:
    if arr.size <= lag:
        return 0.0
    x = arr[:-lag].astype(np.float64, copy=False)
    y = arr[lag:].astype(np.float64, copy=False)
    clip_abs = 1e6
    x = np.clip(x, -clip_abs, clip_abs)
    y = np.clip(y, -clip_abs, clip_abs)
    x = x - x.mean()
    y = y - y.mean()

    vx = float(np.mean(x * x))
    vy = float(np.mean(y * y))
    if not math.isfinite(vx) or not math.isfinite(vy) or vx <= 0.0 or vy <= 0.0:
        return 0.0
    denom = math.sqrt(vx * vy) + 1e-12
    cov = float(np.mean(x * y))
    if not math.isfinite(cov) or not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    return float(np.clip(cov / denom, -1.0, 1.0))


def build_zero_padded_array(feats: Any, wanted_dim: int) -> np.ndarray:
    out = np.zeros(wanted_dim, dtype=np.float32)
    arr = np.asarray(feats, dtype=np.float32)
    out[: min(len(arr), wanted_dim)] = arr[:wanted_dim]
    return out


def select_typed_view(raw: bytes, dtype_name: str | None) -> dict[str, Any] | None:
    if dtype_name is None:
        return None

    if dtype_name == "float32":
        np_dtype = np.dtype("<f4")
        elem_size = 4
        uint_dtype = np.uint32
        exp_shift = 23
        exp_bits = 8
        mant_mask = np.uint32((1 << 23) - 1)
        sign_shift = 31
    elif dtype_name == "float64":
        np_dtype = np.dtype("<f8")
        elem_size = 8
        uint_dtype = np.uint64
        exp_shift = 52
        exp_bits = 11
        mant_mask = np.uint64((1 << 52) - 1)
        sign_shift = 63
    else:
        return None

    best: dict[str, Any] | None = None
    best_score = -1e18
    for offset in range(elem_size):
        usable = ((len(raw) - offset) // elem_size) * elem_size
        if usable < elem_size * 64:
            continue

        arr_typed = np.frombuffer(raw[offset: offset + usable], dtype=np_dtype)
        if arr_typed.size > TYPED_MAX_VALUES:
            arr_typed = arr_typed[:TYPED_MAX_VALUES]

        finite_mask = np.isfinite(arr_typed)
        finite_ratio = float(finite_mask.mean())
        if finite_ratio <= 0:
            continue

        arr_valid = arr_typed[finite_mask]
        if arr_valid.size < 64:
            continue

        arr64 = arr_valid.astype(np.float64, copy=False)
        corr1 = lag_corr(arr64, 1)
        zero_frac = float((arr64 == 0).mean())
        score = finite_ratio + 0.20 * abs(corr1) + 0.05 * zero_frac
        if score > best_score:
            best = {
                "values": arr64,
                "bits": arr_valid.view(uint_dtype),
                "offset": offset,
                "finite_ratio": finite_ratio,
                "elem_size": elem_size,
                "exp_shift": exp_shift,
                "exp_bits": exp_bits,
                "mant_mask": mant_mask,
                "sign_shift": sign_shift,
            }
            best_score = score
    return best


def extract_typed_float_features(raw: bytes, original_suffix: str = "", dtype_hint: str | None = None) -> np.ndarray:
    dtype_name = guess_dtype_name(original_suffix, dtype_hint=dtype_hint)
    wanted_dim = 18
    if not USE_TYPED_FLOAT_FEATURES or dtype_name is None:
        return np.zeros(wanted_dim, dtype=np.float32)

    picked = select_typed_view(raw, dtype_name)
    if picked is None:
        return np.zeros(wanted_dim, dtype=np.float32)

    arr = np.clip(picked["values"], -1e6, 1e6)
    bits = picked["bits"]
    exp_shift = picked["exp_shift"]
    exp_bits = picked["exp_bits"]
    mant_mask = picked["mant_mask"]
    sign_shift = picked["sign_shift"]

    abs_arr = np.abs(arr)
    std_val = float(arr.std())
    mean_abs = float(abs_arr.mean())
    nz = abs_arr[abs_arr > 0]
    log_range = 0.0
    if nz.size > 0:
        log_range = float(np.log10(nz.max() + 1e-12) - np.log10(nz.min() + 1e-12))

    zero_frac = float((arr == 0).mean())
    near_zero_thr = max(1e-12, 1e-6 * (mean_abs + 1e-12))
    near_zero_frac = float((abs_arr <= near_zero_thr).mean())

    diff1 = np.diff(arr) if arr.size > 1 else np.zeros(1, dtype=np.float64)
    diff2 = np.diff(arr, 2) if arr.size > 2 else np.zeros(1, dtype=np.float64)
    mean_abs_diff1 = float(np.abs(diff1).mean())
    mean_abs_diff2 = float(np.abs(diff2).mean())
    diff_std_ratio = float(diff1.std()) / (std_val + 1e-12)

    repeat_frac = float((arr[1:] == arr[:-1]).mean()) if arr.size > 1 else 1.0
    sign_change_frac = float((np.sign(diff1[1:]) != np.sign(diff1[:-1])).mean()) if diff1.size > 1 else 0.0
    lag1 = lag_corr(arr, 1)
    lag2 = lag_corr(arr, 2)

    exponents = ((bits >> exp_shift) & ((1 << exp_bits) - 1)).astype(np.int64)
    exp_hist = np.bincount(exponents, minlength=(1 << exp_bits)).astype(np.float32)
    exp_hist /= float(max(1.0, exp_hist.sum()))
    exp_entropy = entropy_from_hist(exp_hist) / math.log2(1 << exp_bits)
    exp_top1 = float(exp_hist.max())

    mantissa = bits & mant_mask
    mant_low8_zero = float(((mantissa & 0xFF) == 0).mean())
    mant_low16_zero = float(((mantissa & 0xFFFF) == 0).mean())
    sign_frac = float(((bits >> sign_shift) & 1).mean())

    feats = np.array(
        [
            picked["finite_ratio"],
            picked["offset"] / max(1.0, picked["elem_size"] - 1),
            math.log1p(arr.size),
            zero_frac,
            near_zero_frac,
            sign_frac,
            min(log_range / 20.0, 1.0),
            min(std_val / (mean_abs + 1e-12), 10.0) / 10.0,
            mean_abs_diff1 / (std_val + 1e-12),
            mean_abs_diff2 / (std_val + 1e-12),
            diff_std_ratio,
            repeat_frac,
            sign_change_frac,
            lag1,
            lag2,
            exp_entropy,
            exp_top1,
            mant_low8_zero + mant_low16_zero,
        ],
        dtype=np.float32,
    )
    feats = np.nan_to_num(feats, nan=0.0, posinf=1e6, neginf=-1e6)
    return build_zero_padded_array(feats, wanted_dim)


def extract_base_features_from_sample(
    raw: bytes,
    file_size: int,
    original_suffix: str = "",
    dtype_hint: str | None = None,
) -> np.ndarray:
    x = np.frombuffer(raw, dtype=np.uint8)
    if len(x) == 0:
        x = np.zeros(1, dtype=np.uint8)

    hist = np.bincount(x, minlength=256).astype(np.float32)
    hist /= float(hist.sum())

    entropy = entropy_from_hist(hist)
    zero_frac = float((x == 0).mean())
    ff_frac = float((x == 255).mean())
    top1 = float(hist.max())
    top4 = float(np.sort(hist)[-4:].sum())
    unique_ratio = float((hist > 0).mean())

    x_f = x.astype(np.float32)
    mean_byte = float(x_f.mean()) / 255.0
    std_byte = float(x_f.std()) / 128.0

    if len(x) > 1:
        same_ratio = float((x[1:] == x[:-1]).mean())
        delta = np.abs(np.diff(x.astype(np.int16))).astype(np.uint8)
        delta_hist = np.bincount(delta, minlength=256).astype(np.float32)
        delta_hist /= float(delta_hist.sum())
        delta_entropy = entropy_from_hist(delta_hist)
        mean_abs_delta = float(delta.mean()) / 255.0
        std_abs_delta = float(delta.astype(np.float32).std()) / 128.0

        xorv = np.bitwise_xor(x[1:], x[:-1])
        xor_hist = np.bincount(xorv, minlength=256).astype(np.float32)
        xor_hist /= float(xor_hist.sum())
        xor_entropy = entropy_from_hist(xor_hist)
    else:
        same_ratio = 1.0
        delta_entropy = 0.0
        mean_abs_delta = 0.0
        std_abs_delta = 0.0
        xor_entropy = 0.0

    window_entropies: list[float] = []
    if len(raw) <= WINDOW_BYTES:
        window_entropies.append(entropy)
    else:
        raw_len = len(raw)
        offsets = np.linspace(0, raw_len - WINDOW_BYTES, NUM_WINDOWS, dtype=np.int64)
        for off in offsets:
            seg = np.frombuffer(raw[int(off): int(off) + WINDOW_BYTES], dtype=np.uint8)
            seg_hist = np.bincount(seg, minlength=256).astype(np.float32)
            seg_hist /= float(seg_hist.sum())
            window_entropies.append(entropy_from_hist(seg_hist))

    window_arr = np.array(window_entropies, dtype=np.float32)
    window_stats = np.array(
        [
            float(window_arr.mean()) / 8.0,
            float(window_arr.std()) / 8.0,
            float(window_arr.min()) / 8.0,
            float(window_arr.max()) / 8.0,
        ],
        dtype=np.float32,
    )

    bits = np.unpackbits(x).reshape(-1, 8).astype(np.float32)
    bit_density = bits.mean(axis=0)

    scalar_features = np.array(
        [
            math.log1p(file_size),
            min(1.0, len(raw) / max(1.0, file_size)),
            mean_byte,
            std_byte,
            entropy / 8.0,
            zero_frac,
            ff_frac,
            top1,
            top4,
            unique_ratio,
            same_ratio,
            mean_abs_delta,
            std_abs_delta,
            delta_entropy / 8.0,
            xor_entropy / 8.0,
        ],
        dtype=np.float32,
    )

    parts = [
        scalar_features,
        window_stats,
        guess_dtype_flags(original_suffix, dtype_hint=dtype_hint),
        extract_typed_float_features(raw, original_suffix, dtype_hint=dtype_hint),
        hist,
        bit_density,
    ]
    if USE_PILOT_FEATURES:
        parts.append(pilot_compression_features(raw))
    return np.concatenate(parts).astype(np.float32)


def extract_base_features_from_bytes(
    blob: bytes,
    original_suffix: str = "",
    dtype_hint: str | None = None,
) -> np.ndarray:
    raw, file_size = sample_raw_bytes(blob)
    return extract_base_features_from_sample(raw, file_size, original_suffix, dtype_hint=dtype_hint)


def extract_features(path: Path, original_suffix: str = "", dtype_hint: str | None = None) -> np.ndarray:
    raw, file_size = read_sample_windows(path)
    return extract_base_features_from_sample(raw, file_size, original_suffix, dtype_hint=dtype_hint)


def dtype_name_from_item(item: dict[str, Any]) -> str | None:
    dtype = item.get("dtype_guess") or item.get("dtype") or item.get("source_dtype") or item.get("dtype_name")
    if dtype:
        s = str(dtype).lower()
        if "float32" in s or s == "f32":
            return "float32"
        if "float64" in s or "double" in s or s in ("f64", "d64"):
            return "float64"
    suffix = str(item.get("original_suffix") or "").lower()
    if "f32" in suffix or "float32" in suffix:
        return "float32"
    if "d64" in suffix or "f64" in suffix or "double" in suffix or "float64" in suffix:
        return "float64"
    return None


def shape_meta_features(item: dict[str, Any]) -> np.ndarray:
    shape = item.get("shape") or []
    if not isinstance(shape, (list, tuple)):
        shape = []
    dims = [int(x) for x in shape[:4] if int(x) >= 0]
    while len(dims) < 4:
        dims.append(0)

    elem = item.get("element_count")
    if elem is None:
        prod = 1
        for dim in shape:
            try:
                prod *= int(dim)
            except Exception:
                prod = 0
                break
        elem = prod if prod > 0 else 0

    shape_source = str(item.get("shape_source") or "unknown").lower()
    dtype = dtype_name_from_item(item)
    feats = [
        min(len(shape), 6) / 6.0,
        math.log1p(float(elem or 0.0)) / 30.0,
        *(math.log1p(float(dim)) / 20.0 for dim in dims),
        float(shape_source == "dataset_table"),
        float(shape_source == "filename_numbers"),
        float(shape_source == "1d_fallback"),
        float(shape_source == "byte_fallback"),
        float(shape_source not in {"dataset_table", "filename_numbers", "1d_fallback", "byte_fallback"}),
        float(dtype == "float32"),
        float(dtype == "float64"),
        float(dtype is None),
    ]
    return np.asarray(feats, dtype=np.float32)


def item_from_context(context: Any, context_metadata: dict[str, Any]) -> dict[str, Any]:
    shape = context_metadata.get("shape")
    if shape is None and getattr(context, "shape", None) is not None:
        shape = list(getattr(context, "shape"))

    dtype_name = (
        context_metadata.get("dtype_name")
        or context_metadata.get("dtype")
        or getattr(context, "dtype_name", None)
    )

    item_size = int(context_metadata.get("typesize") or getattr(context, "typesize", None) or 1)
    element_count = context_metadata.get("element_count")
    if element_count is None and shape:
        prod = 1
        for dim in shape:
            prod *= int(dim)
        element_count = prod
    if element_count is None:
        aligned_size = int(context_metadata.get("aligned_size") or 0)
        element_count = aligned_size // max(1, item_size) if aligned_size > 0 else None

    return {
        "shape": shape or [],
        "element_count": element_count,
        "shape_source": context_metadata.get("shape_source", "unknown"),
        "dtype_name": dtype_name,
        "dtype_guess": dtype_name,
        "original_suffix": context_metadata.get("original_suffix", ""),
    }


def build_feature_vector_from_bytes(
    raw: bytes,
    *,
    original_suffix: str = "",
    dtype_hint: str | None = None,
    item_metadata: dict[str, Any] | None = None,
) -> np.ndarray:
    base = extract_base_features_from_bytes(raw, original_suffix, dtype_hint=dtype_hint)
    meta = shape_meta_features(item_metadata or {})
    return np.concatenate([base, meta]).astype(np.float32)


def build_feature_dict(raw: bytes, context: Any, context_metadata: dict[str, Any]) -> dict[str, float]:
    item = item_from_context(context, context_metadata)
    dtype_hint = item.get("dtype_guess") or item.get("dtype_name")
    vector = build_feature_vector_from_bytes(
        raw or b"",
        original_suffix=str(item.get("original_suffix") or ""),
        dtype_hint=str(dtype_hint) if dtype_hint else None,
        item_metadata=item,
    )
    return {name: float(vector[idx]) for idx, name in enumerate(FULL_FEATURE_NAMES)}
