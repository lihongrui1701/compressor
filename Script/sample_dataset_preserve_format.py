from __future__ import annotations

import argparse
import fnmatch
import io
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


INPUT_DIR = Path("./RawDataset")
DEFAULT_OUTPUT_ROOT = Path("./MediumDataset")

MIN_KB = 10000
MAX_KB = 50000
RANDOM_SEED = 42
FASTA_LINE_WIDTH = 80

SKIP_FILE_NAMES = {
    "readme.md",
    "md5sum.txt",
}

RAW_NUMERIC_SUFFIXES = (
    ".npy",
    ".bin",
    ".f32",
    ".f64",
    ".d64",
    ".i64",
    ".dat",
    ".raw",
)

KNOWN_RAW_SHAPES = {
    "cesm-atm": [(1800, 3600), (26, 1800, 3600)],
    "hurricane-isabel": [(100, 500, 500)],
    "miranda": [(256, 384, 384)],
    "nyx": [(512, 512, 512)],
    "s3d": [(500, 500, 5500)],
    "exafel": [(130, 1480, 1552), (130, 2048)],
}

DTYPE_HINTS = (
    ("float64", np.float64, ("float64", "double", "d64", "f64")),
    ("float32", np.float32, ("float32", "single", "f32")),
    ("int64", np.int64, ("int64_t", "int64", "i64")),
    ("uint64", np.uint64, ("uint64_t", "uint64", "u64")),
    ("int32", np.int32, ("int32_t", "int32", "i32")),
    ("uint32", np.uint32, ("uint32_t", "uint32", "u32")),
    ("int16", np.int16, ("int16_t", "int16", "i16")),
    ("uint16", np.uint16, ("uint16_t", "uint16", "u16")),
    ("int8", np.int8, ("int8_t", "int8", "i8")),
    ("uint8", np.uint8, ("uint8_t", "uint8", "u8")),
)


@dataclass
class Config:
    input_dir: Path
    output_root: Path
    output_dir: Path
    output_json: Path
    min_bytes: int
    max_bytes: int
    random_seed: int
    shuffle_output: bool
    include_archives: bool
    path_patterns: list[str]
    max_files: int | None
    max_pieces_per_file: int | None

    @property
    def target_bytes(self) -> int:
        return (self.min_bytes + self.max_bytes) // 2


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def product_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def dataset_key(rel_path: Path) -> str:
    if not rel_path.parts:
        return "unknown"
    return rel_path.parts[0].lower()


def get_original_suffix(name: str) -> str:
    if "." in name:
        return name.split(".", 1)[1]
    return ""


def get_output_suffix(name: str) -> str:
    original_suffix = get_original_suffix(name)
    return f".{original_suffix}" if original_suffix else ""


def matches_patterns(rel_path: Path, patterns: list[str]) -> bool:
    if not patterns:
        return True
    rel_posix = rel_path.as_posix()
    return any(fnmatch.fnmatch(rel_posix, pattern) for pattern in patterns)


def path_is_archive(rel_path: Path) -> bool:
    return any(part.lower() == "_archives" for part in rel_path.parts)


def guess_text_encoding_open(path: Path):
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def parse_dtype_from_text(text: str) -> tuple[np.dtype | None, str | None]:
    lower = text.lower()
    for dtype_name, dtype, tokens in DTYPE_HINTS:
        if any(token in lower for token in tokens):
            return np.dtype(dtype), dtype_name
    return None, None


def unique_preserve_order(values: Iterable[tuple[int, ...]]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def dimension_candidates_from_text(text: str) -> list[tuple[int, ...]]:
    lower = text.lower()
    candidates: list[tuple[int, ...]] = []

    for match in re.finditer(r"(\d+(?:x\d+){1,3})", lower):
        dims = tuple(int(part) for part in match.group(1).split("x"))
        candidates.append(dims)

    for match in re.finditer(r"_(\d+)_(\d+)_(\d+)(?=$|[._])", lower):
        candidates.append(tuple(int(match.group(i)) for i in range(1, 4)))

    for match in re.finditer(r"_(\d+)_(\d+)(?=$|[._])", lower):
        candidates.append(tuple(int(match.group(i)) for i in range(1, 3)))

    for match in re.finditer(r"_(\d+)(?=$|[._])", lower):
        candidates.append((int(match.group(1)),))

    return unique_preserve_order(candidates)


def best_2d_shape(n: int) -> tuple[tuple[int, int] | None, float | None]:
    if n <= 0:
        return None, None
    side = int(math.sqrt(n))
    while side >= 1:
        if n % side == 0:
            other = n // side
            score = max(side, other) / max(min(side, other), 1)
            return (side, other), score
        side -= 1
    return None, None


def best_3d_shape(n: int) -> tuple[tuple[int, int, int] | None, float | None]:
    if n <= 0:
        return None, None
    start = int(round(n ** (1 / 3)))
    best_shape: tuple[int, int, int] | None = None
    best_score: float | None = None

    for a in range(start, 0, -1):
        if n % a != 0:
            continue
        shape2, score2 = best_2d_shape(n // a)
        if shape2 is None or score2 is None:
            continue
        shape3 = tuple(sorted((a, shape2[0], shape2[1])))
        score3 = max(shape3) / max(min(shape3), 1)
        if best_shape is None or score3 < best_score:
            best_shape = shape3
            best_score = score3
        if score3 == 1:
            break
    return best_shape, best_score


def detect_source_format(path: Path, rel_path: Path) -> tuple[str | None, str | None]:
    lower_name = path.name.lower()
    lower_suffix = "".join(path.suffixes).lower()

    if lower_name in SKIP_FILE_NAMES:
        return None, "metadata_file"

    if path_is_archive(rel_path):
        return "archive", None

    if lower_name.endswith(".json"):
        return None, "json_metadata"
    if lower_name.endswith(".jsonl"):
        return None, "json_metadata"
    if lower_name.endswith(".npy"):
        return "npy", None
    if lower_name.endswith((".fastq", ".fq")):
        return "fastq", None
    if lower_name.endswith((".fna", ".fasta", ".fa")):
        return "fasta", None
    if lower_name.endswith(".cif"):
        return "mmcif", None
    if lower_name.endswith((".tif", ".tiff")):
        return "tiff", None
    if lower_name.endswith(".nc"):
        return "netcdf", None
    if lower_name.endswith(RAW_NUMERIC_SUFFIXES):
        return "raw_numeric", None

    dtype, _ = parse_dtype_from_text(path.name)
    if dtype is not None:
        return "raw_numeric", None

    if lower_suffix in (".txt", ".csv", ".tsv"):
        return "text_lines", None

    return None, "unknown_or_unsupported"


def choose_rounds(total_bytes: int, ndim: int, min_bytes: int, max_bytes: int) -> int:
    target = (min_bytes + max_bytes) / 2
    best_in_range_rounds: int | None = None
    best_in_range_gap = float("inf")
    best_fallback_rounds = 0
    best_fallback_gap = float("inf")

    for rounds in range(0, 12):
        sample_bytes = total_bytes / ((2 ** ndim) ** rounds)
        gap = abs(sample_bytes - target)
        if gap < best_fallback_gap:
            best_fallback_gap = gap
            best_fallback_rounds = rounds
        if min_bytes <= sample_bytes <= max_bytes and gap < best_in_range_gap:
            best_in_range_rounds = rounds
            best_in_range_gap = gap

    return best_in_range_rounds if best_in_range_rounds is not None else best_fallback_rounds


def stride_sample_piece(arr: np.ndarray, offsets: tuple[int, ...]) -> np.ndarray:
    slices = tuple(slice(offset, None, 2) for offset in offsets)
    return arr[slices]


def iter_stride_samples(
    arr: np.ndarray,
    rounds: int,
    path_trace: list[list[int]] | None = None,
):
    if path_trace is None:
        path_trace = []
    if rounds == 0:
        yield arr, path_trace
        return
    for offsets in product([0, 1], repeat=arr.ndim):
        sampled = stride_sample_piece(arr, offsets)
        yield from iter_stride_samples(
            sampled,
            rounds - 1,
            path_trace + [list(offsets)],
        )


def build_raw_meta(path: Path, rel_path: Path) -> dict[str, Any]:
    size_bytes = path.stat().st_size
    dtype, dtype_name = parse_dtype_from_text(path.name)
    dtype_from = "filename" if dtype is not None else None
    shape_from = None

    candidates: list[tuple[tuple[int, ...], str]] = []
    key = dataset_key(rel_path)
    for shape in KNOWN_RAW_SHAPES.get(key, []):
        candidates.append((shape, "dataset_table"))

    for shape in dimension_candidates_from_text(rel_path.as_posix()):
        candidates.append((shape, "path_or_name"))

    if dtype is None:
        fallback_candidates: list[tuple[np.dtype, str]] = []
        for guessed_dtype in (np.dtype(np.float32), np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.int64)):
            if size_bytes % guessed_dtype.itemsize == 0:
                fallback_candidates.append((guessed_dtype, guessed_dtype.name))
        for guessed_dtype, guessed_name in fallback_candidates:
            count = size_bytes // guessed_dtype.itemsize
            matched_shape = None
            matched_shape_from = None
            for shape, source in candidates:
                if product_int(shape) == count:
                    matched_shape = shape
                    matched_shape_from = source
                    break
            if matched_shape is not None:
                return {
                    "dtype": guessed_dtype,
                    "dtype_name": guessed_name,
                    "dtype_from": "heuristic",
                    "shape": matched_shape,
                    "shape_from": matched_shape_from,
                    "used_size_bytes": size_bytes,
                    "tail_dropped_bytes": 0,
                }
        raise ValueError(f"Cannot infer dtype for raw numeric file: {rel_path.as_posix()}")

    used_size_bytes = size_bytes - (size_bytes % dtype.itemsize)
    count = used_size_bytes // dtype.itemsize

    shape: tuple[int, ...] | None = None
    for candidate, source in candidates:
        if product_int(candidate) == count:
            shape = candidate
            shape_from = source
            break

    if shape is None:
        if count <= 0:
            raise ValueError(f"Raw numeric file is empty after dtype alignment: {rel_path.as_posix()}")
        if count >= 64:
            shape3, score3 = best_3d_shape(count)
            if shape3 is not None and score3 is not None and score3 <= 16:
                shape = shape3
                shape_from = "heuristic_3d"
        if shape is None and count >= 16:
            shape2, score2 = best_2d_shape(count)
            if shape2 is not None and score2 is not None and score2 <= 64:
                shape = shape2
                shape_from = "heuristic_2d"
        if shape is None:
            shape = (count,)
            shape_from = "1d_fallback"

    return {
        "dtype": dtype,
        "dtype_name": dtype_name,
        "dtype_from": dtype_from,
        "shape": tuple(int(value) for value in shape),
        "shape_from": shape_from,
        "used_size_bytes": used_size_bytes,
        "tail_dropped_bytes": size_bytes - used_size_bytes,
    }


class PieceWriter:
    """切片写入 `output_root / <RawDataset 顶层目录名> / tmp_*`，与 `finalize_output` 配合重命名为 `subdir/000001.*`。"""

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.records: list[dict[str, Any]] = []
        self.temp_counter = 1

    @staticmethod
    def dataset_subdir(rel_path: Path) -> str:
        if not rel_path.parts:
            return "unknown"
        # 保留 RawDataset 下第一级目录名大小写，如 CESM-ATM/
        return rel_path.parts[0]

    def write_bytes(self, rel_path: Path, output_suffix: str, payload: bytes, record: dict[str, Any]) -> None:
        sub = self.dataset_subdir(rel_path)
        temp_name = f"tmp_{self.temp_counter:06d}{output_suffix}"
        target_dir = self.output_root / sub
        target_dir.mkdir(parents=True, exist_ok=True)
        temp_path = target_dir / temp_name
        temp_path.write_bytes(payload)
        self.records.append(
            {
                "dataset_subdir": sub,
                "temp_name": temp_name,
                "output_suffix": output_suffix,
                "record": record,
            }
        )
        self.temp_counter += 1


def base_record(
    *,
    rel_path: Path,
    path: Path,
    source_format: str,
    split_method: str,
    piece_index: int,
    slice_size_bytes: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "source_file": rel_path.as_posix(),
        "original_suffix": get_original_suffix(path.name),
        "source_format": source_format,
        "split_method": split_method,
        "sampling_method": split_method,
        "piece_index_in_source": piece_index,
        "source_size_bytes": path.stat().st_size,
        "slice_size_bytes": slice_size_bytes,
    }
    if extra:
        payload.update(extra)
    return payload


def maybe_stop(piece_count: int, config: Config) -> bool:
    return config.max_pieces_per_file is not None and piece_count >= config.max_pieces_per_file


def split_array_contiguously(
    *,
    array: np.ndarray,
    path: Path,
    rel_path: Path,
    writer: PieceWriter,
    config: Config,
    output_suffix: str,
    source_format: str,
    dtype: np.dtype,
    shape: tuple[int, ...],
    shape_from: str,
    dtype_from: str,
    tail_dropped_bytes: int = 0,
) -> int:
    piece_count = 0

    if array.ndim == 1:
        elements_per_piece = max(1, config.target_bytes // dtype.itemsize)
        for start in range(0, array.shape[0], elements_per_piece):
            stop = min(array.shape[0], start + elements_per_piece)
            sample_arr = np.ascontiguousarray(array[start:stop])
            payload = sample_arr.tobytes()
            writer.write_bytes(
                rel_path,
                output_suffix,
                payload,
                base_record(
                    rel_path=rel_path,
                    path=path,
                    source_format=source_format,
                    split_method="contiguous_slice",
                    piece_index=piece_count,
                    slice_size_bytes=len(payload),
                    extra={
                        "dtype": str(dtype),
                        "ndim": 1,
                        "source_shape": list(shape),
                        "sample_shape": list(sample_arr.shape),
                        "shape_from": shape_from,
                        "dtype_from": dtype_from,
                        "element_start": int(start),
                        "element_stop": int(stop),
                        "tail_dropped_bytes": tail_dropped_bytes,
                    },
                ),
            )
            piece_count += 1
            if maybe_stop(piece_count, config):
                break
        return piece_count

    slab_bytes = dtype.itemsize * product_int(array.shape[1:])
    slab_count = max(1, config.target_bytes // max(1, slab_bytes))
    for start in range(0, array.shape[0], slab_count):
        stop = min(array.shape[0], start + slab_count)
        sample_arr = np.ascontiguousarray(array[start:stop])
        payload = sample_arr.tobytes()
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format=source_format,
                split_method="axis0_slice",
                piece_index=piece_count,
                slice_size_bytes=len(payload),
                extra={
                    "dtype": str(dtype),
                    "ndim": array.ndim,
                    "source_shape": list(shape),
                    "sample_shape": list(sample_arr.shape),
                    "shape_from": shape_from,
                    "dtype_from": dtype_from,
                    "axis0_start": int(start),
                    "axis0_stop": int(stop),
                    "tail_dropped_bytes": tail_dropped_bytes,
                },
            ),
        )
        piece_count += 1
        if maybe_stop(piece_count, config):
            break
    return piece_count


def split_raw_numeric(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)

    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r")
        if array.ndim not in (1, 2, 3):
            raise ValueError(f"Unsupported npy ndim={array.ndim}: {rel_path.as_posix()}")
        piece_count = 0
        shape = tuple(array.shape)
        ndim = array.ndim
        dtype = np.dtype(array.dtype)
        total_bytes = int(array.nbytes)
        if total_bytes <= config.max_bytes:
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            payload = buffer.getvalue()
            writer.write_bytes(
                rel_path,
                output_suffix,
                payload,
                base_record(
                    rel_path=rel_path,
                    path=path,
                    source_format="npy",
                    split_method="copy_small_file",
                    piece_index=0,
                    slice_size_bytes=len(payload),
                    extra={
                        "dtype": str(dtype),
                        "ndim": ndim,
                        "source_shape": list(shape),
                        "sample_shape": list(shape),
                        "shape_from": "npy",
                        "dtype_from": "npy",
                    },
                ),
            )
            piece_count = 1
        elif ndim in (2, 3):
            rounds = choose_rounds(total_bytes, ndim, config.min_bytes, config.max_bytes)
            for sample_index, (sample_arr, sampling_path) in enumerate(iter_stride_samples(array, rounds)):
                sample_arr = np.ascontiguousarray(sample_arr)
                buffer = io.BytesIO()
                np.save(buffer, sample_arr, allow_pickle=False)
                payload = buffer.getvalue()
                writer.write_bytes(
                    rel_path,
                    output_suffix,
                    payload,
                    base_record(
                        rel_path=rel_path,
                        path=path,
                        source_format="npy",
                        split_method="stride_2_sampling",
                        piece_index=sample_index,
                        slice_size_bytes=len(payload),
                        extra={
                            "dtype": str(dtype),
                            "ndim": ndim,
                            "source_shape": list(shape),
                            "sample_shape": list(sample_arr.shape),
                            "shape_from": "npy",
                            "dtype_from": "npy",
                            "sampling_rounds": rounds,
                            "sampling_path": sampling_path,
                        },
                    ),
                )
                piece_count += 1
                if maybe_stop(piece_count, config):
                    break
        else:
            piece_count = split_array_contiguously(
                array=array,
                path=path,
                rel_path=rel_path,
                writer=writer,
                config=config,
                output_suffix=output_suffix,
                source_format="npy",
                dtype=dtype,
                shape=shape,
                shape_from="npy",
                dtype_from="npy",
            )
        return {"pieces": piece_count, "format": "npy"}

    meta = build_raw_meta(path, rel_path)
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["dtype"])
    ndim = len(shape)
    piece_count = 0
    array = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        shape=shape,
    )
    total_bytes = int(product_int(shape) * dtype.itemsize)

    if total_bytes <= config.max_bytes:
        payload = array.tobytes()
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="raw_numeric",
                split_method="copy_small_file",
                piece_index=0,
                slice_size_bytes=len(payload),
                extra={
                    "dtype": str(dtype),
                    "ndim": ndim,
                    "source_shape": list(shape),
                    "sample_shape": list(shape),
                    "shape_from": meta["shape_from"],
                    "dtype_from": meta["dtype_from"],
                    "tail_dropped_bytes": meta["tail_dropped_bytes"],
                },
            ),
        )
        piece_count = 1
    elif ndim in (2, 3) and dtype.kind in ("f", "i", "u"):
        rounds = choose_rounds(total_bytes, ndim, config.min_bytes, config.max_bytes)
        for sample_index, (sample_arr, sampling_path) in enumerate(iter_stride_samples(array, rounds)):
            sample_arr = np.ascontiguousarray(sample_arr)
            payload = sample_arr.tobytes()
            writer.write_bytes(
                rel_path,
                output_suffix,
                payload,
                base_record(
                    rel_path=rel_path,
                    path=path,
                    source_format="raw_numeric",
                    split_method="stride_2_sampling",
                    piece_index=sample_index,
                    slice_size_bytes=len(payload),
                    extra={
                        "dtype": str(dtype),
                        "ndim": ndim,
                        "source_shape": list(shape),
                        "sample_shape": list(sample_arr.shape),
                        "shape_from": meta["shape_from"],
                        "dtype_from": meta["dtype_from"],
                        "sampling_rounds": rounds,
                        "sampling_path": sampling_path,
                        "tail_dropped_bytes": meta["tail_dropped_bytes"],
                    },
                ),
            )
            piece_count += 1
            if maybe_stop(piece_count, config):
                break
    else:
        piece_count = split_array_contiguously(
            array=array,
            path=path,
            rel_path=rel_path,
            writer=writer,
            config=config,
            output_suffix=output_suffix,
            source_format="raw_numeric",
            dtype=dtype,
            shape=shape,
            shape_from=meta["shape_from"],
            dtype_from=meta["dtype_from"],
            tail_dropped_bytes=meta["tail_dropped_bytes"],
        )

    return {"pieces": piece_count, "format": "raw_numeric"}


def split_fastq(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    target_bytes = config.target_bytes
    piece_reads = 0
    piece_bytes = 0
    piece_index = 0
    total_reads = 0
    buffer: list[str] = []
    start_read_index = 0

    def flush_piece() -> None:
        nonlocal piece_reads, piece_bytes, piece_index, buffer, start_read_index
        if piece_reads == 0:
            return
        text = "".join(buffer)
        payload = text.encode("utf-8")
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="fastq",
                split_method="read_chunk",
                piece_index=piece_index,
                slice_size_bytes=len(payload),
                extra={
                    "read_start": start_read_index,
                    "read_stop": start_read_index + piece_reads,
                    "read_count": piece_reads,
                },
            ),
        )
        piece_index += 1
        start_read_index += piece_reads
        piece_reads = 0
        piece_bytes = 0
        buffer = []

    with guess_text_encoding_open(path) as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()
            if not quality:
                raise ValueError(f"Incomplete FASTQ record in {rel_path.as_posix()}")
            record_text = header + sequence + plus + quality
            buffer.append(record_text)
            piece_reads += 1
            piece_bytes += len(record_text.encode("utf-8"))
            total_reads += 1
            if piece_bytes >= target_bytes:
                flush_piece()
                if maybe_stop(piece_index, config):
                    break
    if not maybe_stop(piece_index, config):
        flush_piece()

    return {"pieces": piece_index, "format": "fastq", "records": total_reads}


def wrap_fasta_sequence(sequence: str, width: int = FASTA_LINE_WIDTH) -> str:
    return "\n".join(sequence[idx:idx + width] for idx in range(0, len(sequence), width))


def split_fasta(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    target_bases = max(1, int(config.target_bytes * FASTA_LINE_WIDTH / (FASTA_LINE_WIDTH + 1)))
    piece_index = 0
    total_records = 0

    current_header: str | None = None
    current_fragments: list[str] = []
    current_bases = 0
    current_start_base = 1
    record_chunk_index = 0
    record_position = 1

    def flush_piece() -> None:
        nonlocal piece_index, current_fragments, current_bases, current_start_base, record_chunk_index
        if current_header is None or current_bases == 0:
            return
        sequence = "".join(current_fragments)
        chunk_header = (
            f"{current_header} | chunk={record_chunk_index} "
            f"bases={current_start_base}-{current_start_base + current_bases - 1}"
        )
        text = f"{chunk_header}\n{wrap_fasta_sequence(sequence)}\n"
        payload = text.encode("utf-8")
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="fasta",
                split_method="sequence_window",
                piece_index=piece_index,
                slice_size_bytes=len(payload),
                extra={
                    "record_header": current_header[1:] if current_header.startswith(">") else current_header,
                    "record_position": record_position,
                    "chunk_index_in_record": record_chunk_index,
                    "base_start": current_start_base,
                    "base_stop": current_start_base + current_bases - 1,
                    "base_count": current_bases,
                },
            ),
        )
        piece_index += 1
        current_start_base += current_bases
        current_fragments = []
        current_bases = 0
        record_chunk_index += 1

    with guess_text_encoding_open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_piece()
                if maybe_stop(piece_index, config):
                    break
                current_header = line
                current_fragments = []
                current_bases = 0
                current_start_base = 1
                record_chunk_index = 0
                total_records += 1
                record_position = total_records
                continue
            if current_header is None:
                raise ValueError(f"FASTA sequence appeared before header in {rel_path.as_posix()}")
            current_fragments.append(line)
            current_bases += len(line)
            if current_bases >= target_bases:
                flush_piece()
                if maybe_stop(piece_index, config):
                    break
    if not maybe_stop(piece_index, config):
        flush_piece()

    return {"pieces": piece_index, "format": "fasta", "records": total_records}


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def split_json(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    serialized = json_bytes(payload)
    if len(serialized) <= config.max_bytes:
        writer.write_bytes(
            rel_path,
            output_suffix,
            serialized,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="json",
                split_method="copy_small_file",
                piece_index=0,
                slice_size_bytes=len(serialized),
            ),
        )
        return {"pieces": 1, "format": "json"}

    items: list[Any]
    container_kind: str
    base_payload: dict[str, Any] | None = None

    if isinstance(payload, list):
        items = list(payload)
        container_kind = "list"
    elif isinstance(payload, dict) and "datasets" in payload and isinstance(payload["datasets"], dict):
        items = list(payload["datasets"].items())
        container_kind = "datasets_dict"
        base_payload = {key: value for key, value in payload.items() if key != "datasets"}
    elif isinstance(payload, dict):
        items = list(payload.items())
        container_kind = "dict"
    else:
        writer.write_bytes(
            rel_path,
            output_suffix,
            serialized,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="json",
                split_method="copy_fallback",
                piece_index=0,
                slice_size_bytes=len(serialized),
            ),
        )
        return {"pieces": 1, "format": "json"}

    piece_index = 0
    chunk: list[Any] = []

    def chunk_payload(entries: list[Any]) -> Any:
        if container_kind == "list":
            return list(entries)
        if container_kind == "datasets_dict":
            return {**(base_payload or {}), "datasets": dict(entries)}
        return dict(entries)

    for item in items:
        test_chunk = chunk + [item]
        if chunk and len(json_bytes(chunk_payload(test_chunk))) > config.max_bytes:
            payload_bytes = json_bytes(chunk_payload(chunk))
            writer.write_bytes(
                rel_path,
                output_suffix,
                payload_bytes,
                base_record(
                    rel_path=rel_path,
                    path=path,
                    source_format="json",
                    split_method="json_chunk",
                    piece_index=piece_index,
                    slice_size_bytes=len(payload_bytes),
                    extra={"entry_count": len(chunk)},
                ),
            )
            piece_index += 1
            if maybe_stop(piece_index, config):
                return {"pieces": piece_index, "format": "json"}
            chunk = [item]
        else:
            chunk = test_chunk

    if chunk:
        payload_bytes = json_bytes(chunk_payload(chunk))
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload_bytes,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="json",
                split_method="json_chunk",
                piece_index=piece_index,
                slice_size_bytes=len(payload_bytes),
                extra={"entry_count": len(chunk)},
            ),
        )
        piece_index += 1

    return {"pieces": piece_index, "format": "json"}


def split_jsonl(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    piece_index = 0
    current_lines: list[str] = []
    current_bytes = 0
    total_lines = 0

    def flush() -> None:
        nonlocal piece_index, current_lines, current_bytes
        if not current_lines:
            return
        text = "".join(current_lines)
        payload = text.encode("utf-8")
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="jsonl",
                split_method="line_chunk",
                piece_index=piece_index,
                slice_size_bytes=len(payload),
                extra={"line_count": len(current_lines)},
            ),
        )
        piece_index += 1
        current_lines = []
        current_bytes = 0

    with guess_text_encoding_open(path) as handle:
        for line in handle:
            current_lines.append(line)
            current_bytes += len(line.encode("utf-8"))
            total_lines += 1
            if current_bytes >= config.target_bytes:
                flush()
                if maybe_stop(piece_index, config):
                    break
    if not maybe_stop(piece_index, config):
        flush()

    return {"pieces": piece_index, "format": "jsonl", "records": total_lines}


def extract_loop_name(tag_lines: list[str], fallback: str) -> str:
    for tag in tag_lines:
        tag = tag.strip()
        if tag.startswith("_"):
            return tag.lstrip("_").replace(".", "_")
    return fallback


def mmcif_piece_bytes(data_name: str, section_lines: list[str]) -> bytes:
    text = f"{data_name}\n#\n" + "".join(section_lines)
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def split_mmcif(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    if path.stat().st_size <= config.max_bytes:
        payload = path.read_bytes()
        writer.write_bytes(
            rel_path,
            get_output_suffix(path.name),
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="mmcif",
                split_method="copy_small_file",
                piece_index=0,
                slice_size_bytes=len(payload),
            ),
        )
        return {"pieces": 1, "format": "mmcif"}

    output_suffix = get_output_suffix(path.name)
    piece_index = 0
    section_lines: list[str] = []
    section_bytes = 0
    data_name = f"data_{path.stem}"

    def flush_section(reason: str, extra: dict[str, Any] | None = None) -> None:
        nonlocal piece_index, section_lines, section_bytes
        if not section_lines:
            return
        payload = mmcif_piece_bytes(data_name, section_lines)
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="mmcif",
                split_method=reason,
                piece_index=piece_index,
                slice_size_bytes=len(payload),
                extra=extra,
            ),
        )
        piece_index += 1
        section_lines = []
        section_bytes = 0

    with guess_text_encoding_open(path) as handle:
        first_line = handle.readline()
        if first_line and first_line.startswith("data_"):
            data_name = first_line.strip() or data_name
        elif first_line:
            section_lines.append(first_line)
            section_bytes += len(first_line.encode("utf-8"))

        pending_line: str | None = None
        while True:
            line = pending_line if pending_line is not None else handle.readline()
            pending_line = None
            if not line:
                break

            if line.startswith("loop_"):
                loop_header = [line]
                while True:
                    next_line = handle.readline()
                    if not next_line:
                        break
                    if next_line.startswith("_"):
                        loop_header.append(next_line)
                    else:
                        pending_line = next_line
                        break

                loop_name = extract_loop_name(loop_header[1:], "loop")
                loop_rows: list[str] = []
                loop_bytes = sum(len(item.encode("utf-8")) for item in loop_header)
                row_count = 0
                emitted_loop_chunks = False

                while True:
                    row_line = pending_line if pending_line is not None else handle.readline()
                    pending_line = None
                    if not row_line or row_line.strip() == "#":
                        if loop_rows:
                            if emitted_loop_chunks or loop_bytes >= config.target_bytes:
                                payload = mmcif_piece_bytes(data_name, loop_header + loop_rows + ["#\n"])
                                writer.write_bytes(
                                    rel_path,
                                    output_suffix,
                                    payload,
                                    base_record(
                                        rel_path=rel_path,
                                        path=path,
                                        source_format="mmcif",
                                        split_method="mmcif_loop_chunk",
                                        piece_index=piece_index,
                                        slice_size_bytes=len(payload),
                                        extra={
                                            "loop_name": loop_name,
                                            "row_count": row_count,
                                        },
                                    ),
                                )
                                piece_index += 1
                                if maybe_stop(piece_index, config):
                                    return {"pieces": piece_index, "format": "mmcif"}
                            else:
                                section_lines.extend(loop_header + loop_rows + ["#\n"])
                                section_bytes += loop_bytes + len("#\n".encode("utf-8"))
                                if section_bytes >= config.target_bytes:
                                    flush_section("mmcif_section_chunk")
                                    if maybe_stop(piece_index, config):
                                        return {"pieces": piece_index, "format": "mmcif"}
                        break

                    loop_rows.append(row_line)
                    row_count += 1
                    loop_bytes += len(row_line.encode("utf-8"))
                    if loop_bytes >= config.target_bytes:
                        if not emitted_loop_chunks and section_lines:
                            flush_section("mmcif_section_chunk")
                            if maybe_stop(piece_index, config):
                                return {"pieces": piece_index, "format": "mmcif"}
                        payload = mmcif_piece_bytes(data_name, loop_header + loop_rows + ["#\n"])
                        writer.write_bytes(
                            rel_path,
                            output_suffix,
                            payload,
                            base_record(
                                rel_path=rel_path,
                                path=path,
                                source_format="mmcif",
                                split_method="mmcif_loop_chunk",
                                piece_index=piece_index,
                                slice_size_bytes=len(payload),
                                extra={
                                    "loop_name": loop_name,
                                    "row_count": row_count,
                                },
                            ),
                        )
                        piece_index += 1
                        if maybe_stop(piece_index, config):
                            return {"pieces": piece_index, "format": "mmcif"}
                        emitted_loop_chunks = True
                        loop_rows = []
                        loop_bytes = sum(len(item.encode("utf-8")) for item in loop_header)
                        row_count = 0
                continue

            section_lines.append(line)
            section_bytes += len(line.encode("utf-8"))
            if line.strip() == "#":
                if section_bytes >= config.target_bytes:
                    flush_section("mmcif_section_chunk")
                    if maybe_stop(piece_index, config):
                        return {"pieces": piece_index, "format": "mmcif"}

    flush_section("mmcif_section_chunk")
    return {"pieces": piece_index, "format": "mmcif"}


def split_tiff(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    if tifffile is None:
        payload = path.read_bytes()
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="tiff",
                split_method="copy_no_tifffile",
                piece_index=0,
                slice_size_bytes=len(payload),
            ),
        )
        return {"pieces": 1, "format": "tiff"}

    if path.stat().st_size <= config.max_bytes:
        payload = path.read_bytes()
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="tiff",
                split_method="copy_small_file",
                piece_index=0,
                slice_size_bytes=len(payload),
            ),
        )
        return {"pieces": 1, "format": "tiff"}

    array = tifffile.imread(path)
    if array.ndim < 2:
        payload = path.read_bytes()
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="tiff",
                split_method="copy_non_image_array",
                piece_index=0,
                slice_size_bytes=len(payload),
            ),
        )
        return {"pieces": 1, "format": "tiff"}

    row_bytes = array.shape[1] * np.dtype(array.dtype).itemsize
    rows_per_piece = max(1, config.target_bytes // max(1, row_bytes))
    piece_index = 0

    for start in range(0, array.shape[0], rows_per_piece):
        stop = min(array.shape[0], start + rows_per_piece)
        chunk = array[start:stop]
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, chunk)
        payload = buffer.getvalue()
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="tiff",
                split_method="row_crop",
                piece_index=piece_index,
                slice_size_bytes=len(payload),
                extra={
                    "source_shape": list(array.shape),
                    "sample_shape": list(chunk.shape),
                    "row_start": int(start),
                    "row_stop": int(stop),
                    "dtype": str(array.dtype),
                },
            ),
        )
        piece_index += 1
        if maybe_stop(piece_index, config):
            break

    return {"pieces": piece_index, "format": "tiff"}


def split_netcdf(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    payload = path.read_bytes()
    split_method = "copy_small_file" if len(payload) <= config.max_bytes else "copy_netcdf_hdf5"
    writer.write_bytes(
        rel_path,
        output_suffix,
        payload,
        base_record(
            rel_path=rel_path,
            path=path,
            source_format="netcdf",
            split_method=split_method,
            piece_index=0,
            slice_size_bytes=len(payload),
            extra={"uses_h5py": h5py is not None},
        ),
    )
    return {"pieces": 1, "format": "netcdf"}


def split_text_lines(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    output_suffix = get_output_suffix(path.name)
    piece_index = 0
    current_lines: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal piece_index, current_lines, current_bytes
        if not current_lines:
            return
        payload = "".join(current_lines).encode("utf-8")
        writer.write_bytes(
            rel_path,
            output_suffix,
            payload,
            base_record(
                rel_path=rel_path,
                path=path,
                source_format="text_lines",
                split_method="line_chunk",
                piece_index=piece_index,
                slice_size_bytes=len(payload),
                extra={"line_count": len(current_lines)},
            ),
        )
        piece_index += 1
        current_lines = []
        current_bytes = 0

    with guess_text_encoding_open(path) as handle:
        for line in handle:
            current_lines.append(line)
            current_bytes += len(line.encode("utf-8"))
            if current_bytes >= config.target_bytes:
                flush()
                if maybe_stop(piece_index, config):
                    break
    if not maybe_stop(piece_index, config):
        flush()

    return {"pieces": piece_index, "format": "text_lines"}


def process_one_file(path: Path, rel_path: Path, writer: PieceWriter, config: Config) -> dict[str, Any]:
    source_format, reason = detect_source_format(path, rel_path)
    if source_format is None:
        return {"status": "skipped", "reason": reason}

    if source_format == "archive":
        reason = "archive_not_split" if config.include_archives else "archive_skipped"
        return {"status": "skipped", "reason": reason}

    if source_format in ("raw_numeric", "npy"):
        info = split_raw_numeric(path, rel_path, writer, config)
    elif source_format == "fastq":
        info = split_fastq(path, rel_path, writer, config)
    elif source_format == "fasta":
        info = split_fasta(path, rel_path, writer, config)
    elif source_format == "json":
        info = split_json(path, rel_path, writer, config)
    elif source_format == "jsonl":
        info = split_jsonl(path, rel_path, writer, config)
    elif source_format == "mmcif":
        info = split_mmcif(path, rel_path, writer, config)
    elif source_format == "tiff":
        info = split_tiff(path, rel_path, writer, config)
    elif source_format == "netcdf":
        info = split_netcdf(path, rel_path, writer, config)
    elif source_format == "text_lines":
        info = split_text_lines(path, rel_path, writer, config)
    else:
        return {"status": "skipped", "reason": f"unhandled_format:{source_format}"}

    info["status"] = "ok"
    return info


def finalize_output(writer: PieceWriter, output_json: Path, output_root: Path, random_seed: int, shuffle_output: bool) -> None:
    """按数据集子目录写入 `output_root/<subdir>/<编号>.<后缀>`；默认顺序编号（不 shuffle）。"""
    items = list(writer.records)
    rng = random.Random(random_seed)
    if shuffle_output:
        rng.shuffle(items)
        per_folder: dict[str, int] = defaultdict(int)
        final_payload: dict[str, Any] = {}
        for global_idx, item in enumerate(items, 1):
            sub = item["dataset_subdir"]
            per_folder[sub] += 1
            local_n = per_folder[sub]
            final_name = f"{local_n:06d}{item['output_suffix']}"
            rel_slice = f"{sub}/{final_name}".replace("\\", "/")
            target_dir = output_root / sub
            (target_dir / item["temp_name"]).rename(target_dir / final_name)
            record = dict(item["record"])
            record["slice_file"] = rel_slice
            final_payload[str(global_idx)] = record
        save_json(output_json, final_payload)
        return

    dir_order: list[str] = []
    for it in items:
        d = it["dataset_subdir"]
        if d not in dir_order:
            dir_order.append(d)
    by_dir: dict[str, list[dict[str, Any]]] = {d: [] for d in dir_order}
    for it in items:
        by_dir[it["dataset_subdir"]].append(it)
    for d in dir_order:
        by_dir[d].sort(key=lambda x: x["temp_name"])

    final_payload = {}
    global_idx = 0
    for d in dir_order:
        for local_idx, item in enumerate(by_dir[d], 1):
            global_idx += 1
            final_name = f"{local_idx:06d}{item['output_suffix']}"
            rel_slice = f"{d}/{final_name}".replace("\\", "/")
            target_dir = output_root / d
            (target_dir / item["temp_name"]).rename(target_dir / final_name)
            record = dict(item["record"])
            record["slice_file"] = rel_slice
            final_payload[str(global_idx)] = record

    save_json(output_json, final_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 RawDataset 分割为 MediumDataset/<数据集>/<编号>.<后缀>，保留原分割逻辑；默认顺序编号。",
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-kb", type=int, default=MIN_KB)
    parser.add_argument("--max-kb", type=int, default=MAX_KB)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--include-archives", action="store_true")
    parser.add_argument("--path-pattern", action="append", default=[], help="fnmatch on relative input path")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-pieces-per-file", type=int, default=None, help="useful for smoke tests")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="打乱切片最终顺序后再编号（旧版 Dataset/ flat 目录时的默认行为类似）",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="兼容旧参数：指定后强制顺序编号（与当前默认相同）",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    output_json = args.output_root / "data.json"
    return Config(
        input_dir=args.input_dir,
        output_root=args.output_root,
        output_dir=args.output_root,
        output_json=output_json,
        min_bytes=args.min_kb * 1024,
        max_bytes=args.max_kb * 1024,
        random_seed=args.seed,
        shuffle_output=bool(args.shuffle) and not bool(args.no_shuffle),
        include_archives=args.include_archives,
        path_patterns=list(args.path_pattern),
        max_files=args.max_files,
        max_pieces_per_file=args.max_pieces_per_file,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if config.min_bytes > config.max_bytes:
        raise ValueError("--min-kb cannot be larger than --max-kb")
    if not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir.resolve()}")

    shutil.rmtree(config.output_root, ignore_errors=True)
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.output_json.parent.mkdir(parents=True, exist_ok=True)

    all_files = sorted(path for path in config.input_dir.rglob("*") if path.is_file())
    selected_files: list[Path] = []
    for path in all_files:
        rel_path = path.relative_to(config.input_dir)
        if not matches_patterns(rel_path, config.path_patterns):
            continue
        selected_files.append(path)
        if config.max_files is not None and len(selected_files) >= config.max_files:
            break

    writer = PieceWriter(config.output_root)
    outcome_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()
    skipped_files: list[dict[str, str]] = []

    print(f"Input directory:  {config.input_dir.resolve()}")
    print(f"Output root:      {config.output_root.resolve()}")
    print(f"Selected files:   {len(selected_files)}")
    print(f"Target size:      {config.min_bytes} ~ {config.max_bytes} bytes")
    if config.max_pieces_per_file is not None:
        print(f"Max pieces/file:  {config.max_pieces_per_file}")
    print()

    for index, path in enumerate(selected_files, 1):
        rel_path = path.relative_to(config.input_dir)
        print(f"[{index}/{len(selected_files)}] {rel_path.as_posix()}", flush=True)
        result = process_one_file(path, rel_path, writer, config)
        if result["status"] == "ok":
            fmt = str(result.get("format", "unknown"))
            outcome_counter["processed_files"] += 1
            split_counter[fmt] += int(result.get("pieces", 0))
            print(f"  -> format={fmt} pieces={result.get('pieces', 0)}")
        else:
            reason = str(result.get("reason", "unknown"))
            outcome_counter["skipped_files"] += 1
            outcome_counter[f"skip:{reason}"] += 1
            skipped_files.append({"path": rel_path.as_posix(), "reason": reason})
            print(f"  -> skipped ({reason})")

    meta = {
        "script": Path(__file__).name,
        "layout": "medium_dataset_by_raw_topdir",
        "input_dir": str(config.input_dir),
        "output_root": str(config.output_root),
        "min_bytes": config.min_bytes,
        "max_bytes": config.max_bytes,
        "target_bytes": config.target_bytes,
        "random_seed": config.random_seed,
        "shuffle_output": config.shuffle_output,
        "include_archives": config.include_archives,
        "path_patterns": config.path_patterns,
        "max_files": config.max_files,
        "max_pieces_per_file": config.max_pieces_per_file,
        "selected_files": len(selected_files),
        "processed_files": outcome_counter["processed_files"],
        "skipped_files": outcome_counter["skipped_files"],
        "piece_count_by_format": dict(split_counter),
        "skipped": skipped_files[:200],
    }

    finalize_output(
        writer=writer,
        output_json=config.output_json,
        output_root=config.output_root,
        random_seed=config.random_seed,
        shuffle_output=config.shuffle_output,
    )

    with config.output_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload = {"_meta": meta, **payload}
    save_json(config.output_json, payload)

    print()
    print("Done")
    print(f"Output root (含各数据集子目录): {config.output_root.resolve()}")
    print(f"Output metadata:    {config.output_json}")
    print(f"Generated pieces:   {len(writer.records)}")
    if split_counter:
        print("Pieces by format:")
        for fmt, count in sorted(split_counter.items()):
            print(f"  {fmt}: {count}")
    if outcome_counter["skipped_files"]:
        print("Skip summary:")
        for key, value in sorted(outcome_counter.items()):
            if key.startswith("skip:"):
                print(f"  {key[5:]}: {value}")


if __name__ == "__main__":
    main()
