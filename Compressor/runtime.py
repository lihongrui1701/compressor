from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR / "Script"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_registry import (  # noqa: E402
    ComponentUnavailable,
    COMPRESSOR_FUNCS,
    FILTER_FUNCS,
    PipelineContext,
    canonical_name,
    element_size,
    guess_dtype,
    guess_shape,
    normalize_context,
    parse_pipeline_name,
)


def canonical_pipeline_name(name: str) -> str:
    parts = [canonical_name(part) for part in str(name).split("->") if part.strip()]
    return "->".join(parts)


def build_context_for_file(
    file_path: Path,
    raw: bytes,
    *,
    dtype_name: str | None = None,
    shape: tuple[int, ...] | None = None,
    endian: str = "little",
) -> tuple[PipelineContext, dict]:
    dtype_name = canonical_name(dtype_name) if dtype_name else guess_dtype(file_path.suffix)
    typesize = element_size(dtype_name)
    used_size = len(raw)
    shape_source = "user_override" if shape else ("byte_fallback" if dtype_name is None else "1d_fallback")

    if typesize and typesize > 1:
        used_size = len(raw) - (len(raw) % typesize)
    if shape is None and dtype_name is not None:
        inferred_shape, guessed_source = guess_shape(
            file_path=file_path,
            dtype_name=dtype_name,
            dataset_name=file_path.parent.name or "unknown",
            used_size=used_size,
        )
        if inferred_shape is not None:
            shape = tuple(int(item) for item in inferred_shape)
            shape_source = guessed_source
    if shape is None:
        shape = (used_size if used_size > 0 else len(raw),)

    ctx = PipelineContext(
        dtype_name=dtype_name,
        shape=tuple(shape),
        typesize=typesize or 1,
        endian=endian,
    )
    ctx = normalize_context(ctx, raw)
    metadata = {
        "dtype_name": ctx.dtype_name,
        "shape": list(ctx.shape) if ctx.shape is not None else None,
        "typesize": ctx.typesize,
        "endian": ctx.endian,
        "shape_source": shape_source,
        "original_suffix": file_path.suffix,
        "aligned_size": used_size,
        "element_count": used_size // max(1, ctx.typesize or 1),
    }
    return ctx, metadata
