from __future__ import annotations

from dataclasses import dataclass

from .runtime import canonical_pipeline_name


@dataclass(frozen=True)
class DeployablePipeline:
    pipeline_id: int
    name: str


_PIPELINE_ROWS = [
    (1, "brotli"),
    (2, "bzip2"),
    (3, "delta->brotli"),
    (4, "delta->bzip2"),
    (5, "delta->libdeflate"),
    (6, "delta->xz_lzma2"),
    (7, "delta->zlib"),
    (8, "delta_byte_array->xz_lzma2"),
    (9, "delta_length_byte_array->xz_lzma2"),
    (10, "dictionary_categorize->bzip2"),
    (11, "higher_order_delta->xz_lzma2"),
    (12, "libdeflate"),
    (13, "lorenzo_residual->xz_lzma2"),
    (14, "packbits->bzip2"),
    (15, "rle->brotli"),
    (16, "rle->xz_lzma2"),
    (17, "rle_bitpack_hybrid->snappy"),
    (18, "rle_bitpack_hybrid->xz_lzma2"),
    (19, "shuffle->bzip2"),
    (20, "shuffle->xz_lzma2"),
    (21, "snappy"),
    (22, "tdt->brotli"),
    (23, "tdt->bzip2"),
    (24, "tdt->xz_lzma2"),
    (25, "xz_lzma2"),
    (26, "zlib"),
    (27, "zstd"),
]

DEPLOYABLE_PIPELINES = tuple(
    DeployablePipeline(pipeline_id=pipeline_id, name=canonical_pipeline_name(name))
    for pipeline_id, name in _PIPELINE_ROWS
)

PIPELINE_NAME_TO_ID = {item.name: item.pipeline_id for item in DEPLOYABLE_PIPELINES}
PIPELINE_ID_TO_NAME = {item.pipeline_id: item.name for item in DEPLOYABLE_PIPELINES}
