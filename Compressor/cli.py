from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import CPSS_EXTENSION, FIXED_HEADER_SIZE
from .container import build_header, decode_metadata, encode_metadata, parse_header
from .deploy_registry import DEPLOYABLE_PIPELINES, PIPELINE_ID_TO_NAME
from .reversible import compress_with_pipeline, decompress_from_pipeline_id
from .runtime import PipelineContext, build_context_for_file, canonical_pipeline_name
from .selector import SELECTOR_MODE_CODES, deployable_pipeline_names, select_pipeline


def parse_shape(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def default_output_for_compress(input_path: Path) -> Path:
    return input_path.with_suffix(input_path.suffix + CPSS_EXTENSION)


def default_output_for_decompress(input_path: Path) -> Path:
    if input_path.suffix.lower() == CPSS_EXTENSION:
        return input_path.with_suffix("")
    return input_path.with_name(input_path.name + ".restored")


def command_compress(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output_for_compress(input_path)
    raw = input_path.read_bytes()
    context, context_meta = build_context_for_file(
        input_path,
        raw,
        dtype_name=args.dtype,
        shape=parse_shape(args.shape),
        endian=args.endian,
    )
    candidates = (
        [canonical_pipeline_name(item) for item in args.pipelines.split(",") if item.strip()]
        if args.pipelines
        else deployable_pipeline_names()
    )
    selection = select_pipeline(
        raw,
        context,
        context_meta,
        mode=args.selector,
        top_k=args.top_k,
        candidates=candidates,
        model_dir=Path(args.model_dir).resolve() if args.model_dir else None,
    )
    selection.artifact.selector_metadata.update(
        {
            "requested_mode": args.selector,
            "resolved_mode": selection.selector_mode,
            "fallback_used": bool(selection.fallback_used),
            "ranking": selection.ranking,
        }
    )
    metadata = selection.artifact.metadata
    metadata_blob = encode_metadata(metadata)
    header_blob = build_header(
        selector_mode=SELECTOR_MODE_CODES.get(selection.selector_mode, 0),
        pipeline_id=selection.artifact.pipeline_id,
        metadata_size=len(metadata_blob),
        tail_size=len(selection.artifact.tail),
        payload_size=len(selection.artifact.payload),
        original_size=len(raw),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(header_blob + metadata_blob + selection.artifact.tail + selection.artifact.payload)

    print(f"input           : {input_path}")
    print(f"output          : {output_path}")
    print(f"selector        : {selection.selector_mode}")
    print(f"pipeline_id     : {selection.artifact.pipeline_id}")
    print(f"pipeline_name   : {selection.artifact.pipeline_name}")
    print(f"original_size   : {len(raw)}")
    print(f"aligned_size    : {selection.artifact.aligned_size}")
    print(f"tail_size       : {len(selection.artifact.tail)}")
    print(f"payload_size    : {len(selection.artifact.payload)}")
    print(f"metadata_size   : {len(metadata_blob)}")


def command_decompress(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else default_output_for_decompress(input_path)
    blob = input_path.read_bytes()
    header = parse_header(blob[:FIXED_HEADER_SIZE])
    offset = FIXED_HEADER_SIZE
    metadata_blob = blob[offset:offset + header.metadata_size]
    offset += header.metadata_size
    tail = blob[offset:offset + header.tail_size]
    offset += header.tail_size
    payload = blob[offset:offset + header.payload_size]
    metadata = decode_metadata(metadata_blob)
    raw = decompress_from_pipeline_id(payload, header.pipeline_id, metadata, tail)
    if len(raw) != header.original_size:
        raise ValueError(f"restored size mismatch: expected {header.original_size}, got {len(raw)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw)

    print(f"input         : {input_path}")
    print(f"output        : {output_path}")
    print(f"pipeline_id   : {header.pipeline_id}")
    print(f"pipeline_name : {metadata.get('pipeline_name') or PIPELINE_ID_TO_NAME.get(header.pipeline_id)}")
    print(f"restored_size : {len(raw)}")


def command_inspect(args: argparse.Namespace) -> None:
    input_path = Path(args.input).resolve()
    blob = input_path.read_bytes()
    header = parse_header(blob[:FIXED_HEADER_SIZE])
    metadata_blob = blob[FIXED_HEADER_SIZE:FIXED_HEADER_SIZE + header.metadata_size]
    metadata = decode_metadata(metadata_blob)
    print(json.dumps(
        {
            "header": {
                "version": header.version,
                "selector_mode": header.selector_mode,
                "pipeline_id": header.pipeline_id,
                "metadata_size": header.metadata_size,
                "tail_size": header.tail_size,
                "payload_size": header.payload_size,
                "original_size": header.original_size,
            },
            "metadata": metadata,
        },
        ensure_ascii=False,
        indent=2,
    ))


def command_check_env(args: argparse.Namespace) -> None:
    sample = np.linspace(0, 1, 1024, dtype=np.float32).tobytes()
    context = PipelineContext(dtype_name="float32", shape=(1024,), typesize=4)
    context_meta = {
        "dtype_name": "float32",
        "shape": [1024],
        "typesize": 4,
        "endian": "little",
        "shape_source": "synthetic",
    }
    print("Available deployable pipelines:")
    for row in DEPLOYABLE_PIPELINES:
        try:
            artifact = compress_with_pipeline(
                sample,
                row.name,
                context,
                context_metadata=context_meta,
                selector_metadata={"mode": "check_env"},
            )
            print(f"  OK   {row.pipeline_id:02d} {row.name:<32} payload={len(artifact.payload)}")
        except Exception as exc:
            print(f"  FAIL {row.pipeline_id:02d} {row.name:<32} reason={exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPSS scientific-data compressor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_compress = sub.add_parser("compress", help="compress a file into .cpss")
    p_compress.add_argument("input", type=str)
    p_compress.add_argument("-o", "--output", type=str, default=None)
    p_compress.add_argument(
        "--selector",
        type=str,
        default="model",
        choices=["auto", "exhaustive", "model", "hybrid"],
    )
    p_compress.add_argument("--top-k", type=int, default=3)
    p_compress.add_argument("--model-dir", type=str, default=None, help="optional nnmax model directory override")
    p_compress.add_argument("--dtype", type=str, default=None, help="optional dtype override, e.g. float32")
    p_compress.add_argument("--shape", type=str, default=None, help="optional shape override, e.g. 100,500,500")
    p_compress.add_argument("--endian", type=str, default="little", choices=["little", "big"])
    p_compress.add_argument("--pipelines", type=str, default=None, help="comma-separated deployable pipeline whitelist")
    p_compress.set_defaults(func=command_compress)

    p_decompress = sub.add_parser("decompress", help="restore a .cpss file")
    p_decompress.add_argument("input", type=str)
    p_decompress.add_argument("-o", "--output", type=str, default=None)
    p_decompress.set_defaults(func=command_decompress)

    p_inspect = sub.add_parser("inspect", help="inspect CPSS header and metadata")
    p_inspect.add_argument("input", type=str)
    p_inspect.set_defaults(func=command_inspect)

    p_check = sub.add_parser("check-env", help="probe deployable pipeline availability")
    p_check.set_defaults(func=command_check_env)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
