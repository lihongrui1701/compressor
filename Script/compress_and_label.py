"""
功能：读取 `Dataset/data.json` 中各切片元数据，对 preset 管线逐条压缩，写回最佳管线标签并导出 `train.json`/`test.json` 扁平表。
改版：
  - 2026-03-28：补充模块头注释（功能 / 改版 / 改版特点）。
  - 2026-03-28：默认跑满「压缩器 + 单 filter×压缩器」全量管线（约 420 条，接近 400 条全量）；多线程消费切片队列（同 `benchmark_slices_parallel_workers` 模式）；train/test 用独立 Random 洗牌划分。
  - 2026-04-06：切片级进度（已完成/总数、百分比、ETA），`--progress-every` 控制打印间隔。
  - 2026-04-06：`_slice_worker_loop` 内单切片管线进度条（已完成管线/总管线），`--slice-pipeline-progress-every` 控制打印频率。
改版特点：
  - 默认 `compressors_plus_single_filter_x_compressor`；`--workers` 并行度；`PIPELINE_PRESET`/`PIPELINES`/`--preset` 仍可覆盖；train/test 随机划分（`COMPRESS_LABEL_SPLIT_SEED` 可复现）。
"""

from __future__ import annotations

import argparse
import os
import queue
import random
import secrets
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_registry import (
    BENCHMARK_CURATED_EXCLUDED_COMPRESSORS,
    PipelineContext,
    choose_best_pipeline,
    flatten_result,
    guess_dtype,
    guess_shape,
    load_json,
    pipeline_config_from_env,
    pipeline_source_summary,
    prepare_bytes,
    save_json,
    execute_pipeline,
)


DATASET_ROOT = Path("./Dataset")
# 扁平切片默认子目录（兼容旧导入）；**路径拼接请用 resolve_slice_path(slice_file)**，因 slice_file 已为相对 Dataset/ 的路径（如 dataset/000001.fastq）。
DATASET_DIR = DATASET_ROOT / "dataset"
DATA_JSON = Path("./Dataset/data.json")
TRAIN_JSON = Path("./Dataset/train.json")
TEST_JSON = Path("./Dataset/test.json")

TEST_RATIO = 0.2
# 全量注册组合：裸压缩器 + 各单 filter→压缩器（当前仓库约 420 条，见 resolve_pipeline_names）
DEFAULT_PRESET = "compressors_plus_single_filter_x_compressor"
DEFAULT_WORKERS = 6


def resolve_train_test_split_seed() -> int:
    """训练/测试划分随机种子。未设置 COMPRESS_LABEL_SPLIT_SEED 时每次运行不同，保证划分随机；设置整数可复现同一划分。"""
    raw = os.environ.get("COMPRESS_LABEL_SPLIT_SEED", "").strip()
    if raw:
        return int(raw)
    return secrets.randbelow(2**63)


def source_dataset_name(source_file: str) -> str:
    parts = Path(source_file).parts
    return parts[0] if parts else "unknown"


def resolve_slice_path(slice_file: str) -> Path:
    """slice_file 为相对项目根目录下 `Dataset/` 的路径（如 `dataset/000001.fastq`、`CESM-ATM/000001.f32`）。"""
    return DATASET_ROOT / Path(str(slice_file).replace("\\", "/"))


def build_context(record: dict) -> PipelineContext:
    dtype_name = guess_dtype(record.get("original_suffix", ""))
    source_name = source_dataset_name(record.get("source_file", ""))
    source_path = Path(record.get("source_file", record.get("slice_file", "")))
    raw_path = resolve_slice_path(record["slice_file"])
    used_bytes, raw_size, used_size = prepare_bytes(raw_path, dtype_name)
    shape, shape_source = guess_shape(source_path, dtype_name, source_name, used_size)
    element_count = None
    if dtype_name:
        item_size = 8 if dtype_name in ("float64", "uint64", "int64") else 4 if dtype_name in ("float32", "uint32", "int32") else 2 if dtype_name in ("uint16", "int16") else 1
        element_count = used_size // item_size
    else:
        item_size = 1
        shape = (used_size,)
        shape_source = "byte_fallback"
    if shape is None:
        shape = (element_count or used_size,)
        shape_source = "1d_fallback"

    record["dtype_guess"] = dtype_name
    record["original_size_bytes"] = raw_size
    record["used_size_bytes"] = used_size
    record["tail_dropped_bytes"] = raw_size - used_size
    record["element_count"] = element_count
    record["shape"] = list(shape) if shape is not None else None
    record["shape_source"] = shape_source

    context = PipelineContext(dtype_name=dtype_name, shape=tuple(shape) if shape is not None else None, typesize=item_size)
    return context, used_bytes


def process_one_slice(
    record: dict,
    pipeline_names: list[str],
    *,
    on_pipeline_progress: Callable[[int, int], None] | None = None,
) -> dict:
    context, used_bytes = build_context(record)
    compression: dict[str, dict] = {}
    n_pipe = len(pipeline_names)
    for i, pipeline_name in enumerate(pipeline_names):
        compression[pipeline_name] = execute_pipeline(used_bytes, pipeline_name, context=context)
        if on_pipeline_progress is not None:
            on_pipeline_progress(i + 1, n_pipe)
    best_pipeline, best_ratio = choose_best_pipeline(compression)
    record["compression"] = compression
    record["label"] = best_pipeline
    record["best_pipeline"] = best_pipeline
    record["best_compression_ratio"] = best_ratio
    record["split"] = None
    return record


def build_flat_record(slice_id: str, record: dict, pipeline_names: list[str]) -> dict:
    row = {
        "id": slice_id,
        "slice_file": record["slice_file"],
        "slice_path": str(resolve_slice_path(record["slice_file"])),
        "source_file": record.get("source_file"),
        "original_suffix": record.get("original_suffix"),
        "dtype_guess": record.get("dtype_guess"),
        "original_size_bytes": record.get("original_size_bytes"),
        "used_size_bytes": record.get("used_size_bytes"),
        "tail_dropped_bytes": record.get("tail_dropped_bytes"),
        "element_count": record.get("element_count"),
        "shape": record.get("shape"),
        "shape_source": record.get("shape_source"),
        "label": record.get("label"),
        "best_pipeline": record.get("best_pipeline"),
        "best_compression_ratio": record.get("best_compression_ratio"),
        "split": record.get("split"),
    }
    for pipeline_name in pipeline_names:
        info = record["compression"][pipeline_name]
        row.update(flatten_result(pipeline_name, info))
    return row


def split_train_test(data: dict[str, dict], *, split_seed: int) -> tuple[list[str], list[str]]:
    ids = [slice_id for slice_id, record in data.items() if slice_id != "_meta" and record.get("label")]
    rng = random.Random(split_seed)
    rng.shuffle(ids)

    if len(ids) <= 1:
        test_count = 0
    else:
        test_count = max(1, int(len(ids) * TEST_RATIO))
        test_count = min(len(ids) - 1, test_count)

    test_ids = ids[:test_count]
    train_ids = ids[test_count:]

    for slice_id, record in data.items():
        if slice_id == "_meta":
            continue
        record["split"] = None
    for slice_id in train_ids:
        data[slice_id]["split"] = "train"
    for slice_id in test_ids:
        data[slice_id]["split"] = "test"
    return train_ids, test_ids


def _format_eta_sec(sec: float) -> str:
    if sec < 0 or not (sec < 1e308):
        return "?"
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}min"
    return f"{sec / 3600:.2f}h"


def _pipeline_bar(done: int, total: int, width: int = 24) -> str:
    """ASCII 进度条：当前已完成管线数 / 该切片总管线数。"""
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, max(0, int(round(width * done / total))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _slice_worker_loop(
    job_q: queue.Queue,
    records: dict[str, dict],
    pipeline_names: list[str],
    done_counter: list[int],
    progress_lock: threading.Lock,
    total: int,
    t0_mono: list[float],
    progress_every: int,
    worker_id: int,
    slice_pipeline_progress_every: int,
) -> None:
    """与 `benchmark_slices_parallel_workers.worker_loop` 类似：队列中为 slice_id，逐个跑满全 preset。"""
    while True:
        slice_id = job_q.get()
        try:
            if slice_id is None:
                return
            record = records[slice_id]
            slice_file = record["slice_file"]
            ptot = len(pipeline_names)
            step = max(1, slice_pipeline_progress_every)

            def on_pipeline_progress(cur: int, ptot_: int) -> None:
                if slice_pipeline_progress_every <= 0:
                    return
                show = cur == 1 or cur == ptot_ or (cur % step == 0)
                if not show:
                    return
                bar = _pipeline_bar(cur, ptot_)
                pct = 100.0 * cur / ptot_
                with progress_lock:
                    print(
                        f"[W{worker_id}] 本切片管线 {cur}/{ptot_} {bar} {pct:5.1f}% | {slice_file}",
                        flush=True,
                    )

            cb = on_pipeline_progress if slice_pipeline_progress_every > 0 else None
            records[slice_id] = process_one_slice(record, pipeline_names, on_pipeline_progress=cb)
            with progress_lock:
                done_counter[0] += 1
                n = done_counter[0]
                should_print = progress_every <= 1 or (n % progress_every == 0) or n == total
                eta_part = ""
                if should_print and n >= 3 and total > 0:
                    elapsed = time.monotonic() - t0_mono[0]
                    rate = n / elapsed
                    if rate > 1e-9:
                        eta_part = f" ETA≈{_format_eta_sec((total - n) / rate)}"

                if should_print:
                    pct = 100.0 * n / total if total else 0.0
                    path = records[slice_id]["slice_file"]
                    print(
                        f"切片进度 {n}/{total} ({pct:.1f}%){eta_part} | {path}",
                        flush=True,
                    )
        finally:
            job_q.task_done()


def main() -> None:
    ap = argparse.ArgumentParser(description="全量管线压缩打标 + 随机划分 train/test")
    ap.add_argument(
        "--preset",
        type=str,
        default=None,
        help=f"覆盖环境变量中的 PIPELINE_PRESET（默认脚本内: {DEFAULT_PRESET}）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行处理切片的线程数（默认 {DEFAULT_WORKERS}，与 benchmark_slices_parallel_workers 一致）",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1,
        metavar="N",
        help="每完成 N 条切片打印一行进度（含已完成/总数、百分比、ETA）；1 表示每条都打印，设大数可减少日志量",
    )
    ap.add_argument(
        "--slice-pipeline-progress-every",
        type=int,
        default=14,
        metavar="M",
        help="单条切片内：每跑完 M 条管线打印一行进度条（另在 1 与全部完成时各打一行）；0 关闭；1 表示每条管线都打",
    )
    args = ap.parse_args()

    if args.preset:
        os.environ["PIPELINE_PRESET"] = args.preset

    raw = load_json(DATA_JSON)
    pipeline_cfg = pipeline_config_from_env(default_preset=DEFAULT_PRESET)
    pipeline_names = pipeline_cfg["pipeline_names"]
    split_seed = resolve_train_test_split_seed()

    print(f"Pipeline preset: {pipeline_cfg['preset']}", flush=True)
    print(f"Pipeline count: {len(pipeline_names)}", flush=True)
    print(f"Parallel workers: {max(1, args.workers)}", flush=True)
    print(f"Benchmark curated excludes (compressors): {', '.join(BENCHMARK_CURATED_EXCLUDED_COMPRESSORS)}", flush=True)
    print(f"Train/test split seed: {split_seed} (set COMPRESS_LABEL_SPLIT_SEED to fix)", flush=True)
    preview = ", ".join(pipeline_names[:5])
    if preview:
        print(f"Preview: {preview}{' ...' if len(pipeline_names) > 5 else ''}", flush=True)

    records = {k: v for k, v in raw.items() if k != "_meta"}
    slice_ids = sorted(records.keys(), key=lambda x: int(x))
    total = len(slice_ids)
    workers_n = max(1, args.workers)
    progress_every = max(1, args.progress_every)

    slice_pipe_prog = max(0, args.slice_pipeline_progress_every)
    print(f"待处理切片总数: {total}（每条顺序跑 {len(pipeline_names)} 条管线）", flush=True)
    if slice_pipe_prog > 0:
        print(
            f"单切片管线进度: 约每 {max(1, slice_pipe_prog)} 条管线一行（加首尾），工作线程前缀 [W0]…",
            flush=True,
        )

    job_q: queue.Queue = queue.Queue()
    for sid in slice_ids:
        job_q.put(sid)
    for _ in range(workers_n):
        job_q.put(None)

    done_counter = [0]
    progress_lock = threading.Lock()
    t0_mono = [time.monotonic()]
    threads: list[threading.Thread] = []
    for wid in range(workers_n):
        th = threading.Thread(
            target=_slice_worker_loop,
            args=(
                job_q,
                records,
                pipeline_names,
                done_counter,
                progress_lock,
                total,
                t0_mono,
                progress_every,
                wid,
                slice_pipe_prog,
            ),
            daemon=True,
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    meta = {
        "pipeline_preset": pipeline_cfg["preset"],
        "pipeline_names": pipeline_names,
        "parallel_workers": workers_n,
        "benchmark_curated_excluded_compressors": list(BENCHMARK_CURATED_EXCLUDED_COMPRESSORS),
        "filter_names": sorted({name for pipeline_name in pipeline_names for name in pipeline_name.split('->')[:-1]}),
        "compressor_names": sorted({pipeline_name.split('->')[-1] for pipeline_name in pipeline_names}),
        "source_summary": pipeline_source_summary(),
        "test_ratio": TEST_RATIO,
        "train_test_split_seed": split_seed,
    }

    output = {"_meta": meta, **records}
    train_ids, test_ids = split_train_test(output, split_seed=split_seed)

    save_json(DATA_JSON, output)
    train_data = [build_flat_record(slice_id, output[slice_id], pipeline_names) for slice_id in train_ids]
    test_data = [build_flat_record(slice_id, output[slice_id], pipeline_names) for slice_id in test_ids]
    save_json(TRAIN_JSON, train_data)
    save_json(TEST_JSON, test_data)

    print()
    print(f"Detailed data: {DATA_JSON}", flush=True)
    print(f"Train split: {TRAIN_JSON} ({len(train_data)})", flush=True)
    print(f"Test split: {TEST_JSON} ({len(test_data)})", flush=True)


if __name__ == "__main__":
    main()
