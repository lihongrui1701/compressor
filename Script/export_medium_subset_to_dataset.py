"""
从 MediumDataset 按 slice_file 顶级目录各抽取至多 N 条，整体打乱后复制到 Dataset/，
并写入新的 data.json（键从 1 起连续编号，slice_file 相对路径与原文件一致）。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

SKIP_TOP_NAMES = {"_meta"}


def top_folder(slice_file: str) -> str:
    s = slice_file.replace("\\", "/").strip("/")
    if "/" not in s:
        return "(flat)"
    return s.split("/", 1)[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="每目录抽样复制到 Dataset")
    ap.add_argument("--source-dir", type=Path, default=Path("./MediumDataset"))
    ap.add_argument("--dest-dir", type=Path, default=Path("./Dataset"))
    ap.add_argument("--per-folder", type=int, default=200, help="每个顶级目录最多抽取条数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_root = args.source_dir.resolve()
    dst_root = args.dest_dir.resolve()
    src_json = src_root / "data.json"

    raw = load_json(src_json)
    src_meta = raw.get("_meta")
    records: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if k in SKIP_TOP_NAMES or not isinstance(v, dict) or "slice_file" not in v:
            continue
        records[k] = v

    by_fold: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for sid, rec in records.items():
        by_fold[top_folder(rec["slice_file"])].append((sid, rec))

    rng = random.Random(args.seed)
    picked: list[tuple[str, dict[str, Any], str]] = []
    summary: list[dict[str, Any]] = []

    for fold in sorted(by_fold.keys(), key=str.lower):
        rows = by_fold[fold][:]
        n_take = min(args.per_folder, len(rows))
        rng.shuffle(rows)
        chosen = rows[:n_take]
        summary.append({"folder": fold, "available": len(rows), "picked": n_take})
        for sid, rec in chosen:
            picked.append((sid, rec, fold))

    rng.shuffle(picked)

    out: dict[str, Any] = {
        "_meta": {
            "script": "export_medium_subset_to_dataset.py",
            "source_root": str(src_root.as_posix()),
            "dest_root": str(dst_root.as_posix()),
            "per_folder": args.per_folder,
            "seed": args.seed,
            "dry_run": args.dry_run,
            "per_folder_summary": summary,
            "total_picked": len(picked),
            "source_medium_meta_keys": list(src_meta.keys())[:20] if isinstance(src_meta, dict) else None,
        }
    }

    n_ok = n_skip = 0
    new_id = 0
    for _old_sid, rec, _ in picked:
        rel = rec["slice_file"].replace("\\", "/")
        src_path = src_root / rel
        dst_path = dst_root / rel
        if not src_path.is_file():
            n_skip += 1
            print(f"  [缺失源文件] {src_path}")
            continue
        if not args.dry_run:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
        new_id += 1
        rec_copy = json.loads(json.dumps(rec))
        out[str(new_id)] = rec_copy
        n_ok += 1

    out["_meta"]["files_copied"] = n_ok
    out["_meta"]["records_skipped_missing_file"] = n_skip

    if not args.dry_run:
        save_json(dst_root / "data.json", out)
        print(f"已写入 {dst_root / 'data.json'}，共 {n_ok} 条（跳过缺失 {n_skip}）")
    else:
        print(f"[dry-run] 将复制 {n_ok} 个文件到 {dst_root}（跳过缺失 {n_skip}），data.json 条目 {n_ok}")

    print("各目录抽取:")
    for row in summary:
        print(f"  {row['folder']!r}: {row['picked']}/{row['available']}")


if __name__ == "__main__":
    main()
