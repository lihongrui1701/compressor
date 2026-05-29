"""
将 ./Dataset 下各顶级子目录中的切片打乱，按序号命名为 dataset/000001.<original_suffix>，
统一移动到 ./Dataset/dataset/，并重写 data.json（键 1..N 与打乱顺序一致）。

"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def tail_from_record(rec: dict[str, Any]) -> str:
    t = rec.get("original_suffix")
    if isinstance(t, str) and t:
        return t
    name = Path(rec["slice_file"].replace("\\", "/")).name
    if "." in name:
        return name.split(".", 1)[1]
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description="打乱并扁平化到 Dataset/dataset/")
    ap.add_argument("--dataset-root", type=Path, default=Path("./Dataset"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="若已全部位于 dataset/ 下，仍按当前文件重新打乱并改写 JSON（需磁盘上文件与 JSON 一致）",
    )
    args = ap.parse_args()

    root = args.dataset_root.resolve()
    json_path = root / "data.json"

    raw = load_json(json_path)
    prev_meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else {}

    records: list[dict[str, Any]] = []
    for k, v in raw.items():
        if k == "_meta" or not isinstance(v, dict) or "slice_file" not in v:
            continue
        records.append(v)

    if not records:
        print("无切片记录，退出")
        return

    n_under_dataset = sum(
        1 for r in records if str(r["slice_file"]).replace("\\", "/").startswith("dataset/")
    )
    if n_under_dataset == len(records) and not args.force:
        print("全部已在 dataset/ 下；若要重新打乱请使用 --force")
        return

    rng = random.Random(args.seed)
    rng.shuffle(records)

    planned: list[tuple[Path, str, dict[str, Any]]] = []
    for new_i, rec in enumerate(records, start=1):
        old_rel = rec["slice_file"].replace("\\", "/")
        src = root / old_rel
        tail = tail_from_record(rec)
        new_rel = f"dataset/{new_i:06d}.{tail}"
        new_rec = copy.deepcopy(rec)
        new_rec["slice_file"] = new_rel
        planned.append((src, new_rel, new_rec))

    missing = [p for p, _, _ in planned if not p.is_file()]
    if missing:
        print(f"缺失源文件 {len(missing)} 个，例: {missing[0]}")
        return

    seen_dst: set[str] = set()
    for _s, rel, _ in planned:
        if rel in seen_dst:
            print(f"目标路径重复: {rel}")
            return
        seen_dst.add(rel)

    if args.dry_run:
        print(f"[dry-run] 将移动 {len(planned)} 个文件到 {root / 'dataset'}，并重写 data.json")
        return

    flat_dir = root / "dataset"
    flat_dir.mkdir(parents=True, exist_ok=True)
    staging = root / "_flatten_staging"
    staging.mkdir(exist_ok=True)

    # 先移到暂存目录，避免个别 src 在 dataset/ 下时与目标名冲突
    staging_moves: list[tuple[Path, Path]] = []
    for i, (src, _new_rel, _) in enumerate(planned):
        tmp = staging / f"_{i:05d}_{src.name}"
        shutil.move(str(src), str(tmp))
        staging_moves.append((tmp, src))

    for (tmp, _orig_src), (_, new_rel, _) in zip(staging_moves, planned, strict=True):
        dst = root / new_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp), str(dst))

    try:
        staging.rmdir()
    except OSError:
        pass

    for p in list(root.iterdir()):
        if p.name in ("data.json", "dataset", "_flatten_staging"):
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    out: dict[str, Any] = {
        "_meta": {
            **prev_meta,
            "script_flatten": "flatten_dataset_to_dataset_folder.py",
            "layout": "flat_under_dataset_subdir",
            "dataset_flat_dir": "dataset",
            "shuffle_seed": args.seed,
            "record_count": len(planned),
        }
    }
    for new_i, (_src, _rel, new_rec) in enumerate(planned, start=1):
        out[str(new_i)] = new_rec

    save_json(json_path, out)
    print(f"完成：{len(planned)} 条已写入 {json_path}，文件位于 {flat_dir}")


if __name__ == "__main__":
    main()
