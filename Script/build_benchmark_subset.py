from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_FILES = [
    "Landsat-QA_PIXEL/LC09_L2SR_069075_20260326_20260330_02_T1_QA_PIXEL.TIF",
    "OISST/oisst-avhrr-v02r01.20240301.nc",
    "RCSB-mmCIF/8OTZ.cif",
    "RCSB-mmCIF/9A1M.cif",
    "RCSB-mmCIF/9A1N.cif",
    "RCSB-mmCIF/9A1O.cif",
    "RCSB-mmCIF/9D5N.cif",
    "RefSeq-Ecoli/ncbi_dataset/data/GCF_000005845.2/GCF_000005845.2_ASM584v2_genomic.fna",
    "VIIRS-I1-SDR-Swath/SVI01_j01_d20240301_t0019343_e0020589_b32555_c20240301010545210000_oebc_ops.h5",
    "VIIRS-Surface-Albedo/SURFALB_v2r2_j01_s202412191643079_e202412191644324_c202412191713523.nc",
]

ONE_GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the curated <=1 GiB benchmark subset.")
    parser.add_argument("--source", type=Path, default=Path("RawDataset"))
    parser.add_argument("--dest", type=Path, default=Path("BenchmarkDataset1GB"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    dest = args.dest.resolve()
    total = 0

    for rel in DEFAULT_FILES:
        src = source / rel
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        total += src.stat().st_size
        print(f"{rel}\t{src.stat().st_size}", flush=True)

    print(f"TOTAL\t{total}", flush=True)
    print(f"TOTAL_GiB\t{total / ONE_GIB:.6f}", flush=True)
    if total > ONE_GIB:
        raise RuntimeError("subset exceeded 1 GiB")


if __name__ == "__main__":
    main()
