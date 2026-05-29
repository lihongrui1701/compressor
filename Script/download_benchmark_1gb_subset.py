from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from download_raw_datasets import (
    ARCHIVE_DIR,
    RAW_DIR,
    RCSB_TOP_IDS,
    download,
    ensure_dir,
    ensure_landsat_qa_pixel,
    ensure_oisst,
    ensure_refseq_ecoli,
    ensure_viirs_i1_sdr_swath,
    ensure_viirs_surface_albedo,
    extract_gz,
    log,
)


SELECTED_RCSB_IDS = ["8OTZ", "9A1M", "9A1N", "9A1O", "9D5N"]
ONE_GIB = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the curated <=1 GiB benchmark subset.")
    parser.add_argument(
        "--remove-archives",
        action="store_true",
        help="Remove temporary archives after extraction to keep the on-disk footprint smaller.",
    )
    return parser.parse_args()


def ensure_selected_rcsb() -> None:
    archive_dir = ensure_dir(ARCHIVE_DIR / "RCSB-mmCIF")
    extract_dir = ensure_dir(RAW_DIR / "RCSB-mmCIF")
    for pdb_id in SELECTED_RCSB_IDS:
        output_path = extract_dir / f"{pdb_id}.cif"
        if output_path.exists() and output_path.stat().st_size > 0:
            log(f"Skipping RCSB {pdb_id}: payload already present")
            continue
        archive_path = archive_dir / f"{pdb_id}.cif.gz"
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        download(url, archive_path, enforce_max_archive=True)
        extract_gz(archive_path, output_path)


def payload_total_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and "_archives" not in path.parts
    )


def main() -> None:
    args = parse_args()
    ensure_dir(RAW_DIR)
    ensure_dir(ARCHIVE_DIR)

    ensure_oisst()
    ensure_viirs_surface_albedo()
    ensure_viirs_i1_sdr_swath()
    ensure_landsat_qa_pixel()
    ensure_refseq_ecoli()
    ensure_selected_rcsb()

    total = payload_total_bytes(RAW_DIR)
    print(f"PAYLOAD_TOTAL_BYTES={total}", flush=True)
    print(f"PAYLOAD_TOTAL_GiB={total / ONE_GIB:.6f}", flush=True)

    if total > ONE_GIB:
        raise RuntimeError("Curated payload exceeded 1 GiB.")

    if args.remove_archives and ARCHIVE_DIR.exists():
        shutil.rmtree(ARCHIVE_DIR)
        log("Removed RawDataset/_archives")


if __name__ == "__main__":
    main()
