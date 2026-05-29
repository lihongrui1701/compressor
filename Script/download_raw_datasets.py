#!/usr/bin/env python3
"""
功能：从公开 URL 下载示例科学/遥感/测序等原始数据到 `RawDataset`，支持解压与体积上限控制，便于复现基准输入。
改版：
  - 2026-03-28：补充模块头注释（功能 / 改版 / 改版特点）。
改版特点：
  - 多数据源常量配置；大文件可限制 `MAX_ARCHIVE_BYTES`；归档解压到 `_archives` 等约定目录。
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "RawDataset"
ARCHIVE_DIR = RAW_DIR / "_archives"
MAX_ARCHIVE_BYTES = 50 * 1024 ** 3

OISST_URL = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/"
    "v2.1/access/avhrr/202403/oisst-avhrr-v02r01.20240301.nc"
)
ERA5_PRESSURE_LEVEL_URL = (
    "https://data.rda.ucar.edu/d633000/e5.oper.an.pl/202408/"
    "e5.oper.an.pl.128_130_t.ll025sc.2024080200_2024080223.nc"
)
VIIRS_SURFACE_ALBEDO_URL = (
    "https://noaa-nesdis-n20-pds.s3.amazonaws.com/JPSSRR_SurfAlb/2024/12/19/"
    "SURFALB_v2r2_j01_s202412191643079_e202412191644324_c202412191713523.nc"
)
VIIRS_I1_SDR_SWATH_URL = (
    "https://noaa-nesdis-n20-pds.s3.amazonaws.com/VIIRS-I1-SDR/2024/03/01/"
    "SVI01_j01_d20240301_t0019343_e0020589_b32555_c20240301010545210000_oebc_ops.h5"
)
LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_QA_ASSET = (
    "https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/"
    "oli-tirs/2026/069/075/LC09_L2SR_069075_20260326_20260330_02_T1/"
    "LC09_L2SR_069075_20260326_20260330_02_T1_QA_PIXEL.TIF"
)
IGSR_SAMPLE = "NA19466"
IGSR_RUN = "ERR009256"
IGSR_FASTQ_URLS = [
    "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR009/ERR009256/ERR009256_1.fastq.gz",
    "https://ftp.sra.ebi.ac.uk/vol1/fastq/ERR009/ERR009256/ERR009256_2.fastq.gz",
]
RCSB_TOP_IDS = [
    "9FQR",
    "8GLV",
    "9E5C",
    "8CKB",
    "9MJN",
    "9Y6S",
    "8J07",
    "7Y7A",
    "3J3Q",
    "3J3Y",
    "8QO1",
    "8QO0",
    "9A1M",
    "9A1N",
    "9A1O",
    "8IYJ",
    "9IJJ",
    "9MKB",
    "8OTZ",
    "9D5N",
]


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    raise RuntimeError(message)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_glob_exists(base: Path, pattern: str) -> bool:
    return any(base.glob(pattern))


def dir_has_payload(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def get_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Codex dataset downloader)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def remote_content_length(url: str) -> int | None:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        return None
    proc = subprocess.run(
        [curl, "-I", "-L", "-s", url],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    length = None
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            raw = line.split(":", 1)[1].strip()
            if raw.isdigit():
                length = int(raw)
    return length


def download(url: str, dest: Path, *, enforce_max_archive: bool = False) -> Path:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        fail("curl.exe is required but was not found in PATH.")

    ensure_dir(dest.parent)

    if enforce_max_archive:
        length = remote_content_length(url)
        if length is not None and length > MAX_ARCHIVE_BYTES:
            fail(
                f"Refusing to download {url} because Content-Length "
                f"{length / 1024 ** 3:.2f} GiB exceeds 50 GiB."
            )

    cmd = [
        curl,
        "--fail",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-delay",
        "5",
        "--output",
        str(dest),
        url,
    ]
    if dest.exists() and dest.stat().st_size > 0:
        cmd[1:1] = ["--continue-at", "-"]

    log(f"Downloading {dest.name}")
    subprocess.run(cmd, check=True)
    return dest


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    with tarfile.open(archive_path, "r:*") as tar:
        for member in tar.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if dest_dir.resolve() not in member_path.parents and member_path != dest_dir.resolve():
                fail(f"Unsafe tar member detected in {archive_path}: {member.name}")
        try:
            tar.extractall(dest_dir, filter="data")
        except TypeError:
            tar.extractall(dest_dir)


def extract_zip(archive_path: Path, dest_dir: Path) -> None:
    ensure_dir(dest_dir)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest_dir)


def extract_gz(archive_path: Path, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    log(f"Extracting {archive_path.name} -> {output_path.name}")
    with gzip.open(archive_path, "rb") as src, output_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def unwrap_dataset_wrapper(dest_dir: Path) -> None:
    wrapper = dest_dir / "dataset"
    if not wrapper.is_dir():
        return
    log(f"Normalizing wrapper directory under {dest_dir}")
    for child in list(wrapper.iterdir()):
        target = dest_dir / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))
    try:
        wrapper.rmdir()
    except OSError:
        pass


def landsat_signed_url() -> str:
    token = get_json(
        f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{LANDSAT_COLLECTION}"
    )["token"]
    return f"{LANDSAT_QA_ASSET}?{token}"


def ensure_file_dataset(name: str, url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        log(f"Skipping {name}: file already present")
        return
    ensure_dir(target.parent)
    download(url, target, enforce_max_archive=False)


def ensure_tar_dataset(
    name: str,
    url: str,
    archive_path: Path,
    extract_dir: Path,
    done_pattern: str,
) -> None:
    if path_glob_exists(extract_dir, done_pattern):
        log(f"Skipping {name}: extracted payload already present")
        return
    download(url, archive_path, enforce_max_archive=True)
    log(f"Extracting {archive_path.name}")
    safe_extract_tar(archive_path, extract_dir)
    unwrap_dataset_wrapper(extract_dir)


def ensure_zip_dataset(
    name: str,
    url: str,
    archive_path: Path,
    extract_dir: Path,
    done_pattern: str,
) -> None:
    if path_glob_exists(extract_dir, done_pattern):
        log(f"Skipping {name}: extracted payload already present")
        return
    download(url, archive_path, enforce_max_archive=True)
    log(f"Extracting {archive_path.name}")
    extract_zip(archive_path, extract_dir)


def ensure_oisst() -> None:
    ensure_file_dataset(
        "OISST v2.1",
        OISST_URL,
        RAW_DIR / "OISST" / "oisst-avhrr-v02r01.20240301.nc",
    )


def ensure_era5_pressure_level() -> None:
    ensure_file_dataset(
        "ERA5 pressure-level temperature",
        ERA5_PRESSURE_LEVEL_URL,
        RAW_DIR
        / "ERA5-Pressure-Level"
        / "e5.oper.an.pl.128_130_t.ll025sc.2024080200_2024080223.nc",
    )


def ensure_viirs_surface_albedo() -> None:
    ensure_file_dataset(
        "VIIRS Surface Albedo",
        VIIRS_SURFACE_ALBEDO_URL,
        RAW_DIR
        / "VIIRS-Surface-Albedo"
        / "SURFALB_v2r2_j01_s202412191643079_e202412191644324_c202412191713523.nc",
    )


def ensure_viirs_i1_sdr_swath() -> None:
    ensure_file_dataset(
        "VIIRS I1 SDR swath",
        VIIRS_I1_SDR_SWATH_URL,
        RAW_DIR
        / "VIIRS-I1-SDR-Swath"
        / "SVI01_j01_d20240301_t0019343_e0020589_b32555_c20240301010545210000_oebc_ops.h5",
    )


def ensure_cesm_atm_dataset1() -> None:
    target = RAW_DIR / "CESM-ATM"
    if path_glob_exists(target, "1800x3600/*.f32"):
        log("Skipping CESM-ATM Dataset1: extracted payload already present")
        return
    ensure_tar_dataset(
        "CESM-ATM Dataset1",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz",
        ARCHIVE_DIR / "CESM-ATM" / "SDRBENCH-CESM-ATM-1800x3600.tar.gz",
        target,
        "1800x3600/*.f32",
    )


def ensure_cesm_atm_dataset2() -> None:
    ensure_tar_dataset(
        "CESM-ATM Dataset2",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/CESM-ATM/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz",
        ARCHIVE_DIR / "CESM-ATM" / "SDRBENCH-CESM-ATM-26x1800x3600.tar.gz",
        RAW_DIR / "CESM-ATM",
        "**/*26x1800x3600*",
    )


def ensure_hurricane_isabel() -> None:
    target = RAW_DIR / "Hurricane-ISABEL"
    if path_glob_exists(target, "100x500x500/*.f32"):
        log("Skipping Hurricane ISABEL: extracted payload already present")
        return
    ensure_tar_dataset(
        "Hurricane ISABEL",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz",
        ARCHIVE_DIR / "Hurricane-ISABEL" / "SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz",
        target,
        "100x500x500/*.f32",
    )


def ensure_exafel_dataset2() -> None:
    ensure_tar_dataset(
        "EXAFEL Dataset2",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/EXAFEL/SDRBENCH-EXAFEL-130x1480x1552.tar.gz",
        ARCHIVE_DIR / "EXAFEL" / "SDRBENCH-EXAFEL-130x1480x1552.tar.gz",
        RAW_DIR / "EXAFEL",
        "**/*",
    )


def ensure_nyx() -> None:
    ensure_tar_dataset(
        "NYX 512^3",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz",
        ARCHIVE_DIR / "NYX" / "SDRBENCH-EXASKY-NYX-512x512x512.tar.gz",
        RAW_DIR / "NYX",
        "**/*",
    )


def ensure_s3d() -> None:
    ensure_tar_dataset(
        "S3D",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/S3D/SDRBENCH-S3D.tar.gz",
        ARCHIVE_DIR / "S3D" / "SDRBENCH-S3D.tar.gz",
        RAW_DIR / "S3D",
        "**/*",
    )


def ensure_miranda() -> None:
    target = RAW_DIR / "Miranda"
    if path_glob_exists(target, "**/*.d64"):
        log("Skipping Miranda: extracted payload already present")
        return
    ensure_tar_dataset(
        "Miranda",
        "https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/"
        "raw-data/Miranda/SDRBENCH-Miranda-256x384x384.tar.gz",
        ARCHIVE_DIR / "Miranda" / "SDRBENCH-Miranda-256x384x384.tar.gz",
        target,
        "**/*.d64",
    )


def ensure_landsat_qa_pixel() -> None:
    target = (
        RAW_DIR
        / "Landsat-QA_PIXEL"
        / "LC09_L2SR_069075_20260326_20260330_02_T1_QA_PIXEL.TIF"
    )
    if target.exists() and target.stat().st_size > 0:
        log("Skipping Landsat QA_PIXEL: file already present")
        return
    ensure_file_dataset("Landsat QA_PIXEL", landsat_signed_url(), target)


def ensure_refseq_ecoli() -> None:
    ensure_zip_dataset(
        "RefSeq FASTA E. coli ASM584v2",
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        "GCF_000005845.2/download?include_annotation_type=GENOME_FASTA",
        ARCHIVE_DIR / "RefSeq" / "GCF_000005845.2_ncbi_dataset.zip",
        RAW_DIR / "RefSeq-Ecoli",
        "**/*_genomic.fna",
    )


def ensure_refseq_human() -> None:
    ensure_zip_dataset(
        "RefSeq FASTA Human GRCh38.p14",
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
        "GCF_000001405.40/download?include_annotation_type=GENOME_FASTA",
        ARCHIVE_DIR / "RefSeq" / "GCF_000001405.40_ncbi_dataset.zip",
        RAW_DIR / "RefSeq-Human",
        "**/*_genomic.fna",
    )


def ensure_igsr_fastq() -> None:
    extract_dir = RAW_DIR / "IGSR-FASTQ" / IGSR_SAMPLE
    expected = [
        extract_dir / f"{IGSR_RUN}_1.fastq",
        extract_dir / f"{IGSR_RUN}_2.fastq",
    ]
    if all(p.exists() and p.stat().st_size > 0 for p in expected):
        log("Skipping IGSR FASTQ: extracted payload already present")
        return

    archive_dir = ensure_dir(ARCHIVE_DIR / "IGSR-FASTQ" / IGSR_SAMPLE)
    ensure_dir(extract_dir)
    for url in IGSR_FASTQ_URLS:
        archive_path = archive_dir / Path(url).name
        download(url, archive_path, enforce_max_archive=True)
        extract_gz(archive_path, extract_dir / archive_path.stem)


def ensure_rcsb_mmcif() -> None:
    archive_dir = ensure_dir(ARCHIVE_DIR / "RCSB-mmCIF")
    extract_dir = ensure_dir(RAW_DIR / "RCSB-mmCIF")
    expected = [extract_dir / f"{pdb_id}.cif" for pdb_id in RCSB_TOP_IDS]
    if all(p.exists() and p.stat().st_size > 0 for p in expected):
        log("Skipping RCSB mmCIF: extracted payload already present")
        return

    for pdb_id in RCSB_TOP_IDS:
        archive_path = archive_dir / f"{pdb_id}.cif.gz"
        output_path = extract_dir / f"{pdb_id}.cif"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
        download(url, archive_path, enforce_max_archive=True)
        extract_gz(archive_path, output_path)


DATASET_FUNCS = {
    "oisst_v2_1": ensure_oisst,
    "era5_pressure_level": ensure_era5_pressure_level,
    "viirs_surface_albedo": ensure_viirs_surface_albedo,
    "viirs_i1_sdr_swath": ensure_viirs_i1_sdr_swath,
    "cesm_atm_dataset1": ensure_cesm_atm_dataset1,
    "cesm_atm_dataset2": ensure_cesm_atm_dataset2,
    "hurricane_isabel": ensure_hurricane_isabel,
    "exafel_dataset2": ensure_exafel_dataset2,
    "nyx_512": ensure_nyx,
    "s3d": ensure_s3d,
    "miranda_small": ensure_miranda,
    "landsat_qa_pixel": ensure_landsat_qa_pixel,
    "refseq_ecoli": ensure_refseq_ecoli,
    "refseq_human": ensure_refseq_human,
    "igsr_fastq": ensure_igsr_fastq,
    "rcsb_mmcif": ensure_rcsb_mmcif,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the benchmark datasets into RawDataset."
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(DATASET_FUNCS),
        help="Only process the selected dataset keys.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dir(RAW_DIR)
    ensure_dir(ARCHIVE_DIR)

    selected = args.only or list(DATASET_FUNCS)
    log(f"Processing {len(selected)} dataset targets")
    for key in selected:
        log(f"[{key}]")
        DATASET_FUNCS[key]()
    log("All requested dataset tasks finished.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted.")
        raise
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
