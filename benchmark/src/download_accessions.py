import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import boto3
import zstandard as zstd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from tqdm import tqdm

LOGAN_BUCKET = "logan-pub"


def download_logan(srr_id: str, topdir: Path):
    target_dir = Path(topdir) / srr_id
    target_dir.mkdir(parents=True, exist_ok=True)
    key = f"c/{srr_id}/{srr_id}.contigs.fa.zst"
    dest = target_dir / f"{srr_id}.contigs.fa.zst"
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    try:
        s3.download_file(LOGAN_BUCKET, key, str(dest))
    except ClientError as e:
        if target_dir.exists():
            shutil.rmtree(target_dir)
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return
        raise


def decompress_and_cleanup(zst_path: Path):
    """Worker function to decompress a .fa.zst file and remove the original."""
    # .with_suffix('') removes the final extension (.zst), leaving .fa
    fa_path = zst_path.with_suffix("")

    # Instantiate the decompressor INSIDE the worker to avoid pickling issues
    dctx = zstd.ZstdDecompressor()

    with (
        open(zst_path, "rb") as compressed_file,
        open(fa_path, "wb") as uncompressed_file,
    ):
        # copy_stream is highly optimized for memory-efficient file decompression
        dctx.copy_stream(compressed_file, uncompressed_file)

    # Delete the original .fa.zst file
    zst_path.unlink()

    return fa_path


def download_accessions(
    accession_ids: list[str], accessions_dir: Path, num_procs: int | None = None
):
    if num_procs is None:
        num_procs = os.cpu_count()
        assert num_procs is not None

    # Use less cores for downloading to not saturate bandwidth
    if num_procs >= 4:
        download_procs = 4
    else:
        download_procs = 1

    with ProcessPoolExecutor(max_workers=download_procs) as download_executor:
        futures = [
            download_executor.submit(download_logan, srr_id, accessions_dir)
            for srr_id in accession_ids
        ]

        for future in tqdm(
            as_completed(futures),
            desc="Downloading accs from Logan...",
            total=len(futures),
        ):
            future.result()

    logan_paths = list(accessions_dir.rglob("*.fa.zst"))
    with ProcessPoolExecutor() as decompress_executor:
        futures = [
            decompress_executor.submit(decompress_and_cleanup, logan_path)
            for logan_path in logan_paths
        ]
        for f in tqdm(
            as_completed(futures), desc="Decompressing Files...", total=len(futures)
        ):
            f.result()
