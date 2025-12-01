import time
import subprocess
import shutil
import tempfile
import os
import traceback
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from jsonargparse import auto_cli
from tqdm import tqdm


def run_art(seq_record, tmp_dir, art_exe_path, coverage=5, read_len=150):
    """
    Run ART Illumina read simulator on a single transcript sequence.
    Returns paths to simulated reads.
    """
    seq_path = tmp_dir / "seq.fasta"
    SeqIO.write(seq_record, seq_path, "fasta")

    reads_prefix = tmp_dir / "tmp"

    # Define the command and its arguments as a list
    command = [
        art_exe_path,
        "-ss",
        "HS25",
        "-i",
        str(seq_path),
        "-o",
        str(reads_prefix),
        "-l",
        str(read_len),
        "-f",
        str(coverage),
        "-p",
        "-m",
        "400",
        "-s",
        "40",
        "-na",
    ]
    subprocess.run(command, check=True, capture_output=True)

    # ART outputs .fq files with suffixes like _1.fq and _2.fq
    return [(tmp_dir / "tmp1.fq").resolve(), (tmp_dir / "tmp2.fq").resolve()]


def build_unitigs(read_files, tmp_dir, cuttlefish_exe_path):
    """
    Build unitigs from simulated reads using Cuttlefish.
    """
    output_prefix = tmp_dir / "unitigs"
    command = [
        cuttlefish_exe_path,
        "build",
        "-w",
        str(tmp_dir),
        "-s",
        read_files[0],
        read_files[1],
        "-t",
        "1",
        "-k",
        "31",
        "-o",
        str(output_prefix),
        "--read",
    ]
    subprocess.run(
        command,
        check=True,
        capture_output=True,
    )
    unitig_path = output_prefix.with_suffix(".fa")
    unitigs = [str(rec.seq) for rec in SeqIO.parse(unitig_path, "fasta")]
    return unitigs


def process_batch(
    seq_batch: List[SeqIO.SeqRecord],
    art_exe_path: str,
    cuttlefish_exe_path: str,
    base_tmp_dir: Path,
    schema: pa.Schema,
    batch_index: int,
):
    pid = os.getpid()
    tmp_dir = Path(tempfile.mkdtemp(dir=base_tmp_dir))
    batch_path = base_tmp_dir / f"data_part_{pid}_{batch_index}.parquet"
    transcript_ids, query_seqs, unitigs_list = [], [], []
    try:
        for seq_record in seq_batch:
            try:
                reads = run_art(seq_record, tmp_dir, art_exe_path)
                unitigs = build_unitigs(reads, tmp_dir, cuttlefish_exe_path)
                transcript_ids.append(seq_record.id)
                query_seqs.append(str(seq_record.seq))
                unitigs_list.append(unitigs)

                # Remove temporary files
                for f in tmp_dir.glob("*"):
                    if f.is_file():
                        f.unlink()
                    elif f.is_dir():
                        shutil.rmtree(f)
            except Exception:
                traceback.print_exc()
                continue
        if transcript_ids:
            table = pa.table([transcript_ids, query_seqs, unitigs_list], schema=schema)
            pq.write_table(table, batch_path)
            return str(batch_path)
        return None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def process_transcriptome(
    transcriptome_fasta: str,
    num_workers: int,
    batch_size: int = 100,
    cuttlefish_exe_path: Optional[str] = None,
):
    """
    Stream through the transcriptome, generate (query, [unitigs]) pairs,
    and write them to a parquet file
    """

    # Get the path to the directory containing this script
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    art_exe_path = (
        data_dir / "tools" / "art_bin_MountRainier" / "art_illumina"
    ).resolve()
    if cuttlefish_exe_path is None:
        cuttlefish_exe_path = (
            data_dir / "tools" / "cuttlefish" / "bin" / "cuttlefish"
        ).resolve()
        assert cuttlefish_exe_path.is_file(), "Cuttlefish not installed"
    else:
        cuttlefish_exe_path = Path(cuttlefish_exe_path).resolve()

    tmp_dir = (data_dir / "tmp").resolve()
    tmp_dir.mkdir(exist_ok=True)

    # Open ParquetWriter
    schema = pa.schema(
        [
            ("transcript_id", pa.string()),
            ("query_seq", pa.string()),
            ("unitigs", pa.list_(pa.string())),  # nested list column
        ]
    )
    part_files = []
    futures = []
    batch_records = []
    batch_counter = 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        print("Building work queue")
        for i, seq_record in enumerate(SeqIO.parse(transcriptome_fasta, "fasta")):
            batch_records.append(seq_record)
            if len(batch_records) >= batch_size:
                futures.append(
                    executor.submit(
                        process_batch,
                        batch_records,
                        art_exe_path,
                        cuttlefish_exe_path,
                        tmp_dir,
                        schema,
                        batch_counter,
                    )
                )
                batch_counter += 1
                batch_records = []

        # Any leftover sequences
        if batch_records:
            futures.append(
                executor.submit(
                    process_batch,
                    batch_records,
                    art_exe_path,
                    cuttlefish_exe_path,
                    tmp_dir,
                    schema,
                    batch_counter,
                )
            )
            batch_counter += 1
            batch_records = []

        print(f"Added {batch_counter} batches to work queue")
        print("Starting work...")
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = f.result()
            if result:
                part_files.append(result)

    end = time.time()
    elapsed = end - start
    print(f"Total time: {elapsed}")
    final_path = data_dir / "data.parquet"
    tables = [pq.read_table(p) for p in part_files if Path(p).exists()]
    if tables:
        pq.write_table(pa.concat_tables(tables), final_path)


if __name__ == "__main__":
    auto_cli(process_transcriptome)
