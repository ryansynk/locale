import time
import subprocess
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from Bio import SeqIO
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


def build_unitigs(read_files, tmp_dir, cuttlefish_exe_path, cuttlefish_threads):
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
        str(cuttlefish_threads),
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


def process_transcriptome(
    transcriptome_fasta: str,
    output_path: str,
    cuttlefish_threads: int = 1,
    cuttlefish_exe_path: str = None,
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
    else:
        cuttlefish_exe_path = Path(cuttlefish_exe_path)

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
    output_path = Path(output_path)
    writer = pq.ParquetWriter((output_path).resolve(), schema)
    entries_written = 0
    try:
        start = time.time()
        total_time_reads = 0.0
        total_time_unitigs = 0.0
        total_time_writes = 0.0
        seq_records = list(SeqIO.parse(transcriptome_fasta, "fasta"))
        for seq_record in tqdm(seq_records, desc="Writing dataset"):
            # Run ART + Cuttlefish
            try:
                reads_start = time.time()
                reads = run_art(seq_record, tmp_dir, art_exe_path)
                total_time_reads += time.time() - reads_start
                unitigs_start = time.time()
                unitigs = build_unitigs(
                    reads, tmp_dir, cuttlefish_exe_path, cuttlefish_threads
                )
                total_time_unitigs += time.time() - unitigs_start
            except subprocess.CalledProcessError as e:
                print(f"Error processing {seq_record.id}: {e}")
                continue

            table = pa.table(
                [[seq_record.id], [str(seq_record.seq)], [unitigs]], schema=schema
            )

            # Remove the temporary FASTA and read files
            for f in tmp_dir.glob("*"):
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)

            start_writes = time.time()
            writer.write_table(table)
            entries_written += 1
            total_time_writes += time.time() - start_writes

        total_time = time.time() - start
        print(f"TOTAL TIME = {total_time:2f}s")
        print(
            f"ART TIME = {total_time_reads:2f}s, {100*(total_time_reads / total_time):2f}%"
        )
        print(
            f"CUTTLEFISH TIME = {total_time_unitigs:2f}s, {100*(total_time_unitigs / total_time):2f}%"
        )
        print(
            f"WRITES TIME = {total_time_writes:2f}s, {100*(total_time_writes / total_time):2f}%"
        )

    finally:
        writer.close()


if __name__ == "__main__":
    auto_cli(process_transcriptome)
