import subprocess
import tempfile
from pathlib import Path

import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .base_index import BaseIndex
from .config import ExperimentConfig, MantisConfig


class MantisIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, MantisConfig)
        self.model_cfg: MantisConfig = cfg.model
        self.mantis_exe: str = self.model_cfg.executable
        self.squeakr_exe: str = self.model_cfg.squeakr_executable
        self.seqtk_exe: str = self.model_cfg.seqtk_executable
        self.k = self.model_cfg.k
        self.index_prefix_path: Path | None = None
        print(f"Running mantis with executable = {self.mantis_exe}")

    def load(self, index_path: Path):
        self.index_prefix_path = index_path

    def build(self, accessions: list[Path], index_path: Path):
        index_path: Path = index_path.resolve()
        index_path.mkdir(exist_ok=True, parents=True)
        fastq_files_path = index_path / "fastq_files"
        fastq_files_path.mkdir(exist_ok=True, parents=True)
        squeakr_files_path = index_path / "squeakr_files"
        squeakr_files_path.mkdir(exist_ok=True, parents=True)

        # Convert to fastq
        fastq_accessions = []
        for path in accessions:
            out_fname = path.stem + ".fastq"
            out_path = fastq_files_path / out_fname
            with open(out_path, "w") as fout:
                subprocess.run(
                    [self.seqtk_exe, "seq", "-F", "I", str(path)],
                    stdout=fout,
                    check=True,
                )
            fastq_accessions.append(out_path)

        squeakr_accessions: list[Path] = []
        num_threads = self.model_cfg.num_threads
        log_slots = self.model_cfg.log_slots
        for path in fastq_accessions:
            squeakr_out_fname: str = path.stem + ".squeakr"
            squeakr_out_path: Path = squeakr_files_path / squeakr_out_fname
            subprocess.run(
                [
                    self.squeakr_exe,
                    "count",
                    "-e",
                    "-k",
                    str(self.k),
                    "-c",
                    "1",
                    "--no-counts",
                    "-s",
                    str(log_slots),
                    "-t",
                    str(num_threads),
                    "-o",
                    str(squeakr_out_path),
                    str(path),
                ],
                check=True,
            )
            squeakr_accessions.append(squeakr_out_path)

        manifest_path = index_path / "squeakr_manifest.lst"
        with open(manifest_path, "w") as f:
            for path in squeakr_accessions:
                f.write(f"{str(path)}\n")

        subprocess.run(
            [
                self.mantis_exe,
                "build",
                "-s",
                str(log_slots),
                "-i",
                str(manifest_path),
                "-o",
                str(index_path) + "/",
            ],
            check=True,
        )

        subprocess.run(
            [
                self.mantis_exe,
                "mst",
                "-p",
                str(index_path) + "/",
                "-t",
                str(num_threads),
                "-k",
            ],
            check=True,
        )
        self.manifest_path = manifest_path
        self.index_prefix_path = index_path

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        queries = queries.with_row_index()
        queries = queries.with_columns(
            pl.col("query_sequence").str.len_chars().alias("query_length")
        )
        records = [
            SeqRecord(Seq(seq), id=qid, description="")
            for qid, seq in zip(
                queries["query_id"].to_list(),
                queries["query_sequence"].to_list(),
            )
        ]
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".fa", delete=True
            ) as fasta_tmp,
            tempfile.NamedTemporaryFile(
                mode="r", suffix=".tsv", delete=True
            ) as result_tmp,
        ):
            # Write queries as FASTA
            SeqIO.write(records, fasta_tmp, "fasta")
            fasta_tmp.flush()

            subprocess.run(
                [
                    self.mantis_exe,
                    "query",
                    "-k",
                    str(self.k),
                    "-p",
                    str(self.index_prefix_path) + "/",
                    "-o",
                    result_tmp.name,
                    fasta_tmp.name,
                ],
                check=True,
            )
            rows = []
            current_query = None
            for line in result_tmp:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                first = parts[0]
                if first.startswith("seq") and "/" not in first:
                    # Header line for a new query
                    current_query = first
                else:
                    # Hit line: extract accession from the squeakr filename
                    accession = Path(first).name.split(".")[0]
                    count = int(parts[1])
                    rows.append((current_query, accession, count))

            df = pl.DataFrame(
                rows,
                schema={
                    "query_index": pl.Utf8,
                    "accession": pl.Utf8,
                    "score": pl.Float64,
                },
                orient="row",
            )
        df = df.with_columns(pl.col("query_index").str.slice(3).str.to_integer())
        df = (
            df.with_columns(pl.struct("accession", "score").alias("result"))
            .group_by("query_index")
            .agg(pl.col("result").alias("results"))
        )
        df = queries.join(
            df, left_on="index", right_on="query_index", how="left"
        ).select("query_id", "results")
        df = df.with_columns(
            pl.col("results").fill_null(pl.lit([], dtype=df.schema["results"]))
        )
        assert len(df) == len(queries)
        return df

    def indexed_accessions(self) -> list[Path]:
        with open(self.manifest_path) as f:
            contig_paths = [Path(line.strip()) for line in f if line.strip()]
        return contig_paths

    def save(self, output_path: Path):
        pass
