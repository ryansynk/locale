"""MMseqs2 alignment baseline over ONE database holding every accession's contigs.

Logan contig ids are '<accession>_<n>', so a hit's accession is its target id
up to the first underscore and the contig files can be fed to createdb as they
are. An earlier version built one database per accession and searched them one
after another; createindex's k-mer lookup table is ~9 GB whatever the database
holds, so 50 accessions reported 440 GB of "index" and every query batch ran 50
searches. One database gives one index, one search per batch, and an index
size that reflects the data.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .base_index import BaseIndex
from .config import ExperimentConfig, MMseqs2Config

DB_DIR = "db"
DB_PREFIX = "targetDB"
MANIFEST_FILE = "contig_manifest.txt"
RESULTS_DTYPE = pl.List(pl.Struct({"accession": pl.String, "score": pl.Float64}))


class MMseqs2Index(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, MMseqs2Config)
        self.model_cfg: MMseqs2Config = cfg.model
        self.mmseqs_exe: str = self.model_cfg.executable
        self.max_seqs = self.model_cfg.max_seqs
        self.index_path: Path | None = None
        self.db_prefix: Path | None = None
        self.manifest_path: Path | None = None
        self.tmp_dir: Path | None = None
        print(f"Running mmseqs with executable = {self.mmseqs_exe}")

    def _threads_args(self) -> list[str]:
        # mmseqs defaults to every hardware thread, which oversubscribes a
        # partial SLURM allocation; honour the allocation when there is one.
        n = os.environ.get("SLURM_CPUS_PER_TASK")
        return ["--threads", n] if n else []

    def _set_paths(self, index_path: Path):
        self.index_path = index_path.resolve()
        self.db_prefix = self.index_path / DB_DIR / DB_PREFIX
        self.manifest_path = self.index_path / MANIFEST_FILE
        self.tmp_dir = self.index_path / "tmp"

    def load(self, index_path: Path):
        self._set_paths(index_path)
        for required in (
            self.db_prefix,
            self.db_prefix.with_suffix(".index"),
            self.manifest_path,
        ):
            if not required.exists():
                raise FileNotFoundError(
                    f"mmseqs index under {self.index_path} is missing {required}"
                )

    def build(self, accessions: list[Path], index_path: Path):
        self._set_paths(index_path)
        self.db_prefix.parent.mkdir(exist_ok=True, parents=True)
        self.tmp_dir.mkdir(exist_ok=True)
        with open(self.manifest_path, "w") as f:
            for path in accessions:
                f.write(f"{path}\n")
        # One sequence database over every contig file; ids keep the
        # '<accession>_<n>' form the search relies on.
        subprocess.run(
            [self.mmseqs_exe, "createdb", *map(str, accessions), str(self.db_prefix)],
            check=True,
        )
        # One precomputed k-mer index for the nucleotide search.
        subprocess.run(
            [
                self.mmseqs_exe,
                "createindex",
                str(self.db_prefix),
                str(self.tmp_dir),
                "--search-type",
                "3",
                *self._threads_args(),
            ],
            check=True,
        )
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        records = [
            SeqRecord(Seq(seq), id=qid, description="")
            for qid, seq in zip(
                queries["query_id"].to_list(), queries["query_sequence"].to_list()
            )
        ]
        self.tmp_dir.mkdir(exist_ok=True)
        with (
            tempfile.NamedTemporaryFile(mode="w", suffix=".fa") as fasta_tmp,
            tempfile.NamedTemporaryFile(mode="r", suffix=".m8") as result_tmp,
        ):
            SeqIO.write(records, fasta_tmp, "fasta")
            fasta_tmp.flush()
            subprocess.run(
                [
                    self.mmseqs_exe,
                    "easy-search",
                    fasta_tmp.name,
                    str(self.db_prefix),
                    result_tmp.name,
                    str(self.tmp_dir),
                    "--search-type",
                    "3",
                    "--max-seqs",
                    str(self.max_seqs),
                    "--format-output",
                    "query,target,bits",
                    *self._threads_args(),
                ],
                check=True,
            )
            if Path(result_tmp.name).stat().st_size == 0:
                return queries.select(
                    "query_id", pl.lit([], dtype=RESULTS_DTYPE).alias("results")
                )
            hits = pl.read_csv(
                result_tmp.name,
                separator="\t",
                has_header=False,
                schema=pl.Schema(
                    {"query_id": pl.String, "contig_id": pl.String, "score": pl.Float64}
                ),
            )
        # Best-scoring contig per (query, accession), accessions score-descending.
        df = (
            hits.with_columns(
                pl.col("contig_id").str.split("_").list.get(0).alias("accession")
            )
            .group_by("query_id", "accession")
            .agg(pl.col("score").max())
            .group_by("query_id")
            .agg(
                pl.struct("accession", "score")
                .sort_by("score", descending=True)
                .alias("results")
            )
        )
        df = queries.join(df, on="query_id", how="left").select("query_id", "results")
        df = df.with_columns(
            pl.col("results").fill_null(pl.lit([], dtype=RESULTS_DTYPE))
        )
        assert len(df) == len(queries)
        return df

    def indexed_accessions(self) -> list[Path]:
        with open(self.manifest_path) as f:
            return [Path(line.strip()) for line in f if line.strip()]

    def save(self, output_path: Path):
        pass

    def index_size_gb(self, index_path: Path):
        db_dir = index_path / DB_DIR
        total = sum(p.stat().st_size for p in db_dir.rglob("*") if p.is_file())
        return total / (1024**3)
