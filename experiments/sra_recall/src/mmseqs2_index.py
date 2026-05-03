import subprocess
import tempfile
from pathlib import Path

import polars as pl
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .base_index import BaseIndex
from .config import ExperimentConfig, MMseqs2Config


class MMseqs2Index(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, MMseqs2Config)
        self.model_cfg: MMseqs2Config = cfg.model
        self.mmseqs_exe: str = self.model_cfg.executable
        self.max_seqs = self.model_cfg.max_seqs
        self.index_prefix_path: Path | None = None
        print(f"Running mmseqs with executable = {self.mmseqs_exe}")

    def load(self, index_path: Path):
        self.index_prefix_path = index_path
        self.tmp_dir = index_path / "tmp"
        self.databases_path = index_path / "databases"
        self.databases = [
            p / "targetDB" for p in self.databases_path.iterdir() if p.is_dir()
        ]

    def build(self, accessions: list[Path], index_path: Path):
        index_path: Path = index_path.resolve()
        index_path.mkdir(exist_ok=True, parents=True)
        self.tmp_dir = index_path / "tmp"
        self.databases_path = index_path / "databases"
        self.databases_path.mkdir(exist_ok=True, parents=True)

        self.databases = []
        for acc_path in accessions:
            acc = acc_path.name.split(".")[0]
            db_dir = self.databases_path / acc
            db_dir.mkdir(exist_ok=True, parents=True)
            db_prefix = db_dir / "targetDB"
            subprocess.run(
                [self.mmseqs_exe, "createdb", str(acc_path), str(db_prefix)], check=True
            )
            subprocess.run(
                [
                    self.mmseqs_exe,
                    "createindex",
                    str(db_prefix),
                    str(self.tmp_dir),
                    "--search-type",
                    "3",
                ],
                check=True,
            )
            self.databases.append(db_prefix)

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
        all_results = []
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".fa", delete=True
        ) as fasta_tmp:
            # Write queries as FASTA
            SeqIO.write(records, fasta_tmp, "fasta")
            fasta_tmp.flush()
            for database in self.databases:
                with tempfile.NamedTemporaryFile(
                    mode="r", suffix=".m8", delete=True
                ) as result_tmp:
                    subprocess.run(
                        [
                            self.mmseqs_exe,
                            "easy-search",
                            fasta_tmp.name,
                            str(database),
                            result_tmp.name,
                            self.tmp_dir,
                            "--format-output",
                            "query,target,bits",
                        ],
                        check=True,
                    )
                    if Path(result_tmp.name).stat().st_size == 0:
                        continue
                    results_df = pl.read_csv(
                        result_tmp.name,
                        separator="\t",
                        schema=pl.Schema(
                            {
                                "query_id": pl.String,
                                "contig_id": pl.String,
                                "bit_score": pl.Float64,
                            }
                        ),
                    )
                    results_df = (
                        results_df.group_by("query_id")
                        .agg(
                            pl.col("contig_id", "bit_score").get(
                                pl.col("bit_score").arg_max()
                            )
                        )
                        .sort(["bit_score", "query_id"], descending=True)
                    )
                    results_df = results_df.with_columns(
                        pl.col("contig_id")
                        .str.split("_")
                        .list.get(0)
                        .alias("accession")
                    ).rename({"bit_score": "score"})
                    all_results.append(results_df)
        df = pl.concat(all_results).drop("contig_id")
        df = (
            df.with_columns(pl.struct("accession", "score").alias("result"))
            .group_by("query_id")
            .agg(pl.col("result").alias("results"))
        )
        df = queries.join(df, on="query_id", how="left").select("query_id", "results")
        df = df.with_columns(
            pl.col("results").fill_null(pl.lit([], dtype=df.schema["results"]))
        )
        assert len(df) == len(queries)
        return df

    def indexed_accessions(self) -> list[Path]:
        pass

    def save(self, output_path: Path):
        pass

    def index_size_gb(self, index_path: Path):
        total = sum(
            p.stat().st_size
            for p in (index_path / "databases").rglob("*")
            if p.is_file()
        )
        return total / (1024**3)
