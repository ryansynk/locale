import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
from metagraph.client import GraphClient

from .base_index import BaseIndex
from .config import ExperimentConfig, MetagraphConfig


class MetagraphIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, MetagraphConfig)
        self.port: int = get_free_port()
        self.graph_path: Path | None = None
        self.annotation_path: Path | None = None
        self.model_cfg: MetagraphConfig = cfg.model
        self.executable: str = self.model_cfg.executable
        self.k = self.model_cfg.k
        print(f"Running metagraph with executable = {self.executable}")

    def load(self, index_path: Path):
        self.manifest_path = index_path / "contig_manifest.txt"
        self.graph_path = index_path / "graph_primary.dbg"
        self.annotation_path = index_path / "annotation.relaxed.row_diff_brwt.annodbg"

        base_metagraph_cmd = shlex.split(self.executable)
        full_cmd = base_metagraph_cmd + [
            "server_query",
            "-i",
            str(self.graph_path),
            "-a",
            str(self.annotation_path),
            "--port",
            str(self.port),
        ]

        self.query_server_proc = subprocess.Popen(full_cmd)
        # Good ol reliable sleep
        # Kludge -- I need to wait for server to start but don't want to figure out how
        # to communicate with metagraph process to see if its ready
        time.sleep(5)
        self.graph_client = GraphClient("127.0.0.1", self.port, api_path="")

    def build(self, accessions: list[Path], index_path: Path):
        index_path: Path = index_path.resolve()
        index_path.mkdir(exist_ok=True, parents=True)
        bash_script_path = Path(__file__).parent / "build_metagraph.sh"
        # os.cpu_count() reports the whole node, not the cgroup, so under a
        # partial SLURM allocation it oversubscribes -- build_metagraph.sh
        # divides this by 8 to size annotate's parallelism.
        num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or os.cpu_count()
        if not Path(bash_script_path).is_file():
            print(
                f"Error: Bash script not found at {bash_script_path}", file=sys.stderr
            )
            sys.exit(1)

        manifest_path = index_path / "contig_manifest.txt"
        with open(manifest_path, "w") as f:
            for path in accessions:
                f.write(f"{str(path)}\n")

        command = [
            "bash",
            str(bash_script_path),
            str(self.model_cfg.executable),
            str(self.k),
            str(num_threads),
            str(manifest_path),
            str(index_path),
        ]

        print(f"Executing command: {' '.join(command)}")

        try:
            process = subprocess.Popen(
                command, stdout=sys.stdout, stderr=sys.stderr, text=True
            )
            process.wait()

            if process.returncode == 0:
                print("\nPipeline completed successfully.")
            else:
                print(
                    f"\nPipeline failed with exit code {process.returncode}.",
                    file=sys.stderr,
                )
                sys.exit(process.returncode)

        except Exception as e:
            print(f"An error occurred while executing the script: {e}", file=sys.stderr)
            sys.exit(1)

        self.manifest_path = manifest_path
        self.graph_path = index_path / "graph_primary.dbg"
        self.annotation_path = index_path / "annotation.relaxed.row_diff_brwt.annodbg"
        self.load(index_path)

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        queries = queries.with_row_index()
        queries = queries.with_columns(
            pl.col("query_sequence").str.len_chars().alias("query_length")
        )
        results = self.graph_client.search(queries["query_sequence"].to_list())
        df = pl.from_pandas(results)
        df = df.with_columns(
            pl.col("sample").str.split("/").list.get(-2).alias("accession")
        ).drop("sample")

        df = (
            df.rename({"kmer_count": "score"})
            .with_columns(pl.col("score").cast(pl.Float64))
            .select(
                pl.col("seq_description"),
                pl.struct("accession", "score").alias("result"),
            )
            .group_by("seq_description")
            .agg(pl.col("result").alias("results"))
        )
        df = df.with_columns(pl.col("seq_description").cast(pl.Int32))
        df = queries.join(
            df, left_on="index", right_on="seq_description", how="left"
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

    def index_size_gb(self, index_path: Path):
        manifest_gb = (index_path / "contig_manifest.txt").stat().st_size / (1024**3)
        graph_gb = (index_path / "graph_primary.dbg").stat().st_size / (1024**3)
        annotation_gb = (
            index_path / "annotation.relaxed.row_diff_brwt.annodbg"
        ).stat().st_size / (1024**3)
        return manifest_gb + graph_gb + annotation_gb


def get_free_port():
    # Create a new socket using IPv4 and TCP
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind to all interfaces ('') on port 0.
        # The OS will automatically find an available port.
        s.bind(("", 0))

        # Retrieve the assigned port number
        port = s.getsockname()[1]

    return port
