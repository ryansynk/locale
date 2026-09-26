"""Metagraph k-mer baseline: built by build_metagraph.sh, queried via server_query.

Multi-node: run_benchmark builds one complete metagraph index per node over
that node's stripe of accessions (index_path/shard_<r>/) and merge_shards
writes graphs.csv listing every shard under one name. server_query queries all
rows sharing a name together and appends their results. That is exact here:
an accession's k-mer count for a query does not depend on which other
accessions share its graph, and each accession lives in exactly one shard.
The union is cut back to TOP_LABELS per query so a sharded index returns what
one joint index would.
"""

import atexit
import os
import shlex
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests
from metagraph.client import GraphClient

from .base_index import BaseIndex
from .config import ExperimentConfig, MetagraphConfig

GRAPH_FILE = "graph_primary.dbg"
# row_diff annotations load these two files next to the graph at query time.
GRAPH_SIDECARS = (".anchors", ".rd_succ")
ANNOTATION_FILE = "annotation.relaxed.row_diff_brwt.annodbg"
MANIFEST_FILE = "contig_manifest.txt"
# server_query index list, one '<name>,<graph>,<annotation>' row per shard.
GRAPHS_CSV = "graphs.csv"
CSV_INDEX_NAME = "all"
# Labels requested per query. Every shard answers with its own top TOP_LABELS
# by k-mer count, so the global top TOP_LABELS is a subset of the union.
TOP_LABELS = 100
# The Python client default: report every label sharing at least one k-mer.
DISCOVERY_FRACTION = 0.0

RESULTS_DTYPE = pl.List(pl.Struct({"accession": pl.String, "score": pl.Float64}))


class MetagraphIndex(BaseIndex):
    server_parallel: int = 1  # default for instances built without __init__ (tests)

    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, MetagraphConfig)
        self.port: int = get_free_port()
        self.model_cfg: MetagraphConfig = cfg.model
        self.executable: str = self.model_cfg.executable
        self.k = self.model_cfg.k
        self.server_parallel: int = self.model_cfg.server_parallel
        self.index_path: Path | None = None
        self.manifest_path: Path | None = None
        self.query_server_proc: subprocess.Popen | None = None
        self.graph_client: GraphClient | None = None
        print(f"Running metagraph with executable = {self.executable}")

    # ------------------------------------------------------------------ layout
    @staticmethod
    def index_files(index_path: Path) -> list[tuple[Path, Path]]:
        """(graph, annotation) pairs load() serves: every graphs.csv row, or
        the single pair of a one-node build."""
        csv_path = index_path / GRAPHS_CSV
        if csv_path.exists():
            pairs = []
            for line in csv_path.read_text().splitlines():
                if not line.strip():
                    continue
                _name, graph, anno = line.split(",")
                pairs.append((Path(graph), Path(anno)))
            return pairs
        return [(index_path / GRAPH_FILE, index_path / ANNOTATION_FILE)]

    @staticmethod
    def _required_files(index_path: Path) -> list[Path]:
        files = [index_path / MANIFEST_FILE]
        for graph, anno in MetagraphIndex.index_files(index_path):
            files.append(graph)
            files.extend(graph.with_name(graph.name + ext) for ext in GRAPH_SIDECARS)
            files.append(anno)
        return files

    @staticmethod
    def merge_shards(index_path: Path, num_nodes: int):
        """Point one query server at every shard and join the manifests.

        Nothing is copied: graphs.csv references the shard files in place, so
        the shards must stay where they are for as long as the index is used.
        """
        index_path = index_path.resolve()
        rows: list[str] = []
        accessions: list[str] = []
        for rank in range(num_nodes):
            shard = index_path / f"shard_{rank}"
            graph, anno = shard / GRAPH_FILE, shard / ANNOTATION_FILE
            missing = [
                str(p) for p in MetagraphIndex._required_files(shard) if not p.exists()
            ]
            if missing:
                raise FileNotFoundError(f"shard_{rank} is incomplete: {missing}")
            rows.append(f"{CSV_INDEX_NAME},{graph},{anno}")
            accessions.extend(
                line.strip()
                for line in (shard / MANIFEST_FILE).read_text().splitlines()
                if line.strip()
            )
        if len(set(accessions)) != len(accessions):
            raise ValueError("the same contig file appears in more than one shard")
        # Write both files whole, then rename, so a reader never sees a partial
        # index list.
        for name, text in (
            (MANIFEST_FILE, "".join(f"{a}\n" for a in sorted(accessions))),
            (GRAPHS_CSV, "".join(f"{r}\n" for r in rows)),
        ):
            tmp = index_path / (name + ".tmp")
            tmp.write_text(text)
            tmp.replace(index_path / name)
        print(f"[merge] {num_nodes} metagraph shards listed in {index_path / GRAPHS_CSV}")

    # ------------------------------------------------------------------- build
    def build(self, accessions: list[Path], index_path: Path):
        index_path = index_path.resolve()
        index_path.mkdir(exist_ok=True, parents=True)
        bash_script_path = Path(__file__).parent / "build_metagraph.sh"
        if not bash_script_path.is_file():
            print(f"Error: Bash script not found at {bash_script_path}", file=sys.stderr)
            sys.exit(1)
        # os.cpu_count() reports the whole node, not the cgroup, so under a
        # partial SLURM allocation it oversubscribes -- build_metagraph.sh
        # divides this by 8 to size annotate's parallelism.
        num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or os.cpu_count()

        manifest_path = index_path / MANIFEST_FILE
        with open(manifest_path, "w") as f:
            for path in accessions:
                f.write(f"{str(path)}\n")

        command = [
            "bash",
            str(bash_script_path),
            str(self.executable),
            str(self.k),
            str(num_threads),
            str(manifest_path),
            str(index_path),
            str(node_memory_gb()),
        ]
        print(f"Executing command: {' '.join(command)}")
        try:
            process = subprocess.Popen(
                command, stdout=sys.stdout, stderr=sys.stderr, text=True
            )
            process.wait()
        except Exception as e:
            print(f"An error occurred while executing the script: {e}", file=sys.stderr)
            sys.exit(1)
        if process.returncode != 0:
            print(
                f"\nPipeline failed with exit code {process.returncode}.",
                file=sys.stderr,
            )
            sys.exit(process.returncode)
        print("\nPipeline completed successfully.")
        # No load() here: run_benchmark calls load() on the index it will
        # search, which for a multi-node build is the merged one, not this
        # node's shard.
        self.manifest_path = manifest_path

    def save(self, output_path: Path):
        pass

    # -------------------------------------------------------------------- load
    def load(self, index_path: Path):
        index_path = index_path.resolve()
        self.index_path = index_path
        self.manifest_path = index_path / MANIFEST_FILE
        missing = [str(p) for p in self._required_files(index_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"metagraph index under {index_path} is missing {missing}")

        cmd = shlex.split(self.executable) + ["server_query"]
        if (index_path / GRAPHS_CSV).exists():
            cmd.append(str(index_path / GRAPHS_CSV))
        else:
            graph, anno = self.index_files(index_path)[0]
            cmd += ["-i", str(graph), "-a", str(anno)]
        cmd += ["--port", str(self.port)]
        if self.server_parallel > 1:
            cmd += ["-p", str(self.server_parallel)]
        print(f"Starting metagraph query server: {' '.join(cmd)}")
        self.query_server_proc = subprocess.Popen(cmd)
        atexit.register(self.stop_server)
        self.graph_client = GraphClient("127.0.0.1", self.port, api_path="")
        # server_query answers only once every graph and annotation is in RAM;
        # loading is roughly linear in index bytes and a minute per GB is
        # generous for Lustre.
        timeout = 600 + 60 * self.index_size_gb(index_path)
        wait_for_server(self.query_server_proc, self.graph_client, self.port, timeout)

    def stop_server(self):
        proc = self.query_server_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    # ------------------------------------------------------------------ search
    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        queries = queries.with_row_index()
        seqs = queries["query_sequence"].to_list()
        n_req = max(1, min(self.server_parallel, len(seqs)))
        if n_req == 1:
            results = [self.graph_client.search(
                seqs, top_labels=TOP_LABELS, discovery_fraction=DISCOVERY_FRACTION
            )]
        else:
            # Contiguous slices, one request each; seq_description is the
            # position within a request, so shift it back to the batch.
            bounds = [len(seqs) * r // n_req for r in range(n_req + 1)]

            def _one(r):
                c = GraphClient("127.0.0.1", self.port, api_path="")
                out = c.search(
                    seqs[bounds[r] : bounds[r + 1]],
                    top_labels=TOP_LABELS,
                    discovery_fraction=DISCOVERY_FRACTION,
                )
                if len(out):
                    out["seq_description"] = (
                        out["seq_description"].astype(int) + bounds[r]
                    ).astype(str)
                return out

            with ThreadPoolExecutor(max_workers=n_req) as pool:
                results = list(pool.map(_one, range(n_req)))
        results = [r for r in results if len(r)]
        df = pl.concat([pl.from_pandas(r) for r in results]) if results else pl.DataFrame()
        if len(df) == 0:
            return queries.select(
                "query_id", pl.lit([], dtype=RESULTS_DTYPE).alias("results")
            )
        # --anno-filename labels are the contig paths, laid out as
        # <accessions_dir>/<accession>/<accession>.contigs.fa.
        df = df.with_columns(
            pl.col("sample").str.split("/").list.get(-2).alias("accession"),
            pl.col("kmer_count").cast(pl.Float64).alias("score"),
            pl.col("seq_description").cast(pl.Int32),
        )
        # A sharded server appends each shard's top TOP_LABELS without
        # re-ranking; sort and cut so the list matches a single joint index.
        df = df.group_by("seq_description").agg(
            pl.struct("accession", "score")
            .sort_by("score", descending=True)
            .head(TOP_LABELS)
            .alias("results")
        )
        df = queries.join(
            df, left_on="index", right_on="seq_description", how="left"
        ).select("query_id", "results")
        df = df.with_columns(pl.col("results").fill_null(pl.lit([], dtype=RESULTS_DTYPE)))
        assert len(df) == len(queries)
        return df

    def indexed_accessions(self) -> list[Path]:
        with open(self.manifest_path) as f:
            return [Path(line.strip()) for line in f if line.strip()]

    def index_size_gb(self, index_path: Path):
        # Everything the query server reads: manifest, every shard's graph
        # with its row_diff sidecars, and its annotation.
        total = sum(p.stat().st_size for p in self._required_files(index_path))
        return total / (1024**3)


def node_memory_gb() -> int:
    """RAM budget handed to build_metagraph.sh: the SLURM per-node limit when
    set, else physical RAM. The script keeps metagraph's caps below it."""
    slurm_mb = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mb:
        return max(1, int(slurm_mb) // 1024)
    return max(1, os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024**3))


def wait_for_server(
    proc: subprocess.Popen,
    client: GraphClient,
    port: int,
    timeout: float,
    poll_interval: float = 5.0,
):
    """Block until server_query answers a probe search, or raise.

    Replaces a fixed sleep: a large index takes minutes to load, and the first
    real request would otherwise fail on a closed port.
    """
    start = time.time()
    last_error = "no connection attempt yet"
    next_report = 60.0
    while True:
        if proc.poll() is not None:
            raise RuntimeError(
                f"metagraph server_query exited with code {proc.returncode} before serving"
            )
        elapsed = time.time() - start
        if elapsed > timeout:
            proc.terminate()
            raise TimeoutError(
                f"metagraph server_query on port {port} not ready after {timeout:.0f}s "
                f"(last error: {last_error})"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=poll_interval):
                pass
            client.search("ACGT" * 16, top_labels=1)
            print(f"metagraph query server ready after {elapsed:.0f}s")
            return
        except (OSError, requests.RequestException, RuntimeError, ValueError) as e:
            last_error = f"{type(e).__name__}: {e}"
        if elapsed >= next_report:
            print(f"waiting for metagraph query server ({elapsed:.0f}s): {last_error}")
            next_report += 60.0
        time.sleep(poll_interval)


def get_free_port():
    # Bind port 0 on all interfaces so the OS picks a free port, then release it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
