"""List the contigs behind every saved top-k hit of one benchmark run.

Read the run's hits (<results_path>/hits/mut*.parquet: one row per query with
[{accession, score, vector_id}, ...]) and map each vector id back to the
contig the index cut it from. A vector id is an fbin row; meta.parquet gives
its accession, and walking that accession's FASTA in build order with the
index's chunking rule gives the contig.

Output: one parquet with a row per distinct contig: accession, contig_id.
Whether a contig is a true hit or a wrong one is left to the consumer, which
has the truth.

Reads the dataset's FASTAs once; run sra500-size datasets on a CPU node.

  uv run python scripts/hit_contigs.py \\
      --bench_config configs/ablation_sweep/data_reference.yaml \\
      --out /pscratch/sd/r/rsynk/locale-data/constructed/sra50/hit_contigs.parquet
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import polars as pl
from Bio import SeqIO
from jsonargparse import CLI, auto_cli

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmark/, for `src`

from run_benchmark import results_path  # noqa: E402
from src.config import DONE_FILE, DenseMethod, ExperimentConfig, read_config  # noqa: E402
from src.dense_index import _chunks_for_length  # noqa: E402


def load_run(bench_config: str) -> tuple[ExperimentConfig, tuple]:
    """The run's config and its index's chunking (W, step, end_tail, min_len)."""
    cfg = CLI(ExperimentConfig, as_positional=False, args=["--config", bench_config])
    if not isinstance(cfg.model, DenseMethod):
        raise ValueError(f"{bench_config}: not a dense method")
    index_path = cfg.model.index_path(cfg.index_dir)
    if not (index_path / DONE_FILE).exists():
        raise FileNotFoundError(f"{index_path} is not .done")
    ident = read_config(index_path)
    w = ident["max_seq_len"]
    step = w - ident["chunk_overlap"]
    return cfg, (w, step, ident.get("end_window_min_tail_bp"), ident.get("min_seq_len"))


def hit_windows(cfg: ExperimentConfig) -> dict[str, list[int]]:
    """{accession: sorted local fbin rows} of every distinct window in the run's hits."""
    index_path = cfg.model.index_path(cfg.index_dir)
    files = sorted((results_path(cfg) / "hits").glob("mut*.parquet"))
    if not files:
        raise FileNotFoundError(f"no hits under {results_path(cfg) / 'hits'}")
    meta = pl.read_parquet(index_path / "meta.parquet").rename({"srr_id": "accession"})
    hits = (
        pl.concat(pl.read_parquet(f).select("hits") for f in files)
        .explode("hits")
        .unnest("hits")
        .join(meta, on="accession", how="left")
        .with_columns((pl.col("vector_id") - pl.col("start_row")).alias("local_row"))
    )
    bad = hits.filter(
        (pl.col("local_row") < 0) | (pl.col("local_row") >= pl.col("num_rows"))
    )
    if bad.height:
        raise ValueError(f"{bad.height} hits outside their accession's rows")
    print(f"{hits.height:,} hits")
    return {
        a: sorted(set(r))
        for a, r in hits.group_by("accession").agg("local_row").iter_rows()
    }


def replay_accession(chunking: tuple, path: str, wanted: list[int]) -> set[str]:
    """Contig ids of the sorted wanted rows, walking the FASTA in fbin row
    order and counting windows per contig with the index's rule."""
    found, end, j = set(), 0, 0
    for rec in SeqIO.parse(path, "fasta"):
        end += _chunks_for_length(len(rec.seq), *chunking)
        while j < len(wanted) and wanted[j] < end:
            found.add(str(rec.id))
            j += 1
        if j == len(wanted):
            break
    if j < len(wanted):
        raise RuntimeError(f"{path}: row {wanted[j]} beyond the {end} replayed")
    return found


def main(bench_config: str, out: str):
    """
    Args:
        bench_config: run_benchmark yaml of a dense run with saved hits.
        out: output parquet (accession, contig_id).
    """
    cfg, chunking = load_run(bench_config)
    wanted = hit_windows(cfg)
    accs = sorted(wanted)
    paths = [
        str(Path(cfg.dataset_dir) / "logan_accessions" / a / f"{a}.contigs.fa")
        for a in accs
    ]
    print(f"replaying {len(accs)} accessions with chunking {chunking}")
    with ProcessPoolExecutor() as ex:
        found = list(
            ex.map(
                partial(replay_accession, chunking), paths, [wanted[a] for a in accs]
            )
        )
    table = pl.DataFrame(
        {
            "accession": [a for a, f in zip(accs, found) for _ in f],
            "contig_id": [c for f in found for c in sorted(f)],
        }
    )
    table.write_parquet(out)
    print(f"{table.height:,} contigs; wrote {out}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
