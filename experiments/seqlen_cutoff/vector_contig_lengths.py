"""Map every vector of a dense index to the length of the contig it came from.

The fbin row order is deterministic: accessions in meta.parquet order, records
in FASTA order, and per record the chunks benchmark/src/dense_index.py's
_iter_chunks yields (one whole chunk for a contig at most max_seq_len long,
else the strided windows plus the optional end-aligned window). So the map is
rebuilt here from the FASTAs and the index's config.json, without re-embedding,
and validated two ways:

  * per accession, the chunk count must equal meta.parquet's num_rows (this is
    exactly the count build() preallocated for it, and build() writes rows in
    submission order, so the sum also matches the fbin header);
  * optionally (--spot_check N) N random rows are re-embedded from the
    reconstructed chunk strings and compared against the fbin rows.

Output: a little-endian uint32 array with one entry per fbin row, the length
in bp of the record that row's chunk was cut from (so every strided or end
window of a long contig carries that contig's full length, and a cutoff c
removes "every vector of a contig shorter than c"), plus a JSON sidecar with
the index identity, the row count and the fraction of rows below a few
cutoffs.

    cd benchmark
    uv run python ../experiments/seqlen_cutoff/vector_contig_lengths.py \
        --index_path indexes/sra500/locale@8vqiabk9-endwin50 \
        --dataset_dir /pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle \
        --output ../experiments/seqlen_cutoff/sra500_endwin50_contig_len.u32

Reading every contig of a 500-accession set is a CPU job: run it on a compute
node, not a login node. --limit N processes only the first N accessions in
meta order (smoke test; the output then covers only those rows).
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli
from tqdm import tqdm

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

from src.config import PAPER_CHECKPOINT, EncoderConfig, read_config  # noqa: E402
from src.dense_index import _chunks_for_length, chunk_sequence  # noqa: E402
from src.fbin import _load_fbin_mmap  # noqa: E402

REPORT_CUTOFFS = (45, 50, 64, 77, 100, 128, 150, 200, 256, 500)


def _iter_fasta_lengths(fasta_path: Path):
    """Yield each record's sequence length, by plain text scan (SeqIO is too
    slow for Logan files with millions of records). Mirrors
    dense_index._count_file_chunks: line.strip() per sequence line."""
    seq_len = 0
    in_record = False
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if in_record:
                    yield seq_len
                in_record = True
                seq_len = 0
            else:
                seq_len += len(line.strip())
        if in_record:
            yield seq_len


def _iter_fasta_records(fasta_path: Path):
    """Yield each record's sequence string (same parser as above)."""
    parts: list[str] = []
    in_record = False
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if in_record:
                    yield "".join(parts)
                in_record = True
                parts = []
            else:
                parts.append(line.strip())
        if in_record:
            yield "".join(parts)


def _accession_fasta(dataset_dir: Path, srr_id: str) -> Path:
    """The one *.contigs.fa run_benchmark indexes for this accession: it
    globs dataset_dir/logan_accessions/**/*.contigs.fa and takes the parent
    directory's name as the accession."""
    candidates = sorted(
        (dataset_dir / "logan_accessions" / srr_id).glob("*.contigs.fa")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one *.contigs.fa for {srr_id}, found {candidates}"
        )
    return candidates[0]


def _accession_lengths(
    fasta_path: str,
    chunk_size: int,
    step_size: int,
    end_window_min_tail_bp: int | None,
    expected_rows: int,
) -> np.ndarray:
    """uint32 contig length per fbin row of one accession, in row order."""
    out = np.empty(expected_rows, dtype=np.uint32)
    pos = 0
    for seq_len in _iter_fasta_lengths(Path(fasta_path)):
        n = _chunks_for_length(seq_len, chunk_size, step_size, end_window_min_tail_bp)
        if pos + n > expected_rows:
            raise ValueError(
                f"{fasta_path}: chunk count exceeds meta.parquet num_rows "
                f"{expected_rows} (chunking config does not match the index?)"
            )
        out[pos : pos + n] = seq_len
        pos += n
    if pos != expected_rows:
        raise ValueError(
            f"{fasta_path}: {pos} chunks but meta.parquet says {expected_rows} rows"
        )
    return out


def _chunk_strings_for_rows(
    fasta_path: Path,
    local_rows: list[int],
    chunk_size: int,
    chunk_overlap: int,
    end_window_min_tail_bp: int | None,
) -> dict[int, str]:
    """The exact chunk string _iter_chunks yielded for each local row index
    (row offset within the accession)."""
    wanted = sorted(set(local_rows))
    found: dict[int, str] = {}
    step_size = chunk_size - chunk_overlap
    pos = 0
    wi = 0
    for seq in _iter_fasta_records(fasta_path):
        n = _chunks_for_length(len(seq), chunk_size, step_size, end_window_min_tail_bp)
        while wi < len(wanted) and wanted[wi] < pos + n:
            if len(seq) <= chunk_size:
                chunks = [seq]
            else:
                chunks = chunk_sequence(
                    seq, "", chunk_size, chunk_overlap, "stride", end_window_min_tail_bp
                )
            assert len(chunks) == n
            found[wanted[wi]] = chunks[wanted[wi] - pos]
            wi += 1
        if wi == len(wanted):
            break
        pos += n
    if len(found) != len(wanted):
        raise ValueError(f"{fasta_path}: rows {set(wanted) - found.keys()} not found")
    return found


def _spot_check(
    index_path: Path,
    dataset_dir: Path,
    meta: pl.DataFrame,
    identity: dict,
    n: int,
    seed: int,
    checkpoint_path: str | None,
    device: str,
    min_cos: float,
) -> None:
    """Re-embed n random rows from their reconstructed chunks and compare."""
    import torch

    from src.encoders import DenseEncoder

    if identity["name"] == "locale" and checkpoint_path is None:
        if (identity["checkpoint"], identity["step"]) != (
            PAPER_CHECKPOINT["ckpt_id"],
            PAPER_CHECKPOINT["step"],
        ):
            raise ValueError(
                "index was built from a non-paper checkpoint; pass --checkpoint_path"
            )
    enc_cfg = EncoderConfig(
        name=identity["name"],
        checkpoint_path=checkpoint_path,
        pooling=identity["pooling"],
        max_seq_len=identity["max_seq_len"],
        chunk_overlap=identity["chunk_overlap"],
        end_window_min_tail_bp=identity.get("end_window_min_tail_bp"),
        device=device,
    )
    starts = meta["start_row"].to_numpy()
    counts = meta["num_rows"].to_numpy()
    total = int(starts[-1] + counts[-1])
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(total, size=min(n, total), replace=False))
    acc_idx = np.searchsorted(starts, rows, side="right") - 1

    chunks: dict[int, str] = {}
    for ai in np.unique(acc_idx):
        srr_id = meta["srr_id"][int(ai)]
        local = [int(r - starts[ai]) for r in rows[acc_idx == ai]]
        strings = _chunk_strings_for_rows(
            _accession_fasta(dataset_dir, srr_id),
            local,
            enc_cfg.max_seq_len,
            enc_cfg.chunk_overlap,
            enc_cfg.end_window_min_tail_bp,
        )
        for lr, s in strings.items():
            chunks[int(starts[ai] + lr)] = s

    encoder = DenseEncoder(enc_cfg)
    seqs = [chunks[int(r)] for r in rows]
    fresh = encoder.encode(seqs).float().cpu().numpy()
    stored = np.asarray(_load_fbin_mmap(index_path / "embeddings.fbin")[rows])
    cos = (fresh * stored).sum(1) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored, axis=1)
    )
    worst = float(cos.min())
    print(
        f"[spot check] {len(rows)} rows re-embedded on {device}: "
        f"cos min {worst:.6f} mean {cos.mean():.6f} "
        f"(chunk lengths {min(map(len, seqs))}-{max(map(len, seqs))} bp)"
    )
    if worst < min_cos:
        bad = rows[cos < min_cos]
        raise SystemExit(
            f"[FAIL] {len(bad)} rows below cos {min_cos}: rows {bad[:10].tolist()} "
            "-- the reconstructed chunk order does not match the fbin"
        )
    del torch


def main(
    index_path: str,
    dataset_dir: str,
    output: str,
    num_workers: int | None = None,
    limit: int | None = None,
    spot_check: int = 0,
    spot_check_seed: int = 0,
    checkpoint_path: str | None = None,
    device: str = "cuda",
    min_cos: float = 0.999,
):
    """
    Args:
        index_path: the encoder's index directory (config.json, meta.parquet,
            embeddings.fbin).
        dataset_dir: benchmark bundle holding logan_accessions/<acc>/*.contigs.fa.
        output: where the uint32 lengths go; <output>.json gets the sidecar.
        num_workers: FASTA-walking processes (default: all cores of this task).
        limit: only the first N accessions in meta.parquet order (smoke test).
        spot_check: re-embed this many random rows and compare with the fbin
            (needs the checkpoint; GPU unless --device cpu with a tiny N).
        checkpoint_path: for the spot check, when the index is not from the
            paper checkpoint.
        min_cos: spot-check tolerance (GPU float noise sits around 1e-6).
    """
    index_path = Path(index_path).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    output = Path(output).resolve()

    identity = read_config(index_path)
    if identity is None:
        raise FileNotFoundError(f"{index_path} has no config.json")
    chunk_size = identity["max_seq_len"]
    step_size = chunk_size - identity["chunk_overlap"]
    end_window = identity.get("end_window_min_tail_bp")
    if identity.get("min_seq_len") is not None:
        raise NotImplementedError(
            "index was built with a build-time length filter; this script "
            "reproduces unfiltered chunking only"
        )

    meta = pl.read_parquet(index_path / "meta.parquet")
    with open(index_path / "embeddings.fbin", "rb") as f:
        n_header, d = (int(x) for x in np.frombuffer(f.read(8), dtype=np.uint32))
    starts = meta["start_row"].to_numpy()
    counts = meta["num_rows"].to_numpy()
    if not (starts[0] == 0 and np.all(starts[1:] == starts[:-1] + counts[:-1])):
        raise ValueError("meta.parquet rows are not contiguous")
    if int(starts[-1] + counts[-1]) != n_header:
        raise ValueError(
            f"meta.parquet covers {int(starts[-1] + counts[-1])} rows, fbin header "
            f"says {n_header}: a build batch was dropped, the row order is not "
            "recoverable from the FASTAs"
        )
    print(f"{index_path.name}: {n_header:,} x {d} vectors, {len(meta)} accessions")

    work = meta if limit is None else meta.head(limit)
    n_rows = int(work["num_rows"].sum())
    output.parent.mkdir(parents=True, exist_ok=True)
    out = np.memmap(output, dtype="<u4", mode="w+", shape=(n_rows,))

    if num_workers is None:
        num_workers = len(os.sched_getaffinity(0))
    num_workers = max(1, min(num_workers, len(work)))
    ctx_jobs = []
    # Largest accessions first so one huge file does not serialize the tail.
    order = sorted(range(len(work)), key=lambda i: -int(counts[i]))
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for i in order:
            srr_id = work["srr_id"][i]
            fut = pool.submit(
                _accession_lengths,
                str(_accession_fasta(dataset_dir, srr_id)),
                chunk_size,
                step_size,
                end_window,
                int(counts[i]),
            )
            ctx_jobs.append((fut, i))
        by_future = {f: i for f, i in ctx_jobs}
        for fut in tqdm(
            as_completed(by_future), total=len(by_future), desc="Accessions"
        ):
            i = by_future[fut]
            arr = fut.result()
            s = int(starts[i])
            out[s : s + len(arr)] = arr
    out.flush()

    arr = np.asarray(out)
    fractions = {str(c): float((arr < c).mean()) for c in REPORT_CUTOFFS}
    sidecar = {
        "index_path": str(index_path),
        "index_identity": identity,
        "dtype": "<u4",
        "n_rows": n_rows,
        "n_rows_in_index": n_header,
        "accessions": len(work),
        "limit": limit,
        "fraction_rows_with_contig_len_below": fractions,
    }
    with open(output.with_suffix(output.suffix + ".json"), "w") as f:
        json.dump(sidecar, f, indent=1)
    print(f"wrote {n_rows:,} lengths -> {output}")
    print("fraction of rows from contigs shorter than cutoff:")
    for c, frac in fractions.items():
        print(f"  < {c:>4} bp: {frac:.3f}")

    if spot_check:
        _spot_check(
            index_path,
            dataset_dir,
            work,
            identity,
            spot_check,
            spot_check_seed,
            checkpoint_path,
            device,
            min_cos,
        )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
