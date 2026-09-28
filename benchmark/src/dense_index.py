"""DenseIndex: chunk accessions, embed them into embeddings.fbin + meta.parquet,
embed queries, and hand the query vectors to an index engine.

Everything after "I have query vectors" is the engine's (src/engines.py):
every engine reads the encoder's fbin and returns per-chunk (scores, ids);
this class unions a query's chunks
(positions and, with both_strands, strands) into its hits frame and regroups
hits to accessions (topk_regroup). The exhaustive reference protocol is the
one non-top-k path: it scores every accession through the engine's
slot_scores.
"""

import copy
import faulthandler
import multiprocessing as mp
import os
import sys
import threading
import traceback
import time
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    process,
)
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from Bio import SeqIO
from tqdm import tqdm

from .base_index import BaseIndex
from .config import DenseMethod, ExhaustiveIndex, ExperimentConfig
from .encoders import DenseEncoder, batched  # noqa: F401  (re-exported for tests)
from .engines import make_engine
from .fbin import _create_fbin_memmap, _load_fbin_mmap  # noqa: F401  (re-exported for tests)
from .topk_regroup import build_hits_frame, regroup_topk_hits

# IUPAC complement; case is preserved. Anything else (gaps, '*') maps to itself,
# which the encoders' own tokenizers already have to cope with in the forward
# strand.
_COMPLEMENT = str.maketrans(
    "ACGTUNRYSWKMBDHVacgtunryswkmbdhv", "TGCAANYRSWMKVHDBtgcaanyrswmkvhdb"
)


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _chunks_for_length(seq_len: int, chunk_size: int, step_size: int) -> int:
    """Chunk count _iter_chunks yields for one sequence, from its length alone.

    Mirrors the two branches exactly: sequences of at most chunk_size (including
    empty ones) pass through _iter_chunks whole as a single chunk; longer ones go
    through chunk_sequence's stride mode, which yields
    range(0, seq_len - chunk_size + 1, step_size) chunks.
    """
    if seq_len <= chunk_size:
        return 1
    return (seq_len - chunk_size) // step_size + 1


def _count_file_chunks(fasta_path: str, chunk_size: int, step_size: int) -> int:
    """Count the chunks one FASTA contributes without materializing any of them.

    Plain text scan instead of SeqIO: only per-record sequence lengths are
    needed, and Logan assemblies can have >10M records per file, where
    SeqRecord construction alone dominates the runtime.
    """
    total = 0
    seq_len = 0
    in_record = False
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if in_record:
                    total += _chunks_for_length(seq_len, chunk_size, step_size)
                in_record = True
                seq_len = 0
            else:
                seq_len += len(line.strip())
        if in_record:
            total += _chunks_for_length(seq_len, chunk_size, step_size)
    return total


def count_total_chunks(
    accessions: list[Path], chunk_size: int, chunk_overlap: int
) -> int:
    """Parallel arithmetic replacement for `sum(1 for _ in _iter_chunks(...))`."""
    step_size = chunk_size - chunk_overlap
    if step_size <= 0:
        raise ValueError("The overlap must be strictly less than the chunk size.")
    # Fork, not spawn: workers only read files (no torch/CUDA use), and fork
    # skips re-importing this module's heavy dependencies in every worker.
    num_workers = min(len(accessions), len(os.sched_getaffinity(0)))
    ctx = mp.get_context("fork")
    total = 0
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = [
            executor.submit(_count_file_chunks, str(acc), chunk_size, step_size)
            for acc in accessions
        ]
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Counting chunks"
        ):
            total += f.result()
    return total


# Global variable to hold the model instance per worker process
_worker_encoder = None


def _init_worker(encoder_cfg, gpu_queue):
    """Initializes the model once per worker process on a specific GPU."""
    global _worker_encoder
    faulthandler.enable(file=sys.stderr)  # dump traceback on SIGSEGV/SIGFPE/etc.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device_id = gpu_queue.get()

    local_cfg = copy.copy(encoder_cfg)
    local_cfg.device = f"cuda:{device_id}"

    # Initialize and keep in memory
    _worker_encoder = DenseEncoder(local_cfg)


def _process_sequence_batch(batch):
    """Encodes a batch of (srr_id, sequence) tuples."""
    global _worker_encoder
    assert isinstance(_worker_encoder, DenseEncoder)
    try:
        # Unzip the batch into IDs and sequences
        srr_ids = [item[0] for item in batch]
        sequences = [item[1] for item in batch]

        # Encode the batch (encoder.encode handles its own internal batching if needed,
        # but ideally this function's batch size matches your optimal GPU batch size)
        embeddings = _worker_encoder.encode(sequences).cpu()

        # Return the mapping to be re-assembled by the main process
        return srr_ids, embeddings
    except Exception as e:
        # Print the full traceback directly to the console from the worker
        print("\n--- WORKER ERROR ---\n", file=sys.stderr)
        traceback.print_exc()
        print("--------------------\n", file=sys.stderr)
        raise e  # Re-raise so the main thread knows it failed


class DenseIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, DenseMethod)
        self.no_search: bool = cfg.no_search
        self.cfg = cfg
        self.method: DenseMethod = cfg.model
        self.encoder_cfg = cfg.model.encoder
        self.engine = make_engine(cfg.model.index)
        # Scoring protocol: top_k vectors regrouped to accessions unless the
        # index is the exhaustive reference.
        self.exhaustive: bool = isinstance(cfg.model.index, ExhaustiveIndex)
        self.top_k: int | None = getattr(cfg.model.index, "top_k", None)
        self.both_strands: bool = cfg.model.encoder.both_strands

        # Unconditional: build() needs the encoder for embed_dim and the chunking
        # attributes for _iter_chunks, so a no_search (build-only) run crashed
        # when these lived behind `if not self.no_search`.
        enc_cfg = copy.copy(cfg.model.encoder)
        if self.engine.WORKER_EMBED:
            # The engine's GPU workers embed the queries on their own cards;
            # this process stays off CUDA (the cards are ~97% full of index).
            enc_cfg.device = "cpu"
        self.model = DenseEncoder(enc_cfg)
        # stride is the only supported chunking mode (legacy column in run_benchmark).
        self.chunk_type: Literal["stride"] = "stride"
        self.chunk_overlap: int = cfg.model.encoder.chunk_overlap

    # ------------------------------------------------------------------ #
    # load / build
    # ------------------------------------------------------------------ #

    @staticmethod
    def _devices(fallback: str) -> list[str]:
        if torch.cuda.is_available():
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return [fallback]

    def load(self, index_path: Path):
        """index_path is the encoder's directory; the engine's artifact is
        index_path/<index_label>."""
        meta = pl.read_parquet(index_path / "meta.parquet")
        self.acc_names_flat = meta["srr_id"].to_list()
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        self.acc_offsets = starts + [starts[-1] + counts[-1]] if starts else [0]
        self.n_vectors: int = int(self.acc_offsets[-1])
        self.engine.load(
            index_path,
            index_path / self.method.index_label,
            self._devices(self.encoder_cfg.device),
            encoder_cfg=self.encoder_cfg if self.engine.WORKER_EMBED else None,
        )

    def _iter_chunks(self, accessions: list[Path]):
        """Yield (srr_id, sequence_chunk) for every chunk across all accessions."""
        for accession in accessions:
            srr_id = accession.parent.stem
            for record in SeqIO.parse(accession, "fasta"):
                seq = str(record.seq)
                contig_id = str(record.id)
                if len(seq) <= self.encoder_cfg.max_seq_len:
                    yield srr_id, seq
                else:
                    for chunk in chunk_sequence(
                        seq,
                        contig_id,
                        self.encoder_cfg.max_seq_len,
                        self.chunk_overlap,
                        self.chunk_type,
                    ):
                        yield srr_id, chunk

    def build(self, accessions: list[Path], index_path: Path):
        """Embed every chunk of ``accessions`` into index_path/embeddings.fbin
        (+ meta.parquet). The engine's own artifact, if any, is built by
        engine.build afterwards (run_benchmark / build_*.py)."""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for building the index.")

        print("Pre-counting chunks (parallel length scan)...")
        total_chunks = count_total_chunks(
            accessions, self.encoder_cfg.max_seq_len, self.chunk_overlap
        )
        print(f"Total chunks: {total_chunks:,}")

        embed_dim = self.model.encode(["ACGT"]).shape[1]

        index_path.mkdir(exist_ok=True, parents=True)
        raw_mmap = _create_fbin_memmap(
            index_path / "embeddings.fbin", total_chunks, embed_dim
        )

        print(f"Embedding across {num_gpus} GPUs...")
        ctx = mp.get_context("spawn")
        m = ctx.Manager()
        gpu_queue = m.Queue()
        for i in range(num_gpus):
            gpu_queue.put(i)

        submission_batch_size = self.encoder_cfg.batch_size * 4
        rows: list[dict] = []
        offset = 0

        # Window size caps how many completed-but-uncollected results sit in RAM.
        # Submitting all futures at once lets workers race far ahead of the main
        # loop, causing unbounded result accumulation that triggers OOM kills.
        window_size = num_gpus * 2
        batch_iter = iter(batched(self._iter_chunks(accessions), submission_batch_size))
        total_batches = -(-total_chunks // submission_batch_size)  # ceil div

        def _next_future(ex):
            batch = next(batch_iter, None)
            return (
                ex.submit(_process_sequence_batch, batch) if batch is not None else None
            )

        with ProcessPoolExecutor(
            max_workers=num_gpus,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.encoder_cfg, gpu_queue),
        ) as executor:
            # Seed the window
            window = [
                f
                for _ in range(window_size)
                if (f := _next_future(executor)) is not None
            ]

            # Process in submission order so accession chunks land contiguously.
            with tqdm(total=total_batches, desc="Embedding batches") as pbar:
                while window:
                    future = window.pop(0)
                    try:
                        srr_ids, embeddings = future.result()
                    except process.BrokenProcessPool:
                        print("\n[!] A worker died abruptly. Halting.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as e:
                        print(f"Worker failed: {e}")
                        pbar.update(1)
                        nxt = _next_future(executor)
                        if nxt:
                            window.append(nxt)
                        continue

                    arr = embeddings.numpy()
                    srr_groups: dict[str, list[int]] = defaultdict(list)
                    for i, srr_id in enumerate(srr_ids):
                        srr_groups[srr_id].append(i)
                    for srr_id, indices in srr_groups.items():
                        chunk = arr[indices]
                        n = len(chunk)
                        raw_mmap[offset : offset + n] = chunk
                        rows.append(
                            {"srr_id": srr_id, "start_row": offset, "num_rows": n}
                        )
                        offset += n

                    pbar.update(1)
                    nxt = _next_future(executor)
                    if nxt:
                        window.append(nxt)

        raw_mmap.flush()
        del raw_mmap

        # Merge consecutive rows for the same accession into single entries
        # (a batch boundary may split one accession across two consecutive rows).
        final_rows: list[dict] = []
        for r in rows:
            if final_rows and final_rows[-1]["srr_id"] == r["srr_id"]:
                final_rows[-1]["num_rows"] += r["num_rows"]
            else:
                final_rows.append(dict(r))

        pl.DataFrame(final_rows).write_parquet(index_path / "meta.parquet")
        self._streamed_to = index_path
        print(
            f"Built: {offset:,} vectors, {len(final_rows)} accessions -> {index_path}"
        )

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def search(
        self, queries: pl.DataFrame, acc_indices: list[int] | None = None
    ) -> pl.DataFrame:
        """Per-accession results frame for ``queries`` (see topk_regroup).

        Default protocol: the engine retrieves each query's top_k nearest
        index vectors (topk_hits), the hits are grouped by accession and each
        accession scored by its max hit; accessions with no hit get
        MISS_SCORE. run_benchmark calls topk_hits itself to persist the raw
        hits and shard the scan across nodes; this is the single-node wrapper
        that also serves the timing runs.

        ``ExhaustiveIndex`` is the reference protocol: every accession is
        scored by the max over all its vectors (a long query is chunked
        without overlap and scored as the sum of per-chunk maxima; a query at
        most max_seq_len long is one chunk, so that reduces to a plain max).
        With both_strands the query and its reverse complement are scored
        separately and the accession keeps the larger. acc_indices restricts
        the exhaustive scan to that subset of accessions -- used by multi-node
        search, where each node scores a stride of the index -- and the
        results then cover only those accessions.
        """
        if not self.exhaustive:
            if acc_indices is not None:
                raise NotImplementedError(
                    "acc_indices is only supported by the exhaustive search path"
                )
            hits = self.topk_hits(queries)
            df = regroup_topk_hits(hits, self.acc_names_flat)
            assert len(df) == len(queries)
            return df
        if acc_indices is None:
            acc_indices = list(range(len(self.acc_names_flat)))
        queries = queries.with_row_index()
        query_chunk_features, query_indices, chunk_strand = self._embed_queries(queries)
        n_strands = 2 if self.both_strands else 1
        n_chunks = len(query_chunk_features)
        n_queries = len(queries)

        # Slot = (query, strand): chunk maxima are summed within a slot and
        # a query keeps its best strand.
        chunk_to_query = np.zeros(n_chunks, dtype=np.int64)
        for qi, (s, e) in enumerate(query_indices):
            chunk_to_query[s:e] = qi
        chunk_to_slot = chunk_to_query * n_strands + chunk_strand.numpy().astype(np.int64)
        slot_scores = self.engine.slot_scores(
            query_chunk_features.float().cpu().numpy(),
            chunk_to_slot,
            n_queries * n_strands,
            self.acc_offsets,
            acc_indices,
        )
        scores_cpu = slot_scores.reshape(n_queries, n_strands, -1).max(axis=1)

        accession_names = [self.acc_names_flat[i] for i in acc_indices]
        scores_df = []
        for i in range(scores_cpu.shape[0]):
            for j in range(scores_cpu.shape[1]):
                scores_df.append(
                    {
                        "query_idx": i,
                        "accession": accession_names[j],
                        "score": float(scores_cpu[i, j]),
                    }
                )

        scores_df = pl.from_dicts(scores_df)
        scores_df = (
            scores_df.with_columns(pl.struct("accession", "score").alias("result"))
            .group_by("query_idx")
            .agg(pl.col("result").alias("results"))
        )
        df = queries.join(
            scores_df, left_on="index", right_on="query_idx", how="left"
        ).select("query_id", "results")

        assert len(df) == len(queries)
        return df

    @torch.no_grad()
    def topk_hits(
        self, queries: pl.DataFrame, vec_range: tuple[int, int] | None = None
    ) -> pl.DataFrame:
        """Each query's top_k vector hits from the engine.

        Returns one row per query: ``query_id`` and ``hits``, a
        score-descending list of at most top_k ``{accession, score,
        vector_id}`` structs. This is the artifact run_benchmark persists:
        smaller k are its prefixes, and regroup_topk_hits turns any prefix
        into the standard per-accession results. A long query's chunks each
        keep their own top-k; the query's list is their union, deduplicated
        by vector (max score) and cut back to top_k. With both_strands a
        query's reverse-complement chunks are simply more chunks of that
        query. vec_range (row sharding, SHARDABLE engines only) makes the
        scan shardable across nodes; node 0 merges with merge_topk_hits.
        """
        if self.exhaustive:
            raise NotImplementedError("topk_hits has no meaning under exhaustive")
        if vec_range is not None and not self.engine.SHARDABLE:
            raise NotImplementedError(f"{type(self.engine).__name__} cannot be row-sharded")
        t0 = time.time()
        query_chunk_features, query_indices, _ = self._embed_queries(queries)
        q = query_chunk_features.float().cpu().numpy()
        embed_s = time.time() - t0
        scores, ids = self.engine.topk_hits(q, self.top_k, vec_range)
        t1 = time.time()
        hits = self._hits_from_chunk_topk(queries, query_indices, scores, ids)
        timings = getattr(self.engine, "last_timings", None)
        if timings:
            self.last_timings = {
                "embed_s": embed_s,
                "n_chunks": len(q),
                **timings,
                "hits_s": time.time() - t1,
            }
            print(f"search timings: {self.last_timings}")
        return hits

    def _hits_from_chunk_topk(
        self,
        queries: pl.DataFrame,
        query_indices: list[tuple[int, int]],
        top_vals: np.ndarray,
        top_ids: np.ndarray,
    ) -> pl.DataFrame:
        """Per-chunk (n_chunks, k) top-k -> per-query hits frame.

        Maps chunk rows back to their query and lets build_hits_frame union a
        query's chunks (positions and, with both_strands, strands), dedup by
        vector and cut to top_k. Padding slots (id -1 / -inf) fall out
        there.
        """
        top_vals = np.asarray(top_vals)
        top_ids = np.asarray(top_ids)
        n_chunks, k = top_ids.shape
        chunk_to_query = np.zeros(n_chunks, dtype=np.int64)
        for qi, (cs, ce) in enumerate(query_indices):
            chunk_to_query[cs:ce] = qi
        return build_hits_frame(
            query_ids=queries["query_id"].to_list(),
            flat_query_pos=np.repeat(chunk_to_query, k),
            flat_vector_ids=top_ids.ravel().astype(np.int64),
            flat_scores=top_vals.ravel().astype(np.float64),
            acc_offsets=np.asarray(self.acc_offsets, dtype=np.int64),
            acc_names=self.acc_names_flat,
            top_k=self.top_k,
        )

    def num_vectors(self) -> int:
        return self.n_vectors

    def indexed_accessions(self) -> list[str]:
        return self.acc_names_flat

    def save(self, output_path: Path):
        print(f"Index already written to {output_path} during build.")

    def index_size_gb(self, index_path: Path):
        return self.engine.size_gb(index_path, index_path / self.method.index_label)

    @staticmethod
    def merge_shards(
        index_path: Path,
        num_nodes: int,
        copy_workers: int = 8,
        buf_bytes: int = 64 << 20,
    ):
        """Concatenate per-node shard fbins into one file, without loading into RAM.

        Each shard owns a disjoint byte range of the merged file, so shards are
        copied concurrently as raw byte streams (fbin is header + contiguous
        float32 rows). A progress file records fully-copied shards, making an
        interrupted merge resumable; completed ranks are recorded only after an
        fsync, so a crash can never mark a partially-written shard done.
        """
        header_bytes = 8
        all_meta = []
        shard_sizes: list[int] = []
        total_vectors = 0
        embed_dim = None

        for rank in range(num_nodes):
            shard_path = index_path / f"shard_{rank}"
            fbin_path = shard_path / "embeddings.fbin"
            with open(fbin_path, "rb") as f:
                n, d = (int(x) for x in np.frombuffer(f.read(8), dtype=np.uint32))
            expected = header_bytes + n * d * 4
            actual = fbin_path.stat().st_size
            if actual != expected:
                raise ValueError(
                    f"{fbin_path}: size {actual} != {expected} for header ({n}, {d})"
                )
            if embed_dim is None:
                embed_dim = d
            elif d != embed_dim:
                raise ValueError(f"{fbin_path}: dim {d} != {embed_dim}")
            meta = pl.read_parquet(shard_path / "meta.parquet")
            meta = meta.with_columns(
                (pl.col("start_row") + total_vectors).alias("start_row")
            )
            all_meta.append(meta)
            shard_sizes.append(n)
            total_vectors += n
            print(f"  Shard {rank}: {len(meta)} accessions")

        if embed_dim is None:
            raise ValueError("No valid shards found")

        row_bytes = embed_dim * 4
        merged_path = index_path / "embeddings.fbin"
        progress_path = index_path / ".merge_progress"
        expected_size = header_bytes + total_vectors * row_bytes

        done_ranks: set[int] = set()
        if (
            progress_path.exists()
            and merged_path.exists()
            and merged_path.stat().st_size == expected_size
        ):
            done_ranks = {int(x) for x in progress_path.read_text().split()}
            print(f"Resuming merge: {len(done_ranks)} shards already streamed")
        else:
            # Sparse pre-allocation: header, then seek-and-touch the last byte
            with open(merged_path, "wb") as f:
                np.array([total_vectors, embed_dim], dtype=np.uint32).tofile(f)
                f.seek(total_vectors * row_bytes - 1, 1)
                f.write(b"\x00")
            progress_path.write_text("")

        shard_offsets = [
            header_bytes + sum(shard_sizes[:r]) * row_bytes for r in range(num_nodes)
        ]
        progress_lock = threading.Lock()

        def _copy_shard(rank: int) -> int:
            src_path = index_path / f"shard_{rank}" / "embeddings.fbin"
            target = shard_sizes[rank] * row_bytes
            with open(src_path, "rb") as src, open(merged_path, "r+b") as out:
                src.seek(header_bytes)
                out.seek(shard_offsets[rank])
                copied = 0
                while copied < target:
                    chunk = src.read(min(buf_bytes, target - copied))
                    if not chunk:
                        raise IOError(f"{src_path} truncated at byte {copied}")
                    out.write(chunk)
                    copied += len(chunk)
                out.flush()
                os.fsync(out.fileno())
            with progress_lock:
                with open(progress_path, "a") as pf:
                    pf.write(f"{rank}\n")
            return rank

        todo = [r for r in range(num_nodes) if r not in done_ranks]
        if todo:
            with ThreadPoolExecutor(max_workers=min(copy_workers, len(todo))) as pool:
                futures = [pool.submit(_copy_shard, r) for r in todo]
                for f in tqdm(
                    as_completed(futures), total=len(futures), desc="Merging shards"
                ):
                    rank = f.result()
                    print(f"  Streamed shard {rank} ({shard_sizes[rank]} vectors)")

        pl.concat(all_meta).write_parquet(index_path / "meta.parquet")
        progress_path.unlink(missing_ok=True)
        print(f"Merged {num_nodes} shards -> {total_vectors} vectors")

    # ------------------------------------------------------------------ #
    # query embedding
    # ------------------------------------------------------------------ #

    def _embed_queries(
        self, queries
    ) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
        """Embed every query, chunking any that exceed max_seq_len.

        Queries are split into non-overlapping max_seq_len windows, so a query
        shorter than that yields exactly one chunk identical to itself. With
        both_strands the query's reverse complement is chunked the same way
        and its chunks follow the forward ones inside the query's range, so
        the second strand costs one more embedding per chunk and nothing
        else -- every caller that reduces over a query's range sees both
        strands. Returns the (n_chunks, dim) features, per query the
        [start, end) range of rows it owns, and per row its strand (0
        forward, 1 reverse complement) for callers that must not sum across
        strands.
        """
        query_chunks: list[str] = []
        query_indices: list[tuple[int, int]] = []
        strands: list[int] = []
        prev_idx = 0
        max_len = self.encoder_cfg.max_seq_len
        for query in queries["query_sequence"].to_list():
            variants = [query]
            if self.both_strands:
                variants.append(reverse_complement(query))
            num_chunks = 0
            for strand, seq in enumerate(variants):
                chunked = [seq[i : i + max_len] for i in range(0, len(seq), max_len)]
                query_chunks.extend(chunked)
                strands.extend([strand] * len(chunked))
                num_chunks += len(chunked)
            query_indices.append((prev_idx, prev_idx + num_chunks))
            prev_idx += num_chunks

        query_features = self._encode_chunks(query_chunks)
        return query_features, query_indices, torch.tensor(strands, dtype=torch.long)

    _query_replicas: list | None = None

    def _encode_chunks(self, chunks: list[str]) -> torch.Tensor:
        """Encode on self.model, on the engine's workers (WORKER_EMBED), or
        split across query_embed_gpus replicas.

        Replicas (DenseEncoder on cuda:1..n-1, same checkpoint) are built on
        first use; each takes a contiguous slice and results are concatenated
        in order on the CPU.
        """
        engine = getattr(self, "engine", None)
        if engine is not None and engine.WORKER_EMBED:
            return torch.from_numpy(engine.embed(chunks))
        n_gpu = min(getattr(self.encoder_cfg, "query_embed_gpus", 1), torch.cuda.device_count())
        if n_gpu <= 1 or len(chunks) < 2 * n_gpu:
            return self.model.encode(chunks)
        if self._query_replicas is None:
            replicas = [self.model]
            for i in range(1, n_gpu):
                rcfg = copy.copy(self.encoder_cfg)
                rcfg.device = f"cuda:{i}"
                replicas.append(DenseEncoder(rcfg))
            self._query_replicas = replicas
        bounds = [len(chunks) * i // n_gpu for i in range(n_gpu + 1)]

        def _run(i):
            with torch.cuda.device(i):
                return self._query_replicas[i].encode(chunks[bounds[i] : bounds[i + 1]]).cpu()

        with ThreadPoolExecutor(max_workers=n_gpu) as pool:
            return torch.cat(list(pool.map(_run, range(n_gpu))))


def chunk_sequence(
    seq: str,
    contig_id: str,
    chunk_size: int,
    chunk_overlap: int,
    chunk_type: Literal["stride"],
):
    if chunk_type == "stride":
        if chunk_overlap >= chunk_size:
            raise ValueError("The overlap must be strictly less than the chunk size.")
        if chunk_size <= 0:
            raise ValueError("Chunk size (c) must be greater than 0.")

        step_size = chunk_size - chunk_overlap

        # Generate chunks of exactly size c
        chunks = [
            seq[i : i + chunk_size]
            for i in range(0, len(seq) - chunk_size + 1, step_size)
        ]
    else:
        raise ValueError(f"Incorrect chunk_type recieved: {chunk_type}")

    return chunks
