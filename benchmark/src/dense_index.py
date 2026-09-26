import copy
import faulthandler
import multiprocessing as mp
import os
import random
import sys
import threading
import traceback
import time
from collections import defaultdict
from contextlib import contextmanager
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
from .config import DenseConfig, ExperimentConfig

# Encoders and the fbin helpers moved to their own modules so an index can use one
# without importing this one. Re-exported because tests/ and other callers import
# them from here.
from .encoders import DenseEncoder, _strings_to_one_hot, batched  # noqa: F401
from .fbin import _create_fbin_memmap, _load_fbin_mmap, read_fbin_rows  # noqa: F401
from .rabitq import RaBitQIndex, build_rabitq_index  # noqa: F401
from .topk_regroup import build_hits_frame, regroup_topk_hits

# cuVS CAGRA serializes to a single file rather than DiskANN's directory of them.
CAGRA_INDEX_FILE = "cagra_index.bin"

# Build/search knobs carried over from the DiskANN parameters they replace:
# graph_degree is the same 64, complexity=128 becomes the build-time candidate
# list (intermediate_graph_degree) and the search-time one (itopk_size).
CAGRA_GRAPH_DEGREE = 64
CAGRA_INTERMEDIATE_GRAPH_DEGREE = 128
CAGRA_ITOPK_SIZE = 128

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


def _init_worker(cfg, gpu_queue):
    """Initializes the model once per worker process on a specific GPU."""
    global _worker_encoder
    faulthandler.enable(file=sys.stderr)  # dump traceback on SIGSEGV/SIGFPE/etc.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device_id = gpu_queue.get()

    local_cfg = copy.deepcopy(cfg)
    local_cfg.model.device = f"cuda:{device_id}"

    # Initialize and keep in memory
    _worker_encoder = DenseEncoder(local_cfg.model)


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


class _FbinBlockLoader:
    """Brings row blocks of an fbin onto one device: pread into a pinned host
    buffer, then a single H2D copy. One instance per scanning thread; each
    holds its own descriptor (positional reads, no shared offset) and one
    (block_rows, d) staging buffer, so a 2M-row block is 6 GB of pinned
    memory per GPU. See read_fbin_rows for why this replaces memmap slicing.
    """

    def __init__(self, path: Path, d: int, block_rows: int, device: torch.device):
        self.fd = os.open(path, os.O_RDONLY)
        self.device = device
        self.buf = torch.empty(
            (block_rows, d), dtype=torch.float32, pin_memory=device.type == "cuda"
        )

    def __call__(self, bs: int, be: int) -> torch.Tensor:
        """Rows [bs, be) as a (be - bs, d) tensor on the device."""
        read_fbin_rows(self.fd, bs, be, self.buf.numpy())
        return self.buf[: be - bs].to(self.device)

    def close(self) -> None:
        os.close(self.fd)


class DenseIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, DenseConfig)
        self.no_search: bool = cfg.no_search
        self.k = cfg.model.k
        self.cfg = cfg
        self.use_ann: bool = cfg.model.use_ann
        self.use_rabitq: bool = cfg.model.use_rabitq
        self.use_ivf: bool = cfg.model.use_ivf
        self.use_ivfpq: bool = cfg.model.use_ivfpq
        # Scoring protocol, see DenseConfig: top_k vectors regrouped to
        # accessions unless exhaustive.
        self.exhaustive: bool = cfg.model.exhaustive
        self.top_k: int = cfg.model.top_k
        self.both_strands: bool = cfg.model.both_strands
        self.rabitq_sample_rows: int = cfg.model.rabitq_sample_rows
        self.model_cfg = cfg.model

        # Unconditional: build() needs the encoder for embed_dim and the chunking
        # attributes for _iter_chunks, so a no_search (build-only) run crashed
        # when these lived behind `if not self.no_search`.
        if cfg.model.use_ivfpq:
            # The GPU IVF-PQ workers embed the queries on their own cards;
            # this process stays off CUDA (the cards are ~97% full of index).
            cpu_cfg = copy.copy(cfg.model)
            cpu_cfg.device = "cpu"
            self.model = DenseEncoder(cpu_cfg)
        else:
            self.model = DenseEncoder(cfg.model)
        # chunk_type was dropped from DenseConfig during the v1 cleanup;
        # stride is the only supported mode (see the hardcoded "chunkstride"
        # index tag in config.py and the legacy column in run_benchmark.py).
        self.chunk_type: Literal["stride", "exact_chunk"] = "stride"
        self.chunk_overlap: int = cfg.model.chunk_overlap
        self.contig_align_intervals: dict[str, list[tuple[int, int]]] | None = None

    def load(self, index_path: Path):
        meta = pl.read_parquet(index_path / "meta.parquet")
        self.acc_names_flat = meta["srr_id"].to_list()
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        self.acc_offsets = starts + [starts[-1] + counts[-1]] if starts else [0]
        self.n_vectors: int = int(self.acc_offsets[-1])
        if self.use_ann:
            # Imported here, not at module scope: cuvs pulls in the CUDA runtime, and
            # run_benchmark imports this module on nodes that have no GPU.
            from cuvs.neighbors import cagra

            cagra_path = index_path / CAGRA_INDEX_FILE
            if not cagra_path.exists():
                self.construct_ann_index(index_path)

            self.index = cagra.load(str(cagra_path))
            self.all_embeddings = None
        elif self.use_ivfpq:
            from .ivfpq_gpu import IVFPQGPUSearcher, ivfpq_dir, list_shard_files

            m = self.model_cfg
            d = ivfpq_dir(index_path, m.ivfpq_pq_dim, m.ivfpq_pq_bits, m.ivfpq_lists_per_shard)
            self.ivf_index_files = list_shard_files(d, m.ivfpq_num_shards)
            self.ivfpq = IVFPQGPUSearcher(
                self.ivf_index_files,
                index_path / "embeddings.fbin",
                encoder_cfg=copy.copy(m),
            )
            self.all_embeddings = None
        elif self.use_ivf:
            # Built by build_ivf.py (multi-node, one shard per rank); the first
            # load on a node with enough RAM folds the shards into one file.
            from .ivf_rabitq import (
                IVFRaBitQSearcher,
                list_shards,
                merge_ivf_shards,
                merged_path,
            )

            m = self.model_cfg
            ivf_dir = index_path / "ivf"
            merged = merged_path(ivf_dir, m.ivf_nlist, m.ivf_nb_bits)
            shards = list_shards(ivf_dir, m.ivf_nlist, m.ivf_nb_bits)
            if m.ivf_fastscan:
                # FastScan shards cannot be merged (faiss merge_from bug at
                # this size), so they are searched side by side.
                files = shards or [merged]
            else:
                if not merged.exists():
                    if not shards:
                        raise FileNotFoundError(
                            f"no IVF index at {merged} and no shards to merge; "
                            "build it with build_ivf.py"
                        )
                    merge_ivf_shards(ivf_dir, m.ivf_nlist, m.ivf_nb_bits, len(shards))
                files = [merged]
            self.ivf = IVFRaBitQSearcher(
                files,
                index_path / "embeddings.fbin",
                quantizer=m.ivf_quantizer,
                fastscan=m.ivf_fastscan,
            )
            self.ivf_index_files = files
            self.all_embeddings = None
        elif self.use_rabitq:
            # Single-node build if missing (a multi-node run builds the shards
            # before load, see run_benchmark). Only metadata is read here; the
            # codes for a row range are brought onto the GPUs by topk_hits.
            rabitq_dir = index_path / "rabitq"
            if not (rabitq_dir / "meta.json").exists():
                print("RaBitQ index not found, building from embeddings.fbin...")
                build_rabitq_index(
                    index_path / "embeddings.fbin",
                    rabitq_dir,
                    centroid_sample_rows=self.rabitq_sample_rows,
                )
            self.rabitq_index = RaBitQIndex.open(rabitq_dir)
            self.all_embeddings = None
        else:
            mmap = _load_fbin_mmap(index_path / "embeddings.fbin")
            self._mmap = mmap  # keep reference to prevent GC closing the mapping
            self.all_embeddings = torch.from_numpy(mmap)
            # The scans read blocks through _block_loader (pread), not by
            # slicing this memmap; it stays for shape and small lookups.
            self._fbin_path = index_path / "embeddings.fbin"
            print(
                f"Loaded {len(starts)} accessions ({mmap.shape[0]} vectors) [memory-mapped]"
            )

    def _iter_chunks(self, accessions: list[Path]):
        """Yield (srr_id, sequence_chunk) for every chunk across all accessions."""
        for accession in accessions:
            srr_id = accession.parent.stem
            for record in SeqIO.parse(accession, "fasta"):
                seq = str(record.seq)
                contig_id = str(record.id)
                if len(seq) <= self.model_cfg.max_seq_len:
                    yield srr_id, seq
                else:
                    for chunk in chunk_sequence(
                        seq,
                        contig_id,
                        self.model_cfg.max_seq_len,
                        self.chunk_overlap,
                        self.chunk_type,
                        self.contig_align_intervals,
                    ):
                        yield srr_id, chunk

    def build(self, accessions: list[Path], index_path: Path):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No GPUs available for building the index.")

        print("Pre-counting chunks (parallel length scan)...")
        total_chunks = count_total_chunks(
            accessions, self.model_cfg.max_seq_len, self.chunk_overlap
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

        submission_batch_size = self.model_cfg.batch_size * 4
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
            initargs=(self.cfg, gpu_queue),
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

        if self.use_rabitq:
            print("Building RaBitQ quantized index...")
            build_rabitq_index(
                index_path / "embeddings.fbin",
                index_path / "rabitq",
                centroid_sample_rows=self.rabitq_sample_rows,
            )

    @torch.no_grad()
    def search(
        self, queries: pl.DataFrame, acc_indices: list[int] | None = None
    ) -> pl.DataFrame:
        """Per-accession results frame for ``queries`` (see topk_regroup).

        Default protocol: the configured vector engine retrieves each query's
        top_k nearest index vectors (topk_hits), the hits are grouped by
        accession and each accession scored by its max hit;
        accessions with no hit get MISS_SCORE. run_benchmark calls topk_hits
        itself to persist the raw hits and shard the scan across nodes; this
        is the single-node wrapper that also serves the timing runs.

        ``exhaustive`` is the reference protocol: every accession is scored
        by the max over all its vectors (a long query is chunked without
        overlap and scored as the sum of per-chunk maxima; a query at most
        max_seq_len long is one chunk, so that reduces to a plain max). With
        both_strands the query and its reverse complement are scored
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
        assert not (self.use_ann or self.use_rabitq)
        if acc_indices is None:
            acc_indices = list(range(len(self.acc_names_flat)))
        queries = queries.with_row_index()
        query_chunk_features, query_indices, chunk_strand = self._embed_queries(queries)
        query_chunk_features = query_chunk_features.to(self.model_cfg.device)
        n_strands = 2 if self.both_strands else 1
        assert self.all_embeddings is not None
        n_chunks = len(query_chunk_features)
        n_queries = len(queries)
        n_acc = len(acc_indices)

        # Per-accession scoring (2 GPU ops per accession instead of
        # n_queries), parallelized across all GPUs: accessions are dealt
        # round-robin to one thread per device. GPU ops and the pread
        # block loads release the GIL, so the threads also overlap the
        # multi-TB index read. Each accession streams through its GPU in blocks:
        # the largest ones (~30M+ vectors, 90+ GB fp32) do not fit on a
        # 40 GB A100 as a single slice, and maximum() over block maxima
        # equals the full max.
        block_rows = 2_000_000
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = [self.model_cfg.device]
        qcf_cpu = query_chunk_features.cpu()
        # Slot = (query, strand): chunk maxima are summed within a slot and
        # a query keeps its best strand.
        chunk_to_query_cpu = torch.zeros(n_chunks, dtype=torch.long)
        for qi, (s, e) in enumerate(query_indices):
            chunk_to_query_cpu[s:e] = qi
        chunk_to_slot_cpu = chunk_to_query_cpu * n_strands + chunk_strand.long()

        acc_offsets = self.acc_offsets
        # Threads write disjoint columns, so unsynchronized writes are safe
        scores_cpu = np.zeros((n_queries, n_acc), dtype=np.float32)

        def _score_accessions(dev_idx: int):
            dev = torch.device(devices[dev_idx])
            q = qcf_cpu.to(dev)
            c2s = chunk_to_slot_cpu.to(dev)
            positions = range(dev_idx, n_acc, len(devices))
            if dev_idx == 0:
                positions = tqdm(
                    positions, desc=f"Scoring accessions ({len(devices)} GPUs)"
                )
            with self._block_loader(block_rows, dev) as load_block:
                for pos in positions:
                    i = acc_indices[pos]
                    s, e = acc_offsets[i], acc_offsets[i + 1]
                    if e <= s:
                        continue
                    chunk_maxes = torch.full(
                        (n_chunks,), float("-inf"), device=dev, dtype=q.dtype
                    )
                    for bs in range(s, e, block_rows):
                        be = min(bs + block_rows, e)
                        logits = q @ load_block(bs, be).T
                        chunk_maxes = torch.maximum(
                            chunk_maxes, logits.max(dim=-1).values
                        )
                        del logits
                    slot_scores = torch.zeros(
                        n_queries * n_strands, device=dev, dtype=q.dtype
                    )
                    slot_scores.scatter_add_(0, c2s, chunk_maxes)
                    acc_scores = slot_scores.view(n_queries, n_strands).max(dim=1)
                    scores_cpu[:, pos] = acc_scores.values.cpu().numpy()

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            # list() propagates any worker exception
            list(pool.map(_score_accessions, range(len(devices))))

        accession_names = [self.acc_names_flat[i] for i in acc_indices]
        # scores_col = []
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
    def exact_topk_hits(
        self,
        queries: pl.DataFrame,
        vec_range: tuple[int, int] | None = None,
        block_rows: int = 2_000_000,
    ) -> pl.DataFrame:
        """Exact top-k nearest index vectors per query, streamed off the fbin.

        Scans index rows [vec_range) -- the whole index when None -- and returns
        one row per query: ``query_id`` and ``hits``, a score-descending list of
        at most ``self.top_k`` ``{accession, score, vector_id}`` structs.
        This is the artifact run_benchmark persists: smaller k are its
        prefixes, and regroup_topk_hits turns any prefix into the standard
        per-accession results.

        The range is split evenly across the node's GPUs; each streams its
        sub-range in ``block_rows`` blocks (2M x 768 fp32 = 6 GB, the same tile
        the streaming dense path uses; read with pread, see _block_loader) and folds every block's top-k into a
        running (n_chunks, top_k) buffer, so the full 7 TB index never has to
        fit anywhere. A long query's chunks each keep their own top-k; the
        query's list is their union, deduplicated by vector (max score) and cut
        back to top_k. vec_range makes the scan shardable: each node's
        hits are exact over its own range, so node 0 merges them with
        merge_topk_hits. With both_strands a query's reverse-complement chunks
        are simply more chunks of that query, so its list is the union of both
        strands' hits.
        """
        assert self.all_embeddings is not None
        n_vecs = self.all_embeddings.shape[0]
        start, end = (0, n_vecs) if vec_range is None else vec_range
        if not (0 <= start <= end <= n_vecs):
            raise ValueError(f"vec_range {vec_range} outside [0, {n_vecs}]")
        top_k = self.top_k

        query_chunk_features, query_indices, _ = self._embed_queries(queries)
        qcf_cpu = query_chunk_features.cpu().float()
        n_chunks = len(qcf_cpu)

        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = [self.model_cfg.device]
        n_dev = len(devices)
        per_dev = -(-(end - start) // n_dev) if end > start else 0

        def _scan(dev_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            dev = torch.device(devices[dev_idx])
            s = start + dev_idx * per_dev
            e = min(s + per_dev, end)
            q = qcf_cpu.to(dev)
            best_vals = torch.full(
                (n_chunks, top_k), float("-inf"), device=dev, dtype=q.dtype
            )
            best_ids = torch.full((n_chunks, top_k), -1, device=dev, dtype=torch.int64)
            blocks = range(s, e, block_rows)
            if dev_idx == 0:
                blocks = tqdm(
                    blocks,
                    desc=f"Exact top-{top_k} vector scan ({n_dev} device(s), "
                    f"{end - start:,} vectors)",
                )
            with self._block_loader(block_rows, dev) as load_block:
                for bs in blocks:
                    be = min(bs + block_rows, e)
                    logits = q @ load_block(bs, be).T  # (n_chunks, be-bs)
                    k = min(top_k, be - bs)
                    vals, idx = torch.topk(logits, k, dim=-1)
                    # The logits tile is the big allocation (2024 chunks x 2M
                    # rows fp32 = 15 GB with 1000 both-strand queries). Drop it
                    # before the next block is loaded; otherwise the previous
                    # tile is still bound when the next matmul allocates and
                    # the peak is two tiles plus a block, which OOMs a 40 GB
                    # A100.
                    del logits
                    cand_vals = torch.cat([best_vals, vals], dim=1)
                    cand_ids = torch.cat([best_ids, idx + bs], dim=1)
                    best_vals, pos = torch.topk(cand_vals, top_k, dim=-1)
                    best_ids = torch.gather(cand_ids, 1, pos)
                    del vals, idx, cand_vals, cand_ids, pos
            return best_vals.cpu(), best_ids.cpu()

        with ThreadPoolExecutor(max_workers=n_dev) as pool:
            shard_results = list(pool.map(_scan, range(n_dev)))

        # Per-device buffers are exact over disjoint sub-ranges; fold them into
        # per-chunk global top-k, then let build_hits_frame union a query's
        # chunks. Padding slots (id -1, -inf) fall out in that union.
        all_vals = torch.cat([r[0] for r in shard_results], dim=1)
        all_ids = torch.cat([r[1] for r in shard_results], dim=1)
        k = min(top_k, all_vals.shape[1])
        top_vals, top_pos = torch.topk(all_vals, k, dim=-1)
        top_ids = torch.gather(all_ids, 1, top_pos)

        return self._hits_from_chunk_topk(queries, query_indices, top_vals, top_ids)

    def _hits_from_chunk_topk(
        self,
        queries: pl.DataFrame,
        query_indices: list[tuple[int, int]],
        top_vals: torch.Tensor,
        top_ids: torch.Tensor,
    ) -> pl.DataFrame:
        """Per-chunk (n_chunks, k) top-k -> per-query hits frame.

        Maps chunk rows back to their query and lets build_hits_frame union a
        query's chunks (positions and, with both_strands, strands), dedup by
        vector and cut to top_k. Padding slots (id -1 / -inf) fall out
        there.
        """
        n_chunks, k = top_ids.shape
        chunk_to_query = np.zeros(n_chunks, dtype=np.int64)
        for qi, (cs, ce) in enumerate(query_indices):
            chunk_to_query[cs:ce] = qi
        return build_hits_frame(
            query_ids=queries["query_id"].to_list(),
            flat_query_pos=np.repeat(chunk_to_query, k),
            flat_vector_ids=top_ids.numpy().ravel(),
            flat_scores=top_vals.numpy().ravel().astype(np.float64),
            acc_offsets=np.asarray(self.acc_offsets, dtype=np.int64),
            acc_names=self.acc_names_flat,
            top_k=self.top_k,
        )

    @torch.no_grad()
    def rabitq_topk_hits(
        self, queries: pl.DataFrame, vec_range: tuple[int, int] | None = None
    ) -> pl.DataFrame:
        """Top-k by 1-bit RaBitQ estimated inner product over rows [vec_range).

        Same contract as exact_topk_hits (hits frame with global vector ids,
        score-descending, at most top_k per query) so the multi-node merge,
        the hits artifact and the re-scoring all apply unchanged; the scores
        are estimates, so validate the candidate set against an exact run.
        The range's packed codes are loaded onto this node's devices on first
        use -- one node of an N-node search holds 1/N of the codes.
        """
        assert self.rabitq_index is not None
        n_vecs = self.rabitq_index.n
        start, end = (0, n_vecs) if vec_range is None else vec_range
        if not (0 <= start <= end <= n_vecs):
            raise ValueError(f"vec_range {vec_range} outside [0, {n_vecs}]")
        query_chunk_features, query_indices, _ = self._embed_queries(queries)
        if end == start:
            n_chunks = len(query_chunk_features)
            return self._hits_from_chunk_topk(
                queries,
                query_indices,
                torch.empty((n_chunks, 0)),
                torch.empty((n_chunks, 0), dtype=torch.long),
            )
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = [self.model_cfg.device]
        self.rabitq_index.load_rows(start, end, devices)
        scores, ids = self.rabitq_index.search(
            query_chunk_features.cpu().float(), self.top_k
        )
        return self._hits_from_chunk_topk(queries, query_indices, scores, ids)

    @torch.no_grad()
    def ann_topk_hits(self, queries: pl.DataFrame) -> pl.DataFrame:
        """Top-k by CAGRA graph search; same hits contract as the scans.

        The graph holds the whole index, so there is no vec_range: an ANN
        search runs on one node. Results are approximate (a true neighbor the
        graph walk misses is absent from the list rather than mis-scored).
        """
        from cuvs.neighbors import cagra

        query_chunk_features, query_indices, _ = self._embed_queries(queries)
        q = query_chunk_features.to(self.model_cfg.device).float().contiguous()
        n_chunks = len(q)
        top_k = self.top_k
        # cuVS reads and writes through the CUDA array interface, which torch
        # tensors implement -- passing the output buffers in keeps the results
        # in torch and avoids a cupy dependency just to move them back.
        identifiers = torch.empty((n_chunks, top_k), dtype=torch.int64, device=q.device)
        distances = torch.empty((n_chunks, top_k), dtype=torch.float32, device=q.device)
        cagra.search(
            # itopk_size must be at least k for CAGRA to return k neighbors.
            cagra.SearchParams(itopk_size=max(CAGRA_ITOPK_SIZE, top_k)),
            self.index,
            q,
            top_k,
            neighbors=identifiers,
            distances=distances,
        )
        # inner_product "distances" are the raw similarities, largest first.
        return self._hits_from_chunk_topk(
            queries, query_indices, distances.cpu(), identifiers.cpu()
        )

    @torch.no_grad()
    def ivf_topk_hits(self, queries: pl.DataFrame) -> pl.DataFrame:
        """Top-k by IVF-RaBitQ candidate scan + exact fp32 rerank.

        Same hits contract as the scans; like CAGRA it holds the whole index,
        so there is no vec_range. Per-stage wall times of the last call are
        kept in self.last_timings (embed, 1-bit scan, rerank).
        """
        m = self.model_cfg
        t0 = time.time()
        query_chunk_features, query_indices, _ = self._embed_queries(queries)
        q = query_chunk_features.float().cpu().numpy()
        timings = {"embed_s": time.time() - t0, "n_chunks": len(q)}
        scores, ids = self.ivf.search(
            q,
            self.top_k,
            nprobe=m.ivf_nprobe,
            rerank=m.ivf_rerank,
            qb=m.ivf_qb,
            timings=timings,
        )
        self.last_timings = timings
        print(f"IVF search timings: {timings}")
        return self._hits_from_chunk_topk(
            queries,
            query_indices,
            torch.from_numpy(scores),
            torch.from_numpy(ids),
        )

    @torch.no_grad()
    def ivfpq_topk_hits(self, queries: pl.DataFrame) -> pl.DataFrame:
        """Top-k by GPU IVF-PQ (cuVS) over every shard, optional exact rerank.

        Same hits contract as the scans; the whole index is on this node's
        GPUs, so there is no vec_range. Stage wall times of the last call are
        kept in self.last_timings.
        """
        m = self.model_cfg
        t0 = time.time()
        query_chunk_features, query_indices, _ = self._embed_queries(queries)
        q = query_chunk_features.float().cpu().numpy()
        timings = {"embed_s": time.time() - t0, "n_chunks": len(q)}
        scores, ids = self.ivfpq.search(
            q,
            self.top_k,
            n_probes=m.ivfpq_nprobe,
            rerank=m.ivfpq_rerank,
            lut_dtype=m.ivfpq_lut,
            timings=timings,
        )
        t1 = time.time()
        hits = self._hits_from_chunk_topk(
            queries, query_indices, torch.from_numpy(scores), torch.from_numpy(ids)
        )
        timings["hits_s"] = time.time() - t1
        self.last_timings = timings
        print(f"IVF-PQ search timings: {timings}")
        return hits

    def topk_hits(
        self, queries: pl.DataFrame, vec_range: tuple[int, int] | None = None
    ) -> pl.DataFrame:
        """Each query's top_k vector hits from the configured engine.

        Exact fp32 scan by default; use_rabitq and use_ann select the others.
        Only the two scans accept vec_range (the multi-node row sharding).
        """
        if self.exhaustive:
            raise NotImplementedError("topk_hits has no meaning under exhaustive")
        if self.use_rabitq:
            return self.rabitq_topk_hits(queries, vec_range)
        if self.use_ann:
            if vec_range is not None:
                raise NotImplementedError("CAGRA search cannot be row-sharded")
            return self.ann_topk_hits(queries)
        if self.use_ivf:
            if vec_range is not None:
                raise NotImplementedError("IVF search cannot be row-sharded")
            return self.ivf_topk_hits(queries)
        if self.use_ivfpq:
            if vec_range is not None:
                raise NotImplementedError("IVF-PQ search cannot be row-sharded")
            return self.ivfpq_topk_hits(queries)
        return self.exact_topk_hits(queries, vec_range)

    # Engine flag defaults for instances built without __init__ (tests).
    use_ivf: bool = False
    use_ivfpq: bool = False

    # Set by load() when the embeddings live in an fbin; None for in-memory
    # embeddings (tests), which the scans then slice directly.
    _fbin_path: Path | None = None

    @contextmanager
    def _block_loader(self, block_rows: int, device: torch.device):
        """Yields load(bs, be) -> (be - bs, d) tensor of index rows on device.

        Reads through _FbinBlockLoader when the index is file-backed, else
        slices all_embeddings. One loader per thread: call inside the thread.
        """
        if self._fbin_path is None:
            emb = self.all_embeddings
            yield lambda bs, be: emb[bs:be].to(device)
            return
        loader = _FbinBlockLoader(
            self._fbin_path, self.all_embeddings.shape[1], block_rows, device
        )
        try:
            yield loader
        finally:
            loader.close()

    def num_vectors(self) -> int:
        return self.n_vectors

    def indexed_accessions(self) -> list[str]:
        return self.acc_names_flat

    def save(self, output_path: Path):
        print(f"Index already written to {output_path} during build.")

    def index_size_gb(self, index_path: Path):
        total = 0
        if self.use_ivf or self.use_ivfpq:
            # The in-memory index (codes + ids + centroids); the rerank reads
            # the fbin from disk and is not counted, like a DB's raw store.
            return sum(f.stat().st_size for f in self.ivf_index_files) / (1024**3)
        if self.use_rabitq:
            rabitq_dir = index_path / "rabitq"
            assert rabitq_dir.exists()
            total += sum(f.stat().st_size for f in rabitq_dir.iterdir()) / (1024**3)
        else:
            meta_gb = (index_path / "meta.parquet").stat().st_size / (1024**3)
            embeds_gb = (index_path / "embeddings.fbin").stat().st_size / (1024**3)
            total = meta_gb + embeds_gb

        return total

    def construct_ann_index(self, index_path: Path):
        """Build a CAGRA graph over embeddings.fbin and serialize it.

        The whole base has to fit in GPU memory for the build; ``build_algo="ace"``
        is cuVS's out-of-core path if it stops fitting.
        """
        from cuvs.neighbors import cagra

        print("Constructing CAGRA index")
        base = _load_fbin_mmap(index_path / "embeddings.fbin")
        index = cagra.build(
            cagra.IndexParams(
                # Every encoder ends in normalize(), so all vectors are unit-norm and
                # maximum inner product is the same ranking DiskANN's "mips" gave.
                metric="inner_product",
                intermediate_graph_degree=CAGRA_INTERMEDIATE_GRAPH_DEGREE,
                graph_degree=CAGRA_GRAPH_DEGREE,
            ),
            base,
        )
        # Serialize the dataset alongside the graph so load() is self-contained --
        # search needs the vectors on device to score candidates.
        cagra.save(str(index_path / CAGRA_INDEX_FILE), index, include_dataset=True)

    @staticmethod
    def merge_shards(
        index_path: Path,
        num_nodes: int,
        use_rabitq: bool = False,
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

        if use_rabitq:
            print("Building RaBitQ quantized index from merged embeddings...")
            build_rabitq_index(index_path / "embeddings.fbin", index_path / "rabitq")

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
        max_len = self.model_cfg.max_seq_len
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
        """Encode on self.model, or split across query_embed_gpus replicas.

        Replicas (DenseEncoder on cuda:1..n-1, same checkpoint) are built on
        first use; each takes a contiguous slice and results are concatenated
        in order on the CPU.
        """
        ivfpq = getattr(self, "ivfpq", None)
        if ivfpq is not None and ivfpq.has_encoder:
            return torch.from_numpy(ivfpq.embed(chunks))
        n_gpu = min(getattr(self.model_cfg, "query_embed_gpus", 1), torch.cuda.device_count())
        if n_gpu <= 1 or len(chunks) < 2 * n_gpu:
            return self.model.encode(chunks)
        if self._query_replicas is None:
            replicas = [self.model]
            for i in range(1, n_gpu):
                rcfg = copy.copy(self.model_cfg)
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
    chunk_type: Literal["stride", "exact_chunk"],
    contig_align_intervals: dict[str, list[tuple[int, int]]] | None,
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
    elif chunk_type == "exact_chunk":
        assert contig_align_intervals
        align_intervals = contig_align_intervals.get(contig_id, [])
        chunks = []
        covered = []

        for iv_start, iv_end in align_intervals:
            # chunk must start early enough to reach iv_start, and late enough to cover iv_end
            lo = max(0, iv_end - chunk_size)
            hi = min(iv_start, len(seq) - chunk_size)
            chunk_start = random.randint(lo, max(lo, hi))
            chunks.append(seq[chunk_start : chunk_start + chunk_size])
            covered.append((chunk_start, chunk_start + chunk_size))

        # Merge covered intervals to find uncovered regions
        covered.sort()
        merged: list[tuple[int, int]] = []
        for s, e in covered:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        uncovered_regions: list[tuple[int, int]] = []
        prev = 0
        for s, e in merged:
            if prev < s:
                uncovered_regions.append((prev, s))
            prev = e
        if prev < len(seq):
            uncovered_regions.append((prev, len(seq)))

        # Evenly chunk each uncovered region, including any leftover
        for r_start, r_end in uncovered_regions:
            for i in range(r_start, r_end, chunk_size):
                chunks.append(seq[i : i + chunk_size])
    else:
        raise ValueError(f"Incorrect chunk_type recieved: {chunk_type}")

    return chunks
