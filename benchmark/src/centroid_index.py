"""Centroid-count index: each accession is a bag of k-means centroids.

Build clusters a sample of read embeddings into ``num_centroids`` centroids, assigns
every base vector to its nearest one, and stores each accession as a length-K vector
of ``log1p`` centroid counts. Search embeds the query, probes the ``nprobe`` nearest
centroids, and scores an accession by the weighted sum of its log counts over those
centroids.

The index is two small arrays instead of the full base: at K=4096 and 500 accessions
that is ~12.6 MB of centroids plus ~8.2 MB of counts, against 502 GB of embeddings.

Two properties the implementation leans on:

- Every encoder ends in ``nn.functional.normalize(..., dim=1)``, so all vectors are
  unit-norm. Cosine similarity *is* the dot product, nearest-centroid *is*
  ``argmax(X @ C.T)``, and k-means is spherical k-means. No distance math anywhere.
- ``meta.parquet`` describes accessions as contiguous run-length row ranges, so a
  global row id maps to an accession with a single ``searchsorted`` over
  ``acc_offsets`` -- the idiom DenseIndex.search already uses.

Deliberately imports ``.encoders`` and ``.fbin`` rather than ``.dense_index``: this
index needs an encoder and an fbin reader, not diskannpy.
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from .base_index import BaseIndex
from .config import CentroidConfig, ExperimentConfig
from .encoders import DenseEncoder

from .fbin import _create_fbin_memmap, _load_fbin_mmap


class CentroidIndex(BaseIndex):
    # Files save() writes into the index directory and load() reads back.
    CENTROIDS_FILE = "centroids.fbin"
    COUNTS_FILE = "counts.npy"
    META_FILE = "meta.parquet"

    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, CentroidConfig)
        self.cfg = cfg
        self.model_cfg: CentroidConfig = cfg.model
        self.no_search: bool = cfg.no_search

        # Populated by build() or load(): the searchable state, all on the host.
        self.centroids: torch.Tensor | None = None  # (K, d) unit-norm
        self.counts: np.ndarray | None = None  # (n_acc, K) int64 raw assignments
        self.log_counts: torch.Tensor | None = None  # (n_acc, K) float32 log1p
        self.acc_names_flat: list[str] = []

        if not self.no_search:
            self.model = DenseEncoder(self.model_cfg.encoder)

    def load(self, index_path: Path):
        # np.array(...) copies the memmap into RAM: at K=4096 the centroids are
        # ~12 MB, and a torch tensor over a read-only memmap is not writable.
        self.centroids = torch.from_numpy(
            np.array(_load_fbin_mmap(index_path / self.CENTROIDS_FILE))
        )
        self.counts = np.load(index_path / self.COUNTS_FILE)
        self.log_counts = torch.from_numpy(np.log1p(self.counts).astype(np.float32))
        self.acc_names_flat = pl.read_parquet(index_path / self.META_FILE)[
            "srr_id"
        ].to_list()
        assert self.log_counts.shape == (
            len(self.acc_names_flat),
            self.centroids.shape[0],
        )
        print(
            f"Loaded {len(self.acc_names_flat)} accessions x "
            f"{self.centroids.shape[0]} centroids ({self.counts.sum()} vectors)"
        )

    def build(self, accessions: list[Path], index_path: Path):
        # if you pass in the embeddings already in construction, then dont rebuild them
        if self.model_cfg.source_index_dir:
            source_dir = Path(self.model_cfg.source_index_dir)
            embeddings = _load_fbin_mmap(source_dir / "embeddings.fbin")
            n, d = embeddings.shape
        else:
            # Manually generate embeddings, write them, etc
            raise NotImplementedError("No implementation of encoding yet")

        # select, without replacement, self.model_cfg.sample_size embeddings and load them
        # into memory as pytorch tensor
        k = min(self.model_cfg.sample_size, n)
        rng = np.random.default_rng(self.model_cfg.random_seed)
        if k >= n:
            row_ids = np.arange(n, dtype=np.int64)
        else:
            # rejection sampling to save memory. draw with replacement, then take unique, then resample
            row_ids = np.unique(rng.integers(0, n, size=k, dtype=np.int64))
            while row_ids.size < k:
                extra = rng.integers(0, n, size=2 * (k - row_ids.size), dtype=np.int64)
                row_ids = np.unique(np.concatenate([row_ids, extra]))
        row_ids = np.sort(rng.permutation(row_ids)[:k])
        sample = torch.from_numpy(embeddings[row_ids])

        from cuvs.cluster.kmeans import KMeansParams, fit

        device = self.model_cfg.device
        params = KMeansParams(
            n_clusters=self.model_cfg.num_centroids,
            max_iter=self.model_cfg.kmeans_iters,
            tol=1e-4,
        )
        # fit wants a CUDA array; sample is still on the host at this point.
        centroids, inertia, n_iter = fit(params, sample.to(device))
        # cuVS k-means is L2, so it returns cluster means whose norms vary, and it
        # returns them as a raft device_ndarray. Renormalizing makes
        # argmax(X @ C.T) the true nearest centroid -- the spherical k-means this
        # index assumes, and what search probes with.
        centroids = torch.nn.functional.normalize(
            torch.as_tensor(centroids, device=device), dim=1
        )  # (num_centroids, D)

        # The accession layout is the source index's: index_path does not exist
        # yet, and this index has no rows of its own. The meta is carried over in
        # save() so the built index describes its accessions on its own.
        self._source_meta = pl.read_parquet(source_dir / self.META_FILE)
        starts = self._source_meta["start_row"].to_list()
        num_rows = self._source_meta["num_rows"].to_list()
        acc_offsets = starts + [starts[-1] + num_rows[-1]] if starts else [0]
        assert acc_offsets[-1] == n, f"meta covers {acc_offsets[-1]} of {n} rows"

        counts = self._accumulate_counts(embeddings, centroids, acc_offsets)
        assert counts.sum() == n, f"assigned {counts.sum()} of {n} vectors"
        self.centroids = centroids.cpu()
        self.counts = counts
        self.log_counts = torch.from_numpy(np.log1p(counts).astype(np.float32))
        self.acc_names_flat = self._source_meta["srr_id"].to_list()

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        assert self.centroids is not None
        assert self.log_counts is not None

        queries = queries.with_row_index()
        query_chunk_features, query_indices = self._embed_queries(queries)
        device = self.model_cfg.device
        query_chunk_features = query_chunk_features.to(device)
        centroids = self.centroids.to(device)
        log_counts = self.log_counts.to(device)  # (n_acc, K)
        n_chunks = query_chunk_features.shape[0]
        n_queries = len(queries)

        sims, probe_idx = torch.topk(
            query_chunk_features @ centroids.T, k=self.model_cfg.nprobe
        )  # both (n_chunks, nprobe)
        weights = self._probe_weights(sims)

        # log_counts.T[probe_idx] gathers, per chunk, the (nprobe, n_acc) block of
        # log counts under its probed centroids; the einsum is the weighted sum
        # over probes. At 500 queries x 32 probes x 500 accessions this is 8M
        # floats -- tiny beside the dense index's per-accession matmuls.
        probed = log_counts.T[probe_idx]  # (n_chunks, nprobe, n_acc)
        chunk_scores = torch.einsum("cp,cpa->ca", weights, probed)  # (n_chunks, n_acc)

        # A query longer than max_seq_len owns several chunks; sum them, matching
        # DenseIndex.search's sum-of-chunk-scores reduction.
        chunk_to_query = torch.zeros(n_chunks, dtype=torch.long, device=device)
        for qi, (s, e) in enumerate(query_indices):
            chunk_to_query[s:e] = qi
        scores = torch.zeros(
            n_queries, chunk_scores.shape[1], device=device, dtype=chunk_scores.dtype
        )
        scores.index_add_(0, chunk_to_query, chunk_scores)  # (n_queries, n_acc)
        scores_cpu = scores.float().cpu().numpy()

        scores_df = []
        for i in range(scores_cpu.shape[0]):
            for j in range(scores_cpu.shape[1]):
                scores_df.append(
                    {
                        "query_idx": i,
                        "accession": self.acc_names_flat[j],
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

    def indexed_accessions(self) -> list[str]:
        return self.acc_names_flat

    def save(self, output_path: Path):
        assert self.centroids is not None and self.counts is not None
        output_path.mkdir(parents=True, exist_ok=True)
        centroids = self.centroids.cpu().numpy().astype(np.float32)
        out = _create_fbin_memmap(output_path / self.CENTROIDS_FILE, *centroids.shape)
        out[:] = centroids
        out.flush()
        # Raw counts, not log1p: they are exact integers and load() derives the
        # rest, so a different scoring transform later needs no rebuild.
        np.save(output_path / self.COUNTS_FILE, self.counts)
        self._source_meta.write_parquet(output_path / self.META_FILE)

    def index_size_gb(self, index_path: Path):
        return sum(
            (index_path / f).stat().st_size
            for f in (self.CENTROIDS_FILE, self.COUNTS_FILE, self.META_FILE)
        ) / (1024**3)

    def _probe_weights(self, sims: torch.Tensor) -> torch.Tensor:
        """Weight each probed centroid by ``probe_weight``: (n_chunks, nprobe)."""
        mode = self.model_cfg.probe_weight
        if mode == "sim":
            return sims
        if mode == "softmax":
            return torch.softmax(sims / self.model_cfg.softmax_temperature, dim=1)
        if mode == "uniform":
            return torch.ones_like(sims)
        raise ValueError(
            f"probe_weight expected sim, softmax, or uniform. Got = {mode}"
        )

    def _accumulate_counts(self, base, centroids, acc_offsets):
        """Assign every base vector to a centroid -> raw counts ``(n_acc, K)``.

        Streams the base in ``tile_rows`` tiles, read-bound rather than
        compute-bound: the matmul is ~1 PFLOP against ~500 GB of NFS reads.
        """
        device = self.model_cfg.device
        n, d = base.shape
        K = centroids.shape[0]
        n_acc = len(acc_offsets) - 1

        centroids_t = centroids.to(device).T.contiguous()  # (d, K)
        acc_offsets_gpu = torch.as_tensor(acc_offsets, device=device)
        # Flat (accession, centroid) histogram, kept on device for the whole scan:
        # n_acc * K is only ~2M bins at K=4096 and 500 accessions.
        counts = torch.zeros(n_acc * K, dtype=torch.int64, device=device)

        tile_rows = self.model_cfg.tile_rows
        # Bounds each similarity block at ~1 GB when K=4096; a full 500k-row tile
        # would otherwise allocate 8.2 GB in a single matmul.
        mm_chunk = 65_536

        # Basic slicing of a memmap is a view, so copyto faults the pages straight
        # into pinned memory with no intermediate array.
        # Pinning needs a CUDA allocator; on a CPU device it is neither possible
        # nor useful.
        staging = torch.empty(
            (tile_rows, d),
            dtype=torch.float32,
            pin_memory=torch.device(device).type == "cuda",
        )
        staging_np = staging.numpy()

        for start in tqdm(range(0, n, tile_rows), desc="Assigning centroids"):
            end = min(start + tile_rows, n)
            rows_read = end - start
            np.copyto(staging_np[:rows_read], base[start:end])
            tile = staging[:rows_read].to(device)

            for c0 in range(0, rows_read, mm_chunk):
                c1 = min(c0 + mm_chunk, rows_read)
                centroid_idx = (tile[c0:c1] @ centroids_t).argmax(dim=1)
                row_ids = torch.arange(start + c0, start + c1, device=device)
                acc_idx = torch.searchsorted(acc_offsets_gpu, row_ids, right=True) - 1
                counts += torch.bincount(
                    acc_idx * K + centroid_idx, minlength=n_acc * K
                )

        return counts.view(n_acc, K).cpu().numpy()

    def _embed_queries(self, queries) -> tuple[torch.Tensor, list[tuple[int, int]]]:
        """Embed every query, chunking any that exceed max_seq_len.

        Queries are split into non-overlapping max_seq_len windows, so a query
        shorter than that yields exactly one chunk identical to itself. Returns
        the (n_chunks, dim) features and, per query, the [start, end) range of
        rows it owns -- callers that need per-query results reduce over that
        range rather than assuming one row per query.
        """
        query_chunks: list[str] = []
        query_indices: list[tuple[int, int]] = []
        prev_idx = 0
        for query in queries["query_sequence"].to_list():
            max_seq_len = self.model_cfg.encoder.max_seq_len
            chunked_query = [
                query[i : (i + max_seq_len)] for i in range(0, len(query), max_seq_len)
            ]
            num_chunks = len(chunked_query)
            query_indices.append((prev_idx, prev_idx + num_chunks))
            query_chunks.extend(chunked_query)
            prev_idx = prev_idx + num_chunks

        query_features = self.model.encode(query_chunks)
        return query_features, query_indices
