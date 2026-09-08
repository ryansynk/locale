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

# Unused until the bodies land: _load_fbin_mmap opens the source base and the
# centroids, _create_fbin_memmap writes centroids.fbin.
from .fbin import _create_fbin_memmap, _load_fbin_mmap  # noqa: F401

from cuvs.cluster.kmeans import KMeansParams, fit

# Files written into the index directory by build().
CENTROIDS_FILE = "centroids.fbin"
COUNTS_FILE = "counts.npy"
META_FILE = "meta.parquet"


class CentroidIndex(BaseIndex):
    def __init__(self, cfg: ExperimentConfig):
        assert isinstance(cfg.model, CentroidConfig)
        self.cfg = cfg
        self.model_cfg: CentroidConfig = cfg.model
        self.no_search: bool = cfg.no_search

        """
        encoder: DenseConfig = field(default_factory=DenseConfig)
        self.source_index_dir: Optional[str] = None
        self.num_centroids: int = cfg.num_centroids
        nprobe: int = 32
        sample_size: int = 1_000_000
        kmeans_iters: int = 25
        # Rows per streaming tile in the assignment pass. 500k x 768 float32 ~ 1.5 GB,
        # matching the tile size vecdb_dataset/ground_truth.py settled on for the same
        # NFS-read-bound scan.
        tile_rows: int = 500_000
        probe_weight: Literal["sim", "softmax", "uniform"] = "sim"
        softmax_temperature: float = 0.05
        random_seed: int = 0
        """

        # Populated by load(): the searchable state.
        self.centroids: torch.Tensor | None = None  # (K, d) unit-norm
        self.log_counts: torch.Tensor | None = None  # (n_acc, K) float32
        self.acc_names_flat: list[str] = []

        if not self.no_search:
            self.model = DenseEncoder(self.model_cfg.encoder)

    def load(self, index_path: Path):
        self.centroids = torch.load(index_path / "centroids.pt")
        self.log_counts = torch.load(index_path / "log_counts.pt")

        meta = pl.read_parquet(index_path / "meta.parquet")
        self.acc_names_flat = meta["srr_id"].to_list()
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        self.acc_offsets = starts + [starts[-1] + counts[-1]] if starts else [0]
        mmap = _load_fbin_mmap(index_path / "embeddings.fbin")
        self._mmap = mmap  # keep reference to prevent GC closing the mapping
        self.all_embeddings = torch.from_numpy(mmap)
        print(
            f"Loaded {len(starts)} accessions ({mmap.shape[0]} vectors) [memory-mapped]"
        )

    def build(self, accessions: list[Path], index_path: Path):
        # if you pass in the embeddings already in construction, then dont rebuild them
        if self.model_cfg.source_index_dir:
            embeddings = _load_fbin_mmap(
                Path(self.model_cfg.source_index_dir) / "embeddings.fbin"
            )
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

        device = self.model_cfg.encoder.device
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

        meta = pl.read_parquet(index_path / "meta.parquet")
        acc_names_flat = meta["srr_id"].to_list()
        starts = meta["start_row"].to_list()
        counts = meta["num_rows"].to_list()
        acc_offsets = starts + [starts[-1] + counts[-1]] if starts else [0]

        counts = self._accumulate_counts(embeddings, centroids, acc_offsets)
        assert counts.sum() == n, f"assigned {counts.sum()} of {n} vectors"
        # Populated by load(): the searchable state.
        self.centroids = centroids
        self.log_counts = np.log1p(counts).astype(np.float32)
        self.acc_names_flat = acc_names_flat

    def search(self, queries: pl.DataFrame) -> pl.DataFrame:
        assert (
            queries.filter(
                pl.col("query_sequence").str.len_chars() > self.model_cfg.max_len
            ).height
            == 0
        ), "CURRENT IMPLEMENTATION DOES NOT SUPPORT QUERIES LONGER THAN max_len"
        assert self.centroids is not None
        assert self.log_counts is not None

        queries = queries.with_row_index()
        query_features, query_indices = self._embed_queries(queries)
        query_features = query_features.to(self.model_cfg.device)
        centroids = self.centroids.to(query_features.device)

        vals, indexes = torch.topk(
            query_features @ centroids.T, k=self.model_cfg.nprobe
        )  # (num_queries, nprobe)

        # This loop definitely doesn't need to exist
        query_acc_scores = []
        for q_idx in range(vals.shape[0]):
            v = vals[q_idx, :]
            i = indexes[q_idx, :]
            query_acc_scores.append(
                torch.dot(v, self.log_counts[:, i])
            )  # score for one query over all accessions
        query_acc_scores = torch.cat(query_acc_scores, dim=1)  # (num_queries, num_accs)

        scores_df = []
        for i in range(query_acc_scores.shape[0]):
            for j in range(query_acc_scores.shape[1]):
                scores_df.append(
                    {
                        "query_idx": i,
                        "accession": self.acc_names_flat[j],
                        "score": float(query_acc_scores[i, j]),
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
        self.acc_names_flat
        self.centroids
        self.log_counts

    def index_size_gb(self, index_path: Path):
        pass

    def _accumulate_counts(self, base, centroids, acc_offsets):
        """Assign every base vector to a centroid -> raw counts ``(n_acc, K)``.

        Streams the base in ``tile_rows`` tiles, read-bound rather than
        compute-bound: the matmul is ~1 PFLOP against ~500 GB of NFS reads.
        """
        device = self.model_cfg.encoder.device
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
        staging = torch.empty((tile_rows, d), dtype=torch.float32, pin_memory=True)
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
            chunked_query = [
                query[i : (i + self.model_cfg.max_seq_len)]
                for i in range(0, len(query), self.model_cfg.max_seq_len)
            ]
            num_chunks = len(chunked_query)
            query_indices.append((prev_idx, prev_idx + num_chunks))
            query_chunks.extend(chunked_query)
            prev_idx = prev_idx + num_chunks

        query_features = self.model.encode(query_chunks)
        return query_features, query_indices
