"""Merged epsilon-net index: one greedy net over every accession's own net
(src/epsilonnet.py, built at per_accession_epsilon), so accessions that share
sequence share centers.

Artifact in the index directory:
    embeddings.fbin  the merged centers, in greedy_net's order
    counts.pt        sparse CSR (n_centers, n_accessions): counts[c, a] = how
                     many of accession a's level-1 centers center c covers
    net_ids.pt       the centers' encoder-fbin rows (provenance; not read by
                     search)
    level1/          the per-accession net, only when no finished sibling
                     epsilonnet index at per_accession_epsilon existed

Search is ExactEngine over the centers; each hit center is expanded into the
accessions in its row of counts, an accession scores its best hit center, and
each query keeps its top_k accessions. An accession is returned as its first
encoder-fbin row, so the accession regrouping downstream is unchanged.
"""

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import torch

from .config import (
    CONFIG_FILE,
    DONE_FILE,
    EpsilonNetIndex,
    ExactIndex,
    MergedEpsilonNetIndex,
)
from .epsilonnet import NET_FBIN, NET_IDS, EpsilonNetEngine
from .exact import ExactEngine
from .fbin import _create_fbin_memmap, _load_fbin_mmap, read_fbin_rows
from .greedy_net import greedy_net

COUNTS = "counts.pt"  # sparse CSR (n_centers, n_accessions)


class MergedEpsilonNetEngine:
    # Multi-node row sharding would split the encoder fbin's row count
    # (run_benchmark), not the net's; one node's GPUs scan the net instead.
    SHARDABLE = False
    WORKER_EMBED = False

    def __init__(self, cfg: MergedEpsilonNetIndex):
        self.cfg = cfg
        self.exact: ExactEngine | None = None
        self.counts: torch.Tensor | None = None  # sparse CSR, see module docstring
        self.starts: torch.Tensor | None = None  # each accession's first encoder row

    def _level1(self, fbin_dir: Path, index_dir: Path) -> Path:
        """The per-accession net this index merges: the first finished
        epsilonnet index at per_accession_epsilon over the same encoder fbin,
        else a private one built (once) under index_dir/level1."""
        want = {"engine": "epsilonnet", "epsilon": self.cfg.per_accession_epsilon}
        for cfg_file in sorted(fbin_dir.glob(f"*/{CONFIG_FILE}")):
            d = cfg_file.parent
            if json.loads(cfg_file.read_text()) == want and (d / DONE_FILE).exists():
                print(f"level 1: reusing {d}")
                return d
        d = index_dir / "level1"
        if not (d / DONE_FILE).exists():
            level1_cfg = EpsilonNetIndex(epsilon=self.cfg.per_accession_epsilon)
            EpsilonNetEngine(level1_cfg).build(fbin_dir, d, 0, 1)
            (d / DONE_FILE).touch()
        return d

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        if num_shards > 1:
            raise ValueError(
                f"{type(self).__name__} build is single-process (num_shards must "
                "be 1); it already uses every visible GPU"
            )
        meta = pl.read_parquet(fbin_dir / "meta.parquet")
        level1 = self._level1(fbin_dir, index_dir)

        # Level-1 centers (every accession's own net): their encoder-fbin rows
        # and their vectors, all in host memory.
        l1_rows = torch.load(level1 / NET_IDS).numpy()
        path = level1 / NET_FBIN
        d = _load_fbin_mmap(path).shape[1]
        x = np.empty((len(l1_rows), d), dtype=np.float32)
        fd = os.open(path, os.O_RDONLY)
        try:
            read_fbin_rows(fd, 0, len(l1_rows), x)
        finally:
            os.close(fd)

        # Level 2: one greedy net over all level-1 centers. x stays in host
        # memory; the GPU holds only the merged centers and one batch.
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=dev).manual_seed(0)
        ids, assign, sim = greedy_net(
            torch.from_numpy(x),
            self.cfg.epsilon,
            generator=gen,
            device=dev,
            progress=True,
        )
        ids, assign, sim = ids.cpu().numpy(), assign.cpu(), sim.cpu()

        # Centers in greedy_net's order: row c of embeddings.fbin is center c,
        # the index assign already uses. net_ids.pt (provenance only, not
        # read by search) is their encoder-fbin rows in that order, unsorted.
        index_dir.mkdir(parents=True, exist_ok=True)
        out = _create_fbin_memmap(index_dir / NET_FBIN, len(ids), d)
        out[:] = x[ids]
        out.flush()
        del out
        torch.save(torch.from_numpy(l1_rows[ids]), index_dir / NET_IDS)

        # counts[c, a]: accession a's level-1 centers assigned to center c.
        # coalesce sums the duplicate (c, a) entries.
        starts = meta["start_row"].to_numpy()
        acc = torch.from_numpy(np.searchsorted(starts, l1_rows, side="right") - 1)
        counts = (
            torch.sparse_coo_tensor(
                torch.stack([assign, acc]),
                torch.ones(len(assign), dtype=torch.int32),
                (len(ids), len(starts)),
            )
            .coalesce()
            .to_sparse_csr()
        )
        torch.save(counts, index_dir / COUNTS)

        n = int(meta["num_rows"].sum())
        m = max(len(ids), 1)
        print(
            f"merged net: {len(ids):,} centers from {len(l1_rows):,} level-1 "
            f"centers ({len(l1_rows) / m:.3f}x), {n:,} vectors ({n / m:.3f}x "
            f"overall); {counts.values().numel():,} (center, accession) entries; "
            f"worst cover cosine {sim.min().item():.4f}"
        )

    def load(
        self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None
    ) -> None:
        self.exact = ExactEngine(ExactIndex(top_k=self.cfg.top_k))
        self.exact.load(index_dir, index_dir, devices)  # the centers' own fbin
        # Row c = row c of embeddings.fbin; column a = row a of meta.parquet.
        self.counts = torch.load(index_dir / COUNTS)
        meta = pl.read_parquet(fbin_dir / "meta.parquet")
        self.starts = torch.from_numpy(meta["start_row"].to_numpy().astype(np.int64))
        if self.counts.layout != torch.sparse_csr:
            raise ValueError(
                f"{index_dir / COUNTS}: expected sparse CSR, got {self.counts.layout}"
            )
        n_centers, n_acc = self.counts.shape
        if n_centers != self.exact.all_embeddings.shape[0]:
            raise ValueError(
                f"{index_dir}: {NET_FBIN} and {COUNTS} disagree on centers"
            )
        if n_acc != len(self.starts):
            raise ValueError(
                f"{index_dir / COUNTS}: {n_acc} accessions, meta.parquet has {len(self.starts)}"
            )

    def topk_hits(
        self,
        query_vecs: np.ndarray,
        top_k: int,
        vec_range: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.exact is not None
        scores, rows = self.exact.topk_hits(query_vecs, top_k)
        scores, rows = torch.from_numpy(scores), torch.from_numpy(rows)
        n_q, k = rows.shape
        crow = self.counts.crow_indices()
        acc_of = self.counts.col_indices()

        # Hit h = (query h // k, center c[h]) owns entries crow[c]:crow[c+1]
        # of acc_of, one per accession at that center; padding hits own none.
        c = rows.ravel().clamp(min=0)
        lens = torch.where(rows.ravel() >= 0, crow[c + 1] - crow[c], 0)
        hit = torch.repeat_interleave(torch.arange(n_q * k), lens)  # hit of each entry
        first = torch.cumsum(lens, 0) - lens  # where each hit's run starts
        entry = crow[c][hit] + torch.arange(len(hit)) - first[hit]
        acc = acc_of[entry]
        query = hit // k

        # Best score per (query, accession), then each query's top accessions.
        n_acc = len(self.starts)
        acc_score = torch.full((n_q * n_acc,), -torch.inf, dtype=scores.dtype)
        acc_score.scatter_reduce_(
            0, query * n_acc + acc, scores.ravel()[hit], reduce="amax"
        )
        top_s, top_a = acc_score.view(n_q, n_acc).topk(min(top_k, n_acc), dim=1)
        ids = torch.where(torch.isfinite(top_s), self.starts[top_a], -1)
        return top_s.numpy(), ids.numpy()

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        """The merged centers plus the center x accession count matrix, in GiB
        like ExactEngine."""
        files = (index_dir / NET_FBIN, index_dir / COUNTS)
        return sum(f.stat().st_size for f in files) / (1024**3)
