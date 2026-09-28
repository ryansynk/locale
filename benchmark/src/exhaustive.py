"""Reference protocol: every accession scored by the max over ALL its vectors.

Not a top-k engine -- there are no vector hits to persist -- so ``topk_hits``
raises and the scoring entry point is ``slot_scores``: per (query, strand)
slot and accession, the sum over the slot's chunks of each chunk's max inner
product against the accession's rows. DenseIndex reduces slots to queries
(sum over a long query's chunks, max over strands).

Accessions are dealt round-robin to one thread per device; each accession
streams through its device in blocks (the largest ones, ~30M+ vectors, 90+ GB
fp32, do not fit a 40 GB A100 as one slice) and maximum() over block maxima
equals the full max. GPU ops and the pread block loads release the GIL, so
the threads also overlap the multi-TB index read.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import ExhaustiveIndex
from .exact import BLOCK_ROWS, block_loader
from .fbin import _load_fbin_mmap


class ExhaustiveEngine:
    SHARDABLE = False  # run_benchmark shards the exhaustive scan by accession, not by row
    WORKER_EMBED = False

    def __init__(self, cfg: ExhaustiveIndex):
        self.cfg = cfg
        self.all_embeddings: torch.Tensor | None = None
        self.fbin_path: Path | None = None
        self.devices: list[str] = ["cpu"]

    def build(self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int) -> None:
        pass

    def load(self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None) -> None:
        mmap = _load_fbin_mmap(fbin_dir / "embeddings.fbin")
        self._mmap = mmap
        self.all_embeddings = torch.from_numpy(mmap)
        self.fbin_path = fbin_dir / "embeddings.fbin"
        self.devices = list(devices)
        print(f"Loaded {mmap.shape[0]:,} vectors [memory-mapped]")

    def topk_hits(self, query_vecs, top_k, vec_range=None):
        raise NotImplementedError("the exhaustive protocol has no vector hits")

    @torch.no_grad()
    def slot_scores(
        self,
        query_vecs: np.ndarray,
        chunk_to_slot: np.ndarray,
        n_slots: int,
        acc_offsets: list[int],
        acc_indices: list[int],
        block_rows: int = BLOCK_ROWS,
    ) -> np.ndarray:
        """(n_slots, len(acc_indices)) scores; column j is accession acc_indices[j]."""
        assert self.all_embeddings is not None
        devices = self.devices
        qcf_cpu = torch.as_tensor(np.ascontiguousarray(query_vecs, dtype=np.float32))
        c2s_cpu = torch.as_tensor(np.asarray(chunk_to_slot, dtype=np.int64))
        n_chunks = len(qcf_cpu)
        n_acc = len(acc_indices)
        # Threads write disjoint columns, so unsynchronized writes are safe
        scores_cpu = np.zeros((n_slots, n_acc), dtype=np.float32)

        def _score_accessions(dev_idx: int):
            dev = torch.device(devices[dev_idx])
            q = qcf_cpu.to(dev)
            c2s = c2s_cpu.to(dev)
            positions = range(dev_idx, n_acc, len(devices))
            if dev_idx == 0:
                positions = tqdm(
                    positions, desc=f"Scoring accessions ({len(devices)} device(s))"
                )
            with block_loader(self.fbin_path, self.all_embeddings, block_rows, dev) as load_block:
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
                    slot = torch.zeros(n_slots, device=dev, dtype=q.dtype)
                    slot.scatter_add_(0, c2s, chunk_maxes)
                    scores_cpu[:, pos] = slot.cpu().numpy()

        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            # list() propagates any worker exception
            list(pool.map(_score_accessions, range(len(devices))))
        return scores_cpu

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        meta_gb = (fbin_dir / "meta.parquet").stat().st_size / (1024**3)
        embeds_gb = (fbin_dir / "embeddings.fbin").stat().st_size / (1024**3)
        return meta_gb + embeds_gb
