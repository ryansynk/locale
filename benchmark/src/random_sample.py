"""Random-subsample baseline for the epsilon net: same artifact (net_ids.pt,
sorted global fbin rows) and same exact scan over the kept rows, but the rows
are a uniform random subset of each accession instead of a cover."""

from pathlib import Path

import polars as pl
import torch

from .config import RandomSampleIndex
from .epsilonnet import EpsilonNetEngine


class RandomSampleEngine(EpsilonNetEngine):
    """load / topk_hits / size_gb are EpsilonNetEngine's; only build differs.
    self.cfg is a RandomSampleIndex."""

    cfg: RandomSampleIndex

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        # Needs only meta.parquet: the choice of rows never looks at the vectors.
        if num_shards > 1:
            raise ValueError("randomsample build is single-node (num_shards must be 1)")
        meta = pl.read_parquet(fbin_dir / "meta.parquet")
        gen = torch.Generator().manual_seed(self.cfg.seed)
        kept = []
        for start, n in zip(meta["start_row"].to_list(), meta["num_rows"].to_list()):
            k = max(1, round(n / self.cfg.compression_ratio)) if n else 0
            kept.append(torch.randperm(n, generator=gen)[:k] + start)
        index_dir.mkdir(parents=True, exist_ok=True)
        ids = torch.cat(kept).sort().values
        torch.save(ids, index_dir / "net_ids.pt")
        print(f"kept {len(ids):,} of {sum(meta['num_rows']):,} vectors")
