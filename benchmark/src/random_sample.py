"""Random-subsample baseline for the epsilon net: same artifact (net_ids.pt +
the kept rows' own embeddings.fbin) and same exact scan over the kept rows,
but the rows are a uniform random subset of each accession instead of a
cover."""

from pathlib import Path

import polars as pl
import torch

from .config import RandomSampleIndex
from .epsilonnet import EpsilonNetEngine


class RandomSampleEngine(EpsilonNetEngine):
    """Everything but the row choice is EpsilonNetEngine's.
    self.cfg is a RandomSampleIndex."""

    cfg: RandomSampleIndex

    def _select(self, fbin_dir: Path, meta: pl.DataFrame) -> torch.Tensor:
        # Needs only meta.parquet: the choice of rows never looks at the vectors.
        gen = torch.Generator().manual_seed(self.cfg.seed)
        kept = []
        for start, n in zip(meta["start_row"].to_list(), meta["num_rows"].to_list()):
            k = max(1, round(n / self.cfg.compression_ratio)) if n else 0
            kept.append(torch.randperm(n, generator=gen)[:k] + start)
        return torch.cat(kept)
