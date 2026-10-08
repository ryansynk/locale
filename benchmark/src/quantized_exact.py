"""Exact top-k over a 2-byte copy of the encoder fbin: fp16 or bf16 storage,
half the bytes of the fp32 fbin, scanned by ExactEngine unchanged.

``build`` streams the encoder fbin once and writes every row rounded to the
configured dtype (round-to-nearest-even, torch's .to) under the same uint32
[n, d] header, into the index's own directory. ``load`` hands that file to
ExactEngine.load_file, whose block loader reads the 2-byte rows (half the
read and host-to-device bytes) and upcasts each block to fp32 on the device
before the matmul, so the rounding of the stored values is the only
difference from the exact scan. Rows keep their encoder-fbin ids, so
vec_range sharding and the hits cache work as for exact.

fp16 has 10 mantissa bits to bf16's 7, but a far narrower range (normal
values ~6e-5 to 65504; bf16 keeps fp32's): for unit-norm embeddings
(|x_i| <= 1) the top of the range never binds and fp16 is the more precise
of the two. A value fp16
cannot hold fails the build instead of being stored as inf.
"""

from pathlib import Path

import numpy as np
import torch

from .config import ExactIndex, QuantizedExactIndex
from .exact import BLOCK_ROWS, ExactEngine
from .fbin import _load_fbin_mmap
from .fbin_stream import transform_fbin

TORCH_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def quantized_file(dtype: str) -> str:
    return f"embeddings.{dtype}bin"


def round_to(x: torch.Tensor, dtype: str) -> torch.Tensor:
    """``x`` rounded to ``dtype`` ("fp16" / "bf16"; torch's round-to-nearest-
    even). A value the dtype cannot hold raises instead of becoming inf."""
    y = x.to(TORCH_DTYPES[dtype])
    if not torch.isfinite(y).all():
        bad = x[~torch.isfinite(y)].abs().max().item()
        raise ValueError(f"{dtype} cannot hold |x| = {bad:g}")
    return y


class QuantizedExactEngine:
    SHARDABLE = True  # topk_hits accepts vec_range (multi-node row sharding)
    WORKER_EMBED = False

    def __init__(self, cfg: QuantizedExactIndex):
        if cfg.dtype not in TORCH_DTYPES:
            raise ValueError(
                f"dtype must be one of {list(TORCH_DTYPES)}, got {cfg.dtype}"
            )
        self.cfg = cfg
        self.dtype = TORCH_DTYPES[cfg.dtype]
        self.exact: ExactEngine | None = None

    def build(
        self, fbin_dir: Path, index_dir: Path, shard: int, num_shards: int
    ) -> None:
        if num_shards > 1:
            raise ValueError(
                f"{type(self).__name__} build is single-process (num_shards must be 1)"
            )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        src_path = fbin_dir / "embeddings.fbin"
        d = _load_fbin_mmap(src_path).shape[1]
        dtype = self.cfg.dtype
        index_dir.mkdir(parents=True, exist_ok=True)
        transform_fbin(
            src_path,
            index_dir / quantized_file(self.cfg.dtype),
            lambda x: round_to(x, dtype),
            d,
            self.dtype,
            device,
            desc=f"Quantizing to {self.cfg.dtype}",
        )

    def load(
        self, fbin_dir: Path, index_dir: Path, devices: list[str], encoder_cfg=None
    ) -> None:
        self.exact = ExactEngine(ExactIndex(top_k=self.cfg.top_k))
        self.exact.load_file(
            index_dir / quantized_file(self.cfg.dtype), devices, self.dtype
        )
        n_src, d = _load_fbin_mmap(fbin_dir / "embeddings.fbin").shape
        if tuple(self.exact.all_embeddings.shape) != (n_src, d):
            raise ValueError(
                f"{index_dir}: {quantized_file(self.cfg.dtype)} is "
                f"{tuple(self.exact.all_embeddings.shape)}, encoder fbin is "
                f"{(n_src, d)}"
            )

    @torch.no_grad()
    def topk_hits(
        self,
        query_vecs: np.ndarray,
        top_k: int,
        vec_range: tuple[int, int] | None = None,
        block_rows: int = BLOCK_ROWS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """ExactEngine.topk_hits over the quantized rows: fp32 queries against
        each block upcast to fp32, so the scores are exact inner products with
        the rounded vectors."""
        assert self.exact is not None
        return self.exact.topk_hits(query_vecs, top_k, vec_range, block_rows)

    def size_gb(self, fbin_dir: Path, index_dir: Path) -> float:
        """The quantized vectors, in GiB like ExactEngine."""
        path = index_dir / quantized_file(self.cfg.dtype)
        return path.stat().st_size / (1024**3)
