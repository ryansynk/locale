"""Stream an fbin through a per-row transform into a new fbin-layout file.

Shared by the engines that keep every encoder row but store it differently
(pca: projected fp32; quantized_exact: fp16 / bf16). Rows keep their
encoder-fbin ids, so ExactEngine's scan over the output needs no remapping.
"""

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .exact import BLOCK_ROWS
from .fbin import FBIN_HEADER_BYTES, _load_fbin_mmap, read_fbin_rows


def pwrite_all(fd: int, offset: int, arr: np.ndarray, piece_bytes: int = 64 << 20):
    """Write the C-contiguous ``arr`` at ``offset`` of ``fd`` in positional
    writes of ``piece_bytes``, retrying short writes (the write-side twin of
    fbin.pread_into)."""
    mv = memoryview(np.ascontiguousarray(arr)).cast("B")
    done = 0
    while done < len(mv):
        got = os.pwrite(fd, mv[done : done + piece_bytes], offset + done)
        if got <= 0:
            raise IOError(f"short write at byte {offset + done}")
        done += got


def transform_fbin(
    src_path: Path,
    out_path: Path,
    transform: Callable[[torch.Tensor], torch.Tensor],
    out_cols: int,
    device: str,
    desc: str,
    block_rows: int = BLOCK_ROWS,
) -> None:
    """Write ``transform(rows)`` for every row of the fp32 fbin at ``src_path``
    to ``out_path``: a uint32 [n, out_cols] header, then the values in the
    dtype ``transform`` returns (raw bytes, so bfloat16 works).

    ``transform`` maps an (m, d) fp32 block on ``device`` to an (m, out_cols)
    tensor; every block must come back in the same dtype.

    Double-buffered: a reader thread preads block i + 1 into one host buffer
    while block i, in the other, is transformed and pwritten. The output goes
    out in large pwrites instead of through a memmap, whose page-at-a-time
    faults ran at 0.19 GB/s on Lustre (login node, 2026-10-08).
    """
    n_src, d = _load_fbin_mmap(src_path).shape
    blocks = [(bs, min(bs + block_rows, n_src)) for bs in range(0, n_src, block_rows)]
    bufs = [
        torch.empty(
            (min(block_rows, n_src), d),
            dtype=torch.float32,
            pin_memory=device != "cpu",
        )
        for _ in range(2)
    ]

    def read_block(i: int) -> None:
        bs, be = blocks[i]
        read_fbin_rows(src_fd, bs, be, bufs[i % 2].numpy())

    # Postfix: time the main thread spent waiting on the reader, transforming
    # and writing, so the log shows which side is the bottleneck.
    t_wait = t_proj = t_write = 0.0
    row_bytes = None  # from the first block's output dtype
    pbar = tqdm(total=n_src, desc=desc, unit="vec", unit_scale=True)
    src_fd = os.open(src_path, os.O_RDONLY)
    out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
    try:
        pwrite_all(out_fd, 0, np.array([n_src, out_cols], dtype=np.uint32))
        with ThreadPoolExecutor(1) as reader:
            pending = reader.submit(read_block, 0)
            for i, (bs, be) in enumerate(blocks):
                t0 = time.time()
                pending.result()
                if i + 1 < len(blocks):  # into the other buffer
                    pending = reader.submit(read_block, i + 1)
                t1 = time.time()
                y = transform(bufs[i % 2][: be - bs].to(device)).cpu()
                if y.shape != (be - bs, out_cols):
                    raise ValueError(
                        f"transform gave {tuple(y.shape)}, want {(be - bs, out_cols)}"
                    )
                if row_bytes is None:
                    row_bytes = out_cols * y.element_size()
                elif out_cols * y.element_size() != row_bytes:
                    raise ValueError("transform changed dtype between blocks")
                t2 = time.time()
                pwrite_all(
                    out_fd,
                    FBIN_HEADER_BYTES + bs * row_bytes,
                    y.contiguous().view(torch.uint8).numpy(),
                )
                t_wait += t1 - t0
                t_proj += t2 - t1
                t_write += time.time() - t2
                pbar.set_postfix(
                    wait_read=f"{t_wait:.0f}s",
                    transform=f"{t_proj:.0f}s",
                    write=f"{t_write:.0f}s",
                )
                pbar.update(be - bs)
    finally:
        os.close(src_fd)
        os.close(out_fd)
        pbar.close()
    print(
        f"Wrote {n_src:,} x {out_cols} to {out_path.name}: waited on reads "
        f"{t_wait:.0f}s, transform {t_proj:.0f}s, write {t_write:.0f}s"
    )
