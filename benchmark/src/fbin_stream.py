"""Stream an fbin through a per-row transform into a new fbin-layout file.

Shared by the engines that keep every encoder row but store it differently
(pca: projected, fp32 / fp16 / bf16; quantized_exact: fp16 / bf16). Rows keep
their encoder-fbin ids, so ExactEngine's scan over the output needs no
remapping.

The output is created once at full size (``create_fbin_file``) and then
filled by row range (``transform_rows``), so disjoint ranges can be written
by different threads, devices or nodes: positional writes into one
preallocated file need no coordination beyond who writes which rows.
"""

import os
import threading
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


def fbin_file_bytes(n: int, cols: int, dtype: torch.dtype) -> int:
    return FBIN_HEADER_BYTES + n * cols * dtype.itemsize


def create_fbin_file(path: Path, n: int, cols: int, dtype: torch.dtype) -> None:
    """Create (truncating) ``path`` as an fbin-layout file of n x cols
    ``dtype`` values: the uint32 [n, cols] header, then a hole of the full
    data size for transform_rows to fill. On Lustre the file takes its
    directory's default striping."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o664)
    try:
        pwrite_all(fd, 0, np.array([n, cols], dtype=np.uint32))
        os.ftruncate(fd, fbin_file_bytes(n, cols, dtype))
        os.fsync(fd)
    finally:
        os.close(fd)


def transform_rows(
    src_path: Path,
    out_path: Path,
    transform: Callable[[torch.Tensor], torch.Tensor],
    out_cols: int,
    out_dtype: torch.dtype,
    device: str,
    start: int,
    end: int,
    block_rows: int = BLOCK_ROWS,
    progress: Callable[[int, float, float, float], None] | None = None,
) -> tuple[float, float, float]:
    """Write ``transform(rows)`` for rows [start, end) of the fp32 fbin at
    ``src_path`` into the same rows of ``out_path``, an existing file from
    create_fbin_file. Returns seconds (waiting on reads, transforming,
    writing); ``progress(rows, wait, transform, write)`` is called per block.

    ``transform`` maps an (m, d) fp32 block on ``device`` to an
    (m, out_cols) ``out_dtype`` tensor. The output is fsynced before return,
    so a caller may record the range as done.

    Double-buffered: a reader thread preads block i + 1 into one host buffer
    while block i, in the other, is transformed and pwritten. The output goes
    out in large pwrites instead of through a memmap, whose page-at-a-time
    faults ran at 0.19 GB/s on Lustre (login node, 2026-10-08).
    """
    n_src, d = _load_fbin_mmap(src_path).shape
    if not 0 <= start <= end <= n_src:
        raise ValueError(f"rows [{start}, {end}) outside [0, {n_src}]")
    expect = fbin_file_bytes(n_src, out_cols, out_dtype)
    if os.path.getsize(out_path) != expect:
        raise ValueError(
            f"{out_path} is {os.path.getsize(out_path)} bytes, expected {expect} "
            f"for {n_src} x {out_cols} {out_dtype}: create it with create_fbin_file"
        )
    row_bytes = out_cols * out_dtype.itemsize
    blocks = [(bs, min(bs + block_rows, end)) for bs in range(start, end, block_rows)]
    if not blocks:
        return 0.0, 0.0, 0.0
    bufs = [
        torch.empty(
            (min(block_rows, end - start), d),
            dtype=torch.float32,
            pin_memory=device != "cpu",
        )
        for _ in range(2)
    ]

    def read_block(i: int) -> None:
        bs, be = blocks[i]
        read_fbin_rows(src_fd, bs, be, bufs[i % 2].numpy())

    t_wait = t_proj = t_write = 0.0
    src_fd = os.open(src_path, os.O_RDONLY)
    out_fd = os.open(out_path, os.O_WRONLY)  # no O_TRUNC: other rows are others'
    try:
        with ThreadPoolExecutor(1) as reader:
            pending = reader.submit(read_block, 0)
            for i, (bs, be) in enumerate(blocks):
                t0 = time.time()
                pending.result()
                if i + 1 < len(blocks):  # into the other buffer
                    pending = reader.submit(read_block, i + 1)
                t1 = time.time()
                y = transform(bufs[i % 2][: be - bs].to(device)).cpu()
                if y.shape != (be - bs, out_cols) or y.dtype != out_dtype:
                    raise ValueError(
                        f"transform gave {tuple(y.shape)} {y.dtype}, want "
                        f"{(be - bs, out_cols)} {out_dtype}"
                    )
                t2 = time.time()
                pwrite_all(
                    out_fd,
                    FBIN_HEADER_BYTES + bs * row_bytes,
                    y.contiguous().view(torch.uint8).numpy(),
                )
                t3 = time.time()
                t_wait += t1 - t0
                t_proj += t2 - t1
                t_write += t3 - t2
                if progress is not None:
                    progress(be - bs, t1 - t0, t2 - t1, t3 - t2)
        os.fsync(out_fd)
    finally:
        os.close(src_fd)
        os.close(out_fd)
    return t_wait, t_proj, t_write


def transform_rows_parallel(
    src_path: Path,
    out_path: Path,
    make_transform: Callable[[str], Callable[[torch.Tensor], torch.Tensor]],
    out_cols: int,
    out_dtype: torch.dtype,
    devices: list[str],
    start: int,
    end: int,
    desc: str,
    block_rows: int = BLOCK_ROWS,
) -> None:
    """transform_rows over [start, end) split into one contiguous sub-range
    per device, each in its own thread with ``make_transform(device)``.

    Each thread reads its own region of the source, so a node streams as many
    regions as it has GPUs; on Lustre that is what lifts a node's read rate
    (one stream per GPU, fbin.pread_into). Progress is one shared bar.
    """
    n = end - start
    cuts = [start + n * i // len(devices) for i in range(len(devices) + 1)]
    lock = threading.Lock()
    totals = [0.0, 0.0, 0.0]
    pbar = tqdm(total=n, desc=desc, unit="vec", unit_scale=True)

    def progress(rows: int, wait: float, proj: float, write: float) -> None:
        with lock:
            totals[0] += wait
            totals[1] += proj
            totals[2] += write
            # summed over threads: per-thread seconds, not wall time
            pbar.set_postfix(
                wait_read=f"{totals[0]:.0f}s",
                transform=f"{totals[1]:.0f}s",
                write=f"{totals[2]:.0f}s",
            )
            pbar.update(rows)

    def run(i: int) -> None:
        transform_rows(
            src_path,
            out_path,
            make_transform(devices[i]),
            out_cols,
            out_dtype,
            devices[i],
            cuts[i],
            cuts[i + 1],
            block_rows,
            progress,
        )

    t0 = time.time()
    try:
        with ThreadPoolExecutor(len(devices)) as pool:
            for f in [pool.submit(run, i) for i in range(len(devices))]:
                f.result()  # re-raises the first failure
    finally:
        pbar.close()
    print(
        f"Wrote rows [{start:,}, {end:,}) x {out_cols} {out_dtype} to "
        f"{out_path.name} on {len(devices)} device(s) in {time.time() - t0:.0f}s "
        f"(thread-seconds: waited on reads {totals[0]:.0f}, transform "
        f"{totals[1]:.0f}, write {totals[2]:.0f})"
    )


def transform_fbin(
    src_path: Path,
    out_path: Path,
    transform: Callable[[torch.Tensor], torch.Tensor],
    out_cols: int,
    out_dtype: torch.dtype,
    device: str,
    desc: str,
    block_rows: int = BLOCK_ROWS,
) -> None:
    """Create ``out_path`` and write ``transform(rows)`` for every row of the
    fp32 fbin at ``src_path``, on one device (see transform_rows)."""
    n_src = _load_fbin_mmap(src_path).shape[0]
    create_fbin_file(out_path, n_src, out_cols, out_dtype)
    transform_rows_parallel(
        src_path,
        out_path,
        lambda _dev: transform,
        out_cols,
        out_dtype,
        [device],
        0,
        n_src,
        desc,
        block_rows,
    )
