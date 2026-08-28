import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_dc, Path_drw


def _load_fbin_mmap(path: Path) -> np.memmap:
    """Memory-map an fbin file and return a read-only float32 array."""
    with open(path, "rb") as f:
        n, d = np.frombuffer(f.read(8), dtype=np.uint32)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(int(n), int(d)))


def shard_fbin(embeds_dir: Path_drw, out: Path_dc, rows_per_shard: int = 325520):
    """Shards output fbin into multiple files

    Given output fbin of embeddings, creates shards... By default, rows_per_shard
    is selected to approximate 1GB per shard.
    """
    embeds_dir: Path = Path(embeds_dir)
    embeddings = embeds_dir / "embeddings.fbin"
    meta = embeds_dir / "meta.parquet"
    out: Path = Path(out)
    out.mkdir()

    embeddings_out = out / "base"
    embeddings_out.mkdir()
    mmap = _load_fbin_mmap(embeddings)
    n, d = mmap.shape

    # Check that meta file agrees with embeds
    written = pl.read_parquet(meta)["num_rows"].sum()
    if written != n:
        raise RuntimeError(
            f"{n - written} unwritten rows: header n={n}, meta={written}"
        )

    # Stream embeddings and write output shards
    num_vecs_written = 0
    num_shards_written = 0
    for i, start in enumerate(range(0, n, rows_per_shard)):
        end = min(start + rows_per_shard, n)
        with open(embeddings_out / f"embeddings-{i:05d}.fbin", "wb") as f:
            # write n,d header
            np.array([end - start, d], dtype=np.uint32).tofile(f)
            # write data
            mmap[start:end].tofile(f)
        num_vecs_written += end - start
        num_shards_written += 1

    assert num_vecs_written == n, "Num vecs written does not match num inputs!"

    # Copy meta file
    shutil.copy(meta, out / "meta.parquet")

    # Write output manifest
    manifest = {
        "n": num_vecs_written,
        "dim": d,
        "num_shards": num_shards_written,
        "rows_per_shard": rows_per_shard,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    auto_cli(shard_fbin, as_positional=False)
