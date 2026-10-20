"""Shard a merged embedding file into ~1 GB pieces for a Hugging Face bundle.

Successor of the deleted vecdb_dataset/shard_fbin.py (commit 097ec39) that
published the sra50 / sra500 fp32 bundles. It now also handles the 2-byte
files written by quantized_exact / pca (embeddings.{fp16,bf16}bin): same
uint32 [n, d] header, 2 bytes per value, copied verbatim as uint16 so no
rounding happens.

Output layout (what the dataset repo serves under embeddings/<label>/):

    base/embeddings-00000.<ext> ...   one header per shard (its own row count)
    meta.parquet                      srr_id, start_row, num_rows (global rows)
    manifest.json                     n, dim, dtype, num_shards, rows_per_shard,
                                      metric, normalized, plus encoder / pca
                                      sidecar contents when present
    projector.pt                      copied when the source is a pca index

Shards concatenate in filename order; a reader adds each shard's row count to
recover the global row ids that meta.parquet and the published hits use.

Usage (compute node; it streams the whole file once, ~7 min for 212 GB):
    uv run python -m hf_dataset.shard_fbin --src <index_dir>/pca-d128-bf16 \
        --out /pscratch/.../embeddings/locale-pca128-bf16
    --src may also be the encoder index dir itself (fp32 embeddings.fbin).
    hf_dataset/export_bundle.sbatch runs this as its first step.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_dc, Path_drw

NP_DTYPES = {"fp32": np.float32, "fp16": np.uint16, "bf16": np.uint16}


def _find_source(src: Path) -> tuple[Path, str]:
    """The embedding file in ``src`` and its dtype label."""
    for dtype in ("fp32", "fp16", "bf16"):
        name = "embeddings.fbin" if dtype == "fp32" else f"embeddings.{dtype}bin"
        if (src / name).exists():
            return src / name, dtype
    raise FileNotFoundError(f"no embeddings.{{fbin,fp16bin,bf16bin}} in {src}")


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def shard_fbin(
    src: Path_drw,
    out: Path_dc,
    rows_per_shard: int | None = None,
    normalized: bool | None = None,
):
    """
    Args:
        src: directory holding embeddings.fbin / embeddings.<dtype>bin. For a
            pca or quantized sub-index, meta.parquet is taken from its parent
            (the encoder index dir) since the row ids are shared.
        out: output bundle directory (created; must not exist).
        rows_per_shard: default 1e9 bytes of values per shard: 325520 rows
            at 768-d fp32 (the sra50 / sra500 bundles), 3906250 at 128-d bf16.
        normalized: recorded in the manifest. Defaults to True for a plain
            encoder index (unit-norm encoder output) and False for a pca
            projection, which does not preserve the norm.
    """
    src = Path(src)
    out = Path(out)
    emb_path, dtype = _find_source(src)
    meta_path = src / "meta.parquet"
    if not meta_path.exists():
        meta_path = src.parent / "meta.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.parquet in {src} or {src.parent}")
    sub_cfg = _read_json(src / "config.json")
    is_pca = bool(sub_cfg and sub_cfg.get("engine") == "pca")
    enc_cfg = _read_json((src.parent if is_pca or sub_cfg else src) / "config.json")
    if normalized is None:
        normalized = not is_pca

    with open(emb_path, "rb") as f:
        n, d = (int(v) for v in np.frombuffer(f.read(8), dtype=np.uint32))
    mmap = np.memmap(emb_path, dtype=NP_DTYPES[dtype], mode="r", offset=8, shape=(n, d))
    if rows_per_shard is None:
        rows_per_shard = 10**9 // (d * np.dtype(NP_DTYPES[dtype]).itemsize)
    meta = pl.read_parquet(meta_path)
    written = int(meta["num_rows"].sum())
    if written != n:
        raise RuntimeError(f"header n={n} but meta.parquet sums to {written}")
    last = meta.sort("start_row").tail(1)
    if int(last["start_row"][0] + last["num_rows"][0]) != n:
        raise RuntimeError("meta.parquet rows do not tile [0, n)")

    ext = emb_path.name.split(".", 1)[1]
    out.mkdir(parents=True)
    base = out / "base"
    base.mkdir()
    num_shards = 0
    vecs = 0
    for i, start in enumerate(range(0, n, rows_per_shard)):
        end = min(start + rows_per_shard, n)
        with open(base / f"embeddings-{i:05d}.{ext}", "wb") as f:
            np.array([end - start, d], dtype=np.uint32).tofile(f)
            mmap[start:end].tofile(f)
        vecs += end - start
        num_shards += 1
        print(f"shard {i:05d}: rows {start}-{end}", flush=True)
    assert vecs == n, f"wrote {vecs} rows, expected {n}"

    shutil.copy(meta_path, out / "meta.parquet")
    manifest = {
        "n": n,
        "dim": d,
        "dtype": dtype,
        "num_shards": num_shards,
        "rows_per_shard": rows_per_shard,
        "metric": "ip",
        "normalized": normalized,
    }
    if enc_cfg:
        manifest["encoder"] = enc_cfg
    if is_pca:
        shutil.copy(src / "projector.pt", out / "projector.pt")
        manifest["pca"] = {
            "source_dim": enc_cfg["embed_dim"]
            if enc_cfg and "embed_dim" in enc_cfg
            else 768,
            "pca_dim": sub_cfg["pca_dim"],
            "projector": "projector.pt",
            "centered": False,
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"done: {num_shards} shards, {n} x {d} {dtype} -> {out}")


if __name__ == "__main__":
    auto_cli(shard_fbin, as_positional=False)
