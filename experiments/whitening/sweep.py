"""Post-hoc whitening sweep on sra500: does raising the effective dimension of
the embedding lower the unrelated-score floor without losing the true pairs?

Whitening with exponent alpha: fit mean mu and covariance eigenpairs (U, lam)
on a strided sample of the index, then score a read q against a window x by
the cosine of W(q - mu) and W(x - mu) with W = diag(lam^-alpha) U^T. alpha 0
is the plain (centred) cosine, 0.5 is full whitening, in between is partial.
Nothing is materialised: with M = U diag(lam^-2alpha) U^T the score is
((M(q-mu)) . (x-mu)) / (|W(q-mu)| |W(x-mu)|), so one pass over the original
fp32 fbin serves every alpha -- per block, a (B x d) @ (d x d) matmul per
variant gives both the per-row whitened norms and the logits. The variant
"raw" (alpha 0, no centring) is the validation: it must reproduce the saved
exact top-100 run at the same cutoff.

Only rows whose contig is at least --cutoff bp long compete (the length map
from experiments/seqlen_cutoff), matching the deployed min_seq_len.

Results go to <out_dir>/<encoder_label>/whiten-<variant>/top<k>/mut<rate>.parquet
(+ hits/, config.json), the standard layout, so score.py / print_results.py
read them. The fitted whitening is saved to <out_dir>/whitening.pt.

Single GPU node, all its GPUs, from the benchmark directory:
  uv run python ../experiments/whitening/sweep.py \
      --bench_config configs/sra500/perlmutter_locale_sra500v2_endwin50.yaml \
      --lengths ../experiments/seqlen_cutoff/sra500_endwin50_contig_len.u32
"""

import copy
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import CLI, auto_cli
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE.parents[1] / "benchmark"
sys.path.insert(0, str(BENCHMARK_DIR))

import run_benchmark as rb  # noqa: E402
from src.config import (  # noqa: E402
    DONE_FILE,
    DenseMethod,
    ExactIndex,
    ExperimentConfig,
    results_file_name,
    write_config,
)
from src.dense_index import DenseIndex  # noqa: E402
from src.exact import block_loader  # noqa: E402
from src.topk_regroup import regroup_topk_hits  # noqa: E402

DEFAULT_ALPHAS = [0.0, 0.125, 0.25, 0.375, 0.5]
DEFAULT_RATES = [0.0, 0.05, 0.10]
BLOCK_ROWS = 1_000_000  # 1M x 768 fp32 = 3 GB; three such tensors + logits per GPU
EIG_FLOOR = 1e-4  # eigenvalues below this fraction of the largest are clamped


# --------------------------------------------------------------------------
def fit_whitening(
    all_embeddings: torch.Tensor, lengths, cutoff: int, n_sample: int, seed: int
):
    """Mean and covariance eigenpairs of a strided sample of surviving rows."""
    n = all_embeddings.shape[0]
    keep = np.flatnonzero(np.asarray(lengths) >= cutoff)
    rng = np.random.default_rng(seed)
    # strided over the survivors (covers every accession), jittered
    pick = keep[
        np.sort(rng.choice(len(keep), size=min(n_sample, len(keep)), replace=False))
    ]
    t0 = time.time()
    X = all_embeddings[torch.from_numpy(pick)].double().numpy()
    print(f"whitening fit: read {len(X):,} of {n:,} rows in {time.time() - t0:.0f}s")
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    mu = X.mean(0)
    C = np.cov(X - mu, rowvar=False)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    lam, U = np.clip(lam[order], 0, None), U[:, order]
    pr = lam.sum() ** 2 / (lam**2).sum()
    print(
        f"whitening fit: |mu| {np.linalg.norm(mu):.3f}, participation ratio {pr:.0f}, "
        f"top/median eigenvalue {lam[0] / np.median(lam):.0f}, dims below floor: {(lam < EIG_FLOOR * lam[0]).sum()}"
    )
    return mu, U, lam


def variant_matrix(U, lam, alpha: float) -> np.ndarray:
    """M = U diag(lam^-2alpha) U^T (identity at alpha 0), fp64."""
    if alpha == 0:
        return np.eye(len(lam))
    lam_c = np.maximum(lam, EIG_FLOOR * lam[0])
    return (U * lam_c ** (-2 * alpha)) @ U.T


def variants_from(mu, U, lam, alphas):
    """[(tag, M, mu_used)]: 'raw' (uncentred cosine, the validation), then
    alpha<a> centred for each alpha."""
    d = len(lam)
    out = [("raw", np.eye(d), np.zeros(d))]
    for a in alphas:
        out.append((f"a{a:g}", variant_matrix(U, lam, a), mu))
    return out


@torch.no_grad()
def scan_topk_whitened(
    engine,
    query_vecs: np.ndarray,
    lengths,
    cutoff: int,
    variants,
    top_k: int,
    block_rows: int = BLOCK_ROWS,
):
    """Exact top-k per variant over rows with lengths >= cutoff.

    query_vecs: (n_chunks, d) raw unit query embeddings (both strands as rows,
    as DenseIndex._embed_queries lays them out). Returns {tag: (scores, ids)}.
    """
    assert engine.all_embeddings is not None
    n_vecs, d = engine.all_embeddings.shape
    devices = engine.devices
    n_dev = len(devices)
    per_dev = -(-n_vecs // n_dev)
    Q = np.asarray(query_vecs, dtype=np.float64)
    # per variant: whitened, normalised queries folded through M -> (n_chunks, d)
    qw = []
    for tag, M, mu_v in variants:
        Qc = Q - mu_v
        QM = Qc @ M  # M symmetric
        qn = np.sqrt(np.maximum((QM * Qc).sum(1), 1e-12))
        qw.append(torch.as_tensor((QM / qn[:, None]).astype(np.float32)))
    mats = [torch.as_tensor(M.astype(np.float32)) for _, M, _ in variants]
    mus = [torch.as_tensor(mu_v.astype(np.float32)) for _, _, mu_v in variants]
    n_chunks = len(Q)

    def _scan(dev_idx):
        dev = torch.device(devices[dev_idx])
        s, e = dev_idx * per_dev, min((dev_idx + 1) * per_dev, n_vecs)
        qw_d = [t.to(dev) for t in qw]
        M_d = [m.to(dev) for m in mats]
        mu_d = [m.to(dev) for m in mus]
        best_vals = [
            torch.full((n_chunks, top_k), float("-inf"), device=dev) for _ in variants
        ]
        best_ids = [
            torch.full((n_chunks, top_k), -1, device=dev, dtype=torch.int64)
            for _ in variants
        ]
        blocks = range(s, e, block_rows)
        if dev_idx == 0:
            blocks = tqdm(
                blocks,
                desc=f"whitening scan x{len(variants)} variants ({n_dev} GPUs, {n_vecs:,} rows)",
            )
        with block_loader(
            engine.fbin_path, engine.all_embeddings, block_rows, dev
        ) as load_block:
            for bs in blocks:
                be = min(bs + block_rows, e)
                X = load_block(bs, be)  # (B, d) fp32
                dead = torch.from_numpy(np.asarray(lengths[bs:be]) < cutoff).to(dev)
                dead_cols = dead.nonzero().squeeze(1)
                k = min(top_k, be - bs)
                for vi in range(len(variants)):
                    Xc = X - mu_d[vi] if vi > 0 else X
                    XM = Xc @ M_d[vi] if vi > 0 else Xc  # raw: M = I
                    nrm = torch.sqrt(torch.clamp((XM * Xc).sum(1), min=1e-12))
                    logits = (qw_d[vi] @ Xc.T) / nrm[None, :]
                    if dead_cols.numel():
                        logits.index_fill_(1, dead_cols, float("-inf"))
                    vals, idx = torch.topk(logits, k, dim=-1)
                    cand_vals = torch.cat([best_vals[vi], vals], dim=1)
                    cand_ids = torch.cat([best_ids[vi], idx + bs], dim=1)
                    best_vals[vi], pos = torch.topk(cand_vals, top_k, dim=-1)
                    best_ids[vi] = torch.gather(cand_ids, 1, pos)
                    del Xc, XM, nrm, logits, vals, idx, cand_vals, cand_ids, pos
                del X, dead, dead_cols
        out = []
        for bv, bi in zip(best_vals, best_ids):
            bi = torch.where(torch.isinf(bv), torch.full_like(bi, -1), bi)
            out.append((bv.cpu(), bi.cpu()))
        return out

    with ThreadPoolExecutor(max_workers=n_dev) as pool:
        shard_results = list(pool.map(_scan, range(n_dev)))
    result = {}
    for vi, (tag, _, _) in enumerate(variants):
        all_vals = torch.cat([r[vi][0] for r in shard_results], dim=1)
        all_ids = torch.cat([r[vi][1] for r in shard_results], dim=1)
        top_vals, top_pos = torch.topk(all_vals, min(top_k, all_vals.shape[1]), dim=-1)
        result[tag] = (top_vals.numpy(), torch.gather(all_ids, 1, top_pos).numpy())
    return result


# --------------------------------------------------------------------------
def main(
    bench_config: str,
    lengths: str,
    alphas: list[float] = DEFAULT_ALPHAS,
    rates: list[float] = DEFAULT_RATES,
    cutoff: int = 100,
    top_k: int = 100,
    n_sample: int = 200_000,
    out_dir: str | None = None,
    block_rows: int = BLOCK_ROWS,
):
    """
    Args:
        bench_config: run_benchmark yaml of the index to search (dense, exact).
        lengths: uint32 contig length per fbin row (vector_contig_lengths.py).
        alphas: whitening exponents; 0 = centred cosine, 0.5 = full whitening.
        rates: mutation rates (each needs queries_mut<rate>.parquet).
        cutoff: min contig length in bp for a row to compete (deployed: 100).
        top_k: vectors per query before regrouping to accessions.
        n_sample: index rows sampled to fit the whitening.
        out_dir: results root (default experiments/whitening/results).
        block_rows: fbin rows per GPU block.
    """
    out = Path(out_dir) if out_dir else HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    cfg: ExperimentConfig = CLI(
        ExperimentConfig,
        as_positional=False,
        args=["--config", bench_config, "--shard", "0", "--num_shards", "1"],
    )
    if not (
        isinstance(cfg.model, DenseMethod) and isinstance(cfg.model.index, ExactIndex)
    ):
        raise ValueError("config must be a dense method with the exact engine")
    cfg.model.index.top_k = top_k
    index_path = cfg.model.index_path(cfg.index_dir)
    if not (index_path / DONE_FILE).exists():
        raise FileNotFoundError(f"{index_path} is not .done")
    bundle = Path(cfg.dataset_dir).resolve()

    index = DenseIndex(cfg)
    index.load(index_path)
    n_vecs = index.num_vectors()
    lengths_arr = np.memmap(lengths, dtype="<u4", mode="r")
    if lengths_arr.shape[0] != n_vecs:
        raise ValueError(
            f"{lengths} has {lengths_arr.shape[0]:,} entries, index has {n_vecs:,} rows"
        )
    n_alive = int((np.asarray(lengths_arr) >= cutoff).sum())
    print(f"cutoff {cutoff}: {n_alive:,} of {n_vecs:,} rows compete")

    wfile = out / "whitening.pt"
    if wfile.exists():
        w = torch.load(wfile, weights_only=False)
        mu, U, lam = w["mu"], w["U"], w["lam"]
        print(f"loaded whitening fit from {wfile}")
    else:
        mu, U, lam = fit_whitening(
            index.engine.all_embeddings, lengths_arr, cutoff, n_sample, cfg.random_seed
        )
        torch.save(
            {
                "mu": mu,
                "U": U,
                "lam": lam,
                "n_sample": n_sample,
                "cutoff": cutoff,
                "index_path": str(index_path),
            },
            wfile,
        )
    variants = variants_from(mu, U, lam, alphas)
    with open(out / "sweep.json", "w") as f:
        json.dump(
            {
                "bench_config": str(Path(bench_config).resolve()),
                "index_path": str(index_path),
                "cutoff": cutoff,
                "top_k": top_k,
                "alphas": alphas,
                "variants": [v[0] for v in variants],
                "rates": rates,
                "n_rows": n_vecs,
                "n_alive": n_alive,
                "participation_ratio": float(lam.sum() ** 2 / (lam**2).sum()),
                "eig_floor": EIG_FLOOR,
            },
            f,
            indent=1,
        )

    def results_dir(tag):
        return out / cfg.model.encoder_label / f"whiten-{tag}" / f"top{top_k}"

    for rate in rates:
        files = {
            tag: results_dir(tag) / results_file_name(rate) for tag, _, _ in variants
        }
        if all(p.exists() for p in files.values()):
            print(f"rate {rate}: all variants done, skipping")
            continue
        queries = pl.read_parquet(bundle / rb.query_file_name(rate))
        queries = queries.sample(
            min(cfg.num_queries, len(queries)), seed=cfg.random_seed
        )
        t0 = time.time()
        q_feats, query_indices, _ = index._embed_queries(queries)
        q = q_feats.float().cpu().numpy()
        print(
            f"rate {rate}: {len(queries)} queries -> {len(q)} chunks embedded in {time.time() - t0:.0f}s; scanning"
        )
        per_variant = scan_topk_whitened(
            index.engine, q, lengths_arr, cutoff, variants, top_k, block_rows
        )
        print(f"rate {rate}: scan done in {time.time() - t0:.0f}s")
        for tag, (scores, ids) in per_variant.items():
            hits = index._hits_from_chunk_topk(queries, query_indices, scores, ids)
            results = regroup_topk_hits(hits, index.indexed_accessions())
            cfg_v = copy.deepcopy(cfg)
            cfg_v.model.index_label = f"whiten-{tag}"
            cfg_v.model.search_label = f"top{top_k}"
            cfg_v.mutation_rate = rate
            results = rb._annotate_results(results, cfg_v, index, index_path, -1.0)
            d = results_dir(tag)
            d.mkdir(parents=True, exist_ok=True)
            write_config(
                d,
                {
                    **rb.run_search_identity(cfg),
                    "min_seq_len": cutoff,
                    "whitening": tag,
                },
            )
            results.write_parquet(files[tag])
            (d / "hits").mkdir(exist_ok=True)
            rb._annotate_results(hits, cfg_v, index, index_path, -1.0).write_parquet(
                d / "hits" / results_file_name(rate)
            )
            print(f"rate {rate} {tag}: wrote {files[tag]}")

    print(f"done; score with:\n  uv run python {HERE / 'score.py'} --results_dir {out}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
