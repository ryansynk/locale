"""Read-length survival probe on sra500v2 (2026-10-27).

How does retrieval degrade as a read gets shorter relative to the 256 bp index
window, with and without sequencing noise? Synthetic reads are cropped from
real sra500 contigs at fixed lengths (nested crops from one random anchor per
parent contig), mutated with mutation-simulator exactly as the benchmark's
query files were (SNP rate r, insertion and deletion rates r/10), embedded,
and ranked against the WHOLE index of each checkpoint by a streaming exact
scan with the both-strands protocol (max over the two strand embeddings per
vector, top-100 vectors kept per query, regrouped by accession).

Per query the outputs are
  true_strided   max cosine over the parent contig's index windows (W=256,
                 stride 106, end window at tail > 50 bp): the realistic
                 score of the true accession through its own contig
  true_centered  a 256 bp window centred on the read (placement ceiling)
  exact          cosine to the clean read of the same length (noise only,
                 no dilution; 1.0 at rate 0)
  true_scan      the parent accession's best vector in the scan's top-100
                 (>= true_strided when another contig of the accession is
                 closer; null when the accession is outside the top-100)
  best_irr       best vector of any accession not deemed relevant
  rank_acc       accession-level rank of the parent accession (1 = found
                 first); 101 means it is outside the top-100 vectors
  margin         true_strided - best_irr (> 0: found at rank 1 through the
                 parent contig alone)

Relevance: sra500 accessions share contigs (median 2 relevant accessions per
benchmark query), so an accession holding a near-identical window would be a
false "impostor". Any accession with a vector at cosine >= dup_threshold
(0.97) to the parent's CLEAN 256 bp window at rate 0 is treated as relevant
alongside the parent accession, per checkpoint; n_rel_extra records how many
were added. Accessions that share only a sub-span shorter than 256 bp are not
caught, so survival at the shortest lengths is slightly underestimated.

Index pools: the two projection-head indexes are minlen100 already; the
8vqiabk9 endwin50 index holds every contig, so rows whose contig is shorter
than 100 bp are masked with the per-row contig-length sidecar from
experiments/seqlen_cutoff, which makes its background the same population
the sra500_minlen results were searched against.

Stages (jsonargparse subcommands; `--help` on each):
  prepare  CPU: sample parents, crop, mutate -> <out>/parents.parquet, queries.parquet
  scan     GPU: embed + stream one checkpoint's index -> <out>/per_query_<ckpt>.parquet
  report   CPU: tables by checkpoint x rate x length -> <out>/summary.txt

Run (prepare and report are fine on a login node; scan needs a GPU node):
  cd benchmark
  uv run --no-sync python ../experiments/length_survival/survival.py prepare
  srun -A m5408_g -C gpu -q interactive -N 1 --gpus=1 -c 32 -t 1:00:00 \
      uv run --no-sync python ../experiments/length_survival/survival.py scan --checkpoints '[pc6t1x8j,qpms6u7j,8vqiabk9]'
  uv run --no-sync python ../experiments/length_survival/survival.py report
Defaults: 1000 parents from 50 accessions, lengths 50..256, rates 0/5/10%, 2
draws per rate -> 40,000 queries per checkpoint; ~5 min per projected index
and ~15 min for 8vqiabk9 on one A100 (I/O bound on the 480 GB fbin).
"""

import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl
import torch
from jsonargparse import auto_cli

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "benchmark"
sys.path.insert(0, str(BENCH))
from src.config import EncoderConfig  # noqa: E402
from src.dense_index import _end_tail, reverse_complement  # noqa: E402
from src.encoders import DenseEncoder  # noqa: E402
from src.fbin import FBIN_HEADER_BYTES, read_fbin_rows  # noqa: E402

BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")
MUTATION_SIMULATOR = "/pscratch/sd/r/rsynk/locale-data/.venv/bin/mutation-simulator"
INDEXES = BENCH / "indexes/sra500"
W, STEP, END_TAIL = 256, 106, 50  # the endwin50 index chunking
CHECKPOINTS = {
    "pc6t1x8j": {
        "ckpt": ROOT / "checkpoints/pc6t1x8j/checkpoint21380.pth.tar",
        "project": True,
        "index": INDEXES / "locale@pc6t1x8j-proj-endwin50-minlen100",
    },
    "qpms6u7j": {
        "ckpt": ROOT / "checkpoints/qpms6u7j/checkpoint4276.pth.tar",
        "project": True,
        "index": INDEXES / "locale@qpms6u7j-proj-endwin50-minlen100",
    },
    "8vqiabk9": {
        "ckpt": ROOT / "checkpoints/8vqiabk9/checkpoint5859.pth.tar",
        "project": False,
        "index": INDEXES / "locale@8vqiabk9-endwin50",
        "contig_len_u32": ROOT / "experiments/seqlen_cutoff/sra500_endwin50_contig_len.u32",
        "min_seq_len": 100,
    },
}
DEFAULT_LENGTHS = [50, 64, 75, 100, 125, 150, 200, 256]
DEFAULT_RATES = [0.0, 0.05, 0.10]
MISSING_RANK = 101


# --------------------------------------------------------------------------- #
# prepare
# --------------------------------------------------------------------------- #
def _long_contigs(args):
    """(accession, min_len) -> list of (contig_id, seq) with len >= min_len.
    Logan FASTA: one sequence line per record, id = first header token."""
    acc, min_len = args
    out, cid = [], None
    with open(BUNDLE / "logan_accessions" / acc / f"{acc}.contigs.fa") as f:
        for line in f:
            if line[0] == ">":
                cid = line[1:].split(maxsplit=1)[0]
            else:
                s = line.strip().upper()
                if len(s) >= min_len and set(s) <= set("ACGT"):
                    out.append((cid, s))
    return acc, out


def _mutate(reads: pl.DataFrame, rate: float, work: Path, tag: str) -> dict:
    """Run mutation-simulator on the reads (ids must be dot-free: it truncates
    names at the first '.'); SNP rate `rate`, indel rates rate/10 each, as
    locale-data/benchmark/mutate_queries.py. Returns {read_id: mutated seq}."""
    work.mkdir(parents=True, exist_ok=True)
    fa = work / f"{tag}.fa"
    with open(fa, "w") as f:
        for rid, seq in reads.select("read_id", "sequence").iter_rows():
            f.write(f">{rid}\n{seq}\n")
    base = work / f"{tag}_out"
    cmd = [MUTATION_SIMULATOR, "--no-progress", "--no-color", "--quiet",
           "--output", str(base), str(fa), "args",
           "--snp", str(rate), "--insert", str(rate / 10), "--deletion", str(rate / 10)]
    subprocess.run(cmd, check=True)
    out, rid = {}, None
    with open(base.with_name(base.name + "_ms.fa")) as f:
        for line in f:
            if line[0] == ">":
                rid = line[1:].split(maxsplit=1)[0]
                out[rid] = ""
            else:
                out[rid] += line.strip().upper()
    missing = set(reads["read_id"]) - out.keys()
    if missing:
        raise RuntimeError(f"mutation-simulator dropped {len(missing)} reads, e.g. {sorted(missing)[:3]}")
    return out


def prepare(
    out: Path = HERE / "run",
    n_parents: int = 1000,
    n_parent_accs: int = 50,
    lengths: list[int] = DEFAULT_LENGTHS,
    rates: list[float] = DEFAULT_RATES,
    n_draws: int = 2,
    seed: int = 0,
    parent_accessions: list[str] | None = None,
):
    """Sample parent contigs (>= 256 bp, ACGT only) from random accessions,
    crop nested reads of every length from one random anchor per parent, and
    mutate each read n_draws times per non-zero rate.

    Args:
        out: run directory (parents.parquet, queries.parquet, mutations/).
        n_parents: parent contigs in total, spread evenly over the accessions.
        n_parent_accs: accessions to draw parents from.
        lengths: read lengths in bp (all <= 256).
        rates: SNP rates; 0.0 is the clean read, one draw.
        n_draws: independent mutation draws per read per non-zero rate.
        seed: numpy seed for accession, contig and anchor choice (the
            simulator itself is unseeded; mutations/ keeps its output).
        parent_accessions: fixed accession list instead of a random draw.
    """
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert max(lengths) <= W, "lengths must be <= 256"
    rng = np.random.default_rng(seed)
    accs = BUNDLE.joinpath("accs.txt").read_text().split()
    if parent_accessions is None:
        parent_accessions = sorted(rng.choice(accs, size=n_parent_accs, replace=False).tolist())
    per_acc = int(np.ceil(n_parents / len(parent_accessions)))

    with ProcessPoolExecutor(max_workers=min(32, len(parent_accessions))) as ex:
        pools = dict(ex.map(_long_contigs, [(a, W) for a in parent_accessions]))
    rows = []
    for acc in parent_accessions:
        pool = pools[acc]
        if len(pool) < per_acc:
            print(f"[warn] {acc}: only {len(pool)} contigs >= {W} bp (wanted {per_acc})")
        for j in rng.choice(len(pool), size=min(per_acc, len(pool)), replace=False):
            cid, seq = pool[j]
            start = int(rng.integers(0, len(seq) - W + 1))
            rows.append({"parent_id": f"p{len(rows)}", "accession": acc, "contig_id": cid,
                         "contig_len": len(seq), "start": start, "contig": seq})
    parents = pl.DataFrame(rows)
    parents.write_parquet(out / "parents.parquet")
    print(f"{len(parents)} parents from {len(parent_accessions)} accessions "
          f"(median contig {int(parents['contig_len'].median())} bp) in {time.time() - t0:.0f}s")

    reads = pl.DataFrame(
        [{"read_id": f"{p['parent_id']}_L{L}", "parent_id": p["parent_id"], "accession": p["accession"],
          "L": L, "sequence": p["contig"][p["start"]: p["start"] + L]}
         for p in parents.iter_rows(named=True) for L in lengths]
    )
    frames = [reads.with_columns(pl.lit(0.0).alias("rate"), pl.lit(0).alias("draw"))]
    for rate in rates:
        if rate == 0.0:
            continue
        for d in range(n_draws):
            tag = f"mut{round(rate * 100):02d}pct_d{d}"
            mut = _mutate(reads, rate, out / "mutations", tag)
            frames.append(reads.with_columns(
                pl.col("read_id").replace_strict(mut).alias("sequence"),
                pl.lit(rate).alias("rate"), pl.lit(d).alias("draw")))
            print(f"  {tag}: {len(mut)} reads mutated ({time.time() - t0:.0f}s)")
    queries = pl.concat(frames).with_columns(
        (pl.col("read_id") + "_r" + pl.col("rate").round(2).cast(pl.String) + "_d" + pl.col("draw").cast(pl.String)).alias("query_id")
    )
    queries.write_parquet(out / "queries.parquet")
    json.dump({"n_parents": len(parents), "parent_accessions": parent_accessions, "lengths": lengths,
               "rates": rates, "n_draws": n_draws, "seed": seed}, open(out / "prepare.json", "w"), indent=1)
    print(f"{len(queries)} queries written to {out} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- #
# scan
# --------------------------------------------------------------------------- #
def _index_windows(contig: str, start: int) -> list[str]:
    """The endwin50 index's 256 bp windows of `contig` that overlap
    [start, start + 256) (the anchor span every read is cropped from)."""
    n = len(contig)
    starts = list(range(0, n - W + 1, STEP))
    if _end_tail(n, W, STEP) > END_TAIL:
        starts.append(n - W)
    return [contig[s: s + W] for s in starts if s < start + W and s + W > start]


def _centered_window(contig: str, start: int, L: int) -> str:
    st = min(max(start + L // 2 - W // 2, 0), len(contig) - W)
    return contig[st: st + W]


def _stream_topk(fbin: Path, Q: torch.Tensor, k: int, block_rows: int, q_chunk: int,
                 mask_u32: np.ndarray | None, min_seq_len: int | None, max_rows: int | None):
    """Exact scan of the fbin: per query the top-k (score, row) over all rows,
    score = max over the query's two strand embeddings Q[:, 0] / Q[:, 1]
    (n, 2, d). Rows whose contig length (mask_u32) is below min_seq_len are
    skipped. Blocks are pread on a prefetch thread; logits in fp16."""
    dev = Q.device
    with open(fbin, "rb") as f:
        n_rows, d = (int(x) for x in np.frombuffer(f.read(FBIN_HEADER_BYTES), dtype=np.uint32))
    if max_rows:
        n_rows = min(n_rows, max_rows)
    assert d == Q.shape[-1], f"index dim {d} != query dim {Q.shape[-1]}"
    nq = Q.shape[0]
    top_s = torch.full((nq, k), -float("inf"), device=dev, dtype=torch.float32)
    top_i = torch.full((nq, k), -1, device=dev, dtype=torch.int64)
    fd = os.open(fbin, os.O_RDONLY)
    bufs = [np.empty((block_rows, d), dtype=np.float32) for _ in range(2)]
    blocks = list(range(0, n_rows, block_rows))

    def load(bi, buf):
        s = blocks[bi]
        return read_fbin_rows(fd, s, min(s + block_rows, n_rows), buf)

    t0 = time.time()
    pending = {}
    def prefetch(bi, buf):
        pending[bi] = load(bi, buf)
    thr = threading.Thread(target=prefetch, args=(0, bufs[0])); thr.start()
    for bi, s in enumerate(blocks):
        thr.join()
        blk = pending.pop(bi)
        if bi + 1 < len(blocks):
            thr = threading.Thread(target=prefetch, args=(bi + 1, bufs[(bi + 1) % 2])); thr.start()
        X = torch.from_numpy(blk).to(dev, non_blocking=False).to(torch.float16)
        keep = None
        if mask_u32 is not None:
            keep = torch.from_numpy((mask_u32[s: s + len(blk)] >= min_seq_len).copy()).to(dev)
        kk = min(k, X.shape[0])
        for q0 in range(0, nq, q_chunk):
            qf, qr = Q[q0: q0 + q_chunk, 0], Q[q0: q0 + q_chunk, 1]
            logit = qf @ X.T
            torch.maximum(logit, qr @ X.T, out=logit)
            if keep is not None:
                logit.masked_fill_(~keep, -float("inf"))
            bs, bi_ = torch.topk(logit, kk, dim=1)
            cs = torch.cat([top_s[q0: q0 + q_chunk], bs.float()], 1)
            ci = torch.cat([top_i[q0: q0 + q_chunk], bi_ + s], 1)
            ms, mi = torch.topk(cs, k, dim=1)
            top_s[q0: q0 + q_chunk] = ms
            top_i[q0: q0 + q_chunk] = torch.gather(ci, 1, mi)
            del logit
        if bi % 10 == 0 or bi == len(blocks) - 1:
            done = min(s + block_rows, n_rows)
            print(f"    scanned {done:,}/{n_rows:,} rows ({time.time() - t0:.0f}s)", flush=True)
    os.close(fd)
    return top_s.cpu().numpy(), top_i.cpu().numpy()


def scan(
    checkpoints: list[str] = ["pc6t1x8j", "qpms6u7j", "8vqiabk9"],
    out: Path = HERE / "run",
    device: str = "cuda",
    top_k: int = 100,
    dup_threshold: float = 0.97,
    block_rows: int = 1_000_000,
    q_chunk: int = 4096,
    embed_batch: int = 512,
    max_rows: int | None = None,
):
    """Embed the prepared queries and parent windows with each checkpoint and
    rank every query against that checkpoint's whole sra500 index.

    Args:
        checkpoints: keys of CHECKPOINTS to run, one after another.
        out: run directory written by `prepare`.
        device: cuda or cpu (cpu only for smoke tests with max_rows).
        top_k: vectors kept per query (the benchmark's 100).
        dup_threshold: cosine to the clean 256 bp parent window above which
            another accession's vector marks that accession relevant.
        block_rows: fbin rows per streamed block (1M x 768 fp32 = 3 GB host).
        q_chunk: queries per matmul (q_chunk x block_rows fp16 logits, twice).
        embed_batch: encoder batch size.
        max_rows: scan only the first rows of the fbin (smoke tests).
    """
    out = Path(out)
    parents = pl.read_parquet(out / "parents.parquet")
    queries = pl.read_parquet(out / "queries.parquet")
    pid_to_parent = {p["parent_id"]: p for p in parents.iter_rows(named=True)}
    for name in checkpoints:
        t0 = time.time()
        spec = CHECKPOINTS[name]
        print(f"=== {name}: {spec['ckpt'].name}, project={spec['project']}, index {spec['index'].name}")
        enc = DenseEncoder(EncoderConfig(name="locale", checkpoint_path=str(spec["ckpt"]),
                                         project=spec["project"], batch_size=embed_batch, device=device))
        meta = pl.read_parquet(spec["index"] / "meta.parquet").sort("start_row")
        acc_start = meta["start_row"].to_numpy()
        acc_names = meta["srr_id"].to_list()

        def embed(seqs):
            return enc.encode(seqs).to(torch.float16)

        # queries, both strands
        seqs = queries["sequence"].to_list()
        E = torch.stack([embed(seqs), embed([reverse_complement(x) for x in seqs])], 1)  # (n, 2, d)
        print(f"  embedded {len(seqs)} queries x 2 strands ({time.time() - t0:.0f}s)")
        qpos = {q: i for i, q in enumerate(queries["query_id"].to_list())}

        # parent windows: strided index windows + one centred window per length
        win_strided, win_centered = {}, {}
        for p in parents.iter_rows(named=True):
            win_strided[p["parent_id"]] = _index_windows(p["contig"], p["start"])
        for q in queries.filter(pl.col("rate") == 0.0).iter_rows(named=True):
            p = pid_to_parent[q["parent_id"]]
            win_centered[(q["parent_id"], q["L"])] = _centered_window(p["contig"], p["start"], q["L"])
        uniq = sorted({w for ws in win_strided.values() for w in ws} | set(win_centered.values()))
        WE = embed(uniq)
        widx = {w: i for i, w in enumerate(uniq)}
        print(f"  embedded {len(uniq)} parent windows ({time.time() - t0:.0f}s)")

        def best_vs(qi, wins):
            ids = torch.tensor([widx[w] for w in wins], device=E.device)
            return (E[qi] @ WE[ids].T).max().item()  # max over strands and windows

        # the whole index
        mask = None
        if spec.get("contig_len_u32"):
            mask = np.memmap(spec["contig_len_u32"], dtype="<u4", mode="r")
        top_s, top_i = _stream_topk(spec["index"] / "embeddings.fbin", E, top_k, block_rows, q_chunk,
                                    mask, spec.get("min_seq_len"), max_rows)
        top_acc = np.searchsorted(acc_start, top_i, side="right") - 1
        print(f"  scan done ({time.time() - t0:.0f}s)")

        # relevant accessions per parent from the clean 256 bp query
        relevant = {}
        for q in queries.filter((pl.col("rate") == 0.0) & (pl.col("L") == W)).iter_rows(named=True):
            i = qpos[q["query_id"]]
            rel = {q["accession"]} | {acc_names[a] for a, s in zip(top_acc[i], top_s[i]) if s >= dup_threshold}
            relevant[q["parent_id"]] = rel
        rows = []
        for q in queries.iter_rows(named=True):
            i = qpos[q["query_id"]]
            rel = relevant[q["parent_id"]]
            acc_best = {}
            for a, s in zip(top_acc[i], top_s[i]):
                if s == -np.inf:
                    continue
                an = acc_names[a]
                acc_best[an] = max(acc_best.get(an, -np.inf), float(s))
            irr = [s for a, s in acc_best.items() if a not in rel]
            kth = float(top_s[i][-1])
            true_scan = acc_best.get(q["accession"])
            t_str = best_vs(i, win_strided[q["parent_id"]])
            ref = true_scan if true_scan is not None else t_str
            n_above = sum(s > ref for s in irr)
            rank = 1 + n_above if true_scan is not None else MISSING_RANK
            clean_id = f"{q['parent_id']}_L{q['L']}_r0.0_d0"
            rows.append({
                "query_id": q["query_id"], "parent_id": q["parent_id"], "accession": q["accession"],
                "L": q["L"], "rate": q["rate"], "draw": q["draw"], "qlen_actual": len(q["sequence"]),
                "true_strided": t_str,
                "true_centered": best_vs(i, [win_centered[(q["parent_id"], q["L"])]]),
                "exact": float((E[i] @ E[qpos[clean_id]].T).max().item()),
                "true_scan": true_scan,
                "best_irr": max(irr) if irr else kth, "best_irr_is_bound": not irr,
                "kth_score": kth, "n_rel_extra": len(rel) - 1,
                "rank_acc": rank, "in_top100": true_scan is not None,
            })
        pq = pl.DataFrame(rows).with_columns(
            (pl.col("true_strided") - pl.col("best_irr")).alias("margin"), pl.lit(name).alias("ckpt"))
        pq.write_parquet(out / f"per_query_{name}.parquet")
        print(f"  wrote per_query_{name}.parquet ({len(pq)} rows, {time.time() - t0:.0f}s)")
        del enc, E, WE
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def report(out: Path = HERE / "run", isolated_below: float | None = 0.75):
    """Survival and score tables by checkpoint x rate x length -> summary.txt.

    Args:
        out: run directory holding per_query_<ckpt>.parquet files.
        isolated_below: also report the subset of parents whose clean 256 bp
            window has no unflagged accession scoring >= this cosine (under
            the first checkpoint listed, alphabetically). sra500 accessions
            come in related groups: 3 of 4 parents have another accession
            holding a window at cosine >= 0.75 to theirs (a partial overlap
            the 0.97 duplicate rule does not catch), and those accessions
            outrank the parent on short reads. The subset has the benchmark's
            ~0.6 background and gives the clean length curve. None skips it.
    """
    out = Path(out)
    files = sorted(out.glob("per_query_*.parquet"))
    if not files:
        raise SystemExit(f"no per_query_*.parquet under {out}; run scan first")
    pq = pl.concat([pl.read_parquet(f) for f in files])
    subsets = {"all parents": pq}
    if isolated_below is not None:
        ck0 = pq["ckpt"].unique().sort()[0]
        ref = pq.filter((pl.col("ckpt") == ck0) & (pl.col("rate") == 0.0) & (pl.col("L") == W))
        iso = set(ref.filter(pl.col("best_irr") < isolated_below)["parent_id"].to_list())
        subsets[f"isolated parents (no other accession >= {isolated_below} to the clean window under {ck0}): {len(iso)} of {len(ref)}"] = (
            pq.filter(pl.col("parent_id").map_elements(lambda p: p in iso, return_dtype=pl.Boolean)))
    def table(df):
        return (
            df.group_by("ckpt", "rate", "L").agg(
                pl.len().alias("n"),
                (pl.col("rank_acc") == 1).mean().alias("P_rank1"),
                (pl.col("rank_acc") <= 10).mean().alias("P_top10"),
                pl.col("in_top100").mean().alias("P_hit100"),
                (pl.col("margin") > 0).mean().alias("P_margin>0"),
                pl.col("true_strided").mean().alias("true_strided"),
                pl.col("true_centered").mean().alias("true_centered"),
                pl.col("exact").mean().alias("exact"),
                pl.col("best_irr").mean().alias("background"),
                pl.col("margin").median().alias("margin_med"),
                pl.col("n_rel_extra").mean().alias("rel_extra"),
            ).sort("ckpt", "rate", "L")
        )

    lines = ["Read-length survival on sra500v2 (whole-index exact scan, both strands, top-100 regroup)",
             "P_rank1: parent accession ranked first; P_top10: within 10; P_hit100: any of its vectors in the",
             "top-100; P_margin>0: parent contig's own window beats every irrelevant vector; scores are mean",
             "cosines (true_strided = realistic index window, true_centered = placement ceiling, exact = clean",
             "read of the same length); background = mean best irrelevant score; rel_extra = accessions",
             "added as relevant by the duplicate rule.", ""]
    tabs = {}
    with pl.Config(tbl_rows=1000, tbl_cols=20, tbl_width_chars=220, float_precision=3):
        for name, df in subsets.items():
            tab = tabs[name] = table(df)
            lines.append(f"=============== {name} ===============")
            for ck in tab["ckpt"].unique().sort():
                lines.append(f"##### {ck} #####")
                lines.append(str(tab.filter(pl.col("ckpt") == ck).drop("ckpt")))
                lines.append("")
            piv = tab.pivot(on="ckpt", index=["rate", "L"], values="P_rank1").sort("rate", "L")
            lines.append("##### P_rank1 side by side #####")
            lines.append(str(piv))
            lines.append("")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    tabs["all parents"].write_parquet(out / "summary.parquet")
    for name, tab in tabs.items():
        if name != "all parents":
            tab.write_parquet(out / "summary_isolated.parquet")
    print(text)


if __name__ == "__main__":
    auto_cli([prepare, scan, report])
