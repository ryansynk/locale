"""Score the whitening sweep: per variant and noise level, recall@k over
accessions, AUPRC, and the floor-vs-true-pair picture from the hits (median
best irrelevant score, median best true score, fraction of queries whose true
accession outranks every irrelevant one).

  uv run python ../experiments/whitening/score.py --results_dir ../experiments/whitening/results
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(HERE.parent / "seqlen_cutoff"))
import plot_results as pr  # noqa: E402
from score_sweep import per_query_ap  # noqa: E402

BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")
KS = [1, 10, 50]


def main(results_dir: str, bundle: str = str(BUNDLE)):
    bundle = Path(bundle)
    out = Path(results_dir)
    q = pl.read_parquet(bundle / "queries.parquet")
    truth = pr.raw_read_oracle_results(q).select("query_id", "results")
    tsets = {
        r["query_id"]: {t["accession"] for t in r["results"]}
        for r in truth.iter_rows(named=True)
    }
    accs = (bundle / "accs.txt").read_text().splitlines()
    rows = []
    for f in sorted(out.glob("*/whiten-*/top*/mut*.parquet")):
        tag = f.parents[1].name.removeprefix("whiten-")
        rate = float(f.stem.removeprefix("mut"))
        res = pl.read_parquet(f)
        rec = {k: [] for k in KS}
        best_irr, best_true, wins = [], [], []
        for qid, rl in res.select("query_id", "results").iter_rows():
            T = tsets.get(qid)
            if not T:
                continue
            ranked = sorted(rl, key=lambda x: -x["score"])
            names = [x["accession"] for x in ranked]
            for k in KS:
                rec[k].append(len(T & set(names[:k])) / len(T))
            bi = max(
                (x["score"] for x in ranked if x["accession"] not in T), default=np.nan
            )
            bt = max(
                (x["score"] for x in ranked if x["accession"] in T), default=np.nan
            )
            best_irr.append(bi)
            best_true.append(bt)
            wins.append(bt > bi)
        ap = per_query_ap(res.select("query_id", "results"), truth, accs)["ap"].mean()
        rows.append(
            {
                "variant": tag,
                "rate": rate,
                **{f"R@{k}": float(np.mean(v)) for k, v in rec.items()},
                "auprc": float(ap),
                "floor_med": float(np.nanmedian(best_irr)),
                "floor_p90": float(np.nanquantile(best_irr, 0.9)),
                "true_med": float(np.nanmedian(best_true)),
                "margin_med": float(
                    np.nanmedian(np.array(best_true) - np.array(best_irr))
                ),
                "true_wins": float(np.mean(wins)),
            }
        )
    df = pl.DataFrame(rows).sort("rate", "variant")
    with pl.Config(tbl_cols=20, tbl_width_chars=220, float_precision=3, tbl_rows=100):
        print(df)
        print("\nrecall@10 by variant (columns = noise):")
        print(df.pivot(index="variant", on="rate", values="R@10").sort("variant"))
        print("\nmedian floor (best irrelevant accession score):")
        print(df.pivot(index="variant", on="rate", values="floor_med").sort("variant"))
        print("\nmedian true-pair score:")
        print(df.pivot(index="variant", on="rate", values="true_med").sort("variant"))
    df.write_csv(out / "scores.csv")
    print(f"\nwrote {out / 'scores.csv'}")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
