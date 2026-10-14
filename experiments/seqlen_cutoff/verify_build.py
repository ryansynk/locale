"""Check that an index built with EncoderConfig.min_seq_len returns the same
exact top-k results as the post-hoc masked scan of the unfiltered index.

The sweep (sweep_search.py) masks vectors from short contigs inside the scan
of the full index; the build-time filter never embeds them. The two index the
same embeddings for the surviving contigs, so at the same cutoff the exact
top-k hits must agree vector for vector (up to float ties) and the regrouped
per-accession results must agree score for score. This compares both for
every mutation rate that has results on both sides.

    cd benchmark
    uv run python ../experiments/seqlen_cutoff/verify_build.py \
        --built results/sra500/locale@8vqiabk9-endwin50-minlen100/exact/top100 \
        --sweep results/sra500_minlen/locale@8vqiabk9-endwin50/exact-minlen100/top100 \
        --built_index indexes/sra500/locale@8vqiabk9-endwin50-minlen100 \
        --lengths ../experiments/seqlen_cutoff/sra500_endwin50_contig_len.u32 \
        --cutoff 100

Hit vector ids differ between the two indexes (the built one has fewer
rows), so hits are compared after mapping the sweep's ids through the
unfiltered length map: the built index's row r is the r-th surviving row of
the unfiltered index, in order, because both builds walk the same FASTAs in
the same order and only drop rows.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import auto_cli


def _results(d: Path, rate: float) -> pl.DataFrame:
    return pl.read_parquet(d / f"mut{rate:.2f}.parquet").select("query_id", "results")


def _hits(d: Path, rate: float) -> pl.DataFrame | None:
    p = d / "hits" / f"mut{rate:.2f}.parquet"
    return pl.read_parquet(p).select("query_id", "hits") if p.exists() else None


def main(
    built: str,
    sweep: str,
    cutoff: int,
    built_index: str | None = None,
    lengths: str | None = None,
    rates: list[float] = [0.0, 0.05, 0.10],
    score_tol: float = 1e-5,
):
    """
    Args:
        built: results dir of the min_seq_len build (mut<rate>.parquet, hits/).
        sweep: results dir of the sweep at the same cutoff.
        cutoff: the cutoff both used (bp).
        built_index: the built index dir; its meta.parquet is checked for
            zero-row accessions and against the unfiltered length map.
        lengths: unfiltered length map (vector_contig_lengths.py), needed to
            compare hit vector ids.
        score_tol: largest per-accession score difference tolerated (ties
            between vectors at equal cosine are the only expected source).
    """
    built, sweep = Path(built), Path(sweep)
    ok = True

    surviving = None
    if lengths is not None:
        lens = np.memmap(lengths, dtype="<u4", mode="r")
        surviving = np.flatnonzero(np.asarray(lens) >= cutoff)
        print(
            f"length map: {len(lens):,} rows, {len(surviving):,} survive cutoff {cutoff}"
        )
        if built_index is not None:
            meta = pl.read_parquet(Path(built_index) / "meta.parquet")
            n_built = int(meta["num_rows"].sum())
            n_empty = int((meta["num_rows"] == 0).sum())
            print(
                f"built index: {n_built:,} rows over {meta.height} accessions "
                f"({n_empty} with zero rows)"
            )
            if n_built != len(surviving):
                print(
                    f"[FAIL] built index has {n_built:,} rows, expected {len(surviving):,}"
                )
                ok = False

    for rate in rates:
        if not (built / f"mut{rate:.2f}.parquet").exists():
            print(f"rate {rate}: no built results, skipped")
            continue
        a = _results(built, rate).sort("query_id")
        b = _results(sweep, rate).sort("query_id")
        if a["query_id"].to_list() != b["query_id"].to_list():
            print(f"[FAIL] rate {rate}: query sets differ")
            ok = False
            continue
        ea = a.explode("results").unnest("results")
        eb = b.explode("results").unnest("results")
        if ea["accession"].to_list() != eb["accession"].to_list():
            print(f"[FAIL] rate {rate}: accession lists differ")
            ok = False
            continue
        sa, sb = ea["score"].to_numpy(), eb["score"].to_numpy()
        scored_a, scored_b = sa > -2, sb > -2
        n_member = int((scored_a != scored_b).sum())
        both = scored_a & scored_b
        max_diff = float(np.abs(sa[both] - sb[both]).max()) if both.any() else 0.0
        line = (
            f"rate {rate}: {int(scored_a.sum())} scored accessions built vs "
            f"{int(scored_b.sum())} sweep, {n_member} membership mismatches, "
            f"max |score diff| {max_diff:.2e}"
        )
        if n_member or max_diff > score_tol:
            print("[FAIL] " + line)
            ok = False
        else:
            print("[ok]   " + line)

        ha, hb = _hits(built, rate), _hits(sweep, rate)
        if ha is None or hb is None or surviving is None:
            continue
        ha = ha.sort("query_id").explode("hits").unnest("hits")
        hb = hb.sort("query_id").explode("hits").unnest("hits")
        # Sweep ids are rows of the unfiltered index; map to built rows.
        mapped = np.searchsorted(surviving, hb["vector_id"].to_numpy())
        hb = hb.with_columns(pl.Series("vector_id", mapped))
        key = ["query_id", "vector_id"]
        only_a = ha.join(hb, on=key, how="anti").height
        only_b = hb.join(ha, on=key, how="anti").height
        n = ha.height
        print(
            f"        hits: {n:,} built vs {hb.height:,} sweep; "
            f"{only_a} only in built, {only_b} only in sweep (ties expected to be few)"
        )

    print("VERIFIED" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
