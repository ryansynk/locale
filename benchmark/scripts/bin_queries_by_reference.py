"""Bin benchmark queries by their best alignment identity to the LOCALE
reference-genome training set, and write one queries parquet per bin.

Every clean query (queries.fa) is aligned with minimap2 to each reference
assembly under refs_dir (the NCBI datasets layout,
ncbi_dataset/data/<accession>/<accession>_*_genomic.fna). Identity is defined
exactly as the bundle's ground truth defines it (locale-data
finalize_query_dataset.py): exact matches ('=' ops of an --eqx CIGAR) over the
full read length, from the best alignment, so soft clipping counts against
the read. Alignment and best-record selection follow locale-data
get_query_alignments_minimap.py, which built the bundle ground truth: the
map-ont preset (minimap2 default, what that script got by passing no -x),
every mapped record considered, and a primary alignment winning a tie on
exact matches. A 0.90 bin edge therefore means what the bundle 0.90 relevance
cutoff means. The best identity over all assemblies puts a query in one bin;
queries with no alignment anywhere go to "unaligned". Mutated copies of a
query inherit its bin, so only the clean reads are aligned.

Outputs under out_dir:
  alignments/<accession>.sam        raw minimap2 output (kept; skipped if present)
  alignments.parquet                best alignment per (query, assembly)
  query_bins.parquet                query_id, best accession/organism, identity, bin
  queries/queries_<bin>.parquet     queries.parquet rows of that bin (full schema),
                                    ready for print_results.py / score_recall.py

  uv run python scripts/bin_queries_by_reference.py \\
      --refs_dir /pscratch/sd/r/rsynk/reference_data/refs \\
      --bundle /pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle \\
      --out_dir results/sra500_binned
"""

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")
MINIMAP2 = "/pscratch/sd/r/rsynk/minimap2-2.30_x64-linux/minimap2"


def read_fasta_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    name = None
    for line in open(path):
        if line.startswith(">"):
            name = line[1:].split()[0]
            lengths[name] = 0
        else:
            lengths[name] += len(line.strip())
    return lengths


def assembly_organisms(refs_dir: Path) -> dict[str, str]:
    """accession -> organism name from NCBI's assembly_data_report.jsonl."""
    out = {}
    for report in refs_dir.rglob("assembly_data_report.jsonl"):
        for line in open(report):
            rec = json.loads(line)
            out[rec["accession"]] = rec.get("organism", {}).get(
                "organismName", rec["accession"]
            )
    return out


def find_assemblies(refs_dir: Path) -> list[tuple[str, Path]]:
    fnas = sorted(refs_dir.rglob("*_genomic.fna"))
    if not fnas:
        raise FileNotFoundError(f"no *_genomic.fna under {refs_dir}")
    # The accession is the containing directory in the NCBI datasets layout.
    # Deriving it from the filename breaks on assembly names that contain
    # underscores (pea is GCF_024323335.1_CAAS_Psat_ZW6_1.0_genomic.fna, which
    # a filename rsplit turns into "GCF_024323335.1_CAAS_Psat_ZW6").
    return [(f.parent.name, f) for f in fnas]


def align(
    minimap2: str, fna: Path, queries_fa: Path, sam: Path, threads: int, preset: str
):
    if sam.exists():
        print(f"[{fna.name}] {sam} exists, reusing", flush=True)
        return
    # -I 8G keeps every assembly (largest: pea, 4.3 Gbp) in one index part;
    # a split index would report hits per part and break best-per-query.
    cmd = [
        minimap2, "-x", preset, "-a", "--eqx", "--sam-hit-only",
        "-I", "8G", "-t", str(threads), str(fna), str(queries_fa),
    ]
    print(f"[{fna.name}] {' '.join(cmd)}", flush=True)
    tmp = sam.with_suffix(".sam.tmp")
    with open(tmp, "w") as out, open(sam.with_suffix(".log"), "w") as err:
        subprocess.run(cmd, stdout=out, stderr=err, check=True)
    tmp.rename(sam)


def best_alignments(sam: Path, read_len: dict[str, int], accession: str) -> list[dict]:
    """Best alignment per query in one SAM, by exact matches.

    Mirrors locale-data get_query_alignments_minimap.py: every mapped record
    counts (secondary and supplementary included), the winner is the highest
    '=' count, and a primary alignment breaks a tie.
    """
    best: dict[str, dict] = {}
    for line in open(sam):
        if line.startswith("@"):
            continue
        f = line.split("\t", 6)
        qid, flag, target, pos, cigar = f[0], int(f[1]), f[2], int(f[3]), f[5]
        if flag & 0x4:  # unmapped
            continue
        matches = sum(int(n) for n, op in CIGAR_RE.findall(cigar) if op == "=")
        is_primary = not (flag & 0x900)
        cur = best.get(qid)
        if (
            cur is None
            or matches > cur["matches"]
            or (matches == cur["matches"] and is_primary and not cur["is_primary"])
        ):
            best[qid] = {
                "query_id": qid,
                "accession": accession,
                "is_primary": is_primary,
                "matches": matches,
                "read_length": read_len[qid],
                "identity": matches / read_len[qid],
                "target": target,
                "target_pos": pos - 1,  # SAM POS is 1-based
                "strand": "-" if flag & 0x10 else "+",
            }
    return list(best.values())


def bin_label(identity: float | None, edges: list[float]) -> str:
    if identity is None:
        return "unaligned"
    if identity < edges[0]:
        return f"lt{edges[0]:.2f}"
    for lo, hi in zip(edges, edges[1:]):
        if lo <= identity < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return f"ge{edges[-1]:.2f}"


def main(
    refs_dir: Path,
    bundle: Path,
    out_dir: Path,
    edges: list[float] = [0.90, 0.95, 0.99],
    minimap2: str = MINIMAP2,
    preset: str = "map-ont",
    jobs: int = 4,
    threads: int | None = None,
):
    """
    Args:
        refs_dir: directory holding the NCBI datasets download (rglob'd for
            *_genomic.fna and assembly_data_report.jsonl).
        bundle: benchmark bundle with queries.fa and queries.parquet.
        out_dir: where alignments, query_bins.parquet and queries/ go.
        edges: ascending identity bin edges; bins are unaligned, <e0,
            [e0,e1), ..., >=eN.
        minimap2: minimap2 executable.
        preset: minimap2 -x preset. map-ont is the minimap2 default and
            what the ground-truth pipeline used; asm5 is far stricter and
            pushes most reads into "unaligned".
        jobs: assemblies aligned concurrently. Each holds its index in RAM,
            ~3.5 bytes/bp: the five large genomes are pea 3.8, human 3.3,
            mouse 2.7, peanut 2.6 and zebrafish 1.4 Gbp, so 4 at once peaks
            near 45 GB and all 7 near 50 GB. Either fits a 512 GB CPU node.
        threads: minimap2 threads per job (default: SLURM_CPUS_PER_TASK or
            os.cpu_count(), divided by jobs).
    """
    import os

    refs_dir, bundle, out_dir = Path(refs_dir), Path(bundle), Path(out_dir)
    edges = sorted(edges)
    queries_fa = bundle / "queries.fa"
    queries = pl.read_parquet(bundle / "queries.parquet")
    read_len = read_fasta_lengths(queries_fa)
    missing = set(queries["query_id"]) - set(read_len)
    if missing:
        raise ValueError(f"{len(missing)} query_ids in queries.parquet not in queries.fa")

    total_threads = threads or (
        int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 8
    )
    per_job = max(1, total_threads // jobs) if threads is None else threads
    assemblies = find_assemblies(refs_dir)
    organisms = assembly_organisms(refs_dir)
    print(f"{len(assemblies)} assemblies, {len(read_len)} queries, "
          f"{jobs} jobs x {per_job} threads", flush=True)

    aln_dir = out_dir / "alignments"
    aln_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(
            lambda a: align(minimap2, a[1], queries_fa, aln_dir / f"{a[0]}.sam", per_job, preset),
            assemblies,
        ))

    rows = []
    for acc, _ in assemblies:
        rows.extend(best_alignments(aln_dir / f"{acc}.sam", read_len, acc))
    schema = {
        "query_id": pl.String, "accession": pl.String, "is_primary": pl.Boolean,
        "matches": pl.Int64, "read_length": pl.Int64, "identity": pl.Float64,
        "target": pl.String, "target_pos": pl.Int64, "strand": pl.String,
    }
    alns = pl.DataFrame(rows, schema=schema).with_columns(
        pl.col("accession").replace_strict(organisms, default=None).alias("organism")
    )
    alns.write_parquet(out_dir / "alignments.parquet")

    best = (
        alns.sort(["query_id", "identity", "matches", "is_primary", "accession"],
                  descending=[False, True, True, True, False])
        .unique("query_id", keep="first", maintain_order=True)
        .select("query_id", "accession", "organism", "identity", "matches",
                "read_length", "is_primary")
    )
    bins = (
        queries.select("query_id", pl.col("accession").alias("source_run"))
        .join(best, on="query_id", how="left")
        .with_columns(
            pl.col("identity")
            .map_elements(lambda x: bin_label(x, edges), return_dtype=pl.String)
            .fill_null("unaligned")
            .alias("bin")
        )
    )
    bins.write_parquet(out_dir / "query_bins.parquet")

    labels = ["unaligned", f"lt{edges[0]:.2f}"] + [
        f"{lo:.2f}-{hi:.2f}" for lo, hi in zip(edges, edges[1:])
    ] + [f"ge{edges[-1]:.2f}"]
    qdir = out_dir / "queries"
    qdir.mkdir(exist_ok=True)
    print("\nqueries per bin:")
    for label in labels:
        ids = bins.filter(pl.col("bin") == label)["query_id"]
        sub = queries.filter(pl.col("query_id").is_in(ids.implode()))
        print(f"  {label:12s} {len(sub):4d}")
        if len(sub):
            sub.write_parquet(qdir / f"queries_{label}.parquet")
    print("\norganism x bin (best-aligned assembly organism; null = unaligned):")
    print(
        bins.group_by("organism", "bin").len().sort("organism", "bin")
        .pivot(on="bin", index="organism", values="len").fill_null(0)
    )
    print(f"\nwrote {out_dir / 'query_bins.parquet'} and {qdir}/queries_<bin>.parquet")


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
