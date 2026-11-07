"""sra500 queries whose source run's organism is neither a LOCALE
reference-genome training organism nor "human metagenome".

Organism per run comes from the Metagraph 100-studies table; training
organisms from the reference_genome_dataset.parquet `organism` column
(8vqiabk9's dataset_path). Writes queries_unseen.parquet with the full
queries.parquet schema, so print_results.py / score_recall.py accept it
as --raw_read_queries_path / --queries and score only these query_ids
(every metric joins results to this file on query_id, inner).
"""

from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
BUNDLE = Path("/pscratch/sd/r/rsynk/locale-data/constructed/sra500/bundle")
TABLE = Path("/pscratch/sd/r/rsynk/locale-data/accessions/TableS12_SRA_Public_100studies.tsv")
REF = Path("/pscratch/sd/r/rsynk/locale_data/reference_genome_dataset.parquet")

train_orgs = set(
    pl.scan_parquet(REF).select("organism").unique().collect()["organism"].to_list()
)
excluded = train_orgs | {"human metagenome"}
print("excluded organisms:", sorted(excluded))

organism = (
    pl.read_csv(TABLE, separator="\t", infer_schema_length=0)
    .select(pl.col("acc").alias("accession"), "organism")
    .unique("accession")
)
queries = pl.read_parquet(BUNDLE / "queries.parquet")
q = queries.join(organism, on="accession", how="left")
assert q["organism"].null_count() == 0, "run without organism metadata"
keep = q.filter(~pl.col("organism").is_in(list(excluded)))
print(f"kept {len(keep)} / {len(queries)} queries")
print(keep.group_by("organism").len().sort("len", descending=True))

out = HERE / "queries_unseen.parquet"
keep.select(queries.columns).write_parquet(out)
keep.select("query_id", "accession", "organism").write_csv(HERE / "queries_unseen_organisms.csv")
print("wrote", out)
