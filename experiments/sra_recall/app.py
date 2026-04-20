import sys
from pathlib import Path

import polars as pl
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from plot_results import (
    calculate_recall_precision_df,
    gencode_oracle_results,
    get_ground_truth,
    plot_auprc,
    plot_contig_len_hit_at_k,
    plot_recall_at_k,
    plot_recall_precision,
    plot_recall_vs_noise_bar,
    plot_recall_vs_noise_line,
    plot_recall_vs_time,
    raw_read_oracle_results,
)

RESULTS_SCHEMA = pl.Schema(
    {
        "query_id": pl.String,
        "results": pl.List(pl.Struct({"accession": pl.String, "score": pl.Float64})),
        "model": pl.String,
        "mutation_rate": pl.Float64,
        "query_type": pl.String,
        "checkpoint": pl.String,
        "max_len": pl.Int64,
        "checkpoint_step_num": pl.Int64,
        "chunk_type": pl.String,
        "avg_time": pl.Float64,
    }
)

st.set_page_config(layout="wide", page_title="SRA Recall Benchmark")
st.title("SRA Recall Benchmark")

# ── Sidebar: paths ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Paths")
    results_dir = st.text_input(
        "Results directory",
        value="results",
    )
    raw_read_queries_path = st.text_input(
        "Raw read queries parquet",
        value="/pscratch/sd/r/rsynk/rawbert_data/data/sra_recall/raw_read_queries_final.parquet",
    )
    gencode_queries_path = st.text_input(
        "Gencode queries parquet",
        value="/pscratch/sd/r/rsynk/rawbert_data/data/sra_recall/gencode/gencode_queries.parquet",
    )
    st.header("Plot options")
    k = st.number_input("k (for hit@k / recall@k plots)", min_value=1, value=7, step=1)
    time_mut_rate = st.number_input(
        "Mutation rate for recall-vs-time plot",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
    )


# ── Load all experiment metadata (fast — no results column) ──────────────────
@st.cache_data
def scan_experiments(results_dir: str) -> pl.DataFrame:
    frames = []
    for f in Path(results_dir).rglob("*.parquet"):
        df = pl.read_parquet(
            f,
            columns=[
                "model",
                "mutation_rate",
                "query_type",
                "checkpoint_step_num",
                "chunk_type",
            ],
        )
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).unique()


meta = scan_experiments(results_dir)

if meta.is_empty():
    st.warning(f"No parquet files found in '{results_dir}'.")
    st.stop()

all_models = sorted(meta["model"].unique().to_list())
all_mutation_rates = sorted(meta["mutation_rate"].unique().to_list())
all_query_types = sorted(meta["query_type"].unique().to_list())

# ── Sidebar: filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    selected_models = st.multiselect("Models", all_models, default=all_models)
    selected_mutation_rates = st.multiselect(
        "Mutation rates", all_mutation_rates, default=all_mutation_rates
    )
    selected_query_types = st.multiselect(
        "Query types", all_query_types, default=all_query_types
    )

if not selected_models:
    st.info("Select at least one model in the sidebar.")
    st.stop()


# ── Load full data for selected filters ──────────────────────────────────────
@st.cache_data
def load_data(
    results_dir: str,
    models: tuple[str, ...],
    mutation_rates: tuple[float, ...],
    query_types: tuple[str, ...],
) -> pl.DataFrame:
    frames = []
    for f in Path(results_dir).rglob("*.parquet"):
        df = pl.read_parquet(f, schema=RESULTS_SCHEMA)
        df = df.filter(
            pl.col("model").is_in(list(models))
            & pl.col("mutation_rate").is_in(list(mutation_rates))
            & pl.col("query_type").is_in(list(query_types))
        )
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return pl.DataFrame(schema=RESULTS_SCHEMA)
    return pl.concat(frames)


@st.cache_data
def load_ground_truth(
    raw_read_queries_path: str,
    gencode_queries_path: str,
    mutation_rates: tuple[float, ...],
):
    raw_read_queries_df = pl.read_parquet(raw_read_queries_path)
    gencode_queries_df = pl.read_parquet(gencode_queries_path)
    raw_read_oracle_data = raw_read_oracle_results(raw_read_queries_df)
    gencode_oracle_data = gencode_oracle_results(gencode_queries_df)
    combos = pl.DataFrame({"mutation_rate": list(mutation_rates)})
    raw_read_oracle_data = raw_read_oracle_data.join(combos, how="cross")
    gencode_oracle_data = gencode_oracle_data.join(combos, how="cross")
    ground_truth = get_ground_truth(raw_read_queries_df, gencode_oracle_data, combos)
    return raw_read_oracle_data, gencode_oracle_data, ground_truth


@st.cache_data
def compute_recall_precision(
    results_dir: str,
    raw_read_queries_path: str,
    gencode_queries_path: str,
    models: tuple[str, ...],
    mutation_rates: tuple[float, ...],
    query_types: tuple[str, ...],
):
    data = load_data(results_dir, models, mutation_rates, query_types)
    raw_read_oracle_data, gencode_oracle_data, ground_truth = load_ground_truth(
        raw_read_queries_path, gencode_queries_path, mutation_rates
    )
    data = pl.concat([data, raw_read_oracle_data, gencode_oracle_data], how="diagonal")
    recall_precision_df = calculate_recall_precision_df(ground_truth, data)
    return recall_precision_df, ground_truth, data


with st.spinner("Computing recall/precision metrics..."):
    try:
        recall_precision_df, ground_truth, full_data = compute_recall_precision(
            results_dir,
            raw_read_queries_path,
            gencode_queries_path,
            tuple(sorted(selected_models)),
            tuple(sorted(selected_mutation_rates)),
            tuple(sorted(selected_query_types)),
        )
    except Exception as e:
        st.error(f"Error computing metrics: {e}")
        st.stop()


# ── Tabs: one per plot type ───────────────────────────────────────────────────
tabs = st.tabs(
    [
        "Precision-Recall",
        "Recall @ K",
        "AUPRC",
        "Recall vs Mutation Rate",
        "Recall vs Mutation Rate (Bar)",
        "Recall vs Time",
        "Contig Len Hit@K",
    ]
)


def show_charts(charts: list[tuple[str, object]]):
    if not charts:
        st.info("No data for current selection.")
        return
    for label, chart in charts:
        st.subheader(label)
        st.altair_chart(chart, use_container_width=False)


with tabs[0]:
    show_charts(plot_recall_precision(recall_precision_df))

with tabs[1]:
    show_charts(plot_recall_at_k(recall_precision_df))

with tabs[2]:
    show_charts(plot_auprc(recall_precision_df))

with tabs[3]:
    show_charts(plot_recall_vs_noise_line(recall_precision_df, k=k))

with tabs[4]:
    show_charts(plot_recall_vs_noise_bar(recall_precision_df, k=k))

with tabs[5]:
    show_charts(
        plot_recall_vs_time(recall_precision_df, k=k, mutation_rate=time_mut_rate)
    )

with tabs[6]:
    show_charts(plot_contig_len_hit_at_k(ground_truth, full_data, k=k))
