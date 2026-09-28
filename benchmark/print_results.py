"""Print benchmark result tables without plotting.

plot_results.py's main() interleaves tables and figures, and two of its figure
calls are hardcoded to mutation_rate=10 (plot_recall_at_k_vs_k_line's default at
line 447, plot_r_precision_vs_time's at line 595). On a single-rate run those
filters empty the frame and pl.from_dicts([]) raises, which kills the process
before the AUPRC and systems tables ever print.

This reuses the same functions -- no metric logic is duplicated -- but calls
only the four that emit tables, so it works with any set of mutation rates.
"""

import tempfile
from pathlib import Path

import polars as pl
from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr
from matplotlib import pyplot as plt

import plot_results as pr

# The two table-emitting functions also savefig. We throw the PDFs away, so skip
# the LaTeX round-trip that makes that slow.
plt.rcParams["text.usetex"] = False

def main(
    results_dir: str,
    raw_read_queries_path: Path_fr,
    accessions: Path_fr,
    k: int = 7,
    bootstrap_samples: int = 10000,
    full_model_names: bool | None = None,
):
    """
    Args:
        full_model_names: keep full model ids in the R-precision and recall
            tables instead of the paper's first-token display names ("LOCALE").
            The default (None) decides from the data: full ids whenever the
            short names would pool distinct models -- e.g. the full-dense
            baseline and exact top-k runs at several k are all "LOCALE" and would otherwise be averaged into one row.
            The AUPRC table always uses full ids.
    """
    results_dir = Path(results_dir)
    with open(accessions) as f:
        accs = f.read().splitlines()

    data = pr.load_results([results_dir])

    oracle = pr.raw_read_oracle_results(pl.read_parquet(Path(raw_read_queries_path)))
    oracle = oracle.join(data.select("mutation_rate").unique(), how="cross")

    rates = sorted(data["mutation_rate"].unique().to_list())
    models = sorted(data["model"].unique().to_list())
    print(f"Loaded {len(data)} rows")
    print(f"models        : {models}")
    print(f"mutation rates: {rates}")

    if full_model_names is None:
        n_display = data.select(pr._display_model()).n_unique()
        full_model_names = n_display < len(models)
        if full_model_names:
            print(
                "Several models share a first-token display name; keeping full "
                "model ids in the R-precision/recall tables so they are not pooled."
            )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pr.plot_r_precision_vs_noise_line(
            data,
            oracle,
            accs,
            tmp,
            bootstrap_samples,
            print_data=True,
            short_model_names=not full_model_names,
        )
        pr.plot_recall_at_k_vs_noise_line(
            data,
            oracle,
            accs,
            k,
            tmp,
            bootstrap_samples,
            print_data=True,
            short_model_names=not full_model_names,
        )
    pr.print_auprc(data, oracle, accs, bootstrap_samples)
    print("=========== SYSTEMS DATA =============")
    pr.print_systems_data(data)


if __name__ == "__main__":
    auto_cli(main)
