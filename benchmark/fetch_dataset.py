"""Download a published benchmark bundle into a dataset_dir.

Usage: python fetch_dataset.py <name> <dataset_dir>

Fetches accs.txt, every per-rate query file and the bundle's json metadata;
run_benchmark.py reads them from dataset_dir and downloads nothing itself
(the contigs under logan_accessions/ are fetched on the first run).
"""

import sys

from huggingface_hub import snapshot_download

DATASETS = {
    "sra50": "rsynk/locale-benchmark-sra50",
    "sra500": "rsynk/locale-benchmark-sra500",
    # 2026-09 rebuild (locale-data list-first pipeline): fresh seeded draws
    # disjoint from the training/validation runs, source run required in the
    # relevant set, both strands aligned (`strand` column).
    "sra50v2": "rsynk/locale-benchmark-sra50-v2",
    "sra500v2": "rsynk/locale-benchmark-sra500-v2",
    "sra55viral": "rsynk/locale-benchmark-sra55viral",
}

if __name__ == "__main__":
    name, dataset_dir = sys.argv[1:]
    path = snapshot_download(
        DATASETS[name],
        repo_type="dataset",
        local_dir=dataset_dir,
        allow_patterns=["accs.txt", "queries_mut*.parquet", "*.json"],
    )
    print(f"{name} -> {path}")
