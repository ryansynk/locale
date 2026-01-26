from pathlib import Path

import torch.distributed as dist
from Bio.SeqIO.FastaIO import FastaIterator
from torch.utils.data import IterableDataset, get_worker_info

from .augmenter import SequenceAugmenter


class Batcher(IterableDataset):
    def __init__(self, data_dir, augment_config):
        self.file_paths = list(Path(data_dir).resolve().glob("*.fasta"))
        self.augmenter = SequenceAugmenter(augment_config)
        if not self.file_paths:
            raise FileNotFoundError(f"No .fasta files found in {data_dir}")

    def __iter__(self):
        # 1. IDENTIFY GLOBAL CONTEXT (Node Level)
        if dist.is_available() and dist.is_initialized():  # ty: ignore[possibly-missing-attribute]
            # We are in a multi-node DDP setup
            # Total nodes * GPUs
            world_size = dist.get_world_size()  # ty: ignore[possibly-missing-attribute]
            # My unique ID across the whole cluster
            global_rank = dist.get_rank()  # ty: ignore[possibly-missing-attribute]
        else:
            # Single GPU/CPU debugging
            world_size = 1
            global_rank = 0

        # 2. IDENTIFY LOCAL CONTEXT (Process Level)
        worker_info = get_worker_info()
        if worker_info is None:
            # Main process loading (num_workers=0)
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        # 3. CALCULATE TOTAL GLOBAL WORKERS
        # This is the critical math. We view the entire cluster as one giant list of workers.
        total_workers = world_size * num_workers

        # My unique index in the ENTIRE cluster (0 to total_workers - 1)
        # e.g., Node 2, Worker 3 might be global_worker_id = 67
        my_global_worker_id = (global_rank * num_workers) + worker_id

        # 4. ASSIGN SHARDS
        # Round-robin distribution across the entire cluster
        my_files = [
            f
            for i, f in enumerate(self.file_paths)
            if i % total_workers == my_global_worker_id
        ]

        # --- CRITICAL WARNING ---
        if len(my_files) == 0:
            print(
                f"WARNING: Global Worker {my_global_worker_id} has NO files to read! "
                f"You have {len(self.file_paths)} shards but {total_workers} total workers. "
                "Create more shards!"
            )

        # 5. READ
        for file_path in my_files:
            with open(file_path, "r") as handle:
                for record in FastaIterator(handle):
                    if len(record.seq) < self.augmenter.cfg.min_seq_length:
                        continue

                    query, target = self.augmenter(record.seq)
                    yield query, target
