"""Finish merging a multi-node dense index whose shards are all built.

Usage: python finish_merge.py <index_path> <num_shards>

Runs the (resumable) shard merge and marks the index .done so a subsequent
run_benchmark invocation skips straight to search. Safe to rerun: completed
shard copies are recorded in <index_path>/.merge_progress and skipped.
"""

import sys
from pathlib import Path

from src.dense_index import DenseIndex

index_path = Path(sys.argv[1]).resolve()
num_shards = int(sys.argv[2])

# A completed merge deletes its progress file, so rerunning merge_shards after
# success would restart the copy from scratch — skip when already done.
if (index_path / ".done").exists():
    print(f"Index already merged and marked .done: {index_path}")
    sys.exit(0)

missing = [r for r in range(num_shards) if not (index_path / f"shard_{r}" / ".done").exists()]
if missing:
    sys.exit(f"Refusing to merge: shards not complete: {missing}")

DenseIndex.merge_shards(index_path, num_shards)
(index_path / ".done").touch()
print(f"Index marked .done: {index_path}")
