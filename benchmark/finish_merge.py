"""Finish merging a multi-node index whose shards are all built.

Usage: python finish_merge.py <index_path> <num_shards>

Runs the index type's shard merge and marks the index .done so a subsequent
run_benchmark invocation skips straight to search. The type is read off
shard_0/: a dense shard holds embeddings.fbin, a metagraph shard
graph_primary.dbg. Safe to rerun: the dense merge records completed shard
copies in <index_path>/.merge_progress and skips them; the metagraph merge
only writes graphs.csv and the joined manifest.
"""

import sys
from pathlib import Path

from src.dense_index import DenseIndex
from src.metagraph_index import GRAPH_FILE, MetagraphIndex

index_path = Path(sys.argv[1]).resolve()
num_shards = int(sys.argv[2])

# A completed dense merge deletes its progress file, so rerunning merge_shards
# after success would restart the copy from scratch — skip when already done.
if (index_path / ".done").exists():
    print(f"Index already merged and marked .done: {index_path}")
    sys.exit(0)

missing = [r for r in range(num_shards) if not (index_path / f"shard_{r}" / ".done").exists()]
if missing:
    sys.exit(f"Refusing to merge: shards not complete: {missing}")

shard0 = index_path / "shard_0"
if (shard0 / "embeddings.fbin").exists():
    index_cls = DenseIndex
elif (shard0 / GRAPH_FILE).exists():
    index_cls = MetagraphIndex
else:
    sys.exit(f"Cannot tell the index type from {shard0}: no embeddings.fbin or {GRAPH_FILE}")

index_cls.merge_shards(index_path, num_shards)
(index_path / ".done").touch()
print(f"Index marked .done: {index_path}")
