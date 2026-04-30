from jsonargparse import auto_cli
from jsonargparse.typing import Path_fr
from pathlib import Path
import sys
import time

sys.path.insert(0, "/pscratch/sd/r/rsynk/ParlayANN/python")
import wrapper as pann_wp


def main(queries: Path_fr, embeddings: Path_fr, graph: Path_fr):
    queries = Path(queries)
    embeddings = Path(embeddings)
    graph = Path(graph)

    index = pann_wp.load_index(
        "mips", "float", str(embeddings), str(graph), use_quant=False
    )
    start = time.time()
    neighbors, distances = index.batch_search_from_string(
        str(queries), 10, 128, False, 1000
    )
    similarities = -distances
    elapsed = time.time() - start
    print(f"search time = {elapsed}")
    breakpoint()


if __name__ == "__main__":
    auto_cli(main)
