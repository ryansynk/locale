import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

from .config import MMSeqs2Config


def _write_fasta(sequences: list[str], ids: list[str], path: Path):
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for seq_id, seq in zip(ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")


class MMSeqs2Searcher:
    """
    Wraps the mmseqs2 CLI to perform nucleotide sequence search.

    Unlike the encoder/indexer pattern used by other baselines, mmseqs2 is a
    monolithic C++ tool where encoding, indexing, and searching all happen in
    a single CLI call. This class writes sequences to temporary FASTA files,
    invokes `mmseqs easy-search`, and parses the results back into the tensor
    format expected by the benchmark.
    """

    def __init__(self, cfg: MMSeqs2Config):
        self.sensitivity = cfg.sensitivity
        self.search_type = cfg.search_type
        self.threads = cfg.threads
        self.mmseqs_binary = cfg.mmseqs_binary

    def search(
        self,
        query_seqs: list[str],
        target_seqs: list[str],
        topk: int,
    ) -> torch.Tensor:
        """
        Run mmseqs2 easy-search and return top-k target indices per query.

        Args:
            query_seqs: List of query DNA sequences.
            target_seqs: List of target DNA sequences.
            topk: Number of top hits to return per query.
            valid_targets_mask: Boolean tensor (num_queries, num_targets).
                If provided, hits to masked-out targets are skipped.

        Returns:
            torch.Tensor of shape (num_queries, topk) with integer target indices.
            Padded with -1 where fewer than topk hits are found.
        """
        # Use integer IDs so we can map results back to indices unambiguously
        query_int_ids = [str(i) for i in range(len(query_seqs))]
        target_int_ids = [str(i) for i in range(len(target_seqs))]

        # Build lookup from integer string ID -> index
        target_id_to_idx = {tid: i for i, tid in enumerate(target_int_ids)}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            query_fasta = tmpdir / "queries.fasta"
            target_fasta = tmpdir / "targets.fasta"
            result_file = tmpdir / "results.m8"

            _write_fasta(query_seqs, query_int_ids, query_fasta)
            _write_fasta(target_seqs, target_int_ids, target_fasta)

            # mmseqs easy-search: query.fasta target.fasta result.m8 tmp/
            # --search-type 3 = nucleotide search
            # --format-output query,target,evalue = minimal output for parsing
            # --max-seqs N = max results per query (request extra to account for masking)
            cmd = [
                self.mmseqs_binary,
                "easy-search",
                str(query_fasta),
                str(target_fasta),
                str(result_file),
                str(tmpdir / "tmp"),
                "--search-type",
                str(self.search_type),
                "-s",
                str(self.sensitivity),
                "--threads",
                str(self.threads),
                "--max-seqs",
                str(topk),
                "--format-output",
                "query,target,evalue",
            ]

            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"mmseqs2 failed (exit {result.returncode}):\n{result.stderr}"
                )

            # Parse m8 results: each line is "query_id\ttarget_id\tevalue"
            # Results are already sorted by e-value (best first) per query
            hits: dict[int, list[int]] = defaultdict(list)
            if result_file.exists():
                for line in result_file.read_text().splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    q_idx = int(parts[0])
                    t_idx = target_id_to_idx[parts[1]]
                    hits[q_idx].append(t_idx)

        # Build predictions tensor, applying valid_targets_mask if provided
        num_queries = len(query_seqs)
        predictions = torch.full((num_queries, topk), -1, dtype=torch.long)

        for q_idx in range(num_queries):
            count = 0
            for t_idx in hits.get(q_idx, []):
                predictions[q_idx, count] = t_idx
                count += 1
                if count >= topk:
                    break

        return predictions
