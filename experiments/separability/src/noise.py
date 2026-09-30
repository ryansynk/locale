"""Read-level noise for the query side of the probe.

Two conventions exist in this project and they are not the same thing, so both
are implemented and the report says which produced which number.

``repo`` -- ``lae.training.batcher.Augmenter.augment(seq, identity)``, imported,
not reimplemented. This is the augmentation the paper checkpoint was actually
trained against: it picks ``round(len * (1 - identity))`` positions and makes
each a substitution / insertion / deletion with probability 0.4 / 0.3 / 0.3.
Total divergence is pinned at the target, and indels are 30% of all edits --
a mutation *rate* r means identity 1 - r.

``benchmark`` -- the convention ``locale-data/benchmark/mutate_queries.py`` uses
when it shells out to ``mutation-simulator``: SNP rate r with insertion and
deletion rates r/10 each, indel length 1-2 bp. Reimplemented here in numpy
rather than shelling out, because mutation-simulator is not on PATH in this
environment and because it has no seed flag -- the docstring of
mutate_queries.py says so outright, which makes it unusable for a probe that
has to be rerunnable. The reimplementation follows the documented rates, not
the tool's internals, and the report labels it a reimplementation.

Note the two are not interchangeable at the same nominal r: ``repo`` yields
identity 1-r by construction, ``benchmark`` yields roughly 1 - 1.2r because the
indel rates are added on top of the SNP rate. ``realized_identity`` reports
what actually happened so the curves can be read against real divergence.

Determinism: Augmenter uses the global torch RNG with no generator argument
(see the note in memory about torch seeding), so this module seeds the global
RNG immediately before each call from a per-sequence derived seed. That makes a
rerun reproducible within a process and across processes, at the cost of
perturbing global torch RNG state -- which nothing else in this probe depends on.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lae.training.batcher import Augmenter  # noqa: E402

BASES = np.array(list("ACGT"))
CONVENTIONS = ("repo", "benchmark")
# mutate_queries.py: "insertion and deletion rates r/10 each".
INDEL_FRACTION = 0.1
# mutate_queries.py: "Indel lengths use the simulator defaults (1 to 2 bases)."
INDEL_LEN = (1, 2)


def _seed_for(seq: str, rate: float, salt: int) -> int:
    h = hashlib.sha1(f"{salt}|{rate:.4f}|{seq}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def mutate_repo(seq: str, rate: float, salt: int = 0) -> str:
    """Augmenter.augment at target identity 1 - rate."""
    return mutate_repo_with_identity(seq, rate, salt)[0]


def mutate_repo_with_identity(
    seq: str, rate: float, salt: int = 0
) -> tuple[str, float]:
    """As mutate_repo, but also the identity Augmenter actually achieved.

    Worth a separate entry point because ``_uniform_random_mutation`` computes
    the realized identity itself, exactly, as a by-product of counting its own
    edits -- and it is free. Recovering the same number afterwards with an
    alignment costs O(n*m) in Python, which for a 6.5 kb clbB is ~42M cell
    operations per sequence and dominated the whole probe's runtime.
    """
    if rate <= 0:
        return seq, 1.0
    torch.manual_seed(_seed_for(seq, rate, salt))
    return Augmenter._uniform_random_mutation(seq, 1.0 - rate)


def mutate_benchmark(seq: str, rate: float, salt: int = 0) -> str:
    """SNP rate `rate`, insertion and deletion rates `rate/10` each."""
    if rate <= 0:
        return seq
    rng = np.random.default_rng(_seed_for(seq, rate, salt))
    arr = np.frombuffer(seq.encode(), dtype="S1").astype("U1")
    n = len(arr)

    snp = rng.random(n) < rate
    if snp.any():
        # Substitute with a different base: draw an offset in 1..3 over the
        # ACGT ring, which can never land back on the original.
        idx = np.searchsorted(BASES, arr[snp])
        # Non-ACGT characters (rare, filtered upstream) would misindex; clamp.
        idx = np.clip(idx, 0, 3)
        arr[snp] = BASES[(idx + rng.integers(1, 4, size=int(snp.sum()))) % 4]

    indel_rate = rate * INDEL_FRACTION
    do_ins = rng.random(n) < indel_rate
    do_del = rng.random(n) < indel_rate

    out: list[str] = []
    i = 0
    while i < n:
        if do_del[i]:
            i += int(rng.integers(INDEL_LEN[0], INDEL_LEN[1] + 1))
            continue
        if do_ins[i]:
            k = int(rng.integers(INDEL_LEN[0], INDEL_LEN[1] + 1))
            out.append("".join(rng.choice(BASES, size=k)))
        out.append(arr[i])
        i += 1
    return "".join(out)


def mutate(seq: str, rate: float, convention: str = "repo", salt: int = 0) -> str:
    if convention == "repo":
        return mutate_repo(seq, rate, salt)
    if convention == "benchmark":
        return mutate_benchmark(seq, rate, salt)
    raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")


def mutate_all(
    seqs: list[str], rate: float, convention: str = "repo", salt: int = 0
) -> list[str]:
    return [mutate(s, rate, convention, salt + i) for i, s in enumerate(seqs)]


def realized_identity(
    original: str, mutated: str, max_len: int = 1500
) -> float:
    """Edit-distance identity, 1 - lev/max(len), on the same denominator
    Augmenter._uniform_random_mutation reports against.

    Full Levenshtein DP in Python, so it is O(n*m) and is capped: only the
    first ``max_len`` bases of each sequence are compared. Mutations are
    i.i.d. along the sequence in both conventions, so a 1500 bp prefix is an
    unbiased estimate of the whole sequence's identity, and the alternative --
    42M Python cell operations for one 6.5 kb gene -- dominated the runtime of
    the entire probe.

    For the ``repo`` convention prefer ``mutate_repo_with_identity``, which
    gets the exact figure from the Augmenter for free.
    """
    a, b = original[:max_len], mutated[:max_len]
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))
