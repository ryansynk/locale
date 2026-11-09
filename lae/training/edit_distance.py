"""Exact pairwise edit distances for the edit-distance training loss.

The loss regresses the embedding distance of every (query, key) pair in the
local batch onto the pair's *global* edit distance (edlib mode "NW"): the
whole of each crop counts, including the ends that stick out past the
overlap. For a 103 bp crop contained in a 256 bp crop that is >= 153 before
any mutation, so under this target a short-in-long containment positive can
be as far apart as a random pair (measured 2026-11-08: a third of the paper
recipe's positives normalize above 0.45 against ~0.54 for random pairs).
That is the point of the experiment, not a bug.

Everything here runs in the dataloader workers (collate), on CPU: edlib's
binding holds the GIL, and 64 x 64 pairs of <= 256 bp cost ~20 ms there.
"""

from typing import Literal

import edlib  # ty: ignore unresolved-import
import numpy as np

Normalization = Literal["max_len", "sum_len", "none"]


def edit_distance_matrix(a: list[str], b: list[str]) -> np.ndarray:
    """Global (Levenshtein, edlib NW) edit distance of every a[i] to every b[j].

    Returns an int32 array of shape (len(a), len(b)). Every character counts
    as its own symbol (an N matches only an N)."""
    out = np.empty((len(a), len(b)), dtype=np.int32)
    for i, s in enumerate(a):
        for j, t in enumerate(b):
            out[i, j] = edlib.align(s, t, mode="NW", task="distance")["editDistance"]
    return out


def length_matrix(a: list[str], b: list[str], how: Normalization) -> np.ndarray:
    """Per-pair normalizer matching ``how``: the longer length, the summed
    length, or all ones."""
    la = np.array([len(s) for s in a], dtype=np.int32)[:, None]
    lb = np.array([len(t) for t in b], dtype=np.int32)[None, :]
    if how == "max_len":
        return np.maximum(la, lb)
    if how == "sum_len":
        return la + lb
    if how == "none":
        return np.ones((len(a), len(b)), dtype=np.int32)
    raise ValueError(f"Unknown normalization {how!r}")


def normalized_edit_distance(
    a: list[str], b: list[str], how: Normalization = "max_len"
) -> np.ndarray:
    """``edit_distance_matrix(a, b) / length_matrix(a, b, how)`` as float32.
    With "max_len" or "sum_len" every entry is in [0, 1] (an NW distance never
    exceeds the longer length), 0 only for identical strings."""
    dist = edit_distance_matrix(a, b)
    return (dist / length_matrix(a, b, how)).astype(np.float32)
