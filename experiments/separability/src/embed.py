"""LOCALE embedding for the separability probe -- a thin mirror of dense_index.py.

The one rule here is that this file invents nothing. It reuses
``benchmark/src/encoders.py::LOCALEEncoder`` verbatim (checkpoint loading,
tokenizer, mean pooling over the attention mask, final ``F.normalize(dim=1)``)
and reuses ``benchmark/src/dense_index.py::chunk_sequence`` for tiling. What is
added is only the bookkeeping this probe needs: which window came from which
gene.

Facts read out of the repo rather than assumed, because the probe brief guessed
some of them wrong:

  * The native window is **256 bp**, not a kilobase. ``EncoderConfig.max_seq_len``
    defaults to 256 and every benchmark config in benchmark/configs/ sets 256,
    as does ``configs/config.yaml``'s ``augment_config.max_len`` for the paper
    training run. A 54 kb pks island is therefore ~500 windows, not one vector.
  * Embeddings are **768-d**, the DNABERT-2 backbone hidden size -- not the
    128 of ``TrainConfig.dim``. ``LOCALEEncoder`` mean-pools
    ``model.bert_q(**tokens)[0]`` and never touches the MoCo projection head,
    which exists only for the training objective. They are L2-normalised, so
    cosine == inner product and a "radius" in this report is always cosine
    similarity, higher = closer.
  * ``benchmark/configs/locale_config.yaml`` no longer exists; the paper
    checkpoint is pinned in ``benchmark/src/config.py::PAPER_CHECKPOINT`` and is
    what an unset ``checkpoint_path`` downloads.

Tiling follows the *index* side of dense_index (``DenseIndex._iter_chunks``):
a sequence at most one window long is embedded whole, and anything longer goes
through ``chunk_sequence`` in stride mode with ``chunk_overlap`` (150 by
default, so step 106), which drops a ragged tail rather than embedding a
12 bp fragment. Every sequence in this probe is both a query and a target, so
using one tiling on both sides keeps the similarity matrix symmetric.

Reverse strand: dense_index embeds each query's reverse complement too. This
probe does not, because every sequence here is a CDS pulled out by
``feature.extract``, which already returns the coding strand -- orientation is
normalised at extraction and the second strand would only double the compute.
``reverse_complement`` is re-exported so a caller can check that claim.
"""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "benchmark") not in sys.path:
    # Same trick as tests/conftest.py: benchmark/ on the path makes the
    # benchmark's own `src` package importable without turning this directory
    # (also named src/) into a competing top-level `src`.
    sys.path.insert(0, str(REPO / "benchmark"))

from src.config import PAPER_CHECKPOINT, EncoderConfig  # noqa: E402
from src.dense_index import chunk_sequence, reverse_complement  # noqa: E402,F401
from src.encoders import LOCALEEncoder  # noqa: E402

# The model's native window and the benchmark's stride, taken from the
# dataclass defaults so this probe tracks the repo instead of hardcoding 256.
WINDOW = EncoderConfig.max_seq_len
OVERLAP = EncoderConfig.chunk_overlap
PAPER_CKPT_ID = f"{PAPER_CHECKPOINT['ckpt_id']}/{PAPER_CHECKPOINT['step']}"


def encoder_config(
    device: str = "cuda",
    batch_size: int = 256,
    checkpoint_path: str | None = None,
) -> EncoderConfig:
    """EncoderConfig matching the paper benchmark runs (pooling/window/stride).

    ``checkpoint_path=None`` is the paper checkpoint, downloaded from the Hub at
    the revision pinned in PAPER_CHECKPOINT.
    """
    return EncoderConfig(
        name="locale",
        checkpoint_path=checkpoint_path,
        pooling="mean",
        max_seq_len=WINDOW,
        chunk_overlap=OVERLAP,
        batch_size=batch_size,
        both_strands=False,
        device=device,
    )


def load_encoder(cfg: EncoderConfig | None = None, **kw) -> LOCALEEncoder:
    return LOCALEEncoder(cfg or encoder_config(**kw))


def tile(seq: str, window: int = WINDOW, overlap: int = OVERLAP) -> list[str]:
    """Windows for one sequence, exactly as DenseIndex._iter_chunks yields them."""
    if len(seq) <= window:
        return [seq]
    return chunk_sequence(seq, "probe", window, overlap, "stride")


def tile_many(
    seqs: list[str], window: int = WINDOW, overlap: int = OVERLAP
) -> tuple[list[str], np.ndarray]:
    """Flatten many sequences into windows.

    Returns the window strings and, per window, the index of the sequence it
    came from -- the array every max-over-windows reduction in sweep.py groups
    by.
    """
    windows: list[str] = []
    owner: list[int] = []
    for i, s in enumerate(seqs):
        w = tile(s, window, overlap)
        windows.extend(w)
        owner.extend([i] * len(w))
    return windows, np.asarray(owner, dtype=np.int64)


@torch.no_grad()
def embed(encoder: LOCALEEncoder, seqs: list[str]) -> np.ndarray:
    """(n, 128) float32 L2-normalised embeddings. Empty input -> (0, dim)."""
    if not seqs:
        dim = int(encoder.model.bert_q.config.hidden_size)
        return np.zeros((0, dim), dtype=np.float32)
    out = encoder.encode(seqs)
    return out.detach().float().cpu().numpy()


def embed_windows(
    encoder: LOCALEEncoder,
    seqs: list[str],
    window: int = WINDOW,
    overlap: int = OVERLAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile, embed, and return (vectors, owner) for a list of gene sequences."""
    windows, owner = tile_many(seqs, window, overlap)
    return embed(encoder, windows), owner


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) cosine similarity. Inputs are already unit norm; renormalising
    anyway costs nothing and keeps this honest if a caller passes raw vectors."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return (an @ bn.T).astype(np.float32)


def group_starts(owner: np.ndarray) -> np.ndarray:
    """Row index where each owner's block begins.

    ``tile_many`` emits windows in sequence order, so ``owner`` is sorted and
    every owner owns one contiguous block. That is what lets the max reductions
    below use ``np.maximum.reduceat`` instead of a scatter; this function
    asserts the property rather than assuming it, because a caller that
    reorders windows would otherwise get silently wrong maxima.
    """
    owner = np.asarray(owner)
    if owner.size == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(np.diff(owner) < 0):
        raise ValueError("owner must be sorted (windows grouped by sequence)")
    expected = np.arange(int(owner[-1]) + 1)
    starts = np.searchsorted(owner, expected, side="left")
    if not np.array_equal(np.unique(owner), expected):
        raise ValueError("owner must cover 0..n-1 with no empty groups")
    return starts.astype(np.int64)


def max_over_groups(sim: np.ndarray, owner: np.ndarray, axis: int) -> np.ndarray:
    """Collapse windows to their owning sequence by max, along one axis."""
    starts = group_starts(owner)
    if starts.size == 0:
        shape = list(sim.shape)
        shape[axis] = 0
        return np.zeros(shape, dtype=np.float32)
    return np.maximum.reduceat(sim, starts, axis=axis)


def max_pool_similarity(
    sim: np.ndarray, row_owner: np.ndarray, col_owner: np.ndarray
) -> np.ndarray:
    """Window-level similarity -> gene-level, by max over window pairs.

    This is the late-interaction reduction the brief asks for, and it is
    deliberately NOT a mean pool: pooled accession representation is an open
    problem in this codebase and nothing here needs it solved. Every
    gene-level number in the report is a max over windows and is labelled as
    such.
    """
    return max_over_groups(max_over_groups(sim, col_owner, axis=1), row_owner, axis=0)


def describe_runtime() -> dict:
    """What actually ran -- the report has to state the attention path."""
    try:
        import flash_attn  # noqa: F401

        flash = True
        flash_version = getattr(flash_attn, "__version__", "unknown")
    except Exception:
        flash = False
        flash_version = None
    try:
        import triton  # noqa: F401

        has_triton = True
    except Exception:
        # lae/modeling/bert_layers.py falls further back to plain PyTorch
        # attention without Triton, and says so in a UserWarning.
        has_triton = False
    if flash:
        path = "flash"
    elif has_triton:
        path = "native ALiBi (Triton)"
    else:
        path = "native ALiBi (PyTorch, no Triton)"
    return {
        "flash_attn_installed": flash,
        "flash_attn_version": flash_version,
        "triton_available": has_triton,
        "attention_path": path,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "window_bp": WINDOW,
        "chunk_overlap_bp": OVERLAP,
        "checkpoint": PAPER_CKPT_ID,
    }
