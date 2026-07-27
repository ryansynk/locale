"""Every trainable backbone parameter must receive a gradient.

A parameter left trainable but unused makes DDP's reducer abort at step 2
("parameters that were not used in producing loss") and puts a None into the
grad-norm logging. Both only surface under torchrun, so they cost a live
allocation to discover -- this reproduces them on CPU in one process.

Downloads model weights, so it is opt-in:

    LAE_RUN_BACKBONE_TESTS=1 uv run python -m pytest tests/test_backbone_gradients.py
"""

import os

import pytest
import torch
import torch.nn.functional as F

from lae.modeling.backbones import BACKBONES, get_tokenizer
from lae.modeling.model import LOCALE

pytestmark = pytest.mark.skipif(
    os.environ.get("LAE_RUN_BACKBONE_TESTS") != "1",
    reason="downloads backbone weights; set LAE_RUN_BACKBONE_TESTS=1 to run",
)

SEQS_Q = ["ACGTACGTAAGCTTGGCATCA" * 3, "TTGACCAGTTAGGCATTACAG" * 2]
SEQS_K = ["ACGTACGTAAGCTTGGCATCT" * 3, "TTGACCAGTTAGGCATTACAT" * 2]

# dnabert2's flash-attn path needs a GPU, so it cannot run in this CPU check.
CPU_BACKBONES = [name for name in BACKBONES if name != "dnabert2"]


def _forward_backward(backbone: str) -> LOCALE:
    model = LOCALE(pooling="mean", dim=128, K=0, m=0.9995, T=0.05, backbone=backbone)
    model.train()
    tokenizer = get_tokenizer(backbone)
    query = tokenizer(SEQS_Q, return_tensors="pt", padding=True)
    key = tokenizer(SEQS_K, return_tensors="pt", padding=True)

    logits, labels = model(query, key, 0.88, is_distributed=False, filter_aligned=False)
    F.cross_entropy(logits, labels).backward()
    return model


@pytest.mark.parametrize("backbone", CPU_BACKBONES)
def test_every_trainable_param_receives_grad(backbone):
    model = _forward_backward(backbone)
    starved = [
        name
        for name, param in model.bert_q.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not starved, (
        f"{backbone}: {len(starved)} trainable params got no gradient — DDP will "
        f"abort at step 2. Freeze them in build_backbone: {starved}"
    )


@pytest.mark.parametrize("backbone", CPU_BACKBONES)
def test_optimizer_param_group_has_no_none_grads(backbone):
    """Mirrors the grad-norm logging in training.py, which crashed on a None."""
    model = _forward_backward(backbone)
    trainable = [p for p in model.bert_q.parameters() if p.requires_grad]
    assert trainable, f"{backbone}: nothing trainable"
    torch.nn.utils.get_total_norm([p.grad for p in trainable])


@pytest.mark.parametrize("backbone", CPU_BACKBONES)
def test_backbone_actually_trains(backbone):
    """Guard against over-freezing: the encoder body must still learn."""
    model = _forward_backward(backbone)
    trainable = sum(p.numel() for p in model.bert_q.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.bert_q.parameters())
    assert trainable / total > 0.9, (
        f"{backbone}: only {trainable}/{total} params trainable — too much frozen"
    )
