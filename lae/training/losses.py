"""Training objectives other than the InfoNCE in LOCALE.forward."""

import torch
import torch.nn.functional as F


def spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation of two flat tensors (ties broken by order)."""
    x = x.flatten().float()
    y = y.flatten().float()
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return (rx * ry).sum() / (rx.norm() * ry.norm()).clamp_min(1e-12)


def edit_distance_loss(
    q: torch.Tensor,
    k: torch.Tensor,
    n: torch.Tensor | None,
    target_qk: torch.Tensor,
    target_qn: torch.Tensor | None,
    log_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """MSE between a predicted and the exact (normalized) edit distance of
    every local (query, key) and (query, hard negative) pair.

    q, k, n are unit-norm embeddings; the prediction is
    ``exp(log_scale) * (1 - cos)``, monotone in the inner product the benchmark
    ranks by, and pinned to 0 for identical embeddings. The scalar scale is a
    model parameter (so DDP keeps it in sync across ranks) and lets the
    embedding geometry choose how far "unrelated" is.

    Returns the loss, the (b, b [+ b']) prediction matrix, the matching target
    matrix, and scalar diagnostics.
    """
    scale = log_scale.exp()
    cos_qk = q @ k.T  # (b, b); diagonal = the positive pairs
    pred = scale * (1.0 - cos_qk)
    target = target_qk
    if n is not None:
        assert target_qn is not None
        pred = torch.cat([pred, scale * (1.0 - q @ n.T)], dim=1)
        target = torch.cat([target_qk, target_qn], dim=1)

    loss = F.mse_loss(pred, target)

    with torch.no_grad():
        b = q.shape[0]
        eye = torch.eye(b, dtype=torch.bool, device=q.device)
        off = ~eye
        if n is not None:
            off = torch.cat(
                [off, torch.ones(b, n.shape[0], dtype=torch.bool, device=q.device)],
                dim=1,
            )
            eye = torch.cat(
                [eye, torch.zeros(b, n.shape[0], dtype=torch.bool, device=q.device)],
                dim=1,
            )
        metrics = {
            "train/ed_scale": scale.item(),
            "train/ed_spearman": spearman(pred, target).item(),
            "train/ed_target_pos": target[eye].mean().item(),
            "train/ed_target_neg": target[off].mean().item(),
            "train/ed_pred_pos": pred[eye].mean().item(),
            "train/ed_pred_neg": pred[off].mean().item(),
            # Fraction of positives whose target is farther than the batch's
            # median negative: the containment-overhang effect, see
            # lae/training/edit_distance.py.
            "train/ed_pos_beyond_median_neg": (
                target[eye] > target[off].median()
            ).float().mean().item(),
        }
    return loss, pred, target, metrics
