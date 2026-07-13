"""
losses.py
=========
Training objectives. All are differentiable and (the primary one) invariant to
global rotation and translation.

PRIMARY: pairwise-distance-matrix MSE. We compare the NxN inter-agent distance
matrix of the realized configuration to the target shape's distance matrix. Because
distances are unchanged by rotating or translating the whole formation, this loss
treats every rotated/translated copy of the target as equally correct -- the target
is an orbit, not a single point. (That is also why we need a damping term.)

REGULARIZER: velocity damping near the end of the rollout, so the formation parks
instead of drifting/spinning. The invariant loss alone cannot pin down global
rigid-body motion, so without this the swarm can settle into a slow rotation.

STRETCH (use_chamfer): permutation-invariant loss. We Kabsch/Procrustes-align the
realized points to the target (optimal rigid transform via SVD), then take a
symmetric nearest-neighbour (Chamfer) distance, so any agent may fill any slot.
"""

from __future__ import annotations

import torch

from shapes import pairwise_distance_matrix


def distance_matrix_loss(pos: torch.Tensor, target_dm: torch.Tensor) -> torch.Tensor:
    """
    Pairwise-distance-matrix MSE between the realized config `pos` (N,2) and a
    precomputed target distance matrix `target_dm` (N,N). Rotation/translation
    invariant by construction.
    """
    dm = pairwise_distance_matrix(pos)
    return torch.mean((dm - target_dm) ** 2)


def formation_hold_loss(pos_history, target_dm: torch.Tensor, tail: int) -> torch.Tensor:
    """
    Distance-matrix MSE averaged over the last `tail` recorded positions of the
    rollout, instead of only the very last state. Penalizing the whole tail makes
    "stay in formation" part of the objective rather than just "arrive": a
    trajectory that reaches the target and wobbles (or passes through it) scores
    worse than one that parks there, which is what turns the target into a genuine
    attractor that survives long horizons (issue #2). tail=1 recovers the old
    end-state-only loss.

    `pos_history` entries may be (N, 2) or batched (B, N, 2); `target_dm` must
    broadcast against the resulting (tail, ..., N, N) distance matrices.
    """
    tail_pos = pos_history[-tail:] if tail < len(pos_history) else pos_history
    return distance_matrix_loss(torch.stack(tail_pos), target_dm)


def damping_loss(vel_history, tail: int) -> torch.Tensor:
    """
    Mean squared speed over the last `tail` steps of the rollout. Drives the
    formation to a standstill at the end (which is why headings need the
    persistent fallback -- see graph.py).
    """
    tail_vels = vel_history[-tail:] if tail < len(vel_history) else vel_history
    sq = torch.stack([torch.mean(v ** 2) for v in tail_vels])
    return torch.mean(sq)


def kabsch_align(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Optimal rigid alignment (rotation + translation, no scaling) of `src` onto
    `dst` via orthogonal Procrustes / Kabsch. Returns the aligned `src`.
    Both are (N, 2).
    """
    src_c = src - src.mean(dim=0, keepdim=True)
    dst_c = dst - dst.mean(dim=0, keepdim=True)
    h = src_c.t() @ dst_c                       # (2, 2) covariance
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.det(vt.t() @ u.t()))   # reflection correction
    diag = torch.diag(torch.tensor([1.0, d], dtype=src.dtype, device=src.device))
    rot = vt.t() @ diag @ u.t()                 # (2, 2) rotation
    return src_c @ rot.t() + dst.mean(dim=0, keepdim=True)


def chamfer_loss(pos: torch.Tensor, target_pts: torch.Tensor) -> torch.Tensor:
    """
    Permutation-invariant aligned Chamfer distance. Kabsch-align `pos` to the
    target points, then average the symmetric nearest-neighbour squared distances.
    """
    aligned = kabsch_align(pos, target_pts)
    d = torch.cdist(aligned, target_pts) ** 2   # (N, N)
    forward = d.min(dim=1).values.mean()
    backward = d.min(dim=0).values.mean()
    return 0.5 * (forward + backward)
