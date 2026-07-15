"""
losses.py
=========
Training objectives. All are differentiable.

PRIMARY (arena_mode none/walls): pairwise-distance-matrix MSE. We compare the NxN
inter-agent distance matrix of the realized configuration to the target shape's
distance matrix. Because distances are unchanged by rotating or translating the
whole formation, this loss treats every rotated/translated copy of the target as
equally correct -- the target is an orbit, not a single point. (That is also why
we need a damping term.)

PRIMARY (arena_mode fixed -- issue #4, method 1): plain coordinate MSE against the
target points at their FIXED location, centered in the flight area. Deliberately
NOT invariant: the whole point of method 1 is that the goal shape lives at one
absolute place inside the 3m x 3m arena, so being in the right shape at the wrong
place is wrong. (The model then needs absolute position as an input -- see
model.py -- which the lighthouse positioning system physically provides.)

REGULARIZER: velocity damping near the end of the rollout, so the formation parks
instead of drifting/spinning. The invariant loss alone cannot pin down global
rigid-body motion, so without this the swarm can settle into a slow rotation.

COLLISION AVOIDANCE (issue #4): a hinge penalty on every pair of drones at EVERY
rollout step (not just the tail -- a mid-transit crash is just as fatal as one in
formation). Zero whenever all pairs keep their distance, quadratically increasing
once any pair gets closer than the safety distance.

ARENA CONTAINMENT (issue #4, fixed mode): a hinge penalty for any drone whose
safety disk crosses the arena boundary at any rollout step. In walls mode this is
unnecessary -- the wall force in sim.py handles containment physically.

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


def fixed_formation_loss(pos: torch.Tensor, target_pts: torch.Tensor) -> torch.Tensor:
    """
    Coordinate MSE between the realized config `pos` (..., N, 2) and the target
    points AT THEIR FIXED ARENA LOCATION (issue #4, method 1). Agent k is
    compared against slot k directly, so this pins down position, rotation, and
    assignment all at once -- no invariance anywhere.
    """
    return torch.mean((pos - target_pts) ** 2)


def fixed_formation_hold_loss(pos_history, target_pts: torch.Tensor,
                              tail: int) -> torch.Tensor:
    """
    `fixed_formation_loss` averaged over the last `tail` recorded positions --
    the fixed-target counterpart of `formation_hold_loss`, keeping "stay parked
    on the target" part of the objective (issue #2). `pos_history` entries may
    be (N, 2) or batched (B, N, 2); `target_pts` must broadcast against them.
    """
    tail_pos = pos_history[-tail:] if tail < len(pos_history) else pos_history
    return fixed_formation_loss(torch.stack(tail_pos), target_pts)


def separation_loss(pos_history, min_dist: float) -> torch.Tensor:
    """
    COLLISION-AVOIDANCE hinge (issue #4). For every rollout step and every pair
    of drones, penalize

        relu(min_dist - ||pos_i - pos_j||)^2

    i.e. exactly zero while every pair keeps at least `min_dist` between centers
    (so it never fights the formation loss once the swarm is safely spread), and
    growing quadratically as a pair penetrates the safety distance. It is
    averaged over ALL recorded steps and all ordered pairs, because a collision
    in transit breaks a real drone just as surely as one in formation.

    `min_dist` should be 2*drone_radius (touching safety disks) plus a small
    training margin, so the loss starts pushing slightly before an actual
    collision would occur.
    """
    pos = torch.stack(pos_history)                       # (T, ..., N, 2)
    dm = pairwise_distance_matrix(pos)                   # (T, ..., N, N)
    n = dm.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=dm.device)
    violation = torch.relu(min_dist - dm) ** 2
    violation = violation.masked_fill(eye, 0.0)          # ignore self-distances
    # Mean over the N*(N-1) real pairs (not N*N), every step, every batch item.
    return violation.sum(dim=(-2, -1)).mean() / (n * (n - 1))


def bounds_loss(pos_history, arena_half: float, drone_radius: float) -> torch.Tensor:
    """
    ARENA CONTAINMENT hinge (issue #4, fixed mode). Penalize, at every rollout
    step, any drone whose safety disk pokes outside the square flight area:

        relu(|coordinate| - (arena_half - drone_radius))^2   per axis.

    Zero for every drone flying safely inside. The fixed targets already sit
    well inside the arena, so this only shapes TRANSIT trajectories, which the
    fixed-target loss alone says nothing about.
    """
    pos = torch.stack(pos_history)                       # (T, ..., N, 2)
    overshoot = torch.relu(pos.abs() - (arena_half - drone_radius))
    return torch.mean(overshoot ** 2)


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
