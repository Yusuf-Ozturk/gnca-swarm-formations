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

HEADING-RATE (rotation-fix branch): a hinge penalty on excess heading angular
speed between consecutive rollout steps, above a physically-motivated yaw-rate
threshold. The differentiable, LEARNED counterpart to sim.py's hard
max_turn_deg clamp -- rather than clamping the simulated heading after the
fact, this pressures the policy to produce accelerations that don't demand
sharp turns in the first place.
"""

from __future__ import annotations

import math

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


def separation_loss(pos_history, min_dist: float, dt: float) -> torch.Tensor:
    """
    COLLISION-AVOIDANCE loss (issue #4), measured as VIOLATION EXPOSURE. For
    every rollout step and every pair of drones, take the dimensionless
    penetration-depth hinge

        relu(1 - ||pos_i - pos_j|| / min_dist)^2      (0 when safe, 1 at overlap)

    then average over pairs and SUM over time weighted by dt -- i.e. the loss is
    the (squared-depth-weighted) pair-seconds spent inside the safety distance.

    Two properties matter here, both learned the hard way:
      * exactly zero while every pair keeps `min_dist`, so at convergence it
        never fights the formation loss;
      * summed over time, NOT averaged. A time-averaged penalty divides a brief
        transit collision by the whole horizon (T up to 120), shrinking it to
        the same order as the converged formation loss -- training then happily
        trades a mid-flight crash for a marginally straighter path (measured:
        8/8 colliding rollouts at convergence). The exposure sum keeps a
        collision's cost independent of how long the rollout happens to be.

    `min_dist` should be 2*drone_radius (touching safety disks) plus a training
    margin, so the loss pushes back well before an actual collision.
    """
    pos = torch.stack(pos_history)                       # (T, ..., N, 2)
    dm = pairwise_distance_matrix(pos)                   # (T, ..., N, N)
    n = dm.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=dm.device)
    violation = torch.relu(1.0 - dm / min_dist) ** 2
    violation = violation.masked_fill(eye, 0.0)          # ignore self-distances
    # Mean over the N*(N-1) real pairs and any batch dim; sum over time (dim 0).
    per_step = violation.sum(dim=(-2, -1)) / (n * (n - 1))   # (T, ...) pair mean
    return (per_step * dt).sum(dim=0).mean()


def bounds_loss(pos_history, arena_half: float, drone_radius: float,
                dt: float) -> torch.Tensor:
    """
    ARENA CONTAINMENT loss (issue #4, fixed mode), as VIOLATION EXPOSURE.
    Penalize any drone whose safety disk pokes outside the square flight area:

        relu(|coordinate| - (arena_half - drone_radius))^2   per axis,

    averaged over agents/axes and SUMMED over time weighted by dt -- the same
    exposure construction as separation_loss, and for the same reason: a
    time-averaged hinge divides one brief boundary excursion by the whole
    horizon, making it cheaper than the detour that avoids it (measured:
    transit overshoots up to 0.3m past the wall at convergence). Zero for
    every drone flying safely inside; the fixed targets already sit well
    inside the arena, so this only shapes TRANSIT trajectories.
    """
    pos = torch.stack(pos_history)                       # (T, ..., N, 2)
    overshoot = torch.relu(pos.abs() - (arena_half - drone_radius)) ** 2
    per_step = overshoot.mean(dim=tuple(range(1, overshoot.dim())))  # (T,)
    return (per_step * dt).sum()


def heading_rate_loss(heading_history, dt: float, threshold_deg: float = 180.0) -> torch.Tensor:
    """
    HEADING-RATE regularizer (rotation-fix branch), as VIOLATION EXPOSURE --
    same construction as separation_loss/bounds_loss and for the same reason:
    a plain per-step mean would let a trajectory that's fine 90% of the time
    and spikes to ~1800 deg/s the other 10% look nearly as good as one that
    never spikes, since the spike gets diluted by the rest of the (up to
    120-step) horizon.

    For every pair of consecutive rollout steps, compute the signed angle
    between the heading unit vectors via atan2(cross, dot) (numerically
    cleaner near +-1 than arccos of the dot product), convert to an angular
    SPEED in deg/s, and hinge-penalize the excess above `threshold_deg`:

        relu(|angle| * (180/pi) / dt - threshold_deg)^2

    Zero whenever every step's turn stays within the threshold (a physically-
    motivated yaw-rate limit -- no Crazyflie can spin its heading past
    ~threshold_deg deg/s), growing quadratically beyond it. This is the
    differentiable, LEARNED counterpart to sim.py's hard max_turn_deg clamp:
    instead of clamping the simulated heading after the fact, it pressures the
    POLICY to produce accelerations that don't demand sharp turns to begin
    with, so gradients flow back through whatever caused the turn (a sharp
    velocity reversal, a safety-filter correction, ...), not just through the
    heading follower.

    `heading_history` entries are unit heading vectors, (N, 2) or batched
    (B, N, 2); there must be at least 2 entries (T >= 2).
    """
    h = torch.stack(heading_history)                      # (T, ..., N, 2)
    h_prev, h_next = h[:-1], h[1:]
    cross = h_prev[..., 0] * h_next[..., 1] - h_prev[..., 1] * h_next[..., 0]
    dot = (h_prev * h_next).sum(dim=-1)
    angle = torch.atan2(cross, dot)                       # (T-1, ...) signed radians
    angular_speed_deg = torch.abs(angle) * (180.0 / math.pi) / dt
    violation = torch.relu(angular_speed_deg - threshold_deg) ** 2  # (T-1, ..., N)
    per_step = violation.mean(dim=tuple(range(1, violation.dim())))  # (T-1,)
    return (per_step * dt).sum()


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
