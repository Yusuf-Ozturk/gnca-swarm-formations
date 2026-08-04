"""
losses.py
=========
The complete training objective:

    total = formation + ramp * ( separation_weight * soft + collision_weight * hard )

and nothing else. No damping term, no bounds term, no heading regularizer -- if
the swarm is to come to rest in formation, gamma has to discover that itself.

FORMATION -- Kabsch-aligned optimal-assignment MSE.
    Drones are identical (model.py), so "drone k belongs at slot k" is not a
    statement the swarm can act on: relabel the drones and gamma's outputs
    relabel with them, which means a labelled loss asks for something no shared
    rule can deliver. (Measured on the previous labelled model: dropping the
    per-agent identity input while keeping the labelled loss cost 11x-361x in
    final error, with the loss curve flat.) So the target is a point SET and the
    loss is invariant to which drone fills which position:

        L = min over the 4! = 24 assignments pi, of
            mean_k || R(theta*_pi) (p_k - p_bar) - (t_pi(k) - t_bar) ||^2

    where R(theta*_pi) is the optimal rotation for that assignment, in closed
    form (2D Procrustes: theta* = atan2(sum of cross products, sum of dots) --
    exact, no SVD, and it vectorizes over batch, time and assignment at once).
    Translation is removed by centering both point sets. So the loss scores the
    SHAPE alone: any position, any orientation, any assignment of drones to
    positions is equally correct, and only the geometry is graded.

    Why exact assignment rather than a soft/Chamfer surrogate: Chamfer lets two
    drones collapse onto one target position while a third goes unfilled and
    still scores well. A permutation is a bijection, so every position gets
    exactly one drone.

    The loss is averaged over the last `hold_tail` steps rather than scored at
    the final state alone. That is not a parking penalty -- it is measuring the
    formation where it has to hold, so a swarm that sweeps through the right
    shape at speed scores worse than one that settles into it.

SEPARATION / COLLISION -- two-tier violation exposure.
    Per step and per pair, the dimensionless penetration-depth hinge

        relu(1 - d_ij / d_min)^2         (0 when clear, 1 when centers coincide)

    averaged over pairs, then SUMMED over time weighted by dt: the loss is
    "pair-seconds spent inside the safety distance". Summed rather than averaged
    because a time-average divides a brief mid-flight collision by the whole
    horizon, which makes crashing cheaper than the detour that avoids it.

    Two tiers with a shared ramp: a soft one below 2r + margin that shapes a
    buffer, and a hard one below 2r itself -- an actual collision -- priced an
    order of magnitude higher. Both are exactly zero once the swarm keeps its
    distance, so at convergence they never fight the formation term.
"""

from __future__ import annotations

import itertools
from typing import List, Sequence

import torch

from shapes import pairwise_distance_matrix

# All 4! assignments of drones to target positions, built once.
_PERMS_CACHE: dict = {}


def all_permutations(n: int, device=None) -> torch.Tensor:
    """(n!, n) LongTensor of every assignment of n drones to n target slots."""
    key = (n, str(device))
    if key not in _PERMS_CACHE:
        _PERMS_CACHE[key] = torch.tensor(list(itertools.permutations(range(n))),
                                         dtype=torch.long, device=device)
    return _PERMS_CACHE[key]


def assignment_loss(pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pose- and permutation-invariant formation error.

    Args:
        pos:    (..., N, 2) realized positions (any leading batch/time dims).
        target: (N, 2) target point set.
    Returns:
        scalar mean over the leading dims of the best-assignment aligned MSE.

    Method, all vectorized: center both sets, then for each of the N! candidate
    assignments compute the closed-form optimal 2D rotation and its residual MSE,
    and take the minimum. The min is a subgradient operation -- gradients flow
    through the winning assignment, which is the standard treatment and is
    well-behaved here because the winner changes rarely once a shape forms.
    """
    n = target.shape[0]
    perms = all_permutations(n, pos.device)                  # (P, N)

    p = pos - pos.mean(dim=-2, keepdim=True)                 # (..., N, 2)
    t = target - target.mean(dim=0, keepdim=True)            # (N, 2)
    t_perm = t[perms]                                        # (P, N, 2)

    # Broadcast: (..., 1, N, 2) against (P, N, 2) -> (..., P, N, 2)
    p_e = p.unsqueeze(-3)
    # 2D Procrustes in closed form, per assignment:
    #   theta* = atan2( sum_k (p_k x t_k) , sum_k (p_k . t_k) )
    cross = (p_e[..., 0] * t_perm[..., 1] - p_e[..., 1] * t_perm[..., 0]).sum(-1)
    dot = (p_e * t_perm).sum(dim=(-1, -2))
    ang = torch.atan2(cross, dot)                            # (..., P)
    c, s = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)
    rot = torch.stack([c * p_e[..., 0] - s * p_e[..., 1],
                       s * p_e[..., 0] + c * p_e[..., 1]], dim=-1)  # (..., P, N, 2)

    mse = ((rot - t_perm) ** 2).sum(-1).mean(-1)             # (..., P)
    return mse.min(dim=-1).values.mean()


def formation_loss(pos_history: Sequence[torch.Tensor], target: torch.Tensor,
                   hold_tail: int) -> torch.Tensor:
    """`assignment_loss` averaged over the last `hold_tail` recorded states."""
    tail = pos_history[-hold_tail:] if hold_tail < len(pos_history) else pos_history
    return assignment_loss(torch.stack(list(tail)), target)


def separation_loss(pos_history: Sequence[torch.Tensor], min_dist: float,
                    dt: float) -> torch.Tensor:
    """
    Violation exposure below `min_dist`: squared penetration depth, mean over
    pairs, summed over time x dt. Zero whenever every pair stays clear.
    """
    pos = torch.stack(list(pos_history))                 # (T, ..., N, 2)
    dm = pairwise_distance_matrix(pos)                   # (T, ..., N, N)
    n = dm.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=dm.device)
    violation = torch.relu(1.0 - dm / min_dist) ** 2
    violation = violation.masked_fill(eye, 0.0)
    per_step = violation.sum(dim=(-2, -1)) / (n * (n - 1))
    return (per_step * dt).sum(dim=0).mean()


def collision_stats(pos_history: Sequence[torch.Tensor], drone_radius: float):
    """
    Diagnostic (no gradient): (min separation over the rollout, min separation at
    the FINAL state, number of colliding steps). Collision = centers closer than
    2 * drone_radius.
    """
    with torch.no_grad():
        pos = torch.stack(list(pos_history))
        dm = pairwise_distance_matrix(pos)
        n = dm.shape[-1]
        dm = dm + torch.eye(n, device=dm.device) * 1e9
        flat = dm.flatten(start_dim=1) if dm.dim() > 2 else dm.flatten(1)
        per_step = flat.min(dim=-1).values                # (T, ...) -> min per step
        while per_step.dim() > 1:
            per_step = per_step.min(dim=-1).values
        coll = 2.0 * drone_radius
        return (float(per_step.min()), float(per_step[-1]),
                int((per_step < coll).sum()))


def total_loss(cfg, pos_history, target, sep_scale: float = 1.0) -> torch.Tensor:
    """
    THE COMPLETE OBJECTIVE (see module docstring).

        total = formation
              + sep_scale * separation_weight * exposure below 2r + margin
              + sep_scale * collision_weight  * exposure below 2r

    `sep_scale` is the curriculum ramp: at full weight from epoch 0 the collision
    terms dominate and the swarm learns to avoid before it can form at all, so
    the weight ramps in over the first `separation_ramp` fraction of training.
    """
    total = formation_loss(pos_history, target, cfg.hold_tail)
    radius2 = 2.0 * cfg.drone_radius
    if cfg.separation_weight > 0:
        total = total + sep_scale * cfg.separation_weight * separation_loss(
            pos_history, radius2 + cfg.separation_margin, cfg.dt)
    if cfg.collision_weight > 0:
        total = total + sep_scale * cfg.collision_weight * separation_loss(
            pos_history, radius2, cfg.dt)
    return total


if __name__ == "__main__":
    # Self-check: the loss must be blind to pose and to labelling, and must fire
    # on a genuinely wrong shape.
    import math
    from shapes import get_shape

    tgt = get_shape("wedge", 0.8)

    print(f"exact target                  : {assignment_loss(tgt, tgt).item():.3e}")

    th = math.radians(53.0)
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    moved = tgt @ R.T + torch.tensor([3.0, -7.0])
    print(f"rotated 53 deg + shifted (3,-7): {assignment_loss(moved, tgt).item():.3e}")

    shuffled = moved[torch.tensor([2, 0, 3, 1])]
    print(f"...and drones relabelled       : {assignment_loss(shuffled, tgt).item():.3e}")

    wrong = get_shape("square", 0.8)
    print(f"a square scored against wedge  : {assignment_loss(wrong, tgt).item():.4f}")

    for name, val in (("exact", assignment_loss(tgt, tgt)),
                      ("moved", assignment_loss(moved, tgt)),
                      ("shuffled", assignment_loss(shuffled, tgt))):
        assert val < 1e-10, f"{name} should be invariant, got {val}"
    assert assignment_loss(wrong, tgt) > 0.01
    print("losses.py self-check passed")


def align_target_to(pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Map the target point set INTO the swarm's own frame, for visualization.

    Returns (N, 2) target positions rotated/translated (and reordered) to the
    best-scoring assignment for `pos`, so `aligned[k]` is the position drone k is
    being graded against. Purely a drawing aid -- the loss itself never needs the
    pose, but a viewer does, otherwise the dashed target sits somewhere unrelated
    to where the swarm actually settled.
    """
    n = target.shape[0]
    perms = all_permutations(n, pos.device)
    p_bar, t_bar = pos.mean(0, keepdim=True), target.mean(0, keepdim=True)
    p, t = pos - p_bar, target - t_bar
    t_perm = t[perms]                                        # (P, N, 2)

    cross = (p[None, :, 0] * t_perm[..., 1] - p[None, :, 1] * t_perm[..., 0]).sum(-1)
    dot = (p[None] * t_perm).sum(dim=(-1, -2))
    ang = torch.atan2(cross, dot)                            # (P,) rotation of p onto t
    c, s = torch.cos(ang).unsqueeze(-1), torch.sin(ang).unsqueeze(-1)
    rot_p = torch.stack([c * p[None, :, 0] - s * p[None, :, 1],
                         s * p[None, :, 0] + c * p[None, :, 1]], dim=-1)
    best = int(((rot_p - t_perm) ** 2).sum(-1).mean(-1).argmin())

    # Undo that rotation on the target instead, so it lands on the swarm.
    a = -ang[best]
    ca, sa = torch.cos(a), torch.sin(a)
    tb = t_perm[best]
    return torch.stack([ca * tb[:, 0] - sa * tb[:, 1],
                        sa * tb[:, 0] + ca * tb[:, 1]], dim=-1) + p_bar
