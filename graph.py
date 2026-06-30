"""
graph.py
========
Dynamic interaction-graph construction for the swarm.

Two non-obvious pieces live here, both flagged below:

  1. THE CONE / FORWARD FIELD-OF-VIEW. Each agent i only "sees" agents that fall
     inside an angular sector ahead of it. "Ahead" is relative to agent i's own
     heading, which keeps the whole rule rotation/translation equivariant.

  2. THE PERSISTENT-HEADING FALLBACK. At convergence velocities -> 0, so the
     instantaneous heading (velocity direction) is undefined. We keep a per-agent
     heading that only updates when the agent is moving fast enough, and otherwise
     holds its last valid value. The cone is defined by this persistent heading.

Edges are DIRECTED and asymmetric: i seeing j does NOT imply j sees i. We return a
boolean adjacency mask `adj[i, j] == True` meaning "j is in i's cone" (i is the
receiver, j is the sender). A row with no True entries means that agent has no
neighbours this step and will receive no messages -- expected, not a bug.
"""

from __future__ import annotations

import torch


def update_heading(
    heading: torch.Tensor,
    vel: torch.Tensor,
    speed_eps: float = 1e-3,
) -> torch.Tensor:
    """
    PERSISTENT-HEADING FALLBACK.

    Update each agent's persistent unit heading from its current velocity, but
    ONLY for agents whose speed exceeds `speed_eps`. Slow/stationary agents keep
    their previous heading. This is what lets the cone stay well-defined even when
    the formation has parked and velocities have decayed to ~0.

    Args:
        heading: (N, 2) previous persistent unit headings.
        vel:     (N, 2) current velocities.
    Returns:
        (N, 2) updated unit headings.
    """
    speed = torch.linalg.norm(vel, dim=-1, keepdim=True)  # (N, 1)
    moving = (speed > speed_eps).float()                  # (N, 1) gate
    # Normalize velocity where it is meaningful; fall back to old heading otherwise.
    safe_speed = torch.clamp(speed, min=speed_eps)
    new_dir = vel / safe_speed
    updated = moving * new_dir + (1.0 - moving) * heading
    # Renormalize for numerical hygiene (old heading is already ~unit).
    updated = updated / torch.clamp(
        torch.linalg.norm(updated, dim=-1, keepdim=True), min=1e-8
    )
    return updated


def init_heading(vel: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Initialize persistent headings from the (small random) initial velocities.
    For any agent whose initial speed is ~0 we draw a random unit vector so the
    cone is well-defined from step 0.
    """
    n = vel.shape[0]
    speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
    moving = (speed > 1e-6).float()
    safe = torch.clamp(speed, min=1e-6)
    from_vel = vel / safe

    angles = torch.rand(n, generator=generator) * 2.0 * torch.pi
    rand_dir = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

    heading = moving * from_vel + (1.0 - moving) * rand_dir
    heading = heading / torch.clamp(
        torch.linalg.norm(heading, dim=-1, keepdim=True), min=1e-8
    )
    return heading


def cone_adjacency(
    pos: torch.Tensor,
    heading: torch.Tensor,
    sensing_range: float,
    half_angle_rad: float,
) -> torch.Tensor:
    """
    CONE / FORWARD FIELD-OF-VIEW edge construction.

    Build the directed adjacency mask for one timestep. `adj[i, j]` is True iff
    agent j lies inside agent i's forward cone:
        ||pos_j - pos_i|| <= sensing_range
        AND angle( heading_i , pos_j - pos_i ) <= half_angle

    Args:
        pos:            (N, 2) positions.
        heading:        (N, 2) persistent unit headings (receiver's facing dir).
        sensing_range:  scalar radius.
        half_angle_rad: scalar cone half-angle in radians.
    Returns:
        adj: (N, N) boolean. Self-loops are excluded.
    """
    n = pos.shape[0]
    # rel[i, j] = pos_j - pos_i  (vector from receiver i to sender j)
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N, N, 2)
    dist = torch.linalg.norm(rel, dim=-1)              # (N, N)

    # Unit direction from i to j (guard against the zero self-vector).
    safe_dist = torch.clamp(dist, min=1e-8).unsqueeze(-1)
    rel_unit = rel / safe_dist                         # (N, N, 2)

    # cos of angle between heading_i and direction(i->j).
    # heading_i broadcasts across j (dim 1).
    cos_angle = (heading.unsqueeze(1) * rel_unit).sum(dim=-1)  # (N, N)
    cos_thresh = torch.cos(torch.tensor(half_angle_rad, dtype=pos.dtype))

    within_range = dist <= sensing_range
    within_cone = cos_angle >= cos_thresh

    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    adj = within_range & within_cone & (~eye)
    return adj


def knn_adjacency(pos: torch.Tensor, k: int) -> torch.Tensor:
    """
    k-NEAREST-NEIGHBOUR edge construction -- the BASELINE perception to compare
    against the cone FOV.

    Unlike the cone, kNN is OMNIDIRECTIONAL and heading-independent: agent i
    connects to its k closest agents regardless of which way i is facing. It uses
    only inter-agent distances, so it is rotation/translation invariant just like
    the cone, which makes for a clean apples-to-apples comparison -- the only thing
    that changes is *what each agent is allowed to see*.

    Same convention as `cone_adjacency`: `adj[i, j]` True means j is one of i's
    neighbours (i is the receiver). Edges are still directed/asymmetric -- j being
    among i's k nearest does not imply i is among j's k nearest. Self is excluded,
    and every agent always has exactly min(k, N-1) neighbours (no empty cones here:
    that asymmetry of information is precisely the point of the comparison).

    Args:
        pos: (N, 2) positions.
        k:   number of neighbours per agent.
    Returns:
        adj: (N, N) boolean. Self-loops excluded.
    """
    n = pos.shape[0]
    rel = pos.unsqueeze(0) - pos.unsqueeze(1)          # (N, N, 2), rel[i,j]=pos_j-pos_i
    dist = torch.linalg.norm(rel, dim=-1)              # (N, N)
    # Exclude self by pushing the diagonal to +inf before taking nearest.
    dist = dist + torch.eye(n, device=pos.device) * 1e9
    k_eff = min(k, n - 1)
    nn_idx = dist.topk(k_eff, dim=1, largest=False).indices  # (N, k_eff)
    adj = torch.zeros(n, n, dtype=torch.bool, device=pos.device)
    rows = torch.arange(n, device=pos.device).unsqueeze(1).expand_as(nn_idx)
    adj[rows.reshape(-1), nn_idx.reshape(-1)] = True
    return adj


def build_edges(adj: torch.Tensor):
    """
    Convert a (N, N) boolean adjacency mask into edge-index tensors.

    Returns (recv, send) where each is a 1-D LongTensor of equal length and edge e
    connects sender `send[e]` -> receiver `recv[e]`. Convention: messages flow from
    sender j to receiver i for every True entry adj[i, j].
    """
    recv, send = torch.nonzero(adj, as_tuple=True)
    return recv, send
