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

Both `cone_adjacency` and `circular_adjacency` (below) are pure range/bearing
sensing rules: an edge exists only if agent j is within a physical sensing radius
of agent i (plus, for the cone, within its forward angular sector). This is what a
real drone's onboard sensor (camera FOV, lidar, UWB ranging) can actually measure
locally. There is deliberately no k-nearest-neighbours mode here: kNN requires
agent i to globally rank *every other agent* by distance and keep exactly k of
them regardless of how far away they are, which is not something a drone's sensor
can do -- it has no notion of "keep exactly my k closest neighbours no matter the
distance", only "who is within my sensing range".

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

    Accepts an optional leading batch dimension so a whole batch of independent
    swarms can be processed in one vectorized call instead of one Python-level call
    per swarm (see `block_diag_adjacency` below for how the result is then combined
    into a single graph for the model).

    Args:
        pos:            (..., N, 2) positions.
        heading:        (..., N, 2) persistent unit headings (receiver's facing dir).
        sensing_range:  scalar radius.
        half_angle_rad: scalar cone half-angle in radians.
    Returns:
        adj: (..., N, N) boolean. Self-loops are excluded.
    """
    n = pos.shape[-2]
    # rel[..., i, j] = pos_j - pos_i  (vector from receiver i to sender j)
    rel = pos.unsqueeze(-3) - pos.unsqueeze(-2)        # (..., N, N, 2)
    dist = torch.linalg.norm(rel, dim=-1)              # (..., N, N)

    # Unit direction from i to j (guard against the zero self-vector).
    safe_dist = torch.clamp(dist, min=1e-8).unsqueeze(-1)
    rel_unit = rel / safe_dist                         # (..., N, N, 2)

    # cos of angle between heading_i and direction(i->j).
    # heading_i broadcasts across j (second-to-last dim).
    cos_angle = (heading.unsqueeze(-2) * rel_unit).sum(dim=-1)  # (..., N, N)
    cos_thresh = torch.cos(torch.tensor(half_angle_rad, dtype=pos.dtype))

    within_range = dist <= sensing_range
    within_cone = cos_angle >= cos_thresh

    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    adj = within_range & within_cone & (~eye)
    return adj


def circular_adjacency(pos: torch.Tensor, sensing_range: float) -> torch.Tensor:
    """
    CIRCULAR / 360-DEGREE FIELD-OF-VIEW edge construction -- the omnidirectional
    counterpart to the forward cone, and the BASELINE perception to compare against
    it.

    Agent i connects to every agent within `sensing_range` of it, in every
    direction, regardless of heading. This is exactly `cone_adjacency` with the
    angular gate removed (equivalently, a cone with half_angle = 180 degrees): a
    pure omnidirectional range sensor, physically meaningful for a drone equipped
    with e.g. lidar or UWB ranging rather than a forward-facing camera. Unlike kNN,
    an agent's neighbour count is whatever the geometry gives it (0 up to N-1), not
    a fixed k -- distance is the only criterion, never a global rank.

    Same convention as `cone_adjacency`: `adj[i, j]` True means j is one of i's
    neighbours (i is the receiver). Edges are directed but, since the range test is
    symmetric (dist(i, j) == dist(j, i)), the resulting adjacency is symmetric too.
    Also accepts an optional leading batch dimension, same as `cone_adjacency`.

    Args:
        pos:           (..., N, 2) positions.
        sensing_range: scalar radius.
    Returns:
        adj: (..., N, N) boolean. Self-loops excluded.
    """
    n = pos.shape[-2]
    rel = pos.unsqueeze(-3) - pos.unsqueeze(-2)        # (..., N, N, 2), rel[...,i,j]=pos_j-pos_i
    dist = torch.linalg.norm(rel, dim=-1)              # (..., N, N)
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    adj = (dist <= sensing_range) & (~eye)
    return adj


def block_diag_adjacency(adj_batch: torch.Tensor) -> torch.Tensor:
    """
    Combine a (B, N, N) batch of independent-swarm adjacency masks into one
    block-diagonal (B*N, B*N) adjacency describing B disjoint graphs at once --
    swarms never connect to each other, only within their own block.

    This is what lets `model.py`'s edge-list message passing run a whole training
    batch as a single call (one flattened multi-graph) instead of looping over the
    batch in Python and calling the model B separate times, which is the dominant
    overhead at this problem's small per-agent scale.

    Args:
        adj_batch: (B, N, N) boolean, one adjacency mask per swarm.
    Returns:
        (B*N, B*N) boolean block-diagonal adjacency.
    """
    return torch.block_diag(*adj_batch.unbind(0))


def build_edges(adj: torch.Tensor):
    """
    Convert a (N, N) boolean adjacency mask into edge-index tensors.

    Returns (recv, send) where each is a 1-D LongTensor of equal length and edge e
    connects sender `send[e]` -> receiver `recv[e]`. Convention: messages flow from
    sender j to receiver i for every True entry adj[i, j].
    """
    recv, send = torch.nonzero(adj, as_tuple=True)
    return recv, send
