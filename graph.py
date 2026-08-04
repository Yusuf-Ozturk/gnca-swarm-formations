"""
graph.py
========
Perception: which drones does each drone see this step?

Two switchable sensing rules, both pure range/bearing tests -- the only things a
real onboard sensor can measure locally:

  * CONE (forward field of view). Drone i sees j iff j is within `sensing_range`
    AND within `half_angle` of i's own heading. Directed and asymmetric: i seeing
    j does not imply j sees i. A drone whose cone is empty receives no messages
    that step and must act on its own state alone -- expected, not a bug.

  * CIRCULAR (360-degree). Drone i sees every j within `sensing_range`,
    regardless of orientation. Same range test, no bearing test.

Heading is now a genuine STATE VARIABLE, integrated from the yaw acceleration the
model outputs (see sim.py), not a quantity derived from the velocity direction.
That removes an entire class of machinery the velocity-derived version needed --
persistent-heading fallback for parked drones, EMA smoothing of a noisy velocity
direction, a scanning self-rotation, a post-hoc yaw clamp. Heading is simply
theta: always defined, and the drone turns because gamma decided to turn.

Convention throughout: `adj[i, j] == True` means "j is visible to i", i is the
receiver and j the sender. Self-loops excluded.
"""

from __future__ import annotations

import torch


def heading_vector(theta: torch.Tensor) -> torch.Tensor:
    """(...,) heading angles -> (..., 2) unit vectors [cos, sin]."""
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def cone_adjacency(pos: torch.Tensor, theta: torch.Tensor,
                   sensing_range: float, half_angle_rad: float) -> torch.Tensor:
    """
    FORWARD FIELD OF VIEW.

    adj[i, j] iff  ||p_j - p_i|| <= sensing_range
              and  angle( heading_i , p_j - p_i ) <= half_angle.

    Args:
        pos:   (..., N, 2) positions.
        theta: (..., N)    heading angles in radians.
    Returns:
        (..., N, N) boolean adjacency.
    """
    rel = pos.unsqueeze(-3) - pos.unsqueeze(-2)           # rel[..., i, j] = p_j - p_i
    dist = torch.linalg.norm(rel, dim=-1)                 # (..., N, N)
    rel_unit = rel / dist.clamp(min=1e-8).unsqueeze(-1)

    h = heading_vector(theta)                             # (..., N, 2)
    cos_angle = (h.unsqueeze(-2) * rel_unit).sum(dim=-1)  # receiver i broadcast over j
    cos_thresh = torch.cos(torch.as_tensor(half_angle_rad, dtype=pos.dtype))

    n = pos.shape[-2]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    return (dist <= sensing_range) & (cos_angle >= cos_thresh) & (~eye)


def circular_adjacency(pos: torch.Tensor, sensing_range: float) -> torch.Tensor:
    """
    OMNIDIRECTIONAL (360-degree) sensing: the range test alone, heading ignored.
    The baseline the cone is compared against -- the only thing that differs
    between the two runs is what each drone is allowed to see.
    """
    rel = pos.unsqueeze(-3) - pos.unsqueeze(-2)
    dist = torch.linalg.norm(rel, dim=-1)
    n = pos.shape[-2]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device)
    return (dist <= sensing_range) & (~eye)


def block_diag_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    (B, N, N) per-swarm adjacencies -> (B*N, B*N) block-diagonal mask, so a whole
    training batch of independent swarms becomes one multi-graph and the model
    runs once per step instead of B times. Swarms never share edges.
    """
    b, n, _ = adj.shape
    out = torch.zeros(b * n, b * n, dtype=torch.bool, device=adj.device)
    idx = torch.arange(b, device=adj.device) * n
    for k in range(b):                    # B is small (128); an explicit loop is
        o = int(idx[k])                   # clearer than index gymnastics here
        out[o:o + n, o:o + n] = adj[k]
    return out


def build_edges(adj: torch.Tensor):
    """(N, N) mask -> (recv, send) index tensors; a message flows send -> recv."""
    recv, send = torch.nonzero(adj, as_tuple=True)
    return recv, send
