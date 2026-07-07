"""
sim.py
======
Shared rollout dynamics used by both training (train.py) and visualization
(viz.py). Keeping the integrator in one place guarantees the demo animates the
exact same physics the model was trained on.

State per agent: position(2), velocity(2). Plus a persistent heading(2) used only
to build the perception cone (see graph.py).

`step`/`rollout` transparently support a BATCH of independent swarms: pass (B, N, 2)
tensors instead of (N, 2) and a (B, z_dim) latent code instead of (z_dim,), and every
swarm in the batch is simulated with one vectorized call per timestep instead of a
Python loop over B separate single-swarm calls (see `block_diag_adjacency` in
graph.py and `GammaGNCA.forward` in model.py for how the batch is flattened into one
multi-graph for the model).

Integration (semi-implicit Euler, fixed dt):
    accel = gamma(pos, vel, cone_adjacency(pos, heading), z_shape)
    vel  <- (vel + dt * accel) * (1 - drag)
    pos  <- pos + dt * vel
    heading <- update_heading(heading, vel, max_turn_rad_per_step)  # rate-limited yaw
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from graph import (
    cone_adjacency,
    circular_adjacency,
    block_diag_adjacency,
    update_heading,
    init_heading,
)


@dataclass
class SimConfig:
    n: int = 12
    dt: float = 0.1
    drag: float = 0.02             # mild velocity damping each step (stability)
    perception: str = "cone"       # "cone" (forward FOV) or "circular" (360-degree FOV)
    half_angle_deg: float = 100.0  # cone half-angle (perception == "cone")
    sensing_range: float = 1.0     # sensing radius (both "cone" and "circular")
    init_box: float = 1.0          # half-width of initial position box
    init_vel_std: float = 0.05     # std of small random initial velocities
    speed_eps: float = 1e-3        # heading-update threshold
    max_turn_deg: float = 180.0    # max heading yaw rate, degrees/sec (<=0 = unlimited)

    @property
    def half_angle_rad(self) -> float:
        return math.radians(self.half_angle_deg)

    @property
    def max_turn_rad_per_step(self) -> float | None:
        """Max heading rotation (radians) allowed in one `dt`-sized step, or None."""
        if self.max_turn_deg <= 0:
            return None
        return math.radians(self.max_turn_deg) * self.dt


def build_adjacency(pos, heading, cfg: "SimConfig"):
    """Dispatch to the configured perception model -> directed adjacency mask."""
    if cfg.perception == "circular":
        return circular_adjacency(pos, cfg.sensing_range)
    return cone_adjacency(pos, heading, cfg.sensing_range, cfg.half_angle_rad)


def random_init(cfg: SimConfig, generator: Optional[torch.Generator] = None):
    """Sample a fresh random initial state (pos, vel, heading)."""
    pos = (torch.rand(cfg.n, 2, generator=generator) * 2 - 1) * cfg.init_box
    vel = torch.randn(cfg.n, 2, generator=generator) * cfg.init_vel_std
    heading = init_heading(vel, generator=generator)
    return pos, vel, heading


def step(model, pos, vel, heading, z, cfg: SimConfig):
    """
    Advance the simulation one timestep. Returns (pos, vel, heading).

    `pos`/`vel`/`heading` may be (N, 2) for a single swarm, or (B, N, 2) for a
    BATCH of B independent swarms -- in the batched case, `z` must be (B, z_dim)
    (one latent code per swarm). Batched swarms are flattened into a single
    block-diagonal multi-graph (see `graph.block_diag_adjacency`) so the whole
    batch is handled by one call to the model instead of a Python loop over B.
    """
    adj = build_adjacency(pos, heading, cfg)
    if pos.dim() == 3:
        b, n, _ = pos.shape
        flat_adj = block_diag_adjacency(adj)
        accel = model(pos.reshape(b * n, 2), vel.reshape(b * n, 2), flat_adj, z)
        accel = accel.reshape(b, n, 2)
    else:
        accel = model(pos, vel, adj, z)
    vel = (vel + cfg.dt * accel) * (1.0 - cfg.drag)
    pos = pos + cfg.dt * vel
    heading = update_heading(heading, vel, cfg.speed_eps, cfg.max_turn_rad_per_step)
    return pos, vel, heading


def rollout(
    model,
    pos,
    vel,
    heading,
    z,
    steps: int,
    cfg: SimConfig,
    record: bool = False,
):
    """
    Run `steps` simulation steps.

    If record=False (training), returns the final (pos, vel, heading) and a list of
    the velocities visited (for the damping regularizer). If record=True (viz),
    also returns the full per-step trajectory of positions and headings.
    """
    traj_pos: List[torch.Tensor] = []
    traj_heading: List[torch.Tensor] = []
    vel_history: List[torch.Tensor] = []

    if record:
        traj_pos.append(pos.detach().clone())
        traj_heading.append(heading.detach().clone())

    for _ in range(steps):
        pos, vel, heading = step(model, pos, vel, heading, z, cfg)
        vel_history.append(vel)
        if record:
            traj_pos.append(pos.detach().clone())
            traj_heading.append(heading.detach().clone())

    if record:
        return pos, vel, heading, traj_pos, traj_heading
    return pos, vel, heading, vel_history
