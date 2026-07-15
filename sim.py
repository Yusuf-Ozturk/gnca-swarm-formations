"""
sim.py
======
Shared rollout dynamics used by both training (train.py) and visualization
(viz.py). Keeping the integrator in one place guarantees the demo animates the
exact same physics the model was trained on.

State per agent: position(2), velocity(2). Plus a persistent heading(2) used only
to build the perception cone (see graph.py), and a smoothed_vel(2) EMA state that
heading is derived from (see graph.py point 3 -- raw instantaneous velocity is too
noisy a signal to build heading from directly).

`step`/`rollout` transparently support a BATCH of independent swarms: pass (B, N, 2)
tensors instead of (N, 2) and a (B, z_dim) latent code instead of (z_dim,), and every
swarm in the batch is simulated with one vectorized call per timestep instead of a
Python loop over B separate single-swarm calls (see `block_diag_adjacency` in
graph.py and `GammaGNCA.forward` in model.py for how the batch is flattened into one
multi-graph for the model).

Integration (semi-implicit Euler, fixed dt):
    accel = gamma(pos, vel, cone_adjacency(pos, heading), z_shape)
    accel += wall_accel(pos)                                # arena_mode == "walls" only
    vel  <- (vel + dt * accel) * (1 - drag)
    pos  <- pos + dt * vel
    smoothed_vel <- update_smoothed_vel(smoothed_vel, vel)  # EMA low-pass filter
    heading <- update_heading(heading, smoothed_vel)        # persistent-heading fallback
    heading <- apply_self_rotation(heading, self_rotation)  # scanning term (graph.py point 4)

UNITS (issue #4): one simulation unit is one METER and dt is in seconds, so the
default dt=0.1 is a 10 Hz control rate. The physical deployment target is a
3m x 3m lighthouse flight area centered on the origin (arena_half = 1.5), with
Crazyflie drones treated as 10 cm-radius safety disks (drone_radius = 0.1).
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
    update_smoothed_vel,
    apply_self_rotation,
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
    heading_smoothing: float = 0.2  # EMA beta for velocity->heading low-pass filter (1.0 = off)
    self_rotation_deg: float = 15.0  # heading scanning rate, deg/sec (0 = off)
    # --- drone flight area (issue #4); 1 sim unit = 1 meter ---
    arena_mode: str = "none"       # "none" | "fixed" (method 1) | "walls" (method 2)
    arena_half: float = 1.5        # half-width of the square flight area (3m x 3m)
    wall_margin: float = 0.3       # wall force activates within this distance [walls]
    wall_strength: float = 0.5     # wall force scale, accel = k*(1/d - 1/margin) [walls]
    drone_radius: float = 0.1      # per-drone safety-disk radius (10 cm)
    min_start_dist: float = 0.0    # min pairwise distance enforced on inits (0 = off)
    max_speed: float = 0.0         # hard per-drone speed cap, m/s (0 = uncapped)

    @property
    def half_angle_rad(self) -> float:
        return math.radians(self.half_angle_deg)

    @property
    def self_rotation_rad_per_step(self) -> float:
        """Fixed heading rotation (radians) applied in one `dt`-sized step."""
        return math.radians(self.self_rotation_deg) * self.dt


def build_adjacency(pos, heading, cfg: "SimConfig"):
    """Dispatch to the configured perception model -> directed adjacency mask."""
    if cfg.perception == "circular":
        return circular_adjacency(pos, cfg.sensing_range)
    return cone_adjacency(pos, heading, cfg.sensing_range, cfg.half_angle_rad)


def wall_accel(pos: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """
    REPULSIVE WALLS (issue #4, method 2). The four walls of the square flight
    area x, y = +/- arena_half each push a drone toward the arena center with an
    acceleration inversely proportional to its distance from that wall, but only
    once the drone is closer than `wall_margin`:

        a_inward = wall_strength * (1/d - 1/wall_margin)      for d < wall_margin

    Subtracting 1/wall_margin makes the force vanish continuously exactly at the
    activation distance (no kick when crossing the threshold), and it diverges as
    d -> 0 so the boundary is effectively impenetrable -- a drone that reaches a
    wall bounces back in. `d` is clamped to a small positive floor so a state
    that starts (or numerically ends up) outside the arena is pushed strongly
    back inside instead of producing an infinite/NaN force. The whole expression
    is differentiable, so during BPTT the model trains *through* the wall force
    and learns to re-form the (still position/rotation-invariant) shape after
    bouncing.

    Args:
        pos: (..., N, 2) positions.
    Returns:
        (..., N, 2) wall acceleration (zeros for every drone outside the margin).
    """
    d_floor = 0.02  # caps the max wall accel at ~wall_strength * 50
    # Distance to the nearer wall along each axis (negative = outside the arena).
    dist = (cfg.arena_half - pos.abs()).clamp(min=d_floor)   # (..., N, 2)
    magnitude = (1.0 / dist - 1.0 / cfg.wall_margin).clamp(min=0.0)
    return -torch.sign(pos) * cfg.wall_strength * magnitude  # push toward center


def random_init(cfg: SimConfig, generator: Optional[torch.Generator] = None):
    """
    Sample a fresh random initial state (pos, vel, heading, smoothed_vel).

    If cfg.min_start_dist > 0, initial positions are rejection-sampled so that
    every pair of drones starts at least that far apart (issue #4: physical
    drones can never be placed, or spawned in training, inside each other's
    safety disks). Only the offending agents are re-drawn, so this converges in
    a handful of rounds for any reasonable density.
    """
    pos = (torch.rand(cfg.n, 2, generator=generator) * 2 - 1) * cfg.init_box
    if cfg.min_start_dist > 0:
        for _ in range(200):
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.linalg.norm(diff, dim=-1)
            dist.fill_diagonal_(float("inf"))
            too_close = (dist.min(dim=1).values < cfg.min_start_dist)
            if not too_close.any():
                break
            resample = (torch.rand(int(too_close.sum()), 2, generator=generator)
                        * 2 - 1) * cfg.init_box
            pos[too_close] = resample
    vel = torch.randn(cfg.n, 2, generator=generator) * cfg.init_vel_std
    heading = init_heading(vel, generator=generator)
    smoothed_vel = vel.clone()  # no EMA history yet at t=0
    return pos, vel, heading, smoothed_vel


def step(model, pos, vel, heading, smoothed_vel, z, cfg: SimConfig):
    """
    Advance the simulation one timestep. Returns (pos, vel, heading, smoothed_vel).

    `pos`/`vel`/`heading`/`smoothed_vel` may be (N, 2) for a single swarm, or
    (B, N, 2) for a BATCH of B independent swarms -- in the batched case, `z` must
    be (B, z_dim) (one latent code per swarm). Batched swarms are flattened into a
    single block-diagonal multi-graph (see `graph.block_diag_adjacency`) so the
    whole batch is handled by one call to the model instead of a Python loop over B.
    """
    adj = build_adjacency(pos, heading, cfg)
    if pos.dim() == 3:
        b, n, _ = pos.shape
        flat_adj = block_diag_adjacency(adj)
        accel = model(pos.reshape(b * n, 2), vel.reshape(b * n, 2), flat_adj, z)
        accel = accel.reshape(b, n, 2)
    else:
        accel = model(pos, vel, adj, z)
    if cfg.arena_mode == "walls":
        accel = accel + wall_accel(pos, cfg)
    vel = (vel + cfg.dt * accel) * (1.0 - cfg.drag)
    if cfg.max_speed > 0:
        # SPEED CAP (issue #4). Physical: a Crazyflie can't do much more than
        # ~1 m/s indoors, so an uncapped point-mass would report unflyable
        # trajectories. Numerical: it also bounds the rollout dynamics -- an
        # uncapped agent can cross the whole wall-force band (wall_margin) in
        # one dt and slingshot out of the arena, which is exactly how the
        # walls-mode training was observed to diverge. Rescaling the vector
        # (rather than clamping components) preserves direction and stays
        # differentiable almost everywhere.
        speed = torch.linalg.norm(vel, dim=-1, keepdim=True)
        vel = vel * torch.clamp(cfg.max_speed / speed.clamp(min=1e-8), max=1.0)
    pos = pos + cfg.dt * vel
    smoothed_vel = update_smoothed_vel(smoothed_vel, vel, cfg.heading_smoothing)
    heading = update_heading(heading, smoothed_vel, cfg.speed_eps)
    heading = apply_self_rotation(heading, cfg.self_rotation_rad_per_step)
    return pos, vel, heading, smoothed_vel


def rollout(
    model,
    pos,
    vel,
    heading,
    smoothed_vel,
    z,
    steps: int,
    cfg: SimConfig,
    record: bool = False,
):
    """
    Run `steps` simulation steps.

    If record=False (training), returns the final (pos, vel, heading, smoothed_vel)
    plus lists of the velocities and positions visited (for the damping and
    formation-hold regularizers; both stay attached to the autograd graph). If
    record=True (viz), instead returns the full per-step detached trajectory of
    positions and headings.
    """
    traj_pos: List[torch.Tensor] = []
    traj_heading: List[torch.Tensor] = []
    vel_history: List[torch.Tensor] = []
    pos_history: List[torch.Tensor] = []

    if record:
        traj_pos.append(pos.detach().clone())
        traj_heading.append(heading.detach().clone())

    for _ in range(steps):
        pos, vel, heading, smoothed_vel = step(model, pos, vel, heading, smoothed_vel, z, cfg)
        vel_history.append(vel)
        pos_history.append(pos)
        if record:
            traj_pos.append(pos.detach().clone())
            traj_heading.append(heading.detach().clone())

    if record:
        return pos, vel, heading, smoothed_vel, traj_pos, traj_heading
    return pos, vel, heading, smoothed_vel, vel_history, pos_history
