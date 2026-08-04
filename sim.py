"""
sim.py
======
The simulated flight dynamics. Shared by training and visualization, so the demo
always animates exactly the physics the model was trained on.

STATE per drone: position p (2), heading angle theta, forward speed s, yaw rate
omega. Velocity is never an independent variable -- it is always s * [cos theta,
sin theta]. A drone moves where its nose points.

ONE STEP (semi-implicit Euler, fixed dt):

    a_lin, a_ang = gamma(neighbours in body frame, s, omega)   # the ONLY control
    s     <- clip( s + dt * a_lin , 0 , max_speed )            # forward-only
    omega <- clip( omega + dt * a_ang , +-max_yaw_rate )
    theta <- theta + dt * omega
    p     <- p + dt * s * [cos theta, sin theta]

What is deliberately NOT here, per the model rules:

  * no drag -- nothing damps the swarm for free; parking must be learned;
  * no reactive safety filter and no geofence clamp -- both were controllers that
    overrode gamma's output, and collision avoidance is now the loss's job alone
    (losses.separation_loss);
  * no arena walls -- the swarm flies in free space;
  * no self-rotation, no heading smoothing, no persistent-heading fallback --
    heading is a state gamma steers directly.

The remaining clamps are an ACTUATION ENVELOPE, not a controller: they state what
a Crazyflie can physically do (accelerate at ~2 m/s^2, fly at ~1 m/s, yaw at ~180
deg/s), and gamma may choose anything inside it. They also keep 120-step
backpropagation-through-time numerically bounded now that drag is gone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from graph import block_diag_adjacency, circular_adjacency, cone_adjacency


@dataclass
class SimConfig:
    n: int = 4
    dt: float = 0.1                 # seconds per step
    perception: str = "cone"        # "cone" (forward FOV) or "circular" (360 deg)
    half_angle_deg: float = 75.0    # cone half-angle => 150 deg total FOV
    sensing_range: float = 1.2      # meters
    # --- actuation envelope (physical limits, not control logic) ---
    max_speed: float = 1.0              # m/s
    max_accel: float = 2.0              # m/s^2 forward acceleration
    max_yaw_rate_deg: float = 180.0     # deg/s
    max_yaw_accel_deg: float = 360.0    # deg/s^2
    # --- initial conditions ---
    init_box: float = 1.0           # half-width of the random start box, meters
    min_start_dist: float = 0.5     # resample starts tighter than this
    drone_radius: float = 0.1       # safety-disk radius; collision = 2*this

    @property
    def half_angle_rad(self) -> float:
        return math.radians(self.half_angle_deg)

    @property
    def max_yaw_rate(self) -> float:
        return math.radians(self.max_yaw_rate_deg)

    @property
    def max_yaw_accel(self) -> float:
        return math.radians(self.max_yaw_accel_deg)


def build_adjacency(pos, theta, cfg: SimConfig):
    """Dispatch to the configured sensing rule -> directed adjacency mask."""
    if cfg.perception == "circular":
        return circular_adjacency(pos, cfg.sensing_range)
    return cone_adjacency(pos, theta, cfg.sensing_range, cfg.half_angle_rad)


def velocity(s: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """World-frame velocity from the non-holonomic state: v = s * heading."""
    return s.unsqueeze(-1) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def random_init(cfg: SimConfig, batch: int = 1,
                generator: Optional[torch.Generator] = None):
    """
    Sample a fresh start: random positions in a box, random headings, at rest.

    Positions are resampled until every pair is at least `min_start_dist` apart,
    so a run never *begins* in a collision -- otherwise the collision loss would
    be paying for the sampler's mistake rather than the policy's. Drones start at
    s = omega = 0, so every bit of motion in a rollout is something gamma caused.
    """
    n = cfg.n
    pos = torch.empty(batch, n, 2)
    for b in range(batch):
        cand = (torch.rand(n, 2, generator=generator) * 2 - 1) * cfg.init_box
        for _ in range(1000):
            d = torch.linalg.norm(cand.unsqueeze(0) - cand.unsqueeze(1), dim=-1)
            d = d + torch.eye(n) * 1e9
            if cfg.min_start_dist <= 0 or float(d.min()) >= cfg.min_start_dist:
                break
            cand = (torch.rand(n, 2, generator=generator) * 2 - 1) * cfg.init_box
        pos[b] = cand
    theta = torch.rand(batch, n, generator=generator) * 2 * math.pi
    s = torch.zeros(batch, n)
    omega = torch.zeros(batch, n)
    if batch == 1:
        return pos[0], theta[0], s[0], omega[0]
    return pos, theta, s, omega


def step(model, pos, theta, s, omega, cfg: SimConfig):
    """
    Advance one timestep. Accepts a single swarm (N, ...) or a batch (B, N, ...);
    a batch is flattened into one block-diagonal multi-graph so the model runs
    once per step rather than once per swarm.
    """
    adj = build_adjacency(pos, theta, cfg)
    if pos.dim() == 3:
        b, n, _ = pos.shape
        a = model(pos.reshape(b * n, 2), theta.reshape(b * n), s.reshape(b * n),
                  omega.reshape(b * n), block_diag_adjacency(adj))
        a = a.reshape(b, n, 2)
    else:
        a = model(pos, theta, s, omega, adj)
    a_lin, a_ang = a[..., 0], a[..., 1]

    # Actuation envelope. Clamping is differentiable almost everywhere; gradient
    # simply stops flowing through a channel asking for more than the drone has.
    a_lin = a_lin.clamp(-cfg.max_accel, cfg.max_accel)
    a_ang = a_ang.clamp(-cfg.max_yaw_accel, cfg.max_yaw_accel)

    s = (s + cfg.dt * a_lin).clamp(0.0, cfg.max_speed)      # forward-only
    omega = (omega + cfg.dt * a_ang).clamp(-cfg.max_yaw_rate, cfg.max_yaw_rate)
    theta = theta + cfg.dt * omega
    pos = pos + cfg.dt * velocity(s, theta)
    return pos, theta, s, omega


def rollout(model, pos, theta, s, omega, steps: int, cfg: SimConfig,
            record: bool = False):
    """
    Run `steps` timesteps.

    Always returns the full position history: the losses need every step -- the
    formation term averages over the tail, the collision term integrates over the
    whole rollout. With record=True it also returns heading and speed histories
    for visualization and diagnostics.
    """
    pos_history: List[torch.Tensor] = [pos]
    theta_history: List[torch.Tensor] = [theta]
    s_history: List[torch.Tensor] = [s]

    for _ in range(steps):
        pos, theta, s, omega = step(model, pos, theta, s, omega, cfg)
        pos_history.append(pos)
        if record:
            theta_history.append(theta)
            s_history.append(s)

    if record:
        return pos, theta, s, omega, pos_history, theta_history, s_history
    return pos, theta, s, omega, pos_history
