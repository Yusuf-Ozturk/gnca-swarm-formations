"""
model.py
========
gamma -- the one and only controller. Every drone runs this same function, with
the same weights, every timestep. There is nothing else in the loop: no safety
filter, no geofence, no wall force, no hand-tuned fallback.

    m_{i<-j} = phi_msg( [ R(-theta_i)(p_j - p_i) , R(-theta_i)(v_j - v_i) , d_ij ] )
    a_i      = mean_j  m_{i<-j}                      (permutation-invariant pool)
    a_lin_i , a_ang_i = phi_upd( [ a_i , s_i , omega_i ] )

Four properties are structural, not incidental:

  1. DRONES ARE PERFECTLY IDENTICAL. No per-agent identity embedding, no index
     input, no per-agent parameter of any kind. gamma is one shared function of a
     drone's observations, so relabelling the swarm relabels the accelerations
     and nothing else. This is why the training loss must be permutation-
     invariant (losses.py): a labelled slot loss is unlearnable by identical
     agents -- measured, it plateaus at ~11x-361x the error of the labelled model.

  2. NO ABSOLUTE INFORMATION. Edges carry only differences p_j - p_i, v_j - v_i.
     No drone is told its own position, and there is no global frame anywhere in
     the rule. Translation invariance is therefore exact by construction.

  3. BODY FRAME. Those differences are rotated by -theta_i into the receiver's
     own heading frame, and the outputs are a forward acceleration and a yaw
     acceleration -- both defined relative to that same frame. So the whole rule
     is exactly rotation-equivariant too: rotate the world, and every drone's
     trajectory rotates with it. The body frame is also what makes a yaw command
     meaningful -- a drone can only decide to "turn toward that neighbour" if it
     knows where that neighbour is relative to its own nose.

  4. PROPRIOCEPTION ONLY. The sole non-neighbour inputs are the drone's own
     forward speed s and yaw rate omega -- exactly what an onboard IMU reads.
     They are not absolute quantities, and they are what lets an isolated drone
     (empty cone, no messages) still damp itself instead of coasting blind.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from graph import build_edges


def rotate_into_body_frame(vec: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Rotate world-frame vectors by -theta, i.e. express them in the body frame of
    a drone whose nose points along theta. `vec` is (..., 2) and `theta` (...,).

    In the body frame the +x axis is "straight ahead" and +y is "to my left", so
    a neighbour at body-frame (0.5, -0.2) is half a meter ahead and slightly to
    the right -- which is the coordinate system a yaw command has to live in.
    """
    c, s = torch.cos(theta), torch.sin(theta)
    x, y = vec[..., 0], vec[..., 1]
    return torch.stack([c * x + s * y, -s * x + c * y], dim=-1)


class GammaSwarm(nn.Module):
    """
    The shared per-drone update rule.

    One network is trained per target formation (there is no shape-conditioning
    input), so a trained gamma *is* the formation: it encodes "fly into a wedge"
    in its weights and nothing else selects behaviour at runtime.

    Args:
        hidden:   width of the update MLP.
        msg_dim:  width of the message embedding.
        a_lin_scale / a_ang_scale: output scaling, in m/s^2 and rad/s^2. The
            integrator clamps to the physical envelope; these only set the
            natural magnitude of a freshly initialized network's outputs.
    """

    def __init__(
        self,
        hidden: int = 128,
        msg_dim: int = 128,
        a_lin_scale: float = 1.0,
        a_ang_scale: float = 1.0,
    ):
        super().__init__()
        self.msg_dim = msg_dim
        self.a_lin_scale = a_lin_scale
        self.a_ang_scale = a_ang_scale

        # Edge feature = [body-frame rel_pos(2), body-frame rel_vel(2), dist(1)].
        self.phi_msg = nn.Sequential(
            nn.Linear(5, msg_dim), nn.SiLU(),
            nn.Linear(msg_dim, msg_dim), nn.SiLU(),
        )
        # Own feature = [pooled message(msg_dim), own speed(1), own yaw rate(1)].
        self.phi_upd = nn.Sequential(
            nn.Linear(msg_dim + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.out = nn.Linear(hidden, 2)          # (a_lin, a_ang)

        # Start from a standstill policy: zero output means early rollouts stay
        # bounded and BPTT through up to 120 steps does not explode at epoch 0.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pos: torch.Tensor, theta: torch.Tensor, s: torch.Tensor,
                omega: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Compute (a_lin, a_ang) for every drone.

        Accepts a single swarm, (N, ...), or a whole batch flattened into one
        block-diagonal multi-graph, (B*N, ...) with `adj` block-diagonal.

        Args:
            pos:   (M, 2) positions.
            theta: (M,)   heading angles.
            s:     (M,)   forward speeds.
            omega: (M,)   yaw rates.
            adj:   (M, M) boolean adjacency, adj[i, j] = "i sees j".
        Returns:
            (M, 2) tensor of [forward acceleration, yaw acceleration].
        """
        m = pos.shape[0]
        # World-frame velocity is fully determined by (s, theta): motion is along
        # the heading, always. This is the non-holonomic constraint made explicit.
        vel = s.unsqueeze(-1) * torch.stack([torch.cos(theta), torch.sin(theta)], -1)

        agg = pos.new_zeros((m, self.msg_dim))
        recv, send = build_edges(adj)
        if recv.numel() > 0:
            rel_pos = pos[send] - pos[recv]                      # (E, 2) world
            rel_vel = vel[send] - vel[recv]                      # (E, 2) world
            dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
            # BODY FRAME: everything the receiver sees is expressed in its own
            # heading frame, so the rule carries no global orientation.
            th_r = theta[recv]
            edge_feat = torch.cat([
                rotate_into_body_frame(rel_pos, th_r),
                rotate_into_body_frame(rel_vel, th_r),
                dist,
            ], dim=-1)
            msg = self.phi_msg(edge_feat)                        # (E, msg_dim)

            # Mean pooling: permutation-invariant, and invariant to how many
            # neighbours happen to be visible. Drones with an empty cone keep an
            # all-zero pooled message and fall back on proprioception alone.
            agg = agg.index_add(0, recv, msg)
            deg = torch.zeros(m, device=pos.device, dtype=pos.dtype).index_add(
                0, recv, torch.ones_like(recv, dtype=pos.dtype))
            agg = agg / deg.clamp(min=1.0).unsqueeze(-1)

        own = torch.cat([agg, s.unsqueeze(-1), omega.unsqueeze(-1)], dim=-1)
        h = self.phi_upd(own)
        a = self.out(h)
        return torch.stack([a[:, 0] * self.a_lin_scale,
                            a[:, 1] * self.a_ang_scale], dim=-1)
