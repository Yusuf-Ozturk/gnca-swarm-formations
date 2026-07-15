"""
model.py
========
The learned update function gamma: a single shared message-passing GNN layer plus
a per-node MLP that outputs each agent's next acceleration. One network serves all
shapes; the target shape is selected by a learned latent code injected via FiLM.

Three non-obvious pieces, flagged inline:

  1. RELATIVE-COORDINATE EQUIVARIANCE. The message function only ever sees RELATIVE
     spatial quantities per edge -- (pos_j - pos_i) and (vel_j - vel_i) -- never
     absolute coordinates. Differencing removes the global origin, so the rule is
     translation-invariant by construction (and, being free of any global frame,
     has no preferred orientation -- the rotation-invariant training loss does the
     rest).

  2. FiLM CONDITIONING. A per-shape latent code z_shape is mapped by a linear layer
     to a (scale, shift) pair that modulates a hidden layer of the post-aggregation
     MLP: h <- scale * h + shift. Swapping z_shape swaps the target formation while
     reusing the exact same gamma weights. Adding a new shape later = learning one
     new z_shape vector, no retraining of gamma.

  3. AGENT IDENTITY. The PRIMARY loss assigns agent k to a fixed labeled slot k, but
     a shared anonymous rule is permutation-symmetric and cannot decide which agent
     becomes which slot -- it averages over the ambiguity and never forms a clean
     shape. So each agent carries a small constant learned ID embedding: it is fed
     into the agent's own features AND attached to every message it sends, so a
     receiver can recognise *who* it sees and triangulate its own target slot. IDs
     are constant scalars, not coordinates, so translation/rotation behaviour is
     unaffected.

  4. ABSOLUTE POSITION (issue #4, arena_mode == "fixed" only). Piece 1 makes the
     rule translation-invariant by construction -- which also makes it physically
     INCAPABLE of steering to a target at one fixed place in the arena: it cannot
     tell where it is. When the goal shape is fixed and centered in the 3m x 3m
     flight area (method 1), each agent's own absolute position (in arena-centered
     coordinates) is appended to its own features, deliberately trading away
     translation invariance for absolute addressability. This is physically
     honest: the lighthouse positioning system gives every Crazyflie exactly this
     measurement. In the invariant modes ("none"/"walls") the flag stays off and
     nothing changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from graph import build_edges


class FiLM(nn.Module):
    """
    FiLM CONDITIONING layer.

    Maps a conditioning vector z (here, the per-shape latent code) to per-feature
    (scale, shift) and applies h <- scale * h + shift elementwise. We initialize so
    that at the start scale ~= 1 and shift ~= 0 (an identity modulation), which
    keeps early training stable.
    """

    def __init__(self, cond_dim: int, feature_dim: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * feature_dim)
        # Identity-ish init: scale->1, shift->0.
        nn.init.zeros_(self.to_scale_shift.weight)
        with torch.no_grad():
            self.to_scale_shift.bias[:feature_dim] = 1.0   # scale bias
            self.to_scale_shift.bias[feature_dim:] = 0.0   # shift bias
        self.feature_dim = feature_dim

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_scale_shift(z).chunk(2, dim=-1)
        return scale * h + shift


class GammaGNCA(nn.Module):
    """
    The shared GNCA update rule.

    A single message-passing step:
        message_{i<-j} = phi_msg( [pos_j-pos_i, vel_j-vel_i, dist, id_j] )
        agg_i          = pool_j  message_{i<-j}            (perm-invariant)
        accel_i        = phi_upd( [agg_i, vel_i, speed_i, id_i] , FiLM(z_shape) )

    pool is selected by `aggregation`:
      * "mean"    -- plain average of neighbour messages (the original rule).
      * "softmax" -- DISTANCE-WEIGHTED attention, w_ij = softmax_j(-d_ij / temp).
        Plain mean pooling structurally dilutes collision threats: an agent in a
        compact swarm sees most teammates at once, and the alarm message from a
        neighbour 0.25m away is averaged 1:1 with half a dozen benign messages
        from over a meter away -- measured on trained checkpoints, that model
        formed perfect shapes yet kept grazing through the collision distance
        during launch/merge transients (issue #4). With temp = 0.3m, a sender at
        0.3m carries ~e^3 ~ 20x the weight of one at 1.2m, so the imminent
        threat dominates the pooled message exactly when it should, while far
        neighbours still tie-break gently. Both pools are permutation-invariant
        and use only relative quantities, so equivariance is unaffected.

    own features use only the agent's OWN velocity (a relative-frame quantity), its
    speed, and its constant ID -- never absolute position -- so an empty-cone agent
    still produces a sensible (identity-aware) acceleration. The one deliberate
    exception is absolute_pos=True (fixed-arena targets, header note 4), where the
    agent's own absolute position is appended so it can steer to a fixed slot.
    """

    def __init__(
        self,
        n_agents: int = 12,
        hidden: int = 64,
        msg_dim: int = 64,
        z_dim: int = 8,
        id_dim: int = 8,
        n_shapes: int = 4,
        accel_scale: float = 1.0,
        absolute_pos: bool = False,
        aggregation: str = "mean",
        softmax_temp: float = 0.3,
    ):
        super().__init__()
        self.accel_scale = accel_scale
        self.n_agents = n_agents
        # ABSOLUTE POSITION input (header note 4): only for fixed-arena targets.
        self.absolute_pos = absolute_pos
        # Message pooling: "mean" or distance-weighted "softmax" (class docstring).
        self.aggregation = aggregation
        self.softmax_temp = softmax_temp

        # --- Per-shape latent codes (the only thing that differs across shapes). ---
        # An opaque learned descriptor per preset, NOT the target coordinates.
        self.z_shape = nn.Embedding(n_shapes, z_dim)
        nn.init.normal_(self.z_shape.weight, std=0.1)

        # --- Constant per-agent identity embeddings (see header note 3). ---
        self.agent_id = nn.Embedding(n_agents, id_dim)
        nn.init.normal_(self.agent_id.weight, std=1.0)
        self.register_buffer("_ids", torch.arange(n_agents))

        # --- Message function phi_msg over RELATIVE edge features (+ sender id). ---
        # Edge feature = [rel_pos(2), rel_vel(2), dist(1), id_sender(id_dim)].
        self.phi_msg = nn.Sequential(
            nn.Linear(2 + 2 + 1 + id_dim, msg_dim),
            nn.SiLU(),
            nn.Linear(msg_dim, msg_dim),
            nn.SiLU(),
        )
        self.msg_dim = msg_dim

        # --- Update function: aggregated message + own velocity + own id -> accel. ---
        # Own input = [agg(msg_dim), own_vel(2), own_speed(1), own_id(id_dim)]
        # (+ own absolute pos(2) when absolute_pos, see header note 4).
        self.in_upd = nn.Linear(msg_dim + 2 + 1 + id_dim + (2 if absolute_pos else 0),
                                hidden)
        self.film = FiLM(z_dim, hidden)
        self.mid_upd = nn.Linear(hidden, hidden)
        self.out_upd = nn.Linear(hidden, 2)  # 2D acceleration
        self.act = nn.SiLU()

        # Start with tiny accelerations so early rollouts are stable.
        nn.init.zeros_(self.out_upd.weight)
        nn.init.zeros_(self.out_upd.bias)

    def get_z(self, shape_id: int | torch.Tensor) -> torch.Tensor:
        """Look up the latent code for a shape id (int or LongTensor scalar)."""
        if not torch.is_tensor(shape_id):
            shape_id = torch.tensor(shape_id, dtype=torch.long)
        return self.z_shape(shape_id)

    def forward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        adj: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute next acceleration for every agent.

        Supports both a single swarm and a BATCH of independent swarms flattened
        into one multi-graph (see `graph.block_diag_adjacency`): the batch size is
        inferred from the node count (`pos.shape[0] // self.n_agents`), since every
        swarm always has exactly `self.n_agents` agents and swarms in a batch never
        share edges. This lets `sim.py` run a whole training batch as one call
        instead of looping over the batch in Python.

        Args:
            pos: (n_agents * B, 2) positions, B swarms concatenated (B=1 for a
                 single swarm).
            vel: (n_agents * B, 2) velocities, same layout as pos.
            adj: (n_agents * B, n_agents * B) boolean adjacency; block-diagonal
                 across swarms when B > 1 (adj[i,j] True => j in i's cone).
            z:   (z_dim,) latent code shared by every node, OR (B, z_dim) one
                 latent code per swarm (matching the B inferred above).
        Returns:
            accel: (n_agents * B, 2) accelerations.
        """
        n = pos.shape[0]
        num_blocks = n // self.n_agents
        ids = self.agent_id(self._ids)  # (n_agents, id_dim) constant per agent
        if num_blocks > 1:
            ids = ids.tile(num_blocks, 1)  # (N, id_dim), repeated per swarm block
        recv, send = build_edges(adj)        # both (E,)

        agg = pos.new_zeros((n, self.msg_dim))
        if recv.numel() > 0:
            # RELATIVE-COORDINATE EQUIVARIANCE: spatial edge features are differences.
            rel_pos = pos[send] - pos[recv]            # (E, 2)
            rel_vel = vel[send] - vel[recv]            # (E, 2)
            dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)  # (E, 1)
            id_send = ids[send]                        # (E, id_dim) who is seen
            edge_feat = torch.cat([rel_pos, rel_vel, dist, id_send], dim=-1)
            msg = self.phi_msg(edge_feat)              # (E, msg_dim)

            if self.aggregation == "softmax":
                # DISTANCE-WEIGHTED attention pooling (class docstring): per
                # receiver, w_ij = softmax_j(-d_ij / temp), so the nearest
                # sender dominates. Numerically stabilized by subtracting each
                # receiver's max logit (detached: the shift cancels in the
                # softmax, so it must carry no gradient of its own).
                logit = (-dist / self.softmax_temp).squeeze(-1)      # (E,)
                mx = torch.full((n,), float("-inf"), dtype=logit.dtype,
                                device=pos.device)
                mx = mx.scatter_reduce(0, recv, logit.detach(), reduce="amax")
                w = torch.exp(logit - mx[recv])                      # (E,)
                denom = torch.zeros(n, device=pos.device).index_add(0, recv, w)
                w = (w / denom[recv].clamp(min=1e-8)).unsqueeze(-1)  # (E, 1)
                agg = agg.index_add(0, recv, w * msg)  # empty rows stay zero
            else:
                # Permutation-invariant mean of messages per receiver.
                agg = agg.index_add(0, recv, msg)
                deg = torch.zeros(n, device=pos.device).index_add(
                    0, recv, torch.ones_like(recv, dtype=pos.dtype)
                )
                deg = deg.clamp(min=1.0).unsqueeze(-1)
                agg = agg / deg  # empty-cone rows stay zero (deg clamp keeps them 0)

        own_speed = torch.linalg.norm(vel, dim=-1, keepdim=True)  # (N, 1)
        own_parts = [agg, vel, own_speed, ids]
        if self.absolute_pos:
            own_parts.append(pos)  # arena-centered coordinates (header note 4)
        own_feat = torch.cat(own_parts, dim=-1)  # (N, msg_dim+3+id[+2])

        h = self.act(self.in_upd(own_feat))
        # FiLM CONDITIONING applied to this hidden layer.
        if z.dim() == 1:
            z_b = z.unsqueeze(0).expand(n, -1)  # one latent code, broadcast to all nodes
        else:
            z_b = z.repeat_interleave(self.n_agents, dim=0)  # (B, z_dim) -> per-node
        h = self.film(h, z_b)
        h = self.act(self.mid_upd(h))
        accel = self.out_upd(h) * self.accel_scale
        return accel
