"""
train.py
========
Train the shared GNCA rule gamma to drive a swarm into every preset formation.

Key training ingredients:
  * BPTT over a rollout whose length t is sampled randomly per batch (t in
    [t_min, t_max]) so the target must be an ATTRACTOR reached after any number of
    steps, not memorized at exactly t. The horizon spans the real deployment hold
    window (t_max steps ~= 12s at dt=0.1), so holding formation over that window
    is optimized directly instead of hoped for by extrapolation (issue #2).
  * A FORMATION-HOLD TAIL: the formation loss is averaged over the last hold_tail
    steps of the rollout, not just the final state, so drifting/wobbling around
    the target inside the tail is itself penalized (see losses.py).
  * A REPLAY CACHE (a la Mordvintsev / Grattarola): we stash the end-states of
    rollouts and re-seed some training samples from them, so the model learns to
    keep refining already-near-target states (and doesn't settle into a periodic
    jitter). Fresh random inits are periodically re-injected to avoid forgetting
    how to form a shape from scratch.
  * Every preset is trained every epoch so the single shared gamma learns all the
    z_shape codes.
  * Cosine learning-rate decay (lr -> lr_min) so late training refines the learned
    attractor with small steps instead of jittering it with full-size updates.
  * DRONE-DEPLOYMENT terms (issue #4): a COLLISION-AVOIDANCE hinge on every drone
    pair at every rollout step, an ARENA containment story selected by arena_mode
    ("fixed": fixed centered targets + coordinate loss + bounds hinge; "walls":
    repulsive boundary force in the physics, invariant loss unchanged), and a
    post-training collision check on every evaluation rollout. See training_loss
    below for the exact objective.

Run:  python train.py            (uses config.yaml defaults; ~a few min on CPU)
"""

from __future__ import annotations

import random
import time
from collections import deque

import torch

from config import load_config, sim_config_from
from losses import (bounds_loss, chamfer_loss, damping_loss, distance_matrix_loss,
                    fixed_formation_hold_loss, fixed_formation_loss,
                    formation_hold_loss, separation_loss)
from model import GammaGNCA
from shapes import (PRESET_NAMES, get_shape, all_target_distance_matrices,
                    pairwise_distance_matrix)
from sim import random_init, rollout


class ReplayCache:
    """
    REPLAY CACHE. A bounded pool of rollout end-states, keyed by shape id. Each
    entry is a (pos, vel, heading, smoothed_vel) tuple (detached). We sample from
    it to continue refining near-converged states, and periodically overwrite
    entries with fresh random inits so the model never forgets the from-scratch
    task.
    """

    def __init__(self, size: int, n_shapes: int):
        self.pools = [deque(maxlen=size) for _ in range(n_shapes)]

    def add(self, shape_id: int, state):
        self.pools[shape_id].append(tuple(s.detach().clone() for s in state))

    def sample(self, shape_id: int):
        pool = self.pools[shape_id]
        if not pool:
            return None
        return random.choice(pool)

    def __len__(self):
        return sum(len(p) for p in self.pools)


def make_sample(cache, shape_id, cfg, sim_cfg, gen, force_fresh: bool):
    """Either re-seed from the replay cache or draw a fresh random init."""
    if (not force_fresh) and random.random() < cfg.replay_prob:
        s = cache.sample(shape_id)
        if s is not None:
            pos, vel, heading, smoothed_vel = (t.clone() for t in s)
            return pos, vel, heading, smoothed_vel
    return random_init(sim_cfg, generator=gen)


def evaluate(model, cfg, sim_cfg, target_dms, target_pts):
    """
    Final per-shape formation error from fresh random inits (no grad), measured
    with the metric each mode is actually trained/judged on: coordinate MSE vs
    the fixed targets in "fixed" arena mode, distance-matrix MSE otherwise.

    Also runs the issue #4 COLLISION CHECK on every evaluation rollout: reports,
    per shape, the smallest center-to-center distance seen at any step of any
    rollout and how many rollouts ever put two drones closer than
    2*drone_radius (touching safety disks).
    """
    fixed = getattr(cfg, "arena_mode", "none") == "fixed"
    collision_dist = 2.0 * sim_cfg.drone_radius
    model.eval()
    errs, safety = {}, {}
    with torch.no_grad():
        for sid, name in enumerate(PRESET_NAMES):
            z = model.get_z(sid)
            vals, min_seps, n_collided = [], [], 0
            for _ in range(8):
                pos, vel, heading, smoothed_vel = random_init(sim_cfg)
                pos, vel, heading, smoothed_vel, _, pos_history = rollout(
                    model, pos, vel, heading, smoothed_vel, z, cfg.t_max, sim_cfg
                )
                if fixed:
                    vals.append(fixed_formation_loss(pos, target_pts[name]).item())
                else:
                    vals.append(distance_matrix_loss(pos, target_dms[name]).item())
                dm = pairwise_distance_matrix(torch.stack(pos_history))
                dm = dm + torch.eye(cfg.n) * 1e9   # mask self-distances
                min_sep = dm.min().item()
                min_seps.append(min_sep)
                n_collided += int(min_sep < collision_dist)
            errs[name] = sum(vals) / len(vals)
            safety[name] = {"min_separation": min(min_seps),
                            "collided_rollouts": n_collided, "rollouts": len(min_seps)}
    model.train()
    return errs, safety


def training_loss(cfg, pos, pos_history, vel_history, target_dm_batch,
                  target_pts, target_pts_batch, shape_ids):
    """
    THE COMPLETE TRAINING OBJECTIVE:

        total  =  1.0 * formation
                + cfg.damping_weight    * damping
                + cfg.separation_weight * separation          (issue #4)
                + cfg.bounds_weight     * bounds  [arena_mode == "fixed" only]

    * formation (weight fixed at 1.0 -- it is the reference scale everything
      else is weighted against):
        arena_mode none/walls -- distance-matrix MSE vs the target shape
                          (rotation/translation invariant; in "walls" mode the
                          repulsive boundary lives in the PHYSICS, sim.wall_accel,
                          not in the loss), averaged over the last cfg.hold_tail
                          rollout steps (losses.formation_hold_loss), so each
                          tail step effectively contributes 1/hold_tail;
        arena_mode fixed  -- coordinate MSE vs the targets at their FIXED,
                          arena-centered location (losses.fixed_formation_hold_loss),
                          same hold_tail averaging, NOT invariant (issue #4
                          method 1);
        --use_chamfer  -- Kabsch-aligned Chamfer on the END state only (the
                          per-item SVD isn't batched, so a tail x batch loop
                          would dominate the step). Invariant modes only:
                          alignment would erase exactly what "fixed" pins down.
    * damping (weight cfg.damping_weight):
        mean squared speed over the last cfg.damping_tail steps
        (losses.damping_loss). Parks the swarm: the rotation/translation-
        invariant formation term treats every rigid-motion copy of the target
        as equally correct, so without this the swarm may drift or spin.
    * separation (weight cfg.separation_weight, 0 disables):
        collision-avoidance hinge over EVERY rollout step and drone pair
        (losses.separation_loss), active below
        2*cfg.drone_radius + cfg.separation_margin. Zero once the swarm keeps
        safe distances, so at convergence it does not fight the formation term.
    * bounds (weight cfg.bounds_weight, "fixed" mode only, 0 disables):
        containment hinge over every rollout step (losses.bounds_loss),
        penalizing safety disks that cross the 3m x 3m boundary in transit.
        Unused in "walls" mode, where sim.wall_accel enforces containment
        physically, and in "none" mode, which has no arena.

    Nothing else in config.yaml is a loss weight -- t_min/t_max set the BPTT
    horizon distribution, replay_* set the fresh-vs-resumed batch mixture, and
    lr/lr_min/grad_clip belong to the optimizer.
    """
    arena_mode = getattr(cfg, "arena_mode", "none")
    if cfg.use_chamfer:
        form_loss = torch.mean(torch.stack([
            chamfer_loss(pos[b], target_pts[PRESET_NAMES[shape_ids[b]]])
            for b in range(cfg.batch_size)
        ]))
    elif arena_mode == "fixed":
        form_loss = fixed_formation_hold_loss(pos_history, target_pts_batch,
                                              cfg.hold_tail)
    else:
        form_loss = formation_hold_loss(pos_history, target_dm_batch, cfg.hold_tail)
    total = form_loss + cfg.damping_weight * damping_loss(vel_history, cfg.damping_tail)

    sep_weight = getattr(cfg, "separation_weight", 0.0)
    if sep_weight > 0:
        min_dist = 2.0 * getattr(cfg, "drone_radius", 0.1) \
            + getattr(cfg, "separation_margin", 0.0)
        total = total + sep_weight * separation_loss(pos_history, min_dist)

    bounds_weight = getattr(cfg, "bounds_weight", 0.0)
    if arena_mode == "fixed" and bounds_weight > 0:
        total = total + bounds_weight * bounds_loss(
            pos_history, cfg.arena_half, getattr(cfg, "drone_radius", 0.1))
    return total


def train_model(cfg, verbose: bool = True):
    """
    Run the full training loop for a given config and return
    (model, loss_curve, errors, sim_cfg). Factored out of main() so that
    compare.py can train several perception modes with one call each.
    """
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    sim_cfg = sim_config_from(cfg)
    n_shapes = len(PRESET_NAMES)
    arena_mode = getattr(cfg, "arena_mode", "none")
    if arena_mode == "fixed" and cfg.use_chamfer:
        raise ValueError("use_chamfer is incompatible with arena_mode='fixed': "
                         "Kabsch alignment erases the fixed position/rotation "
                         "that method 1 is meant to enforce.")

    # Precompute target distance matrices and target point sets. The canonical
    # shapes are origin-centered, which in "fixed" mode IS the required placement:
    # centered in the arena (the arena is origin-centered too, issue #4).
    target_dms = all_target_distance_matrices(cfg.n, cfg.shape_scale)
    target_pts = {name: get_shape(name, cfg.n, cfg.shape_scale) for name in PRESET_NAMES}

    model = GammaGNCA(
        n_agents=cfg.n,
        hidden=cfg.hidden,
        msg_dim=cfg.msg_dim,
        z_dim=cfg.z_dim,
        id_dim=cfg.id_dim,
        n_shapes=n_shapes,
        accel_scale=cfg.accel_scale,
        absolute_pos=(arena_mode == "fixed"),  # model.py header note 4
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.lr_min
    )
    cache = ReplayCache(cfg.replay_size, n_shapes)

    # Round-robin shape assignment per batch slot is fixed across epochs, so
    # precompute it once, along with the per-slot target distance matrices and the
    # per-slot z_shape ids (used to look up one latent code per batch item).
    shape_ids = [b % n_shapes for b in range(cfg.batch_size)]
    shape_id_tensor = torch.tensor(shape_ids, dtype=torch.long)
    target_dm_batch = torch.stack([target_dms[PRESET_NAMES[sid]] for sid in shape_ids])
    target_pts_batch = torch.stack([target_pts[PRESET_NAMES[sid]] for sid in shape_ids])

    loss_curve = []
    t0 = time.time()
    if verbose:
        print(f"Training {n_shapes} presets, perception={cfg.perception}, "
              f"N={cfg.n}, batch_size={cfg.batch_size}, {cfg.epochs} epochs on CPU...")

    for epoch in range(cfg.epochs):
        # Randomized horizon: same t for the whole batch keeps it simple/fast.
        t_steps = random.randint(cfg.t_min, cfg.t_max)
        opt.zero_grad()

        # Assemble one BATCH of initial states (still one Python call per slot,
        # since the replay cache is per-shape and cheap -- the expensive part,
        # rolled out below, runs as a single vectorized call for the whole batch).
        pos_list, vel_list, heading_list, smoothed_vel_list = [], [], [], []
        for b, shape_id in enumerate(shape_ids):
            force_fresh = b < cfg.replay_fresh  # guarantee some from-scratch inits
            p, v, h, sv = make_sample(cache, shape_id, cfg, sim_cfg, gen, force_fresh)
            pos_list.append(p); vel_list.append(v); heading_list.append(h)
            smoothed_vel_list.append(sv)
        pos = torch.stack(pos_list)                  # (B, N, 2)
        vel = torch.stack(vel_list)                  # (B, N, 2)
        heading = torch.stack(heading_list)          # (B, N, 2)
        smoothed_vel = torch.stack(smoothed_vel_list)  # (B, N, 2)
        z = model.z_shape(shape_id_tensor)   # (B, z_dim), one latent code per swarm

        pos, vel, heading, smoothed_vel, vel_history, pos_history = rollout(
            model, pos, vel, heading, smoothed_vel, z, t_steps, sim_cfg
        )

        loss = training_loss(cfg, pos, pos_history, vel_history,
                             target_dm_batch, target_pts, target_pts_batch, shape_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        # Stash each swarm's end-state for future replay.
        for b, shape_id in enumerate(shape_ids):
            cache.add(shape_id, (pos[b], vel[b], heading[b], smoothed_vel[b]))

        loss_curve.append(loss.item())
        if verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  t={t_steps:2d}  loss={loss.item():.4f}"
                  f"  ({time.time() - t0:5.1f}s)")

    errs, safety = evaluate(model, cfg, sim_cfg, target_dms, target_pts)
    if verbose:
        metric = ("fixed-target position MSE" if arena_mode == "fixed"
                  else "distance-matrix error")
        print(f"\nFinal per-shape {metric} (fresh inits) + collision check "
              f"(collision = centers < {2 * sim_cfg.drone_radius:.2f}m apart):")
        for name, e in errs.items():
            s = safety[name]
            print(f"  {name:9s}: {e:.4f}   min separation {s['min_separation']:.3f}m, "
                  f"collided in {s['collided_rollouts']}/{s['rollouts']} rollouts")
        print(f"Total train wall time: {time.time() - t0:.1f}s")
    return model, loss_curve, errs, sim_cfg, safety


def main():
    cfg = load_config()
    model, loss_curve, errs, sim_cfg, safety = train_model(cfg, verbose=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": vars(cfg),
            "loss_curve": loss_curve,
            "final_errors": errs,
            "final_safety": safety,
            "preset_names": PRESET_NAMES,
        },
        cfg.checkpoint,
    )
    print(f"\nSaved checkpoint -> {cfg.checkpoint}")

    # Save the loss curve plot too.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(loss_curve)
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss (log)")
        plt.title("GNCA training loss")
        plt.tight_layout()
        plt.savefig("loss_curve.png", dpi=110)
        print("Saved loss curve -> loss_curve.png")
    except Exception as e:  # plotting is non-essential
        print(f"(skipped loss plot: {e})")


if __name__ == "__main__":
    main()
