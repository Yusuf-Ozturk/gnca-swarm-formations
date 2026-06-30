"""
train.py
========
Train the shared GNCA rule gamma to drive a swarm into every preset formation.

Key training ingredients:
  * BPTT over a rollout whose length t is sampled randomly per batch (t in
    [t_min, t_max]) so the target must be an ATTRACTOR reached after any number of
    steps, not memorized at exactly t.
  * A REPLAY CACHE (a la Mordvintsev / Grattarola): we stash the end-states of
    rollouts and re-seed some training samples from them, so the model learns to
    keep refining already-near-target states (and doesn't settle into a periodic
    jitter). Fresh random inits are periodically re-injected to avoid forgetting
    how to form a shape from scratch.
  * Every preset is trained every epoch so the single shared gamma learns all the
    z_shape codes.

Run:  python train.py            (uses config.yaml defaults; ~a few min on CPU)
"""

from __future__ import annotations

import random
import time
from collections import deque

import torch

from config import load_config, sim_config_from
from losses import chamfer_loss, damping_loss, distance_matrix_loss
from model import GammaGNCA
from shapes import PRESET_NAMES, get_shape, all_target_distance_matrices
from sim import random_init, rollout


class ReplayCache:
    """
    REPLAY CACHE. A bounded pool of rollout end-states, keyed by shape id. Each
    entry is a (pos, vel, heading) tuple (detached). We sample from it to continue
    refining near-converged states, and periodically overwrite entries with fresh
    random inits so the model never forgets the from-scratch task.
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
            pos, vel, heading = (t.clone() for t in s)
            return pos, vel, heading
    return random_init(sim_cfg, generator=gen)


def evaluate(model, cfg, sim_cfg, target_dms):
    """Final per-shape distance-matrix error from fresh random inits (no grad)."""
    model.eval()
    errs = {}
    with torch.no_grad():
        for sid, name in enumerate(PRESET_NAMES):
            z = model.get_z(sid)
            vals = []
            for _ in range(8):
                pos, vel, heading = random_init(sim_cfg)
                pos, vel, heading, _ = rollout(
                    model, pos, vel, heading, z, cfg.t_max, sim_cfg
                )
                vals.append(distance_matrix_loss(pos, target_dms[name]).item())
            errs[name] = sum(vals) / len(vals)
    model.train()
    return errs


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

    # Precompute target distance matrices and target point sets.
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
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    cache = ReplayCache(cfg.replay_size, n_shapes)

    loss_curve = []
    t0 = time.time()
    if verbose:
        print(f"Training {n_shapes} presets, perception={cfg.perception}, "
              f"N={cfg.n}, {cfg.epochs} epochs on CPU...")

    for epoch in range(cfg.epochs):
        # Randomized horizon: same t for the whole batch keeps it simple/fast.
        t_steps = random.randint(cfg.t_min, cfg.t_max)
        opt.zero_grad()
        batch_loss = 0.0

        for b in range(cfg.batch_size):
            # Round-robin over shapes so every z_shape is trained each epoch.
            shape_id = b % n_shapes
            name = PRESET_NAMES[shape_id]
            z = model.get_z(shape_id)

            force_fresh = b < cfg.replay_fresh  # guarantee some from-scratch inits
            pos, vel, heading = make_sample(cache, shape_id, cfg, sim_cfg, gen, force_fresh)

            pos, vel, heading, vel_history = rollout(
                model, pos, vel, heading, z, t_steps, sim_cfg
            )

            if cfg.use_chamfer:
                form_loss = chamfer_loss(pos, target_pts[name])
            else:
                form_loss = distance_matrix_loss(pos, target_dms[name])
            damp = damping_loss(vel_history, cfg.damping_tail)
            loss = form_loss + cfg.damping_weight * damp
            batch_loss = batch_loss + loss

            # Stash the end-state for future replay.
            cache.add(shape_id, (pos, vel, heading))

        batch_loss = batch_loss / cfg.batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        loss_curve.append(batch_loss.item())
        if verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  t={t_steps:2d}  loss={batch_loss.item():.4f}"
                  f"  ({time.time() - t0:5.1f}s)")

    errs = evaluate(model, cfg, sim_cfg, target_dms)
    if verbose:
        print("\nFinal per-shape distance-matrix error (fresh inits):")
        for name, e in errs.items():
            print(f"  {name:9s}: {e:.4f}")
        print(f"Total train wall time: {time.time() - t0:.1f}s")
    return model, loss_curve, errs, sim_cfg


def main():
    cfg = load_config()
    model, loss_curve, errs, sim_cfg = train_model(cfg, verbose=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": vars(cfg),
            "loss_curve": loss_curve,
            "final_errors": errs,
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
