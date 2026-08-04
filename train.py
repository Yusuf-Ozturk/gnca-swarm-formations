"""
train.py
========
Train one gamma for one formation.

There is no shape-conditioning input in the model, so a trained network *is* a
formation: `--shape wedge --perception cone` produces one network that flies four
identical drones into a wedge using a forward field of view, and nothing else.
The full study is this command run once per (shape, perception) pair.

Protocol:
  * BPTT over a rollout whose length t is drawn fresh each epoch from
    [t_min, t_max] (4-12 s at dt=0.1), so the formation must be an ATTRACTOR
    reached after any number of steps rather than a state memorized at exactly t.
  * FRESH RANDOM INITS ONLY -- no replay cache. Every training swarm starts from
    a new random configuration at rest, so nothing in the objective is
    conditioned on states the model itself produced earlier.
  * COLLISION CURRICULUM: both collision tiers ramp linearly from 0 to full
    weight over the first `separation_ramp` fraction of training. At full weight
    from epoch 0 they dominate, and the swarm learns to keep its distance before
    it can form anything at all.
  * Cosine-annealed learning rate, gradients clipped at `grad_clip`.

Run:  python train.py --shape square --perception cone
      python train.py --shape wedge --perception circular --epochs 500
"""

from __future__ import annotations

import math
import os
import random
import time

import torch

from config import load_config, sim_config_from
from losses import assignment_loss, collision_stats, total_loss
from model import GammaSwarm
from shapes import PRESET_NAMES, get_shape
from sim import random_init, rollout


def _extra_args(parser):
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from its checkpoint")
    parser.add_argument("--save_every", type=int, default=50,
                        help="write an in-progress checkpoint every N epochs (0 = off)")


def cosine_lr(epoch: int, cfg) -> float:
    """Cosine anneal from cfg.lr down to cfg.lr_min across the run."""
    if cfg.epochs <= 1:
        return cfg.lr
    frac = epoch / (cfg.epochs - 1)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + math.cos(math.pi * frac))


def ramp_scale(epoch: int, cfg) -> float:
    """Collision-curriculum weight in [0, 1] (see module docstring)."""
    if cfg.separation_ramp <= 0:
        return 1.0
    return min(1.0, epoch / max(1.0, cfg.separation_ramp * cfg.epochs))


def evaluate(model, sim_cfg, target, n_runs: int = 8, steps: int = 200):
    """
    Final quality from fresh random inits, no gradient: the pose- and
    permutation-invariant formation error at the settled state, plus the
    collision check -- worst separation over the whole flight, worst separation
    at the FINAL state, and how many steps contained a collision.
    """
    model.eval()
    errs, worst, worst_final, coll_steps = [], float("inf"), float("inf"), 0
    with torch.no_grad():
        for k in range(n_runs):
            g = torch.Generator().manual_seed(10_000 + k)
            pos, theta, s, omega = random_init(sim_cfg, generator=g)
            pos, theta, s, omega, hist = rollout(
                model, pos, theta, s, omega, steps, sim_cfg)
            errs.append(assignment_loss(pos, target).item())
            mn, fin, cs = collision_stats(hist, sim_cfg.drone_radius)
            worst = min(worst, mn)
            worst_final = min(worst_final, fin)
            coll_steps += cs
    model.train()
    return {
        "formation_error": sum(errs) / len(errs),
        "min_separation_m": worst,
        "final_min_separation_m": worst_final,
        "collision_steps": coll_steps,
        "n_runs": n_runs,
        "eval_steps": steps,
    }


def save_checkpoint(model, cfg, loss_curve, stats, path, opt=None):
    """
    Write a checkpoint. Called periodically during training, not only at the end,
    and it carries the optimizer state so an interrupted run RESUMES rather than
    restarting: this environment restarts containers mid-run, and a fresh Adam
    state after a resume would be a silent change to the training protocol.
    """
    torch.save({
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict() if opt is not None else None,
        "cfg": vars(cfg),
        "loss_curve": loss_curve,
        "eval": stats,
        "shape": cfg.shape,
        "epochs_done": len(loss_curve),
    }, path)


def train_model(cfg, verbose: bool = True):
    """Train one network. Returns (model, loss_curve, eval_stats, sim_cfg)."""
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    sim_cfg = sim_config_from(cfg)
    target = get_shape(cfg.shape, cfg.shape_scale)

    model = GammaSwarm(hidden=cfg.hidden, msg_dim=cfg.msg_dim,
                       a_lin_scale=cfg.a_lin_scale, a_ang_scale=cfg.a_ang_scale)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Training gamma for shape='{cfg.shape}', perception={cfg.perception}, "
              f"N={cfg.n}, {n_params} params, {cfg.epochs} epochs on CPU...")

    loss_curve = []
    start_epoch = 0
    # RESUME: pick up an interrupted run exactly where it stopped, restoring the
    # optimizer state and the RNG stream position (the per-epoch horizon and the
    # init sampler both draw from it, so replaying the consumed draws keeps a
    # resumed run on the same trajectory as an uninterrupted one).
    if getattr(cfg, "resume", False) and os.path.exists(cfg.checkpoint):
        ck = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        if ck.get("opt_state"):
            opt.load_state_dict(ck["opt_state"])
        loss_curve = list(ck.get("loss_curve", []))
        start_epoch = len(loss_curve)
        for _ in range(start_epoch):
            random.randint(cfg.t_min, cfg.t_max)
            random_init(sim_cfg, batch=cfg.batch_size, generator=gen)
        if verbose:
            print(f"  resuming from {cfg.checkpoint} at epoch {start_epoch}")

    t0 = time.time()
    for epoch in range(start_epoch, cfg.epochs):
        for group in opt.param_groups:
            group["lr"] = cosine_lr(epoch, cfg)
        t_steps = random.randint(cfg.t_min, cfg.t_max)
        sep_scale = ramp_scale(epoch, cfg)

        pos, theta, s, omega = random_init(sim_cfg, batch=cfg.batch_size, generator=gen)
        _, _, _, _, pos_history = rollout(model, pos, theta, s, omega, t_steps, sim_cfg)
        loss = total_loss(cfg, pos_history, target, sep_scale)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        loss_curve.append(loss.item())
        # Periodic checkpoint: a run takes about an hour, and a container restart
        # mid-run would otherwise discard every epoch of it.
        save_every = getattr(cfg, "save_every", 0)
        if save_every and (epoch + 1) % save_every == 0:
            save_checkpoint(model, cfg, loss_curve, None, cfg.checkpoint, opt)
        if verbose and (epoch % 25 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  t={t_steps:3d}  ramp={sep_scale:.2f}  "
                  f"loss={loss.item():.4f}  ({time.time() - t0:6.1f}s)")

    stats = evaluate(model, sim_cfg, target)
    if verbose:
        print(f"\nFinal (fresh inits, {stats['n_runs']} runs x {stats['eval_steps']} steps):")
        print(f"  formation error (aligned, best assignment): "
              f"{stats['formation_error']:.5f}")
        print(f"  min separation any step : {stats['min_separation_m']:.3f} m")
        print(f"  min separation at final : {stats['final_min_separation_m']:.3f} m")
        print(f"  colliding steps (< {2 * sim_cfg.drone_radius:.2f} m): "
              f"{stats['collision_steps']}")
        print(f"Total train wall time: {time.time() - t0:.1f}s")
    return model, loss_curve, stats, sim_cfg


def main():
    cfg = load_config(extra_args=_extra_args)
    if cfg.shape not in PRESET_NAMES:
        raise SystemExit(f"--shape must be one of {PRESET_NAMES}, got '{cfg.shape}'")
    model, loss_curve, stats, sim_cfg = train_model(cfg, verbose=not cfg.quiet)

    save_checkpoint(model, cfg, loss_curve, stats, cfg.checkpoint, opt)
    print(f"\nSaved checkpoint -> {cfg.checkpoint}")


if __name__ == "__main__":
    main()
