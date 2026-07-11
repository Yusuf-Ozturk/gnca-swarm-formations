"""
compare.py
==========
Head-to-head comparison of the two perception models -- the forward-FOV CONE vs
the omnidirectional CIRCULAR (360-degree) FOV -- under otherwise identical
settings.

It trains one GNCA per perception mode (everything else equal: same N, same shapes,
same epochs, same seed) and then evaluates both on the SAME shared set of random
initial conditions, reporting the per-shape distance-matrix error. Each model is
rolled out with its OWN perception graph -- the only thing that differs between the
two runs is what each agent is allowed to see.

Run:  python compare.py                  (uses config.yaml epochs; ~train x2)
      python compare.py --epochs 500 --sensing_range 1.2
Output: a printed table + compare_perception.png (grouped bar chart).
"""

from __future__ import annotations

import copy

import torch

from config import load_config, sim_config_from
from losses import distance_matrix_loss
from shapes import PRESET_NAMES, all_target_distance_matrices
from sim import random_init, rollout
from train import train_model


def shared_inits(sim_cfg, n_inits: int, base_seed: int = 1000):
    """A fixed list of (pos, vel, heading, smoothed_vel) inits reused for every
    model -> fair test."""
    inits = []
    for i in range(n_inits):
        g = torch.Generator().manual_seed(base_seed + i)
        inits.append(random_init(sim_cfg, generator=g))
    return inits


def eval_on(model, sim_cfg, inits, target_dms, t_steps):
    """Per-shape mean distance-matrix error of one model over the shared inits."""
    model.eval()
    errs = {}
    with torch.no_grad():
        for sid, name in enumerate(PRESET_NAMES):
            z = model.get_z(sid)
            vals = []
            for (pos, vel, heading, smoothed_vel) in inits:
                p, v, h, sv, _ = rollout(model, pos.clone(), vel.clone(), heading.clone(),
                                         smoothed_vel.clone(), z, t_steps, sim_cfg)
                vals.append(distance_matrix_loss(p, target_dms[name]).item())
            errs[name] = sum(vals) / len(vals)
    return errs


def main():
    cfg = load_config()
    target_dms = all_target_distance_matrices(cfg.n, cfg.shape_scale)

    results = {}
    sim_cfgs = {}
    for mode in ("cone", "circular"):
        print(f"\n===== Training perception = {mode} =====")
        cfg_m = copy.copy(cfg)
        cfg_m.perception = mode
        model, _, _, sim_cfg = train_model(cfg_m, verbose=True)
        results[mode] = model
        sim_cfgs[mode] = sim_cfg

    # Identical inits for both; evaluate each under its own perception graph.
    inits = shared_inits(sim_cfgs["cone"], n_inits=16)
    errs = {m: eval_on(results[m], sim_cfgs[m], inits, target_dms, cfg.t_max)
            for m in ("cone", "circular")}

    # --- table ---
    print("\n================ PERCEPTION COMPARISON ================")
    print(f"(per-shape distance-matrix error, mean over {len(inits)} shared inits, "
          f"lower = better)\n")
    print(f"{'shape':10s}{'cone (FOV)':>14s}{'circular':>12s}{'winner':>10s}")
    for name in PRESET_NAMES:
        c, k = errs["cone"][name], errs["circular"][name]
        win = "cone" if c < k else "circular"
        print(f"{name:10s}{c:14.4f}{k:12.4f}{win:>10s}")
    mc = sum(errs["cone"].values()) / len(PRESET_NAMES)
    mk = sum(errs["circular"].values()) / len(PRESET_NAMES)
    print("-" * 46)
    print(f"{'MEAN':10s}{mc:14.4f}{mk:12.4f}{('cone' if mc < mk else 'circular'):>10s}")
    print(f"\ncone half_angle={cfg.half_angle_deg} deg, sensing_range={cfg.sensing_range}")

    # --- grouped bar chart ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(PRESET_NAMES))
        w = 0.38
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(x - w / 2, [errs["cone"][n] for n in PRESET_NAMES], w,
               label="cone (forward FOV)", color="tab:blue")
        ax.bar(x + w / 2, [errs["circular"][n] for n in PRESET_NAMES], w,
               label="circular (360-degree FOV)", color="tab:orange")
        ax.set_xticks(x); ax.set_xticklabels(PRESET_NAMES)
        ax.set_ylabel("distance-matrix error (lower = better)")
        ax.set_title("Perception comparison: forward-FOV cone vs circular FOV")
        ax.legend()
        plt.tight_layout()
        plt.savefig("compare_perception.png", dpi=110)
        print("\nSaved chart -> compare_perception.png")
    except Exception as e:
        print(f"(skipped chart: {e})")

    # Save both checkpoints so viz.py can animate either.
    for mode in ("cone", "circular"):
        cfg_m = copy.copy(cfg); cfg_m.perception = mode
        torch.save(
            {"model_state": results[mode].state_dict(), "cfg": vars(cfg_m),
             "final_errors": errs[mode], "preset_names": PRESET_NAMES,
             "loss_curve": []},
            f"checkpoint_{mode}.pt",
        )
    print("Saved checkpoint_cone.pt and checkpoint_circular.pt")
    print("Animate either with:  python viz.py --checkpoint checkpoint_circular.pt")


if __name__ == "__main__":
    main()
