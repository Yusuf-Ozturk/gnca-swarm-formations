"""
sweep_range.py
==============
Train the circular (360-degree FOV) perception model for several sensing ranges
and collect every output, classified into one folder per range.

For each range in --ranges (default 0.6,1.0,1.4) it:
  * trains a fresh GNCA (everything else equal: same N, shapes, epochs, seed),
  * evaluates per-shape distance-matrix error on a SHARED set of random inits
    (identical across all ranges, so the only variable is the sensing radius),
  * saves into  runs/circular_r<range>/ :
        checkpoint.pt          (loadable by viz.py)
        loss_curve.png
        errors.txt
        convergence_<shape>.gif   (4)
        switching.gif             (square -> hexagon @ step 30)
  * and writes the cross-range roll-up at  runs/ :
        circular_range_summary.txt
        circular_range_comparison.png

Run:  python sweep_range.py                       (uses config.yaml epochs; trains x len(ranges))
      python sweep_range.py --ranges 0.6,1.0,1.4 --epochs 500
"""

from __future__ import annotations

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare import eval_on, shared_inits
from config import load_config, sim_config_from
from shapes import PRESET_NAMES, all_target_distance_matrices, get_shape
from train import train_model
from viz import animate, simulate

RUNS_DIR = "runs"


def _ranges_arg(parser):
    parser.add_argument("--ranges", default="0.6,1.0,1.4",
                        help="comma-separated sensing_range values to sweep, e.g. 0.6,1.0,1.4")
    # viz/render knobs (mirror viz.py defaults, since we render gifs here too)
    parser.add_argument("--total_steps", type=int, default=70)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--switch_step", type=int, default=30)
    parser.add_argument("--draw_cone", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=True)


def save_checkpoint(model, cfg_r, errs, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": vars(cfg_r),
            "final_errors": errs,
            "preset_names": PRESET_NAMES,
            "loss_curve": [],
        },
        path,
    )


def plot_loss(loss_curve, path, r):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_curve)
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("loss (log)")
    plt.title(f"Circular FOV training loss (range={r})")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def render_gifs(model, sim_cfg, cfg, out_dir):
    """Convergence gif per shape + the dynamic-switch gif, written into out_dir."""
    for sid, name in enumerate(PRESET_NAMES):
        poses, headings = simulate(model, sim_cfg, [(0, sid)], cfg.total_steps,
                                   seed=sid + 1)
        target = get_shape(name, cfg.n, cfg.shape_scale)
        target = target - target.mean(0) + torch.tensor(poses[-1].mean(0))
        animate(poses, headings, sim_cfg,
                title=f"Convergence: {name}  [circular range={sim_cfg.sensing_range}]",
                outfile=os.path.join(out_dir, f"convergence_{name}.gif"),
                fps=cfg.fps, draw_cone=cfg.draw_cone, target_pts=target.numpy())

    sid_from, sid_to = PRESET_NAMES.index("square"), PRESET_NAMES.index("hexagon")
    poses, headings = simulate(model, sim_cfg, [(0, sid_from), (cfg.switch_step, sid_to)],
                               cfg.total_steps, seed=42)

    def switch_label(frame):
        return "hexagon" if frame >= cfg.switch_step else "square"

    animate(poses, headings, sim_cfg,
            title=f"Dynamic switch: square -> hexagon @ step {cfg.switch_step}  "
                  f"[circular range={sim_cfg.sensing_range}]",
            outfile=os.path.join(out_dir, "switching.gif"),
            fps=cfg.fps, draw_cone=cfg.draw_cone,
            switch_step=cfg.switch_step, switch_label=switch_label)


def main():
    cfg = load_config(extra_args=_ranges_arg)
    ranges = [float(x) for x in str(cfg.ranges).split(",") if x.strip()]
    os.makedirs(RUNS_DIR, exist_ok=True)

    target_dms = all_target_distance_matrices(cfg.n, cfg.shape_scale)
    # Shared inits (perception-independent) reused for every range -> a fair comparison.
    inits = shared_inits(sim_config_from(cfg), n_inits=16)

    all_errs = {}
    for r in ranges:
        print(f"\n=============== circular FOV sweep: range = {r} ===============")
        cfg_r = copy.copy(cfg)
        cfg_r.perception = "circular"
        cfg_r.sensing_range = r
        model, loss_curve, _, sim_cfg, _ = train_model(cfg_r, verbose=True)

        out_dir = os.path.join(RUNS_DIR, f"circular_r{r}")
        os.makedirs(out_dir, exist_ok=True)

        errs = eval_on(model, sim_cfg, inits, target_dms, cfg.t_max)
        all_errs[r] = errs

        save_checkpoint(model, cfg_r, errs, os.path.join(out_dir, "checkpoint.pt"))
        plot_loss(loss_curve, os.path.join(out_dir, "loss_curve.png"), r)
        with open(os.path.join(out_dir, "errors.txt"), "w") as f:
            f.write(f"circular FOV range={r}  (per-shape distance-matrix error, "
                    f"mean over {len(inits)} shared inits)\n")
            for name in PRESET_NAMES:
                f.write(f"  {name:9s}: {errs[name]:.5f}\n")
            f.write(f"  {'MEAN':9s}: {sum(errs.values())/len(PRESET_NAMES):.5f}\n")
        print(f"  rendering gifs -> {out_dir}/ ...")
        render_gifs(model, sim_cfg, cfg, out_dir)
        print(f"  collected outputs for range={r} in {out_dir}/")

    # ---------- cross-range roll-up ----------
    lines = []
    header = f"{'shape':10s}" + "".join(f"{'r='+str(r):>12s}" for r in ranges)
    lines.append("Circular FOV sweep - distance-matrix error (lower = better)\n")
    lines.append(header)
    lines.append("-" * len(header))
    for name in PRESET_NAMES:
        row = f"{name:10s}" + "".join(f"{all_errs[r][name]:12.5f}" for r in ranges)
        lines.append(row)
    means = {r: sum(all_errs[r].values()) / len(PRESET_NAMES) for r in ranges}
    lines.append("-" * len(header))
    lines.append(f"{'MEAN':10s}" + "".join(f"{means[r]:12.5f}" for r in ranges))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(RUNS_DIR, "circular_range_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # grouped bar chart: x = shapes, one bar series per range
    x = np.arange(len(PRESET_NAMES))
    w = 0.8 / len(ranges)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(ranges):
        ax.bar(x + (i - (len(ranges) - 1) / 2) * w,
               [all_errs[r][n] for n in PRESET_NAMES], w, label=f"r={r}")
    ax.set_xticks(x); ax.set_xticklabels(PRESET_NAMES)
    ax.set_ylabel("distance-matrix error (lower = better)")
    ax.set_title("Circular FOV perception: effect of sensing range")
    ax.legend(title="sensing range")
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "circular_range_comparison.png"), dpi=110)
    plt.close()
    print(f"\nSaved roll-up -> {RUNS_DIR}/circular_range_summary.txt and "
          f"{RUNS_DIR}/circular_range_comparison.png")
    print("Animate any run with, e.g.:  "
          "python viz.py --checkpoint runs/circular_r1.0/checkpoint.pt")


if __name__ == "__main__":
    main()
