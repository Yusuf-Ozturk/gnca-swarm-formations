"""
sweep_knn.py
============
Train the kNN perception model for several values of k and collect every output,
classified into one folder per k.

For each k in --ks (default 2,4,6) it:
  * trains a fresh GNCA (everything else equal: same N, shapes, epochs, seed),
  * evaluates per-shape distance-matrix error on a SHARED set of random inits
    (identical across all k, so the only variable is the neighbour count),
  * saves into  runs/knn_k<k>/ :
        checkpoint.pt          (loadable by viz.py)
        loss_curve.png
        errors.txt
        convergence_<shape>.gif   (4)
        switching.gif             (square -> hexagon @ step 30)
  * and writes the cross-k roll-up at  runs/ :
        knn_k_summary.txt
        knn_k_comparison.png

Run:  python sweep_knn.py                 (uses config.yaml epochs; trains x len(ks))
      python sweep_knn.py --ks 2,4,6 --epochs 500
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


def _ks_arg(parser):
    parser.add_argument("--ks", default="2,4,6",
                        help="comma-separated k values to sweep, e.g. 2,4,6")
    # viz/render knobs (mirror viz.py defaults, since we render gifs here too)
    parser.add_argument("--total_steps", type=int, default=70)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--switch_step", type=int, default=30)
    parser.add_argument("--draw_cone", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=True)


def save_checkpoint(model, cfg_k, errs, path):
    torch.save(
        {
            "model_state": model.state_dict(),
            "cfg": vars(cfg_k),
            "final_errors": errs,
            "preset_names": PRESET_NAMES,
            "loss_curve": [],
        },
        path,
    )


def plot_loss(loss_curve, path, k):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_curve)
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("loss (log)")
    plt.title(f"kNN training loss (k={k})")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def render_gifs(model, sim_cfg, cfg, out_dir):
    """Convergence gif per shape + the dynamic-switch gif, written into out_dir."""
    for sid, name in enumerate(PRESET_NAMES):
        poses, headings = simulate(model, sim_cfg, [(0, sid)], cfg.total_steps,
                                   seed=sid + 1)
        target = get_shape(name, cfg.n, cfg.shape_scale)
        target = target - target.mean(0) + torch.tensor(poses[-1].mean(0))
        animate(poses, headings, sim_cfg,
                title=f"Convergence: {name}  [knn k={sim_cfg.knn_k}]",
                outfile=os.path.join(out_dir, f"convergence_{name}.gif"),
                fps=cfg.fps, draw_cone=cfg.draw_cone, target_pts=target.numpy())

    sid_from, sid_to = PRESET_NAMES.index("square"), PRESET_NAMES.index("hexagon")
    poses, headings = simulate(model, sim_cfg, [(0, sid_from), (cfg.switch_step, sid_to)],
                               cfg.total_steps, seed=42)

    def switch_label(frame):
        return "hexagon" if frame >= cfg.switch_step else "square"

    animate(poses, headings, sim_cfg,
            title=f"Dynamic switch: square -> hexagon @ step {cfg.switch_step}  "
                  f"[knn k={sim_cfg.knn_k}]",
            outfile=os.path.join(out_dir, "switching.gif"),
            fps=cfg.fps, draw_cone=cfg.draw_cone,
            switch_step=cfg.switch_step, switch_label=switch_label)


def main():
    cfg = load_config(extra_args=_ks_arg)
    ks = [int(x) for x in str(cfg.ks).split(",") if x.strip()]
    os.makedirs(RUNS_DIR, exist_ok=True)

    target_dms = all_target_distance_matrices(cfg.n, cfg.shape_scale)
    # Shared inits (perception-independent) reused for every k -> a fair comparison.
    inits = shared_inits(sim_config_from(cfg), n_inits=16)

    all_errs = {}
    for k in ks:
        print(f"\n=================== kNN sweep: k = {k} ===================")
        cfg_k = copy.copy(cfg)
        cfg_k.perception = "knn"
        cfg_k.knn_k = k
        model, loss_curve, _, sim_cfg = train_model(cfg_k, verbose=True)

        out_dir = os.path.join(RUNS_DIR, f"knn_k{k}")
        os.makedirs(out_dir, exist_ok=True)

        errs = eval_on(model, sim_cfg, inits, target_dms, cfg.t_max)
        all_errs[k] = errs

        save_checkpoint(model, cfg_k, errs, os.path.join(out_dir, "checkpoint.pt"))
        plot_loss(loss_curve, os.path.join(out_dir, "loss_curve.png"), k)
        with open(os.path.join(out_dir, "errors.txt"), "w") as f:
            f.write(f"kNN k={k}  (per-shape distance-matrix error, "
                    f"mean over {len(inits)} shared inits)\n")
            for name in PRESET_NAMES:
                f.write(f"  {name:9s}: {errs[name]:.5f}\n")
            f.write(f"  {'MEAN':9s}: {sum(errs.values())/len(PRESET_NAMES):.5f}\n")
        print(f"  rendering gifs -> {out_dir}/ ...")
        render_gifs(model, sim_cfg, cfg, out_dir)
        print(f"  collected outputs for k={k} in {out_dir}/")

    # ---------- cross-k roll-up ----------
    lines = []
    header = f"{'shape':10s}" + "".join(f"{'k='+str(k):>12s}" for k in ks)
    lines.append("kNN perception sweep - distance-matrix error (lower = better)\n")
    lines.append(header)
    lines.append("-" * len(header))
    for name in PRESET_NAMES:
        row = f"{name:10s}" + "".join(f"{all_errs[k][name]:12.5f}" for k in ks)
        lines.append(row)
    means = {k: sum(all_errs[k].values()) / len(PRESET_NAMES) for k in ks}
    lines.append("-" * len(header))
    lines.append(f"{'MEAN':10s}" + "".join(f"{means[k]:12.5f}" for k in ks))
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(RUNS_DIR, "knn_k_summary.txt"), "w") as f:
        f.write(summary + "\n")

    # grouped bar chart: x = shapes, one bar series per k
    x = np.arange(len(PRESET_NAMES))
    w = 0.8 / len(ks)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, k in enumerate(ks):
        ax.bar(x + (i - (len(ks) - 1) / 2) * w,
               [all_errs[k][n] for n in PRESET_NAMES], w, label=f"k={k}")
    ax.set_xticks(x); ax.set_xticklabels(PRESET_NAMES)
    ax.set_ylabel("distance-matrix error (lower = better)")
    ax.set_title("kNN perception: effect of neighbour count k")
    ax.legend(title="neighbours")
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "knn_k_comparison.png"), dpi=110)
    plt.close()
    print(f"\nSaved roll-up -> {RUNS_DIR}/knn_k_summary.txt and "
          f"{RUNS_DIR}/knn_k_comparison.png")
    print("Animate any run with, e.g.:  "
          "python viz.py --checkpoint runs/knn_k4/checkpoint.pt")


if __name__ == "__main__":
    main()
