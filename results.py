"""
results.py
==========
Evaluate every trained gamma in runs/ and write the study's results.

Per checkpoint, from `n_seeds` fresh random inits rolled out for `steps`:

  * FORMATION -- the pose- and permutation-invariant assignment error, sampled
    over time so convergence and any late drift are both visible.
  * COLLISIONS -- the whole point of the collision loss, reported twice:
    over every step of the flight, and at the FINAL state on its own. A breach
    during the fly-in and a parked formation that overlaps are different
    failures; the second means the learned attractor itself is unsafe.
  * EQUILIBRIUM -- mean forward speed and yaw rate over the last second. There
    is no damping term in the objective and no drag in the physics, so coming to
    rest is something gamma either discovered or did not. These numbers are how
    we tell, rather than assuming a low formation error implies a parked swarm.
  * FLYABILITY -- peak speed and yaw rate against the actuation envelope.

Outputs:
    results/<shape>_<perception>/summary.md, metrics.json, animation.mp4
    results/README.md      -- the cone-vs-circular comparison across all shapes
    results/collisions.png -- final-state separation against the collision line

Run:  python results.py                      # everything in runs/
      python results.py --no_animations
"""

from __future__ import annotations

import glob
import json
import math
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import load_config, sim_config_from
from losses import assignment_loss
from shapes import PRESET_NAMES, get_shape, min_slot_spacing
from viz import animate, load_model, simulate

RESULTS_DIR = "results"
RUNS_DIR = "runs"
PERCEPTIONS = ["cone", "circular"]


def _args(parser):
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=300,
                        help="rollout length for evaluation (300 = 30 s at dt=0.1)")
    parser.add_argument("--sample_every_s", type=float, default=5.0)
    parser.add_argument("--no_animations", action="store_true")


def evaluate_checkpoint(path, cfg):
    """Roll out one checkpoint over several seeds and collect every metric."""
    model, ckpt = load_model(path)
    saved = ckpt["cfg"]
    sim_cfg = sim_config_from(SimpleNamespace(**saved))
    shape = ckpt.get("shape", saved.get("shape"))
    target = get_shape(shape, saved["shape_scale"])
    coll_dist = 2 * sim_cfg.drone_radius
    every = max(1, round(cfg.sample_every_s / sim_cfg.dt))

    per_seed = []
    for seed in range(1, cfg.n_seeds + 1):
        poses, thetas, speeds = simulate(model, sim_cfg, cfg.eval_steps, seed=seed)
        P = np.array(poses)                                  # (T+1, N, 2)
        S = np.array(speeds)                                 # (T+1, N)
        TH = np.unwrap(np.array(thetas), axis=0)             # (T+1, N)

        d = np.linalg.norm(P[:, :, None] - P[:, None, :], axis=-1)
        d[:, np.arange(P.shape[1]), np.arange(P.shape[1])] = np.inf
        min_sep_per_step = d.min(axis=(1, 2))

        # formation error vs time
        curve, times = [], []
        for t in range(0, len(poses), every):
            curve.append(assignment_loss(torch.from_numpy(P[t]), target).item())
            times.append(t * sim_cfg.dt)
        final_err = assignment_loss(torch.from_numpy(P[-1]), target).item()

        # equilibrium over the last simulated second
        tail = max(1, round(1.0 / sim_cfg.dt))
        yaw_rate = np.abs(np.diff(TH, axis=0)) / sim_cfg.dt   # rad/s
        per_seed.append({
            "seed": seed,
            "times_s": times,
            "error_curve": curve,
            "final_error": final_err,
            "min_separation_m": float(min_sep_per_step.min()),
            "final_min_separation_m": float(min_sep_per_step[-1]),
            "collision_steps": int((min_sep_per_step < coll_dist).sum()),
            "final_mean_speed_mps": float(S[-tail:].mean()),
            "final_mean_yaw_rate_deg_s": float(np.degrees(yaw_rate[-tail:]).mean()),
            "max_speed_mps": float(S.max()),
            "max_yaw_rate_deg_s": float(np.degrees(yaw_rate).max()),
        })

    agg = {
        "shape": shape,
        "perception": sim_cfg.perception,
        "checkpoint": os.path.basename(path),
        "config": {k: saved.get(k) for k in
                   ("n", "dt", "perception", "half_angle_deg", "sensing_range",
                    "max_speed", "max_accel", "max_yaw_rate_deg", "max_yaw_accel_deg",
                    "hidden", "msg_dim", "epochs", "lr", "t_min", "t_max", "hold_tail",
                    "separation_weight", "collision_weight", "separation_margin",
                    "separation_ramp", "drone_radius", "shape_scale", "seed")},
        "eval_steps": cfg.eval_steps,
        "n_seeds": cfg.n_seeds,
        "target_min_slot_spacing_m": min_slot_spacing(shape, saved["shape_scale"]),
        "per_seed": per_seed,
        "final_error_mean": float(np.mean([s["final_error"] for s in per_seed])),
        "final_error_worst": float(np.max([s["final_error"] for s in per_seed])),
        "min_separation_worst": float(np.min([s["min_separation_m"] for s in per_seed])),
        "final_min_separation_worst":
            float(np.min([s["final_min_separation_m"] for s in per_seed])),
        "collision_steps_total": int(np.sum([s["collision_steps"] for s in per_seed])),
        "final_speed_mean": float(np.mean([s["final_mean_speed_mps"] for s in per_seed])),
        "final_yaw_rate_mean":
            float(np.mean([s["final_mean_yaw_rate_deg_s"] for s in per_seed])),
        "max_speed": float(np.max([s["max_speed_mps"] for s in per_seed])),
        "max_yaw_rate": float(np.max([s["max_yaw_rate_deg_s"] for s in per_seed])),
        "training_loss_curve": ckpt.get("loss_curve", [])[-1:],
    }
    return agg, model, sim_cfg, target, ckpt


def write_summary(agg, outdir):
    m = agg
    coll = 2 * m["config"]["drone_radius"]
    lines = [f"# {m['shape']} / {m['perception']}\n",
             f"Checkpoint: `{m['checkpoint']}`  --  {m['n_seeds']} seeds x "
             f"{m['eval_steps']} steps ({m['eval_steps'] * m['config']['dt']:.0f} s)\n",
             "## Config\n", "| key | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in m["config"].items()]

    lines += ["\n## Formation (pose- and permutation-invariant assignment error)\n",
              "| seed | " + " | ".join(f"{t:.0f}s" for t in m["per_seed"][0]["times_s"])
              + " | final |", "|---|" + "---|" * (len(m["per_seed"][0]["times_s"]) + 1)]
    for s in m["per_seed"]:
        lines.append(f"| {s['seed']} | " + " | ".join(f"{e:.4f}" for e in s["error_curve"])
                     + f" | {s['final_error']:.4f} |")
    lines.append(f"\nmean final error **{m['final_error_mean']:.5f}**, "
                 f"worst seed {m['final_error_worst']:.5f}")

    lines += [f"\n## Collisions (collision = centers < {coll:.2f} m)\n",
              "| seed | min sep, any step | min sep, FINAL | colliding steps |",
              "|---|---|---|---|"]
    for s in m["per_seed"]:
        flag = " **COLLISION**" if s["final_min_separation_m"] < coll else ""
        lines.append(f"| {s['seed']} | {s['min_separation_m']:.3f} | "
                     f"{s['final_min_separation_m']:.3f}{flag} | {s['collision_steps']} |")
    final_clear = m["final_min_separation_worst"] >= coll
    all_clear = m["collision_steps_total"] == 0
    lines += [f"\nTarget slots are {m['target_min_slot_spacing_m']:.2f} m apart, so a "
              f"perfect formation is collision-free by construction.\n",
              "**Final state: " + ("no collisions.**" if final_clear
                                   else "COLLISION.**"),
              "**All steps: " + ("no collisions.**" if all_clear
                                 else f"{m['collision_steps_total']} colliding steps.**")]

    lines += ["\n## Equilibrium (no damping term, no drag -- gamma had to find this)\n",
              "| seed | mean speed, last 1 s | mean yaw rate, last 1 s | peak speed | "
              "peak yaw rate |", "|---|---|---|---|---|"]
    for s in m["per_seed"]:
        lines.append(f"| {s['seed']} | {s['final_mean_speed_mps']:.4f} m/s | "
                     f"{s['final_mean_yaw_rate_deg_s']:.1f} deg/s | "
                     f"{s['max_speed_mps']:.2f} m/s | {s['max_yaw_rate_deg_s']:.0f} deg/s |")
    lines.append(f"\nmean residual speed **{m['final_speed_mean']:.4f} m/s** "
                 f"(envelope {m['config']['max_speed']} m/s), residual yaw rate "
                 f"**{m['final_yaw_rate_mean']:.1f} deg/s** "
                 f"(envelope {m['config']['max_yaw_rate_deg']} deg/s)")

    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_comparison(all_metrics):
    """results/README.md: cone vs circular for every shape, plus the verdict."""
    shapes = [s for s in PRESET_NAMES if any(m["shape"] == s for m in all_metrics)]
    by = {(m["shape"], m["perception"]): m for m in all_metrics}
    coll = 2 * all_metrics[0]["config"]["drone_radius"]
    n_seeds = all_metrics[0]["n_seeds"]
    steps = all_metrics[0]["eval_steps"]
    dt = all_metrics[0]["config"]["dt"]

    lines = [f"# Four identical drones: formation flight under two sensing models\n",
             f"Every drone runs the same gamma with the same weights and no identity, "
             f"no absolute position and no shape code. Motion is non-holonomic -- gamma "
             f"outputs a forward acceleration and a yaw acceleration, and a drone only "
             f"ever moves along its own heading. One network per formation.\n",
             f"Evaluation: {n_seeds} fresh random inits per run, "
             f"{steps} steps ({steps * dt:.0f} s) each.\n",
             "![final-state separation](collisions.png)\n",
             "## Formation error (aligned best assignment, lower = better)\n",
             "| shape | cone | circular | better |", "|---|---|---|---|"]
    for sh in shapes:
        c, o = by.get((sh, "cone")), by.get((sh, "circular"))
        if not (c and o):
            continue
        win = "cone" if c["final_error_mean"] < o["final_error_mean"] else "circular"
        lines.append(f"| {sh} | {c['final_error_mean']:.5f} | "
                     f"{o['final_error_mean']:.5f} | {win} |")

    lines += [f"\n## Collisions (collision = centers closer than {coll:.2f} m)\n",
              "| shape | perception | min sep any step | min sep FINAL | colliding steps "
              "| verdict |", "|---|---|---|---|---|---|"]
    any_final, any_all = False, False
    for sh in shapes:
        for p in PERCEPTIONS:
            m = by.get((sh, p))
            if not m:
                continue
            fin_bad = m["final_min_separation_worst"] < coll
            all_bad = m["collision_steps_total"] > 0
            any_final |= fin_bad
            any_all |= all_bad
            verdict = ("**COLLISION at final**" if fin_bad else
                       ("collision in transit" if all_bad else "clear"))
            lines.append(f"| {sh} | {p} | {m['min_separation_worst']:.3f} | "
                         f"{m['final_min_separation_worst']:.3f} | "
                         f"{m['collision_steps_total']} | {verdict} |")
    lines += ["", "**Final-state verdict: " +
              ("at least one run parks in a collision.**" if any_final else
               "no collisions in any run.**"),
              "**All-steps verdict: " +
              (f"collisions occur during transit.**" if any_all else
               "no colliding step in any run.**")]

    lines += ["\n## Equilibrium: does the swarm actually stop?\n",
              "There is no damping term in the loss and no drag in the physics, so a "
              "swarm at rest is something gamma learned, not something it was given.\n",
              "| shape | perception | residual speed (m/s) | residual yaw rate (deg/s) |",
              "|---|---|---|---|"]
    for sh in shapes:
        for p in PERCEPTIONS:
            m = by.get((sh, p))
            if m:
                lines.append(f"| {sh} | {p} | {m['final_speed_mean']:.4f} | "
                             f"{m['final_yaw_rate_mean']:.1f} |")

    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_collisions(all_metrics):
    shapes = [s for s in PRESET_NAMES if any(m["shape"] == s for m in all_metrics)]
    by = {(m["shape"], m["perception"]): m for m in all_metrics}
    coll = 2 * all_metrics[0]["config"]["drone_radius"]
    x = np.arange(len(shapes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, p in enumerate(PERCEPTIONS):
        vals = [by[(s, p)]["final_min_separation_worst"] if (s, p) in by else np.nan
                for s in shapes]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=p)
        for b, v in zip(bars, vals):
            if v == v and v < coll:
                b.set_color("tab:red")
                ax.annotate("COLLISION", (b.get_x() + b.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=7, color="tab:red",
                            rotation=90)
    ax.axhline(coll, color="tab:red", ls="--", label=f"collision distance {coll:.2f} m")
    ax.set_xticks(x); ax.set_xticklabels(shapes)
    ax.set_ylabel("worst final-state separation (m)")
    ax.set_title("Four drones: closest pair at the final state (higher = safer)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "collisions.png"), dpi=110)
    plt.close()


def main():
    cfg = load_config(extra_args=_args)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(RUNS_DIR, "gamma_*.pt")))
    if not paths:
        raise SystemExit(f"no checkpoints in {RUNS_DIR}/")

    all_metrics = []
    for path in paths:
        agg, model, sim_cfg, target, ckpt = evaluate_checkpoint(path, cfg)
        label = f"{agg['shape']}_{agg['perception']}"
        outdir = os.path.join(RESULTS_DIR, label)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(agg, f, indent=1)
        write_summary(agg, outdir)

        if ckpt.get("loss_curve"):
            plt.figure(figsize=(7, 4))
            plt.plot(ckpt["loss_curve"])
            plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss (log)")
            plt.title(f"training loss -- {label}")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=110)
            plt.close()

        if not cfg.no_animations:
            poses, thetas, _ = simulate(model, sim_cfg, cfg.eval_steps, seed=1)
            animate(poses, thetas, sim_cfg,
                    title=f"{agg['shape']} [{agg['perception']}]",
                    outfile=os.path.join(outdir, "animation.mp4"), target=target)
        print(f"[{label}] final error {agg['final_error_mean']:.5f}  "
              f"final min sep {agg['final_min_separation_worst']:.3f} m  "
              f"colliding steps {agg['collision_steps_total']}  "
              f"residual speed {agg['final_speed_mean']:.4f} m/s")
        all_metrics.append(agg)

    write_comparison(all_metrics)
    plot_collisions(all_metrics)
    print(f"\nwrote {RESULTS_DIR}/README.md and collisions.png")


if __name__ == "__main__":
    main()
