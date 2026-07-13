"""
make_results.py
===============
Generate a detailed, reproducible results folder for a trained checkpoint, and a
head-to-head comparison between perception modes.

Per checkpoint (one folder per perception mode under results/):

  results/<label>/
    summary.md        -- config, final training errors, hold + heading summaries
    drift_<shape>.csv -- 60s multi-seed drift data (distance-matrix error vs time)
    drift_tables.md   -- the same data as full markdown tables with per-seed stats
    loss_curve.png    -- training loss curve stored in the checkpoint
    metrics.json      -- everything machine-readable (consumed by --compare)

The drift protocol is the multi-seed check from issue #2: from a fresh random
init per seed, roll the swarm out for --steps steps (default 600 = 60s at
dt=0.1) and record the distance-matrix error every 2 simulated seconds, plus the
heading angular-speed distribution (the issue #3 metric) over the same rollouts.

Optionally (--animations) render the mode's animation set into
results/<label>/animations/: per-shape convergence gifs, the runtime shape-switch
demo, and 60s hold gifs for square (the issue #2 reference) and line (the
hardest shape for the distance-matrix loss -- bending a colinear formation is
nearly invisible to it).

Run:
  python make_results.py --checkpoint checkpoint.pt                 # -> results/cone/
  python make_results.py --checkpoint checkpoint_circular.pt        # -> results/circular/
  python make_results.py --checkpoint checkpoint.pt --animations    # gifs only
  python make_results.py --compare                                  # -> results/README.md + chart

The label defaults to the checkpoint's perception mode; --label overrides it.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from config import load_config, sim_config_from
from losses import distance_matrix_loss
from shapes import PRESET_NAMES, get_shape, pairwise_distance_matrix
from viz import load_model, simulate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SAMPLE_PERIOD_S = 2.0  # drift is sampled every 2 simulated seconds
# Config keys worth echoing into each summary (perception + training levers).
_SUMMARY_KEYS = [
    "perception", "half_angle_deg", "sensing_range", "self_rotation_deg",
    "heading_smoothing", "n", "dt", "hidden", "msg_dim", "epochs", "lr",
    "lr_min", "t_min", "t_max", "hold_tail", "damping_weight", "seed",
]


def _extra_args(parser):
    parser.add_argument("--label", default=None,
                        help="results subfolder name (default: checkpoint's perception)")
    parser.add_argument("--steps", type=int, default=600,
                        help="rollout length for the drift check (600 = 60s at dt=0.1)")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--compare", action="store_true",
                        help="build results/README.md from existing metrics.json files")
    parser.add_argument("--animations", action="store_true",
                        help="render the animation set into results/<label>/animations/")


def drift_and_heading(model, sim_cfg, shape_name, sid, n, shape_scale, steps, n_seeds):
    """
    The issue #2 drift protocol plus the issue #3 heading metric, per shape.
    Returns (times, {seed: [errors]}, heading_stats_dict).
    """
    target_dm = pairwise_distance_matrix(get_shape(shape_name, n, shape_scale))
    sample_every = max(1, round(SAMPLE_PERIOD_S / sim_cfg.dt))
    errors = {}
    times = []
    ang_all = []
    for seed in range(1, n_seeds + 1):
        poses, headings = simulate(model, sim_cfg, [(0, sid)], steps, seed=seed)
        poses = np.array(poses)
        errs, ts = [], []
        for s in range(sample_every, steps + 1, sample_every):
            errs.append(distance_matrix_loss(torch.from_numpy(poses[s]), target_dm).item())
            ts.append(s * sim_cfg.dt)
        errors[seed] = errs
        times = ts
        h = np.array(headings)
        angles = np.arctan2(h[..., 1], h[..., 0])
        dtheta = np.diff(angles, axis=0)
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        ang_all.append(np.degrees(np.abs(dtheta)) / sim_cfg.dt)
    ang = np.concatenate(ang_all)
    heading_stats = {
        "mean_deg_s": float(ang.mean()),
        "p90_deg_s": float(np.percentile(ang, 90)),
        "max_deg_s": float(ang.max()),
        "frac_over_180_deg_s": float((ang > 180).mean()),
    }
    return times, errors, heading_stats


def _err_at(times, errs, t):
    """Error at the sample closest to time t (seconds)."""
    idx = int(np.argmin(np.abs(np.array(times) - t)))
    return errs[idx]


def write_mode_results(cfg):
    model, ckpt = load_model(cfg)
    saved = ckpt["cfg"]
    # Simulate with exactly the physics/perception the checkpoint was trained on.
    sim_cfg = sim_config_from(SimpleNamespace(**saved))
    label = cfg.label or saved.get("perception", "cone")
    outdir = os.path.join(RESULTS_DIR, label)
    os.makedirs(outdir, exist_ok=True)

    n, shape_scale = saved["n"], saved["shape_scale"]
    metrics = {
        "label": label,
        "checkpoint": os.path.basename(cfg.checkpoint),
        "config": {k: saved.get(k) for k in _SUMMARY_KEYS},
        "drift_steps": cfg.steps,
        "n_seeds": cfg.n_seeds,
        "final_training_errors": ckpt.get("final_errors", {}),
        "drift": {},
        "heading": {},
    }

    tables_md = [f"# 60s drift tables -- {label}\n",
                 f"Distance-matrix error vs simulated time; {cfg.n_seeds} seeds, "
                 f"{cfg.steps} steps at dt={sim_cfg.dt} (protocol from issue #2).\n"]

    for sid, name in enumerate(ckpt["preset_names"]):
        print(f"[{label}] drift check: {name} ...")
        times, errors, heading = drift_and_heading(
            model, sim_cfg, name, sid, n, shape_scale, cfg.steps, cfg.n_seeds)
        metrics["drift"][name] = {"times_s": times,
                                  "errors": {str(s): e for s, e in errors.items()}}
        metrics["heading"][name] = heading

        with open(os.path.join(outdir, f"drift_{name}.csv"), "w") as f:
            f.write("t_s," + ",".join(f"seed{s}" for s in errors) + "\n")
            for i, t in enumerate(times):
                f.write(f"{t:g}," + ",".join(f"{errors[s][i]:.6f}" for s in errors) + "\n")

        tables_md.append(f"\n## {name}\n")
        tables_md.append("| t (s) |" + "".join(f" seed {s} |" for s in errors))
        tables_md.append("|---|" + "---|" * len(errors))
        for i, t in enumerate(times):
            tables_md.append(f"| {t:g} |" + "".join(f" {errors[s][i]:.4f} |" for s in errors))
        tables_md.append("\nPer-seed: " + "; ".join(
            f"seed {s}: min {min(e):.4f}, final {e[-1]:.4f}" for s, e in errors.items()))

    with open(os.path.join(outdir, "drift_tables.md"), "w") as f:
        f.write("\n".join(tables_md) + "\n")

    if ckpt.get("loss_curve"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.plot(ckpt["loss_curve"])
        plt.yscale("log")
        plt.xlabel("epoch"); plt.ylabel("loss (log)")
        plt.title(f"Training loss -- {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=110)
        plt.close()

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    _write_summary_md(outdir, metrics)
    print(f"[{label}] wrote {outdir}/")
    return metrics


def _write_summary_md(outdir, m):
    lines = [f"# Results -- {m['label']} perception\n",
             f"Checkpoint: `{m['checkpoint']}`\n", "## Config\n",
             "| key | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in m["config"].items()]

    lines += ["\n## Final training error (fresh inits, rolled t_max steps)\n",
              "| shape | distance-matrix error |", "|---|---|"]
    fte = m["final_training_errors"]
    lines += [f"| {s} | {e:.4f} |" for s, e in fte.items()]
    if fte:
        lines.append(f"| **mean** | **{sum(fte.values()) / len(fte):.4f}** |")

    lines += [f"\n## Formation hold over 60s ({m['n_seeds']} seeds, worst seed shown)\n",
              "| shape | err @10s | err @30s | err @60s | worst final/min |",
              "|---|---|---|---|---|"]
    for shape, d in m["drift"].items():
        times = d["times_s"]
        per_seed = list(d["errors"].values())
        w10 = max(_err_at(times, e, 10) for e in per_seed)
        w30 = max(_err_at(times, e, 30) for e in per_seed)
        w60 = max(_err_at(times, e, 60) for e in per_seed)
        ratio = max(e[-1] / min(e) for e in per_seed)
        lines.append(f"| {shape} | {w10:.4f} | {w30:.4f} | {w60:.4f} | {ratio:.1f}x |")

    lines += ["\n## Heading angular speed during the 60s holds (issue #3 metric)\n",
              "| shape | mean (deg/s) | p90 | max | steps >180 deg/s |",
              "|---|---|---|---|---|"]
    for shape, h in m["heading"].items():
        lines.append(f"| {shape} | {h['mean_deg_s']:.1f} | {h['p90_deg_s']:.1f} | "
                     f"{h['max_deg_s']:.0f} | {h['frac_over_180_deg_s'] * 100:.1f}% |")

    lines.append("\nFull per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.")
    lines.append("Animations (generate with `--animations`): `animations/` -- per-shape "
                 "convergence, runtime shape switching, and 60s holds for square + line.")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_animations(cfg):
    """
    Render the animation set for one checkpoint into results/<label>/animations/.
    Every gif is drawn with the perception model the checkpoint was trained with
    (cone wedges or circular edges), at fps = 1/dt so playback is real time.
    """
    from viz import animate

    model, ckpt = load_model(cfg)
    saved = ckpt["cfg"]
    sim_cfg = sim_config_from(SimpleNamespace(**saved))
    label = cfg.label or saved.get("perception", "cone")
    outdir = os.path.join(RESULTS_DIR, label, "animations")
    os.makedirs(outdir, exist_ok=True)
    names = ckpt["preset_names"]
    fps = round(1 / sim_cfg.dt)
    n, shape_scale = saved["n"], saved["shape_scale"]

    # Per-shape convergence from a fresh random init (seeds match viz.py).
    for sid, name in enumerate(names):
        poses, headings = simulate(model, sim_cfg, [(0, sid)], 70, seed=sid + 1)
        target = get_shape(name, n, shape_scale)
        target = target - target.mean(0) + torch.tensor(poses[-1].mean(0))
        animate(poses, headings, sim_cfg,
                title=f"Convergence: {name}  [{label}]",
                outfile=os.path.join(outdir, f"convergence_{name}.gif"),
                fps=fps, draw_cone=True, target_pts=target.numpy())

    # Runtime shape switching (the key demo): square -> hexagon without reset.
    switch_step = 30
    schedule = [(0, names.index("square")), (switch_step, names.index("hexagon"))]
    poses, headings = simulate(model, sim_cfg, schedule, 70, seed=42)
    animate(poses, headings, sim_cfg,
            title=f"Dynamic switch: square -> hexagon @ step {switch_step}  [{label}]",
            outfile=os.path.join(outdir, "switching.gif"),
            fps=fps, draw_cone=True, switch_step=switch_step,
            switch_label=lambda f: "hexagon" if f >= switch_step else "square")

    # 60s holds (issue #2 protocol, seed 1): square as the reference case, line as
    # the shape most prone to slow late drift. 600 frames at full fps makes a
    # ~15MB gif; every 2nd frame at fps/2 keeps playback real-time at half the
    # size, which is plenty for a hold demo where the swarm barely moves.
    for name in ("square", "line"):
        sid = names.index(name)
        poses, headings = simulate(model, sim_cfg, [(0, sid)], cfg.steps, seed=1)
        animate(poses[::2], headings[::2], sim_cfg,
                title=f"{cfg.steps * sim_cfg.dt:.0f}s hold: {name}  [{label}]",
                outfile=os.path.join(outdir, f"hold_60s_{name}.gif"),
                fps=max(1, fps // 2), draw_cone=True,
                target_pts=get_shape(name, n, shape_scale).numpy())
    print(f"[{label}] wrote {outdir}/")


def write_comparison():
    """Build results/README.md (+ chart) from every results/*/metrics.json present."""
    modes = {}
    for label in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, label, "metrics.json")
        if os.path.isfile(path):
            with open(path) as f:
                modes[label] = json.load(f)
    if len(modes) < 2:
        raise SystemExit(f"need at least 2 results/<label>/metrics.json, found {len(modes)}")

    labels = list(modes)
    shapes = list(next(iter(modes.values()))["drift"])
    lines = ["# Perception comparison: forward-FOV cone vs circular (360-degree) FOV\n",
             "Both models share every hyperparameter (same seed, epochs, horizons, "
             "sensing range, N) -- the only difference is what each agent may see: "
             "a forward cone tied to its heading, or an omnidirectional disk. "
             "Generated by `make_results.py`; per-mode details live in " +
             ", ".join(f"[`{l}/`]({l}/summary.md)" for l in labels) + ".\n"]

    cfg_rows = [("perception", *[modes[l]["config"]["perception"] for l in labels]),
                ("half_angle_deg", *[modes[l]["config"]["half_angle_deg"] for l in labels]),
                ("sensing_range", *[modes[l]["config"]["sensing_range"] for l in labels]),
                ("epochs / t_min / t_max / hold_tail",
                 *[f"{modes[l]['config']['epochs']} / {modes[l]['config']['t_min']} / "
                   f"{modes[l]['config']['t_max']} / {modes[l]['config']['hold_tail']}"
                   for l in labels])]
    lines += ["## Setup\n", "| | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in cfg_rows]

    lines += ["\n## Formation quality: final training error (fresh inits, lower = better)\n",
              "| shape | " + " | ".join(labels) + " | winner |",
              "|---|" + "---|" * (len(labels) + 1)]
    for shape in shapes:
        vals = [modes[l]["final_training_errors"].get(shape) for l in labels]
        win = labels[int(np.argmin(vals))]
        lines.append(f"| {shape} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {win} |")
    means = [np.mean(list(modes[l]["final_training_errors"].values())) for l in labels]
    lines.append("| **mean** | " + " | ".join(f"**{v:.4f}**" for v in means) +
                 f" | **{labels[int(np.argmin(means))]}** |")

    for t_probe in (10, 30, 60):
        lines += [f"\n## Formation hold: worst-seed error at {t_probe}s\n",
                  "| shape | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        for shape in shapes:
            row = []
            for l in labels:
                d = modes[l]["drift"][shape]
                row.append(max(_err_at(d["times_s"], e, t_probe) for e in d["errors"].values()))
            lines.append(f"| {shape} | " + " | ".join(f"{v:.4f}" for v in row) + " |")

    lines += ["\n## Heading angular speed during the 60s holds (mean deg/s, issue #3 metric)\n",
              "| shape | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for shape in shapes:
        row = [modes[l]["heading"][shape]["mean_deg_s"] for l in labels]
        lines.append(f"| {shape} | " + " | ".join(f"{v:.1f}" for v in row) + " |")

    with open(os.path.join(RESULTS_DIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(shapes))
    w = 0.8 / len(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for panel, (ax, title) in enumerate(zip(
            axes, ["final training error (fresh inits)", "worst-seed error @60s hold"])):
        for i, l in enumerate(labels):
            if panel == 0:
                vals = [modes[l]["final_training_errors"].get(s, 0) for s in shapes]
            else:
                vals = [max(_err_at(modes[l]["drift"][s]["times_s"], e, 60)
                            for e in modes[l]["drift"][s]["errors"].values()) for s in shapes]
            ax.bar(x + (i - (len(labels) - 1) / 2) * w, vals, w, label=l)
        ax.set_xticks(x); ax.set_xticklabels(shapes)
        ax.set_ylabel("distance-matrix error (lower = better)")
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "compare_modes.png"), dpi=110)
    plt.close()
    print(f"wrote {RESULTS_DIR}/README.md and compare_modes.png")


def main():
    cfg = load_config(extra_args=_extra_args)
    if cfg.compare:
        write_comparison()
    elif cfg.animations:
        write_animations(cfg)
    else:
        write_mode_results(cfg)


if __name__ == "__main__":
    main()
