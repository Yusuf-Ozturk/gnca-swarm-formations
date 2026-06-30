"""
viz.py
======
Produce two matplotlib animations from a trained checkpoint:

  1. convergence.gif    -- from a random init, the swarm forms each preset.
  2. switching.gif      -- THE KEY DEMO: start forming a square, then at step
                           --switch_step swap z_shape to hexagon WITHOUT resetting
                           positions, and watch it re-converge.

Each agent is drawn as a dot with its heading arrow and (faintly) its perception
cone, so the forward field-of-view model is visible.

Run:  python viz.py                 (reads checkpoint.pt, writes both gifs)
      python viz.py --switch_from square --switch_to hexagon --switch_step 30
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Wedge

from config import load_config, sim_config_from
from graph import knn_adjacency
from model import GammaGNCA
from shapes import PRESET_NAMES, get_shape
from sim import random_init, step


def _viz_args(parser):
    parser.add_argument("--switch_from", default="square")
    parser.add_argument("--switch_to", default="hexagon")
    parser.add_argument("--switch_step", type=int, default=30)
    parser.add_argument("--total_steps", type=int, default=70)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--draw_cone", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=True)


def load_model(cfg):
    ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["cfg"]
    model = GammaGNCA(
        n_agents=saved["n"],
        hidden=saved["hidden"],
        msg_dim=saved["msg_dim"],
        z_dim=saved["z_dim"],
        id_dim=saved["id_dim"],
        n_shapes=len(ckpt["preset_names"]),
        accel_scale=saved["accel_scale"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def simulate(model, sim_cfg, shape_schedule, total_steps, seed=0):
    """
    Roll out, optionally switching the active shape id partway through.

    shape_schedule: list of (start_step, shape_id). The active shape is the last
    entry whose start_step <= current step.
    Returns lists of positions and headings per step (numpy arrays).
    """
    gen = torch.Generator().manual_seed(seed)
    pos, vel, heading = random_init(sim_cfg, generator=gen)

    poses, headings = [pos.numpy().copy()], [heading.numpy().copy()]
    with torch.no_grad():
        for s in range(total_steps):
            sid = shape_schedule[0][1]
            for start, sched_id in shape_schedule:
                if s >= start:
                    sid = sched_id
            z = model.get_z(sid)
            pos, vel, heading = step(model, pos, vel, heading, z, sim_cfg)
            poses.append(pos.numpy().copy())
            headings.append(heading.numpy().copy())
    return poses, headings


def animate(poses, headings, sim_cfg, title, outfile, fps, draw_cone,
            target_pts=None, switch_step=None, switch_label=None):
    """Render an animation of a trajectory to a gif."""
    allp = np.concatenate(poses, axis=0)
    pad = 0.6
    xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title)

    n = poses[0].shape[0]
    p0, h0 = poses[0], headings[0]
    scat = ax.scatter(p0[:, 0], p0[:, 1], s=80, c="tab:blue", zorder=3)
    quiv = ax.quiver(p0[:, 0], p0[:, 1], h0[:, 0], h0[:, 1],
                     color="tab:red", scale=12, width=0.005, zorder=4)
    # Draw whichever perception model was used: cone wedges, or kNN edges.
    perception = sim_cfg.perception
    cone_patches = []
    edge_lc = None
    half_angle = sim_cfg.half_angle_deg
    rng = sim_cfg.sensing_range
    if draw_cone and perception == "cone":
        for _ in range(n):
            w = Wedge((0, 0), rng, 0, 0, alpha=0.06, color="tab:green", zorder=1)
            ax.add_patch(w)
            cone_patches.append(w)
    elif draw_cone and perception == "knn":
        edge_lc = LineCollection([], colors="tab:green", alpha=0.3,
                                 linewidths=0.8, zorder=1)
        ax.add_collection(edge_lc)

    target_scat = None
    if target_pts is not None:
        target_scat = ax.scatter(
            target_pts[:, 0], target_pts[:, 1], s=120, facecolors="none",
            edgecolors="gray", linestyle="--", zorder=2, label="target slot",
        )

    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)

    def update(frame):
        p = poses[frame]
        h = headings[frame]
        scat.set_offsets(p)
        quiv.set_offsets(p)
        quiv.set_UVC(h[:, 0], h[:, 1])
        if draw_cone and perception == "cone":
            for k, w in enumerate(cone_patches):
                ang = math.degrees(math.atan2(h[k, 1], h[k, 0]))
                w.set_center((p[k, 0], p[k, 1]))
                w.set_theta1(ang - half_angle)
                w.set_theta2(ang + half_angle)
        elif draw_cone and perception == "knn":
            # Recompute the kNN graph each frame and draw it as edge segments.
            adj = knn_adjacency(torch.from_numpy(p), sim_cfg.knn_k)
            recv, send = torch.nonzero(adj, as_tuple=True)
            segs = [[p[i], p[j]] for i, j in zip(recv.tolist(), send.tolist())]
            edge_lc.set_segments(segs)
        label = f"step {frame}"
        if switch_step is not None and switch_label is not None:
            label += f"   shape: {switch_label(frame)}"
            if frame == switch_step:
                label += "  <-- SWITCHED"
        txt.set_text(label)
        return scat, quiv, txt

    anim = FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps, blit=False)
    anim.save(outfile, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  wrote {outfile}")


def main():
    cfg = load_config(extra_args=_viz_args)
    sim_cfg = sim_config_from(cfg)
    model, ckpt = load_model(cfg)
    names = ckpt["preset_names"]

    # Visualize with the SAME perception the checkpoint was trained with, so the
    # animation (and the drawn graph) matches the dynamics the model learned.
    saved = ckpt["cfg"]
    sim_cfg.perception = saved.get("perception", sim_cfg.perception)
    sim_cfg.knn_k = saved.get("knn_k", sim_cfg.knn_k)
    print(f"Perception model: {sim_cfg.perception}")
    tag = sim_cfg.perception  # filename/title suffix so cone vs knn don't clobber

    # --- 1. Convergence: one panel-per-shape gif each, forming from scratch. ---
    print("Rendering convergence animations...")
    for sid, name in enumerate(names):
        poses, headings = simulate(
            model, sim_cfg, [(0, sid)], cfg.total_steps, seed=sid + 1
        )
        target = get_shape(name, cfg.n, cfg.shape_scale)
        # Center target on the swarm's final centroid for visual comparison.
        target = target - target.mean(0) + torch.tensor(poses[-1].mean(0))
        animate(
            poses, headings, sim_cfg,
            title=f"Convergence: {name}  [{tag}]",
            outfile=f"convergence_{name}_{tag}.gif",
            fps=cfg.fps, draw_cone=cfg.draw_cone,
            target_pts=target.numpy(),
        )

    # --- 2. Dynamic switching: the key demo. ---
    print("Rendering dynamic-switch animation...")
    sid_from = names.index(cfg.switch_from)
    sid_to = names.index(cfg.switch_to)
    schedule = [(0, sid_from), (cfg.switch_step, sid_to)]
    poses, headings = simulate(model, sim_cfg, schedule, cfg.total_steps, seed=42)

    def switch_label(frame):
        return cfg.switch_to if frame >= cfg.switch_step else cfg.switch_from

    animate(
        poses, headings, sim_cfg,
        title=f"Dynamic switch: {cfg.switch_from} -> {cfg.switch_to} "
              f"@ step {cfg.switch_step}  [{tag}]",
        outfile=f"switching_{tag}.gif",
        fps=cfg.fps, draw_cone=cfg.draw_cone,
        switch_step=cfg.switch_step, switch_label=switch_label,
    )

    print(f"\nDone. Key demo: switching_{tag}.gif")


if __name__ == "__main__":
    main()
