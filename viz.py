"""
viz.py
======
Produce two matplotlib animations from a trained checkpoint (saved as .mp4;
the writer is chosen from the output extension, so .gif paths still work):

  1. convergence_<shape>_<tag>.mp4 -- from a random init, the swarm forms each
                           preset.
  2. switching_<tag>.mp4 -- THE KEY DEMO: start forming a square, then at step
                           --switch_step swap z_shape to hexagon WITHOUT resetting
                           positions, and watch it re-converge.

Each agent is drawn as a dot with its heading arrow and (faintly) its perception
cone, so the forward field-of-view model is visible.

Run:  python viz.py                 (reads checkpoint.pt, writes the videos)
      python viz.py --switch_from square --switch_to hexagon --switch_step 30
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle, Wedge

from config import load_config, sim_config_from
from graph import circular_adjacency
from model import GammaGNCA
from shapes import PRESET_NAMES, get_shape
from sim import random_init, step


def _viz_args(parser):
    parser.add_argument("--switch_from", default="square")
    parser.add_argument("--switch_to", default="hexagon")
    parser.add_argument("--switch_step", type=int, default=30)
    parser.add_argument("--total_steps", type=int, default=70)
    # Keep fps == 1/dt so playback runs in real simulated time; otherwise motion
    # looks artificially slowed/sped up. Update this if config.yaml's dt changes.
    parser.add_argument("--fps", type=int, default=10)
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
        # Fixed-arena checkpoints carry an extra absolute-position input
        # (model.py note 4); pre-drone checkpoints predate the key -> off.
        absolute_pos=(saved.get("arena_mode", "none") == "fixed"),
        aggregation=saved.get("aggregation", "mean"),
        softmax_temp=saved.get("softmax_temp", 0.3),
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
    pos, vel, heading, smoothed_vel = random_init(sim_cfg, generator=gen)

    poses, headings = [pos.numpy().copy()], [heading.numpy().copy()]
    with torch.no_grad():
        for s in range(total_steps):
            sid = shape_schedule[0][1]
            for start, sched_id in shape_schedule:
                if s >= start:
                    sid = sched_id
            z = model.get_z(sid)
            pos, vel, heading, smoothed_vel = step(
                model, pos, vel, heading, smoothed_vel, z, sim_cfg
            )
            poses.append(pos.numpy().copy())
            headings.append(heading.numpy().copy())
    return poses, headings


def animate(poses, headings, sim_cfg, title, outfile, fps, draw_cone,
            target_pts=None, switch_step=None, switch_label=None):
    """Render a trajectory animation; the writer is picked from the extension
    (.mp4 -> H.264 via ffmpeg, anything else -> Pillow gif)."""
    bounded = sim_cfg.arena_mode != "none"
    allp = np.concatenate(poses, axis=0)
    pad = 0.6
    xlim = (allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ylim = (allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    if bounded:
        # Always show the whole flight area (plus anything that escaped it).
        wall_pad = 0.2
        xlim = (min(xlim[0], -sim_cfg.arena_half - wall_pad),
                max(xlim[1], sim_cfg.arena_half + wall_pad))
        ylim = (min(ylim[0], -sim_cfg.arena_half - wall_pad),
                max(ylim[1], sim_cfg.arena_half + wall_pad))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title)

    n = poses[0].shape[0]
    p0, h0 = poses[0], headings[0]
    if bounded:
        # The 3m x 3m flight area (issue #4): solid walls, plus the wall-force
        # onset band in "walls" mode.
        ax.add_patch(Rectangle((-sim_cfg.arena_half, -sim_cfg.arena_half),
                               2 * sim_cfg.arena_half, 2 * sim_cfg.arena_half,
                               fill=False, edgecolor="black", linewidth=1.5, zorder=1))
        if sim_cfg.arena_mode == "walls":
            inner = sim_cfg.arena_half - sim_cfg.wall_margin
            ax.add_patch(Rectangle((-inner, -inner), 2 * inner, 2 * inner,
                                   fill=False, edgecolor="black", linewidth=0.8,
                                   linestyle=":", alpha=0.5, zorder=1))
    # Per-drone safety disks (issue #4): radius = drone_radius; a disk turns red
    # whenever its drone is in COLLISION (another center closer than 2*radius).
    safety_circles = []
    if sim_cfg.drone_radius > 0:
        for k in range(n):
            c = Circle((p0[k, 0], p0[k, 1]), sim_cfg.drone_radius,
                       facecolor="tab:blue", alpha=0.25, edgecolor="none", zorder=2)
            ax.add_patch(c)
            safety_circles.append(c)
    scat = ax.scatter(p0[:, 0], p0[:, 1], s=80, c="tab:blue", zorder=3)
    quiv = ax.quiver(p0[:, 0], p0[:, 1], h0[:, 0], h0[:, 1],
                     color="tab:red", scale=12, width=0.005, zorder=4)
    # Draw whichever perception model was used: cone wedges, or circular edges.
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
    elif draw_cone and perception == "circular":
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
        if safety_circles:
            dist = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
            np.fill_diagonal(dist, np.inf)
            colliding = dist.min(axis=1) < 2 * sim_cfg.drone_radius
            for k, c in enumerate(safety_circles):
                c.center = (p[k, 0], p[k, 1])
                c.set_facecolor("tab:red" if colliding[k] else "tab:blue")
                c.set_alpha(0.5 if colliding[k] else 0.25)
        if draw_cone and perception == "cone":
            for k, w in enumerate(cone_patches):
                ang = math.degrees(math.atan2(h[k, 1], h[k, 0]))
                w.set_center((p[k, 0], p[k, 1]))
                w.set_theta1(ang - half_angle)
                w.set_theta2(ang + half_angle)
        elif draw_cone and perception == "circular":
            # Recompute the circular (360-degree) graph each frame and draw it.
            adj = circular_adjacency(torch.from_numpy(p), sim_cfg.sensing_range)
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
    if outfile.endswith(".mp4"):
        # H.264 via the ffmpeg binary bundled with imageio-ffmpeg (no system
        # install needed) -- far smaller files than gif at the same quality.
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(outfile, writer=writer)
    plt.close(fig)
    print(f"  wrote {outfile}")


def main():
    cfg = load_config(extra_args=_viz_args)
    sim_cfg = sim_config_from(cfg)
    model, ckpt = load_model(cfg)
    names = ckpt["preset_names"]

    # Visualize with the SAME perception and arena physics the checkpoint was
    # trained with, so the animation (and the drawn graph/walls) matches the
    # dynamics the model learned.
    saved = ckpt["cfg"]
    sim_cfg.perception = saved.get("perception", sim_cfg.perception)
    for key in ("arena_mode", "arena_half", "wall_margin", "wall_strength",
                "drone_radius", "min_start_dist", "max_speed", "max_accel",
                "safety_filter", "hard_bounds", "max_turn_deg"):
        setattr(sim_cfg, key, saved.get(key, getattr(sim_cfg, key)))
    print(f"Perception model: {sim_cfg.perception}, arena: {sim_cfg.arena_mode}")
    tag = sim_cfg.perception  # filename/title suffix so cone vs circular don't clobber
    if sim_cfg.arena_mode != "none":
        tag = f"{sim_cfg.perception}_{sim_cfg.arena_mode}"

    # --- 1. Convergence: one video per shape, forming from scratch. ---
    print("Rendering convergence animations...")
    for sid, name in enumerate(names):
        poses, headings = simulate(
            model, sim_cfg, [(0, sid)], cfg.total_steps, seed=sid + 1
        )
        target = get_shape(name, cfg.n, cfg.shape_scale)
        if sim_cfg.arena_mode != "fixed":
            # Center target on the swarm's final centroid for visual comparison.
            # (In "fixed" mode the target's true, arena-centered spot IS the goal.)
            target = target - target.mean(0) + torch.tensor(poses[-1].mean(0))
        animate(
            poses, headings, sim_cfg,
            title=f"Convergence: {name}  [{tag}]",
            outfile=f"convergence_{name}_{tag}.mp4",
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
        outfile=f"switching_{tag}.mp4",
        fps=cfg.fps, draw_cone=cfg.draw_cone,
        switch_step=cfg.switch_step, switch_label=switch_label,
    )

    print(f"\nDone. Key demo: switching_{tag}.mp4")


if __name__ == "__main__":
    main()
