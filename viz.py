"""
viz.py
======
Animate a trained gamma.

What is drawn, and why each element is there:

  * each drone as a dot with its HEADING ARROW. Heading is a real state variable
    now -- the drone flies where the arrow points, always -- so the arrow is the
    single most informative thing on screen: turning is the only way this swarm
    can change direction.
  * its SAFETY DISK (radius = drone_radius). Two disks touching is exactly the
    collision condition, and any drone inside another's collision distance turns
    RED, with the offending pair named in the corner. A collision cannot be
    missed, including on a paused final frame.
  * its PERCEPTION: cone wedges for the forward FOV, or neighbour segments for
    circular sensing, recomputed every frame from the drawn state.
  * the TARGET, dashed, mapped into the swarm's own frame by the same
    best-assignment alignment the loss uses (losses.align_target_to). The loss is
    pose- and permutation-invariant, so there is no "correct" place to draw the
    target a priori -- drawing it where the loss actually grades it is the only
    honest option.

Run:  python viz.py --checkpoint runs/gamma_wedge_cone.pt
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
from matplotlib.patches import Circle, Wedge

from config import load_config, sim_config_from
from graph import circular_adjacency
from losses import align_target_to
from model import GammaSwarm
from shapes import get_shape
from sim import random_init, step


def _viz_args(parser):
    parser.add_argument("--steps", type=int, default=300, help="rollout length")
    # NB: --seed is already exposed as a config key (the training seed);
    # the rollout seed is separate so a demo can be re-drawn without
    # implying the model was trained differently.
    parser.add_argument("--viz_seed", type=int, default=1)
    parser.add_argument("--outfile", default=None)
    parser.add_argument("--still", action="store_true",
                        help="write one PNG of the whole flight instead of an animation")
    parser.add_argument("--draw_perception",
                        type=lambda s: s.lower() in ("1", "true", "yes"), default=True)


def load_model(checkpoint: str):
    """Rebuild gamma from a checkpoint, with the config it was trained under."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved = ckpt["cfg"]
    model = GammaSwarm(hidden=saved["hidden"], msg_dim=saved["msg_dim"],
                       a_lin_scale=saved["a_lin_scale"],
                       a_ang_scale=saved["a_ang_scale"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def simulate(model, sim_cfg, steps: int, seed: int = 1):
    """Roll out from a fresh random init. Returns (positions, headings, speeds)."""
    gen = torch.Generator().manual_seed(seed)
    pos, theta, s, omega = random_init(sim_cfg, generator=gen)
    poses, thetas, speeds = [pos.numpy().copy()], [theta.numpy().copy()], [s.numpy().copy()]
    with torch.no_grad():
        for _ in range(steps):
            pos, theta, s, omega = step(model, pos, theta, s, omega, sim_cfg)
            poses.append(pos.numpy().copy())
            thetas.append(theta.numpy().copy())
            speeds.append(s.numpy().copy())
    return poses, thetas, speeds


def animate(poses, thetas, sim_cfg, title, outfile, target=None, fps=None):
    """Render a rollout. Writer is chosen from the extension (.mp4 or .gif)."""
    fps = fps or round(1 / sim_cfg.dt)
    allp = np.concatenate(poses, axis=0)
    pad = 0.8
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)

    n = poses[0].shape[0]
    p0, th0 = poses[0], thetas[0]
    coll_dist = 2 * sim_cfg.drone_radius

    disks = []
    for k in range(n):
        c = Circle((p0[k, 0], p0[k, 1]), sim_cfg.drone_radius, facecolor="tab:blue",
                   alpha=0.25, edgecolor="none", zorder=2)
        ax.add_patch(c)
        disks.append(c)
    scat = ax.scatter(p0[:, 0], p0[:, 1], s=70, c="tab:blue", zorder=3)
    quiv = ax.quiver(p0[:, 0], p0[:, 1], np.cos(th0), np.sin(th0),
                     color="tab:red", scale=14, width=0.005, zorder=4)

    cone_patches, edge_lc = [], None
    if sim_cfg.perception == "cone":
        for _ in range(n):
            w = Wedge((0, 0), sim_cfg.sensing_range, 0, 0, alpha=0.07,
                      color="tab:green", zorder=1)
            ax.add_patch(w)
            cone_patches.append(w)
    else:
        edge_lc = LineCollection([], colors="tab:green", alpha=0.3, linewidths=0.8,
                                 zorder=1)
        ax.add_collection(edge_lc)

    target_scat = None
    if target is not None:
        target_scat = ax.scatter([], [], s=130, facecolors="none", edgecolors="gray",
                                 linestyle="--", zorder=2)

    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)
    coll_txt = ax.text(0.02, 0.03, "", transform=ax.transAxes, va="bottom",
                       fontsize=10, color="tab:red", fontweight="bold")

    def update(frame):
        p, th = poses[frame], thetas[frame]
        scat.set_offsets(p)
        quiv.set_offsets(p)
        quiv.set_UVC(np.cos(th), np.sin(th))

        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        hit = d.min(axis=1) < coll_dist
        for k, c in enumerate(disks):
            c.center = (p[k, 0], p[k, 1])
            c.set_facecolor("tab:red" if hit[k] else "tab:blue")
            c.set_alpha(0.5 if hit[k] else 0.25)
        scat.set_color(np.where(hit, "tab:red", "tab:blue"))
        if hit.any():
            iu = np.triu_indices(len(p), k=1)
            w = int(np.argmin(d[iu]))
            i, j = int(iu[0][w]), int(iu[1][w])
            coll_txt.set_text(f"!! COLLISION  {i}-{j} at {d[i, j]:.3f} m "
                              f"(< {coll_dist:.2f} m)")
        else:
            coll_txt.set_text("")

        if cone_patches:
            for k, w in enumerate(cone_patches):
                ang = math.degrees(th[k])
                w.set_center((p[k, 0], p[k, 1]))
                w.set_theta1(ang - sim_cfg.half_angle_deg)
                w.set_theta2(ang + sim_cfg.half_angle_deg)
        elif edge_lc is not None:
            adj = circular_adjacency(torch.from_numpy(p), sim_cfg.sensing_range)
            recv, send = torch.nonzero(adj, as_tuple=True)
            edge_lc.set_segments([[p[i], p[j]] for i, j in
                                  zip(recv.tolist(), send.tolist())])
        if target_scat is not None:
            # Re-align every frame: the target has no canonical pose, so it is
            # drawn where the loss currently grades this configuration.
            aligned = align_target_to(torch.from_numpy(p), target).numpy()
            target_scat.set_offsets(aligned)

        txt.set_text(f"step {frame}   t = {frame * sim_cfg.dt:.1f} s")
        return scat, quiv, txt, coll_txt

    anim = FuncAnimation(fig, update, frames=len(poses), interval=1000 / fps, blit=False)
    if outfile.endswith(".mp4"):
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)
    anim.save(outfile, writer=writer)
    plt.close(fig)
    print(f"  wrote {outfile}")


def still(poses, thetas, sim_cfg, title, outfile, target=None):
    """One PNG summarising a whole flight: trails, final state, aligned target.
    More practical than a video on a terminal-only machine."""
    P = np.array(poses)
    n = P.shape[1]
    coll = 2 * sim_cfg.drone_radius
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect("equal")
    colors = plt.cm.tab10(np.arange(n) % 10)
    for k in range(n):
        ax.plot(P[:, k, 0], P[:, k, 1], color=colors[k], alpha=0.35, lw=1.2)
        ax.scatter(P[0, k, 0], P[0, k, 1], color=colors[k], s=25, alpha=0.5)
    last, th = P[-1], thetas[-1]
    d = np.linalg.norm(last[:, None, :] - last[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    hit = d.min(axis=1) < coll
    for k in range(n):
        ax.add_patch(Circle((last[k, 0], last[k, 1]), sim_cfg.drone_radius,
                            facecolor="tab:red" if hit[k] else colors[k],
                            alpha=0.45, edgecolor="none"))
    ax.scatter(last[:, 0], last[:, 1], c=np.where(hit, "tab:red", "black"), s=45, zorder=4)
    ax.quiver(last[:, 0], last[:, 1], np.cos(th), np.sin(th), color="tab:red",
              scale=14, width=0.005, zorder=5)
    if target is not None:
        al = align_target_to(torch.from_numpy(last), target).numpy()
        ax.scatter(al[:, 0], al[:, 1], s=150, facecolors="none", edgecolors="gray",
                   linestyle="--", zorder=3)
    ax.set_title(f"{title}\nmin final separation {d.min():.3f} m "
                 f"({'COLLISION' if d.min() < coll else 'clear'}, threshold {coll:.2f} m)",
                 fontsize=10, color="tab:red" if d.min() < coll else "black")
    plt.tight_layout(); plt.savefig(outfile, dpi=110); plt.close(fig)
    print(f"  wrote {outfile}")


def main():
    cfg = load_config(extra_args=_viz_args)
    model, ckpt = load_model(cfg.checkpoint)
    saved = ckpt["cfg"]
    from types import SimpleNamespace
    sim_cfg = sim_config_from(SimpleNamespace(**saved))   # animate the trained physics
    shape = ckpt.get("shape", saved.get("shape"))
    target = get_shape(shape, saved["shape_scale"])

    poses, thetas, speeds = simulate(model, sim_cfg, cfg.steps, seed=cfg.viz_seed)
    if cfg.still:
        still(poses, thetas, sim_cfg,
              title=f"{shape} [{sim_cfg.perception}]  {cfg.steps} steps",
              outfile=cfg.outfile or f"{shape}_{sim_cfg.perception}.png", target=target)
        return
    out = cfg.outfile or f"{shape}_{sim_cfg.perception}.mp4"
    animate(poses, thetas, sim_cfg,
            title=f"{shape} [{sim_cfg.perception}]  seed {cfg.viz_seed}",
            outfile=out, target=target)


if __name__ == "__main__":
    main()
