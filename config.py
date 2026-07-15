"""
config.py
=========
Tiny config loader: read config.yaml, then let any field be overridden from the
command line (argparse). Returns a plain namespace plus a SimConfig.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import yaml

from sim import SimConfig

_DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(argv=None, extra_args=None) -> SimpleNamespace:
    """
    Load config.yaml and apply CLI overrides. `extra_args` is an optional callable
    that adds parser arguments specific to a script (e.g. viz options).
    """
    with open(_DEFAULT_YAML, "r") as f:
        base = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="GNCA swarm formation config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=_DEFAULT_YAML,
                        help="path to an alternate yaml config")
    # Auto-expose every yaml key as an optional override with the right type.
    for key, val in base.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=None)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=None)
    if extra_args is not None:
        extra_args(parser)

    args = parser.parse_args(argv)

    # If an alternate config path was given, reload from it.
    if args.config != _DEFAULT_YAML:
        with open(args.config, "r") as f:
            base = yaml.safe_load(f)

    cfg = dict(base)
    for key in base:
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    # Carry through any extra (non-yaml) args too.
    for key, v in vars(args).items():
        if key not in cfg and key != "config":
            cfg[key] = v

    return SimpleNamespace(**cfg)


def sim_config_from(cfg: SimpleNamespace) -> SimConfig:
    """
    Extract the simulation-physics subset into a SimConfig dataclass.

    Keys added after a checkpoint was trained may be missing from its saved cfg
    (e.g. the issue #4 arena/drone keys on pre-drone checkpoints), so every
    lookup falls back to the SimConfig default -- old checkpoints keep their
    original unbounded, no-wall physics.
    """
    defaults = SimConfig()
    def get(key):
        return getattr(cfg, key, getattr(defaults, key))
    return SimConfig(
        n=get("n"),
        dt=get("dt"),
        drag=get("drag"),
        perception=get("perception"),
        half_angle_deg=get("half_angle_deg"),
        sensing_range=get("sensing_range"),
        init_box=get("init_box"),
        init_vel_std=get("init_vel_std"),
        speed_eps=get("speed_eps"),
        heading_smoothing=get("heading_smoothing"),
        self_rotation_deg=get("self_rotation_deg"),
        arena_mode=get("arena_mode"),
        arena_half=get("arena_half"),
        wall_margin=get("wall_margin"),
        wall_strength=get("wall_strength"),
        drone_radius=get("drone_radius"),
        min_start_dist=get("min_start_dist"),
        max_speed=get("max_speed"),
        max_accel=get("max_accel"),
        safety_filter=get("safety_filter"),
    )
