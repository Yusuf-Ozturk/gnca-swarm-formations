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
    """Extract the simulation-physics subset into a SimConfig dataclass."""
    return SimConfig(
        n=cfg.n,
        dt=cfg.dt,
        drag=cfg.drag,
        perception=cfg.perception,
        half_angle_deg=cfg.half_angle_deg,
        sensing_range=cfg.sensing_range,
        init_box=cfg.init_box,
        init_vel_std=cfg.init_vel_std,
        speed_eps=cfg.speed_eps,
        max_turn_deg=cfg.max_turn_deg,
        max_accel=cfg.max_accel,
    )
