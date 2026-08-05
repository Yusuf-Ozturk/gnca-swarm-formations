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
    Extract the flight-physics subset into a SimConfig dataclass. Any key missing
    from a saved checkpoint config falls back to the SimConfig default.
    """
    defaults = SimConfig()

    def get(key):
        return getattr(cfg, key, getattr(defaults, key))

    return SimConfig(
        n=get("n"),
        dt=get("dt"),
        perception=get("perception"),
        half_angle_deg=get("half_angle_deg"),
        sensing_range=get("sensing_range"),
        max_speed=get("max_speed"),
        max_accel=get("max_accel"),
        max_yaw_rate_deg=get("max_yaw_rate_deg"),
        max_yaw_accel_deg=get("max_yaw_accel_deg"),
        init_box=get("init_box"),
        min_start_dist=get("min_start_dist"),
        init_speed_min=get("init_speed_min"),
        init_speed_max=get("init_speed_max"),
        drone_radius=get("drone_radius"),
    )
