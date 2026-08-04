"""
shapes.py
=========
Target formations for a FOUR-drone swarm.

Every preset is a set of exactly 4 points, centered at the origin, expressed in
meters (1 simulation unit = 1 m). The training loss is permutation-invariant and
pose-invariant (Kabsch-aligned optimal assignment, see losses.py), so a preset is
a *point set*, not a labelled list: there is no "slot k belongs to drone k" here.
Drones are identical and interchangeable by construction.

Two consequences of that invariance shaped this file:

  1. NO DEGENERATE PAIRS. Under a pose-invariant loss a diamond is the same
     object as a square (rotate 45 degrees), so shipping both would mean training
     two networks on one target. The four presets below are pairwise distinct up
     to rotation, translation and reflection.

  2. N=4 NATIVE. The presets are defined directly for four drones rather than
     sampled off a perimeter designed for a larger swarm -- perimeter sampling at
     N=4 produced misleading sets (a "hexagon" that is really a rhombus, a
     "triangle" that is three vertices plus one edge midpoint).

Every preset also keeps its tightest pair well above the 0.2 m collision distance
(2 x drone_radius), so a perfectly formed shape is never itself a collision --
`min_slot_spacing()` reports the margin and `__main__` asserts it.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch


def square(scale: float = 1.0) -> torch.Tensor:
    """Four corners of a square, side 2*scale."""
    s = scale
    return _recenter(torch.tensor(
        [(-s, -s), (s, -s), (s, s), (-s, s)], dtype=torch.float32))


def line(scale: float = 1.0) -> torch.Tensor:
    """Four collinear points, evenly spaced, total span 3*scale."""
    xs = torch.linspace(-1.5 * scale, 1.5 * scale, 4)
    return _recenter(torch.stack([xs, torch.zeros(4)], dim=1))


def wedge(scale: float = 1.0, half_angle_deg: float = 35.0) -> torch.Tensor:
    """
    A V / arrowhead: one leader at the apex and three followers trailing along
    the two arms. With four drones the V is necessarily ASYMMETRIC (two on one
    arm, one on the other) -- the same way a real skein of geese flies -- which
    also makes it chiral, and therefore distinct from every other preset here
    even under a reflection-tolerant loss.
    """
    a = math.radians(half_angle_deg)
    d = 0.9 * scale                                  # rank spacing along an arm
    left = (-math.sin(a), -math.cos(a))
    right = (math.sin(a), -math.cos(a))
    pts = [
        (0.0, 0.0),                                  # leader
        (left[0] * d, left[1] * d),                  # left arm, rank 1
        (right[0] * d, right[1] * d),                # right arm, rank 1
        (left[0] * 2 * d, left[1] * 2 * d),          # left arm, rank 2
    ]
    return _recenter(torch.tensor(pts, dtype=torch.float32))


def triangle_centroid(scale: float = 1.0) -> torch.Tensor:
    """
    Equilateral triangle (circumradius `scale`) with the fourth drone at the
    centroid. The interior point is what makes this a genuine 4-drone shape
    rather than a triangle with a spare drone parked on an edge.
    """
    verts = [(scale * math.cos(math.radians(deg)), scale * math.sin(math.radians(deg)))
             for deg in (90.0, 210.0, 330.0)]
    return _recenter(torch.tensor(verts + [(0.0, 0.0)], dtype=torch.float32))


def _recenter(pts: torch.Tensor) -> torch.Tensor:
    return pts - pts.mean(dim=0, keepdim=True)


_GENERATORS = {
    "square": square,
    "line": line,
    "wedge": wedge,
    "triangle_centroid": triangle_centroid,
}

# Canonical ordering. One network is trained per name (there is no shape code in
# the model -- see model.py), so this list is just the sweep's task list.
PRESET_NAMES: List[str] = ["square", "line", "wedge", "triangle_centroid"]

N_DRONES = 4


def get_shape(name: str, scale: float = 1.0) -> torch.Tensor:
    """Return the (4, 2) target point set for a named preset, in meters."""
    if name not in _GENERATORS:
        raise KeyError(f"Unknown shape '{name}'. Available: {list(_GENERATORS)}")
    pts = _GENERATORS[name](scale)
    assert pts.shape == (N_DRONES, 2), f"{name} must define exactly {N_DRONES} points"
    return pts


def pairwise_distance_matrix(points: torch.Tensor) -> torch.Tensor:
    """(..., N, 2) points -> (..., N, N) Euclidean inter-point distances."""
    diff = points.unsqueeze(-2) - points.unsqueeze(-3)
    return torch.linalg.norm(diff, dim=-1)


def min_slot_spacing(name: str, scale: float = 1.0) -> float:
    """Tightest pair in a perfectly formed preset -- the collision headroom."""
    d = pairwise_distance_matrix(get_shape(name, scale))
    n = d.shape[0]
    d = d + torch.eye(n) * 1e9
    return float(d.min())


def all_targets(scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """Every preset's point set, keyed by name."""
    return {name: get_shape(name, scale) for name in PRESET_NAMES}


if __name__ == "__main__":
    # Sanity check: distinct shapes, all collision-free by a wide margin.
    collision_dist = 0.2
    scale = 0.8
    print(f"scale={scale} m, collision distance={collision_dist} m\n")
    for name in PRESET_NAMES:
        pts = get_shape(name, scale)
        gap = min_slot_spacing(name, scale)
        print(f"{name:18s} min slot spacing {gap:.3f} m "
              f"({gap / collision_dist:.1f}x the collision distance)")
        print("   " + "  ".join(f"({x:+.2f},{y:+.2f})" for x, y in pts.tolist()))
        assert gap > collision_dist, f"{name} target is itself a collision"

    # Pairwise distinctness up to pose: compare sorted pairwise-distance spectra.
    import itertools
    for a, b in itertools.combinations(PRESET_NAMES, 2):
        sa = torch.sort(pairwise_distance_matrix(get_shape(a, scale)).flatten()).values
        sb = torch.sort(pairwise_distance_matrix(get_shape(b, scale)).flatten()).values
        assert not torch.allclose(sa, sb, atol=1e-4), f"{a} and {b} are the same shape"
    print("\nall presets distinct up to rotation/translation; self-check passed")
