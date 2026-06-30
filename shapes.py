"""
shapes.py
=========
Preset target formations for the swarm, each defined as a canonical set of N
target points centered at the origin with roughly unit scale. We also expose the
target *pairwise distance matrix* for each shape, which is what the (rotation- and
translation-invariant) training loss actually compares against.

Each preset returns an (N, 2) tensor. For the MVP we assign agent k to target
slot k (a fixed labeling), so the NxN distance matrices are directly comparable.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch


# ----------------------------------------------------------------------------
# Canonical shape generators. Each returns an (N, 2) float tensor centered at
# the origin. They try to place N points sensibly even when N is not a "nice"
# number for the shape (e.g. a hexagon with N=12 -> two rings).
# ----------------------------------------------------------------------------

def _recenter(pts: torch.Tensor) -> torch.Tensor:
    return pts - pts.mean(dim=0, keepdim=True)


def square(n: int, scale: float = 1.0) -> torch.Tensor:
    """N points spread evenly around the perimeter of a square."""
    # Distribute points around the perimeter parameterized by arclength.
    side = 2.0 * scale
    perim = 4.0 * side
    pts = []
    for k in range(n):
        s = (k / n) * perim
        # Walk around the square starting at the bottom-left corner.
        if s < side:
            x, y = -scale + s, -scale
        elif s < 2 * side:
            x, y = scale, -scale + (s - side)
        elif s < 3 * side:
            x, y = scale - (s - 2 * side), scale
        else:
            x, y = -scale, scale - (s - 3 * side)
        pts.append((x, y))
    return _recenter(torch.tensor(pts, dtype=torch.float32))


def triangle(n: int, scale: float = 1.0) -> torch.Tensor:
    """N points spread evenly around an equilateral triangle perimeter."""
    # Triangle vertices (pointing up).
    verts = [
        (0.0, scale),
        (-scale * math.sin(math.radians(60)), -scale * 0.5),
        (scale * math.sin(math.radians(60)), -scale * 0.5),
    ]
    # Build the three edges and sample by arclength.
    edges = [(verts[i], verts[(i + 1) % 3]) for i in range(3)]
    edge_len = math.dist(edges[0][0], edges[0][1])
    perim = 3.0 * edge_len
    pts = []
    for k in range(n):
        s = (k / n) * perim
        e = int(s // edge_len)
        t = (s - e * edge_len) / edge_len
        (x0, y0), (x1, y1) = edges[e]
        pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return _recenter(torch.tensor(pts, dtype=torch.float32))


def hexagon(n: int, scale: float = 1.0) -> torch.Tensor:
    """N points around a regular hexagon perimeter."""
    verts = [
        (scale * math.cos(math.radians(60 * i)), scale * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]
    edges = [(verts[i], verts[(i + 1) % 6]) for i in range(6)]
    edge_len = math.dist(edges[0][0], edges[0][1])
    perim = 6.0 * edge_len
    pts = []
    for k in range(n):
        s = (k / n) * perim
        e = int(s // edge_len)
        e = min(e, 5)
        t = (s - e * edge_len) / edge_len
        (x0, y0), (x1, y1) = edges[e]
        pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return _recenter(torch.tensor(pts, dtype=torch.float32))


def line(n: int, scale: float = 1.0) -> torch.Tensor:
    """N points evenly spaced on a horizontal line segment."""
    xs = torch.linspace(-1.5 * scale, 1.5 * scale, n)
    ys = torch.zeros(n)
    return _recenter(torch.stack([xs, ys], dim=1))


# Registry of preset name -> generator function.
_GENERATORS = {
    "square": square,
    "hexagon": hexagon,
    "triangle": triangle,
    "line": line,
}

# Canonical ordering of presets; the index into this list is the z_shape id.
PRESET_NAMES: List[str] = ["square", "hexagon", "triangle", "line"]


def get_shape(name: str, n: int, scale: float = 1.0) -> torch.Tensor:
    """Return the (N, 2) target points for a named preset."""
    if name not in _GENERATORS:
        raise KeyError(f"Unknown shape '{name}'. Available: {list(_GENERATORS)}")
    return _GENERATORS[name](n, scale)


def pairwise_distance_matrix(points: torch.Tensor) -> torch.Tensor:
    """
    Given (..., N, 2) points, return the (..., N, N) matrix of Euclidean
    inter-point distances. This quantity is invariant to global rotation and
    translation, which is exactly what makes it a good training target.
    """
    diff = points.unsqueeze(-2) - points.unsqueeze(-3)  # (..., N, N, 2)
    return torch.linalg.norm(diff, dim=-1)              # (..., N, N)


def all_target_distance_matrices(n: int, scale: float = 1.0) -> Dict[str, torch.Tensor]:
    """Precompute the target distance matrix for every preset."""
    return {
        name: pairwise_distance_matrix(get_shape(name, n, scale))
        for name in PRESET_NAMES
    }


if __name__ == "__main__":
    # Quick visual sanity check of the presets.
    import matplotlib.pyplot as plt

    n = 12
    fig, axes = plt.subplots(1, len(PRESET_NAMES), figsize=(4 * len(PRESET_NAMES), 4))
    for ax, name in zip(axes, PRESET_NAMES):
        p = get_shape(name, n)
        ax.scatter(p[:, 0], p[:, 1])
        for k, (x, y) in enumerate(p):
            ax.annotate(str(k), (x, y))
        ax.set_title(name)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("shapes_preview.png", dpi=100)
    print("Wrote shapes_preview.png")
