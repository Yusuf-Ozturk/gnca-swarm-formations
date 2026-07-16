# Results -- drone_fixed_cone

Checkpoint: `checkpoint_drone_fixed_cone.pt`

## Config

| key | value |
|---|---|
| perception | cone |
| half_angle_deg | 75.0 |
| sensing_range | 1.2 |
| self_rotation_deg | 15.0 |
| heading_smoothing | 0.2 |
| n | 8 |
| dt | 0.1 |
| hidden | 128 |
| msg_dim | 128 |
| epochs | 2000 |
| lr | 0.002 |
| lr_min | 0.0003 |
| t_min | 40 |
| t_max | 120 |
| hold_tail | 10 |
| damping_weight | 0.1 |
| seed | 0 |
| arena_mode | fixed |
| arena_half | 1.5 |
| wall_margin | 0.3 |
| wall_strength | 0.5 |
| drone_radius | 0.1 |
| separation_weight | 10.0 |
| separation_margin | 0.1 |
| min_start_dist | 0.5 |
| shape_scale | 0.8 |

## Final training error (fresh inits, rolled t_max steps)

| shape | fixed-target position MSE |
|---|---|
| square | 0.0032 |
| hexagon | 0.0027 |
| triangle | 0.0112 |
| line | 0.0021 |
| **mean** | **0.0048** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.298 | 0 | 0/5 | 1.09 | 0 | 1.395 |
| hexagon | 0.290 | 0 | 0/5 | 1.09 | 0 | 1.449 |
| triangle | 0.268 | 0 | 0/5 | 1.04 | 0 | 1.031 |
| line | 0.290 | 0 | 0/5 | 1.08 | 0 | 1.320 |
| switch square->hexagon | 0.268 | 0 | 0/5 | 1.15 | 0 | 1.395 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0047 | 0.0034 | 0.0068 | 3.1x |
| hexagon | 0.0031 | 0.0031 | 0.0033 | 1.5x |
| triangle | 0.0063 | 0.0061 | 0.0059 | 1.4x |
| line | 0.0023 | 0.0022 | 0.0026 | 1.9x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 349.4 | 1222.7 | 1800 | 38.1% |
| hexagon | 184.2 | 470.4 | 1800 | 22.6% |
| triangle | 254.7 | 902.4 | 1800 | 30.7% |
| line | 300.3 | 896.3 | 1800 | 40.7% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
