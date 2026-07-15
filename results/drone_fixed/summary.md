# Results -- drone_fixed

Checkpoint: `checkpoint_drone_fixed.pt`

## Config

| key | value |
|---|---|
| perception | circular |
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
| square | 0.0065 |
| hexagon | 0.0037 |
| triangle | 0.0033 |
| line | 0.0033 |
| **mean** | **0.0042** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.294 | 0 | 0/5 | 1.08 | 0 | 1.500 |
| hexagon | 0.297 | 0 | 0/5 | 1.06 | 0 | 1.185 |
| triangle | 0.284 | 0 | 0/5 | 1.10 | 0 | 1.000 |
| line | 0.262 | 0 | 0/5 | 1.06 | 0 | 1.481 |
| switch square->hexagon | 0.250 | 0 | 0/5 | 1.10 | 0 | 1.500 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0071 | 0.0071 | 0.0071 | 1.1x |
| hexagon | 0.0035 | 0.0035 | 0.0035 | 1.1x |
| triangle | 0.0034 | 0.0032 | 0.0032 | 1.0x |
| line | 0.0035 | 0.0032 | 0.0031 | 1.0x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 23.9 | 31.6 | 1683 | 1.1% |
| hexagon | 33.0 | 65.6 | 1784 | 3.0% |
| triangle | 137.7 | 71.9 | 1800 | 7.6% |
| line | 26.3 | 39.0 | 1783 | 1.4% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
