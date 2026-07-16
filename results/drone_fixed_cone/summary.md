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
| damping_weight | 0.2 |
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
| max_turn_deg | 180.0 |
| heading_rate_weight | 5e-06 |
| heading_rate_threshold_deg | 180.0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | fixed-target position MSE |
|---|---|
| square | 0.0007 |
| hexagon | 0.0058 |
| triangle | 0.0064 |
| line | 0.0020 |
| **mean** | **0.0037** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.275 | 0 | 0/5 | 1.10 | 0 | 1.304 |
| hexagon | 0.290 | 0 | 0/5 | 1.11 | 0 | 1.283 |
| triangle | 0.250 | 0 | 0/5 | 1.00 | 0 | 1.000 |
| line | 0.267 | 0 | 0/5 | 1.07 | 0 | 1.392 |
| switch square->hexagon | 0.275 | 0 | 0/5 | 1.10 | 0 | 1.365 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0008 | 0.0009 | 0.0009 | 1.6x |
| hexagon | 0.0075 | 0.0070 | 0.0068 | 1.6x |
| triangle | 0.0076 | 0.0069 | 0.0077 | 1.5x |
| line | 0.0031 | 0.0024 | 0.0021 | 1.5x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 108.5 | 195.0 | 195 | 29.8% |
| hexagon | 115.3 | 195.0 | 195 | 29.9% |
| triangle | 67.3 | 165.0 | 195 | 8.8% |
| line | 110.1 | 195.0 | 195 | 16.9% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
