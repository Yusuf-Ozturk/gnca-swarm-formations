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
| square | 0.0018 |
| hexagon | 0.0070 |
| triangle | 0.0058 |
| line | 0.0029 |
| **mean** | **0.0044** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.279 | 0 | 0/5 | 1.06 | 17 | 1.631 |
| hexagon | 0.275 | 0 | 0/5 | 1.07 | 0 | 1.243 |
| triangle | 0.289 | 0 | 0/5 | 1.14 | 0 | 1.116 |
| line | 0.292 | 0 | 0/5 | 1.00 | 9 | 1.719 |
| switch square->hexagon | 0.273 | 0 | 0/5 | 1.13 | 27 | 1.822 |

**Verdict: SAFETY VIOLATIONS FOUND -- see rows above.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0018 | 0.0018 | 0.0018 | 1.0x |
| hexagon | 0.0073 | 0.0075 | 0.0077 | 1.3x |
| triangle | 0.0112 | 0.0061 | 0.0060 | 1.2x |
| line | 0.0029 | 0.0029 | 0.0029 | 1.0x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 127.6 | 102.6 | 1800 | 6.9% |
| hexagon | 260.0 | 697.9 | 1797 | 42.3% |
| triangle | 208.9 | 479.4 | 1795 | 33.7% |
| line | 23.3 | 27.6 | 1751 | 1.1% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
