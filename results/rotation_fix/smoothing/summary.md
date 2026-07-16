# Results -- rotation_fix/smoothing

Checkpoint: `checkpoint_rotationfix_smoothing.pt`

## Config

| key | value |
|---|---|
| perception | cone |
| half_angle_deg | 75.0 |
| sensing_range | 1.2 |
| self_rotation_deg | 15.0 |
| heading_smoothing | 0.08 |
| n | 8 |
| dt | 0.1 |
| hidden | 128 |
| msg_dim | 128 |
| epochs | 1000 |
| lr | 0.002 |
| lr_min | 0.0003 |
| t_min | 40 |
| t_max | 120 |
| hold_tail | 10 |
| damping_weight | 0.1 |
| seed | 0 |
| arena_mode | walls |
| arena_half | 1.5 |
| wall_margin | 0.3 |
| wall_strength | 0.5 |
| drone_radius | 0.1 |
| separation_weight | 10.0 |
| separation_margin | 0.1 |
| min_start_dist | 0.5 |
| shape_scale | 0.8 |
| max_turn_deg | 0.0 |
| heading_rate_weight | 0.0 |
| heading_rate_threshold_deg | 180.0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.3004 |
| hexagon | 0.1409 |
| triangle | 0.0918 |
| line | 0.3708 |
| **mean** | **0.2260** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.262 | 0 | 0/5 | 1.00 | 0 | 1.487 |
| hexagon | 0.250 | 0 | 0/5 | 1.06 | 0 | 1.479 |
| triangle | 0.250 | 0 | 0/5 | 1.08 | 0 | 1.491 |
| line | 0.249 | 0 | 0/5 | 1.03 | 0 | 1.483 |
| switch square->hexagon | 0.250 | 0 | 0/5 | 1.00 | 0 | 1.477 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.3019 | 0.3969 | 0.3090 | 1.2x |
| hexagon | 0.1563 | 0.1617 | 0.1642 | 1.2x |
| triangle | 0.1014 | 0.1145 | 0.1106 | 1.2x |
| line | 0.4223 | 0.4025 | 0.4016 | 1.4x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 88.2 | 195.8 | 1791 | 11.3% |
| hexagon | 169.8 | 402.2 | 1799 | 25.1% |
| triangle | 207.3 | 530.5 | 1798 | 29.0% |
| line | 145.2 | 342.4 | 1797 | 21.5% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
