# Results -- rotation_fix/damping

Checkpoint: `checkpoint_rotationfix_damping.pt`

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
| epochs | 1000 |
| lr | 0.002 |
| lr_min | 0.0003 |
| t_min | 40 |
| t_max | 120 |
| hold_tail | 10 |
| damping_weight | 0.4 |
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
| square | 0.4880 |
| hexagon | 0.2888 |
| triangle | 0.1665 |
| line | 0.4340 |
| **mean** | **0.3443** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.250 | 0 | 0/5 | 1.07 | 0 | 1.500 |
| hexagon | 0.250 | 0 | 0/5 | 1.10 | 0 | 1.500 |
| triangle | 0.246 | 0 | 0/5 | 1.07 | 0 | 1.500 |
| line | 0.249 | 0 | 0/5 | 1.05 | 0 | 1.489 |
| switch square->hexagon | 0.249 | 0 | 0/5 | 1.07 | 0 | 1.497 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.4751 | 0.4784 | 0.4944 | 1.6x |
| hexagon | 0.3513 | 0.4054 | 0.4093 | 2.5x |
| triangle | 0.2211 | 0.1908 | 0.1753 | 1.4x |
| line | 0.5226 | 0.6681 | 0.4685 | 1.5x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 146.3 | 350.3 | 1800 | 21.6% |
| hexagon | 139.9 | 327.3 | 1798 | 20.5% |
| triangle | 165.3 | 395.4 | 1798 | 24.3% |
| line | 157.5 | 381.8 | 1800 | 22.0% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
