# Results -- rotation_fix/lossreg

Checkpoint: `checkpoint_rotationfix_lossreg.pt`

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
| heading_rate_weight | 1e-05 |
| heading_rate_threshold_deg | 180.0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.7210 |
| hexagon | 0.4222 |
| triangle | 0.3148 |
| line | 0.5266 |
| **mean** | **0.4962** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.250 | 0 | 0/5 | 0.50 | 0 | 1.349 |
| hexagon | 0.249 | 0 | 0/5 | 0.48 | 0 | 1.331 |
| triangle | 0.249 | 0 | 0/5 | 0.48 | 0 | 1.333 |
| line | 0.249 | 0 | 0/5 | 0.47 | 0 | 1.336 |
| switch square->hexagon | 0.249 | 0 | 0/5 | 0.46 | 0 | 1.347 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.7645 | 0.9091 | 1.0071 | 2.1x |
| hexagon | 0.6160 | 0.3940 | 0.3880 | 1.5x |
| triangle | 0.4975 | 0.2716 | 0.2674 | 1.4x |
| line | 0.7518 | 0.6919 | 0.6502 | 1.3x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 64.7 | 137.5 | 1798 | 6.6% |
| hexagon | 50.8 | 112.8 | 1800 | 4.6% |
| triangle | 47.3 | 103.2 | 1797 | 3.9% |
| line | 45.4 | 99.0 | 1797 | 3.9% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
