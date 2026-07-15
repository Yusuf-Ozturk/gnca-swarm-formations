# Results -- drone_walls

Checkpoint: `checkpoint_drone_walls.pt`

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
| arena_mode | walls |
| arena_half | 1.5 |
| wall_margin | 0.3 |
| wall_strength | 0.5 |
| drone_radius | 0.1 |
| separation_weight | 10.0 |
| separation_margin | 0.1 |
| min_start_dist | 0.5 |
| shape_scale | 0.8 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.0095 |
| hexagon | 0.0066 |
| triangle | 0.0163 |
| line | 0.0272 |
| **mean** | **0.0149** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.282 | 0 | 0/5 | 1.04 | 0 | 1.485 |
| hexagon | 0.262 | 0 | 0/5 | 1.04 | 0 | 1.467 |
| triangle | 0.275 | 0 | 0/5 | 1.06 | 0 | 1.454 |
| line | 0.282 | 0 | 0/5 | 1.10 | 0 | 1.493 |
| switch square->hexagon | 0.282 | 0 | 0/5 | 1.04 | 0 | 1.485 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0056 | 0.0042 | 0.0042 | 1.5x |
| hexagon | 0.0065 | 0.0094 | 0.0096 | 1.7x |
| triangle | 0.0166 | 0.0180 | 0.0180 | 1.1x |
| line | 0.0292 | 0.0271 | 0.0301 | 1.8x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 238.1 | 1551.8 | 1799 | 16.1% |
| hexagon | 212.2 | 819.9 | 1800 | 22.1% |
| triangle | 193.3 | 378.2 | 1799 | 37.2% |
| line | 127.6 | 292.6 | 1799 | 18.6% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
