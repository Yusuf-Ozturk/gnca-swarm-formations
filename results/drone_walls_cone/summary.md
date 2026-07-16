# Results -- drone_walls_cone

Checkpoint: `checkpoint_drone_walls_cone.pt`

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
| square | 0.4348 |
| hexagon | 0.1945 |
| triangle | 0.1067 |
| line | 0.3781 |
| **mean** | **0.2785** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.250 | 0 | 0/5 | 1.11 | 0 | 1.500 |
| hexagon | 0.250 | 0 | 0/5 | 1.10 | 0 | 1.500 |
| triangle | 0.249 | 0 | 0/5 | 1.06 | 0 | 1.496 |
| line | 0.250 | 0 | 0/5 | 1.08 | 0 | 1.500 |
| switch square->hexagon | 0.250 | 0 | 0/5 | 1.10 | 0 | 1.500 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.8261 | 0.8809 | 0.5030 | 1.8x |
| hexagon | 0.2376 | 0.5165 | 0.2306 | 1.6x |
| triangle | 0.1448 | 0.1645 | 0.1566 | 1.5x |
| line | 0.4023 | 0.4134 | 0.3983 | 1.1x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 137.4 | 297.1 | 1798 | 19.2% |
| hexagon | 212.4 | 530.0 | 1800 | 31.4% |
| triangle | 352.3 | 996.2 | 1800 | 51.4% |
| line | 323.5 | 902.7 | 1800 | 47.9% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
