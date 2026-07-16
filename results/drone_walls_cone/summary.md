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
| damping_weight | 0.2 |
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
| max_turn_deg | 180.0 |
| heading_rate_weight | 5e-06 |
| heading_rate_threshold_deg | 180.0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.3050 |
| hexagon | 0.1588 |
| triangle | 0.1079 |
| line | 0.4273 |
| **mean** | **0.2498** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.287 | 0 | 0/5 | 1.07 | 0 | 1.494 |
| hexagon | 0.265 | 0 | 0/5 | 1.08 | 0 | 1.500 |
| triangle | 0.250 | 0 | 0/5 | 1.04 | 0 | 1.500 |
| line | 0.251 | 0 | 0/5 | 1.05 | 0 | 1.499 |
| switch square->hexagon | 0.275 | 0 | 0/5 | 1.05 | 0 | 1.500 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.9237 | 0.3213 | 0.5013 | 2.6x |
| hexagon | 0.5984 | 0.2145 | 0.1865 | 1.4x |
| triangle | 0.1461 | 0.1544 | 0.1111 | 1.5x |
| line | 0.4330 | 0.5095 | 0.4997 | 1.4x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 111.3 | 195.0 | 195 | 17.3% |
| hexagon | 143.0 | 195.0 | 195 | 31.5% |
| triangle | 150.2 | 195.0 | 195 | 32.1% |
| line | 144.2 | 195.0 | 195 | 29.9% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
