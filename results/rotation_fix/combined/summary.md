# Results -- rotation_fix/combined

Checkpoint: `checkpoint_rotationfix_combined.pt`

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
| square | 0.3010 |
| hexagon | 0.1616 |
| triangle | 0.0890 |
| line | 0.4258 |
| **mean** | **0.2444** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.263 | 0 | 0/5 | 1.08 | 0 | 1.493 |
| hexagon | 0.262 | 0 | 0/5 | 1.10 | 0 | 1.500 |
| triangle | 0.250 | 0 | 0/5 | 1.10 | 0 | 1.500 |
| line | 0.258 | 0 | 0/5 | 1.05 | 0 | 1.485 |
| switch square->hexagon | 0.260 | 0 | 0/5 | 1.05 | 0 | 1.491 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.3369 | 0.3078 | 0.3468 | 1.4x |
| hexagon | 0.2360 | 0.2212 | 0.3251 | 2.1x |
| triangle | 0.0947 | 0.1411 | 0.1115 | 1.3x |
| line | 0.4523 | 0.4755 | 0.4576 | 1.4x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 130.2 | 195.0 | 195 | 24.3% |
| hexagon | 146.5 | 195.0 | 195 | 20.8% |
| triangle | 160.1 | 195.0 | 195 | 31.0% |
| line | 156.9 | 195.0 | 195 | 24.9% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
