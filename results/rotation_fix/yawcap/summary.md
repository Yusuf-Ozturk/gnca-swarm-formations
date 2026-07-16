# Results -- rotation_fix/yawcap

Checkpoint: `checkpoint_rotationfix_yawcap.pt`

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
| max_turn_deg | 180.0 |
| heading_rate_weight | 0.0 |
| heading_rate_threshold_deg | 180.0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.7003 |
| hexagon | 0.3948 |
| triangle | 0.1863 |
| line | 0.5468 |
| **mean** | **0.4571** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.248 | 0 | 0/5 | 1.06 | 0 | 1.500 |
| hexagon | 0.245 | 0 | 0/5 | 1.05 | 0 | 1.500 |
| triangle | 0.244 | 0 | 0/5 | 1.03 | 0 | 1.500 |
| line | 0.239 | 0 | 0/5 | 1.04 | 0 | 1.493 |
| switch square->hexagon | 0.244 | 0 | 0/5 | 1.05 | 0 | 1.500 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.8267 | 1.1306 | 1.0606 | 2.0x |
| hexagon | 0.9784 | 0.9647 | 0.4418 | 1.9x |
| triangle | 0.2320 | 0.2081 | 0.2101 | 1.6x |
| line | 0.6019 | 1.2263 | 0.6342 | 2.2x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 142.3 | 195.0 | 195 | 35.4% |
| hexagon | 117.8 | 195.0 | 195 | 26.0% |
| triangle | 154.1 | 195.0 | 195 | 39.0% |
| line | 129.9 | 195.0 | 195 | 28.2% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
