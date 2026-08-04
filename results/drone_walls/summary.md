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
| n | 4 |
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
| max_turn_deg | 0.0 |
| heading_rate_weight | 0.0 |
| heading_rate_threshold_deg | 180.0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.0931 |
| hexagon | 0.0007 |
| triangle | 0.0006 |
| line | 0.0181 |
| **mean** | **0.0281** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | final sep (m) | final-state collisions | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|---|---|
| square | 0.285 | 0 | 0/5 | 1.207 | 0/5 | 1.00 | 0 | 1.429 |
| hexagon | 0.288 | 0 | 0/5 | 1.003 | 0/5 | 1.07 | 0 | 1.464 |
| triangle | 0.322 | 0 | 0/5 | 0.622 | 0/5 | 1.02 | 0 | 1.469 |
| line | 0.346 | 0 | 0/5 | 0.798 | 0/5 | 1.01 | 0 | 1.471 |
| switch square->hexagon | 0.285 | 0 | 0/5 | 1.003 | 0/5 | 1.00 | 0 | 1.467 |

**Final-state verdict (the parked formation): NO COLLISIONS -- every run ends with all drones clear.**

**Verdict (all steps, including transit): COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.2747 | 0.0963 | 0.1118 | 4.3x |
| hexagon | 0.0008 | 0.0008 | 0.0007 | 1.1x |
| triangle | 0.0008 | 0.0007 | 0.0007 | 1.2x |
| line | 0.1459 | 0.0027 | 0.0008 | 1.0x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 55.8 | 134.4 | 1799 | 6.7% |
| hexagon | 115.9 | 256.5 | 1800 | 13.6% |
| triangle | 400.6 | 1460.1 | 1799 | 33.8% |
| line | 60.7 | 140.7 | 1793 | 7.8% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
