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
| square | 0.1595 |
| hexagon | 0.0872 |
| triangle | 0.0545 |
| line | 0.0536 |
| **mean** | **0.0887** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | final sep (m) | final-state collisions | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|---|---|
| square | 0.352 | 0 | 0/5 | 1.178 | 0/5 | 1.00 | 0 | 1.479 |
| hexagon | 0.352 | 0 | 0/5 | 0.653 | 0/5 | 1.01 | 0 | 1.487 |
| triangle | 0.339 | 0 | 0/5 | 0.366 | 0/5 | 1.03 | 0 | 1.488 |
| line | 0.332 | 0 | 0/5 | 0.621 | 0/5 | 1.00 | 0 | 1.482 |
| switch square->hexagon | 0.332 | 0 | 0/5 | 0.481 | 0/5 | 1.00 | 0 | 1.479 |

**Final-state verdict (the parked formation): NO COLLISIONS -- every run ends with all drones clear.**

**Verdict (all steps, including transit): COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.3471 | 0.1387 | 0.1721 | 4.3x |
| hexagon | 0.1992 | 0.0497 | 0.0424 | 3.2x |
| triangle | 0.1033 | 0.0528 | 0.0748 | 4.5x |
| line | 0.0726 | 0.1102 | 0.0628 | 7.3x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 98.2 | 195.0 | 195 | 18.5% |
| hexagon | 137.3 | 195.0 | 195 | 33.9% |
| triangle | 150.1 | 195.0 | 195 | 43.4% |
| line | 140.9 | 195.0 | 195 | 29.1% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
