# Results -- drone_fixed

Checkpoint: `checkpoint_drone_fixed.pt`

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
| arena_mode | fixed |
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

| shape | fixed-target position MSE |
|---|---|
| square | 0.0006 |
| hexagon | 0.0011 |
| triangle | 0.0006 |
| line | 0.0020 |
| **mean** | **0.0011** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | final sep (m) | final-state collisions | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|---|---|
| square | 0.333 | 0 | 0/5 | 1.549 | 0/5 | 1.07 | 0 | 1.265 |
| hexagon | 0.311 | 0 | 0/5 | 1.003 | 0/5 | 1.00 | 0 | 1.110 |
| triangle | 0.278 | 0 | 0/5 | 0.599 | 0/5 | 1.01 | 0 | 1.246 |
| line | 0.344 | 0 | 0/5 | 0.763 | 0/5 | 1.00 | 0 | 1.229 |
| switch square->hexagon | 0.333 | 0 | 0/5 | 1.003 | 0/5 | 1.07 | 0 | 1.265 |

**Final-state verdict (the parked formation): NO COLLISIONS -- every run ends with all drones clear.**

**Verdict (all steps, including transit): COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0006 | 0.0006 | 0.0006 | 1.1x |
| hexagon | 0.0011 | 0.0011 | 0.0011 | 1.2x |
| triangle | 0.0006 | 0.0006 | 0.0008 | 1.7x |
| line | 0.0021 | 0.0021 | 0.0021 | 1.2x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 19.7 | 23.3 | 1581 | 0.4% |
| hexagon | 20.3 | 15.0 | 1769 | 0.7% |
| triangle | 276.3 | 1650.3 | 1799 | 18.1% |
| line | 31.5 | 36.6 | 1790 | 2.6% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
