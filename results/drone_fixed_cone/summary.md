# Results -- drone_fixed_cone

Checkpoint: `checkpoint_drone_fixed_cone.pt`

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
| arena_mode | fixed |
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

| shape | fixed-target position MSE |
|---|---|
| square | 0.0001 |
| hexagon | 0.0003 |
| triangle | 0.0004 |
| line | 0.0003 |
| **mean** | **0.0003** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | final sep (m) | final-state collisions | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|---|---|
| square | 0.348 | 0 | 0/5 | 1.574 | 0/5 | 1.01 | 0 | 1.112 |
| hexagon | 0.335 | 0 | 0/5 | 1.010 | 0/5 | 1.00 | 0 | 0.991 |
| triangle | 0.285 | 0 | 0/5 | 0.562 | 0/5 | 1.00 | 0 | 0.991 |
| line | 0.300 | 0 | 0/5 | 0.757 | 0/5 | 1.00 | 0 | 1.297 |
| switch square->hexagon | 0.303 | 0 | 0/5 | 1.024 | 0/5 | 1.01 | 0 | 1.112 |

**Final-state verdict (the parked formation): NO COLLISIONS -- every run ends with all drones clear.**

**Verdict (all steps, including transit): COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0001 | 0.0001 | 0.0001 | 1.2x |
| hexagon | 0.0006 | 0.0005 | 0.0005 | 5.8x |
| triangle | 0.0007 | 0.0007 | 0.0007 | 4.0x |
| line | 0.0002 | 0.0003 | 0.0003 | 4.2x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 19.5 | 15.0 | 195 | 1.1% |
| hexagon | 116.6 | 195.0 | 195 | 16.0% |
| triangle | 110.8 | 188.1 | 195 | 11.7% |
| line | 79.7 | 195.0 | 195 | 15.7% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
