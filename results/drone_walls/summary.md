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
| square | 0.1856 |
| hexagon | 0.1039 |
| triangle | 0.0752 |
| line | 0.1588 |
| **mean** | **0.1309** |

## Safety check (issue #4): every 60s run, every seed

Collision = two drone centers closer than 2 x drone_radius = 0.20m (overlapping 10cm safety disks). Out-of-bounds = any drone outside the 3m x 3m flight area.

| run | min separation (m) | collision steps | collided runs | max speed (m/s) | OOB steps | max \|coord\| (m) |
|---|---|---|---|---|---|---|
| square | 0.295 | 0 | 0/5 | 1.03 | 0 | 1.489 |
| hexagon | 0.283 | 0 | 0/5 | 1.04 | 0 | 1.491 |
| triangle | 0.250 | 0 | 0/5 | 1.07 | 0 | 1.482 |
| line | 0.250 | 0 | 0/5 | 1.07 | 0 | 1.493 |
| switch square->hexagon | 0.285 | 0 | 0/5 | 1.03 | 0 | 1.491 |

**Verdict: COLLISION-FREE (and in-bounds) on every run.**

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.1885 | 0.1822 | 0.1814 | 1.0x |
| hexagon | 0.1068 | 0.1049 | 0.1052 | 1.0x |
| triangle | 0.0986 | 0.0832 | 0.0789 | 1.1x |
| line | 0.1763 | 0.1697 | 0.1696 | 1.4x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 114.6 | 222.0 | 1800 | 12.0% |
| hexagon | 239.4 | 1552.7 | 1800 | 16.5% |
| triangle | 50.8 | 103.9 | 1795 | 5.4% |
| line | 189.1 | 480.6 | 1800 | 25.3% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
