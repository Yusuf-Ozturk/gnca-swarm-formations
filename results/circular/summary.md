# Results -- circular perception

Checkpoint: `checkpoint_circular.pt`

## Config

| key | value |
|---|---|
| perception | circular |
| half_angle_deg | 75.0 |
| sensing_range | 1.5 |
| self_rotation_deg | 15.0 |
| heading_smoothing | 0.2 |
| n | 12 |
| dt | 0.1 |
| hidden | 128 |
| msg_dim | 128 |
| epochs | 3000 |
| lr | 0.003 |
| lr_min | 0.0003 |
| t_min | 40 |
| t_max | 120 |
| hold_tail | 10 |
| damping_weight | 0.1 |
| seed | 0 |

## Final training error (fresh inits, rolled t_max steps)

| shape | distance-matrix error |
|---|---|
| square | 0.0001 |
| hexagon | 0.0001 |
| triangle | 0.0000 |
| line | 0.0001 |
| **mean** | **0.0001** |

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0005 | 0.0001 | 0.0001 | 1.7x |
| hexagon | 0.0003 | 0.0000 | 0.0000 | 1.3x |
| triangle | 0.0000 | 0.0000 | 0.0000 | 2.9x |
| line | 0.0001 | 0.0001 | 0.0001 | 1.9x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 122.3 | 287.7 | 1799 | 19.3% |
| hexagon | 42.9 | 95.3 | 1791 | 4.1% |
| triangle | 24.2 | 60.0 | 1779 | 2.0% |
| line | 6.2 | 11.9 | 1772 | 0.5% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
Animations (generate with `--animations`): `animations/` -- per-shape convergence, runtime shape switching, and 60s holds for square + line.
