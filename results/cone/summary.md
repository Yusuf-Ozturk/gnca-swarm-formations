# Results -- cone perception

Checkpoint: `checkpoint.pt`

## Config

| key | value |
|---|---|
| perception | cone |
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
| square | 0.0093 |
| hexagon | 0.0146 |
| triangle | 0.0049 |
| line | 0.1453 |
| **mean** | **0.0435** |

## Formation hold over 60s (5 seeds, worst seed shown)

| shape | err @10s | err @30s | err @60s | worst final/min |
|---|---|---|---|---|
| square | 0.0353 | 0.0069 | 0.0069 | 2.9x |
| hexagon | 0.0318 | 0.0104 | 0.0092 | 2.1x |
| triangle | 0.0225 | 0.0041 | 0.0295 | 29.5x |
| line | 0.0808 | 0.0730 | 0.1892 | 9.2x |

## Heading angular speed during the 60s holds (issue #3 metric)

| shape | mean (deg/s) | p90 | max | steps >180 deg/s |
|---|---|---|---|---|
| square | 10.8 | 18.8 | 1799 | 0.9% |
| hexagon | 13.9 | 28.5 | 1755 | 1.0% |
| triangle | 10.7 | 18.3 | 1793 | 0.6% |
| line | 10.2 | 20.9 | 1798 | 0.6% |

Full per-seed time series: `drift_tables.md` / `drift_<shape>.csv`.
