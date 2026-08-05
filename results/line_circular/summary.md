# line / circular

Checkpoint: `gamma_line_circular.pt`  --  5 seeds x 300 steps (30 s)

## Config

| key | value |
|---|---|
| n | 4 |
| dt | 0.1 |
| perception | circular |
| half_angle_deg | 75.0 |
| sensing_range | 3.0 |
| max_speed | 1.0 |
| max_accel | 2.0 |
| max_yaw_rate_deg | 180.0 |
| max_yaw_accel_deg | 360.0 |
| hidden | 128 |
| msg_dim | 128 |
| epochs | 2000 |
| lr | 0.003 |
| t_min | 40 |
| t_max | 120 |
| hold_tail | 10 |
| separation_weight | 10.0 |
| collision_weight | 100.0 |
| separation_margin | 0.1 |
| separation_ramp | 0.5 |
| drone_radius | 0.1 |
| shape_scale | 0.8 |
| seed | 0 |

## Formation (pose- and permutation-invariant assignment error)

| seed | 0s | 5s | 10s | 15s | 20s | 25s | 30s | final |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2388 | 0.0262 | 0.0136 | 0.0079 | 0.0077 | 0.0076 | 0.0078 | 0.0078 |
| 2 | 0.0640 | 0.0209 | 0.0078 | 0.0070 | 0.0083 | 0.0056 | 0.0092 | 0.0092 |
| 3 | 0.3920 | 0.0313 | 0.0061 | 0.0166 | 0.0055 | 0.0146 | 0.0052 | 0.0052 |
| 4 | 0.5030 | 0.2230 | 0.0995 | 0.0408 | 0.0184 | 0.0114 | 0.0113 | 0.0113 |
| 5 | 0.9001 | 0.2006 | 0.0818 | 0.0268 | 0.0132 | 0.0102 | 0.0049 | 0.0049 |

mean final error **0.00768**, worst seed 0.01130

## Collisions (collision = centers < 0.20 m)

| seed | min sep, any step | min sep, FINAL | colliding steps |
|---|---|---|---|
| 1 | 0.529 | 0.711 | 0 |
| 2 | 0.494 | 0.732 | 0 |
| 3 | 0.453 | 0.818 | 0 |
| 4 | 0.540 | 0.785 | 0 |
| 5 | 0.439 | 0.622 | 0 |

Target slots are 0.80 m apart, so a perfect formation is collision-free by construction.

**Final state: no collisions.**
**All steps: no collisions.**

## Equilibrium (no damping term, no drag -- gamma had to find this)

| seed | mean speed, last 1 s | mean yaw rate, last 1 s | peak speed | peak yaw rate |
|---|---|---|---|---|
| 1 | 0.1372 m/s | 180.0 deg/s | 0.85 m/s | 180 deg/s |
| 2 | 0.1514 m/s | 180.0 deg/s | 0.81 m/s | 180 deg/s |
| 3 | 0.1416 m/s | 180.0 deg/s | 1.00 m/s | 180 deg/s |
| 4 | 0.1646 m/s | 180.0 deg/s | 0.91 m/s | 180 deg/s |
| 5 | 0.1269 m/s | 180.0 deg/s | 1.00 m/s | 180 deg/s |

mean residual speed **0.1443 m/s** (envelope 1.0 m/s), residual yaw rate **180.0 deg/s** (envelope 180.0 deg/s)
