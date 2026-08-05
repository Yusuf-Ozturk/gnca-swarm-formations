# triangle_centroid / circular

Checkpoint: `gamma_triangle_centroid_circular.pt`  --  5 seeds x 300 steps (30 s)

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
| 1 | 0.3836 | 0.1323 | 0.1046 | 0.1945 | 0.2444 | 0.2910 | 0.0512 | 0.0512 |
| 2 | 0.3847 | 0.1997 | 0.1163 | 0.0570 | 0.0737 | 0.1508 | 0.0662 | 0.0662 |
| 3 | 0.5160 | 0.2359 | 0.2046 | 0.1215 | 0.1071 | 0.1510 | 0.1777 | 0.1777 |
| 4 | 1.3373 | 0.2420 | 0.1941 | 0.1729 | 0.1317 | 0.2118 | 0.1780 | 0.1780 |
| 5 | 0.8197 | 0.0661 | 0.1040 | 0.1006 | 0.1034 | 0.1107 | 0.1445 | 0.1445 |

mean final error **0.12350**, worst seed 0.17802

## Collisions (collision = centers < 0.20 m)

| seed | min sep, any step | min sep, FINAL | colliding steps |
|---|---|---|---|
| 1 | 0.011 | 0.353 | 113 |
| 2 | 0.016 | 0.380 | 53 |
| 3 | 0.020 | 0.128 **COLLISION** | 72 |
| 4 | 0.039 | 0.168 **COLLISION** | 66 |
| 5 | 0.030 | 0.364 | 40 |

Target slots are 0.80 m apart, so a perfect formation is collision-free by construction.

**Final state: COLLISION.**
**All steps: 344 colliding steps.**

## Equilibrium (no damping term, no drag -- gamma had to find this)

| seed | mean speed, last 1 s | mean yaw rate, last 1 s | peak speed | peak yaw rate |
|---|---|---|---|---|
| 1 | 0.5955 m/s | 87.3 deg/s | 1.00 m/s | 180 deg/s |
| 2 | 0.5874 m/s | 97.4 deg/s | 1.00 m/s | 180 deg/s |
| 3 | 0.2908 m/s | 32.7 deg/s | 1.00 m/s | 180 deg/s |
| 4 | 0.6135 m/s | 78.0 deg/s | 1.00 m/s | 180 deg/s |
| 5 | 0.7473 m/s | 83.1 deg/s | 1.00 m/s | 180 deg/s |

mean residual speed **0.5669 m/s** (envelope 1.0 m/s), residual yaw rate **75.7 deg/s** (envelope 180.0 deg/s)
