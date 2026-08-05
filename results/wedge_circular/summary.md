# wedge / circular

Checkpoint: `gamma_wedge_circular.pt`  --  5 seeds x 300 steps (30 s)

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
| 1 | 0.3099 | 0.3089 | 0.3089 | 0.3089 | 0.3089 | 0.3089 | 0.3089 | 0.3089 |
| 2 | 0.2316 | 0.2305 | 0.2305 | 0.2305 | 0.2305 | 0.2305 | 0.2305 | 0.2305 |
| 3 | 0.4596 | 0.4596 | 0.4596 | 0.4596 | 0.4596 | 0.4596 | 0.4596 | 0.4596 |
| 4 | 1.0279 | 1.0290 | 1.0290 | 1.0290 | 1.0290 | 1.0290 | 1.0290 | 1.0290 |
| 5 | 0.9278 | 0.9311 | 0.9311 | 0.9311 | 0.9311 | 0.9311 | 0.9311 | 0.9311 |

mean final error **0.59182**, worst seed 1.02902

## Collisions (collision = centers < 0.20 m)

| seed | min sep, any step | min sep, FINAL | colliding steps |
|---|---|---|---|
| 1 | 0.771 | 0.771 | 0 |
| 2 | 0.530 | 0.530 | 0 |
| 3 | 0.566 | 0.566 | 0 |
| 4 | 0.571 | 0.571 | 0 |
| 5 | 1.290 | 1.290 | 0 |

Target slots are 0.72 m apart, so a perfect formation is collision-free by construction.

**Final state: no collisions.**
**All steps: no collisions.**

## Equilibrium (no damping term, no drag -- gamma had to find this)

| seed | mean speed, last 1 s | mean yaw rate, last 1 s | peak speed | peak yaw rate |
|---|---|---|---|---|
| 1 | 0.0000 m/s | 180.0 deg/s | 0.29 m/s | 180 deg/s |
| 2 | 0.0000 m/s | 180.0 deg/s | 0.27 m/s | 180 deg/s |
| 3 | 0.0000 m/s | 180.0 deg/s | 0.19 m/s | 180 deg/s |
| 4 | 0.0000 m/s | 180.0 deg/s | 0.29 m/s | 180 deg/s |
| 5 | 0.0000 m/s | 180.0 deg/s | 0.28 m/s | 180 deg/s |

mean residual speed **0.0000 m/s** (envelope 1.0 m/s), residual yaw rate **180.0 deg/s** (envelope 180.0 deg/s)
