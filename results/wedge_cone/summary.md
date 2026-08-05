# wedge / cone

Checkpoint: `gamma_wedge_cone.pt`  --  5 seeds x 300 steps (30 s)

## Config

| key | value |
|---|---|
| n | 4 |
| dt | 0.1 |
| perception | cone |
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
| 1 | 0.3099 | 0.0854 | 0.0857 | 0.0857 | 0.0857 | 0.0857 | 0.0857 | 0.0857 |
| 2 | 0.2316 | 0.0594 | 0.0594 | 0.0594 | 0.0594 | 0.0594 | 0.0594 | 0.0594 |
| 3 | 0.4596 | 0.1716 | 0.0507 | 0.0507 | 0.0507 | 0.0507 | 0.0507 | 0.0507 |
| 4 | 1.0279 | 0.0823 | 0.0752 | 0.0752 | 0.0752 | 0.0752 | 0.0752 | 0.0752 |
| 5 | 0.9278 | 0.0817 | 0.0817 | 0.0817 | 0.0817 | 0.0817 | 0.0817 | 0.0817 |

mean final error **0.07055**, worst seed 0.08573

## Collisions (collision = centers < 0.20 m)

| seed | min sep, any step | min sep, FINAL | colliding steps |
|---|---|---|---|
| 1 | 0.743 | 0.743 | 0 |
| 2 | 0.529 | 0.529 | 0 |
| 3 | 0.527 | 0.528 | 0 |
| 4 | 0.546 | 0.546 | 0 |
| 5 | 0.629 | 0.629 | 0 |

Target slots are 0.72 m apart, so a perfect formation is collision-free by construction.

**Final state: no collisions.**
**All steps: no collisions.**

## Equilibrium (no damping term, no drag -- gamma had to find this)

| seed | mean speed, last 1 s | mean yaw rate, last 1 s | peak speed | peak yaw rate |
|---|---|---|---|---|
| 1 | 0.0000 m/s | 3.4 deg/s | 1.00 m/s | 89 deg/s |
| 2 | 0.0000 m/s | 42.3 deg/s | 1.00 m/s | 180 deg/s |
| 3 | 0.0000 m/s | 45.5 deg/s | 1.00 m/s | 180 deg/s |
| 4 | 0.0000 m/s | 0.3 deg/s | 1.00 m/s | 65 deg/s |
| 5 | 0.0000 m/s | 9.3 deg/s | 1.00 m/s | 91 deg/s |

mean residual speed **0.0000 m/s** (envelope 1.0 m/s), residual yaw rate **20.2 deg/s** (envelope 180.0 deg/s)
