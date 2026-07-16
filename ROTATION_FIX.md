# Fixing unrealistic heading rotation speed

**Branch:** `rotation-fix` &nbsp;|&nbsp; **Status:** sweep in progress

## The problem

`make_results.py`'s issue #3 heading metric (mean/p90/max angular speed, and
the fraction of steps exceeding 180 deg/s) showed every drone-deployment
checkpoint swinging its heading far faster than a Crazyflie can physically
yaw:

| checkpoint | mean deg/s (range across shapes) | worst-case max | steps >180 deg/s |
|---|---|---|---|
| `drone_fixed` (circular) | 24–138 | up to 1800 | 1.1–7.6% |
| `drone_walls` (circular) | 51–239 | up to 1800 | 5.4–16.5% |
| `drone_fixed_cone` | 137–349 | up to 1800 | 22.6–38.1% |
| `drone_walls_cone` | 137–352 | up to 1800 | 19.2–51.4% |

1800 deg/s is the measurement ceiling — a ~180-degree flip in a single 0.1s
control step. No real Crazyflie yaw authority comes close.

**Important asymmetry, established before this branch started:** heading only
drives the forward-FOV **cone** perception model's sensing direction;
**circular** perception is omnidirectional and ignores heading entirely (see
main README). So a fast-swinging heading arrow on a circular checkpoint is
cosmetic — nothing about real deployment (assuming lighthouse/omnidirectional
sensing) needs a Crazyflie to track it. It is a genuine physical problem only
for the **cone** checkpoints, where heading = the direction an actual
forward-facing sensor would have to point. This is why the comparison sweep
below trains on `config_drone_walls_cone.yaml` — the fully rotation-invariant,
worst-numbers case, and the one where the fix actually matters.

## Prior art in this exact repo

A hard yaw-rate cap was tried once before (commit `e1be1c1`) and reverted
(`34a5dce`). The revert was **not** because capping the follower is a bad
idea — it was because the cap was combined with *removing* `drag`/
`damping_loss`, which let the *target* direction (raw velocity) itself reverse
near-instantly (~3600 deg/s); no amount of patching the follower fixed a
noisy source. Today's baseline keeps `drag`, `damping_loss`, and EMA-smoothed
heading (`heading_smoothing`) intact, so revisiting a cap is a different,
safer experiment this time — not a repeat of that failure.

## Methods compared

All five variants below start from `config_drone_walls_cone.yaml`, differ in
exactly one axis each (or a deliberate combination), and train for 1000
epochs (half the production schedule, to keep the comparison sweep
tractable) with identical seed/architecture/loss-curriculum otherwise.

### 0. Baseline
The existing `checkpoint_drone_walls_cone.pt`, unchanged. Reference point.

### 1. Stronger EMA smoothing (`heading_smoothing: 0.2 -> 0.08`)
**Mechanism:** `heading_smoothing` is the EMA beta blending raw velocity into
the smoothed signal heading is derived from (`update_smoothed_vel` in
`graph.py`): `smoothed_vel <- beta*vel + (1-beta)*smoothed_vel_prev`. Lowering
beta weights history more heavily, damping out sharp velocity reversals
before they ever reach the heading follower. **Cost:** zero new code, zero
new failure modes — purely a hyperparameter already in the pipeline. Heading
becomes systematically laggier relative to true velocity direction, which
(for cone perception) means the sensing cone points at a slightly stale
direction more of the time, not just during sharp turns.

### 2. Hard yaw-rate cap (`max_turn_deg: 180`)
**Mechanism:** `graph.update_heading` now accepts `max_turn_rad` — instead of
snapping straight to the target direction, it rotates the *current* heading
toward the target by at most that many radians this step (signed angle via
`atan2(cross, dot)`, clamped, reapplied as a 2D rotation). A hard,
deployment-time guarantee: whatever the policy learned, the simulated
(and eventually the physically-flown) heading literally cannot exceed the
cap. **Cost:** during the fastest reactive maneuvers (e.g. the safety
filter's abrupt velocity corrections), the capped heading can noticeably lag
true velocity direction, so a cone agent's actual sensing window points
somewhere slightly stale exactly when an evasive maneuver is happening.

### 3. Learned heading-rate penalty (`heading_rate_weight: 1e-5`, threshold 180 deg/s)
**Mechanism:** a new differentiable loss term, `losses.heading_rate_loss` —
a violation-exposure hinge (same construction as the collision/bounds losses:
`relu(angular_speed_deg - threshold)^2`, summed over time x dt, not
averaged, so a brief spike isn't diluted by a long horizon) on the angular
speed between consecutive *unsmoothed* heading steps. Required adding
`heading_history` tracking through `sim.rollout()` (previously only
`vel_history`/`pos_history` were kept). Ramped on the same curriculum as
`separation_weight` (0 -> full over the first half of training).
**Difference from the hard cap:** this doesn't clamp anything — it pressures
the POLICY itself to produce accelerations that don't demand sharp turns,
so gradients reach whatever upstream decision (an abrupt reversal, a
safety-filter correction) caused the fast turn in the first place. In
principle this could fix the problem at its source rather than papering over
the symptom; in practice it depends on the model having enough capacity/
training signal to find such trajectories.

### 4. Stronger damping (`damping_weight: 0.1 -> 0.4`, `drag: 0.02 -> 0.05`)
**Mechanism:** per the historical postmortem, damping/drag aren't just about
parking at the end of a rollout — they keep velocity's direction changing
*smoothly throughout*, which is what makes "heading = velocity direction" a
trustworthy signal at all. If the safety filter's abrupt corrections (or
ordinary collision-avoidance maneuvers) are injecting large velocity swings
that damping isn't currently strong enough to smooth out, turning damping up
is the most conservative lever — it doesn't touch heading machinery at all,
just makes the underlying velocity signal calmer. **Cost:** stronger damping
also fights the model's ability to move quickly/decisively, which could hurt
formation-convergence speed or hold quality.

### 5. Combined (`max_turn_deg: 180` + `heading_rate_weight: 5e-6` + `damping_weight: 0.2`)
The hard cap as the non-negotiable guarantee, plus a lighter-weight version
of the learned penalty and a moderate damping increase, on the theory that
each mechanism covers the others' gaps: the cap guarantees the ceiling even
early in training or for a novel initial condition; the loss nudges the
policy not to want to hit that ceiling; extra damping keeps the input signal
calmer to begin with.

## Results

*(sweep in progress — table below fills in as each variant's 5-seed, 60s
`make_results.py` run completes)*

| variant | heading mean (deg/s) | heading p90 | steps >180 deg/s | formation error | collision-free? |
|---|---|---|---|---|---|
| 0. baseline | | | | | |
| 1. smoothing | | | | | |
| 2. yawcap | | | | | |
| 3. lossreg | | | | | |
| 4. damping | | | | | |
| 5. combined | | | | | |

## Recommendation

*(pending sweep completion)*
