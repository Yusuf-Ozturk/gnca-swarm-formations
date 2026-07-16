# Fixing unrealistic heading rotation speed

**Branch:** `rotation-fix` &nbsp;|&nbsp; **Status:** done. Both cone
production checkpoints retrained and verified (see "Production result" at
the bottom).

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

All six trained/evaluated on `config_drone_walls_cone.yaml` at 1000 epochs
(the comparison-sweep schedule — half the 2000-epoch production schedule),
5 seeds, 60s rollouts + the switching transient. Numbers below are averaged
across the four preset shapes for readability; full per-shape tables are in
each `results/rotation_fix/<variant>/summary.md`.

| variant | heading mean (deg/s) | heading p90 | heading max | steps >180 deg/s | formation error @60s (worst seed) | collision-free? |
|---|---|---|---|---|---|---|
| 0. baseline | 256.4 | 681.5 | **1800** | 37.5% | 0.322 | yes |
| 1. smoothing | 152.6 | 367.7 | 1799 | 21.7% | **0.246** | yes |
| 2. yawcap | 136.0 | **195.0** | **195** | 32.2% | 0.587 | yes |
| 3. lossreg | **52.1** | **113.1** | 1800 | **4.8%** | 0.578 | yes |
| 4. damping | 152.2 | 363.7 | 1800 | 22.1% | 0.387 | yes |
| 5. combined | 148.4 | 195.0 | **195** | 25.3% | 0.310 | yes |

(Bold = best in column. "Formation error" is the multi-seed worst-seed
distance-matrix error at t=60s — lower is better; the metric this whole
project has used throughout, so it's directly comparable to every other
`summary.md` in the repo.)

**Every variant stays collision-free** — none of the heading interventions
touched the (independent) collision-avoidance stack, as expected since they
operate on a different part of the pipeline.

**Two structurally different kinds of improvement showed up, and no single
mechanism gets both:**

* **Ceiling-bounding** (`max_turn_deg`, alone or combined): the only
  mechanisms that touch the worst case at all. Both `yawcap` and `combined`
  land at *exactly* 195 deg/s max (180 cap + 15 deg/s `self_rotation_deg`
  layered on afterward) — a hard, verifiable guarantee, not a statistical
  tendency. `smoothing`, `lossreg`, and `damping` all leave the max at
  ~1800 deg/s regardless of how much they improve the mean: a rare-but-real
  velocity reversal still produces a near-instant heading flip when nothing
  physically prevents it.
* **Typical-case improvement** (`heading_rate_weight`, alone): `lossreg` cuts
  mean/p90/frac>180 by 3-8x over baseline — by far the largest such
  improvement of any variant — because it pressures the *policy* itself, not
  a downstream filter. But it doesn't bound the ceiling, and it has the
  steepest formation-quality cost of the three individual mechanisms tested
  (0.578, nearly double baseline's 0.322).

**Why `combined` doesn't just add both wins together:** `heading_history` is
recorded from the *already-capped* simulated heading (the cap runs inside
`sim.step` before the value is stashed for the loss to see later). Once
`max_turn_deg` is active, `heading_rate_loss` can only ever observe angular
speeds up to ~195 deg/s — the large uncapped excursions that `lossreg`-alone
was learning to suppress simply never appear in `combined`'s training
signal. So the two mechanisms compete for the same variable rather than
attacking independent failure modes: the cap "uses up" the loss's signal.
This is why `combined`'s typical-case numbers (mean 148.4, 25.3% >180) land
close to `yawcap`-alone's rather than splitting the difference toward
`lossreg`-alone's.

**Formation quality, ranked:** smoothing (0.246) > combined (0.310) ≈
baseline (0.322) > damping (0.387) > lossreg (0.578) ≈ yawcap (0.587). The
cap and the loss are both individually expensive; only combining the cap
with a *reduced* loss weight and moderately higher damping recovered
baseline-competitive formation quality — `damping_weight: 0.2` in particular
seems to be doing real work stabilizing what the cap and loss would
otherwise destabilize together (compare `combined`'s 0.310 to what a cap+loss
run without the extra damping might look like — not tested in this sweep).

## Recommendation

**Ship `combined`** (`max_turn_deg: 180`, `heading_rate_weight: 5e-6`,
`heading_rate_threshold_deg: 180`, `damping_weight: 0.2`) as the production
default for both cone-perception drone presets. Rationale:

* The hard ceiling is the property that actually matters for real
  deployment. A Crazyflie's yaw servo loop cannot execute an 1800 deg/s
  command *regardless* of how good the rest of the trajectory looks — a
  policy that "needs" that is undeployable no matter how favorable its mean/
  p90 statistics are. Only `max_turn_deg` variants provide this as a
  guarantee rather than a tendency, so any production config must include it.
* Of the two ceiling-bounding variants, `combined` recovers baseline-level
  formation quality (0.310 vs `yawcap`-alone's 0.587) at the cost of somewhat
  worse typical-case heading behavior than `lossreg`-alone achieves without
  a cap. That trade is the right one: an undeployable policy with excellent
  average behavior is still undeployable.
* `heading_smoothing: 0.08` (from `smoothing`) is cheap (zero new code, zero
  interaction risk) and produces the single best formation-quality number in
  the sweep. It's worth adopting **in addition to** `combined`'s settings —
  the sweep didn't test smoothing stacked with the cap+loss+damping
  combination, so this is a follow-up worth trying, not something already
  validated here.

**Follow-up worth trying, not done in this sweep** (to keep it bounded):
`combined` + `heading_smoothing: 0.08` together (cheap, likely composes
cleanly since smoothing acts upstream of everything else); and re-testing
`combined` with the loss weight restored toward `lossreg`-alone's 1e-5, now
that `damping_weight: 0.2` has demonstrated it can stabilize training under
multiple simultaneous secondary objectives — since the cap "using up" the
loss's signal (see above) means a higher weight might still have headroom to
push the *capped* typical-case numbers down further without necessarily
repeating the formation-quality cost `lossreg`-alone paid.

**Not applied to the circular-perception presets** (`config_drone_fixed.yaml`
/ `config_drone_walls.yaml`): heading only drives the cone sensing model;
circular perception ignores it entirely (`circular_adjacency` never reads
heading), so a fast-swinging heading arrow there is cosmetic, not a real yaw
demand on the physical drone. Retrofitting the fix would only smooth the
rendered arrow, not fix anything that affects real deployment.

## Production result

`config_drone_fixed_cone.yaml` and `config_drone_walls_cone.yaml` were updated
with the `combined` settings (`max_turn_deg: 180`, `heading_rate_weight: 5e-6`,
`heading_rate_threshold_deg: 180`, `damping_weight: 0.2`, up from 0.1) and
retrained at the full 2000-epoch production schedule. Both checkpoints and
their full 60s/5-seed results + animations were regenerated
(`results/drone_fixed_cone/`, `results/drone_walls_cone/`).

| | heading mean (deg/s) | heading max | formation error @60s (worst seed) | collision-free? |
|---|---|---|---|---|
| `drone_fixed_cone` (before) | 137–349 | ~1800 | 0.003–0.011 | yes |
| `drone_fixed_cone` (after) | **67–115** | **195** | 0.001–0.008 | yes |
| `drone_walls_cone` (before) | 137–352 | ~1800 | 0.11–0.88 | yes |
| `drone_walls_cone` (after) | **111–150** | **195** | 0.11–0.50 | yes |

Both checkpoints now hold the exact same hard ceiling verified in the sweep
(195 deg/s = 180 cap + 15 deg/s `self_rotation_deg`), with mean heading speed
roughly halved, and formation quality at least as good as before the fix (the
production runs used the full 2000-epoch schedule vs. the sweep's 1000, so
these aren't directly comparable to the sweep table above — they're a fresh,
independent confirmation that the fix holds up at full training length on
both arena methods, not just the `walls_cone` config the sweep was tuned on).
Both remain collision-free across every 60s multi-seed run and the
shape-switching transient.
