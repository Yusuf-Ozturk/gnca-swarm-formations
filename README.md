# GNCA Swarm Formations

Train a **Graph Neural Cellular Automaton (GNCA)** to drive a 2D swarm of agents
into target formation shapes — and switch the target shape *at runtime* without
resetting the swarm.

A single shared update rule `gamma` (a message-passing GNN layer + per-node MLP) is
applied synchronously to every agent for `t` steps. It outputs each agent's
acceleration; velocity and position integrate with simple Euler steps. The
interaction graph is rebuilt every timestep from a **forward field-of-view cone**.
One network handles **all** shapes — the target is selected by a learned per-shape
latent code injected via **FiLM**.

```
pip install -r requirements.txt
python train.py      # trains all presets, saves checkpoint.pt  (~1.5 h on CPU)
python viz.py        # writes convergence_*_cone.gif and switching_cone.gif
python compare.py    # trains cone vs circular perception and charts the difference
```

## What you get

* `convergence_<shape>_<perception>.gif` — from a random init, the swarm forms each
  preset (square, hexagon, triangle, line), with each agent's heading arrow and its
  perception drawn (cone wedges for the forward FOV, neighbour edges for the
  circular FOV).
* **`switching_<perception>.gif`** — the key demo: the swarm forms a **square**, then
  at step 30 the shape code is swapped to **hexagon** *without resetting positions*,
  and it re-converges.
* `loss_curve.png` — training loss, plus per-shape final distance-matrix error
  printed at the end of training.
* `compare_perception.png` — grouped bar chart of cone-FOV vs circular-FOV error per
  shape.

## Perception models: forward-FOV cone vs circular (360-degree) FOV

There are **two switchable vision models**, selected by `--perception`, and both are
pure **range/bearing sensing rules**: an edge exists only if agent j is physically
within agent i's sensing radius (and, for the cone, within its forward angular
sector). That is what a real drone's onboard sensor (camera FOV, lidar, UWB ranging)
can actually measure locally.

* **`cone`** (default) — each agent sees only agents inside an angular sector
  *ahead* of its own heading (`half_angle_deg`, `sensing_range`). Directed,
  asymmetric, and partially observed: a meaningful fraction of agents have an
  *empty* cone at any step and act on their own state alone.
* **`circular`** — each agent sees every agent within `sensing_range`, in every
  direction, omnidirectional and heading-independent (equivalently, a cone with a
  180-degree half-angle). Uses only inter-agent distances, so it is
  rotation/translation invariant just like the cone — the *only* thing that changes
  between the two is **what each agent may see**.

There is deliberately **no kNN mode**: connecting to a fixed count of "k nearest"
agents regardless of how far away they are isn't something a drone's sensor can do
locally — a real sensor only knows *who is within range*, not *global rank order
among everyone else*. `circular` is the physically-meaningful, distance-based
replacement for that omnidirectional baseline.

```bash
python train.py --perception cone                          # the hard, partially-observed setting
python train.py --perception circular --sensing_range 1.0  # the omnidirectional baseline
python compare.py                                           # train both, print table + bar chart
python viz.py --checkpoint checkpoint_circular.pt           # animate the circular model (draws edges)
```

**What the comparison shows:** see [`results/README.md`](results/README.md) for the
current head-to-head numbers (formation quality, 60s hold stability, and heading
noise for both modes under identical training), with full per-seed drift data under
[`results/cone/`](results/cone/summary.md) and
[`results/circular/`](results/circular/summary.md). The circular sensor hands every
agent full local proximity information, so it forms near-perfect shapes; the cone
reaches clearly *recognizable* but looser formations. This is the expected and
interesting result: the cone is a genuinely hard partial-observability problem,
forward-only and often blind. The interesting research question the repo is set up
to probe is *how close a forward-FOV swarm can get to the circular-FOV ceiling* —
try widening the cone (`--half_angle_deg`), shrinking the sensing range, or
training longer. Regenerate the folder with:

```bash
python make_results.py --checkpoint checkpoint.pt            # -> results/cone/
python make_results.py --checkpoint checkpoint_circular.pt   # -> results/circular/
python make_results.py --compare                             # -> results/README.md
```

## Repo layout

| file | role |
|------|------|
| `model.py`  | `gamma`: relative-coordinate message passing + FiLM shape conditioning + per-agent identity → acceleration |
| `graph.py`  | cone (forward-FOV) **and** circular (360-degree FOV) edge construction + persistent-heading fallback |
| `shapes.py` | preset target point sets and their pairwise-distance matrices |
| `losses.py` | distance-matrix MSE, fixed-target MSE, collision-avoidance + containment hinges, velocity damping, optional Kabsch/Chamfer |
| `sim.py`    | shared Euler-integrator rollout (dispatches the perception model; repulsive-wall force in `walls` arena mode) |
| `train.py`  | BPTT over deployment-length horizons, formation-hold tail loss, replay cache, multi-shape training, checkpointing |
| `viz.py`    | the two animations (draws cone wedges or circular edges to match the checkpoint) |
| `compare.py`| trains cone vs circular and reports the head-to-head error |
| `sweep_range.py` | sweeps circular `sensing_range` values and reports the head-to-head error per range |
| `make_results.py` | writes a detailed `results/<mode>/` folder per checkpoint (drift tables/CSVs, heading metrics, per-run collision/bounds checks, loss curve) and the cone-vs-circular comparison in `results/` |
| `config.yaml` / `config.py` | all hyperparameters; every key is a CLI override |
| `config_drone_fixed.yaml` / `config_drone_walls.yaml` | drone-deployment presets for the 3m×3m Crazyflie arena (issue #4), one per arena method |

## The four non-obvious parts (all commented in code)

1. **Relative-coordinate equivariance** (`model.py`). Each edge `(i,j)` sees only
   `pos_j - pos_i` and `vel_j - vel_i`, never absolute coordinates. Differencing
   removes the global origin → translation-invariant by construction; with no global
   frame anywhere in the rule, the rotation-invariant loss does the rest.

2. **FiLM conditioning** (`model.py`). A per-shape latent code `z_shape` (an opaque
   learned vector, **not** the target coordinates) is mapped to `(scale, shift)` and
   modulates a hidden layer of the post-aggregation MLP: `h ← scale*h + shift`.
   Switching shapes = swapping `z_shape`. Adding a new shape later only needs a new
   `z_shape`, not retraining `gamma`.

3. **Replay cache** (`train.py`). Rollout end-states are stashed and some training
   samples are re-seeded from them, so the model learns to keep refining
   near-converged states (a stable fixed point, not a jittery orbit). Fresh random
   inits are always re-injected so it never forgets forming from scratch.

4. **Persistent-heading fallback** (`graph.py`). The cone is defined relative to
   each agent's *own* heading. At convergence velocities → 0, so the instantaneous
   heading is undefined. Each agent keeps a persistent heading that updates from
   velocity only when speed > ε, and otherwise holds its last valid value. Because
   this network corrects overshoot like a free point-mass (not a fixed-nose
   vehicle), raw instantaneous velocity can flip direction almost instantly and is
   too noisy a signal to build heading from directly — so heading is derived from
   an EMA-smoothed velocity (`heading_smoothing` in `config.yaml`) instead of raw
   velocity, fixing the noise at its source rather than rate-limiting the heading
   that's derived from it.

## A fifth design note: agent identity

The primary loss assigns agent `k` to a fixed labeled slot `k`. A purely anonymous
shared rule is permutation-symmetric and literally cannot decide which agent becomes
which slot — it averages over the ambiguity and never forms a clean shape (we
verified this: loss plateaus). So each agent carries a small **constant learned ID
embedding**, fed into its own features and attached to every message it sends, so a
neighbour can recognise *who* it sees and triangulate its own target. IDs are
constant scalars, not coordinates, so translation/rotation behaviour is unaffected.

## Loss

The whole objective (see `training_loss` in `train.py`):

```
total = 1.0 * formation + damping_weight * damping
      + separation_weight * separation                 # collision avoidance, issue #4
      + bounds_weight * bounds                         # arena_mode == "fixed" only
```

* **Primary** — pairwise-distance-matrix MSE: compare the N×N inter-agent distance
  matrix of the realized configuration to the target shape's. Invariant to global
  rotation/translation, fully differentiable, no inner optimization. It is averaged
  over the last `hold_tail` rollout steps (not just the final state), so *staying*
  in formation is optimized, not just arriving — a trajectory that reaches the
  target and wobbles scores worse than one that parks there. (With
  `arena_mode: fixed` the primary term is instead a plain coordinate MSE against
  the targets at their fixed arena-centered location — see the drone section below.)
* **Damping regularizer** — small penalty on speed over the last few rollout steps,
  so the formation *parks* instead of drifting/spinning (the invariant loss alone
  treats every rotated/translated copy as equally correct, so it can't pin down
  rigid-body motion).
* **Separation (collision avoidance, issue #4)** — hinge penalty
  `relu(2*drone_radius + separation_margin − dist)²` on **every pair at every
  rollout step** (a mid-transit crash breaks a drone just as surely as one in
  formation). Exactly zero once all pairs keep safe distance, so it never fights
  the formation term at convergence.
* **Bounds (`arena_mode: fixed` only)** — exposure penalty (same time-integrated
  construction as separation) for any drone whose safety disk crosses the
  flight-area boundary during transit, backed by a hard geofence clamp at the
  arena edge (`hard_bounds`, the containment guarantee). Unneeded in `walls`
  mode, where the repulsive boundary force lives in the physics instead.
* **Stretch (`--use_chamfer true`)** — permutation-invariant Kabsch/Procrustes-aligned
  Chamfer loss, so any agent can fill any slot (invariant modes only).

## Drone deployment: the 3m×3m Crazyflie arena (issue #4)

The physical target is a swarm of Bitcraze Crazyflies flying under a lighthouse
positioning system in a **3m×3m flight area centered on the origin**. Units are
literal: **1 simulation unit = 1 meter, `dt` is seconds** (0.1 = 10 Hz), and each
drone is a 10cm×10cm quad modeled as a **10cm-radius safety disk**
(`drone_radius: 0.1` — the disk fully contains the body, whose half-diagonal is
7.1cm). A **collision** is two centers closer than `2*drone_radius` = 0.2m.

Two *independent* ways to keep the swarm inside the walls, selected by
`arena_mode` (each has a ready preset):

```bash
python train.py --config config_drone_fixed.yaml    # method 1 -> checkpoint_drone_fixed.pt
python train.py --config config_drone_walls.yaml    # method 2 -> checkpoint_drone_walls.pt
python make_results.py --config config_drone_fixed.yaml --checkpoint checkpoint_drone_fixed.pt --label drone_fixed
python make_results.py --config config_drone_walls.yaml --checkpoint checkpoint_drone_walls.pt --label drone_walls
```

* **Method 1 — `arena_mode: fixed`**: the goal shape is **fixed and centered in
  the arena**; the loss is coordinate MSE against those fixed points (no
  position/rotation invariance), plus a containment hinge on transit. Because a
  translation-invariant rule physically *cannot* steer to an absolute spot, this
  mode appends each agent's own absolute position to its inputs (`model.py`,
  note 4) — exactly the measurement lighthouse provides.
* **Method 2 — `arena_mode: walls`**: the loss stays fully
  position/rotation-invariant (form the shape anywhere), and each wall pushes
  drones toward the center with acceleration `wall_strength * (1/d − 1/wall_margin)`
  once a drone is within `wall_margin` of it (`sim.wall_accel`) — inverse
  proportional to wall distance, continuous at onset, divergent at the wall, so
  the swarm **bounces** off the boundary. The force is differentiable, so BPTT
  trains straight through the bounces.

**Collision avoidance** is layered, mirroring how a real deployment stacks
defenses:

1. **The loss (primary)** — the two-tier separation objective above shapes whole
   trajectories so close approaches become rare in the first place, plus
   collision-free takeoff spacing (`min_start_dist: 0.5`).
2. **A reactive safety filter (the guarantee)** — `safety_filter: true` in the
   drone presets adds the closing-velocity filter from `sim.apply_safety_filter`:
   for any pair closer than 0.45m, a ramped fraction of the pair's *closing*
   velocity is cancelled symmetrically, reaching full cancellation at 0.25m, so
   the 0.2m collision distance is never breached (worst-case head-on at
   2 m/s closing bottoms out at 0.25m; a 4-way max-speed pileup at 0.28m).
   Tangential motion passes through untouched, so forming/reshuffling is
   unaffected; separating pairs are never touched. Training runs *through* the
   filter (it is differentiable), so the policy learns to cooperate with it —
   exactly like the onboard reactive layer you would (and should) run on the
   real Crazyflie commander beneath any learned controller. Disable it with
   `--safety_filter false` to measure what the loss achieves alone: across
   many training recipes, loss-only converged to *rare, brief* launch-window
   grazes but never to strictly zero — physical drones need zero.
3. **Verification on every result run** — `make_results.py` records, for every
   shape and seed (and the shape-switching transient), the minimum pairwise
   separation over the whole 60s rollout, collision steps, peak speed, and (for
   arena modes) out-of-bounds steps — `results/<label>/summary.md` prints the
   table with an explicit **COLLISION-FREE / VIOLATIONS verdict**, and
   `train.py` prints the same check after training. Animations draw each
   drone's safety disk (it flashes red on contact) plus the arena walls, so a
   violation is also visible at a glance.

The drone presets use `n: 8` and `shape_scale: 0.8` so every preset shape fits
the arena with ≥0.3m wall clearance and every pair of target slots is ≥0.34m
apart (comfortably above the 0.28m separation-loss distance), and `perception:
circular` — collision avoidance with a forward-only cone is unsafe (an agent
can't see a neighbour approaching from behind), and omnidirectional relative
sensing is what lighthouse positions shared over the Crazyflie radio actually
give you. They also cap per-drone speed at `max_speed: 1.0` m/s inside the
simulator itself: that is both a real Crazyflie actuation limit (so every
reported trajectory is flyable) and what keeps training through the wall force
stable — an uncapped agent can cross the whole `wall_margin` band in a single
`dt` and slingshot out of the arena. Formation *hold* over the deployment window is already part of the
objective on this branch (issue #2): training rollouts span the real 4–12s
window (`t_min=40`–`t_max=120`) with the formation-hold tail loss, and the 60s
drift tables in `results/` measure it directly.

## Common overrides

```bash
python train.py --epochs 1200 --n 12 --half_angle_deg 45 --sensing_range 1.4
python train.py --use_chamfer true
python viz.py --switch_from triangle --switch_to line --switch_step 25
```

Run `python train.py --help` to see every exposed parameter.

## Notes / limitations

* Defaults converge to clearly **recognizable** (not pixel-perfect) formations;
  train longer (`--epochs` above 3000) for crisper shapes.
* Training rollouts span the real deployment hold window (`t_min=40` to `t_max=120`
  steps = 4-12 s at `dt=0.1`), so holding a formation over that window is part of
  the objective itself rather than extrapolation past a short training horizon —
  the target deployment (a Crazyflie swarm holding formation for 5-10 s) sits
  inside what BPTT directly optimizes. Shorter horizons (the old `t_max=40`) train
  ~3x faster but the formation quietly drifts apart after ~10-15 s (issue #2).
* The cone FOV is a genuinely hard, partially-observed perception setting — ~20% of
  agents have an empty cone at any step (they then act on their own state only,
  which is expected, not a bug). If training ever struggles, that's the first thing
  to inspect.
* `gamma`'s message MLP gives **exact** translation invariance; rotation is handled
  by removing the global frame plus the rotation-invariant loss (a generic MLP on
  relative vectors is not exactly rotation-equivariant — an EGNN-style radial
  message would be, at the cost of simplicity).
