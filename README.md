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
| `losses.py` | distance-matrix MSE, velocity damping, optional Kabsch/Chamfer |
| `sim.py`    | shared Euler-integrator rollout (dispatches the perception model) |
| `train.py`  | BPTT over deployment-length horizons, formation-hold tail loss, replay cache, multi-shape training, checkpointing |
| `viz.py`    | the two animations (draws cone wedges or circular edges to match the checkpoint) |
| `compare.py`| trains cone vs circular and reports the head-to-head error |
| `sweep_range.py` | sweeps circular `sensing_range` values and reports the head-to-head error per range |
| `make_results.py` | writes a detailed `results/<mode>/` folder per checkpoint (drift tables/CSVs, heading metrics, loss curve) and the cone-vs-circular comparison in `results/` |
| `config.yaml` / `config.py` | all hyperparameters; every key is a CLI override |

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

* **Primary** — pairwise-distance-matrix MSE: compare the N×N inter-agent distance
  matrix of the realized configuration to the target shape's. Invariant to global
  rotation/translation, fully differentiable, no inner optimization. It is averaged
  over the last `hold_tail` rollout steps (not just the final state), so *staying*
  in formation is optimized, not just arriving — a trajectory that reaches the
  target and wobbles scores worse than one that parks there.
* **Damping regularizer** — small penalty on speed over the last few rollout steps,
  so the formation *parks* instead of drifting/spinning (the invariant loss alone
  treats every rotated/translated copy as equally correct, so it can't pin down
  rigid-body motion).
* **Stretch (`--use_chamfer true`)** — permutation-invariant Kabsch/Procrustes-aligned
  Chamfer loss, so any agent can fill any slot.

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
