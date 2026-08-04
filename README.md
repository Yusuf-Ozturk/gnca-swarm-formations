# Four identical drones learning to fly in formation

Four drones, one shared control rule, no labels. A single learned function `gamma`
runs on every drone with the same weights; it sees only its neighbours' *relative*
positions and velocities, and it outputs a forward acceleration and a yaw
acceleration. A drone moves where its nose points. Nothing else is in the loop.

```
pip install -r requirements.txt
python train.py --shape wedge --perception cone     # one network, one formation
python results.py                                   # evaluate everything in runs/
python viz.py --checkpoint runs/gamma_wedge_cone.pt # animate one
```

## The rules this model obeys

These are hard constraints on the design, not defaults:

1. **The drones are perfectly identical.** No per-agent identity embedding, no
   index input, no per-agent parameter. Relabel the swarm and `gamma`'s outputs
   relabel with it — nothing distinguishes drone 0 from drone 3.
2. **No absolute information.** No drone is told its own position, anyone else's
   position, or any global reference. Every spatial quantity that reaches the
   network is a difference between two drones.
3. **`gamma` is the only controller.** No reactive safety filter, no geofence, no
   wall force, no hand-written fallback for any situation.
4. **Motion is non-holonomic.** Velocity is always `s · [cos θ, sin θ]`. To go
   somewhere else, a drone must turn first.
5. **Collision avoidance is a loss term**, not a runtime override.
6. **No drag.** Nothing damps the swarm for free; coming to rest has to be
   learned.

## The math

**State** per drone: position `p`, heading `θ`, forward speed `s ≥ 0`, yaw rate `ω`.

**Perception** — one of two range/bearing rules, the only difference between the
two arms of the study:

```
cone      i sees j iff  ‖pⱼ−pᵢ‖ ≤ R  and  ∠(headingᵢ, pⱼ−pᵢ) ≤ 75°   (150° FOV)
circular  i sees j iff  ‖pⱼ−pᵢ‖ ≤ R                                   (360°)
```

**The controller** (`model.py`), applied to every drone simultaneously each step:

```
mᵢ←ⱼ = φ_msg( [ R(−θᵢ)(pⱼ−pᵢ) , R(−θᵢ)(vⱼ−vᵢ) , ‖pⱼ−pᵢ‖ ] )
aᵢ    = mean_j mᵢ←ⱼ
a_lin, a_ang = φ_upd( [ aᵢ , sᵢ , ωᵢ ] )
```

`R(−θᵢ)` rotates into the receiver's **body frame**: +x is straight ahead, +y is
to its left. This is what makes a yaw command meaningful — a drone can only
decide to turn *toward* something if it knows where that thing is relative to its
own nose — and it makes the whole rule exactly rotation-equivariant. Combined
with the differencing, `gamma` is invariant to where the swarm is and which way
it is facing, by construction rather than by training. The only non-neighbour
inputs are `s` and `ω`, which is what an onboard IMU reads.

**Dynamics** (`sim.py`), semi-implicit Euler at `dt = 0.1 s`:

```
s     ← clip( s + dt·a_lin , 0 , 1.0 m/s )
ω     ← clip( ω + dt·a_ang , ±180 °/s )
θ     ← θ + dt·ω
p     ← p + dt·s·[cos θ, sin θ]
```

The clips are an **actuation envelope**, not control logic: they state what a
Crazyflie can physically do, and `gamma` may choose anything inside them.

**The objective** (`losses.py`) is two terms and nothing else:

```
total = formation + ramp · ( 10 · exposure below 0.30 m + 100 · exposure below 0.20 m )
```

*Formation* is a **Kabsch-aligned optimal-assignment MSE**. Because the drones
are identical, "drone k belongs at slot k" is not something a shared rule can
act on, so the target is a point *set*:

```
L = min over all 4! = 24 assignments π of
    mean_k ‖ R(θ*_π)(p_k − p̄) − (t_π(k) − t̄) ‖²
```

with `θ*_π` the closed-form optimal 2D rotation for that assignment
(`atan2(Σ cross, Σ dot)` — exact, no SVD, and it vectorizes over batch, time and
assignment together). Any position, any orientation, any assignment of drones to
positions is equally correct; only the geometry is graded. It is averaged over
the last 10 steps, so the formation is measured where it has to *hold*.

*Collision* is violation exposure: the squared penetration depth below a distance,
averaged over pairs and **summed** over time × dt — "pair-seconds spent too
close." Summed rather than averaged because a time-average divides a brief
mid-flight collision by the whole horizon, which would make crashing cheaper than
the detour that avoids it. Two tiers (a soft buffer at 0.30 m, an actual collision
at 0.20 m priced 10× higher) ramp in over the first half of training: at full
weight from epoch 0 they dominate, and the swarm learns to keep apart before it
can form anything at all.

Note what is *absent*: no damping term, no bounds term, no heading regularizer.
If the swarm parks, `gamma` found that equilibrium on its own — `results.py`
measures residual speed and yaw rate precisely because that is now an empirical
question rather than something the loss handed over.

## Trained vs preset

**Trained** — ~50.8k parameters, one network per formation:

| tensor | shape |
|---|---|
| `phi_msg` | 5 → 128 → 128 |
| `phi_upd` | 130 → 128 → 128 |
| `out` | 128 → 2 |

**Preset** — the target point sets (analytic, `shapes.py`), the perception rule,
`dt`, the actuation envelope, and the loss weights. Nothing else.

## The formations

Four point sets designed for N=4 (`python shapes.py` prints and checks them).
Because the loss is pose-invariant, presets must be distinct *up to rotation* —
a diamond would be the same object as a square, so there isn't one:

| shape | tightest pair | note |
|---|---|---|
| `square` | 1.60 m | four corners |
| `line` | 0.80 m | four collinear, evenly spaced |
| `wedge` | 0.72 m | a V: leader plus three trailing, necessarily asymmetric at N=4 |
| `triangle_centroid` | 0.80 m | equilateral triangle with a drone at the centre |

Every one is collision-free by construction — the tightest is 3.6× the 0.20 m
collision distance — so a collision is always a formation failure, never an
artefact of an impossible target.

## Repo layout

| file | role |
|---|---|
| `model.py` | `gamma`: body-frame message passing → (forward accel, yaw accel) |
| `sim.py` | non-holonomic state and integrator; the actuation envelope |
| `graph.py` | cone and circular sensing; heading is a state, not a derived quantity |
| `shapes.py` | the four N=4 target point sets |
| `losses.py` | assignment loss + two-tier collision exposure |
| `train.py` | BPTT over a randomized horizon, fresh inits only, collision curriculum |
| `results.py` | evaluation: formation, collisions (all steps *and* final), equilibrium |
| `viz.py` | animation with heading arrows, safety disks, and collision marking |
| `config.yaml` / `config.py` | every hyperparameter; each key is a CLI override |

## Results

The study is 4 shapes × {cone, circular} = 8 networks trained under identical
settings. Generated results — per-run summaries, the head-to-head comparison, the
final-state collision verdict, and one animation per run — live in
[`results/`](results/README.md), written by `python results.py`.
