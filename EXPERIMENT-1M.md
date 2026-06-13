# Experiment: scaling toward 1M boids (NEGATIVE RESULT — do not merge)

Branch `experiment/1m-scaling`. Goal: push the sim toward 1M and find how far it
goes "without killing the logic." Conclusion: **a faithful 1M murmuration is not
interactive in this flock volume, and the changes that make flocking *honest* at
high N regress the 150k case we actually care about.** Master keeps the shipped
half-res sky (150k@4K = 180fps) and none of this.

## What was changed
- `GRID_Y` 8 → 16 (more cells, lower mean occupancy; Y cell still ≥ zone radius).
- `slots_for(n)` — bird-count-aware `SLOTS_PER_CELL` (was fixed 16).
- Adaptive timestep: `MAX_STEPS` 32 → 4 + one clamped catch-up step (kills the
  fixed-240Hz death-spiral).

## Measured (RTX 2070 SUPER, ~3840×2031, aurora on)
| config | birds | result |
|---|---|---|
| master (slots 16, GRID_Y 8) | 150k | **180 fps** |
| branch, slots→256 | 1M | 1.2 fps (1024 ms) — *worse* than master's 2 fps |
| branch, slots→256 | 150k | 9 fps |
| branch, slots cap 32 | 150k | **34 fps** (28 ms) — 5× worse than master |
| branch, slots cap 32 | 1M | ~5 fps |

## Why it fails — the O(N × density) wall, confirmed empirically
A murmuration is **densely clumped** (that's the point). Raising the slot cap so
birds see all their neighbours means dense-centre birds honestly visit 30–250
neighbours × 27 cells. The synthesis estimated cost from *mean* occupancy, but
the flock's density is wildly non-uniform — the centre cells dominate. So:

- **Cheap high-N = approximate flocking.** Master's 16-slot cap is what keeps
  150k/1M "running" — by silently dropping most neighbours in dense cells. At 1M
  that's ~97% dropped, which is also why 1M doesn't *look* like a murmuration
  regardless of fps.
- **Honest high-N = not interactive.** Visiting the real neighbour set at the
  real density is the fundamental O(N × density) cost: 1M → 1–5 fps.

There is no free lunch in a fixed volume. The only escape that keeps the logic
intact is a **larger flock volume** (raise `FLOCK_EXTENT` + box so 1M has 10k's
density) — but that changes the scale/look of the flock; it's an aesthetic
decision, not an optimisation. A fixed-K neighbour subsample makes 1M cheap but
degrades alignment/cohesion coherence — simplifying boids toward particles, the
one thing the project forbids.

## Verdict
- **150k is the sweet spot** and master already serves it well (180fps@4K after
  the half-res sky). Weak GPUs get the headroom.
- **1M is a stress ceiling, not a target.** Faithful 1M needs a bigger volume
  (a look change), not more optimisation.
- The **adaptive timestep** is the one idea here worth salvaging *on its own*
  (robustness against the death-spiral on weak GPUs) — but it must be isolated
  and re-tested without the slot/grid changes before it touches master, and it
  introduces variable-dt (cosmetic effect on freedom-noise/wing-phase).
