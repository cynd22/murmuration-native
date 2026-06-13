# NEXT — topological flocking (the path to 1M)

**Status: planned, not started. Resume here.** Written 2026-06-13 so the plan
survives a context reset.

## Why
Metric-distance flocking (the current build) is **O(N × local density)**. A
murmuration clumps, so at high N each bird honestly sees hundreds of neighbours
→ quadratic blow-up. Measured this session: honest 1M = 1–5 fps, and even a
modest slot-cap bump regressed 150k from 180 → 34 fps. Full data in
[EXPERIMENT-1M.md](EXPERIMENT-1M.md).

**Topological flocking** — each bird interacts with its **k nearest neighbours**
(k≈7, the real-starling result) instead of everyone within a radius — caps work
per bird at a constant → **linear in N**. It's the principled door to 1M and is
arguably more biologically accurate. It IS a behavioural change, so keep the
metric version and A/B with your eyes (project rule: don't iterate blind).

## Plan (incremental — each step verifiable in isolation; DO NOT skip the gates)

1. **Candidate gathering, exposed.** Reuse the existing 3×3×3 grid stencil
   (already in `shaders/velocity.wgsl`, the 27-cell loop) to emit, per bird, its
   cell + 26 neighbour candidate set. Step one is just *surfacing* the read
   pattern you already have. **Verify on its own:** does each bird see the right
   candidates? (debug count / spot-check a known bird).

2. **k-buffer select** — the one genuinely new primitive, and it's small. Fixed
   7-slot insertion, branchless-ish, keep the nearest by distance. **Test
   STANDALONE** with known inputs (a handful of points → assert the 7 nearest)
   before it touches flocking.

3. **Swap metric → topological.** separation / alignment / cohesion now average
   over the k selected neighbours instead of everyone-in-radius. **Keep BOTH**
   (runtime toggle), A/B against the metric version, watch the flock with your
   eyes.

4. **Boundary cases.** Fewer than 7 candidates at flock edges; distance ties.
   This is where a first pass goes subtly wrong — give it its own pass.

## Implementation pointers
- Read pattern: `shaders/velocity.wgsl` (the `for dx/dy/dz` 27-cell loop). The
  grid (`grid.wgsl` / `params.rs`) already gives cell→bird lookup. Don't rebuild it.
- Add a runtime toggle `metric | topological` (mirror the upperMid
  relative/absolute A/B pattern in `audio.rs`) so you can flip live, and expose
  `k` (default 7) in the egui panel for visual tuning.
- Decouple two things that are currently fused: the grid cell size still needs a
  spatial scale for *candidate gathering*, but the *force* no longer needs a
  metric radius. So the old "cell_size ≥ zone_radius" constraint relaxes — but
  candidates_scanned per bird must stay small for 1M to be linear, so revisit
  grid resolution alongside this.
- The `experiment/1m-scaling` branch holds the adaptive-timestep + GRID_Y
  changes (slots scaling was a negative result; the **adaptive timestep is worth
  salvaging to master on its own** — see below).

## How to run this next session — RAW, not heavy orchestration
This task is **small, sequential, and eyes-in-the-loop** (you judge the flock
visually at step 3). That's the opposite of what subagents/workflows are good at
(breadth, parallel fan-out, adversarial verification of many findings). So:
**use Claude raw, one step at a time, honouring the verification gates above.**
The only plausible subagent is vetting the step-2 k-select WGSL in isolation, but
it's small enough to just write + standalone-test inline. **Do NOT reach for
ultracode here** — it's the wrong tool for narrow, gated, sequential work. (The
earlier ultracode use was right *because* it was broad parallel perf analysis;
this isn't.)

## Other open threads (separate from the 1M task)
- **Laptop PipeWire capture fix:** on a pure-PipeWire box `PULSE_SOURCE` is a
  no-op (libpulse-only); must also set `PIPEWIRE_NODE=<sink node>` (verified on
  the AZenbook — it links to the monitor ports). The fix is currently stranded
  on the laptop clone — consolidate into canonical `src/capture.rs` and push.
- **Laptop RADV segfault:** crashes on the AMD RENOIR iGPU. Bisect with
  `--no-sky` (skips the fluid sim + most of the sky shader), `coredumpctl gdb`
  for the crash frame, check `mesa` / `vulkan-radeon` version. Prime suspects:
  the fluid (`rgba16float` compute), the half-res offscreen sky target, or a
  mesa driver bug.
- **Adaptive timestep → master**, isolated from the slot change that regressed
  it: lets weak GPUs degrade to a steady lower framerate instead of stuttering.
