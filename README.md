# Murmuration — Rust/wgpu port

Native port of `index_onsets.html`. Same boid forces, same dark-trough colour
layer, same camera — but compute shaders instead of fragment-shader GPGPU.

## Run

```sh
cd rust-port
cargo run                          # 10,000 birds, vsync, sim 240 Hz
cargo run -- --birds 50000         # more birds
cargo run -- --uncapped            # no vsync (fps benchmark / 240Hz displays)
cargo run -- --sim-hz 120          # different fixed sim tick rate
```

Esc quits. FPS lives in the title bar (no UI chrome over the visual).

## Measured on the 2070 SUPER (2026-06-13, uncapped)

| Birds | Render fps |
|---|---|
| 10,000 | ~2,300 |
| 50,000 | ~1,180 |

The WebGL build capped out around 60 fps at 10k birds. The win comes from the
grid build: compute shaders can scatter with atomics (O(birds)), where the
fragment-shader version had every cell scan every bird (O(cells × birds),
~327M texture reads/frame at 10k).

## How iteration works (no compile-time loop)

- **All forces and visuals are WGSL files in `shaders/`, hot-reloaded on save.**
  Edit `velocity.wgsl` while the app runs — the flock changes in well under a
  second. Broken WGSL logs the error and keeps the old pipeline; the app never
  dies on a typo.
- **Tunables** live in `src/params.rs` (`Settings::default()`) for now —
  changing those does need a rebuild (~3 s). They're slated to move to an egui
  panel + TOML preset in a later stage.
- Rust code is a thin harness (window, device, buffers, fixed-timestep loop)
  and rarely needs touching.

## Architecture notes

- **Fixed-timestep sim (default 240 Hz), decoupled from render rate.** The
  flock behaves identically on a 60 Hz and a 240 Hz display; uncapped render
  just shows more frames of the same trajectory. Velocity damping is converted
  per-tick (`0.98^(60·dt)`) so drag-per-second is rate-independent — the HTML
  build's damping was per-frame and would have behaved differently at other
  framerates.
- **Per tick:** clear grid counters → grid build (atomic scatter) → velocity
  (3×3×3 cell stencil, all forces) → position (integrate + wing phase).
  Position/velocity are double-buffered.
- **Non-sRGB surface format on purpose.** The HTML build wrote raw shader
  values into an sRGB-interpreted canvas; doing the same here reproduces its
  exact look, including the dark trough.
- `Params.time` is **milliseconds** (the freedom-noise hash constants were
  tuned against `performance.now()` ms in the HTML build).
- Spatial grid is 16×8×16 cells × 16 slots (HTML parity). Average density at
  50k birds is ~24/cell — past ~32k birds, raise `SLOTS_PER_CELL` in
  `src/params.rs` or dense regions lose local cohesion (same safety-valve
  semantics as the HTML build).

## Audio

Run `feeder_onsets.py` (same one the HTML uses); the app connects to
`ws://localhost:8766` automatically, reconnects every second if the feeder
isn't up, and shows `audio ✓/✗` in the title bar. Override with `--ws <url>`.

The full mapping layer is ported (`src/audio.rs`): subBass→attract/separation/
freedom(inverse)/verticality, treble+upperMid→maxSpeed, upperMid→turn-rate+shear,
mid→alignment, lowMid→cohesion, bass→paletteT (dark-trough swell + softened
kick punctuation), air→twinkle (off by default), bass-onset→shockwave (off by
default). All floors/ceilings/alphas are the HTML build's Kiseia-tuned values.
Smoothing alphas are converted per tick (`1-(1-α)^(60·dt)`) so envelope time
constants are identical at any sim rate.

## Status / roadmap

- [x] Stage 0–1: scaffold, GPU sim, bird+ground rendering, hot reload, fixed timestep
- [x] Stage 2: feeder websocket client + smoothing layer + full band→uniform mappings
- [ ] Stage 3: egui control panel (`H` to hide) + TOML presets (mapping
      tunables currently live in `src/audio.rs` `Mapping::default()`)
- [ ] BPM/beat phase: add to the *feeder* (aubio via Arch's `python-aubio` —
      PyPI aubio is dead on Python 3.14; recreate the feeder venv with
      `--system-site-packages`), so HTML and Rust clients both get it through
      the same socket. Carry bpm / phase / nextBeatIn / confidence.
