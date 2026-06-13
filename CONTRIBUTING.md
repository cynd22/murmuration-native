# Contributing

- All flock behaviour and visuals live in `shaders/*.wgsl`, hot-reloaded on save —
  iterate there with the app running; broken WGSL logs and keeps the old pipeline.
- The force application order in `shaders/velocity.wgsl` is load-bearing
  (separation/alignment/cohesion before environmental forces, damping last).
  Don't reorder it without comparing the flock visually before and after.
- Rust code is a thin harness (window, buffers, fixed timestep) and rarely needs touching;
  tunables live in `src/params.rs` and `src/audio.rs`.
- Design rationale (audio mappings, colour layer, environmental forces) lives in the
  parent project: https://github.com/cynd22/murmuration-visualiser
