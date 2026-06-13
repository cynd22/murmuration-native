# Murmuration

An audio-reactive 3D starling murmuration, in native Rust + WebGPU. Tens of
thousands of GPU boids flock in real time and the music *shapes* how they move —
the flock tightens on kicks, carves on guitar solos, disperses when things go
quiet, and banks through its turns so dark "agitation bands" ripple across the
mass — all over an audio-reactive fluid sky that billows in time with the beat.

It's the native port of a single-file WebGL visualiser, rebuilt on compute
shaders: ~50,000 birds at well over 1000 fps on a mid-range GPU, where the
browser version capped out around 60.

![murmuration](https://github.com/cynd22/murmuration-native) <!-- add a screenshot/gif here -->

---

## Download (no build needed)

Grab the latest **`murmuration-linux-x86_64.tar.gz`** from the
[Releases page](https://github.com/cynd22/murmuration-native/releases), then:

```sh
tar xzf murmuration-linux-x86_64.tar.gz
cd murmuration
./murmuration
```

Keep the `shaders/` folder next to the `murmuration` binary — the tarball already
has them together. To *run* (not build) you only need your GPU's **Vulkan driver**;
the other libraries it uses (libxkbcommon, ALSA, D-Bus) ship with almost every
desktop already:

- **Arch:** `sudo pacman -S --needed vulkan-icd-loader vulkan-intel` *(Intel)* — or `vulkan-radeon` *(AMD)* / `nvidia-utils` *(NVIDIA)*
- **Debian / Ubuntu / Mint:** `sudo apt install libvulkan1 mesa-vulkan-drivers`
- **Fedora:** `sudo dnf install vulkan-loader mesa-vulkan-drivers`

If it won't launch, run it **from a terminal** so you can read the error (and on a
weak/integrated GPU, start it with `./murmuration --sky-div 4` or `--no-sky` — see
[Controls](#controls)). Full library list is in [BUILDING.md](BUILDING.md).

---

## Build from source (also quick)

You need three things: the Rust toolchain, a few system libraries, and (optionally)
the audio feeder. Copy-paste the block for your distro.

### 1. Install Rust

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# accept the defaults, then restart your terminal (or run: source "$HOME/.cargo/env")
```

### 2. Install system libraries

**Arch / Manjaro / EndeavourOS:**
```sh
sudo pacman -S --needed base-devel libxkbcommon wayland vulkan-icd-loader
# then your GPU's Vulkan driver:
#   NVIDIA:        sudo pacman -S nvidia-utils
#   AMD / Intel:   sudo pacman -S vulkan-radeon   # AMD
#                  sudo pacman -S vulkan-intel     # Intel
```

**Debian / Ubuntu / Pop!_OS / Mint:**
```sh
sudo apt update
sudo apt install build-essential libxkbcommon-dev libwayland-dev \
                 libvulkan1 mesa-vulkan-drivers
# NVIDIA users: also install the proprietary driver via your Driver Manager
```

**Fedora:**
```sh
sudo dnf install @development-tools libxkbcommon-devel wayland-devel \
                 vulkan-loader mesa-vulkan-drivers
```

### 3. Build and run

```sh
git clone https://github.com/cynd22/murmuration-native.git
cd murmuration-native
cargo run --release
```

The first build downloads and compiles dependencies and takes a few minutes —
that's normal, it's a one-time cost. After that it launches in a second or two.

> **Just play music and it reacts** — no setup, no extra process. The
> visualiser captures your system audio itself (see below). With nothing
> playing you get a calm, drifting flock.

Full per-distro detail, NVIDIA/Optimus troubleshooting, and CI build notes live
in [BUILDING.md](BUILDING.md).

---

## Audio — it just works

The visualiser captures your **system audio** directly (via cpal) — whatever
you're playing through your speakers/headphones: Spotify, a browser, a local
player, anything. No feeder, no Python, no config. On PipeWire/PulseAudio it
auto-selects your output's *monitor* source, so the title bar shows `audio ✓`
the moment something plays.

If it ever grabs the wrong input (e.g. a mic on an unusual setup):

```sh
cargo run --release -- --list-devices         # see inputs; monitors are marked
cargo run --release -- --device "<substring>" # force a specific one
```

Fallbacks if no monitor is auto-found: `pactl set-default-source $(pactl get-default-sink).monitor`,
or pick "Monitor of …" in `pavucontrol` → Recording.

### Now playing

A bottom-right card shows the current track (title, artist, album, album art) with
a seekable position bar and prev / play-pause / next buttons — driven over **MPRIS**,
so it works with any Linux player (VLC, mpv, browsers, Spotify, …) and controls
playback for real. Press **P** to toggle it (independent of the H control panel).

---

## Controls

Press **H** to show/hide the control panel and all chrome — during playback
there's nothing on screen but the flock and the sky. Press **P** to toggle the
now-playing card on its own.

The panel has live sliders for:

- **flock size** — 1k to 150k birds, applied on demand (respawns the flock)
- **solo response (upperMid)** — the guitar-solo dimension, with live signal
  bars so you can *see* the flock deciding to carve (see below)
- **audio mappings** — every band's floor/ceiling (attract, separation, speed,
  alignment, cohesion, freedom, vertical lift, shockwave, twinkle)
- **colour** — palette, dark-trough depth, bass swell, kick punctuation
- **sky** — procedural clouds, stars, beat pulse, and the **fluid sky** (amount,
  heat glow, injection strength, swirl, cloud lifetime)
- **camera** — height, distance, FOV
- **realism** — bird **banking** (roll into turns) and dark **agitation bands** that sweep the flock through hard turns

Command-line flags:

```sh
cargo run --release -- --birds 100000    # bigger flock (default is 50,000)
cargo run --release -- --uncapped        # disable vsync (benchmark / >60Hz displays)
cargo run --release -- --sky-div 4       # cheap quarter-res sky (weak / integrated GPUs)
cargo run --release -- --no-sky          # start with the sky off entirely
cargo run --release -- --sim-hz 120      # change the fixed simulation tick rate
cargo run --release -- --list-devices    # list audio inputs and exit
cargo run --release -- --device "monitor"  # force a capture device by name
cargo run --release -- --ws ws://host:8766 # advanced: use an external feeder instead of native capture
```

`Esc` quits.

---

## What makes it react the way it does

The whole design principle is **transform the audio, don't normalise it** — every
reactive element is a function of the music, so different songs genuinely look
different rather than converging on one "visualiser look."

- **The flock** — subBass drives attraction/separation/freedom (kicks pull the
  murmuration tight, then it breathes back out), treble drives speed, mid drives
  heading-unity, lowMid drives cohesion. Colour rides the bass with a dark
  "trough" so the birds go near-black in quiet passages and ignite on drops.
- **Guitar solos (the upperMid fix)** — absolute treble/upper-mid level carries
  almost no solo information in a dense mix (vocals, rhythm guitar and cymbals
  keep that band permanently lit). So the solo axes — how hard the flock turns
  and stretches into filaments — are driven by **contrast above the song's own
  rolling baseline**: a solo *departing* from what the track has been doing,
  which self-calibrates per song. There's an A/B toggle to the old absolute mode
  and live bars showing envelope vs. baseline vs. drive.
- **The fluid sky** — a real 2D stable-fluids simulation (advection + vorticity
  confinement + Jacobi pressure solve) on the GPU, but with **no energy of its
  own**: subBass kicks fire rising thermal plumes from the horizon, treble
  onsets stir little vortices, mid is the wind, and when the beat tracker is
  confident the plumes *pump in beat phase*. Silence → the sky settles and
  fades. The dye is lit by the same palette as the birds, so sky and flock share
  one colour identity and the dark trough owns the whole frame.
- **Aurora borealis** — an optional corona-view aurora across the top of the sky
  (real green→teal→magenta colours, rays converging overhead, driven by the same
  fluid field + bass/beat). Independent on/off from the plumes; both in the sky
  panel.
- **Beat tracking** — an autocorrelation tempo tracker estimates BPM/phase/
  confidence from the onset novelty; the visualiser breathes the horizon glow and
  pumps the fluid plumes on the beat, and ignores it gracefully on beatless/
  ambient material (confidence falls to zero).

---

## Performance (RTX 2070 SUPER, uncapped)

| Birds | Render fps |
|---|---|
| 10,000 | ~2,300 |
| 50,000 | ~1,180 |

The win over the WebGL build is the spatial-hash grid: compute shaders scatter
each bird into its cell with atomics (O(birds)), where the fragment-shader
version had to scan every bird for every cell (O(cells × birds) — ~327M texture
reads/frame at 10k).

---

## Hacking on it

**Everything visual is a hot-reloaded WGSL file in `shaders/`.** Edit
`velocity.wgsl` (flocking forces), `bird.wgsl` (bird shape + colour),
`background.wgsl` (sky), or `fluid.wgsl` (the fluid sim) while the app is
running and the change appears in well under a second. Broken WGSL logs the
error and keeps the previous shader — the app never dies on a typo. The Rust
side is a thin harness (window, device, buffers, the fixed-timestep loop) you
rarely need to touch. See [CONTRIBUTING.md](CONTRIBUTING.md).

### Architecture notes

- **Fixed-timestep sim (default 240 Hz), decoupled from render rate.** The flock
  behaves identically on a 60 Hz and a 240 Hz display; uncapped render just shows
  more frames of the same trajectory. Velocity damping is converted per tick
  (`0.98^(60·dt)`) so drag-per-second is framerate-independent.
- **Per sim tick:** clear grid counters → grid build (atomic scatter) → velocity
  (3×3×3 cell stencil, all forces) → position (integrate + wing phase). Position
  and velocity are double-buffered.
- **The fluid runs at a fixed 60 Hz** on its own accumulator — a visual element,
  not physics-critical — so its ~30 tiny dispatches stay cheap.
- **Non-sRGB surface format on purpose**, reproducing the original WebGL look
  (including the dark trough) where raw shader values hit an sRGB canvas.
- **Sub-LSB dithering** on the sky and fog gradients — slow dark gradients band
  badly on 8-bit output, and a tiny time-varying hash noise breaks the bands
  invisibly.
- Spatial grid is 16×8×16 cells × 16 slots. Past ~32k birds, raise
  `SLOTS_PER_CELL` in `src/params.rs` or dense regions start dropping neighbours.

### Status

- [x] GPU boid sim, bird + ground rendering, WGSL hot reload, fixed timestep
- [x] Native system-audio capture (cpal) + full DSP in-process — no external feeder
- [x] Smoothing layer + full band→uniform mappings; optional `--ws` external feeder
- [x] egui control panel (`H` hides chrome), live flock-size slider
- [x] Solo-contrast upperMid mode with live diagnostics
- [x] Realtime autocorrelation beat tracking (tempo / phase / confidence)
- [x] Audio-reactive sky: procedural clouds, stars, fluid plumes, aurora borealis
- [x] MPRIS now-playing card with transport + seek (`P` toggles)
- [x] Flight realism: bird banking + dark agitation bands
- [ ] Save/load presets (TOML)

---

## License

MIT — see [LICENSE](LICENSE).

Built by Kiseia ([@cynd22](https://github.com/cynd22)), with Claude.
