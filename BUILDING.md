# Building on Linux

## 1. Install Rust

Install via [rustup](https://rustup.rs/) (distro Rust packages are often too old —
this project uses the 2024 edition, which needs **Rust 1.85+**):

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable
```

## 2. System packages

The app is built on wgpu (Vulkan on Linux) + winit (Wayland and X11 both
supported, picked at runtime). You need a working Vulkan driver, the Vulkan
loader, and libxkbcommon; Wayland users also need the Wayland client library
(almost certainly already installed on any Wayland desktop).

It also captures system audio (ALSA, via cpal) and reads now-playing metadata
(D-Bus / MPRIS), so it needs the **ALSA** and **D-Bus** development headers at
build time. No Python or external feeder is required — capture is in-process.

### Arch

```sh
sudo pacman -S --needed gcc pkgconf libxkbcommon wayland vulkan-icd-loader \
               alsa-lib dbus
# plus the Vulkan driver for your GPU:
sudo pacman -S vulkan-radeon        # AMD
sudo pacman -S vulkan-intel         # Intel
sudo pacman -S nvidia-utils         # NVIDIA (proprietary driver)
```

### Debian / Ubuntu

```sh
sudo apt install build-essential pkg-config libxkbcommon-dev libwayland-dev \
                 libvulkan1 mesa-vulkan-drivers libasound2-dev libdbus-1-dev
```

`mesa-vulkan-drivers` covers AMD and Intel GPUs. On NVIDIA, the proprietary
driver packages (`nvidia-driver-*`) ship their own Vulkan ICD — you still want
`libvulkan1` (the loader), but not `mesa-vulkan-drivers`.

### Fedora

```sh
sudo dnf install gcc pkgconf-pkg-config libxkbcommon-devel wayland-devel \
                 vulkan-loader mesa-vulkan-drivers alsa-lib-devel dbus-devel
```

Same NVIDIA note as above: with the RPM Fusion NVIDIA driver you get the
Vulkan ICD from the driver; `vulkan-loader` is still required.

### NVIDIA notes

- Recent proprietary drivers (535+) work fine on both X11 and Wayland.
- If the window opens but stays black or the app can't find an adapter, check
  `vulkaninfo --summary` (package `vulkan-tools`) — the app needs at least one
  Vulkan-capable device listed.
- Hybrid (Optimus) laptops: prefix with `prime-run` (Arch) or set
  `__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia` to run on the
  dGPU.

## 3. Build and run

```sh
cargo build --release
```

The binary lands at `target/release/murmuration`.

**Important:** the app loads its WGSL shaders from disk at runtime (that's what
makes them hot-reloadable). It looks for `./shaders` or `./rust-port/shaders`
relative to the **working directory**, so run it from the repo root:

```sh
./target/release/murmuration
# or simply
cargo run --release
```

If you move the binary elsewhere, keep a copy of the `shaders/` directory next
to wherever you launch it from.

## 4. Flags

| Flag | Meaning | Default |
|---|---|---|
| `--birds N` | flock size | 10000 |
| `--uncapped` | disable vsync (benchmarking / high-refresh displays) | vsync on |
| `--sim-hz N` | fixed simulation tick rate | 240 |
| `--list-devices` | print audio input devices (monitors marked) and exit | — |
| `--device SUBSTR` | force a capture device by name substring | auto (monitor) |
| `--ws URL` | use an external feeder instead of native capture | native capture |

`Esc` quits. `H` toggles the control panel, `P` toggles the now-playing card.
FPS is shown in the title bar.

## 5. Audio

Capture is **native and automatic** — the app records your system output via
cpal/ALSA and runs the full FFT/onset/beat analysis in-process. On
PipeWire/PulseAudio it auto-selects your default sink's *monitor*, so just play
something and the flock reacts (`audio ✓` in the title bar). No feeder, no Python.

If auto-selection grabs the wrong input on an unusual setup:

```sh
./murmuration --list-devices                 # see inputs; monitors are marked
./murmuration --device "monitor"             # force one by name substring
# or set the recording source system-wide:
pactl set-default-source $(pactl get-default-sink).monitor
# or pick "Monitor of ..." in pavucontrol -> Recording
```

Advanced: `--ws ws://host:8766` makes the app consume an external feeder over a
websocket instead (the JSON band/onset format used by the parent
[cynd22/murmuration-visualiser](https://github.com/cynd22/murmuration-visualiser)),
e.g. to drive it from another machine.

The now-playing card (title/artist/art/transport, `P` to toggle) reads any
MPRIS-compatible player over D-Bus — VLC, mpv, browsers, Spotify, and so on.
