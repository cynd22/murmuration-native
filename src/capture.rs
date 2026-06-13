//! System-audio (loopback) capture via cpal.
//!
//! This is the in-process replacement for the external Python feeder's capture
//! stage: it grabs whatever is currently playing out of the default output and
//! hands the rest of the pipeline mono f32 sample chunks. The DSP / onset /
//! beat analysis lives in `dsp.rs`; this module only does I/O + downmix.
//!
//! ## Why "PulseAudio Sound Server", not a "*.monitor" device
//!
//! On every current Linux desktop (PipeWire-with-pulse, or classic PulseAudio)
//! the cpal ALSA host does NOT surface PipeWire/Pulse monitor node names like
//! `alsa_output...analog-stereo.monitor`. It only exposes generic ALSA PCM
//! descriptions ("PulseAudio Sound Server", "PipeWire Sound Server", the raw
//! HD-Audio cards, etc). So we cannot pick "the monitor" by name through cpal.
//!
//! What works instead: the Pulse/PipeWire *default source* is normally the
//! default sink's monitor, so capturing from the cpal "PulseAudio Sound Server"
//! device records the system output mix with zero user setup — the same
//! mechanism the Python feeder relied on. We additionally set `PULSE_SOURCE`
//! to a concrete `.monitor` (discovered via `pactl`) before opening the stream,
//! which pins the capture to the output monitor even if the user has set some
//! other default source (e.g. a USB mic). That's a best-effort nicety: if
//! `pactl` isn't present we just fall back to the default-source routing.
//!
//! ## Lifetime
//!
//! cpal stops the stream the instant its `Stream` handle is dropped. The
//! analysis side only ever sees the channel `Receiver`, so if we returned the
//! `Stream` and the caller dropped it, capture would die silently. We instead
//! move the `Stream` into a detached thread that parks forever — it lives for
//! the whole process and is torn down by the OS at exit.

use std::sync::mpsc::{Receiver, Sender};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};

/// Handle the analysis side holds: a stream of mono f32 chunks plus the real
/// capture sample rate (NEVER assume 44100 — this machine is 48000).
pub struct Capture {
    /// Mono f32 chunks, downmixed from however many channels the device has.
    /// Chunk sizes follow cpal's callback buffer, not `BLOCK_SIZE`; the caller
    /// is expected to accumulate them into fixed analysis frames.
    pub rx: Receiver<Vec<f32>>,
    /// The device's actual sample rate, to be fed into the analyser.
    pub sample_rate: u32,
}

/// Best-effort human-readable name for a device. `Device::name()` was removed
/// in cpal 0.18; the replacement is `description().name()`.
fn device_name(dev: &Device) -> String {
    dev.description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

/// True if the lowercased device name looks like a system-audio monitor.
/// Real `*.monitor` names are not exposed by the cpal ALSA host, but some
/// hosts/platforms do name them, so the check stays for portability.
fn looks_like_monitor(name_lc: &str) -> bool {
    name_lc.contains("monitor") || name_lc.contains("loopback")
}

/// Run `pactl <args...>` and return stdout as a String, or None on any failure
/// (pactl missing, non-zero exit, non-UTF8). Used only to discover a concrete
/// monitor source name and to set `PULSE_SOURCE`; never required for capture.
fn pactl(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("pactl").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

/// Discover the best Pulse/PipeWire monitor source name to capture, preferring
/// the *default sink's* monitor so we follow whatever the user is listening to.
/// Returns None if pactl is unavailable or there are no monitor sources.
fn discover_monitor_source() -> Option<String> {
    // Collect all "...monitor" source names from `pactl list short sources`.
    let short = pactl(&["list", "short", "sources"])?;
    let mut monitors: Vec<String> = Vec::new();
    for line in short.lines() {
        // Columns: index \t name \t driver \t spec \t state
        if let Some(name) = line.split('\t').nth(1) {
            if name.contains(".monitor") {
                monitors.push(name.to_string());
            }
        }
    }
    if monitors.is_empty() {
        return None;
    }

    // Prefer the default sink's monitor (what the user actually hears).
    if let Some(sink) = pactl(&["get-default-sink"]) {
        let candidate = format!("{}.monitor", sink.trim());
        if monitors.iter().any(|m| m == &candidate) {
            return Some(candidate);
        }
    }
    // Otherwise: if the default *source* is itself a monitor, take it; else any.
    if let Some(src) = pactl(&["get-default-source"]) {
        let src = src.trim().to_string();
        if src.ends_with(".monitor") && monitors.iter().any(|m| m == &src) {
            return Some(src);
        }
    }
    monitors.into_iter().next()
}

/// Print all input devices to stdout, flagging ones that are (or, on Linux,
/// route to) system-audio. Used by a `--list-devices` style flag so a user on
/// an unusual setup can find the right `--device` substring.
pub fn list_devices() {
    let host = cpal::default_host();
    println!("Input devices (host: {:?}):", host.id());

    let default_name = host
        .default_input_device()
        .map(|d| device_name(&d))
        .unwrap_or_else(|| "<none>".to_string());

    let devices = match host.input_devices() {
        Ok(d) => d,
        Err(e) => {
            println!("  (failed to enumerate input devices: {e})");
            return;
        }
    };

    // On Linux the "pulse"/"pipewire" server devices route to the default
    // source, which is normally the output monitor — flag them as such.
    let monitor_src = discover_monitor_source();

    let mut any = false;
    for dev in devices {
        any = true;
        let name = device_name(&dev);
        let name_lc = name.to_lowercase();

        let mut tags: Vec<&str> = Vec::new();
        if name == default_name {
            tags.push("default");
        }
        if looks_like_monitor(&name_lc) {
            tags.push("monitor/loopback");
        } else if name_lc.contains("pulse") || name_lc.contains("pipewire") {
            // These follow the default/PULSE_SOURCE routing -> system audio.
            tags.push("system-audio (via monitor source)");
        }

        // Probe the default input config so the user sees rate / format / chans.
        let cfg = match dev.default_input_config() {
            Ok(c) => format!(
                "ch={} rate={} fmt={:?}",
                c.channels(),
                c.sample_rate(),
                c.sample_format()
            ),
            Err(e) => format!("(unavailable: {e})"),
        };

        if tags.is_empty() {
            println!("  - {name}  [{cfg}]");
        } else {
            println!("  - {name}  [{cfg}]  <{}>", tags.join(", "));
        }
    }
    if !any {
        println!("  (no input devices found)");
    }

    if let Some(src) = monitor_src {
        println!();
        println!("Detected system-audio monitor source: {src}");
        println!(
            "Capturing the \"PulseAudio Sound Server\" / default input records this monitor."
        );
    }
}

/// Pick the device to capture from, given an optional name-substring override.
///
/// Priority:
///   1. explicit `device_substr` -> first input device whose name contains it
///      (case-insensitive); error if nothing matches so the user knows.
///   2. a device whose name itself looks like a monitor/loopback (rare via the
///      ALSA host, but free and correct where hosts expose it).
///   3. the "pulse" server device (PulseAudio Sound Server) — captures the
///      default source, i.e. the output monitor on a stock desktop.
///   4. the "pipewire" server device.
///   5. the host default input (with a warning — may be a real mic).
///
/// Returns `(device, auto_is_system_audio)`. `auto_is_system_audio` is false
/// only for case 5, so the caller can warn that it might be a microphone.
fn pick_device(
    host: &cpal::Host,
    device_substr: &Option<String>,
) -> Result<(Device, bool), String> {
    let devices: Vec<Device> = host
        .input_devices()
        .map_err(|e| format!("failed to enumerate input devices: {e}"))?
        .collect();

    if let Some(sub) = device_substr {
        let sub_lc = sub.to_lowercase();
        let found = devices
            .into_iter()
            .find(|d| device_name(d).to_lowercase().contains(&sub_lc));
        return match found {
            Some(d) => Ok((d, true)),
            None => Err(format!("no input device name contains {sub:?}")),
        };
    }

    let (mut monitor_named, mut pulse, mut pipewire) = (None, None, None);
    for dev in devices {
        let n = device_name(&dev).to_lowercase();
        if looks_like_monitor(&n) {
            monitor_named.get_or_insert(dev);
        } else if n.contains("pulse") {
            pulse.get_or_insert(dev);
        } else if n.contains("pipewire") {
            pipewire.get_or_insert(dev);
        }
    }

    if let Some(d) = monitor_named.or(pulse).or(pipewire) {
        return Ok((d, true));
    }

    host.default_input_device()
        .map(|d| (d, false))
        .ok_or_else(|| "no input device available".to_string())
}

/// Start system-audio capture.
///
/// * `device_substr == None` auto-selects a system-audio monitor/loopback input
///   when one exists (see [`pick_device`]); on Linux it additionally pins
///   `PULSE_SOURCE` to the output monitor so the cpal pulse device follows the
///   sound the user actually hears.
/// * `device_substr == Some(s)` selects the first input device whose name
///   contains `s` (case-insensitive).
///
/// The returned [`Capture`] yields mono f32 chunks on `rx`; the cpal stream is
/// kept alive for the program lifetime inside a detached parked thread.
pub fn start(device_substr: Option<String>) -> Result<Capture, String> {
    let host = cpal::default_host();

    // On Linux, before opening a pulse/pipewire device, pin the capture to the
    // output monitor via PULSE_SOURCE so we get system audio even if the user's
    // default *source* is a mic. Skip this when the user gave an explicit
    // device override (they're asking for something specific) or already set
    // PULSE_SOURCE themselves. Best-effort: silent if pactl is missing.
    if device_substr.is_none() && std::env::var_os("PULSE_SOURCE").is_none() {
        if let Some(monitor) = discover_monitor_source() {
            log::info!("pinning PULSE_SOURCE to monitor: {monitor}");
            // SAFETY: set before the audio thread/stream is created; no other
            // thread reads the environment concurrently at this point.
            unsafe {
                std::env::set_var("PULSE_SOURCE", &monitor);
            }
        }
    }

    let (device, is_system_audio) = pick_device(&host, &device_substr)?;
    let name = device_name(&device);

    if device_substr.is_none() && !is_system_audio {
        log::warn!(
            "no monitor/loopback/pulse input found; falling back to default input \
             '{name}' — this may be a microphone, not system audio. \
             Run with --list-devices, or set the default source to a monitor \
             (e.g. `pactl set-default-source $(pactl get-default-sink).monitor`)."
        );
    }

    let supported = device
        .default_input_config()
        .map_err(|e| format!("default_input_config failed for '{name}': {e}"))?;
    let sample_rate = supported.sample_rate(); // u32 alias in cpal 0.18
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    // build_input_stream takes the StreamConfig BY VALUE in cpal 0.18.
    let config: StreamConfig = supported.config();

    log::info!(
        "capturing system audio from '{name}' @ {sample_rate} Hz, {channels} ch, {sample_format:?}"
    );

    let (tx, rx): (Sender<Vec<f32>>, Receiver<Vec<f32>>) = std::sync::mpsc::channel();

    let err_fn = |e| log::error!("audio stream error: {e}");

    // Each variant gets its own cloned sender and a per-sample -> f32 converter.
    // The closure downmixes every interleaved `channels`-wide frame to one mono
    // sample by averaging, then sends the whole chunk. A dead receiver (caller
    // shutting down) just means our send fails — ignore it.
    let ch = channels.max(1);
    macro_rules! mono_stream {
        ($t:ty, $to_f32:expr) => {{
            let tx = tx.clone();
            device.build_input_stream(
                config,
                move |data: &[$t], _: &cpal::InputCallbackInfo| {
                    let mut mono = Vec::with_capacity(data.len() / ch);
                    for frame in data.chunks(ch) {
                        let sum: f32 = frame.iter().map(|&s| $to_f32(s)).sum();
                        mono.push(sum / ch as f32);
                    }
                    let _ = tx.send(mono);
                },
                err_fn,
                None,
            )
        }};
    }

    let stream = match sample_format {
        SampleFormat::F32 => mono_stream!(f32, |s: f32| s),
        SampleFormat::I16 => mono_stream!(i16, |s: i16| s as f32 / i16::MAX as f32),
        SampleFormat::U16 => {
            mono_stream!(u16, |s: u16| (s as f32 / u16::MAX as f32) * 2.0 - 1.0)
        }
        // The probe found I32 cards too; supporting it is free and avoids a
        // hard failure if the chosen device only offers 32-bit int.
        SampleFormat::I32 => mono_stream!(i32, |s: i32| s as f32 / i32::MAX as f32),
        other => return Err(format!("unsupported sample format: {other:?}")),
    }
    .map_err(|e| format!("build_input_stream failed for '{name}': {e}"))?;

    stream
        .play()
        .map_err(|e| format!("stream.play failed for '{name}': {e}"))?;

    // CRITICAL lifetime guard: cpal stops capture the moment the Stream is
    // dropped. Move it into a detached thread that parks forever so it lives as
    // long as the process. `Stream` is !Send on some hosts, but the ALSA host's
    // Stream is Send; if a future host makes it !Send this won't compile and we
    // revisit (Box::leak on the main thread is the fallback).
    std::thread::Builder::new()
        .name("cpal-capture-keepalive".to_string())
        .spawn(move || {
            let _keep_alive = stream;
            loop {
                std::thread::park();
            }
        })
        .map_err(|e| format!("failed to spawn capture keep-alive thread: {e}"))?;

    Ok(Capture { rx, sample_rate })
}