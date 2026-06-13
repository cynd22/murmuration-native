"""
Parallel audio feeder with multi-band onset detection.

PROTOTYPE / EXPLORATION — runs alongside the main `feeder.py` on a different
WS port (default 8766) so the live visualiser is unaffected. Open the
companion `initial-testing/onset-test.html` to see the per-band onset events firing live.

Same audio capture as feeder.py (PipeWire monitor on Linux, sounddevice
PortAudio backend). Same FFT band split. Adds rectified spectral flux
onset detection per band: when a band's frame-to-frame magnitude jump
exceeds an adaptive threshold (mean of recent flux × multiplier), an
onset event fires for that band. Independent per-band detection means
we can distinguish kick / snare / hi-hat events instead of conflating
them into a single bass-spike heuristic.

Run alongside the main feeder:

    # terminal 1 — main feeder, drives the live visualiser
    python feeder.py

    # terminal 2 — this prototype, drives initial-testing/onset-test.html
    python feeder_onsets.py

Both processes can share the same PipeWire monitor source; PipeWire handles
multi-consumer capture cleanly. Open initial-testing/onset-test.html in a separate browser
tab; the live visualiser keeps running in its own tab on the main feeder.

If this prototype validates well by ear+eye, the onset detection logic
gets folded into feeder.py and replaces (or augments) the bass-spike
envelope used to drive the visualiser's shockwave / colour pops / etc.
"""

import argparse
import asyncio
import collections
import json
import os
import subprocess
import sys
import time

import numpy as np
import sounddevice as sd
import websockets


SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
DEFAULT_PORT = 8766

BANDS = [
    ("subBass",   20,    60),
    ("bass",      60,    250),
    ("lowMid",    250,   500),
    ("mid",       500,   2000),
    ("upperMid",  2000,  4000),
    ("treble",    4000,  8000),
    ("air",       8000,  20000),
]

AGC_DECAY = 0.9995
AGC_FLOOR = 1e-4

# =====================================================================
# ONSET DETECTOR TUNING — 2026-05-08
# =====================================================================
# The pre-tuning values (commented OLD below) produced ~7 onsets/second
# per band on Tell Me and Bodyfeeling — far too dense to function as
# "discrete kick punctuation" in the visualiser's colour layer or the
# (currently disabled) shockwave. Symptom: the bass-onset envelope
# stayed continuously elevated during heavy sections instead of
# punching on actual kicks, which forced Kiseia to tune the visualiser's
# onsetBoost down to 0.05 just to keep colour from flickering chaotically.
#
# The post-tuning values (NEW, currently active below) target ~2 onsets/
# second per band — actual kick density for typical electronic music.
# Three knobs adjusted together:
#
#   THRESHOLD_MULT 2.0 → 3.5: flux must exceed rolling mean by 3.5×
#   (not 2×) to count as an onset. Filters out sustained bass guitar
#   note attacks and synth content that's just slightly above baseline,
#   keeping only real spectral spikes.
#
#   REFRACTORY 0.05 → 0.15s: minimum 150ms between consecutive onsets
#   on the same band. Kick drums physically don't fire faster than
#   ~150ms apart in typical electronic music tempos, so this is a
#   hard musical floor. Catches false-positive double-fires from a
#   single transient's edges.
#
#   FLUX_FLOOR 0.0008 → 0.002: raises the absolute minimum flux level
#   to consider, suppressing low-energy false triggers during quiet
#   sections.
#
# To REVERT (if this tuning kills onset detection too aggressively or
# misses kicks on a quieter track): swap the OLD values back in for
# the NEW ones below. Restart the feeder to pick up changes.
# =====================================================================

ONSET_HISTORY_SECONDS = 1.0
ONSET_ENVELOPE_DECAY = 0.88

# OLD (pre-2026-05-08) — kept for easy revert:
# ONSET_THRESHOLD_MULT = 2.0
# ONSET_REFRACTORY_SECONDS = 0.05
# ONSET_FLUX_FLOOR = 0.0008

# NEW (2026-05-08):
ONSET_THRESHOLD_MULT = 3.5
ONSET_REFRACTORY_SECONDS = 0.15
ONSET_FLUX_FLOOR = 0.002

# =====================================================================
# BEAT TRACKER TUNING — 2026-06-13
# =====================================================================
# Rolling-autocorrelation tempo + phase tracker over the summed per-band
# spectral flux (the "novelty" signal). Pure numpy — no aubio/librosa.
# See the BeatTracker class docstring for the algorithm walkthrough;
# these knobs control its windows, prior, and confidence shaping.
#
#   BEAT_NOVELTY_SECONDS: tempo analysis window. 8s ≈ 16 beats at
#   120 BPM — long enough for a stable autocorrelation peak, short
#   enough to follow a tempo change within a phrase or two.
#
#   BEAT_PHASE_SECONDS: phase comb window. Deliberately shorter (4s)
#   than the tempo window so the beat grid re-anchors on *recent*
#   kicks and tracks drummer/DJ drift instead of averaging over it.
#
#   BEAT_RETRACK_SECONDS: how often the heavy autocorrelation runs.
#   0.5s keeps per-frame cost trivial (the per-frame path is just a
#   ring-buffer append plus phase extrapolation).
#
#   BEAT_PRIOR_BPM / BEAT_PRIOR_SIGMA_LOG2: log-Gaussian tempo prior.
#   Centred at 120 BPM with sigma 0.4 octaves — 60 and 240 BPM are
#   each "one octave away" and penalised symmetrically, which is what
#   breaks the half/double-tempo ties autocorrelation can't resolve.
#
#   BEAT_FOLD_LO/HI_BPM: estimates are octave-folded into 70–180 BPM,
#   the range where "the beat" usually lives perceptually.
#
#   BEAT_SILENCE_FLOOR: mean summed flux below this → confidence
#   forced to 0. Absolute (pre-AGC raw magnitudes), so it scales with
#   capture gain — if a very quiet source reads confidence 0 on an
#   obvious beat, lower this first.
#
#   BEAT_CONF_BASE / BEAT_CONF_SCALE: raw confidence is
#   (peak / sum of positive autocorrelation over the search range)
#   × harmonic support at 2× the peak lag (see _retrack for why the
#   support factor is needed — peak/sum alone has a fat noise tail).
#   Measured on synthetic streams at this buffer length: white noise
#   ≤ 0.011 (max over 60 trials), weak/messy beats ~0.016, steady
#   humanised beats 0.04–0.17. conf = clamp((raw - BASE) * SCALE)
#   with BASE 0.012 / SCALE 8 maps noise → 0 and a steady beat →
#   comfortably above 0.15. Known bias: the peak/sum factor shrinks
#   at fast tempi because more harmonic peaks (2×, 3×… the beat lag)
#   fall inside the fixed search range and dilute the denominator —
#   confidence on a solid 170+ BPM track reads lower than the same
#   groove at 90. If drum&bass reads implausibly low-confidence,
#   this is why; lower BASE before touching anything else.
# =====================================================================

BEAT_NOVELTY_SECONDS = 8.0
BEAT_PHASE_SECONDS = 4.0
BEAT_RETRACK_SECONDS = 0.5
BEAT_LAG_MIN_SECONDS = 0.25      # 240 BPM — autocorrelation search range
BEAT_LAG_MAX_SECONDS = 2.0       # 30 BPM
BEAT_PRIOR_BPM = 120.0
BEAT_PRIOR_SIGMA_LOG2 = 0.4
BEAT_FOLD_LO_BPM = 70.0
BEAT_FOLD_HI_BPM = 180.0
BEAT_MEDIAN_COUNT = 8            # BPM median filter — ~4s of estimates
BEAT_SILENCE_FLOOR = 2e-4
BEAT_CONF_BASE = 0.012
BEAT_CONF_SCALE = 8.0
BEAT_MIN_BEAT_CONFIDENCE = 0.05  # below this, never fire the `beat` flag


def _pactl(*args):
    return subprocess.run(
        ["pactl", *args], capture_output=True, text=True, check=True
    ).stdout


def list_pipewire_monitors():
    out = _pactl("list", "sources", "short")
    monitors = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        if ".monitor" in name:
            monitors.append(name)
    return monitors


def pick_pipewire_monitor(name_hint=None):
    monitors = list_pipewire_monitors()
    if not monitors:
        raise RuntimeError("No '.monitor' sources found via pactl. Is PipeWire running?")
    if name_hint:
        matches = [n for n in monitors if name_hint.lower() in n.lower()]
        if not matches:
            raise RuntimeError(f"No monitor source matching {name_hint!r}. Available: {monitors}")
        return matches[0]
    default_sink = None
    for line in _pactl("info").splitlines():
        if line.startswith("Default Sink:"):
            default_sink = line.split(":", 1)[1].strip()
            break
    if default_sink:
        candidate = f"{default_sink}.monitor"
        if candidate in monitors:
            return candidate
    return monitors[0]


def find_loopback_device(name_hint=None):
    devices = sd.query_devices()
    if sys.platform.startswith("linux"):
        monitor = pick_pipewire_monitor(name_hint)
        os.environ["PULSE_SOURCE"] = monitor
        for idx, dev in enumerate(devices):
            if dev["name"] == "pulse" and dev["max_input_channels"] > 0:
                return idx, dev, monitor
        raise RuntimeError("PortAudio doesn't see a 'pulse' device. Is pipewire-pulse running?")
    if not name_hint:
        raise RuntimeError("Auto-detect Linux-only. Pass --device on Windows/macOS.")
    for idx, dev in enumerate(devices):
        if name_hint.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx, dev, dev["name"]
    raise RuntimeError(f"No input device matching {name_hint!r}.")


class Analyser:
    """Same FFT + AGC as feeder.py, but also exposes the *raw* per-band
    magnitudes (pre-AGC) so the onset detector can do spectral flux on
    them — normalized values are unreliable for flux because the AGC
    rescales the floor over time, which would muddy onset detection."""

    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.window = np.hanning(block_size).astype(np.float32)
        freqs = np.fft.rfftfreq(block_size, d=1.0 / sample_rate)
        self.band_slices = []
        for name, lo, hi in BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            idxs = np.flatnonzero(mask)
            if idxs.size == 0:
                self.band_slices.append((name, slice(0, 0)))
            else:
                self.band_slices.append((name, slice(idxs[0], idxs[-1] + 1)))
        self.agc_max = {name: AGC_FLOOR for name, _, _ in BANDS}
        self.amp_agc_max = AGC_FLOOR

    def process(self, block_mono):
        rms = float(np.sqrt(np.mean(block_mono ** 2)))
        self.amp_agc_max = max(rms, self.amp_agc_max * AGC_DECAY)
        amp_norm = min(1.0, rms / max(self.amp_agc_max, AGC_FLOOR))

        spec = np.abs(np.fft.rfft(block_mono * self.window))

        bands_norm = {}
        bands_raw = {}
        for name, sl in self.band_slices:
            val = float(spec[sl].mean()) if sl.stop > sl.start else 0.0
            bands_raw[name] = val
            self.agc_max[name] = max(val, self.agc_max[name] * AGC_DECAY)
            bands_norm[name] = min(1.0, val / max(self.agc_max[name], AGC_FLOOR))

        return {
            "t": time.time(),
            "amp": amp_norm,
            "bands": bands_norm,
            "_raw_bands": bands_raw,  # internal — not sent over WS
        }


class OnsetDetector:
    """Multi-band rectified spectral flux onset detector.

    For each band, computes flux = max(0, current_raw - previous_raw) per
    frame. Maintains a rolling history of recent flux values per band; an
    onset fires when the current flux exceeds (mean_of_history * threshold_mult)
    AND clears an absolute floor AND respects a refractory period since the
    last onset for that band.

    The envelope per band starts at the relative flux strength on each
    onset and decays smoothly between, giving a 0..1 signal you can drive
    a uniform with — same shape as the bass-spike envelope already used
    in the visualiser, but per-band and onset-driven instead of bass-only
    differential-driven.

    Also returns the per-frame flux summed across all bands — the
    "novelty" signal the BeatTracker autocorrelates. Returned here rather
    than recomputed there so flux is only ever computed in one place.
    """

    def __init__(self, band_names, sample_rate=43.0):
        self.band_names = list(band_names)
        history_size = max(10, int(ONSET_HISTORY_SECONDS * sample_rate))
        self.flux_history = {n: collections.deque(maxlen=history_size) for n in self.band_names}
        self.prev_raw = {n: 0.0 for n in self.band_names}
        self.envelopes = {n: 0.0 for n in self.band_names}
        self.last_onset_t = {n: -1.0 for n in self.band_names}
        self.onset_count = {n: 0 for n in self.band_names}

    def process(self, raw_bands, t):
        onsets = {}
        total_flux = 0.0
        for name in self.band_names:
            current = raw_bands.get(name, 0.0)
            prev = self.prev_raw[name]

            flux = max(0.0, current - prev)
            total_flux += flux
            self.flux_history[name].append(flux)

            fired = False
            if len(self.flux_history[name]) >= 10 and flux > ONSET_FLUX_FLOOR:
                hist = self.flux_history[name]
                mean = sum(hist) / len(hist)
                threshold = max(ONSET_FLUX_FLOOR, mean * ONSET_THRESHOLD_MULT)

                if (flux > threshold and
                        t - self.last_onset_t[name] > ONSET_REFRACTORY_SECONDS):
                    fired = True
                    # Envelope bumps to (relative flux strength), capped at 1.
                    relative = flux / (mean * 4.0 + 1e-6)
                    self.envelopes[name] = max(self.envelopes[name], min(1.0, relative))
                    self.last_onset_t[name] = t
                    self.onset_count[name] += 1

            onsets[name] = fired
            self.envelopes[name] *= ONSET_ENVELOPE_DECAY
            self.prev_raw[name] = current

        return onsets, dict(self.envelopes), total_flux


class BeatTracker:
    """Realtime tempo + beat-phase tracker over the onset-novelty signal.

    Four stages, all pure numpy:

    1. NOVELTY. Each frame, the OnsetDetector's rectified spectral flux
       summed across all bands becomes one "how much new spectral energy
       just arrived" value. Kicks, snares and hats all spike it; sustained
       pads barely move it. The last BEAT_NOVELTY_SECONDS (~8s) live in a
       ring buffer.

    2. TEMPO — runs every BEAT_RETRACK_SECONDS (~0.5s), not per frame.
       The mean-subtracted novelty buffer is autocorrelated over lags of
       0.25–2.0s (240–30 BPM): a periodic beat makes the novelty
       self-similar at multiples of the beat period, so the
       autocorrelation peaks there. Peaks are weighted by a log-Gaussian
       prior centred at BEAT_PRIOR_BPM — in log2 space, so half and
       double tempo are penalised symmetrically — which is what resolves
       the half/double-tempo ambiguity autocorrelation alone can't. The
       winning lag gets parabolic interpolation for sub-frame precision
       (one frame at 43fps is ~3 BPM of quantisation around 120), the BPM
       is octave-folded into BEAT_FOLD_LO..HI, and a median over the last
       BEAT_MEDIAN_COUNT estimates kills residual jitter.

       Confidence = (peak / sum of positive autocorrelation over the
       search range) × harmonic support at 2× the peak lag: a steady
       beat concentrates correlation mass at the beat lag AND its
       multiples, while noise spreads it flat and can't fake the
       second lag. Rescaled via BEAT_CONF_BASE/SCALE so uncorrelated
       novelty sits near 0, and forced to 0 outright when the buffer
       is near-silent.

    3. PHASE — same 0.5s cadence. The last BEAT_PHASE_SECONDS of novelty
       is comb-correlated against pulse trains at the chosen period for
       every candidate offset in quarter-frame steps; the offset whose
       pulses land on the flux spikes wins and anchors the beat grid
       (an absolute fractional frame index where a beat occurred).

    4. PER FRAME — trivial cost: extrapolate from the anchor to get
       phase (0..1 within the beat), nextBeatIn (seconds until the next
       predicted beat), and a one-frame `beat` flag on the frame nearest
       each predicted beat instant. The flag has a half-period refractory
       and is gated on confidence so silence never "beats".

    The output dict shape is a wire contract — the Rust client parses
    {bpm, period, phase, nextBeatIn, confidence, beat} exactly. Floats
    are rounded to 4 decimals to keep the payload small.
    """

    def __init__(self, frame_rate=SAMPLE_RATE / BLOCK_SIZE):
        self.frame_rate = frame_rate
        self.frame_dt = 1.0 / frame_rate
        self.novelty = collections.deque(maxlen=int(BEAT_NOVELTY_SECONDS * frame_rate))
        self.lag_min = max(2, int(round(BEAT_LAG_MIN_SECONDS * frame_rate)))
        self.lag_max = int(round(BEAT_LAG_MAX_SECONDS * frame_rate))
        self.retrack_frames = max(1, int(round(BEAT_RETRACK_SECONDS * frame_rate)))
        # Need enough buffer that the longest lag still has decent overlap
        # before the first estimate; ~4s ≈ 2× the 2s max lag.
        self.min_track_frames = int(2 * BEAT_LAG_MAX_SECONDS * frame_rate)
        self.frames_since_track = 0
        self.frame_idx = -1

        self.bpm_history = collections.deque(maxlen=BEAT_MEDIAN_COUNT)
        self.bpm = 0.0
        self.period_frames = 0.0
        self.confidence = 0.0
        self.beat_anchor = 0.0      # absolute (fractional) frame index of a beat
        self.last_beat_fire = -1e9  # frame index of the last fired `beat` flag
        self.have_estimate = False

    def process(self, novelty):
        """Feed one frame of summed flux; returns the per-frame beat dict."""
        self.frame_idx += 1
        self.novelty.append(float(novelty))

        self.frames_since_track += 1
        if (self.frames_since_track >= self.retrack_frames
                and len(self.novelty) >= self.min_track_frames):
            self._retrack()
            self.frames_since_track = 0

        if not self.have_estimate:
            return {"bpm": 0.0, "period": 0.0, "phase": 0.0,
                    "nextBeatIn": 0.0, "confidence": 0.0, "beat": False}

        # Cheap per-frame path: extrapolate phase from the anchored grid.
        period_s = self.period_frames * self.frame_dt
        phase = ((self.frame_idx - self.beat_anchor) / self.period_frames) % 1.0
        next_in = (1.0 - phase) * period_s

        fired = False
        frames_from_beat = min(phase, 1.0 - phase) * self.period_frames
        if (self.confidence >= BEAT_MIN_BEAT_CONFIDENCE
                and frames_from_beat <= 0.5
                and self.frame_idx - self.last_beat_fire > 0.5 * self.period_frames):
            fired = True
            self.last_beat_fire = self.frame_idx

        return {
            "bpm": round(self.bpm, 4),
            "period": round(period_s, 4),
            "phase": round(phase, 4),
            "nextBeatIn": round(next_in, 4),
            "confidence": round(self.confidence, 4),
            "beat": fired,
        }

    def _retrack(self):
        """Tempo estimate (autocorrelation + prior + fold + median),
        confidence, then phase re-anchor. The 'heavy' path — ~0.1M numpy
        flops at 2Hz, irrelevant next to the per-frame FFT."""
        x = np.asarray(self.novelty, dtype=np.float64)
        if float(x.mean()) < BEAT_SILENCE_FLOOR:
            self.confidence = 0.0
            return
        x = x - x.mean()

        n = x.size
        lag_max = min(self.lag_max, n // 2)
        if lag_max - self.lag_min < 4:
            return
        ac = np.correlate(x, x, mode="full")[n - 1:]
        if ac[0] <= 1e-12:
            self.confidence = 0.0
            return
        ac = ac / ac[0]

        lags = np.arange(self.lag_min, lag_max + 1)
        bpms = 60.0 * self.frame_rate / lags
        prior = np.exp(-0.5 * (np.log2(bpms / BEAT_PRIOR_BPM) / BEAT_PRIOR_SIGMA_LOG2) ** 2)
        seg = ac[self.lag_min:lag_max + 1]
        weighted = seg * prior
        i = int(np.argmax(weighted))

        # Parabolic interpolation around the peak for sub-frame lag precision.
        delta = 0.0
        if 0 < i < weighted.size - 1:
            a, b, c = weighted[i - 1], weighted[i], weighted[i + 1]
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-12:
                delta = max(-0.5, min(0.5, 0.5 * (a - c) / denom))
        lag_star = lags[i] + delta

        bpm = 60.0 * self.frame_rate / lag_star
        while bpm < BEAT_FOLD_LO_BPM:
            bpm *= 2.0
        while bpm > BEAT_FOLD_HI_BPM:
            bpm /= 2.0

        self.bpm_history.append(bpm)
        self.bpm = float(np.median(self.bpm_history))
        self.period_frames = 60.0 * self.frame_rate / self.bpm

        # Confidence = (peak / positive-AC mass) × harmonic support.
        # The first factor alone has a fat noise tail: over an 8s buffer,
        # noise occasionally produces a single lucky AC spike that the
        # prior-weighted argmax then picks. The support factor checks a
        # second, deterministically-chosen lag — 2× the peak lag (or ½×
        # when 2× falls outside the search range) — where a real beat is
        # also self-similar but a lucky noise spike has no reason to be.
        # ±2-frame window on the support lag tolerates tempo jitter.
        pos_sum = float(np.clip(seg, 0.0, None).sum())
        peak_ratio = max(0.0, float(seg[i])) / (pos_sum + 1e-9)
        lag_n = int(round(lag_star))
        support_lag = lag_n * 2 if lag_n * 2 <= lag_max else max(self.lag_min, lag_n // 2)
        j = support_lag - self.lag_min
        window = seg[max(0, j - 2):j + 3]
        support = float(np.clip(window, 0.0, 1.0).max()) if window.size else 0.0
        raw = peak_ratio * support
        self.confidence = max(0.0, min(1.0, (raw - BEAT_CONF_BASE) * BEAT_CONF_SCALE))

        self._rephase()
        self.have_estimate = True

    def _rephase(self):
        """Anchor the beat grid: comb-correlate recent novelty against
        pulse trains at the current period over all candidate offsets
        (quarter-frame steps, linear interpolation between frames); the
        offset whose pulses sit on the flux spikes wins. Mean rather than
        sum so offsets fitting one fewer pulse in the window aren't
        penalised."""
        w = min(len(self.novelty), int(BEAT_PHASE_SECONDS * self.frame_rate))
        p = self.period_frames
        if p <= 1.0 or w < p + 2:
            return
        y = np.asarray(self.novelty, dtype=np.float64)[-w:]

        best_off, best_score = 0.0, -np.inf
        for off in np.arange(0.0, p, 0.25):
            pos = np.arange(off, w - 1.0, p)
            i0 = pos.astype(int)
            frac = pos - i0
            score = float((y[i0] * (1.0 - frac) + y[i0 + 1] * frac).mean())
            if score > best_score:
                best_score, best_off = score, off

        # y[0] is absolute frame (frame_idx - w + 1); the first pulse of
        # the winning train is a beat instant.
        self.beat_anchor = self.frame_idx - (w - 1) + best_off


async def _broadcast(clients, msg):
    if not clients:
        return
    payload = json.dumps(msg, separators=(",", ":"))
    await asyncio.gather(*(c.send(payload) for c in clients), return_exceptions=True)


def _make_ws_handler(clients):
    async def handle_client(ws):
        clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
    return handle_client


async def feeder_loop(device_idx, port, channels):
    clients = set()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    analyser = Analyser()
    detector = OnsetDetector([name for name, _, _ in BANDS])
    beat_tracker = BeatTracker()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata.reshape(-1)
        try:
            loop.call_soon_threadsafe(queue.put_nowait, mono.copy())
        except asyncio.QueueFull:
            pass

    stream = sd.InputStream(
        device=device_idx,
        channels=channels,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=audio_callback,
    )

    server = await websockets.serve(_make_ws_handler(clients), "localhost", port)
    print(f"[onsets] listening on ws://localhost:{port}")
    print(f"[onsets] capturing from device #{device_idx} ({channels}ch @ {SAMPLE_RATE}Hz, block {BLOCK_SIZE})")
    print(f"[onsets] onset detection: spectral flux per band, threshold = mean * {ONSET_THRESHOLD_MULT}, refractory {ONSET_REFRACTORY_SECONDS*1000:.0f}ms")
    print(f"[onsets] beat tracking: {BEAT_NOVELTY_SECONDS:.0f}s novelty autocorrelation every {BEAT_RETRACK_SECONDS}s, fold {BEAT_FOLD_LO_BPM:.0f}-{BEAT_FOLD_HI_BPM:.0f} BPM")

    with stream:
        try:
            while True:
                block = await queue.get()
                result = analyser.process(block)
                onsets, envelopes, novelty = detector.process(result.pop("_raw_bands"), result["t"])
                result["onsets"] = onsets
                result["onset_envelopes"] = envelopes
                result["beat"] = beat_tracker.process(novelty)
                await _broadcast(clients, result)
        except asyncio.CancelledError:
            pass
        finally:
            server.close()
            await server.wait_closed()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", help="substring of input device name")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    idx, dev, source_label = find_loopback_device(args.device)
    print(f"[onsets] PortAudio device #{idx}: {dev['name']!r}")
    if source_label and source_label != dev["name"]:
        print(f"[onsets] capturing source: {source_label}")
    channels = max(1, min(2, dev["max_input_channels"]))

    try:
        asyncio.run(feeder_loop(idx, args.port, channels))
    except KeyboardInterrupt:
        print("\n[onsets] stopped")


if __name__ == "__main__":
    main()
