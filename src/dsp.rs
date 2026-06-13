//! Audio DSP — a faithful Rust port of `audio-feeder/feeder_onsets.py`.
//!
//! Three sequential components process one mono audio block (1024 samples)
//! per call, in this fixed order:
//!
//!   1. `Analyser`     — Hann window → real FFT → per-band magnitude means;
//!                       a peak-follower AGC normalises each band and the
//!                       overall RMS amplitude into 0..1. Also exposes the
//!                       *raw* (pre-AGC) band means the onset detector needs.
//!   2. `OnsetDetector`— rectified spectral flux per band → adaptive
//!                       threshold → per-band onset bools + decaying 0..1
//!                       envelopes, plus a single summed `novelty` scalar.
//!   3. `BeatTracker`  — a novelty ring buffer; periodic autocorrelation
//!                       tempo estimate (log-Gaussian prior, octave fold,
//!                       median filter), comb-correlation phase anchoring,
//!                       and cheap per-frame phase extrapolation emitting
//!                       {bpm, period, phase, next_beat_in, confidence, beat}.
//!
//! Downstream visual tuning was calibrated against the Python feeder, so the
//! magnitude scale and every constant below are reproduced exactly. The single
//! most important fidelity point: the FFT and autocorrelation are *unnormalised*
//! (no 1/N), because absolute thresholds (`ONSET_FLUX_FLOOR`, `BEAT_SILENCE_FLOOR`,
//! `BEAT_CONF_BASE`) compare directly against linear magnitudes — an FFT scale
//! mismatch would desync them. realfft is unnormalised (matches NumPy rfft).
//!
//! Cadence: capture is fixed at SAMPLE_RATE=44100, BLOCK_SIZE=1024, so the
//! nominal frame rate is 44100/1024 = 43.06640625 fps. Note (deliberately,
//! matching the Python): the OnsetDetector's history length uses the *rounded*
//! 43.0 fps the Python constructor defaults to, while the BeatTracker uses the
//! exact 43.06640625. They are genuinely different in the source — do not unify.

use std::collections::VecDeque;
use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

/// Samples per processed block (PortAudio capture block, Python BLOCK_SIZE).
pub const BLOCK_SIZE: usize = 1024;
/// Number of frequency bands (subBass..air).
pub const N_BANDS: usize = 7;

/// Band edges in Hz, half-open [lo, hi). Order is load-bearing everywhere:
/// 0=subBass 1=bass 2=lowMid 3=mid 4=upperMid 5=treble 6=air.
const BAND_EDGES: [(f32, f32); N_BANDS] = [
    (20.0, 60.0),     // subBass
    (60.0, 250.0),    // bass
    (250.0, 500.0),   // lowMid
    (500.0, 2000.0),  // mid
    (2000.0, 4000.0), // upperMid
    (4000.0, 8000.0), // treble
    (8000.0, 20000.0),// air
];

// --- AGC constants (Analyser) ---
const AGC_DECAY: f32 = 0.9995;
const AGC_FLOOR: f32 = 1e-4;

// --- Onset constants (OnsetDetector) ---
const ONSET_HISTORY_SECONDS: f32 = 1.0;
const ONSET_ENVELOPE_DECAY: f32 = 0.88;
const ONSET_THRESHOLD_MULT: f32 = 3.5;
const ONSET_REFRACTORY_SECONDS: f32 = 0.15;
const ONSET_FLUX_FLOOR: f32 = 0.002;
/// The Python OnsetDetector is constructed with the default `sample_rate=43.0`
/// (the call site passes no override), so its history length rounds against 43.0,
/// NOT the exact 43.066. history_size = max(10, int(1.0 * 43.0)) = 43.
const ONSET_FRAME_RATE: f32 = 43.0;

// --- Beat tracker constants (BeatTracker) ---
const BEAT_NOVELTY_SECONDS: f32 = 8.0;
const BEAT_PHASE_SECONDS: f32 = 4.0;
const BEAT_RETRACK_SECONDS: f32 = 0.5;
const BEAT_LAG_MIN_SECONDS: f32 = 0.25; // 240 BPM
const BEAT_LAG_MAX_SECONDS: f32 = 2.0; // 30 BPM
const BEAT_PRIOR_BPM: f32 = 120.0;
const BEAT_PRIOR_SIGMA_LOG2: f32 = 0.4;
const BEAT_FOLD_LO_BPM: f32 = 70.0;
const BEAT_FOLD_HI_BPM: f32 = 180.0;
const BEAT_MEDIAN_COUNT: usize = 8;
const BEAT_SILENCE_FLOOR: f32 = 2e-4;
const BEAT_CONF_BASE: f32 = 0.012;
const BEAT_CONF_SCALE: f32 = 8.0;
const BEAT_MIN_BEAT_CONFIDENCE: f32 = 0.05;

// =====================================================================
// COMPONENT 1 — Analyser
// =====================================================================

/// One block of analysis: AGC-normalised bands (0..1), raw pre-AGC band
/// means (linear magnitude, for onset flux), and overall amplitude (0..1).
#[derive(Clone, Copy)]
pub struct Frame {
    pub bands: [f32; N_BANDS],
    pub raw_bands: [f32; N_BANDS],
    pub amp: f32,
}

pub struct Analyser {
    /// realfft plan for a length-BLOCK_SIZE real transform (513 complex bins out).
    fft: Arc<dyn RealToComplex<f32>>,
    /// Reusable FFT input buffer (windowed samples).
    fft_in: Vec<f32>,
    /// Reusable FFT output buffer (513 complex bins).
    fft_out: Vec<Complex<f32>>,
    /// Symmetric Hann window, length BLOCK_SIZE (matches NumPy `np.hanning`).
    window: [f32; BLOCK_SIZE],
    /// Inclusive..exclusive bin range per band, resolved from the bin frequencies.
    band_bins: [(usize, usize); N_BANDS],
    /// Per-band peak-follower AGC running maxima (one per band).
    agc_max: [f32; N_BANDS],
    /// Overall-amplitude peak-follower AGC running maximum.
    amp_agc_max: f32,
}

impl Analyser {
    pub fn new(sample_rate: u32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(BLOCK_SIZE);
        let fft_in = fft.make_input_vec();
        let fft_out = fft.make_output_vec();

        // Symmetric Hann window: w[n] = 0.5 - 0.5*cos(2*pi*n/(M-1)), n=0..M-1.
        // CRITICAL: denominator is M-1 (=1023), matching NumPy `np.hanning`.
        // Endpoints w[0] and w[M-1] are exactly 0. A "Hann/M" variant would
        // shift magnitudes and desync the absolute thresholds downstream.
        let mut window = [0.0f32; BLOCK_SIZE];
        let denom = (BLOCK_SIZE - 1) as f32;
        for (n, w) in window.iter_mut().enumerate() {
            *w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / denom).cos();
        }

        // Resolve each band to a contiguous bin range [first, last+1) from the
        // half-open frequency mask freqs[k] in [lo, hi). freqs[k] = k*fs/N.
        // 513 bins for a 1024-point real FFT (N/2 + 1), bin 0 = DC.
        let n_bins = BLOCK_SIZE / 2 + 1;
        let bin_hz = sample_rate as f32 / BLOCK_SIZE as f32;
        let mut band_bins = [(0usize, 0usize); N_BANDS];
        for (b, &(lo, hi)) in BAND_EDGES.iter().enumerate() {
            let mut first: Option<usize> = None;
            let mut last: usize = 0;
            for k in 0..n_bins {
                let f = k as f32 * bin_hz;
                if f >= lo && f < hi {
                    if first.is_none() {
                        first = Some(k);
                    }
                    last = k;
                }
            }
            band_bins[b] = match first {
                Some(f) => (f, last + 1), // contiguous [first, last+1)
                None => (0, 0),           // empty band
            };
        }

        Self {
            fft,
            fft_in,
            fft_out,
            window,
            band_bins,
            agc_max: [AGC_FLOOR; N_BANDS],
            amp_agc_max: AGC_FLOOR,
        }
    }

    /// Process one BLOCK_SIZE mono block. `block.len()` is expected to equal
    /// BLOCK_SIZE; shorter is zero-padded, longer is truncated (defensive).
    pub fn process(&mut self, block: &[f32]) -> Frame {
        // --- Overall amplitude from the *un-windowed* time-domain block. ---
        // rms = sqrt(mean(x^2)) over the raw mono samples (NO window here; the
        // window is only for the FFT path).
        let mut sum_sq = 0.0f64;
        for i in 0..BLOCK_SIZE {
            let x = block.get(i).copied().unwrap_or(0.0) as f64;
            sum_sq += x * x;
        }
        let rms = (sum_sq / BLOCK_SIZE as f64).sqrt() as f32;
        let amp = agc_step(&mut self.amp_agc_max, rms);

        // --- FFT path: window the block, real FFT, magnitude spectrum. ---
        for i in 0..BLOCK_SIZE {
            self.fft_in[i] = block.get(i).copied().unwrap_or(0.0) * self.window[i];
        }
        // realfft is unnormalised (matches NumPy rfft, norm=None). 513 bins out.
        self.fft
            .process(&mut self.fft_in, &mut self.fft_out)
            .expect("realfft length invariant holds");

        // Per band: arithmetic mean of the linear magnitude |re+i*im| over the
        // band's bins. No sqrt-of-power, no log, no sum — plain mean. Empty → 0.
        let mut raw_bands = [0.0f32; N_BANDS];
        let mut bands = [0.0f32; N_BANDS];
        for b in 0..N_BANDS {
            let (start, stop) = self.band_bins[b];
            let val = if stop > start {
                let mut acc = 0.0f64;
                for k in start..stop {
                    acc += self.fft_out[k].norm() as f64; // sqrt(re^2 + im^2)
                }
                (acc / (stop - start) as f64) as f32
            } else {
                0.0
            };
            raw_bands[b] = val;
            bands[b] = agc_step(&mut self.agc_max[b], val);
        }

        Frame {
            bands,
            raw_bands,
            amp,
        }
    }
}

/// One peak-follower AGC step shared by the bands and the overall amplitude.
/// Instantaneous attack, geometric decay at AGC_DECAY/frame, then normalise the
/// current value by max(running_max, AGC_FLOOR) and clamp the ratio to <= 1.0.
/// The ratio is naturally >= 0 (val >= 0, denom > 0), so there is no lower clamp.
#[inline]
fn agc_step(running_max: &mut f32, val: f32) -> f32 {
    // Decay the stored max, then snap up to `val` if it exceeds the decayed max.
    *running_max = val.max(*running_max * AGC_DECAY);
    (val / running_max.max(AGC_FLOOR)).min(1.0)
}

// =====================================================================
// COMPONENT 2 — OnsetDetector
// =====================================================================

/// Per-band decaying onset envelopes (0..1) and the single summed novelty.
#[derive(Clone, Copy)]
pub struct OnsetOut {
    pub envelopes: [f32; N_BANDS],
    pub novelty: f32,
}

pub struct OnsetDetector {
    /// Per-band ring of recent rectified flux values (maxlen = history_size).
    flux_history: [VecDeque<f32>; N_BANDS],
    /// Capacity of each flux ring (= 43 with the Python default 43.0 fps).
    history_size: usize,
    /// Previous frame's raw band value, for the flux difference.
    prev_raw: [f32; N_BANDS],
    /// Decaying per-band onset envelopes (0..1).
    envelopes: [f32; N_BANDS],
    /// Wall-clock time of the last onset per band (-1.0 = never; first allowed).
    last_onset_t: [f32; N_BANDS],
}

impl OnsetDetector {
    pub fn new() -> Self {
        // history_size = max(10, int(ONSET_HISTORY_SECONDS * 43.0)) = 43.
        let history_size = (ONSET_HISTORY_SECONDS * ONSET_FRAME_RATE) as usize; // int() truncates
        let history_size = history_size.max(10);
        Self {
            flux_history: std::array::from_fn(|_| VecDeque::with_capacity(history_size)),
            history_size,
            prev_raw: [0.0; N_BANDS],
            envelopes: [0.0; N_BANDS],
            last_onset_t: [-1.0; N_BANDS],
        }
    }

    /// One frame of onset detection over the Analyser's *raw* (pre-AGC) band
    /// means. `t_secs` is monotonically-increasing wall-clock seconds; only its
    /// deltas matter (refractory check). Returns the per-band envelopes and the
    /// summed novelty (the BeatTracker's input).
    pub fn process(&mut self, raw_bands: &[f32; N_BANDS], t_secs: f32) -> OnsetOut {
        let mut total_flux = 0.0f32;

        for b in 0..N_BANDS {
            let current = raw_bands[b];
            let prev = self.prev_raw[b];

            // Rectified spectral flux: positive part of the frame-to-frame
            // increase on the raw magnitude. Decreases clip to 0.
            let flux = (current - prev).max(0.0);
            total_flux += flux;

            // Append THEN compute the mean, so the mean includes this frame.
            let hist = &mut self.flux_history[b];
            if hist.len() == self.history_size {
                hist.pop_front();
            }
            hist.push_back(flux);

            // Gate: need >=10 frames of history AND flux above the absolute floor.
            if hist.len() >= 10 && flux > ONSET_FLUX_FLOOR {
                let sum: f32 = hist.iter().sum();
                let mean = sum / hist.len() as f32;
                // Threshold = max(floor, mean * 3.5): flux must beat both.
                let threshold = ONSET_FLUX_FLOOR.max(mean * ONSET_THRESHOLD_MULT);

                // Refractory: >150 ms (wall-clock) since this band's last onset.
                if flux > threshold && t_secs - self.last_onset_t[b] > ONSET_REFRACTORY_SECONDS {
                    // Envelope bump magnitude uses a DIFFERENT denominator
                    // (mean*4, not mean*3.5). Envelope takes the max, so a
                    // weaker later onset never pulls a still-high envelope down.
                    let relative = flux / (mean * 4.0 + 1e-6);
                    self.envelopes[b] = self.envelopes[b].max(relative.min(1.0));
                    self.last_onset_t[b] = t_secs;
                }
            }

            // Decay EVERY frame, AFTER the possible bump (so a freshly bumped
            // envelope is immediately multiplied by 0.88 this same frame).
            self.envelopes[b] *= ONSET_ENVELOPE_DECAY;

            self.prev_raw[b] = current;
        }

        OnsetOut {
            envelopes: self.envelopes,
            novelty: total_flux,
        }
    }
}

impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new()
    }
}

// =====================================================================
// COMPONENT 3 — BeatTracker
// =====================================================================

/// Per-frame beat estimate. `period` is seconds/beat; floats are reported
/// rounded to 4 decimals to match the Python wire payload. `beat` is a one-frame
/// pulse on a predicted beat instant.
#[derive(Clone, Copy)]
pub struct BeatOut {
    pub bpm: f32,
    pub period: f32,
    pub phase: f32,
    pub next_beat_in: f32,
    pub confidence: f32,
    pub beat: bool,
}

pub struct BeatTracker {
    frame_rate: f32,
    frame_dt: f32,

    /// Novelty ring (maxlen = int(8 * frame_rate) = 344).
    novelty: VecDeque<f32>,
    novelty_cap: usize,

    /// Autocorrelation lag search bounds, in frames.
    lag_min: usize, // = 11  (~234.9 BPM)
    lag_max: usize, // = 86  (~30.05 BPM)
    /// Retrack runs once >= this many frames have elapsed since the last retrack.
    retrack_frames: usize, // = 22
    /// No retrack until the novelty buffer holds at least this many frames.
    min_track_frames: usize, // = 172
    frames_since_track: usize,

    /// Absolute frame counter; -1 so the first process() lands on 0.
    frame_idx: i64,

    /// Median filter over the last few folded BPM estimates.
    bpm_history: VecDeque<f32>,

    bpm: f32,
    period_frames: f32,
    confidence: f32,
    /// Absolute (fractional) frame index of a beat instant — the phase grid anchor.
    beat_anchor: f32,
    /// Frame index of the last fired `beat` flag (-1e9 = never).
    last_beat_fire: f32,
    /// False until the first fully-completed retrack anchors a grid.
    have_estimate: bool,
}

impl BeatTracker {
    pub fn new(frame_rate: f32) -> Self {
        // All integer constants below replicate Python int()/round() semantics
        // (int truncates toward zero; round is half-to-even — but none of these
        // arguments sit on a .5 boundary, so plain rounding agrees).
        let novelty_cap = (BEAT_NOVELTY_SECONDS * frame_rate) as usize; // int() = 344
        let lag_min = 2usize.max(py_round(BEAT_LAG_MIN_SECONDS * frame_rate) as usize); // 11
        let lag_max = py_round(BEAT_LAG_MAX_SECONDS * frame_rate) as usize; // 86
        let retrack_frames = 1usize.max(py_round(BEAT_RETRACK_SECONDS * frame_rate) as usize); // 22
        let min_track_frames = (2.0 * BEAT_LAG_MAX_SECONDS * frame_rate) as usize; // int() = 172

        Self {
            frame_rate,
            frame_dt: 1.0 / frame_rate,
            novelty: VecDeque::with_capacity(novelty_cap),
            novelty_cap,
            lag_min,
            lag_max,
            retrack_frames,
            min_track_frames,
            frames_since_track: 0,
            frame_idx: -1,
            bpm_history: VecDeque::with_capacity(BEAT_MEDIAN_COUNT),
            bpm: 0.0,
            period_frames: 0.0,
            confidence: 0.0,
            beat_anchor: 0.0,
            last_beat_fire: -1e9,
            have_estimate: false,
        }
    }

    /// One frame: push novelty, maybe retrack, then extrapolate phase from the
    /// anchored grid and emit the estimate. `t_secs` is accepted for interface
    /// symmetry but unused — the BeatTracker runs entirely on its integer frame
    /// counter (the Python tracker likewise uses frame_idx, not wall-clock).
    pub fn process(&mut self, novelty: f32, _t_secs: f32) -> BeatOut {
        self.frame_idx += 1;
        if self.novelty.len() == self.novelty_cap {
            self.novelty.pop_front();
        }
        self.novelty.push_back(novelty);

        // Retrack gate: >=retrack_frames since last retrack AND buffer full enough.
        self.frames_since_track += 1;
        if self.frames_since_track >= self.retrack_frames
            && self.novelty.len() >= self.min_track_frames
        {
            self.retrack();
            self.frames_since_track = 0;
        }

        if !self.have_estimate {
            return BeatOut {
                bpm: 0.0,
                period: 0.0,
                phase: 0.0,
                next_beat_in: 0.0,
                confidence: 0.0,
                beat: false,
            };
        }

        // Per-frame extrapolation from the anchored grid.
        let period_s = self.period_frames * self.frame_dt;
        // Fraction-through-current-beat in [0,1). Python's float % is non-negative
        // (sign of divisor); use rem_euclid to match for any sign of the numerator.
        let phase = ((self.frame_idx as f32 - self.beat_anchor) / self.period_frames).rem_euclid(1.0);
        let next_in = (1.0 - phase) * period_s;

        // Beat flag: confidence-gated, near a beat instant, and at least half a
        // period since the last fire (refractory).
        let mut fired = false;
        let frames_from_beat = phase.min(1.0 - phase) * self.period_frames;
        if self.confidence >= BEAT_MIN_BEAT_CONFIDENCE
            && frames_from_beat <= 0.5
            && self.frame_idx as f32 - self.last_beat_fire > 0.5 * self.period_frames
        {
            fired = true;
            self.last_beat_fire = self.frame_idx as f32;
        }

        BeatOut {
            bpm: round4(self.bpm),
            period: round4(period_s),
            phase: round4(phase),
            next_beat_in: round4(next_in),
            confidence: round4(self.confidence),
            beat: fired,
        }
    }

    /// Heavy path (~every 0.5 s): autocorrelation tempo estimate with a
    /// log-Gaussian prior, parabolic peak refinement, octave fold, median
    /// filter, harmonic-support confidence, then phase re-anchor.
    fn retrack(&mut self) {
        let n = self.novelty.len();
        let mut x: Vec<f64> = self.novelty.iter().map(|&v| v as f64).collect();

        // Silence floor on the RAW (non-centred) mean. Below it: confidence 0
        // and early-return WITHOUT touching have_estimate / bpm / anchor — the
        // last known grid keeps extrapolating (the beat flag is suppressed by
        // the confidence gate anyway). Note this floor is on pre-AGC absolute
        // novelty, so it scales with capture gain.
        let mean_raw = x.iter().sum::<f64>() / n as f64;
        if (mean_raw as f32) < BEAT_SILENCE_FLOOR {
            self.confidence = 0.0;
            return;
        }

        // Mean-subtract (zero-centre) AFTER the silence check.
        for v in x.iter_mut() {
            *v -= mean_raw;
        }

        // Clamp the upper lag to half the buffer; bail if the lag range is tiny.
        let lag_max = self.lag_max.min(n / 2);
        if lag_max < self.lag_min || lag_max - self.lag_min < 4 {
            return; // too little lag range — leave confidence unchanged
        }

        // Unnormalised autocorrelation at non-negative lags:
        //   ac[k] = sum_{m=0}^{n-1-k} x[m] * x[m+k],  k = 0..n-1.
        // We only need ac[0] (for normalisation) and ac[lag_min..=lag_max].
        let ac0: f64 = x.iter().map(|&v| v * v).sum();
        if ac0 <= 1e-12 {
            self.confidence = 0.0; // zero energy
            return;
        }

        // seg[idx] corresponds to lag = lag_min + idx, normalised by ac[0].
        let n_lags = lag_max - self.lag_min + 1;
        let mut seg = vec![0.0f64; n_lags];
        for (idx, lag) in (self.lag_min..=lag_max).enumerate() {
            let mut acc = 0.0f64;
            for m in 0..(n - lag) {
                acc += x[m] * x[m + lag];
            }
            seg[idx] = acc / ac0;
        }

        // Log-Gaussian prior over candidate BPMs, centred at 120 BPM,
        // sigma 0.4 octaves — this breaks half/double-tempo ambiguity.
        // weighted[idx] = seg[idx] * prior(bpm_of_lag).
        let mut weighted = vec![0.0f64; n_lags];
        let mut best_i = 0usize;
        let mut best_w = f64::NEG_INFINITY;
        for (idx, lag) in (self.lag_min..=lag_max).enumerate() {
            let bpm = 60.0 * self.frame_rate as f64 / lag as f64;
            let z = (bpm / BEAT_PRIOR_BPM as f64).log2() / BEAT_PRIOR_SIGMA_LOG2 as f64;
            let prior = (-0.5 * z * z).exp();
            let w = seg[idx] * prior;
            weighted[idx] = w;
            if w > best_w {
                best_w = w;
                best_i = idx;
            }
        }

        // Parabolic interpolation around the peak (on the PRIOR-WEIGHTED values),
        // for sub-frame lag precision. Only when the peak is interior.
        let mut delta = 0.0f64;
        if best_i > 0 && best_i < n_lags - 1 {
            let a = weighted[best_i - 1];
            let b = weighted[best_i];
            let c = weighted[best_i + 1];
            let denom = a - 2.0 * b + c;
            if denom.abs() > 1e-12 {
                delta = (0.5 * (a - c) / denom).clamp(-0.5, 0.5);
            }
        }
        let lag_star = (self.lag_min + best_i) as f64 + delta;

        // Octave-fold the instantaneous BPM into [70, 180].
        let mut bpm = 60.0 * self.frame_rate as f64 / lag_star;
        while bpm < BEAT_FOLD_LO_BPM as f64 {
            bpm *= 2.0;
        }
        while bpm > BEAT_FOLD_HI_BPM as f64 {
            bpm /= 2.0;
        }

        // Median filter the folded BPM; report and derive period from the MEDIAN.
        if self.bpm_history.len() == BEAT_MEDIAN_COUNT {
            self.bpm_history.pop_front();
        }
        self.bpm_history.push_back(bpm as f32);
        self.bpm = median(&self.bpm_history);
        self.period_frames = 60.0 * self.frame_rate / self.bpm;

        // --- Confidence: peak's share of positive AC mass, weighted by support
        //     at a harmonic lag (2x, or 0.5x near the top of the search range). ---
        let pos_sum: f64 = seg.iter().map(|&v| v.max(0.0)).sum();
        let peak_ratio = seg[best_i].max(0.0) / (pos_sum + 1e-9);

        let lag_n = py_round(lag_star as f32) as i64;
        // Harmonic support lag: 2x if it fits, else 0.5x (floored, clamped >= lag_min).
        let support_lag = if (lag_n * 2) as usize <= lag_max {
            (lag_n * 2) as usize
        } else {
            (self.lag_min as i64).max(lag_n / 2) as usize
        };
        // ±2-frame window around the support lag, within seg bounds.
        let j = support_lag as i64 - self.lag_min as i64;
        let lo = (j - 2).max(0) as usize;
        let hi = ((j + 3).max(0) as usize).min(n_lags); // exclusive, auto-truncate at end
        let support = if hi > lo {
            seg[lo..hi]
                .iter()
                .map(|&v| v.clamp(0.0, 1.0))
                .fold(0.0f64, f64::max)
        } else {
            0.0
        };

        let raw = peak_ratio * support;
        self.confidence =
            (((raw - BEAT_CONF_BASE as f64) * BEAT_CONF_SCALE as f64).clamp(0.0, 1.0)) as f32;

        self.rephase();
        self.have_estimate = true;
    }

    /// Phase anchoring via comb correlation: slide a pulse train (spacing =
    /// period_frames) across the most-recent window of novelty, score each
    /// quarter-frame offset by the MEAN interpolated novelty under its pulses,
    /// and anchor the grid to the winning offset.
    fn rephase(&mut self) {
        // Window = min(buffer_len, int(4 * frame_rate)) = min(len, 172).
        let phase_window = (BEAT_PHASE_SECONDS * self.frame_rate) as usize; // int() = 172
        let w = self.novelty.len().min(phase_window);
        let p = self.period_frames as f64;
        // Skip on too-short period or too-short window (keep the old anchor).
        if p <= 1.0 || (w as f64) < p + 2.0 {
            return;
        }

        // y = the most recent w novelty samples (the buffer's tail).
        let start = self.novelty.len() - w;
        let y: Vec<f64> = self.novelty.iter().skip(start).map(|&v| v as f64).collect();

        // Offset search: off = 0, 0.25, 0.5, ... up to (not incl.) p.
        // Score = MEAN over pulses of linearly-interpolated novelty at each
        // pulse position pos = off, off+p, off+2p, ... up to (not incl.) w-1.
        // Mean (not sum) so offsets that fit one fewer pulse aren't penalised.
        let mut best_off = 0.0f64;
        let mut best_score = f64::NEG_INFINITY;
        let mut off = 0.0f64;
        while off < p {
            let mut acc = 0.0f64;
            let mut count = 0usize;
            let mut pos = off;
            while pos < w as f64 - 1.0 {
                let i0 = pos as usize; // floor (pos >= 0)
                let frac = pos - i0 as f64;
                acc += y[i0] * (1.0 - frac) + y[i0 + 1] * frac;
                count += 1;
                pos += p;
            }
            if count > 0 {
                let score = acc / count as f64;
                if score > best_score {
                    best_score = score;
                    best_off = off;
                }
            }
            off += 0.25;
        }

        // y[0] is absolute frame (frame_idx - (w-1)); the winning train's first
        // pulse (y[best_off]) is therefore a beat instant at this absolute frame.
        self.beat_anchor = self.frame_idx as f32 - (w as f32 - 1.0) + best_off as f32;
    }
}

// --- small numeric helpers replicating Python/NumPy semantics ---

/// Python `round(x)` for the half-to-even rule, returning an i64. Used only for
/// values that never sit on a .5 boundary in practice, but kept faithful.
#[inline]
fn py_round(x: f32) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        // exactly .5 → round to even
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// `round(x, 4)` — round to 4 decimal places (matches the Python wire payload).
#[inline]
fn round4(x: f32) -> f32 {
    (x * 10_000.0).round() / 10_000.0
}

/// NumPy-style median: for even counts, the mean of the two central order
/// statistics (NOT the lower-middle); for odd counts, the middle value.
fn median(values: &VecDeque<f32>) -> f32 {
    let mut v: Vec<f32> = values.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 44100;
    /// Exact frame rate the BeatTracker uses (44100/1024).
    const FRAME_RATE: f32 = SR as f32 / BLOCK_SIZE as f32;

    fn band(name: &str) -> usize {
        match name {
            "subBass" => 0,
            "bass" => 1,
            "lowMid" => 2,
            "mid" => 3,
            "upperMid" => 4,
            "treble" => 5,
            "air" => 6,
            _ => panic!("bad band"),
        }
    }

    /// Verifies the resolved bin ranges match the spec's verification table.
    #[test]
    fn band_bins_match_spec() {
        let a = Analyser::new(SR);
        assert_eq!(a.band_bins[0], (1, 2), "subBass single bin");
        assert_eq!(a.band_bins[1], (2, 6), "bass");
        assert_eq!(a.band_bins[2], (6, 12), "lowMid");
        assert_eq!(a.band_bins[3], (12, 47), "mid");
        assert_eq!(a.band_bins[4], (47, 93), "upperMid");
        assert_eq!(a.band_bins[5], (93, 186), "treble");
        assert_eq!(a.band_bins[6], (186, 465), "air");
    }

    /// The symmetric Hann window has exact-zero endpoints and a unity centre.
    #[test]
    fn hann_window_symmetric() {
        let a = Analyser::new(SR);
        assert!(a.window[0].abs() < 1e-7, "w[0] must be 0");
        assert!(a.window[BLOCK_SIZE - 1].abs() < 1e-7, "w[M-1] must be 0");
        // Centre (between the two middle samples) is ~1.0.
        let mid = a.window[BLOCK_SIZE / 2];
        assert!(mid > 0.999, "centre near unity, got {mid}");
    }

    /// (a) Silence → all bands ~0 and amp ~0 (after many frames so AGC settles).
    #[test]
    fn silence_gives_zero_bands() {
        let mut a = Analyser::new(SR);
        let silent = vec![0.0f32; BLOCK_SIZE];
        let mut last = a.process(&silent);
        for _ in 0..200 {
            last = a.process(&silent);
        }
        for b in 0..N_BANDS {
            assert!(
                last.bands[b] < 1e-3,
                "band {b} should be ~0 on silence, got {}",
                last.bands[b]
            );
            assert!(
                last.raw_bands[b] < 1e-6,
                "raw band {b} should be ~0 on silence, got {}",
                last.raw_bands[b]
            );
        }
        assert!(last.amp < 1e-3, "amp should be ~0 on silence, got {}", last.amp);
    }

    /// (b) A 100 Hz sine concentrates energy in subBass/bass, negligible in treble.
    #[test]
    fn sine_100hz_energy_in_low_bands() {
        let mut a = Analyser::new(SR);
        let freq = 100.0f32;
        let mut phase = 0.0f32;
        let dp = 2.0 * std::f32::consts::PI * freq / SR as f32;
        let mut last = Frame {
            bands: [0.0; N_BANDS],
            raw_bands: [0.0; N_BANDS],
            amp: 0.0,
        };
        // Feed contiguous blocks so the phase is continuous across blocks.
        for _ in 0..50 {
            let mut block = vec![0.0f32; BLOCK_SIZE];
            for s in block.iter_mut() {
                *s = 0.5 * phase.sin();
                phase += dp;
            }
            last = a.process(&block);
        }
        // 100 Hz lands in the bass band (60..250). subBass (20..60, only bin 1
        // at 43 Hz) and bass should carry the bulk via spectral leakage; treble
        // (4000..8000) must be negligible.
        let low = last.raw_bands[band("subBass")] + last.raw_bands[band("bass")];
        let treble = last.raw_bands[band("treble")];
        assert!(low > 0.0, "low-band energy must be present");
        assert!(
            treble < low * 1e-3,
            "treble {treble} must be negligible vs low {low}"
        );
        // Bass should specifically dominate (100 Hz is inside it).
        assert!(
            last.raw_bands[band("bass")] > last.raw_bands[band("mid")],
            "bass must exceed mid for a 100 Hz tone"
        );
        // Normalised bass should be near its AGC ceiling for a steady tone.
        assert!(
            last.bands[band("bass")] > 0.9,
            "normalised bass should saturate for a steady tone, got {}",
            last.bands[band("bass")]
        );
    }

    /// Onset detector fires on a sharp rise and the envelope then decays.
    #[test]
    fn onset_fires_and_decays() {
        let mut od = OnsetDetector::new();
        let dt = 1.0 / FRAME_RATE;
        let mut t = 0.0f32;
        let bass = band("bass");

        // Quiet floor for >10 frames so history fills.
        for _ in 0..20 {
            let mut raw = [0.0f32; N_BANDS];
            raw[bass] = 0.001;
            od.process(&raw, t);
            t += dt;
        }
        // A sharp jump on the bass band → onset.
        let mut raw = [0.0f32; N_BANDS];
        raw[bass] = 0.5;
        let out = od.process(&raw, t);
        t += dt;
        assert!(out.envelopes[bass] > 0.1, "envelope should bump on onset");
        assert!(out.novelty > 0.0, "novelty should be positive on a rise");

        // Hold steady (flux ~0) → envelope decays toward 0.
        let env_after_onset = out.envelopes[bass];
        let mut env = env_after_onset;
        for _ in 0..20 {
            let mut raw = [0.0f32; N_BANDS];
            raw[bass] = 0.5;
            env = od.process(&raw, t).envelopes[bass];
            t += dt;
        }
        assert!(env < env_after_onset * 0.5, "envelope must decay when steady");
    }

    // --- BeatTracker helpers ---

    /// Builds a frame-rate impulse novelty stream at `bpm` with mild noise, runs
    /// it through a fresh BeatTracker, and returns (final BeatOut, max confidence).
    fn run_beat_stream(bpm: f32, seconds: f32, noise_amp: f32, seed: u64) -> (BeatOut, f32) {
        let mut bt = BeatTracker::new(FRAME_RATE);
        let period_frames = 60.0 / bpm * FRAME_RATE;
        let total = (seconds * FRAME_RATE) as usize;
        let mut rng = seed;
        let dt = 1.0 / FRAME_RATE;
        let mut out = BeatOut {
            bpm: 0.0,
            period: 0.0,
            phase: 0.0,
            next_beat_in: 0.0,
            confidence: 0.0,
            beat: false,
        };
        let mut max_conf = 0.0f32;
        let mut next_beat = 0.0f32;
        for f in 0..total {
            // xorshift noise in [0, noise_amp).
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let r = (rng as f32 / u64::MAX as f32) * noise_amp;
            // Impulse when this frame index crosses the next scheduled beat.
            let mut nov = r;
            if (f as f32) >= next_beat {
                nov += 1.0;
                next_beat += period_frames;
            }
            out = bt.process(nov, f as f32 * dt);
            if out.confidence > max_conf {
                max_conf = out.confidence;
            }
        }
        (out, max_conf)
    }

    /// (c) 130 BPM impulse stream converges to ~130 BPM and is much more
    /// confident than a pure white-noise novelty stream.
    #[test]
    fn beat_tracker_converges_to_130_and_beats_noise() {
        let (out, conf_beat) = run_beat_stream(130.0, 10.0, 0.05, 0x1234_5678);
        assert!(
            (out.bpm - 130.0).abs() <= 2.0,
            "bpm should converge to 130 +/- 2, got {}",
            out.bpm
        );

        // Pure white-noise novelty (no periodic structure) → low confidence.
        let mut bt = BeatTracker::new(FRAME_RATE);
        let total = (10.0 * FRAME_RATE) as usize;
        let mut rng: u64 = 0x9E37_79B9_7F4A_7C15;
        let dt = 1.0 / FRAME_RATE;
        let mut max_conf_noise = 0.0f32;
        for f in 0..total {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Keep the noise mean well above BEAT_SILENCE_FLOOR so confidence is
            // actually computed (not short-circuited to 0 by the silence gate);
            // this makes the comparison a real periodicity test.
            let nov = 0.3 + (rng as f32 / u64::MAX as f32) * 0.4;
            let o = bt.process(nov, f as f32 * dt);
            if o.confidence > max_conf_noise {
                max_conf_noise = o.confidence;
            }
        }

        assert!(
            conf_beat > max_conf_noise + 0.1,
            "beat confidence {conf_beat} must be much higher than noise confidence {max_conf_noise}"
        );
        // And the beat stream should reach a solidly-confident level.
        assert!(
            conf_beat > 0.15,
            "steady beat confidence should be solid, got {conf_beat}"
        );
    }

    /// (d) next_beat_in decreases between consecutive beats (the countdown).
    #[test]
    fn next_beat_in_decreases_between_beats() {
        let mut bt = BeatTracker::new(FRAME_RATE);
        let bpm = 120.0f32;
        let period_frames = 60.0 / bpm * FRAME_RATE;
        let total = (12.0 * FRAME_RATE) as usize;
        let dt = 1.0 / FRAME_RATE;
        let mut next_beat = 0.0f32;

        // Warm up until we have an estimate and a real countdown.
        let mut series: Vec<(f32, bool)> = Vec::new();
        for f in 0..total {
            let mut nov = 0.02f32; // mild floor so we're above silence
            if (f as f32) >= next_beat {
                nov += 1.0;
                next_beat += period_frames;
            }
            let o = bt.process(nov, f as f32 * dt);
            if o.confidence > 0.0 {
                series.push((o.next_beat_in, o.beat));
            }
        }
        assert!(series.len() > 100, "need a usable run of estimates");

        // Find a window strictly between two beat instants and confirm the
        // countdown is (weakly) monotonically decreasing there.
        // Locate two successive frames where `beat` fired.
        let beats: Vec<usize> = series
            .iter()
            .enumerate()
            .filter(|(_, (_, b))| *b)
            .map(|(i, _)| i)
            .collect();
        assert!(beats.len() >= 2, "expected multiple beat fires, got {}", beats.len());
        // Take the interval just after the first fire up to just before the next.
        let a = beats[0] + 1;
        let b = beats[1];
        assert!(b > a + 2, "need room between beats");
        let mut prev = series[a].0;
        let mut decreased = false;
        for k in (a + 1)..b {
            let cur = series[k].0;
            // Allow tiny rounding wiggle (values are round4'd).
            assert!(
                cur <= prev + 1e-3,
                "next_beat_in should not increase mid-beat: {prev} -> {cur}"
            );
            if cur < prev - 1e-4 {
                decreased = true;
            }
            prev = cur;
        }
        assert!(decreased, "next_beat_in should visibly decrease between beats");
    }

    /// Sanity: before the first retrack the tracker reports all-zero / no-beat.
    #[test]
    fn beat_tracker_silent_before_first_retrack() {
        let mut bt = BeatTracker::new(FRAME_RATE);
        // Fewer than min_track_frames (172) → never retracks → all zeros.
        for f in 0..100 {
            let o = bt.process(0.5, f as f32 / FRAME_RATE);
            assert_eq!(o.bpm, 0.0);
            assert!(!o.beat);
            assert_eq!(o.confidence, 0.0);
        }
    }
}
