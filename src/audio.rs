//! Audio layer: websocket client for feeder_onsets.py + the smoothing /
//! mapping engine — a faithful port of the HTML build's audio driving block.
//!
//! Channel split (load-bearing, see docs/audio-mappings.md): MOTION is driven
//! by subBass (20–60 Hz), COLOUR by bass (60–250 Hz). Don't re-unify them.
//!
//! The HTML's smoothing alphas were per-frame at 60 fps; here smoothing runs
//! per sim tick, so every alpha is converted with
//!   alpha_tick = 1 - (1 - alpha_60)^(60 * dt)
//! which preserves the exact time constants at any tick rate. The feeder
//! pushes ~43 msg/s; between messages the envelopes keep easing toward the
//! latest value, which is the same zero-order-hold the HTML had.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

#[derive(Deserialize, Default, Clone, Copy)]
pub struct Bands {
    #[serde(default, rename = "subBass")]
    pub sub_bass: f32,
    #[serde(default)]
    pub bass: f32,
    #[serde(default, rename = "lowMid")]
    pub low_mid: f32,
    #[serde(default)]
    pub mid: f32,
    #[serde(default, rename = "upperMid")]
    pub upper_mid: f32,
    #[serde(default)]
    pub treble: f32,
    #[serde(default)]
    pub air: f32,
}

#[derive(Deserialize)]
struct FeederMsg {
    #[serde(default)]
    bands: Option<Bands>,
    #[serde(default)]
    onset_envelopes: Option<Bands>,
}

#[derive(Default)]
struct Shared {
    bands: Bands,
    onset_envelopes: Bands,
    connected: bool,
    last_msg: Option<Instant>,
}

pub struct AudioClient {
    shared: Arc<Mutex<Shared>>,
}

impl AudioClient {
    /// Spawns a background thread that connects (and reconnects, 1s backoff,
    /// same as the HTML) and keeps `shared` holding the latest frame.
    pub fn connect(url: String) -> Self {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let thread_shared = shared.clone();
        std::thread::spawn(move || loop {
            match tungstenite::connect(&url) {
                Ok((mut socket, _)) => {
                    log::info!("audio feeder connected ({url})");
                    thread_shared.lock().unwrap().connected = true;
                    loop {
                        match socket.read() {
                            Ok(msg) => {
                                if let Ok(text) = msg.into_text() {
                                    if let Ok(parsed) =
                                        serde_json::from_str::<FeederMsg>(text.as_str())
                                    {
                                        let mut s = thread_shared.lock().unwrap();
                                        if let Some(b) = parsed.bands {
                                            s.bands = b;
                                        }
                                        if let Some(e) = parsed.onset_envelopes {
                                            s.onset_envelopes = e;
                                        }
                                        s.last_msg = Some(Instant::now());
                                    }
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    thread_shared.lock().unwrap().connected = false;
                    log::warn!("audio feeder disconnected — retrying");
                }
                Err(_) => {
                    // Feeder not running yet — keep trying quietly.
                }
            }
            std::thread::sleep(Duration::from_secs(1));
        });
        Self { shared }
    }

    fn snapshot(&self) -> (Bands, Bands, bool) {
        let s = self.shared.lock().unwrap();
        let fresh = s.connected
            && s.last_msg
                .is_some_and(|t| t.elapsed() < Duration::from_millis(500));
        (s.bands, s.onset_envelopes, fresh)
    }
}

/// All mapping tunables — values are the HTML build's Kiseia-tuned defaults.
pub struct Mapping {
    pub enabled: bool,
    // subBass → attract (fast asymmetric) + separation/verticality (slow symmetric)
    pub sub_bass_attack: f32,
    pub sub_bass_release: f32,
    pub separation_smoothing: f32,
    pub attract_floor: f32,
    pub attract_ceiling: f32,
    pub separation_floor: f32,
    pub separation_ceiling: f32,
    pub vertical_lift: f32,
    // subBass → freedom (inverse)
    pub freedom_enabled: bool,
    pub freedom_floor: f32,   // freedom @ subBass=1 (compressed unity)
    pub freedom_ceiling: f32, // freedom @ subBass=0 (restless calm)
    // treble → maxSpeed
    pub treble_attack: f32,
    pub treble_release: f32,
    pub max_speed_floor: f32,
    pub max_speed_ceiling: f32,
    // mid → alignment, lowMid → cohesion (slow symmetric)
    pub mid_smoothing: f32,
    pub alignment_floor: f32,
    pub alignment_ceiling: f32,
    pub low_mid_smoothing: f32,
    pub cohesion_floor: f32,
    pub cohesion_ceiling: f32,
    // upperMid → turn-rate + shear + partial speed
    pub upper_mid_attack: f32,
    pub upper_mid_release: f32,
    pub upper_mid_gate: f32,
    pub upper_mid_curve: f32,
    pub turn_rate_enabled: bool,
    pub turn_floor: f32,
    pub turn_ceiling: f32,
    pub shear_enabled: bool,
    pub shear_floor: f32,
    pub shear_ceiling: f32,
    pub upper_mid_speed_enabled: bool,
    pub upper_mid_speed_factor: f32,
    // air → twinkle
    pub air_smoothing: f32,
    pub twinkle_strength: f32,
    pub twinkle_invert: bool,
    pub twinkle_enabled: bool,
    // bass onset → shockwave (master toggle default OFF, per the HTML)
    pub shockwave_enabled: bool,
    // colour: bass → paletteT (dark-trough swell + kick punctuation)
    pub t_scale: f32,
    pub t_offset: f32,
    pub onset_boost: f32,
    pub onset_smoothing: f32,
}

impl Default for Mapping {
    fn default() -> Self {
        Self {
            enabled: true,
            sub_bass_attack: 0.5,
            sub_bass_release: 0.05,
            separation_smoothing: 0.03,
            attract_floor: 0.8,
            attract_ceiling: 3.0,
            separation_floor: 14.0,
            separation_ceiling: 18.0,
            vertical_lift: 80.0,
            freedom_enabled: true,
            freedom_floor: 0.3,
            freedom_ceiling: 1.3,
            treble_attack: 0.4,
            treble_release: 0.15,
            max_speed_floor: 9.0,
            max_speed_ceiling: 16.0,
            mid_smoothing: 0.05,
            alignment_floor: 25.0,
            alignment_ceiling: 38.0,
            low_mid_smoothing: 0.05,
            cohesion_floor: 35.0,
            cohesion_ceiling: 80.0,
            upper_mid_attack: 0.5,
            upper_mid_release: 0.08,
            upper_mid_gate: 0.08,
            upper_mid_curve: 1.6,
            turn_rate_enabled: true,
            turn_floor: 5.0,
            turn_ceiling: 18.0,
            shear_enabled: true,
            shear_floor: 0.0,
            shear_ceiling: 1.2,
            upper_mid_speed_enabled: true,
            upper_mid_speed_factor: 0.5,
            air_smoothing: 0.1,
            twinkle_strength: 0.20,
            twinkle_invert: true,
            twinkle_enabled: false,
            shockwave_enabled: false,
            t_scale: 0.45,
            t_offset: 0.11,
            onset_boost: 0.05,
            onset_smoothing: 0.12,
        }
    }
}

/// Per-tick outputs — what actually gets written into SimParams / render
/// uniforms this tick.
pub struct Driven {
    pub attract: f32,
    pub separation: f32,
    pub alignment: f32,
    pub cohesion: f32,
    pub freedom: f32,
    pub max_speed: f32,
    pub max_turn_rate: f32,
    pub shear: f32,
    pub shape_center_y: f32,
    pub shock_strength: f32,
    pub palette_t: f32,
    pub twinkle: f32,
    pub connected: bool,
}

#[derive(Default)]
struct Smoothed {
    bass: f32,
    bass_onset_for_colour: f32,
    sub_bass: f32,
    sub_bass_for_sep: f32,
    treble: f32,
    mid: f32,
    low_mid: f32,
    upper_mid: f32,
    air: f32,
}

pub struct AudioEngine {
    client: AudioClient,
    pub mapping: Mapping,
    sm: Smoothed,
}

/// Convert a per-frame-at-60fps alpha into this tick's alpha.
fn alpha(alpha_60: f32, dt: f32) -> f32 {
    1.0 - (1.0 - alpha_60).powf(60.0 * dt)
}

fn ease(current: &mut f32, target: f32, alpha_60: f32, dt: f32) {
    let a = alpha(alpha_60, dt);
    *current += (target - *current) * a;
}

impl AudioEngine {
    pub fn new(url: String) -> Self {
        Self {
            client: AudioClient::connect(url),
            mapping: Mapping::default(),
            sm: Smoothed::default(),
        }
    }

    /// Advance all envelopes by one sim tick and produce the driven values.
    /// Static fallbacks (mapping disabled) match effectController defaults.
    pub fn tick(&mut self, dt: f32, defaults: &crate::params::Settings) -> Driven {
        let (bands, envelopes, fresh) = self.client.snapshot();
        let m = &self.mapping;

        if !m.enabled {
            return Driven {
                attract: defaults.attraction_strength,
                separation: defaults.separation,
                alignment: defaults.alignment,
                cohesion: defaults.cohesion,
                freedom: defaults.freedom,
                max_speed: defaults.max_speed,
                max_turn_rate: 1000.0,
                shear: 0.0,
                shape_center_y: 0.0,
                shock_strength: 0.0,
                palette_t: m.t_offset,
                twinkle: 0.0,
                connected: fresh,
            };
        }

        // === Envelopes (always computed, exactly as the HTML does).
        // bass: fast asymmetric — COLOUR only.
        let ba = if bands.bass > self.sm.bass {
            m.sub_bass_attack
        } else {
            m.sub_bass_release
        };
        ease(&mut self.sm.bass, bands.bass, ba, dt);

        // subBass: fast asymmetric — attract + freedom(inverse).
        let sba = if bands.sub_bass > self.sm.sub_bass {
            m.sub_bass_attack
        } else {
            m.sub_bass_release
        };
        ease(&mut self.sm.sub_bass, bands.sub_bass, sba, dt);
        let t_attract = self.sm.sub_bass.clamp(0.0, 1.0);

        // subBass: slow symmetric — separation + verticality.
        ease(&mut self.sm.sub_bass_for_sep, bands.sub_bass, m.separation_smoothing, dt);
        let t_sep = self.sm.sub_bass_for_sep.clamp(0.0, 1.0);

        // treble: fast asymmetric — maxSpeed.
        let ta = if bands.treble > self.sm.treble {
            m.treble_attack
        } else {
            m.treble_release
        };
        ease(&mut self.sm.treble, bands.treble, ta, dt);
        let t_speed = self.sm.treble.clamp(0.0, 1.0);

        // mid / lowMid: slow symmetric — alignment / cohesion.
        ease(&mut self.sm.mid, bands.mid, m.mid_smoothing, dt);
        let t_align = self.sm.mid.clamp(0.0, 1.0);
        ease(&mut self.sm.low_mid, bands.low_mid, m.low_mid_smoothing, dt);
        let t_cohesion = self.sm.low_mid.clamp(0.0, 1.0);

        // air: medium symmetric — twinkle.
        ease(&mut self.sm.air, bands.air, m.air_smoothing, dt);
        let t_air = self.sm.air.clamp(0.0, 1.0);

        // upperMid: fast attack / slow release, then gate + perceptual curve.
        let uma = if bands.upper_mid > self.sm.upper_mid {
            m.upper_mid_attack
        } else {
            m.upper_mid_release
        };
        ease(&mut self.sm.upper_mid, bands.upper_mid, uma, dt);
        let um_gated =
            ((self.sm.upper_mid - m.upper_mid_gate) / (1.0 - m.upper_mid_gate).max(1e-4)).max(0.0);
        let t_upper_mid = um_gated.min(1.0).powf(m.upper_mid_curve);

        // bass onset envelope → colour punctuation (rise-time softened).
        ease(&mut self.sm.bass_onset_for_colour, envelopes.bass, m.onset_smoothing, dt);

        // === Driven values.
        let attract = m.attract_floor + (m.attract_ceiling - m.attract_floor) * t_attract;
        let separation =
            m.separation_floor + (m.separation_ceiling - m.separation_floor) * t_sep;
        let freedom = if m.freedom_enabled {
            let t_freedom = 1.0 - t_attract;
            m.freedom_floor + (m.freedom_ceiling - m.freedom_floor) * t_freedom
        } else {
            defaults.freedom
        };
        let shape_center_y = m.vertical_lift * t_sep;

        // maxSpeed ← treble (full span) + upperMid (partial span, additive).
        let speed_span = m.max_speed_ceiling - m.max_speed_floor;
        let speed_base = m.max_speed_floor + speed_span * t_speed;
        let speed_add = if m.upper_mid_speed_enabled {
            speed_span * m.upper_mid_speed_factor * t_upper_mid
        } else {
            0.0
        };

        let max_turn_rate = if m.turn_rate_enabled {
            m.turn_floor + (m.turn_ceiling - m.turn_floor) * t_upper_mid
        } else {
            1000.0
        };
        let shear = if m.shear_enabled {
            m.shear_floor + (m.shear_ceiling - m.shear_floor) * t_upper_mid
        } else {
            0.0
        };

        let alignment =
            m.alignment_floor + (m.alignment_ceiling - m.alignment_floor) * t_align;
        let cohesion = m.cohesion_floor + (m.cohesion_ceiling - m.cohesion_floor) * t_cohesion;

        let twinkle = if m.twinkle_enabled {
            let sign = if m.twinkle_invert { -1.0 } else { 1.0 };
            t_air * m.twinkle_strength * sign
        } else {
            0.0
        };

        let shock_strength = if m.shockwave_enabled {
            envelopes.bass
        } else {
            0.0
        };

        // Colour: dark-trough swell (sustained bass) + kick punctuation
        // (softened onset envelope) on top of the trough offset.
        let palette_t =
            m.t_offset + self.sm.bass * m.t_scale + self.sm.bass_onset_for_colour * m.onset_boost;

        Driven {
            attract,
            separation,
            alignment,
            cohesion,
            freedom,
            max_speed: speed_base + speed_add,
            max_turn_rate,
            shear,
            shape_center_y,
            shock_strength,
            palette_t,
            twinkle,
            connected: fresh,
        }
    }
}
