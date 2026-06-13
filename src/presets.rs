//! Named tuning presets, saved as TOML in the user's config dir.
//!
//! Presets live in `~/.config/murmuration/presets/<name>.toml` — deliberately
//! NOT next to the binary, so re-downloading the binary to update never touches
//! them. A preset is the whole look + tuning snapshot: flock forces, every audio
//! mapping, sky / fluid / aurora, banking + dark bands, camera, and bird count.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::audio::Mapping;
use crate::fluid::FluidSettings;
use crate::params::Settings;
use crate::render::CamSettings;
use crate::ui::UiState;

/// The panel knobs that live in UiState (not in Settings/Mapping/FluidSettings).
/// `cam` is declared LAST so TOML emits the scalar fields before the nested
/// `[ui.cam]` table (TOML requires values before sub-tables).
#[derive(Serialize, Deserialize)]
pub struct PresetUi {
    pub palette_idx: usize,
    pub palette_intensity: f32,
    pub bg_intensity: f32,
    pub bg_clouds: f32,
    pub bg_stars: f32,
    pub bg_beat_pulse: f32,
    pub fluid_mix: f32,
    pub fluid_heat: f32,
    pub aurora: f32,
    pub band_strength: f32,
    pub um_baseline_secs: f32,
    pub birds: u32,
    pub cam: CamSettings,
}

#[derive(Serialize, Deserialize)]
pub struct Preset {
    pub settings: Settings,
    pub mapping: Mapping,
    pub fluid: FluidSettings,
    pub ui: PresetUi,
}

/// `~/.config/murmuration/presets/`, created on demand. None if HOME is unset.
fn preset_dir() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    let dir = base.join("murmuration").join("presets");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

/// Filename-safe form of a preset name (the name IS the filename stem).
fn sanitize(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let s = s.trim().to_string();
    if s.is_empty() { "preset".to_string() } else { s }
}

/// Names of saved presets (sorted, case-insensitive), from the `.toml` files.
pub fn list() -> Vec<String> {
    let Some(dir) = preset_dir() else {
        return Vec::new();
    };
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("toml") {
                p.file_stem().and_then(|s| s.to_str()).map(str::to_string)
            } else {
                None
            }
        })
        .collect();
    names.sort_by_key(|s| s.to_lowercase());
    names
}

/// Write a preset; returns the saved (sanitized) name on success.
pub fn save(name: &str, preset: &Preset) -> Result<String, String> {
    let dir = preset_dir().ok_or("no config directory (HOME unset)")?;
    let name = sanitize(name);
    let text = toml::to_string_pretty(preset).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}.toml")), text).map_err(|e| e.to_string())?;
    Ok(name)
}

pub fn load(name: &str) -> Option<Preset> {
    let dir = preset_dir()?;
    let text = std::fs::read_to_string(dir.join(format!("{name}.toml"))).ok()?;
    toml::from_str(&text).ok()
}

/// Snapshot the live state into a preset.
pub fn capture(s: &Settings, m: &Mapping, fl: &FluidSettings, ui: &UiState) -> Preset {
    Preset {
        settings: s.clone(),
        mapping: m.clone(),
        fluid: fl.clone(),
        ui: PresetUi {
            palette_idx: ui.palette_idx,
            palette_intensity: ui.palette_intensity,
            bg_intensity: ui.bg_intensity,
            bg_clouds: ui.bg_clouds,
            bg_stars: ui.bg_stars,
            bg_beat_pulse: ui.bg_beat_pulse,
            fluid_mix: ui.fluid_mix,
            fluid_heat: ui.fluid_heat,
            aurora: ui.aurora,
            band_strength: ui.band_strength,
            um_baseline_secs: ui.um_baseline_secs,
            birds: ui.pending_birds,
            cam: ui.cam,
        },
    }
}

/// Apply a loaded preset to the live state. Sets pending_birds + apply_birds so
/// the next frame respawns the flock at the preset's count. palette_idx is
/// clamped so an old/hand-edited preset can't index past the palette list.
pub fn apply(p: Preset, s: &mut Settings, m: &mut Mapping, fl: &mut FluidSettings, ui: &mut UiState) {
    *s = p.settings;
    *m = p.mapping;
    *fl = p.fluid;
    ui.palette_idx = p
        .ui
        .palette_idx
        .min(crate::render::PALETTE_PRESETS.len().saturating_sub(1));
    ui.palette_intensity = p.ui.palette_intensity;
    ui.bg_intensity = p.ui.bg_intensity;
    ui.bg_clouds = p.ui.bg_clouds;
    ui.bg_stars = p.ui.bg_stars;
    ui.bg_beat_pulse = p.ui.bg_beat_pulse;
    ui.fluid_mix = p.ui.fluid_mix;
    ui.fluid_heat = p.ui.fluid_heat;
    ui.aurora = p.ui.aurora;
    ui.band_strength = p.ui.band_strength;
    ui.um_baseline_secs = p.ui.um_baseline_secs;
    ui.cam = p.ui.cam;
    ui.pending_birds = p.ui.birds;
    ui.apply_birds = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_roundtrip() {
        let p = Preset {
            settings: Settings::default(),
            mapping: Mapping::default(),
            fluid: FluidSettings::default(),
            ui: PresetUi {
                palette_idx: 2,
                palette_intensity: 0.65,
                bg_intensity: 1.0,
                bg_clouds: 0.7,
                bg_stars: 0.7,
                bg_beat_pulse: 0.6,
                fluid_mix: 0.85,
                fluid_heat: 1.0,
                aurora: 0.7,
                band_strength: 0.6,
                um_baseline_secs: 6.0,
                birds: 50_000,
                cam: CamSettings::default(),
            },
        };
        // Serialize must succeed (catches TOML's values-before-tables ordering).
        let text = toml::to_string_pretty(&p).expect("serialize");
        let back: Preset = toml::from_str(&text).expect("deserialize");
        assert_eq!(back.ui.birds, 50_000);
        assert_eq!(back.ui.palette_idx, 2);
        assert!((back.settings.separation - p.settings.separation).abs() < 1e-6);
        assert!((back.mapping.t_offset - p.mapping.t_offset).abs() < 1e-6);
        assert!((back.fluid.vorticity - p.fluid.vorticity).abs() < 1e-6);
        assert!((back.ui.cam.fov_deg - p.ui.cam.fov_deg).abs() < 1e-6);
    }
}
