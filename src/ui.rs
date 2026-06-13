//! egui control panel. `H` toggles all UI chrome (the no-chrome-during-
//! playback rule); everything else is a translucent floating window styled to
//! sit quietly over the dusk sky.

use crate::audio::{Driven, Mapping};
use crate::params::Settings;
use crate::render::{CamSettings, PALETTE_PRESETS};
use winit::window::Window;

pub struct UiState {
    pub visible: bool,
    pub cam: CamSettings,
    pub palette_idx: usize,
    pub palette_intensity: f32,
    pub bg_intensity: f32,
    pub bg_clouds: f32,
    pub bg_stars: f32,
    pub bg_beat_pulse: f32,
    pub pending_birds: u32,
    pub apply_birds: bool,
    pub um_baseline_secs: f32, // UI-friendly form of Mapping::um_baseline_alpha
}

impl UiState {
    pub fn new(birds: u32) -> Self {
        Self {
            visible: true,
            cam: CamSettings::default(),
            palette_idx: 0, // ocean
            palette_intensity: 0.65,
            bg_intensity: 1.0,
            bg_clouds: 0.7,
            bg_stars: 0.7,
            bg_beat_pulse: 0.6,
            pending_birds: birds,
            apply_birds: false,
            um_baseline_secs: 6.0,
        }
    }

    pub fn bg(&self) -> [f32; 4] {
        [
            self.bg_intensity,
            self.bg_clouds,
            self.bg_stars,
            self.bg_beat_pulse,
        ]
    }
}

pub struct Ui {
    pub ctx: egui::Context,
    winit_state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
}

impl Ui {
    pub fn new(window: &Window, device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();

        // Quiet, modern dark style: translucent near-black panels, soft
        // corners, a desaturated ice-blue accent that matches the palette.
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(8, 12, 22, 235);
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(8, 12, 22, 235);
        visuals.window_corner_radius = egui::CornerRadius::same(10);
        visuals.window_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(90, 130, 180, 60));
        visuals.selection.bg_fill = egui::Color32::from_rgb(64, 110, 160);
        ctx.set_visuals(visuals);

        let winit_state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );
        let renderer = egui_wgpu::Renderer::new(
            device,
            format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                depth_stencil_format: None,
                dithering: false,
                ..Default::default()
            },
        );

        Self {
            ctx,
            winit_state,
            renderer,
        }
    }

    pub fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.winit_state.on_window_event(window, event).consumed
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run_and_paint(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        window: &Window,
        target: &wgpu::TextureView,
        size: [u32; 2],
        state: &mut UiState,
        mapping: &mut Mapping,
        settings: &mut Settings,
        diag: &Driven,
        fps: f32,
        birds: u32,
        sim_hz: f64,
    ) {
        let raw = self.winit_state.take_egui_input(window);
        let visible = state.visible;
        let out = self.ctx.run_ui(raw, |ctx| {
            if visible {
                panel(ctx, state, mapping, settings, diag, fps, birds, sim_hz);
            }
        });
        self.winit_state
            .handle_platform_output(window, out.platform_output);

        let prims = self.ctx.tessellate(out.shapes, out.pixels_per_point);
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: size,
            pixels_per_point: out.pixels_per_point,
        };
        for (id, delta) in &out.textures_delta.set {
            self.renderer.update_texture(device, queue, *id, delta);
        }
        self.renderer.update_buffers(device, queue, encoder, &prims, &sd);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            let mut pass = pass.forget_lifetime();
            self.renderer.render(&mut pass, &prims, &sd);
        }
        for id in &out.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}

fn slider(ui: &mut egui::Ui, v: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str) {
    ui.add(egui::Slider::new(v, range).text(label));
}

#[allow(clippy::too_many_arguments)]
fn panel(
    ctx: &egui::Context,
    state: &mut UiState,
    m: &mut Mapping,
    s: &mut Settings,
    d: &Driven,
    fps: f32,
    birds: u32,
    sim_hz: f64,
) {
    egui::Window::new("murmuration")
        .default_width(330.0)
        .resizable(true)
        .show(ctx, |ui| {
            // === Status strip.
            ui.horizontal(|ui| {
                let (dot, txt) = if d.connected {
                    (egui::Color32::from_rgb(80, 220, 130), "audio")
                } else {
                    (egui::Color32::from_rgb(220, 80, 80), "no audio")
                };
                ui.colored_label(dot, "●");
                ui.label(txt);
                ui.separator();
                ui.label(format!("{fps:.0} fps"));
                ui.separator();
                ui.label(format!("{birds} birds @ {sim_hz:.0} Hz"));
                if d.beat_conf > 0.12 {
                    ui.separator();
                    ui.label(format!("{:.0} bpm ({:.0}%)", d.bpm, d.beat_conf * 100.0));
                }
            });
            ui.add_space(4.0);
            ui.small("press H to hide everything");

            // === Performance.
            egui::CollapsingHeader::new("flock size").show(ui, |ui| {
                ui.add(
                    egui::Slider::new(&mut state.pending_birds, 1_000..=150_000)
                        .logarithmic(true)
                        .text("birds"),
                );
                if ui.button("apply (respawns flock)").clicked() {
                    state.apply_birds = true;
                }
            });

            // === Solo lab — the upperMid fix, with live signal readout.
            egui::CollapsingHeader::new("solo response (upperMid)")
                .default_open(true)
                .show(ui, |ui| {
                    ui.checkbox(&mut m.upper_mid_relative, "relative (contrast above song baseline)");
                    if m.upper_mid_relative {
                        slider(ui, &mut state.um_baseline_secs, 2.0..=15.0, "baseline memory (s)");
                        m.um_baseline_alpha = 1.0 - (-(1.0 / (60.0 * state.um_baseline_secs))).exp();
                        slider(ui, &mut m.um_contrast_gain, 1.0..=8.0, "contrast gain");
                        slider(ui, &mut m.um_contrast_deadzone, 0.0..=0.15, "deadzone");
                    } else {
                        slider(ui, &mut m.upper_mid_gate, 0.0..=0.5, "gate (absolute mode)");
                    }
                    slider(ui, &mut m.upper_mid_curve, 0.5..=3.0, "perceptual curve");
                    slider(ui, &mut m.turn_floor, 1.0..=20.0, "turn rate floor (rad/s)");
                    slider(ui, &mut m.turn_ceiling, 5.0..=40.0, "turn rate ceiling");
                    ui.checkbox(&mut m.shear_enabled, "shear (filament stretch)");
                    slider(ui, &mut m.shear_ceiling, 0.0..=3.0, "shear ceiling");
                    slider(ui, &mut m.upper_mid_speed_factor, 0.0..=1.5, "speed contribution");
                    ui.add_space(4.0);
                    ui.label("live signal:");
                    ui.add(egui::ProgressBar::new(d.um_raw).text(format!("envelope {:.2}", d.um_raw)));
                    ui.add(egui::ProgressBar::new(d.um_baseline).text(format!("baseline {:.2}", d.um_baseline)));
                    ui.add(egui::ProgressBar::new(d.um_t).text(format!("SOLO DRIVE {:.2}", d.um_t)));
                });

            // === Audio mappings (floors/ceilings for the other bands).
            egui::CollapsingHeader::new("audio mappings").show(ui, |ui| {
                ui.checkbox(&mut m.enabled, "audio drives the flock");
                slider(ui, &mut m.attract_floor, 0.0..=3.0, "attract floor (quiet)");
                slider(ui, &mut m.attract_ceiling, 1.0..=6.0, "attract ceiling (sub hits)");
                slider(ui, &mut m.separation_floor, 5.0..=30.0, "separation floor");
                slider(ui, &mut m.separation_ceiling, 10.0..=40.0, "separation ceiling");
                slider(ui, &mut m.max_speed_floor, 3.0..=15.0, "speed floor");
                slider(ui, &mut m.max_speed_ceiling, 8.0..=30.0, "speed ceiling (treble)");
                slider(ui, &mut m.alignment_floor, 10.0..=50.0, "alignment floor");
                slider(ui, &mut m.alignment_ceiling, 20.0..=70.0, "alignment ceiling (mid)");
                slider(ui, &mut m.cohesion_floor, 10.0..=70.0, "cohesion floor");
                slider(ui, &mut m.cohesion_ceiling, 30.0..=120.0, "cohesion ceiling (lowMid)");
                slider(ui, &mut m.freedom_floor, 0.0..=1.0, "freedom @ sub=1 (unified)");
                slider(ui, &mut m.freedom_ceiling, 0.5..=2.5, "freedom @ sub=0 (restless)");
                slider(ui, &mut m.vertical_lift, 0.0..=300.0, "vertical lift (sustained sub)");
                ui.checkbox(&mut m.shockwave_enabled, "shockwave (bass onsets → ground push)");
                ui.checkbox(&mut m.twinkle_enabled, "air twinkle");
                if m.twinkle_enabled {
                    ui.checkbox(&mut m.twinkle_invert, "twinkle darkens (preserves trough)");
                    slider(ui, &mut m.twinkle_strength, 0.0..=0.6, "twinkle strength");
                }
            });

            // === Colour.
            egui::CollapsingHeader::new("colour").show(ui, |ui| {
                egui::ComboBox::from_label("palette")
                    .selected_text(PALETTE_PRESETS[state.palette_idx].0)
                    .show_ui(ui, |ui| {
                        for (i, (name, _)) in PALETTE_PRESETS.iter().enumerate() {
                            ui.selectable_value(&mut state.palette_idx, i, *name);
                        }
                    });
                slider(ui, &mut m.t_offset, 0.0..=1.0, "t offset (trough depth)");
                slider(ui, &mut m.t_scale, 0.0..=2.0, "t scale (bass swell)");
                slider(ui, &mut m.onset_boost, 0.0..=0.5, "kick punctuation");
                slider(ui, &mut m.onset_smoothing, 0.01..=0.5, "kick rise-time");
                slider(ui, &mut state.palette_intensity, 0.0..=2.5, "intensity");
            });

            // === Background.
            egui::CollapsingHeader::new("sky").show(ui, |ui| {
                slider(ui, &mut state.bg_intensity, 0.0..=1.5, "sky effects (0 = classic)");
                slider(ui, &mut state.bg_clouds, 0.0..=1.5, "clouds");
                slider(ui, &mut state.bg_stars, 0.0..=1.5, "stars");
                slider(ui, &mut state.bg_beat_pulse, 0.0..=1.5, "beat pulse (needs BPM lock)");
            });

            // === Camera.
            egui::CollapsingHeader::new("camera").show(ui, |ui| {
                slider(ui, &mut state.cam.pov_height, -400.0..=300.0, "height");
                slider(ui, &mut state.cam.distance, 1200.0..=5000.0, "distance");
                slider(ui, &mut state.cam.fov_deg, 10.0..=60.0, "fov");
            });

            // === Static fallbacks (only matter with audio mappings off).
            egui::CollapsingHeader::new("static flock (no-audio fallbacks)").show(ui, |ui| {
                slider(ui, &mut s.separation, 1.0..=60.0, "separation");
                slider(ui, &mut s.alignment, 1.0..=80.0, "alignment");
                slider(ui, &mut s.cohesion, 1.0..=120.0, "cohesion");
                slider(ui, &mut s.freedom, 0.0..=2.5, "freedom");
                slider(ui, &mut s.attraction_strength, 0.0..=6.0, "attraction");
                slider(ui, &mut s.max_speed, 3.0..=30.0, "max speed");
            });
        });
}
