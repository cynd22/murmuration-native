//! Murmuration — Rust/wgpu port of the audio-reactive starling visualiser.
//!
//! Design pillars:
//!  - Fixed-timestep simulation (default 240 Hz) decoupled from render rate —
//!    the flock behaves identically on a 60 Hz and a 240 Hz display.
//!  - All forces/visuals live in shaders/*.wgsl, hot-reloaded on save —
//!    iteration never touches the Rust compiler.
//!
//! Flags: --birds N (default 10000), --uncapped (no vsync), --sim-hz N (default 240).

mod audio;
mod capture;
mod dsp;
mod fluid;
mod hot;
mod nowplaying;
mod params;
mod render;
mod sim;
mod ui;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct Args {
    birds: u32,
    uncapped: bool,
    sim_hz: f64,
    /// External feeder URL. None = native in-process capture (the default).
    ws_url: Option<String>,
    /// Capture device name substring (native mode). None = auto-pick monitor.
    device: Option<String>,
    /// Print input devices and exit.
    list_devices: bool,
    /// Benchmark: run uncapped, measure frame times for N seconds, print, exit.
    bench: Option<f64>,
    /// Force the window/render size (physical px) — e.g. for a 4K bench.
    win: Option<(u32, u32)>,
    /// Start with all sky effects off (isolate the fullscreen sky cost).
    no_sky: bool,
    /// Sky render-resolution divisor: 1=full, 2=half (default), 4=quarter
    /// (near-free sky for weak GPUs). Clamped to 1..=8.
    sky_div: u32,
}

fn parse_args() -> Args {
    let mut args = Args {
        birds: 10_000,
        uncapped: false,
        sim_hz: 240.0,
        ws_url: None,
        device: None,
        list_devices: false,
        bench: None,
        win: None,
        no_sky: false,
        sky_div: render::SKY_DIV_DEFAULT,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--birds" => {
                args.birds = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.birds)
            }
            "--uncapped" => args.uncapped = true,
            "--ws" => args.ws_url = it.next(),
            "--device" => args.device = it.next(),
            "--list-devices" => args.list_devices = true,
            "--no-sky" => args.no_sky = true,
            "--sky-div" => {
                args.sky_div = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.sky_div)
            }
            "--bench" => {
                args.bench = Some(it.next().and_then(|v| v.parse().ok()).unwrap_or(6.0))
            }
            "--win" => {
                let w = it.next().and_then(|v| v.parse().ok());
                let h = it.next().and_then(|v| v.parse().ok());
                if let (Some(w), Some(h)) = (w, h) {
                    args.win = Some((w, h));
                }
            }
            "--sim-hz" => {
                args.sim_hz = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(args.sim_hz)
            }
            other => log::warn!("unknown arg: {other}"),
        }
    }
    args
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    sim: sim::Sim,
    renderer: render::Renderer,
    sky_div: u32,
    fluid: fluid::Fluid,
    fluid_settings: fluid::FluidSettings,
    fluid_acc: f64,
    settings: params::Settings,
    audio: audio::AudioEngine,
    // Last tick's driven values — read at render time (most frames run zero
    // ticks at high fps, so this carries between them).
    last_driven: audio::Driven,
    ui: ui::Ui,
    ui_state: ui::UiState,
    sources: hot::Sources,
    fps_current: f32,
    watcher: hot::ShaderWatcher,
    shader_dir: std::path::PathBuf,

    tick_dt: f64,
    accumulator: f64,
    sim_time_s: f64,
    last_frame: Instant,

    fps_frames: u32,
    fps_since: Instant,

    // Benchmark mode: measure frame times for bench_secs (after a 1s warmup),
    // print a parseable line, exit.
    bench_secs: Option<f64>,
    bench_start: Option<Instant>,
    bench_dts: Vec<f32>,
}

impl State {
    fn new(event_loop: &ActiveEventLoop, args: &Args) -> State {
        let attrs = Window::default_attributes().with_title("murmuration");
        let attrs = match args.win {
            Some((w, h)) => attrs.with_inner_size(winit::dpi::PhysicalSize::new(w, h)),
            None => attrs.with_inner_size(winit::dpi::LogicalSize::new(1600, 900)),
        };
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no suitable GPU adapter");
        log::info!("adapter: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("murmuration device"),
            ..Default::default()
        }))
        .expect("request device");

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Non-sRGB format on purpose — see render.rs.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let present_mode = if args.uncapped || args.bench.is_some() {
            [wgpu::PresentMode::Mailbox, wgpu::PresentMode::Immediate]
                .into_iter()
                .find(|m| caps.present_modes.contains(m))
                .unwrap_or(wgpu::PresentMode::AutoNoVsync)
        } else {
            wgpu::PresentMode::Fifo
        };
        log::info!("surface format {format:?}, present mode {present_mode:?}");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        // Shader dir resolution. An installed copy keeps shaders/ next to the
        // binary, so it works launched from anywhere (app menu, any CWD); the
        // dev tree falls back to CWD-relative paths (cargo run from rust-port/).
        let mut shader_candidates: Vec<std::path::PathBuf> = Vec::new();
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                shader_candidates.push(dir.join("shaders"));
            }
        }
        shader_candidates.push(std::path::PathBuf::from("shaders"));
        shader_candidates.push(std::path::PathBuf::from("rust-port/shaders"));
        let shader_dir = shader_candidates
            .into_iter()
            .find(|p| p.join("common.wgsl").exists())
            .expect("shaders/ not found (looked next to the binary and in ./shaders)");

        let sources = hot::Sources::load(&shader_dir).expect("read shaders");
        let sim = sim::Sim::new(&device, args.birds, &sources)
            .unwrap_or_else(|e| panic!("initial sim shader compile failed:\n{e}"));
        let fluid = fluid::Fluid::new(&device, &sources)
            .unwrap_or_else(|e| panic!("initial fluid shader compile failed:\n{e}"));
        let renderer = render::Renderer::new(
            &device,
            format,
            config.width,
            config.height,
            &sim,
            &fluid,
            &sources,
            args.sky_div,
        )
        .unwrap_or_else(|e| panic!("initial render shader compile failed:\n{e}"));

        let watcher = hot::ShaderWatcher::new(shader_dir.clone()).expect("shader watcher");
        let ui = ui::Ui::new(&window, &device, format);

        log::info!(
            "{} birds, sim {} Hz fixed timestep",
            args.birds,
            args.sim_hz
        );

        let mut ui_state = ui::UiState::new(args.birds);
        if args.no_sky {
            ui_state.bg_intensity = 0.0;
            ui_state.aurora = 0.0;
            ui_state.fluid_mix = 0.0;
        }

        State {
            ui,
            ui_state,
            window,
            surface,
            device,
            queue,
            config,
            sim,
            renderer,
            sky_div: args.sky_div,
            fluid,
            fluid_settings: fluid::FluidSettings::default(),
            fluid_acc: 0.0,
            settings: params::Settings::default(),
            audio: match &args.ws_url {
                Some(url) => audio::AudioEngine::websocket(url.clone()),
                None => audio::AudioEngine::native(args.device.clone()),
            },
            last_driven: audio::Driven {
                palette_t: 0.11, // trough until audio arrives
                ..Default::default()
            },
            sources,
            fps_current: 0.0,
            watcher,
            shader_dir,
            tick_dt: 1.0 / args.sim_hz,
            accumulator: 0.0,
            sim_time_s: 0.0,
            last_frame: Instant::now(),
            fps_frames: 0,
            fps_since: Instant::now(),
            bench_secs: args.bench,
            bench_start: None,
            bench_dts: Vec::new(),
        }
    }

    fn print_bench_and_exit(&self) -> ! {
        let mut dts = self.bench_dts.clone();
        dts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = dts.len().max(1);
        let sum: f32 = dts.iter().sum();
        let avg_fps = n as f32 / sum.max(1e-6);
        // 1%-low = mean frametime of the slowest 1% of frames, as fps.
        let tail = (n / 100).max(1);
        let slow = dts.iter().rev().take(tail).sum::<f32>() / tail as f32;
        let low1_fps = 1.0 / slow.max(1e-6);
        let median_ms = dts[n / 2] * 1000.0;
        eprintln!(
            "BENCH birds={} res={}x{} frames={} avg_fps={:.1} 1%low_fps={:.1} median_ms={:.2}",
            self.sim.n, self.config.width, self.config.height, n, avg_fps, low1_fps, median_ms
        );
        std::process::exit(0);
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(&self.device, self.config.width, self.config.height);
    }

    fn hot_reload(&mut self) {
        if !self.watcher.take_changed() {
            return;
        }
        match hot::Sources::load(&self.shader_dir) {
            Ok(sources) => {
                match self.sim.rebuild(&self.device, &sources) {
                    Ok(()) => log::info!("sim shaders reloaded"),
                    Err(e) => log::error!("sim shader error (keeping old):\n{e}"),
                }
                match self.renderer.rebuild(&self.device, &sources) {
                    Ok(()) => log::info!("render shaders reloaded"),
                    Err(e) => log::error!("render shader error (keeping old):\n{e}"),
                }
                match self.fluid.rebuild(&self.device, &sources) {
                    Ok(()) => log::info!("fluid shaders reloaded"),
                    Err(e) => log::error!("fluid shader error (keeping old):\n{e}"),
                }
                self.sources = sources;
            }
            Err(e) => log::error!("shader read failed: {e}"),
        }
    }

    /// Rebuild sim + renderer for a new flock size (respawns the flock).
    fn apply_bird_count(&mut self) {
        let n = self.ui_state.pending_birds.max(64);
        match sim::Sim::new(&self.device, n, &self.sources) {
            Ok(new_sim) => match render::Renderer::new(
                &self.device,
                self.config.format,
                self.config.width,
                self.config.height,
                &new_sim,
                &self.fluid,
                &self.sources,
                self.sky_div,
            ) {
                Ok(new_renderer) => {
                    self.sim = new_sim;
                    self.renderer = new_renderer;
                    log::info!("flock resized to {n} birds");
                }
                Err(e) => log::error!("renderer rebuild failed: {e}"),
            },
            Err(e) => log::error!("sim rebuild failed: {e}"),
        }
    }

    fn frame(&mut self) {
        self.hot_reload();
        if self.ui_state.apply_birds {
            self.ui_state.apply_birds = false;
            self.apply_bird_count();
        }

        // === Fixed-timestep accumulator. Render rate floats with the display;
        // simulation always advances in exact tick_dt slices.
        let now = Instant::now();
        let raw_dt = (now - self.last_frame).as_secs_f64();
        let frame_dt = raw_dt.min(0.25);
        self.last_frame = now;

        // Benchmark: 1s warmup, then record frame times for bench_secs, print, exit.
        if let Some(bsecs) = self.bench_secs {
            let start = *self.bench_start.get_or_insert(now);
            let elapsed = (now - start).as_secs_f64();
            if elapsed > 1.0 {
                self.bench_dts.push(raw_dt as f32);
            }
            if elapsed > 1.0 + bsecs {
                self.print_bench_and_exit();
            }
        }

        self.accumulator += frame_dt;

        let mut steps = 0;
        while self.accumulator >= self.tick_dt && steps < 32 {
            // Audio envelopes advance at sim rate; driven values land in this
            // tick's params (motion) and are stashed for render (colour/sky).
            let d = self.audio.tick(self.tick_dt as f32, &self.settings);
            self.last_driven = d;

            let mut p = self.settings.sim_params(
                self.tick_dt as f32,
                (self.sim_time_s * 1000.0) as f32,
                self.sim.n,
                self.ui_state.cam.position().into(),
            );
            p.shape_center = [0.0, d.shape_center_y, 0.0, d.attract];
            p.separation_dist = d.separation;
            p.alignment_dist = d.alignment;
            p.cohesion_dist = d.cohesion;
            p.freedom = d.freedom;
            p.max_speed = d.max_speed;
            p.max_turn_rate = d.max_turn_rate;
            p.shear = d.shear;
            p.shock_strength = d.shock_strength;
            self.queue.write_buffer(&self.sim.params_buf, 0, bytemuck::bytes_of(&p));

            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.sim.step(&mut encoder);
            self.queue.submit(Some(encoder.finish()));

            self.sim_time_s += self.tick_dt;
            self.accumulator -= self.tick_dt;
            steps += 1;
        }
        if steps == 32 {
            // Severe stall — drop the backlog rather than spiral.
            self.accumulator = 0.0;
        }

        // === Render current state.
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Timeout => return,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => panic!("surface validation error"),
        };
        let view = frame.texture.create_view(&Default::default());

        let aspect = self.config.width as f32 / self.config.height as f32;
        let d = self.last_driven;
        let style = render::FrameStyle {
            cam: self.ui_state.cam,
            palette: render::PALETTE_PRESETS[self.ui_state.palette_idx].1,
            palette_t: d.palette_t,
            palette_intensity: self.ui_state.palette_intensity,
            twinkle: d.twinkle,
            time_ms: (self.sim_time_s * 1000.0) as f32,
            bands: d.bg_bands,
            beat: [
                d.beat_phase,
                d.beat_conf,
                d.bpm / 200.0,
                self.sim_time_s as f32,
            ],
            bg: self.ui_state.bg(),
            bg2: [
                self.ui_state.fluid_mix,
                self.ui_state.fluid_heat,
                self.ui_state.aurora,
                0.0,
            ],
        };
        self.renderer.update_uniforms(&self.queue, aspect, &style);

        let mut encoder = self.device.create_command_encoder(&Default::default());

        // Fluid sky: fixed 60 Hz on its own accumulator (drop backlog on stall).
        self.fluid_acc = (self.fluid_acc + frame_dt).min(0.1);
        if self.fluid_acc >= 1.0 / 60.0 && self.ui_state.fluid_mix > 0.001 {
            self.fluid_acc -= 1.0 / 60.0;
            self.fluid.step(
                &self.device,
                &self.queue,
                &mut encoder,
                1.0 / 60.0,
                &d,
                &self.fluid_settings,
                self.sim_time_s as f32,
            );
        }

        self.renderer.render(&mut encoder, &view, self.sim.flip);
        self.ui.run_and_paint(
            &self.device,
            &self.queue,
            &mut encoder,
            &self.window,
            &view,
            [self.config.width, self.config.height],
            &mut self.ui_state,
            &mut self.audio.mapping,
            &mut self.settings,
            &mut self.fluid_settings,
            &d,
            self.fps_current,
            self.sim.n,
            1.0 / self.tick_dt,
        );
        self.queue.submit(Some(encoder.finish()));
        frame.present();

        // === FPS in the title bar (the no-UI-chrome rule: nothing on screen).
        self.fps_frames += 1;
        let elapsed = self.fps_since.elapsed().as_secs_f64();
        if elapsed >= 0.5 {
            let fps = self.fps_frames as f64 / elapsed;
            self.fps_current = fps as f32;
            self.window.set_title(&format!(
                "murmuration — {} birds — {:.0} fps — sim {:.0} Hz — audio {}",
                self.sim.n,
                fps,
                1.0 / self.tick_dt,
                if self.last_driven.connected { "✓" } else { "✗" }
            ));
            self.fps_frames = 0;
            self.fps_since = Instant::now();
        }
    }
}

struct App {
    args: Args,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            self.state = Some(State::new(event_loop, &self.args));
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        let window = state.window.clone();
        let ui_consumed = state.ui.on_event(&window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref c),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !ui_consumed && c.eq_ignore_ascii_case("h") => {
                state.ui_state.visible = !state.ui_state.visible;
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref c),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !ui_consumed && c.eq_ignore_ascii_case("p") => {
                state.ui_state.now_playing_visible = !state.ui_state.now_playing_visible;
            }
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = parse_args();

    // --list-devices: print capture inputs (marking system-audio monitors) and exit.
    if args.list_devices {
        capture::list_devices();
        return;
    }

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App { args, state: None })
        .expect("event loop run");
}
