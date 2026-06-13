//! GPU fluid sky — the Rust side of shaders/fluid.wgsl.
//!
//! A small (384x192) stable-fluids field whose ONLY energy input is the
//! music: subBass onsets fire rising plumes from the bottom edge, treble
//! onsets kick small vortex pairs, mid is lateral wind, and a confident beat
//! makes the plume breathe in phase. The dye texture (density + heat) is
//! sampled by the sky shader and lit with the bird palette.
//!
//! Runs at a fixed 60 Hz on its own accumulator — visual element, not
//! physics-critical, and 60 Hz keeps the per-frame cost negligible
//! (~30 tiny dispatches).

use crate::audio::Driven;
use crate::hot::Sources;

pub const FLUID_W: u32 = 384;
pub const FLUID_H: u32 = 192;
const JACOBI_ITERS: usize = 20;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidParams {
    dt: f32,
    vorticity: f32,
    vel_dissipation: f32,
    dye_dissipation: f32,
    wind: [f32; 2],
    texel: [f32; 2],
    splats: [[f32; 4]; 8],
}

/// UI-tunable fluid behaviour.
pub struct FluidSettings {
    pub vorticity: f32,
    pub dye_keep: f32,   // per-step dye keep factor (closer to 1 = longer-lived clouds)
    pub plume: f32,      // master strength of audio injection
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self {
            vorticity: 18.0,
            dye_keep: 0.994,
            plume: 1.0,
        }
    }
}

struct Pipelines {
    splat_vel: wgpu::ComputePipeline,
    splat_dye: wgpu::ComputePipeline,
    curl: wgpu::ComputePipeline,
    vorticity: wgpu::ComputePipeline,
    divergence: wgpu::ComputePipeline,
    jacobi: wgpu::ComputePipeline,
    subtract: wgpu::ComputePipeline,
    advect_vel: wgpu::ComputePipeline,
    advect_dye: wgpu::ComputePipeline,
}

pub struct Fluid {
    params_buf: wgpu::Buffer,
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    pipelines: Pipelines,
    sampler: wgpu::Sampler,
    vel: [wgpu::Texture; 2],
    dye: [wgpu::Texture; 2],
    pressure: [wgpu::Texture; 2],
    curl_tex: wgpu::Texture,
    div_tex: wgpu::Texture,
    /// Stable view for the sky shader — dye is copied here after each step so
    /// the render bind group never has to chase the ping-pong.
    pub display: wgpu::Texture,
    pub display_view: wgpu::TextureView,
    pub sampler_render: wgpu::Sampler,
    vel_i: usize,
    dye_i: usize,
    pr_i: usize,
    // Cheap deterministic rng for splat placement (no Date/rand dependency).
    rng: u64,
    prev_beat_phase: f32,
}

fn mk_tex(device: &wgpu::Device, label: &str, extra: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: FLUID_W,
            height: FLUID_H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | extra,
        view_formats: &[],
    })
}

impl Fluid {
    pub fn new(device: &wgpu::Device, sources: &Sources) -> Result<Self, wgpu::Error> {
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fluid params"),
            size: std::mem::size_of::<FluidParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluid bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fluid layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fluid sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sampler_render = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fluid render sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let vel = [mk_tex(device, "vel A", wgpu::TextureUsages::empty()),
                   mk_tex(device, "vel B", wgpu::TextureUsages::empty())];
        let dye = [mk_tex(device, "dye A", wgpu::TextureUsages::COPY_SRC),
                   mk_tex(device, "dye B", wgpu::TextureUsages::COPY_SRC)];
        let pressure = [mk_tex(device, "pressure A", wgpu::TextureUsages::empty()),
                        mk_tex(device, "pressure B", wgpu::TextureUsages::empty())];
        let curl_tex = mk_tex(device, "curl", wgpu::TextureUsages::empty());
        let div_tex = mk_tex(device, "divergence", wgpu::TextureUsages::empty());
        let display = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fluid display"),
            size: wgpu::Extent3d {
                width: FLUID_W,
                height: FLUID_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let display_view = display.create_view(&Default::default());

        let pipelines = build_pipelines(device, &layout, sources)?;

        Ok(Self {
            params_buf,
            bgl,
            layout,
            pipelines,
            sampler,
            vel,
            dye,
            pressure,
            curl_tex,
            div_tex,
            display,
            display_view,
            sampler_render,
            vel_i: 0,
            dye_i: 0,
            pr_i: 0,
            rng: 0xC10D_5EED,
            prev_beat_phase: 0.0,
        })
    }

    pub fn rebuild(&mut self, device: &wgpu::Device, sources: &Sources) -> Result<(), wgpu::Error> {
        self.pipelines = build_pipelines(device, &self.layout, sources)?;
        Ok(())
    }

    fn rand(&mut self) -> f32 {
        self.rng = self
            .rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.rng >> 40) & 0xFFFFFF) as f32 / 16_777_216.0
    }

    /// Compose this step's audio forcing and upload params.
    fn write_params(&mut self, queue: &wgpu::Queue, dt: f32, d: &Driven, s: &FluidSettings, time_s: f32) {
        let mut splats = [[0.0f32; 4]; 8];

        // Beat-edge detection: phase wrapped since last step + confident
        // tracker = give this plume a kick multiplier.
        let beat_kick = if d.beat_conf > 0.18 && d.beat_phase < self.prev_beat_phase {
            1.0 + 1.5 * d.beat_conf
        } else {
            1.0
        };
        self.prev_beat_phase = d.beat_phase;

        // Splat 0: the main thermal plume. Wanders slowly along the bottom;
        // strength rides subBass level + subBass onset envelope. When the
        // beat tracker is locked, the plume PUMPS in beat phase — sharp
        // injection right at each beat instant, near-silent between — so the
        // sky breathes the tempo even through sustained bass. Texture-space
        // up is -y.
        let sub = d.bg_bands[0];
        let beat_pump = if d.beat_conf > 0.15 {
            let ph = (d.beat_phase * std::f32::consts::TAU).cos() * 0.5 + 0.5;
            0.4 + 1.6 * ph.powi(4) * d.beat_conf.min(1.0)
        } else {
            1.0
        };
        let plume_power =
            (0.25 * sub + 1.2 * d.onset_sub + 0.5 * d.onset_bass) * s.plume * beat_kick * beat_pump;
        if plume_power > 0.01 {
            let x = 0.5 + 0.30 * (time_s * 0.11).sin();
            splats[0] = [x, 0.96, 0.10, 0.0];
            splats[1] = [0.0, -1.6 * plume_power, 0.9 * plume_power, 1.2 * plume_power];
        }

        // Splats 1-2: treble-onset vortex kicks — small, randomly placed in
        // the mid-sky, opposing horizontal forces so they shear the field.
        if d.onset_treble > 0.05 {
            let strength = 0.5 * d.onset_treble * s.plume;
            let x = 0.15 + 0.7 * self.rand();
            let y = 0.25 + 0.45 * self.rand();
            let dir = if self.rand() > 0.5 { 1.0 } else { -1.0 };
            splats[2] = [x, y, 0.045, 0.0];
            splats[3] = [dir * 0.9 * strength, -0.15 * strength, 0.08 * strength, 0.3 * strength];
            splats[4] = [x + dir * 0.06, (y + 0.05).min(0.9), 0.045, 0.0];
            splats[5] = [-dir * 0.9 * strength, 0.05 * strength, 0.0, 0.0];
        }

        // Splat 3: a faint broad haze refresh driven by bass — keeps SOME
        // material in the sky during sustained music even between kicks.
        let bass = d.bg_bands[1];
        if bass > 0.05 {
            splats[6] = [0.5, 0.75, 0.45, 0.0];
            splats[7] = [0.0, -0.04 * bass * s.plume, 0.05 * bass * s.plume, 0.0];
        }

        let p = FluidParams {
            dt,
            vorticity: s.vorticity,
            vel_dissipation: 0.985,
            dye_dissipation: s.dye_keep,
            wind: [0.015 * d.bg_bands[2], 0.0], // mid → lateral drift
            texel: [1.0 / FLUID_W as f32, 1.0 / FLUID_H as f32],
            splats,
        };
        queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }

    fn bind(&self, device: &wgpu::Device, a: &wgpu::Texture, b: &wgpu::Texture, out: &wgpu::Texture) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluid pass"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&a.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&b.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&out.create_view(&Default::default())) },
            ],
        })
    }

    /// Encode one 60Hz fluid step. write_params must have run first.
    pub fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
        d: &Driven,
        s: &FluidSettings,
        time_s: f32,
    ) {
        self.write_params(queue, dt, d, s, time_s);
        let wg = (FLUID_W.div_ceil(8), FLUID_H.div_ceil(8));

        // Pass order: splat → curl/vorticity → project → advect → display.
        let mut runs: Vec<(&wgpu::ComputePipeline, wgpu::BindGroup)> = Vec::with_capacity(9 + JACOBI_ITERS);

        let v = self.vel_i;
        runs.push((&self.pipelines.splat_vel, self.bind(device, &self.vel[v], &self.vel[v], &self.vel[v ^ 1])));
        let v = v ^ 1;
        let dy = self.dye_i;
        runs.push((&self.pipelines.splat_dye, self.bind(device, &self.dye[dy], &self.dye[dy], &self.dye[dy ^ 1])));
        let dy = dy ^ 1;

        runs.push((&self.pipelines.curl, self.bind(device, &self.vel[v], &self.vel[v], &self.curl_tex)));
        runs.push((&self.pipelines.vorticity, self.bind(device, &self.vel[v], &self.curl_tex, &self.vel[v ^ 1])));
        let v = v ^ 1;

        runs.push((&self.pipelines.divergence, self.bind(device, &self.vel[v], &self.vel[v], &self.div_tex)));
        let mut pr = self.pr_i;
        for _ in 0..JACOBI_ITERS {
            runs.push((&self.pipelines.jacobi, self.bind(device, &self.pressure[pr], &self.div_tex, &self.pressure[pr ^ 1])));
            pr ^= 1;
        }
        runs.push((&self.pipelines.subtract, self.bind(device, &self.vel[v], &self.pressure[pr], &self.vel[v ^ 1])));
        let v = v ^ 1;

        runs.push((&self.pipelines.advect_vel, self.bind(device, &self.vel[v], &self.vel[v], &self.vel[v ^ 1])));
        let v = v ^ 1;
        runs.push((&self.pipelines.advect_dye, self.bind(device, &self.vel[v], &self.dye[dy], &self.dye[dy ^ 1])));
        let dy = dy ^ 1;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            for (pipeline, bg) in &runs {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg.0, wg.1, 1);
            }
        }

        encoder.copy_texture_to_texture(
            self.dye[dy].as_image_copy(),
            self.display.as_image_copy(),
            wgpu::Extent3d {
                width: FLUID_W,
                height: FLUID_H,
                depth_or_array_layers: 1,
            },
        );

        self.vel_i = v;
        self.dye_i = dy;
        self.pr_i = pr;
    }
}

fn build_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    sources: &Sources,
) -> Result<Pipelines, wgpu::Error> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fluid.wgsl"),
        source: wgpu::ShaderSource::Wgsl(sources.fluid.as_str().into()),
    });
    let mk = |label: &str, entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines = Pipelines {
        splat_vel: mk("fluid splat vel", "splat_vel"),
        splat_dye: mk("fluid splat dye", "splat_dye"),
        curl: mk("fluid curl", "curl"),
        vorticity: mk("fluid vorticity", "vorticity"),
        divergence: mk("fluid divergence", "divergence"),
        jacobi: mk("fluid jacobi", "jacobi"),
        subtract: mk("fluid subtract", "subtract_gradient"),
        advect_vel: mk("fluid advect vel", "advect_vel"),
        advect_dye: mk("fluid advect dye", "advect_dye"),
    };
    match pollster::block_on(scope.pop()) {
        Some(err) => Err(err),
        None => Ok(pipelines),
    }
}
