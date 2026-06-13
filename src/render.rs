//! Bird + ground rendering and the camera.
//!
//! The surface uses a NON-sRGB format deliberately: the HTML build's
//! ShaderMaterial wrote raw colour values into an sRGB-interpreted canvas, so
//! writing the same raw values to a non-sRGB swapchain reproduces its exact
//! look (including the dark-trough behaviour at low palette values).

use crate::hot::Sources;
use glam::{Mat4, Vec3};

// Sky 0x0a1428, raw (non-sRGB surface).
pub const SKY: wgpu::Color = wgpu::Color {
    r: 0.03922,
    g: 0.07843,
    b: 0.15686,
    a: 1.0,
};

/// Camera — defaults match cameraSettings in the HTML build; UI-adjustable.
#[derive(Clone, Copy)]
pub struct CamSettings {
    pub pov_height: f32,
    pub distance: f32,
    pub fov_deg: f32,
}

impl Default for CamSettings {
    fn default() -> Self {
        Self {
            pov_height: -300.0,
            distance: 3500.0,
            fov_deg: 18.0,
        }
    }
}

impl CamSettings {
    pub fn position(&self) -> Vec3 {
        Vec3::new(0.0, self.pov_height, self.distance)
    }
}

pub const CAM_LOOK_AT: Vec3 = Vec3::new(0.0, 250.0, 0.0);

/// iq cosine palette presets — same table as the HTML build. `ocean` first:
/// it's the load-bearing default (dark trough at low t).
pub const PALETTE_PRESETS: [(&str, [[f32; 4]; 4]); 5] = [
    ("ocean", [
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5, 0.0],
        [0.0, 0.5, 0.333, 0.0],
        [0.0, 0.5, 0.667, 0.0],
    ]),
    ("sunset", [
        [0.5, 0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5, 0.0],
        [1.0, 0.7, 0.4, 0.0],
        [0.0, 0.15, 0.20, 0.0],
    ]),
    ("fire", [
        [0.5, 0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5, 0.0],
        [1.0, 1.0, 0.5, 0.0],
        [0.8, 0.9, 0.30, 0.0],
    ]),
    ("rose", [
        [0.8, 0.5, 0.4, 0.0],
        [0.2, 0.4, 0.2, 0.0],
        [2.0, 1.0, 1.0, 0.0],
        [0.0, 0.25, 0.25, 0.0],
    ]),
    ("spectrum", [
        [0.5, 0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 0.33, 0.67, 0.0],
    ]),
];

/// Everything the renderer needs for one frame's uniforms.
pub struct FrameStyle {
    pub cam: CamSettings,
    pub palette: [[f32; 4]; 4],
    pub palette_t: f32,
    pub palette_intensity: f32,
    pub twinkle: f32,
    pub time_ms: f32,
    pub bands: [f32; 4], // smoothed subBass, bass, mid, air
    pub beat: [f32; 4],  // phase, confidence, bpm/200, time seconds
    pub bg: [f32; 4],    // intensity, clouds, stars, beat pulse
    pub bg2: [f32; 4],   // fluid mix, fluid heat glow, unused, unused
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    palette_a: [f32; 4],
    palette_b: [f32; 4],
    palette_c: [f32; 4],
    palette_d: [f32; 4],
    palette_t: f32,
    palette_intensity: f32,
    palette_enabled: f32,
    time: f32,
    num_birds: f32,
    twinkle_amount: f32,
    _pad0: f32,
    _pad1: f32,
    bands: [f32; 4],
    beat: [f32; 4],
    bg: [f32; 4],
    bg2: [f32; 4],
}

pub struct Renderer {
    uniforms_buf: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    layout: wgpu::PipelineLayout,
    bird_pipeline: wgpu::RenderPipeline,
    ground_pipeline: wgpu::RenderPipeline,
    background_pipeline: wgpu::RenderPipeline,
    // Half-res sky: the fullscreen sky pass renders here, then a cheap blit
    // upscales it before ground+birds (full res). ~4x cheaper sky at 4K.
    #[allow(dead_code)] // held only to keep the texture alive behind sky_view
    sky_tex: wgpu::Texture,
    sky_view: wgpu::TextureView,
    sky_sampler: wgpu::Sampler,
    blit_bgl: wgpu::BindGroupLayout,
    blit_bg: wgpu::BindGroup,
    blit_pipeline: wgpu::RenderPipeline,
    sky_div: u32,
    depth: wgpu::TextureView,
    format: wgpu::TextureFormat,
    n: u32,
}

impl Renderer {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sim: &crate::sim::Sim,
        fluid: &crate::fluid::Fluid,
        sources: &Sources,
        sky_div: u32,
    ) -> Result<Self, wgpu::Error> {
        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render uniforms"),
            size: std::mem::size_of::<RenderUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let mk_bg = |cur: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("render bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniforms_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: sim.pos[cur].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sim.vel[cur].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&fluid.display_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&fluid.sampler_render),
                    },
                ],
            })
        };
        let bind_groups = [mk_bg(0), mk_bg(1)];

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });

        let (bird_pipeline, ground_pipeline, background_pipeline) =
            build_pipelines(device, &layout, format, sources)?;

        let (sky_tex, sky_view) = create_sky_target(device, format, width, height, sky_div);
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky upscale sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky blit bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_bg = make_blit_bg(device, &blit_bgl, &sky_view, &sky_sampler);
        let blit_pipeline = build_blit_pipeline(device, &blit_bgl, format)?;

        Ok(Self {
            uniforms_buf,
            bind_groups,
            layout,
            bird_pipeline,
            ground_pipeline,
            background_pipeline,
            sky_tex,
            sky_view,
            sky_sampler,
            blit_bgl,
            blit_bg,
            blit_pipeline,
            sky_div,
            depth: create_depth(device, width, height),
            format,
            n: sim.n,
        })
    }

    pub fn rebuild(&mut self, device: &wgpu::Device, sources: &Sources) -> Result<(), wgpu::Error> {
        let (bird, ground, background) =
            build_pipelines(device, &self.layout, self.format, sources)?;
        self.bird_pipeline = bird;
        self.ground_pipeline = ground;
        self.background_pipeline = background;
        Ok(())
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth = create_depth(device, width, height);
        let (sky_tex, sky_view) = create_sky_target(device, self.format, width, height, self.sky_div);
        self.sky_tex = sky_tex;
        self.sky_view = sky_view;
        self.blit_bg = make_blit_bg(device, &self.blit_bgl, &self.sky_view, &self.sky_sampler);
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue, aspect: f32, style: &FrameStyle) {
        let view = Mat4::look_at_rh(style.cam.position(), CAM_LOOK_AT, Vec3::Y);
        let proj = Mat4::perspective_rh(style.cam.fov_deg.to_radians(), aspect, 1.0, 5000.0);
        let view_proj = proj * view;
        let u = RenderUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            palette_a: style.palette[0],
            palette_b: style.palette[1],
            palette_c: style.palette[2],
            palette_d: style.palette[3],
            palette_t: style.palette_t,
            palette_intensity: style.palette_intensity,
            palette_enabled: 1.0,
            time: style.time_ms,
            num_birds: self.n as f32,
            twinkle_amount: style.twinkle,
            _pad0: 0.0,
            _pad1: 0.0,
            bands: style.bands,
            beat: style.beat,
            bg: style.bg,
            bg2: style.bg2,
        };
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, flip: usize) {
        // Pass A: render the (expensive, low-frequency) sky into the half-res
        // offscreen target. No depth needed — the fullscreen triangle covers it.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sky (half-res)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.sky_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SKY),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_bind_group(0, &self.bind_groups[flip], &[]);
            pass.set_pipeline(&self.background_pipeline);
            pass.draw(0..3, 0..1);
        }

        // Pass B: full-res scene — upscale the sky, then ground + birds on top.
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(SKY),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_bind_group(0, &self.blit_bg, &[]);
            pass.set_pipeline(&self.blit_pipeline);
            pass.draw(0..3, 0..1);

            pass.set_bind_group(0, &self.bind_groups[flip], &[]);
            pass.set_pipeline(&self.ground_pipeline);
            pass.draw(0..6, 0..1);
            pass.set_pipeline(&self.bird_pipeline);
            pass.draw(0..self.n * 9, 0..1);
        }
    }
}

fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn build_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    sources: &Sources,
) -> Result<(wgpu::RenderPipeline, wgpu::RenderPipeline, wgpu::RenderPipeline), wgpu::Error> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    // bird.wgsl declares RenderUniforms + palette(); background.wgsl rides in
    // the same module so they stay in sync automatically.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bird.wgsl + background.wgsl"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{}", sources.bird, sources.background).into(),
        ),
    });

    let depth_state = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: Default::default(),
        bias: Default::default(),
    };
    // Background: drawn at the far plane, never writes depth.
    let bg_depth_state = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: Default::default(),
        bias: Default::default(),
    };

    let mk = |label: &str, vs: &str, fs: &str, depth: Option<&wgpu::DepthStencilState>| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // birds are double-sided triangles
                ..Default::default()
            },
            depth_stencil: depth.cloned(),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    let _ = &bg_depth_state; // (the sky now renders depthless into the offscreen target)
    let bird = mk("birds", "vs_bird", "fs_bird", Some(&depth_state));
    let ground = mk("ground", "vs_ground", "fs_ground", Some(&depth_state));
    // Background renders into the half-res offscreen sky target (no depth there).
    let background = mk("background", "vs_bg", "fs_bg", None);

    match pollster::block_on(scope.pop()) {
        Some(err) => Err(err),
        None => Ok((bird, ground, background)),
    }
}

pub const SKY_DIV_DEFAULT: u32 = 2; // sky rendered at 1/div resolution, then upscaled

// Standalone upscale-blit shader (separate module so its group-0 texture/sampler
// don't collide with the RenderUniforms binding in the bird+background module).
const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var sky_tex: texture_2d<f32>;
@group(0) @binding(1) var sky_samp: sampler;
struct VO { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VO {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let c = p[vi];
    var o: VO;
    o.clip = vec4<f32>(c, 0.99999, 1.0);
    // Framebuffer is top-left origin; flip y so the upscaled sky matches.
    o.uv = vec2<f32>(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5);
    return o;
}
@fragment
fn fs_main(in: VO) -> @location(0) vec4<f32> {
    return textureSampleLevel(sky_tex, sky_samp, in.uv, 0.0);
}
"#;

fn create_sky_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    div: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let div = div.clamp(1, 8);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sky half-res target"),
        size: wgpu::Extent3d {
            width: (width / div).max(1),
            height: (height / div).max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn make_blit_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    sky_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sky blit bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(sky_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn build_blit_pipeline(
    device: &wgpu::Device,
    blit_bgl: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, wgpu::Error> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sky blit"),
        source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("blit layout"),
        bind_group_layouts: &[Some(blit_bgl)],
        ..Default::default()
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sky blit pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        // In the scene pass (which has depth): fill the background, no depth test/write.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    match pollster::block_on(scope.pop()) {
        Some(err) => Err(err),
        None => Ok(pipeline),
    }
}
