//! Bird + ground rendering and the camera.
//!
//! The surface uses a NON-sRGB format deliberately: the HTML build's
//! ShaderMaterial wrote raw colour values into an sRGB-interpreted canvas, so
//! writing the same raw values to a non-sRGB swapchain reproduces its exact
//! look (including the dark-trough behaviour at low palette values).

use crate::hot::Sources;
use glam::{Mat4, Vec3};

// Camera defaults — match cameraSettings in the HTML build.
pub const CAM_FOV_DEG: f32 = 18.0;
pub const CAM_POS: Vec3 = Vec3::new(0.0, -300.0, 3500.0);
pub const CAM_LOOK_AT: Vec3 = Vec3::new(0.0, 250.0, 0.0);

// Sky 0x0a1428, raw (non-sRGB surface).
pub const SKY: wgpu::Color = wgpu::Color {
    r: 0.03922,
    g: 0.07843,
    b: 0.15686,
    a: 1.0,
};

// Ocean palette — the load-bearing default. paletteT sits at tOffset during
// silence, which is the dark trough.
const PALETTE_A: [f32; 4] = [0.0, 0.5, 0.5, 0.0];
const PALETTE_B: [f32; 4] = [0.0, 0.5, 0.5, 0.0];
const PALETTE_C: [f32; 4] = [0.0, 0.5, 0.333, 0.0];
const PALETTE_D: [f32; 4] = [0.0, 0.5, 0.667, 0.0];
const PALETTE_INTENSITY: f32 = 0.65;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RenderUniforms {
    view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
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
}

pub struct Renderer {
    uniforms_buf: wgpu::Buffer,
    bind_groups: [wgpu::BindGroup; 2],
    layout: wgpu::PipelineLayout,
    bird_pipeline: wgpu::RenderPipeline,
    ground_pipeline: wgpu::RenderPipeline,
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
        sources: &Sources,
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
                ],
            })
        };
        let bind_groups = [mk_bg(0), mk_bg(1)];

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });

        let (bird_pipeline, ground_pipeline) =
            build_pipelines(device, &layout, format, &sources.bird)?;

        Ok(Self {
            uniforms_buf,
            bind_groups,
            layout,
            bird_pipeline,
            ground_pipeline,
            depth: create_depth(device, width, height),
            format,
            n: sim.n,
        })
    }

    pub fn rebuild(&mut self, device: &wgpu::Device, sources: &Sources) -> Result<(), wgpu::Error> {
        let (bird, ground) = build_pipelines(device, &self.layout, self.format, &sources.bird)?;
        self.bird_pipeline = bird;
        self.ground_pipeline = ground;
        Ok(())
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth = create_depth(device, width, height);
    }

    pub fn update_uniforms(
        &self,
        queue: &wgpu::Queue,
        aspect: f32,
        time_ms: f32,
        palette_t: f32,
        twinkle: f32,
    ) {
        let view = Mat4::look_at_rh(CAM_POS, CAM_LOOK_AT, Vec3::Y);
        let proj = Mat4::perspective_rh(CAM_FOV_DEG.to_radians(), aspect, 1.0, 5000.0);
        let u = RenderUniforms {
            view_proj: (proj * view).to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            palette_a: PALETTE_A,
            palette_b: PALETTE_B,
            palette_c: PALETTE_C,
            palette_d: PALETTE_D,
            palette_t,
            palette_intensity: PALETTE_INTENSITY,
            palette_enabled: 1.0,
            time: time_ms,
            num_birds: self.n as f32,
            twinkle_amount: twinkle,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        queue.write_buffer(&self.uniforms_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, flip: usize) {
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

        pass.set_bind_group(0, &self.bind_groups[flip], &[]);
        pass.set_pipeline(&self.ground_pipeline);
        pass.draw(0..6, 0..1);
        pass.set_pipeline(&self.bird_pipeline);
        pass.draw(0..self.n * 9, 0..1);
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
    bird_src: &str,
) -> Result<(wgpu::RenderPipeline, wgpu::RenderPipeline), wgpu::Error> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bird.wgsl"),
        source: wgpu::ShaderSource::Wgsl(bird_src.into()),
    });

    let depth_state = wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgpu::CompareFunction::Less),
        stencil: Default::default(),
        bias: Default::default(),
    };

    let mk = |label: &str, vs: &str, fs: &str| {
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
            depth_stencil: Some(depth_state.clone()),
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

    let bird = mk("birds", "vs_bird", "fs_bird");
    let ground = mk("ground", "vs_ground", "fs_ground");

    match pollster::block_on(scope.pop()) {
        Some(err) => Err(err),
        None => Ok((bird, ground)),
    }
}
