//! GPU boid simulation: storage buffers, compute pipelines, per-tick dispatch.
//!
//! Per tick: clear grid counters → grid build (scatter, atomics) → velocity
//! (flocking + environmental forces over the 3x3x3 cell stencil) → position
//! (integrate + wing phase). Position/velocity are double-buffered; `flip`
//! indexes the buffers holding current state.

use crate::hot::Sources;
use crate::params::{CELLS, SLOTS_PER_CELL};
use wgpu::util::DeviceExt;

struct Pipelines {
    grid: wgpu::ComputePipeline,
    velocity: wgpu::ComputePipeline,
    position: wgpu::ComputePipeline,
}

pub struct Sim {
    pub n: u32,
    pub pos: [wgpu::Buffer; 2],
    pub vel: [wgpu::Buffer; 2],
    pub params_buf: wgpu::Buffer,
    grid_count: wgpu::Buffer,
    // Held only by the bind groups after init; field kept for future debug readback.
    _grid_cells: wgpu::Buffer,
    grid_layout: wgpu::PipelineLayout,
    vel_layout: wgpu::PipelineLayout,
    pos_layout: wgpu::PipelineLayout,
    grid_bg: [wgpu::BindGroup; 2],
    vel_bg: [wgpu::BindGroup; 2],
    pos_bg: [wgpu::BindGroup; 2],
    pipelines: Pipelines,
    /// Index of the buffers holding CURRENT state (read side of next step).
    pub flip: usize,
}

// Deterministic spawn — same flock every run makes visual A/B against the
// HTML build meaningful.
fn splitmix(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z >> 40) & 0xFFFFFF) as f32 / 16_777_216.0
}

impl Sim {
    pub fn new(device: &wgpu::Device, n: u32, sources: &Sources) -> Result<Self, wgpu::Error> {
        let mut rng: u64 = 0x5EED_5EED;
        let mut init_pos = Vec::with_capacity(n as usize * 4);
        let mut init_vel = Vec::with_capacity(n as usize * 4);
        for _ in 0..n {
            // Same spawn volume as the HTML build: ±400 on each axis.
            init_pos.push(splitmix(&mut rng) * 800.0 - 400.0);
            init_pos.push(splitmix(&mut rng) * 800.0 - 400.0);
            init_pos.push(splitmix(&mut rng) * 800.0 - 400.0);
            init_pos.push(splitmix(&mut rng) * 62.83); // wing phase
            init_vel.push(splitmix(&mut rng) * 10.0 - 5.0);
            init_vel.push(splitmix(&mut rng) * 10.0 - 5.0);
            init_vel.push(splitmix(&mut rng) * 10.0 - 5.0);
            init_vel.push(0.0);
        }

        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let mk_state = |label: &str, data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: storage,
            })
        };
        let pos = [mk_state("pos A", &init_pos), mk_state("pos B", &init_pos)];
        let vel = [mk_state("vel A", &init_vel), mk_state("vel B", &init_vel)];

        let grid_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid count"),
            size: CELLS as u64 * 4,
            usage: storage,
            mapped_at_creation: false,
        });
        let grid_cells = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grid cells"),
            size: (CELLS * SLOTS_PER_CELL) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: std::mem::size_of::<crate::params::SimParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_entry = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let grid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("grid bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // positions
                storage_entry(2, false), // grid_count (atomic)
                storage_entry(3, false), // grid_cells
            ],
        });
        let vel_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("velocity bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // pos_in
                storage_entry(2, true),  // vel_in
                storage_entry(3, false), // vel_out
                storage_entry(4, true),  // grid_count
                storage_entry(5, true),  // grid_cells
            ],
        });
        let pos_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("position bgl"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // pos_in
                storage_entry(2, true),  // vel_new
                storage_entry(3, false), // pos_out
            ],
        });

        let mk_layout = |label: &str, bgl: &wgpu::BindGroupLayout| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                ..Default::default()
            })
        };
        let grid_layout = mk_layout("grid layout", &grid_bgl);
        let vel_layout = mk_layout("velocity layout", &vel_bgl);
        let pos_layout = mk_layout("position layout", &pos_bgl);

        let mk_grid_bg = |cur: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("grid bg"),
                layout: &grid_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pos[cur].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: grid_count.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: grid_cells.as_entire_binding() },
                ],
            })
        };
        let mk_vel_bg = |cur: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("velocity bg"),
                layout: &vel_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pos[cur].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: vel[cur].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: vel[cur ^ 1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: grid_count.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: grid_cells.as_entire_binding() },
                ],
            })
        };
        let mk_pos_bg = |cur: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("position bg"),
                layout: &pos_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: pos[cur].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: vel[cur ^ 1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: pos[cur ^ 1].as_entire_binding() },
                ],
            })
        };

        let grid_bg = [mk_grid_bg(0), mk_grid_bg(1)];
        let vel_bg = [mk_vel_bg(0), mk_vel_bg(1)];
        let pos_bg = [mk_pos_bg(0), mk_pos_bg(1)];

        let pipelines =
            build_pipelines(device, sources, &grid_layout, &vel_layout, &pos_layout)?;

        Ok(Self {
            n,
            pos,
            vel,
            params_buf,
            grid_count,
            _grid_cells: grid_cells,
            grid_layout,
            vel_layout,
            pos_layout,
            grid_bg,
            vel_bg,
            pos_bg,
            pipelines,
            flip: 0,
        })
    }

    /// Swap in newly compiled shaders; on WGSL error the old pipelines stay.
    pub fn rebuild(&mut self, device: &wgpu::Device, sources: &Sources) -> Result<(), wgpu::Error> {
        self.pipelines = build_pipelines(
            device,
            sources,
            &self.grid_layout,
            &self.vel_layout,
            &self.pos_layout,
        )?;
        Ok(())
    }

    /// Encode one fixed-dt tick. Params must already hold this tick's values.
    pub fn step(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let cur = self.flip;
        let workgroups = self.n.div_ceil(64);

        encoder.clear_buffer(&self.grid_count, 0, None);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.grid);
            pass.set_bind_group(0, &self.grid_bg[cur], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.velocity);
            pass.set_bind_group(0, &self.vel_bg[cur], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipelines.position);
            pass.set_bind_group(0, &self.pos_bg[cur], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.flip ^= 1;
    }
}

fn build_pipelines(
    device: &wgpu::Device,
    sources: &Sources,
    grid_layout: &wgpu::PipelineLayout,
    vel_layout: &wgpu::PipelineLayout,
    pos_layout: &wgpu::PipelineLayout,
) -> Result<Pipelines, wgpu::Error> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let mk_module = |label: &str, src: String| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        })
    };
    let grid_mod = mk_module("grid.wgsl", sources.with_common(&sources.grid));
    let vel_mod = mk_module("velocity.wgsl", sources.with_common(&sources.velocity));
    let pos_mod = mk_module("position.wgsl", sources.with_common(&sources.position));

    let mk_pipeline = |label: &str,
                       layout: &wgpu::PipelineLayout,
                       module: &wgpu::ShaderModule,
                       entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines = Pipelines {
        grid: mk_pipeline("grid build", grid_layout, &grid_mod, "build"),
        velocity: mk_pipeline("velocity", vel_layout, &vel_mod, "velocity_update"),
        position: mk_pipeline("position", pos_layout, &pos_mod, "position_update"),
    };

    match pollster::block_on(scope.pop()) {
        Some(err) => Err(err),
        None => Ok(pipelines),
    }
}
