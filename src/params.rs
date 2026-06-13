//! Simulation tunables and the GPU-side uniform struct.
//!
//! `Settings` holds every tunable in CPU-friendly form (defaults match the
//! HTML build's effectController + constants exactly). `SimParams` is the
//! byte-for-byte mirror of the WGSL `Params` struct in shaders/common.wgsl —
//! if you change one, change the other.

// === Spatial-hash grid. GRID_Y raised 8->16 (experiment): the flock is oblate
// (Y attraction x2.5) so Y is the packing axis, yet it had the fewest/fattest
// cells. Y cell = 2600/16 = 162.5, still >= the loud zone-radius ceiling (~136)
// so the 3x3x3 stencil stays correct; per-cell Y occupancy roughly halves.
pub const GRID_X: u32 = 16;
pub const GRID_Y: u32 = 16;
pub const GRID_Z: u32 = 16;
pub const SLOTS_PER_CELL: u32 = 16; // minimum / low-tier fallback; see slots_for()
pub const CELLS: u32 = GRID_X * GRID_Y * GRID_Z;

/// Slots-per-cell scaled to bird count so the cap stops silently dropping
/// neighbours at high N (~2.5x mean occupancy headroom, power-of-two, clamped).
/// 10k->16, 150k->~128, 1M->256. grid_slots is a runtime uniform, so this only
/// affects the grid_cells buffer size and the value handed to the shaders.
pub fn slots_for(n: u32) -> u32 {
    // NOTE (experiment finding): scaling slots up to mean*2.5 collapsed perf —
    // a murmuration is densely clumped, so dense-cell birds then honestly visit
    // 100+ neighbours x 27 cells (the O(N x density) wall). A modest cap (32)
    // restores some neighbour honesty at 150k at ~2x dense-cell cost; going
    // higher is the cliff. 16 stays the floor for parity with master.
    let mean = (2.5 * n as f32 / CELLS as f32).ceil() as u32;
    mean.max(SLOTS_PER_CELL).next_power_of_two().clamp(16, 32)
}

// World extent the grid covers — slightly larger than the bounding box so edge
// birds hash into real cells. If you change the box, change this too.
pub const FLOCK_EXTENT: [f32; 3] = [800.0, 1300.0, 800.0];

#[derive(Clone)]
pub struct Settings {
    pub separation: f32,
    pub alignment: f32,
    pub cohesion: f32,
    pub freedom: f32,
    pub attraction_strength: f32,
    pub shape_center: [f32; 3],
    pub drift: [f32; 3],
    pub min_speed: f32,
    pub max_speed: f32,
    /// Per-frame damping multiplier AT 60 FPS — the value the HTML build used.
    /// Converted to the actual tick rate in `sim_params` so drag-per-second is
    /// identical at any sim Hz.
    pub velocity_damping_60: f32,
    /// Radians/sec heading cap. Large value = clamp never fires (mapping off).
    pub max_turn_rate: f32,
    pub shear: f32,
    pub ground_y: f32,
    pub ground_repel_height: f32,
    pub ground_repel_strength: f32,
    pub shock_strength: f32,
    pub shock_source_y: f32,
    pub shock_reach: f32,
    pub box_min: [f32; 3],
    pub box_max: [f32; 3],
    pub box_margin: f32,
    pub box_strength: f32,
    pub camera_repel_radius: f32,
    pub camera_repel_strength: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            separation: 14.0,
            alignment: 35.0,
            cohesion: 44.375,
            freedom: 0.6,
            attraction_strength: 1.9,
            shape_center: [0.0, 0.0, 0.0],
            drift: [0.1, 0.0, 0.0],
            min_speed: 2.7,
            max_speed: 9.0,
            velocity_damping_60: 0.98,
            max_turn_rate: 1000.0,
            shear: 0.0,
            ground_y: -300.0,
            ground_repel_height: 120.0,
            ground_repel_strength: 25.0,
            shock_strength: 0.0,
            shock_source_y: -200.0,
            shock_reach: 250.0,
            box_min: [-770.0, -250.0, -770.0],
            box_max: [770.0, 1250.0, 770.0],
            box_margin: 80.0,
            box_strength: 9.0,
            camera_repel_radius: 700.0,
            camera_repel_strength: 60.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub dt: f32,
    pub time_ms: f32, // milliseconds — the freedom-noise hash was tuned on ms
    pub num_birds: u32,
    pub grid_slots: u32,
    pub grid_dim: [f32; 4],
    pub cell_size: [f32; 4],
    pub world_min: [f32; 4],
    pub separation_dist: f32,
    pub alignment_dist: f32,
    pub cohesion_dist: f32,
    pub freedom: f32,
    pub shape_center: [f32; 4], // w = attraction strength
    pub drift: [f32; 4],        // w = min speed
    pub max_speed: f32,
    pub damping_per_tick: f32,
    pub max_turn_rate: f32,
    pub shear: f32,
    pub ground_y: f32,
    pub ground_repel_height: f32,
    pub ground_repel_strength: f32,
    pub shock_strength: f32,
    pub shock_source_y: f32,
    pub shock_reach: f32,
    pub box_margin: f32,
    pub box_strength: f32,
    pub box_min: [f32; 4],
    pub box_max: [f32; 4],
    pub camera_pos: [f32; 4], // w = camera repel radius
    pub camera_repel_strength: f32,
    pub tex_width: f32,
    pub _pad0: f32,
    pub _pad1: f32,
}

impl Settings {
    pub fn sim_params(
        &self,
        dt: f32,
        time_ms: f32,
        num_birds: u32,
        camera_pos: [f32; 3],
    ) -> SimParams {
        let cell_size = [
            FLOCK_EXTENT[0] * 2.0 / GRID_X as f32,
            FLOCK_EXTENT[1] * 2.0 / GRID_Y as f32,
            FLOCK_EXTENT[2] * 2.0 / GRID_Z as f32,
        ];
        SimParams {
            dt,
            time_ms,
            num_birds,
            grid_slots: slots_for(num_birds),
            grid_dim: [GRID_X as f32, GRID_Y as f32, GRID_Z as f32, 0.0],
            cell_size: [cell_size[0], cell_size[1], cell_size[2], 0.0],
            world_min: [-FLOCK_EXTENT[0], -FLOCK_EXTENT[1], -FLOCK_EXTENT[2], 0.0],
            separation_dist: self.separation,
            alignment_dist: self.alignment,
            cohesion_dist: self.cohesion,
            freedom: self.freedom,
            shape_center: [
                self.shape_center[0],
                self.shape_center[1],
                self.shape_center[2],
                self.attraction_strength,
            ],
            drift: [self.drift[0], self.drift[1], self.drift[2], self.min_speed],
            max_speed: self.max_speed,
            // 0.98/frame at 60fps == 0.98^(60*dt) per tick at any rate.
            damping_per_tick: self.velocity_damping_60.powf(60.0 * dt),
            max_turn_rate: self.max_turn_rate,
            shear: self.shear,
            ground_y: self.ground_y,
            ground_repel_height: self.ground_repel_height,
            ground_repel_strength: self.ground_repel_strength,
            shock_strength: self.shock_strength,
            shock_source_y: self.shock_source_y,
            shock_reach: self.shock_reach,
            box_margin: self.box_margin,
            box_strength: self.box_strength,
            box_min: [self.box_min[0], self.box_min[1], self.box_min[2], 0.0],
            box_max: [self.box_max[0], self.box_max[1], self.box_max[2], 0.0],
            camera_pos: [
                camera_pos[0],
                camera_pos[1],
                camera_pos[2],
                self.camera_repel_radius,
            ],
            camera_repel_strength: self.camera_repel_strength,
            tex_width: (num_birds as f32).sqrt().ceil(),
            _pad0: 0.0,
            _pad1: 0.0,
        }
    }
}
