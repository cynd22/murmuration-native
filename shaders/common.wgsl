// Shared prelude — prepended to every compute shader at load time (WGSL has no
// includes). The Params struct layout MUST match SimParams in src/params.rs
// field-for-field; vec3 values travel as vec4 to sidestep alignment surprises.

struct Params {
    // row 0
    dt: f32,            // fixed sim timestep (seconds)
    time: f32,          // sim time in MILLISECONDS — the freedom-noise hash was tuned on ms
    num_birds: u32,
    grid_slots: u32,    // max birds tracked per cell (overflow dropped)
    // rows 1-3
    grid_dim: vec4<f32>,   // xyz = cell counts (16,8,16)
    cell_size: vec4<f32>,  // xyz = world-space cell size
    world_min: vec4<f32>,  // xyz = grid anchor corner
    // row 4 — flocking radii (zone model: separation | alignment | cohesion)
    separation_dist: f32,
    alignment_dist: f32,
    cohesion_dist: f32,
    freedom: f32,
    // rows 5-6 — environmental steering
    shape_center: vec4<f32>,  // xyz = attractor point, w = attraction strength
    drift: vec4<f32>,         // xyz = wind velocity bias, w = min speed
    // row 7
    max_speed: f32,
    damping_per_tick: f32,    // velocity damping pre-converted for this dt
    max_turn_rate: f32,       // radians/sec heading cap (large = off)
    shear: f32,               // anisotropic separation along travel axis
    // row 8 — ground + shockwave
    ground_y: f32,
    ground_repel_height: f32,
    ground_repel_strength: f32,
    shock_strength: f32,
    // row 9
    shock_source_y: f32,
    shock_reach: f32,
    box_margin: f32,
    box_strength: f32,
    // rows 10-12
    box_min: vec4<f32>,
    box_max: vec4<f32>,
    camera_pos: vec4<f32>,    // xyz = camera world pos, w = repel radius
    // row 13
    camera_repel_strength: f32,
    tex_width: f32,           // ceil(sqrt(num_birds)) — preserves the GLSL rand() uv character
    bank_amount: f32,         // banking roll gain (cosmetic; stored bank → vel.w)
    _pad1: f32,
}

const PI: f32 = 3.141592653589793;
const PI_2: f32 = 6.283185307179586;

// Same hash the GLSL build used, so the freedom force keeps its exact character.
fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn cell_of(pos: vec3<f32>, p_world_min: vec3<f32>, p_cell_size: vec3<f32>, p_grid_dim: vec3<f32>) -> vec3<i32> {
    let c = floor((pos - p_world_min) / p_cell_size);
    return clamp(vec3<i32>(c), vec3<i32>(0), vec3<i32>(p_grid_dim) - vec3<i32>(1));
}
