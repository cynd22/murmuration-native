// Spatial-hash grid build. ONE thread per bird scatters its own index into its
// cell via an atomic counter — O(birds), vs the WebGL version's O(cells × birds)
// gather (every cell-pixel scanned every bird because fragment shaders can't
// scatter). This pass is why the port has the performance headroom it has.
// grid_count is zeroed by CommandEncoder::clear_buffer before this dispatch.

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> grid_count: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> grid_cells: array<u32>;

@compute @workgroup_size(64)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_birds) {
        return;
    }
    let c = cell_of(positions[i].xyz, p.world_min.xyz, p.cell_size.xyz, p.grid_dim.xyz);
    let gx = u32(p.grid_dim.x);
    let gy = u32(p.grid_dim.y);
    let cell_idx = u32(c.x) + u32(c.y) * gx + u32(c.z) * gx * gy;
    let slot = atomicAdd(&grid_count[cell_idx], 1u);
    // Birds past grid_slots in a dense cell are dropped for this tick — same
    // safety-valve semantics as the WebGL build (no crash, just unseen by
    // neighbours for one tick).
    if (slot < p.grid_slots) {
        grid_cells[cell_idx * p.grid_slots + slot] = i;
    }
}
