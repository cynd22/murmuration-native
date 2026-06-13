// Position integration + wing-flap phase. Reads the velocity the velocity pass
// just wrote (vel_new), so position update uses this tick's velocity — same
// ordering as the WebGL build. w channel carries the wing phase: advance rate
// scales with horizontal speed and climb so fast/climbing birds beat harder.

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_new: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> pos_out: array<vec4<f32>>;

@compute @workgroup_size(64)
fn position_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_birds) {
        return;
    }

    let position = pos_in[i].xyz;
    let phase = pos_in[i].w;
    let velocity = vel_new[i].xyz;

    let new_phase = (phase + p.dt
        + length(velocity.xz) * p.dt * 3.0
        + max(velocity.y, 0.0) * p.dt * 6.0) % 62.83;

    pos_out[i] = vec4<f32>(position + velocity * p.dt * 15.0, new_phase);
}
