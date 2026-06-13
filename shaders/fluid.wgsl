// 2D stable-fluids sky simulation (Stam advection + vorticity confinement +
// Jacobi pressure solve — the classic WebGL-fluid-simulation recipe), run on
// small rgba16float textures. One bind group layout serves every pass; the
// Rust side swaps which textures sit in slots a/b/out per pass.
//
// AUDIO IS THE ONLY ENERGY SOURCE. The fluid has no self-sustaining forcing:
// no audio → splats stop → dye dissipates → flat sky. SubBass/kicks drive
// rising plumes from the bottom edge, treble stirs small vortices, mid is
// lateral wind, beats pulse the plume. This is what keeps a "physics sim"
// from becoming a normaliser — the dynamics are a transform of the music.
//
// Field layout (all rgba16float):
//   velocity: xy = uv-space velocity (v positive = downward in texture space)
//   dye:      x  = cloud density, y = "heat" (recent-injection marker, used
//                  by the sky shader to brighten fresh plumes)
//   curl/divergence/pressure: x only.

struct FluidParams {
    dt: f32,
    vorticity: f32,        // confinement strength (swirl)
    vel_dissipation: f32,  // per-step velocity keep-factor
    dye_dissipation: f32,  // per-step dye keep-factor
    wind: vec2<f32>,       // uv/sec lateral drift (mid band)
    texel: vec2<f32>,      // 1 / texture size
    // 4 splats: [i*2] = (x, y, radius, unused), [i*2+1] = (fx, fy, dye, heat)
    splats: array<vec4<f32>, 8>,
}

@group(0) @binding(0) var<uniform> fp: FluidParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var tex_a: texture_2d<f32>;
@group(0) @binding(3) var tex_b: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;

fn uv_of(gid: vec3<u32>) -> vec2<f32> {
    return (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) * fp.texel;
}

fn in_bounds(gid: vec3<u32>) -> bool {
    let size = vec2<f32>(1.0, 1.0) / fp.texel;
    return f32(gid.x) < size.x && f32(gid.y) < size.y;
}

// === Splat passes: gaussian injection of force (velocity) or dye.

@compute @workgroup_size(8, 8)
fn splat_vel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let uv = uv_of(gid);
    var v = textureSampleLevel(tex_a, samp, uv, 0.0);
    // Aspect-correct the distance so splats are round on a wide texture.
    let aspect = fp.texel.y / fp.texel.x;
    for (var i = 0; i < 4; i++) {
        let pr = fp.splats[i * 2];
        let fd = fp.splats[i * 2 + 1];
        if (pr.z <= 0.0) { continue; }
        var d = uv - pr.xy;
        d.x *= aspect;
        let g = exp(-dot(d, d) / (pr.z * pr.z));
        v = vec4<f32>(v.xy + fd.xy * g, v.zw);
    }
    // Wind: constant gentle bias rather than a splat.
    v = vec4<f32>(v.xy + fp.wind * fp.dt, v.zw);
    textureStore(out_tex, vec2<i32>(gid.xy), v);
}

@compute @workgroup_size(8, 8)
fn splat_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let uv = uv_of(gid);
    var dye = textureSampleLevel(tex_a, samp, uv, 0.0);
    let aspect = fp.texel.y / fp.texel.x;
    for (var i = 0; i < 4; i++) {
        let pr = fp.splats[i * 2];
        let fd = fp.splats[i * 2 + 1];
        if (pr.z <= 0.0) { continue; }
        var d = uv - pr.xy;
        d.x *= aspect;
        let g = exp(-dot(d, d) / (pr.z * pr.z));
        dye.x += fd.z * g;
        dye.y += fd.w * g;
    }
    textureStore(out_tex, vec2<i32>(gid.xy), dye);
}

// === Vorticity confinement: measure curl, then push energy back into the
// swirls that numerical diffusion smears out. This is what makes it billow.

@compute @workgroup_size(8, 8)
fn curl(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(tex_a, p - vec2<i32>(1, 0), 0).y;
    let r = textureLoad(tex_a, p + vec2<i32>(1, 0), 0).y;
    let t = textureLoad(tex_a, p - vec2<i32>(0, 1), 0).x;
    let b = textureLoad(tex_a, p + vec2<i32>(0, 1), 0).x;
    textureStore(out_tex, p, vec4<f32>(0.5 * ((r - l) - (b - t)), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn vorticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let l = abs(textureLoad(tex_b, p - vec2<i32>(1, 0), 0).x);
    let r = abs(textureLoad(tex_b, p + vec2<i32>(1, 0), 0).x);
    let t = abs(textureLoad(tex_b, p - vec2<i32>(0, 1), 0).x);
    let b = abs(textureLoad(tex_b, p + vec2<i32>(0, 1), 0).x);
    let c = textureLoad(tex_b, p, 0).x;
    var force = 0.5 * vec2<f32>(abs(t) - abs(b), abs(r) - abs(l));
    force /= length(force) + 1e-4;
    force *= fp.vorticity * c;
    force.y = -force.y;
    var v = textureLoad(tex_a, p, 0);
    textureStore(out_tex, p, vec4<f32>(v.xy + force * fp.dt, v.zw));
}

// === Incompressibility: divergence → Jacobi pressure relax → project.

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(tex_a, p - vec2<i32>(1, 0), 0).x;
    let r = textureLoad(tex_a, p + vec2<i32>(1, 0), 0).x;
    let t = textureLoad(tex_a, p - vec2<i32>(0, 1), 0).y;
    let b = textureLoad(tex_a, p + vec2<i32>(0, 1), 0).y;
    textureStore(out_tex, p, vec4<f32>(0.5 * (r - l + b - t), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(tex_a, p - vec2<i32>(1, 0), 0).x;
    let r = textureLoad(tex_a, p + vec2<i32>(1, 0), 0).x;
    let t = textureLoad(tex_a, p - vec2<i32>(0, 1), 0).x;
    let b = textureLoad(tex_a, p + vec2<i32>(0, 1), 0).x;
    let div = textureLoad(tex_b, p, 0).x;
    textureStore(out_tex, p, vec4<f32>((l + r + t + b - div) * 0.25, 0.0, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn subtract_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let p = vec2<i32>(gid.xy);
    let l = textureLoad(tex_b, p - vec2<i32>(1, 0), 0).x;
    let r = textureLoad(tex_b, p + vec2<i32>(1, 0), 0).x;
    let t = textureLoad(tex_b, p - vec2<i32>(0, 1), 0).x;
    let b = textureLoad(tex_b, p + vec2<i32>(0, 1), 0).x;
    var v = textureLoad(tex_a, p, 0);
    textureStore(out_tex, p, vec4<f32>(v.xy - 0.5 * vec2<f32>(r - l, b - t), v.zw));
}

// === Semi-Lagrangian advection (sample upstream along the velocity field).

@compute @workgroup_size(8, 8)
fn advect_vel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let uv = uv_of(gid);
    let v = textureSampleLevel(tex_a, samp, uv, 0.0).xy;
    let prev = uv - v * fp.dt;
    let sampled = textureSampleLevel(tex_a, samp, prev, 0.0);
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(sampled.xy * fp.vel_dissipation, sampled.zw));
}

@compute @workgroup_size(8, 8)
fn advect_dye(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (!in_bounds(gid)) { return; }
    let uv = uv_of(gid);
    let v = textureSampleLevel(tex_a, samp, uv, 0.0).xy;
    let prev = uv - v * fp.dt;
    let dye = textureSampleLevel(tex_b, samp, prev, 0.0);
    // Heat (y) cools ~3x faster than density so fresh plumes read brighter.
    textureStore(
        out_tex,
        vec2<i32>(gid.xy),
        vec4<f32>(
            dye.x * fp.dye_dissipation,
            dye.y * (fp.dye_dissipation * fp.dye_dissipation * fp.dye_dissipation),
            dye.zw,
        ),
    );
}
