// Bird + ground rendering. Birds use vertex pulling: no vertex buffer, the
// vertex shader derives (bird index, vertex index) from the global vertex id
// and reads position/velocity straight from the sim's storage buffers — the
// wgpu equivalent of the single-mesh custom-geometry path (one draw call, no
// per-bird CPU work).
//
// Colour layer: Inigo Quilez cosine palette multiplied against the depth-grey
// base — the multiply is load-bearing for the dark-trough effect (near-zero
// palette values black the birds out during quiet sections).

struct RenderUniforms {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,  // clip → world rays for the sky shader
    palette_a: vec4<f32>,
    palette_b: vec4<f32>,
    palette_c: vec4<f32>,
    palette_d: vec4<f32>,
    palette_t: f32,
    palette_intensity: f32,
    palette_enabled: f32,
    time: f32,
    num_birds: f32,
    twinkle_amount: f32,
    _pad0: f32,
    _pad1: f32,
    bands: vec4<f32>,  // smoothed (subBass, bass, mid, air)
    beat: vec4<f32>,   // (phase 0..1, confidence, bpm/200, time seconds)
    bg: vec4<f32>,     // (intensity, cloud amount, star amount, beat-pulse amount)
    bg2: vec4<f32>,    // (fluid mix, fluid heat glow, unused, unused)
}

@group(0) @binding(0) var<uniform> u: RenderUniforms;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;

// Three triangles per bird (body + two wings), pre-scaled by the 0.2 the
// WebGL build baked into its geometry.
const BIRD_VERTS = array<vec3<f32>, 9>(
    // body
    vec3<f32>(0.0, 0.0, -4.0),
    vec3<f32>(0.0, 0.8, -4.0),
    vec3<f32>(0.0, 0.0, 6.0),
    // left wing
    vec3<f32>(0.0, 0.0, -3.0),
    vec3<f32>(-4.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, 3.0),
    // right wing
    vec3<f32>(0.0, 0.0, 3.0),
    vec3<f32>(4.0, 0.0, 0.0),
    vec3<f32>(0.0, 0.0, -3.0),
);

struct BirdOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_z: f32,
}

@vertex
fn vs_bird(@builtin(vertex_index) vi: u32) -> BirdOut {
    let bird = vi / 9u;
    let vert = vi % 9u;

    let pos_phase = positions[bird];
    let bird_pos = pos_phase.xyz;
    let phase = pos_phase.w;

    var v = BIRD_VERTS[vert];

    // Wing flap — wing tips (verts 4 and 7) ride a sine on the per-bird phase.
    if (vert == 4u || vert == 7u) {
        v.y = sin(phase) * 5.0;
    }

    // The WebGL build's mesh carried a baked rotation.y = PI/2; fold it in.
    v = vec3<f32>(v.z, v.y, -v.x);

    // Orient along velocity — direct port of the birdVS rotation math.
    var vel = velocities[bird].xyz;
    let vlen = length(vel);
    if (vlen > 0.00001) {
        vel /= vlen;
    } else {
        vel = vec3<f32>(1.0, 0.0, 0.0);
    }
    vel.z = -vel.z;
    let xz = max(length(vel.xz), 0.00001);
    let x = sqrt(max(1.0 - vel.y * vel.y, 0.0));

    let cosry = vel.x / xz;
    let sinry = vel.z / xz;
    let cosrz = x;
    let sinrz = vel.y;

    let maty = mat3x3<f32>(
        vec3<f32>(cosry, 0.0, -sinry),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(sinry, 0.0, cosry),
    );
    let matz = mat3x3<f32>(
        vec3<f32>(cosrz, sinrz, 0.0),
        vec3<f32>(-sinrz, cosrz, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );

    let world = maty * matz * v + bird_pos;

    // Per-bird plumage colour — dark blue-grey gradient across the flock with
    // deterministic per-bird jitter (same constants as BirdGeometry).
    let t = f32(bird) / u.num_birds;
    let seed = f32(((bird % 233280u) * 9301u + 49297u) % 233280u) / 233280.0;
    let jitter = (seed - 0.5) * 0.08;
    let color = vec3<f32>(
        0.18 + t * 0.22 + jitter,
        0.20 + t * 0.22 + jitter,
        0.26 + t * 0.24 + jitter * 0.5,
    );

    var out: BirdOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.color = color;
    out.world_z = world.z;
    return out;
}

fn palette(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(6.28318 * (c * t + d));
}

@fragment
fn fs_bird(in: BirdOut) -> @location(0) vec4<f32> {
    // Depth-based grey: closer = brighter, scaled by per-bird variation.
    let z2 = 0.2 + (1000.0 - in.world_z) / 1000.0 * in.color.x;
    let base = vec3<f32>(z2);

    let pal = palette(u.palette_t, u.palette_a.xyz, u.palette_b.xyz, u.palette_c.xyz, u.palette_d.xyz);
    let colored = base * pal * u.palette_intensity;
    var final_color = mix(base, colored, u.palette_enabled);

    // Air-band shimmer — additive, zero when no air content, so it never
    // disturbs the dark trough.
    if (u.twinkle_amount > 0.001) {
        let twinkle_seed = in.color.x * 7.31 + in.color.y * 11.7;
        let twinkle =
            (sin(u.time * 0.020 + twinkle_seed * 6.28) * 0.5 + 0.5) *
            (sin(u.time * 0.013 + twinkle_seed * 17.0) * 0.5 + 0.5);
        final_color += vec3<f32>(twinkle * u.twinkle_amount);
    }

    return vec4<f32>(final_color, 1.0);
}

// === Ground plane — 8000x8000 quad at ground height, fogged into the sky so
// the horizon dissolves at distance (linear fog on view-space depth, matching
// THREE.Fog 600..3500).

const GROUND_Y: f32 = -300.0;
const GROUND_COLOR = vec3<f32>(0.01569, 0.03529, 0.07059); // 0x040912
const SKY_COLOR = vec3<f32>(0.03922, 0.07843, 0.15686);    // 0x0a1428
const FOG_NEAR: f32 = 600.0;
const FOG_FAR: f32 = 3500.0;

struct GroundOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) fog_depth: f32,
}

@vertex
fn vs_ground(@builtin(vertex_index) vi: u32) -> GroundOut {
    // Two triangles from vertex id: (0,0) (1,0) (0,1) / (1,0) (1,1) (0,1).
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let c = corners[vi] * 4000.0;
    let world = vec3<f32>(c.x, GROUND_Y, c.y);

    var out: GroundOut;
    out.clip = u.view_proj * vec4<f32>(world, 1.0);
    out.fog_depth = -(u.view * vec4<f32>(world, 1.0)).z;
    return out;
}

@fragment
fn fs_ground(in: GroundOut) -> @location(0) vec4<f32> {
    let f = clamp((in.fog_depth - FOG_NEAR) / (FOG_FAR - FOG_NEAR), 0.0, 1.0);
    var col = mix(GROUND_COLOR, SKY_COLOR, f);
    // Sub-LSB dither — the fog ramp bands on 8-bit output without it.
    var q = fract(in.clip.xy * vec2<f32>(0.1031, 0.1369) + fract(u.time * 0.001));
    q += dot(q, q.yx + 33.33);
    col += vec3<f32>((fract((q.x + q.y) * q.x) - 0.5) * (1.6 / 255.0));
    return vec4<f32>(col, 1.0);
}
