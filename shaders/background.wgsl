// Audio-reactive sky. Compiled into the same module as bird.wgsl (the Rust
// side concatenates them), so RenderUniforms `u` is already declared.
//
// Design intent (NOT a self-consistent cloud sim — every visible parameter is
// a direct transform of the audio, so different songs paint different skies):
//   - Base: the dusk-navy gradient the project has always had. Quiet music
//     leaves the sky almost exactly as it was — the dark trough extends to
//     the whole frame.
//   - Clouds: two layers of drifting FBM, tinted by the SAME cosine palette
//     as the birds at the SAME palette_t — so the sky ignites with the flock
//     on drops and goes near-black with it in build-ups. Bass swells cloud
//     luminance; mid widens the band of sky they occupy.
//   - Horizon glow: subBass lifts a storm-light along the horizon line.
//   - Stars: sparse hash stars above the cloud band, shimmering with air.
//   - Beat pulse: when BPM confidence is high, the horizon glow breathes in
//     beat phase. Subtle by design; scale with bg.w.
// Everything is multiplied by bg.x (master intensity) — 0 restores the flat
// classic sky exactly.

fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let s = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 5; i++) {
        sum += vnoise(p) * amp;
        p = p * 2.03 + vec2<f32>(17.3, 9.1);
        amp *= 0.5;
    }
    return sum;
}

struct BgOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_bg(@builtin(vertex_index) vi: u32) -> BgOut {
    // Fullscreen triangle.
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0),
    );
    var out: BgOut;
    out.clip = vec4<f32>(pts[vi], 0.99999, 1.0); // at the far plane, behind everything
    out.ndc = pts[vi];
    return out;
}

@fragment
fn fs_bg(in: BgOut) -> @location(0) vec4<f32> {
    // World-space view ray for this pixel.
    let near = u.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let far = u.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let t_sec = u.beat.w;
    let sub_bass = u.bands.x;
    let bass = u.bands.y;
    let mid = u.bands.z;
    let air = u.bands.w;

    // Base dusk gradient: classic sky colour, slightly deeper at zenith and
    // slightly lifted toward the horizon.
    let sky = vec3<f32>(0.03922, 0.07843, 0.15686);
    let horizon_band = 1.0 - clamp(abs(dir.y) * 3.0, 0.0, 1.0); // 1 at horizon → 0 up/down
    var col = sky * (0.85 + 0.35 * horizon_band);

    // Shared palette colour — the sky's tint IS the flock's tint.
    let pal = palette(u.palette_t, u.palette_a.xyz, u.palette_b.xyz, u.palette_c.xyz, u.palette_d.xyz);

    if (dir.y > 0.001) {
        // Project the ray onto a high "cloud plane" for stable cloud shapes.
        let cp = dir.xz / (dir.y + 0.12);

        // Two drifting FBM layers; drift direction matches the flock's wind.
        let drift1 = vec2<f32>(t_sec * 0.006, t_sec * 0.0011);
        let drift2 = vec2<f32>(t_sec * -0.0023, t_sec * 0.0034);
        let n1 = fbm(cp * 0.9 + drift1);
        let n2 = fbm(cp * 2.1 + drift2 + n1 * 0.35);
        // Cloud coverage: mid content widens/blooms the deck (busy mix =
        // fuller sky); the shaping keeps a clear band near the zenith.
        let coverage = 0.32 + 0.25 * mid;
        let cloud = smoothstep(1.0 - coverage, 1.0 - coverage + 0.45, n1 * 0.65 + n2 * 0.35);
        // Luminance: a dim ambient floor so clouds are always faintly there,
        // plus bass-driven palette light — clouds ignite WITH the birds.
        let cloud_light = sky * 0.6 + pal * (0.10 + 0.55 * bass) * u.palette_intensity;
        let cloud_amount = cloud * u.bg.y * (0.35 + 0.65 * smoothstep(0.0, 0.5, dir.y));
        col = mix(col, cloud_light, clamp(cloud_amount, 0.0, 1.0) * u.bg.x);

        // Stars: only well above the horizon, occluded by clouds, shimmering
        // with air content. Sparse by construction.
        let star_cell = floor(dir.xz / max(dir.y, 0.05) * 90.0);
        let star_h = hash21(star_cell);
        if (star_h > 0.997) {
            let tw = 0.5 + 0.5 * sin(t_sec * (1.5 + star_h * 3.0) + star_h * 40.0);
            let star = (star_h - 0.997) / 0.003 * tw * (0.25 + 0.75 * air);
            col += vec3<f32>(star * 0.5) * (1.0 - cloud) * smoothstep(0.08, 0.3, dir.y) * u.bg.z * u.bg.x;
        }
    }

    // Horizon storm-glow: subBass lifts light along the horizon; the beat
    // breathes it when the tracker is confident. pow shapes the pulse so it
    // taps on the beat instant rather than sloshing sinusoidally.
    let pulse = pow(0.5 + 0.5 * cos(u.beat.x * 6.28318), 3.0);
    let beat_breathe = 1.0 + u.bg.w * u.beat.y * (pulse - 0.5);
    let glow_strength = (0.12 + 0.9 * sub_bass) * beat_breathe;
    let glow = pal * glow_strength * pow(horizon_band, 3.0) * 0.45;
    col += glow * u.bg.x;

    // Below the horizon fade toward the ground colour so the plane blends in.
    if (dir.y < 0.0) {
        col = mix(col, vec3<f32>(0.01569, 0.03529, 0.07059), clamp(-dir.y * 6.0, 0.0, 1.0));
    }

    return vec4<f32>(col, 1.0);
}
