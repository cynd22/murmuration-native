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

// Fluid sky field — dye.x = cloud density, dye.y = heat (fresh injection).
@group(0) @binding(3) var fluid_tex: texture_2d<f32>;
@group(0) @binding(4) var fluid_samp: sampler;

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

// Real aurora-borealis colour ramp, base of the curtain -> ray tips:
// oxygen green (the dominant 557.7nm emission) -> teal -> the high-altitude
// oxygen/nitrogen red-magenta. Fixed naturalistic colours; the audio drives
// brightness and motion, not hue, so it reads as real aurora rather than an
// RGB light show.
fn aurora_color(t: f32) -> vec3<f32> {
    let green   = vec3<f32>(0.05, 1.00, 0.35);
    let teal    = vec3<f32>(0.10, 0.80, 0.75);
    let magenta = vec3<f32>(0.70, 0.18, 0.85);
    let c1 = mix(green, teal, smoothstep(0.0, 0.5, t));
    return mix(c1, magenta, smoothstep(0.5, 1.0, t));
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

    // === Fluid layer. The stable-fluids dye field, screen-aligned, masked to
    // the sky. Density is lit by the shared palette (dark-trough applies);
    // heat (recent injection) adds a brighter core so fresh sub-bass plumes
    // read as glowing thermals that cool as they rise and disperse.
    if (u.bg2.x > 0.001) {
        let fuv = vec2<f32>(in.ndc.x * 0.5 + 0.5, 1.0 - (in.ndc.y * 0.5 + 0.5));
        let f = textureSampleLevel(fluid_tex, fluid_samp, fuv, 0.0);
        let density = clamp(f.x, 0.0, 2.0);
        let heat = clamp(f.y, 0.0, 2.0);
        let sky_mask = smoothstep(-0.02, 0.12, dir.y);
        // Heat shifts hue along the SAME palette family — fresh kick plumes
        // arrive in a sibling colour and settle back into the haze tint as
        // they cool. Curated-family variation, never full-RGB.
        let pal_hot = palette(u.palette_t + heat * 0.18, u.palette_a.xyz, u.palette_b.xyz, u.palette_c.xyz, u.palette_d.xyz);
        let fluid_light = pal_hot * (0.18 + 0.5 * heat) * u.palette_intensity + sky * 0.3;
        let amount = clamp(density * u.bg2.x, 0.0, 1.0) * sky_mask * u.bg.x;
        col = mix(col, fluid_light, amount);
        // Heat cores push a little extra light additively.
        col += pal_hot * heat * u.bg2.y * 0.15 * sky_mask * u.bg.x;
    }

    // === Aurora borealis — CORONA VIEW (looking up from below). INDEPENDENT
    // layer (own on/off via bg2.z), separate from the bottom fluid plumes
    // (bg2.x), which keep their own switch. Real aurora colours (green base ->
    // teal -> magenta toward the zenith). Instead of a flat band seen edge-on
    // ("across the atmosphere"), this projects flowing curtains onto a sky dome
    // and accumulates them over depth: as we look up (dir.y grows) the plane
    // coordinate shrinks, so every layer converges overhead — the free-flowing
    // rays-from-the-zenith look you get standing underneath. Driven by the
    // fluid field + bass/beat/air.
    if (u.bg2.z > 0.001 && dir.y > 0.015) {
        let proj = dir.xz / (dir.y + 0.07); // converges overhead, streaks to horizon
        var dens = 0.0;
        let STEPS = 5;
        for (var k = 0; k < STEPS; k = k + 1) {
            let fk = f32(k);
            let scl = 0.55 + fk * 0.45;
            let dr = vec2<f32>(t_sec * (0.018 + fk * 0.004), t_sec * 0.010);
            // Domain-warped FBM -> free-flowing curtains that ripple and drift.
            let warp = fbm(proj * scl * 0.5 + dr);
            let d = fbm(proj * scl + vec2<f32>(warp * 1.3, warp * 0.6) - dr);
            // The fluid field warps/brightens the curtains with the music.
            let fsamp = textureSampleLevel(
                fluid_tex, fluid_samp,
                vec2<f32>(fract(proj.x * 0.04 + 0.5), fract(proj.y * 0.04 + 0.5)), 0.0
            ).x;
            let curtain = pow(smoothstep(0.40, 0.92, d + fsamp * 0.10), 1.5);
            // Nearer (lower) layers slightly stronger; far layers fade -> depth.
            dens += curtain * (1.0 - fk / f32(STEPS) * 0.55);
        }
        // Fine vertical shimmer from treble/air; fade in over the upper sky.
        let shimmer = 1.0 + air * 0.5 * sin(proj.x * 5.0 + t_sec * 3.0);
        let height_mask = smoothstep(0.02, 0.15, dir.y);
        // Colour green (low) -> magenta (toward the zenith).
        let ct = clamp(dir.y * 3.0, 0.0, 1.0);
        let apulse = pow(0.5 + 0.5 * cos(u.beat.x * 6.28318), 3.0);
        let drive = 0.22 + 0.85 * bass + 0.6 * apulse * u.beat.y * u.bg.w;
        var aglow = aurora_color(ct) * dens * shimmer * height_mask * drive * 0.16 * u.bg2.z;
        // Soft roll-off: the banding was bright green hard-clipping to 1.0 (a
        // saturated flat patch with a sharp edge). Rolling off keeps the
        // gradient smooth; the final TPDF dither cleans up the rest.
        aglow = aglow / (1.0 + aglow * 0.7);
        col += aglow * u.bg.x;
    }

    // Horizon storm-glow: subBass lifts light along the horizon; the beat
    // breathes it when the tracker is confident. pow shapes the pulse so it
    // taps on the beat instant rather than sloshing sinusoidally.
    let pulse = pow(0.5 + 0.5 * cos(u.beat.x * 6.28318), 3.0);
    let beat_breathe = 1.0 + u.bg.w * u.beat.y * (pulse - 0.5);
    let glow_strength = (0.12 + 0.9 * sub_bass) * beat_breathe;
    // Glow hue rides subBass the same way the birds' palette rides bass —
    // hits push the horizon along the palette family, releases settle it back.
    let pal_glow = palette(u.palette_t + sub_bass * 0.14, u.palette_a.xyz, u.palette_b.xyz, u.palette_c.xyz, u.palette_d.xyz);
    let glow = pal_glow * glow_strength * pow(horizon_band, 3.0) * 0.45;
    col += glow * u.bg.x;

    // Below the horizon fade toward the ground colour so the plane blends in.
    if (dir.y < 0.0) {
        col = mix(col, vec3<f32>(0.01569, 0.03529, 0.07059), clamp(-dir.y * 6.0, 0.0, 1.0));
    }

    // TPDF dither: the sky is slow, dark gradients on an 8-bit surface, so
    // banding is guaranteed without this — most visible where the beat-breathing
    // horizon glow brightens a near-black ramp. Triangular-PDF (two independent
    // hashes summed) at ±1 LSB makes the quantization error white and
    // signal-independent, which uniform-PDF dither doesn't — it's what fully
    // kills the bands. Animated per-frame so the eye time-averages it to clean.
    let seed = in.ndc + fract(u.beat.w) * vec2<f32>(13.7, 7.1);
    let r1 = hash21(seed * vec2<f32>(1741.3, 921.7));
    let r2 = hash21(seed * vec2<f32>(913.7, 1733.1) + 19.19);
    col += vec3<f32>((r1 + r2 - 1.0) * (1.0 / 255.0));

    return vec4<f32>(col, 1.0);
}
