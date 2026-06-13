// Boid velocity update — faithful port of the WebGL velocity fragment shader.
// Force order is preserved exactly (it matters: the turn-rate clamp constrains
// steering but not the safety forces applied after it):
//   shockwave → [pre-steer snapshot] → attraction → drift → flocking (grid) →
//   freedom → turn-rate clamp → ground → box → camera bubble → damping →
//   min/max speed.

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> grid_count: array<u32>;
@group(0) @binding(5) var<storage, read> grid_cells: array<u32>;

@compute @workgroup_size(64)
fn velocity_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.num_birds) {
        return;
    }

    let dt = p.dt;
    let self_position = pos_in[i].xyz;
    let self_velocity = vel_in[i].xyz;
    var velocity = self_velocity;
    var limit = p.max_speed;

    // Zone model: one radius partitioned into separation / alignment / cohesion
    // shells by squared-distance percentage.
    let zone_radius = p.separation_dist + p.alignment_dist + p.cohesion_dist;
    let separation_thresh = p.separation_dist / zone_radius;
    let alignment_thresh = (p.separation_dist + p.alignment_dist) / zone_radius;
    let zone_radius_squared = zone_radius * zone_radius;

    // === Audio shockwave: vertical push from a low source plane, falling off
    // with height. Bottom-of-flock birds feel the kick; top barely does.
    // Event force — applied before the pre-steer snapshot so it bypasses the
    // turn-rate cap.
    if (p.shock_strength > 0.0) {
        let height_above = self_position.y - p.shock_source_y;
        let falloff = clamp(1.0 - height_above / p.shock_reach, 0.0, 1.0);
        velocity.y += falloff * p.shock_strength * 100.0 * dt;
        limit += 5.0 * p.shock_strength * falloff;
    }

    let pre_steer_velocity = velocity;

    // === Shape-center attraction. Y scaled 2.5x — this produces the oblate
    // "pancake" murmuration silhouette.
    var to_center = p.shape_center.xyz - self_position;
    to_center.y *= 2.5;
    let to_center_len = length(to_center);
    if (to_center_len > 0.0001) {
        velocity += (to_center / to_center_len) * dt * p.shape_center.w;
    }

    // === Drift / wind.
    velocity += p.drift.xyz * dt;

    // === Flocking over the spatial-hash grid: 3x3x3 cell stencil.
    let my_cell = cell_of(self_position, p.world_min.xyz, p.cell_size.xyz, p.grid_dim.xyz);
    let gx = u32(p.grid_dim.x);
    let gy = u32(p.grid_dim.y);
    let gdim = vec3<i32>(p.grid_dim.xyz);

    for (var dx: i32 = -1; dx <= 1; dx++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dz: i32 = -1; dz <= 1; dz++) {

                let n_cell = my_cell + vec3<i32>(dx, dy, dz);
                if (any(n_cell < vec3<i32>(0)) || any(n_cell >= gdim)) {
                    continue;
                }
                let cell_idx = u32(n_cell.x) + u32(n_cell.y) * gx + u32(n_cell.z) * gx * gy;
                let count = min(grid_count[cell_idx], p.grid_slots);

                for (var s: u32 = 0u; s < count; s++) {

                    let j = grid_cells[cell_idx * p.grid_slots + s];
                    if (j == i) {
                        continue;
                    }

                    let bird_position = pos_in[j].xyz;
                    let dir = bird_position - self_position;
                    let dist = length(dir);
                    if (dist < 0.0001) {
                        continue;
                    }
                    let dist_squared = dist * dist;
                    if (dist_squared > zone_radius_squared) {
                        continue;
                    }

                    let percent = dist_squared / zone_radius_squared;

                    if (percent < separation_thresh) {
                        // Separation — move apart for comfort.
                        let f = (separation_thresh / percent - 1.0) * dt;
                        var sep = (dir / dist) * f;
                        // Shear: extra separation along our own travel axis so the
                        // flock stretches into a filament (0 = isotropic, off).
                        if (p.shear > 0.0) {
                            let sv_len = length(self_velocity);
                            if (sv_len > 0.0001) {
                                let v_axis = self_velocity / sv_len;
                                let along = dot(dir / dist, v_axis);
                                sep += v_axis * (along * f * p.shear);
                            }
                        }
                        velocity -= sep;

                    } else if (percent < alignment_thresh) {
                        // Alignment — fly the same direction.
                        let thresh_delta = alignment_thresh - separation_thresh;
                        let adjusted = (percent - separation_thresh) / thresh_delta;
                        let bird_velocity = vel_in[j].xyz;
                        let bv_len = length(bird_velocity);
                        if (bv_len > 0.00001) {
                            let f = (0.5 - cos(adjusted * PI_2) * 0.5 + 0.5) * dt;
                            velocity += (bird_velocity / bv_len) * f;
                        }

                    } else {
                        // Cohesion — move closer.
                        let thresh_delta = 1.0 - alignment_thresh;
                        var adjusted = 1.0;
                        if (thresh_delta != 0.0) {
                            adjusted = (percent - alignment_thresh) / thresh_delta;
                        }
                        let f = (0.5 - (cos(adjusted * PI_2) * -0.5 + 0.5)) * dt;
                        velocity += (dir / dist) * f;
                    }
                }
            }
        }
    }

    // === Freedom / individuality — per-bird random perturbation. Y attenuated
    // to 30% so random kicks don't fight the oblate flattening. uv derived from
    // bird index the same way the texture build derived it, so the noise keeps
    // its character.
    {
        let uv = (vec2<f32>(f32(i % u32(p.tex_width)), f32(i / u32(p.tex_width))) + 0.5) / p.tex_width;
        let rx = rand(uv + vec2<f32>(p.time * 0.013, 1.7)) * 2.0 - 1.0;
        let ry = rand(uv + vec2<f32>(p.time * 0.017, 3.1)) * 2.0 - 1.0;
        let rz = rand(uv + vec2<f32>(p.time * 0.011, 5.3)) * 2.0 - 1.0;
        velocity += vec3<f32>(rx, ry * 0.3, rz) * dt * p.freedom;
    }

    // === Turn-rate limit. Clamp how far the heading rotated during steering;
    // speed (length) is preserved, only direction is constrained. Applied
    // BEFORE the safety forces so those can always pull a bird out of danger.
    {
        let pre_len = length(pre_steer_velocity);
        let post_len = length(velocity);
        if (pre_len > 0.0001 && post_len > 0.0001) {
            let old_dir = pre_steer_velocity / pre_len;
            let new_dir = velocity / post_len;
            let cos_a = clamp(dot(old_dir, new_dir), -1.0, 1.0);
            let angle = acos(cos_a);
            let max_angle = p.max_turn_rate * dt;
            if (angle > max_angle) {
                var perp = new_dir - old_dir * cos_a;
                let perp_len = length(perp);
                if (perp_len > 0.00001) {
                    perp /= perp_len;
                    let clamped_dir = old_dir * cos(max_angle) + perp * sin(max_angle);
                    velocity = clamped_dir * post_len;
                }
            }
        }
    }

    // === Ground avoidance — quadratic upward push near the floor.
    {
        let above_ground = self_position.y - p.ground_y;
        if (above_ground < p.ground_repel_height) {
            let t = 1.0 - max(above_ground, 0.0) / p.ground_repel_height;
            velocity.y += t * t * p.ground_repel_strength * dt;
        }
    }

    // === Bounding box — per-axis quadratic inward push near each face.
    {
        let from_min = self_position - p.box_min.xyz;
        let to_max = p.box_max.xyz - self_position;

        if (from_min.x < p.box_margin) {
            let t = 1.0 - max(from_min.x, 0.0) / p.box_margin;
            velocity.x += t * t * p.box_strength * dt;
        }
        if (to_max.x < p.box_margin) {
            let t = 1.0 - max(to_max.x, 0.0) / p.box_margin;
            velocity.x -= t * t * p.box_strength * dt;
        }
        if (from_min.y < p.box_margin) {
            let t = 1.0 - max(from_min.y, 0.0) / p.box_margin;
            velocity.y += t * t * p.box_strength * dt;
        }
        if (to_max.y < p.box_margin) {
            let t = 1.0 - max(to_max.y, 0.0) / p.box_margin;
            velocity.y -= t * t * p.box_strength * dt;
        }
        if (from_min.z < p.box_margin) {
            let t = 1.0 - max(from_min.z, 0.0) / p.box_margin;
            velocity.z += t * t * p.box_strength * dt;
        }
        if (to_max.z < p.box_margin) {
            let t = 1.0 - max(to_max.z, 0.0) / p.box_margin;
            velocity.z -= t * t * p.box_strength * dt;
        }
    }

    // === Camera safety bubble — last-resort repulsion so nothing clips the lens.
    {
        let from_camera = self_position - p.camera_pos.xyz;
        let dist_from_camera = length(from_camera);
        if (dist_from_camera < p.camera_pos.w && dist_from_camera > 0.0001) {
            let t = 1.0 - dist_from_camera / p.camera_pos.w;
            velocity += (from_camera / dist_from_camera) * t * t * p.camera_repel_strength * dt;
        }
    }

    // === Damping (pre-converted on CPU so the drag-per-second is identical at
    // any tick rate — fixes the framerate-dependent damping the WebGL build had).
    velocity *= p.damping_per_tick;

    // === Min speed — real birds can't idle; the flock stalls and looks dead
    // without a floor.
    let speed = length(velocity);
    if (speed < p.drift.w && speed > 0.0001) {
        velocity *= p.drift.w / speed;
    }

    // === Max speed.
    if (length(velocity) > limit) {
        velocity = normalize(velocity) * limit;
    }

    vel_out[i] = vec4<f32>(velocity, 0.0);
}


