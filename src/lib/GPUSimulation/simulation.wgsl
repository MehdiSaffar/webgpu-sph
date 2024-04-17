struct UBO {
    N: u32,
    SMOOTHING_RADIUS: f32,
    BASE_DENSITY: f32,
    NORMALIZATION_DENSITY: f32,
    NORMALIZATION_NEAR_DENSITY: f32,
    NORMALIZATION_VISCOUS_FORCE: f32,
    SCENE_SIZE: vec2f,
    DAMPING_COEFF: f32,
    GRAVITY: f32,
    DENSITY_TO_PRESSURE_FACTOR: f32,
    NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR: f32,
    TIME_STEP: f32,
    RADIUS: f32,

    INTERACTION_STRENGTH: f32,
    INTERACTION_POS: vec2f,

    DENSITY_KERNEL: u32,
};

const SPIKY = 0;
const SOFT = 1;

const cell_offsets: array<vec2i, 9> = array(
    vec2i(-1, -1),
    vec2i(-1, 0),
    vec2i(-1, 1),
    vec2i(0, -1),
    vec2i(0, 0),
    vec2i(0, 1),
    vec2i(1, -1),
    vec2i(1, 0),
    vec2i(1, 1),
);

@group(0) @binding(0) var<uniform> ubo : UBO;
@group(0) @binding(1) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> near_densities: array<f32>;
@group(0) @binding(5) var<storage, read_write> densities: array<f32>;
@group(0) @binding(6) var<storage, read_write> spatial_lookup: array<vec2u>;
@group(0) @binding(7) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(8) var<storage, read_write> ranges: array<f32>;
@group(0) @binding(9) var<storage, read_write> forces_mag: array<f32>;
@group(0) @binding(10) var<storage, read_write> velocities_mag: array<f32>;

fn get_cell_coords(pos: vec2f) -> vec2i {
    return vec2i(pos / ubo.SMOOTHING_RADIUS);
}

fn get_cell_hash(pos: vec2i) -> i32 {
    return pos.x * 15823 + pos.y * 9737333;
}

fn get_cell_key(hash: i32) -> u32 {
    return u32(hash % i32(ubo.N));
}

fn resolve_collisions(i: u32) {
    let top_cond = (positions[i].y + ubo.RADIUS) > ubo.SCENE_SIZE.y;
    let bottom_cond = (positions[i].y - ubo.RADIUS) < 0.0;
    let left_cond = (positions[i].x - ubo.RADIUS) < 0.0;
    let right_cond = (positions[i].x + ubo.RADIUS) > ubo.SCENE_SIZE.x;

    if left_cond {
        velocities[i].x = abs(velocities[i].x) * ubo.DAMPING_COEFF;
    }
    if right_cond {
        velocities[i].x = -abs(velocities[i].x) * ubo.DAMPING_COEFF;
    }
    if top_cond {
        velocities[i].y = -abs(velocities[i].y) * ubo.DAMPING_COEFF;
    }
    if bottom_cond {
        velocities[i].y = abs(velocities[i].y) * ubo.DAMPING_COEFF;
    }

    positions[i].x = clamp(positions[i].x, ubo.RADIUS, ubo.SCENE_SIZE.x - ubo.RADIUS);
    positions[i].y = clamp(positions[i].y, ubo.RADIUS, ubo.SCENE_SIZE.y - ubo.RADIUS);
}


// ------- KERNELS -------
// CORRECT only if 0 <= dist <= ubo.SMOOTHING_RADIUS
fn density_kernel(dist: f32) -> f32 {
    if ubo.DENSITY_KERNEL == SPIKY {
        return ubo.NORMALIZATION_DENSITY * pow(ubo.SMOOTHING_RADIUS - dist, 2);
    } else {
        return ubo.NORMALIZATION_DENSITY * pow(pow(ubo.SMOOTHING_RADIUS, 2) - pow(dist, 2), 3);
    }
}

fn density_kernel_derivative(dist: f32) -> f32 {
    if ubo.DENSITY_KERNEL == SPIKY {
        return -2.0 * ubo.NORMALIZATION_DENSITY * (ubo.SMOOTHING_RADIUS - dist);
    } else {
        return -6.0 * dist * ubo.NORMALIZATION_DENSITY * pow(pow(ubo.SMOOTHING_RADIUS, 2) - pow(dist, 2), 2);
    }
}

fn near_density_kernel(dist: f32) -> f32 {
    if ubo.DENSITY_KERNEL == SPIKY {
        return ubo.NORMALIZATION_NEAR_DENSITY * pow(ubo.SMOOTHING_RADIUS - dist, 3);
    } else {
        return ubo.NORMALIZATION_NEAR_DENSITY * pow(pow(ubo.SMOOTHING_RADIUS, 2) - pow(dist, 2), 4) ;
    }
}

fn near_density_kernel_derivative(dist: f32) -> f32 {
    if ubo.DENSITY_KERNEL == SPIKY {
        return -3.0 * ubo.NORMALIZATION_NEAR_DENSITY * pow(ubo.SMOOTHING_RADIUS - dist, 2);
    } else {
        return -8.0 * dist * ubo.NORMALIZATION_NEAR_DENSITY * pow(pow(ubo.SMOOTHING_RADIUS, 2) - pow(dist, 2), 3) ;
    }
}

fn viscosity_kernel(dist: f32) -> f32 {
    return ubo.NORMALIZATION_VISCOUS_FORCE * pow(pow(ubo.SMOOTHING_RADIUS, 2) - pow(dist, 2), 3);
}

// ------- CONVERTERS ------- 

fn density_to_pressure(density: f32) -> f32 {
    return ubo.DENSITY_TO_PRESSURE_FACTOR * (density - ubo.BASE_DENSITY);
}

fn near_density_to_near_pressure(near_density: f32) -> f32 {
    return ubo.NEAR_DENSITY_TO_NEAR_PRESSURE_FACTOR * near_density;
}

fn compute_densities_dumb(i: u32) -> f32 {
    var density = 0.0;

    for (var j = 0u; j < ubo.N; j++) {
        let dist = distance(predicted_positions[i], predicted_positions[j]);
        if dist <= ubo.SMOOTHING_RADIUS {
            density += density_kernel(dist);
        }
    }

    return density;
}

fn compute_densities_smart(i: u32) -> vec2f {
    var density = 0.0;
    var near_density = 0.0;

    let cell_pos = get_cell_coords(predicted_positions[i]);
    let cell_key = get_cell_key(get_cell_hash(cell_pos));
    for (var offset_idx = 0u; offset_idx < 9u; offset_idx++) {
        let next_cell_pos = cell_pos + cell_offsets[offset_idx];
        let next_cell_key = get_cell_key(get_cell_hash(next_cell_pos));

        if start_indices[next_cell_key] == ubo.N {
            continue;
        }

        let start_idx = start_indices[next_cell_key];
        for (var k = u32(start_idx); k < ubo.N; k++) {
            let sk_cell_key = spatial_lookup[k].x;
            let j = spatial_lookup[k].y;
            if sk_cell_key != next_cell_key {
                break;
            }

            let dist = distance(predicted_positions[i], predicted_positions[j]);
            if dist <= ubo.SMOOTHING_RADIUS {
                density += density_kernel(dist);
                near_density += near_density_kernel(dist);
            }
        }
    }

    return vec2f(density, near_density);
}

fn compute_force(i: u32, j: u32, dist: f32) -> vec2f {
    let direction = (predicted_positions[j] - predicted_positions[i]) / dist;

    let shared_pressure = (density_to_pressure(densities[i]) + density_to_pressure(densities[j])) / 2.0;
    let pressure_slope = density_kernel_derivative(dist);
    let pressure_force = (direction * pressure_slope * shared_pressure) / densities[j];

    let shared_near_pressure = (near_density_to_near_pressure(near_densities[i]) + near_density_to_near_pressure(near_densities[j])) / 2.0;
    let near_pressure_slope = near_density_kernel_derivative(dist);
    let near_pressure_force = (direction * near_pressure_slope * shared_near_pressure) / near_densities[j];

    let viscosity_force = (velocities[j] - velocities[i]) * viscosity_kernel(dist) / densities[j];

    return near_pressure_force + pressure_force + viscosity_force;
}

fn compute_forces_dumb(i: u32) -> vec2f {
    var forces = vec2f(0.0, -ubo.GRAVITY * densities[i]);

    for (var j = 0u; j < ubo.N; j++) {
        let dist = distance(predicted_positions[i], predicted_positions[j]);
        if 0 < dist && dist <= ubo.SMOOTHING_RADIUS {
            forces += compute_force(i, j, dist);
        }
    }

    return forces;
}

fn compute_interaction_force(i: u32) -> vec2f {
    if ubo.INTERACTION_STRENGTH == 0.0 {
        return vec2f(0.0);
    }

    let dist = distance(predicted_positions[i], ubo.INTERACTION_POS);
    if dist > 10 {
        return vec2f(0.0);
    }

    let direction = (ubo.INTERACTION_POS - predicted_positions[i]) / dist;

    return direction * ubo.INTERACTION_STRENGTH;
}

fn compute_forces_smart(i: u32) -> vec2f {
    var forces = vec2f(0.0, -ubo.GRAVITY * densities[i]);

    let cell_pos = get_cell_coords(predicted_positions[i]);
    let cell_key = get_cell_key(get_cell_hash(cell_pos));
    for (var offset_idx = 0u; offset_idx < 9u; offset_idx++) {
        let next_cell_key = get_cell_key(get_cell_hash(cell_pos + cell_offsets[offset_idx]));

        if start_indices[next_cell_key] == ubo.N {
            continue;
        }

        for (var k = u32(start_indices[next_cell_key]); k < ubo.N; k++) {
            let sk_cell_key = spatial_lookup[k].x;
            let j = u32(spatial_lookup[k].y);
            if sk_cell_key != next_cell_key {
                break;
            }

            let dist = distance(predicted_positions[i], predicted_positions[j]);
            if 0 < dist && dist <= ubo.SMOOTHING_RADIUS {
                forces += compute_force(i, j, dist);
            }
        }
    }

    return forces;
}

// ------- COMPUTE KERNELS ------- 

const SIZE = 256;

@compute @workgroup_size(SIZE)
fn spatial_lookup_pass_one(
    @builtin(global_invocation_id) gid: vec3u
) {
    let i = gid.x;

    let cell_coords = get_cell_coords(predicted_positions[i]);
    let cell_key = get_cell_key(get_cell_hash(cell_coords));

    spatial_lookup[i] = vec2u(cell_key, i);
    start_indices[i] = ubo.N;
}

@compute @workgroup_size(SIZE)
fn spatial_lookup_pass_two(
    @builtin(global_invocation_id) gid: vec3u
) {
    let i = gid.x;

    let cell_key = spatial_lookup[i].x;
    if (i == 0) || (cell_key != spatial_lookup[i - 1].x) {
        start_indices[cell_key] = i;
    }
}

@compute @workgroup_size(SIZE)
fn compute_densities(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;

    let ds = compute_densities_smart(i);
    // let densities = compute_densities_dumb(i)

    densities[i] = ds.x;
    near_densities[i] = ds.y;
}

@compute @workgroup_size(SIZE)
fn compute_forces(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;

    var forces = compute_forces_smart(i);
    forces += compute_interaction_force(i);
    // forces[i] = compute_forces_dumb(i);
    forces_mag[i] = length(forces);

    velocities[i] += (forces / densities[i]) * ubo.TIME_STEP;
    velocities_mag[i] = length(velocities[i]);

    positions[i] += velocities[i] * ubo.TIME_STEP;


    resolve_collisions(i);
}

@compute @workgroup_size(SIZE)
fn predict_positions(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;

    predicted_positions[i] = positions[i] + velocities[i] * ubo.TIME_STEP;
}