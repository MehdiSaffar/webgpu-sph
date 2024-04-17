const DENSITY = 0;
const PRESSURE = 1;
const FORCE = 2;
const VELOCITY = 3;

struct RenderUBO {
    SCENE_SIZE: vec2f,
    VIEWPORT_SIZE: vec2f,
    SELECTED_PROPERTY: u32,
    MIN_COLOR: vec3f,
    MAX_COLOR: vec3f,
    RADIUS: f32,
};

@group(0) @binding(0) var<uniform> ubo : RenderUBO;
@group(0) @binding(1) var<storage> ranges : array<f32>;
@group(0) @binding(2) var<storage, read> densities: array<f32>;
@group(0) @binding(3) var<storage, read> pressures: array<f32>;
@group(0) @binding(4) var<storage, read> forces_mag: array<f32>;
@group(0) @binding(5) var<storage, read> velocities_mag: array<f32>;

struct InVertex {
  @location(0) pos: vec2f,
};

struct OutVertex {
    @builtin(position) pos: vec4f,
    @location(0) center_pos: vec4f,
    @location(1) uv: vec2f,
    @location(2) center_uv: vec2f,
    @location(3) value: f32,
}

const quad_points = array(
    vec2f(-1, -1),
    vec2f(1, -1),
    vec2f(-1, 1),
    vec2f(-1, 1),
    vec2f(1, -1),
    vec2f(1, 1),
);
 
@vertex 
fn vs(iv: InVertex, @builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> OutVertex {
    var ov: OutVertex;

    let scale = ubo.VIEWPORT_SIZE.x / ubo.SCENE_SIZE.x;

    var pos = (iv.pos + quad_points[vi] * ubo.RADIUS) * scale;
    pos = pos / ubo.VIEWPORT_SIZE * 2 - 1;
    pos.y = -pos.y;
    ov.pos = vec4f(pos, 0, 1);

    var center_pos = iv.pos * scale;
    center_pos = center_pos / ubo.VIEWPORT_SIZE * 2 - 1;
    center_pos.y = -center_pos.y;
    ov.center_pos = vec4f(center_pos, 0, 1);

    ov.uv = quad_points[vi] * 0.5 + 0.5;
    ov.center_uv = vec2f(0.5, 0.5);

    var value = 0.0;

    if ubo.SELECTED_PROPERTY == DENSITY {
        value = densities[ii];
    } else if ubo.SELECTED_PROPERTY == PRESSURE {
        value = pressures[ii];
    } else if ubo.SELECTED_PROPERTY == FORCE {
        value = forces_mag[ii];
    } else if ubo.SELECTED_PROPERTY == VELOCITY {
        value = velocities_mag[ii];
    }

    let min_value = ranges[ubo.SELECTED_PROPERTY * 2];
    let max_value = ranges[ubo.SELECTED_PROPERTY * 2+1];
    ov.value = (value - min_value) / (max_value - min_value);
    return ov;
}
 
@fragment
fn fs(ov: OutVertex) -> @location(0) vec4f {
    let dist = distance(ov.uv, ov.center_uv);
    if dist > 0.5 {
        discard;
    }

    // let alpha = 1.0 - pow(dist / 0.5, 1);
    let alpha = 1.0;

    return mix(vec4f(ubo.MIN_COLOR, alpha), vec4f(ubo.MAX_COLOR, alpha), ov.value);
}
