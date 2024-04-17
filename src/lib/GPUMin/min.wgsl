
@group(0) @binding(0) var<uniform> size: u32;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;

@compute @workgroup_size(1)
fn compute_min(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x * size;
    input[i] = min(input[i], input[i + size/2]);
}

@compute @workgroup_size(1)
fn compute_max(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x * size;
    input[i] = max(input[i], input[i + size/2]);
}
