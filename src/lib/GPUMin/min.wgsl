const WK_SIZE: u32 = $[WK_SIZE];
const POS_INF = 3.40282e+38;
const NEG_INF = -3.40282e+38;

struct UBO {
    size: u32,
    start: u32,
    end: u32,
}

@group(0) @binding(0) var<uniform> ubo: UBO;
@group(0) @binding(1) var<storage, read_write> input: array<f32>;


@compute @workgroup_size(WK_SIZE)
fn init_min(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x;
    if i < ubo.start || i >= ubo.end {
        input[i] = POS_INF;
    }
}

@compute @workgroup_size(WK_SIZE)
fn init_max(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x;
    if i < ubo.start || i >= ubo.end {
        input[i] = NEG_INF;
    }
}

@compute @workgroup_size(WK_SIZE)
fn compute_min(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x * ubo.size;
    input[i] = min(input[i], input[i + ubo.size / 2]);
}

@compute @workgroup_size(WK_SIZE)
fn compute_max(@builtin(global_invocation_id) global_id: vec3u) {
    let i = global_id.x * ubo.size;
    input[i] = max(input[i], input[i + ubo.size / 2]);
}
