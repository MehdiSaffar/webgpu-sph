const WK_SIZE: u32 = $[WK_SIZE];

@group(0) @binding(0) var<uniform> tonic: vec2u;
@group(0) @binding(1) var<storage, read_write> input: array<vec2i>;
var<workgroup> shared_data: array<vec2i, WK_SIZE>;

fn lt(a: vec2i, b: vec2i) -> bool {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

@compute @workgroup_size(WK_SIZE, 1, 1)
fn sort_all(
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(workgroup_id) group_id: vec3u,
) {
    let lid = local_id.x;
    let gid = global_id.x;
    shared_data[lid] = input[gid];

    workgroupBarrier();
    storageBarrier();

    let offset = group_id.x * WK_SIZE;
    for (var k = 2u; k <= WK_SIZE; k = k << 1u) {
        for (var j = k >> 1u; j > 0u; j = j >> 1u) {
            let ixj = (gid ^ j) - offset;
            if ixj > lid {
                if (gid & k) == 0u {
                    if lt(shared_data[ixj], shared_data[lid]) {
                        let tmp = shared_data[lid];
                        shared_data[lid] = shared_data[ixj];
                        shared_data[ixj] = tmp;
                    }
                } else {
                    if lt(shared_data[lid], shared_data[ixj]) {
                        let tmp = shared_data[lid];
                        shared_data[lid] = shared_data[ixj];
                        shared_data[ixj] = tmp;
                    }
                }
            }
            workgroupBarrier();
            storageBarrier();
        }
    }

    input[gid] = shared_data[lid];
}

@compute @workgroup_size(WK_SIZE, 1, 1)
fn sort_chunk(@builtin(global_invocation_id) global_id: vec3u) {
    let gid = global_id.x;
    let ixj = gid ^ tonic.y;

    if ixj > gid {
        if (gid & tonic.x) == 0u {
            if lt(input[ixj], input[gid]) {
                let tmp = input[gid];
                input[gid] = input[ixj];
                input[ixj] = tmp;
            }
        } else {
            if lt(input[gid], input[ixj]) {
                let tmp = input[gid];
                input[gid] = input[ixj];
                input[ixj] = tmp;
            }
        }
    }
}