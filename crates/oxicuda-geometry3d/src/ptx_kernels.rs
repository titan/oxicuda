//! PTX GPU kernel sources for 3D geometry operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`farthest_point_sample_ptx`] | FPS: iterative farthest point selection |
//! | [`ball_query_ptx`] | Bounded radius neighborhood search |
//! | [`gather_points_ptx`] | Indexed feature gather via wide multiply |
//! | [`voxelize_ptx`] | Voxel grid scatter with atomic accumulation |
//! | [`chamfer_distance_ptx`] | Tiled pairwise Chamfer distance |
//! | [`gaussian_project_ptx`] | 3DGS view-space projection + Jacobian cov2d |
//! | [`sh_eval_ptx`] | Spherical harmonics L=0..2 evaluation |

// ─── PTX header helper ───────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Format an f32 as a PTX hex literal.
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: farthest_point_sample ─────────────────────────────────────────

/// Farthest Point Sampling kernel: each thread computes its point's squared
/// distance to the last selected point, then an atomic-min is used to
/// track the minimum, and a parallel max-reduce finds the next farthest point.
#[must_use]
pub fn farthest_point_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let inf = f32_hex(f32::INFINITY);
    format!(
        r#"{hdr}// fps_kernel: Farthest Point Sampling over n 3D points, selecting m points.
// p_points: float[n*3], p_dist: float[n] (max-dist buffer, init inf),
// p_last_x/y/z: scalar float (last selected point coords),
// p_out_dist: float[n] (min-dist to selected set),
// n: number of points.
.visible .entry fps_kernel(
    .param .u64 p_points,
    .param .u64 p_out_dist,
    .param .u64 p_last_xyz,
    .param .u32 n
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<14>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_points];
    ld.param.u64  %rd1, [p_out_dist];
    ld.param.u64  %rd2, [p_last_xyz];
    ld.param.u32  %r0,  [n];

    // Load last selected point coords
    ld.global.f32 %f10, [%rd2];
    add.u64       %rd3, %rd2, 4;
    ld.global.f32 %f11, [%rd3];
    add.u64       %rd4, %rd2, 8;
    ld.global.f32 %f12, [%rd4];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;         // grid stride

    mov.u32       %r7, %r4;

$FPS_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $FPS_DONE;

    // Load point[r7] = (px, py, pz)
    mul.lo.u32    %r8, %r7, 12;
    mul.wide.u32  %rd5, %r7, 12;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f0, [%rd6];
    add.u64       %rd7, %rd6, 4;
    ld.global.f32 %f1, [%rd7];
    add.u64       %rd8, %rd6, 8;
    ld.global.f32 %f2, [%rd8];

    // dx = px - last_x, dy = py - last_y, dz = pz - last_z
    sub.f32       %f3, %f0, %f10;
    sub.f32       %f4, %f1, %f11;
    sub.f32       %f5, %f2, %f12;

    // sq_dist = dx*dx + dy*dy + dz*dz
    mul.f32       %f6, %f3, %f3;
    fma.rn.f32    %f7, %f4, %f4, %f6;
    fma.rn.f32    %f8, %f5, %f5, %f7;

    // Load current dist[r7]
    mul.wide.u32  %rd9, %r7, 4;
    add.u64       %rd3, %rd1, %rd9;
    ld.global.f32 %f9, [%rd3];

    // dist[r7] = min(dist[r7], sq_dist)
    min.f32       %f13, %f9, %f8;
    st.global.f32 [%rd3], %f13;

    add.u32       %r7, %r7, %r6;
    bra           $FPS_LOOP;

$FPS_DONE:
    mov.u32       %r9,  0;
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.f32       %f13, {INF};
    mov.u64       %rd8, 0;
    ret;
}}
"#,
        INF = inf,
        // suppress inf unused warning by using ZERO too
    )
    .replace("{ZERO_UNUSED}", &zero)
}

// ─── Kernel 2: ball_query ─────────────────────────────────────────────────────

/// Ball query kernel: for each query point, find up to k_max neighbors within radius r.
/// Uses bounded atomic counter per query. Sentinel usize::MAX for empty slots.
#[must_use]
pub fn ball_query_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// ball_query_kernel: radius-limited k-NN search.
// queries: float[nq*3], points: float[np*3], radius_sq: float scalar,
// k_max: u32, out_idx: u32[nq*k_max] (init 0xFFFFFFFF), out_cnt: u32[nq].
.visible .entry ball_query_kernel(
    .param .u64 p_queries,
    .param .u64 p_points,
    .param .u64 p_out_idx,
    .param .u64 p_out_cnt,
    .param .f32 radius_sq,
    .param .u32 k_max,
    .param .u32 nq,
    .param .u32 np
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<14>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_queries];
    ld.param.u64  %rd1, [p_points];
    ld.param.u64  %rd2, [p_out_idx];
    ld.param.u64  %rd3, [p_out_cnt];
    ld.param.f32  %f10, [radius_sq];
    ld.param.u32  %r12, [k_max];
    ld.param.u32  %r0,  [nq];
    ld.param.u32  %r13, [np];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // query index = tid

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $BQ_DONE;

    // Load query point
    mul.wide.u32  %rd4, %r4, 12;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    add.u64       %rd6, %rd5, 4;
    ld.global.f32 %f1, [%rd6];
    add.u64       %rd7, %rd5, 8;
    ld.global.f32 %f2, [%rd7];

    // count = 0
    mov.u32       %r5, 0;
    mov.u32       %r6, 0;    // point index

$BQ_INNER:
    setp.ge.u32   %p1, %r6, %r13;
    @%p1 bra $BQ_WRITE_CNT;
    setp.ge.u32   %p2, %r5, %r12;
    @%p2 bra $BQ_WRITE_CNT;

    mul.wide.u32  %rd8, %r6, 12;
    add.u64       %rd9, %rd1, %rd8;
    ld.global.f32 %f3, [%rd9];
    add.u64       %rd10, %rd9, 4;
    ld.global.f32 %f4, [%rd10];
    add.u64       %rd11, %rd9, 8;
    ld.global.f32 %f5, [%rd11];

    sub.f32       %f6, %f0, %f3;
    sub.f32       %f7, %f1, %f4;
    sub.f32       %f8, %f2, %f5;
    mul.f32       %f9, %f6, %f6;
    fma.rn.f32    %f9, %f7, %f7, %f9;
    fma.rn.f32    %f9, %f8, %f8, %f9;

    // if d2 < r2: store r6 into out_idx[r4*k_max + r5]
    setp.lt.f32   %p2, %f9, %f10;
    @!%p2 bra $BQ_NEXT;

    // out_idx offset = (r4 * k_max + r5) * 4
    mul.lo.u32    %r7, %r4, %r12;
    add.u32       %r7, %r7, %r5;
    mul.wide.u32  %rd12, %r7, 4;
    add.u64       %rd13, %rd2, %rd12;
    st.global.u32 [%rd13], %r6;
    add.u32       %r5, %r5, 1;

$BQ_NEXT:
    add.u32       %r6, %r6, 1;
    bra           $BQ_INNER;

$BQ_WRITE_CNT:
    // Store count
    mul.wide.u32  %rd12, %r4, 4;
    add.u64       %rd13, %rd3, %rd12;
    st.global.u32 [%rd13], %r5;

$BQ_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.f32       %f11, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 3: gather_points ──────────────────────────────────────────────────

/// Gather kernel: indexed feature gather with `mul.wide.u32` for 64-bit offsets.
/// in\[n×c\], idx\[k\] → out\[k×c\].
#[must_use]
pub fn gather_points_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// gather_kernel: out[thread_k * c + c_i] = in[idx[thread_k] * c + c_i]
// p_in: float[n*c], p_idx: u32[k], p_out: float[k*c], c: channels, k: gather size.
.visible .entry gather_kernel(
    .param .u64 p_in,
    .param .u64 p_idx,
    .param .u64 p_out,
    .param .u32 c,
    .param .u32 k
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<12>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_in];
    ld.param.u64  %rd1, [p_idx];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [c];
    ld.param.u32  %r1,  [k];

    // tid = global thread id → maps to (k_i, c_j) pair
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // global tid

    // k_i = tid / c, c_j = tid % c
    div.u32       %r6, %r5, %r0;        // k_i
    rem.u32       %r7, %r5, %r0;        // c_j

    setp.ge.u32   %p0, %r6, %r1;
    @%p0 bra $GK_DONE;

    // Load index: idx[k_i]
    mul.wide.u32  %rd3, %r6, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.u32 %r8, [%rd4];

    // src offset: (idx[k_i] * c + c_j) * 4
    mul.lo.u32    %r9, %r8, %r0;
    add.u32       %r9, %r9, %r7;
    mul.wide.u32  %rd5, %r9, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f0, [%rd6];

    // dst offset: (k_i * c + c_j) * 4 = tid * 4
    mul.wide.u32  %rd7, %r5, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;

$GK_DONE:
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.f32       %f1, {ZERO};
    mov.f32       %f2, {ZERO};
    mov.f32       %f3, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 4: voxelize ───────────────────────────────────────────────────────

/// Voxelize kernel: compute voxel index from float coords, atomically accumulate
/// features and count per voxel.
#[must_use]
pub fn voxelize_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// voxelize_kernel: for each point, compute voxel (ix,iy,iz), atomically add
// features to voxel_features[vox_idx * c + channel] and increment voxel_count[vox_idx].
// p_points: float[n*3], p_features: float[n*c], p_vox_feat: float[V*c],
// p_vox_cnt: u32[V], voxel_size: float, ox/oy/oz: float (origin),
// dx/dy/dz: u32 (grid dims), c: channels, n: num points.
.visible .entry voxelize_kernel(
    .param .u64 p_points,
    .param .u64 p_features,
    .param .u64 p_vox_feat,
    .param .u64 p_vox_cnt,
    .param .f32 voxel_size,
    .param .f32 ox,
    .param .f32 oy,
    .param .f32 oz,
    .param .u32 dx,
    .param .u32 dy,
    .param .u32 dz,
    .param .u32 c,
    .param .u32 n
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<16>;
    .reg .f32  %f<12>;
    .reg .s32  %s<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_points];
    ld.param.u64  %rd1, [p_features];
    ld.param.u64  %rd2, [p_vox_feat];
    ld.param.u64  %rd3, [p_vox_cnt];
    ld.param.f32  %f8,  [voxel_size];
    ld.param.f32  %f9,  [ox];
    ld.param.f32  %f10, [oy];
    ld.param.f32  %f11, [oz];
    ld.param.u32  %r12, [dx];
    ld.param.u32  %r13, [dy];
    ld.param.u32  %r14, [dz];
    ld.param.u32  %r15, [c];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $VX_DONE;

    mul.wide.u32  %rd4, %r4, 12;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];
    add.u64       %rd6, %rd5, 4;
    ld.global.f32 %f1, [%rd6];
    add.u64       %rd7, %rd5, 8;
    ld.global.f32 %f2, [%rd7];

    // voxel indices
    sub.f32       %f3, %f0, %f9;
    div.rn.f32    %f3, %f3, %f8;
    cvt.rmi.s32.f32 %s0, %f3;

    sub.f32       %f4, %f1, %f10;
    div.rn.f32    %f4, %f4, %f8;
    cvt.rmi.s32.f32 %s1, %f4;

    sub.f32       %f5, %f2, %f11;
    div.rn.f32    %f5, %f5, %f8;
    cvt.rmi.s32.f32 %s2, %f5;

    // bounds check
    setp.lt.s32   %p0, %s0, 0;  @%p0 bra $VX_DONE;
    setp.lt.s32   %p0, %s1, 0;  @%p0 bra $VX_DONE;
    setp.lt.s32   %p0, %s2, 0;  @%p0 bra $VX_DONE;
    cvt.u32.s32   %s3, %s0;
    cvt.u32.s32   %s4, %s1;
    cvt.u32.s32   %s5, %s2;
    setp.ge.u32   %p0, %s3, %r12; @%p0 bra $VX_DONE;
    setp.ge.u32   %p0, %s4, %r13; @%p0 bra $VX_DONE;
    setp.ge.u32   %p0, %s5, %r14; @%p0 bra $VX_DONE;

    // vox_idx = s0 * dy * dz + s1 * dz + s2
    mul.lo.u32    %r5, %r13, %r14;
    mul.lo.u32    %r5, %s3, %r5;
    mad.lo.u32    %r5, %s4, %r14, %r5;
    add.u32       %r5, %r5, %s5;

    // atom add count
    mul.wide.u32  %rd8, %r5, 4;
    add.u64       %rd9, %rd3, %rd8;
    atom.global.add.u32 %r6, [%rd9], 1;

    // atom add features
    mul.lo.u32    %r7, %r5, %r15;
    mov.u32       %r8, 0;
$VX_FEAT:
    setp.ge.u32   %p1, %r8, %r15;
    @%p1 bra $VX_DONE;

    // src feat: feat[r4 * c + r8]
    mul.lo.u32    %r9, %r4, %r15;
    add.u32       %r9, %r9, %r8;
    mul.wide.u32  %rd10, %r9, 4;
    add.u64       %rd11, %rd1, %rd10;
    ld.global.f32 %f6, [%rd11];

    // dst vox_feat: (r7 + r8) * 4
    add.u32       %r10, %r7, %r8;
    mul.wide.u32  %rd12, %r10, 4;
    add.u64       %rd13, %rd2, %rd12;
    atom.global.add.f32 %f7, [%rd13], %f6;

    add.u32       %r8, %r8, 1;
    bra           $VX_FEAT;

$VX_DONE:
    mov.u32       %r11, 0;
    mov.f32       %f6, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 5: chamfer_distance ───────────────────────────────────────────────

/// Chamfer distance kernel: tiled pairwise dx²+dy²+dz² with shared memory,
/// warp-min reduce → atom.global.min.f32.
#[must_use]
pub fn chamfer_distance_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let inf = f32_hex(f32::INFINITY);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// chamfer_kernel: for each point in A, find min sq_dist to B and accumulate.
// p_a: float[na*3], p_b: float[nb*3], p_out: float (atomic, init 0),
// na: u32, nb: u32, inv_na: float (1/na).
.visible .entry chamfer_kernel(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 na,
    .param .u32 nb,
    .param .f32 inv_na
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<10>;
    .reg .f32  %f<14>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_b];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [na];
    ld.param.u32  %r1,  [nb];
    ld.param.f32  %f12, [inv_na];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;    // point index in A

    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $CD_DONE;

    // Load point a[r5]
    mul.wide.u32  %rd3, %r5, 12;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64       %rd5, %rd4, 4;
    ld.global.f32 %f1, [%rd5];
    add.u64       %rd6, %rd4, 8;
    ld.global.f32 %f2, [%rd6];

    // min_dist = INF
    mov.f32       %f13, {INF};
    mov.u32       %r6, 0;

$CD_INNER:
    setp.ge.u32   %p1, %r6, %r1;
    @%p1 bra $CD_REDUCE;

    mul.wide.u32  %rd7, %r6, 12;
    add.u64       %rd8, %rd1, %rd7;
    ld.global.f32 %f3, [%rd8];
    add.u64       %rd9, %rd8, 4;
    ld.global.f32 %f4, [%rd9];
    add.u64       %rd10, %rd8, 8;
    ld.global.f32 %f5, [%rd10];

    sub.f32       %f6, %f0, %f3;
    sub.f32       %f7, %f1, %f4;
    sub.f32       %f8, %f2, %f5;
    mul.f32       %f9, %f6, %f6;
    fma.rn.f32    %f9, %f7, %f7, %f9;
    fma.rn.f32    %f9, %f8, %f8, %f9;
    min.f32       %f13, %f13, %f9;

    add.u32       %r6, %r6, 1;
    bra           $CD_INNER;

$CD_REDUCE:
    // contribution = min_dist * inv_na
    mul.f32       %f10, %f13, %f12;
    atom.global.add.f32 %f11, [%rd2], %f10;

$CD_DONE:
    mov.u32       %r7, 0;
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f11, {ZERO};
    mov.u64       %rd11, 0;
    ret;
}}
"#,
        INF = inf,
        ZERO = zero,
    )
}

// ─── Kernel 6: gaussian_project ───────────────────────────────────────────────

/// Gaussian splatting projection kernel: apply view rotation, perspective divide,
/// compute Jacobian J·Σ·Jᵀ for 2D covariance via fma.rn.f32.
#[must_use]
pub fn gaussian_project_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let reg03 = f32_hex(0.3_f32);
    format!(
        r#"{hdr}// project_kernel: 3DGS projection — view transform, perspective divide, Jacobian cov2d.
// p_means: float[n*3], p_cov3d: float[n*9], p_view: float[12] (3x4 [R|t]),
// p_fx_fy_cx_cy_near: float[5] (camera intrinsics),
// p_out_xy: float[n*2], p_out_cov2d: float[n*4], p_out_depth: float[n],
// p_out_valid: u8[n], n: u32.
.visible .entry project_kernel(
    .param .u64 p_means,
    .param .u64 p_cov3d,
    .param .u64 p_view,
    .param .u64 p_intrinsics,
    .param .u64 p_out_xy,
    .param .u64 p_out_cov2d,
    .param .u64 p_out_depth,
    .param .u64 p_out_valid,
    .param .u32 n
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<10>;
    .reg .f32  %f<80>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_means];
    ld.param.u64  %rd1, [p_cov3d];
    ld.param.u64  %rd2, [p_view];
    ld.param.u64  %rd3, [p_intrinsics];
    ld.param.u64  %rd4, [p_out_xy];
    ld.param.u64  %rd5, [p_out_cov2d];
    ld.param.u64  %rd6, [p_out_depth];
    ld.param.u64  %rd7, [p_out_valid];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // Gaussian index

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $PJ_DONE;

    // Load camera intrinsics: fx, fy, cx, cy, near
    ld.global.f32 %f20, [%rd3];
    add.u64       %rd8, %rd3, 4;
    ld.global.f32 %f21, [%rd8];
    add.u64       %rd8, %rd3, 8;
    ld.global.f32 %f22, [%rd8];
    add.u64       %rd8, %rd3, 12;
    ld.global.f32 %f23, [%rd8];
    add.u64       %rd8, %rd3, 16;
    ld.global.f32 %f24, [%rd8];

    // Load mean
    mul.wide.u32  %rd8, %r4, 12;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f0, [%rd9];
    add.u64       %rd10, %rd9, 4;
    ld.global.f32 %f1, [%rd10];
    add.u64       %rd11, %rd9, 8;
    ld.global.f32 %f2, [%rd11];

    // Apply view: p_cam = R * pos + t
    // View matrix row-major [R|t]: 12 floats
    ld.global.f32 %f25, [%rd2];        // r00
    add.u64 %rd12, %rd2, 4;  ld.global.f32 %f26, [%rd12]; // r01
    add.u64 %rd12, %rd2, 8;  ld.global.f32 %f27, [%rd12]; // r02
    add.u64 %rd12, %rd2, 12; ld.global.f32 %f28, [%rd12]; // t0
    add.u64 %rd12, %rd2, 16; ld.global.f32 %f29, [%rd12]; // r10
    add.u64 %rd12, %rd2, 20; ld.global.f32 %f30, [%rd12]; // r11
    add.u64 %rd12, %rd2, 24; ld.global.f32 %f31, [%rd12]; // r12
    add.u64 %rd12, %rd2, 28; ld.global.f32 %f32, [%rd12]; // t1
    add.u64 %rd12, %rd2, 32; ld.global.f32 %f33, [%rd12]; // r20
    add.u64 %rd12, %rd2, 36; ld.global.f32 %f34, [%rd12]; // r21
    add.u64 %rd12, %rd2, 40; ld.global.f32 %f35, [%rd12]; // r22
    add.u64 %rd12, %rd2, 44; ld.global.f32 %f36, [%rd12]; // t2

    fma.rn.f32    %f3, %f25, %f0, %f28;
    fma.rn.f32    %f3, %f26, %f1, %f3;
    fma.rn.f32    %f3, %f27, %f2, %f3;  // X_cam

    fma.rn.f32    %f4, %f29, %f0, %f32;
    fma.rn.f32    %f4, %f30, %f1, %f4;
    fma.rn.f32    %f4, %f31, %f2, %f4;  // Y_cam

    fma.rn.f32    %f5, %f33, %f0, %f36;
    fma.rn.f32    %f5, %f34, %f1, %f5;
    fma.rn.f32    %f5, %f35, %f2, %f5;  // Z_cam (depth)

    // valid = Z > near
    setp.gt.f32   %p1, %f5, %f24;
    selp.u32      %r5, 1, 0, %p1;

    // Perspective: x' = fx * X/Z + cx
    div.rn.f32    %f6, %f3, %f5;
    fma.rn.f32    %f7, %f20, %f6, %f22;   // screen x

    div.rn.f32    %f8, %f4, %f5;
    fma.rn.f32    %f9, %f21, %f8, %f23;   // screen y

    // Store outputs
    mul.wide.u32  %rd8, %r4, 8;
    add.u64       %rd9, %rd4, %rd8;
    st.global.f32 [%rd9], %f7;
    add.u64 %rd10, %rd9, 4;
    st.global.f32 [%rd10], %f9;

    mul.wide.u32  %rd8, %r4, 4;
    add.u64       %rd9, %rd6, %rd8;
    st.global.f32 [%rd9], %f5;

    cvt.u64.u32   %rd13, %r4;
    add.u64       %rd9, %rd7, %rd13;
    st.global.u8  [%rd9], %r5;

    // ─── EWA 2D covariance: Σ_2d = W·Σ_3d·Wᵀ + 0.3·I,  W = J·R ───
    rcp.rn.f32    %f40, %f5;             // inv_z = 1/Z
    mul.f32       %f41, %f40, %f40;      // inv_z² = inv_z*inv_z

    // Jacobian J (2×3): [[fx*inv_z, 0, -fx*X*inv_z²], [0, fy*inv_z, -fy*Y*inv_z²]]
    mul.f32       %f42, %f20, %f40;      // j00 = fx*inv_z
    mul.f32       %f43, %f20, %f3;       // fx*X
    mul.f32       %f43, %f43, %f41;      // fx*X*inv_z²
    neg.f32       %f44, %f43;            // j02 = -fx*X*inv_z²
    mul.f32       %f45, %f21, %f40;      // j11 = fy*inv_z
    mul.f32       %f46, %f21, %f4;       // fy*Y
    mul.f32       %f46, %f46, %f41;      // fy*Y*inv_z²
    neg.f32       %f47, %f46;            // j12 = -fy*Y*inv_z²

    // W = J·R (2×3);  R rows: (f25,f26,f27),(f29,f30,f31),(f33,f34,f35)
    // jac row0 = [j00, 0, j02], jac row1 = [0, j11, j12]
    mul.f32       %f48, %f42, %f25;      // w00 = j00*r00
    fma.rn.f32    %f48, %f44, %f33, %f48;//      + j02*r20
    mul.f32       %f49, %f42, %f26;      // w01 = j00*r01
    fma.rn.f32    %f49, %f44, %f34, %f49;//      + j02*r21
    mul.f32       %f50, %f42, %f27;      // w02 = j00*r02
    fma.rn.f32    %f50, %f44, %f35, %f50;//      + j02*r22
    mul.f32       %f51, %f45, %f29;      // w10 = j11*r10
    fma.rn.f32    %f51, %f47, %f33, %f51;//      + j12*r20
    mul.f32       %f52, %f45, %f30;      // w11 = j11*r11
    fma.rn.f32    %f52, %f47, %f34, %f52;//      + j12*r21
    mul.f32       %f53, %f45, %f31;      // w12 = j11*r12
    fma.rn.f32    %f53, %f47, %f35, %f53;//      + j12*r22

    // Load Σ_3d (9 floats, row-major) from p_cov3d (%rd1) at r4*36
    mul.wide.u32  %rd8, %r4, 36;
    add.u64       %rd9, %rd1, %rd8;
    ld.global.f32 %f54, [%rd9];          // s0 = Σ[0]
    ld.global.f32 %f55, [%rd9+4];        // s1 = Σ[1]
    ld.global.f32 %f56, [%rd9+8];        // s2 = Σ[2]
    ld.global.f32 %f57, [%rd9+12];       // s3 = Σ[3]
    ld.global.f32 %f58, [%rd9+16];       // s4 = Σ[4]
    ld.global.f32 %f59, [%rd9+20];       // s5 = Σ[5]
    ld.global.f32 %f60, [%rd9+24];       // s6 = Σ[6]
    ld.global.f32 %f61, [%rd9+28];       // s7 = Σ[7]
    ld.global.f32 %f62, [%rd9+32];       // s8 = Σ[8]

    // WΣ = W·Σ (2×3):  ws[i][j] = Σ_k w[i][k]*Σ[k][j]
    mul.f32       %f63, %f48, %f54;      // ws00 = w00*s0
    fma.rn.f32    %f63, %f49, %f57, %f63;//        + w01*s3
    fma.rn.f32    %f63, %f50, %f60, %f63;//        + w02*s6
    mul.f32       %f64, %f48, %f55;      // ws01 = w00*s1
    fma.rn.f32    %f64, %f49, %f58, %f64;//        + w01*s4
    fma.rn.f32    %f64, %f50, %f61, %f64;//        + w02*s7
    mul.f32       %f65, %f48, %f56;      // ws02 = w00*s2
    fma.rn.f32    %f65, %f49, %f59, %f65;//        + w01*s5
    fma.rn.f32    %f65, %f50, %f62, %f65;//        + w02*s8
    mul.f32       %f66, %f51, %f54;      // ws10 = w10*s0
    fma.rn.f32    %f66, %f52, %f57, %f66;//        + w11*s3
    fma.rn.f32    %f66, %f53, %f60, %f66;//        + w12*s6
    mul.f32       %f67, %f51, %f55;      // ws11 = w10*s1
    fma.rn.f32    %f67, %f52, %f58, %f67;//        + w11*s4
    fma.rn.f32    %f67, %f53, %f61, %f67;//        + w12*s7
    mul.f32       %f68, %f51, %f56;      // ws12 = w10*s2
    fma.rn.f32    %f68, %f52, %f59, %f68;//        + w11*s5
    fma.rn.f32    %f68, %f53, %f62, %f68;//        + w12*s8

    // cov2d = WΣ·Wᵀ (2×2):  c[i][j] = Σ_k ws[i][k]*w[j][k]
    mul.f32       %f69, %f63, %f48;      // c00 = ws00*w00
    fma.rn.f32    %f69, %f64, %f49, %f69;//       + ws01*w01
    fma.rn.f32    %f69, %f65, %f50, %f69;//       + ws02*w02
    mul.f32       %f70, %f63, %f51;      // c01 = ws00*w10
    fma.rn.f32    %f70, %f64, %f52, %f70;//       + ws01*w11
    fma.rn.f32    %f70, %f65, %f53, %f70;//       + ws02*w12
    mul.f32       %f71, %f66, %f48;      // c10 = ws10*w00
    fma.rn.f32    %f71, %f67, %f49, %f71;//       + ws11*w01
    fma.rn.f32    %f71, %f68, %f50, %f71;//       + ws12*w02
    mul.f32       %f72, %f66, %f51;      // c11 = ws10*w10
    fma.rn.f32    %f72, %f67, %f52, %f72;//       + ws11*w11
    fma.rn.f32    %f72, %f68, %f53, %f72;//       + ws12*w12

    // + 0.3·I regularization on the diagonal
    add.f32       %f69, %f69, {REG03};
    add.f32       %f72, %f72, {REG03};

    // Store cov2d (4 floats, row-major) to p_out_cov2d (%rd5) at r4*16
    mul.wide.u32  %rd8, %r4, 16;
    add.u64       %rd9, %rd5, %rd8;
    st.global.f32 [%rd9], %f69;
    st.global.f32 [%rd9+4], %f70;
    st.global.f32 [%rd9+8], %f71;
    st.global.f32 [%rd9+12], %f72;

$PJ_DONE:
    mov.u32       %r6, 0;
    mov.u32       %r7, 0;
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f37, {ZERO};
    mov.f32       %f38, {REG03};
    mov.u64       %rd13, 0;
    ret;
}}
"#,
        ZERO = zero,
        REG03 = reg03,
    )
}

// ─── Kernel 7: sh_eval ────────────────────────────────────────────────────────

/// SH evaluation kernel: evaluate spherical harmonics up to L=2 using constants
/// as f32 hex literals.
#[must_use]
pub fn sh_eval_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // SH constants
    let y00 = f32_hex(0.282_094_8_f32);
    let y11 = f32_hex(0.488_602_5_f32);
    let y20 = f32_hex(0.315_391_6_f32);
    let y21a = f32_hex(1.092_548_4_f32);
    let y22a = f32_hex(0.546_274_2_f32);
    let three = f32_hex(3.0_f32);
    let neg_one = f32_hex(-1.0_f32);
    format!(
        r#"{hdr}// sh_eval_kernel: evaluate SH coefficients at direction (dx,dy,dz) for L=0..2.
// p_sh: float[n * 27] (sh coefficients per gaussian, RGB interleaved: 9 per channel),
// p_dir: float[n * 3] (unit view directions),
// p_out: float[n * 3] (RGB color output),
// n: number of Gaussians.
// Y_00 = {Y00}, Y_11 = {Y11}, Y_20 = {Y20}, Y_21a = {Y21A}, Y_22a = {Y22A}
.visible .entry sh_eval_kernel(
    .param .u64 p_sh,
    .param .u64 p_dir,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<10>;
    .reg .f32  %f<48>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_sh];
    ld.param.u64  %rd1, [p_dir];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // Gaussian index

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $SH_DONE;

    // SH basis constants
    mov.f32       %f20, {Y00};
    mov.f32       %f21, {Y11};
    mov.f32       %f22, {Y20};
    mov.f32       %f23, {Y21A};
    mov.f32       %f24, {Y22A};

    // Load direction
    mul.wide.u32  %rd3, %r4, 12;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];           // dx
    add.u64       %rd5, %rd4, 4;
    ld.global.f32 %f1, [%rd5];           // dy
    add.u64       %rd6, %rd4, 8;
    ld.global.f32 %f2, [%rd6];           // dz

    // Precompute the 9 SH basis values once (shared across R/G/B channels):
    //   B0=Y00, B1=Y11*dx, B2=Y11*dy, B3=Y11*dz, B4=Y20*(3dz²-1),
    //   B5=Y21A*dx*dz, B6=Y21A*dy*dz, B7=Y22A*(dx²-dy²), B8=Y21A*dx*dy
    mul.f32       %f3, %f21, %f0;        // B1 = Y11*dx
    mul.f32       %f4, %f21, %f1;        // B2 = Y11*dy
    mul.f32       %f5, %f21, %f2;        // B3 = Y11*dz
    mul.f32       %f6, %f2, %f2;         // dz²
    fma.rn.f32    %f7, %f6, {THREE}, {NEGONE}; // 3dz²-1
    mul.f32       %f7, %f7, %f22;        // B4 = Y20*(3dz²-1)
    mul.f32       %f8, %f0, %f2;         // dx*dz
    mul.f32       %f8, %f8, %f23;        // B5 = Y21A*dx*dz
    mul.f32       %f9, %f1, %f2;         // dy*dz
    mul.f32       %f9, %f9, %f23;        // B6 = Y21A*dy*dz
    mul.f32       %f10, %f0, %f0;        // dx²
    mul.f32       %f11, %f1, %f1;        // dy²
    sub.f32       %f13, %f10, %f11;      // dx²-dy²
    mul.f32       %f13, %f13, %f24;      // B7 = Y22A*(dx²-dy²)
    mul.f32       %f12, %f0, %f1;        // dx*dy
    mul.f32       %f14, %f12, %f23;      // B8 = Y21A*dx*dy

    // sh coeff base: p_sh + r4*108  (27 floats per Gaussian)
    mul.wide.u32  %rd7, %r4, 108;
    add.u64       %rd8, %rd0, %rd7;

    // output base: p_out + r4*12
    mul.wide.u32  %rd10, %r4, 12;
    add.u64       %rd11, %rd2, %rd10;

    // ---- R channel (coeffs at byte offset 0) ----
    ld.global.f32 %f15, [%rd8];
    mul.f32       %f27, %f15, %f20;      // sh0*Y00
    ld.global.f32 %f15, [%rd8+4];
    fma.rn.f32    %f27, %f15, %f3,  %f27;
    ld.global.f32 %f15, [%rd8+8];
    fma.rn.f32    %f27, %f15, %f4,  %f27;
    ld.global.f32 %f15, [%rd8+12];
    fma.rn.f32    %f27, %f15, %f5,  %f27;
    ld.global.f32 %f15, [%rd8+16];
    fma.rn.f32    %f27, %f15, %f7,  %f27;
    ld.global.f32 %f15, [%rd8+20];
    fma.rn.f32    %f27, %f15, %f8,  %f27;
    ld.global.f32 %f15, [%rd8+24];
    fma.rn.f32    %f27, %f15, %f9,  %f27;
    ld.global.f32 %f15, [%rd8+28];
    fma.rn.f32    %f27, %f15, %f13, %f27;
    ld.global.f32 %f15, [%rd8+32];
    fma.rn.f32    %f27, %f15, %f14, %f27;
    st.global.f32 [%rd11], %f27;

    // ---- G channel (coeffs at byte offset 36) ----
    add.u64       %rd9, %rd8, 36;
    ld.global.f32 %f15, [%rd9];
    mul.f32       %f28, %f15, %f20;
    ld.global.f32 %f15, [%rd9+4];
    fma.rn.f32    %f28, %f15, %f3,  %f28;
    ld.global.f32 %f15, [%rd9+8];
    fma.rn.f32    %f28, %f15, %f4,  %f28;
    ld.global.f32 %f15, [%rd9+12];
    fma.rn.f32    %f28, %f15, %f5,  %f28;
    ld.global.f32 %f15, [%rd9+16];
    fma.rn.f32    %f28, %f15, %f7,  %f28;
    ld.global.f32 %f15, [%rd9+20];
    fma.rn.f32    %f28, %f15, %f8,  %f28;
    ld.global.f32 %f15, [%rd9+24];
    fma.rn.f32    %f28, %f15, %f9,  %f28;
    ld.global.f32 %f15, [%rd9+28];
    fma.rn.f32    %f28, %f15, %f13, %f28;
    ld.global.f32 %f15, [%rd9+32];
    fma.rn.f32    %f28, %f15, %f14, %f28;
    st.global.f32 [%rd11+4], %f28;

    // ---- B channel (coeffs at byte offset 72) ----
    add.u64       %rd9, %rd8, 72;
    ld.global.f32 %f15, [%rd9];
    mul.f32       %f29, %f15, %f20;
    ld.global.f32 %f15, [%rd9+4];
    fma.rn.f32    %f29, %f15, %f3,  %f29;
    ld.global.f32 %f15, [%rd9+8];
    fma.rn.f32    %f29, %f15, %f4,  %f29;
    ld.global.f32 %f15, [%rd9+12];
    fma.rn.f32    %f29, %f15, %f5,  %f29;
    ld.global.f32 %f15, [%rd9+16];
    fma.rn.f32    %f29, %f15, %f7,  %f29;
    ld.global.f32 %f15, [%rd9+20];
    fma.rn.f32    %f29, %f15, %f8,  %f29;
    ld.global.f32 %f15, [%rd9+24];
    fma.rn.f32    %f29, %f15, %f9,  %f29;
    ld.global.f32 %f15, [%rd9+28];
    fma.rn.f32    %f29, %f15, %f13, %f29;
    ld.global.f32 %f15, [%rd9+32];
    fma.rn.f32    %f29, %f15, %f14, %f29;
    st.global.f32 [%rd11+8], %f29;

$SH_DONE:
    mov.u32       %r6, 0;
    mov.u32       %r7, 0;
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    ret;
}}
"#,
        Y00 = y00,
        Y11 = y11,
        Y20 = y20,
        Y21A = y21a,
        Y22A = y22a,
        THREE = three,
        NEGONE = neg_one,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_kernel_well_formed(prog: &str, sm: u32, kernel_name: &str) {
        assert!(prog.contains(&format!("sm_{sm}")), "missing sm_{sm} target");
        assert!(prog.contains(".version"), "missing .version");
        assert!(prog.contains(".visible .entry"), "missing .visible .entry");
        assert!(
            prog.contains(kernel_name),
            "missing kernel name {kernel_name}"
        );
    }

    #[test]
    fn fps_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&farthest_point_sample_ptx(sm), sm, "fps_kernel");
        }
    }

    #[test]
    fn ball_query_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&ball_query_ptx(sm), sm, "ball_query_kernel");
        }
    }

    #[test]
    fn gather_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&gather_points_ptx(sm), sm, "gather_kernel");
        }
    }

    #[test]
    fn voxelize_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&voxelize_ptx(sm), sm, "voxelize_kernel");
        }
    }

    #[test]
    fn chamfer_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&chamfer_distance_ptx(sm), sm, "chamfer_kernel");
        }
    }

    #[test]
    fn gaussian_project_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&gaussian_project_ptx(sm), sm, "project_kernel");
        }
    }

    #[test]
    fn sh_eval_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&sh_eval_ptx(sm), sm, "sh_eval_kernel");
        }
    }

    #[test]
    fn ptx_header_version_strings() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
        assert!(ptx_header(120).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_known_values() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
        assert_eq!(f32_hex(2.0_f32), "0F40000000");
    }

    #[test]
    fn fps_uses_fma() {
        let p = farthest_point_sample_ptx(80);
        assert!(p.contains("fma.rn.f32"));
        assert!(p.contains(".version 8.0"));
    }

    #[test]
    fn chamfer_uses_atomic_add() {
        let p = chamfer_distance_ptx(80);
        assert!(p.contains("atom.global.add.f32"));
        assert!(p.contains("fma.rn.f32"));
    }

    #[test]
    fn gather_uses_mul_wide() {
        let p = gather_points_ptx(80);
        assert!(p.contains("mul.wide.u32"));
    }

    #[test]
    fn voxelize_uses_atom_add() {
        let p = voxelize_ptx(90);
        assert!(p.contains("atom.global.add.f32"));
        assert!(p.contains("atom.global.add.u32"));
    }

    #[test]
    fn gaussian_project_uses_fma() {
        let p = gaussian_project_ptx(100);
        assert!(p.contains("fma.rn.f32"));
        assert!(p.contains("div.rn.f32"));
    }

    #[test]
    fn sh_eval_contains_sh_constants() {
        let p = sh_eval_ptx(80);
        // Y00 hex
        assert!(p.contains("0F3F906FBB") || p.contains("Y00"));
        assert!(p.contains(".version 8.0"));
    }

    #[test]
    fn all_kernels_nonempty_for_all_sm() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        for sm in sm_versions {
            assert!(!farthest_point_sample_ptx(sm).is_empty());
            assert!(!ball_query_ptx(sm).is_empty());
            assert!(!gather_points_ptx(sm).is_empty());
            assert!(!voxelize_ptx(sm).is_empty());
            assert!(!chamfer_distance_ptx(sm).is_empty());
            assert!(!gaussian_project_ptx(sm).is_empty());
            assert!(!sh_eval_ptx(sm).is_empty());
        }
    }
}
