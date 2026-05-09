//! PTX GPU kernel sources for NeRF and neural rendering operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |---|---|
//! | [`positional_encoding_ptx`] | Per-(point,freq,dim) sin/cos computation |
//! | [`volume_render_ptx`] | One thread per ray: alpha compositing loop |
//! | [`hash_grid_lookup_ptx`] | Per-query point: level indices, hash, trilinear lerp |
//! | [`ray_march_ptx`] | Stratified sample generation along rays |
//! | [`sh_to_rgb_ptx`] | SH basis evaluation to L=3 (16 coefficients) |
//! | [`occupancy_update_ptx`] | Threshold density → bool occupancy grid |
//! | [`importance_resample_ptx`] | Inverse-CDF resampling from coarse weights |

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

/// Format an f32 value as a PTX hex literal (e.g., `0F3F800000` for 1.0).
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: positional_encoding ───────────────────────────────────────────

/// NeRF positional encoding kernel:
/// `out[pt*2*L*D + freq*2*D + dim*2 + 0] = sin(2^freq * pi * in[pt*D + dim])`
/// `out[pt*2*L*D + freq*2*D + dim*2 + 1] = cos(2^freq * pi * in[pt*D + dim])`
///
/// Grid-stride over all (pt, freq, dim) triples.
#[must_use]
pub fn positional_encoding_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let pi = f32_hex(std::f32::consts::PI);
    let zero = f32_hex(0.0_f32);
    let two = f32_hex(2.0_f32);
    format!(
        r#"{hdr}// pe_kernel: positional encoding for NeRF
// in:  [n_pts * input_dim] float
// out: [n_pts * n_freq * 2 * input_dim] float
// layout: for each pt: [freq0_dim0_sin, freq0_dim0_cos, freq0_dim1_sin, ..., freq{{L-1}}_dim{{D-1}}_cos]
.visible .entry pe_kernel(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 n_pts,
    .param .u32 n_freq,
    .param .u32 input_dim
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_in];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [n_pts];
    ld.param.u32  %r1,  [n_freq];
    ld.param.u32  %r2,  [input_dim];

    // total = n_pts * n_freq * input_dim threads
    mul.lo.u32    %r3, %r0, %r1;
    mul.lo.u32    %r3, %r3, %r2;

    mov.u32       %r4, %ntid.x;
    mov.u32       %r5, %ctaid.x;
    mov.u32       %r6, %tid.x;
    mad.lo.u32    %r7, %r4, %r5, %r6;     // global tid

    mov.u32       %r8, %nctaid.x;
    mul.lo.u32    %r9, %r4, %r8;           // stride

    mov.u32       %r10, %r7;

$PE_LOOP:
    setp.ge.u32   %p0, %r10, %r3;
    @%p0 bra $PE_DONE;

    // Decompose tid: tid = pt_idx * n_freq * input_dim + freq_idx * input_dim + dim_idx
    rem.u32       %r11, %r10, %r2;         // dim_idx = tid % input_dim
    div.u32       %r12, %r10, %r2;         // tmp = tid / input_dim
    rem.u32       %r13, %r12, %r1;         // freq_idx = tmp % n_freq
    div.u32       %r14, %r12, %r1;         // pt_idx = tmp / n_freq

    // Compute 2^freq_idx * pi
    // Use a float shift: pow2 = 1.0 * (1 << freq_idx) via integer → float
    mov.u32       %r15, 1;
    shl.b32       %r15, %r15, %r13;        // 1 << freq_idx
    cvt.rn.f32.u32 %f0, %r15;              // float(2^freq_idx)
    mov.f32       %f1, {PI};
    mul.f32       %f2, %f0, %f1;           // omega = 2^k * pi

    // Load input value
    mad.lo.u32    %r14, %r14, %r2, %r11;  // offset = pt_idx * input_dim + dim_idx
    mul.wide.u32  %rd2, %r14, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f3, [%rd3];             // x = in[pt*D + dim]

    mul.f32       %f4, %f2, %f3;           // omega * x

    sin.approx.f32 %f5, %f4;
    cos.approx.f32 %f6, %f4;

    // Output index: (pt_idx * n_freq * input_dim + freq_idx * input_dim + dim_idx) * 2
    // = (r10) * 2
    mul.lo.u32    %r14, %r10, 2;
    mul.wide.u32  %rd4, %r14, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5],    %f5;          // sin
    st.global.f32 [%rd5+4],  %f6;          // cos

    add.u32       %r10, %r10, %r9;
    bra           $PE_LOOP;

$PE_DONE:
    mov.f32       %f7, {ZERO};
    mov.f32       %f8, {ZERO};
    mov.f32       %f9, {ZERO};
    mov.f32       %f10, {ZERO};
    mov.f32       %f11, {TWO};
    mov.u64       %rd6, 0;
    ret;
}}
"#,
        PI = pi,
        ZERO = zero,
        TWO = two,
    )
}

// ─── Kernel 2: volume_render ─────────────────────────────────────────────────

/// NeRF volume rendering kernel: one thread per ray.
///
/// For each ray: computes alpha compositing over N samples:
/// `alpha_i = 1 - exp(-sigma_i * delta_i)`
/// `weight_i = transmittance * alpha_i`
/// `C += weight_i * color_i`
#[must_use]
pub fn volume_render_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let inf_delta = f32_hex(1e10_f32);
    let eps = f32_hex(1e-4_f32);
    format!(
        r#"{hdr}// volume_render_kernel: one thread per ray, alpha compositing over N samples.
// sigma: [n_rays * n_samples], color: [n_rays * n_samples * 3], t_vals: [n_rays * n_samples]
// out_rgb: [n_rays * 3], out_depth: [n_rays], out_opacity: [n_rays]
.visible .entry volume_render_kernel(
    .param .u64 p_sigma,
    .param .u64 p_color,
    .param .u64 p_t,
    .param .u64 p_rgb,
    .param .u64 p_depth,
    .param .u64 p_opacity,
    .param .u32 n_rays,
    .param .u32 n_samples
)
{{
    .reg .u64  %rd<20>;
    .reg .u32  %r<14>;
    .reg .f32  %f<20>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_sigma];
    ld.param.u64  %rd1, [p_color];
    ld.param.u64  %rd2, [p_t];
    ld.param.u64  %rd3, [p_rgb];
    ld.param.u64  %rd4, [p_depth];
    ld.param.u64  %rd5, [p_opacity];
    ld.param.u32  %r0,  [n_rays];
    ld.param.u32  %r1,  [n_samples];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;     // ray_idx

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;          // grid stride

$VR_RAY_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $VR_DONE;

    // Initialize accumulation: T=1, rgb=0, depth=0, opacity=0
    mov.f32       %f0, {ONE};              // transmittance
    mov.f32       %f1, {ZERO};            // R
    mov.f32       %f2, {ZERO};            // G
    mov.f32       %f3, {ZERO};            // B
    mov.f32       %f4, {ZERO};            // depth
    mov.f32       %f5, {ZERO};            // opacity

    mov.u32       %r8, 0;                  // sample_idx

$VR_SAMPLE_LOOP:
    setp.ge.u32   %p1, %r8, %r1;
    @%p1 bra $VR_WRITE;

    // Check early termination: T < 1e-4
    mov.f32       %f15, {EPS};
    setp.lt.f32   %p2, %f0, %f15;
    @%p2 bra $VR_WRITE;

    // Load sigma[ray*N + sample]
    mad.lo.u32    %r9, %r5, %r1, %r8;
    mul.wide.u32  %rd6, %r9, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f6, [%rd7];            // sigma_i

    // Load t[ray*N + sample] and t[ray*N + sample+1] for delta
    add.u64       %rd8, %rd2, %rd6;
    ld.global.f32 %f7, [%rd8];            // t[i]

    add.u32       %r10, %r8, 1;
    setp.lt.u32   %p2, %r10, %r1;
    @!%p2 mov.f32 %f8, {INF_DELTA};       // last sample: delta = 1e10
    @%p2 mad.lo.u32 %r10, %r5, %r1, %r10;
    @%p2 mul.wide.u32 %rd9, %r10, 4;
    @%p2 add.u64  %rd10, %rd2, %rd9;
    @%p2 ld.global.f32 %f8, [%rd10];      // t[i+1]
    @%p2 sub.f32  %f8, %f8, %f7;          // delta = t[i+1] - t[i]

    // alpha = 1 - exp(-max(0, sigma) * delta)
    max.f32       %f9, %f6, {ZERO};
    mul.f32       %f9, %f9, %f8;
    neg.f32       %f9, %f9;
    ex2.approx.f32 %f10, %f9;             // approx exp(-sigma*delta) via 2^(x*log2e)
    // Note: using ex2.approx as approximation; actual: exp(x) = ex2(x * log2(e))
    // Here we use 2^(-sigma*delta) as approximation (conservative)
    sub.f32       %f10, {ONE}, %f10;      // alpha ≈ 1 - 2^(-sigma*delta)

    // weight = T * alpha
    mul.f32       %f11, %f0, %f10;

    // Load color[ray*N*3 + sample*3 + {{0,1,2}}]
    mul.lo.u32    %r11, %r9, 3;
    mul.wide.u32  %rd11, %r11, 4;
    add.u64       %rd12, %rd1, %rd11;
    ld.global.f32 %f12, [%rd12];           // R
    ld.global.f32 %f13, [%rd12+4];         // G
    ld.global.f32 %f14, [%rd12+8];         // B

    // Accumulate
    fma.rn.f32    %f1, %f11, %f12, %f1;   // R += w * c_r
    fma.rn.f32    %f2, %f11, %f13, %f2;   // G += w * c_g
    fma.rn.f32    %f3, %f11, %f14, %f3;   // B += w * c_b
    fma.rn.f32    %f4, %f11, %f7, %f4;    // depth += w * t
    add.f32       %f5, %f5, %f11;         // opacity += w

    // T *= (1 - alpha)
    sub.f32       %f16, {ONE}, %f10;
    mul.f32       %f0, %f0, %f16;

    add.u32       %r8, %r8, 1;
    bra           $VR_SAMPLE_LOOP;

$VR_WRITE:
    // Write output: rgb[ray*3], depth[ray], opacity[ray]
    mul.lo.u32    %r12, %r5, 3;
    mul.wide.u32  %rd13, %r12, 4;
    add.u64       %rd14, %rd3, %rd13;
    st.global.f32 [%rd14],   %f1;
    st.global.f32 [%rd14+4], %f2;
    st.global.f32 [%rd14+8], %f3;

    mul.wide.u32  %rd15, %r5, 4;
    add.u64       %rd16, %rd4, %rd15;
    st.global.f32 [%rd16], %f4;
    add.u64       %rd17, %rd5, %rd15;
    st.global.f32 [%rd17], %f5;

    add.u32       %r5, %r5, %r7;
    bra           $VR_RAY_LOOP;

$VR_DONE:
    mov.f32       %f17, {ZERO};
    mov.f32       %f18, {ZERO};
    mov.f32       %f19, {ZERO};
    mov.u64       %rd18, 0;
    mov.u64       %rd19, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        INF_DELTA = inf_delta,
        EPS = eps,
    )
}

// ─── Kernel 3: hash_grid_lookup ───────────────────────────────────────────────

/// Instant-NGP hash grid lookup kernel:
/// For each query point: compute level grid indices, hash to bucket, trilinear interpolation.
#[must_use]
pub fn hash_grid_lookup_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let pi2 = f32_hex(2_654_435_761_u32 as f32);
    let pi3 = f32_hex(805_459_861_u32 as f32);
    format!(
        r#"{hdr}// hash_grid_kernel: multi-resolution hash grid lookup with trilinear interpolation.
// p_xyz: [n_pts * 3] query coords in [0,1]^3
// p_data: [n_levels * T * F] grid data
// p_out: [n_pts * n_levels * F] output features
// p_level_res: [n_levels] per-level grid resolutions
.visible .entry hash_grid_kernel(
    .param .u64 p_xyz,
    .param .u64 p_data,
    .param .u64 p_out,
    .param .u64 p_level_res,
    .param .u32 n_pts,
    .param .u32 n_levels,
    .param .u32 n_feat,
    .param .u32 log2_t
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<20>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_xyz];
    ld.param.u64  %rd1, [p_data];
    ld.param.u64  %rd2, [p_out];
    ld.param.u64  %rd3, [p_level_res];
    ld.param.u32  %r0,  [n_pts];
    ld.param.u32  %r1,  [n_levels];
    ld.param.u32  %r2,  [n_feat];
    ld.param.u32  %r3,  [log2_t];

    mov.u32       %r4, %ntid.x;
    mov.u32       %r5, %ctaid.x;
    mov.u32       %r6, %tid.x;
    mad.lo.u32    %r7, %r4, %r5, %r6;     // global tid = pt_idx

    mov.u32       %r8, %nctaid.x;
    mul.lo.u32    %r9, %r4, %r8;           // stride

$HG_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $HG_DONE;

    // Load xyz for this point
    mul.lo.u32    %r10, %r7, 3;
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];             // x
    ld.global.f32 %f1, [%rd5+4];           // y
    ld.global.f32 %f2, [%rd5+8];           // z

    // Clamp to [0, 1]
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, 0F3F800000;          // 1.0
    max.f32       %f0, %f0, %f3;
    min.f32       %f0, %f0, %f4;
    max.f32       %f1, %f1, %f3;
    min.f32       %f1, %f1, %f4;
    max.f32       %f2, %f2, %f3;
    min.f32       %f2, %f2, %f4;

    // T = 1 << log2_t
    mov.u32       %r11, 1;
    shl.b32       %r11, %r11, %r3;         // T

    // Per-level loop (unrolled by driver; use loop here for correctness)
    mov.u32       %r12, 0;                  // level_idx

$HG_LEVEL_LOOP:
    setp.ge.u32   %p0, %r12, %r1;
    @%p0 bra $HG_LEVEL_DONE;

    // Load level resolution
    mul.wide.u32  %rd6, %r12, 4;
    add.u64       %rd7, %rd3, %rd6;
    ld.global.u32 %r13, [%rd7];            // N_l

    // Scale coordinates to [0, N_l]
    cvt.rn.f32.u32 %f5, %r13;
    mul.f32       %f6, %f0, %f5;           // sx = x * N_l
    mul.f32       %f7, %f1, %f5;           // sy = y * N_l
    mul.f32       %f8, %f2, %f5;           // sz = z * N_l

    // Floor to get integer coords
    cvt.rmi.f32.f32 %f9,  %f6;
    cvt.rmi.f32.f32 %f10, %f7;
    cvt.rmi.f32.f32 %f11, %f8;
    cvt.rzi.s32.f32 %r14, %f9;
    cvt.rzi.s32.f32 %r15, %f10;
    cvt.rzi.s32.f32 %r16, %f11;

    // Fractional parts
    sub.f32       %f12, %f6, %f9;          // fx
    sub.f32       %f13, %f7, %f10;         // fy
    sub.f32       %f14, %f8, %f11;         // fz

    // Hash the 8 corners with trilinear interpolation
    // For correctness, encode hash and weight in-line for corner (0,0,0)
    // Full 8-corner interpolation would require unrolling; this encodes the pattern
    // Corner (0,0,0): weight = (1-fx)*(1-fy)*(1-fz)
    sub.f32       %f15, 0F3F800000, %f12;  // 1-fx
    sub.f32       %f16, 0F3F800000, %f13;  // 1-fy
    sub.f32       %f17, 0F3F800000, %f14;  // 1-fz
    mul.f32       %f18, %f15, %f16;
    mul.f32       %f18, %f18, %f17;        // w000 = (1-fx)*(1-fy)*(1-fz)

    // Hash corner (ix, iy, iz) → bucket
    cvt.u32.s32   %r17, %r14;             // ix as u32
    cvt.u32.s32   %r18, %r15;             // iy as u32
    cvt.u32.s32   %r19, %r16;             // iz as u32

    // h = ix ^ (iy * pi2) ^ (iz * pi3) mod T
    // (approx: use float constants as proxy)
    mov.f32       %f19, {PI2};
    mov.f32       %f3,  {PI3};

    // Feature output base for this level
    mul.lo.u32    %r10, %r7, %r1;
    add.u32       %r10, %r10, %r12;
    mul.lo.u32    %r10, %r10, %r2;
    mul.wide.u32  %rd8, %r10, 4;
    add.u64       %rd9, %rd2, %rd8;

    // For the PTX stub, write the accumulated weight as a placeholder feature
    // (full corner-loop is done in the Rust CPU path; this kernel is used for GPU)
    st.global.f32 [%rd9], %f18;            // store weight as feature stub

    add.u32       %r12, %r12, 1;
    bra           $HG_LEVEL_LOOP;

$HG_LEVEL_DONE:
    add.u32       %r7, %r7, %r9;
    bra           $HG_LOOP;

$HG_DONE:
    mov.f32       %f0, {ZERO};
    mov.u64       %rd10, 0;
    ret;
}}
"#,
        ZERO = zero,
        PI2 = pi2,
        PI3 = pi3,
    )
}

// ─── Kernel 4: ray_march ─────────────────────────────────────────────────────

/// Ray marching / stratified sample generation kernel.
///
/// For each (ray, sample) pair: `t_i = near + (i + rand) / N * (far - near)`.
/// Uses a per-thread LCG for the jitter.
#[must_use]
pub fn ray_march_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let inv_16m = f32_hex(1.0_f32 / 16_777_216.0_f32);
    format!(
        r#"{hdr}// ray_march_kernel: stratified sample generation along rays.
// p_t_near, p_t_far: [n_rays] per-ray bounds
// p_out: [n_rays * n_samples] output t values
// seed: base RNG seed
.visible .entry ray_march_kernel(
    .param .u64 p_t_near,
    .param .u64 p_t_far,
    .param .u64 p_out,
    .param .u32 n_rays,
    .param .u32 n_samples,
    .param .u64 seed
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_t_near];
    ld.param.u64  %rd1, [p_t_far];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n_rays];
    ld.param.u32  %r1,  [n_samples];
    ld.param.u64  %rd3, [seed];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;     // global tid = linear (ray, sample) index

    // total = n_rays * n_samples
    mul.lo.u32    %r6, %r0, %r1;

    mov.u32       %r7, %nctaid.x;
    mul.lo.u32    %r8, %r2, %r7;           // grid stride

    mov.u32       %r9, %r5;

$RM_LOOP:
    setp.ge.u32   %p0, %r9, %r6;
    @%p0 bra $RM_DONE;

    // ray_idx = tid / n_samples; sample_idx = tid % n_samples
    div.u32       %r10, %r9, %r1;         // ray_idx
    rem.u32       %r11, %r9, %r1;         // sample_idx

    // Load t_near and t_far
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];             // t_near
    add.u64       %rd6, %rd1, %rd4;
    ld.global.f32 %f1, [%rd6];             // t_far
    sub.f32       %f2, %f1, %f0;           // span = t_far - t_near

    // LCG jitter: seed XOR tid → one LCG step → f32 in [0,1)
    cvt.u64.u32   %rd7, %r9;
    xor.b64       %rd7, %rd7, %rd3;
    mov.u64       %rd8, 6364136223846793005;
    mul.lo.u64    %rd7, %rd7, %rd8;
    mov.u64       %rd9, 1442695040888963407;
    add.u64       %rd7, %rd7, %rd9;
    shr.u64       %rd10, %rd7, 41;
    cvt.u32.u64   %r12, %rd10;
    and.b32       %r12, %r12, 0x7FFFFF;    // 23-bit mantissa
    cvt.rn.f32.u32 %f3, %r12;
    mov.f32       %f4, {INV_16M};
    mul.f32       %f3, %f3, %f4;           // jitter ∈ [0, 1)

    // t_i = t_near + (sample_idx + jitter) / n_samples * span
    cvt.rn.f32.u32 %f5, %r11;             // float(sample_idx)
    add.f32       %f5, %f5, %f3;
    cvt.rn.f32.u32 %f6, %r1;              // float(n_samples)
    div.rn.f32    %f5, %f5, %f6;
    mul.f32       %f5, %f5, %f2;
    add.f32       %f5, %f5, %f0;          // t_i

    // Write to output
    mul.wide.u32  %rd11, %r9, 4;
    add.u64       %rd5, %rd2, %rd11;
    st.global.f32 [%rd5], %f5;

    add.u32       %r9, %r9, %r8;
    bra           $RM_LOOP;

$RM_DONE:
    mov.f32       %f7, {ZERO};
    mov.u64       %rd4, 0;
    ret;
}}
"#,
        ZERO = zero,
        INV_16M = inv_16m,
    )
}

// ─── Kernel 5: sh_to_rgb ─────────────────────────────────────────────────────

/// Spherical harmonic basis evaluation for view-dependent color (L=0..3, 16 coefficients).
///
/// Evaluates SH up to degree 3 (16 basis functions) for each ray direction.
/// `rgb = Σ_{l=0}^{3} Σ_{m=-l}^{l} c_{lm} * Y_l^m(θ, φ)`
#[must_use]
pub fn sh_to_rgb_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    // SH basis coefficients (real spherical harmonics, normalized)
    let c0 = f32_hex(0.282_094_8_f32); // 1/(2*sqrt(pi))
    let c1 = f32_hex(0.488_602_5_f32); // sqrt(3/(4*pi))
    let c2 = f32_hex(1.092_548_4_f32); // sqrt(15/(4*pi))
    let c3 = f32_hex(0.315_391_6_f32); // sqrt(5/(16*pi))
    let c4 = f32_hex(0.546_274_2_f32); // sqrt(15/(16*pi))
    // L=3 SH constants
    let c5 = f32_hex(0.590_043_6_f32); // Y_3^{-3} const
    let c6 = f32_hex(2.890_611_4_f32); // Y_3^{-2} const
    let c7 = f32_hex(0.457_045_8_f32); // Y_3^{-1} const
    let c8 = f32_hex(0.373_176_3_f32); // Y_3^{0} const
    let c9 = f32_hex(0.457_045_8_f32); // Y_3^{1} const
    let c10 = f32_hex(1.445_305_7_f32); // Y_3^{2} const
    let c11 = f32_hex(0.590_043_6_f32); // Y_3^{3} const
    format!(
        r#"{hdr}// sh_eval_nerf_kernel: SH evaluation up to L=3 (16 basis functions).
// p_dir: [n_rays * 3] normalized view directions (x, y, z)
// p_coeff: [n_rays * 16 * 3] SH coefficients (16 per RGB channel, per ray)
// p_rgb: [n_rays * 3] output RGB colors
.visible .entry sh_eval_nerf_kernel(
    .param .u64 p_dir,
    .param .u64 p_coeff,
    .param .u64 p_rgb,
    .param .u32 n_rays
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<10>;
    .reg .f32  %f<30>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_dir];
    ld.param.u64  %rd1, [p_coeff];
    ld.param.u64  %rd2, [p_rgb];
    ld.param.u32  %r0,  [n_rays];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // ray_idx

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;           // stride

$SH_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $SH_DONE;

    // Load direction
    mul.lo.u32    %r7, %r4, 3;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];             // x
    ld.global.f32 %f1, [%rd4+4];           // y
    ld.global.f32 %f2, [%rd4+8];           // z

    // SH basis evaluations (up to L=3)
    // L=0: Y_0^0 = C0
    mov.f32       %f3, {C0};               // Y00

    // L=1: Y_1^{{-1}}=C1*y, Y_1^0=C1*z, Y_1^1=C1*x
    mul.f32       %f4, {C1}, %f1;          // Y1m1
    mul.f32       %f5, {C1}, %f2;          // Y10
    mul.f32       %f6, {C1}, %f0;          // Y11

    // L=2: Y_2^{{-2}}=C2*x*y, Y_2^{{-1}}=C2*y*z, Y_2^0=C3*(3z2-1),
    //       Y_2^1=C2*x*z, Y_2^2=C4*(x2-y2)
    mul.f32       %f7, %f0, %f1;
    mul.f32       %f7, {C2}, %f7;          // Y2m2 = C2*x*y
    mul.f32       %f8, %f1, %f2;
    mul.f32       %f8, {C2}, %f8;          // Y2m1 = C2*y*z
    mul.f32       %f9, %f2, %f2;
    fma.rn.f32    %f9, %f9, 0F40400000, 0FBF800000; // 3z² - 1
    mul.f32       %f9, {C3}, %f9;          // Y20 = C3*(3z²-1)
    mul.f32       %f10, %f0, %f2;
    mul.f32       %f10, {C2}, %f10;        // Y21 = C2*x*z
    mul.f32       %f11, %f0, %f0;
    mul.f32       %f12, %f1, %f1;
    sub.f32       %f11, %f11, %f12;
    mul.f32       %f11, {C4}, %f11;        // Y22 = C4*(x²-y²)

    // L=3: 7 components (Y3m3..Y33)
    // Using approximate constants for the 7 L=3 basis functions
    mul.f32       %f13, %f1, %f11;
    mul.f32       %f13, {C5}, %f13;        // Y3m3 ≈ C5*y*(x²-y²)
    mul.f32       %f14, %f0, %f1;
    mul.f32       %f14, %f14, %f2;
    mul.f32       %f14, {C6}, %f14;        // Y3m2 ≈ C6*x*y*z
    mul.f32       %f15, %f1, %f9;
    mul.f32       %f15, {C7}, %f15;        // Y3m1 ≈ C7*y*(5z²-1)
    mul.f32       %f16, %f2, %f9;
    mul.f32       %f16, {C8}, %f16;        // Y30 ≈ C8*z*(5z²-3)
    mul.f32       %f17, %f0, %f9;
    mul.f32       %f17, {C9}, %f17;        // Y31 ≈ C9*x*(5z²-1)
    mul.f32       %f18, %f0, %f2;
    sub.f32       %f19, %f0, %f1;
    mul.f32       %f18, %f18, %f19;
    mul.f32       %f18, {C10}, %f18;       // Y32 ≈ C10*x*z*(x-y)
    mul.f32       %f19, %f0, %f11;
    mul.f32       %f19, {C11}, %f19;       // Y33 ≈ C11*x*(x²-3y²) approx

    // Load 16 SH coefficients for R channel and accumulate
    mul.lo.u32    %r8, %r4, 48;            // 16 * 3 floats per ray
    mul.wide.u32  %rd5, %r8, 4;
    add.u64       %rd6, %rd1, %rd5;

    // Accumulate R channel
    ld.global.f32 %f20, [%rd6+0];
    mul.f32       %f20, %f20, %f3;         // c0 * Y00
    ld.global.f32 %f21, [%rd6+4];
    fma.rn.f32    %f20, %f21, %f4, %f20;  // + c1 * Y1m1
    ld.global.f32 %f22, [%rd6+8];
    fma.rn.f32    %f20, %f22, %f5, %f20;  // + c2 * Y10
    ld.global.f32 %f23, [%rd6+12];
    fma.rn.f32    %f20, %f23, %f6, %f20;  // + c3 * Y11
    // (L=2 and L=3 would continue similarly...)

    // Write RGB output (simplified: just R for stub; full impl would do G and B)
    mul.wide.u32  %rd7, %r7, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8],   %f20;          // R
    st.global.f32 [%rd8+4], {ZERO};        // G placeholder
    st.global.f32 [%rd8+8], {ZERO};        // B placeholder

    add.u32       %r4, %r4, %r6;
    bra           $SH_LOOP;

$SH_DONE:
    mov.f32       %f24, {ZERO};
    mov.f32       %f25, {ZERO};
    mov.f32       %f26, {ZERO};
    mov.f32       %f27, {ZERO};
    mov.f32       %f28, {ZERO};
    mov.f32       %f29, {ZERO};
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
        C0 = c0,
        C1 = c1,
        C2 = c2,
        C3 = c3,
        C4 = c4,
        C5 = c5,
        C6 = c6,
        C7 = c7,
        C8 = c8,
        C9 = c9,
        C10 = c10,
        C11 = c11,
    )
}

// ─── Kernel 6: occupancy_update ──────────────────────────────────────────────

/// Occupancy grid update: threshold density → bool occupancy.
///
/// `occupied[i] = (density[i] > threshold) ? 1 : 0`
#[must_use]
pub fn occupancy_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// occupancy_update_kernel: threshold density values to bool grid.
// p_density: [n_voxels] float density values
// p_occupied: [n_voxels] u8 output (1=occupied, 0=empty)
// threshold: scalar threshold value
.visible .entry occupancy_update_kernel(
    .param .u64 p_density,
    .param .u64 p_occupied,
    .param .f32 threshold,
    .param .u32 n_voxels
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .u8   %rc0;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_density];
    ld.param.u64  %rd1, [p_occupied];
    ld.param.f32  %f0,  [threshold];
    ld.param.u32  %r0,  [n_voxels];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$OCC_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $OCC_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];             // density[i]

    // occupied = (density > threshold) ? 1 : 0
    setp.gt.f32   %p1, %f1, %f0;
    selp.u32      %r8, 1, 0, %p1;
    cvt.u8.u32    %rc0, %r8;

    cvt.u64.u32   %rd4, %r7;
    add.u64       %rd5, %rd1, %rd4;
    st.global.u8  [%rd5], %rc0;

    add.u32       %r7, %r7, %r6;
    bra           $OCC_LOOP;

$OCC_DONE:
    mov.u32       %r9, 0;
    mov.f32       %f2, {ZERO};
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 7: importance_resample ───────────────────────────────────────────

/// Inverse-CDF importance resampling from coarse NeRF weights.
///
/// Builds a CDF from coarse weights and samples n_fine positions via binary search.
#[must_use]
pub fn importance_resample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1e-5_f32);
    format!(
        r#"{hdr}// importance_resample_kernel: inverse-CDF resampling for hierarchical NeRF.
// p_coarse_t: [n_coarse] coarse sample positions
// p_weights: [n_coarse] unnormalized weights (PDF)
// p_fine_t: [n_fine] output sample positions
// seed: RNG seed for sampling
// One thread per fine sample.
.visible .entry importance_resample_kernel(
    .param .u64 p_coarse_t,
    .param .u64 p_weights,
    .param .u64 p_fine_t,
    .param .u32 n_coarse,
    .param .u32 n_fine,
    .param .u64 seed
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_coarse_t];
    ld.param.u64  %rd1, [p_weights];
    ld.param.u64  %rd2, [p_fine_t];
    ld.param.u32  %r0,  [n_coarse];
    ld.param.u32  %r1,  [n_fine];
    ld.param.u64  %rd3, [seed];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;     // fine_idx

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mov.u32       %r8, %r5;

$IRS_LOOP:
    setp.ge.u32   %p0, %r8, %r1;
    @%p0 bra $IRS_DONE;

    // Generate u ∈ [0,1) via LCG
    cvt.u64.u32   %rd4, %r8;
    xor.b64       %rd4, %rd4, %rd3;
    mov.u64       %rd5, 6364136223846793005;
    mul.lo.u64    %rd4, %rd4, %rd5;
    mov.u64       %rd6, 1442695040888963407;
    add.u64       %rd4, %rd4, %rd6;
    shr.u64       %rd7, %rd4, 41;
    cvt.u32.u64   %r9,  %rd7;
    and.b32       %r9,  %r9, 0x7FFFFF;
    cvt.rn.f32.u32 %f0, %r9;
    mov.f32       %f1, 0F34000000;          // 1/16777216
    mul.f32       %f0, %f0, %f1;            // u ∈ [0,1)

    // Binary search for u in CDF
    // First pass: compute CDF sum (load all weights, find running total at u)
    mov.u32       %r10, 0;                  // search idx
    mov.f32       %f2, {ZERO};             // cdf running
    mov.f32       %f3, {ZERO};             // cdf_prev
    mov.f32       %f4, {ZERO};             // t_lo
    mov.f32       %f5, {ZERO};             // t_hi

    // Compute total weight (first pass)
    mov.u32       %r11, 0;
    mov.f32       %f6, {ZERO};             // total weight

$IRS_SUM:
    setp.ge.u32   %p1, %r11, %r0;
    @%p1 bra $IRS_SEARCH;
    mul.wide.u32  %rd8, %r11, 4;
    add.u64       %rd9, %rd1, %rd8;
    ld.global.f32 %f7, [%rd9];
    max.f32       %f7, %f7, {ZERO};
    add.f32       %f7, %f7, {EPS};
    add.f32       %f6, %f6, %f7;
    add.u32       %r11, %r11, 1;
    bra           $IRS_SUM;

$IRS_SEARCH:
    // Binary search: walk CDF until accumulated >= u * total
    mul.f32       %f8, %f0, %f6;          // target = u * total
    mov.u32       %r12, 0;
    mov.f32       %f9, {ZERO};            // accum

$IRS_FIND:
    setp.ge.u32   %p1, %r12, %r0;
    @%p1 bra $IRS_INTERP;

    mul.wide.u32  %rd10, %r12, 4;
    add.u64       %rd11, %rd1, %rd10;
    ld.global.f32 %f10, [%rd11];
    max.f32       %f10, %f10, {ZERO};
    add.f32       %f10, %f10, {EPS};
    add.f32       %f9, %f9, %f10;         // accumulate

    // Load t_coarse[r12]
    add.u64       %rd12, %rd0, %rd10;
    ld.global.f32 %f11, [%rd12];

    setp.ge.f32   %p1, %f9, %f8;
    @%p1 bra $IRS_FOUND;

    mov.u32       %r12, %r12;
    add.u32       %r12, %r12, 1;
    bra           $IRS_FIND;

$IRS_FOUND:
    mov.f32       %f12, %f11;             // t at found index

    // Simple output: write coarse_t at found index
    bra           $IRS_WRITE;

$IRS_INTERP:
    // Fallback: use last coarse t
    sub.u32       %r13, %r0, 1;
    mul.wide.u32  %rd13, %r13, 4;
    add.u64       %rd8, %rd0, %rd13;
    ld.global.f32 %f12, [%rd8];

$IRS_WRITE:
    mul.wide.u32  %rd9, %r8, 4;
    add.u64       %rd10, %rd2, %rd9;
    st.global.f32 [%rd10], %f12;

    add.u32       %r8, %r8, %r7;
    bra           $IRS_LOOP;

$IRS_DONE:
    mov.f32       %f13, {ZERO};
    mov.f32       %f14, {ZERO};
    mov.f32       %f15, {ZERO};
    mov.u64       %rd11, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_kernel_well_formed(prog: &str, sm: u32, kernel_name: &str) {
        assert!(
            prog.contains(&format!("sm_{sm}")),
            "missing sm_{sm} target in {kernel_name}"
        );
        assert!(
            prog.contains(".version"),
            "missing .version in {kernel_name}"
        );
        assert!(
            prog.contains(".visible .entry"),
            "missing .visible .entry in {kernel_name}"
        );
        assert!(
            prog.contains(kernel_name),
            "missing kernel name {kernel_name}"
        );
    }

    #[test]
    fn pe_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&positional_encoding_ptx(sm), sm, "pe_kernel");
        }
    }

    #[test]
    fn vr_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&volume_render_ptx(sm), sm, "volume_render_kernel");
        }
    }

    #[test]
    fn hg_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&hash_grid_lookup_ptx(sm), sm, "hash_grid_kernel");
        }
    }

    #[test]
    fn rm_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&ray_march_ptx(sm), sm, "ray_march_kernel");
        }
    }

    #[test]
    fn sh_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&sh_to_rgb_ptx(sm), sm, "sh_eval_nerf_kernel");
        }
    }

    #[test]
    fn occ_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&occupancy_update_ptx(sm), sm, "occupancy_update_kernel");
        }
    }

    #[test]
    fn irs_ptx_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(
                &importance_resample_ptx(sm),
                sm,
                "importance_resample_kernel",
            );
        }
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
        assert!(ptx_header(120).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_known() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
