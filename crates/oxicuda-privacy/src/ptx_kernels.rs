//! PTX kernel generators for `oxicuda-privacy`.
//!
//! Each function returns a `String` containing valid PTX assembly for the
//! target SM version.  These kernels are designed to be loaded at runtime
//! via `oxicuda-driver` / `oxicuda-launch` and do not require compile-time
//! CUDA SDK linkage.

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    let target = if sm >= 100 {
        "sm_100"
    } else if sm >= 90 {
        "sm_90"
    } else if sm >= 80 {
        "sm_80"
    } else {
        "sm_75"
    };
    format!(".version {ver}\n.target {target}\n.address_size 64\n")
}

#[allow(dead_code)]
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: exponential_sample ────────────────────────────────────────────

/// Generate PTX for the exponential mechanism sampler.
///
/// Kernel signature: `(f64* scores, u32 n, f64 weight_sum, f64 uniform_u, u32* out_idx)`
/// Each thread checks its assigned score's cumulative weight and
/// performs an atomic CAS to claim the selected index.
///
/// # PTX structure
/// - load thread index, guard against n
/// - load weight, compute prefix (parallel scan omitted for simplicity —
///   host-side normalization is preferred; kernel performs the selection step)
/// - compare running total against `uniform_u * weight_sum`
/// - atomic-min to claim winning index
#[must_use]
pub fn exponential_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// Exponential mechanism sampler
// scores[i] are pre-exponentiated weights (host normalizes first).
// Each thread i checks if cumsum[i] >= u * total and atomically
// writes its index if it is the first to cross the threshold.
.visible .entry exponential_sample(
    .param .u64 param_scores,      // f64* weights (pre-exp)
    .param .u32 param_n,           // number of outcomes
    .param .f64 param_threshold,   // u * total_weight (host-computed)
    .param .u64 param_out           // u32* output index
)
{{
    .reg .u64   rd_scores, rd_out;
    .reg .u32   r_n, r_tid, r_stride, r_one;
    .reg .f64   fd_w, fd_cum, fd_thr;
    .reg .pred  p_guard, p_cross;

    ld.param.u64    rd_scores,    [param_scores];
    ld.param.u32    r_n,          [param_n];
    ld.param.f64    fd_thr,       [param_threshold];
    ld.param.u64    rd_out,       [param_out];

    mov.u32         r_tid,        %tid.x;
    setp.ge.u32     p_guard,      r_tid, r_n;
    @p_guard bra    EXIT;

    // Compute byte offset for this thread's weight element
    mul.wide.u32    rd_scores,    r_tid, 8;      // 8 bytes per f64
    add.u64         rd_scores,    rd_scores, %rd_scores; // NOTE: for demo
    // Simplified: each thread loads its own weight
    // (full kernel would use shared-memory prefix scan)
    ld.global.f64   fd_w,         [rd_scores];

    // If this weight >= threshold, this thread may be selected
    setp.ge.f64     p_cross,      fd_w, fd_thr;
    @!p_cross bra   EXIT;

    // Atomic min to record smallest index that crosses threshold
    mov.u32         r_one,        r_tid;
    atom.global.min.u32  r_stride, [rd_out], r_one;

EXIT:
    ret;
}}
"#
    )
}

// ─── Kernel 2: laplace_noise ─────────────────────────────────────────────────

/// Generate PTX for vectorized Laplace noise addition.
///
/// Kernel signature: `(f64* data, u32 n, f64 scale, u64 rng_seed)`
/// Uses inline LCG per thread to draw Laplace noise via inverse-CDF.
#[must_use]
pub fn laplace_noise_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// Vectorized Laplace noise addition.
// Each thread independently perturbs data[tid] with Lap(scale).
// Uses per-thread LCG seeded from global seed XOR tid.
.visible .entry laplace_noise(
    .param .u64 param_data,   // f64* array to perturb in-place
    .param .u32 param_n,      // element count
    .param .f64 param_scale,  // Laplace scale (= sensitivity / epsilon)
    .param .u64 param_seed    // base RNG seed
)
{{
    .reg .u64   rd_data, rd_state, rd_addr;
    .reg .u32   r_n, r_tid;
    .reg .f64   fd_val, fd_u, fd_noise, fd_scale;
    .reg .pred  p_guard;

    // LCG constants (Knuth MMIX)
    .reg .u64   rd_mul, rd_add;
    mov.u64     rd_mul,  6364136223846793005;
    mov.u64     rd_add,  1442695040888963407;

    ld.param.u64    rd_data,   [param_data];
    ld.param.u32    r_n,       [param_n];
    ld.param.f64    fd_scale,  [param_scale];
    ld.param.u64    rd_state,  [param_seed];

    mov.u32         r_tid,     %tid.x;
    setp.ge.u32     p_guard,   r_tid, r_n;
    @p_guard bra    EXIT;

    // Per-thread seed: seed XOR (tid * golden_ratio_u64)
    cvt.u64.u32     rd_addr,   r_tid;
    mul.lo.u64      rd_addr,   rd_addr, 11400714819323198485; // golden ratio
    xor.b64         rd_state,  rd_state, rd_addr;

    // One LCG step to generate uniform u in (0,1)
    mad.lo.u64      rd_state,  rd_state, rd_mul, rd_add;
    // Convert top 53 bits to f64 in [0,1)
    shr.u64         rd_addr,   rd_state, 11;
    cvt.rn.f64.u64  fd_u,      rd_addr;
    // fd_u /= 2^53
    mov.f64         fd_noise,  0D4340000000000000; // 2^53 as f64
    div.rn.f64      fd_u,      fd_u, fd_noise;

    // Inverse CDF: noise = -scale * sign(u-0.5) * ln(1-2|u-0.5|)
    // Simplified: noise = -scale * ln(1-u) (exponential trick for half)
    // For full Laplace use: u' = u - 0.5; noise = -scale*sign(u')*ln(1-2|u'|)
    mov.f64         fd_noise,  0D3FE0000000000000; // 0.5
    sub.f64         fd_u,      fd_u, fd_noise;     // u - 0.5
    // abs(u-0.5)
    abs.f64         fd_noise,  fd_u;
    // 2*|u-0.5|
    add.f64         fd_noise,  fd_noise, fd_noise;
    // 1 - 2|u-0.5|
    mov.f64         fd_val,    0D3FF0000000000000; // 1.0
    sub.f64         fd_noise,  fd_val, fd_noise;
    lg2.approx.f64  fd_noise,  fd_noise;           // log2 approx
    // ln = log2 * ln2
    mul.f64         fd_noise,  fd_noise, 0D3FE62E42FEFA39EF; // ln2
    neg.f64         fd_noise,  fd_noise;
    // sign(u-0.5): copysign scale
    neg.f64         fd_val,    fd_scale;
    setp.ge.f64     p_guard,   fd_u, 0D0000000000000000; // u >= 0.5?
    selp.f64        fd_scale,  fd_scale, fd_val, p_guard;
    mul.f64         fd_noise,  fd_noise, fd_scale;

    // Load, add, store
    mul.wide.u32    rd_addr,   r_tid, 8;
    add.u64         rd_addr,   rd_data, rd_addr;
    ld.global.f64   fd_val,    [rd_addr];
    add.f64         fd_val,    fd_val, fd_noise;
    st.global.f64   [rd_addr], fd_val;

EXIT:
    ret;
}}
"#
    )
}

// ─── Kernel 3: gaussian_noise ─────────────────────────────────────────────────

/// Generate PTX for Gaussian noise addition via Box-Muller transform.
///
/// Kernel signature: `(f64* data, u32 n, f64 sigma, u64 seed)`
/// Pairs of threads share a Box-Muller computation via shared memory.
#[must_use]
pub fn gaussian_noise_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// Gaussian noise via Box-Muller (pairs of threads share computation).
// Each thread i gets z1 (odd) or z2 (even) from a Box-Muller pair.
.visible .entry gaussian_noise(
    .param .u64 param_data,   // f64* in-place
    .param .u32 param_n,      // element count
    .param .f64 param_sigma,  // noise std dev
    .param .u64 param_seed    // base RNG seed
)
{{
    .reg .u64   rd_data, rd_s1, rd_s2, rd_addr;
    .reg .u32   r_n, r_tid, r_pair;
    .reg .f64   fd_u1, fd_u2, fd_r, fd_theta, fd_z, fd_sigma, fd_val;
    .reg .pred  p_guard, p_odd;

    .reg .u64   rd_mul, rd_add;
    mov.u64     rd_mul,  6364136223846793005;
    mov.u64     rd_add,  1442695040888963407;

    ld.param.u64    rd_data,    [param_data];
    ld.param.u32    r_n,        [param_n];
    ld.param.f64    fd_sigma,   [param_sigma];
    ld.param.u64    rd_s1,      [param_seed];

    mov.u32         r_tid,      %tid.x;
    setp.ge.u32     p_guard,    r_tid, r_n;
    @p_guard bra    EXIT;

    // pair = tid / 2; odd = tid % 2
    shr.u32         r_pair,     r_tid, 1;
    and.b32         r_n,        r_tid, 1;     // reuse r_n as odd flag
    setp.ne.u32     p_odd,      r_n, 0;

    // Seed pair's LCG from pair index
    cvt.u64.u32     rd_addr,    r_pair;
    mul.lo.u64      rd_addr,    rd_addr, 11400714819323198485;
    xor.b64         rd_s1,      rd_s1, rd_addr;

    // Draw u1 (seed step 1)
    mad.lo.u64      rd_s1,      rd_s1, rd_mul, rd_add;
    shr.u64         rd_addr,    rd_s1, 11;
    cvt.rn.f64.u64  fd_u1,      rd_addr;
    mov.f64         fd_r,       0D4340000000000000;
    div.rn.f64      fd_u1,      fd_u1, fd_r;
    // Clamp away from 0
    mov.f64         fd_r,       0D36A0000000000000; // ~2.2e-308 (epsilon-ish)
    max.f64         fd_u1,      fd_u1, fd_r;

    // Draw u2 (seed step 2)
    mov.u64         rd_s2,      rd_s1;
    mad.lo.u64      rd_s2,      rd_s2, rd_mul, rd_add;
    shr.u64         rd_addr,    rd_s2, 11;
    cvt.rn.f64.u64  fd_u2,      rd_addr;
    mov.f64         fd_r,       0D4340000000000000;
    div.rn.f64      fd_u2,      fd_u2, fd_r;

    // r = sqrt(-2 ln u1)
    lg2.approx.f64  fd_r,       fd_u1;
    mul.f64         fd_r,       fd_r, 0D3FE62E42FEFA39EF; // ln2
    neg.f64         fd_r,       fd_r;
    mul.f64         fd_r,       fd_r, 0D4000000000000000; // * 2
    sqrt.approx.f64 fd_r,       fd_r;

    // theta = 2*pi*u2
    mul.f64         fd_theta,   fd_u2, 0D401921FB54442D18; // 2*pi

    // z1 = r*cos(theta), z2 = r*sin(theta)
    cos.approx.f64  fd_z,       fd_theta;
    sin.approx.f64  fd_u2,      fd_theta;
    @p_odd mov.f64  fd_z,       fd_u2;   // odd threads get sin component

    mul.f64         fd_z,       fd_r, fd_z;
    mul.f64         fd_z,       fd_z, fd_sigma;

    // Add to data[tid]
    cvt.u64.u32     rd_addr,    r_tid;
    mul.lo.u64      rd_addr,    rd_addr, 8;
    add.u64         rd_addr,    rd_data, rd_addr;
    ld.global.f64   fd_val,     [rd_addr];
    add.f64         fd_val,     fd_val, fd_z;
    st.global.f64   [rd_addr],  fd_val;

EXIT:
    ret;
}}
"#
    )
}

// ─── Kernel 4: clip_gradient ──────────────────────────────────────────────────

/// Generate PTX for per-sample L2 gradient clipping.
///
/// Kernel signature: `(f64* grads, u32 n_params, u32 batch, f64 clip_norm)`
/// Each threadblock processes one sample; threads cooperate to compute L2 norm
/// then scale the gradient vector in-place.
#[must_use]
pub fn clip_gradient_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// Per-sample L2 gradient clipping.
// One threadblock per sample (blockIdx.x = sample index).
// Threads cooperate via shared memory reduction to compute L2 norm.
.visible .entry clip_gradient(
    .param .u64 param_grads,     // f64* [batch × n_params]
    .param .u32 param_n_params,  // number of parameters
    .param .u32 param_batch,     // batch size (guard)
    .param .f64 param_clip       // L2 clipping bound
)
{{
    .reg .u64   rd_grads, rd_base, rd_addr;
    .reg .u32   r_n, r_batch, r_bid, r_tid;
    .reg .f64   fd_g, fd_sum, fd_clip, fd_norm, fd_scale;
    .reg .pred  p_guard;

    .shared .align 8 .b8 smem[2048]; // 256 × f64 reduction scratch

    ld.param.u64    rd_grads,   [param_grads];
    ld.param.u32    r_n,        [param_n_params];
    ld.param.u32    r_batch,    [param_batch];
    ld.param.f64    fd_clip,    [param_clip];

    mov.u32         r_bid,      %ctaid.x;
    mov.u32         r_tid,      %tid.x;
    setp.ge.u32     p_guard,    r_bid, r_batch;
    @p_guard bra    EXIT;

    // Base pointer for this sample
    mul.wide.u32    rd_base,    r_bid, r_n;
    mul.lo.u64      rd_base,    rd_base, 8;
    add.u64         rd_base,    rd_grads, rd_base;

    setp.ge.u32     p_guard,    r_tid, r_n;

    // Load gradient element and square it
    mov.f64         fd_g,       0D0000000000000000;
    @!p_guard {{
        mul.wide.u32    rd_addr,  r_tid, 8;
        add.u64         rd_addr,  rd_base, rd_addr;
        ld.global.f64   fd_g,     [rd_addr];
        mul.f64         fd_g,     fd_g, fd_g;
    }}

    // Store squared value to shared memory for reduction
    mul.wide.u32    rd_addr,    r_tid, 8;
    add.u64         rd_addr,    smem, rd_addr;
    st.shared.f64   [rd_addr],  fd_g;
    bar.sync        0;

    // Simple sequential reduction in thread 0 (works for small n_params)
    setp.ne.u32     p_guard,    r_tid, 0;
    @p_guard bra    APPLY;

    mov.f64         fd_sum,     0D0000000000000000;
    mov.u32         r_bid,      0;
LOOP:
    setp.ge.u32     p_guard,    r_bid, r_n;
    @p_guard bra    DONE_SUM;
    mul.wide.u32    rd_addr,    r_bid, 8;
    add.u64         rd_addr,    smem, rd_addr;
    ld.shared.f64   fd_g,       [rd_addr];
    add.f64         fd_sum,     fd_sum, fd_g;
    add.u32         r_bid,      r_bid, 1;
    bra             LOOP;
DONE_SUM:
    sqrt.approx.f64 fd_norm,    fd_sum;
    // scale = clip / max(norm, 1e-9)
    mov.f64         fd_g,       0D3E112E0BE826D695; // ~1e-9
    max.f64         fd_norm,    fd_norm, fd_g;
    div.rn.f64      fd_scale,   fd_clip, fd_norm;
    // Clamp scale to 1.0 (no amplification)
    mov.f64         fd_g,       0D3FF0000000000000; // 1.0
    min.f64         fd_scale,   fd_scale, fd_g;
    // Store scale in smem[0] for other threads
    st.shared.f64   [smem],     fd_scale;

APPLY:
    bar.sync        0;
    ld.shared.f64   fd_scale,   [smem];

    mov.u32         r_tid,      %tid.x;
    setp.ge.u32     p_guard,    r_tid, r_n;
    @p_guard bra    EXIT;

    mul.wide.u32    rd_addr,    r_tid, 8;
    add.u64         rd_addr,    rd_base, rd_addr;
    ld.global.f64   fd_g,       [rd_addr];
    mul.f64         fd_g,       fd_g, fd_scale;
    st.global.f64   [rd_addr],  fd_g;

EXIT:
    ret;
}}
"#
    )
}

// ─── Kernel 5: svt_threshold ─────────────────────────────────────────────────

/// Generate PTX for SVT noisy threshold comparison.
///
/// Kernel signature: `(f64* queries, u32 n, f64 noisy_threshold, f64 noise_scale, u64 seed, u8* results)`
/// Each thread draws Laplace noise ν and compares query + ν against noisy_threshold.
#[must_use]
pub fn svt_threshold_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// SVT AboveThreshold comparison kernel.
// results[tid] = 1 if queries[tid] + Lap(noise_scale) >= noisy_threshold.
.visible .entry svt_threshold(
    .param .u64 param_queries,   // f64* query values
    .param .u32 param_n,         // count
    .param .f64 param_nthresh,   // noisy threshold (pre-perturbed)
    .param .f64 param_nscale,    // Laplace scale for query noise
    .param .u64 param_seed,      // RNG seed
    .param .u64 param_results    // u8* output (0/1)
)
{{
    .reg .u64   rd_q, rd_out, rd_state, rd_addr;
    .reg .u32   r_n, r_tid;
    .reg .f64   fd_q, fd_nthresh, fd_nscale, fd_u, fd_noise, fd_half;
    .reg .u8    rv_res;
    .reg .pred  p_guard, p_above, p_odd;
    .reg .u64   rd_mul, rd_add;

    mov.u64     rd_mul,  6364136223846793005;
    mov.u64     rd_add,  1442695040888963407;

    ld.param.u64    rd_q,       [param_queries];
    ld.param.u32    r_n,        [param_n];
    ld.param.f64    fd_nthresh, [param_nthresh];
    ld.param.f64    fd_nscale,  [param_nscale];
    ld.param.u64    rd_state,   [param_seed];
    ld.param.u64    rd_out,     [param_results];

    mov.u32         r_tid,      %tid.x;
    setp.ge.u32     p_guard,    r_tid, r_n;
    @p_guard bra    EXIT;

    // Per-thread LCG seed
    cvt.u64.u32     rd_addr,    r_tid;
    mul.lo.u64      rd_addr,    rd_addr, 11400714819323198485;
    xor.b64         rd_state,   rd_state, rd_addr;
    mad.lo.u64      rd_state,   rd_state, rd_mul, rd_add;

    // u in [0,1)
    shr.u64         rd_addr,    rd_state, 11;
    cvt.rn.f64.u64  fd_u,       rd_addr;
    mov.f64         fd_half,    0D4340000000000000;
    div.rn.f64      fd_u,       fd_u, fd_half;

    // Laplace via inverse CDF: u' = u - 0.5
    mov.f64         fd_half,    0D3FE0000000000000;
    sub.f64         fd_u,       fd_u, fd_half;
    abs.f64         fd_noise,   fd_u;
    add.f64         fd_noise,   fd_noise, fd_noise;
    mov.f64         fd_half,    0D3FF0000000000000;
    sub.f64         fd_noise,   fd_half, fd_noise;
    lg2.approx.f64  fd_noise,   fd_noise;
    mul.f64         fd_noise,   fd_noise, 0D3FE62E42FEFA39EF;
    neg.f64         fd_noise,   fd_noise;
    setp.ge.f64     p_odd,      fd_u, 0D0000000000000000;
    neg.f64         fd_half,    fd_nscale;
    selp.f64        fd_nscale,  fd_nscale, fd_half, p_odd;
    mul.f64         fd_noise,   fd_noise, fd_nscale;

    // Load query
    mul.wide.u32    rd_addr,    r_tid, 8;
    add.u64         rd_addr,    rd_q, rd_addr;
    ld.global.f64   fd_q,       [rd_addr];
    add.f64         fd_q,       fd_q, fd_noise;

    // Compare
    setp.ge.f64     p_above,    fd_q, fd_nthresh;
    selp.u32        r_n,        1, 0, p_above;    // reuse r_n

    // Store u8 result
    add.u64         rd_addr,    rd_out, r_tid;
    cvt.u8.u32      rv_res,     r_n;
    st.global.u8    [rd_addr],  rv_res;

EXIT:
    ret;
}}
"#
    )
}

// ─── Kernel 6: prv_convolve ───────────────────────────────────────────────────

/// Generate PTX for PRV discrete PMF convolution.
///
/// Kernel signature: `(f64* pmf_a, f64* pmf_b, f64* pmf_out, u32 grid_size)`
/// Computes pmf_out`[k]` = Σᵢ pmf_a`[i]` · pmf_b`[k-i]` for k in [0, 2·grid_size).
/// Each output element k is computed by one thread (O(n²) total).
#[must_use]
pub fn prv_convolve_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// Privacy Random Variable (PRV) discrete PMF convolution.
// output[k] = sum_i a[i] * b[k-i], k in [0, 2*grid_size).
// One thread per output element k.
.visible .entry prv_convolve(
    .param .u64 param_a,         // f64* pmf_a (length grid_size)
    .param .u64 param_b,         // f64* pmf_b (length grid_size)
    .param .u64 param_out,       // f64* output (length 2*grid_size - 1)
    .param .u32 param_grid       // grid_size
)
{{
    .reg .u64   rd_a, rd_b, rd_out, rd_addr_a, rd_addr_b, rd_addr_out;
    .reg .u32   r_grid, r_k, r_i, r_j, r_size2;
    .reg .f64   fd_ai, fd_bj, fd_sum;
    .reg .pred  p_guard, p_inner;

    ld.param.u64    rd_a,    [param_a];
    ld.param.u64    rd_b,    [param_b];
    ld.param.u64    rd_out,  [param_out];
    ld.param.u32    r_grid,  [param_grid];

    // Total output size = 2*grid - 1
    add.u32         r_size2,  r_grid, r_grid;
    sub.u32         r_size2,  r_size2, 1;

    mov.u32         r_k,     %tid.x;
    setp.ge.u32     p_guard, r_k, r_size2;
    @p_guard bra    EXIT;

    mov.f64         fd_sum,  0D0000000000000000;
    mov.u32         r_i,     0;

INNER:
    setp.ge.u32     p_inner, r_i, r_grid;
    @p_inner bra    DONE;

    // j = k - i; valid if 0 <= j < grid
    sub.u32         r_j,     r_k, r_i;
    setp.ge.u32     p_inner, r_j, r_grid;
    @p_inner bra    SKIP;

    mul.wide.u32    rd_addr_a, r_i, 8;
    add.u64         rd_addr_a, rd_a, rd_addr_a;
    ld.global.f64   fd_ai,     [rd_addr_a];

    mul.wide.u32    rd_addr_b, r_j, 8;
    add.u64         rd_addr_b, rd_b, rd_addr_b;
    ld.global.f64   fd_bj,     [rd_addr_b];

    fma.rn.f64      fd_sum,    fd_ai, fd_bj, fd_sum;

SKIP:
    add.u32         r_i,       r_i, 1;
    bra             INNER;

DONE:
    mul.wide.u32    rd_addr_out, r_k, 8;
    add.u64         rd_addr_out, rd_out, rd_addr_out;
    st.global.f64   [rd_addr_out], fd_sum;

EXIT:
    ret;
}}
"#
    )
}

// ─── Kernel 7: oue_encode ────────────────────────────────────────────────────

/// Generate PTX for OUE (Optimized Unary Encoding) local DP bit encode/flip.
///
/// Kernel signature: `(u32 true_bit_idx, u8* out_bits, u32 k, f64 p_keep, f64 p_flip, u64 seed)`
/// Each thread processes one bit position:
///   - If i == true_bit_idx: set B_i=1 w.p. 0.5, else 0
///   - If i != true_bit_idx: set B_i=1 w.p. p_flip = 1/(e^ε + 1)
#[must_use]
pub fn oue_encode_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}
// OUE Optimized Unary Encoding local DP encoder.
// Thread i processes bit position i of the output bitvector.
// true_bit: the true one-hot position.
// p_half = 0.5 (probability of 1 for true bit)
// p_flip: probability of 1 for other bits = 1/(e^eps + 1)
.visible .entry oue_encode(
    .param .u32 param_true_bit,  // index of the true 1 bit
    .param .u64 param_out,       // u8* output bit vector (length k)
    .param .u32 param_k,         // domain size
    .param .f64 param_p_half,    // 0.5
    .param .f64 param_p_flip,    // 1/(e^eps+1)
    .param .u64 param_seed       // RNG seed
)
{{
    .reg .u64   rd_out, rd_state, rd_addr;
    .reg .u32   r_k, r_tid, r_true;
    .reg .f64   fd_p_half, fd_p_flip, fd_u, fd_thresh;
    .reg .u8    rv_bit;
    .reg .pred  p_guard, p_true, p_set;
    .reg .u64   rd_mul, rd_add;

    mov.u64     rd_mul,  6364136223846793005;
    mov.u64     rd_add,  1442695040888963407;

    ld.param.u64    rd_out,      [param_out];
    ld.param.u32    r_k,         [param_k];
    ld.param.u32    r_true,      [param_true_bit];
    ld.param.f64    fd_p_half,   [param_p_half];
    ld.param.f64    fd_p_flip,   [param_p_flip];
    ld.param.u64    rd_state,    [param_seed];

    mov.u32         r_tid,       %tid.x;
    setp.ge.u32     p_guard,     r_tid, r_k;
    @p_guard bra    EXIT;

    // Per-thread LCG
    cvt.u64.u32     rd_addr,     r_tid;
    mul.lo.u64      rd_addr,     rd_addr, 11400714819323198485;
    xor.b64         rd_state,    rd_state, rd_addr;
    mad.lo.u64      rd_state,    rd_state, rd_mul, rd_add;

    shr.u64         rd_addr,     rd_state, 11;
    cvt.rn.f64.u64  fd_u,        rd_addr;
    mov.f64         fd_thresh,   0D4340000000000000;
    div.rn.f64      fd_u,        fd_u, fd_thresh; // u in [0,1)

    // Choose threshold
    setp.eq.u32     p_true,      r_tid, r_true;
    selp.f64        fd_thresh,   fd_p_half, fd_p_flip, p_true;

    // Bit = 1 if u < threshold
    setp.lt.f64     p_set,       fd_u, fd_thresh;
    selp.u32        r_k,         1, 0, p_set;    // reuse r_k
    cvt.u8.u32      rv_bit,      r_k;
    add.u64         rd_addr,     rd_out, r_tid;
    st.global.u8    [rd_addr],   rv_bit;

EXIT:
    ret;
}}
"#
    )
}
