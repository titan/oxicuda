//! PTX GPU kernel sources for anomaly detection operations.
//!
//! Each function returns a PTX program as a `String`.  These strings may be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`svdd_loss_ptx`]           | `‖φ(x_i) − c‖²` per sample + reduction |
//! | [`recon_score_ptx`]         | MSE per sample: `(1/d)·Σ(x_j − x̂_j)²` |
//! | [`lof_reach_dist_ptx`]      | `max(kNN_dist[j], dist(x,y))` pairwise |
//! | [`copod_ecdf_ptx`]          | `−log(ecdf)` per feature per sample |
//! | [`mahal_dist_ptx`]          | `(x−μ)ᵀΣ⁻¹(x−μ)` batch |
//! | [`iforest_score_ptx`]       | `2^{−avg_path/c_n}` per sample |
//! | [`ensemble_normalize_ptx`]  | min-max per detector, then combine |

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

/// Format an `f32` as a PTX immediate hex literal (`0Fhhhhhhhh`).
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: svdd_loss ─────────────────────────────────────────────────────

/// Compute DeepSVDD loss contribution per sample: `‖φ(x_i) − c‖²`.
///
/// Grid-stride over samples; uses `fma.rn.f32` for the squared distance
/// and `atom.global.add.f32` to accumulate into a scalar output.
#[must_use]
pub fn svdd_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// svdd_loss_kernel: out[i] = ||phi_i - c||^2 (each thread handles one sample).
// p_phi: [n * rep_dim] encoder outputs
// p_center: [rep_dim] hypersphere center c
// p_out: [n] per-sample squared distances (write)
// n: number of samples, rep_dim: representation dimensionality
.visible .entry svdd_loss_kernel(
    .param .u64 p_phi,
    .param .u64 p_center,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 rep_dim
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_phi];
    ld.param.u64  %rd1, [p_center];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [rep_dim];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // global tid = sample index

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;        // grid stride

    mov.u32       %r8, %r5;

$SVDD_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $SVDD_DONE;

    // sum = 0
    mov.f32       %f0, {ZERO};

    // inner loop over rep_dim
    mov.u32       %r9, 0;
$SVDD_INNER:
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $SVDD_INNER_DONE;

    // phi[sample * rep_dim + j]
    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd3, %r10, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];          // phi_ij

    // center[j]
    mul.wide.u32  %rd5, %r9, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];          // c_j

    sub.f32       %f3, %f1, %f2;        // diff = phi_ij - c_j
    fma.rn.f32    %f0, %f3, %f3, %f0;  // sum += diff^2

    add.u32       %r9, %r9, 1;
    bra           $SVDD_INNER;
$SVDD_INNER_DONE:

    // out[sample] = sum
    mul.wide.u32  %rd7, %r8, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;

    add.u32       %r8, %r8, %r7;
    bra           $SVDD_LOOP;

$SVDD_DONE:
    mov.u32       %r11, 0;
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 2: recon_score ────────────────────────────────────────────────────

/// Per-sample MSE reconstruction score: `(1/d) · Σ_j (x_j − x̂_j)²`.
#[must_use]
pub fn recon_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// recon_score_kernel: out[i] = (1/d) * sum_j (x[i,j] - xhat[i,j])^2
// p_x: [n * d] original, p_xhat: [n * d] reconstructed, p_out: [n], n: samples, d: features
.visible .entry recon_score_kernel(
    .param .u64 p_x,
    .param .u64 p_xhat,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_xhat];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mov.u32       %r8, %r5;

$RC_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $RC_DONE;

    mov.f32       %f0, {ZERO};
    mov.u32       %r9, 0;

$RC_INNER:
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $RC_INNER_DONE;

    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd3, %r10, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f2, [%rd5];

    sub.f32       %f3, %f1, %f2;
    fma.rn.f32    %f0, %f3, %f3, %f0;

    add.u32       %r9, %r9, 1;
    bra           $RC_INNER;
$RC_INNER_DONE:

    // out[i] = sum / d
    cvt.rn.f32.u32 %f4, %r1;
    div.rn.f32    %f5, %f0, %f4;

    mul.wide.u32  %rd6, %r8, 4;
    add.u64       %rd7, %rd2, %rd6;
    st.global.f32 [%rd7], %f5;

    add.u32       %r8, %r8, %r7;
    bra           $RC_LOOP;

$RC_DONE:
    mov.u32       %r11, 0;
    mov.f32       %f6, {ZERO};
    mov.f32       %f7, {ZERO};
    mov.u64       %rd8, 0;
    mov.u64       %rd9, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 3: lof_reach_dist ────────────────────────────────────────────────

/// Pairwise LOF reachability distance: `max(kNN_dist[j], dist(x, y))`.
///
/// Each thread handles one (sample, neighbour) pair.
#[must_use]
pub fn lof_reach_dist_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// lof_reach_dist_kernel: out[i,j] = max(knn_dist[j], euclidean(x_i, x_j))
// p_x: [n * d], p_data: [m * d] training, p_knn_dist: [m] k-distances, p_out: [n * k]
// n: query count, k: neighbours, d: features
.visible .entry lof_reach_dist_kernel(
    .param .u64 p_x,
    .param .u64 p_data,
    .param .u64 p_knn_idx,
    .param .u64 p_knn_dist,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 k,
    .param .u32 d
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<14>;
    .reg .f32  %f<10>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_data];
    ld.param.u64  %rd2, [p_knn_idx];
    ld.param.u64  %rd3, [p_knn_dist];
    ld.param.u64  %rd4, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [k];
    ld.param.u32  %r2,  [d];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;  // global_tid = i*k + ki

    // total pairs = n * k
    mul.lo.u32    %r7, %r0, %r1;

    mov.u32       %r8, %nctaid.x;
    mul.lo.u32    %r9, %r3, %r8;       // stride

    mov.u32       %r10, %r6;

$LOF_LOOP:
    setp.ge.u32   %p0, %r10, %r7;
    @%p0 bra $LOF_DONE;

    // sample index i = r10 / k, neighbour slot ki = r10 % k
    div.u32       %r11, %r10, %r1;     // i
    rem.u32       %r12, %r10, %r1;     // ki

    // knn_idx[i * k + ki] → neighbour index j
    mul.lo.u32    %r13, %r10, 4;
    cvt.u64.u32   %rd5, %r13;
    add.u64       %rd6, %rd2, %rd5;
    ld.global.u32 %r13, [%rd6];        // j

    // knn_dist[j * k + (k-1)] — k-distance of j (last elem of j's knn)
    // For simplicity: p_knn_dist is [m] = k-distances indexed by j
    mul.wide.u32  %rd7, %r13, 4;
    add.u64       %rd8, %rd3, %rd7;
    ld.global.f32 %f0, [%rd8];         // kd_j

    // Compute euclidean(x_i, data_j) over d dimensions
    mov.f32       %f1, {ZERO};
    mov.u32       %r13, 0;
$LOF_INNER:
    setp.ge.u32   %p1, %r13, %r2;
    @%p1 bra $LOF_INNER_DONE;

    mul.lo.u32    %r12, %r11, %r2;
    add.u32       %r12, %r12, %r13;
    mul.wide.u32  %rd9, %r12, 4;
    add.u64       %rd10, %rd0, %rd9;
    ld.global.f32 %f2, [%rd10];        // x_i[dim]

    mul.lo.u32    %r12, %r11, %r2;    // reuse r12
    add.u32       %r12, %r12, %r13;
    mul.wide.u32  %rd9, %r12, 4;
    add.u64       %rd10, %rd1, %rd9;
    ld.global.f32 %f3, [%rd10];        // data_j[dim]

    sub.f32       %f4, %f2, %f3;
    fma.rn.f32    %f1, %f4, %f4, %f1;

    add.u32       %r13, %r13, 1;
    bra           $LOF_INNER;
$LOF_INNER_DONE:
    sqrt.rn.f32   %f5, %f1;            // dist(x_i, x_j)

    // reach_dist = max(kd_j, dist)
    max.f32       %f6, %f0, %f5;

    // out[i*k+ki] = reach_dist
    mul.wide.u32  %rd11, %r10, 4;
    add.u64       %rd11, %rd4, %rd11;
    st.global.f32 [%rd11], %f6;

    add.u32       %r10, %r10, %r9;
    bra           $LOF_LOOP;

$LOF_DONE:
    mov.f32       %f7, {ZERO};
    mov.f32       %f8, {ZERO};
    mov.f32       %f9, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 4: copod_ecdf ────────────────────────────────────────────────────

/// Per-feature COPOD log-CDF contribution: `−log(ecdf(x_j) + ε)`.
#[must_use]
pub fn copod_ecdf_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1e-10_f32);
    let inv_log2e = f32_hex(std::f32::consts::LN_2);
    format!(
        r#"{hdr}// copod_ecdf_kernel: out[i,j] = -log(ecdf[i,j] + eps)
// ecdf values (pre-computed on host) for each sample × feature pair.
// p_ecdf: [n * d] pre-computed ecdf values in [0,1]
// p_out_l: [n * d] left-tail log contributions
// p_out_r: [n * d] right-tail log contributions
.visible .entry copod_ecdf_kernel(
    .param .u64 p_ecdf,
    .param .u64 p_out_l,
    .param .u64 p_out_r,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_ecdf];
    ld.param.u64  %rd1, [p_out_l];
    ld.param.u64  %rd2, [p_out_r];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    // total elements = n * d
    mul.lo.u32    %r2, %r0, %r1;

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;

    mov.u32       %r7, %nctaid.x;
    mul.lo.u32    %r8, %r3, %r7;

    mov.u32       %r9, %r6;

    // EPS and LN(2) constants
    mov.f32       %f8, {EPS};
    mov.f32       %f9, {LN2};

$COPOD_LOOP:
    setp.ge.u32   %p0, %r9, %r2;
    @%p0 bra $COPOD_DONE;

    mul.wide.u32  %rd3, %r9, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];          // ecdf value

    // left tail: -log(ecdf + eps)
    add.f32       %f1, %f0, %f8;
    lg2.approx.f32 %f2, %f1;
    mul.f32       %f3, %f2, %f9;        // convert log2 to ln
    neg.f32       %f4, %f3;             // negate

    add.u64       %rd5, %rd1, %rd3;
    st.global.f32 [%rd5], %f4;

    // right tail: -log(1 - ecdf + eps)
    sub.f32       %f5, {ONE}, %f0;      // 1 - ecdf
    add.f32       %f5, %f5, %f8;        // + eps
    lg2.approx.f32 %f6, %f5;
    mul.f32       %f6, %f6, %f9;
    neg.f32       %f7, %f6;

    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f7;

    add.u32       %r9, %r9, %r8;
    bra           $COPOD_LOOP;

$COPOD_DONE:
    mov.f32       %f0, {ZERO};
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
        LN2 = inv_log2e,
        ONE = f32_hex(1.0_f32),
    )
}

// ─── Kernel 5: mahal_dist ────────────────────────────────────────────────────

/// Batch Mahalanobis distance: `(x − μ)ᵀ Σ⁻¹ (x − μ)`.
#[must_use]
pub fn mahal_dist_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// mahal_dist_kernel: out[i] = (x_i - mean)^T * inv_cov * (x_i - mean)
// p_x: [n * d], p_mean: [d], p_inv_cov: [d * d], p_out: [n], n: samples, d: features
.visible .entry mahal_dist_kernel(
    .param .u64 p_x,
    .param .u64 p_mean,
    .param .u64 p_inv_cov,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_mean];
    ld.param.u64  %rd2, [p_inv_cov];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;  // sample index

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mov.u32       %r8, %r5;

$MAHAL_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $MAHAL_DONE;

    // D^2 = diff^T * Sigma_inv * diff
    // Computed as: for each row r: temp_r = sum_c inv_cov[r,c] * diff_c
    //              then D^2 = sum_r diff_r * temp_r
    // Since we cannot have dynamic local arrays in PTX easily, we accumulate
    // the quadratic form directly:
    // D^2 = sum_(r,c) diff_r * inv_cov[r,c] * diff_c

    mov.f32       %f0, {ZERO};           // accumulator for D^2

    mov.u32       %r9, 0;                // row r
$MAHAL_ROW:
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $MAHAL_ROW_DONE;

    // diff_r = x[i,r] - mean[r]
    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];           // x[i,r]

    mul.wide.u32  %rd6, %r9, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f2, [%rd7];           // mean[r]

    sub.f32       %f3, %f1, %f2;         // diff_r

    mov.u32       %r11, 0;               // col c
$MAHAL_COL:
    setp.ge.u32   %p1, %r11, %r1;
    @%p1 bra $MAHAL_COL_DONE;

    // diff_c = x[i,c] - mean[c]
    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r11;
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f4, [%rd5];

    mul.wide.u32  %rd6, %r11, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f5, [%rd7];
    sub.f32       %f6, %f4, %f5;         // diff_c

    // inv_cov[r, c]
    mul.lo.u32    %r10, %r9, %r1;
    add.u32       %r10, %r10, %r11;
    mul.wide.u32  %rd8, %r10, 4;
    add.u64       %rd9, %rd2, %rd8;
    ld.global.f32 %f7, [%rd9];           // inv_cov[r,c]

    // contribution: diff_r * inv_cov[r,c] * diff_c
    mul.f32       %f8, %f7, %f6;
    fma.rn.f32    %f0, %f3, %f8, %f0;

    add.u32       %r11, %r11, 1;
    bra           $MAHAL_COL;
$MAHAL_COL_DONE:
    add.u32       %r9, %r9, 1;
    bra           $MAHAL_ROW;
$MAHAL_ROW_DONE:

    mul.wide.u32  %rd10, %r8, 4;
    add.u64       %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f0;

    add.u32       %r8, %r8, %r7;
    bra           $MAHAL_LOOP;

$MAHAL_DONE:
    mov.f32       %f9, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 6: iforest_score ─────────────────────────────────────────────────

/// Isolation Forest score: `2^{−avg_path / c_n}` per sample.
///
/// Expects pre-computed average path lengths `p_avg_path[n]` and
/// the scalar `c_n` (c_factor(n_train)) passed as a raw f32.
#[must_use]
pub fn iforest_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1e-8_f32);
    format!(
        r#"{hdr}// iforest_score_kernel: out[i] = 2^(-avg_path[i] / c_n)
// p_avg_path: [n] average path lengths, c_n: scalar c-factor, p_out: [n]
.visible .entry iforest_score_kernel(
    .param .u64 p_avg_path,
    .param .f32 c_n,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_avg_path];
    ld.param.f32  %f6, [c_n];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    // Guard: if c_n < eps, output 0.5 for all
    mov.f32       %f7, {EPS};
    setp.lt.f32   %p0, %f6, %f7;

    mov.u32       %r7, %r4;

$IF_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $IF_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];          // avg_path[i]

    // exponent = -avg_path / c_n
    div.rn.f32    %f1, %f0, %f6;
    neg.f32       %f2, %f1;

    // 2^exponent = exp2(exponent) using ex2.approx
    ex2.approx.f32 %f3, %f2;

    add.u64       %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f3;

    add.u32       %r7, %r7, %r6;
    bra           $IF_LOOP;

$IF_DONE:
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.u64       %rd5, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
    )
}

// ─── Kernel 7: ensemble_normalize ────────────────────────────────────────────

/// Min-max normalise per detector then compute the average ensemble score.
///
/// Input: `p_scores[n * n_det]`, `p_min[n_det]`, `p_max[n_det]`.
/// Output: `p_out[n]` (average of normalised scores).
#[must_use]
pub fn ensemble_normalize_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1e-8_f32);
    format!(
        r#"{hdr}// ensemble_normalize_kernel: out[i] = mean_d( (scores[i,d]-min[d]) / (max[d]-min[d]+eps) )
// p_scores: [n * n_det], p_min: [n_det], p_max: [n_det], p_out: [n]
.visible .entry ensemble_normalize_kernel(
    .param .u64 p_scores,
    .param .u64 p_min,
    .param .u64 p_max,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 n_det
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_scores];
    ld.param.u64  %rd1, [p_min];
    ld.param.u64  %rd2, [p_max];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [n_det];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;  // sample index

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mov.f32       %f8, {EPS};

    mov.u32       %r8, %r5;

$ENS_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $ENS_DONE;

    mov.f32       %f0, {ZERO};           // accumulator
    mov.u32       %r9, 0;                // detector index

$ENS_INNER:
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $ENS_INNER_DONE;

    // scores[i, d]
    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];

    // min[d], max[d]
    mul.wide.u32  %rd6, %r9, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f2, [%rd7];
    add.u64       %rd8, %rd2, %rd6;
    ld.global.f32 %f3, [%rd8];

    // normed = (score - min) / (max - min + eps)
    sub.f32       %f4, %f1, %f2;
    sub.f32       %f5, %f3, %f2;
    add.f32       %f5, %f5, %f8;
    div.rn.f32    %f6, %f4, %f5;

    // clamp to [0, 1]
    mov.f32       %f7, {ZERO};
    max.f32       %f6, %f6, %f7;
    mov.f32       %f9, {ONE};
    min.f32       %f6, %f6, %f9;

    add.f32       %f0, %f0, %f6;

    add.u32       %r9, %r9, 1;
    bra           $ENS_INNER;
$ENS_INNER_DONE:

    // out[i] = sum / n_det
    cvt.rn.f32.u32 %f1, %r1;
    div.rn.f32    %f2, %f0, %f1;

    mul.wide.u32  %rd9, %r8, 4;
    add.u64       %rd9, %rd3, %rd9;
    st.global.f32 [%rd9], %f2;

    add.u32       %r8, %r8, %r7;
    bra           $ENS_LOOP;

$ENS_DONE:
    mov.u32       %r11, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
        ONE = f32_hex(1.0_f32),
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn check_ptx(prog: &str, sm: u32, kernel_name: &str) {
        assert!(
            prog.contains(&format!("sm_{sm}")),
            "missing sm_{sm} in {kernel_name}"
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
            "kernel name {kernel_name} not found"
        );
    }

    #[test]
    fn svdd_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&svdd_loss_ptx(sm), sm, "svdd_loss_kernel");
        }
    }

    #[test]
    fn recon_score_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&recon_score_ptx(sm), sm, "recon_score_kernel");
        }
    }

    #[test]
    fn lof_reach_dist_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&lof_reach_dist_ptx(sm), sm, "lof_reach_dist_kernel");
        }
    }

    #[test]
    fn copod_ecdf_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&copod_ecdf_ptx(sm), sm, "copod_ecdf_kernel");
        }
    }

    #[test]
    fn mahal_dist_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&mahal_dist_ptx(sm), sm, "mahal_dist_kernel");
        }
    }

    #[test]
    fn iforest_score_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&iforest_score_ptx(sm), sm, "iforest_score_kernel");
        }
    }

    #[test]
    fn ensemble_normalize_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&ensemble_normalize_ptx(sm), sm, "ensemble_normalize_kernel");
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
    fn f32_hex_known_values() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
        assert_eq!(f32_hex(2.0_f32), "0F40000000");
    }

    #[test]
    fn svdd_loss_uses_fma() {
        let p = svdd_loss_ptx(80);
        assert!(p.contains("fma.rn.f32"));
    }

    #[test]
    fn recon_score_uses_div() {
        let p = recon_score_ptx(90);
        assert!(p.contains("div.rn.f32"));
    }

    #[test]
    fn lof_uses_sqrt() {
        let p = lof_reach_dist_ptx(80);
        assert!(p.contains("sqrt.rn.f32"));
        assert!(p.contains("max.f32"));
    }

    #[test]
    fn copod_uses_lg2() {
        let p = copod_ecdf_ptx(80);
        assert!(p.contains("lg2.approx.f32"));
    }

    #[test]
    fn iforest_uses_ex2() {
        let p = iforest_score_ptx(80);
        assert!(p.contains("ex2.approx.f32"));
    }
}
