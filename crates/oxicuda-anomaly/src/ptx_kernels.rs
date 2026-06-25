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
//! | [`fused_knn_lof_ptx`]       | fused kNN min-reduction + reach-dist (sm_80+) |
//! | [`abod_batch_ptx`]          | batched angle-variance ABOF over query batches |
//! | [`fast_mcd_cstep_ptx`]      | device-side MCD C-step Mahalanobis for all `n` |

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

// ─── Kernel 8: fused_knn_lof ─────────────────────────────────────────────────

/// Fused k-nearest-neighbour + LOF reachability kernel for `sm_80` and above.
///
/// One thread block cooperates on a single query point. Each thread walks a
/// grid-stride slice of the reference set, computing the squared Euclidean
/// distance to the query and maintaining a thread-local running minimum. The
/// per-thread minima are reduced with `shfl.sync.down.b32` warp shuffles into a
/// warp-level minimum, then the warp leaders combine through a small shared
/// staging buffer (`.shared` "scratch") into the block minimum `d(x, nn₁)`. The
/// same pass tracks the k-distance of the matched neighbour from the
/// pre-computed `p_knn_dist[m]` table and forms the LOF reachability distance
/// `reach = max(knn_dist[nn], d(x, nn₁))` in shared memory, so the k-NN min
/// reduction and the reach-distance update are fused into a single launch.
///
/// The `cp.async`-free shared staging buffer is intentionally one warp wide
/// (32 slots) to stay bank-conflict free on Ampere/Hopper; the warp-shuffle
/// reduction uses `shfl.sync.down.b32` (requires `sm_80+`).
///
/// Layout:
/// * `p_query`     `[d]` — single query vector.
/// * `p_data`      `[m * d]` — reference set (row-major).
/// * `p_knn_dist`  `[m]` — pre-computed reference k-distances.
/// * `p_out`       `[2]` — `out[0] = d(x, nn₁)`, `out[1] = reach-distance`.
/// * `m` reference count, `d` feature dimensionality.
#[must_use]
pub fn fused_knn_lof_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let big = f32_hex(1.0e30_f32);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// fused_knn_lof_kernel (sm_80+): block-cooperative nearest-neighbour search
// fused with the LOF reachability-distance update in shared memory.
// out[0] = min_j dist(query, data_j); out[1] = max(knn_dist[argmin], out[0]).
.visible .entry fused_knn_lof_kernel(
    .param .u64 p_query,
    .param .u64 p_data,
    .param .u64 p_knn_dist,
    .param .u64 p_out,
    .param .u32 m,
    .param .u32 d
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<16>;
    .reg .pred %p0, %p1, %p2;

    // 32-slot shared staging buffers: one per warp lane, bank-conflict free.
    .shared .align 4 .f32 s_dist[32];
    .shared .align 4 .u32 s_idx[32];

    ld.param.u64  %rd0, [p_query];
    ld.param.u64  %rd1, [p_data];
    ld.param.u64  %rd2, [p_knn_dist];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [m];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %tid.x;            // lane within block
    mov.u32       %r3, %ntid.x;           // block width (grid stride)

    // thread-local running min and its reference index
    mov.f32       %f0, {BIG};             // best squared distance
    mov.u32       %r4, 0;                 // best reference index

    mov.u32       %r5, %r2;               // reference index walked by this thread
$FKL_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $FKL_LOOP_DONE;

    // squared Euclidean distance query .. data[r5]
    mov.f32       %f1, {ZERO};
    mul.lo.u32    %r6, %r5, %r1;          // row offset = r5 * d
    mov.u32       %r7, 0;                 // dimension
$FKL_DIM:
    setp.ge.u32   %p1, %r7, %r1;
    @%p1 bra $FKL_DIM_DONE;

    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f2, [%rd5];            // query[dim]

    add.u32       %r8, %r6, %r7;
    mul.wide.u32  %rd6, %r8, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f3, [%rd7];            // data[r5, dim]

    sub.f32       %f4, %f2, %f3;
    fma.rn.f32    %f1, %f4, %f4, %f1;

    add.u32       %r7, %r7, 1;
    bra           $FKL_DIM;
$FKL_DIM_DONE:

    setp.lt.f32   %p2, %f1, %f0;
    @%p2 mov.f32  %f0, %f1;
    @%p2 mov.u32  %r4, %r5;

    add.u32       %r5, %r5, %r3;
    bra           $FKL_LOOP;
$FKL_LOOP_DONE:

    // ── intra-warp min reduction via shfl.sync.down.b32 (sm_80+) ──
    mov.u32       %r9, 16;                // offset
$FKL_WARP:
    setp.eq.u32   %p0, %r9, 0;
    @%p0 bra $FKL_WARP_DONE;
    // shuffle the partner's distance and index down the warp
    mov.b32       %r10, %f0;
    shfl.sync.down.b32 %r11, %r10, %r9, 31, -1;
    mov.b32       %f5, %r11;
    shfl.sync.down.b32 %r12, %r4, %r9, 31, -1;
    setp.lt.f32   %p1, %f5, %f0;
    @%p1 mov.f32  %f0, %f5;
    @%p1 mov.u32  %r4, %r12;
    shr.u32       %r9, %r9, 1;
    bra           $FKL_WARP;
$FKL_WARP_DONE:

    // lane 0 of each warp writes to shared staging
    and.b32       %r13, %r2, 31;          // lane id
    shr.u32       %r14, %r2, 5;           // warp id
    setp.ne.u32   %p0, %r13, 0;
    @%p0 bra $FKL_SKIP_STORE;
    mul.wide.u32  %rd8, %r14, 4;
    mov.u64       %rd9, s_dist;
    add.u64       %rd10, %rd9, %rd8;
    st.shared.f32 [%rd10], %f0;
    mov.u64       %rd11, s_idx;
    add.u64       %rd12, %rd11, %rd8;
    st.shared.u32 [%rd12], %r4;
$FKL_SKIP_STORE:
    bar.sync      0;

    // thread 0 reduces the per-warp minima sequentially
    setp.ne.u32   %p0, %r2, 0;
    @%p0 bra $FKL_DONE;

    // number of warps = ceil(ntid.x / 32)
    add.u32       %r15, %r3, 31;
    shr.u32       %r15, %r15, 5;          // warp count

    mov.u64       %rd9, s_dist;
    ld.shared.f32 %f0, [%rd9];            // warp 0 min
    mov.u64       %rd11, s_idx;
    ld.shared.u32 %r4, [%rd11];

    mov.u32       %r16, 1;
$FKL_FINAL:
    setp.ge.u32   %p1, %r16, %r15;
    @%p1 bra $FKL_FINAL_DONE;
    mul.wide.u32  %rd8, %r16, 4;
    add.u64       %rd10, %rd9, %rd8;
    ld.shared.f32 %f6, [%rd10];
    add.u64       %rd12, %rd11, %rd8;
    ld.shared.u32 %r17, [%rd12];
    setp.lt.f32   %p2, %f6, %f0;
    @%p2 mov.f32  %f0, %f6;
    @%p2 mov.u32  %r4, %r17;
    add.u32       %r16, %r16, 1;
    bra           $FKL_FINAL;
$FKL_FINAL_DONE:

    // out[0] = sqrt(min squared distance) = d(x, nn1)
    sqrt.rn.f32   %f7, %f0;
    st.global.f32 [%rd3], %f7;

    // reach = max(knn_dist[argmin], d(x, nn1)) — fused reach-distance update
    mul.wide.u32  %rd13, %r4, 4;
    add.u64       %rd14, %rd2, %rd13;
    ld.global.f32 %f8, [%rd14];           // knn_dist[argmin]
    max.f32       %f9, %f8, %f7;
    add.u64       %rd15, %rd3, 4;          // out[1]
    st.global.f32 [%rd15], %f9;

$FKL_DONE:
    mov.f32       %f10, {ZERO};
    mov.u32       %r18, 0;
    mov.u32       %r19, 0;
    ret;
}}
"#,
        BIG = big,
        ZERO = zero,
    )
}

// ─── Kernel 9: abod_batch ────────────────────────────────────────────────────

/// Batched Angle-Based Outlier Factor (ABOF) kernel.
///
/// One thread handles one query in the batch. For query `p` and every unordered
/// pair `{a, b}` of reference points it accumulates the reciprocal-distance
/// weighted three-vector inner product
///
/// ```text
/// f(a, b) = ⟨p − a, p − b⟩ / (‖p − a‖² · ‖p − b‖²)
/// ```
///
/// in a streaming (stream-k) fashion: running sums of `f` and `f²` are kept in
/// registers so the variance `ABOF = E[f²] − E[f]²` is produced in a single
/// pass with no intermediate storage. The reciprocal `1 / (‖p − a‖²·‖p − b‖²)`
/// is the reciprocal-distance weight; a guard adds `ε` to the denominator so
/// coincident reference points do not divide by zero. The kernel writes the
/// anomaly score `1 / (ABOF + ε)` so that higher means more anomalous, matching
/// the CPU [`crate::distance::abod::Abod`] convention.
///
/// Layout:
/// * `p_query` `[nq * d]` — batch of query vectors.
/// * `p_data`  `[m * d]` — reference set.
/// * `p_out`   `[nq]` — per-query anomaly scores.
/// * `nq` query count, `m` reference count, `d` feature dimensionality.
#[must_use]
pub fn abod_batch_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps = f32_hex(1.0e-10_f32);
    format!(
        r#"{hdr}// abod_batch_kernel: stream-k angle-variance ABOF over a batch of queries.
// out[q] = 1 / (Var_pairs[<p-a,p-b> / (||p-a||^2 ||p-b||^2)] + eps).
.visible .entry abod_batch_kernel(
    .param .u64 p_query,
    .param .u64 p_data,
    .param .u64 p_out,
    .param .u32 nq,
    .param .u32 m,
    .param .u32 d
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<24>;
    .reg .f32  %f<24>;
    .reg .pred %p0, %p1, %p2, %p3;

    ld.param.u64  %rd0, [p_query];
    ld.param.u64  %rd1, [p_data];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [nq];
    ld.param.u32  %r1,  [m];
    ld.param.u32  %r2,  [d];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;     // global query index

    mov.u32       %r7, %nctaid.x;
    mul.lo.u32    %r8, %r3, %r7;          // grid stride

    mov.f32       %f20, {EPS};

    mov.u32       %r9, %r6;
$ABB_QUERY:
    setp.ge.u32   %p0, %r9, %r0;
    @%p0 bra $ABB_QUERY_DONE;

    // running ABOF accumulators
    mov.f32       %f0, {ZERO};            // sum f
    mov.f32       %f1, {ZERO};            // sum f^2
    mov.u32       %r10, 0;                // pair count

    mul.lo.u32    %r11, %r9, %r2;         // query row offset

    mov.u32       %r12, 0;                // index a
$ABB_A:
    setp.ge.u32   %p1, %r12, %r1;
    @%p1 bra $ABB_A_DONE;

    add.u32       %r13, %r12, 1;          // index b = a + 1
$ABB_B:
    setp.ge.u32   %p2, %r13, %r1;
    @%p2 bra $ABB_B_DONE;

    // accumulate dot = <p-a, p-b>, na2 = ||p-a||^2, nb2 = ||p-b||^2
    mov.f32       %f2, {ZERO};            // dot
    mov.f32       %f3, {ZERO};            // na2
    mov.f32       %f4, {ZERO};            // nb2
    mul.lo.u32    %r14, %r12, %r2;        // a row offset
    mul.lo.u32    %r15, %r13, %r2;        // b row offset
    mov.u32       %r16, 0;                // dimension
$ABB_DIM:
    setp.ge.u32   %p3, %r16, %r2;
    @%p3 bra $ABB_DIM_DONE;

    add.u32       %r17, %r11, %r16;
    mul.wide.u32  %rd3, %r17, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f5, [%rd4];            // p[dim]

    add.u32       %r18, %r14, %r16;
    mul.wide.u32  %rd5, %r18, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f6, [%rd6];            // a[dim]

    add.u32       %r19, %r15, %r16;
    mul.wide.u32  %rd7, %r19, 4;
    add.u64       %rd8, %rd1, %rd7;
    ld.global.f32 %f7, [%rd8];            // b[dim]

    sub.f32       %f8, %f5, %f6;          // pa = p - a
    sub.f32       %f9, %f5, %f7;          // pb = p - b
    fma.rn.f32    %f2, %f8, %f9, %f2;     // dot += pa*pb
    fma.rn.f32    %f3, %f8, %f8, %f3;     // na2 += pa*pa
    fma.rn.f32    %f4, %f9, %f9, %f4;     // nb2 += pb*pb

    add.u32       %r16, %r16, 1;
    bra           $ABB_DIM;
$ABB_DIM_DONE:

    // f = dot / (na2 * nb2 + eps)  (reciprocal-distance weight)
    mul.f32       %f10, %f3, %f4;
    add.f32       %f10, %f10, %f20;
    div.rn.f32    %f11, %f2, %f10;

    add.f32       %f0, %f0, %f11;         // sum f
    fma.rn.f32    %f1, %f11, %f11, %f1;   // sum f^2
    add.u32       %r10, %r10, 1;

    add.u32       %r13, %r13, 1;
    bra           $ABB_B;
$ABB_B_DONE:
    add.u32       %r12, %r12, 1;
    bra           $ABB_A;
$ABB_A_DONE:

    // ABOF = E[f^2] - E[f]^2 ; out = 1 / (ABOF + eps)
    setp.eq.u32   %p1, %r10, 0;
    @%p1 bra $ABB_WRITE_ZERO;

    cvt.rn.f32.u32 %f12, %r10;            // pair count as f32
    div.rn.f32    %f13, %f0, %f12;        // mean f
    div.rn.f32    %f14, %f1, %f12;        // mean f^2
    mul.f32       %f15, %f13, %f13;
    sub.f32       %f16, %f14, %f15;       // variance
    max.f32       %f16, %f16, {ZERO};     // clamp >= 0
    add.f32       %f16, %f16, %f20;
    mov.f32       %f17, {ONE};
    div.rn.f32    %f18, %f17, %f16;       // 1 / (ABOF + eps)
    bra           $ABB_STORE;

$ABB_WRITE_ZERO:
    mov.f32       %f18, {ZERO};

$ABB_STORE:
    mul.wide.u32  %rd9, %r9, 4;
    add.u64       %rd10, %rd2, %rd9;
    st.global.f32 [%rd10], %f18;

    add.u32       %r9, %r9, %r8;
    bra           $ABB_QUERY;
$ABB_QUERY_DONE:
    mov.f32       %f19, {ZERO};
    mov.u32       %r20, 0;
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
        ONE = f32_hex(1.0_f32),
    )
}

// ─── Kernel 10: fast_mcd_cstep ───────────────────────────────────────────────

/// FastMCD C-step kernel: device-side Mahalanobis distance for all `n` points.
///
/// Replaces the host-side per-point loop in
/// [`crate::density::fast_mcd`]: in a single launch every sample receives its
/// squared Mahalanobis distance `(xᵢ − μ)ᵀ Σ⁻¹ (xᵢ − μ)` with respect to the
/// current robust location `μ` and the inverse robust scatter `Σ⁻¹`. One thread
/// owns one sample and evaluates the quadratic form directly as the double sum
/// `Σ_r Σ_c diffᵣ · Σ⁻¹[r,c] · diff_c`, so the whole C-step distance vector is
/// produced on the device and the host only performs the subsequent
/// partial-sort / h-subset selection.
///
/// Layout:
/// * `p_x`       `[n * d]` — samples (row-major).
/// * `p_mean`    `[d]` — current robust location `μ`.
/// * `p_inv_cov` `[d * d]` — inverse robust scatter `Σ⁻¹` (row-major).
/// * `p_out`     `[n]` — per-sample squared Mahalanobis distances.
/// * `n` sample count, `d` feature dimensionality.
#[must_use]
pub fn fast_mcd_cstep_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// fast_mcd_cstep_kernel: out[i] = (x_i - mean)^T * inv_cov * (x_i - mean)
// for every sample i in one launch (device-side MCD C-step distance pass).
.visible .entry fast_mcd_cstep_kernel(
    .param .u64 p_x,
    .param .u64 p_mean,
    .param .u64 p_inv_cov,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<14>;
    .reg .f32  %f<12>;
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
    mad.lo.u32    %r5, %r2, %r3, %r4;     // sample index

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;          // grid stride

    mov.u32       %r8, %r5;
$MCD_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $MCD_DONE;

    mov.f32       %f0, {ZERO};            // D^2 accumulator
    mov.u32       %r9, 0;                 // row r
$MCD_ROW:
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $MCD_ROW_DONE;

    // diff_r = x[i,r] - mean[r]
    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];
    mul.wide.u32  %rd6, %r9, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f2, [%rd7];
    sub.f32       %f3, %f1, %f2;          // diff_r

    mov.u32       %r11, 0;                // col c
$MCD_COL:
    setp.ge.u32   %p1, %r11, %r1;
    @%p1 bra $MCD_COL_DONE;

    // diff_c = x[i,c] - mean[c]
    mul.lo.u32    %r12, %r8, %r1;
    add.u32       %r12, %r12, %r11;
    mul.wide.u32  %rd8, %r12, 4;
    add.u64       %rd9, %rd0, %rd8;
    ld.global.f32 %f4, [%rd9];
    mul.wide.u32  %rd10, %r11, 4;
    add.u64       %rd11, %rd1, %rd10;
    ld.global.f32 %f5, [%rd11];
    sub.f32       %f6, %f4, %f5;          // diff_c

    // inv_cov[r,c]
    mul.lo.u32    %r12, %r9, %r1;
    add.u32       %r12, %r12, %r11;
    mul.wide.u32  %rd12, %r12, 4;
    add.u64       %rd13, %rd2, %rd12;
    ld.global.f32 %f7, [%rd13];

    // D^2 += diff_r * inv_cov[r,c] * diff_c
    mul.f32       %f8, %f7, %f6;
    fma.rn.f32    %f0, %f3, %f8, %f0;

    add.u32       %r11, %r11, 1;
    bra           $MCD_COL;
$MCD_COL_DONE:
    add.u32       %r9, %r9, 1;
    bra           $MCD_ROW;
$MCD_ROW_DONE:

    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f0;

    add.u32       %r8, %r8, %r7;
    bra           $MCD_LOOP;
$MCD_DONE:
    mov.f32       %f9, {ZERO};
    mov.u32       %r13, 0;
    ret;
}}
"#,
        ZERO = zero,
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

    #[test]
    fn fused_knn_lof_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&fused_knn_lof_ptx(sm), sm, "fused_knn_lof_kernel");
        }
    }

    #[test]
    fn fused_knn_lof_uses_shfl_and_shared() {
        // Warp-level distance min reduction + shared-memory reach-dist staging.
        let p = fused_knn_lof_ptx(80);
        assert!(p.contains("shfl.sync.down.b32"), "missing warp shuffle");
        assert!(p.contains(".shared"), "missing shared-memory staging");
        assert!(p.contains("bar.sync"), "missing block barrier");
        // fused reach-distance update: max(knn_dist, d(x,nn1))
        assert!(p.contains("max.f32"), "missing reach-dist max");
        assert!(p.contains("sqrt.rn.f32"), "missing distance sqrt");
    }

    #[test]
    fn abod_batch_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&abod_batch_ptx(sm), sm, "abod_batch_kernel");
        }
    }

    #[test]
    fn abod_batch_uses_fma_and_div() {
        // Three-vector inner product + reciprocal-distance weight + variance.
        let p = abod_batch_ptx(80);
        // three FMA accumulators: dot, na2, nb2 plus the f^2 running sum
        assert!(
            p.matches("fma.rn.f32").count() >= 4,
            "expected >=4 fma accumulations for the 3-vector inner product"
        );
        assert!(
            p.contains("div.rn.f32"),
            "missing reciprocal-distance weight"
        );
    }

    #[test]
    fn fast_mcd_cstep_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            check_ptx(&fast_mcd_cstep_ptx(sm), sm, "fast_mcd_cstep_kernel");
        }
    }

    #[test]
    fn fast_mcd_cstep_uses_quadratic_form() {
        // Double-sum quadratic form diff_r * inv_cov[r,c] * diff_c via fma.
        let p = fast_mcd_cstep_ptx(90);
        assert!(p.contains("fma.rn.f32"), "missing quadratic-form fma");
        assert!(
            p.contains("inv_cov"),
            "missing inverse-covariance comment/param"
        );
        assert!(p.contains("p_inv_cov"), "missing inv-cov parameter");
    }

    #[test]
    fn new_kernels_share_header() {
        // All three new kernels carry the unified PTX header for every SM.
        for sm in [75_u32, 80, 90, 120] {
            for prog in [
                fused_knn_lof_ptx(sm),
                abod_batch_ptx(sm),
                fast_mcd_cstep_ptx(sm),
            ] {
                assert!(prog.contains(".address_size 64"));
                assert!(prog.contains(&format!(".target sm_{sm}")));
            }
        }
    }
}
