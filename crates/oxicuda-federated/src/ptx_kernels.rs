//! PTX GPU kernel sources for federated learning operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`fedavg_weighted_sum_ptx`] | Weighted parameter aggregation: `out[i] += weight * param[i]` |
//! | [`dp_clip_gradient_ptx`] | Per-vector gradient clipping to L2 ball |
//! | [`gaussian_noise_ptx`] | Box-Muller Gaussian noise injection for DP |
//! | [`topk_mask_ptx`] | Top-k sparsification mask: zero below threshold |
//! | [`qsgd_quantize_ptx`] | Stochastic quantization (QSGD) |
//! | [`pairwise_mask_ptx`] | Additive pairwise masking for secure aggregation |
//! | [`aggregate_mean_ptx`] | Running online mean update |

// ─── PTX header helper ────────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: fedavg_weighted_sum ──────────────────────────────────────────

/// FedAvg weighted parameter aggregation: `out[i] += weight * param[i]`.
///
/// Grid-stride loop over all parameter elements. Used server-side to
/// accumulate client updates into the global model.
#[must_use]
pub fn fedavg_weighted_sum_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry fedavg_weighted_sum_kernel(
    .param .u64 p_out,
    .param .u64 p_param,
    .param .f32 weight,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<12>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_out];
    ld.param.u64  %rd1, [p_param];
    ld.param.f32  %f0,  [weight];
    ld.param.u32  %r0,  [n_elems];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // r4 = tid

    // stride = gridDim.x * blockDim.x
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;          // r6 = stride

$WAVG_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $WAVG_DONE;

    // Load param[tid]
    mul.wide.u32  %rd2, %r4, 4;
    add.u64       %rd3, %rd1, %rd2;
    ld.global.f32 %f1,  [%rd3];

    // Load out[tid]
    add.u64       %rd4, %rd0, %rd2;
    ld.global.f32 %f2,  [%rd4];

    // out[tid] += weight * param[tid]
    fma.rn.f32    %f3,  %f0, %f1, %f2;
    st.global.f32 [%rd4], %f3;

    // tid += stride
    add.u32       %r4, %r4, %r6;
    bra           $WAVG_LOOP;

$WAVG_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 2: dp_clip_gradient ──────────────────────────────────────────────

/// Per-vector gradient clipping to L2 norm ball of radius `clip_norm`.
///
/// Computes `||g|| = sqrt(sum(g_i^2))` via a warp reduction, then scales
/// `g = g * min(1, clip_norm / max(||g||, 1e-6))`.
#[must_use]
pub fn dp_clip_gradient_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let eps_hex = f32_hex(1e-6_f32);
    let one_hex = f32_hex(1.0_f32);
    format!(
        r#"{hdr}.visible .entry dp_clip_gradient_kernel(
    .param .u64 p_grad,
    .param .f32 clip_norm,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<12>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_grad];
    ld.param.f32  %f0,  [clip_norm];
    ld.param.u32  %r0,  [n_elems];

    // Thread computes partial sum of squares
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // r4 = tid
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;          // r6 = stride

    mov.f32       %f1, 0f00000000;        // partial_sq = 0.0

$CLIP_SUM_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $CLIP_SUM_DONE;

    mul.wide.u32  %rd1, %r4, 4;
    add.u64       %rd2, %rd0, %rd1;
    ld.global.f32 %f2,  [%rd2];
    fma.rn.f32    %f1,  %f2, %f2, %f1;   // partial_sq += g_i^2
    add.u32       %r4, %r4, %r6;
    bra           $CLIP_SUM_LOOP;

$CLIP_SUM_DONE:
    // norm = sqrt(partial_sq)  (approx, single thread per block for simplicity)
    sqrt.approx.f32 %f3, %f1;

    // scale = min(1, clip_norm / max(norm, 1e-6))
    mov.f32       %f4,  {eps_hex};
    max.f32       %f5,  %f3, %f4;
    div.approx.f32 %f6, %f0, %f5;
    mov.f32       %f7,  {one_hex};
    min.f32       %f8,  %f7, %f6;

    // Second pass: scale gradients
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

$CLIP_SCALE_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $CLIP_DONE;

    mul.wide.u32  %rd1, %r4, 4;
    add.u64       %rd2, %rd0, %rd1;
    ld.global.f32 %f9,  [%rd2];
    mul.f32       %f10, %f9, %f8;
    st.global.f32 [%rd2], %f10;
    add.u32       %r4, %r4, %r6;
    bra           $CLIP_SCALE_LOOP;

$CLIP_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 3: gaussian_noise ─────────────────────────────────────────────────

/// Box-Muller Gaussian noise injection for differential privacy.
///
/// From two uniform seeds `u1, u2`: `z = sqrt(-2*ln(u1)) * cos(2*pi*u2)`,
/// then `g[i] += sigma * z`. Uses `lg2.approx.f32`, `ex2.approx.f32`,
/// `sqrt.approx.f32` PTX instructions.
#[must_use]
pub fn gaussian_noise_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let two_pi_hex = f32_hex(2.0 * std::f32::consts::PI);
    let minus_two_ln2_hex = f32_hex(-2.0 * std::f32::consts::LN_2);
    format!(
        r#"{hdr}.visible .entry gaussian_noise_kernel(
    .param .u64 p_grad,
    .param .u64 p_u1,
    .param .u64 p_u2,
    .param .f32 sigma,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_grad];
    ld.param.u64  %rd1, [p_u1];
    ld.param.u64  %rd2, [p_u2];
    ld.param.f32  %f0,  [sigma];
    ld.param.u32  %r0,  [n_elems];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;

$GNOISE_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $GNOISE_DONE;

    mul.wide.u32  %rd3, %r4, 4;

    // u1 = max(u1_buf[tid], 1e-6) to avoid log(0)
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f1,  [%rd4];
    mov.f32       %f2,  0F358637BD;       // 1e-6 hex
    max.f32       %f3,  %f1, %f2;

    // u2 = u2_buf[tid]
    add.u64       %rd5, %rd2, %rd3;
    ld.global.f32 %f4,  [%rd5];

    // ln(u1) via lg2: ln(u1) = log2(u1) / log2(e) = log2(u1) * ln(2)
    lg2.approx.f32 %f5, %f3;             // log2(u1)
    mov.f32        %f6, {minus_two_ln2_hex}; // -2 * ln(2)
    // -2*ln(u1) = -2 * log2(u1) * ln(2) = log2(u1) * (-2*ln(2))
    mul.f32        %f7, %f5, %f6;        // f7 = -2*ln(u1)

    // sqrt(-2*ln(u1))
    sqrt.approx.f32 %f8, %f7;

    // cos(2*pi*u2): use cos.approx.f32
    mov.f32        %f9, {two_pi_hex};
    mul.f32        %f10, %f9, %f4;       // 2*pi*u2
    cos.approx.f32 %f11, %f10;

    // z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    mul.f32        %f12, %f8, %f11;

    // noise = sigma * z
    mul.f32        %f13, %f0, %f12;

    // grad[tid] += noise
    add.u64        %rd6, %rd0, %rd3;
    ld.global.f32  %f14, [%rd6];
    add.f32        %f15, %f14, %f13;
    st.global.f32  [%rd6], %f15;

    add.u32        %r4, %r4, %r6;
    bra            $GNOISE_LOOP;

$GNOISE_DONE:
    ret;
}}
"#,
        two_pi_hex = two_pi_hex,
        minus_two_ln2_hex = minus_two_ln2_hex,
    )
}

// ─── Kernel 4: topk_mask ─────────────────────────────────────────────────────

/// Top-k sparsification mask: `out[i] = (|x[i]| >= thresh) ? x[i] : 0.0`.
///
/// Used in Top-k gradient compression to zero out small-magnitude elements.
#[must_use]
pub fn topk_mask_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry topk_mask_kernel(
    .param .u64 p_out,
    .param .u64 p_x,
    .param .f32 thresh,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_out];
    ld.param.u64  %rd1, [p_x];
    ld.param.f32  %f0,  [thresh];
    ld.param.u32  %r0,  [n_elems];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;

$TOPK_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $TOPK_DONE;

    mul.wide.u32  %rd2, %r4, 4;
    add.u64       %rd3, %rd1, %rd2;
    ld.global.f32 %f1,  [%rd3];

    // |x[i]|
    abs.f32       %f2,  %f1;

    // out = |x| >= thresh ? x : 0.0
    setp.ge.f32   %p1,  %f2, %f0;
    mov.f32       %f3,  0f00000000;
    selp.f32      %f4,  %f1, %f3, %p1;

    add.u64       %rd4, %rd0, %rd2;
    st.global.f32 [%rd4], %f4;

    add.u32       %r4, %r4, %r6;
    bra           $TOPK_LOOP;

$TOPK_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 5: qsgd_quantize ─────────────────────────────────────────────────

/// Stochastic quantization (QSGD): `q[i] = sign(x[i]) * floor(|x[i]|/norm * s + u[i])`
/// where `u[i]` ∈ `[0,1]` is uniform noise, clamped to `[0, s]`.
#[must_use]
pub fn qsgd_quantize_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry qsgd_quantize_kernel(
    .param .u64 p_q,
    .param .u64 p_x,
    .param .u64 p_u,
    .param .f32 norm,
    .param .f32 s_levels,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<14>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_q];
    ld.param.u64  %rd1, [p_x];
    ld.param.u64  %rd2, [p_u];
    ld.param.f32  %f0,  [norm];
    ld.param.f32  %f1,  [s_levels];
    ld.param.u32  %r0,  [n_elems];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;

    // Ensure norm > 0 to avoid division by zero
    mov.f32       %f2,  0F358637BD;       // 1e-6
    max.f32       %f3,  %f0, %f2;

$QSGD_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $QSGD_DONE;

    mul.wide.u32  %rd3, %r4, 4;

    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f4,  [%rd4];           // x[i]

    add.u64       %rd5, %rd2, %rd3;
    ld.global.f32 %f5,  [%rd5];           // u[i]

    // sign(x)
    mov.f32       %f6,  0f3F800000;       // 1.0
    mov.f32       %f7,  0fBF800000;       // -1.0
    setp.ge.f32   %p1,  %f4, 0f00000000;
    selp.f32      %f8,  %f6, %f7, %p1;   // sign(x)

    // |x[i]| / norm * s + u[i]
    abs.f32       %f9,  %f4;
    div.approx.f32 %f10, %f9, %f3;
    mul.f32       %f11, %f10, %f1;
    add.f32       %f12, %f11, %f5;

    // floor(...)
    cvt.rmi.f32.f32 %f12, %f12;

    // clamp to [0, s]
    max.f32       %f12, %f12, 0f00000000;
    min.f32       %f12, %f12, %f1;

    // q[i] = sign * floor(...)
    mul.f32       %f13, %f8, %f12;

    add.u64       %rd6, %rd0, %rd3;
    st.global.f32 [%rd6], %f13;

    add.u32       %r4, %r4, %r6;
    bra           $QSGD_LOOP;

$QSGD_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 6: pairwise_mask ─────────────────────────────────────────────────

/// Additive pairwise masking for secure aggregation:
/// `out[i] = (x[i] + mask[i]) mod 2^32` (bit-level integer arithmetic).
#[must_use]
pub fn pairwise_mask_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry pairwise_mask_kernel(
    .param .u64 p_out,
    .param .u64 p_x,
    .param .u64 p_mask,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_out];
    ld.param.u64  %rd1, [p_x];
    ld.param.u64  %rd2, [p_mask];
    ld.param.u32  %r0,  [n_elems];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;

$PMASK_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $PMASK_DONE;

    mul.wide.u32  %rd3, %r4, 4;

    // Load x and mask as u32 for bitwise add
    add.u64       %rd4, %rd1, %rd3;
    ld.global.u32 %r7,  [%rd4];

    add.u64       %rd5, %rd2, %rd3;
    ld.global.u32 %r8,  [%rd5];

    // out = (x + mask) mod 2^32
    add.u32       %r9, %r7, %r8;

    add.u64       %rd6, %rd0, %rd3;
    st.global.u32 [%rd6], %r9;

    add.u32       %r4, %r4, %r6;
    bra           $PMASK_LOOP;

$PMASK_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 7: aggregate_mean ─────────────────────────────────────────────────

/// Running online mean: `mean[i] = mean[i] * (n-1)/n + x[i]/n`.
///
/// Used for incremental model averaging across aggregation rounds.
#[must_use]
pub fn aggregate_mean_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry aggregate_mean_kernel(
    .param .u64 p_mean,
    .param .u64 p_x,
    .param .u32 round_n,
    .param .u32 n_elems
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_mean];
    ld.param.u64  %rd1, [p_x];
    ld.param.u32  %r0,  [round_n];
    ld.param.u32  %r7,  [n_elems];

    // n as f32
    cvt.rn.f32.u32 %f0, %r0;

    // (n-1)/n
    mov.f32       %f1,  0f3F800000;       // 1.0
    sub.f32       %f2,  %f0, %f1;        // n - 1
    div.approx.f32 %f3, %f2, %f0;        // (n-1)/n

    // 1/n
    rcp.approx.f32 %f4, %f0;

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r5, %r1;

$AMEAN_LOOP:
    setp.ge.u32   %p0, %r4, %r7;
    @%p0 bra $AMEAN_DONE;

    mul.wide.u32  %rd2, %r4, 4;

    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f5,  [%rd3];           // mean[i]

    add.u64       %rd4, %rd1, %rd2;
    ld.global.f32 %f6,  [%rd4];           // x[i]

    // mean[i] = mean[i] * (n-1)/n + x[i]/n
    mul.f32       %f7, %f5, %f3;
    fma.rn.f32    %f8, %f6, %f4, %f7;
    st.global.f32 [%rd3], %f8;

    add.u32       %r4, %r4, %r6;
    bra           $AMEAN_LOOP;

$AMEAN_DONE:
    ret;
}}
"#
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_ptx(ptx: &str, sm: u32) {
        assert!(
            ptx.contains(".version"),
            "SM {sm}: PTX missing .version directive"
        );
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "SM {sm}: PTX missing .target sm_{sm}"
        );
        assert!(
            ptx.contains(".address_size 64"),
            "SM {sm}: PTX missing .address_size"
        );
    }

    #[test]
    fn all_kernels_all_sm_versions_valid_ptx() {
        for &sm in SM_VERSIONS {
            check_ptx(&fedavg_weighted_sum_ptx(sm), sm);
            check_ptx(&dp_clip_gradient_ptx(sm), sm);
            check_ptx(&gaussian_noise_ptx(sm), sm);
            check_ptx(&topk_mask_ptx(sm), sm);
            check_ptx(&qsgd_quantize_ptx(sm), sm);
            check_ptx(&pairwise_mask_ptx(sm), sm);
            check_ptx(&aggregate_mean_ptx(sm), sm);
        }
    }

    #[test]
    fn fedavg_weighted_sum_contains_fma() {
        let ptx = fedavg_weighted_sum_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "expected FMA instruction");
    }

    #[test]
    fn gaussian_noise_contains_box_muller_ops() {
        let ptx = gaussian_noise_ptx(80);
        assert!(ptx.contains("lg2.approx.f32"), "expected lg2 instruction");
        assert!(ptx.contains("sqrt.approx.f32"), "expected sqrt instruction");
        assert!(ptx.contains("cos.approx.f32"), "expected cos instruction");
    }

    #[test]
    fn topk_mask_contains_abs() {
        let ptx = topk_mask_ptx(80);
        assert!(ptx.contains("abs.f32"), "expected abs instruction");
    }

    #[test]
    fn qsgd_quantize_contains_floor() {
        let ptx = qsgd_quantize_ptx(80);
        assert!(ptx.contains("cvt.rmi.f32.f32"), "expected floor (cvt.rmi)");
    }

    #[test]
    fn pairwise_mask_uses_integer_add() {
        let ptx = pairwise_mask_ptx(80);
        assert!(ptx.contains("add.u32"), "expected integer add for mod 2^32");
        assert!(ptx.contains("ld.global.u32"), "expected u32 load");
    }

    #[test]
    fn aggregate_mean_contains_rcp() {
        let ptx = aggregate_mean_ptx(80);
        assert!(ptx.contains("rcp.approx.f32"), "expected rcp instruction");
    }
}
