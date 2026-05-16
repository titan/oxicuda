//! GPU PTX kernels for compressed-sensing operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM≥100 → 8.7 (Blackwell), SM≥90 → 8.4 (Hopper),
//!     SM≥80  → 8.0 (Ampere),    else → 7.5 (Turing).
//!
//! IMPORTANT: PTX kernel bodies use **string concatenation** (NOT `format!()`) for
//! sections containing `%rd`, `%r`, `%f`, `%fd` register names, which Rust's format macro
//! would misinterpret as unused format arguments in edition 2024.

/// Build a PTX file header string for the given SM version.
fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Compute correlations `c[j] = sum_i phi[i, j] * r[i]` for `Φᵀ r`.
///
/// Signature: `correlate_kernel(c, phi, r, m, n)` where phi is row-major m×n.
/// Grid = (ceil(n/256), 1, 1), Block = (256, 1, 1).
#[must_use]
pub fn correlate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry correlate_kernel(\n\
        .param .u64 p_c,\n\
        .param .u64 p_phi,\n\
        .param .u64 p_r,\n\
        .param .u32 p_m,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_c];\n\
        ld.param.u64  %rd1, [p_phi];\n\
        ld.param.u64  %rd2, [p_r];\n\
        ld.param.u32  %r0,  [p_m];\n\
        ld.param.u32  %r1,  [p_n];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $CR_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r6, 0;\n\
    \n\
    $CR_LOOP:\n\
        setp.ge.u32   %p0, %r6, %r0;\n\
        @%p0 bra $CR_WRITE;\n\
    \n\
        // phi[i, j] index = i * n + j\n\
        mul.lo.u32    %r7, %r6, %r1;\n\
        add.u32       %r7, %r7, %r5;\n\
        mul.wide.u32  %rd3, %r7, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
    \n\
        // r[i]\n\
        mul.wide.u32  %rd5, %r6, 4;\n\
        add.u64       %rd6, %rd2, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
    \n\
        add.u32       %r6, %r6, 1;\n\
        bra $CR_LOOP;\n\
    \n\
    $CR_WRITE:\n\
        mul.wide.u32  %rd7, %r5, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        st.global.f32 [%rd8], %f0;\n\
    \n\
    $CR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Hard threshold: `out[i] = x[i]` if `|x[i]| > threshold` else 0.
///
/// Signature: `hard_threshold_kernel(out, x, threshold, n)`.
/// (Host computes `threshold` via partial sort to keep top-K.)
#[must_use]
pub fn hard_threshold_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry hard_threshold_kernel(\n\
        .param .u64 p_out,\n\
        .param .u64 p_x,\n\
        .param .f32 p_threshold,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_out];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.f32  %f0,  [p_threshold];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $HT_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        abs.f32       %f2, %f1;\n\
        setp.gt.f32   %p0, %f2, %f0;\n\
        selp.f32      %f3, %f1, 0f00000000, %p0;\n\
    \n\
        add.u64       %rd4, %rd0, %rd2;\n\
        st.global.f32 [%rd4], %f3;\n\
    \n\
    $HT_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Element-wise soft-thresholding: `y[i] = sign(x[i]) * max(|x[i]| - lambda, 0)`.
///
/// Signature: `soft_threshold_kernel(y, x, lambda, n)`.
#[must_use]
pub fn soft_threshold_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry soft_threshold_kernel(\n\
        .param .u64 p_y,\n\
        .param .u64 p_x,\n\
        .param .f32 p_lambda,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_y];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.f32  %f0,  [p_lambda];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $ST_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        abs.f32       %f2, %f1;\n\
        sub.f32       %f3, %f2, %f0;\n\
        max.f32       %f4, %f3, 0f00000000;\n\
        setp.lt.f32   %p0, %f1, 0f00000000;\n\
        neg.f32       %f5, %f4;\n\
        selp.f32      %f6, %f5, %f4, %p0;\n\
    \n\
        add.u64       %rd4, %rd0, %rd2;\n\
        st.global.f32 [%rd4], %f6;\n\
    \n\
    $ST_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// IHT update step: `x[i] = x[i] + mu * Phi^T (y - Phi x)[i]` then host hard-thresholds.
///
/// Signature: `iht_step_kernel(x, grad, mu, n)` where grad = Phi^T (y - Phi x) is precomputed.
#[must_use]
pub fn iht_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry iht_step_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_grad,\n\
        .param .f32 p_mu,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_grad];\n\
        ld.param.f32  %f0,  [p_mu];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $IH_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        add.u64       %rd4, %rd1, %rd2;\n\
        ld.global.f32 %f2, [%rd4];\n\
    \n\
        fma.rn.f32    %f3, %f0, %f2, %f1;\n\
        st.global.f32 [%rd3], %f3;\n\
    \n\
    $IH_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// AMP Onsager correction: `z_new[i] = residual[i] + (b / m) * z_prev[i]`.
///
/// Signature: `amp_onsager_kernel(z_new, residual, z_prev, b_over_m, m_dim)`.
#[must_use]
pub fn amp_onsager_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry amp_onsager_kernel(\n\
        .param .u64 p_z_new,\n\
        .param .u64 p_residual,\n\
        .param .u64 p_z_prev,\n\
        .param .f32 p_b_over_m,\n\
        .param .u32 p_m_dim\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_z_new];\n\
        ld.param.u64  %rd1, [p_residual];\n\
        ld.param.u64  %rd2, [p_z_prev];\n\
        ld.param.f32  %f0,  [p_b_over_m];\n\
        ld.param.u32  %r0,  [p_m_dim];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $AO_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
        add.u64       %rd5, %rd2, %rd3;\n\
        ld.global.f32 %f2, [%rd5];\n\
    \n\
        fma.rn.f32    %f3, %f0, %f2, %f1;\n\
        add.u64       %rd6, %rd0, %rd3;\n\
        st.global.f32 [%rd6], %f3;\n\
    \n\
    $AO_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// SVT per-singular-value soft-thresholding: `sigma_new[i] = max(sigma[i] - tau, 0)`.
///
/// Signature: `svt_threshold_kernel(sigma_out, sigma_in, tau, n)`.
#[must_use]
pub fn svt_threshold_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry svt_threshold_kernel(\n\
        .param .u64 p_sigma_out,\n\
        .param .u64 p_sigma_in,\n\
        .param .f32 p_tau,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_sigma_out];\n\
        ld.param.u64  %rd1, [p_sigma_in];\n\
        ld.param.f32  %f0,  [p_tau];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $SV_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
    \n\
        sub.f32       %f2, %f1, %f0;\n\
        max.f32       %f3, %f2, 0f00000000;\n\
    \n\
        add.u64       %rd4, %rd0, %rd2;\n\
        st.global.f32 [%rd4], %f3;\n\
    \n\
    $SV_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// 1D TV gradient (forward difference): `grad[i] = x[i+1] - x[i]` for i in `[0, n-1)`.
///
/// Signature: `tv_grad_kernel(grad, x, n)`. Boundary: `grad[n-1] = 0`.
#[must_use]
pub fn tv_grad_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry tv_grad_kernel(\n\
        .param .u64 p_grad,\n\
        .param .u64 p_x,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_grad];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $TV_DONE;\n\
    \n\
        // i == n - 1 -> write 0\n\
        sub.u32       %r5, %r0, 1;\n\
        setp.ge.u32   %p0, %r4, %r5;\n\
        @%p0 bra $TV_BOUND;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        add.u32       %r6, %r4, 1;\n\
        mul.wide.u32  %rd4, %r6, 4;\n\
        add.u64       %rd5, %rd1, %rd4;\n\
        ld.global.f32 %f2, [%rd5];\n\
        sub.f32       %f3, %f2, %f1;\n\
        add.u64       %rd6, %rd0, %rd2;\n\
        st.global.f32 [%rd6], %f3;\n\
        bra $TV_DONE;\n\
    \n\
    $TV_BOUND:\n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd6, %rd0, %rd2;\n\
        mov.f32       %f4, 0f00000000;\n\
        st.global.f32 [%rd6], %f4;\n\
    \n\
    $TV_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

#[cfg(test)]
mod tests {
    use super::*;

    type KernelFn = fn(u32) -> String;

    fn all_kernels() -> Vec<(&'static str, KernelFn)> {
        vec![
            ("correlate", correlate_ptx),
            ("hard_threshold", hard_threshold_ptx),
            ("soft_threshold", soft_threshold_ptx),
            ("iht_step", iht_step_ptx),
            ("amp_onsager", amp_onsager_ptx),
            ("svt_threshold", svt_threshold_ptx),
            ("tv_grad", tv_grad_ptx),
        ]
    }

    #[test]
    fn ptx_header_versions() {
        assert!(ptx_header(75).contains("7.5"));
        assert!(ptx_header(80).contains("8.0"));
        assert!(ptx_header(90).contains("8.4"));
        assert!(ptx_header(100).contains("8.7"));
    }

    #[test]
    fn ptx_all_kernels_non_empty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            for (name, f) in all_kernels() {
                let s = f(sm);
                assert!(!s.is_empty(), "kernel {name} sm={sm} produced empty string");
                assert!(
                    s.contains(".visible .entry"),
                    "kernel {name} sm={sm} missing entry"
                );
                assert!(s.contains("ret"), "kernel {name} sm={sm} missing ret");
            }
        }
    }
}
