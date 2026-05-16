//! GPU PTX kernels for convex optimization operations.
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

/// AXPY: y = alpha * x + y.
///
/// Signature: `axpy_kernel(y, x, alpha, n)`
/// Grid = (ceil(n/256), 1, 1), Block = (256, 1, 1).
#[must_use]
pub fn axpy_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry axpy_kernel(\n\
        .param .u64 p_y,\n\
        .param .u64 p_x,\n\
        .param .f32 p_alpha,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_y];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.f32  %f0,  [p_alpha];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $AX_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        add.u64       %rd4, %rd0, %rd2;\n\
        ld.global.f32 %f2, [%rd4];\n\
        fma.rn.f32    %f3, %f0, %f1, %f2;\n\
        st.global.f32 [%rd4], %f3;\n\
    \n\
    $AX_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Element-wise soft-thresholding: `y[i] = sign(x[i]) * max(|x[i]| - lambda, 0)`.
///
/// Signature: `soft_threshold_kernel(y, x, lambda, n)`
/// Grid = (ceil(n/256), 1, 1), Block = (256, 1, 1).
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
        // abs_x = abs(x)\n\
        abs.f32       %f2, %f1;\n\
        // thr = max(abs_x - lambda, 0)\n\
        sub.f32       %f3, %f2, %f0;\n\
        max.f32       %f4, %f3, 0f00000000;\n\
        // sign mask: if x >= 0 then thr else -thr\n\
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

/// Simplex projection helper: per-thread element `y[i] = max(x[i] - tau, 0)`.
///
/// Signature: `simplex_proj_kernel(y, x, tau, n)`
/// (Host computes tau via sort+search; kernel applies element-wise.)
#[must_use]
pub fn simplex_proj_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry simplex_proj_kernel(\n\
        .param .u64 p_y,\n\
        .param .u64 p_x,\n\
        .param .f32 p_tau,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_y];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.f32  %f0,  [p_tau];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $SP_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        sub.f32       %f2, %f1, %f0;\n\
        max.f32       %f3, %f2, 0f00000000;\n\
        add.u64       %rd4, %rd0, %rd2;\n\
        st.global.f32 [%rd4], %f3;\n\
    \n\
    $SP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Gradient step: x = x - alpha * g.
///
/// Signature: `gradient_step_kernel(x, g, alpha, n)`
#[must_use]
pub fn gradient_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry gradient_step_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_g,\n\
        .param .f32 p_alpha,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_g];\n\
        ld.param.f32  %f0,  [p_alpha];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $GS_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        add.u64       %rd4, %rd1, %rd2;\n\
        ld.global.f32 %f2, [%rd4];\n\
        // x = x - alpha * g\n\
        neg.f32       %f3, %f0;\n\
        fma.rn.f32    %f4, %f3, %f2, %f1;\n\
        st.global.f32 [%rd3], %f4;\n\
    \n\
    $GS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// FISTA momentum extrapolation: `y[i] = x_new[i] + ((t_k-1)/t_kp1) * (x_new[i] - x_old[i])`.
///
/// Signature: `fista_extrapolate_kernel(y, x_new, x_old, beta, n)` where beta = (t_k-1)/t_kp1.
#[must_use]
pub fn fista_extrapolate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry fista_extrapolate_kernel(\n\
        .param .u64 p_y,\n\
        .param .u64 p_x_new,\n\
        .param .u64 p_x_old,\n\
        .param .f32 p_beta,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_y];\n\
        ld.param.u64  %rd1, [p_x_new];\n\
        ld.param.u64  %rd2, [p_x_old];\n\
        ld.param.f32  %f0,  [p_beta];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $FE_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
        add.u64       %rd5, %rd2, %rd3;\n\
        ld.global.f32 %f2, [%rd5];\n\
        // delta = x_new - x_old\n\
        sub.f32       %f3, %f1, %f2;\n\
        // y = x_new + beta * delta\n\
        fma.rn.f32    %f4, %f0, %f3, %f1;\n\
        add.u64       %rd6, %rd0, %rd3;\n\
        st.global.f32 [%rd6], %f4;\n\
    \n\
    $FE_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// ADMM dual variable update: u = u + (Ax + Bz - c).
///
/// Signature: `admm_dual_update_kernel(u, residual, n)` where residual = Ax + Bz - c (precomputed).
#[must_use]
pub fn admm_dual_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry admm_dual_update_kernel(\n\
        .param .u64 p_u,\n\
        .param .u64 p_residual,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_u];\n\
        ld.param.u64  %rd1, [p_residual];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $AD_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        add.u64       %rd4, %rd1, %rd2;\n\
        ld.global.f32 %f2, [%rd4];\n\
        add.f32       %f3, %f1, %f2;\n\
        st.global.f32 [%rd3], %f3;\n\
    \n\
    $AD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// L2 ball projection: x = x * min(1, r/||x||). Per-thread scale operation.
///
/// Signature: `proj_l2_ball_kernel(x, scale, n)` where scale = min(1, r/||x||) (precomputed).
#[must_use]
pub fn proj_l2_ball_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry proj_l2_ball_kernel(\n\
        .param .u64 p_x,\n\
        .param .f32 p_scale,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.f32  %f0,  [p_scale];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $PL_DONE;\n\
    \n\
        mul.wide.u32  %rd2, %r4, 4;\n\
        add.u64       %rd3, %rd0, %rd2;\n\
        ld.global.f32 %f1, [%rd3];\n\
        mul.f32       %f2, %f0, %f1;\n\
        st.global.f32 [%rd3], %f2;\n\
    \n\
    $PL_DONE:\n\
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
            ("axpy", axpy_ptx),
            ("soft_threshold", soft_threshold_ptx),
            ("simplex_proj", simplex_proj_ptx),
            ("gradient_step", gradient_step_ptx),
            ("fista_extrapolate", fista_extrapolate_ptx),
            ("admm_dual_update", admm_dual_update_ptx),
            ("proj_l2_ball", proj_l2_ball_ptx),
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
