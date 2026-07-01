//! GPU PTX kernels for survival analysis operations.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on SM version.
//! PTX ISA is selected by SM:
//!     SM>=100 → 8.7 (Blackwell), SM>=90 → 8.4 (Hopper),
//!     SM>=80  → 8.0 (Ampere),    else → 7.5 (Turing).
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

/// Kaplan-Meier per-time step kernel.
///
/// Signature: `km_step_kernel(d, n, s_out, n_steps, s_init)`
/// Computes S(t_i) = s_init * Π_{k<=i} (1 - d_k/n_k) by parallel prefix on log-domain.
/// For simplicity each thread independently scans contributions up to its index.
#[must_use]
pub fn km_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry km_step_kernel(\n\
        .param .u64 p_d,\n\
        .param .u64 p_n,\n\
        .param .u64 p_s_out,\n\
        .param .u32 p_n_steps,\n\
        .param .f32 p_s_init\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_d];\n\
        ld.param.u64  %rd1, [p_n];\n\
        ld.param.u64  %rd2, [p_s_out];\n\
        ld.param.u32  %r0,  [p_n_steps];\n\
        ld.param.f32  %f0,  [p_s_init];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $KM_DONE;\n\
    \n\
        mov.f32       %f1, %f0;\n\
        mov.u32       %r5, 0;\n\
    \n\
    $KM_LOOP:\n\
        setp.gt.u32   %p0, %r5, %r4;\n\
        @%p0 bra $KM_WRITE;\n\
    \n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f2, [%rd4];\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f32 %f3, [%rd5];\n\
    \n\
        // factor = 1 - d/n\n\
        div.approx.f32 %f4, %f2, %f3;\n\
        mov.f32       %f5, 0f3F800000;\n\
        sub.f32       %f6, %f5, %f4;\n\
        mul.f32       %f1, %f1, %f6;\n\
    \n\
        add.u32       %r5, %r5, 1;\n\
        bra $KM_LOOP;\n\
    \n\
    $KM_WRITE:\n\
        mul.wide.u32  %rd6, %r4, 4;\n\
        add.u64       %rd7, %rd2, %rd6;\n\
        st.global.f32 [%rd7], %f1;\n\
    \n\
    $KM_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Cox risk sum kernel: compute exp(β·x_j) sum over risk set.
///
/// Signature: `cox_risk_sum_kernel(eta, mask, out, n)`
/// `out[0] = Σ_{j: mask[j]=1} exp(eta[j])`
#[must_use]
pub fn cox_risk_sum_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cox_risk_sum_kernel(\n\
        .param .u64 p_eta,\n\
        .param .u64 p_mask,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<12>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_eta];\n\
        ld.param.u64  %rd1, [p_mask];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $CRS_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f0, [%rd4];\n\
        mov.f32       %f5, 0f00000000;\n\
        setp.eq.f32   %p0, %f0, %f5;\n\
        @%p0 bra $CRS_DONE;\n\
    \n\
        add.u64       %rd5, %rd0, %rd3;\n\
        ld.global.f32 %f1, [%rd5];\n\
        // exp(eta) = ex2(eta * log2(e)); ex2 is base-2, so scale first.\n\
        mul.f32       %f3, %f1, 0f3FB8AA3B;\n\
        ex2.approx.f32 %f2, %f3;\n\
    \n\
        // atomic add into out[0]\n\
        red.global.add.f32 [%rd2], %f2;\n\
    \n\
    $CRS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Cox partial-likelihood score (gradient) accumulation kernel.
///
/// Signature: `cox_score_kernel(eta, x, mask, d_event, score, n, p)`
/// `score[k] += Σ_{j ∈ R} w_j * x[j,k]`  (later normalised by host)
#[must_use]
pub fn cox_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cox_score_kernel(\n\
        .param .u64 p_eta,\n\
        .param .u64 p_x,\n\
        .param .u64 p_mask,\n\
        .param .u64 p_score,\n\
        .param .u32 p_n,\n\
        .param .u32 p_p\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_eta];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.u64  %rd2, [p_mask];\n\
        ld.param.u64  %rd3, [p_score];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_p];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $CSC_DONE;\n\
    \n\
        mul.wide.u32  %rd4, %r5, 4;\n\
        add.u64       %rd5, %rd2, %rd4;\n\
        ld.global.f32 %f0, [%rd5];\n\
        mov.f32       %f9, 0f00000000;\n\
        setp.eq.f32   %p0, %f0, %f9;\n\
        @%p0 bra $CSC_DONE;\n\
    \n\
        add.u64       %rd6, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd6];\n\
        // exp(eta) = ex2(eta * log2(e)); ex2 is base-2, so scale first.\n\
        mul.f32       %f5, %f1, 0f3FB8AA3B;\n\
        ex2.approx.f32 %f2, %f5;\n\
    \n\
        // for k=0..p: score[k] += w * x[row*p + k]\n\
        mov.u32       %r6, 0;\n\
    $CSC_LOOP:\n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        @%p0 bra $CSC_DONE;\n\
    \n\
        mul.lo.u32    %r7, %r5, %r1;\n\
        add.u32       %r7, %r7, %r6;\n\
        mul.wide.u32  %rd7, %r7, 4;\n\
        add.u64       %rd8, %rd1, %rd7;\n\
        ld.global.f32 %f3, [%rd8];\n\
        mul.f32       %f4, %f2, %f3;\n\
    \n\
        mul.wide.u32  %rd9, %r6, 4;\n\
        add.u64       %rd10, %rd3, %rd9;\n\
        red.global.add.f32 [%rd10], %f4;\n\
    \n\
        add.u32       %r6, %r6, 1;\n\
        bra $CSC_LOOP;\n\
    \n\
    $CSC_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Cox Fisher information matrix accumulation kernel.
///
/// Signature: `cox_info_kernel(eta, x, mask, info, n, p)`
/// `info[k,l] += Σ w_j * x[j,k] * x[j,l]`
#[must_use]
pub fn cox_info_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry cox_info_kernel(\n\
        .param .u64 p_eta,\n\
        .param .u64 p_x,\n\
        .param .u64 p_mask,\n\
        .param .u64 p_info,\n\
        .param .u32 p_n,\n\
        .param .u32 p_p\n\
    )\n\
    {\n\
        .reg .u64  %rd<14>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<12>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_eta];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.u64  %rd2, [p_mask];\n\
        ld.param.u64  %rd3, [p_info];\n\
        ld.param.u32  %r0,  [p_n];\n\
        ld.param.u32  %r1,  [p_p];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $CIN_DONE;\n\
    \n\
        mul.wide.u32  %rd4, %r5, 4;\n\
        add.u64       %rd5, %rd2, %rd4;\n\
        ld.global.f32 %f0, [%rd5];\n\
        mov.f32       %f10, 0f00000000;\n\
        setp.eq.f32   %p0, %f0, %f10;\n\
        @%p0 bra $CIN_DONE;\n\
    \n\
        add.u64       %rd6, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd6];\n\
        // exp(eta) = ex2(eta * log2(e)); ex2 is base-2, so scale first.\n\
        mul.f32       %f6, %f1, 0f3FB8AA3B;\n\
        ex2.approx.f32 %f2, %f6;\n\
    \n\
        mov.u32       %r6, 0;\n\
    $CIN_K:\n\
        setp.ge.u32   %p0, %r6, %r1;\n\
        @%p0 bra $CIN_DONE;\n\
    \n\
        mul.lo.u32    %r7, %r5, %r1;\n\
        add.u32       %r7, %r7, %r6;\n\
        mul.wide.u32  %rd7, %r7, 4;\n\
        add.u64       %rd8, %rd1, %rd7;\n\
        ld.global.f32 %f3, [%rd8];\n\
    \n\
        mov.u32       %r8, 0;\n\
    $CIN_L:\n\
        setp.ge.u32   %p0, %r8, %r1;\n\
        @%p0 bra $CIN_K_END;\n\
    \n\
        mul.lo.u32    %r9, %r5, %r1;\n\
        add.u32       %r9, %r9, %r8;\n\
        mul.wide.u32  %rd9, %r9, 4;\n\
        add.u64       %rd10, %rd1, %rd9;\n\
        ld.global.f32 %f4, [%rd10];\n\
    \n\
        mul.f32       %f5, %f3, %f4;\n\
        mul.f32       %f5, %f5, %f2;\n\
    \n\
        mul.lo.u32    %r10, %r6, %r1;\n\
        add.u32       %r10, %r10, %r8;\n\
        mul.wide.u32  %rd11, %r10, 4;\n\
        add.u64       %rd12, %rd3, %rd11;\n\
        red.global.add.f32 [%rd12], %f5;\n\
    \n\
        add.u32       %r8, %r8, 1;\n\
        bra $CIN_L;\n\
    \n\
    $CIN_K_END:\n\
        add.u32       %r6, %r6, 1;\n\
        bra $CIN_K;\n\
    \n\
    $CIN_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Log-rank observed minus expected per-time kernel.
///
/// Signature: `logrank_oe_kernel(d_group, n_group, d_total, n_total, oe_out, n_times)`
/// `oe[t] = d_group[t] - n_group[t] * d_total[t] / n_total[t]`
#[must_use]
pub fn logrank_oe_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry logrank_oe_kernel(\n\
        .param .u64 p_d_g,\n\
        .param .u64 p_n_g,\n\
        .param .u64 p_d_t,\n\
        .param .u64 p_n_t,\n\
        .param .u64 p_oe,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_d_g];\n\
        ld.param.u64  %rd1, [p_n_g];\n\
        ld.param.u64  %rd2, [p_d_t];\n\
        ld.param.u64  %rd3, [p_n_t];\n\
        ld.param.u64  %rd4, [p_oe];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $LR_DONE;\n\
    \n\
        mul.wide.u32  %rd5, %r4, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f0, [%rd6];\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.f32 %f1, [%rd6];\n\
        add.u64       %rd6, %rd2, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
        add.u64       %rd6, %rd3, %rd5;\n\
        ld.global.f32 %f3, [%rd6];\n\
    \n\
        div.approx.f32 %f4, %f2, %f3;\n\
        mul.f32       %f5, %f1, %f4;\n\
        sub.f32       %f6, %f0, %f5;\n\
    \n\
        add.u64       %rd6, %rd4, %rd5;\n\
        st.global.f32 [%rd6], %f6;\n\
    \n\
    $LR_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Brier score elementwise IPCW kernel.
///
/// Signature: `brier_score_kernel(t, delta, s_pred, w, t_star, out, n)`
/// `out[i] = w[i] * (indicator - s_pred[i])^2`
#[must_use]
pub fn brier_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry brier_score_kernel(\n\
        .param .u64 p_t,\n\
        .param .u64 p_delta,\n\
        .param .u64 p_s,\n\
        .param .u64 p_w,\n\
        .param .f32 p_tstar,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<12>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_t];\n\
        ld.param.u64  %rd1, [p_delta];\n\
        ld.param.u64  %rd2, [p_s];\n\
        ld.param.u64  %rd3, [p_w];\n\
        ld.param.f32  %f0,  [p_tstar];\n\
        ld.param.u64  %rd4, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $BS_DONE;\n\
    \n\
        mul.wide.u32  %rd5, %r4, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f1, [%rd6];\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
        add.u64       %rd6, %rd2, %rd5;\n\
        ld.global.f32 %f3, [%rd6];\n\
        add.u64       %rd6, %rd3, %rd5;\n\
        ld.global.f32 %f4, [%rd6];\n\
    \n\
        // indicator = (t <= t_star && delta==1) ? 1 : 0\n\
        mov.f32       %f10, 0f00000000;\n\
        mov.f32       %f11, 0f3F800000;\n\
        setp.le.f32   %p0, %f1, %f0;\n\
        setp.gt.f32   %p1, %f2, %f10;\n\
        and.pred      %p0, %p0, %p1;\n\
        selp.f32      %f5, %f11, %f10, %p0;\n\
    \n\
        sub.f32       %f6, %f5, %f3;\n\
        mul.f32       %f7, %f6, %f6;\n\
        mul.f32       %f8, %f4, %f7;\n\
    \n\
        add.u64       %rd6, %rd4, %rd5;\n\
        st.global.f32 [%rd6], %f8;\n\
    \n\
    $BS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// RMST integration kernel (rectangle rule).
///
/// Signature: `rmst_integrate_kernel(t, s, tau, out, n)`
/// `out[i] = (min(t[i+1], tau) - t[i]) * s[i] if t[i] < tau else 0`
#[must_use]
pub fn rmst_integrate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry rmst_integrate_kernel(\n\
        .param .u64 p_t,\n\
        .param .u64 p_s,\n\
        .param .f32 p_tau,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_t];\n\
        ld.param.u64  %rd1, [p_s];\n\
        ld.param.f32  %f0,  [p_tau];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        sub.u32       %r5, %r0, 1;\n\
        setp.ge.u32   %p0, %r4, %r5;\n\
        @%p0 bra $RM_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
        add.u32       %r6, %r4, 1;\n\
        mul.wide.u32  %rd5, %r6, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f2, [%rd6];\n\
    \n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f3, [%rd4];\n\
    \n\
        // upper = min(t[i+1], tau)\n\
        setp.lt.f32   %p0, %f2, %f0;\n\
        selp.f32      %f4, %f2, %f0, %p0;\n\
        sub.f32       %f5, %f4, %f1;\n\
        mov.f32       %f6, 0f00000000;\n\
        setp.le.f32   %p0, %f5, %f6;\n\
        selp.f32      %f5, %f6, %f5, %p0;\n\
    \n\
        mul.f32       %f7, %f3, %f5;\n\
        add.u64       %rd4, %rd2, %rd3;\n\
        st.global.f32 [%rd4], %f7;\n\
    \n\
    $RM_DONE:\n\
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
            ("km_step", km_step_ptx),
            ("cox_risk_sum", cox_risk_sum_ptx),
            ("cox_score", cox_score_ptx),
            ("cox_info", cox_info_ptx),
            ("logrank_oe", logrank_oe_ptx),
            ("brier_score", brier_score_ptx),
            ("rmst_integrate", rmst_integrate_ptx),
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

    #[test]
    fn ptx_seven_kernels() {
        assert_eq!(all_kernels().len(), 7);
    }
}
