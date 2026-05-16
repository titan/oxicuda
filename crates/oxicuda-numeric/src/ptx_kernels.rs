//! GPU PTX kernels for numerical analysis primitives.
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

/// Horner-rule polynomial evaluation kernel.
///
/// Signature: `horner_eval_kernel(coeff, x_arr, out, degree, n_points)`
/// Each thread evaluates one point: `out[tid] = a_n·x^n + a_{n-1}·x^{n-1} + … + a_0` via
/// Horner's nested form: `(((a_n·x + a_{n-1})·x + a_{n-2})·x + …)`.
#[must_use]
pub fn horner_eval_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry horner_eval_kernel(\n\
        .param .u64 p_coeff,\n\
        .param .u64 p_x,\n\
        .param .u64 p_out,\n\
        .param .u32 p_degree,\n\
        .param .u32 p_n_points\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_coeff];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_degree];\n\
        ld.param.u32  %r1,  [p_n_points];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $HE_DONE;\n\
    \n\
        // load x[tid]\n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.f32 %f0, [%rd4];\n\
    \n\
        // start with leading coefficient (index = degree)\n\
        mul.wide.u32  %rd5, %r0, 4;\n\
        add.u64       %rd6, %rd0, %rd5;\n\
        ld.global.f32 %f1, [%rd6];\n\
    \n\
        // i = degree - 1\n\
        sub.u32       %r6, %r0, 1;\n\
    \n\
    $HE_LOOP:\n\
        setp.lt.s32   %p0, %r6, 0;\n\
        @%p0 bra $HE_WRITE;\n\
    \n\
        mul.wide.u32  %rd7, %r6, 4;\n\
        add.u64       %rd8, %rd0, %rd7;\n\
        ld.global.f32 %f2, [%rd8];\n\
        // acc = acc * x + a_i\n\
        fma.rn.f32    %f1, %f1, %f0, %f2;\n\
    \n\
        sub.u32       %r6, %r6, 1;\n\
        bra $HE_LOOP;\n\
    \n\
    $HE_WRITE:\n\
        mul.wide.u32  %rd9, %r5, 4;\n\
        add.u64       %rd0, %rd2, %rd9;\n\
        st.global.f32 [%rd0], %f1;\n\
    \n\
    $HE_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Fused RK4 update kernel.
///
/// Signature: `rk4_stage_kernel(y, k1, k2, k3, k4, out, h, n)`
/// `out[i] = y[i] + (h/6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])`
#[must_use]
pub fn rk4_stage_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry rk4_stage_kernel(\n\
        .param .u64 p_y,\n\
        .param .u64 p_k1,\n\
        .param .u64 p_k2,\n\
        .param .u64 p_k3,\n\
        .param .u64 p_k4,\n\
        .param .u64 p_out,\n\
        .param .f32 p_h,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<14>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<14>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_y];\n\
        ld.param.u64  %rd1, [p_k1];\n\
        ld.param.u64  %rd2, [p_k2];\n\
        ld.param.u64  %rd3, [p_k3];\n\
        ld.param.u64  %rd4, [p_k4];\n\
        ld.param.u64  %rd5, [p_out];\n\
        ld.param.f32  %f0,  [p_h];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $RK_DONE;\n\
    \n\
        mul.wide.u32  %rd6, %r4, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f1, [%rd7];\n\
        add.u64       %rd8, %rd1, %rd6;\n\
        ld.global.f32 %f2, [%rd8];\n\
        add.u64       %rd9, %rd2, %rd6;\n\
        ld.global.f32 %f3, [%rd9];\n\
        add.u64       %rd10, %rd3, %rd6;\n\
        ld.global.f32 %f4, [%rd10];\n\
        add.u64       %rd11, %rd4, %rd6;\n\
        ld.global.f32 %f5, [%rd11];\n\
    \n\
        // sum = k1 + 2*k2 + 2*k3 + k4\n\
        add.f32       %f6, %f3, %f4;\n\
        add.f32       %f6, %f6, %f6;\n\
        add.f32       %f7, %f2, %f5;\n\
        add.f32       %f8, %f6, %f7;\n\
    \n\
        // h/6\n\
        mov.f32       %f9, 0f3E2AAAAB;\n\
        mul.f32       %f10, %f0, %f9;\n\
        fma.rn.f32    %f11, %f10, %f8, %f1;\n\
    \n\
        add.u64       %rd12, %rd5, %rd6;\n\
        st.global.f32 [%rd12], %f11;\n\
    \n\
    $RK_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Bisection step kernel — compute midpoint and update the bracket.
///
/// Signature: `bisection_step_kernel(a_arr, b_arr, fa_arr, fb_arr, mid, fmid, n)`
/// For each item, `mid[i] = (a[i] + b[i]) / 2`; the host then re-evaluates fmid and updates.
#[must_use]
pub fn bisection_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry bisection_step_kernel(\n\
        .param .u64 p_a,\n\
        .param .u64 p_b,\n\
        .param .u64 p_mid,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_a];\n\
        ld.param.u64  %rd1, [p_b];\n\
        ld.param.u64  %rd2, [p_mid];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $BI_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f0, [%rd4];\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        // mid = 0.5 * (a + b)\n\
        add.f32       %f2, %f0, %f1;\n\
        mov.f32       %f3, 0f3F000000;\n\
        mul.f32       %f4, %f2, %f3;\n\
    \n\
        add.u64       %rd6, %rd2, %rd3;\n\
        st.global.f32 [%rd6], %f4;\n\
    \n\
    $BI_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Gauss quadrature accumulation kernel: out = Σ w_i · f(x_i).
///
/// Signature: `gauss_quad_accumulate_kernel(weights, fvals, out, n)`
/// Each thread holds one (w_i, f_i) pair; a reduction is performed in shared memory.
/// For simplicity each thread writes its w_i·f_i to its output slot — host or another reduction
/// kernel sums the result.
#[must_use]
pub fn gauss_quad_accumulate_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry gauss_quad_accumulate_kernel(\n\
        .param .u64 p_w,\n\
        .param .u64 p_f,\n\
        .param .u64 p_out,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<6>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_w];\n\
        ld.param.u64  %rd1, [p_f];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $GQ_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f0, [%rd4];\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f32 %f1, [%rd5];\n\
    \n\
        mul.f32       %f2, %f0, %f1;\n\
    \n\
        add.u64       %rd6, %rd2, %rd3;\n\
        st.global.f32 [%rd6], %f2;\n\
    \n\
    $GQ_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Cubic-spline evaluation kernel.
///
/// Signature: `spline_eval_kernel(xs, ys, m_coef, x_eval, out, idx, n_query, n_nodes)`
/// Given precomputed second derivatives `m_coef` and node arrays `(xs, ys)`, evaluate the
/// natural cubic spline at the query points `x_eval[k]` using piece index `idx[k]`.
/// `out[k] = A·ys[i] + B·ys[i+1] + ((A³-A)·m[i] + (B³-B)·m[i+1])·h²/6`
/// `where h=xs[i+1]-xs[i], A=(xs[i+1]-x)/h, B=1-A`
#[must_use]
pub fn spline_eval_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry spline_eval_kernel(\n\
        .param .u64 p_xs,\n\
        .param .u64 p_ys,\n\
        .param .u64 p_m,\n\
        .param .u64 p_xe,\n\
        .param .u64 p_out,\n\
        .param .u64 p_idx,\n\
        .param .u32 p_nq,\n\
        .param .u32 p_nn\n\
    )\n\
    {\n\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<14>;\n\
        .reg .f32  %f<20>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_xs];\n\
        ld.param.u64  %rd1, [p_ys];\n\
        ld.param.u64  %rd2, [p_m];\n\
        ld.param.u64  %rd3, [p_xe];\n\
        ld.param.u64  %rd4, [p_out];\n\
        ld.param.u64  %rd5, [p_idx];\n\
        ld.param.u32  %r0,  [p_nq];\n\
        ld.param.u32  %r1,  [p_nn];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r0;\n\
        @%p0 bra $SP_DONE;\n\
    \n\
        // read piece index i\n\
        mul.wide.u32  %rd6, %r5, 4;\n\
        add.u64       %rd7, %rd5, %rd6;\n\
        ld.global.u32 %r6, [%rd7];\n\
    \n\
        mul.wide.u32  %rd8, %r6, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        ld.global.f32 %f0, [%rd9];\n\
        add.u64       %rd10, %rd9, 4;\n\
        ld.global.f32 %f1, [%rd10];\n\
        add.u64       %rd11, %rd1, %rd8;\n\
        ld.global.f32 %f2, [%rd11];\n\
        add.u64       %rd12, %rd11, 4;\n\
        ld.global.f32 %f3, [%rd12];\n\
        add.u64       %rd13, %rd2, %rd8;\n\
        ld.global.f32 %f4, [%rd13];\n\
        add.u64       %rd14, %rd13, 4;\n\
        ld.global.f32 %f5, [%rd14];\n\
    \n\
        // x = xe[tid]\n\
        add.u64       %rd15, %rd3, %rd6;\n\
        ld.global.f32 %f6, [%rd15];\n\
    \n\
        // h = x1 - x0\n\
        sub.f32       %f7, %f1, %f0;\n\
        // A = (x1 - x) / h\n\
        sub.f32       %f8, %f1, %f6;\n\
        div.approx.f32 %f9, %f8, %f7;\n\
        // B = 1 - A\n\
        mov.f32       %f10, 0f3F800000;\n\
        sub.f32       %f11, %f10, %f9;\n\
    \n\
        // base = A*y0 + B*y1\n\
        mul.f32       %f12, %f9, %f2;\n\
        fma.rn.f32    %f13, %f11, %f3, %f12;\n\
    \n\
        // (A^3 - A) * m0\n\
        mul.f32       %f14, %f9, %f9;\n\
        mul.f32       %f14, %f14, %f9;\n\
        sub.f32       %f14, %f14, %f9;\n\
        mul.f32       %f14, %f14, %f4;\n\
        // (B^3 - B) * m1\n\
        mul.f32       %f15, %f11, %f11;\n\
        mul.f32       %f15, %f15, %f11;\n\
        sub.f32       %f15, %f15, %f11;\n\
        mul.f32       %f15, %f15, %f5;\n\
        // sum * h^2 / 6\n\
        add.f32       %f16, %f14, %f15;\n\
        mul.f32       %f17, %f7, %f7;\n\
        mul.f32       %f16, %f16, %f17;\n\
        mov.f32       %f18, 0f3E2AAAAB;\n\
        mul.f32       %f16, %f16, %f18;\n\
    \n\
        add.f32       %f19, %f13, %f16;\n\
    \n\
        add.u64       %rd6, %rd4, %rd6;\n\
        st.global.f32 [%rd6], %f19;\n\
    \n\
    $SP_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Central finite difference gradient kernel.
///
/// Signature: `central_diff_kernel(f_plus, f_minus, out, h, n)`
/// `out[i] = (f_plus[i] - f_minus[i]) / (2*h)`
#[must_use]
pub fn central_diff_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry central_diff_kernel(\n\
        .param .u64 p_fp,\n\
        .param .u64 p_fm,\n\
        .param .u64 p_out,\n\
        .param .f32 p_h,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<8>;\n\
        .reg .u32  %r<10>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_fp];\n\
        ld.param.u64  %rd1, [p_fm];\n\
        ld.param.u64  %rd2, [p_out];\n\
        ld.param.f32  %f0,  [p_h];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $CD_DONE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
        add.u64       %rd5, %rd1, %rd3;\n\
        ld.global.f32 %f2, [%rd5];\n\
    \n\
        sub.f32       %f3, %f1, %f2;\n\
        add.f32       %f4, %f0, %f0;\n\
        div.approx.f32 %f5, %f3, %f4;\n\
    \n\
        add.u64       %rd6, %rd2, %rd3;\n\
        st.global.f32 [%rd6], %f5;\n\
    \n\
    $CD_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

/// Bessel function downward recurrence step (Miller's algorithm).
///
/// Signature: `bessel_recurrence_kernel(j_arr, x_arr, n_order, n_points)`
/// For each x, performs the recurrence J_{n-1}(x) = (2n/x) J_n(x) - J_{n+1}(x).
/// j_arr is treated as a 2-D array of shape (n_points, n_order+1) and updated in-place.
#[must_use]
pub fn bessel_recurrence_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry bessel_recurrence_kernel(\n\
        .param .u64 p_j,\n\
        .param .u64 p_x,\n\
        .param .u32 p_n_order,\n\
        .param .u32 p_n_points\n\
    )\n\
    {\n\
        .reg .u64  %rd<10>;\n\
        .reg .u32  %r<14>;\n\
        .reg .f32  %f<10>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_j];\n\
        ld.param.u64  %rd1, [p_x];\n\
        ld.param.u32  %r0,  [p_n_order];\n\
        ld.param.u32  %r1,  [p_n_points];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $BR_DONE;\n\
    \n\
        // x[tid]\n\
        mul.wide.u32  %rd2, %r5, 4;\n\
        add.u64       %rd3, %rd1, %rd2;\n\
        ld.global.f32 %f0, [%rd3];\n\
    \n\
        // stride in j: (n_order + 1) per point\n\
        add.u32       %r6, %r0, 1;\n\
        mul.lo.u32    %r7, %r5, %r6;\n\
    \n\
        // n = n_order; iterate downward to 1\n\
        mov.u32       %r8, %r0;\n\
    \n\
    $BR_LOOP:\n\
        setp.le.u32   %p0, %r8, 0;\n\
        @%p0 bra $BR_DONE;\n\
    \n\
        // J_{n}\n\
        add.u32       %r9, %r7, %r8;\n\
        mul.wide.u32  %rd4, %r9, 4;\n\
        add.u64       %rd5, %rd0, %rd4;\n\
        ld.global.f32 %f1, [%rd5];\n\
        // J_{n+1}\n\
        add.u32       %r10, %r9, 1;\n\
        mul.wide.u32  %rd6, %r10, 4;\n\
        add.u64       %rd7, %rd0, %rd6;\n\
        ld.global.f32 %f2, [%rd7];\n\
    \n\
        // J_{n-1} = (2n/x) * J_n - J_{n+1}\n\
        cvt.rn.f32.u32 %f3, %r8;\n\
        add.f32       %f3, %f3, %f3;\n\
        div.approx.f32 %f4, %f3, %f0;\n\
        mul.f32       %f5, %f4, %f1;\n\
        sub.f32       %f6, %f5, %f2;\n\
    \n\
        sub.u32       %r11, %r9, 1;\n\
        mul.wide.u32  %rd8, %r11, 4;\n\
        add.u64       %rd9, %rd0, %rd8;\n\
        st.global.f32 [%rd9], %f6;\n\
    \n\
        sub.u32       %r8, %r8, 1;\n\
        bra $BR_LOOP;\n\
    \n\
    $BR_DONE:\n\
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
            ("horner_eval", horner_eval_ptx),
            ("rk4_stage", rk4_stage_ptx),
            ("bisection_step", bisection_step_ptx),
            ("gauss_quad_accumulate", gauss_quad_accumulate_ptx),
            ("spline_eval", spline_eval_ptx),
            ("central_diff", central_diff_ptx),
            ("bessel_recurrence", bessel_recurrence_ptx),
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
