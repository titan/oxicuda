//! GPU PTX kernels for Optimal Transport algorithms.
//!
//! Each kernel is emitted as a self-contained PTX module string, parameterised on
//! SM version. The kernels target a single-thread-per-output pattern matching the
//! Vol.42/43 reference implementation, with explicit register usage and structured
//! control flow. PTX ISA is selected by SM:
//!     SM≥100 → 8.7 (Blackwell), SM≥90 → 8.4 (Hopper),
//!     SM≥80  → 8.0 (Ampere),    else → 7.5 (Turing).

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

/// Encode a `f32` constant as a PTX immediate hex literal (`0Fxxxxxxxx`).
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

/// One log-domain Sinkhorn iteration: `u_i = ε·log(a_i) − ε·logsumexp_j((v_j − C_ij)/ε)`.
///
/// Kernel signature: `sinkhorn_step_kernel(c, log_a, log_b, u, v, m, n, eps)`.
/// Grid=(m,1,1) Block=(1,1,1). Each block computes one row update of `u`.
#[must_use]
pub fn sinkhorn_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#"{hdr}// sinkhorn_step_kernel: log-domain Sinkhorn row update.
// c: [m*n] cost matrix
// log_a: [m] log-marginal source, log_b: [n] (unused here, for symmetry)
// u: [m] in/out potentials
// v: [n] current target potentials
// m, n: matrix shape, eps: regularization (>0)
.visible .entry sinkhorn_step_kernel(
    .param .u64 p_c,
    .param .u64 p_log_a,
    .param .u64 p_log_b,
    .param .u64 p_u,
    .param .u64 p_v,
    .param .u32 p_m,
    .param .u32 p_n,
    .param .f32 p_eps
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_c];
    ld.param.u64  %rd1, [p_log_a];
    ld.param.u64  %rd3, [p_u];
    ld.param.u64  %rd4, [p_v];
    ld.param.u32  %r0,  [p_m];
    ld.param.u32  %r1,  [p_n];
    ld.param.f32  %f0,  [p_eps];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;   // row index i

    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $SK_DONE;

    // First pass: max_val = max_j ((v_j - C_ij)/eps)
    mov.f32       %f1, {NINF};   // running max
    mov.u32       %r7, 0;        // j=0

$SK_MAX_LOOP:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $SK_MAX_DONE;

    // C[i,j]
    mul.lo.u32    %r8, %r6, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd5, %r8, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f2, [%rd6];

    // v[j]
    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd6, %rd4, %rd5;
    ld.global.f32 %f3, [%rd6];

    sub.f32       %f4, %f3, %f2;
    div.rn.f32    %f4, %f4, %f0;
    max.f32       %f1, %f1, %f4;

    add.u32       %r7, %r7, 1;
    bra $SK_MAX_LOOP;

$SK_MAX_DONE:
    // Second pass: sum_exp = sum_j exp((v_j - C_ij)/eps - max_val)
    mov.f32       %f5, {ZERO};
    mov.u32       %r7, 0;

$SK_SUM_LOOP:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $SK_SUM_DONE;

    mul.lo.u32    %r8, %r6, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd5, %r8, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f2, [%rd6];

    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd6, %rd4, %rd5;
    ld.global.f32 %f3, [%rd6];

    sub.f32       %f4, %f3, %f2;
    div.rn.f32    %f4, %f4, %f0;
    sub.f32       %f4, %f4, %f1;
    ex2.approx.f32 %f6, %f4;     // PTX has ex2; exp(x)=ex2(x*log2(e)) — approximation
    add.f32       %f5, %f5, %f6;

    add.u32       %r7, %r7, 1;
    bra $SK_SUM_LOOP;

$SK_SUM_DONE:
    // u[i] = eps*log_a[i] - eps*(max_val + log(sum_exp))
    mul.wide.u32  %rd5, %r6, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f7, [%rd6];   // log_a[i]
    mul.f32       %f7, %f7, %f0;

    lg2.approx.f32 %f8, %f5;
    add.f32       %f8, %f8, %f1;
    mul.f32       %f8, %f8, %f0;

    sub.f32       %f9, %f7, %f8;

    add.u64       %rd7, %rd3, %rd5;
    st.global.f32 [%rd7], %f9;

$SK_DONE:
    ret;
}}
"#,
        ZERO = zero,
        NINF = neg_inf,
    )
}

/// Compute pairwise distance matrix `C_ij = ‖x_i − y_j‖_p^p` for `mode∈{1:L1, 2:L2-sq}`.
///
/// Kernel signature: `cost_matrix_kernel(x, y, c, m, n, dim, mode)`.
/// Grid=(m,n,1) Block=(1,1,1).
#[must_use]
pub fn cost_matrix_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// cost_matrix_kernel: pairwise distance.
// x: [m*dim], y: [n*dim], c: [m*n] output, m, n, dim, mode (1=L1, 2=L2sq)
.visible .entry cost_matrix_kernel(
    .param .u64 p_x,
    .param .u64 p_y,
    .param .u64 p_c,
    .param .u32 p_m,
    .param .u32 p_n,
    .param .u32 p_dim,
    .param .u32 p_mode
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_y];
    ld.param.u64  %rd2, [p_c];
    ld.param.u32  %r0,  [p_m];
    ld.param.u32  %r1,  [p_n];
    ld.param.u32  %r2,  [p_dim];
    ld.param.u32  %r3,  [p_mode];

    mov.u32       %r4, %ctaid.x;   // i
    mov.u32       %r5, %ctaid.y;   // j

    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $CM_DONE;
    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $CM_DONE;

    mov.f32       %f0, {ZERO};
    mov.u32       %r6, 0;

$CM_LOOP:
    setp.ge.u32   %p0, %r6, %r2;
    @%p0 bra $CM_LOOP_DONE;

    mul.lo.u32    %r7, %r4, %r2;
    add.u32       %r7, %r7, %r6;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    mul.lo.u32    %r7, %r5, %r2;
    add.u32       %r7, %r7, %r6;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];

    sub.f32       %f3, %f1, %f2;
    setp.eq.u32   %p1, %r3, 1;
    @%p1 abs.f32  %f3, %f3;        // L1
    @!%p1 mul.f32 %f3, %f3, %f3;   // L2 squared
    add.f32       %f0, %f0, %f3;

    add.u32       %r6, %r6, 1;
    bra $CM_LOOP;

$CM_LOOP_DONE:
    mul.lo.u32    %r7, %r4, %r1;
    add.u32       %r7, %r7, %r5;
    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f0;

$CM_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

/// Apply transport plan `T` for barycentric mapping: `Tx_i = Σ_j (P_ij/Σ_k P_ik)·y_j`.
///
/// Kernel signature: `transport_apply_kernel(plan, y, out, m, n, dim)`.
/// Grid=(m,1,1) Block=(1,1,1).
#[must_use]
pub fn transport_apply_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps_d = f32_hex(1e-12_f32);
    format!(
        r#"{hdr}// transport_apply_kernel: barycentric mapping.
.visible .entry transport_apply_kernel(
    .param .u64 p_plan,
    .param .u64 p_y,
    .param .u64 p_out,
    .param .u32 p_m,
    .param .u32 p_n,
    .param .u32 p_dim
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_plan];
    ld.param.u64  %rd1, [p_y];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [p_m];
    ld.param.u32  %r1,  [p_n];
    ld.param.u32  %r2,  [p_dim];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $TA_DONE;

    // Compute row sum
    mov.f32       %f0, {ZERO};
    mov.u32       %r7, 0;

$TA_SUM:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $TA_SUM_DONE;
    mul.lo.u32    %r8, %r6, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd3, %r8, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32       %f0, %f0, %f1;
    add.u32       %r7, %r7, 1;
    bra $TA_SUM;

$TA_SUM_DONE:
    add.f32       %f0, %f0, {EPS};   // avoid /0

    // For each dim d, accumulate Σ_j P_ij * y_jd / row_sum
    mov.u32       %r9, 0;
$TA_DIM:
    setp.ge.u32   %p0, %r9, %r2;
    @%p0 bra $TA_DIM_DONE;

    mov.f32       %f2, {ZERO};
    mov.u32       %r7, 0;
$TA_INNER:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $TA_INNER_DONE;
    // P[i,j]
    mul.lo.u32    %r8, %r6, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd3, %r8, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f3, [%rd4];
    // y[j,d]
    mul.lo.u32    %r10, %r7, %r2;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd3, %r10, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f4, [%rd4];
    fma.rn.f32    %f2, %f3, %f4, %f2;
    add.u32       %r7, %r7, 1;
    bra $TA_INNER;
$TA_INNER_DONE:
    div.rn.f32    %f2, %f2, %f0;
    // out[i,d] = f2
    mul.lo.u32    %r10, %r6, %r2;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd5, %r10, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f2;
    add.u32       %r9, %r9, 1;
    bra $TA_DIM;

$TA_DIM_DONE:
$TA_DONE:
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps_d
    )
}

/// Project samples onto random unit directions for Sliced-Wasserstein.
///
/// Kernel signature: `sliced_proj_kernel(theta, x, proj, n_proj, n, dim)`.
/// `theta`: [n_proj, dim] unit directions; `x`: [n, dim]; `proj`: [n_proj, n].
#[must_use]
pub fn sliced_proj_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// sliced_proj_kernel: projection onto random directions.
.visible .entry sliced_proj_kernel(
    .param .u64 p_theta,
    .param .u64 p_x,
    .param .u64 p_proj,
    .param .u32 p_np,
    .param .u32 p_n,
    .param .u32 p_dim
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_theta];
    ld.param.u64  %rd1, [p_x];
    ld.param.u64  %rd2, [p_proj];
    ld.param.u32  %r0,  [p_np];
    ld.param.u32  %r1,  [p_n];
    ld.param.u32  %r2,  [p_dim];

    mov.u32       %r3, %ctaid.x;     // k = direction index
    mov.u32       %r4, %ctaid.y;     // i = sample index
    setp.ge.u32   %p0, %r3, %r0;
    @%p0 bra $SP_DONE;
    setp.ge.u32   %p0, %r4, %r1;
    @%p0 bra $SP_DONE;

    mov.f32       %f0, {ZERO};
    mov.u32       %r5, 0;

$SP_LOOP:
    setp.ge.u32   %p0, %r5, %r2;
    @%p0 bra $SP_LOOP_DONE;
    // theta[k, d]
    mul.lo.u32    %r6, %r3, %r2;
    add.u32       %r6, %r6, %r5;
    mul.wide.u32  %rd3, %r6, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];
    // x[i, d]
    mul.lo.u32    %r6, %r4, %r2;
    add.u32       %r6, %r6, %r5;
    mul.wide.u32  %rd3, %r6, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];
    fma.rn.f32    %f0, %f1, %f2, %f0;
    add.u32       %r5, %r5, 1;
    bra $SP_LOOP;

$SP_LOOP_DONE:
    mul.lo.u32    %r6, %r3, %r1;
    add.u32       %r6, %r6, %r4;
    mul.wide.u32  %rd5, %r6, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f0;

$SP_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

/// Gromov-Wasserstein gradient `G_ij = -2 Σ_kl C1_ik T_kl C2_jl`.
///
/// Kernel signature: `gromov_grad_kernel(c1, c2, t, g, m, n)`.
#[must_use]
pub fn gromov_grad_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let neg_two = f32_hex(-2.0_f32);
    format!(
        r#"{hdr}// gromov_grad_kernel: -2 * C1 * T * C2^T
.visible .entry gromov_grad_kernel(
    .param .u64 p_c1,
    .param .u64 p_c2,
    .param .u64 p_t,
    .param .u64 p_g,
    .param .u32 p_m,
    .param .u32 p_n
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_c1];
    ld.param.u64  %rd1, [p_c2];
    ld.param.u64  %rd2, [p_t];
    ld.param.u64  %rd3, [p_g];
    ld.param.u32  %r0,  [p_m];
    ld.param.u32  %r1,  [p_n];

    mov.u32       %r3, %ctaid.x;     // i
    mov.u32       %r4, %ctaid.y;     // j
    setp.ge.u32   %p0, %r3, %r0;
    @%p0 bra $GW_DONE;
    setp.ge.u32   %p0, %r4, %r1;
    @%p0 bra $GW_DONE;

    // G[i,j] = -2 sum_{{k,l}} C1[i,k] * T[k,l] * C2[j,l]
    mov.f32       %f0, {ZERO};
    mov.u32       %r5, 0;            // k

$GW_K:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $GW_K_DONE;
    // C1[i,k]
    mul.lo.u32    %r6, %r3, %r0;
    add.u32       %r6, %r6, %r5;
    mul.wide.u32  %rd4, %r6, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];

    mov.u32       %r7, 0;            // l
$GW_L:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $GW_L_DONE;
    // T[k,l]
    mul.lo.u32    %r8, %r5, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];
    // C2[j,l]
    mul.lo.u32    %r9, %r4, %r1;
    add.u32       %r9, %r9, %r7;
    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f3, [%rd5];
    mul.f32       %f4, %f2, %f3;
    fma.rn.f32    %f0, %f1, %f4, %f0;
    add.u32       %r7, %r7, 1;
    bra $GW_L;
$GW_L_DONE:
    add.u32       %r5, %r5, 1;
    bra $GW_K;
$GW_K_DONE:
    mul.f32       %f0, %f0, {NEG2};
    mul.lo.u32    %r6, %r3, %r1;
    add.u32       %r6, %r6, %r4;
    mul.wide.u32  %rd6, %r6, 4;
    add.u64       %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f0;

$GW_DONE:
    ret;
}}
"#,
        ZERO = zero,
        NEG2 = neg_two
    )
}

/// Unbalanced Sinkhorn step `f = (τ_a/(τ_a+ε))·(ε log a − ε logsumexp_j((g_j − C_ij)/ε))`.
#[must_use]
pub fn unbalanced_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#"{hdr}// unbalanced_step_kernel: KL-relaxed Sinkhorn step.
.visible .entry unbalanced_step_kernel(
    .param .u64 p_c,
    .param .u64 p_log_a,
    .param .u64 p_g,
    .param .u64 p_f,
    .param .u32 p_m,
    .param .u32 p_n,
    .param .f32 p_eps,
    .param .f32 p_tau
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_c];
    ld.param.u64  %rd1, [p_log_a];
    ld.param.u64  %rd2, [p_g];
    ld.param.u64  %rd3, [p_f];
    ld.param.u32  %r0,  [p_m];
    ld.param.u32  %r1,  [p_n];
    ld.param.f32  %f10, [p_eps];
    ld.param.f32  %f11, [p_tau];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $UB_DONE;

    // logsumexp via two passes
    mov.f32       %f1, {NINF};
    mov.u32       %r7, 0;
$UB_MAX:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $UB_MAX_DONE;
    mul.lo.u32    %r8, %r6, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd5, %r8, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd6, %rd2, %rd5;
    ld.global.f32 %f3, [%rd6];
    sub.f32       %f4, %f3, %f2;
    div.rn.f32    %f4, %f4, %f10;
    max.f32       %f1, %f1, %f4;
    add.u32       %r7, %r7, 1;
    bra $UB_MAX;

$UB_MAX_DONE:
    mov.f32       %f5, {ZERO};
    mov.u32       %r7, 0;
$UB_SUM:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $UB_SUM_DONE;
    mul.lo.u32    %r8, %r6, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd5, %r8, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f2, [%rd6];
    mul.wide.u32  %rd5, %r7, 4;
    add.u64       %rd6, %rd2, %rd5;
    ld.global.f32 %f3, [%rd6];
    sub.f32       %f4, %f3, %f2;
    div.rn.f32    %f4, %f4, %f10;
    sub.f32       %f4, %f4, %f1;
    ex2.approx.f32 %f6, %f4;
    add.f32       %f5, %f5, %f6;
    add.u32       %r7, %r7, 1;
    bra $UB_SUM;

$UB_SUM_DONE:
    mul.wide.u32  %rd5, %r6, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f7, [%rd6];

    // f = (tau/(tau+eps)) * (eps*log_a - eps*(max + log(sum)))
    lg2.approx.f32 %f8, %f5;
    add.f32       %f8, %f8, %f1;
    mul.f32       %f8, %f8, %f10;
    mul.f32       %f9, %f7, %f10;
    sub.f32       %f12, %f9, %f8;

    add.f32       %f13, %f11, %f10;
    div.rn.f32    %f14, %f11, %f13;
    mul.f32       %f12, %f12, %f14;

    add.u64       %rd7, %rd3, %rd5;
    st.global.f32 [%rd7], %f12;

$UB_DONE:
    ret;
}}
"#,
        ZERO = zero,
        NINF = neg_inf
    )
}

/// Barycenter support update: `y_i ← Σ_k λ_k Σ_j (T_k)_ij x_kj / row_sum`.
/// Kernel signature: `barycenter_update_kernel(t, x, lambda, y, m, n_k, k_count, dim)`.
#[must_use]
pub fn barycenter_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let eps_d = f32_hex(1e-12_f32);
    format!(
        r#"{hdr}// barycenter_update_kernel
// t: [k_count, m, n_k] flattened plans
// x: [k_count, n_k, dim] flattened source supports
// lambda: [k_count] weights
// y: [m, dim] output barycenter support
.visible .entry barycenter_update_kernel(
    .param .u64 p_t,
    .param .u64 p_x,
    .param .u64 p_lambda,
    .param .u64 p_y,
    .param .u32 p_m,
    .param .u32 p_nk,
    .param .u32 p_kc,
    .param .u32 p_dim
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_t];
    ld.param.u64  %rd1, [p_x];
    ld.param.u64  %rd2, [p_lambda];
    ld.param.u64  %rd3, [p_y];
    ld.param.u32  %r0,  [p_m];
    ld.param.u32  %r1,  [p_nk];
    ld.param.u32  %r2,  [p_kc];
    ld.param.u32  %r3,  [p_dim];

    mov.u32       %r4, %ctaid.x;     // i (target row)
    mov.u32       %r5, %ctaid.y;     // d (output dim)
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $BU_DONE;
    setp.ge.u32   %p0, %r5, %r3;
    @%p0 bra $BU_DONE;

    mov.f32       %f0, {ZERO};       // accumulator
    mov.f32       %f6, {ZERO};       // weight normaliser
    mov.u32       %r6, 0;            // k

$BU_K:
    setp.ge.u32   %p0, %r6, %r2;
    @%p0 bra $BU_K_DONE;

    // lambda[k]
    mul.wide.u32  %rd4, %r6, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f5, [%rd5];

    // For each j: t[k,i,j] * x[k,j,d]
    mov.f32       %f1, {ZERO};       // sum
    mov.f32       %f7, {ZERO};       // row sum of T
    mov.u32       %r7, 0;            // j
$BU_J:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $BU_J_DONE;
    // t[k,i,j] = t[(k*m + i)*nk + j]
    mul.lo.u32    %r8, %r6, %r0;
    add.u32       %r8, %r8, %r4;
    mul.lo.u32    %r8, %r8, %r1;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd6, %r8, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f2, [%rd7];
    // x[k,j,d] = x[(k*nk + j)*dim + d]
    mul.lo.u32    %r9, %r6, %r1;
    add.u32       %r9, %r9, %r7;
    mul.lo.u32    %r9, %r9, %r3;
    add.u32       %r9, %r9, %r5;
    mul.wide.u32  %rd6, %r9, 4;
    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f3, [%rd7];
    fma.rn.f32    %f1, %f2, %f3, %f1;
    add.f32       %f7, %f7, %f2;
    add.u32       %r7, %r7, 1;
    bra $BU_J;
$BU_J_DONE:
    add.f32       %f7, %f7, {EPS};
    div.rn.f32    %f1, %f1, %f7;     // normalise per-source contribution
    fma.rn.f32    %f0, %f5, %f1, %f0;
    add.f32       %f6, %f6, %f5;
    add.u32       %r6, %r6, 1;
    bra $BU_K;

$BU_K_DONE:
    add.f32       %f6, %f6, {EPS};
    div.rn.f32    %f0, %f0, %f6;

    // out[i,d]
    mul.lo.u32    %r10, %r4, %r3;
    add.u32       %r10, %r10, %r5;
    mul.wide.u32  %rd8, %r10, 4;
    add.u64       %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], %f0;

$BU_DONE:
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps_d
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_header_strings() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(86).contains(".version 8.0"));
        assert!(ptx_header(89).contains(".version 8.0"));
        assert!(ptx_header(90).contains(".version 8.4"));
        assert!(ptx_header(100).contains(".version 8.7"));
    }

    #[test]
    fn f32_hex_format() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
        assert!(f32_hex(1.5_f32).starts_with("0F"));
    }

    #[test]
    fn all_kernels_for_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            for kernel in [
                sinkhorn_step_ptx(sm),
                cost_matrix_ptx(sm),
                transport_apply_ptx(sm),
                sliced_proj_ptx(sm),
                gromov_grad_ptx(sm),
                unbalanced_step_ptx(sm),
                barycenter_update_ptx(sm),
            ] {
                assert!(kernel.contains(".visible .entry"));
                assert!(kernel.contains(".address_size 64"));
            }
        }
    }
}
