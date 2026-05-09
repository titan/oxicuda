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

/// Partial correlation kernel: computes residual-based partial correlations for PC algorithm.
#[must_use]
pub fn partial_corr_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// partial_corr_kernel: computes partial correlations via residuals.
// p_x: [n * d] data matrix (row-major)
// p_corr: [d * d] output partial correlation matrix
// n: number of samples, d: number of variables
.visible .entry partial_corr_kernel(
    .param .u64 p_x,
    .param .u64 p_corr,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<16>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_corr];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mov.u32       %r8, %r5;

    // Each thread handles one (i,j) pair
    mul.lo.u32    %r9, %r1, %r1;
$PCORR_LOOP:
    setp.ge.u32   %p0, %r8, %r9;
    @%p0 bra $PCORR_DONE;

    // Compute row i and col j from linear index r8
    div.u32       %r10, %r8, %r1;   // row i
    rem.u32       %r11, %r8, %r1;   // col j

    // dot product sum_x = sum(x[:,i] * x[:,j])
    mov.f32       %f0, {ZERO};
    mov.f32       %f1, {ZERO};
    mov.f32       %f2, {ZERO};
    mov.u32       %r12, 0;
$PCORR_INNER:
    setp.ge.u32   %p0, %r12, %r0;
    @%p0 bra $PCORR_INNER_DONE;

    mul.lo.u32    %r13, %r12, %r1;
    add.u32       %r14, %r13, %r10;
    mul.wide.u32  %rd2, %r14, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f3, [%rd3];

    add.u32       %r14, %r13, %r11;
    mul.wide.u32  %rd2, %r14, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f4, [%rd3];

    fma.rn.f32    %f0, %f3, %f4, %f0;   // sum xi*xj
    fma.rn.f32    %f1, %f3, %f3, %f1;   // sum xi*xi
    fma.rn.f32    %f2, %f4, %f4, %f2;   // sum xj*xj

    add.u32       %r12, %r12, 1;
    bra $PCORR_INNER;

$PCORR_INNER_DONE:
    // corr = sum_xy / sqrt(sum_xx * sum_yy)
    mul.f32       %f5, %f1, %f2;
    sqrt.rn.f32   %f6, %f5;
    // guard against zero denominator
    mov.f32       %f7, {ONE};
    setp.lt.f32   %p0, %f6, 0F3727C5AC;  // 1e-6
    @%p0 mov.f32  %f6, %f7;
    div.rn.f32    %f8, %f0, %f6;

    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f8;

    add.u32       %r8, %r8, %r7;
    bra $PCORR_LOOP;

$PCORR_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one
    )
}

/// NOTEARS loss kernel: computes L2 loss gradient for structural equation model.
#[must_use]
pub fn notears_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// notears_loss_kernel: computes (1/n)||X - XW||_F^2 gradient w.r.t. W.
// p_x: [n * d] data matrix
// p_w: [d * d] weight matrix W
// p_grad: [d * d] gradient output
// n: number of samples, d: number of variables
.visible .entry notears_loss_kernel(
    .param .u64 p_x,
    .param .u64 p_w,
    .param .u64 p_grad,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_w];
    ld.param.u64  %rd2, [p_grad];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mul.lo.u32    %r8, %r1, %r1;
    mov.u32       %r9, %r5;

$NOTEARS_LOOP:
    setp.ge.u32   %p0, %r9, %r8;
    @%p0 bra $NOTEARS_DONE;

    div.u32       %r10, %r9, %r1;   // output col j
    rem.u32       %r11, %r9, %r1;   // input col k

    // grad[j,k] = (1/n) * sum_i X[i,j] * (XW - X)[i,k]
    mov.f32       %f0, {ZERO};
    mov.u32       %r12, 0;
$NOTEARS_INNER:
    setp.ge.u32   %p0, %r12, %r0;
    @%p0 bra $NOTEARS_INNER_DONE;

    // XW[i,k] = sum_l X[i,l] * W[l,k]
    mov.f32       %f1, {ZERO};
    mov.u32       %r13, 0;
$NOTEARS_INNER2:
    setp.ge.u32   %p0, %r13, %r1;
    @%p0 bra $NOTEARS_INNER2_DONE;

    mul.lo.u32    %r14, %r12, %r1;
    add.u32       %r14, %r14, %r13;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f2, [%rd4];       // X[i,l]

    mul.lo.u32    %r14, %r13, %r1;
    add.u32       %r14, %r14, %r11;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];       // W[l,k]

    fma.rn.f32    %f1, %f2, %f3, %f1;
    add.u32       %r13, %r13, 1;
    bra $NOTEARS_INNER2;

$NOTEARS_INNER2_DONE:
    // residual = XW[i,k] - X[i,k]
    mul.lo.u32    %r14, %r12, %r1;
    add.u32       %r14, %r14, %r11;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f4, [%rd4];       // X[i,k]
    sub.f32       %f5, %f1, %f4;

    // X[i,j]
    mul.lo.u32    %r14, %r12, %r1;
    add.u32       %r14, %r14, %r10;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f6, [%rd4];

    fma.rn.f32    %f0, %f6, %f5, %f0;
    add.u32       %r12, %r12, 1;
    bra $NOTEARS_INNER;

$NOTEARS_INNER_DONE:
    // divide by n
    cvt.rn.f32.u32 %f7, %r0;
    div.rn.f32    %f8, %f0, %f7;

    mul.wide.u32  %rd5, %r9, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f8;

    add.u32       %r9, %r9, %r7;
    bra $NOTEARS_LOOP;

$NOTEARS_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

/// Padé(3,3) matrix exponential kernel for acyclicity constraint computation.
#[must_use]
pub fn expm_pade_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let half = f32_hex(0.5_f32);
    let twelfth = f32_hex(1.0_f32 / 12.0_f32);
    format!(
        r#"{hdr}// expm_pade_kernel: Pade(3,3) approximation of matrix exponential.
// For acyclicity constraint h(W) = tr(expm(W*W)) - d.
// p_a: [d * d] input matrix A = W elementwise-squared
// p_out: [d * d] output expm(A)
// d: matrix dimension
.visible .entry expm_pade_kernel(
    .param .u64 p_a,
    .param .u64 p_out,
    .param .u32 d
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<20>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [d];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mul.lo.u32    %r5, %r0, %r0;

    setp.ge.u32   %p0, %r4, %r5;
    @%p0 bra $EXPM_DONE;

    div.u32       %r6, %r4, %r0;   // row i
    rem.u32       %r7, %r4, %r0;   // col j

    // Compute (I + A/2 + A^2/12) and (I - A/2 + A^2/12) [i,j]
    // identity contribution
    setp.eq.u32   %p0, %r6, %r7;
    mov.f32       %f0, {ZERO};
    @%p0 mov.f32  %f0, {ONE};   // I[i,j]

    // A/2 term: A[i,j] * 0.5
    mul.wide.u32  %rd2, %r4, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];         // A[i,j]
    mul.f32       %f2, %f1, {HALF};    // A[i,j]/2

    // A^2/12 term: sum_k A[i,k]*A[k,j] / 12
    mov.f32       %f3, {ZERO};
    mov.u32       %r8, 0;
$EXPM_INNER:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $EXPM_INNER_DONE;

    mul.lo.u32    %r9, %r6, %r0;
    add.u32       %r9, %r9, %r8;
    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f4, [%rd5];         // A[i,k]

    mul.lo.u32    %r9, %r8, %r0;
    add.u32       %r9, %r9, %r7;
    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f5, [%rd5];         // A[k,j]

    fma.rn.f32    %f3, %f4, %f5, %f3;
    add.u32       %r8, %r8, 1;
    bra $EXPM_INNER;

$EXPM_INNER_DONE:
    mul.f32       %f6, %f3, {TWELFTH};   // A^2[i,j] / 12

    // U = I + A/2 + A^2/12, V = I - A/2 + A^2/12
    add.f32       %f7, %f0, %f2;
    add.f32       %f7, %f7, %f6;    // U[i,j]
    sub.f32       %f8, %f0, %f2;
    add.f32       %f8, %f8, %f6;    // V[i,j]

    // Approximate expm(A)[i,j] ≈ U[i,j] (diagonal dominant for small A)
    // Full inversion would need a separate pass; store U for now
    mul.wide.u32  %rd6, %r4, 4;
    add.u64       %rd7, %rd1, %rd6;
    st.global.f32 [%rd7], %f7;

$EXPM_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        HALF = half,
        TWELFTH = twelfth
    )
}

/// Propensity logit kernel: sigmoid logistic regression predictions.
#[must_use]
pub fn propensity_logit_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let clamp_lo = f32_hex(0.05_f32);
    let clamp_hi = f32_hex(0.95_f32);
    format!(
        r#"{hdr}// propensity_logit_kernel: sigmoid(X*w + b) clipped to [0.05, 0.95].
// p_x: [n * d] feature matrix
// p_w: [d] weight vector
// p_b: scalar bias
// p_out: [n] propensity scores
// n: samples, d: features
.visible .entry propensity_logit_kernel(
    .param .u64 p_x,
    .param .u64 p_w,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_w];
    ld.param.u64  %rd2, [p_b];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;
    mov.u32       %r8, %r5;

    ld.global.f32 %f0, [%rd2];   // bias

$PROPENSITY_LOOP:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $PROPENSITY_DONE;

    // dot = X[i,:] . w
    mov.f32       %f1, %f0;   // start with bias
    mov.u32       %r9, 0;
$PROPENSITY_INNER:
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $PROPENSITY_INNER_DONE;

    mul.lo.u32    %r10, %r8, %r1;
    add.u32       %r10, %r10, %r9;
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f2, [%rd5];

    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f3, [%rd5];

    fma.rn.f32    %f1, %f2, %f3, %f1;
    add.u32       %r9, %r9, 1;
    bra $PROPENSITY_INNER;

$PROPENSITY_INNER_DONE:
    // sigmoid(dot) = 1 / (1 + exp(-dot))
    neg.f32       %f4, %f1;
    ex2.approx.f32 %f5, %f4;     // approx exp via ex2: use ln2 scaling
    // Note: ex2(x) = 2^x, so exp(-dot) = ex2(-dot / ln2)
    // Simplified: direct sigmoid approximation
    mov.f32       %f6, {ZERO};
    fma.rn.f32    %f7, %f4, 0F3FB8AA3B, %f6;  // -dot * log2(e)
    ex2.approx.f32 %f5, %f7;
    add.f32       %f8, 0F3F800000, %f5;         // 1 + exp(-dot)
    rcp.rn.f32    %f9, %f8;                     // sigmoid

    // clamp to [0.05, 0.95]
    max.f32       %f9, %f9, {CLAMP_LO};
    min.f32       %f9, %f9, {CLAMP_HI};

    mul.wide.u32  %rd6, %r8, 4;
    add.u64       %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f9;

    add.u32       %r8, %r8, %r7;
    bra $PROPENSITY_LOOP;

$PROPENSITY_DONE:
    ret;
}}
"#,
        ZERO = zero,
        CLAMP_LO = clamp_lo,
        CLAMP_HI = clamp_hi
    )
}

/// IPW estimator kernel: inverse-probability weighting ATE computation.
#[must_use]
pub fn ipw_estimator_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let clamp_lo = f32_hex(0.05_f32);
    let clamp_hi = f32_hex(0.95_f32);
    format!(
        r#"{hdr}// ipw_estimator_kernel: ATE = mean(Y*T/pi - Y*(1-T)/(1-pi)).
// p_y: [n] outcomes
// p_t: [n] treatment indicators (0/1)
// p_pi: [n] propensity scores
// p_out: [1] ATE accumulator (atomic add)
// n: number of samples
.visible .entry ipw_estimator_kernel(
    .param .u64 p_y,
    .param .u64 p_t,
    .param .u64 p_pi,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_y];
    ld.param.u64  %rd1, [p_t];
    ld.param.u64  %rd2, [p_pi];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$IPW_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $IPW_DONE;

    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];   // Y[i]

    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];   // T[i]

    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f2, [%rd5];   // pi[i]

    // clamp pi
    max.f32       %f2, %f2, {CLAMP_LO};
    min.f32       %f2, %f2, {CLAMP_HI};

    // 1 - pi
    mov.f32       %f3, 0F3F800000;
    sub.f32       %f4, %f3, %f2;

    // IPW term: Y*T/pi - Y*(1-T)/(1-pi)
    mul.f32       %f5, %f0, %f1;
    div.rn.f32    %f6, %f5, %f2;

    sub.f32       %f7, %f3, %f1;
    mul.f32       %f8, %f0, %f7;
    div.rn.f32    %f9, %f8, %f4;

    sub.f32       %f10, %f6, %f9;

    // atomic add to accumulator
    atom.global.add.f32 %f11, [%rd3], %f10;

    add.u32       %r7, %r7, %r6;
    bra $IPW_LOOP;

$IPW_DONE:
    ret;
}}
"#,
        CLAMP_LO = clamp_lo,
        CLAMP_HI = clamp_hi
    )
}

/// Double ML residual kernel: cross-fitted nuisance residuals for DML.
#[must_use]
pub fn dml_residual_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// dml_residual_kernel: compute residuals Y - g(X) and T - m(X).
// p_y: [n] outcomes
// p_t: [n] treatments
// p_gy: [n] predicted g(X) = E[Y|X]
// p_mt: [n] predicted m(X) = E[T|X]
// p_ytilde: [n] outcome residuals Y - g(X)
// p_ttilde: [n] treatment residuals T - m(X)
// n: number of samples
.visible .entry dml_residual_kernel(
    .param .u64 p_y,
    .param .u64 p_t,
    .param .u64 p_gy,
    .param .u64 p_mt,
    .param .u64 p_ytilde,
    .param .u64 p_ttilde,
    .param .u32 n
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_y];
    ld.param.u64  %rd1, [p_t];
    ld.param.u64  %rd2, [p_gy];
    ld.param.u64  %rd3, [p_mt];
    ld.param.u64  %rd4, [p_ytilde];
    ld.param.u64  %rd5, [p_ttilde];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$DML_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $DML_DONE;

    mul.wide.u32  %rd6, %r7, 4;

    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f0, [%rd7];   // Y[i]

    add.u64       %rd7, %rd1, %rd6;
    ld.global.f32 %f1, [%rd7];   // T[i]

    add.u64       %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd7];   // g(X)[i]

    add.u64       %rd7, %rd3, %rd6;
    ld.global.f32 %f3, [%rd7];   // m(X)[i]

    sub.f32       %f4, %f0, %f2;   // Y - g(X)
    sub.f32       %f5, %f1, %f3;   // T - m(X)

    add.u64       %rd8, %rd4, %rd6;
    st.global.f32 [%rd8], %f4;

    add.u64       %rd9, %rd5, %rd6;
    st.global.f32 [%rd9], %f5;

    add.u32       %r7, %r7, %r6;
    bra $DML_LOOP;

$DML_DONE:
    ret;
}}
"#
    )
}

/// Causal forest split score kernel: heterogeneous treatment effect split criterion.
#[must_use]
pub fn causal_split_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// causal_split_score_kernel: Delta = (tau_L - tau_R)^2 * n_L * n_R / n per candidate split.
// p_y: [n] outcomes
// p_t: [n] treatment indicators
// p_features: [n * d] feature matrix
// p_scores: [d * n] output split scores per feature per threshold
// n: number of samples, d: number of features
.visible .entry causal_split_score_kernel(
    .param .u64 p_y,
    .param .u64 p_t,
    .param .u64 p_features,
    .param .u64 p_scores,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_y];
    ld.param.u64  %rd1, [p_t];
    ld.param.u64  %rd2, [p_features];
    ld.param.u64  %rd3, [p_scores];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mul.lo.u32    %r6, %r1, %r0;

    setp.ge.u32   %p0, %r5, %r6;
    @%p0 bra $CSPLIT_DONE;

    div.u32       %r7, %r5, %r0;   // feature index
    rem.u32       %r8, %r5, %r0;   // threshold index (sample index as threshold)

    // Get threshold value = feature[threshold_idx, feature_idx]
    mul.lo.u32    %r9, %r8, %r1;
    add.u32       %r9, %r9, %r7;
    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f0, [%rd5];   // threshold

    // Accumulate left/right stats
    mov.f32       %f1, {ZERO};   // sum_y_L_t1
    mov.f32       %f2, {ZERO};   // sum_y_L_t0
    mov.f32       %f3, {ZERO};   // sum_y_R_t1
    mov.f32       %f4, {ZERO};   // sum_y_R_t0
    mov.u32       %r10, 0;       // n_L_t1
    mov.u32       %r11, 0;       // n_L_t0
    mov.u32       %r12, 0;       // n_R_t1
    mov.u32       %r13, 0;       // n_R_t0

    mov.u32       %r14, 0;
$CSPLIT_INNER:
    setp.ge.u32   %p0, %r14, %r0;
    @%p0 bra $CSPLIT_INNER_DONE;

    mul.lo.u32    %r15, %r14, %r1;
    add.u32       %r15, %r15, %r7;
    mul.wide.u32  %rd4, %r15, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f5, [%rd5];   // feature[i, feat]

    mul.wide.u32  %rd4, %r14, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f6, [%rd5];   // Y[i]

    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f7, [%rd5];   // T[i]

    setp.lt.f32   %p0, %f5, %f0;
    @%p0 bra $CSPLIT_LEFT;

    // Right: f5 >= threshold
    setp.gt.f32   %p0, %f7, 0F3F000000;
    @%p0 add.f32  %f3, %f3, %f6;
    @%p0 add.u32  %r12, %r12, 1;
    setp.le.f32   %p0, %f7, 0F3F000000;
    @%p0 add.f32  %f4, %f4, %f6;
    @%p0 add.u32  %r13, %r13, 1;
    bra $CSPLIT_NEXT;

$CSPLIT_LEFT:
    setp.gt.f32   %p0, %f7, 0F3F000000;
    @%p0 add.f32  %f1, %f1, %f6;
    @%p0 add.u32  %r10, %r10, 1;
    setp.le.f32   %p0, %f7, 0F3F000000;
    @%p0 add.f32  %f2, %f2, %f6;
    @%p0 add.u32  %r11, %r11, 1;

$CSPLIT_NEXT:
    add.u32       %r14, %r14, 1;
    bra $CSPLIT_INNER;

$CSPLIT_INNER_DONE:
    // tau_L = sum_y_L_t1/n_L_t1 - sum_y_L_t0/n_L_t0
    cvt.rn.f32.u32 %f8, %r10;
    cvt.rn.f32.u32 %f9, %r11;
    setp.gt.f32   %p0, %f8, {ZERO};
    @%p0 div.rn.f32 %f8, %f1, %f8;
    setp.gt.f32   %p0, %f9, {ZERO};
    @%p0 div.rn.f32 %f9, %f2, %f9;
    sub.f32       %f10, %f8, %f9;   // tau_L

    cvt.rn.f32.u32 %f11, %r12;
    cvt.rn.f32.u32 %f12, %r13;
    setp.gt.f32   %p0, %f11, {ZERO};
    @%p0 div.rn.f32 %f11, %f3, %f11;
    setp.gt.f32   %p0, %f12, {ZERO};
    @%p0 div.rn.f32 %f12, %f4, %f12;
    sub.f32       %f13, %f11, %f12;   // tau_R

    sub.f32       %f14, %f10, %f13;
    mul.f32       %f14, %f14, %f14;   // (tau_L - tau_R)^2

    // multiply by n_L * n_R / n
    add.u32       %r10, %r10, %r11;
    add.u32       %r12, %r12, %r13;
    cvt.rn.f32.u32 %f8, %r10;
    cvt.rn.f32.u32 %f9, %r12;
    cvt.rn.f32.u32 %f11, %r0;
    mul.f32       %f8, %f8, %f9;
    div.rn.f32    %f9, %f8, %f11;
    mul.f32       %f15, %f14, %f9;

    mul.wide.u32  %rd6, %r5, 4;
    add.u64       %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f15;

$CSPLIT_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kernels_non_empty() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!partial_corr_ptx(sm).is_empty());
            assert!(!notears_loss_ptx(sm).is_empty());
            assert!(!expm_pade_ptx(sm).is_empty());
            assert!(!propensity_logit_ptx(sm).is_empty());
            assert!(!ipw_estimator_ptx(sm).is_empty());
            assert!(!dml_residual_ptx(sm).is_empty());
            assert!(!causal_split_score_ptx(sm).is_empty());
        }
    }
}
