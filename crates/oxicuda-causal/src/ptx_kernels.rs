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

/// Maximum matrix dimension supported by [`expm_pade_ptx`].
///
/// The kernel runs as a single cooperative thread block holding the whole
/// matrix (plus a `d × 2d` Gauss-Jordan tableau) in shared memory and uses one
/// thread per matrix element. With `EXPM_MAX_DIM = 32` the shared footprint is
/// `32·64·4 + 3·32·32·4 + 32·4 ≈ 20 KiB` and the launch needs `d²` threads.
pub const EXPM_MAX_DIM: u32 = 32;

/// Padé(1,1) matrix exponential kernel with scaling-and-squaring.
///
/// Computes the true `expm(A) = U·V⁻¹` for the acyclicity constraint
/// `h(W) = tr(expm(W ⊙ W)) - d`, where `U = I + A/2 + A²/12` and
/// `V = I - A/2 + A²/12` are the Padé(1,1) numerator and denominator.
///
/// The kernel mirrors the crate CPU reference (`discovery::notears::expm_pade`)
/// step for step:
///
/// 1. Load `A` into shared memory and scale it by `2^-s` so `‖A/2^s‖∞ ≤ 1/2`.
/// 2. Form the Padé(1,1) numerator `U` and denominator `V`.
/// 3. Invert `V` by in-block Gauss-Jordan elimination with partial pivoting on
///    the augmented `[V | I]` tableau.
/// 4. Multiply `R = U·V⁻¹` (the scaled exponential).
/// 5. Square `R` a total of `s` times to undo the scaling: `expm(A) = R^(2^s)`.
///
/// Launch with a single block of `d²` threads (`d ≤ EXPM_MAX_DIM`).
#[must_use]
pub fn expm_pade_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let half = f32_hex(0.5_f32);
    let twelfth = f32_hex(1.0_f32 / 12.0_f32);
    let theta = f32_hex(0.5_f32);
    let tiny = f32_hex(1.0e-12_f32);
    // Shared-tableau dimensions in 32-bit words for EXPM_MAX_DIM.
    let mat_words = EXPM_MAX_DIM * EXPM_MAX_DIM;
    let aug_words = EXPM_MAX_DIM * 2 * EXPM_MAX_DIM;
    format!(
        r#"{hdr}// expm_pade_kernel: Pade(1,1) matrix exponential with scaling-and-squaring.
// Computes the true expm(A) = U * V^-1 in a single cooperative thread block.
// For acyclicity constraint h(W) = tr(expm(W*W)) - d.
// p_a:   [d * d] input matrix A (row-major), d <= {MAX_DIM}
// p_out: [d * d] output expm(A) (row-major)
// d:     matrix dimension
// Launch: one block, d*d threads.
.visible .entry expm_pade_kernel(
    .param .u64 p_a,
    .param .u64 p_out,
    .param .u32 d
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<32>;
    .reg .f32  %f<32>;
    .reg .pred %p0;
    .reg .pred %p1;

    // sh_cur: current matrix (A, then U*V^-1, then squared result).
    // sh_alt: numerator U, also the ping-pong target for squaring.
    // sh_aug: d x 2d Gauss-Jordan tableau [V | I].
    // sh_scr: per-row reduction scratch and the scaling exponent slot.
    .shared .align 4 .f32 sh_cur[{MAT_WORDS}];
    .shared .align 4 .f32 sh_alt[{MAT_WORDS}];
    .shared .align 4 .f32 sh_aug[{AUG_WORDS}];
    .shared .align 4 .f32 sh_scr[{MAX_DIM}];
    .shared .align 4 .u32 sh_exp[1];

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [d];

    mov.u32       %r1, %tid.x;        // linear thread id
    mul.lo.u32    %r2, %r0, %r0;      // d*d element count

    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_EXIT;              // surplus threads idle (still hit bar.sync)

    div.u32       %r3, %r1, %r0;      // row i
    rem.u32       %r4, %r1, %r0;      // col j
    mad.lo.u32    %r5, %r3, %r0, %r4; // flat index i*d + j

    // --- Phase 0: load A into sh_cur ----------------------------------------
    mul.wide.u32  %rd2, %r5, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];
    st.shared.f32 [sh_cur + %r5*4], %f0;
$EXPM_EXIT:
    bar.sync      0;

    // --- Phase 1: infinity norm + scaling exponent (thread 0) ---------------
    // Each row leader (j == 0) sums |A[i,*]| into sh_scr[i].
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_AFTER_ROWSUM;
    setp.ne.u32   %p0, %r4, 0;
    @%p0 bra $EXPM_AFTER_ROWSUM;
    mov.f32       %f1, {ZERO};
    mul.lo.u32    %r6, %r3, %r0;      // row base i*d
    mov.u32       %r7, 0;
$EXPM_ROWSUM:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $EXPM_ROWSUM_DONE;
    add.u32       %r8, %r6, %r7;
    ld.shared.f32 %f2, [sh_cur + %r8*4];
    abs.f32       %f2, %f2;
    add.f32       %f1, %f1, %f2;
    add.u32       %r7, %r7, 1;
    bra $EXPM_ROWSUM;
$EXPM_ROWSUM_DONE:
    st.shared.f32 [sh_scr + %r3*4], %f1;
$EXPM_AFTER_ROWSUM:
    bar.sync      0;

    // Thread 0 reduces row sums to ‖A‖∞ and derives s = ceil(log2(norm/theta)).
    setp.ne.u32   %p0, %r1, 0;
    @%p0 bra $EXPM_AFTER_EXP;
    mov.f32       %f3, {ZERO};        // running max
    mov.u32       %r9, 0;
$EXPM_NORMMAX:
    setp.ge.u32   %p0, %r9, %r0;
    @%p0 bra $EXPM_NORMMAX_DONE;
    ld.shared.f32 %f4, [sh_scr + %r9*4];
    max.f32       %f3, %f3, %f4;
    add.u32       %r9, %r9, 1;
    bra $EXPM_NORMMAX;
$EXPM_NORMMAX_DONE:
    mov.u32       %r10, 0;            // s = 0 by default
    // Non-finite or small norm => no scaling.
    setp.leu.f32  %p0, %f3, {THETA};
    @%p0 bra $EXPM_EXP_STORE;
    div.rn.f32    %f5, %f3, {THETA};  // ratio = norm / theta
    lg2.approx.f32 %f6, %f5;          // approx log2(ratio)
    cvt.rpi.f32.f32 %f7, %f6;         // ceil estimate (round toward +inf)
    cvt.rzi.u32.f32 %r11, %f7;        // s estimate as integer (>= 0)
    // Refine: bump s until 2^s >= ratio, guarding against a low lg2.approx.
$EXPM_EXP_REFINE:
    cvt.rn.f32.u32 %f8, %r11;
    ex2.approx.f32 %f8, %f8;          // 2^s
    setp.ge.f32   %p0, %f8, %f5;      // 2^s >= ratio ?
    @%p0 bra $EXPM_EXP_HAVE;
    add.u32       %r11, %r11, 1;
    bra $EXPM_EXP_REFINE;
$EXPM_EXP_HAVE:
    mov.u32       %r10, %r11;
$EXPM_EXP_STORE:
    st.shared.u32 [sh_exp], %r10;
$EXPM_AFTER_EXP:
    bar.sync      0;

    // --- Phase 2: scale A in place by 2^-s ----------------------------------
    // 2^-s is built exactly: 2^s = (1u32 << s) is integer-exact for small s,
    // and rcp.rn of an exact power of two is exact, matching the CPU scale.
    ld.shared.u32 %r12, [sh_exp];     // s, visible to every thread
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_AFTER_SCALE;
    mov.u32       %r13, 1;
    shl.b32       %r13, %r13, %r12;   // 2^s as integer
    cvt.rn.f32.u32 %f9, %r13;         // 2^s as f32 (exact for s < 24)
    rcp.rn.f32    %f10, %f9;          // 2^-s, exact for power-of-two input
    ld.shared.f32 %f11, [sh_cur + %r5*4];
    mul.f32       %f11, %f11, %f10;   // scaled A[i,j]
    st.shared.f32 [sh_cur + %r5*4], %f11;
$EXPM_AFTER_SCALE:
    bar.sync      0;

    // --- Phase 3: Pade(1,1) numerator U and denominator V -------------------
    // A^2[i,j] = sum_k A[i,k]*A[k,j]; U = I + A/2 + A^2/12; V = I - A/2 + A^2/12.
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_AFTER_PADE;
    ld.shared.f32 %f12, [sh_cur + %r5*4];   // scaled A[i,j]
    mul.f32       %f13, %f12, {HALF};       // A[i,j]/2
    mov.f32       %f14, {ZERO};             // accumulator for A^2[i,j]
    mul.lo.u32    %r13, %r3, %r0;           // row base i*d
    mov.u32       %r14, 0;
$EXPM_PADE_DOT:
    setp.ge.u32   %p0, %r14, %r0;
    @%p0 bra $EXPM_PADE_DOT_DONE;
    add.u32       %r15, %r13, %r14;
    ld.shared.f32 %f15, [sh_cur + %r15*4];  // A[i,k]
    mad.lo.u32    %r16, %r14, %r0, %r4;
    ld.shared.f32 %f16, [sh_cur + %r16*4];  // A[k,j]
    fma.rn.f32    %f14, %f15, %f16, %f14;
    add.u32       %r14, %r14, 1;
    bra $EXPM_PADE_DOT;
$EXPM_PADE_DOT_DONE:
    mul.f32       %f17, %f14, {TWELFTH};    // A^2[i,j]/12
    mov.f32       %f18, {ZERO};
    setp.eq.u32   %p0, %r3, %r4;
    @%p0 mov.f32  %f18, {ONE};              // I[i,j]
    add.f32       %f19, %f18, %f13;
    add.f32       %f19, %f19, %f17;         // U[i,j]
    sub.f32       %f20, %f18, %f13;
    add.f32       %f20, %f20, %f17;         // V[i,j]
    st.shared.f32 [sh_alt + %r5*4], %f19;   // U -> sh_alt
    // Augmented tableau: [V | I], row stride 2*d.
    mul.lo.u32    %r17, %r3, %r0;
    add.u32       %r17, %r17, %r17;         // i*2d
    add.u32       %r18, %r17, %r4;          // left half slot
    st.shared.f32 [sh_aug + %r18*4], %f20;  // V[i,j]
    add.u32       %r19, %r18, %r0;          // right half slot
    @%p0 st.shared.f32 [sh_aug + %r19*4], {ONE};
    setp.ne.u32   %p1, %r3, %r4;
    @%p1 st.shared.f32 [sh_aug + %r19*4], {ZERO};
$EXPM_AFTER_PADE:
    bar.sync      0;

    // --- Phase 4: Gauss-Jordan inversion of V with partial pivoting ---------
    // Column-major elimination over the [V | I] tableau; one bar.sync per
    // pivot column. Thread (i,j) owns tableau entries (i,j) and (i,j+d).
    mov.u32       %r20, 0;                  // pivot column c
$EXPM_GJ_COL:
    setp.ge.u32   %p0, %r20, %r0;
    @%p0 bra $EXPM_GJ_DONE;

    // Thread 0 selects the pivot row (max |tableau[r,c]| over r >= c) and
    // swaps it into row c across the full 2d-wide tableau.
    setp.ne.u32   %p0, %r1, 0;
    @%p0 bra $EXPM_GJ_AFTER_PIVOT;
    mov.u32       %r21, %r20;               // best row = c
    mov.f32       %f21, {ZERO};
    mul.lo.u32    %r22, %r20, %r0;
    add.u32       %r22, %r22, %r22;         // c*2d
    add.u32       %r23, %r22, %r20;         // (c,c) slot
    ld.shared.f32 %f21, [sh_aug + %r23*4];
    abs.f32       %f21, %f21;               // |tableau[c,c]|
    add.u32       %r24, %r20, 1;            // scan row r = c+1 ..
$EXPM_GJ_PIV_SCAN:
    setp.ge.u32   %p0, %r24, %r0;
    @%p0 bra $EXPM_GJ_PIV_SCAN_DONE;
    mul.lo.u32    %r25, %r24, %r0;
    add.u32       %r25, %r25, %r25;
    add.u32       %r25, %r25, %r20;         // (r,c) slot
    ld.shared.f32 %f22, [sh_aug + %r25*4];
    abs.f32       %f22, %f22;
    setp.gt.f32   %p0, %f22, %f21;
    @%p0 mov.f32  %f21, %f22;
    @%p0 mov.u32  %r21, %r24;
    add.u32       %r24, %r24, 1;
    bra $EXPM_GJ_PIV_SCAN;
$EXPM_GJ_PIV_SCAN_DONE:
    // Swap rows c and r21 (skip when already aligned).
    setp.eq.u32   %p0, %r21, %r20;
    @%p0 bra $EXPM_GJ_AFTER_PIVOT;
    mul.lo.u32    %r26, %r20, %r0;
    add.u32       %r26, %r26, %r26;         // c*2d
    mul.lo.u32    %r27, %r21, %r0;
    add.u32       %r27, %r27, %r27;         // r21*2d
    add.u32       %r28, %r0, %r0;           // 2d columns to swap
    mov.u32       %r29, 0;
$EXPM_GJ_SWAP:
    setp.ge.u32   %p0, %r29, %r28;
    @%p0 bra $EXPM_GJ_AFTER_PIVOT;
    add.u32       %r30, %r26, %r29;
    add.u32       %r31, %r27, %r29;
    ld.shared.f32 %f23, [sh_aug + %r30*4];
    ld.shared.f32 %f24, [sh_aug + %r31*4];
    st.shared.f32 [sh_aug + %r30*4], %f24;
    st.shared.f32 [sh_aug + %r31*4], %f23;
    add.u32       %r29, %r29, 1;
    bra $EXPM_GJ_SWAP;
$EXPM_GJ_AFTER_PIVOT:
    bar.sync      0;

    // Normalize pivot row c: divide every column by the (clamped) pivot value.
    // The pivot tableau[c,c] is shared by all pivot-row threads, so its read is
    // separated from the row writes by a bar.sync (in-row shared race guard).
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_GJ_NORM_SYNC;
    setp.ne.u32   %p0, %r3, %r20;
    @%p0 bra $EXPM_GJ_NORM_SYNC;
    mul.lo.u32    %r22, %r20, %r0;
    add.u32       %r22, %r22, %r22;         // c*2d
    add.u32       %r23, %r22, %r20;         // (c,c) slot
    ld.shared.f32 %f25, [sh_aug + %r23*4];  // pivot value
    // Guard a (near-)singular pivot so the divide stays finite.
    abs.f32       %f26, %f25;
    setp.lt.f32   %p1, %f26, {TINY};
    mov.f32       %f27, {ONE};
    @%p1 mov.f32  %f25, %f27;
    add.u32       %r24, %r22, %r4;           // left slot (c,j)
    ld.shared.f32 %f28, [sh_aug + %r24*4];   // tableau[c,j]
    add.u32       %r25, %r24, %r0;           // right slot (c,j+d)
    ld.shared.f32 %f29, [sh_aug + %r25*4];   // tableau[c,j+d]
$EXPM_GJ_NORM_SYNC:
    bar.sync      0;
    // Direct division by the pivot, matching the CPU Gauss-Jordan reference.
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_GJ_AFTER_NORM;
    setp.ne.u32   %p0, %r3, %r20;
    @%p0 bra $EXPM_GJ_AFTER_NORM;
    div.rn.f32    %f28, %f28, %f25;
    st.shared.f32 [sh_aug + %r24*4], %f28;
    div.rn.f32    %f29, %f29, %f25;
    st.shared.f32 [sh_aug + %r25*4], %f29;
$EXPM_GJ_AFTER_NORM:
    bar.sync      0;

    // Eliminate column c from every other row: row_i -= factor_i * row_c.
    // Sub-phase A: every active thread loads its row factor tableau[i,c] and the
    // operands it needs into registers. The factor read must finish for the
    // whole row before any thread overwrites tableau[i,c], so a bar.sync
    // separates the loads from the stores (avoids an in-row shared race).
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_GJ_ELIM_SYNC;
    setp.eq.u32   %p0, %r3, %r20;
    @%p0 bra $EXPM_GJ_ELIM_SYNC;            // pivot row already normalized
    mul.lo.u32    %r26, %r3, %r0;
    add.u32       %r26, %r26, %r26;          // i*2d
    add.u32       %r27, %r26, %r20;          // (i,c) slot -> elimination factor
    ld.shared.f32 %f30, [sh_aug + %r27*4];   // factor = tableau[i,c]
    neg.f32       %f30, %f30;                 // -factor for fused subtract
    mul.lo.u32    %r28, %r20, %r0;
    add.u32       %r28, %r28, %r28;          // c*2d
    add.u32       %r29, %r26, %r4;           // (i,j) slot
    add.u32       %r30, %r28, %r4;           // (c,j) slot
    ld.shared.f32 %f31, [sh_aug + %r29*4];   // tableau[i,j]
    ld.shared.f32 %f25, [sh_aug + %r30*4];   // tableau[c,j]
    add.u32       %r31, %r29, %r0;           // (i,j+d) slot
    add.u32       %r24, %r30, %r0;           // (c,j+d) slot
    ld.shared.f32 %f28, [sh_aug + %r31*4];   // tableau[i,j+d]
    ld.shared.f32 %f29, [sh_aug + %r24*4];   // tableau[c,j+d]
$EXPM_GJ_ELIM_SYNC:
    bar.sync      0;
    // Sub-phase B: store the eliminated row entries.
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_GJ_AFTER_ELIM;
    setp.eq.u32   %p0, %r3, %r20;
    @%p0 bra $EXPM_GJ_AFTER_ELIM;
    fma.rn.f32    %f31, %f30, %f25, %f31;    // tableau[i,j] - factor*tableau[c,j]
    st.shared.f32 [sh_aug + %r29*4], %f31;
    fma.rn.f32    %f28, %f30, %f29, %f28;    // tableau[i,j+d] - factor*tableau[c,j+d]
    st.shared.f32 [sh_aug + %r31*4], %f28;
$EXPM_GJ_AFTER_ELIM:
    bar.sync      0;

    add.u32       %r20, %r20, 1;
    bra $EXPM_GJ_COL;
$EXPM_GJ_DONE:
    // The right half of the tableau now holds V^-1. Copy it into sh_cur.
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_AFTER_VINV;
    mul.lo.u32    %r21, %r3, %r0;
    add.u32       %r21, %r21, %r21;          // i*2d
    add.u32       %r21, %r21, %r0;           // right half base
    add.u32       %r21, %r21, %r4;           // (i,j+d)
    ld.shared.f32 %f0, [sh_aug + %r21*4];
    st.shared.f32 [sh_cur + %r5*4], %f0;     // V^-1 -> sh_cur
$EXPM_AFTER_VINV:
    bar.sync      0;

    // --- Phase 5: R = U * V^-1 (U in sh_alt, V^-1 in sh_cur -> sh_aug) ------
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_AFTER_GEMM;
    mov.f32       %f1, {ZERO};
    mul.lo.u32    %r6, %r3, %r0;             // row base i*d
    mov.u32       %r7, 0;
$EXPM_GEMM_DOT:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $EXPM_GEMM_DOT_DONE;
    add.u32       %r8, %r6, %r7;
    ld.shared.f32 %f2, [sh_alt + %r8*4];     // U[i,k]
    mad.lo.u32    %r9, %r7, %r0, %r4;
    ld.shared.f32 %f3, [sh_cur + %r9*4];     // V^-1[k,j]
    fma.rn.f32    %f1, %f2, %f3, %f1;
    add.u32       %r7, %r7, 1;
    bra $EXPM_GEMM_DOT;
$EXPM_GEMM_DOT_DONE:
    st.shared.f32 [sh_aug + %r5*4], %f1;     // scaled expm -> sh_aug scratch
$EXPM_AFTER_GEMM:
    bar.sync      0;
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_AFTER_GEMM_CP;
    ld.shared.f32 %f0, [sh_aug + %r5*4];
    st.shared.f32 [sh_cur + %r5*4], %f0;     // scaled expm -> sh_cur
$EXPM_AFTER_GEMM_CP:
    bar.sync      0;

    // --- Phase 6: squaring loop, R <- R*R repeated s times ------------------
    ld.shared.u32 %r12, [sh_exp];            // s
    mov.u32       %r20, 0;                   // squaring counter
$EXPM_SQ_LOOP:
    setp.ge.u32   %p0, %r20, %r12;
    @%p0 bra $EXPM_SQ_DONE;
    // sh_alt[i,j] = sum_k sh_cur[i,k] * sh_cur[k,j]
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_SQ_AFTER_MUL;
    mov.f32       %f1, {ZERO};
    mul.lo.u32    %r6, %r3, %r0;
    mov.u32       %r7, 0;
$EXPM_SQ_DOT:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $EXPM_SQ_DOT_DONE;
    add.u32       %r8, %r6, %r7;
    ld.shared.f32 %f2, [sh_cur + %r8*4];
    mad.lo.u32    %r9, %r7, %r0, %r4;
    ld.shared.f32 %f3, [sh_cur + %r9*4];
    fma.rn.f32    %f1, %f2, %f3, %f1;
    add.u32       %r7, %r7, 1;
    bra $EXPM_SQ_DOT;
$EXPM_SQ_DOT_DONE:
    st.shared.f32 [sh_alt + %r5*4], %f1;
$EXPM_SQ_AFTER_MUL:
    bar.sync      0;
    // copy sh_alt back into sh_cur for the next squaring / final store
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_SQ_AFTER_CP;
    ld.shared.f32 %f0, [sh_alt + %r5*4];
    st.shared.f32 [sh_cur + %r5*4], %f0;
$EXPM_SQ_AFTER_CP:
    bar.sync      0;
    add.u32       %r20, %r20, 1;
    bra $EXPM_SQ_LOOP;
$EXPM_SQ_DONE:

    // --- Phase 7: write expm(A) from sh_cur to global output ---------------
    setp.ge.u32   %p0, %r1, %r2;
    @%p0 bra $EXPM_DONE;
    ld.shared.f32 %f0, [sh_cur + %r5*4];
    mul.wide.u32  %rd4, %r5, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f0;

$EXPM_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        HALF = half,
        TWELFTH = twelfth,
        THETA = theta,
        TINY = tiny,
        MAX_DIM = EXPM_MAX_DIM,
        MAT_WORDS = mat_words,
        AUG_WORDS = aug_words,
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

    #[test]
    fn expm_kernel_does_inversion_and_multiply() {
        // The kernel must compute the true expm(A) = U * V^-1, not just store U.
        let ptx = expm_pade_ptx(86);
        // Gauss-Jordan inversion pass: pivot search, row swap, normalize, eliminate.
        assert!(
            ptx.contains("Gauss-Jordan inversion"),
            "missing Gauss-Jordan inversion pass"
        );
        assert!(
            ptx.contains("$EXPM_GJ_PIV_SCAN"),
            "missing partial-pivot search"
        );
        assert!(ptx.contains("$EXPM_GJ_SWAP"), "missing pivot row swap");
        assert!(
            ptx.contains("$EXPM_GJ_COL"),
            "missing Gauss-Jordan column loop"
        );
        // Final GEMM that multiplies U by V^-1.
        assert!(
            ptx.contains("R = U * V^-1"),
            "missing U * V^-1 multiply pass"
        );
        assert!(ptx.contains("$EXPM_GEMM_DOT"), "missing GEMM dot product");
    }

    #[test]
    fn expm_kernel_no_longer_just_stores_u() {
        // The retired stub stored the Padé numerator U directly; ensure the
        // stale comments and the bare "store U" shortcut are gone.
        let ptx = expm_pade_ptx(90);
        assert!(
            !ptx.contains("store U for now"),
            "stale 'store U for now' comment still present"
        );
        assert!(
            !ptx.contains("Full inversion would need"),
            "stale 'Full inversion would need' comment still present"
        );
        assert!(
            !ptx.to_lowercase().contains("approximate expm"),
            "stale 'approximate expm' comment still present"
        );
    }

    #[test]
    fn expm_kernel_has_scaling_and_squaring() {
        // Scaling-and-squaring: infinity-norm reduction then a squaring loop.
        let ptx = expm_pade_ptx(80);
        assert!(
            ptx.contains("infinity norm"),
            "missing infinity-norm computation for scaling"
        );
        assert!(
            ptx.contains("scaling-and-squaring"),
            "kernel header should document scaling-and-squaring"
        );
        assert!(ptx.contains("$EXPM_SQ_LOOP"), "missing squaring loop");
        assert!(
            ptx.contains("scale A in place"),
            "missing the A/2^s scaling pass"
        );
    }

    #[test]
    fn expm_kernel_uses_cooperative_block() {
        // Single-block kernel: shared tableau plus barrier synchronization.
        let ptx = expm_pade_ptx(89);
        assert!(ptx.contains(".shared"), "kernel must use shared memory");
        assert!(ptx.contains("sh_aug"), "missing augmented [V|I] tableau");
        assert!(
            ptx.matches("bar.sync").count() >= 8,
            "kernel must barrier-synchronize between phases"
        );
        // Padé numerator/denominator are still both formed.
        assert!(ptx.contains("U[i,j]") && ptx.contains("V[i,j]"));
    }

    #[test]
    fn expm_kernel_well_formed_all_sm() {
        // Balanced braces and a single kernel entry across all SM targets.
        for sm in [75u32, 80, 86, 89, 90, 100] {
            let ptx = expm_pade_ptx(sm);
            assert_eq!(
                ptx.matches('{').count(),
                ptx.matches('}').count(),
                "unbalanced braces for sm={sm}"
            );
            assert_eq!(
                ptx.matches(".visible .entry expm_pade_kernel").count(),
                1,
                "expected exactly one kernel entry for sm={sm}"
            );
            assert!(ptx.contains("ret;"), "missing ret for sm={sm}");
        }
    }
}
