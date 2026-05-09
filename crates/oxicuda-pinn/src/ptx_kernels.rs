//! PTX GPU kernel sources for Physics-Informed Neural Network operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`pinn_residual_ptx`]    | MSE residual accumulation: `Σ r_i²` |
//! | [`spectral_conv_ptx`]    | Complex spectral convolution for FNO |
//! | [`dual_op_ptx`]          | Dual-number elementwise multiply (forward-mode AD) |
//! | [`adjoint_ode_ptx`]      | Euler adjoint step: `a += h * da/dt` |
//! | [`branch_trunk_dot_ptx`] | DeepONet branch·trunk inner product |
//! | [`siren_forward_ptx`]    | SIREN layer: `sin(ω₀·(Wx+b))` |
//! | [`lhs_sample_ptx`]       | Latin hypercube sample generation |

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

/// Format an f32 as a PTX hex literal.
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: pinn_residual ─────────────────────────────────────────────────

/// Accumulate squared PDE residuals: `out_sum += Σ residuals[i]²`.
///
/// Grid-stride; uses `mul.f32` and `atom.global.add.f32`.
#[must_use]
pub fn pinn_residual_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// pinn_residual_kernel: out_sum += sum_i( residuals[i]^2 )
// Grid-stride kernel; each thread accumulates its portion into out_sum atomically.
.visible .entry pinn_residual_kernel(
    .param .u64 residuals,
    .param .u64 out_sum,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [residuals];
    ld.param.u64  %rd1, [out_sum];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$PR_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $PR_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;       // &residuals[i]

    ld.global.f32 %f0, [%rd3];            // r_i
    mul.f32       %f1, %f0, %f0;          // r_i^2
    fma.rn.f32    %f2, %f0, %f0, {ZERO}; // fma variant
    atom.global.add.f32 %f3, [%rd1], %f1;

    add.u32       %r7, %r7, %r6;
    bra           $PR_LOOP;

$PR_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.f32       %f6, {ZERO};
    mov.f32       %f7, {ZERO};
    mov.u64       %rd4, 0;
    mov.u64       %rd5, 0;
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 2: spectral_conv ─────────────────────────────────────────────────

/// Spectral convolution for FNO: complex multiply of Fourier-mode arrays.
///
/// `out_real = a_real*w_real - a_imag*w_imag`
/// `out_imag = a_real*w_imag + a_imag*w_real`
/// Uses `fma.rn.f32` for precision.
#[must_use]
pub fn spectral_conv_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let neg_one = f32_hex(-1.0_f32);
    format!(
        r#"{hdr}// spectral_conv_kernel: complex multiply of Fourier modes for FNO.
// out_real[i] = a_real[i]*w_real[i] - a_imag[i]*w_imag[i]
// out_imag[i] = a_real[i]*w_imag[i] + a_imag[i]*w_real[i]
.visible .entry spectral_conv_kernel(
    .param .u64 a_real,
    .param .u64 a_imag,
    .param .u64 w_real,
    .param .u64 w_imag,
    .param .u64 out_real,
    .param .u64 out_imag,
    .param .u32 n
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<10>;
    .reg .f32  %f<14>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [a_real];
    ld.param.u64  %rd1, [a_imag];
    ld.param.u64  %rd2, [w_real];
    ld.param.u64  %rd3, [w_imag];
    ld.param.u64  %rd4, [out_real];
    ld.param.u64  %rd5, [out_imag];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$SC_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $SC_DONE;

    mul.wide.u32  %rd6, %r7, 4;
    add.u64       %rd7,  %rd0, %rd6;   // &a_real[i]
    add.u64       %rd8,  %rd1, %rd6;   // &a_imag[i]
    add.u64       %rd9,  %rd2, %rd6;   // &w_real[i]
    add.u64       %rd10, %rd3, %rd6;   // &w_imag[i]
    add.u64       %rd11, %rd4, %rd6;   // &out_real[i]
    add.u64       %rd12, %rd5, %rd6;   // &out_imag[i]

    ld.global.f32 %f0, [%rd7];   // a_real
    ld.global.f32 %f1, [%rd8];   // a_imag
    ld.global.f32 %f2, [%rd9];   // w_real
    ld.global.f32 %f3, [%rd10];  // w_imag

    // out_real = a_real*w_real - a_imag*w_imag  (fma: a_real*w_real + (-1)*a_imag*w_imag)
    mul.f32       %f4, %f0, %f2;         // a_real*w_real
    mov.f32       %f12, {NEG_ONE};
    fma.rn.f32    %f5, %f12, %f1, %f4;  // -= a_imag*w_imag via fma
    // more precisely: out_real = fma(-a_imag, w_imag, a_real*w_real)
    fma.rn.f32    %f6, %f1, %f3, {ZERO}; // a_imag*w_imag (temp)
    sub.f32       %f7, %f4, %f6;         // a_real*w_real - a_imag*w_imag

    // out_imag = a_real*w_imag + a_imag*w_real
    fma.rn.f32    %f8, %f0, %f3, {ZERO}; // a_real*w_imag
    fma.rn.f32    %f9, %f1, %f2, %f8;    // += a_imag*w_real

    st.global.f32 [%rd11], %f7;
    st.global.f32 [%rd12], %f9;

    add.u32       %r7, %r7, %r6;
    bra           $SC_LOOP;

$SC_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f10, {ZERO};
    mov.f32       %f11, {ZERO};
    mov.f32       %f13, {ZERO};
    mov.u64       %rd13, 0;
    ret;
}}
"#,
        ZERO = zero,
        NEG_ONE = neg_one,
    )
}

// ─── Kernel 3: dual_op ───────────────────────────────────────────────────────

/// Dual-number element-wise multiply (forward-mode AD product rule).
///
/// `out_val  = a_val * b_val`
/// `out_dval = a_val * b_dval + a_dval * b_val` via `fma.rn.f32`.
#[must_use]
pub fn dual_op_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// dual_mul_kernel: elementwise dual-number multiply (product rule).
// out_val[i]  = a_val[i] * b_val[i]
// out_dval[i] = a_val[i]*b_dval[i] + a_dval[i]*b_val[i]
.visible .entry dual_mul_kernel(
    .param .u64 a_val,
    .param .u64 a_dval,
    .param .u64 b_val,
    .param .u64 b_dval,
    .param .u64 out_val,
    .param .u64 out_dval,
    .param .u32 n
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [a_val];
    ld.param.u64  %rd1, [a_dval];
    ld.param.u64  %rd2, [b_val];
    ld.param.u64  %rd3, [b_dval];
    ld.param.u64  %rd4, [out_val];
    ld.param.u64  %rd5, [out_dval];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$DM_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $DM_DONE;

    mul.wide.u32  %rd6, %r7, 4;
    add.u64       %rd7,  %rd0, %rd6;
    add.u64       %rd8,  %rd1, %rd6;
    add.u64       %rd9,  %rd2, %rd6;
    add.u64       %rd10, %rd3, %rd6;
    add.u64       %rd11, %rd4, %rd6;
    add.u64       %rd12, %rd5, %rd6;

    ld.global.f32 %f0, [%rd7];   // av
    ld.global.f32 %f1, [%rd8];   // ad
    ld.global.f32 %f2, [%rd9];   // bv
    ld.global.f32 %f3, [%rd10];  // bd

    mul.f32       %f4, %f0, %f2;          // out_val = av*bv
    fma.rn.f32    %f5, %f0, %f3, {ZERO}; // av*bd
    fma.rn.f32    %f6, %f1, %f2, %f5;    // out_dval = av*bd + ad*bv

    st.global.f32 [%rd11], %f4;
    st.global.f32 [%rd12], %f6;

    add.u32       %r7, %r7, %r6;
    bra           $DM_LOOP;

$DM_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f7,  {ZERO};
    mov.f32       %f8,  {ZERO};
    mov.f32       %f9,  {ZERO};
    mov.f32       %f10, {ZERO};
    mov.f32       %f11, {ZERO};
    mov.u64       %rd13, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 4: adjoint_ode ────────────────────────────────────────────────────

/// Euler adjoint step: `a[i] += h * dadt[i]`.
///
/// Used for the continuous adjoint ODE backward pass.
/// Uses `fma.rn.f32`.
#[must_use]
pub fn adjoint_ode_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// adjoint_step_kernel: a[i] += h * dadt[i]
// Euler step for the adjoint ODE in the continuous adjoint method.
.visible .entry adjoint_step_kernel(
    .param .u64 a,
    .param .u64 dadt,
    .param .f32 h,
    .param .u32 n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [a];
    ld.param.u64  %rd1, [dadt];
    ld.param.f32  %f0,  [h];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$AS_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $AS_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;   // &a[i]
    add.u64       %rd4, %rd1, %rd2;   // &dadt[i]

    ld.global.f32 %f1, [%rd3];   // a_i
    ld.global.f32 %f2, [%rd4];   // dadt_i

    // a[i] += h * dadt[i]
    fma.rn.f32    %f3, %f0, %f2, %f1;
    st.global.f32 [%rd3], %f3;

    add.u32       %r7, %r7, %r6;
    bra           $AS_LOOP;

$AS_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.f32       %f6, {ZERO};
    mov.f32       %f7, {ZERO};
    mov.u64       %rd5, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 5: branch_trunk_dot ───────────────────────────────────────────────

/// DeepONet branch·trunk inner product with warp-shuffle reduction.
///
/// `out = Σ_k branch[k] * trunk[k]`
/// Warp-level sum via `shfl.sync.bfly.b32`, final accumulation via
/// `atom.global.add.f32`.
#[must_use]
pub fn branch_trunk_dot_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// branch_trunk_dot_kernel: inner product of branch[p] and trunk[p].
// Uses warp-shuffle reduction (shfl.sync.bfly.b32) then atomic add.
.visible .entry branch_trunk_dot_kernel(
    .param .u64 branch,
    .param .u64 trunk,
    .param .u64 out,
    .param .u32 p
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [branch];
    ld.param.u64  %rd1, [trunk];
    ld.param.u64  %rd2, [out];
    ld.param.u32  %r0,  [p];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;   // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;         // grid stride

    mov.f32       %f0, {ZERO};           // partial sum

    mov.u32       %r7, %r4;

$BT_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $BT_REDUCE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;

    ld.global.f32 %f1, [%rd4];   // branch[k]
    ld.global.f32 %f2, [%rd5];   // trunk[k]

    fma.rn.f32    %f0, %f1, %f2, %f0;   // partial += branch*trunk

    add.u32       %r7, %r7, %r6;
    bra           $BT_LOOP;

$BT_REDUCE:
    // Warp-shuffle butterfly reduction (32 lanes)
    shfl.sync.bfly.b32  %f3, %f0, 16, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f3;
    shfl.sync.bfly.b32  %f4, %f0,  8, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f4;
    shfl.sync.bfly.b32  %f5, %f0,  4, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f5;
    shfl.sync.bfly.b32  %f6, %f0,  2, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f6;
    shfl.sync.bfly.b32  %f7, %f0,  1, 31, 0xFFFFFFFF;
    add.f32       %f0, %f0, %f7;

    // Lane 0 writes result
    and.b32       %r8, %r3, 31;   // lane_id = tid & 31
    setp.ne.u32   %p0, %r8, 0;
    @%p0 bra $BT_DONE;

    atom.global.add.f32 %f8, [%rd2], %f0;

$BT_DONE:
    mov.u32       %r9,  0;
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.f32       %f9, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 6: siren_forward ──────────────────────────────────────────────────

/// SIREN layer forward pass: `out[i] = sin(omega_0 * (dot(w[i,:], x) + b[i]))`.
///
/// Uses `sin.approx.f32` for the sine activation.
#[must_use]
pub fn siren_forward_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// siren_forward_kernel: out[i] = sin(omega0 * (dot(w[i,:], x) + b[i]))
// w: [dout x din] row-major, x: [din], b: [dout], out: [dout]
.visible .entry siren_forward_kernel(
    .param .u64 w,
    .param .u64 x,
    .param .u64 b,
    .param .u64 out,
    .param .u32 din,
    .param .u32 dout,
    .param .f32 omega0
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<10>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [w];
    ld.param.u64  %rd1, [x];
    ld.param.u64  %rd2, [b];
    ld.param.u64  %rd3, [out];
    ld.param.u32  %r0,  [din];
    ld.param.u32  %r1,  [dout];
    ld.param.f32  %f8,  [omega0];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // output neuron index i

    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $SF_DONE;

    // Compute dot product: acc = dot(w[i,:], x)
    mul.lo.u32    %r6, %r5, %r0;         // row offset = i * din
    mov.u32       %r7, 0;                // j = 0
    mov.f32       %f0, {ZERO};           // acc = 0

$SF_DOT:
    setp.ge.u32   %p1, %r7, %r0;
    @%p1 bra $SF_BIAS;

    add.u32       %r8, %r6, %r7;         // flat index i*din + j
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd0, %rd4;      // &w[i,j]
    mul.wide.u32  %rd6, %r7, 4;
    add.u64       %rd7, %rd1, %rd6;      // &x[j]

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd7];
    fma.rn.f32    %f0, %f1, %f2, %f0;   // acc += w[i,j]*x[j]

    add.u32       %r7, %r7, 1;
    bra           $SF_DOT;

$SF_BIAS:
    mul.wide.u32  %rd8, %r5, 4;
    add.u64       %rd9, %rd2, %rd8;      // &b[i]
    ld.global.f32 %f3, [%rd9];           // b[i]
    add.f32       %f4, %f0, %f3;         // acc += b[i]
    mul.f32       %f5, %f8, %f4;         // omega0 * acc
    sin.approx.f32 %f6, %f5;            // sin(omega0 * acc)

    mul.wide.u32  %rd4, %r5, 4;
    add.u64       %rd5, %rd3, %rd4;      // &out[i]
    st.global.f32 [%rd5], %f6;

$SF_DONE:
    mov.u32       %r9, 0;
    mov.f32       %f7, {ZERO};
    mov.f32       %f9, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 7: lhs_sample ────────────────────────────────────────────────────

/// Latin Hypercube Sampling kernel.
///
/// For each thread (i, j): LCG step on seed, cell = perm[j*n+i],
/// `out[i*dim+j] = (cell + state/MAX) / n`.
#[must_use]
pub fn lhs_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let u32max_inv = f32_hex(1.0_f32 / (u32::MAX as f32 + 1.0));
    format!(
        r#"{hdr}// lhs_sample_kernel: Latin Hypercube Sampling.
// Each thread handles one (sample i, dimension j) pair.
// state = LCG(seed + tid); cell = perm[j*n + i]; out = (cell + state/MAX) / n
.visible .entry lhs_sample_kernel(
    .param .u64 perm,
    .param .u64 seed,
    .param .u32 n,
    .param .u32 dim,
    .param .u64 out
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [perm];
    ld.param.u64  %rd1, [seed];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [dim];
    ld.param.u64  %rd2, [out];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // global tid = i*dim + j

    // derive i and j from global tid
    rem.u32       %r6, %r5, %r1;         // j = tid % dim
    div.u32       %r7, %r5, %r1;         // i = tid / dim

    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $LHS_DONE;

    // LCG: state = a*(seed + tid) + c
    cvt.u64.u32   %rd3, %r5;
    add.u64       %rd4, %rd1, %rd3;
    mov.u64       %rd5, 6364136223846793005;
    mul.lo.u64    %rd4, %rd4, %rd5;
    mov.u64       %rd6, 1442695040888963407;
    add.u64       %rd4, %rd4, %rd6;
    shr.u64       %rd7, %rd4, 33;
    cvt.u32.u64   %r8, %rd7;             // random u32

    // cell = perm[j*n + i]
    mul.lo.u32    %r9, %r6, %r0;         // j*n
    add.u32       %r9, %r9, %r7;         // j*n + i
    mul.wide.u32  %rd3, %r9, 4;
    add.u64       %rd5, %rd0, %rd3;
    ld.global.u32 %r10, [%rd5];          // cell = perm[j*n+i]

    // out = (cell + state/MAX_U32) / n
    cvt.rn.f32.u32  %f0, %r10;           // cell as f32
    cvt.rn.f32.u32  %f1, %r8;            // rand as f32
    mov.f32         %f2, {U32MAX_INV};
    mul.f32         %f3, %f1, %f2;        // state/MAX
    add.f32         %f4, %f0, %f3;        // cell + frac
    cvt.rn.f32.u32  %f5, %r0;            // n as f32
    div.rn.f32      %f6, %f4, %f5;        // / n

    mul.wide.u32  %rd8, %r5, 4;
    add.u64       %rd9, %rd2, %rd8;
    st.global.f32 [%rd9], %f6;

$LHS_DONE:
    mov.u32       %r11, 0;
    mov.f32       %f7, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        U32MAX_INV = u32max_inv,
    )
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_kernel_well_formed(prog: &str, sm: u32, kernel_name: &str) {
        assert!(prog.contains(&format!("sm_{sm}")), "missing sm_{sm} target");
        assert!(prog.contains(".version"), "missing .version");
        assert!(prog.contains(".visible .entry"), "missing .visible .entry");
        assert!(
            prog.contains(kernel_name),
            "missing kernel name {kernel_name}"
        );
    }

    #[test]
    fn pinn_residual_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&pinn_residual_ptx(sm), sm, "pinn_residual_kernel");
        }
    }

    #[test]
    fn spectral_conv_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&spectral_conv_ptx(sm), sm, "spectral_conv_kernel");
        }
    }

    #[test]
    fn dual_op_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&dual_op_ptx(sm), sm, "dual_mul_kernel");
        }
    }

    #[test]
    fn adjoint_ode_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&adjoint_ode_ptx(sm), sm, "adjoint_step_kernel");
        }
    }

    #[test]
    fn branch_trunk_dot_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&branch_trunk_dot_ptx(sm), sm, "branch_trunk_dot_kernel");
        }
    }

    #[test]
    fn siren_forward_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&siren_forward_ptx(sm), sm, "siren_forward_kernel");
        }
    }

    #[test]
    fn lhs_sample_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&lhs_sample_ptx(sm), sm, "lhs_sample_kernel");
        }
    }

    #[test]
    fn ptx_header_version_strings() {
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
    fn pinn_residual_uses_fma_and_atomic() {
        let p = pinn_residual_ptx(80);
        assert!(p.contains("fma.rn.f32"));
        assert!(p.contains("atom.global.add.f32"));
    }

    #[test]
    fn spectral_conv_uses_fma() {
        let p = spectral_conv_ptx(90);
        assert!(p.contains("fma.rn.f32"));
    }

    #[test]
    fn dual_op_uses_fma() {
        let p = dual_op_ptx(80);
        assert!(p.contains("fma.rn.f32"));
    }

    #[test]
    fn adjoint_ode_uses_fma() {
        let p = adjoint_ode_ptx(100);
        assert!(p.contains("fma.rn.f32"));
    }

    #[test]
    fn branch_trunk_dot_uses_shfl() {
        let p = branch_trunk_dot_ptx(80);
        assert!(p.contains("shfl.sync.bfly.b32"));
        assert!(p.contains("atom.global.add.f32"));
    }

    #[test]
    fn siren_uses_sin_approx() {
        let p = siren_forward_ptx(86);
        assert!(p.contains("sin.approx.f32"));
    }

    #[test]
    fn lhs_sample_uses_lcg() {
        let p = lhs_sample_ptx(80);
        assert!(p.contains("6364136223846793005"));
        assert!(p.contains("1442695040888963407"));
    }

    #[test]
    fn sm_80_has_correct_version() {
        let p = pinn_residual_ptx(80);
        assert!(p.contains(".version 8.0"));
        assert!(p.contains("sm_80"));
    }

    #[test]
    fn sm_90_has_correct_version() {
        let p = spectral_conv_ptx(90);
        assert!(p.contains(".version 8.4"));
        assert!(p.contains("sm_90"));
    }
}
