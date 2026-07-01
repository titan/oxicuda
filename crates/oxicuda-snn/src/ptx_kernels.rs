//! GPU PTX kernels for Spiking Neural Networks.
//!
//! Each kernel is emitted as a self-contained PTX module string parameterised on
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

/// Leaky Integrate-and-Fire step: `v ← β·v + I; s = (v ≥ v_th); v ← (1-s)·v + s·v_rest`.
///
/// Kernel signature: `lif_step_kernel(v, current, spikes, n, beta, v_th, v_rest, reset_mode)`.
/// `reset_mode = 0` → hard reset to `v_rest`; `reset_mode = 1` → soft (subtractive) reset.
#[must_use]
pub fn lif_step_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// lif_step_kernel: LIF membrane update + spike + reset
.visible .entry lif_step_kernel(
    .param .u64 p_v,
    .param .u64 p_i,
    .param .u64 p_s,
    .param .u32 p_n,
    .param .f32 p_beta,
    .param .f32 p_vth,
    .param .f32 p_vrest,
    .param .u32 p_reset_mode
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<8>;
    .reg .f32  %f<10>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_v];
    ld.param.u64  %rd1, [p_i];
    ld.param.u64  %rd2, [p_s];
    ld.param.u32  %r0,  [p_n];
    ld.param.f32  %f0,  [p_beta];
    ld.param.f32  %f1,  [p_vth];
    ld.param.f32  %f2,  [p_vrest];
    ld.param.u32  %r1,  [p_reset_mode];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $LIF_DONE;

    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f3, [%rd4];        // v
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f4, [%rd5];        // I
    fma.rn.f32    %f5, %f0, %f3, %f4; // v_new = β·v + I

    setp.ge.f32   %p1, %f5, %f1;
    selp.f32      %f6, {ONE}, {ZERO}, %p1;  // spike

    setp.eq.u32   %p2, %r1, 0;
    // hard reset: v = (1-s)·v_new + s·v_rest = v_new + s·(v_rest - v_new)
    sub.f32       %f7, %f2, %f5;
    fma.rn.f32    %f8, %f6, %f7, %f5;
    // soft reset: v = v_new - s·v_th
    fma.rn.f32    %f9, %f6, %f1, {ZERO};
    sub.f32       %f9, %f5, %f9;
    selp.f32      %f7, %f8, %f9, %p2;

    st.global.f32 [%rd4], %f7;
    add.u64       %rd6, %rd2, %rd3;
    st.global.f32 [%rd6], %f6;

$LIF_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one
    )
}

/// Surrogate gradient: branches by `mode∈{0:sigmoid, 1:atan, 2:triangle, 3:super_spike, 4:fast_sigmoid}`.
#[must_use]
pub fn surrogate_grad_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let pi = f32_hex(std::f32::consts::PI);
    format!(
        r#"{hdr}// surrogate_grad_kernel: per-element surrogate derivative
.visible .entry surrogate_grad_kernel(
    .param .u64 p_v,
    .param .u64 p_g,
    .param .u32 p_n,
    .param .f32 p_vth,
    .param .f32 p_alpha,
    .param .u32 p_mode
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<8>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_v];
    ld.param.u64  %rd1, [p_g];
    ld.param.u32  %r0,  [p_n];
    ld.param.f32  %f0,  [p_vth];
    ld.param.f32  %f1,  [p_alpha];
    ld.param.u32  %r1,  [p_mode];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $SG_DONE;

    mul.wide.u32  %rd2, %r5, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f2, [%rd3];
    sub.f32       %f3, %f2, %f0;     // (v - v_th)
    mul.f32       %f4, %f1, %f3;     // α·(v-v_th)

    setp.eq.u32   %p1, %r1, 0;
    @%p1 bra $SG_SIGMOID;
    setp.eq.u32   %p1, %r1, 1;
    @%p1 bra $SG_ATAN;
    setp.eq.u32   %p1, %r1, 2;
    @%p1 bra $SG_TRI;
    setp.eq.u32   %p1, %r1, 3;
    @%p1 bra $SG_SUPER;

    // fast sigmoid: α / (1 + |α(v-v_th)|)^2
    abs.f32       %f5, %f4;
    add.f32       %f5, %f5, {ONE};
    mul.f32       %f5, %f5, %f5;
    div.rn.f32    %f6, %f1, %f5;
    bra $SG_STORE;

$SG_SIGMOID:
    // σ = 1/(1+exp(-α(v-v_th))); g = α·σ·(1-σ)
    neg.f32       %f5, %f4;
    mul.f32       %f5, %f5, 0F3FB8AA3B;  // *log2(e)
    ex2.approx.f32 %f5, %f5;
    add.f32       %f7, %f5, {ONE};
    div.rn.f32    %f8, {ONE}, %f7;       // σ
    sub.f32       %f9, {ONE}, %f8;
    mul.f32       %f10, %f8, %f9;
    mul.f32       %f6, %f1, %f10;
    bra $SG_STORE;

$SG_ATAN:
    // g = α / (π·(1 + (α(v-v_th))^2))
    mul.f32       %f5, %f4, %f4;        // (α(v-v_th))^2
    add.f32       %f5, %f5, {ONE};      // 1 + (α(v-v_th))^2
    mul.f32       %f5, %f5, {PI};       // π·(1 + (α(v-v_th))^2)
    div.rn.f32    %f6, %f1, %f5;        // α / (π·(1 + (α(v-v_th))^2))
    bra $SG_STORE;

$SG_TRI:
    // g = max(0, 1 - |v-v_th|/α)
    abs.f32       %f5, %f3;
    div.rn.f32    %f5, %f5, %f1;
    sub.f32       %f6, {ONE}, %f5;
    setp.lt.f32   %p1, %f6, {ZERO};
    selp.f32      %f6, {ZERO}, %f6, %p1;
    bra $SG_STORE;

$SG_SUPER:
    // g = α / (1 + |v-v_th|·α)^2
    abs.f32       %f5, %f3;
    mul.f32       %f5, %f5, %f1;
    add.f32       %f5, %f5, {ONE};
    mul.f32       %f5, %f5, %f5;
    div.rn.f32    %f6, %f1, %f5;

$SG_STORE:
    add.u64       %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f6;

$SG_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        PI = pi
    )
}

/// STDP weight delta from pre/post traces.
/// `Δw[i,j] += A_+·x_pre[i]·post_spike[j] − A_−·y_post[j]·pre_spike[i]`.
#[must_use]
pub fn stdp_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// stdp_update_kernel: pair-based weight delta
.visible .entry stdp_update_kernel(
    .param .u64 p_w,
    .param .u64 p_x_pre,
    .param .u64 p_y_post,
    .param .u64 p_pre_spike,
    .param .u64 p_post_spike,
    .param .u32 p_n_pre,
    .param .u32 p_n_post,
    .param .f32 p_a_plus,
    .param .f32 p_a_minus
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_w];
    ld.param.u64  %rd1, [p_x_pre];
    ld.param.u64  %rd2, [p_y_post];
    ld.param.u64  %rd3, [p_pre_spike];
    ld.param.u64  %rd4, [p_post_spike];
    ld.param.u32  %r0,  [p_n_pre];
    ld.param.u32  %r1,  [p_n_post];
    ld.param.f32  %f0,  [p_a_plus];
    ld.param.f32  %f1,  [p_a_minus];

    mov.u32       %r2, %ctaid.x;     // i
    mov.u32       %r3, %ctaid.y;     // j
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $STDP_DONE;
    setp.ge.u32   %p0, %r3, %r1;
    @%p0 bra $STDP_DONE;

    // x_pre[i]
    mul.wide.u32  %rd5, %r2, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];
    // y_post[j]
    mul.wide.u32  %rd5, %r3, 4;
    add.u64       %rd6, %rd2, %rd5;
    ld.global.f32 %f3, [%rd6];
    // pre_spike[i]
    mul.wide.u32  %rd5, %r2, 4;
    add.u64       %rd6, %rd3, %rd5;
    ld.global.f32 %f4, [%rd6];
    // post_spike[j]
    mul.wide.u32  %rd5, %r3, 4;
    add.u64       %rd6, %rd4, %rd5;
    ld.global.f32 %f5, [%rd6];

    // Δw = A_+ · x_pre · post_spike − A_− · y_post · pre_spike
    mul.f32       %f6, %f2, %f5;
    mul.f32       %f6, %f6, %f0;
    mul.f32       %f7, %f3, %f4;
    fma.rn.f32    %f7, %f7, %f1, {ZERO};
    sub.f32       %f6, %f6, %f7;

    // w[i,j] += Δw
    mul.lo.u32    %r4, %r2, %r1;
    add.u32       %r4, %r4, %r3;
    mul.wide.u32  %rd5, %r4, 4;
    add.u64       %rd7, %rd0, %rd5;
    ld.global.f32 %f4, [%rd7];
    add.f32       %f4, %f4, %f6;
    st.global.f32 [%rd7], %f4;

$STDP_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

/// 2D spiking convolution forward: each output channel sums a `kh×kw` window of
/// `in_c` channels then applies a LIF nonlinearity over the running membrane.
#[must_use]
pub fn spike_conv_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// spike_conv_kernel: 2D spiking convolution forward
.visible .entry spike_conv_kernel(
    .param .u64 p_in,
    .param .u64 p_w,
    .param .u64 p_v,
    .param .u64 p_out,
    .param .u32 p_oc,
    .param .u32 p_oh,
    .param .u32 p_ow,
    .param .u32 p_ic,
    .param .u32 p_kh,
    .param .u32 p_kw,
    .param .f32 p_vth
)
{{
    .reg .u64  %rd<14>;
    .reg .u32  %r<24>;
    .reg .f32  %f<10>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_in];
    ld.param.u64  %rd1, [p_w];
    ld.param.u64  %rd2, [p_v];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [p_oc];
    ld.param.u32  %r1,  [p_oh];
    ld.param.u32  %r2,  [p_ow];
    ld.param.u32  %r3,  [p_ic];
    ld.param.u32  %r4,  [p_kh];
    ld.param.u32  %r5,  [p_kw];
    ld.param.f32  %f9,  [p_vth];

    mov.u32       %r6, %ctaid.x;     // (oc index)
    mov.u32       %r7, %ctaid.y;     // (oh index)
    mov.u32       %r8, %ctaid.z;     // (ow index)
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $SC_DONE;

    mov.f32       %f0, {ZERO};       // accumulator
    mov.u32       %r9, 0;            // ic loop
$SC_IC:
    setp.ge.u32   %p0, %r9, %r3;
    @%p0 bra $SC_IC_DONE;
    mov.u32       %r10, 0;           // kh
$SC_KH:
    setp.ge.u32   %p0, %r10, %r4;
    @%p0 bra $SC_KH_DONE;
    mov.u32       %r11, 0;           // kw
$SC_KW:
    setp.ge.u32   %p0, %r11, %r5;
    @%p0 bra $SC_KW_DONE;

    // input pixel index = ((ic*ih + (oh+kh)) * iw) + (ow+kw); ih = oh+kh+1, iw=ow+kw+1 (no padding)
    add.u32       %r12, %r7, %r10;   // ih_pos
    add.u32       %r13, %r8, %r11;   // iw_pos
    add.u32       %r14, %r1, %r4;    // ih = oh+kh
    add.u32       %r15, %r2, %r5;    // iw = ow+kw
    mul.lo.u32    %r16, %r9, %r14;
    add.u32       %r16, %r16, %r12;
    mul.lo.u32    %r16, %r16, %r15;
    add.u32       %r16, %r16, %r13;
    mul.wide.u32  %rd4, %r16, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f1, [%rd5];

    // weight[oc, ic, kh, kw]
    mul.lo.u32    %r17, %r6, %r3;
    add.u32       %r17, %r17, %r9;
    mul.lo.u32    %r17, %r17, %r4;
    add.u32       %r17, %r17, %r10;
    mul.lo.u32    %r17, %r17, %r5;
    add.u32       %r17, %r17, %r11;
    mul.wide.u32  %rd4, %r17, 4;
    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];

    fma.rn.f32    %f0, %f1, %f2, %f0;
    add.u32       %r11, %r11, 1;
    bra $SC_KW;
$SC_KW_DONE:
    add.u32       %r10, %r10, 1;
    bra $SC_KH;
$SC_KH_DONE:
    add.u32       %r9, %r9, 1;
    bra $SC_IC;
$SC_IC_DONE:

    // out[oc, oh, ow]
    mul.lo.u32    %r18, %r6, %r1;
    add.u32       %r18, %r18, %r7;
    mul.lo.u32    %r18, %r18, %r2;
    add.u32       %r18, %r18, %r8;
    mul.wide.u32  %rd6, %r18, 4;
    add.u64       %rd7, %rd2, %rd6;
    ld.global.f32 %f3, [%rd7];
    add.f32       %f3, %f3, %f0;     // membrane accumulate
    setp.ge.f32   %p1, %f3, %f9;
    selp.f32      %f4, {ONE}, {ZERO}, %p1;
    sub.f32       %f5, %f3, %f9;
    selp.f32      %f3, %f5, %f3, %p1;
    st.global.f32 [%rd7], %f3;

    add.u64       %rd8, %rd3, %rd6;
    st.global.f32 [%rd8], %f4;

$SC_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one
    )
}

/// Per-step rate encoding: `out[t,i] = (rng_state(t,i) < value[i]) ? 1 : 0`.
#[must_use]
pub fn rate_encode_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let inv2_24 = f32_hex(1.0_f32 / 16_777_216.0_f32);
    format!(
        r#"{hdr}// rate_encode_kernel: Bernoulli rate encoding via inline LCG
.visible .entry rate_encode_kernel(
    .param .u64 p_value,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_t_steps,
    .param .u64 p_seed
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<10>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_value];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [p_n];
    ld.param.u32  %r1,  [p_t_steps];
    ld.param.u64  %rd2, [p_seed];

    mov.u32       %r2, %ctaid.x;     // t
    mov.u32       %r3, %ctaid.y;     // i
    setp.ge.u32   %p0, %r2, %r1;
    @%p0 bra $RE_DONE;
    setp.ge.u32   %p0, %r3, %r0;
    @%p0 bra $RE_DONE;

    // state = seed XOR (t * 6364136223846793005 + i * 1442695040888963407)
    cvt.u64.u32   %rd3, %r2;
    mul.lo.u64    %rd3, %rd3, 6364136223846793005;
    cvt.u64.u32   %rd4, %r3;
    mul.lo.u64    %rd4, %rd4, 1442695040888963407;
    add.u64       %rd5, %rd3, %rd4;
    xor.b64       %rd5, %rd5, %rd2;
    // advance once: state = state*M + A
    mul.lo.u64    %rd5, %rd5, 6364136223846793005;
    add.u64       %rd5, %rd5, 1442695040888963407;

    shr.u64       %rd6, %rd5, 33;
    xor.b64       %rd6, %rd6, %rd5;
    cvt.u32.u64   %r4, %rd6;
    shr.u32       %r4, %r4, 8;
    cvt.rn.f32.u32 %f0, %r4;
    mul.f32       %f0, %f0, {INV};   // u in [0,1)

    mul.wide.u32  %rd7, %r3, 4;
    add.u64       %rd8, %rd0, %rd7;
    ld.global.f32 %f1, [%rd8];
    setp.lt.f32   %p1, %f0, %f1;
    selp.f32      %f2, {ONE}, {ZERO}, %p1;

    mul.lo.u32    %r5, %r2, %r0;
    add.u32       %r5, %r5, %r3;
    mul.wide.u32  %rd9, %r5, 4;
    add.u64       %rd9, %rd1, %rd9;
    st.global.f32 [%rd9], %f2;

$RE_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        INV = inv2_24
    )
}

/// Poisson sampling: `out[i] = (lcg < rate[i]·dt) ? 1 : 0` per timestep.
#[must_use]
pub fn poisson_sample_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let inv2_24 = f32_hex(1.0_f32 / 16_777_216.0_f32);
    format!(
        r#"{hdr}// poisson_sample_kernel: per-step Bernoulli p=rate*dt
.visible .entry poisson_sample_kernel(
    .param .u64 p_rate,
    .param .u64 p_state,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .f32 p_dt
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<8>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_rate];
    ld.param.u64  %rd1, [p_state];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [p_n];
    ld.param.f32  %f0,  [p_dt];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $PS_DONE;

    mul.wide.u32  %rd3, %r4, 8;       // u64 state per neuron
    add.u64       %rd4, %rd1, %rd3;
    ld.global.u64 %rd5, [%rd4];

    mul.lo.u64    %rd5, %rd5, 6364136223846793005;
    add.u64       %rd5, %rd5, 1442695040888963407;
    st.global.u64 [%rd4], %rd5;

    shr.u64       %rd6, %rd5, 33;
    xor.b64       %rd6, %rd6, %rd5;
    cvt.u32.u64   %r5, %rd6;
    shr.u32       %r5, %r5, 8;
    cvt.rn.f32.u32 %f1, %r5;
    mul.f32       %f1, %f1, {INV};

    mul.wide.u32  %rd7, %r4, 4;
    add.u64       %rd8, %rd0, %rd7;
    ld.global.f32 %f2, [%rd8];
    mul.f32       %f2, %f2, %f0;
    setp.lt.f32   %p1, %f1, %f2;
    selp.f32      %f3, {ONE}, {ZERO}, %p1;

    add.u64       %rd9, %rd2, %rd7;
    st.global.f32 [%rd9], %f3;

$PS_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        INV = inv2_24
    )
}

/// BPTT gradient accumulator for SNN. `dL/dW += dL/dv_t · I_t^T` and propagates `dL/dv_{t-1}`.
#[must_use]
pub fn bptt_accum_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// bptt_accum_kernel: outer-product accumulation dW += dv·I^T
.visible .entry bptt_accum_kernel(
    .param .u64 p_dv,
    .param .u64 p_input,
    .param .u64 p_dw,
    .param .u32 p_out,
    .param .u32 p_in
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_dv];
    ld.param.u64  %rd1, [p_input];
    ld.param.u64  %rd2, [p_dw];
    ld.param.u32  %r0,  [p_out];
    ld.param.u32  %r1,  [p_in];

    mov.u32       %r2, %ctaid.x;     // i (out)
    mov.u32       %r3, %ctaid.y;     // j (in)
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $BP_DONE;
    setp.ge.u32   %p0, %r3, %r1;
    @%p0 bra $BP_DONE;

    mul.wide.u32  %rd3, %r2, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

    mul.wide.u32  %rd3, %r3, 4;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f1, [%rd5];

    mul.lo.u32    %r4, %r2, %r1;
    add.u32       %r4, %r4, %r3;
    mul.wide.u32  %rd6, %r4, 4;
    add.u64       %rd7, %rd2, %rd6;
    ld.global.f32 %f2, [%rd7];
    fma.rn.f32    %f2, %f0, %f1, %f2;
    st.global.f32 [%rd7], %f2;

$BP_DONE:
    ret;
}}
"#,
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
                lif_step_ptx(sm),
                surrogate_grad_ptx(sm),
                stdp_update_ptx(sm),
                spike_conv_ptx(sm),
                rate_encode_ptx(sm),
                poisson_sample_ptx(sm),
                bptt_accum_ptx(sm),
            ] {
                assert!(kernel.contains(".visible .entry"));
                assert!(kernel.contains(".address_size 64"));
            }
        }
    }
}
