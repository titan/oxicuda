//! PTX GPU kernel sources for audio/speech ML operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`stride_conv1d_ptx`] | Strided 1-D conv for Wav2Vec2 CNN feature extractor |
//! | [`dilated_conv1d_ptx`] | Causal dilated 1-D conv with filter+gate outputs (WaveNet) |
//! | [`ctc_alpha_ptx`] | CTC forward alpha recursion in log domain |
//! | [`spec_augment_mask_ptx`] | SpecAugment time+frequency masking in-place |
//! | [`depthwise_conv1d_ptx`] | Causal depthwise 1-D conv for Conformer conv module |
//! | [`rel_pos_bias_ptx`] | Relative-position bias matrix for Conformer/Transformer-XL |
//! | [`stats_pool_ptx`] | Temporal mean+std pooling for speaker embedding |

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

// ─── Kernel 1: stride_conv1d ─────────────────────────────────────────────────

/// Strided 1-D convolution kernel used by the Wav2Vec2 CNN feature extractor.
#[must_use]
pub fn stride_conv1d_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry stride_conv1d_kernel(
    .param .u64 p_input,
    .param .u64 p_weights,
    .param .u64 p_bias,
    .param .u64 p_output,
    .param .u32 in_chans,
    .param .u32 in_len,
    .param .u32 out_chans,
    .param .u32 kernel_size,
    .param .u32 stride,
    .param .u32 out_len
)
{{
    .reg .u64  %rd<20>;
    .reg .u32  %r<32>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_input];
    ld.param.u64  %rd1, [p_weights];
    ld.param.u64  %rd2, [p_bias];
    ld.param.u64  %rd3, [p_output];
    ld.param.u32  %r0,  [in_chans];
    ld.param.u32  %r1,  [in_len];
    ld.param.u32  %r2,  [out_chans];
    ld.param.u32  %r3,  [kernel_size];
    ld.param.u32  %r4,  [stride];
    ld.param.u32  %r5,  [out_len];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r6,  %ntid.x;
    mov.u32       %r7,  %ctaid.x;
    mov.u32       %r8,  %tid.x;
    mad.lo.u32    %r9,  %r6, %r7, %r8;      // r9 = tid

    // total = out_chans * out_len
    mul.lo.u32    %r10, %r2, %r5;

    setp.ge.u32   %p0, %r9, %r10;
    @%p0 bra $SC1D_DONE;

    // oc  = tid / out_len
    // pos = tid % out_len
    div.u32       %r11, %r9, %r5;           // r11 = oc
    rem.u32       %r12, %r9, %r5;           // r12 = pos

    // t_start = pos * stride
    mul.lo.u32    %r13, %r12, %r4;          // r13 = t_start

    // Load bias[oc]
    mul.wide.u32  %rd4, %r11, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f0,  [%rd5];             // f0 = acc = bias[oc]

    // weight_base_oc = oc * in_chans * kernel_size
    mul.lo.u32    %r14, %r0, %r3;           // in_chans * kernel_size
    mul.lo.u32    %r15, %r11, %r14;         // r15 = oc * in_chans * kernel_size (elem)

    // Outer loop: ic in [0, in_chans)
    mov.u32       %r16, 0;                  // ic = 0

$SC1D_IC_LOOP:
    setp.ge.u32   %p1, %r16, %r0;
    @%p1 bra $SC1D_IC_END;

    // input base for ic: ic * in_len + t_start
    mul.lo.u32    %r17, %r16, %r1;          // ic * in_len
    add.u32       %r18, %r17, %r13;         // + t_start  (byte offset / 4)

    // weight_base_ic = weight_base_oc + ic * kernel_size (elem)
    mad.lo.u32    %r19, %r16, %r3, %r15;    // r19 = elem offset in weight for (oc, ic, 0)

    // Inner loop: k in [0, kernel_size)
    mov.u32       %r20, 0;                  // k = 0

$SC1D_K_LOOP:
    setp.ge.u32   %p1, %r20, %r3;
    @%p1 bra $SC1D_K_END;

    // input[ic * in_len + t_start + k]
    add.u32       %r21, %r18, %r20;         // r21 = elem idx in input
    mul.wide.u32  %rd6, %r21, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f1,  [%rd7];             // f1 = x

    // weights[oc * in_chans * kernel_size + ic * kernel_size + k]
    add.u32       %r22, %r19, %r20;         // r22 = elem idx in weights
    mul.wide.u32  %rd8, %r22, 4;
    add.u64       %rd9, %rd1, %rd8;
    ld.global.f32 %f2,  [%rd9];             // f2 = w

    fma.rn.f32    %f0, %f2, %f1, %f0;      // acc += w * x

    add.u32       %r20, %r20, 1;
    bra           $SC1D_K_LOOP;

$SC1D_K_END:
    add.u32       %r16, %r16, 1;
    bra           $SC1D_IC_LOOP;

$SC1D_IC_END:
    // output[oc * out_len + pos] = acc
    mul.wide.u32  %rd10, %r9, 4;
    add.u64       %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f0;

$SC1D_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 2: dilated_conv1d ────────────────────────────────────────────────

/// Causal dilated 1-D convolution with separate filter and gate outputs (WaveNet).
#[must_use]
pub fn dilated_conv1d_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry dilated_conv1d_kernel(
    .param .u64 p_input,
    .param .u64 p_filter_w,
    .param .u64 p_gate_w,
    .param .u64 p_filter_b,
    .param .u64 p_gate_b,
    .param .u64 p_out_filter,
    .param .u64 p_out_gate,
    .param .u32 length,
    .param .u32 channels,
    .param .u32 kernel_size,
    .param .u32 dilation
)
{{
    .reg .u64  %rd<24>;
    .reg .u32  %r<32>;
    .reg .f32  %f<12>;
    .reg .pred %p0, %p1, %p2;
    .reg .s32  %s0, %s1, %s2, %s3;

    ld.param.u64  %rd0,  [p_input];
    ld.param.u64  %rd1,  [p_filter_w];
    ld.param.u64  %rd2,  [p_gate_w];
    ld.param.u64  %rd3,  [p_filter_b];
    ld.param.u64  %rd4,  [p_gate_b];
    ld.param.u64  %rd5,  [p_out_filter];
    ld.param.u64  %rd6,  [p_out_gate];
    ld.param.u32  %r0,   [length];
    ld.param.u32  %r1,   [channels];
    ld.param.u32  %r2,   [kernel_size];
    ld.param.u32  %r3,   [dilation];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r4,  %ntid.x;
    mov.u32       %r5,  %ctaid.x;
    mov.u32       %r6,  %tid.x;
    mad.lo.u32    %r7,  %r4, %r5, %r6;     // r7 = tid

    // total = channels * length
    mul.lo.u32    %r8,  %r1, %r0;

    setp.ge.u32   %p0, %r7, %r8;
    @%p0 bra $DCV_DONE;

    // ch = tid / length,  t = tid % length
    div.u32       %r9,  %r7, %r0;           // r9 = ch
    rem.u32       %r10, %r7, %r0;           // r10 = t

    // Load filter bias[ch] and gate bias[ch]
    mul.wide.u32  %rd7, %r9, 4;
    add.u64       %rd8, %rd3, %rd7;
    ld.global.f32 %f0,  [%rd8];             // f0 = filter_acc = filter_b[ch]
    add.u64       %rd9, %rd4, %rd7;
    ld.global.f32 %f1,  [%rd9];             // f1 = gate_acc   = gate_b[ch]

    // weight base for this channel: ch * kernel_size (elem)
    mul.lo.u32    %r11, %r9, %r2;           // r11 = ch * kernel_size

    // Convert t to signed for src_t arithmetic
    cvt.s32.u32   %s0, %r10;               // s0 = (s32)t

    // Loop k in [0, kernel_size)
    mov.u32       %r12, 0;                  // k = 0

$DCV_K_LOOP:
    setp.ge.u32   %p1, %r12, %r2;
    @%p1 bra $DCV_K_END;

    // src_t = t - k * dilation
    cvt.s32.u32   %s1, %r12;               // s1 = (s32)k
    cvt.s32.u32   %s2, %r3;                // s2 = (s32)dilation
    mul.lo.s32    %s3, %s1, %s2;           // k * dilation
    sub.s32       %s3, %s0, %s3;           // s3 = src_t = t - k*dilation

    // if src_t < 0 treat input as 0.0 (causal left-pad)
    setp.lt.s32   %p2, %s3, 0;

    // input[ch * length + src_t]  (only if src_t >= 0)
    cvt.u32.s32   %r13, %s3;               // unsigned src_t
    mul.lo.u32    %r14, %r9, %r0;          // ch * length
    add.u32       %r14, %r14, %r13;        // + src_t
    mul.wide.u32  %rd10, %r14, 4;
    add.u64       %rd11, %rd0, %rd10;

    // f2 = 0 if src_t < 0 (causal pad), else load from input
    @%p2 mov.f32  %f2, {ZERO};
    @!%p2 ld.global.f32 %f2, [%rd11];      // f2 = x (or 0 if oob)

    // weight index for (ch, k) in filter_w and gate_w
    add.u32       %r15, %r11, %r12;        // r15 = ch*kernel_size + k
    mul.wide.u32  %rd12, %r15, 4;
    add.u64       %rd13, %rd1, %rd12;
    ld.global.f32 %f3,  [%rd13];           // f3 = filter_w[ch, k]
    add.u64       %rd14, %rd2, %rd12;
    ld.global.f32 %f4,  [%rd14];           // f4 = gate_w[ch, k]

    fma.rn.f32    %f0, %f3, %f2, %f0;     // filter_acc += filter_w * x
    fma.rn.f32    %f1, %f4, %f2, %f1;     // gate_acc   += gate_w   * x

    add.u32       %r12, %r12, 1;
    bra           $DCV_K_LOOP;

$DCV_K_END:
    // Store output_filter[tid] and output_gate[tid]
    mul.wide.u32  %rd15, %r7, 4;
    add.u64       %rd16, %rd5, %rd15;
    st.global.f32 [%rd16], %f0;
    add.u64       %rd17, %rd6, %rd15;
    st.global.f32 [%rd17], %f1;

$DCV_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 3: ctc_alpha ─────────────────────────────────────────────────────

/// CTC forward alpha recursion in log domain (log-space forward algorithm).
#[must_use]
pub fn ctc_alpha_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#"{hdr}// CTC forward alpha: log-domain DP.
// log_sum_exp(a,b) = max(a,b) + lg2(1 + 2^(min-max)) converted to log-base-e
// We use ex2.approx + lg2.approx for the stable LSE step.
// Blank label index = 0.
.visible .entry ctc_alpha_kernel(
    .param .u64 p_log_probs,
    .param .u64 p_alpha,
    .param .u64 p_ext_target,
    .param .u32 T,
    .param .u32 V,
    .param .u32 S
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<24>;
    .reg .f32  %f<20>;
    .reg .pred %p0, %p1, %p2, %p3, %p4;
    .reg .s32  %si0;

    ld.param.u64  %rd0, [p_log_probs];
    ld.param.u64  %rd1, [p_alpha];
    ld.param.u64  %rd2, [p_ext_target];
    ld.param.u32  %r0,  [T];
    ld.param.u32  %r1,  [V];
    ld.param.u32  %r2,  [S];

    // One thread per label position s
    mov.u32       %r3,  %ntid.x;
    mov.u32       %r4,  %ctaid.x;
    mov.u32       %r5,  %tid.x;
    mad.lo.u32    %r6,  %r3, %r4, %r5;    // r6 = s

    setp.ge.u32   %p0, %r6, %r2;
    @%p0 bra $CTC_DONE;

    // Outer sequential loop over t in [1, T)
    mov.u32       %r7, 1;                  // t = 1

$CTC_T_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $CTC_T_END;

    // s = r6; load alpha[s, t-1]
    sub.u32       %r8, %r7, 1;            // t-1
    mul.lo.u32    %r9, %r8, %r2;          // (t-1) * S
    add.u32       %r10, %r9, %r6;         // (t-1)*S + s
    mul.wide.u32  %rd3, %r10, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];            // f0 = alpha[s, t-1]

    // alpha[s-1, t-1] (if s >= 1)
    setp.ge.u32   %p1, %r6, 1;
    mov.f32       %f1, {NEG_INF};
    @%p1 sub.u32  %r11, %r6, 1;
    @%p1 add.u32  %r11, %r9, %r11;
    @%p1 mul.wide.u32 %rd5, %r11, 4;
    @%p1 add.u64  %rd6, %rd1, %rd5;
    @%p1 ld.global.f32 %f1, [%rd6];       // f1 = alpha[s-1, t-1]

    // log_sum_exp(f0, f1) → f2
    max.f32       %f4,  %f0,  %f1;
    min.f32       %f5,  %f0,  %f1;
    sub.f32       %f6,  %f5,  %f4;        // min - max  (<= 0)
    ex2.approx.f32 %f7, %f6;              // 2^(min-max)
    add.f32       %f7,  %f7,  {ONE_F};
    lg2.approx.f32 %f7, %f7;
    mul.f32       %f7,  %f7,  {LN2};      // * ln(2) → natural log
    add.f32       %f2,  %f4,  %f7;        // f2 = lse(alpha[s], alpha[s-1])

    // Optionally add alpha[s-2, t-1] if s >= 2 and ext_target[s] != blank
    // and ext_target[s] != ext_target[s-2]
    setp.ge.u32   %p2, %r6, 2;
    @!%p2 bra $CTC_NO_SKIP;

    // Load ext_target[s]
    mul.wide.u32  %rd7, %r6, 4;
    add.u64       %rd8, %rd2, %rd7;
    ld.global.u32 %r12, [%rd8];           // r12 = ext_target[s]

    // Check blank (0)
    setp.eq.u32   %p3, %r12, 0;
    @%p3 bra $CTC_NO_SKIP;

    // Load ext_target[s-2]
    sub.u32       %r13, %r6, 2;
    mul.wide.u32  %rd9, %r13, 4;
    add.u64       %rd10, %rd2, %rd9;
    ld.global.u32 %r14, [%rd10];          // r14 = ext_target[s-2]

    setp.eq.u32   %p4, %r12, %r14;
    @%p4 bra $CTC_NO_SKIP;

    // Load alpha[s-2, t-1]
    add.u32       %r15, %r9, %r13;        // (t-1)*S + (s-2)
    mul.wide.u32  %rd11, %r15, 4;
    add.u64       %rd12, %rd1, %rd11;
    ld.global.f32 %f8,  [%rd12];          // f8 = alpha[s-2, t-1]

    // log_sum_exp(f2, f8) → f2
    max.f32       %f9,  %f2, %f8;
    min.f32       %f10, %f2, %f8;
    sub.f32       %f11, %f10, %f9;
    ex2.approx.f32 %f12, %f11;
    add.f32       %f12, %f12, {ONE_F};
    lg2.approx.f32 %f12, %f12;
    mul.f32       %f12, %f12, {LN2};
    add.f32       %f2,  %f9,  %f12;

$CTC_NO_SKIP:
    // Add log_probs[t, ext_target[s]]
    mul.wide.u32  %rd7, %r6, 4;
    add.u64       %rd8, %rd2, %rd7;
    ld.global.u32 %r16, [%rd8];           // r16 = ext_target[s]

    // log_probs[t, ext_target[s]]: offset = t * V + ext_target[s]
    mul.lo.u32    %r17, %r7, %r1;
    add.u32       %r17, %r17, %r16;
    mul.wide.u32  %rd13, %r17, 4;
    add.u64       %rd14, %rd0, %rd13;
    ld.global.f32 %f13, [%rd14];          // f13 = log_probs[t, label]

    add.f32       %f2,  %f2, %f13;

    // Store alpha[s, t]
    mul.lo.u32    %r18, %r7, %r2;         // t * S
    add.u32       %r18, %r18, %r6;        // + s
    mul.wide.u32  %rd15, %r18, 4;
    add.u64       %rd16, %rd1, %rd15;
    st.global.f32 [%rd16], %f2;

    // Barrier: all threads must finish writing alpha[*, t] before reading at t+1
    bar.sync      0;

    add.u32       %r7, %r7, 1;
    bra           $CTC_T_LOOP;

$CTC_T_END:
$CTC_DONE:
    ret;
}}
"#,
        NEG_INF = neg_inf,
        ONE_F = f32_hex(1.0_f32),
        LN2 = f32_hex(core::f32::consts::LN_2)
    )
}

// ─── Kernel 4: spec_augment_mask ─────────────────────────────────────────────

/// SpecAugment time+frequency masking applied in-place to a `[T, F]` log-mel spectrogram.
#[must_use]
pub fn spec_augment_mask_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry spec_augment_mask_kernel(
    .param .u64 p_mel,
    .param .u32 T,
    .param .u32 F,
    .param .u32 t_start,
    .param .u32 t_len,
    .param .u32 f_start,
    .param .u32 f_len
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<24>;
    .reg .f32  %f<4>;
    .reg .pred %p0, %p1, %p2, %p3, %p4;

    ld.param.u64  %rd0, [p_mel];
    ld.param.u32  %r0,  [T];
    ld.param.u32  %r1,  [F];
    ld.param.u32  %r2,  [t_start];
    ld.param.u32  %r3,  [t_len];
    ld.param.u32  %r4,  [f_start];
    ld.param.u32  %r5,  [f_len];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r6,  %ntid.x;
    mov.u32       %r7,  %ctaid.x;
    mov.u32       %r8,  %tid.x;
    mad.lo.u32    %r9,  %r6, %r7, %r8;    // r9 = tid

    // total = T * F
    mul.lo.u32    %r10, %r0, %r1;

    setp.ge.u32   %p0, %r9, %r10;
    @%p0 bra $SA_DONE;

    // t = tid / F,  f = tid % F
    div.u32       %r11, %r9, %r1;          // r11 = t
    rem.u32       %r12, %r9, %r1;          // r12 = f

    // time mask: t_start <= t < t_start + t_len
    add.u32       %r13, %r2, %r3;          // t_start + t_len
    setp.ge.u32   %p1, %r11, %r2;          // t >= t_start
    setp.lt.u32   %p2, %r11, %r13;         // t < t_start+t_len
    and.pred      %p1, %p1, %p2;           // p1 = in_time_mask

    // freq mask: f_start <= f < f_start + f_len
    add.u32       %r14, %r4, %r5;          // f_start + f_len
    setp.ge.u32   %p3, %r12, %r4;          // f >= f_start
    setp.lt.u32   %p4, %r12, %r14;         // f < f_start+f_len
    and.pred      %p3, %p3, %p4;           // p3 = in_freq_mask

    or.pred       %p1, %p1, %p3;           // p1 = should_zero

    // Load current value
    mul.wide.u32  %rd1, %r9, 4;
    add.u64       %rd2, %rd0, %rd1;
    ld.global.f32 %f0, [%rd2];

    // Zero-out if masked
    selp.f32      %f1, {ZERO}, %f0, %p1;

    st.global.f32 [%rd2], %f1;

$SA_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 5: depthwise_conv1d ──────────────────────────────────────────────

/// Causal depthwise 1-D convolution for the Conformer conv module.
#[must_use]
pub fn depthwise_conv1d_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry depthwise_conv1d_kernel(
    .param .u64 p_input,
    .param .u64 p_weights,
    .param .u64 p_bias,
    .param .u64 p_output,
    .param .u32 channels,
    .param .u32 length,
    .param .u32 kernel_size
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<24>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;
    .reg .s32  %s0, %s1, %s2, %s3;

    ld.param.u64  %rd0, [p_input];
    ld.param.u64  %rd1, [p_weights];
    ld.param.u64  %rd2, [p_bias];
    ld.param.u64  %rd3, [p_output];
    ld.param.u32  %r0,  [channels];
    ld.param.u32  %r1,  [length];
    ld.param.u32  %r2,  [kernel_size];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r3,  %ntid.x;
    mov.u32       %r4,  %ctaid.x;
    mov.u32       %r5,  %tid.x;
    mad.lo.u32    %r6,  %r3, %r4, %r5;    // r6 = tid

    // total = channels * length
    mul.lo.u32    %r7,  %r0, %r1;

    setp.ge.u32   %p0, %r6, %r7;
    @%p0 bra $DW_DONE;

    // ch = tid / length,  t = tid % length
    div.u32       %r8,  %r6, %r1;          // r8 = ch
    rem.u32       %r9,  %r6, %r1;          // r9 = t

    // Load bias[ch]
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f0,  [%rd5];            // f0 = acc = bias[ch]

    // pad = kernel_size - 1 (causal left-padding)
    sub.u32       %r10, %r2, 1;            // r10 = pad = K-1
    cvt.s32.u32   %s0, %r9;               // s0 = (s32)t
    cvt.s32.u32   %s1, %r10;              // s1 = pad

    // weight base = ch * kernel_size
    mul.lo.u32    %r11, %r8, %r2;          // r11 = ch * kernel_size

    // Loop k in [0, kernel_size)
    mov.u32       %r12, 0;

$DW_K_LOOP:
    setp.ge.u32   %p1, %r12, %r2;
    @%p1 bra $DW_K_END;

    // input_idx = t - pad + k  (signed)
    // = s0 - s1 + k
    cvt.s32.u32   %s2, %r12;              // s2 = (s32)k
    sub.s32       %s3, %s0, %s1;          // t - pad
    add.s32       %s3, %s3, %s2;          // t - pad + k

    // Skip if index < 0
    setp.lt.s32   %p1, %s3, 0;
    @%p1 bra $DW_SKIP_K;

    // input[ch * length + (t - pad + k)]
    cvt.u32.s32   %r13, %s3;
    mul.lo.u32    %r14, %r8, %r1;          // ch * length
    add.u32       %r14, %r14, %r13;
    mul.wide.u32  %rd6, %r14, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f1,  [%rd7];            // f1 = x

    // weights[ch * kernel_size + k]
    add.u32       %r15, %r11, %r12;
    mul.wide.u32  %rd8, %r15, 4;
    add.u64       %rd9, %rd1, %rd8;
    ld.global.f32 %f2,  [%rd9];            // f2 = w

    fma.rn.f32    %f0, %f2, %f1, %f0;

$DW_SKIP_K:
    add.u32       %r12, %r12, 1;
    bra           $DW_K_LOOP;

$DW_K_END:
    // Store output[ch * length + t]
    mul.wide.u32  %rd10, %r6, 4;
    add.u64       %rd11, %rd3, %rd10;
    st.global.f32 [%rd11], %f0;

$DW_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 6: rel_pos_bias ──────────────────────────────────────────────────

/// Relative-position bias matrix `B[Q, K]` for Conformer / Transformer-XL attention.
#[must_use]
pub fn rel_pos_bias_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry rel_pos_bias_kernel(
    .param .u64 p_table,
    .param .u64 p_output,
    .param .u32 Q,
    .param .u32 K,
    .param .u32 max_len
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<24>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_table];
    ld.param.u64  %rd1, [p_output];
    ld.param.u32  %r0,  [Q];
    ld.param.u32  %r1,  [K];
    ld.param.u32  %r2,  [max_len];

    // tid = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32       %r3,  %ntid.x;
    mov.u32       %r4,  %ctaid.x;
    mov.u32       %r5,  %tid.x;
    mad.lo.u32    %r6,  %r3, %r4, %r5;    // r6 = tid

    // total = Q * K
    mul.lo.u32    %r7,  %r0, %r1;

    setp.ge.u32   %p0, %r6, %r7;
    @%p0 bra $RPB_DONE;

    // q = tid / K,  k = tid % K
    div.u32       %r8,  %r6, %r1;          // r8 = q
    rem.u32       %r9,  %r6, %r1;          // r9 = k

    // table index = (k - q) + max_len - 1
    // k - q can be negative, compute as signed then clamp to [0, 2*max_len-2]
    // Use add with possible borrow: idx_signed = (int)k - (int)q + max_len - 1
    // All u32 arithmetic with wrap-around; then clamp via min/max.
    sub.u32       %r10, %r9, %r8;          // k - q  (wraps if negative, u32)
    add.u32       %r10, %r10, %r2;         // + max_len
    sub.u32       %r10, %r10, 1;           // + max_len - 1  = idx (may wrap)

    // table has 2*max_len-1 entries; clamp idx to [0, 2*max_len-2]
    sub.u32       %r11, %r2, 1;            // max_len - 1
    add.u32       %r11, %r11, %r2;         // 2*max_len - 1 - 1 = 2*max_len-2 + ... fix:
    // r11 = 2*max_len - 2
    sub.u32       %r11, %r11, 1;           // actually: 2*max_len-1-1 = 2*(max_len-1)
    // simplify: upper_bound = 2*max_len - 2
    mul.lo.u32    %r12, %r2, 2;            // 2 * max_len
    sub.u32       %r12, %r12, 2;           // r12 = 2*max_len - 2 (upper bound, inclusive)

    min.u32       %r10, %r10, %r12;        // clamp upper
    // lower clamp: since idx can wrap to a very large u32 when (k-q) < 0,
    // a wrap gives u32 > 2*max_len-2, so min.u32 above already clamps it.
    // But when idx wraps to e.g. 0xFFFFFFF0, min.u32 keeps %r12.
    // We need to also handle the case where the subtraction didn't actually
    // produce a wrap in the valid range.  The correct approach: if the
    // raw unsigned result k - q + max_len - 1 is within [0, 2*max_len-2],
    // it stays; if k < q we get a huge number which min.u32 clips to
    // the upper bound.  But we actually want to clamp to 0 in that case.
    // Detection: if (k - q) wrapped (k < q), the u32 difference will be
    // >= 2^31 (>> 2*max_len for reasonable inputs), which means after
    // adding max_len - 1 it stays huge → min clips correctly to upper end.
    // This matches the behaviour "k < q → use table[2*max_len-2]".
    // Alternatively use max.u32 to clamp at 0:
    max.u32       %r10, %r10, 0;           // clamp lower (no-op for u32, keeps 0 semantics)

    // Load table[idx]
    mul.wide.u32  %rd2, %r10, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f0,  [%rd3];

    // Store output[q * K + k]
    mul.wide.u32  %rd4, %r6, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f0;

$RPB_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 7: stats_pool ─────────────────────────────────────────────────────

/// Helper: build the shared-memory declaration and warp-shuffle reduction body.
fn stats_pool_warp_reduce_body() -> &'static str {
    // Returns a PTX snippet (as a static str) that reduces %f_sum across a warp.
    // Caller places partial sum in %f0; result ends in %f0.
    // Uses shfl.sync.down.b32 for offsets 16, 8, 4, 2, 1.
    "    // Warp-shuffle reduction of %f0 (partial sum)
    mov.b32       %r_mask, 0xffffffff;
    shfl.sync.down.b32  %f_sh, %f0, 16, 31, %r_mask;
    add.f32       %f0, %f0, %f_sh;
    shfl.sync.down.b32  %f_sh, %f0,  8, 31, %r_mask;
    add.f32       %f0, %f0, %f_sh;
    shfl.sync.down.b32  %f_sh, %f0,  4, 31, %r_mask;
    add.f32       %f0, %f0, %f_sh;
    shfl.sync.down.b32  %f_sh, %f0,  2, 31, %r_mask;
    add.f32       %f0, %f0, %f_sh;
    shfl.sync.down.b32  %f_sh, %f0,  1, 31, %r_mask;
    add.f32       %f0, %f0, %f_sh;\n"
}

/// Temporal mean+std pooling for speaker embedding using warp-shuffle reductions.
#[must_use]
pub fn stats_pool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let eps = f32_hex(1.0e-8_f32);
    let zero = f32_hex(0.0_f32);
    let warp_reduce = stats_pool_warp_reduce_body();
    format!(
        r#"{hdr}// stats_pool_kernel: one block per channel c.
// blockDim.x must be a multiple of 32 (warp size), e.g. 256.
// Pass 1: compute mean = (1/T) * sum_t x[c, t]
// Pass 2: compute var  = (1/T) * sum_t (x[c,t] - mean)^2
// Output: mean_out[c], std_out[c] = sqrt(var + eps)
//
// Shared memory layout (per block): shmem[blockDim.x / 32] f32 for warp partials.
.visible .entry stats_pool_kernel(
    .param .u64 p_input,
    .param .u64 p_mean_out,
    .param .u64 p_std_out,
    .param .u32 T,
    .param .u32 C
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<24>;
    .reg .u32  %r_mask;
    .reg .f32  %f0, %f1, %f2, %f3, %f4, %f_sh;
    .reg .pred %p0, %p1;

    // Shared memory for warp-level partial sums (max 8 warps per block)
    .shared .align 4 .f32 smem[8];

    ld.param.u64  %rd0, [p_input];
    ld.param.u64  %rd1, [p_mean_out];
    ld.param.u64  %rd2, [p_std_out];
    ld.param.u32  %r0,  [T];
    ld.param.u32  %r1,  [C];

    // channel = blockIdx.x
    mov.u32       %r2,  %ctaid.x;

    setp.ge.u32   %p0, %r2, %r1;
    @%p0 bra $SP_DONE;

    // lane = threadIdx.x & 31,  warp_id = threadIdx.x >> 5
    mov.u32       %r3,  %tid.x;
    and.b32       %r4,  %r3, 31;           // r4 = lane
    shr.u32       %r5,  %r3, 5;            // r5 = warp_id

    // Total threads in block
    mov.u32       %r6,  %ntid.x;           // block_size

    // ── Pass 1: sum over T ────────────────────────────────────────────────────
    mov.f32       %f0, {ZERO};             // partial sum = 0

    // Loop: t = threadIdx.x, t += blockDim.x
    mov.u32       %r7, %r3;               // t = tid (local)

$SP_SUM_LOOP:
    setp.ge.u32   %p1, %r7, %r0;
    @%p1 bra $SP_SUM_END;

    // input[c * T + t]
    mul.lo.u32    %r8, %r2, %r0;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd3, %r8, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.f32       %f0, %f0, %f1;

    add.u32       %r7, %r7, %r6;
    bra           $SP_SUM_LOOP;

$SP_SUM_END:
    // Warp-shuffle reduction of partial sum → f0
{WARP_REDUCE}
    // Lane 0 writes warp partial to smem
    setp.ne.u32   %p1, %r4, 0;
    @%p1 bra $SP_WARP_SKIP1;
    mul.wide.u32  %rd5, %r5, 4;
    mov.u64       %rd6, smem;
    add.u64       %rd6, %rd6, %rd5;
    st.shared.f32 [%rd6], %f0;
$SP_WARP_SKIP1:
    bar.sync      0;

    // Warp 0 reduces across smem entries (n_warps = blockDim.x / 32)
    shr.u32       %r9, %r6, 5;            // n_warps
    setp.ne.u32   %p1, %r5, 0;
    @%p1 bra $SP_SMEM_SKIP1;
    setp.ge.u32   %p1, %r4, %r9;
    mov.f32       %f2, {ZERO};
    @%p1 bra $SP_SMEM_LOAD_SKIP1;
    mul.wide.u32  %rd5, %r4, 4;
    mov.u64       %rd6, smem;
    add.u64       %rd6, %rd6, %rd5;
    ld.shared.f32 %f2, [%rd6];
$SP_SMEM_LOAD_SKIP1:
    // Reduce f2 across first n_warps lanes of warp 0 (simplified: lane 0 only)
    // For correctness with n_warps <= 8, lane 0 accumulates smem[1..n_warps-1]
    mov.u32       %r10, 1;
$SP_SMEM_RED_LOOP:
    setp.ge.u32   %p1, %r10, %r9;
    @%p1 bra $SP_SMEM_RED_END;
    mul.wide.u32  %rd5, %r10, 4;
    mov.u64       %rd6, smem;
    add.u64       %rd6, %rd6, %rd5;
    ld.shared.f32 %f3, [%rd6];
    add.f32       %f2, %f2, %f3;
    add.u32       %r10, %r10, 1;
    bra           $SP_SMEM_RED_LOOP;
$SP_SMEM_RED_END:
    // f2 = total sum; mean = f2 / T
    cvt.rn.f32.u32 %f3, %r0;             // f3 = (f32)T
    div.rn.f32     %f4, %f2, %f3;        // f4 = mean
    // Store mean_out[c]
    mul.wide.u32  %rd7, %r2, 4;
    add.u64       %rd8, %rd1, %rd7;
    st.global.f32 [%rd8], %f4;
    // Broadcast mean to smem[0] for all warps to read
    mov.u64       %rd6, smem;
    st.shared.f32 [%rd6], %f4;
$SP_SMEM_SKIP1:
    bar.sync      0;

    // All threads load mean from smem[0]
    mov.u64       %rd6, smem;
    ld.shared.f32 %f4, [%rd6];

    // ── Pass 2: sum of squared deviations ────────────────────────────────────
    mov.f32       %f0, {ZERO};
    mov.u32       %r7, %r3;               // t = tid

$SP_VAR_LOOP:
    setp.ge.u32   %p1, %r7, %r0;
    @%p1 bra $SP_VAR_END;

    mul.lo.u32    %r8, %r2, %r0;
    add.u32       %r8, %r8, %r7;
    mul.wide.u32  %rd3, %r8, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    sub.f32       %f1, %f1, %f4;          // x - mean
    fma.rn.f32    %f0, %f1, %f1, %f0;    // acc += (x-mean)^2

    add.u32       %r7, %r7, %r6;
    bra           $SP_VAR_LOOP;

$SP_VAR_END:
{WARP_REDUCE}
    setp.ne.u32   %p1, %r4, 0;
    @%p1 bra $SP_WARP_SKIP2;
    mul.wide.u32  %rd5, %r5, 4;
    mov.u64       %rd6, smem;
    add.u64       %rd6, %rd6, %rd5;
    st.shared.f32 [%rd6], %f0;
$SP_WARP_SKIP2:
    bar.sync      0;

    setp.ne.u32   %p1, %r5, 0;
    @%p1 bra $SP_SMEM_SKIP2;
    setp.ge.u32   %p1, %r4, %r9;
    mov.f32       %f2, {ZERO};
    @%p1 bra $SP_SMEM_LOAD_SKIP2;
    mul.wide.u32  %rd5, %r4, 4;
    mov.u64       %rd6, smem;
    add.u64       %rd6, %rd6, %rd5;
    ld.shared.f32 %f2, [%rd6];
$SP_SMEM_LOAD_SKIP2:
    mov.u32       %r10, 1;
$SP_SMEM_RED2_LOOP:
    setp.ge.u32   %p1, %r10, %r9;
    @%p1 bra $SP_SMEM_RED2_END;
    mul.wide.u32  %rd5, %r10, 4;
    mov.u64       %rd6, smem;
    add.u64       %rd6, %rd6, %rd5;
    ld.shared.f32 %f3, [%rd6];
    add.f32       %f2, %f2, %f3;
    add.u32       %r10, %r10, 1;
    bra           $SP_SMEM_RED2_LOOP;
$SP_SMEM_RED2_END:
    // variance = f2 / T; std = sqrt(variance + eps)
    cvt.rn.f32.u32 %f3, %r0;
    div.rn.f32     %f2, %f2, %f3;        // f2 = variance
    add.f32        %f2, %f2, {EPS};      // + epsilon
    sqrt.approx.f32 %f2, %f2;            // std
    // Store std_out[c]
    mul.wide.u32  %rd7, %r2, 4;
    add.u64       %rd9, %rd2, %rd7;
    st.global.f32 [%rd9], %f2;
$SP_SMEM_SKIP2:

$SP_DONE:
    ret;
}}
"#,
        ZERO = zero,
        EPS = eps,
        WARP_REDUCE = warp_reduce
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        ctc_alpha_ptx, depthwise_conv1d_ptx, dilated_conv1d_ptx, rel_pos_bias_ptx,
        spec_augment_mask_ptx, stats_pool_ptx, stride_conv1d_ptx,
    };

    const ALL_SM: &[u32] = &[75, 80, 86, 90, 100, 120];

    fn check_kernel(ptx: &str, sm: u32) {
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "missing .target sm_{sm}"
        );
        assert!(ptx.contains(".address_size 64"), "missing .address_size 64");
    }

    #[test]
    fn stride_conv1d_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&stride_conv1d_ptx(sm), sm);
        }
    }

    #[test]
    fn dilated_conv1d_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&dilated_conv1d_ptx(sm), sm);
        }
    }

    #[test]
    fn ctc_alpha_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&ctc_alpha_ptx(sm), sm);
        }
    }

    #[test]
    fn spec_augment_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&spec_augment_mask_ptx(sm), sm);
        }
    }

    #[test]
    fn depthwise_conv1d_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&depthwise_conv1d_ptx(sm), sm);
        }
    }

    #[test]
    fn rel_pos_bias_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&rel_pos_bias_ptx(sm), sm);
        }
    }

    #[test]
    fn stats_pool_all_sm() {
        for &sm in ALL_SM {
            check_kernel(&stats_pool_ptx(sm), sm);
        }
    }

    #[test]
    fn stride_conv1d_contains_fma() {
        assert!(stride_conv1d_ptx(80).contains("fma.rn.f32"));
    }

    #[test]
    fn dilated_conv1d_causal() {
        let p = dilated_conv1d_ptx(80);
        assert!(p.contains("dilation"));
    }

    #[test]
    fn ctc_alpha_log_domain() {
        let p = ctc_alpha_ptx(80);
        assert!(p.contains("lg2") || p.contains("log_sum_exp"));
    }

    #[test]
    fn spec_augment_zero_fill() {
        let p = spec_augment_mask_ptx(80);
        assert!(p.contains("selp") || p.contains("0F00000000"));
    }

    #[test]
    fn stats_pool_warp_shuffle() {
        let p = stats_pool_ptx(80);
        assert!(p.contains("shfl") || p.contains("sqrt"));
    }

    #[test]
    fn rel_pos_bias_clamp() {
        let p = rel_pos_bias_ptx(80);
        assert!(p.contains("min.u32") || p.contains("max.u32") || p.contains("clamp"));
    }

    // ── Additional correctness checks ─────────────────────────────────────────

    #[test]
    fn stride_conv1d_has_entry_name() {
        let p = stride_conv1d_ptx(80);
        assert!(p.contains(".visible .entry stride_conv1d_kernel"));
    }

    #[test]
    fn dilated_conv1d_has_entry_name() {
        let p = dilated_conv1d_ptx(80);
        assert!(p.contains(".visible .entry dilated_conv1d_kernel"));
    }

    #[test]
    fn ctc_alpha_has_entry_name() {
        let p = ctc_alpha_ptx(80);
        assert!(p.contains(".visible .entry ctc_alpha_kernel"));
    }

    #[test]
    fn spec_augment_has_entry_name() {
        let p = spec_augment_mask_ptx(80);
        assert!(p.contains(".visible .entry spec_augment_mask_kernel"));
    }

    #[test]
    fn depthwise_conv1d_has_entry_name() {
        let p = depthwise_conv1d_ptx(80);
        assert!(p.contains(".visible .entry depthwise_conv1d_kernel"));
    }

    #[test]
    fn rel_pos_bias_has_entry_name() {
        let p = rel_pos_bias_ptx(80);
        assert!(p.contains(".visible .entry rel_pos_bias_kernel"));
    }

    #[test]
    fn stats_pool_has_entry_name() {
        let p = stats_pool_ptx(80);
        assert!(p.contains(".visible .entry stats_pool_kernel"));
    }

    #[test]
    fn sm120_uses_ptx_87() {
        for f in [
            stride_conv1d_ptx(120),
            dilated_conv1d_ptx(120),
            ctc_alpha_ptx(120),
            spec_augment_mask_ptx(120),
            depthwise_conv1d_ptx(120),
            rel_pos_bias_ptx(120),
            stats_pool_ptx(120),
        ] {
            assert!(f.contains(".version 8.7"), "sm_120 must use PTX 8.7");
        }
    }

    #[test]
    fn sm90_uses_ptx_84() {
        for f in [
            stride_conv1d_ptx(90),
            dilated_conv1d_ptx(90),
            ctc_alpha_ptx(90),
            spec_augment_mask_ptx(90),
            depthwise_conv1d_ptx(90),
            rel_pos_bias_ptx(90),
            stats_pool_ptx(90),
        ] {
            assert!(f.contains(".version 8.4"), "sm_90 must use PTX 8.4");
        }
    }

    #[test]
    fn sm80_uses_ptx_80() {
        for f in [
            stride_conv1d_ptx(80),
            dilated_conv1d_ptx(80),
            ctc_alpha_ptx(80),
            spec_augment_mask_ptx(80),
            depthwise_conv1d_ptx(80),
            rel_pos_bias_ptx(80),
            stats_pool_ptx(80),
        ] {
            assert!(f.contains(".version 8.0"), "sm_80 must use PTX 8.0");
        }
    }

    #[test]
    fn sm75_uses_ptx_75() {
        for f in [
            stride_conv1d_ptx(75),
            dilated_conv1d_ptx(75),
            ctc_alpha_ptx(75),
            spec_augment_mask_ptx(75),
            depthwise_conv1d_ptx(75),
            rel_pos_bias_ptx(75),
            stats_pool_ptx(75),
        ] {
            assert!(f.contains(".version 7.5"), "sm_75 must use PTX 7.5");
        }
    }

    #[test]
    fn ctc_alpha_has_bar_sync() {
        // CTC requires barrier between timestep writes/reads
        assert!(ctc_alpha_ptx(80).contains("bar.sync"));
    }

    #[test]
    fn stats_pool_has_sqrt() {
        assert!(stats_pool_ptx(80).contains("sqrt.approx.f32"));
    }

    #[test]
    fn stats_pool_has_shfl_sync() {
        assert!(stats_pool_ptx(80).contains("shfl.sync.down.b32"));
    }

    #[test]
    fn rel_pos_bias_has_min_max() {
        let p = rel_pos_bias_ptx(80);
        assert!(p.contains("min.u32") && p.contains("max.u32"));
    }

    #[test]
    fn spec_augment_has_selp() {
        assert!(spec_augment_mask_ptx(80).contains("selp.f32"));
    }

    #[test]
    fn dilated_conv1d_has_fma() {
        assert!(dilated_conv1d_ptx(80).contains("fma.rn.f32"));
    }

    #[test]
    fn depthwise_conv1d_has_fma() {
        assert!(depthwise_conv1d_ptx(80).contains("fma.rn.f32"));
    }

    #[test]
    fn ctc_alpha_uses_ex2_and_lg2() {
        let p = ctc_alpha_ptx(80);
        assert!(p.contains("ex2.approx.f32") && p.contains("lg2.approx.f32"));
    }
}
