//! PTX GPU kernel sources for State Space Model (SSM) operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`selective_scan_ptx`] | Mamba S6 selective scan (sequential per thread) |
//! | [`parallel_scan_ptx`] | Warp-level (A,b) associative prefix scan |
//! | [`depthwise_conv1d_ptx`] | Causal 1-D depthwise convolution |
//! | [`wkv_forward_ptx`] | RWKV WKV numerically-stable forward pass |
//! | [`ssd_chunk_ptx`] | Mamba-2 SSD chunk computation |
//! | [`hippo_legendre_ptx`] | HiPPO-LegS forward Euler coefficient update |
//! | [`rms_norm_silu_ptx`] | Fused RMSNorm + SiLU gate |

// ─── Hex encoding ────────────────────────────────────────────────────────────

/// Encode a `f32` as a PTX hexadecimal float literal (e.g., `0F3F800000` = 1.0f).
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── PTX header helper ───────────────────────────────────────────────────────

fn ptx_header(sm: u32) -> String {
    let ptx_ver = if sm >= 100 {
        "8.7"
    } else if sm >= 90 {
        "8.4"
    } else if sm >= 80 {
        "8.0"
    } else {
        "7.5"
    };
    format!(".version {ptx_ver}\n.target sm_{sm}\n.address_size 64\n\n")
}

// ─── Kernel 1: selective_scan ─────────────────────────────────────────────────

/// Mamba S6 selective scan kernel — one thread handles ONE channel-dimension
/// slice through the entire time axis.
///
/// Implements the S6 recurrence per channel `d`:
/// ```text
/// h = 0
/// for t in 0..seq_len:
///     u_t    = p_u   [t * d_model + d]
///     a_bar  = p_a_bar[t * d_model + d]   // pre-computed exp(Δ·A)
///     b_bar  = p_b_bar[t * d_model + d]   // pre-computed Δ·B·u
///     c      = p_c   [t * d_model + d]
///     h      = a_bar * h + b_bar * u_t    // state update
///     out[t * d_model + d] = c * h        // output projection
/// ```
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_u` | `u64` (→ `f32*`) | Input tensor `[seq_len × d_model]` |
/// | `p_delta` | `u64` (→ `f32*`) | Δ values (unused in recurrence but kept for ABI) |
/// | `p_a_bar` | `u64` (→ `f32*`) | Pre-computed `exp(Δ·A)` `[seq_len × d_model]` |
/// | `p_b_bar` | `u64` (→ `f32*`) | Pre-computed `Δ·B` `[seq_len × d_model]` |
/// | `p_c` | `u64` (→ `f32*`) | C projection `[seq_len × d_model]` |
/// | `p_out` | `u64` (→ `f32*`) | Output `[seq_len × d_model]` |
/// | `seq_len` | `u32` | Sequence length |
/// | `d_model` | `u32` | Model / channel dimension |
///
/// Launch: `grid = ceil(d_model / 256)`, `block = 256`.
pub fn selective_scan_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry selective_scan(
    .param .u64 p_u,
    .param .u64 p_delta,
    .param .u64 p_a_bar,
    .param .u64 p_b_bar,
    .param .u64 p_c,
    .param .u64 p_out,
    .param .u32 seq_len,
    .param .u32 d_model
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<8>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    // Load base pointers
    ld.param.u64  %rd0, [p_u];
    ld.param.u64  %rd1, [p_delta];
    ld.param.u64  %rd2, [p_a_bar];
    ld.param.u64  %rd3, [p_b_bar];
    ld.param.u64  %rd4, [p_c];
    ld.param.u64  %rd5, [p_out];

    // Load scalar params
    ld.param.u32  %r0, [seq_len];
    ld.param.u32  %r1, [d_model];

    // Grid-stride: channel index d = blockDim.x * blockIdx.x + threadIdx.x
    mov.u32        %r2, %ntid.x;
    mov.u32        %r3, %ctaid.x;
    mov.u32        %r4, %tid.x;
    mad.lo.u32     %r5, %r2, %r3, %r4;   // r5 = d (channel index)

$SS_OUTER:
    setp.ge.u32    %p0, %r5, %r1;
    @%p0 bra $SS_DONE;

    // Initialise hidden state h = 0.0
    mov.f32        %f0, {ZERO};           // f0 = h

    // r6 = t (time step counter), r7 = t_stride = d_model (row stride)
    mov.u32        %r6, 0;

$SS_TLOOP:
    setp.ge.u32    %p0, %r6, %r0;
    @%p0 bra $SS_TSAVE;

    // byte offset for element [t * d_model + d]:
    //   r_off32 = r6 * r1 + r5
    //   rd_off  = r_off32 * 4
    mad.lo.u32     %r7, %r6, %r1, %r5;
    mul.wide.u32   %rd6, %r7, 4;

    // u_t = p_u[t * d_model + d]
    add.u64        %rd7, %rd0, %rd6;
    ld.global.f32  %f1, [%rd7];

    // a_bar = p_a_bar[t * d_model + d]
    add.u64        %rd7, %rd2, %rd6;
    ld.global.f32  %f2, [%rd7];

    // b_bar = p_b_bar[t * d_model + d]
    add.u64        %rd7, %rd3, %rd6;
    ld.global.f32  %f3, [%rd7];

    // c = p_c[t * d_model + d]
    add.u64        %rd7, %rd4, %rd6;
    ld.global.f32  %f4, [%rd7];

    // h = a_bar * h + b_bar * u_t
    //   = fma(a_bar, h, b_bar * u_t)
    mul.f32        %f5, %f3, %f1;         // b_bar * u_t
    fma.rn.f32     %f0, %f2, %f0, %f5;   // h = a_bar * h + b_bar*u_t

    // out[t * d_model + d] = c * h  (written in-place below after loop)
    // We write immediately to avoid a second pass.
    mul.f32        %f6, %f4, %f0;         // c * h
    add.u64        %rd7, %rd5, %rd6;
    st.global.f32  [%rd7], %f6;

    // t++
    add.u32        %r6, %r6, 1;
    bra            $SS_TLOOP;

$SS_TSAVE:
    // Grid stride: advance channel index by blockDim * gridDim
    mov.u32        %r2, %ntid.x;
    mov.u32        %r3, %nctaid.x;
    mul.lo.u32     %r3, %r2, %r3;
    add.u32        %r5, %r5, %r3;
    bra            $SS_OUTER;

$SS_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 2: parallel_scan ──────────────────────────────────────────────────

/// Warp-level (A, b) associative prefix scan using `shfl.sync.down`.
///
/// Each thread starts with `(a_i, b_i)` loaded from `p_a[i]`, `p_b[i]`.
/// The inclusive prefix-scan monoid is `(a, b) · (a', b') = (a·a', a·b' + b)`,
/// which recovers hidden states from discretised A/B matrices.
///
/// Warp butterfly: for stride in `[1, 2, 4, 8, 16]`:
/// ```text
/// (a_left, b_left) = shfl.sync.down(a, b, stride)
/// a_new = a_i * a_left
/// b_new = fma(a_i, b_left, b_i)
/// if lane >= stride: (a_i, b_i) = (a_new, b_new)
/// ```
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_a` | `u64` (→ `f32*`) | Diagonal A values `[n]` |
/// | `p_b` | `u64` (→ `f32*`) | Input-coupled B values `[n]` |
/// | `p_out_a` | `u64` (→ `f32*`) | Output prefix-scan A `[n]` |
/// | `p_out_b` | `u64` (→ `f32*`) | Output prefix-scan B `[n]` |
/// | `n` | `u32` | Number of elements (should be multiple of 32 for full warps) |
///
/// Launch: `grid = ceil(n / 32)`, `block = 32` (one warp per block).
pub fn parallel_scan_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}.visible .entry parallel_scan(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out_a,
    .param .u64 p_out_b,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<8>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_b];
    ld.param.u64  %rd2, [p_out_a];
    ld.param.u64  %rd3, [p_out_b];
    ld.param.u32  %r0,  [n];

    // Grid-stride at warp granularity
    // global warp id = (blockIdx.x * blockDim.x + threadIdx.x) maps directly to element
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // r4 = global thread index = element index

    // lane = threadIdx.x & 31
    and.b32       %r5, %r3, 31;           // r5 = lane

$PS_OUTER:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $PS_DONE;

    // Load a_i, b_i
    mul.wide.u32  %rd4, %r4, 4;

    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];            // f0 = a_i

    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f1, [%rd5];            // f1 = b_i

    // ── Warp-level inclusive prefix scan ─────────────────────────────────────
    // stride = 1
    shfl.sync.down.b32  %f2, %f0, 1, 31, 0xFFFFFFFF;   // a_left
    shfl.sync.down.b32  %f3, %f1, 1, 31, 0xFFFFFFFF;   // b_left
    setp.ge.u32   %p1, %r5, 1;
    mul.f32        %f4, %f0, %f2;          // a_new = a_i * a_left
    fma.rn.f32     %f5, %f0, %f3, %f1;    // b_new = a_i * b_left + b_i
    @%p1 mov.f32   %f0, %f4;
    @%p1 mov.f32   %f1, %f5;

    // stride = 2
    shfl.sync.down.b32  %f2, %f0, 2, 31, 0xFFFFFFFF;
    shfl.sync.down.b32  %f3, %f1, 2, 31, 0xFFFFFFFF;
    setp.ge.u32   %p1, %r5, 2;
    mul.f32        %f4, %f0, %f2;
    fma.rn.f32     %f5, %f0, %f3, %f1;
    @%p1 mov.f32   %f0, %f4;
    @%p1 mov.f32   %f1, %f5;

    // stride = 4
    shfl.sync.down.b32  %f2, %f0, 4, 31, 0xFFFFFFFF;
    shfl.sync.down.b32  %f3, %f1, 4, 31, 0xFFFFFFFF;
    setp.ge.u32   %p1, %r5, 4;
    mul.f32        %f4, %f0, %f2;
    fma.rn.f32     %f5, %f0, %f3, %f1;
    @%p1 mov.f32   %f0, %f4;
    @%p1 mov.f32   %f1, %f5;

    // stride = 8
    shfl.sync.down.b32  %f2, %f0, 8, 31, 0xFFFFFFFF;
    shfl.sync.down.b32  %f3, %f1, 8, 31, 0xFFFFFFFF;
    setp.ge.u32   %p1, %r5, 8;
    mul.f32        %f4, %f0, %f2;
    fma.rn.f32     %f5, %f0, %f3, %f1;
    @%p1 mov.f32   %f0, %f4;
    @%p1 mov.f32   %f1, %f5;

    // stride = 16
    shfl.sync.down.b32  %f2, %f0, 16, 31, 0xFFFFFFFF;
    shfl.sync.down.b32  %f3, %f1, 16, 31, 0xFFFFFFFF;
    setp.ge.u32   %p1, %r5, 16;
    mul.f32        %f4, %f0, %f2;
    fma.rn.f32     %f5, %f0, %f3, %f1;
    @%p1 mov.f32   %f0, %f4;
    @%p1 mov.f32   %f1, %f5;

    // Store results
    add.u64       %rd5, %rd2, %rd4;
    st.global.f32 [%rd5], %f0;

    add.u64       %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f1;

    // Grid stride: advance by blockDim * gridDim
    mov.u32       %r1, %ntid.x;
    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r6;
    add.u32       %r4, %r4, %r6;
    bra           $PS_OUTER;

$PS_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 3: depthwise_conv1d ───────────────────────────────────────────────

/// Causal 1-D depthwise convolution.
///
/// Implements: `y[c, t] = Σ_{k=0}^{K-1} w[c, k] * x[c, t-k]`
/// with zero-padding for `t - k < 0`.
///
/// One thread per `(channel, time_step)` pair.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_x` | `u64` (→ `f32*`) | Input `[channels × seq_len]` (row-major: c outer, t inner) |
/// | `p_w` | `u64` (→ `f32*`) | Kernel weights `[channels × kernel_size]` |
/// | `p_y` | `u64` (→ `f32*`) | Output `[channels × seq_len]` |
/// | `seq_len` | `u32` | Input / output sequence length |
/// | `channels` | `u32` | Number of independent channels |
/// | `kernel_size` | `u32` | Convolution kernel size K |
///
/// Launch: `grid = ceil(channels * seq_len / 256)`, `block = 256`.
pub fn depthwise_conv1d_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry depthwise_conv1d(
    .param .u64 p_x,
    .param .u64 p_w,
    .param .u64 p_y,
    .param .u32 seq_len,
    .param .u32 channels,
    .param .u32 kernel_size
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<4>;
    .reg .pred %p0, %p1, %p2;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_w];
    ld.param.u64  %rd2, [p_y];
    ld.param.u32  %r0,  [seq_len];
    ld.param.u32  %r1,  [channels];
    ld.param.u32  %r2,  [kernel_size];

    // Grid-stride: one thread per (channel, time_step)
    // global_tid maps to flat index = c * seq_len + t
    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;    // r6 = flat index

    // total = channels * seq_len
    mul.lo.u32    %r7, %r1, %r0;

$DC_OUTER:
    setp.ge.u32   %p0, %r6, %r7;
    @%p0 bra $DC_DONE;

    // c = r6 / seq_len,  t = r6 % seq_len
    div.u32       %r8,  %r6, %r0;         // r8  = c
    rem.u32       %r9,  %r6, %r0;         // r9  = t

    // Accumulator
    mov.f32       %f0, {ZERO};

    // Inner loop over k = 0 .. kernel_size-1
    mov.u32       %r10, 0;                 // k = 0

$DC_KLOOP:
    setp.ge.u32   %p1, %r10, %r2;
    @%p1 bra $DC_KEND;

    // t_src = t - k  (signed arithmetic: use subtraction then check)
    // If t < k, t_src would be negative → zero-pad (skip add)
    setp.lt.u32   %p2, %r9, %r10;
    @%p2 bra $DC_KSKIP;

    sub.u32       %r11, %r9, %r10;        // r11 = t_src = t - k

    // x[c * seq_len + t_src]
    mad.lo.u32    %r12, %r8, %r0, %r11;
    mul.wide.u32  %rd3, %r12, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    // w[c * kernel_size + k]
    mad.lo.u32    %r13, %r8, %r2, %r10;
    mul.wide.u32  %rd5, %r13, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f2, [%rd6];

    // acc += w * x
    fma.rn.f32    %f0, %f2, %f1, %f0;

$DC_KSKIP:
    add.u32       %r10, %r10, 1;
    bra           $DC_KLOOP;

$DC_KEND:
    // Store y[c * seq_len + t] = acc
    mul.wide.u32  %rd7, %r6, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f0;

    // Grid stride
    mov.u32       %r3, %ntid.x;
    mov.u32       %r14, %nctaid.x;
    mul.lo.u32    %r14, %r3, %r14;
    add.u32       %r6, %r6, %r14;
    bra           $DC_OUTER;

$DC_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 4: wkv_forward ────────────────────────────────────────────────────

/// RWKV WKV numerically-stable forward pass.
///
/// Computes the WKV attention mechanism for one channel per thread,
/// sequential over time.  Uses the running-max trick for numerical stability:
///
/// ```text
/// p_t = max(w + k_{t-1}, k_t)          // new running exponent pivot
/// e1 = exp(w + k_{t-1} - p_t)          // prev state scaled
/// e2 = exp(k_t - p_t)                  // new state scale
/// eu = exp(u + k_t - p_t)              // u bonus
///
/// wkv_t = (e1 * (a / b) + eu * v_t) / (e1 + eu)
///       = (e1 * a + eu * v_t) / (e1 * b + eu)
///
/// a_new = e1 * a + e2 * v_t
/// b_new = e1 * b + e2
/// ```
///
/// Uses `ex2.approx.f32` + `lg2.approx.f32` + `rcp.approx.f32` + `fma.rn.f32`.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_k` | `u64` (→ `f32*`) | Keys `[seq_len × channels]` |
/// | `p_v` | `u64` (→ `f32*`) | Values `[seq_len × channels]` |
/// | `p_w` | `u64` (→ `f32*`) | Channel-wise time-decay `[channels]` |
/// | `p_u` | `u64` (→ `f32*`) | Bonus `u` vector `[channels]` |
/// | `p_out` | `u64` (→ `f32*`) | Output `[seq_len × channels]` |
/// | `seq_len` | `u32` | Sequence length |
/// | `channels` | `u32` | Number of channels |
///
/// Launch: `grid = ceil(channels / 256)`, `block = 256`.
pub fn wkv_forward_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // log2(e) used to convert natural exp → base-2: exp(x) = ex2(x * log2e)
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}.visible .entry wkv_forward(
    .param .u64 p_k,
    .param .u64 p_v,
    .param .u64 p_w,
    .param .u64 p_u,
    .param .u64 p_out,
    .param .u32 seq_len,
    .param .u32 channels
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<8>;
    .reg .f32  %f<24>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_k];
    ld.param.u64  %rd1, [p_v];
    ld.param.u64  %rd2, [p_w];
    ld.param.u64  %rd3, [p_u];
    ld.param.u64  %rd4, [p_out];
    ld.param.u32  %r0,  [seq_len];
    ld.param.u32  %r1,  [channels];

    // Grid-stride: one thread per channel
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;    // r5 = c (channel index)

$WKV_OUTER:
    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $WKV_DONE;

    // Load channel-wise w[c] and u[c]
    mul.wide.u32  %rd5, %r5, 4;

    add.u64       %rd6, %rd2, %rd5;
    ld.global.f32 %f0, [%rd6];            // f0 = w (time decay, negative)

    add.u64       %rd6, %rd3, %rd5;
    ld.global.f32 %f1, [%rd6];            // f1 = u (bonus)

    // Running state: a, b, p_prev (running max exponent)
    mov.f32       %f2, {ZERO};            // f2 = a  (numerator running sum)
    mov.f32       %f3, {ZERO};            // f3 = b  (denominator running sum)
    mov.f32       %f4, {NEG_INF};         // f4 = p_prev (running max pivot)

    // r6 = t (time step)
    mov.u32       %r6, 0;

$WKV_TLOOP:
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $WKV_TEND;

    // Flat offset [t * channels + c]
    mad.lo.u32    %r7, %r6, %r1, %r5;
    mul.wide.u32  %rd5, %r7, 4;

    // Load k_t, v_t
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f5, [%rd6];            // f5 = k_t

    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f6, [%rd6];            // f6 = v_t

    // p_new = max(w + p_prev, k_t)
    //       Note: w is the per-step decay (w_c), p_prev is the running max pivot.
    //       The RWKV formulation uses: max(w + k_{{t-1}}, k_t) for stability.
    //       We use p_prev to track the cumulative pivot.
    add.f32       %f7, %f0, %f4;          // f7 = w + p_prev
    max.f32       %f8, %f7, %f5;          // f8 = p_new = max(w + p_prev, k_t)

    // e1 = exp(w + p_prev - p_new)
    //    = ex2( (w + p_prev - p_new) * log2e )
    sub.f32       %f9, %f7, %f8;          // f9  = w + p_prev - p_new
    mul.f32       %f10, %f9, {LOG2E};
    ex2.approx.f32 %f10, %f10;            // f10 = e1

    // e2 = exp(k_t - p_new)
    sub.f32       %f11, %f5, %f8;         // f11 = k_t - p_new
    mul.f32       %f12, %f11, {LOG2E};
    ex2.approx.f32 %f12, %f12;            // f12 = e2

    // eu = exp(u + k_t - p_new)
    add.f32       %f13, %f1, %f11;        // f13 = u + k_t - p_new
    mul.f32       %f14, %f13, {LOG2E};
    ex2.approx.f32 %f14, %f14;            // f14 = eu

    // wkv_t = (e1 * a + eu * v_t) / (e1 * b + eu)
    mul.f32       %f15, %f10, %f2;        // f15 = e1 * a
    fma.rn.f32    %f16, %f14, %f6, %f15; // f16 = e1*a + eu*v_t  (numerator)

    mul.f32       %f17, %f10, %f3;        // f17 = e1 * b
    add.f32       %f18, %f17, %f14;       // f18 = e1*b + eu      (denominator)

    rcp.approx.f32 %f19, %f18;
    mul.f32        %f20, %f16, %f19;      // f20 = wkv_t

    // Store out[t * channels + c]
    add.u64       %rd6, %rd4, %rd5;
    st.global.f32 [%rd6], %f20;

    // Update running state
    // a_new = e1 * a + e2 * v_t
    fma.rn.f32    %f21, %f12, %f6, %f15; // f21 = e1*a + e2*v_t
    mov.f32       %f2, %f21;

    // b_new = e1 * b + e2
    add.f32       %f22, %f17, %f12;       // f22 = e1*b + e2
    mov.f32       %f3, %f22;

    // p_prev = p_new
    mov.f32       %f4, %f8;

    add.u32       %r6, %r6, 1;
    bra           $WKV_TLOOP;

$WKV_TEND:
    // Grid stride: advance channel index
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %nctaid.x;
    mul.lo.u32    %r3, %r2, %r3;
    add.u32       %r5, %r5, %r3;
    bra           $WKV_OUTER;

$WKV_DONE:
    ret;
}}
"#,
        LOG2E = log2e,
        NEG_INF = neg_inf,
        ZERO = zero,
    )
}

// ─── Kernel 5: ssd_chunk ──────────────────────────────────────────────────────

/// Mamba-2 SSD chunk computation.
///
/// For one chunk of length `chunk_len`, computes the causal output:
/// ```text
/// Y[i] = C[i] * Σ_{j=0}^{i} (Π_{k=j+1}^{i} A_k) * B[j] * x[j]
/// ```
///
/// One thread per output position `i` (assumes `chunk_len ≤ 256`).
/// Each thread iterates `j` from 0 to `i`, accumulating `a_prod = Π A_k`
/// and the weighted sum.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_a` | `u64` (→ `f32*`) | Scalar A per step `[chunk_len]` |
/// | `p_b_vec` | `u64` (→ `f32*`) | B projection values `[chunk_len × state_dim]` |
/// | `p_c_vec` | `u64` (→ `f32*`) | C projection values `[chunk_len × state_dim]` |
/// | `p_x` | `u64` (→ `f32*`) | Input values `[chunk_len × state_dim]` |
/// | `p_out` | `u64` (→ `f32*`) | Output `[chunk_len × state_dim]` |
/// | `chunk_len` | `u32` | Number of steps in the chunk (≤ 256) |
/// | `state_dim` | `u32` | State / feature dimension |
///
/// Launch: `grid = ceil(chunk_len * state_dim / 256)`, `block = 256`.
pub fn ssd_chunk_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}.visible .entry ssd_chunk(
    .param .u64 p_a,
    .param .u64 p_b_vec,
    .param .u64 p_c_vec,
    .param .u64 p_x,
    .param .u64 p_out,
    .param .u32 chunk_len,
    .param .u32 state_dim
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_b_vec];
    ld.param.u64  %rd2, [p_c_vec];
    ld.param.u64  %rd3, [p_x];
    ld.param.u64  %rd4, [p_out];
    ld.param.u32  %r0,  [chunk_len];
    ld.param.u32  %r1,  [state_dim];

    // Grid-stride: one thread per (i, s) pair
    // flat_tid = i * state_dim + s
    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;    // r5 = flat tid

    mul.lo.u32    %r6, %r0, %r1;          // r6 = chunk_len * state_dim

$SSD_OUTER:
    setp.ge.u32   %p0, %r5, %r6;
    @%p0 bra $SSD_DONE;

    // i = r5 / state_dim,  s = r5 % state_dim
    div.u32       %r7, %r5, %r1;          // r7 = i
    rem.u32       %r8, %r5, %r1;          // r8 = s

    // Load C[i, s] = p_c_vec[i * state_dim + s]
    mad.lo.u32    %r9, %r7, %r1, %r8;
    mul.wide.u32  %rd5, %r9, 4;
    add.u64       %rd6, %rd2, %rd5;
    ld.global.f32 %f0, [%rd6];            // f0 = C_i_s

    // Accumulator for sum_{{j=0}}^{{i}} a_prod * B[j,s] * x[j,s]
    mov.f32       %f1, {ZERO};            // f1 = acc

    // Inner loop: j from 0 to i (inclusive)
    // a_prod = product_{{k=j+1..i}} A_k — computed by iterating j downward:
    // Start j = i, a_prod = 1. For each j, multiply a_prod * A_j before advancing left.
    // This way we accumulate in reverse to get the correct product.
    // j = i: a_prod starts at 1 (no A factors from j+1=i+1 to i, empty product)
    mov.u32       %r10, %r7;              // r10 = j = i
    mov.f32       %f2, {ONE};             // f2 = a_prod = 1 (empty product for j=i)

$SSD_JLOOP:
    // Load B[j, s] = p_b_vec[j * state_dim + s]
    mad.lo.u32    %r11, %r10, %r1, %r8;
    mul.wide.u32  %rd5, %r11, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f3, [%rd6];            // f3 = B_j_s

    // Load x[j, s] = p_x[j * state_dim + s]
    add.u64       %rd7, %rd3, %rd5;
    ld.global.f32 %f4, [%rd7];            // f4 = x_j_s

    // acc += a_prod * B[j,s] * x[j,s]
    mul.f32       %f5, %f3, %f4;          // B * x
    fma.rn.f32    %f1, %f2, %f5, %f1;    // acc += a_prod * B*x

    // Check j == 0 (stop condition)
    setp.eq.u32   %p1, %r10, 0;
    @%p1 bra $SSD_JEND;

    // For the next j (j-1), update a_prod *= A_j
    // a_prod for position j-1 is a_prod(j) * A_j
    mul.wide.u32  %rd5, %r10, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f6, [%rd6];            // f6 = A_j (scalar per time step)
    mul.f32       %f2, %f2, %f6;          // a_prod *= A_j

    sub.u32       %r10, %r10, 1;          // j--
    bra           $SSD_JLOOP;

$SSD_JEND:
    // out[i, s] = C[i, s] * acc
    mul.f32       %f7, %f0, %f1;

    mul.wide.u32  %rd5, %r5, 4;
    add.u64       %rd8, %rd4, %rd5;
    st.global.f32 [%rd8], %f7;

    // Grid stride
    mov.u32       %r2, %ntid.x;
    mov.u32       %r12, %nctaid.x;
    mul.lo.u32    %r12, %r2, %r12;
    add.u32       %r5, %r5, %r12;
    bra           $SSD_OUTER;

$SSD_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
    )
}

// ─── Kernel 6: hippo_legendre ─────────────────────────────────────────────────

/// HiPPO-LegS forward Euler coefficient update.
///
/// Approximates the update step:
/// ```text
/// c_n(t+Δ) ≈ c_n(t) * (1 - Δ*(n+1)) + Δ * sqrt(2n+1) * u
/// ```
///
/// One thread per coefficient index `n` in `[0, n_coeffs)`.
/// Uses `sqrt.approx.f32` on the float `(2n+1)` to compute the HiPPO-LegS
/// input coupling strength.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_c` | `u64` (→ `f32*`) | Current coefficient vector `[n_coeffs]` |
/// | `p_c_out` | `u64` (→ `f32*`) | Updated coefficient vector `[n_coeffs]` |
/// | `u_val` | `f32` | Scalar input value `u(t)` |
/// | `delta` | `f32` | Forward Euler step size Δ |
/// | `n_coeffs` | `u32` | Number of HiPPO polynomial coefficients |
///
/// Launch: `grid = ceil(n_coeffs / 256)`, `block = 256`.
pub fn hippo_legendre_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    let two = f32_hex(2.0_f32);
    format!(
        r#"{hdr}.visible .entry hippo_legendre(
    .param .u64 p_c,
    .param .u64 p_c_out,
    .param .f32 u_val,
    .param .f32 delta,
    .param .u32 n_coeffs
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<6>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_c];
    ld.param.u64  %rd1, [p_c_out];
    ld.param.f32  %f0,  [u_val];
    ld.param.f32  %f1,  [delta];
    ld.param.u32  %r0,  [n_coeffs];

    // Grid-stride: one thread per coefficient n
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // r4 = n

$HL_OUTER:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $HL_DONE;

    // Load c_n
    mul.wide.u32  %rd2, %r4, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f2, [%rd3];            // f2 = c_n

    // Compute (n+1) as float
    add.u32       %r5, %r4, 1;
    cvt.rn.f32.u32 %f3, %r5;              // f3 = float(n+1)

    // decay factor: (1 - delta * (n+1))
    mul.f32       %f4, %f1, %f3;          // delta * (n+1)
    sub.f32       %f5, {ONE}, %f4;        // 1 - delta*(n+1)

    // Compute sqrt(2n+1)
    // 2n+1 = 2*(n+1) - 1 = 2*f3 - 1
    mul.f32       %f6, {TWO}, %f3;        // 2*(n+1)
    sub.f32       %f7, %f6, {ONE};        // 2n+1
    sqrt.approx.f32 %f8, %f7;             // sqrt(2n+1)

    // c_n_new = c_n * (1 - delta*(n+1)) + delta * sqrt(2n+1) * u
    //         = fma(c_n, decay, delta * sqrt(2n+1) * u)
    mul.f32       %f9, %f1, %f8;          // delta * sqrt(2n+1)
    mul.f32       %f10, %f9, %f0;         // delta * sqrt(2n+1) * u
    fma.rn.f32    %f11, %f2, %f5, %f10;  // c_n * decay + input_coupling

    // Store c_out[n]
    add.u64       %rd4, %rd1, %rd2;
    st.global.f32 [%rd4], %f11;

    // Grid stride
    mov.u32       %r1, %ntid.x;
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r5, %r1, %r5;
    add.u32       %r4, %r4, %r5;
    bra           $HL_OUTER;

$HL_DONE:
    ret;
}}
"#,
        ONE = one,
        TWO = two,
    )
}

// ─── Kernel 7: rms_norm_silu ──────────────────────────────────────────────────

/// Fused RMSNorm + SiLU gate for Mamba block output.
///
/// Computes:
/// ```text
/// rms(x) = sqrt(mean(x²) + ε)
/// out[i]  = (x[i] / rms(x) * g[i]) * silu(z[i])
/// silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
/// ```
///
/// Two-pass kernel:
/// - **Pass 1**: each thread loads `x[tid]`, squares it, then a warp butterfly
///   sum with `shfl.sync.bfly.b32` reduces to the RMS.  Only the first lane
///   of each warp does the sqrt + rcp to produce `rms_inv`.  The value is
///   broadcast back with `shfl.sync.bfly.b32` mask=`0xFFFFFFFF`, idx=`0`.
/// - **Pass 2**: each thread computes the fused output.
///
/// # Parameters
///
/// | Param | Type | Description |
/// |-------|------|-------------|
/// | `p_x` | `u64` (→ `f32*`) | Input `x` `[n]` |
/// | `p_g` | `u64` (→ `f32*`) | Scale gate `g` `[n]` |
/// | `p_z` | `u64` (→ `f32*`) | SiLU gate input `z` `[n]` |
/// | `p_out` | `u64` (→ `f32*`) | Output `[n]` |
/// | `n` | `u32` | Number of elements |
/// | `eps` | `f32` | RMSNorm epsilon |
///
/// Launch: `grid = ceil(n / 32)`, `block = 32` (one warp per block).
/// The warp-internal reduction handles exactly the 32-element window.
/// For `n` not divisible by 32 the last block's out-of-bounds threads are
/// guarded by the `setp.ge` bound check.
pub fn rms_norm_silu_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let inv32 = f32_hex(1.0_f32 / 32.0_f32);
    let log2e = f32_hex(std::f32::consts::LOG2_E);
    format!(
        r#"{hdr}.visible .entry rms_norm_silu(
    .param .u64 p_x,
    .param .u64 p_g,
    .param .u64 p_z,
    .param .u64 p_out,
    .param .u32 n,
    .param .f32 eps
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<8>;
    .reg .f32  %f<20>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_g];
    ld.param.u64  %rd2, [p_z];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [eps];

    // Grid-stride: each thread maps to one element index.
    // Block = 32 (one warp).  All 32 lanes always execute the warp shuffle
    // together; out-of-bounds lanes contribute 0 to the sum.
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;    // r4 = global tid (element index)

    // Compute stride = blockDim * gridDim once (reused in loop footer)
    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r1, %r6;          // r7 = stride

$RN_OUTER:
    // ── Early exit when all 32 warp lanes are out of bounds ───────────────────
    // The warp starts at r4 (lane 0 base); lane 31 is r4 + 31.
    // If even lane 0 is past n, the entire warp is done.
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $RN_DONE;

    // ── Pass 1: load x[i] or 0; compute x², warp-reduce ──────────────────────
    // Each lane conditionally loads x[i]; out-of-bounds contribute 0.
    setp.lt.u32   %p1, %r4, %r0;          // p1 = (i < n)
    mov.f32       %f1, {ZERO};            // default x = 0
    mov.f32       %f2, {ZERO};            // default x² = 0

    @%p1 mul.wide.u32  %rd4, %r4, 4;
    @%p1 add.u64       %rd5, %rd0, %rd4;
    @%p1 ld.global.f32 %f1, [%rd5];       // f1 = x[i]  (if in-bounds)
    @%p1 mul.f32       %f2, %f1, %f1;     // f2 = x[i]²

    // Warp all-reduce via butterfly (shfl.sync.bfly): sum of squares
    shfl.sync.bfly.b32  %f3, %f2, 16, 31, 0xFFFFFFFF;
    add.f32       %f2, %f2, %f3;
    shfl.sync.bfly.b32  %f3, %f2,  8, 31, 0xFFFFFFFF;
    add.f32       %f2, %f2, %f3;
    shfl.sync.bfly.b32  %f3, %f2,  4, 31, 0xFFFFFFFF;
    add.f32       %f2, %f2, %f3;
    shfl.sync.bfly.b32  %f3, %f2,  2, 31, 0xFFFFFFFF;
    add.f32       %f2, %f2, %f3;
    shfl.sync.bfly.b32  %f3, %f2,  1, 31, 0xFFFFFFFF;
    add.f32       %f2, %f2, %f3;           // f2 = warp-wide sum of x²

    // mean_sq = sum / 32
    mul.f32       %f4, %f2, {INV32};       // f4 = mean(x²)

    // rms_inv = 1 / sqrt(mean_sq + eps)
    add.f32       %f5, %f4, %f0;           // f5 = mean_sq + eps
    sqrt.approx.f32 %f6, %f5;              // f6 = rms
    rcp.approx.f32  %f7, %f6;              // f7 = rms_inv

    // ── Pass 2: fused norm + SiLU gate (in-bounds only) ──────────────────────
    @!%p1 bra $RN_STRIDE;

    // x_norm = x[i] * rms_inv
    mul.f32       %f8, %f1, %f7;           // f8 = x[i] / rms

    // Load g[i] and scale
    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f9, [%rd5];             // f9 = g[i]
    mul.f32       %f10, %f8, %f9;          // f10 = x_norm * g[i]

    // Load z[i] for SiLU gate
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f11, [%rd5];            // f11 = z[i]

    // silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
    // exp(-z) = ex2(-z * log2e)
    neg.f32       %f12, %f11;              // -z
    mul.f32       %f13, %f12, {LOG2E};     // -z * log2e
    ex2.approx.f32 %f14, %f13;             // exp(-z)
    add.f32       %f15, {ONE}, %f14;       // 1 + exp(-z)
    rcp.approx.f32 %f16, %f15;             // 1 / (1 + exp(-z))
    mul.f32       %f17, %f11, %f16;        // z * sigmoid(z) = silu(z)

    // out[i] = (x_norm * g[i]) * silu(z[i])
    mul.f32       %f18, %f10, %f17;
    add.u64       %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f18;

$RN_STRIDE:
    // All lanes advance by stride; loop exit check is at top of $RN_OUTER.
    add.u32       %r4, %r4, %r7;
    bra           $RN_OUTER;

$RN_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        INV32 = inv32,
        LOG2E = log2e,
    )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];

    // ── Helper ───────────────────────────────────────────────────────────────

    fn check_visible_entry(ptx: &str, sm: u32, entry_name: &str) {
        assert!(
            ptx.contains(&format!(".target sm_{sm}")),
            "missing .target sm_{sm} in kernel '{entry_name}'"
        );
        assert!(
            ptx.contains(".address_size 64"),
            "missing .address_size 64 in kernel '{entry_name}'"
        );
        assert!(
            ptx.contains(".visible .entry"),
            "missing .visible .entry in kernel '{entry_name}'"
        );
        assert!(
            ptx.contains(entry_name),
            "missing entry name '{entry_name}'"
        );
    }

    // ── f32_hex ───────────────────────────────────────────────────────────────

    #[test]
    fn f32_hex_one() {
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }

    #[test]
    fn f32_hex_zero() {
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
    }

    #[test]
    fn f32_hex_neg_one() {
        assert_eq!(f32_hex(-1.0_f32), "0FBF800000");
    }

    #[test]
    fn f32_hex_log2e() {
        // 1.4426950 -> 0x3FB8AA3B
        let h = f32_hex(std::f32::consts::LOG2_E);
        assert!(h.starts_with("0F"), "should start with 0F");
        assert_eq!(h.len(), 10, "should be 10 chars");
    }

    // ── ptx_header ────────────────────────────────────────────────────────────

    #[test]
    fn ptx_header_sm75_version_75() {
        let h = super::ptx_header(75);
        assert!(h.contains(".version 7.5"), "sm75 → PTX 7.5");
        assert!(h.contains(".target sm_75"));
    }

    #[test]
    fn ptx_header_sm80_version_80() {
        let h = super::ptx_header(80);
        assert!(h.contains(".version 8.0"), "sm80 → PTX 8.0");
        assert!(h.contains(".target sm_80"));
    }

    #[test]
    fn ptx_header_sm90_version_84() {
        let h = super::ptx_header(90);
        assert!(h.contains(".version 8.4"), "sm90 → PTX 8.4");
        assert!(h.contains(".target sm_90"));
    }

    #[test]
    fn ptx_header_sm120_version_87() {
        let h = super::ptx_header(120);
        assert!(h.contains(".version 8.7"), "sm120 → PTX 8.7");
        assert!(h.contains(".target sm_120"));
    }

    // ── selective_scan ────────────────────────────────────────────────────────

    #[test]
    fn selective_scan_version_and_target_sm75() {
        let ptx = selective_scan_ptx(75);
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_75"));
    }

    #[test]
    fn selective_scan_version_and_target_sm120() {
        let ptx = selective_scan_ptx(120);
        assert!(ptx.contains(".version 8.7"));
        assert!(ptx.contains(".target sm_120"));
    }

    #[test]
    fn selective_scan_has_visible_entry_and_params() {
        let ptx = selective_scan_ptx(80);
        check_visible_entry(&ptx, 80, "selective_scan");
        assert!(ptx.contains("p_u"), "must have p_u param");
        assert!(ptx.contains("p_a_bar"), "must have p_a_bar param");
        assert!(ptx.contains("p_b_bar"), "must have p_b_bar param");
        assert!(ptx.contains("p_c"), "must have p_c param");
        assert!(ptx.contains("p_out"), "must have p_out param");
        assert!(ptx.contains("seq_len"), "must have seq_len param");
        assert!(ptx.contains("d_model"), "must have d_model param");
    }

    #[test]
    fn selective_scan_uses_fma_rn_f32() {
        let ptx = selective_scan_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "S6 scan must use fma");
    }

    #[test]
    fn selective_scan_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = selective_scan_ptx(sm);
            check_visible_entry(&ptx, sm, "selective_scan");
        }
    }

    // ── parallel_scan ─────────────────────────────────────────────────────────

    #[test]
    fn parallel_scan_version_and_target_sm80() {
        let ptx = parallel_scan_ptx(80);
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn parallel_scan_version_and_target_sm90() {
        let ptx = parallel_scan_ptx(90);
        assert!(ptx.contains(".version 8.4"));
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn parallel_scan_has_visible_entry() {
        let ptx = parallel_scan_ptx(80);
        check_visible_entry(&ptx, 80, "parallel_scan");
    }

    #[test]
    fn parallel_scan_has_shfl_sync_down() {
        let ptx = parallel_scan_ptx(80);
        assert!(
            ptx.contains("shfl.sync.down.b32"),
            "warp prefix scan must use shfl.sync.down.b32"
        );
    }

    #[test]
    fn parallel_scan_uses_fma_and_mul() {
        let ptx = parallel_scan_ptx(80);
        assert!(
            ptx.contains("fma.rn.f32"),
            "must use fma.rn.f32 for b combine"
        );
        assert!(ptx.contains("mul.f32"), "must use mul.f32 for a combine");
    }

    #[test]
    fn parallel_scan_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = parallel_scan_ptx(sm);
            check_visible_entry(&ptx, sm, "parallel_scan");
        }
    }

    // ── depthwise_conv1d ──────────────────────────────────────────────────────

    #[test]
    fn depthwise_conv1d_version_and_target_sm75() {
        let ptx = depthwise_conv1d_ptx(75);
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_75"));
    }

    #[test]
    fn depthwise_conv1d_version_and_target_sm120() {
        let ptx = depthwise_conv1d_ptx(120);
        assert!(ptx.contains(".version 8.7"));
        assert!(ptx.contains(".target sm_120"));
    }

    #[test]
    fn depthwise_conv1d_has_visible_entry_and_params() {
        let ptx = depthwise_conv1d_ptx(80);
        check_visible_entry(&ptx, 80, "depthwise_conv1d");
        assert!(ptx.contains("kernel_size"), "must have kernel_size param");
        assert!(ptx.contains("seq_len"), "must have seq_len param");
        assert!(ptx.contains("channels"), "must have channels param");
    }

    #[test]
    fn depthwise_conv1d_uses_fma_rn_f32() {
        let ptx = depthwise_conv1d_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "causal conv must use fma");
    }

    #[test]
    fn depthwise_conv1d_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = depthwise_conv1d_ptx(sm);
            check_visible_entry(&ptx, sm, "depthwise_conv1d");
        }
    }

    // ── wkv_forward ───────────────────────────────────────────────────────────

    #[test]
    fn wkv_forward_version_and_target_sm80() {
        let ptx = wkv_forward_ptx(80);
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn wkv_forward_version_and_target_sm100() {
        let ptx = wkv_forward_ptx(100);
        assert!(ptx.contains(".version 8.7"));
        assert!(ptx.contains(".target sm_100"));
    }

    #[test]
    fn wkv_forward_has_visible_entry_and_params() {
        let ptx = wkv_forward_ptx(80);
        check_visible_entry(&ptx, 80, "wkv_forward");
        assert!(ptx.contains("p_k"), "must have p_k param");
        assert!(ptx.contains("p_v"), "must have p_v param");
        assert!(ptx.contains("seq_len"), "must have seq_len param");
        assert!(ptx.contains("channels"), "must have channels param");
    }

    #[test]
    fn wkv_forward_uses_ex2_lg2_rcp_fma() {
        let ptx = wkv_forward_ptx(80);
        assert!(ptx.contains("ex2.approx.f32"), "WKV must use ex2 for exp");
        assert!(
            ptx.contains("rcp.approx.f32"),
            "WKV must use rcp for division"
        );
        assert!(ptx.contains("fma.rn.f32"), "WKV must use fma");
    }

    #[test]
    fn wkv_forward_contains_max_for_stability() {
        let ptx = wkv_forward_ptx(80);
        assert!(
            ptx.contains("max.f32"),
            "WKV running-max trick needs max.f32"
        );
    }

    #[test]
    fn wkv_forward_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = wkv_forward_ptx(sm);
            check_visible_entry(&ptx, sm, "wkv_forward");
        }
    }

    // ── ssd_chunk ─────────────────────────────────────────────────────────────

    #[test]
    fn ssd_chunk_version_and_target_sm86() {
        let ptx = ssd_chunk_ptx(86);
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_86"));
    }

    #[test]
    fn ssd_chunk_version_and_target_sm120() {
        let ptx = ssd_chunk_ptx(120);
        assert!(ptx.contains(".version 8.7"));
        assert!(ptx.contains(".target sm_120"));
    }

    #[test]
    fn ssd_chunk_has_visible_entry_and_params() {
        let ptx = ssd_chunk_ptx(80);
        check_visible_entry(&ptx, 80, "ssd_chunk");
        assert!(ptx.contains("chunk_len"), "must have chunk_len param");
        assert!(ptx.contains("state_dim"), "must have state_dim param");
        assert!(ptx.contains("p_a"), "must have p_a param");
        assert!(ptx.contains("p_b_vec"), "must have p_b_vec param");
        assert!(ptx.contains("p_c_vec"), "must have p_c_vec param");
    }

    #[test]
    fn ssd_chunk_uses_fma_rn_f32() {
        let ptx = ssd_chunk_ptx(80);
        assert!(ptx.contains("fma.rn.f32"), "SSD chunk must use fma");
    }

    #[test]
    fn ssd_chunk_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = ssd_chunk_ptx(sm);
            check_visible_entry(&ptx, sm, "ssd_chunk");
        }
    }

    // ── hippo_legendre ────────────────────────────────────────────────────────

    #[test]
    fn hippo_legendre_version_and_target_sm75() {
        let ptx = hippo_legendre_ptx(75);
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_75"));
    }

    #[test]
    fn hippo_legendre_version_and_target_sm90() {
        let ptx = hippo_legendre_ptx(90);
        assert!(ptx.contains(".version 8.4"));
        assert!(ptx.contains(".target sm_90"));
    }

    #[test]
    fn hippo_legendre_has_visible_entry_and_params() {
        let ptx = hippo_legendre_ptx(80);
        check_visible_entry(&ptx, 80, "hippo_legendre");
        assert!(ptx.contains("u_val"), "must have u_val param");
        assert!(ptx.contains("delta"), "must have delta param");
        assert!(ptx.contains("n_coeffs"), "must have n_coeffs param");
    }

    #[test]
    fn hippo_legendre_uses_sqrt_approx_and_fma() {
        let ptx = hippo_legendre_ptx(80);
        assert!(
            ptx.contains("sqrt.approx.f32"),
            "HiPPO-LegS must use sqrt.approx.f32 for sqrt(2n+1)"
        );
        assert!(ptx.contains("fma.rn.f32"), "HiPPO-LegS must use fma");
    }

    #[test]
    fn hippo_legendre_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = hippo_legendre_ptx(sm);
            check_visible_entry(&ptx, sm, "hippo_legendre");
        }
    }

    // ── rms_norm_silu ─────────────────────────────────────────────────────────

    #[test]
    fn rms_norm_silu_version_and_target_sm80() {
        let ptx = rms_norm_silu_ptx(80);
        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn rms_norm_silu_version_and_target_sm120() {
        let ptx = rms_norm_silu_ptx(120);
        assert!(ptx.contains(".version 8.7"));
        assert!(ptx.contains(".target sm_120"));
    }

    #[test]
    fn rms_norm_silu_has_visible_entry_and_params() {
        let ptx = rms_norm_silu_ptx(80);
        check_visible_entry(&ptx, 80, "rms_norm_silu");
        assert!(ptx.contains("p_x"), "must have p_x param");
        assert!(ptx.contains("p_g"), "must have p_g param");
        assert!(ptx.contains("p_z"), "must have p_z param");
        assert!(ptx.contains("p_out"), "must have p_out param");
        assert!(ptx.contains("eps"), "must have eps param");
    }

    #[test]
    fn rms_norm_silu_uses_sqrt_rcp_ex2_shfl_bfly() {
        let ptx = rms_norm_silu_ptx(80);
        assert!(
            ptx.contains("sqrt.approx.f32"),
            "RMSNorm must use sqrt.approx.f32"
        );
        assert!(
            ptx.contains("rcp.approx.f32"),
            "RMSNorm must use rcp.approx.f32"
        );
        assert!(
            ptx.contains("ex2.approx.f32"),
            "SiLU must use ex2.approx.f32 for exp"
        );
        assert!(
            ptx.contains("shfl.sync.bfly.b32"),
            "warp sum must use shfl.sync.bfly.b32"
        );
    }

    #[test]
    fn rms_norm_silu_all_sm_versions() {
        for &sm in SM_VERSIONS {
            let ptx = rms_norm_silu_ptx(sm);
            check_visible_entry(&ptx, sm, "rms_norm_silu");
        }
    }

    // ── Cross-kernel sanity: all 7 kernels produce non-empty PTX ─────────────

    #[test]
    fn all_kernels_nonempty_for_all_sm_versions() {
        for &sm in SM_VERSIONS {
            assert!(!selective_scan_ptx(sm).is_empty());
            assert!(!parallel_scan_ptx(sm).is_empty());
            assert!(!depthwise_conv1d_ptx(sm).is_empty());
            assert!(!wkv_forward_ptx(sm).is_empty());
            assert!(!ssd_chunk_ptx(sm).is_empty());
            assert!(!hippo_legendre_ptx(sm).is_empty());
            assert!(!rms_norm_silu_ptx(sm).is_empty());
        }
    }

    #[test]
    fn all_kernels_contain_ret_instruction() {
        let sm = 80;
        for ptx in [
            selective_scan_ptx(sm),
            parallel_scan_ptx(sm),
            depthwise_conv1d_ptx(sm),
            wkv_forward_ptx(sm),
            ssd_chunk_ptx(sm),
            hippo_legendre_ptx(sm),
            rms_norm_silu_ptx(sm),
        ] {
            assert!(ptx.contains("ret;"), "every kernel must end with ret;");
        }
    }

    #[test]
    fn all_kernels_use_mul_wide_u32_for_byte_offsets() {
        let sm = 80;
        for ptx in [
            selective_scan_ptx(sm),
            parallel_scan_ptx(sm),
            depthwise_conv1d_ptx(sm),
            wkv_forward_ptx(sm),
            ssd_chunk_ptx(sm),
            hippo_legendre_ptx(sm),
            rms_norm_silu_ptx(sm),
        ] {
            assert!(
                ptx.contains("mul.wide.u32"),
                "must use mul.wide.u32 for byte offset computation"
            );
        }
    }

    #[test]
    fn all_kernels_use_grid_stride_loop() {
        let sm = 80;
        // Grid-stride loops use nctaid.x to compute the stride
        for ptx in [
            selective_scan_ptx(sm),
            parallel_scan_ptx(sm),
            depthwise_conv1d_ptx(sm),
            wkv_forward_ptx(sm),
            ssd_chunk_ptx(sm),
            hippo_legendre_ptx(sm),
            rms_norm_silu_ptx(sm),
        ] {
            assert!(
                ptx.contains("%nctaid.x"),
                "must reference %nctaid.x for grid-stride stride computation"
            );
        }
    }

    #[test]
    fn f32_constants_use_hex_not_bare_float() {
        // Kernels that inject literal float constants must use f32_hex encoding.
        // The "0F" prefix is the PTX hexadecimal float notation.
        // parallel_scan_ptx operates purely on memory-loaded values and injects
        // no literal float constants — it is excluded from this check.
        let sm = 80;
        for (name, ptx) in [
            ("selective_scan", selective_scan_ptx(sm)),
            ("depthwise_conv1d", depthwise_conv1d_ptx(sm)),
            ("wkv_forward", wkv_forward_ptx(sm)),
            ("ssd_chunk", ssd_chunk_ptx(sm)),
            ("hippo_legendre", hippo_legendre_ptx(sm)),
            ("rms_norm_silu", rms_norm_silu_ptx(sm)),
        ] {
            // PTX hex floats start with "0F" followed by 8 hex digits
            assert!(
                ptx.contains("0F"),
                "kernel '{name}': float constants must be hex-encoded (0F...)"
            );
        }
    }
}
