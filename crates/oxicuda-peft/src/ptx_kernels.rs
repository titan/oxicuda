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

/// Fused low-rank matmul: computes `B·(A·x)` where A∈ℝ^{r×n}, B∈ℝ^{m×r}.
///
/// Kernel signature: `lora_matmul_kernel(x, a, b, out, n, r, m)`
/// where `n`=in_dim, `r`=rank, `m`=out_dim.
/// Grid=(m/32+1,1,1) Block=(32,1,1).
#[must_use]
pub fn lora_matmul_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// lora_matmul_kernel: fused B*(A*x) low-rank matmul.
// x: [n] input vector
// a: [r*n] A matrix (row-major)
// b: [m*r] B matrix (row-major)
// out: [m] output vector
// n: in_dim, r: rank, m: out_dim
.visible .entry lora_matmul_kernel(
    .param .u64 p_x,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_r,
    .param .u32 p_m
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_a];
    ld.param.u64  %rd2, [p_b];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [p_n];
    ld.param.u32  %r1,  [p_r];
    ld.param.u32  %r2,  [p_m];

    // Global thread id = output row index
    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;   // tid_global = row of output

    setp.ge.u32   %p0, %r6, %r2;
    @%p0 bra $LORA_DONE;

    // For each rank dimension ri: compute tmp[ri] = A[ri,:] . x
    // Then out[row] = B[row,:] . tmp
    // Note: we compute B[row, :] dot (A . x) directly without storing tmp

    mov.f32       %f0, {ZERO};   // accumulator for out[row]
    mov.u32       %r7, 0;        // ri = 0

$LORA_RANK_LOOP:
    setp.ge.u32   %p0, %r7, %r1;
    @%p0 bra $LORA_RANK_DONE;

    // Compute tmp_ri = sum_j A[ri, j] * x[j]
    mov.f32       %f1, {ZERO};
    mov.u32       %r8, 0;

$LORA_INNER_A:
    setp.ge.u32   %p0, %r8, %r0;
    @%p0 bra $LORA_INNER_A_DONE;

    // A[ri, j] address = a + (ri*n + j) * 4
    mul.lo.u32    %r9, %r7, %r0;
    add.u32       %r9, %r9, %r8;
    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f2, [%rd5];       // A[ri, j]

    // x[j]
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f3, [%rd5];       // x[j]

    fma.rn.f32    %f1, %f2, %f3, %f1;
    add.u32       %r8, %r8, 1;
    bra $LORA_INNER_A;

$LORA_INNER_A_DONE:
    // Load B[row, ri] and accumulate
    mul.lo.u32    %r9, %r6, %r1;
    add.u32       %r9, %r9, %r7;
    mul.wide.u32  %rd4, %r9, 4;
    add.u64       %rd5, %rd2, %rd4;
    ld.global.f32 %f4, [%rd5];       // B[row, ri]

    fma.rn.f32    %f0, %f4, %f1, %f0;

    add.u32       %r7, %r7, 1;
    bra $LORA_RANK_LOOP;

$LORA_RANK_DONE:
    mul.wide.u32  %rd6, %r6, 4;
    add.u64       %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %f0;

$LORA_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

/// Element-wise IA³ scaling: `out[i] = x[i] * scale[i]`.
///
/// Kernel signature: `ia3_scale_kernel(x, scale, out, n)`.
#[must_use]
pub fn ia3_scale_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// ia3_scale_kernel: element-wise multiply x by scale vector.
// x:     [n] input vector
// scale: [n] learned scale
// out:   [n] output vector
// n:     length
.visible .entry ia3_scale_kernel(
    .param .u64 p_x,
    .param .u64 p_scale,
    .param .u64 p_out,
    .param .u32 p_n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_scale];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [p_n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$IA3_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $IA3_DONE;

    mul.wide.u32  %rd3, %r7, 4;

    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];   // x[tid]

    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];   // scale[tid]

    mul.f32       %f2, %f0, %f1;

    add.u64       %rd5, %rd2, %rd3;
    st.global.f32 [%rd5], %f2;

    add.u32       %r7, %r7, %r6;
    bra $IA3_LOOP;

$IA3_DONE:
    ret;
}}
"#
    )
}

/// Replicate a prefix tensor across the batch dimension.
///
/// Kernel signature: `prefix_expand_kernel(prefix, out, batch, seq, dim)`
/// where `seq` = num_virtual_tokens × num_heads × head_dim (flattened prefix length).
#[must_use]
pub fn prefix_expand_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// prefix_expand_kernel: tile prefix [seq, dim] into out [batch*seq, dim].
// prefix: [seq * dim] source prefix tensor
// out:    [batch * seq * dim] expanded output
// batch:  batch size
// seq:    virtual token count * heads * head_dim (row count in prefix)
// dim:    final feature dimension
.visible .entry prefix_expand_kernel(
    .param .u64 p_prefix,
    .param .u64 p_out,
    .param .u32 p_batch,
    .param .u32 p_seq,
    .param .u32 p_dim
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<16>;
    .reg .f32  %f<2>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_prefix];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [p_batch];
    ld.param.u32  %r1,  [p_seq];
    ld.param.u32  %r2,  [p_dim];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;   // global tid

    mov.u32       %r7, %nctaid.x;
    mul.lo.u32    %r8, %r3, %r7;

    // total output elements = batch * seq * dim
    mul.lo.u32    %r9, %r0, %r1;
    mul.lo.u32    %r9, %r9, %r2;

    mov.u32       %r10, %r6;

$PEXP_LOOP:
    setp.ge.u32   %p0, %r10, %r9;
    @%p0 bra $PEXP_DONE;

    // row = tid / dim,  col = tid % dim
    div.u32       %r11, %r10, %r2;   // row in expanded (0..batch*seq)
    rem.u32       %r12, %r10, %r2;   // col

    // src_row = row % seq  (tile over batch)
    rem.u32       %r13, %r11, %r1;

    // Load prefix[src_row * dim + col]
    mul.lo.u32    %r14, %r13, %r2;
    add.u32       %r14, %r14, %r12;
    mul.wide.u32  %rd2, %r14, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f0, [%rd3];

    // Store out[tid]
    mul.wide.u32  %rd4, %r10, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f0;

    add.u32       %r10, %r10, %r8;
    bra $PEXP_LOOP;

$PEXP_DONE:
    ret;
}}
"#
    )
}

/// Bottleneck adapter forward pass: `x → W_down → GELU → W_up + x`.
///
/// Kernel signature: `adapter_forward_kernel(x, w_down, w_up, b_down, b_up, out, n, bot, seq)`
/// where `n`=in_dim, `bot`=bottleneck_dim, `seq`=sequence length (token count).
#[must_use]
pub fn adapter_forward_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let gelu_c0 = f32_hex(0.7978845608_f32); // sqrt(2/pi)
    let gelu_c1 = f32_hex(0.044715_f32);
    format!(
        r#"{hdr}// adapter_forward_kernel: bottleneck FFN with GELU + residual.
// x:      [seq * n] input tokens (row-major)
// w_down: [bot * n] down-projection
// w_up:   [n * bot] up-projection
// b_down: [bot] down bias
// b_up:   [n] up bias
// out:    [seq * n] output
// n: in_dim, bot: bottleneck_dim, seq: sequence length
.visible .entry adapter_forward_kernel(
    .param .u64 p_x,
    .param .u64 p_wdown,
    .param .u64 p_wup,
    .param .u64 p_bdown,
    .param .u64 p_bup,
    .param .u64 p_out,
    .param .u32 p_n,
    .param .u32 p_bot,
    .param .u32 p_seq
)
{{
    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<16>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_x];
    ld.param.u64  %rd1, [p_wdown];
    ld.param.u64  %rd2, [p_wup];
    ld.param.u64  %rd3, [p_bdown];
    ld.param.u64  %rd4, [p_bup];
    ld.param.u64  %rd5, [p_out];
    ld.param.u32  %r0,  [p_n];
    ld.param.u32  %r1,  [p_bot];
    ld.param.u32  %r2,  [p_seq];

    // Each thread handles one output element: (tok, out_col)
    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;

    mul.lo.u32    %r7, %r2, %r0;   // total = seq * n
    setp.ge.u32   %p0, %r6, %r7;
    @%p0 bra $ADAP_DONE;

    div.u32       %r8, %r6, %r0;   // token index
    rem.u32       %r9, %r6, %r0;   // out_col

    // Load residual x[tok, out_col]
    mul.wide.u32  %rd6, %r6, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f0, [%rd7];     // residual

    // Down projection + bias: h[bi] = sum_j W_down[bi, j] * x[tok, j] + b_down[bi]
    // Then GELU
    // Up projection: out[tok, out_col] = sum_bi W_up[out_col, bi] * gelu(h[bi]) + b_up[out_col] + residual

    // Load b_up[out_col]
    mul.wide.u32  %rd8, %r9, 4;
    add.u64       %rd9, %rd4, %rd8;
    ld.global.f32 %f1, [%rd9];     // b_up[out_col]

    mov.f32       %f2, %f1;        // accumulator starts with b_up
    mov.u32       %r10, 0;         // bi = 0

$ADAP_BOT_LOOP:
    setp.ge.u32   %p0, %r10, %r1;
    @%p0 bra $ADAP_BOT_DONE;

    // Down: h_bi = b_down[bi] + sum_j w_down[bi,j]*x[tok,j]
    mul.wide.u32  %rd10, %r10, 4;
    add.u64       %rd11, %rd3, %rd10;
    ld.global.f32 %f3, [%rd11];    // b_down[bi]

    mov.u32       %r11, 0;
$ADAP_DOWN_INNER:
    setp.ge.u32   %p0, %r11, %r0;
    @%p0 bra $ADAP_DOWN_DONE;

    mul.lo.u32    %r12, %r10, %r0;
    add.u32       %r12, %r12, %r11;
    mul.wide.u32  %rd12, %r12, 4;
    add.u64       %rd13, %rd1, %rd12;
    ld.global.f32 %f4, [%rd13];    // w_down[bi, j]

    mul.lo.u32    %r12, %r8, %r0;
    add.u32       %r12, %r12, %r11;
    mul.wide.u32  %rd12, %r12, 4;
    add.u64       %rd13, %rd0, %rd12;
    ld.global.f32 %f5, [%rd13];    // x[tok, j]

    fma.rn.f32    %f3, %f4, %f5, %f3;
    add.u32       %r11, %r11, 1;
    bra $ADAP_DOWN_INNER;

$ADAP_DOWN_DONE:
    // GELU(h_bi): approx tanh-based GELU
    // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    mul.f32       %f6, %f3, %f3;
    mul.f32       %f6, %f6, %f3;              // x^3
    fma.rn.f32    %f6, %f6, {GELU_C1}, %f3;  // x + 0.044715*x^3
    mul.f32       %f6, %f6, {GELU_C0};        // sqrt(2/pi) * (...)
    // tanh approximation via ex2: tanh(x) = (e^2x - 1)/(e^2x + 1)
    add.f32       %f7, %f6, %f6;              // 2x
    mul.f32       %f7, %f7, 0F3FB8AA3B;       // * log2(e)
    ex2.approx.f32 %f8, %f7;                  // 2^(2x * log2e) = e^(2x)
    add.f32       %f9, %f8, 0F3F800000;       // e^2x + 1
    sub.f32       %f10, %f8, 0F3F800000;      // e^2x - 1
    div.rn.f32    %f11, %f10, %f9;            // tanh(...)
    add.f32       %f11, %f11, 0F3F800000;     // 1 + tanh(...)
    mul.f32       %f11, %f11, 0F3F000000;     // 0.5 * (1 + tanh(...))
    mul.f32       %f12, %f3, %f11;            // gelu(h_bi)

    // Up: accumulate W_up[out_col, bi] * gelu(h_bi)
    mul.lo.u32    %r12, %r9, %r1;
    add.u32       %r12, %r12, %r10;
    mul.wide.u32  %rd12, %r12, 4;
    add.u64       %rd13, %rd2, %rd12;
    ld.global.f32 %f13, [%rd13];    // w_up[out_col, bi]

    fma.rn.f32    %f2, %f13, %f12, %f2;

    add.u32       %r10, %r10, 1;
    bra $ADAP_BOT_LOOP;

$ADAP_BOT_DONE:
    // out = up_result + residual
    add.f32       %f2, %f2, %f0;

    mul.wide.u32  %rd14, %r6, 4;
    add.u64       %rd15, %rd5, %rd14;
    st.global.f32 [%rd15], %f2;

$ADAP_DONE:
    ret;
}}
"#,
        GELU_C0 = gelu_c0,
        GELU_C1 = gelu_c1
    )
}

/// NF4 dequantization: look up packed 4-bit codes in the NF4 table and scale by absmax.
///
/// Kernel signature: `nf4_dequant_kernel(codes, absmax, out, n_blocks, block_size)`
/// where `codes` contains packed nibbles (2 per byte).
#[must_use]
pub fn nf4_dequant_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    // NF4 table as hex immediates
    let nf4: [f32; 16] = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.3949468731880188,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ];
    format!(
        r#"{hdr}// nf4_dequant_kernel: unpack 4-bit NF4 codes and scale by absmax.
// codes:      [n_blocks * block_size / 2] packed nibbles (lo=even, hi=odd)
// absmax:     [n_blocks] per-block scale factors
// out:        [n_blocks * block_size] dequantized f32 values
// n_blocks:   number of quantization blocks
// block_size: elements per block (must be even)
.visible .entry nf4_dequant_kernel(
    .param .u64 p_codes,
    .param .u64 p_absmax,
    .param .u64 p_out,
    .param .u32 p_nblocks,
    .param .u32 p_bsize
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<20>;
    .reg .f32  %f<8>;
    .reg .pred %p0;
    // NF4 lookup table in registers (via setp/selp chain)
    .reg .f32  %nf<16>;

    ld.param.u64  %rd0, [p_codes];
    ld.param.u64  %rd1, [p_absmax];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [p_nblocks];
    ld.param.u32  %r1,  [p_bsize];

    // Initialize NF4 table entries in registers
    mov.f32       %nf0,  {NF0};
    mov.f32       %nf1,  {NF1};
    mov.f32       %nf2,  {NF2};
    mov.f32       %nf3,  {NF3};
    mov.f32       %nf4,  {NF4};
    mov.f32       %nf5,  {NF5};
    mov.f32       %nf6,  {NF6};
    mov.f32       %nf7,  {NF7};
    mov.f32       %nf8,  {NF8};
    mov.f32       %nf9,  {NF9};
    mov.f32       %nf10, {NF10};
    mov.f32       %nf11, {NF11};
    mov.f32       %nf12, {NF12};
    mov.f32       %nf13, {NF13};
    mov.f32       %nf14, {NF14};
    mov.f32       %nf15, {NF15};

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;   // global tid = byte index

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;         // stride

    // Total bytes = n_blocks * block_size / 2
    mul.lo.u32    %r8, %r0, %r1;
    shr.u32       %r8, %r8, 1;

    mov.u32       %r9, %r5;

$NF4_LOOP:
    setp.ge.u32   %p0, %r9, %r8;
    @%p0 bra $NF4_DONE;

    // Load one byte (contains 2 nibbles)
    mul.wide.u32  %rd3, %r9, 1;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.u8  %r10, [%rd4];

    // lo nibble (even element index)
    and.b32       %r11, %r10, 0xF;
    // hi nibble (odd element index)
    shr.u32       %r12, %r10, 4;

    // Element indices in the output
    shl.b32       %r13, %r9, 1;   // elem0 = byte_idx * 2
    add.u32       %r14, %r13, 1;  // elem1 = byte_idx * 2 + 1

    // Block index and absmax
    div.u32       %r15, %r13, %r1;
    mul.wide.u32  %rd5, %r15, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.f32 %f0, [%rd6];    // absmax[block]

    // Table lookup for lo nibble via selp chain
    mov.f32       %f1, %nf0;
    setp.eq.u32   %p0, %r11, 1;  @%p0 mov.f32 %f1, %nf1;
    setp.eq.u32   %p0, %r11, 2;  @%p0 mov.f32 %f1, %nf2;
    setp.eq.u32   %p0, %r11, 3;  @%p0 mov.f32 %f1, %nf3;
    setp.eq.u32   %p0, %r11, 4;  @%p0 mov.f32 %f1, %nf4;
    setp.eq.u32   %p0, %r11, 5;  @%p0 mov.f32 %f1, %nf5;
    setp.eq.u32   %p0, %r11, 6;  @%p0 mov.f32 %f1, %nf6;
    setp.eq.u32   %p0, %r11, 7;  @%p0 mov.f32 %f1, %nf7;
    setp.eq.u32   %p0, %r11, 8;  @%p0 mov.f32 %f1, %nf8;
    setp.eq.u32   %p0, %r11, 9;  @%p0 mov.f32 %f1, %nf9;
    setp.eq.u32   %p0, %r11, 10; @%p0 mov.f32 %f1, %nf10;
    setp.eq.u32   %p0, %r11, 11; @%p0 mov.f32 %f1, %nf11;
    setp.eq.u32   %p0, %r11, 12; @%p0 mov.f32 %f1, %nf12;
    setp.eq.u32   %p0, %r11, 13; @%p0 mov.f32 %f1, %nf13;
    setp.eq.u32   %p0, %r11, 14; @%p0 mov.f32 %f1, %nf14;
    setp.eq.u32   %p0, %r11, 15; @%p0 mov.f32 %f1, %nf15;
    mul.f32       %f1, %f1, %f0;

    // Table lookup for hi nibble
    mov.f32       %f2, %nf0;
    setp.eq.u32   %p0, %r12, 1;  @%p0 mov.f32 %f2, %nf1;
    setp.eq.u32   %p0, %r12, 2;  @%p0 mov.f32 %f2, %nf2;
    setp.eq.u32   %p0, %r12, 3;  @%p0 mov.f32 %f2, %nf3;
    setp.eq.u32   %p0, %r12, 4;  @%p0 mov.f32 %f2, %nf4;
    setp.eq.u32   %p0, %r12, 5;  @%p0 mov.f32 %f2, %nf5;
    setp.eq.u32   %p0, %r12, 6;  @%p0 mov.f32 %f2, %nf6;
    setp.eq.u32   %p0, %r12, 7;  @%p0 mov.f32 %f2, %nf7;
    setp.eq.u32   %p0, %r12, 8;  @%p0 mov.f32 %f2, %nf8;
    setp.eq.u32   %p0, %r12, 9;  @%p0 mov.f32 %f2, %nf9;
    setp.eq.u32   %p0, %r12, 10; @%p0 mov.f32 %f2, %nf10;
    setp.eq.u32   %p0, %r12, 11; @%p0 mov.f32 %f2, %nf11;
    setp.eq.u32   %p0, %r12, 12; @%p0 mov.f32 %f2, %nf12;
    setp.eq.u32   %p0, %r12, 13; @%p0 mov.f32 %f2, %nf13;
    setp.eq.u32   %p0, %r12, 14; @%p0 mov.f32 %f2, %nf14;
    setp.eq.u32   %p0, %r12, 15; @%p0 mov.f32 %f2, %nf15;
    mul.f32       %f2, %f2, %f0;

    // Store even element
    mul.wide.u32  %rd7, %r13, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f1;

    // Store odd element
    mul.wide.u32  %rd7, %r14, 4;
    add.u64       %rd8, %rd2, %rd7;
    st.global.f32 [%rd8], %f2;

    add.u32       %r9, %r9, %r7;
    bra $NF4_LOOP;

$NF4_DONE:
    ret;
}}
"#,
        NF0 = f32_hex(nf4[0]),
        NF1 = f32_hex(nf4[1]),
        NF2 = f32_hex(nf4[2]),
        NF3 = f32_hex(nf4[3]),
        NF4 = f32_hex(nf4[4]),
        NF5 = f32_hex(nf4[5]),
        NF6 = f32_hex(nf4[6]),
        NF7 = f32_hex(nf4[7]),
        NF8 = f32_hex(nf4[8]),
        NF9 = f32_hex(nf4[9]),
        NF10 = f32_hex(nf4[10]),
        NF11 = f32_hex(nf4[11]),
        NF12 = f32_hex(nf4[12]),
        NF13 = f32_hex(nf4[13]),
        NF14 = f32_hex(nf4[14]),
        NF15 = f32_hex(nf4[15])
    )
}

/// LoRA weight merge: `W += scale * B * A` (outer-product accumulate over rank).
///
/// Kernel signature: `lora_merge_kernel(w, a, b, scale, n, r, m)`
/// where n=in_dim, r=rank, m=out_dim; each thread handles one (row, col) element of the m×n update.
#[must_use]
pub fn lora_merge_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// lora_merge_kernel: W[row,col] += scale * sum_ri B[row,ri]*A[ri,col]
// w:     [m * n] weight matrix (updated in-place)
// a:     [r * n] A matrix
// b:     [m * r] B matrix
// scale: scalar LoRA scale factor (alpha/r)
// n: in_dim, r: rank, m: out_dim
.visible .entry lora_merge_kernel(
    .param .u64 p_w,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .f32 p_scale,
    .param .u32 p_n,
    .param .u32 p_r,
    .param .u32 p_m
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_w];
    ld.param.u64  %rd1, [p_a];
    ld.param.u64  %rd2, [p_b];
    ld.param.f32  %f0,  [p_scale];
    ld.param.u32  %r0,  [p_n];
    ld.param.u32  %r1,  [p_r];
    ld.param.u32  %r2,  [p_m];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;

    // Total elements = m * n
    mul.lo.u32    %r7, %r2, %r0;

    setp.ge.u32   %p0, %r6, %r7;
    @%p0 bra $MERGE_DONE;

    div.u32       %r8, %r6, %r0;   // row
    rem.u32       %r9, %r6, %r0;   // col

    // Compute delta = scale * sum_ri B[row,ri] * A[ri,col]
    mov.f32       %f1, {ZERO};
    mov.u32       %r10, 0;

$MERGE_RANK_LOOP:
    setp.ge.u32   %p0, %r10, %r1;
    @%p0 bra $MERGE_RANK_DONE;

    // B[row, ri]
    mul.lo.u32    %r11, %r8, %r1;
    add.u32       %r11, %r11, %r10;
    mul.wide.u32  %rd3, %r11, 4;
    add.u64       %rd4, %rd2, %rd3;
    ld.global.f32 %f2, [%rd4];

    // A[ri, col]
    mul.lo.u32    %r11, %r10, %r0;
    add.u32       %r11, %r11, %r9;
    mul.wide.u32  %rd3, %r11, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f3, [%rd4];

    fma.rn.f32    %f1, %f2, %f3, %f1;

    add.u32       %r10, %r10, 1;
    bra $MERGE_RANK_LOOP;

$MERGE_RANK_DONE:
    mul.f32       %f4, %f1, %f0;   // scale * delta

    // Load W[row, col] and add
    mul.wide.u32  %rd5, %r6, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f5, [%rd6];
    add.f32       %f5, %f5, %f4;
    st.global.f32 [%rd6], %f5;

$MERGE_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

/// Soft-prompt concatenation: `out = [prompt; seq]` along the token (row) dimension.
///
/// Kernel signature: `prompt_concat_kernel(prompt, seq, out, p, s, d)`
/// where p=num_prompt_tokens, s=seq_len, d=embed_dim.
#[must_use]
pub fn prompt_concat_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// prompt_concat_kernel: prepend soft-prompt to sequence embeddings.
// prompt: [p * d] soft prompt embeddings
// seq:    [s * d] input sequence embeddings
// out:    [(p + s) * d] concatenated output
// p: num_prompt_tokens, s: seq_len, d: embed_dim
.visible .entry prompt_concat_kernel(
    .param .u64 p_prompt,
    .param .u64 p_seq,
    .param .u64 p_out,
    .param .u32 p_p,
    .param .u32 p_s,
    .param .u32 p_d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<2>;
    .reg .pred %p0;
    .reg .pred %p1;

    ld.param.u64  %rd0, [p_prompt];
    ld.param.u64  %rd1, [p_seq];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [p_p];
    ld.param.u32  %r1,  [p_s];
    ld.param.u32  %r2,  [p_d];

    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;

    mov.u32       %r7, %nctaid.x;
    mul.lo.u32    %r8, %r3, %r7;

    add.u32       %r9, %r0, %r1;   // total_rows = p + s
    mul.lo.u32    %r10, %r9, %r2;  // total = (p+s)*d

    mov.u32       %r11, %r6;

$PCAT_LOOP:
    setp.ge.u32   %p0, %r11, %r10;
    @%p0 bra $PCAT_DONE;

    div.u32       %r12, %r11, %r2;  // row
    rem.u32       %r13, %r11, %r2;  // col

    // if row < p: load from prompt, else from seq
    setp.lt.u32   %p1, %r12, %r0;
    @%p1 bra $PCAT_FROM_PROMPT;

    // From sequence: seq_row = row - p
    sub.u32       %r14, %r12, %r0;
    mul.lo.u32    %r14, %r14, %r2;
    add.u32       %r14, %r14, %r13;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    bra $PCAT_STORE;

$PCAT_FROM_PROMPT:
    mul.lo.u32    %r14, %r12, %r2;
    add.u32       %r14, %r14, %r13;
    mul.wide.u32  %rd3, %r14, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];

$PCAT_STORE:
    mul.wide.u32  %rd5, %r11, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f0;

    add.u32       %r11, %r11, %r8;
    bra $PCAT_LOOP;

$PCAT_DONE:
    ret;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kernels_non_empty() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!lora_matmul_ptx(sm).is_empty());
            assert!(!ia3_scale_ptx(sm).is_empty());
            assert!(!prefix_expand_ptx(sm).is_empty());
            assert!(!adapter_forward_ptx(sm).is_empty());
            assert!(!nf4_dequant_ptx(sm).is_empty());
            assert!(!lora_merge_ptx(sm).is_empty());
            assert!(!prompt_concat_ptx(sm).is_empty());
        }
    }
}
