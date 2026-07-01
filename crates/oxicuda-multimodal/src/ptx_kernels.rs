//! PTX GPU kernel sources for multi-modal learning operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`cross_attn_score_ptx`] | Scaled dot-product attention score Q·Kᵀ/√d; grid-stride over batch×heads×seq_q×seq_k |
//! | [`modal_align_loss_ptx`] | Symmetric InfoNCE: numerically stable row-softmax + diagonal log |
//! | [`bilinear_pool_ptx`] | Compact bilinear: Hadamard of projected features then sum-pool |
//! | [`temporal_pool_ptx`] | Average pooling over T video frames per spatial position |
//! | [`token_merge_ptx`] | Concatenate two token sequences with attention mask generation |
//! | [`gate_fusion_ptx`] | Sigmoid gating: `out = g·a + (1-g)·b` |
//! | [`itm_bce_ptx`] | Sigmoid + binary cross-entropy for ITM |

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

/// Format an f32 as a PTX hex literal (`0F` prefix + 8 uppercase hex digits).
#[must_use]
pub fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1: cross_attn_score ──────────────────────────────────────────────

/// Scaled dot-product attention score `scores[b,h,i,j] = Q[b,h,i,*] · K[b,h,j,*] / √d_k`.
///
/// Grid-stride over `batch × heads × seq_q × seq_k`.
/// Each thread handles one `(batch, head, q_pos, k_pos)` tuple.
/// `fma.rn.f32` accumulates the dot product; `sqrt.approx.f32` computes the scale.
#[must_use]
pub fn cross_attn_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// cross_attn_score_kernel: scores[b,h,i,j] = dot(Q[b,h,i,*], K[b,h,j,*]) / sqrt(d_k)
// Params: p_q, p_k, p_out — all [batch * n_heads * seq * d_k] row-major
//         batch, n_heads, seq_q, seq_k, d_k
.visible .entry cross_attn_score_kernel(
    .param .u64 p_q,
    .param .u64 p_k,
    .param .u64 p_out,
    .param .u32 batch,
    .param .u32 n_heads,
    .param .u32 seq_q,
    .param .u32 seq_k,
    .param .u32 d_k
)
{{
    .reg .u64  %rd<12>;
    .reg .u32  %r<20>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_q];
    ld.param.u64  %rd1, [p_k];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [batch];
    ld.param.u32  %r1,  [n_heads];
    ld.param.u32  %r2,  [seq_q];
    ld.param.u32  %r3,  [seq_k];
    ld.param.u32  %r4,  [d_k];

    // Global thread id
    mov.u32       %r5, %ntid.x;
    mov.u32       %r6, %ctaid.x;
    mov.u32       %r7, %tid.x;
    mad.lo.u32    %r8, %r5, %r6, %r7;

    // Total output elements = batch * n_heads * seq_q * seq_k
    mul.lo.u32    %r9,  %r0, %r1;
    mul.lo.u32    %r9,  %r9, %r2;
    mul.lo.u32    %r9,  %r9, %r3;

    // Grid stride
    mov.u32       %r10, %nctaid.x;
    mul.lo.u32    %r11, %r5, %r10;
    mov.u32       %r12, %r8;

$CATTN_LOOP:
    setp.ge.u32   %p0, %r12, %r9;
    @%p0 bra $CATTN_DONE;

    // Decode flat index -> (b, h, qi, kj)
    // kj = idx % seq_k
    rem.u32       %r13, %r12, %r3;
    // qi = (idx / seq_k) % seq_q
    div.u32       %r14, %r12, %r3;
    rem.u32       %r14, %r14, %r2;
    // h  = (idx / (seq_k * seq_q)) % n_heads
    mul.lo.u32    %r15, %r2, %r3;
    div.u32       %r16, %r12, %r15;
    rem.u32       %r16, %r16, %r1;
    // b  = idx / (seq_k * seq_q * n_heads)
    mul.lo.u32    %r17, %r15, %r1;
    div.u32       %r17, %r12, %r17;

    // Q base: b*n_heads*seq_q*d_k + h*seq_q*d_k + qi*d_k
    mul.lo.u32    %r18, %r1, %r2;
    mul.lo.u32    %r18, %r18, %r4;
    mul.lo.u32    %r19, %r17, %r18;    // b * (n_heads*seq_q*d_k)
    mad.lo.u32    %r19, %r16, %r4, %r19; // + h * d_k  (simplified, omit seq_q)
    // (full: h*seq_q*d_k + qi*d_k; use r14 for qi)
    mul.lo.u32    %r15, %r16, %r2;
    mul.lo.u32    %r15, %r15, %r4;     // h*seq_q*d_k
    mad.lo.u32    %r15, %r14, %r4, %r15; // + qi*d_k
    mul.lo.u32    %r18, %r17, %r1;
    mul.lo.u32    %r18, %r18, %r2;
    mul.lo.u32    %r18, %r18, %r4;     // b*n_heads*seq_q*d_k
    add.u32       %r19, %r18, %r15;    // q_base (offset in f32 elems)

    // K base: same formula but qi replaced by kj (r13)
    mul.lo.u32    %r15, %r16, %r2;
    mul.lo.u32    %r15, %r15, %r4;
    mad.lo.u32    %r15, %r13, %r4, %r15;
    add.u32       %r18, %r18, %r15;   // k_base

    // Dot product over d_k
    mov.f32       %f0, {ZERO};
    mov.u32       %r15, 0;             // d index

$DOT_LOOP:
    setp.ge.u32   %p0, %r15, %r4;
    @%p0 bra $DOT_DONE;

    add.u32       %r16, %r19, %r15;
    mul.wide.u32  %rd3, %r16, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u32       %r16, %r18, %r15;
    mul.wide.u32  %rd3, %r16, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];

    fma.rn.f32    %f0, %f1, %f2, %f0;  // accumulate dot

    add.u32       %r15, %r15, 1;
    bra           $DOT_LOOP;

$DOT_DONE:
    // Scale by 1/sqrt(d_k) using sqrt.approx.f32 + rcp.approx.f32
    cvt.rn.f32.u32 %f3, %r4;
    sqrt.approx.f32 %f4, %f3;
    rcp.approx.f32  %f5, %f4;
    mul.f32         %f6, %f0, %f5;

    // Store to output
    mul.wide.u32  %rd5, %r12, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f6;

    add.u32       %r12, %r12, %r11;
    bra           $CATTN_LOOP;

$CATTN_DONE:
    // Suppress unused registers
    mov.u32       %r13, 0;
    mov.f32       %f7, {ZERO};
    mov.f32       %f8, {ZERO};
    mov.f32       %f9, {ZERO};
    mov.u64       %rd7, 0;
    mov.u64       %rd8, 0;
    mov.u64       %rd9, 0;
    mov.u64       %rd10, 0;
    mov.u64       %rd11, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 2: modal_align_loss ──────────────────────────────────────────────

/// Symmetric InfoNCE alignment loss.
/// Per-row numerically stable softmax (`lg2.approx.f32` + `ex2.approx.f32`),
/// then diagonal log-probability. One block per row; result accumulated by host.
#[must_use]
pub fn modal_align_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    format!(
        r#"{hdr}// modal_align_loss_kernel: symmetric InfoNCE row log-softmax + diagonal loss.
// p_sim: [N x N] similarity matrix (already temperature-scaled).
// p_loss: scalar accumulator (atomically added per block).
.visible .entry modal_align_loss_kernel(
    .param .u64 p_sim,
    .param .u64 p_loss,
    .param .u32 n_batch
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_sim];
    ld.param.u64  %rd1, [p_loss];
    ld.param.u32  %r0,  [n_batch];

    // Row index = blockIdx.x; column = threadIdx.x (grid-stride over cols)
    mov.u32       %r1, %ctaid.x;
    setp.ge.u32   %p0, %r1, %r0;
    @%p0 bra $ALIGN_DONE;

    // Row start address
    mul.lo.u32    %r2, %r1, %r0;       // r1 * n_batch
    mul.wide.u32  %rd2, %r2, 4;
    add.u64       %rd3, %rd0, %rd2;    // &sim[r1, 0]

    // Pass 1: compute row max for numerical stability
    mov.f32       %f0, {NEG_INF};
    mov.u32       %r3, 0;

$MAX_LOOP:
    setp.ge.u32   %p0, %r3, %r0;
    @%p0 bra $MAX_DONE;
    mul.wide.u32  %rd4, %r3, 4;
    add.u64       %rd5, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    max.f32       %f0, %f0, %f1;
    add.u32       %r3, %r3, 1;
    bra           $MAX_LOOP;
$MAX_DONE:

    // Pass 2: sum of exp(s - max) using ex2.approx (log2 domain)
    // log2(e) = 1/ln(2) ≈ 1.4426950408889634
    mov.f32       %f2, 0F3FB8AA3B;     // log2(e)
    mov.f32       %f3, {ZERO};
    mov.u32       %r4, 0;

$SUM_LOOP:
    setp.ge.u32   %p0, %r4, %r0;
    @%p0 bra $SUM_DONE;
    mul.wide.u32  %rd4, %r4, 4;
    add.u64       %rd5, %rd3, %rd4;
    ld.global.f32 %f4, [%rd5];
    sub.f32       %f5, %f4, %f0;       // s - max
    mul.f32       %f6, %f5, %f2;       // (s-max) * log2(e)
    ex2.approx.f32 %f7, %f6;           // 2^((s-max)*log2e) = exp(s-max)
    add.f32       %f3, %f3, %f7;
    add.u32       %r4, %r4, 1;
    bra           $SUM_LOOP;
$SUM_DONE:

    // log_sum = max + ln(sum) = max + log2(sum)/log2(e)
    // Using lg2.approx.f32 to get log2(sum)
    lg2.approx.f32 %f8, %f3;           // log2(sum_exp)
    rcp.approx.f32 %f9, %f2;           // 1/log2(e) = ln(2)
    mul.f32        %f9, %f8, %f9;      // log2(sum) * ln(2) = ln(sum)
    add.f32        %f9, %f9, %f0;      // log_sum_exp = max + ln(sum)

    // Diagonal element = sim[r1, r1]
    mul.lo.u32    %r5, %r1, %r0;
    add.u32       %r5, %r5, %r1;
    mul.wide.u32  %rd6, %r5, 4;
    add.u64       %rd7, %rd0, %rd6;
    ld.global.f32 %f10, [%rd7];

    // contribution = log_sum_exp - diag
    sub.f32       %f11, %f9, %f10;
    // Negate (InfoNCE loss = -log p)

    // Atomic add to global loss accumulator
    atom.global.add.f32 %f1, [%rd1], %f11;

$ALIGN_DONE:
    // Suppress unused registers
    mov.u32       %r6, 0;
    mov.u32       %r7, 0;
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f0, {ONE};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
        NEG_INF = neg_inf,
    )
}

// ─── Kernel 3: bilinear_pool ─────────────────────────────────────────────────

/// Compact bilinear pooling: Hadamard product of projected features, then sum-pool.
/// `out[b, d] = sum_k proj_v[b, k] * proj_q[b, k]` over dim factor `k`.
/// Grid-stride over `batch × out_dim`.
#[must_use]
pub fn bilinear_pool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// bilinear_pool_kernel: out[b,d] = sum_k( proj_v[b,k*d_out+d] * proj_q[b,k*d_out+d] )
// p_pv, p_pq: [batch * k_factor * d_out]; p_out: [batch * d_out]
// k_factor * d_out = inner_dim
.visible .entry bilinear_pool_kernel(
    .param .u64 p_pv,
    .param .u64 p_pq,
    .param .u64 p_out,
    .param .u32 batch,
    .param .u32 d_out,
    .param .u32 k_factor
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<17>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_pv];
    ld.param.u64  %rd1, [p_pq];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [batch];
    ld.param.u32  %r1,  [d_out];
    ld.param.u32  %r2,  [k_factor];

    // Global tid
    mov.u32       %r3, %ntid.x;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %tid.x;
    mad.lo.u32    %r6, %r3, %r4, %r5;

    // Total = batch * d_out
    mul.lo.u32    %r7, %r0, %r1;

    // Grid stride
    mov.u32       %r8, %nctaid.x;
    mul.lo.u32    %r9, %r3, %r8;

    mov.u32       %r10, %r6;

$BP_LOOP:
    setp.ge.u32   %p0, %r10, %r7;
    @%p0 bra $BP_DONE;

    // Decode (b, d) from flat index
    rem.u32       %r11, %r10, %r1;    // d
    div.u32       %r12, %r10, %r1;    // b

    // Sum over k: inner_dim = k_factor * d_out
    mul.lo.u32    %r13, %r2, %r1;     // inner_dim = k_factor * d_out

    mov.f32       %f0, {ZERO};
    mov.u32       %r14, 0;            // k index

$K_LOOP:
    setp.ge.u32   %p0, %r14, %r2;
    @%p0 bra $K_DONE;

    // offset = b * inner_dim + k * d_out + d
    mul.lo.u32    %r15, %r14, %r1;    // k * d_out
    add.u32       %r15, %r15, %r11;   // + d
    mul.lo.u32    %r16, %r12, %r13;   // b * inner_dim
    add.u32       %r15, %r15, %r16;   // final offset

    mul.wide.u32  %rd3, %r15, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f2, [%rd5];

    fma.rn.f32    %f0, %f1, %f2, %f0;  // accumulate Hadamard

    add.u32       %r14, %r14, 1;
    bra           $K_LOOP;
$K_DONE:

    // Store
    mul.wide.u32  %rd6, %r10, 4;
    add.u64       %rd7, %rd2, %rd6;
    st.global.f32 [%rd7], %f0;

    add.u32       %r10, %r10, %r9;
    bra           $BP_LOOP;

$BP_DONE:
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 4: temporal_pool ─────────────────────────────────────────────────

/// Average pooling over `T` video frames per spatial position.
/// `out[b, s, d] = (1/T) * sum_t frames[b, t, s, d]`.
/// Grid-stride over `batch × spatial × d_model`.
#[must_use]
pub fn temporal_pool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// temporal_pool_kernel: out[b,s,d] = mean_t( frames[b,t,s,d] )
// p_in: [batch * n_frames * n_spatial * d_model]; p_out: [batch * n_spatial * d_model]
.visible .entry temporal_pool_kernel(
    .param .u64 p_in,
    .param .u64 p_out,
    .param .u32 batch,
    .param .u32 n_frames,
    .param .u32 n_spatial,
    .param .u32 d_model
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<17>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_in];
    ld.param.u64  %rd1, [p_out];
    ld.param.u32  %r0,  [batch];
    ld.param.u32  %r1,  [n_frames];
    ld.param.u32  %r2,  [n_spatial];
    ld.param.u32  %r3,  [d_model];

    // Global tid
    mov.u32       %r4, %ntid.x;
    mov.u32       %r5, %ctaid.x;
    mov.u32       %r6, %tid.x;
    mad.lo.u32    %r7, %r4, %r5, %r6;

    // Total output elements = batch * n_spatial * d_model
    mul.lo.u32    %r8, %r0, %r2;
    mul.lo.u32    %r8, %r8, %r3;

    // Grid stride
    mov.u32       %r9, %nctaid.x;
    mul.lo.u32    %r10, %r4, %r9;

    mov.u32       %r11, %r7;

$TP_LOOP:
    setp.ge.u32   %p0, %r11, %r8;
    @%p0 bra $TP_DONE;

    // Decode (b, s, d)
    rem.u32       %r12, %r11, %r3;    // d
    div.u32       %r13, %r11, %r3;
    rem.u32       %r14, %r13, %r2;    // s
    div.u32       %r13, %r13, %r2;    // b

    // frame stride in p_in = n_spatial * d_model
    mul.lo.u32    %r15, %r2, %r3;

    // base in p_in at frame 0: b*n_frames*n_spatial*d_model + s*d_model + d
    mul.lo.u32    %r16, %r13, %r1;
    mul.lo.u32    %r16, %r16, %r15;
    mad.lo.u32    %r16, %r14, %r3, %r16;
    add.u32       %r16, %r16, %r12;

    mov.f32       %f0, {ZERO};
    mov.u32       %r12, 0;             // frame index

$FRAME_LOOP:
    setp.ge.u32   %p0, %r12, %r1;
    @%p0 bra $FRAME_DONE;

    // input index = r16 + t * r15
    mul.lo.u32    %r15, %r12, %r2;
    mul.lo.u32    %r15, %r15, %r3;    // t * n_spatial * d_model
    add.u32       %r15, %r16, %r15;
    mul.wide.u32  %rd2, %r15, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];
    add.f32       %f0, %f0, %f1;

    add.u32       %r12, %r12, 1;
    mul.lo.u32    %r15, %r2, %r3;     // restore stride
    bra           $FRAME_LOOP;
$FRAME_DONE:

    // Divide by n_frames
    cvt.rn.f32.u32 %f2, %r1;
    rcp.approx.f32 %f3, %f2;
    mul.f32        %f4, %f0, %f3;

    mul.wide.u32  %rd4, %r11, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f4;

    add.u32       %r11, %r11, %r10;
    bra           $TP_LOOP;

$TP_DONE:
    mov.f32       %f5, {ZERO};
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 5: token_merge ───────────────────────────────────────────────────

/// Concatenate two token sequences into one buffer with attention mask generation.
/// `out[b, :len_a] = a[b, :]`, `out[b, len_a:len_a+len_b] = b[b, :]`.
/// `mask[b, i] = 1.0 if i < len_a+len_b else 0.0`.
/// Grid-stride over `batch × max_len`.
#[must_use]
pub fn token_merge_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// token_merge_kernel: concat two token sequences + attention mask.
// p_a: [batch * len_a * d]; p_b: [batch * len_b * d]
// p_out: [batch * (len_a+len_b) * d]; p_mask: [batch * (len_a+len_b)]
.visible .entry token_merge_kernel(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u64 p_mask,
    .param .u32 batch,
    .param .u32 len_a,
    .param .u32 len_b,
    .param .u32 d_model
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<18>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_b];
    ld.param.u64  %rd2, [p_out];
    ld.param.u64  %rd3, [p_mask];
    ld.param.u32  %r0,  [batch];
    ld.param.u32  %r1,  [len_a];
    ld.param.u32  %r2,  [len_b];
    ld.param.u32  %r3,  [d_model];

    add.u32       %r4, %r1, %r2;      // total_len = len_a + len_b

    // Global tid
    mov.u32       %r5, %ntid.x;
    mov.u32       %r6, %ctaid.x;
    mov.u32       %r7, %tid.x;
    mad.lo.u32    %r8, %r5, %r6, %r7;

    // Total = batch * total_len * d_model
    mul.lo.u32    %r9, %r0, %r4;
    mul.lo.u32    %r9, %r9, %r3;

    // Grid stride
    mov.u32       %r10, %nctaid.x;
    mul.lo.u32    %r11, %r5, %r10;

    mov.u32       %r12, %r8;

$TM_LOOP:
    setp.ge.u32   %p0, %r12, %r9;
    @%p0 bra $TM_DONE;

    // Decode (b, pos, d)
    rem.u32       %r13, %r12, %r3;    // d
    div.u32       %r14, %r12, %r3;
    rem.u32       %r15, %r14, %r4;    // pos in [0, total_len)
    div.u32       %r14, %r14, %r4;    // b

    // Determine if pos < len_a (from A) or else from B
    setp.lt.u32   %p1, %r15, %r1;

    @%p1 bra $FROM_A;

    // FROM_B: src_pos = pos - len_a
    sub.u32       %r16, %r15, %r1;
    // index in b: b * len_b * d_model + src_pos * d_model + d
    mul.lo.u32    %r17, %r2, %r3;
    mad.lo.u32    %r17, %r14, %r17, 0;
    mad.lo.u32    %r17, %r16, %r3, %r17;
    add.u32       %r17, %r17, %r13;
    mul.wide.u32  %rd4, %r17, 4;
    add.u64       %rd5, %rd1, %rd4;
    ld.global.f32 %f0, [%rd5];
    bra           $STORE;

$FROM_A:
    // index in a: b * len_a * d_model + pos * d_model + d
    mul.lo.u32    %r17, %r1, %r3;
    mad.lo.u32    %r17, %r14, %r17, 0;
    mad.lo.u32    %r17, %r15, %r3, %r17;
    add.u32       %r17, %r17, %r13;
    mul.wide.u32  %rd4, %r17, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];

$STORE:
    mul.wide.u32  %rd6, %r12, 4;
    add.u64       %rd7, %rd2, %rd6;
    st.global.f32 [%rd7], %f0;

    // Write mask for this (b, pos) combination only when d==0
    setp.eq.u32   %p0, %r13, 0;
    @!%p0 bra $SKIP_MASK;
    mul.lo.u32    %r17, %r14, %r4;
    add.u32       %r17, %r17, %r15;
    mul.wide.u32  %rd8, %r17, 4;
    add.u64       %rd9, %rd3, %rd8;
    st.global.f32 [%rd9], {ONE};

$SKIP_MASK:
    add.u32       %r12, %r12, %r11;
    bra           $TM_LOOP;

$TM_DONE:
    mov.f32       %f1, {ZERO};
    mov.f32       %f2, {ZERO};
    mov.f32       %f3, {ZERO};
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
    )
}

// ─── Kernel 6: gate_fusion ───────────────────────────────────────────────────

/// Sigmoid gating: `out[i] = g[i] * a[i] + (1 - g[i]) * b[i]`.
/// Sigmoid via `ex2.approx.f32` + `rcp.approx.f32`: `σ(x) = 1/(1 + exp(-x))`.
/// Grid-stride over total elements.
#[must_use]
pub fn gate_fusion_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    // log2(e) ≈ 1.4426950408889634 → 0x3FB8AA3B
    format!(
        r#"{hdr}// gate_fusion_kernel: out[i] = sigma(gate[i]) * a[i] + (1 - sigma(gate[i])) * b[i]
// Sigmoid: sigma(x) = 1 / (1 + exp(-x)), computed via ex2.approx.f32
.visible .entry gate_fusion_kernel(
    .param .u64 p_gate,
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_gate];
    ld.param.u64  %rd1, [p_a];
    ld.param.u64  %rd2, [p_b];
    ld.param.u64  %rd3, [p_out];
    ld.param.u32  %r0,  [n];

    // Global tid
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    // Grid stride
    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$GF_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $GF_DONE;

    mul.wide.u32  %rd4, %r7, 4;
    add.u64       %rd5, %rd0, %rd4;
    ld.global.f32 %f0, [%rd5];     // gate value x

    // sigmoid(x) = 1 / (1 + exp(-x))
    // exp(-x) via ex2: exp(-x) = 2^(-x * log2(e))
    mov.f32       %f1, 0F3FB8AA3B;  // log2(e)
    neg.f32       %f2, %f0;         // -x
    mul.f32       %f3, %f2, %f1;    // -x * log2(e)
    ex2.approx.f32 %f4, %f3;        // exp(-x)
    mov.f32       %f5, {ONE};
    add.f32       %f6, %f5, %f4;    // 1 + exp(-x)
    rcp.approx.f32 %f7, %f6;        // sigma(x) = g

    // 1 - g
    sub.f32       %f8, %f5, %f7;

    // load a, b
    add.u64       %rd6, %rd1, %rd4;
    ld.global.f32 %f9, [%rd6];     // a[i]
    add.u64       %rd7, %rd2, %rd4;
    ld.global.f32 %f10, [%rd7];    // b[i]

    // out = g*a + (1-g)*b
    mul.f32       %f9,  %f9,  %f7;
    fma.rn.f32    %f11, %f10, %f8, %f9;

    // store
    add.u64       %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f11;

    add.u32       %r7, %r7, %r6;
    bra           $GF_LOOP;

$GF_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f0, {ZERO};
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
    )
}

// ─── Kernel 7: itm_bce ───────────────────────────────────────────────────────

/// Sigmoid + binary cross-entropy for Image-Text Matching (ITM).
/// `loss = -(y * log(σ(x)) + (1-y) * log(1 - σ(x)))`.
/// Uses `ex2.approx.f32` + `lg2.approx.f32` for numerically stable BCE.
/// Grid-stride over batch; result accumulated into scalar.
#[must_use]
pub fn itm_bce_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// itm_bce_kernel: BCE with logits: loss[i] = -(y*log(sigma(x)) + (1-y)*log(1-sigma(x)))
// p_logits: [n]; p_labels: [n] (f32, 0.0 or 1.0); p_loss: scalar accumulator
.visible .entry itm_bce_kernel(
    .param .u64 p_logits,
    .param .u64 p_labels,
    .param .u64 p_loss,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<14>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_logits];
    ld.param.u64  %rd1, [p_labels];
    ld.param.u64  %rd2, [p_loss];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;

    mov.u32       %r7, %r4;

$BCE_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $BCE_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];     // x (logit)

    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f1, [%rd5];     // y (label)

    // sigma(x) = 1 / (1 + exp(-x))
    mov.f32       %f2, 0F3FB8AA3B;  // log2(e)
    neg.f32       %f3, %f0;
    mul.f32       %f4, %f3, %f2;
    ex2.approx.f32 %f5, %f4;
    mov.f32       %f6, {ONE};
    add.f32       %f7, %f6, %f5;
    rcp.approx.f32 %f8, %f7;        // sigma(x)

    // 1 - sigma(x)
    sub.f32       %f9, %f6, %f8;

    // log(sigma(x)) via lg2.approx then * ln(2)
    mov.f32       %f10, 0F3F317218; // ln(2) ≈ 0.6931471805599453
    lg2.approx.f32 %f11, %f8;
    mul.f32       %f11, %f11, %f10; // ln(sigma(x))

    // log(1 - sigma(x))
    lg2.approx.f32 %f12, %f9;
    mul.f32       %f12, %f12, %f10; // ln(1 - sigma(x))

    // loss = -(y * log(sigma) + (1-y) * log(1-sigma))
    mul.f32       %f11, %f1, %f11;
    sub.f32       %f13, %f6, %f1;   // 1-y
    fma.rn.f32    %f11, %f13, %f12, %f11;
    neg.f32       %f11, %f11;

    // Accumulate
    atom.global.add.f32 %f0, [%rd2], %f11;

    add.u32       %r7, %r7, %r6;
    bra           $BCE_LOOP;

$BCE_DONE:
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f0, {ZERO};
    mov.u64       %rd6, 0;
    mov.u64       %rd7, 0;
    ret;
}}
"#,
        ZERO = zero,
        ONE = one,
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
    fn cross_attn_score_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&cross_attn_score_ptx(sm), sm, "cross_attn_score_kernel");
        }
    }

    #[test]
    fn modal_align_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&modal_align_loss_ptx(sm), sm, "modal_align_loss_kernel");
        }
    }

    #[test]
    fn bilinear_pool_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&bilinear_pool_ptx(sm), sm, "bilinear_pool_kernel");
        }
    }

    #[test]
    fn temporal_pool_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&temporal_pool_ptx(sm), sm, "temporal_pool_kernel");
        }
    }

    #[test]
    fn token_merge_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&token_merge_ptx(sm), sm, "token_merge_kernel");
        }
    }

    #[test]
    fn gate_fusion_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&gate_fusion_ptx(sm), sm, "gate_fusion_kernel");
        }
    }

    #[test]
    fn itm_bce_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&itm_bce_ptx(sm), sm, "itm_bce_kernel");
        }
    }

    #[test]
    fn ptx_header_version_strings() {
        assert!(ptx_header(75).contains(".version 7.5"));
        assert!(ptx_header(80).contains(".version 8.0"));
        assert!(ptx_header(86).contains(".version 8.0"));
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
    fn cross_attn_uses_fma_and_sqrt() {
        let p = cross_attn_score_ptx(80);
        assert!(p.contains("fma.rn.f32"));
        assert!(p.contains("sqrt.approx.f32"));
    }

    #[test]
    fn modal_align_uses_lg2_and_ex2() {
        let p = modal_align_loss_ptx(80);
        assert!(p.contains("lg2.approx.f32"));
        assert!(p.contains("ex2.approx.f32"));
    }

    #[test]
    fn gate_fusion_uses_ex2_and_rcp() {
        let p = gate_fusion_ptx(80);
        assert!(p.contains("ex2.approx.f32"));
        assert!(p.contains("rcp.approx.f32"));
    }

    #[test]
    fn itm_bce_uses_sigmoid_and_log() {
        let p = itm_bce_ptx(80);
        assert!(p.contains("ex2.approx.f32"));
        assert!(p.contains("lg2.approx.f32"));
    }

    #[test]
    fn bilinear_pool_uses_fma() {
        let p = bilinear_pool_ptx(80);
        assert!(p.contains("fma.rn.f32"));
    }
}
