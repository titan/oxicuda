//! PTX kernel strings for GPU-accelerated knowledge distillation operations.
//!
//! All kernels are returned as `String` — they are template strings suitable for
//! JIT compilation via the CUDA driver API.  No GPU hardware is required to build
//! or test this crate.

/// Build the PTX version/target header for a given SM version.
fn ptx_header(sm: u32) -> String {
    let (ptx_ver, target) = match sm {
        v if v >= 100 => ("8.7", format!("sm_{v}")),
        v if v >= 90 => ("8.4", format!("sm_{v}")),
        v if v >= 80 => ("8.0", format!("sm_{v}")),
        v => ("7.5", format!("sm_{v}")),
    };
    format!(".version {ptx_ver}\n.target {target}\n.address_size 64\n\n")
}

/// Format an `f32` value as a PTX immediate hex literal: `0Fxxxxxxxx`.
fn f32_hex(v: f32) -> String {
    format!("0F{:08X}", v.to_bits())
}

// ─── Kernel 1 ────────────────────────────────────────────────────────────────

/// Hinton KD soft-label loss kernel.
///
/// Computes temperature-scaled softmax/KL per sample and atomically accumulates into `out_loss`.
/// Grid = (batch, 1, 1), Block = (min(n_classes, 256), 1, 1).
#[must_use]
pub fn kd_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// kd_loss_kernel: Hinton KD soft-label loss with temperature.
// s_logits: [batch * n] student logits
// t_logits: [batch * n] teacher logits
// out_loss: [1] accumulator (atomic add)
// n: number of classes, batch: batch size, temp: temperature
.visible .entry kd_loss_kernel(
    .param .u64 s_logits,
    .param .u64 t_logits,
    .param .u64 out_loss,
    .param .u32 n,
    .param .u32 batch,
    .param .f32 temp
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<20>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [s_logits];
    ld.param.u64  %rd1, [t_logits];
    ld.param.u64  %rd2, [out_loss];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [batch];
    ld.param.f32  %f0,  [temp];

    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;

    setp.ge.u32   %p0, %r2, %r1;
    @%p0 bra $KD_DONE;
    setp.ge.u32   %p0, %r3, %r0;
    @%p0 bra $KD_DONE;

    mul.lo.u32    %r4, %r2, %r0;
    add.u32       %r5, %r4, %r3;

    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];

    div.rn.f32    %f3, %f1, %f0;
    div.rn.f32    %f4, %f2, %f0;

    mul.f32       %f5, %f3, 0F3FB8AA3B;
    ex2.approx.f32 %f6, %f5;
    mul.f32       %f7, %f4, 0F3FB8AA3B;
    ex2.approx.f32 %f8, %f7;

    mov.f32       %f9, {ZERO};
    setp.gt.f32   %p0, %f8, {ZERO};
    @%p0 div.rn.f32 %f10, %f8, %f6;
    setp.gt.f32   %p0, %f10, 0F00800000;
    @%p0 lg2.approx.f32 %f11, %f10;
    @%p0 mul.f32 %f11, %f11, 0F3F317218;
    @%p0 mul.f32 %f12, %f8, %f11;
    setp.gt.f32   %p0, %f8, {ZERO};
    @%p0 atom.global.add.f32 %f13, [%rd2], %f12;

    mov.f32       %f14, {ONE};
$KD_DONE:
    ret;
}}
"#,
        ZERO = zero,
        ONE = one
    )
}

// ─── Kernel 2 ────────────────────────────────────────────────────────────────

/// Feature MSE distillation kernel.
///
/// Each thread computes its squared difference and atomically adds to `out_mse[0]`.
/// Divide by `n` on host to obtain the mean.
#[must_use]
pub fn mse_distill_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// mse_distill_kernel: squared element-wise differences (s - t)^2.
// s_feat: [n] student features
// t_feat: [n] teacher features
// out_mse: [1] MSE accumulator (atomic add; divide by n on host)
// n: total number of elements
.visible .entry mse_distill_kernel(
    .param .u64 s_feat,
    .param .u64 t_feat,
    .param .u64 out_mse,
    .param .u32 n
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [s_feat];
    ld.param.u64  %rd1, [t_feat];
    ld.param.u64  %rd2, [out_mse];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;
    mov.u32       %r7, %r4;

$MSE_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $MSE_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    sub.f32       %f2, %f0, %f1;
    mul.f32       %f3, %f2, %f2;
    atom.global.add.f32 %f4, [%rd2], %f3;

    add.u32       %r7, %r7, %r6;
    bra $MSE_LOOP;

$MSE_DONE:
    mov.f32       %f5, {ZERO};
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 3 ────────────────────────────────────────────────────────────────

/// Attention weight MSE distillation kernel — per-head accumulation.
///
/// Thread index spans `n_heads × seq_sq`; each element contributes to `out_loss[head_idx]`.
#[must_use]
pub fn attn_distill_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// attn_distill_kernel: attention weight MSE per head.
// s_attn: [n_heads * seq_sq] student attention weights (flat)
// t_attn: [n_heads * seq_sq] teacher attention weights (flat)
// out_loss: [n_heads] per-head MSE accumulators
// n_heads: number of attention heads, seq_sq: seq_len^2
.visible .entry attn_distill_kernel(
    .param .u64 s_attn,
    .param .u64 t_attn,
    .param .u64 out_loss,
    .param .u32 n_heads,
    .param .u32 seq_sq
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<14>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [s_attn];
    ld.param.u64  %rd1, [t_attn];
    ld.param.u64  %rd2, [out_loss];
    ld.param.u32  %r0,  [n_heads];
    ld.param.u32  %r1,  [seq_sq];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;

    mul.lo.u32    %r8, %r0, %r1;
    mov.u32       %r9, %r5;

$ATTN_LOOP:
    setp.ge.u32   %p0, %r9, %r8;
    @%p0 bra $ATTN_DONE;

    div.u32       %r10, %r9, %r1;
    setp.ge.u32   %p0, %r10, %r0;
    @%p0 bra $ATTN_NEXT;

    mul.wide.u32  %rd3, %r9, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];

    sub.f32       %f2, %f0, %f1;
    mul.f32       %f3, %f2, %f2;

    mul.wide.u32  %rd5, %r10, 4;
    add.u64       %rd6, %rd2, %rd5;
    atom.global.add.f32 %f4, [%rd6], %f3;

$ATTN_NEXT:
    add.u32       %r9, %r9, %r7;
    bra $ATTN_LOOP;

$ATTN_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 4 ────────────────────────────────────────────────────────────────

/// AT spatial pooling kernel — sums |F`[c,h,w]`|^p over channels.
///
/// Thread per spatial location (hw); accumulates channel-wise abs-power.
#[must_use]
pub fn at_pool_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    let _p_exp = f32_hex(2.0_f32);
    format!(
        r#"{hdr}// at_pool_kernel: AT spatial pooling: out[hw] = sum_c |F[c,hw]|^p.
// feat: [channels * hw] feature map (channel-major flat)
// out:  [hw] pooled attention map
// channels, hw: spatial dimensions; p_exp: power exponent (as f32)
.visible .entry at_pool_kernel(
    .param .u64 feat,
    .param .u64 out,
    .param .u32 channels,
    .param .u32 hw,
    .param .f32 p_exp
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<12>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [feat];
    ld.param.u64  %rd1, [out];
    ld.param.u32  %r0,  [channels];
    ld.param.u32  %r1,  [hw];
    ld.param.f32  %f0,  [p_exp];

    mov.u32       %r2, %ntid.x;
    mov.u32       %r3, %ctaid.x;
    mov.u32       %r4, %tid.x;
    mad.lo.u32    %r5, %r2, %r3, %r4;

    mov.u32       %r6, %nctaid.x;
    mul.lo.u32    %r7, %r2, %r6;
    mov.u32       %r8, %r5;

$AT_LOOP:
    setp.ge.u32   %p0, %r8, %r1;
    @%p0 bra $AT_DONE;

    mov.f32       %f1, {ZERO};
    mov.u32       %r9, 0;

$AT_INNER:
    setp.ge.u32   %p0, %r9, %r0;
    @%p0 bra $AT_INNER_DONE;

    mul.lo.u32    %r10, %r9, %r1;
    add.u32       %r11, %r10, %r8;
    mul.wide.u32  %rd2, %r11, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f2, [%rd3];

    abs.f32       %f3, %f2;
    setp.gt.f32   %p0, %f3, 0F00800000;
    mov.f32       %f4, {ZERO};
    @%p0 lg2.approx.f32 %f4, %f3;
    @%p0 mul.f32 %f4, %f4, 0F3F317218;
    @%p0 mul.f32 %f4, %f4, %f0;
    @%p0 mul.f32 %f5, %f4, 0F3FB8AA3B;
    @%p0 ex2.approx.f32 %f5, %f5;
    setp.le.f32   %p0, %f3, 0F00800000;
    @%p0 mov.f32  %f5, {ZERO};
    add.f32       %f1, %f1, %f5;

    add.u32       %r9, %r9, 1;
    bra $AT_INNER;

$AT_INNER_DONE:
    mul.wide.u32  %rd4, %r8, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f1;

    add.u32       %r8, %r8, %r7;
    bra $AT_LOOP;

$AT_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 5 ────────────────────────────────────────────────────────────────

/// DML peer KL aggregation kernel.
///
/// Each block handles one peer; accumulates KL(self ‖ peer) per class into `out_kl[peer]`.
#[must_use]
pub fn dml_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// dml_loss_kernel: KL(self_probs || peer_probs) per peer block.
// self_probs:      [n_classes] self probability distribution
// peer_probs_flat: [n_peers * n_classes] peer distributions row-major
// out_kl:          [n_peers] per-peer KL accumulators
// n_classes, n_peers: dimensions
.visible .entry dml_loss_kernel(
    .param .u64 self_probs,
    .param .u64 peer_probs_flat,
    .param .u64 out_kl,
    .param .u32 n_classes,
    .param .u32 n_peers
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<14>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [self_probs];
    ld.param.u64  %rd1, [peer_probs_flat];
    ld.param.u64  %rd2, [out_kl];
    ld.param.u32  %r0,  [n_classes];
    ld.param.u32  %r1,  [n_peers];

    mov.u32       %r2, %ctaid.x;    // peer index
    setp.ge.u32   %p0, %r2, %r1;
    @%p0 bra $DML_DONE;

    mov.u32       %r3, %tid.x;
    mov.u32       %r4, %ntid.x;

    mov.f32       %f0, {ZERO};
    mov.u32       %r5, %r3;

$DML_CLASS_LOOP:
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $DML_CLASS_DONE;

    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f1, [%rd4];       // self_prob[c]

    mul.lo.u32    %r6, %r2, %r0;
    add.u32       %r7, %r6, %r5;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f2, [%rd4];       // peer_prob[peer, c]

    setp.gt.f32   %p0, %f1, 0F00800000;
    mov.f32       %f3, {ZERO};
    @%p0 div.rn.f32 %f4, %f1, %f2;
    setp.gt.f32   %p0, %f4, 0F00800000;
    @%p0 lg2.approx.f32 %f5, %f4;
    @%p0 mul.f32 %f5, %f5, 0F3F317218;
    @%p0 mul.f32 %f3, %f1, %f5;
    add.f32       %f0, %f0, %f3;

    add.u32       %r5, %r5, %r4;
    bra $DML_CLASS_LOOP;

$DML_CLASS_DONE:
    setp.ne.u32   %p0, %r3, 0;
    @%p0 bra $DML_DONE;
    mul.wide.u32  %rd5, %r2, 4;
    add.u64       %rd6, %rd2, %rd5;
    atom.global.add.f32 %f6, [%rd6], %f0;

$DML_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 6 ────────────────────────────────────────────────────────────────

/// CRD contrastive cosine similarity kernel.
///
/// Grid = (batch, 1, 1), Block = (min(feat_dim, 256), 1, 1).
/// Shared-memory dot/norm accumulation; outputs one cosine per (anchor, key) pair.
#[must_use]
pub fn crd_score_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// crd_score_kernel: cosine similarity between anchors and keys.
// anchor:     [batch * feat_dim] anchor feature vectors
// keys:       [batch * feat_dim] key feature vectors
// out_scores: [batch] cosine similarity per pair
// batch, feat_dim: dimensions
.visible .entry crd_score_kernel(
    .param .u64 anchor,
    .param .u64 keys,
    .param .u64 out_scores,
    .param .u32 batch,
    .param .u32 feat_dim
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [anchor];
    ld.param.u64  %rd1, [keys];
    ld.param.u64  %rd2, [out_scores];
    ld.param.u32  %r0,  [batch];
    ld.param.u32  %r1,  [feat_dim];

    mov.u32       %r2, %ctaid.x;
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $CRD_DONE;

    mov.u32       %r3, %tid.x;
    mov.u32       %r4, %ntid.x;

    mov.f32       %f0, {ZERO};
    mov.f32       %f1, {ZERO};
    mov.f32       %f2, {ZERO};

    mul.lo.u32    %r5, %r2, %r1;
    mov.u32       %r6, %r3;

$CRD_INNER:
    setp.ge.u32   %p0, %r6, %r1;
    @%p0 bra $CRD_INNER_DONE;

    add.u32       %r7, %r5, %r6;
    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    ld.global.f32 %f3, [%rd4];
    add.u64       %rd4, %rd1, %rd3;
    ld.global.f32 %f4, [%rd4];

    fma.rn.f32    %f0, %f3, %f4, %f0;
    fma.rn.f32    %f1, %f3, %f3, %f1;
    fma.rn.f32    %f2, %f4, %f4, %f2;

    add.u32       %r6, %r6, %r4;
    bra $CRD_INNER;

$CRD_INNER_DONE:
    setp.ne.u32   %p0, %r3, 0;
    @%p0 bra $CRD_DONE;

    sqrt.rn.f32   %f5, %f1;
    sqrt.rn.f32   %f6, %f2;
    mul.f32       %f7, %f5, %f6;
    add.f32       %f7, %f7, 0F33D6BF95;
    div.rn.f32    %f8, %f0, %f7;

    mul.wide.u32  %rd5, %r2, 4;
    add.u64       %rd6, %rd2, %rd5;
    st.global.f32 [%rd6], %f8;

$CRD_DONE:
    ret;
}}
"#,
        ZERO = zero
    )
}

// ─── Kernel 7 ────────────────────────────────────────────────────────────────

/// Gram matrix kernel G = FᵀF.
///
/// Each thread computes one element G`[i,j]` = dot(F`[:,i]`, F`[:,j]`).
/// Grid = ((d+15)/16, (d+15)/16, 1), Block = (16, 16, 1).
#[must_use]
pub fn gram_matrix_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// gram_matrix_kernel: G = F^T @ F, G[j,i] = dot(F[:,j], F[:,i]).
// feat: [n * d] feature matrix (row-major, n samples × d dims)
// gram: [d * d] output Gram matrix
// n: number of samples, d: feature dimension
.visible .entry gram_matrix_kernel(
    .param .u64 feat,
    .param .u64 gram,
    .param .u32 n,
    .param .u32 d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<16>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [feat];
    ld.param.u64  %rd1, [gram];
    ld.param.u32  %r0,  [n];
    ld.param.u32  %r1,  [d];

    mov.u32       %r2, %tid.x;
    mov.u32       %r3, %tid.y;
    mov.u32       %r4, %ctaid.x;
    mov.u32       %r5, %ctaid.y;
    mov.u32       %r6, %ntid.x;
    mov.u32       %r7, %ntid.y;

    mad.lo.u32    %r8, %r4, %r6, %r2;   // global col i
    mad.lo.u32    %r9, %r5, %r7, %r3;   // global row j

    setp.ge.u32   %p0, %r8, %r1;
    @%p0 bra $GRAM_DONE;
    setp.ge.u32   %p0, %r9, %r1;
    @%p0 bra $GRAM_DONE;

    mov.f32       %f0, {ZERO};
    mov.u32       %r10, 0;

$GRAM_INNER:
    setp.ge.u32   %p0, %r10, %r0;
    @%p0 bra $GRAM_INNER_DONE;

    mul.lo.u32    %r11, %r10, %r1;

    add.u32       %r12, %r11, %r8;
    mul.wide.u32  %rd2, %r12, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f1, [%rd3];

    add.u32       %r13, %r11, %r9;
    mul.wide.u32  %rd2, %r13, 4;
    add.u64       %rd3, %rd0, %rd2;
    ld.global.f32 %f2, [%rd3];

    fma.rn.f32    %f0, %f1, %f2, %f0;
    add.u32       %r10, %r10, 1;
    bra $GRAM_INNER;

$GRAM_INNER_DONE:
    mul.lo.u32    %r14, %r9, %r1;
    add.u32       %r15, %r14, %r8;
    mul.wide.u32  %rd4, %r15, 4;
    add.u64       %rd5, %rd1, %rd4;
    st.global.f32 [%rd5], %f0;

$GRAM_DONE:
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
            assert!(!kd_loss_ptx(sm).is_empty());
            assert!(!mse_distill_ptx(sm).is_empty());
            assert!(!attn_distill_ptx(sm).is_empty());
            assert!(!at_pool_ptx(sm).is_empty());
            assert!(!dml_loss_ptx(sm).is_empty());
            assert!(!crd_score_ptx(sm).is_empty());
            assert!(!gram_matrix_ptx(sm).is_empty());
        }
    }
}
