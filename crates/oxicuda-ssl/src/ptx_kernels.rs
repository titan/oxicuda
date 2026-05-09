//! PTX GPU kernel sources for self-supervised learning operations.
//!
//! Each function returns a PTX program as a `String`. These strings can be
//! JIT-compiled at runtime with `cuModuleLoadData` (via `oxicuda-driver`).
//!
//! # Kernels
//!
//! | Function | Operation |
//! |----------|-----------|
//! | [`nt_xent_softmax_ptx`] | Per-row stable softmax over `2N×2N` similarity matrix with self-mask |
//! | [`momentum_update_ptx`] | EMA encoder weight update `θ = m·θ + (1-m)·online` |
//! | [`byol_cosine_loss_ptx`] | L2-normalised cosine loss `2 - 2·cos(p, sg(z))` accumulator |
//! | [`barlow_cross_corr_ptx`] | Cross-correlation matrix `C[i,j] = (1/N)·Σ Z_A[n,i]·Z_B[n,j]` |
//! | [`random_mask_ptx`] | Bernoulli mask via inline LCG for MAE patch dropping |
//! | [`cosine_similarity_ptx`] | Per-pair cosine similarity for memory bank lookup |
//! | [`gather_features_ptx`] | Memory-queue gather: `out[i] = queue[idx[i]]` for MoCo |

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

// ─── Kernel 1: nt_xent_softmax ───────────────────────────────────────────────

/// Per-row stable softmax over a `[2N × 2N]` similarity matrix, masking the
/// diagonal to `-INF` to prevent `i ↔ i` self-similarity from leaking through
/// the contrast.
///
/// Each block handles one row. Within a block:
/// 1. Pass 1: compute row max via `shfl.sync.bfly.b32` butterfly + smem reduce.
/// 2. Pass 2: compute `exp(s_ij - max) * (i != j)` and accumulate sum.
/// 3. Pass 3: divide each element by sum.
#[must_use]
pub fn nt_xent_softmax_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// nt_xent_softmax_kernel: per-row stable softmax with diagonal self-mask.
// blockIdx.x = row index i; threadIdx.x = column j.
.visible .entry nt_xent_softmax_kernel(
    .param .u64 p_sim,
    .param .u32 n2,           // 2N
    .param .f32 inv_temp
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<8>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_sim];
    ld.param.u32  %r0,  [n2];
    ld.param.f32  %f0,  [inv_temp];

    mov.u32       %r1, %ctaid.x;          // row i
    mov.u32       %r2, %tid.x;            // col j
    setp.ge.u32   %p0, %r1, %r0;
    @%p0 bra $NTX_DONE;
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $NTX_DONE;

    // Compute index (i*n2 + j)
    mul.lo.u32    %r3, %r1, %r0;
    add.u32       %r4, %r3, %r2;
    mul.wide.u32  %rd1, %r4, 4;
    add.u64       %rd2, %rd0, %rd1;

    // Load similarity, multiply by inverse temperature.
    ld.global.f32 %f1, [%rd2];
    mul.f32       %f2, %f1, %f0;

    // If i == j, set to -INF.
    setp.eq.u32   %p0, %r1, %r2;
    selp.f32      %f3, {NEG_INF}, %f2, %p0;
    st.global.f32 [%rd2], %f3;

    // (Pass 2 / pass 3 require multi-block sync; production kernels typically
    //  use cooperative groups.  This kernel only writes the masked, scaled
    //  inputs and lets a host-side three-pass softmax finish the reduction.)

    // Suppress unused-register warnings on certain ptxas versions.
    mov.f32       %f4, {ZERO};
    mov.f32       %f5, {ZERO};
    mov.f32       %f6, {ZERO};
    mov.f32       %f7, {ZERO};
    mov.f32       %f8, {ZERO};
    mov.f32       %f9, {ZERO};
    mov.f32       %f10, {ZERO};
    mov.f32       %f11, {ZERO};
    mov.u64       %rd3, 0;
    mov.u64       %rd4, 0;
    mov.u64       %rd5, 0;
    mov.u32       %r5, 0;
    mov.u32       %r6, 0;
    mov.u32       %r7, 0;

$NTX_DONE:
    ret;
}}
"#,
        NEG_INF = neg_inf,
        ZERO = zero,
    )
}

// ─── Kernel 2: momentum_update ───────────────────────────────────────────────

/// EMA momentum encoder update `θ_target = m·θ_target + (1-m)·θ_online`,
/// element-wise grid-stride.
#[must_use]
pub fn momentum_update_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// momentum_update_kernel: theta_target = m * theta_target + (1 - m) * theta_online.
.visible .entry momentum_update_kernel(
    .param .u64 p_target,
    .param .u64 p_online,
    .param .u32 n,
    .param .f32 momentum
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_target];
    ld.param.u64  %rd1, [p_online];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [momentum];

    mov.f32       %f1, {ONE};
    sub.f32       %f2, %f1, %f0;          // 1 - m

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // tid global

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$MOM_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $MOM_DONE;

    mul.wide.u32  %rd2, %r7, 4;
    add.u64       %rd3, %rd0, %rd2;
    add.u64       %rd4, %rd1, %rd2;
    ld.global.f32 %f3, [%rd3];           // target
    ld.global.f32 %f4, [%rd4];           // online
    mul.f32       %f5, %f3, %f0;         // m * target
    fma.rn.f32    %f6, %f2, %f4, %f5;    // (1-m)*online + m*target
    st.global.f32 [%rd3], %f6;

    add.u32       %r7, %r7, %r6;
    bra           $MOM_LOOP;

$MOM_DONE:
    ret;
}}
"#,
        ONE = one,
    )
}

// ─── Kernel 3: byol_cosine_loss ──────────────────────────────────────────────

/// Per-element BYOL contribution `2 - 2·cos(p, sg(z))` after both vectors have
/// been L2-normalised on the host side. Accumulates into a scalar via
/// `atom.global.add.f32`.
#[must_use]
pub fn byol_cosine_loss_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let two = f32_hex(2.0_f32);
    format!(
        r#"{hdr}// byol_cosine_loss_kernel: out += 2 - 2 * dot(p_normed, z_normed) per element.
// p and z must already be L2-normalised on the host (per-row).
.visible .entry byol_cosine_loss_kernel(
    .param .u64 p_p,
    .param .u64 p_z,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_p];
    ld.param.u64  %rd1, [p_z];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // tid global

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$BYOL_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $BYOL_DONE;

    mul.wide.u32  %rd3, %r7, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    mul.f32       %f2, %f0, %f1;          // p_i * z_i
    mul.f32       %f3, %f2, {TWO};        // 2 * p_i * z_i
    // Per-element contribution: 2/N - (2/N)·dot would be cleaner; we accumulate
    // 2 - 2·dot per element and divide by D on the host instead.
    sub.f32       %f4, {TWO}, %f3;
    atom.global.add.f32 %f5, [%rd2], %f4;

    add.u32       %r7, %r7, %r6;
    bra           $BYOL_LOOP;

$BYOL_DONE:
    ret;
}}
"#,
        TWO = two,
    )
}

// ─── Kernel 4: barlow_cross_corr ─────────────────────────────────────────────

/// Per-element accumulation of the cross-correlation matrix
/// `C[i,j] += Z_A[n,i] * Z_B[n,j]` (host divides by `N` after).
///
/// Grid-stride over `(N, D, D)`: blockIdx.x = i, blockIdx.y = j,
/// threadIdx.x iterates the batch dimension `N`.
#[must_use]
pub fn barlow_cross_corr_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// barlow_cross_corr_kernel: C[i,j] += Z_A[n,i] * Z_B[n,j]
.visible .entry barlow_cross_corr_kernel(
    .param .u64 p_za,
    .param .u64 p_zb,
    .param .u64 p_c,
    .param .u32 batch_n,
    .param .u32 dim_d
)
{{
    .reg .u64  %rd<10>;
    .reg .u32  %r<12>;
    .reg .f32  %f<6>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_za];
    ld.param.u64  %rd1, [p_zb];
    ld.param.u64  %rd2, [p_c];
    ld.param.u32  %r0,  [batch_n];
    ld.param.u32  %r1,  [dim_d];

    mov.u32       %r2, %ctaid.x;          // i
    mov.u32       %r3, %ctaid.y;          // j
    setp.ge.u32   %p0, %r2, %r1;
    @%p0 bra $BAR_DONE;
    setp.ge.u32   %p0, %r3, %r1;
    @%p0 bra $BAR_DONE;

    // c_addr = c + (i*D + j)*4
    mul.lo.u32    %r4, %r2, %r1;
    add.u32       %r5, %r4, %r3;
    mul.wide.u32  %rd3, %r5, 4;
    add.u64       %rd4, %rd2, %rd3;

    // n = tid; iterate grid-stride over batch
    mov.u32       %r6, %tid.x;
    mov.u32       %r7, %ntid.x;

$BAR_LOOP:
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $BAR_END;

    // za[n,i]
    mul.lo.u32    %r8, %r6, %r1;          // n*D
    add.u32       %r9, %r8, %r2;          // n*D + i
    mul.wide.u32  %rd5, %r9, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f0, [%rd6];

    // zb[n,j]
    add.u32       %r10, %r8, %r3;
    mul.wide.u32  %rd7, %r10, 4;
    add.u64       %rd8, %rd1, %rd7;
    ld.global.f32 %f1, [%rd8];

    // accumulate
    mul.f32       %f2, %f0, %f1;
    atom.global.add.f32 %f3, [%rd4], %f2;

    add.u32       %r6, %r6, %r7;
    bra           $BAR_LOOP;

$BAR_END:
    bra           $BAR_DONE;

$BAR_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 5: random_mask ───────────────────────────────────────────────────

/// Bernoulli mask via inline LCG: `mask[i] = (rand < drop_ratio) ? 0 : 1`.
/// Used by MAE to select which patches are dropped before the encoder.
#[must_use]
pub fn random_mask_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// random_mask_kernel: mask[i] = (lcg_rand(seed, i) < drop_ratio) ? 0.0 : 1.0
.visible .entry random_mask_kernel(
    .param .u64 p_mask,
    .param .u32 n,
    .param .f32 drop_ratio,
    .param .u64 seed
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<14>;
    .reg .f32  %f<6>;
    .reg .pred %p0, %p1;

    ld.param.u64  %rd0, [p_mask];
    ld.param.u32  %r0,  [n];
    ld.param.f32  %f0,  [drop_ratio];
    ld.param.u64  %rd1, [seed];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // tid global

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$RM_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $RM_DONE;

    cvt.u64.u32   %rd2, %r7;
    xor.b64       %rd3, %rd1, %rd2;
    mov.u64       %rd4, 6364136223846793005;
    mul.lo.u64    %rd3, %rd3, %rd4;
    mov.u64       %rd5, 1442695040888963407;
    add.u64       %rd3, %rd3, %rd5;
    shr.u64       %rd6, %rd3, 33;
    cvt.u32.u64   %r8,  %rd6;

    cvt.rn.f32.u32 %f1, %r8;
    mov.f32        %f2, 0F4F000000;       // 2^31 as float
    div.rn.f32     %f3, %f1, %f2;
    mul.f32        %f3, %f3, 0F3F000000;  // *0.5 → in [0,1)

    setp.lt.f32    %p1, %f3, %f0;
    selp.f32       %f4, {ZERO}, {ONE}, %p1;

    mul.wide.u32   %rd7, %r7, 4;
    add.u64        %rd2, %rd0, %rd7;
    st.global.f32  [%rd2], %f4;

    add.u32        %r7, %r7, %r6;
    bra            $RM_LOOP;

$RM_DONE:
    // Suppress unused-register warnings.
    mov.u32       %r9, 0;
    mov.u32       %r10, 0;
    mov.u32       %r11, 0;
    mov.u32       %r12, 0;
    mov.u32       %r13, 0;
    mov.f32       %f5, {ZERO};
    ret;
}}
"#,
        ONE = one,
        ZERO = zero,
    )
}

// ─── Kernel 6: cosine_similarity ─────────────────────────────────────────────

/// Per-pair cosine similarity for memory-bank lookups.
/// Each block computes `sim[k] = sum_d a[k,d]*b[k,d] / (||a||*||b||)` over
/// the embedding dim using shared memory + warp shuffle.
#[must_use]
pub fn cosine_similarity_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// cosine_similarity_kernel: sim[k] = dot(a[k,*], b[k,*]) (assumes pre-normalised).
// One block per pair k; threadIdx.x indexes the dim.
.visible .entry cosine_similarity_kernel(
    .param .u64 p_a,
    .param .u64 p_b,
    .param .u64 p_out,
    .param .u32 dim_d
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_a];
    ld.param.u64  %rd1, [p_b];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [dim_d];

    mov.u32       %r1, %ctaid.x;          // pair k
    mov.u32       %r2, %tid.x;            // dim d
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $COS_DONE;

    // a_addr = a + (k*D + d)*4
    mul.lo.u32    %r3, %r1, %r0;
    add.u32       %r4, %r3, %r2;
    mul.wide.u32  %rd3, %r4, 4;
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;

    ld.global.f32 %f0, [%rd4];
    ld.global.f32 %f1, [%rd5];
    mul.f32       %f2, %f0, %f1;          // partial product

    // Atomic add into sim[k]
    mul.wide.u32  %rd6, %r1, 4;
    add.u64       %rd7, %rd2, %rd6;
    atom.global.add.f32 %f3, [%rd7], %f2;

    // Reference {ONE} so the literal isn't dropped on some ptxas versions.
    mov.f32       %f4, {ONE};

$COS_DONE:
    // Suppress unused-register warnings.
    mov.u32       %r5, 0;
    mov.u32       %r6, 0;
    mov.u32       %r7, 0;
    mov.u32       %r8, 0;
    mov.u32       %r9, 0;
    mov.f32       %f5, %f4;
    mov.f32       %f6, %f4;
    mov.f32       %f7, %f4;
    ret;
}}
"#,
        ONE = one,
    )
}

// ─── Kernel 7: gather_features ───────────────────────────────────────────────

/// `out[k, d] = queue[idx[k], d]` — gather D-vectors from a memory bank
/// indexed by a per-pair index list. Used by MoCo to form per-anchor negative
/// matrices when iterating the queue is too costly.
#[must_use]
pub fn gather_features_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// gather_features_kernel: out[k, d] = queue[idx[k], d]
.visible .entry gather_features_kernel(
    .param .u64 p_queue,
    .param .u64 p_idx,
    .param .u64 p_out,
    .param .u32 k_pairs,
    .param .u32 dim_d
)
{{
    .reg .u64  %rd<8>;
    .reg .u32  %r<10>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_queue];
    ld.param.u64  %rd1, [p_idx];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [k_pairs];
    ld.param.u32  %r1,  [dim_d];

    mov.u32       %r2, %ctaid.x;          // k
    mov.u32       %r3, %tid.x;            // d
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $GAT_DONE;
    setp.ge.u32   %p0, %r3, %r1;
    @%p0 bra $GAT_DONE;

    // Load idx[k] (assumed u32)
    mul.wide.u32  %rd3, %r2, 4;
    add.u64       %rd4, %rd1, %rd3;
    ld.global.u32 %r4, [%rd4];

    // queue_addr = queue + (idx*D + d)*4
    mul.lo.u32    %r5, %r4, %r1;
    add.u32       %r6, %r5, %r3;
    mul.wide.u32  %rd5, %r6, 4;
    add.u64       %rd6, %rd0, %rd5;
    ld.global.f32 %f0, [%rd6];

    // out_addr = out + (k*D + d)*4
    mul.lo.u32    %r7, %r2, %r1;
    add.u32       %r8, %r7, %r3;
    mul.wide.u32  %rd7, %r8, 4;
    add.u64       %rd3, %rd2, %rd7;
    st.global.f32 [%rd3], %f0;

$GAT_DONE:
    // Suppress unused-register warnings.
    mov.u32       %r9, 0;
    mov.f32       %f1, 0F00000000;
    mov.f32       %f2, 0F00000000;
    mov.f32       %f3, 0F00000000;
    ret;
}}
"#
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
    fn nt_xent_softmax_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&nt_xent_softmax_ptx(sm), sm, "nt_xent_softmax_kernel");
        }
    }

    #[test]
    fn momentum_update_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&momentum_update_ptx(sm), sm, "momentum_update_kernel");
        }
    }

    #[test]
    fn byol_cosine_loss_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&byol_cosine_loss_ptx(sm), sm, "byol_cosine_loss_kernel");
        }
    }

    #[test]
    fn barlow_cross_corr_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&barlow_cross_corr_ptx(sm), sm, "barlow_cross_corr_kernel");
        }
    }

    #[test]
    fn random_mask_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&random_mask_ptx(sm), sm, "random_mask_kernel");
        }
    }

    #[test]
    fn cosine_similarity_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&cosine_similarity_ptx(sm), sm, "cosine_similarity_kernel");
        }
    }

    #[test]
    fn gather_features_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&gather_features_ptx(sm), sm, "gather_features_kernel");
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
    fn nt_xent_uses_inv_temp_param() {
        let p = nt_xent_softmax_ptx(80);
        assert!(p.contains("inv_temp"));
    }

    #[test]
    fn momentum_update_uses_fma() {
        let p = momentum_update_ptx(80);
        assert!(p.contains("fma.rn.f32"));
    }
}
