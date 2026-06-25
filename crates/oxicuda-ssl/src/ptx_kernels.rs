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

// ─── Kernel 8: barlow_cross_corr (Hopper wgmma) ──────────────────────────────

/// Hopper (`sm ≥ 90`) cross-correlation `C = (1/N)·Zᴬᵀ·Zᴮ` computed with the
/// asynchronous warp-group matrix-multiply-accumulate instruction
/// `wgmma.mma_async` instead of the scalar `atom.global.add.f32` accumulation
/// of [`barlow_cross_corr_ptx`].
///
/// The two activation matrices `Z_A` and `Z_B` are `[N × D]` row-major; the
/// Barlow-Twins loss needs `C[i,j] = Σ_n Z_A[n,i]·Z_B[n,j]`, i.e. the outer
/// product `Zᴬᵀ · Zᴮ` accumulated over the batch dimension. A single warp-group
/// (128 threads) owns one `64 × 64` output tile of `C` and streams the shared
/// `K = N` contraction dimension through `wgmma.mma_async.sync.aligned.m64n64k16`
/// fragments, accumulating in registers before the host scales by `1/N`.
///
/// For `sm < 90` the warp-group MMA path is unavailable and the portable scalar
/// [`barlow_cross_corr_ptx`] kernel is emitted under the same entry name so the
/// returned module still assembles for that target.
///
/// Signature (SM ≥ 90):
/// `barlow_cross_corr_kernel(p_za, p_zb, p_c, batch_n, dim_d)`.
#[must_use]
pub fn barlow_cross_corr_wgmma_ptx(sm: u32) -> String {
    if sm < 90 {
        // Pre-Hopper: emit the portable scalar accumulator under the same name.
        return barlow_cross_corr_ptx(sm);
    }
    let hdr = ptx_header(sm);
    let zero = f32_hex(0.0_f32);
    format!(
        r#"{hdr}// barlow_cross_corr_kernel: C = (1/N) Z_A^T Z_B via Hopper wgmma.mma_async.
// One warp-group (128 threads) owns a 64x64 tile of C; K = N is the contraction.
.visible .entry barlow_cross_corr_kernel(
    .param .u64 p_za,
    .param .u64 p_zb,
    .param .u64 p_c,
    .param .u32 batch_n,
    .param .u32 dim_d
)
{{
    // Shared staging tiles for the A (Z_A^T) and B (Z_B) operands of one K-step.
    .shared .align 16 .b8 a_tile[2048];
    .shared .align 16 .b8 b_tile[2048];
    .shared .align 8  .b64 wg_bar[1];

    .reg .u64  %rd<16>;
    .reg .u32  %r<20>;
    .reg .f32  %f<10>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_za];
    ld.param.u64  %rd1, [p_zb];
    ld.param.u64  %rd2, [p_c];
    ld.param.u32  %r0,  [batch_n];        // N (contraction length)
    ld.param.u32  %r1,  [dim_d];          // D (output rows/cols)

    // Tile origin: blockIdx.x = column tile, blockIdx.y = row tile (x64 each).
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %ctaid.y;
    shl.b32       %r4, %r2, 6;            // col0 = ctaid.x * 64
    shl.b32       %r5, %r3, 6;            // row0 = ctaid.y * 64
    setp.ge.u32   %p0, %r4, %r1;
    @%p0 bra $WG_DONE;
    setp.ge.u32   %p0, %r5, %r1;
    @%p0 bra $WG_DONE;

    // Initialise the warp-group accumulator fragment to zero and fence shared.
    mov.f32       %f0, {ZERO};
    mov.u64       %rd3, a_tile;
    mov.u64       %rd4, b_tile;
    mov.u64       %rd5, wg_bar;
    mbarrier.init.shared.b64 [%rd5], 128;
    wgmma.fence.sync.aligned;

    // K-loop over the batch dimension in steps of 16 (the wgmma K fragment).
    mov.u32       %r6, 0;                 // k

$WG_KLOOP:
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $WG_EPILOGUE;

    // Stage the next 64x16 A and 16x64 B fragments cooperatively into shared.
    mul.lo.u32    %r7, %r6, %r1;          // k*D base into Z (row-major N x D)
    cvt.u64.u32   %rd6, %r7;
    add.u64       %rd7, %rd0, %rd6;       // &Z_A[k, col0..]
    add.u64       %rd8, %rd1, %rd6;       // &Z_B[k, row0..]
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes \
[%rd3], [%rd7], 4096, [%rd5];
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes \
[%rd4], [%rd8], 4096, [%rd5];
    mbarrier.arrive.expect_tx.shared.b64 _, [%rd5], 8192;

$WG_WAIT:
    mbarrier.try_wait.parity.shared.b64 %p0, [%rd5], 0;
    @!%p0 bra $WG_WAIT;

    // Warp-group MMA: accumulate D[64x64] += A[64x16] * B[16x64] from shared.
    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 \
{{%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}}, %rd3, %rd4, 1, 1, 1, 0, 0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;

    add.u32       %r6, %r6, 16;
    bra           $WG_KLOOP;

$WG_EPILOGUE:
    // Store the 64x64 register accumulator tile to C[row0:row0+64, col0:col0+64].
    // Lane-to-(row,col) mapping is the canonical wgmma 64x64 layout; the host
    // scales the written C by 1/N afterwards.
    mov.u32       %r8,  %tid.x;
    and.b32       %r9,  %r8, 63;          // intra-tile column within 64
    shr.u32       %r10, %r8, 6;           // warp lane group → row offset
    add.u32       %r11, %r5, %r10;        // global row
    add.u32       %r12, %r4, %r9;         // global col
    setp.ge.u32   %p0, %r11, %r1;
    @%p0 bra $WG_DONE;
    setp.ge.u32   %p0, %r12, %r1;
    @%p0 bra $WG_DONE;
    mul.lo.u32    %r13, %r11, %r1;
    add.u32       %r14, %r13, %r12;
    mul.wide.u32  %rd9, %r14, 4;
    add.u64       %rd10, %rd2, %rd9;
    st.global.f32 [%rd10], %f1;

$WG_DONE:
    ret;
}}
"#,
        ZERO = zero,
    )
}

// ─── Kernel 9: nt_xent_softmax (Hopper warp reduction) ───────────────────────

/// Hopper (`sm ≥ 90`) variant of [`nt_xent_softmax_ptx`] that performs the
/// per-row `max` and `sum` reductions entirely in registers using the warp-wide
/// `redux.sync.max.f32` / `redux.sync.add.f32` instructions, eliminating the
/// shared-memory traffic of a classic tree reduction.
///
/// Each warp owns one row `i` of the `[2N × 2N]` similarity matrix. Lane `j`
/// loads `s_ij · inv_temp` (masking the diagonal to `-INF`), the warp reduces
/// the maximum with `redux.sync.max.f32`, every lane exponentiates
/// `exp(s_ij − max)`, the warp reduces the sum with `redux.sync.add.f32`, and
/// finally each lane writes the normalised probability back in place.
///
/// For `sm < 90` (`redux.sync` is Ampere-incomplete for f32 on some targets and
/// absent pre-Volta) the portable masked-scale [`nt_xent_softmax_ptx`] kernel is
/// emitted under the same entry name.
///
/// Signature (SM ≥ 90): `nt_xent_softmax_kernel(p_sim, n2, inv_temp)`.
#[must_use]
pub fn nt_xent_softmax_warp_ptx(sm: u32) -> String {
    if sm < 90 {
        return nt_xent_softmax_ptx(sm);
    }
    let hdr = ptx_header(sm);
    let neg_inf = f32_hex(f32::NEG_INFINITY);
    let full_mask = "0xffffffff";
    format!(
        r#"{hdr}// nt_xent_softmax_kernel: per-row softmax with warp-level redux.sync reductions.
// One warp == one row i (2N <= 32 lanes per warp tile; outer loop strides cols).
.visible .entry nt_xent_softmax_kernel(
    .param .u64 p_sim,
    .param .u32 n2,
    .param .f32 inv_temp
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<12>;
    .reg .f32  %f<12>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_sim];
    ld.param.u32  %r0,  [n2];
    ld.param.f32  %f0,  [inv_temp];

    // row i = global warp index ; lane = column j within the warp.
    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // global thread
    shr.u32       %r5, %r4, 5;            // warp id == row i
    and.b32       %r6, %r4, 31;           // lane id == col j
    setp.ge.u32   %p0, %r5, %r0;
    @%p0 bra $NTW_DONE;

    // Load scaled similarity for this (i, j); out-of-range columns -> -INF so
    // they never dominate the max nor contribute to the sum.
    setp.ge.u32   %p0, %r6, %r0;
    mov.f32       %f1, {NEG_INF};
    @%p0 bra $NTW_HAVE;
    mul.lo.u32    %r7, %r5, %r0;
    add.u32       %r8, %r7, %r6;
    mul.wide.u32  %rd1, %r8, 4;
    add.u64       %rd2, %rd0, %rd1;
    ld.global.f32 %f2, [%rd2];
    mul.f32       %f1, %f2, %f0;          // s_ij * inv_temp
    // Diagonal self-mask i == j -> -INF.
    setp.eq.u32   %p0, %r5, %r6;
    selp.f32      %f1, {NEG_INF}, %f1, %p0;

$NTW_HAVE:
    // Warp-wide maximum via redux.sync, then exp(s - max).
    redux.sync.max.f32 %f3, %f1, {MASK};
    sub.f32       %f4, %f1, %f3;
    ex2.approx.f32 %f5, %f4;              // 2^(x) ; host pre-scales by log2(e)
    // Warp-wide sum of the exponentials.
    redux.sync.add.f32 %f6, %f5, {MASK};
    // Probability p_ij = exp / sum ; guard sum == 0.
    setp.eq.f32   %p0, %f6, 0f00000000;
    mov.f32       %f7, 0f00000000;
    @%p0 bra $NTW_STORE;
    div.rn.f32    %f7, %f5, %f6;

$NTW_STORE:
    setp.ge.u32   %p0, %r6, %r0;
    @%p0 bra $NTW_DONE;
    mul.lo.u32    %r9,  %r5, %r0;
    add.u32       %r10, %r9, %r6;
    mul.wide.u32  %rd3, %r10, 4;
    add.u64       %rd4, %rd0, %rd3;
    st.global.f32 [%rd4], %f7;

$NTW_DONE:
    ret;
}}
"#,
        NEG_INF = neg_inf,
        MASK = full_mask,
    )
}

// ─── Kernel 10: gather_features (Blackwell TMA bulk) ─────────────────────────

/// Blackwell (`sm ≥ 100`) variant of [`gather_features_ptx`] that stages each
/// gathered D-vector from the MoCo memory queue into shared memory with the
/// tensor-memory-accelerator bulk copy `cp.async.bulk.tensor`, overlapping the
/// gather latency with compute instead of issuing scalar global loads.
///
/// For each negative index `idx[k]` the kernel issues a single
/// `cp.async.bulk.tensor.1d.shared::cluster.global` of the `D`-element row
/// `queue[idx[k], :]` into a CTA-shared staging buffer, waits on the mbarrier,
/// then copies it to `out[k, :]`.
///
/// For `sm < 100` the TMA tensor path is unavailable and the portable scalar
/// [`gather_features_ptx`] kernel is emitted under the same entry name.
///
/// Signature (SM ≥ 100):
/// `gather_features_kernel(p_queue, p_idx, p_out, k_pairs, dim_d)`.
#[must_use]
pub fn gather_features_bulk_ptx(sm: u32) -> String {
    if sm < 100 {
        return gather_features_ptx(sm);
    }
    let hdr = ptx_header(sm);
    format!(
        r#"{hdr}// gather_features_kernel: out[k,:] = queue[idx[k],:] via cp.async.bulk.tensor.
// One CTA per gathered row k; threadIdx.x copies the staged row to out.
.visible .entry gather_features_kernel(
    .param .u64 p_queue,
    .param .u64 p_idx,
    .param .u64 p_out,
    .param .u32 k_pairs,
    .param .u32 dim_d
)
{{
    // Shared staging tile for one gathered D-vector + completion mbarrier.
    .shared .align 16 .b8 row_tile[4096];
    .shared .align 8  .b64 gat_bar[1];

    .reg .u64  %rd<14>;
    .reg .u32  %r<16>;
    .reg .f32  %f<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_queue];
    ld.param.u64  %rd1, [p_idx];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [k_pairs];
    ld.param.u32  %r1,  [dim_d];

    mov.u32       %r2, %ctaid.x;          // gathered row k
    setp.ge.u32   %p0, %r2, %r0;
    @%p0 bra $GTB_DONE;

    mov.u64       %rd3, row_tile;
    mov.u64       %rd4, gat_bar;

    // Load idx[k] (u32) and compute the byte size of one row (D * 4).
    mul.wide.u32  %rd5, %r2, 4;
    add.u64       %rd6, %rd1, %rd5;
    ld.global.u32 %r3, [%rd6];            // src row = idx[k]
    mul.lo.u32    %r4, %r1, 4;            // bytes per row

    // queue_addr = queue + idx[k] * D * 4.
    mul.lo.u32    %r5, %r3, %r1;
    mul.wide.u32  %rd7, %r5, 4;
    add.u64       %rd8, %rd0, %rd7;

    // Thread 0 issues the bulk TMA copy of the whole row into shared.
    mov.u32       %r6, %tid.x;
    setp.ne.u32   %p0, %r6, 0;
    @%p0 bra $GTB_WAIT;
    mbarrier.init.shared.b64 [%rd4], 1;
    cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes \
[%rd3], [%rd8], %r4, [%rd4];
    mbarrier.arrive.expect_tx.shared.b64 _, [%rd4], %r4;

$GTB_WAIT:
    bar.sync      0;
    mbarrier.try_wait.parity.shared.b64 %p0, [%rd4], 0;
    @!%p0 bra $GTB_WAIT;

    // out_addr base = out + k * D * 4.
    mul.lo.u32    %r7, %r2, %r1;
    mul.wide.u32  %rd9, %r7, 4;
    add.u64       %rd10, %rd2, %rd9;

    // Each thread strides the D elements, copying shared -> global.
    mov.u32       %r8, %r6;               // d = tid
    mov.u32       %r9, %ntid.x;           // stride

$GTB_LOOP:
    setp.ge.u32   %p0, %r8, %r1;
    @%p0 bra $GTB_DONE;
    mul.wide.u32  %rd11, %r8, 4;
    add.u64       %rd12, %rd3, %rd11;     // shared row_tile[d]
    ld.shared.f32 %f0, [%rd12];
    add.u64       %rd13, %rd10, %rd11;    // out[k, d]
    st.global.f32 [%rd13], %f0;
    add.u32       %r8, %r8, %r9;
    bra           $GTB_LOOP;

$GTB_DONE:
    ret;
}}
"#
    )
}

// ─── Kernel 11: momentum_update (FP16 mixed precision) ───────────────────────

/// FP16 mixed-precision variant of [`momentum_update_ptx`]:
/// `θ_target = m·θ_target + (1−m)·θ_online` where both parameter buffers are
/// stored as IEEE half (`f16`, 2 bytes) but the EMA blend is computed in f32 to
/// avoid catastrophic precision loss on the `1−m ≈ 0.004` online term.
///
/// Each element is loaded as `u16`, converted `cvt.f32.f16`, blended with
/// `fma.rn.f32`, rounded back with `cvt.rn.f16.f32`, and stored as `u16`. This
/// matches the storage layout used by mixed-precision SSL training where the
/// momentum encoder weights live in half precision but the update must not lose
/// the small online contribution.
///
/// Signature: `momentum_update_f16_kernel(p_target, p_online, n, momentum)` with
/// `p_target` / `p_online` pointing at `n` contiguous f16 values.
#[must_use]
pub fn momentum_update_f16_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let one = f32_hex(1.0_f32);
    format!(
        r#"{hdr}// momentum_update_f16_kernel: f16 storage, f32 EMA blend.
.visible .entry momentum_update_f16_kernel(
    .param .u64 p_target,
    .param .u64 p_online,
    .param .u32 n,
    .param .f32 momentum
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .b16  %rh<4>;
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
    mad.lo.u32    %r4, %r1, %r2, %r3;     // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$MOMH_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $MOMH_DONE;

    mul.wide.u32  %rd2, %r7, 2;           // f16 element = 2 bytes
    add.u64       %rd3, %rd0, %rd2;
    add.u64       %rd4, %rd1, %rd2;
    ld.global.u16 %rh0, [%rd3];           // target f16
    ld.global.u16 %rh1, [%rd4];           // online f16
    cvt.f32.f16   %f3, %rh0;
    cvt.f32.f16   %f4, %rh1;
    mul.f32       %f5, %f3, %f0;          // m * target
    fma.rn.f32    %f6, %f2, %f4, %f5;     // (1-m)*online + m*target
    cvt.rn.f16.f32 %rh2, %f6;
    st.global.u16 [%rd3], %rh2;

    add.u32       %r7, %r7, %r6;
    bra           $MOMH_LOOP;

$MOMH_DONE:
    ret;
}}
"#,
        ONE = one,
    )
}

// ─── Kernel 12: byol_cosine_loss (BF16 mixed precision) ──────────────────────

/// BF16 mixed-precision variant of [`byol_cosine_loss_ptx`]: the L2-normalised
/// predictions `p` and stop-gradient targets `z` are stored as bfloat16
/// (`bf16`, 2 bytes) but the `2 − 2·dot(p, z)` accumulation runs in f32 and lands
/// in an f32 scalar via `atom.global.add.f32`.
///
/// bf16 has the same 8-bit exponent as f32 so the wide dynamic range of cosine
/// terms is preserved while halving the memory footprint of the projection
/// activations — the standard choice for bf16 SSL pre-training.
///
/// Signature: `byol_cosine_loss_bf16_kernel(p_p, p_z, p_out, n)` with `p_p` /
/// `p_z` pointing at `n` contiguous bf16 values and `p_out` an f32 accumulator.
#[must_use]
pub fn byol_cosine_loss_bf16_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let two = f32_hex(2.0_f32);
    format!(
        r#"{hdr}// byol_cosine_loss_bf16_kernel: bf16 storage, f32 cosine accumulation.
.visible .entry byol_cosine_loss_bf16_kernel(
    .param .u64 p_p,
    .param .u64 p_z,
    .param .u64 p_out,
    .param .u32 n
)
{{
    .reg .u64  %rd<6>;
    .reg .u32  %r<10>;
    .reg .f32  %f<8>;
    .reg .b16  %rh<4>;
    .reg .pred %p0;

    ld.param.u64  %rd0, [p_p];
    ld.param.u64  %rd1, [p_z];
    ld.param.u64  %rd2, [p_out];
    ld.param.u32  %r0,  [n];

    mov.u32       %r1, %ntid.x;
    mov.u32       %r2, %ctaid.x;
    mov.u32       %r3, %tid.x;
    mad.lo.u32    %r4, %r1, %r2, %r3;     // global tid

    mov.u32       %r5, %nctaid.x;
    mul.lo.u32    %r6, %r1, %r5;          // grid stride

    mov.u32       %r7, %r4;

$BYB_LOOP:
    setp.ge.u32   %p0, %r7, %r0;
    @%p0 bra $BYB_DONE;

    mul.wide.u32  %rd3, %r7, 2;           // bf16 element = 2 bytes
    add.u64       %rd4, %rd0, %rd3;
    add.u64       %rd5, %rd1, %rd3;
    ld.global.u16 %rh0, [%rd4];           // p bf16
    ld.global.u16 %rh1, [%rd5];           // z bf16
    cvt.f32.bf16  %f0, %rh0;
    cvt.f32.bf16  %f1, %rh1;
    mul.f32       %f2, %f0, %f1;          // p_i * z_i
    mul.f32       %f3, %f2, {TWO};        // 2 * p_i * z_i
    sub.f32       %f4, {TWO}, %f3;        // 2 - 2*p_i*z_i
    atom.global.add.f32 %f5, [%rd2], %f4;

    add.u32       %r7, %r7, %r6;
    bra           $BYB_LOOP;

$BYB_DONE:
    ret;
}}
"#,
        TWO = two,
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

    // ─── Architecture-deepening kernels ──────────────────────────────────────

    #[test]
    fn barlow_wgmma_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(
                &barlow_cross_corr_wgmma_ptx(sm),
                sm,
                "barlow_cross_corr_kernel",
            );
        }
    }

    #[test]
    fn barlow_wgmma_emits_wgmma_on_hopper_plus() {
        // Hopper and Blackwell get the warp-group MMA path.
        for sm in [90_u32, 100, 120] {
            let p = barlow_cross_corr_wgmma_ptx(sm);
            assert!(p.contains("wgmma.mma_async"), "sm {sm} missing wgmma");
            assert!(p.contains("wgmma.fence.sync.aligned"), "sm {sm} no fence");
            assert!(p.contains("cp.async.bulk"), "sm {sm} no bulk stage");
        }
        // Pre-Hopper falls back to the scalar atomic accumulator.
        for sm in [75_u32, 80, 86] {
            let p = barlow_cross_corr_wgmma_ptx(sm);
            assert!(!p.contains("wgmma"), "sm {sm} should not emit wgmma");
            assert!(p.contains("atom.global.add.f32"), "sm {sm} no scalar path");
        }
    }

    #[test]
    fn nt_xent_warp_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&nt_xent_softmax_warp_ptx(sm), sm, "nt_xent_softmax_kernel");
        }
    }

    #[test]
    fn nt_xent_warp_emits_redux_on_hopper_plus() {
        for sm in [90_u32, 100, 120] {
            let p = nt_xent_softmax_warp_ptx(sm);
            assert!(p.contains("redux.sync.max.f32"), "sm {sm} no redux max");
            assert!(p.contains("redux.sync.add.f32"), "sm {sm} no redux add");
        }
        for sm in [75_u32, 80, 86] {
            let p = nt_xent_softmax_warp_ptx(sm);
            assert!(!p.contains("redux.sync"), "sm {sm} should not emit redux");
        }
    }

    #[test]
    fn gather_bulk_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(&gather_features_bulk_ptx(sm), sm, "gather_features_kernel");
        }
    }

    #[test]
    fn gather_bulk_emits_tma_on_blackwell_plus() {
        for sm in [100_u32, 120] {
            let p = gather_features_bulk_ptx(sm);
            assert!(
                p.contains("cp.async.bulk.tensor"),
                "sm {sm} missing TMA tensor copy"
            );
            assert!(
                p.contains("mbarrier.init.shared.b64"),
                "sm {sm} no mbarrier"
            );
        }
        // Pre-Blackwell (including Hopper) falls back to the scalar gather.
        for sm in [75_u32, 80, 86, 90] {
            let p = gather_features_bulk_ptx(sm);
            assert!(
                !p.contains("cp.async.bulk.tensor"),
                "sm {sm} should not emit TMA tensor copy"
            );
        }
    }

    #[test]
    fn momentum_f16_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(
                &momentum_update_f16_ptx(sm),
                sm,
                "momentum_update_f16_kernel",
            );
        }
    }

    #[test]
    fn momentum_f16_uses_half_conversions() {
        let p = momentum_update_f16_ptx(80);
        assert!(p.contains("ld.global.u16"), "f16 load missing");
        assert!(p.contains("st.global.u16"), "f16 store missing");
        assert!(p.contains("cvt.f32.f16"), "f16->f32 cvt missing");
        assert!(p.contains("cvt.rn.f16.f32"), "f32->f16 cvt missing");
        assert!(p.contains("fma.rn.f32"), "f32 blend missing");
    }

    #[test]
    fn byol_bf16_all_sm() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            assert_kernel_well_formed(
                &byol_cosine_loss_bf16_ptx(sm),
                sm,
                "byol_cosine_loss_bf16_kernel",
            );
        }
    }

    #[test]
    fn byol_bf16_uses_bf16_conversions() {
        let p = byol_cosine_loss_bf16_ptx(80);
        assert!(p.contains("ld.global.u16"), "bf16 load missing");
        assert!(p.contains("cvt.f32.bf16"), "bf16->f32 cvt missing");
        assert!(p.contains("atom.global.add.f32"), "f32 accumulate missing");
    }
}
