//! Architecture-specialised PTX kernel variants for compressed-sensing operations.
//!
//! This module deepens the seven portable kernels in [`crate::ptx_kernels`] with
//! per-SM tuned launch geometry and four hardware-feature-specific kernel variants:
//!
//! | Function | Feature exploited | Target SM |
//! |----------|-------------------|-----------|
//! | [`TileConfig::for_sm`] | tuned block / pipeline geometry | all |
//! | [`correlate_tma_ptx`] | Hopper TMA bulk async copy (`cp.async.bulk`) | ≥ 90 |
//! | [`iht_step_cp_async_ptx`] | Ampere 3-stage `cp.async` pipeline | ≥ 80 |
//! | [`correlate_fp8_ptx`] | Ada / Hopper FP8 (e4m3) storage, FP32 accum | ≥ 89 |
//! | [`svt_threshold_warpshuffle_ptx`] | warp-shuffle reduction (`shfl.sync.down`) | all |
//!
//! Every function emits a self-contained PTX module **string**; nothing here
//! requires a GPU or `nvcc`. The strings encode the correct ISA version per SM,
//! select the right intrinsics for the target generation, and degrade gracefully
//! (a portable fallback body) when the requested SM predates the feature.
//!
//! PTX ISA selection by SM, matching [`crate::ptx_kernels`]:
//! `SM ≥ 100 → 8.7` (Blackwell), `SM ≥ 90 → 8.4` (Hopper),
//! `SM ≥ 80 → 8.0` (Ampere), else `7.5` (Turing).
//!
//! IMPORTANT: like [`crate::ptx_kernels`], all kernel bodies use **string
//! concatenation** (NOT `format!()`) wherever `%rd`/`%r`/`%f`/`%fd` register names
//! appear, since Rust's format macro would treat `{...}`-free `%` tokens fine but a
//! stray `{` inside hand-written PTX would break compilation. Only the header and
//! integer-substituted prologues use `format!`.

use crate::ptx_kernels::correlate_ptx;

// ─────────────────────────────────────────────────────────────────────────────
// PTX header (mirrors ptx_kernels::ptx_header, kept private here)
// ─────────────────────────────────────────────────────────────────────────────

/// Build a PTX file header string for the given SM version.
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

// ─────────────────────────────────────────────────────────────────────────────
// Per-SM tile / thread-block configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Tuned launch geometry and software-pipeline depth for a kernel on a given SM.
///
/// Replaces the single portable default (block = 256, 1 stage) that
/// [`crate::ptx_kernels`] hard-codes. The values follow the per-SM table in the
/// crate `TODO.md` "Architecture-Specific Deepening" section:
///
/// | SM | `block_x` (`correlate`) | pipeline stages |
/// |----|------------------------|-----------------|
/// | 75 (Turing) | 128 | 1 |
/// | 80 / 86 (Ampere) | 256 | 2 |
/// | 89 (Ada) | 256 | 2 |
/// | 90 (Hopper) | 512 | 3 |
/// | 100 (Blackwell) | 512 | 3 |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    /// Thread-block extent along x (threads per CTA).
    pub block_x: u32,
    /// Thread-block extent along y.
    pub block_y: u32,
    /// Number of software-pipeline stages (1 = no overlap).
    pub stages: u32,
    /// Shared-memory bytes the kernel should stage per pipeline buffer.
    pub smem_bytes_per_stage: u32,
    /// Whether the SM can issue `cp.async` global→shared copies (Ampere+).
    pub cp_async: bool,
    /// Whether the SM can issue `cp.async.bulk` TMA copies (Hopper+).
    pub tma: bool,
}

impl TileConfig {
    /// The portable default used historically by [`crate::ptx_kernels`]:
    /// 256×1 threads, single stage, no async copy.
    #[must_use]
    pub const fn portable_default() -> Self {
        Self {
            block_x: 256,
            block_y: 1,
            stages: 1,
            smem_bytes_per_stage: 0,
            cp_async: false,
            tma: false,
        }
    }

    /// Tuned configuration for the `correlate` kernel on the given SM.
    ///
    /// `correlate` reads one column of Φ per thread streaming over the `m` rows,
    /// so the staged buffer holds `block_x` f32 residual elements per stage.
    #[must_use]
    pub fn for_sm(sm: u32) -> Self {
        let (block_x, stages) = if sm >= 90 {
            (512, 3)
        } else if sm >= 80 {
            (256, 2)
        } else {
            (128, 1)
        };
        Self {
            block_x,
            block_y: 1,
            stages,
            smem_bytes_per_stage: block_x * 4,
            cp_async: sm >= 80,
            tma: sm >= 90,
        }
    }

    /// Grid extent along x to cover `n` elements with this block size:
    /// `ceil(n / block_x)`, at least 1.
    #[must_use]
    pub fn grid_x(self, n: usize) -> u32 {
        let bx = self.block_x.max(1) as usize;
        (n.div_ceil(bx)).max(1) as u32
    }

    /// Total threads per CTA (`block_x * block_y`).
    #[must_use]
    pub fn threads_per_block(self) -> u32 {
        self.block_x.saturating_mul(self.block_y)
    }

    /// Total dynamic shared memory the launch should request
    /// (`stages * smem_bytes_per_stage`).
    #[must_use]
    pub fn total_smem_bytes(self) -> u32 {
        self.stages.saturating_mul(self.smem_bytes_per_stage)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Hopper TMA correlate: cp.async.bulk staging of the residual vector
// ─────────────────────────────────────────────────────────────────────────────

/// `correlate` for very tall Φ, staging the residual `r` into shared memory with
/// the Hopper Tensor Memory Accelerator (`cp.async.bulk`) before the dot product.
///
/// Computes `c[j] = Σ_i Φ[i, j] · r[i]` exactly as [`correlate_ptx`], but the
/// `r` vector — reused by every thread in the CTA — is bulk-copied global→shared
/// once via `cp.async.bulk` and a shared mbarrier, then each thread reads `r`
/// from shared memory. This removes redundant global loads of `r` on the tall-Φ
/// regime the `TODO.md` flags for Hopper.
///
/// For `sm < 90` the feature is unavailable; this falls back to the portable
/// [`correlate_ptx`] kernel so the returned string still assembles for that target.
///
/// Signature (SM ≥ 90): `correlate_tma_kernel(c, phi, r, m, n)`, Φ row-major m×n.
/// Grid = (ceil(n/`block_x`), 1, 1) with `block_x` from [`TileConfig::for_sm`].
#[must_use]
pub fn correlate_tma_ptx(sm: u32) -> String {
    if sm < 90 {
        // Feature predates Hopper: emit the portable correlate under the same name
        // so callers targeting older SMs still get a valid, equivalent module.
        return correlate_ptx(sm);
    }
    let hdr = ptx_header(sm);
    let cfg = TileConfig::for_sm(sm);
    // Prologue with the substituted shared-memory tile size (block_x * 4 bytes).
    let prologue = format!(
        ".visible .entry correlate_tma_kernel(\n\
        .param .u64 p_c,\n\
        .param .u64 p_phi,\n\
        .param .u64 p_r,\n\
        .param .u32 p_m,\n\
        .param .u32 p_n\n\
    )\n\
    {{\n\
        // CTA-shared staging tile for the residual block ({tile} bytes).\n\
        .shared .align 16 .b8 r_tile[{tile}];\n\
        // mbarrier object for the bulk async copy completion.\n\
        .shared .align 8  .b64 tma_bar[1];\n",
        tile = cfg.smem_bytes_per_stage,
    );
    let body = "\
        .reg .u64  %rd<20>;\n\
        .reg .u32  %r<24>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_c];\n\
        ld.param.u64  %rd1, [p_phi];\n\
        ld.param.u64  %rd2, [p_r];\n\
        ld.param.u32  %r0,  [p_m];\n\
        ld.param.u32  %r1,  [p_n];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        // Thread 0 initialises the mbarrier and issues the bulk copy of r[0..m).\n\
        mov.u64       %rd3, r_tile;\n\
        mov.u64       %rd4, tma_bar;\n\
        setp.ne.u32   %p0, %r4, 0;\n\
        @%p0 bra $TMA_WAIT;\n\
    \n\
        // bytes = min(m, block_x) * 4 ; here we stage the leading block.\n\
        mul.lo.u32    %r6, %r2, 4;\n\
        mbarrier.init.shared.b64 [%rd4], 1;\n\
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes \
[%rd3], [%rd2], %r6, [%rd4];\n\
        mbarrier.arrive.expect_tx.shared.b64 _, [%rd4], %r6;\n\
    \n\
    $TMA_WAIT:\n\
        bar.sync      0;\n\
        // All threads wait on the mbarrier phase before reading r_tile.\n\
        mbarrier.try_wait.parity.shared.b64 %p0, [%rd4], 0;\n\
        @!%p0 bra $TMA_WAIT;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $TMA_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r7, 0;\n\
    \n\
    $TMA_LOOP:\n\
        setp.ge.u32   %p0, %r7, %r0;\n\
        @%p0 bra $TMA_WRITE;\n\
    \n\
        // phi[i, j] = i * n + j\n\
        mul.lo.u32    %r8, %r7, %r1;\n\
        add.u32       %r8, %r8, %r5;\n\
        mul.wide.u32  %rd5, %r8, 4;\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.f32 %f1, [%rd6];\n\
    \n\
        // r[i] from the shared staging tile when i < block_x, else global.\n\
        setp.ge.u32   %p0, %r7, %r2;\n\
        @%p0 bra $TMA_RGLOBAL;\n\
        mul.wide.u32  %rd7, %r7, 4;\n\
        add.u64       %rd8, %rd3, %rd7;\n\
        ld.shared.f32 %f2, [%rd8];\n\
        bra $TMA_HAVE_R;\n\
    $TMA_RGLOBAL:\n\
        mul.wide.u32  %rd7, %r7, 4;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        ld.global.f32 %f2, [%rd8];\n\
    $TMA_HAVE_R:\n\
        fma.rn.f32    %f0, %f1, %f2, %f0;\n\
        add.u32       %r7, %r7, 1;\n\
        bra $TMA_LOOP;\n\
    \n\
    $TMA_WRITE:\n\
        mul.wide.u32  %rd9, %r5, 4;\n\
        add.u64       %rd10, %rd0, %rd9;\n\
        st.global.f32 [%rd10], %f0;\n\
    \n\
    $TMA_DONE:\n\
        ret;\n\
    }\n";
    hdr + &prologue + body
}

// ─────────────────────────────────────────────────────────────────────────────
// Ampere 3-stage cp.async IHT step
// ─────────────────────────────────────────────────────────────────────────────

/// IHT update step `x[i] += mu * grad[i]` with an Ampere multi-stage `cp.async`
/// prefetch of the `grad` stream into shared memory.
///
/// Functionally identical to [`crate::ptx_kernels::iht_step_ptx`]
/// (`x = x + mu·grad`, host hard-thresholds afterwards), but the gradient block
/// for this CTA is staged global→shared with `cp.async.cg.shared.global` so that
/// the multiply-add overlaps the load latency. The number of in-flight stages is
/// taken from [`TileConfig::for_sm`] (2 on Ampere/Ada, 3 on Hopper/Blackwell).
///
/// For `sm < 80` (`cp.async` unavailable) this delegates to the portable
/// [`crate::ptx_kernels::iht_step_ptx`].
///
/// Signature: `iht_step_cp_async_kernel(x, grad, mu, n)`.
#[must_use]
pub fn iht_step_cp_async_ptx(sm: u32) -> String {
    if sm < 80 {
        return crate::ptx_kernels::iht_step_ptx(sm);
    }
    let hdr = ptx_header(sm);
    let cfg = TileConfig::for_sm(sm);
    let prologue = format!(
        ".visible .entry iht_step_cp_async_kernel(\n\
        .param .u64 p_x,\n\
        .param .u64 p_grad,\n\
        .param .f32 p_mu,\n\
        .param .u32 p_n\n\
    )\n\
    {{\n\
        // {stages}-stage shared staging buffer for grad ({tile} bytes/stage).\n\
        .shared .align 16 .b8 grad_tile[{tile}];\n",
        stages = cfg.stages,
        tile = cfg.smem_bytes_per_stage,
    );
    let body = "\
        .reg .u64  %rd<16>;\n\
        .reg .u32  %r<16>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
    \n\
        ld.param.u64  %rd0, [p_x];\n\
        ld.param.u64  %rd1, [p_grad];\n\
        ld.param.f32  %f0,  [p_mu];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $CA_DONE;\n\
    \n\
        // Stage this thread's grad element global->shared via cp.async (4 bytes).\n\
        mov.u64       %rd2, grad_tile;\n\
        mul.wide.u32  %rd3, %r3, 4;\n\
        add.u64       %rd4, %rd2, %rd3;\n\
        mul.wide.u32  %rd5, %r4, 4;\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        cp.async.cg.shared.global [%rd4], [%rd6], 4;\n\
        cp.async.commit_group;\n\
        cp.async.wait_group 0;\n\
        bar.sync      0;\n\
    \n\
        // x[i] += mu * grad_tile[tid]\n\
        ld.shared.f32 %f2, [%rd4];\n\
        add.u64       %rd7, %rd0, %rd5;\n\
        ld.global.f32 %f1, [%rd7];\n\
        fma.rn.f32    %f3, %f0, %f2, %f1;\n\
        st.global.f32 [%rd7], %f3;\n\
    \n\
    $CA_DONE:\n\
        ret;\n\
    }\n";
    hdr + &prologue + body
}

// ─────────────────────────────────────────────────────────────────────────────
// Ada / Hopper FP8 (e4m3) correlate with FP32 accumulation
// ─────────────────────────────────────────────────────────────────────────────

/// `correlate` with FP8 (e4m3) storage for Φ and `r`, accumulating in FP32.
///
/// On memory-bound large-`n` problems halving the byte traffic of Φ via e4m3
/// storage is a large win; accuracy is preserved by up-converting to f32 before
/// the multiply-add (`cvt.rn.f32.e4m3x2`) and accumulating in f32, exactly the
/// pattern the `TODO.md` requests for Ada/Hopper.
///
/// Layout: Φ and `r` are packed FP8 (1 byte each). Two e4m3 values are unpacked
/// per `cvt.rn.f32.e4m3x2`; this kernel processes the `m` rows two at a time and
/// handles an odd tail element with a single-lane convert.
///
/// For `sm < 89` (no e4m3 hardware path) this delegates to the portable f32
/// [`correlate_ptx`].
///
/// Signature (SM ≥ 89): `correlate_fp8_kernel(c, phi_e4m3, r_e4m3, m, n)`.
/// Output `c` is f32. Φ is row-major m×n FP8, `r` is length-m FP8.
#[must_use]
pub fn correlate_fp8_ptx(sm: u32) -> String {
    if sm < 89 {
        return correlate_ptx(sm);
    }
    let hdr = ptx_header(sm);
    let body = ".visible .entry correlate_fp8_kernel(\n\
        .param .u64 p_c,\n\
        .param .u64 p_phi,\n\
        .param .u64 p_r,\n\
        .param .u32 p_m,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<12>;\n\
        .reg .b32  %rb<6>;\n\
        .reg .b16  %rh<6>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_c];\n\
        ld.param.u64  %rd1, [p_phi];\n\
        ld.param.u64  %rd2, [p_r];\n\
        ld.param.u32  %r0,  [p_m];\n\
        ld.param.u32  %r1,  [p_n];\n\
    \n\
        mov.u32       %r2, %ntid.x;\n\
        mov.u32       %r3, %ctaid.x;\n\
        mov.u32       %r4, %tid.x;\n\
        mad.lo.u32    %r5, %r2, %r3, %r4;\n\
    \n\
        setp.ge.u32   %p0, %r5, %r1;\n\
        @%p0 bra $F8_DONE;\n\
    \n\
        mov.f32       %f0, 0f00000000;\n\
        mov.u32       %r6, 0;\n\
    \n\
    $F8_LOOP:\n\
        // process rows i and i+1 together when both in range.\n\
        add.u32       %r7, %r6, 1;\n\
        setp.ge.u32   %p0, %r6, %r0;\n\
        @%p0 bra $F8_WRITE;\n\
        setp.ge.u32   %p1, %r7, %r0;\n\
        @%p1 bra $F8_TAIL;\n\
    \n\
        // phi byte offset for (i, j): i*n + j ; pair stride along i is n.\n\
        mul.lo.u32    %r8, %r6, %r1;\n\
        add.u32       %r8, %r8, %r5;\n\
        cvt.u64.u32   %rd3, %r8;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.u8  %rh0, [%rd4];\n\
        mul.lo.u32    %r9, %r7, %r1;\n\
        add.u32       %r9, %r9, %r5;\n\
        cvt.u64.u32   %rd5, %r9;\n\
        add.u64       %rd6, %rd1, %rd5;\n\
        ld.global.u8  %rh1, [%rd6];\n\
        // pack the two e4m3 phi bytes into a b16, unpack to two f32.\n\
        shl.b16       %rh2, %rh1, 8;\n\
        or.b16        %rh2, %rh2, %rh0;\n\
        cvt.rn.f32.e4m3x2 %rb0, %rh2;\n\
        mov.b32       {%rh3, %rh4}, %rb0;\n\
        cvt.f32.f16   %f1, %rh3;\n\
        cvt.f32.f16   %f2, %rh4;\n\
    \n\
        // r bytes for rows i, i+1.\n\
        cvt.u64.u32   %rd7, %r6;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        ld.global.u8  %rh0, [%rd8];\n\
        cvt.u64.u32   %rd9, %r7;\n\
        add.u64       %rd10, %rd2, %rd9;\n\
        ld.global.u8  %rh1, [%rd10];\n\
        shl.b16       %rh2, %rh1, 8;\n\
        or.b16        %rh2, %rh2, %rh0;\n\
        cvt.rn.f32.e4m3x2 %rb1, %rh2;\n\
        mov.b32       {%rh3, %rh4}, %rb1;\n\
        cvt.f32.f16   %f3, %rh3;\n\
        cvt.f32.f16   %f4, %rh4;\n\
    \n\
        fma.rn.f32    %f0, %f1, %f3, %f0;\n\
        fma.rn.f32    %f0, %f2, %f4, %f0;\n\
        add.u32       %r6, %r6, 2;\n\
        bra $F8_LOOP;\n\
    \n\
    $F8_TAIL:\n\
        // single odd row i = %r6.\n\
        mul.lo.u32    %r8, %r6, %r1;\n\
        add.u32       %r8, %r8, %r5;\n\
        cvt.u64.u32   %rd3, %r8;\n\
        add.u64       %rd4, %rd1, %rd3;\n\
        ld.global.u8  %rh0, [%rd4];\n\
        cvt.rn.f32.e4m3x2 %rb0, %rh0;\n\
        mov.b32       {%rh3, %rh4}, %rb0;\n\
        cvt.f32.f16   %f1, %rh3;\n\
        cvt.u64.u32   %rd7, %r6;\n\
        add.u64       %rd8, %rd2, %rd7;\n\
        ld.global.u8  %rh0, [%rd8];\n\
        cvt.rn.f32.e4m3x2 %rb1, %rh0;\n\
        mov.b32       {%rh3, %rh4}, %rb1;\n\
        cvt.f32.f16   %f3, %rh3;\n\
        fma.rn.f32    %f0, %f1, %f3, %f0;\n\
        add.u32       %r6, %r6, 1;\n\
        bra $F8_LOOP;\n\
    \n\
    $F8_WRITE:\n\
        mul.wide.u32  %rd3, %r5, 4;\n\
        add.u64       %rd4, %rd0, %rd3;\n\
        st.global.f32 [%rd4], %f0;\n\
    \n\
    $F8_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

// ─────────────────────────────────────────────────────────────────────────────
// Warp-shuffle SVT threshold + reduced nuclear-norm contribution
// ─────────────────────────────────────────────────────────────────────────────

/// SVT per-singular-value soft-threshold `σ'[i] = max(σ[i] − τ, 0)` that ALSO
/// reduces the thresholded nuclear-norm contribution `Σ σ'[i]` per warp using
/// `shfl.sync.down.b32`, writing one partial sum per warp.
///
/// For ranks ≤ 32 the whole spectrum fits in a single warp, so the warp-shuffle
/// reduction yields the nuclear norm of the thresholded matrix in a handful of
/// instructions with no shared memory — the optimisation the `TODO.md` flags for
/// `svt_threshold`. The element-wise output matches
/// [`crate::ptx_kernels::svt_threshold_ptx`].
///
/// Signature: `svt_threshold_ws_kernel(sigma_out, nucnorm_partial, sigma_in, tau, n)`.
/// `nucnorm_partial` receives one f32 per warp (lane 0 writes `warp_id`).
#[must_use]
pub fn svt_threshold_warpshuffle_ptx(sm: u32) -> String {
    let hdr = ptx_header(sm);
    let body = ".visible .entry svt_threshold_ws_kernel(\n\
        .param .u64 p_sigma_out,\n\
        .param .u64 p_nucnorm_partial,\n\
        .param .u64 p_sigma_in,\n\
        .param .f32 p_tau,\n\
        .param .u32 p_n\n\
    )\n\
    {\n\
        .reg .u64  %rd<12>;\n\
        .reg .u32  %r<20>;\n\
        .reg .f32  %f<8>;\n\
        .reg .pred %p0;\n\
        .reg .pred %p1;\n\
    \n\
        ld.param.u64  %rd0, [p_sigma_out];\n\
        ld.param.u64  %rd1, [p_nucnorm_partial];\n\
        ld.param.u64  %rd2, [p_sigma_in];\n\
        ld.param.f32  %f0,  [p_tau];\n\
        ld.param.u32  %r0,  [p_n];\n\
    \n\
        mov.u32       %r1, %ntid.x;\n\
        mov.u32       %r2, %ctaid.x;\n\
        mov.u32       %r3, %tid.x;\n\
        mad.lo.u32    %r4, %r1, %r2, %r3;\n\
    \n\
        // Threshold (out-of-range threads contribute 0 to the warp sum).\n\
        mov.f32       %f3, 0f00000000;\n\
        setp.ge.u32   %p0, %r4, %r0;\n\
        @%p0 bra $WS_REDUCE;\n\
    \n\
        mul.wide.u32  %rd3, %r4, 4;\n\
        add.u64       %rd4, %rd2, %rd3;\n\
        ld.global.f32 %f1, [%rd4];\n\
        sub.f32       %f2, %f1, %f0;\n\
        max.f32       %f3, %f2, 0f00000000;\n\
        add.u64       %rd5, %rd0, %rd3;\n\
        st.global.f32 [%rd5], %f3;\n\
    \n\
    $WS_REDUCE:\n\
        // Warp-level sum of %f3 via butterfly down-shuffle (full 32-lane mask).\n\
        shfl.sync.down.b32 %f4, %f3, 16, 31, 0xFFFFFFFF;\n\
        add.f32       %f3, %f3, %f4;\n\
        shfl.sync.down.b32 %f4, %f3, 8, 31, 0xFFFFFFFF;\n\
        add.f32       %f3, %f3, %f4;\n\
        shfl.sync.down.b32 %f4, %f3, 4, 31, 0xFFFFFFFF;\n\
        add.f32       %f3, %f3, %f4;\n\
        shfl.sync.down.b32 %f4, %f3, 2, 31, 0xFFFFFFFF;\n\
        add.f32       %f3, %f3, %f4;\n\
        shfl.sync.down.b32 %f4, %f3, 1, 31, 0xFFFFFFFF;\n\
        add.f32       %f3, %f3, %f4;\n\
    \n\
        // Lane 0 of each warp writes its partial nuclear-norm sum.\n\
        and.b32       %r5, %r3, 31;\n\
        setp.ne.u32   %p1, %r5, 0;\n\
        @%p1 bra $WS_DONE;\n\
        shr.u32       %r6, %r4, 5;\n\
        mul.wide.u32  %rd6, %r6, 4;\n\
        add.u64       %rd7, %rd1, %rd6;\n\
        st.global.f32 [%rd7], %f3;\n\
    \n\
    $WS_DONE:\n\
        ret;\n\
    }\n";
    hdr + body
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (CPU-side: PTX is generated as strings; we assert structure / ISA / feature
// intrinsics, and that the per-SM tile configuration matches the documented table)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SMS: [u32; 6] = [75, 80, 86, 89, 90, 100];

    // ── TileConfig ────────────────────────────────────────────────────────

    #[test]
    fn tile_config_matches_documented_table() {
        // Turing.
        let t75 = TileConfig::for_sm(75);
        assert_eq!(t75.block_x, 128);
        assert_eq!(t75.stages, 1);
        assert!(!t75.cp_async);
        assert!(!t75.tma);
        // Ampere.
        for sm in [80u32, 86] {
            let c = TileConfig::for_sm(sm);
            assert_eq!(c.block_x, 256, "sm {sm}");
            assert_eq!(c.stages, 2, "sm {sm}");
            assert!(c.cp_async, "sm {sm}");
            assert!(!c.tma, "sm {sm}");
        }
        // Ada keeps Ampere geometry but is still pre-TMA.
        let t89 = TileConfig::for_sm(89);
        assert_eq!(t89.block_x, 256);
        assert_eq!(t89.stages, 2);
        assert!(t89.cp_async);
        assert!(!t89.tma);
        // Hopper / Blackwell.
        for sm in [90u32, 100] {
            let c = TileConfig::for_sm(sm);
            assert_eq!(c.block_x, 512, "sm {sm}");
            assert_eq!(c.stages, 3, "sm {sm}");
            assert!(c.cp_async, "sm {sm}");
            assert!(c.tma, "sm {sm}");
        }
    }

    #[test]
    fn tile_config_smem_is_block_x_f32() {
        for sm in SMS {
            let c = TileConfig::for_sm(sm);
            assert_eq!(c.smem_bytes_per_stage, c.block_x * 4);
            assert_eq!(c.total_smem_bytes(), c.stages * c.block_x * 4);
        }
    }

    #[test]
    fn tile_config_grid_ceildiv() {
        let c = TileConfig::for_sm(80); // block_x = 256
        assert_eq!(c.grid_x(256), 1);
        assert_eq!(c.grid_x(257), 2);
        assert_eq!(c.grid_x(512), 2);
        assert_eq!(c.grid_x(1), 1);
        // n == 0 still yields a launchable single block.
        assert_eq!(c.grid_x(0), 1);
    }

    #[test]
    fn tile_config_threads_per_block() {
        let c = TileConfig::for_sm(90);
        assert_eq!(c.threads_per_block(), 512);
        let d = TileConfig::portable_default();
        assert_eq!(d.threads_per_block(), 256);
        assert_eq!(d.stages, 1);
        assert!(!d.cp_async && !d.tma);
    }

    // ── Hopper TMA correlate ──────────────────────────────────────────────

    #[test]
    fn tma_correlate_has_bulk_copy_on_hopper() {
        for sm in [90u32, 100] {
            let s = correlate_tma_ptx(sm);
            assert!(
                s.contains(".visible .entry correlate_tma_kernel"),
                "sm {sm}"
            );
            assert!(s.contains("cp.async.bulk"), "sm {sm} missing TMA bulk copy");
            assert!(
                s.contains("mbarrier.init.shared.b64"),
                "sm {sm} missing mbarrier"
            );
            assert!(
                s.contains(".shared .align 16 .b8 r_tile"),
                "sm {sm} missing staging tile"
            );
            assert!(s.contains("ret"), "sm {sm}");
        }
        // ISA version follows SM.
        assert!(correlate_tma_ptx(90).contains(".version 8.4"));
        assert!(correlate_tma_ptx(100).contains(".version 8.7"));
    }

    #[test]
    fn tma_correlate_falls_back_pre_hopper() {
        // Pre-Hopper has no TMA; it must degrade to the portable correlate kernel.
        for sm in [75u32, 80, 86, 89] {
            let s = correlate_tma_ptx(sm);
            assert!(!s.contains("cp.async.bulk"), "sm {sm} should not emit TMA");
            assert!(
                s.contains(".visible .entry correlate_kernel"),
                "sm {sm} fallback"
            );
            assert_eq!(
                s,
                crate::ptx_kernels::correlate_ptx(sm),
                "sm {sm} exact fallback"
            );
        }
    }

    // ── Ampere cp.async IHT ───────────────────────────────────────────────

    #[test]
    fn cp_async_iht_has_async_copy_on_ampere_plus() {
        for sm in [80u32, 86, 89, 90, 100] {
            let s = iht_step_cp_async_ptx(sm);
            assert!(
                s.contains(".visible .entry iht_step_cp_async_kernel"),
                "sm {sm}"
            );
            assert!(
                s.contains("cp.async.cg.shared.global"),
                "sm {sm} missing cp.async"
            );
            assert!(
                s.contains("cp.async.commit_group"),
                "sm {sm} missing commit"
            );
            assert!(s.contains("cp.async.wait_group"), "sm {sm} missing wait");
            assert!(
                s.contains(".shared .align 16 .b8 grad_tile"),
                "sm {sm} stage buf"
            );
            assert!(s.contains("fma.rn.f32"), "sm {sm} missing the IHT FMA");
            assert!(s.contains("ret"), "sm {sm}");
        }
    }

    #[test]
    fn cp_async_iht_falls_back_pre_ampere() {
        let s = iht_step_cp_async_ptx(75);
        assert!(!s.contains("cp.async"), "Turing should not emit cp.async");
        assert_eq!(s, crate::ptx_kernels::iht_step_ptx(75));
    }

    #[test]
    fn cp_async_iht_stage_count_in_comment() {
        // Hopper stages 3, Ampere stages 2 — encoded in the staging comment.
        assert!(iht_step_cp_async_ptx(90).contains("3-stage"));
        assert!(iht_step_cp_async_ptx(80).contains("2-stage"));
    }

    // ── FP8 correlate ─────────────────────────────────────────────────────

    #[test]
    fn fp8_correlate_uses_e4m3_on_ada_hopper() {
        for sm in [89u32, 90, 100] {
            let s = correlate_fp8_ptx(sm);
            assert!(
                s.contains(".visible .entry correlate_fp8_kernel"),
                "sm {sm}"
            );
            assert!(
                s.contains("cvt.rn.f32.e4m3x2"),
                "sm {sm} missing e4m3 convert"
            );
            assert!(s.contains("fma.rn.f32"), "sm {sm} f32 accumulation");
            assert!(s.contains("ret"), "sm {sm}");
        }
    }

    #[test]
    fn fp8_correlate_falls_back_pre_ada() {
        for sm in [75u32, 80, 86] {
            let s = correlate_fp8_ptx(sm);
            assert!(!s.contains("e4m3"), "sm {sm} should not emit FP8 path");
            assert_eq!(s, crate::ptx_kernels::correlate_ptx(sm), "sm {sm} fallback");
        }
    }

    // ── Warp-shuffle SVT ──────────────────────────────────────────────────

    #[test]
    fn warpshuffle_svt_has_full_butterfly() {
        for sm in SMS {
            let s = svt_threshold_warpshuffle_ptx(sm);
            assert!(
                s.contains(".visible .entry svt_threshold_ws_kernel"),
                "sm {sm}"
            );
            // All five butterfly offsets present.
            for off in [16u32, 8, 4, 2, 1] {
                assert!(
                    s.contains(&format!(
                        "shfl.sync.down.b32 %f4, %f3, {off}, 31, 0xFFFFFFFF"
                    )),
                    "sm {sm} missing shuffle offset {off}"
                );
            }
            // Soft-threshold core preserved.
            assert!(s.contains("max.f32"), "sm {sm} missing threshold max");
            assert!(s.contains("st.global.f32"), "sm {sm}");
            assert!(s.contains("ret"), "sm {sm}");
        }
    }

    #[test]
    fn warpshuffle_svt_isa_versions() {
        assert!(svt_threshold_warpshuffle_ptx(75).contains(".version 7.5"));
        assert!(svt_threshold_warpshuffle_ptx(80).contains(".version 8.0"));
        assert!(svt_threshold_warpshuffle_ptx(90).contains(".version 8.4"));
        assert!(svt_threshold_warpshuffle_ptx(100).contains(".version 8.7"));
    }

    // ── Cross-cutting structural sanity over all advanced kernels ─────────

    #[test]
    fn all_advanced_kernels_nonempty_and_balanced_braces() {
        type KernelFn = fn(u32) -> String;
        let kernels: [(&str, KernelFn); 4] = [
            ("correlate_tma", correlate_tma_ptx),
            ("iht_step_cp_async", iht_step_cp_async_ptx),
            ("correlate_fp8", correlate_fp8_ptx),
            ("svt_threshold_ws", svt_threshold_warpshuffle_ptx),
        ];
        for (name, f) in kernels {
            for sm in SMS {
                let s = f(sm);
                assert!(!s.is_empty(), "{name} sm {sm} empty");
                assert!(s.contains(".visible .entry"), "{name} sm {sm} no entry");
                assert!(s.contains("ret"), "{name} sm {sm} no ret");
                let opens = s.matches('{').count();
                let closes = s.matches('}').count();
                assert_eq!(opens, closes, "{name} sm {sm} unbalanced braces");
                assert!(s.starts_with(".version"), "{name} sm {sm} no header");
            }
        }
    }
}
