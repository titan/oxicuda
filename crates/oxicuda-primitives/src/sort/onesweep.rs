//! Onesweep radix sort — single-kernel-per-pass decoupled-lookback scatter.
//!
//! The classic radix pipeline ([`crate::sort::radix_sort`]) launches three
//! kernels per pass (count, scan, scatter).  CUB's **onesweep** removes the
//! middle scan kernel entirely: each scatter block resolves its global per-digit
//! offset on the fly via a *decoupled-lookback* chained scan over the
//! predecessor blocks' published per-digit aggregates.  One pass = one kernel
//! (plus a one-time global-histogram precursor shared by all passes).
//!
//! # Offset decomposition
//!
//! For an element with digit `d` in block `b`, its output position is:
//!
//! ```text
//! out_pos = global_base[d]            // start of digit d across the whole array
//!         + block_prefix[b][d]        // count of digit d in blocks 0..b  (lookback)
//!         + local_rank                // rank of this element among digit-d
//!                                     //   elements within block b
//! ```
//!
//! * `global_base[d]` is the exclusive prefix sum of the **global** digit
//!   histogram for this pass; the caller computes it once with
//!   [`crate::device::reduce`]-style counting or a `256`-bin histogram, then the
//!   exclusive scan of those `RADIX` counts.
//! * `block_prefix[b][d]` is obtained by the lookback loop reading predecessor
//!   blocks' published per-digit aggregates from a descriptor array.
//! * `local_rank` is a shared-memory atomic rank within the block.
//!
//! # Descriptor layout
//!
//! `status[b]` (one `u32` per block) and `agg[b * RADIX + d]` /
//! `prefix[b * RADIX + d]` (one slot per (block, digit)).  The status flags are
//! the same `X` / `A` / `P` as [`crate::device::decoupled_scan`], applied to the
//! whole per-block digit vector at once: a block publishes all `RADIX`
//! aggregates, fences, then flips its single status word.
//!
//! This template emits the **onesweep pass** kernel.  Only executing it on real
//! hardware (where inter-block lookback ordering is enforced) is GPU-gated; the
//! PTX generation and the CPU model below are fully testable.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::sort::onesweep::{OnesweepConfig, OnesweepTemplate};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = OnesweepConfig::new(PtxType::U32, 4, 256).expect("valid config");
//! let ptx = OnesweepTemplate::new(cfg).generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(ptx.contains("onesweep_pass_r4_u32_bs256"));
//! assert!(ptx.contains("OSW_LOOKBACK"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

/// Descriptor flag: nothing published.
pub const FLAG_X: u32 = 0;
/// Descriptor flag: per-digit aggregates available.
pub const FLAG_A: u32 = 1;
/// Descriptor flag: per-digit inclusive prefixes available.
pub const FLAG_P: u32 = 2;

/// Configuration for a onesweep radix pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OnesweepConfig {
    /// Key type (`U32` or `U64`).
    pub ty: PtxType,
    /// Radix bits per pass (`4` → 16 buckets, `8` → 256 buckets).
    pub radix_bits: u32,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl OnesweepConfig {
    /// Create a configuration, validating the key type, radix, and block size.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] if the key type is not
    /// `U32`/`U64`, `radix_bits` is not `4` or `8`, or `block_size` is invalid.
    pub fn new(ty: PtxType, radix_bits: u32, block_size: u32) -> PrimitivesResult<Self> {
        if !matches!(ty, PtxType::U32 | PtxType::U64) {
            return Err(PrimitivesError::InvalidArgument(format!(
                "radix key type must be U32 or U64, got {ty:?}"
            )));
        }
        if radix_bits != 4 && radix_bits != 8 {
            return Err(PrimitivesError::InvalidArgument(format!(
                "radix_bits must be 4 or 8, got {radix_bits}"
            )));
        }
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self {
            ty,
            radix_bits,
            block_size,
        })
    }

    /// Number of digit buckets (`2^radix_bits`).
    #[must_use]
    pub fn radix(&self) -> u32 {
        1 << self.radix_bits
    }

    /// Digit mask (`radix - 1`).
    #[must_use]
    pub fn digit_mask(&self) -> u32 {
        self.radix() - 1
    }

    /// Number of passes to fully sort the key type.
    #[must_use]
    pub fn passes(&self) -> u32 {
        let bits = if self.ty == PtxType::U64 { 64 } else { 32 };
        bits / self.radix_bits
    }

    /// Bytes per key.
    #[must_use]
    pub fn elem_bytes(&self) -> u32 {
        if self.ty == PtxType::U64 { 8 } else { 4 }
    }

    /// Number of thread blocks for `n` elements.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(u64::from(self.block_size))
    }

    /// Descriptor scratch bytes: `status[num_blocks]` (u32) plus
    /// `agg[num_blocks * radix]` and `prefix[num_blocks * radix]` (u32 each).
    #[must_use]
    pub fn descriptor_bytes(&self, n: u64) -> u64 {
        let nb = self.num_blocks(n);
        let radix = u64::from(self.radix());
        nb * 4 + nb * radix * 4 * 2
    }

    /// Generated kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "onesweep_pass_r{}_{}_bs{}",
            self.radix_bits,
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX generator for the onesweep radix-sort pass kernel.
pub struct OnesweepTemplate {
    /// Configuration.
    pub cfg: OnesweepConfig,
}

impl OnesweepTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: OnesweepConfig) -> Self {
        Self { cfg }
    }

    /// Generate the onesweep pass kernel.
    ///
    /// Params:
    /// `(keys_out, keys_in, global_base, status, agg, prefix, n, num_blocks, shift)`.
    /// `global_base` is the `radix`-length exclusive scan of the global digit
    /// histogram for this pass; the descriptor arrays are zero-initialised.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let radix = self.cfg.radix();
        let mask = self.cfg.digit_mask();
        let is64 = self.cfg.ty == PtxType::U64;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("onesweep_pass", e);

        let mut out = ptx_header(sm);
        // Shared: per-block digit histogram (radix bins) reused as rank counters.
        writeln!(out, ".shared .align 4 .u32 osw_hist[{radix}];").map_err(ferr)?;
        // Shared: resolved per-digit block-exclusive prefix.
        writeln!(out, ".shared .align 4 .u32 osw_bprefix[{radix}];").map_err(ferr)?;
        // Shared: each thread's digit, for a STABLE within-block local rank.
        writeln!(out, ".shared .align 4 .u32 osw_sdig[{bs}];").map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_keys_out,\n    \
             .param .u64 param_keys_in,\n    \
             .param .u64 param_global_base,\n    \
             .param .u64 param_status,\n    \
             .param .u64 param_agg,\n    \
             .param .u64 param_prefix,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_num_blocks,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %key;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %ltid, %bid, %nb, %shift, %digit, %old, %i, %d, %flag, %probe;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %agg_v, %pre_v, %base_v, %local_rank, %gbase, %cnt, %tp, %other;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_kin, %ptr_kout, %ptr_gbase;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_status, %ptr_agg, %ptr_prefix, %addr, %hist_addr;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %smem_h, %smem_p, %smem_d, %sdig_addr, %out64;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .pred   %p, %oob, %init_p, %is_first, %done, %eq;"
        )
        .map_err(ferr)?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(ferr)?;
        }

        writeln!(out, "    ld.param.u64 %ptr_kout,   [param_keys_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_kin,    [param_keys_in];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_gbase,  [param_global_base];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_status, [param_status];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_agg,    [param_agg];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_prefix, [param_prefix];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,           [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %nb,          [param_num_blocks];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %shift,       [param_shift];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_h, osw_hist;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_p, osw_bprefix;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_d, osw_sdig;").map_err(ferr)?;

        // Phase 1: zero the shared histogram (strided over radix bins).
        writeln!(out, "    mov.u32      %i, %ltid;").map_err(ferr)?;
        writeln!(out, "OSW_ZERO:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %init_p, %i, {radix};").map_err(ferr)?;
        writeln!(out, "    @%init_p bra OSW_ZERO_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %hist_addr, %i, 4, %smem_h;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%hist_addr], 0;").map_err(ferr)?;
        writeln!(out, "    add.u32      %i, %i, {bs};").map_err(ferr)?;
        writeln!(out, "    bra OSW_ZERO;").map_err(ferr)?;
        writeln!(out, "OSW_ZERO_DONE:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;

        // Phase 2: build the per-digit histogram and a STABLE local rank.
        //
        // The histogram counter (`atom.shared.add`) is only used for the block
        // AGGREGATE; its return value (the arbitrary atomic order) must NOT be
        // used as the element's local rank — that would make the sort unstable
        // and break the multi-pass LSD chain. Each thread instead caches its
        // digit and computes `local_rank = #{ t' < tid : digit[t'] == digit }`.
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %local_rank, 0;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %digit, 0;").map_err(ferr)?;
        writeln!(out, "    @%oob bra OSW_OOB_SDIG;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %key, [%addr];").map_err(ferr)?;
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(ferr)?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;").map_err(ferr)?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(ferr)?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(ferr)?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0x{mask:X};").map_err(ferr)?;
        // Cache digit and count it (atomic return discarded).
        writeln!(out, "    mad.wide.u32   %sdig_addr, %ltid, 4, %smem_d;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%sdig_addr], %digit;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %hist_addr, %digit, 4, %smem_h;").map_err(ferr)?;
        writeln!(out, "    atom.shared.add.u32 %old, [%hist_addr], 1;").map_err(ferr)?;
        writeln!(out, "    bra OSW_HIST_DONE;").map_err(ferr)?;
        writeln!(out, "OSW_OOB_SDIG:").map_err(ferr)?;
        // Out-of-range lanes store a sentinel that never matches a real digit.
        writeln!(out, "    mov.u32      %old, 0xFFFFFFFF;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %sdig_addr, %ltid, 4, %smem_d;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%sdig_addr], %old;").map_err(ferr)?;
        writeln!(out, "OSW_HIST_DONE:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        // Stable rank: count earlier lanes in this block sharing the same digit.
        writeln!(out, "    @%oob bra OSW_RANK_DONE;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tp, 0;").map_err(ferr)?;
        writeln!(out, "OSW_RANK:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %p, %tp, %ltid;").map_err(ferr)?;
        writeln!(out, "    @%p bra OSW_RANK_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %sdig_addr, %tp, 4, %smem_d;").map_err(ferr)?;
        writeln!(out, "    ld.shared.u32 %other, [%sdig_addr];").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %eq, %other, %digit;").map_err(ferr)?;
        writeln!(out, "    @%eq add.u32 %local_rank, %local_rank, 1;").map_err(ferr)?;
        writeln!(out, "    add.u32      %tp, %tp, 1;").map_err(ferr)?;
        writeln!(out, "    bra OSW_RANK;").map_err(ferr)?;
        writeln!(out, "OSW_RANK_DONE:").map_err(ferr)?;

        // Phase 3: thread 0 publishes per-digit aggregates and runs the lookback
        // per digit to fill osw_bprefix[d].  (Serial over digits and blocks; a
        // warp-parallel form is a perf refinement that does not change results.)
        writeln!(out, "    setp.ne.u32  %p, %ltid, 0;").map_err(ferr)?;
        writeln!(out, "    @%p bra OSW_APPLY;").map_err(ferr)?;

        // Publish this block's per-digit aggregates: agg[bid*radix + d] = hist[d].
        writeln!(out, "    mov.u32      %d, 0;").map_err(ferr)?;
        writeln!(out, "OSW_PUB:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %init_p, %d, {radix};").map_err(ferr)?;
        writeln!(out, "    @%init_p bra OSW_PUB_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %hist_addr, %d, 4, %smem_h;").map_err(ferr)?;
        writeln!(out, "    ld.shared.u32 %cnt, [%hist_addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u32   %i, %bid, {radix}, %d;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %addr, %i, 4, %ptr_agg;").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %cnt;").map_err(ferr)?;
        writeln!(out, "    add.u32      %d, %d, 1;").map_err(ferr)?;
        writeln!(out, "    bra OSW_PUB;").map_err(ferr)?;
        writeln!(out, "OSW_PUB_DONE:").map_err(ferr)?;
        writeln!(out, "    membar.gl;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %addr, %bid, 4, %ptr_status;").map_err(ferr)?;
        writeln!(out, "    atom.global.exch.b32 %flag, [%addr], {FLAG_A};").map_err(ferr)?;

        // For each digit d: walk predecessors to accumulate block-exclusive prefix.
        writeln!(out, "    setp.eq.u32  %is_first, %bid, 0;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %d, 0;").map_err(ferr)?;
        writeln!(out, "OSW_DIGIT:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %init_p, %d, {radix};").map_err(ferr)?;
        writeln!(out, "    @%init_p bra OSW_DIGIT_DONE;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %pre_v, 0;").map_err(ferr)?; // accumulated block prefix
        writeln!(out, "    @%is_first bra OSW_DIGIT_STORE;").map_err(ferr)?;
        writeln!(out, "    sub.u32      %probe, %bid, 1;").map_err(ferr)?;
        writeln!(out, "OSW_LOOKBACK:").map_err(ferr)?;
        writeln!(out, "OSW_LB_SPIN:").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %addr, %probe, 4, %ptr_status;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %flag, [%addr];").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %p, %flag, {FLAG_X};").map_err(ferr)?;
        writeln!(out, "    @%p bra OSW_LB_SPIN;").map_err(ferr)?;
        writeln!(out, "    membar.gl;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u32   %i, %probe, {radix}, %d;").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %done, %flag, {FLAG_P};").map_err(ferr)?;
        writeln!(out, "    @%done bra OSW_LB_P;").map_err(ferr)?;
        // A: add predecessor aggregate, keep walking.
        writeln!(out, "    mad.wide.u32   %addr, %i, 4, %ptr_agg;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %agg_v, [%addr];").map_err(ferr)?;
        writeln!(out, "    add.u32      %pre_v, %pre_v, %agg_v;").map_err(ferr)?;
        writeln!(out, "    setp.eq.u32  %p, %probe, 0;").map_err(ferr)?;
        writeln!(out, "    @%p bra OSW_DIGIT_STORE;").map_err(ferr)?;
        writeln!(out, "    sub.u32      %probe, %probe, 1;").map_err(ferr)?;
        writeln!(out, "    bra OSW_LOOKBACK;").map_err(ferr)?;
        writeln!(out, "OSW_LB_P:").map_err(ferr)?;
        // Inclusive prefix of this predecessor already covers all earlier blocks;
        // add it to the aggregates accumulated while walking past intervening
        // A-blocks, then stop.
        writeln!(out, "    mad.wide.u32   %addr, %i, 4, %ptr_prefix;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %agg_v, [%addr];").map_err(ferr)?;
        writeln!(out, "    add.u32      %pre_v, %pre_v, %agg_v;").map_err(ferr)?;
        writeln!(out, "OSW_DIGIT_STORE:").map_err(ferr)?;
        // Store block-exclusive prefix for digit d, and publish inclusive prefix.
        writeln!(out, "    mad.wide.u32   %hist_addr, %d, 4, %smem_p;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%hist_addr], %pre_v;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %hist_addr, %d, 4, %smem_h;").map_err(ferr)?;
        writeln!(out, "    ld.shared.u32 %cnt, [%hist_addr];").map_err(ferr)?;
        writeln!(out, "    add.u32      %agg_v, %pre_v, %cnt;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u32   %i, %bid, {radix}, %d;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %addr, %i, 4, %ptr_prefix;").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %agg_v;").map_err(ferr)?;
        writeln!(out, "    add.u32      %d, %d, 1;").map_err(ferr)?;
        writeln!(out, "    bra OSW_DIGIT;").map_err(ferr)?;
        writeln!(out, "OSW_DIGIT_DONE:").map_err(ferr)?;
        writeln!(out, "    membar.gl;").map_err(ferr)?;
        writeln!(out, "    mad.wide.u32   %addr, %bid, 4, %ptr_status;").map_err(ferr)?;
        writeln!(out, "    atom.global.exch.b32 %flag, [%addr], {FLAG_P};").map_err(ferr)?;

        // Phase 4: all threads scatter using global_base + block_prefix + rank.
        writeln!(out, "OSW_APPLY:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    @%oob ret;").map_err(ferr)?;
        // gbase = global_base[digit]
        writeln!(out, "    mad.wide.u32   %addr, %digit, 4, %ptr_gbase;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %gbase, [%addr];").map_err(ferr)?;
        // bprefix = osw_bprefix[digit]
        writeln!(out, "    mad.wide.u32   %hist_addr, %digit, 4, %smem_p;").map_err(ferr)?;
        writeln!(out, "    ld.shared.u32 %base_v, [%hist_addr];").map_err(ferr)?;
        writeln!(out, "    add.u32      %old, %gbase, %base_v;").map_err(ferr)?;
        writeln!(out, "    add.u32      %old, %old, %local_rank;").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %out64, %old;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %out64, {eb}, %ptr_kout;").map_err(ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %key;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU reference ─────────────────────────────────────────────────────────────

/// Host model of one onesweep pass over `u32` keys with the given `radix_bits`
/// and `block_size`.  Reproduces the exact offset decomposition
/// (`global_base + block_prefix + local_rank`) the GPU kernel computes, so the
/// output is a stable per-digit counting-sort pass.
#[must_use]
pub fn reference_onesweep_pass_u32(
    input: &[u32],
    shift: u32,
    radix_bits: u32,
    block_size: u32,
) -> Vec<u32> {
    let radix = 1usize << radix_bits;
    let mask = (radix as u32) - 1;
    let n = input.len();
    let nb = n.div_ceil(block_size as usize);

    // Global histogram → exclusive base offsets.
    let mut global_hist = vec![0u32; radix];
    for &k in input {
        global_hist[((k >> shift) & mask) as usize] += 1;
    }
    let mut global_base = vec![0u32; radix];
    let mut running = 0u32;
    for d in 0..radix {
        global_base[d] = running;
        running += global_hist[d];
    }

    // Per-block per-digit counts.
    let mut block_counts = vec![vec![0u32; radix]; nb];
    for (i, &k) in input.iter().enumerate() {
        let b = i / block_size as usize;
        block_counts[b][((k >> shift) & mask) as usize] += 1;
    }
    // block_prefix[b][d] = sum over b' < b of block_counts[b'][d].
    let mut block_prefix = vec![vec![0u32; radix]; nb];
    for d in 0..radix {
        let mut acc = 0u32;
        for b in 0..nb {
            block_prefix[b][d] = acc;
            acc += block_counts[b][d];
        }
    }

    // Scatter with local ranks.
    let mut out = vec![0u32; n];
    let mut local_rank = vec![vec![0u32; radix]; nb];
    for (i, &k) in input.iter().enumerate() {
        let b = i / block_size as usize;
        let d = ((k >> shift) & mask) as usize;
        let pos = global_base[d] + block_prefix[b][d] + local_rank[b][d];
        local_rank[b][d] += 1;
        out[pos as usize] = k;
    }
    out
}

/// Full onesweep sort over `u32` keys by chaining `reference_onesweep_pass_u32`.
#[must_use]
pub fn reference_onesweep_sort_u32(input: &[u32], radix_bits: u32, block_size: u32) -> Vec<u32> {
    let passes = 32 / radix_bits;
    let mut data = input.to_vec();
    for p in 0..passes {
        data = reference_onesweep_pass_u32(&data, p * radix_bits, radix_bits, block_size);
    }
    data
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    #[test]
    fn config_validation() {
        assert!(OnesweepConfig::new(PtxType::F32, 4, 256).is_err());
        assert!(OnesweepConfig::new(PtxType::U32, 5, 256).is_err());
        assert!(OnesweepConfig::new(PtxType::U32, 4, 100).is_err());
        let c = OnesweepConfig::new(PtxType::U32, 8, 256).expect("valid config");
        assert_eq!(c.radix(), 256);
        assert_eq!(c.digit_mask(), 0xFF);
        assert_eq!(c.passes(), 4);
    }

    #[test]
    fn passes_for_u64() {
        let c4 = OnesweepConfig::new(PtxType::U64, 4, 256).expect("valid config");
        assert_eq!(c4.passes(), 16);
        let c8 = OnesweepConfig::new(PtxType::U64, 8, 256).expect("valid config");
        assert_eq!(c8.passes(), 8);
    }

    #[test]
    fn descriptor_bytes_layout() {
        let c = OnesweepConfig::new(PtxType::U32, 4, 256).expect("valid config");
        // 600 elements → ceil(600/256)=3 blocks, radix 16.
        // status: 3*4=12 ; agg+prefix: 3*16*4*2 = 384 ; total 396.
        assert_eq!(c.num_blocks(600), 3);
        assert_eq!(c.descriptor_bytes(600), 12 + 384);
    }

    #[test]
    fn ptx_has_lookback_and_offset_decomposition() {
        let c = OnesweepConfig::new(PtxType::U32, 4, 256).expect("valid config");
        let ptx = OnesweepTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("onesweep_pass_r4_u32_bs256"), "PTX: {ptx}");
        assert!(ptx.contains("OSW_LOOKBACK"), "PTX: {ptx}");
        assert!(ptx.contains("OSW_LB_SPIN"), "PTX: {ptx}");
        assert!(ptx.contains("atom.global.exch.b32"), "PTX: {ptx}");
        assert!(ptx.contains("membar.gl"), "PTX: {ptx}");
        // offset = gbase + bprefix + local_rank
        assert!(
            ptx.contains("add.u32      %old, %gbase, %base_v"),
            "PTX: {ptx}"
        );
        assert!(
            ptx.contains("add.u32      %old, %old, %local_rank"),
            "PTX: {ptx}"
        );
        // 4-bit mask.
        assert!(
            ptx.contains("and.b32      %digit, %digit, 0xF"),
            "PTX: {ptx}"
        );
    }

    #[test]
    fn ptx_8bit_uses_ff_mask_and_256_bins() {
        let c = OnesweepConfig::new(PtxType::U32, 8, 256).expect("valid config");
        let ptx = OnesweepTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("osw_hist[256]"), "PTX: {ptx}");
        assert!(
            ptx.contains("and.b32      %digit, %digit, 0xFF"),
            "PTX: {ptx}"
        );
    }

    #[test]
    fn reference_single_pass_is_stable_counting_sort() {
        // Two-block input (block_size 4, n=8); check digit-0 (4-bit) pass.
        let data = [0x13u32, 0x21, 0x05, 0x13, 0x37, 0x21, 0x44, 0x13];
        let out = reference_onesweep_pass_u32(&data, 0, 4, 4);
        // Sorted by low nibble, stable within equal nibble.
        // nibbles: 3,1,5,3,7,1,4,3 → order of nibble values 1,1,3,3,3,4,5,7
        let nibbles: Vec<u32> = out.iter().map(|&k| k & 0xF).collect();
        assert_eq!(nibbles, vec![1, 1, 3, 3, 3, 4, 5, 7]);
        // Stability: the three 0x_3 (=0x13) entries keep relative order (all 0x13).
        assert_eq!(out[2], 0x13);
        assert_eq!(out[3], 0x13);
        assert_eq!(out[4], 0x13);
    }

    #[test]
    fn reference_full_sort_4bit_and_8bit() {
        let mut rng = 0x9E37_79B9u32;
        let mut data = Vec::new();
        for _ in 0..777 {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push(rng);
        }
        let mut expected = data.clone();
        expected.sort_unstable();

        let s4 = reference_onesweep_sort_u32(&data, 4, 64);
        assert_eq!(s4, expected, "4-bit onesweep");
        let s8 = reference_onesweep_sort_u32(&data, 8, 128);
        assert_eq!(s8, expected, "8-bit onesweep");
    }
}
