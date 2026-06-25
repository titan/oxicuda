//! Key+value, descending, and floating-point variants of LSD radix sort.
//!
//! The base [`crate::sort::radix_sort`] module sorts `u32` / `u64` keys in
//! ascending order.  This module extends it along three independent axes that
//! CUB's `DeviceRadixSort` also supports:
//!
//! 1. **Key+value pairs** — [`RadixPairsTemplate`] emits a scatter kernel that
//!    moves an associated value payload alongside each key, enabling
//!    sort-by-key for stable joins and gather-index permutation.
//! 2. **Descending order** — [`SortOrder::Descending`] inverts each 4-bit digit
//!    (`d → 15 - d`) consistently in the count and scatter kernels, producing a
//!    descending result directly (no post-pass reverse).
//! 3. **Floating-point keys** — [`FloatTwiddleTemplate`] emits the standard
//!    order-preserving bijection that reinterprets `f32` / `f64` as unsigned so
//!    the integer radix passes sort them correctly, plus the exact inverse for
//!    the final un-twiddle.
//!
//! The count and scan kernels for the key channel are unchanged from the base
//! module *except* for the digit inversion under descending order, so this
//! module re-emits a digit-inverting count kernel and pairs it with the base
//! scan kernel.
//!
//! # Float twiddle bijection
//!
//! For an IEEE-754 bit pattern `x` with the sign bit at the top:
//!
//! ```text
//! forward:  (x & sign_mask) != 0  ?  ~x        :  x | sign_mask
//! inverse:  (y & sign_mask) != 0  ?  y & ~sign :  ~y
//! ```
//!
//! This maps negative floats below positive floats while preserving the
//! ordering within each sign, exactly as `cub::DeviceRadixSort` does internally.

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

/// Number of digit buckets per radix pass (2^4 = 16).
pub const RADIX_SIZE: u32 = 16;

// ─── Sort order ────────────────────────────────────────────────────────────────

/// Ordering produced by the radix sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortOrder {
    /// Smallest key first (the base module's behaviour).
    Ascending,
    /// Largest key first.
    Descending,
}

impl SortOrder {
    fn name(self) -> &'static str {
        match self {
            Self::Ascending => "asc",
            Self::Descending => "desc",
        }
    }
}

// ─── Key+value config ──────────────────────────────────────────────────────────

/// Configuration for a key+value radix-sort pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RadixPairsConfig {
    /// Key type (`U32` or `U64`).
    pub key_ty: PtxType,
    /// Value (payload) type.
    pub val_ty: PtxType,
    /// Ordering.
    pub order: SortOrder,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl RadixPairsConfig {
    /// Create a configuration, validating the key type and block size.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] if the key type is not
    /// `U32`/`U64`, or `block_size` is not a power of two in `[32, 1024]`.
    pub fn new(
        key_ty: PtxType,
        val_ty: PtxType,
        order: SortOrder,
        block_size: u32,
    ) -> PrimitivesResult<Self> {
        if !matches!(key_ty, PtxType::U32 | PtxType::U64) {
            return Err(PrimitivesError::InvalidArgument(format!(
                "radix key type must be U32 or U64, got {key_ty:?}"
            )));
        }
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self {
            key_ty,
            val_ty,
            order,
            block_size,
        })
    }

    /// Number of radix passes for the key type.
    #[must_use]
    pub fn passes(&self) -> u32 {
        match self.key_ty {
            PtxType::U64 => 16,
            _ => 8,
        }
    }

    /// Bytes per key.
    #[must_use]
    pub fn key_bytes(&self) -> u32 {
        if self.key_ty == PtxType::U64 { 8 } else { 4 }
    }

    /// Bytes per value.
    #[must_use]
    pub fn val_bytes(&self) -> u32 {
        match self.val_ty {
            PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => 8,
            _ => 4,
        }
    }

    /// Kernel name for the (digit-inverting if descending) count pass.
    #[must_use]
    pub fn count_kernel_name(&self) -> String {
        format!(
            "radix_pairs_count_{}_{}_bs{}",
            self.order.name(),
            ptx_type_str(self.key_ty),
            self.block_size
        )
    }

    /// Kernel name for the key+value scatter pass.
    #[must_use]
    pub fn scatter_kernel_name(&self) -> String {
        format!(
            "radix_pairs_scatter_{}_{}_{}_bs{}",
            self.order.name(),
            ptx_type_str(self.key_ty),
            ptx_type_str(self.val_ty),
            self.block_size
        )
    }
}

/// PTX generator for key+value / descending radix sort passes.
pub struct RadixPairsTemplate {
    /// Configuration.
    pub cfg: RadixPairsConfig,
}

impl RadixPairsTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: RadixPairsConfig) -> Self {
        Self { cfg }
    }

    /// Generate `(count_ptx, scatter_ptx)`.  Pair these with the base
    /// [`crate::sort::radix_sort::RadixSortTemplate`] scan kernel.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<(String, String)> {
        Ok((
            self.generate_count_kernel(sm)?,
            self.generate_scatter_kernel(sm)?,
        ))
    }

    // ── Count: per-block histogram, optionally inverting the digit ───────────

    fn generate_count_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.count_kernel_name();
        let ty = ptx_type_str(self.cfg.key_ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.key_bytes();
        let is64 = self.cfg.key_ty == PtxType::U64;
        let desc = self.cfg.order == SortOrder::Descending;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("radix_pairs_count", e);

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align 4 .u32 cnt_hist[{RADIX_SIZE}];").map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_counts,\n    \
             .param .u64 param_keys,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{ty}   %key;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %shift, %digit, %old, %flat_idx;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u64    %n, %gid, %ptr_in, %ptr_cnt, %addr;").map_err(ferr)?;
        writeln!(out, "    .reg .u64    %smem_base, %hist_addr;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p;").map_err(ferr)?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(ferr)?;
        }

        writeln!(out, "    ld.param.u64 %ptr_cnt, [param_counts];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_keys];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %shift,    [param_shift];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_base, cnt_hist;").map_err(ferr)?;

        writeln!(out, "    setp.ge.u32  %p, %tid, {RADIX_SIZE};").map_err(ferr)?;
        writeln!(out, "    @%p bra CNT_INIT_DONE;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %hist_addr, %tid, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%hist_addr], 0;").map_err(ferr)?;
        writeln!(out, "CNT_INIT_DONE:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;

        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p bra CNT_FLUSH;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(ferr)?;
        writeln!(out, "    ld.global.{ty} %key, [%addr];").map_err(ferr)?;
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(ferr)?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;").map_err(ferr)?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(ferr)?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(ferr)?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0xF;").map_err(ferr)?;
        if desc {
            // Invert the digit so larger keys fall into lower buckets.
            writeln!(out, "    sub.u32      %digit, {}, %digit;", RADIX_SIZE - 1).map_err(ferr)?;
        }
        writeln!(out, "    mad.lo.u64   %hist_addr, %digit, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    atom.shared.add.u32 %old, [%hist_addr], 1;").map_err(ferr)?;

        writeln!(out, "CNT_FLUSH:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;
        writeln!(out, "    setp.ge.u32  %p, %tid, {RADIX_SIZE};").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u32   %flat_idx, %bid, {RADIX_SIZE}, %tid;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %flat_idx, 4, %ptr_cnt;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %hist_addr, %tid, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    ld.shared.u32 %old, [%hist_addr];").map_err(ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %old;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }

    // ── Scatter: move key AND value to the ranked output slot ────────────────

    fn generate_scatter_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.scatter_kernel_name();
        let kty = ptx_type_str(self.cfg.key_ty);
        let vty = ptx_type_str(self.cfg.val_ty);
        let bs = self.cfg.block_size;
        let keb = self.cfg.key_bytes();
        let veb = self.cfg.val_bytes();
        let is64 = self.cfg.key_ty == PtxType::U64;
        let desc = self.cfg.order == SortOrder::Descending;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("radix_pairs_scatter", e);

        let mut out = ptx_header(sm);
        writeln!(out, ".shared .align 4 .u32 block_offs[{RADIX_SIZE}];").map_err(ferr)?;
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_keys_out,\n    \
             .param .u64 param_vals_out,\n    \
             .param .u64 param_keys_in,\n    \
             .param .u64 param_vals_in,\n    \
             .param .u64 param_offsets,\n    \
             .param .u64 param_n,\n    \
             .param .u32 param_shift\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{kty}   %key;").map_err(ferr)?;
        writeln!(out, "    .reg .{vty}   %val;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u32    %tid, %bid, %shift, %digit, %out_pos, %flat_init, %off_val;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_kin, %ptr_vin, %ptr_kout, %ptr_vout, %ptr_off;"
        )
        .map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %addr, %smem_base, %smem_addr, %out64;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .pred   %p;").map_err(ferr)?;
        if is64 {
            writeln!(out, "    .reg .u64    %shift64, %key_shifted;").map_err(ferr)?;
        }

        writeln!(out, "    ld.param.u64 %ptr_kout, [param_keys_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_vout, [param_vals_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_kin,  [param_keys_in];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_vin,  [param_vals_in];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_off,  [param_offsets];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,         [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u32 %shift,     [param_shift];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %smem_base, block_offs;").map_err(ferr)?;

        writeln!(out, "    setp.ge.u32  %p, %tid, {RADIX_SIZE};").map_err(ferr)?;
        writeln!(out, "    @%p bra SCT_LOAD_DONE;").map_err(ferr)?;
        writeln!(
            out,
            "    mad.lo.u32   %flat_init, %bid, {RADIX_SIZE}, %tid;"
        )
        .map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %flat_init, 4, %ptr_off;").map_err(ferr)?;
        writeln!(out, "    ld.global.u32 %off_val, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %smem_addr, %tid, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    st.shared.u32 [%smem_addr], %off_val;").map_err(ferr)?;
        writeln!(out, "SCT_LOAD_DONE:").map_err(ferr)?;
        writeln!(out, "    bar.sync 0;").map_err(ferr)?;

        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{kty} %key, [%addr];").map_err(ferr)?;
        if is64 {
            writeln!(out, "    cvt.u64.u32  %shift64, %shift;").map_err(ferr)?;
            writeln!(out, "    shr.u64      %key_shifted, %key, %shift64;").map_err(ferr)?;
            writeln!(out, "    cvt.u32.u64  %digit, %key_shifted;").map_err(ferr)?;
        } else {
            writeln!(out, "    shr.u32      %digit, %key, %shift;").map_err(ferr)?;
        }
        writeln!(out, "    and.b32      %digit, %digit, 0xF;").map_err(ferr)?;
        if desc {
            writeln!(out, "    sub.u32      %digit, {}, %digit;", RADIX_SIZE - 1).map_err(ferr)?;
        }
        writeln!(out, "    mad.lo.u64   %smem_addr, %digit, 4, %smem_base;").map_err(ferr)?;
        writeln!(out, "    atom.shared.add.u32 %out_pos, [%smem_addr], 1;").map_err(ferr)?;
        writeln!(out, "    cvt.u64.u32  %out64, %out_pos;").map_err(ferr)?;

        // Write key.
        writeln!(out, "    mad.lo.u64   %addr, %out64, {keb}, %ptr_kout;").map_err(ferr)?;
        writeln!(out, "    st.global.{kty} [%addr], %key;").map_err(ferr)?;

        // Move the value payload from the same source index to the same output slot.
        writeln!(out, "    mad.lo.u64   %addr, %gid, {veb}, %ptr_vin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{vty} %val, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %out64, {veb}, %ptr_vout;").map_err(ferr)?;
        writeln!(out, "    st.global.{vty} [%addr], %val;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── Float twiddle ─────────────────────────────────────────────────────────────

/// Configuration for the floating-point radix-sort twiddle bijection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FloatTwiddleConfig {
    /// Float type (`F32` or `F64`).
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl FloatTwiddleConfig {
    /// Create a configuration, validating the type and block size.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] if `ty` is not `F32`/`F64`,
    /// or `block_size` is not a power of two in `[32, 1024]`.
    pub fn new(ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        if !matches!(ty, PtxType::F32 | PtxType::F64) {
            return Err(PrimitivesError::InvalidArgument(format!(
                "float twiddle type must be F32 or F64, got {ty:?}"
            )));
        }
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self { ty, block_size })
    }

    /// `(int register type, byte width, sign mask literal, sign-mask complement literal)`.
    ///
    /// PTX has no bitwise-NOT operator on immediates, so the complement of the
    /// sign mask (used by the inverse twiddle) is supplied as a precomputed
    /// literal rather than `~mask`.
    fn int_params(&self) -> (&'static str, u32, &'static str, &'static str) {
        if self.ty == PtxType::F64 {
            ("b64", 8, "0x8000000000000000", "0x7FFFFFFFFFFFFFFF")
        } else {
            ("b32", 4, "0x80000000", "0x7FFFFFFF")
        }
    }

    /// Kernel name for the forward twiddle.
    #[must_use]
    pub fn forward_kernel_name(&self) -> String {
        format!(
            "radix_float_twiddle_fwd_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }

    /// Kernel name for the inverse (un-twiddle) pass.
    #[must_use]
    pub fn inverse_kernel_name(&self) -> String {
        format!(
            "radix_float_twiddle_inv_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

/// PTX generator for the order-preserving float→unsigned twiddle.
pub struct FloatTwiddleTemplate {
    /// Configuration.
    pub cfg: FloatTwiddleConfig,
}

impl FloatTwiddleTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: FloatTwiddleConfig) -> Self {
        Self { cfg }
    }

    /// Generate `(forward_ptx, inverse_ptx)`.
    ///
    /// Both kernels operate in place over an array of length `n`, reinterpreting
    /// the float bits as `bN`.  Run the forward kernel before the radix passes
    /// and the inverse kernel after.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<(String, String)> {
        Ok((
            self.generate_kernel(sm, true)?,
            self.generate_kernel(sm, false)?,
        ))
    }

    fn generate_kernel(&self, sm: SmVersion, forward: bool) -> PrimitivesResult<String> {
        let name = if forward {
            self.cfg.forward_kernel_name()
        } else {
            self.cfg.inverse_kernel_name()
        };
        let (bty, eb, sign_mask, sign_mask_compl) = self.cfg.int_params();
        let bs = self.cfg.block_size;
        let op = if forward {
            "twiddle_fwd"
        } else {
            "twiddle_inv"
        };
        let ferr = move |e: std::fmt::Error| PrimitivesError::ptx(op, e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_data,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{bty}   %x, %y, %notx, %masked;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %tid, %bid;").map_err(ferr)?;
        writeln!(out, "    .reg .u64    %n, %gid, %ptr, %addr;").map_err(ferr)?;
        writeln!(out, "    .reg .pred   %oob, %neg;").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr, [param_data];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,   [param_n];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%oob ret;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr;").map_err(ferr)?;
        writeln!(out, "    ld.global.{bty} %x, [%addr];").map_err(ferr)?;

        // Test sign bit: (x & sign_mask) != 0.
        writeln!(out, "    and.{bty}     %masked, %x, {sign_mask};").map_err(ferr)?;
        writeln!(out, "    setp.ne.{bty} %neg, %masked, 0;").map_err(ferr)?;
        writeln!(out, "    not.{bty}     %notx, %x;").map_err(ferr)?;
        if forward {
            // neg → ~x ; pos → x | sign_mask
            writeln!(out, "    or.{bty}      %y, %x, {sign_mask};").map_err(ferr)?;
            writeln!(out, "    selp.{bty}    %y, %notx, %y, %neg;").map_err(ferr)?;
        } else {
            // neg → x & ~sign_mask ; pos → ~x
            writeln!(out, "    and.{bty}     %masked, %x, {sign_mask_compl};").map_err(ferr)?;
            writeln!(out, "    selp.{bty}    %y, %masked, %notx, %neg;").map_err(ferr)?;
        }
        writeln!(out, "    st.global.{bty} [%addr], %y;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU references ────────────────────────────────────────────────────────────

/// Host reference for sorting `(key, value)` pairs by key.  Returns the values
/// permuted into key order, matching the GPU key+value scatter result.
#[must_use]
pub fn reference_sort_pairs_by_key(
    keys: &[u64],
    values: &[u64],
    order: SortOrder,
) -> (Vec<u64>, Vec<u64>) {
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    // Stable sort by key (LSD radix is stable for ascending; descending here
    // mirrors digit-inversion, which is also stable within equal keys).
    idx.sort_by(|&a, &b| match order {
        SortOrder::Ascending => keys[a].cmp(&keys[b]),
        SortOrder::Descending => keys[b].cmp(&keys[a]),
    });
    let sk = idx.iter().map(|&i| keys[i]).collect();
    let sv = idx.iter().map(|&i| values[i]).collect();
    (sk, sv)
}

/// Forward float→u32 twiddle bijection for `f32`.
#[must_use]
pub fn twiddle_f32_forward(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// Inverse u32→`f32` twiddle bijection.
#[must_use]
pub fn twiddle_f32_inverse(y: u32) -> f32 {
    let bits = if y & 0x8000_0000 != 0 {
        y & 0x7FFF_FFFF
    } else {
        !y
    };
    f32::from_bits(bits)
}

/// Forward float→u64 twiddle bijection for `f64`.
#[must_use]
pub fn twiddle_f64_forward(x: f64) -> u64 {
    let bits = x.to_bits();
    if bits & 0x8000_0000_0000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000_0000_0000
    }
}

/// Inverse u64→`f64` twiddle bijection.
#[must_use]
pub fn twiddle_f64_inverse(y: u64) -> f64 {
    let bits = if y & 0x8000_0000_0000_0000 != 0 {
        y & 0x7FFF_FFFF_FFFF_FFFF
    } else {
        !y
    };
    f64::from_bits(bits)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    #[test]
    fn pairs_config_rejects_bad_key_type() {
        assert!(
            RadixPairsConfig::new(PtxType::F32, PtxType::U32, SortOrder::Ascending, 256).is_err()
        );
        assert!(
            RadixPairsConfig::new(PtxType::U32, PtxType::F32, SortOrder::Ascending, 256).is_ok()
        );
    }

    #[test]
    fn pairs_names_encode_order_and_types() {
        let c = RadixPairsConfig::new(PtxType::U64, PtxType::F32, SortOrder::Descending, 128)
            .expect("valid config");
        assert_eq!(c.count_kernel_name(), "radix_pairs_count_desc_u64_bs128");
        assert_eq!(
            c.scatter_kernel_name(),
            "radix_pairs_scatter_desc_u64_f32_bs128"
        );
        assert_eq!(c.passes(), 16);
        assert_eq!(c.key_bytes(), 8);
        assert_eq!(c.val_bytes(), 4);
    }

    #[test]
    fn count_descending_inverts_digit() {
        let c = RadixPairsConfig::new(PtxType::U32, PtxType::U32, SortOrder::Descending, 256)
            .expect("valid config");
        let (count, _) = RadixPairsTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(
            count.contains("sub.u32      %digit, 15, %digit"),
            "PTX: {count}"
        );
    }

    #[test]
    fn count_ascending_does_not_invert() {
        let c = RadixPairsConfig::new(PtxType::U32, PtxType::U32, SortOrder::Ascending, 256)
            .expect("valid config");
        let (count, _) = RadixPairsTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(!count.contains("sub.u32      %digit, 15"), "PTX: {count}");
    }

    #[test]
    fn scatter_moves_key_and_value() {
        let c = RadixPairsConfig::new(PtxType::U32, PtxType::U64, SortOrder::Ascending, 256)
            .expect("valid config");
        let (_, scatter) = RadixPairsTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(scatter.contains("param_vals_in"), "PTX: {scatter}");
        assert!(scatter.contains("param_vals_out"), "PTX: {scatter}");
        // value type is u64 → 8-byte loads/stores.
        assert!(scatter.contains("ld.global.u64 %val"), "PTX: {scatter}");
        assert!(
            scatter.contains("st.global.u64 [%addr], %val"),
            "PTX: {scatter}"
        );
    }

    #[test]
    fn float_twiddle_config_rejects_non_float() {
        assert!(FloatTwiddleConfig::new(PtxType::U32, 256).is_err());
        assert!(FloatTwiddleConfig::new(PtxType::F32, 256).is_ok());
        assert!(FloatTwiddleConfig::new(PtxType::F64, 256).is_ok());
    }

    #[test]
    fn float_twiddle_fwd_inv_ptx() {
        let c = FloatTwiddleConfig::new(PtxType::F32, 256).expect("valid config");
        let (fwd, inv) = FloatTwiddleTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(
            fwd.contains("radix_float_twiddle_fwd_f32_bs256"),
            "PTX: {fwd}"
        );
        assert!(fwd.contains("or.b32      %y, %x, 0x80000000"), "PTX: {fwd}");
        assert!(fwd.contains("not.b32"), "PTX: {fwd}");
        assert!(
            inv.contains("radix_float_twiddle_inv_f32_bs256"),
            "PTX: {inv}"
        );
        assert!(
            inv.contains("and.b32     %masked, %x, 0x7FFFFFFF"),
            "PTX: {inv}"
        );
    }

    #[test]
    fn float_twiddle_f64_uses_64bit() {
        let c = FloatTwiddleConfig::new(PtxType::F64, 256).expect("valid config");
        let (fwd, _) = FloatTwiddleTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(fwd.contains("0x8000000000000000"), "PTX: {fwd}");
        assert!(fwd.contains(".reg .b64"), "PTX: {fwd}");
    }

    #[test]
    fn twiddle_f32_is_order_preserving_bijection() {
        let vals = [
            f32::NEG_INFINITY,
            -1.0e30,
            -1.0,
            -0.0,
            0.0,
            1.0,
            42.5,
            1.0e30,
            f32::INFINITY,
        ];
        // Round-trip.
        for &v in &vals {
            let rt = twiddle_f32_inverse(twiddle_f32_forward(v));
            assert_eq!(rt.to_bits(), v.to_bits(), "round-trip failed for {v}");
        }
        // Order preservation: sorting by twiddled key matches float order
        // (excluding -0.0/0.0 which twiddle to adjacent codes).
        let mut sorted = vals;
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaNs"));
        let mut by_key = vals;
        by_key.sort_by_key(|&x| twiddle_f32_forward(x));
        for (a, b) in sorted.iter().zip(by_key.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "ordering mismatch {a} vs {b}");
        }
    }

    #[test]
    fn twiddle_f64_round_trip() {
        for &v in &[-2.5_f64, 0.0, 1.0, 1.0e300, f64::NEG_INFINITY] {
            let rt = twiddle_f64_inverse(twiddle_f64_forward(v));
            assert_eq!(rt.to_bits(), v.to_bits());
        }
    }

    #[test]
    fn reference_pairs_ascending_descending() {
        let keys = [3u64, 1, 2, 1];
        let vals = [30u64, 10, 20, 11];
        let (ak, av) = reference_sort_pairs_by_key(&keys, &vals, SortOrder::Ascending);
        assert_eq!(ak, vec![1, 1, 2, 3]);
        // Stable: the two key=1 entries keep input order (10 before 11).
        assert_eq!(av, vec![10, 11, 20, 30]);
        let (dk, _dv) = reference_sort_pairs_by_key(&keys, &vals, SortOrder::Descending);
        assert_eq!(dk, vec![3, 2, 1, 1]);
    }
}
