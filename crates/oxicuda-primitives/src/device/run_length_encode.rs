//! Device-wide run-length encoding — compact identical consecutive runs.
//!
//! Given an input array such as `[7, 7, 7, 3, 3, 9]`, run-length encoding
//! produces two parallel output arrays:
//!
//! * **unique values** — `[7, 3, 9]`
//! * **run lengths**    — `[3, 2, 1]`
//!
//! plus a single scalar **number of runs** (`3`).  This primitive mirrors
//! `cub::DeviceRunLengthEncode::Encode` and is used by sparse matrix formats
//! (CSR row pointers) and tokenizers (BPE merge counting).
//!
//! # Pipeline
//!
//! The full pipeline uses three GPU passes plus one exclusive scan:
//!
//! | # | Kernel        | Purpose                                                       |
//! |---|---------------|---------------------------------------------------------------|
//! | 1 | **head**      | Write `head[i] = 1` when `in[i] != in[i-1]` (`head[0] = 1`)    |
//! |   | *scan*        | Exclusive prefix sum of `head` → `run_idx[i]` (use `DeviceScan`)|
//! | 2 | **gather**    | For each head element, scatter `in[i]` to `unique[run_idx[i]]` and record its start position |
//! | 3 | **lengths**   | `len[r] = start[r+1] - start[r]` (last run uses `n`)          |
//!
//! The number of runs equals `run_idx[n-1] + head[n-1]`, which the caller reads
//! back from the last element of the inclusive scan (or `exclusive[n-1] + head[n-1]`).
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::device::run_length_encode::{
//!     DeviceRunLengthEncodeConfig, DeviceRunLengthEncodeTemplate,
//! };
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = DeviceRunLengthEncodeConfig::new(PtxType::U32, 256).expect("valid config");
//! let t = DeviceRunLengthEncodeTemplate::new(cfg);
//! let (head_ptx, gather_ptx, lengths_ptx) = t.generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(head_ptx.contains("rle_head_u32_bs256"));
//! assert!(gather_ptx.contains("rle_gather_u32_bs256"));
//! assert!(lengths_ptx.contains("rle_lengths_u32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for device-wide run-length encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceRunLengthEncodeConfig {
    /// Element type of the input / unique-output arrays.
    pub ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl DeviceRunLengthEncodeConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] if `block_size` is not a
    /// power of two in `[32, 1024]`.
    pub fn new(ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self { ty, block_size })
    }

    /// Bytes per data element.
    #[must_use]
    pub fn elem_bytes(&self) -> u32 {
        match self.ty {
            PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => 8,
            _ => 4,
        }
    }

    /// Number of thread blocks needed for `n` elements.
    #[must_use]
    pub fn num_blocks(&self, n: u64) -> u64 {
        n.div_ceil(u64::from(self.block_size))
    }

    /// Bytes of scratch needed for the per-element head-flag and start-position
    /// arrays: two `u32` arrays plus one `u64` run-index array, all of length
    /// `n` (the unique/length outputs are at most `n` elements too, but are
    /// caller-owned).
    #[must_use]
    pub fn workspace_bytes(&self, n: u64) -> u64 {
        // head[n] (u32) + scanned run_idx[n] (u64) + start[n] (u64)
        n * 4 + n * 8 + n * 8
    }

    /// Kernel name for the head-flag pass.
    #[must_use]
    pub fn head_kernel_name(&self) -> String {
        format!("rle_head_{}_bs{}", ptx_type_str(self.ty), self.block_size)
    }

    /// Kernel name for the gather (scatter-unique + record-start) pass.
    #[must_use]
    pub fn gather_kernel_name(&self) -> String {
        format!("rle_gather_{}_bs{}", ptx_type_str(self.ty), self.block_size)
    }

    /// Kernel name for the lengths-from-starts pass.
    #[must_use]
    pub fn lengths_kernel_name(&self) -> String {
        format!(
            "rle_lengths_{}_bs{}",
            ptx_type_str(self.ty),
            self.block_size
        )
    }
}

// ─── Template ────────────────────────────────────────────────────────────────

/// PTX code generator for device-wide run-length encoding.
///
/// Produces three PTX kernels: **head**, **gather**, and **lengths**.  Between
/// the head and gather kernels the caller must run an exclusive prefix scan
/// (sum) on the `u32` head-flag array via
/// [`crate::device::scan::DeviceScanTemplate`].
pub struct DeviceRunLengthEncodeTemplate {
    /// Configuration.
    pub cfg: DeviceRunLengthEncodeConfig,
}

impl DeviceRunLengthEncodeTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: DeviceRunLengthEncodeConfig) -> Self {
        Self { cfg }
    }

    /// Generate all three PTX kernels as `(head_ptx, gather_ptx, lengths_ptx)`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on a formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<(String, String, String)> {
        let head = self.generate_head_kernel(sm)?;
        let gather = self.generate_gather_kernel(sm)?;
        let lengths = self.generate_lengths_kernel(sm)?;
        Ok((head, gather, lengths))
    }

    // ── Kernel 1: head flags ─────────────────────────────────────────────────
    //
    // head[i] = (i == 0) ? 1 : (in[i] != in[i-1]) ? 1 : 0

    fn generate_head_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.head_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("rle_head", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_heads,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(&ferr)?;
        writeln!(out, "{{").map_err(&ferr)?;
        writeln!(out, "    .reg .{ty}   %cur, %prev;").map_err(&ferr)?;
        writeln!(out, "    .reg .u32    %head, %ltid, %bid;").map_err(&ferr)?;
        writeln!(
            out,
            "    .reg .u64    %n, %gid, %ptr_in, %ptr_out, %addr, %prev_idx;"
        )
        .map_err(&ferr)?;
        writeln!(out, "    .reg .pred   %oob, %is_first, %diff;").map_err(&ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_out, [param_heads];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,  [param_input];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %n,        [param_n];").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(&ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(&ferr)?;
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(&ferr)?;
        writeln!(out, "    @%oob ret;").map_err(&ferr)?;

        // First element is always a run head.
        writeln!(out, "    setp.eq.u64  %is_first, %gid, 0;").map_err(&ferr)?;
        writeln!(out, "    @%is_first bra RLE_HEAD_ONE;").map_err(&ferr)?;

        // Load cur and prev, compare.
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(&ferr)?;
        writeln!(out, "    ld.global.{ty} %cur, [%addr];").map_err(&ferr)?;
        writeln!(out, "    sub.u64      %prev_idx, %gid, 1;").map_err(&ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %prev_idx, {eb}, %ptr_in;").map_err(&ferr)?;
        writeln!(out, "    ld.global.{ty} %prev, [%addr];").map_err(&ferr)?;
        writeln!(out, "    setp.ne.{ty} %diff, %cur, %prev;").map_err(&ferr)?;
        writeln!(out, "    selp.u32     %head, 1, 0, %diff;").map_err(&ferr)?;
        writeln!(out, "    bra RLE_HEAD_STORE;").map_err(&ferr)?;

        writeln!(out, "RLE_HEAD_ONE:").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %head, 1;").map_err(&ferr)?;

        writeln!(out, "RLE_HEAD_STORE:").map_err(&ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_out;").map_err(&ferr)?;
        writeln!(out, "    st.global.u32 [%addr], %head;").map_err(&ferr)?;
        writeln!(out, "    ret;").map_err(&ferr)?;
        writeln!(out, "}}").map_err(&ferr)?;

        Ok(out)
    }

    // ── Kernel 2: gather unique values + record run start positions ───────────
    //
    // For each i where head[i] == 1:
    //   r = run_idx[i]   (exclusive scan of head)
    //   unique[r] = in[i]
    //   start[r]  = i

    fn generate_gather_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.gather_kernel_name();
        let ty = ptx_type_str(self.cfg.ty);
        let bs = self.cfg.block_size;
        let eb = self.cfg.elem_bytes();
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("rle_gather", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_unique,\n    \
             .param .u64 param_starts,\n    \
             .param .u64 param_input,\n    \
             .param .u64 param_heads,\n    \
             .param .u64 param_run_idx,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(&ferr)?;
        writeln!(out, "{{").map_err(&ferr)?;
        writeln!(out, "    .reg .{ty}   %val;").map_err(&ferr)?;
        writeln!(out, "    .reg .u32    %head, %ltid, %bid;").map_err(&ferr)?;
        writeln!(out, "    .reg .u64    %n, %gid, %run, %addr;").map_err(&ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_uniq, %ptr_start, %ptr_in, %ptr_head, %ptr_ridx;"
        )
        .map_err(&ferr)?;
        writeln!(out, "    .reg .pred   %oob, %keep;").map_err(&ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_uniq,  [param_unique];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_start, [param_starts];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_in,    [param_input];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_head,  [param_heads];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_ridx,  [param_run_idx];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %n,          [param_n];").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(&ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %gid, %ltid;
    mad.wide.u32   %gid, %bid, {bs}, %gid;"
        )
        .map_err(&ferr)?;
        writeln!(out, "    setp.ge.u64  %oob, %gid, %n;").map_err(&ferr)?;
        writeln!(out, "    @%oob ret;").map_err(&ferr)?;

        // Skip non-head elements.
        writeln!(out, "    mad.lo.u64   %addr, %gid, 4, %ptr_head;").map_err(&ferr)?;
        writeln!(out, "    ld.global.u32 %head, [%addr];").map_err(&ferr)?;
        writeln!(out, "    setp.ne.u32  %keep, %head, 0;").map_err(&ferr)?;
        writeln!(out, "    @!%keep ret;").map_err(&ferr)?;

        // Output run index from exclusive scan of head flags (u64).
        writeln!(out, "    mad.lo.u64   %addr, %gid, 8, %ptr_ridx;").map_err(&ferr)?;
        writeln!(out, "    ld.global.u64 %run, [%addr];").map_err(&ferr)?;

        // Load input value.
        writeln!(out, "    mad.lo.u64   %addr, %gid, {eb}, %ptr_in;").map_err(&ferr)?;
        writeln!(out, "    ld.global.{ty} %val, [%addr];").map_err(&ferr)?;

        // unique[run] = val
        writeln!(out, "    mad.lo.u64   %addr, %run, {eb}, %ptr_uniq;").map_err(&ferr)?;
        writeln!(out, "    st.global.{ty} [%addr], %val;").map_err(&ferr)?;

        // start[run] = gid
        writeln!(out, "    mad.lo.u64   %addr, %run, 8, %ptr_start;").map_err(&ferr)?;
        writeln!(out, "    st.global.u64 [%addr], %gid;").map_err(&ferr)?;
        writeln!(out, "    ret;").map_err(&ferr)?;
        writeln!(out, "}}").map_err(&ferr)?;

        Ok(out)
    }

    // ── Kernel 3: run lengths from consecutive start positions ────────────────
    //
    // len[r] = (r + 1 < num_runs) ? start[r+1] - start[r] : n - start[r]

    fn generate_lengths_kernel(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.lengths_kernel_name();
        let bs = self.cfg.block_size;
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("rle_lengths", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_lengths,\n    \
             .param .u64 param_starts,\n    \
             .param .u64 param_num_runs,\n    \
             .param .u64 param_n\n)"
        )
        .map_err(&ferr)?;
        writeln!(out, "{{").map_err(&ferr)?;
        writeln!(out, "    .reg .u32    %ltid, %bid;").map_err(&ferr)?;
        writeln!(
            out,
            "    .reg .u64    %num_runs, %n, %r, %start, %next, %len, %addr, %next_r;"
        )
        .map_err(&ferr)?;
        writeln!(out, "    .reg .u64    %ptr_len, %ptr_start;").map_err(&ferr)?;
        writeln!(out, "    .reg .pred   %oob, %is_last;").map_err(&ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_len,   [param_lengths];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_start, [param_starts];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %num_runs,  [param_num_runs];").map_err(&ferr)?;
        writeln!(out, "    ld.param.u64 %n,          [param_n];").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %ltid, %tid.x;").map_err(&ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(&ferr)?;
        writeln!(
            out,
            "    cvt.u64.u32   %r, %ltid;
    mad.wide.u32   %r, %bid, {bs}, %r;"
        )
        .map_err(&ferr)?;
        writeln!(out, "    setp.ge.u64  %oob, %r, %num_runs;").map_err(&ferr)?;
        writeln!(out, "    @%oob ret;").map_err(&ferr)?;

        // start[r]
        writeln!(out, "    mad.lo.u64   %addr, %r, 8, %ptr_start;").map_err(&ferr)?;
        writeln!(out, "    ld.global.u64 %start, [%addr];").map_err(&ferr)?;

        // Last run: next boundary = n.
        writeln!(out, "    add.u64      %next_r, %r, 1;").map_err(&ferr)?;
        writeln!(out, "    setp.ge.u64  %is_last, %next_r, %num_runs;").map_err(&ferr)?;
        writeln!(out, "    @%is_last bra RLE_LEN_LAST;").map_err(&ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %next_r, 8, %ptr_start;").map_err(&ferr)?;
        writeln!(out, "    ld.global.u64 %next, [%addr];").map_err(&ferr)?;
        writeln!(out, "    bra RLE_LEN_COMPUTE;").map_err(&ferr)?;
        writeln!(out, "RLE_LEN_LAST:").map_err(&ferr)?;
        writeln!(out, "    mov.u64      %next, %n;").map_err(&ferr)?;

        writeln!(out, "RLE_LEN_COMPUTE:").map_err(&ferr)?;
        writeln!(out, "    sub.u64      %len, %next, %start;").map_err(&ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %r, 8, %ptr_len;").map_err(&ferr)?;
        writeln!(out, "    st.global.u64 [%addr], %len;").map_err(&ferr)?;
        writeln!(out, "    ret;").map_err(&ferr)?;
        writeln!(out, "}}").map_err(&ferr)?;

        Ok(out)
    }
}

// ─── CPU reference ─────────────────────────────────────────────────────────────

/// Host-side reference implementation of run-length encoding, used to
/// cross-check the GPU pipeline during development and in unit tests.
///
/// Returns `(unique_values, run_lengths)` with one entry per run.
#[must_use]
pub fn reference_run_length_encode<T: PartialEq + Copy>(input: &[T]) -> (Vec<T>, Vec<u64>) {
    let mut values = Vec::new();
    let mut lengths = Vec::new();
    let mut iter = input.iter().copied();
    if let Some(first) = iter.next() {
        let mut cur = first;
        let mut count: u64 = 1;
        for v in iter {
            if v == cur {
                count += 1;
            } else {
                values.push(cur);
                lengths.push(count);
                cur = v;
                count = 1;
            }
        }
        values.push(cur);
        lengths.push(count);
    }
    (values, lengths)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    fn cfg(ty: PtxType) -> DeviceRunLengthEncodeConfig {
        DeviceRunLengthEncodeConfig::new(ty, 256).expect("valid config")
    }

    #[test]
    fn config_rejects_bad_block_size() {
        assert!(DeviceRunLengthEncodeConfig::new(PtxType::U32, 100).is_err());
        assert!(DeviceRunLengthEncodeConfig::new(PtxType::U32, 16).is_err());
        assert!(DeviceRunLengthEncodeConfig::new(PtxType::U32, 2048).is_err());
        assert!(DeviceRunLengthEncodeConfig::new(PtxType::U32, 256).is_ok());
    }

    #[test]
    fn kernel_names_contain_type_and_block() {
        let c = cfg(PtxType::U32);
        assert!(c.head_kernel_name().contains("rle_head_u32_bs256"));
        assert!(c.gather_kernel_name().contains("rle_gather_u32_bs256"));
        assert!(c.lengths_kernel_name().contains("rle_lengths_u32_bs256"));
    }

    #[test]
    fn elem_bytes_64bit() {
        assert_eq!(cfg(PtxType::U32).elem_bytes(), 4);
        assert_eq!(cfg(PtxType::U64).elem_bytes(), 8);
        assert_eq!(cfg(PtxType::F64).elem_bytes(), 8);
    }

    #[test]
    fn head_ptx_compares_neighbours() {
        let t = DeviceRunLengthEncodeTemplate::new(cfg(PtxType::U32));
        let ptx = t
            .generate_head_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("setp.ne.u32 %diff"), "PTX: {ptx}");
        assert!(ptx.contains("RLE_HEAD_ONE"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.u32"), "PTX: {ptx}");
    }

    #[test]
    fn head_ptx_f32_uses_f32_compare() {
        let t = DeviceRunLengthEncodeTemplate::new(cfg(PtxType::F32));
        let ptx = t
            .generate_head_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("setp.ne.f32"), "PTX: {ptx}");
        assert!(ptx.contains("ld.global.f32"), "PTX: {ptx}");
    }

    #[test]
    fn gather_ptx_scatters_unique_and_start() {
        let t = DeviceRunLengthEncodeTemplate::new(cfg(PtxType::U32));
        let ptx = t
            .generate_gather_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("param_unique"), "PTX: {ptx}");
        assert!(ptx.contains("param_starts"), "PTX: {ptx}");
        assert!(ptx.contains("param_run_idx"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.u64"), "PTX: {ptx}"); // start[run] = gid
        assert!(ptx.contains("st.global.u32"), "PTX: {ptx}"); // unique[run] = val
    }

    #[test]
    fn lengths_ptx_handles_last_run() {
        let t = DeviceRunLengthEncodeTemplate::new(cfg(PtxType::U32));
        let ptx = t
            .generate_lengths_kernel(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("RLE_LEN_LAST"), "PTX: {ptx}");
        assert!(ptx.contains("sub.u64      %len"), "PTX: {ptx}");
    }

    #[test]
    fn generate_all_three_succeeds() {
        let t = DeviceRunLengthEncodeTemplate::new(cfg(PtxType::U64));
        let (h, g, l) = t
            .generate(SmVersion::Sm90)
            .expect("PTX generation should succeed in test");
        assert!(!h.is_empty() && !g.is_empty() && !l.is_empty());
        // u64 element type → 8-byte data stride in gather.
        assert!(g.contains("ld.global.u64"), "gather PTX: {g}");
    }

    #[test]
    fn workspace_bytes_scales_with_n() {
        let c = cfg(PtxType::U32);
        assert!(c.workspace_bytes(1000) > c.workspace_bytes(500));
        // 4 + 8 + 8 = 20 bytes/element.
        assert_eq!(c.workspace_bytes(10), 200);
    }

    #[test]
    fn reference_basic() {
        let (vals, lens) = reference_run_length_encode(&[7u32, 7, 7, 3, 3, 9]);
        assert_eq!(vals, vec![7, 3, 9]);
        assert_eq!(lens, vec![3, 2, 1]);
    }

    #[test]
    fn reference_empty_and_single() {
        let (v0, l0) = reference_run_length_encode::<u32>(&[]);
        assert!(v0.is_empty() && l0.is_empty());
        let (v1, l1) = reference_run_length_encode(&[5u32]);
        assert_eq!(v1, vec![5]);
        assert_eq!(l1, vec![1]);
    }

    #[test]
    fn reference_all_distinct_and_all_equal() {
        let (vd, ld) = reference_run_length_encode(&[1u32, 2, 3, 4]);
        assert_eq!(vd, vec![1, 2, 3, 4]);
        assert_eq!(ld, vec![1, 1, 1, 1]);

        let (ve, le) = reference_run_length_encode(&[8u32, 8, 8, 8, 8]);
        assert_eq!(ve, vec![8]);
        assert_eq!(le, vec![5]);
    }
}
