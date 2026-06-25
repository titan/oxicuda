//! Key+value merge: the value-carrying co-rank merge kernel.
//!
//! This module provides the key+value counterpart to
//! [`crate::sort::merge_sort`], plus the standalone `DeviceMergeKeysValues`
//! primitive (`cub::DeviceMerge::MergeKeys` / `MergePairs`).
//!
//! Both use the **same** O(log n) co-rank binary-search merge kernel; they
//! differ only in how the caller drives it:
//!
//! * **Merge-sort pass** — launch repeatedly with a doubling `merge_len` over
//!   one array that has already been block-sorted, exactly like
//!   [`crate::sort::merge_sort`], but each thread also moves the value paired
//!   with the key it selects.
//! * **Standalone merge** — launch once with `merge_len` equal to the length of
//!   the (single) left run to merge two adjacent sorted key+value runs into one
//!   sorted output.  This is the stable-join building block.
//!
//! The co-rank search compares **keys only**; the value is a passive payload
//! moved to the same output slot as its key, which keeps the merge stable.
//!
//! # Example
//!
//! ```
//! use oxicuda_primitives::sort::merge_pairs::{MergePairsConfig, MergePairsTemplate};
//! use oxicuda_ptx::ir::PtxType;
//! use oxicuda_ptx::arch::SmVersion;
//!
//! let cfg = MergePairsConfig::new(PtxType::U32, PtxType::F32, 256).expect("valid config");
//! let ptx = MergePairsTemplate::new(cfg).generate(SmVersion::Sm80).expect("PTX gen");
//! assert!(ptx.contains("merge_pairs_u32_f32_bs256"));
//! ```

use std::fmt::Write as FmtWrite;

use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

use crate::error::{PrimitivesError, PrimitivesResult};
use crate::ptx_helpers::{ptx_header, ptx_type_str};

fn elem_bytes(ty: PtxType) -> u32 {
    match ty {
        PtxType::F64 | PtxType::U64 | PtxType::S64 | PtxType::B64 => 8,
        _ => 4,
    }
}

/// Configuration for the key+value co-rank merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MergePairsConfig {
    /// Key type (drives the comparison).
    pub key_ty: PtxType,
    /// Value (payload) type.
    pub val_ty: PtxType,
    /// Threads per block (power of 2, `32`–`1024`).
    pub block_size: u32,
}

impl MergePairsConfig {
    /// Create a configuration, validating `block_size`.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::InvalidArgument`] for an invalid `block_size`.
    pub fn new(key_ty: PtxType, val_ty: PtxType, block_size: u32) -> PrimitivesResult<Self> {
        if !(32..=1024).contains(&block_size) || !block_size.is_power_of_two() {
            return Err(PrimitivesError::InvalidArgument(format!(
                "block_size must be a power of two in [32, 1024], got {block_size}"
            )));
        }
        Ok(Self {
            key_ty,
            val_ty,
            block_size,
        })
    }

    /// Bytes per key.
    #[must_use]
    pub fn key_bytes(&self) -> u32 {
        elem_bytes(self.key_ty)
    }

    /// Bytes per value.
    #[must_use]
    pub fn val_bytes(&self) -> u32 {
        elem_bytes(self.val_ty)
    }

    /// Generated kernel name.
    #[must_use]
    pub fn kernel_name(&self) -> String {
        format!(
            "merge_pairs_{}_{}_bs{}",
            ptx_type_str(self.key_ty),
            ptx_type_str(self.val_ty),
            self.block_size
        )
    }
}

/// PTX generator for the key+value co-rank merge kernel.
pub struct MergePairsTemplate {
    /// Configuration.
    pub cfg: MergePairsConfig,
}

impl MergePairsTemplate {
    /// Create a new template.
    #[must_use]
    pub fn new(cfg: MergePairsConfig) -> Self {
        Self { cfg }
    }

    /// Generate the merge PTX kernel.
    ///
    /// Params: `(keys_out, vals_out, keys_in, vals_in, n, merge_len)`.  Each
    /// thread produces one output element of the merged key+value stream.
    ///
    /// # Errors
    ///
    /// Returns [`PrimitivesError::PtxGeneration`] on formatting failure.
    pub fn generate(&self, sm: SmVersion) -> PrimitivesResult<String> {
        let name = self.cfg.kernel_name();
        let kty = ptx_type_str(self.cfg.key_ty);
        let vty = ptx_type_str(self.cfg.val_ty);
        let bs = self.cfg.block_size;
        let keb = self.cfg.key_bytes();
        let veb = self.cfg.val_bytes();
        let cmp_le = format!("setp.le.{kty}");
        let ferr = |e: std::fmt::Error| PrimitivesError::ptx("merge_pairs", e);

        let mut out = ptx_header(sm);
        writeln!(
            out,
            ".visible .entry {name}(\n    \
             .param .u64 param_keys_out,\n    \
             .param .u64 param_vals_out,\n    \
             .param .u64 param_keys_in,\n    \
             .param .u64 param_vals_in,\n    \
             .param .u64 param_n,\n    \
             .param .u64 param_merge_len\n)"
        )
        .map_err(ferr)?;
        writeln!(out, "{{").map_err(ferr)?;
        writeln!(out, "    .reg .{kty}   %ak, %bj, %a_km1;").map_err(ferr)?;
        writeln!(out, "    .reg .{vty}   %av, %bv;").map_err(ferr)?;
        writeln!(out, "    .reg .u64    %n, %merge_len, %gid, %local_pos;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %merge_id, %pair_start, %left_start, %right_start;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u64    %lo, %hi, %mid, %lo_min, %hi_cand;").map_err(ferr)?;
        writeln!(out, "    .reg .u64    %k, %j, %k_m1;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .u64    %ptr_kin, %ptr_vin, %ptr_kout, %ptr_vout, %addr;"
        )
        .map_err(ferr)?;
        writeln!(out, "    .reg .u64    %two_ml, %out_addr, %src_idx;").map_err(ferr)?;
        writeln!(out, "    .reg .u32    %tid, %bid;").map_err(ferr)?;
        writeln!(
            out,
            "    .reg .pred   %p, %a_leq_b, %k_valid, %j_valid, %akm1_le_bj;"
        )
        .map_err(ferr)?;

        writeln!(out, "    ld.param.u64 %ptr_kout, [param_keys_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_vout, [param_vals_out];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_kin,  [param_keys_in];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %ptr_vin,  [param_vals_in];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %n,         [param_n];").map_err(ferr)?;
        writeln!(out, "    ld.param.u64 %merge_len, [param_merge_len];").map_err(ferr)?;
        writeln!(out, "    mov.u32      %tid, %tid.x;").map_err(ferr)?;
        writeln!(out, "    mov.u32      %bid, %ctaid.x;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %gid, %bid, {bs}, %tid;").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %gid, %n;").map_err(ferr)?;
        writeln!(out, "    @%p ret;").map_err(ferr)?;

        // pair_start, local_pos, left/right run starts.
        writeln!(out, "    add.u64      %two_ml, %merge_len, %merge_len;").map_err(ferr)?;
        writeln!(out, "    div.u64      %merge_id, %gid, %two_ml;").map_err(ferr)?;
        writeln!(out, "    mul.lo.u64   %pair_start, %merge_id, %two_ml;").map_err(ferr)?;
        writeln!(out, "    sub.u64      %local_pos, %gid, %pair_start;").map_err(ferr)?;
        writeln!(out, "    mov.u64      %left_start, %pair_start;").map_err(ferr)?;
        writeln!(
            out,
            "    add.u64      %right_start, %pair_start, %merge_len;"
        )
        .map_err(ferr)?;

        // lo = max(0, local_pos - merge_len), hi = min(local_pos, merge_len).
        writeln!(out, "    setp.gt.u64  %p, %local_pos, %merge_len;").map_err(ferr)?;
        writeln!(out, "    sub.u64      %lo_min, %local_pos, %merge_len;").map_err(ferr)?;
        writeln!(out, "    selp.u64     %lo, %lo_min, 0, %p;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u64  %p, %local_pos, %merge_len;").map_err(ferr)?;
        writeln!(out, "    selp.u64     %hi, %local_pos, %merge_len, %p;").map_err(ferr)?;

        // Co-rank binary search on keys.
        writeln!(out, "MP_BSEARCH:").map_err(ferr)?;
        writeln!(out, "    setp.ge.u64  %p, %lo, %hi;").map_err(ferr)?;
        writeln!(out, "    @%p bra MP_BSEARCH_DONE;").map_err(ferr)?;
        writeln!(out, "    add.u64      %mid, %lo, %hi;").map_err(ferr)?;
        writeln!(out, "    add.u64      %mid, %mid, 1;").map_err(ferr)?;
        writeln!(out, "    shr.u64      %mid, %mid, 1;").map_err(ferr)?;
        writeln!(out, "    sub.u64      %j, %local_pos, %mid;").map_err(ferr)?;
        writeln!(out, "    sub.u64      %k_m1, %mid, 1;").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %left_start, %k_m1;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %addr, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{kty} %a_km1, [%addr];").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %right_start, %j;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %addr, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u64  %j_valid, %j, %merge_len;").map_err(ferr)?;
        writeln!(out, "    @%j_valid ld.global.{kty} %bj, [%addr];").map_err(ferr)?;
        writeln!(out, "    @!%j_valid bra MP_TAKE_A;").map_err(ferr)?;
        writeln!(out, "    {cmp_le}    %akm1_le_bj, %a_km1, %bj;").map_err(ferr)?;
        writeln!(out, "    @!%akm1_le_bj bra MP_TAKE_B;").map_err(ferr)?;
        writeln!(out, "MP_TAKE_A:").map_err(ferr)?;
        writeln!(out, "    mov.u64      %lo, %mid;").map_err(ferr)?;
        writeln!(out, "    bra MP_BSEARCH;").map_err(ferr)?;
        writeln!(out, "MP_TAKE_B:").map_err(ferr)?;
        writeln!(out, "    mov.u64      %hi, %k_m1;").map_err(ferr)?;
        writeln!(out, "    bra MP_BSEARCH;").map_err(ferr)?;
        writeln!(out, "MP_BSEARCH_DONE:").map_err(ferr)?;

        // k = lo, j = local_pos - lo.
        writeln!(out, "    mov.u64      %k, %lo;").map_err(ferr)?;
        writeln!(out, "    sub.u64      %j, %local_pos, %lo;").map_err(ferr)?;

        // Validity of A[k] and B[j].
        writeln!(out, "    setp.lt.u64  %k_valid, %k, %merge_len;").map_err(ferr)?;
        writeln!(out, "    add.u64      %hi_cand, %left_start, %k;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u64  %p, %hi_cand, %n;").map_err(ferr)?;
        writeln!(out, "    and.pred     %k_valid, %k_valid, %p;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u64  %j_valid, %j, %merge_len;").map_err(ferr)?;
        writeln!(out, "    add.u64      %hi_cand, %right_start, %j;").map_err(ferr)?;
        writeln!(out, "    setp.lt.u64  %p, %hi_cand, %n;").map_err(ferr)?;
        writeln!(out, "    and.pred     %j_valid, %j_valid, %p;").map_err(ferr)?;

        // Output address.
        writeln!(out, "    mad.lo.u64   %out_addr, %gid, {keb}, %ptr_kout;").map_err(ferr)?;

        // Decide A vs B (branch-based).
        writeln!(out, "    @!%k_valid bra MP_USE_B;").map_err(ferr)?;
        writeln!(out, "    @!%j_valid bra MP_USE_A;").map_err(ferr)?;
        // Load both keys to compare.
        writeln!(out, "    add.u64      %addr, %left_start, %k;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %addr, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{kty} %ak, [%addr];").map_err(ferr)?;
        writeln!(out, "    add.u64      %addr, %right_start, %j;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %addr, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{kty} %bj, [%addr];").map_err(ferr)?;
        writeln!(out, "    {cmp_le}     %a_leq_b, %ak, %bj;").map_err(ferr)?;
        writeln!(out, "    @%a_leq_b  bra MP_USE_A;").map_err(ferr)?;

        // ── Use B[j]: write key and value ────────────────────────────────────
        writeln!(out, "MP_USE_B:").map_err(ferr)?;
        writeln!(out, "    add.u64      %src_idx, %right_start, %j;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %src_idx, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{kty} %bj, [%addr];").map_err(ferr)?;
        writeln!(out, "    st.global.{kty} [%out_addr], %bj;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %src_idx, {veb}, %ptr_vin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{vty} %bv, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {veb}, %ptr_vout;").map_err(ferr)?;
        writeln!(out, "    st.global.{vty} [%addr], %bv;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;

        // ── Use A[k]: write key and value ────────────────────────────────────
        writeln!(out, "MP_USE_A:").map_err(ferr)?;
        writeln!(out, "    add.u64      %src_idx, %left_start, %k;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %src_idx, {keb}, %ptr_kin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{kty} %ak, [%addr];").map_err(ferr)?;
        writeln!(out, "    st.global.{kty} [%out_addr], %ak;").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %src_idx, {veb}, %ptr_vin;").map_err(ferr)?;
        writeln!(out, "    ld.global.{vty} %av, [%addr];").map_err(ferr)?;
        writeln!(out, "    mad.lo.u64   %addr, %gid, {veb}, %ptr_vout;").map_err(ferr)?;
        writeln!(out, "    st.global.{vty} [%addr], %av;").map_err(ferr)?;
        writeln!(out, "    ret;").map_err(ferr)?;
        writeln!(out, "}}").map_err(ferr)?;
        Ok(out)
    }
}

// ─── CPU reference ─────────────────────────────────────────────────────────────

/// Host reference for stably merging two sorted key+value runs.
///
/// `(left_keys, left_vals)` and `(right_keys, right_vals)` must each already be
/// sorted ascending by key.  On ties, left-run elements come first (stable).
#[must_use]
pub fn reference_merge_pairs(
    left_keys: &[u64],
    left_vals: &[u64],
    right_keys: &[u64],
    right_vals: &[u64],
) -> (Vec<u64>, Vec<u64>) {
    let mut keys = Vec::with_capacity(left_keys.len() + right_keys.len());
    let mut vals = Vec::with_capacity(left_vals.len() + right_vals.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < left_keys.len() && j < right_keys.len() {
        if left_keys[i] <= right_keys[j] {
            keys.push(left_keys[i]);
            vals.push(left_vals[i]);
            i += 1;
        } else {
            keys.push(right_keys[j]);
            vals.push(right_vals[j]);
            j += 1;
        }
    }
    while i < left_keys.len() {
        keys.push(left_keys[i]);
        vals.push(left_vals[i]);
        i += 1;
    }
    while j < right_keys.len() {
        keys.push(right_keys[j]);
        vals.push(right_vals[j]);
        j += 1;
    }
    (keys, vals)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::{arch::SmVersion, ir::PtxType};

    #[test]
    fn config_validation_and_name() {
        assert!(MergePairsConfig::new(PtxType::U32, PtxType::F32, 100).is_err());
        let c = MergePairsConfig::new(PtxType::U32, PtxType::F32, 256).expect("valid config");
        assert_eq!(c.kernel_name(), "merge_pairs_u32_f32_bs256");
        assert_eq!(c.key_bytes(), 4);
        assert_eq!(c.val_bytes(), 4);
    }

    #[test]
    fn val_bytes_64bit() {
        let c = MergePairsConfig::new(PtxType::U32, PtxType::U64, 256).expect("valid config");
        assert_eq!(c.val_bytes(), 8);
    }

    #[test]
    fn merge_ptx_has_corank_search_and_moves_values() {
        let c = MergePairsConfig::new(PtxType::U32, PtxType::F32, 256).expect("valid config");
        let ptx = MergePairsTemplate::new(c)
            .generate(SmVersion::Sm80)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("MP_BSEARCH"), "PTX: {ptx}");
        assert!(ptx.contains("MP_USE_A"), "PTX: {ptx}");
        assert!(ptx.contains("MP_USE_B"), "PTX: {ptx}");
        assert!(ptx.contains("param_vals_in"), "PTX: {ptx}");
        assert!(ptx.contains("param_vals_out"), "PTX: {ptx}");
        // value channel uses f32 loads/stores.
        assert!(ptx.contains("ld.global.f32"), "PTX: {ptx}");
        assert!(ptx.contains("st.global.f32"), "PTX: {ptx}");
        // key channel uses u32 with setp.le for comparison.
        assert!(ptx.contains("setp.le.u32"), "PTX: {ptx}");
    }

    #[test]
    fn merge_ptx_64bit_key_uses_8byte_stride() {
        let c = MergePairsConfig::new(PtxType::U64, PtxType::U32, 256).expect("valid config");
        let ptx = MergePairsTemplate::new(c)
            .generate(SmVersion::Sm90)
            .expect("PTX generation should succeed in test");
        assert!(ptx.contains("ld.global.u64 %ak"), "PTX: {ptx}");
        assert!(ptx.contains("setp.le.u64"), "PTX: {ptx}");
    }

    #[test]
    fn reference_merge_stable_on_ties() {
        // Left key 2 (val 20) must precede right key 2 (val 99) on a tie.
        let (k, v) = reference_merge_pairs(&[1, 2, 5], &[10, 20, 50], &[2, 3], &[99, 30]);
        assert_eq!(k, vec![1, 2, 2, 3, 5]);
        assert_eq!(v, vec![10, 20, 99, 30, 50]);
    }

    #[test]
    fn reference_merge_one_side_empty() {
        let (k, v) = reference_merge_pairs(&[], &[], &[1, 2], &[10, 20]);
        assert_eq!(k, vec![1, 2]);
        assert_eq!(v, vec![10, 20]);
        let (k2, v2) = reference_merge_pairs(&[3, 4], &[30, 40], &[], &[]);
        assert_eq!(k2, vec![3, 4]);
        assert_eq!(v2, vec![30, 40]);
    }
}
