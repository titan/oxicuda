//! Host (CPU) reference implementations for cross-checking GPU primitive output.
//!
//! Every device template in this crate generates PTX that runs on the GPU.
//! During development it is invaluable to run the *same algorithm* on the CPU
//! over the input and compare results.  This module provides that "host
//! reference mode" for the core primitives.
//!
//! These references operate directly on plain slices (`&[T]`) — the crate
//! deliberately avoids pulling in `ndarray`; a slice is the natural shape for a
//! flat device buffer mirrored on the host.
//!
//! # Coverage
//!
//! | Function                         | GPU template mirrored                         |
//! |----------------------------------|-----------------------------------------------|
//! | [`reference_reduce`]             | [`crate::device::reduce`]                     |
//! | [`reference_scan`]               | [`crate::device::scan`] / `decoupled_scan`    |
//! | [`reference_histogram_modulo`]   | [`crate::device::histogram`] (`Modulo`)       |
//! | [`reference_histogram_even`]     | [`crate::device::histogram`] (`EvenRange`)    |
//! | [`reference_select`]             | [`crate::device::select`]                     |
//!
//! Run-length-encode, segmented, partition, unique, and the sort references
//! live next to their respective templates and are re-exported here.
//!
//! # Boundary sizes
//!
//! These references are the natural oracle for the warp/block/cross-block
//! boundary sizes `n ∈ {1, 31, 32, 33, 1023, 1024, 1025, …}` that GPU scans and
//! reductions must get right.

use crate::ptx_helpers::ReduceOp;

// Re-export the references that live beside their templates.
pub use crate::device::decoupled_scan::reference_decoupled_scan_u64;
pub use crate::device::partition::{reference_partition, reference_select_unique};
pub use crate::device::run_length_encode::reference_run_length_encode;
pub use crate::device::segmented::{reference_segmented_reduce_u64, reference_segmented_scan_u64};
pub use crate::sort::merge_pairs::reference_merge_pairs;
pub use crate::sort::onesweep::{reference_onesweep_pass_u32, reference_onesweep_sort_u32};
pub use crate::sort::radix_sort_8bit::reference_radix8_sort_u32;
pub use crate::sort::radix_sort_pairs::{
    reference_sort_pairs_by_key, twiddle_f32_forward, twiddle_f32_inverse, twiddle_f64_forward,
    twiddle_f64_inverse,
};
pub use crate::sort::segmented_sort::reference_segmented_sort_u64;

/// Output direction for [`reference_scan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HostScanKind {
    /// `out[i]` includes `in[i]`.
    Inclusive,
    /// `out[i]` excludes `in[i]`.
    Exclusive,
}

// ─── Integer references (u64-domain, exact) ─────────────────────────────────────

fn apply_u64(op: ReduceOp, a: u64, b: u64) -> u64 {
    match op {
        ReduceOp::Sum => a.wrapping_add(b),
        ReduceOp::Product => a.wrapping_mul(b),
        ReduceOp::Min => a.min(b),
        ReduceOp::Max => a.max(b),
        ReduceOp::And => a & b,
        ReduceOp::Or => a | b,
        ReduceOp::Xor => a ^ b,
    }
}

fn identity_u64(op: ReduceOp) -> u64 {
    match op {
        ReduceOp::Sum | ReduceOp::Or | ReduceOp::Xor => 0,
        ReduceOp::Product => 1,
        ReduceOp::Min => u64::MAX,
        ReduceOp::Max => 0,
        ReduceOp::And => u64::MAX,
    }
}

/// Host reference for device-wide reduction over integer keys.
///
/// Returns the identity element for an empty input.
#[must_use]
pub fn reference_reduce(op: ReduceOp, data: &[u64]) -> u64 {
    data.iter()
        .copied()
        .fold(identity_u64(op), |acc, v| apply_u64(op, acc, v))
}

/// Host reference for device-wide prefix scan over integer keys.
#[must_use]
pub fn reference_scan(op: ReduceOp, kind: HostScanKind, data: &[u64]) -> Vec<u64> {
    let mut out = Vec::with_capacity(data.len());
    let mut acc = identity_u64(op);
    for &v in data {
        match kind {
            HostScanKind::Exclusive => {
                out.push(acc);
                acc = apply_u64(op, acc, v);
            }
            HostScanKind::Inclusive => {
                acc = apply_u64(op, acc, v);
                out.push(acc);
            }
        }
    }
    out
}

// ─── Floating-point references ──────────────────────────────────────────────────

/// Host reference for device-wide `f64` reduction.  `Sum`/`Product`/`Min`/`Max`
/// are supported; bitwise ops are meaningless for floats and return `f64::NAN`.
#[must_use]
pub fn reference_reduce_f64(op: ReduceOp, data: &[f64]) -> f64 {
    match op {
        ReduceOp::Sum => data.iter().copied().sum(),
        ReduceOp::Product => data.iter().copied().product(),
        ReduceOp::Min => data.iter().copied().fold(f64::INFINITY, f64::min),
        ReduceOp::Max => data.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ReduceOp::And | ReduceOp::Or | ReduceOp::Xor => f64::NAN,
    }
}

// ─── Histogram references ───────────────────────────────────────────────────────

/// Host reference for `Modulo`-mode histogram: `bin = value % num_bins`.
#[must_use]
pub fn reference_histogram_modulo(data: &[u32], num_bins: u32) -> Vec<u32> {
    let mut bins = vec![0u32; num_bins as usize];
    if num_bins == 0 {
        return bins;
    }
    for &v in data {
        bins[(v % num_bins) as usize] += 1;
    }
    bins
}

/// Host reference for `EvenRange`-mode histogram over integer keys.
///
/// Values in `[lo, hi)` map linearly into `num_bins` equal-width bins; values
/// outside the range are dropped (matching the GPU kernel's bounds guard).
#[must_use]
pub fn reference_histogram_even(data: &[u32], lo: u32, hi: u32, num_bins: u32) -> Vec<u32> {
    let mut bins = vec![0u32; num_bins as usize];
    if num_bins == 0 || hi <= lo {
        return bins;
    }
    let range = u64::from(hi) - u64::from(lo);
    for &v in data {
        if v < lo || v >= hi {
            continue;
        }
        let offset = u64::from(v) - u64::from(lo);
        let bin = (offset * u64::from(num_bins) / range) as u32;
        let bin = bin.min(num_bins - 1);
        bins[bin as usize] += 1;
    }
    bins
}

// ─── Select reference ───────────────────────────────────────────────────────────

/// Host reference for device-wide stream compaction (keep where `pred` holds),
/// preserving input order.
#[must_use]
pub fn reference_select<T: Copy>(data: &[T], pred: impl Fn(T) -> bool) -> Vec<T> {
    data.iter().copied().filter(|&x| pred(x)).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_sum_and_max() {
        let data = [1u64, 5, 3, 9, 2];
        assert_eq!(reference_reduce(ReduceOp::Sum, &data), 20);
        assert_eq!(reference_reduce(ReduceOp::Max, &data), 9);
        assert_eq!(reference_reduce(ReduceOp::Min, &data), 1);
        assert_eq!(reference_reduce(ReduceOp::Xor, &[3u64, 5, 6]), 3 ^ 5 ^ 6);
    }

    #[test]
    fn reduce_empty_is_identity() {
        assert_eq!(reference_reduce(ReduceOp::Sum, &[]), 0);
        assert_eq!(reference_reduce(ReduceOp::Product, &[]), 1);
        assert_eq!(reference_reduce(ReduceOp::Min, &[]), u64::MAX);
        assert_eq!(reference_reduce(ReduceOp::And, &[]), u64::MAX);
    }

    #[test]
    fn scan_boundary_sizes() {
        // The GPU scan must be correct at warp/block boundaries; the reference
        // is the oracle for those sizes.
        for &n in &[1usize, 31, 32, 33, 1023, 1024, 1025] {
            let data: Vec<u64> = (1..=n as u64).collect();
            let inc = reference_scan(ReduceOp::Sum, HostScanKind::Inclusive, &data);
            assert_eq!(inc.len(), n);
            // Inclusive sum of 1..=n is n*(n+1)/2 at the last element.
            assert_eq!(
                *inc.last().expect("nonempty"),
                (n as u64) * (n as u64 + 1) / 2
            );
            let exc = reference_scan(ReduceOp::Sum, HostScanKind::Exclusive, &data);
            assert_eq!(exc[0], 0);
            if n > 1 {
                assert_eq!(exc[1], 1);
            }
        }
    }

    #[test]
    fn reduce_f64_ops() {
        let data = [1.0_f64, 2.0, 4.0];
        assert!((reference_reduce_f64(ReduceOp::Sum, &data) - 7.0).abs() < 1e-12);
        assert!((reference_reduce_f64(ReduceOp::Product, &data) - 8.0).abs() < 1e-12);
        assert!((reference_reduce_f64(ReduceOp::Max, &data) - 4.0).abs() < 1e-12);
        assert!((reference_reduce_f64(ReduceOp::Min, &data) - 1.0).abs() < 1e-12);
        assert!(reference_reduce_f64(ReduceOp::And, &data).is_nan());
    }

    #[test]
    fn histogram_modulo_counts() {
        let data = [0u32, 1, 2, 3, 4, 5, 8];
        let h = reference_histogram_modulo(&data, 4);
        // bins: 0→{0,4,8}=3, 1→{1,5}=2, 2→{2}=1, 3→{3}=1
        assert_eq!(h, vec![3, 2, 1, 1]);
    }

    #[test]
    fn histogram_even_range_and_clamp() {
        let data = [0u32, 2, 4, 6, 8, 10, 99];
        // [0,10) into 5 bins of width 2: 0,2→b0,b1 ... 99 is OOB (dropped).
        let h = reference_histogram_even(&data, 0, 10, 5);
        // 0→0, 2→1, 4→2, 6→3, 8→4, 10 OOB, 99 OOB.
        assert_eq!(h, vec![1, 1, 1, 1, 1]);
        assert_eq!(h.iter().sum::<u32>(), 5);
    }

    #[test]
    fn select_keeps_predicate() {
        let kept = reference_select(&[1i32, -2, 3, -4, 5], |x| x > 0);
        assert_eq!(kept, vec![1, 3, 5]);
        let nonzero = reference_select(&[0u32, 7, 0, 9], |x| x != 0);
        assert_eq!(nonzero, vec![7, 9]);
    }

    #[test]
    fn reexports_are_callable() {
        // Smoke-test a couple of the re-exported references to confirm wiring.
        let (vals, lens) = reference_run_length_encode(&[1u32, 1, 2]);
        assert_eq!(vals, vec![1, 2]);
        assert_eq!(lens, vec![2, 1]);
        assert_eq!(reference_select_unique(&[5u32, 5, 6]), vec![5, 6]);
        let scan = reference_decoupled_scan_u64(
            ReduceOp::Sum,
            crate::device::decoupled_scan::ScanKind::Inclusive,
            &[1, 2, 3],
            256,
        );
        assert_eq!(scan, vec![1, 3, 6]);
    }
}
