//! Sparse-gradient fast path for PackNet / Piggyback mask application.
//!
//! When a Piggyback or PackNet mask is pruned past ~90 % the dense
//! representation wastes time and bandwidth iterating over inactive weights
//! that contribute nothing. This module stores **only the active indices**
//! and exposes four primitives:
//!
//! * [`SparseActiveMask::from_dense`] — derive a sparse view from a dense
//!   `[f64]` mask by a magnitude threshold,
//! * [`sparse_mask_apply`]            — zero-out inactive positions in place,
//! * [`sparse_mask_backward`]         — propagate an upstream gradient through
//!   the binary mask,
//! * [`sparse_mask_compact`] /
//!   [`sparse_mask_scatter`]          — round-trip between dense and the
//!   active-index-compact representation for downstream sparse ops.
//!
//! For the dense path see [`super::packnet::apply_mask`] and
//! [`super::piggyback::piggyback_forward`].

#![forbid(unsafe_code)]

use crate::error::{ContinualError, ContinualResult};

// ─── SparseActiveMask ────────────────────────────────────────────────────────

/// Sparse representation of a binary / pruned mask.
///
/// `active_indices` is kept sorted ascending so that the in-place apply
/// kernel can use a simple two-pointer sweep over the value buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseActiveMask {
    pub n_weights: usize,
    pub active_indices: Vec<usize>,
    pub sparsity: f64,
}

impl SparseActiveMask {
    /// Build a sparse view from a dense mask `[f64]`.
    ///
    /// A weight at index `i` is **active** iff `mask[i].abs() > threshold`,
    /// so `threshold = 0.0` keeps every non-zero entry and `threshold = 0.5`
    /// splits a {0, 1}-valued binary mask cleanly.
    ///
    /// # Errors
    /// * [`ContinualError::Internal`] if `threshold` is negative or not finite.
    pub fn from_dense(mask: &[f64], threshold: f64) -> ContinualResult<Self> {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(ContinualError::Internal(format!(
                "sparse_mask_apply: threshold must be >= 0 and finite, got {threshold}"
            )));
        }
        let n_weights = mask.len();
        let mut active_indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v.abs() > threshold).then_some(i))
            .collect();
        active_indices.sort_unstable();
        let sparsity = if n_weights == 0 {
            0.0
        } else {
            1.0 - (active_indices.len() as f64) / (n_weights as f64)
        };
        Ok(Self {
            n_weights,
            active_indices,
            sparsity,
        })
    }

    /// Number of active (kept) weights.
    #[must_use]
    #[inline]
    pub fn n_active(&self) -> usize {
        self.active_indices.len()
    }

    /// True when the dense buffer is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n_weights == 0
    }

    /// True iff the fast path is worthwhile (`sparsity > 0.9`).
    #[must_use]
    #[inline]
    pub fn is_sparse_enough(&self) -> bool {
        self.sparsity > 0.9
    }
}

// ─── Kernels ─────────────────────────────────────────────────────────────────

#[inline]
fn check_len(values_len: usize, mask: &SparseActiveMask) -> ContinualResult<()> {
    if values_len != mask.n_weights {
        return Err(ContinualError::DimensionMismatch {
            expected: mask.n_weights,
            got: values_len,
        });
    }
    Ok(())
}

/// In-place element-wise multiply by the binary mask: inactive positions are
/// zeroed, active positions keep their incoming value.
///
/// Uses a two-pointer sweep over `values` so each element is touched at most
/// once even when the active set is much smaller than `n_weights`.
pub fn sparse_mask_apply(values: &mut [f64], mask: &SparseActiveMask) -> ContinualResult<()> {
    check_len(values.len(), mask)?;
    let mut prev = 0usize;
    for &i in &mask.active_indices {
        for v in values.iter_mut().take(i).skip(prev) {
            *v = 0.0;
        }
        prev = i + 1;
    }
    for v in values.iter_mut().skip(prev) {
        *v = 0.0;
    }
    Ok(())
}

/// Propagate an upstream gradient through the binary mask.
///
/// Because the binary mask is the identity on active positions and zero
/// elsewhere, the gradient w.r.t. the unmasked weights is exactly the
/// upstream gradient at active indices and zero at inactive indices.
pub fn sparse_mask_backward(
    upstream_grad: &[f64],
    mask: &SparseActiveMask,
) -> ContinualResult<Vec<f64>> {
    check_len(upstream_grad.len(), mask)?;
    let mut out = vec![0.0_f64; mask.n_weights];
    for &i in &mask.active_indices {
        out[i] = upstream_grad[i];
    }
    Ok(out)
}

/// Compact-representation forward: returns the values located at the active
/// indices only, in active-index order. `result.len() == mask.n_active()`.
pub fn sparse_mask_compact(values: &[f64], mask: &SparseActiveMask) -> ContinualResult<Vec<f64>> {
    check_len(values.len(), mask)?;
    Ok(mask.active_indices.iter().map(|&i| values[i]).collect())
}

/// Inverse of [`sparse_mask_compact`]: scatter a compact buffer back into a
/// dense layout of length `n_weights` with zeros at inactive positions.
///
/// # Errors
/// * [`ContinualError::DimensionMismatch`] if `compact.len() != mask.n_active()`.
pub fn sparse_mask_scatter(compact: &[f64], mask: &SparseActiveMask) -> ContinualResult<Vec<f64>> {
    if compact.len() != mask.n_active() {
        return Err(ContinualError::DimensionMismatch {
            expected: mask.n_active(),
            got: compact.len(),
        });
    }
    let mut out = vec![0.0_f64; mask.n_weights];
    for (k, &i) in mask.active_indices.iter().enumerate() {
        out[i] = compact[k];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_mask_yields_empty_sparse_view() {
        let m = SparseActiveMask::from_dense(&[], 0.0)
            .expect("sparse mask should construct from valid dense mask");
        assert_eq!(m.n_weights, 0);
        assert!(m.is_empty());
        assert_eq!(m.n_active(), 0);
        assert_eq!(m.sparsity, 0.0);
        assert!(!m.is_sparse_enough());
    }

    #[test]
    fn all_active_mask() {
        let dense = vec![1.0_f64; 8];
        let m = SparseActiveMask::from_dense(&dense, 0.0)
            .expect("sparse mask should construct from valid dense mask");
        assert_eq!(m.n_active(), 8);
        assert_eq!(m.active_indices, (0..8).collect::<Vec<_>>());
        assert!(m.sparsity.abs() < 1e-15);
        assert!(!m.is_sparse_enough());
    }

    #[test]
    fn all_inactive_mask() {
        let dense = vec![0.0_f64; 8];
        let m = SparseActiveMask::from_dense(&dense, 0.0)
            .expect("sparse mask should construct from valid dense mask");
        assert_eq!(m.n_active(), 0);
        assert!((m.sparsity - 1.0).abs() < 1e-15);
        assert!(m.is_sparse_enough());
    }

    #[test]
    fn threshold_filters_correctly() {
        let dense = vec![0.1_f64, 0.6, -0.4, 0.9, 0.5, -0.5];
        let m = SparseActiveMask::from_dense(&dense, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        assert_eq!(m.active_indices, vec![1, 3]);
        assert!((m.sparsity - 4.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn threshold_negative_or_nan_errors() {
        let dense = vec![1.0_f64; 4];
        assert!(SparseActiveMask::from_dense(&dense, -0.1).is_err());
        assert!(SparseActiveMask::from_dense(&dense, f64::NAN).is_err());
        assert!(SparseActiveMask::from_dense(&dense, f64::INFINITY).is_err());
    }

    #[test]
    fn is_sparse_enough_boundary() {
        let mut dense = vec![0.0_f64; 100];
        dense[0..9].fill(1.0);
        let m = SparseActiveMask::from_dense(&dense, 0.0)
            .expect("sparse mask should construct from valid dense mask");
        assert!(m.is_sparse_enough(), "9/100 active → sparsity 0.91 > 0.9");

        dense[9] = 1.0;
        let m = SparseActiveMask::from_dense(&dense, 0.0)
            .expect("sparse mask should construct from valid dense mask");
        assert!(
            !m.is_sparse_enough(),
            "10/100 active → sparsity 0.9, not strictly greater"
        );
    }

    #[test]
    fn apply_zeroes_inactive_only() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let mut values = vec![10.0_f64, 20.0, 30.0, 40.0, 50.0];
        sparse_mask_apply(&mut values, &m)
            .expect("sparse mask application should succeed with matching dimensions");
        assert_eq!(values, vec![10.0, 0.0, 30.0, 0.0, 50.0]);
    }

    #[test]
    fn apply_all_active_is_identity_on_values() {
        let dense_mask = vec![1.0_f64; 6];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let mut values = vec![1.5_f64, -2.5, 3.5, -4.5, 5.5, -6.5];
        let copy = values.clone();
        sparse_mask_apply(&mut values, &m)
            .expect("sparse mask application should succeed with matching dimensions");
        assert_eq!(values, copy);
    }

    #[test]
    fn apply_all_inactive_zeroes_everything() {
        let dense_mask = vec![0.0_f64; 6];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let mut values = vec![1.0_f64; 6];
        sparse_mask_apply(&mut values, &m)
            .expect("sparse mask application should succeed with matching dimensions");
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn apply_length_mismatch_errors() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let mut values = vec![1.0_f64; 5];
        let err = sparse_mask_apply(&mut values, &m).unwrap_err();
        assert!(matches!(
            err,
            ContinualError::DimensionMismatch {
                expected: 3,
                got: 5
            }
        ));
    }

    #[test]
    fn backward_zero_fills_inactive_and_keeps_active() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let upstream = vec![7.0_f64, 8.0, 9.0, 10.0, 11.0];
        let grad =
            sparse_mask_backward(&upstream, &m).expect("sparse mask backward should succeed");
        assert_eq!(grad, vec![7.0, 0.0, 9.0, 0.0, 11.0]);
    }

    #[test]
    fn backward_length_mismatch_errors() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let upstream = vec![1.0_f64; 4];
        assert!(sparse_mask_backward(&upstream, &m).is_err());
    }

    #[test]
    fn compact_round_trip_preserves_active_values() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let values = vec![10.0_f64, 99.0, 30.0, 99.0, 50.0, 99.0, 70.0];
        let compact =
            sparse_mask_compact(&values, &m).expect("sparse mask compaction should succeed");
        assert_eq!(compact, vec![10.0, 30.0, 50.0, 70.0]);
        let scattered =
            sparse_mask_scatter(&compact, &m).expect("sparse mask scatter should succeed");
        assert_eq!(scattered, vec![10.0, 0.0, 30.0, 0.0, 50.0, 0.0, 70.0]);
    }

    #[test]
    fn compact_length_mismatch_errors() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let values = vec![1.0_f64, 2.0];
        assert!(sparse_mask_compact(&values, &m).is_err());
    }

    #[test]
    fn scatter_length_mismatch_errors() {
        let dense_mask = vec![1.0_f64, 0.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let compact = vec![1.0_f64, 2.0, 3.0];
        let err = sparse_mask_scatter(&compact, &m).unwrap_err();
        assert!(matches!(
            err,
            ContinualError::DimensionMismatch {
                expected: 2,
                got: 3
            }
        ));
    }

    #[test]
    fn sparsity_computed_correctly_for_typical_packnet_density() {
        let mut dense = vec![0.0_f64; 1000];
        for i in (0..1000).step_by(20) {
            dense[i] = 1.0;
        }
        let m = SparseActiveMask::from_dense(&dense, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        assert_eq!(m.n_active(), 50);
        assert!((m.sparsity - 0.95).abs() < 1e-12);
        assert!(m.is_sparse_enough());
    }

    #[test]
    fn apply_followed_by_backward_consistent_with_mul() {
        let dense_mask = vec![0.0_f64, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let m = SparseActiveMask::from_dense(&dense_mask, 0.5)
            .expect("sparse mask should construct from valid dense mask");
        let original = vec![1.1_f64, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8];

        let mut applied = original.clone();
        sparse_mask_apply(&mut applied, &m)
            .expect("sparse mask application should succeed with matching dimensions");
        let back =
            sparse_mask_backward(&original, &m).expect("sparse mask backward should succeed");
        assert_eq!(applied, back, "for binary masks: apply == backward");
    }
}
