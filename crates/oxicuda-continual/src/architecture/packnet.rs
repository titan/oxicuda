//! PackNet: Iterative pruning and task-specific binary masks.
//!
//! Implements the method from:
//! Mallya & Lazebnik. "PackNet: Adding Multiple Tasks to a Single Network
//! by Iterative Pruning." CVPR 2018.
//!
//! PackNet assigns exclusive subnetworks to each task by pruning the
//! model after each task and freezing the pruned weights. Subsequent tasks
//! only modify the remaining free capacity.

use crate::error::{ContinualError, ContinualResult};

/// Binary weight mask for a single task in PackNet.
///
/// `mask[i] = 1` means the weight is active (kept) for this task,
/// `mask[i] = 0` means it is pruned/frozen.
#[derive(Debug, Clone)]
pub struct PackNetMask {
    /// Binary mask: 1 = keep, 0 = prune/freeze.
    pub mask: Vec<u8>,
    /// Task identifier this mask was created for.
    pub task_id: usize,
    /// Fraction of weights pruned (0 = none, 1 = all).
    pub sparsity: f32,
}

impl PackNetMask {
    /// Number of parameters covered by this mask.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mask.len()
    }

    /// True if the mask covers no parameters.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    /// Count the number of active (kept) weights.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.mask.iter().filter(|&&b| b != 0).count()
    }
}

/// Prune weights by L1-magnitude and return a binary mask.
///
/// Keeps the top `(1 - sparsity_fraction)` fraction of weights by absolute
/// value; sets the rest to masked (0).
///
/// Returns `Err` if `sparsity_fraction` is not in `[0, 1)` or `weights` is empty.
pub fn prune_weights_l1(
    weights: &[f32],
    sparsity_fraction: f32,
    task_id: usize,
) -> ContinualResult<PackNetMask> {
    if !sparsity_fraction.is_finite() || !(0.0..1.0).contains(&sparsity_fraction) {
        return Err(ContinualError::InvalidSparsityFraction {
            fraction: sparsity_fraction,
        });
    }
    if weights.is_empty() {
        return Err(ContinualError::EmptyInput);
    }
    let n = weights.len();
    // Number of weights to prune
    let n_prune = (n as f32 * sparsity_fraction).floor() as usize;
    let n_keep = n - n_prune;

    // Sort indices by |weight| descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        weights[b]
            .abs()
            .partial_cmp(&weights[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Top n_keep indices are kept (mask=1), rest are pruned (mask=0)
    let mut mask = vec![0u8; n];
    for &idx in indices.iter().take(n_keep) {
        mask[idx] = 1;
    }

    Ok(PackNetMask {
        mask,
        task_id,
        sparsity: sparsity_fraction,
    })
}

/// Apply a PackNet mask in-place: `weights[i] *= mask[i]`.
///
/// Masked positions (mask=0) are zeroed out.
pub fn apply_mask(weights: &mut [f32], mask: &PackNetMask) -> ContinualResult<()> {
    if weights.len() != mask.mask.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: mask.mask.len(),
            got: weights.len(),
        });
    }
    for (w, &m) in weights.iter_mut().zip(mask.mask.iter()) {
        if m == 0 {
            *w = 0.0;
        }
    }
    Ok(())
}

/// Compute the combined frozen mask across all previous tasks.
///
/// A weight is frozen (result mask = 1) if it was active in ANY previous task
/// mask. The caller should NOT modify frozen weights for the current task.
///
/// `current_task`: index of the task currently being trained (excluded from freeze).
pub fn freeze_task_weights(masks: &[PackNetMask], current_task: usize) -> Vec<u8> {
    if masks.is_empty() {
        return vec![];
    }
    let n = masks[0].mask.len();
    let mut frozen = vec![0u8; n];
    for mask in masks {
        if mask.task_id == current_task {
            continue;
        }
        for (f, &m) in frozen.iter_mut().zip(mask.mask.iter()) {
            if m != 0 {
                *f = 1;
            }
        }
    }
    frozen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_sparsity_fraction_respected() {
        let weights: Vec<f32> = (1..=10).map(|i| i as f32).collect();
        let mask = prune_weights_l1(&weights, 0.5, 0)
            .expect("L1 pruning should succeed with valid weight and sparsity");
        let n_keep = mask.n_active();
        // 0.5 sparsity → keep 5 weights
        assert_eq!(n_keep, 5, "Should keep 50% of weights");
    }

    #[test]
    fn prune_keeps_largest_weights() {
        let weights = vec![0.1_f32, 5.0, 0.01, 3.0, 0.001];
        let mask = prune_weights_l1(&weights, 0.6, 0)
            .expect("L1 pruning should succeed with valid sparsity");
        // Keep top 40% = 2 weights → indices 1 (5.0) and 3 (3.0)
        assert_eq!(mask.mask[1], 1, "Largest weight should be kept");
        assert_eq!(mask.mask[3], 1, "Second largest weight should be kept");
        assert_eq!(mask.mask[0], 0);
        assert_eq!(mask.mask[2], 0);
        assert_eq!(mask.mask[4], 0);
    }

    #[test]
    fn apply_mask_zeroes_pruned_weights() {
        let mut weights = vec![1.0_f32; 8];
        let mask = PackNetMask {
            mask: vec![1, 0, 1, 0, 1, 0, 1, 0],
            task_id: 0,
            sparsity: 0.5,
        };
        apply_mask(&mut weights, &mask)
            .expect("mask application should succeed with matching dimensions");
        for (w, m) in weights.iter().zip(mask.mask.iter()) {
            if *m == 0 {
                assert_eq!(*w, 0.0, "Masked weight should be zero");
            } else {
                assert_eq!(*w, 1.0, "Kept weight should be unchanged");
            }
        }
    }

    #[test]
    fn freeze_task_weights_combines_masks() {
        let mask0 = PackNetMask {
            mask: vec![1, 0, 1, 0],
            task_id: 0,
            sparsity: 0.5,
        };
        let mask1 = PackNetMask {
            mask: vec![0, 1, 0, 1],
            task_id: 1,
            sparsity: 0.5,
        };
        // Current task = 2, freeze all of task0 and task1
        let frozen = freeze_task_weights(&[mask0, mask1], 2);
        assert_eq!(frozen, vec![1, 1, 1, 1]);
    }

    #[test]
    fn freeze_excludes_current_task() {
        let mask0 = PackNetMask {
            mask: vec![1, 0, 1, 0],
            task_id: 0,
            sparsity: 0.5,
        };
        let mask1 = PackNetMask {
            mask: vec![0, 1, 0, 1],
            task_id: 1,
            sparsity: 0.5,
        };
        // Current task = 1, only freeze task0
        let frozen = freeze_task_weights(&[mask0, mask1], 1);
        assert_eq!(frozen, vec![1, 0, 1, 0]);
    }

    #[test]
    fn prune_invalid_sparsity_returns_err() {
        let weights = vec![1.0_f32; 10];
        assert!(prune_weights_l1(&weights, -0.1, 0).is_err());
        assert!(prune_weights_l1(&weights, 1.0, 0).is_err());
        assert!(prune_weights_l1(&weights, f32::NAN, 0).is_err());
    }

    #[test]
    fn prune_empty_returns_err() {
        assert!(prune_weights_l1(&[], 0.5, 0).is_err());
    }

    #[test]
    fn apply_mask_dimension_mismatch() {
        let mut weights = vec![1.0_f32; 4];
        let mask = PackNetMask {
            mask: vec![1, 0, 1],
            task_id: 0,
            sparsity: 0.33,
        };
        assert!(apply_mask(&mut weights, &mask).is_err());
    }

    #[test]
    fn packnet_mask_n_active() {
        let mask = PackNetMask {
            mask: vec![1, 0, 1, 1, 0],
            task_id: 0,
            sparsity: 0.4,
        };
        assert_eq!(mask.n_active(), 3);
    }
}
