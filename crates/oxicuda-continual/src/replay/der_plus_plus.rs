//! Dark Experience Replay ++ (DER++) loss computation.
//!
//! Implements the loss from:
//! Buzzega et al. "Dark Experience for General Continual Learning:
//! a Strong, Simple Baseline." NeurIPS 2020.
//!
//! The DER++ objective for a mini-batch of `n_current` new samples and
//! `n_mem` replayed memory samples is:
//!
//! ```text
//! L = CE(z_cur, y_cur)
//!   + α · MSE(z_mem, z̃_mem)
//!   + β · CE(z_mem, y_mem)
//! ```
//!
//! where:
//! - `z_cur`  — current model logits on new data  (`n_current × n_classes`)
//! - `y_cur`  — ground-truth labels for new data
//! - `z_mem`  — current model logits on memory samples (`n_mem × n_classes`)
//! - `z̃_mem` — logits saved when a sample was first stored in memory
//! - `y_mem`  — labels for memory samples
//!
//! The MSE term distils "dark knowledge" from the stored logits (hence the
//! name), while the CE term on memory explicitly prevents forgetting.

use crate::error::{ContinualError, ContinualResult};

// ─── DerPpLoss ────────────────────────────────────────────────────────────────

/// DER++ loss weighting configuration.
///
/// `alpha` controls the dark-experience MSE distillation weight; `beta`
/// controls the cross-entropy weight on replayed memory samples.
#[derive(Debug, Clone)]
pub struct DerPpLoss {
    /// Weight for the MSE distillation term (α ≥ 0).
    pub alpha: f32,
    /// Weight for the memory cross-entropy term (β ≥ 0).
    pub beta: f32,
}

impl DerPpLoss {
    /// Construct a [`DerPpLoss`] with the given α and β.
    ///
    /// # Errors
    ///
    /// Returns [`ContinualError::InvalidLambda`] if either `alpha` or `beta`
    /// is negative or non-finite.
    pub fn new(alpha: f32, beta: f32) -> ContinualResult<Self> {
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: alpha });
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(ContinualError::InvalidLambda { lambda: beta });
        }
        Ok(Self { alpha, beta })
    }

    /// Compute the full DER++ loss.
    ///
    /// # Arguments
    ///
    /// - `current_logits`  — flat `[n_current × n_classes]`
    /// - `current_labels`  — `[n_current]`, each in `[0, n_classes)`
    /// - `mem_logits`      — flat `[n_mem × n_classes]` (current model on memory)
    /// - `mem_labels`      — `[n_mem]`, each in `[0, n_classes)`
    /// - `mem_old_logits`  — flat `[n_mem × n_classes]` (stored teacher logits)
    /// - `n_current`       — number of new-task samples (must be ≥ 1)
    /// - `n_mem`           — number of memory samples (may be 0)
    /// - `n_classes`       — number of output classes (must be ≥ 1)
    ///
    /// # Errors
    ///
    /// - [`ContinualError::EmptyInput`] if `n_current == 0`.
    /// - [`ContinualError::DimensionMismatch`] if any slice length is
    ///   inconsistent with `(n_current | n_mem) × n_classes`.
    /// - [`ContinualError::DimensionMismatch`] if any label ≥ `n_classes`.
    pub fn compute(
        &self,
        current_logits: &[f32],
        current_labels: &[usize],
        mem_logits: &[f32],
        mem_labels: &[usize],
        mem_old_logits: &[f32],
        n_current: usize,
        n_mem: usize,
        n_classes: usize,
    ) -> ContinualResult<f32> {
        // ── Guard: n_current must be at least 1 ──────────────────────────────
        if n_current == 0 {
            return Err(ContinualError::EmptyInput);
        }

        // ── Validate current slice dimensions ─────────────────────────────────
        let expected_cur = n_current * n_classes;
        if current_logits.len() != expected_cur {
            return Err(ContinualError::DimensionMismatch {
                expected: expected_cur,
                got: current_logits.len(),
            });
        }
        if current_labels.len() != n_current {
            return Err(ContinualError::DimensionMismatch {
                expected: n_current,
                got: current_labels.len(),
            });
        }

        // ── Validate memory slice dimensions (only when n_mem > 0) ────────────
        if n_mem > 0 {
            let expected_mem = n_mem * n_classes;
            if mem_logits.len() != expected_mem {
                return Err(ContinualError::DimensionMismatch {
                    expected: expected_mem,
                    got: mem_logits.len(),
                });
            }
            if mem_labels.len() != n_mem {
                return Err(ContinualError::DimensionMismatch {
                    expected: n_mem,
                    got: mem_labels.len(),
                });
            }
            if mem_old_logits.len() != expected_mem {
                return Err(ContinualError::DimensionMismatch {
                    expected: expected_mem,
                    got: mem_old_logits.len(),
                });
            }
        }

        // ── Validate labels ───────────────────────────────────────────────────
        for &label in current_labels {
            if label >= n_classes {
                return Err(ContinualError::DimensionMismatch {
                    expected: n_classes.saturating_sub(1),
                    got: label,
                });
            }
        }
        for &label in mem_labels {
            if label >= n_classes {
                return Err(ContinualError::DimensionMismatch {
                    expected: n_classes.saturating_sub(1),
                    got: label,
                });
            }
        }

        // ── Term 1: CE on new-task data ───────────────────────────────────────
        let ce_current = cross_entropy_mean(current_logits, current_labels, n_current, n_classes)?;

        // ── Term 2 & 3: MSE + CE on memory (only when n_mem > 0) ─────────────
        let (mse_mem, ce_mem) = if n_mem > 0 {
            let mse = mse_mean(mem_logits, mem_old_logits)?;
            let ce = cross_entropy_mean(mem_logits, mem_labels, n_mem, n_classes)?;
            (mse, ce)
        } else {
            (0.0_f32, 0.0_f32)
        };

        let total = ce_current + self.alpha * mse_mem + self.beta * ce_mem;

        if !total.is_finite() {
            return Err(ContinualError::NanEncountered {
                location: "DerPpLoss::compute",
            });
        }

        Ok(total)
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Numerically stable softmax cross-entropy averaged over `n` samples.
///
/// For each sample `i`:
/// ```text
/// CE_i = -log(softmax(logits[i])[label[i]])
///      = -logits[i][label] + log(Σ_j exp(logits[i][j] - max_i))
/// ```
/// (max is subtracted from every logit before exp for numerical stability.)
fn cross_entropy_mean(
    logits: &[f32],
    labels: &[usize],
    n: usize,
    n_classes: usize,
) -> ContinualResult<f32> {
    let mut total_ce = 0.0_f32;

    for (i, &label) in labels.iter().enumerate().take(n) {
        let start = i * n_classes;
        let row = &logits[start..start + n_classes];

        // Subtract max for numerical stability.
        let max_z = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exps: Vec<f32> = row.iter().map(|&z| (z - max_z).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();

        // CE = -(logit[label] - max_z) + log(sum_exp)
        let log_prob = (row[label] - max_z) - sum_exp.max(1e-30_f32).ln();
        let ce = -log_prob;

        if !ce.is_finite() {
            return Err(ContinualError::NanEncountered {
                location: "cross_entropy_mean",
            });
        }

        total_ce += ce;
    }

    Ok(total_ce / n as f32)
}

/// Mean squared error averaged over all elements of two equal-length slices.
fn mse_mean(a: &[f32], b: &[f32]) -> ContinualResult<f32> {
    if a.len() != b.len() {
        return Err(ContinualError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.is_empty() {
        return Ok(0.0_f32);
    }
    let sum_sq: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum();
    let mse = sum_sq / a.len() as f32;
    if !mse.is_finite() {
        return Err(ContinualError::NanEncountered {
            location: "mse_mean",
        });
    }
    Ok(mse)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers ─────────────────────────────────────────────────────────────────

    /// Create uniform logits (all zeros → uniform softmax).
    fn uniform_logits(n: usize, n_classes: usize) -> Vec<f32> {
        vec![0.0_f32; n * n_classes]
    }

    fn uniform_labels(n: usize, n_classes: usize) -> Vec<usize> {
        (0..n).map(|i| i % n_classes).collect()
    }

    // ── Test 1: loss is finite on valid inputs ────────────────────────────────

    #[test]
    fn loss_finite() {
        let loss_fn = DerPpLoss::new(0.5, 1.0).expect("DerPpLoss::new should succeed");
        let n_cur = 4;
        let n_mem = 2;
        let n_cls = 3;

        let cur_logits = vec![
            1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0,
        ];
        let cur_labels = vec![0_usize, 1, 2, 0];
        let mem_logits = vec![0.2_f32, 0.6, 0.2, 0.1, 0.1, 0.8];
        let mem_labels = vec![1_usize, 2];
        let mem_old = vec![0.3_f32, 0.5, 0.2, 0.0, 0.2, 0.8];

        let result = loss_fn
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_logits,
                &mem_labels,
                &mem_old,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("compute should succeed");

        assert!(result.is_finite(), "loss should be finite, got {result}");
    }

    // ── Test 2: CE loss is always non-negative ────────────────────────────────

    #[test]
    fn loss_nonneg() {
        let loss_fn = DerPpLoss::new(1.0, 1.0).expect("DerPpLoss::new should succeed");
        let n_cur = 3;
        let n_cls = 4;

        let result = loss_fn
            .compute(
                &uniform_logits(n_cur, n_cls),
                &uniform_labels(n_cur, n_cls),
                &[],
                &[],
                &[],
                n_cur,
                0,
                n_cls,
            )
            .expect("compute should succeed");

        assert!(
            result >= 0.0,
            "DER++ loss must be non-negative, got {result}"
        );
    }

    // ── Test 3: alpha=0 → MSE term vanishes regardless of memory logits ───────

    #[test]
    fn alpha_0_no_mse() {
        let loss_fn = DerPpLoss::new(0.0, 0.0).expect("DerPpLoss::new should succeed");
        let n_cur = 2;
        let n_cls = 3;

        let cur_logits = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let cur_labels = vec![0_usize, 1];
        let n_mem = 2;
        let mem_labels = vec![0_usize, 1];

        // mem_logits == mem_old → MSE = 0 already.
        let mem_eq = uniform_logits(n_mem, n_cls);
        let loss_eq = loss_fn
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_eq,
                &mem_labels,
                &mem_eq,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("compute eq should succeed");

        // mem_logits very different from mem_old → but alpha=0 so MSE should be suppressed.
        let mem_cur = vec![10.0_f32, 0.0, 0.0, 0.0, 0.0, 10.0];
        let mem_old = vec![-10.0_f32, 0.0, 0.0, 0.0, 0.0, -10.0];
        let loss_diff = loss_fn
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_cur,
                &mem_labels,
                &mem_old,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("compute diff should succeed");

        assert!(
            (loss_eq - loss_diff).abs() < 1e-5,
            "alpha=0, beta=0 → MSE and mem-CE both suppressed; losses should be equal: {loss_eq} vs {loss_diff}"
        );
    }

    // ── Test 4: beta=0 → memory CE term vanishes ─────────────────────────────

    #[test]
    fn beta_0_no_mem_ce() {
        let n_cur = 2;
        let n_cls = 3;
        let n_mem = 2;
        let cur_logits = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let cur_labels = vec![0_usize, 1];
        let mem_labels_a = vec![0_usize, 1];
        // mem_labels_b targets wrong classes (high CE if beta != 0).
        let mem_labels_b = vec![2_usize, 0];
        let mem_logits = vec![5.0_f32, 0.0, 0.0, 0.0, 5.0, 0.0];
        let mem_old = uniform_logits(n_mem, n_cls);

        let loss_fn = DerPpLoss::new(0.0, 0.0).expect("DerPpLoss::new should succeed");

        let loss_a = loss_fn
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_logits,
                &mem_labels_a,
                &mem_old,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("compute a should succeed");

        let loss_b = loss_fn
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_logits,
                &mem_labels_b,
                &mem_old,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("compute b should succeed");

        assert!(
            (loss_a - loss_b).abs() < 1e-5,
            "beta=0 → memory CE must not affect total loss: {loss_a} vs {loss_b}"
        );
    }

    // ── Test 5: n_current=0 returns Err(EmptyInput) ───────────────────────────

    #[test]
    fn empty_current_error() {
        let loss_fn = DerPpLoss::new(0.5, 0.5).expect("DerPpLoss::new should succeed");
        let result = loss_fn.compute(&[], &[], &[], &[], &[], 0, 0, 3);
        assert!(
            matches!(result, Err(ContinualError::EmptyInput)),
            "n_current=0 should return EmptyInput, got {result:?}"
        );
    }

    // ── Test 6: label >= n_classes returns Err(DimensionMismatch) ─────────────

    #[test]
    fn label_out_of_range_error() {
        let loss_fn = DerPpLoss::new(0.5, 0.5).expect("DerPpLoss::new should succeed");
        let n_cur = 2;
        let n_cls = 3;
        let cur_logits = uniform_logits(n_cur, n_cls);
        // Label 3 is out of range for n_classes=3.
        let bad_labels = vec![0_usize, 3];

        let result = loss_fn.compute(&cur_logits, &bad_labels, &[], &[], &[], n_cur, 0, n_cls);
        assert!(
            result.is_err(),
            "label >= n_classes should return Err, got {result:?}"
        );
        match result.unwrap_err() {
            ContinualError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, n_cls - 1);
                assert_eq!(got, 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ── Test 7: larger alpha increases MSE contribution proportionally ────────

    #[test]
    fn alpha_beta_weight_correctly() {
        let n_cur = 1;
        let n_cls = 2;
        let n_mem = 1;
        // Current data: all zeros.
        let cur_logits = vec![0.0_f32; n_cur * n_cls];
        let cur_labels = vec![0_usize];
        // Memory logits differ significantly from old logits → large MSE.
        let mem_logits = vec![3.0_f32, 0.0];
        let mem_labels = vec![0_usize];
        let mem_old = vec![-3.0_f32, 0.0];

        let loss_low = DerPpLoss::new(0.1, 0.0)
            .expect("low alpha")
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_logits,
                &mem_labels,
                &mem_old,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("low alpha compute");

        let loss_high = DerPpLoss::new(10.0, 0.0)
            .expect("high alpha")
            .compute(
                &cur_logits,
                &cur_labels,
                &mem_logits,
                &mem_labels,
                &mem_old,
                n_cur,
                n_mem,
                n_cls,
            )
            .expect("high alpha compute");

        assert!(
            loss_high > loss_low,
            "larger alpha must give larger total loss: high={loss_high}, low={loss_low}"
        );
    }

    // ── Test 8: n_classes mismatch in current_logits returns Err ──────────────

    #[test]
    fn n_classes_mismatch_error() {
        let loss_fn = DerPpLoss::new(0.5, 0.5).expect("DerPpLoss::new should succeed");
        let n_cur = 2;
        let n_cls = 4;
        // Supply only 3 logits per sample instead of 4.
        let wrong_logits = vec![0.0_f32; n_cur * 3];
        let labels = vec![0_usize; n_cur];

        let result = loss_fn.compute(&wrong_logits, &labels, &[], &[], &[], n_cur, 0, n_cls);
        assert!(
            result.is_err(),
            "logit length mismatch should return Err, got {result:?}"
        );
        match result.unwrap_err() {
            ContinualError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, n_cur * n_cls);
                assert_eq!(got, n_cur * 3);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    // ── Test 9: negative alpha returns Err(InvalidLambda) ────────────────────

    #[test]
    fn negative_alpha_returns_err() {
        let result = DerPpLoss::new(-1.0, 0.5);
        assert!(
            matches!(result, Err(ContinualError::InvalidLambda { .. })),
            "negative alpha should return InvalidLambda, got {result:?}"
        );
    }

    // ── Test 10: negative beta returns Err(InvalidLambda) ────────────────────

    #[test]
    fn negative_beta_returns_err() {
        let result = DerPpLoss::new(0.5, -0.1);
        assert!(
            matches!(result, Err(ContinualError::InvalidLambda { .. })),
            "negative beta should return InvalidLambda, got {result:?}"
        );
    }

    // ── Test 11: zero memory (n_mem=0) succeeds and MSE/CE-mem = 0 ───────────

    #[test]
    fn zero_memory_succeeds() {
        let loss_fn = DerPpLoss::new(1.0, 1.0).expect("DerPpLoss::new should succeed");
        let n_cur = 2;
        let n_cls = 3;

        let result = loss_fn
            .compute(
                &uniform_logits(n_cur, n_cls),
                &uniform_labels(n_cur, n_cls),
                &[],
                &[],
                &[],
                n_cur,
                0,
                n_cls,
            )
            .expect("zero memory should succeed");

        assert!(
            result.is_finite(),
            "zero-memory loss should be finite, got {result}"
        );
        assert!(
            result >= 0.0,
            "zero-memory loss should be non-negative, got {result}"
        );
    }

    // ── Test 12: confident correct prediction gives lower CE than uniform ─────

    #[test]
    fn confident_correct_lower_ce_than_uniform() {
        let loss_fn = DerPpLoss::new(0.0, 0.0).expect("DerPpLoss::new should succeed");
        let n_cls = 4;

        // Confident prediction on class 0.
        let confident = vec![10.0_f32, 0.0, 0.0, 0.0];
        let loss_confident = loss_fn
            .compute(&confident, &[0], &[], &[], &[], 1, 0, n_cls)
            .expect("confident compute");

        // Uniform prediction.
        let uniform = vec![0.0_f32; n_cls];
        let loss_uniform = loss_fn
            .compute(&uniform, &[0], &[], &[], &[], 1, 0, n_cls)
            .expect("uniform compute");

        assert!(
            loss_confident < loss_uniform,
            "confident correct prediction must have lower CE: {loss_confident} vs {loss_uniform}"
        );
    }
}
