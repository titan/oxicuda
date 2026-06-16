//! Adversarial transferability matrix.
//!
//! Transferability measures how well adversarial examples crafted against one
//! ("source") model fool a *different* ("target") model. Given `N` models and,
//! for each source model, a batch of adversarial examples crafted to fool it,
//! the **transfer matrix** `T ∈ [0, 1]^{N×N}` records
//!
//! ```text
//! T[i, j] = attack-success-rate of (examples crafted on source i) evaluated
//!           on target model j
//!         = fraction of source-i examples that target j mis-classifies.
//! ```
//!
//! The diagonal `T[i, i]` is the *white-box* success rate (the attack against
//! the very model it was crafted on); off-diagonal entries quantify
//! *black-box transfer*. White-box success is typically the highest in each
//! row, and strongly transferable attacks (or near-identical models) push the
//! off-diagonal up towards the diagonal.
//!
//! Two builders are provided:
//!
//! * [`transferability_matrix_from_predictions`] — when target predictions on
//!   every source's adversarial batch have already been computed.
//! * [`transferability_matrix`] — when models are given as prediction closures
//!   and adversarial example batches as flattened input vectors.
//!
//! Both reuse [`crate::metrics::asr::attack_success_rate`] for the per-cell
//! rate.

use crate::error::{AdvError, AdvResult};
use crate::metrics::asr::attack_success_rate;

/// A model's prediction closure: maps a flattened input to its predicted class.
///
/// Used by [`transferability_matrix`] so heterogeneous models can be passed as
/// `&[&ModelPredict]` (trait objects of differing concrete closure types).
pub type ModelPredict<'a> = dyn Fn(&[f32]) -> AdvResult<usize> + 'a;

/// Square `N×N` adversarial transferability matrix (row-major).
///
/// `rates[i * n_models + j]` is `T[i, j]` — the success rate of source-`i`
/// adversarial examples on target model `j`.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferMatrix {
    /// Row-major `N×N` success rates, each in `[0, 1]`.
    pub rates: Vec<f32>,
    /// Side length `N` (number of models).
    pub n_models: usize,
}

impl TransferMatrix {
    /// Success rate of source-`src` examples on target model `tgt`.
    ///
    /// # Errors
    /// * [`AdvError::DimensionMismatch`] — `src` or `tgt` out of range.
    pub fn get(&self, src: usize, tgt: usize) -> AdvResult<f32> {
        if src >= self.n_models || tgt >= self.n_models {
            return Err(AdvError::DimensionMismatch {
                expected: self.n_models,
                got: src.max(tgt) + 1,
            });
        }
        Ok(self.rates[src * self.n_models + tgt])
    }

    /// The white-box diagonal `[T[0,0], …, T[N-1,N-1]]`.
    #[must_use]
    pub fn diagonal(&self) -> Vec<f32> {
        (0..self.n_models)
            .map(|i| self.rates[i * self.n_models + i])
            .collect()
    }

    /// Mean of the diagonal (white-box) success rates. `0` for an empty matrix.
    #[must_use]
    pub fn mean_diagonal(&self) -> f32 {
        if self.n_models == 0 {
            return 0.0;
        }
        let sum: f32 = self.diagonal().iter().sum();
        sum / self.n_models as f32
    }

    /// Mean of the off-diagonal (black-box transfer) success rates. `0` when
    /// there are fewer than two models (no off-diagonal entries).
    #[must_use]
    pub fn mean_off_diagonal(&self) -> f32 {
        if self.n_models < 2 {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        for i in 0..self.n_models {
            for j in 0..self.n_models {
                if i != j {
                    sum += self.rates[i * self.n_models + j];
                }
            }
        }
        let count = self.n_models * (self.n_models - 1);
        sum / count as f32
    }
}

/// Build a transferability matrix from pre-computed target predictions.
///
/// * `preds[src][tgt]` — predicted classes of **target** model `tgt` evaluated
///   on the adversarial batch crafted against **source** model `src`. Its
///   length is the number of examples crafted on source `src`.
/// * `labels[src]` — ground-truth labels for source-`src`'s adversarial batch.
///
/// `T[src, tgt]` is the fraction of source-`src` examples that target `tgt`
/// mis-classifies (i.e. the attack success rate on that target).
///
/// # Errors
/// * [`AdvError::EmptyInput`]         — no models, or a source has no examples.
/// * [`AdvError::DimensionMismatch`]  — `preds`/`labels` ragged, a source's row
///   does not have `N` target entries, or a prediction vector length differs
///   from its source's label count.
pub fn transferability_matrix_from_predictions(
    preds: &[Vec<Vec<usize>>],
    labels: &[Vec<usize>],
) -> AdvResult<TransferMatrix> {
    let n = preds.len();
    if n == 0 {
        return Err(AdvError::EmptyInput);
    }
    if labels.len() != n {
        return Err(AdvError::DimensionMismatch {
            expected: n,
            got: labels.len(),
        });
    }

    let mut rates = vec![0.0_f32; n * n];
    for (src, src_row) in preds.iter().enumerate() {
        if src_row.len() != n {
            return Err(AdvError::DimensionMismatch {
                expected: n,
                got: src_row.len(),
            });
        }
        let src_labels = &labels[src];
        if src_labels.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        for (tgt, tgt_pred) in src_row.iter().enumerate() {
            // `attack_success_rate` validates equal-length & non-empty inputs.
            rates[src * n + tgt] = attack_success_rate(tgt_pred, src_labels)?;
        }
    }
    Ok(TransferMatrix { rates, n_models: n })
}

/// Build a transferability matrix from model prediction closures and per-source
/// adversarial example batches.
///
/// * `models[j]` — prediction closure of model `j`: maps a flattened input to
///   its predicted class.
/// * `adv_examples[i]` — the adversarial batch crafted against source model
///   `i`; each element is one flattened input vector.
/// * `labels[i]` — ground-truth labels for source-`i`'s batch.
///
/// Each model in `models` is evaluated on every source's batch; the resulting
/// predictions are reduced through
/// [`transferability_matrix_from_predictions`].
///
/// # Errors
/// * [`AdvError::EmptyInput`]         — no models, or a source has no examples.
/// * [`AdvError::DimensionMismatch`]  — `models`, `adv_examples` and `labels`
///   do not all have the same length `N`, or a batch's example count differs
///   from its label count.
/// * Any error returned by a model closure.
pub fn transferability_matrix(
    models: &[&ModelPredict<'_>],
    adv_examples: &[Vec<Vec<f32>>],
    labels: &[Vec<usize>],
) -> AdvResult<TransferMatrix> {
    let n = models.len();
    if n == 0 {
        return Err(AdvError::EmptyInput);
    }
    if adv_examples.len() != n {
        return Err(AdvError::DimensionMismatch {
            expected: n,
            got: adv_examples.len(),
        });
    }
    if labels.len() != n {
        return Err(AdvError::DimensionMismatch {
            expected: n,
            got: labels.len(),
        });
    }

    // Materialise predictions: preds[src][tgt] = model[tgt] over batch[src].
    let mut preds: Vec<Vec<Vec<usize>>> = Vec::with_capacity(n);
    for (src, batch) in adv_examples.iter().enumerate() {
        if batch.is_empty() {
            return Err(AdvError::EmptyInput);
        }
        if batch.len() != labels[src].len() {
            return Err(AdvError::DimensionMismatch {
                expected: batch.len(),
                got: labels[src].len(),
            });
        }
        let mut src_row: Vec<Vec<usize>> = Vec::with_capacity(n);
        for model in models {
            let mut tgt_pred = Vec::with_capacity(batch.len());
            for example in batch {
                tgt_pred.push(model(example)?);
            }
            src_row.push(tgt_pred);
        }
        preds.push(src_row);
    }
    transferability_matrix_from_predictions(&preds, labels)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn all_in_unit(m: &TransferMatrix) -> bool {
        m.rates.iter().all(|&r| (0.0..=1.0).contains(&r))
    }

    // ── from_predictions: manual rates ──────────────────────────────────────

    #[test]
    fn from_predictions_matches_manual_rates() {
        // 2 models, 2 examples per source.
        // Source 0 labels = [0, 1].
        //   target 0 preds = [1, 0]  → both wrong → ASR 1.0
        //   target 1 preds = [0, 1]  → both right → ASR 0.0
        // Source 1 labels = [1, 1].
        //   target 0 preds = [1, 0]  → one wrong → ASR 0.5
        //   target 1 preds = [0, 0]  → both wrong → ASR 1.0
        let preds = vec![
            vec![vec![1_usize, 0], vec![0, 1]],
            vec![vec![1_usize, 0], vec![0, 0]],
        ];
        let labels = vec![vec![0_usize, 1], vec![1_usize, 1]];
        let m = transferability_matrix_from_predictions(&preds, &labels)
            .expect("transferability_matrix_from_predictions should succeed");
        assert_eq!(m.n_models, 2);
        assert!((m.get(0, 0).expect("get should succeed") - 1.0).abs() < 1e-6);
        assert!(m.get(0, 1).expect("get should succeed").abs() < 1e-6);
        assert!((m.get(1, 0).expect("get should succeed") - 0.5).abs() < 1e-6);
        assert!((m.get(1, 1).expect("get should succeed") - 1.0).abs() < 1e-6);
    }

    // ── shape N×N ────────────────────────────────────────────────────────────

    #[test]
    fn matrix_is_square_n_by_n() {
        let preds = vec![
            vec![vec![0_usize], vec![0], vec![0]],
            vec![vec![0_usize], vec![0], vec![0]],
            vec![vec![0_usize], vec![0], vec![0]],
        ];
        let labels = vec![vec![1_usize], vec![1], vec![1]];
        let m = transferability_matrix_from_predictions(&preds, &labels)
            .expect("transferability_matrix_from_predictions should succeed");
        assert_eq!(m.n_models, 3);
        assert_eq!(m.rates.len(), 9);
    }

    // ── rates in [0, 1] & finite ────────────────────────────────────────────

    #[test]
    fn rates_in_unit_interval_and_finite() {
        let preds = vec![
            vec![vec![1_usize, 1, 0], vec![0, 1, 1]],
            vec![vec![1_usize, 0, 0], vec![1, 1, 1]],
        ];
        let labels = vec![vec![0_usize, 1, 0], vec![1_usize, 1, 0]];
        let m = transferability_matrix_from_predictions(&preds, &labels)
            .expect("transferability_matrix_from_predictions should succeed");
        assert!(all_in_unit(&m));
        assert!(m.rates.iter().all(|r| r.is_finite()));
    }

    // ── closure builder: diagonal ≥ off-diagonal ────────────────────────────

    #[test]
    fn closure_builder_diagonal_dominates() {
        // model k predicts class 1 iff x[0] >= k, else class 0.
        // Source k's adversarial batch sits in [k, k+1): it flips model k (and
        // every lower-threshold model) to class 1 while the true label is 0,
        // but leaves higher-threshold models predicting 0 (correct).
        let m0 = |x: &[f32]| -> AdvResult<usize> { Ok((x[0] >= 0.0) as usize) };
        let m1 = |x: &[f32]| -> AdvResult<usize> { Ok((x[0] >= 1.0) as usize) };
        let m2 = |x: &[f32]| -> AdvResult<usize> { Ok((x[0] >= 2.0) as usize) };
        let models: [&ModelPredict<'_>; 3] = [&m0, &m1, &m2];

        let adv = vec![
            vec![vec![0.4_f32], vec![0.6]], // source 0 → in [0,1)
            vec![vec![1.4_f32], vec![1.6]], // source 1 → in [1,2)
            vec![vec![2.4_f32], vec![2.6]], // source 2 → in [2,3)
        ];
        let labels = vec![vec![0_usize, 0], vec![0, 0], vec![0, 0]];
        let m = transferability_matrix(&models, &adv, &labels)
            .expect("transferability_matrix should succeed");

        // Diagonal is full white-box success.
        for d in m.diagonal() {
            assert!((d - 1.0).abs() < 1e-6);
        }
        // Each diagonal entry ≥ every off-diagonal entry in its row.
        for i in 0..m.n_models {
            let diag = m.get(i, i).expect("get should succeed");
            for j in 0..m.n_models {
                assert!(diag + 1e-6 >= m.get(i, j).expect("get should succeed"));
            }
        }
        assert!(m.mean_diagonal() >= m.mean_off_diagonal());
    }

    // ── identical models ⇒ symmetric high transfer ──────────────────────────

    #[test]
    fn identical_models_symmetric_high_transfer() {
        let model = |x: &[f32]| -> AdvResult<usize> { Ok((x[0] >= 0.5) as usize) };
        let models: [&ModelPredict<'_>; 3] = [&model, &model, &model];
        // All examples flip the (shared) model: x[0] >= 0.5 predicts 1, label 0.
        let adv = vec![
            vec![vec![0.9_f32], vec![0.8]],
            vec![vec![0.7_f32], vec![0.95]],
            vec![vec![0.6_f32], vec![0.99]],
        ];
        let labels = vec![vec![0_usize, 0], vec![0, 0], vec![0, 0]];
        let m = transferability_matrix(&models, &adv, &labels)
            .expect("transferability_matrix should succeed");
        // Every cell is 1.0 ⇒ symmetric and maximal transfer.
        for i in 0..m.n_models {
            for j in 0..m.n_models {
                assert!((m.get(i, j).expect("get should succeed") - 1.0).abs() < 1e-6);
                assert!(
                    (m.get(i, j).expect("get should succeed")
                        - m.get(j, i).expect("get should succeed"))
                    .abs()
                        < 1e-6
                );
            }
        }
        assert!((m.mean_off_diagonal() - 1.0).abs() < 1e-6);
    }

    // ── error handling ──────────────────────────────────────────────────────

    #[test]
    fn empty_models_rejected() {
        let preds: Vec<Vec<Vec<usize>>> = vec![];
        let labels: Vec<Vec<usize>> = vec![];
        assert!(matches!(
            transferability_matrix_from_predictions(&preds, &labels).unwrap_err(),
            AdvError::EmptyInput
        ));
    }

    #[test]
    fn ragged_rows_rejected() {
        // Source 0 row has only 1 target entry but N = 2.
        let preds = vec![vec![vec![0_usize]], vec![vec![0_usize], vec![0]]];
        let labels = vec![vec![1_usize], vec![1]];
        assert!(matches!(
            transferability_matrix_from_predictions(&preds, &labels).unwrap_err(),
            AdvError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn label_count_mismatch_rejected() {
        let preds = vec![vec![vec![0_usize, 1]]];
        // Prediction length 2 but only 1 label → attack_success_rate errors.
        let labels = vec![vec![1_usize]];
        assert!(transferability_matrix_from_predictions(&preds, &labels).is_err());
    }

    #[test]
    fn get_out_of_range_errors() {
        let preds = vec![vec![vec![0_usize]]];
        let labels = vec![vec![1_usize]];
        let m = transferability_matrix_from_predictions(&preds, &labels)
            .expect("transferability_matrix_from_predictions should succeed");
        assert!(m.get(1, 0).is_err());
        assert!(m.get(0, 1).is_err());
    }

    #[test]
    fn closure_builder_propagates_model_error() {
        let good = |x: &[f32]| -> AdvResult<usize> { Ok((x[0] >= 0.5) as usize) };
        let bad = |_x: &[f32]| -> AdvResult<usize> { Err(AdvError::AttackFailedAll) };
        let models: [&ModelPredict<'_>; 2] = [&good, &bad];
        let adv = vec![vec![vec![0.9_f32]], vec![vec![0.1_f32]]];
        let labels = vec![vec![0_usize], vec![0]];
        assert!(matches!(
            transferability_matrix(&models, &adv, &labels).unwrap_err(),
            AdvError::AttackFailedAll
        ));
    }
}
