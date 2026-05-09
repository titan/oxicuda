//! Isotonic regression (Pool Adjacent Violators algorithm) for non-parametric
//! monotone recalibration.
//!
//! Given (uncalibrated_score, target) pairs, fit a piecewise-constant non-decreasing
//! function `g: [0,1] → [0,1]` minimising `Σ_i w_i · (g(x_i) − y_i)²` subject to
//! `g` being non-decreasing in `x`. PAV runs in `O(n log n)` (sort) plus
//! `O(n)` (merging) and is the canonical post-hoc recalibrator for binary
//! probabilistic predictions (Zadrozny & Elkan 2001).

use crate::error::{BayesError, BayesResult};

/// Fitted piecewise-constant non-decreasing recalibrator.
///
/// The function consists of `K` blocks; block `k` covers `x ∈ [x[k], x[k+1])`
/// (with `x[K]` = +∞) and outputs the constant `y[k]`.
#[derive(Debug, Clone, PartialEq)]
pub struct IsotonicRegressor {
    /// Block lower endpoints in ascending order (length `K`). The first
    /// is `≤ min(input)` (clamped to 0.0 in [`Self::predict`]).
    pub thresholds: Vec<f32>,
    /// Output value of each block (length `K`). Non-decreasing.
    pub values: Vec<f32>,
}

impl IsotonicRegressor {
    /// Identity-like constant function `y = 0.5`. Useful as a fallback.
    #[must_use]
    pub fn constant(y: f32) -> Self {
        Self {
            thresholds: vec![0.0],
            values: vec![y],
        }
    }

    /// Fit a non-decreasing PAV regressor from raw scores and 0/1 targets.
    ///
    /// All weights are set to 1.0.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] when `scores.is_empty()`.
    /// - [`BayesError::DimensionMismatch`] when lengths disagree.
    /// - [`BayesError::IsotonicNotMonotone`] if the output is not weakly
    ///   monotone (defensive — should not happen with PAV).
    pub fn fit(scores: &[f32], targets: &[f32]) -> BayesResult<Self> {
        Self::fit_weighted(scores, targets, &vec![1.0_f32; scores.len()])
    }

    /// Fit with per-sample weights (must be positive).
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] when inputs are empty.
    /// - [`BayesError::DimensionMismatch`] when lengths disagree.
    /// - [`BayesError::NanEncountered`] when a non-positive or non-finite weight
    ///   is encountered.
    /// - [`BayesError::IsotonicNotMonotone`] if PAV fails to produce monotone output.
    pub fn fit_weighted(scores: &[f32], targets: &[f32], weights: &[f32]) -> BayesResult<Self> {
        if scores.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if scores.len() != targets.len() || scores.len() != weights.len() {
            return Err(BayesError::DimensionMismatch {
                expected: scores.len(),
                got: targets.len().min(weights.len()),
            });
        }
        for &w in weights {
            if !(w.is_finite() && w > 0.0) {
                return Err(BayesError::NanEncountered {
                    location: "isotonic.fit_weighted: non-positive weight",
                });
            }
        }

        // Sort by score ascending.
        let n = scores.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let xs: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();
        let ys: Vec<f32> = idx.iter().map(|&i| targets[i]).collect();
        let ws: Vec<f32> = idx.iter().map(|&i| weights[i]).collect();

        // PAV: maintain blocks (sum, weight, x_min)
        let mut block_x: Vec<f32> = Vec::with_capacity(n);
        let mut block_y: Vec<f32> = Vec::with_capacity(n); // mean
        let mut block_w: Vec<f32> = Vec::with_capacity(n);

        for i in 0..n {
            block_x.push(xs[i]);
            block_y.push(ys[i]);
            block_w.push(ws[i]);
            // Merge backwards while monotonicity is violated.
            while block_y.len() >= 2 {
                let last = block_y.len() - 1;
                if block_y[last - 1] <= block_y[last] {
                    break;
                }
                let w1 = block_w[last - 1];
                let w2 = block_w[last];
                let merged_w = w1 + w2;
                let merged_y = (w1 * block_y[last - 1] + w2 * block_y[last]) / merged_w;
                block_y[last - 1] = merged_y;
                block_w[last - 1] = merged_w;
                // Keep the lower x as block start.
                block_x[last - 1] = block_x[last - 1].min(block_x[last]);
                block_y.pop();
                block_w.pop();
                block_x.pop();
            }
        }

        // Defensive: verify weak monotonicity.
        for i in 1..block_y.len() {
            if block_y[i] + 1e-6 < block_y[i - 1] {
                return Err(BayesError::IsotonicNotMonotone);
            }
        }

        Ok(Self {
            thresholds: block_x,
            values: block_y,
        })
    }

    /// Apply the fitted function pointwise: returns the value of the block
    /// containing `x` (binary search in `thresholds`).
    #[must_use]
    pub fn predict_one(&self, x: f32) -> f32 {
        if x.is_nan() {
            return f32::NAN;
        }
        // Binary search for the largest threshold <= x; default to first block.
        let mut lo = 0usize;
        let mut hi = self.thresholds.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.thresholds[mid] <= x {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        let block = if lo == 0 { 0 } else { lo - 1 };
        self.values[block]
    }

    /// Apply pointwise to an entire vector.
    #[must_use]
    pub fn predict(&self, xs: &[f32]) -> Vec<f32> {
        xs.iter().map(|&x| self.predict_one(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pav_already_monotone_input_unchanged() {
        let xs = vec![0.1_f32, 0.2, 0.3, 0.4];
        let ys = vec![0.0_f32, 0.0, 1.0, 1.0];
        let r = IsotonicRegressor::fit(&xs, &ys).unwrap();
        // already monotone, two blocks expected (after merging the equal-mean groups)
        assert!(!r.values.is_empty());
        for w in r.values.windows(2) {
            assert!(w[0] <= w[1] + 1e-6);
        }
    }

    #[test]
    fn pav_reverse_inputs_collapses_to_mean() {
        let xs = vec![0.1_f32, 0.2, 0.3, 0.4];
        let ys = vec![1.0_f32, 1.0, 0.0, 0.0];
        let r = IsotonicRegressor::fit(&xs, &ys).unwrap();
        // PAV will pool everything into one block at mean = 0.5
        for &v in &r.values {
            assert!((v - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn pav_predict_within_blocks() {
        let xs = vec![0.1_f32, 0.5, 0.9];
        let ys = vec![0.0_f32, 0.5, 1.0];
        let r = IsotonicRegressor::fit(&xs, &ys).unwrap();
        // x below first threshold falls into first block
        assert!((r.predict_one(0.05) - r.values[0]).abs() < 1e-6);
        // x above last threshold uses last block
        let last_v = *r.values.last().unwrap();
        assert!((r.predict_one(0.99) - last_v).abs() < 1e-6);
    }

    #[test]
    fn pav_random_data_produces_monotone_output() {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..50 {
            let x = i as f32 / 50.0;
            // y has noise but increasing trend
            let y = if i < 25 { 0.0 } else { 1.0 };
            xs.push(x);
            ys.push(y);
        }
        let r = IsotonicRegressor::fit(&xs, &ys).unwrap();
        let probs = r.predict(&xs);
        for w in probs.windows(2) {
            assert!(w[0] <= w[1] + 1e-6);
        }
    }

    #[test]
    fn pav_weighted_fit_respects_weights() {
        let xs = vec![0.1_f32, 0.2];
        let ys = vec![0.0_f32, 1.0];
        let ws = vec![3.0_f32, 1.0];
        // Weighted mean = 0.25
        let r = IsotonicRegressor::fit_weighted(&xs, &ys, &ws).unwrap();
        // since ys are already monotone (0,1), no merging needed
        assert_eq!(r.values, vec![0.0, 1.0]);
        // try the reverse case
        let r2 = IsotonicRegressor::fit_weighted(&xs, &[1.0, 0.0], &ws).unwrap();
        assert_eq!(r2.values.len(), 1);
        assert!((r2.values[0] - 0.75).abs() < 1e-5); // (3*1 + 1*0)/4
    }

    #[test]
    fn pav_constant_constructor() {
        let r = IsotonicRegressor::constant(0.7);
        for x in [-1.0_f32, 0.0, 0.5, 1.0, 5.0] {
            assert!((r.predict_one(x) - 0.7).abs() < 1e-6);
        }
    }

    #[test]
    fn pav_rejects_empty() {
        let r = IsotonicRegressor::fit(&[], &[]);
        assert!(r.is_err());
    }

    #[test]
    fn pav_rejects_length_mismatch() {
        let r = IsotonicRegressor::fit(&[0.1_f32], &[0.0_f32, 1.0_f32]);
        assert!(r.is_err());
    }

    #[test]
    fn pav_rejects_non_positive_weight() {
        let xs = vec![0.1_f32];
        let ys = vec![0.0_f32];
        let ws = vec![0.0_f32];
        assert!(IsotonicRegressor::fit_weighted(&xs, &ys, &ws).is_err());
    }

    #[test]
    fn pav_predict_one_handles_nan() {
        let r = IsotonicRegressor::constant(0.5);
        assert!(r.predict_one(f32::NAN).is_nan());
    }
}
