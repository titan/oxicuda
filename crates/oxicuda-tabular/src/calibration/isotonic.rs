//! Isotonic-regression probability calibration via the Pool-Adjacent-Violators
//! Algorithm (PAVA).
//!
//! Given raw classifier scores and binary `{0, 1}` labels, PAVA fits the
//! non-decreasing step function `g` minimising the weighted squared error
//! `Σ_i (g(s_i) − y_i)²` subject to monotonicity in `s`. The resulting
//! `(score, calibrated_probability)` knots define a monotone calibration map;
//! [`IsotonicCalibrator::predict`] linearly interpolates between adjacent knots
//! and clamps queries outside the fitted score range (Zadrozny & Elkan 2002).

use crate::error::{TabularError, TabularResult};

/// Fitted isotonic probability calibrator.
///
/// The map is represented by ascending knot scores `x_thresholds` and their
/// associated calibrated probabilities `y_values` (non-decreasing, each in
/// `[0, 1]`). Predictions interpolate linearly between knots.
#[derive(Debug, Clone, PartialEq)]
pub struct IsotonicCalibrator {
    /// Ascending, deduplicated knot scores.
    x_thresholds: Vec<f32>,
    /// Calibrated probability at each knot (non-decreasing, clamped to `[0, 1]`).
    y_values: Vec<f32>,
}

impl IsotonicCalibrator {
    /// Fit the calibrator from `n` `(score, label)` pairs.
    ///
    /// `labels` should contain `{0, 1}` values (any finite value is accepted
    /// and pooled). PAVA runs in `O(n log n)` (the sort) plus `O(n)` pooling.
    ///
    /// # Errors
    /// Returns [`TabularError::EmptyInput`] if `n == 0`, and
    /// [`TabularError::DimensionMismatch`] if either slice is shorter than `n`.
    pub fn fit(scores: &[f32], labels: &[f32], n: usize) -> TabularResult<Self> {
        if n == 0 {
            return Err(TabularError::EmptyInput);
        }
        if scores.len() < n || labels.len() < n {
            return Err(TabularError::DimensionMismatch {
                expected: n,
                got: scores.len().min(labels.len()),
            });
        }

        // Sort indices by score ascending.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // PAVA over blocks: each block stores (sum_y, weight, x_min, x_max).
        let mut block_sum: Vec<f32> = Vec::with_capacity(n);
        let mut block_w: Vec<f32> = Vec::with_capacity(n);
        let mut block_xmin: Vec<f32> = Vec::with_capacity(n);
        let mut block_xmax: Vec<f32> = Vec::with_capacity(n);

        for &i in &idx {
            let x = scores[i];
            let y = labels[i];
            block_sum.push(y);
            block_w.push(1.0);
            block_xmin.push(x);
            block_xmax.push(x);
            // Merge while the previous block's mean exceeds the new block's
            // mean, or while the two blocks share the same score (tied inputs
            // must be pooled so the fitted map is single-valued there).
            while block_sum.len() >= 2 {
                let k = block_sum.len() - 1;
                let mean_prev = block_sum[k - 1] / block_w[k - 1];
                let mean_cur = block_sum[k] / block_w[k];
                let tied = block_xmin[k] <= block_xmax[k - 1];
                if mean_prev <= mean_cur && !tied {
                    break;
                }
                // Pool block k into block k-1.
                block_sum[k - 1] += block_sum[k];
                block_w[k - 1] += block_w[k];
                block_xmax[k - 1] = block_xmax[k];
                block_sum.pop();
                block_w.pop();
                block_xmin.pop();
                block_xmax.pop();
            }
        }

        // Build knots: place the pooled mean at both ends of each block's score
        // span so that interpolation reproduces the step plateau across ties.
        let mut x_thresholds: Vec<f32> = Vec::with_capacity(block_sum.len() * 2);
        let mut y_values: Vec<f32> = Vec::with_capacity(block_sum.len() * 2);
        for b in 0..block_sum.len() {
            let mean = (block_sum[b] / block_w[b]).clamp(0.0, 1.0);
            // Left endpoint of the block.
            push_knot(&mut x_thresholds, &mut y_values, block_xmin[b], mean);
            // Right endpoint (only if distinct from the left).
            if block_xmax[b] > block_xmin[b] {
                push_knot(&mut x_thresholds, &mut y_values, block_xmax[b], mean);
            }
        }

        // Enforce non-decreasing y across knots (defensive against ties at
        // identical scores that may have been deduplicated).
        for i in 1..y_values.len() {
            if y_values[i] < y_values[i - 1] {
                y_values[i] = y_values[i - 1];
            }
        }

        Ok(Self {
            x_thresholds,
            y_values,
        })
    }

    /// Map a single raw score to a calibrated probability in `[0, 1]`.
    ///
    /// Scores below the smallest knot return the first calibrated value, scores
    /// above the largest knot return the last; interior scores are linearly
    /// interpolated between the two bracketing knots.
    #[must_use]
    pub fn predict(&self, score: f32) -> f32 {
        let xs = &self.x_thresholds;
        let ys = &self.y_values;
        if xs.is_empty() {
            return 0.5;
        }
        // Clamp below / above the fitted range.
        if score <= xs[0] {
            return ys[0];
        }
        let last = xs.len() - 1;
        if score >= xs[last] {
            return ys[last];
        }
        // Binary search for the bracketing interval [xs[lo], xs[lo + 1]).
        let mut lo = 0usize;
        let mut hi = last;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if xs[mid] <= score {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let x0 = xs[lo];
        let x1 = xs[lo + 1];
        let y0 = ys[lo];
        let y1 = ys[lo + 1];
        let span = x1 - x0;
        if span <= 0.0 {
            return y0;
        }
        let t = (score - x0) / span;
        (y0 + t * (y1 - y0)).clamp(0.0, 1.0)
    }

    /// Calibrate a batch of scores.
    #[must_use]
    pub fn predict_batch(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.predict(s)).collect()
    }

    /// Number of fitted knots.
    #[must_use]
    pub fn n_knots(&self) -> usize {
        self.x_thresholds.len()
    }
}

/// Append a `(x, y)` knot, coalescing with the previous knot when the score is
/// identical (keeping the most recent — and hence non-decreasing — `y`).
fn push_knot(xs: &mut Vec<f32>, ys: &mut Vec<f32>, x: f32, y: f32) {
    if let (Some(&last_x), Some(last_y)) = (xs.last(), ys.last_mut())
        && (x - last_x).abs() <= f32::EPSILON
    {
        *last_y = y;
        return;
    }
    xs.push(x);
    ys.push(y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotone_output() {
        // A calibrator fit on increasing labels-with-score must be monotone.
        let scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let labels = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 8).expect("fit should succeed");
        let mut prev = f32::NEG_INFINITY;
        let mut q = -0.5_f32;
        while q <= 1.5 {
            let p = cal.predict(q);
            assert!(p >= prev - 1e-6, "not monotone at {q}: {p} < {prev}");
            prev = p;
            q += 0.05;
        }
    }

    #[test]
    fn predict_in_0_1() {
        let scores = [0.0, 0.3, 0.6, 0.9, 1.0];
        let labels = [0.0, 0.0, 1.0, 1.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 5).expect("fit should succeed");
        for &q in &[-10.0, -1.0, 0.0, 0.5, 1.0, 10.0] {
            let p = cal.predict(q);
            assert!((0.0..=1.0).contains(&p), "p={p} for q={q}");
        }
    }

    #[test]
    fn fit_finite() {
        let scores = [0.2, 0.5, 0.1, 0.9, 0.4, 0.7];
        let labels = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 6).expect("fit should succeed");
        assert!(cal.y_values.iter().all(|v| v.is_finite()));
        assert!(cal.x_thresholds.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn perfectly_calibrated_identity() {
        // Scores that already equal the empirical label frequency should be
        // reproduced closely. Two scores, each with a 50/50 label mix → 0.5.
        let scores = [0.5, 0.5, 0.5, 0.5];
        let labels = [0.0, 1.0, 0.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 4).expect("fit should succeed");
        assert!((cal.predict(0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn predict_batch_shape() {
        let scores = [0.1, 0.4, 0.7, 0.9];
        let labels = [0.0, 0.0, 1.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 4).expect("fit should succeed");
        let out = cal.predict_batch(&[0.0, 0.2, 0.8, 1.0]);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn extrapolation_clamps() {
        let scores = [0.2, 0.4, 0.6, 0.8];
        let labels = [0.0, 0.0, 1.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 4).expect("fit should succeed");
        // Below the smallest score → first y; above the largest → last y.
        let below = cal.predict(-5.0);
        let above = cal.predict(5.0);
        assert!((below - cal.y_values[0]).abs() < 1e-6);
        assert!((above - cal.y_values[cal.y_values.len() - 1]).abs() < 1e-6);
    }

    #[test]
    fn empty_input_error() {
        let res = IsotonicCalibrator::fit(&[], &[], 0);
        assert!(matches!(res, Err(TabularError::EmptyInput)));
    }

    #[test]
    fn dimension_mismatch_error() {
        let res = IsotonicCalibrator::fit(&[0.1, 0.2], &[1.0], 2);
        assert!(matches!(res, Err(TabularError::DimensionMismatch { .. })));
    }

    #[test]
    fn single_point() {
        let cal = IsotonicCalibrator::fit(&[0.7], &[1.0], 1).expect("fit should succeed");
        // A single knot → constant map at the fitted value.
        assert!((cal.predict(0.0) - 1.0).abs() < 1e-6);
        assert!((cal.predict(0.7) - 1.0).abs() < 1e-6);
        assert!((cal.predict(1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn interpolation_works() {
        // Two distinct plateaus at y=0 (score 0.2) and y=1 (score 0.8); a query
        // midway should interpolate strictly between them.
        let scores = [0.2, 0.8];
        let labels = [0.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels, 2).expect("fit should succeed");
        let mid = cal.predict(0.5);
        assert!(mid > 0.0 && mid < 1.0, "mid={mid} not interpolated");
        assert!((mid - 0.5).abs() < 0.2, "mid={mid} not near 0.5");
    }

    #[test]
    fn violators_pooled() {
        // Out-of-order labels must be pooled into a monotone fit.
        let scores = [0.1, 0.2, 0.3, 0.4];
        let labels = [1.0, 0.0, 1.0, 0.0]; // non-monotone
        let cal = IsotonicCalibrator::fit(&scores, &labels, 4).expect("fit should succeed");
        // The pooled mean should be 0.5 throughout (average of all labels).
        for &q in &[0.1, 0.25, 0.4] {
            assert!((cal.predict(q) - 0.5).abs() < 1e-6, "q={q}");
        }
    }
}
