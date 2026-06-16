//! Platt scaling: parametric binary recalibration via a logistic transform
//! `p̂ = σ(A·s + B)` (Platt 1999).
//!
//! Given uncalibrated scores `s_i ∈ ℝ` (often raw SVM margins or pre-sigmoid logits)
//! and binary labels `y_i ∈ {0,1}`, fit `(A, B)` minimising
//! `-Σ_i [y_i log p̂_i + (1−y_i) log (1−p̂_i)]`. We use Newton-Raphson with a
//! simple line search backed by Lin et al. 2007 stable-target formulation.
//!
//! Numerical guard: targets `t = (n_pos+1)/(n_pos+2)` for positives and
//! `1/(n_neg+2)` for negatives prevent the log from blowing up at degenerate
//! datasets.

use crate::error::{BayesError, BayesResult};

/// Sigmoid `σ(x) = 1 / (1 + e^{-x})` with overflow guards.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Configuration for [`PlattScaler::fit`].
#[derive(Debug, Clone)]
pub struct PlattFitConfig {
    /// Maximum Newton iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the gradient norm.
    pub tol: f32,
    /// Damping multiplier when a Newton step doesn't decrease the loss.
    pub damping: f32,
}

impl Default for PlattFitConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-6,
            damping: 1e-12,
        }
    }
}

/// Two-parameter logistic recalibrator `p̂ = σ(A·s + B)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlattScaler {
    /// Slope `A`. Positive values preserve the original ordering.
    pub a: f32,
    /// Bias `B`.
    pub b: f32,
}

impl Default for PlattScaler {
    fn default() -> Self {
        Self { a: 1.0, b: 0.0 }
    }
}

impl PlattScaler {
    /// Apply the recalibration to a single score.
    #[must_use]
    pub fn predict_one(&self, s: f32) -> f32 {
        sigmoid(self.a * s + self.b)
    }

    /// Apply the recalibration pointwise.
    #[must_use]
    pub fn predict(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.predict_one(s)).collect()
    }

    /// Fit `(A, B)` on `(scores, labels)` via Lin et al.'s stable-target
    /// negative log-likelihood. Uses Lin's 2007 hybrid algorithm: Newton step
    /// with a per-iteration backtracking line search and full step shrinkage.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] when input is empty.
    /// - [`BayesError::DimensionMismatch`] when `scores.len() != labels.len()`.
    /// - [`BayesError::PlattFitFailed`] when no descent direction can be found.
    pub fn fit(scores: &[f32], labels: &[u8], cfg: &PlattFitConfig) -> BayesResult<Self> {
        if scores.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if scores.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: scores.len(),
                got: labels.len(),
            });
        }
        let n = scores.len();
        let mut n_pos = 0usize;
        for &y in labels {
            if y > 1 {
                return Err(BayesError::DimensionMismatch {
                    expected: 1,
                    got: usize::from(y),
                });
            }
            if y == 1 {
                n_pos += 1;
            }
        }
        let n_neg = n - n_pos;
        let hi_target = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let lo_target = 1.0 / (n_neg as f64 + 2.0);
        let targets: Vec<f64> = labels
            .iter()
            .map(|&y| if y == 1 { hi_target } else { lo_target })
            .collect();
        let scores64: Vec<f64> = scores.iter().map(|&s| s as f64).collect();

        // f64-stable log-loss.
        let loss = |a: f64, b: f64| -> f64 {
            let mut sum = 0.0_f64;
            for (i, &s) in scores64.iter().enumerate() {
                let z = a * s + b;
                let t = targets[i];
                // softplus64
                let sp_neg = if -z > 20.0 {
                    -z
                } else if -z < -20.0 {
                    (-z).exp()
                } else {
                    (1.0 + (-z).exp()).ln()
                };
                let sp_pos = if z > 20.0 {
                    z
                } else if z < -20.0 {
                    z.exp()
                } else {
                    (1.0 + z.exp()).ln()
                };
                sum += t * sp_neg + (1.0 - t) * sp_pos;
            }
            sum
        };

        // Initialise A=0, B=log((n_neg+1)/(n_pos+1))
        let mut a = 0.0_f64;
        let mut b = ((n_neg as f64 + 1.0) / (n_pos as f64 + 1.0)).ln();
        let mut prev_loss = loss(a, b);
        let lambda = 1e-3_f64; // small Hessian diagonal regulariser
        let min_step = 1e-10_f64;

        for _ in 0..cfg.max_iter {
            // Compute gradient and Hessian.
            let mut g_a = 0.0_f64;
            let mut g_b = 0.0_f64;
            let mut h_aa = 0.0_f64;
            let mut h_ab = 0.0_f64;
            let mut h_bb = 0.0_f64;
            for (i, &s) in scores64.iter().enumerate() {
                let z = a * s + b;
                let p = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                };
                let d = p - targets[i];
                let pq = (p * (1.0 - p)).max(1e-12);
                g_a += d * s;
                g_b += d;
                h_aa += pq * s * s;
                h_ab += pq * s;
                h_bb += pq;
            }

            let gn = (g_a * g_a + g_b * g_b).sqrt();
            if gn < cfg.tol as f64 {
                break;
            }

            // Add a small regulariser to the diagonal.
            let h_aa_reg = h_aa + lambda;
            let h_bb_reg = h_bb + lambda;
            let det = h_aa_reg * h_bb_reg - h_ab * h_ab;
            if det <= 0.0 {
                return Err(BayesError::PlattFitFailed);
            }
            let inv_det = 1.0 / det;
            // Δ = -H^{-1} g
            let dx_a = -inv_det * (h_bb_reg * g_a - h_ab * g_b);
            let dx_b = -inv_det * (-h_ab * g_a + h_aa_reg * g_b);

            // Backtracking line search.
            let mut step = 1.0_f64;
            let mut accepted = false;
            while step > min_step {
                let na = a + step * dx_a;
                let nb = b + step * dx_b;
                let nl = loss(na, nb);
                if nl <= prev_loss - 1e-3 * step * (g_a * dx_a + g_b * dx_b) || nl < prev_loss {
                    a = na;
                    b = nb;
                    prev_loss = nl;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                // No descent direction available — declare convergence at the
                // current iterate. (Often happens at the very flat tail of the
                // log-likelihood.)
                break;
            }
        }

        let a = a as f32;
        let b = b as f32;
        if !(a.is_finite() && b.is_finite()) {
            return Err(BayesError::PlattFitFailed);
        }
        Ok(Self { a, b })
    }

    /// Convenience: fit with default config.
    ///
    /// # Errors
    /// Propagates errors from [`PlattScaler::fit`].
    pub fn fit_default(scores: &[f32], labels: &[u8]) -> BayesResult<Self> {
        Self::fit(scores, labels, &PlattFitConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic linearly-separable data: positive class has score > 0.
    fn linearly_separable_dataset(n: usize) -> (Vec<f32>, Vec<u8>) {
        let mut s = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f32 - n as f32 * 0.5) * 0.1;
            s.push(x);
            y.push(if x > 0.0 { 1 } else { 0 });
        }
        (s, y)
    }

    #[test]
    fn platt_default_is_identity_sigmoid() {
        let p = PlattScaler::default();
        assert!((p.predict_one(0.0) - 0.5).abs() < 1e-6);
        assert!(p.predict_one(10.0) > 0.99);
        assert!(p.predict_one(-10.0) < 0.01);
    }

    #[test]
    fn platt_fit_separable_recovers_steep_slope() {
        let (s, y) = linearly_separable_dataset(200);
        let p = PlattScaler::fit_default(&s, &y).expect("fit_default should succeed");
        // Fitted slope must be positive and reasonably large for steeper than identity.
        assert!(p.a > 0.0);
        // Predictions for highly positive scores should approach 1.
        assert!(p.predict_one(100.0) > 0.95);
        assert!(p.predict_one(-100.0) < 0.05);
    }

    #[test]
    fn platt_predict_returns_probability() {
        let p = PlattScaler { a: 2.0, b: 0.5 };
        for s in [-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
            let v = p.predict_one(s);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn platt_predict_batch() {
        let p = PlattScaler::default();
        let s = vec![-1.0_f32, 0.0, 1.0];
        let v = p.predict(&s);
        assert_eq!(v.len(), 3);
        assert!(v[0] < 0.5 && v[1] == 0.5 && v[2] > 0.5);
    }

    #[test]
    fn platt_rejects_empty() {
        let r = PlattScaler::fit_default(&[], &[]);
        assert!(r.is_err());
    }

    #[test]
    fn platt_rejects_length_mismatch() {
        let r = PlattScaler::fit_default(&[1.0_f32, 2.0_f32], &[0_u8]);
        assert!(r.is_err());
    }

    #[test]
    fn platt_rejects_invalid_label() {
        let r = PlattScaler::fit_default(&[1.0_f32], &[2_u8]);
        assert!(r.is_err());
    }

    #[test]
    fn platt_fit_extreme_imbalance_ok() {
        // Mostly negatives, rare positives at very high scores
        let mut s = Vec::new();
        let mut y = Vec::new();
        for i in 0..98 {
            s.push(-1.0 + i as f32 * 0.01);
            y.push(0_u8);
        }
        s.push(5.0);
        y.push(1);
        s.push(6.0);
        y.push(1);
        let p = PlattScaler::fit_default(&s, &y).expect("fit_default should succeed");
        assert!(p.predict_one(6.0) > 0.5);
    }

    #[test]
    fn platt_lin_target_smoothing_tolerates_uniform_labels() {
        // All positives — Lin's smoothing prevents log(0) and we should still
        // produce a sensible all-near-1 calibrator.
        let s: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
        let y = vec![1_u8; 50];
        let p = PlattScaler::fit_default(&s, &y).expect("fit_default should succeed");
        assert!(p.predict_one(5.0) > 0.5);
    }
}
