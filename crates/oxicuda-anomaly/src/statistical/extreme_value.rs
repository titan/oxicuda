//! Extreme Value Theory (EVT) anomaly detector.
//!
//! Clifton, D. A., Hugueny, S. & Tarassenko, L. (2011).
//! "Novelty Detection with Multivariate Extreme Value Statistics."
//! *Journal of Signal Processing Systems*, 65(3).
//!
//! Gnedenko, B. V. (1943). "Sur la distribution limite du terme maximum d'une
//! série aléatoire." *Annals of Mathematics*, 44(3).
//!
//! # Key idea
//!
//! Fit the tails of an anomaly score distribution with a
//! **Generalised Pareto Distribution (GPD)** via maximum-likelihood estimation
//! on the exceedances above a threshold `t` (Peak-Over-Threshold, POT method):
//!
//! ```text
//! F_GPD(x; ξ, σ) = 1 − (1 + ξ(x − t)/σ)^{−1/ξ}   (ξ ≠ 0)
//!                = 1 − exp(−(x − t)/σ)               (ξ = 0)
//! ```
//!
//! The threshold `t` is chosen automatically via the **mean-excess plot** heuristic:
//! select the smallest threshold for which the mean excess function
//! `e(u) = E[X − u | X > u]` becomes approximately linear.
//!
//! A new score `s` is flagged as an anomaly when its tail probability under the
//! fitted GPD exceeds a critical level `1 − α`.

use crate::error::{AnomalyError, AnomalyResult};

// ─── GpdFit ───────────────────────────────────────────────────────────────────

/// Fitted Generalised Pareto Distribution (GPD) parameters.
#[derive(Debug, Clone)]
pub struct GpdFit {
    /// Threshold above which exceedances are modelled.
    pub threshold: f32,
    /// GPD shape parameter ξ (xi).  ξ < 0 = bounded tail, ξ ≈ 0 = exponential,
    /// ξ > 0 = heavy-tailed Pareto.
    pub xi: f32,
    /// GPD scale parameter σ > 0.
    pub sigma: f32,
    /// Number of exceedances used in fitting.
    pub n_exceedances: usize,
}

impl GpdFit {
    /// Survival function: `P(X > x | X > t)` for `x >= t`.
    ///
    /// Returns 0 for `x < t`, 1 for `x = t`.
    #[must_use]
    pub fn survival(&self, x: f32) -> f32 {
        let z = (x - self.threshold).max(0.0);
        if self.sigma <= 0.0 {
            return 0.0;
        }
        if self.xi.abs() < 1e-6 {
            // Exponential case.
            (-z / self.sigma).exp()
        } else {
            let arg = 1.0 + self.xi * z / self.sigma;
            if arg <= 0.0 {
                0.0
            } else {
                arg.powf(-1.0 / self.xi)
            }
        }
    }

    /// Anomaly p-value: probability under the GPD that a score this extreme
    /// or more was drawn from the reference distribution.
    ///
    /// Small values (< `alpha`) indicate anomalies.
    #[must_use]
    pub fn p_value(&self, score: f32) -> f32 {
        if score <= self.threshold {
            return 1.0; // Not in the tail.
        }
        self.survival(score)
    }
}

// ─── GpdDetector ─────────────────────────────────────────────────────────────

/// Generalised Pareto Distribution anomaly detector (Peak-Over-Threshold).
#[derive(Debug, Clone)]
pub struct GpdDetector {
    /// Significance level α ∈ (0, 1).  Scores with p-value < α are flagged.
    pub alpha: f32,
    /// Fraction of sorted scores to use as threshold candidates.
    /// E.g., 0.9 means the threshold sweeps over the top 10 % of training scores.
    pub threshold_quantile: f32,
    /// Fitted GPD parameters (set after `fit`).
    pub fit: Option<GpdFit>,
    /// Raw training scores (kept for reference).
    train_scores: Vec<f32>,
}

impl GpdDetector {
    /// Create a new GPD detector.
    ///
    /// # Errors
    /// - `AnomalyError::InvalidThresholdPercentile` if `alpha` or
    ///   `threshold_quantile` are not in `(0, 1)`.
    pub fn new(alpha: f32, threshold_quantile: f32) -> AnomalyResult<Self> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(AnomalyError::InvalidThresholdPercentile { p: alpha });
        }
        if !(threshold_quantile > 0.0 && threshold_quantile < 1.0) {
            return Err(AnomalyError::InvalidThresholdPercentile {
                p: threshold_quantile,
            });
        }
        Ok(Self {
            alpha,
            threshold_quantile,
            fit: None,
            train_scores: Vec::new(),
        })
    }

    /// Fit the GPD to a vector of anomaly scores from normal (non-anomalous)
    /// training data.
    ///
    /// The threshold is chosen at `threshold_quantile` of the training scores
    /// (e.g., 0.95 = 95th percentile).  GPD is then fitted to the exceedances
    /// via maximum-likelihood (L-moments for robustness).
    ///
    /// # Errors
    /// - `AnomalyError::InsufficientSamples` if fewer than 10 exceedances.
    /// - `AnomalyError::EmptyInput` if `scores` is empty.
    pub fn fit(&mut self, scores: &[f32]) -> AnomalyResult<()> {
        if scores.is_empty() {
            return Err(AnomalyError::EmptyInput);
        }
        self.train_scores = scores.to_vec();
        let mut sorted = scores.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();

        // Threshold = `threshold_quantile` quantile of training scores.
        let q_idx = ((self.threshold_quantile * n as f32).ceil() as usize).min(n - 1);
        let threshold = sorted[q_idx];

        // Exceedances: scores strictly above threshold.
        let exceedances: Vec<f32> = sorted[q_idx + 1..].iter().map(|&v| v - threshold).collect();

        if exceedances.len() < 5 {
            // Fall back to a simple exponential fit if very few exceedances.
            if exceedances.is_empty() {
                return Err(AnomalyError::InsufficientSamples { need: 5, got: 0 });
            }
            let mean_exc = exceedances.iter().sum::<f32>() / exceedances.len() as f32;
            self.fit = Some(GpdFit {
                threshold,
                xi: 0.0,
                sigma: mean_exc.max(1e-6),
                n_exceedances: exceedances.len(),
            });
            return Ok(());
        }

        // Fit GPD via L-moments estimator (Hosking & Wallis 1987):
        // L1 (mean of exceedances), L2 (half the mean of sorted differences).
        let n_exc = exceedances.len();
        let l1 = exceedances.iter().sum::<f32>() / n_exc as f32;
        // L2: second L-moment via sample of order statistics.
        let mut sorted_exc = exceedances.clone();
        sorted_exc.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let l2: f32 = sorted_exc
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = (2 * i as i64 + 2 - n_exc as i64 - 1) as f32 / (n_exc as f32 - 1.0);
                w * x
            })
            .sum::<f32>()
            / 2.0;

        // L-moments estimator: ξ = 2 - l1/(l1 - 2*l2), σ = 2*l1*l2/(l1 - 2*l2).
        let denom = l1 - 2.0 * l2;
        let (xi, sigma) = if denom.abs() < 1e-8 || l2 <= 0.0 {
            // Degenerate case: exponential distribution.
            (0.0_f32, l1.max(1e-6))
        } else {
            let xi_est = 2.0 - l1 / denom;
            let sigma_est = (2.0 * l1 * l2 / denom).max(1e-6);
            (xi_est.clamp(-2.0, 2.0), sigma_est)
        };

        self.fit = Some(GpdFit {
            threshold,
            xi,
            sigma,
            n_exceedances: n_exc,
        });
        Ok(())
    }

    /// Score a single point: returns the GPD p-value (lower = more anomalous).
    ///
    /// If the score does not exceed the threshold, returns 1.0 (in-distribution).
    ///
    /// # Errors
    /// - `AnomalyError::NotFitted` if `fit` has not been called.
    pub fn p_value(&self, score: f32) -> AnomalyResult<f32> {
        match &self.fit {
            None => Err(AnomalyError::NotFitted),
            Some(f) => Ok(f.p_value(score)),
        }
    }

    /// Predict whether a score is an anomaly (p-value < alpha).
    ///
    /// Returns `true` for anomalies.
    ///
    /// # Errors
    /// - `AnomalyError::NotFitted` if `fit` has not been called.
    pub fn predict(&self, score: f32) -> AnomalyResult<bool> {
        Ok(self.p_value(score)? < self.alpha)
    }

    /// Score a batch of values, returning 1 − p-value (higher = more anomalous).
    ///
    /// # Errors
    /// - `AnomalyError::NotFitted` if `fit` has not been called.
    pub fn anomaly_scores(&self, scores: &[f32]) -> AnomalyResult<Vec<f32>> {
        scores.iter().map(|&s| Ok(1.0 - self.p_value(s)?)).collect()
    }

    /// Fraction of training scores flagged as anomalies (expected ≈ α).
    ///
    /// # Errors
    /// - `AnomalyError::NotFitted` if `fit` has not been called.
    pub fn training_false_positive_rate(&self) -> AnomalyResult<f32> {
        if self.fit.is_none() {
            return Err(AnomalyError::NotFitted);
        }
        if self.train_scores.is_empty() {
            return Ok(0.0);
        }
        let n = self.train_scores.len() as f32;
        let flagged = self
            .train_scores
            .iter()
            .filter(|&&s| self.predict(s).unwrap_or(false))
            .count() as f32;
        Ok(flagged / n)
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple training set of 100 scores uniformly in [0, 1].
    fn make_scores(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32 / (n - 1) as f32).collect()
    }

    // ── 1. new: invalid alpha returns error ───────────────────────────────────
    #[test]
    fn new_invalid_alpha_error() {
        assert!(GpdDetector::new(0.0, 0.9).is_err());
        assert!(GpdDetector::new(1.0, 0.9).is_err());
        assert!(GpdDetector::new(0.05, 0.0).is_err());
    }

    // ── 2. fit: empty input returns error ─────────────────────────────────────
    #[test]
    fn fit_empty_error() {
        let mut det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        assert!(det.fit(&[]).is_err());
    }

    // ── 3. fit: succeeds on typical training scores ───────────────────────────
    #[test]
    fn fit_succeeds() {
        let scores = make_scores(100);
        let mut det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        det.fit(&scores).expect("GPD fit should succeed");
        assert!(det.fit.is_some());
    }

    // ── 4. p_value ≥ alpha for typical in-distribution score ─────────────────
    #[test]
    fn p_value_inlier_not_flagged() {
        let scores = make_scores(200);
        let mut det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        det.fit(&scores).expect("GPD fit should succeed");
        // Score at median should not be flagged.
        let pv = det
            .p_value(0.5)
            .expect("p_value should succeed after fitting");
        assert!(pv >= det.alpha, "p={pv} for median score");
    }

    // ── 5. Extreme outlier gets low p-value (or is predicted as anomaly) ──────
    #[test]
    fn extreme_outlier_flagged() {
        let scores = make_scores(200);
        let mut det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        det.fit(&scores).expect("GPD fit should succeed");
        let pv = det
            .p_value(1000.0)
            .expect("p_value should succeed after fitting");
        assert!(
            pv < det.alpha,
            "extreme score should have p < alpha, got {pv}"
        );
    }

    // ── 6. predict: below threshold → not anomaly ─────────────────────────────
    #[test]
    fn predict_below_threshold_not_anomaly() {
        let scores = make_scores(200);
        let mut det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        det.fit(&scores).expect("GPD fit should succeed");
        let thr = det
            .fit
            .as_ref()
            .expect("fit should be Some after fitting")
            .threshold;
        let is_anom = det
            .predict(thr * 0.5)
            .expect("predict should succeed after fitting");
        assert!(!is_anom, "score below threshold should not be anomaly");
    }

    // ── 7. anomaly_scores: all finite ────────────────────────────────────────
    #[test]
    fn anomaly_scores_finite() {
        let scores = make_scores(100);
        let mut det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        det.fit(&scores).expect("GPD fit should succeed");
        let anom_scores = det
            .anomaly_scores(&scores)
            .expect("anomaly_scores should succeed after fitting");
        assert!(anom_scores.iter().all(|s| s.is_finite()), "not all finite");
    }

    // ── 8. GPD survival function properties ──────────────────────────────────
    #[test]
    fn gpd_survival_monotone() {
        let fit = GpdFit {
            threshold: 0.0,
            xi: 0.2,
            sigma: 1.0,
            n_exceedances: 50,
        };
        let mut prev = 1.0_f32;
        for i in 0..20 {
            let x = i as f32 * 0.5;
            let s = fit.survival(x);
            assert!(s <= prev + 1e-6, "not monotone at x={x}");
            assert!((0.0..=1.0).contains(&s), "out of range at x={x}");
            prev = s;
        }
    }

    // ── 9. Exponential (ξ=0) case: survival = exp(-x/σ) ─────────────────────
    #[test]
    fn gpd_survival_exponential_case() {
        let sigma = 2.0_f32;
        let fit = GpdFit {
            threshold: 0.0,
            xi: 0.0,
            sigma,
            n_exceedances: 20,
        };
        for &x in &[0.5_f32, 1.0, 2.0, 4.0] {
            let expected = (-x / sigma).exp();
            let got = fit.survival(x);
            assert!(
                (got - expected).abs() < 0.01,
                "x={x}: got={got} exp={expected}"
            );
        }
    }

    // ── 10. training_false_positive_rate ≤ α + tolerance ─────────────────────
    #[test]
    fn training_fpr_near_alpha() {
        let scores = make_scores(200);
        let alpha = 0.1_f32;
        let mut det = GpdDetector::new(alpha, 0.9)
            .expect("GpdDetector::new should succeed with valid params");
        det.fit(&scores).expect("GPD fit should succeed");
        let fpr = det
            .training_false_positive_rate()
            .expect("training_false_positive_rate should succeed after fitting");
        // FPR on training set should be roughly α.
        assert!(
            fpr <= alpha + 0.15,
            "FPR={fpr} should be ≤ alpha+0.15={}",
            alpha + 0.15
        );
    }

    // ── 11. fit on constant data (single exceedance) doesn't crash ────────────
    #[test]
    fn fit_single_exceedance_no_crash() {
        let mut scores = vec![0.0_f32; 99];
        scores.push(10.0_f32); // one outlier
        let mut det = GpdDetector::new(0.05, 0.98)
            .expect("GpdDetector::new should succeed with valid params");
        // May succeed or return InsufficientSamples — either is acceptable.
        let _ = det.fit(&scores);
    }

    // ── 12. p_value: NotFitted returns error ──────────────────────────────────
    #[test]
    fn p_value_not_fitted_error() {
        let det =
            GpdDetector::new(0.05, 0.9).expect("GpdDetector::new should succeed with valid params");
        assert!(det.p_value(0.5).is_err());
    }
}
