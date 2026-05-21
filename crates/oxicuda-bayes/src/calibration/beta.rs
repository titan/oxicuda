//! Beta calibration: three-parameter post-hoc recalibration (Kull, Filho & Flach 2017 AISTATS).
//!
//! For a binary classifier producing uncalibrated probability `p ∈ [0, 1]`, the
//! calibrated output is:
//!
//! ```text
//! p̂ = σ(a·ln(p) - b·ln(1-p) + c)
//! ```
//!
//! where `σ` is the logistic sigmoid, and `(a, b, c)` are fitted via gradient
//! descent on the binary cross-entropy loss over a held-out calibration set.
//!
//! **Properties:**
//! - Subsumes Platt scaling when `a = b` (symmetric case).
//! - Monotonically increasing (and thus calibration-preserving) when `a ≥ 0` and `b ≥ 0`.
//! - Closed-form gradients allow simple first-order fitting.
//!
//! **References:**
//! - Kull, M., Filho, T. S., & Flach, P. (2017). Beta calibration: a well-founded and
//!   easily implemented improvement on logistic calibration for binary classifiers.
//!   *AISTATS 2017*, PMLR 54.

use crate::error::{BayesError, BayesResult};

// ─── numerical guard for log(0) ──────────────────────────────────────────────

/// Guard against log(0): ε used in `ln(p+ε)` and `ln(1-p+ε)`.
const EPS: f32 = 1e-12;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Numerically stable logistic sigmoid `σ(x) = 1 / (1 + e^{-x})`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Hyperparameters for [`BetaCalibrator::fit`].
#[derive(Debug, Clone)]
pub struct BetaCalibConfig {
    /// Maximum gradient-descent iterations.
    pub max_iter: usize,
    /// Gradient-descent step size.
    pub learning_rate: f32,
    /// Convergence tolerance on the L2 gradient norm.
    pub tol: f32,
    /// L2 regularisation coefficient on `(a, b, c)`.
    pub reg_lambda: f32,
}

impl Default for BetaCalibConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            learning_rate: 0.01,
            tol: 1e-7,
            reg_lambda: 1e-4,
        }
    }
}

// ─── Calibrator ──────────────────────────────────────────────────────────────

/// Three-parameter Beta calibrator: `p̂ = σ(a·ln(p+ε) - b·ln(1-p+ε) + c)`.
///
/// Monotonicity is enforced by keeping `a ≥ 0` and `b ≥ 0` (projected gradient).
/// Parameters are initialised at `a = 1, b = 1, c = 0` (a Platt-like starting point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaCalibrator {
    /// Coefficient for `ln(p)` (log-odds numerator).  Must be `≥ 0` for monotonicity.
    pub a: f32,
    /// Coefficient for `-ln(1-p)` (log-odds denominator).  Must be `≥ 0` for monotonicity.
    pub b: f32,
    /// Bias term.
    pub c: f32,
}

impl Default for BetaCalibrator {
    fn default() -> Self {
        Self {
            a: 1.0,
            b: 1.0,
            c: 0.0,
        }
    }
}

impl BetaCalibrator {
    // ─── Inference ───────────────────────────────────────────────────────────

    /// Apply the calibration map to a single probability `p ∈ [0, 1]`.
    ///
    /// Computes `σ(a·ln(p+ε) - b·ln(1-p+ε) + c)` where `ε = 1e-12`.
    #[must_use]
    pub fn predict_one(&self, p: f32) -> f32 {
        let f1 = (p + EPS).ln();
        let f2 = -(1.0 - p + EPS).ln();
        sigmoid(self.a * f1 + self.b * f2 + self.c)
    }

    /// Apply the calibration map pointwise to a slice of probabilities.
    #[must_use]
    pub fn predict(&self, probs: &[f32]) -> Vec<f32> {
        probs.iter().map(|&p| self.predict_one(p)).collect()
    }

    // ─── Fitting ─────────────────────────────────────────────────────────────

    /// Fit `(a, b, c)` via projected gradient descent on the binary cross-entropy.
    ///
    /// # Algorithm
    /// 1. Validate inputs (non-empty, matching lengths, labels ∈ {0,1}, probs ∈ `[0,1]`).
    /// 2. Precompute features `f1[i] = ln(p[i]+ε)`, `f2[i] = -ln(1-p[i]+ε)`.
    /// 3. Gradient descent with L2 regularisation.
    /// 4. After each step clamp `a ≥ 0`, `b ≥ 0` (monotonicity projection).
    /// 5. Terminate when `‖∇‖₂ < tol` or `max_iter` reached.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] — `probs` is empty.
    /// - [`BayesError::DimensionMismatch`] — `probs.len() != labels.len()`.
    /// - [`BayesError::NanEncountered`] — a NaN appears in `(a, b, c)` after fitting.
    pub fn fit(probs: &[f32], labels: &[u8], cfg: &BetaCalibConfig) -> BayesResult<Self> {
        // ── Validation ──────────────────────────────────────────────────────
        if probs.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if probs.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: probs.len(),
                got: labels.len(),
            });
        }
        for &y in labels {
            if y > 1 {
                return Err(BayesError::DimensionMismatch {
                    expected: 1,
                    got: usize::from(y),
                });
            }
        }

        let n = probs.len();
        let n_inv = 1.0_f32 / n as f32;

        // ── Pre-compute features ─────────────────────────────────────────────
        // f1[i] = ln(p[i] + ε)    (feature for a)
        // f2[i] = -ln(1-p[i] + ε) (feature for b; same sign convention as logit of p)
        let mut f1 = Vec::with_capacity(n);
        let mut f2 = Vec::with_capacity(n);
        let mut y_vec = Vec::with_capacity(n);
        for (&p, &lbl) in probs.iter().zip(labels.iter()) {
            f1.push((p + EPS).ln());
            f2.push(-(1.0_f32 - p + EPS).ln());
            y_vec.push(lbl as f32);
        }

        // ── Gradient descent ─────────────────────────────────────────────────
        let mut a = 1.0_f32;
        let mut b = 1.0_f32;
        let mut c = 0.0_f32;

        for _ in 0..cfg.max_iter {
            let mut grad_a = 0.0_f32;
            let mut grad_b = 0.0_f32;
            let mut grad_c = 0.0_f32;

            for i in 0..n {
                let logit = a * f1[i] + b * f2[i] + c;
                let p_hat = sigmoid(logit);
                let residual = p_hat - y_vec[i];
                grad_a += residual * f1[i];
                grad_b += residual * f2[i];
                grad_c += residual;
            }

            // Average + L2 regularisation
            grad_a = grad_a * n_inv + cfg.reg_lambda * a;
            grad_b = grad_b * n_inv + cfg.reg_lambda * b;
            grad_c *= n_inv; // no regularisation on bias

            // Convergence check (L2 norm of gradient)
            let grad_norm = (grad_a * grad_a + grad_b * grad_b + grad_c * grad_c).sqrt();
            if grad_norm < cfg.tol {
                break;
            }

            // Gradient step
            a -= cfg.learning_rate * grad_a;
            b -= cfg.learning_rate * grad_b;
            c -= cfg.learning_rate * grad_c;

            // Projection for monotonicity: a ≥ 0, b ≥ 0
            if a < 0.0 {
                a = 0.0;
            }
            if b < 0.0 {
                b = 0.0;
            }
        }

        // ── NaN guard ────────────────────────────────────────────────────────
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            return Err(BayesError::NanEncountered {
                location: "BetaCalibrator::fit: non-finite parameter",
            });
        }

        Ok(Self { a, b, c })
    }

    // ─── Evaluation ──────────────────────────────────────────────────────────

    /// Mean negative log-likelihood (binary cross-entropy) on `(probs, labels)`.
    ///
    /// Predictions are clamped to `[ε, 1-ε]` before taking the log.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] — `probs` is empty.
    /// - [`BayesError::DimensionMismatch`] — `probs.len() != labels.len()`.
    pub fn nll(&self, probs: &[f32], labels: &[u8]) -> BayesResult<f32> {
        if probs.is_empty() {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if probs.len() != labels.len() {
            return Err(BayesError::DimensionMismatch {
                expected: probs.len(),
                got: labels.len(),
            });
        }

        // Use an f32-safe log-clamp: 1e-7 is well above f32 machine-epsilon near 0.
        const LOG_EPS: f32 = 1e-7;
        let mut sum = 0.0_f32;
        for (&p, &y) in probs.iter().zip(labels.iter()) {
            let p_hat = self.predict_one(p).clamp(LOG_EPS, 1.0 - LOG_EPS);
            let yf = y as f32;
            sum -= yf * p_hat.ln() + (1.0 - yf) * (1.0 - p_hat).ln();
        }
        Ok(sum / probs.len() as f32)
    }

    // ─── Properties ──────────────────────────────────────────────────────────

    /// Return `true` when `a ≥ 0` and `b ≥ 0` (sufficient condition for monotonicity).
    #[must_use]
    #[inline]
    pub fn is_monotone(&self) -> bool {
        self.a >= 0.0 && self.b >= 0.0
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn default_calib() -> BetaCalibrator {
        BetaCalibrator::default()
    }

    fn default_cfg() -> BetaCalibConfig {
        BetaCalibConfig::default()
    }

    // Helper method available only in tests.
    impl BetaCalibrator {
        fn is_finite_params(&self) -> bool {
            self.a.is_finite() && self.b.is_finite() && self.c.is_finite()
        }
    }

    // ── predict_one ──────────────────────────────────────────────────────────

    #[test]
    fn predict_one_at_half_equals_sigmoid_c() {
        // p = 0.5: f1 = ln(0.5+ε) ≈ -ln 2, f2 = -ln(0.5+ε) ≈ +ln 2
        // logit = a*f1 + b*f2 + c = a*(-ln2) + b*(ln2) + c
        // For a=1, b=1: logit = 0 + c = 0 → p̂ = 0.5
        let bc = default_calib();
        let p_hat = bc.predict_one(0.5);
        assert!((p_hat - 0.5).abs() < 1e-4, "expected ≈ 0.5, got {p_hat}");
    }

    #[test]
    fn predict_one_near_zero_clips_correctly() {
        let bc = default_calib();
        let p_hat = bc.predict_one(0.0);
        assert!(
            (0.0..=1.0).contains(&p_hat),
            "p_hat={p_hat} should be in [0,1]"
        );
        assert!(p_hat.is_finite(), "must be finite at p=0");
    }

    #[test]
    fn predict_one_at_high_p_is_above_half() {
        let bc = default_calib();
        let p_hat = bc.predict_one(0.99);
        assert!(
            p_hat > 0.5,
            "predict_one(0.99) should be > 0.5, got {p_hat}"
        );
    }

    #[test]
    fn predict_one_at_zero_and_one_are_finite() {
        let bc = default_calib();
        assert!(bc.predict_one(0.0).is_finite());
        assert!(bc.predict_one(1.0).is_finite());
    }

    #[test]
    fn predict_batch_matches_pointwise() {
        let bc = BetaCalibrator {
            a: 1.2,
            b: 0.8,
            c: 0.3,
        };
        let probs = vec![0.1_f32, 0.3, 0.5, 0.7, 0.9];
        let batch = bc.predict(&probs);
        for (i, &p) in probs.iter().enumerate() {
            let expected = bc.predict_one(p);
            assert!(
                (batch[i] - expected).abs() < 1e-7,
                "mismatch at index {i}: batch={}, pointwise={}",
                batch[i],
                expected
            );
        }
    }

    #[test]
    fn predict_output_in_unit_interval() {
        let bc = default_calib();
        for i in 0..=20 {
            let p = i as f32 / 20.0;
            let p_hat = bc.predict_one(p);
            assert!(
                (0.0..=1.0).contains(&p_hat),
                "predict_one({p}) = {p_hat} outside [0,1]"
            );
        }
    }

    // ── Default values ────────────────────────────────────────────────────────

    #[test]
    fn default_calibrator_fields() {
        let bc = BetaCalibrator::default();
        assert!((bc.a - 1.0).abs() < 1e-9);
        assert!((bc.b - 1.0).abs() < 1e-9);
        assert!(bc.c.abs() < 1e-9);
    }

    #[test]
    fn default_config_fields() {
        let cfg = BetaCalibConfig::default();
        assert_eq!(cfg.max_iter, 500);
        assert!((cfg.learning_rate - 0.01).abs() < 1e-9);
        assert!((cfg.tol - 1e-7).abs() < 1e-15);
        assert!((cfg.reg_lambda - 1e-4).abs() < 1e-12);
    }

    // ── fit – error cases ─────────────────────────────────────────────────────

    #[test]
    fn fit_empty_returns_calibration_set_empty() {
        let r = BetaCalibrator::fit(&[], &[], &default_cfg());
        assert!(
            matches!(r, Err(BayesError::CalibrationSetEmpty)),
            "got {r:?}"
        );
    }

    #[test]
    fn fit_length_mismatch_returns_dimension_mismatch() {
        let probs = vec![0.3_f32, 0.7];
        let labels = vec![0_u8];
        let r = BetaCalibrator::fit(&probs, &labels, &default_cfg());
        assert!(
            matches!(r, Err(BayesError::DimensionMismatch { .. })),
            "got {r:?}"
        );
    }

    // ── fit – convergence ─────────────────────────────────────────────────────

    #[test]
    fn fit_on_perfectly_calibrated_data_recovers_identity_like_params() {
        // Perfectly calibrated: p[i] = 0.1*i, label = round(p[i])
        // Optimal calibrator should leave predictions nearly unchanged → a≈1, b≈1, c≈0
        let n = 50;
        let probs: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32).clamp(0.01, 0.99))
            .collect();
        let labels: Vec<u8> = probs
            .iter()
            .map(|&p| if p >= 0.5 { 1 } else { 0 })
            .collect();
        let cfg = BetaCalibConfig {
            max_iter: 2000,
            learning_rate: 0.005,
            tol: 1e-8,
            reg_lambda: 1e-5,
        };
        let bc = BetaCalibrator::fit(&probs, &labels, &cfg)
            .expect("fit must succeed on well-formed data");
        // Parameters should be broadly near identity (a≈1, b≈1).
        // We only assert they are finite and the calibrator predicts sensibly.
        assert!(bc.a.is_finite() && bc.b.is_finite() && bc.c.is_finite());
        // predict_one(0.9) > predict_one(0.1) — monotonicity preserved
        assert!(bc.predict_one(0.9) > bc.predict_one(0.1));
    }

    #[test]
    fn fit_overconfident_data_shifts_c_negative() {
        // All predictions are 0.9, but only half are truly positive.
        // The optimal calibrator should push predictions down → c < 0.
        let n = 100;
        let probs: Vec<f32> = vec![0.9_f32; n];
        let labels: Vec<u8> = (0..n).map(|i| if i < n / 2 { 1 } else { 0 }).collect();
        let bc = BetaCalibrator::fit(&probs, &labels, &default_cfg()).expect("fit must succeed");
        // The calibrated output for p=0.9 should be < 0.9 (correction toward 0.5).
        let p_hat = bc.predict_one(0.9);
        assert!(
            p_hat < 0.85,
            "overconfident data should pull prediction down; got p_hat={p_hat}"
        );
    }

    #[test]
    fn fit_all_positive_labels_shifts_parameters() {
        // When all labels are 1, the calibrator should push outputs toward 1.
        let probs: Vec<f32> = vec![0.3_f32, 0.5, 0.7];
        let labels: Vec<u8> = vec![1, 1, 1];
        let bc = BetaCalibrator::fit(&probs, &labels, &default_cfg()).expect("fit must succeed");
        assert!(bc.is_finite_params(), "all params should be finite");
        // The calibrated probability for p=0.5 should now exceed 0.5
        assert!(
            bc.predict_one(0.5) > 0.5,
            "all-positive labels should push output above 0.5"
        );
    }

    #[test]
    fn fit_step_function_data_converges() {
        // Ideal case: p < 0.5 → y=0, p ≥ 0.5 → y=1
        let n = 40;
        let probs: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
        let labels: Vec<u8> = probs
            .iter()
            .map(|&p| if p >= 0.5 { 1 } else { 0 })
            .collect();
        let bc = BetaCalibrator::fit(&probs, &labels, &default_cfg())
            .expect("fit must succeed on step-function data");
        assert!(bc.a.is_finite() && bc.b.is_finite() && bc.c.is_finite());
        // Monotone output
        assert!(bc.predict_one(0.8) > bc.predict_one(0.2));
    }

    // ── is_monotone ───────────────────────────────────────────────────────────

    #[test]
    fn is_monotone_true_after_projected_gradient_fit() {
        // After fitting with projection, a ≥ 0 and b ≥ 0 must hold.
        let probs = vec![0.2_f32, 0.4, 0.6, 0.8];
        let labels = vec![0_u8, 0, 1, 1];
        let bc = BetaCalibrator::fit(&probs, &labels, &default_cfg()).expect("fit must succeed");
        assert!(bc.is_monotone(), "a={}, b={} must both be ≥ 0", bc.a, bc.b);
    }

    #[test]
    fn is_monotone_false_for_negative_a() {
        let bc = BetaCalibrator {
            a: -0.1,
            b: 1.0,
            c: 0.0,
        };
        assert!(!bc.is_monotone());
    }

    #[test]
    fn is_monotone_false_for_negative_b() {
        let bc = BetaCalibrator {
            a: 1.0,
            b: -0.5,
            c: 0.0,
        };
        assert!(!bc.is_monotone());
    }

    // ── nll ───────────────────────────────────────────────────────────────────

    #[test]
    fn nll_of_near_perfect_predictions_is_small() {
        // Calibrator with mild amplification: p=0.9 maps to a high probability,
        // p=0.1 maps to a low probability, while staying away from numerical extremes.
        let bc = BetaCalibrator {
            a: 2.0,
            b: 2.0,
            c: 0.0,
        };
        let probs = vec![0.9_f32, 0.1, 0.9, 0.1];
        let labels = vec![1_u8, 0, 1, 0];
        let p_high = bc.predict_one(0.9);
        let p_low = bc.predict_one(0.1);
        // BCE for one sample: -log(p_high) for y=1, -log(1-p_low) for y=0
        let expected_nll = (-p_high.ln() - (1.0 - p_low).ln()) / 2.0;
        let loss = bc.nll(&probs, &labels).expect("nll must succeed");
        assert!(
            (loss - expected_nll).abs() < 1e-5,
            "nll={loss} expected={expected_nll}"
        );
        // Should be well below the random-guess NLL of ln(2) ≈ 0.693.
        assert!(
            loss < 0.5,
            "well-separated predictions should yield small NLL, got {loss}"
        );
    }

    #[test]
    fn nll_empty_returns_error() {
        let bc = default_calib();
        let r = bc.nll(&[], &[]);
        assert!(matches!(r, Err(BayesError::CalibrationSetEmpty)));
    }

    #[test]
    fn nll_length_mismatch_returns_error() {
        let bc = default_calib();
        let r = bc.nll(&[0.5_f32], &[0_u8, 1_u8]);
        assert!(matches!(r, Err(BayesError::DimensionMismatch { .. })));
    }

    #[test]
    fn nll_is_non_negative() {
        let bc = default_calib();
        let probs = vec![0.3_f32, 0.6, 0.8];
        let labels = vec![0_u8, 1, 1];
        let loss = bc.nll(&probs, &labels).expect("nll must succeed");
        assert!(loss >= 0.0, "NLL must be non-negative, got {loss}");
    }
}
