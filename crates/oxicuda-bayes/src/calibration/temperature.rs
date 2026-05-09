//! Temperature scaling: post-hoc recalibration via a single learnable scalar T (Guo 2017).
//!
//! Given uncalibrated logits `z ∈ ℝ^{N×K}` and integer labels `y ∈ {0,..,K-1}^N`,
//! the optimisation
//! ```text
//!     T* = argmin_T  -(1/N) Σ_n  log softmax(z_n / T)[y_n]
//! ```
//! yields a temperature that minimises NLL on a held-out set without changing
//! the argmax (so accuracy is preserved). The resulting `softmax(z / T*)`
//! probabilities are typically much better calibrated.
//!
//! We solve for T via golden-section search on `T ∈ (0, T_max]` since the NLL
//! is unimodal in T for a fixed dataset (Guo 2017 §4.2). No autodiff is
//! required and the scalar-search is a few hundred float operations.

use crate::error::{BayesError, BayesResult};

/// In-place row-wise softmax over `[N, K]` logits.
fn row_softmax(logits: &mut [f32], k: usize) {
    let n = logits.len() / k;
    for i in 0..n {
        let row = &mut logits[i * k..(i + 1) * k];
        let mut m = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > m {
                m = v;
            }
        }
        let mut s = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            s += *v;
        }
        let inv = 1.0_f32 / s.max(1e-30);
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// NLL of `logits/T` against `labels`. Returns +inf for invalid T.
fn nll_at(logits: &[f32], labels: &[usize], k: usize, t: f32) -> f32 {
    if !(t.is_finite() && t > 0.0) {
        return f32::INFINITY;
    }
    let n = labels.len();
    let mut total = 0.0_f64;
    for (i, &y) in labels.iter().enumerate() {
        let row = &logits[i * k..(i + 1) * k];
        // log-softmax stable: log p_y = z_y/T − logsumexp(z/T)
        let mut m = f32::NEG_INFINITY;
        for &v in row.iter() {
            let scaled = v / t;
            if scaled > m {
                m = scaled;
            }
        }
        let mut s = 0.0_f64;
        for &v in row.iter() {
            s += ((v / t - m) as f64).exp();
        }
        let log_z = (m as f64) + s.ln();
        let log_p = (row[y] / t) as f64 - log_z;
        total += -log_p;
    }
    (total / n as f64) as f32
}

/// Configuration for [`TemperatureScaler::fit`].
#[derive(Debug, Clone)]
pub struct TemperatureFitConfig {
    /// Lower bracket for the search (must be > 0). Default 0.05.
    pub t_lo: f32,
    /// Upper bracket for the search. Default 10.0.
    pub t_hi: f32,
    /// Maximum number of golden-section iterations. Default 64.
    pub max_iter: usize,
    /// Absolute tolerance on the bracket width. Default 1e-4.
    pub tol: f32,
}

impl Default for TemperatureFitConfig {
    fn default() -> Self {
        Self {
            t_lo: 0.05,
            t_hi: 10.0,
            max_iter: 64,
            tol: 1e-4,
        }
    }
}

/// Post-hoc temperature scaler for multi-class probabilistic classifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct TemperatureScaler {
    /// Learned temperature; `1.0` means an identity transform.
    pub temperature: f32,
}

impl Default for TemperatureScaler {
    fn default() -> Self {
        Self { temperature: 1.0 }
    }
}

impl TemperatureScaler {
    /// New scaler initialised to a fixed temperature. Errors if `T <= 0` or non-finite.
    ///
    /// # Errors
    /// [`BayesError::InvalidTemperature`] if `t <= 0` or not finite.
    pub fn new(t: f32) -> BayesResult<Self> {
        if !(t.is_finite() && t > 0.0) {
            return Err(BayesError::InvalidTemperature { temp: t });
        }
        Ok(Self { temperature: t })
    }

    /// Apply the temperature: produce calibrated probabilities `softmax(z / T)`.
    /// Returns a freshly allocated `[N·K]` row-major buffer.
    ///
    /// # Errors
    /// - [`BayesError::DimensionMismatch`] if `logits.len()` is not a multiple of `n_classes`
    ///   or `n_classes == 0`.
    pub fn apply(&self, logits: &[f32], n_classes: usize) -> BayesResult<Vec<f32>> {
        if n_classes == 0 || logits.is_empty() || logits.len() % n_classes != 0 {
            return Err(BayesError::DimensionMismatch {
                expected: n_classes,
                got: logits.len(),
            });
        }
        let inv = 1.0_f32 / self.temperature;
        let mut out: Vec<f32> = logits.iter().map(|v| v * inv).collect();
        row_softmax(&mut out, n_classes);
        Ok(out)
    }

    /// Fit the temperature by minimising NLL on `(logits, labels)` with
    /// golden-section search on `[t_lo, t_hi]`.
    ///
    /// Returns the resulting scaler.
    ///
    /// # Errors
    /// - [`BayesError::CalibrationSetEmpty`] if `labels.is_empty()`.
    /// - [`BayesError::DimensionMismatch`] if `logits.len() != labels.len()*n_classes`.
    /// - [`BayesError::TemperatureNotFinite`] if optimisation produces a non-finite T.
    /// - [`BayesError::InvalidTemperature`] if `t_lo <= 0`, `t_hi <= t_lo`,
    ///   or any out-of-range label is found.
    pub fn fit(
        logits: &[f32],
        labels: &[usize],
        n_classes: usize,
        cfg: &TemperatureFitConfig,
    ) -> BayesResult<Self> {
        if labels.is_empty() || n_classes == 0 {
            return Err(BayesError::CalibrationSetEmpty);
        }
        if logits.len() != labels.len() * n_classes {
            return Err(BayesError::DimensionMismatch {
                expected: labels.len() * n_classes,
                got: logits.len(),
            });
        }
        if cfg.t_lo <= 0.0 || !cfg.t_lo.is_finite() || cfg.t_hi <= cfg.t_lo || !cfg.t_hi.is_finite()
        {
            return Err(BayesError::InvalidTemperature { temp: cfg.t_lo });
        }
        for &y in labels {
            if y >= n_classes {
                return Err(BayesError::DimensionMismatch {
                    expected: n_classes,
                    got: y + 1,
                });
            }
        }

        // Golden-section search on T in [t_lo, t_hi].
        let phi: f32 = 0.5 * (5.0_f32.sqrt() - 1.0); // 0.6180339...
        let mut a = cfg.t_lo;
        let mut b = cfg.t_hi;
        let mut c = b - phi * (b - a);
        let mut d = a + phi * (b - a);
        let mut fc = nll_at(logits, labels, n_classes, c);
        let mut fd = nll_at(logits, labels, n_classes, d);
        for _ in 0..cfg.max_iter {
            if (b - a) <= cfg.tol {
                break;
            }
            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = b - phi * (b - a);
                fc = nll_at(logits, labels, n_classes, c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + phi * (b - a);
                fd = nll_at(logits, labels, n_classes, d);
            }
        }
        let t = 0.5 * (a + b);
        if !(t.is_finite() && t > 0.0) {
            return Err(BayesError::TemperatureNotFinite);
        }
        Ok(Self { temperature: t })
    }

    /// Convenience: fit a scaler with default golden-section config.
    ///
    /// # Errors
    /// Propagates errors from [`TemperatureScaler::fit`].
    pub fn fit_default(logits: &[f32], labels: &[usize], n_classes: usize) -> BayesResult<Self> {
        Self::fit(logits, labels, n_classes, &TemperatureFitConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::metrics::{
        expected_calibration_error, negative_log_likelihood, top1_confidences,
    };

    /// Synthetic over-confident dataset: large logits with only modest accuracy.
    fn over_confident_dataset(n: usize) -> (Vec<f32>, Vec<usize>) {
        // 3 classes; logits drawn so argmax is class 0 with margin 5;
        // labels are ~70% class 0, the rest randomised (but not class 0).
        let mut logits = Vec::with_capacity(n * 3);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            logits.extend_from_slice(&[5.0, 0.0, 0.0]);
            labels.push(if i % 10 < 7 { 0 } else { 1 + (i % 2) });
        }
        (logits, labels)
    }

    #[test]
    fn temperature_default_is_identity() {
        let scaler = TemperatureScaler::default();
        assert!((scaler.temperature - 1.0).abs() < 1e-7);
    }

    #[test]
    fn temperature_apply_softmax_normalised() {
        let scaler = TemperatureScaler::new(2.0).unwrap();
        let logits = vec![1.0_f32, 2.0, 3.0];
        let p = scaler.apply(&logits, 3).unwrap();
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn temperature_apply_preserves_argmax() {
        let scaler = TemperatureScaler::new(0.3).unwrap();
        let logits = vec![1.0_f32, 5.0, 2.0, 3.0, 0.5, 0.1];
        let p = scaler.apply(&logits, 3).unwrap();
        // first row argmax should still be index 1
        let row0 = &p[0..3];
        let mut best = 0;
        let mut bv = row0[0];
        for (k, &v) in row0.iter().enumerate().skip(1) {
            if v > bv {
                bv = v;
                best = k;
            }
        }
        assert_eq!(best, 1);
    }

    #[test]
    fn temperature_fit_reduces_ece_when_overconfident() {
        let (logits, labels) = over_confident_dataset(200);
        let scaler = TemperatureScaler::fit_default(&logits, &labels, 3).unwrap();
        assert!(scaler.temperature > 0.0);
        let scaled = scaler.apply(&logits, 3).unwrap();
        let baseline = scaler; // copy of struct for use after move
        let p_before = TemperatureScaler::default().apply(&logits, 3).unwrap();
        let (c_before, ok_before) = top1_confidences(&p_before, &labels, 3).unwrap();
        let (c_after, ok_after) = top1_confidences(&scaled, &labels, 3).unwrap();
        let ece_before = expected_calibration_error(&c_before, &ok_before, 10).unwrap();
        let ece_after = expected_calibration_error(&c_after, &ok_after, 10).unwrap();
        assert!(
            ece_after <= ece_before + 1e-4,
            "ECE should not increase: before={ece_before}, after={ece_after}"
        );
        assert!(baseline.temperature > 0.0);
    }

    #[test]
    fn temperature_fit_lowers_nll() {
        let (logits, labels) = over_confident_dataset(200);
        let nll_before = negative_log_likelihood(
            &TemperatureScaler::default().apply(&logits, 3).unwrap(),
            &labels,
            3,
        )
        .unwrap();
        let scaler = TemperatureScaler::fit_default(&logits, &labels, 3).unwrap();
        let nll_after =
            negative_log_likelihood(&scaler.apply(&logits, 3).unwrap(), &labels, 3).unwrap();
        assert!(nll_after <= nll_before + 1e-4);
    }

    #[test]
    fn temperature_invalid_construction_rejected() {
        assert!(TemperatureScaler::new(0.0).is_err());
        assert!(TemperatureScaler::new(-1.0).is_err());
        assert!(TemperatureScaler::new(f32::NAN).is_err());
        assert!(TemperatureScaler::new(f32::INFINITY).is_err());
    }

    #[test]
    fn temperature_fit_rejects_empty() {
        let r = TemperatureScaler::fit_default(&[], &[], 3);
        assert!(r.is_err());
    }

    #[test]
    fn temperature_fit_rejects_dim_mismatch() {
        let r = TemperatureScaler::fit_default(&[1.0_f32, 2.0], &[0_usize], 3);
        assert!(r.is_err());
    }

    #[test]
    fn temperature_fit_rejects_invalid_label() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let labels = vec![5_usize];
        let r = TemperatureScaler::fit_default(&logits, &labels, 3);
        assert!(r.is_err());
    }

    #[test]
    fn temperature_apply_rejects_bad_shape() {
        let scaler = TemperatureScaler::new(1.0).unwrap();
        let r = scaler.apply(&[1.0_f32, 2.0, 3.0], 2);
        assert!(r.is_err());
    }

    #[test]
    fn temperature_fit_with_custom_bracket() {
        let (logits, labels) = over_confident_dataset(50);
        let cfg = TemperatureFitConfig {
            t_lo: 0.5,
            t_hi: 4.0,
            max_iter: 32,
            tol: 1e-3,
        };
        let s = TemperatureScaler::fit(&logits, &labels, 3, &cfg).unwrap();
        assert!(s.temperature >= cfg.t_lo - 1e-4);
        assert!(s.temperature <= cfg.t_hi + 1e-4);
    }
}
