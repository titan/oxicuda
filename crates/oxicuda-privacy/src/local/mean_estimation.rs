//! Locally Differentially Private mean estimation (Duchi-Jordan-Wainwright 2018).
//!
//! Reference: Duchi JC, Jordan MI, Wainwright MJ (2018) "Minimax Optimal
//! Procedures for Locally Private Estimation", Journal of the American
//! Statistical Association 113(521):182–201. We implement the bounded scalar
//! / bounded vector unbiased mean mechanism described in Section 2.2 and
//! analysed in Theorem 1 (see also Algorithm 1 of the same paper).
//!
//! # Scalar mechanism
//! Given an input `x in [-R, R]`, the local mechanism returns a single bit
//! that is then mapped to the value `+B` or `-B` where
//!
//! ```text
//! B = R * (e^eps + 1) / (e^eps - 1)
//! ```
//!
//! The bit is sampled as follows: draw `U ~ Uniform(0, 1)`, set
//!
//! ```text
//! p = 1/2 + (x / (2R)) * (e^eps - 1) / (e^eps + 1)
//! ```
//!
//! and report `Z = +B` if `U < p`, else `Z = -B`. The mechanism is
//! - **unbiased**: `E[Z | X = x] = x` (Section 2.2),
//! - **eps-LDP**: the likelihood ratio between any two inputs is bounded by
//!   `e^eps` because the output takes only two values.
//!
//! # Vector mechanism
//! For `x in R^d` with `||x||_inf <= R`, the same mechanism is applied
//! coordinate-wise with **independent** randomness. By basic composition the
//! total local privacy cost is `d * eps`; documenting this responsibility on
//! the caller follows Duchi-Jordan-Wainwright (2018) Section 2.4 / Remark
//! after Theorem 1.
//!
//! # Aggregation
//! The trusted aggregator simply averages reports coordinate-wise:
//! `mu_hat_j = (1/n) * sum_i reports_{i, j}`. Because each report is unbiased
//! per-coordinate, so is the average; its variance is `B^2 / n` per
//! coordinate (Theorem 1 of the reference).

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

/// Tolerance for out-of-range checks: inputs are allowed to be within this
/// many ULP-scale units beyond the configured radius (handles harmless
/// floating-point drift on data already clipped to `[-R, R]`).
const RADIUS_TOLERANCE: f64 = 1e-9;

/// Configuration for the Duchi-Jordan-Wainwright LDP mean mechanism.
#[derive(Debug, Clone, Copy)]
pub struct LdpMeanConfig {
    /// Per-coordinate local privacy parameter epsilon > 0.
    pub epsilon: f64,
    /// Input radius R > 0 (each coordinate must satisfy `|x_j| <= R`).
    pub radius: f64,
}

impl LdpMeanConfig {
    /// Validate and construct an `LdpMeanConfig`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `epsilon <= 0` or non-finite.
    /// - `InvalidParameter` if `radius <= 0` or non-finite.
    pub fn new(epsilon: f64, radius: f64) -> PrivacyResult<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "epsilon must be > 0 and finite, got {epsilon}"
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "radius must be > 0 and finite, got {radius}"
            )));
        }
        Ok(Self { epsilon, radius })
    }
}

/// Duchi-Jordan-Wainwright LDP mean estimator.
#[derive(Debug, Clone, Copy)]
pub struct LdpMean {
    cfg: LdpMeanConfig,
}

impl LdpMean {
    /// Construct a new estimator after validating the configuration.
    ///
    /// # Errors
    /// Propagates `LdpMeanConfig::new` errors.
    pub fn new(cfg: LdpMeanConfig) -> PrivacyResult<Self> {
        // Re-validate explicitly so callers cannot smuggle in an unchecked
        // struct literal.
        let cfg = LdpMeanConfig::new(cfg.epsilon, cfg.radius)?;
        Ok(Self { cfg })
    }

    /// Active configuration.
    #[must_use]
    pub fn config(&self) -> &LdpMeanConfig {
        &self.cfg
    }

    /// Bias-correction amplitude `B = R * (e^eps + 1) / (e^eps - 1)`.
    #[must_use]
    pub fn amplitude(&self) -> f64 {
        let e = self.cfg.epsilon.exp();
        self.cfg.radius * (e + 1.0) / (e - 1.0)
    }

    /// Privatise a single scalar input `x in [-R, R]`.
    ///
    /// Returns either `+B` or `-B` with the calibrated probability so that
    /// `E[Z | X = x] = x` and the mechanism is eps-LDP.
    ///
    /// # Errors
    /// `InvalidParameter` if `|x|` exceeds `radius` (beyond a small
    /// floating-point tolerance) or `x` is non-finite.
    pub fn privatise_scalar(&self, x: f64, handle: &mut PrivacyHandle) -> PrivacyResult<f64> {
        if !x.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "input must be finite, got {x}"
            )));
        }
        let r = self.cfg.radius;
        if x.abs() > r + RADIUS_TOLERANCE {
            return Err(PrivacyError::InvalidParameter(format!(
                "input {x} outside radius +/- {r}"
            )));
        }
        // Clamp microscopic floating-point drift so the probability stays
        // strictly inside [0, 1] without altering meaningful values.
        let x_clamped = x.clamp(-r, r);
        let e = self.cfg.epsilon.exp();
        let bias = (e - 1.0) / (e + 1.0);
        let p_plus = 0.5 + (x_clamped / (2.0 * r)) * bias;
        let b = self.amplitude();
        let u = handle.rng.next_f64();
        Ok(if u < p_plus { b } else { -b })
    }

    /// Privatise a vector input by applying the scalar mechanism independently
    /// per coordinate. The composed local privacy cost is `dim * epsilon`
    /// (basic composition) — this is the caller's bookkeeping concern.
    ///
    /// # Errors
    /// - `EmptyInput` if `x` is empty.
    /// - Propagates `privatise_scalar` errors (out-of-range per coordinate).
    pub fn privatise_vector(
        &self,
        x: &[f64],
        handle: &mut PrivacyHandle,
    ) -> PrivacyResult<Vec<f64>> {
        if x.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let mut out = Vec::with_capacity(x.len());
        for &v in x.iter() {
            out.push(self.privatise_scalar(v, handle)?);
        }
        Ok(out)
    }

    /// Aggregate a batch of (vector) reports by coordinate-wise averaging:
    /// `mu_hat_j = (1/n) sum_i reports[i][j]`.
    ///
    /// # Errors
    /// - `EmptyInput` if `reports` is empty or the first report is empty.
    /// - `DimensionMismatch` if reports have differing lengths.
    pub fn aggregate(reports: &[Vec<f64>]) -> PrivacyResult<Vec<f64>> {
        if reports.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        let dim = reports[0].len();
        if dim == 0 {
            return Err(PrivacyError::EmptyInput);
        }
        let mut acc = vec![0.0f64; dim];
        for r in reports.iter() {
            if r.len() != dim {
                return Err(PrivacyError::DimensionMismatch {
                    expected: dim,
                    got: r.len(),
                });
            }
            for (a, &v) in acc.iter_mut().zip(r.iter()) {
                *a += v;
            }
        }
        let inv = 1.0 / (reports.len() as f64);
        for a in acc.iter_mut() {
            *a *= inv;
        }
        Ok(acc)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Unbiased mean: sample 20k privatised reports of a fixed value and
    //    verify the average is close to the true value within 3*B/sqrt(n).
    #[test]
    fn test_scalar_unbiased_mean_20k_reports() {
        let cfg = LdpMeanConfig::new(1.0, 1.0).expect("cfg");
        let m = LdpMean::new(cfg).expect("new");
        let mut handle = PrivacyHandle::new(80, 314_159);
        let true_x = 0.4f64;
        let n = 20_000usize;
        let reports: Vec<f64> = (0..n)
            .map(|_| m.privatise_scalar(true_x, &mut handle).expect("ok"))
            .collect();
        let mean = reports.iter().sum::<f64>() / (n as f64);
        let tol = 3.0 * m.amplitude() / (n as f64).sqrt();
        assert!(
            (mean - true_x).abs() < tol,
            "mean {mean} too far from {true_x} (tol {tol})"
        );
    }

    // 2. Out-of-bounds input errors.
    #[test]
    fn test_out_of_bounds_input_errors() {
        let cfg = LdpMeanConfig::new(2.0, 1.0).expect("cfg");
        let m = LdpMean::new(cfg).expect("new");
        let mut handle = PrivacyHandle::new(80, 0);
        assert!(m.privatise_scalar(1.5, &mut handle).is_err());
        assert!(m.privatise_scalar(-1.5, &mut handle).is_err());
        assert!(m.privatise_scalar(f64::NAN, &mut handle).is_err());
        assert!(m.privatise_scalar(f64::INFINITY, &mut handle).is_err());
    }

    // 3. Non-positive epsilon errors.
    #[test]
    fn test_non_positive_epsilon_errors() {
        assert!(LdpMeanConfig::new(0.0, 1.0).is_err());
        assert!(LdpMeanConfig::new(-1.0, 1.0).is_err());
        assert!(LdpMeanConfig::new(f64::NAN, 1.0).is_err());
        assert!(LdpMeanConfig::new(f64::INFINITY, 1.0).is_err());
    }

    // 4. Non-positive radius errors.
    #[test]
    fn test_non_positive_radius_errors() {
        assert!(LdpMeanConfig::new(1.0, 0.0).is_err());
        assert!(LdpMeanConfig::new(1.0, -2.0).is_err());
        assert!(LdpMeanConfig::new(1.0, f64::NAN).is_err());
        assert!(LdpMeanConfig::new(1.0, f64::INFINITY).is_err());
    }

    // 5. Vector mean unbiased per coordinate.
    #[test]
    fn test_vector_unbiased_per_coordinate() {
        let cfg = LdpMeanConfig::new(2.0, 1.0).expect("cfg");
        let m = LdpMean::new(cfg).expect("new");
        let mut handle = PrivacyHandle::new(80, 27_182);
        let true_x = [0.2f64, -0.5, 0.0, 0.8];
        let n = 20_000usize;
        let reports: Vec<Vec<f64>> = (0..n)
            .map(|_| m.privatise_vector(&true_x, &mut handle).expect("ok"))
            .collect();
        let mean = LdpMean::aggregate(&reports).expect("agg");
        let tol = 3.0 * m.amplitude() / (n as f64).sqrt();
        for (a, b) in mean.iter().zip(true_x.iter()) {
            assert!(
                (a - b).abs() < tol,
                "coord mean {a} vs truth {b}, tol {tol}"
            );
        }
    }

    // 6. Dimension mismatch in aggregate errors.
    #[test]
    fn test_aggregate_dim_mismatch_errors() {
        let reports = vec![vec![1.0, 2.0, 3.0], vec![0.0, 0.0]];
        let r = LdpMean::aggregate(&reports);
        assert!(matches!(r, Err(PrivacyError::DimensionMismatch { .. })));
    }

    // 7. Empty reports errors.
    #[test]
    fn test_empty_reports_errors() {
        let empty: Vec<Vec<f64>> = vec![];
        assert!(matches!(
            LdpMean::aggregate(&empty),
            Err(PrivacyError::EmptyInput)
        ));
        // Also an inner empty vector.
        let inner_empty: Vec<Vec<f64>> = vec![vec![]];
        assert!(matches!(
            LdpMean::aggregate(&inner_empty),
            Err(PrivacyError::EmptyInput)
        ));
    }

    // 8. Aggregate length matches first-report length.
    #[test]
    fn test_aggregate_length_matches_input() {
        let reports = vec![vec![1.0; 7], vec![2.0; 7], vec![-3.0; 7]];
        let out = LdpMean::aggregate(&reports).expect("ok");
        assert_eq!(out.len(), 7);
        // Each coord is the mean of [1, 2, -3] = 0.
        for v in out {
            assert!(v.abs() < 1e-12);
        }
    }

    // 9. Deterministic for fixed seed.
    #[test]
    fn test_deterministic_for_fixed_seed() {
        let cfg = LdpMeanConfig::new(1.5, 1.0).expect("cfg");
        let a = LdpMean::new(cfg).expect("a");
        let b = LdpMean::new(cfg).expect("b");
        let mut h_a = PrivacyHandle::new(80, 7);
        let mut h_b = PrivacyHandle::new(80, 7);
        let xs = [-0.3f64, 0.1, 0.7, -0.9, 0.0, 0.5, -0.5, 0.25];
        for &x in xs.iter() {
            let za = a.privatise_scalar(x, &mut h_a).expect("a");
            let zb = b.privatise_scalar(x, &mut h_b).expect("b");
            assert!((za - zb).abs() < 1e-15);
        }
    }

    // 10. Scalar privatisation always returns +B or -B exactly.
    #[test]
    fn test_scalar_returns_plus_or_minus_b_exactly() {
        let cfg = LdpMeanConfig::new(0.7, 2.5).expect("cfg");
        let m = LdpMean::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 99_991);
        let b = m.amplitude();
        for _ in 0..1_000 {
            let z = m.privatise_scalar(0.1, &mut handle).expect("ok");
            assert!(
                (z - b).abs() < 1e-12 || (z + b).abs() < 1e-12,
                "z {z} is neither +B nor -B (B={b})"
            );
        }
    }

    // 11. Smaller eps amplifies noise (B grows as eps -> 0).
    #[test]
    fn test_small_epsilon_amplifies_noise() {
        let m_small = LdpMean::new(LdpMeanConfig::new(0.1, 1.0).expect("a")).expect("a2");
        let m_large = LdpMean::new(LdpMeanConfig::new(5.0, 1.0).expect("b")).expect("b2");
        assert!(
            m_small.amplitude() > m_large.amplitude() * 10.0,
            "B(eps=0.1)={} should dominate B(eps=5)={}",
            m_small.amplitude(),
            m_large.amplitude()
        );
        // Empirical variance check on a large sample of x=0.
        let mut handle = PrivacyHandle::new(80, 1_001);
        let n = 4_000usize;
        let mut var = |m: &LdpMean| {
            let zs: Vec<f64> = (0..n)
                .map(|_| m.privatise_scalar(0.0, &mut handle).expect("ok"))
                .collect();
            let mu = zs.iter().sum::<f64>() / (n as f64);
            zs.iter().map(|&z| (z - mu).powi(2)).sum::<f64>() / (n as f64)
        };
        let v_small = var(&m_small);
        let v_large = var(&m_large);
        assert!(
            v_small > v_large * 10.0,
            "Var(eps=0.1)={v_small} should be ≫ Var(eps=5)={v_large}"
        );
    }

    // 12. Large eps approaches the input value: the amplitude B -> R = 1
    //     and the unbiased mechanism reports +R or -R with probabilities
    //     close to (1 + x/R)/2 and (1 - x/R)/2 (so the expectation equals x).
    #[test]
    fn test_large_epsilon_approaches_input() {
        // At eps=30, exp(eps) is astronomical so B -> R and the sample mean
        // over a moderate batch is very close to x with low variance.
        let cfg = LdpMeanConfig::new(30.0, 1.0).expect("cfg");
        let m = LdpMean::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 555);
        // B should be essentially R.
        assert!(
            (m.amplitude() - 1.0).abs() < 1e-6,
            "B should approach R, got {}",
            m.amplitude()
        );
        let true_x = 0.6f64;
        let n = 5_000usize;
        let reports: Vec<f64> = (0..n)
            .map(|_| m.privatise_scalar(true_x, &mut handle).expect("ok"))
            .collect();
        let mean = reports.iter().sum::<f64>() / (n as f64);
        // With B≈R, std per report is sqrt(1 - x^2) ≈ 0.8, so 3 sigma / sqrt(n) ≈ 0.034.
        assert!(
            (mean - true_x).abs() < 0.05,
            "mean {mean} not close to {true_x}"
        );
    }

    // 13. Single-element aggregate equals that element.
    #[test]
    fn test_single_element_aggregate_equals_element() {
        let only = vec![vec![1.5f64, -2.5, 0.0, 7.0]];
        let out = LdpMean::aggregate(&only).expect("ok");
        assert_eq!(out, only[0]);
    }

    // 14. privatise_vector on an empty input errors.
    #[test]
    fn test_privatise_vector_empty_errors() {
        let cfg = LdpMeanConfig::new(1.0, 1.0).expect("cfg");
        let m = LdpMean::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 0);
        let r = m.privatise_vector(&[], &mut handle);
        assert!(matches!(r, Err(PrivacyError::EmptyInput)));
    }

    // 15. Boundary inputs +/- R are accepted (no false positive on the
    //     radius check thanks to the tolerance).
    #[test]
    fn test_boundary_inputs_accepted() {
        let cfg = LdpMeanConfig::new(1.0, 1.0).expect("cfg");
        let m = LdpMean::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 42);
        assert!(m.privatise_scalar(1.0, &mut handle).is_ok());
        assert!(m.privatise_scalar(-1.0, &mut handle).is_ok());
        // And just-inside-tolerance values.
        assert!(m.privatise_scalar(1.0 + 1e-12, &mut handle).is_ok());
        assert!(m.privatise_scalar(-1.0 - 1e-12, &mut handle).is_ok());
    }
}
