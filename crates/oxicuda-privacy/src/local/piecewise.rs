//! Piecewise mechanism for local differentially private mean estimation.
//!
//! Reference: Wang, Xiao, Yang & Yi (2019), "Collecting and Analyzing
//! Multidimensional Data with Local Differential Privacy", *International
//! Conference on Data Engineering* (ICDE), §4.
//!
//! # Motivation
//! For continuous numerical attributes bounded in `[-R, R]`, the
//! Duchi-Jordan-Wainwright (2018) mechanism reports only `±B` with
//! `B = R · (e^ε + 1) / (e^ε − 1)` — large amplitude at small ε. The
//! Wang et al. **piecewise mechanism** replaces the two-point output with a
//! continuous distribution supported on `[−C, C]` whose density is
//! piecewise-uniform (constant `p` on a narrow "high" region near a
//! configurable centre and constant `p / e^ε` on the surrounding "low"
//! region). The conditional variance is strictly smaller than the
//! Duchi-Jordan-Wainwright (2018) variance at the same ε for `ε ≳ 1.2`,
//! yielding lower mean-squared error for both scalar and vector means.
//!
//! # Construction (§4.2 of Wang et al. 2019, scalar form)
//! Let `e_half = exp(ε/2)`. Define
//!
//! ```text
//! C    = (e_half + 1) / (e_half − 1)
//! ```
//!
//! For input `t ∈ [−1, 1]` (the rescaled `x / R`) the high-density
//! window is `[L_t, R_t]` with
//!
//! ```text
//! L_t = (C + 1) / 2 · t − (C − 1) / 2
//! R_t = L_t + (C − 1)
//! ```
//!
//! Sampling proceeds in two stages:
//! 1. Draw `U ~ Uniform([0, 1))`. With probability `p = e_half / (e_half + 1)`
//!    sample `V` uniformly from the high-density region `[L_t, R_t]`.
//! 2. Otherwise sample `V` uniformly from the two-piece low-density region
//!    `[−C, L_t] ∪ [R_t, C]` — total length `C + 1` — and add the offset
//!    that skips over `[L_t, R_t]`.
//!
//! The output is `Z = V · R` (rescaled back to the original input range);
//! `E[Z | X = x] = x` (Theorem 2 of Wang et al.) and the variance per
//! report is at most `R² · (4 e_half / (e_half − 1)²)` (Theorem 3).
//!
//! # Vector privatisation
//! For `x ∈ R^d` with `||x||_∞ ≤ R`, the mechanism is applied
//! coordinate-wise with independent randomness. The composed per-record
//! cost under basic composition is `d · ε`; the caller is responsible for
//! budgeting. (Wang et al. §4.3 also describe a randomised sub-coordinate
//! sampling variant; we expose the straightforward per-coordinate form here
//! to mirror the Duchi LDP API in `local::mean_estimation`.)

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::PrivacyHandle;

/// Tolerance for the radius check (matches the convention used by
/// `local::mean_estimation`).
const RADIUS_TOLERANCE: f64 = 1e-9;

/// Configuration for the piecewise mechanism.
#[derive(Debug, Clone, Copy)]
pub struct PiecewiseConfig {
    /// Per-coordinate local privacy parameter `ε > 0`.
    pub epsilon: f64,
    /// Input radius `R > 0` (each coordinate must satisfy `|x| ≤ R`).
    pub radius: f64,
}

impl PiecewiseConfig {
    /// Validate and construct a `PiecewiseConfig`.
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `epsilon ≤ 0` or non-finite.
    /// - `InvalidParameter` if `radius ≤ 0` or non-finite.
    pub fn new(epsilon: f64, radius: f64) -> PrivacyResult<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "radius must be > 0 and finite, got {radius}"
            )));
        }
        Ok(Self { epsilon, radius })
    }
}

/// Piecewise mechanism (Wang-Xiao-Yang-Yi 2019, §4) for LDP mean estimation.
#[derive(Debug, Clone, Copy)]
pub struct PiecewiseMechanism {
    /// Active configuration.
    pub cfg: PiecewiseConfig,
}

impl PiecewiseMechanism {
    /// Construct after revalidating the configuration.
    ///
    /// # Errors
    /// Propagates `PiecewiseConfig::new` errors.
    pub fn new(cfg: PiecewiseConfig) -> PrivacyResult<Self> {
        let cfg = PiecewiseConfig::new(cfg.epsilon, cfg.radius)?;
        Ok(Self { cfg })
    }

    /// Active configuration.
    #[must_use]
    pub fn config(&self) -> &PiecewiseConfig {
        &self.cfg
    }

    /// Output half-width `C = (e^(ε/2) + 1) / (e^(ε/2) − 1)` in the
    /// normalised `[−C, C]` scale; multiply by `radius` for the actual
    /// output bound `radius · C`.
    #[must_use]
    pub fn output_c(&self) -> f64 {
        let e_half = (self.cfg.epsilon * 0.5).exp();
        (e_half + 1.0) / (e_half - 1.0)
    }

    /// Privatise a single scalar input `x ∈ [−R, R]`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `x` is non-finite.
    /// - `InvalidParameter` if `|x| > radius + tolerance`.
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
        // Rescale to t ∈ [−1, 1]. The radius check passed, so the clamp only
        // trims microscopic FP drift.
        let t = (x / r).clamp(-1.0, 1.0);
        let e_half = (self.cfg.epsilon * 0.5).exp();
        let c = (e_half + 1.0) / (e_half - 1.0);
        // High-density window for the current input.
        let l_t = 0.5 * (c + 1.0) * t - 0.5 * (c - 1.0);
        let r_t = l_t + (c - 1.0);
        // Probability of landing in the high-density region.
        let p_high = e_half / (e_half + 1.0);
        let u = handle.rng.next_f64();
        let v = if u < p_high {
            // Inside [L_t, R_t]: scale a fresh uniform into the window.
            let u2 = handle.rng.next_f64();
            l_t + u2 * (r_t - l_t)
        } else {
            // Outside: draw from the two-piece region [−C, L_t] ∪ [R_t, C]
            // whose total length is (L_t − (−C)) + (C − R_t) = C + 1.
            // Sample u3 ∈ [0, C + 1) and place it in the correct piece.
            let total = c + 1.0;
            let u3 = handle.rng.next_f64() * total;
            let left_length = l_t + c; // (L_t − (−C))
            if u3 < left_length {
                -c + u3
            } else {
                r_t + (u3 - left_length)
            }
        };
        Ok(v * r)
    }

    /// Privatise a vector input by applying `privatise_scalar` per coordinate
    /// with independent randomness. Per-record composition cost is `d · ε`;
    /// callers are responsible for budgeting (see Wang et al. §4.3).
    ///
    /// # Errors
    /// - `EmptyInput` if `x` is empty.
    /// - Propagates `privatise_scalar` errors.
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

    /// Coordinate-wise mean of a collection of vector reports:
    /// `μ̂_j = (1/n) Σ_i reports[i][j]`.
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
    use crate::local::{LdpMean, LdpMeanConfig};

    fn variance(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let mu = samples.iter().sum::<f64>() / n;
        samples.iter().map(|&v| (v - mu).powi(2)).sum::<f64>() / n
    }

    // 1. Empirical mean of 50K reports ≈ true mean within 4·B/√n where
    //    B = radius · C bounds the output magnitude.
    #[test]
    fn test_scalar_unbiased_mean_50k_reports() {
        let cfg = PiecewiseConfig::new(1.0, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 314_159);
        let true_x = 0.3f64;
        let n = 50_000usize;
        let reports: Vec<f64> = (0..n)
            .map(|_| m.privatise_scalar(true_x, &mut handle).expect("ok"))
            .collect();
        let mean = reports.iter().sum::<f64>() / (n as f64);
        let b = cfg.radius * m.output_c();
        let tol = 4.0 * b / (n as f64).sqrt();
        assert!(
            (mean - true_x).abs() < tol,
            "mean {mean} too far from {true_x} (tol {tol})"
        );
    }

    // 2. Output magnitudes bounded by radius · C.
    #[test]
    fn test_output_bounded_by_radius_times_c() {
        let cfg = PiecewiseConfig::new(2.0, 2.5).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 42);
        let bound = cfg.radius * m.output_c();
        // Allow a generous numeric slack for FP rounding inside the
        // boundary-piece sampler.
        for _ in 0..10_000 {
            let z = m.privatise_scalar(0.0, &mut handle).expect("ok");
            assert!(z.abs() <= bound + 1e-9, "|{z}| > bound {bound}");
        }
    }

    // 3. |x| > radius returns InvalidParameter.
    #[test]
    fn test_out_of_range_input_errors() {
        let cfg = PiecewiseConfig::new(1.0, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 0);
        assert!(matches!(
            m.privatise_scalar(1.5, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            m.privatise_scalar(-1.5, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            m.privatise_scalar(f64::NAN, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            m.privatise_scalar(f64::INFINITY, &mut handle),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 4. epsilon ≤ 0 returns NonPositiveEpsilon.
    #[test]
    fn test_nonpositive_epsilon_errors() {
        assert!(matches!(
            PiecewiseConfig::new(0.0, 1.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            PiecewiseConfig::new(-1.0, 1.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            PiecewiseConfig::new(f64::NAN, 1.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
        assert!(matches!(
            PiecewiseConfig::new(f64::INFINITY, 1.0),
            Err(PrivacyError::NonPositiveEpsilon(_))
        ));
    }

    // 5. radius ≤ 0 returns InvalidParameter.
    #[test]
    fn test_nonpositive_radius_errors() {
        assert!(matches!(
            PiecewiseConfig::new(1.0, 0.0),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            PiecewiseConfig::new(1.0, -2.0),
            Err(PrivacyError::InvalidParameter(_))
        ));
        assert!(matches!(
            PiecewiseConfig::new(1.0, f64::NAN),
            Err(PrivacyError::InvalidParameter(_))
        ));
    }

    // 6. Vector mean per-coordinate unbiased.
    #[test]
    fn test_vector_mean_unbiased_per_coordinate() {
        let cfg = PiecewiseConfig::new(2.0, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 27_182);
        let true_x = [0.2f64, -0.5, 0.0, 0.8];
        let n = 20_000usize;
        let reports: Vec<Vec<f64>> = (0..n)
            .map(|_| m.privatise_vector(&true_x, &mut handle).expect("ok"))
            .collect();
        let mean = PiecewiseMechanism::aggregate(&reports).expect("agg");
        let b = cfg.radius * m.output_c();
        let tol = 4.0 * b / (n as f64).sqrt();
        for (a, b) in mean.iter().zip(true_x.iter()) {
            assert!(
                (a - b).abs() < tol,
                "coord mean {a} vs truth {b}, tol {tol}"
            );
        }
    }

    // 7. privatise_vector on empty input errors.
    #[test]
    fn test_privatise_vector_empty_errors() {
        let cfg = PiecewiseConfig::new(1.0, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 0);
        assert!(matches!(
            m.privatise_vector(&[], &mut handle),
            Err(PrivacyError::EmptyInput)
        ));
    }

    // 8. aggregate empty errors.
    #[test]
    fn test_aggregate_empty_errors() {
        let empty: Vec<Vec<f64>> = vec![];
        assert!(matches!(
            PiecewiseMechanism::aggregate(&empty),
            Err(PrivacyError::EmptyInput)
        ));
        let inner_empty: Vec<Vec<f64>> = vec![vec![]];
        assert!(matches!(
            PiecewiseMechanism::aggregate(&inner_empty),
            Err(PrivacyError::EmptyInput)
        ));
    }

    // 9. aggregate dim mismatch errors.
    #[test]
    fn test_aggregate_dim_mismatch_errors() {
        let reports = vec![vec![1.0, 2.0, 3.0], vec![0.0, 0.0]];
        assert!(matches!(
            PiecewiseMechanism::aggregate(&reports),
            Err(PrivacyError::DimensionMismatch { .. })
        ));
    }

    // 10. Deterministic given a fixed seed.
    #[test]
    fn test_deterministic_for_fixed_seed() {
        let cfg = PiecewiseConfig::new(1.5, 1.0).expect("cfg");
        let a = PiecewiseMechanism::new(cfg).expect("a");
        let b = PiecewiseMechanism::new(cfg).expect("b");
        let mut h_a = PrivacyHandle::new(80, 7);
        let mut h_b = PrivacyHandle::new(80, 7);
        let xs = [-0.3f64, 0.1, 0.7, -0.9, 0.0, 0.5, -0.5, 0.25];
        for &x in xs.iter() {
            let za = a.privatise_scalar(x, &mut h_a).expect("a");
            let zb = b.privatise_scalar(x, &mut h_b).expect("b");
            assert!((za - zb).abs() < 1e-15, "za {za} != zb {zb}");
        }
    }

    // 11. x = 0 → output centred at 0 (mean over 20K within tolerance).
    #[test]
    fn test_x_zero_output_centred() {
        let cfg = PiecewiseConfig::new(1.0, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 999);
        let n = 20_000usize;
        let reports: Vec<f64> = (0..n)
            .map(|_| m.privatise_scalar(0.0, &mut handle).expect("ok"))
            .collect();
        let mean = reports.iter().sum::<f64>() / (n as f64);
        let b = cfg.radius * m.output_c();
        let tol = 4.0 * b / (n as f64).sqrt();
        assert!(mean.abs() < tol, "mean {mean} not near 0 (tol {tol})");
    }

    // 12. x = +radius → empirical mean positive (sign of expectation).
    #[test]
    fn test_x_positive_radius_mean_positive() {
        let cfg = PiecewiseConfig::new(1.5, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 12_345);
        let n = 10_000usize;
        let reports: Vec<f64> = (0..n)
            .map(|_| m.privatise_scalar(1.0, &mut handle).expect("ok"))
            .collect();
        let mean = reports.iter().sum::<f64>() / (n as f64);
        assert!(mean > 0.5, "x=+1 should give mean > 0.5, got {mean}");
    }

    // 13. x = −radius → empirical mean negative.
    #[test]
    fn test_x_negative_radius_mean_negative() {
        let cfg = PiecewiseConfig::new(1.5, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 23_456);
        let n = 10_000usize;
        let reports: Vec<f64> = (0..n)
            .map(|_| m.privatise_scalar(-1.0, &mut handle).expect("ok"))
            .collect();
        let mean = reports.iter().sum::<f64>() / (n as f64);
        assert!(mean < -0.5, "x=-1 should give mean < -0.5, got {mean}");
    }

    // 14. Lower variance than Duchi at the same ε on the same input x = 0.
    //
    // Wang-Xiao-Yang-Yi (2019, §5) prove the piecewise variance is strictly
    // smaller than Duchi's once `ε ≥ ε₀ ≈ 0.61`; we test ε = 2 where the gap
    // is unambiguous.
    #[test]
    fn test_variance_lower_than_duchi() {
        let eps = 2.0f64;
        let radius = 1.0f64;
        let pcfg = PiecewiseConfig::new(eps, radius).expect("pcfg");
        let pm = PiecewiseMechanism::new(pcfg).expect("pm");
        let dcfg = LdpMeanConfig::new(eps, radius).expect("dcfg");
        let dm = LdpMean::new(dcfg).expect("dm");
        let mut h_p = PrivacyHandle::new(80, 91_827);
        let mut h_d = PrivacyHandle::new(80, 91_827);
        let n = 50_000usize;
        let p_samples: Vec<f64> = (0..n)
            .map(|_| pm.privatise_scalar(0.0, &mut h_p).expect("ok"))
            .collect();
        let d_samples: Vec<f64> = (0..n)
            .map(|_| dm.privatise_scalar(0.0, &mut h_d).expect("ok"))
            .collect();
        let var_p = variance(&p_samples);
        let var_d = variance(&d_samples);
        assert!(
            var_p < var_d,
            "piecewise var {var_p} should be < Duchi var {var_d}"
        );
    }

    // 15. Boundary inputs ±R accepted (no false positive on the tolerance).
    #[test]
    fn test_boundary_inputs_accepted() {
        let cfg = PiecewiseConfig::new(1.0, 1.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 42);
        assert!(m.privatise_scalar(1.0, &mut handle).is_ok());
        assert!(m.privatise_scalar(-1.0, &mut handle).is_ok());
        assert!(m.privatise_scalar(1.0 + 1e-12, &mut handle).is_ok());
        assert!(m.privatise_scalar(-1.0 - 1e-12, &mut handle).is_ok());
    }

    // 16. Output values strictly inside [−B, B] (sanity).
    #[test]
    fn test_output_strictly_in_bracket() {
        let cfg = PiecewiseConfig::new(0.7, 3.0).expect("cfg");
        let m = PiecewiseMechanism::new(cfg).expect("m");
        let mut handle = PrivacyHandle::new(80, 77);
        let bound = cfg.radius * m.output_c();
        for _ in 0..5_000 {
            let z = m.privatise_scalar(0.5, &mut handle).expect("ok");
            assert!(z >= -bound - 1e-9, "z {z} < -{bound}");
            assert!(z <= bound + 1e-9, "z {z} > {bound}");
        }
    }

    // 17. Single-element aggregate equals that element.
    #[test]
    fn test_single_element_aggregate_equals_element() {
        let only = vec![vec![1.5f64, -2.5, 0.0, 7.0]];
        let out = PiecewiseMechanism::aggregate(&only).expect("ok");
        assert_eq!(out, only[0]);
    }

    // 18. output_c shrinks as ε grows (C → 1 as ε → ∞).
    #[test]
    fn test_output_c_shrinks_with_epsilon() {
        let small =
            PiecewiseMechanism::new(PiecewiseConfig::new(0.5, 1.0).expect("s")).expect("s2");
        let medium =
            PiecewiseMechanism::new(PiecewiseConfig::new(2.0, 1.0).expect("m")).expect("m2");
        let large =
            PiecewiseMechanism::new(PiecewiseConfig::new(8.0, 1.0).expect("l")).expect("l2");
        assert!(
            small.output_c() > medium.output_c(),
            "C(0.5)={} should > C(2)={}",
            small.output_c(),
            medium.output_c()
        );
        assert!(
            medium.output_c() > large.output_c(),
            "C(2)={} should > C(8)={}",
            medium.output_c(),
            large.output_c()
        );
        // C(ε → ∞) → 1.
        assert!(
            (large.output_c() - 1.0).abs() < 0.1,
            "C(8) should approach 1, got {}",
            large.output_c()
        );
    }
}
