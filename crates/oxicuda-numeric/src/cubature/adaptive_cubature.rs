//! Adaptive multi-dimensional cubature with the Berntsen–Espelid–Genz error estimator.
//!
//! This module wraps the Genz–Malik (1980) fully-symmetric degree-7 basic rule with the
//! robust *local* error estimator of Berntsen, Espelid and Genz (1991), and exposes it
//! through the [`AdaptiveCubature`] driver that subdivides the integration region until
//! the requested absolute **or** relative tolerance is met, or a function-evaluation
//! budget is exhausted.
//!
//! # The basic rule
//!
//! On a `d`-dimensional hyperrectangle `∏[aᵢ, bᵢ]` the Genz–Malik basic rule is a
//! fully-symmetric formula that evaluates the integrand on five "generators":
//!
//! * the centre,
//! * the `2d` axis points at `±λ₂`,
//! * the `2d` axis points at `±λ₃`,
//! * the `2d(d-1)` face points at `(±λ₄, ±λ₄)`,
//! * the `2ᵈ` vertices at `(±λ₅, …, ±λ₅)`.
//!
//! Combining these evaluations with one set of weights gives a degree-7 estimate of the
//! integral.
//!
//! # The Berntsen–Espelid–Genz error estimate
//!
//! Rather than the bare degree-5 / degree-7 difference, BEG-1991 forms several *null
//! rules* (linear combinations of the same evaluations that integrate the constant — and
//! some higher monomials — to zero). The magnitudes of the null-rule results, denoted
//! `E₃ ≥ E₂ ≥ E₁`, probe successively higher-frequency content of the integrand. The
//! estimator then chooses, *per region*, between an "asymptotic" (4th-power) and a
//! "non-asymptotic" (linear) extrapolation depending on whether the integrand looks
//! smooth there:
//!
//! ```text
//! r = max(E₁/E₂, E₂/E₃)                       (smoothness ratio; 0 if a denom is 0)
//! E = (10 r) · max(E₁, E₂, E₃)                 if r ≥ 1   (non-asymptotic, cautious)
//! E = (10 r⁴) · max(E₁, E₂, E₃)                if r < 1   (asymptotic, optimistic)
//! ```
//!
//! followed by a small absolute safety floor. This makes the error estimate far more
//! reliable for non-smooth or oscillatory integrands than the single-difference Genz–Malik
//! estimate, at no extra integrand evaluations.
//!
//! # The driver
//!
//! [`AdaptiveCubature`] keeps a heap of regions keyed by error, repeatedly bisecting the
//! region with the largest estimated error along its most-varying axis (a globally
//! adaptive strategy). It stops when the *total* estimated error satisfies
//! `E_total ≤ max(abs_tol, rel_tol · |I_total|)` or when adding another subdivision would
//! exceed `max_eval` integrand evaluations.
//!
//! References:
//! - A. C. Genz and A. A. Malik, "Remarks on algorithm 006: An adaptive algorithm for
//!   numerical integration over an N-dimensional rectangular region",
//!   *J. Comput. Appl. Math.* 6 (1980), 295–302.
//! - J. Berntsen, T. O. Espelid, A. Genz, "An adaptive algorithm for the approximate
//!   calculation of multiple integrals", *ACM Trans. Math. Softw.* 17 (1991), 437–451.

use crate::error::{NumericError, NumericResult};

/// One sub-region of the integration domain together with its local estimates.
#[derive(Debug, Clone)]
struct Region {
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Degree-7 estimate of the integral over this region.
    value: f64,
    /// Berntsen–Espelid–Genz local error estimate.
    error: f64,
    /// Axis along which this region should next be bisected.
    split_dim: usize,
}

/// Result of an [`AdaptiveCubature`] run.
#[derive(Debug, Clone, Copy)]
pub struct CubatureResult {
    /// Approximation to the integral.
    pub value: f64,
    /// Estimated absolute error of [`value`](Self::value).
    pub error: f64,
    /// Number of integrand evaluations consumed.
    pub evaluations: usize,
    /// Number of sub-regions in the final partition.
    pub regions: usize,
    /// Whether the requested tolerance was met before the evaluation budget ran out.
    pub converged: bool,
}

/// Adaptive multi-dimensional cubature driver.
///
/// Configured with an absolute tolerance, a relative tolerance and a maximum number of
/// integrand evaluations. Construct with [`AdaptiveCubature::new`] (which validates the
/// parameters) and integrate with [`AdaptiveCubature::integrate`].
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveCubature {
    /// Absolute error target.
    pub abs_tol: f64,
    /// Relative error target (relative to `|I|`).
    pub rel_tol: f64,
    /// Maximum number of integrand evaluations.
    pub max_eval: usize,
}

impl AdaptiveCubature {
    /// Create and validate an adaptive cubature configuration.
    ///
    /// # Errors
    /// Returns [`NumericError::InvalidConfiguration`] when both tolerances are
    /// non-positive, when either tolerance is negative or non-finite, or when
    /// `max_eval == 0`.
    pub fn new(abs_tol: f64, rel_tol: f64, max_eval: usize) -> NumericResult<Self> {
        if !abs_tol.is_finite() || abs_tol < 0.0 {
            return Err(NumericError::InvalidConfiguration(format!(
                "AdaptiveCubature: abs_tol must be finite and ≥ 0, got {abs_tol}"
            )));
        }
        if !rel_tol.is_finite() || rel_tol < 0.0 {
            return Err(NumericError::InvalidConfiguration(format!(
                "AdaptiveCubature: rel_tol must be finite and ≥ 0, got {rel_tol}"
            )));
        }
        if abs_tol == 0.0 && rel_tol == 0.0 {
            return Err(NumericError::InvalidConfiguration(
                "AdaptiveCubature: at least one of abs_tol, rel_tol must be > 0".to_string(),
            ));
        }
        if max_eval == 0 {
            return Err(NumericError::InvalidConfiguration(
                "AdaptiveCubature: max_eval must be ≥ 1".to_string(),
            ));
        }
        Ok(Self {
            abs_tol,
            rel_tol,
            max_eval,
        })
    }

    /// Integrate `f` over the hyperrectangle `∏[lo[i], hi[i]]`.
    ///
    /// `f` maps a point in ℝᵈ to a scalar (and may itself fail, propagating the error).
    /// Returns the approximation, its estimated error, evaluation count and a convergence
    /// flag.
    ///
    /// # Errors
    /// Returns [`NumericError::DimensionMismatch`] when `lo` and `hi` differ in length,
    /// [`NumericError::EmptyInput`] when the domain is zero-dimensional,
    /// [`NumericError::InvalidParameter`] when any `lo[i] > hi[i]` or a bound is
    /// non-finite, and propagates any error returned by `f`.
    pub fn integrate<F>(&self, f: F, lo: &[f64], hi: &[f64]) -> NumericResult<CubatureResult>
    where
        F: Fn(&[f64]) -> NumericResult<f64>,
    {
        let d = lo.len();
        if d != hi.len() {
            return Err(NumericError::DimensionMismatch { a: d, b: hi.len() });
        }
        if d == 0 {
            return Err(NumericError::EmptyInput);
        }
        for i in 0..d {
            if !lo[i].is_finite() || !hi[i].is_finite() {
                return Err(NumericError::InvalidParameter(format!(
                    "AdaptiveCubature: non-finite bound at axis {i}: [{}, {}]",
                    lo[i], hi[i]
                )));
            }
            if lo[i] > hi[i] {
                return Err(NumericError::InvalidParameter(format!(
                    "AdaptiveCubature: lo > hi at axis {i}: [{}, {}]",
                    lo[i], hi[i]
                )));
            }
        }

        // Per-region cost (integrand evaluations) of one basic-rule application.
        let evals_per_region = basic_rule_point_count(d);

        // Initial region.
        let mut evaluations = 0usize;
        let (value0, error0, dim0) = self.basic_rule(&f, lo, hi)?;
        evaluations += evals_per_region;

        let mut regions = vec![Region {
            lo: lo.to_vec(),
            hi: hi.to_vec(),
            value: value0,
            error: error0,
            split_dim: dim0,
        }];
        let mut total_value = value0;
        let mut total_error = error0;

        loop {
            let tol = self.abs_tol.max(self.rel_tol * total_value.abs());
            if total_error <= tol {
                return Ok(CubatureResult {
                    value: total_value,
                    error: total_error,
                    evaluations,
                    regions: regions.len(),
                    converged: true,
                });
            }
            // Stop if another bisection (two more rule applications) would overrun budget.
            if evaluations + 2 * evals_per_region > self.max_eval {
                return Ok(CubatureResult {
                    value: total_value,
                    error: total_error,
                    evaluations,
                    regions: regions.len(),
                    converged: false,
                });
            }

            // Select the region carrying the most error.
            let mut worst = 0usize;
            let mut worst_err = regions[0].error;
            for (i, r) in regions.iter().enumerate().skip(1) {
                if r.error > worst_err {
                    worst_err = r.error;
                    worst = i;
                }
            }
            let parent = regions.swap_remove(worst);

            // Bisect along the parent's chosen split axis.
            let axis = parent.split_dim;
            let mid = 0.5 * (parent.lo[axis] + parent.hi[axis]);
            let mut left_hi = parent.hi.clone();
            left_hi[axis] = mid;
            let mut right_lo = parent.lo.clone();
            right_lo[axis] = mid;

            let (v_l, e_l, d_l) = self.basic_rule(&f, &parent.lo, &left_hi)?;
            let (v_r, e_r, d_r) = self.basic_rule(&f, &right_lo, &parent.hi)?;
            evaluations += 2 * evals_per_region;

            // Update running totals (subtract the parent, add the two children).
            total_value += (v_l + v_r) - parent.value;
            total_error += (e_l + e_r) - parent.error;

            regions.push(Region {
                lo: parent.lo,
                hi: left_hi,
                value: v_l,
                error: e_l,
                split_dim: d_l,
            });
            regions.push(Region {
                lo: right_lo,
                hi: parent.hi,
                value: v_r,
                error: e_r,
                split_dim: d_r,
            });
        }
    }

    /// Apply the Genz–Malik basic rule with the Berntsen–Espelid–Genz error estimate to a
    /// single hyperrectangle. Returns `(value, error, split_dim)`.
    fn basic_rule<F>(&self, f: &F, lo: &[f64], hi: &[f64]) -> NumericResult<(f64, f64, usize)>
    where
        F: Fn(&[f64]) -> NumericResult<f64>,
    {
        let d = lo.len();
        let df = d as f64;
        let center: Vec<f64> = (0..d).map(|i| 0.5 * (lo[i] + hi[i])).collect();
        let half: Vec<f64> = (0..d).map(|i| 0.5 * (hi[i] - lo[i])).collect();
        let vol: f64 = half.iter().map(|h| 2.0 * h).product();

        // Generator radii (Genz–Malik 1980).
        let lambda2 = (9.0_f64 / 70.0).sqrt();
        let lambda3 = (9.0_f64 / 10.0).sqrt();
        let lambda4 = lambda3;
        let lambda5 = (9.0_f64 / 19.0).sqrt();

        // Degree-7 weights.
        let w1 = (12_824.0 - 9120.0 * df + 400.0 * df * df) / 19_683.0;
        let w2 = 980.0 / 6561.0;
        let w3 = (1820.0 - 400.0 * df) / 19_683.0;
        let w4 = 200.0 / 19_683.0;
        let w5 = 6859.0 / 19_683.0 / 2_f64.powi(d as i32);

        // Sums of integrand evaluations by generator class.
        let f_center = f(&center)?;
        let mut sum2 = 0.0; // ±λ₂ axis points
        let mut sum3 = 0.0; // ±λ₃ axis points
        let mut sum4 = 0.0; // face points (±λ₄, ±λ₄)
        let mut sum5 = 0.0; // vertex points (±λ₅,…)
        // Per-axis fourth-difference indicator for the split-axis choice.
        let mut diff_axis = vec![0.0_f64; d];

        for i in 0..d {
            let mut p = center.clone();
            p[i] = center[i] + lambda2 * half[i];
            let f2p = f(&p)?;
            p[i] = center[i] - lambda2 * half[i];
            let f2m = f(&p)?;
            sum2 += f2p + f2m;

            p[i] = center[i] + lambda3 * half[i];
            let f3p = f(&p)?;
            p[i] = center[i] - lambda3 * half[i];
            let f3m = f(&p)?;
            sum3 += f3p + f3m;

            // Fourth divided difference along axis i (Genz–Malik split heuristic):
            // |f(+λ₃) - 2f(0) + f(-λ₃) - (λ₃/λ₂)²(f(+λ₂) - 2f(0) + f(-λ₂))|.
            let ratio = (lambda3 / lambda2).powi(2);
            let diff = (f3p + f3m - 2.0 * f_center) - ratio * (f2p + f2m - 2.0 * f_center);
            diff_axis[i] = diff.abs();
        }

        for i in 0..d {
            for j in (i + 1)..d {
                for &sx in &[-1.0_f64, 1.0] {
                    for &sy in &[-1.0_f64, 1.0] {
                        let mut p = center.clone();
                        p[i] += sx * lambda4 * half[i];
                        p[j] += sy * lambda4 * half[j];
                        sum4 += f(&p)?;
                    }
                }
            }
        }

        let total_vertices = 1u32 << d;
        for vi in 0..total_vertices {
            let mut p = center.clone();
            for k in 0..d {
                let s = if (vi >> k) & 1 == 1 { 1.0 } else { -1.0 };
                p[k] += s * lambda5 * half[k];
            }
            sum5 += f(&p)?;
        }

        // Degree-7 integral estimate.
        let value = vol * (w1 * f_center + w2 * sum2 + w3 * sum3 + w4 * sum4 + w5 * sum5);

        // --- Berntsen–Espelid–Genz null-rule error estimate ---------------------------
        // Three null rules of decreasing degree formed from the centre and the two axis
        // generator sums. Each integrates a constant (and successively fewer monomials) to
        // zero; the magnitude of its action measures the corresponding spectral content.
        // The coefficients below are the standard Genz–Malik null-rule weights (the same
        // family used by Cubpack / HIntLib for the 7-point 1-D embedded structure).
        let two_d = 2.0 * df;
        // E₃ (lowest degree, most local): centre vs all axis evaluations.
        let null3 = (sum2 + sum3 - two_d * 2.0 * f_center).abs();
        // E₂ (middle degree): contrast the two axis radii.
        let null2 = (sum3 - 7.0 * sum2 + 6.0 * two_d * f_center).abs();
        // E₁ (highest degree, smoothest): the degree-5/degree-7 GM difference.
        let null1 = {
            // Embedded degree-5 weights (Genz–Malik 1980, eq. for the lower rule).
            let p1 = (729.0 - 950.0 * df + 50.0 * df * df) / 729.0;
            let p2 = 245.0 / 486.0;
            let p3 = (265.0 - 100.0 * df) / 1458.0;
            let p4 = 25.0 / 729.0;
            let lower = vol * (p1 * f_center + p2 * sum2 + p3 * sum3 + p4 * sum4);
            (value - lower).abs()
        };

        // `null1` is the degree-5/degree-7 difference and is already scaled by `vol`
        // (both `value` and `lower` carry the volume factor). The lower-degree null rules
        // are formed from raw evaluation sums, so scale them by the region volume to put
        // all three error probes on the same footing as `value`.
        let e_high = null1;
        let e_mid = vol.abs() * null2;
        let e_low = vol.abs() * null3;

        let error = berntsen_espelid_genz_error(e_high, e_mid, e_low);

        // Choose the split axis with the largest fourth difference.
        let mut split_dim = 0usize;
        let mut best = diff_axis[0];
        for (i, &v) in diff_axis.iter().enumerate().skip(1) {
            if v > best {
                best = v;
                split_dim = i;
            }
        }

        Ok((value, error, split_dim))
    }
}

/// Number of integrand evaluations consumed by one Genz–Malik basic-rule application in
/// dimension `d`: `1 + 4d + 2d(d-1) + 2ᵈ`.
#[must_use]
fn basic_rule_point_count(d: usize) -> usize {
    let dd = d;
    1 + 4 * dd + 2 * dd * (dd.saturating_sub(1)) + (1usize << dd)
}

/// Combine the three null-rule magnitudes into a single local error estimate following
/// the Berntsen–Espelid–Genz (1991) heuristic.
///
/// `e_high ≥` smoothness probe of highest degree (`E₁`), `e_mid` (`E₂`), `e_low` (`E₃`)
/// of lowest degree. Returns the estimated absolute error of the basic rule.
#[must_use]
fn berntsen_espelid_genz_error(e_high: f64, e_mid: f64, e_low: f64) -> f64 {
    let e1 = e_high;
    let e2 = e_mid;
    let e3 = e_low;
    let e_max = e1.max(e2).max(e3);
    if e_max == 0.0 {
        return 0.0;
    }
    // Smoothness ratio: how fast the null-rule magnitudes decay with degree.
    let r1 = if e2 > 0.0 { e1 / e2 } else { 0.0 };
    let r2 = if e3 > 0.0 { e2 / e3 } else { 0.0 };
    let r = r1.max(r2);
    // Asymptotic (smooth) regime: r < 1 → quartic extrapolation; otherwise cautious.
    let scaled = if r >= 1.0 {
        10.0 * r * e_max
    } else {
        10.0 * r.powi(4) * e_max
    };
    // Never claim less error than the leading (smoothest) null rule resolves, and apply a
    // tiny absolute safety floor proportional to the dominant null rule.
    scaled.max(0.5 * e1).max(1.0e-15 * e_max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn point_count_formula() {
        // d=1: 1 + 4 + 0 + 2 = 7 ; d=2: 1 + 8 + 4 + 4 = 17 ; d=3: 1+12+12+8 = 33.
        assert_eq!(basic_rule_point_count(1), 7);
        assert_eq!(basic_rule_point_count(2), 17);
        assert_eq!(basic_rule_point_count(3), 33);
    }

    #[test]
    fn constant_integral_2d() {
        let ac = AdaptiveCubature::new(1.0e-10, 0.0, 10_000).expect("cfg");
        let f = |_x: &[f64]| -> NumericResult<f64> { Ok(3.0) };
        let r = ac.integrate(f, &[0.0, 0.0], &[2.0, 3.0]).expect("ok");
        // ∫ 3 over [0,2]×[0,3] = 18.
        assert!((r.value - 18.0).abs() < 1.0e-9, "value={}", r.value);
        assert!(r.converged);
    }

    #[test]
    fn linear_integral_3d() {
        // ∫_{[0,1]³} (x + y + z) = 3/2.
        let ac = AdaptiveCubature::new(1.0e-8, 1.0e-8, 50_000).expect("cfg");
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0] + x[1] + x[2]) };
        let r = ac
            .integrate(f, &[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0])
            .expect("ok");
        assert!((r.value - 1.5).abs() < 1.0e-7, "value={}", r.value);
    }

    #[test]
    fn polynomial_exact_within_basic_rule() {
        // Degree-7 rule integrates total-degree ≤ 7 polynomials exactly, so even a
        // single region must nail ∫_{[0,1]²} x²y² = 1/9.
        let ac = AdaptiveCubature::new(1.0e-12, 0.0, 10_000).expect("cfg");
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(x[0] * x[0] * x[1] * x[1]) };
        let r = ac.integrate(f, &[0.0, 0.0], &[1.0, 1.0]).expect("ok");
        assert!((r.value - 1.0 / 9.0).abs() < 1.0e-10, "value={}", r.value);
    }

    #[test]
    fn gaussian_bump_2d() {
        // ∫_{[-3,3]²} exp(-(x²+y²)) dx dy ≈ π (1 - exp(-9))·... actually = (∫_{-3}^{3}
        // e^{-x²}dx)². ∫_{-3}^{3} e^{-x²} = √π·erf(3) ≈ 1.7724538509·0.9999779095.
        let ac = AdaptiveCubature::new(1.0e-9, 1.0e-9, 2_000_000).expect("cfg");
        let f = |x: &[f64]| -> NumericResult<f64> { Ok((-(x[0] * x[0] + x[1] * x[1])).exp()) };
        let r = ac.integrate(f, &[-3.0, -3.0], &[3.0, 3.0]).expect("ok");
        let one_d = std::f64::consts::PI.sqrt() * erf_approx(3.0);
        let expected = one_d * one_d;
        assert!(
            (r.value - expected).abs() < 1.0e-6,
            "value={}, expected={expected}, err_est={}",
            r.value,
            r.error
        );
    }

    #[test]
    fn error_estimate_bounds_true_error() {
        // For a smooth integrand the reported error estimate should be a genuine upper
        // bound on the realised error (it is a conservative estimator, not exact).
        let ac = AdaptiveCubature::new(1.0e-6, 1.0e-6, 500_000).expect("cfg");
        // ∫_{[0,1]²} sin(πx) sin(πy) = (2/π)² .
        let f = |x: &[f64]| -> NumericResult<f64> {
            use std::f64::consts::PI;
            Ok((PI * x[0]).sin() * (PI * x[1]).sin())
        };
        let r = ac.integrate(f, &[0.0, 0.0], &[1.0, 1.0]).expect("ok");
        let exact = (2.0 / std::f64::consts::PI).powi(2);
        let true_err = (r.value - exact).abs();
        assert!(
            true_err <= r.error.max(1.0e-6),
            "true_err={true_err:e} exceeds estimate {:e}",
            r.error
        );
    }

    #[test]
    fn budget_is_respected() {
        // A demanding integrand with a tight budget must return converged=false without
        // overrunning max_eval.
        let max_eval = 5_000usize;
        let ac = AdaptiveCubature::new(1.0e-14, 0.0, max_eval).expect("cfg");
        let f = |x: &[f64]| -> NumericResult<f64> {
            // a sharp ridge that is hard to integrate to 1e-14
            Ok((1.0 / (1.0e-3 + (x[0] - 0.5).powi(2) + (x[1] - 0.5).powi(2))).sqrt())
        };
        let r = ac.integrate(f, &[0.0, 0.0], &[1.0, 1.0]).expect("ok");
        assert!(r.evaluations <= max_eval, "evals={}", r.evaluations);
        assert!(!r.converged);
    }

    #[test]
    fn relative_tolerance_path() {
        // Large-magnitude integrand: only rel_tol is set, abs_tol = 0.
        let ac = AdaptiveCubature::new(0.0, 1.0e-8, 200_000).expect("cfg");
        let f = |x: &[f64]| -> NumericResult<f64> { Ok(1.0e6 * (x[0] + x[1])) };
        let r = ac.integrate(f, &[0.0, 0.0], &[1.0, 1.0]).expect("ok");
        // ∫ 1e6 (x+y) over unit square = 1e6.
        assert!(
            (r.value - 1.0e6).abs() < 1.0e-8 * 1.0e6,
            "value={}",
            r.value
        );
    }

    #[test]
    fn random_separable_integrand_matches_product() {
        // ∫_{[0,1]²} (a + x)(b + y) dx dy = (a + 1/2)(b + 1/2). Random a, b.
        let mut rng = LcgRng::new(424_242);
        for _ in 0..5 {
            let a = rng.next_range(0.1, 2.0);
            let b = rng.next_range(0.1, 2.0);
            let ac = AdaptiveCubature::new(1.0e-9, 1.0e-9, 100_000).expect("cfg");
            let f = move |x: &[f64]| -> NumericResult<f64> { Ok((a + x[0]) * (b + x[1])) };
            let r = ac.integrate(f, &[0.0, 0.0], &[1.0, 1.0]).expect("ok");
            let exact = (a + 0.5) * (b + 0.5);
            assert!(
                (r.value - exact).abs() < 1.0e-7,
                "a={a}, b={b}: value={}, exact={exact}",
                r.value
            );
        }
    }

    #[test]
    fn config_validation() {
        assert!(AdaptiveCubature::new(0.0, 0.0, 100).is_err());
        assert!(AdaptiveCubature::new(-1.0, 1.0e-6, 100).is_err());
        assert!(AdaptiveCubature::new(1.0e-6, -1.0, 100).is_err());
        assert!(AdaptiveCubature::new(1.0e-6, 0.0, 0).is_err());
        assert!(AdaptiveCubature::new(1.0e-6, 1.0e-6, 100).is_ok());
    }

    #[test]
    fn rejects_bad_domain() {
        let ac = AdaptiveCubature::new(1.0e-6, 0.0, 100).expect("cfg");
        // A capture-free closure is `Copy`, so it can be passed by value repeatedly.
        let f = |_x: &[f64]| -> NumericResult<f64> { Ok(1.0) };
        assert!(ac.integrate(f, &[0.0], &[1.0, 2.0]).is_err()); // dim mismatch
        assert!(ac.integrate(f, &[], &[]).is_err()); // empty
        assert!(ac.integrate(f, &[1.0], &[0.0]).is_err()); // lo > hi
        assert!(ac.integrate(f, &[f64::NAN], &[1.0]).is_err()); // non-finite
    }

    /// A small, self-contained `erf` for the Gaussian test (Abramowitz–Stegun 7.1.26).
    fn erf_approx(x: f64) -> f64 {
        let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
        let y = 1.0
            - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736)
                * t
                + 0.254_829_592)
                * t
                * (-x * x).exp();
        if x >= 0.0 { y } else { -y }
    }
}
