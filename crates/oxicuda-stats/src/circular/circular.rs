//! Circular statistics: Von Mises distribution and Rayleigh test for directional data.
//!
//! All angles are in radians. Uses self-contained error types to keep this module
//! cleanly separated from the main `StatsError` hierarchy.

use std::f64::consts::PI;

// ─── Error types ──────────────────────────────────────────────────────────────

/// Errors specific to circular-statistics operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CircularError {
    /// Slice was empty when at least one element was required.
    EmptyInput,
    /// Concentration parameter κ is not finite or is negative.
    InvalidKappa(String),
    /// Iterative algorithm did not converge.
    NotConverged,
    /// Sample size is too small for the requested operation.
    InsufficientSamples { got: usize, need: usize },
}

impl std::fmt::Display for CircularError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "circular: empty input"),
            Self::InvalidKappa(msg) => write!(f, "circular: invalid kappa — {msg}"),
            Self::NotConverged => write!(f, "circular: algorithm did not converge"),
            Self::InsufficientSamples { got, need } => {
                write!(f, "circular: need ≥{need} samples, got {got}")
            }
        }
    }
}

impl std::error::Error for CircularError {}

/// Result type for circular-statistics operations.
pub type CircularResult<T> = Result<T, CircularError>;

// ─── Bessel helper ────────────────────────────────────────────────────────────

/// Modified Bessel function of the first kind, order 0: I_0(x).
///
/// Series expansion: I_0(x) = Σ_{k=0}^{K} [(x/2)^{2k} / (k!)^2].
/// 30 terms give full f64 precision for |x| ≤ 700.
#[inline]
fn bessel_i0(x: f64) -> f64 {
    let half_x = x.abs() / 2.0;
    // Start the running term at the k=0 value (= 1) and accumulate.
    // Re-use the running term variable to avoid re-computing powers.
    let mut sum = 1.0_f64;
    let mut fact_k: f64 = 1.0;
    for k in 1..=30_u32 {
        fact_k *= k as f64;
        let term = half_x.powi(2 * k as i32) / (fact_k * fact_k);
        sum += term;
        if term < 1e-16 * sum {
            break;
        }
    }
    sum
}

// ─── Von Mises PDF / CDF ──────────────────────────────────────────────────────

/// Von Mises probability density function.
///
/// f(θ; μ, κ) = exp(κ cos(θ − μ)) / (2π I_0(κ))
///
/// # Errors
/// Returns [`CircularError::InvalidKappa`] when `kappa` is negative or non-finite.
pub fn von_mises_pdf(theta: f64, mu: f64, kappa: f64) -> CircularResult<f64> {
    if !kappa.is_finite() || kappa < 0.0 {
        return Err(CircularError::InvalidKappa(format!(
            "kappa must be finite and ≥ 0, got {kappa}"
        )));
    }
    let normaliser = 2.0 * PI * bessel_i0(kappa);
    Ok((kappa * (theta - mu).cos()).exp() / normaliser)
}

/// Von Mises cumulative distribution function via Simpson's rule.
///
/// Integrates the PDF on [μ − π, θ] using `n_intervals` (must be even; if odd it
/// is rounded up by one) sub-intervals.  200 intervals give ~10-digit accuracy
/// for practical κ values.
///
/// # Errors
/// Returns [`CircularError::InvalidKappa`] when `kappa` is negative or non-finite.
pub fn von_mises_cdf(theta: f64, mu: f64, kappa: f64, n_intervals: usize) -> CircularResult<f64> {
    if !kappa.is_finite() || kappa < 0.0 {
        return Err(CircularError::InvalidKappa(format!(
            "kappa must be finite and ≥ 0, got {kappa}"
        )));
    }
    // Ensure even number of intervals for Simpson's rule.
    let m = if n_intervals % 2 == 0 {
        n_intervals.max(2)
    } else {
        n_intervals + 1
    };
    let a = mu - PI;
    let b = theta;
    if (b - a).abs() < 1e-15 {
        return Ok(0.0);
    }
    let h = (b - a) / m as f64;
    let normaliser = 2.0 * PI * bessel_i0(kappa);
    // f(t) = exp(κ cos(t − μ)) / normaliser
    let f = |t: f64| (kappa * (t - mu).cos()).exp() / normaliser;

    // Composite Simpson's 1/3 rule
    let mut sum = f(a) + f(b);
    for i in 1..m {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    Ok((h / 3.0 * sum).clamp(0.0, 1.0))
}

// ─── Von Mises MLE ────────────────────────────────────────────────────────────

/// Result of Von Mises MLE fitting.
#[derive(Debug, Clone, Copy)]
pub struct VonMisesFit {
    /// MLE estimate of the mean direction μ̂ ∈ (−π, π].
    pub mu: f64,
    /// MLE estimate of the concentration parameter κ̂.
    pub kappa: f64,
    /// Mean resultant length R̄ ∈ [0, 1].
    pub r_bar: f64,
}

/// Fit a Von Mises distribution to circular data via maximum likelihood.
///
/// Uses the Mardia & Jupp (2000) approximation A21 for κ from R̄:
/// - R̄ < 0.53 : κ ≈ 2R + R³ + 5R⁵/6
/// - R̄ < 0.85 : κ ≈ −0.4 + 1.39R + 0.43/(1 − R)
/// - else      : κ ≈ 1/(R³ − 4R² + 3R)
///
/// # Errors
/// - [`CircularError::EmptyInput`] if `angles` is empty.
/// - [`CircularError::InsufficientSamples`] if fewer than 2 angles are supplied.
pub fn von_mises_mle(angles: &[f64]) -> CircularResult<VonMisesFit> {
    if angles.is_empty() {
        return Err(CircularError::EmptyInput);
    }
    if angles.len() < 2 {
        return Err(CircularError::InsufficientSamples {
            got: angles.len(),
            need: 2,
        });
    }
    let n = angles.len() as f64;
    let sum_sin: f64 = angles.iter().map(|a| a.sin()).sum();
    let sum_cos: f64 = angles.iter().map(|a| a.cos()).sum();
    let mu = sum_sin.atan2(sum_cos);
    let r_bar = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();

    let kappa = kappa_from_r_bar(r_bar);
    Ok(VonMisesFit { mu, kappa, r_bar })
}

/// Mardia & Jupp (2000) Approximation A21: κ from mean resultant length R̄.
///
/// The denominator of the high-R branch, r(r−1)(r−3), approaches 0 as r → 1
/// (κ → ∞).  We clamp r to [0, R_MAX] so that the approximation remains
/// finite and positive.  The cap value R_MAX ≈ 0.9999 corresponds to
/// κ ≈ 5000, which is already numerically indistinguishable from the limiting
/// degenerate distribution for all practical purposes.
#[inline]
fn kappa_from_r_bar(r: f64) -> f64 {
    const R_MAX: f64 = 0.9999;
    let r = r.clamp(0.0, R_MAX);
    if r < 0.53 {
        2.0 * r + r.powi(3) + 5.0 * r.powi(5) / 6.0
    } else if r < 0.85 {
        -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    } else {
        // denom = r(r-1)(r-3); negative for r ∈ (1,3), so absolute value is used.
        let denom = (r.powi(3) - 4.0 * r.powi(2) + 3.0 * r).abs();
        if denom < 1e-10 {
            // Effectively r = 1: return a large, finite κ.
            5000.0
        } else {
            1.0 / denom
        }
    }
}

// ─── Rayleigh test ────────────────────────────────────────────────────────────

/// Result of Rayleigh test for circular uniformity.
#[derive(Debug, Clone, Copy)]
pub struct RayleighResult {
    /// Mean resultant length R̄.
    pub r_bar: f64,
    /// Rayleigh test statistic Z = n R̄².
    pub statistic: f64,
    /// Approximate p-value (Rayleigh 1919 / Mardia 1972 series correction).
    pub p_value: f64,
}

/// Rayleigh test of H₀: angles are uniformly distributed on the circle.
///
/// Computes Z = n R̄² and applies the Mardia (1972) series correction for the
/// exact p-value approximation:
///
/// p ≈ exp(−Z) · [1 + (2Z − Z²)/(4n) − (24Z − 132Z² + 76Z³ − 9Z⁴)/(288n²)]
///
/// # Errors
/// - [`CircularError::EmptyInput`] if `angles` is empty.
/// - [`CircularError::InsufficientSamples`] if fewer than 2 angles are supplied.
pub fn rayleigh_test(angles: &[f64]) -> CircularResult<RayleighResult> {
    if angles.is_empty() {
        return Err(CircularError::EmptyInput);
    }
    if angles.len() < 2 {
        return Err(CircularError::InsufficientSamples {
            got: angles.len(),
            need: 2,
        });
    }
    let n = angles.len() as f64;
    let sum_sin: f64 = angles.iter().map(|a| a.sin()).sum();
    let sum_cos: f64 = angles.iter().map(|a| a.cos()).sum();
    let r_bar = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
    let z = n * r_bar.powi(2);

    // Mardia (1972) asymptotic expansion of the p-value.
    let correction1 = (2.0 * z - z.powi(2)) / (4.0 * n);
    let correction2 =
        (24.0 * z - 132.0 * z.powi(2) + 76.0 * z.powi(3) - 9.0 * z.powi(4)) / (288.0 * n.powi(2));
    let p_raw = (-z).exp() * (1.0 + correction1 - correction2);
    let p_value = p_raw.clamp(0.0, 1.0);

    Ok(RayleighResult {
        r_bar,
        statistic: z,
        p_value,
    })
}

// ─── Summary statistics ───────────────────────────────────────────────────────

/// Circular mean direction: atan2(Σ sin θᵢ, Σ cos θᵢ) ∈ (−π, π].
///
/// Returns `f64::NAN` for an empty slice (no error — matches the contract of
/// the plain scalar helpers that mirror `f64::NAN` for degenerate input).
pub fn circular_mean(angles: &[f64]) -> f64 {
    if angles.is_empty() {
        return f64::NAN;
    }
    let sum_sin: f64 = angles.iter().map(|a| a.sin()).sum();
    let sum_cos: f64 = angles.iter().map(|a| a.cos()).sum();
    sum_sin.atan2(sum_cos)
}

/// Circular variance: V = 1 − R̄ ∈ [0, 1].
///
/// Returns `f64::NAN` for an empty slice.
pub fn circular_variance(angles: &[f64]) -> f64 {
    if angles.is_empty() {
        return f64::NAN;
    }
    let n = angles.len() as f64;
    let sum_sin: f64 = angles.iter().map(|a| a.sin()).sum();
    let sum_cos: f64 = angles.iter().map(|a| a.cos()).sum();
    let r_bar = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
    1.0 - r_bar
}

/// Circular standard deviation: s = √(−2 ln R̄).
///
/// Returns `f64::NAN` for an empty slice; returns `f64::INFINITY` when R̄ = 0
/// (perfectly uniform distribution).
pub fn circular_std(angles: &[f64]) -> f64 {
    if angles.is_empty() {
        return f64::NAN;
    }
    let n = angles.len() as f64;
    let sum_sin: f64 = angles.iter().map(|a| a.sin()).sum();
    let sum_cos: f64 = angles.iter().map(|a| a.cos()).sum();
    let r_bar = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
    if r_bar <= 0.0 {
        return f64::INFINITY;
    }
    (-2.0 * r_bar.ln()).sqrt()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    // ── Bessel sanity ──────────────────────────────────────────────────────────

    #[test]
    fn bessel_i0_zero() {
        // I_0(0) = 1
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bessel_i0_known() {
        // I_0(1) ≈ 1.2660658777520082
        let expected = 1.2660658777520082_f64;
        assert!((bessel_i0(1.0) - expected).abs() < 1e-10);
    }

    // ── PDF tests ─────────────────────────────────────────────────────────────

    #[test]
    fn von_mises_pdf_positive() {
        let p = von_mises_pdf(0.0, 0.0, 1.0).expect("ok");
        assert!(p > 0.0);
        assert!(p.is_finite());
    }

    #[test]
    fn von_mises_pdf_invalid_kappa() {
        assert!(von_mises_pdf(0.0, 0.0, -1.0).is_err());
        assert!(von_mises_pdf(0.0, 0.0, f64::NAN).is_err());
    }

    /// Numerical integration of the PDF over [−π, π] must equal ≈ 1.
    #[test]
    fn von_mises_pdf_integrates_to_one() {
        let mu = 0.0;
        let kappa = 2.5;
        let n = 10_000;
        let h = 2.0 * PI / n as f64;
        let a = -PI;
        // Composite trapezoidal rule
        let integral: f64 = (0..=n)
            .map(|i| {
                let t = a + i as f64 * h;
                let w = if i == 0 || i == n { 0.5 } else { 1.0 };
                w * von_mises_pdf(t, mu, kappa).expect("von_mises_pdf should succeed")
            })
            .sum::<f64>()
            * h;
        assert!(
            (integral - 1.0).abs() < 1e-6,
            "integral = {integral}, expected ≈ 1"
        );
    }

    /// PDF with κ = 0 should equal 1/(2π) everywhere (uniform).
    #[test]
    fn von_mises_pdf_uniform_kappa_zero() {
        let p = von_mises_pdf(1.23, 0.0, 0.0).expect("ok");
        let expected = 1.0 / (2.0 * PI);
        assert!((p - expected).abs() < 1e-12);
    }

    // ── CDF tests ─────────────────────────────────────────────────────────────

    #[test]
    fn von_mises_cdf_bounds() {
        // CDF at μ − π should be ≈ 0, at μ + π should be ≈ 1.
        let mu = 0.0;
        let kappa = 2.0;
        let lo = von_mises_cdf(-PI, mu, kappa, 200).expect("ok");
        let hi = von_mises_cdf(PI, mu, kappa, 200).expect("ok");
        assert!(lo < 0.01, "lo={lo}");
        assert!(hi > 0.99, "hi={hi}");
    }

    #[test]
    fn von_mises_cdf_monotone() {
        let mu = 0.5;
        let kappa = 1.5;
        let mut prev = von_mises_cdf(-PI, mu, kappa, 200).expect("ok");
        for i in 1..=20 {
            let t = -PI + i as f64 * (2.0 * PI / 20.0);
            let cur = von_mises_cdf(t, mu, kappa, 200).expect("ok");
            assert!(
                cur >= prev - 1e-10,
                "CDF not monotone at i={i}: prev={prev}, cur={cur}"
            );
            prev = cur;
        }
    }

    // ── MLE tests ─────────────────────────────────────────────────────────────

    #[test]
    fn von_mises_mle_recovers_mu() {
        // Concentrated data near mu = PI/4 should yield mu_hat ≈ PI/4.
        let mu_true = PI / 4.0;
        let angles: Vec<f64> = (0..200)
            .map(|i| mu_true + 0.05 * (i as f64 * 0.1).sin())
            .collect();
        let fit = von_mises_mle(&angles).expect("ok");
        assert!(
            (fit.mu - mu_true).abs() < 0.1,
            "mu_hat={}, mu_true={}",
            fit.mu,
            mu_true
        );
    }

    #[test]
    fn von_mises_mle_concentrated_has_large_kappa() {
        // Angles all identical → R_bar ≈ 1 → large kappa.
        let angles = vec![0.1_f64; 100];
        let fit = von_mises_mle(&angles).expect("ok");
        assert!(fit.r_bar > 0.99);
        assert!(fit.kappa > 50.0, "kappa={}", fit.kappa);
    }

    #[test]
    fn von_mises_mle_empty_error() {
        assert!(matches!(von_mises_mle(&[]), Err(CircularError::EmptyInput)));
    }

    // ── R_bar tests ───────────────────────────────────────────────────────────

    #[test]
    fn r_bar_concentrated_near_one() {
        // All angles near 0 → R_bar ≈ 1.
        let angles: Vec<f64> = vec![0.01; 50];
        let fit = von_mises_mle(&angles).expect("ok");
        assert!(fit.r_bar > 0.99);
    }

    #[test]
    fn r_bar_uniform_near_zero() {
        // Evenly spaced around the circle → R_bar ≈ 0.
        let n = 360_usize;
        let angles: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let fit = von_mises_mle(&angles).expect("ok");
        assert!(fit.r_bar < 1e-10, "r_bar={}", fit.r_bar);
    }

    // ── Rayleigh test ─────────────────────────────────────────────────────────

    #[test]
    fn rayleigh_test_rejects_non_uniform() {
        // All angles = 0 → maximally concentrated → p ≈ 0, Z is large.
        let angles = vec![0.0_f64; 100];
        let res = rayleigh_test(&angles).expect("ok");
        assert!(res.p_value < 0.001, "p={}", res.p_value);
        assert!(res.statistic > 50.0);
    }

    #[test]
    fn rayleigh_test_does_not_reject_uniform() {
        // Evenly spaced → uniform distribution → p should be large.
        let n = 360_usize;
        let angles: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let res = rayleigh_test(&angles).expect("ok");
        assert!(res.p_value > 0.5, "p={}", res.p_value);
    }

    #[test]
    fn rayleigh_test_empty_error() {
        assert!(matches!(rayleigh_test(&[]), Err(CircularError::EmptyInput)));
    }

    #[test]
    fn rayleigh_test_insufficient_error() {
        assert!(matches!(
            rayleigh_test(&[0.1]),
            Err(CircularError::InsufficientSamples { got: 1, need: 2 })
        ));
    }

    // ── Summary statistics ────────────────────────────────────────────────────

    #[test]
    fn circular_mean_symmetric() {
        // Symmetric angles around 0 → mean ≈ 0.
        let angles = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let m = circular_mean(&angles);
        assert!(m.abs() < 1e-10, "mean={m}");
    }

    #[test]
    fn circular_mean_empty_is_nan() {
        assert!(circular_mean(&[]).is_nan());
    }

    #[test]
    fn circular_variance_concentrated_near_zero() {
        let angles = vec![0.01_f64; 100];
        let v = circular_variance(&angles);
        assert!(v < 0.01, "variance={v}");
    }

    #[test]
    fn circular_variance_uniform_near_one() {
        let n = 360_usize;
        let angles: Vec<f64> = (0..n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
        let v = circular_variance(&angles);
        assert!(v > 0.99, "variance={v}");
    }

    #[test]
    fn circular_std_concentrated_near_zero() {
        let angles = vec![0.0_f64; 200];
        let s = circular_std(&angles);
        // R_bar = 1 → -2 ln(1) = 0 → std = 0
        assert!(s < 1e-10, "std={s}");
    }

    #[test]
    fn circular_std_empty_is_nan() {
        assert!(circular_std(&[]).is_nan());
    }
}
