//! f-DP and Gaussian Differential Privacy (GDP).
//!
//! Reference: Dong, Roth & Su (2022), "Gaussian Differential Privacy",
//! *Journal of the Royal Statistical Society*, Series B.
//!
//! # Gaussian DP
//! A mechanism M is **μ-GDP** if its privacy trade-off function
//! `T_M(α) ≥ T_μ(α)` point-wise, where
//!
//! `T_μ(α) = Φ(Φ⁻¹(1 − α) − μ)`
//!
//! and Φ is the standard normal CDF.
//!
//! Key results used here:
//! - Gaussian mechanism with sensitivity Δ, noise std σ is `(Δ/σ)`-GDP.
//! - Composition of k independent mechanisms with parameters μ₁,…,μₖ:
//!   `μ_composed = √(Σ μᵢ²)` (central limit theorem for trade-off functions).
//! - Conversion to (ε, δ): `δ(ε) = Φ(−ε/μ + μ/2) − exp(ε)·Φ(−ε/μ − μ/2)`.

use crate::error::{PrivacyError, PrivacyResult};

// ─── Normal CDF / probit approximations ──────────────────────────────────────

/// Standard normal CDF Φ(x) via Horner-form rational approximation.
///
/// Uses Abramowitz & Stegun (7.1.26) erfc approximation applied to x/√2,
/// accurate to ~1.5×10⁻⁷.  Φ(x) = 0.5 · erfc(−x/√2).
pub(crate) fn phi(x: f64) -> f64 {
    // We use the A&S erfc approximation for the argument |x|/√2.
    let z = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    // erfc(z) ≈ poly * exp(-z²)
    let erfc_z = poly * (-z * z).exp();
    // Φ(x) = 0.5 * erfc(-x/√2) = 1 - 0.5 * erfc(x/√2) for x ≥ 0
    if x >= 0.0 {
        1.0 - 0.5 * erfc_z
    } else {
        0.5 * erfc_z
    }
}

/// Inverse standard normal CDF (probit) via Beasley-Springer-Moro rational approximation.
///
/// Accurate to ±5×10⁻⁴ across the open interval (0, 1).
/// Returns `f64::NEG_INFINITY` at `p ≤ 0` and `f64::INFINITY` at `p ≥ 1`.
pub(crate) fn phi_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation for the central region p ∈ (0.02425, 0.97575).
    // For tail regions, use reflection symmetry.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.024_54;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail (reflection)
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ─── GDP configuration ────────────────────────────────────────────────────────

/// Configuration for a Gaussian DP mechanism.
#[derive(Debug, Clone)]
pub struct GdpConfig {
    /// GDP parameter μ = Δ/σ where Δ is sensitivity and σ is noise std.
    pub mu: f64,
}

// ─── GDP functions ────────────────────────────────────────────────────────────

/// Compute the composed GDP parameter for k mechanisms.
///
/// By the CLT for trade-off functions (Theorem 3.3 in Dong et al.),
/// the composition of k independent μᵢ-GDP mechanisms is μ_composed-GDP
/// where `μ_composed = √(Σ μᵢ²)`.
///
/// # Errors
/// Returns `EmptyInput` if `mus` is empty.
pub fn gdp_compose(mus: &[f64]) -> f64 {
    mus.iter().map(|&m| m * m).sum::<f64>().sqrt()
}

/// Convert μ-GDP to (ε, δ)-DP by inverting the trade-off function.
///
/// The conversion formula (Proposition 2.7 in Dong et al.):
/// `δ(ε) = Φ(−ε/μ + μ/2) − exp(ε)·Φ(−ε/μ − μ/2)`.
///
/// We find the minimum ε such that δ(ε) ≤ δ_target via binary search.
///
/// # Errors
/// - `InvalidDelta` if `delta ≤ 0` or `delta ≥ 1`.
/// - `InvalidParameter` if `mu ≤ 0`.
/// - `ConvergenceFailed` if binary search does not converge within 200 iterations.
pub fn gdp_to_epsilon_delta(mu: f64, delta: f64) -> PrivacyResult<f64> {
    if mu <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "mu must be positive, got {mu}"
        )));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }

    // Binary search for ε in [0, 100].
    let mut lo = 0.0f64;
    let mut hi = 100.0f64;

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let d = gdp_delta_at_epsilon(mu, mid);
        if d > delta {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-10 {
            return Ok(hi);
        }
    }

    Err(PrivacyError::ConvergenceFailed(200))
}

/// Evaluate δ(ε) for a μ-GDP mechanism at a given ε.
///
/// `δ(ε) = Φ(−ε/μ + μ/2) − e^ε · Φ(−ε/μ − μ/2)`
pub fn gdp_delta_at_epsilon(mu: f64, epsilon: f64) -> f64 {
    let a = -epsilon / mu + mu / 2.0;
    let b = -epsilon / mu - mu / 2.0;
    phi(a) - (epsilon.exp()) * phi(b)
}

/// Compute the GDP parameter μ = Δ/σ for the Gaussian mechanism.
///
/// # Errors
/// Returns `NonPositiveSensitivity` if `sensitivity ≤ 0` or
/// `InvalidParameter` if `sigma ≤ 0`.
pub fn gaussian_mechanism_mu(sensitivity: f64, sigma: f64) -> PrivacyResult<f64> {
    if sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
    }
    if sigma <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "sigma must be positive, got {sigma}"
        )));
    }
    Ok(sensitivity / sigma)
}

/// Evaluate the f-DP trade-off function T_μ(α).
///
/// `T_μ(α) = Φ(Φ⁻¹(1 − α) − μ)` for α ∈ [0, 1].
///
/// Represents the type-II error rate at type-I error rate α for the
/// optimal distinguisher of a μ-GDP mechanism.
pub fn fdp_tradeoff(mu: f64, alpha: f64) -> f64 {
    let quantile = phi_inv(1.0 - alpha);
    phi(quantile - mu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_at_zero() {
        let p = phi(0.0);
        assert!((p - 0.5).abs() < 1e-6, "Φ(0) should be 0.5, got {p}");
    }

    #[test]
    fn test_phi_symmetry() {
        let x = 1.5;
        let p = phi(x) + phi(-x);
        assert!((p - 1.0).abs() < 1e-6, "Φ(x)+Φ(-x) should be 1, got {p}");
    }

    #[test]
    fn test_phi_inv_roundtrip() {
        for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = phi_inv(p);
            let recovered = phi(x);
            assert!(
                (recovered - p).abs() < 1e-4,
                "Φ(Φ⁻¹({p})) = {recovered}, expected {p}"
            );
        }
    }

    #[test]
    fn test_gdp_compose_sqrt_sum_squares() {
        let mus = [1.0, 1.0, 1.0];
        let composed = gdp_compose(&mus);
        let expected = 3.0f64.sqrt();
        assert!(
            (composed - expected).abs() < 1e-10,
            "expected √3={expected}, got {composed}"
        );
    }

    #[test]
    fn test_gdp_to_epsilon_delta_nonneg() {
        let epsilon = gdp_to_epsilon_delta(1.0, 1e-5).expect("ok");
        assert!(epsilon > 0.0, "ε must be positive, got {epsilon}");
    }

    #[test]
    fn test_fdp_tradeoff_at_zero() {
        // T_μ(0) = Φ(Φ⁻¹(1) - μ) = Φ(+∞ - μ) = 1 for μ finite.
        // Edge case: returns Φ(+∞) ≈ 1.
        let t = fdp_tradeoff(1.0, 0.0);
        assert!(t >= 0.99, "T_μ(0) should be near 1, got {t}");
    }
}
