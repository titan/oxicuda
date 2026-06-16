//! Rényi Differential Privacy (RDP) accountant for the Laplace mechanism.
//!
//! Reference: Mironov (2017), "Rényi Differential Privacy", IEEE CSF,
//! Proposition 6 / Table II.
//!
//! # RDP bound for Laplace
//! The Laplace mechanism with noise scale `b` and L1-sensitivity `Δ` satisfies,
//! for a Rényi order `α > 1`, the exact Rényi divergence
//!
//! `ε_R(α) = (1/(α−1)) · ln( (α/(2α−1)) · e^{(α−1)·t} + ((α−1)/(2α−1)) · e^{−α·t} )`,
//!
//! where `t = Δ / b` is the pure-DP parameter `ε₀` of the Laplace mechanism
//! (so `b = Δ / ε₀`).  The bound interpolates between a KL bound as `α → 1` and
//! the pure-DP value `ε₀` as `α → ∞` (where `ε_R(α) → t`).
//!
//! # Composition
//! At a *fixed* order `α`, RDP composes additively: `ε_R^total(α) = Σ ε_R^i(α)`.
//!
//! # Conversion to (ε, δ)-DP
//! Mironov's 2017 conversion: an `(α, ε_R)`-RDP guarantee implies
//! `(ε, δ)`-DP with
//!
//! `ε = ε_R + ln(1/δ) / (α − 1)`.
//!
//! Optimising over a grid of orders `α` yields the tightest `ε` for a target
//! `δ`.

use crate::error::{PrivacyError, PrivacyResult};

/// Configuration for the Laplace-mechanism RDP accountant.
#[derive(Debug, Clone)]
pub struct RdpLaplaceConfig {
    /// L1 sensitivity `Δ > 0`.
    pub sensitivity: f64,
    /// Noise scale `b > 0`.
    pub scale: f64,
}

impl RdpLaplaceConfig {
    /// Construct and validate an `RdpLaplaceConfig`.
    ///
    /// # Errors
    /// - `NonPositiveSensitivity` if `sensitivity ≤ 0`.
    /// - `InvalidParameter` if `scale ≤ 0`.
    pub fn new(sensitivity: f64, scale: f64) -> PrivacyResult<Self> {
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        if scale <= 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "scale must be positive, got {scale}"
            )));
        }
        Ok(Self { sensitivity, scale })
    }
}

/// Compute the RDP-ε of the Laplace mechanism at order `alpha`.
///
/// Uses `t = sensitivity / scale` and the closed form of Mironov (2017,
/// Prop 6).  The two exponentials are combined via a factored log-sum-exp:
/// since `(α−1)·t > −α·t` for `t > 0, α > 1`, the larger exponent
/// `(α−1)·t` is factored out for numerical stability.
///
/// # Errors
/// - `InvalidParameter` if `alpha ≤ 1`.
pub fn rdp_epsilon(cfg: &RdpLaplaceConfig, alpha: f64) -> PrivacyResult<f64> {
    if alpha <= 1.0 || alpha.is_nan() {
        return Err(PrivacyError::InvalidParameter(format!(
            "alpha must be > 1, got {alpha}"
        )));
    }
    let t = cfg.sensitivity / cfg.scale;
    let a = alpha;
    let denom = 2.0 * a - 1.0;
    let coef_hi = a / denom; // weight on e^{(α−1)t}
    let coef_lo = (a - 1.0) / denom; // weight on e^{−αt}

    // Factor out the larger exponent (α−1)·t:
    //   ε_R = (1/(α−1)) · [ (α−1)·t + ln( coef_hi + coef_lo · e^{−(2α−1)·t} ) ].
    let inner = coef_hi + coef_lo * (-denom * t).exp();
    let log_term = (a - 1.0) * t + inner.ln();
    Ok(log_term / (a - 1.0))
}

/// Compose RDP guarantees at a *fixed* order `α`: returns `Σ εᵢ`.
///
/// All entries must correspond to the **same** Rényi order `α`; RDP
/// composition is exact and additive there.
#[must_use]
pub fn rdp_compose(epsilons: &[f64]) -> f64 {
    epsilons.iter().sum()
}

/// Convert an `(α, ε_R)`-RDP guarantee to `(ε, δ)`-DP (Mironov 2017).
///
/// `ε = ε_R + ln(1/δ) / (α − 1)`.
///
/// # Errors
/// - `InvalidParameter` if `alpha ≤ 1`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
pub fn rdp_to_epsilon_delta(rdp_eps: f64, alpha: f64, delta: f64) -> PrivacyResult<f64> {
    if alpha <= 1.0 || alpha.is_nan() {
        return Err(PrivacyError::InvalidParameter(format!(
            "alpha must be > 1, got {alpha}"
        )));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    Ok(rdp_eps + (1.0 / delta).ln() / (alpha - 1.0))
}

/// Given `(α, ε_R(α))` pairs, return the minimum converted `(ε, δ)`-DP value.
///
/// Each pair is converted via [`rdp_to_epsilon_delta`] and the smallest
/// resulting `ε` is returned — the tightest guarantee over the order grid.
///
/// # Errors
/// - `EmptyMechanismList` if the pair list is empty.
/// - `InvalidParameter` / `InvalidDelta` if any conversion has invalid inputs.
pub fn optimal_epsilon(
    _cfg: &RdpLaplaceConfig,
    rdp_eps_at_alphas: &[(f64, f64)],
    delta: f64,
) -> PrivacyResult<f64> {
    if rdp_eps_at_alphas.is_empty() {
        return Err(PrivacyError::EmptyMechanismList);
    }
    let mut best = f64::INFINITY;
    for &(alpha, rdp_eps) in rdp_eps_at_alphas {
        let eps = rdp_to_epsilon_delta(rdp_eps, alpha, delta)?;
        if eps < best {
            best = eps;
        }
    }
    Ok(best)
}

/// Map a grid of orders to `(α, ε_R(α))` pairs for the configured mechanism.
///
/// # Errors
/// - `InvalidParameter` if any `α ≤ 1`.
pub fn rdp_curve(cfg: &RdpLaplaceConfig, alphas: &[f64]) -> PrivacyResult<Vec<(f64, f64)>> {
    let mut out = Vec::with_capacity(alphas.len());
    for &alpha in alphas {
        let eps = rdp_epsilon(cfg, alpha)?;
        out.push((alpha, eps));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        assert!(RdpLaplaceConfig::new(0.0, 1.0).is_err());
        assert!(RdpLaplaceConfig::new(-1.0, 1.0).is_err());
        assert!(RdpLaplaceConfig::new(1.0, 0.0).is_err());
        assert!(RdpLaplaceConfig::new(1.0, -2.0).is_err());
        assert!(RdpLaplaceConfig::new(1.0, 2.0).is_ok());
    }

    #[test]
    fn test_rdp_epsilon_rejects_low_alpha() {
        let cfg = RdpLaplaceConfig::new(1.0, 1.0).expect("ok");
        assert!(rdp_epsilon(&cfg, 1.0).is_err());
        assert!(rdp_epsilon(&cfg, 0.5).is_err());
        assert!(rdp_epsilon(&cfg, 1.0001).is_ok());
    }

    #[test]
    fn test_rdp_epsilon_positive_and_increasing_in_t() {
        // Larger t (smaller scale) ⇒ larger ε_R at fixed α.
        let cfg_small_t = RdpLaplaceConfig::new(1.0, 4.0).expect("ok"); // t = 0.25
        let cfg_large_t = RdpLaplaceConfig::new(1.0, 1.0).expect("ok"); // t = 1.0
        let alpha = 8.0;
        let e_small = rdp_epsilon(&cfg_small_t, alpha).expect("ok");
        let e_large = rdp_epsilon(&cfg_large_t, alpha).expect("ok");
        assert!(e_small > 0.0, "ε_R must be positive, got {e_small}");
        assert!(e_large > 0.0, "ε_R must be positive, got {e_large}");
        assert!(
            e_large > e_small,
            "larger t should give larger ε_R: {e_large} > {e_small}"
        );
    }

    #[test]
    fn test_rdp_epsilon_approaches_t_at_large_alpha() {
        // ε_R(α) → t = Δ/b as α → ∞.
        let cfg = RdpLaplaceConfig::new(1.0, 2.0).expect("ok"); // t = 0.5
        let t = 0.5;
        let e = rdp_epsilon(&cfg, 200.0).expect("ok");
        assert!(
            (e - t).abs() < 0.05 * t.max(1.0),
            "ε_R(200)={e} should be near t={t}"
        );
    }

    #[test]
    fn test_rdp_epsilon_monotone_in_alpha() {
        // ε_R generally increases toward t as α grows.
        let cfg = RdpLaplaceConfig::new(1.0, 2.0).expect("ok"); // t = 0.5
        let alphas = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let mut prev = f64::NEG_INFINITY;
        for &a in &alphas {
            let e = rdp_epsilon(&cfg, a).expect("ok");
            assert!(
                e >= prev - 1e-9,
                "ε_R should be non-decreasing in α: {e} < {prev} at α={a}"
            );
            prev = e;
        }
    }

    #[test]
    fn test_rdp_compose_sums() {
        let eps = [0.1, 0.2, 0.3];
        let total = rdp_compose(&eps);
        assert!((total - 0.6).abs() < 1e-12, "expected 0.6, got {total}");
    }

    #[test]
    fn test_rdp_to_epsilon_delta_rejects_bad_inputs() {
        assert!(rdp_to_epsilon_delta(0.1, 1.0, 1e-5).is_err());
        assert!(rdp_to_epsilon_delta(0.1, 0.5, 1e-5).is_err());
        assert!(rdp_to_epsilon_delta(0.1, 2.0, 0.0).is_err());
        assert!(rdp_to_epsilon_delta(0.1, 2.0, 1.0).is_err());
        assert!(rdp_to_epsilon_delta(0.1, 2.0, 1e-5).is_ok());
    }

    #[test]
    fn test_rdp_to_epsilon_delta_adds_positive_slack() {
        let rdp_eps = 0.3;
        let eps = rdp_to_epsilon_delta(rdp_eps, 4.0, 1e-5).expect("ok");
        assert!(eps > rdp_eps, "converted ε={eps} must exceed ε_R={rdp_eps}");
    }

    #[test]
    fn test_rdp_to_epsilon_delta_slack_decreases_with_alpha() {
        // For fixed ε_R, the additive slack ln(1/δ)/(α−1) shrinks as α grows.
        let rdp_eps = 0.3;
        let delta = 1e-5;
        let eps_small_alpha = rdp_to_epsilon_delta(rdp_eps, 2.0, delta).expect("ok");
        let eps_large_alpha = rdp_to_epsilon_delta(rdp_eps, 64.0, delta).expect("ok");
        assert!(
            eps_large_alpha < eps_small_alpha,
            "slack should shrink with α: {eps_large_alpha} < {eps_small_alpha}"
        );
    }

    #[test]
    fn test_optimal_epsilon_is_min() {
        let cfg = RdpLaplaceConfig::new(1.0, 2.0).expect("ok");
        let delta = 1e-5;
        let alphas = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let curve = rdp_curve(&cfg, &alphas).expect("ok");
        let opt = optimal_epsilon(&cfg, &curve, delta).expect("ok");
        // Optimal must be ≤ every single conversion.
        for &(alpha, rdp_eps) in &curve {
            let single = rdp_to_epsilon_delta(rdp_eps, alpha, delta).expect("ok");
            assert!(
                opt <= single + 1e-12,
                "optimal {opt} must be ≤ single conversion {single}"
            );
        }
    }

    #[test]
    fn test_optimal_epsilon_rejects_empty() {
        let cfg = RdpLaplaceConfig::new(1.0, 2.0).expect("ok");
        assert!(optimal_epsilon(&cfg, &[], 1e-5).is_err());
    }

    #[test]
    fn test_rdp_curve_length_matches() {
        let cfg = RdpLaplaceConfig::new(1.0, 2.0).expect("ok");
        let alphas = [2.0, 3.0, 4.0, 5.0];
        let curve = rdp_curve(&cfg, &alphas).expect("ok");
        assert_eq!(curve.len(), alphas.len());
        for (i, &a) in alphas.iter().enumerate() {
            assert!((curve[i].0 - a).abs() < 1e-12);
        }
    }

    #[test]
    fn test_end_to_end_realistic() {
        // ε₀ = Δ/b = 0.5 (Δ=1, b=2). Sweep α, convert at δ=1e-5.
        let cfg = RdpLaplaceConfig::new(1.0, 2.0).expect("ok");
        let delta = 1e-5;
        let alphas = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let curve = rdp_curve(&cfg, &alphas).expect("ok");
        let opt = optimal_epsilon(&cfg, &curve, delta).expect("ok");
        assert!(opt.is_finite());
        assert!(
            opt > 0.0 && opt < 20.0,
            "optimal ε={opt} should be a reasonable positive finite number"
        );
    }
}
