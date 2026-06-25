//! zCDP ↔ RDP conversion, verified against the Gaussian mechanism.
//!
//! References:
//! - Bun & Steinke (2016), "Concentrated Differential Privacy: Simplifications,
//!   Extensions, and Lower Bounds", TCC.  Proposition 1.4 establishes the exact
//!   relationship between `ρ`-zCDP and the Rényi-divergence curve.
//! - Mironov (2017), "Rényi Differential Privacy", IEEE CSF.  Provides the
//!   `(ε, δ)` conversion of an RDP curve.
//!
//! # The zCDP ↔ RDP bridge
//! `ρ`-zCDP is *defined* by the linear Rényi curve
//!
//! ```text
//!     ε_R(α) = ρ · α        for all α ∈ (1, ∞).
//! ```
//!
//! Therefore zCDP and RDP are two views of the same object: a mechanism is
//! `ρ`-zCDP **iff** it is `(α, ρα)`-RDP for every order `α > 1`.  This module
//! makes the bridge explicit:
//!
//! - [`zcdp_to_rdp_curve`] samples the linear curve `ε_R(α) = ρα` on a grid.
//! - [`rdp_curve_to_zcdp`] inverts a *general* RDP curve to the tightest `ρ`
//!   such that `ε_R(α) ≤ ρα` over the grid, i.e. `ρ = maxₐ ε_R(α)/α`.
//! - [`zcdp_epsilon_via_rdp`] converts `ρ`-zCDP to `(ε, δ)`-DP by optimising the
//!   Mironov RDP→(ε, δ) bound `ε = ρα + ln(1/δ)/(α−1)` over `α`, which has the
//!   closed-form optimum `α* = 1 + √(ln(1/δ)/ρ)` giving
//!   `ε = ρ + 2√(ρ·ln(1/δ))` — recovering the Bun–Steinke Lemma 3.5 bound and
//!   thereby *verifying* the bridge numerically.
//!
//! The Gaussian mechanism with L2-sensitivity `Δ` and noise std `σ` has the
//! exact RDP curve `ε_R(α) = α·Δ²/(2σ²) = α·ρ` with `ρ = Δ²/(2σ²)`, so feeding
//! its curve through [`rdp_curve_to_zcdp`] recovers `ρ` to grid precision (tested).

use crate::error::{PrivacyError, PrivacyResult};

/// Default grid of Rényi orders for the zCDP ↔ RDP bridge.
///
/// Mirrors the order grid used by the Gaussian RDP accountant so the two
/// modules report consistent `(ε, δ)` values.
const DEFAULT_ORDERS: &[f64] = &[
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    20.0, 24.0, 28.0, 32.0, 48.0, 64.0,
];

/// Return the default Rényi-order grid used by this module.
#[must_use]
pub fn default_orders() -> Vec<f64> {
    DEFAULT_ORDERS.to_vec()
}

/// Sample the RDP curve `ε_R(α) = ρ·α` induced by a `ρ`-zCDP guarantee.
///
/// Returns `(α, ε_R(α))` pairs over `orders`.  Every order must satisfy
/// `α > 1`.
///
/// # Errors
/// - `InvalidParameter` if `rho < 0` or any order `α ≤ 1`.
/// - `EmptyInput` if `orders` is empty.
pub fn zcdp_to_rdp_curve(rho: f64, orders: &[f64]) -> PrivacyResult<Vec<(f64, f64)>> {
    if rho < 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "rho must be ≥ 0, got {rho}"
        )));
    }
    if orders.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let mut curve = Vec::with_capacity(orders.len());
    for &alpha in orders {
        if alpha <= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "Rényi order must be > 1, got {alpha}"
            )));
        }
        curve.push((alpha, rho * alpha));
    }
    Ok(curve)
}

/// Invert a general RDP curve to the tightest `ρ`-zCDP guarantee it implies.
///
/// Given `(α, ε_R(α))` pairs, a `ρ`-zCDP guarantee with
/// `ρ = maxₐ ε_R(α)/α` dominates the curve (`ρα ≥ ε_R(α)` at every grid order),
/// and is the smallest such `ρ` justified by the grid.  For an exactly-linear
/// curve (e.g. the Gaussian mechanism) every ratio equals `ρ`, so the result is
/// exact.
///
/// # Errors
/// - `EmptyInput` if `curve` is empty.
/// - `InvalidParameter` if any order `α ≤ 1` or any `ε_R(α) < 0`.
pub fn rdp_curve_to_zcdp(curve: &[(f64, f64)]) -> PrivacyResult<f64> {
    if curve.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let mut rho = 0.0f64;
    for &(alpha, rdp_eps) in curve {
        if alpha <= 1.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "Rényi order must be > 1, got {alpha}"
            )));
        }
        if rdp_eps < 0.0 {
            return Err(PrivacyError::InvalidParameter(format!(
                "RDP epsilon must be ≥ 0, got {rdp_eps}"
            )));
        }
        let ratio = rdp_eps / alpha;
        if ratio > rho {
            rho = ratio;
        }
    }
    Ok(rho)
}

/// Convert `ρ`-zCDP to `(ε, δ)`-DP by optimising the Mironov RDP→(ε, δ) bound.
///
/// For each grid order `α`, the RDP→DP conversion gives
/// `ε(α) = ρα + ln(1/δ)/(α−1)`; the reported `ε` minimises this over `orders`.
/// The grid is augmented with the analytic optimum
/// `α* = 1 + √(ln(1/δ)/ρ)` so the result matches the closed-form Bun–Steinke
/// bound `ε = ρ + 2√(ρ·ln(1/δ))` whenever `α*` lies above 1.
///
/// # Errors
/// - `InvalidParameter` if `rho ≤ 0`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
/// - `EmptyInput` if `orders` is empty.
pub fn zcdp_epsilon_via_rdp(rho: f64, delta: f64, orders: &[f64]) -> PrivacyResult<f64> {
    if rho <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "rho must be positive, got {rho}"
        )));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    if orders.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    let log_inv_delta = (1.0 / delta).ln();

    // Analytic optimum α* = 1 + √(ln(1/δ)/ρ).
    let alpha_star = 1.0 + (log_inv_delta / rho).sqrt();

    let mut best = f64::INFINITY;
    let eval = |alpha: f64, best: &mut f64| {
        if alpha > 1.0 {
            let eps = rho * alpha + log_inv_delta / (alpha - 1.0);
            if eps < *best {
                *best = eps;
            }
        }
    };
    for &alpha in orders {
        eval(alpha, &mut best);
    }
    eval(alpha_star, &mut best);

    if best.is_finite() {
        Ok(best)
    } else {
        Err(PrivacyError::ConvergenceFailed(orders.len()))
    }
}

/// Closed-form `(ε, δ)` bound for `ρ`-zCDP: `ε = ρ + 2√(ρ·ln(1/δ))`.
///
/// Provided for cross-checking [`zcdp_epsilon_via_rdp`] against the analytic
/// Bun–Steinke Lemma 3.5 value.
///
/// # Errors
/// - `InvalidParameter` if `rho ≤ 0`.
/// - `InvalidDelta` if `delta ∉ (0, 1)`.
pub fn zcdp_epsilon_closed_form(rho: f64, delta: f64) -> PrivacyResult<f64> {
    if rho <= 0.0 {
        return Err(PrivacyError::InvalidParameter(format!(
            "rho must be positive, got {rho}"
        )));
    }
    if !(delta > 0.0 && delta < 1.0) {
        return Err(PrivacyError::InvalidDelta(delta));
    }
    Ok(rho + 2.0 * (rho * (1.0 / delta).ln()).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn test_zcdp_to_rdp_curve_is_linear() {
        let rho = 0.3;
        let orders = default_orders();
        let curve = zcdp_to_rdp_curve(rho, &orders).expect("curve");
        for &(alpha, rdp_eps) in &curve {
            assert!(
                approx(rdp_eps, rho * alpha, 1e-12),
                "ε_R({alpha}) should equal ρα = {}, got {rdp_eps}",
                rho * alpha
            );
        }
    }

    #[test]
    fn test_gaussian_curve_recovers_rho() {
        // Gaussian mechanism RDP curve ε_R(α) = α·Δ²/(2σ²) = α·ρ.
        let sensitivity = 1.5;
        let sigma = 2.0;
        let rho_true = sensitivity * sensitivity / (2.0 * sigma * sigma);
        let orders = default_orders();
        let curve: Vec<(f64, f64)> = orders.iter().map(|&a| (a, a * rho_true)).collect();
        let rho_rec = rdp_curve_to_zcdp(&curve).expect("rho");
        assert!(
            approx(rho_rec, rho_true, 1e-12),
            "recovered ρ={rho_rec} should equal {rho_true}"
        );
    }

    #[test]
    fn test_roundtrip_zcdp_rdp_zcdp() {
        let rho = 0.42;
        let orders = default_orders();
        let curve = zcdp_to_rdp_curve(rho, &orders).expect("curve");
        let rho_back = rdp_curve_to_zcdp(&curve).expect("rho");
        assert!(
            approx(rho_back, rho, 1e-12),
            "roundtrip ρ {rho} -> curve -> {rho_back}"
        );
    }

    #[test]
    fn test_epsilon_via_rdp_matches_closed_form() {
        // The grid-optimised bound must match the analytic Bun–Steinke value
        // because the analytic optimum α* is injected into the search.
        for &rho in &[0.05f64, 0.1, 0.25, 0.5, 1.0] {
            for &delta in &[1e-3f64, 1e-5, 1e-7] {
                let orders = default_orders();
                let via = zcdp_epsilon_via_rdp(rho, delta, &orders).expect("via");
                let cf = zcdp_epsilon_closed_form(rho, delta).expect("cf");
                assert!(
                    approx(via, cf, 1e-9),
                    "ρ={rho} δ={delta}: via-RDP={via} vs closed-form={cf}"
                );
            }
        }
    }

    #[test]
    fn test_tighter_rho_gives_smaller_epsilon() {
        let orders = default_orders();
        let e_small = zcdp_epsilon_via_rdp(0.1, 1e-5, &orders).expect("a");
        let e_large = zcdp_epsilon_via_rdp(0.5, 1e-5, &orders).expect("b");
        assert!(
            e_small < e_large,
            "smaller ρ should give smaller ε: {e_small} < {e_large}"
        );
    }

    #[test]
    fn test_nonlinear_curve_takes_max_ratio() {
        // A curve where the ratio peaks at α=4 should report that peak ratio.
        let curve = vec![(2.0, 0.2), (4.0, 0.6), (8.0, 0.8)];
        // ratios: 0.1, 0.15, 0.1  → max 0.15.
        let rho = rdp_curve_to_zcdp(&curve).expect("rho");
        assert!(approx(rho, 0.15, 1e-12), "expected 0.15, got {rho}");
    }

    #[test]
    fn test_invalid_inputs() {
        assert!(zcdp_to_rdp_curve(-1.0, &[2.0]).is_err());
        assert!(zcdp_to_rdp_curve(0.1, &[1.0]).is_err());
        assert!(zcdp_to_rdp_curve(0.1, &[]).is_err());
        assert!(rdp_curve_to_zcdp(&[]).is_err());
        assert!(zcdp_epsilon_via_rdp(0.0, 1e-5, &[2.0]).is_err());
        assert!(zcdp_epsilon_via_rdp(0.1, 0.0, &[2.0]).is_err());
        assert!(zcdp_epsilon_via_rdp(0.1, 1.0, &[2.0]).is_err());
    }
}
