//! Rényi Differential Privacy (RDP) accountant for the Gaussian mechanism.
//!
//! Reference: Mironov (2017), "Rényi Differential Privacy", IEEE CSF —
//! Proposition 7 / Corollary 3 (the Gaussian mechanism).  Conversion to
//! `(ε, δ)`-DP additionally uses Canonne, Kamath & Steinke (2020),
//! "The Discrete Gaussian for Differential Privacy", Lemma 21.
//!
//! # RDP of the Gaussian mechanism
//! For the Gaussian mechanism with **noise multiplier** `σ = noise_std / Δ`
//! (so the additive noise has standard deviation `σ·Δ` for an `L2`-sensitivity-`Δ`
//! query), the Rényi divergence at order `α > 1` is exactly
//!
//! ```text
//!     ε_RDP(α) = α / (2 σ²).
//! ```
//!
//! # Composition
//! At a *fixed* order `α`, RDP composes **additively**: composing `k` Gaussian
//! steps with multipliers `σ₁, …, σ_k` gives
//! `ε_RDP(α) = Σ_k α / (2 σ_k²)`.  In particular `k` identical steps with
//! multiplier `σ` give `ε_RDP(α) = k · α / (2 σ²)` — linear in `k`.
//!
//! # Conversion to (ε, δ)-DP
//! Two valid conversions are evaluated at each grid order and the tighter is
//! taken:
//! - Mironov (2017): `ε = ε_RDP(α) + ln(1/δ) / (α − 1)`.
//! - Canonne–Kamath–Steinke (2020):
//!   `ε = ε_RDP(α) + ln((α − 1)/α) − (ln δ + ln α) / (α − 1)`.
//!
//! The reported `ε(δ)` minimises the (per-order) tighter bound over the
//! accountant's internal grid of Rényi orders `α`.

use crate::error::{PrivacyError, PrivacyResult};

/// Default grid of Rényi orders used by [`RenyiDpAccountant::new`].
///
/// Spans the small-order regime that dominates moderate-privacy budgets
/// (`α ≈ 1.25 … 8`) through the large-order tail that dominates tight-privacy
/// (small-`δ`, many-step) budgets (`α ≈ 16 … 64`).
const DEFAULT_ORDERS: &[f64] = &[
    1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
    20.0, 24.0, 28.0, 32.0, 48.0, 64.0,
];

/// Rényi-DP accountant for the (sub)Gaussian mechanism.
///
/// Accumulates per-order RDP `ε_RDP(α)` over an internal grid of Rényi orders
/// as Gaussian steps are composed, then converts to the tightest `(ε, δ)`-DP
/// guarantee on demand.
#[derive(Debug, Clone)]
pub struct RenyiDpAccountant {
    /// Internal grid of Rényi orders `α > 1`.
    orders: Vec<f64>,
    /// Accumulated RDP value `ε_RDP(α)` aligned element-wise with `orders`.
    rdp: Vec<f64>,
}

impl Default for RenyiDpAccountant {
    fn default() -> Self {
        Self::new()
    }
}

impl RenyiDpAccountant {
    /// Create a fresh accountant over the `DEFAULT_ORDERS` grid with zero
    /// accumulated RDP.
    #[must_use]
    pub fn new() -> Self {
        Self {
            orders: DEFAULT_ORDERS.to_vec(),
            rdp: vec![0.0; DEFAULT_ORDERS.len()],
        }
    }

    /// Create a fresh accountant over a custom grid of Rényi orders.
    ///
    /// # Errors
    /// - `EmptyInput` if `orders` is empty.
    /// - `InvalidParameter` if any order is non-finite or `≤ 1`.
    pub fn with_orders(orders: &[f64]) -> PrivacyResult<Self> {
        if orders.is_empty() {
            return Err(PrivacyError::EmptyInput);
        }
        for &a in orders {
            if a <= 1.0 || !a.is_finite() {
                return Err(PrivacyError::InvalidParameter(format!(
                    "Rényi order α must be finite and > 1, got {a}"
                )));
            }
        }
        Ok(Self {
            orders: orders.to_vec(),
            rdp: vec![0.0; orders.len()],
        })
    }

    /// RDP-ε of a *single* Gaussian step at order `alpha`: `α / (2 σ²)`.
    ///
    /// # Errors
    /// - `InvalidParameter` if `sigma` is non-finite or `≤ 0`, or if `alpha` is
    ///   non-finite or `≤ 1`.
    pub fn single_step_rdp(sigma: f64, alpha: f64) -> PrivacyResult<f64> {
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise multiplier σ must be finite and > 0, got {sigma}"
            )));
        }
        if alpha <= 1.0 || !alpha.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "Rényi order α must be finite and > 1, got {alpha}"
            )));
        }
        Ok(alpha / (2.0 * sigma * sigma))
    }

    /// Accumulate one Gaussian step with noise multiplier `sigma` into the
    /// per-order RDP curve.
    ///
    /// # Errors
    /// - `InvalidParameter` if `sigma` is non-finite or `≤ 0`.
    pub fn add_gaussian_step(&mut self, sigma: f64) -> PrivacyResult<()> {
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise multiplier σ must be finite and > 0, got {sigma}"
            )));
        }
        let inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
        for (r, &a) in self.rdp.iter_mut().zip(self.orders.iter()) {
            *r += a * inv_two_sigma_sq;
        }
        Ok(())
    }

    /// Convenience: accumulate `steps` identical Gaussian steps with multiplier
    /// `sigma` in one call (RDP is additive, so this equals calling
    /// [`add_gaussian_step`] `steps` times).
    ///
    /// [`add_gaussian_step`]: Self::add_gaussian_step
    ///
    /// # Errors
    /// - `InvalidParameter` if `sigma` is non-finite or `≤ 0`.
    pub fn compose(&mut self, steps: usize, sigma: f64) -> PrivacyResult<()> {
        if sigma <= 0.0 || !sigma.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "noise multiplier σ must be finite and > 0, got {sigma}"
            )));
        }
        if steps == 0 {
            return Ok(());
        }
        let factor = steps as f64 / (2.0 * sigma * sigma);
        for (r, &a) in self.rdp.iter_mut().zip(self.orders.iter()) {
            *r += a * factor;
        }
        Ok(())
    }

    /// Read-only view of the internal Rényi-order grid.
    #[must_use]
    pub fn orders(&self) -> &[f64] {
        &self.orders
    }

    /// Read-only view of the accumulated RDP values aligned with [`orders`].
    ///
    /// [`orders`]: Self::orders
    #[must_use]
    pub fn rdp_values(&self) -> &[f64] {
        &self.rdp
    }

    /// Accumulated RDP-ε at a specific grid order `alpha`.
    ///
    /// `alpha` must match one of the orders this accountant tracks (within a
    /// tight relative tolerance).
    ///
    /// # Errors
    /// - `InvalidParameter` if `alpha` is not present in the order grid.
    pub fn rdp_at(&self, alpha: f64) -> PrivacyResult<f64> {
        for (&a, &r) in self.orders.iter().zip(self.rdp.iter()) {
            if (a - alpha).abs() <= 1e-12 * a.max(1.0) {
                return Ok(r);
            }
        }
        Err(PrivacyError::InvalidParameter(format!(
            "order α={alpha} is not in the accountant's grid"
        )))
    }

    /// Convert the accumulated RDP curve to the tightest `(ε, δ)`-DP `ε` for the
    /// target `delta`, minimising over the internal order grid.
    ///
    /// # Errors
    /// - `InvalidDelta` if `delta ∉ (0, 1)`.
    /// - `InvalidParameter` if no grid order yields a finite `ε`.
    pub fn epsilon(&self, delta: f64) -> PrivacyResult<f64> {
        if !(delta > 0.0 && delta < 1.0) {
            return Err(PrivacyError::InvalidDelta(delta));
        }
        // The identity mechanism (no composed steps) is exactly (0, 0)-DP; the
        // finite-order conversion carries slack, so short-circuit it.
        if self.rdp.iter().all(|&r| r == 0.0) {
            return Ok(0.0);
        }
        let ln_inv_delta = (1.0 / delta).ln();
        let ln_delta = delta.ln();
        let mut best = f64::INFINITY;
        for (&a, &rdp_eps) in self.orders.iter().zip(self.rdp.iter()) {
            // Mironov (2017) conversion.
            let mironov = rdp_eps + ln_inv_delta / (a - 1.0);
            // Canonne–Kamath–Steinke (2020) conversion (tighter for moderate α).
            let cks = rdp_eps + ((a - 1.0) / a).ln() - (ln_delta + a.ln()) / (a - 1.0);
            let candidate = mironov.min(cks);
            if candidate.is_finite() && candidate < best {
                best = candidate;
            }
        }
        if !best.is_finite() {
            return Err(PrivacyError::InvalidParameter(
                "RDP → (ε, δ) conversion produced no finite ε".into(),
            ));
        }
        Ok(best.max(0.0))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // (e) σ ≤ 0 is rejected by every accumulation entry point.
    #[test]
    fn rejects_non_positive_sigma() {
        let mut acc = RenyiDpAccountant::new();
        assert!(acc.add_gaussian_step(0.0).is_err());
        assert!(acc.add_gaussian_step(-1.0).is_err());
        assert!(acc.add_gaussian_step(f64::INFINITY).is_err());
        assert!(acc.compose(5, 0.0).is_err());
        assert!(acc.compose(5, -2.0).is_err());
        assert!(RenyiDpAccountant::single_step_rdp(0.0, 8.0).is_err());
        assert!(RenyiDpAccountant::single_step_rdp(-1.0, 8.0).is_err());
    }

    // (e) δ ∉ (0,1) → InvalidDelta.
    #[test]
    fn rejects_bad_delta() {
        let mut acc = RenyiDpAccountant::new();
        acc.add_gaussian_step(1.0).expect("step");
        assert!(matches!(
            acc.epsilon(0.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(matches!(
            acc.epsilon(1.0),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(matches!(
            acc.epsilon(-0.1),
            Err(PrivacyError::InvalidDelta(_))
        ));
        assert!(acc.epsilon(1e-5).is_ok());
    }

    // with_orders validates the custom grid.
    #[test]
    fn with_orders_validation() {
        assert!(RenyiDpAccountant::with_orders(&[]).is_err());
        assert!(RenyiDpAccountant::with_orders(&[1.0, 2.0]).is_err());
        assert!(RenyiDpAccountant::with_orders(&[0.5]).is_err());
        assert!(RenyiDpAccountant::with_orders(&[f64::NAN]).is_err());
        assert!(RenyiDpAccountant::with_orders(&[2.0, 4.0, 8.0]).is_ok());
    }

    // (b) composing k identical steps gives ε_RDP(α) = k · α / (2 σ²), linear in k.
    #[test]
    fn rdp_is_linear_in_steps() {
        let sigma = 2.0;
        let alpha = 8.0;
        let per_step = alpha / (2.0 * sigma * sigma); // = 8 / 8 = 1.0
        for k in [1_usize, 3, 5, 10, 25] {
            let mut acc = RenyiDpAccountant::new();
            acc.compose(k, sigma).expect("compose");
            let got = acc.rdp_at(alpha).expect("rdp_at");
            let want = k as f64 * per_step;
            assert!(
                (got - want).abs() < 1e-12,
                "k={k}: rdp_at(8)={got}, want {want}"
            );
        }
    }

    // add_gaussian_step and compose agree, and composition is additive.
    #[test]
    fn add_step_matches_compose_and_is_additive() {
        let sigma = 1.5;
        let mut by_steps = RenyiDpAccountant::new();
        for _ in 0..7 {
            by_steps.add_gaussian_step(sigma).expect("step");
        }
        let mut by_compose = RenyiDpAccountant::new();
        by_compose.compose(7, sigma).expect("compose");
        for (a, b) in by_steps.rdp_values().iter().zip(by_compose.rdp_values()) {
            assert!(
                (a - b).abs() < 1e-12,
                "step vs compose mismatch: {a} vs {b}"
            );
        }

        // Split composition equals single composition.
        let mut split = RenyiDpAccountant::new();
        split.compose(3, sigma).expect("c");
        split.compose(4, sigma).expect("c");
        for (a, b) in split.rdp_values().iter().zip(by_compose.rdp_values()) {
            assert!((a - b).abs() < 1e-12, "non-additive: {a} vs {b}");
        }
    }

    // single_step_rdp returns the closed form α/(2σ²).
    #[test]
    fn single_step_closed_form() {
        for &sigma in &[0.5, 1.0, 2.0, 5.0] {
            for &alpha in &[2.0, 8.0, 32.0] {
                let got = RenyiDpAccountant::single_step_rdp(sigma, alpha).expect("ok");
                let want = alpha / (2.0 * sigma * sigma);
                assert!((got - want).abs() < 1e-12, "σ={sigma} α={alpha}");
            }
        }
    }

    // (a) one step with large σ → small ε; fresh accountant → ε = 0.
    #[test]
    fn single_large_sigma_small_epsilon() {
        let fresh = RenyiDpAccountant::new();
        assert!(
            fresh.epsilon(1e-5).expect("eps").abs() < 1e-12,
            "no steps ⇒ ε=0"
        );

        let mut acc = RenyiDpAccountant::new();
        acc.add_gaussian_step(10.0).expect("step");
        let eps = acc.epsilon(1e-5).expect("eps");
        assert!(
            eps > 0.0 && eps < 1.0,
            "σ=10 single step should give small ε, got {eps}"
        );
    }

    // (c) epsilon decreases as σ increases and as δ increases.
    #[test]
    fn epsilon_monotonic_in_sigma_and_delta() {
        let mut lo = RenyiDpAccountant::new();
        let mut hi = RenyiDpAccountant::new();
        lo.compose(100, 1.0).expect("c");
        hi.compose(100, 4.0).expect("c");
        let eps_lo = lo.epsilon(1e-5).expect("e");
        let eps_hi = hi.epsilon(1e-5).expect("e");
        assert!(eps_hi < eps_lo, "larger σ → smaller ε: {eps_hi} < {eps_lo}");

        let eps_tight = lo.epsilon(1e-8).expect("e");
        let eps_loose = lo.epsilon(1e-3).expect("e");
        assert!(
            eps_loose < eps_tight,
            "larger δ → smaller ε: {eps_loose} < {eps_tight}"
        );
    }

    // (d) k steps gives larger ε than a single step.
    #[test]
    fn more_steps_larger_epsilon() {
        let mut one = RenyiDpAccountant::new();
        let mut many = RenyiDpAccountant::new();
        one.add_gaussian_step(1.0).expect("s");
        many.compose(10, 1.0).expect("c");
        let eps_one = one.epsilon(1e-5).expect("e");
        let eps_many = many.epsilon(1e-5).expect("e");
        assert!(
            eps_many > eps_one,
            "10 steps > 1 step: {eps_many} > {eps_one}"
        );
    }

    // Sanity: DP-SGD regime σ=1, sensitivity 1, one step, δ=1e-5 ⇒ ε a few units.
    #[test]
    fn dp_sgd_single_step_magnitude() {
        let mut acc = RenyiDpAccountant::new();
        acc.add_gaussian_step(1.0).expect("step");
        let eps = acc.epsilon(1e-5).expect("eps");
        assert!(
            eps > 2.0 && eps < 8.0,
            "σ=1 single-step ε at δ=1e-5 should be a few units, got {eps}"
        );
    }

    // rdp_at rejects orders outside the grid.
    #[test]
    fn rdp_at_rejects_off_grid_order() {
        let acc = RenyiDpAccountant::new();
        assert!(acc.rdp_at(5.5).is_err());
        assert!(acc.rdp_at(8.0).is_ok());
    }
}
