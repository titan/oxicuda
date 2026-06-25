//! ADMM with adaptive penalty `ρ` via residual balancing (Boyd et al. 2011, §3.4.1).
//!
//! Solves the generic two-block problem
//!
//! ```text
//!   min  f(x) + g(z)   s.t.  A x + B z = c,
//! ```
//!
//! in scaled form (scaled dual `u = y / ρ`):
//!
//! ```text
//!   x_{k+1} = argmin_x  f(x) + (ρ/2) ‖A x + B z_k − c + u_k‖²
//!   z_{k+1} = argmin_z  g(z) + (ρ/2) ‖A x_{k+1} + B z − c + u_k‖²
//!   u_{k+1} = u_k + (A x_{k+1} + B z_{k+1} − c).
//! ```
//!
//! The primal and dual residuals are
//!
//! ```text
//!   r_k = A x_{k+1} + B z_{k+1} − c,                (primal feasibility)
//!   s_k = ρ Aᵀ B (z_{k+1} − z_k).                   (dual feasibility)
//! ```
//!
//! # Residual balancing
//!
//! A fixed penalty `ρ` trades off the two residuals: a large `ρ` drives the primal
//! residual down at the expense of the dual residual, and vice-versa.  The Boyd
//! residual-balancing scheme adapts `ρ` to keep `‖r_k‖` and `‖s_k‖` within a factor
//! `μ` of each other:
//!
//! ```text
//!   ρ_{k+1} = τ⁺ ρ_k          if ‖r_k‖ > μ ‖s_k‖,
//!   ρ_{k+1} = ρ_k / τ⁻        if ‖s_k‖ > μ ‖r_k‖,
//!   ρ_{k+1} = ρ_k             otherwise,
//! ```
//!
//! with `μ > 1` (typically `10`) and `τ⁺, τ⁻ > 1` (typically `2`).  Because the
//! **scaled** dual `u = y / ρ` is stored, every change `ρ_k → ρ_{k+1}` must rescale
//! `u_{k+1} ← (ρ_k / ρ_{k+1}) u_{k+1}` so the unscaled multiplier `y = ρ u` is
//! preserved.  Updates are throttled to at most once per [`AdaptiveRhoConfig::adapt_every`]
//! iterations to avoid oscillation.
//!
//! The standard ε-feasibility stopping rule (Boyd §3.3.1) is used:
//!
//! ```text
//!   ‖r_k‖ ≤ √p · ε_abs + ε_rel · max(‖A x‖, ‖B z‖, ‖c‖),
//!   ‖s_k‖ ≤ √n · ε_abs + ε_rel · ‖Aᵀ y‖.
//! ```
//!
//! # References
//!
//! - S. Boyd, N. Parikh, E. Chu, B. Peleato & J. Eckstein (2011), "Distributed
//!   Optimization and Statistical Learning via the Alternating Direction Method of
//!   Multipliers", *Foundations and Trends in Machine Learning* 3(1):1-122, §3.4.1.
//! - B. Wohlberg (2017), "ADMM Penalty Parameter Selection by Residual Balancing".

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};

/// Configuration for [`adaptive_rho_admm`].
#[derive(Debug, Clone)]
pub struct AdaptiveRhoConfig {
    /// Initial penalty `ρ_0 > 0`.
    pub rho0: f64,
    /// Maximum number of outer ADMM iterations.
    pub max_iter: usize,
    /// Absolute tolerance `ε_abs` in the Boyd feasibility test.
    pub eps_abs: f64,
    /// Relative tolerance `ε_rel` in the Boyd feasibility test.
    pub eps_rel: f64,
    /// Imbalance threshold `μ > 1`: adapt when one residual exceeds `μ ×` the other.
    pub mu: f64,
    /// Multiplicative increase factor `τ⁺ > 1` applied when the primal residual dominates.
    pub tau_incr: f64,
    /// Multiplicative decrease factor `τ⁻ > 1` applied when the dual residual dominates.
    pub tau_decr: f64,
    /// Update `ρ` at most once every `adapt_every` iterations (`≥ 1`).
    pub adapt_every: usize,
    /// Lower clamp on `ρ` to prevent underflow.
    pub rho_min: f64,
    /// Upper clamp on `ρ` to prevent overflow.
    pub rho_max: f64,
}

impl Default for AdaptiveRhoConfig {
    fn default() -> Self {
        Self {
            rho0: 1.0,
            max_iter: 1000,
            eps_abs: 1e-8,
            eps_rel: 1e-6,
            mu: 10.0,
            tau_incr: 2.0,
            tau_decr: 2.0,
            adapt_every: 1,
            rho_min: 1e-8,
            rho_max: 1e8,
        }
    }
}

impl AdaptiveRhoConfig {
    /// Validate the configuration, returning a descriptive error on the first
    /// invariant violation.
    ///
    /// # Errors
    /// Returns [`CvxError::InvalidParameter`] for any non-finite or out-of-range field.
    pub fn validate(&self) -> CvxResult<()> {
        let positive_finite = |v: f64| v.is_finite() && v > 0.0;
        let nonneg_finite = |v: f64| v.is_finite() && v >= 0.0;
        if !positive_finite(self.rho0) {
            return Err(CvxError::InvalidParameter(format!(
                "rho0 must be a positive finite number, got {}",
                self.rho0
            )));
        }
        if self.max_iter == 0 {
            return Err(CvxError::InvalidParameter("max_iter must be ≥ 1".into()));
        }
        if !nonneg_finite(self.eps_abs) || !nonneg_finite(self.eps_rel) {
            return Err(CvxError::InvalidParameter(
                "eps_abs and eps_rel must be non-negative and finite".into(),
            ));
        }
        if !(self.mu.is_finite() && self.mu > 1.0) {
            return Err(CvxError::InvalidParameter(format!(
                "mu must be > 1, got {}",
                self.mu
            )));
        }
        let valid_tau = |v: f64| v.is_finite() && v > 1.0;
        if !(valid_tau(self.tau_incr) && valid_tau(self.tau_decr)) {
            return Err(CvxError::InvalidParameter(
                "tau_incr and tau_decr must be finite and > 1".into(),
            ));
        }
        if self.adapt_every == 0 {
            return Err(CvxError::InvalidParameter("adapt_every must be ≥ 1".into()));
        }
        if !positive_finite(self.rho_min) || self.rho_min > self.rho_max {
            return Err(CvxError::InvalidParameter(
                "require 0 < rho_min ≤ rho_max".into(),
            ));
        }
        Ok(())
    }
}

/// Result returned by [`adaptive_rho_admm`].
#[derive(Debug, Clone)]
pub struct AdaptiveRhoResult {
    /// Primal block `x` (length `an`).
    pub x: Vec<f64>,
    /// Primal block `z` (length `bn`).
    pub z: Vec<f64>,
    /// Scaled dual `u = y / ρ` (length `am`).
    pub u: Vec<f64>,
    /// Unscaled dual multiplier `y = ρ u` (length `am`).
    pub y: Vec<f64>,
    /// Final penalty `ρ` reached after adaptation.
    pub rho: f64,
    /// Iterations performed.
    pub iter: usize,
    /// Final primal residual `‖r_k‖₂`.
    pub pri_residual: f64,
    /// Final dual residual `‖s_k‖₂`.
    pub dual_residual: f64,
    /// Number of times `ρ` was changed.
    pub rho_updates: usize,
    /// Whether the Boyd ε-feasibility stopping rule fired.
    pub converged: bool,
}

/// ADMM with Boyd residual-balancing adaptive penalty.
///
/// The `x_update(z, u, rho)` closure must return the new `x` minimising
/// `f(x) + (ρ/2)‖A x + B z − c + u‖²` (typically a prox or a small linear solve),
/// and likewise `z_update(x, u, rho)`.  Both receive the **current** penalty `ρ`
/// because the augmented-Lagrangian sub-problems depend on it.
///
/// # Parameters
/// * `a`, `am`, `an` — constraint matrix `A` (row-major `am × an`).
/// * `b`, `bn`       — constraint matrix `B` (row-major `am × bn`).
/// * `c`             — constraint right-hand side (length `am`).
/// * `x_update`      — solver for the `x`-block sub-problem.
/// * `z_update`      — solver for the `z`-block sub-problem.
/// * `config`        — penalty / tolerance settings.
///
/// # Errors
/// * [`CvxError::InvalidParameter`] for an invalid configuration.
/// * [`CvxError::ShapeMismatch`] / [`CvxError::DimensionMismatch`] for inconsistent sizes
///   (including a sub-problem closure returning a wrongly-sized vector).
#[allow(clippy::too_many_arguments)]
pub fn adaptive_rho_admm<X, Z>(
    a: &[f64],
    am: usize,
    an: usize,
    b: &[f64],
    bn: usize,
    c: &[f64],
    x_update: X,
    z_update: Z,
    config: &AdaptiveRhoConfig,
) -> CvxResult<AdaptiveRhoResult>
where
    X: Fn(&[f64], &[f64], f64) -> CvxResult<Vec<f64>>,
    Z: Fn(&[f64], &[f64], f64) -> CvxResult<Vec<f64>>,
{
    config.validate()?;
    if a.len() != am * an {
        return Err(CvxError::ShapeMismatch {
            expected: vec![am, an],
            got: vec![a.len()],
        });
    }
    if b.len() != am * bn {
        return Err(CvxError::ShapeMismatch {
            expected: vec![am, bn],
            got: vec![b.len()],
        });
    }
    if c.len() != am {
        return Err(CvxError::DimensionMismatch { a: c.len(), b: am });
    }

    let mut x = vec![0.0_f64; an];
    let mut z = vec![0.0_f64; bn];
    let mut u = vec![0.0_f64; am];
    let mut rho = config.rho0;

    let sqrt_p = (am as f64).sqrt();
    let sqrt_n = ((an + bn) as f64).sqrt();
    let c_norm = norm2(c);

    let mut pri_norm = f64::INFINITY;
    let mut dual_norm = f64::INFINITY;
    let mut rho_updates = 0usize;
    let mut iters = 0usize;
    let mut converged = false;

    for it in 0..config.max_iter {
        iters = it + 1;

        // ── block updates ───────────────────────────────────────────────────
        let x_new = x_update(&z, &u, rho)?;
        if x_new.len() != an {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: an,
            });
        }
        let z_new = z_update(&x_new, &u, rho)?;
        if z_new.len() != bn {
            return Err(CvxError::DimensionMismatch {
                a: z_new.len(),
                b: bn,
            });
        }

        // ── primal residual r = A x + B z − c ───────────────────────────────
        let ax = mat_vec(a, am, an, &x_new)?;
        let bz = mat_vec(b, am, bn, &z_new)?;
        let mut r = vec![0.0_f64; am];
        for i in 0..am {
            r[i] = ax[i] + bz[i] - c[i];
        }

        // ── scaled-dual update u ← u + r ────────────────────────────────────
        for i in 0..am {
            u[i] += r[i];
        }

        // ── dual residual s = ρ Aᵀ B (z_new − z) ────────────────────────────
        let mut dz = vec![0.0_f64; bn];
        for i in 0..bn {
            dz[i] = z_new[i] - z[i];
        }
        let b_dz = mat_vec(b, am, bn, &dz)?;
        let at_bdz = mat_t_vec(a, am, an, &b_dz)?;
        let s: Vec<f64> = at_bdz.iter().map(|v| rho * v).collect();

        pri_norm = norm2(&r);
        dual_norm = norm2(&s);

        // ── Boyd ε-feasibility stopping rule ────────────────────────────────
        let ax_norm = norm2(&ax);
        let bz_norm = norm2(&bz);
        let eps_pri = sqrt_p * config.eps_abs + config.eps_rel * ax_norm.max(bz_norm).max(c_norm);
        // y = ρ u; ‖Aᵀ y‖ scales the dual tolerance.
        let y_now: Vec<f64> = u.iter().map(|ui| rho * ui).collect();
        let at_y = mat_t_vec(a, am, an, &y_now)?;
        let eps_dual = sqrt_n * config.eps_abs + config.eps_rel * norm2(&at_y);

        x = x_new;
        z = z_new;

        if pri_norm <= eps_pri && dual_norm <= eps_dual {
            converged = true;
            break;
        }

        // ── residual-balancing penalty update (throttled) ───────────────────
        if (it + 1) % config.adapt_every == 0 {
            let old_rho = rho;
            if pri_norm > config.mu * dual_norm {
                rho = (rho * config.tau_incr).min(config.rho_max);
            } else if dual_norm > config.mu * pri_norm {
                rho = (rho / config.tau_decr).max(config.rho_min);
            }
            if rho != old_rho {
                // Preserve the unscaled multiplier y = ρ u ⇒ rescale u.
                let factor = old_rho / rho;
                for ui in &mut u {
                    *ui *= factor;
                }
                rho_updates += 1;
            }
        }
    }

    let y: Vec<f64> = u.iter().map(|ui| rho * ui).collect();
    Ok(AdaptiveRhoResult {
        x,
        z,
        u,
        y,
        rho,
        iter: iters,
        pri_residual: pri_norm,
        dual_residual: dual_norm,
        rho_updates,
        converged,
    })
}

#[cfg(test)]
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
mod tests {
    use super::*;
    use crate::prox_ops::l1::soft_threshold;

    // x-update for the consensus split x − z = 0 of  min ½‖x − d‖² + λ‖z‖₁:
    //   argmin_x ½‖x − d‖² + (ρ/2)‖x − z + u‖² = (d + ρ(z − u)) / (1 + ρ).
    fn make_lasso(
        d: Vec<f64>,
        lambda: f64,
    ) -> (
        impl Fn(&[f64], &[f64], f64) -> CvxResult<Vec<f64>>,
        impl Fn(&[f64], &[f64], f64) -> CvxResult<Vec<f64>>,
    ) {
        let d_x = d.clone();
        let xu = move |z: &[f64], u: &[f64], rho: f64| -> CvxResult<Vec<f64>> {
            Ok((0..d_x.len())
                .map(|i| (d_x[i] + rho * (z[i] - u[i])) / (1.0 + rho))
                .collect())
        };
        let zu = move |x: &[f64], u: &[f64], rho: f64| -> CvxResult<Vec<f64>> {
            Ok((0..x.len())
                .map(|i| soft_threshold(x[i] + u[i], lambda / rho))
                .collect())
        };
        (xu, zu)
    }

    fn identity_consensus(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // A = I, B = −I, c = 0  ⇒  x − z = 0.
        let mut a = vec![0.0_f64; n * n];
        let mut b = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
            b[i * n + i] = -1.0;
        }
        (a, b, vec![0.0_f64; n])
    }

    #[test]
    fn lasso_matches_soft_threshold_optimum() {
        // min ½‖x − d‖² + λ‖z‖₁ s.t. x = z  ⇒  z* = soft_threshold(d, λ).
        let d = vec![3.0_f64, -2.0, 0.5, 5.0, -0.3];
        let lambda = 1.0;
        let (xu, zu) = make_lasso(d.clone(), lambda);
        let (a, b, c) = identity_consensus(d.len());
        let cfg = AdaptiveRhoConfig {
            rho0: 1.0,
            max_iter: 2000,
            eps_abs: 1e-10,
            eps_rel: 1e-10,
            ..Default::default()
        };
        let res =
            adaptive_rho_admm(&a, d.len(), d.len(), &b, d.len(), &c, xu, zu, &cfg).expect("solves");
        for i in 0..d.len() {
            let want = soft_threshold(d[i], lambda);
            assert!(
                (res.z[i] - want).abs() < 1e-5,
                "z[{i}]={} want {want}",
                res.z[i]
            );
        }
        assert!(res.converged, "did not hit ε-feasibility");
        assert!(res.pri_residual < 1e-4 && res.dual_residual < 1e-4);
    }

    #[test]
    fn rho_actually_adapts_from_imbalanced_start() {
        // Start with an extreme ρ so a balancing move is forced.
        let d = vec![4.0_f64, -1.0, 2.0];
        let (xu, zu) = make_lasso(d.clone(), 0.5);
        let (a, b, c) = identity_consensus(d.len());
        let cfg = AdaptiveRhoConfig {
            rho0: 1e4, // huge ⇒ dual residual dominates early ⇒ ρ should shrink
            max_iter: 1000,
            eps_abs: 1e-9,
            eps_rel: 1e-9,
            mu: 10.0,
            tau_incr: 2.0,
            tau_decr: 2.0,
            adapt_every: 1,
            ..Default::default()
        };
        let res =
            adaptive_rho_admm(&a, d.len(), d.len(), &b, d.len(), &c, xu, zu, &cfg).expect("solves");
        assert!(res.rho_updates > 0, "ρ never adapted");
        assert!(res.rho < cfg.rho0, "ρ should have shrunk from a huge start");
        // Still converges to the correct optimum.
        for i in 0..d.len() {
            let want = soft_threshold(d[i], 0.5);
            assert!((res.z[i] - want).abs() < 1e-4);
        }
    }

    #[test]
    fn scaled_dual_consistent_with_unscaled() {
        // y = ρ u must hold exactly in the returned result.
        let d = vec![2.0_f64, -3.0];
        let (xu, zu) = make_lasso(d.clone(), 1.0);
        let (a, b, c) = identity_consensus(d.len());
        let cfg = AdaptiveRhoConfig {
            rho0: 5.0,
            max_iter: 500,
            ..Default::default()
        };
        let res =
            adaptive_rho_admm(&a, d.len(), d.len(), &b, d.len(), &c, xu, zu, &cfg).expect("solves");
        for i in 0..d.len() {
            assert!((res.y[i] - res.rho * res.u[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn converges_at_least_as_fast_as_fixed_bad_rho() {
        // Adaptive ρ from a bad start should beat the fixed-but-bad penalty.
        let d = vec![3.0_f64, -2.0, 0.5, 1.0, -4.0, 2.5];
        let lambda = 0.8;
        let (a, b, c) = identity_consensus(d.len());

        let cfg_adapt = AdaptiveRhoConfig {
            rho0: 1e3,
            max_iter: 5000,
            eps_abs: 1e-9,
            eps_rel: 1e-9,
            ..Default::default()
        };
        let (xu1, zu1) = make_lasso(d.clone(), lambda);
        let adapt = adaptive_rho_admm(&a, d.len(), d.len(), &b, d.len(), &c, xu1, zu1, &cfg_adapt)
            .expect("solves");

        // Fixed bad penalty: same ρ0, never adapts (adapt_every huge).
        let cfg_fixed = AdaptiveRhoConfig {
            rho0: 1e3,
            max_iter: 5000,
            eps_abs: 1e-9,
            eps_rel: 1e-9,
            adapt_every: usize::MAX,
            ..Default::default()
        };
        let (xu2, zu2) = make_lasso(d.clone(), lambda);
        let fixed = adaptive_rho_admm(&a, d.len(), d.len(), &b, d.len(), &c, xu2, zu2, &cfg_fixed)
            .expect("solves");

        assert!(adapt.converged, "adaptive run failed to converge");
        // Both reach the same optimum...
        for i in 0..d.len() {
            let want = soft_threshold(d[i], lambda);
            assert!((adapt.z[i] - want).abs() < 1e-4);
        }
        // ...but adaptive needs strictly fewer iterations from the bad start.
        assert!(
            adapt.iter < fixed.iter,
            "adaptive {} vs fixed {}",
            adapt.iter,
            fixed.iter
        );
    }

    #[test]
    fn rejects_bad_config() {
        let d = vec![1.0_f64];
        let (xu, zu) = make_lasso(d.clone(), 1.0);
        let (a, b, c) = identity_consensus(1);
        let bad = AdaptiveRhoConfig {
            mu: 0.5, // must be > 1
            ..Default::default()
        };
        let r = adaptive_rho_admm(&a, 1, 1, &b, 1, &c, xu, zu, &bad);
        assert!(matches!(r, Err(CvxError::InvalidParameter(_))), "{r:?}");
    }

    #[test]
    fn rejects_shape_mismatch() {
        let d = vec![1.0_f64, 2.0];
        let (xu, zu) = make_lasso(d.clone(), 1.0);
        let (_a, b, c) = identity_consensus(2);
        let bad_a = vec![1.0, 0.0, 0.0]; // not 2×2
        let cfg = AdaptiveRhoConfig::default();
        let r = adaptive_rho_admm(&bad_a, 2, 2, &b, 2, &c, xu, zu, &cfg);
        assert!(matches!(r, Err(CvxError::ShapeMismatch { .. })), "{r:?}");
    }
}
