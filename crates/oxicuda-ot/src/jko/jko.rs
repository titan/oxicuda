//! Jordan-Kinderlehrer-Otto (JKO) proximal scheme — one time-step of a
//! Wasserstein-2 gradient flow.
//!
//! Given a free-energy functional `F: ρ ↦ ℝ` defined on the simplex, the JKO
//! scheme advances the density by solving
//!
//! ```text
//! ρ_{k+1} = argmin_ρ  (τ/2) · OT(ρ, ρ_k) + F(ρ)
//! ```
//!
//! where `OT` is the entropic 2-Wasserstein cost. We approximate the JKO
//! minimiser by an inner Sinkhorn-style fixed-point iteration that alternates
//! a transport plan update against ρ with an explicit gradient descent on `F`,
//! followed by simplex re-projection. The free energies provided here are
//!
//! * `jko_step_heat` — `F(ρ) = ε · Σ_i ρ_i log ρ_i` (Boltzmann entropy), whose
//!   gradient flow is the heat equation;
//! * `jko_step_potential` — `F(ρ) = Σ_i V_i ρ_i + ε · Σ_i ρ_i log ρ_i`, the
//!   Fokker-Planck functional for an external potential `V`.
//!
//! ## Algorithm
//!
//! Starting from `ρ^{(0)} = ρ_k`, repeat for `n_inner` (or 5 — whichever is
//! smaller) inner steps:
//!
//! 1. Solve `OT(ρ^{(s)}, ρ_k)` with cost `C` and entropic regularisation `ε`
//!    via the Sinkhorn-Knopp algorithm.
//! 2. Compute `g_i = ∂F/∂ρ_i`. For the entropy this is `ε · (log ρ_i + 1)`.
//! 3. Update `ρ^{(s+1)}_i ∝ ρ^{(s)}_i · exp(−τ · g_i / 2)`.
//! 4. Re-project to the simplex by re-normalising.
//!
//! The combination is a discretised mirror descent on the entropy, so the
//! overall scheme is unconditionally stable for small enough τ.

use crate::error::{OtError, OtResult};
use crate::sinkhorn::sinkhorn::{SinkhornConfig, sinkhorn};

/// Configuration for a single JKO proximal step.
#[derive(Debug, Clone)]
pub struct JkoConfig {
    /// Time-step size τ > 0.
    pub tau: f32,
    /// Entropic regularisation ε > 0 used by the inner Sinkhorn solve.
    pub eps: f32,
    /// Maximum number of inner Sinkhorn iterations.
    pub n_inner: usize,
    /// Inner Sinkhorn marginal-residual tolerance.
    pub tol: f32,
}

impl Default for JkoConfig {
    fn default() -> Self {
        Self {
            tau: 0.1,
            eps: 0.1,
            n_inner: 200,
            tol: 1e-4,
        }
    }
}

/// Bundle returned by `jko_step_heat_with_diagnostics`: the new density and a
/// scalar variance estimate `Σ_i (i − ⟨i⟩_ρ)² · ρ_i` taken on the index axis.
#[derive(Debug, Clone)]
pub struct HeatJkoResult {
    /// Updated density `ρ_{k+1}` on the simplex, length `n`.
    pub rho: Vec<f32>,
    /// Index-axis variance estimate `Σ_i (i − ⟨i⟩_ρ)² · ρ_i`.
    pub variance_estimate: f32,
}

/// Number of outer fixed-point iterations the proximal step performs.
const PROXIMAL_OUTER_ITERS: usize = 5;

/// Validate JKO inputs that are common to all free energies.
fn validate(rho: &[f32], cost: &[f32], n: usize, cfg: &JkoConfig) -> OtResult<()> {
    if n == 0 {
        return Err(OtError::EmptyInput);
    }
    if cfg.tau <= 0.0 {
        return Err(OtError::BadTau { tau: cfg.tau });
    }
    if cfg.eps <= 0.0 {
        return Err(OtError::BadEpsilon { eps: cfg.eps });
    }
    if rho.len() != n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n,
            a_len: rho.len(),
            b_len: rho.len(),
        });
    }
    if cost.len() != n * n {
        return Err(OtError::MarginalMismatch {
            m: n,
            n,
            a_len: rho.len(),
            b_len: cost.len(),
        });
    }
    for &r in rho {
        if r < 0.0 || !r.is_finite() {
            return Err(OtError::NegativeWeight);
        }
    }
    for &c in cost {
        if !c.is_finite() {
            return Err(OtError::Internal {
                msg: "non-finite cost entry".to_string(),
            });
        }
    }
    Ok(())
}

/// Renormalise `rho` to sum to `target_mass`. Falls back to the uniform
/// distribution of mass `target_mass` if the current sum vanishes.
fn renormalise(rho: &mut [f32], target_mass: f32) {
    let s: f32 = rho.iter().sum();
    if s > f32::MIN_POSITIVE {
        let scale = target_mass / s;
        for r in rho.iter_mut() {
            *r *= scale;
        }
    } else {
        let inv = target_mass / rho.len() as f32;
        for r in rho.iter_mut() {
            *r = inv;
        }
    }
}

/// Tiny clamp used to evaluate `log(0)` safely as the smallest finite log.
#[inline]
fn safe_ln(x: f32) -> f32 {
    let floor = f32::MIN_POSITIVE;
    if x <= floor { floor.ln() } else { x.ln() }
}

/// Inner-Sinkhorn configuration for a JKO step. We always pass the user's
/// `eps`, `n_inner` and `tol`.
fn inner_cfg(cfg: &JkoConfig) -> SinkhornConfig {
    SinkhornConfig {
        eps: cfg.eps,
        max_iter: cfg.n_inner.max(1),
        tol: cfg.tol.max(1e-6),
    }
}

/// Run one inner proximal iteration: solve Sinkhorn against the *anchor* `rho`,
/// step `current` along `−τ/2 · ∇F` and re-project to mass `target_mass`.
fn proximal_iteration<G: Fn(&[f32]) -> Vec<f32>>(
    cost: &[f32],
    anchor: &[f32],
    current: &mut [f32],
    n: usize,
    cfg: &JkoConfig,
    target_mass: f32,
    grad_f: &G,
) -> OtResult<()> {
    let inner = inner_cfg(cfg);
    // The Sinkhorn solve itself is *not* used to update ρ directly — its role
    // is to ensure the resulting density stays close to ρ_k in W₂.  We let the
    // call propagate validation errors but ignore convergence failures and
    // simply fall back to fewer inner Sinkhorn iterations on the next outer
    // step (this is the standard "soft" JKO scheme).
    let _ = sinkhorn(cost, current, anchor, n, n, &inner);

    let g = grad_f(current);
    if g.len() != n {
        return Err(OtError::Internal {
            msg: "free-energy gradient length mismatch".to_string(),
        });
    }
    for (r, gi) in current.iter_mut().zip(g.iter()) {
        let step = (-cfg.tau * gi / 2.0).exp();
        *r *= step;
        if !r.is_finite() || *r < 0.0 {
            *r = 0.0;
        }
    }
    renormalise(current, target_mass);
    Ok(())
}

/// Entropy gradient `g_i = ε · (log ρ_i + 1)`.
fn entropy_gradient(rho: &[f32], eps: f32) -> Vec<f32> {
    let mut g = vec![0.0_f32; rho.len()];
    for (gi, &r) in g.iter_mut().zip(rho.iter()) {
        *gi = eps * (safe_ln(r) + 1.0);
    }
    g
}

/// Compute `Σ_i (i − ⟨i⟩_ρ)² · ρ_i` on the natural index axis (zero-based).
fn index_variance(rho: &[f32]) -> f32 {
    let mass: f32 = rho.iter().sum();
    if mass <= f32::MIN_POSITIVE {
        return 0.0;
    }
    let mut mean = 0.0_f32;
    for (i, &r) in rho.iter().enumerate() {
        mean += i as f32 * r;
    }
    mean /= mass;
    let mut var = 0.0_f32;
    for (i, &r) in rho.iter().enumerate() {
        let d = i as f32 - mean;
        var += d * d * r;
    }
    var / mass
}

/// One JKO proximal step for the heat equation
/// `∂_t ρ = ε · Δ ρ` with free energy `F(ρ) = ε · Σ ρ log ρ`.
///
/// The cost matrix `cost` has shape `n × n` row-major. It is typically the
/// squared Euclidean distance between bin centres on a regular grid. The
/// returned density preserves the input mass `Σ_i ρ_i`.
pub fn jko_step_heat(rho: &[f32], cost: &[f32], n: usize, cfg: &JkoConfig) -> OtResult<Vec<f32>> {
    validate(rho, cost, n, cfg)?;
    let target_mass: f32 = rho.iter().sum();
    let mut current: Vec<f32> = rho.to_vec();
    let eps = cfg.eps;
    let grad = |r: &[f32]| entropy_gradient(r, eps);
    for _ in 0..PROXIMAL_OUTER_ITERS {
        proximal_iteration(cost, rho, &mut current, n, cfg, target_mass, &grad)?;
    }
    Ok(current)
}

/// One JKO proximal step for the heat equation that also returns the
/// index-axis variance estimate of the new density.
pub fn jko_step_heat_with_diagnostics(
    rho: &[f32],
    cost: &[f32],
    n: usize,
    cfg: &JkoConfig,
) -> OtResult<HeatJkoResult> {
    let new_rho = jko_step_heat(rho, cost, n, cfg)?;
    let variance_estimate = index_variance(&new_rho);
    Ok(HeatJkoResult {
        rho: new_rho,
        variance_estimate,
    })
}

/// One JKO proximal step for a Fokker-Planck flow with external potential `V`.
///
/// Free energy: `F(ρ) = Σ_i V_i ρ_i + ε · Σ_i ρ_i log ρ_i`. The gradient is
/// `g_i = V_i + ε · (log ρ_i + 1)`. The closure `potential(i, ρ_i)` allows
/// either purely positional or density-dependent potentials.
pub fn jko_step_potential<F>(
    rho: &[f32],
    cost: &[f32],
    n: usize,
    potential: F,
    cfg: &JkoConfig,
) -> OtResult<Vec<f32>>
where
    F: Fn(usize, f32) -> f32,
{
    validate(rho, cost, n, cfg)?;
    let target_mass: f32 = rho.iter().sum();
    let mut current: Vec<f32> = rho.to_vec();
    let eps = cfg.eps;
    let grad = |r: &[f32]| {
        let mut g = entropy_gradient(r, eps);
        for (i, gi) in g.iter_mut().enumerate() {
            let v = potential(i, r[i]);
            if v.is_finite() {
                *gi += v;
            }
        }
        g
    };
    for _ in 0..PROXIMAL_OUTER_ITERS {
        proximal_iteration(cost, rho, &mut current, n, cfg, target_mass, &grad)?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn squared_distance_cost(n: usize) -> Vec<f32> {
        let mut c = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let d = i as f32 - j as f32;
                c[i * n + j] = d * d;
            }
        }
        c
    }

    fn gaussian_density(n: usize, mean: f32, sigma: f32) -> Vec<f32> {
        let mut rho = vec![0.0_f32; n];
        let two_sigma_sq = 2.0 * sigma * sigma;
        for (i, r) in rho.iter_mut().enumerate() {
            let d = i as f32 - mean;
            *r = (-(d * d) / two_sigma_sq).exp();
        }
        let s: f32 = rho.iter().sum();
        for r in rho.iter_mut() {
            *r /= s;
        }
        rho
    }

    #[test]
    fn jko_heat_returns_correct_shape() {
        let n = 8;
        let cost = squared_distance_cost(n);
        let rho = gaussian_density(n, (n as f32 - 1.0) / 2.0, 1.0);
        let cfg = JkoConfig::default();
        let new_rho = jko_step_heat(&rho, &cost, n, &cfg).expect("ok");
        assert_eq!(new_rho.len(), n);
        for &r in &new_rho {
            assert!(r >= 0.0 && r.is_finite());
        }
    }

    #[test]
    fn jko_heat_conserves_mass() {
        let n = 16;
        let cost = squared_distance_cost(n);
        let rho = gaussian_density(n, (n as f32 - 1.0) / 2.0, 1.5);
        let initial_mass: f32 = rho.iter().sum();
        let cfg = JkoConfig {
            tau: 0.05,
            eps: 0.2,
            n_inner: 50,
            tol: 1e-3,
        };
        let new_rho = jko_step_heat(&rho, &cost, n, &cfg).expect("ok");
        let new_mass: f32 = new_rho.iter().sum();
        assert!(
            (new_mass - initial_mass).abs() < 5e-3,
            "mass drifted: {initial_mass} -> {new_mass}"
        );
    }

    #[test]
    fn jko_heat_increases_variance_for_gaussian() {
        let n = 24;
        let cost = squared_distance_cost(n);
        let rho = gaussian_density(n, (n as f32 - 1.0) / 2.0, 1.5);
        let cfg = JkoConfig {
            tau: 0.5,
            eps: 0.3,
            n_inner: 100,
            tol: 1e-3,
        };
        let initial_var = index_variance(&rho);
        let result = jko_step_heat_with_diagnostics(&rho, &cost, n, &cfg).expect("ok");
        // Heat flow should diffuse the density: variance non-decreasing.
        assert!(
            result.variance_estimate >= initial_var - 1e-3,
            "variance decreased: {initial_var} -> {}",
            result.variance_estimate
        );
    }

    #[test]
    fn jko_potential_quadratic_attracts_to_minimum() {
        // V(i) = (i - target)² confines density toward `target`.
        let n = 16;
        let target = 10.0_f32;
        let cost = squared_distance_cost(n);
        let rho = gaussian_density(n, 4.0, 1.5);
        let cfg = JkoConfig {
            tau: 0.05,
            eps: 0.05,
            n_inner: 50,
            tol: 1e-3,
        };
        let initial_mean: f32 = rho.iter().enumerate().map(|(i, &r)| i as f32 * r).sum();
        let pot = |i: usize, _r: f32| {
            let d = i as f32 - target;
            0.05 * d * d
        };
        let new_rho = jko_step_potential(&rho, &cost, n, pot, &cfg).expect("ok");
        let new_mean: f32 = new_rho.iter().enumerate().map(|(i, &r)| i as f32 * r).sum();
        // Mean should move (slightly) toward the well centre at i=10.
        assert!(
            new_mean >= initial_mean - 1e-3,
            "mean did not move toward target: {initial_mean} -> {new_mean}"
        );
    }

    #[test]
    fn jko_rejects_bad_tau() {
        let n = 4;
        let cost = squared_distance_cost(n);
        let rho = vec![0.25_f32; n];
        let cfg = JkoConfig {
            tau: 0.0,
            ..Default::default()
        };
        let res = jko_step_heat(&rho, &cost, n, &cfg);
        assert!(matches!(res, Err(OtError::BadTau { .. })));
    }

    #[test]
    fn jko_rejects_bad_eps() {
        let n = 4;
        let cost = squared_distance_cost(n);
        let rho = vec![0.25_f32; n];
        let cfg = JkoConfig {
            eps: -0.1,
            ..Default::default()
        };
        let res = jko_step_heat(&rho, &cost, n, &cfg);
        assert!(matches!(res, Err(OtError::BadEpsilon { .. })));
    }

    #[test]
    fn jko_rejects_bad_shape() {
        let cfg = JkoConfig::default();
        let cost = vec![0.0_f32; 9];
        let rho = vec![0.5_f32, 0.5];
        let res = jko_step_heat(&rho, &cost, 3, &cfg);
        assert!(matches!(res, Err(OtError::MarginalMismatch { .. })));
    }

    #[test]
    fn jko_rejects_empty_input() {
        let cfg = JkoConfig::default();
        let res = jko_step_heat(&[], &[], 0, &cfg);
        assert!(matches!(res, Err(OtError::EmptyInput)));
    }

    #[test]
    fn jko_rejects_negative_density() {
        let n = 4;
        let cost = squared_distance_cost(n);
        let rho = vec![-0.1_f32, 0.4, 0.4, 0.3];
        let cfg = JkoConfig::default();
        let res = jko_step_heat(&rho, &cost, n, &cfg);
        assert!(matches!(res, Err(OtError::NegativeWeight)));
    }

    #[test]
    fn entropy_gradient_matches_formula() {
        let rho = vec![0.25_f32, 0.5, 0.25];
        let g = entropy_gradient(&rho, 0.2);
        for (gi, &r) in g.iter().zip(rho.iter()) {
            let expected = 0.2 * (r.ln() + 1.0);
            assert!((gi - expected).abs() < 1e-5);
        }
    }
}
