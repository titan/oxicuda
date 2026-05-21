//! Imai-Keele-Tingley causal-mediation decomposition.
//!
//! Reference: Imai, K., Keele, L., & Tingley, D. (2010). "A general approach
//! to causal mediation analysis." *Psychological Methods*, 15(4), 309-334.
//! See also Imai, K., Keele, L., & Yamamoto, T. (2010). "Identification,
//! inference, and sensitivity analysis for causal mediation effects."
//! *Statistical Science*, 25(1), 51-71.
//!
//! # Setting
//!
//! Let `T_i` be a treatment indicator, `M_i` a continuous mediator, `Y_i` a
//! continuous outcome, and `X_i ∈ R^d` a vector of pre-treatment covariates.
//! Under *sequential ignorability* (Imai-Keele-Tingley 2010 Assumption 1):
//!
//! 1. `{Y_i(t', m), M_i(t)} ⫫ T_i  |  X_i = x` and
//! 2. `Y_i(t', m) ⫫ M_i(t)  |  T_i = t, X_i = x`,
//!
//! the *average causal mediation effect* (ACME, aka the indirect effect) and
//! the *average direct effect* (ADE) are non-parametrically identified.
//!
//! # Algorithm — parametric estimator (IKT §3.2)
//!
//! Two parametric working models are fitted by ridge OLS,
//!
//! ```text
//!   Mediator model:  M = α_m + β_t · T + Σ_k β_{xk} · X_k + ε_m
//!   Outcome  model:  Y = α_y + γ_t · T + γ_m · M + γ_{tm} · (T · M)
//!                       + Σ_k γ_{xk} · X_k + ε_y
//! ```
//!
//! For each sample `i` and each pair of *counterfactual treatment values*
//! `(t_med, t_out) ∈ {0, 1}²`, the conditional expectations are
//!
//! ```text
//!   M̂(t_med, x_i)        = α_m + β_t · t_med + Σ_k β_{xk} · x_{ik}
//!   Ŷ(t_out, M̂, x_i)     = α_y + γ_t · t_out + γ_m · M̂
//!                          + γ_{tm} · t_out · M̂ + Σ_k γ_{xk} · x_{ik}
//! ```
//!
//! The two natural causal estimands (per IKT 2010 eq. 4) are
//!
//! ```text
//!   δ̂(t_out) = (1/n) Σ_i [ Ŷ(t_out, M̂(1, x_i), x_i)
//!                          − Ŷ(t_out, M̂(0, x_i), x_i) ]
//!   ζ̂(t_med) = (1/n) Σ_i [ Ŷ(1, M̂(t_med, x_i), x_i)
//!                          − Ŷ(0, M̂(t_med, x_i), x_i) ]
//! ```
//!
//! and we report the symmetric averages
//!
//! ```text
//!   ACME = (δ̂(0) + δ̂(1)) / 2
//!   ADE  = (ζ̂(0) + ζ̂(1)) / 2
//!   Total = ACME + ADE
//!   prop_mediated = ACME / Total
//! ```
//!
//! ## Monte-Carlo confidence intervals
//!
//! Imai-Keele-Tingley recommend a *quasi-Bayesian / parametric bootstrap*
//! that draws coefficient vectors from the asymptotic sampling distribution
//! `N(θ̂, V̂)` where `V̂` is the (ridge-regularised) sandwich covariance,
//!
//! ```text
//!   V̂ = σ̂² · (Z^T Z + λ I)^{-1}.
//! ```
//!
//! We approximate `V̂` by its diagonal — `V̂_jj ≈ σ̂² · D_jj` where `D` is
//! the inverse of `Z^T Z + λ I` evaluated on the diagonal — and sample each
//! coefficient independently as `θ_j ~ N(θ̂_j, V̂_jj)`.  This is the
//! "independent-components" approximation noted in the implementation plan;
//! it provides an upper-tail-conservative band that converges to the exact
//! distribution when off-diagonal terms of `V̂` are negligible (well-scaled
//! design).
//!
//! Each Monte-Carlo draw recomputes `(ACME, ADE)` from the perturbed
//! coefficients and the 2.5 % / 97.5 % empirical quantiles of the
//! resulting samples form the reported 95 % CIs.  Determinism is preserved
//! via `LcgRng` seeded from `cfg.seed`.

use crate::error::{CausalError, CausalResult};
use crate::handle::LcgRng;

/// Configuration for [`Mediation::estimate`].
#[derive(Debug, Clone)]
pub struct MediationConfig {
    /// Ridge penalty for both the mediator and outcome ridge OLS solves.
    /// Must be strictly positive.
    pub ridge_lambda: f64,
    /// Number of Monte-Carlo simulations used to construct ACME / ADE 95 %
    /// confidence intervals.  Must be `≥ 100`.
    pub n_simulations: usize,
    /// Seed for the deterministic [`LcgRng`] used by the parametric
    /// bootstrap.
    pub seed: u64,
}

impl Default for MediationConfig {
    fn default() -> Self {
        Self {
            ridge_lambda: 1e-3,
            n_simulations: 1_000,
            seed: 0,
        }
    }
}

/// Output of [`Mediation::estimate`].
#[derive(Debug, Clone)]
pub struct MediationResult {
    /// Average causal mediation (indirect) effect.
    pub acme: f64,
    /// Average direct effect.
    pub ade: f64,
    /// Total effect: `acme + ade`.
    pub total_effect: f64,
    /// Proportion mediated: `acme / total_effect`.  Returns `f64::NAN` when
    /// the total effect is exactly zero.
    pub prop_mediated: f64,
    /// 95 % Monte-Carlo CI for ACME: `(2.5 %, 97.5 %)` quantiles.
    pub acme_ci: (f64, f64),
    /// 95 % Monte-Carlo CI for ADE: `(2.5 %, 97.5 %)` quantiles.
    pub ade_ci: (f64, f64),
    /// Number of observations used.
    pub n: usize,
}

/// Zero-sized handle for the IKT mediation estimator.
pub struct Mediation;

impl Mediation {
    /// Estimate ACME, ADE and 95 % Monte-Carlo CIs from observational data.
    ///
    /// # Parameters
    /// - `y`: continuous outcomes, length `n`.
    /// - `t`: treatment indicators (each in `{0.0, 1.0}`), length `n`.
    /// - `m`: continuous mediator values, length `n`.
    /// - `x`: row-major covariate matrix; `x[i].len() == d` for every `i`.
    /// - `cfg`: see [`MediationConfig`].
    ///
    /// # Errors
    /// - [`CausalError::DimensionMismatch`] for empty inputs or row-width
    ///   mismatches.
    /// - [`CausalError::IncompatibleData`] for `n < 5`, `ridge_lambda ≤ 0`,
    ///   `n_simulations < 100`, or `t[i] ∉ {0.0, 1.0}`.
    /// - [`CausalError::MatrixSingular`] if either ridge-augmented normal
    ///   equation is rank-deficient.
    pub fn estimate(
        y: &[f64],
        t: &[f64],
        m: &[f64],
        x: &[Vec<f64>],
        cfg: &MediationConfig,
    ) -> CausalResult<MediationResult> {
        // ---- shape validation ------------------------------------------
        let n = y.len();
        if n == 0 || t.is_empty() || m.is_empty() || x.is_empty() {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: 0,
            });
        }
        if t.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: t.len(),
            });
        }
        if m.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: m.len(),
            });
        }
        if x.len() != n {
            return Err(CausalError::DimensionMismatch {
                expected: n,
                got: x.len(),
            });
        }
        let d = x[0].len();
        if d == 0 {
            return Err(CausalError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        for row in x.iter() {
            if row.len() != d {
                return Err(CausalError::DimensionMismatch {
                    expected: d,
                    got: row.len(),
                });
            }
        }

        // ---- config / numeric validation ------------------------------
        if cfg.ridge_lambda <= 0.0 || !cfg.ridge_lambda.is_finite() {
            return Err(CausalError::IncompatibleData);
        }
        if cfg.n_simulations < 100 {
            return Err(CausalError::IncompatibleData);
        }
        if n < 5 {
            return Err(CausalError::IncompatibleData);
        }
        for &ti in t {
            if !(ti == 0.0 || ti == 1.0) {
                return Err(CausalError::IncompatibleData);
            }
        }
        for &mi in m {
            if !mi.is_finite() {
                return Err(CausalError::IncompatibleData);
            }
        }
        for &yi in y {
            if !yi.is_finite() {
                return Err(CausalError::IncompatibleData);
            }
        }
        for row in x.iter() {
            for &xij in row.iter() {
                if !xij.is_finite() {
                    return Err(CausalError::IncompatibleData);
                }
            }
        }

        // ---- design matrices -------------------------------------------
        // Mediator design: z_m = [1, t, x_1, ..., x_d]  ⇒  p_m = d + 2
        let p_m = d + 2;
        let mut z_m = vec![0.0_f64; n * p_m];
        for i in 0..n {
            let r = i * p_m;
            z_m[r] = 1.0;
            z_m[r + 1] = t[i];
            for j in 0..d {
                z_m[r + 2 + j] = x[i][j];
            }
        }
        // Outcome design: z_y = [1, t, m, t·m, x_1, ..., x_d]  ⇒  p_y = d + 4
        let p_y = d + 4;
        let mut z_y = vec![0.0_f64; n * p_y];
        for i in 0..n {
            let r = i * p_y;
            z_y[r] = 1.0;
            z_y[r + 1] = t[i];
            z_y[r + 2] = m[i];
            z_y[r + 3] = t[i] * m[i];
            for j in 0..d {
                z_y[r + 4 + j] = x[i][j];
            }
        }

        // ---- ridge OLS solves -----------------------------------------
        let (beta_m, gram_inv_m_diag) =
            ridge_solve_with_inv_diag(&z_m, m, n, p_m, cfg.ridge_lambda)?;
        let (beta_y, gram_inv_y_diag) =
            ridge_solve_with_inv_diag(&z_y, y, n, p_y, cfg.ridge_lambda)?;

        // ---- residual variances σ̂²_m and σ̂²_y -----------------------
        let sigma2_m = residual_variance(&z_m, m, &beta_m, n, p_m);
        let sigma2_y = residual_variance(&z_y, y, &beta_y, n, p_y);

        // ---- point estimates -------------------------------------------
        let (acme, ade) = compute_acme_ade(&beta_m, &beta_y, x, d);
        let total = acme + ade;
        let prop_mediated = if total.abs() < f64::EPSILON {
            f64::NAN
        } else {
            acme / total
        };

        // ---- Monte-Carlo CIs ------------------------------------------
        let mut rng = LcgRng::new(cfg.seed);
        let mut acme_draws = Vec::with_capacity(cfg.n_simulations);
        let mut ade_draws = Vec::with_capacity(cfg.n_simulations);
        // Per-coefficient sampling SDs (independent-components approx).
        let sd_m: Vec<f64> = (0..p_m)
            .map(|j| (sigma2_m * gram_inv_m_diag[j].max(0.0)).sqrt())
            .collect();
        let sd_y: Vec<f64> = (0..p_y)
            .map(|j| (sigma2_y * gram_inv_y_diag[j].max(0.0)).sqrt())
            .collect();
        for _ in 0..cfg.n_simulations {
            let mut sample_m = vec![0.0_f64; p_m];
            for j in 0..p_m {
                let z = rng.next_normal() as f64;
                sample_m[j] = beta_m[j] + sd_m[j] * z;
            }
            let mut sample_y = vec![0.0_f64; p_y];
            for j in 0..p_y {
                let z = rng.next_normal() as f64;
                sample_y[j] = beta_y[j] + sd_y[j] * z;
            }
            let (a, d_eff) = compute_acme_ade(&sample_m, &sample_y, x, d);
            acme_draws.push(a);
            ade_draws.push(d_eff);
        }
        let acme_ci = empirical_ci(&mut acme_draws, 0.025, 0.975);
        let ade_ci = empirical_ci(&mut ade_draws, 0.025, 0.975);

        Ok(MediationResult {
            acme,
            ade,
            total_effect: total,
            prop_mediated,
            acme_ci,
            ade_ci,
            n,
        })
    }
}

// =====================================================================
// Core estimand computation
// =====================================================================

/// Given mediator and outcome coefficient vectors, return `(ACME, ADE)`
/// averaged over the symmetric `t_med`/`t_out` choices.
fn compute_acme_ade(beta_m: &[f64], beta_y: &[f64], x: &[Vec<f64>], d: usize) -> (f64, f64) {
    // beta_m layout: [α_m, β_t, β_x1, ..., β_xd]              (length d+2)
    // beta_y layout: [α_y, γ_t, γ_m, γ_tm, γ_x1, ..., γ_xd]   (length d+4)
    let n = x.len();
    let inv_n = 1.0 / n as f64;

    // Precompute Σ β_xk · x_ik (mediator) and Σ γ_xk · x_ik (outcome)
    // per sample.  Stored once and reused four times below.
    let mut mu_m_base = vec![0.0_f64; n]; // α_m + Σ β_xk x_ik
    let mut mu_y_base = vec![0.0_f64; n]; // α_y + Σ γ_xk x_ik
    for i in 0..n {
        let mut s_m = beta_m[0]; // α_m
        let mut s_y = beta_y[0]; // α_y
        for j in 0..d {
            s_m += beta_m[2 + j] * x[i][j];
            s_y += beta_y[4 + j] * x[i][j];
        }
        mu_m_base[i] = s_m;
        mu_y_base[i] = s_y;
    }

    // M̂(t_med, x_i) for t_med ∈ {0, 1}.
    let mut delta_sum = 0.0_f64; // accumulator for δ̂(0) + δ̂(1)
    let mut zeta_sum = 0.0_f64; // accumulator for ζ̂(0) + ζ̂(1)
    for i in 0..n {
        let m_at_1 = mu_m_base[i] + beta_m[1]; // β_t · 1
        let m_at_0 = mu_m_base[i]; // β_t · 0

        // Y(t_out, M, x_i) = α_y + γ_t·t_out + γ_m·M + γ_tm·t_out·M + Σ γ_xk x_ik
        let y_of = |t_out: f64, m_val: f64| -> f64 {
            mu_y_base[i] + beta_y[1] * t_out + beta_y[2] * m_val + beta_y[3] * t_out * m_val
        };

        // δ̂(0): Y(0, M(1)) − Y(0, M(0))
        delta_sum += y_of(0.0, m_at_1) - y_of(0.0, m_at_0);
        // δ̂(1): Y(1, M(1)) − Y(1, M(0))
        delta_sum += y_of(1.0, m_at_1) - y_of(1.0, m_at_0);

        // ζ̂(0): Y(1, M(0)) − Y(0, M(0))
        zeta_sum += y_of(1.0, m_at_0) - y_of(0.0, m_at_0);
        // ζ̂(1): Y(1, M(1)) − Y(0, M(1))
        zeta_sum += y_of(1.0, m_at_1) - y_of(0.0, m_at_1);
    }
    // Each summand divides by n, and we average the two t-values.
    let acme = 0.5 * (delta_sum * inv_n);
    let ade = 0.5 * (zeta_sum * inv_n);
    (acme, ade)
}

/// Empirical `(lo, hi)` quantiles of `draws` using nearest-rank ordering.
fn empirical_ci(draws: &mut [f64], lo: f64, hi: f64) -> (f64, f64) {
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = draws.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let pick = |q: f64| -> f64 {
        let idx = ((q * n as f64).floor() as usize).min(n - 1);
        draws[idx]
    };
    (pick(lo), pick(hi))
}

// =====================================================================
// Ridge OLS plumbing — Gauss-Jordan with partial pivoting + diagonal
// extraction of (Z^T Z + λI)^{-1}.
// =====================================================================

/// Solve `(Z^T Z + λI) β = Z^T y` and additionally return the diagonal of
/// `(Z^T Z + λI)^{-1}`, used for the independent-components covariance
/// approximation.
fn ridge_solve_with_inv_diag(
    z: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    lambda: f64,
) -> CausalResult<(Vec<f64>, Vec<f64>)> {
    let mut ztz = vec![0.0_f64; p * p];
    let mut zty = vec![0.0_f64; p];
    for row in 0..n {
        for i in 0..p {
            let zri = z[row * p + i];
            for j in 0..p {
                ztz[i * p + j] += zri * z[row * p + j];
            }
            zty[i] += zri * y[row];
        }
    }
    for i in 0..p {
        ztz[i * p + i] += lambda;
    }
    let beta = gauss_jordan_solve(&ztz, &zty, p)?;
    let diag = gauss_jordan_inv_diag(&ztz, p)?;
    Ok((beta, diag))
}

/// Residual variance estimator `σ̂² = (1/(n − p)) · Σ_i (y_i − ŷ_i)²`,
/// with the standard degrees-of-freedom adjustment.  Returns `0.0` when
/// `n ≤ p` (degenerate case — caller checks `n` upstream).
fn residual_variance(z: &[f64], y: &[f64], beta: &[f64], n: usize, p: usize) -> f64 {
    if n <= p {
        return 0.0;
    }
    let mut rss = 0.0_f64;
    for i in 0..n {
        let mut pred = 0.0_f64;
        for j in 0..p {
            pred += beta[j] * z[i * p + j];
        }
        let r = y[i] - pred;
        rss += r * r;
    }
    let dof = (n - p) as f64;
    rss / dof
}

/// Solve `A β = b` by Gauss-Jordan elimination with partial pivoting.
fn gauss_jordan_solve(a: &[f64], b: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let cols = p + 1;
    let mut m = vec![0.0_f64; p * cols];
    for i in 0..p {
        for j in 0..p {
            m[i * cols + j] = a[i * p + j];
        }
        m[i * cols + p] = b[i];
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..p {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-15 {
            return Err(CausalError::MatrixSingular);
        }
        if piv != col {
            for k in 0..cols {
                m.swap(col * cols + k, piv * cols + k);
            }
        }
        let pv = m[col * cols + col];
        for k in 0..cols {
            m[col * cols + k] /= pv;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = m[r * cols + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in 0..cols {
                let v = m[col * cols + k];
                m[r * cols + k] -= f * v;
            }
        }
    }
    let mut x = vec![0.0_f64; p];
    for i in 0..p {
        x[i] = m[i * cols + p];
    }
    Ok(x)
}

/// Return the diagonal of `A^{-1}` for the symmetric positive-definite
/// `p × p` matrix `A`, computed by Gauss-Jordan elimination on the
/// `(A | I)` augmented matrix and extracting `A^{-1}_{j,j}`.
fn gauss_jordan_inv_diag(a: &[f64], p: usize) -> CausalResult<Vec<f64>> {
    let cols = 2 * p;
    let mut m = vec![0.0_f64; p * cols];
    for i in 0..p {
        for j in 0..p {
            m[i * cols + j] = a[i * p + j];
        }
        m[i * cols + p + i] = 1.0;
    }
    for col in 0..p {
        let mut piv = col;
        let mut best = m[col * cols + col].abs();
        for r in (col + 1)..p {
            let v = m[r * cols + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-15 {
            return Err(CausalError::MatrixSingular);
        }
        if piv != col {
            for k in 0..cols {
                m.swap(col * cols + k, piv * cols + k);
            }
        }
        let pv = m[col * cols + col];
        for k in 0..cols {
            m[col * cols + k] /= pv;
        }
        for r in 0..p {
            if r == col {
                continue;
            }
            let f = m[r * cols + col];
            if f.abs() < 1e-18 {
                continue;
            }
            for k in 0..cols {
                let v = m[col * cols + k];
                m[r * cols + k] -= f * v;
            }
        }
    }
    let mut diag = vec![0.0_f64; p];
    for i in 0..p {
        diag[i] = m[i * cols + p + i];
    }
    Ok(diag)
}

// =====================================================================
// Helpers re-exported for sibling test module.
// =====================================================================

#[cfg(test)]
#[inline]
pub(super) fn compute_acme_ade_for_tests(
    beta_m: &[f64],
    beta_y: &[f64],
    x: &[Vec<f64>],
    d: usize,
) -> (f64, f64) {
    compute_acme_ade(beta_m, beta_y, x, d)
}

#[cfg(test)]
#[inline]
pub(super) fn empirical_ci_for_tests(draws: &mut [f64], lo: f64, hi: f64) -> (f64, f64) {
    empirical_ci(draws, lo, hi)
}

// tests live in `mediation_tests.rs` (registered from `effect/mod.rs`).
