//! Penalized Cox proportional hazards models.
//!
//! Implements two classical penalty regimes on top of the partial log-likelihood:
//!
//! - **Ridge (L2):**  maximise `ℓ(β) − (λ/2)‖β‖²`
//!   Adds `λ` to the diagonal of the Fisher information and subtracts `λ β` from
//!   the score, giving a modified Newton-Raphson step that is always well-conditioned.
//!
//! - **Lasso (L1):**  maximise `ℓ(β) − λ‖β‖₁`
//!   Uses cyclic coordinate descent with a second-order Taylor approximation of the
//!   partial likelihood.  Each coordinate is updated by soft-thresholding the
//!   unconstrained Newton step, producing sparse solutions for large λ.
//!
//! The function [`fit_penalized_cox`] is the top-level entry point.

use crate::cox::breslow_ties::breslow_log_likelihood;
use crate::cox::efron_ties::efron_log_likelihood;
use crate::cox::newton_raphson::TieMethod;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;

// ─── public types ─────────────────────────────────────────────────────────────

/// Which penalty to apply to the Cox partial log-likelihood.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyType {
    /// L2 (ridge) penalty: `(λ/2)‖β‖²`.
    Ridge,
    /// L1 (lasso) penalty: `λ‖β‖₁`, fitted via coordinate descent.
    Lasso,
}

/// Configuration for [`fit_penalized_cox`].
#[derive(Debug, Clone, Copy)]
pub struct PenalizedCoxConfig {
    /// Penalty type.
    pub penalty: PenaltyType,
    /// Non-negative regularisation strength λ.
    pub lambda: f64,
    /// Tie-handling method for the partial likelihood.
    pub tie: TieMethod,
    /// Convergence tolerance (on max |Δβ| for lasso; max |score| for ridge).
    pub tol: f64,
    /// Maximum outer iterations.
    pub max_iter: usize,
}

impl Default for PenalizedCoxConfig {
    fn default() -> Self {
        Self {
            penalty: PenaltyType::Ridge,
            lambda: 0.01,
            tie: TieMethod::Breslow,
            tol: 1.0e-6,
            max_iter: 100,
        }
    }
}

/// Outcome of a penalised Cox fit.
#[derive(Debug, Clone)]
pub struct PenalizedCoxFit {
    /// Coefficient vector β̂ (length p).
    pub coefficients: Vec<f64>,
    /// Unpenalised partial log-likelihood at β̂.
    pub log_likelihood: f64,
    /// Penalty term evaluated at β̂: `(λ/2)‖β̂‖²` for ridge, `λ‖β̂‖₁` for lasso.
    pub penalty: f64,
    /// Number of outer iterations consumed.
    pub iterations: usize,
    /// Whether the algorithm met the convergence criterion.
    pub converged: bool,
}

// ─── helpers ──────────────────────────────────────────────────────────────────

/// Dispatch to Breslow or Efron likelihood, returning `(loglik, score, info)`.
#[inline]
fn loglik(
    data: &Dataset,
    beta: &[f64],
    tie: TieMethod,
) -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
    match tie {
        TieMethod::Breslow => breslow_log_likelihood(data, beta),
        TieMethod::Efron => efron_log_likelihood(data, beta),
    }
}

/// Soft-threshold operator: `sign(x) * (|x| − threshold).max(0)`.
#[inline]
fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

// ─── ridge (L2) via Newton-Raphson ────────────────────────────────────────────

/// Maximise `ℓ(β) − (λ/2)‖β‖²` using a penalised Newton-Raphson loop.
///
/// The penalised score is `U_ridge = U − λ β` and the penalised information
/// matrix is `I_ridge = I + λ Iₚ` (adding λ to each diagonal element).
/// The step equation `I_ridge Δβ = U_ridge` is solved via Cholesky decomposition.
fn fit_ridge(
    data: &Dataset,
    lambda: f64,
    tie: TieMethod,
    tol: f64,
    max_iter: usize,
) -> SurvivalResult<PenalizedCoxFit> {
    let p = data.n_features();
    let mut beta = vec![0.0_f64; p];

    let (mut ll, mut score, mut info) = loglik(data, &beta, tie)?;

    let mut converged = false;
    let mut iter = 0usize;

    for it in 0..max_iter {
        iter = it + 1;

        // Penalised score: U_ridge = U - λ β
        let penalised_score: Vec<f64> = score
            .iter()
            .zip(beta.iter())
            .map(|(u, b)| u - lambda * b)
            .collect();

        // Check convergence on max |penalised score|
        let max_ps = penalised_score
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_ps < tol {
            converged = true;
            break;
        }

        // Penalised information: I_ridge = I + λ Iₚ
        let mut penalised_info = info.clone();
        for d in 0..p {
            penalised_info[d * p + d] += lambda;
        }

        // Solve (I + λ I_p) Δβ = U_ridge
        let delta = match cholesky_solve(&penalised_info, &penalised_score, p) {
            Ok(d) => d,
            Err(_) => {
                // Add a tiny ridge boost and retry once
                let mut boosted = penalised_info.clone();
                for d in 0..p {
                    boosted[d * p + d] += 1.0e-8;
                }
                cholesky_solve(&boosted, &penalised_score, p)?
            }
        };

        // Line search with step-halving
        let mut step = 1.0_f64;
        let mut accepted = false;
        let penalised_ll = ll - 0.5 * lambda * beta.iter().map(|b| b * b).sum::<f64>();

        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();

            if let Ok((ll_new, sc_new, info_new)) = loglik(data, &trial, tie) {
                let penalised_ll_new =
                    ll_new - 0.5 * lambda * trial.iter().map(|b| b * b).sum::<f64>();
                if ll_new.is_finite() && penalised_ll_new > penalised_ll - 1.0e-10 {
                    beta = trial;
                    ll = ll_new;
                    score = sc_new;
                    info = info_new;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }

        if !accepted {
            break;
        }
    }

    // Final convergence check in case we exited via !accepted
    if !converged {
        let penalised_score: Vec<f64> = score
            .iter()
            .zip(beta.iter())
            .map(|(u, b)| u - lambda * b)
            .collect();
        let max_ps = penalised_score
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_ps < tol {
            converged = true;
        }
    }

    let penalty_val = 0.5 * lambda * beta.iter().map(|b| b * b).sum::<f64>();

    Ok(PenalizedCoxFit {
        coefficients: beta,
        log_likelihood: ll,
        penalty: penalty_val,
        iterations: iter,
        converged,
    })
}

// ─── lasso (L1) via cyclic coordinate descent ─────────────────────────────────

/// Maximise `ℓ(β) − λ‖β‖₁` using cyclic coordinate descent with a
/// second-order Taylor approximation of the partial log-likelihood.
///
/// At each outer iteration we compute the full score `U` and Fisher information
/// diagonal `I_jj`, then update each coordinate in turn:
///
/// ```text
/// u_j  = U_j − (I_jj + ε) β_j           (working response for coord j)
/// β_j* = β_j + u_j / (I_jj + ε)         (unconstrained Newton step)
/// β_j  ← soft_threshold(β_j*, λ / (I_jj + ε))
/// ```
///
/// Convergence: `max |Δβ_j| < tol` over a full cycle through all p coordinates.
fn fit_lasso(
    data: &Dataset,
    lambda: f64,
    tie: TieMethod,
    tol: f64,
    max_iter: usize,
) -> SurvivalResult<PenalizedCoxFit> {
    let p = data.n_features();
    // Numerical stability floor for diagonal information elements
    const DIAG_EPS: f64 = 1.0e-8;

    let mut beta = vec![0.0_f64; p];
    let mut converged = false;
    let mut iter = 0usize;

    for it in 0..max_iter {
        iter = it + 1;

        // Evaluate partial log-likelihood at current β
        let (ll_cur, score, info) = loglik(data, &beta, tie)?;

        if !ll_cur.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "lasso coordinate descent: log-likelihood became non-finite".to_string(),
            ));
        }

        // Extract information diagonal
        let info_diag: Vec<f64> = (0..p).map(|j| info[j * p + j]).collect();

        // Cyclic coordinate update — track max change for convergence
        let mut max_delta = 0.0_f64;

        for j in 0..p {
            let i_jj = info_diag[j] + DIAG_EPS;
            // Working response: contribution to score excluding the current coordinate's penalty
            let u_j = score[j] - i_jj * beta[j];
            // Unconstrained Newton step for coordinate j
            let beta_j_star = beta[j] + u_j / i_jj;
            // Apply soft-thresholding
            let threshold = lambda / i_jj;
            let beta_j_new = soft_threshold(beta_j_star, threshold);
            let delta = (beta_j_new - beta[j]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            beta[j] = beta_j_new;
        }

        if max_delta < tol {
            converged = true;
            break;
        }
    }

    // Evaluate final log-likelihood
    let (ll_final, _, _) = loglik(data, &beta, tie)?;

    let penalty_val = lambda * beta.iter().map(|b| b.abs()).sum::<f64>();

    Ok(PenalizedCoxFit {
        coefficients: beta,
        log_likelihood: ll_final,
        penalty: penalty_val,
        iterations: iter,
        converged,
    })
}

// ─── public entry point ───────────────────────────────────────────────────────

/// Fit a penalised Cox proportional hazards model.
///
/// Dispatches to `fit_ridge` (Newton-Raphson) or `fit_lasso` (coordinate
/// descent) depending on `config.penalty`.
///
/// # Errors
///
/// Returns [`SurvivalError::InvalidParameter`] for negative λ or a dataset
/// without covariates, [`SurvivalError::EmptyDataset`] for an empty dataset,
/// and various numerical errors if the algorithm encounters instability.
pub fn fit_penalized_cox(
    data: &Dataset,
    config: &PenalizedCoxConfig,
) -> SurvivalResult<PenalizedCoxFit> {
    // Validate inputs
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if config.lambda < 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "lambda must be non-negative, got {}",
            config.lambda
        )));
    }
    let p = data.n_features();
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "dataset has no covariates".to_string(),
        ));
    }
    if data.n_events() == 0 {
        return Err(SurvivalError::NoEvents);
    }

    match config.penalty {
        PenaltyType::Ridge => {
            fit_ridge(data, config.lambda, config.tie, config.tol, config.max_iter)
        }
        PenaltyType::Lasso => {
            fit_lasso(data, config.lambda, config.tie, config.tol, config.max_iter)
        }
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cox::cox_ph::{CoxPhConfig, fit_cox_ph};
    use crate::data::{Dataset, Observation};
    use crate::handle::LcgRng;

    /// Build a small synthetic dataset with 5 observations and 2 covariates.
    /// Three events (status true) and two censored.
    fn make_small_dataset() -> Dataset {
        let obs = vec![
            Observation::new(5.0, true).expect("ok"),
            Observation::new(3.0, true).expect("ok"),
            Observation::new(8.0, false).expect("ok"),
            Observation::new(2.0, true).expect("ok"),
            Observation::new(6.0, false).expect("ok"),
        ];
        let cov = vec![
            vec![1.0, 0.5],
            vec![0.0, -1.0],
            vec![-1.0, 0.3],
            vec![2.0, -0.5],
            vec![0.5, 1.0],
        ];
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    /// Build a large synthetic dataset (n=200, 1 covariate) using LcgRng so that
    /// the unpenalised Cox PH has a well-conditioned information matrix.
    fn make_large_dataset(seed: u64) -> Dataset {
        let mut rng = LcgRng::new(seed);
        let n = 200usize;
        let beta_true = 0.8_f64;
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            let lambda = (beta_true * x).exp();
            let t = rng.next_exponential(lambda).max(1.0e-6);
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(vec![x]);
        }
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    // ── 1. ridge_shrinks_coefficients_toward_zero ─────────────────────────────

    #[test]
    fn ridge_shrinks_coefficients_toward_zero() {
        let data = make_large_dataset(1001);

        // Unpenalised Cox
        let unpen = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");

        // Ridge with large penalty
        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 10.0,
            ..Default::default()
        };
        let ridge = fit_penalized_cox(&data, &cfg).expect("ok");

        let norm_unpen = unpen.coefficients.iter().map(|b| b * b).sum::<f64>().sqrt();
        let norm_ridge = ridge.coefficients.iter().map(|b| b * b).sum::<f64>().sqrt();
        assert!(
            norm_ridge < norm_unpen,
            "ridge norm={norm_ridge} should be < unpenalised norm={norm_unpen}"
        );
    }

    // ── 2. ridge_lambda_zero_matches_unpenalized ──────────────────────────────

    #[test]
    fn ridge_lambda_zero_matches_unpenalized() {
        let data = make_large_dataset(2002);

        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 0.0,
            tol: 1.0e-8,
            max_iter: 200,
            ..Default::default()
        };
        let ridge = fit_penalized_cox(&data, &cfg).expect("ok");

        let cox_cfg = CoxPhConfig {
            tol: 1.0e-8,
            max_iter: 200,
            ..Default::default()
        };
        let unpen = fit_cox_ph(&data, cox_cfg).expect("ok");

        assert_eq!(ridge.coefficients.len(), unpen.coefficients.len());
        for (r, u) in ridge.coefficients.iter().zip(unpen.coefficients.iter()) {
            assert!(
                (r - u).abs() < 1.0e-4,
                "lambda=0 ridge={r}, unpenalised={u}"
            );
        }
    }

    // ── 3. lasso_shrinks_toward_zero ─────────────────────────────────────────

    #[test]
    fn lasso_shrinks_toward_zero() {
        let data = make_large_dataset(3003);

        let unpen = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");

        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Lasso,
            lambda: 5.0,
            ..Default::default()
        };
        let lasso = fit_penalized_cox(&data, &cfg).expect("ok");

        let l1_unpen: f64 = unpen.coefficients.iter().map(|b| b.abs()).sum();
        let l1_lasso: f64 = lasso.coefficients.iter().map(|b| b.abs()).sum();
        assert!(
            l1_lasso < l1_unpen,
            "lasso l1={l1_lasso} should be < unpenalised l1={l1_unpen}"
        );
    }

    // ── 4. lasso_sparsity_at_large_lambda ────────────────────────────────────

    #[test]
    fn lasso_sparsity_at_large_lambda() {
        let data = make_small_dataset();

        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Lasso,
            lambda: 100.0,
            max_iter: 500,
            ..Default::default()
        };
        let lasso = fit_penalized_cox(&data, &cfg).expect("ok");

        // With very large λ, at least one coefficient should be essentially zero
        let has_sparse = lasso.coefficients.iter().any(|b| b.abs() < 1.0e-6);
        assert!(
            has_sparse,
            "expected sparsity with lambda=100, got {:?}",
            lasso.coefficients
        );
    }

    // ── 5. ridge_converges ────────────────────────────────────────────────────

    #[test]
    fn ridge_converges() {
        let data = make_large_dataset(5005);
        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 1.0,
            ..Default::default()
        };
        let fit = fit_penalized_cox(&data, &cfg).expect("ok");
        assert!(
            fit.converged,
            "ridge should converge on well-conditioned data"
        );
    }

    // ── 6. lasso_converges ────────────────────────────────────────────────────

    #[test]
    fn lasso_converges() {
        let data = make_large_dataset(6006);
        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Lasso,
            lambda: 1.0,
            // Lasso coordinate descent on n=200 data with moderate λ needs more
            // outer iterations because each per-coordinate Newton step is small.
            max_iter: 1000,
            ..Default::default()
        };
        let fit = fit_penalized_cox(&data, &cfg).expect("ok");
        assert!(
            fit.converged,
            "lasso should converge on well-conditioned data"
        );
    }

    // ── 7. penalized_fit_log_likelihood_finite ────────────────────────────────

    #[test]
    fn penalized_fit_log_likelihood_finite() {
        let data = make_small_dataset();

        for &penalty in &[PenaltyType::Ridge, PenaltyType::Lasso] {
            let cfg = PenalizedCoxConfig {
                penalty,
                lambda: 0.5,
                ..Default::default()
            };
            let fit = fit_penalized_cox(&data, &cfg).expect("ok");
            assert!(
                fit.log_likelihood.is_finite(),
                "log-likelihood should be finite for {penalty:?}"
            );
        }
    }

    // ── 8. ridge_positive_lambda_reduces_penalized_objective ─────────────────

    #[test]
    fn ridge_positive_lambda_reduces_likelihood() {
        // The penalised objective ℓ(β̂_pen) − penalty ≤ ℓ(β̂_unpen) by optimality
        // of the unpenalised estimate; equivalently, the unpenalised log-likelihood
        // at the penalised solution should be ≤ that at the unpenalised solution.
        let data = make_large_dataset(8008);

        let unpen = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");

        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 2.0,
            tol: 1.0e-8,
            max_iter: 200,
            ..Default::default()
        };
        let ridge = fit_penalized_cox(&data, &cfg).expect("ok");

        // Penalised solution sacrifices some likelihood for regularisation
        assert!(
            ridge.log_likelihood <= unpen.log_likelihood + 1.0e-6,
            "ridge loglik={} should be ≤ unpenalised loglik={} (tol 1e-6)",
            ridge.log_likelihood,
            unpen.log_likelihood
        );
    }

    // ── 9. invalid_lambda_returns_error ──────────────────────────────────────

    #[test]
    fn invalid_lambda_returns_error() {
        let data = make_small_dataset();
        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: -0.1,
            ..Default::default()
        };
        let result = fit_penalized_cox(&data, &cfg);
        assert!(result.is_err(), "negative lambda should return an error");
        assert!(matches!(result, Err(SurvivalError::InvalidParameter(_))));
    }

    // ── 10. empty_dataset_returns_error ──────────────────────────────────────

    #[test]
    fn empty_dataset_returns_error() {
        // Dataset::new rejects empty observations, so we test via a no-covariates path
        // by constructing a dataset that has no covariates (p = 0).
        let obs = vec![
            Observation::new(1.0, true).expect("ok"),
            Observation::new(2.0, false).expect("ok"),
        ];
        // Dataset with no covariates triggers InvalidParameter("no covariates")
        let data = Dataset::new(obs, None, None).expect("ok");
        let cfg = PenalizedCoxConfig::default();
        let result = fit_penalized_cox(&data, &cfg);
        assert!(result.is_err());
    }

    // ── 11. penalty_value_is_non_negative ────────────────────────────────────

    #[test]
    fn penalty_value_is_non_negative() {
        let data = make_small_dataset();
        for &penalty in &[PenaltyType::Ridge, PenaltyType::Lasso] {
            let cfg = PenalizedCoxConfig {
                penalty,
                lambda: 1.0,
                ..Default::default()
            };
            let fit = fit_penalized_cox(&data, &cfg).expect("ok");
            assert!(
                fit.penalty >= 0.0,
                "penalty value must be non-negative, got {}",
                fit.penalty
            );
        }
    }

    // ── 12. ridge_efron_also_converges ────────────────────────────────────────

    #[test]
    fn ridge_efron_also_converges() {
        let data = make_large_dataset(1200);
        let cfg = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 0.5,
            tie: TieMethod::Efron,
            ..Default::default()
        };
        let fit = fit_penalized_cox(&data, &cfg).expect("ok");
        assert!(fit.converged);
        assert!(fit.log_likelihood.is_finite());
    }

    // ── 13. ridge_penalty_grows_with_lambda ───────────────────────────────────

    #[test]
    fn ridge_penalty_grows_with_lambda() {
        let data = make_large_dataset(1300);
        let cfg_small = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 0.1,
            ..Default::default()
        };
        let cfg_large = PenalizedCoxConfig {
            penalty: PenaltyType::Ridge,
            lambda: 10.0,
            ..Default::default()
        };
        let fit_small = fit_penalized_cox(&data, &cfg_small).expect("ok");
        let fit_large = fit_penalized_cox(&data, &cfg_large).expect("ok");
        // Larger λ → smaller ‖β‖²  (more regularisation → more shrinkage)
        let norm_small: f64 = fit_small.coefficients.iter().map(|b| b * b).sum::<f64>();
        let norm_large: f64 = fit_large.coefficients.iter().map(|b| b * b).sum::<f64>();
        assert!(
            norm_large <= norm_small + 1.0e-10,
            "larger lambda should give smaller or equal ‖β‖²: small={norm_small}, large={norm_large}"
        );
    }
}
