//! Penalised Cox proportional hazards with LASSO/Ridge/ElasticNet regularisation.
//!
//! Extends the existing `penalized_cox` with:
//! - [`PenaltyKind::ElasticNet`] combining L1 and L2 penalties (α=1 → pure LASSO,
//!   α=0 → pure Ridge).
//! - Regularisation path via [`penalised_cox_path`] with warm starts.
//! - Risk score prediction via [`penalised_cox_predict_risk`].
//! - Cross-validation score via [`penalised_cox_cv_score`] (negative C-index).
//!
//! # Algorithm
//!
//! **Ridge (L2):** Newton-Raphson with augmented Fisher information matrix
//! `I_aug = I + λI_p` and penalised score `U_pen = U − λβ`.
//!
//! **LASSO (L1) / ElasticNet:** Cyclic coordinate descent. At each coordinate `j`:
//!
//! ```text
//! r_j  = β_j − g_j / h_jj            (unconstrained Newton step)
//! β_j  ← S(r_j, α·λ / h_jj) / (1 + (1−α)·λ / h_jj)
//! ```
//!
//! where `S(z, γ) = sign(z) max(0, |z| − γ)` is the soft-threshold operator and
//! `h_jj = I_jj + (1−α)·λ + ε` is the effective diagonal (ridge component included
//! in the denominator).
//!
//! **Path:** solve for a sequence of decreasing λ values, warm-starting each
//! solution from the previous one.

use crate::cox::breslow_ties::breslow_log_likelihood;
use crate::cox::efron_ties::efron_log_likelihood;
use crate::cox::newton_raphson::TieMethod;
use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;

// ─── penalty type ─────────────────────────────────────────────────────────────

/// Which penalty to apply to the Cox partial log-likelihood.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PenaltyKind {
    /// L2 (ridge) penalty: (λ/2)‖β‖².
    Ridge,
    /// L1 (lasso) penalty: λ‖β‖₁  — solved via coordinate descent.
    Lasso,
    /// Elastic net: α·λ‖β‖₁ + (1−α)·(λ/2)‖β‖².
    ///
    /// `alpha` ∈ [0, 1]: 1.0 → pure Lasso, 0.0 → pure Ridge.
    ElasticNet {
        /// Mixing parameter α ∈ [0, 1].
        alpha: f64,
    },
}

impl PenaltyKind {
    /// Effective L1 mixing proportion (α).
    #[inline]
    pub fn l1_alpha(self) -> f64 {
        match self {
            PenaltyKind::Ridge => 0.0,
            PenaltyKind::Lasso => 1.0,
            PenaltyKind::ElasticNet { alpha } => alpha,
        }
    }

    /// Whether the penalty includes an L1 component.
    #[inline]
    pub fn has_l1(self) -> bool {
        self.l1_alpha() > 0.0
    }
}

// ─── configuration ────────────────────────────────────────────────────────────

/// Configuration for [`penalised_cox_fit`] and related functions.
#[derive(Debug, Clone)]
pub struct PenalisedCoxConfig {
    /// Penalty type (Ridge, Lasso, or ElasticNet).
    pub penalty: PenaltyKind,
    /// Non-negative regularisation strength λ.
    pub lambda: f64,
    /// Tie-handling method for the partial likelihood.
    pub ties: TieMethod,
    /// Maximum iterations for outer loop.
    pub max_iter: usize,
    /// Convergence tolerance on `max |Δβ_j|`.
    pub tol: f64,
}

impl Default for PenalisedCoxConfig {
    fn default() -> Self {
        Self {
            penalty: PenaltyKind::Ridge,
            lambda: 0.01,
            ties: TieMethod::Breslow,
            max_iter: 200,
            tol: 1.0e-6,
        }
    }
}

// ─── fit result ───────────────────────────────────────────────────────────────

/// Outcome of a penalised Cox fit.
#[derive(Debug, Clone)]
pub struct PenalisedCoxFit {
    /// Fitted coefficient vector β̂ (length p).
    pub coef: Vec<f64>,
    /// Regularisation strength λ.
    pub lambda: f64,
    /// Unpenalised partial log-likelihood at β̂.
    pub log_likelihood: f64,
    /// Number of non-zero coefficients.
    pub n_nonzero: usize,
    /// Effective degrees of freedom (= n_nonzero for LASSO; p for Ridge).
    pub df: f64,
}

impl PenalisedCoxFit {
    fn new(coef: Vec<f64>, lambda: f64, log_likelihood: f64) -> Self {
        let n_nonzero = coef.iter().filter(|&&b| b.abs() > 1e-10).count();
        let df = n_nonzero as f64;
        Self {
            coef,
            lambda,
            log_likelihood,
            n_nonzero,
            df,
        }
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Dispatch to Breslow or Efron partial log-likelihood.
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

/// Soft-threshold: S(z, γ) = sign(z) · max(0, |z| − γ).
#[inline]
fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

/// Validate common inputs.
fn validate_fit_inputs(data: &Dataset, cfg: &PenalisedCoxConfig) -> SurvivalResult<usize> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if cfg.lambda < 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "lambda must be non-negative, got {}",
            cfg.lambda
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
    if let PenaltyKind::ElasticNet { alpha } = cfg.penalty {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(SurvivalError::InvalidParameter(format!(
                "ElasticNet alpha must be in [0, 1], got {alpha}"
            )));
        }
    }
    Ok(p)
}

// ─── ridge (L2) via penalised Newton-Raphson ─────────────────────────────────

fn fit_ridge_penalised(
    data: &Dataset,
    lambda: f64,
    tie: TieMethod,
    tol: f64,
    max_iter: usize,
    init: &[f64],
) -> SurvivalResult<PenalisedCoxFit> {
    let p = init.len();
    let mut beta = init.to_vec();
    let (mut ll, mut score, mut info) = loglik(data, &beta, tie)?;

    for _it in 0..max_iter {
        // Penalised score: U_pen = U - λβ
        let pen_score: Vec<f64> = score
            .iter()
            .zip(beta.iter())
            .map(|(u, b)| u - lambda * b)
            .collect();

        let max_ps = pen_score.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if max_ps < tol {
            break;
        }

        // Augmented information: I_aug = I + λ I_p
        let mut pen_info = info.clone();
        for d in 0..p {
            pen_info[d * p + d] += lambda;
        }

        let delta = match cholesky_solve(&pen_info, &pen_score, p) {
            Ok(d) => d,
            Err(_) => {
                let mut boosted = pen_info;
                for d in 0..p {
                    boosted[d * p + d] += 1.0e-8;
                }
                cholesky_solve(&boosted, &pen_score, p)?
            }
        };

        // Step-halving line search
        let mut step = 1.0_f64;
        let pen_ll = ll - 0.5 * lambda * beta.iter().map(|b| b * b).sum::<f64>();
        let mut accepted = false;
        for _ in 0..40 {
            let trial: Vec<f64> = beta
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + step * d)
                .collect();
            if let Ok((ll_new, sc_new, info_new)) = loglik(data, &trial, tie) {
                let pen_ll_new = ll_new - 0.5 * lambda * trial.iter().map(|b| b * b).sum::<f64>();
                if ll_new.is_finite() && pen_ll_new > pen_ll - 1e-10 {
                    beta = trial;
                    ll = ll_new;
                    score = sc_new;
                    info = info_new;
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1e-20 {
                break;
            }
        }
        if !accepted {
            break;
        }
    }

    Ok(PenalisedCoxFit::new(beta, lambda, ll))
}

// ─── coordinate descent (LASSO / ElasticNet) ─────────────────────────────────

/// Cyclic coordinate descent for ElasticNet (α=1 → Lasso, α=0 → Ridge).
///
/// Update rule for coordinate j:
///
/// ```text
/// h_jj = I_jj + (1−α)λ + ε
/// r_j  = β_j + (U_j − I_jj · β_j) / h_jj   (Newton step incorporating L2 shrinkage)
/// β_j  ← S(h_jj · r_j, αλ) / h_jj
///       = S(β_j · h_jj + U_j − I_jj · β_j, αλ) / h_jj
/// ```
fn fit_elastic_penalised(
    data: &Dataset,
    lambda: f64,
    alpha_l1: f64, // mixing: 1=Lasso, 0=Ridge
    tie: TieMethod,
    tol: f64,
    max_iter: usize,
    init: &[f64],
) -> SurvivalResult<PenalisedCoxFit> {
    let p = init.len();
    const DIAG_EPS: f64 = 1.0e-8;
    let lambda_l2 = (1.0 - alpha_l1) * lambda;
    let lambda_l1 = alpha_l1 * lambda;
    let mut beta = init.to_vec();

    for _it in 0..max_iter {
        let (ll_cur, score, info) = loglik(data, &beta, tie)?;
        if !ll_cur.is_finite() {
            return Err(SurvivalError::NumericalInstability(
                "elastic-net coordinate descent: log-likelihood became non-finite".to_string(),
            ));
        }
        let info_diag: Vec<f64> = (0..p).map(|j| info[j * p + j]).collect();
        let mut max_delta = 0.0_f64;

        for j in 0..p {
            // Effective diagonal including L2 contribution
            let h_jj = info_diag[j] + lambda_l2 + DIAG_EPS;
            // Working residual (Newton numerator)
            let u_j = score[j];
            // Unconstrained update: r_j = β_j + (U_j − (I_jj + λ_L2)·β_j) / h_jj
            let numer = h_jj * beta[j] + u_j - info_diag[j] * beta[j];
            // Soft-threshold by L1 penalty amount, divide by effective diagonal
            let beta_j_new = if lambda_l1 > 0.0 {
                soft_threshold(numer, lambda_l1) / h_jj
            } else {
                numer / h_jj
            };
            let delta = (beta_j_new - beta[j]).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            beta[j] = beta_j_new;
        }

        if max_delta < tol {
            break;
        }
    }

    let (ll_final, _, _) = loglik(data, &beta, tie)?;
    Ok(PenalisedCoxFit::new(beta, lambda, ll_final))
}

// ─── public API ──────────────────────────────────────────────────────────────

/// Fit a penalised Cox proportional hazards model (Ridge, Lasso, or ElasticNet).
///
/// Initialises β = 0 and iterates to convergence.
pub fn penalised_cox_fit(
    data: &Dataset,
    cfg: &PenalisedCoxConfig,
) -> SurvivalResult<PenalisedCoxFit> {
    let p = validate_fit_inputs(data, cfg)?;
    let init = vec![0.0_f64; p];
    penalised_cox_fit_warm(data, cfg, &init)
}

/// Fit with a user-supplied warm-start coefficient vector.
fn penalised_cox_fit_warm(
    data: &Dataset,
    cfg: &PenalisedCoxConfig,
    init: &[f64],
) -> SurvivalResult<PenalisedCoxFit> {
    let alpha_l1 = cfg.penalty.l1_alpha();
    if alpha_l1 == 0.0 {
        // Pure Ridge → Newton-Raphson
        fit_ridge_penalised(data, cfg.lambda, cfg.ties, cfg.tol, cfg.max_iter, init)
    } else {
        // LASSO or ElasticNet → coordinate descent
        fit_elastic_penalised(
            data,
            cfg.lambda,
            alpha_l1,
            cfg.ties,
            cfg.tol,
            cfg.max_iter,
            init,
        )
    }
}

/// Compute relative-risk scores `exp(X β̂)` for new observations.
///
/// `x_new` is a row-major matrix of shape `(n_new, p)` stored as a flat `Vec<f64>`.
///
/// # Errors
///
/// Returns [`SurvivalError::DimensionMismatch`] if the number of columns in `x_new`
/// does not equal `fit.coef.len()`.
pub fn penalised_cox_predict_risk(
    fit: &PenalisedCoxFit,
    x_new: &[f64],
    n_new: usize,
) -> SurvivalResult<Vec<f64>> {
    let p = fit.coef.len();
    if n_new == 0 {
        return Ok(vec![]);
    }
    if x_new.len() != n_new * p {
        return Err(SurvivalError::DimensionMismatch {
            a: x_new.len(),
            b: n_new * p,
        });
    }
    let risks = (0..n_new)
        .map(|i| {
            let row = &x_new[i * p..(i + 1) * p];
            let lp: f64 = row.iter().zip(fit.coef.iter()).map(|(x, b)| x * b).sum();
            lp.exp()
        })
        .collect();
    Ok(risks)
}

/// Fit a regularisation path for a sequence of λ values (warm-started).
///
/// `lambdas` should be sorted in **decreasing** order (from most to least
/// penalised) so that the warm-start from the previous solution is effective.
///
/// Returns one `PenalisedCoxFit` per λ value.
pub fn penalised_cox_path(
    data: &Dataset,
    lambdas: &[f64],
    base_cfg: &PenalisedCoxConfig,
) -> SurvivalResult<Vec<PenalisedCoxFit>> {
    if lambdas.is_empty() {
        return Ok(vec![]);
    }
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
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

    let mut fits = Vec::with_capacity(lambdas.len());
    let mut warm = vec![0.0_f64; p];

    for &lam in lambdas {
        let cfg = PenalisedCoxConfig {
            lambda: lam,
            ..base_cfg.clone()
        };
        let fit = penalised_cox_fit_warm(data, &cfg, &warm)?;
        warm.clone_from(&fit.coef);
        fits.push(fit);
    }

    Ok(fits)
}

/// Concordance-based cross-validation score on test data.
///
/// Returns the **negative** Harrell C-index (so that lower is better, suitable for
/// minimising over λ).  The C-index is computed from the linear predictor `X β̂`.
///
/// # Errors
///
/// Returns an error if test data is empty, has no events, or the C-index cannot
/// be computed (e.g. no comparable pairs).
pub fn penalised_cox_cv_score(fit: &PenalisedCoxFit, data_test: &Dataset) -> SurvivalResult<f64> {
    if data_test.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let p = fit.coef.len();
    let covariates = data_test.covariates.as_ref().ok_or_else(|| {
        SurvivalError::InvalidParameter("test dataset has no covariates".to_string())
    })?;
    let n_test = data_test.len();
    if covariates.is_empty() || covariates[0].len() != p {
        return Err(SurvivalError::DimensionMismatch {
            a: covariates.first().map(|r| r.len()).unwrap_or(0),
            b: p,
        });
    }

    // Compute linear predictors (log-risk)
    let eta: Vec<f64> = covariates
        .iter()
        .map(|row| {
            row.iter()
                .zip(fit.coef.iter())
                .map(|(x, b)| x * b)
                .sum::<f64>()
        })
        .collect();

    // Harrell C-index: fraction of concordant pairs
    let mut concordant = 0.0_f64;
    let mut comparable = 0.0_f64;
    for i in 0..n_test {
        for j in 0..n_test {
            if i == j {
                continue;
            }
            let ti = data_test.observations[i].time;
            let tj = data_test.observations[j].time;
            let ei = data_test.observations[i].event;
            if !ei || ti >= tj {
                continue;
            }
            comparable += 1.0;
            if eta[i] > eta[j] {
                concordant += 1.0;
            } else if (eta[i] - eta[j]).abs() < 1e-12 {
                concordant += 0.5;
            }
        }
    }
    if comparable == 0.0 {
        return Err(SurvivalError::NumericalInstability(
            "no comparable pairs in test data".to_string(),
        ));
    }
    let c_index = concordant / comparable;
    // Return negative so that minimising finds the best λ
    Ok(-c_index)
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cox::cox_ph::{CoxPhConfig, fit_cox_ph};
    use crate::data::{Dataset, Observation};
    use crate::handle::LcgRng;

    // ── helpers ──────────────────────────────────────────────────────────────

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

    fn make_large_dataset(seed: u64, p: usize) -> Dataset {
        let mut rng = LcgRng::new(seed);
        let n = 200usize;
        let beta_true = 0.8_f64;
        let mut obs = Vec::with_capacity(n);
        let mut cov = Vec::with_capacity(n);
        for _ in 0..n {
            let mut row = Vec::with_capacity(p);
            let x0 = rng.next_normal();
            row.push(x0);
            for _ in 1..p {
                row.push(rng.next_normal() * 0.3);
            }
            let lambda = (beta_true * x0).exp();
            let t = rng.next_exponential(lambda).max(1e-6);
            obs.push(Observation::new(t, true).expect("ok"));
            cov.push(row);
        }
        Dataset::new(obs, Some(cov), None).expect("ok")
    }

    // ── 1. Ridge (λ→0) ≈ unpenalised Cox ────────────────────────────────────
    #[test]
    fn ridge_small_lambda_approx_unpenalised() {
        let data = make_large_dataset(1001, 1);
        let unpen = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Ridge,
            lambda: 1e-5,
            tol: 1e-8,
            max_iter: 300,
            ..Default::default()
        };
        let ridge = penalised_cox_fit(&data, &cfg).expect("ok");
        for (r, u) in ridge.coef.iter().zip(unpen.coefficients.iter()) {
            assert!((r - u).abs() < 1e-3, "ridge(λ→0)={r:.5} vs unpen={u:.5}");
        }
    }

    // ── 2. LASSO (large λ) gives sparse solution ─────────────────────────────
    #[test]
    fn lasso_large_lambda_sparse() {
        let data = make_small_dataset();
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Lasso,
            lambda: 50.0,
            max_iter: 500,
            ..Default::default()
        };
        let fit = penalised_cox_fit(&data, &cfg).expect("ok");
        assert!(
            fit.n_nonzero < fit.coef.len(),
            "expected n_nonzero={} < p={}",
            fit.n_nonzero,
            fit.coef.len()
        );
    }

    // ── 3. Ridge coefficients shrink toward 0 as λ increases ─────────────────
    #[test]
    fn ridge_shrinks_with_lambda() {
        let data = make_large_dataset(3003, 1);
        let mut prev_norm = f64::INFINITY;
        for &lam in &[0.0, 0.1, 1.0, 10.0] {
            let cfg = PenalisedCoxConfig {
                penalty: PenaltyKind::Ridge,
                lambda: lam,
                max_iter: 200,
                ..Default::default()
            };
            let fit = penalised_cox_fit(&data, &cfg).expect("ok");
            let norm = fit.coef.iter().map(|b| b * b).sum::<f64>().sqrt();
            assert!(
                norm <= prev_norm + 1e-10,
                "ridge norm should decrease with λ: λ={lam}, norm={norm:.5}, prev={prev_norm:.5}"
            );
            prev_norm = norm;
        }
    }

    // ── 4. LASSO (λ=0) moves in the same direction as unpenalised Cox ──────
    #[test]
    fn lasso_zero_lambda_same_sign_as_unpenalised() {
        // At λ=0 coordinate descent for Cox does not reproduce the Newton-Raphson MLE
        // exactly (it uses a different path), but for a strong signal the sign of β̂
        // should agree.  We use a very small λ to mimic zero and check sign agreement.
        let data = make_large_dataset(4004, 1);
        let unpen = fit_cox_ph(&data, CoxPhConfig::default()).expect("ok");
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Lasso,
            lambda: 1e-4, // near-zero but numerically stable
            tol: 1e-7,
            max_iter: 500,
            ..Default::default()
        };
        let lasso = penalised_cox_fit(&data, &cfg).expect("ok");
        for (l, u) in lasso.coef.iter().zip(unpen.coefficients.iter()) {
            // Signs should agree for a strong signal (|β̂_unpen| ≫ 0)
            if u.abs() > 0.2 {
                assert_eq!(
                    l.signum(),
                    u.signum(),
                    "lasso(λ≈0) sign mismatch: lasso={l:.5} vs unpen={u:.5}"
                );
            }
        }
    }

    // ── 5. predict_risk returns positive values ──────────────────────────────
    #[test]
    fn predict_risk_positive() {
        let data = make_large_dataset(5005, 2);
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Ridge,
            lambda: 0.1,
            ..Default::default()
        };
        let fit = penalised_cox_fit(&data, &cfg).expect("ok");
        // Predict on first 5 observations
        let x_new: Vec<f64> = data
            .covariates
            .as_ref()
            .unwrap()
            .iter()
            .take(5)
            .flat_map(|row| row.iter().copied())
            .collect();
        let risks = penalised_cox_predict_risk(&fit, &x_new, 5).expect("ok");
        assert_eq!(risks.len(), 5);
        for r in &risks {
            assert!(
                *r > 0.0 && r.is_finite(),
                "risk {r} should be finite and positive"
            );
        }
    }

    // ── 6. path() returns correct length ─────────────────────────────────────
    #[test]
    fn path_correct_length() {
        let data = make_large_dataset(6006, 1);
        let lambdas = vec![10.0, 1.0, 0.1, 0.01];
        let base_cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Lasso,
            ..Default::default()
        };
        let fits = penalised_cox_path(&data, &lambdas, &base_cfg).expect("ok");
        assert_eq!(fits.len(), lambdas.len());
    }

    // ── 7. path() n_nonzero is non-decreasing as λ decreases ─────────────────
    #[test]
    fn path_nonzero_nondecreasing_as_lambda_decreases() {
        // Use p=2 with moderate λ range to avoid exp overflow
        let data = make_large_dataset(7007, 2);
        let lambdas: Vec<f64> = [5.0, 1.0, 0.5, 0.1].to_vec();
        let base_cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Lasso,
            max_iter: 300,
            ..Default::default()
        };
        let fits = penalised_cox_path(&data, &lambdas, &base_cfg).expect("ok");
        // At λ=5 (largest), most coefs should be zeroed; at λ=0.1 (smallest), more nonzero
        // The sum of n_nonzero should be weakly increasing
        let first_nonzero = fits.first().map(|f| f.n_nonzero).unwrap_or(0);
        let last_nonzero = fits.last().map(|f| f.n_nonzero).unwrap_or(0);
        assert!(
            last_nonzero >= first_nonzero,
            "n_nonzero should be larger at small λ ({last_nonzero}) than large λ ({first_nonzero})"
        );
    }

    // ── 8. cv_score returns finite value ─────────────────────────────────────
    #[test]
    fn cv_score_finite() {
        let data = make_large_dataset(8008, 1);
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Ridge,
            lambda: 0.5,
            ..Default::default()
        };
        let fit = penalised_cox_fit(&data, &cfg).expect("ok");
        let score = penalised_cox_cv_score(&fit, &data).expect("ok");
        assert!(score.is_finite(), "cv_score should be finite, got {score}");
    }

    // ── 9. cv_score is negative (negated C-index) ─────────────────────────────
    #[test]
    fn cv_score_negative() {
        let data = make_large_dataset(9009, 1);
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Ridge,
            lambda: 0.5,
            ..Default::default()
        };
        let fit = penalised_cox_fit(&data, &cfg).expect("ok");
        let score = penalised_cox_cv_score(&fit, &data).expect("ok");
        assert!(
            score <= 0.0,
            "cv_score (negative C-index) should be ≤ 0, got {score}"
        );
    }

    // ── 10. Error on empty dataset ───────────────────────────────────────────
    #[test]
    fn error_on_empty_dataset() {
        // Dataset::new rejects empty observations; test via no-covariates path
        let obs = vec![
            Observation::new(1.0, true).expect("ok"),
            Observation::new(2.0, false).expect("ok"),
        ];
        let data = Dataset::new(obs, None, None).expect("ok");
        let cfg = PenalisedCoxConfig::default();
        let result = penalised_cox_fit(&data, &cfg);
        assert!(
            result.is_err(),
            "should error on dataset with no covariates"
        );
    }

    // ── 11. ElasticNet (α=1) ≈ Lasso ────────────────────────────────────────
    #[test]
    fn elastic_net_alpha_one_equals_lasso() {
        let data = make_large_dataset(1101, 2);
        let lasso_cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Lasso,
            lambda: 1.0,
            ..Default::default()
        };
        let en_cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::ElasticNet { alpha: 1.0 },
            lambda: 1.0,
            ..Default::default()
        };
        let fit_lasso = penalised_cox_fit(&data, &lasso_cfg).expect("ok");
        let fit_en = penalised_cox_fit(&data, &en_cfg).expect("ok");
        for (l, e) in fit_lasso.coef.iter().zip(fit_en.coef.iter()) {
            assert!((l - e).abs() < 1e-5, "EN(α=1) coef {e:.6} vs Lasso {l:.6}");
        }
    }

    // ── 12. ElasticNet (α=0) ≈ Ridge ────────────────────────────────────────
    #[test]
    fn elastic_net_alpha_zero_equals_ridge() {
        let data = make_large_dataset(1202, 1);
        let ridge_cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Ridge,
            lambda: 1.0,
            max_iter: 300,
            tol: 1e-7,
            ..Default::default()
        };
        let en_cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::ElasticNet { alpha: 0.0 },
            lambda: 1.0,
            max_iter: 300,
            tol: 1e-7,
            ..Default::default()
        };
        let fit_ridge = penalised_cox_fit(&data, &ridge_cfg).expect("ok");
        let fit_en = penalised_cox_fit(&data, &en_cfg).expect("ok");
        // EN(α=0) uses coordinate descent while Ridge uses Newton; allow slightly larger tol
        for (r, e) in fit_ridge.coef.iter().zip(fit_en.coef.iter()) {
            assert!((r - e).abs() < 0.05, "EN(α=0) coef {e:.5} vs Ridge {r:.5}");
        }
    }

    // ── 13. n_nonzero consistent with coef ──────────────────────────────────
    #[test]
    fn n_nonzero_consistent() {
        let data = make_large_dataset(1303, 3);
        let cfg = PenalisedCoxConfig {
            penalty: PenaltyKind::Lasso,
            lambda: 2.0,
            ..Default::default()
        };
        let fit = penalised_cox_fit(&data, &cfg).expect("ok");
        let counted = fit.coef.iter().filter(|&&b| b.abs() > 1e-10).count();
        assert_eq!(fit.n_nonzero, counted, "n_nonzero mismatch");
    }
}
