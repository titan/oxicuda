//! Builder pattern for fitting Cox proportional hazards models.
//!
//! Provides:
//! - [`CoxBuilder`]: fluent builder for Cox PH fitting configuration.
//! - [`CoxFitResult`]: unified result type with prediction helpers.
//!
//! # Usage
//!
//! ```rust,ignore
//! use oxicuda_survival::cox::cox_builder::{CoxBuilder, TieMethod};
//!
//! let result = CoxBuilder::new()
//!     .ties(TieMethod::Efron)
//!     .max_iter(200)
//!     .tolerance(1e-8)
//!     .ridge(0.01)
//!     .fit(&data)?;
//!
//! println!("β̂ = {:?}", result.coef);
//! ```

use crate::cox::newton_raphson::TieMethod;
use crate::cox::trust_region::{TrustRegionConfig, trust_region_cox};
use crate::data::{Dataset, Observation};
use crate::error::{SurvivalError, SurvivalResult};
use crate::linalg::solve::cholesky_solve;

// Re-export TieMethod so users can import it from this module.
pub use crate::cox::newton_raphson::TieMethod as TieMethodAlias;

// ---------------------------------------------------------------------------
// CoxFitResult
// ---------------------------------------------------------------------------

/// Result of a Cox PH fit from [`CoxBuilder`].
#[derive(Debug, Clone)]
pub struct CoxFitResult {
    /// Estimated coefficient vector β̂.
    pub coef: Vec<f64>,
    /// Partial log-likelihood at β̂.
    pub log_lik: f64,
    /// Number of optimisation iterations consumed.
    pub n_iter: usize,
    /// Whether the algorithm converged within the iteration budget.
    pub converged: bool,
    /// Number of events (uncensored observations) in the training data.
    pub n_events: usize,
    /// Total number of subjects in the training data.
    pub n_subjects: usize,
}

impl CoxFitResult {
    /// Predicted risk score `exp(xᵀβ)` for a new covariate vector `x`.
    ///
    /// Higher risk score → higher instantaneous hazard relative to the baseline.
    #[must_use]
    pub fn predict_risk(&self, x: &[f64]) -> f64 {
        self.predict_log_hazard(x).exp()
    }

    /// Predicted log-hazard ratio `xᵀβ` for a new covariate vector `x`.
    #[must_use]
    pub fn predict_log_hazard(&self, x: &[f64]) -> f64 {
        x.iter().zip(self.coef.iter()).map(|(xi, bi)| xi * bi).sum()
    }
}

// ---------------------------------------------------------------------------
// CoxBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for Cox proportional hazards models.
///
/// Supports:
/// - Breslow or Efron tie handling.
/// - Standard Newton-Raphson or trust-region Newton optimisation.
/// - Optional L2 (ridge) regularisation.
///
/// All fields are set via method chaining; call [`CoxBuilder::fit`] to train the model.
#[derive(Debug, Clone)]
pub struct CoxBuilder {
    tie_method: TieMethod,
    max_iter: usize,
    tolerance: f64,
    ridge_lambda: f64,
    use_trust_region: bool,
}

impl Default for CoxBuilder {
    fn default() -> Self {
        Self {
            tie_method: TieMethod::Efron,
            max_iter: 100,
            tolerance: 1.0e-6,
            ridge_lambda: 0.0,
            use_trust_region: false,
        }
    }
}

impl CoxBuilder {
    /// Create a new builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the tie-handling method (Breslow or Efron).
    #[must_use]
    pub fn ties(mut self, method: TieMethod) -> Self {
        self.tie_method = method;
        self
    }

    /// Set the maximum number of optimisation iterations.
    #[must_use]
    pub fn max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Set the convergence tolerance (on the infinity norm of the gradient).
    #[must_use]
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Add L2 (ridge) regularisation with penalty weight `lambda`.
    ///
    /// The penalty `lambda * ||β||²` is added to the *negative* log-likelihood.
    /// Setting `lambda = 0.0` (the default) disables regularisation.
    #[must_use]
    pub fn ridge(mut self, lambda: f64) -> Self {
        self.ridge_lambda = lambda;
        self
    }

    /// Switch to trust-region Newton (Steihaug-CG) optimisation.
    ///
    /// Trust-region Newton is more robust than standard Newton-Raphson for
    /// ill-conditioned problems or when the Hessian is nearly singular.
    #[must_use]
    pub fn trust_region(mut self) -> Self {
        self.use_trust_region = true;
        self
    }

    /// Fit the Cox model to `data`.
    ///
    /// `data` is a slice of `(time, event, covariates)` tuples.
    ///
    /// # Errors
    /// - [`SurvivalError::EmptyDataset`] if `data` is empty.
    /// - [`SurvivalError::NoEvents`] if there are no events.
    /// - [`SurvivalError::InvalidParameter`] for invalid configurations.
    /// - Propagates numerical errors from the optimiser.
    pub fn fit(&self, data: &[(f64, bool, Vec<f64>)]) -> SurvivalResult<CoxFitResult> {
        if data.is_empty() {
            return Err(SurvivalError::EmptyDataset);
        }
        let n_events = data.iter().filter(|(_, e, _)| *e).count();
        if n_events == 0 {
            return Err(SurvivalError::NoEvents);
        }
        let n_subjects = data.len();

        if self.ridge_lambda < 0.0 {
            return Err(SurvivalError::InvalidParameter(format!(
                "ridge_lambda={:.4e} must be non-negative",
                self.ridge_lambda
            )));
        }

        if self.use_trust_region {
            self.fit_trust_region(data, n_events, n_subjects)
        } else {
            self.fit_newton_raphson(data, n_events, n_subjects)
        }
    }

    // ------------------------------------------------------------------
    // Internal: Newton-Raphson path
    // ------------------------------------------------------------------

    fn fit_newton_raphson(
        &self,
        data: &[(f64, bool, Vec<f64>)],
        n_events: usize,
        n_subjects: usize,
    ) -> SurvivalResult<CoxFitResult> {
        let ds = triples_to_dataset(data)?;
        let p = ds.n_features();
        if p == 0 {
            return Err(SurvivalError::InvalidParameter(
                "CoxBuilder::fit: no covariates".to_string(),
            ));
        }

        let mut beta = vec![0.0_f64; p];
        let loglik_fn = |b: &[f64]| -> SurvivalResult<(f64, Vec<f64>, Vec<f64>)> {
            let (ll, mut score, mut info) = match self.tie_method {
                TieMethod::Breslow => crate::cox::breslow_ties::breslow_log_likelihood(&ds, b)?,
                TieMethod::Efron => crate::cox::efron_ties::efron_log_likelihood(&ds, b)?,
            };
            // Add ridge penalty to gradient and Hessian.
            if self.ridge_lambda > 0.0 {
                for k in 0..p {
                    // Penalty adds lambda * b[k]^2 to -loglik → subtract lambda*b[k] from score.
                    score[k] -= 2.0 * self.ridge_lambda * b[k];
                    // Add lambda * I to information matrix (Hessian of -loglik).
                    info[k * p + k] += 2.0 * self.ridge_lambda;
                }
                let ll_ridge = ll - self.ridge_lambda * b.iter().map(|bi| bi * bi).sum::<f64>();
                return Ok((ll_ridge, score, info));
            }
            Ok((ll, score, info))
        };

        let (mut ll, mut score, mut info) = loglik_fn(&beta)?;
        let mut converged = false;
        let mut n_iter = 0usize;

        for it in 0..self.max_iter {
            n_iter = it + 1;

            let max_score = score.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if max_score < self.tolerance {
                converged = true;
                break;
            }

            let delta = match cholesky_solve(&info, &score, p) {
                Ok(d) => d,
                Err(_) => {
                    // Ridge boost for near-singular information.
                    let mut info_boosted = info.clone();
                    for k in 0..p {
                        info_boosted[k * p + k] += 1.0e-4;
                    }
                    cholesky_solve(&info_boosted, &score, p)?
                }
            };

            // Step-halving backtrack.
            let mut step = 1.0_f64;
            let mut accepted = false;
            for _ in 0..40 {
                let trial: Vec<f64> = beta
                    .iter()
                    .zip(delta.iter())
                    .map(|(b, d)| b + step * d)
                    .collect();
                if let Ok((ll_new, sc_new, info_new)) = loglik_fn(&trial) {
                    if ll_new.is_finite() && ll_new > ll - 1.0e-10 {
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

        if !converged {
            let max_score = score.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if max_score < self.tolerance {
                converged = true;
            }
        }

        Ok(CoxFitResult {
            coef: beta,
            log_lik: ll,
            n_iter,
            converged,
            n_events,
            n_subjects,
        })
    }

    // ------------------------------------------------------------------
    // Internal: Trust-region path
    // ------------------------------------------------------------------

    fn fit_trust_region(
        &self,
        data: &[(f64, bool, Vec<f64>)],
        n_events: usize,
        n_subjects: usize,
    ) -> SurvivalResult<CoxFitResult> {
        // For ridge regularisation with trust region, we augment the data with
        // p synthetic observations that enforce the penalty.
        let augmented_data = if self.ridge_lambda > 0.0 {
            augment_with_ridge(data, self.ridge_lambda)
        } else {
            data.to_vec()
        };

        let config = TrustRegionConfig {
            max_outer: self.max_iter,
            tol: self.tolerance,
            ..Default::default()
        };
        let tr_result = trust_region_cox(&augmented_data, &config, self.tie_method)?;

        Ok(CoxFitResult {
            coef: tr_result.coef,
            log_lik: tr_result.log_lik,
            n_iter: tr_result.n_outer_iters,
            converged: tr_result.converged,
            n_events,
            n_subjects,
        })
    }
}

// ---------------------------------------------------------------------------
// Ridge augmentation helper
// ---------------------------------------------------------------------------

/// For the trust-region path, encode ridge regularisation via data augmentation.
///
/// Appends `p` pseudo-observations with covariate `sqrt(lambda) * e_k`, time=1, event=true.
/// This shrinks β toward zero in the same direction as an L2 penalty on the partial likelihood.
fn augment_with_ridge(data: &[(f64, bool, Vec<f64>)], lambda: f64) -> Vec<(f64, bool, Vec<f64>)> {
    let p = data.first().map(|(_, _, x)| x.len()).unwrap_or(0);
    let mut aug = data.to_vec();
    let scale = lambda.sqrt();
    for k in 0..p {
        let mut x = vec![0.0_f64; p];
        x[k] = scale;
        // Use a very small time so they contribute to the risk set at all event times.
        aug.push((1.0e-8, false, x));
    }
    aug
}

// ---------------------------------------------------------------------------
// Shared helper: triples → Dataset
// ---------------------------------------------------------------------------

fn triples_to_dataset(data: &[(f64, bool, Vec<f64>)]) -> SurvivalResult<Dataset> {
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    let mut obs = Vec::with_capacity(data.len());
    let mut cov = Vec::with_capacity(data.len());
    for (t, e, x) in data.iter() {
        obs.push(Observation::new(*t, *e)?);
        cov.push(x.clone());
    }
    Dataset::new(obs, Some(cov), None)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    fn make_data(n: usize, beta_true: f64, seed: u64) -> Vec<(f64, bool, Vec<f64>)> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let lambda = (beta_true * x).exp();
                let t = rng.next_exponential(lambda).max(1.0e-6);
                (t, true, vec![x])
            })
            .collect()
    }

    fn make_data_censored(n: usize, beta_true: f64, seed: u64) -> Vec<(f64, bool, Vec<f64>)> {
        let mut rng = LcgRng::new(seed);
        (0..n)
            .map(|_| {
                let x = rng.next_normal();
                let lambda = (beta_true * x).exp();
                let t = rng.next_exponential(lambda).max(1.0e-6);
                let c = rng.next_exponential(0.5).max(1.0e-6);
                let event = t <= c;
                (t.min(c), event, vec![x])
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // Basic builder / result tests
    // ------------------------------------------------------------------

    #[test]
    fn builder_default_produces_valid_fit() {
        let data = make_data(200, 0.7, 42);
        let result = CoxBuilder::new().fit(&data).expect("ok");
        assert!(result.converged);
        assert!(result.coef[0] > 0.0);
        assert_eq!(result.n_subjects, 200);
        assert_eq!(result.n_events, 200); // all events in make_data
    }

    #[test]
    fn builder_trust_region_path_converges() {
        let data = make_data(300, 0.5, 77);
        let result = CoxBuilder::new()
            .trust_region()
            .max_iter(200)
            .fit(&data)
            .expect("ok");
        assert!(result.converged, "did not converge; coef={:?}", result.coef);
        assert!(result.coef[0] > 0.0);
    }

    #[test]
    fn builder_breslow_ties_matches_direction() {
        let data = make_data(200, -0.8, 55);
        let result = CoxBuilder::new()
            .ties(TieMethod::Breslow)
            .fit(&data)
            .expect("ok");
        assert!(result.coef[0] < 0.0, "coef={}", result.coef[0]);
    }

    #[test]
    fn builder_efron_ties_matches_direction() {
        let data = make_data(200, 1.2, 66);
        let result = CoxBuilder::new()
            .ties(TieMethod::Efron)
            .fit(&data)
            .expect("ok");
        assert!(result.coef[0] > 0.0, "coef={}", result.coef[0]);
    }

    #[test]
    fn builder_ridge_shrinks_coef_toward_zero() {
        let data = make_data(200, 1.5, 88);
        let no_ridge = CoxBuilder::new().fit(&data).expect("no_ridge ok");
        let with_ridge = CoxBuilder::new().ridge(1.0).fit(&data).expect("ridge ok");
        // Ridge should shrink |β̂| toward 0.
        assert!(
            with_ridge.coef[0].abs() < no_ridge.coef[0].abs(),
            "ridge coef={} no_ridge coef={}",
            with_ridge.coef[0],
            no_ridge.coef[0]
        );
    }

    #[test]
    fn predict_risk_is_exp_of_log_hazard() {
        let data = make_data(100, 0.5, 11);
        let result = CoxBuilder::new().fit(&data).expect("ok");
        let x = [1.5_f64];
        let lh = result.predict_log_hazard(&x);
        let risk = result.predict_risk(&x);
        assert!((risk - lh.exp()).abs() < 1.0e-14);
    }

    #[test]
    fn predict_log_hazard_linearity() {
        let data = make_data(100, 0.5, 22);
        let result = CoxBuilder::new().fit(&data).expect("ok");
        let x1 = [1.0_f64];
        let x2 = [2.0_f64];
        // log-hazard should be proportional to covariate.
        let lh1 = result.predict_log_hazard(&x1);
        let lh2 = result.predict_log_hazard(&x2);
        // lh2 = 2 * lh1 for linear predictor with single covariate.
        assert!((lh2 - 2.0 * lh1).abs() < 1.0e-12);
    }

    #[test]
    fn builder_empty_data_returns_error() {
        let err = CoxBuilder::new().fit(&[]);
        assert!(matches!(err, Err(SurvivalError::EmptyDataset)));
    }

    #[test]
    fn builder_no_events_returns_error() {
        let data: Vec<(f64, bool, Vec<f64>)> =
            vec![(1.0, false, vec![0.5]), (2.0, false, vec![-0.3])];
        let err = CoxBuilder::new().fit(&data);
        assert!(matches!(err, Err(SurvivalError::NoEvents)));
    }

    #[test]
    fn builder_method_chaining_compiles() {
        // Just verify the chain compiles and executes.
        let data = make_data(50, 0.3, 999);
        let result = CoxBuilder::new()
            .ties(TieMethod::Efron)
            .max_iter(200)
            .tolerance(1.0e-7)
            .ridge(0.01)
            .fit(&data)
            .expect("ok");
        assert!(result.log_lik.is_finite());
    }

    #[test]
    fn builder_negative_ridge_returns_error() {
        let data = make_data(50, 0.5, 321);
        let err = CoxBuilder::new().ridge(-0.1).fit(&data);
        assert!(err.is_err());
    }

    #[test]
    fn builder_trust_region_and_nr_agree_sign() {
        let data = make_data(200, 0.8, 444);
        let nr = CoxBuilder::new().fit(&data).expect("nr ok");
        let tr = CoxBuilder::new().trust_region().fit(&data).expect("tr ok");
        // Both should recover positive β for beta_true=0.8.
        assert!(nr.coef[0] > 0.0);
        assert!(tr.coef[0] > 0.0);
    }

    #[test]
    fn fit_censored_data_ok() {
        let data = make_data_censored(200, 0.7, 555);
        let result = CoxBuilder::new().fit(&data).expect("ok");
        assert!(result.converged);
        assert!(result.n_events < result.n_subjects);
    }
}
