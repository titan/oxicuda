//! Discrete-time survival analysis via logistic and complementary log-log link functions.
//!
//! Assumes survival time is measured in discrete intervals `[0,1), [1,2), …, [T-1,T)`.
//! At each interval the conditional hazard probability is:
//!
//! ```text
//! h(t | x) = P(T = t | T ≥ t, x)
//! ```
//!
//! Two link functions are supported:
//! - **Logistic:** `h(t | x) = σ(α_t + x^T β)`  where `σ(z) = 1 / (1 + exp(-z))`
//! - **Complementary log-log (cloglog):** `h(t | x) = 1 - exp(-exp(α_t + x^T β))`
//!
//! Parameters `θ = (α_1, …, α_T, β_1, …, β_p)` are estimated by direct maximisation
//! of the discrete-time log-likelihood using gradient ascent with Armijo backtracking.
//!
//! # References
//! - Allison PD (1982). Discrete-time methods for the analysis of event histories.
//!   *Sociological Methodology* 13: 61–98.
//! - Singer JD, Willett JB (1993). It's about time: using discrete-time survival analysis
//!   to study duration and the timing of events. *Journal of Educational Statistics* 18: 155–195.

use crate::data::Dataset;
use crate::error::{SurvivalError, SurvivalResult};

// ──────────────────────────────────────────────────────────────────────────────
// Link function enum
// ──────────────────────────────────────────────────────────────────────────────

/// Link function for the discrete-time hazard model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscreteTimeLink {
    /// Logistic link: `h = σ(η) = 1 / (1 + exp(-η))`.
    Logistic,
    /// Complementary log-log link: `h = 1 - exp(-exp(η))`.
    CLogLog,
}

// ──────────────────────────────────────────────────────────────────────────────
// Link function application
// ──────────────────────────────────────────────────────────────────────────────

/// Apply the chosen link function to linear predictor `eta`.
///
/// - Logistic: `σ(η) = 1 / (1 + exp(-η))` (numerically stable)
/// - CLogLog:  `1 - exp(-exp(η))` (clipped to `[0, 1]`)
#[inline]
fn apply_link(eta: f64, link: DiscreteTimeLink) -> f64 {
    match link {
        DiscreteTimeLink::Logistic => {
            // Stable sigmoid: avoids overflow for large |η|.
            if eta >= 0.0 {
                let e = (-eta).exp();
                1.0 / (1.0 + e)
            } else {
                let e = eta.exp();
                e / (1.0 + e)
            }
        }
        DiscreteTimeLink::CLogLog => {
            // 1 - exp(-exp(η)), clipped to [0, 1].
            let v = 1.0 - (-eta.exp()).exp();
            v.clamp(0.0, 1.0)
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for `fit_discrete_time`.
#[derive(Debug, Clone)]
pub struct DiscreteTimeConfig {
    /// Link function to use (logistic or cloglog).
    pub link: DiscreteTimeLink,
    /// Convergence tolerance: stop when `max |gradient component| < tol`.
    pub tol: f64,
    /// Maximum number of gradient-ascent iterations.
    pub max_iter: usize,
    /// Initial learning rate for Armijo backtracking line search.
    pub lr: f64,
}

impl Default for DiscreteTimeConfig {
    fn default() -> Self {
        Self {
            link: DiscreteTimeLink::Logistic,
            tol: 1.0e-5,
            max_iter: 200,
            lr: 0.01,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Fit result
// ──────────────────────────────────────────────────────────────────────────────

/// Fitted discrete-time survival model.
#[derive(Debug, Clone)]
pub struct DiscreteTimeFit {
    /// Interval-specific intercepts `α_1, …, α_T` (one per distinct observed time point).
    pub alpha: Vec<f64>,
    /// Covariate coefficients `β_1, …, β_p`.
    pub beta: Vec<f64>,
    /// Sorted distinct observed time points used to index `alpha`.
    pub time_points: Vec<f64>,
    /// Log-likelihood at the final parameter estimates.
    pub log_likelihood: f64,
    /// Number of gradient-ascent iterations taken.
    pub n_iter: usize,
    /// Whether the optimiser declared convergence.
    pub converged: bool,
    /// Link function used during fitting (needed for prediction).
    pub link: DiscreteTimeLink,
}

impl DiscreteTimeFit {
    // ── Internal helpers ──────────────────────────────────────────────────────

    /// Compute hazard probability at each model time point given covariate vector `x`.
    fn hazard_at_each_time_point(&self, x: &[f64]) -> SurvivalResult<Vec<f64>> {
        let beta_dot_x: f64 = self.beta.iter().zip(x.iter()).map(|(b, xi)| b * xi).sum();
        self.alpha
            .iter()
            .map(|&a| {
                let eta = a + beta_dot_x;
                Ok(apply_link(eta, self.link))
            })
            .collect()
    }

    // ── Public prediction methods ─────────────────────────────────────────────

    /// Predict the survival probability `S(t | x)` at each requested time.
    ///
    /// `S(t | x) = Π_{j: t_j ≤ t} (1 - h(t_j | x))`
    ///
    /// For times before the first observed time point `S = 1`.  For times after
    /// the last time point the product is taken over all model time points.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != beta.len()`.
    pub fn predict_survival(&self, x: &[f64], times: &[f64]) -> SurvivalResult<Vec<f64>> {
        if x.len() != self.beta.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: x.len(),
                b: self.beta.len(),
            });
        }
        let hazards = self.hazard_at_each_time_point(x)?;
        times
            .iter()
            .map(|&t| {
                let mut s = 1.0_f64;
                for (j, &tj) in self.time_points.iter().enumerate() {
                    if tj > t {
                        break;
                    }
                    s *= 1.0 - hazards[j];
                    s = s.max(0.0);
                }
                Ok(s)
            })
            .collect()
    }

    /// Predict the conditional hazard probability `h(t | x)` at each requested time.
    ///
    /// For times not exactly matching a model time point, returns the hazard at the
    /// nearest lower time point, or 0 if no model time point is ≤ `t`.
    ///
    /// # Errors
    /// - `DimensionMismatch` if `x.len() != beta.len()`.
    pub fn predict_hazard(&self, x: &[f64], times: &[f64]) -> SurvivalResult<Vec<f64>> {
        if x.len() != self.beta.len() {
            return Err(SurvivalError::DimensionMismatch {
                a: x.len(),
                b: self.beta.len(),
            });
        }
        let hazards = self.hazard_at_each_time_point(x)?;
        times
            .iter()
            .map(|&t| {
                // Find the count of time points ≤ t (= index of first time point > t).
                let idx = self.time_points.partition_point(|&tp| tp <= t);
                if idx == 0 {
                    Ok(0.0)
                } else {
                    Ok(hazards[idx - 1])
                }
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Log-likelihood
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the discrete-time log-likelihood.
///
/// Parameters:
/// - `theta[0..T]`    = `α_1, …, α_T`  (interval-specific intercepts)
/// - `theta[T..T+p]`  = `β_1, …, β_p`  (covariate coefficients)
///
/// For observation `i` with event time `T_i` and status `s_i`:
///
/// ```text
/// log L_i = Σ_{t < T_i} log(1 - h(t|x_i))
///         + s_i · log h(T_i|x_i)
///         + (1 - s_i) · log(1 - h(T_i|x_i))
/// ```
///
/// `log(1 - h)` is computed via `(-h).ln_1p()` for numerical stability.
fn discrete_ll(
    data: &Dataset,
    theta: &[f64],
    time_points: &[f64],
    p: usize,
    link: DiscreteTimeLink,
) -> f64 {
    let big_t = time_points.len();
    let alpha = &theta[..big_t];
    let beta = &theta[big_t..big_t + p];

    let mut ll = 0.0_f64;

    for (i, obs) in data.observations.iter().enumerate() {
        let t_i = obs.time;
        let s_i = obs.event;

        // β · xᵢ computed once per observation
        let beta_dot_x: f64 = if p > 0 {
            match &data.covariates {
                Some(cov) => beta.iter().zip(cov[i].iter()).map(|(b, xi)| b * xi).sum(),
                None => 0.0,
            }
        } else {
            0.0
        };

        // Iterate over model time points ≤ t_i
        for (j, &tj) in time_points.iter().enumerate() {
            if tj > t_i + 1.0e-10 {
                // Past the observation time — stop
                break;
            }

            let eta = alpha[j] + beta_dot_x;
            let h = apply_link(eta, link);

            // Clamp h away from 0 and 1 to avoid log(0) / log(0-ε)
            let h_safe = h.clamp(1.0e-15, 1.0 - 1.0e-15);

            let at_event_time = (tj - t_i).abs() < 1.0e-10;

            if at_event_time {
                // Interval of event/censoring
                if s_i {
                    ll += h_safe.ln();
                } else {
                    ll += (-h_safe).ln_1p(); // log(1 - h)
                }
                break;
            } else {
                // Interval strictly before event/censoring: log(1 - h)
                ll += (-h_safe).ln_1p();
            }
        }
    }

    if ll.is_finite() {
        ll
    } else {
        f64::NEG_INFINITY
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Finite-difference gradient (central differences, h = 1e-6)
// ──────────────────────────────────────────────────────────────────────────────

/// Central-difference gradient of the log-likelihood with step size `fd_h`.
fn finite_diff_gradient(
    data: &Dataset,
    theta: &[f64],
    time_points: &[f64],
    p: usize,
    link: DiscreteTimeLink,
    fd_h: f64,
) -> Vec<f64> {
    let dim = theta.len();
    let mut grad = vec![0.0_f64; dim];
    let mut theta_p = theta.to_vec();
    let mut theta_m = theta.to_vec();

    for j in 0..dim {
        theta_p[j] = theta[j] + fd_h;
        theta_m[j] = theta[j] - fd_h;
        let lp = discrete_ll(data, &theta_p, time_points, p, link);
        let lm = discrete_ll(data, &theta_m, time_points, p, link);
        grad[j] = (lp - lm) / (2.0 * fd_h);
        theta_p[j] = theta[j];
        theta_m[j] = theta[j];
    }
    grad
}

// ──────────────────────────────────────────────────────────────────────────────
// Main fitting function
// ──────────────────────────────────────────────────────────────────────────────

/// Fit a discrete-time survival model with the chosen link function.
///
/// # Algorithm
/// 1. Extract the sorted distinct observed time points `t_1 < … < t_T`.
/// 2. Initialise `θ = 0`.
/// 3. Run gradient ascent with Armijo backtracking: step = `θ ← θ + lr * ∇ℓ(θ)`;
///    halve `lr` until ℓ increases sufficiently (up to 50 halvings).
/// 4. Convergence declared when `max |∇ℓ(θ)| < tol`.
///
/// # Errors
/// - `EmptyDataset` if the dataset has no observations.
/// - `InvalidParameter` if any config value is non-positive.
/// - `NumericalInstability` if the final log-likelihood is non-finite.
pub fn fit_discrete_time(
    data: &Dataset,
    config: &DiscreteTimeConfig,
) -> SurvivalResult<DiscreteTimeFit> {
    // ── Validation ────────────────────────────────────────────────────────────
    if data.is_empty() {
        return Err(SurvivalError::EmptyDataset);
    }
    if config.tol <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "tol must be positive".to_string(),
        ));
    }
    if config.max_iter == 0 {
        return Err(SurvivalError::InvalidParameter(
            "max_iter must be positive".to_string(),
        ));
    }
    if config.lr <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "lr must be positive".to_string(),
        ));
    }

    // ── Collect distinct, sorted time points ─────────────────────────────────
    let mut time_points: Vec<f64> = data.observations.iter().map(|o| o.time).collect();
    time_points.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    time_points.dedup_by(|a, b| (*a - *b).abs() < 1.0e-10);

    let big_t = time_points.len();
    let p = data.n_features();
    let dim = big_t + p;

    // ── Initialise parameters to zero ────────────────────────────────────────
    let mut theta = vec![0.0_f64; dim];

    // ── Gradient ascent with Armijo backtracking ──────────────────────────────
    const FD_H: f64 = 1.0e-6;
    // Armijo sufficient-increase constant c: ℓ(θ + step·g) ≥ ℓ(θ) + c·step·‖g‖²
    const ARMIJO_C: f64 = 1.0e-4;
    const MAX_HALVINGS: usize = 50;

    let link = config.link;
    let mut ll_cur = discrete_ll(data, &theta, &time_points, p, link);
    let mut grad = finite_diff_gradient(data, &theta, &time_points, p, link, FD_H);

    let mut n_iter = 0usize;
    let mut converged = false;

    let mut theta_new = vec![0.0_f64; dim];

    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Convergence check: max absolute gradient component
        let max_grad = grad.iter().map(|g| g.abs()).fold(0.0_f64, f64::max);
        if max_grad < config.tol {
            converged = true;
            break;
        }

        // Armijo backtracking along the gradient direction (steepest ascent)
        let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
        let mut step = config.lr;
        let mut accepted = false;

        for _ in 0..MAX_HALVINGS {
            for j in 0..dim {
                theta_new[j] = theta[j] + step * grad[j];
            }
            let ll_new = discrete_ll(data, &theta_new, &time_points, p, link);
            if ll_new.is_finite() && ll_new >= ll_cur + ARMIJO_C * step * grad_norm_sq {
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1.0e-20 {
                break;
            }
        }

        if accepted {
            theta.copy_from_slice(&theta_new);
            ll_cur = discrete_ll(data, &theta, &time_points, p, link);
            grad = finite_diff_gradient(data, &theta, &time_points, p, link, FD_H);
        } else {
            // Cannot make further progress — exit early without converged flag
            break;
        }
    }

    let log_likelihood = discrete_ll(data, &theta, &time_points, p, link);
    if !log_likelihood.is_finite() {
        return Err(SurvivalError::NumericalInstability(
            "non-finite log-likelihood at final parameters".to_string(),
        ));
    }

    let alpha = theta[..big_t].to_vec();
    let beta = theta[big_t..].to_vec();

    Ok(DiscreteTimeFit {
        alpha,
        beta,
        time_points,
        log_likelihood,
        n_iter,
        converged,
        link,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, Observation};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn obs(t: f64, event: bool) -> Observation {
        Observation::new(t, event).expect("valid observation")
    }

    /// Small synthetic discrete-time dataset:
    /// 15 observations at integer times 1–4 with one covariate.
    fn make_discrete_dataset() -> Dataset {
        let observations = vec![
            obs(1.0, true),
            obs(1.0, true),
            obs(2.0, true),
            obs(2.0, false),
            obs(2.0, true),
            obs(3.0, false),
            obs(3.0, true),
            obs(3.0, true),
            obs(4.0, false),
            obs(4.0, true),
            obs(1.0, false),
            obs(2.0, true),
            obs(3.0, false),
            obs(4.0, true),
            obs(4.0, false),
        ];
        let covariates = vec![
            vec![0.5_f64],
            vec![-0.5],
            vec![1.0],
            vec![-1.0],
            vec![0.2],
            vec![0.8],
            vec![-0.3],
            vec![0.6],
            vec![-0.7],
            vec![0.4],
            vec![1.2],
            vec![-0.9],
            vec![0.1],
            vec![-0.2],
            vec![0.7],
        ];
        Dataset::new(observations, Some(covariates), None).expect("valid dataset")
    }

    /// Simple no-covariate dataset.
    fn make_no_cov_dataset() -> Dataset {
        let observations = vec![
            obs(1.0, true),
            obs(2.0, true),
            obs(2.0, false),
            obs(3.0, true),
            obs(3.0, false),
            obs(4.0, true),
            obs(4.0, false),
            obs(1.0, false),
            obs(3.0, true),
            obs(4.0, false),
        ];
        Dataset::new(observations, None, None).expect("valid dataset")
    }

    /// All-censored no-covariate dataset.
    fn make_all_censored_dataset() -> Dataset {
        let observations = vec![
            obs(1.0, false),
            obs(2.0, false),
            obs(3.0, false),
            obs(4.0, false),
            obs(2.0, false),
        ];
        Dataset::new(observations, None, None).expect("valid dataset")
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    /// Logistic link fit returns Ok on small synthetic data.
    #[test]
    fn discrete_logistic_fit_basic() {
        let data = make_discrete_dataset();
        let config = DiscreteTimeConfig::default();
        let result = fit_discrete_time(&data, &config);
        assert!(
            result.is_ok(),
            "logistic fit should return Ok, got {:?}",
            result
        );
    }

    /// CLogLog link fit returns Ok on small synthetic data.
    #[test]
    fn discrete_cloglog_fit_basic() {
        let data = make_discrete_dataset();
        let config = DiscreteTimeConfig {
            link: DiscreteTimeLink::CLogLog,
            ..Default::default()
        };
        let result = fit_discrete_time(&data, &config);
        assert!(
            result.is_ok(),
            "cloglog fit should return Ok, got {:?}",
            result
        );
    }

    /// `predict_survival` returns values in `(0, 1]` for all requested times.
    #[test]
    fn discrete_survival_in_01() {
        let data = make_no_cov_dataset();
        let config = DiscreteTimeConfig {
            max_iter: 100,
            ..Default::default()
        };
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        let times: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
        let s = fit.predict_survival(&[], &times).expect("predict ok");
        for (i, &si) in s.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&si),
                "S(t[{i}]) = {si} must be in [0, 1]"
            );
        }
    }

    /// `predict_survival` is non-increasing in `t`.
    #[test]
    fn discrete_survival_monotone() {
        let data = make_no_cov_dataset();
        let config = DiscreteTimeConfig {
            max_iter: 100,
            ..Default::default()
        };
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        let times: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = fit.predict_survival(&[], &times).expect("predict ok");
        for w in s.windows(2) {
            assert!(
                w[0] >= w[1] - 1.0e-9,
                "S(t) must be non-increasing: got {} then {}",
                w[0],
                w[1]
            );
        }
    }

    /// `predict_hazard` returns values in `[0, 1)` for model time points.
    #[test]
    fn discrete_hazard_in_01() {
        let data = make_no_cov_dataset();
        let config = DiscreteTimeConfig {
            max_iter: 100,
            ..Default::default()
        };
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        let times: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let h = fit.predict_hazard(&[], &times).expect("predict ok");
        for (i, &hi) in h.iter().enumerate() {
            assert!(
                (0.0..1.0).contains(&hi),
                "h(t[{i}]) = {hi} must be in [0, 1)"
            );
        }
    }

    /// The final `log_likelihood` field must be finite.
    #[test]
    fn discrete_loglik_finite() {
        let data = make_discrete_dataset();
        let config = DiscreteTimeConfig::default();
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        assert!(
            fit.log_likelihood.is_finite(),
            "log_likelihood must be finite, got {}",
            fit.log_likelihood
        );
    }

    /// With generous iteration budget and loose tolerance, easy data should converge.
    #[test]
    fn discrete_converged_simple_data() {
        let data = make_no_cov_dataset();
        let config = DiscreteTimeConfig {
            tol: 1.0e-4,
            max_iter: 500,
            lr: 0.05,
            ..Default::default()
        };
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        assert!(
            fit.converged,
            "should converge on simple data; n_iter={}",
            fit.n_iter
        );
    }

    /// `alpha.len()` must equal `time_points.len()` after fitting.
    #[test]
    fn discrete_alpha_length_matches_time_points() {
        let data = make_discrete_dataset();
        let config = DiscreteTimeConfig::default();
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        assert_eq!(
            fit.alpha.len(),
            fit.time_points.len(),
            "alpha.len() must equal time_points.len()"
        );
    }

    /// Constructing an empty `Dataset` returns an error (no empty-dataset fit path needed).
    #[test]
    fn discrete_empty_dataset_returns_error() {
        let result = Dataset::new(vec![], None, None);
        assert!(
            result.is_err(),
            "empty dataset construction must return an error"
        );
    }

    /// All-censored data: the log-likelihood is well-defined (no `log h` terms).
    #[test]
    fn discrete_no_events_valid() {
        let data = make_all_censored_dataset();
        let config = DiscreteTimeConfig {
            max_iter: 50,
            ..Default::default()
        };
        let result = fit_discrete_time(&data, &config);
        assert!(
            result.is_ok(),
            "all-censored data should still fit, got {:?}",
            result
        );
        let fit = result.expect("ok");
        assert!(
            fit.log_likelihood.is_finite(),
            "log_likelihood must be finite"
        );
    }

    /// CLogLog survival must also be non-increasing.
    #[test]
    fn discrete_cloglog_survival_monotone() {
        let data = make_no_cov_dataset();
        let config = DiscreteTimeConfig {
            link: DiscreteTimeLink::CLogLog,
            max_iter: 100,
            ..Default::default()
        };
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        let times: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let s = fit.predict_survival(&[], &times).expect("predict ok");
        for w in s.windows(2) {
            assert!(
                w[0] >= w[1] - 1.0e-9,
                "cloglog S(t) must be non-increasing: {} then {}",
                w[0],
                w[1]
            );
        }
    }

    /// `time_points` are strictly increasing regardless of observation order.
    #[test]
    fn discrete_time_points_sorted() {
        let data = Dataset::from_arrays(
            &[4.0, 1.0, 3.0, 2.0, 1.0, 3.0],
            &[true, false, true, true, true, false],
        )
        .expect("ok");
        let config = DiscreteTimeConfig {
            max_iter: 50,
            ..Default::default()
        };
        let fit = fit_discrete_time(&data, &config).expect("fit ok");
        for w in fit.time_points.windows(2) {
            assert!(w[0] < w[1], "time_points must be strictly increasing");
        }
    }
}
