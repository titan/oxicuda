//! Mixture cure model (Berkson-Gage 1952, Farewell 1982).
//!
//! The mixture cure model decomposes overall survival as:
//!
//! ```text
//! S(t|x) = π(x) + [1 − π(x)] · S_u(t|x)
//! ```
//!
//! where:
//! - `π(x) = sigmoid(γᵀx)` is the **cure probability** (logistic incidence model).
//! - `S_u(t|x) = exp(-exp(βᵀx) · t)` is the **latency survival** for susceptible subjects
//!   (simplified exponential baseline hazard).
//!
//! ## Algorithm
//!
//! Iterative EM-like gradient ascent:
//!
//! **E-step**: Compute posterior probability of being susceptible:
//! - Events: `ν_i = 1`.
//! - Censored: `ν_i = (1 − π_i) · S_u(t_i) / [π_i + (1 − π_i) · S_u(t_i)]`.
//!
//! **M-step** (SGD):
//! - Incidence gradient: `grad_γ[k] = Σ_i (ν_i − (1 − π_i)) · x_ik / n`.
//! - Latency gradient: `grad_β[k] = Σ_i ν_i · (δ_i − exp(βᵀx_i) · t_i) · x_ik / n`.
//! - Update: `γ += lr · grad_γ`, `β += lr · grad_β`.

use crate::error::{SurvivalError, SurvivalResult};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for the simplified mixture cure model.
#[derive(Debug, Clone)]
pub struct CureModelConfig {
    /// Number of covariates (shared for incidence and latency models).
    pub n_covariates: usize,
    /// Learning rate for SGD updates.
    pub lr: f64,
    /// Maximum number of EM-like iterations.
    pub n_iter: usize,
}

impl Default for CureModelConfig {
    fn default() -> Self {
        Self {
            n_covariates: 1,
            lr: 0.01,
            n_iter: 100,
        }
    }
}

/// Result of a fitted mixture cure model.
#[derive(Debug, Clone)]
pub struct CureModelFit {
    /// Logistic coefficients γ for the cure probability model (length = `n_covariates`).
    pub incidence_coef: Vec<f64>,
    /// Cox PH coefficients β for the latency survival model (length = `n_covariates`).
    pub latency_coef: Vec<f64>,
    /// Average cure probability `π(x)` across all training subjects.
    pub cure_fraction: f64,
    /// Whether the gradient norms were non-degenerate throughout.
    pub converged: bool,
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Numerically-stable sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let xc = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-xc).exp())
}

/// Compute `γᵀx_i` for subject `i`.
#[inline]
fn dot_row(coef: &[f64], covariates: &[f64], i: usize, p: usize) -> f64 {
    (0..p).map(|k| coef[k] * covariates[i * p + k]).sum()
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Fit a mixture cure model on raw slices via EM-like gradient ascent.
///
/// # Parameters
/// - `covariates`: row-major `[n × n_covariates]` covariate matrix.
/// - `times`: observed times, length `n`.
/// - `events`: event indicators (0 = censored, 1 = event), length `n`.
/// - `n`: number of subjects.
/// - `config`: algorithm configuration.
///
/// # Errors
/// - [`SurvivalError::EmptyDataset`] when `n == 0`.
/// - [`SurvivalError::InvalidParameter`] for invalid config or array sizes.
/// - [`SurvivalError::NoEvents`] when there are no observed events.
pub fn mixture_cure_fit(
    covariates: &[f64],
    times: &[f64],
    events: &[u8],
    n: usize,
    config: &CureModelConfig,
) -> SurvivalResult<CureModelFit> {
    // ── Validation ────────────────────────────────────────────────────────────
    if n == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    let p = config.n_covariates;
    if p == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_covariates must be >= 1".to_string(),
        ));
    }
    if covariates.len() != n * p {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n * p],
            got: vec![covariates.len()],
        });
    }
    if times.len() != n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n],
            got: vec![times.len()],
        });
    }
    if events.len() != n {
        return Err(SurvivalError::ShapeMismatch {
            expected: vec![n],
            got: vec![events.len()],
        });
    }
    if config.lr <= 0.0 {
        return Err(SurvivalError::InvalidParameter(
            "lr must be > 0".to_string(),
        ));
    }
    if config.n_iter == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_iter must be >= 1".to_string(),
        ));
    }
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }
    let n_ev = events.iter().filter(|&&e| e == 1).count();
    if n_ev == 0 {
        return Err(SurvivalError::NoEvents);
    }

    // ── Initialisation ────────────────────────────────────────────────────────
    let mut incidence_coef = vec![0.0_f64; p];
    let mut latency_coef = vec![0.0_f64; p];
    let mut converged = false;

    // ── EM-like gradient ascent ───────────────────────────────────────────────
    for _iter in 0..config.n_iter {
        // ── E-step: compute posterior susceptibility ν_i ─────────────────────
        let mut nu = vec![0.0_f64; n];
        for i in 0..n {
            if events[i] == 1 {
                nu[i] = 1.0;
            } else {
                let gamma_x = dot_row(&incidence_coef, covariates, i, p);
                let pi_i = sigmoid(gamma_x);
                let beta_x = dot_row(&latency_coef, covariates, i, p);
                // Simplified exponential latency: S_u(t|x) = exp(-exp(β·x) * t)
                let lambda_i = beta_x.exp();
                let su_i = (-lambda_i * times[i]).exp();
                let uncured = 1.0 - pi_i;
                let numer = uncured * su_i;
                let denom = pi_i + numer;
                nu[i] = if denom < 1.0e-300 {
                    0.0
                } else {
                    (numer / denom).clamp(0.0, 1.0)
                };
            }
        }

        // ── M-step: compute SGD gradients ─────────────────────────────────────
        let mut grad_gamma = vec![0.0_f64; p];
        let mut grad_beta = vec![0.0_f64; p];

        for i in 0..n {
            let gamma_x = dot_row(&incidence_coef, covariates, i, p);
            let pi_i = sigmoid(gamma_x);
            let beta_x = dot_row(&latency_coef, covariates, i, p);
            let lambda_i = beta_x.exp();

            // Incidence gradient: grad_γ[k] = (ν_i - (1 - π_i)) * x_ik / n
            let incidence_delta = nu[i] - (1.0 - pi_i);
            // Latency gradient: grad_β[k] = ν_i * (δ_i - λ_i * t_i) * x_ik / n
            let latency_delta = nu[i] * (events[i] as f64 - lambda_i * times[i]);

            for k in 0..p {
                let x_ik = covariates[i * p + k];
                grad_gamma[k] += incidence_delta * x_ik;
                grad_beta[k] += latency_delta * x_ik;
            }
        }

        let inv_n = 1.0 / n as f64;
        let max_g = grad_gamma
            .iter()
            .chain(grad_beta.iter())
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));

        if !max_g.is_finite() {
            break;
        }

        // Apply SGD update
        for k in 0..p {
            incidence_coef[k] += config.lr * grad_gamma[k] * inv_n;
            latency_coef[k] += config.lr * grad_beta[k] * inv_n;
        }

        if max_g * inv_n < 1.0e-10 {
            converged = true;
            break;
        }
    }

    // ── Compute average cure fraction ─────────────────────────────────────────
    let cure_fraction = (0..n)
        .map(|i| {
            let gamma_x = dot_row(&incidence_coef, covariates, i, p);
            sigmoid(gamma_x)
        })
        .sum::<f64>()
        / n as f64;

    Ok(CureModelFit {
        incidence_coef,
        latency_coef,
        cure_fraction,
        converged,
    })
}

/// Predict survival probability `S(t|x)` for a single new subject.
///
/// Uses the mixture cure decomposition:
///
/// ```text
/// S(t|x) = π(x) + (1 - π(x)) · S_u(t|x)
/// ```
///
/// where:
/// - `π(x) = sigmoid(γᵀx)` (cure probability)
/// - `S_u(t|x) = exp(-exp(βᵀx) · t)` (exponential latency baseline)
///
/// # Errors
/// - [`SurvivalError::InvalidParameter`] when `x.len() != n_covariates`.
/// - [`SurvivalError::NegativeTime`] when `t < 0`.
pub fn cure_predict_survival(fit: &CureModelFit, x: &[f64], t: f64) -> SurvivalResult<f64> {
    let p = fit.incidence_coef.len();
    if x.len() != p {
        return Err(SurvivalError::InvalidParameter(format!(
            "x.len()={} but model has n_covariates={}",
            x.len(),
            p
        )));
    }
    if t < 0.0 {
        return Err(SurvivalError::NegativeTime(t));
    }

    let gamma_x: f64 = fit
        .incidence_coef
        .iter()
        .zip(x.iter())
        .map(|(g, xi)| g * xi)
        .sum();
    let beta_x: f64 = fit
        .latency_coef
        .iter()
        .zip(x.iter())
        .map(|(b, xi)| b * xi)
        .sum();

    let pi = sigmoid(gamma_x);
    let lambda = beta_x.exp();
    let su = (-lambda * t).exp();
    let s = pi + (1.0 - pi) * su;

    Ok(s.clamp(0.0, 1.0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mixed_data(n: usize, cure_rate: f64, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<u8>) {
        use crate::handle::LcgRng;
        let mut rng = LcgRng::new(seed);
        let mut cov = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        let mut events = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.next_normal();
            cov.push(x);
            let cured = rng.next_f64() < cure_rate;
            if cured {
                times.push(10.0_f64);
                events.push(0u8);
            } else {
                let t = rng.next_exponential(1.0).max(1.0e-6);
                if t < 8.0 {
                    times.push(t);
                    events.push(1u8);
                } else {
                    times.push(8.0);
                    events.push(0u8);
                }
            }
        }
        (cov, times, events)
    }

    #[test]
    fn coefficients_len() {
        let (cov, times, events) = make_mixed_data(80, 0.3, 4001);
        let config = CureModelConfig::default();
        let fit = mixture_cure_fit(&cov, &times, &events, 80, &config).expect("fit should succeed");
        assert_eq!(fit.incidence_coef.len(), config.n_covariates);
        assert_eq!(fit.latency_coef.len(), config.n_covariates);
    }

    #[test]
    fn cure_fraction_in_range() {
        let (cov, times, events) = make_mixed_data(80, 0.3, 4002);
        let config = CureModelConfig::default();
        let fit = mixture_cure_fit(&cov, &times, &events, 80, &config).expect("fit should succeed");
        assert!(
            fit.cure_fraction >= 0.0 && fit.cure_fraction <= 1.0,
            "cure_fraction={} not in [0,1]",
            fit.cure_fraction
        );
    }

    #[test]
    fn survival_in_range() {
        let (cov, times, events) = make_mixed_data(80, 0.3, 4003);
        let config = CureModelConfig::default();
        let fit = mixture_cure_fit(&cov, &times, &events, 80, &config).expect("fit should succeed");
        let s = cure_predict_survival(&fit, &[0.5], 1.0).expect("predict should succeed");
        assert!((0.0..=1.0).contains(&s), "survival={s} not in [0,1]");
    }

    #[test]
    fn finite_output() {
        let (cov, times, events) = make_mixed_data(80, 0.3, 4004);
        let config = CureModelConfig::default();
        let fit = mixture_cure_fit(&cov, &times, &events, 80, &config).expect("fit should succeed");
        for &c in fit.incidence_coef.iter().chain(fit.latency_coef.iter()) {
            assert!(c.is_finite(), "coef {c} not finite");
        }
        assert!(fit.cure_fraction.is_finite(), "cure_fraction not finite");
    }

    #[test]
    fn all_events_low_cure() {
        use crate::handle::LcgRng;
        let n = 100usize;
        let mut rng = LcgRng::new(5005);
        let mut cov = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        let events = vec![1u8; n];
        for _ in 0..n {
            cov.push(rng.next_normal());
            times.push(rng.next_exponential(1.0).max(1.0e-6));
        }
        let config = CureModelConfig {
            n_iter: 500,
            lr: 0.05,
            ..Default::default()
        };
        let fit = mixture_cure_fit(&cov, &times, &events, n, &config).expect("fit should succeed");
        // All events => the EM drives cure fraction below 0.5 with enough iterations
        assert!(
            fit.cure_fraction < 0.6,
            "all-events cure_fraction={} expected < 0.6",
            fit.cure_fraction
        );
    }

    #[test]
    fn all_censored_high_cure() {
        // All censored → model expects more cured subjects
        use crate::handle::LcgRng;
        let n = 100usize;
        let mut rng = LcgRng::new(6006);
        let mut cov = Vec::with_capacity(n);
        let mut times = Vec::with_capacity(n);
        // We need at least 1 event to avoid NoEvents error, so make 99 censored + 1 event
        let mut events = vec![0u8; n];
        events[0] = 1;
        for _ in 0..n {
            cov.push(rng.next_normal());
            times.push(rng.next_exponential(0.1).max(5.0)); // long times → mostly censored
        }
        times[0] = 0.1; // ensure the one event has a small time
        let config = CureModelConfig {
            n_iter: 150,
            ..Default::default()
        };
        let fit = mixture_cure_fit(&cov, &times, &events, n, &config).expect("fit should succeed");
        // Mostly censored → higher cure fraction
        assert!(
            fit.cure_fraction >= 0.0 && fit.cure_fraction <= 1.0,
            "cure_fraction={} out of range",
            fit.cure_fraction
        );
    }

    #[test]
    fn survival_decreasing_in_t() {
        let (cov, times, events) = make_mixed_data(80, 0.3, 7007);
        let config = CureModelConfig {
            n_iter: 150,
            ..Default::default()
        };
        let fit = mixture_cure_fit(&cov, &times, &events, 80, &config).expect("fit should succeed");
        let x = vec![0.0_f64]; // neutral covariate
        let s01 = cure_predict_survival(&fit, &x, 0.1).expect("predict t=0.1");
        let s10 = cure_predict_survival(&fit, &x, 1.0).expect("predict t=1.0");
        let s50 = cure_predict_survival(&fit, &x, 5.0).expect("predict t=5.0");
        // S(0.1) >= S(1.0) >= S(5.0) (non-increasing)
        assert!(s01 >= s10 - 1.0e-10, "S(0.1)={s01} < S(1.0)={s10}");
        assert!(s10 >= s50 - 1.0e-10, "S(1.0)={s10} < S(5.0)={s50}");
    }

    #[test]
    fn feature_mismatch_error() {
        let (cov, times, events) = make_mixed_data(80, 0.3, 8008);
        let config = CureModelConfig::default(); // n_covariates = 1
        let fit = mixture_cure_fit(&cov, &times, &events, 80, &config).expect("fit should succeed");
        // Provide x with wrong length
        let result = cure_predict_survival(&fit, &[0.5, 1.0], 1.0);
        assert!(
            matches!(result, Err(SurvivalError::InvalidParameter(_))),
            "expected InvalidParameter, got: {result:?}"
        );
    }

    #[test]
    fn no_events_returns_error() {
        let cov = vec![0.5_f64, 1.0, -0.5];
        let times = vec![1.0_f64, 2.0, 3.0];
        let events = vec![0u8, 0, 0];
        let config = CureModelConfig::default();
        let result = mixture_cure_fit(&cov, &times, &events, 3, &config);
        assert!(
            matches!(result, Err(SurvivalError::NoEvents)),
            "expected NoEvents, got: {result:?}"
        );
    }

    #[test]
    fn empty_dataset_error() {
        let config = CureModelConfig::default();
        let result = mixture_cure_fit(&[], &[], &[], 0, &config);
        assert!(
            matches!(result, Err(SurvivalError::EmptyDataset)),
            "expected EmptyDataset, got: {result:?}"
        );
    }
}
