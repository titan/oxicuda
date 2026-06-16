//! Nonparametric Bayesian Survival model via Dirichlet Process prior.
//!
//! Implements the Ferguson (1973) / Hjort (1990) approach: a Dirichlet Process
//! is placed over the survival distribution.  Posterior sampling is performed
//! by drawing Beta-distributed incremental hazards at each unique event time
//! via the stick-breaking / Bayesian bootstrap construction with a
//! concentration parameter alpha.
//!
//! Reference:
//! - Ferguson, T.S. (1973). "A Bayesian Analysis of Some Nonparametric
//!   Problems." *Annals of Statistics*, 1(2), 209–230.
//! - Hjort, N.L. (1990). "Nonparametric Bayes Estimators Based on Beta
//!   Processes in Models for Life History Data." *Annals of Statistics*,
//!   18(3), 1259–1294.

use crate::error::{SurvivalError, SurvivalResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Public configuration & result types
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the Dirichlet Process survival posterior sampler.
#[derive(Debug, Clone)]
pub struct DpSurvivalConfig {
    /// Concentration parameter alpha > 0.  Larger values give more weight to
    /// the (uniform) Dirichlet Process base measure, i.e. more prior influence.
    pub concentration: f64,
    /// Stick-breaking truncation level (carried in the struct for API
    /// completeness; the current algorithm uses Beta-increment sampling
    /// conditioned on observed event times and does not iterate over sticks).
    pub n_sticks: usize,
    /// Number of posterior Monte-Carlo samples.
    pub n_samples: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

/// Posterior summary of the Dirichlet Process survival estimator.
#[derive(Debug, Clone)]
pub struct DpSurvivalPosterior {
    /// Unique, sorted event times at which posterior quantities are evaluated.
    pub times: Vec<f64>,
    /// Posterior mean survival probability S(t) at each event time.
    pub mean_survival: Vec<f64>,
    /// 2.5th percentile of the posterior survival distribution at each time.
    pub ci_lower: Vec<f64>,
    /// 97.5th percentile of the posterior survival distribution at each time.
    pub ci_upper: Vec<f64>,
    /// Configuration used to produce this posterior.
    pub config: DpSurvivalConfig,
}

// ─────────────────────────────────────────────────────────────────────────────
// Core estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Fit a Dirichlet Process survival posterior by Monte-Carlo Beta-increment
/// sampling at each unique event time.
///
/// # Arguments
///
/// * `times`  – observed follow-up times (>= 0).
/// * `events` – `true` if the subject experienced the event, `false` for
///   censoring.
/// * `cfg`    – sampler configuration.
///
/// # Algorithm
///
/// For each posterior sample the incremental hazard at event time t_i is
/// drawn from Beta(d_i + α/K,  n_i − d_i + α·(1 − 1/K))  where d_i is the
/// number of events, n_i the number at risk, K = |{unique event times}|, and
/// α is the concentration.  The survival is then the cumulative product of
/// (1 − h_i) clamped to [0, 1].  Posterior mean and 2.5/97.5 percentiles are
/// reported.
pub fn dp_survival_posterior(
    times: &[f64],
    events: &[bool],
    cfg: &DpSurvivalConfig,
) -> SurvivalResult<DpSurvivalPosterior> {
    // ── 1. Validate inputs ───────────────────────────────────────────────────
    let n_obs = times.len();
    if n_obs == 0 {
        return Err(SurvivalError::EmptyDataset);
    }
    if events.len() != n_obs {
        return Err(SurvivalError::DimensionMismatch {
            a: n_obs,
            b: events.len(),
        });
    }
    for &t in times {
        if t < 0.0 {
            return Err(SurvivalError::NegativeTime(t));
        }
    }
    if !events.iter().any(|&e| e) {
        return Err(SurvivalError::NoEvents);
    }
    if cfg.concentration <= 0.0 {
        return Err(SurvivalError::InvalidParameter(format!(
            "concentration must be > 0, got {}",
            cfg.concentration
        )));
    }
    if cfg.n_samples == 0 {
        return Err(SurvivalError::InvalidParameter(
            "n_samples must be > 0".to_owned(),
        ));
    }

    // ── 2. Unique event times, at-risk counts, event counts ─────────────────
    // Collect only the times where at least one event occurred.
    let mut event_times: Vec<f64> = times
        .iter()
        .zip(events.iter())
        .filter_map(|(&t, &e)| if e { Some(t) } else { None })
        .collect();
    event_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    event_times.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON * a.abs().max(b.abs()).max(1.0));

    let k = event_times.len(); // K = number of unique event times

    // d_i = event count at each unique time; n_i = at-risk count.
    let mut d_counts: Vec<f64> = vec![0.0; k];
    let mut n_counts: Vec<f64> = vec![0.0; k];

    for (idx, &t_star) in event_times.iter().enumerate() {
        for (&t, &e) in times.iter().zip(events.iter()) {
            // at risk: subject with time >= t_star
            if t >= t_star {
                n_counts[idx] += 1.0;
            }
            // event at exactly t_star
            if e && (t - t_star).abs() < f64::EPSILON * t_star.abs().max(1.0) {
                d_counts[idx] += 1.0;
            }
        }
    }

    // ── 3. Monte-Carlo posterior sampling ───────────────────────────────────
    // Storage: flat row-major matrix, shape [n_samples × k].
    let n_samples = cfg.n_samples;
    let mut survival_mat: Vec<f64> = Vec::with_capacity(n_samples * k);

    let mut rng = LcgRng::new(cfg.seed);
    let k_f = k.max(1) as f64;
    let alpha = cfg.concentration;

    for _ in 0..n_samples {
        let mut log_surv_acc: f64 = 0.0; // accumulate log(1 − h_i)

        for i in 0..k {
            let d_i = d_counts[i];
            let n_i = n_counts[i];

            // Beta parameters (Hjort 1990, equation 4.2 in Ferguson's sense).
            let a_beta = d_i + alpha / k_f;
            let b_beta = (n_i - d_i) + alpha * (1.0 - 1.0 / k_f);

            // Clamp to avoid degenerate Beta parameters.
            let a_beta = a_beta.max(1.0e-10);
            let b_beta = b_beta.max(1.0e-10);

            let h_i = sample_beta(a_beta, b_beta, &mut rng);

            // log(1 − h_i), guard against log(0).
            let one_minus_h = (1.0 - h_i).clamp(1.0e-300, 1.0);
            log_surv_acc += one_minus_h.ln();

            // S(t_i) = exp(Σ_{j<=i} log(1 − h_j))
            let s_val = log_surv_acc.exp().clamp(0.0, 1.0);
            survival_mat.push(s_val);
        }
    }

    // ── 4. Posterior summaries ───────────────────────────────────────────────
    let mut mean_survival = Vec::with_capacity(k);
    let mut ci_lower = Vec::with_capacity(k);
    let mut ci_upper = Vec::with_capacity(k);

    for i in 0..k {
        // Gather the i-th column across all samples.
        let mut col: Vec<f64> = (0..n_samples).map(|s| survival_mat[s * k + i]).collect();

        let mean = col.iter().sum::<f64>() / n_samples as f64;
        mean_survival.push(mean);

        // Sort for quantile computation.
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lo = empirical_quantile(&col, 0.025);
        let hi = empirical_quantile(&col, 0.975);
        ci_lower.push(lo);
        ci_upper.push(hi);
    }

    Ok(DpSurvivalPosterior {
        times: event_times,
        mean_survival,
        ci_lower,
        ci_upper,
        config: cfg.clone(),
    })
}

/// Predict posterior-mean survival at time `t` by step-function interpolation.
///
/// Returns 1.0 if `t` is before the first observed event time or if there are
/// no event times recorded.
#[must_use]
pub fn dp_predict_survival(posterior: &DpSurvivalPosterior, t: f64) -> f64 {
    if posterior.times.is_empty() {
        return 1.0;
    }
    if t < posterior.times[0] {
        return 1.0;
    }
    // Find the last index where times[i] <= t.
    let idx = posterior
        .times
        .partition_point(|&ti| ti <= t)
        .saturating_sub(1);
    posterior.mean_survival[idx]
}

// ─────────────────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Draw a sample from Beta(a, b) using Gamma variates.
///
/// Each Gamma variate is obtained via the Marsaglia-Tsang (2000) method.
/// Result is clamped to `[1e-15, 1 − 1e-15]`.
fn sample_beta(a: f64, b: f64, rng: &mut LcgRng) -> f64 {
    let ga = sample_gamma(a, rng);
    let gb = sample_gamma(b, rng);
    let denom = ga + gb;
    if denom <= 0.0 || !denom.is_finite() {
        return 0.5; // fallback for degenerate cases
    }
    (ga / denom).clamp(1.0e-15, 1.0 - 1.0e-15)
}

/// Draw a sample from Gamma(shape, 1) using the Marsaglia-Tsang (2000)
/// squeeze algorithm.
///
/// For shape < 1 we use the relation Gamma(a) = Gamma(a+1) · U^(1/a)
/// where U ~ Uniform(0,1).
fn sample_gamma(shape: f64, rng: &mut LcgRng) -> f64 {
    if shape < 1.0 {
        // Boost to shape+1 then scale back.
        let g = sample_gamma(shape + 1.0, rng);
        let u = rng.next_f64().max(1.0e-300);
        return g * u.powf(1.0 / shape);
    }

    // Marsaglia-Tsang squeeze for shape >= 1.
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        // Draw X ~ N(0,1).
        let x = rng.next_normal();
        let v_inner = 1.0 + c * x;
        if v_inner <= 0.0 {
            continue;
        }
        let v = v_inner * v_inner * v_inner; // v = (1 + c·x)^3

        let u = rng.next_f64().max(1.0e-300);

        // Squeeze acceptance: cheap test first.
        let x2 = x * x;
        if u < 1.0 - 0.0331 * x2 * x2 {
            return d * v;
        }
        // Full log acceptance.
        if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Compute the `p`-th empirical quantile of a **sorted** slice using linear
/// interpolation (Type 7 in R / numpy convention).
fn empirical_quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> DpSurvivalConfig {
        DpSurvivalConfig {
            concentration: 1.0,
            n_sticks: 10,
            n_samples: 50,
            seed: 42,
        }
    }

    fn default_data() -> (Vec<f64>, Vec<bool>) {
        (
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![true, false, true, true, false],
        )
    }

    /// 1. Posterior times vector must be non-empty after fitting.
    #[test]
    fn posterior_has_times() {
        let (times, events) = default_data();
        let cfg = default_cfg();
        let posterior = dp_survival_posterior(&times, &events, &cfg).expect("fit failed");
        assert!(!posterior.times.is_empty(), "times must be non-empty");
    }

    /// 2. Posterior mean survival before the first event time ≈ 1.0
    ///    (queried via dp_predict_survival at t = 0).
    #[test]
    fn posterior_mean_at_t0_approx_1() {
        let (times, events) = default_data();
        let cfg = default_cfg();
        let posterior = dp_survival_posterior(&times, &events, &cfg).expect("fit failed");
        let s0 = dp_predict_survival(&posterior, 0.0);
        assert!((s0 - 1.0).abs() < 1.0e-10, "S(0) should be 1.0, got {s0}");
    }

    /// 3. All posterior mean_survival values must lie in [0, 1].
    #[test]
    fn survival_in_0_1() {
        let (times, events) = default_data();
        let cfg = default_cfg();
        let posterior = dp_survival_posterior(&times, &events, &cfg).expect("fit failed");
        for (i, &s) in posterior.mean_survival.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&s),
                "mean_survival[{i}] = {s} out of [0,1]"
            );
        }
    }

    /// 4. CI ordering: ci_lower[i] <= mean_survival[i] <= ci_upper[i] for all i.
    #[test]
    fn ci_ordering() {
        let (times, events) = default_data();
        let cfg = default_cfg();
        let posterior = dp_survival_posterior(&times, &events, &cfg).expect("fit failed");
        let n = posterior.times.len();
        for i in 0..n {
            assert!(
                posterior.ci_lower[i] <= posterior.mean_survival[i] + 1.0e-12,
                "ci_lower[{i}]={} > mean_survival[{i}]={}",
                posterior.ci_lower[i],
                posterior.mean_survival[i]
            );
            assert!(
                posterior.mean_survival[i] <= posterior.ci_upper[i] + 1.0e-12,
                "mean_survival[{i}]={} > ci_upper[{i}]={}",
                posterior.mean_survival[i],
                posterior.ci_upper[i]
            );
        }
    }

    /// 5. Small concentration (data-driven): should fit without crashing.
    #[test]
    fn concentration_small() {
        let (times, events) = default_data();
        let cfg = DpSurvivalConfig {
            concentration: 0.01,
            n_sticks: 10,
            n_samples: 50,
            seed: 42,
        };
        let result = dp_survival_posterior(&times, &events, &cfg);
        assert!(
            result.is_ok(),
            "small concentration fit failed: {:?}",
            result
        );
        let posterior = result.expect("should be ok");
        assert!(!posterior.times.is_empty());
    }

    /// 6. Large concentration (prior-dominant): should fit without crashing.
    #[test]
    fn large_concentration() {
        let (times, events) = default_data();
        let cfg = DpSurvivalConfig {
            concentration: 100.0,
            n_sticks: 10,
            n_samples: 50,
            seed: 42,
        };
        let result = dp_survival_posterior(&times, &events, &cfg);
        assert!(
            result.is_ok(),
            "large concentration fit failed: {:?}",
            result
        );
        let posterior = result.expect("should be ok");
        assert!(!posterior.times.is_empty());
    }

    /// 7. All-censored data (no events) must return NoEvents error.
    #[test]
    fn empty_data_errors() {
        let times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let events = vec![false, false, false, false, false];
        let cfg = default_cfg();
        let result = dp_survival_posterior(&times, &events, &cfg);
        assert!(
            matches!(result, Err(SurvivalError::NoEvents)),
            "expected NoEvents error, got {:?}",
            result
        );
    }

    /// 8. dp_predict_survival at t = 0 should return 1.0.
    #[test]
    fn predict_before_first_event() {
        let (times, events) = default_data();
        let cfg = default_cfg();
        let posterior = dp_survival_posterior(&times, &events, &cfg).expect("fit failed");
        let s = dp_predict_survival(&posterior, 0.0);
        assert!((s - 1.0).abs() < 1.0e-10, "S(0.0) should be 1.0, got {s}");
    }
}
