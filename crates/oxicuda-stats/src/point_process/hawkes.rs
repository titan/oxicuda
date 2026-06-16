//! Univariate Hawkes self-exciting point process with an exponential kernel.
//!
//! A Hawkes process (Hawkes, 1971) is a point process whose conditional
//! intensity is driven by its own past events:
//!
//! ```text
//! λ(t) = μ + Σ_{tᵢ < t} α · e^{−β (t − tᵢ)}
//! ```
//!
//! - `μ > 0` is the **background** (immigrant) rate;
//! - `α ≥ 0` is the **jump** size each event adds to the intensity;
//! - `β > 0` is the **decay** rate of the excitation.
//!
//! The **branching ratio** `n = α / β` is the expected number of offspring per
//! event; the process is stationary (non-explosive) iff `n < 1`.
//!
//! # Recursive O(n) log-likelihood (Ogata, 1981)
//!
//! For an ordered realisation `0 ≤ t₁ < … < t_N ≤ T`, the exact log-likelihood is
//!
//! ```text
//! ℓ(μ,α,β) = Σᵢ ln λ(tᵢ⁻) − ∫₀ᵀ λ(s) ds
//!          = Σᵢ ln( μ + α Aᵢ ) − μT − (α/β) Σᵢ (1 − e^{−β (T − tᵢ)})
//! ```
//!
//! where `λ(tᵢ⁻)` excludes the event at `tᵢ` itself and `Aᵢ` obeys the linear
//! recursion
//!
//! ```text
//! A₁ = 0,   Aᵢ = e^{−β (tᵢ − t_{i−1})} (1 + A_{i−1})   (i ≥ 2).
//! ```
//!
//! This computes both the summed log-intensity **and** the compensator
//! `Λ(T)=∫₀ᵀλ` in a single O(N) pass, versus the naïve O(N²) double sum.
//!
//! # Simulation
//!
//! [`hawkes_simulate`] uses **Ogata's thinning** algorithm: propose the next
//! point from a homogeneous Poisson process whose rate is the current upper
//! bound on `λ`, then accept it with probability `λ(t)/λ̄`.
//!
//! # References
//! - Hawkes, A.G. (1971). "Spectra of some self-exciting and mutually exciting
//!   point processes". *Biometrika* 58(1):83–90.
//! - Ogata, Y. (1981). "On Lewis' simulation method for point processes".
//!   *IEEE Trans. Information Theory* 27(1):23–31.
//! - Laub, P.J., Taimre, T. & Pollett, P.K. (2015). "Hawkes Processes". arXiv.

use crate::error::{StatsError, StatsResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters `(μ, α, β)` of an exponential-kernel Hawkes process.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HawkesParams {
    /// Background (immigrant) intensity `μ > 0`.
    pub mu: f64,
    /// Excitation jump `α ≥ 0`.
    pub alpha: f64,
    /// Excitation decay `β > 0`.
    pub beta: f64,
}

impl HawkesParams {
    /// Construct parameters, validating positivity / non-negativity.
    pub fn new(mu: f64, alpha: f64, beta: f64) -> StatsResult<Self> {
        if !(mu.is_finite() && alpha.is_finite() && beta.is_finite()) {
            return Err(StatsError::NumericalInstability(
                "Hawkes parameters must be finite".to_string(),
            ));
        }
        if mu <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "mu".to_string(),
                reason: "background rate must be > 0".to_string(),
            });
        }
        if alpha < 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "alpha".to_string(),
                reason: "excitation jump must be ≥ 0".to_string(),
            });
        }
        if beta <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "beta".to_string(),
                reason: "decay rate must be > 0".to_string(),
            });
        }
        Ok(Self { mu, alpha, beta })
    }

    /// Branching ratio `n = α / β` (expected offspring per event).
    #[must_use]
    pub fn branching_ratio(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Whether the process is stationary / sub-critical (`α/β < 1`).
    #[must_use]
    pub fn is_stationary(&self) -> bool {
        self.branching_ratio() < 1.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Intensity & compensator
// ─────────────────────────────────────────────────────────────────────────────

/// Conditional intensity `λ(t)` given the history `events` (strictly before `t`).
///
/// Only events with `tᵢ < t` contribute. With an empty history this is `μ`.
/// Always satisfies `λ(t) ≥ μ` because every kernel term is non-negative.
#[must_use]
pub fn hawkes_intensity(t: f64, events: &[f64], params: &HawkesParams) -> f64 {
    let mut excite = 0.0;
    for &ti in events {
        if ti < t {
            excite += (-params.beta * (t - ti)).exp();
        }
    }
    params.mu + params.alpha * excite
}

/// Compensator `Λ(t) = ∫₀ᵗ λ(s) ds` (the integrated intensity).
///
/// ```text
/// Λ(t) = μ t + (α/β) Σ_{tᵢ ≤ t} (1 − e^{−β (t − tᵢ)}).
/// ```
///
/// This is non-negative and non-decreasing in `t`.
#[must_use]
pub fn hawkes_compensator(t: f64, events: &[f64], params: &HawkesParams) -> f64 {
    let ratio = params.alpha / params.beta;
    let mut sum = 0.0;
    for &ti in events {
        if ti <= t {
            sum += 1.0 - (-params.beta * (t - ti)).exp();
        }
    }
    params.mu * t + ratio * sum
}

// ─────────────────────────────────────────────────────────────────────────────
// Log-likelihood
// ─────────────────────────────────────────────────────────────────────────────

/// Exact O(N) recursive log-likelihood of an ordered event sequence on `[0, T]`.
///
/// `events` must be sorted ascending and lie within `[0, T]`. Returns an error
/// for an unsorted sequence, out-of-range times, or `T ≤ 0`.
pub fn hawkes_log_likelihood(
    events: &[f64],
    horizon: f64,
    params: &HawkesParams,
) -> StatsResult<f64> {
    if horizon <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "horizon".to_string(),
            reason: "observation horizon T must be > 0".to_string(),
        });
    }
    // Validate ordering and range.
    let mut prev = f64::NEG_INFINITY;
    for (i, &t) in events.iter().enumerate() {
        if !t.is_finite() {
            return Err(StatsError::NonFiniteValue(i));
        }
        if t < 0.0 || t > horizon {
            return Err(StatsError::InvalidParameter {
                name: "events".to_string(),
                reason: format!("event {t} at index {i} outside [0, {horizon}]"),
            });
        }
        if t < prev {
            return Err(StatsError::InvalidParameter {
                name: "events".to_string(),
                reason: format!("events must be sorted ascending (index {i})"),
            });
        }
        prev = t;
    }

    let mu = params.mu;
    let alpha = params.alpha;
    let beta = params.beta;
    let n = events.len();

    // Σ ln λ(tᵢ⁻) via the A-recursion.
    let mut sum_log_intensity = 0.0;
    let mut a_prev = 0.0; // A₁ = 0
    for i in 0..n {
        let a_i = if i == 0 {
            0.0
        } else {
            (-beta * (events[i] - events[i - 1])).exp() * (1.0 + a_prev)
        };
        let lambda = mu + alpha * a_i;
        // λ ≥ μ > 0, so the log is always defined.
        sum_log_intensity += lambda.max(f64::MIN_POSITIVE).ln();
        a_prev = a_i;
    }

    // Compensator Λ(T) = μ T + (α/β) Σ (1 − e^{−β (T − tᵢ)}).
    let mut comp_sum = 0.0;
    for &t in events {
        comp_sum += 1.0 - (-beta * (horizon - t)).exp();
    }
    let compensator = mu * horizon + (alpha / beta) * comp_sum;

    Ok(sum_log_intensity - compensator)
}

/// Reference O(N²) log-likelihood via the explicit double sum.
///
/// Numerically equivalent to [`hawkes_log_likelihood`]; provided for testing and
/// validation of the recursive form. Same input contract.
pub fn hawkes_log_likelihood_naive(
    events: &[f64],
    horizon: f64,
    params: &HawkesParams,
) -> StatsResult<f64> {
    if horizon <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "horizon".to_string(),
            reason: "observation horizon T must be > 0".to_string(),
        });
    }
    let mu = params.mu;
    let alpha = params.alpha;
    let beta = params.beta;

    // Σ ln( μ + α Σ_{j<i} e^{−β(tᵢ − tⱼ)} ).
    let mut sum_log_intensity = 0.0;
    for (i, &ti) in events.iter().enumerate() {
        let mut excite = 0.0;
        for &tj in events.iter().take(i) {
            excite += (-beta * (ti - tj)).exp();
        }
        let lambda = mu + alpha * excite;
        sum_log_intensity += lambda.max(f64::MIN_POSITIVE).ln();
    }

    let mut comp_sum = 0.0;
    for &t in events {
        comp_sum += 1.0 - (-beta * (horizon - t)).exp();
    }
    let compensator = mu * horizon + (alpha / beta) * comp_sum;

    Ok(sum_log_intensity - compensator)
}

// ─────────────────────────────────────────────────────────────────────────────
// MLE
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`hawkes_mle`].
#[derive(Debug, Clone)]
pub struct HawkesMleConfig {
    /// Initial guess `(μ, α, β)`.
    pub init: HawkesParams,
    /// Maximum coordinate-ascent sweeps (default 400).
    pub max_iter: usize,
    /// Convergence tolerance on the log-likelihood change (default 1e-8).
    pub tol: f64,
    /// Penalty subtracted from the objective when `α/β ≥ 1` to enforce
    /// stationarity (default 1e6 per unit of overshoot). Set to `0.0` to allow
    /// super-critical fits.
    pub stationarity_penalty: f64,
}

impl Default for HawkesMleConfig {
    fn default() -> Self {
        Self {
            init: HawkesParams {
                mu: 1.0,
                alpha: 0.5,
                beta: 1.0,
            },
            max_iter: 400,
            tol: 1e-8,
            stationarity_penalty: 1e6,
        }
    }
}

/// Result of [`hawkes_mle`].
#[derive(Debug, Clone)]
pub struct HawkesMleResult {
    /// Estimated parameters.
    pub params: HawkesParams,
    /// Maximised (penalised) log-likelihood.
    pub log_likelihood: f64,
    /// Branching ratio `α̂ / β̂`.
    pub branching_ratio: f64,
    /// Whether the estimate satisfies the stationarity condition `α̂/β̂ < 1`.
    pub stationary: bool,
    /// Number of optimisation sweeps executed.
    pub n_iter: usize,
    /// Whether the objective converged within `tol`.
    pub converged: bool,
}

/// Penalised objective: log-likelihood minus a stationarity penalty.
fn penalised_ll(events: &[f64], horizon: f64, params: &HawkesParams, penalty: f64) -> Option<f64> {
    let ll = hawkes_log_likelihood(events, horizon, params).ok()?;
    if !ll.is_finite() {
        return None;
    }
    let ratio = params.branching_ratio();
    let pen = if penalty > 0.0 && ratio >= 1.0 {
        penalty * (ratio - 1.0 + 1e-6)
    } else {
        0.0
    };
    Some(ll - pen)
}

/// Maximum-likelihood estimation of `(μ, α, β)` by bounded coordinate ascent.
///
/// A derivative-free golden-section line search is run on each coordinate in
/// turn (`μ`, then `α`, then `β`), keeping every parameter strictly positive and
/// penalising super-critical (`α/β ≥ 1`) configurations. This is robust and
/// needs no analytic Hessian while still recovering the generating parameters of
/// simulated data to good accuracy.
///
/// Requires at least two events.
pub fn hawkes_mle(
    events: &[f64],
    horizon: f64,
    cfg: &HawkesMleConfig,
) -> StatsResult<HawkesMleResult> {
    if events.len() < 2 {
        return Err(StatsError::InsufficientSampleSize {
            got: events.len(),
            need: 2,
        });
    }
    if horizon <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "horizon".to_string(),
            reason: "observation horizon T must be > 0".to_string(),
        });
    }
    // Validate the sequence once via the likelihood at the initial point.
    let _ = hawkes_log_likelihood(events, horizon, &cfg.init)?;

    let mut mu = cfg.init.mu.max(1e-6);
    let mut alpha = cfg.init.alpha.max(0.0);
    let mut beta = cfg.init.beta.max(1e-6);
    let penalty = cfg.stationarity_penalty;

    let eval = |mu: f64, alpha: f64, beta: f64| -> f64 {
        let p = HawkesParams { mu, alpha, beta };
        penalised_ll(events, horizon, &p, penalty).unwrap_or(f64::NEG_INFINITY)
    };

    let mut best = eval(mu, alpha, beta);
    if !best.is_finite() {
        return Err(StatsError::NumericalInstability(
            "initial Hawkes log-likelihood is not finite".to_string(),
        ));
    }

    let mut converged = false;
    let mut n_iter = 0usize;

    for sweep in 0..cfg.max_iter {
        n_iter = sweep + 1;
        let prev_best = best;

        // μ-search on [1e-6, 10·current].
        mu = golden_section(1e-6, (mu * 10.0).max(1.0), |m| eval(m, alpha, beta), 60);

        // α-search on [0, 10·current] (allow zero excitation).
        alpha = golden_section(0.0, (alpha * 10.0).max(1.0), |a| eval(mu, a, beta), 60);

        // β-search on [1e-6, 10·current].
        beta = golden_section(1e-6, (beta * 10.0).max(1.0), |b| eval(mu, alpha, b), 60);

        // Best objective after this full sweep over all three coordinates.
        best = eval(mu, alpha, beta);

        if (best - prev_best).abs() < cfg.tol {
            converged = true;
            break;
        }
    }

    let params = HawkesParams { mu, alpha, beta };
    let log_likelihood = hawkes_log_likelihood(events, horizon, &params)?;
    let ratio = params.branching_ratio();

    Ok(HawkesMleResult {
        params,
        log_likelihood,
        branching_ratio: ratio,
        stationary: ratio < 1.0,
        n_iter,
        converged,
    })
}

/// Golden-section maximisation of a unimodal-ish `f` on `[lo, hi]`.
fn golden_section<F: Fn(f64) -> f64>(lo: f64, hi: f64, f: F, iters: usize) -> f64 {
    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0; // 1/φ ≈ 0.618
    let mut a = lo;
    let mut b = hi;
    if b <= a {
        return lo.max(0.0);
    }
    let mut c = b - inv_phi * (b - a);
    let mut d = a + inv_phi * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..iters {
        if fc < fd {
            a = c;
            c = d;
            fc = fd;
            d = a + inv_phi * (b - a);
            fd = f(d);
        } else {
            b = d;
            d = c;
            fd = fc;
            c = b - inv_phi * (b - a);
            fc = f(c);
        }
        if (b - a).abs() < 1e-10 {
            break;
        }
    }
    0.5 * (a + b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Simulation (Ogata thinning)
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate a Hawkes process on `[0, T]` by Ogata's thinning algorithm.
///
/// Returns the ordered event times. `seed` drives the workspace [`LcgRng`].
/// Requires a stationary specification when `params.alpha > 0`; a super-critical
/// process may yield an unbounded number of points, so an explicit safety cap of
/// `max_events` aborts runaway simulations with an error.
pub fn hawkes_simulate(
    params: &HawkesParams,
    horizon: f64,
    seed: u64,
    max_events: usize,
) -> StatsResult<Vec<f64>> {
    if horizon <= 0.0 {
        return Err(StatsError::InvalidParameter {
            name: "horizon".to_string(),
            reason: "T must be > 0".to_string(),
        });
    }
    let mut rng = LcgRng::new(seed);
    let mut events: Vec<f64> = Vec::new();
    let mut t = 0.0_f64;

    while t < horizon {
        // Upper bound λ̄ on [t, ∞): intensity is largest immediately after t and
        // decays, so the current intensity (just after t) bounds the near future.
        let lambda_bar = hawkes_intensity(t + 1e-12, &events, params).max(params.mu);
        if lambda_bar <= 0.0 {
            break;
        }
        // Next homogeneous-Poisson candidate.
        let u = rng.next_f64().max(1e-300);
        let w = -u.ln() / lambda_bar;
        t += w;
        if t >= horizon {
            break;
        }
        // Thinning acceptance.
        let lambda_t = hawkes_intensity(t, &events, params);
        let d = rng.next_f64();
        if d * lambda_bar <= lambda_t {
            events.push(t);
            if events.len() > max_events {
                return Err(StatsError::NumericalInstability(format!(
                    "Hawkes simulation exceeded max_events ({max_events}); \
                     is the process super-critical (α/β = {})?",
                    params.branching_ratio()
                )));
            }
        }
    }
    Ok(events)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p(mu: f64, alpha: f64, beta: f64) -> HawkesParams {
        HawkesParams::new(mu, alpha, beta).expect("valid params")
    }

    // (a) Recursive O(n) log-likelihood equals the O(n²) double sum.
    #[test]
    fn recursive_equals_naive() {
        let events = vec![0.3, 0.7, 1.1, 2.5, 3.0, 4.2, 5.9, 6.0, 7.7, 9.1];
        let horizon = 10.0;
        for params in [p(0.5, 0.8, 1.5), p(1.0, 0.2, 0.6), p(0.3, 1.0, 2.0)] {
            let fast = hawkes_log_likelihood(&events, horizon, &params).expect("fast");
            let slow = hawkes_log_likelihood_naive(&events, horizon, &params).expect("slow");
            assert!(
                (fast - slow).abs() < 1e-8,
                "recursive {fast} vs naive {slow}"
            );
        }
    }

    // (b) MLE recovers (μ, α, β) from data simulated by thinning.
    #[test]
    fn mle_recovers_parameters() {
        // Generate a long, sub-critical realisation.
        let true_p = p(0.6, 0.5, 1.4);
        let horizon = 4000.0;
        let events = hawkes_simulate(&true_p, horizon, 2024, 1_000_000).expect("sim");
        assert!(
            events.len() > 200,
            "need enough events, got {}",
            events.len()
        );

        let cfg = HawkesMleConfig {
            init: p(1.0, 0.4, 1.0),
            ..Default::default()
        };
        let res = hawkes_mle(&events, horizon, &cfg).expect("mle");

        // Recovery within generous tolerance (stochastic data).
        assert!(
            (res.params.mu - true_p.mu).abs() < 0.25,
            "mu {} vs {}",
            res.params.mu,
            true_p.mu
        );
        // Branching ratio is the most stable identifiable quantity.
        assert!(
            (res.branching_ratio - true_p.branching_ratio()).abs() < 0.2,
            "branching {} vs {}",
            res.branching_ratio,
            true_p.branching_ratio()
        );
        assert!(res.stationary, "fitted process should be stationary");
    }

    // (c) Stationarity requires α/β < 1; the penalty steers MLE sub-critical.
    #[test]
    fn stationarity_flag_and_penalty() {
        let sub = p(1.0, 0.5, 1.0);
        assert!(sub.is_stationary());
        assert!((sub.branching_ratio() - 0.5).abs() < 1e-12);

        let sup = p(1.0, 2.0, 1.0);
        assert!(!sup.is_stationary());
        assert!(sup.branching_ratio() > 1.0);

        // With the penalty active, an MLE on sub-critical data stays sub-critical.
        let true_p = p(0.8, 0.4, 1.2);
        let events = hawkes_simulate(&true_p, 2000.0, 77, 1_000_000).expect("sim");
        let cfg = HawkesMleConfig::default();
        let res = hawkes_mle(&events, 2000.0, &cfg).expect("mle");
        assert!(res.branching_ratio < 1.0, "ratio {}", res.branching_ratio);
    }

    // (d) Intensity λ(t) ≥ μ everywhere.
    #[test]
    fn intensity_at_least_mu() {
        let params = p(0.7, 0.9, 1.3);
        let events = vec![0.2, 0.5, 1.0, 1.8, 3.3];
        for k in 0..200 {
            let t = k as f64 * 0.05;
            let lam = hawkes_intensity(t, &events, &params);
            assert!(lam >= params.mu - 1e-12, "λ({t})={lam} < μ");
        }
    }

    // (e) Compensator Λ(t) is non-decreasing.
    #[test]
    fn compensator_non_decreasing() {
        let params = p(0.5, 0.7, 1.1);
        let events = vec![0.4, 0.9, 1.5, 2.7, 3.6, 4.0];
        let mut prev = f64::NEG_INFINITY;
        for k in 0..200 {
            let t = k as f64 * 0.03;
            let comp = hawkes_compensator(t, &events, &params);
            assert!(comp >= prev - 1e-12, "compensator decreased at t={t}");
            assert!(comp >= 0.0, "compensator negative");
            prev = comp;
        }
    }

    // (f) Clustering: a Hawkes count is over-dispersed vs a Poisson of equal mean.
    #[test]
    fn hawkes_burstier_than_poisson() {
        // Strong excitation → pronounced clustering.
        let params = p(0.5, 0.8, 1.0);
        let window = 5.0_f64;
        let horizon = window * 400.0;

        // Count events per fixed window for the Hawkes realisation.
        let events = hawkes_simulate(&params, horizon, 999, 5_000_000).expect("sim");
        let n_windows = (horizon / window) as usize;
        let mut counts = vec![0usize; n_windows];
        for &t in &events {
            let idx = ((t / window) as usize).min(n_windows - 1);
            counts[idx] += 1;
        }
        let mean = counts.iter().sum::<usize>() as f64 / n_windows as f64;
        let var = counts
            .iter()
            .map(|&c| {
                let d = c as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n_windows as f64;
        // Fano factor = Var/Mean; > 1 indicates over-dispersion (clustering).
        // For Poisson it is exactly 1. Require a clear excess.
        let fano = var / mean.max(1e-9);
        assert!(
            fano > 1.2,
            "Hawkes Fano factor {fano} should exceed Poisson's 1.0 (mean {mean}, var {var})"
        );
    }

    // (g) Empty history → λ(t) = μ.
    #[test]
    fn empty_history_intensity_is_mu() {
        let params = p(1.3, 0.5, 2.0);
        assert!((hawkes_intensity(0.0, &[], &params) - params.mu).abs() < 1e-15);
        assert!((hawkes_intensity(5.0, &[], &params) - params.mu).abs() < 1e-15);
        // Compensator with no events is exactly μ t.
        assert!((hawkes_compensator(3.0, &[], &params) - params.mu * 3.0).abs() < 1e-12);
    }

    // Parameter validation.
    #[test]
    fn invalid_params_error() {
        assert!(HawkesParams::new(0.0, 0.5, 1.0).is_err()); // μ ≤ 0
        assert!(HawkesParams::new(1.0, -0.1, 1.0).is_err()); // α < 0
        assert!(HawkesParams::new(1.0, 0.5, 0.0).is_err()); // β ≤ 0
        assert!(HawkesParams::new(f64::NAN, 0.5, 1.0).is_err());
    }

    // Unsorted events rejected.
    #[test]
    fn unsorted_events_error() {
        let params = p(1.0, 0.5, 1.0);
        let events = vec![1.0, 0.5, 2.0];
        assert!(hawkes_log_likelihood(&events, 3.0, &params).is_err());
    }

    // Single / empty event likelihood still defined; MLE needs ≥ 2.
    #[test]
    fn mle_needs_two_events() {
        let cfg = HawkesMleConfig::default();
        assert!(hawkes_mle(&[1.0], 2.0, &cfg).is_err());
        assert!(hawkes_mle(&[], 2.0, &cfg).is_err());
    }

    // Simulation respects the horizon and returns sorted events.
    #[test]
    fn simulation_sorted_and_in_horizon() {
        let params = p(1.0, 0.3, 1.0);
        let horizon = 50.0;
        let events = hawkes_simulate(&params, horizon, 5, 1_000_000).expect("sim");
        for w in events.windows(2) {
            assert!(w[0] <= w[1], "events not sorted");
        }
        assert!(events.iter().all(|&t| (0.0..horizon).contains(&t)));
    }

    // Likelihood increases for parameters closer to the truth (sanity of the
    // objective surface used by the MLE).
    #[test]
    fn likelihood_prefers_truth() {
        let true_p = p(0.7, 0.5, 1.3);
        let events = hawkes_simulate(&true_p, 3000.0, 314, 1_000_000).expect("sim");
        let ll_true = hawkes_log_likelihood(&events, 3000.0, &true_p).expect("ll");
        // A clearly wrong parameter set should score lower.
        let wrong = p(3.0, 0.1, 5.0);
        let ll_wrong = hawkes_log_likelihood(&events, 3000.0, &wrong).expect("ll");
        assert!(
            ll_true > ll_wrong,
            "truth ll {ll_true} should exceed wrong ll {ll_wrong}"
        );
    }
}
