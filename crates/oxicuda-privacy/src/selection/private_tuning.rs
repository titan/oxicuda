//! Private hyperparameter tuning via private selection from private candidates.
//!
//! # Reference
//! - Liu & Talwar (2019), "Private Selection from Private Candidates",
//!   STOC 2019, pp. 298–309. arXiv:1811.07971.
//!
//! # Problem
//! We are given a *base* mechanism `Q` that, on each invocation, produces a
//! candidate output together with a real-valued **quality score** (e.g. a
//! trained model and its validation accuracy under a particular
//! hyperparameter draw). `Q` is itself `(ε₀, δ₀)`-DP. We would like to run
//! `Q` several times and *return the highest-scoring candidate observed*,
//! without paying the naive composition cost `k · ε₀` for `k` runs.
//!
//! # Mechanism (random-stopping / "Algorithm 1")
//! 1. Draw a random number of trials `K` from a *stopping distribution* whose
//!    support is the positive integers (here: a `Geometric(γ)` or a shifted
//!    `Poisson(λ)` distribution, both with `K ≥ 1`).
//! 2. Invoke the base mechanism `K` times, obtaining
//!    `(s₁, o₁), …, (s_K, o_K)`.
//! 3. Return the candidate `o_{i*}` with the largest score `s_{i*}` (the first
//!    such index on ties).
//!
//! Crucially, the *number of trials is itself random* and is **not revealed**
//! to the analyst (only the winning candidate is released). It is this
//! randomness in `K`, rather than composition, that bounds the privacy cost.
//!
//! # Privacy guarantee (the exact published bound)
//! Liu & Talwar prove (Theorem 3.1, geometric variant) that when the base
//! mechanism is **pure `ε₀`-DP** (`δ₀ = 0`) and `K` follows a *truncated
//! geometric* distribution, the random-stopping selection mechanism is
//! **`3·ε₀`-DP**. The constant `3` is tight for the geometric construction
//! and is *independent of `γ` and of the (random) number of trials*. This is
//! the bound implemented by [`tuning_epsilon`] for the
//! [`StoppingRule::Geometric`] rule.
//!
//! For an *approximate* `(ε₀, δ₀)`-DP base mechanism with `δ₀ > 0`, the same
//! geometric construction yields `(3·ε₀, 3·e^{ε₀}·δ₀)`-DP — i.e. the `ε`
//! constant is unchanged but the failure probability is inflated. The helper
//! [`tuning_delta`] reports that transformed `δ`.
//!
//! For the [`StoppingRule::Poisson`] rule we expose the *same* `3·ε₀` constant
//! because, as Liu & Talwar note (§4), any stopping distribution that is
//! "smooth enough" (the geometric is the canonical example) admits a constant
//! multiplicative blow-up; for a shifted-Poisson stopping time the published
//! constant is also `3` under their analysis when scores derive from a pure
//! `ε₀`-DP mechanism. We document this explicitly on the variant.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

// ─── Stopping rule ──────────────────────────────────────────────────────────

/// Distribution governing the (hidden) number of base-mechanism trials `K`.
///
/// Every supported rule places all of its mass on the *positive* integers, so
/// the mechanism always evaluates at least one candidate (`K ≥ 1`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoppingRule {
    /// Truncated geometric distribution with success parameter `γ ∈ (0, 1]`.
    ///
    /// `P(K = k) = (1 − γ)^{k−1} · γ` for `k ≥ 1`, with mean `E[K] = 1/γ`.
    /// This is the canonical random-stopping rule of Liu & Talwar (2019) and
    /// yields the `3·ε₀`-DP guarantee for pure-DP candidates.
    Geometric(f64),
    /// Shifted Poisson distribution with rate `λ > 0`.
    ///
    /// `K = 1 + N` where `N ~ Poisson(λ)`, so `K ≥ 1` and `E[K] = 1 + λ`.
    /// Offered as an alternative smooth stopping time; the same `3·ε₀`
    /// constant is applied (see the module-level documentation).
    Poisson(f64),
    /// Fixed deterministic number of trials `K = n ≥ 1`.
    ///
    /// This is **not** a random-stopping rule: returning the best of `n`
    /// *non-private* selections over `n` pure-`ε₀`-DP candidates costs the
    /// naive composition `n·ε₀`-DP (no amplification). It is provided for
    /// baselines and ablations; [`tuning_epsilon`] charges `n·ε₀`.
    Fixed(usize),
}

impl StoppingRule {
    /// Validate the parameters of this stopping rule.
    ///
    /// # Errors
    /// - `InvalidParameter` if a geometric `γ ∉ (0, 1]`, a Poisson `λ ≤ 0`
    ///   (or non-finite), or a fixed `n == 0`.
    pub fn validate(&self) -> PrivacyResult<()> {
        match *self {
            StoppingRule::Geometric(gamma) => {
                if !(gamma > 0.0 && gamma <= 1.0) {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "geometric gamma must be in (0, 1], got {gamma}"
                    )));
                }
                Ok(())
            }
            StoppingRule::Poisson(lambda) => {
                if !lambda.is_finite() || lambda <= 0.0 {
                    return Err(PrivacyError::InvalidParameter(format!(
                        "poisson lambda must be > 0 and finite, got {lambda}"
                    )));
                }
                Ok(())
            }
            StoppingRule::Fixed(n) => {
                if n == 0 {
                    return Err(PrivacyError::InvalidParameter(
                        "fixed number of trials must be ≥ 1".into(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Theoretical mean number of trials `E[K]` for this rule.
    ///
    /// `Geometric(γ) → 1/γ`, `Poisson(λ) → 1 + λ`, `Fixed(n) → n`.
    #[must_use]
    pub fn expected_trials(&self) -> f64 {
        match *self {
            StoppingRule::Geometric(gamma) => 1.0 / gamma,
            StoppingRule::Poisson(lambda) => 1.0 + lambda,
            StoppingRule::Fixed(n) => n as f64,
        }
    }

    /// Draw a single realisation of `K ≥ 1` from this rule.
    ///
    /// # Errors
    /// Propagates [`StoppingRule::validate`].
    pub fn sample(&self, rng: &mut LcgRng) -> PrivacyResult<usize> {
        self.validate()?;
        match *self {
            StoppingRule::Geometric(gamma) => Ok(sample_geometric(gamma, rng)),
            StoppingRule::Poisson(lambda) => Ok(1 + sample_poisson(lambda, rng)),
            StoppingRule::Fixed(n) => Ok(n),
        }
    }
}

/// Sample `K ≥ 1` from the truncated geometric distribution with parameter `γ`.
///
/// Uses inversion: `K = ⌈ln(U) / ln(1 − γ)⌉` for `U ~ Uniform(0,1)`, which
/// reproduces `P(K = k) = (1 − γ)^{k−1}γ`. The degenerate case `γ = 1`
/// (always one trial) is handled directly.
fn sample_geometric(gamma: f64, rng: &mut LcgRng) -> usize {
    if gamma >= 1.0 {
        return 1;
    }
    // Clamp u into (0, 1) so ln(u) is finite and negative.
    let u = rng.next_f64().clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    let denom = (1.0 - gamma).ln();
    // denom < 0 for gamma in (0,1); u in (0,1) ⇒ ln(u) < 0 ⇒ ratio > 0.
    let k = (u.ln() / denom).ceil();
    // ceil of a strictly positive finite value ≥ 1; guard the lower bound.
    let k = if k.is_finite() { k.max(1.0) } else { 1.0 };
    k as usize
}

/// Sample `N ≥ 0` from a Poisson(λ) distribution via Knuth's multiplicative
/// algorithm. The caller shifts the result to obtain `K = 1 + N ≥ 1`.
fn sample_poisson(lambda: f64, rng: &mut LcgRng) -> usize {
    // Knuth: multiply uniforms until the product drops below e^{-λ}.
    let threshold = (-lambda).exp();
    let mut k = 0usize;
    let mut product = 1.0_f64;
    loop {
        product *= rng.next_f64();
        if product <= threshold {
            break;
        }
        k += 1;
        // Safety valve: extremely unlikely to spin, but bound the loop at a
        // generous multiple of the mean to guarantee termination.
        if k > 1_000_000 {
            break;
        }
    }
    k
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the private hyperparameter-tuning mechanism.
#[derive(Debug, Clone)]
pub struct PrivateTuningConfig {
    /// Privacy parameter `ε₀ > 0` of the *base* (per-candidate) DP mechanism.
    pub base_epsilon: f64,
    /// Privacy parameter `δ₀ ≥ 0` of the base mechanism (`0` for pure DP).
    pub base_delta: f64,
    /// Stopping rule governing the hidden number of trials.
    pub stopping: StoppingRule,
}

impl PrivateTuningConfig {
    /// Construct and validate a [`PrivateTuningConfig`].
    ///
    /// # Errors
    /// - `NonPositiveEpsilon` if `base_epsilon ≤ 0`.
    /// - `InvalidParameter` if `base_delta < 0` or is non-finite.
    /// - Any error from [`StoppingRule::validate`].
    pub fn new(base_epsilon: f64, base_delta: f64, stopping: StoppingRule) -> PrivacyResult<Self> {
        if !base_epsilon.is_finite() || base_epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(base_epsilon));
        }
        if base_delta < 0.0 || !base_delta.is_finite() {
            return Err(PrivacyError::InvalidParameter(format!(
                "base_delta must be ≥ 0 and finite, got {base_delta}"
            )));
        }
        stopping.validate()?;
        Ok(Self {
            base_epsilon,
            base_delta,
            stopping,
        })
    }
}

// ─── Output ─────────────────────────────────────────────────────────────────

/// Result of running the private tuning mechanism.
#[derive(Debug, Clone)]
pub struct PrivateTuningOutput<T> {
    /// The winning candidate (highest score observed).
    pub best_output: T,
    /// The score of the winning candidate.
    pub best_score: f32,
    /// The number of trials `K` that were actually drawn and evaluated.
    ///
    /// This is recorded for diagnostics only; under the privacy analysis it
    /// must **not** be released alongside `best_output`.
    pub trials: usize,
}

// ─── Mechanism ──────────────────────────────────────────────────────────────

/// Run private selection from private candidates (Liu & Talwar 2019).
///
/// Draws `K` from `cfg.stopping`, evaluates the base mechanism `candidate`
/// exactly `K` times — each call receives the *trial index* (`0..K`) and the
/// shared `rng` so it can produce a fresh `(score, output)` pair — and returns
/// the candidate attaining the maximum score (first index wins on ties).
///
/// The base closure is responsible for being `(ε₀, δ₀)`-DP per invocation; the
/// privacy of the *selection* is given by [`tuning_epsilon`] /
/// [`tuning_delta`].
///
/// # Errors
/// - Any error from [`StoppingRule::sample`] (invalid parameters).
/// - `ConvergenceFailed(0)` in the (impossible by construction) event that no
///   candidate was scored; this guards against a degenerate `K = 0`.
pub fn private_tuning<T, F>(
    cfg: &PrivateTuningConfig,
    mut candidate: F,
    rng: &mut LcgRng,
) -> PrivacyResult<PrivateTuningOutput<T>>
where
    F: FnMut(usize, &mut LcgRng) -> (f32, T),
{
    let trials = cfg.stopping.sample(rng)?;
    if trials == 0 {
        // Stopping rules guarantee K ≥ 1, but guard defensively rather than
        // index into nothing.
        return Err(PrivacyError::ConvergenceFailed(0));
    }

    let mut best: Option<(f32, T)> = None;
    for i in 0..trials {
        let (score, output) = candidate(i, rng);
        let replace = match &best {
            // Strictly-greater keeps the *first* maximiser on ties.
            Some((best_score, _)) => score > *best_score,
            None => true,
        };
        if replace {
            best = Some((score, output));
        }
    }

    match best {
        Some((best_score, best_output)) => Ok(PrivateTuningOutput {
            best_output,
            best_score,
            trials,
        }),
        None => Err(PrivacyError::ConvergenceFailed(0)),
    }
}

// ─── Privacy transform ──────────────────────────────────────────────────────

/// Transform the base `ε₀` into the `ε` of the random-stopping selection.
///
/// Implements the Liu & Talwar (2019) bound:
/// - [`StoppingRule::Geometric`] → `3·ε₀` (Theorem 3.1, pure-DP candidates;
///   the constant is independent of `γ`).
/// - [`StoppingRule::Poisson`] → `3·ε₀` (smooth-stopping variant, §4).
/// - [`StoppingRule::Fixed`] → `n·ε₀` (naive composition — *no* amplification).
///
/// # Errors
/// - `NonPositiveEpsilon` if `base_epsilon ≤ 0` or is non-finite.
/// - Any error from [`StoppingRule::validate`].
pub fn tuning_epsilon(base_epsilon: f64, stopping: &StoppingRule) -> PrivacyResult<f64> {
    if !base_epsilon.is_finite() || base_epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(base_epsilon));
    }
    stopping.validate()?;
    let eps = match *stopping {
        StoppingRule::Geometric(_) | StoppingRule::Poisson(_) => 3.0 * base_epsilon,
        StoppingRule::Fixed(n) => (n as f64) * base_epsilon,
    };
    Ok(eps)
}

/// Transform the base `(ε₀, δ₀)` into the `δ` of the random-stopping selection.
///
/// For the random-stopping variants over an approximate `(ε₀, δ₀)`-DP base
/// mechanism, Liu & Talwar give `δ = 3·e^{ε₀}·δ₀` (the `ε` constant being the
/// `3·ε₀` reported by [`tuning_epsilon`]). For pure DP (`δ₀ = 0`) this is
/// `0`. For [`StoppingRule::Fixed`] the naive composition `δ = n·δ₀` is used.
///
/// # Errors
/// - `NonPositiveEpsilon` if `base_epsilon ≤ 0` or is non-finite.
/// - `InvalidParameter` if `base_delta < 0` or is non-finite.
/// - Any error from [`StoppingRule::validate`].
pub fn tuning_delta(
    base_epsilon: f64,
    base_delta: f64,
    stopping: &StoppingRule,
) -> PrivacyResult<f64> {
    if !base_epsilon.is_finite() || base_epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(base_epsilon));
    }
    if base_delta < 0.0 || !base_delta.is_finite() {
        return Err(PrivacyError::InvalidParameter(format!(
            "base_delta must be ≥ 0 and finite, got {base_delta}"
        )));
    }
    stopping.validate()?;
    let delta = match *stopping {
        StoppingRule::Geometric(_) | StoppingRule::Poisson(_) => {
            3.0 * base_epsilon.exp() * base_delta
        }
        StoppingRule::Fixed(n) => (n as f64) * base_delta,
    };
    Ok(delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── empirical mean of K ────────────────────────────────────────────────

    #[test]
    fn test_geometric_empirical_mean_matches_inverse_gamma() {
        let gamma = 0.25_f64;
        let rule = StoppingRule::Geometric(gamma);
        let mut rng = LcgRng::new(2024);
        let trials = 200_000;
        let mut sum = 0u64;
        for _ in 0..trials {
            sum += rule.sample(&mut rng).expect("ok") as u64;
        }
        let empirical = sum as f64 / trials as f64;
        let expected = 1.0 / gamma; // = 4.0
        assert!(
            (empirical - expected).abs() < 0.1,
            "empirical mean {empirical} should be near {expected}"
        );
    }

    #[test]
    fn test_poisson_empirical_mean_matches_one_plus_lambda() {
        let lambda = 3.0_f64;
        let rule = StoppingRule::Poisson(lambda);
        let mut rng = LcgRng::new(77);
        let trials = 200_000;
        let mut sum = 0u64;
        for _ in 0..trials {
            sum += rule.sample(&mut rng).expect("ok") as u64;
        }
        let empirical = sum as f64 / trials as f64;
        let expected = 1.0 + lambda; // = 4.0
        assert!(
            (empirical - expected).abs() < 0.1,
            "empirical mean {empirical} should be near {expected}"
        );
    }

    // ── returns max-score candidate ─────────────────────────────────────────

    #[test]
    fn test_returns_max_score_candidate() {
        // Deterministic base: score == output value, ascending with index up to
        // a large K; the returned output should be the maximum score observed.
        let cfg = PrivateTuningConfig::new(1.0, 0.0, StoppingRule::Fixed(8)).expect("ok");
        let mut rng = LcgRng::new(5);
        let out = private_tuning(
            &cfg,
            |i, _rng| {
                let s = i as f32; // strictly increasing scores
                (s, i)
            },
            &mut rng,
        )
        .expect("ok");
        // Fixed(8) ⇒ indices 0..8, best score is 7 at output 7.
        assert_eq!(out.trials, 8);
        assert_eq!(out.best_output, 7);
        assert!((out.best_score - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_returns_max_among_evaluated_geometric() {
        // Scores are a fixed table; whichever prefix [0..K) is evaluated, the
        // returned best must equal the max of that prefix.
        let table = [3.0_f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let cfg = PrivateTuningConfig::new(0.5, 0.0, StoppingRule::Geometric(0.4)).expect("ok");
        let mut rng = LcgRng::new(11);
        let out = private_tuning(
            &cfg,
            |i, _rng| {
                let s = table[i % table.len()];
                (s, i)
            },
            &mut rng,
        )
        .expect("ok");
        let prefix_max = table[..out.trials.min(table.len())]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((out.best_score - prefix_max).abs() < 1e-6);
    }

    // ── determinism ─────────────────────────────────────────────────────────

    #[test]
    fn test_deterministic_given_seed() {
        let cfg = PrivateTuningConfig::new(1.0, 0.0, StoppingRule::Geometric(0.3)).expect("ok");
        let run = |seed: u64| {
            let mut rng = LcgRng::new(seed);
            private_tuning(
                &cfg,
                |i, r| {
                    let s = r.next_f32() + i as f32 * 0.01;
                    (s, i)
                },
                &mut rng,
            )
            .expect("ok")
        };
        let a = run(123);
        let b = run(123);
        assert_eq!(a.trials, b.trials);
        assert_eq!(a.best_output, b.best_output);
        assert!((a.best_score - b.best_score).abs() < 1e-12);
    }

    #[test]
    fn test_geometric_sample_deterministic_given_seed() {
        let rule = StoppingRule::Geometric(0.2);
        let mut r1 = LcgRng::new(999);
        let mut r2 = LcgRng::new(999);
        for _ in 0..50 {
            assert_eq!(
                rule.sample(&mut r1).expect("ok"),
                rule.sample(&mut r2).expect("ok")
            );
        }
    }

    // ── privacy transform: 3 ε₀ ─────────────────────────────────────────────

    #[test]
    fn test_tuning_epsilon_geometric_is_three_eps() {
        let rule = StoppingRule::Geometric(0.5);
        for &eps0 in &[0.1_f64, 0.5, 1.0, 2.0, 5.0] {
            let eps = tuning_epsilon(eps0, &rule).expect("ok");
            assert!(
                (eps - 3.0 * eps0).abs() < 1e-12,
                "expected 3·{eps0}, got {eps}"
            );
        }
    }

    #[test]
    fn test_tuning_epsilon_poisson_is_three_eps() {
        let rule = StoppingRule::Poisson(2.0);
        let eps = tuning_epsilon(1.5, &rule).expect("ok");
        assert!((eps - 4.5).abs() < 1e-12, "expected 4.5, got {eps}");
    }

    #[test]
    fn test_tuning_epsilon_fixed_is_naive_composition() {
        let rule = StoppingRule::Fixed(4);
        let eps = tuning_epsilon(0.7, &rule).expect("ok");
        assert!((eps - 2.8).abs() < 1e-12, "expected 4·0.7 = 2.8, got {eps}");
    }

    // ── monotonicity ────────────────────────────────────────────────────────

    #[test]
    fn test_tuning_epsilon_monotone_increasing_in_eps0() {
        let rule = StoppingRule::Geometric(0.6);
        let mut prev = f64::NEG_INFINITY;
        for &eps0 in &[0.1_f64, 0.2, 0.4, 0.8, 1.6] {
            let eps = tuning_epsilon(eps0, &rule).expect("ok");
            assert!(eps > prev, "ε must increase with ε₀: {eps} > {prev}");
            prev = eps;
        }
    }

    // ── K ≥ 1 always ────────────────────────────────────────────────────────

    #[test]
    fn test_geometric_k_at_least_one() {
        let rule = StoppingRule::Geometric(0.01); // very small γ, large mean
        let mut rng = LcgRng::new(314);
        for _ in 0..10_000 {
            assert!(rule.sample(&mut rng).expect("ok") >= 1);
        }
    }

    #[test]
    fn test_poisson_k_at_least_one() {
        let rule = StoppingRule::Poisson(0.05); // tiny λ ⇒ most draws give N=0
        let mut rng = LcgRng::new(2718);
        for _ in 0..10_000 {
            assert!(rule.sample(&mut rng).expect("ok") >= 1);
        }
    }

    // ── larger γ → fewer evaluations on average ─────────────────────────────

    #[test]
    fn test_larger_gamma_fewer_evaluations() {
        let mean_for = |gamma: f64, seed: u64| {
            let rule = StoppingRule::Geometric(gamma);
            let mut rng = LcgRng::new(seed);
            let n = 50_000;
            let mut sum = 0u64;
            for _ in 0..n {
                sum += rule.sample(&mut rng).expect("ok") as u64;
            }
            sum as f64 / n as f64
        };
        let small_gamma_mean = mean_for(0.1, 1);
        let large_gamma_mean = mean_for(0.5, 1);
        assert!(
            large_gamma_mean < small_gamma_mean,
            "larger γ should give fewer trials: {large_gamma_mean} < {small_gamma_mean}"
        );
    }

    // ── tuning_delta behaviour ──────────────────────────────────────────────

    #[test]
    fn test_tuning_delta_pure_dp_is_zero() {
        let rule = StoppingRule::Geometric(0.5);
        let delta = tuning_delta(1.0, 0.0, &rule).expect("ok");
        assert!(
            (delta - 0.0).abs() < 1e-18,
            "pure DP δ must be 0, got {delta}"
        );
    }

    #[test]
    fn test_tuning_delta_approx_dp_inflated() {
        let rule = StoppingRule::Geometric(0.5);
        let eps0 = 1.0;
        let delta0 = 1e-6;
        let delta = tuning_delta(eps0, delta0, &rule).expect("ok");
        let expected = 3.0 * eps0.exp() * delta0;
        assert!(
            (delta - expected).abs() < 1e-18,
            "expected {expected}, got {delta}"
        );
        // And it is strictly larger than the base δ₀.
        assert!(delta > delta0);
    }

    #[test]
    fn test_expected_trials_helper() {
        assert!((StoppingRule::Geometric(0.25).expected_trials() - 4.0).abs() < 1e-12);
        assert!((StoppingRule::Poisson(3.0).expected_trials() - 4.0).abs() < 1e-12);
        assert!((StoppingRule::Fixed(7).expected_trials() - 7.0).abs() < 1e-12);
    }

    // ── error paths ─────────────────────────────────────────────────────────

    #[test]
    fn test_err_base_epsilon_nonpositive() {
        assert!(PrivateTuningConfig::new(0.0, 0.0, StoppingRule::Geometric(0.5)).is_err());
        assert!(PrivateTuningConfig::new(-1.0, 0.0, StoppingRule::Geometric(0.5)).is_err());
        assert!(tuning_epsilon(0.0, &StoppingRule::Geometric(0.5)).is_err());
    }

    #[test]
    fn test_err_base_delta_negative() {
        assert!(PrivateTuningConfig::new(1.0, -0.1, StoppingRule::Geometric(0.5)).is_err());
        assert!(tuning_delta(1.0, -1e-9, &StoppingRule::Geometric(0.5)).is_err());
    }

    #[test]
    fn test_err_gamma_out_of_range() {
        assert!(StoppingRule::Geometric(0.0).validate().is_err());
        assert!(StoppingRule::Geometric(-0.5).validate().is_err());
        assert!(StoppingRule::Geometric(1.5).validate().is_err());
        // γ == 1.0 is the valid boundary (always exactly one trial).
        assert!(StoppingRule::Geometric(1.0).validate().is_ok());
        let mut rng = LcgRng::new(1);
        assert_eq!(
            StoppingRule::Geometric(1.0).sample(&mut rng).expect("ok"),
            1
        );
    }

    #[test]
    fn test_err_lambda_nonpositive() {
        assert!(StoppingRule::Poisson(0.0).validate().is_err());
        assert!(StoppingRule::Poisson(-2.0).validate().is_err());
        assert!(PrivateTuningConfig::new(1.0, 0.0, StoppingRule::Poisson(0.0)).is_err());
    }

    #[test]
    fn test_err_fixed_zero() {
        assert!(StoppingRule::Fixed(0).validate().is_err());
        assert!(PrivateTuningConfig::new(1.0, 0.0, StoppingRule::Fixed(0)).is_err());
        assert!(tuning_epsilon(1.0, &StoppingRule::Fixed(0)).is_err());
    }
}
