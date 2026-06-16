//! Report-Noisy-Max with Gumbel noise and one-shot Top-k selection.
//!
//! References:
//! - Durfee & Rogers, "Practical Differentially Private Top-k Selection with
//!   Pay-what-you-get Composition", NeurIPS 2019.
//! - The Gumbel-max trick: adding `Gumbel(0, β)` noise to log-weights and taking
//!   the argmax samples *exactly* from the softmax over those weights.
//!
//! # Gumbel-max ≡ Exponential mechanism
//! The exponential mechanism selects candidate `i` with probability proportional
//! to `exp(ε·q_i / (2·Δq))`. Equivalently, drawing
//!
//! `g_i ~ Gumbel(0, β)` with `β = 2·Δq / ε`
//!
//! and returning `argmax_i (q_i + g_i)` yields *exactly* the exponential
//! mechanism's distribution. This is "Report Noisy Max with Gumbel noise".
//!
//! # One-shot Top-k
//! Returning the indices of the **k largest** Gumbel-perturbed scores gives a
//! differentially-private ordered top-k. Under the monotone setting of
//! Durfee & Rogers (2019), releasing the top-k via a single Gumbel perturbation
//! satisfies `(k·ε, 0)`-DP (peeling each rank costs ε), with the noise scale set
//! per the *per-query* budget `ε`.
//!
//! Unlike Laplace Report-Noisy-Max, Gumbel noise makes the argmax provably
//! equivalent to the exponential mechanism, giving the tightest pure-DP
//! selection without an explicit softmax normalisation.

use crate::error::{PrivacyError, PrivacyResult};
use crate::handle::LcgRng;

/// Configuration for the Gumbel-max selection / top-k mechanism.
#[derive(Debug, Clone)]
pub struct GumbelMaxConfig {
    /// Per-query privacy parameter ε > 0.
    pub epsilon: f64,
    /// Global sensitivity Δq > 0 of the quality function.
    pub sensitivity: f64,
}

impl GumbelMaxConfig {
    /// Construct and validate a `GumbelMaxConfig`.
    ///
    /// # Errors
    /// Returns `NonPositiveEpsilon` or `NonPositiveSensitivity` on invalid params.
    pub fn new(epsilon: f64, sensitivity: f64) -> PrivacyResult<Self> {
        if epsilon <= 0.0 {
            return Err(PrivacyError::NonPositiveEpsilon(epsilon));
        }
        if sensitivity <= 0.0 {
            return Err(PrivacyError::NonPositiveSensitivity(sensitivity));
        }
        Ok(Self {
            epsilon,
            sensitivity,
        })
    }

    /// Gumbel scale β = 2·Δq / ε so the argmax matches the exponential mechanism.
    #[must_use]
    pub fn gumbel_scale(&self) -> f64 {
        2.0 * self.sensitivity / self.epsilon
    }
}

/// Draw a single `Gumbel(0, scale)` sample via inverse-CDF transform.
///
/// `G = location − scale · ln(−ln(U))` with `U ~ Uniform(0,1)`; here
/// `location = 0`. `U` is clamped away from the open endpoints to keep the
/// double logarithm finite.
fn gumbel_sample(scale: f64, rng: &mut LcgRng) -> f64 {
    // u ∈ (0, 1): clamp away from 0 (ln 0) and from 1 (ln(−ln 1)=ln 0).
    let u = rng.next_f64().clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    -scale * (-(u.ln())).ln()
}

/// Report-Noisy-Max with Gumbel noise: add `Gumbel(0, 2Δq/ε)` to each score and
/// return the argmax index. Provides `(ε, 0)`-DP and is distributionally
/// identical to the exponential mechanism.
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `NonPositiveEpsilon` / `NonPositiveSensitivity` for an invalid config.
pub fn gumbel_max(scores: &[f64], cfg: &GumbelMaxConfig, rng: &mut LcgRng) -> PrivacyResult<usize> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if cfg.sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
    }

    let scale = cfg.gumbel_scale();
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &s) in scores.iter().enumerate() {
        let noisy = s + gumbel_sample(scale, rng);
        if noisy > best_val {
            best_val = noisy;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

/// One-shot differentially-private Top-k via a single Gumbel perturbation.
///
/// Adds `Gumbel(0, 2Δq/ε)` noise to every score once, then returns the indices
/// of the `k` largest perturbed scores in **descending** order. Satisfies
/// `(k·ε, 0)`-DP in the monotone top-k setting (Durfee & Rogers 2019).
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `InvalidParameter` if `k == 0`.
/// - `IndexOutOfRange(k, len)` if `k > scores.len()`.
/// - Config validation errors as above.
pub fn gumbel_top_k(
    scores: &[f64],
    k: usize,
    cfg: &GumbelMaxConfig,
    rng: &mut LcgRng,
) -> PrivacyResult<Vec<usize>> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if k == 0 {
        return Err(PrivacyError::InvalidParameter("k must be ≥ 1".into()));
    }
    if k > scores.len() {
        return Err(PrivacyError::IndexOutOfRange(k, scores.len()));
    }
    if cfg.epsilon <= 0.0 {
        return Err(PrivacyError::NonPositiveEpsilon(cfg.epsilon));
    }
    if cfg.sensitivity <= 0.0 {
        return Err(PrivacyError::NonPositiveSensitivity(cfg.sensitivity));
    }

    let scale = cfg.gumbel_scale();
    let mut noisy: Vec<(usize, f64)> = scores
        .iter()
        .map(|&s| s + gumbel_sample(scale, rng))
        .enumerate()
        .collect();

    // Sort by descending perturbed score; stable on the index for tie-break.
    noisy.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    Ok(noisy.into_iter().take(k).map(|(i, _)| i).collect())
}

/// Total `(ε_total, 0)`-DP cost of a one-shot Gumbel top-k release: `k · ε`.
#[must_use]
pub fn gumbel_top_k_epsilon(cfg: &GumbelMaxConfig, k: usize) -> f64 {
    cfg.epsilon * k as f64
}

/// Empirical argmax-selection probabilities over `trials` runs of [`gumbel_max`].
///
/// # Errors
/// - `EmptyInput` if `scores` is empty.
/// - `InvalidParameter` if `trials == 0`.
/// - Config validation errors as above.
pub fn gumbel_max_empirical_probs(
    scores: &[f64],
    cfg: &GumbelMaxConfig,
    trials: usize,
    rng: &mut LcgRng,
) -> PrivacyResult<Vec<f64>> {
    if scores.is_empty() {
        return Err(PrivacyError::EmptyInput);
    }
    if trials == 0 {
        return Err(PrivacyError::InvalidParameter("trials must be ≥ 1".into()));
    }
    let mut counts = vec![0u64; scores.len()];
    for _ in 0..trials {
        let idx = gumbel_max(scores, cfg, rng)?;
        counts[idx] += 1;
    }
    let inv = 1.0 / trials as f64;
    Ok(counts.iter().map(|&c| c as f64 * inv).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> GumbelMaxConfig {
        GumbelMaxConfig::new(1.0, 1.0).expect("valid config")
    }

    #[test]
    fn config_validation() {
        assert!(GumbelMaxConfig::new(0.0, 1.0).is_err());
        assert!(GumbelMaxConfig::new(1.0, 0.0).is_err());
        assert!(GumbelMaxConfig::new(-1.0, 1.0).is_err());
        assert!(GumbelMaxConfig::new(1.0, -1.0).is_err());
    }

    #[test]
    fn gumbel_scale_formula() {
        let c = GumbelMaxConfig::new(2.0, 3.0).expect("ok");
        // β = 2·Δ/ε = 2·3/2 = 3
        assert!((c.gumbel_scale() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn argmax_returns_valid_index() {
        let scores = vec![1.0, 5.0, 2.0, 0.5];
        let c = cfg();
        let mut rng = LcgRng::new(11);
        for _ in 0..200 {
            let idx = gumbel_max(&scores, &c, &mut rng).expect("ok");
            assert!(idx < scores.len());
        }
    }

    #[test]
    fn argmax_empty_errors() {
        let c = cfg();
        let mut rng = LcgRng::new(0);
        assert!(gumbel_max(&[], &c, &mut rng).is_err());
    }

    #[test]
    fn argmax_single_candidate() {
        let scores = vec![3.5];
        let c = cfg();
        let mut rng = LcgRng::new(2);
        for _ in 0..30 {
            assert_eq!(gumbel_max(&scores, &c, &mut rng).expect("ok"), 0);
        }
    }

    #[test]
    fn deterministic_same_seed() {
        let scores = vec![1.0, 2.0, 0.0, 3.0, 1.5];
        let c = cfg();
        let mut a = LcgRng::new(555);
        let mut b = LcgRng::new(555);
        for _ in 0..100 {
            assert_eq!(
                gumbel_max(&scores, &c, &mut a).expect("ok"),
                gumbel_max(&scores, &c, &mut b).expect("ok")
            );
        }
    }

    #[test]
    fn high_epsilon_picks_best() {
        let scores = vec![0.0, 0.0, 8.0, 1.0];
        let c = GumbelMaxConfig::new(5.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(404);
        let probs = gumbel_max_empirical_probs(&scores, &c, 4000, &mut rng).expect("ok");
        assert!(probs[2] > 0.9, "best prob = {}", probs[2]);
    }

    #[test]
    fn empirical_matches_exponential_softmax() {
        // Gumbel-max must reproduce softmax(ε·q/(2Δ)). Check the empirical
        // distribution against the closed-form exponential weights.
        let scores = vec![0.0, 1.0, 2.0];
        let c = GumbelMaxConfig::new(1.0, 1.0).expect("ok");
        let scale = c.epsilon / (2.0 * c.sensitivity);
        let weights: Vec<f64> = scores.iter().map(|&s| (s * scale).exp()).collect();
        let total: f64 = weights.iter().sum();
        let expected: Vec<f64> = weights.iter().map(|&w| w / total).collect();

        let mut rng = LcgRng::new(9001);
        let probs = gumbel_max_empirical_probs(&scores, &c, 20000, &mut rng).expect("ok");
        for (p, e) in probs.iter().zip(expected.iter()) {
            assert!((p - e).abs() < 0.03, "empirical {p} vs softmax {e}");
        }
    }

    #[test]
    fn top_k_returns_k_distinct_indices() {
        let scores = vec![1.0, 5.0, 2.0, 4.0, 3.0];
        let c = GumbelMaxConfig::new(8.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(17);
        let top = gumbel_top_k(&scores, 3, &c, &mut rng).expect("ok");
        assert_eq!(top.len(), 3);
        let mut sorted = top.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "indices must be distinct");
    }

    #[test]
    fn top_k_high_epsilon_recovers_true_ordering() {
        // With large ε the noisy top-k should match the true descending order.
        let scores = vec![10.0, 1.0, 7.0, 3.0, 9.0];
        let c = GumbelMaxConfig::new(50.0, 1.0).expect("ok");
        let mut rng = LcgRng::new(321);
        let top = gumbel_top_k(&scores, 3, &c, &mut rng).expect("ok");
        // True top-3 by score: indices 0 (10), 4 (9), 2 (7).
        assert_eq!(top, vec![0, 4, 2]);
    }

    #[test]
    fn top_k_errors() {
        let c = cfg();
        let mut rng = LcgRng::new(0);
        assert!(gumbel_top_k(&[], 1, &c, &mut rng).is_err());
        assert!(gumbel_top_k(&[1.0, 2.0], 0, &c, &mut rng).is_err());
        assert!(gumbel_top_k(&[1.0, 2.0], 3, &c, &mut rng).is_err());
    }

    #[test]
    fn top_k_epsilon_cost() {
        let c = GumbelMaxConfig::new(0.5, 1.0).expect("ok");
        assert!((gumbel_top_k_epsilon(&c, 4) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn empirical_zero_trials_errors() {
        let c = cfg();
        let mut rng = LcgRng::new(0);
        assert!(gumbel_max_empirical_probs(&[1.0, 2.0], &c, 0, &mut rng).is_err());
    }
}
