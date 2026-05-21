//! Typical (locally typical) decoding for autoregressive sequence
//! generation.
//!
//! Reference: Meister, C., Pimentel, T., Wiher, G. & Cotterell, R. (2022).
//! *Typical Decoding for Natural Language Generation*. TACL 2023 (arXiv 2202.00666).
//! <https://arxiv.org/abs/2202.00666>.
//!
//! # Algorithm
//!
//! Given logits `z ∈ ℝᵛ`, temperature `T > 0`, mass threshold `τ ∈ (0, 1]`
//! and a lower bound `min_tokens ≥ 1`:
//!
//! ```text
//! 1. p_i  = softmax(z_i / T)                       (numerically-stable)
//! 2. H    = -Σ_i p_i log p_i                       (conditional entropy)
//! 3. c_i  = |-log p_i - H|                         (information-content gap)
//! 4. sort indices by c_i ascending
//! 5. cumulate p along this order until ≥ τ
//! 6. enforce min_tokens; renormalise the kept set
//! 7. sample
//! ```
//!
//! Locally-typical decoding selects tokens whose surprisal is closest to
//! the expected information content of the next-token distribution.
//! Setting `τ = 1.0` retains every token (full softmax sampling);
//! peaked distributions select the argmax (its surprisal matches the
//! near-zero entropy); uniform distributions retain every token (every
//! surprisal equals the entropy `log V`).
//!
//! Probabilities below `f64::MIN_POSITIVE` are floored to a tiny epsilon
//! before taking the log, so the surprisal stays finite without distorting
//! the categorical sampler.

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Configuration for [`typical_sample`] and [`typical_sample_batch`].
#[derive(Debug, Clone, Copy)]
pub struct TypicalConfig {
    /// Typical-decoding cumulative mass threshold, in `(0, 1]`.
    pub tau: f64,
    /// Softmax temperature (`> 0`).
    pub temperature: f64,
    /// Minimum number of tokens to keep (`≥ 1`).
    pub min_tokens: usize,
}

impl Default for TypicalConfig {
    fn default() -> Self {
        Self {
            tau: 0.95,
            temperature: 1.0,
            min_tokens: 1,
        }
    }
}

impl TypicalConfig {
    fn validate(&self) -> SeqResult<()> {
        if !self.tau.is_finite() || self.tau <= 0.0 || self.tau > 1.0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "typical: tau must be in (0, 1], got {}",
                self.tau
            )));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "temperature".to_string(),
                value: self.temperature,
            });
        }
        if self.min_tokens == 0 {
            return Err(SeqError::InvalidConfiguration(
                "typical: min_tokens must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Numerically-stable softmax of `logits / temperature`.
fn softmax_scaled(logits: &[f64], temperature: f64) -> SeqResult<Vec<f64>> {
    let scaled: Vec<f64> = logits.iter().map(|&z| z / temperature).collect();
    let max_z = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_z.is_finite() {
        return Err(SeqError::NumericalInstability(
            "typical: all logits non-finite".to_string(),
        ));
    }
    let mut probs = vec![0.0_f64; scaled.len()];
    let mut sum = 0.0_f64;
    for (i, &z) in scaled.iter().enumerate() {
        let w = (z - max_z).exp();
        probs[i] = w;
        sum += w;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SeqError::NumericalInstability(
            "typical: softmax denominator non-positive".to_string(),
        ));
    }
    for q in probs.iter_mut() {
        *q /= sum;
    }
    Ok(probs)
}

/// Compute the Shannon entropy `H = -Σ p_i log p_i` of a probability
/// vector.  Public for testing and downstream use.
pub fn entropy(probs: &[f64]) -> f64 {
    let mut h = 0.0_f64;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.ln();
        }
    }
    h
}

const LOG_FLOOR: f64 = 1.0e-300;

/// Sample a single token id from `logits` using typical decoding.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `logits` is empty.
/// * [`SeqError::InvalidConfiguration`] / [`SeqError::InvalidParameter`]
///   if `cfg` is malformed.
pub fn typical_sample(logits: &[f64], cfg: &TypicalConfig, rng: &mut LcgRng) -> SeqResult<usize> {
    cfg.validate()?;
    if logits.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let probs = softmax_scaled(logits, cfg.temperature)?;
    let h = entropy(&probs);

    // |−log p_i − H|.  Floor p before log to avoid −∞.
    let mut gaps: Vec<(usize, f64)> = probs
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let surprisal = -(p.max(LOG_FLOOR)).ln();
            (i, (surprisal - h).abs())
        })
        .collect();

    // Sort indices by ascending information-content gap, breaking ties
    // by index for determinism.
    gaps.sort_by(|&(ia, ga), &(ib, gb)| {
        ga.partial_cmp(&gb)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ia.cmp(&ib))
    });

    // Cumulate p along the typical-set order until >= tau.
    let mut cum = 0.0_f64;
    let mut m = 0usize;
    for (rank, &(idx, _)) in gaps.iter().enumerate() {
        cum += probs[idx];
        if cum >= cfg.tau {
            m = rank + 1;
            break;
        }
    }
    if m == 0 {
        m = gaps.len();
    }
    let m_eff = m.max(cfg.min_tokens).min(gaps.len());

    // Renormalise the kept set.
    let mut kept_probs = vec![0.0_f64; m_eff];
    let mut kept_sum = 0.0_f64;
    for slot in 0..m_eff {
        let q = probs[gaps[slot].0];
        kept_probs[slot] = q;
        kept_sum += q;
    }
    if !kept_sum.is_finite() || kept_sum <= 0.0 {
        return Err(SeqError::NumericalInstability(
            "typical: kept mass zero".to_string(),
        ));
    }
    for q in kept_probs.iter_mut() {
        *q /= kept_sum;
    }

    let chosen_slot = rng.sample_categorical(&kept_probs);
    Ok(gaps[chosen_slot].0)
}

/// Batch variant of [`typical_sample`] for `n` independent rows of length
/// `vocab` in a flat `logits` buffer.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `logits` is empty, `n == 0`, or `vocab == 0`.
/// * [`SeqError::ShapeMismatch`] if `logits.len() != n * vocab`.
pub fn typical_sample_batch(
    logits: &[f64],
    n: usize,
    vocab: usize,
    cfg: &TypicalConfig,
    rng: &mut LcgRng,
) -> SeqResult<Vec<usize>> {
    cfg.validate()?;
    if logits.is_empty() || n == 0 || vocab == 0 {
        return Err(SeqError::EmptyInput);
    }
    if logits.len() != n * vocab {
        return Err(SeqError::ShapeMismatch {
            expected: n * vocab,
            got: logits.len(),
        });
    }
    let mut out = Vec::with_capacity(n);
    for b in 0..n {
        let row = &logits[b * vocab..(b + 1) * vocab];
        out.push(typical_sample(row, cfg, rng)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_tau_rejected() {
        let mut rng = LcgRng::new(0);
        for tau in [0.0_f64, -0.1, 1.1, f64::NAN] {
            let cfg = TypicalConfig {
                tau,
                temperature: 1.0,
                min_tokens: 1,
            };
            let err = typical_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
            assert!(matches!(err, SeqError::InvalidConfiguration(_)));
        }
    }

    #[test]
    fn nonpositive_temperature_rejected() {
        let mut rng = LcgRng::new(0);
        for t in [0.0_f64, -0.5, f64::NAN] {
            let cfg = TypicalConfig {
                tau: 0.9,
                temperature: t,
                min_tokens: 1,
            };
            let err = typical_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
            assert!(matches!(err, SeqError::InvalidParameter { .. }));
        }
    }

    #[test]
    fn zero_min_tokens_rejected() {
        let cfg = TypicalConfig {
            tau: 0.9,
            temperature: 1.0,
            min_tokens: 0,
        };
        let mut rng = LcgRng::new(0);
        let err = typical_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::InvalidConfiguration(_)));
    }

    #[test]
    fn empty_logits_rejected() {
        let cfg = TypicalConfig::default();
        let mut rng = LcgRng::new(0);
        let err = typical_sample(&[], &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn tau_one_keeps_everything() {
        let logits = vec![0.0_f64; 5];
        let cfg = TypicalConfig {
            tau: 1.0,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let mut counts = [0usize; 5];
        for _ in 0..5000 {
            counts[typical_sample(&logits, &cfg, &mut rng).expect("ok")] += 1;
        }
        for c in counts {
            assert!(c > 700, "counts = {counts:?}");
        }
    }

    #[test]
    fn uniform_logits_all_typical() {
        // For a uniform distribution over v tokens, H = log v and every
        // surprisal also equals log v, so all gaps are zero.  Sampling at
        // any tau > 0 must keep them all and produce a uniform output.
        let logits = vec![0.0_f64; 4];
        let probs = softmax_scaled(&logits, 1.0).expect("ok");
        let h = entropy(&probs);
        assert!((h - (4.0_f64).ln()).abs() < 1e-12);
        for &p in &probs {
            let gap = (-p.ln() - h).abs();
            assert!(gap < 1e-12, "gap = {gap}");
        }
    }

    #[test]
    fn peaked_logits_picks_peak() {
        // For a strongly peaked distribution the entropy is ~0; the argmax
        // has surprisal ~0, so its gap is the smallest and it is "most
        // typical".  Verify the argmax wins.
        let logits = vec![20.0, 0.0, 0.0, 0.0, 0.0];
        let cfg = TypicalConfig {
            tau: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        for _ in 0..200 {
            assert_eq!(typical_sample(&logits, &cfg, &mut rng).expect("ok"), 0);
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let logits = vec![0.5, -1.0, 1.2, 0.0, 2.3];
        let cfg = TypicalConfig {
            tau: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng_a = LcgRng::new(0xC0DE);
        let mut rng_b = LcgRng::new(0xC0DE);
        for _ in 0..200 {
            let a = typical_sample(&logits, &cfg, &mut rng_a).expect("ok");
            let b = typical_sample(&logits, &cfg, &mut rng_b).expect("ok");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn min_tokens_lower_bound() {
        // With tau very small, the kept set would otherwise be just the
        // single most-typical token; min_tokens = 3 must force 3 to be
        // kept.  Use a moderately spread distribution so that the 3
        // most-typical tokens are well-defined and the remaining ones
        // are excluded.
        let logits = vec![3.0_f64, 0.0, -1.0, -20.0, -25.0];
        let cfg = TypicalConfig {
            tau: 1.0e-6,
            temperature: 1.0,
            min_tokens: 3,
        };
        let mut rng = LcgRng::new(0);
        let mut seen_4 = false;
        let mut seen_lower = false;
        for _ in 0..1000 {
            let t = typical_sample(&logits, &cfg, &mut rng).expect("ok");
            if t == 3 || t == 4 {
                seen_lower = true;
            }
            if t < 3 {
                seen_4 = true;
            }
        }
        assert!(seen_4);
        assert!(!seen_lower, "tail tokens leaked into the typical set");
    }

    #[test]
    fn batch_correctness() {
        // Row 0: strongly peaked on idx 0; row 1: strongly peaked on idx 2.
        let logits = vec![20.0, -5.0, -5.0, -5.0, -5.0, 20.0];
        let cfg = TypicalConfig {
            tau: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let out = typical_sample_batch(&logits, 2, 3, &cfg, &mut rng).expect("ok");
        assert_eq!(out, vec![0, 2]);
    }

    #[test]
    fn batch_shape_mismatch() {
        let logits = vec![0.0_f64; 5];
        let cfg = TypicalConfig::default();
        let mut rng = LcgRng::new(0);
        let err = typical_sample_batch(&logits, 2, 3, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::ShapeMismatch { .. }));
    }

    #[test]
    fn numerically_stable_softmax() {
        // Logits that would overflow without max-subtraction.
        let logits = vec![1.0e6_f64, 1.0, 1.0];
        let cfg = TypicalConfig {
            tau: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let t = typical_sample(&logits, &cfg, &mut rng).expect("ok");
        assert_eq!(t, 0);
    }

    #[test]
    fn symmetric_distribution_sanity() {
        // Symmetric two-mode distribution: tokens 0 and 4 are equally
        // probable, tokens 1 and 3 are equally probable but lower, token
        // 2 is the lowest.  Their gaps come in matching pairs, so over
        // many samples token 0 and token 4 must be drawn at comparable
        // rates and the bulk of mass should sit on the two modes.
        let logits = vec![2.0_f64, 0.5, -1.0, 0.5, 2.0];
        let cfg = TypicalConfig {
            tau: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let mut counts = [0usize; 5];
        for _ in 0..6000 {
            counts[typical_sample(&logits, &cfg, &mut rng).expect("ok")] += 1;
        }
        let lo = counts[0].min(counts[4]) as f64;
        let hi = counts[0].max(counts[4]) as f64;
        assert!(hi / lo < 1.5, "asymmetric: counts = {counts:?}");
    }

    #[test]
    fn entropy_known_values() {
        // Uniform over 5: H = log 5 ≈ 1.6094.
        let p_uniform = vec![0.2_f64; 5];
        let h_u = entropy(&p_uniform);
        assert!((h_u - (5.0_f64).ln()).abs() < 1e-12);

        // Peaked: p = [1-4ε, ε, ε, ε, ε] with ε = 1e-12 — H ≈ 0.
        let eps = 1.0e-12;
        let p_peak = vec![1.0 - 4.0 * eps, eps, eps, eps, eps];
        let h_p = entropy(&p_peak);
        assert!(h_p.abs() < 1.0e-9, "peaked entropy = {h_p}");
    }

    #[test]
    fn single_element_vocab() {
        let logits = vec![0.5];
        let cfg = TypicalConfig::default();
        let mut rng = LcgRng::new(0);
        for _ in 0..10 {
            assert_eq!(typical_sample(&logits, &cfg, &mut rng).expect("ok"), 0);
        }
    }
}
