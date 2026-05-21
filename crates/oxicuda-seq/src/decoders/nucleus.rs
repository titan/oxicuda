//! Nucleus (top-`p`) sampling for autoregressive sequence decoding.
//!
//! Reference: Holtzman, A., Buys, J., Du, L., Forbes, M. & Choi, Y. (2020).
//! *The Curious Case of Neural Text Degeneration*. ICLR 2020.
//! <https://arxiv.org/abs/1904.09751>.
//!
//! # Algorithm
//!
//! Given logits `z ∈ ℝᵛ`, temperature `T > 0`, nucleus mass `p ∈ (0, 1]`
//! and a lower bound `min_tokens ≥ 1`:
//!
//! ```text
//! q_i        = softmax(z_i / T)                   (numerically-stable)
//! sort q in descending order, breaking ties by index
//! find the smallest m such that  Σ_{j≤m} q_{(j)}  ≥  p
//! m'         = max(m, min_tokens)
//! keep the first m' indices, renormalise their probabilities, sample
//! ```
//!
//! Setting `p = 1.0` recovers full temperature-scaled softmax sampling
//! (statistically); setting `min_tokens = 1` and `p → 0⁺` collapses to
//! greedy decoding (the argmax must always survive).
//!
//! The softmax is computed using the standard max-subtraction trick so
//! that logits like `[1e6, 1, 1]` do not overflow.

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Configuration for [`nucleus_sample`] and [`nucleus_sample_batch`].
#[derive(Debug, Clone, Copy)]
pub struct NucleusConfig {
    /// Cumulative-probability threshold defining the "nucleus", in `(0, 1]`.
    pub p: f64,
    /// Softmax temperature (`> 0`).
    pub temperature: f64,
    /// Minimum number of tokens to keep (`≥ 1`).  This safeguards against
    /// pathologically tiny nuclei when `p` is very small.
    pub min_tokens: usize,
}

impl Default for NucleusConfig {
    fn default() -> Self {
        Self {
            p: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        }
    }
}

impl NucleusConfig {
    fn validate(&self) -> SeqResult<()> {
        if !self.p.is_finite() || self.p <= 0.0 || self.p > 1.0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "nucleus: p must be in (0, 1], got {}",
                self.p
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
                "nucleus: min_tokens must be >= 1".to_string(),
            ));
        }
        Ok(())
    }
}

/// Sample a single token id from `logits` using nucleus (top-`p`) sampling.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `logits` is empty.
/// * [`SeqError::InvalidConfiguration`] / [`SeqError::InvalidParameter`]
///   if `cfg` is malformed.
pub fn nucleus_sample(logits: &[f64], cfg: &NucleusConfig, rng: &mut LcgRng) -> SeqResult<usize> {
    cfg.validate()?;
    if logits.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let v = logits.len();

    // Numerically-stable softmax with max-subtraction *after* temperature
    // scaling.
    let scaled: Vec<f64> = logits.iter().map(|&z| z / cfg.temperature).collect();
    let max_z = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_z.is_finite() {
        return Err(SeqError::NumericalInstability(
            "nucleus: all logits non-finite".to_string(),
        ));
    }
    let mut probs = vec![0.0_f64; v];
    let mut sum = 0.0_f64;
    for (i, &z) in scaled.iter().enumerate() {
        let w = (z - max_z).exp();
        probs[i] = w;
        sum += w;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SeqError::NumericalInstability(
            "nucleus: softmax denominator non-positive".to_string(),
        ));
    }
    for q in probs.iter_mut() {
        *q /= sum;
    }

    // Sort indices by descending probability.
    let mut order: Vec<usize> = (0..v).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });

    // Find smallest m such that cumulative >= p.
    let mut cum = 0.0_f64;
    let mut m = 0usize;
    for (rank, &idx) in order.iter().enumerate() {
        cum += probs[idx];
        if cum >= cfg.p {
            m = rank + 1;
            break;
        }
    }
    if m == 0 {
        m = order.len();
    }
    let m_eff = m.max(cfg.min_tokens).min(order.len());

    // Renormalise over the kept set.
    let mut kept_probs = vec![0.0_f64; m_eff];
    let mut kept_sum = 0.0_f64;
    for slot in 0..m_eff {
        let q = probs[order[slot]];
        kept_probs[slot] = q;
        kept_sum += q;
    }
    if !kept_sum.is_finite() || kept_sum <= 0.0 {
        return Err(SeqError::NumericalInstability(
            "nucleus: kept mass zero".to_string(),
        ));
    }
    for q in kept_probs.iter_mut() {
        *q /= kept_sum;
    }

    let chosen_slot = rng.sample_categorical(&kept_probs);
    Ok(order[chosen_slot])
}

/// Batch variant of [`nucleus_sample`] for `n` independent rows of length
/// `vocab` in a flat `logits` buffer.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `logits` is empty, `n == 0`, or `vocab == 0`.
/// * [`SeqError::ShapeMismatch`] if `logits.len() != n * vocab`.
pub fn nucleus_sample_batch(
    logits: &[f64],
    n: usize,
    vocab: usize,
    cfg: &NucleusConfig,
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
        out.push(nucleus_sample(row, cfg, rng)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_p_rejected() {
        let mut rng = LcgRng::new(0);
        for p in [0.0_f64, -0.1, 1.1, f64::NAN] {
            let cfg = NucleusConfig {
                p,
                temperature: 1.0,
                min_tokens: 1,
            };
            let err = nucleus_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
            assert!(matches!(err, SeqError::InvalidConfiguration(_)));
        }
    }

    #[test]
    fn nonpositive_temperature_rejected() {
        let mut rng = LcgRng::new(0);
        for t in [0.0_f64, -0.5, f64::NAN] {
            let cfg = NucleusConfig {
                p: 0.9,
                temperature: t,
                min_tokens: 1,
            };
            let err = nucleus_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
            assert!(matches!(err, SeqError::InvalidParameter { .. }));
        }
    }

    #[test]
    fn zero_min_tokens_rejected() {
        let cfg = NucleusConfig {
            p: 0.9,
            temperature: 1.0,
            min_tokens: 0,
        };
        let mut rng = LcgRng::new(0);
        let err = nucleus_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::InvalidConfiguration(_)));
    }

    #[test]
    fn empty_logits_rejected() {
        let cfg = NucleusConfig::default();
        let mut rng = LcgRng::new(0);
        let err = nucleus_sample(&[], &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn p_one_is_full_softmax() {
        // With p = 1.0, every token is kept; statistically all should
        // appear when the distribution is uniform.
        let logits = vec![0.0; 4];
        let cfg = NucleusConfig {
            p: 1.0,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[nucleus_sample(&logits, &cfg, &mut rng).expect("ok")] += 1;
        }
        for c in counts {
            assert!(c > 800, "counts = {counts:?}");
        }
    }

    #[test]
    fn p_half_truncates_to_top_half() {
        // 4-token distribution where the top two probabilities together
        // already exceed 0.5; with p = 0.5 we should only ever see tokens
        // 0 and 1.
        let logits = vec![2.0, 1.0, 0.0, -1.0];
        let cfg = NucleusConfig {
            p: 0.5,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(123);
        for _ in 0..500 {
            let t = nucleus_sample(&logits, &cfg, &mut rng).expect("ok");
            assert!(t < 2, "token {t} should not appear with p=0.5");
        }
    }

    #[test]
    fn min_tokens_lower_bound() {
        // p ≈ 0 ⇒ nucleus contains only the argmax — but min_tokens = 3
        // forces the top-3 to remain.  Verify all of the top-3 indices
        // can be sampled and none of the bottom-2.
        let logits = vec![5.0, 1.5, 1.0, -4.0, -10.0];
        let cfg = NucleusConfig {
            p: 0.001,
            temperature: 1.0,
            min_tokens: 3,
        };
        let mut rng = LcgRng::new(0);
        let mut seen = [false; 5];
        for _ in 0..1000 {
            let t = nucleus_sample(&logits, &cfg, &mut rng).expect("ok");
            seen[t] = true;
        }
        assert!(seen[0] && seen[1] && seen[2]);
        assert!(!seen[3] && !seen[4]);
    }

    #[test]
    fn min_tokens_collapses_to_argmax() {
        // min_tokens = 1 with tiny p must always pick the argmax.
        let logits = vec![5.0, 1.5, 1.0, -4.0, -10.0];
        let cfg = NucleusConfig {
            p: 0.001,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(7);
        for _ in 0..200 {
            assert_eq!(nucleus_sample(&logits, &cfg, &mut rng).expect("ok"), 0);
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let logits = vec![0.5, -1.0, 1.2, 0.0, 2.3];
        let cfg = NucleusConfig {
            p: 0.8,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng_a = LcgRng::new(2026);
        let mut rng_b = LcgRng::new(2026);
        for _ in 0..200 {
            let a = nucleus_sample(&logits, &cfg, &mut rng_a).expect("ok");
            let b = nucleus_sample(&logits, &cfg, &mut rng_b).expect("ok");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn batch_correctness() {
        // Row 0 strongly favours index 0; row 1 strongly favours index 2.
        let logits = vec![10.0, -10.0, -10.0, -10.0, -10.0, 10.0];
        let cfg = NucleusConfig {
            p: 0.5,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let out = nucleus_sample_batch(&logits, 2, 3, &cfg, &mut rng).expect("ok");
        assert_eq!(out, vec![0, 2]);
    }

    #[test]
    fn batch_shape_mismatch() {
        let logits = vec![0.0_f64; 5];
        let cfg = NucleusConfig::default();
        let mut rng = LcgRng::new(0);
        let err = nucleus_sample_batch(&logits, 2, 3, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::ShapeMismatch { .. }));
    }

    #[test]
    fn numerically_stable_softmax() {
        // Without max-subtraction, exp(1e6) overflows.  After subtraction
        // we should still return a valid index and never panic.
        let logits = vec![1.0e6, 1.0, 1.0];
        let cfg = NucleusConfig {
            p: 0.9,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        let t = nucleus_sample(&logits, &cfg, &mut rng).expect("ok");
        assert_eq!(t, 0);
    }

    #[test]
    fn single_element_vocab() {
        let logits = vec![2.71];
        let cfg = NucleusConfig {
            p: 0.5,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        for _ in 0..10 {
            assert_eq!(nucleus_sample(&logits, &cfg, &mut rng).expect("ok"), 0);
        }
    }

    #[test]
    fn nucleus_never_picks_truncated_token() {
        // Bottom tokens have ~0 mass; nucleus at p=0.95 should still
        // exclude the very-bottom tokens.
        let logits = vec![3.0_f64, 2.5, 2.0, -10.0, -10.0, -10.0];
        let cfg = NucleusConfig {
            p: 0.95,
            temperature: 1.0,
            min_tokens: 1,
        };
        let mut rng = LcgRng::new(0);
        for _ in 0..500 {
            let t = nucleus_sample(&logits, &cfg, &mut rng).expect("ok");
            assert!(t < 3, "got truncated token {t}");
        }
    }
}
