//! Top-`k` sampling for autoregressive sequence decoding.
//!
//! Reference: Fan, A., Lewis, M., & Dauphin, Y. (2018).
//! *Hierarchical Neural Story Generation*. ACL 2018.
//! <https://aclanthology.org/P18-1082/>.
//!
//! # Algorithm
//!
//! Given a vector of logits `z ∈ ℝᵛ` and a temperature `T > 0`:
//!
//! ```text
//! z'_i  = z_i / T
//! S     = arg-top-k by z'
//! p_i   = softmax over S, zero elsewhere
//! tok   = Categorical(p)
//! ```
//!
//! The softmax is computed in a numerically-stable way by subtracting the
//! maximum logit of the top-`k` set before exponentiation.
//!
//! Setting `k = 1` collapses to greedy / argmax decoding (deterministic
//! regardless of the RNG state); setting `k ≥ vocab` recovers full
//! temperature-scaled softmax sampling.

use crate::error::{SeqError, SeqResult};
use crate::handle::LcgRng;

/// Configuration for [`top_k_sample`] and [`top_k_sample_batch`].
#[derive(Debug, Clone, Copy)]
pub struct TopKConfig {
    /// Number of highest-probability tokens to keep (`k ≥ 1`).
    pub k: usize,
    /// Softmax temperature (`> 0`).  Higher values flatten the
    /// distribution, lower values sharpen it.
    pub temperature: f64,
}

impl Default for TopKConfig {
    fn default() -> Self {
        Self {
            k: 50,
            temperature: 1.0,
        }
    }
}

impl TopKConfig {
    fn validate(&self) -> SeqResult<()> {
        if self.k == 0 {
            return Err(SeqError::InvalidConfiguration(
                "top-k: k must be >= 1".to_string(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(SeqError::InvalidParameter {
                name: "temperature".to_string(),
                value: self.temperature,
            });
        }
        Ok(())
    }
}

/// Sample a single token id from `logits` using top-`k` sampling.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `logits` is empty.
/// * [`SeqError::InvalidConfiguration`] / [`SeqError::InvalidParameter`]
///   if `cfg` is malformed.
pub fn top_k_sample(logits: &[f64], cfg: &TopKConfig, rng: &mut LcgRng) -> SeqResult<usize> {
    cfg.validate()?;
    if logits.is_empty() {
        return Err(SeqError::EmptyInput);
    }
    let v = logits.len();
    let k_eff = cfg.k.min(v);

    // Temperature-scale once.
    let scaled: Vec<f64> = logits.iter().map(|&z| z / cfg.temperature).collect();

    // Argmax fast-path when k == 1.
    if k_eff == 1 {
        return Ok(argmax(&scaled));
    }

    // Partial-sort indices by descending logit value, keeping the top k.
    let indices = top_k_indices(&scaled, k_eff);

    // Numerically-stable softmax over the surviving k.
    let max_z = indices
        .iter()
        .map(|&i| scaled[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut probs = vec![0.0_f64; k_eff];
    let mut sum = 0.0_f64;
    for (slot, &i) in indices.iter().enumerate() {
        let w = (scaled[i] - max_z).exp();
        probs[slot] = w;
        sum += w;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SeqError::NumericalInstability(
            "top-k: softmax denominator non-positive".to_string(),
        ));
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    let chosen_slot = rng.sample_categorical(&probs);
    Ok(indices[chosen_slot])
}

/// Batch variant of [`top_k_sample`] that processes `n` independent rows
/// of length `vocab` from a flat `logits` buffer.
///
/// # Errors
///
/// * [`SeqError::EmptyInput`] if `logits` is empty or `n == 0` or `vocab == 0`.
/// * [`SeqError::ShapeMismatch`] if `logits.len() != n * vocab`.
pub fn top_k_sample_batch(
    logits: &[f64],
    n: usize,
    vocab: usize,
    cfg: &TopKConfig,
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
        out.push(top_k_sample(row, cfg, rng)?);
    }
    Ok(out)
}

/// Return the index of the maximum element.  `xs` is assumed non-empty.
#[inline]
fn argmax(xs: &[f64]) -> usize {
    let mut best = 0usize;
    let mut best_v = xs[0];
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// Return the indices of the top-`k` elements of `xs` in descending order.
/// `k` is assumed to satisfy `1 <= k <= xs.len()`.
fn top_k_indices(xs: &[f64], k: usize) -> Vec<usize> {
    // O(v log k) maintain a small min-heap of (value, index) for the top-k.
    // For simplicity (and because k is generally small), we sort all and
    // take a prefix.  This stays O(v log v) but avoids hand-rolling a heap
    // and keeps the code readable.
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(k);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_softmax(logits: &[f64], t: f64) -> Vec<f64> {
        let scaled: Vec<f64> = logits.iter().map(|&z| z / t).collect();
        let m = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scaled.iter().map(|&z| (z - m).exp()).collect();
        let s: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / s).collect()
    }

    #[test]
    fn k_zero_rejected() {
        let cfg = TopKConfig {
            k: 0,
            temperature: 1.0,
        };
        let mut rng = LcgRng::new(0);
        let err = top_k_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::InvalidConfiguration(_)));
    }

    #[test]
    fn nonpositive_temperature_rejected() {
        let mut rng = LcgRng::new(0);
        for t in [0.0_f64, -1.0, f64::NAN] {
            let cfg = TopKConfig {
                k: 2,
                temperature: t,
            };
            let err = top_k_sample(&[0.1, 0.2], &cfg, &mut rng).unwrap_err();
            assert!(matches!(err, SeqError::InvalidParameter { .. }));
        }
    }

    #[test]
    fn empty_logits_rejected() {
        let cfg = TopKConfig::default();
        let mut rng = LcgRng::new(0);
        let err = top_k_sample(&[], &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn k_at_least_vocab_full_softmax() {
        // k >= vocab should behave like full softmax sampling.  Verify
        // that with high temperature both tokens have a chance.
        let logits = vec![0.0, 0.0, 0.0];
        let cfg = TopKConfig {
            k: 10,
            temperature: 1.0,
        };
        let mut rng = LcgRng::new(42);
        let mut counts = [0usize; 3];
        for _ in 0..3000 {
            let tok = top_k_sample(&logits, &cfg, &mut rng).expect("sample ok");
            counts[tok] += 1;
        }
        for c in counts {
            assert!(
                c > 700,
                "every token should be sampled: counts = {counts:?}"
            );
        }
    }

    #[test]
    fn k_one_is_argmax() {
        let logits = vec![-1.0, 4.5, 2.0, 4.5_f64.next_down()];
        let cfg = TopKConfig {
            k: 1,
            temperature: 0.7,
        };
        let mut rng_a = LcgRng::new(0);
        let mut rng_b = LcgRng::new(999_999);
        let tok_a = top_k_sample(&logits, &cfg, &mut rng_a).expect("sample ok");
        let tok_b = top_k_sample(&logits, &cfg, &mut rng_b).expect("sample ok");
        assert_eq!(tok_a, 1);
        assert_eq!(tok_b, 1, "k=1 must be deterministic regardless of rng");
    }

    #[test]
    fn deterministic_with_seed() {
        let logits = vec![0.5, 1.2, -0.3, 0.8, 2.1];
        let cfg = TopKConfig {
            k: 3,
            temperature: 1.0,
        };
        let mut rng_a = LcgRng::new(123);
        let mut rng_b = LcgRng::new(123);
        for _ in 0..200 {
            let a = top_k_sample(&logits, &cfg, &mut rng_a).expect("ok");
            let b = top_k_sample(&logits, &cfg, &mut rng_b).expect("ok");
            assert_eq!(a, b);
        }
    }

    #[test]
    fn distribution_matches_renormalised_softmax() {
        // Use logits such that top-3 of 5 are well-separated from the
        // bottom-2; assert chi-square goodness-of-fit at 5% significance
        // against the theoretical renormalised softmax over the kept set.
        let logits = vec![3.0_f64, 1.0, 0.0, -2.0, -5.0];
        let cfg = TopKConfig {
            k: 3,
            temperature: 1.0,
        };
        // Expected probabilities = softmax over indices {0, 1, 2}.
        let full = full_softmax(&logits[..3], 1.0);
        let n_samples = 6000usize;
        let mut rng = LcgRng::new(7);
        let mut counts = [0usize; 3];
        for _ in 0..n_samples {
            let t = top_k_sample(&logits, &cfg, &mut rng).expect("ok");
            assert!(t < 3, "top-k must never pick a truncated index");
            counts[t] += 1;
        }
        let mut chi2 = 0.0_f64;
        for i in 0..3 {
            let expected = full[i] * n_samples as f64;
            let diff = counts[i] as f64 - expected;
            chi2 += diff * diff / expected;
        }
        // df = 2, 99th percentile ≈ 9.21.
        assert!(chi2 < 9.21, "chi-square = {chi2}");
    }

    #[test]
    fn batch_correctness() {
        // Two rows, each strongly peaked on a different index.
        let logits = vec![10.0, -10.0, -10.0, -10.0, 10.0, -10.0];
        let cfg = TopKConfig {
            k: 2,
            temperature: 1.0,
        };
        let mut rng = LcgRng::new(0);
        let out = top_k_sample_batch(&logits, 2, 3, &cfg, &mut rng).expect("ok");
        assert_eq!(out, vec![0, 1]);
    }

    #[test]
    fn batch_empty_rejected() {
        let cfg = TopKConfig::default();
        let mut rng = LcgRng::new(0);
        assert!(matches!(
            top_k_sample_batch(&[], 0, 3, &cfg, &mut rng).unwrap_err(),
            SeqError::EmptyInput
        ));
        assert!(matches!(
            top_k_sample_batch(&[0.0, 0.0], 1, 0, &cfg, &mut rng).unwrap_err(),
            SeqError::EmptyInput
        ));
    }

    #[test]
    fn batch_shape_mismatch_rejected() {
        let logits = vec![0.0_f64; 5];
        let cfg = TopKConfig::default();
        let mut rng = LcgRng::new(0);
        let err = top_k_sample_batch(&logits, 2, 3, &cfg, &mut rng).unwrap_err();
        assert!(matches!(err, SeqError::ShapeMismatch { .. }));
    }

    #[test]
    fn high_temperature_flattens() {
        // With very high temperature and k=full, even strongly peaked
        // logits should produce a near-uniform sample distribution.
        let logits = vec![5.0, 0.0, 0.0, 0.0];
        let cfg = TopKConfig {
            k: 4,
            temperature: 50.0,
        };
        let mut rng = LcgRng::new(1);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[top_k_sample(&logits, &cfg, &mut rng).expect("ok")] += 1;
        }
        for c in counts {
            // Uniform would give 1000; allow generous slack.
            assert!(c > 700);
        }
    }

    #[test]
    fn low_temperature_sharpens() {
        // With low temperature, sampling should overwhelmingly pick
        // the largest logit even at k = full.
        let logits = vec![3.0, 1.0, 0.0, -1.0];
        let cfg = TopKConfig {
            k: 4,
            temperature: 0.05,
        };
        let mut rng = LcgRng::new(0);
        let mut argmax_count = 0usize;
        for _ in 0..1000 {
            if top_k_sample(&logits, &cfg, &mut rng).expect("ok") == 0 {
                argmax_count += 1;
            }
        }
        assert!(argmax_count > 980);
    }

    #[test]
    fn top_k_never_picks_truncated_token() {
        // Logits with a clear top-2 vs the rest; with k=2, the chosen
        // token must always be one of the two largest.
        let logits = vec![5.0, 4.5, -3.0, -4.0, -10.0];
        let cfg = TopKConfig {
            k: 2,
            temperature: 1.0,
        };
        let mut rng = LcgRng::new(42);
        for _ in 0..500 {
            let t = top_k_sample(&logits, &cfg, &mut rng).expect("ok");
            assert!(t == 0 || t == 1, "got truncated token {t}");
        }
    }

    #[test]
    fn single_vocab_returns_zero() {
        let logits = vec![2.71_f64];
        let cfg = TopKConfig {
            k: 5,
            temperature: 1.0,
        };
        let mut rng = LcgRng::new(0);
        for _ in 0..10 {
            assert_eq!(top_k_sample(&logits, &cfg, &mut rng).expect("ok"), 0);
        }
    }
}
