//! Top-k / top-p (nucleus) sampling helpers for RLHF rollout generation.
//!
//! References:
//! * Fan et al. 2018, "Hierarchical Neural Story Generation", arXiv:1805.04833
//!   (top-k truncation).
//! * Holtzman et al. 2020, "The Curious Case of Neural Text Degeneration",
//!   arXiv:1904.09751 (top-p / nucleus sampling).
//!
//! During on-policy RLHF the policy must *generate* candidate continuations
//! token by token. Greedy decoding collapses diversity (and starves PPO of
//! exploration), while raw multinomial sampling over the full vocabulary draws
//! from the long, unreliable tail. Truncated sampling fixes both: restrict the
//! candidate set to the `k` highest-probability tokens (top-k) or the smallest
//! set whose cumulative probability ≥ `p` (top-p), apply a temperature, and
//! renormalise before drawing.
//!
//! All routines here are deterministic given the crate's [`LcgRng`] and take
//! **logits** (un-normalised scores); the softmax / temperature / truncation /
//! renormalisation pipeline is performed internally with the standard
//! numerically-stable max-subtraction. Sampling uses inverse-CDF on the
//! renormalised distribution with the full-range unit uniform
//! `next_u32() as f64 / 2^32`.

use crate::error::{RlhfError, RlhfResult};
use crate::handle::LcgRng;

// ── Config ──────────────────────────────────────────────────────────────────

/// Truncation strategy for sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum TruncationMode {
    /// Keep the `k` highest-probability tokens (`k ≥ 1`). `k ≥ vocab` keeps all.
    TopK(usize),
    /// Keep the smallest set of tokens whose cumulative probability ≥ `p`
    /// (`0 < p ≤ 1`). Always keeps at least one token.
    TopP(f32),
    /// Apply top-k first, then top-p on the surviving tokens.
    TopKThenP { k: usize, p: f32 },
}

/// Configuration for truncated sampling.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Softmax temperature `T > 0`. Higher → flatter (more random); `T → 0`
    /// approaches greedy. Must be positive and finite.
    pub temperature: f32,
    /// Truncation strategy.
    pub mode: TruncationMode,
}

impl SamplingConfig {
    fn validate(&self, vocab: usize) -> RlhfResult<()> {
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(RlhfError::InvalidTemp {
                temp: self.temperature,
            });
        }
        match &self.mode {
            TruncationMode::TopK(k) => {
                if *k == 0 {
                    return Err(RlhfError::DimensionMismatch {
                        expected: 1,
                        got: 0,
                    });
                }
            }
            TruncationMode::TopP(p) => {
                if !p.is_finite() || *p <= 0.0 || *p > 1.0 {
                    return Err(RlhfError::InvalidLambda { lambda: *p });
                }
            }
            TruncationMode::TopKThenP { k, p } => {
                if *k == 0 {
                    return Err(RlhfError::DimensionMismatch {
                        expected: 1,
                        got: 0,
                    });
                }
                if !p.is_finite() || *p <= 0.0 || *p > 1.0 {
                    return Err(RlhfError::InvalidLambda { lambda: *p });
                }
            }
        }
        let _ = vocab;
        Ok(())
    }
}

// ── Softmax with temperature ────────────────────────────────────────────────

/// Numerically-stable softmax of `logits / temperature`.
fn softmax_temp(logits: &[f32], temperature: f32) -> RlhfResult<Vec<f32>> {
    if logits.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        let scaled = l / temperature;
        if scaled > max {
            max = scaled;
        }
    }
    let mut probs = Vec::with_capacity(logits.len());
    let mut sum = 0.0_f64;
    for &l in logits {
        let e = ((l / temperature - max) as f64).exp();
        probs.push(e);
        sum += e;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(RlhfError::Internal {
            msg: "softmax normaliser is non-positive or non-finite".to_string(),
        });
    }
    Ok(probs.into_iter().map(|e| (e / sum) as f32).collect())
}

// ── Sorted-index helper ─────────────────────────────────────────────────────

/// Indices `0..n` sorted by descending probability, ties broken by ascending
/// index (deterministic).
fn argsort_desc(probs: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

// ── Build truncated distribution ────────────────────────────────────────────

/// The renormalised, truncated probability distribution over a subset of the
/// vocabulary.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedDistribution {
    /// Surviving token ids, ordered by descending original probability.
    pub token_ids: Vec<usize>,
    /// Renormalised probabilities aligned with `token_ids` (sum to 1).
    pub probs: Vec<f32>,
}

/// Build the truncated, renormalised distribution from `logits` under `cfg`.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for empty logits, [`RlhfError::InvalidTemp`]
/// / [`RlhfError::InvalidLambda`] / [`RlhfError::DimensionMismatch`] for an
/// invalid config, and [`RlhfError::NanEncountered`] for NaN logits.
pub fn build_truncated_distribution(
    logits: &[f32],
    cfg: &SamplingConfig,
) -> RlhfResult<TruncatedDistribution> {
    cfg.validate(logits.len())?;
    let probs = softmax_temp(logits, cfg.temperature)?;
    let order = argsort_desc(&probs);

    // Apply top-k truncation: keep the first min(k, n) ranked tokens.
    let after_k: Vec<usize> = match &cfg.mode {
        TruncationMode::TopK(k) | TruncationMode::TopKThenP { k, .. } => {
            let keep = (*k).min(order.len());
            order[..keep].to_vec()
        }
        TruncationMode::TopP(_) => order.clone(),
    };

    // Apply top-p (nucleus) truncation over the surviving ranked tokens.
    let kept: Vec<usize> = match &cfg.mode {
        TruncationMode::TopK(_) => after_k,
        TruncationMode::TopP(p) | TruncationMode::TopKThenP { p, .. } => {
            let mut cum = 0.0_f32;
            let mut out = Vec::new();
            for &tid in &after_k {
                out.push(tid);
                cum += probs[tid];
                if cum >= *p {
                    break;
                }
            }
            // Always at least one token (guaranteed since after_k is non-empty).
            out
        }
    };

    if kept.is_empty() {
        return Err(RlhfError::Internal {
            msg: "truncation produced an empty candidate set".to_string(),
        });
    }

    // Renormalise the surviving probabilities to sum to 1.
    let mass: f32 = kept.iter().map(|&t| probs[t]).sum();
    if mass <= 0.0 {
        return Err(RlhfError::Internal {
            msg: "surviving probability mass is non-positive".to_string(),
        });
    }
    let renorm: Vec<f32> = kept.iter().map(|&t| probs[t] / mass).collect();
    Ok(TruncatedDistribution {
        token_ids: kept,
        probs: renorm,
    })
}

// ── Sampling ────────────────────────────────────────────────────────────────

/// Full-range unit uniform draw in `[0, 1)` from the crate RNG.
#[inline]
fn unit_uniform(rng: &mut LcgRng) -> f64 {
    rng.next_u32() as f64 / 4_294_967_296.0_f64
}

/// Draw one token id from `logits` using the configured truncated distribution.
///
/// Sampling is inverse-CDF on the renormalised distribution. Deterministic for a
/// given RNG state.
///
/// # Errors
///
/// Propagates errors from [`build_truncated_distribution`].
pub fn sample_token(logits: &[f32], cfg: &SamplingConfig, rng: &mut LcgRng) -> RlhfResult<usize> {
    let dist = build_truncated_distribution(logits, cfg)?;
    let u = unit_uniform(rng) as f32;
    let mut cum = 0.0_f32;
    for (&tid, &p) in dist.token_ids.iter().zip(dist.probs.iter()) {
        cum += p;
        if u < cum {
            return Ok(tid);
        }
    }
    // Floating-point fallback: return the last (lowest-prob) survivor.
    dist.token_ids.last().copied().ok_or(RlhfError::EmptyInput)
}

/// Greedy decode: the argmax token id (ties → lowest index). Independent of
/// temperature and truncation.
///
/// # Errors
///
/// Returns [`RlhfError::EmptyInput`] for empty logits and
/// [`RlhfError::NanEncountered`] for NaN logits.
pub fn greedy_token(logits: &[f32]) -> RlhfResult<usize> {
    if logits.is_empty() {
        return Err(RlhfError::EmptyInput);
    }
    let mut best = 0_usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() {
            return Err(RlhfError::NanEncountered);
        }
        if l > best_val {
            best_val = l;
            best = i;
        }
    }
    Ok(best)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn topk(k: usize) -> SamplingConfig {
        SamplingConfig {
            temperature: 1.0,
            mode: TruncationMode::TopK(k),
        }
    }

    // 1. softmax_temp sums to 1 and is monotone in logit.
    #[test]
    fn softmax_normalised_and_monotone() {
        let logits = [0.0_f32, 1.0, 2.0, 3.0];
        let p = softmax_temp(&logits, 1.0).expect("softmax");
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "softmax sum {s}");
        for w in p.windows(2) {
            assert!(w[1] > w[0], "softmax should increase with logit");
        }
    }

    // 2. top-k keeps exactly k tokens (k < vocab).
    #[test]
    fn topk_keeps_k_tokens() {
        let logits = [3.0_f32, 1.0, 2.0, 0.0, -1.0];
        let dist = build_truncated_distribution(&logits, &topk(2)).expect("dist");
        assert_eq!(dist.token_ids.len(), 2);
        // Highest logits are indices 0 (3.0) then 2 (2.0).
        assert_eq!(dist.token_ids, vec![0, 2]);
        let s: f32 = dist.probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "renorm sum {s}");
    }

    // 3. top-k with k >= vocab keeps all tokens.
    #[test]
    fn topk_large_keeps_all() {
        let logits = [3.0_f32, 1.0, 2.0];
        let dist = build_truncated_distribution(&logits, &topk(10)).expect("dist");
        assert_eq!(dist.token_ids.len(), 3);
    }

    // 4. top-p keeps the smallest nucleus exceeding p.
    #[test]
    fn topp_keeps_nucleus() {
        // Probabilities ~ [0.6439, 0.2369, 0.0871, 0.0321] for logits 3,2,1,0.
        let logits = [3.0_f32, 2.0, 1.0, 0.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            mode: TruncationMode::TopP(0.8),
        };
        let dist = build_truncated_distribution(&logits, &cfg).expect("dist");
        // 0.6439 < 0.8, +0.2369 = 0.8808 >= 0.8 → keep top 2.
        assert_eq!(dist.token_ids, vec![0, 1]);
    }

    // 5. top-p always keeps at least one token even with tiny p.
    #[test]
    fn topp_keeps_at_least_one() {
        let logits = [3.0_f32, 2.0, 1.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            mode: TruncationMode::TopP(0.01),
        };
        let dist = build_truncated_distribution(&logits, &cfg).expect("dist");
        assert_eq!(dist.token_ids.len(), 1);
        assert_eq!(dist.token_ids[0], 0);
        assert!((dist.probs[0] - 1.0).abs() < 1e-6);
    }

    // 6. TopKThenP applies k first then p.
    #[test]
    fn topk_then_p_composes() {
        let logits = [4.0_f32, 3.0, 2.0, 1.0, 0.0];
        let cfg = SamplingConfig {
            temperature: 1.0,
            mode: TruncationMode::TopKThenP { k: 3, p: 0.9 },
        };
        let dist = build_truncated_distribution(&logits, &cfg).expect("dist");
        // First keep top 3 (indices 0,1,2), then top-p within those.
        assert!(dist.token_ids.len() <= 3);
        assert!(dist.token_ids.iter().all(|&t| t < 3));
    }

    // 7. Sampling is deterministic for a fixed RNG seed.
    #[test]
    fn sampling_deterministic_per_seed() {
        let logits = [2.0_f32, 1.0, 0.5, 0.0];
        let cfg = topk(3);
        let mut r1 = LcgRng::new(123);
        let mut r2 = LcgRng::new(123);
        for _ in 0..50 {
            let a = sample_token(&logits, &cfg, &mut r1).expect("sample");
            let b = sample_token(&logits, &cfg, &mut r2).expect("sample");
            assert_eq!(a, b, "same seed must give same draws");
        }
    }

    // 8. Sampled tokens always lie within the truncated candidate set.
    #[test]
    fn samples_within_candidate_set() {
        let logits = [3.0_f32, 1.0, 2.0, 0.0, -5.0];
        let cfg = topk(2); // candidates: indices 0 and 2
        let mut rng = LcgRng::new(7);
        for _ in 0..200 {
            let t = sample_token(&logits, &cfg, &mut rng).expect("sample");
            assert!(t == 0 || t == 2, "sampled token {t} outside top-2 set");
        }
    }

    // 9. Greedy returns the argmax (ties → lowest index).
    #[test]
    fn greedy_returns_argmax() {
        let logits = [1.0_f32, 5.0, 5.0, 2.0];
        assert_eq!(greedy_token(&logits).expect("greedy"), 1);
    }

    // 10. Empirical: with top-1 the sampler always returns the argmax.
    #[test]
    fn top1_is_greedy() {
        let logits = [0.5_f32, 3.0, 1.0];
        let cfg = topk(1);
        let mut rng = LcgRng::new(99);
        for _ in 0..100 {
            assert_eq!(sample_token(&logits, &cfg, &mut rng).expect("sample"), 1);
        }
    }

    // 11. Higher temperature flattens the distribution (entropy increases).
    #[test]
    fn higher_temperature_flattens() {
        let logits = [3.0_f32, 1.0, 0.0];
        let cold = softmax_temp(&logits, 0.5).expect("cold");
        let hot = softmax_temp(&logits, 5.0).expect("hot");
        // Max probability should be larger when colder.
        let max_cold = cold.iter().cloned().fold(f32::MIN, f32::max);
        let max_hot = hot.iter().cloned().fold(f32::MIN, f32::max);
        assert!(max_cold > max_hot, "cold max {max_cold} vs hot {max_hot}");
    }

    // 12. Invalid temperature rejected.
    #[test]
    fn invalid_temperature_errors() {
        let logits = [1.0_f32, 2.0];
        let cfg = SamplingConfig {
            temperature: 0.0,
            mode: TruncationMode::TopK(2),
        };
        assert!(matches!(
            build_truncated_distribution(&logits, &cfg),
            Err(RlhfError::InvalidTemp { .. })
        ));
    }

    // 13. k = 0 rejected.
    #[test]
    fn topk_zero_errors() {
        let logits = [1.0_f32, 2.0];
        assert!(matches!(
            build_truncated_distribution(&logits, &topk(0)),
            Err(RlhfError::DimensionMismatch { .. })
        ));
    }

    // 14. p out of (0,1] rejected.
    #[test]
    fn topp_out_of_range_errors() {
        let logits = [1.0_f32, 2.0];
        let bad_hi = SamplingConfig {
            temperature: 1.0,
            mode: TruncationMode::TopP(1.5),
        };
        let bad_lo = SamplingConfig {
            temperature: 1.0,
            mode: TruncationMode::TopP(0.0),
        };
        assert!(matches!(
            build_truncated_distribution(&logits, &bad_hi),
            Err(RlhfError::InvalidLambda { .. })
        ));
        assert!(matches!(
            build_truncated_distribution(&logits, &bad_lo),
            Err(RlhfError::InvalidLambda { .. })
        ));
    }

    // 15. NaN logit rejected.
    #[test]
    fn nan_logit_errors() {
        let logits = [1.0_f32, f32::NAN, 2.0];
        assert!(matches!(
            build_truncated_distribution(&logits, &topk(2)),
            Err(RlhfError::NanEncountered)
        ));
        assert!(matches!(
            greedy_token(&logits),
            Err(RlhfError::NanEncountered)
        ));
    }

    // 16. Empty logits rejected.
    #[test]
    fn empty_logits_errors() {
        let logits: [f32; 0] = [];
        assert!(matches!(
            build_truncated_distribution(&logits, &topk(2)),
            Err(RlhfError::EmptyInput)
        ));
        assert!(matches!(greedy_token(&logits), Err(RlhfError::EmptyInput)));
    }
}
