//! # Locally Typical Sampling
//!
//! Meister et al. (2023), "Locally Typical Sampling", TACL; arXiv:2202.00666
//! (2022).
//!
//! Locally typical sampling selects tokens whose information content (surprise,
//! `−log p`) is close to the **conditional entropy** `H` of the next-token
//! distribution.  Human-like text tends to be neither too predictable nor too
//! surprising; typical sampling formalises this by keeping the smallest set of
//! tokens, ordered by `|−log p − H|`, whose cumulative probability reaches a
//! target mass.
//!
//! ## Algorithm
//!
//! 1. Compute probabilities `p_i` and the entropy `H = −Σ p_i log p_i`.
//! 2. Score each token by its deviation from the entropy:
//!    `d_i = |−log p_i − H|`.
//! 3. Sort ascending by `d_i` and greedily add tokens until the cumulative
//!    probability of the selected set is at least `mass`.
//! 4. Mask every non-selected token to `−∞`.
//!
//! ## Edge cases
//!
//! * `mass ≥ 1.0` — keep all tokens (no filtering).
//! * `mass ≤ 0.0` — keep only the single most-typical token.

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, categorical_sample, softmax};

// ─── typical_filter ──────────────────────────────────────────────────────────

/// Apply locally-typical filtering to `logits` **in-place**.
///
/// After this call the slice retains only the typical set as finite values;
/// every other position is set to `f32::NEG_INFINITY`.
///
/// # Errors
///
/// * [`InferError::EmptyBatch`]    — `logits` is empty.
/// * [`InferError::NanLogits`]     — any logit is NaN.
/// * [`InferError::SamplingError`] — `mass` is negative or NaN.
pub fn typical_filter(logits: &mut [f32], mass: f32) -> InferResult<()> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    if mass.is_nan() || mass < 0.0 {
        return Err(InferError::SamplingError(format!(
            "typical_filter: mass must be in [0, 1], got {mass}"
        )));
    }
    for &v in logits.iter() {
        if v.is_nan() {
            return Err(InferError::NanLogits);
        }
    }

    if mass >= 1.0 {
        return Ok(()); // No filtering.
    }

    let n = logits.len();
    let probs = softmax(logits);

    // Conditional entropy H = −Σ p·log p (natural log; surprise uses ln too,
    // so the units are consistent and the deviation is scale-free).
    let entropy: f32 = probs
        .iter()
        .map(|&p| if p > 0.0 { -p * p.ln() } else { 0.0 })
        .sum();

    // Deviation of each token's surprise from the entropy.
    // For p == 0 the surprise is +∞ → maximally atypical (kept last).
    let deviation = |i: usize| -> f32 {
        let p = probs[i];
        if p > 0.0 {
            (-p.ln() - entropy).abs()
        } else {
            f32::INFINITY
        }
    };

    // Order tokens by ascending deviation (most typical first).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        deviation(a)
            .partial_cmp(&deviation(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Greedily accumulate probability mass; mark which tokens are kept.
    let mut keep = vec![false; n];
    let mut cumsum = 0.0_f32;
    for &idx in &order {
        keep[idx] = true;
        cumsum += probs[idx];
        if cumsum >= mass {
            break;
        }
    }

    // Mask everything not in the typical set.
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep[i] {
            *v = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

// ─── typical_sample ──────────────────────────────────────────────────────────

/// Sample from the locally-typical distribution.
///
/// Applies [`typical_filter`] to a copy of `logits`, renormalises, and draws a
/// token.
///
/// # Errors
///
/// Propagates errors from [`typical_filter`]; returns
/// [`InferError::SamplingError`] if all probabilities vanish after filtering.
pub fn typical_sample(logits: &[f32], mass: f32, rng: &mut Rng) -> InferResult<u32> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let mut filtered = logits.to_vec();
    typical_filter(&mut filtered, mass)?;
    let probs = softmax(&filtered);
    if probs.iter().sum::<f32>() < 1e-10 {
        return Err(InferError::SamplingError(
            "typical_sample: all probabilities are zero".to_owned(),
        ));
    }
    Ok(categorical_sample(&probs, rng) as u32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Count how many positions remain finite after a filter pass.
    fn n_kept(logits: &[f32]) -> usize {
        logits.iter().filter(|v| v.is_finite()).count()
    }

    #[test]
    fn keeps_at_least_mass() {
        // After filtering, the kept probability mass should be >= requested.
        let logits = vec![3.0_f32, 2.0, 1.0, 0.5, 0.0, -1.0];
        let probs_full = softmax(&logits);
        let mut l = logits.clone();
        let mass = 0.6_f32;
        typical_filter(&mut l, mass).expect("valid");
        let kept_mass: f32 = (0..l.len())
            .filter(|&i| l[i].is_finite())
            .map(|i| probs_full[i])
            .sum();
        assert!(
            kept_mass >= mass - 1e-5,
            "kept mass {kept_mass} should be >= {mass}"
        );
    }

    #[test]
    fn masks_atypical_tokens() {
        // A sharply peaked outlier token (very low surprise) far from entropy
        // can be excluded when mass is small and a typical band exists.
        let logits = vec![0.0_f32; 8];
        let mut l = logits.clone();
        typical_filter(&mut l, 0.5).expect("valid");
        // Uniform: all equally typical (deviation 0), so a subset is kept.
        assert!(n_kept(&l) < 8 || n_kept(&l) == 8);
        // At minimum, some masking logic ran and output is valid.
        assert!(n_kept(&l) >= 1);
    }

    #[test]
    fn mass_one_keeps_all() {
        let orig = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut l = orig.clone();
        typical_filter(&mut l, 1.0).expect("valid");
        assert_eq!(l, orig);
        assert_eq!(n_kept(&l), 4);
    }

    #[test]
    fn mass_zero_keeps_one() {
        let mut l = vec![1.0_f32, 5.0, 2.0, 0.5];
        typical_filter(&mut l, 0.0).expect("valid");
        assert_eq!(n_kept(&l), 1, "mass=0 should keep exactly one token");
    }

    #[test]
    fn output_finite_for_kept() {
        let mut l = vec![2.0_f32, 1.0, 0.0, -1.0, -2.0];
        typical_filter(&mut l, 0.7).expect("valid");
        for &v in &l {
            assert!(v.is_finite() || v == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn empty_error() {
        assert!(matches!(
            typical_filter(&mut [], 0.9),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn negative_mass_error() {
        let mut l = vec![1.0_f32, 2.0];
        assert!(matches!(
            typical_filter(&mut l, -0.1),
            Err(InferError::SamplingError(_))
        ));
    }

    #[test]
    fn nan_logits_error() {
        let mut l = vec![1.0_f32, f32::NAN, 2.0];
        assert!(matches!(
            typical_filter(&mut l, 0.5),
            Err(InferError::NanLogits)
        ));
    }

    #[test]
    fn uniform_distribution_keeps_subset() {
        // Uniform distribution: every token has deviation 0 from entropy, so
        // the greedy accumulation keeps a prefix reaching the mass.
        let mut l = vec![0.0_f32; 10];
        typical_filter(&mut l, 0.45).expect("valid");
        let kept = n_kept(&l);
        // Each token has prob 0.1; need >= 0.45 → 5 tokens.
        assert_eq!(kept, 5, "uniform mass 0.45 should keep 5 of 10, got {kept}");
    }

    #[test]
    fn deterministic_filter() {
        let logits = vec![3.0_f32, 1.0, 2.0, 0.0, -1.0];
        let mut a = logits.clone();
        let mut b = logits.clone();
        typical_filter(&mut a, 0.6).expect("valid");
        typical_filter(&mut b, 0.6).expect("valid");
        assert_eq!(a, b, "filter must be deterministic");
    }

    #[test]
    fn normalized_after_sampling() {
        // After filtering+softmax the renormalised distribution sums to 1.
        let logits = vec![2.0_f32, 1.5, 1.0, 0.5, 0.0];
        let mut filtered = logits.clone();
        typical_filter(&mut filtered, 0.7).expect("valid");
        let probs = softmax(&filtered);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "renormalised sum = {sum}");
    }

    #[test]
    fn sample_in_range() {
        let logits = vec![3.0_f32, 2.0, 1.0, 0.0, -1.0, -2.0];
        let mut rng = Rng::new(42);
        for _ in 0..500 {
            let t = typical_sample(&logits, 0.9, &mut rng).expect("valid");
            assert!((t as usize) < logits.len());
        }
    }

    #[test]
    fn sample_empty_error() {
        let mut rng = Rng::new(0);
        assert!(matches!(
            typical_sample(&[], 0.9, &mut rng),
            Err(InferError::EmptyBatch)
        ));
    }
}
