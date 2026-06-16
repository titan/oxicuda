//! # Epsilon Sampling
//!
//! Hewitt et al. (2022), "Truncation Sampling as Language Model Desmoothing",
//! Findings of EMNLP 2022.
//!
//! Epsilon sampling is an **absolute-probability truncation** strategy: every
//! token whose probability falls below a fixed threshold `ε` is removed before
//! sampling.  Unlike nucleus (top-p) sampling — which keeps a cumulative mass —
//! or min-p sampling — which scales the threshold by the most-probable token —
//! epsilon sampling applies the *same* floor regardless of the distribution's
//! peakedness, directly implementing the desmoothing view that the tail of a
//! neural LM is an artefact of smoothing and should be cut at a constant
//! probability.
//!
//! ## Algorithm
//!
//! 1. Convert logits to probabilities `p_i`.
//! 2. Keep every token with `p_i >= ε`; always retain at least the single
//!    most-probable token so the distribution never becomes empty.
//! 3. Mask the rest to `−∞`, then renormalise and sample.
//!
//! ## Edge cases
//!
//! * `ε <= 0.0` — keep all tokens (no filtering).
//! * `ε >= 1.0` — keep only the single most-probable token.

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, categorical_sample, softmax};

// ─── epsilon_filter ──────────────────────────────────────────────────────────

/// Apply epsilon (absolute-probability) filtering to `logits` **in-place**.
///
/// Tokens with probability below `epsilon` are set to `f32::NEG_INFINITY`; the
/// single most-probable token is always preserved.
///
/// # Errors
///
/// * [`InferError::EmptyBatch`]    — `logits` is empty.
/// * [`InferError::NanLogits`]     — any logit is NaN.
/// * [`InferError::SamplingError`] — `epsilon` is negative or NaN.
pub fn epsilon_filter(logits: &mut [f32], epsilon: f32) -> InferResult<()> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    if epsilon.is_nan() || epsilon < 0.0 {
        return Err(InferError::SamplingError(format!(
            "epsilon_filter: epsilon must be >= 0, got {epsilon}"
        )));
    }
    for &v in logits.iter() {
        if v.is_nan() {
            return Err(InferError::NanLogits);
        }
    }

    if epsilon <= 0.0 {
        return Ok(()); // No filtering.
    }

    let probs = softmax(logits);

    // Index of the most-probable token, always retained.
    let mut argmax = 0_usize;
    let mut max_p = probs[0];
    for (i, &p) in probs.iter().enumerate().skip(1) {
        if p > max_p {
            max_p = p;
            argmax = i;
        }
    }

    for (i, v) in logits.iter_mut().enumerate() {
        if i != argmax && probs[i] < epsilon {
            *v = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

// ─── epsilon_sample ──────────────────────────────────────────────────────────

/// Sample from the epsilon-truncated distribution.
///
/// Applies [`epsilon_filter`] to a copy of `logits`, renormalises, and draws a
/// token.
///
/// # Errors
///
/// Propagates errors from [`epsilon_filter`]; returns
/// [`InferError::SamplingError`] if all probabilities vanish after filtering.
pub fn epsilon_sample(logits: &[f32], epsilon: f32, rng: &mut Rng) -> InferResult<u32> {
    if logits.is_empty() {
        return Err(InferError::EmptyBatch);
    }
    let mut filtered = logits.to_vec();
    epsilon_filter(&mut filtered, epsilon)?;
    let probs = softmax(&filtered);
    if probs.iter().sum::<f32>() < 1e-10 {
        return Err(InferError::SamplingError(
            "epsilon_sample: all probabilities are zero".to_owned(),
        ));
    }
    Ok(categorical_sample(&probs, rng) as u32)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn n_kept(logits: &[f32]) -> usize {
        logits.iter().filter(|v| v.is_finite()).count()
    }

    #[test]
    fn keeps_high_prob_tokens() {
        // Two dominant tokens, several negligible ones.
        let logits = vec![5.0_f32, 5.0, -5.0, -5.0, -5.0];
        let mut l = logits.clone();
        epsilon_filter(&mut l, 0.05).expect("valid");
        // The two peaks (prob ≈ 0.5 each) survive; the tiny tail is masked.
        assert!(l[0].is_finite() && l[1].is_finite());
        assert!(n_kept(&l) <= 2, "low-prob tail should be masked");
    }

    #[test]
    fn masks_below_threshold() {
        let logits = vec![10.0_f32, 0.0, 0.0, 0.0];
        let mut l = logits.clone();
        epsilon_filter(&mut l, 0.1).expect("valid");
        // Token 0 dominates (prob ≈ 1); the rest are below 0.1.
        assert_eq!(n_kept(&l), 1);
        assert!(l[0].is_finite());
    }

    #[test]
    fn epsilon_zero_keeps_all() {
        let orig = vec![1.0_f32, 2.0, 3.0];
        let mut l = orig.clone();
        epsilon_filter(&mut l, 0.0).expect("valid");
        assert_eq!(l, orig);
    }

    #[test]
    fn epsilon_large_keeps_only_top() {
        let mut l = vec![1.0_f32, 5.0, 2.0, 0.5];
        epsilon_filter(&mut l, 0.99).expect("valid");
        // No token reaches prob 0.99, so only the argmax is retained.
        assert_eq!(n_kept(&l), 1);
        assert!(l[1].is_finite(), "argmax token must survive");
    }

    #[test]
    fn always_keeps_argmax() {
        // Even with a huge epsilon, the top token is never removed.
        let mut l = vec![0.0_f32, 0.1, 0.0];
        epsilon_filter(&mut l, 1.0).expect("valid");
        assert!(l[1].is_finite(), "argmax must always survive");
        assert!(n_kept(&l) >= 1);
    }

    #[test]
    fn empty_error() {
        assert!(matches!(
            epsilon_filter(&mut [], 0.1),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn negative_epsilon_error() {
        let mut l = vec![1.0_f32, 2.0];
        assert!(matches!(
            epsilon_filter(&mut l, -0.1),
            Err(InferError::SamplingError(_))
        ));
    }

    #[test]
    fn nan_logits_error() {
        let mut l = vec![1.0_f32, f32::NAN];
        assert!(matches!(
            epsilon_filter(&mut l, 0.1),
            Err(InferError::NanLogits)
        ));
    }

    #[test]
    fn output_finite_or_neg_inf() {
        let mut l = vec![3.0_f32, 1.0, -1.0, -3.0];
        epsilon_filter(&mut l, 0.2).expect("valid");
        for &v in &l {
            assert!(v.is_finite() || v == f32::NEG_INFINITY);
        }
    }

    #[test]
    fn deterministic_filter() {
        let logits = vec![2.0_f32, 1.0, 0.5, 0.0, -1.0];
        let mut a = logits.clone();
        let mut b = logits.clone();
        epsilon_filter(&mut a, 0.15).expect("valid");
        epsilon_filter(&mut b, 0.15).expect("valid");
        assert_eq!(a, b);
    }

    #[test]
    fn normalized_after_filter() {
        let logits = vec![2.0_f32, 1.5, 1.0, 0.5, 0.0];
        let mut filtered = logits.clone();
        epsilon_filter(&mut filtered, 0.1).expect("valid");
        let probs = softmax(&filtered);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn sample_in_range() {
        let logits = vec![3.0_f32, 2.0, 1.0, 0.0, -1.0];
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let t = epsilon_sample(&logits, 0.05, &mut rng).expect("valid");
            assert!((t as usize) < logits.len());
        }
    }

    #[test]
    fn sample_dominant_token() {
        let logits = vec![0.0_f32, 100.0, 0.0];
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let t = epsilon_sample(&logits, 0.1, &mut rng).expect("valid");
            assert_eq!(t, 1, "dominant token should be sampled");
        }
    }
}
