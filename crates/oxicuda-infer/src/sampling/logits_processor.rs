//! # Logits Processor
//!
//! Applies a chain of logit transformations before sampling:
//!
//! 1. **Repetition penalty** — down-weight tokens that have already appeared.
//! 2. **Presence penalty** — subtract a flat penalty for each unique repeated token.
//! 3. **Temperature scaling** — divide all logits by a temperature scalar.
//! 4. **Top-K filtering** — keep only the K highest logits (optional).
//! 5. **Top-P (nucleus) filtering** — keep the smallest set whose cumulative
//!    softmax probability ≥ p (optional).
//!
//! The processor is purely functional on the logit slice; all state is in
//! the config.
//!
//! ## References
//!
//! * Keskar et al. (2019) — "CTRL: A Conditional Transformer Language Model
//!   for Controllable Generation" (repetition penalty).
//! * Holtzman et al. (2020) — "The Curious Case of Neural Text Degeneration"
//!   (nucleus / top-p sampling).

use crate::error::{InferError, InferResult};
use crate::sampling::{Rng, categorical_sample, softmax};

// ─── LogitsProcessorConfig ───────────────────────────────────────────────────

/// Configuration for the logits processing pipeline.
#[derive(Debug, Clone)]
pub struct LogitsProcessorConfig {
    /// Temperature: logits are divided by this value before sampling.
    /// Must be strictly positive.
    pub temperature: f32,
    /// Top-K: if `Some(k)` and `k > 0`, only the top-K logits are kept.
    /// `None` or `Some(0)` disables top-K filtering.
    pub top_k: Option<usize>,
    /// Top-P (nucleus): if `Some(p)`, keep the smallest set of tokens
    /// whose cumulative probability ≥ p.  Must be in `(0, 1]`.
    pub top_p: Option<f32>,
    /// Repetition penalty: values > 1.0 reduce likelihood of repeated tokens.
    /// Must be ≥ 1.0.
    pub repetition_penalty: f32,
    /// Presence penalty: flat penalty subtracted for each unique repeated token.
    pub presence_penalty: f32,
}

impl Default for LogitsProcessorConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
        }
    }
}

// ─── LogitsProcessor ─────────────────────────────────────────────────────────

/// Stateless logit transformation pipeline.
pub struct LogitsProcessor {
    config: LogitsProcessorConfig,
}

impl LogitsProcessor {
    /// Build a new processor, validating all configuration fields.
    ///
    /// # Errors
    ///
    /// * [`InferError::InvalidConfig`] — temperature ≤ 0.0, repetition_penalty
    ///   < 1.0, or top_p not in `(0, 1]`.
    pub fn new(config: LogitsProcessorConfig) -> InferResult<Self> {
        if config.temperature <= 0.0 {
            return Err(InferError::InvalidConfig(
                "temperature must be strictly positive",
            ));
        }
        if config.repetition_penalty < 1.0 {
            return Err(InferError::InvalidConfig(
                "repetition_penalty must be >= 1.0",
            ));
        }
        if let Some(p) = config.top_p {
            if p <= 0.0 || p > 1.0 {
                return Err(InferError::InvalidConfig(
                    "top_p must be in the range (0, 1]",
                ));
            }
        }
        Ok(Self { config })
    }

    // ── process ──────────────────────────────────────────────────────────────

    /// Apply the full logit-processing pipeline in-place.
    ///
    /// The pipeline is:
    /// 1. Repetition penalty (divide positive / multiply negative logits).
    /// 2. Presence penalty (subtract flat constant for unique repeated tokens).
    /// 3. Temperature scaling.
    /// 4. Top-K filtering.
    /// 5. Top-P (nucleus) filtering.
    ///
    /// # Errors
    ///
    /// * [`InferError::NanLogits`] — a NaN is produced or detected.
    pub fn process(&self, logits: &mut [f32], input_ids: &[usize]) -> InferResult<()> {
        // ── 1. Repetition penalty ────────────────────────────────────────────
        let rep = self.config.repetition_penalty;
        if rep != 1.0 {
            for &t in input_ids {
                if t < logits.len() {
                    let v = logits[t];
                    logits[t] = if v > 0.0 { v / rep } else { v * rep };
                }
            }
        }

        // ── 2. Presence penalty ──────────────────────────────────────────────
        let pp = self.config.presence_penalty;
        if pp != 0.0 {
            // Collect unique token ids.
            let mut seen: Vec<usize> = input_ids.to_vec();
            seen.sort_unstable();
            seen.dedup();
            for t in seen {
                if t < logits.len() {
                    logits[t] -= pp;
                }
            }
        }

        // ── 3. Temperature ───────────────────────────────────────────────────
        let temp = self.config.temperature;
        if (temp - 1.0).abs() > f32::EPSILON {
            for v in logits.iter_mut() {
                *v /= temp;
            }
        }

        // ── 4. Top-K ─────────────────────────────────────────────────────────
        if let Some(k) = self.config.top_k {
            if k > 0 && k < logits.len() {
                apply_top_k(logits, k);
            }
        }

        // ── 5. Top-P (nucleus) ───────────────────────────────────────────────
        if let Some(p) = self.config.top_p {
            apply_top_p(logits, p);
        }

        // ── NaN guard ────────────────────────────────────────────────────────
        if logits.iter().any(|v| v.is_nan()) {
            return Err(InferError::NanLogits);
        }

        Ok(())
    }

    // ── sample ───────────────────────────────────────────────────────────────

    /// Compute softmax over `logits` and draw a categorical sample.
    ///
    /// # Errors
    ///
    /// * [`InferError::SamplingError`] — logits slice is empty or all
    ///   probabilities are zero.
    pub fn sample(&self, logits: &[f32], rng: &mut Rng) -> InferResult<usize> {
        if logits.is_empty() {
            return Err(InferError::SamplingError(
                "logits slice is empty".to_owned(),
            ));
        }
        let probs = softmax(logits);
        let total: f32 = probs.iter().sum();
        if total < 1e-10 {
            return Err(InferError::SamplingError(
                "all probabilities are zero after softmax".to_owned(),
            ));
        }
        Ok(categorical_sample(&probs, rng))
    }
}

// ─── apply_top_k (internal) ──────────────────────────────────────────────────

/// In-place top-K filter: zero-mask all but the K largest logits.
///
/// Logits below the K-th largest are set to `f32::NEG_INFINITY`.
fn apply_top_k(logits: &mut [f32], k: usize) {
    // Build a sorted copy to find the kth-largest threshold.
    let mut sorted: Vec<f32> = logits.iter().cloned().filter(|v| !v.is_nan()).collect();
    // Sort descending (guaranteed NaN-free here).
    sorted.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = if k <= sorted.len() {
        sorted[k - 1]
    } else {
        return; // k >= vocab: nothing to mask.
    };

    // Keep at most k values.
    let mut kept = 0_usize;
    for v in logits.iter_mut() {
        if !v.is_nan() && *v >= threshold && kept < k {
            kept += 1;
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ─── apply_top_p (internal) ──────────────────────────────────────────────────

/// In-place nucleus (top-P) filter.
///
/// Tokens beyond the smallest set whose cumulative softmax probability ≥ p
/// are set to `f32::NEG_INFINITY`.
fn apply_top_p(logits: &mut [f32], p: f32) {
    let n = logits.len();
    if n == 0 {
        return;
    }

    // Softmax over only the finite logits (treat -inf as probability 0).
    let probs = softmax(logits);

    // Sort indices by probability descending.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find the cutoff: the smallest prefix with cumprob >= p.
    let mut cumprob = 0.0_f32;
    let mut cutoff_idx = 0_usize; // last index to KEEP (inclusive).
    for (rank, &idx) in indices.iter().enumerate() {
        cumprob += probs[idx];
        cutoff_idx = rank;
        if cumprob >= p {
            break;
        }
    }

    // Mask all tokens beyond the cutoff.
    let keep_set: std::collections::HashSet<usize> =
        indices[..=cutoff_idx].iter().copied().collect();
    for (i, v) in logits.iter_mut().enumerate() {
        if !keep_set.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_proc() -> LogitsProcessor {
        LogitsProcessor::new(LogitsProcessorConfig::default()).expect("default config is valid")
    }

    fn proc_with(config: LogitsProcessorConfig) -> LogitsProcessor {
        LogitsProcessor::new(config).expect("test config is valid")
    }

    // ── 1. temperature_1_no_change ────────────────────────────────────────────
    #[test]
    fn temperature_1_no_change() {
        let proc = default_proc();
        let original = vec![1.0_f32, -2.0, 3.5, 0.0];
        let mut logits = original.clone();
        proc.process(&mut logits, &[])
            .expect("process with no input_ids");
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "expected unchanged logits, got {logits:?}"
            );
        }
    }

    // ── 2. temperature_scales ─────────────────────────────────────────────────
    #[test]
    fn temperature_scales() {
        let proc = proc_with(LogitsProcessorConfig {
            temperature: 2.0,
            ..Default::default()
        });
        let original = vec![2.0_f32, 4.0, -6.0];
        let mut logits = original.clone();
        proc.process(&mut logits, &[])
            .expect("process with temperature=2");
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!(
                (a - b / 2.0).abs() < 1e-6,
                "expected halved logits, orig={b} got={a}"
            );
        }
    }

    // ── 3. top_k_zeros_rest ───────────────────────────────────────────────────
    #[test]
    fn top_k_zeros_rest() {
        let proc = proc_with(LogitsProcessorConfig {
            top_k: Some(2),
            ..Default::default()
        });
        let mut logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        proc.process(&mut logits, &[])
            .expect("process with top_k=2");
        let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(
            finite_count, 2,
            "top_k=2 should keep exactly 2 finite logits"
        );
    }

    // ── 4. top_k_1_deterministic ─────────────────────────────────────────────
    #[test]
    fn top_k_1_deterministic() {
        let proc = proc_with(LogitsProcessorConfig {
            top_k: Some(1),
            ..Default::default()
        });
        let mut logits = vec![0.0_f32, -1.0, 5.0, 2.0];
        proc.process(&mut logits, &[])
            .expect("process with top_k=1");
        // Only index 2 (highest) should be finite.
        assert!(logits[2].is_finite(), "argmax should remain finite");
        assert!(!logits[0].is_finite(), "non-max logit[0] should be -inf");
        assert!(!logits[1].is_finite(), "non-max logit[1] should be -inf");
        assert!(!logits[3].is_finite(), "non-max logit[3] should be -inf");
    }

    // ── 5. top_p_covers_p_mass ────────────────────────────────────────────────
    #[test]
    fn top_p_covers_p_mass() {
        let proc = proc_with(LogitsProcessorConfig {
            top_p: Some(1.0),
            ..Default::default()
        });
        let mut logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        proc.process(&mut logits, &[])
            .expect("process with top_p=1.0");
        // All tokens should remain (top_p=1.0 = no filtering).
        let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(finite_count, 4, "top_p=1.0 should keep all tokens finite");
    }

    // ── 6. rep_penalty_reduces_repeated ──────────────────────────────────────
    #[test]
    fn rep_penalty_reduces_repeated() {
        let proc = proc_with(LogitsProcessorConfig {
            repetition_penalty: 2.0,
            ..Default::default()
        });
        let mut logits = vec![0.0_f32, 5.0, -3.0, 1.0];
        let logits_before_token1 = logits[1];
        let logits_before_token2 = logits[2];
        proc.process(&mut logits, &[1, 2])
            .expect("process with rep penalty");
        // token 1: positive logit → divided by 2 → smaller magnitude
        assert!(
            logits[1] < logits_before_token1,
            "repeated positive logit should decrease"
        );
        // token 2: negative logit → multiplied by 2 → more negative
        assert!(
            logits[2] < logits_before_token2,
            "repeated negative logit should become more negative"
        );
    }

    // ── 7. sample_in_range ────────────────────────────────────────────────────
    #[test]
    fn sample_in_range() {
        let proc = default_proc();
        let vocab_size = 16_usize;
        let logits: Vec<f32> = (0..vocab_size).map(|i| i as f32).collect();
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let idx = proc
                .sample(&logits, &mut rng)
                .expect("valid logits for sampling");
            assert!(
                idx < vocab_size,
                "sampled index {idx} >= vocab_size {vocab_size}"
            );
        }
    }

    // ── 8. process_output_finite ──────────────────────────────────────────────
    #[test]
    fn process_output_finite() {
        let proc = proc_with(LogitsProcessorConfig {
            temperature: 1.5,
            top_k: Some(3),
            repetition_penalty: 1.2,
            presence_penalty: 0.1,
            ..Default::default()
        });
        let mut logits = vec![1.0_f32, 3.0, 2.5, -1.0, 0.5];
        let input_ids = vec![0_usize, 1];
        proc.process(&mut logits, &input_ids)
            .expect("process should succeed with valid config");
        // No NaN in output.
        for &v in &logits {
            assert!(!v.is_nan(), "output contains NaN: {logits:?}");
        }
    }

    // ── 9. top_k_0_ok ────────────────────────────────────────────────────────
    #[test]
    fn top_k_0_ok() {
        // top_k=Some(0) means disabled — all logits remain finite.
        let proc = proc_with(LogitsProcessorConfig {
            top_k: Some(0),
            ..Default::default()
        });
        let original = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut logits = original.clone();
        proc.process(&mut logits, &[])
            .expect("top_k=0 should be a no-op");
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "top_k=0 should not change logits, got {logits:?}"
            );
        }
    }

    // ── 10. top_p_1_no_change ────────────────────────────────────────────────
    #[test]
    fn top_p_1_no_change() {
        let proc = proc_with(LogitsProcessorConfig {
            top_p: Some(1.0),
            ..Default::default()
        });
        let original = vec![0.5_f32, 1.0, 2.0, -0.5];
        let mut logits = original.clone();
        proc.process(&mut logits, &[])
            .expect("top_p=1.0 should keep all tokens");
        // Every logit should remain finite (no masking with top_p=1.0).
        let finite_count = logits.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(
            finite_count,
            original.len(),
            "top_p=1.0 should keep all {len} tokens finite",
            len = original.len()
        );
    }

    // ── Validation tests ──────────────────────────────────────────────────────

    #[test]
    fn invalid_temperature_zero() {
        let result = LogitsProcessor::new(LogitsProcessorConfig {
            temperature: 0.0,
            ..Default::default()
        });
        assert!(
            matches!(result, Err(InferError::InvalidConfig(_))),
            "temperature=0 should be InvalidConfig"
        );
    }

    #[test]
    fn invalid_rep_penalty_less_than_one() {
        let result = LogitsProcessor::new(LogitsProcessorConfig {
            repetition_penalty: 0.5,
            ..Default::default()
        });
        assert!(
            matches!(result, Err(InferError::InvalidConfig(_))),
            "rep_penalty < 1.0 should be InvalidConfig"
        );
    }

    #[test]
    fn invalid_top_p_zero() {
        let result = LogitsProcessor::new(LogitsProcessorConfig {
            top_p: Some(0.0),
            ..Default::default()
        });
        assert!(
            matches!(result, Err(InferError::InvalidConfig(_))),
            "top_p=0.0 should be InvalidConfig"
        );
    }

    #[test]
    fn invalid_top_p_gt_one() {
        let result = LogitsProcessor::new(LogitsProcessorConfig {
            top_p: Some(1.1),
            ..Default::default()
        });
        assert!(
            matches!(result, Err(InferError::InvalidConfig(_))),
            "top_p > 1.0 should be InvalidConfig"
        );
    }

    #[test]
    fn sample_empty_logits_error() {
        let proc = default_proc();
        let mut rng = Rng::new(0);
        assert!(
            matches!(
                proc.sample(&[], &mut rng),
                Err(InferError::SamplingError(_))
            ),
            "sampling empty logits should return SamplingError"
        );
    }
}
