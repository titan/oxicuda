//! CTC forward algorithm in the log domain.
//!
//! Computes the CTC log-likelihood for a given log-probability sequence and
//! label target using the standard forward-variable recursion over the
//! blank-interleaved extended target `l'`.

use crate::error::{AudioError, AudioResult};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp of two log-domain values.
///
/// Returns `NEG_INFINITY` when both inputs are `NEG_INFINITY`.
#[inline]
fn log_sum_exp2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    // m + ln(1 + exp(other - m))
    m + (1.0_f32 + (a.min(b) - m).exp()).ln()
}

// ─── Validation ──────────────────────────────────────────────────────────────

/// Validate all inputs before running the forward recursion.
fn validate_inputs(
    log_probs: &[f32],
    t: usize,
    v: usize,
    target: &[usize],
    blank: usize,
) -> AudioResult<()> {
    if v == 0 {
        return Err(AudioError::InvalidVocabSize(v));
    }
    if t == 0 {
        return Err(AudioError::InvalidSequenceLength(t));
    }
    if blank >= v {
        return Err(AudioError::BlankOutOfRange { blank, vocab: v });
    }
    if log_probs.len() != t * v {
        return Err(AudioError::ShapeMismatch {
            msg: format!(
                "log_probs length {} does not match T({}) * V({})",
                log_probs.len(),
                t,
                v
            ),
        });
    }
    for (i, &lbl) in target.iter().enumerate() {
        if lbl >= v {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "target[{}] = {} is out of range for vocabulary size {}",
                    i, lbl, v
                ),
            });
        }
        if lbl == blank {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "target[{}] = {} equals blank index {}; targets must not contain blank",
                    i, lbl, blank
                ),
            });
        }
    }
    Ok(())
}

// ─── Forward algorithm ───────────────────────────────────────────────────────

/// Build the blank-interleaved extended target `l'`.
///
/// For target `[a, b, c]` the extended target is
/// `[blank, a, blank, b, blank, c, blank]` with length `2 * |target| + 1`.
fn build_extended_target(target: &[usize], blank: usize) -> Vec<usize> {
    let s = 2 * target.len() + 1;
    let mut l_prime = vec![blank; s];
    for (i, &lbl) in target.iter().enumerate() {
        l_prime[2 * i + 1] = lbl;
    }
    l_prime
}

/// Compute CTC log-likelihood using the forward algorithm in the log domain.
///
/// # Parameters
///
/// - `log_probs`: Row-major `[T, V]` slice of log-softmax values.
/// - `t`: Number of time steps.
/// - `v`: Vocabulary size (including blank).
/// - `target`: Label sequence (each index in `[0, V)`, must not equal `blank`).
/// - `blank`: Blank label index.
///
/// # Errors
///
/// Returns:
/// - [`AudioError::InvalidVocabSize`] if `v == 0`.
/// - [`AudioError::InvalidSequenceLength`] if `t == 0`.
/// - [`AudioError::BlankOutOfRange`] if `blank >= v`.
/// - [`AudioError::ShapeMismatch`] if `log_probs.len() != t * v`, or any
///   target index is out-of-range or equals `blank`.
pub fn ctc_forward_log(
    log_probs: &[f32],
    t: usize,
    v: usize,
    target: &[usize],
    blank: usize,
) -> AudioResult<f32> {
    validate_inputs(log_probs, t, v, target, blank)?;

    let l_prime = build_extended_target(target, blank);
    let s_len = l_prime.len(); // 2 * |target| + 1

    // Initialise alpha in log domain; everything starts as -inf.
    let mut alpha = vec![f32::NEG_INFINITY; s_len];

    // t = 0: only first two positions of l' are reachable.
    alpha[0] = log_probs[blank]; // log_probs[0 * v + blank]
    if !target.is_empty() {
        alpha[1] = log_probs[l_prime[1]]; // log_probs[0 * v + l'[1]]
    }

    // Iterate over remaining time steps.
    for ts in 1..t {
        let row_offset = ts * v;
        // We must read the *previous* column while writing the current one,
        // so we swap into a temporary buffer.
        let prev = alpha.clone();
        alpha.fill(f32::NEG_INFINITY);

        for s in 0..s_len {
            // Start with contribution from same position and one step back.
            let mut val = log_sum_exp2(
                prev[s],
                if s > 0 {
                    prev[s - 1]
                } else {
                    f32::NEG_INFINITY
                },
            );

            // If this position carries a non-blank label that differs from
            // the label two positions back, we may also come from `s - 2`.
            if s >= 2 && l_prime[s] != blank && l_prime[s] != l_prime[s - 2] {
                val = log_sum_exp2(val, prev[s - 2]);
            }

            // Add emission probability.
            alpha[s] = val + log_probs[row_offset + l_prime[s]];
        }
    }

    // The total log-likelihood sums over all valid terminal positions in l'.
    // For non-empty targets: last blank l'[S-1] and last label l'[S-2].
    // For empty target (S=1): only position 0 (the single blank) is terminal.
    let ll = if s_len == 1 {
        alpha[0]
    } else {
        log_sum_exp2(alpha[s_len - 1], alpha[s_len - 2])
    };
    Ok(ll)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Uniform log-probability helper: each frame is a uniform distribution.
    fn uniform_log_probs(t: usize, v: usize) -> Vec<f32> {
        let lp = -(v as f32).ln();
        vec![lp; t * v]
    }

    // Softmax-normalise a raw logit row and return log.
    fn row_log_softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| (e / sum).ln()).collect()
    }

    // ── log_sum_exp2 unit tests ───────────────────────────────────────────────

    #[test]
    fn log_sum_exp2_both_neg_inf() {
        let result = log_sum_exp2(f32::NEG_INFINITY, f32::NEG_INFINITY);
        assert_eq!(result, f32::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp2_one_neg_inf_left() {
        let result = log_sum_exp2(f32::NEG_INFINITY, -3.0_f32);
        assert!((result - (-3.0_f32)).abs() < 1e-6);
    }

    #[test]
    fn log_sum_exp2_one_neg_inf_right() {
        let result = log_sum_exp2(-5.0_f32, f32::NEG_INFINITY);
        assert!((result - (-5.0_f32)).abs() < 1e-6);
    }

    #[test]
    fn log_sum_exp2_equal_values() {
        // log(e^a + e^a) = a + log(2)
        let a = -2.0_f32;
        let result = log_sum_exp2(a, a);
        let expected = a + 2.0_f32.ln();
        assert!(
            (result - expected).abs() < 1e-5,
            "got {result}, expected {expected}"
        );
    }

    // ── Error path tests ──────────────────────────────────────────────────────

    #[test]
    fn ctc_forward_invalid_vocab() {
        let lp = vec![0.0_f32; 0];
        let err = ctc_forward_log(&lp, 1, 0, &[], 0).unwrap_err();
        assert_eq!(err, AudioError::InvalidVocabSize(0));
    }

    #[test]
    fn ctc_forward_invalid_t() {
        let lp = vec![0.0_f32; 3];
        let err = ctc_forward_log(&lp, 0, 3, &[], 0).unwrap_err();
        assert_eq!(err, AudioError::InvalidSequenceLength(0));
    }

    #[test]
    fn ctc_forward_invalid_blank() {
        let lp = uniform_log_probs(1, 3);
        let err = ctc_forward_log(&lp, 1, 3, &[], 5).unwrap_err();
        assert_eq!(err, AudioError::BlankOutOfRange { blank: 5, vocab: 3 });
    }

    #[test]
    fn ctc_forward_target_equals_blank_error() {
        let lp = uniform_log_probs(2, 4);
        // blank = 0; target contains 0 → error
        let err = ctc_forward_log(&lp, 2, 4, &[0], 0).unwrap_err();
        assert!(matches!(err, AudioError::ShapeMismatch { .. }));
    }

    #[test]
    fn ctc_forward_target_out_of_range() {
        let lp = uniform_log_probs(2, 4);
        // vocab = 4, target contains index 10 → error
        let err = ctc_forward_log(&lp, 2, 4, &[10], 0).unwrap_err();
        assert!(matches!(err, AudioError::ShapeMismatch { .. }));
    }

    // ── Valid path tests ──────────────────────────────────────────────────────

    #[test]
    fn ctc_forward_empty_target_returns_sum_of_blanks() {
        // With no target labels, l' = [blank] (length 1).
        // The only valid path is all-blank, so log-likelihood = sum of log_probs[:,blank].
        let t = 4;
        let v = 3;
        let blank = 0;
        // Give the blank a fixed log-prob of ln(0.5) per frame, others don't matter.
        let lp_blank = 0.5_f32.ln();
        let mut lp = vec![0.0_f32; t * v];
        for ts in 0..t {
            // Simple: blank gets lp_blank, normalise the rest uniformly.
            let lp_other = ((1.0 - 0.5) / 2.0_f32).ln();
            lp[ts * v] = lp_blank;
            lp[ts * v + 1] = lp_other;
            lp[ts * v + 2] = lp_other;
        }
        let ll = ctc_forward_log(&lp, t, v, &[], blank).expect("should succeed for empty target");
        assert!(ll.is_finite(), "expected finite log-likelihood, got {ll}");
        // For empty target, alpha[0] after each step accumulates blank prob.
        // expected = sum of lp_blank over T frames
        let expected = (t as f32) * lp_blank;
        assert!((ll - expected).abs() < 1e-4, "ll={ll}, expected={expected}");
    }

    #[test]
    fn ctc_forward_single_symbol_one_frame_valid() {
        // T=1, V=3, blank=0, target=[1].
        // l' = [blank, 1, blank] (S=3). At t=0:
        //   alpha[0] = lp[blank]   (blank-only path → decodes to empty, not [1])
        //   alpha[1] = lp[1]       (label path → decodes to [1])
        //   alpha[2] = NEG_INF     (trailing blank needs at least 2 frames)
        // The CTC return for target [1] is log_sum_exp(alpha[S-1], alpha[S-2])
        //   = log_sum_exp(NEG_INF, lp[1]) = lp[1].
        let v = 3;
        let blank = 0;
        let lp = row_log_softmax(&[1.0_f32, 3.0, 0.5]);
        let ll = ctc_forward_log(&lp, 1, v, &[1], blank).expect("should succeed");
        assert!(ll.is_finite(), "expected finite, got {ll}");
        // At T=1 the only reachable terminal state for target=[1] is alpha[1]=lp[1].
        let expected = lp[1];
        assert!((ll - expected).abs() < 1e-5, "ll={ll}, expected={expected}");
    }

    #[test]
    fn ctc_forward_single_frame_single_label() {
        // T=1, V=3, blank=0, target=[1].
        // Same reasoning: only lp[1] contributes at T=1 for target=[1].
        let v = 3;
        let blank = 0;
        let raw = [2.0_f32, 5.0, 1.0];
        let lp = row_log_softmax(&raw);
        let ll = ctc_forward_log(&lp, 1, v, &[1], blank).expect("should succeed");
        let expected = lp[1]; // only the non-blank label path is a valid terminal
        assert!(ll.is_finite());
        assert!((ll - expected).abs() < 1e-5, "ll={ll}, expected={expected}");
    }

    #[test]
    fn ctc_forward_basic_valid() {
        // T=5, V=4, blank=0, target=[1,2].
        let t = 5;
        let v = 4;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let ll = ctc_forward_log(&lp, t, v, &[1, 2], blank)
            .expect("should return finite log-likelihood");
        assert!(ll.is_finite(), "expected finite, got {ll}");
        assert!(ll < 0.0, "log-likelihood must be ≤ 0; got {ll}");
    }

    #[test]
    fn ctc_forward_finite_random() {
        // Random but normalised rows, target of length 3.
        let t = 7;
        let v = 6;
        let blank = 0;
        // Deterministic pseudo-random via simple LCG.
        let mut seed: u64 = 0xdeadbeef_u64;
        let mut next = || -> f32 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f32 / u32::MAX as f32
        };
        let mut lp = Vec::with_capacity(t * v);
        for _ in 0..t {
            let raw: Vec<f32> = (0..v).map(|_| next() + 0.1).collect();
            lp.extend(row_log_softmax(&raw));
        }
        let ll = ctc_forward_log(&lp, t, v, &[1, 2, 3], blank)
            .expect("should succeed with normalised rows");
        assert!(ll.is_finite(), "expected finite, got {ll}");
    }

    #[test]
    fn ctc_forward_repeated_labels() {
        // target=[1,1]: l' = [0, 1, 0, 1, 0].
        // The two 1-labels are separated by blanks so they are valid distinct emissions.
        let t = 6;
        let v = 3;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let ll = ctc_forward_log(&lp, t, v, &[1, 1], blank).expect("repeated labels must be valid");
        assert!(ll.is_finite(), "expected finite, got {ll}");
    }

    #[test]
    fn ctc_forward_longer_target_possible() {
        // T=10, |target|=3 → S=7; well within T so paths exist.
        let t = 10;
        let v = 5;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let ll = ctc_forward_log(&lp, t, v, &[1, 2, 3], blank).expect("should succeed");
        assert!(ll.is_finite(), "expected finite, got {ll}");
    }

    #[test]
    fn ctc_forward_log_likelihood_is_nonpositive() {
        // Log-likelihood of a probability must be ≤ 0.
        let t = 4;
        let v = 4;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let ll = ctc_forward_log(&lp, t, v, &[1, 2], blank).expect("should succeed");
        assert!(
            ll <= 0.0_f32 + 1e-5,
            "log-likelihood should be ≤ 0; got {ll}"
        );
    }

    #[test]
    fn ctc_forward_shape_mismatch_wrong_len() {
        // Provide a slice that doesn't match T * V.
        let lp = vec![0.0_f32; 5]; // should be T=2, V=3 → 6
        let err = ctc_forward_log(&lp, 2, 3, &[], 0).unwrap_err();
        assert!(matches!(err, AudioError::ShapeMismatch { .. }));
    }
}
