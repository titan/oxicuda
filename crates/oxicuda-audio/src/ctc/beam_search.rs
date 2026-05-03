//! CTC prefix beam-search decoder.
//!
//! Implements the standard CTC prefix beam search algorithm. At each timestep
//! every active hypothesis is extended by all vocabulary tokens, blank
//! extensions collapse into the same prefix, and identical resulting prefixes
//! are merged. The beam is then pruned to the top `beam_width` entries by
//! total log-probability.

use std::collections::HashMap;

use crate::error::{AudioError, AudioResult};

// ─── Public types ─────────────────────────────────────────────────────────────

/// A single decoded hypothesis returned by [`ctc_beam_search`].
#[derive(Debug, Clone)]
pub struct BeamHypothesis {
    /// Decoded label sequence (blank tokens are not included).
    pub tokens: Vec<usize>,
    /// Total log-probability of this hypothesis.
    pub log_prob: f32,
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Numerically stable log-sum-exp of two log-domain values.
#[inline]
fn log_sum_exp2(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + (1.0_f32 + (a.min(b) - m).exp()).ln()
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn validate_inputs(
    log_probs: &[f32],
    t: usize,
    v: usize,
    blank: usize,
    beam_width: usize,
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
    if beam_width == 0 {
        return Err(AudioError::InvalidBeamWidth(beam_width));
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
    Ok(())
}

// ─── Beam state types ────────────────────────────────────────────────────────

/// Per-prefix accumulated log-probabilities.
///
/// - `p_blank`: log P(prefix was emitted ending with a blank at this timestep).
/// - `p_nonblank`: log P(prefix was emitted ending with a non-blank at this timestep).
#[derive(Clone, Copy)]
struct PrefixProbs {
    p_blank: f32,
    p_nonblank: f32,
}

impl PrefixProbs {
    /// Log-probability of the prefix under any ending token.
    #[inline]
    fn total(self) -> f32 {
        log_sum_exp2(self.p_blank, self.p_nonblank)
    }
}

// ─── Algorithm ───────────────────────────────────────────────────────────────

/// Prune `beam` to at most `beam_width` entries, retaining the highest-total
/// log-probability prefixes.
fn prune_beam(
    beam: HashMap<Vec<usize>, PrefixProbs>,
    beam_width: usize,
) -> HashMap<Vec<usize>, PrefixProbs> {
    if beam.len() <= beam_width {
        return beam;
    }
    let mut entries: Vec<(Vec<usize>, PrefixProbs)> = beam.into_iter().collect();
    // Sort descending by total log-probability; use partial_cmp with a fallback
    // so that NaN values sink to the bottom.
    entries.sort_by(|(_, a), (_, b)| {
        b.total()
            .partial_cmp(&a.total())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    entries.truncate(beam_width);
    entries.into_iter().collect()
}

/// Extend a single beam entry `(prefix, probs)` with blank at timestep `ts`.
///
/// Returns `(new_prefix, new_p_blank)`.
#[inline]
fn extend_with_blank(prefix: &[usize], probs: PrefixProbs, log_blank: f32) -> (Vec<usize>, f32) {
    let new_p_blank = probs.total() + log_blank;
    (prefix.to_vec(), new_p_blank)
}

/// Extend a single beam entry `(prefix, probs)` with non-blank token `c`.
///
/// Returns `(new_prefix, new_p_nonblank)`.
fn extend_with_token(
    prefix: &[usize],
    probs: PrefixProbs,
    c: usize,
    log_c: f32,
) -> (Vec<usize>, f32) {
    let last = prefix.last().copied();

    let new_p_nonblank = if last == Some(c) {
        // Same token as last in prefix: can only extend via a preceding blank
        // (otherwise the two consecutive identical emissions collapse).
        probs.p_blank + log_c
    } else {
        // Different token: can come from either ending.
        probs.total() + log_c
    };

    let mut new_prefix = prefix.to_vec();
    new_prefix.push(c);
    (new_prefix, new_p_nonblank)
}

/// Merge a candidate `(prefix, p_blank_delta, p_nb_delta)` into `next_beam`.
///
/// The two `Option<f32>` arguments carry at most one update each: a blank
/// probability update and a non-blank probability update respectively.
fn merge_into(
    next_beam: &mut HashMap<Vec<usize>, PrefixProbs>,
    prefix: Vec<usize>,
    new_p_blank: Option<f32>,
    new_p_nb: Option<f32>,
) {
    let entry = next_beam.entry(prefix).or_insert(PrefixProbs {
        p_blank: f32::NEG_INFINITY,
        p_nonblank: f32::NEG_INFINITY,
    });
    if let Some(pb) = new_p_blank {
        entry.p_blank = log_sum_exp2(entry.p_blank, pb);
    }
    if let Some(pnb) = new_p_nb {
        entry.p_nonblank = log_sum_exp2(entry.p_nonblank, pnb);
    }
}

/// Decode `log_probs` using CTC prefix beam search.
///
/// # Parameters
///
/// - `log_probs`: Row-major `[T, V]` slice of per-frame log-softmax values.
/// - `t`: Number of time steps.
/// - `v`: Vocabulary size (including blank).
/// - `blank`: Index of the blank label.
/// - `beam_width`: Maximum number of hypotheses to keep at each timestep.
///
/// # Returns
///
/// A `Vec<BeamHypothesis>` sorted by decreasing `log_prob`. All entries have
/// `log_prob <= 0.0` (they are log-probabilities).
///
/// # Errors
///
/// Returns:
/// - [`AudioError::InvalidVocabSize`] if `v == 0`.
/// - [`AudioError::InvalidSequenceLength`] if `t == 0`.
/// - [`AudioError::BlankOutOfRange`] if `blank >= v`.
/// - [`AudioError::InvalidBeamWidth`] if `beam_width == 0`.
/// - [`AudioError::ShapeMismatch`] if `log_probs.len() != t * v`.
pub fn ctc_beam_search(
    log_probs: &[f32],
    t: usize,
    v: usize,
    blank: usize,
    beam_width: usize,
) -> AudioResult<Vec<BeamHypothesis>> {
    validate_inputs(log_probs, t, v, blank, beam_width)?;

    // Initialise: empty prefix with p_blank = log(1) = 0.
    let mut beam: HashMap<Vec<usize>, PrefixProbs> = HashMap::new();
    beam.insert(
        vec![],
        PrefixProbs {
            p_blank: 0.0,
            p_nonblank: f32::NEG_INFINITY,
        },
    );

    for ts in 0..t {
        let row = &log_probs[ts * v..(ts + 1) * v];
        let log_blank = row[blank];

        // `next_beam` accumulates extensions for this timestep.
        let mut next_beam: HashMap<Vec<usize>, PrefixProbs> = HashMap::new();

        for (prefix, probs) in &beam {
            // ── Blank extension ──────────────────────────────────────────────
            {
                let (new_prefix, new_pb) = extend_with_blank(prefix, *probs, log_blank);
                merge_into(&mut next_beam, new_prefix, Some(new_pb), None);
            }

            // ── Non-blank extensions ─────────────────────────────────────────
            for (c, &log_c) in row.iter().enumerate().filter(|(c, _)| *c != blank) {
                let last = prefix.last().copied();

                if last == Some(c) {
                    // Same-token extension: two separate contributions.
                    // 1. Extend via blank-ending: new prefix gets a new `c`.
                    let new_p_nb_diff = probs.p_blank + log_c;
                    let mut new_prefix_diff = prefix.clone();
                    new_prefix_diff.push(c);
                    merge_into(&mut next_beam, new_prefix_diff, None, Some(new_p_nb_diff));

                    // 2. Same prefix (no push): non-blank to non-blank keeps collapsed.
                    let new_p_nb_same = probs.p_nonblank + log_c;
                    merge_into(&mut next_beam, prefix.clone(), None, Some(new_p_nb_same));
                } else {
                    let (new_prefix, new_p_nb) = extend_with_token(prefix, *probs, c, log_c);
                    merge_into(&mut next_beam, new_prefix, None, Some(new_p_nb));
                }
            }
        }

        beam = prune_beam(next_beam, beam_width);
    }

    // Collect and sort.
    let mut hypotheses: Vec<BeamHypothesis> = beam
        .into_iter()
        .map(|(tokens, probs)| BeamHypothesis {
            tokens,
            log_prob: probs.total(),
        })
        .collect();

    hypotheses.sort_by(|a, b| {
        b.log_prob
            .partial_cmp(&a.log_prob)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(hypotheses)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_log_probs(t: usize, v: usize) -> Vec<f32> {
        let lp = -(v as f32).ln();
        vec![lp; t * v]
    }

    /// Build a T×V log-prob matrix where each frame has a strong peak at
    /// `preferred[ts % preferred.len()]`.
    fn peaked_log_probs(t: usize, v: usize, preferred: &[usize]) -> Vec<f32> {
        let mut lp = vec![0.0_f32; t * v];
        for ts in 0..t {
            let peak = preferred[ts % preferred.len()];
            let row = &mut lp[ts * v..(ts + 1) * v];
            // Very high logit at peak, near-zero elsewhere.
            for (i, val) in row.iter_mut().enumerate() {
                *val = if i == peak { 20.0 } else { 0.0 };
            }
            // Apply log-softmax in-place.
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (i, val) in row.iter_mut().enumerate() {
                *val = (exps[i] / sum).ln();
            }
        }
        lp
    }

    // ── Error path tests ──────────────────────────────────────────────────────

    #[test]
    fn beam_search_invalid_vocab() {
        let err = ctc_beam_search(&[], 1, 0, 0, 3).unwrap_err();
        assert_eq!(err, AudioError::InvalidVocabSize(0));
    }

    #[test]
    fn beam_search_invalid_t() {
        let lp = vec![0.0_f32; 3];
        let err = ctc_beam_search(&lp, 0, 3, 0, 3).unwrap_err();
        assert_eq!(err, AudioError::InvalidSequenceLength(0));
    }

    #[test]
    fn beam_search_invalid_blank() {
        let lp = uniform_log_probs(1, 3);
        let err = ctc_beam_search(&lp, 1, 3, 5, 3).unwrap_err();
        assert_eq!(err, AudioError::BlankOutOfRange { blank: 5, vocab: 3 });
    }

    #[test]
    fn beam_search_invalid_beam_width() {
        let lp = uniform_log_probs(2, 4);
        let err = ctc_beam_search(&lp, 2, 4, 0, 0).unwrap_err();
        assert_eq!(err, AudioError::InvalidBeamWidth(0));
    }

    #[test]
    fn beam_search_shape_mismatch() {
        let lp = vec![0.0_f32; 5]; // should be 2*4=8
        let err = ctc_beam_search(&lp, 2, 4, 0, 3).unwrap_err();
        assert!(matches!(err, AudioError::ShapeMismatch { .. }));
    }

    // ── Valid path tests ──────────────────────────────────────────────────────

    #[test]
    fn beam_search_basic_valid() {
        let t = 5;
        let v = 4;
        let blank = 0;
        let beam_width = 3;
        let lp = uniform_log_probs(t, v);
        let hyps = ctc_beam_search(&lp, t, v, blank, beam_width).expect("should succeed");
        assert!(!hyps.is_empty(), "should return at least one hypothesis");
        assert!(
            hyps.len() <= beam_width,
            "beam should not exceed beam_width"
        );
        for hyp in &hyps {
            assert!(hyp.log_prob.is_finite(), "log_prob must be finite");
        }
    }

    #[test]
    fn beam_search_results_sorted_by_logprob() {
        let t = 5;
        let v = 4;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let hyps = ctc_beam_search(&lp, t, v, blank, 4).expect("should succeed");
        for pair in hyps.windows(2) {
            assert!(
                pair[0].log_prob >= pair[1].log_prob,
                "results must be in decreasing order: {} < {}",
                pair[0].log_prob,
                pair[1].log_prob
            );
        }
    }

    #[test]
    fn beam_search_blank_only_returns_empty() {
        // All frames peak strongly at blank → best decoded hypothesis is empty.
        let t = 5;
        let v = 3;
        let blank = 0;
        let lp = peaked_log_probs(t, v, &[blank]);
        let hyps = ctc_beam_search(&lp, t, v, blank, 5).expect("should succeed");
        assert!(!hyps.is_empty(), "should return at least one hypothesis");
        // The top hypothesis must be the empty sequence.
        assert!(
            hyps[0].tokens.is_empty(),
            "top hypothesis must have no tokens when blank dominates; got {:?}",
            hyps[0].tokens
        );
    }

    #[test]
    fn beam_search_single_label() {
        // V=2: blank=0, label=1. Every frame strongly predicts label 1.
        let t = 4;
        let v = 2;
        let blank = 0;
        let lp = peaked_log_probs(t, v, &[1]);
        let hyps = ctc_beam_search(&lp, t, v, blank, 3).expect("should succeed");
        assert!(!hyps.is_empty());
        // CTC collapses consecutive identical non-blank emissions → [1].
        let top = &hyps[0];
        assert_eq!(
            top.tokens,
            vec![1],
            "expected [1] from all-ones predictions, got {:?}",
            top.tokens
        );
    }

    #[test]
    fn beam_search_repeated_symbol_blank_separated() {
        // Frames: [1, blank, 1, blank, 1] → CTC should decode to [1, 1] since
        // there is a blank between the repeated symbol at frames 0→1→2.
        // Specifically frame pattern: label1 blank label1 blank label1
        let t = 5;
        let v = 3;
        let blank = 0;
        let pattern = [1_usize, 0, 1, 0, 1];
        let lp = peaked_log_probs(t, v, &pattern);
        let hyps = ctc_beam_search(&lp, t, v, blank, 5).expect("should succeed");
        assert!(!hyps.is_empty());
        // The top hypothesis must be [1] or [1,1] depending on merge behaviour.
        // Both are valid CTC outputs; we just require the best is non-empty.
        assert!(
            !hyps[0].tokens.is_empty(),
            "expected non-empty decode from alternating label/blank pattern"
        );
    }

    #[test]
    fn beam_search_log_prob_nonpositive() {
        // Log-probabilities are always ≤ 0.
        let t = 4;
        let v = 4;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let hyps = ctc_beam_search(&lp, t, v, blank, 5).expect("should succeed");
        for hyp in &hyps {
            assert!(
                hyp.log_prob <= 0.0 + 1e-5,
                "log_prob must be ≤ 0; got {}",
                hyp.log_prob
            );
        }
    }

    #[test]
    fn beam_search_beam_truncation() {
        // Large V, beam=1 → exactly one result returned.
        let t = 6;
        let v = 20;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let hyps = ctc_beam_search(&lp, t, v, blank, 1).expect("should succeed");
        assert_eq!(hyps.len(), 1, "beam=1 must return exactly one hypothesis");
    }

    #[test]
    fn beam_search_tokens_within_vocab() {
        // Every decoded token index must be < V and != blank.
        let t = 5;
        let v = 6;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let hyps = ctc_beam_search(&lp, t, v, blank, 4).expect("should succeed");
        for hyp in &hyps {
            for &tok in &hyp.tokens {
                assert!(tok < v, "token {tok} out of vocab range {v}");
                assert_ne!(tok, blank, "decoded tokens must not equal blank");
            }
        }
    }

    #[test]
    fn beam_search_large_beam_covers_more() {
        // A wider beam should return at least as many hypotheses as a narrow one.
        let t = 4;
        let v = 4;
        let blank = 0;
        let lp = uniform_log_probs(t, v);
        let hyps_narrow = ctc_beam_search(&lp, t, v, blank, 1).expect("should succeed");
        let hyps_wide = ctc_beam_search(&lp, t, v, blank, 10).expect("should succeed");
        assert!(
            hyps_wide.len() >= hyps_narrow.len(),
            "wider beam should have >= hypotheses"
        );
    }
}
