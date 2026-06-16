//! # Beam Search Decoding
//!
//! Functional beam search implementation that takes a vocabulary distribution
//! for the first step plus a step function, and returns the top `beam_width`
//! completed (or truncated at `max_length`) candidate sequences.
//!
//! ## Algorithm
//!
//! 1. **Initialisation** — log-softmax the initial logits, take the top
//!    `beam_width` tokens as seed beams.
//! 2. **Expansion** — for each live beam call `step_fn(&tokens)` to obtain
//!    next-token logits, log-softmax them, and expand all `vocab_size`
//!    continuations.
//! 3. **Pruning** — keep the top `beam_width` candidates by cumulative score.
//! 4. **EOS handling** — beams that emit `eos_token` are moved to the
//!    completed set and not expanded further.
//! 5. **Termination** — stop when all beams are completed or the longest
//!    live beam reaches `max_length`.
//! 6. **Length normalisation** — `score /= max((len/6)^alpha, 1.0)`.
//! 7. **Return** — `beam_width` candidates sorted by normalised score descending.
//!
//! ## References
//!
//! * Wu et al. (2016) — "Google's Neural Machine Translation System" (length
//!   normalisation).

use crate::error::{InferError, InferResult};

// ─── BeamConfig ──────────────────────────────────────────────────────────────

/// Hyper-parameters for the functional beam search.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Number of parallel beams (search width ≥ 1).
    pub beam_width: usize,
    /// Maximum total sequence length (including all generated tokens).
    pub max_length: usize,
    /// Length-penalty exponent `α` used in `(len/6)^α`.
    /// 0.0 disables length penalty; common values are 0.6–1.0.
    pub length_penalty: f32,
    /// Token id that signals end-of-sequence.
    pub eos_token: usize,
}

// ─── BeamCandidate ────────────────────────────────────────────────────────────

/// A completed or truncated candidate hypothesis from beam search.
#[derive(Debug, Clone)]
pub struct BeamCandidate {
    /// Generated token sequence.
    pub tokens: Vec<usize>,
    /// Length-normalised log-probability score (higher = better).
    pub score: f32,
}

// ─── beam_search ─────────────────────────────────────────────────────────────

/// Run beam search and return `beam_width` candidates sorted by score descending.
///
/// # Arguments
///
/// * `initial_logits` — vocabulary-sized logit vector for the first step.
/// * `step_fn` — given the current token sequence, returns vocabulary-sized
///   logits for the next token.
/// * `vocab_size` — number of tokens in the vocabulary.
/// * `config` — beam search hyperparameters.
///
/// # Errors
///
/// * [`InferError::BeamSearchError`] — `vocab_size == 0` or `beam_width == 0`.
pub fn beam_search(
    initial_logits: &[f32],
    step_fn: impl Fn(&[usize]) -> Vec<f32>,
    vocab_size: usize,
    config: &BeamConfig,
) -> InferResult<Vec<BeamCandidate>> {
    // ── Validation ───────────────────────────────────────────────────────────
    if vocab_size == 0 {
        return Err(InferError::BeamSearchError(
            "vocab_size must be > 0".to_owned(),
        ));
    }
    if config.beam_width == 0 {
        return Err(InferError::BeamSearchError(
            "beam_width must be > 0".to_owned(),
        ));
    }

    // ── Step 0: Seed beams from initial logits ───────────────────────────────
    let log_probs_0 = log_softmax(initial_logits);

    // Gather (score, token) pairs and take the top beam_width.
    let mut seed_candidates: Vec<(f32, usize)> = log_probs_0
        .iter()
        .enumerate()
        .map(|(tok, &lp)| (lp, tok))
        .collect();
    seed_candidates
        .sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Live beams: (tokens, cumulative_score)
    let beam_width = config.beam_width.min(vocab_size);
    let mut live: Vec<(Vec<usize>, f32)> = seed_candidates
        .iter()
        .take(beam_width)
        .map(|&(score, tok)| {
            let finished = tok == config.eos_token;
            (vec![tok], score, finished)
        })
        .map(|(toks, sc, _)| (toks, sc))
        .collect();

    let mut completed: Vec<(Vec<usize>, f32)> = Vec::new();

    // Move any initial EOS beams to completed immediately.
    let mut still_live: Vec<(Vec<usize>, f32)> = Vec::new();
    for (toks, sc) in live {
        if toks.last().copied() == Some(config.eos_token) {
            completed.push((toks, sc));
        } else {
            still_live.push((toks, sc));
        }
    }
    live = still_live;

    // ── Expansion loop ───────────────────────────────────────────────────────
    while !live.is_empty() && live[0].0.len() < config.max_length {
        // Build the expanded candidate list across all live beams.
        let mut candidates: Vec<(Vec<usize>, f32)> = Vec::with_capacity(live.len() * vocab_size);

        for (tokens, cum_score) in &live {
            let next_logits = step_fn(tokens);
            let log_probs = log_softmax(&next_logits);

            for (tok, &lp) in log_probs.iter().enumerate() {
                let new_score = cum_score + lp;
                let mut new_tokens = tokens.clone();
                new_tokens.push(tok);
                candidates.push((new_tokens, new_score));
            }
        }

        // Sort all candidates by cumulative score (descending).
        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top beam_width candidates.
        let mut next_live: Vec<(Vec<usize>, f32)> = Vec::with_capacity(config.beam_width);
        for (toks, sc) in candidates.into_iter().take(config.beam_width) {
            if toks.last().copied() == Some(config.eos_token) {
                completed.push((toks, sc));
            } else {
                next_live.push((toks, sc));
            }
        }
        live = next_live;
    }

    // Move any remaining live beams (hit max_length) to completed.
    for (toks, sc) in live {
        completed.push((toks, sc));
    }

    // ── Length normalisation ─────────────────────────────────────────────────
    let alpha = config.length_penalty;
    let mut result: Vec<BeamCandidate> = completed
        .into_iter()
        .map(|(tokens, raw_score)| {
            let len = tokens.len() as f32;
            let penalty = (len / 6.0_f32).max(1.0_f32).powf(alpha);
            let norm_score = raw_score / penalty;
            BeamCandidate {
                tokens,
                score: norm_score,
            }
        })
        .collect();

    // Sort by normalised score descending (best first).
    result.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return exactly beam_width candidates (trim if we accumulated more
    // due to early EOS beams across multiple steps).
    result.truncate(config.beam_width);

    Ok(result)
}

// ─── log_softmax (local) ─────────────────────────────────────────────────────

/// Numerically stable log-softmax, implemented locally (not imported from
/// the sampling module to keep this module independent).
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = logits.iter().map(|&x| (x - max).exp()).sum::<f32>().ln();
    logits.iter().map(|&x| (x - max) - log_sum_exp).collect()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant step function: always returns the same logits.
    fn const_step(logits: Vec<f32>) -> impl Fn(&[usize]) -> Vec<f32> {
        move |_| logits.clone()
    }

    /// Peaked logits: high value at one token, zero elsewhere.
    fn peaked(vocab: usize, peak: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; vocab];
        v[peak] = 10.0;
        v
    }

    fn uniform(vocab: usize) -> Vec<f32> {
        vec![1.0_f32; vocab]
    }

    // ── 1. output_len ─────────────────────────────────────────────────────────
    #[test]
    fn output_len() {
        let vocab = 8_usize;
        let cfg = BeamConfig {
            beam_width: 3,
            max_length: 4,
            length_penalty: 0.0,
            eos_token: 0,
        };
        let result = beam_search(&uniform(vocab), const_step(uniform(vocab)), vocab, &cfg)
            .expect("valid beam search config and logits");
        assert_eq!(
            result.len(),
            cfg.beam_width,
            "beam_search should return exactly beam_width candidates"
        );
    }

    // ── 2. scores_descending ──────────────────────────────────────────────────
    #[test]
    fn scores_descending() {
        let vocab = 8_usize;
        let cfg = BeamConfig {
            beam_width: 3,
            max_length: 4,
            length_penalty: 0.0,
            eos_token: 0,
        };
        let result = beam_search(&uniform(vocab), const_step(uniform(vocab)), vocab, &cfg)
            .expect("valid config for descending-score test");
        for w in result.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "scores not descending: {} < {}",
                w[0].score,
                w[1].score
            );
        }
    }

    // ── 3. all_tokens_in_range ────────────────────────────────────────────────
    #[test]
    fn all_tokens_in_range() {
        let vocab = 10_usize;
        let cfg = BeamConfig {
            beam_width: 4,
            max_length: 5,
            length_penalty: 0.0,
            eos_token: 0,
        };
        let result = beam_search(&uniform(vocab), const_step(uniform(vocab)), vocab, &cfg)
            .expect("valid config for token-range test");
        for candidate in &result {
            for &tok in &candidate.tokens {
                assert!(tok < vocab, "token {tok} >= vocab_size {vocab}");
            }
        }
    }

    // ── 4. beam_1_is_greedy ───────────────────────────────────────────────────
    #[test]
    fn beam_1_is_greedy() {
        let vocab = 8_usize;
        let eos = 99_usize; // out of vocab range — won't occur
        let cfg = BeamConfig {
            beam_width: 1,
            max_length: 3,
            length_penalty: 0.0,
            eos_token: eos,
        };
        // Always peak at token 5.
        let logits = peaked(vocab, 5);
        let result = beam_search(&logits, const_step(logits.clone()), vocab, &cfg)
            .expect("valid greedy beam search config");
        assert_eq!(result.len(), 1);
        // Every token should be 5 (greedy argmax).
        for &tok in &result[0].tokens {
            assert_eq!(tok, 5, "beam_width=1 should follow greedy argmax");
        }
    }

    // ── 5. eos_stops_beam ─────────────────────────────────────────────────────
    #[test]
    fn eos_stops_beam() {
        let vocab = 4_usize;
        let eos = 0_usize;
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 10,
            length_penalty: 0.0,
            eos_token: eos,
        };
        // Always predict EOS (token 0) strongly.
        let logits = peaked(vocab, eos);
        let result = beam_search(&logits, const_step(logits.clone()), vocab, &cfg)
            .expect("valid eos_stops_beam config");
        // EOS should be the last token and NOT followed by more tokens.
        for candidate in &result {
            assert!(
                !candidate.tokens.is_empty(),
                "candidate should have at least one token"
            );
            // If EOS appears, it should only be at the end.
            let eos_positions: Vec<usize> = candidate
                .tokens
                .iter()
                .enumerate()
                .filter_map(|(i, &t)| if t == eos { Some(i) } else { None })
                .collect();
            if !eos_positions.is_empty() {
                let last_eos = *eos_positions.last().expect("eos_positions is non-empty");
                assert_eq!(
                    last_eos,
                    candidate.tokens.len() - 1,
                    "EOS must only appear at the final position"
                );
            }
        }
    }

    // ── 6. length_penalty_affects_score ──────────────────────────────────────
    #[test]
    fn length_penalty_affects_score() {
        // Verify that length_penalty=2.0 produces different scores than
        // length_penalty=0.0 when sequences run to max_length (length > 6).
        //
        // Raw scores are negative log-probabilities.  With length_penalty=2.0
        // and max_length=10, the divisor is (10/6)^2 ≈ 2.78.  Dividing a
        // negative number by a value > 1.0 makes it less negative (higher),
        // so the penalised score should be *greater* (closer to 0) than the
        // raw score which is used directly when length_penalty=0.0.
        let vocab = 4_usize;
        let eos = 99_usize; // will never occur — sequences run to max_length
        let base_cfg = BeamConfig {
            beam_width: 1,
            max_length: 10, // longer than 6, so (len/6)^alpha > 1.0
            length_penalty: 0.0,
            eos_token: eos,
        };
        let penalised_cfg = BeamConfig {
            length_penalty: 2.0,
            ..base_cfg.clone()
        };
        let logits = uniform(vocab);
        let result_no_penalty = beam_search(&logits, const_step(logits.clone()), vocab, &base_cfg)
            .expect("no-penalty beam search");
        let result_penalised =
            beam_search(&logits, const_step(logits.clone()), vocab, &penalised_cfg)
                .expect("penalised beam search");
        // Scores differ: with penalty the negative score is divided by > 1.0,
        // making it less negative.
        assert!(
            (result_penalised[0].score - result_no_penalty[0].score).abs() > 1e-6,
            "length_penalty=2.0 and length_penalty=0.0 should produce different scores; \
             penalised={} no_penalty={}",
            result_penalised[0].score,
            result_no_penalty[0].score
        );
        // With negative raw scores, dividing by penalty > 1 gives HIGHER (less negative) score.
        assert!(
            result_penalised[0].score > result_no_penalty[0].score,
            "with negative log-prob scores and penalty > 1, penalised score should be \
             greater (less negative); penalised={} no_penalty={}",
            result_penalised[0].score,
            result_no_penalty[0].score
        );
    }

    // ── 7. max_length_bounded ─────────────────────────────────────────────────
    #[test]
    fn max_length_bounded() {
        let vocab = 6_usize;
        let max_len = 5_usize;
        let eos = 99_usize; // never occurs
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: max_len,
            length_penalty: 0.0,
            eos_token: eos,
        };
        let logits = uniform(vocab);
        let result = beam_search(&logits, const_step(logits.clone()), vocab, &cfg)
            .expect("valid max_length config");
        for candidate in &result {
            assert!(
                candidate.tokens.len() <= max_len,
                "candidate length {} > max_length {}",
                candidate.tokens.len(),
                max_len
            );
        }
    }

    // ── 8. vocab_size_0_error ─────────────────────────────────────────────────
    #[test]
    fn vocab_size_0_error() {
        let cfg = BeamConfig {
            beam_width: 2,
            max_length: 4,
            length_penalty: 0.0,
            eos_token: 0,
        };
        let result = beam_search(&[], |_| vec![], 0, &cfg);
        assert!(
            matches!(result, Err(InferError::BeamSearchError(_))),
            "vocab_size=0 should return BeamSearchError"
        );
    }

    // ── 9. beam_0_error ──────────────────────────────────────────────────────
    #[test]
    fn beam_0_error() {
        let vocab = 4_usize;
        let cfg = BeamConfig {
            beam_width: 0,
            max_length: 4,
            length_penalty: 0.0,
            eos_token: 0,
        };
        let logits = uniform(vocab);
        let result = beam_search(&logits, const_step(logits.clone()), vocab, &cfg);
        assert!(
            matches!(result, Err(InferError::BeamSearchError(_))),
            "beam_width=0 should return BeamSearchError"
        );
    }

    // ── log_softmax sanity ────────────────────────────────────────────────────
    #[test]
    fn log_softmax_sums_to_one() {
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        let lsm = log_softmax(&logits);
        let sum_exp: f64 = lsm.iter().map(|&x| (x as f64).exp()).sum();
        assert!(
            (sum_exp - 1.0).abs() < 1e-5,
            "sum of exp(log_softmax) should be 1.0, got {sum_exp}"
        );
    }
}
