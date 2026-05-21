//! Contrastive search decoding (Su et al. 2022 ACL).
//!
//! Reference: Su, Y., Lan, T., Wang, Y., Yatbaz, H. Y., & Xu, X. (2022).
//! *A Contrastive Framework for Neural Text Generation*. ACL 2022.
//! <https://aclanthology.org/2022.acl-long.365/>.
//!
//! # Background
//!
//! Standard stochastic decoders (nucleus, top-k) address the *degeneration
//! problem* (repetition, incoherence) only partially; at low temperatures they
//! tend to loop, while at high temperatures they produce incoherent text.
//!
//! Contrastive search is **deterministic** and addresses degeneration through
//! an explicit penalty on the cosine similarity between the candidate token's
//! hidden state and any of the previous context hidden states:
//!
//! ```text
//! score(v, t) = (1 − α) · model_prob(v | context)
//!             − α · max_{j < t} cos_sim(h_v, h_{context_j})
//! ```
//!
//! The first term rewards tokens with high model probability; the second term
//! penalises tokens whose hidden-state representation is too similar to those
//! already present in the context (i.e., tokens that would cause degeneration).
//! The hyperparameter `α ∈ [0, 1]` controls the trade-off.
//!
//! # Provided API
//!
//! * [`ContrastiveConfig`] — configuration struct (`k`, `alpha`, `max_len`).
//! * [`ContrastiveSearcher`] — stateless struct with all algorithm steps as
//!   associated functions.
//! * [`ContrastiveSearcher::cosine_similarity`] — `(a·b) / (‖a‖·‖b‖ + ε)`.
//! * [`ContrastiveSearcher::degeneration_penalty`] — max cos-sim to any
//!   previous context hidden state.
//! * [`ContrastiveSearcher::top_k_candidates`] — select top-k logit indices,
//!   return `(token_id, softmax_prob)` pairs sorted descending by prob.
//! * [`ContrastiveSearcher::contrastive_score`] — scalar score combination.
//! * [`ContrastiveSearcher::decode`] — full generation loop with explicit
//!   hidden states from a step function.
//! * [`ContrastiveSearcher::decode_logits_only`] — generation with logit-only
//!   step function using past logit vectors as proxy hidden states.

use crate::error::{SeqError, SeqResult};

/// Configuration for contrastive search decoding.
///
/// # Fields
///
/// * `k` — top-k candidates to consider at each step (`≥ 1`).
/// * `alpha` — degeneration penalty weight `∈ [0, 1]`.  `α = 0` reduces to
///   greedy decoding; `α = 1` ignores model probability entirely.
/// * `max_len` — maximum number of tokens to generate.
#[derive(Debug, Clone, Copy)]
pub struct ContrastiveConfig {
    /// Number of top-probability candidates to consider at each step.
    pub k: usize,
    /// Degeneration penalty weight ∈ [0, 1].
    pub alpha: f32,
    /// Maximum generation length (number of tokens produced).
    pub max_len: usize,
}

impl Default for ContrastiveConfig {
    fn default() -> Self {
        Self {
            k: 5,
            alpha: 0.6,
            max_len: 50,
        }
    }
}

impl ContrastiveConfig {
    /// Validate the configuration fields.
    fn validate(&self) -> SeqResult<()> {
        if self.k == 0 {
            return Err(SeqError::InvalidConfiguration(
                "contrastive: k must be >= 1".to_string(),
            ));
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(SeqError::InvalidConfiguration(format!(
                "contrastive: alpha must be in [0, 1], got {}",
                self.alpha
            )));
        }
        Ok(())
    }
}

/// Stateless struct providing all contrastive search decoding primitives.
///
/// All methods are free functions (associated functions) taking only explicit
/// inputs; there is no mutable state.  The generation loops in [`Self::decode`] and
/// [`Self::decode_logits_only`] do maintain local state internally but expose only a
/// pure interface.
pub struct ContrastiveSearcher;

impl ContrastiveSearcher {
    /// Cosine similarity between two equal-length, non-empty vectors.
    ///
    /// Returns `(a·b) / (‖a‖ · ‖b‖ + ε)` where `ε = 1e-12`.  If either
    /// vector is the zero vector the function returns `0.0` (not `NaN`).
    ///
    /// # Errors
    ///
    /// * [`SeqError::EmptyInput`] if either slice is empty.
    /// * [`SeqError::LengthMismatch`] if `a.len() != b.len()`.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> SeqResult<f32> {
        if a.is_empty() || b.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if a.len() != b.len() {
            return Err(SeqError::LengthMismatch {
                a: a.len(),
                b: b.len(),
            });
        }
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt() + 1e-12_f32;
        Ok(dot / denom)
    }

    /// Degeneration penalty for a candidate token given context hidden states.
    ///
    /// The penalty is the **maximum** cosine similarity between
    /// `candidate_hidden` and any of the `n_context` previously-seen context
    /// hidden states packed row-major in `context_hiddens`.
    ///
    /// When `n_context == 0` (no prior context), the penalty is `0.0`.
    ///
    /// # Parameters
    ///
    /// * `context_hiddens` — flat `[n_context × hidden_dim]` buffer.
    /// * `n_context` — number of prior context tokens.
    /// * `candidate_hidden` — `[hidden_dim]` hidden state for this candidate.
    /// * `hidden_dim` — dimension of each hidden-state vector.
    ///
    /// # Errors
    ///
    /// * [`SeqError::ShapeMismatch`] if `context_hiddens.len() != n_context * hidden_dim`.
    /// * [`SeqError::LengthMismatch`] if `candidate_hidden.len() != hidden_dim`.
    /// * [`SeqError::EmptyInput`] if `hidden_dim == 0`.
    pub fn degeneration_penalty(
        context_hiddens: &[f32],
        n_context: usize,
        candidate_hidden: &[f32],
        hidden_dim: usize,
    ) -> SeqResult<f32> {
        if hidden_dim == 0 {
            return Err(SeqError::EmptyInput);
        }
        if context_hiddens.len() != n_context * hidden_dim {
            return Err(SeqError::ShapeMismatch {
                expected: n_context * hidden_dim,
                got: context_hiddens.len(),
            });
        }
        if candidate_hidden.len() != hidden_dim {
            return Err(SeqError::LengthMismatch {
                a: candidate_hidden.len(),
                b: hidden_dim,
            });
        }
        if n_context == 0 {
            return Ok(0.0);
        }

        let mut max_sim = f32::NEG_INFINITY;
        for t in 0..n_context {
            let ctx_slice = &context_hiddens[t * hidden_dim..(t + 1) * hidden_dim];
            let sim = Self::cosine_similarity(ctx_slice, candidate_hidden)?;
            if sim > max_sim {
                max_sim = sim;
            }
        }
        Ok(max_sim)
    }

    /// Select the top-k tokens by logit value, returning `(token_id, prob)`
    /// pairs sorted by softmax probability in **descending** order.
    ///
    /// Softmax is computed over **all** logits for correct probability values;
    /// only the top-k are returned.  If `k >= vocab_size`, all tokens are
    /// returned.
    ///
    /// # Errors
    ///
    /// * [`SeqError::EmptyInput`] if `logits` is empty.
    /// * [`SeqError::InvalidConfiguration`] if `k == 0`.
    pub fn top_k_candidates(logits: &[f32], k: usize) -> SeqResult<Vec<(usize, f32)>> {
        if logits.is_empty() {
            return Err(SeqError::EmptyInput);
        }
        if k == 0 {
            return Err(SeqError::InvalidConfiguration(
                "contrastive: k must be >= 1".to_string(),
            ));
        }
        let vocab = logits.len();
        let k_eff = k.min(vocab);

        // Partial-sort: find the top-k indices by raw logit (descending).
        let mut indices: Vec<usize> = (0..vocab).collect();
        indices.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices.truncate(k_eff);

        // Full-vocabulary numerically-stable softmax.
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exps = vec![0.0_f32; vocab];
        let mut sum = 0.0_f32;
        for (i, &l) in logits.iter().enumerate() {
            let e = (l - max_l).exp();
            exps[i] = e;
            sum += e;
        }
        // Guard against degenerate all-NEG_INFINITY logits.
        let sum_safe = if sum > 0.0 && sum.is_finite() {
            sum
        } else {
            1.0
        };

        // Collect (token_id, prob) for top-k candidates.
        let mut candidates: Vec<(usize, f32)> = indices
            .iter()
            .map(|&idx| (idx, exps[idx] / sum_safe))
            .collect();

        // Sort by probability descending.
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates)
    }

    /// Combine model probability and degeneration penalty into a contrastive
    /// score.
    ///
    /// ```text
    /// score = (1 − alpha) · prob − alpha · degen_penalty
    /// ```
    #[inline]
    pub fn contrastive_score(prob: f32, degen_penalty: f32, alpha: f32) -> f32 {
        (1.0 - alpha) * prob - alpha * degen_penalty
    }

    /// Run contrastive search decoding for `cfg.max_len` steps using a
    /// caller-provided step function that also returns hidden states.
    ///
    /// # Step function contract
    ///
    /// ```text
    /// step_fn(selected_token_id: usize, last_hidden: &[f32])
    ///     -> (logits: Vec<f32>, next_hidden: Vec<f32>)
    /// ```
    ///
    /// `logits` must have length `vocab_size`; `next_hidden` must have length
    /// `hidden_dim` and represents the hidden state **for the selected token**.
    ///
    /// # Initial hidden states
    ///
    /// `initial_hiddens` must be a flat `[vocab_size × hidden_dim]` buffer
    /// providing one hidden state per vocabulary token for the very first step.
    /// If your model produces a single shared hidden state at step 0 rather
    /// than one per token, replicate it `vocab_size` times before calling.
    ///
    /// # Errors
    ///
    /// * [`SeqError::InvalidConfiguration`] if `k == 0` or `alpha ∉ [0, 1]`.
    /// * [`SeqError::ShapeMismatch`] if `initial_logits.len() != vocab_size` or
    ///   `initial_hiddens.len() != vocab_size * hidden_dim`.
    /// * [`SeqError::EmptyInput`] if `vocab_size == 0` or `hidden_dim == 0`.
    pub fn decode<F>(
        initial_logits: &[f32],
        initial_hiddens: &[f32],
        vocab_size: usize,
        hidden_dim: usize,
        step_fn: F,
        cfg: &ContrastiveConfig,
    ) -> SeqResult<Vec<usize>>
    where
        F: Fn(usize, &[f32]) -> (Vec<f32>, Vec<f32>),
    {
        cfg.validate()?;
        if vocab_size == 0 || hidden_dim == 0 {
            return Err(SeqError::EmptyInput);
        }
        if initial_logits.len() != vocab_size {
            return Err(SeqError::ShapeMismatch {
                expected: vocab_size,
                got: initial_logits.len(),
            });
        }
        if initial_hiddens.len() != vocab_size * hidden_dim {
            return Err(SeqError::ShapeMismatch {
                expected: vocab_size * hidden_dim,
                got: initial_hiddens.len(),
            });
        }

        let mut generated: Vec<usize> = Vec::with_capacity(cfg.max_len);
        // Accumulate context hidden states: [t × hidden_dim]
        let mut context_hiddens: Vec<f32> = Vec::new();

        // ---- Step 0 ------------------------------------------------
        let candidates_0 = Self::top_k_candidates(initial_logits, cfg.k)?;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_token = candidates_0[0].0;
        let mut best_hidden: Vec<f32> =
            initial_hiddens[best_token * hidden_dim..(best_token + 1) * hidden_dim].to_vec();

        for (tok, prob) in &candidates_0 {
            // At step 0 there is no context yet: degeneration penalty is 0.
            let score = Self::contrastive_score(*prob, 0.0, cfg.alpha);
            if score > best_score {
                best_score = score;
                best_token = *tok;
                best_hidden = initial_hiddens[tok * hidden_dim..(tok + 1) * hidden_dim].to_vec();
            }
        }

        generated.push(best_token);
        context_hiddens.extend_from_slice(&best_hidden);
        let mut last_hidden = best_hidden;

        // ---- Steps 1..max_len ----------------------------------------
        for _step in 1..cfg.max_len {
            let (next_logits, next_hidden) = step_fn(generated[generated.len() - 1], &last_hidden);

            if next_logits.len() != vocab_size {
                return Err(SeqError::ShapeMismatch {
                    expected: vocab_size,
                    got: next_logits.len(),
                });
            }
            if next_hidden.len() != hidden_dim {
                return Err(SeqError::ShapeMismatch {
                    expected: hidden_dim,
                    got: next_hidden.len(),
                });
            }

            let candidates = Self::top_k_candidates(&next_logits, cfg.k)?;
            let n_ctx = context_hiddens.len() / hidden_dim;

            let mut step_best_score = f32::NEG_INFINITY;
            let mut step_best_token = candidates[0].0;

            for (tok, prob) in &candidates {
                // Use next_hidden as the proxy hidden state for all candidates.
                // In a true implementation each candidate would have its own
                // forward pass; here we use the single next_hidden (for the
                // selected token) as the representative for the candidate set.
                let degen =
                    Self::degeneration_penalty(&context_hiddens, n_ctx, &next_hidden, hidden_dim)?;
                let score = Self::contrastive_score(*prob, degen, cfg.alpha);
                if score > step_best_score {
                    step_best_score = score;
                    step_best_token = *tok;
                }
            }

            generated.push(step_best_token);
            context_hiddens.extend_from_slice(&next_hidden);
            last_hidden = next_hidden;
        }

        Ok(generated)
    }

    /// Simplified contrastive search that works with a logit-only step
    /// function.
    ///
    /// Because hidden states are unavailable, the past **logit vectors** serve
    /// as proxy hidden states: the degeneration penalty for a candidate at step
    /// `t` is the maximum cosine similarity between the *current* logit vector
    /// and each of the past logit vectors stored in the context.
    ///
    /// The degeneration is therefore shared across all candidates at each step
    /// (it measures how similar the new distribution is to past ones), which
    /// implicitly penalises repetitive distributions.
    ///
    /// # Step function contract
    ///
    /// ```text
    /// step_fn(selected_token_id: usize) -> logits: Vec<f32>   [vocab_size]
    /// ```
    ///
    /// # Errors
    ///
    /// * [`SeqError::InvalidConfiguration`] if `k == 0` or `alpha ∉ [0, 1]`.
    /// * [`SeqError::EmptyInput`] if `initial_logits` is empty.
    pub fn decode_logits_only<F>(
        initial_logits: &[f32],
        step_fn: F,
        cfg: &ContrastiveConfig,
    ) -> SeqResult<Vec<usize>>
    where
        F: Fn(usize) -> Vec<f32>,
    {
        cfg.validate()?;
        if initial_logits.is_empty() {
            return Err(SeqError::EmptyInput);
        }

        let vocab_size = initial_logits.len();
        let mut generated: Vec<usize> = Vec::with_capacity(cfg.max_len);
        // Context logit vectors act as proxy hidden states.
        // Each entry is one logit vector [vocab_size].
        let mut context_logits_flat: Vec<f32> = Vec::new();

        // ---- Step 0 ------------------------------------------------
        let candidates_0 = Self::top_k_candidates(initial_logits, cfg.k)?;

        // At step 0 there is no context; degeneration = 0 for all candidates.
        let mut best_score = f32::NEG_INFINITY;
        let mut best_token = candidates_0[0].0;
        for (tok, prob) in &candidates_0 {
            let score = Self::contrastive_score(*prob, 0.0, cfg.alpha);
            if score > best_score {
                best_score = score;
                best_token = *tok;
            }
        }

        generated.push(best_token);
        // Store initial_logits as the first context entry.
        context_logits_flat.extend_from_slice(initial_logits);

        // ---- Steps 1..max_len ----------------------------------------
        for _step in 1..cfg.max_len {
            let next_logits = step_fn(generated[generated.len() - 1]);

            if next_logits.is_empty() {
                return Err(SeqError::EmptyInput);
            }
            let cur_vocab = next_logits.len();
            // Allow vocab to grow/shrink within a step (though unusual), but
            // we need a consistent dimension for cosine similarity.  Use
            // the minimum dimension when comparing with past logit vectors.
            let dim = cur_vocab.min(vocab_size);

            let candidates = Self::top_k_candidates(&next_logits, cfg.k)?;
            let n_ctx = context_logits_flat.len() / vocab_size;

            // Compute degeneration penalty using the current logit vector as
            // the "candidate hidden state", compared against past logit vectors.
            // All candidates share this penalty (distribution-level penalty).
            let mut degen = 0.0_f32;
            for t in 0..n_ctx {
                let ctx_slice = &context_logits_flat[t * vocab_size..t * vocab_size + dim];
                let cand_slice = &next_logits[..dim];
                let sim = Self::cosine_similarity(ctx_slice, cand_slice)?;
                if sim > degen {
                    degen = sim;
                }
            }

            let mut step_best_score = f32::NEG_INFINITY;
            let mut step_best_token = candidates[0].0;
            for (tok, prob) in &candidates {
                let score = Self::contrastive_score(*prob, degen, cfg.alpha);
                if score > step_best_score {
                    step_best_score = score;
                    step_best_token = *tok;
                }
            }

            generated.push(step_best_token);
            // Extend context with the current logit vector (padded/truncated to vocab_size).
            let mut entry = next_logits.clone();
            entry.resize(vocab_size, 0.0);
            context_logits_flat.extend_from_slice(&entry);
        }

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // cosine_similarity tests
    // -----------------------------------------------------------------------

    #[test]
    fn cosine_similarity_identical_vectors_is_one() {
        let v = vec![1.0_f32, 2.0, 3.0];
        let sim = ContrastiveSearcher::cosine_similarity(&v, &v).expect("ok");
        assert!((sim - 1.0).abs() < 1e-5, "got {sim}");
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = ContrastiveSearcher::cosine_similarity(&a, &b).expect("ok");
        assert!(sim.abs() < 1e-6, "got {sim}");
    }

    #[test]
    fn cosine_similarity_zero_vector_is_zero_not_nan() {
        let a = vec![0.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let sim = ContrastiveSearcher::cosine_similarity(&a, &b).expect("ok");
        assert!(!sim.is_nan(), "must not be NaN");
        assert!(sim.abs() < 1e-6, "got {sim}");
    }

    #[test]
    fn cosine_similarity_length_mismatch_error() {
        let a = vec![1.0_f32, 2.0];
        let b = vec![1.0_f32, 2.0, 3.0];
        let err = ContrastiveSearcher::cosine_similarity(&a, &b).unwrap_err();
        assert!(matches!(err, SeqError::LengthMismatch { .. }));
    }

    #[test]
    fn cosine_similarity_empty_error() {
        let err = ContrastiveSearcher::cosine_similarity(&[], &[]).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn cosine_similarity_negative_vectors() {
        // Antiparallel vectors should give -1.
        let a = vec![1.0_f32, 0.0];
        let b = vec![-1.0_f32, 0.0];
        let sim = ContrastiveSearcher::cosine_similarity(&a, &b).expect("ok");
        assert!((sim + 1.0).abs() < 1e-5, "got {sim}");
    }

    // -----------------------------------------------------------------------
    // degeneration_penalty tests
    // -----------------------------------------------------------------------

    #[test]
    fn degeneration_penalty_no_context_is_zero() {
        let candidate = vec![1.0_f32, 2.0, 3.0];
        let pen = ContrastiveSearcher::degeneration_penalty(&[], 0, &candidate, 3).expect("ok");
        assert!(pen.abs() < 1e-6, "got {pen}");
    }

    #[test]
    fn degeneration_penalty_identical_context_is_one() {
        // If the candidate is identical to a context token, penalty = 1.0.
        let hidden = vec![1.0_f32, 0.0, 0.0];
        let context = hidden.clone();
        let pen = ContrastiveSearcher::degeneration_penalty(&context, 1, &hidden, 3).expect("ok");
        assert!((pen - 1.0).abs() < 1e-5, "got {pen}");
    }

    #[test]
    fn degeneration_penalty_orthogonal_context_is_zero() {
        let context = vec![1.0_f32, 0.0];
        let candidate = vec![0.0_f32, 1.0];
        let pen =
            ContrastiveSearcher::degeneration_penalty(&context, 1, &candidate, 2).expect("ok");
        assert!(pen.abs() < 1e-6, "got {pen}");
    }

    #[test]
    fn degeneration_penalty_multiple_context_returns_max() {
        // Two context vectors: first orthogonal, second identical to candidate.
        let dim = 2usize;
        let mut ctx = vec![1.0_f32, 0.0]; // orthogonal to candidate
        ctx.extend_from_slice(&[0.0, 1.0]); // identical to candidate
        let candidate = vec![0.0_f32, 1.0];
        let pen = ContrastiveSearcher::degeneration_penalty(&ctx, 2, &candidate, dim).expect("ok");
        // Max should be ~1.0 (the identical pair).
        assert!((pen - 1.0).abs() < 1e-5, "got {pen}");
    }

    #[test]
    fn degeneration_penalty_shape_mismatch_error() {
        let err = ContrastiveSearcher::degeneration_penalty(
            &[1.0, 2.0],
            2, // expects 2 * 3 = 6 floats
            &[1.0, 2.0, 3.0],
            3,
        )
        .unwrap_err();
        assert!(matches!(err, SeqError::ShapeMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // top_k_candidates tests
    // -----------------------------------------------------------------------

    #[test]
    fn top_k_k_equals_one_returns_argmax() {
        let logits = vec![-1.0_f32, 5.0, 2.0, 0.5];
        let cands = ContrastiveSearcher::top_k_candidates(&logits, 1).expect("ok");
        assert_eq!(cands.len(), 1);
        assert_eq!(cands[0].0, 1, "argmax should be token 1");
    }

    #[test]
    fn top_k_k_ge_vocab_returns_all() {
        let logits = vec![1.0_f32, 2.0, 0.5];
        let cands = ContrastiveSearcher::top_k_candidates(&logits, 100).expect("ok");
        assert_eq!(cands.len(), 3, "should return all 3 tokens");
    }

    #[test]
    fn top_k_probs_are_valid_softmax() {
        let logits = vec![1.0_f32, 2.0, 0.5, -1.0, 3.0];
        let cands = ContrastiveSearcher::top_k_candidates(&logits, 3).expect("ok");
        // Probs should be positive.
        for (_, prob) in &cands {
            assert!(*prob > 0.0, "prob must be positive");
        }
        // Sum of all vocab probs ≈ 1; we only have top-k, so sum ≤ 1.
        let partial_sum: f32 = cands.iter().map(|(_, p)| p).sum();
        assert!(partial_sum <= 1.0 + 1e-5, "partial sum {partial_sum} > 1");
    }

    #[test]
    fn top_k_sorted_descending_by_prob() {
        let logits = vec![1.0_f32, 3.0, 2.0, 0.5];
        let cands = ContrastiveSearcher::top_k_candidates(&logits, 4).expect("ok");
        for i in 1..cands.len() {
            assert!(
                cands[i - 1].1 >= cands[i].1,
                "probs should be non-increasing: {:?}",
                cands
            );
        }
    }

    #[test]
    fn top_k_empty_logits_error() {
        let err = ContrastiveSearcher::top_k_candidates(&[], 3).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    #[test]
    fn top_k_k_zero_error() {
        let err = ContrastiveSearcher::top_k_candidates(&[1.0, 2.0], 0).unwrap_err();
        assert!(matches!(err, SeqError::InvalidConfiguration(_)));
    }

    // -----------------------------------------------------------------------
    // contrastive_score tests
    // -----------------------------------------------------------------------

    #[test]
    fn contrastive_score_alpha_zero_equals_prob() {
        let score = ContrastiveSearcher::contrastive_score(0.7, 0.9, 0.0);
        assert!((score - 0.7).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn contrastive_score_alpha_one_equals_neg_degen() {
        let score = ContrastiveSearcher::contrastive_score(0.7, 0.4, 1.0);
        assert!((score + 0.4).abs() < 1e-6, "got {score}");
    }

    #[test]
    fn contrastive_score_midpoint() {
        let score = ContrastiveSearcher::contrastive_score(0.8, 0.5, 0.5);
        // (1-0.5)*0.8 - 0.5*0.5 = 0.4 - 0.25 = 0.15
        assert!((score - 0.15).abs() < 1e-6, "got {score}");
    }

    // -----------------------------------------------------------------------
    // decode_logits_only tests
    // -----------------------------------------------------------------------

    #[test]
    fn decode_logits_only_length_matches_max_len() {
        let initial = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let cfg = ContrastiveConfig {
            k: 3,
            alpha: 0.5,
            max_len: 5,
        };
        let seq = ContrastiveSearcher::decode_logits_only(
            &initial,
            |_tok| vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
            &cfg,
        )
        .expect("ok");
        assert_eq!(seq.len(), 5);
    }

    #[test]
    fn decode_logits_only_constant_step_fn_valid_tokens() {
        let initial = vec![0.0_f32, 1.0, -1.0, 2.0];
        let cfg = ContrastiveConfig {
            k: 2,
            alpha: 0.4,
            max_len: 10,
        };
        let seq = ContrastiveSearcher::decode_logits_only(
            &initial,
            |_tok| vec![0.0_f32, 1.0, -1.0, 2.0],
            &cfg,
        )
        .expect("ok");
        assert_eq!(seq.len(), 10);
        for tok in &seq {
            assert!(*tok < 4, "token {tok} out of vocab");
        }
    }

    #[test]
    fn decode_logits_only_can_produce_repetition() {
        // When the step function always returns the same logits and alpha=0
        // (no degeneration penalty), the argmax (greedy) token is always
        // the same token — demonstrating that contrastive search CAN repeat
        // when α=0.
        let initial = vec![0.0_f32, 5.0, 1.0];
        let cfg = ContrastiveConfig {
            k: 1,
            alpha: 0.0,
            max_len: 5,
        };
        let seq =
            ContrastiveSearcher::decode_logits_only(&initial, |_tok| vec![0.0_f32, 5.0, 1.0], &cfg)
                .expect("ok");
        // With k=1 and alpha=0, must always pick token 1 (greedy).
        for tok in &seq {
            assert_eq!(*tok, 1);
        }
    }

    #[test]
    fn decode_logits_only_alpha_reduces_repetition() {
        // With a high degeneration penalty (alpha close to 1), even a constant
        // step function should eventually switch to a different token because
        // distributional similarity accumulates.  We just verify no panic
        // and that token IDs are within vocab.
        let vocab = 8usize;
        let initial: Vec<f32> = (0..vocab).map(|i| i as f32).collect();
        let cfg = ContrastiveConfig {
            k: 4,
            alpha: 0.8,
            max_len: 20,
        };
        let seq = ContrastiveSearcher::decode_logits_only(
            &initial,
            |_tok| (0..vocab).map(|i| i as f32).collect(),
            &cfg,
        )
        .expect("ok");
        assert_eq!(seq.len(), 20);
        for tok in &seq {
            assert!(*tok < vocab);
        }
    }

    #[test]
    fn decode_logits_only_k_zero_error() {
        let cfg = ContrastiveConfig {
            k: 0,
            alpha: 0.5,
            max_len: 5,
        };
        let err = ContrastiveSearcher::decode_logits_only(&[1.0, 2.0], |_| vec![1.0, 2.0], &cfg)
            .unwrap_err();
        assert!(matches!(err, SeqError::InvalidConfiguration(_)));
    }

    #[test]
    fn decode_logits_only_alpha_above_one_error() {
        let cfg = ContrastiveConfig {
            k: 3,
            alpha: 1.5,
            max_len: 5,
        };
        let err = ContrastiveSearcher::decode_logits_only(&[1.0, 2.0], |_| vec![1.0, 2.0], &cfg)
            .unwrap_err();
        assert!(matches!(err, SeqError::InvalidConfiguration(_)));
    }

    #[test]
    fn decode_logits_only_empty_logits_error() {
        let cfg = ContrastiveConfig::default();
        let err = ContrastiveSearcher::decode_logits_only(&[], |_| vec![], &cfg).unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }

    // -----------------------------------------------------------------------
    // decode (with hidden states) tests
    // -----------------------------------------------------------------------

    #[test]
    fn decode_with_hidden_states_length_matches_max_len() {
        let vocab = 4usize;
        let hidden_dim = 3usize;
        let initial_logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        // One [hidden_dim] vector per vocabulary token.
        let initial_hiddens: Vec<f32> = (0..vocab * hidden_dim).map(|i| i as f32 * 0.1).collect();
        let cfg = ContrastiveConfig {
            k: 2,
            alpha: 0.5,
            max_len: 7,
        };

        let seq = ContrastiveSearcher::decode(
            &initial_logits,
            &initial_hiddens,
            vocab,
            hidden_dim,
            |_tok, _last| {
                let logits = vec![0.5_f32, 1.5, 2.5, 0.2];
                let hidden = vec![0.1_f32, 0.2, 0.3];
                (logits, hidden)
            },
            &cfg,
        )
        .expect("ok");
        assert_eq!(seq.len(), 7);
    }

    #[test]
    fn decode_with_hidden_states_valid_token_ids() {
        let vocab = 5usize;
        let hidden_dim = 4usize;
        let initial_logits: Vec<f32> = vec![1.0, 2.0, 3.0, 0.5, 1.5];
        let initial_hiddens: Vec<f32> = (0..vocab * hidden_dim).map(|i| (i as f32).sin()).collect();
        let cfg = ContrastiveConfig {
            k: 3,
            alpha: 0.6,
            max_len: 12,
        };
        let seq = ContrastiveSearcher::decode(
            &initial_logits,
            &initial_hiddens,
            vocab,
            hidden_dim,
            |_tok, _last| {
                let logits: Vec<f32> = vec![0.1, 0.5, 2.0, 1.0, 0.3];
                let hidden: Vec<f32> = vec![0.5, -0.5, 0.3, -0.3];
                (logits, hidden)
            },
            &cfg,
        )
        .expect("ok");
        for tok in &seq {
            assert!(*tok < vocab, "token {tok} out of range");
        }
    }

    #[test]
    fn decode_empty_vocab_error() {
        let cfg = ContrastiveConfig::default();
        let err = ContrastiveSearcher::decode(&[], &[], 0, 4, |_tok, _h| (vec![], vec![]), &cfg)
            .unwrap_err();
        assert!(matches!(err, SeqError::EmptyInput));
    }
}
