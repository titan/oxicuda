//! # Contrastive Search decoding
//!
//! Su et al. (2022), "A Contrastive Framework for Neural Text Generation"
//! (NeurIPS). <https://arxiv.org/abs/2202.06417>
//!
//! Contrastive search picks the next token by balancing two objectives over the
//! top-`k` candidates of the model's probability distribution:
//!
//! 1. **Model confidence** — prefer high-probability tokens `p(v | x_{<t})`.
//! 2. **Degeneration penalty** — discourage tokens whose hidden representation
//!    is too similar to any *previously generated* token, measured by the
//!    maximum cosine similarity to the context representations.
//!
//! The selected token maximises
//!
//! ```text
//! score(v) = (1 − α)·p(v | x_{<t})  −  α·max_{j<t} cos( h_v , h_{x_j} )
//! ```
//!
//! where `α ∈ [0, 1]` is the *penalty_alpha* and `h_v` is the candidate token's
//! representation. With `α = 0` it reduces to greedy decoding.
//!
//! This module is representation-agnostic: the caller supplies the candidate
//! token representations and the accumulated context representations as flat
//! `&[f32]` slices (row-major `[n × hidden]`). The cosine similarity is computed
//! with an ε-guard for zero-norm vectors.

use crate::error::{InferError, InferResult};
use crate::sampling::softmax;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for contrastive search.
#[derive(Debug, Clone, Copy)]
pub struct ContrastiveSearchConfig {
    /// Number of candidate tokens `k` considered each step (must be ≥ 1).
    pub top_k: usize,
    /// Degeneration penalty weight `α ∈ [0, 1]`.
    pub penalty_alpha: f32,
}

impl Default for ContrastiveSearchConfig {
    fn default() -> Self {
        Self {
            top_k: 4,
            penalty_alpha: 0.6,
        }
    }
}

// ─── Cosine similarity ───────────────────────────────────────────────────────

/// Cosine similarity between two equal-length vectors, ε-guarded.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-8 { 0.0 } else { dot / denom }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Select the next token via contrastive search.
///
/// # Arguments
/// * `logits`        — `[vocab]` raw next-token logits.
/// * `cand_reprs`    — `[vocab × hidden]` representations for *each* vocab token
///   (row-major). Only the rows of the chosen top-k candidates are read.
/// * `context_reprs` — `[ctx_len × hidden]` representations of previously
///   generated tokens (row-major). May be empty for the first step.
/// * `hidden`        — representation dimensionality.
/// * `cfg`           — contrastive-search configuration.
///
/// # Returns
/// The selected token id (index into `logits`).
///
/// # Errors
/// * [`InferError::InvalidConfig`]   — `top_k == 0`, `penalty_alpha` outside
///   `[0, 1]`, or `hidden == 0`.
/// * [`InferError::DimensionMismatch`] — `cand_reprs`/`context_reprs` not a
///   multiple of `hidden`, or `cand_reprs` shorter than `vocab × hidden`.
/// * [`InferError::SamplingError`]   — empty `logits`.
/// * [`InferError::NanLogits`]       — a NaN candidate score.
pub fn contrastive_search_select(
    logits: &[f32],
    cand_reprs: &[f32],
    context_reprs: &[f32],
    hidden: usize,
    cfg: ContrastiveSearchConfig,
) -> InferResult<usize> {
    if cfg.top_k == 0 {
        return Err(InferError::InvalidConfig("top_k must be >= 1"));
    }
    if !(0.0..=1.0).contains(&cfg.penalty_alpha) {
        return Err(InferError::InvalidConfig("penalty_alpha must be in [0, 1]"));
    }
    if hidden == 0 {
        return Err(InferError::InvalidConfig("hidden must be >= 1"));
    }
    let vocab = logits.len();
    if vocab == 0 {
        return Err(InferError::SamplingError(
            "logits slice is empty".to_owned(),
        ));
    }
    if cand_reprs.len() != vocab * hidden {
        return Err(InferError::DimensionMismatch {
            expected: vocab * hidden,
            got: cand_reprs.len(),
        });
    }
    if context_reprs.len() % hidden != 0 {
        return Err(InferError::DimensionMismatch {
            expected: hidden,
            got: context_reprs.len(),
        });
    }

    // Probabilities for the model-confidence term.
    let probs = softmax(logits);

    // Identify the top-k candidate token ids by logit.
    let k = cfg.top_k.min(vocab);
    let mut idx: Vec<usize> = (0..vocab).collect();
    idx.sort_unstable_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let candidates = &idx[..k];

    let alpha = cfg.penalty_alpha;
    let mut best_token = candidates[0];
    let mut best_score = f32::NEG_INFINITY;

    for &tok in candidates {
        let repr = &cand_reprs[tok * hidden..(tok + 1) * hidden];
        // Degeneration penalty: max cosine similarity to any context token.
        let mut max_sim = 0.0_f32;
        for ctx in context_reprs.chunks_exact(hidden) {
            let sim = cosine_similarity(repr, ctx);
            if sim > max_sim {
                max_sim = sim;
            }
        }
        let score = (1.0 - alpha) * probs[tok] - alpha * max_sim;
        if score.is_nan() {
            return Err(InferError::NanLogits);
        }
        if score > best_score {
            best_score = score;
            best_token = tok;
        }
    }

    Ok(best_token)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> ContrastiveSearchConfig {
        ContrastiveSearchConfig::default()
    }

    /// Build per-token one-hot-ish representations: token i → unit vector e_i
    /// (padded into `hidden` dims). `vocab <= hidden` for simplicity.
    fn onehot_reprs(vocab: usize, hidden: usize) -> Vec<f32> {
        let mut r = vec![0.0_f32; vocab * hidden];
        for i in 0..vocab {
            r[i * hidden + (i % hidden)] = 1.0;
        }
        r
    }

    #[test]
    fn alpha_zero_is_greedy() {
        // With α=0 the penalty vanishes ⇒ pick the argmax logit.
        let vocab = 5;
        let hidden = 5;
        let logits = vec![0.1_f32, 5.0, 0.3, 0.2, 0.0];
        let cand = onehot_reprs(vocab, hidden);
        let cfg = ContrastiveSearchConfig {
            top_k: 5,
            penalty_alpha: 0.0,
        };
        let tok = contrastive_search_select(&logits, &cand, &[], hidden, cfg).expect("ok");
        assert_eq!(tok, 1, "α=0 should pick argmax logit");
    }

    #[test]
    fn empty_context_picks_highest_prob_candidate() {
        // No context ⇒ penalty 0 ⇒ highest-prob among top-k.
        let vocab = 4;
        let hidden = 4;
        let logits = vec![1.0_f32, 3.0, 2.0, 0.5];
        let cand = onehot_reprs(vocab, hidden);
        let tok =
            contrastive_search_select(&logits, &cand, &[], hidden, default_cfg()).expect("ok");
        assert_eq!(tok, 1, "should pick highest-prob token, got {tok}");
    }

    #[test]
    fn penalty_avoids_repetition() {
        // Token 1 has the highest logit, but its representation equals the
        // context token's representation (cosine sim = 1), so a strong penalty
        // should steer selection toward a different candidate.
        let vocab = 4;
        let hidden = 4;
        let logits = vec![2.0_f32, 2.1, 2.0, 0.0];
        let cand = onehot_reprs(vocab, hidden);
        // Context is exactly token 1's representation.
        let ctx = cand[hidden..2 * hidden].to_vec();
        let cfg = ContrastiveSearchConfig {
            top_k: 3,
            penalty_alpha: 0.9,
        };
        let tok = contrastive_search_select(&logits, &cand, &ctx, hidden, cfg).expect("ok");
        assert_ne!(tok, 1, "high penalty should avoid the repeated token");
    }

    #[test]
    fn returns_valid_token_index() {
        let vocab = 8;
        let hidden = 8;
        let logits: Vec<f32> = (0..vocab).map(|i| i as f32 * 0.3).collect();
        let cand = onehot_reprs(vocab, hidden);
        let ctx = onehot_reprs(2, hidden);
        let tok =
            contrastive_search_select(&logits, &cand, &ctx, hidden, default_cfg()).expect("ok");
        assert!(tok < vocab, "token {tok} out of range");
    }

    #[test]
    fn top_k_one_picks_argmax() {
        let vocab = 6;
        let hidden = 6;
        let logits = vec![0.0_f32, 0.0, 9.0, 0.0, 0.0, 0.0];
        let cand = onehot_reprs(vocab, hidden);
        let cfg = ContrastiveSearchConfig {
            top_k: 1,
            penalty_alpha: 0.6,
        };
        // Even with a context, top_k=1 forces the single argmax candidate.
        let ctx = cand[2 * hidden..3 * hidden].to_vec();
        let tok = contrastive_search_select(&logits, &cand, &ctx, hidden, cfg).expect("ok");
        assert_eq!(tok, 2);
    }

    #[test]
    fn cosine_similarity_identical_is_one() {
        let a = vec![1.0_f32, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_orthogonal_is_zero() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_zero_vector_guarded() {
        let a = vec![0.0_f32, 0.0];
        let b = vec![1.0_f32, 1.0];
        // No NaN; ε-guard returns 0.
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn higher_alpha_more_penalty_sensitive() {
        // Construct a case where increasing α flips the selection.
        let vocab = 3;
        let hidden = 3;
        // token 0 has a much higher logit (and prob) but matches the context;
        // token 1 is distinct. A small α keeps token 0; a large α flips to 1.
        let logits = vec![3.0_f32, 0.0, -10.0];
        let cand = onehot_reprs(vocab, hidden);
        let ctx = cand[0..hidden].to_vec(); // context == token 0 repr
        let low = ContrastiveSearchConfig {
            top_k: 2,
            penalty_alpha: 0.1,
        };
        let high = ContrastiveSearchConfig {
            top_k: 2,
            penalty_alpha: 0.95,
        };
        let tok_low = contrastive_search_select(&logits, &cand, &ctx, hidden, low).expect("ok");
        let tok_high = contrastive_search_select(&logits, &cand, &ctx, hidden, high).expect("ok");
        assert_eq!(tok_low, 0, "low α keeps the high-prob repeated token");
        assert_eq!(tok_high, 1, "high α avoids the repeated token");
    }

    #[test]
    fn multiple_context_tokens_use_max_similarity() {
        // The penalty uses the *max* similarity over all context tokens.
        let vocab = 3;
        let hidden = 3;
        let logits = vec![1.0_f32, 1.0, 1.0];
        let cand = onehot_reprs(vocab, hidden);
        // Context contains token 0 and token 2 representations.
        let mut ctx = cand[0..hidden].to_vec();
        ctx.extend_from_slice(&cand[2 * hidden..3 * hidden]);
        let cfg = ContrastiveSearchConfig {
            top_k: 3,
            penalty_alpha: 0.8,
        };
        let tok = contrastive_search_select(&logits, &cand, &ctx, hidden, cfg).expect("ok");
        // Tokens 0 and 2 are penalised (sim=1); token 1 is free ⇒ selected.
        assert_eq!(tok, 1);
    }

    #[test]
    fn err_top_k_zero() {
        assert!(matches!(
            contrastive_search_select(
                &[1.0],
                &[1.0],
                &[],
                1,
                ContrastiveSearchConfig {
                    top_k: 0,
                    penalty_alpha: 0.5,
                },
            ),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_alpha_out_of_range() {
        assert!(matches!(
            contrastive_search_select(
                &[1.0],
                &[1.0],
                &[],
                1,
                ContrastiveSearchConfig {
                    top_k: 1,
                    penalty_alpha: 1.5,
                },
            ),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_empty_logits() {
        assert!(matches!(
            contrastive_search_select(&[], &[], &[], 4, default_cfg()),
            Err(InferError::SamplingError(_))
        ));
    }

    #[test]
    fn err_cand_repr_dim_mismatch() {
        // logits has 4 tokens, hidden=4 ⇒ need 16 cand values, give 8.
        assert!(matches!(
            contrastive_search_select(&[1.0, 2.0, 3.0, 4.0], &[0.0; 8], &[], 4, default_cfg()),
            Err(InferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_context_dim_mismatch() {
        let vocab = 4;
        let hidden = 4;
        let cand = onehot_reprs(vocab, hidden);
        // Context length not a multiple of hidden.
        assert!(matches!(
            contrastive_search_select(
                &[1.0, 2.0, 3.0, 4.0],
                &cand,
                &[1.0, 2.0, 3.0],
                hidden,
                default_cfg()
            ),
            Err(InferError::DimensionMismatch { .. })
        ));
    }
}
