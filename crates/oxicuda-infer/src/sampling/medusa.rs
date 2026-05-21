//! # Medusa Speculative Decoding
//!
//! Implements the multi-head, tree-verified speculative-decoding scheme from:
//!
//! > Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with
//! > Multiple Decoding Heads" (arXiv 2401.10774, 2024).
//!
//! ## Concept
//!
//! Whereas single-draft speculative decoding (see
//! [`crate::sampling::speculative`]) uses a separate small *draft model* to
//! propose one linear chain of tokens, **Medusa** augments the base model with
//! several lightweight *decoding heads*.  Head `h` predicts the token at
//! relative position `h + 1` directly from the current hidden state.  Taking
//! the `top_k_per_head` candidates from each head and combining them across
//! heads produces a *candidate token tree* — many possible continuations of
//! length `n_heads` — which the base model verifies in a single forward pass.
//! The longest candidate prefix that the base model would itself have produced
//! is accepted, yielding several tokens per step.
//!
//! ## This implementation
//!
//! * [`MedusaDecoder::build_candidates`] forms the candidate paths.  For each
//!   head it selects the `top_k_per_head` highest-logit token indices, then
//!   combines heads via a Cartesian product, scoring each path by the **sum**
//!   of its per-head logits and keeping the `max_candidates` highest-scoring
//!   paths.  To avoid the `k^n_heads` combinatorial blow-up, the product is
//!   built incrementally with a beam that retains the best `max_candidates`
//!   partial paths after each head — which, because the joint score is additive
//!   and per-head choices are independent, yields exactly the global top
//!   `max_candidates` complete paths.
//! * [`MedusaDecoder::verify`] consults a base-model acceptance oracle
//!   `base_accept(path) -> usize` (how many leading tokens of `path` the base
//!   model accepts) and returns the longest accepted prefix, breaking ties in
//!   favour of the earlier candidate.
//! * [`MedusaDecoder::acceptance_rate`] reports `total_accepted /
//!   total_proposed` across all verification steps — the Medusa analogue of the
//!   single-draft [`crate::sampling::speculative::SpeculativeDecoder`] statistic.

use crate::error::{InferError, InferResult};
use std::cmp::Ordering;

// ─── MedusaConfig ──────────────────────────────────────────────────────────────

/// Configuration for the Medusa multi-head candidate tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MedusaConfig {
    /// Number of Medusa decoding heads (candidate path length).  Must be ≥ 1.
    pub n_heads: usize,
    /// Number of top tokens taken from each head.  Must be ≥ 1.
    pub top_k_per_head: usize,
    /// Maximum number of candidate paths kept after combination.  Must be ≥ 1.
    pub max_candidates: usize,
}

impl MedusaConfig {
    /// Convenience constructor.
    #[must_use]
    pub fn new(n_heads: usize, top_k_per_head: usize, max_candidates: usize) -> Self {
        Self {
            n_heads,
            top_k_per_head,
            max_candidates,
        }
    }
}

// ─── MedusaDecoder ──────────────────────────────────────────────────────────────

/// Stateful Medusa speculative decoder.
///
/// Holds the [`MedusaConfig`] and accumulates acceptance statistics across
/// verification steps.
#[derive(Debug, Clone)]
pub struct MedusaDecoder {
    /// Immutable configuration.
    cfg: MedusaConfig,
    /// Total candidate tokens examined across all `verify` calls.
    total_proposed: usize,
    /// Total tokens accepted across all `verify` calls.
    total_accepted: usize,
}

/// A partial or complete candidate path under construction, with its running
/// joint logit (sum of per-head logits).
#[derive(Debug, Clone)]
struct ScoredPath {
    /// Token indices chosen so far (one per head processed).
    tokens: Vec<usize>,
    /// Sum of the per-head logits of `tokens`.
    score: f32,
}

impl MedusaDecoder {
    /// Create a new decoder.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::InvalidConfig`] if `n_heads`, `top_k_per_head`, or
    /// `max_candidates` is zero.
    pub fn new(cfg: MedusaConfig) -> InferResult<Self> {
        if cfg.n_heads == 0 {
            return Err(InferError::InvalidConfig("medusa: n_heads must be ≥ 1"));
        }
        if cfg.top_k_per_head == 0 {
            return Err(InferError::InvalidConfig(
                "medusa: top_k_per_head must be ≥ 1",
            ));
        }
        if cfg.max_candidates == 0 {
            return Err(InferError::InvalidConfig(
                "medusa: max_candidates must be ≥ 1",
            ));
        }
        Ok(Self {
            cfg,
            total_proposed: 0,
            total_accepted: 0,
        })
    }

    /// Access the configuration.
    #[must_use]
    pub fn config(&self) -> &MedusaConfig {
        &self.cfg
    }

    /// Build candidate token paths from per-head logits.
    ///
    /// `head_logits` is `n_heads × vocab` in row-major order: row `h` holds the
    /// vocabulary logits of head `h`, which predicts the token at relative
    /// position `h + 1`.  Returns up to `max_candidates` paths, each of length
    /// `n_heads`, ordered by **descending** joint logit (sum of per-head
    /// logits); ties are broken lexicographically by token index so the result
    /// is deterministic.
    ///
    /// # Errors
    ///
    /// * [`InferError::InvalidConfig`]      — `vocab == 0`.
    /// * [`InferError::DimensionMismatch`]  — `head_logits.len() != n_heads * vocab`.
    /// * [`InferError::NanLogits`]          — any logit is NaN.
    pub fn build_candidates(
        &self,
        head_logits: &[f32],
        vocab: usize,
    ) -> InferResult<Vec<Vec<usize>>> {
        if vocab == 0 {
            return Err(InferError::InvalidConfig("medusa: vocab must be ≥ 1"));
        }
        let expected = self.cfg.n_heads * vocab;
        if head_logits.len() != expected {
            return Err(InferError::DimensionMismatch {
                expected,
                got: head_logits.len(),
            });
        }
        for &v in head_logits {
            if v.is_nan() {
                return Err(InferError::NanLogits);
            }
        }

        // Beam over partial paths; retain at most `max_candidates` partials
        // after each head so the final set is exactly the global top
        // `max_candidates` complete paths (additive, independent scores).
        let mut beam: Vec<ScoredPath> = vec![ScoredPath {
            tokens: Vec::with_capacity(self.cfg.n_heads),
            score: 0.0,
        }];

        for head in 0..self.cfg.n_heads {
            let row = &head_logits[head * vocab..(head + 1) * vocab];
            let top = Self::top_k_indices(row, self.cfg.top_k_per_head);

            let mut expanded: Vec<ScoredPath> = Vec::with_capacity(beam.len() * top.len());
            for partial in &beam {
                for &(tok, logit) in &top {
                    let mut tokens = partial.tokens.clone();
                    tokens.push(tok);
                    expanded.push(ScoredPath {
                        tokens,
                        score: partial.score + logit,
                    });
                }
            }
            Self::sort_paths(&mut expanded);
            expanded.truncate(self.cfg.max_candidates);
            beam = expanded;
        }

        Ok(beam.into_iter().map(|p| p.tokens).collect())
    }

    /// Verify candidate paths against a base-model acceptance oracle.
    ///
    /// `base_accept(path)` returns how many leading tokens of `path` the base
    /// model accepts (`0..=path.len()`).  The candidate with the maximum
    /// accepted-prefix length is selected (ties resolved in favour of the
    /// earlier candidate), and that prefix is returned.  Statistics are
    /// updated: `total_proposed` grows by the number of candidate paths
    /// examined and `total_accepted` by the accepted-prefix length.
    ///
    /// # Errors
    ///
    /// Returns [`InferError::EmptyBatch`] if `candidates` is empty.
    pub fn verify(
        &mut self,
        candidates: &[Vec<usize>],
        base_accept: &dyn Fn(&[usize]) -> usize,
    ) -> InferResult<Vec<usize>> {
        if candidates.is_empty() {
            return Err(InferError::EmptyBatch);
        }

        let mut best_len = 0_usize;
        let mut best_path: &[usize] = &candidates[0][..0];
        for candidate in candidates {
            // Clamp to the candidate length so a misbehaving oracle cannot make
            // us return more tokens than the candidate actually contains.
            let accepted = base_accept(candidate).min(candidate.len());
            if accepted > best_len {
                best_len = accepted;
                best_path = &candidate[..accepted];
            }
        }

        self.total_proposed += candidates.len();
        self.total_accepted += best_len;
        Ok(best_path.to_vec())
    }

    /// Empirical acceptance rate `total_accepted / total_proposed`.
    ///
    /// Returns `0.0` before any verification step.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_proposed == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_proposed as f64
        }
    }

    /// Total candidate paths examined so far.
    #[must_use]
    pub fn total_proposed(&self) -> usize {
        self.total_proposed
    }

    /// Total tokens accepted so far.
    #[must_use]
    pub fn total_accepted(&self) -> usize {
        self.total_accepted
    }

    // ─── helpers ────────────────────────────────────────────────────────────

    /// Return the `k` highest-logit `(index, logit)` pairs of `row`, sorted by
    /// descending logit then ascending index.  `k` is clamped to `row.len()`.
    fn top_k_indices(row: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut pairs: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        pairs.sort_unstable_by(|&(ia, la), &(ib, lb)| Self::cmp_desc_logit(la, ia, lb, ib));
        pairs.truncate(k.min(row.len()));
        pairs
    }

    /// Sort complete/partial paths by descending joint score, breaking ties by
    /// lexicographic token order for determinism.
    fn sort_paths(paths: &mut [ScoredPath]) {
        paths.sort_unstable_by(|a, b| {
            match b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal) {
                Ordering::Equal => a.tokens.cmp(&b.tokens),
                other => other,
            }
        });
    }

    /// Total order: higher logit first, then lower index first.
    fn cmp_desc_logit(la: f32, ia: usize, lb: f32, ib: usize) -> Ordering {
        match lb.partial_cmp(&la).unwrap_or(Ordering::Equal) {
            Ordering::Equal => ia.cmp(&ib),
            other => other,
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat `n_heads × vocab` logit buffer from per-head rows.
    fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
        rows.iter().flatten().copied().collect()
    }

    fn decoder(n_heads: usize, top_k: usize, max_c: usize) -> MedusaDecoder {
        MedusaDecoder::new(MedusaConfig::new(n_heads, top_k, max_c)).expect("valid medusa config")
    }

    #[test]
    fn candidates_count_capped() {
        // 2 heads, top-2 each → 4 cartesian paths, capped at max_candidates = 3.
        let d = decoder(2, 2, 3);
        let rows = vec![vec![1.0_f32, 2.0, 0.5, 0.1], vec![0.2_f32, 3.0, 1.0, 0.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 4)
            .expect("valid head logits");
        assert!(cands.len() <= 3, "must not exceed max_candidates");
    }

    #[test]
    fn each_path_has_n_heads_length() {
        let d = decoder(3, 2, 10);
        let rows = vec![
            vec![1.0_f32, 2.0, 0.5],
            vec![0.2_f32, 3.0, 1.0],
            vec![2.0_f32, 0.1, 0.4],
        ];
        let cands = d
            .build_candidates(&flatten(&rows), 3)
            .expect("valid head logits");
        for path in &cands {
            assert_eq!(path.len(), 3, "each path length must equal n_heads");
        }
    }

    #[test]
    fn only_top_k_tokens_appear() {
        // Head 0 top-2 = {1 (5.0), 2 (4.0)}; head 1 top-2 = {0 (9.0), 3 (8.0)}.
        let d = decoder(2, 2, 100);
        let rows = vec![vec![1.0_f32, 5.0, 4.0, 0.0], vec![9.0_f32, 1.0, 2.0, 8.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 4)
            .expect("valid head logits");
        for path in &cands {
            assert!(path[0] == 1 || path[0] == 2, "head 0 token must be top-2");
            assert!(path[1] == 0 || path[1] == 3, "head 1 token must be top-2");
        }
    }

    #[test]
    fn paths_ordered_by_descending_joint_logit() {
        let d = decoder(2, 2, 100);
        let rows = vec![vec![1.0_f32, 5.0, 4.0, 0.0], vec![9.0_f32, 1.0, 2.0, 8.0]];
        let flat = flatten(&rows);
        let cands = d.build_candidates(&flat, 4).expect("valid head logits");
        // Recompute joint scores and assert non-increasing order.
        let score = |path: &[usize]| -> f32 { flat[path[0]] + flat[4 + path[1]] };
        for w in cands.windows(2) {
            assert!(
                score(&w[0]) >= score(&w[1]),
                "paths must be ordered by descending joint logit"
            );
        }
        // Best path is the joint argmax: head0 token 1 (5.0) + head1 token 0 (9.0).
        assert_eq!(cands[0], vec![1, 0]);
    }

    #[test]
    fn verify_accept_all_returns_full_path() {
        let mut d = decoder(3, 2, 10);
        let rows = vec![
            vec![1.0_f32, 2.0, 0.5],
            vec![0.2_f32, 3.0, 1.0],
            vec![2.0_f32, 0.1, 0.4],
        ];
        let cands = d
            .build_candidates(&flatten(&rows), 3)
            .expect("valid head logits");
        let accept_all = |p: &[usize]| p.len();
        let out = d.verify(&cands, &accept_all).expect("non-empty candidates");
        assert_eq!(out.len(), 3, "accept-all should return a full n_heads path");
    }

    #[test]
    fn verify_accept_none_returns_empty() {
        let mut d = decoder(2, 2, 10);
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 0.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 2)
            .expect("valid head logits");
        let accept_none = |_: &[usize]| 0_usize;
        let out = d
            .verify(&cands, &accept_none)
            .expect("non-empty candidates");
        assert!(out.is_empty(), "accept-none should return an empty prefix");
    }

    #[test]
    fn verify_returns_longest_accepted_prefix() {
        let mut d = decoder(3, 2, 10);
        let rows = vec![
            vec![1.0_f32, 2.0, 0.5],
            vec![0.2_f32, 3.0, 1.0],
            vec![2.0_f32, 0.1, 0.4],
        ];
        let cands = d
            .build_candidates(&flatten(&rows), 3)
            .expect("valid head logits");
        // Oracle: accept a prefix length equal to the first token index, so the
        // candidate beginning with token 2 (if present) gives the longest.
        let oracle = |p: &[usize]| p.first().copied().unwrap_or(0).min(p.len());
        let out = d.verify(&cands, &oracle).expect("non-empty candidates");
        let best_first = cands
            .iter()
            .map(|p| p[0].min(p.len()))
            .max()
            .expect("candidates non-empty");
        assert_eq!(
            out.len(),
            best_first,
            "must return the longest accepted prefix"
        );
    }

    #[test]
    fn acceptance_rate_in_unit_interval() {
        let mut d = decoder(2, 2, 4);
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 0.5]];
        let cands = d
            .build_candidates(&flatten(&rows), 2)
            .expect("valid head logits");
        let oracle = |_: &[usize]| 1_usize;
        d.verify(&cands, &oracle).expect("non-empty candidates");
        let rate = d.acceptance_rate();
        assert!((0.0..=1.0).contains(&rate), "rate out of range: {rate}");
    }

    #[test]
    fn deterministic_build() {
        let d = decoder(3, 2, 5);
        let rows = vec![
            vec![1.0_f32, 2.0, 0.5, 0.3],
            vec![0.2_f32, 3.0, 1.0, 0.1],
            vec![2.0_f32, 0.1, 0.4, 0.9],
        ];
        let flat = flatten(&rows);
        let a = d.build_candidates(&flat, 4).expect("valid head logits");
        let b = d.build_candidates(&flat, 4).expect("valid head logits");
        assert_eq!(a, b, "build_candidates must be deterministic");
    }

    #[test]
    fn single_head_candidates_are_single_tokens() {
        let d = decoder(1, 3, 10);
        let rows = vec![vec![1.0_f32, 5.0, 4.0, 2.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 4)
            .expect("valid head logits");
        assert!(
            cands.iter().all(|p| p.len() == 1),
            "single head → length-1 paths"
        );
        // Top token (index 1, logit 5.0) should be first.
        assert_eq!(cands[0], vec![1]);
        assert!(
            cands.len() <= 3,
            "at most top_k_per_head distinct single tokens"
        );
    }

    #[test]
    fn max_candidates_one_keeps_only_best() {
        let d = decoder(2, 2, 1);
        let rows = vec![vec![1.0_f32, 5.0, 4.0], vec![9.0_f32, 1.0, 2.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 3)
            .expect("valid head logits");
        assert_eq!(cands.len(), 1, "max_candidates=1 keeps a single path");
        // Best joint = head0 token 1 (5.0) + head1 token 0 (9.0).
        assert_eq!(cands[0], vec![1, 0]);
    }

    #[test]
    fn err_n_heads_zero() {
        assert!(matches!(
            MedusaDecoder::new(MedusaConfig::new(0, 2, 4)),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_top_k_zero() {
        assert!(matches!(
            MedusaDecoder::new(MedusaConfig::new(2, 0, 4)),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_max_candidates_zero() {
        assert!(matches!(
            MedusaDecoder::new(MedusaConfig::new(2, 2, 0)),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_head_logits_wrong_length() {
        let d = decoder(2, 2, 4);
        // 2 heads × vocab 4 → expects 8 logits, give 6.
        let logits = vec![0.0_f32; 6];
        assert!(matches!(
            d.build_candidates(&logits, 4),
            Err(InferError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn err_vocab_zero() {
        let d = decoder(2, 2, 4);
        assert!(matches!(
            d.build_candidates(&[], 0),
            Err(InferError::InvalidConfig(_))
        ));
    }

    #[test]
    fn err_nan_logits() {
        let d = decoder(1, 2, 4);
        let logits = vec![1.0_f32, f32::NAN, 0.0];
        assert!(matches!(
            d.build_candidates(&logits, 3),
            Err(InferError::NanLogits)
        ));
    }

    #[test]
    fn top_k_larger_than_vocab_clamps() {
        // top_k_per_head = 5 but vocab = 2 → at most 2 tokens per head.
        let d = decoder(1, 5, 10);
        let rows = vec![vec![1.0_f32, 2.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 2)
            .expect("valid head logits");
        assert_eq!(cands.len(), 2, "clamped to vocab size");
    }

    #[test]
    fn acceptance_rate_zero_before_verify() {
        let d = decoder(2, 2, 4);
        assert_eq!(d.acceptance_rate(), 0.0);
        assert_eq!(d.total_proposed(), 0);
        assert_eq!(d.total_accepted(), 0);
    }

    #[test]
    fn two_heads_cartesian_product_size() {
        // Large max_candidates so nothing is pruned: product = top_k^n_heads.
        let d = decoder(2, 2, 1000);
        let rows = vec![vec![1.0_f32, 2.0, 0.5, 0.1], vec![0.2_f32, 3.0, 1.0, 0.0]];
        let cands = d
            .build_candidates(&flatten(&rows), 4)
            .expect("valid head logits");
        assert_eq!(cands.len(), 4, "2 heads × top-2 → 4 candidate paths");
    }

    #[test]
    fn verify_empty_candidates_errors() {
        let mut d = decoder(2, 2, 4);
        let accept_all = |p: &[usize]| p.len();
        assert!(matches!(
            d.verify(&[], &accept_all),
            Err(InferError::EmptyBatch)
        ));
    }

    #[test]
    fn verify_accumulates_statistics() {
        let mut d = decoder(2, 2, 4);
        let rows = vec![vec![1.0_f32, 2.0], vec![3.0_f32, 0.5]];
        let cands = d
            .build_candidates(&flatten(&rows), 2)
            .expect("valid head logits");
        let n = cands.len();
        let oracle = |_: &[usize]| 2_usize;
        d.verify(&cands, &oracle).expect("non-empty candidates");
        assert_eq!(d.total_proposed(), n, "proposed grows by candidate count");
        assert_eq!(d.total_accepted(), 2, "accepted grows by prefix length");
    }

    #[test]
    fn verify_ties_prefer_first_candidate() {
        // Two candidates accept the same length; the first must win. We detect
        // this by giving them distinct token contents.
        let mut d = decoder(2, 2, 10);
        let cands = vec![vec![7_usize, 8], vec![1_usize, 2]];
        let oracle = |_: &[usize]| 1_usize; // both accept length 1
        let out = d.verify(&cands, &oracle).expect("non-empty candidates");
        assert_eq!(out, vec![7], "tie should resolve to the first candidate");
    }
}
