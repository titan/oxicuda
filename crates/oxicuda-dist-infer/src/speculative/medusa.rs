//! Medusa multi-head speculative decoding (Cai et al. 2024).
//!
//! Medusa augments a frozen base LLM with several lightweight *decoding heads*,
//! each predicting a token at a future position. Head `i` (0-indexed) predicts
//! the token at position `t + 1 + i` directly from the *current* hidden state
//! `h_t`, instead of running the full transformer autoregressively. The base
//! model then verifies the speculated continuations in a single forward pass.
//!
//! ```text
//!                         ┌── head[0] ──► token at t+1  (top-k candidates)
//!  hidden state h_t ──────┼── head[1] ──► token at t+2  (top-k candidates)
//!                         ├── head[2] ──► token at t+3  (top-k candidates)
//!                         └── ...
//! ```
//!
//! Each head is an affine projection `logits = W_i · h_t + b_i` over the
//! vocabulary. The top-`k` tokens of every head form per-position candidate
//! sets; their Cartesian product (capped) yields *tree-structured* candidate
//! continuations that the base model checks in parallel.
//!
//! # Reference
//! - Cai, Li, Geng, Peng, Lee, Chen, Dao (2024) "Medusa: Simple LLM Inference
//!   Acceleration Framework with Multiple Decoding Heads." arXiv:2401.10774.

use crate::error::{DistInferError, DistInferResult};

/// Maximum number of tree-structured candidate continuations produced by
/// [`MedusaHeads::build_tree_candidates`]. Real Medusa uses a learned sparse
/// tree; this reference implementation caps the dense Cartesian product so the
/// candidate set stays bounded regardless of `top_k`/`n_heads`.
const MAX_TREE_CANDIDATES: usize = 64;

// ─── DistInferRng ─────────────────────────────────────────────────────────────

/// Minimal reproducible 64-bit linear-congruential generator.
///
/// Uses Knuth's MMIX constants (multiplier `6364136223846793005`, increment
/// `1442695040888963407`). Self-contained so the distributed-inference crate
/// needs no external RNG dependency for deterministic weight initialisation.
#[derive(Debug, Clone)]
pub struct DistInferRng {
    state: u64,
}

impl DistInferRng {
    /// Create a generator from a 64-bit seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    /// Advance the state and return the next 64-bit value.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Sample a float uniformly in `[0, 1)` using the top 24 bits.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    /// Sample an approximately standard-normal value via the central limit
    /// theorem (sum of 12 uniforms minus 6 → mean 0, variance 1). Cheap and
    /// dependency-free; adequate for deterministic weight initialisation.
    #[inline]
    pub fn next_gaussian(&mut self) -> f32 {
        let sum: f32 = (0..12).map(|_| self.next_f32()).sum();
        sum - 6.0
    }
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for a bank of Medusa decoding heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MedusaConfig {
    /// Number of decoding heads (each predicts one future position).
    pub n_heads: usize,
    /// Vocabulary size (output dimension of each head).
    pub vocab_size: usize,
    /// Hidden-state dimension (input dimension of each head).
    pub d_model: usize,
    /// Number of top candidate tokens retained per head.
    pub top_k: usize,
}

// ─── MedusaHeads ────────────────────────────────────────────────────────────

/// A bank of Medusa decoding heads operating on a shared hidden state.
///
/// Weights are stored row-major per head: `head_w[i]` has length
/// `vocab_size * d_model` and is laid out as `[vocab_size][d_model]`.
#[derive(Debug, Clone)]
pub struct MedusaHeads {
    /// Per-head weight matrices, `[n_heads][vocab_size × d_model]`.
    head_w: Vec<Vec<f32>>,
    /// Per-head bias vectors, `[n_heads][vocab_size]`.
    head_b: Vec<Vec<f32>>,
    /// Static configuration.
    config: MedusaConfig,
}

impl MedusaHeads {
    /// Construct a head bank with randomly initialised weights.
    ///
    /// Weights are drawn from a small Gaussian (scaled by `1/sqrt(d_model)`,
    /// Glorot-style) so per-head logits stay well-conditioned. Biases start at
    /// zero, matching the usual Medusa head initialisation.
    ///
    /// # Errors
    ///
    /// * [`DistInferError::InvalidWorldSize`] if `n_heads == 0`.
    /// * [`DistInferError::DimensionMismatch`] if `vocab_size == 0`,
    ///   `d_model == 0`, or `top_k > vocab_size`.
    pub fn new(config: MedusaConfig, rng: &mut DistInferRng) -> DistInferResult<Self> {
        if config.n_heads == 0 {
            return Err(DistInferError::InvalidWorldSize {
                world_size: 0,
                reason: "MedusaConfig.n_heads must be ≥ 1",
            });
        }
        if config.vocab_size == 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if config.d_model == 0 {
            return Err(DistInferError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if config.top_k == 0 || config.top_k > config.vocab_size {
            return Err(DistInferError::DimensionMismatch {
                expected: config.vocab_size,
                got: config.top_k,
            });
        }

        let scale = 1.0_f32 / (config.d_model as f32).sqrt();
        let mut head_w = Vec::with_capacity(config.n_heads);
        let mut head_b = Vec::with_capacity(config.n_heads);
        for _ in 0..config.n_heads {
            let mut w = Vec::with_capacity(config.vocab_size * config.d_model);
            for _ in 0..config.vocab_size * config.d_model {
                w.push(rng.next_gaussian() * scale);
            }
            head_w.push(w);
            head_b.push(vec![0.0_f32; config.vocab_size]);
        }

        Ok(Self {
            head_w,
            head_b,
            config,
        })
    }

    /// Number of decoding heads.
    #[must_use]
    pub fn n_heads(&self) -> usize {
        self.config.n_heads
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    /// Hidden dimension expected for the input `hidden` slice.
    #[must_use]
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Compute logits `W_i · h + b_i` for head `i` over the full vocabulary.
    fn head_logits(&self, head: usize, hidden: &[f32]) -> Vec<f32> {
        let d = self.config.d_model;
        let v = self.config.vocab_size;
        let w = &self.head_w[head];
        let b = &self.head_b[head];
        let mut logits = vec![0.0_f32; v];
        for (row, logit) in logits.iter_mut().enumerate() {
            let base = row * d;
            let mut acc = 0.0_f32;
            for (j, &h) in hidden.iter().enumerate() {
                acc += w[base + j] * h;
            }
            *logit = acc + b[row];
        }
        logits
    }

    /// Return the indices of the `top_k` largest logits, descending by value.
    ///
    /// Ties break toward the lower token id so the output is deterministic.
    fn top_k_indices(&self, logits: &[f32]) -> Vec<usize> {
        let k = self.config.top_k;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        // Partial-sort would suffice, but vocabulary sizes here are small;
        // a full stable sort keeps tie-breaking explicit and clear.
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(k);
        idx
    }

    /// Predict candidate tokens for each head.
    ///
    /// For head `i`, computes `logits = W_i · hidden + b_i` and returns its
    /// `top_k` candidate token ids. The result has shape `[n_heads][top_k]`,
    /// describing candidate tokens for positions `t+1 .. t+n_heads`.
    ///
    /// # Errors
    ///
    /// [`DistInferError::DimensionMismatch`] if `hidden.len() != d_model`.
    pub fn predict_heads(&self, hidden: &[f32]) -> DistInferResult<Vec<Vec<usize>>> {
        if hidden.len() != self.config.d_model {
            return Err(DistInferError::DimensionMismatch {
                expected: self.config.d_model,
                got: hidden.len(),
            });
        }
        let mut out = Vec::with_capacity(self.config.n_heads);
        for head in 0..self.config.n_heads {
            let logits = self.head_logits(head, hidden);
            out.push(self.top_k_indices(&logits));
        }
        Ok(out)
    }

    /// Build tree-structured candidate continuations.
    ///
    /// Forms the Cartesian product of the per-head top-`k` candidate sets — one
    /// token per head, in head order — yielding candidate continuations of
    /// length `n_heads`. The total number of continuations is capped at
    /// `MAX_TREE_CANDIDATES` (the highest-ranked combinations are kept, since
    /// head 0 varies slowest in the enumeration order).
    ///
    /// Each returned inner vector has length `n_heads`.
    ///
    /// # Errors
    ///
    /// [`DistInferError::DimensionMismatch`] if `hidden.len() != d_model`.
    pub fn build_tree_candidates(&self, hidden: &[f32]) -> DistInferResult<Vec<Vec<usize>>> {
        let per_head = self.predict_heads(hidden)?;

        // Enumerate the Cartesian product in row-major (head-0 = most
        // significant) order, stopping once the cap is reached.
        let mut candidates: Vec<Vec<usize>> = vec![Vec::with_capacity(self.config.n_heads)];
        for head_cands in &per_head {
            let mut next = Vec::with_capacity(candidates.len() * head_cands.len());
            for prefix in &candidates {
                for &tok in head_cands {
                    if next.len() >= MAX_TREE_CANDIDATES {
                        break;
                    }
                    let mut cont = prefix.clone();
                    cont.push(tok);
                    next.push(cont);
                }
                if next.len() >= MAX_TREE_CANDIDATES {
                    break;
                }
            }
            candidates = next;
        }
        Ok(candidates)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n_heads: usize, vocab: usize, d: usize, k: usize) -> MedusaConfig {
        MedusaConfig {
            n_heads,
            vocab_size: vocab,
            d_model: d,
            top_k: k,
        }
    }

    fn make(n_heads: usize, vocab: usize, d: usize, k: usize, seed: u64) -> MedusaHeads {
        let mut rng = DistInferRng::new(seed);
        MedusaHeads::new(cfg(n_heads, vocab, d, k), &mut rng).expect("valid config")
    }

    #[test]
    fn predict_heads_shape() {
        let heads = make(3, 16, 8, 4, 1);
        let hidden = vec![0.5_f32; 8];
        let preds = heads.predict_heads(&hidden).expect("predict");
        assert_eq!(preds.len(), 3, "one candidate set per head");
        for row in &preds {
            assert_eq!(row.len(), 4, "top_k candidates per head");
        }
    }

    #[test]
    fn predict_heads_in_range() {
        let heads = make(4, 32, 8, 5, 2);
        let hidden: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let preds = heads.predict_heads(&hidden).expect("predict");
        for row in &preds {
            for &tok in row {
                assert!(tok < 32, "token {tok} out of vocab range");
            }
        }
    }

    #[test]
    fn top_k_respected() {
        // Each head returns exactly top_k distinct tokens.
        let heads = make(2, 20, 6, 7, 3);
        let hidden = vec![1.0_f32; 6];
        let preds = heads.predict_heads(&hidden).expect("predict");
        for row in &preds {
            assert_eq!(row.len(), 7);
            let mut sorted = row.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 7, "top_k tokens must be distinct");
        }
    }

    #[test]
    fn tree_candidates_nonempty() {
        let heads = make(3, 16, 8, 2, 4);
        let hidden = vec![0.3_f32; 8];
        let cands = heads.build_tree_candidates(&hidden).expect("tree");
        assert!(!cands.is_empty(), "must produce ≥ 1 continuation");
        for c in &cands {
            assert_eq!(c.len(), 3, "continuation spans all heads");
        }
    }

    #[test]
    fn different_hidden_different_preds() {
        let heads = make(3, 64, 8, 3, 5);
        let h1 = vec![1.0_f32; 8];
        let h2: Vec<f32> = (0..8).map(|i| -(i as f32)).collect();
        let p1 = heads.predict_heads(&h1).expect("p1");
        let p2 = heads.predict_heads(&h2).expect("p2");
        assert_ne!(p1, p2, "distinct hidden states should yield distinct preds");
    }

    #[test]
    fn n_heads_0_error() {
        let mut rng = DistInferRng::new(6);
        let err = MedusaHeads::new(cfg(0, 16, 8, 4), &mut rng);
        assert!(matches!(err, Err(DistInferError::InvalidWorldSize { .. })));
    }

    #[test]
    fn vocab_size_0_error() {
        let mut rng = DistInferRng::new(7);
        let err = MedusaHeads::new(cfg(3, 0, 8, 1), &mut rng);
        assert!(matches!(err, Err(DistInferError::DimensionMismatch { .. })));
    }

    #[test]
    fn top_k_gt_vocab_error() {
        let mut rng = DistInferRng::new(8);
        let err = MedusaHeads::new(cfg(3, 4, 8, 5), &mut rng);
        assert!(matches!(err, Err(DistInferError::DimensionMismatch { .. })));
    }

    #[test]
    fn hidden_dim_mismatch_error() {
        let heads = make(2, 16, 8, 3, 9);
        let hidden = vec![0.0_f32; 7]; // wrong: d_model is 8
        let err = heads.predict_heads(&hidden);
        assert!(matches!(err, Err(DistInferError::DimensionMismatch { .. })));
    }

    #[test]
    fn candidates_in_range() {
        let heads = make(3, 12, 8, 3, 10);
        let hidden = vec![0.2_f32; 8];
        let cands = heads.build_tree_candidates(&hidden).expect("tree");
        for c in &cands {
            for &tok in c {
                assert!(tok < 12, "tree token {tok} out of range");
            }
        }
    }

    #[test]
    fn tree_candidates_capped() {
        // 4 heads × top_k 4 = 256 dense combos, must be capped.
        let heads = make(4, 64, 8, 4, 11);
        let hidden = vec![0.5_f32; 8];
        let cands = heads.build_tree_candidates(&hidden).expect("tree");
        assert!(cands.len() <= MAX_TREE_CANDIDATES);
        assert!(!cands.is_empty());
    }

    #[test]
    fn n_heads_accessor() {
        let heads = make(5, 16, 8, 2, 12);
        assert_eq!(heads.n_heads(), 5);
        assert_eq!(heads.vocab_size(), 16);
        assert_eq!(heads.d_model(), 8);
    }

    #[test]
    fn d_model_0_error() {
        let mut rng = DistInferRng::new(13);
        let err = MedusaHeads::new(cfg(2, 16, 0, 1), &mut rng);
        assert!(matches!(err, Err(DistInferError::DimensionMismatch { .. })));
    }

    #[test]
    fn rng_reproducible() {
        let mut a = DistInferRng::new(42);
        let mut b = DistInferRng::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
