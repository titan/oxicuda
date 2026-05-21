//! Prompt Pool / L2P — Learning to Prompt for Continual Learning.
//!
//! Reference: Wang Z, Zhang Z, Lee C-Y, Zhang H, Sun R, Ren X, Su G, Perot V,
//! Dy J, Pfister T (2022) "Learning to Prompt for Continual Learning", *CVPR 2022*:
//! 139–149. <https://arxiv.org/abs/2112.08654>
//!
//! ## Design
//!
//! L2P maintains a pool of `M` learnable prompts, each paired with a learnable key.
//! Given an input query feature `q ∈ ℝ^{key_dim}` (e.g. the `[CLS]` embedding of a
//! frozen backbone), the model:
//!
//! 1. Scores every pool key against the query by cosine similarity.
//! 2. Selects the top-`N` keys (descending score, lowest index breaks ties).
//! 3. Prepends the corresponding `N` prompts (each `prompt_len` tokens of width
//!    `embed_dim`) to the input sequence, in selection order.
//!
//! A key-matching loss `(1/N) Σ_{m∈selected} (1 − cosine(q, key_m))` pulls the
//! selected keys toward the query, encouraging instance-conditioned prompt routing.
//!
//! The keys and prompts are the only learnable parameters; the backbone stays frozen.

use crate::error::{PeftError, PeftResult};
use crate::handle::LcgRng;

/// Small constant guarding the cosine-similarity denominator against division by zero.
const COSINE_EPS: f32 = 1e-8;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for a [`PromptPool`].
#[derive(Debug, Clone)]
pub struct PromptPoolConfig {
    /// Pool size `M`: number of (key, prompt) pairs in the pool.
    pub pool_size: usize,
    /// Number of prompts selected per query `N` (`1 ≤ top_n ≤ pool_size`).
    pub top_n: usize,
    /// Number of prompt tokens per pool prompt.
    pub prompt_len: usize,
    /// Embedding dimension of each prompt token.
    pub embed_dim: usize,
    /// Dimension of the key vectors and the query feature.
    pub key_dim: usize,
}

// ---------------------------------------------------------------------------
// Prompt pool
// ---------------------------------------------------------------------------

/// A pool of `M` learnable (key, prompt) pairs for L2P-style prompt selection.
///
/// `keys` is stored row-major as `pool_size × key_dim`; `prompts` is stored
/// row-major as `pool_size × (prompt_len · embed_dim)`.
#[derive(Debug, Clone)]
pub struct PromptPool {
    /// Pool keys, flat row-major shape `pool_size × key_dim`.
    pub(crate) keys: Vec<f32>,
    /// Pool prompts, flat row-major shape `pool_size × (prompt_len · embed_dim)`.
    pub(crate) prompts: Vec<f32>,
    /// Pool configuration.
    pub cfg: PromptPoolConfig,
}

impl PromptPool {
    /// Construct a new prompt pool with random initialization.
    ///
    /// - Keys are sampled from N(0, 0.02) (small so cosine routing starts roughly
    ///   isotropic, mirroring the prompt-tuning init style in this crate).
    /// - Prompts are sampled from N(0, 0.02).
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] if any constraint is violated:
    /// `pool_size ≥ 1`, `1 ≤ top_n ≤ pool_size`, `prompt_len ≥ 1`,
    /// `embed_dim ≥ 1`, `key_dim ≥ 1`.
    pub fn new(cfg: PromptPoolConfig, rng: &mut LcgRng) -> PeftResult<Self> {
        if cfg.pool_size == 0 {
            return Err(PeftError::Internal {
                msg: "pool_size must be >= 1".to_string(),
            });
        }
        if cfg.top_n == 0 {
            return Err(PeftError::Internal {
                msg: "top_n must be >= 1".to_string(),
            });
        }
        if cfg.top_n > cfg.pool_size {
            return Err(PeftError::Internal {
                msg: format!(
                    "top_n ({}) must not exceed pool_size ({})",
                    cfg.top_n, cfg.pool_size
                ),
            });
        }
        if cfg.prompt_len == 0 {
            return Err(PeftError::Internal {
                msg: "prompt_len must be >= 1".to_string(),
            });
        }
        if cfg.embed_dim == 0 {
            return Err(PeftError::Internal {
                msg: "embed_dim must be >= 1".to_string(),
            });
        }
        if cfg.key_dim == 0 {
            return Err(PeftError::Internal {
                msg: "key_dim must be >= 1".to_string(),
            });
        }

        let mut keys = vec![0.0_f32; cfg.pool_size * cfg.key_dim];
        rng.fill_normal(&mut keys);
        for v in keys.iter_mut() {
            *v *= 0.02;
        }

        let block = cfg.prompt_len * cfg.embed_dim;
        let mut prompts = vec![0.0_f32; cfg.pool_size * block];
        rng.fill_normal(&mut prompts);
        for v in prompts.iter_mut() {
            *v *= 0.02;
        }

        Ok(Self { keys, prompts, cfg })
    }

    /// Cosine similarity of the query to each pool key.
    ///
    /// Returns `pool_size` scores, each in `[-1, 1]`. A query (or key) of all
    /// zeros yields a score of `0` via the `COSINE_EPS` guard, never `NaN`.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if `query.len() ≠ key_dim`.
    pub fn key_scores(&self, query: &[f32]) -> PeftResult<Vec<f32>> {
        if query.len() != self.cfg.key_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.key_dim,
                got: query.len(),
            });
        }

        let key_dim = self.cfg.key_dim;
        let q_norm = l2_norm(query);

        let scores = (0..self.cfg.pool_size)
            .map(|m| {
                let base = m * key_dim;
                let key = &self.keys[base..base + key_dim];
                cosine(query, key, q_norm)
            })
            .collect();
        Ok(scores)
    }

    /// Indices of the `top_n` keys by cosine score, descending.
    ///
    /// Ties are broken by the lower index (stable descending sort by score).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::key_scores`].
    pub fn select(&self, query: &[f32]) -> PeftResult<Vec<usize>> {
        let scores = self.key_scores(query)?;
        let mut indices: Vec<usize> = (0..self.cfg.pool_size).collect();
        // Sort by descending score; on ties keep the lower index first. Using a
        // total comparator that falls back to ascending index makes the order
        // fully deterministic regardless of the sort's stability guarantees.
        indices.sort_by(|&a, &b| match scores[b].partial_cmp(&scores[a]) {
            Some(std::cmp::Ordering::Equal) | None => a.cmp(&b),
            Some(other) => other,
        });
        indices.truncate(self.cfg.top_n);
        Ok(indices)
    }

    /// Gather the selected prompts concatenated in selection order.
    ///
    /// Returns a flat row-major matrix of shape `(top_n · prompt_len) × embed_dim`,
    /// i.e. `top_n · prompt_len · embed_dim` values.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::select`].
    pub fn selected_prompts(&self, query: &[f32]) -> PeftResult<Vec<f32>> {
        let selected = self.select(query)?;
        let block = self.cfg.prompt_len * self.cfg.embed_dim;
        let mut out = Vec::with_capacity(selected.len() * block);
        for &m in &selected {
            let base = m * block;
            out.extend_from_slice(&self.prompts[base..base + block]);
        }
        Ok(out)
    }

    /// Key-matching loss `(1/N) Σ_{m∈selected} (1 − cosine(query, key_m))`.
    ///
    /// The loss lies in `[0, 2]` (since cosine ∈ `[-1, 1]`) and is minimized when
    /// the selected keys align with the query.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::select`] / [`Self::key_scores`].
    pub fn matching_loss(&self, query: &[f32]) -> PeftResult<f32> {
        let scores = self.key_scores(query)?;
        let selected = self.select(query)?;
        let n = selected.len();
        if n == 0 {
            // Unreachable given `top_n ≥ 1` validated at construction, but handled
            // defensively to avoid division by zero.
            return Ok(0.0);
        }
        let sum: f32 = selected.iter().map(|&m| 1.0 - scores[m]).sum();
        Ok(sum / n as f32)
    }

    /// Total prefix length contributed to the sequence: `top_n · prompt_len` tokens.
    #[inline]
    #[must_use]
    pub fn prefix_len(&self) -> usize {
        self.cfg.top_n * self.cfg.prompt_len
    }

    /// Total number of learnable parameters:
    /// `pool_size · key_dim + pool_size · prompt_len · embed_dim`.
    #[inline]
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.cfg.pool_size * self.cfg.key_dim
            + self.cfg.pool_size * self.cfg.prompt_len * self.cfg.embed_dim
    }
}

// ---------------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------------

/// L2 norm of a slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Cosine similarity `⟨q, k⟩ / (‖q‖·‖k‖ + ε)`, reusing a precomputed `‖q‖`.
fn cosine(q: &[f32], k: &[f32], q_norm: f32) -> f32 {
    let dot: f32 = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum();
    let k_norm = l2_norm(k);
    dot / (q_norm * k_norm + COSINE_EPS)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(
        pool_size: usize,
        top_n: usize,
        prompt_len: usize,
        embed_dim: usize,
        key_dim: usize,
    ) -> PromptPoolConfig {
        PromptPoolConfig {
            pool_size,
            top_n,
            prompt_len,
            embed_dim,
            key_dim,
        }
    }

    /// Helper: overwrite pool key `m` with the given vector.
    fn set_key(pool: &mut PromptPool, m: usize, v: &[f32]) {
        let key_dim = pool.cfg.key_dim;
        let base = m * key_dim;
        pool.keys[base..base + key_dim].copy_from_slice(v);
    }

    // ── Test 1: key_scores length and cosine range ─────────────────────────
    #[test]
    fn key_scores_length_and_range() {
        let mut rng = LcgRng::new(1);
        let pool = PromptPool::new(cfg(6, 2, 3, 4, 5), &mut rng).unwrap();
        let query = vec![0.3_f32, -0.1, 0.7, 0.2, -0.4];
        let scores = pool.key_scores(&query).unwrap();
        assert_eq!(scores.len(), 6);
        for &s in &scores {
            assert!(
                (-1.0 - 1e-5..=1.0 + 1e-5).contains(&s),
                "cosine score {s} outside [-1, 1]"
            );
        }
    }

    // ── Test 2: select returns top_n distinct indices ──────────────────────
    #[test]
    fn select_returns_distinct_indices() {
        let mut rng = LcgRng::new(2);
        let pool = PromptPool::new(cfg(8, 4, 2, 3, 4), &mut rng).unwrap();
        let query = vec![0.5_f32, 0.5, 0.5, 0.5];
        let sel = pool.select(&query).unwrap();
        assert_eq!(sel.len(), 4);
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            4,
            "selected indices must be distinct: {sel:?}"
        );
        for &i in &sel {
            assert!(i < 8, "index {i} out of range");
        }
    }

    // ── Test 3: query equal to key m → m selected first ────────────────────
    #[test]
    fn query_equal_to_key_selected_first() {
        let mut rng = LcgRng::new(3);
        let mut pool = PromptPool::new(cfg(5, 2, 2, 3, 4), &mut rng).unwrap();
        // Make key 3 a clear, large-magnitude direction.
        let target = [1.0_f32, 2.0, -1.0, 0.5];
        set_key(&mut pool, 3, &target);
        // Other keys point elsewhere so they cannot tie.
        set_key(&mut pool, 0, &[-1.0, 0.0, 0.0, 0.0]);
        set_key(&mut pool, 1, &[0.0, -1.0, 0.0, 0.0]);
        set_key(&mut pool, 2, &[0.0, 0.0, 0.0, -1.0]);
        set_key(&mut pool, 4, &[-1.0, -2.0, 1.0, -0.5]);
        let sel = pool.select(&target).unwrap();
        assert_eq!(sel[0], 3, "key equal to query must rank first, got {sel:?}");
    }

    // ── Test 4: selected_prompts length ────────────────────────────────────
    #[test]
    fn selected_prompts_length() {
        let mut rng = LcgRng::new(4);
        let top_n = 3;
        let prompt_len = 4;
        let embed_dim = 6;
        let pool = PromptPool::new(cfg(7, top_n, prompt_len, embed_dim, 5), &mut rng).unwrap();
        let query = vec![0.1_f32; 5];
        let out = pool.selected_prompts(&query).unwrap();
        assert_eq!(out.len(), top_n * prompt_len * embed_dim);
    }

    // ── Test 5: matching_loss within [0, 2] ────────────────────────────────
    #[test]
    fn matching_loss_in_range() {
        let mut rng = LcgRng::new(5);
        let pool = PromptPool::new(cfg(10, 3, 2, 4, 6), &mut rng).unwrap();
        let query = vec![0.2_f32, -0.5, 0.1, 0.9, -0.3, 0.4];
        let loss = pool.matching_loss(&query).unwrap();
        assert!(
            (0.0 - 1e-5..=2.0 + 1e-5).contains(&loss),
            "matching loss {loss} outside [0, 2]"
        );
    }

    // ── Test 6: matching_loss == 0 when query equals all selected keys ─────
    #[test]
    fn matching_loss_zero_when_keys_equal_query() {
        let mut rng = LcgRng::new(6);
        let mut pool = PromptPool::new(cfg(3, 3, 2, 3, 4), &mut rng).unwrap();
        // top_n == pool_size → all keys selected; make every key == query.
        let query = [0.4_f32, -0.2, 0.7, 0.1];
        for m in 0..3 {
            set_key(&mut pool, m, &query);
        }
        let loss = pool.matching_loss(&query).unwrap();
        assert!(loss.abs() < 1e-5, "expected loss≈0, got {loss}");
    }

    // ── Test 7: prefix_len == top_n * prompt_len ───────────────────────────
    #[test]
    fn prefix_len_formula() {
        let mut rng = LcgRng::new(7);
        let pool = PromptPool::new(cfg(9, 4, 5, 2, 3), &mut rng).unwrap();
        assert_eq!(pool.prefix_len(), 4 * 5);
    }

    // ── Test 8: top_n == pool_size selects all ─────────────────────────────
    #[test]
    fn top_n_equals_pool_size_selects_all() {
        let mut rng = LcgRng::new(8);
        let pool = PromptPool::new(cfg(5, 5, 2, 3, 4), &mut rng).unwrap();
        let query = vec![0.3_f32, 0.1, -0.2, 0.6];
        let sel = pool.select(&query).unwrap();
        assert_eq!(sel.len(), 5);
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    // ── Test 9: deterministic given seed ───────────────────────────────────
    #[test]
    fn deterministic_same_seed() {
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        let p1 = PromptPool::new(cfg(6, 2, 3, 4, 5), &mut r1).unwrap();
        let p2 = PromptPool::new(cfg(6, 2, 3, 4, 5), &mut r2).unwrap();
        assert_eq!(p1.keys, p2.keys);
        assert_eq!(p1.prompts, p2.prompts);
        let q = vec![0.5_f32; 5];
        assert_eq!(
            p1.selected_prompts(&q).unwrap(),
            p2.selected_prompts(&q).unwrap()
        );
    }

    // ── Test 10: err — pool_size = 0 ───────────────────────────────────────
    #[test]
    fn err_pool_size_zero() {
        let mut rng = LcgRng::new(10);
        let res = PromptPool::new(cfg(0, 1, 2, 3, 4), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 11: err — top_n = 0 ───────────────────────────────────────────
    #[test]
    fn err_top_n_zero() {
        let mut rng = LcgRng::new(11);
        let res = PromptPool::new(cfg(4, 0, 2, 3, 4), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 12: err — top_n > pool_size ───────────────────────────────────
    #[test]
    fn err_top_n_exceeds_pool_size() {
        let mut rng = LcgRng::new(12);
        let res = PromptPool::new(cfg(3, 5, 2, 3, 4), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 13: err — prompt_len = 0 ──────────────────────────────────────
    #[test]
    fn err_prompt_len_zero() {
        let mut rng = LcgRng::new(13);
        let res = PromptPool::new(cfg(4, 2, 0, 3, 4), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 14: err — embed_dim = 0 ───────────────────────────────────────
    #[test]
    fn err_embed_dim_zero() {
        let mut rng = LcgRng::new(14);
        let res = PromptPool::new(cfg(4, 2, 3, 0, 4), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 15: err — key_dim = 0 ─────────────────────────────────────────
    #[test]
    fn err_key_dim_zero() {
        let mut rng = LcgRng::new(15);
        let res = PromptPool::new(cfg(4, 2, 3, 4, 0), &mut rng);
        assert!(matches!(res, Err(PeftError::Internal { .. })));
    }

    // ── Test 16: err — query wrong length ──────────────────────────────────
    #[test]
    fn err_query_wrong_length() {
        let mut rng = LcgRng::new(16);
        let pool = PromptPool::new(cfg(4, 2, 3, 4, 5), &mut rng).unwrap();
        let res = pool.key_scores(&[0.0_f32; 4]); // key_dim = 5
        assert!(matches!(res, Err(PeftError::DimensionMismatch { .. })));
        let res2 = pool.select(&[0.0_f32; 6]);
        assert!(matches!(res2, Err(PeftError::DimensionMismatch { .. })));
    }

    // ── Test 17: query of zeros handled (no NaN via ε) ─────────────────────
    #[test]
    fn zero_query_no_nan() {
        let mut rng = LcgRng::new(17);
        let pool = PromptPool::new(cfg(5, 2, 2, 3, 4), &mut rng).unwrap();
        let query = vec![0.0_f32; 4];
        let scores = pool.key_scores(&query).unwrap();
        for &s in &scores {
            assert!(s.is_finite(), "score must be finite, got {s}");
            assert!(s.abs() < 1e-4, "zero query → cosine≈0, got {s}");
        }
        let loss = pool.matching_loss(&query).unwrap();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");
    }

    // ── Test 18: selection order is by descending score ────────────────────
    #[test]
    fn selection_order_descending_score() {
        let mut rng = LcgRng::new(18);
        let pool = PromptPool::new(cfg(7, 4, 2, 3, 5), &mut rng).unwrap();
        let query = vec![0.7_f32, -0.2, 0.4, 0.1, -0.6];
        let scores = pool.key_scores(&query).unwrap();
        let sel = pool.select(&query).unwrap();
        for w in sel.windows(2) {
            assert!(
                scores[w[0]] >= scores[w[1]] - 1e-6,
                "selection not in descending score order: {} then {}",
                scores[w[0]],
                scores[w[1]]
            );
        }
    }

    // ── Test 19: tie-break by lowest index ─────────────────────────────────
    #[test]
    fn tie_break_lowest_index() {
        let mut rng = LcgRng::new(19);
        let mut pool = PromptPool::new(cfg(4, 2, 2, 3, 3), &mut rng).unwrap();
        // Make keys 0 and 2 identical (and aligned with the query) so they tie.
        let dir = [1.0_f32, 0.0, 0.0];
        set_key(&mut pool, 0, &dir);
        set_key(&mut pool, 2, &dir);
        // Keys 1 and 3 point away so they score lower.
        set_key(&mut pool, 1, &[-1.0, 0.0, 0.0]);
        set_key(&mut pool, 3, &[0.0, -1.0, 0.0]);
        let sel = pool.select(&dir).unwrap();
        // Both tied winners selected, lower index (0) before higher (2).
        assert_eq!(
            sel,
            vec![0, 2],
            "tie must break by lowest index, got {sel:?}"
        );
    }

    // ── Test 20: selected_prompts matches gathered blocks in order ─────────
    #[test]
    fn selected_prompts_matches_blocks() {
        let mut rng = LcgRng::new(20);
        let prompt_len = 2;
        let embed_dim = 3;
        let pool = PromptPool::new(cfg(5, 3, prompt_len, embed_dim, 4), &mut rng).unwrap();
        let query = vec![0.2_f32, 0.5, -0.1, 0.3];
        let sel = pool.select(&query).unwrap();
        let gathered = pool.selected_prompts(&query).unwrap();
        let block = prompt_len * embed_dim;
        for (slot, &m) in sel.iter().enumerate() {
            let src = &pool.prompts[m * block..m * block + block];
            let dst = &gathered[slot * block..slot * block + block];
            assert_eq!(src, dst, "block {slot} (pool index {m}) mismatch");
        }
    }

    // ── Test 21: num_params formula ────────────────────────────────────────
    #[test]
    fn num_params_formula() {
        let mut rng = LcgRng::new(21);
        let pool = PromptPool::new(cfg(6, 2, 3, 4, 5), &mut rng).unwrap();
        let expected = 6 * 5 + 6 * 3 * 4;
        assert_eq!(pool.num_params(), expected);
    }
}
