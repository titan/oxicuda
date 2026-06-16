//! Learning to Prompt (L2P) continual learning via prompt-pool retrieval.
//!
//! Implements the method from:
//! Wang et al. "Learning to Prompt for Continual Learning."
//! CVPR 2022.
//!
//! L2P maintains a **prompt pool** of learnable prompt embeddings, each
//! associated with a key vector. At inference and training time, the input
//! feature `x` is matched against all keys via cosine similarity and the
//! top-K most relevant prompts are prepended to the transformer input.
//!
//! # Layout conventions
//!
//! - `prompt_pool`: flat `[pool_size × prompt_len × d_model]` row-major.
//! - `keys`:        flat `[pool_size × d_model]` row-major, L2-normalised.

use crate::error::{ContinualError, ContinualResult};
use crate::handle::LcgRng;

/// Type alias for the crate-local RNG used in L2P operations.
pub type ContRng = LcgRng;

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the L2P prompt pool.
#[derive(Debug, Clone)]
pub struct L2pConfig {
    /// Number of learnable prompts in the pool (`M` in the paper).
    pub pool_size: usize,
    /// Number of tokens per individual prompt (`L_p`).
    pub prompt_len: usize,
    /// Transformer embedding dimension (`d`).
    pub d_model: usize,
    /// Number of prompts retrieved per forward call (`G`).
    pub top_k: usize,
}

// ─── L2P prompt pool ──────────────────────────────────────────────────────────

/// L2P prompt pool with key-based cosine-similarity retrieval.
#[derive(Debug, Clone)]
pub struct L2p {
    /// Flat prompt embeddings: `[pool_size × prompt_len × d_model]`.
    prompt_pool: Vec<f32>,
    /// L2-normalised key vectors: `[pool_size × d_model]`.
    keys: Vec<f32>,
    /// Configuration snapshot.
    config: L2pConfig,
}

impl L2p {
    /// Create a new L2P prompt pool.
    ///
    /// The prompt pool is initialised from N(0, 1) samples; the keys are
    /// likewise sampled from N(0, 1) and then L2-normalised.
    ///
    /// # Errors
    ///
    /// - [`ContinualError::DimensionMismatch`] if `d_model == 0`
    ///   (expected ≥ 1, got 0).
    /// - [`ContinualError::DimensionMismatch`] if `top_k > pool_size`
    ///   (expected = pool_size, got = top_k).
    pub fn new(config: L2pConfig, rng: &mut ContRng) -> ContinualResult<Self> {
        if config.d_model == 0 {
            return Err(ContinualError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if config.top_k > config.pool_size {
            return Err(ContinualError::DimensionMismatch {
                expected: config.pool_size,
                got: config.top_k,
            });
        }

        // Allocate and fill prompt pool with N(0,1) values.
        let pool_len = config.pool_size * config.prompt_len * config.d_model;
        let mut prompt_pool = vec![0.0_f32; pool_len];
        rng.fill_normal(&mut prompt_pool);

        // Allocate keys, fill with N(0,1), and L2-normalise each key.
        let key_len = config.pool_size * config.d_model;
        let mut keys = vec![0.0_f32; key_len];
        rng.fill_normal(&mut keys);

        for i in 0..config.pool_size {
            let start = i * config.d_model;
            let end = start + config.d_model;
            l2_normalize_in_place(&mut keys[start..end]);
        }

        Ok(Self {
            prompt_pool,
            keys,
            config,
        })
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Query the pool for the top-K prompt indices most similar to `x`.
    ///
    /// `x` must have length `d_model`.  Since the keys are pre-L2-normalised,
    /// the cosine similarity simplifies to `dot(x / ‖x‖, key_i)`.
    ///
    /// Returns a `Vec<usize>` of length `top_k` sorted by *decreasing*
    /// cosine similarity.
    ///
    /// # Errors
    ///
    /// - [`ContinualError::DimensionMismatch`] if `x.len() != d_model`.
    /// - [`ContinualError::EmptyInput`] if `x` is empty (d_model == 0 is
    ///   already rejected at construction, but defensive).
    pub fn query(&self, x: &[f32]) -> ContinualResult<Vec<usize>> {
        let d = self.config.d_model;
        if x.len() != d {
            return Err(ContinualError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }

        // Compute ‖x‖ and guard against zero-norm queries.
        let norm_x = l2_norm(x);
        let inv_norm_x = if norm_x > 1e-12 { 1.0 / norm_x } else { 0.0 };

        // Compute cosine similarity with every key (dot of normalised x and key).
        let mut sims: Vec<(usize, f32)> = (0..self.config.pool_size)
            .map(|i| {
                let key_start = i * d;
                let sim = dot_product(x, &self.keys[key_start..key_start + d]) * inv_norm_x;
                (i, sim)
            })
            .collect();

        // Partial sort descending: bring the top_k largest to the front.
        let k = self.config.top_k;
        sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(sims[..k].iter().map(|&(idx, _)| idx).collect())
    }

    /// Retrieve the concatenated prompt embeddings for the given indices.
    ///
    /// Returns a flat `Vec<f32>` of shape `[k × prompt_len × d_model]`.
    ///
    /// # Errors
    ///
    /// - [`ContinualError::DimensionMismatch`] if any index ≥ `pool_size`.
    pub fn retrieve_prompts(&self, prompt_ids: &[usize]) -> ContinualResult<Vec<f32>> {
        let pool_size = self.config.pool_size;
        let prompt_stride = self.config.prompt_len * self.config.d_model;

        let mut out = Vec::with_capacity(prompt_ids.len() * prompt_stride);
        for &idx in prompt_ids {
            if idx >= pool_size {
                return Err(ContinualError::DimensionMismatch {
                    expected: pool_size.saturating_sub(1),
                    got: idx,
                });
            }
            let start = idx * prompt_stride;
            let end = start + prompt_stride;
            out.extend_from_slice(&self.prompt_pool[start..end]);
        }
        Ok(out)
    }

    /// Full L2P forward pass.
    ///
    /// 1. Query the top-K prompt indices using cosine similarity.
    /// 2. Retrieve the corresponding prompt embeddings (`[k×prompt_len × d_model]` flat).
    /// 3. Mean-pool the retrieved prompt tokens to produce a single `[d_model]` vector.
    /// 4. Return `x + mean_pooled` (residual addition).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::query`] and [`Self::retrieve_prompts`].
    /// Returns [`ContinualError::NanEncountered`] if any output element is
    /// non-finite.
    pub fn forward(&self, x: &[f32]) -> ContinualResult<Vec<f32>> {
        let d = self.config.d_model;
        if x.len() != d {
            return Err(ContinualError::DimensionMismatch {
                expected: d,
                got: x.len(),
            });
        }

        let prompt_ids = self.query(x)?;
        let prompts = self.retrieve_prompts(&prompt_ids)?;

        // Mean-pool over k * prompt_len tokens, each of length d_model.
        let n_tokens = prompts.len() / d;
        let inv_n = if n_tokens > 0 {
            1.0 / n_tokens as f32
        } else {
            0.0
        };

        let mut mean_pooled = vec![0.0_f32; d];
        for t in 0..n_tokens {
            let tok_start = t * d;
            for j in 0..d {
                mean_pooled[j] += prompts[tok_start + j];
            }
        }
        for v in &mut mean_pooled {
            *v *= inv_n;
        }

        // Residual addition: x + mean_pooled.
        let mut out = Vec::with_capacity(d);
        for j in 0..d {
            let val = x[j] + mean_pooled[j];
            if !val.is_finite() {
                return Err(ContinualError::NanEncountered {
                    location: "L2p::forward",
                });
            }
            out.push(val);
        }

        Ok(out)
    }

    /// Compute the key diversity regulariser.
    ///
    /// Encourages the prompt keys to be mutually orthogonal by penalising
    /// large squared pairwise dot products:
    ///
    /// ```text
    /// R = Σ_{i < j} dot(key_i, key_j)²
    /// ```
    ///
    /// Since keys are L2-normalised, `dot(key_i, key_j) ∈ [-1, 1]` and
    /// `R ∈ [0, C(pool_size, 2)]`.  A smaller `R` indicates more diverse
    /// (orthogonal) keys.
    #[must_use]
    pub fn diversity_regularizer(&self) -> f32 {
        let d = self.config.d_model;
        let m = self.config.pool_size;
        let mut reg = 0.0_f32;
        for i in 0..m {
            for j in (i + 1)..m {
                let ki_start = i * d;
                let kj_start = j * d;
                let sim = dot_product(
                    &self.keys[ki_start..ki_start + d],
                    &self.keys[kj_start..kj_start + d],
                );
                reg += sim * sim;
            }
        }
        reg
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Return a reference to the raw key matrix (`[pool_size × d_model]` flat).
    #[must_use]
    pub fn keys(&self) -> &[f32] {
        &self.keys
    }

    /// Return a reference to the raw prompt pool (`[pool_size × prompt_len × d_model]` flat).
    #[must_use]
    pub fn prompt_pool(&self) -> &[f32] {
        &self.prompt_pool
    }

    /// Return a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &L2pConfig {
        &self.config
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Compute ‖v‖₂.
#[inline]
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// L2-normalise `v` in-place.  If the norm is below `1e-12` the vector is
/// left unchanged (all-zero is valid — it will not cause NaN).
#[inline]
fn l2_normalize_in_place(v: &mut [f32]) {
    let n = l2_norm(v);
    if n > 1e-12 {
        let inv_n = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv_n;
        }
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> ContRng {
        LcgRng::new(42)
    }

    fn default_config() -> L2pConfig {
        L2pConfig {
            pool_size: 8,
            prompt_len: 4,
            d_model: 16,
            top_k: 3,
        }
    }

    // ── Test 1: query returns exactly top_k indices ───────────────────────────

    #[test]
    fn query_returns_k() {
        let mut rng = make_rng();
        let cfg = default_config();
        let top_k = cfg.top_k;
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let ids = l2p.query(&x).expect("query should succeed");

        assert_eq!(
            ids.len(),
            top_k,
            "query must return exactly top_k={top_k} indices"
        );
    }

    // ── Test 2: all returned indices are < pool_size ──────────────────────────

    #[test]
    fn query_in_range() {
        let mut rng = make_rng();
        let cfg = default_config();
        let pool_size = cfg.pool_size;
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        let x: Vec<f32> = vec![1.0; 16];
        let ids = l2p.query(&x).expect("query should succeed");

        for &idx in &ids {
            assert!(
                idx < pool_size,
                "returned index {idx} must be < pool_size={pool_size}"
            );
        }
    }

    // ── Test 3: retrieve_prompts returns vec of correct length ────────────────

    #[test]
    fn retrieve_shape() {
        let mut rng = make_rng();
        let cfg = default_config();
        let prompt_len = cfg.prompt_len;
        let d_model = cfg.d_model;
        let k = cfg.top_k;
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        let ids: Vec<usize> = (0..k).collect();
        let prompts = l2p
            .retrieve_prompts(&ids)
            .expect("retrieve_prompts should succeed");

        let expected_len = k * prompt_len * d_model;
        assert_eq!(
            prompts.len(),
            expected_len,
            "retrieved prompts should have length k*prompt_len*d_model={expected_len}"
        );
    }

    // ── Test 4: forward output contains no NaN or inf ─────────────────────────

    #[test]
    fn forward_finite() {
        let mut rng = make_rng();
        let cfg = default_config();
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        let x: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let out = l2p.forward(&x).expect("forward should succeed");

        for (j, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "forward output[{j}] = {v} is not finite");
        }
    }

    // ── Test 5: each key has L2 norm ≈ 1.0 ───────────────────────────────────

    #[test]
    fn keys_normalized_approx() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d = cfg.d_model;
        let m = cfg.pool_size;
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        for i in 0..m {
            let start = i * d;
            let norm = l2_norm(&l2p.keys()[start..start + d]);
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "key {i} has norm {norm}, expected ≈ 1.0"
            );
        }
    }

    // ── Test 6: diversity_regularizer is non-negative ─────────────────────────

    #[test]
    fn diversity_positive() {
        let mut rng = make_rng();
        let cfg = default_config();
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");
        let reg = l2p.diversity_regularizer();
        assert!(
            reg >= 0.0,
            "diversity_regularizer must be non-negative, got {reg}"
        );
    }

    // ── Test 7: top_k > pool_size at construction returns Err ─────────────────

    #[test]
    fn top_k_gt_pool_size_error() {
        let mut rng = make_rng();
        let cfg = L2pConfig {
            pool_size: 2,
            prompt_len: 4,
            d_model: 8,
            top_k: 3,
        };
        let result = L2p::new(cfg, &mut rng);
        assert!(
            result.is_err(),
            "L2p::new should return Err when top_k > pool_size"
        );
        match result.unwrap_err() {
            ContinualError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 2, "expected should be pool_size=2");
                assert_eq!(got, 3, "got should be top_k=3");
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    // ── Test 8: d_model == 0 at construction returns Err ─────────────────────

    #[test]
    fn d_model_zero_error() {
        let mut rng = make_rng();
        let cfg = L2pConfig {
            pool_size: 4,
            prompt_len: 2,
            d_model: 0,
            top_k: 1,
        };
        let result = L2p::new(cfg, &mut rng);
        assert!(
            result.is_err(),
            "L2p::new should return Err when d_model == 0"
        );
        match result.unwrap_err() {
            ContinualError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    // ── Test 9: very different query vectors retrieve different prompt sets ────

    #[test]
    fn different_queries_different_prompts() {
        // Use a large pool so there is room for different top-k sets.
        let mut rng = LcgRng::new(777);
        let cfg = L2pConfig {
            pool_size: 16,
            prompt_len: 2,
            d_model: 32,
            top_k: 2,
        };
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        // Two queries that are nearly antipodal in high-dimensional space.
        let x_pos: Vec<f32> = vec![1.0_f32; 32];
        let x_neg: Vec<f32> = vec![-1.0_f32; 32];

        let ids_pos = l2p.query(&x_pos).expect("query x_pos should succeed");
        let ids_neg = l2p.query(&x_neg).expect("query x_neg should succeed");

        // At least one index should differ between the two retrieval results.
        let any_differ = ids_pos.iter().zip(ids_neg.iter()).any(|(a, b)| a != b)
            || ids_pos.iter().collect::<std::collections::HashSet<_>>()
                != ids_neg.iter().collect::<std::collections::HashSet<_>>();
        assert!(
            any_differ,
            "Antipodal queries should retrieve at least partially different prompts\n\
             pos: {ids_pos:?}\n neg: {ids_neg:?}"
        );
    }

    // ── Test 10: forward output has length d_model ────────────────────────────

    #[test]
    fn forward_shape() {
        let mut rng = make_rng();
        let cfg = default_config();
        let d = cfg.d_model;
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        let x: Vec<f32> = vec![0.5_f32; d];
        let out = l2p.forward(&x).expect("forward should succeed");

        assert_eq!(
            out.len(),
            d,
            "forward output must have length d_model={d}, got {}",
            out.len()
        );
    }

    // ── Test 11: retrieve_prompts with out-of-range index returns Err ─────────

    #[test]
    fn retrieve_out_of_range_error() {
        let mut rng = make_rng();
        let cfg = default_config();
        let pool_size = cfg.pool_size;
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");

        let result = l2p.retrieve_prompts(&[pool_size]);
        assert!(
            result.is_err(),
            "retrieve_prompts should error on index >= pool_size"
        );
    }

    // ── Test 12: diversity_regularizer is finite ──────────────────────────────

    #[test]
    fn diversity_finite() {
        let mut rng = LcgRng::new(99);
        let cfg = default_config();
        let l2p = L2p::new(cfg, &mut rng).expect("L2p::new should succeed");
        let reg = l2p.diversity_regularizer();
        assert!(
            reg.is_finite(),
            "diversity_regularizer must be finite, got {reg}"
        );
    }
}
