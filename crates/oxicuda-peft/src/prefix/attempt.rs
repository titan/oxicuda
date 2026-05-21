//! ATTEMPT: Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts.
//!
//! Reference: Asai A, Sadeghian P, Hajishirzi H (2022) "ATTEMPT: Parameter-Efficient Multi-task
//! Tuning via Attentional Mixtures of Soft Prompts", *EMNLP 2022*: 6655–6672.
//! <https://arxiv.org/abs/2202.08906>
//!
//! ## Design
//!
//! Unlike SPoT, which uses cosine similarity for initialization-only retrieval, ATTEMPT uses a
//! **learnable attention mechanism** at runtime. Given a task representation vector
//! `input_repr ∈ ℝ^{prompt_dim}`, the router:
//!
//! 1. Projects to a query: `q = W_query · input_repr ∈ ℝ^{key_dim}`.
//! 2. Computes scaled dot-product scores against per-source key vectors:
//!    `s_k = (q · key_k) / sqrt(key_dim)`.
//! 3. Applies temperature-scaled softmax to obtain attention weights `α`.
//! 4. Returns the weighted mixture of source soft prompts: `mix = Σ_k α_k · P_k`.
//!
//! The routing is differentiable and part of the forward pass, enabling joint optimization
//! of the router and the prompts during multi-task training.

use super::prompt_tuning::SoftPrompt;
use crate::error::{PeftError, PeftResult};
use crate::handle::PeftHandle;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for an [`AttemptRouter`].
#[derive(Clone, Debug)]
pub struct AttemptConfig {
    /// Number of virtual soft prompt tokens.
    pub num_tokens: usize,
    /// Embedding dimension per token (= `prompt_dim` in the paper).
    pub prompt_dim: usize,
    /// Attention key / query dimension.
    pub key_dim: usize,
    /// Softmax temperature `τ > 0`; lower values sharpen the distribution.
    pub temperature: f32,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// ATTEMPT attentional soft-prompt router.
///
/// Holds a learned query projection matrix, per-source key vectors, and per-source
/// soft prompt embeddings. At routing time it computes an attention mixture of the
/// source prompts using the task representation as input.
#[derive(Debug)]
pub struct AttemptRouter {
    /// Query projection `W_q ∈ ℝ^{key_dim × prompt_dim}` (row-major).
    pub(crate) query_proj: Vec<f32>,
    /// Per-source key vectors stacked: `(num_sources × key_dim)` row-major.
    pub(crate) source_keys: Vec<f32>,
    /// Per-source prompt embeddings: `source_prompts[k].len() == num_tokens * prompt_dim`.
    pub(crate) source_prompts: Vec<Vec<f32>>,
    /// Router configuration.
    pub cfg: AttemptConfig,
    /// Number of source prompts.
    num_sources: usize,
}

impl AttemptRouter {
    /// Construct a new ATTEMPT router with random initialization.
    ///
    /// - `query_proj`: Kaiming-uniform `U(-limit, limit)` where `limit = sqrt(6/prompt_dim)`.
    /// - `source_keys`: N(0, 0.02) per entry.
    /// - `source_prompts`: N(0, 0.02) per entry.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::Internal`] if any dimension constraint is violated:
    /// `num_sources ≥ 1`, `num_tokens ≥ 1`, `prompt_dim ≥ 1`, `key_dim ≥ 1`,
    /// `temperature > 0`.
    pub fn new(
        cfg: AttemptConfig,
        num_sources: usize,
        handle: &mut PeftHandle,
    ) -> PeftResult<Self> {
        if num_sources == 0 {
            return Err(PeftError::Internal {
                msg: "num_sources must be >= 1".to_string(),
            });
        }
        if cfg.num_tokens == 0 {
            return Err(PeftError::Internal {
                msg: "num_tokens must be >= 1".to_string(),
            });
        }
        if cfg.prompt_dim == 0 {
            return Err(PeftError::Internal {
                msg: "prompt_dim must be >= 1".to_string(),
            });
        }
        if cfg.key_dim == 0 {
            return Err(PeftError::Internal {
                msg: "key_dim must be >= 1".to_string(),
            });
        }
        if cfg.temperature <= 0.0 || cfg.temperature.is_nan() {
            return Err(PeftError::Internal {
                msg: format!(
                    "temperature must be > 0 and finite, got {}",
                    cfg.temperature
                ),
            });
        }

        let rng = &mut handle.rng;

        // Query projection: Kaiming-uniform on [-limit, limit]
        let limit = (6.0_f32 / cfg.prompt_dim as f32).sqrt();
        let qp_size = cfg.key_dim * cfg.prompt_dim;
        let query_proj = (0..qp_size)
            .map(|_| rng.next_f32() * 2.0 * limit - limit)
            .collect();

        // Source keys: N(0, 0.02)
        let sk_size = num_sources * cfg.key_dim;
        let source_keys = (0..sk_size).map(|_| rng.next_normal() * 0.02).collect();

        // Source prompts: N(0, 0.02)
        let prompt_size = cfg.num_tokens * cfg.prompt_dim;
        let source_prompts = (0..num_sources)
            .map(|_| (0..prompt_size).map(|_| rng.next_normal() * 0.02).collect())
            .collect();

        Ok(Self {
            query_proj,
            source_keys,
            source_prompts,
            cfg,
            num_sources,
        })
    }

    /// Compute attention weights `α_k` for each source prompt given `input_repr`.
    ///
    /// Algorithm:
    /// 1. Query: `q[d] = Σ_p W_q[d, p] · input_repr[p]`.
    /// 2. Score: `s_k = (Σ_d q[d] · key_k[d]) / sqrt(key_dim)`.
    /// 3. Temperature: `s_k /= temperature`.
    /// 4. Numerically-stable softmax (max-shift).
    ///
    /// Returns `Vec<f32>` of length `num_sources` summing to ≈ 1.0.
    ///
    /// # Errors
    ///
    /// Returns [`PeftError::DimensionMismatch`] if `input_repr.len() ≠ prompt_dim`.
    pub fn attention_weights(&self, input_repr: &[f32]) -> PeftResult<Vec<f32>> {
        if input_repr.len() != self.cfg.prompt_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.prompt_dim,
                got: input_repr.len(),
            });
        }

        let key_dim = self.cfg.key_dim;
        let prompt_dim = self.cfg.prompt_dim;

        // Step 1: project input_repr to query q in R^{key_dim}
        let q: Vec<f32> = (0..key_dim)
            .map(|d| {
                let row_base = d * prompt_dim;
                self.query_proj[row_base..row_base + prompt_dim]
                    .iter()
                    .zip(input_repr.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f32>()
            })
            .collect();

        // Step 2 & 3: scaled dot-product scores + temperature scaling
        let scale = 1.0 / ((key_dim as f32).sqrt() * self.cfg.temperature);
        let scores: Vec<f32> = (0..self.num_sources)
            .map(|k| {
                let key_base = k * key_dim;
                let s: f32 = q
                    .iter()
                    .zip(self.source_keys[key_base..key_base + key_dim].iter())
                    .map(|(qd, kd)| qd * kd)
                    .sum();
                s * scale
            })
            .collect();

        // Step 4: numerically-stable softmax
        Ok(softmax_stable(&scores))
    }

    /// Compute the attention-weighted mixture of source prompts.
    ///
    /// Returns a [`SoftPrompt`] of shape `(num_tokens × prompt_dim)`.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::attention_weights`].
    pub fn route(&self, input_repr: &[f32]) -> PeftResult<SoftPrompt> {
        let alpha = self.attention_weights(input_repr)?;
        let prompt_size = self.cfg.num_tokens * self.cfg.prompt_dim;
        let mut mix = vec![0.0_f32; prompt_size];
        for (k, &w) in alpha.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            for (mix_j, &p_j) in mix.iter_mut().zip(self.source_prompts[k].iter()) {
                *mix_j += w * p_j;
            }
        }
        Ok(SoftPrompt {
            num_tokens: self.cfg.num_tokens,
            embed_dim: self.cfg.prompt_dim,
            embeddings: mix,
        })
    }

    /// Route using only the top-`k` sources by attention weight; non-top-k weights
    /// are zeroed and the remaining weights are re-normalized to sum to 1.
    ///
    /// # Errors
    ///
    /// - [`PeftError::Internal`] if `k == 0`.
    /// - [`PeftError::WeightCountMismatch`] if `k > num_sources`.
    /// - Propagates errors from [`Self::attention_weights`].
    pub fn route_top_k(&self, input_repr: &[f32], k: usize) -> PeftResult<SoftPrompt> {
        if k == 0 {
            return Err(PeftError::Internal {
                msg: "k must be >= 1 for route_top_k".to_string(),
            });
        }
        if k > self.num_sources {
            return Err(PeftError::WeightCountMismatch {
                weights: k,
                adapters: self.num_sources,
            });
        }

        let mut alpha = self.attention_weights(input_repr)?;

        // Sort indices by descending weight (softmax is monotone so this is also
        // descending score order)
        let mut indices: Vec<usize> = (0..self.num_sources).collect();
        indices.sort_by(|&a, &b| {
            alpha[b]
                .partial_cmp(&alpha[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Zero out all but top-k
        for &idx in indices.iter().skip(k) {
            alpha[idx] = 0.0;
        }

        // Re-normalize
        let sum: f32 = alpha.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for w in alpha.iter_mut() {
                *w *= inv;
            }
        }

        // Weighted sum
        let prompt_size = self.cfg.num_tokens * self.cfg.prompt_dim;
        let mut mix = vec![0.0_f32; prompt_size];
        for (k_idx, &w) in alpha.iter().enumerate() {
            if w == 0.0 {
                continue;
            }
            for (mix_j, &p_j) in mix.iter_mut().zip(self.source_prompts[k_idx].iter()) {
                *mix_j += w * p_j;
            }
        }

        Ok(SoftPrompt {
            num_tokens: self.cfg.num_tokens,
            embed_dim: self.cfg.prompt_dim,
            embeddings: mix,
        })
    }

    /// Number of source prompts held in this router.
    #[inline]
    #[must_use]
    pub fn num_sources(&self) -> usize {
        self.num_sources
    }

    /// Total number of learnable parameters:
    /// `key_dim * prompt_dim + num_sources * key_dim + num_sources * num_tokens * prompt_dim`.
    #[inline]
    #[must_use]
    pub fn total_params(&self) -> usize {
        self.cfg.key_dim * self.cfg.prompt_dim
            + self.num_sources * self.cfg.key_dim
            + self.num_sources * self.cfg.num_tokens * self.cfg.prompt_dim
    }
}

// ---------------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------------

/// Numerically-stable softmax (max-shift before exponentiation).
fn softmax_stable(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = out.iter().sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in out.iter_mut() {
        *v *= inv;
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn handle(seed: u64) -> PeftHandle {
        PeftHandle::new(80, seed)
    }

    fn cfg(
        num_tokens: usize,
        prompt_dim: usize,
        key_dim: usize,
        temperature: f32,
    ) -> AttemptConfig {
        AttemptConfig {
            num_tokens,
            prompt_dim,
            key_dim,
            temperature,
        }
    }

    // ── Test 1: num_sources = 0 → error ────────────────────────────────────
    #[test]
    fn zero_num_sources_errors() {
        let mut h = handle(1);
        let res = AttemptRouter::new(cfg(4, 8, 4, 1.0), 0, &mut h);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for num_sources=0, got {:?}",
            res
        );
    }

    // ── Test 2: temperature = 0 → error ────────────────────────────────────
    #[test]
    fn non_positive_temperature_errors() {
        let mut h = handle(2);
        let res = AttemptRouter::new(cfg(4, 8, 4, 0.0), 3, &mut h);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for temperature=0, got {:?}",
            res
        );
    }

    // ── Test 3: negative temperature → error ───────────────────────────────
    #[test]
    fn negative_temperature_errors() {
        let mut h = handle(3);
        let res = AttemptRouter::new(cfg(4, 8, 4, -1.0), 2, &mut h);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for temperature<0, got {:?}",
            res
        );
    }

    // ── Test 4: single source → route returns that source's prompt exactly ──
    #[test]
    fn single_source_route_returns_source_prompt() {
        let mut h = handle(4);
        let router = AttemptRouter::new(cfg(2, 4, 2, 1.0), 1, &mut h).unwrap();
        let source_prompt = router.source_prompts[0].clone();
        let input_repr = vec![1.0_f32, 0.0, 0.0, 0.0];
        let result = router.route(&input_repr).unwrap();
        for (r, &p) in result.embeddings.iter().zip(source_prompt.iter()) {
            assert!((r - p).abs() < 1e-5, "single source: got {r} expected {p}");
        }
    }

    // ── Test 5: identical source_keys → weights nearly uniform ─────────────
    #[test]
    fn identical_source_keys_uniform_weights() {
        let num_sources = 4;
        let num_tokens = 2;
        let prompt_dim = 4;
        let key_dim = 2;
        let mut h = handle(5);
        let mut router = AttemptRouter::new(
            cfg(num_tokens, prompt_dim, key_dim, 1.0),
            num_sources,
            &mut h,
        )
        .unwrap();
        // Force all source keys to be identical
        let common_key = vec![0.5_f32; key_dim];
        for k in 0..num_sources {
            let base = k * key_dim;
            router.source_keys[base..base + key_dim].copy_from_slice(&common_key);
        }
        let input_repr = vec![1.0_f32; prompt_dim];
        let weights = router.attention_weights(&input_repr).unwrap();
        let expected = 1.0 / num_sources as f32;
        for &w in &weights {
            assert!(
                (w - expected).abs() < 1e-5,
                "expected uniform weight {expected}, got {w}"
            );
        }
    }

    // ── Test 6: attention_weights sum ≈ 1.0 ────────────────────────────────
    #[test]
    fn attention_weights_sum_to_one() {
        let mut h = handle(6);
        let router = AttemptRouter::new(cfg(3, 8, 4, 1.0), 5, &mut h).unwrap();
        let input_repr = vec![0.3_f32; 8];
        let weights = router.attention_weights(&input_repr).unwrap();
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "attention weights should sum to 1, got {sum}"
        );
    }

    // ── Test 7: route output length ─────────────────────────────────────────
    #[test]
    fn route_output_length_correct() {
        let num_tokens = 4;
        let prompt_dim = 8;
        let mut h = handle(7);
        let router = AttemptRouter::new(cfg(num_tokens, prompt_dim, 4, 1.0), 3, &mut h).unwrap();
        let input_repr = vec![0.0_f32; prompt_dim];
        let result = router.route(&input_repr).unwrap();
        assert_eq!(result.embeddings.len(), num_tokens * prompt_dim);
        assert_eq!(result.num_tokens, num_tokens);
        assert_eq!(result.embed_dim, prompt_dim);
    }

    // ── Test 8: route_top_k k=1 → single dominant source ───────────────────
    /// With strongly different source keys, top-1 should closely match the
    /// winning source prompt.
    #[test]
    fn route_top_k_one_returns_dominant_source() {
        let num_tokens = 2;
        let prompt_dim = 4;
        let key_dim = 4;
        let num_sources = 3;
        let mut h = handle(8);
        let mut router = AttemptRouter::new(
            cfg(num_tokens, prompt_dim, key_dim, 0.01), // very sharp
            num_sources,
            &mut h,
        )
        .unwrap();

        // Zero query_proj then set one entry so q[0] = input_repr[0]
        for v in router.query_proj.iter_mut() {
            *v = 0.0;
        }
        // Zero all source keys
        for v in router.source_keys.iter_mut() {
            *v = 0.0;
        }
        // source 1 gets key dimension 0 = 100 → large score when q[0] > 0
        router.source_keys[key_dim] = 100.0;
        // query_proj[0,0] = 1 so q[0] = input_repr[0]
        router.query_proj[0] = 1.0;

        // Give each source a distinct constant prompt
        let prompt_size = num_tokens * prompt_dim;
        router.source_prompts[0] = vec![0.0_f32; prompt_size];
        router.source_prompts[1] = vec![5.0_f32; prompt_size];
        router.source_prompts[2] = vec![0.0_f32; prompt_size];

        let input_repr = vec![1.0_f32, 0.0, 0.0, 0.0]; // q[0] = 1
        let result = router.route_top_k(&input_repr, 1).unwrap();

        // Source 1 should win (score[1] = 1.0 * 100.0 / sqrt(4) / 0.01 >> 0)
        for &v in &result.embeddings {
            assert!(
                (v - 5.0).abs() < 1e-4,
                "top-1 should return source 1 prompt (5.0), got {v}"
            );
        }
    }

    // ── Test 9: route_top_k k=0 → error ────────────────────────────────────
    #[test]
    fn route_top_k_zero_errors() {
        let mut h = handle(9);
        let router = AttemptRouter::new(cfg(2, 4, 2, 1.0), 3, &mut h).unwrap();
        let res = router.route_top_k(&[0.0_f32; 4], 0);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for k=0, got {:?}",
            res
        );
    }

    // ── Test 10: route_top_k k > num_sources → error ────────────────────────
    #[test]
    fn route_top_k_exceeds_sources_errors() {
        let mut h = handle(10);
        let router = AttemptRouter::new(cfg(2, 4, 2, 1.0), 2, &mut h).unwrap();
        let res = router.route_top_k(&[0.0_f32; 4], 5);
        assert!(
            matches!(res, Err(PeftError::WeightCountMismatch { .. })),
            "expected WeightCountMismatch, got {:?}",
            res
        );
    }

    // ── Test 11: input_repr wrong dim → error ───────────────────────────────
    #[test]
    fn wrong_input_repr_dim_errors() {
        let mut h = handle(11);
        let router = AttemptRouter::new(cfg(2, 8, 4, 1.0), 3, &mut h).unwrap();
        let bad = vec![1.0_f32; 5]; // prompt_dim = 8
        let res = router.attention_weights(&bad);
        assert!(
            matches!(res, Err(PeftError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            res
        );
    }

    // ── Test 12: deterministic — same seed → same route ─────────────────────
    #[test]
    fn deterministic_same_seed() {
        let input_repr = vec![0.5_f32; 8];
        let mut h1 = handle(42);
        let mut h2 = handle(42);
        let r1 = AttemptRouter::new(cfg(3, 8, 4, 1.0), 4, &mut h1).unwrap();
        let r2 = AttemptRouter::new(cfg(3, 8, 4, 1.0), 4, &mut h2).unwrap();
        let out1 = r1.route(&input_repr).unwrap();
        let out2 = r2.route(&input_repr).unwrap();
        for (v1, v2) in out1.embeddings.iter().zip(out2.embeddings.iter()) {
            assert_eq!(v1, v2, "determinism failed: {v1} vs {v2}");
        }
    }

    // ── Test 13: total_params formula ───────────────────────────────────────
    #[test]
    fn total_params_formula_correct() {
        let num_sources = 3;
        let num_tokens = 5;
        let prompt_dim = 8;
        let key_dim = 4;
        let mut h = handle(13);
        let router = AttemptRouter::new(
            cfg(num_tokens, prompt_dim, key_dim, 1.0),
            num_sources,
            &mut h,
        )
        .unwrap();
        let expected =
            key_dim * prompt_dim + num_sources * key_dim + num_sources * num_tokens * prompt_dim;
        assert_eq!(router.total_params(), expected);
    }

    // ── Test 14: large temperature → nearly uniform weights ─────────────────
    #[test]
    fn large_temperature_nearly_uniform() {
        let num_sources = 4;
        let mut h = handle(14);
        let router = AttemptRouter::new(
            cfg(2, 4, 4, 1000.0), // very large temperature
            num_sources,
            &mut h,
        )
        .unwrap();
        let input_repr = vec![1.0_f32; 4];
        let weights = router.attention_weights(&input_repr).unwrap();
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_w = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let ratio = max_w / min_w.max(1e-10);
        assert!(
            ratio < 2.0,
            "with large temperature, weights should be nearly uniform, max/min ratio={ratio}"
        );
    }

    // ── Test 15: small temperature → one weight dominates ───────────────────
    #[test]
    fn small_temperature_dominant_weight() {
        let num_sources = 4;
        let mut h = handle(15);
        let router = AttemptRouter::new(
            cfg(2, 4, 4, 0.001), // very small temperature
            num_sources,
            &mut h,
        )
        .unwrap();
        let input_repr = vec![1.0_f32; 4];
        let weights = router.attention_weights(&input_repr).unwrap();
        let max_w = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_w > 0.9,
            "with tiny temperature, one weight should dominate (>0.9), max={max_w}"
        );
    }

    // ── Test 16: two identical source prompts → route = that prompt ──────────
    #[test]
    fn two_identical_source_prompts_route_equals_prompt() {
        let num_tokens = 2;
        let prompt_dim = 4;
        let prompt_size = num_tokens * prompt_dim;
        let mut h = handle(16);
        let mut router =
            AttemptRouter::new(cfg(num_tokens, prompt_dim, 4, 1.0), 2, &mut h).unwrap();
        // Force both source prompts to be the same constant
        let common: Vec<f32> = (0..prompt_size).map(|i| i as f32 * 0.1).collect();
        router.source_prompts[0] = common.clone();
        router.source_prompts[1] = common.clone();

        let input_repr = vec![0.3_f32; prompt_dim];
        let result = router.route(&input_repr).unwrap();
        for (r, &p) in result.embeddings.iter().zip(common.iter()) {
            assert!(
                (r - p).abs() < 1e-5,
                "identical source prompts: got {r}, expected {p}"
            );
        }
    }

    // ── Test 17: num_sources() accessor ─────────────────────────────────────
    #[test]
    fn num_sources_accessor() {
        let mut h = handle(17);
        let router = AttemptRouter::new(cfg(2, 4, 2, 1.0), 7, &mut h).unwrap();
        assert_eq!(router.num_sources(), 7);
    }

    // ── Test 18: route_top_k k=num_sources matches route ────────────────────
    #[test]
    fn route_top_k_all_matches_route() {
        let num_sources = 4;
        let mut h = handle(18);
        let router = AttemptRouter::new(cfg(3, 8, 4, 1.0), num_sources, &mut h).unwrap();
        let input_repr = vec![0.5_f32; 8];
        let full = router.route(&input_repr).unwrap();
        let top_k = router.route_top_k(&input_repr, num_sources).unwrap();
        for (a, b) in full.embeddings.iter().zip(top_k.embeddings.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "top_k(k=num_sources) should match route, {a} vs {b}"
            );
        }
    }
}
