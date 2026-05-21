//! SPoT (Soft Prompt Transfer) — Vu-Lester-Briskie-Liu-Chaturvedi-Iyyer 2022 ACL.
//!
//! Reference: Vu T, Lester B, Briskie O, Salemi A, Barua R, Gupta S, Chaturvedi S,
//! Iyyer M (2022) "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer",
//! ACL 2022: 4643–4663. <https://arxiv.org/abs/2110.07904>
//!
//! SPoT solves cross-task prompt initialization: given a library of source-task
//! soft prompts and their corresponding task embeddings, it retrieves a weighted
//! mixture of the most similar source prompts (by cosine similarity) to initialize
//! a target-task prompt. The result is a `SoftPrompt` ready for fine-tuning.
//!
//! ## Algorithm (§3 of Vu et al. 2022)
//!
//! 1. Compute cosine similarities between the target task embedding and each
//!    source task embedding.
//! 2. Apply temperature-scaled softmax to obtain retrieval weights.
//! 3. Form the transferred prompt as a weighted sum of source prompts.
//!
//! Optional: restrict to top-`k` sources before re-normalizing.

use super::prompt_tuning::SoftPrompt;
use crate::error::{PeftError, PeftResult};

/// Configuration for [`SoftPromptLibrary`].
#[derive(Debug, Clone)]
pub struct SpotConfig {
    /// Number of virtual prompt tokens.
    pub num_tokens: usize,
    /// Dimension of each prompt token embedding.
    pub prompt_embed_dim: usize,
    /// Dimension of the task embedding vectors.
    pub task_embed_dim: usize,
    /// Softmax temperature `τ > 0`; lower values sharpen the distribution.
    pub temperature: f32,
}

/// A single source task with its task embedding and pre-trained soft prompt.
#[derive(Debug, Clone)]
pub struct SourceTask {
    /// Task embedding vector, shape `task_embed_dim`.
    pub task_embedding: Vec<f32>,
    /// Soft prompt embeddings, flat row-major shape `num_tokens × prompt_embed_dim`.
    pub prompt: Vec<f32>,
    /// Human-readable task identifier.
    pub task_id: String,
}

/// A library of source-task soft prompts used to initialize target prompts via
/// cosine-similarity retrieval.
#[derive(Debug)]
pub struct SoftPromptLibrary {
    /// Source tasks in the library.
    pub(crate) sources: Vec<SourceTask>,
    /// Configuration.
    pub cfg: SpotConfig,
}

impl SoftPromptLibrary {
    /// Build a library from a non-empty list of source tasks.
    ///
    /// # Errors
    ///
    /// - [`PeftError::Internal`] if `sources` is empty.
    /// - [`PeftError::DimensionMismatch`] if any source task embedding length
    ///   ≠ `cfg.task_embed_dim` or any source prompt length
    ///   ≠ `cfg.num_tokens × cfg.prompt_embed_dim`.
    pub fn new(cfg: SpotConfig, sources: Vec<SourceTask>) -> PeftResult<Self> {
        if sources.is_empty() {
            return Err(PeftError::Internal {
                msg: "empty source library".to_string(),
            });
        }
        let expected_prompt_len = cfg.num_tokens * cfg.prompt_embed_dim;
        for src in &sources {
            if src.task_embedding.len() != cfg.task_embed_dim {
                return Err(PeftError::DimensionMismatch {
                    expected: cfg.task_embed_dim,
                    got: src.task_embedding.len(),
                });
            }
            if src.prompt.len() != expected_prompt_len {
                return Err(PeftError::DimensionMismatch {
                    expected: expected_prompt_len,
                    got: src.prompt.len(),
                });
            }
        }
        Ok(Self { sources, cfg })
    }

    /// Compute cosine-similarity softmax retrieval weights for the target task.
    ///
    /// Returns a `Vec<f32>` of length `num_sources`, summing to 1.0.
    ///
    /// # Errors
    ///
    /// - [`PeftError::DimensionMismatch`] if `target.len() ≠ task_embed_dim`.
    /// - [`PeftError::Internal`] if `temperature ≤ 0`.
    pub fn similarity_weights(&self, target_task_embedding: &[f32]) -> PeftResult<Vec<f32>> {
        if target_task_embedding.len() != self.cfg.task_embed_dim {
            return Err(PeftError::DimensionMismatch {
                expected: self.cfg.task_embed_dim,
                got: target_task_embedding.len(),
            });
        }
        if self.cfg.temperature <= 0.0 || self.cfg.temperature.is_nan() {
            return Err(PeftError::Internal {
                msg: format!(
                    "SPoT temperature must be > 0 and finite, got {}",
                    self.cfg.temperature
                ),
            });
        }

        let target = target_task_embedding;
        let norm_target = l2_norm(target);

        let mut scores = Vec::with_capacity(self.sources.len());
        for src in &self.sources {
            let dot = dot_product(target, &src.task_embedding);
            let norm_src = l2_norm(&src.task_embedding);
            let cosine = if norm_target < 1e-10 || norm_src < 1e-10 {
                0.0_f32
            } else {
                dot / (norm_target * norm_src)
            };
            scores.push(cosine / self.cfg.temperature);
        }

        Ok(softmax_max_shift(&scores))
    }

    /// Compute a transfer-initialized `SoftPrompt` as a weighted sum of all
    /// source prompts, using cosine-similarity softmax weights.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::similarity_weights`].
    pub fn initialize_target(&self, target_task_embedding: &[f32]) -> PeftResult<SoftPrompt> {
        let weights = self.similarity_weights(target_task_embedding)?;
        let prompt_len = self.cfg.num_tokens * self.cfg.prompt_embed_dim;
        let prompt_data = weighted_sum_prompts(&self.sources, &weights, prompt_len);
        Ok(SoftPrompt {
            num_tokens: self.cfg.num_tokens,
            embed_dim: self.cfg.prompt_embed_dim,
            embeddings: prompt_data,
        })
    }

    /// Transfer-initialize using only the top-`k` most similar source prompts.
    ///
    /// The weights of the non-top-k sources are zeroed out and the remaining
    /// weights are re-normalized to sum to 1.
    ///
    /// # Errors
    ///
    /// - [`PeftError::Internal`] if `k == 0`.
    /// - [`PeftError::WeightCountMismatch`] if `k > num_sources`.
    /// - Propagates errors from [`Self::similarity_weights`].
    pub fn top_k_initialize(
        &self,
        target_task_embedding: &[f32],
        k: usize,
    ) -> PeftResult<SoftPrompt> {
        if k == 0 {
            return Err(PeftError::Internal {
                msg: "k must be >= 1".to_string(),
            });
        }
        let n = self.sources.len();
        if k > n {
            return Err(PeftError::WeightCountMismatch {
                weights: k,
                adapters: n,
            });
        }

        let mut weights = self.similarity_weights(target_task_embedding)?;

        // Sort indices by descending cosine similarity score (i.e., descending
        // softmax weight; the softmax is monotone so rank order is preserved).
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Zero out all but top-k.
        for &idx in indices.iter().skip(k) {
            weights[idx] = 0.0;
        }

        // Re-normalize the top-k weights.
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for w in weights.iter_mut() {
                *w *= inv;
            }
        }

        let prompt_len = self.cfg.num_tokens * self.cfg.prompt_embed_dim;
        let prompt_data = weighted_sum_prompts(&self.sources, &weights, prompt_len);
        Ok(SoftPrompt {
            num_tokens: self.cfg.num_tokens,
            embed_dim: self.cfg.prompt_embed_dim,
            embeddings: prompt_data,
        })
    }

    /// Return the number of source tasks in the library.
    #[must_use]
    pub fn num_sources(&self) -> usize {
        self.sources.len()
    }
}

// ---------------------------------------------------------------------------
// internal helpers
// ---------------------------------------------------------------------------

/// L2 norm of a slice.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Dot product of two equal-length slices.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Numerically-stable softmax with max shift.
fn softmax_max_shift(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = vec![0.0_f32; scores.len()];
    let mut sum = 0.0_f32;
    for (slot, &s) in out.iter_mut().zip(scores.iter()) {
        let e = (s - m).exp();
        *slot = e;
        sum += e;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for slot in out.iter_mut() {
        *slot *= inv;
    }
    out
}

/// Compute `Σ_i weights[i] * sources[i].prompt` (element-wise weighted sum).
fn weighted_sum_prompts(sources: &[SourceTask], weights: &[f32], prompt_len: usize) -> Vec<f32> {
    let mut result = vec![0.0_f32; prompt_len];
    for (src, &w) in sources.iter().zip(weights.iter()) {
        if w == 0.0 {
            continue;
        }
        for (r, &p) in result.iter_mut().zip(src.prompt.iter()) {
            *r += w * p;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cfg(num_tokens: usize, embed_dim: usize, task_dim: usize, temp: f32) -> SpotConfig {
        SpotConfig {
            num_tokens,
            prompt_embed_dim: embed_dim,
            task_embed_dim: task_dim,
            temperature: temp,
        }
    }

    fn make_source(task_emb: Vec<f32>, prompt: Vec<f32>, id: &str) -> SourceTask {
        SourceTask {
            task_embedding: task_emb,
            prompt,
            task_id: id.to_string(),
        }
    }

    // ── Test 1 ─────────────────────────────────────────────────────────────
    /// Empty source library → error.
    #[test]
    fn empty_sources_errors() {
        let cfg = make_cfg(4, 8, 3, 1.0);
        let res = SoftPromptLibrary::new(cfg, vec![]);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal, got {:?}",
            res
        );
    }

    // ── Test 2 ─────────────────────────────────────────────────────────────
    /// Single source with unit task embedding → `initialize_target` returns
    /// exactly that source's prompt (weight ≈ 1.0).
    #[test]
    fn single_source_returns_exact_prompt() {
        let task_emb = vec![1.0_f32, 0.0, 0.0];
        let prompt: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let cfg = make_cfg(2, 4, 3, 1.0);
        let src = make_source(task_emb.clone(), prompt.clone(), "task_a");
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();

        let result = lib.initialize_target(&task_emb).unwrap();
        assert_eq!(result.embeddings.len(), 8);
        for (r, p) in result.embeddings.iter().zip(prompt.iter()) {
            assert!(
                (r - p).abs() < 1e-5,
                "single source result mismatch: {r} vs {p}"
            );
        }
    }

    // ── Test 3 ─────────────────────────────────────────────────────────────
    /// Two identical task embeddings → each weight ≈ 0.5.
    #[test]
    fn two_identical_task_embeddings_equal_weights() {
        let task_emb = vec![0.6_f32, 0.8, 0.0];
        let prompt_a = vec![1.0_f32; 6];
        let prompt_b = vec![0.0_f32; 6];
        let cfg = make_cfg(2, 3, 3, 1.0);
        let src_a = make_source(task_emb.clone(), prompt_a, "a");
        let src_b = make_source(task_emb.clone(), prompt_b, "b");
        let lib = SoftPromptLibrary::new(cfg, vec![src_a, src_b]).unwrap();

        let weights = lib.similarity_weights(&task_emb).unwrap();
        assert_eq!(weights.len(), 2);
        assert!(
            (weights[0] - 0.5).abs() < 1e-5,
            "expected w[0]≈0.5, got {}",
            weights[0]
        );
        assert!(
            (weights[1] - 0.5).abs() < 1e-5,
            "expected w[1]≈0.5, got {}",
            weights[1]
        );
    }

    // ── Test 4 ─────────────────────────────────────────────────────────────
    /// `target.len()` mismatch → `DimensionMismatch`.
    #[test]
    fn target_dim_mismatch_errors() {
        let cfg = make_cfg(2, 4, 3, 1.0);
        let src = make_source(vec![1.0, 0.0, 0.0], vec![0.0_f32; 8], "x");
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();

        let bad_target = vec![1.0_f32; 5]; // task_embed_dim = 3
        let res = lib.similarity_weights(&bad_target);
        assert!(
            matches!(res, Err(PeftError::DimensionMismatch { .. })),
            "expected DimensionMismatch, got {:?}",
            res
        );
    }

    // ── Test 5 ─────────────────────────────────────────────────────────────
    /// `num_sources()` returns the correct count.
    #[test]
    fn num_sources_correct() {
        let cfg = make_cfg(2, 4, 3, 1.0);
        let srcs: Vec<SourceTask> = (0..5)
            .map(|i| make_source(vec![i as f32, 0.0, 0.0], vec![0.0_f32; 8], &format!("t{i}")))
            .collect();
        let lib = SoftPromptLibrary::new(cfg, srcs).unwrap();
        assert_eq!(lib.num_sources(), 5);
    }

    // ── Test 6 ─────────────────────────────────────────────────────────────
    /// `initialize_target` output has length `num_tokens × prompt_embed_dim`.
    #[test]
    fn initialize_target_output_length() {
        let num_tokens = 4;
        let embed_dim = 6;
        let task_dim = 3;
        let cfg = make_cfg(num_tokens, embed_dim, task_dim, 1.0);
        let src = make_source(
            vec![1.0, 0.0, 0.0],
            vec![0.5_f32; num_tokens * embed_dim],
            "src",
        );
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();
        let result = lib.initialize_target(&[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(result.embeddings.len(), num_tokens * embed_dim);
        assert_eq!(result.num_tokens, num_tokens);
        assert_eq!(result.embed_dim, embed_dim);
    }

    // ── Test 7 ─────────────────────────────────────────────────────────────
    /// `temperature = 0.0` → error.
    #[test]
    fn zero_temperature_errors() {
        let cfg = make_cfg(2, 4, 3, 0.0);
        let src = make_source(vec![1.0, 0.0, 0.0], vec![0.0_f32; 8], "x");
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();
        let res = lib.similarity_weights(&[1.0, 0.0, 0.0]);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for temperature=0, got {:?}",
            res
        );
    }

    // ── Test 8 ─────────────────────────────────────────────────────────────
    /// `top_k_initialize(k=1)` → the most similar source dominates (weight → 1).
    #[test]
    fn top_k_one_argmax_source_dominates() {
        let task_dim = 3;
        let cfg = make_cfg(2, 3, task_dim, 0.1); // low temperature sharpens
        // Source A is much more similar to target than source B.
        let target = vec![1.0_f32, 0.0, 0.0];
        let src_a = make_source(vec![1.0_f32, 0.0, 0.0], vec![2.0_f32; 6], "a");
        let src_b = make_source(vec![0.0_f32, 1.0, 0.0], vec![0.0_f32; 6], "b");
        let lib = SoftPromptLibrary::new(cfg, vec![src_a, src_b]).unwrap();

        let result = lib.top_k_initialize(&target, 1).unwrap();
        // Source A has cosine=1 with target, source B has cosine=0.
        // top-1 must be A, so prompt ≈ [2, 2, 2, 2, 2, 2].
        for &v in &result.embeddings {
            assert!(
                (v - 2.0).abs() < 1e-5,
                "expected 2.0 (source A prompt), got {v}"
            );
        }
    }

    // ── Test 9 ─────────────────────────────────────────────────────────────
    /// `top_k_initialize(k = num_sources)` matches `initialize_target`.
    #[test]
    fn top_k_all_sources_matches_initialize_target() {
        let task_dim = 4;
        let cfg = make_cfg(3, 5, task_dim, 1.0);
        let target = vec![0.5_f32, 0.3, 0.1, 0.0];
        let srcs: Vec<SourceTask> = (0..4)
            .map(|i| {
                let emb: Vec<f32> = (0..task_dim)
                    .map(|j| if j == i { 1.0 } else { 0.0 })
                    .collect();
                make_source(emb, vec![i as f32 * 0.1; 15], &format!("s{i}"))
            })
            .collect();
        let lib = SoftPromptLibrary::new(cfg, srcs).unwrap();

        let full = lib.initialize_target(&target).unwrap();
        let top_k = lib.top_k_initialize(&target, 4).unwrap();

        for (a, b) in full.embeddings.iter().zip(top_k.embeddings.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "top_k(k=num_sources) differs from initialize_target: {a} vs {b}"
            );
        }
    }

    // ── Test 10 ────────────────────────────────────────────────────────────
    /// `top_k_initialize(k > num_sources)` → error.
    #[test]
    fn top_k_exceeds_num_sources_errors() {
        let cfg = make_cfg(2, 4, 3, 1.0);
        let src = make_source(vec![1.0, 0.0, 0.0], vec![0.0_f32; 8], "x");
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();
        let res = lib.top_k_initialize(&[1.0, 0.0, 0.0], 5);
        assert!(
            matches!(res, Err(PeftError::WeightCountMismatch { .. })),
            "expected WeightCountMismatch, got {:?}",
            res
        );
    }

    // ── Test 11 ────────────────────────────────────────────────────────────
    /// `top_k_initialize(k = 0)` → error.
    #[test]
    fn top_k_zero_errors() {
        let cfg = make_cfg(2, 4, 3, 1.0);
        let src = make_source(vec![1.0, 0.0, 0.0], vec![0.0_f32; 8], "x");
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();
        let res = lib.top_k_initialize(&[1.0, 0.0, 0.0], 0);
        assert!(
            matches!(res, Err(PeftError::Internal { .. })),
            "expected Internal for k=0, got {:?}",
            res
        );
    }

    // ── Test 12 ────────────────────────────────────────────────────────────
    /// Source prompt with wrong length in `new()` → `DimensionMismatch`.
    #[test]
    fn source_prompt_dim_mismatch_in_new() {
        let cfg = make_cfg(2, 4, 3, 1.0); // expects prompt len = 2*4 = 8
        let bad_src = make_source(vec![1.0, 0.0, 0.0], vec![0.0_f32; 5], "bad");
        let res = SoftPromptLibrary::new(cfg, vec![bad_src]);
        assert!(
            matches!(res, Err(PeftError::DimensionMismatch { .. })),
            "expected DimensionMismatch for bad prompt length, got {:?}",
            res
        );
    }

    // ── Test 13 ────────────────────────────────────────────────────────────
    /// Source task embedding with wrong length in `new()` → `DimensionMismatch`.
    #[test]
    fn source_task_emb_dim_mismatch_in_new() {
        let cfg = make_cfg(2, 4, 3, 1.0); // expects task_embed_dim = 3
        let bad_src = make_source(vec![1.0_f32, 0.0], vec![0.0_f32; 8], "bad");
        let res = SoftPromptLibrary::new(cfg, vec![bad_src]);
        assert!(
            matches!(res, Err(PeftError::DimensionMismatch { .. })),
            "expected DimensionMismatch for bad task embedding, got {:?}",
            res
        );
    }

    // ── Test 14 ────────────────────────────────────────────────────────────
    /// Weights from `similarity_weights` sum to ≈ 1.0.
    #[test]
    fn similarity_weights_sum_to_one() {
        let task_dim = 4;
        let cfg = make_cfg(2, 3, task_dim, 1.0);
        let srcs: Vec<SourceTask> = (0..6)
            .map(|i| {
                let emb: Vec<f32> = (0..task_dim)
                    .map(|j| if j == i % task_dim { 1.0 } else { 0.0 })
                    .collect();
                make_source(emb, vec![0.0_f32; 6], &format!("t{i}"))
            })
            .collect();
        let lib = SoftPromptLibrary::new(cfg, srcs).unwrap();

        let target = vec![0.3_f32, 0.7, 0.1, 0.9];
        let weights = lib.similarity_weights(&target).unwrap();
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax weights should sum to 1, got {sum}"
        );
    }

    // ── Test 15 ────────────────────────────────────────────────────────────
    /// Different target embeddings produce different weights.
    #[test]
    fn different_targets_give_different_weights() {
        let task_dim = 3;
        let cfg = make_cfg(2, 4, task_dim, 1.0);
        let src_a = make_source(vec![1.0_f32, 0.0, 0.0], vec![0.0_f32; 8], "a");
        let src_b = make_source(vec![0.0_f32, 1.0, 0.0], vec![1.0_f32; 8], "b");
        let lib = SoftPromptLibrary::new(cfg, vec![src_a, src_b]).unwrap();

        let target1 = vec![1.0_f32, 0.0, 0.0];
        let target2 = vec![0.0_f32, 1.0, 0.0];

        let w1 = lib.similarity_weights(&target1).unwrap();
        let w2 = lib.similarity_weights(&target2).unwrap();

        // The two weight vectors should differ substantially.
        let diff: f32 = w1.iter().zip(w2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.1,
            "different targets should produce different weights, diff = {diff}"
        );
    }

    // ── Test 16 ────────────────────────────────────────────────────────────
    /// Near-zero target norm → cosine treated as 0 (no divide-by-zero panic).
    #[test]
    fn near_zero_target_norm_no_panic() {
        let task_dim = 3;
        let cfg = make_cfg(2, 3, task_dim, 1.0);
        let src = make_source(vec![1.0_f32, 0.0, 0.0], vec![0.5_f32; 6], "x");
        let lib = SoftPromptLibrary::new(cfg, vec![src]).unwrap();

        // Near-zero target → cosine = 0 → all weights equal.
        let tiny = vec![1e-12_f32; task_dim];
        let weights = lib.similarity_weights(&tiny).unwrap();
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "weights should still sum to 1 with tiny target, sum={sum}"
        );
    }
}
