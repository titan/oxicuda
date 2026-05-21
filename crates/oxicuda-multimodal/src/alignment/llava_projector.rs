//! LLaVA-style visual projector.
//!
//! Reference: Liu, Li, Wu, Lee 2023, *Visual Instruction Tuning* (LLaVA).
//!
//! LLaVA bridges a frozen CLIP visual encoder and a frozen large language
//! model (LLM) by inserting a small trainable projector `g_\theta` that maps
//! the CLIP visual feature space `\R^{vision\_dim}` to the LLM's text
//! embedding space `\R^{llm\_dim}`. The projected vectors are then fed to the
//! LLM as if they were ordinary text-token embeddings, allowing the language
//! model to attend over visual content with no architectural changes.
//!
//! In the original paper the projector is a single linear layer
//! (`mlp_depth = 1`); subsequent LLaVA-1.5 work replaces it with a two-layer
//! MLP using GELU (`mlp_depth = 2`), which is the de-facto baseline today.
//! This module supports both and any deeper stack by parameterising the depth.
//!
//! ```text
//!   visual_token [vision_dim]
//!         │
//!         ▼  linear(vision_dim → hidden_dim)     ── if mlp_depth ≥ 2
//!         ▼  GELU
//!         ▼  linear(hidden_dim → hidden_dim)     ── repeated (mlp_depth − 2) times
//!         ▼  GELU                                ── (each followed by GELU)
//!         ▼  linear(hidden_dim → llm_dim)        ── (always the last layer)
//!   projected_token [llm_dim]
//! ```
//!
//! When `mlp_depth = 1` the projector collapses to a single linear layer
//! mapping `vision_dim → llm_dim` (no GELU and no hidden state).

use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the LLaVA visual projector.
#[derive(Debug, Clone)]
pub struct LlavaProjectorConfig {
    /// Dimensionality of the CLIP visual feature tokens.
    pub vision_dim: usize,
    /// Dimensionality of the LLM's text-embedding space.
    pub llm_dim: usize,
    /// Hidden dimensionality between successive MLP layers
    /// (used only when `mlp_depth ≥ 2`).
    pub hidden_dim: usize,
    /// Number of linear layers in the projector. `1` is a plain linear
    /// projection (LLaVA original); `2` is the LLaVA-1.5 MLP with GELU;
    /// deeper values stack additional hidden GELU layers.
    pub mlp_depth: usize,
}

impl LlavaProjectorConfig {
    /// Tiny preset for unit testing: `vision_dim=8`, `llm_dim=16`,
    /// `hidden_dim=12`, `mlp_depth=2`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            vision_dim: 8,
            llm_dim: 16,
            hidden_dim: 12,
            mlp_depth: 2,
        }
    }

    /// Validate the configuration.
    fn validate(&self) -> MmResult<()> {
        if self.vision_dim == 0 || self.llm_dim == 0 || self.hidden_dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.mlp_depth == 0 {
            return Err(MultiModalError::InvalidLayerCount);
        }
        Ok(())
    }
}

// ─── Layer shapes ────────────────────────────────────────────────────────────

/// Return `(in_dim, out_dim)` for the `layer_index`-th linear layer of a
/// projector with the given configuration.
///
/// Layer 0 starts at `vision_dim`, the last layer ends at `llm_dim`, and
/// intermediate layers all use `hidden_dim`.
fn layer_shape(cfg: &LlavaProjectorConfig, layer_index: usize) -> (usize, usize) {
    let last = cfg.mlp_depth - 1;
    if cfg.mlp_depth == 1 {
        (cfg.vision_dim, cfg.llm_dim)
    } else if layer_index == 0 {
        (cfg.vision_dim, cfg.hidden_dim)
    } else if layer_index == last {
        (cfg.hidden_dim, cfg.llm_dim)
    } else {
        (cfg.hidden_dim, cfg.hidden_dim)
    }
}

// ─── Projector ───────────────────────────────────────────────────────────────

/// LLaVA visual projector — an `mlp_depth`-layer MLP with GELU activations
/// between layers, mapping CLIP visual feature tokens to the LLM's text
/// embedding space.
#[derive(Debug, Clone)]
pub struct LlavaProjector {
    /// Per-layer weight matrices, each `[in_dim × out_dim]` row-major.
    weights: Vec<Vec<f32>>,
    /// Per-layer bias vectors, each `[out_dim]`.
    biases: Vec<Vec<f32>>,
    /// Frozen copy of the configuration.
    cfg: LlavaProjectorConfig,
}

impl LlavaProjector {
    /// Construct a new projector with weights drawn from a deterministic
    /// normal distribution (mean 0, std `1/sqrt(fan_in)`) and zero biases.
    pub fn new(cfg: LlavaProjectorConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;

        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(cfg.mlp_depth);
        let mut biases: Vec<Vec<f32>> = Vec::with_capacity(cfg.mlp_depth);

        for layer_index in 0..cfg.mlp_depth {
            let (in_dim, out_dim) = layer_shape(&cfg, layer_index);
            let scale = 1.0_f32 / (in_dim as f32).sqrt();
            let mut w = vec![0.0_f32; in_dim * out_dim];
            rng.fill_normal(&mut w);
            for v in w.iter_mut() {
                *v *= scale;
            }
            weights.push(w);
            biases.push(vec![0.0_f32; out_dim]);
        }

        Ok(Self {
            weights,
            biases,
            cfg,
        })
    }

    /// Borrow the projector configuration.
    #[must_use]
    pub fn config(&self) -> &LlavaProjectorConfig {
        &self.cfg
    }

    /// Project a single CLIP visual token (`vision_dim`) to an LLM
    /// embedding (`llm_dim`).
    pub fn project_one(&self, visual_token: &[f32]) -> MmResult<Vec<f32>> {
        if visual_token.len() != self.cfg.vision_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.cfg.vision_dim,
                got: visual_token.len(),
            });
        }
        let mut state = visual_token.to_vec();
        let last = self.cfg.mlp_depth - 1;
        for layer_index in 0..self.cfg.mlp_depth {
            let (in_dim, out_dim) = layer_shape(&self.cfg, layer_index);
            let weight = match self.weights.get(layer_index) {
                Some(w) => w,
                None => {
                    return Err(MultiModalError::Internal(format!(
                        "llava projector: missing weight at layer {layer_index}",
                    )));
                }
            };
            let bias = match self.biases.get(layer_index) {
                Some(b) => b,
                None => {
                    return Err(MultiModalError::Internal(format!(
                        "llava projector: missing bias at layer {layer_index}",
                    )));
                }
            };
            let mut next = vec![0.0_f32; out_dim];
            for o in 0..out_dim {
                let mut acc = match bias.get(o) {
                    Some(b) => *b,
                    None => 0.0_f32,
                };
                for i in 0..in_dim {
                    let w_io = match weight.get(i * out_dim + o) {
                        Some(w) => *w,
                        None => 0.0_f32,
                    };
                    let s_i = match state.get(i) {
                        Some(s) => *s,
                        None => 0.0_f32,
                    };
                    acc += s_i * w_io;
                }
                if layer_index < last {
                    next[o] = gelu_tanh(acc);
                } else {
                    next[o] = acc;
                }
            }
            state = next;
        }
        if state.len() != self.cfg.llm_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: self.cfg.llm_dim,
                got: state.len(),
            });
        }
        Ok(state)
    }

    /// Project `n_tokens` CLIP visual tokens (`n_tokens × vision_dim`) to
    /// `n_tokens × llm_dim` LLM embeddings (row-major).
    pub fn project_tokens(&self, visual_tokens: &[f32], n_tokens: usize) -> MmResult<Vec<f32>> {
        if n_tokens == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if visual_tokens.len() != n_tokens * self.cfg.vision_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_tokens * self.cfg.vision_dim,
                got: visual_tokens.len(),
            });
        }
        let mut out = Vec::with_capacity(n_tokens * self.cfg.llm_dim);
        for i in 0..n_tokens {
            let start = i * self.cfg.vision_dim;
            let end = start + self.cfg.vision_dim;
            let projected = self.project_one(&visual_tokens[start..end])?;
            out.extend_from_slice(&projected);
        }
        Ok(out)
    }

    /// Total number of learnable parameters (all weights + all biases).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let mut total = 0_usize;
        for layer_index in 0..self.cfg.mlp_depth {
            let (in_dim, out_dim) = layer_shape(&self.cfg, layer_index);
            total += in_dim * out_dim + out_dim;
        }
        total
    }
}

// ─── Activation ──────────────────────────────────────────────────────────────

/// Approximate GELU activation using the `tanh` approximation.
///
/// `GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))`.
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    let k = 0.044_715_f32;
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let inner = c * (x + k * x.powi(3));
    0.5 * x * (1.0 + inner.tanh())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_proj(seed: u64, cfg: LlavaProjectorConfig) -> LlavaProjector {
        let mut rng = LcgRng::new(seed);
        match LlavaProjector::new(cfg, &mut rng) {
            Ok(p) => p,
            Err(e) => panic!("llava projector should construct: {e:?}"),
        }
    }

    // ── 1: mlp_depth = 1 is a single linear with the right n_params ──────────
    #[test]
    fn depth_one_is_single_linear_n_params() {
        let cfg = LlavaProjectorConfig {
            vision_dim: 8,
            llm_dim: 16,
            hidden_dim: 12,
            mlp_depth: 1,
        };
        let proj = make_proj(1, cfg.clone());
        // single linear: vision_dim * llm_dim weights + llm_dim biases
        let expected = cfg.vision_dim * cfg.llm_dim + cfg.llm_dim;
        assert_eq!(proj.n_params(), expected);
        assert_eq!(proj.weights.len(), 1);
        assert_eq!(proj.biases.len(), 1);
        assert_eq!(proj.weights[0].len(), cfg.vision_dim * cfg.llm_dim);
        assert_eq!(proj.biases[0].len(), cfg.llm_dim);
    }

    // ── 2: mlp_depth = 2 has GELU and matches the n_params formula ──────────
    #[test]
    fn depth_two_has_gelu_and_correct_n_params() {
        let cfg = LlavaProjectorConfig {
            vision_dim: 8,
            llm_dim: 16,
            hidden_dim: 12,
            mlp_depth: 2,
        };
        let proj = make_proj(2, cfg.clone());
        // layer0: vision_dim × hidden_dim + hidden_dim
        // layer1: hidden_dim × llm_dim + llm_dim
        let expected = cfg.vision_dim * cfg.hidden_dim
            + cfg.hidden_dim
            + cfg.hidden_dim * cfg.llm_dim
            + cfg.llm_dim;
        assert_eq!(proj.n_params(), expected);
        // With a non-zero input, the GELU non-linearity must alter the
        // output relative to a pure linear pipeline (we cannot easily
        // construct an oracle, so instead we verify finiteness and that
        // a sign-symmetric input does NOT produce a sign-symmetric output,
        // which is a hallmark of a non-linear activation in between).
        let v: Vec<f32> = (0..cfg.vision_dim).map(|i| (i as f32) - 4.0).collect();
        let nv: Vec<f32> = v.iter().map(|x| -x).collect();
        let y_pos = proj.project_one(&v).expect("positive input");
        let y_neg = proj.project_one(&nv).expect("negative input");
        let mut anti_symmetric = true;
        for (a, b) in y_pos.iter().zip(y_neg.iter()) {
            if (a + b).abs() > 1e-4 {
                anti_symmetric = false;
                break;
            }
        }
        assert!(
            !anti_symmetric,
            "depth=2 with GELU must not produce a sign-symmetric output",
        );
    }

    // ── 3: project_one length equals llm_dim ────────────────────────────────
    #[test]
    fn project_one_output_length_is_llm_dim() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(3, cfg.clone());
        let v = vec![0.1_f32; cfg.vision_dim];
        let y = proj.project_one(&v).expect("project_one");
        assert_eq!(y.len(), cfg.llm_dim);
        assert!(y.iter().all(|x| x.is_finite()));
    }

    // ── 4: project_tokens length equals n_tokens * llm_dim ──────────────────
    #[test]
    fn project_tokens_output_length() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(4, cfg.clone());
        let n = 5;
        let v = vec![0.2_f32; n * cfg.vision_dim];
        let y = proj.project_tokens(&v, n).expect("project_tokens");
        assert_eq!(y.len(), n * cfg.llm_dim);
    }

    // ── 5: project_one deterministic given same seed and same input ────────
    #[test]
    fn project_one_deterministic_given_seed() {
        let cfg = LlavaProjectorConfig::tiny();
        let a = make_proj(5, cfg.clone());
        let b = make_proj(5, cfg.clone());
        let v: Vec<f32> = (0..cfg.vision_dim).map(|i| i as f32 * 0.1).collect();
        let ya = a.project_one(&v).expect("project_one a");
        let yb = b.project_one(&v).expect("project_one b");
        assert_eq!(ya, yb);
    }

    // ── 6: Changing the input changes the output ───────────────────────────
    #[test]
    fn changing_input_changes_output() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(6, cfg.clone());
        let v1: Vec<f32> = (0..cfg.vision_dim).map(|i| i as f32 * 0.1).collect();
        let v2: Vec<f32> = (0..cfg.vision_dim)
            .map(|i| (i as f32 * 0.1) + 1.0)
            .collect();
        let y1 = proj.project_one(&v1).expect("project_one y1");
        let y2 = proj.project_one(&v2).expect("project_one y2");
        let diff: f32 = y1.iter().zip(y2.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "output should respond to input: diff={diff}");
    }

    // ── 7: mlp_depth = 3 works and matches n_params ─────────────────────────
    #[test]
    fn depth_three_works_and_n_params_match() {
        let cfg = LlavaProjectorConfig {
            vision_dim: 6,
            llm_dim: 10,
            hidden_dim: 8,
            mlp_depth: 3,
        };
        let proj = make_proj(7, cfg.clone());
        // layer0: 6*8 + 8 = 56
        // layer1: 8*8 + 8 = 72
        // layer2: 8*10 + 10 = 90
        let expected = cfg.vision_dim * cfg.hidden_dim
            + cfg.hidden_dim
            + cfg.hidden_dim * cfg.hidden_dim
            + cfg.hidden_dim
            + cfg.hidden_dim * cfg.llm_dim
            + cfg.llm_dim;
        assert_eq!(proj.n_params(), expected);
        let v = vec![0.3_f32; cfg.vision_dim];
        let y = proj.project_one(&v).expect("project_one depth=3");
        assert_eq!(y.len(), cfg.llm_dim);
        assert!(y.iter().all(|x| x.is_finite()));
    }

    // ── 8: vision_dim = 0 errors ────────────────────────────────────────────
    #[test]
    fn vision_dim_zero_errors() {
        let mut rng = LcgRng::new(8);
        let cfg = LlavaProjectorConfig {
            vision_dim: 0,
            llm_dim: 16,
            hidden_dim: 12,
            mlp_depth: 2,
        };
        let err = LlavaProjector::new(cfg, &mut rng).expect_err("vision_dim=0 must err");
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 9: llm_dim = 0 errors ───────────────────────────────────────────────
    #[test]
    fn llm_dim_zero_errors() {
        let mut rng = LcgRng::new(9);
        let cfg = LlavaProjectorConfig {
            vision_dim: 8,
            llm_dim: 0,
            hidden_dim: 12,
            mlp_depth: 2,
        };
        let err = LlavaProjector::new(cfg, &mut rng).expect_err("llm_dim=0 must err");
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 10: hidden_dim = 0 errors ───────────────────────────────────────────
    #[test]
    fn hidden_dim_zero_errors() {
        let mut rng = LcgRng::new(10);
        let cfg = LlavaProjectorConfig {
            vision_dim: 8,
            llm_dim: 16,
            hidden_dim: 0,
            mlp_depth: 2,
        };
        let err = LlavaProjector::new(cfg, &mut rng).expect_err("hidden_dim=0 must err");
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 11: mlp_depth = 0 errors ────────────────────────────────────────────
    #[test]
    fn mlp_depth_zero_errors() {
        let mut rng = LcgRng::new(11);
        let cfg = LlavaProjectorConfig {
            vision_dim: 8,
            llm_dim: 16,
            hidden_dim: 12,
            mlp_depth: 0,
        };
        let err = LlavaProjector::new(cfg, &mut rng).expect_err("mlp_depth=0 must err");
        assert!(matches!(err, MultiModalError::InvalidLayerCount));
    }

    // ── 12: visual_token wrong length errors ────────────────────────────────
    #[test]
    fn visual_token_wrong_len_errors() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(12, cfg.clone());
        let bad = vec![0.0_f32; cfg.vision_dim + 1];
        let err = proj.project_one(&bad).expect_err("wrong len must err");
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 13: visual_tokens wrong length errors ───────────────────────────────
    #[test]
    fn visual_tokens_wrong_len_errors() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(13, cfg.clone());
        let bad = vec![0.0_f32; 3 * cfg.vision_dim];
        let err = proj
            .project_tokens(&bad, 4)
            .expect_err("len does not match n_tokens");
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 14: n_tokens = 1 works ──────────────────────────────────────────────
    #[test]
    fn n_tokens_one_works() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(14, cfg.clone());
        let v = vec![0.1_f32; cfg.vision_dim];
        let y = proj.project_tokens(&v, 1).expect("project_tokens n=1");
        assert_eq!(y.len(), cfg.llm_dim);
    }

    // ── 15: project_one vs project_tokens row-equivalence ───────────────────
    #[test]
    fn project_one_matches_project_tokens() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(15, cfg.clone());
        let n = 4;
        // Build a batch with deterministic, distinct rows.
        let mut batch = vec![0.0_f32; n * cfg.vision_dim];
        for i in 0..n {
            for k in 0..cfg.vision_dim {
                batch[i * cfg.vision_dim + k] = ((i + 1) as f32 * 0.13 + k as f32 * 0.07).sin();
            }
        }
        let batched = proj
            .project_tokens(&batch, n)
            .expect("project_tokens batched");
        for i in 0..n {
            let row = &batch[i * cfg.vision_dim..(i + 1) * cfg.vision_dim];
            let one = proj.project_one(row).expect("project_one row");
            for (idx, (a, b)) in one
                .iter()
                .zip(batched[i * cfg.llm_dim..(i + 1) * cfg.llm_dim].iter())
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-5,
                    "row {i} col {idx}: per-row {a} vs batched {b}",
                );
            }
        }
    }

    // ── 16: project_tokens with n_tokens=0 errors ───────────────────────────
    #[test]
    fn project_tokens_zero_n_errors() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(16, cfg);
        let err = proj
            .project_tokens(&[], 0)
            .expect_err("n_tokens=0 must err");
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    // ── 17: config getter exposes the original config ───────────────────────
    #[test]
    fn config_getter_returns_input() {
        let cfg = LlavaProjectorConfig::tiny();
        let proj = make_proj(17, cfg.clone());
        let got = proj.config();
        assert_eq!(got.vision_dim, cfg.vision_dim);
        assert_eq!(got.llm_dim, cfg.llm_dim);
        assert_eq!(got.hidden_dim, cfg.hidden_dim);
        assert_eq!(got.mlp_depth, cfg.mlp_depth);
    }

    // ── 18: depth = 1 collapses to a single linear layer (no GELU) ──────────
    #[test]
    fn depth_one_is_linear_no_gelu() {
        // For a single linear layer, project_one(a*x) == a*project_one(x)
        // up to the bias term. Since we initialise biases to zero, the
        // mapping is exactly linear and homogeneous.
        let cfg = LlavaProjectorConfig {
            vision_dim: 8,
            llm_dim: 16,
            hidden_dim: 12,
            mlp_depth: 1,
        };
        let proj = make_proj(18, cfg.clone());
        let v: Vec<f32> = (0..cfg.vision_dim)
            .map(|i| (i as f32 + 1.0) * 0.1)
            .collect();
        let v3: Vec<f32> = v.iter().map(|x| 3.0 * x).collect();
        let y1 = proj.project_one(&v).expect("project_one v");
        let y3 = proj.project_one(&v3).expect("project_one 3v");
        for (a, b) in y1.iter().zip(y3.iter()) {
            assert!(
                (3.0 * a - b).abs() < 1e-4,
                "depth=1 must be linear: 3 * {a} != {b}",
            );
        }
    }
}
