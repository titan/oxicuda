//! AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.
//!
//! Reference: Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive
//! Neural Networks", AAAI 2019.
//!
//! Each raw feature is mapped to an embedding vector (continuous: `x_j * W_j + b_j`).
//! L stacked multi-head self-attention layers with optional residual connections and LayerNorm
//! transform the feature tokens. The final token matrix is flattened and fed to a linear
//! classifier.

use crate::attention::saint::multihead_attention;
use crate::error::{TabularError, TabularResult};
use crate::handle::LcgRng;

// ─── AutoIntConfig ────────────────────────────────────────────────────────────

/// Configuration for `AutoInt`.
#[derive(Debug, Clone)]
pub struct AutoIntConfig {
    /// Number of input features (F).
    pub n_features: usize,
    /// Embedding dimension per feature (d). All features share the same dim.
    pub embed_dim: usize,
    /// Number of attention heads (`embed_dim % n_heads == 0`).
    pub n_heads: usize,
    /// Number of self-attention layers (default 3).
    pub n_attn_layers: usize,
    /// Number of output classes (1 for regression).
    pub n_classes: usize,
    /// Whether to add a residual connection in each attention layer (default true).
    pub use_residual: bool,
}

impl Default for AutoIntConfig {
    fn default() -> Self {
        Self {
            n_features: 8,
            embed_dim: 16,
            n_heads: 2,
            n_attn_layers: 3,
            n_classes: 2,
            use_residual: true,
        }
    }
}

// ─── AutoIntLayerWeights ──────────────────────────────────────────────────────

/// Weight bundle for a single AutoInt attention layer.
#[derive(Debug, Clone)]
pub struct AutoIntLayerWeights {
    /// Q projection: `[embed_dim × embed_dim]`, row-major.
    pub wq: Vec<f32>,
    /// K projection: `[embed_dim × embed_dim]`.
    pub wk: Vec<f32>,
    /// V projection: `[embed_dim × embed_dim]`.
    pub wv: Vec<f32>,
    /// Output projection: `[embed_dim × embed_dim]`.
    pub wo: Vec<f32>,
    /// Layer norm gamma (scale): `[embed_dim]`.
    pub ln_gamma: Vec<f32>,
    /// Layer norm beta (shift): `[embed_dim]`.
    pub ln_beta: Vec<f32>,
}

impl AutoIntLayerWeights {
    /// Create randomly initialised weights for one attention layer.
    ///
    /// Uses Kaiming-uniform init: U(-k, k) where k = sqrt(6 / embed_dim).
    /// Layer-norm gamma = ones, beta = zeros.
    pub fn new_random(embed_dim: usize, rng: &mut LcgRng) -> Self {
        let k = (6.0_f32 / embed_dim as f32).sqrt();
        let n = embed_dim * embed_dim;

        let mut fill_kaiming =
            |size: usize| -> Vec<f32> { (0..size).map(|_| rng.next_f32() * 2.0 * k - k).collect() };

        Self {
            wq: fill_kaiming(n),
            wk: fill_kaiming(n),
            wv: fill_kaiming(n),
            wo: fill_kaiming(n),
            ln_gamma: vec![1.0_f32; embed_dim],
            ln_beta: vec![0.0_f32; embed_dim],
        }
    }
}

// ─── AutoIntWeights ───────────────────────────────────────────────────────────

/// Full weight bundle for AutoInt.
#[derive(Debug, Clone)]
pub struct AutoIntWeights {
    /// Continuous feature embedding weights: `[n_features × embed_dim]`, row-major.
    pub cont_w: Vec<f32>,
    /// Continuous feature embedding biases: `[n_features × embed_dim]`.
    pub cont_b: Vec<f32>,
    /// Per-layer attention weights (length == n_attn_layers).
    pub layers: Vec<AutoIntLayerWeights>,
    /// Final classifier weight: `[n_classes × (n_features * embed_dim)]`.
    pub cls_w: Vec<f32>,
    /// Final classifier bias: `[n_classes]`.
    pub cls_b: Vec<f32>,
}

impl AutoIntWeights {
    /// Create randomly initialised weights for an AutoInt model.
    ///
    /// - `cont_w`, `cont_b`: Kaiming-uniform U(±sqrt(6/(embed_dim+1))).
    /// - Per-layer weights via `AutoIntLayerWeights::new_random`.
    /// - `cls_w`: Kaiming-uniform U(±sqrt(6/(n_features*embed_dim + n_classes))).
    /// - `cls_b`: zeros.
    pub fn new_random(cfg: &AutoIntConfig, rng: &mut LcgRng) -> Self {
        let n_feat = cfg.n_features;
        let ed = cfg.embed_dim;

        // Continuous embedding init
        let k_cont = (6.0_f32 / (ed as f32 + 1.0)).sqrt();
        let embed_size = n_feat * ed;
        let cont_w: Vec<f32> = (0..embed_size)
            .map(|_| rng.next_f32() * 2.0 * k_cont - k_cont)
            .collect();
        let cont_b = vec![0.0_f32; embed_size];

        // Attention layer weights
        let layers: Vec<AutoIntLayerWeights> = (0..cfg.n_attn_layers)
            .map(|_| AutoIntLayerWeights::new_random(ed, rng))
            .collect();

        // Classifier weight
        let flat_dim = n_feat * ed;
        let k_cls = (6.0_f32 / (flat_dim as f32 + cfg.n_classes as f32)).sqrt();
        let cls_w: Vec<f32> = (0..cfg.n_classes * flat_dim)
            .map(|_| rng.next_f32() * 2.0 * k_cls - k_cls)
            .collect();
        let cls_b = vec![0.0_f32; cfg.n_classes];

        Self {
            cont_w,
            cont_b,
            layers,
            cls_w,
            cls_b,
        }
    }
}

// ─── Layer normalisation ──────────────────────────────────────────────────────

/// Layer normalisation: `(x - mean) / sqrt(var + eps) * gamma + beta`.
///
/// - `x`, `gamma`, `beta`: length-n slices.
/// - `eps`: small constant for numerical stability (typically 1e-5).
pub fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    x.iter()
        .zip(gamma.iter().zip(beta.iter()))
        .map(|(&xi, (&g, &b))| (xi - mean) / (var + eps).sqrt() * g + b)
        .collect()
}

// ─── AutoInt ──────────────────────────────────────────────────────────────────

/// AutoInt model (inference only).
pub struct AutoInt {
    /// Model configuration.
    pub config: AutoIntConfig,
}

impl AutoInt {
    /// Construct a new `AutoInt` instance, validating the configuration.
    pub fn new(config: AutoIntConfig) -> TabularResult<Self> {
        if config.n_features == 0 {
            return Err(TabularError::InvalidFeatureCount { n: 0 });
        }
        if config.embed_dim == 0 {
            return Err(TabularError::InvalidEmbedDim { dim: 0 });
        }
        if !config.embed_dim.is_multiple_of(config.n_heads) {
            return Err(TabularError::InvalidAttentionDim {
                dim: config.embed_dim,
            });
        }
        Ok(Self { config })
    }

    /// Map continuous input features to per-feature embedding tokens.
    ///
    /// - `x`: `[n_samples × n_features]` flat row-major.
    /// - `cont_w`, `cont_b`: `[n_features × embed_dim]` row-major.
    /// - Returns: `[n_samples × n_features × embed_dim]`.
    pub fn tokenize(&self, x: &[f32], cont_w: &[f32], cont_b: &[f32]) -> TabularResult<Vec<f32>> {
        let n_feat = self.config.n_features;
        let ed = self.config.embed_dim;

        if !x.len().is_multiple_of(n_feat) {
            return Err(TabularError::DimensionMismatch {
                expected: (x.len() / n_feat) * n_feat,
                got: x.len(),
            });
        }
        let n_samples = x.len() / n_feat;

        let expected_w = n_feat * ed;
        if cont_w.len() != expected_w {
            return Err(TabularError::DimensionMismatch {
                expected: expected_w,
                got: cont_w.len(),
            });
        }
        if cont_b.len() != expected_w {
            return Err(TabularError::DimensionMismatch {
                expected: expected_w,
                got: cont_b.len(),
            });
        }

        let mut out = vec![0.0_f32; n_samples * n_feat * ed];
        for s in 0..n_samples {
            for j in 0..n_feat {
                let xj = x[s * n_feat + j];
                let w_row = &cont_w[j * ed..(j + 1) * ed];
                let b_row = &cont_b[j * ed..(j + 1) * ed];
                let out_row = &mut out[(s * n_feat + j) * ed..(s * n_feat + j + 1) * ed];
                for (d, o) in out_row.iter_mut().enumerate() {
                    *o = xj * w_row[d] + b_row[d];
                }
            }
        }
        Ok(out)
    }

    /// Apply one self-attention layer with optional residual + LayerNorm (single sample).
    ///
    /// - `h`: `[n_features × embed_dim]`.
    /// - Returns: updated `h` of the same shape.
    pub fn attention_layer(
        &self,
        h: &[f32],
        layer_weights: &AutoIntLayerWeights,
    ) -> TabularResult<Vec<f32>> {
        let n_feat = self.config.n_features;
        let ed = self.config.embed_dim;
        let n_heads = self.config.n_heads;

        let expected = n_feat * ed;
        if h.len() != expected {
            return Err(TabularError::DimensionMismatch {
                expected,
                got: h.len(),
            });
        }

        // Multi-head self-attention: seq_len = n_features
        let attn_out = multihead_attention(
            h,
            &layer_weights.wq,
            &layer_weights.wk,
            &layer_weights.wv,
            &layer_weights.wo,
            n_feat,
            ed,
            n_heads,
        )?;

        // Residual connection
        let residual: Vec<f32> = if self.config.use_residual {
            h.iter()
                .zip(attn_out.iter())
                .map(|(&hi, &ai)| hi + ai)
                .collect()
        } else {
            attn_out
        };

        // LayerNorm applied token-by-token
        let mut h_new = vec![0.0_f32; n_feat * ed];
        for f in 0..n_feat {
            let tok = &residual[f * ed..(f + 1) * ed];
            let normed = layer_norm(tok, &layer_weights.ln_gamma, &layer_weights.ln_beta, 1e-5);
            h_new[f * ed..(f + 1) * ed].copy_from_slice(&normed);
        }
        Ok(h_new)
    }

    /// Run the full AutoInt forward pass on one sample.
    ///
    /// - `x`: `[n_features]`.
    /// - Returns logits: `[n_classes]`.
    pub fn forward_single(&self, x: &[f32], weights: &AutoIntWeights) -> TabularResult<Vec<f32>> {
        let n_feat = self.config.n_features;
        let ed = self.config.embed_dim;

        if x.len() != n_feat {
            return Err(TabularError::DimensionMismatch {
                expected: n_feat,
                got: x.len(),
            });
        }

        // Step 1: Tokenize → [n_features × embed_dim]
        let tokens = self.tokenize(x, &weights.cont_w, &weights.cont_b)?;
        let mut h = tokens;

        // Step 2: Apply attention layers
        for layer_w in &weights.layers {
            h = self.attention_layer(&h, layer_w)?;
        }

        // Step 3: Flatten → [n_features * embed_dim]
        // h is already flat as [n_feat * ed]

        // Step 4: Linear classifier
        let flat_dim = n_feat * ed;
        let mut logits = weights.cls_b.clone();
        for (c, logit) in logits.iter_mut().enumerate() {
            for (d, &hd) in h.iter().enumerate() {
                *logit += weights.cls_w[c * flat_dim + d] * hd;
            }
        }
        Ok(logits)
    }

    /// Run the full AutoInt forward pass on a batch.
    ///
    /// - `x`: `[n_samples × n_features]`.
    /// - Returns logits: `[n_samples × n_classes]`.
    pub fn forward(&self, x: &[f32], weights: &AutoIntWeights) -> TabularResult<Vec<f32>> {
        let n_feat = self.config.n_features;
        let n_classes = self.config.n_classes;

        if !x.len().is_multiple_of(n_feat) {
            return Err(TabularError::DimensionMismatch {
                expected: (x.len() / n_feat) * n_feat,
                got: x.len(),
            });
        }
        let n_samples = x.len() / n_feat;

        let mut all_logits = Vec::with_capacity(n_samples * n_classes);
        for s in 0..n_samples {
            let row = &x[s * n_feat..(s + 1) * n_feat];
            let logits = self.forward_single(row, weights)?;
            all_logits.extend_from_slice(&logits);
        }
        Ok(all_logits)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    // ── layer_norm tests ──────────────────────────────────────────────────────

    #[test]
    fn layer_norm_basic_mean_std() {
        // Input [0, 1, 2, 3]: mean = 1.5, var = 5/4 = 1.25
        let x = [0.0_f32, 1.0, 2.0, 3.0];
        let gamma = [1.0_f32; 4];
        let beta = [0.0_f32; 4];
        let out = layer_norm(&x, &gamma, &beta, 1e-8);
        // Expected normalised values
        let mean = 1.5_f32;
        let std = 1.25_f32.sqrt();
        for (i, (&xi, &oi)) in x.iter().zip(out.iter()).enumerate() {
            let expected = (xi - mean) / std;
            assert!(
                (oi - expected).abs() < 1e-4,
                "index {i}: expected {expected}, got {oi}"
            );
        }
    }

    #[test]
    fn layer_norm_all_same_input_zeros() {
        // All-same input → normalised to ~0 (std≈0 + eps)
        let x = [3.0_f32; 5];
        let gamma = [1.0_f32; 5];
        let beta = [0.0_f32; 5];
        let out = layer_norm(&x, &gamma, &beta, 1e-5);
        for &v in &out {
            assert!(v.abs() < 1e-2, "expected ~0, got {v}");
        }
    }

    #[test]
    fn layer_norm_gamma_zero_produces_zeros() {
        let x = [1.0_f32, 2.0, 3.0];
        let gamma = [0.0_f32; 3];
        let beta = [0.0_f32; 3];
        let out = layer_norm(&x, &gamma, &beta, 1e-5);
        for &v in &out {
            assert_eq!(v, 0.0_f32, "expected 0 with gamma=0, got {v}");
        }
    }

    #[test]
    fn layer_norm_beta_shifts_output() {
        let x = [0.0_f32, 0.0, 0.0, 0.0];
        let gamma = [1.0_f32; 4];
        let beta = [5.0_f32; 4];
        let out = layer_norm(&x, &gamma, &beta, 1e-5);
        // All-same → normalised to 0, then shifted by beta=5
        for &v in &out {
            assert!((v - 5.0).abs() < 1e-3, "expected 5.0, got {v}");
        }
    }

    #[test]
    fn layer_norm_gamma_scales_output() {
        let x = [0.0_f32, 1.0, 2.0, 3.0];
        let gamma = [2.0_f32; 4];
        let beta = [0.0_f32; 4];
        let out_scaled = layer_norm(&x, &gamma, &beta, 1e-8);
        let gamma_one = [1.0_f32; 4];
        let out_unit = layer_norm(&x, &gamma_one, &beta, 1e-8);
        for (&s, &u) in out_scaled.iter().zip(out_unit.iter()) {
            assert!((s - 2.0 * u).abs() < 1e-5, "scaling mismatch");
        }
    }

    // ── AutoInt::new validation ───────────────────────────────────────────────

    #[test]
    fn autoint_new_n_features_zero_is_err() {
        let cfg = AutoIntConfig {
            n_features: 0,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 2,
            n_classes: 2,
            use_residual: true,
        };
        assert!(AutoInt::new(cfg).is_err());
    }

    #[test]
    fn autoint_new_embed_dim_zero_is_err() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 0,
            n_heads: 1,
            n_attn_layers: 2,
            n_classes: 2,
            use_residual: true,
        };
        assert!(AutoInt::new(cfg).is_err());
    }

    #[test]
    fn autoint_new_embed_dim_not_divisible_by_n_heads_is_err() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 7, // 7 % 3 != 0
            n_heads: 3,
            n_attn_layers: 2,
            n_classes: 2,
            use_residual: true,
        };
        assert!(AutoInt::new(cfg).is_err());
    }

    // ── tokenize tests ────────────────────────────────────────────────────────

    #[test]
    fn tokenize_output_shape() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 2,
            n_classes: 2,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(1);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.5_f32; 3 * 4]; // 3 samples × 4 features
        let out = model
            .tokenize(&x, &weights.cont_w, &weights.cont_b)
            .unwrap();
        assert_eq!(out.len(), 3 * 4 * 8, "shape mismatch");
    }

    #[test]
    fn tokenize_single_sample_correct_values() {
        // n_features=2, embed_dim=3
        let cfg = AutoIntConfig {
            n_features: 2,
            embed_dim: 3,
            n_heads: 1,
            n_attn_layers: 0,
            n_classes: 1,
            use_residual: false,
        };
        let model = AutoInt::new(cfg).unwrap();
        // cont_w: feat0=[1,2,3], feat1=[4,5,6]  cont_b: all zeros
        let cont_w = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cont_b = vec![0.0_f32; 6];
        let x = vec![2.0_f32, 3.0]; // x[0]=2, x[1]=3
        let out = model.tokenize(&x, &cont_w, &cont_b).unwrap();
        // token0 = 2 * [1,2,3] = [2,4,6]
        // token1 = 3 * [4,5,6] = [12,15,18]
        let expected = [2.0_f32, 4.0, 6.0, 12.0, 15.0, 18.0];
        for (i, (&o, &e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((o - e).abs() < 1e-5, "index {i}: expected {e}, got {o}");
        }
    }

    // ── attention_layer tests ─────────────────────────────────────────────────

    #[test]
    fn attention_layer_output_shape() {
        let cfg = AutoIntConfig {
            n_features: 5,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 1,
            n_classes: 2,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(99);
        let layer_w = AutoIntLayerWeights::new_random(8, &mut rng);
        let h = vec![0.1_f32; 5 * 8];
        let out = model.attention_layer(&h, &layer_w).unwrap();
        assert_eq!(out.len(), 5 * 8, "attention_layer output shape mismatch");
    }

    #[test]
    fn attention_layer_with_residual_changes_output() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 4,
            n_heads: 2,
            n_attn_layers: 1,
            n_classes: 2,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(42);
        let layer_w = AutoIntLayerWeights::new_random(4, &mut rng);
        let h: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let out = model.attention_layer(&h, &layer_w).unwrap();
        // Output should differ from input due to attention transform
        let all_equal = h
            .iter()
            .zip(out.iter())
            .all(|(&a, &b)| (a - b).abs() < 1e-8);
        assert!(!all_equal, "attention_layer should modify input");
    }

    // ── forward_single tests ──────────────────────────────────────────────────

    #[test]
    fn forward_single_output_shape() {
        let cfg = AutoIntConfig {
            n_features: 6,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 2,
            n_classes: 3,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(7);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.3_f32; 6];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 3, "forward_single output shape mismatch");
    }

    #[test]
    fn forward_single_regression_n_classes_1() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 1,
            n_classes: 1,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(3);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.5_f32; 4];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 1, "regression output should have 1 element");
    }

    // ── forward (batch) tests ─────────────────────────────────────────────────

    #[test]
    fn forward_batch_output_shape() {
        let cfg = AutoIntConfig {
            n_features: 5,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 2,
            n_classes: 3,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(11);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.2_f32; 7 * 5]; // 7 samples × 5 features
        let logits = model.forward(&x, &weights).unwrap();
        assert_eq!(logits.len(), 7 * 3, "forward batch output shape mismatch");
    }

    #[test]
    fn forward_batch_output_is_finite() {
        let cfg = AutoIntConfig::default();
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(55);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x: Vec<f32> = (0..4 * 8).map(|i| (i as f32) * 0.01).collect();
        let logits = model.forward(&x, &weights).unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
    }

    #[test]
    fn forward_single_vs_batch_n_samples_1() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 2,
            n_classes: 2,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(21);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.1_f32, 0.2, 0.3, 0.4];
        let single = model.forward_single(&x, &weights).unwrap();
        let batch = model.forward(&x, &weights).unwrap();
        assert_eq!(single.len(), batch.len());
        for (&s, &b) in single.iter().zip(batch.iter()) {
            assert!((s - b).abs() < 1e-5, "single vs batch mismatch: {s} vs {b}");
        }
    }

    #[test]
    fn forward_batch_4_samples() {
        let cfg = AutoIntConfig {
            n_features: 3,
            embed_dim: 4,
            n_heads: 2,
            n_attn_layers: 1,
            n_classes: 2,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(88);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.5_f32; 4 * 3]; // 4 samples × 3 features
        let logits = model.forward(&x, &weights).unwrap();
        assert_eq!(logits.len(), 4 * 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    // ── weight initialisation tests ───────────────────────────────────────────

    #[test]
    fn autoint_weights_new_random_finite() {
        let cfg = AutoIntConfig::default();
        let mut rng = LcgRng::new(77);
        let w = AutoIntWeights::new_random(&cfg, &mut rng);
        assert!(w.cont_w.iter().all(|v| v.is_finite()));
        assert!(w.cont_b.iter().all(|v| v.is_finite()));
        for layer in &w.layers {
            assert!(layer.wq.iter().all(|v| v.is_finite()));
            assert!(layer.wk.iter().all(|v| v.is_finite()));
            assert!(layer.wv.iter().all(|v| v.is_finite()));
            assert!(layer.wo.iter().all(|v| v.is_finite()));
        }
        assert!(w.cls_w.iter().all(|v| v.is_finite()));
        assert!(w.cls_b.iter().all(|v| v.is_finite()));
    }

    // ── n_attn_layers=0 (no attention) ───────────────────────────────────────

    #[test]
    fn forward_zero_attn_layers() {
        let cfg = AutoIntConfig {
            n_features: 3,
            embed_dim: 4,
            n_heads: 2,
            n_attn_layers: 0,
            n_classes: 2,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(5);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![1.0_f32, 2.0, 3.0];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 2);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_one_attn_layer() {
        let cfg = AutoIntConfig {
            n_features: 4,
            embed_dim: 8,
            n_heads: 2,
            n_attn_layers: 1,
            n_classes: 3,
            use_residual: true,
        };
        let model = AutoInt::new(cfg).unwrap();
        let mut rng = LcgRng::new(9);
        let weights = AutoIntWeights::new_random(&model.config, &mut rng);
        let x = vec![0.1_f32; 4];
        let logits = model.forward_single(&x, &weights).unwrap();
        assert_eq!(logits.len(), 3);
    }
}
