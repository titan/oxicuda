//! Flamingo GATED XATTN-DENSE layer — visual-language cross-attention block
//! inserted into a frozen language model.
//!
//! Reference: Alayrac et al. 2022, "Flamingo: a Visual Language Model for
//! Few-Shot Learning".
//!
//! The layer cross-attends language tokens to visual features and then applies
//! a position-wise feed-forward network. Both contributions are scaled by
//! `tanh(alpha)` gates that are **initialised to zero**, so at initialisation
//! the layer is an exact identity (`tanh(0) = 0`). This is critical for stably
//! inserting the new layer into a pre-trained, frozen LM without disrupting it.
//!
//! ```text
//! attn_out = CrossAttention(LayerNorm(x) as Q, vis as K/V)
//! y        = x + tanh(alpha_attn) * attn_out
//! ffn_out  = FFN(LayerNorm(y))
//! z        = y + tanh(alpha_ffn)  * ffn_out
//! ```

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::{FeedForward, LayerNorm};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a Flamingo gated cross-attention layer.
#[derive(Debug, Clone)]
pub struct FlamingoGatedConfig {
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Number of attention heads. Must divide `d_model`.
    pub n_heads: usize,
    /// Hidden dimension of the feed-forward network.
    pub ffn_dim: usize,
}

impl FlamingoGatedConfig {
    /// Tiny preset for testing: `d_model=8`, `n_heads=2`, `ffn_dim=16`.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            d_model: 8,
            n_heads: 2,
            ffn_dim: 16,
        }
    }

    /// Validate the configuration, returning the appropriate error variant.
    fn validate(&self) -> MmResult<()> {
        if self.d_model == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.n_heads == 0 || self.d_model % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.d_model,
            });
        }
        if self.ffn_dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        Ok(())
    }
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// Learnable parameters for a Flamingo gated cross-attention layer.
#[derive(Debug, Clone)]
pub struct FlamingoGatedWeights {
    /// Cross-attention projections (Q from text, K/V from visuals).
    pub cross_attn: CrossAttnWeights,
    /// Feed-forward network (`d_model → ffn_dim → d_model`).
    pub ffn: FeedForward,
    /// LayerNorm applied before cross-attention.
    pub ln_attn: LayerNorm,
    /// LayerNorm applied before the feed-forward network.
    pub ln_ffn: LayerNorm,
    /// Attention gate (passed through `tanh`); zero at initialisation.
    pub alpha_attn: f32,
    /// Feed-forward gate (passed through `tanh`); zero at initialisation.
    pub alpha_ffn: f32,
}

impl FlamingoGatedWeights {
    /// Randomly initialise the projection / FFN weights, with both gates set
    /// to `0.0` (identity at init).
    fn random(cfg: &FlamingoGatedConfig, rng: &mut LcgRng) -> Self {
        let d = cfg.d_model;
        let attn_scale = (1.0 / d as f32).sqrt();
        let ffn_in_scale = (1.0 / d as f32).sqrt();
        let ffn_out_scale = (1.0 / cfg.ffn_dim as f32).sqrt();

        let cross_attn = CrossAttnWeights {
            w_q: gaussian_vec(d * d, attn_scale, rng),
            w_k: gaussian_vec(d * d, attn_scale, rng),
            w_v: gaussian_vec(d * d, attn_scale, rng),
            w_o: gaussian_vec(d * d, attn_scale, rng),
        };
        let ffn = FeedForward {
            w1: gaussian_vec(d * cfg.ffn_dim, ffn_in_scale, rng),
            b1: vec![0.0_f32; cfg.ffn_dim],
            w2: gaussian_vec(cfg.ffn_dim * d, ffn_out_scale, rng),
            b2: vec![0.0_f32; d],
            d_model: d,
            d_ff: cfg.ffn_dim,
        };

        Self {
            cross_attn,
            ffn,
            ln_attn: LayerNorm::ones(d),
            ln_ffn: LayerNorm::ones(d),
            alpha_attn: 0.0,
            alpha_ffn: 0.0,
        }
    }
}

/// Allocate a vector of `len` N(0, `scale`²) samples.
fn gaussian_vec(len: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut v = vec![0.0_f32; len];
    rng.fill_normal(&mut v);
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

// ─── FlamingoGatedLayer ────────────────────────────────────────────────────────

/// Flamingo GATED XATTN-DENSE layer.
#[derive(Debug, Clone)]
pub struct FlamingoGatedLayer {
    pub cfg: FlamingoGatedConfig,
    pub weights: FlamingoGatedWeights,
}

impl FlamingoGatedLayer {
    /// Create a new layer with randomly initialised weights and both gates set
    /// to `0.0` (so the layer is an exact identity at initialisation).
    pub fn new(cfg: FlamingoGatedConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = FlamingoGatedWeights::random(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Forward pass.
    ///
    /// - `x`: `[n_text × d_model]` row-major — language tokens.
    /// - `vis`: `[n_vis × d_model]` row-major — visual features.
    ///
    /// Returns `[n_text × d_model]`.
    pub fn forward(
        &self,
        x: &[f32],
        n_text: usize,
        vis: &[f32],
        n_vis: usize,
    ) -> MmResult<Vec<f32>> {
        let d = self.cfg.d_model;

        if n_text == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if n_vis == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if x.len() != n_text * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_text * d,
                got: x.len(),
            });
        }
        if vis.len() != n_vis * d {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_vis * d,
                got: vis.len(),
            });
        }

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, d, 0.0)?;
        let cross_attn = CrossAttention::with_weights(attn_cfg, self.weights.cross_attn.clone());

        // ── Gated cross-attention: y = x + tanh(alpha_attn) * xattn(x, vis) ──
        let ln_x = self.weights.ln_attn.forward(x, n_text)?;
        let attn_out = cross_attn.forward(&ln_x, vis, vis, n_text, n_vis)?;
        let gate_attn = self.weights.alpha_attn.tanh();
        let mut y = vec![0.0_f32; n_text * d];
        for i in 0..(n_text * d) {
            y[i] = x[i] + gate_attn * attn_out[i];
        }

        // ── Gated FFN: z = y + tanh(alpha_ffn) * FFN(y) ──────────────────────
        let ln_y = self.weights.ln_ffn.forward(&y, n_text)?;
        let ffn_out = self.weights.ffn.forward(&ln_y, n_text)?;
        let gate_ffn = self.weights.alpha_ffn.tanh();
        let mut z = y;
        for (zi, fi) in z.iter_mut().zip(ffn_out.iter()) {
            *zi += gate_ffn * fi;
        }

        Ok(z)
    }

    /// Return the attention gate value (raw, pre-`tanh`).
    #[must_use]
    pub fn alpha_attn(&self) -> f32 {
        self.weights.alpha_attn
    }

    /// Return the feed-forward gate value (raw, pre-`tanh`).
    #[must_use]
    pub fn alpha_ffn(&self) -> f32 {
        self.weights.alpha_ffn
    }

    /// Set both gate values.
    pub fn set_gates(&mut self, alpha_attn: f32, alpha_ffn: f32) {
        self.weights.alpha_attn = alpha_attn;
        self.weights.alpha_ffn = alpha_ffn;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(seed: u64) -> FlamingoGatedLayer {
        let mut rng = LcgRng::new(seed);
        FlamingoGatedLayer::new(FlamingoGatedConfig::tiny(), &mut rng)
            .expect("value should be present")
    }

    #[test]
    fn default_gates_make_identity() {
        // THE Flamingo property: zero-init gates ⇒ forward(x, ...) == x exactly.
        let layer = make_layer(1);
        let d = layer.cfg.d_model;
        let n_text = 4;
        let n_vis = 5;
        let x: Vec<f32> = (0..(n_text * d)).map(|i| (i as f32) * 0.03 - 0.5).collect();
        let vis = vec![0.7_f32; n_vis * d];
        let out = layer
            .forward(&x, n_text, &vis, n_vis)
            .expect("forward should succeed");
        assert_eq!(out, x);
    }

    #[test]
    fn default_gate_values_are_zero() {
        let layer = make_layer(2);
        assert_eq!(layer.alpha_attn(), 0.0);
        assert_eq!(layer.alpha_ffn(), 0.0);
    }

    #[test]
    fn set_gates_updates_values() {
        let mut layer = make_layer(3);
        layer.set_gates(0.5, -0.25);
        assert_eq!(layer.alpha_attn(), 0.5);
        assert_eq!(layer.alpha_ffn(), -0.25);
    }

    #[test]
    fn nonzero_gates_change_output() {
        let mut layer = make_layer(4);
        let d = layer.cfg.d_model;
        let n_text = 3;
        let n_vis = 4;
        let x: Vec<f32> = (0..(n_text * d)).map(|i| (i as f32) * 0.02).collect();
        let vis = vec![0.4_f32; n_vis * d];

        let identity_out = layer
            .forward(&x, n_text, &vis, n_vis)
            .expect("forward should succeed");
        layer.set_gates(1.0, 1.0);
        let gated_out = layer
            .forward(&x, n_text, &vis, n_vis)
            .expect("forward should succeed");

        assert_ne!(identity_out, gated_out);
    }

    #[test]
    fn gate_zero_ignores_vis() {
        // With alpha_attn = 0, changing the visual features must NOT change the
        // output (the attention contribution is fully gated off).
        let layer = make_layer(5);
        let d = layer.cfg.d_model;
        let n_text = 4;
        let n_vis = 5;
        let x: Vec<f32> = (0..(n_text * d)).map(|i| (i as f32) * 0.05).collect();

        let vis_a = vec![0.1_f32; n_vis * d];
        let vis_b = vec![0.9_f32; n_vis * d];
        let out_a = layer
            .forward(&x, n_text, &vis_a, n_vis)
            .expect("forward should succeed");
        let out_b = layer
            .forward(&x, n_text, &vis_b, n_vis)
            .expect("forward should succeed");

        assert_eq!(out_a, out_b);
    }

    #[test]
    fn gate_nonzero_uses_vis() {
        // With alpha_attn > 0, changing the visual features MUST change the
        // output (the gating is wired through cross-attention).
        let mut layer = make_layer(6);
        layer.set_gates(0.8, 0.0);
        let d = layer.cfg.d_model;
        let n_text = 4;
        let n_vis = 5;
        let x: Vec<f32> = (0..(n_text * d)).map(|i| (i as f32) * 0.05).collect();

        let vis_a = vec![0.1_f32; n_vis * d];
        let mut vis_b = vec![0.1_f32; n_vis * d];
        for (i, v) in vis_b.iter_mut().enumerate() {
            *v = 0.1 + (i as f32) * 0.07;
        }
        let out_a = layer
            .forward(&x, n_text, &vis_a, n_vis)
            .expect("forward should succeed");
        let out_b = layer
            .forward(&x, n_text, &vis_b, n_vis)
            .expect("forward should succeed");

        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-4,
            "open attention gate should respond to visuals, diff={diff}"
        );
    }

    #[test]
    fn output_length_correct() {
        let mut layer = make_layer(7);
        layer.set_gates(0.3, 0.3);
        let d = layer.cfg.d_model;
        let n_text = 6;
        let n_vis = 3;
        let x = vec![0.2_f32; n_text * d];
        let vis = vec![0.5_f32; n_vis * d];
        let out = layer
            .forward(&x, n_text, &vis, n_vis)
            .expect("forward should succeed");
        assert_eq!(out.len(), n_text * d);
    }

    #[test]
    fn deterministic_given_seed() {
        let mut a = make_layer(8);
        let mut b = make_layer(8);
        a.set_gates(0.5, 0.5);
        b.set_gates(0.5, 0.5);
        let d = a.cfg.d_model;
        let x = vec![0.15_f32; 4 * d];
        let vis = vec![0.3_f32; 5 * d];
        let out_a = a.forward(&x, 4, &vis, 5).expect("forward should succeed");
        let out_b = b.forward(&x, 4, &vis, 5).expect("forward should succeed");
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn output_finite() {
        let mut layer = make_layer(9);
        layer.set_gates(0.6, 0.6);
        let d = layer.cfg.d_model;
        let x = vec![0.3_f32; 5 * d];
        let vis = vec![0.4_f32; 6 * d];
        let out = layer
            .forward(&x, 5, &vis, 6)
            .expect("forward should succeed");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn large_alpha_saturates_tanh() {
        // A very large gate ⇒ tanh ≈ 1, so the gated output ≈ x + attn_out.
        // A huge gate and an even huger gate should give nearly identical output.
        let mut layer = make_layer(10);
        let d = layer.cfg.d_model;
        let n_text = 4;
        let n_vis = 5;
        let x = vec![0.2_f32; n_text * d];
        let vis: Vec<f32> = (0..(n_vis * d)).map(|i| (i as f32) * 0.03).collect();

        layer.set_gates(50.0, 50.0);
        let out_big = layer
            .forward(&x, n_text, &vis, n_vis)
            .expect("forward should succeed");
        layer.set_gates(500.0, 500.0);
        let out_huge = layer
            .forward(&x, n_text, &vis, n_vis)
            .expect("forward should succeed");

        let diff: f32 = out_big
            .iter()
            .zip(out_huge.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff < 1e-4, "tanh should saturate near 1.0, diff={diff}");
    }

    #[test]
    fn d_model_not_divisible_by_heads_errors() {
        let mut rng = LcgRng::new(11);
        let cfg = FlamingoGatedConfig {
            d_model: 10,
            n_heads: 3,
            ffn_dim: 16,
        };
        let err = FlamingoGatedLayer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }

    #[test]
    fn x_wrong_length_errors() {
        let layer = make_layer(12);
        let d = layer.cfg.d_model;
        let x = vec![0.1_f32; 3 * d]; // claim 4 text tokens below
        let vis = vec![0.2_f32; 5 * d];
        let err = layer.forward(&x, 4, &vis, 5).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn vis_wrong_length_errors() {
        let layer = make_layer(13);
        let d = layer.cfg.d_model;
        let x = vec![0.1_f32; 4 * d];
        let vis = vec![0.2_f32; 3 * d]; // claim 5 visual tokens below
        let err = layer.forward(&x, 4, &vis, 5).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    #[test]
    fn n_text_zero_errors() {
        let layer = make_layer(14);
        let d = layer.cfg.d_model;
        let vis = vec![0.2_f32; 5 * d];
        let err = layer.forward(&[], 0, &vis, 5).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn n_vis_zero_errors() {
        let layer = make_layer(15);
        let d = layer.cfg.d_model;
        let x = vec![0.1_f32; 4 * d];
        let err = layer.forward(&x, 4, &[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    #[test]
    fn ffn_dim_zero_errors() {
        let mut rng = LcgRng::new(16);
        let cfg = FlamingoGatedConfig {
            d_model: 8,
            n_heads: 2,
            ffn_dim: 0,
        };
        let err = FlamingoGatedLayer::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    #[test]
    fn single_head_works() {
        let mut rng = LcgRng::new(17);
        let cfg = FlamingoGatedConfig {
            d_model: 8,
            n_heads: 1,
            ffn_dim: 16,
        };
        let mut layer = FlamingoGatedLayer::new(cfg, &mut rng).expect("new should succeed");
        layer.set_gates(0.5, 0.5);
        let x = vec![0.2_f32; 4 * 8];
        let vis = vec![0.3_f32; 5 * 8];
        let out = layer
            .forward(&x, 4, &vis, 5)
            .expect("forward should succeed");
        assert_eq!(out.len(), 4 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn single_vis_token_works() {
        let mut layer = make_layer(18);
        layer.set_gates(0.4, 0.4);
        let d = layer.cfg.d_model;
        let x = vec![0.2_f32; 4 * d];
        let vis = vec![0.3_f32; d];
        let out = layer
            .forward(&x, 4, &vis, 1)
            .expect("forward should succeed");
        assert_eq!(out.len(), 4 * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
