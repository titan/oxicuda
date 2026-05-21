//! PerceiverIO — A General Architecture for Structured Inputs & Outputs.
//!
//! Reference: Jaegle et al. 2021, "Perceiver IO: A General Architecture for
//! Structured Inputs & Outputs".
//!
//! PerceiverIO decouples compute from input/output size by routing all
//! information through a small, fixed-size *latent array*:
//!
//! 1. **Encode** — a learnable latent array of shape `n_latents × latent_dim`
//!    cross-attends over the (potentially huge) input array. Because Q comes
//!    from the latents, the output of this step has shape
//!    `n_latents × latent_dim` — **independent of `n_inputs`** — which is the
//!    defining property of the Perceiver family.
//! 2. **Process** — `n_self_layers` blocks of latent self-attention followed
//!    by a position-wise feed-forward network refine the latents in place
//!    (pre-norm + residual).
//! 3. **Decode** — a learnable *output query array* of shape
//!    `n_outputs × output_dim` cross-attends over the refined latents to
//!    produce the final structured output, of shape `n_outputs × output_dim`.
//!
//! Because the encode step projects K/V from `input_dim` to `latent_dim` and
//! the decode step projects Q from `output_dim` to `latent_dim`, the user is
//! free to choose `input_dim`, `latent_dim`, and `output_dim` independently;
//! divisibility constraints apply only to `latent_dim` (which carries all
//! attention) and `n_heads`.
//!
//! ```text
//!  inputs [n_in × input_dim] ──► K, V (projected to latent_dim)
//!  latents [n_latents × latent_dim] ──► Q
//!                          └────── cross-attn ─────► refined_latents
//!                                                       │
//!                            n_self_layers × { self-attn + FFN }
//!                                                       │
//!  out_queries [n_outputs × output_dim] ──► Q (projected to latent_dim)
//!  refined_latents ──► K, V (no projection needed)
//!                          └────── cross-attn ─────► outputs [n_outputs × output_dim]
//! ```

use crate::cross_attn::cross_attention::{CrossAttention, CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::self_cross_block::{FeedForward, LayerNorm};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the Perceiver IO architecture.
#[derive(Debug, Clone)]
pub struct PerceiverIoConfig {
    /// Dimensionality of each input token.
    pub input_dim: usize,
    /// Dimensionality of the latent space (carries all attention compute).
    pub latent_dim: usize,
    /// Number of latent tokens (fixed bottleneck size).
    pub n_latents: usize,
    /// Number of attention heads. Must divide `latent_dim`.
    pub n_heads: usize,
    /// Number of latent self-attention + FFN layers.
    pub n_self_layers: usize,
    /// Dimensionality of each output token.
    pub output_dim: usize,
    /// Number of output tokens to emit.
    pub n_outputs: usize,
}

impl PerceiverIoConfig {
    /// Tiny preset for unit testing.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            input_dim: 6,
            latent_dim: 8,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 3,
        }
    }

    /// Validate the configuration.
    fn validate(&self) -> MmResult<()> {
        if self.input_dim == 0 || self.latent_dim == 0 || self.output_dim == 0 {
            return Err(MultiModalError::InvalidFeatureDim);
        }
        if self.n_latents == 0 || self.n_outputs == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if self.n_heads == 0 || self.latent_dim % self.n_heads != 0 {
            return Err(MultiModalError::InvalidHeads {
                heads: self.n_heads,
                d_model: self.latent_dim,
            });
        }
        Ok(())
    }
}

// ─── Self-attention layer weights ──────────────────────────────────────────────

/// One layer of latent self-attention + FFN.
#[derive(Debug, Clone)]
pub struct PerceiverSelfLayer {
    /// Multi-head self-attention weights over the latent array (`latent_dim`).
    pub self_attn: CrossAttnWeights,
    /// Feed-forward network `latent_dim → 4·latent_dim → latent_dim`.
    pub ffn: FeedForward,
    /// LayerNorm applied before self-attention.
    pub ln_self: LayerNorm,
    /// LayerNorm applied before the FFN.
    pub ln_ffn: LayerNorm,
}

// ─── Weights ─────────────────────────────────────────────────────────────────

/// All learnable parameters of a Perceiver IO model.
#[derive(Debug, Clone)]
pub struct PerceiverIoWeights {
    /// Learnable latent array `[n_latents × latent_dim]`.
    pub latents: Vec<f32>,
    /// Input projection `[input_dim × latent_dim]` (row-major).
    pub input_proj: Vec<f32>,
    /// Encoder cross-attention weights (Q = latent, K/V = projected inputs).
    pub encode_attn: CrossAttnWeights,
    /// LayerNorm before the encode cross-attention (applied to latents).
    pub ln_encode_q: LayerNorm,
    /// LayerNorm before the encode cross-attention (applied to inputs).
    pub ln_encode_kv: LayerNorm,
    /// Latent self-attention / FFN stack.
    pub self_layers: Vec<PerceiverSelfLayer>,
    /// Learnable output query array `[n_outputs × output_dim]`.
    pub out_queries: Vec<f32>,
    /// Output query projection `[output_dim × latent_dim]` (row-major).
    pub query_proj: Vec<f32>,
    /// Decoder cross-attention weights (Q = projected queries, K/V = latents).
    pub decode_attn: CrossAttnWeights,
    /// LayerNorm before the decode cross-attention (applied to queries).
    pub ln_decode_q: LayerNorm,
    /// LayerNorm before the decode cross-attention (applied to latents).
    pub ln_decode_kv: LayerNorm,
    /// Output projection `[latent_dim × output_dim]` (row-major).
    pub output_proj: Vec<f32>,
}

impl PerceiverIoWeights {
    /// Randomly initialise all weights from N(0, scale²) Gaussian noise.
    fn random(cfg: &PerceiverIoConfig, rng: &mut LcgRng) -> Self {
        let input_dim = cfg.input_dim;
        let latent_dim = cfg.latent_dim;
        let output_dim = cfg.output_dim;
        let ff = latent_dim.max(4) * 4;

        let latent_scale = (1.0 / latent_dim as f32).sqrt();
        let input_proj_scale = (1.0 / input_dim as f32).sqrt();
        let output_proj_scale = (1.0 / latent_dim as f32).sqrt();
        let query_proj_scale = (1.0 / output_dim as f32).sqrt();
        let ffn_in_scale = (1.0 / latent_dim as f32).sqrt();
        let ffn_out_scale = (1.0 / ff as f32).sqrt();

        let latents = gaussian_vec(cfg.n_latents * latent_dim, latent_scale, rng);
        let input_proj = gaussian_vec(input_dim * latent_dim, input_proj_scale, rng);
        let encode_attn = random_attn_weights(latent_dim, latent_scale, rng);

        let mut self_layers = Vec::with_capacity(cfg.n_self_layers);
        for _ in 0..cfg.n_self_layers {
            let self_attn = random_attn_weights(latent_dim, latent_scale, rng);
            let ffn = FeedForward {
                w1: gaussian_vec(latent_dim * ff, ffn_in_scale, rng),
                b1: vec![0.0_f32; ff],
                w2: gaussian_vec(ff * latent_dim, ffn_out_scale, rng),
                b2: vec![0.0_f32; latent_dim],
                d_model: latent_dim,
                d_ff: ff,
            };
            self_layers.push(PerceiverSelfLayer {
                self_attn,
                ffn,
                ln_self: LayerNorm::ones(latent_dim),
                ln_ffn: LayerNorm::ones(latent_dim),
            });
        }

        let out_queries = gaussian_vec(cfg.n_outputs * output_dim, query_proj_scale, rng);
        let query_proj = gaussian_vec(output_dim * latent_dim, query_proj_scale, rng);
        let decode_attn = random_attn_weights(latent_dim, latent_scale, rng);
        let output_proj = gaussian_vec(latent_dim * output_dim, output_proj_scale, rng);

        Self {
            latents,
            input_proj,
            encode_attn,
            ln_encode_q: LayerNorm::ones(latent_dim),
            ln_encode_kv: LayerNorm::ones(latent_dim),
            self_layers,
            out_queries,
            query_proj,
            decode_attn,
            ln_decode_q: LayerNorm::ones(latent_dim),
            ln_decode_kv: LayerNorm::ones(latent_dim),
            output_proj,
        }
    }
}

fn gaussian_vec(len: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut v = vec![0.0_f32; len];
    rng.fill_normal(&mut v);
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

fn random_attn_weights(d: usize, scale: f32, rng: &mut LcgRng) -> CrossAttnWeights {
    CrossAttnWeights {
        w_q: gaussian_vec(d * d, scale, rng),
        w_k: gaussian_vec(d * d, scale, rng),
        w_v: gaussian_vec(d * d, scale, rng),
        w_o: gaussian_vec(d * d, scale, rng),
    }
}

// ─── PerceiverIo ─────────────────────────────────────────────────────────────

/// Perceiver IO module.
#[derive(Debug, Clone)]
pub struct PerceiverIo {
    pub cfg: PerceiverIoConfig,
    pub weights: PerceiverIoWeights,
}

impl PerceiverIo {
    /// Create a new PerceiverIo with randomly initialised weights.
    pub fn new(cfg: PerceiverIoConfig, rng: &mut LcgRng) -> MmResult<Self> {
        cfg.validate()?;
        let weights = PerceiverIoWeights::random(&cfg, rng);
        Ok(Self { cfg, weights })
    }

    /// Encode: latent array cross-attends over the inputs.
    ///
    /// `inputs`: `[n_inputs × input_dim]` row-major. The output shape is
    /// `n_latents × latent_dim` — **independent of `n_inputs`**.
    pub fn encode(&self, inputs: &[f32], n_inputs: usize) -> MmResult<Vec<f32>> {
        let input_dim = self.cfg.input_dim;
        let latent_dim = self.cfg.latent_dim;
        let n_latents = self.cfg.n_latents;

        if n_inputs == 0 {
            return Err(MultiModalError::EmptyInput);
        }
        if inputs.len() != n_inputs * input_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_inputs * input_dim,
                got: inputs.len(),
            });
        }

        // Project inputs into latent_dim so the cross-attention can run in a
        // uniform dimension. The K/V sequence has length `n_inputs` and the Q
        // sequence has length `n_latents`.
        let proj_inputs = matmul_seq(
            inputs,
            &self.weights.input_proj,
            n_inputs,
            input_dim,
            latent_dim,
        )?;

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, latent_dim, 0.0)?;
        let attn = CrossAttention::with_weights(attn_cfg, self.weights.encode_attn.clone());

        // Pre-norm both sides.
        let ln_q = self
            .weights
            .ln_encode_q
            .forward(&self.weights.latents, n_latents)?;
        let ln_kv = self.weights.ln_encode_kv.forward(&proj_inputs, n_inputs)?;

        let cross_out = attn.forward(&ln_q, &ln_kv, &ln_kv, n_latents, n_inputs)?;

        // Residual: out = latents + cross_attn(LN(latents), LN(proj_inputs))
        let out = add_vecs(&self.weights.latents, &cross_out)?;
        Ok(out)
    }

    /// Process: apply `n_self_layers` of latent self-attention + FFN.
    /// Shape is preserved at `n_latents × latent_dim`.
    pub fn process(&self, latents: &[f32]) -> MmResult<Vec<f32>> {
        let latent_dim = self.cfg.latent_dim;
        let n_latents = self.cfg.n_latents;
        if latents.len() != n_latents * latent_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_latents * latent_dim,
                got: latents.len(),
            });
        }

        if self.weights.self_layers.is_empty() {
            return Ok(latents.to_vec());
        }

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, latent_dim, 0.0)?;

        let mut x = latents.to_vec();
        for layer in &self.weights.self_layers {
            // ── Self-attention (pre-norm + residual) ────────────────────────
            let self_attn = CrossAttention::with_weights(attn_cfg.clone(), layer.self_attn.clone());
            let ln_x = layer.ln_self.forward(&x, n_latents)?;
            let attn_out = self_attn.forward(&ln_x, &ln_x, &ln_x, n_latents, n_latents)?;
            add_in_place(&mut x, &attn_out)?;

            // ── FFN (pre-norm + residual) ───────────────────────────────────
            let ln_f = layer.ln_ffn.forward(&x, n_latents)?;
            let ffn_out = layer.ffn.forward(&ln_f, n_latents)?;
            add_in_place(&mut x, &ffn_out)?;
        }
        Ok(x)
    }

    /// Decode: output queries cross-attend over the refined latents.
    pub fn decode(&self, latents: &[f32]) -> MmResult<Vec<f32>> {
        let latent_dim = self.cfg.latent_dim;
        let output_dim = self.cfg.output_dim;
        let n_latents = self.cfg.n_latents;
        let n_outputs = self.cfg.n_outputs;

        if latents.len() != n_latents * latent_dim {
            return Err(MultiModalError::DimensionMismatch {
                expected: n_latents * latent_dim,
                got: latents.len(),
            });
        }

        // Project output queries into latent_dim so attention has matching K/V.
        let proj_queries = matmul_seq(
            &self.weights.out_queries,
            &self.weights.query_proj,
            n_outputs,
            output_dim,
            latent_dim,
        )?;

        let attn_cfg = CrossAttnConfig::new(self.cfg.n_heads, latent_dim, 0.0)?;
        let attn = CrossAttention::with_weights(attn_cfg, self.weights.decode_attn.clone());

        let ln_q = self.weights.ln_decode_q.forward(&proj_queries, n_outputs)?;
        let ln_kv = self.weights.ln_decode_kv.forward(latents, n_latents)?;

        let cross_out = attn.forward(&ln_q, &ln_kv, &ln_kv, n_outputs, n_latents)?;

        // Project back into output_dim and return.
        let out = matmul_seq(
            &cross_out,
            &self.weights.output_proj,
            n_outputs,
            latent_dim,
            output_dim,
        )?;
        Ok(out)
    }

    /// Full forward: `encode → process → decode`.
    pub fn forward(&self, inputs: &[f32], n_inputs: usize) -> MmResult<Vec<f32>> {
        let encoded = self.encode(inputs, n_inputs)?;
        let processed = self.process(&encoded)?;
        self.decode(&processed)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Matrix multiply: `A [rows × in_dim] × W [in_dim × out_dim]` → `[rows × out_dim]`.
fn matmul_seq(
    a: &[f32],
    w: &[f32],
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) -> MmResult<Vec<f32>> {
    if a.len() != rows * in_dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: rows * in_dim,
            got: a.len(),
        });
    }
    if w.len() != in_dim * out_dim {
        return Err(MultiModalError::DimensionMismatch {
            expected: in_dim * out_dim,
            got: w.len(),
        });
    }
    let mut out = vec![0.0_f32; rows * out_dim];
    for r in 0..rows {
        for o in 0..out_dim {
            let mut acc = 0.0_f32;
            for i in 0..in_dim {
                acc += a[r * in_dim + i] * w[i * out_dim + o];
            }
            out[r * out_dim + o] = acc;
        }
    }
    Ok(out)
}

/// Add two equally-sized vectors element-wise.
fn add_vecs(a: &[f32], b: &[f32]) -> MmResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(MultiModalError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
}

fn add_in_place(acc: &mut [f32], delta: &[f32]) -> MmResult<()> {
    if acc.len() != delta.len() {
        return Err(MultiModalError::DimensionMismatch {
            expected: acc.len(),
            got: delta.len(),
        });
    }
    for (a, d) in acc.iter_mut().zip(delta.iter()) {
        *a += *d;
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_perceiver(seed: u64) -> PerceiverIo {
        let mut rng = LcgRng::new(seed);
        match PerceiverIo::new(PerceiverIoConfig::tiny(), &mut rng) {
            Ok(p) => p,
            Err(e) => panic!("tiny PerceiverIo should construct: {e:?}"),
        }
    }

    // ── 1: encode output length == n_latents * latent_dim, IS INDEPENDENT OF
    //       n_inputs ─────────────────────────────────────────────────────────
    #[test]
    fn encode_output_length_is_size_independent() {
        let p = make_perceiver(1);
        let cfg = &p.cfg;
        let in3 = vec![0.1_f32; 3 * cfg.input_dim];
        let in9 = vec![0.1_f32; 9 * cfg.input_dim];
        let e3 = p.encode(&in3, 3).unwrap();
        let e9 = p.encode(&in9, 9).unwrap();
        let expected = cfg.n_latents * cfg.latent_dim;
        assert_eq!(e3.len(), expected);
        assert_eq!(e9.len(), expected);
        assert_eq!(e3.len(), e9.len());
    }

    // ── 2: process preserves shape ──────────────────────────────────────────
    #[test]
    fn process_preserves_shape() {
        let p = make_perceiver(2);
        let cfg = &p.cfg;
        let latents = vec![0.2_f32; cfg.n_latents * cfg.latent_dim];
        let out = p.process(&latents).unwrap();
        assert_eq!(out.len(), cfg.n_latents * cfg.latent_dim);
    }

    // ── 3: decode output length == n_outputs * output_dim ────────────────────
    #[test]
    fn decode_output_length() {
        let p = make_perceiver(3);
        let cfg = &p.cfg;
        let latents = vec![0.15_f32; cfg.n_latents * cfg.latent_dim];
        let out = p.decode(&latents).unwrap();
        assert_eq!(out.len(), cfg.n_outputs * cfg.output_dim);
    }

    // ── 4: forward output length == n_outputs * output_dim ───────────────────
    #[test]
    fn forward_output_length() {
        let p = make_perceiver(4);
        let cfg = &p.cfg;
        let inputs = vec![0.1_f32; 5 * cfg.input_dim];
        let out = p.forward(&inputs, 5).unwrap();
        assert_eq!(out.len(), cfg.n_outputs * cfg.output_dim);
    }

    // ── 5: n_self_layers = 0 → process is identity ───────────────────────────
    #[test]
    fn n_self_layers_zero_process_is_identity() {
        let mut rng = LcgRng::new(5);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 8,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 0,
            output_dim: 10,
            n_outputs: 3,
        };
        let p = PerceiverIo::new(cfg, &mut rng).unwrap();
        let latents: Vec<f32> = (0..p.cfg.n_latents * p.cfg.latent_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let out = p.process(&latents).unwrap();
        assert_eq!(out, latents);
    }

    // ── 6: deterministic given seed ─────────────────────────────────────────
    #[test]
    fn deterministic_given_seed() {
        let a = make_perceiver(6);
        let b = make_perceiver(6);
        let inputs = vec![0.3_f32; 5 * a.cfg.input_dim];
        let out_a = a.forward(&inputs, 5).unwrap();
        let out_b = b.forward(&inputs, 5).unwrap();
        assert_eq!(out_a, out_b);
    }

    // ── 7: output finite ────────────────────────────────────────────────────
    #[test]
    fn forward_output_finite() {
        let p = make_perceiver(7);
        let inputs = vec![0.25_f32; 6 * p.cfg.input_dim];
        let out = p.forward(&inputs, 6).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 8: latent_dim % n_heads != 0 → Err ──────────────────────────────────
    #[test]
    fn latent_dim_not_divisible_errors() {
        let mut rng = LcgRng::new(8);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 9,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 3,
        };
        let err = PerceiverIo::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidHeads { .. }));
    }

    // ── 9: inputs wrong length → Err ────────────────────────────────────────
    #[test]
    fn inputs_wrong_length_errors() {
        let p = make_perceiver(9);
        let inputs = vec![0.1_f32; 3 * p.cfg.input_dim]; // claim 4 tokens
        let err = p.encode(&inputs, 4).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 10: single input token (n_inputs = 1) works ─────────────────────────
    #[test]
    fn single_input_token_works() {
        let p = make_perceiver(10);
        let inputs = vec![0.4_f32; p.cfg.input_dim];
        let out = p.forward(&inputs, 1).unwrap();
        assert_eq!(out.len(), p.cfg.n_outputs * p.cfg.output_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 11: single latent works ─────────────────────────────────────────────
    #[test]
    fn single_latent_works() {
        let mut rng = LcgRng::new(11);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 8,
            n_latents: 1,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 3,
        };
        let p = PerceiverIo::new(cfg, &mut rng).unwrap();
        let inputs = vec![0.2_f32; 5 * p.cfg.input_dim];
        let out = p.forward(&inputs, 5).unwrap();
        assert_eq!(out.len(), p.cfg.n_outputs * p.cfg.output_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── 12: changing inputs changes encoded latents ─────────────────────────
    #[test]
    fn changing_inputs_changes_encoded_latents() {
        let p = make_perceiver(12);
        let in_a = vec![0.1_f32; 4 * p.cfg.input_dim];
        let mut in_b = vec![0.1_f32; 4 * p.cfg.input_dim];
        for (i, v) in in_b.iter_mut().enumerate() {
            *v = 0.1 + (i as f32 * 0.07).sin();
        }
        let e_a = p.encode(&in_a, 4).unwrap();
        let e_b = p.encode(&in_b, 4).unwrap();
        let diff: f32 = e_a.iter().zip(e_b.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-4,
            "encoded latents should depend on inputs, diff={diff}"
        );
    }

    // ── 13: input_dim = 0 → Err ─────────────────────────────────────────────
    #[test]
    fn input_dim_zero_errors() {
        let mut rng = LcgRng::new(13);
        let cfg = PerceiverIoConfig {
            input_dim: 0,
            latent_dim: 8,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 3,
        };
        let err = PerceiverIo::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 14: latent_dim = 0 → Err ────────────────────────────────────────────
    #[test]
    fn latent_dim_zero_errors() {
        let mut rng = LcgRng::new(14);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 0,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 3,
        };
        let err = PerceiverIo::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 15: output_dim = 0 → Err ────────────────────────────────────────────
    #[test]
    fn output_dim_zero_errors() {
        let mut rng = LcgRng::new(15);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 8,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 0,
            n_outputs: 3,
        };
        let err = PerceiverIo::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::InvalidFeatureDim));
    }

    // ── 16: n_latents = 0 → Err ─────────────────────────────────────────────
    #[test]
    fn n_latents_zero_errors() {
        let mut rng = LcgRng::new(16);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 8,
            n_latents: 0,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 3,
        };
        let err = PerceiverIo::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    // ── 17: n_outputs = 0 → Err ─────────────────────────────────────────────
    #[test]
    fn n_outputs_zero_errors() {
        let mut rng = LcgRng::new(17);
        let cfg = PerceiverIoConfig {
            input_dim: 6,
            latent_dim: 8,
            n_latents: 4,
            n_heads: 2,
            n_self_layers: 2,
            output_dim: 10,
            n_outputs: 0,
        };
        let err = PerceiverIo::new(cfg, &mut rng).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
    }

    // ── 18: n_inputs = 0 → Err ──────────────────────────────────────────────
    #[test]
    fn n_inputs_zero_errors() {
        let p = make_perceiver(18);
        let err = p.encode(&[], 0).unwrap_err();
        assert!(matches!(err, MultiModalError::EmptyInput));
        let err2 = p.forward(&[], 0).unwrap_err();
        assert!(matches!(err2, MultiModalError::EmptyInput));
    }

    // ── 19: process input length mismatch → Err ──────────────────────────────
    #[test]
    fn process_input_length_mismatch_errors() {
        let p = make_perceiver(19);
        let wrong = vec![0.0_f32; p.cfg.n_latents * p.cfg.latent_dim - 1];
        let err = p.process(&wrong).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 20: decode latents length mismatch → Err ────────────────────────────
    #[test]
    fn decode_input_length_mismatch_errors() {
        let p = make_perceiver(20);
        let wrong = vec![0.0_f32; p.cfg.n_latents * p.cfg.latent_dim + 1];
        let err = p.decode(&wrong).unwrap_err();
        assert!(matches!(err, MultiModalError::DimensionMismatch { .. }));
    }

    // ── 21: encoded latents finite ───────────────────────────────────────────
    #[test]
    fn encode_output_finite() {
        let p = make_perceiver(21);
        let inputs = vec![0.5_f32; 7 * p.cfg.input_dim];
        let out = p.encode(&inputs, 7).unwrap();
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
