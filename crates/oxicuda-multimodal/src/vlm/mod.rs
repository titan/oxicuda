//! Vision-language models (VLMs).
//!
//! End-to-end image+text architectures built on top of the crate's shared
//! encoder / projector / attention primitives:
//!
//! - [`llava_next`] — LLaVA-NeXT (LLaVA-1.6) AnyRes model.
//! - [`qwen_vl`] — Qwen-VL with a position-aware cross-attention resampler.

use crate::cross_attn::cross_attention::{CrossAttnConfig, CrossAttnWeights};
use crate::cross_attn::masked_mha::{MhaArgs, mha_with_weights};
use crate::cross_attn::self_cross_block::{FeedForward, LayerNorm};
use crate::error::{MmResult, MultiModalError};
use crate::handle::LcgRng;

pub mod llava_next;
pub mod qwen_vl;

pub use llava_next::{LlavaNext, LlavaNextConfig, LlavaNextWeights};
pub use qwen_vl::{QwenVl, QwenVlConfig, QwenVlWeights};

// ─── Shared causal language model ────────────────────────────────────────────

/// Weights of one pre-norm causal transformer layer (self-attention + FFN),
/// shared by every VLM language stack in this module.
#[derive(Debug, Clone)]
pub(crate) struct LmLayer {
    pub self_attn: CrossAttnWeights,
    pub ffn: FeedForward,
    pub ln1: LayerNorm,
    pub ln2: LayerNorm,
}

/// Build `n_layers` deterministically-random causal LM layers of width `d`.
pub(crate) fn random_lm_layers(
    n_layers: usize,
    d: usize,
    d_ff: usize,
    n_heads: usize,
    rng: &mut LcgRng,
) -> MmResult<Vec<LmLayer>> {
    let attn_cfg = CrossAttnConfig::new(n_heads, d, 0.0)?;
    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let ffn = FeedForward {
            w1: gaussian_vec(d * d_ff, 1.0 / (d as f32).sqrt(), rng),
            b1: vec![0.0_f32; d_ff],
            w2: gaussian_vec(d_ff * d, 1.0 / (d_ff as f32).sqrt(), rng),
            b2: vec![0.0_f32; d],
            d_model: d,
            d_ff,
        };
        layers.push(LmLayer {
            self_attn: CrossAttnWeights::random(&attn_cfg, rng),
            ffn,
            ln1: LayerNorm::ones(d),
            ln2: LayerNorm::ones(d),
        });
    }
    Ok(layers)
}

/// Run a pre-norm **causal** transformer over a fused `[seq × d]` sequence,
/// adding sinusoidal positions first and applying `final_ln` at the end.
pub(crate) fn run_causal_lm(
    layers: &[LmLayer],
    final_ln: &LayerNorm,
    fused: &[f32],
    seq: usize,
    d: usize,
    n_heads: usize,
) -> MmResult<Vec<f32>> {
    if fused.len() != seq * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: seq * d,
            got: fused.len(),
        });
    }
    let attn_cfg = CrossAttnConfig::new(n_heads, d, 0.0)?;
    let mut x = fused.to_vec();
    add_sinusoidal_pos(&mut x, seq, d);
    for layer in layers {
        let normed = layer.ln1.forward(&x, seq)?;
        let args = MhaArgs {
            query: &normed,
            key: &normed,
            value: &normed,
            q_len: seq,
            kv_len: seq,
            causal: true,
        };
        let (sa, _) = mha_with_weights(&args, &attn_cfg, &layer.self_attn)?;
        for (xi, si) in x.iter_mut().zip(sa.iter()) {
            *xi += si;
        }
        let normed2 = layer.ln2.forward(&x, seq)?;
        let ff = layer.ffn.forward(&normed2, seq)?;
        for (xi, fi) in x.iter_mut().zip(ff.iter()) {
            *xi += fi;
        }
    }
    final_ln.forward(&x, seq)
}

/// Head-averaged causal self-attention weights `[seq × seq]` of the **first**
/// LM layer — exposed so callers can verify the lower-triangular causal mask.
pub(crate) fn first_layer_causal_attention(
    layers: &[LmLayer],
    fused: &[f32],
    seq: usize,
    d: usize,
    n_heads: usize,
) -> MmResult<Vec<f32>> {
    if fused.len() != seq * d {
        return Err(MultiModalError::DimensionMismatch {
            expected: seq * d,
            got: fused.len(),
        });
    }
    let layer = layers.first().ok_or(MultiModalError::InvalidLayerCount)?;
    let attn_cfg = CrossAttnConfig::new(n_heads, d, 0.0)?;
    let mut x = fused.to_vec();
    add_sinusoidal_pos(&mut x, seq, d);
    let normed = layer.ln1.forward(&x, seq)?;
    let args = MhaArgs {
        query: &normed,
        key: &normed,
        value: &normed,
        q_len: seq,
        kv_len: seq,
        causal: true,
    };
    let (_, weights) = mha_with_weights(&args, &attn_cfg, &layer.self_attn)?;
    Ok(weights)
}

/// Apply a `[d × vocab]` LM head (+ bias) to hidden states `[seq × d]`,
/// returning logits `[seq × vocab]`.
pub(crate) fn lm_head_logits(
    hidden: &[f32],
    seq: usize,
    d: usize,
    lm_head: &[f32],
    lm_head_bias: &[f32],
) -> Vec<f32> {
    let vocab = lm_head_bias.len();
    let mut logits = vec![0.0_f32; seq * vocab];
    for s in 0..seq {
        for vi in 0..vocab {
            let mut acc = lm_head_bias[vi];
            for di in 0..d {
                acc += hidden[s * d + di] * lm_head[di * vocab + vi];
            }
            logits[s * vocab + vi] = acc;
        }
    }
    logits
}

// ─── Shared helpers ──────────────────────────────────────────────────────────

/// Allocate `len` deterministic N(0, `scale`²) samples.
pub(crate) fn gaussian_vec(len: usize, scale: f32, rng: &mut LcgRng) -> Vec<f32> {
    let mut v = vec![0.0_f32; len];
    rng.fill_normal(&mut v);
    for x in v.iter_mut() {
        *x *= scale;
    }
    v
}

/// Add standard sinusoidal absolute positional encodings (Vaswani et al. 2017)
/// in-place to a `[seq × d]` row-major sequence. Parameter-free and
/// deterministic, so it never hides whether the *content* of a sequence drives
/// the model's output.
pub(crate) fn add_sinusoidal_pos(x: &mut [f32], seq: usize, d: usize) {
    for pos in 0..seq {
        for i in 0..d {
            let half = (i / 2) as f32;
            let denom = 10_000.0_f32.powf(2.0 * half / d as f32);
            let angle = pos as f32 / denom;
            let pe = if i % 2 == 0 { angle.sin() } else { angle.cos() };
            x[pos * d + i] += pe;
        }
    }
}

/// Add a separable 2-D sinusoidal positional encoding to a grid of `[rows ×
/// cols]` feature vectors of width `d` (row-major, `[rows·cols × d]`). The first
/// half of the channels encodes the row index, the second half the column
/// index — the standard 2-D ViT/resampler scheme. Used to make the Qwen-VL
/// resampler position-aware over patch features.
pub(crate) fn add_2d_sinusoidal_pos(x: &mut [f32], rows: usize, cols: usize, d: usize) {
    let half = d / 2;
    for r in 0..rows {
        for c in 0..cols {
            let base = (r * cols + c) * d;
            // Row component → channels [0, half).
            for i in 0..half {
                let freq = (i / 2) as f32;
                let denom = 10_000.0_f32.powf(2.0 * freq / half.max(1) as f32);
                let angle = r as f32 / denom;
                let pe = if i % 2 == 0 { angle.sin() } else { angle.cos() };
                x[base + i] += pe;
            }
            // Column component → channels [half, d).
            for i in half..d {
                let freq = ((i - half) / 2) as f32;
                let denom = 10_000.0_f32.powf(2.0 * freq / (d - half).max(1) as f32);
                let angle = c as f32 / denom;
                let pe = if (i - half) % 2 == 0 {
                    angle.sin()
                } else {
                    angle.cos()
                };
                x[base + i] += pe;
            }
        }
    }
}
