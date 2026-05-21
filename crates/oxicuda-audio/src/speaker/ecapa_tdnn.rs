//! ECAPA-TDNN speaker embedding (Desplanques et al. 2020, Interspeech).
//!
//! Multi-scale dilated TDNN with Squeeze-and-Excitation attention and
//! Attentive Statistics Pooling (ASTP) for robust speaker embeddings.
//!
//! Architecture (input: `[T × feat_dim]` log-mel features):
//! 1. Initial TDNN conv (kernel, dilation=1) → ReLU → BN → `h_in [T × C]`
//! 2. Three SE-TDNN blocks with dilations `[d0, d1, d2]`:
//!    - Dilated TDNN conv → ReLU → BN
//!    - Squeeze-and-Excitation: global avg → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid → scale
//!    - Residual: h_b = h_b + h_in
//! 3. Multi-scale concat `[h_in, h0, h1, h2]` → 1×1 conv (4C→C) → ReLU → BN → `h_agg`
//! 4. Attentive Statistics Pooling → `[2C]`
//! 5. FC (2C → embed_dim) → BN → L2-normalise → speaker embedding `[embed_dim]`

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Private helpers ─────────────────────────────────────────────────────────

/// ReLU activation.
#[inline]
fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Sigmoid activation.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0_f32 / (1.0_f32 + (-x).exp())
}

/// Kaiming (He) uniform initialisation: uniform in `[-sqrt(6/fan_in), +sqrt(6/fan_in)]`.
fn kaiming_init(rng: &mut LcgRng, fan_in: usize, buf: &mut [f32]) {
    let limit = (6.0_f32 / fan_in as f32).sqrt();
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

/// Batch-normalisation in inference mode.
///
/// `h_norm = (h - mean) / sqrt(var + eps) * gamma + beta`
fn bn_infer(x: &[f32], gamma: &[f32], beta: &[f32], mean: &[f32], var: &[f32]) -> Vec<f32> {
    let c = gamma.len();
    let t = x.len() / c;
    let mut out = vec![0.0_f32; x.len()];
    for ti in 0..t {
        for ci in 0..c {
            let xv = x[ti * c + ci];
            let inv_std = 1.0_f32 / (var[ci] + 1e-5_f32).sqrt();
            out[ti * c + ci] = (xv - mean[ci]) * inv_std * gamma[ci] + beta[ci];
        }
    }
    out
}

/// In-place numerically stable softmax over a slice.
fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

/// Dilated TDNN causal-symmetric convolution.
///
/// Zero-padding for out-of-bounds frame indices.
///
/// - `x`: `[T × in_ch]` flat row-major.
/// - `weight`: `[out_ch × in_ch × kernel_size]` row-major.
/// - `bias`: `[out_ch]`.
/// - Returns `[T × out_ch]` after ReLU.
fn tdnn_dilated_conv(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    t: usize,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    dilation: usize,
) -> Vec<f32> {
    let half = (kernel_size as isize - 1) / 2;
    let mut out = vec![0.0_f32; t * out_ch];

    for t_out in 0..t {
        for o in 0..out_ch {
            let mut acc = bias[o];
            for k in 0..kernel_size {
                let offset = (k as isize - half) * dilation as isize;
                let t_src_raw = t_out as isize + offset;
                // Zero-padding: skip out-of-bounds frames.
                if t_src_raw < 0 || t_src_raw >= t as isize {
                    continue;
                }
                let t_src = t_src_raw as usize;
                for i in 0..in_ch {
                    // weight[o, i, k]
                    acc += weight[(o * in_ch + i) * kernel_size + k] * x[t_src * in_ch + i];
                }
            }
            out[t_out * out_ch + o] = relu(acc);
        }
    }
    out
}

/// Linear projection: `y = W * x + b`.
///
/// - `x_vec`: `[in_dim]`.
/// - `w`: `[out_dim × in_dim]`.
/// - `b`: `[out_dim]`.
/// - Returns `[out_dim]`.
fn linear(x_vec: &[f32], w: &[f32], b: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_dim];
    for o in 0..out_dim {
        let mut acc = b[o];
        for i in 0..in_dim {
            acc += w[o * in_dim + i] * x_vec[i];
        }
        out[o] = acc;
    }
    out
}

// ─── Public types ─────────────────────────────────────────────────────────────

/// Configuration for ECAPA-TDNN.
#[derive(Debug, Clone)]
pub struct EcapaTdnnConfig {
    /// Input feature dimension (typically 80 for log-mel filterbanks).
    pub feat_dim: usize,
    /// Number of channels in TDNN and SE layers.
    pub channels: usize,
    /// Output embedding dimension.
    pub embed_dim: usize,
    /// Kernel size for TDNN convolution layers.
    pub kernel_size: usize,
    /// Dilation rates for the three SE-TDNN blocks.
    pub dilations: [usize; 3],
    /// Reduction ratio for Squeeze-and-Excitation.
    pub se_reduction: usize,
    /// Hidden size for Attentive Statistics Pooling attention network.
    pub asp_hidden_dim: usize,
}

impl Default for EcapaTdnnConfig {
    fn default() -> Self {
        Self {
            feat_dim: 80,
            channels: 512,
            embed_dim: 192,
            kernel_size: 5,
            dilations: [2, 3, 4],
            se_reduction: 8,
            asp_hidden_dim: 128,
        }
    }
}

/// One SE-TDNN residual block.
///
/// Each block applies:
/// 1. Dilated TDNN conv → ReLU → BN
/// 2. Squeeze-and-Excitation channel attention
/// 3. Residual addition with the initial conv output `h_in`
#[derive(Debug)]
pub struct SeTdnnBlock {
    /// Dilated TDNN conv weight `[channels × channels × kernel_size]`.
    pub w_tdnn: Vec<f32>,
    /// Dilated TDNN conv bias `[channels]`.
    pub b_tdnn: Vec<f32>,
    /// BN scale (gamma) `[channels]`.
    pub bn_tdnn_g: Vec<f32>,
    /// BN bias (beta) `[channels]`.
    pub bn_tdnn_b: Vec<f32>,
    /// BN running mean `[channels]`.
    pub bn_tdnn_m: Vec<f32>,
    /// BN running variance `[channels]`.
    pub bn_tdnn_v: Vec<f32>,
    /// SE FC1 weight `[(channels/se_reduction) × channels]`.
    pub se_w1: Vec<f32>,
    /// SE FC1 bias `[channels/se_reduction]`.
    pub se_b1: Vec<f32>,
    /// SE FC2 weight `[channels × (channels/se_reduction)]`.
    pub se_w2: Vec<f32>,
    /// SE FC2 bias `[channels]`.
    pub se_b2: Vec<f32>,
}

impl SeTdnnBlock {
    /// Apply this SE-TDNN block.
    ///
    /// `x`: `[T × channels]`, `h_in`: `[T × channels]` (for residual).
    /// Returns `[T × channels]`.
    fn forward(
        &self,
        x: &[f32],
        h_in: &[f32],
        t: usize,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
        se_reduction: usize,
    ) -> Vec<f32> {
        // 1. Dilated TDNN conv → ReLU.
        let h = tdnn_dilated_conv(
            x,
            &self.w_tdnn,
            &self.b_tdnn,
            t,
            channels,
            channels,
            kernel_size,
            dilation,
        );

        // 2. BN.
        let h = bn_infer(
            &h,
            &self.bn_tdnn_g,
            &self.bn_tdnn_b,
            &self.bn_tdnn_m,
            &self.bn_tdnn_v,
        );

        // 3. SE attention.
        let se_dim = channels / se_reduction;

        // Global average pooling: gap[c] = mean_t h[t, c].
        let mut gap = vec![0.0_f32; channels];
        for ti in 0..t {
            for ci in 0..channels {
                gap[ci] += h[ti * channels + ci];
            }
        }
        let t_inv = 1.0_f32 / t as f32;
        for g in gap.iter_mut() {
            *g *= t_inv;
        }

        // FC1: channels → se_dim, ReLU.
        let mut z = linear(&gap, &self.se_w1, &self.se_b1, channels, se_dim);
        for v in z.iter_mut() {
            *v = relu(*v);
        }

        // FC2: se_dim → channels, Sigmoid → scale vector.
        let scale = linear(&z, &self.se_w2, &self.se_b2, se_dim, channels);
        let scale: Vec<f32> = scale.iter().map(|&v| sigmoid(v)).collect();

        // Apply channel scaling.
        let mut h_se = h;
        for ti in 0..t {
            for ci in 0..channels {
                h_se[ti * channels + ci] *= scale[ci];
            }
        }

        // 4. Residual: h_se + h_in.
        let mut out = h_se;
        for (o, &r) in out.iter_mut().zip(h_in.iter()) {
            *o += r;
        }
        out
    }
}

/// ECAPA-TDNN model for speaker embedding extraction.
#[derive(Debug)]
pub struct EcapaTdnn {
    cfg: EcapaTdnnConfig,
    // Initial TDNN conv: feat_dim → channels.
    w_in: Vec<f32>,
    b_in: Vec<f32>,
    bn_in_g: Vec<f32>,
    bn_in_b: Vec<f32>,
    bn_in_m: Vec<f32>,
    bn_in_v: Vec<f32>,
    // Three SE-TDNN blocks.
    se_blocks: Vec<SeTdnnBlock>,
    // Aggregation: 4*channels → channels (1×1 conv = linear over channel axis).
    w_agg: Vec<f32>,
    b_agg: Vec<f32>,
    bn_agg_g: Vec<f32>,
    bn_agg_b: Vec<f32>,
    bn_agg_m: Vec<f32>,
    bn_agg_v: Vec<f32>,
    // Attentive Statistics Pooling attention network.
    asp_w1: Vec<f32>,
    asp_b1: Vec<f32>,
    asp_w2: Vec<f32>,
    asp_b2: Vec<f32>,
    // Final FC: 2*channels → embed_dim.
    w_fc: Vec<f32>,
    b_fc: Vec<f32>,
    bn_fc_g: Vec<f32>,
    bn_fc_b: Vec<f32>,
    bn_fc_m: Vec<f32>,
    bn_fc_v: Vec<f32>,
}

impl EcapaTdnn {
    /// Create a new ECAPA-TDNN with Kaiming-uniform initialisation.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidEmbedDim`] if `embed_dim == 0` or
    /// [`AudioError::Internal`] if `se_reduction` does not evenly divide `channels`.
    pub fn new(cfg: EcapaTdnnConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        if cfg.embed_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if cfg.channels == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if cfg.feat_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if cfg.channels % cfg.se_reduction != 0 {
            return Err(AudioError::Internal(format!(
                "channels ({}) must be divisible by se_reduction ({})",
                cfg.channels, cfg.se_reduction
            )));
        }

        let c = cfg.channels;
        let fd = cfg.feat_dim;
        let ks = cfg.kernel_size;
        let se_dim = c / cfg.se_reduction;

        // Initial TDNN: feat_dim → channels, kernel=ks, dilation=1.
        let w_in_len = c * fd * ks;
        let mut w_in = vec![0.0_f32; w_in_len];
        kaiming_init(rng, fd * ks, &mut w_in);
        let b_in = vec![0.0_f32; c];
        let bn_in_g = vec![1.0_f32; c];
        let bn_in_b = vec![0.0_f32; c];
        let bn_in_m = vec![0.0_f32; c];
        let bn_in_v = vec![1.0_f32; c];

        // Three SE-TDNN blocks.
        let mut se_blocks = Vec::with_capacity(3);
        for d in &cfg.dilations {
            let w_tdnn_len = c * c * ks;
            let mut w_tdnn = vec![0.0_f32; w_tdnn_len];
            kaiming_init(rng, c * ks, &mut w_tdnn);
            let b_tdnn = vec![0.0_f32; c];
            let bn_tdnn_g = vec![1.0_f32; c];
            let bn_tdnn_b = vec![0.0_f32; c];
            let bn_tdnn_m = vec![0.0_f32; c];
            let bn_tdnn_v = vec![1.0_f32; c];

            let mut se_w1 = vec![0.0_f32; se_dim * c];
            kaiming_init(rng, c, &mut se_w1);
            let se_b1 = vec![0.0_f32; se_dim];

            let mut se_w2 = vec![0.0_f32; c * se_dim];
            kaiming_init(rng, se_dim, &mut se_w2);
            let se_b2 = vec![0.0_f32; c];

            se_blocks.push(SeTdnnBlock {
                w_tdnn,
                b_tdnn,
                bn_tdnn_g,
                bn_tdnn_b,
                bn_tdnn_m,
                bn_tdnn_v,
                se_w1,
                se_b1,
                se_w2,
                se_b2,
            });

            let _ = d; // dilation stored in cfg.dilations, used during forward
        }

        // Aggregation projection: 4*channels → channels (applied per frame).
        // The bias is set to a small positive value (1/sqrt(fan_in)) so that the
        // pre-activation distribution is shifted above zero, preventing ReLU from
        // collapsing all outputs at initialisation.
        let agg_in = 4 * c;
        let mut w_agg = vec![0.0_f32; c * agg_in];
        kaiming_init(rng, agg_in, &mut w_agg);
        let agg_bias_val = 1.0_f32 / (agg_in as f32).sqrt();
        let b_agg = vec![agg_bias_val; c];
        let bn_agg_g = vec![1.0_f32; c];
        let bn_agg_b = vec![0.0_f32; c];
        let bn_agg_m = vec![0.0_f32; c];
        let bn_agg_v = vec![1.0_f32; c];

        // ASP attention.
        let ahd = cfg.asp_hidden_dim;
        let mut asp_w1 = vec![0.0_f32; ahd * c];
        kaiming_init(rng, c, &mut asp_w1);
        let asp_b1 = vec![0.0_f32; ahd];
        let mut asp_w2 = vec![0.0_f32; ahd];
        kaiming_init(rng, ahd, &mut asp_w2);
        let asp_b2 = vec![0.0_f32; 1];

        // Final FC: 2*channels → embed_dim.
        let fc_in = 2 * c;
        let ed = cfg.embed_dim;
        let mut w_fc = vec![0.0_f32; ed * fc_in];
        kaiming_init(rng, fc_in, &mut w_fc);
        let b_fc = vec![0.0_f32; ed];
        let bn_fc_g = vec![1.0_f32; ed];
        let bn_fc_b = vec![0.0_f32; ed];
        let bn_fc_m = vec![0.0_f32; ed];
        let bn_fc_v = vec![1.0_f32; ed];

        Ok(Self {
            cfg,
            w_in,
            b_in,
            bn_in_g,
            bn_in_b,
            bn_in_m,
            bn_in_v,
            se_blocks,
            w_agg,
            b_agg,
            bn_agg_g,
            bn_agg_b,
            bn_agg_m,
            bn_agg_v,
            asp_w1,
            asp_b1,
            asp_w2,
            asp_b2,
            w_fc,
            b_fc,
            bn_fc_g,
            bn_fc_b,
            bn_fc_m,
            bn_fc_v,
        })
    }

    /// Forward pass: extract speaker embedding.
    ///
    /// # Arguments
    ///
    /// - `features`: `[t_frames × feat_dim]` log-mel features, flat row-major.
    /// - `t_frames`: Number of time frames `T`.
    ///
    /// # Returns
    ///
    /// Speaker embedding of length `embed_dim`, L2-normalised.
    ///
    /// # Errors
    ///
    /// - [`AudioError::EmptyInput`] if `T == 0`.
    /// - [`AudioError::DimensionMismatch`] if `features.len() ≠ T × feat_dim`.
    /// - [`AudioError::NonFinite`] if any intermediate value is NaN or infinite.
    pub fn forward(&self, features: &[f32], t_frames: usize) -> AudioResult<Vec<f32>> {
        let c = self.cfg.channels;
        let fd = self.cfg.feat_dim;
        let ks = self.cfg.kernel_size;
        let ed = self.cfg.embed_dim;
        let ahd = self.cfg.asp_hidden_dim;

        if t_frames == 0 {
            return Err(AudioError::EmptyInput {
                msg: "EcapaTdnn::forward: t_frames == 0".into(),
            });
        }
        let expected = t_frames * fd;
        if features.len() != expected {
            return Err(AudioError::DimensionMismatch {
                expected,
                got: features.len(),
            });
        }

        // ── Step 1: Initial TDNN conv ─────────────────────────────────────────
        let h_in_raw = tdnn_dilated_conv(features, &self.w_in, &self.b_in, t_frames, fd, c, ks, 1);
        let h_in = bn_infer(
            &h_in_raw,
            &self.bn_in_g,
            &self.bn_in_b,
            &self.bn_in_m,
            &self.bn_in_v,
        );

        // ── Step 2: Three SE-TDNN blocks ──────────────────────────────────────
        // Each block receives h_in as both its primary input and residual source.
        let se_reduction = self.cfg.se_reduction;
        let mut h_blocks: Vec<Vec<f32>> = Vec::with_capacity(3);
        for (block, &dilation) in self.se_blocks.iter().zip(self.cfg.dilations.iter()) {
            let h_b = block.forward(&h_in, &h_in, t_frames, c, ks, dilation, se_reduction);
            h_blocks.push(h_b);
        }

        // ── Step 3: Multi-scale aggregation ──────────────────────────────────
        // Concat [h_in, h_block[0], h_block[1], h_block[2]] → [T × 4C].
        let agg_in_dim = 4 * c;
        let mut concat = vec![0.0_f32; t_frames * agg_in_dim];
        for ti in 0..t_frames {
            let dst_base = ti * agg_in_dim;
            concat[dst_base..dst_base + c].copy_from_slice(&h_in[ti * c..(ti + 1) * c]);
            concat[dst_base + c..dst_base + 2 * c]
                .copy_from_slice(&h_blocks[0][ti * c..(ti + 1) * c]);
            concat[dst_base + 2 * c..dst_base + 3 * c]
                .copy_from_slice(&h_blocks[1][ti * c..(ti + 1) * c]);
            concat[dst_base + 3 * c..dst_base + 4 * c]
                .copy_from_slice(&h_blocks[2][ti * c..(ti + 1) * c]);
        }

        // 1×1 conv: 4C → C per frame (applied as a linear + ReLU per timestep).
        let mut h_agg_raw = vec![0.0_f32; t_frames * c];
        for ti in 0..t_frames {
            let frame = &concat[ti * agg_in_dim..(ti + 1) * agg_in_dim];
            let proj = linear(frame, &self.w_agg, &self.b_agg, agg_in_dim, c);
            for (o, v) in proj.into_iter().enumerate() {
                h_agg_raw[ti * c + o] = relu(v);
            }
        }
        let h_agg = bn_infer(
            &h_agg_raw,
            &self.bn_agg_g,
            &self.bn_agg_b,
            &self.bn_agg_m,
            &self.bn_agg_v,
        );

        // ── Step 4: Attentive Statistics Pooling ─────────────────────────────
        // Compute per-frame attention energy: e_t = w2 · ReLU(w1 · h_t + b1) + b2.
        let mut energies = vec![0.0_f32; t_frames];
        for ti in 0..t_frames {
            let h_t = &h_agg[ti * c..(ti + 1) * c];
            // FC1: c → ahd, ReLU.
            let mut z1 = linear(h_t, &self.asp_w1, &self.asp_b1, c, ahd);
            for v in z1.iter_mut() {
                *v = relu(*v);
            }
            // FC2: ahd → 1.
            let z2 = linear(&z1, &self.asp_w2, &self.asp_b2, ahd, 1);
            energies[ti] = z2[0];
        }

        // Softmax over T.
        softmax_inplace(&mut energies);

        // Weighted mean: μ = Σ_t α_t * h_t.
        let mut mean_vec = vec![0.0_f32; c];
        for ti in 0..t_frames {
            let alpha_t = energies[ti];
            for ci in 0..c {
                mean_vec[ci] += alpha_t * h_agg[ti * c + ci];
            }
        }

        // Weighted std: σ² = Σ_t α_t * h_t² - μ², element-wise; clamp ≥ 0.
        let mut var_vec = vec![0.0_f32; c];
        for ti in 0..t_frames {
            let alpha_t = energies[ti];
            for ci in 0..c {
                var_vec[ci] += alpha_t * h_agg[ti * c + ci] * h_agg[ti * c + ci];
            }
        }
        let std_vec: Vec<f32> = var_vec
            .iter()
            .zip(mean_vec.iter())
            .map(|(&v2, &mu)| {
                let v = (v2 - mu * mu).max(0.0_f32);
                v.sqrt()
            })
            .collect();

        // Pool = [mean; std] → [2C].
        let mut pool = vec![0.0_f32; 2 * c];
        pool[..c].copy_from_slice(&mean_vec);
        pool[c..].copy_from_slice(&std_vec);

        // ── Step 5: Final FC → BN → L2-normalise ─────────────────────────────
        let fc_in = 2 * c;
        let fc_out = linear(&pool, &self.w_fc, &self.b_fc, fc_in, ed);

        // BN on the [1 × ed] vector (treating as 1 frame of ed channels).
        let fc_bn = bn_infer(
            &fc_out,
            &self.bn_fc_g,
            &self.bn_fc_b,
            &self.bn_fc_m,
            &self.bn_fc_v,
        );

        // L2 normalise.
        let norm_sq: f32 = fc_bn.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt().max(1e-12_f32);
        let embedding: Vec<f32> = fc_bn.iter().map(|v| v / norm).collect();

        // Finite check.
        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(AudioError::NonFinite {
                msg: "EcapaTdnn::forward: embedding contains NaN or Inf".into(),
            });
        }

        Ok(embedding)
    }

    /// Return the total number of trainable parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        let c = self.cfg.channels;
        let fd = self.cfg.feat_dim;
        let ks = self.cfg.kernel_size;
        let se_dim = c / self.cfg.se_reduction;
        let agg_in = 4 * c;
        let ahd = self.cfg.asp_hidden_dim;
        let ed = self.cfg.embed_dim;
        let fc_in = 2 * c;

        let mut n = 0_usize;

        // Initial TDNN conv + BN.
        n += c * fd * ks + c; // w_in + b_in
        n += c + c + c + c; // bn gamma + beta + mean + var

        // Three SE-TDNN blocks.
        for _ in &self.se_blocks {
            n += c * c * ks + c; // w_tdnn + b_tdnn
            n += c + c + c + c; // bn
            n += se_dim * c + se_dim; // se_w1 + se_b1
            n += c * se_dim + c; // se_w2 + se_b2
        }

        // Aggregation.
        n += c * agg_in + c; // w_agg + b_agg
        n += c + c + c + c; // bn

        // ASP.
        n += ahd * c + ahd; // asp_w1 + asp_b1
        n += ahd + 1; // asp_w2 + asp_b2

        // Final FC + BN.
        n += ed * fc_in + ed; // w_fc + b_fc
        n += ed + ed + ed + ed; // bn

        n
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn small_cfg() -> EcapaTdnnConfig {
        EcapaTdnnConfig {
            feat_dim: 16,
            channels: 32,
            embed_dim: 16,
            kernel_size: 3,
            dilations: [2, 3, 4],
            se_reduction: 4,
            asp_hidden_dim: 8,
        }
    }

    fn random_features(t: usize, feat_dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = vec![0.0_f32; t * feat_dim];
        rng.fill_normal(&mut v);
        v
    }

    // ── Output shape ─────────────────────────────────────────────────────────

    #[test]
    fn ecapa_forward_shape() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let ed = cfg.embed_dim;
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(20, cfg.feat_dim, 1);
        let out = net.forward(&feats, 20).unwrap();
        assert_eq!(out.len(), ed, "embedding length must equal embed_dim");
    }

    // ── Output finite ─────────────────────────────────────────────────────────

    #[test]
    fn ecapa_forward_finite() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(30, cfg.feat_dim, 2);
        let out = net.forward(&feats, 30).unwrap();
        assert!(
            out.iter().all(|v| v.is_finite()),
            "embedding must be finite"
        );
    }

    // ── Non-zero ─────────────────────────────────────────────────────────────

    #[test]
    fn ecapa_embedding_nonzero() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(15, cfg.feat_dim, 3);
        let out = net.forward(&feats, 15).unwrap();
        let norm: f32 = out.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(norm > 1e-6, "embedding must be non-zero, norm={norm}");
    }

    // ── Different inputs produce different embeddings ──────────────────────────

    #[test]
    fn ecapa_different_inputs_differ() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let t = 20_usize;
        // Use maximally different inputs: all positive constant vs all negative constant.
        // After TDNN+ReLU the two inputs produce clearly different activations.
        let feats_a = vec![1.0_f32; t * cfg.feat_dim];
        let feats_b = vec![-1.0_f32; t * cfg.feat_dim];
        let out_a = net.forward(&feats_a, t).unwrap();
        let out_b = net.forward(&feats_b, t).unwrap();
        let diff: f32 = out_a
            .iter()
            .zip(out_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "different inputs must produce different embeddings"
        );
    }

    // ── Determinism ───────────────────────────────────────────────────────────

    #[test]
    fn ecapa_deterministic() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let t = 18_usize;
        let feats = random_features(t, cfg.feat_dim, 5);
        let out1 = net.forward(&feats, t).unwrap();
        let out2 = net.forward(&feats, t).unwrap();
        assert_eq!(out1, out2, "forward must be deterministic");
    }

    // ── Small config ──────────────────────────────────────────────────────────

    #[test]
    fn ecapa_small_config() {
        let cfg = EcapaTdnnConfig {
            feat_dim: 16,
            channels: 32,
            embed_dim: 16,
            kernel_size: 3,
            dilations: [2, 3, 4],
            se_reduction: 4,
            asp_hidden_dim: 8,
        };
        let mut rng = LcgRng::new(99);
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(10, cfg.feat_dim, 6);
        let out = net.forward(&feats, 10).unwrap();
        assert_eq!(out.len(), cfg.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Long sequence ─────────────────────────────────────────────────────────

    #[test]
    fn ecapa_long_sequence() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let t = 200_usize;
        let feats = random_features(t, cfg.feat_dim, 7);
        let out = net.forward(&feats, t).unwrap();
        assert_eq!(out.len(), cfg.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Short sequence ────────────────────────────────────────────────────────

    #[test]
    fn ecapa_short_sequence() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let t = 5_usize;
        let feats = random_features(t, cfg.feat_dim, 8);
        let out = net.forward(&feats, t).unwrap();
        assert_eq!(out.len(), cfg.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── n_params positive ─────────────────────────────────────────────────────

    #[test]
    fn ecapa_n_params_positive() {
        let mut rng = make_rng();
        let net = EcapaTdnn::new(small_cfg(), &mut rng).unwrap();
        assert!(net.n_params() > 0, "must have at least one parameter");
    }

    // ── Larger channels → more params ─────────────────────────────────────────

    #[test]
    fn ecapa_n_params_scale_with_channels() {
        let mut rng_a = LcgRng::new(1);
        let cfg_small = small_cfg();
        let n_small = EcapaTdnn::new(cfg_small, &mut rng_a).unwrap().n_params();

        let mut rng_b = LcgRng::new(2);
        let cfg_large = EcapaTdnnConfig {
            channels: 64,
            embed_dim: 32,
            ..small_cfg()
        };
        let n_large = EcapaTdnn::new(cfg_large, &mut rng_b).unwrap().n_params();

        assert!(
            n_large > n_small,
            "larger channels must yield more params: {n_large} vs {n_small}"
        );
    }

    // ── Default config constructs ─────────────────────────────────────────────

    #[test]
    fn ecapa_default_config() {
        let cfg = EcapaTdnnConfig::default();
        let mut rng = LcgRng::new(77);
        // Reduced channels for speed in CI.
        let cfg_fast = EcapaTdnnConfig {
            channels: 32,
            embed_dim: 16,
            asp_hidden_dim: 8,
            ..cfg
        };
        let net = EcapaTdnn::new(cfg_fast.clone(), &mut rng);
        assert!(
            net.is_ok(),
            "default-derived config must construct: {net:?}"
        );
    }

    // ── EmptyInput error ──────────────────────────────────────────────────────

    #[test]
    fn ecapa_err_empty_input() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg, &mut rng).unwrap();
        let err = net.forward(&[], 0).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput { .. }));
    }

    // ── DimensionMismatch error ───────────────────────────────────────────────

    #[test]
    fn ecapa_err_dim_mismatch() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        // Provide wrong number of features.
        let feats = vec![0.0_f32; 5]; // T=3, feat_dim=16 → expected 48, got 5.
        let err = net.forward(&feats, 3).unwrap_err();
        assert!(matches!(err, AudioError::DimensionMismatch { .. }));
    }

    // ── SE-block output shape ─────────────────────────────────────────────────

    #[test]
    fn se_block_output_shape() {
        // Build a single SE-TDNN block and verify the output shape is [T × C].
        let c = 32_usize;
        let ks = 3_usize;
        let se_dim = 8_usize;
        let t = 12_usize;
        let mut rng = LcgRng::new(55);

        let mut w_tdnn = vec![0.0_f32; c * c * ks];
        kaiming_init(&mut rng, c * ks, &mut w_tdnn);
        let mut se_w1 = vec![0.0_f32; se_dim * c];
        kaiming_init(&mut rng, c, &mut se_w1);
        let mut se_w2 = vec![0.0_f32; c * se_dim];
        kaiming_init(&mut rng, se_dim, &mut se_w2);

        let block = SeTdnnBlock {
            w_tdnn,
            b_tdnn: vec![0.0_f32; c],
            bn_tdnn_g: vec![1.0_f32; c],
            bn_tdnn_b: vec![0.0_f32; c],
            bn_tdnn_m: vec![0.0_f32; c],
            bn_tdnn_v: vec![1.0_f32; c],
            se_w1,
            se_b1: vec![0.0_f32; se_dim],
            se_w2,
            se_b2: vec![0.0_f32; c],
        };

        let x = vec![0.1_f32; t * c];
        let h_in = vec![0.05_f32; t * c];
        let out = block.forward(&x, &h_in, t, c, ks, 2, 4);
        assert_eq!(out.len(), t * c, "SE-block output must be [T × C]");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "SE-block output must be finite"
        );
    }

    // ── ASP pool dimension ────────────────────────────────────────────────────

    #[test]
    fn ecapa_asp_mean_std() {
        // Run forward and confirm the final pool (before FC) conceptually has 2*C.
        // We verify this indirectly: embed_dim < 2*C is supported.
        let cfg = EcapaTdnnConfig {
            feat_dim: 16,
            channels: 32,
            embed_dim: 8, // smaller than 2*C=64 → valid
            kernel_size: 3,
            dilations: [2, 3, 4],
            se_reduction: 4,
            asp_hidden_dim: 8,
        };
        let mut rng = LcgRng::new(66);
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(10, cfg.feat_dim, 9);
        let out = net.forward(&feats, 10).unwrap();
        assert_eq!(out.len(), cfg.embed_dim);
    }

    // ── Different feat_dim ────────────────────────────────────────────────────

    #[test]
    fn ecapa_feat_dim_mismatch_small() {
        let cfg = EcapaTdnnConfig {
            feat_dim: 40,
            channels: 32,
            embed_dim: 16,
            kernel_size: 3,
            dilations: [2, 3, 4],
            se_reduction: 4,
            asp_hidden_dim: 8,
        };
        let mut rng = LcgRng::new(33);
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(12, cfg.feat_dim, 11);
        let out = net.forward(&feats, 12).unwrap();
        assert_eq!(out.len(), cfg.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Single frame ──────────────────────────────────────────────────────────

    #[test]
    fn ecapa_single_frame() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(1, cfg.feat_dim, 12);
        let out = net.forward(&feats, 1).unwrap();
        assert_eq!(out.len(), cfg.embed_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── L2 norm ≈ 1.0 ─────────────────────────────────────────────────────────

    #[test]
    fn ecapa_embedding_bounded() {
        let mut rng = make_rng();
        let cfg = small_cfg();
        let net = EcapaTdnn::new(cfg.clone(), &mut rng).unwrap();
        let feats = random_features(25, cfg.feat_dim, 13);
        let out = net.forward(&feats, 25).unwrap();
        let norm_sq: f32 = out.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0_f32).abs() < 1e-5,
            "L2 norm after normalisation must be ≈ 1.0; got {norm}"
        );
    }
}
