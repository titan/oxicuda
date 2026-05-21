//! Conv-TasNet source separation (Luo & Mesgarani 2019, IEEE TASLP).
//!
//! Implements a fully convolutional time-domain audio source separation network
//! using a learned encoder, temporal convolutional network (TCN) separator,
//! and learned decoder for multi-source audio separation.
//!
//! Reference: Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency
//! Magnitude Masking for Speech Separation", IEEE TASLP 2019.
//!
//! Architecture overview:
//! ```text
//! waveform [N] → Encoder (strided conv bank) → features [F, T]
//!                ↓
//!              Layer Norm → 1×1 in_proj
//!              TCN (n_repeats × n_blocks, exp. dilations)
//!              skip sum → 1×1 mask_conv → ReLU → masks [S, F, T]
//!                ↓
//!              features × masks → per-source features
//!                ↓
//!              Decoder (transposed strided conv) → sources [S, N]
//! ```

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Kaiming uniform / layer-norm init ───────────────────────────────────────

fn kaiming_uniform_init(rng: &mut LcgRng, fan_in: usize, buf: &mut [f32]) {
    let limit = (6.0_f32 / fan_in.max(1) as f32).sqrt();
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

// ─── ConvTasNetConfig ────────────────────────────────────────────────────────

/// Configuration for Conv-TasNet.
#[derive(Debug, Clone)]
pub struct ConvTasNetConfig {
    /// Number of audio sources to separate (e.g., 2).
    pub n_sources: usize,
    /// Encoder convolution kernel size (e.g., 16).
    pub enc_kernel: usize,
    /// Encoder stride / decoder stride (e.g., 8).
    pub enc_stride: usize,
    /// Number of encoder filters / feature dimension (e.g., 512).
    pub n_filters: usize,
    /// TCN bottleneck dimension (e.g., 128).
    pub bottleneck_dim: usize,
    /// TCN depthwise convolution hidden dimension (e.g., 512).
    pub hidden_dim: usize,
    /// Number of TCN blocks per repeat (e.g., 8).
    pub n_blocks: usize,
    /// Number of TCN repeats (e.g., 3).
    pub n_repeats: usize,
    /// If true, apply causal padding in TCN depthwise convolutions.
    pub causal: bool,
}

impl ConvTasNetConfig {
    /// Validate and return error on invalid configuration.
    pub fn validate(&self) -> AudioResult<()> {
        if self.n_sources == 0 {
            return Err(AudioError::EmptyInput {
                msg: "n_sources must be > 0".into(),
            });
        }
        if self.enc_kernel == 0 {
            return Err(AudioError::InvalidKernelSize(0));
        }
        if self.enc_stride == 0 {
            return Err(AudioError::InvalidStride(0));
        }
        if self.n_filters == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.bottleneck_dim == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.n_blocks == 0 {
            return Err(AudioError::EmptyInput {
                msg: "n_blocks must be > 0".into(),
            });
        }
        Ok(())
    }
}

// ─── TcnBlockWeights ─────────────────────────────────────────────────────────

/// Weights for a single TCN block.
#[derive(Debug, Clone)]
pub struct TcnBlockWeights {
    /// 1×1 bottleneck_dim → hidden_dim convolution weight: `[hidden_dim × bottleneck_dim]`.
    pub in_conv_w: Vec<f32>,
    /// Bias: `[hidden_dim]`.
    pub in_conv_b: Vec<f32>,
    /// Depthwise kernel weight: `[hidden_dim × k]` (one filter per channel).
    pub dw_conv_w: Vec<f32>,
    /// Depthwise bias: `[hidden_dim]`.
    pub dw_conv_b: Vec<f32>,
    /// 1×1 hidden_dim → bottleneck_dim (residual) weight: `[bottleneck_dim × hidden_dim]`.
    pub pw_conv_w: Vec<f32>,
    /// Bias: `[bottleneck_dim]`.
    pub pw_conv_b: Vec<f32>,
    /// 1×1 hidden_dim → bottleneck_dim (skip connection) weight: `[bottleneck_dim × hidden_dim]`.
    pub skip_w: Vec<f32>,
    /// Bias: `[bottleneck_dim]`.
    pub skip_b: Vec<f32>,
}

// ─── ConvTasNetWeights ───────────────────────────────────────────────────────

/// All learnable weights for Conv-TasNet.
#[derive(Debug, Clone)]
pub struct ConvTasNetWeights {
    /// Encoder conv bank: `[n_filters × enc_kernel]`, row-major.
    pub encoder_w: Vec<f32>,
    /// Decoder transposed-conv bank: `[enc_kernel × n_filters]`, row-major.
    pub decoder_w: Vec<f32>,
    /// Layer norm gamma (scale): `[n_filters]`.
    pub layer_norm_gamma: Vec<f32>,
    /// Layer norm beta (shift): `[n_filters]`.
    pub layer_norm_beta: Vec<f32>,
    /// Input projection 1×1 conv: `[bottleneck_dim × n_filters]`.
    pub in_proj_w: Vec<f32>,
    /// TCN blocks in order (n_repeats * n_blocks).
    pub tcn_blocks: Vec<TcnBlockWeights>,
    /// Mask output conv weight: `[n_sources*n_filters × bottleneck_dim]`.
    pub mask_conv_w: Vec<f32>,
    /// Mask output conv bias: `[n_sources*n_filters]`.
    pub mask_conv_b: Vec<f32>,
}

// ─── SeparationResult ────────────────────────────────────────────────────────

/// Result of source separation.
pub struct SeparationResult {
    /// Separated source waveforms: [n_sources × n_samples] row-major.
    pub sources: Vec<f32>,
    /// Number of separated sources.
    pub n_sources: usize,
    /// Number of audio samples per source.
    pub n_samples: usize,
}

// ─── ConvTasNet ──────────────────────────────────────────────────────────────

/// Conv-TasNet source separation network.
pub struct ConvTasNet {
    /// Configuration.
    pub cfg: ConvTasNetConfig,
    /// All weights.
    pub weights: ConvTasNetWeights,
}

impl ConvTasNet {
    /// Create a new Conv-TasNet with Kaiming-uniform initialised weights.
    pub fn new(cfg: ConvTasNetConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        cfg.validate()?;

        let n_src = cfg.n_sources;
        let enc_k = cfg.enc_kernel;
        let n_filt = cfg.n_filters;
        let bn_dim = cfg.bottleneck_dim;
        let hid_dim = cfg.hidden_dim;
        let n_total_blocks = cfg.n_repeats * cfg.n_blocks;
        let dw_k = 3usize; // depthwise kernel size (fixed = 3, as in paper)

        // Encoder
        let mut encoder_w = vec![0.0f32; n_filt * enc_k];
        kaiming_uniform_init(rng, enc_k, &mut encoder_w);

        // Decoder: enc_kernel × n_filters
        let mut decoder_w = vec![0.0f32; enc_k * n_filt];
        kaiming_uniform_init(rng, n_filt, &mut decoder_w);

        // Layer norm
        let layer_norm_gamma = vec![1.0f32; n_filt];
        let layer_norm_beta = vec![0.0f32; n_filt];

        // Input projection: bottleneck_dim × n_filters
        let mut in_proj_w = vec![0.0f32; bn_dim * n_filt];
        kaiming_uniform_init(rng, n_filt, &mut in_proj_w);

        // TCN blocks
        let mut tcn_blocks = Vec::with_capacity(n_total_blocks);
        for block_idx in 0..n_total_blocks {
            let b_in_repeat = block_idx % cfg.n_blocks;
            let _dilation = 1usize << b_in_repeat; // 2^block_idx_in_repeat

            // in_conv: hidden_dim × bottleneck_dim
            let mut in_conv_w = vec![0.0f32; hid_dim * bn_dim];
            kaiming_uniform_init(rng, bn_dim, &mut in_conv_w);
            let in_conv_b = vec![0.0f32; hid_dim];

            // dw_conv: hidden_dim × dw_k (depthwise: one kernel per channel)
            let mut dw_conv_w = vec![0.0f32; hid_dim * dw_k];
            kaiming_uniform_init(rng, dw_k, &mut dw_conv_w);
            let dw_conv_b = vec![0.0f32; hid_dim];

            // pw_conv: bottleneck_dim × hidden_dim
            let mut pw_conv_w = vec![0.0f32; bn_dim * hid_dim];
            kaiming_uniform_init(rng, hid_dim, &mut pw_conv_w);
            let pw_conv_b = vec![0.0f32; bn_dim];

            // skip: bottleneck_dim × hidden_dim
            let mut skip_w = vec![0.0f32; bn_dim * hid_dim];
            kaiming_uniform_init(rng, hid_dim, &mut skip_w);
            let skip_b = vec![0.0f32; bn_dim];

            tcn_blocks.push(TcnBlockWeights {
                in_conv_w,
                in_conv_b,
                dw_conv_w,
                dw_conv_b,
                pw_conv_w,
                pw_conv_b,
                skip_w,
                skip_b,
            });
        }

        // Mask conv: (n_sources * n_filters) × bottleneck_dim
        let mask_out_ch = n_src * n_filt;
        let mut mask_conv_w = vec![0.0f32; mask_out_ch * bn_dim];
        kaiming_uniform_init(rng, bn_dim, &mut mask_conv_w);
        let mask_conv_b = vec![0.0f32; mask_out_ch];

        Ok(Self {
            cfg,
            weights: ConvTasNetWeights {
                encoder_w,
                decoder_w,
                layer_norm_gamma,
                layer_norm_beta,
                in_proj_w,
                tcn_blocks,
                mask_conv_w,
                mask_conv_b,
            },
        })
    }

    /// Encode waveform into feature representation.
    ///
    /// # Arguments
    /// - `waveform`: raw audio samples.
    ///
    /// # Returns
    /// `(features, n_frames)` where features is `[n_frames × n_filters]` row-major,
    /// `n_frames = (n_samples - enc_kernel) / enc_stride + 1`.
    pub fn encode(&self, waveform: &[f32]) -> AudioResult<(Vec<f32>, usize)> {
        let n = waveform.len();
        let enc_k = self.cfg.enc_kernel;
        let stride = self.cfg.enc_stride;
        let n_filt = self.cfg.n_filters;

        if n < enc_k {
            return Err(AudioError::InvalidSequenceLength(n));
        }

        let n_frames = (n - enc_k) / stride + 1;
        let mut features = vec![0.0f32; n_frames * n_filt];

        for f in 0..n_frames {
            let t_start = f * stride;
            for k in 0..n_filt {
                let mut acc = 0.0f32;
                for t in 0..enc_k {
                    acc += waveform[t_start + t] * self.weights.encoder_w[k * enc_k + t];
                }
                // ReLU activation
                features[f * n_filt + k] = acc.max(0.0);
            }
        }
        Ok((features, n_frames))
    }

    /// TCN separator: features → masks for each source.
    ///
    /// # Arguments
    /// - `features`: encoder output `[n_frames × n_filters]` row-major.
    /// - `n_frames`: number of frames.
    ///
    /// # Returns
    /// Masks `[n_sources × n_frames × n_filters]` row-major (ReLU activation).
    pub fn separate(&self, features: &[f32], n_frames: usize) -> AudioResult<Vec<f32>> {
        let n_filt = self.cfg.n_filters;
        let bn_dim = self.cfg.bottleneck_dim;
        let n_src = self.cfg.n_sources;

        // Layer norm over feature dimension per frame
        let normed = Self::layer_norm(
            features,
            &self.weights.layer_norm_gamma,
            &self.weights.layer_norm_beta,
            n_frames,
            n_filt,
        );

        // Input projection: n_filters → bottleneck_dim via 1×1 conv
        let mut bottleneck =
            pointwise_1x1(&normed, n_frames, n_filt, &self.weights.in_proj_w, bn_dim);

        // TCN blocks: accumulate skip connections
        let mut skip_sum = vec![0.0f32; n_frames * bn_dim];
        let n_total_blocks = self.cfg.n_repeats * self.cfg.n_blocks;

        for block_idx in 0..n_total_blocks {
            let b_in_repeat = block_idx % self.cfg.n_blocks;
            let dilation = 1usize << b_in_repeat;
            let (output, skip) = Self::tcn_block(
                &bottleneck,
                n_frames,
                bn_dim,
                &self.weights.tcn_blocks[block_idx],
                self.cfg.hidden_dim,
                dilation,
                3, // depthwise kernel size
                self.cfg.causal,
            )?;
            for (s, v) in skip_sum.iter_mut().zip(skip.iter()) {
                *s += v;
            }
            bottleneck = output;
        }

        // Mask conv: bottleneck_dim → n_sources * n_filters
        let mask_out_ch = n_src * n_filt;
        let raw_masks = pointwise_1x1(
            &skip_sum,
            n_frames,
            bn_dim,
            &self.weights.mask_conv_w,
            mask_out_ch,
        );
        // Add bias and apply ReLU
        let mut masks = vec![0.0f32; n_src * n_frames * n_filt];
        for t in 0..n_frames {
            for s in 0..n_src {
                for f in 0..n_filt {
                    let raw_idx = t * mask_out_ch + s * n_filt + f;
                    let b = self.weights.mask_conv_b[s * n_filt + f];
                    // ReLU mask (as in TasNet; can be changed to sigmoid)
                    masks[s * n_frames * n_filt + t * n_filt + f] =
                        (raw_masks[raw_idx] + b).max(0.0);
                }
            }
        }
        Ok(masks)
    }

    /// Decode source features back to waveform.
    ///
    /// # Arguments
    /// - `source_features`: `[n_frames × n_filters]` row-major.
    /// - `n_frames`: number of frames.
    ///
    /// # Returns
    /// Waveform of length `n_frames * enc_stride + enc_kernel - enc_stride`.
    pub fn decode(&self, source_features: &[f32], n_frames: usize) -> AudioResult<Vec<f32>> {
        let enc_k = self.cfg.enc_kernel;
        let stride = self.cfg.enc_stride;
        let n_filt = self.cfg.n_filters;
        let n_samples = if n_frames == 0 {
            0
        } else {
            (n_frames - 1) * stride + enc_k
        };
        let mut waveform = vec![0.0f32; n_samples];
        // Transposed conv / overlap-add
        for f in 0..n_frames {
            let t_start = f * stride;
            for k in 0..n_filt {
                let feat_val = source_features[f * n_filt + k];
                for t in 0..enc_k {
                    if t_start + t < n_samples {
                        // decoder_w: [enc_k × n_filters] → decoder_w[t, k]
                        waveform[t_start + t] += feat_val * self.weights.decoder_w[t * n_filt + k];
                    }
                }
            }
        }
        Ok(waveform)
    }

    /// Full forward pass: waveform → separated sources.
    pub fn forward(&self, waveform: &[f32]) -> AudioResult<SeparationResult> {
        let (features, n_frames) = self.encode(waveform)?;

        // Compute masks for all sources
        let masks = self.separate(&features, n_frames)?;

        let n_src = self.cfg.n_sources;
        let n_filt = self.cfg.n_filters;
        let enc_k = self.cfg.enc_kernel;
        let stride = self.cfg.enc_stride;
        let n_samples_out = if n_frames == 0 {
            0
        } else {
            (n_frames - 1) * stride + enc_k
        };

        // Apply masks to encoded features, then decode per source
        let mut all_sources = vec![0.0f32; n_src * n_samples_out];

        for s in 0..n_src {
            // Mask features: [n_frames × n_filters]
            let mut masked = vec![0.0f32; n_frames * n_filt];
            for t in 0..n_frames {
                for f in 0..n_filt {
                    masked[t * n_filt + f] =
                        features[t * n_filt + f] * masks[s * n_frames * n_filt + t * n_filt + f];
                }
            }
            let source_waveform = self.decode(&masked, n_frames)?;
            let src_offset = s * n_samples_out;
            all_sources[src_offset..src_offset + source_waveform.len()]
                .copy_from_slice(&source_waveform);
        }

        Ok(SeparationResult {
            sources: all_sources,
            n_sources: n_src,
            n_samples: n_samples_out,
        })
    }

    /// Single TCN block: 1×1 → PReLU → depthwise dilated conv → PReLU → 1×1 (output + skip).
    ///
    /// # Returns
    /// `(output, skip_connection)` both of shape `[n_frames × bottleneck_dim]`.
    pub fn tcn_block(
        features: &[f32],
        n_frames: usize,
        bottleneck_dim: usize,
        block_weights: &TcnBlockWeights,
        hidden_dim: usize,
        dilation: usize,
        kernel: usize,
        causal: bool,
    ) -> AudioResult<(Vec<f32>, Vec<f32>)> {
        // 1×1 conv: bottleneck_dim → hidden_dim
        let h = pointwise_1x1(
            features,
            n_frames,
            bottleneck_dim,
            &block_weights.in_conv_w,
            hidden_dim,
        );
        // Add bias + PReLU
        let h: Vec<f32> = h
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                let ch = idx % hidden_dim;
                let biased = v + block_weights.in_conv_b[ch];
                Self::prelu_scalar(biased, 0.25)
            })
            .collect();

        // Depthwise dilated conv: hidden_dim channels, kernel size
        let h = depthwise_dilated_conv(
            &h,
            n_frames,
            hidden_dim,
            &block_weights.dw_conv_w,
            &block_weights.dw_conv_b,
            kernel,
            dilation,
            causal,
        );
        // PReLU
        let h: Vec<f32> = h.iter().map(|&v| Self::prelu_scalar(v, 0.25)).collect();

        // Skip connection: 1×1 conv hidden_dim → bottleneck_dim
        let skip_raw = pointwise_1x1(
            &h,
            n_frames,
            hidden_dim,
            &block_weights.skip_w,
            bottleneck_dim,
        );
        let skip: Vec<f32> = skip_raw
            .iter()
            .enumerate()
            .map(|(idx, &v)| {
                let ch = idx % bottleneck_dim;
                v + block_weights.skip_b[ch]
            })
            .collect();

        // Residual output: 1×1 conv hidden_dim → bottleneck_dim
        let res_raw = pointwise_1x1(
            &h,
            n_frames,
            hidden_dim,
            &block_weights.pw_conv_w,
            bottleneck_dim,
        );
        let residual: Vec<f32> = res_raw
            .iter()
            .enumerate()
            .zip(features.iter())
            .map(|((idx, &v), &x)| {
                let ch = idx % bottleneck_dim;
                v + block_weights.pw_conv_b[ch] + x
            })
            .collect();

        Ok((residual, skip))
    }

    /// Layer normalisation: `(x - mean) / (std + 1e-8) * gamma + beta`.
    ///
    /// Normalises over the feature dimension for each frame.
    /// `x`: `[n_frames × dim]` row-major; `gamma`, `beta`: `[dim]`.
    pub fn layer_norm(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        n_frames: usize,
        dim: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_frames * dim];
        for t in 0..n_frames {
            let base = t * dim;
            let row = &x[base..base + dim];
            let mean = row.iter().sum::<f32>() / dim as f32;
            let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let std = (var + 1e-8).sqrt();
            for (d, &v) in row.iter().enumerate() {
                out[base + d] = (v - mean) / std * gamma[d] + beta[d];
            }
        }
        out
    }

    /// PReLU: `max(alpha * x, x)`.
    pub fn prelu(x: &[f32], alpha: f32) -> Vec<f32> {
        x.iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect()
    }

    /// Sigmoid: `1 / (1 + exp(-x))`.
    pub fn sigmoid_act(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
    }

    /// Scale-Invariant Signal-to-Noise Ratio loss.
    ///
    /// Computes `-SI-SNR(estimate, target)` as the scalar loss to minimise.
    /// `SI-SNR = 10 * log10(||s_target||² / (||e_noise||² + 1e-8))`
    /// where `s_target = (ŝ·s / ||s||²) * s` and `e_noise = ŝ - s_target`.
    pub fn si_snr_loss(estimate: &[f32], target: &[f32]) -> AudioResult<f32> {
        if estimate.len() != target.len() {
            return Err(AudioError::DimensionMismatch {
                expected: target.len(),
                got: estimate.len(),
            });
        }
        if estimate.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "si_snr_loss requires non-empty signals".into(),
            });
        }

        let n = estimate.len() as f32;
        // Zero-mean
        let est_mean = estimate.iter().sum::<f32>() / n;
        let tgt_mean = target.iter().sum::<f32>() / n;
        let est_zm: Vec<f32> = estimate.iter().map(|&v| v - est_mean).collect();
        let tgt_zm: Vec<f32> = target.iter().map(|&v| v - tgt_mean).collect();

        let dot: f32 = est_zm.iter().zip(tgt_zm.iter()).map(|(&a, &b)| a * b).sum();
        let tgt_power: f32 = tgt_zm.iter().map(|&v| v * v).sum::<f32>() + 1e-8;
        let scale = dot / tgt_power;

        let s_target: Vec<f32> = tgt_zm.iter().map(|&v| scale * v).collect();
        let e_noise: Vec<f32> = est_zm
            .iter()
            .zip(s_target.iter())
            .map(|(&e, &s)| e - s)
            .collect();

        let s_power: f32 = s_target.iter().map(|&v| v * v).sum();
        let n_power: f32 = e_noise.iter().map(|&v| v * v).sum::<f32>() + 1e-8;

        // SI-SNR in dB, return negative (loss)
        let si_snr_db = 10.0 * (s_power / n_power).log10();
        Ok(-si_snr_db)
    }

    // ── Private scalar helpers ────────────────────────────────────────────────

    #[inline]
    fn prelu_scalar(v: f32, alpha: f32) -> f32 {
        if v >= 0.0 { v } else { alpha * v }
    }
}

// ─── Private conv helpers ─────────────────────────────────────────────────────

/// Pointwise (1×1) convolution: `[n_frames × in_ch]` → `[n_frames × out_ch]`.
fn pointwise_1x1(x: &[f32], n_frames: usize, in_ch: usize, w: &[f32], out_ch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_frames * out_ch];
    for t in 0..n_frames {
        for oc in 0..out_ch {
            let mut acc = 0.0f32;
            for ic in 0..in_ch {
                acc += x[t * in_ch + ic] * w[oc * in_ch + ic];
            }
            out[t * out_ch + oc] = acc;
        }
    }
    out
}

/// Depthwise dilated 1-D convolution.
///
/// Each output channel `c` is computed solely from input channel `c`
/// (depthwise separable). Supports both causal and non-causal padding.
///
/// - `x`: `[n_frames × channels]` row-major.
/// - `w`: `[channels × kernel]` row-major (one kernel per channel).
/// - `b`: `[channels]`.
/// - If `causal=true`: pad `(kernel-1)*dilation` zeros on the left only.
/// - If `causal=false`: pad `(kernel-1)*dilation/2` zeros on each side.
///
/// Returns `[n_frames × channels]` row-major.
fn depthwise_dilated_conv(
    x: &[f32],
    n_frames: usize,
    channels: usize,
    w: &[f32],
    b: &[f32],
    kernel: usize,
    dilation: usize,
    causal: bool,
) -> Vec<f32> {
    let left_pad = if causal {
        (kernel - 1) * dilation
    } else {
        (kernel - 1) * dilation / 2
    };

    let mut out = vec![0.0f32; n_frames * channels];
    for t in 0..n_frames {
        for c in 0..channels {
            let mut acc = b[c];
            for ki in 0..kernel {
                let src_padded = t + ki * dilation;
                if src_padded < left_pad {
                    continue;
                }
                let src_t = src_padded - left_pad;
                if src_t >= n_frames {
                    continue;
                }
                let w_idx = c * kernel + ki;
                acc += x[src_t * channels + c] * w[w_idx];
            }
            out[t * channels + c] = acc;
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg(causal: bool) -> ConvTasNetConfig {
        ConvTasNetConfig {
            n_sources: 2,
            enc_kernel: 4,
            enc_stride: 2,
            n_filters: 8,
            bottleneck_dim: 4,
            hidden_dim: 8,
            n_blocks: 3,
            n_repeats: 2,
            causal,
        }
    }

    #[test]
    fn encode_output_shape() {
        let cfg = tiny_cfg(false);
        let n_samples = 20usize;
        let expected_frames = (n_samples - cfg.enc_kernel) / cfg.enc_stride + 1;
        let mut rng = LcgRng::new(1);
        let model = ConvTasNet::new(cfg.clone(), &mut rng).expect("new ok");
        let wav = vec![0.1f32; n_samples];
        let (features, n_frames) = model.encode(&wav).expect("encode ok");
        assert_eq!(n_frames, expected_frames);
        assert_eq!(features.len(), n_frames * cfg.n_filters);
    }

    #[test]
    fn encode_relu_non_negative() {
        let cfg = tiny_cfg(false);
        let mut rng = LcgRng::new(2);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        let wav = vec![0.5f32; 24];
        let (features, _) = model.encode(&wav).expect("encode ok");
        assert!(
            features.iter().all(|&v| v >= 0.0),
            "encoder output should be non-negative (ReLU)"
        );
    }

    #[test]
    fn separate_output_shape() {
        let cfg = tiny_cfg(false);
        let n_samples = 20usize;
        let n_filters = cfg.n_filters;
        let n_sources = cfg.n_sources;
        let enc_stride = cfg.enc_stride;
        let enc_kernel = cfg.enc_kernel;
        let expected_frames = (n_samples - enc_kernel) / enc_stride + 1;
        let mut rng = LcgRng::new(3);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        let wav = vec![0.1f32; n_samples];
        let (features, n_frames) = model.encode(&wav).expect("encode ok");
        assert_eq!(n_frames, expected_frames);
        let masks = model.separate(&features, n_frames).expect("separate ok");
        assert_eq!(masks.len(), n_sources * n_frames * n_filters);
    }

    #[test]
    fn decode_output_length() {
        let cfg = tiny_cfg(false);
        let n_frames = 5usize;
        let enc_k = cfg.enc_kernel;
        let stride = cfg.enc_stride;
        let n_filt = cfg.n_filters;
        let mut rng = LcgRng::new(4);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        let feats = vec![0.1f32; n_frames * n_filt];
        let wav = model.decode(&feats, n_frames).expect("decode ok");
        let expected = (n_frames - 1) * stride + enc_k;
        assert_eq!(wav.len(), expected);
    }

    #[test]
    fn forward_n_sources_correct() {
        let cfg = tiny_cfg(false);
        let n_src = cfg.n_sources;
        let mut rng = LcgRng::new(5);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        let wav = vec![0.1f32; 32];
        let result = model.forward(&wav).expect("forward ok");
        assert_eq!(result.n_sources, n_src);
    }

    #[test]
    fn forward_finite() {
        let cfg = tiny_cfg(false);
        let mut rng = LcgRng::new(6);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        let wav: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
        let result = model.forward(&wav).expect("forward ok");
        assert!(
            result.sources.iter().all(|v| v.is_finite()),
            "non-finite output"
        );
    }

    #[test]
    fn tcn_block_output_shape() {
        let cfg = tiny_cfg(false);
        let n_frames = 8usize;
        let bn_dim = cfg.bottleneck_dim;
        let hid_dim = cfg.hidden_dim;
        let mut rng = LcgRng::new(7);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        let features = vec![0.1f32; n_frames * bn_dim];
        let (output, skip) = ConvTasNet::tcn_block(
            &features,
            n_frames,
            bn_dim,
            &model.weights.tcn_blocks[0],
            hid_dim,
            1,
            3,
            false,
        )
        .expect("tcn_block ok");
        assert_eq!(output.len(), n_frames * bn_dim);
        assert_eq!(skip.len(), n_frames * bn_dim);
    }

    #[test]
    fn layer_norm_mean_zero() {
        // With gamma=1, beta=0, the output should have mean ≈ 0 per frame
        let n_frames = 4usize;
        let dim = 8usize;
        let x: Vec<f32> = (0..n_frames * dim).map(|i| i as f32).collect();
        let gamma = vec![1.0f32; dim];
        let beta = vec![0.0f32; dim];
        let out = ConvTasNet::layer_norm(&x, &gamma, &beta, n_frames, dim);
        for t in 0..n_frames {
            let mean: f32 = out[t * dim..(t + 1) * dim].iter().sum::<f32>() / dim as f32;
            assert!(mean.abs() < 1e-5, "frame {t} mean = {mean}, expected ≈ 0");
        }
    }

    #[test]
    fn prelu_positive_unchanged() {
        let x = vec![0.0f32, 0.5, 1.0, 2.0, 100.0];
        let out = ConvTasNet::prelu(&x, 0.25);
        for (got, &expected) in out.iter().zip(x.iter()) {
            assert!(
                (got - expected).abs() < 1e-7,
                "prelu positive: {got} != {expected}"
            );
        }
    }

    #[test]
    fn sigmoid_bounded() {
        let x: Vec<f32> = (-20..=20).map(|i| i as f32).collect();
        let out = ConvTasNet::sigmoid_act(&x);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "sigmoid out of [0,1]: {v}");
        }
    }

    #[test]
    fn si_snr_perfect() {
        // When estimate == target, SI-SNR should be very high (loss ≈ -∞ → but finite due to eps)
        // We test that loss is negative (high SI-SNR)
        let signal: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let loss = ConvTasNet::si_snr_loss(&signal, &signal).expect("si_snr ok");
        // Perfect separation → very negative loss (large positive SI-SNR)
        assert!(
            loss < 0.0,
            "perfect SI-SNR loss should be negative, got {loss}"
        );
    }

    #[test]
    fn si_snr_finite() {
        let mut rng = LcgRng::new(99);
        let mut est = vec![0.0f32; 64];
        let mut tgt = vec![0.0f32; 64];
        rng.fill_normal(&mut est);
        rng.fill_normal(&mut tgt);
        let loss = ConvTasNet::si_snr_loss(&est, &tgt).expect("si_snr ok");
        assert!(loss.is_finite(), "SI-SNR loss non-finite: {loss}");
    }

    #[test]
    fn causal_vs_noncausal() {
        let n_samples = 20usize;
        let wav = vec![0.05f32; n_samples];

        let mut rng_c = LcgRng::new(10);
        let model_causal = ConvTasNet::new(tiny_cfg(true), &mut rng_c).expect("causal ok");
        let result_c = model_causal.forward(&wav).expect("causal forward ok");
        assert!(result_c.sources.iter().all(|v| v.is_finite()));

        let mut rng_nc = LcgRng::new(10);
        let model_nc = ConvTasNet::new(tiny_cfg(false), &mut rng_nc).expect("noncausal ok");
        let result_nc = model_nc.forward(&wav).expect("noncausal forward ok");
        assert!(result_nc.sources.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn err_n_sources_zero() {
        let mut cfg = tiny_cfg(false);
        cfg.n_sources = 0;
        assert!(matches!(cfg.validate(), Err(AudioError::EmptyInput { .. })));
    }

    #[test]
    fn err_enc_kernel_zero() {
        let mut cfg = tiny_cfg(false);
        cfg.enc_kernel = 0;
        assert!(matches!(
            cfg.validate(),
            Err(AudioError::InvalidKernelSize(0))
        ));
    }

    #[test]
    fn err_short_waveform() {
        let cfg = tiny_cfg(false);
        let enc_k = cfg.enc_kernel;
        let mut rng = LcgRng::new(11);
        let model = ConvTasNet::new(cfg, &mut rng).expect("new ok");
        // Waveform shorter than enc_kernel
        let short_wav = vec![0.1f32; enc_k - 1];
        let result = model.encode(&short_wav);
        assert!(
            matches!(result, Err(AudioError::InvalidSequenceLength(_))),
            "expected InvalidSequenceLength, got {:?}",
            result
        );
    }
}
