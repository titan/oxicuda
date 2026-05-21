//! HiFi-GAN vocoder generator (Kong et al. 2020, NeurIPS).
//!
//! Implements the HiFi-GAN generator architecture for high-fidelity neural
//! waveform synthesis from mel-spectrograms. The generator uses transposed
//! convolutions for upsampling combined with Multi-Receptive Field Fusion
//! (MRF) consisting of multiple residual dilated conv blocks.
//!
//! Reference: Kong et al., "HiFi-GAN: Generative Adversarial Networks for
//! Efficient and High Fidelity Speech Synthesis", NeurIPS 2020.
//!
//! Architecture overview:
//! ```text
//! mel [T, M] → input_conv (7×1) → LeakyReLU
//!           ↓
//!   for each upsample layer:
//!     TransposeConv1d (stride=r) → LeakyReLU
//!     MRF: Σ ResBlock(k, dilations) / n_resblocks
//!           ↓
//! LeakyReLU → output_conv (7×1) → Tanh → waveform [T*∏r]
//! ```

use crate::error::{AudioError, AudioResult};
use crate::handle::LcgRng;

// ─── Kaiming uniform init ─────────────────────────────────────────────────────

/// Fill `buf` with Kaiming-uniform values U(±√(6/fan_in)).
fn kaiming_uniform_init(rng: &mut LcgRng, fan_in: usize, buf: &mut [f32]) {
    let limit = (6.0_f32 / fan_in.max(1) as f32).sqrt();
    for v in buf.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
}

// ─── HifiGanConfig ────────────────────────────────────────────────────────────

/// Configuration for the HiFi-GAN generator.
#[derive(Debug, Clone)]
pub struct HifiGanConfig {
    /// Number of input mel-spectrogram channels (e.g., 80).
    pub mel_channels: usize,
    /// Upsampling rates per transposed-conv stage (e.g., [8, 8, 2, 2]).
    /// The product gives the total upsampling factor from mel frames to audio samples.
    pub upsample_rates: Vec<usize>,
    /// Number of channels after the initial input convolution (e.g., 512).
    pub upsample_initial_channels: usize,
    /// Kernel sizes for the MRF residual blocks (e.g., [3, 7, 11]).
    pub resblock_kernel_sizes: Vec<usize>,
    /// Dilation series for each residual block kernel (e.g., `[[1,3,5],[1,3,5],[1,3,5]]`).
    /// Outer length must match `resblock_kernel_sizes.len()`.
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
}

impl Default for HifiGanConfig {
    fn default() -> Self {
        Self {
            mel_channels: 80,
            upsample_rates: vec![8, 8, 2, 2],
            upsample_initial_channels: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
        }
    }
}

impl HifiGanConfig {
    /// Validate the configuration and return an error if it is invalid.
    pub fn validate(&self) -> AudioResult<()> {
        if self.mel_channels == 0 {
            return Err(AudioError::InvalidNumMels(0));
        }
        if self.upsample_rates.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "upsample_rates must not be empty".into(),
            });
        }
        if self.upsample_initial_channels == 0 {
            return Err(AudioError::InvalidEmbedDim(0));
        }
        if self.resblock_kernel_sizes.is_empty() {
            return Err(AudioError::EmptyInput {
                msg: "resblock_kernel_sizes must not be empty".into(),
            });
        }
        if self.resblock_kernel_sizes.len() != self.resblock_dilation_sizes.len() {
            return Err(AudioError::WeightShapeMismatch {
                msg: format!(
                    "resblock_kernel_sizes length {} != resblock_dilation_sizes length {}",
                    self.resblock_kernel_sizes.len(),
                    self.resblock_dilation_sizes.len()
                ),
            });
        }
        for &k in &self.resblock_kernel_sizes {
            if k == 0 {
                return Err(AudioError::InvalidKernelSize(0));
            }
        }
        Ok(())
    }
}

// ─── ResBlockWeights ─────────────────────────────────────────────────────────

/// Weights for a single MRF residual block.
///
/// Each residual block has `2 * num_dilations` convolutional layers arranged
/// as pairs: (dilated conv, stride-1 conv) repeated per dilation value.
#[derive(Debug, Clone)]
pub struct ResBlockWeights {
    /// Weight tensors for each conv layer: shape [out_ch × in_ch × k] flattened.
    /// Length = 2 * num_dilations (inner dim).
    pub conv_weights: Vec<Vec<f32>>,
    /// Bias vectors for each conv layer; length = 2 * num_dilations.
    pub conv_biases: Vec<Vec<f32>>,
}

// ─── HifiGanWeights ──────────────────────────────────────────────────────────

/// All learnable weights for the HiFi-GAN generator.
#[derive(Debug, Clone)]
pub struct HifiGanWeights {
    /// Input conv weight: [upsample_initial_channels × mel_channels × 7], flattened.
    pub input_conv_w: Vec<f32>,
    /// Input conv bias: `[upsample_initial_channels]`.
    pub input_conv_b: Vec<f32>,
    /// Upsample transposed-conv weights, one per layer.
    /// Layer i: [in_ch × out_ch × (2*rate)], where in_ch = initial >> i, out_ch = initial >> (i+1).
    pub upsample_weights: Vec<Vec<f32>>,
    /// Upsample transposed-conv biases, one per layer.
    pub upsample_biases: Vec<Vec<f32>>,
    /// MRF residual block weights: indexed `[layer][resblock]`.
    pub resblocks: Vec<Vec<ResBlockWeights>>,
    /// Output conv weight: [1 × channels_last × 7], flattened.
    pub output_conv_w: Vec<f32>,
    /// Output conv bias: `[1]`.
    pub output_conv_b: Vec<f32>,
}

// ─── HifiGanGenerator ────────────────────────────────────────────────────────

/// HiFi-GAN neural vocoder generator.
pub struct HifiGanGenerator {
    /// Configuration.
    pub cfg: HifiGanConfig,
    /// All weights.
    pub weights: HifiGanWeights,
}

impl HifiGanGenerator {
    /// Create a new generator with Kaiming-uniform initialised weights.
    pub fn new(cfg: HifiGanConfig, rng: &mut LcgRng) -> AudioResult<Self> {
        cfg.validate()?;

        let n_ups = cfg.upsample_rates.len();
        let mel_ch = cfg.mel_channels;
        let init_ch = cfg.upsample_initial_channels;
        let input_k = 7usize;

        // Input conv: init_ch × mel_ch × 7
        let input_size = init_ch * mel_ch * input_k;
        let mut input_conv_w = vec![0.0f32; input_size];
        kaiming_uniform_init(rng, mel_ch * input_k, &mut input_conv_w);
        let input_conv_b = vec![0.0f32; init_ch];

        // Upsample transposed conv weights
        let mut upsample_weights = Vec::with_capacity(n_ups);
        let mut upsample_biases = Vec::with_capacity(n_ups);
        for i in 0..n_ups {
            let in_ch = init_ch >> i;
            let out_ch = init_ch >> (i + 1);
            let k = 2 * cfg.upsample_rates[i];
            let w_size = in_ch * out_ch * k;
            let mut w = vec![0.0f32; w_size];
            kaiming_uniform_init(rng, in_ch * k, &mut w);
            upsample_weights.push(w);
            upsample_biases.push(vec![0.0f32; out_ch]);
        }

        // MRF residual block weights: [n_ups][n_resblocks]
        let n_resblocks = cfg.resblock_kernel_sizes.len();
        let mut resblocks = Vec::with_capacity(n_ups);
        for i in 0..n_ups {
            let channels = init_ch >> (i + 1);
            let mut layer_blocks = Vec::with_capacity(n_resblocks);
            for rb_idx in 0..n_resblocks {
                let k = cfg.resblock_kernel_sizes[rb_idx];
                let dilations = &cfg.resblock_dilation_sizes[rb_idx];
                let n_layers = 2 * dilations.len();
                let w_size = channels * channels * k;
                let mut conv_weights = Vec::with_capacity(n_layers);
                let mut conv_biases = Vec::with_capacity(n_layers);
                for _ in 0..n_layers {
                    let mut w = vec![0.0f32; w_size];
                    kaiming_uniform_init(rng, channels * k, &mut w);
                    conv_weights.push(w);
                    conv_biases.push(vec![0.0f32; channels]);
                }
                layer_blocks.push(ResBlockWeights {
                    conv_weights,
                    conv_biases,
                });
            }
            resblocks.push(layer_blocks);
        }

        // Output conv: 1 × channels_last × 7
        let channels_last = init_ch >> n_ups;
        let output_k = 7usize;
        let out_w_size = channels_last * output_k;
        let mut output_conv_w = vec![0.0f32; out_w_size];
        kaiming_uniform_init(rng, channels_last * output_k, &mut output_conv_w);
        let output_conv_b = vec![0.0f32; 1];

        Ok(Self {
            cfg,
            weights: HifiGanWeights {
                input_conv_w,
                input_conv_b,
                upsample_weights,
                upsample_biases,
                resblocks,
                output_conv_w,
                output_conv_b,
            },
        })
    }

    /// Forward pass: mel-spectrogram → audio waveform.
    ///
    /// # Arguments
    /// - `mel`: mel-spectrogram, row-major `[n_frames × mel_channels]`.
    /// - `n_frames`: number of mel frames.
    ///
    /// # Returns
    /// Audio waveform of length `n_frames * ∏(upsample_rates)`.
    pub fn forward(&self, mel: &[f32], n_frames: usize) -> AudioResult<Vec<f32>> {
        let mel_ch = self.cfg.mel_channels;
        if mel_ch == 0 {
            return Err(AudioError::InvalidNumMels(0));
        }
        let expected_len = n_frames * mel_ch;
        if mel.len() != expected_len {
            return Err(AudioError::ShapeMismatch {
                msg: format!(
                    "mel.len()={} != n_frames*mel_channels={}*{}={}",
                    mel.len(),
                    n_frames,
                    mel_ch,
                    expected_len
                ),
            });
        }

        let init_ch = self.cfg.upsample_initial_channels;
        let input_k = 7usize;

        // ── 1. Input conv: [n_frames × mel_ch] → [n_frames_valid × init_ch]
        // We apply 'valid' padding (no padding): output length = n_frames - k + 1
        // But to keep all frames, we use same-style: we just pass it through valid
        // and handle length. For HiFi-GAN, we use same-padding (reflect pad input).
        // Simplified: apply conv with half-padding on each side for same output length.
        let half_pad = (input_k - 1) / 2;
        let padded_mel = pad_reflect_1d(mel, n_frames, mel_ch, half_pad);
        let padded_frames = n_frames + 2 * half_pad;
        // conv1d valid: padded_frames - k + 1 = n_frames + 2*half_pad - (k-1) = n_frames
        let x = Self::conv1d(
            &padded_mel,
            padded_frames,
            mel_ch,
            &self.weights.input_conv_w,
            init_ch,
            input_k,
            &self.weights.input_conv_b,
        );
        let mut current_frames = n_frames;
        let mut x = Self::leaky_relu(&x, 0.1);

        // ── 2. Upsample stages
        let n_ups = self.cfg.upsample_rates.len();
        let n_resblocks = self.cfg.resblock_kernel_sizes.len();

        for i in 0..n_ups {
            let in_ch = init_ch >> i;
            let out_ch = init_ch >> (i + 1);
            let rate = self.cfg.upsample_rates[i];
            let up_k = 2 * rate;

            // TransposeConv1d
            let x_up = Self::conv_transpose1d(
                &x,
                current_frames,
                in_ch,
                &self.weights.upsample_weights[i],
                out_ch,
                up_k,
                rate,
                &self.weights.upsample_biases[i],
            );
            current_frames *= rate;
            let x_up = Self::leaky_relu(&x_up, 0.1);

            // MRF: apply each resblock and average
            let mut mrf_sum = vec![0.0f32; current_frames * out_ch];
            for rb_idx in 0..n_resblocks {
                let k = self.cfg.resblock_kernel_sizes[rb_idx];
                let dilations = &self.cfg.resblock_dilation_sizes[rb_idx];
                let rb_out = Self::resblock_forward(
                    &x_up,
                    current_frames,
                    out_ch,
                    &self.weights.resblocks[i][rb_idx],
                    k,
                    dilations,
                )?;
                for (s, v) in mrf_sum.iter_mut().zip(rb_out.iter()) {
                    *s += v;
                }
            }
            let scale = 1.0 / n_resblocks as f32;
            x = mrf_sum.into_iter().map(|v| v * scale).collect();
        }

        // ── 3. Output conv: LeakyReLU → 7×1 conv → Tanh
        let channels_last = init_ch >> n_ups;
        let output_k = 7usize;
        let x = Self::leaky_relu(&x, 0.1);
        let half_pad = (output_k - 1) / 2;
        let padded_x = pad_reflect_1d(&x, current_frames, channels_last, half_pad);
        let padded_out_frames = current_frames + 2 * half_pad;
        let out = Self::conv1d(
            &padded_x,
            padded_out_frames,
            channels_last,
            &self.weights.output_conv_w,
            1,
            output_k,
            &self.weights.output_conv_b,
        );
        let out = Self::tanh_act(&out);

        Ok(out)
    }

    /// 1-D convolution (valid, stride=1, no padding).
    ///
    /// - `x`: `[seq_len × in_ch]` row-major.
    /// - `w`: `[out_ch × in_ch × k]` row-major.
    /// - `b`: `[out_ch]`.
    ///
    /// Returns `[(seq_len - k + 1) × out_ch]` row-major.
    pub fn conv1d(
        x: &[f32],
        seq_len: usize,
        in_ch: usize,
        w: &[f32],
        out_ch: usize,
        k: usize,
        b: &[f32],
    ) -> Vec<f32> {
        if seq_len < k {
            return Vec::new();
        }
        let out_len = seq_len - k + 1;
        let mut out = vec![0.0f32; out_len * out_ch];
        for t in 0..out_len {
            for oc in 0..out_ch {
                let mut acc = b[oc];
                for ic in 0..in_ch {
                    for ki in 0..k {
                        let x_idx = (t + ki) * in_ch + ic;
                        let w_idx = oc * in_ch * k + ic * k + ki;
                        acc += x[x_idx] * w[w_idx];
                    }
                }
                out[t * out_ch + oc] = acc;
            }
        }
        out
    }

    /// Transposed 1-D convolution for upsampling (stride = `stride`).
    ///
    /// - `x`: `[seq_len × in_ch]` row-major.
    /// - `w`: `[in_ch × out_ch × k]` row-major (transposed-conv weight layout).
    /// - `b`: `[out_ch]`.
    ///
    /// Returns `[(seq_len * stride) × out_ch]` row-major.
    pub fn conv_transpose1d(
        x: &[f32],
        seq_len: usize,
        in_ch: usize,
        w: &[f32],
        out_ch: usize,
        k: usize,
        stride: usize,
        b: &[f32],
    ) -> Vec<f32> {
        let out_len = seq_len * stride;
        let mut out = vec![0.0f32; out_len * out_ch];
        // Initialise output with bias
        for t in 0..out_len {
            for oc in 0..out_ch {
                out[t * out_ch + oc] = b[oc];
            }
        }
        // Overlap-add: each input sample at time t broadcasts into output
        for t in 0..seq_len {
            for ic in 0..in_ch {
                let x_val = x[t * in_ch + ic];
                for ki in 0..k {
                    let out_t = t * stride + ki;
                    if out_t < out_len {
                        for oc in 0..out_ch {
                            let w_idx = ic * out_ch * k + oc * k + ki;
                            out[out_t * out_ch + oc] += x_val * w[w_idx];
                        }
                    }
                }
            }
        }
        out
    }

    /// MRF residual block with dilations.
    ///
    /// For each dilation `d` in `dilations`:
    /// 1. `h = LeakyReLU(x)`
    /// 2. `h = conv1d_dilated(h, kernel, dilation=d)`
    /// 3. `h = LeakyReLU(h)`
    /// 4. `h = conv1d_dilated(h, kernel, dilation=1)`
    /// 5. `x = x + h`
    ///
    /// The 2*num_dilations conv layers are indexed as pairs in `rb_weights`.
    pub fn resblock_forward(
        x: &[f32],
        seq_len: usize,
        channels: usize,
        rb_weights: &ResBlockWeights,
        kernel_size: usize,
        dilations: &[usize],
    ) -> AudioResult<Vec<f32>> {
        let n_layers = 2 * dilations.len();
        if rb_weights.conv_weights.len() < n_layers {
            return Err(AudioError::WeightShapeMismatch {
                msg: format!(
                    "resblock needs {} weight tensors, got {}",
                    n_layers,
                    rb_weights.conv_weights.len()
                ),
            });
        }
        let mut current = x.to_vec();
        for (d_idx, &dil) in dilations.iter().enumerate() {
            let h = Self::leaky_relu(&current, 0.1);
            let h = dilated_conv1d_same(
                &h,
                seq_len,
                channels,
                &rb_weights.conv_weights[2 * d_idx],
                kernel_size,
                dil,
                &rb_weights.conv_biases[2 * d_idx],
            );
            let h = Self::leaky_relu(&h, 0.1);
            let h = dilated_conv1d_same(
                &h,
                seq_len,
                channels,
                &rb_weights.conv_weights[2 * d_idx + 1],
                kernel_size,
                1,
                &rb_weights.conv_biases[2 * d_idx + 1],
            );
            // Residual addition
            for (c, v) in current.iter_mut().zip(h.iter()) {
                *c += v;
            }
        }
        Ok(current)
    }

    /// LeakyReLU: `max(alpha * x, x)`.
    pub fn leaky_relu(x: &[f32], alpha: f32) -> Vec<f32> {
        x.iter()
            .map(|&v| if v >= 0.0 { v } else { alpha * v })
            .collect()
    }

    /// Tanh activation.
    pub fn tanh_act(x: &[f32]) -> Vec<f32> {
        x.iter().map(|&v| v.tanh()).collect()
    }

    /// Approximate total parameter count.
    pub fn n_params(&self) -> usize {
        let mut n = self.weights.input_conv_w.len() + self.weights.input_conv_b.len();
        for (w, b) in self
            .weights
            .upsample_weights
            .iter()
            .zip(self.weights.upsample_biases.iter())
        {
            n += w.len() + b.len();
        }
        for layer_blocks in &self.weights.resblocks {
            for rb in layer_blocks {
                for (w, b) in rb.conv_weights.iter().zip(rb.conv_biases.iter()) {
                    n += w.len() + b.len();
                }
            }
        }
        n += self.weights.output_conv_w.len() + self.weights.output_conv_b.len();
        n
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Dilated 1-D convolution with symmetric zero-padding to preserve sequence length.
///
/// - `x`: `[seq_len × channels]` row-major.
/// - `w`: `[channels × channels × k]` row-major.
/// - `b`: `[channels]`.
///
/// Pads each side with `(k - 1) * dilation / 2` zeros so output length == seq_len.
fn dilated_conv1d_same(
    x: &[f32],
    seq_len: usize,
    channels: usize,
    w: &[f32],
    k: usize,
    dilation: usize,
    b: &[f32],
) -> Vec<f32> {
    let pad = (k - 1) * dilation / 2;
    let mut out = vec![0.0f32; seq_len * channels];
    for t in 0..seq_len {
        for oc in 0..channels {
            let mut acc = b[oc];
            for ki in 0..k {
                let src_padded = t + ki * dilation;
                if src_padded < pad {
                    continue;
                }
                let src_t = src_padded - pad;
                if src_t >= seq_len {
                    continue;
                }
                for ic in 0..channels {
                    let x_idx = src_t * channels + ic;
                    let w_idx = oc * channels * k + ic * k + ki;
                    acc += x[x_idx] * w[w_idx];
                }
            }
            out[t * channels + oc] = acc;
        }
    }
    out
}

/// Reflect-pad a `[seq_len × channels]` tensor by `pad` positions on each side.
///
/// Returns `[(seq_len + 2*pad) × channels]`.
fn pad_reflect_1d(x: &[f32], seq_len: usize, channels: usize, pad: usize) -> Vec<f32> {
    let padded_len = seq_len + 2 * pad;
    let mut out = vec![0.0f32; padded_len * channels];
    for t in 0..padded_len {
        let src_t = if t < pad {
            // Left: reflect around index 0
            pad - t
        } else if t >= seq_len + pad {
            // Right: reflect around last index
            2 * seq_len + pad - 2 - t
        } else {
            t - pad
        };
        // Clamp to valid range
        let src_t = src_t.min(seq_len - 1);
        let dst_base = t * channels;
        let src_base = src_t * channels;
        out[dst_base..dst_base + channels].copy_from_slice(&x[src_base..src_base + channels]);
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> HifiGanConfig {
        HifiGanConfig {
            mel_channels: 8,
            upsample_rates: vec![2, 2],
            upsample_initial_channels: 16,
            resblock_kernel_sizes: vec![3, 7],
            resblock_dilation_sizes: vec![vec![1, 3], vec![1, 3]],
        }
    }

    #[test]
    fn forward_output_length() {
        let cfg = small_cfg();
        let total_ups: usize = cfg.upsample_rates.iter().product();
        let n_frames = 4usize;
        let mut rng = LcgRng::new(1);
        let generator = HifiGanGenerator::new(cfg, &mut rng).expect("new ok");
        let mel = vec![0.1f32; n_frames * generator.cfg.mel_channels];
        let out = generator.forward(&mel, n_frames).expect("forward ok");
        assert_eq!(
            out.len(),
            n_frames * total_ups,
            "output length mismatch: got {} expected {}",
            out.len(),
            n_frames * total_ups
        );
    }

    #[test]
    fn forward_output_finite() {
        let cfg = small_cfg();
        let n_frames = 4usize;
        let mut rng = LcgRng::new(2);
        let generator = HifiGanGenerator::new(cfg, &mut rng).expect("new ok");
        let mel = vec![0.05f32; n_frames * generator.cfg.mel_channels];
        let out = generator.forward(&mel, n_frames).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite output detected"
        );
    }

    #[test]
    fn conv1d_output_shape() {
        let seq_len = 10usize;
        let in_ch = 3usize;
        let out_ch = 5usize;
        let k = 3usize;
        let x = vec![0.1f32; seq_len * in_ch];
        let w = vec![0.1f32; out_ch * in_ch * k];
        let b = vec![0.0f32; out_ch];
        let out = HifiGanGenerator::conv1d(&x, seq_len, in_ch, &w, out_ch, k, &b);
        assert_eq!(out.len(), (seq_len - k + 1) * out_ch);
    }

    #[test]
    fn conv1d_identity_kernel() {
        // Single channel, kernel=[1], output should equal input
        let seq_len = 5usize;
        let in_ch = 1usize;
        let out_ch = 1usize;
        let k = 1usize;
        let x: Vec<f32> = (1..=5).map(|v| v as f32).collect();
        let w = vec![1.0f32]; // identity
        let b = vec![0.0f32];
        let out = HifiGanGenerator::conv1d(&x, seq_len, in_ch, &w, out_ch, k, &b);
        assert_eq!(out.len(), seq_len);
        for (got, expected) in out.iter().zip(x.iter()) {
            assert!(
                (got - expected).abs() < 1e-6,
                "identity conv failed: {got} != {expected}"
            );
        }
    }

    #[test]
    fn conv_transpose1d_output_shape() {
        let seq_len = 5usize;
        let in_ch = 4usize;
        let out_ch = 2usize;
        let k = 4usize;
        let stride = 2usize;
        let x = vec![0.1f32; seq_len * in_ch];
        let w = vec![0.1f32; in_ch * out_ch * k];
        let b = vec![0.0f32; out_ch];
        let out = HifiGanGenerator::conv_transpose1d(&x, seq_len, in_ch, &w, out_ch, k, stride, &b);
        assert_eq!(out.len(), seq_len * stride * out_ch);
    }

    #[test]
    fn resblock_preserves_shape() {
        let seq_len = 8usize;
        let channels = 4usize;
        let k = 3usize;
        let dilations = vec![1usize, 2];
        let n_layers = 2 * dilations.len();
        let w_size = channels * channels * k;
        let conv_weights = vec![vec![0.01f32; w_size]; n_layers];
        let conv_biases = vec![vec![0.0f32; channels]; n_layers];
        let rb = ResBlockWeights {
            conv_weights,
            conv_biases,
        };
        let x = vec![0.1f32; seq_len * channels];
        let out = HifiGanGenerator::resblock_forward(&x, seq_len, channels, &rb, k, &dilations)
            .expect("resblock ok");
        assert_eq!(out.len(), seq_len * channels);
    }

    #[test]
    fn leaky_relu_positive() {
        let x = vec![0.0f32, 0.5, 1.0, 2.0];
        let out = HifiGanGenerator::leaky_relu(&x, 0.1);
        for (got, &expected) in out.iter().zip(x.iter()) {
            assert!(
                (got - expected).abs() < 1e-7,
                "positive pass-through failed"
            );
        }
    }

    #[test]
    fn leaky_relu_negative() {
        let alpha = 0.1f32;
        let x = vec![-1.0f32, -2.0, -0.5];
        let out = HifiGanGenerator::leaky_relu(&x, alpha);
        for (got, &v) in out.iter().zip(x.iter()) {
            let expected = alpha * v;
            assert!(
                (got - expected).abs() < 1e-7,
                "leaky relu negative: {got} != {expected}"
            );
        }
    }

    #[test]
    fn tanh_bounds() {
        let x: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.5).collect();
        let out = HifiGanGenerator::tanh_act(&x);
        for &v in &out {
            assert!((-1.0..=1.0).contains(&v), "tanh out of bounds: {v}");
        }
    }

    #[test]
    fn forward_small_config() {
        let cfg = HifiGanConfig {
            mel_channels: 8,
            upsample_rates: vec![2, 2],
            upsample_initial_channels: 16,
            resblock_kernel_sizes: vec![3],
            resblock_dilation_sizes: vec![vec![1, 3]],
        };
        let n_frames = 4usize;
        let mut rng = LcgRng::new(10);
        let generator = HifiGanGenerator::new(cfg, &mut rng).expect("new ok");
        let mel = vec![0.1f32; n_frames * generator.cfg.mel_channels];
        let out = generator.forward(&mel, n_frames).expect("forward ok");
        assert_eq!(out.len(), n_frames * 4);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn n_params_positive() {
        let cfg = small_cfg();
        let mut rng = LcgRng::new(42);
        let generator = HifiGanGenerator::new(cfg, &mut rng).expect("new ok");
        assert!(generator.n_params() > 0);
    }

    #[test]
    fn default_config_valid() {
        let cfg = HifiGanConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn err_mel_channels_zero() {
        let mut cfg = small_cfg();
        cfg.mel_channels = 0;
        assert!(matches!(cfg.validate(), Err(AudioError::InvalidNumMels(0))));
    }

    #[test]
    fn err_empty_upsample_rates() {
        let mut cfg = small_cfg();
        cfg.upsample_rates = vec![];
        assert!(matches!(cfg.validate(), Err(AudioError::EmptyInput { .. })));
    }

    #[test]
    fn err_wrong_mel_input_size() {
        let cfg = small_cfg();
        let n_frames = 4usize;
        let mut rng = LcgRng::new(7);
        let generator = HifiGanGenerator::new(cfg, &mut rng).expect("new ok");
        // Wrong mel length: should be n_frames * mel_channels, give n_frames * mel_channels - 1
        let bad_mel = vec![0.1f32; n_frames * generator.cfg.mel_channels - 1];
        let result = generator.forward(&bad_mel, n_frames);
        assert!(matches!(result, Err(AudioError::ShapeMismatch { .. })));
    }

    #[test]
    fn resblock_weights_count() {
        let dilations = [1usize, 3, 5];
        let n_expected = 2 * dilations.len();
        let k = 3usize;
        let channels = 4usize;
        let w_size = channels * channels * k;
        let rb = ResBlockWeights {
            conv_weights: vec![vec![0.0f32; w_size]; n_expected],
            conv_biases: vec![vec![0.0f32; channels]; n_expected],
        };
        assert_eq!(rb.conv_weights.len(), n_expected);
        assert_eq!(rb.conv_biases.len(), n_expected);
    }
}
