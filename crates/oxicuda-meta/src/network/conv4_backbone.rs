//! Conv-4 backbone — the canonical four-block convnet used as the few-shot
//! feature extractor in Vinyals et al. 2016 ("Matching Networks for One-Shot
//! Learning") and Snell et al. 2017 ("Prototypical Networks for Few-Shot
//! Learning"), and the de-facto reference encoder on Omniglot and
//! MiniImageNet ever since.
//!
//! Each of the four identical blocks is
//!
//! ```text
//! Conv 3×3 (same-pad, stride 1) → BatchNorm → ReLU → MaxPool 2×2 stride 2
//! ```
//!
//! All four blocks share the same hidden channel width `W` (typically 64).
//! The first block maps `in_channels → W`, the next three keep the channel
//! count at `W`.  The 2×2 max-pool halves the spatial extent at each block, so
//! after four blocks the feature map is `(W × H/16 × W'/16)`, which is then
//! flattened into a `output_dim`-vector for the downstream meta-classifier.
//!
//! This module owns:
//!
//! * four [`Conv4Block`]s, each carrying its 3×3 convolution weights (no
//!   bias — biases are typically omitted when followed by BN), per-channel
//!   BN affine `γ, β`, and the implicit ReLU + 2×2 max-pool;
//! * the [`Conv4Config`] used at construction time.
//!
//! All convolutions are zero-padded "same" 3×3 convolutions, computed via a
//! direct nested-loop implementation (no im2col, no FFT) — the goal is
//! correctness and portability for a Pure-Rust meta-learning library.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for [`Conv4Backbone`].
#[derive(Debug, Clone)]
pub struct Conv4Config {
    /// Number of input channels (e.g. 1 for Omniglot, 3 for MiniImageNet).
    pub in_channels: usize,
    /// Hidden channel width shared by all four conv blocks (typically 64).
    pub width: usize,
    /// Input image height — must be divisible by 16 (four 2×2 max-pools).
    pub input_h: usize,
    /// Input image width — must be divisible by 16 (four 2×2 max-pools).
    pub input_w: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Building blocks
// ─────────────────────────────────────────────────────────────────────────────

/// One of the four blocks of the Conv-4 backbone.
///
/// Holds the 3×3 same-pad convolution weights (`width × prev_channels × 3 × 3`
/// row-major) and the per-channel BN affine parameters γ (`bn_scale`),
/// β (`bn_shift`).  The implicit ReLU and 2×2 stride-2 max-pool are applied
/// inside `Conv4Block::forward`.
#[derive(Debug, Clone)]
pub struct Conv4Block {
    /// Conv 3×3 same-pad weights: layout `[out_c, in_c, kh, kw]` row-major,
    /// total length `width · prev_channels · 9`.
    conv_w: Vec<f32>,
    /// Per-channel BN scale γ (length `width`).
    bn_scale: Vec<f32>,
    /// Per-channel BN shift β (length `width`).
    bn_shift: Vec<f32>,
    /// Input channels into this block (= `width` for blocks 1..=3, =
    /// `in_channels` for block 0).
    in_channels: usize,
    /// Output channels (= `width`).
    out_channels: usize,
    /// Numerical stabiliser for the BN denominator.
    bn_eps: f32,
}

impl Conv4Block {
    /// Construct a single block with Xavier-uniform conv weights and identity
    /// BN affine (γ = 1, β = 0).
    fn new(in_channels: usize, out_channels: usize, rng: &mut LcgRng) -> Self {
        // Fan-in / fan-out for a 3×3 conv layer follow the standard
        // PyTorch convention: fan_in = in_c · 9, fan_out = out_c · 9.
        let fan_in = in_channels * 9;
        let fan_out = out_channels * 9;
        let limit = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
        let n_weights = out_channels * in_channels * 9;
        let mut conv_w = vec![0.0_f32; n_weights];
        for v in conv_w.iter_mut() {
            *v = (rng.next_f32() * 2.0 - 1.0) * limit;
        }
        Self {
            conv_w,
            bn_scale: vec![1.0_f32; out_channels],
            bn_shift: vec![0.0_f32; out_channels],
            in_channels,
            out_channels,
            bn_eps: 1e-5,
        }
    }

    /// Number of trainable parameters in this block — conv weights plus the
    /// BN γ and β.
    fn n_params(&self) -> usize {
        self.conv_w.len() + self.bn_scale.len() + self.bn_shift.len()
    }

    /// Apply this block to a `(in_channels × h × w)` row-major tensor.
    /// Returns a `(out_channels × h/2 × w/2)` row-major tensor.
    fn forward(&self, x: &[f32], h: usize, w: usize) -> MetaResult<Vec<f32>> {
        let expected = self.in_channels * h * w;
        if x.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        if h < 2 || w < 2 {
            return Err(MetaError::BackboneError {
                msg: format!("Conv4Block input too small to pool: {h}x{w}"),
            });
        }
        let conv = self.same_pad_conv3x3(x, h, w);
        let bn = self.batch_norm_relu(&conv, h, w);
        let pooled = self.max_pool_2x2(&bn, h, w);
        Ok(pooled)
    }

    /// 3×3 same-pad convolution with zero padding.  Direct nested-loop
    /// implementation — `out[oc, y, x] = Σ_ic Σ_ky Σ_kx W[oc, ic, ky, kx] ·
    /// in[ic, y + ky − 1, x + kx − 1]`, with zero-padding outside the image.
    fn same_pad_conv3x3(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let oc = self.out_channels;
        let ic = self.in_channels;
        let mut out = vec![0.0_f32; oc * h * w];
        for oc_idx in 0..oc {
            let w_oc_offset = oc_idx * ic * 9;
            for y in 0..h {
                for x_pos in 0..w {
                    let mut acc = 0.0_f32;
                    for ic_idx in 0..ic {
                        let w_ic_offset = w_oc_offset + ic_idx * 9;
                        let x_ic_offset = ic_idx * h * w;
                        for ky in 0..3_usize {
                            let iy = y as isize + ky as isize - 1;
                            if iy < 0 || iy >= h as isize {
                                continue;
                            }
                            let iy_u = iy as usize;
                            let row_offset = x_ic_offset + iy_u * w;
                            let w_row_offset = w_ic_offset + ky * 3;
                            for kx in 0..3_usize {
                                let ix = x_pos as isize + kx as isize - 1;
                                if ix < 0 || ix >= w as isize {
                                    continue;
                                }
                                let ix_u = ix as usize;
                                acc += self.conv_w[w_row_offset + kx] * x[row_offset + ix_u];
                            }
                        }
                    }
                    out[oc_idx * h * w + y * w + x_pos] = acc;
                }
            }
        }
        out
    }

    /// Per-channel batch normalisation over the spatial extent (the
    /// normalisation is over `h · w` positions per channel) followed by a
    /// clip-to-zero ReLU.  No running statistics are stored — the moments
    /// are computed from the current activations, matching the "fresh"
    /// behaviour of meta-learning batch norm.
    fn batch_norm_relu(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let oc = self.out_channels;
        let chan_len = h * w;
        let mut out = vec![0.0_f32; oc * chan_len];
        let inv_n = 1.0_f32 / chan_len as f32;
        for c in 0..oc {
            let chan = &x[c * chan_len..(c + 1) * chan_len];
            let mut mean = 0.0_f32;
            for &v in chan.iter() {
                mean += v;
            }
            mean *= inv_n;
            let mut var = 0.0_f32;
            for &v in chan.iter() {
                let d = v - mean;
                var += d * d;
            }
            var *= inv_n;
            let denom = (var + self.bn_eps).sqrt();
            let gamma = self.bn_scale[c];
            let beta = self.bn_shift[c];
            let out_chan = &mut out[c * chan_len..(c + 1) * chan_len];
            for (o, &v) in out_chan.iter_mut().zip(chan.iter()) {
                let normed = (v - mean) / denom * gamma + beta;
                // Implicit ReLU after BN.
                *o = if normed > 0.0 { normed } else { 0.0 };
            }
        }
        out
    }

    /// 2×2 max-pool with stride 2 over a `(out_channels × h × w)` tensor.
    /// Output shape is `(out_channels × h/2 × w/2)`.
    fn max_pool_2x2(&self, x: &[f32], h: usize, w: usize) -> Vec<f32> {
        let oc = self.out_channels;
        let h_out = h / 2;
        let w_out = w / 2;
        let mut out = vec![0.0_f32; oc * h_out * w_out];
        for c in 0..oc {
            let chan = &x[c * h * w..(c + 1) * h * w];
            for y in 0..h_out {
                for x_pos in 0..w_out {
                    let iy = y * 2;
                    let ix = x_pos * 2;
                    let v00 = chan[iy * w + ix];
                    let v01 = chan[iy * w + ix + 1];
                    let v10 = chan[(iy + 1) * w + ix];
                    let v11 = chan[(iy + 1) * w + ix + 1];
                    let m01 = if v00 > v01 { v00 } else { v01 };
                    let m23 = if v10 > v11 { v10 } else { v11 };
                    let m = if m01 > m23 { m01 } else { m23 };
                    out[c * h_out * w_out + y * w_out + x_pos] = m;
                }
            }
        }
        out
    }

    /// Number of input channels into this block.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Number of output channels from this block.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Append this block's trainable parameters — conv weights, then BN γ, then
    /// BN β — to `out`, in the canonical order used by [`Conv4Backbone::to_params`].
    fn append_params(&self, out: &mut Vec<f32>) {
        out.extend_from_slice(&self.conv_w);
        out.extend_from_slice(&self.bn_scale);
        out.extend_from_slice(&self.bn_shift);
    }

    /// Overwrite this block's trainable parameters from `params[offset..]` in the
    /// same conv-weights / BN-γ / BN-β order produced by [`Self::append_params`],
    /// returning the new offset just past this block's slice.
    fn load_params(&mut self, params: &[f32], offset: usize) -> usize {
        let mut o = offset;
        let cw = self.conv_w.len();
        self.conv_w.copy_from_slice(&params[o..o + cw]);
        o += cw;
        let s = self.bn_scale.len();
        self.bn_scale.copy_from_slice(&params[o..o + s]);
        o += s;
        let b = self.bn_shift.len();
        self.bn_shift.copy_from_slice(&params[o..o + b]);
        o + b
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backbone
// ─────────────────────────────────────────────────────────────────────────────

/// Four-block conv-BN-ReLU-MaxPool backbone.
///
/// `forward(x)` consumes a row-major `(in_channels × input_h × input_w)`
/// tensor and returns a flattened feature vector of length
/// `width · (input_h/16) · (input_w/16)`.
#[derive(Debug, Clone)]
pub struct Conv4Backbone {
    blocks: [Conv4Block; 4],
    cfg: Conv4Config,
}

impl Conv4Backbone {
    /// Construct a Conv-4 backbone with Xavier-initialised conv weights and
    /// identity BN affine.
    ///
    /// Validates that `width > 0`, `in_channels > 0`, and that both
    /// `input_h` and `input_w` are divisible by 16 — otherwise the four 2×2
    /// max-pools would not align.
    pub fn new(cfg: Conv4Config, rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.in_channels == 0 {
            return Err(MetaError::BackboneError {
                msg: "in_channels must be > 0".into(),
            });
        }
        if cfg.width == 0 {
            return Err(MetaError::BackboneError {
                msg: "width must be > 0".into(),
            });
        }
        if cfg.input_h == 0 || !cfg.input_h.is_multiple_of(16) {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!(
                    "input_h ({}) must be a positive multiple of 16",
                    cfg.input_h
                ),
            });
        }
        if cfg.input_w == 0 || !cfg.input_w.is_multiple_of(16) {
            return Err(MetaError::InvalidEpisodeConfig {
                msg: format!(
                    "input_w ({}) must be a positive multiple of 16",
                    cfg.input_w
                ),
            });
        }

        let blocks = [
            Conv4Block::new(cfg.in_channels, cfg.width, rng),
            Conv4Block::new(cfg.width, cfg.width, rng),
            Conv4Block::new(cfg.width, cfg.width, rng),
            Conv4Block::new(cfg.width, cfg.width, rng),
        ];
        Ok(Self { blocks, cfg })
    }

    /// Length of the flattened feature vector returned by [`Self::forward`]:
    /// `width · (input_h/16) · (input_w/16)`.
    pub fn output_dim(&self) -> usize {
        self.cfg.width * (self.cfg.input_h / 16) * (self.cfg.input_w / 16)
    }

    /// Total number of trainable parameters across the four blocks.
    pub fn n_params(&self) -> usize {
        self.blocks.iter().map(|b| b.n_params()).sum()
    }

    /// Read-only access to a specific block (for instrumentation / tests).
    pub fn block(&self, idx: usize) -> MetaResult<&Conv4Block> {
        self.blocks.get(idx).ok_or(MetaError::Internal {
            msg: format!("Conv4Backbone has 4 blocks, asked for index {idx}"),
        })
    }

    /// Forward a single example through the four blocks and flatten the
    /// result.
    ///
    /// `x`: row-major `(in_channels × input_h × input_w)`.
    /// Returns a flattened `output_dim()`-vector.
    pub fn forward(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        let expected = self.cfg.in_channels * self.cfg.input_h * self.cfg.input_w;
        if x.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let mut h = self.cfg.input_h;
        let mut w = self.cfg.input_w;
        let mut cur = x.to_vec();
        for block in self.blocks.iter() {
            cur = block.forward(&cur, h, w)?;
            h /= 2;
            w /= 2;
        }
        // After four blocks `cur` has shape (width × input_h/16 × input_w/16),
        // already flat in row-major layout — no extra copy needed.
        Ok(cur)
    }

    /// Run only the first `n_blocks` blocks and expose the (unflattened) feature
    /// map.  Used by tests that want to inspect intermediate activations
    /// (e.g. ReLU non-negativity).  Returns the `(channels, h, w, data)`
    /// tuple where `data` is `channels · h · w` row-major.
    pub fn forward_partial(
        &self,
        x: &[f32],
        n_blocks: usize,
    ) -> MetaResult<(usize, usize, usize, Vec<f32>)> {
        if n_blocks > self.blocks.len() {
            return Err(MetaError::Internal {
                msg: format!("Conv4Backbone has 4 blocks, asked for first {n_blocks}",),
            });
        }
        let expected = self.cfg.in_channels * self.cfg.input_h * self.cfg.input_w;
        if x.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: x.len(),
            });
        }
        let mut h = self.cfg.input_h;
        let mut w = self.cfg.input_w;
        let mut cur = x.to_vec();
        for block in self.blocks.iter().take(n_blocks) {
            cur = block.forward(&cur, h, w)?;
            h /= 2;
            w /= 2;
        }
        let channels = if n_blocks == 0 {
            self.cfg.in_channels
        } else {
            self.cfg.width
        };
        Ok((channels, h, w, cur))
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &Conv4Config {
        &self.cfg
    }

    /// Flatten every trainable parameter of the backbone into a single
    /// `n_params()`-length `Vec<f32>`, block by block, each block contributing
    /// its conv weights, then BN γ, then BN β.
    ///
    /// This is the flatten half of the MAML inner-loop closure contract — it lets
    /// the meta-learner treat the whole convnet as a flat parameter vector,
    /// exactly like [`crate::network::backbone::MlpBackbone::to_params`].
    pub fn to_params(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_params());
        for block in self.blocks.iter() {
            block.append_params(&mut out);
        }
        out
    }

    /// Overwrite every trainable parameter of the backbone from a flat
    /// `n_params()`-length slice, in the same order produced by
    /// [`Self::to_params`].
    ///
    /// This is the unflatten half of the MAML inner-loop closure contract.
    ///
    /// # Errors
    /// [`MetaError::DimensionMismatch`] if `params.len() != n_params()`.
    pub fn from_params(&mut self, params: &[f32]) -> MetaResult<()> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(MetaError::DimensionMismatch {
                expected,
                got: params.len(),
            });
        }
        let mut offset = 0;
        for block in self.blocks.iter_mut() {
            offset = block.load_params(params, offset);
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Conv4Config {
        Conv4Config {
            in_channels: 1,
            width: 4,
            input_h: 16,
            input_w: 16,
        }
    }

    fn make_backbone(cfg: Conv4Config) -> Conv4Backbone {
        let mut rng = LcgRng::new(2026);
        Conv4Backbone::new(cfg, &mut rng).expect("valid Conv4 cfg")
    }

    // ── construction validation ─────────────────────────────────────────────

    #[test]
    fn new_valid_cfg_succeeds() {
        let mut rng = LcgRng::new(1);
        assert!(Conv4Backbone::new(tiny_cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_input_h_not_divisible_by_16_errs() {
        let mut cfg = tiny_cfg();
        cfg.input_h = 24;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Conv4Backbone::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_input_w_not_divisible_by_16_errs() {
        let mut cfg = tiny_cfg();
        cfg.input_w = 17;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Conv4Backbone::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_width_zero_errs() {
        let mut cfg = tiny_cfg();
        cfg.width = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Conv4Backbone::new(cfg, &mut rng),
            Err(MetaError::BackboneError { .. })
        ));
    }

    #[test]
    fn new_in_channels_zero_errs() {
        let mut cfg = tiny_cfg();
        cfg.in_channels = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Conv4Backbone::new(cfg, &mut rng),
            Err(MetaError::BackboneError { .. })
        ));
    }

    // ── output_dim / n_params ───────────────────────────────────────────────

    #[test]
    fn output_dim_formula_holds() {
        // 1×16×16 with width 4 → 4 · (16/16) · (16/16) = 4
        let bb = make_backbone(tiny_cfg());
        // 4 channels × (16/16) × (16/16) = 4
        assert_eq!(bb.output_dim(), 4);
    }

    #[test]
    fn output_dim_pool_halving() {
        // A 1×32×32 input with width 8 must yield 8 · 2 · 2 = 32 after four
        // 2× max-pools — this confirms the four halvings happen in sequence.
        let cfg = Conv4Config {
            in_channels: 1,
            width: 8,
            input_h: 32,
            input_w: 32,
        };
        let bb = make_backbone(cfg);
        assert_eq!(bb.output_dim(), 8 * 2 * 2);
    }

    #[test]
    fn n_params_formula() {
        // Tiny config: in=1, width=4.  Block 0: 4·1·9 conv + 4 γ + 4 β = 44.
        // Blocks 1..3: 4·4·9 conv + 4 γ + 4 β = 152 each → 3 · 152 = 456.
        // Total = 44 + 456 = 500.
        let bb = make_backbone(tiny_cfg());
        // Block 0: 4·1·9 conv weights + 4 γ + 4 β.
        let block0 = (4_usize * 9) + 4 + 4;
        // Blocks 1..=3 each: 4·4·9 conv weights + 4 γ + 4 β, sum across 3.
        let block1plus = 3 * ((4_usize * 4 * 9) + 4 + 4);
        assert_eq!(bb.n_params(), block0 + block1plus);
    }

    // ── forward correctness ─────────────────────────────────────────────────

    #[test]
    fn forward_output_length_matches_output_dim() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let mut rng = LcgRng::new(11);
        let x: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let y = bb.forward(&x).expect("ok");
        assert_eq!(y.len(), bb.output_dim());
    }

    #[test]
    fn forward_deterministic_with_same_seed() {
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let bb_a = Conv4Backbone::new(tiny_cfg(), &mut rng_a).expect("ok");
        let bb_b = Conv4Backbone::new(tiny_cfg(), &mut rng_b).expect("ok");
        let cfg = bb_a.config().clone();
        let mut rng = LcgRng::new(123);
        let x: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32())
            .collect();
        let y_a = bb_a.forward(&x).expect("ok");
        let y_b = bb_b.forward(&x).expect("ok");
        assert_eq!(y_a, y_b);
    }

    #[test]
    fn forward_single_channel_works() {
        let cfg = Conv4Config {
            in_channels: 1,
            width: 4,
            input_h: 16,
            input_w: 16,
        };
        let bb = make_backbone(cfg);
        let x = vec![0.1_f32; bb.config().in_channels * bb.config().input_h * bb.config().input_w];
        let y = bb.forward(&x).expect("ok");
        assert_eq!(y.len(), bb.output_dim());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_multi_channel_works() {
        let cfg = Conv4Config {
            in_channels: 3,
            width: 8,
            input_h: 16,
            input_w: 16,
        };
        let bb = make_backbone(cfg);
        let cfg_ref = bb.config().clone();
        let mut rng = LcgRng::new(77);
        let x: Vec<f32> = (0..cfg_ref.in_channels * cfg_ref.input_h * cfg_ref.input_w)
            .map(|_| rng.next_f32())
            .collect();
        let y = bb.forward(&x).expect("ok");
        assert_eq!(y.len(), bb.output_dim());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_wrong_length_errs() {
        let bb = make_backbone(tiny_cfg());
        let x = vec![0.0_f32; 10];
        assert!(matches!(
            bb.forward(&x),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn forward_changing_input_changes_output() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let mut rng = LcgRng::new(13);
        let x_a: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let x_b: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let y_a = bb.forward(&x_a).expect("ok");
        let y_b = bb.forward(&x_b).expect("ok");
        assert_ne!(y_a, y_b, "different inputs should give different outputs");
    }

    #[test]
    fn forward_constant_input_gives_constant_interior_per_channel() {
        // A spatially-constant input produces a constant *interior* convolution
        // result per output channel — every interior position sees the full 3×3
        // window of the same constant.  Border positions see fewer non-zero
        // contributions because of zero padding, so they may differ.  BN +
        // ReLU + 2×2 max-pool preserve this property within each pooled
        // interior 2×2 block: pooling four equal interior values still gives
        // a constant per-channel result over the interior pooled cells.
        let cfg = Conv4Config {
            in_channels: 2,
            width: 4,
            input_h: 16,
            input_w: 16,
        };
        let bb = make_backbone(cfg);
        let cfg = bb.config().clone();
        let x = vec![0.5_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        let (channels, h, w, fmap) = bb.forward_partial(&x, 1).expect("ok");
        let chan_len = h * w;
        // Interior pooled cells: indices (y, x) with 1 ≤ y < h-1 and
        // 1 ≤ x < w-1.  These pooled cells consist entirely of pre-pool
        // positions whose 3×3 conv window lay strictly inside the image.
        for c in 0..channels {
            let chan = &fmap[c * chan_len..(c + 1) * chan_len];
            let pivot = chan[(h / 2) * w + (w / 2)];
            for y in 1..(h - 1) {
                for x_pos in 1..(w - 1) {
                    let v = chan[y * w + x_pos];
                    assert!(
                        (v - pivot).abs() < 1e-5,
                        "channel {c} interior at ({y},{x_pos}) not constant: {v} vs {pivot}"
                    );
                }
            }
        }
    }

    #[test]
    fn forward_partial_relu_non_negative() {
        // After the first block the output is post-BN-ReLU, hence ≥ 0.
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let mut rng = LcgRng::new(31);
        let x: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let (_c, _h, _w, fmap) = bb.forward_partial(&x, 1).expect("ok");
        // Note: forward_partial returns the *post-pool* feature map, which is
        // still post-ReLU (max-pool of non-negative values is non-negative).
        for &v in fmap.iter() {
            assert!(v >= 0.0, "ReLU activation must be ≥ 0, got {v}");
        }
    }

    #[test]
    fn forward_partial_dim_halving() {
        // After k blocks the spatial extent has halved k times.
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let x = vec![0.3_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        for k in 1..=4_usize {
            let (channels, h, w, fmap) = bb.forward_partial(&x, k).expect("ok");
            assert_eq!(channels, cfg.width);
            assert_eq!(h, cfg.input_h >> k);
            assert_eq!(w, cfg.input_w >> k);
            assert_eq!(fmap.len(), channels * h * w);
        }
    }

    #[test]
    fn forward_partial_zero_blocks_is_input() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let x = vec![0.25_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        let (channels, h, w, fmap) = bb.forward_partial(&x, 0).expect("ok");
        assert_eq!(channels, cfg.in_channels);
        assert_eq!(h, cfg.input_h);
        assert_eq!(w, cfg.input_w);
        assert_eq!(fmap, x);
    }

    // ── tiny + standard configs ─────────────────────────────────────────────

    #[test]
    fn tiny_config_smoke() {
        let bb = make_backbone(tiny_cfg());
        assert_eq!(bb.output_dim(), 4);
        let x = vec![0.05_f32; 16 * 16];
        let y = bb.forward(&x).expect("ok");
        assert_eq!(y.len(), 4);
        assert!(y.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn standard_width_64_smoke() {
        // The canonical width-64 backbone on 1×16×16 — keeps the test fast
        // but exercises the standard channel count.
        let cfg = Conv4Config {
            in_channels: 1,
            width: 64,
            input_h: 16,
            input_w: 16,
        };
        let bb = make_backbone(cfg);
        let cfg = bb.config().clone();
        let x = vec![0.0_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        let y = bb.forward(&x).expect("ok");
        assert_eq!(y.len(), bb.output_dim());
        assert_eq!(bb.output_dim(), 64); // 64 · 1 · 1
    }

    #[test]
    fn block_accessor_in_out_channels() {
        let bb = make_backbone(tiny_cfg());
        let b0 = bb.block(0).expect("ok");
        assert_eq!(b0.in_channels(), 1);
        assert_eq!(b0.out_channels(), 4);
        let b3 = bb.block(3).expect("ok");
        assert_eq!(b3.in_channels(), 4);
        assert_eq!(b3.out_channels(), 4);
        assert!(matches!(bb.block(4), Err(MetaError::Internal { .. })));
    }

    #[test]
    fn forward_partial_too_many_blocks_errs() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let x = vec![0.0_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        assert!(matches!(
            bb.forward_partial(&x, 5),
            Err(MetaError::Internal { .. })
        ));
    }

    #[test]
    fn forward_partial_wrong_length_errs() {
        let bb = make_backbone(tiny_cfg());
        let x = vec![0.0_f32; 7];
        assert!(matches!(
            bb.forward_partial(&x, 1),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }

    // ── to_params / from_params (MAML flatten/unflatten contract) ────────────

    #[test]
    fn to_params_length_matches_n_params() {
        let bb = make_backbone(tiny_cfg());
        assert_eq!(bb.to_params().len(), bb.n_params());
    }

    #[test]
    fn to_params_from_params_round_trips_exactly() {
        // Flatten, perturb, unflatten, re-flatten: must recover the perturbed
        // vector bit-for-bit (no reordering, no loss).
        let mut bb = make_backbone(tiny_cfg());
        let original = bb.to_params();
        // A deterministic perturbation of every parameter.
        let perturbed: Vec<f32> = original
            .iter()
            .enumerate()
            .map(|(i, &v)| v + (i as f32) * 0.001 - 0.5)
            .collect();
        bb.from_params(&perturbed).expect("from_params ok");
        let reread = bb.to_params();
        assert_eq!(reread, perturbed, "to/from_params must round-trip exactly");
    }

    #[test]
    fn from_params_changes_forward_output() {
        // Loading different parameters must change the forward output — proving
        // the flat vector genuinely drives the convolution weights.
        let mut bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let x = vec![0.3_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        let y_before = bb.forward(&x).expect("forward ok");
        let mut params = bb.to_params();
        for p in params.iter_mut() {
            *p += 0.25;
        }
        bb.from_params(&params).expect("from_params ok");
        let y_after = bb.forward(&x).expect("forward ok");
        assert_ne!(
            y_before, y_after,
            "changing the flat params must change the forward output"
        );
    }

    #[test]
    fn from_params_wrong_length_errs() {
        let mut bb = make_backbone(tiny_cfg());
        assert!(matches!(
            bb.from_params(&[0.0_f32; 3]),
            Err(MetaError::DimensionMismatch { .. })
        ));
    }
}
