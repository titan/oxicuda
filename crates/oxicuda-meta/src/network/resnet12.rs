//! ResNet-12 backbone — the canonical few-shot feature extractor used by the
//! bulk of the modern meta-learning literature (Mishra et al. 2018 "SNAIL",
//! Oreshkin et al. 2018 "TADAM", Lee et al. 2019 "MetaOptNet", Tian et al. 2020
//! "RFS / rethinking few-shot") on MiniImageNet, TieredImageNet, CIFAR-FS and
//! FC100.
//!
//! ResNet-12 is a four-stage residual network.  Each stage is a single
//! *residual block* of the form
//!
//! ```text
//!            ┌────────────────────────────────────────────────┐
//!   x ──┬──▶ Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 → BN ──▶ (+) ──▶ ReLU ──▶ MaxPool 2×2 ──▶ out
//!       │                                                              ▲
//!       └──────────────── Conv1×1 → BN (channel-matching shortcut) ────┘
//! ```
//!
//! i.e. three stacked 3×3 conv-BN(-ReLU) layers with a projection (1×1
//! conv + BN) shortcut so the residual addition is dimension-consistent even
//! when the channel count changes between the block input and output.  After
//! the residual add a final ReLU is applied, followed by a 2×2 stride-2
//! max-pool that halves the spatial extent.  Stacking four such stages halves
//! the spatial extent four times, so a `H × W` input becomes
//! `(width₃ × H/16 × W/16)`, which is flattened into the
//! [`ResNet12::output_dim`]-vector consumed by the downstream meta-classifier.
//!
//! The canonical channel widths are `[64, 160, 320, 640]`; these are
//! configurable through [`ResNet12Config::widths`].
//!
//! All convolutions are zero-padded "same" convolutions computed by a direct
//! nested-loop implementation (no im2col, no FFT) — the goal is correctness and
//! portability for a Pure-Rust meta-learning library.  Batch normalisation uses
//! the activation statistics of the current example (the "fresh"/transductive
//! behaviour standard in episodic meta-learning), with learnable per-channel
//! `γ`/`β`; no running moments are stored.

use crate::error::{MetaError, MetaResult};
use crate::handle::LcgRng;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Hyper-parameters for a [`ResNet12`] backbone.
#[derive(Debug, Clone)]
pub struct ResNet12Config {
    /// Number of input channels (1 for Omniglot greyscale, 3 for ImageNet RGB).
    pub in_channels: usize,
    /// Output channel width of each of the four residual stages.  The canonical
    /// ResNet-12 uses `[64, 160, 320, 640]`.
    pub widths: [usize; 4],
    /// Input image height — must be a positive multiple of 16 (four 2×2 pools).
    pub input_h: usize,
    /// Input image width — must be a positive multiple of 16 (four 2×2 pools).
    pub input_w: usize,
}

impl ResNet12Config {
    /// The canonical ResNet-12 configuration (`[64, 160, 320, 640]` widths)
    /// for a `C × H × W` input.
    pub fn canonical(in_channels: usize, input_h: usize, input_w: usize) -> Self {
        Self {
            in_channels,
            widths: [64, 160, 320, 640],
            input_h,
            input_w,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Convolution / BN / pool primitives (shared by all conv layers)
// ─────────────────────────────────────────────────────────────────────────────

/// 3×3 same-pad convolution with zero padding.
///
/// `weights` are laid out `[out_c, in_c, 3, 3]` row-major.  Computes
/// `out[oc, y, x] = Σ_ic Σ_ky Σ_kx W[oc, ic, ky, kx] · in[ic, y+ky−1, x+kx−1]`
/// with positions outside the image treated as zero.
fn conv3x3_same(
    x: &[f32],
    in_c: usize,
    out_c: usize,
    h: usize,
    w: usize,
    weights: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; out_c * h * w];
    for oc in 0..out_c {
        let w_oc = oc * in_c * 9;
        for y in 0..h {
            for x_pos in 0..w {
                let mut acc = 0.0_f32;
                for ic in 0..in_c {
                    let w_ic = w_oc + ic * 9;
                    let x_ic = ic * h * w;
                    for ky in 0..3_usize {
                        let iy = y as isize + ky as isize - 1;
                        if iy < 0 || iy >= h as isize {
                            continue;
                        }
                        let row = x_ic + iy as usize * w;
                        let w_row = w_ic + ky * 3;
                        for kx in 0..3_usize {
                            let ix = x_pos as isize + kx as isize - 1;
                            if ix < 0 || ix >= w as isize {
                                continue;
                            }
                            acc += weights[w_row + kx] * x[row + ix as usize];
                        }
                    }
                }
                out[oc * h * w + y * w + x_pos] = acc;
            }
        }
    }
    out
}

/// 1×1 convolution (a per-position channel projection) — the residual shortcut.
///
/// `weights` are laid out `[out_c, in_c]` row-major.  Computes
/// `out[oc, y, x] = Σ_ic W[oc, ic] · in[ic, y, x]`.
fn conv1x1(x: &[f32], in_c: usize, out_c: usize, h: usize, w: usize, weights: &[f32]) -> Vec<f32> {
    let plane = h * w;
    let mut out = vec![0.0_f32; out_c * plane];
    for oc in 0..out_c {
        let w_oc = oc * in_c;
        for ic in 0..in_c {
            let wij = weights[w_oc + ic];
            if wij == 0.0 {
                continue;
            }
            let x_ic = &x[ic * plane..(ic + 1) * plane];
            let o_oc = &mut out[oc * plane..(oc + 1) * plane];
            for (o, &xv) in o_oc.iter_mut().zip(x_ic.iter()) {
                *o += wij * xv;
            }
        }
    }
    out
}

/// Per-channel batch normalisation over the spatial plane using the current
/// activation moments, with learnable γ (`scale`) / β (`shift`).  When
/// `apply_relu` is set a clip-to-zero ReLU is fused after the affine.
fn batch_norm(
    x: &[f32],
    channels: usize,
    plane: usize,
    scale: &[f32],
    shift: &[f32],
    eps: f32,
    apply_relu: bool,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; channels * plane];
    let inv_n = 1.0_f32 / plane as f32;
    for c in 0..channels {
        let chan = &x[c * plane..(c + 1) * plane];
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
        let denom = (var + eps).sqrt();
        let gamma = scale[c];
        let beta = shift[c];
        let out_chan = &mut out[c * plane..(c + 1) * plane];
        for (o, &v) in out_chan.iter_mut().zip(chan.iter()) {
            let normed = (v - mean) / denom * gamma + beta;
            *o = if apply_relu && normed < 0.0 {
                0.0
            } else {
                normed
            };
        }
    }
    out
}

/// 2×2 stride-2 max-pool over a `(channels × h × w)` tensor, returning a
/// `(channels × h/2 × w/2)` tensor.
fn max_pool_2x2(x: &[f32], channels: usize, h: usize, w: usize) -> Vec<f32> {
    let h_out = h / 2;
    let w_out = w / 2;
    let mut out = vec![0.0_f32; channels * h_out * w_out];
    for c in 0..channels {
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

/// Initialise a `len`-element He-uniform weight buffer.  The He bound for a
/// conv layer with `fan_in` inputs is `sqrt(6 / fan_in)` — the standard
/// `kaiming_uniform` used for ResNet conv layers.
fn he_uniform(len: usize, fan_in: usize, rng: &mut LcgRng) -> Vec<f32> {
    let limit = (6.0_f32 / fan_in as f32).sqrt();
    let mut w = vec![0.0_f32; len];
    for v in w.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * limit;
    }
    w
}

// ─────────────────────────────────────────────────────────────────────────────
// Residual block
// ─────────────────────────────────────────────────────────────────────────────

/// One residual stage of ResNet-12: three stacked 3×3 conv-BN(-ReLU) layers, a
/// 1×1 conv-BN projection shortcut, the residual addition, a final ReLU, and a
/// 2×2 stride-2 max-pool.
pub struct ResBlock {
    /// Three 3×3 conv weight buffers (`[out_c, mid_in_c, 3, 3]` row-major).
    /// `conv_w[0]` maps `in_c → out_c`; `conv_w[1]`/`conv_w[2]` map `out_c → out_c`.
    conv_w: [Vec<f32>; 3],
    /// Per-channel BN γ for each of the three conv layers (each length `out_c`).
    bn_scale: [Vec<f32>; 3],
    /// Per-channel BN β for each of the three conv layers (each length `out_c`).
    bn_shift: [Vec<f32>; 3],
    /// 1×1 shortcut conv weights (`[out_c, in_c]` row-major).
    shortcut_w: Vec<f32>,
    /// Shortcut BN γ (length `out_c`).
    shortcut_scale: Vec<f32>,
    /// Shortcut BN β (length `out_c`).
    shortcut_shift: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    bn_eps: f32,
}

impl ResBlock {
    fn new(in_channels: usize, out_channels: usize, rng: &mut LcgRng) -> Self {
        // First 3×3 conv: in_channels → out_channels (fan_in = in_channels·9).
        let conv0 = he_uniform(out_channels * in_channels * 9, in_channels * 9, rng);
        // Second / third 3×3 conv: out_channels → out_channels.
        let conv1 = he_uniform(out_channels * out_channels * 9, out_channels * 9, rng);
        let conv2 = he_uniform(out_channels * out_channels * 9, out_channels * 9, rng);
        // 1×1 shortcut: in_channels → out_channels (fan_in = in_channels).
        let shortcut_w = he_uniform(out_channels * in_channels, in_channels, rng);
        Self {
            conv_w: [conv0, conv1, conv2],
            bn_scale: [
                vec![1.0; out_channels],
                vec![1.0; out_channels],
                vec![1.0; out_channels],
            ],
            bn_shift: [
                vec![0.0; out_channels],
                vec![0.0; out_channels],
                vec![0.0; out_channels],
            ],
            shortcut_w,
            shortcut_scale: vec![1.0; out_channels],
            shortcut_shift: vec![0.0; out_channels],
            in_channels,
            out_channels,
            bn_eps: 1e-5,
        }
    }

    /// Number of trainable parameters in this block (three 3×3 convs + their
    /// BN affines, plus the 1×1 shortcut conv + its BN affine).
    fn n_params(&self) -> usize {
        let conv: usize = self.conv_w.iter().map(|c| c.len()).sum();
        let bn: usize = self
            .bn_scale
            .iter()
            .chain(self.bn_shift.iter())
            .map(|b| b.len())
            .sum();
        conv + bn + self.shortcut_w.len() + self.shortcut_scale.len() + self.shortcut_shift.len()
    }

    /// Apply the residual block to a `(in_channels × h × w)` row-major tensor,
    /// returning the post-pool `(out_channels × h/2 × w/2)` tensor.
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
                msg: format!("ResBlock input too small to pool: {h}x{w}"),
            });
        }
        let plane = h * w;

        // Main path: conv0 → BN+ReLU → conv1 → BN+ReLU → conv2 → BN (no ReLU yet).
        let c0 = conv3x3_same(
            x,
            self.in_channels,
            self.out_channels,
            h,
            w,
            &self.conv_w[0],
        );
        let a0 = batch_norm(
            &c0,
            self.out_channels,
            plane,
            &self.bn_scale[0],
            &self.bn_shift[0],
            self.bn_eps,
            true,
        );
        let c1 = conv3x3_same(
            &a0,
            self.out_channels,
            self.out_channels,
            h,
            w,
            &self.conv_w[1],
        );
        let a1 = batch_norm(
            &c1,
            self.out_channels,
            plane,
            &self.bn_scale[1],
            &self.bn_shift[1],
            self.bn_eps,
            true,
        );
        let c2 = conv3x3_same(
            &a1,
            self.out_channels,
            self.out_channels,
            h,
            w,
            &self.conv_w[2],
        );
        let main = batch_norm(
            &c2,
            self.out_channels,
            plane,
            &self.bn_scale[2],
            &self.bn_shift[2],
            self.bn_eps,
            false,
        );

        // Shortcut path: 1×1 conv → BN (channel-matching projection).
        let sc = conv1x1(
            x,
            self.in_channels,
            self.out_channels,
            h,
            w,
            &self.shortcut_w,
        );
        let shortcut = batch_norm(
            &sc,
            self.out_channels,
            plane,
            &self.shortcut_scale,
            &self.shortcut_shift,
            self.bn_eps,
            false,
        );

        // Residual add then final ReLU.
        let mut summed = main;
        for (s, &h_sc) in summed.iter_mut().zip(shortcut.iter()) {
            let v = *s + h_sc;
            *s = if v > 0.0 { v } else { 0.0 };
        }

        // 2×2 stride-2 max-pool.
        Ok(max_pool_2x2(&summed, self.out_channels, h, w))
    }

    /// Number of input channels into this block.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Number of output channels from this block.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backbone
// ─────────────────────────────────────────────────────────────────────────────

/// Four-stage ResNet-12 backbone.
///
/// `forward(x)` consumes a row-major `(in_channels × input_h × input_w)` tensor
/// and returns a flattened feature vector of length
/// `widths[3] · (input_h/16) · (input_w/16)`.
pub struct ResNet12 {
    blocks: [ResBlock; 4],
    cfg: ResNet12Config,
}

impl ResNet12 {
    /// Construct a ResNet-12 backbone with He-uniform conv weights and identity
    /// BN affine (γ = 1, β = 0) throughout.
    ///
    /// # Errors
    /// * [`MetaError::BackboneError`] if `in_channels == 0` or any stage width
    ///   is `0`.
    /// * [`MetaError::InvalidEpisodeConfig`] if `input_h`/`input_w` is not a
    ///   positive multiple of 16.
    pub fn new(cfg: ResNet12Config, rng: &mut LcgRng) -> MetaResult<Self> {
        if cfg.in_channels == 0 {
            return Err(MetaError::BackboneError {
                msg: "in_channels must be > 0".into(),
            });
        }
        if cfg.widths.contains(&0) {
            return Err(MetaError::BackboneError {
                msg: "all stage widths must be > 0".into(),
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
            ResBlock::new(cfg.in_channels, cfg.widths[0], rng),
            ResBlock::new(cfg.widths[0], cfg.widths[1], rng),
            ResBlock::new(cfg.widths[1], cfg.widths[2], rng),
            ResBlock::new(cfg.widths[2], cfg.widths[3], rng),
        ];
        Ok(Self { blocks, cfg })
    }

    /// Construct the canonical `[64, 160, 320, 640]` ResNet-12.
    pub fn canonical(
        in_channels: usize,
        input_h: usize,
        input_w: usize,
        rng: &mut LcgRng,
    ) -> MetaResult<Self> {
        Self::new(
            ResNet12Config::canonical(in_channels, input_h, input_w),
            rng,
        )
    }

    /// Length of the flattened feature vector returned by [`Self::forward`]:
    /// `widths[3] · (input_h/16) · (input_w/16)`.
    pub fn output_dim(&self) -> usize {
        self.cfg.widths[3] * (self.cfg.input_h / 16) * (self.cfg.input_w / 16)
    }

    /// Total number of trainable parameters across the four residual stages.
    pub fn n_params(&self) -> usize {
        self.blocks.iter().map(|b| b.n_params()).sum()
    }

    /// Read-only access to a specific residual block (for instrumentation).
    ///
    /// # Errors
    /// [`MetaError::Internal`] if `idx >= 4`.
    pub fn block(&self, idx: usize) -> MetaResult<&ResBlock> {
        self.blocks.get(idx).ok_or(MetaError::Internal {
            msg: format!("ResNet12 has 4 blocks, asked for index {idx}"),
        })
    }

    /// Read-only access to the configuration.
    pub fn config(&self) -> &ResNet12Config {
        &self.cfg
    }

    /// Forward a single example through the four stages and flatten the result.
    ///
    /// `x`: row-major `(in_channels × input_h × input_w)`.
    /// Returns a flattened [`Self::output_dim`]-vector.
    ///
    /// # Errors
    /// [`MetaError::DimensionMismatch`] if `x.len()` does not equal
    /// `in_channels · input_h · input_w`.
    pub fn forward(&self, x: &[f32]) -> MetaResult<Vec<f32>> {
        let (_c, _h, _w, out) = self.forward_partial(x, 4)?;
        Ok(out)
    }

    /// Run only the first `n_blocks` residual stages and expose the (unflattened)
    /// feature map as `(channels, h, w, data)` with `data` of length
    /// `channels · h · w` row-major.  Used for inspecting intermediate
    /// activations (e.g. residual ReLU non-negativity).
    ///
    /// # Errors
    /// [`MetaError::Internal`] if `n_blocks > 4`; [`MetaError::DimensionMismatch`]
    /// if `x` has the wrong length.
    pub fn forward_partial(
        &self,
        x: &[f32],
        n_blocks: usize,
    ) -> MetaResult<(usize, usize, usize, Vec<f32>)> {
        if n_blocks > self.blocks.len() {
            return Err(MetaError::Internal {
                msg: format!("ResNet12 has 4 blocks, asked for first {n_blocks}"),
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
        let mut channels = self.cfg.in_channels;
        for block in self.blocks.iter().take(n_blocks) {
            cur = block.forward(&cur, h, w)?;
            channels = block.out_channels();
            h /= 2;
            w /= 2;
        }
        Ok((channels, h, w, cur))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> ResNet12Config {
        ResNet12Config {
            in_channels: 1,
            widths: [2, 3, 4, 5],
            input_h: 16,
            input_w: 16,
        }
    }

    fn make_backbone(cfg: ResNet12Config) -> ResNet12 {
        let mut rng = LcgRng::new(2026);
        ResNet12::new(cfg, &mut rng).expect("valid ResNet12 cfg")
    }

    // ── construction validation ──────────────────────────────────────────────

    #[test]
    fn new_valid_cfg_succeeds() {
        let mut rng = LcgRng::new(1);
        assert!(ResNet12::new(tiny_cfg(), &mut rng).is_ok());
    }

    #[test]
    fn new_in_channels_zero_errs() {
        let mut cfg = tiny_cfg();
        cfg.in_channels = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ResNet12::new(cfg, &mut rng),
            Err(MetaError::BackboneError { .. })
        ));
    }

    #[test]
    fn new_zero_width_errs() {
        let mut cfg = tiny_cfg();
        cfg.widths[2] = 0;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ResNet12::new(cfg, &mut rng),
            Err(MetaError::BackboneError { .. })
        ));
    }

    #[test]
    fn new_input_h_not_divisible_by_16_errs() {
        let mut cfg = tiny_cfg();
        cfg.input_h = 24;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ResNet12::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    #[test]
    fn new_input_w_not_divisible_by_16_errs() {
        let mut cfg = tiny_cfg();
        cfg.input_w = 17;
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ResNet12::new(cfg, &mut rng),
            Err(MetaError::InvalidEpisodeConfig { .. })
        ));
    }

    // ── output_dim / n_params ────────────────────────────────────────────────

    #[test]
    fn output_dim_formula_holds() {
        // widths[3]=5, 16/16 = 1 → 5·1·1 = 5.
        let bb = make_backbone(tiny_cfg());
        assert_eq!(bb.output_dim(), 5);
    }

    #[test]
    fn output_dim_pool_halving() {
        // 32×32 input → 32/16 = 2 → widths[3]·2·2.
        let cfg = ResNet12Config {
            in_channels: 1,
            widths: [2, 3, 4, 5],
            input_h: 32,
            input_w: 32,
        };
        let bb = make_backbone(cfg);
        assert_eq!(bb.output_dim(), 5 * 2 * 2);
    }

    #[test]
    fn n_params_formula() {
        // Block 0 (in=1, out=2): conv0 = 2·1·9 = 18, conv1 = conv2 = 2·2·9 = 36
        // each → 18+36+36 = 90 conv weights; BN γ/β = 3·(2+2) = 12; shortcut
        // 1×1 = 2·1 = 2 + BN 2+2 = 4 → 6.  Block 0 = 90 + 12 + 6 = 108.
        let bb = make_backbone(tiny_cfg());
        let block0 = {
            // Block 0 maps in_c=1 → out_c=2.
            let in_c = 1_usize;
            let out_c = 2_usize;
            // conv0 = out·in·9, conv1 = conv2 = out·out·9.
            let conv = out_c * in_c * 9 + out_c * out_c * 9 + out_c * out_c * 9;
            // Three conv layers, each with γ and β of length out_c.
            let bn = 3 * (out_c + out_c);
            // 1×1 shortcut (out·in) + its BN γ/β.
            let sc = (out_c * in_c) + (out_c + out_c);
            conv + bn + sc
        };
        let block0_actual = bb.block(0).expect("block 0").n_params();
        assert_eq!(block0_actual, block0);
        // Total equals the sum of the four blocks.
        let total: usize = (0..4).map(|i| bb.block(i).expect("block").n_params()).sum();
        assert_eq!(bb.n_params(), total);
    }

    // ── forward correctness ──────────────────────────────────────────────────

    #[test]
    fn forward_output_length_matches_output_dim() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let mut rng = LcgRng::new(11);
        let x: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let y = bb.forward(&x).expect("forward ok");
        assert_eq!(y.len(), bb.output_dim());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_deterministic_with_same_seed() {
        let mut rng_a = LcgRng::new(99);
        let mut rng_b = LcgRng::new(99);
        let bb_a = ResNet12::new(tiny_cfg(), &mut rng_a).expect("ok");
        let bb_b = ResNet12::new(tiny_cfg(), &mut rng_b).expect("ok");
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
    fn forward_multi_channel_works() {
        let cfg = ResNet12Config {
            in_channels: 3,
            widths: [4, 6, 8, 10],
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
    fn forward_partial_dim_halving() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let x = vec![0.3_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        for k in 1..=4_usize {
            let (channels, h, w, fmap) = bb.forward_partial(&x, k).expect("ok");
            assert_eq!(channels, cfg.widths[k - 1]);
            assert_eq!(h, cfg.input_h >> k);
            assert_eq!(w, cfg.input_w >> k);
            assert_eq!(fmap.len(), channels * h * w);
        }
    }

    #[test]
    fn forward_partial_residual_relu_non_negative() {
        // After a residual stage the output is post-(residual-add ReLU) and then
        // max-pooled, so every value is ≥ 0.
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let mut rng = LcgRng::new(31);
        let x: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();
        let (_c, _h, _w, fmap) = bb.forward_partial(&x, 1).expect("ok");
        for &v in fmap.iter() {
            assert!(
                v >= 0.0,
                "post-residual ReLU activation must be ≥ 0, got {v}"
            );
        }
    }

    #[test]
    fn forward_partial_zero_blocks_is_input() {
        let bb = make_backbone(tiny_cfg());
        let cfg = bb.config().clone();
        let x: Vec<f32> = (0..cfg.in_channels * cfg.input_h * cfg.input_w)
            .map(|i| i as f32 * 0.01)
            .collect();
        let (channels, h, w, fmap) = bb.forward_partial(&x, 0).expect("ok");
        assert_eq!(channels, cfg.in_channels);
        assert_eq!(h, cfg.input_h);
        assert_eq!(w, cfg.input_w);
        assert_eq!(fmap, x);
    }

    #[test]
    fn block_out_of_range_errs() {
        let bb = make_backbone(tiny_cfg());
        assert!(matches!(bb.block(4), Err(MetaError::Internal { .. })));
    }

    #[test]
    fn canonical_widths() {
        let cfg = ResNet12Config::canonical(3, 16, 16);
        assert_eq!(cfg.widths, [64, 160, 320, 640]);
        assert_eq!(cfg.in_channels, 3);
    }

    #[test]
    fn identity_shortcut_sanity_constant_input_interior() {
        // For a spatially-constant input, the *fully-interior* region of the
        // residual stage is constant per output channel, and BN+ReLU+residual+
        // pool preserve that constancy there.
        //
        // A ResBlock stacks THREE 3×3 same-pad convs before pooling.  Each conv
        // contracts the region whose entire receptive field is interior by one
        // cell on every side, so a pre-pool position is guaranteed constant only
        // if it is at least 3 cells from every border (the input row/col index
        // lies in `[3, h_pre-4]`).  The 2×2 stride-2 pool of a pooled cell
        // `(py, px)` reads pre-pool rows `{2·py, 2·py+1}`; requiring both to lie
        // in `[3, h_pre-4]` (here `[3, 12]`) restricts the safe pooled cells to
        // `py ∈ [2, h_post-3]`.  (The single-conv Conv4 block only needs a
        // 1-cell margin, which is why its analogous test uses `1..h-1`.)
        let cfg = ResNet12Config {
            in_channels: 2,
            widths: [3, 4, 5, 6],
            input_h: 16,
            input_w: 16,
        };
        let bb = make_backbone(cfg);
        let cfg = bb.config().clone();
        let x = vec![0.5_f32; cfg.in_channels * cfg.input_h * cfg.input_w];
        let (channels, h, w, fmap) = bb.forward_partial(&x, 1).expect("ok");
        let chan_len = h * w;
        for c in 0..channels {
            let chan = &fmap[c * chan_len..(c + 1) * chan_len];
            let pivot = chan[(h / 2) * w + (w / 2)];
            // Safe pooled interior: 2 ≤ idx ≤ h_post-3 (i.e. `2..(h-2)`).
            for y in 2..(h - 2) {
                for x_pos in 2..(w - 2) {
                    let v = chan[y * w + x_pos];
                    assert!(
                        (v - pivot).abs() < 1e-4,
                        "channel {c} interior at ({y},{x_pos}) not constant: {v} vs {pivot}"
                    );
                }
            }
        }
    }
}
