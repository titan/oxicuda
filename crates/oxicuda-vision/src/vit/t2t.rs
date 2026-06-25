//! T2T-ViT — Tokens-to-Token Vision Transformer (Yuan 2021, "Tokens-to-Token
//! ViT: Training Vision Transformers from Scratch on ImageNet", ICCV).
//!
//! The defining operation of T2T-ViT is the **soft split** (a.k.a.
//! re-structurization / re-tokenization). Rather than embedding the image with
//! one hard, non-overlapping patch grid (as in vanilla ViT), T2T progressively
//! aggregates neighbouring tokens with *overlapping* windows so that local
//! structure (edges, lines) is modelled and the token length is reduced step by
//! step:
//!
//! 1. **Reshape** the `[L, C]` token sequence back to a 2-D feature map
//!    `[C, H, W]` with `H·W = L`.
//! 2. **Soft split** = unfold the map with a `k × k` window, `stride s`, and
//!    `padding p` (overlapping when `s < k`). Each output location collects a
//!    flattened `C·k·k` neighbourhood — this is exactly `Im2Col`/`nn.Unfold`.
//! 3. The unfolded tensor `[L', C·k·k]` becomes the new (longer-feature, shorter-
//!    length) token sequence, which a transformer then mixes.
//!
//! This module implements the soft-split ([`soft_split`]) and its output-size
//! arithmetic ([`SoftSplitConfig::output_hw`]), plus a re-tokenization helper
//! that chains *reshape → soft split* ([`T2tModule::tokens_to_token`]). The
//! per-step transformer mixing reuses the existing ViT blocks and is left to the
//! caller.
//!
//! Layout: feature maps are flat row-major `[C, H, W]`; token sequences are
//! `[L, C]` row-major.

use crate::error::{VisionError, VisionResult};

// ─── Soft-split config ─────────────────────────────────────────────────────────

/// Parameters of one Tokens-to-Token soft-split (overlapping unfold).
#[derive(Debug, Clone, PartialEq)]
pub struct SoftSplitConfig {
    /// Square window (kernel) size `k`.
    pub kernel: usize,
    /// Stride `s` between successive windows (`s < k` ⇒ overlap).
    pub stride: usize,
    /// Zero-padding `p` applied symmetrically on all four borders.
    pub padding: usize,
}

impl SoftSplitConfig {
    /// Create and validate a `SoftSplitConfig`.
    ///
    /// # Errors
    /// - [`VisionError::Internal`] if `kernel == 0` or `stride == 0`.
    pub fn new(kernel: usize, stride: usize, padding: usize) -> VisionResult<Self> {
        if kernel == 0 {
            return Err(VisionError::Internal(
                "soft-split kernel must be > 0".into(),
            ));
        }
        if stride == 0 {
            return Err(VisionError::Internal(
                "soft-split stride must be > 0".into(),
            ));
        }
        Ok(Self {
            kernel,
            stride,
            padding,
        })
    }

    /// The canonical first-stage T2T config (`k = 7`, `s = 4`, `p = 2`).
    #[must_use]
    pub fn stage1() -> Self {
        Self {
            kernel: 7,
            stride: 4,
            padding: 2,
        }
    }

    /// The canonical deeper-stage T2T config (`k = 3`, `s = 2`, `p = 1`).
    #[must_use]
    pub fn stage_deep() -> Self {
        Self {
            kernel: 3,
            stride: 2,
            padding: 1,
        }
    }

    /// Output spatial size for an `(h, w)` input under this soft split.
    ///
    /// Uses the standard convolution formula
    /// `out = floor((in + 2p − k) / s) + 1`.
    ///
    /// # Errors
    /// - [`VisionError::InvalidImageSize`] if the padded input is smaller than
    ///   the kernel (no valid window position).
    pub fn output_hw(&self, h: usize, w: usize) -> VisionResult<(usize, usize)> {
        let out = |n: usize| -> VisionResult<usize> {
            let padded = n + 2 * self.padding;
            if padded < self.kernel {
                return Err(VisionError::InvalidImageSize {
                    height: h,
                    width: w,
                    channels: 0,
                });
            }
            Ok((padded - self.kernel) / self.stride + 1)
        };
        Ok((out(h)?, out(w)?))
    }

    /// Per-token feature length after a soft split of a `c`-channel map
    /// (`c · k · k`).
    #[must_use]
    #[inline]
    pub fn unfold_dim(&self, c: usize) -> usize {
        c * self.kernel * self.kernel
    }
}

// ─── Soft split (overlapping unfold / Im2Col) ──────────────────────────────────

/// Soft split: unfold a `[C, H, W]` feature map into overlapping `k × k`
/// neighbourhoods.
///
/// Returns `(tokens, out_h, out_w)` where `tokens` is `[out_h·out_w, C·k·k]`
/// row-major: each row is the flattened neighbourhood centred on one output
/// location, laid out as `[c0(k·k), c1(k·k), …]` (channel-major within a row to
/// match `nn.Unfold`'s `(C, kh, kw)` ordering). Out-of-bounds taps (from
/// padding) contribute zeros.
///
/// # Errors
/// - [`VisionError::EmptyInput`] if any dimension is 0.
/// - [`VisionError::DimensionMismatch`] if `map.len() != c * h * w`.
/// - [`VisionError::InvalidImageSize`] if the window does not fit.
pub fn soft_split(
    map: &[f32],
    c: usize,
    h: usize,
    w: usize,
    cfg: &SoftSplitConfig,
) -> VisionResult<(Vec<f32>, usize, usize)> {
    if c == 0 || h == 0 || w == 0 {
        return Err(VisionError::EmptyInput("soft_split map dims"));
    }
    if map.len() != c * h * w {
        return Err(VisionError::DimensionMismatch {
            expected: c * h * w,
            got: map.len(),
        });
    }
    let (out_h, out_w) = cfg.output_hw(h, w)?;
    let k = cfg.kernel;
    let s = cfg.stride;
    let p = cfg.padding;
    let row_dim = cfg.unfold_dim(c);
    let n_out = out_h * out_w;
    let mut tokens = vec![0.0f32; n_out * row_dim];

    for oy in 0..out_h {
        for ox in 0..out_w {
            let out_idx = oy * out_w + ox;
            let row = &mut tokens[out_idx * row_dim..(out_idx + 1) * row_dim];
            // Top-left source coordinate (in padded space, then shift by −p).
            let base_y = (oy * s) as isize - p as isize;
            let base_x = (ox * s) as isize - p as isize;
            for ch in 0..c {
                let plane = &map[ch * h * w..(ch + 1) * h * w];
                let ch_off = ch * k * k;
                for ky in 0..k {
                    let sy = base_y + ky as isize;
                    for kx in 0..k {
                        let sx = base_x + kx as isize;
                        let dst = ch_off + ky * k + kx;
                        if sy >= 0 && sy < h as isize && sx >= 0 && sx < w as isize {
                            row[dst] = plane[sy as usize * w + sx as usize];
                        }
                        // else: padding → leave as 0.0
                    }
                }
            }
        }
    }
    Ok((tokens, out_h, out_w))
}

// ─── Token ⇄ feature-map reshape ───────────────────────────────────────────────

/// Reshape a `[L, C]` token sequence into a `[C, H, W]` feature map.
///
/// Token `l = y·W + x` maps to spatial location `(y, x)`. Requires `H·W == L`.
///
/// # Errors
/// - [`VisionError::DimensionMismatch`] if `tokens.len() != l * c` or
///   `h * w != l`.
pub fn tokens_to_map(
    tokens: &[f32],
    l: usize,
    c: usize,
    h: usize,
    w: usize,
) -> VisionResult<Vec<f32>> {
    if tokens.len() != l * c {
        return Err(VisionError::DimensionMismatch {
            expected: l * c,
            got: tokens.len(),
        });
    }
    if h * w != l {
        return Err(VisionError::DimensionMismatch {
            expected: l,
            got: h * w,
        });
    }
    let mut map = vec![0.0f32; c * h * w];
    for y in 0..h {
        for x in 0..w {
            let tok = y * w + x;
            for ch in 0..c {
                map[ch * h * w + y * w + x] = tokens[tok * c + ch];
            }
        }
    }
    Ok(map)
}

// ─── T2T module ────────────────────────────────────────────────────────────────

/// A Tokens-to-Token re-structurization stage: `reshape → soft split`.
///
/// Stores the soft-split config and the channel count expected on input, and
/// exposes the end-to-end token re-tokenization used between T2T transformer
/// stages.
#[derive(Debug, Clone)]
pub struct T2tModule {
    cfg: SoftSplitConfig,
    in_channels: usize,
}

impl T2tModule {
    /// Create a T2T module operating on `in_channels`-channel token features.
    ///
    /// # Errors
    /// - [`VisionError::EmptyInput`] if `in_channels == 0`.
    pub fn new(cfg: SoftSplitConfig, in_channels: usize) -> VisionResult<Self> {
        if in_channels == 0 {
            return Err(VisionError::EmptyInput("t2t in_channels"));
        }
        Ok(Self { cfg, in_channels })
    }

    /// Input channel count.
    #[must_use]
    #[inline]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Feature length of each output token (`in_channels · k · k`).
    #[must_use]
    #[inline]
    pub fn out_feature_dim(&self) -> usize {
        self.cfg.unfold_dim(self.in_channels)
    }

    /// Re-tokenize: reshape `[L, in_channels]` tokens to a map of size `(h, w)`
    /// then soft-split into `[L', in_channels·k·k]` tokens.
    ///
    /// Returns `(new_tokens, out_h, out_w)`.
    ///
    /// # Errors
    /// Propagates [`tokens_to_map`] and [`soft_split`].
    pub fn tokens_to_token(
        &self,
        tokens: &[f32],
        l: usize,
        h: usize,
        w: usize,
    ) -> VisionResult<(Vec<f32>, usize, usize)> {
        let map = tokens_to_map(tokens, l, self.in_channels, h, w)?;
        soft_split(&map, self.in_channels, h, w, &self.cfg)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::LcgRng;

    #[test]
    fn config_validation_and_output_size() {
        assert!(SoftSplitConfig::new(0, 1, 0).is_err());
        assert!(SoftSplitConfig::new(3, 0, 1).is_err());
        let cfg = SoftSplitConfig::stage1(); // k=7, s=4, p=2
        // 224 → (224 + 4 - 7)/4 + 1 = 56
        let (oh, ow) = cfg.output_hw(224, 224).expect("ok");
        assert_eq!((oh, ow), (56, 56));
        // deep stage: 56 → (56 + 2 - 3)/2 + 1 = 28
        let deep = SoftSplitConfig::stage_deep();
        let (oh2, ow2) = deep.output_hw(56, 56).expect("ok");
        assert_eq!((oh2, ow2), (28, 28));
    }

    #[test]
    fn output_size_too_small_errors() {
        let cfg = SoftSplitConfig::new(7, 4, 0).expect("ok");
        assert!(cfg.output_hw(3, 3).is_err());
    }

    #[test]
    fn unfold_dim_correct() {
        let cfg = SoftSplitConfig::new(3, 2, 1).expect("ok");
        assert_eq!(cfg.unfold_dim(4), 4 * 9);
    }

    #[test]
    fn soft_split_non_overlap_identity_window1() {
        // k=1, s=1, p=0 → soft split is a no-op reshape: tokens == map values.
        let cfg = SoftSplitConfig::new(1, 1, 0).expect("ok");
        let c = 2;
        let h = 3;
        let w = 3;
        let map: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        let (tokens, oh, ow) = soft_split(&map, c, h, w, &cfg).expect("ok");
        assert_eq!((oh, ow), (3, 3));
        // row_dim = c*1*1 = 2. For each output (y,x): [plane0(y,x), plane1(y,x)].
        for y in 0..h {
            for x in 0..w {
                let out_idx = y * w + x;
                let row = &tokens[out_idx * c..(out_idx + 1) * c];
                assert_eq!(row[0], map[y * w + x]);
                assert_eq!(row[1], map[h * w + y * w + x]);
            }
        }
    }

    #[test]
    fn soft_split_padding_produces_zeros_at_border() {
        // k=3, s=1, p=1 on a 1-channel 3×3 of ones. The top-left output window
        // (centred at (0,0)) has its top row and left column out of bounds → 4
        // of its 9 taps must be zero (the 3 padded + corner).
        let cfg = SoftSplitConfig::new(3, 1, 1).expect("ok");
        let c = 1;
        let h = 3;
        let w = 3;
        let map = vec![1.0f32; c * h * w];
        let (tokens, oh, ow) = soft_split(&map, c, h, w, &cfg).expect("ok");
        assert_eq!((oh, ow), (3, 3));
        let row0 = &tokens[0..9]; // top-left output
        // The 3×3 window for output (0,0): rows ky=0 (sy=-1, all pad),
        // ky=1 (sy=0): kx=0 pad, kx=1,2 valid; ky=2 (sy=1): kx=0 pad, kx=1,2 valid.
        let n_zero = row0.iter().filter(|&&v| v == 0.0).count();
        let n_one = row0.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(n_zero, 5, "expected 5 padded taps, got {n_zero}");
        assert_eq!(n_one, 4, "expected 4 valid taps, got {n_one}");
    }

    #[test]
    fn soft_split_overlap_reduces_length() {
        // Overlapping windows: a 4×4 map with k=2,s=2,p=0 → 2×2 = 4 output tokens.
        let cfg = SoftSplitConfig::new(2, 2, 0).expect("ok");
        let c = 1;
        let h = 4;
        let w = 4;
        let map: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let (tokens, oh, ow) = soft_split(&map, c, h, w, &cfg).expect("ok");
        assert_eq!((oh, ow), (2, 2));
        assert_eq!(tokens.len(), 4 * 4); // 4 tokens × (1*2*2)
        // First window (oy=0,ox=0) collects map[(0,0),(0,1),(1,0),(1,1)] = 0,1,4,5.
        assert_eq!(&tokens[0..4], &[0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn soft_split_validation() {
        let cfg = SoftSplitConfig::stage1();
        assert!(soft_split(&[], 0, 4, 4, &cfg).is_err());
        let map = vec![0.0f32; 10];
        assert!(soft_split(&map, 2, 3, 3, &cfg).is_err()); // 2*3*3 != 10
    }

    #[test]
    fn tokens_to_map_roundtrip() {
        // map → (flatten as tokens) → tokens_to_map should recover the map.
        let c = 3;
        let h = 2;
        let w = 4;
        let l = h * w;
        let map: Vec<f32> = (0..c * h * w).map(|i| i as f32).collect();
        // Build tokens [L, C] from the map.
        let mut tokens = vec![0.0f32; l * c];
        for y in 0..h {
            for x in 0..w {
                let tok = y * w + x;
                for ch in 0..c {
                    tokens[tok * c + ch] = map[ch * h * w + y * w + x];
                }
            }
        }
        let recovered = tokens_to_map(&tokens, l, c, h, w).expect("ok");
        assert_eq!(recovered, map);
    }

    #[test]
    fn tokens_to_map_validation() {
        assert!(tokens_to_map(&[0.0; 5], 2, 3, 1, 2).is_err()); // 2*3 != 5
        assert!(tokens_to_map(&[0.0; 6], 2, 3, 2, 2).is_err()); // h*w != l
    }

    #[test]
    fn t2t_module_end_to_end() {
        let cfg = SoftSplitConfig::new(3, 2, 1).expect("ok");
        let c = 4;
        let module = T2tModule::new(cfg, c).expect("ok");
        let h = 8;
        let w = 8;
        let l = h * w;
        let mut rng = LcgRng::new(1);
        let mut tokens = vec![0.0f32; l * c];
        rng.fill_normal(&mut tokens);
        let (new_tokens, oh, ow) = module.tokens_to_token(&tokens, l, h, w).expect("ok");
        // (8 + 2 - 3)/2 + 1 = 4 → 4×4 output.
        assert_eq!((oh, ow), (4, 4));
        assert_eq!(new_tokens.len(), oh * ow * module.out_feature_dim());
        assert_eq!(module.out_feature_dim(), c * 9);
        assert!(new_tokens.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn t2t_module_validation() {
        let cfg = SoftSplitConfig::stage1();
        assert!(T2tModule::new(cfg, 0).is_err());
    }

    #[test]
    fn deterministic() {
        let cfg = SoftSplitConfig::new(3, 2, 1).expect("ok");
        let module = T2tModule::new(cfg, 2).expect("ok");
        let h = 6;
        let w = 6;
        let l = h * w;
        let mut r1 = LcgRng::new(5);
        let mut t1 = vec![0.0f32; l * 2];
        r1.fill_normal(&mut t1);
        let t2 = t1.clone();
        let a = module.tokens_to_token(&t1, l, h, w).expect("ok");
        let b = module.tokens_to_token(&t2, l, h, w).expect("ok");
        assert_eq!(a.0, b.0);
    }
}
