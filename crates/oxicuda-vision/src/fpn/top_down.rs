//! Feature Pyramid Network top-down pathway.
//!
//! Implements the FPN as described in "Feature Pyramid Networks for Object Detection"
//! (Lin et al., 2017).  The forward pass:
//!
//! 1. Apply a 1×1 lateral convolution per backbone level to unify channel counts.
//! 2. Top-down merge: start from the coarsest level, upsample (nearest-neighbour)
//!    and add to the next finer level.
//! 3. Apply a 3×3 "anti-aliasing" (smoothing) convolution to each merged level.

use super::lateral::LateralConv1x1;
use crate::{
    error::{VisionError, VisionResult},
    handle::LcgRng,
};

// ─── FeatureMap ───────────────────────────────────────────────────────────────

/// A single-level feature map in CHW layout.
///
/// `data` is stored as `[channels, height, width]` in row-major order.
#[derive(Debug, Clone)]
pub struct FeatureMap {
    /// Flat data buffer: `[channels × height × width]`.
    pub data: Vec<f32>,
    /// Number of feature channels.
    pub channels: usize,
    /// Spatial height (rows).
    pub height: usize,
    /// Spatial width (columns).
    pub width: usize,
}

impl FeatureMap {
    /// Construct a `FeatureMap`, validating that the data length matches the shape.
    ///
    /// # Errors
    /// `DimensionMismatch` if `data.len() != channels * height * width`.
    pub fn new(data: Vec<f32>, channels: usize, height: usize, width: usize) -> VisionResult<Self> {
        let expected = channels * height * width;
        if data.len() != expected {
            return Err(VisionError::DimensionMismatch {
                expected,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            channels,
            height,
            width,
        })
    }

    /// Sample the value at channel `c`, row `h_idx`, column `w_idx`.
    ///
    /// Panics in debug builds if any index is out of range (same contract as
    /// direct slice indexing — callers are responsible for bounds checking).
    #[inline]
    pub fn at(&self, c: usize, h_idx: usize, w_idx: usize) -> f32 {
        self.data[c * self.height * self.width + h_idx * self.width + w_idx]
    }

    /// Total number of elements.
    #[inline]
    fn len(&self) -> usize {
        self.channels * self.height * self.width
    }
}

// ─── FpnConfig ────────────────────────────────────────────────────────────────

/// Configuration for the Feature Pyramid Network.
pub struct FpnConfig {
    /// Input channel counts per backbone level, ordered **coarse→fine**
    /// (e.g. `[2048, 1024, 512, 256]`).
    pub in_channels: Vec<usize>,
    /// Output channel count (uniform across all FPN levels after lateral convs).
    pub out_channels: usize,
}

impl FpnConfig {
    /// Construct and validate an `FpnConfig`.
    ///
    /// # Errors
    /// - `EmptyInput` if `in_channels` is empty.
    /// - `InvalidImageSize` if `out_channels == 0`.
    pub fn new(in_channels: Vec<usize>, out_channels: usize) -> VisionResult<Self> {
        if in_channels.is_empty() {
            return Err(VisionError::EmptyInput("FpnConfig::in_channels"));
        }
        if out_channels == 0 {
            return Err(VisionError::InvalidImageSize {
                height: 0,
                width: 0,
                channels: out_channels,
            });
        }
        Ok(Self {
            in_channels,
            out_channels,
        })
    }

    /// Number of FPN levels.
    #[inline]
    pub fn n_levels(&self) -> usize {
        self.in_channels.len()
    }
}

// ─── Fpn ──────────────────────────────────────────────────────────────────────

/// Feature Pyramid Network.
///
/// Accepts a list of feature maps at multiple scales (coarsest first) and
/// returns a pyramid with uniform `out_channels` after the top-down path.
pub struct Fpn {
    /// FPN configuration (includes level input channels and output channels).
    pub config: FpnConfig,
    /// 1×1 lateral convolutions, one per backbone level.
    pub lateral_convs: Vec<LateralConv1x1>,
    /// 3×3 smoothing convolution weights per level: each `[out_c × out_c × 9]`.
    pub smooth_weights: Vec<Vec<f32>>,
    /// Biases for the smoothing convolutions: each `[out_c]`.
    pub smooth_biases: Vec<Vec<f32>>,
}

impl Fpn {
    /// Build an FPN with Xavier-initialised weights.
    ///
    /// # Errors
    /// Propagates errors from `FpnConfig` or lateral conv construction.
    pub fn new(cfg: FpnConfig, rng: &mut LcgRng) -> VisionResult<Self> {
        let n = cfg.n_levels();
        let oc = cfg.out_channels;

        // One lateral conv per level: in_channels[l] → out_channels
        let mut lateral_convs = Vec::with_capacity(n);
        for &ic in &cfg.in_channels {
            lateral_convs.push(LateralConv1x1::new(ic, oc, rng)?);
        }

        // Xavier scale for 3×3 conv: fan_in = out_channels * 9
        let smooth_scale = 1.0_f32 / ((oc * 9) as f32).sqrt();
        let mut smooth_weights = Vec::with_capacity(n);
        let mut smooth_biases = Vec::with_capacity(n);
        for _ in 0..n {
            let kernel_size = oc * oc * 9; // [out_c × out_c × 3 × 3]
            let mut w = vec![0.0f32; kernel_size];
            rng.fill_normal(&mut w);
            for v in &mut w {
                *v *= smooth_scale;
            }
            smooth_weights.push(w);
            smooth_biases.push(vec![0.0f32; oc]);
        }

        Ok(Self {
            config: cfg,
            lateral_convs,
            smooth_weights,
            smooth_biases,
        })
    }

    /// Run the FPN forward pass.
    ///
    /// `features` must be ordered **coarse→fine**, one `FeatureMap` per level.
    ///
    /// Returns the FPN pyramid in the same coarse→fine order, with all maps
    /// having `out_channels` channels.
    ///
    /// # Errors
    /// - `DimensionMismatch` if the number of features ≠ number of configured levels.
    /// - `EmptyInput` if `features` is empty.
    /// - Propagates errors from lateral convolutions.
    pub fn forward(&self, features: Vec<FeatureMap>) -> VisionResult<Vec<FeatureMap>> {
        let n = self.config.n_levels();
        if features.is_empty() {
            return Err(VisionError::EmptyInput("Fpn::forward features"));
        }
        if features.len() != n {
            return Err(VisionError::DimensionMismatch {
                expected: n,
                got: features.len(),
            });
        }

        let oc = self.config.out_channels;

        // ── Step 1: lateral convolutions ──────────────────────────────────────
        let mut lateral_maps: Vec<FeatureMap> = Vec::with_capacity(n);
        for (l, feat) in features.iter().enumerate() {
            let h = feat.height;
            let w = feat.width;
            let lateral_data = self.lateral_convs[l].forward(&feat.data, h, w)?;
            lateral_maps.push(FeatureMap::new(lateral_data, oc, h, w)?);
        }

        // ── Step 2: top-down merge ────────────────────────────────────────────
        // merged[L-1] = lateral[L-1]  (coarsest level)
        // merged[l]   = lateral[l] + upsample(merged[l+1]) for l in (L-2)..=0
        let mut merged: Vec<FeatureMap> = Vec::with_capacity(n);
        // Clone the coarsest level to start the top-down path.
        merged.push(lateral_maps[n - 1].clone());

        // Build merged levels from (n-2) down to 0, pushing in reverse order
        // so we'll need to reverse at the end.
        for l in (0..n - 1).rev() {
            let target_h = lateral_maps[l].height;
            let target_w = lateral_maps[l].width;
            // The coarser merged level was added last to our temporary vec.
            let coarser = merged.last().ok_or_else(|| {
                VisionError::Internal("FPN merge: no coarser level pushed".into())
            })?;
            let upsampled = upsample_nearest(coarser, target_h, target_w);
            // Element-wise addition: lateral[l] + upsampled
            let lat = &lateral_maps[l];
            let mut merged_data = vec![0.0f32; lat.len()];
            for (i, v) in merged_data.iter_mut().enumerate() {
                *v = lat.data[i] + upsampled.data[i];
            }
            merged.push(FeatureMap::new(merged_data, oc, target_h, target_w)?);
        }

        // Reverse so that index 0 corresponds to the coarsest level again.
        merged.reverse();

        // ── Step 3: 3×3 smoothing conv ────────────────────────────────────────
        let mut output: Vec<FeatureMap> = Vec::with_capacity(n);
        for (l, fm) in merged.into_iter().enumerate() {
            let smooth_data = conv3x3_same(
                &fm.data,
                oc,
                fm.height,
                fm.width,
                &self.smooth_weights[l],
                &self.smooth_biases[l],
                oc,
            );
            output.push(FeatureMap::new(smooth_data, oc, fm.height, fm.width)?);
        }

        Ok(output)
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Nearest-neighbour 2D upsampling to `(target_h, target_w)`.
///
/// Each output pixel `(c, i, j)` is assigned the value from the nearest
/// input pixel using floor-based index mapping.
fn upsample_nearest(feat: &FeatureMap, target_h: usize, target_w: usize) -> FeatureMap {
    let src_h = feat.height;
    let src_w = feat.width;
    let c = feat.channels;
    let mut out = vec![0.0f32; c * target_h * target_w];

    for ch in 0..c {
        for i in 0..target_h {
            // Map output row → source row (nearest neighbour, floor)
            let src_i = (i * src_h / target_h).min(src_h.saturating_sub(1));
            for j in 0..target_w {
                let src_j = (j * src_w / target_w).min(src_w.saturating_sub(1));
                out[ch * target_h * target_w + i * target_w + j] = feat.at(ch, src_i, src_j);
            }
        }
    }

    // Cannot fail: data.len() == c * target_h * target_w by construction.
    FeatureMap {
        data: out,
        channels: c,
        height: target_h,
        width: target_w,
    }
}

/// 3×3 convolution with same (zero) padding.
///
/// - `feat`:       input `[channels, h, w]`.
/// - `channels`:   `in_channels` (must equal `out_channels` here — smoothing).
/// - `weight`:     `[out_c × in_c × 9]` (kernel laid out as `[oc][ic][ki][kj]`).
/// - `bias`:       `[out_c]`.
/// - `out_channels`: output channel count.
///
/// Returns `[out_channels, h, w]`.
fn conv3x3_same(
    feat: &[f32],
    channels: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_channels * h * w];

    for oc in 0..out_channels {
        for i in 0..h {
            for j in 0..w {
                let mut acc = bias[oc];
                for ic in 0..channels {
                    // 3×3 kernel offsets: ki ∈ {-1, 0, 1}, kj ∈ {-1, 0, 1}
                    for ki in 0..3usize {
                        let src_i = i as isize + ki as isize - 1;
                        if src_i < 0 || src_i >= h as isize {
                            continue; // zero-padding
                        }
                        for kj in 0..3usize {
                            let src_j = j as isize + kj as isize - 1;
                            if src_j < 0 || src_j >= w as isize {
                                continue; // zero-padding
                            }
                            // weight layout: [oc, ic, ki, kj]  → [oc*(ic*9) + ic*9 + ki*3 + kj]
                            let w_idx = oc * channels * 9 + ic * 9 + ki * 3 + kj;
                            let f_idx = ic * h * w + src_i as usize * w + src_j as usize;
                            acc += weight[w_idx] * feat[f_idx];
                        }
                    }
                }
                out[oc * h * w + i * w + j] = acc;
            }
        }
    }

    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> LcgRng {
        LcgRng::new(123)
    }

    /// Build a random feature map for testing.
    fn random_feature_map(rng: &mut LcgRng, channels: usize, h: usize, w: usize) -> FeatureMap {
        let n = channels * h * w;
        let mut data = vec![0.0f32; n];
        rng.fill_normal(&mut data);
        FeatureMap::new(data, channels, h, w).expect("valid feature map")
    }

    // ── FeatureMap ─────────────────────────────────────────────────────────────

    #[test]
    fn feature_map_valid_construction() {
        let data = vec![1.0f32; 3 * 4 * 4];
        let fm = FeatureMap::new(data, 3, 4, 4).expect("valid feature map");
        assert_eq!(fm.channels, 3);
        assert_eq!(fm.height, 4);
        assert_eq!(fm.width, 4);
    }

    #[test]
    fn feature_map_wrong_size_errors() {
        let data = vec![0.0f32; 3 * 4 * 4 - 1];
        let r = FeatureMap::new(data, 3, 4, 4);
        assert!(matches!(r, Err(VisionError::DimensionMismatch { .. })));
    }

    #[test]
    fn feature_map_at_correct_value() {
        // channel 0 = 0.0, channel 1 = 1.0, etc.
        let mut data = vec![0.0f32; 2 * 3 * 3];
        for c in 0..2 {
            for pos in 0..9 {
                data[c * 9 + pos] = c as f32;
            }
        }
        let fm = FeatureMap::new(data, 2, 3, 3).expect("valid feature map");
        assert_eq!(fm.at(0, 1, 1), 0.0);
        assert_eq!(fm.at(1, 0, 0), 1.0);
    }

    // ── FpnConfig ──────────────────────────────────────────────────────────────

    #[test]
    fn fpn_config_valid() {
        let cfg = FpnConfig::new(vec![2048, 1024, 512, 256], 256).expect("valid config");
        assert_eq!(cfg.n_levels(), 4);
    }

    #[test]
    fn fpn_config_empty_in_channels_errors() {
        let r = FpnConfig::new(vec![], 256);
        assert!(r.is_err());
    }

    #[test]
    fn fpn_config_zero_out_channels_errors() {
        let r = FpnConfig::new(vec![512, 256], 0);
        assert!(r.is_err());
    }

    // ── upsample_nearest ──────────────────────────────────────────────────────

    #[test]
    fn upsample_nearest_doubles_size() {
        let data = vec![1.0, 2.0, 3.0, 4.0]; // 1 channel, 2×2
        let fm = FeatureMap::new(data, 1, 2, 2).expect("valid");
        let up = upsample_nearest(&fm, 4, 4);
        assert_eq!(up.height, 4);
        assert_eq!(up.width, 4);
        assert_eq!(up.channels, 1);
        assert_eq!(up.data.len(), 4 * 4);
    }

    #[test]
    fn upsample_nearest_values_replicated() {
        // 1-channel 2×2 map upsampled to 4×4: each pixel maps to a 2×2 block.
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let fm = FeatureMap::new(data, 1, 2, 2).expect("valid");
        let up = upsample_nearest(&fm, 4, 4);
        // top-left 2×2 should all be 1.0 (from src pixel [0,0])
        assert_eq!(up.at(0, 0, 0), 1.0);
        assert_eq!(up.at(0, 0, 1), 1.0);
        assert_eq!(up.at(0, 1, 0), 1.0);
        assert_eq!(up.at(0, 1, 1), 1.0);
        // bottom-right should be 4.0
        assert_eq!(up.at(0, 2, 2), 4.0);
        assert_eq!(up.at(0, 3, 3), 4.0);
    }

    #[test]
    fn upsample_nearest_identity_when_same_size() {
        let mut rng = make_rng();
        let fm = random_feature_map(&mut rng, 4, 5, 7);
        let up = upsample_nearest(&fm, 5, 7);
        for (a, b) in fm.data.iter().zip(up.data.iter()) {
            assert_eq!(*a, *b, "identity upsample should be exact copy");
        }
    }

    // ── FPN forward ───────────────────────────────────────────────────────────

    #[test]
    fn fpn_forward_output_channel_count_uniform() {
        let mut rng = make_rng();
        let cfg = FpnConfig::new(vec![64, 32], 16).expect("valid config");
        let fpn = Fpn::new(cfg, &mut rng).expect("valid FPN");
        let features = vec![
            random_feature_map(&mut rng, 64, 4, 4),
            random_feature_map(&mut rng, 32, 8, 8),
        ];
        let output = fpn.forward(features).expect("FPN forward ok");
        assert_eq!(output.len(), 2, "two output levels");
        for fm in &output {
            assert_eq!(fm.channels, 16, "all output levels should have 16 channels");
        }
    }

    #[test]
    fn fpn_forward_preserves_spatial_dims() {
        let mut rng = make_rng();
        let cfg = FpnConfig::new(vec![32, 16], 8).expect("valid config");
        let fpn = Fpn::new(cfg, &mut rng).expect("valid FPN");
        let features = vec![
            random_feature_map(&mut rng, 32, 3, 3),
            random_feature_map(&mut rng, 16, 6, 6),
        ];
        let output = fpn.forward(features).expect("FPN forward ok");
        assert_eq!(output[0].height, 3);
        assert_eq!(output[0].width, 3);
        assert_eq!(output[1].height, 6);
        assert_eq!(output[1].width, 6);
    }

    #[test]
    fn fpn_forward_three_levels() {
        let mut rng = make_rng();
        let cfg = FpnConfig::new(vec![64, 32, 16], 8).expect("valid config");
        let fpn = Fpn::new(cfg, &mut rng).expect("valid FPN");
        let features = vec![
            random_feature_map(&mut rng, 64, 2, 2),
            random_feature_map(&mut rng, 32, 4, 4),
            random_feature_map(&mut rng, 16, 8, 8),
        ];
        let output = fpn.forward(features).expect("FPN forward 3 levels ok");
        assert_eq!(output.len(), 3);
        for fm in &output {
            assert_eq!(fm.channels, 8);
            assert!(fm.data.iter().all(|v| v.is_finite()), "non-finite output");
        }
    }

    #[test]
    fn fpn_forward_wrong_level_count_errors() {
        let mut rng = make_rng();
        let cfg = FpnConfig::new(vec![64, 32], 16).expect("valid config");
        let fpn = Fpn::new(cfg, &mut rng).expect("valid FPN");
        // Only one feature map provided instead of two
        let features = vec![random_feature_map(&mut rng, 64, 4, 4)];
        let r = fpn.forward(features);
        assert!(
            matches!(
                r,
                Err(VisionError::DimensionMismatch {
                    expected: 2,
                    got: 1
                })
            ),
            "expected DimensionMismatch error"
        );
    }

    #[test]
    fn fpn_forward_empty_features_errors() {
        let mut rng = make_rng();
        let cfg = FpnConfig::new(vec![64, 32], 16).expect("valid config");
        let fpn = Fpn::new(cfg, &mut rng).expect("valid FPN");
        let r = fpn.forward(vec![]);
        assert!(r.is_err(), "expected error for empty features");
    }

    // ── conv3x3_same ──────────────────────────────────────────────────────────

    #[test]
    fn conv3x3_same_output_shape() {
        let feat = vec![0.5f32; 4 * 6 * 6];
        let weight = vec![0.0f32; 4 * 4 * 9];
        let bias = vec![1.0f32; 4]; // constant bias → all outputs = 1
        let out = conv3x3_same(&feat, 4, 6, 6, &weight, &bias, 4);
        assert_eq!(
            out.len(),
            4 * 6 * 6,
            "output size matches input spatial dims"
        );
        // With all-zero weights and bias=1, every output should be exactly 1.
        for v in &out {
            assert!((*v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }
}
