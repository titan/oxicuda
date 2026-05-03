//! CPU reference RoI Align implementation.
//!
//! Matches the semantics of the `roi_align_ptx` kernel: for each (RoI, channel,
//! pooled-row, pooled-col) output element, divide the RoI bin into a grid of
//! `sampling_ratio × sampling_ratio` sample points and average their bilinearly-
//! interpolated values from the feature map.

use crate::error::{VisionError, VisionResult};

// ─── Public API ───────────────────────────────────────────────────────────────

/// CPU reference RoI Align.
///
/// # Parameters
/// - `feat`:           flat `[channels, feat_h, feat_w]` CHW feature map.
/// - `feat_channels`:  number of channels in the feature map.
/// - `feat_h`, `feat_w`: spatial dimensions of the feature map.
/// - `rois`:           flat `[n_rois × 4]` in `[x1, y1, x2, y2]` feature-coordinate format.
/// - `n_rois`:         number of RoIs.
/// - `pooled_h`, `pooled_w`: output spatial dimensions.
/// - `sampling_ratio`: number of sample points per bin dimension (≥1).
///
/// # Returns
/// Flat `[n_rois × channels × pooled_h × pooled_w]`.
///
/// # Errors
/// - `InvalidRoiBox` if any RoI has `x2 ≤ x1` or `y2 ≤ y1`.
/// - `EmptyInput` if `feat_channels == 0`, `feat_h == 0`, or `feat_w == 0`.
/// - `DimensionMismatch` if `feat.len() != feat_channels * feat_h * feat_w`.
/// - `DimensionMismatch` if `rois.len() != n_rois * 4`.
/// - `DimensionMismatch` if `sampling_ratio == 0`.
pub fn roi_align(
    feat: &[f32],
    feat_channels: usize,
    feat_h: usize,
    feat_w: usize,
    rois: &[f32],
    n_rois: usize,
    pooled_h: usize,
    pooled_w: usize,
    sampling_ratio: usize,
) -> VisionResult<Vec<f32>> {
    // ── Validation ────────────────────────────────────────────────────────────
    if feat_channels == 0 || feat_h == 0 || feat_w == 0 {
        return Err(VisionError::EmptyInput("roi_align feature map"));
    }
    if pooled_h == 0 || pooled_w == 0 {
        return Err(VisionError::EmptyInput("roi_align pooled dims"));
    }
    if sampling_ratio == 0 {
        return Err(VisionError::DimensionMismatch {
            expected: 1,
            got: 0,
        });
    }

    let expected_feat = feat_channels * feat_h * feat_w;
    if feat.len() != expected_feat {
        return Err(VisionError::DimensionMismatch {
            expected: expected_feat,
            got: feat.len(),
        });
    }

    let expected_rois = n_rois * 4;
    if rois.len() != expected_rois {
        return Err(VisionError::DimensionMismatch {
            expected: expected_rois,
            got: rois.len(),
        });
    }

    // Validate all RoI coordinates.
    for r in 0..n_rois {
        let x1 = rois[r * 4];
        let y1 = rois[r * 4 + 1];
        let x2 = rois[r * 4 + 2];
        let y2 = rois[r * 4 + 3];
        if x2 <= x1 || y2 <= y1 {
            return Err(VisionError::InvalidRoiBox { x1, y1, x2, y2 });
        }
    }

    // ── Computation ───────────────────────────────────────────────────────────
    let out_size = n_rois * feat_channels * pooled_h * pooled_w;
    let mut out = vec![0.0f32; out_size];

    let sr = sampling_ratio as f32;
    let sr_inv = 1.0 / (sr * sr); // 1 / (sr² ) for averaging

    for r in 0..n_rois {
        let x1 = rois[r * 4];
        let y1 = rois[r * 4 + 1];
        let x2 = rois[r * 4 + 2];
        let y2 = rois[r * 4 + 3];

        let bin_h = (y2 - y1) / pooled_h as f32;
        let bin_w = (x2 - x1) / pooled_w as f32;

        // Step size within a bin for the sampling grid.
        let step_y = bin_h / sr;
        let step_x = bin_w / sr;

        for c in 0..feat_channels {
            for ph in 0..pooled_h {
                for pw in 0..pooled_w {
                    // Start of this bin in feature-map coordinates.
                    let y_start = y1 + ph as f32 * bin_h;
                    let x_start = x1 + pw as f32 * bin_w;

                    let mut sum = 0.0f32;
                    for sy in 0..sampling_ratio {
                        for sx in 0..sampling_ratio {
                            // Sample point centre within the bin.
                            let y = y_start + (sy as f32 + 0.5) * step_y;
                            let x = x_start + (sx as f32 + 0.5) * step_x;
                            sum += bilinear_sample_2d(feat, feat_channels, feat_h, feat_w, c, y, x);
                        }
                    }

                    let out_idx = r * feat_channels * pooled_h * pooled_w
                        + c * pooled_h * pooled_w
                        + ph * pooled_w
                        + pw;
                    out[out_idx] = sum * sr_inv;
                }
            }
        }
    }

    Ok(out)
}

// ─── Bilinear sampling ────────────────────────────────────────────────────────

/// Bilinearly sample a feature map at a sub-pixel location `(y, x)`.
///
/// Returns 0 for any sample falling outside `[0, feat_h) × [0, feat_w)`.
pub fn bilinear_sample_2d(
    feat: &[f32],
    _feat_channels: usize,
    feat_h: usize,
    feat_w: usize,
    channel: usize,
    y: f32,
    x: f32,
) -> f32 {
    // Out-of-bounds returns 0 (zero-padding).
    if y < 0.0 || y >= feat_h as f32 || x < 0.0 || x >= feat_w as f32 {
        return 0.0;
    }

    let y0 = y.floor() as usize;
    let x0 = x.floor() as usize;
    let y1 = (y0 + 1).min(feat_h - 1);
    let x1 = (x0 + 1).min(feat_w - 1);

    let fy = y - y0 as f32; // fractional part in y
    let fx = x - x0 as f32; // fractional part in x

    let spatial = feat_h * feat_w;
    let base = channel * spatial;

    // Four neighbours: (y0, x0), (y0, x1), (y1, x0), (y1, x1)
    let v00 = feat[base + y0 * feat_w + x0];
    let v01 = feat[base + y0 * feat_w + x1];
    let v10 = feat[base + y1 * feat_w + x0];
    let v11 = feat[base + y1 * feat_w + x1];

    // Bilinear interpolation.
    (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a single-channel feature map filled with a constant value.
    fn const_feat(channels: usize, h: usize, w: usize, val: f32) -> Vec<f32> {
        vec![val; channels * h * w]
    }

    // ── bilinear_sample_2d ────────────────────────────────────────────────────

    #[test]
    fn bilinear_exact_pixel_no_interpolation() {
        // Sampling at an integer coordinate should return that pixel exactly.
        let feat = vec![1.0, 2.0, 3.0, 4.0]; // 1ch, 2×2
        let v = bilinear_sample_2d(&feat, 1, 2, 2, 0, 0.0, 0.0);
        assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        let v = bilinear_sample_2d(&feat, 1, 2, 2, 0, 1.0, 1.0);
        assert!((v - 4.0).abs() < 1e-6, "expected 4.0, got {v}");
    }

    #[test]
    fn bilinear_centre_of_2x2_averages_all() {
        // At (0.5, 0.5) we should get (1+2+3+4)/4 = 2.5
        let feat = vec![1.0, 2.0, 3.0, 4.0];
        let v = bilinear_sample_2d(&feat, 1, 2, 2, 0, 0.5, 0.5);
        assert!((v - 2.5).abs() < 1e-5, "expected 2.5, got {v}");
    }

    #[test]
    fn bilinear_out_of_bounds_returns_zero() {
        let feat = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(bilinear_sample_2d(&feat, 1, 2, 2, 0, -0.1, 0.5), 0.0);
        assert_eq!(bilinear_sample_2d(&feat, 1, 2, 2, 0, 0.5, 2.0), 0.0);
        assert_eq!(bilinear_sample_2d(&feat, 1, 2, 2, 0, 2.0, 0.5), 0.0);
    }

    // ── roi_align output shape ────────────────────────────────────────────────

    #[test]
    fn roi_align_output_shape() {
        let feat = const_feat(3, 8, 8, 1.0);
        // One RoI covering the center of the feature map
        let rois = vec![1.0f32, 1.0, 7.0, 7.0];
        let out = roi_align(&feat, 3, 8, 8, &rois, 1, 4, 4, 2).expect("roi_align ok");
        assert_eq!(
            out.len(),
            3 * 4 * 4,
            "output shape [n_rois × channels × ph × pw]"
        );
    }

    #[test]
    fn roi_align_multiple_rois_shape() {
        let feat = const_feat(2, 16, 16, 0.5);
        let rois = vec![
            0.0f32, 0.0, 8.0, 8.0, // roi 0
            4.0, 4.0, 12.0, 12.0, // roi 1
            8.0, 8.0, 16.0, 16.0, // roi 2 (edge: x2=w, y2=h — valid since > x1, y1)
        ];
        let out = roi_align(&feat, 2, 16, 16, &rois, 3, 7, 7, 1).expect("roi_align ok");
        assert_eq!(out.len(), 3 * 2 * 7 * 7);
    }

    // ── Unit box should return (approximate) mean ─────────────────────────────

    #[test]
    fn roi_align_unit_box_constant_map_returns_constant() {
        // A constant feature map: every bilinear sample returns the constant.
        let val = std::f32::consts::PI;
        let feat = const_feat(1, 8, 8, val);
        // RoI covers the entire feature map.
        let rois = vec![0.0f32, 0.0, 8.0, 8.0];
        let out = roi_align(&feat, 1, 8, 8, &rois, 1, 1, 1, 2).expect("roi_align ok");
        // pooled_h=1, pooled_w=1 → single pooled element = average of 4 samples
        assert!(
            (out[0] - val).abs() < 1e-5,
            "expected {val}, got {}",
            out[0]
        );
    }

    #[test]
    fn roi_align_unit_box_mean_check() {
        // With pooled_h=pooled_w=1 and sampling_ratio=1, the result should be
        // the bilinear sample at the bin centre = centre of the entire RoI.
        // For a linearly increasing feature map, this should be the midpoint value.
        let feat: Vec<f32> = (0..64).map(|i| i as f32).collect(); // 1ch, 8×8, 0..63
        let rois = vec![0.0f32, 0.0, 8.0, 8.0];
        let out = roi_align(&feat, 1, 8, 8, &rois, 1, 1, 1, 1).expect("roi_align ok");
        // The single sample point is at (4.0, 4.0). Bilinear interpolation at
        // exact (4, 4) = feat[4*8+4] = 36.0. (Step = 8/1 = 8, centre = 0 + 0.5*8 = 4.)
        assert!(
            (out[0] - 36.0).abs() < 1e-4,
            "expected ~36.0, got {}",
            out[0]
        );
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn roi_align_invalid_roi_box_errors() {
        let feat = const_feat(1, 4, 4, 1.0);
        // x2 <= x1 (degenerate box)
        let rois = vec![3.0f32, 0.0, 1.0, 4.0];
        let r = roi_align(&feat, 1, 4, 4, &rois, 1, 2, 2, 1);
        assert!(
            matches!(r, Err(VisionError::InvalidRoiBox { .. })),
            "expected InvalidRoiBox error"
        );
    }

    #[test]
    fn roi_align_zero_height_roi_errors() {
        let feat = const_feat(1, 4, 4, 1.0);
        // y2 == y1 → degenerate
        let rois = vec![0.0f32, 2.0, 4.0, 2.0];
        let r = roi_align(&feat, 1, 4, 4, &rois, 1, 2, 2, 1);
        assert!(r.is_err(), "expected error for y1==y2");
    }

    #[test]
    fn roi_align_empty_feature_errors() {
        let feat: Vec<f32> = vec![];
        let rois = vec![0.0f32, 0.0, 4.0, 4.0];
        let r = roi_align(&feat, 0, 4, 4, &rois, 1, 2, 2, 1);
        assert!(r.is_err(), "expected error for channels=0");
    }

    #[test]
    fn roi_align_zero_sampling_ratio_errors() {
        let feat = const_feat(1, 4, 4, 1.0);
        let rois = vec![0.0f32, 0.0, 4.0, 4.0];
        let r = roi_align(&feat, 1, 4, 4, &rois, 1, 2, 2, 0);
        assert!(r.is_err(), "expected error for sampling_ratio=0");
    }

    #[test]
    fn roi_align_wrong_feat_size_errors() {
        let feat = vec![0.0f32; 4 * 4 - 1]; // one element short
        let rois = vec![0.0f32, 0.0, 4.0, 4.0];
        let r = roi_align(&feat, 1, 4, 4, &rois, 1, 2, 2, 1);
        assert!(
            matches!(r, Err(VisionError::DimensionMismatch { .. })),
            "expected DimensionMismatch"
        );
    }
}
