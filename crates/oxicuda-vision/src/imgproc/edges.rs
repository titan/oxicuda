//! Classical edge detection: Sobel gradients and the Canny edge detector.
//!
//! These operators consume a **single-channel** (grayscale) `[h × w]` row-major
//! `f32` image and produce gradient / edge maps of the same spatial size. RGB
//! callers should first reduce to luminance (e.g. via the photometric
//! grayscale helper).
//!
//! * [`sobel_gradients`] — horizontal (`Gx`) and vertical (`Gy`) Sobel
//!   responses plus the gradient magnitude `√(Gx²+Gy²)`, with replicate
//!   (clamp-to-edge) border handling.
//! * [`canny`] — the full Canny (1986) pipeline: Sobel gradients → non-maximum
//!   suppression along the quantised gradient direction → hysteresis double
//!   thresholding, yielding a binary edge map (`1.0` edge, `0.0` non-edge).

use crate::error::{VisionError, VisionResult};

/// Sobel horizontal kernel `Gx` (detects vertical edges), row-major 3×3.
pub const SOBEL_GX: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
/// Sobel vertical kernel `Gy` (detects horizontal edges), row-major 3×3.
pub const SOBEL_GY: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

/// Result of a Sobel pass: per-pixel gradient components and magnitude.
#[derive(Debug, Clone)]
pub struct SobelOutput {
    /// Horizontal gradient `Gx`, shape `[h × w]`.
    pub gx: Vec<f32>,
    /// Vertical gradient `Gy`, shape `[h × w]`.
    pub gy: Vec<f32>,
    /// Gradient magnitude `√(Gx² + Gy²)`, shape `[h × w]`.
    pub magnitude: Vec<f32>,
}

#[inline]
fn validate_gray(img: &[f32], h: usize, w: usize) -> VisionResult<()> {
    if h == 0 || w == 0 {
        return Err(VisionError::InvalidImageSize {
            height: h,
            width: w,
            channels: 1,
        });
    }
    if img.len() != h * w {
        return Err(VisionError::DimensionMismatch {
            expected: h * w,
            got: img.len(),
        });
    }
    Ok(())
}

/// Clamp a signed coordinate into `[0, n)` (replicate / clamp-to-edge border).
#[inline]
fn clamp_idx(v: isize, n: usize) -> usize {
    if v < 0 {
        0
    } else if v as usize >= n {
        n - 1
    } else {
        v as usize
    }
}

/// Apply a 3×3 kernel at `(y, x)` with replicate border handling.
#[inline]
fn conv3x3_at(img: &[f32], h: usize, w: usize, y: usize, x: usize, kernel: &[f32; 9]) -> f32 {
    let mut acc = 0.0_f32;
    for ky in 0..3 {
        let sy = clamp_idx(y as isize + ky as isize - 1, h);
        for kx in 0..3 {
            let sx = clamp_idx(x as isize + kx as isize - 1, w);
            acc += kernel[ky * 3 + kx] * img[sy * w + sx];
        }
    }
    acc
}

/// Compute Sobel `Gx`, `Gy`, and gradient magnitude for a grayscale image.
///
/// # Errors
/// Returns [`VisionError::InvalidImageSize`] / [`VisionError::DimensionMismatch`]
/// on shape problems.
pub fn sobel_gradients(img: &[f32], h: usize, w: usize) -> VisionResult<SobelOutput> {
    validate_gray(img, h, w)?;
    let mut gx = vec![0.0_f32; h * w];
    let mut gy = vec![0.0_f32; h * w];
    let mut magnitude = vec![0.0_f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let dx = conv3x3_at(img, h, w, y, x, &SOBEL_GX);
            let dy = conv3x3_at(img, h, w, y, x, &SOBEL_GY);
            let idx = y * w + x;
            gx[idx] = dx;
            gy[idx] = dy;
            magnitude[idx] = (dx * dx + dy * dy).sqrt();
        }
    }
    Ok(SobelOutput { gx, gy, magnitude })
}

/// Quantise a gradient orientation `atan2(gy, gx)` to one of four directions
/// `{0, 1, 2, 3}` corresponding to `{0°, 45°, 90°, 135°}` for non-maximum
/// suppression neighbour selection.
#[inline]
fn quantize_direction(gx: f32, gy: f32) -> u8 {
    // Map angle into [0, 180) degrees.
    let mut angle = gy.atan2(gx).to_degrees();
    if angle < 0.0 {
        angle += 180.0;
    }
    if !(22.5..157.5).contains(&angle) {
        0 // 0° — horizontal gradient, compare left/right
    } else if (22.5..67.5).contains(&angle) {
        1 // 45°
    } else if (67.5..112.5).contains(&angle) {
        2 // 90° — vertical gradient, compare up/down
    } else {
        3 // 135°
    }
}

/// Canny edge detector returning a binary `[h × w]` edge map (`1.0` / `0.0`).
///
/// Pipeline: Sobel gradients → non-maximum suppression along the quantised
/// gradient direction → hysteresis thresholding with `low` / `high`.
/// Pixels whose magnitude exceeds `high` are *strong* edges; pixels above
/// `low` are *weak* edges that are promoted to edges only if connected
/// (8-neighbourhood, transitively) to a strong edge.
///
/// # Errors
/// Returns shape errors via [`sobel_gradients`]; returns
/// [`VisionError::Internal`] if `low > high` or either threshold is negative /
/// non-finite.
pub fn canny(img: &[f32], h: usize, w: usize, low: f32, high: f32) -> VisionResult<Vec<f32>> {
    if !low.is_finite() || !high.is_finite() || low < 0.0 || high < 0.0 || low > high {
        return Err(VisionError::Internal(format!(
            "canny thresholds invalid: low={low}, high={high} (need 0 ≤ low ≤ high)"
        )));
    }
    let sob = sobel_gradients(img, h, w)?;

    // ── Non-maximum suppression ──────────────────────────────────────────────
    let mut nms = vec![0.0_f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mag = sob.magnitude[idx];
            if mag <= 0.0 {
                continue;
            }
            let dir = quantize_direction(sob.gx[idx], sob.gy[idx]);
            // Neighbour offsets along the gradient direction.
            let (dy, dx) = match dir {
                0 => (0isize, 1isize), // horizontal
                1 => (-1, 1),          // 45°
                2 => (1, 0),           // vertical
                _ => (-1, -1),         // 135°
            };
            let y0 = clamp_idx(y as isize - dy, h);
            let x0 = clamp_idx(x as isize - dx, w);
            let y1 = clamp_idx(y as isize + dy, h);
            let x1 = clamp_idx(x as isize + dx, w);
            let m0 = sob.magnitude[y0 * w + x0];
            let m1 = sob.magnitude[y1 * w + x1];
            if mag >= m0 && mag >= m1 {
                nms[idx] = mag;
            }
        }
    }

    // ── Hysteresis double thresholding ──────────────────────────────────────
    // 0 = none, 1 = weak, 2 = strong.
    let mut label = vec![0u8; h * w];
    let mut stack: Vec<usize> = Vec::new();
    for (idx, &m) in nms.iter().enumerate() {
        if m >= high {
            label[idx] = 2;
            stack.push(idx);
        } else if m >= low {
            label[idx] = 1;
        }
    }

    // Promote weak pixels connected (8-neighbour) to strong ones.
    let mut edge = vec![0.0_f32; h * w];
    while let Some(idx) = stack.pop() {
        edge[idx] = 1.0;
        let y = idx / w;
        let x = idx % w;
        for dyi in -1isize..=1 {
            for dxi in -1isize..=1 {
                if dyi == 0 && dxi == 0 {
                    continue;
                }
                let ny = y as isize + dyi;
                let nx = x as isize + dxi;
                if ny < 0 || nx < 0 || ny as usize >= h || nx as usize >= w {
                    continue;
                }
                let nidx = ny as usize * w + nx as usize;
                if label[nidx] == 1 {
                    label[nidx] = 2; // mark visited as strong
                    stack.push(nidx);
                }
            }
        }
    }

    Ok(edge)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// A vertical step edge: left half 0, right half 1.
    fn vertical_step(h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; h * w];
        for y in 0..h {
            for x in 0..w {
                if x >= w / 2 {
                    img[y * w + x] = 1.0;
                }
            }
        }
        img
    }

    /// A horizontal step edge: top half 0, bottom half 1.
    fn horizontal_step(h: usize, w: usize) -> Vec<f32> {
        let mut img = vec![0.0_f32; h * w];
        for y in 0..h {
            for x in 0..w {
                if y >= h / 2 {
                    img[y * w + x] = 1.0;
                }
            }
        }
        img
    }

    #[test]
    fn sobel_output_shapes() {
        let img = vertical_step(8, 8);
        let out = sobel_gradients(&img, 8, 8).expect("ok");
        assert_eq!(out.gx.len(), 64);
        assert_eq!(out.gy.len(), 64);
        assert_eq!(out.magnitude.len(), 64);
    }

    #[test]
    fn sobel_constant_image_zero_gradient() {
        let img = vec![0.4_f32; 8 * 8];
        let out = sobel_gradients(&img, 8, 8).expect("ok");
        assert!(
            out.magnitude.iter().all(|&m| m.abs() < 1e-6),
            "constant image must have zero gradient"
        );
    }

    #[test]
    fn sobel_vertical_edge_detected_by_gx() {
        let img = vertical_step(8, 8);
        let out = sobel_gradients(&img, 8, 8).expect("ok");
        // Column 3/4 boundary: Gx should be large near x = w/2.
        let y = 4;
        let mag_at_edge = out.magnitude[y * 8 + 3].max(out.magnitude[y * 8 + 4]);
        let mag_flat = out.magnitude[y * 8];
        assert!(
            mag_at_edge > mag_flat,
            "edge magnitude {mag_at_edge} should exceed flat {mag_flat}"
        );
    }

    #[test]
    fn sobel_horizontal_edge_strong_gy() {
        let img = horizontal_step(8, 8);
        let out = sobel_gradients(&img, 8, 8).expect("ok");
        let x = 4;
        // Across the horizontal boundary Gy dominates.
        let gy_edge = out.gy[3 * 8 + x].abs().max(out.gy[4 * 8 + x].abs());
        assert!(
            gy_edge > 1.0,
            "Gy at horizontal edge should be large: {gy_edge}"
        );
    }

    #[test]
    fn sobel_magnitude_nonnegative() {
        let mut img = vec![0.0_f32; 6 * 6];
        for (i, v) in img.iter_mut().enumerate() {
            *v = (i % 5) as f32 * 0.1;
        }
        let out = sobel_gradients(&img, 6, 6).expect("ok");
        assert!(out.magnitude.iter().all(|&m| m >= 0.0));
    }

    #[test]
    fn sobel_wrong_size_errors() {
        let img = vec![0.0_f32; 10];
        assert!(matches!(
            sobel_gradients(&img, 8, 8),
            Err(VisionError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn sobel_zero_dim_errors() {
        let img: Vec<f32> = vec![];
        assert!(matches!(
            sobel_gradients(&img, 0, 8),
            Err(VisionError::InvalidImageSize { .. })
        ));
    }

    #[test]
    fn quantize_direction_cardinal() {
        // Pure horizontal gradient → direction 0.
        assert_eq!(quantize_direction(1.0, 0.0), 0);
        // Pure vertical gradient → direction 2 (90°).
        assert_eq!(quantize_direction(0.0, 1.0), 2);
    }

    #[test]
    fn canny_binary_output() {
        let img = vertical_step(16, 16);
        let edges = canny(&img, 16, 16, 0.5, 1.5).expect("ok");
        assert_eq!(edges.len(), 256);
        assert!(
            edges.iter().all(|&e| e == 0.0 || e == 1.0),
            "canny output must be binary"
        );
    }

    #[test]
    fn canny_detects_step_edge() {
        let img = vertical_step(16, 16);
        let edges = canny(&img, 16, 16, 0.5, 1.5).expect("ok");
        let total: f32 = edges.iter().sum();
        assert!(total > 0.0, "canny should find edges on a step image");
    }

    #[test]
    fn canny_constant_image_no_edges() {
        let img = vec![0.7_f32; 16 * 16];
        let edges = canny(&img, 16, 16, 0.5, 1.5).expect("ok");
        let total: f32 = edges.iter().sum();
        assert!(
            total == 0.0,
            "constant image must yield no edges, got {total}"
        );
    }

    #[test]
    fn canny_hysteresis_promotes_weak() {
        // A ramp produces moderate gradients; with a low/high straddling the
        // gradient, weak pixels connected to strong ones become edges.
        let mut img = vec![0.0_f32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                img[y * 16 + x] = x as f32 / 15.0;
            }
        }
        let edges = canny(&img, 16, 16, 0.05, 0.2).expect("ok");
        assert!(edges.contains(&1.0));
    }

    #[test]
    fn canny_invalid_thresholds_error() {
        let img = vertical_step(8, 8);
        assert!(matches!(
            canny(&img, 8, 8, 2.0, 1.0), // low > high
            Err(VisionError::Internal(_))
        ));
        assert!(matches!(
            canny(&img, 8, 8, -1.0, 1.0),
            Err(VisionError::Internal(_))
        ));
    }

    #[test]
    fn canny_higher_threshold_fewer_edges() {
        // Diagonal gradient image.
        let mut img = vec![0.0_f32; 24 * 24];
        for y in 0..24 {
            for x in 0..24 {
                img[y * 24 + x] = ((x + y) as f32 / 46.0).min(1.0);
            }
        }
        let lo = canny(&img, 24, 24, 0.02, 0.05).expect("ok");
        let hi = canny(&img, 24, 24, 0.5, 1.0).expect("ok");
        let n_lo: f32 = lo.iter().sum();
        let n_hi: f32 = hi.iter().sum();
        assert!(
            n_hi <= n_lo,
            "higher thresholds should not produce more edges: {n_hi} vs {n_lo}"
        );
    }

    #[test]
    fn canny_wrong_size_errors() {
        let img = vec![0.0_f32; 5];
        assert!(canny(&img, 8, 8, 0.5, 1.5).is_err());
    }
}
