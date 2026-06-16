//! 2D Gaussian splatting — CPU forward rasterizer.
//!
//! Reference: Huang et al., *"2D Gaussian Splatting for Geometrically Accurate
//! Radiance Fields"*, SIGGRAPH 2024 — the image-plane forward compositing core.
//!
//! Each splat is an anisotropic 2D Gaussian on the image plane with a center
//! `(cx, cy)` in pixel coordinates, a 2×2 symmetric positive-definite (SPD)
//! covariance `Σ`, an RGB color, an opacity `α`, and a depth used for sorting.
//!
//! The image is produced by front-to-back alpha compositing. For each pixel
//! `p` and each splat (sorted by ascending depth):
//!
//! ```text
//! w  = clamp( α · exp(-0.5 · (p - μ)^T Σ^{-1} (p - μ)),  0, 1 )
//! C += T · w · color
//! T *= (1 - w)
//! ```
//!
//! where `T` is the running transmittance (initialised to 1).
//!
//! # Output layout
//!
//! [`rasterize_gaussians_2d`] returns an interleaved, channels-last RGB image
//! of length `width · height · 3`, indexed as `image[(y · width + x) · 3 + c]`.

use crate::error::{Geom3dError, Geom3dResult};

/// A single 2D Gaussian splat.
#[derive(Debug, Clone, Copy)]
pub struct Gaussian2d {
    /// Center `(cx, cy)` in pixel coordinates.
    pub center: [f32; 2],
    /// Symmetric positive-definite 2×2 covariance, row-major `[a, b, c, d]`
    /// representing `[[a, b], [c, d]]` (so `b == c`).
    pub cov: [f32; 4],
    /// RGB color, each channel typically in `[0, 1]`.
    pub color: [f32; 3],
    /// Opacity `α`, typically in `[0, 1]`.
    pub opacity: f32,
    /// Depth used for front-to-back ordering (smaller = nearer the camera).
    pub depth: f32,
}

impl Gaussian2d {
    /// Build a splat from an anisotropic parameterisation `(σ_x, σ_y, θ)`.
    ///
    /// The covariance is `Σ = R(θ) · diag(σ_x², σ_y²) · R(θ)^T`, where `R(θ)`
    /// is the 2D rotation by angle `θ` (radians).
    #[must_use]
    pub fn from_params(
        center: [f32; 2],
        sigma_x: f32,
        sigma_y: f32,
        theta: f32,
        color: [f32; 3],
        opacity: f32,
        depth: f32,
    ) -> Self {
        let (s, c) = theta.sin_cos();
        let vx = sigma_x * sigma_x;
        let vy = sigma_y * sigma_y;
        let a = c * c * vx + s * s * vy;
        let b = c * s * (vx - vy);
        let d = s * s * vx + c * c * vy;
        Self {
            center,
            cov: [a, b, b, d],
            color,
            opacity,
            depth,
        }
    }

    /// Build an isotropic splat with standard deviation `sigma`.
    #[must_use]
    pub fn isotropic(
        center: [f32; 2],
        sigma: f32,
        color: [f32; 3],
        opacity: f32,
        depth: f32,
    ) -> Self {
        let v = sigma * sigma;
        Self {
            center,
            cov: [v, 0.0, 0.0, v],
            color,
            opacity,
            depth,
        }
    }
}

/// Validate that a covariance is symmetric positive-definite.
fn validate_spd(cov: &[f32; 4]) -> Geom3dResult<()> {
    let (a, b, c, d) = (cov[0], cov[1], cov[2], cov[3]);
    if !(a.is_finite() && b.is_finite() && c.is_finite() && d.is_finite()) {
        return Err(Geom3dError::InvalidCovariance {
            reason: "covariance contains non-finite entries",
        });
    }
    if (b - c).abs() > 1e-5 * (1.0 + b.abs() + c.abs()) {
        return Err(Geom3dError::InvalidCovariance {
            reason: "covariance matrix is not symmetric",
        });
    }
    let det = a * d - b * c;
    if a <= 0.0 || det <= 0.0 {
        return Err(Geom3dError::InvalidCovariance {
            reason: "covariance matrix is not positive-definite",
        });
    }
    Ok(())
}

/// Invert a 2×2 matrix `[a, b, c, d]`; returns `None` if (near-)singular.
fn invert2x2(cov: &[f32; 4]) -> Option<[f32; 4]> {
    let det = cov[0] * cov[3] - cov[1] * cov[2];
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        cov[3] * inv_det,
        -cov[1] * inv_det,
        -cov[2] * inv_det,
        cov[0] * inv_det,
    ])
}

/// Rasterize a set of 2D Gaussian splats into an RGB image.
///
/// Splats are composited front-to-back (ascending `depth`). Returns the image
/// as an interleaved channels-last `Vec<f32>` of length `width · height · 3`.
///
/// # Errors
///
/// Returns [`Geom3dError::InvalidCovariance`] if any splat's covariance is not
/// symmetric positive-definite.
pub fn rasterize_gaussians_2d(
    splats: &[Gaussian2d],
    width: usize,
    height: usize,
) -> Geom3dResult<Vec<f32>> {
    // Validate every covariance up front (even off-screen splats).
    for g in splats {
        validate_spd(&g.cov)?;
    }

    let mut image = vec![0.0_f32; width * height * 3];
    if width == 0 || height == 0 {
        return Ok(image);
    }
    let mut transmittance = vec![1.0_f32; width * height];

    // Front-to-back order: nearest (smallest depth) first.
    let mut order: Vec<usize> = (0..splats.len()).collect();
    order.sort_by(|&i, &j| {
        splats[i]
            .depth
            .partial_cmp(&splats[j].depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &gi in &order {
        let g = &splats[gi];
        let inv = match invert2x2(&g.cov) {
            Some(m) => m,
            None => continue,
        };

        let cx = g.center[0];
        let cy = g.center[1];

        // Axis-aligned 3σ bounding box from the marginal standard deviations.
        let rx = 3.0 * g.cov[0].max(0.0).sqrt();
        let ry = 3.0 * g.cov[3].max(0.0).sqrt();
        let x0 = ((cx - rx).floor() as i32).max(0) as usize;
        let y0 = ((cy - ry).floor() as i32).max(0) as usize;
        let x1 = (((cx + rx).ceil() as i32).max(0) as usize).min(width);
        let y1 = (((cy + ry).ceil() as i32).max(0) as usize).min(height);

        for py in y0..y1 {
            let dy = py as f32 + 0.5 - cy;
            for px in x0..x1 {
                let dx = px as f32 + 0.5 - cx;
                // Mahalanobis distance d^T Σ^{-1} d (inv[1] == inv[2]).
                let mah = inv[0] * dx * dx + (inv[1] + inv[2]) * dx * dy + inv[3] * dy * dy;
                let weight = (g.opacity * (-0.5 * mah).exp()).clamp(0.0, 1.0);

                let pix = py * width + px;
                let trans = transmittance[pix];
                if trans <= 1e-6 {
                    continue;
                }
                let contrib = trans * weight;
                let base = pix * 3;
                image[base] += contrib * g.color[0];
                image[base + 1] += contrib * g.color[1];
                image[base + 2] += contrib * g.color[2];
                transmittance[pix] = trans * (1.0 - weight);
            }
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rasterize_output_dims() {
        let img =
            rasterize_gaussians_2d(&[], 16, 12).expect("rasterize_gaussians_2d should succeed");
        assert_eq!(img.len(), 16 * 12 * 3);
        assert!(img.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn single_isotropic_peaks_at_center() {
        // Center on the middle of pixel (7, 7).
        let g = Gaussian2d::isotropic([7.5, 7.5], 2.0, [1.0, 1.0, 1.0], 0.9, 0.0);
        let (w, h) = (16usize, 16usize);
        let img =
            rasterize_gaussians_2d(&[g], w, h).expect("rasterize_gaussians_2d should succeed");

        let intensity = |px: usize, py: usize| -> f32 {
            let base = (py * w + px) * 3;
            img[base] + img[base + 1] + img[base + 2]
        };

        let center = intensity(7, 7);
        // Peak at the center pixel.
        assert!(center > intensity(7, 10));
        assert!(center > intensity(2, 2));
        assert!(intensity(0, 0) < intensity(5, 5));
        // Radial monotonic decrease moving away along +x.
        let mut prev = center;
        for px in 8..14 {
            let cur = intensity(px, 7);
            assert!(cur <= prev + 1e-6, "intensity must not increase outward");
            prev = cur;
        }
        assert!(img.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn two_splats_front_to_back_order() {
        // Near red splat (depth 1) should dominate over far blue splat (depth 5).
        let near = Gaussian2d::isotropic([7.5, 7.5], 3.0, [1.0, 0.0, 0.0], 0.95, 1.0);
        let far = Gaussian2d::isotropic([7.5, 7.5], 3.0, [0.0, 0.0, 1.0], 0.95, 5.0);
        let (w, h) = (16usize, 16usize);
        let img = rasterize_gaussians_2d(&[far, near], w, h)
            .expect("rasterize_gaussians_2d should succeed");

        let base = (7 * w + 7) * 3;
        let red = img[base];
        let blue = img[base + 2];
        assert!(
            red > blue,
            "nearer splat must dominate: red={red}, blue={blue}"
        );
        assert!(red > 0.5, "near opaque splat should contribute strongly");
    }

    #[test]
    fn ordering_independent_of_input_order() {
        let near = Gaussian2d::isotropic([7.5, 7.5], 3.0, [1.0, 0.0, 0.0], 0.95, 1.0);
        let far = Gaussian2d::isotropic([7.5, 7.5], 3.0, [0.0, 0.0, 1.0], 0.95, 5.0);
        let a = rasterize_gaussians_2d(&[far, near], 16, 16)
            .expect("rasterize_gaussians_2d should succeed");
        let b = rasterize_gaussians_2d(&[near, far], 16, 16)
            .expect("rasterize_gaussians_2d should succeed");
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_opacity_contributes_nothing() {
        let g = Gaussian2d::isotropic([7.5, 7.5], 2.0, [1.0, 1.0, 1.0], 0.0, 0.0);
        let img =
            rasterize_gaussians_2d(&[g], 16, 16).expect("rasterize_gaussians_2d should succeed");
        let total: f32 = img.iter().sum();
        assert!(total.abs() < 1e-6, "opacity 0 must contribute nothing");
    }

    #[test]
    fn non_spd_covariance_rejected() {
        // det = 1 - 4 = -3 < 0  →  not positive-definite.
        let g = Gaussian2d {
            center: [4.0, 4.0],
            cov: [1.0, 2.0, 2.0, 1.0],
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
            depth: 0.0,
        };
        let result = rasterize_gaussians_2d(&[g], 8, 8);
        assert!(matches!(result, Err(Geom3dError::InvalidCovariance { .. })));
    }

    #[test]
    fn non_symmetric_covariance_rejected() {
        let g = Gaussian2d {
            center: [4.0, 4.0],
            cov: [4.0, 1.0, -1.0, 4.0],
            color: [1.0, 1.0, 1.0],
            opacity: 1.0,
            depth: 0.0,
        };
        assert!(rasterize_gaussians_2d(&[g], 8, 8).is_err());
    }

    #[test]
    fn from_params_isotropic_matches() {
        let g = Gaussian2d::from_params([0.0, 0.0], 2.0, 2.0, 0.7, [0.0; 3], 1.0, 0.0);
        // Isotropic covariance is rotation-invariant: Σ = diag(4, 4).
        assert!((g.cov[0] - 4.0).abs() < 1e-4);
        assert!((g.cov[3] - 4.0).abs() < 1e-4);
        assert!(g.cov[1].abs() < 1e-4);
        assert!(g.cov[2].abs() < 1e-4);
        validate_spd(&g.cov).expect("validate_spd should succeed");
    }

    #[test]
    fn from_params_anisotropic_is_spd() {
        let g = Gaussian2d::from_params([0.0, 0.0], 3.0, 1.0, 0.9, [0.0; 3], 1.0, 0.0);
        // Off-diagonal terms appear for a rotated anisotropic Gaussian.
        assert!(g.cov[1].abs() > 1e-3);
        assert!((g.cov[1] - g.cov[2]).abs() < 1e-6);
        validate_spd(&g.cov).expect("validate_spd should succeed");
    }
}
