//! Mip-Splatting anti-aliasing filters.
//!
//! Reference: Yu, Chen, Antic, Wang, Niemeyer, Bhattacharyya, Geiger,
//! *"Mip-Splatting: Alias-free 3D Gaussian Splatting"*, CVPR 2024.
//!
//! Standard 3DGS adds a fixed screen-space dilation (`Σ_2d += s·I`) to keep
//! splats at least one pixel wide. That dilation is *not* energy-preserving and
//! introduces erosion/dilation aliasing under resolution or focal-length
//! changes. Mip-Splatting replaces it with two physically-motivated filters:
//!
//! 1. **3D smoothing filter** — convolve each 3D Gaussian with an isotropic
//!    low-pass kernel whose bandwidth is set by the *maximal sampling rate* at
//!    which the primitive is observed across all training views. Concretely the
//!    3D covariance is replaced by `Σ_3d + s²·I`, where `s = τ / f̂` is the
//!    world-space footprint of one pixel at the closest observed depth
//!    (`f̂` = focal length, depth folded into `τ`). The total opacity is
//!    renormalised so energy is conserved:
//!    `α' = α · sqrt( |Σ_3d| / |Σ_3d + s²I| )`.
//!
//! 2. **2D Mip filter** — replaces the screen-space dilation with a 2D
//!    Gaussian low-pass of variance `σ²_mip` (one pixel). Adding it to the
//!    projected covariance and renormalising the opacity by the determinant
//!    ratio yields an *area-correct* (alias-free) footprint:
//!    `Σ'_2d = Σ_2d + σ²_mip·I`,
//!    `α' = α · sqrt( |Σ_2d| / |Σ_2d + σ²_mip I| )`.
//!
//! Both filters are exposed as pure functions over covariance/opacity so they
//! can be composed with [`crate::gaussian::project::project_gaussian`] and any
//! rasterizer.

use crate::error::{Geom3dError, Geom3dResult};

/// Parameters for the Mip-Splatting filters.
#[derive(Debug, Clone)]
pub struct MipSplatConfig {
    /// World-space standard deviation `s` of the 3D smoothing kernel (the
    /// footprint of one pixel at the closest observed sampling rate). Must be
    /// `>= 0`.
    pub world_filter_sigma: f32,
    /// Screen-space standard deviation `σ_mip` of the 2D Mip filter, in pixels
    /// (typically `~1`). Must be `>= 0`.
    pub screen_filter_sigma: f32,
}

impl Default for MipSplatConfig {
    fn default() -> Self {
        Self {
            world_filter_sigma: 0.0,
            screen_filter_sigma: 1.0,
        }
    }
}

impl MipSplatConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Geom3dError::InvalidCovariance`] if either filter standard
    /// deviation is negative or non-finite.
    pub fn validate(&self) -> Geom3dResult<()> {
        if !(self.world_filter_sigma.is_finite() && self.world_filter_sigma >= 0.0) {
            return Err(Geom3dError::InvalidCovariance {
                reason: "world_filter_sigma must be finite and non-negative",
            });
        }
        if !(self.screen_filter_sigma.is_finite() && self.screen_filter_sigma >= 0.0) {
            return Err(Geom3dError::InvalidCovariance {
                reason: "screen_filter_sigma must be finite and non-negative",
            });
        }
        Ok(())
    }
}

/// Determinant of a row-major 3×3 matrix.
fn det3(m: &[f32; 9]) -> f32 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Determinant of a row-major 2×2 matrix `[a, b, c, d]`.
fn det2(m: &[f32; 4]) -> f32 {
    m[0] * m[3] - m[1] * m[2]
}

/// Apply the 3D smoothing filter to a 3D covariance and opacity.
///
/// Returns `(Σ_3d + s²·I, α · sqrt(|Σ_3d| / |Σ_3d + s²I|))`. The opacity
/// renormalisation conserves the integral of the (un-normalised) Gaussian so
/// total radiance is preserved under the low-pass.
///
/// `cov3d` is row-major 3×3; `opacity` is the linear (post-sigmoid) opacity.
///
/// # Errors
///
/// Returns [`Geom3dError::InvalidCovariance`] if the input covariance is
/// singular (non-positive determinant).
pub fn apply_3d_smoothing(
    cov3d: &[f32; 9],
    opacity: f32,
    config: &MipSplatConfig,
) -> Geom3dResult<([f32; 9], f32)> {
    config.validate()?;
    let det_in = det3(cov3d);
    if det_in <= 0.0 {
        return Err(Geom3dError::InvalidCovariance {
            reason: "3D covariance must be positive-definite (det > 0)",
        });
    }
    let s2 = config.world_filter_sigma * config.world_filter_sigma;
    let mut out = *cov3d;
    out[0] += s2;
    out[4] += s2;
    out[8] += s2;
    let det_out = det3(&out);
    let ratio = (det_in / det_out).max(0.0).sqrt();
    Ok((out, opacity * ratio))
}

/// Apply the 2D Mip filter to a projected 2D covariance and opacity.
///
/// Returns `(Σ_2d + σ²_mip·I, α · sqrt(|Σ_2d| / |Σ_2d + σ²_mip I|))`. Unlike
/// the constant `+0.3·I` dilation used by vanilla 3DGS, the determinant-ratio
/// opacity correction makes the footprint area-preserving and therefore
/// alias-free under resolution changes.
///
/// `cov2d` is row-major 2×2; `opacity` is the linear opacity.
///
/// # Errors
///
/// Returns [`Geom3dError::InvalidCovariance`] if the input 2D covariance is
/// singular.
pub fn apply_2d_mip_filter(
    cov2d: &[f32; 4],
    opacity: f32,
    config: &MipSplatConfig,
) -> Geom3dResult<([f32; 4], f32)> {
    config.validate()?;
    let det_in = det2(cov2d);
    if det_in <= 0.0 {
        return Err(Geom3dError::InvalidCovariance {
            reason: "2D covariance must be positive-definite (det > 0)",
        });
    }
    let s2 = config.screen_filter_sigma * config.screen_filter_sigma;
    let out = [cov2d[0] + s2, cov2d[1], cov2d[2], cov2d[3] + s2];
    let det_out = det2(&out);
    let ratio = (det_in / det_out).max(0.0).sqrt();
    Ok((out, opacity * ratio))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_world_sigma_is_identity() {
        let cfg = MipSplatConfig {
            world_filter_sigma: 0.0,
            screen_filter_sigma: 1.0,
        };
        // Symmetric positive-definite row-major 3×3.
        let cov = [2.0, 0.3, 0.0, 0.3, 1.5, 0.0, 0.0, 0.0, 3.0];
        let (out, op) = apply_3d_smoothing(&cov, 0.7, &cfg).expect("smoothing should succeed");
        for (a, b) in out.iter().zip(cov.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((op - 0.7).abs() < 1e-6, "opacity unchanged when s=0");
    }

    #[test]
    fn zero_screen_sigma_is_identity() {
        let cfg = MipSplatConfig {
            world_filter_sigma: 0.0,
            screen_filter_sigma: 0.0,
        };
        let cov = [4.0, 0.5, 0.5, 2.0];
        let (out, op) = apply_2d_mip_filter(&cov, 0.9, &cfg).expect("mip should succeed");
        assert_eq!(out, cov);
        assert!((op - 0.9).abs() < 1e-6);
    }

    #[test]
    fn three_d_filter_grows_covariance_shrinks_opacity() {
        let cfg = MipSplatConfig {
            world_filter_sigma: 0.5,
            screen_filter_sigma: 1.0,
        };
        let cov = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (out, op) = apply_3d_smoothing(&cov, 1.0, &cfg).expect("smoothing should succeed");
        // Isotropic identity → each diagonal grows by s² = 0.25.
        assert!((out[0] - 1.25).abs() < 1e-5);
        assert!((out[4] - 1.25).abs() < 1e-5);
        assert!((out[8] - 1.25).abs() < 1e-5);
        // Energy-preserving opacity: α' = sqrt(1 / 1.25³).
        let expected = (1.0_f32 / 1.25_f32.powi(3)).sqrt();
        assert!((op - expected).abs() < 1e-5, "got {op}, want {expected}");
        assert!(op < 1.0, "low-pass must reduce peak opacity");
    }

    #[test]
    fn two_d_mip_preserves_total_mass() {
        // The total integral of an un-normalised Gaussian α·exp(-½ x^T Σ⁻¹ x)
        // is α·2π·sqrt(|Σ|). The determinant-ratio opacity correction must keep
        // this invariant exactly.
        let cfg = MipSplatConfig {
            world_filter_sigma: 0.0,
            screen_filter_sigma: 1.3,
        };
        let cov = [3.0, 0.4, 0.4, 2.0];
        let alpha = 0.8_f32;
        let (out, op) = apply_2d_mip_filter(&cov, alpha, &cfg).expect("mip should succeed");
        let mass_in = alpha * (det2(&cov)).sqrt();
        let mass_out = op * (det2(&out)).sqrt();
        assert!(
            (mass_in - mass_out).abs() < 1e-5,
            "total mass must be preserved: {mass_in} vs {mass_out}"
        );
    }

    #[test]
    fn mip_footprint_is_at_least_one_pixel() {
        // After the 2D Mip filter the minor-axis variance must be >= σ²_mip,
        // i.e. the splat is never sub-pixel.
        let cfg = MipSplatConfig {
            world_filter_sigma: 0.0,
            screen_filter_sigma: 1.0,
        };
        // A near-degenerate, very thin splat.
        let cov = [0.01, 0.0, 0.0, 0.01];
        let (out, _) = apply_2d_mip_filter(&cov, 1.0, &cfg).expect("mip should succeed");
        assert!(out[0] >= 1.0 - 1e-6);
        assert!(out[3] >= 1.0 - 1e-6);
    }

    #[test]
    fn singular_input_rejected() {
        let cfg = MipSplatConfig::default();
        let singular3 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; // det 0
        assert!(apply_3d_smoothing(&singular3, 1.0, &cfg).is_err());
        let singular2 = [1.0, 1.0, 1.0, 1.0]; // det 0
        assert!(apply_2d_mip_filter(&singular2, 1.0, &cfg).is_err());
    }

    #[test]
    fn negative_sigma_rejected() {
        let cfg = MipSplatConfig {
            world_filter_sigma: -1.0,
            screen_filter_sigma: 1.0,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn determinant_helpers_correct() {
        assert!((det2(&[2.0, 0.0, 0.0, 3.0]) - 6.0).abs() < 1e-6);
        let m = [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0];
        // det = 1·(1·0-4·6) - 2·(0·0-4·5) + 3·(0·6-1·5) = -24 +40 -15 = 1
        assert!((det3(&m) - 1.0).abs() < 1e-4);
    }
}
