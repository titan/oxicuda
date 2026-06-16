//! Spherical harmonics directional encoding for view-dependent NeRF colour.
//!
//! Implements real spherical harmonics up to degree L = 4 (25 basis functions)
//! using explicit polynomial formulae from Sloan 2008 "Stupid Spherical Harmonics
//! (SH) Tricks" and the convention used in Mip-NeRF 360 / NeRF++ pipelines.
//!
//! Degree L gives (L+1)² basis functions, ordered by increasing l then
//! increasing m (m = −l, …, l).

use crate::error::{NerfError, NerfResult};

// ─── ShConfig ────────────────────────────────────────────────────────────────

/// Configuration for the spherical-harmonics encoder.
#[derive(Debug, Clone, Copy)]
pub struct ShConfig {
    /// Maximum SH degree (0 ≤ degree ≤ 4).
    pub degree: usize,
}

// ─── ShEncoder ───────────────────────────────────────────────────────────────

/// Spherical-harmonics encoder for 3-D unit directions.
///
/// Encodes a normalised direction vector as a vector of (degree+1)² real SH
/// basis values, suitable for view-dependent colour prediction in NeRF.
#[derive(Debug, Clone)]
pub struct ShEncoder {
    /// Encoder configuration.
    pub cfg: ShConfig,
    /// Number of SH coefficients: `(degree + 1)²`.
    pub n_coeffs: usize,
}

impl ShEncoder {
    /// Create a new `ShEncoder`.
    ///
    /// # Errors
    ///
    /// Returns `InvalidFeatureDim` if `degree > 4`.
    pub fn new(cfg: ShConfig) -> NerfResult<Self> {
        if cfg.degree > 4 {
            return Err(NerfError::InvalidFeatureDim { dim: cfg.degree });
        }
        let n_coeffs = ShEncoder::n_coeffs_for_degree(cfg.degree);
        Ok(Self { cfg, n_coeffs })
    }

    /// Encode a 3-D direction vector.
    ///
    /// The direction is normalised to the unit sphere before evaluation.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `dir.len() != 3`.
    /// Returns `ZeroRayDirection` if `||dir|| < 1e-10`.
    pub fn encode(&self, dir: &[f32]) -> NerfResult<Vec<f32>> {
        let unit = Self::normalize(dir)?;
        Self::sh_basis(unit[0], unit[1], unit[2], self.cfg.degree)
    }

    /// Normalise a 3-D direction vector to the unit sphere.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `dir.len() != 3`.
    /// Returns `ZeroRayDirection` if `||dir|| < 1e-10`.
    pub fn normalize(dir: &[f32]) -> NerfResult<Vec<f32>> {
        if dir.len() != 3 {
            return Err(NerfError::DimensionMismatch {
                expected: 3,
                got: dir.len(),
            });
        }
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if len < 1e-10 {
            return Err(NerfError::ZeroRayDirection);
        }
        Ok(vec![dir[0] / len, dir[1] / len, dir[2] / len])
    }

    /// Evaluate real SH basis functions at `(x, y, z)` on the unit sphere.
    ///
    /// Uses explicit polynomial formulae for degrees 0–4.  The vector is
    /// ordered by increasing l, then increasing m (m = −l … l).
    ///
    /// # Errors
    ///
    /// Returns `InvalidFeatureDim` if `degree > 4`.
    pub fn sh_basis(x: f32, y: f32, z: f32, degree: usize) -> NerfResult<Vec<f32>> {
        if degree > 4 {
            return Err(NerfError::InvalidFeatureDim { dim: degree });
        }
        let mut out = Vec::with_capacity(ShEncoder::n_coeffs_for_degree(degree));

        // L = 0
        out.push(0.282_095_f32); // Y_0^0

        if degree >= 1 {
            // L = 1
            out.push(-0.488_603 * y); // Y_1^{-1}
            out.push(0.488_603 * z); // Y_1^0
            out.push(-0.488_603 * x); // Y_1^1
        }

        if degree >= 2 {
            // L = 2
            out.push(1.092_548 * x * y); // Y_2^{-2}
            out.push(-1.092_548 * y * z); // Y_2^{-1}
            out.push(0.315_392 * (2.0 * z * z - x * x - y * y)); // Y_2^0
            out.push(-1.092_548 * x * z); // Y_2^1
            out.push(0.546_274 * (x * x - y * y)); // Y_2^2
        }

        if degree >= 3 {
            // L = 3
            out.push(-0.590_044 * y * (3.0 * x * x - y * y)); // Y_3^{-3}
            out.push(2.890_611 * x * y * z); // Y_3^{-2}
            out.push(-0.457_046 * y * (4.0 * z * z - x * x - y * y)); // Y_3^{-1}
            out.push(0.373_176 * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)); // Y_3^0
            out.push(-0.457_046 * x * (4.0 * z * z - x * x - y * y)); // Y_3^1
            out.push(1.445_306 * (x * x - y * y) * z); // Y_3^2
            out.push(-0.590_044 * x * (x * x - 3.0 * y * y)); // Y_3^3
        }

        if degree >= 4 {
            // L = 4
            let x2 = x * x;
            let y2 = y * y;
            let z2 = z * z;
            let z4 = z2 * z2;
            let x4 = x2 * x2;
            let y4 = y2 * y2;

            out.push(2.503_343 * x * y * (x2 - y2)); // Y_4^{-4}
            out.push(-1.770_131 * y * z * (3.0 * x2 - y2)); // Y_4^{-3}
            out.push(0.946_175 * x * y * (7.0 * z2 - 1.0)); // Y_4^{-2}
            out.push(-0.669_047 * y * z * (7.0 * z2 - 3.0)); // Y_4^{-1}
            out.push(0.105_786 * (35.0 * z4 - 30.0 * z2 + 3.0)); // Y_4^0
            out.push(-0.669_047 * x * z * (7.0 * z2 - 3.0)); // Y_4^1
            out.push(0.473_087 * (x2 - y2) * (7.0 * z2 - 1.0)); // Y_4^2
            out.push(-1.770_131 * x * z * (x2 - 3.0 * y2)); // Y_4^3
            out.push(0.625_836 * (x4 - 6.0 * x2 * y2 + y4)); // Y_4^4
        }

        Ok(out)
    }

    /// Reconstruct colour from SH coefficients.
    ///
    /// Computes `color[c] = Σ_{i} coeffs[i * n_channels + c] * basis[i]`
    /// for each colour channel `c = 0..n_channels`.
    ///
    /// `coeffs` must have length `n_coeffs * n_channels` where
    /// `n_coeffs = (degree+1)²`.
    ///
    /// # Errors
    ///
    /// Returns `DimensionMismatch` if `coeffs.len() != n_coeffs * n_channels`.
    /// Returns `ZeroRayDirection` or `DimensionMismatch` propagated from
    /// direction normalisation.
    pub fn sh_color(coeffs: &[f32], dir: &[f32], n_channels: usize) -> NerfResult<Vec<f32>> {
        if dir.len() != 3 {
            return Err(NerfError::DimensionMismatch {
                expected: 3,
                got: dir.len(),
            });
        }
        if n_channels == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }

        // Determine degree from coefficient count
        let n_coeffs_total = coeffs.len();
        if n_coeffs_total == 0 || !n_coeffs_total.is_multiple_of(n_channels) {
            return Err(NerfError::DimensionMismatch {
                expected: n_channels,
                got: n_coeffs_total % n_channels,
            });
        }
        let n_coeffs = n_coeffs_total / n_channels;

        // Infer degree from n_coeffs: (L+1)^2
        let degree =
            infer_degree(n_coeffs).ok_or(NerfError::InvalidFeatureDim { dim: n_coeffs })?;

        // Normalise and evaluate basis
        let unit = ShEncoder::normalize(dir)?;
        let basis = ShEncoder::sh_basis(unit[0], unit[1], unit[2], degree)?;

        // Weighted sum over basis functions per channel
        let mut color = vec![0.0_f32; n_channels];
        for (i, &b) in basis.iter().enumerate() {
            for (c, col) in color.iter_mut().enumerate() {
                *col += coeffs[i * n_channels + c] * b;
            }
        }

        Ok(color)
    }

    /// Number of SH coefficients for a given degree: `(degree + 1)²`.
    #[must_use]
    #[inline]
    pub fn n_coeffs_for_degree(degree: usize) -> usize {
        (degree + 1) * (degree + 1)
    }
}

// ─── Standalone convenience function ─────────────────────────────────────────

/// Evaluate SH basis at `(x, y, z)` without constructing an `ShEncoder`.
///
/// # Errors
///
/// Returns `InvalidFeatureDim` if `degree > 4`.
pub fn evaluate_sh(x: f32, y: f32, z: f32, degree: usize) -> NerfResult<Vec<f32>> {
    ShEncoder::sh_basis(x, y, z, degree)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Infer SH degree from the number of coefficients.
/// Returns `None` if `n_coeffs` is not a perfect square of the form (L+1)².
fn infer_degree(n_coeffs: usize) -> Option<usize> {
    (0..=4).find(|&l| (l + 1) * (l + 1) == n_coeffs)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- n_coeffs ---

    #[test]
    fn degree_0_n_coeffs() {
        assert_eq!(ShEncoder::n_coeffs_for_degree(0), 1);
    }

    #[test]
    fn degree_1_n_coeffs() {
        assert_eq!(ShEncoder::n_coeffs_for_degree(1), 4);
    }

    #[test]
    fn degree_2_n_coeffs() {
        assert_eq!(ShEncoder::n_coeffs_for_degree(2), 9);
    }

    #[test]
    fn degree_3_n_coeffs() {
        assert_eq!(ShEncoder::n_coeffs_for_degree(3), 16);
    }

    #[test]
    fn degree_4_n_coeffs() {
        assert_eq!(ShEncoder::n_coeffs_for_degree(4), 25);
    }

    // --- sh_basis ---

    #[test]
    fn sh_basis_degree0_is_constant() {
        let basis = ShEncoder::sh_basis(0.0, 0.0, 1.0, 0).expect("sh_basis should succeed");
        assert_eq!(basis.len(), 1);
        assert!((basis[0] - 0.282_095).abs() < 1e-5, "Y_0^0 = {}", basis[0]);
    }

    #[test]
    fn sh_basis_degree4_length() {
        let basis = ShEncoder::sh_basis(0.0, 0.0, 1.0, 4).expect("sh_basis should succeed");
        assert_eq!(basis.len(), 25);
    }

    #[test]
    fn sh_basis_orthogonality_spot_check() {
        // For (0, 0, 1): Y_1^{-1} = -0.488603 * y = 0, Y_1^0 = 0.488603 * z = 0.488603
        let basis = ShEncoder::sh_basis(0.0_f32, 0.0, 1.0, 1).expect("sh_basis should succeed");
        // basis[1] = Y_1^{-1} = -0.488603 * 0 = 0
        assert!(basis[1].abs() < 1e-6, "Y_1^{{-1}} at (0,0,1) should be 0");
        // basis[2] = Y_1^0 = 0.488603 * 1 = 0.488603
        assert!((basis[2] - 0.488_603).abs() < 1e-5, "Y_1^0 at (0,0,1)");
        // basis[3] = Y_1^1 = -0.488603 * 0 = 0
        assert!(basis[3].abs() < 1e-6, "Y_1^1 at (0,0,1) should be 0");
    }

    // --- encode ---

    #[test]
    fn encode_output_length() {
        let enc = ShEncoder::new(ShConfig { degree: 3 }).expect("new should succeed");
        let basis = enc
            .encode(&[1.0_f32, 0.0, 0.0])
            .expect("encode should succeed");
        assert_eq!(basis.len(), 16);
    }

    // --- normalize ---

    #[test]
    fn normalize_unit_vector_unchanged() {
        let unit = ShEncoder::normalize(&[0.0_f32, 1.0, 0.0]).expect("normalize should succeed");
        let len = (unit[0] * unit[0] + unit[1] * unit[1] + unit[2] * unit[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-6, "normalized length = {len}");
    }

    #[test]
    fn normalize_scales_correctly() {
        let unit = ShEncoder::normalize(&[0.0_f32, 2.0, 0.0]).expect("normalize should succeed");
        assert!(unit[0].abs() < 1e-6);
        assert!((unit[1] - 1.0).abs() < 1e-6);
        assert!(unit[2].abs() < 1e-6);
    }

    // --- sh_color ---

    #[test]
    fn sh_color_output_length() {
        // degree 1 → 4 coeffs per channel; n_channels = 3 → 12 total coeffs
        let n_coeffs = ShEncoder::n_coeffs_for_degree(1);
        let n_channels = 3;
        let coeffs = vec![0.0_f32; n_coeffs * n_channels];
        let color = ShEncoder::sh_color(&coeffs, &[0.0_f32, 0.0, 1.0], n_channels)
            .expect("sh_color should succeed");
        assert_eq!(color.len(), n_channels);
    }

    #[test]
    fn sh_color_degree0_is_dc() {
        // With degree 0 (1 basis, Y_0^0 = 0.282095), setting coeff = 1/0.282095 gives color ≈ 1.0
        let n_channels = 3;
        // 1 coeff × 3 channels
        let inv_c0 = 1.0_f32 / 0.282_095;
        let coeffs = vec![inv_c0; n_channels];
        let color = ShEncoder::sh_color(&coeffs, &[0.0_f32, 0.0, 1.0], n_channels)
            .expect("sh_color should succeed");
        for (c, &ch) in color.iter().enumerate() {
            assert!((ch - 1.0).abs() < 1e-4, "channel {c}: color = {ch}");
        }
    }

    // --- error cases ---

    #[test]
    fn err_degree_5() {
        assert!(ShEncoder::new(ShConfig { degree: 5 }).is_err());
    }

    #[test]
    fn err_dir_not_3d() {
        let enc = ShEncoder::new(ShConfig { degree: 2 }).expect("new should succeed");
        assert!(enc.encode(&[1.0_f32, 0.0]).is_err());
    }

    #[test]
    fn err_zero_direction() {
        assert!(ShEncoder::normalize(&[0.0_f32, 0.0, 0.0]).is_err());
    }
}
