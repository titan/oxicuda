//! 3D Gaussian primitive for Gaussian splatting.

use crate::error::{Geom3dError, Geom3dResult};

/// SH coefficient counts by degree.
pub const SH_DEG0: usize = 1;
pub const SH_DEG1: usize = 3;
pub const SH_DEG2: usize = 5;
/// Total SH coefficients for L=0..2 per channel.
pub const SH_TOTAL_PER_CHANNEL: usize = SH_DEG0 + SH_DEG1 + SH_DEG2; // = 9

/// SH constants.
const Y00: f32 = 0.282_094_8; // l=0, m=0
const Y11: f32 = 0.488_602_5; // l=1, |m|=1 (×x or ×y or ×z)
const Y20: f32 = 0.315_391_6; // l=2, m=0  ×(3z²-1)
const Y21A: f32 = 1.092_548_4; // l=2, m=±1 ×xz or ×yz
const Y22A: f32 = 0.546_274_2; // l=2, m=±2 ×(x²-y²)
const Y22B: f32 = 1.092_548_4; // l=2, m=±2 ×xy

/// A single 3D Gaussian primitive.
///
/// `sh`: SH coefficients for RGB: total `3 * SH_TOTAL_PER_CHANNEL` = 27 floats.
/// Layout: `[R_coeff_0..9, G_coeff_0..9, B_coeff_0..9]`.
#[derive(Debug, Clone)]
pub struct Gaussian3d {
    pub pos: [f32; 3],
    pub rot: [f32; 4],   // wxyz quaternion
    pub scale: [f32; 3], // log-scale (exp to get actual)
    pub opacity: f32,    // pre-sigmoid
    pub sh: Vec<f32>,    // 27 SH coefficients (9 per RGB channel)
}

/// Convert wxyz quaternion to 3×3 rotation matrix (row-major).
fn quat_to_mat(q: &[f32; 4]) -> [f32; 9] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]
}

/// Multiply two 3×3 matrices (row-major).
fn mat3_mul(a: &[f32; 9], b: &[f32; 9]) -> [f32; 9] {
    let mut c = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    c
}

/// Transpose a 3×3 matrix.
fn mat3_t(m: &[f32; 9]) -> [f32; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Multiply 3×3 by diagonal scale matrix S = diag(sx,sy,sz).
fn mat3_scale(m: &[f32; 9], s: &[f32; 3]) -> [f32; 9] {
    let mut out = *m;
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] *= s[j];
        }
    }
    out
}

impl Gaussian3d {
    /// Create a unit Gaussian at `pos` with identity rotation, zero scale,
    /// sigmoid(0)=0.5 opacity, and constant color.
    pub fn new_unit(pos: [f32; 3]) -> Self {
        let sh = vec![0.0_f32; 3 * SH_TOTAL_PER_CHANNEL]; // zero SH → black by default
        Self {
            pos,
            rot: [1.0, 0.0, 0.0, 0.0], // identity quaternion
            scale: [0.0, 0.0, 0.0],    // log-scale=0 → actual=1
            opacity: 0.0,              // sigmoid(0) = 0.5
            sh,
        }
    }

    /// Returns 3×3 covariance matrix `Σ = R·S·Sᵀ·Rᵀ` (row-major).
    ///
    /// `R = quat_to_mat(rot)`, `S = diag(exp(scale))`.
    pub fn covariance3d(&self) -> Geom3dResult<[f32; 9]> {
        let norm = (self.rot[0] * self.rot[0]
            + self.rot[1] * self.rot[1]
            + self.rot[2] * self.rot[2]
            + self.rot[3] * self.rot[3])
            .sqrt();
        if norm < 1e-7 {
            return Err(Geom3dError::InvalidQuaternion { norm });
        }
        let q = [
            self.rot[0] / norm,
            self.rot[1] / norm,
            self.rot[2] / norm,
            self.rot[3] / norm,
        ];

        let r = quat_to_mat(&q);
        let s = [
            self.scale[0].exp(),
            self.scale[1].exp(),
            self.scale[2].exp(),
        ];

        // RS = R * diag(s)
        let rs = mat3_scale(&r, &s);
        // Σ = RS * (RS)^T
        let rs_t = mat3_t(&rs);
        let cov = mat3_mul(&rs, &rs_t);

        Ok(cov)
    }

    /// Evaluate spherical harmonics color in direction `dir` (unit vec).
    ///
    /// Evaluates L=0..2 SH basis, returns RGB ∈ ℝ³.
    pub fn sh_color(&self, dir: [f32; 3]) -> Geom3dResult<[f32; 3]> {
        let n_coeffs = SH_TOTAL_PER_CHANNEL;
        if self.sh.len() != 3 * n_coeffs {
            return Err(Geom3dError::InvalidShCoefficients {
                expected: 3 * n_coeffs,
                got: self.sh.len(),
            });
        }

        let (x, y, z) = (dir[0], dir[1], dir[2]);

        // SH basis functions evaluated at dir
        let basis = [
            Y00,                       // L=0, m=0
            Y11 * x,                   // L=1, m=-1 (×x using common mapping)
            Y11 * y,                   // L=1, m=0  (×y)
            Y11 * z,                   // L=1, m=1  (×z)
            Y20 * (3.0 * z * z - 1.0), // L=2, m=0
            Y21A * x * z,              // L=2, m=-1 (×xz)
            Y21A * y * z,              // L=2, m=1  (×yz)
            Y22A * (x * x - y * y),    // L=2, m=-2 (×(x²-y²))
            Y22B * x * y,              // L=2, m=2  (×xy)
        ];

        let mut color = [0.0_f32; 3];
        for (ch, val) in color.iter_mut().enumerate() {
            let offset = ch * n_coeffs;
            for (i, &b) in basis.iter().enumerate() {
                *val += self.sh[offset + i] * b;
            }
        }

        Ok(color)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_new_unit() {
        let g = Gaussian3d::new_unit([0.0, 0.0, 0.0]);
        assert_eq!(g.rot, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(g.scale, [0.0, 0.0, 0.0]);
        assert_eq!(g.sh.len(), 27);
    }

    #[test]
    fn gaussian_covariance_identity_rot_unit_scale() {
        // log_scale=0 → exp(0)=1, identity rotation → Σ = I
        let g = Gaussian3d::new_unit([0.0, 0.0, 0.0]);
        let cov = g.covariance3d().expect("covariance3d should succeed");
        // Should be identity
        assert!((cov[0] - 1.0).abs() < 1e-5, "cov[0]={}", cov[0]);
        assert!((cov[4] - 1.0).abs() < 1e-5, "cov[4]={}", cov[4]);
        assert!((cov[8] - 1.0).abs() < 1e-5, "cov[8]={}", cov[8]);
        assert!(cov[1].abs() < 1e-5);
        assert!(cov[2].abs() < 1e-5);
    }

    #[test]
    fn gaussian_covariance_positive_definite() {
        // For any valid Gaussian, Σ should be positive semi-definite
        // Check diagonal elements are positive
        let g = Gaussian3d {
            pos: [0.0, 0.0, 0.0],
            rot: [
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
            ],
            scale: [0.5_f32.ln(), 1.0_f32.ln(), 2.0_f32.ln()],
            opacity: 0.0,
            sh: vec![0.0; 27],
        };
        let cov = g.covariance3d().expect("covariance3d should succeed");
        assert!(cov[0] > 0.0, "cov[0] must be positive");
        assert!(cov[4] > 0.0, "cov[4] must be positive");
        assert!(cov[8] > 0.0, "cov[8] must be positive");
    }

    #[test]
    fn gaussian_sh_color_shape() {
        let g = Gaussian3d::new_unit([0.0, 0.0, 0.0]);
        let color = g
            .sh_color([0.0, 0.0, 1.0])
            .expect("sh_color should succeed");
        assert_eq!(color.len(), 3);
    }

    #[test]
    fn gaussian_sh_color_finite() {
        let g = Gaussian3d {
            pos: [0.0, 0.0, 0.0],
            rot: [1.0, 0.0, 0.0, 0.0],
            scale: [0.0, 0.0, 0.0],
            opacity: 0.0,
            sh: vec![1.0; 27],
        };
        let color = g
            .sh_color([0.577, 0.577, 0.577])
            .expect("sh_color should succeed");
        assert!(color.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gaussian_invalid_quaternion_error() {
        let g = Gaussian3d {
            pos: [0.0, 0.0, 0.0],
            rot: [0.0, 0.0, 0.0, 0.0], // zero quaternion
            scale: [0.0, 0.0, 0.0],
            opacity: 0.0,
            sh: vec![0.0; 27],
        };
        assert!(g.covariance3d().is_err());
    }

    #[test]
    fn gaussian_sh_wrong_size_error() {
        let g = Gaussian3d {
            pos: [0.0, 0.0, 0.0],
            rot: [1.0, 0.0, 0.0, 0.0],
            scale: [0.0, 0.0, 0.0],
            opacity: 0.0,
            sh: vec![0.0; 9], // wrong: should be 27
        };
        assert!(g.sh_color([0.0, 0.0, 1.0]).is_err());
    }
}
