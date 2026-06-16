//! Rigid body transform: rotation + translation.

use crate::error::{Geom3dError, Geom3dResult};

/// Rigid transform: `y = R * x + t`.
///
/// `r`: row-major 3×3 rotation matrix.
/// `t`: translation vector.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidTransform {
    pub r: [f32; 9],
    pub t: [f32; 3],
}

impl RigidTransform {
    /// Identity transform: `R = I`, `t = 0`.
    pub fn identity() -> Self {
        Self {
            r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            t: [0.0, 0.0, 0.0],
        }
    }

    /// Create from axis-angle (Rodrigues' rotation formula).
    ///
    /// `axis` need not be normalized. Returns error if axis is zero-length.
    pub fn from_axis_angle(axis: [f32; 3], angle_rad: f32) -> Geom3dResult<Self> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-8 {
            return Err(Geom3dError::InvalidQuaternion { norm: 0.0 });
        }
        let ux = axis[0] / norm;
        let uy = axis[1] / norm;
        let uz = axis[2] / norm;

        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let t = 1.0 - c;

        let r = [
            t * ux * ux + c,
            t * ux * uy - s * uz,
            t * ux * uz + s * uy,
            t * ux * uy + s * uz,
            t * uy * uy + c,
            t * uy * uz - s * ux,
            t * ux * uz - s * uy,
            t * uy * uz + s * ux,
            t * uz * uz + c,
        ];

        Ok(Self {
            r,
            t: [0.0, 0.0, 0.0],
        })
    }

    /// Inverse: `Rᵀ`, `-Rᵀ t`.
    pub fn inverse(&self) -> Self {
        let rt = [
            self.r[0], self.r[3], self.r[6], self.r[1], self.r[4], self.r[7], self.r[2], self.r[5],
            self.r[8],
        ];
        let neg_rt_t = [
            -(rt[0] * self.t[0] + rt[1] * self.t[1] + rt[2] * self.t[2]),
            -(rt[3] * self.t[0] + rt[4] * self.t[1] + rt[5] * self.t[2]),
            -(rt[6] * self.t[0] + rt[7] * self.t[1] + rt[8] * self.t[2]),
        ];
        Self { r: rt, t: neg_rt_t }
    }

    /// Compose: `(self ∘ other)(x) = self(other(x))`.
    pub fn compose(&self, other: &Self) -> Self {
        // R = self.r * other.r
        let mut r = [0.0_f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r[i * 3 + j] += self.r[i * 3 + k] * other.r[k * 3 + j];
                }
            }
        }
        // t = self.r * other.t + self.t
        let t = [
            self.r[0] * other.t[0] + self.r[1] * other.t[1] + self.r[2] * other.t[2] + self.t[0],
            self.r[3] * other.t[0] + self.r[4] * other.t[1] + self.r[5] * other.t[2] + self.t[1],
            self.r[6] * other.t[0] + self.r[7] * other.t[1] + self.r[8] * other.t[2] + self.t[2],
        ];
        Self { r, t }
    }

    /// Apply transform to a single point.
    pub fn apply_point(&self, p: [f32; 3]) -> [f32; 3] {
        [
            self.r[0] * p[0] + self.r[1] * p[1] + self.r[2] * p[2] + self.t[0],
            self.r[3] * p[0] + self.r[4] * p[1] + self.r[5] * p[2] + self.t[1],
            self.r[6] * p[0] + self.r[7] * p[1] + self.r[8] * p[2] + self.t[2],
        ]
    }

    /// Apply transform to all points in `points [n×3]`.
    pub fn apply_pointcloud(&self, points: &[f32], n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; n * 3];
        for i in 0..n {
            let p = [points[i * 3], points[i * 3 + 1], points[i * 3 + 2]];
            let q = self.apply_point(p);
            out[i * 3] = q[0];
            out[i * 3 + 1] = q[1];
            out[i * 3 + 2] = q[2];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_apply() {
        let tf = RigidTransform::identity();
        let p = [1.0_f32, 2.0, 3.0];
        let q = tf.apply_point(p);
        assert_eq!(q, p);
    }

    #[test]
    fn from_axis_angle_zero_error() {
        assert!(RigidTransform::from_axis_angle([0.0, 0.0, 0.0], 1.0).is_err());
    }

    #[test]
    fn from_axis_angle_90deg_z() {
        let tf = RigidTransform::from_axis_angle([0.0, 0.0, 1.0], std::f32::consts::PI / 2.0)
            .expect("from_axis_angle should succeed");
        let p = [1.0_f32, 0.0, 0.0];
        let q = tf.apply_point(p);
        // 90° around z: [1,0,0] → [0,1,0]
        assert!((q[0] - 0.0).abs() < 1e-5, "x={}", q[0]);
        assert!((q[1] - 1.0).abs() < 1e-5, "y={}", q[1]);
        assert!((q[2] - 0.0).abs() < 1e-5, "z={}", q[2]);
    }

    #[test]
    fn inverse_compose_is_identity() {
        let tf = RigidTransform::from_axis_angle([1.0, 0.5, 0.3], 0.7)
            .expect("from_axis_angle should succeed");
        let tf_inv = tf.inverse();
        let composed = tf.compose(&tf_inv);
        let id = RigidTransform::identity();
        for (a, b) in composed.r.iter().zip(id.r.iter()) {
            assert!((a - b).abs() < 1e-4, "R component mismatch: {} vs {}", a, b);
        }
        for (a, b) in composed.t.iter().zip(id.t.iter()) {
            assert!((a - b).abs() < 1e-4, "t component mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn compose_applies_correctly() {
        // Two 90° z-rotations = 180° rotation
        let half = RigidTransform::from_axis_angle([0.0, 0.0, 1.0], std::f32::consts::PI / 2.0)
            .expect("from_axis_angle should succeed");
        let full = half.compose(&half);
        let p = [1.0_f32, 0.0, 0.0];
        let q = full.apply_point(p);
        // 180° around z: [1,0,0] → [-1,0,0]
        assert!((q[0] + 1.0).abs() < 1e-4, "x={}", q[0]);
        assert!(q[1].abs() < 1e-4, "y={}", q[1]);
    }

    #[test]
    fn apply_pointcloud_shape() {
        let tf = RigidTransform::identity();
        let pts: Vec<f32> = (0..5).flat_map(|i| vec![i as f32, 0.0, 0.0]).collect();
        let out = tf.apply_pointcloud(&pts, 5);
        assert_eq!(out.len(), 15);
        assert_eq!(out, pts);
    }
}
