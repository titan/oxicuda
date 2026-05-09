//! Quaternion type [w, x, y, z] with SLERP and rotation matrix conversion.

use crate::error::{Geom3dError, Geom3dResult};

/// Quaternion in `[w, x, y, z]` convention.
#[derive(Debug, Clone, PartialEq)]
pub struct Quat(pub [f32; 4]);

impl Quat {
    /// Identity quaternion `[1, 0, 0, 0]`.
    pub fn identity() -> Self {
        Self([1.0, 0.0, 0.0, 0.0])
    }

    /// Create from axis-angle representation.
    ///
    /// `axis` need not be normalized. Returns error if axis is zero-length.
    pub fn from_axis_angle(axis: [f32; 3], angle: f32) -> Geom3dResult<Self> {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm < 1e-8 {
            return Err(Geom3dError::InvalidQuaternion { norm: 0.0 });
        }
        let half = angle * 0.5;
        let s = half.sin() / norm;
        Ok(Self([half.cos(), axis[0] * s, axis[1] * s, axis[2] * s]))
    }

    /// Normalize the quaternion. Returns error if near-zero.
    pub fn normalize(&self) -> Geom3dResult<Self> {
        let n = (self.0[0] * self.0[0]
            + self.0[1] * self.0[1]
            + self.0[2] * self.0[2]
            + self.0[3] * self.0[3])
            .sqrt();
        if n < 1e-8 {
            return Err(Geom3dError::InvalidQuaternion { norm: n });
        }
        Ok(Self([
            self.0[0] / n,
            self.0[1] / n,
            self.0[2] / n,
            self.0[3] / n,
        ]))
    }

    /// Hamilton product `self ⊗ other`.
    pub fn mul(&self, other: &Self) -> Self {
        let (w1, x1, y1, z1) = (self.0[0], self.0[1], self.0[2], self.0[3]);
        let (w2, x2, y2, z2) = (other.0[0], other.0[1], other.0[2], other.0[3]);
        Self([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    }

    /// Conjugate: `[w, -x, -y, -z]`.
    pub fn conjugate(&self) -> Self {
        Self([self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    /// Convert to 3×3 rotation matrix (row-major).
    pub fn to_rotation_matrix(&self) -> [f32; 9] {
        let (w, x, y, z) = (self.0[0], self.0[1], self.0[2], self.0[3]);
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

    /// Convert from 3×3 rotation matrix. Forces w ≥ 0 (sign convention).
    pub fn from_rotation_matrix(r: &[f32; 9]) -> Geom3dResult<Self> {
        let trace = r[0] + r[4] + r[8];
        let (w, x, y, z);

        if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            w = 0.25 / s;
            x = (r[7] - r[5]) * s;
            y = (r[2] - r[6]) * s;
            z = (r[3] - r[1]) * s;
        } else if r[0] > r[4] && r[0] > r[8] {
            let s = 2.0 * (1.0 + r[0] - r[4] - r[8]).sqrt();
            w = (r[7] - r[5]) / s;
            x = 0.25 * s;
            y = (r[1] + r[3]) / s;
            z = (r[2] + r[6]) / s;
        } else if r[4] > r[8] {
            let s = 2.0 * (1.0 + r[4] - r[0] - r[8]).sqrt();
            w = (r[2] - r[6]) / s;
            x = (r[1] + r[3]) / s;
            y = 0.25 * s;
            z = (r[5] + r[7]) / s;
        } else {
            let s = 2.0 * (1.0 + r[8] - r[0] - r[4]).sqrt();
            w = (r[3] - r[1]) / s;
            x = (r[2] + r[6]) / s;
            y = (r[5] + r[7]) / s;
            z = 0.25 * s;
        }

        // Force w ≥ 0
        let sign = if w < 0.0 { -1.0_f32 } else { 1.0 };
        let q = Self([w * sign, x * sign, y * sign, z * sign]);
        q.normalize()
    }

    /// Spherical linear interpolation between `a` and `b` at parameter `t ∈ [0,1]`.
    ///
    /// Shortest path: if `dot < 0`, flips `b`'s sign.
    /// Falls back to normalized lerp if `1 - |dot| < 1e-6`.
    pub fn slerp(a: &Self, b_in: &Self, t: f32) -> Geom3dResult<Self> {
        let a = a.normalize()?;
        let b_norm = b_in.normalize()?;

        let dot = a.0[0] * b_norm.0[0]
            + a.0[1] * b_norm.0[1]
            + a.0[2] * b_norm.0[2]
            + a.0[3] * b_norm.0[3];

        // Shortest path
        let (b, dot) = if dot < 0.0 {
            (
                Self([-b_norm.0[0], -b_norm.0[1], -b_norm.0[2], -b_norm.0[3]]),
                -dot,
            )
        } else {
            (b_norm, dot)
        };

        // Fallback to lerp if nearly parallel
        if 1.0 - dot < 1e-6 {
            let lerp = Self([
                a.0[0] + t * (b.0[0] - a.0[0]),
                a.0[1] + t * (b.0[1] - a.0[1]),
                a.0[2] + t * (b.0[2] - a.0[2]),
                a.0[3] + t * (b.0[3] - a.0[3]),
            ]);
            return lerp.normalize();
        }

        let theta = dot.clamp(-1.0, 1.0).acos();
        let sin_theta = theta.sin();
        let s_a = ((1.0 - t) * theta).sin() / sin_theta;
        let s_b = (t * theta).sin() / sin_theta;

        let out = Self([
            s_a * a.0[0] + s_b * b.0[0],
            s_a * a.0[1] + s_b * b.0[1],
            s_a * a.0[2] + s_b * b.0[2],
            s_a * a.0[3] + s_b * b.0[3],
        ]);
        out.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quat_identity() {
        let q = Quat::identity();
        assert_eq!(q.0, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn quat_from_axis_angle_unit_norm() {
        let q = Quat::from_axis_angle([0.0, 0.0, 1.0], std::f32::consts::PI / 2.0).unwrap();
        let n = q.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5, "Quaternion must be unit norm");
    }

    #[test]
    fn quat_from_axis_angle_zero_error() {
        assert!(Quat::from_axis_angle([0.0, 0.0, 0.0], 1.0).is_err());
    }

    #[test]
    fn quat_mul_identity() {
        let q = Quat::from_axis_angle([0.0, 1.0, 0.0], 0.5).unwrap();
        let id = Quat::identity();
        let q_id = q.mul(&id);
        for (a, b) in q.0.iter().zip(q_id.0.iter()) {
            assert!((a - b).abs() < 1e-5, "q * id must equal q");
        }
    }

    #[test]
    fn quat_conjugate_inverse() {
        let q = Quat::from_axis_angle([1.0, 0.0, 0.0], 1.0).unwrap();
        let q_inv = q.conjugate();
        let prod = q.mul(&q_inv);
        // Should be near identity
        assert!((prod.0[0] - 1.0).abs() < 1e-5, "q * q^* should be identity");
        assert!(prod.0[1].abs() < 1e-5);
    }

    #[test]
    fn quat_to_rotation_matrix_identity() {
        let q = Quat::identity();
        let r = q.to_rotation_matrix();
        let expected = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for (a, b) in r.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "Identity quat → identity matrix");
        }
    }

    #[test]
    fn quat_roundtrip_matrix() {
        let q = Quat::from_axis_angle([1.0, 1.0, 0.0], 0.8).unwrap();
        let r = q.to_rotation_matrix();
        let q2 = Quat::from_rotation_matrix(&r).unwrap();
        for (a, b) in q.0.iter().zip(q2.0.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "Roundtrip quat→mat→quat failed: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn quat_slerp_t0_is_a() {
        let a = Quat::from_axis_angle([0.0, 0.0, 1.0], 0.0).unwrap();
        let b = Quat::from_axis_angle([0.0, 0.0, 1.0], 1.0).unwrap();
        let s = Quat::slerp(&a, &b, 0.0).unwrap();
        for (x, y) in s.0.iter().zip(a.normalize().unwrap().0.iter()) {
            assert!((x - y).abs() < 1e-4, "slerp(t=0) should be a");
        }
    }

    #[test]
    fn quat_slerp_t1_is_b() {
        let a = Quat::identity();
        // Use 0.9 radians rather than PI to avoid gimbal issues at t=1 with 180° rotation
        let b = Quat::from_axis_angle([1.0, 0.0, 0.0], 0.9).unwrap();
        let s = Quat::slerp(&a, &b, 1.0).unwrap();
        let b_n = b.normalize().unwrap();
        for (x, y) in s.0.iter().zip(b_n.0.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "slerp(t=1) should be b, got {:?}",
                s.0
            );
        }
    }

    #[test]
    fn quat_slerp_unit_norm() {
        let a = Quat::from_axis_angle([1.0, 0.0, 0.0], 0.3).unwrap();
        let b = Quat::from_axis_angle([0.0, 1.0, 0.0], 1.2).unwrap();
        for ti in 0..=10 {
            let t = ti as f32 / 10.0;
            let s = Quat::slerp(&a, &b, t).unwrap();
            let n = s.0.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                (n - 1.0).abs() < 1e-4,
                "slerp result must be unit norm at t={t}"
            );
        }
    }
}
