//! Ray representation and pinhole camera model.

use crate::error::{NerfError, NerfResult};

// ─── Ray ─────────────────────────────────────────────────────────────────────

/// A 3D ray with origin and (ideally normalized) direction.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Ray origin in world space.
    pub origin: [f32; 3],
    /// Ray direction (should be normalized).
    pub dir: [f32; 3],
}

impl Ray {
    /// Create a ray with the given origin and direction (not normalized).
    ///
    /// # Errors
    ///
    /// Returns `ZeroRayDirection` if `|dir| < 1e-8`.
    pub fn new(origin: [f32; 3], dir: [f32; 3]) -> NerfResult<Self> {
        let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if len_sq < 1e-16 {
            return Err(NerfError::ZeroRayDirection);
        }
        Ok(Self { origin, dir })
    }

    /// Create a ray with automatically normalized direction.
    ///
    /// # Errors
    ///
    /// Returns `ZeroRayDirection` if `|dir| < 1e-8`.
    pub fn normalized(origin: [f32; 3], dir: [f32; 3]) -> NerfResult<Self> {
        let len_sq = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        if len_sq < 1e-16 {
            return Err(NerfError::ZeroRayDirection);
        }
        let inv_len = 1.0 / len_sq.sqrt();
        let ndir = [dir[0] * inv_len, dir[1] * inv_len, dir[2] * inv_len];
        Ok(Self { origin, dir: ndir })
    }

    /// Evaluate the point along the ray at parameter `t`: `origin + t * dir`.
    #[must_use]
    pub fn at(&self, t: f32) -> [f32; 3] {
        [
            self.origin[0] + t * self.dir[0],
            self.origin[1] + t * self.dir[1],
            self.origin[2] + t * self.dir[2],
        ]
    }
}

// ─── PinholeCamera ───────────────────────────────────────────────────────────

/// Pinhole camera intrinsics.
#[derive(Debug, Clone, Copy)]
pub struct PinholeCamera {
    /// Focal length in x (pixels).
    pub fx: f32,
    /// Focal length in y (pixels).
    pub fy: f32,
    /// Principal point x (pixels).
    pub cx: f32,
    /// Principal point y (pixels).
    pub cy: f32,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl PinholeCamera {
    /// Create a new pinhole camera model.
    ///
    /// # Errors
    ///
    /// Returns `InvalidCameraIntrinsics` for non-positive focal lengths or zero dimensions.
    pub fn new(fx: f32, fy: f32, cx: f32, cy: f32, w: u32, h: u32) -> NerfResult<Self> {
        if fx <= 0.0 || fy <= 0.0 {
            return Err(NerfError::InvalidCameraIntrinsics {
                msg: "focal lengths must be positive".into(),
            });
        }
        if w == 0 || h == 0 {
            return Err(NerfError::InvalidCameraIntrinsics {
                msg: "image dimensions must be > 0".into(),
            });
        }
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            width: w,
            height: h,
        })
    }

    /// Generate the ray through pixel `(u, v)` in camera coordinates.
    ///
    /// `c2w` is a row-major 3×4 camera-to-world transform:
    /// ```text
    /// [ R[0,0] R[0,1] R[0,2] t[0]
    ///   R[1,0] R[1,1] R[1,2] t[1]
    ///   R[2,0] R[2,1] R[2,2] t[2] ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `ZeroRayDirection` if the transformed direction is near-zero.
    pub fn ray_through_pixel(&self, u: f32, v: f32, c2w: &[f32; 12]) -> NerfResult<Ray> {
        // Camera-space direction (unnormalized)
        let dx = (u - self.cx) / self.fx;
        let dy = (v - self.cy) / self.fy;
        let dz = 1.0_f32;

        // Rotate to world space using upper-left 3×3 of c2w
        let wx = c2w[0] * dx + c2w[1] * dy + c2w[2] * dz;
        let wy = c2w[4] * dx + c2w[5] * dy + c2w[6] * dz;
        let wz = c2w[8] * dx + c2w[9] * dy + c2w[10] * dz;

        // Origin = translation column of c2w
        let origin = [c2w[3], c2w[7], c2w[11]];

        Ray::normalized(origin, [wx, wy, wz])
    }

    /// Generate all W×H rays for the camera with given `c2w` matrix.
    ///
    /// Output: `width * height` rays in row-major order (left-to-right, top-to-bottom).
    ///
    /// # Errors
    ///
    /// Returns `ZeroRayDirection` if any pixel ray has near-zero direction.
    pub fn generate_rays(&self, c2w: &[f32; 12]) -> NerfResult<Vec<Ray>> {
        let n = (self.width * self.height) as usize;
        let mut rays = Vec::with_capacity(n);
        for row in 0..self.height {
            for col in 0..self.width {
                let u = col as f32 + 0.5;
                let v = row as f32 + 0.5;
                rays.push(self.ray_through_pixel(u, v, c2w)?);
            }
        }
        Ok(rays)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_c2w() -> [f32; 12] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    #[test]
    fn ray_at_origin() {
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let pt = r.at(2.0);
        assert!((pt[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn ray_zero_dir_error() {
        assert!(Ray::new([0.0; 3], [0.0; 3]).is_err());
    }

    #[test]
    fn camera_principal_ray() {
        let cam = PinholeCamera::new(100.0, 100.0, 50.0, 50.0, 100, 100).unwrap();
        let ray = cam.ray_through_pixel(50.5, 50.5, &identity_c2w()).unwrap();
        // Principal ray at (cx, cy) points forward (+z in camera = +z in world for identity)
        assert!(ray.dir[2] > 0.0);
    }

    #[test]
    fn generate_rays_count() {
        let cam = PinholeCamera::new(100.0, 100.0, 50.0, 50.0, 4, 3).unwrap();
        let rays = cam.generate_rays(&identity_c2w()).unwrap();
        assert_eq!(rays.len(), 12);
    }
}
