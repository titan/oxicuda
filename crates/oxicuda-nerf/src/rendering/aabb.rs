//! Ray / axis-aligned bounding box (AABB) intersection via the slab method.
//!
//! Bounded NeRF scenes are usually clipped to a cube `[lo, hi]³` (the "scene
//! box"). Before sampling along a ray we intersect it with this box to obtain the
//! near / far parameters `[t_near, t_far]` that bracket the portion of the ray
//! inside the volume; everything outside is empty space and can be skipped. This
//! is the same near/far computation Instant-NGP and Mip-NeRF 360 use to seed
//! stratified sampling.
//!
//! The **slab method** (Kay & Kajiya 1986) treats the box as the intersection of
//! three pairs of parallel planes (slabs). For each axis `d` the ray enters the
//! slab at `t1 = (lo_d − o_d)/dir_d` and exits at `t2 = (hi_d − o_d)/dir_d`; the
//! ray is inside the box on `[max_d min(t1,t2), min_d max(t1,t2)]`. A hit exists
//! iff that interval is non-empty and ends at a non-negative `t`. Rays parallel to
//! a slab (`dir_d ≈ 0`) are handled by checking whether the origin lies between
//! the slab planes.

use crate::error::{NerfError, NerfResult};
use crate::rendering::ray::Ray;

/// An axis-aligned bounding box `[min, max]` in world space.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    /// Lower corner (per-axis minimum).
    pub min: [f32; 3],
    /// Upper corner (per-axis maximum).
    pub max: [f32; 3],
}

/// Result of a ray / AABB intersection: the parametric entry / exit distances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AabbHit {
    /// Entry distance along the ray (clamped to `>= 0` so it starts at the origin
    /// when the origin is already inside the box).
    pub t_near: f32,
    /// Exit distance along the ray.
    pub t_far: f32,
}

impl Aabb {
    /// Create an AABB from `min` and `max` corners.
    ///
    /// # Errors
    /// Returns [`NerfError::InvalidBounds`] if any `min[d] > max[d]`.
    pub fn new(min: [f32; 3], max: [f32; 3]) -> NerfResult<Self> {
        for d in 0..3 {
            if !min[d].is_finite() || !max[d].is_finite() || max[d] < min[d] {
                return Err(NerfError::InvalidBounds {
                    near: min[d],
                    far: max[d],
                });
            }
        }
        Ok(Self { min, max })
    }

    /// Symmetric cube `[-half, half]³`.
    ///
    /// # Errors
    /// Returns [`NerfError::InvalidBounds`] if `half <= 0` or non-finite.
    pub fn cube(half: f32) -> NerfResult<Self> {
        if !half.is_finite() || half <= 0.0 {
            return Err(NerfError::InvalidBounds {
                near: -half,
                far: half,
            });
        }
        Ok(Self {
            min: [-half; 3],
            max: [half; 3],
        })
    }

    /// Whether a world point lies inside (or on) the box.
    #[must_use]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        (0..3).all(|d| p[d] >= self.min[d] && p[d] <= self.max[d])
    }

    /// Box centre.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }

    /// Intersect this box with `ray`, returning the entry / exit distances if the
    /// ray hits the box at any non-negative `t`.
    ///
    /// `t_near` is clamped to `>= 0`, so a ray whose origin is inside the box
    /// reports `t_near == 0`. Returns `None` when the ray misses the box or only
    /// touches it behind the origin (`t_far < 0`).
    #[must_use]
    pub fn intersect(&self, ray: &Ray) -> Option<AabbHit> {
        let mut t_enter = f32::NEG_INFINITY;
        let mut t_exit = f32::INFINITY;

        for d in 0..3 {
            let o = ray.origin[d];
            let dir = ray.dir[d];
            if dir.abs() < 1e-12 {
                // Ray parallel to this slab: miss unless the origin is between planes.
                if o < self.min[d] || o > self.max[d] {
                    return None;
                }
            } else {
                let inv = 1.0 / dir;
                let mut t1 = (self.min[d] - o) * inv;
                let mut t2 = (self.max[d] - o) * inv;
                if t1 > t2 {
                    core::mem::swap(&mut t1, &mut t2);
                }
                if t1 > t_enter {
                    t_enter = t1;
                }
                if t2 < t_exit {
                    t_exit = t2;
                }
                if t_enter > t_exit {
                    return None;
                }
            }
        }

        // The box is behind the ray entirely.
        if t_exit < 0.0 {
            return None;
        }
        let t_near = t_enter.max(0.0);
        if t_near > t_exit {
            return None;
        }
        Some(AabbHit {
            t_near,
            t_far: t_exit,
        })
    }

    /// Intersect, then clamp the hit interval to a user near/far window
    /// `[t_min, t_max]`. Returns `None` if there is no overlap.
    ///
    /// Useful for combining the scene box with the camera's depth range.
    #[must_use]
    pub fn intersect_clamped(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<AabbHit> {
        let hit = self.intersect(ray)?;
        let t_near = hit.t_near.max(t_min);
        let t_far = hit.t_far.min(t_max);
        if t_near > t_far {
            None
        } else {
            Some(AabbHit { t_near, t_far })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box() -> Aabb {
        Aabb::new([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]).expect("new should succeed")
    }

    #[test]
    fn ray_through_center_hits() {
        let b = unit_box();
        // Origin at z = -3, pointing +z, passes straight through.
        let r = Ray::new([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]).expect("new should succeed");
        let hit = b.intersect(&r).expect("intersect should succeed");
        assert!((hit.t_near - 2.0).abs() < 1e-5, "t_near = {}", hit.t_near);
        assert!((hit.t_far - 4.0).abs() < 1e-5, "t_far = {}", hit.t_far);
    }

    #[test]
    fn ray_missing_box_returns_none() {
        let b = unit_box();
        // Parallel to +z but offset far in x.
        let r = Ray::new([5.0, 0.0, -3.0], [0.0, 0.0, 1.0]).expect("new should succeed");
        assert!(b.intersect(&r).is_none());
    }

    #[test]
    fn ray_origin_inside_box_tnear_zero() {
        let b = unit_box();
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("new should succeed");
        let hit = b.intersect(&r).expect("intersect should succeed");
        assert!(hit.t_near.abs() < 1e-6, "inside box → t_near 0");
        assert!((hit.t_far - 1.0).abs() < 1e-5, "t_far = {}", hit.t_far);
    }

    #[test]
    fn ray_pointing_away_returns_none() {
        let b = unit_box();
        // Origin at z = +3, pointing further +z (box behind).
        let r = Ray::new([0.0, 0.0, 3.0], [0.0, 0.0, 1.0]).expect("new should succeed");
        assert!(b.intersect(&r).is_none());
    }

    #[test]
    fn diagonal_ray_hits() {
        let b = unit_box();
        let r = Ray::normalized([-3.0, -3.0, -3.0], [1.0, 1.0, 1.0])
            .expect("normalized should succeed");
        let hit = b.intersect(&r).expect("intersect should succeed");
        assert!(hit.t_near < hit.t_far);
        // The entry point should be inside the box.
        let p = r.at(hit.t_near + 1e-3);
        assert!(b.contains(p), "entry point not inside box: {:?}", p);
    }

    #[test]
    fn tnear_le_tfar_always() {
        let b = unit_box();
        let r =
            Ray::normalized([2.0, 0.3, -4.0], [-0.2, 0.0, 1.0]).expect("normalized should succeed");
        if let Some(hit) = b.intersect(&r) {
            assert!(hit.t_near <= hit.t_far);
            assert!(hit.t_near >= 0.0);
        }
    }

    #[test]
    fn parallel_ray_inside_slab_hits() {
        let b = unit_box();
        // dir has zero z-component but the ray sweeps x; origin z within [-1,1].
        let r = Ray::new([-3.0, 0.0, 0.5], [1.0, 0.0, 0.0]).expect("new should succeed");
        let hit = b.intersect(&r).expect("intersect should succeed");
        assert!((hit.t_near - 2.0).abs() < 1e-5);
        assert!((hit.t_far - 4.0).abs() < 1e-5);
    }

    #[test]
    fn parallel_ray_outside_slab_misses() {
        let b = unit_box();
        // Zero z-dir but origin z outside [-1, 1] → never enters.
        let r = Ray::new([-3.0, 0.0, 5.0], [1.0, 0.0, 0.0]).expect("new should succeed");
        assert!(b.intersect(&r).is_none());
    }

    #[test]
    fn contains_and_center() {
        let b = Aabb::new([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]).expect("new should succeed");
        assert!(b.contains([1.0, 2.0, 3.0]));
        assert!(!b.contains([3.0, 2.0, 3.0]));
        let c = b.center();
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn intersect_clamped_restricts_interval() {
        let b = unit_box();
        let r = Ray::new([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]).expect("new should succeed");
        // raw hit is [2, 4]; clamp to [2.5, 3.5]
        let hit = b
            .intersect_clamped(&r, 2.5, 3.5)
            .expect("intersect_clamped should succeed");
        assert!((hit.t_near - 2.5).abs() < 1e-5);
        assert!((hit.t_far - 3.5).abs() < 1e-5);
        // Non-overlapping window → None.
        assert!(b.intersect_clamped(&r, 10.0, 20.0).is_none());
    }

    #[test]
    fn cube_constructor() {
        let b = Aabb::cube(2.0).expect("cube should succeed");
        assert_eq!(b.min, [-2.0, -2.0, -2.0]);
        assert_eq!(b.max, [2.0, 2.0, 2.0]);
        assert!(Aabb::cube(0.0).is_err());
        assert!(Aabb::cube(-1.0).is_err());
    }

    #[test]
    fn invalid_bounds_error() {
        assert!(Aabb::new([1.0, 0.0, 0.0], [0.0, 1.0, 1.0]).is_err());
        assert!(Aabb::new([0.0, 0.0, 0.0], [f32::NAN, 1.0, 1.0]).is_err());
    }
}
