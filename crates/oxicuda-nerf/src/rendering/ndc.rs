//! Normalized Device Coordinate (NDC) ray transformation for forward-facing NeRF.
//!
//! Mildenhall et al. (2020) "NeRF: Representing Scenes as Neural Radiance Fields
//! for View Synthesis", ECCV 2020 — Appendix C.
//!
//! For *forward-facing*, unbounded scenes (e.g. the LLFF dataset) NeRF reparametrises
//! rays into a projective **NDC** space so that the infinite view frustum maps onto a
//! finite cube and depth is sampled linearly in disparity (`1/z`) rather than in `z`.
//! A ray `o + t·d` in camera space (looking down `−z`) is rewritten as a new ray
//! `o' + t'·d'` whose parameter `t' ∈ [0, 1]` corresponds to the world depth range
//! `[near, ∞)`. The closed-form transform (paper Eq. 25-26) is
//!
//! ```text
//! o'_x = −(2 f / W) · o_x / o_z
//! o'_y = −(2 f / H) · o_y / o_z
//! o'_z =  1 + 2 n / o_z
//!
//! d'_x = −(2 f / W) · (d_x / d_z − o_x / o_z)
//! d'_y = −(2 f / H) · (d_y / d_z − o_y / o_z)
//! d'_z = −2 n / o_z
//! ```
//!
//! where `f` is the focal length, `W`/`H` the image width/height, and `n` the near
//! plane. Before transforming, ray origins are first shifted to the near plane
//! (`o ← o + t_n·d` with `t_n = −(n + o_z)/d_z`) so that `t' = 0` corresponds to the
//! near plane. After transforming, sampling `t' ∈ [0, 1]` uniformly yields samples
//! that are uniform in disparity in world space.

use crate::error::{NerfError, NerfResult};
use crate::rendering::ray::Ray;

/// Convert a camera-space ray to NDC space (forward-facing NeRF).
///
/// - `ray`     : ray in camera space; the camera looks down `−z`, so `dir[2] < 0`.
/// - `focal`   : focal length in pixels (assumed equal in x and y).
/// - `width`   : image width in pixels.
/// - `height`  : image height in pixels.
/// - `near`    : near-plane distance (`> 0`).
///
/// Returns the NDC-space [`Ray`]; sampling its parameter on `[0, 1]` corresponds to
/// world depth `[near, ∞)`.
///
/// # Errors
/// - [`NerfError::InvalidCameraIntrinsics`] if `focal <= 0`, `width == 0`, `height == 0`.
/// - [`NerfError::InvalidBounds`] if `near <= 0`.
/// - [`NerfError::ZeroRayDirection`] if `dir[2] ≈ 0` (cannot shift to near plane).
/// - [`NerfError::NanEncountered`] if `o_z ≈ 0` after shifting, or any output is non-finite.
pub fn ndc_ray(ray: &Ray, focal: f32, width: u32, height: u32, near: f32) -> NerfResult<Ray> {
    if !focal.is_finite() || focal <= 0.0 {
        return Err(NerfError::InvalidCameraIntrinsics {
            msg: "focal length must be positive".into(),
        });
    }
    if width == 0 || height == 0 {
        return Err(NerfError::InvalidCameraIntrinsics {
            msg: "image dimensions must be > 0".into(),
        });
    }
    if !near.is_finite() || near <= 0.0 {
        return Err(NerfError::InvalidBounds {
            near,
            far: f32::INFINITY,
        });
    }
    if ray.dir[2].abs() < 1e-8 {
        return Err(NerfError::ZeroRayDirection);
    }

    // Shift the ray origin to the near plane: t_n = -(near + o_z) / d_z.
    let t_n = -(near + ray.origin[2]) / ray.dir[2];
    let ox = ray.origin[0] + t_n * ray.dir[0];
    let oy = ray.origin[1] + t_n * ray.dir[1];
    let oz = ray.origin[2] + t_n * ray.dir[2];
    let dx = ray.dir[0];
    let dy = ray.dir[1];
    let dz = ray.dir[2];

    if oz.abs() < 1e-8 {
        return Err(NerfError::NanEncountered {
            context: "ndc_ray: o_z ≈ 0".into(),
        });
    }

    let ax = 2.0 * focal / width as f32;
    let ay = 2.0 * focal / height as f32;

    // Projected origin.
    let o0 = -ax * ox / oz;
    let o1 = -ay * oy / oz;
    let o2 = 1.0 + 2.0 * near / oz;

    // Projected direction.
    let d0 = -ax * (dx / dz - ox / oz);
    let d1 = -ay * (dy / dz - oy / oz);
    let d2 = -2.0 * near / oz;

    let origin = [o0, o1, o2];
    let dir = [d0, d1, d2];
    if origin.iter().chain(dir.iter()).any(|v| !v.is_finite()) {
        return Err(NerfError::NanEncountered {
            context: "ndc_ray: non-finite output".into(),
        });
    }

    // NDC rays are NOT unit-length; preserve raw direction (do not normalize).
    Ok(Ray { origin, dir })
}

/// Convert an NDC-space depth `t' ∈ [0, 1]` back to a world-space depth `z`.
///
/// In NDC the relationship between the parameter and world depth is
/// `t' = 1 − near/z`, i.e. `z = near / (1 − t')`. At `t' = 0` this is `near`, and as
/// `t' → 1` the world depth tends to `+∞` (the far plane at infinity). This is the
/// inverse used to report metric depth after rendering an NDC ray.
///
/// # Errors
/// - [`NerfError::InvalidBounds`] if `near <= 0`.
/// - [`NerfError::NanEncountered`] if `t_ndc >= 1` (depth at infinity / undefined).
pub fn ndc_depth_to_world(t_ndc: f32, near: f32) -> NerfResult<f32> {
    if !near.is_finite() || near <= 0.0 {
        return Err(NerfError::InvalidBounds {
            near,
            far: f32::INFINITY,
        });
    }
    let denom = 1.0 - t_ndc;
    if denom <= 1e-8 {
        return Err(NerfError::NanEncountered {
            context: "ndc_depth_to_world: t_ndc → 1 maps to infinite depth".into(),
        });
    }
    Ok(near / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ndc_origin_on_axis_maps_to_center() {
        // Ray along -z through the optical center → NDC x,y ≈ 0.
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("new should succeed");
        let ndc = ndc_ray(&r, 100.0, 200, 200, 1.0).expect("ndc_ray should succeed");
        assert!(approx(ndc.origin[0], 0.0, 1e-4), "x = {}", ndc.origin[0]);
        assert!(approx(ndc.origin[1], 0.0, 1e-4), "y = {}", ndc.origin[1]);
    }

    #[test]
    fn ndc_origin_z_is_near_plane() {
        // After shifting to near plane, o_z = -near, so o'_z = 1 + 2n/(-n) = -1.
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("new should succeed");
        let ndc = ndc_ray(&r, 100.0, 200, 200, 1.0).expect("ndc_ray should succeed");
        assert!(
            approx(ndc.origin[2], -1.0, 1e-4),
            "o'_z = {}",
            ndc.origin[2]
        );
    }

    #[test]
    fn ndc_output_is_finite() {
        let r = Ray::normalized([0.2, -0.1, 0.0], [0.1, 0.05, -1.0])
            .expect("normalized should succeed");
        let ndc = ndc_ray(&r, 120.0, 256, 192, 0.5).expect("ndc_ray should succeed");
        assert!(ndc.origin.iter().all(|v| v.is_finite()));
        assert!(ndc.dir.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn ndc_direction_z_negative() {
        // d'_z = -2 n / o_z with o_z = -near < 0 → d'_z = +2 (positive here).
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("new should succeed");
        let ndc = ndc_ray(&r, 100.0, 200, 200, 1.0).expect("ndc_ray should succeed");
        // o_z after shift = -1 → d'_z = -2*1/(-1) = 2
        assert!(approx(ndc.dir[2], 2.0, 1e-4), "d'_z = {}", ndc.dir[2]);
    }

    #[test]
    fn ndc_offset_ray_has_nonzero_xy() {
        // A ray offset in x should map to nonzero NDC x.
        let r =
            Ray::normalized([0.5, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("normalized should succeed");
        let ndc = ndc_ray(&r, 100.0, 200, 200, 1.0).expect("ndc_ray should succeed");
        assert!(ndc.origin[0].abs() > 1e-3, "expected nonzero NDC x");
    }

    #[test]
    fn ndc_invalid_focal_errors() {
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("new should succeed");
        assert!(ndc_ray(&r, 0.0, 200, 200, 1.0).is_err());
        assert!(ndc_ray(&r, -10.0, 200, 200, 1.0).is_err());
    }

    #[test]
    fn ndc_invalid_dims_errors() {
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("new should succeed");
        assert!(ndc_ray(&r, 100.0, 0, 200, 1.0).is_err());
        assert!(ndc_ray(&r, 100.0, 200, 0, 1.0).is_err());
    }

    #[test]
    fn ndc_invalid_near_errors() {
        let r = Ray::new([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]).expect("new should succeed");
        assert!(ndc_ray(&r, 100.0, 200, 200, 0.0).is_err());
        assert!(ndc_ray(&r, 100.0, 200, 200, -1.0).is_err());
    }

    #[test]
    fn ndc_zero_z_direction_errors() {
        // dir[2] == 0 → cannot shift to near plane.
        let r = Ray::new([0.0, 0.0, -2.0], [1.0, 0.0, 0.0]).expect("new should succeed");
        assert!(matches!(
            ndc_ray(&r, 100.0, 200, 200, 1.0),
            Err(NerfError::ZeroRayDirection)
        ));
    }

    #[test]
    fn depth_round_trip_at_near() {
        // t' = 0 → world depth == near.
        let z = ndc_depth_to_world(0.0, 0.5).expect("ndc_depth_to_world should succeed");
        assert!(approx(z, 0.5, 1e-5), "z = {z}");
    }

    #[test]
    fn depth_increases_with_t() {
        let near = 1.0_f32;
        let z_a = ndc_depth_to_world(0.3, near).expect("ndc_depth_to_world should succeed");
        let z_b = ndc_depth_to_world(0.6, near).expect("ndc_depth_to_world should succeed");
        assert!(z_b > z_a, "deeper NDC t → larger world depth");
    }

    #[test]
    fn depth_at_infinity_errors() {
        // t' = 1 maps to infinite depth → error.
        assert!(ndc_depth_to_world(1.0, 1.0).is_err());
        assert!(ndc_depth_to_world(0.5, 0.0).is_err());
    }
}
