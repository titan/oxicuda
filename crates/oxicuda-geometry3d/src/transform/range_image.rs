//! LiDAR-style range-image projection.
//!
//! Maps a 3D point cloud to a 2D `(azimuth × elevation)` image whose pixels
//! store the **range** (distance to the origin) of the closest point hashing
//! to that pixel. This is the standard representation used by 360-degree
//! spinning LiDARs (SemanticKITTI, RangeNet++, RangeRCNN, …): a `H × W` matrix
//! that lets convolutional networks process LiDAR scans with a fraction of
//! the cost of voxel- or graph-based pipelines.
//!
//! # Spherical convention
//!
//! A 3D point `p = (x, y, z)` (with `y` being the world up-axis, matching the
//! rest of the crate's yaw/rotation conventions) is projected by:
//!
//! ```text
//! range     = sqrt(x² + y² + z²)
//! azimuth   = atan2(z, x)              ∈ [-π, π]
//! elevation = asin(y / range)          ∈ [-π/2, π/2]
//! ```
//!
//! Pixel coordinates are then:
//!
//! ```text
//! u = floor((azimuth + π) / (2π)        * width )     ∈ [0, width)
//! v = floor((elevation - el_min_rad) / (el_max_rad - el_min_rad) * height )  ∈ [0, height)
//! ```
//!
//! Pixels containing multiple projected points keep the **minimum** range
//! (front-facing surface wins), and pixels with no points are filled with
//! `f32::INFINITY` to act as a clear sentinel for "no return".
//!
//! [`RangeImageProjector::unproject_pixel`] is the analytic inverse: it
//! reconstructs `(x, y, z)` from the centre of pixel `(u, v)` and a range
//! value, so [`RangeImage`] data can round-trip back into world coordinates
//! for visualisation or sensor-aware augmentation.

use crate::error::{Geom3dError, Geom3dResult};

/// Configuration for [`RangeImageProjector`].
///
/// Validation rules (checked in [`RangeImageProjector::new`] and
/// [`RangeImageProjector::project`]):
///
/// * `width >= 1`, `height >= 1` — degenerate dimensions are rejected.
/// * `elev_min_deg < elev_max_deg` — the elevation field-of-view must have
///   positive height.
/// * `elev_max_deg - elev_min_deg <= 180.0` — a single half-sphere cap.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeImageConfig {
    /// Azimuth bin count (width of the resulting image).
    pub width: usize,
    /// Elevation bin count (height of the resulting image).
    pub height: usize,
    /// Minimum elevation in degrees (typically ~−25° for a Velodyne HDL-64).
    pub elev_min_deg: f32,
    /// Maximum elevation in degrees (typically ~+3° for a Velodyne HDL-64).
    pub elev_max_deg: f32,
}

/// Result of a [`RangeImageProjector::project`] call.
///
/// `range` is a flat row-major buffer of length `width * height`. Pixels with
/// no projected point hold `f32::INFINITY`.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeImage {
    /// Per-pixel range values, row-major `[height × width]`.
    pub range: Vec<f32>,
    /// Azimuth bin count.
    pub width: usize,
    /// Elevation bin count.
    pub height: usize,
}

/// Stateless projector that turns 3D point clouds into [`RangeImage`]s and
/// back. The projector stores no per-point data — every call takes the
/// configuration explicitly so the same projector can serve multiple
/// resolutions in a single training step.
#[derive(Debug, Clone)]
pub struct RangeImageProjector;

impl RangeImageProjector {
    /// Validate the configuration and construct a new projector.
    ///
    /// # Errors
    ///
    /// [`Geom3dError::InvalidTopology`] when any of [`RangeImageConfig`]'s
    /// validation rules is violated.
    pub fn new(cfg: RangeImageConfig) -> Geom3dResult<Self> {
        validate_cfg(&cfg)?;
        Ok(Self)
    }

    /// Project `n` 3-D points (`points` is flat row-major `[n × 3]`) into a
    /// range image of the requested resolution.
    ///
    /// Each pixel stores the **minimum** range of the points that fall in it;
    /// empty pixels hold `f32::INFINITY` so downstream code can treat them as
    /// "no return".
    ///
    /// # Errors
    ///
    /// * [`Geom3dError::InvalidTopology`] when `cfg` is invalid.
    /// * [`Geom3dError::DimensionMismatch`] when `points.len() != n * 3`.
    pub fn project(
        &self,
        cfg: &RangeImageConfig,
        points: &[f32],
        n: usize,
    ) -> Geom3dResult<RangeImage> {
        validate_cfg(cfg)?;
        if points.len() != n * 3 {
            return Err(Geom3dError::DimensionMismatch {
                expected: n * 3,
                got: points.len(),
            });
        }
        let total = cfg.width * cfg.height;
        let mut range = vec![f32::INFINITY; total];
        let el_min_rad = cfg.elev_min_deg.to_radians();
        let el_max_rad = cfg.elev_max_deg.to_radians();
        let el_span = el_max_rad - el_min_rad;
        let width = cfg.width;
        let height = cfg.height;

        for i in 0..n {
            let base = i * 3;
            let x = points[base];
            let y = points[base + 1];
            let z = points[base + 2];
            let r = (x * x + y * y + z * z).sqrt();
            if !r.is_finite() {
                continue;
            }
            let az = z.atan2(x);
            let el = if r > 0.0 {
                (y / r).clamp(-1.0, 1.0).asin()
            } else {
                // Point at origin: degenerate; assign to mid pixel.
                0.0
            };

            let u = azimuth_to_bin(az, width);
            let v_opt = elevation_to_bin(el, el_min_rad, el_span, height);
            let Some(v) = v_opt else { continue };
            let idx = v * width + u;
            // Pixel may legitimately be at INFINITY (unfilled); any finite r
            // is strictly smaller.
            if r < range[idx] {
                range[idx] = r;
            }
        }

        Ok(RangeImage {
            range,
            width,
            height,
        })
    }

    /// Inverse of [`RangeImageProjector::project`] for a single pixel.
    ///
    /// Reconstructs the world-coordinate point that projects to the **centre**
    /// of pixel `(u, v)` at the given `range`. Repeatedly composing
    /// `project ∘ unproject_pixel` is a fixed point (modulo binning, see the
    /// `unproject_then_project_roundtrip` test).
    ///
    /// # Errors
    ///
    /// * [`Geom3dError::InvalidTopology`] when `cfg` is invalid.
    /// * [`Geom3dError::DimensionMismatch`] when `u >= width` or
    ///   `v >= height`. (DimensionMismatch is the closest semantic error in
    ///   the crate's error enum for "pixel out of bounds".)
    pub fn unproject_pixel(
        &self,
        cfg: &RangeImageConfig,
        u: usize,
        v: usize,
        range: f32,
    ) -> Geom3dResult<[f32; 3]> {
        validate_cfg(cfg)?;
        if u >= cfg.width {
            return Err(Geom3dError::DimensionMismatch {
                expected: cfg.width,
                got: u,
            });
        }
        if v >= cfg.height {
            return Err(Geom3dError::DimensionMismatch {
                expected: cfg.height,
                got: v,
            });
        }
        if !range.is_finite() || range < 0.0 {
            return Err(Geom3dError::InvalidRadius { radius: range });
        }
        let el_min_rad = cfg.elev_min_deg.to_radians();
        let el_max_rad = cfg.elev_max_deg.to_radians();
        let el_span = el_max_rad - el_min_rad;
        // Bin centre: (u + 0.5) / width gives the centre-of-bin fraction.
        let az = (u as f32 + 0.5) / (cfg.width as f32) * (2.0 * std::f32::consts::PI)
            - std::f32::consts::PI;
        let el = (v as f32 + 0.5) / (cfg.height as f32) * el_span + el_min_rad;
        let cos_el = el.cos();
        let x = range * cos_el * az.cos();
        let y = range * el.sin();
        let z = range * cos_el * az.sin();
        Ok([x, y, z])
    }
}

/// Validate the projector configuration once. Repeated calls are cheap because
/// the function is purely arithmetic — but factoring out validation keeps the
/// per-method error handling uniform.
fn validate_cfg(cfg: &RangeImageConfig) -> Geom3dResult<()> {
    if cfg.width == 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "RangeImage width must be >= 1",
        });
    }
    if cfg.height == 0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "RangeImage height must be >= 1",
        });
    }
    if !cfg.elev_min_deg.is_finite() || !cfg.elev_max_deg.is_finite() {
        return Err(Geom3dError::InvalidTopology {
            reason: "RangeImage elevation bounds must be finite",
        });
    }
    if cfg.elev_min_deg >= cfg.elev_max_deg {
        return Err(Geom3dError::InvalidTopology {
            reason: "RangeImage elev_min_deg must be < elev_max_deg",
        });
    }
    if cfg.elev_max_deg - cfg.elev_min_deg > 180.0 {
        return Err(Geom3dError::InvalidTopology {
            reason: "RangeImage elevation span must be <= 180 degrees",
        });
    }
    Ok(())
}

/// Map an azimuth in `[-π, π]` (with wrap-around) to a bin index in
/// `[0, width)`.
#[inline]
fn azimuth_to_bin(az: f32, width: usize) -> usize {
    let two_pi = 2.0 * std::f32::consts::PI;
    // Wrap into [0, 2π).
    let mut shifted = az + std::f32::consts::PI;
    // Robust wrap for inputs slightly outside [-π, π] (atan2 always returns
    // a value in that range, but defensive wrap costs nothing).
    while shifted < 0.0 {
        shifted += two_pi;
    }
    while shifted >= two_pi {
        shifted -= two_pi;
    }
    let frac = shifted / two_pi;
    let mut bin = (frac * width as f32) as isize;
    if bin < 0 {
        bin = 0;
    }
    let max = width.saturating_sub(1);
    let bin_u = bin as usize;
    bin_u.min(max)
}

/// Map an elevation in radians to a bin in `[0, height)`. Returns `None` if
/// the elevation is outside `[el_min_rad, el_max_rad)` so out-of-FOV points
/// are dropped (matching the LiDAR sensor's physical sweep).
#[inline]
fn elevation_to_bin(el: f32, el_min_rad: f32, el_span: f32, height: usize) -> Option<usize> {
    if !(el_min_rad..(el_min_rad + el_span)).contains(&el) {
        // Allow the exact upper bound to map to the last bin.
        if (el - (el_min_rad + el_span)).abs() < 1e-6 {
            return Some(height.saturating_sub(1));
        }
        return None;
    }
    let frac = (el - el_min_rad) / el_span;
    let bin = (frac * height as f32) as usize;
    Some(bin.min(height.saturating_sub(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference config with HDL-64-like elevation FOV and 64 × 1024 pixels.
    fn reference_cfg() -> RangeImageConfig {
        RangeImageConfig {
            width: 64,
            height: 32,
            elev_min_deg: -25.0,
            elev_max_deg: 3.0,
        }
    }

    #[test]
    fn project_shape_is_width_times_height() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0];
        let img = proj.project(&cfg, &pts, 2).unwrap();
        assert_eq!(img.range.len(), cfg.width * cfg.height);
        assert_eq!(img.width, cfg.width);
        assert_eq!(img.height, cfg.height);
    }

    #[test]
    fn project_empty_cloud_all_infinity() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let img = proj.project(&cfg, &[], 0).unwrap();
        assert!(img.range.iter().all(|&r| r == f32::INFINITY));
        assert_eq!(img.range.len(), cfg.width * cfg.height);
    }

    #[test]
    fn project_single_point_xaxis_lands_at_mid_azimuth() {
        // Point at (1, 0, 0): azimuth = atan2(0, 1) = 0; elevation = 0.
        // With az ∈ [-π, π] mapped to bins [0, width), az = 0 lands at
        // bin = floor((0 + π)/(2π) * width) = floor(width/2) = width/2.
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![1.0_f32, 0.0, 0.0];
        let img = proj.project(&cfg, &pts, 1).unwrap();
        let expected_u = cfg.width / 2;
        let el_min_rad = cfg.elev_min_deg.to_radians();
        let el_max_rad = cfg.elev_max_deg.to_radians();
        let el_span = el_max_rad - el_min_rad;
        let expected_v = ((0.0 - el_min_rad) / el_span * cfg.height as f32) as usize;
        let idx = expected_v * cfg.width + expected_u;
        assert!((img.range[idx] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn project_range_3_4_0_is_5() {
        // (3,4,0): r = sqrt(9+16+0) = 5, atan2(0,3) = 0 → az = 0,
        // elevation = asin(4/5) ≈ 53.13° — needs an elevation FOV that
        // includes that value, so use a custom config.
        let cfg = RangeImageConfig {
            width: 16,
            height: 8,
            elev_min_deg: -90.0,
            elev_max_deg: 90.0,
        };
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![3.0_f32, 4.0, 0.0];
        let img = proj.project(&cfg, &pts, 1).unwrap();
        let min_r = img.range.iter().copied().fold(f32::INFINITY, f32::min);
        assert!((min_r - 5.0).abs() < 1e-4, "got min range {}", min_r);
    }

    #[test]
    fn project_two_points_same_bin_keeps_min_range() {
        // Two points at (1, 0, 0) and (5, 0, 0): same azimuth and elevation,
        // so they share a pixel. The 5-unit one must be replaced by the
        // 1-unit one.
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![5.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let img = proj.project(&cfg, &pts, 2).unwrap();
        let expected_u = cfg.width / 2;
        let el_min_rad = cfg.elev_min_deg.to_radians();
        let el_max_rad = cfg.elev_max_deg.to_radians();
        let el_span = el_max_rad - el_min_rad;
        let expected_v = ((0.0 - el_min_rad) / el_span * cfg.height as f32) as usize;
        let idx = expected_v * cfg.width + expected_u;
        assert!((img.range[idx] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn project_deterministic_for_fixed_input() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts: Vec<f32> = (0..30)
            .flat_map(|i| {
                let f = i as f32;
                vec![f.cos(), f.sin() * 0.1, f.sin()]
            })
            .collect();
        let a = proj.project(&cfg, &pts, 30).unwrap();
        let b = proj.project(&cfg, &pts, 30).unwrap();
        assert_eq!(a, b, "projection must be deterministic");
    }

    #[test]
    fn unproject_then_project_roundtrip() {
        // Pick a config with a full ±90° elevation cap so any (u,v) is valid.
        let cfg = RangeImageConfig {
            width: 64,
            height: 32,
            elev_min_deg: -90.0,
            elev_max_deg: 90.0,
        };
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        // Use distinct (u, v) centres so the projected pixel index matches.
        let cases = [
            (10_usize, 5_usize, 3.5_f32),
            (32, 15, 1.0),
            (50, 20, 4.2),
            (0, 0, 2.0),
            (63, 31, 0.7),
        ];
        for (u, v, r) in cases {
            let p = proj.unproject_pixel(&cfg, u, v, r).unwrap();
            let img = proj.project(&cfg, &p, 1).unwrap();
            let idx = v * cfg.width + u;
            assert!(
                (img.range[idx] - r).abs() < 1e-3,
                "round-trip failed for (u={u}, v={v}, r={r}): pixel range = {}",
                img.range[idx]
            );
        }
    }

    #[test]
    fn project_full_roundtrip_for_clean_point() {
        // Construct a known direction, project it, then unproject the bin to
        // get a point with (approximately) the same range. This tests the
        // dual direction of the round-trip.
        let cfg = RangeImageConfig {
            width: 64,
            height: 32,
            elev_min_deg: -90.0,
            elev_max_deg: 90.0,
        };
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        // Source point at u = 16, v = 16 centre → angle (-π/2 ish), known r.
        let p = proj.unproject_pixel(&cfg, 16, 16, 5.0).unwrap();
        let img = proj.project(&cfg, &p, 1).unwrap();
        let pixel = img.range[16 * cfg.width + 16];
        assert!((pixel - 5.0).abs() < 1e-3, "pixel range {}", pixel);
        // Unproject the projected pixel and confirm range is preserved.
        let q = proj.unproject_pixel(&cfg, 16, 16, pixel).unwrap();
        let r2 = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        assert!((r2 - 5.0).abs() < 1e-3, "unproject range mismatch {}", r2);
    }

    #[test]
    fn err_width_zero() {
        let bad = RangeImageConfig {
            width: 0,
            height: 32,
            elev_min_deg: -25.0,
            elev_max_deg: 3.0,
        };
        assert!(RangeImageProjector::new(bad).is_err());
    }

    #[test]
    fn err_height_zero() {
        let bad = RangeImageConfig {
            width: 16,
            height: 0,
            elev_min_deg: -25.0,
            elev_max_deg: 3.0,
        };
        assert!(RangeImageProjector::new(bad).is_err());
    }

    #[test]
    fn err_elev_range_inverted() {
        let bad = RangeImageConfig {
            width: 16,
            height: 8,
            elev_min_deg: 5.0,
            elev_max_deg: 3.0,
        };
        assert!(RangeImageProjector::new(bad).is_err());
    }

    #[test]
    fn err_elev_range_equal() {
        let bad = RangeImageConfig {
            width: 16,
            height: 8,
            elev_min_deg: 0.0,
            elev_max_deg: 0.0,
        };
        assert!(RangeImageProjector::new(bad).is_err());
    }

    #[test]
    fn err_elev_span_too_large() {
        let bad = RangeImageConfig {
            width: 16,
            height: 8,
            elev_min_deg: -100.0,
            elev_max_deg: 100.0,
        };
        assert!(RangeImageProjector::new(bad).is_err());
    }

    #[test]
    fn err_points_length_mismatch() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![0.0_f32, 0.0, 0.0, 1.0];
        assert!(proj.project(&cfg, &pts, 2).is_err());
    }

    #[test]
    fn err_unproject_u_out_of_range() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        assert!(proj.unproject_pixel(&cfg, cfg.width, 0, 1.0).is_err());
    }

    #[test]
    fn err_unproject_v_out_of_range() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        assert!(proj.unproject_pixel(&cfg, 0, cfg.height, 1.0).is_err());
    }

    #[test]
    fn tiny_resolution_1x1_works() {
        let cfg = RangeImageConfig {
            width: 1,
            height: 1,
            elev_min_deg: -90.0,
            elev_max_deg: 90.0,
        };
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![1.0_f32, 0.0, 0.0, 0.5, 0.5, 0.5];
        let img = proj.project(&cfg, &pts, 2).unwrap();
        assert_eq!(img.range.len(), 1);
        // Both points end up in the only pixel; min range wins.
        let expected_min = 1.0_f32.min((0.5_f32 * 0.5 + 0.25 + 0.25).sqrt());
        assert!((img.range[0] - expected_min).abs() < 1e-4);
    }

    #[test]
    fn negative_depth_points_still_project() {
        // Negative coordinates simply rotate the azimuth (atan2 handles all
        // quadrants), so points "behind" the +x axis still land in valid
        // bins.
        let cfg = RangeImageConfig {
            width: 8,
            height: 4,
            elev_min_deg: -90.0,
            elev_max_deg: 90.0,
        };
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![-1.0_f32, 0.0, 0.0, -2.0, 0.0, -2.0];
        let img = proj.project(&cfg, &pts, 2).unwrap();
        let any_finite = img.range.iter().any(|r| r.is_finite());
        assert!(any_finite, "expected at least one finite pixel");
    }

    #[test]
    fn out_of_fov_elevation_dropped() {
        // Point straight up at (0, 1, 0): elevation = π/2 = 90°. With a
        // restrictive FOV of [-25°, 3°] this is outside and must be dropped
        // — the image must therefore be all-infinity.
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let pts = vec![0.0_f32, 1.0, 0.0];
        let img = proj.project(&cfg, &pts, 1).unwrap();
        assert!(img.range.iter().all(|&r| r == f32::INFINITY));
    }

    #[test]
    fn err_negative_range_unproject() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        assert!(proj.unproject_pixel(&cfg, 0, 0, -1.0).is_err());
    }

    #[test]
    fn unproject_zero_range_at_origin() {
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let p = proj.unproject_pixel(&cfg, 0, 0, 0.0).unwrap();
        assert_eq!(p, [0.0_f32, 0.0, 0.0]);
    }

    #[test]
    fn project_does_not_overflow_for_large_clouds() {
        // Smoke test: a moderately sized random-ish cloud completes without
        // panicking and produces a valid image.
        let cfg = reference_cfg();
        let proj = RangeImageProjector::new(cfg.clone()).unwrap();
        let n = 256;
        let mut pts = Vec::with_capacity(n * 3);
        for i in 0..n {
            let f = i as f32 * 0.1;
            pts.push(f.cos());
            pts.push(0.0);
            pts.push(f.sin());
        }
        let img = proj.project(&cfg, &pts, n).unwrap();
        assert_eq!(img.range.len(), cfg.width * cfg.height);
        let any_finite = img.range.iter().any(|r| r.is_finite());
        assert!(any_finite);
    }
}
