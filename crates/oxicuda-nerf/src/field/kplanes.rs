//! K-Planes: explicit radiance field via factorised coordinate planes.
//!
//! Reference: Fridovich-Keil et al. 2023 CVPR
//! "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance".
//!
//! For a static 3-D scene the volume is factorised into the three coordinate
//! planes `(x, y)`, `(x, z)` and `(y, z)`. A point is queried by projecting it
//! onto each plane, bilinearly interpolating a per-plane feature vector, then
//! combining the three feature vectors with the Hadamard (elementwise) product:
//!
//! ```text
//! f(x, y, z) = f_xy(x, y) ⊙ f_xz(x, z) ⊙ f_yz(y, z)   ∈ ℝ^D
//! ```
//!
//! The fused feature is decoded into density via a linear head + softplus, and
//! into view-dependent colour via a linear head over `[features, SH(view_dir)]`
//! followed by a sigmoid (reusing [`crate::encoding::spherical_harmonics`]).

use crate::encoding::spherical_harmonics::ShEncoder;
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a [`KPlanes`] field.
#[derive(Debug, Clone)]
pub struct KPlanesConfig {
    /// Per-axis grid resolution (must be ≥ 2).
    pub resolution: usize,
    /// Feature-vector dimensionality stored at each grid cell (must be ≥ 1).
    pub feature_dim: usize,
    /// Lower corner of the axis-aligned scene bounds.
    pub bounds_min: [f32; 3],
    /// Upper corner of the axis-aligned scene bounds.
    pub bounds_max: [f32; 3],
    /// Spherical-harmonics degree for view-dependent colour (0 ≤ degree ≤ 4).
    pub sh_degree: usize,
}

// ─── KPlanes ─────────────────────────────────────────────────────────────────

/// K-Planes factorised radiance field over the three coordinate planes.
#[derive(Debug, Clone)]
pub struct KPlanes {
    /// Three feature planes, each `resolution * resolution * feature_dim`.
    ///
    /// Index `[plane_idx]` holds the plane stored row-major as
    /// `(a * resolution + b) * feature_dim + d` where `a` indexes the first
    /// coordinate axis of the plane and `b` the second.
    planes: [Vec<f32>; 3],
    /// Density head weights: `[feature_dim]`.
    density_w: Vec<f32>,
    /// Density head bias.
    density_b: f32,
    /// Colour head weights: `[3 * (feature_dim + n_sh)]` row-major per channel.
    color_w: Vec<f32>,
    /// Colour head bias: `[3]`.
    color_b: [f32; 3],
    /// Number of SH basis functions: `(sh_degree + 1)²`.
    n_sh: usize,
    /// Configuration.
    cfg: KPlanesConfig,
}

impl KPlanes {
    /// Create a new K-Planes field with small random initialisation.
    ///
    /// # Errors
    ///
    /// - [`NerfError::InvalidGridResolution`] if `resolution < 2`.
    /// - [`NerfError::InvalidFeatureDim`] if `feature_dim == 0` or `sh_degree > 4`.
    /// - [`NerfError::InvalidBounds`] if `bounds_min[d] >= bounds_max[d]` for any axis.
    pub fn new(cfg: KPlanesConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.resolution < 2 {
            return Err(NerfError::InvalidGridResolution {
                res: cfg.resolution,
            });
        }
        if cfg.feature_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        if cfg.sh_degree > 4 {
            return Err(NerfError::InvalidFeatureDim { dim: cfg.sh_degree });
        }
        for (&lo, &hi) in cfg.bounds_min.iter().zip(cfg.bounds_max.iter()) {
            if lo >= hi {
                return Err(NerfError::InvalidBounds { near: lo, far: hi });
            }
        }

        let n_sh = ShEncoder::n_coeffs_for_degree(cfg.sh_degree);
        let plane_len = cfg.resolution * cfg.resolution * cfg.feature_dim;

        // Initialise the three planes around 1.0 so the Hadamard product does
        // not collapse to ~0: a small Gaussian perturbation about unity keeps
        // the factorised product well-conditioned at startup.
        let mut make_plane = || -> Vec<f32> {
            let mut p = vec![0.0_f32; plane_len];
            for v in p.iter_mut() {
                let (a, _) = rng.next_normal_pair();
                *v = 1.0 + a * 0.1;
            }
            p
        };
        let planes = [make_plane(), make_plane(), make_plane()];

        // Density head: small Gaussian weights, zero bias.
        let mut density_w = vec![0.0_f32; cfg.feature_dim];
        for w in density_w.iter_mut() {
            let (a, _) = rng.next_normal_pair();
            *w = a * 0.1;
        }

        // Colour head over [features, SH(view_dir)] → 3 channels.
        let color_in = cfg.feature_dim + n_sh;
        let mut color_w = vec![0.0_f32; 3 * color_in];
        let scale = (2.0_f32 / color_in as f32).sqrt();
        for w in color_w.iter_mut() {
            let (a, _) = rng.next_normal_pair();
            *w = a * scale;
        }

        Ok(Self {
            planes,
            density_w,
            density_b: 0.0,
            color_w,
            color_b: [0.0; 3],
            n_sh,
            cfg,
        })
    }

    /// Bilinearly interpolate plane `plane_idx` (0=xy, 1=xz, 2=yz) at the
    /// normalised coordinates `(u, v) ∈ [0, 1]²`.
    ///
    /// `u` and `v` are clamped to `[0, 1]`; the resulting grid position is
    /// clamped to the grid border. Returns a `feature_dim`-length vector.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::Internal`] if `plane_idx > 2`.
    pub fn interpolate_plane(&self, plane_idx: usize, u: f32, v: f32) -> NerfResult<Vec<f32>> {
        if plane_idx > 2 {
            return Err(NerfError::Internal {
                msg: format!("plane_idx {plane_idx} out of range (expected 0..3)"),
            });
        }
        let plane = &self.planes[plane_idx];
        let d = self.cfg.feature_dim;
        let res = self.cfg.resolution;
        let last = res - 1;

        // Map [0, 1] → grid coordinate in [0, res-1].
        let gu = u.clamp(0.0, 1.0) * last as f32;
        let gv = v.clamp(0.0, 1.0) * last as f32;

        let a0 = gu.floor() as usize;
        let b0 = gv.floor() as usize;
        let a1 = (a0 + 1).min(last);
        let b1 = (b0 + 1).min(last);
        let fu = gu - a0 as f32;
        let fv = gv - b0 as f32;

        let w00 = (1.0 - fu) * (1.0 - fv);
        let w01 = (1.0 - fu) * fv;
        let w10 = fu * (1.0 - fv);
        let w11 = fu * fv;

        let base00 = (a0 * res + b0) * d;
        let base01 = (a0 * res + b1) * d;
        let base10 = (a1 * res + b0) * d;
        let base11 = (a1 * res + b1) * d;

        let mut out = vec![0.0_f32; d];
        for (k, o) in out.iter_mut().enumerate() {
            *o = w00 * plane[base00 + k]
                + w01 * plane[base01 + k]
                + w10 * plane[base10 + k]
                + w11 * plane[base11 + k];
        }
        Ok(out)
    }

    /// Normalised `[0, 1]` coordinate of `value` within `[lo, hi]`, clamped.
    #[inline]
    fn normalize_axis(value: f32, lo: f32, hi: f32) -> f32 {
        ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
    }

    /// Query the fused feature vector at `xyz` via the Hadamard product of the
    /// three plane interpolations. Returns a `feature_dim`-length vector.
    ///
    /// Out-of-bounds points are clamped to the scene bounds.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`KPlanes::interpolate_plane`].
    pub fn query_features(&self, xyz: [f32; 3]) -> NerfResult<Vec<f32>> {
        let nx = Self::normalize_axis(xyz[0], self.cfg.bounds_min[0], self.cfg.bounds_max[0]);
        let ny = Self::normalize_axis(xyz[1], self.cfg.bounds_min[1], self.cfg.bounds_max[1]);
        let nz = Self::normalize_axis(xyz[2], self.cfg.bounds_min[2], self.cfg.bounds_max[2]);

        // Plane 0 = xy, plane 1 = xz, plane 2 = yz.
        let f_xy = self.interpolate_plane(0, nx, ny)?;
        let f_xz = self.interpolate_plane(1, nx, nz)?;
        let f_yz = self.interpolate_plane(2, ny, nz)?;

        let mut out = vec![0.0_f32; self.cfg.feature_dim];
        for (k, o) in out.iter_mut().enumerate() {
            *o = f_xy[k] * f_xz[k] * f_yz[k];
        }
        Ok(out)
    }

    /// Query density at `xyz`: `softplus(density_w · features + bias)` (≥ 0).
    ///
    /// # Errors
    ///
    /// - Propagates errors from [`KPlanes::query_features`].
    /// - [`NerfError::NanEncountered`] if a non-finite value is produced.
    pub fn query_density(&self, xyz: [f32; 3]) -> NerfResult<f32> {
        let features = self.query_features(xyz)?;
        let mut acc = self.density_b;
        for (w, f) in self.density_w.iter().zip(features.iter()) {
            acc += w * f;
        }
        let sigma = softplus(acc);
        if !sigma.is_finite() {
            return Err(NerfError::NanEncountered {
                context: "KPlanes::query_density".into(),
            });
        }
        Ok(sigma)
    }

    /// Query view-dependent RGB colour at `xyz` looking along `view_dir`.
    ///
    /// The SH basis is evaluated at the normalised `view_dir`, concatenated
    /// with the fused features, passed through a linear head and a sigmoid,
    /// yielding three RGB values in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// - [`NerfError::ZeroRayDirection`] if `view_dir` is zero / near-zero.
    /// - Propagates errors from [`KPlanes::query_features`] and SH evaluation.
    /// - [`NerfError::NanEncountered`] if a non-finite value is produced.
    pub fn query_color(&self, xyz: [f32; 3], view_dir: [f32; 3]) -> NerfResult<Vec<f32>> {
        let features = self.query_features(xyz)?;

        // Normalise the view direction (guards zero) and evaluate SH basis.
        let unit = ShEncoder::normalize(&view_dir)?;
        let sh = ShEncoder::sh_basis(unit[0], unit[1], unit[2], self.cfg.sh_degree)?;

        let color_in = self.cfg.feature_dim + self.n_sh;
        let mut out = vec![0.0_f32; 3];
        for (ch, o) in out.iter_mut().enumerate() {
            let row = &self.color_w[ch * color_in..(ch + 1) * color_in];
            let mut acc = self.color_b[ch];
            for (w, f) in row.iter().zip(features.iter()) {
                acc += w * f;
            }
            for (w, s) in row[self.cfg.feature_dim..].iter().zip(sh.iter()) {
                acc += w * s;
            }
            let c = sigmoid(acc);
            if !c.is_finite() {
                return Err(NerfError::NanEncountered {
                    context: "KPlanes::query_color".into(),
                });
            }
            *o = c;
        }
        Ok(out)
    }

    /// Total number of learnable parameters.
    ///
    /// `3 · R² · D` (planes) + `D + 1` (density head) + `3 · (D + n_sh) + 3`
    /// (colour head).
    #[must_use]
    pub fn n_params(&self) -> usize {
        let r = self.cfg.resolution;
        let d = self.cfg.feature_dim;
        let plane_params = 3 * r * r * d;
        let density_params = d + 1;
        let color_params = 3 * (d + self.n_sh) + 3;
        plane_params + density_params + color_params
    }

    /// Access the field configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &KPlanesConfig {
        &self.cfg
    }
}

// ─── Activations ───────────────────────────────────────────────────────────────

/// Numerically stable softplus: `log(1 + exp(x))`.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> KPlanesConfig {
        KPlanesConfig {
            resolution: 8,
            feature_dim: 4,
            bounds_min: [-1.0, -1.0, -1.0],
            bounds_max: [1.0, 1.0, 1.0],
            sh_degree: 2,
        }
    }

    fn make_kplanes(seed: u64) -> KPlanes {
        let mut rng = LcgRng::new(seed);
        KPlanes::new(default_cfg(), &mut rng).expect("value should be present")
    }

    /// Overwrite every cell of `plane_idx` with the same `feature` vector.
    fn fill_plane(kp: &mut KPlanes, plane_idx: usize, feature: &[f32]) {
        let res = kp.cfg.resolution;
        let d = kp.cfg.feature_dim;
        for a in 0..res {
            for b in 0..res {
                let base = (a * res + b) * d;
                kp.planes[plane_idx][base..base + d].copy_from_slice(feature);
            }
        }
    }

    /// Set a single grid cell `(a, b)` of `plane_idx` to `feature`.
    fn set_cell(kp: &mut KPlanes, plane_idx: usize, a: usize, b: usize, feature: &[f32]) {
        let res = kp.cfg.resolution;
        let d = kp.cfg.feature_dim;
        let base = (a * res + b) * d;
        kp.planes[plane_idx][base..base + d].copy_from_slice(feature);
    }

    #[test]
    fn interpolate_plane_returns_feature_dim() {
        let kp = make_kplanes(1);
        let f = kp
            .interpolate_plane(0, 0.3, 0.7)
            .expect("interpolate_plane should succeed");
        assert_eq!(f.len(), kp.cfg.feature_dim);
    }

    #[test]
    fn bilinear_at_grid_corner_equals_cell() {
        let mut kp = make_kplanes(2);
        // Set distinct values so the corner can be identified uniquely.
        let target = [9.0_f32, 8.0, 7.0, 6.0];
        // Zero the plane, then place a unique value at cell (3, 5).
        for v in kp.planes[0].iter_mut() {
            *v = 0.0;
        }
        set_cell(&mut kp, 0, 3, 5, &target);
        let last = (kp.cfg.resolution - 1) as f32;
        let u = 3.0 / last;
        let v = 5.0 / last;
        let f = kp
            .interpolate_plane(0, u, v)
            .expect("interpolate_plane should succeed");
        for (got, want) in f.iter().zip(target.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn bilinear_midpoint_is_average_of_four() {
        let mut kp = make_kplanes(3);
        for v in kp.planes[0].iter_mut() {
            *v = 0.0;
        }
        // Four neighbouring cells around the midpoint of (0,0)-(1,1).
        set_cell(&mut kp, 0, 0, 0, &[1.0, 0.0, 0.0, 0.0]);
        set_cell(&mut kp, 0, 0, 1, &[0.0, 1.0, 0.0, 0.0]);
        set_cell(&mut kp, 0, 1, 0, &[0.0, 0.0, 1.0, 0.0]);
        set_cell(&mut kp, 0, 1, 1, &[0.0, 0.0, 0.0, 1.0]);
        // Midpoint between cells 0 and 1 on each axis.
        let last = (kp.cfg.resolution - 1) as f32;
        let u = 0.5 / last;
        let v = 0.5 / last;
        let f = kp
            .interpolate_plane(0, u, v)
            .expect("interpolate_plane should succeed");
        for got in &f {
            assert!((got - 0.25).abs() < 1e-4, "expected 0.25, got {got}");
        }
    }

    #[test]
    fn query_features_length() {
        let kp = make_kplanes(4);
        let f = kp
            .query_features([0.0, 0.0, 0.0])
            .expect("query_features should succeed");
        assert_eq!(f.len(), kp.cfg.feature_dim);
    }

    #[test]
    fn query_features_is_hadamard_product() {
        let mut kp = make_kplanes(5);
        // Constant planes → interpolation returns the constant exactly.
        fill_plane(&mut kp, 0, &[2.0, 3.0, 4.0, 5.0]);
        fill_plane(&mut kp, 1, &[1.0, 2.0, 1.0, 2.0]);
        fill_plane(&mut kp, 2, &[3.0, 1.0, 2.0, 1.0]);
        let f = kp
            .query_features([0.1, -0.2, 0.3])
            .expect("query_features should succeed");
        let expect = [
            2.0 * 1.0 * 3.0,
            3.0 * 2.0 * 1.0,
            4.0 * 1.0 * 2.0,
            5.0 * 2.0 * 1.0,
        ];
        for (got, want) in f.iter().zip(expect.iter()) {
            assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
        }
    }

    #[test]
    fn query_density_non_negative() {
        let kp = make_kplanes(6);
        for &xyz in &[
            [0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5],
            [-0.9, 0.9, -0.1],
            [1.0, 1.0, 1.0],
        ] {
            let d = kp.query_density(xyz).expect("query_density should succeed");
            assert!(d >= 0.0, "density {d} should be >= 0");
        }
    }

    #[test]
    fn query_color_length_and_range() {
        let kp = make_kplanes(7);
        let c = kp
            .query_color([0.1, 0.2, 0.3], [0.0, 0.0, 1.0])
            .expect("query_color should succeed");
        assert_eq!(c.len(), 3);
        for &v in &c {
            assert!((0.0..=1.0).contains(&v), "colour {v} out of [0,1]");
        }
    }

    #[test]
    fn out_of_bounds_is_clamped() {
        let kp = make_kplanes(8);
        // Far outside bounds → must not panic and must equal the clamped border.
        let far = kp
            .query_features([100.0, 100.0, 100.0])
            .expect("query_features should succeed");
        let border = kp
            .query_features([1.0, 1.0, 1.0])
            .expect("query_features should succeed");
        for (a, b) in far.iter().zip(border.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        // Also the negative extreme.
        let _ = kp
            .query_density([-50.0, -50.0, -50.0])
            .expect("query_density should succeed");
    }

    #[test]
    fn deterministic_given_seed() {
        let a = make_kplanes(123);
        let b = make_kplanes(123);
        let fa = a
            .query_features([0.2, 0.3, 0.4])
            .expect("query_features should succeed");
        let fb = b
            .query_features([0.2, 0.3, 0.4])
            .expect("query_features should succeed");
        assert_eq!(fa, fb);
        let da = a
            .query_density([0.2, 0.3, 0.4])
            .expect("query_density should succeed");
        let db = b
            .query_density([0.2, 0.3, 0.4])
            .expect("query_density should succeed");
        assert_eq!(da, db);
    }

    #[test]
    fn n_params_formula() {
        let kp = make_kplanes(9);
        let r = kp.cfg.resolution;
        let d = kp.cfg.feature_dim;
        let n_sh = ShEncoder::n_coeffs_for_degree(kp.cfg.sh_degree);
        let expect = 3 * r * r * d + (d + 1) + (3 * (d + n_sh) + 3);
        assert_eq!(kp.n_params(), expect);
    }

    #[test]
    fn changing_plane_changes_features() {
        let mut kp = make_kplanes(10);
        let before = kp
            .query_features([0.0, 0.0, 0.0])
            .expect("query_features should succeed");
        // Perturb the first cell of plane 0 substantially.
        kp.planes[0][0] += 5.0;
        let after = kp
            .query_features([-1.0, -1.0, -1.0])
            .expect("query_features should succeed");
        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "altering a plane cell should change features");
    }

    #[test]
    fn err_resolution_too_small() {
        let mut rng = LcgRng::new(1);
        let mut cfg = default_cfg();
        cfg.resolution = 1;
        assert!(KPlanes::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_feature_dim_zero() {
        let mut rng = LcgRng::new(1);
        let mut cfg = default_cfg();
        cfg.feature_dim = 0;
        assert!(KPlanes::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_bounds_min_ge_max() {
        let mut rng = LcgRng::new(1);
        let mut cfg = default_cfg();
        cfg.bounds_min[1] = 2.0;
        cfg.bounds_max[1] = 2.0;
        assert!(KPlanes::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_sh_degree_too_large() {
        let mut rng = LcgRng::new(1);
        let mut cfg = default_cfg();
        cfg.sh_degree = 5;
        assert!(KPlanes::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn err_plane_idx_out_of_range() {
        let kp = make_kplanes(11);
        assert!(kp.interpolate_plane(3, 0.5, 0.5).is_err());
    }

    #[test]
    fn err_zero_view_dir() {
        let kp = make_kplanes(12);
        assert!(kp.query_color([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn corner_xyz_queries_succeed() {
        let kp = make_kplanes(13);
        // Both bounds corners must evaluate without error.
        let _ = kp
            .query_density(kp.cfg.bounds_min)
            .expect("query_density should succeed");
        let _ = kp
            .query_density(kp.cfg.bounds_max)
            .expect("query_density should succeed");
        let c = kp
            .query_color(kp.cfg.bounds_max, [1.0, 0.0, 0.0])
            .expect("query_color should succeed");
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn two_distinct_points_distinct_features() {
        let kp = make_kplanes(14);
        let f1 = kp
            .query_features([-0.8, -0.6, -0.4])
            .expect("query_features should succeed");
        let f2 = kp
            .query_features([0.4, 0.6, 0.8])
            .expect("query_features should succeed");
        let differ = f1.iter().zip(f2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(differ, "distinct points should yield distinct features");
    }

    #[test]
    fn sh_degree_zero_color_valid() {
        // sh_degree = 0 must still produce valid RGB.
        let mut cfg = default_cfg();
        cfg.sh_degree = 0;
        let mut rng = LcgRng::new(15);
        let kp = KPlanes::new(cfg, &mut rng).expect("new should succeed");
        let c = kp
            .query_color([0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
            .expect("query_color should succeed");
        assert_eq!(c.len(), 3);
        for &v in &c {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn config_accessor_matches() {
        let kp = make_kplanes(16);
        assert_eq!(kp.config().resolution, 8);
        assert_eq!(kp.config().feature_dim, 4);
    }
}
