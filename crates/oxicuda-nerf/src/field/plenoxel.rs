//! Plenoxels: radiance fields without neural networks.
//!
//! Reference: Yu et al. 2022 CVPR
//! "Plenoxels: Radiance Fields without Neural Networks".
//!
//! A dense voxel grid stores, per voxel, a scalar density and a set of
//! spherical-harmonic colour coefficients (three channels × `n_sh` basis
//! functions). A continuous query trilinearly interpolates the eight
//! surrounding voxels and evaluates the interpolated SH coefficients at the
//! view direction — there is **no** neural network anywhere in the pipeline.
//!
//! Density activation: a [`ReLU`](f32::max) is applied so that an
//! unset (zero-initialised) grid returns exactly `0` density, which is the
//! natural convention for an empty Plenoxel volume.

use crate::encoding::spherical_harmonics::ShEncoder;
use crate::error::{NerfError, NerfResult};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a [`PlenoxelGrid`].
#[derive(Debug, Clone)]
pub struct PlenoxelConfig {
    /// Per-axis voxel resolution (must be ≥ 2).
    pub resolution: usize,
    /// Spherical-harmonics degree for view-dependent colour (0 ≤ degree ≤ 4).
    pub sh_degree: usize,
    /// Lower corner of the axis-aligned scene bounds.
    pub bounds_min: [f32; 3],
    /// Upper corner of the axis-aligned scene bounds.
    pub bounds_max: [f32; 3],
}

// ─── PlenoxelGrid ──────────────────────────────────────────────────────────────

/// Dense Plenoxel voxel grid with per-voxel density and SH colour coefficients.
#[derive(Debug, Clone)]
pub struct PlenoxelGrid {
    /// Per-voxel density, length `resolution³`.
    density: Vec<f32>,
    /// Per-voxel SH coefficients, length `resolution³ * (3 * n_sh)`.
    ///
    /// Within a voxel the layout is channel-major:
    /// `[channel * n_sh + basis_idx]`, channel ∈ {R, G, B}.
    sh_coeffs: Vec<f32>,
    /// Number of SH basis functions per channel: `(sh_degree + 1)²`.
    n_sh: usize,
    /// Configuration.
    cfg: PlenoxelConfig,
}

impl PlenoxelGrid {
    /// Create a new zero-initialised Plenoxel grid.
    ///
    /// # Errors
    ///
    /// - [`NerfError::InvalidGridResolution`] if `resolution < 2`.
    /// - [`NerfError::InvalidFeatureDim`] if `sh_degree > 4`.
    /// - [`NerfError::InvalidBounds`] if `bounds_min[d] >= bounds_max[d]` for any axis.
    pub fn new(cfg: PlenoxelConfig) -> NerfResult<Self> {
        if cfg.resolution < 2 {
            return Err(NerfError::InvalidGridResolution {
                res: cfg.resolution,
            });
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
        let n_voxels = cfg.resolution * cfg.resolution * cfg.resolution;
        let density = vec![0.0_f32; n_voxels];
        let sh_coeffs = vec![0.0_f32; n_voxels * 3 * n_sh];

        Ok(Self {
            density,
            sh_coeffs,
            n_sh,
            cfg,
        })
    }

    /// Number of SH basis functions per colour channel: `(sh_degree + 1)²`.
    #[must_use]
    #[inline]
    pub fn n_sh_per_channel(&self) -> usize {
        self.n_sh
    }

    /// Linear index of voxel `(i, j, k)`: `(i · R + j) · R + k`.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidGridResolution`] if any index is `>= resolution`.
    pub fn voxel_index(&self, i: usize, j: usize, k: usize) -> NerfResult<usize> {
        let res = self.cfg.resolution;
        if i >= res || j >= res || k >= res {
            return Err(NerfError::InvalidGridResolution { res });
        }
        Ok((i * res + j) * res + k)
    }

    /// Set the density and SH coefficients of voxel `(i, j, k)`.
    ///
    /// `sh_coeffs` must have length `3 * n_sh_per_channel()`.
    ///
    /// # Errors
    ///
    /// - [`NerfError::InvalidGridResolution`] via [`Self::voxel_index`] if out of range.
    /// - [`NerfError::DimensionMismatch`] if `sh_coeffs.len() != 3 * n_sh`.
    pub fn set_voxel(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        density: f32,
        sh_coeffs: &[f32],
    ) -> NerfResult<()> {
        let expected = 3 * self.n_sh;
        if sh_coeffs.len() != expected {
            return Err(NerfError::DimensionMismatch {
                expected,
                got: sh_coeffs.len(),
            });
        }
        let idx = self.voxel_index(i, j, k)?;
        self.density[idx] = density;
        let base = idx * expected;
        self.sh_coeffs[base..base + expected].copy_from_slice(sh_coeffs);
        Ok(())
    }

    /// Normalised `[0, 1]` coordinate of `value` within `[lo, hi]`, clamped.
    #[inline]
    fn normalize_axis(value: f32, lo: f32, hi: f32) -> f32 {
        ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
    }

    /// Compute the eight trilinear corner indices and weights for `xyz`.
    ///
    /// Returns `([8 voxel indices], [8 weights])`. Grid positions are clamped
    /// to the voxel-grid border.
    fn trilinear_setup(&self, xyz: [f32; 3]) -> ([usize; 8], [f32; 8]) {
        let res = self.cfg.resolution;
        let last = res - 1;

        let nx = Self::normalize_axis(xyz[0], self.cfg.bounds_min[0], self.cfg.bounds_max[0]);
        let ny = Self::normalize_axis(xyz[1], self.cfg.bounds_min[1], self.cfg.bounds_max[1]);
        let nz = Self::normalize_axis(xyz[2], self.cfg.bounds_min[2], self.cfg.bounds_max[2]);

        let gx = nx * last as f32;
        let gy = ny * last as f32;
        let gz = nz * last as f32;

        let i0 = gx.floor() as usize;
        let j0 = gy.floor() as usize;
        let k0 = gz.floor() as usize;
        let i1 = (i0 + 1).min(last);
        let j1 = (j0 + 1).min(last);
        let k1 = (k0 + 1).min(last);

        let fx = gx - i0 as f32;
        let fy = gy - j0 as f32;
        let fz = gz - k0 as f32;

        // Corner ordering: bit 0 = i, bit 1 = j, bit 2 = k.
        let lin = |i: usize, j: usize, k: usize| (i * res + j) * res + k;
        let indices = [
            lin(i0, j0, k0),
            lin(i1, j0, k0),
            lin(i0, j1, k0),
            lin(i1, j1, k0),
            lin(i0, j0, k1),
            lin(i1, j0, k1),
            lin(i0, j1, k1),
            lin(i1, j1, k1),
        ];
        let weights = [
            (1.0 - fx) * (1.0 - fy) * (1.0 - fz),
            fx * (1.0 - fy) * (1.0 - fz),
            (1.0 - fx) * fy * (1.0 - fz),
            fx * fy * (1.0 - fz),
            (1.0 - fx) * (1.0 - fy) * fz,
            fx * (1.0 - fy) * fz,
            (1.0 - fx) * fy * fz,
            fx * fy * fz,
        ];
        (indices, weights)
    }

    /// Trilinearly interpolate the per-voxel density at `xyz`.
    ///
    /// Out-of-bounds points are clamped to the grid border. The raw
    /// interpolated value is returned (no activation).
    ///
    /// # Errors
    ///
    /// [`NerfError::NanEncountered`] if a non-finite value is produced.
    pub fn trilinear_density(&self, xyz: [f32; 3]) -> NerfResult<f32> {
        let (indices, weights) = self.trilinear_setup(xyz);
        let mut acc = 0.0_f32;
        for (idx, w) in indices.iter().zip(weights.iter()) {
            acc += w * self.density[*idx];
        }
        if !acc.is_finite() {
            return Err(NerfError::NanEncountered {
                context: "PlenoxelGrid::trilinear_density".into(),
            });
        }
        Ok(acc)
    }

    /// Trilinearly interpolate the per-voxel SH coefficient vectors at `xyz`.
    ///
    /// Returns a vector of length `3 * n_sh_per_channel()`. Out-of-bounds
    /// points are clamped to the grid border.
    ///
    /// # Errors
    ///
    /// [`NerfError::NanEncountered`] if a non-finite value is produced.
    pub fn trilinear_sh(&self, xyz: [f32; 3]) -> NerfResult<Vec<f32>> {
        let stride = 3 * self.n_sh;
        let (indices, weights) = self.trilinear_setup(xyz);
        let mut out = vec![0.0_f32; stride];
        for (idx, w) in indices.iter().zip(weights.iter()) {
            let base = idx * stride;
            let coeffs = &self.sh_coeffs[base..base + stride];
            for (o, c) in out.iter_mut().zip(coeffs.iter()) {
                *o += w * c;
            }
        }
        for v in &out {
            if !v.is_finite() {
                return Err(NerfError::NanEncountered {
                    context: "PlenoxelGrid::trilinear_sh".into(),
                });
            }
        }
        Ok(out)
    }

    /// Query density at `xyz`: `ReLU(trilinear_density)` (≥ 0).
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::trilinear_density`].
    pub fn query_density(&self, xyz: [f32; 3]) -> NerfResult<f32> {
        Ok(self.trilinear_density(xyz)?.max(0.0))
    }

    /// Query view-dependent RGB colour at `xyz` looking along `view_dir`.
    ///
    /// The SH coefficients are trilinearly interpolated, the SH basis is
    /// evaluated at the normalised `view_dir`, and each channel sums
    /// `coeff · basis` before a sigmoid maps it to `[0, 1]`.
    ///
    /// # Errors
    ///
    /// - [`NerfError::ZeroRayDirection`] if `view_dir` is zero / near-zero.
    /// - Propagates errors from [`Self::trilinear_sh`] and SH evaluation.
    /// - [`NerfError::NanEncountered`] if a non-finite value is produced.
    pub fn query_color(&self, xyz: [f32; 3], view_dir: [f32; 3]) -> NerfResult<Vec<f32>> {
        let coeffs = self.trilinear_sh(xyz)?;

        let unit = ShEncoder::normalize(&view_dir)?;
        let basis = ShEncoder::sh_basis(unit[0], unit[1], unit[2], self.cfg.sh_degree)?;

        let n_sh = self.n_sh;
        let mut out = vec![0.0_f32; 3];
        for (ch, o) in out.iter_mut().enumerate() {
            let chan = &coeffs[ch * n_sh..(ch + 1) * n_sh];
            let mut acc = 0.0_f32;
            for (c, b) in chan.iter().zip(basis.iter()) {
                acc += c * b;
            }
            let color = sigmoid(acc);
            if !color.is_finite() {
                return Err(NerfError::NanEncountered {
                    context: "PlenoxelGrid::query_color".into(),
                });
            }
            *o = color;
        }
        Ok(out)
    }

    /// Access the grid configuration.
    #[must_use]
    #[inline]
    pub fn config(&self) -> &PlenoxelConfig {
        &self.cfg
    }
}

// ─── Activations ───────────────────────────────────────────────────────────────

/// Logistic sigmoid: `1 / (1 + exp(-x))`.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> PlenoxelConfig {
        PlenoxelConfig {
            resolution: 4,
            sh_degree: 1,
            bounds_min: [-1.0, -1.0, -1.0],
            bounds_max: [1.0, 1.0, 1.0],
        }
    }

    fn make_grid() -> PlenoxelGrid {
        PlenoxelGrid::new(default_cfg()).unwrap()
    }

    /// Normalised coordinate of voxel index `i` along an axis with `res` voxels.
    fn voxel_coord(i: usize, res: usize, lo: f32, hi: f32) -> f32 {
        let t = i as f32 / (res - 1) as f32;
        lo + t * (hi - lo)
    }

    #[test]
    fn new_grid_density_zero_everywhere() {
        let g = make_grid();
        for &xyz in &[
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [0.33, -0.5, 0.7],
            [1.0, 1.0, 1.0],
        ] {
            assert_eq!(g.query_density(xyz).unwrap(), 0.0);
        }
    }

    #[test]
    fn set_voxel_then_density_at_center() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let sh = vec![0.0_f32; n];
        g.set_voxel(2, 1, 3, 0.75, &sh).unwrap();
        // The exact voxel center maps to the integer grid point (2,1,3).
        let res = g.config().resolution;
        let bmin = g.config().bounds_min;
        let bmax = g.config().bounds_max;
        let xyz = [
            voxel_coord(2, res, bmin[0], bmax[0]),
            voxel_coord(1, res, bmin[1], bmax[1]),
            voxel_coord(3, res, bmin[2], bmax[2]),
        ];
        let d = g.trilinear_density(xyz).unwrap();
        assert!((d - 0.75).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn voxel_index_formula() {
        let g = make_grid();
        let res = g.config().resolution;
        assert_eq!(g.voxel_index(1, 2, 3).unwrap(), (res + 2) * res + 3);
        assert_eq!(g.voxel_index(0, 0, 0).unwrap(), 0);
    }

    #[test]
    fn trilinear_midpoint_is_average() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let sh = vec![0.0_f32; n];
        // Two adjacent voxels along i with densities 1.0 and 3.0.
        g.set_voxel(1, 1, 1, 1.0, &sh).unwrap();
        g.set_voxel(2, 1, 1, 3.0, &sh).unwrap();
        let res = g.config().resolution;
        let bmin = g.config().bounds_min;
        let bmax = g.config().bounds_max;
        // Midpoint between voxel i=1 and i=2 (j,k fixed at their grid points).
        let c1 = voxel_coord(1, res, bmin[0], bmax[0]);
        let c2 = voxel_coord(2, res, bmin[0], bmax[0]);
        let xyz = [
            0.5 * (c1 + c2),
            voxel_coord(1, res, bmin[1], bmax[1]),
            voxel_coord(1, res, bmin[2], bmax[2]),
        ];
        let d = g.trilinear_density(xyz).unwrap();
        assert!((d - 2.0).abs() < 1e-4, "expected 2.0, got {d}");
    }

    #[test]
    fn trilinear_sh_length() {
        let g = make_grid();
        let sh = g.trilinear_sh([0.1, 0.2, 0.3]).unwrap();
        assert_eq!(sh.len(), 3 * g.n_sh_per_channel());
    }

    #[test]
    fn query_color_length_and_range() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        // Non-trivial coefficients so the sigmoid output is not exactly 0.5.
        let mut sh = vec![0.0_f32; n];
        for (idx, v) in sh.iter_mut().enumerate() {
            *v = 0.5 * (idx as f32 + 1.0);
        }
        for i in 0..g.config().resolution {
            for j in 0..g.config().resolution {
                for k in 0..g.config().resolution {
                    g.set_voxel(i, j, k, 1.0, &sh).unwrap();
                }
            }
        }
        let c = g.query_color([0.1, 0.2, 0.3], [0.0, 0.0, 1.0]).unwrap();
        assert_eq!(c.len(), 3);
        for &v in &c {
            assert!((0.0..=1.0).contains(&v), "colour {v} out of [0,1]");
        }
    }

    #[test]
    fn n_sh_per_channel_formula() {
        let g = make_grid();
        // sh_degree = 1 → (1+1)² = 4.
        assert_eq!(g.n_sh_per_channel(), 4);

        let cfg2 = PlenoxelConfig {
            resolution: 2,
            sh_degree: 3,
            bounds_min: [0.0, 0.0, 0.0],
            bounds_max: [1.0, 1.0, 1.0],
        };
        let g2 = PlenoxelGrid::new(cfg2).unwrap();
        assert_eq!(g2.n_sh_per_channel(), 16);
    }

    #[test]
    fn set_voxel_wrong_coeff_length_err() {
        let mut g = make_grid();
        let wrong = vec![0.0_f32; 3 * g.n_sh_per_channel() + 1];
        assert!(g.set_voxel(0, 0, 0, 1.0, &wrong).is_err());
    }

    #[test]
    fn voxel_index_out_of_range_err() {
        let g = make_grid();
        let res = g.config().resolution;
        assert!(g.voxel_index(res, 0, 0).is_err());
        assert!(g.voxel_index(0, res, 0).is_err());
        assert!(g.voxel_index(0, 0, res).is_err());
    }

    #[test]
    fn out_of_bounds_xyz_clamped() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let sh = vec![0.0_f32; n];
        let last = g.config().resolution - 1;
        g.set_voxel(last, last, last, 2.0, &sh).unwrap();
        // Far beyond bounds clamps to the (last,last,last) corner voxel.
        let far = g.trilinear_density([100.0, 100.0, 100.0]).unwrap();
        assert!((far - 2.0).abs() < 1e-4, "got {far}");
        // Negative extreme must not panic either.
        let _ = g.query_density([-100.0, -100.0, -100.0]).unwrap();
    }

    #[test]
    fn deterministic_queries() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let sh = vec![0.2_f32; n];
        g.set_voxel(1, 1, 1, 0.5, &sh).unwrap();
        let a = g.query_color([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        let b = g.query_color([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn sh_eval_matches_encoding_dc_term() {
        // The interpolated DC coefficient times Y_0^0 must agree with the
        // shared encoding routine before the sigmoid is applied.
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let mut sh = vec![0.0_f32; n];
        // Set only the DC (basis index 0) of channel 0 across the whole grid.
        sh[0] = 2.0; // channel 0, basis 0
        for i in 0..g.config().resolution {
            for j in 0..g.config().resolution {
                for k in 0..g.config().resolution {
                    g.set_voxel(i, j, k, 1.0, &sh).unwrap();
                }
            }
        }
        let basis = ShEncoder::sh_basis(0.0, 0.0, 1.0, g.config().sh_degree).unwrap();
        let expected_pre_sigmoid = 2.0_f32 * basis[0];
        let expected = sigmoid(expected_pre_sigmoid);
        let c = g.query_color([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap();
        assert!(
            (c[0] - expected).abs() < 1e-4,
            "got {}, want {expected}",
            c[0]
        );
    }

    #[test]
    fn corner_voxel_set_and_query() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let sh = vec![0.0_f32; n];
        g.set_voxel(0, 0, 0, 1.5, &sh).unwrap();
        let d = g.query_density([-1.0, -1.0, -1.0]).unwrap();
        assert!((d - 1.5).abs() < 1e-4, "got {d}");
    }

    #[test]
    fn err_resolution_too_small() {
        let mut cfg = default_cfg();
        cfg.resolution = 1;
        assert!(PlenoxelGrid::new(cfg).is_err());
    }

    #[test]
    fn err_bounds_min_ge_max() {
        let mut cfg = default_cfg();
        cfg.bounds_min[2] = 1.0;
        cfg.bounds_max[2] = 1.0;
        assert!(PlenoxelGrid::new(cfg).is_err());
    }

    #[test]
    fn err_sh_degree_too_large() {
        let mut cfg = default_cfg();
        cfg.sh_degree = 5;
        assert!(PlenoxelGrid::new(cfg).is_err());
    }

    #[test]
    fn err_zero_view_dir() {
        let g = make_grid();
        assert!(g.query_color([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn two_set_voxels_interpolate_distinctly() {
        let mut g = make_grid();
        let n = 3 * g.n_sh_per_channel();
        let sh = vec![0.0_f32; n];
        g.set_voxel(0, 0, 0, 4.0, &sh).unwrap();
        g.set_voxel(3, 3, 3, 0.0, &sh).unwrap();
        // Point near the dense corner should read higher than the far corner.
        let near_dense = g.query_density([-0.9, -0.9, -0.9]).unwrap();
        let near_empty = g.query_density([0.9, 0.9, 0.9]).unwrap();
        assert!(
            near_dense > near_empty,
            "near={near_dense} should exceed far={near_empty}"
        );
    }
}
