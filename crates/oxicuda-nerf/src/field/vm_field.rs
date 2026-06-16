//! TensoRF Vector-Matrix (VM) decomposition radiance field.
//!
//! Chen, Xu, Geng, Yu & Su (2022) "TensoRF: Tensorial Radiance Fields", ECCV 2022.
//!
//! The CP decomposition in [`crate::field::tensorf`] factors the density grid into a
//! sum of rank-one *vector* outer products `v^X ⊗ v^Y ⊗ v^Z`, which is compact but
//! low-capacity. The **vector-matrix (VM)** decomposition is the higher-capacity
//! variant that powers TensoRF's best results (and is structurally the DVGO-style
//! dense factorisation): each rank-one component pairs a 1-D *vector* along one axis
//! with a 2-D *matrix* over the complementary plane,
//!
//! ```text
//! σ(x,y,z) = ReLU( Σ_r  v_r^X(x) · M_r^{YZ}(y,z)
//!                      + v_r^Y(y) · M_r^{XZ}(x,z)
//!                      + v_r^Z(z) · M_r^{XY}(x,y) ) .
//! ```
//!
//! The matrices capture pairwise spatial correlations that a pure CP product cannot,
//! at the cost of `O(R · G²)` parameters per mode instead of `O(R · G)`. Coordinates
//! in `[-1, 1]³` are interpolated linearly along vectors and bilinearly across the
//! plane matrices. A separate set of (vector, matrix) factors with an extra feature
//! axis produces the multi-channel colour features.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

/// Configuration for a TensoRF-VM field.
#[derive(Debug, Clone)]
pub struct VmFieldConfig {
    /// Number of VM rank components `R` (per axis-plane pairing).
    pub rank: usize,
    /// Grid resolution `G` per axis (vectors length `G`, matrices `G × G`).
    pub grid_dim: usize,
    /// Number of colour feature channels.
    pub n_color_feat: usize,
}

/// TensoRF Vector-Matrix radiance field.
#[derive(Debug, Clone)]
pub struct VmField {
    /// Density vectors: 3 axes × `rank` × `grid_dim`, axis-major then rank-major.
    /// Layout `[axis][rank][G]`, flat index `((axis*rank)+r)*G + i`.
    density_vec: Vec<f32>,
    /// Density matrices: 3 planes × `rank` × `grid_dim²`.
    /// Plane order matches the axis it multiplies: axis X ↔ YZ plane, etc.
    density_mat: Vec<f32>,
    /// Colour vectors: 3 axes × `rank` × `grid_dim × n_color_feat`.
    color_vec: Vec<f32>,
    /// Colour matrices: 3 planes × `rank` × `grid_dim²`.
    color_mat: Vec<f32>,
    config: VmFieldConfig,
}

impl VmField {
    /// Create a new TensoRF-VM field with small random initialisation.
    ///
    /// # Errors
    /// Returns [`NerfError::TensorDecompError`] if any dimension is zero.
    pub fn new(cfg: VmFieldConfig, rng: &mut LcgRng) -> NerfResult<Self> {
        if cfg.rank == 0 {
            return Err(NerfError::TensorDecompError {
                msg: "rank must be > 0".into(),
            });
        }
        if cfg.grid_dim == 0 {
            return Err(NerfError::TensorDecompError {
                msg: "grid_dim must be > 0".into(),
            });
        }
        if cfg.n_color_feat == 0 {
            return Err(NerfError::TensorDecompError {
                msg: "n_color_feat must be > 0".into(),
            });
        }

        let g = cfg.grid_dim;
        let r = cfg.rank;
        let nf = cfg.n_color_feat;

        let scale = 0.1_f32;
        let mut draw = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let (a, _) = rng.next_normal_pair();
                    a * scale
                })
                .collect()
        };

        let density_vec = draw(3 * r * g);
        let density_mat = draw(3 * r * g * g);
        let color_vec = draw(3 * r * g * nf);
        let color_mat = draw(3 * r * g * g);

        Ok(Self {
            density_vec,
            density_mat,
            color_vec,
            color_mat,
            config: cfg,
        })
    }

    /// Query density at a point in `[-1, 1]³`.
    ///
    /// Returns `ReLU(Σ_r Σ_axis v_r^axis · M_r^plane)`.
    ///
    /// # Errors
    /// Returns [`NerfError::NanEncountered`] if the result is non-finite.
    pub fn query_density(&self, xyz: [f32; 3]) -> NerfResult<f32> {
        let g = self.config.grid_dim;
        let r = self.config.rank;

        // For each axis a, the complementary plane uses the other two coords.
        // axis 0 (X) ↔ plane (Y, Z); axis 1 (Y) ↔ plane (X, Z); axis 2 (Z) ↔ plane (X, Y).
        let plane_coords = [
            (xyz[1], xyz[2]), // X-vector × YZ-matrix
            (xyz[0], xyz[2]), // Y-vector × XZ-matrix
            (xyz[0], xyz[1]), // Z-vector × XY-matrix
        ];

        let mut sum = 0.0_f32;
        for axis in 0..3 {
            let (pu, pv) = plane_coords[axis];
            for rank_idx in 0..r {
                let vec_base = ((axis * r) + rank_idx) * g;
                let mat_base = ((axis * r) + rank_idx) * g * g;
                let v = interp_vec1(&self.density_vec[vec_base..vec_base + g], xyz[axis]);
                let m = interp_mat2(&self.density_mat[mat_base..mat_base + g * g], g, pu, pv);
                sum += v * m;
            }
        }

        if !sum.is_finite() {
            return Err(NerfError::NanEncountered {
                context: "VmField::query_density".into(),
            });
        }
        Ok(sum.max(0.0))
    }

    /// Query the colour feature vector `[n_color_feat]` at a point in `[-1, 1]³`.
    ///
    /// Each feature channel `k` accumulates
    /// `Σ_r Σ_axis v_r^axis[·, k] · M_r^plane`.
    ///
    /// # Errors
    /// Returns [`NerfError::NanEncountered`] if any output is non-finite.
    pub fn query_color(&self, xyz: [f32; 3]) -> NerfResult<Vec<f32>> {
        let g = self.config.grid_dim;
        let r = self.config.rank;
        let nf = self.config.n_color_feat;

        let plane_coords = [(xyz[1], xyz[2]), (xyz[0], xyz[2]), (xyz[0], xyz[1])];
        let mut out = vec![0.0_f32; nf];

        for axis in 0..3 {
            let (pu, pv) = plane_coords[axis];
            for rank_idx in 0..r {
                let vec_base = ((axis * r) + rank_idx) * g * nf;
                let mat_base = ((axis * r) + rank_idx) * g * g;
                let m = interp_mat2(&self.color_mat[mat_base..mat_base + g * g], g, pu, pv);
                for (k, feat) in out.iter_mut().enumerate() {
                    // Per-feature vector slice: channel k occupies stride nf.
                    let slice = &self.color_vec[vec_base + k..vec_base + k + (g - 1) * nf + 1];
                    let v = interp_vec_strided(slice, nf, xyz[axis]);
                    *feat += v * m;
                }
            }
        }

        if out.iter().any(|v| !v.is_finite()) {
            return Err(NerfError::NanEncountered {
                context: "VmField::query_color".into(),
            });
        }
        Ok(out)
    }

    /// Total number of stored parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.density_vec.len()
            + self.density_mat.len()
            + self.color_vec.len()
            + self.color_mat.len()
    }
}

// ────────────────────────────── interpolation ────────────────────────────────

/// Linear interpolation along a contiguous vector for `coord ∈ [-1, 1]`.
fn interp_vec1(vec: &[f32], coord: f32) -> f32 {
    let g = vec.len();
    if g == 0 {
        return 0.0;
    }
    if g == 1 {
        return vec[0];
    }
    let t = (coord.clamp(-1.0, 1.0) + 1.0) * 0.5 * (g - 1) as f32;
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(g - 1);
    let frac = t - lo as f32;
    vec[lo] * (1.0 - frac) + vec[hi] * frac
}

/// Linear interpolation along a strided vector (`stride` between samples) so that a
/// per-channel slice of an interleaved `[G × nf]` block can be interpolated without
/// a copy. `slice` must cover `(G−1)*stride + 1` elements.
fn interp_vec_strided(slice: &[f32], stride: usize, coord: f32) -> f32 {
    if slice.is_empty() {
        return 0.0;
    }
    let g = (slice.len() - 1) / stride + 1;
    if g == 1 {
        return slice[0];
    }
    let t = (coord.clamp(-1.0, 1.0) + 1.0) * 0.5 * (g - 1) as f32;
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(g - 1);
    let frac = t - lo as f32;
    slice[lo * stride] * (1.0 - frac) + slice[hi * stride] * frac
}

/// Bilinear interpolation in a `g × g` matrix (row-major) for `(u, v) ∈ [-1, 1]²`.
fn interp_mat2(mat: &[f32], g: usize, u: f32, v: f32) -> f32 {
    if g == 0 {
        return 0.0;
    }
    if g == 1 {
        return mat[0];
    }
    let tu = (u.clamp(-1.0, 1.0) + 1.0) * 0.5 * (g - 1) as f32;
    let tv = (v.clamp(-1.0, 1.0) + 1.0) * 0.5 * (g - 1) as f32;
    let u0 = tu.floor() as usize;
    let v0 = tv.floor() as usize;
    let u1 = (u0 + 1).min(g - 1);
    let v1 = (v0 + 1).min(g - 1);
    let fu = tu - u0 as f32;
    let fv = tv - v0 as f32;

    let c00 = mat[u0 * g + v0];
    let c01 = mat[u0 * g + v1];
    let c10 = mat[u1 * g + v0];
    let c11 = mat[u1 * g + v1];

    let c0 = c00 * (1.0 - fv) + c01 * fv;
    let c1 = c10 * (1.0 - fv) + c11 * fv;
    c0 * (1.0 - fu) + c1 * fu
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_field(seed: u64) -> VmField {
        let cfg = VmFieldConfig {
            rank: 4,
            grid_dim: 8,
            n_color_feat: 3,
        };
        let mut rng = LcgRng::new(seed);
        VmField::new(cfg, &mut rng).expect("new should succeed")
    }

    #[test]
    fn construct_param_count() {
        let f = make_field(1);
        // 3*R*G (vec) + 3*R*G² (mat) + 3*R*G*nf (cvec) + 3*R*G² (cmat)
        let g = 8;
        let r = 4;
        let nf = 3;
        let expected = 3 * r * g + 3 * r * g * g + 3 * r * g * nf + 3 * r * g * g;
        assert_eq!(f.param_count(), expected);
    }

    #[test]
    fn density_non_negative() {
        let f = make_field(2);
        let pts: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [0.5, -0.5, 0.5],
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [0.3, 0.7, -0.2],
        ];
        for &p in pts {
            let d = f.query_density(p).expect("query_density should succeed");
            assert!(d >= 0.0, "density {d} negative at {:?}", p);
        }
    }

    #[test]
    fn density_finite() {
        let f = make_field(3);
        for i in 0..10 {
            let t = i as f32 / 9.0 * 2.0 - 1.0;
            let d = f
                .query_density([t, -t, t * 0.5])
                .expect("query_density should succeed");
            assert!(d.is_finite());
        }
    }

    #[test]
    fn color_shape() {
        let f = make_field(4);
        let c = f
            .query_color([0.2, -0.3, 0.4])
            .expect("query_color should succeed");
        assert_eq!(c.len(), 3);
        assert!(c.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn deterministic_queries() {
        let f = make_field(5);
        let a = f
            .query_density([0.1, 0.2, 0.3])
            .expect("query_density should succeed");
        let b = f
            .query_density([0.1, 0.2, 0.3])
            .expect("query_density should succeed");
        assert_eq!(a, b);
        let ca = f
            .query_color([0.1, 0.2, 0.3])
            .expect("query_color should succeed");
        let cb = f
            .query_color([0.1, 0.2, 0.3])
            .expect("query_color should succeed");
        assert_eq!(ca, cb);
    }

    #[test]
    fn different_points_differ() {
        let f = make_field(6);
        let d0 = f
            .query_density([-0.8, -0.8, -0.8])
            .expect("query_density should succeed");
        let d1 = f
            .query_density([0.8, 0.8, 0.8])
            .expect("query_density should succeed");
        // Pre-ReLU sums almost surely differ; if both clamp to 0 the field is
        // degenerate which the random init makes vanishingly unlikely.
        assert!((d0 - d1).abs() > 1e-9 || (d0 == 0.0 && d1 == 0.0));
    }

    #[test]
    fn coords_outside_clamped() {
        // Values outside [-1,1] are clamped, so querying far outside == boundary.
        let f = make_field(7);
        let edge = f
            .query_density([1.0, 1.0, 1.0])
            .expect("query_density should succeed");
        let beyond = f
            .query_density([5.0, 5.0, 5.0])
            .expect("query_density should succeed");
        assert!((edge - beyond).abs() < 1e-5, "{edge} vs {beyond}");
    }

    #[test]
    fn higher_capacity_than_cp_param_count() {
        // VM should have asymptotically more params than CP for the same R,G
        // because of the G² matrices.
        let f = make_field(8);
        let g = 8;
        let r = 4;
        let cp_density = r * 3 * g; // CP density vectors only
        assert!(
            f.param_count() > cp_density,
            "VM must be higher capacity than CP"
        );
    }

    #[test]
    fn rank_zero_errors() {
        let mut rng = LcgRng::new(9);
        let cfg = VmFieldConfig {
            rank: 0,
            grid_dim: 8,
            n_color_feat: 3,
        };
        assert!(VmField::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn grid_zero_errors() {
        let mut rng = LcgRng::new(10);
        let cfg = VmFieldConfig {
            rank: 2,
            grid_dim: 0,
            n_color_feat: 3,
        };
        assert!(VmField::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn color_feat_zero_errors() {
        let mut rng = LcgRng::new(11);
        let cfg = VmFieldConfig {
            rank: 2,
            grid_dim: 8,
            n_color_feat: 0,
        };
        assert!(VmField::new(cfg, &mut rng).is_err());
    }

    #[test]
    fn grid_dim_one_constant() {
        // grid_dim = 1 → vectors and matrices are constants; queries are finite.
        let mut rng = LcgRng::new(12);
        let cfg = VmFieldConfig {
            rank: 2,
            grid_dim: 1,
            n_color_feat: 2,
        };
        let f = VmField::new(cfg, &mut rng).expect("new should succeed");
        let d = f
            .query_density([0.3, -0.4, 0.5])
            .expect("query_density should succeed");
        assert!(d.is_finite() && d >= 0.0);
        let c = f
            .query_color([0.3, -0.4, 0.5])
            .expect("query_color should succeed");
        assert_eq!(c.len(), 2);
    }
}
