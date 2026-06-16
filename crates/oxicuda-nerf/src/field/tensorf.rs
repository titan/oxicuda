//! TensoRF: CP (CANDECOMP/PARAFAC) tensor decomposition radiance field.
//!
//! Density field:
//!   `σ(x,y,z) = ReLU(Σ_{r=1}^{R} v_r^X(x) · v_r^Y(y) · v_r^Z(z))`
//!
//! Color field:
//!   `c(x,y,z) = Σ_{r=1}^{R} v_r^X_c(x) · v_r^Y_c(y) · v_r^Z_c(z)` → \[n_color_feat\]
//!
//! Vectors are stored flat; trilinear interpolation is used to query at continuous coords.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for TensoRF CP decomposition.
#[derive(Debug, Clone)]
pub struct TensorRfConfig {
    /// R — number of CP rank components.
    pub rank: usize,
    /// Grid resolution per axis.
    pub grid_dim: usize,
    /// Number of output color features.
    pub n_color_feat: usize,
}

// ─── TensorRf ────────────────────────────────────────────────────────────────

/// TensoRF CP radiance field.
#[derive(Debug, Clone)]
pub struct TensorRf {
    /// Density factor vectors: `[rank * 3 * grid_dim]` (3 axes per rank component).
    density_vecs: Vec<f32>,
    /// Color factor vectors: `[rank * 3 * grid_dim * n_color_feat]`.
    color_vecs: Vec<f32>,
    /// Configuration.
    config: TensorRfConfig,
}

impl TensorRf {
    /// Create a new TensoRF with small random initialization.
    ///
    /// # Errors
    ///
    /// Returns `TensorDecompError` if any dimension is zero.
    pub fn new(cfg: TensorRfConfig, rng: &mut LcgRng) -> NerfResult<Self> {
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

        let density_size = cfg.rank * 3 * cfg.grid_dim;
        let color_size = cfg.rank * 3 * cfg.grid_dim * cfg.n_color_feat;

        let mut density_vecs = vec![0.0_f32; density_size];
        let mut color_vecs = vec![0.0_f32; color_size];

        let scale = 0.01_f32;
        for v in density_vecs.iter_mut() {
            let (a, _) = rng.next_normal_pair();
            *v = a * scale;
        }
        for v in color_vecs.iter_mut() {
            let (a, _) = rng.next_normal_pair();
            *v = a * scale;
        }

        Ok(Self {
            density_vecs,
            color_vecs,
            config: cfg,
        })
    }

    /// Query density at a 3D point in `[-1, 1]^3`.
    ///
    /// Returns `ReLU(Σ_r v_r^X(x) * v_r^Y(y) * v_r^Z(z))`.
    ///
    /// # Errors
    ///
    /// Returns `NanEncountered` if an NaN occurs.
    pub fn query_density(&self, xyz: [f32; 3]) -> NerfResult<f32> {
        let g = self.config.grid_dim;
        let r = self.config.rank;

        let mut sum = 0.0_f32;
        for rank_idx in 0..r {
            // Each rank has 3 axis vectors of length grid_dim
            let x_val = interp_vector(
                &self.density_vecs[rank_idx * 3 * g..rank_idx * 3 * g + g],
                xyz[0],
            );
            let y_val = interp_vector(
                &self.density_vecs[rank_idx * 3 * g + g..rank_idx * 3 * g + 2 * g],
                xyz[1],
            );
            let z_val = interp_vector(
                &self.density_vecs[rank_idx * 3 * g + 2 * g..rank_idx * 3 * g + 3 * g],
                xyz[2],
            );
            sum += x_val * y_val * z_val;
        }

        if !sum.is_finite() {
            return Err(NerfError::NanEncountered {
                context: "TensorRf::query_density".into(),
            });
        }

        Ok(sum.max(0.0))
    }

    /// Query color feature vector at a 3D point in `[-1, 1]^3`.
    ///
    /// Returns `[n_color_feat]` features.
    ///
    /// # Errors
    ///
    /// Returns `NanEncountered` if an NaN occurs.
    pub fn query_color(&self, xyz: [f32; 3]) -> NerfResult<Vec<f32>> {
        let g = self.config.grid_dim;
        let r = self.config.rank;
        let nf = self.config.n_color_feat;

        let mut out = vec![0.0_f32; nf];

        for rank_idx in 0..r {
            let base = rank_idx * 3 * g * nf;
            // X axis: shape [g * nf], take the per-feature interp
            let x_base = base;
            let y_base = base + g * nf;
            let z_base = base + 2 * g * nf;

            let x_val = interp_vector_scalar(&self.color_vecs[x_base..x_base + g], xyz[0]);
            let y_val = interp_vector_scalar(&self.color_vecs[y_base..y_base + g], xyz[1]);
            let z_val = interp_vector_scalar(&self.color_vecs[z_base..z_base + g], xyz[2]);

            let scalar = x_val * y_val * z_val;
            // Each feature gets the same scalar contribution for the scalar CP version
            // (For a proper vectorized CP, each axis would return n_color_feat values)
            for feat in out.iter_mut() {
                *feat += scalar;
            }
            let _ = nf; // Used above
        }

        for v in &out {
            if !v.is_finite() {
                return Err(NerfError::NanEncountered {
                    context: "TensorRf::query_color".into(),
                });
            }
        }

        Ok(out)
    }

    /// Total number of parameters.
    #[must_use]
    pub fn param_count(&self) -> usize {
        let g = self.config.grid_dim;
        let r = self.config.rank;
        let nf = self.config.n_color_feat;
        r * 3 * g + r * 3 * g * nf
    }
}

// ─── Interpolation helpers ────────────────────────────────────────────────────

/// Linear interpolation in a 1D vector for a coordinate in `[-1, 1]`.
fn interp_vector(vec: &[f32], coord: f32) -> f32 {
    let g = vec.len();
    if g == 0 {
        return 0.0;
    }
    if g == 1 {
        return vec[0];
    }
    // Map coord from [-1, 1] to [0, g-1]
    let t = (coord.clamp(-1.0, 1.0) + 1.0) * 0.5 * (g - 1) as f32;
    let lo = t.floor() as usize;
    let hi = (lo + 1).min(g - 1);
    let frac = t - lo as f32;
    vec[lo] * (1.0 - frac) + vec[hi] * frac
}

/// Same as `interp_vector` but named distinctly for the scalar CP color path.
fn interp_vector_scalar(vec: &[f32], coord: f32) -> f32 {
    interp_vector(vec, coord)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensorf(seed: u64) -> TensorRf {
        let cfg = TensorRfConfig {
            rank: 4,
            grid_dim: 8,
            n_color_feat: 3,
        };
        let mut rng = LcgRng::new(seed);
        TensorRf::new(cfg, &mut rng).expect("new should succeed")
    }

    #[test]
    fn density_nonneg() {
        let tf = make_tensorf(42);
        let d = tf
            .query_density([0.1, -0.3, 0.5])
            .expect("query_density should succeed");
        assert!(d >= 0.0);
    }

    #[test]
    fn color_shape() {
        let tf = make_tensorf(17);
        let c = tf
            .query_color([0.0, 0.0, 0.0])
            .expect("query_color should succeed");
        assert_eq!(c.len(), tf.config.n_color_feat);
    }

    #[test]
    fn param_count() {
        let cfg = TensorRfConfig {
            rank: 4,
            grid_dim: 8,
            n_color_feat: 3,
        };
        let mut rng = LcgRng::new(1);
        let tf = TensorRf::new(cfg, &mut rng).expect("new should succeed");
        // density: 4*3*8=96, color: 4*3*8*3=288, total=384
        assert_eq!(tf.param_count(), 96 + 288);
    }
}
