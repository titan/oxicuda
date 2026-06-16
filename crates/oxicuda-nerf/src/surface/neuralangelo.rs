//! Neuralangelo — high-fidelity neural surface reconstruction.
//!
//! Li, Müller, Morrison, Vaccaro, Marshak, Speierer, Müller & Lin (2023),
//! "Neuralangelo: High-Fidelity Neural Surface Reconstruction", CVPR.
//!
//! Neuralangelo learns a signed-distance field (SDF) on top of an Instant-NGP
//! multiresolution hash grid. Two ingredients make the hash grid usable for
//! smooth surface reconstruction; both are implemented here as faithful CPU
//! cores:
//!
//! 1. **Numerical gradients.** Analytic autodiff gradients of a hash-grid SDF
//!    are *local* to a single trilinear cell, so the gradient is constant inside
//!    each cell and discontinuous across cells — useless for the Eikonal and
//!    curvature regularisers. Neuralangelo's key trick is to estimate `∇f` with
//!    central finite differences using a step `ε` that spans neighbouring grid
//!    cells, coupling gradients across the grid.
//! 2. **Coarse-to-fine optimisation.** Hash-grid levels are activated
//!    progressively over training: only the coarsest levels are visible at the
//!    start, and finer levels switch on (with a smooth ramp) as optimisation
//!    proceeds. The numerical-gradient step `ε` is annealed in lock-step with
//!    the finest active level's cell size.
//!
//! In addition, a **curvature** regulariser penalises the surface's mean
//! curvature, computed as the discrete Laplacian (trace of the Hessian) reusing
//! the same `±ε` SDF samples that produced the gradient. For a unit-gradient SDF
//! the Laplacian equals the sum of principal curvatures.
//!
//! The SDF itself is a geometric-initialised residual field: a base sphere plus
//! a small hash-grid + MLP correction, so `f` is a genuine (approximate) SDF.
//! The finite-difference operators are exposed as free functions and validated
//! against analytic SDFs in the tests.

use crate::encoding::hash_grid::{HashGrid, HashGridConfig};
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;

/// Scale applied to the learned hash-grid SDF residual on top of the geometric
/// (sphere) initialisation.
const RESIDUAL_SCALE: f32 = 0.1;

// ─── Generic finite-difference operators ─────────────────────────────────────

/// Shift `x` by `delta` along a single `axis`.
#[inline]
fn shifted(x: [f32; 3], axis: usize, delta: f32) -> [f32; 3] {
    let mut y = x;
    y[axis] += delta;
    y
}

/// Central-difference numerical gradient of a scalar field `f` at `x`,
/// `∂f/∂x_i ≈ (f(x + ε e_i) − f(x − ε e_i)) / (2ε)`.
///
/// This is Neuralangelo's numerical-gradient operator; the same machinery is
/// exposed generically so it can be applied to any SDF.
pub fn numerical_gradient_of<F>(f: F, x: [f32; 3], eps: f32) -> [f32; 3]
where
    F: Fn([f32; 3]) -> f32,
{
    let inv = 1.0 / (2.0 * eps);
    let mut grad = [0.0_f32; 3];
    for (axis, component) in grad.iter_mut().enumerate() {
        *component = (f(shifted(x, axis, eps)) - f(shifted(x, axis, -eps))) * inv;
    }
    grad
}

/// Discrete Laplacian (trace of the Hessian) of `f` at `x`,
/// `∇²f ≈ Σ_i (f(x + ε e_i) + f(x − ε e_i) − 2 f(x)) / ε²`.
///
/// For a unit-gradient SDF this equals the sum of principal curvatures, so it is
/// the curvature quantity Neuralangelo regularises.
pub fn laplacian_of<F>(f: F, x: [f32; 3], eps: f32) -> f32
where
    F: Fn([f32; 3]) -> f32,
{
    let f0 = f(x);
    let inv = 1.0 / (eps * eps);
    let mut lap = 0.0_f32;
    for axis in 0..3 {
        lap += (f(shifted(x, axis, eps)) + f(shifted(x, axis, -eps)) - 2.0 * f0) * inv;
    }
    lap
}

/// Eikonal residual `|∇f| − 1` of `f` at `x`, estimated with central
/// differences. Zero for an ideal SDF.
pub fn eikonal_residual_of<F>(f: F, x: [f32; 3], eps: f32) -> f32
where
    F: Fn([f32; 3]) -> f32,
{
    let grad = numerical_gradient_of(f, x, eps);
    (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt() - 1.0
}

// ─── SDF decoder ─────────────────────────────────────────────────────────────

/// Tiny two-layer MLP mapping hash-grid features to a scalar SDF residual.
#[derive(Debug, Clone)]
struct SdfDecoder {
    w0: Vec<f32>,
    b0: Vec<f32>,
    w1: Vec<f32>,
    b1: f32,
    in_dim: usize,
    hidden: usize,
}

impl SdfDecoder {
    fn new(in_dim: usize, hidden: usize, rng: &mut LcgRng) -> Self {
        let mut w0 = vec![0.0_f32; hidden * in_dim];
        let mut w1 = vec![0.0_f32; hidden];
        xavier_fill(&mut w0, in_dim, rng);
        xavier_fill(&mut w1, hidden, rng);
        Self {
            w0,
            b0: vec![0.0_f32; hidden],
            w1,
            b1: 0.0,
            in_dim,
            hidden,
        }
    }

    fn forward(&self, feat: &[f32]) -> f32 {
        let mut hidden = vec![0.0_f32; self.hidden];
        for (h_val, (row, &bias)) in hidden
            .iter_mut()
            .zip(self.w0.chunks(self.in_dim).zip(self.b0.iter()))
        {
            let dot: f32 = row.iter().zip(feat.iter()).map(|(&w, &x)| w * x).sum();
            *h_val = (dot + bias).max(0.0);
        }
        self.w1
            .iter()
            .zip(hidden.iter())
            .map(|(&w, &h)| w * h)
            .sum::<f32>()
            + self.b1
    }
}

fn xavier_fill(buf: &mut [f32], fan_in: usize, rng: &mut LcgRng) {
    let scale = (2.0_f32 / fan_in.max(1) as f32).sqrt();
    let mut idx = 0;
    while idx + 1 < buf.len() {
        let (a, b) = rng.next_normal_pair();
        buf[idx] = a * scale;
        buf[idx + 1] = b * scale;
        idx += 2;
    }
    if idx < buf.len() {
        let (a, _) = rng.next_normal_pair();
        buf[idx] = a * scale;
    }
}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Coarse-to-fine schedule and geometry configuration.
#[derive(Debug, Clone)]
pub struct NeuralangeloConfig {
    /// Number of hash-grid levels active from iteration 0.
    pub init_active_levels: usize,
    /// Iterations between activating successive finer levels.
    pub steps_per_level: usize,
    /// Lower corner of the scene bounding box (world → `[0,1]^3`).
    pub aabb_min: [f32; 3],
    /// Upper corner of the scene bounding box.
    pub aabb_max: [f32; 3],
    /// Hidden width of the SDF decoder MLP.
    pub hidden_dim: usize,
    /// Radius of the geometric-initialisation sphere (world units).
    pub sphere_radius: f32,
}

impl Default for NeuralangeloConfig {
    fn default() -> Self {
        Self {
            init_active_levels: 4,
            steps_per_level: 500,
            aabb_min: [-1.0, -1.0, -1.0],
            aabb_max: [1.0, 1.0, 1.0],
            hidden_dim: 32,
            sphere_radius: 0.5,
        }
    }
}

// ─── Neuralangelo ────────────────────────────────────────────────────────────

/// Neuralangelo SDF model: geometric-initialised hash-grid field with
/// numerical gradients, coarse-to-fine level scheduling and curvature support.
#[derive(Debug, Clone)]
pub struct Neuralangelo {
    grid: HashGrid,
    decoder: SdfDecoder,
    config: NeuralangeloConfig,
    extent: [f32; 3],
    center: [f32; 3],
    iteration: usize,
}

impl Neuralangelo {
    /// Build a Neuralangelo model from a hash-grid config and a schedule config.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::Internal`] for a degenerate AABB and propagates
    /// hash-grid construction errors.
    pub fn new(
        grid_cfg: HashGridConfig,
        cfg: NeuralangeloConfig,
        rng: &mut LcgRng,
    ) -> NerfResult<Self> {
        let extent = [
            cfg.aabb_max[0] - cfg.aabb_min[0],
            cfg.aabb_max[1] - cfg.aabb_min[1],
            cfg.aabb_max[2] - cfg.aabb_min[2],
        ];
        if extent.iter().any(|&e| e <= 0.0 || !e.is_finite()) {
            return Err(NerfError::Internal {
                msg: "aabb_max must be strictly greater than aabb_min on every axis".into(),
            });
        }
        let center = [
            0.5 * (cfg.aabb_min[0] + cfg.aabb_max[0]),
            0.5 * (cfg.aabb_min[1] + cfg.aabb_max[1]),
            0.5 * (cfg.aabb_min[2] + cfg.aabb_max[2]),
        ];

        let grid = HashGrid::new(grid_cfg, rng)?;
        let decoder = SdfDecoder::new(grid.output_dim(), cfg.hidden_dim.max(1), rng);

        Ok(Self {
            grid,
            decoder,
            config: cfg,
            extent,
            center,
            iteration: 0,
        })
    }

    /// Current coarse-to-fine training iteration.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Set the coarse-to-fine training iteration (controls the active level mask
    /// used by [`Neuralangelo::sdf`] and friends).
    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    /// Borrow the underlying hash grid.
    #[must_use]
    pub fn grid(&self) -> &HashGrid {
        &self.grid
    }

    /// Coarse-to-fine activation mask over hash-grid levels at the given
    /// iteration. Entries are in `[0, 1]`; the coarsest `init_active_levels`
    /// levels are always fully on, and finer levels ramp up smoothly as the
    /// iteration grows. The mask is monotonically non-decreasing in `iteration`.
    #[must_use]
    pub fn level_mask(&self, iteration: usize) -> Vec<f32> {
        let n_levels = self.grid.config.n_levels;
        let init = self.config.init_active_levels.min(n_levels);
        let steps = self.config.steps_per_level.max(1);
        let progress = iteration as f32 / steps as f32;
        (0..n_levels)
            .map(|level| {
                if level < init {
                    1.0
                } else {
                    let ramp = (progress - (level - init) as f32).clamp(0.0, 1.0);
                    // Smooth (raised-cosine) activation of the newest level.
                    0.5 * (1.0 - (std::f32::consts::PI * ramp).cos())
                }
            })
            .collect()
    }

    /// Numerical-gradient step `ε` tied to the finest active level's cell size at
    /// the given iteration (world units). Anneals as coarse-to-fine progresses.
    #[must_use]
    pub fn progressive_eps(&self, iteration: usize) -> f32 {
        let mask = self.level_mask(iteration);
        let resolutions = self.grid.level_resolutions();
        let finest = mask
            .iter()
            .enumerate()
            .rev()
            .find(|&(_, &m)| m > 0.5)
            .map_or(resolutions[0], |(level, _)| resolutions[level]);
        let extent_mean = (self.extent[0] + self.extent[1] + self.extent[2]) / 3.0;
        extent_mean / finest as f32
    }

    /// Map a world-space point into the grid's `[0,1]^3` domain.
    fn to_grid_coord(&self, x: [f32; 3]) -> [f32; 3] {
        [
            (x[0] - self.config.aabb_min[0]) / self.extent[0],
            (x[1] - self.config.aabb_min[1]) / self.extent[1],
            (x[2] - self.config.aabb_min[2]) / self.extent[2],
        ]
    }

    /// Signed distance at `x` using the current iteration's active level mask.
    #[must_use]
    pub fn sdf(&self, x: [f32; 3]) -> f32 {
        self.eval_sdf(x, self.iteration)
    }

    /// Signed distance at `x` for an explicit coarse-to-fine `iteration`.
    #[must_use]
    pub fn eval_sdf(&self, x: [f32; 3], iteration: usize) -> f32 {
        // Geometric base: a sphere centred in the AABB (gradient magnitude 1).
        let rel = [
            x[0] - self.center[0],
            x[1] - self.center[1],
            x[2] - self.center[2],
        ];
        let radius = (rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt();
        let base = radius - self.config.sphere_radius;

        // Hash-grid residual, gated by the coarse-to-fine level mask.
        let out_dim = self.grid.output_dim();
        let mut feat = match self.grid.query(self.to_grid_coord(x)) {
            Ok(features) => features,
            Err(_) => vec![0.0_f32; out_dim],
        };
        let mask = self.level_mask(iteration);
        let n_feat = self.grid.config.n_features_per_level;
        for (level, &gate) in mask.iter().enumerate() {
            let begin = level * n_feat;
            for value in &mut feat[begin..begin + n_feat] {
                *value *= gate;
            }
        }
        base + RESIDUAL_SCALE * self.decoder.forward(&feat)
    }

    /// Numerical (finite-difference) gradient of the SDF at `x` with step `eps`.
    #[must_use]
    pub fn numerical_gradient(&self, x: [f32; 3], eps: f32) -> [f32; 3] {
        numerical_gradient_of(|p| self.sdf(p), x, eps)
    }

    /// Surface curvature at `x` (discrete Laplacian of the SDF) with step `eps`.
    #[must_use]
    pub fn curvature(&self, x: [f32; 3], eps: f32) -> f32 {
        laplacian_of(|p| self.sdf(p), x, eps)
    }

    /// Eikonal residual `|∇f| − 1` of the SDF at `x` with step `eps`.
    #[must_use]
    pub fn eikonal_residual(&self, x: [f32; 3], eps: f32) -> f32 {
        eikonal_residual_of(|p| self.sdf(p), x, eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_neuralangelo(seed: u64) -> Neuralangelo {
        let grid_cfg = HashGridConfig {
            n_levels: 8,
            n_features_per_level: 2,
            log2_hashmap_size: 10,
            base_resolution: 4,
            max_resolution: 128,
        };
        let cfg = NeuralangeloConfig {
            init_active_levels: 3,
            steps_per_level: 100,
            aabb_min: [-1.0, -1.0, -1.0],
            aabb_max: [1.0, 1.0, 1.0],
            hidden_dim: 16,
            sphere_radius: 0.5,
        };
        let mut rng = LcgRng::new(seed);
        Neuralangelo::new(grid_cfg, cfg, &mut rng).expect("new should succeed")
    }

    fn norm3(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    #[test]
    fn numerical_gradient_matches_analytic_sphere() {
        // Sphere SDF f(x) = |x| - r,  ∇f = x / |x|.
        let sphere = |p: [f32; 3]| norm3(p) - 1.0;
        let x = [0.7_f32, -0.5, 0.3];
        let eps = 1e-3;
        let grad = numerical_gradient_of(sphere, x, eps);
        let mag = norm3(x);
        let analytic = [x[0] / mag, x[1] / mag, x[2] / mag];
        for axis in 0..3 {
            assert!(
                (grad[axis] - analytic[axis]).abs() < 1e-2,
                "axis {axis}: numerical {} vs analytic {}",
                grad[axis],
                analytic[axis]
            );
        }
    }

    #[test]
    fn eikonal_residual_near_zero_for_sphere() {
        let sphere = |p: [f32; 3]| norm3(p) - 1.0;
        let x = [0.6_f32, 0.5, 0.4];
        let residual = eikonal_residual_of(sphere, x, 1e-3);
        assert!(
            residual.abs() < 1e-2,
            "|∇f|-1 should be ~0 for an SDF, got {residual}"
        );
    }

    #[test]
    fn curvature_of_plane_is_zero() {
        // Plane SDF f(x) = y - d → ∇f constant, Laplacian = 0.
        let plane = |p: [f32; 3]| p[1] - 0.3;
        let lap = laplacian_of(plane, [0.1_f32, 0.2, 0.3], 1e-2);
        assert!(lap.abs() < 1e-3, "plane curvature should vanish, got {lap}");
    }

    #[test]
    fn curvature_of_sphere_is_inverse_radius_ish() {
        // Laplacian of |x| in 3D is 2/|x|; at the surface of radius r that is 2/r.
        let sphere = |p: [f32; 3]| norm3(p) - 1.0;
        let x = [0.5_f32, 0.4, 0.3];
        let mag = norm3(x);
        let lap = laplacian_of(sphere, x, 1e-2);
        let expected = 2.0 / mag;
        assert!(
            lap > 0.5,
            "sphere curvature must be clearly nonzero, got {lap}"
        );
        assert!(
            (lap - expected).abs() < 0.5,
            "sphere curvature {lap} should be ≈ 2/|x| = {expected}"
        );
    }

    #[test]
    fn level_mask_grows_monotonically() {
        let neur = make_neuralangelo(1);
        let n_levels = neur.grid().config.n_levels;
        let sum0: f32 = neur.level_mask(0).iter().sum();
        // Few levels at the start: exactly the initial active count.
        assert!(
            (sum0 - 3.0).abs() < 1e-5,
            "start should activate init_active_levels (3), got {sum0}"
        );
        assert!((sum0 as usize) < n_levels);

        // Monotone non-decreasing in iteration.
        let mut prev = sum0;
        for it in [50, 100, 300, 700, 2000, 100_000] {
            let sum: f32 = neur.level_mask(it).iter().sum();
            assert!(
                sum >= prev - 1e-5,
                "level activation must be monotone: {sum} < {prev} at it={it}"
            );
            prev = sum;
        }
        // Eventually all levels are active.
        let sum_final: f32 = neur.level_mask(1_000_000).iter().sum();
        assert!(
            (sum_final - n_levels as f32).abs() < 1e-4,
            "all levels should activate eventually, got {sum_final}"
        );
    }

    #[test]
    fn progressive_eps_anneals_with_iteration() {
        let neur = make_neuralangelo(2);
        let eps_start = neur.progressive_eps(0);
        let eps_mid = neur.progressive_eps(500);
        let eps_late = neur.progressive_eps(1_000_000);
        assert!(eps_start.is_finite() && eps_start > 0.0);
        assert!(
            eps_mid <= eps_start + 1e-6 && eps_late <= eps_mid + 1e-6,
            "eps should anneal (non-increasing): {eps_start} {eps_mid} {eps_late}"
        );
        assert!(
            eps_late < eps_start,
            "finest active level should shrink eps over time"
        );
    }

    #[test]
    fn model_sdf_and_derivatives_are_finite() {
        let mut neur = make_neuralangelo(3);
        neur.set_iteration(1000);
        let points: &[[f32; 3]] = &[
            [0.0, 0.0, 0.0],
            [0.3, -0.2, 0.4],
            [-0.6, 0.1, -0.5],
            [0.5, 0.5, 0.5],
        ];
        for &x in points {
            let value = neur.sdf(x);
            assert!(
                value.is_finite(),
                "sdf must be finite at {x:?}, got {value}"
            );
            let grad = neur.numerical_gradient(x, 1e-2);
            assert!(
                grad.iter().all(|g| g.is_finite()),
                "gradient must be finite"
            );
            let curv = neur.curvature(x, 1e-2);
            assert!(curv.is_finite(), "curvature must be finite");
            let eik = neur.eikonal_residual(x, 1e-2);
            assert!(eik.is_finite(), "eikonal residual must be finite");
        }
    }

    #[test]
    fn model_sdf_is_approximately_eikonal() {
        // The geometric sphere base dominates, so |∇f| ≈ 1 away from the centre.
        let neur = make_neuralangelo(4);
        let x = [0.4_f32, 0.3, -0.2];
        let grad = neur.numerical_gradient(x, 1e-2);
        let mag = norm3(grad);
        assert!(
            (mag - 1.0).abs() < 0.5,
            "geometric-init SDF should be roughly unit-gradient, |∇f|={mag}"
        );
    }

    #[test]
    fn new_rejects_degenerate_aabb() {
        let grid_cfg = HashGridConfig {
            n_levels: 2,
            n_features_per_level: 2,
            log2_hashmap_size: 6,
            base_resolution: 4,
            max_resolution: 8,
        };
        let cfg = NeuralangeloConfig {
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [1.0, 0.0, 1.0], // zero extent on y
            ..Default::default()
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            Neuralangelo::new(grid_cfg, cfg, &mut rng),
            Err(NerfError::Internal { .. })
        ));
    }
}
