//! Zip-NeRF — anti-aliased grid-based neural radiance fields.
//!
//! Barron, Mildenhall, Verbin, Srinivasan & Hedman (2023), "Zip-NeRF:
//! Anti-Aliased Grid-Based Neural Radiance Fields", ICCV.
//!
//! Instant-NGP-style hash grids are fast but alias badly: a single point query
//! cannot represent the spatial footprint of a pixel, so the rendered image
//! flickers under camera motion / scale change. Mip-NeRF solved this for the
//! classical Fourier encoding by integrating over a Gaussian-approximated cone
//! frustum, but that trick does not transfer to a hash grid. Zip-NeRF instead:
//!
//! 1. **Conical-frustum multisampling** — each `[t_i, t_{i+1}]` interval along a
//!    ray is a truncated cone (a *conical frustum*). Rather than querying its
//!    centre, we scatter a small set of multisamples in a hexagonal spiral on
//!    the cone, with the off-axis radius scaled by the cone radius at that depth
//!    (`r(t) = pixel_radius · t`, so the spread grows with `t`).
//! 2. **Hash-grid featurisation** — every multisample is featurised by the shared
//!    multiresolution hash grid via trilinear interpolation (reused verbatim).
//! 3. **Cone-vs-cell reweighting** — to suppress *z-aliasing*, each grid level's
//!    features are downweighted by a Gaussian of the ratio between the cone
//!    radius and that level's cell size `1/N_l`. A wide cone keeps only the
//!    coarse levels; a tight cone keeps everything. The reweighted features are
//!    averaged over the multisamples to obtain the per-interval feature.
//! 4. **Render** — the per-interval features drive the shared density/colour MLP
//!    and are composited with the standard volume-rendering integral.
//!
//! This is a compact, faithful CPU core: the multisample pattern is a fixed
//! hexagonal spiral (the paper additionally jitters it stochastically), and the
//! reweighting uses the closed-form Gaussian footprint attenuation.

use crate::encoding::hash_grid::{HashGrid, HashGridConfig};
use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::network::nerf_mlp::{NerfMlp, NerfMlpConfig};
use crate::rendering::ray::Ray;
use crate::rendering::volume_render::{RenderResult, volume_render};

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for the Zip-NeRF anti-aliased renderer.
#[derive(Debug, Clone)]
pub struct ZipNerfConfig {
    /// Number of multisamples placed inside each conical-frustum interval
    /// (Zip-NeRF default: 6).
    pub n_multisamples: usize,
    /// Lower corner of the scene bounding box (world → `[0,1]^3` mapping).
    pub aabb_min: [f32; 3],
    /// Upper corner of the scene bounding box.
    pub aabb_max: [f32; 3],
    /// Multiplier on the cone radius used for the off-axis multisample spread.
    pub radius_scale: f32,
}

impl Default for ZipNerfConfig {
    fn default() -> Self {
        Self {
            n_multisamples: 6,
            aabb_min: [-1.0, -1.0, -1.0],
            aabb_max: [1.0, 1.0, 1.0],
            radius_scale: 1.0,
        }
    }
}

// ─── Multisample ─────────────────────────────────────────────────────────────

/// A single multisample placed on the conical frustum.
#[derive(Debug, Clone, Copy)]
pub struct Multisample {
    /// World-space position of the multisample.
    pub position: [f32; 3],
    /// Distance `t` along the ray at which this sample sits.
    pub t: f32,
    /// Cone radius at this depth (`pixel_radius · t`).
    pub cone_radius: f32,
}

// ─── ZipNerf ─────────────────────────────────────────────────────────────────

/// Anti-aliased Zip-NeRF renderer: hash grid + density/colour MLP + cone
/// multisampling and reweighting.
#[derive(Debug, Clone)]
pub struct ZipNerf {
    grid: HashGrid,
    mlp: NerfMlp,
    config: ZipNerfConfig,
    /// Per-axis extent `aabb_max - aabb_min`, all strictly positive.
    extent: [f32; 3],
}

impl ZipNerf {
    /// Build a Zip-NeRF renderer from a hash-grid config, a Zip config and the
    /// hidden width of the shared MLP.
    ///
    /// The MLP's positional-feature input dimension is taken from the hash grid
    /// (`n_levels · n_features_per_level`); the view direction is fed raw (3-D).
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidSampleCount`] if `n_multisamples == 0`,
    /// [`NerfError::Internal`] for a degenerate AABB or negative `radius_scale`,
    /// and propagates hash-grid / MLP construction errors.
    pub fn new(
        grid_cfg: HashGridConfig,
        zip_cfg: ZipNerfConfig,
        hidden_dim: usize,
        rng: &mut LcgRng,
    ) -> NerfResult<Self> {
        if zip_cfg.n_multisamples == 0 {
            return Err(NerfError::InvalidSampleCount { n: 0 });
        }
        if !zip_cfg.radius_scale.is_finite() || zip_cfg.radius_scale < 0.0 {
            return Err(NerfError::Internal {
                msg: "radius_scale must be finite and non-negative".into(),
            });
        }
        let extent = [
            zip_cfg.aabb_max[0] - zip_cfg.aabb_min[0],
            zip_cfg.aabb_max[1] - zip_cfg.aabb_min[1],
            zip_cfg.aabb_max[2] - zip_cfg.aabb_min[2],
        ];
        if extent.iter().any(|&e| e <= 0.0 || !e.is_finite()) {
            return Err(NerfError::Internal {
                msg: "aabb_max must be strictly greater than aabb_min on every axis".into(),
            });
        }

        let grid = HashGrid::new(grid_cfg, rng)?;
        let mlp_cfg = NerfMlpConfig {
            xyz_enc_dim: grid.output_dim(),
            dir_enc_dim: 3,
            hidden_dim,
        };
        let mlp = NerfMlp::new(mlp_cfg, rng)?;

        Ok(Self {
            grid,
            mlp,
            config: zip_cfg,
            extent,
        })
    }

    /// Borrow the underlying hash grid.
    #[must_use]
    pub fn grid(&self) -> &HashGrid {
        &self.grid
    }

    /// Number of multisamples produced per interval.
    #[must_use]
    pub fn n_multisamples(&self) -> usize {
        self.config.n_multisamples
    }

    /// Map a world-space point into the grid's `[0,1]^3` domain.
    fn to_grid_coord(&self, world: [f32; 3]) -> [f32; 3] {
        [
            (world[0] - self.config.aabb_min[0]) / self.extent[0],
            (world[1] - self.config.aabb_min[1]) / self.extent[1],
            (world[2] - self.config.aabb_min[2]) / self.extent[2],
        ]
    }

    /// Featurise a single world-space point through the hash grid (no
    /// reweighting). Trilinear interpolation makes this continuous in `world`.
    ///
    /// # Errors
    ///
    /// Propagates [`HashGrid::query`] errors.
    pub fn grid_feature(&self, world: [f32; 3]) -> NerfResult<Vec<f32>> {
        self.grid.query(self.to_grid_coord(world))
    }

    /// Per-level anti-aliasing weights for a cone of the given world-space
    /// radius: `w_l = exp(-½ (r_cone / cell_l)²)` with `cell_l = 1 / N_l`
    /// measured in the normalised grid domain.
    fn level_weights(&self, cone_radius_world: f32) -> Vec<f32> {
        let extent_mean = (self.extent[0] + self.extent[1] + self.extent[2]) / 3.0;
        let cone_norm = if extent_mean > 0.0 {
            cone_radius_world / extent_mean
        } else {
            0.0
        };
        self.grid
            .level_resolutions()
            .iter()
            .map(|&n_l| {
                let ratio = cone_norm * n_l as f32;
                (-0.5 * ratio * ratio).exp()
            })
            .collect()
    }

    /// Build the hexagonal-spiral multisample set for one conical-frustum
    /// interval `[t_near, t_far]` of a ray with the given per-distance pixel
    /// radius.
    #[must_use]
    pub fn multisamples(
        &self,
        ray: &Ray,
        t_near: f32,
        t_far: f32,
        pixel_radius: f32,
    ) -> Vec<Multisample> {
        let count = self.config.n_multisamples;
        let (axis_u, axis_v) = orthonormal_basis(ray.dir);
        let inv_count = 1.0 / count as f32;
        (0..count)
            .map(|j| {
                let frac = (j as f32 + 0.5) * inv_count;
                let t_sample = t_near + frac * (t_far - t_near);
                let cone_radius = pixel_radius * t_sample;
                let offset = self.config.radius_scale * cone_radius;
                let angle = std::f32::consts::TAU * (j as f32) * inv_count;
                let (sin_a, cos_a) = angle.sin_cos();
                let axis_point = ray.at(t_sample);
                let position = [
                    axis_point[0] + offset * (cos_a * axis_u[0] + sin_a * axis_v[0]),
                    axis_point[1] + offset * (cos_a * axis_u[1] + sin_a * axis_v[1]),
                    axis_point[2] + offset * (cos_a * axis_u[2] + sin_a * axis_v[2]),
                ];
                Multisample {
                    position,
                    t: t_sample,
                    cone_radius,
                }
            })
            .collect()
    }

    /// Compute the anti-aliased feature for one conical-frustum interval: query
    /// every multisample, downweight each grid level by the cone-vs-cell
    /// Gaussian, then average over the multisamples.
    ///
    /// The returned vector has length `grid.output_dim()`.
    ///
    /// # Errors
    ///
    /// Propagates hash-grid query errors.
    pub fn featurize_interval(
        &self,
        ray: &Ray,
        t_near: f32,
        t_far: f32,
        pixel_radius: f32,
    ) -> NerfResult<Vec<f32>> {
        let samples = self.multisamples(ray, t_near, t_far, pixel_radius);
        if samples.is_empty() {
            return Err(NerfError::InvalidSampleCount { n: 0 });
        }
        let out_dim = self.grid.output_dim();
        let n_feat = self.grid.config.n_features_per_level;
        let mut acc = vec![0.0_f32; out_dim];

        for sample in &samples {
            let feat = self.grid_feature(sample.position)?;
            let weights = self.level_weights(sample.cone_radius);
            for (level, &weight) in weights.iter().enumerate() {
                let base = level * n_feat;
                for offset in 0..n_feat {
                    acc[base + offset] += weight * feat[base + offset];
                }
            }
        }

        let inv = 1.0 / samples.len() as f32;
        for value in &mut acc {
            *value *= inv;
        }
        Ok(acc)
    }

    /// Anti-aliased volume render of a ray given its `t` interval edges
    /// (`t_edges.len() = n_intervals + 1`) and per-distance pixel radius.
    ///
    /// # Errors
    ///
    /// Returns [`NerfError::InvalidSampleCount`] if fewer than two edges are
    /// supplied, [`NerfError::Internal`] for a negative `pixel_radius`, and
    /// propagates featurisation / MLP / volume-render errors.
    pub fn render(
        &self,
        ray: &Ray,
        t_edges: &[f32],
        pixel_radius: f32,
    ) -> NerfResult<RenderResult> {
        if t_edges.len() < 2 {
            return Err(NerfError::InvalidSampleCount { n: t_edges.len() });
        }
        if !pixel_radius.is_finite() || pixel_radius < 0.0 {
            return Err(NerfError::Internal {
                msg: "pixel_radius must be finite and non-negative".into(),
            });
        }

        let n_intervals = t_edges.len() - 1;
        let mut sigma = Vec::with_capacity(n_intervals);
        let mut color = Vec::with_capacity(n_intervals * 3);
        let mut t_mid = Vec::with_capacity(n_intervals);

        for edge in t_edges.windows(2) {
            let t_near = edge[0];
            let t_far = edge[1];
            let feature = self.featurize_interval(ray, t_near, t_far, pixel_radius)?;
            let (density, rgb) = self.mlp.forward(&feature, &ray.dir)?;
            sigma.push(density);
            color.extend_from_slice(&rgb);
            t_mid.push(0.5 * (t_near + t_far));
        }

        volume_render(&sigma, &color, &t_mid)
    }
}

// ─── Vector helpers ──────────────────────────────────────────────────────────

/// Build an orthonormal basis `(u, v)` spanning the plane perpendicular to
/// `dir` (assumed approximately unit length).
fn orthonormal_basis(dir: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    // Choose a helper axis that is not (near-)parallel to `dir`.
    let helper = if dir[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let axis_u = normalize(cross(helper, dir));
    // `dir × u` is unit when `dir` is unit and orthogonal to `u`.
    let axis_v = cross(dir, axis_u);
    (axis_u, axis_v)
}

#[inline]
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if len_sq > 1e-20 {
        let inv = 1.0 / len_sq.sqrt();
        [v[0] * inv, v[1] * inv, v[2] * inv]
    } else {
        [1.0, 0.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_zip(seed: u64) -> ZipNerf {
        let grid_cfg = HashGridConfig {
            n_levels: 8,
            n_features_per_level: 2,
            log2_hashmap_size: 10,
            base_resolution: 4,
            max_resolution: 64,
        };
        let zip_cfg = ZipNerfConfig {
            n_multisamples: 6,
            aabb_min: [-1.0, -1.0, -1.0],
            aabb_max: [1.0, 1.0, 1.0],
            radius_scale: 1.0,
        };
        let mut rng = LcgRng::new(seed);
        ZipNerf::new(grid_cfg, zip_cfg, 16, &mut rng).expect("new should succeed")
    }

    fn perp_distance(point: [f32; 3], ray: &Ray) -> f32 {
        // Distance of `point` from the ray's infinite axis.
        let rel = [
            point[0] - ray.origin[0],
            point[1] - ray.origin[1],
            point[2] - ray.origin[2],
        ];
        let along = rel[0] * ray.dir[0] + rel[1] * ray.dir[1] + rel[2] * ray.dir[2];
        let proj = [
            rel[0] - along * ray.dir[0],
            rel[1] - along * ray.dir[1],
            rel[2] - along * ray.dir[2],
        ];
        (proj[0] * proj[0] + proj[1] * proj[1] + proj[2] * proj[2]).sqrt()
    }

    fn total_variance(samples: &[Vec<f32>]) -> f32 {
        let n = samples.len();
        assert!(n > 1);
        let dim = samples[0].len();
        let mut var = 0.0_f32;
        for d in 0..dim {
            let mean: f32 = samples.iter().map(|s| s[d]).sum::<f32>() / n as f32;
            let v: f32 = samples
                .iter()
                .map(|s| (s[d] - mean) * (s[d] - mean))
                .sum::<f32>()
                / n as f32;
            var += v;
        }
        var
    }

    #[test]
    fn multisample_count_matches_config() {
        let zn = make_zip(1);
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        let samples = zn.multisamples(&ray, 1.0, 1.2, 0.1);
        assert_eq!(samples.len(), zn.n_multisamples());
        assert_eq!(samples.len(), 6);
    }

    #[test]
    fn spread_scales_with_cone_radius() {
        let zn = make_zip(2);
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        let pixel_radius = 0.1;
        let near = zn.multisamples(&ray, 0.1, 0.2, pixel_radius);
        let far = zn.multisamples(&ray, 4.0, 4.1, pixel_radius);
        let spread_near = near
            .iter()
            .map(|s| perp_distance(s.position, &ray))
            .fold(0.0_f32, f32::max);
        let spread_far = far
            .iter()
            .map(|s| perp_distance(s.position, &ray))
            .fold(0.0_f32, f32::max);
        assert!(
            spread_far > spread_near * 5.0,
            "wider cone at larger t should spread more: near={spread_near} far={spread_far}"
        );
    }

    #[test]
    fn grid_feature_is_continuous() {
        let zn = make_zip(3);
        let base = [0.13_f32, -0.21, 0.37];
        let near = [base[0] + 1e-4, base[1], base[2]];
        let fa = zn.grid_feature(base).expect("grid_feature should succeed");
        let fb = zn.grid_feature(near).expect("grid_feature should succeed");
        let max_diff = fa
            .iter()
            .zip(fb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "trilinear interpolation must be continuous, max_diff={max_diff}"
        );
    }

    #[test]
    fn antialiasing_lowers_variance_vs_single_point() {
        let zn = make_zip(4);
        let mut rng = LcgRng::new(99);
        let dir = [0.0, 0.0, 1.0];
        let t_near = 1.9;
        let t_far = 2.1;
        let pixel_radius = 0.15; // wide cone
        let t_centre = 0.5 * (t_near + t_far);

        let n_jitter = 24;
        let mut single = Vec::with_capacity(n_jitter);
        let mut multi = Vec::with_capacity(n_jitter);
        for _ in 0..n_jitter {
            let origin = [
                rng.next_f32_range(-0.06, 0.06),
                rng.next_f32_range(-0.06, 0.06),
                -2.0,
            ];
            let ray = Ray::normalized(origin, dir).expect("normalized should succeed");
            single.push(
                zn.grid_feature(ray.at(t_centre))
                    .expect("value should be present"),
            );
            multi.push(
                zn.featurize_interval(&ray, t_near, t_far, pixel_radius)
                    .expect("value should be present"),
            );
        }
        let var_single = total_variance(&single);
        let var_multi = total_variance(&multi);
        assert!(
            var_multi < var_single,
            "anti-aliased feature should have lower variance: multi={var_multi} single={var_single}"
        );
    }

    #[test]
    fn render_weights_are_valid() {
        let zn = make_zip(5);
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        let t_edges: Vec<f32> = (0..=12).map(|i| 1.0 + i as f32 * 0.2).collect();
        let res = zn
            .render(&ray, &t_edges, 0.05)
            .expect("render should succeed");
        assert!(
            (0.0..=1.000_01).contains(&res.opacity),
            "opacity (Σ weights) must be in [0,1], got {}",
            res.opacity
        );
        assert!(res.opacity >= 0.0, "weights must be non-negative");
        assert!(res.depth.is_finite(), "depth must be finite");
        for (i, &c) in res.rgb.iter().enumerate() {
            assert!(c.is_finite(), "rgb[{i}] must be finite");
            assert!((0.0..=1.0).contains(&c), "rgb[{i}]={c} out of [0,1]");
        }
    }

    #[test]
    fn render_rejects_short_edges() {
        let zn = make_zip(6);
        let ray =
            Ray::normalized([0.0, 0.0, -2.0], [0.0, 0.0, 1.0]).expect("normalized should succeed");
        assert!(matches!(
            zn.render(&ray, &[1.0], 0.05),
            Err(NerfError::InvalidSampleCount { .. })
        ));
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
        let zip_cfg = ZipNerfConfig {
            n_multisamples: 6,
            aabb_min: [0.0, 0.0, 0.0],
            aabb_max: [0.0, 1.0, 1.0], // zero extent on x
            radius_scale: 1.0,
        };
        let mut rng = LcgRng::new(1);
        assert!(matches!(
            ZipNerf::new(grid_cfg, zip_cfg, 8, &mut rng),
            Err(NerfError::Internal { .. })
        ));
    }

    #[test]
    fn level_weights_downweight_fine_for_wide_cone() {
        let zn = make_zip(7);
        let weights = zn.level_weights(0.3); // wide cone, world units
        // Coarsest level (index 0) must keep more weight than the finest level.
        assert!(
            weights[0] >= weights[weights.len() - 1],
            "coarse level should be weighted >= fine level: {weights:?}"
        );
        assert!(
            weights[weights.len() - 1] < weights[0],
            "a wide cone must strictly downweight the finest level"
        );
    }
}
