//! Block-NeRF / Mega-NeRF scene partitioning for very large scenes.
//!
//! Tancik et al. (2022) "Block-NeRF: Scalable Large Scene Neural View Synthesis"
//! and Turki et al. (2022) "Mega-NeRF: Scalable Construction of Large-Scale NeRFs".
//!
//! A scene too large for a single MLP is split into spatially-localised
//! sub-models ("blocks"). Each block owns a centroid and an influence radius and
//! is responsible for the radiance inside its region. Regions overlap, so a 3D
//! point can be covered by several blocks; their predictions are merged with the
//! **inverse-distance weighting** used by Block-NeRF:
//!
//! ```text
//! w_b(x) = clip( d_b(x), 0, 1 )^{-p}            (d_b = ‖x − c_b‖ / radius_b)
//! merged(x) = Σ_b w_b(x)·f_b(x) / Σ_b w_b(x).
//! ```
//!
//! Blocks whose region does not contain the point are excluded
//! (visibility/relevance culling, the inexpensive analogue of Block-NeRF's
//! learned visibility network). Rendering routes each ray sample to its relevant
//! blocks, merges the per-block `(σ, rgb)`, and runs the standard NeRF
//! [`crate::rendering::volume_render::volume_render`] integral on the merged
//! field.
//!
//! The block fields here are randomly-initialised [`TinyNerf`] MLPs over a
//! per-block *local* positional encoding (coordinates are centred on the block),
//! exactly the partition-of-responsibility a trained ensemble would learn. All
//! randomness uses the crate [`LcgRng`]; no external crates.

use crate::error::{NerfError, NerfResult};
use crate::handle::LcgRng;
use crate::network::tiny_nerf::TinyNerf;
use crate::rendering::volume_render::{RenderResult, volume_render};

// ─── Config ────────────────────────────────────────────────────────────────────

/// Configuration for a partitioned Block-NeRF scene.
#[derive(Debug, Clone)]
pub struct BlockNerfConfig {
    /// Number of blocks per axis (total blocks = `grid^3`).
    pub grid: usize,
    /// Influence-radius multiplier in units of the block half-spacing. Values
    /// `> 1` create overlap between neighbouring blocks. Default `1.5`.
    pub overlap: f32,
    /// Inverse-distance exponent `p` for blending (Block-NeRF uses `p ≥ 1`).
    pub idw_power: f32,
    /// Frequency levels for each block's local positional encoding.
    pub pos_freq: usize,
    /// Hidden width of each block MLP.
    pub hidden_dim: usize,
}

impl Default for BlockNerfConfig {
    fn default() -> Self {
        Self {
            grid: 2,
            overlap: 1.5,
            idw_power: 2.0,
            pos_freq: 4,
            hidden_dim: 32,
        }
    }
}

// ─── Block ─────────────────────────────────────────────────────────────────────

/// A single localised sub-model: centroid, radius, and a local NeRF MLP.
#[derive(Debug, Clone)]
pub struct Block {
    /// World-space centroid `c_b`.
    pub centroid: [f32; 3],
    /// Influence radius `radius_b` (points beyond this are not the block's job).
    pub radius: f32,
    /// Per-block radiance MLP over local (centred) positional encoding.
    field: TinyNerf,
}

impl Block {
    /// Relevance distance `d_b(x) = ‖x − c_b‖ / radius_b`.
    #[must_use]
    pub fn relevance_distance(&self, x: [f32; 3]) -> f32 {
        let dx = x[0] - self.centroid[0];
        let dy = x[1] - self.centroid[1];
        let dz = x[2] - self.centroid[2];
        (dx * dx + dy * dy + dz * dz).sqrt() / self.radius
    }

    /// Whether `x` lies inside this block's influence region.
    #[must_use]
    pub fn covers(&self, x: [f32; 3]) -> bool {
        self.relevance_distance(x) <= 1.0
    }
}

// ─── BlockNerfScene ────────────────────────────────────────────────────────────

/// A scene partitioned into a grid of overlapping [`Block`] sub-models.
#[derive(Debug, Clone)]
pub struct BlockNerfScene {
    blocks: Vec<Block>,
    cfg: BlockNerfConfig,
    pos_cfg: crate::encoding::positional::PosEncConfig,
    /// Scene bounds: min corner and per-axis size.
    bounds_min: [f32; 3],
    bounds_size: [f32; 3],
}

impl BlockNerfScene {
    /// Partition an axis-aligned scene bounding box `[min, max]` into
    /// `grid³` overlapping blocks.
    ///
    /// # Errors
    ///
    /// - [`NerfError::InvalidGridResolution`] if `grid == 0`.
    /// - [`NerfError::InvalidFreqLevels`] if `pos_freq == 0`.
    /// - [`NerfError::InvalidFeatureDim`] if `hidden_dim == 0`.
    /// - [`NerfError::InvalidBounds`] if any axis has `max <= min`.
    /// - [`NerfError::InvalidEmbeddingConfig`] for non-positive `overlap`/`idw_power`.
    pub fn new(
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        cfg: BlockNerfConfig,
        rng: &mut LcgRng,
    ) -> NerfResult<Self> {
        if cfg.grid == 0 {
            return Err(NerfError::InvalidGridResolution { res: 0 });
        }
        if cfg.pos_freq == 0 {
            return Err(NerfError::InvalidFreqLevels { levels: 0 });
        }
        if cfg.hidden_dim == 0 {
            return Err(NerfError::InvalidFeatureDim { dim: 0 });
        }
        if !cfg.overlap.is_finite() || cfg.overlap <= 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "overlap must be positive".into(),
            });
        }
        if !cfg.idw_power.is_finite() || cfg.idw_power <= 0.0 {
            return Err(NerfError::InvalidEmbeddingConfig {
                msg: "idw_power must be positive".into(),
            });
        }
        let mut bounds_size = [0.0_f32; 3];
        for ax in 0..3 {
            // Reject degenerate or non-finite extents (NaN comparisons are false,
            // so test the finite, strictly-positive size directly).
            let size = bounds_max[ax] - bounds_min[ax];
            if !size.is_finite() || size <= 0.0 {
                return Err(NerfError::InvalidBounds {
                    near: bounds_min[ax],
                    far: bounds_max[ax],
                });
            }
            bounds_size[ax] = size;
        }

        let g = cfg.grid;
        let pos_cfg = crate::encoding::positional::PosEncConfig {
            n_freq: cfg.pos_freq,
            include_input: true,
            input_dim: 3,
        };

        // Block centroids sit at cell centres of the grid; the influence radius
        // is the half-cell diagonal scaled by `overlap` so neighbours overlap.
        let cell = [
            bounds_size[0] / g as f32,
            bounds_size[1] / g as f32,
            bounds_size[2] / g as f32,
        ];
        let half_diag = 0.5 * (cell[0] * cell[0] + cell[1] * cell[1] + cell[2] * cell[2]).sqrt();
        let radius = (half_diag * cfg.overlap).max(1.0e-6);

        let mut blocks = Vec::with_capacity(g * g * g);
        for ix in 0..g {
            for iy in 0..g {
                for iz in 0..g {
                    let centroid = [
                        bounds_min[0] + (ix as f32 + 0.5) * cell[0],
                        bounds_min[1] + (iy as f32 + 0.5) * cell[1],
                        bounds_min[2] + (iz as f32 + 0.5) * cell[2],
                    ];
                    let field = TinyNerf::new(pos_cfg.output_dim(), cfg.hidden_dim, rng);
                    blocks.push(Block {
                        centroid,
                        radius,
                        field,
                    });
                }
            }
        }

        Ok(Self {
            blocks,
            cfg,
            pos_cfg,
            bounds_min,
            bounds_size,
        })
    }

    /// Number of blocks (`grid³`).
    #[must_use]
    pub fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Borrow the blocks.
    #[must_use]
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Indices of blocks whose region covers `x` (relevance culling).
    #[must_use]
    pub fn relevant_blocks(&self, x: [f32; 3]) -> Vec<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| b.covers(x))
            .map(|(i, _)| i)
            .collect()
    }

    /// Evaluate one block's field at a world point (centring the coordinate on
    /// the block and normalising by its radius before encoding).
    fn eval_block(&self, idx: usize, x: [f32; 3]) -> NerfResult<(f32, [f32; 3])> {
        let b = &self.blocks[idx];
        let local = [
            (x[0] - b.centroid[0]) / b.radius,
            (x[1] - b.centroid[1]) / b.radius,
            (x[2] - b.centroid[2]) / b.radius,
        ];
        let pe = crate::encoding::positional::positional_encode(&local, &self.pos_cfg)?;
        b.field.forward(&pe)
    }

    /// Inverse-distance weight of a block at `x`:
    /// `w = (clip(d, eps, 1))^{-p}`, zero outside the region.
    fn idw_weight(&self, idx: usize, x: [f32; 3]) -> f32 {
        let d = self.blocks[idx].relevance_distance(x);
        if d > 1.0 {
            return 0.0;
        }
        let clipped = d.clamp(1.0e-4, 1.0);
        clipped.powf(-self.cfg.idw_power)
    }

    /// Merge all relevant blocks' `(σ, rgb)` at a world point via Block-NeRF
    /// inverse-distance weighting. Returns zero density for empty regions.
    ///
    /// # Errors
    ///
    /// Propagates encoding / field-forward errors.
    pub fn query_point(&self, x: [f32; 3]) -> NerfResult<(f32, [f32; 3])> {
        let relevant = self.relevant_blocks(x);
        if relevant.is_empty() {
            return Ok((0.0, [0.0, 0.0, 0.0]));
        }
        let mut w_sum = 0.0_f32;
        let mut sigma = 0.0_f32;
        let mut rgb = [0.0_f32; 3];
        for &idx in &relevant {
            let w = self.idw_weight(idx, x);
            if w == 0.0 {
                continue;
            }
            let (s, c) = self.eval_block(idx, x)?;
            w_sum += w;
            sigma += w * s;
            rgb[0] += w * c[0];
            rgb[1] += w * c[1];
            rgb[2] += w * c[2];
        }
        if w_sum <= 0.0 {
            return Ok((0.0, [0.0, 0.0, 0.0]));
        }
        sigma /= w_sum;
        rgb[0] /= w_sum;
        rgb[1] /= w_sum;
        rgb[2] /= w_sum;
        Ok((sigma, rgb))
    }

    /// Render a ray through the partitioned scene.
    ///
    /// For each sample position `o + t·d`, the merged field is queried (block
    /// routing + IDW blend) and the resulting `(σ, rgb)` sequence is volume
    /// rendered. `t_vals` must be ascending sample distances.
    ///
    /// # Errors
    ///
    /// - [`NerfError::ZeroRayDirection`] for a near-zero direction.
    /// - [`NerfError::EmptyInput`] for no samples.
    /// - Propagates field / volume-render errors.
    pub fn render_ray(
        &self,
        origin: [f32; 3],
        dir: [f32; 3],
        t_vals: &[f32],
    ) -> NerfResult<RenderResult> {
        let dn = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        if dn < 1.0e-8 {
            return Err(NerfError::ZeroRayDirection);
        }
        if t_vals.is_empty() {
            return Err(NerfError::EmptyInput);
        }
        let n = t_vals.len();
        let mut sigma = vec![0.0_f32; n];
        let mut color = vec![0.0_f32; n * 3];
        for (i, &t) in t_vals.iter().enumerate() {
            let x = [
                origin[0] + t * dir[0],
                origin[1] + t * dir[1],
                origin[2] + t * dir[2],
            ];
            let (s, c) = self.query_point(x)?;
            sigma[i] = s;
            color[i * 3] = c[0];
            color[i * 3 + 1] = c[1];
            color[i * 3 + 2] = c[2];
        }
        volume_render(&sigma, &color, t_vals)
    }

    /// Scene minimum corner.
    #[must_use]
    pub fn bounds_min(&self) -> [f32; 3] {
        self.bounds_min
    }

    /// Scene size per axis.
    #[must_use]
    pub fn bounds_size(&self) -> [f32; 3] {
        self.bounds_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(seed: u64) -> BlockNerfScene {
        let mut rng = LcgRng::new(seed);
        BlockNerfScene::new(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            BlockNerfConfig::default(),
            &mut rng,
        )
        .expect("scene")
    }

    #[test]
    fn block_count_is_grid_cubed() {
        let s = scene(1);
        assert_eq!(s.n_blocks(), 8); // grid = 2 → 2³
    }

    #[test]
    fn centroids_partition_bounds() {
        let s = scene(2);
        // With grid=2 on [-1,1]³, centroids should be at ±0.5 on each axis.
        for b in s.blocks() {
            for ax in 0..3 {
                assert!(
                    (b.centroid[ax].abs() - 0.5).abs() < 1e-5,
                    "centroid {:?}",
                    b.centroid
                );
            }
        }
    }

    #[test]
    fn overlap_yields_multi_block_coverage_at_center() {
        let s = scene(3);
        // The scene centre is equidistant from all 8 block centroids; with
        // overlap > 1 it should be covered by all of them.
        let relevant = s.relevant_blocks([0.0, 0.0, 0.0]);
        assert_eq!(relevant.len(), 8, "center should be covered by all blocks");
    }

    #[test]
    fn point_near_one_corner_favours_nearest_block() {
        let s = scene(4);
        let x = [0.45, 0.45, 0.45]; // near the (+,+,+) block centroid (0.5,0.5,0.5)
        let relevant = s.relevant_blocks(x);
        assert!(!relevant.is_empty());
        // Nearest block must have the largest IDW weight.
        let mut best = (usize::MAX, f32::NEG_INFINITY);
        for &idx in &relevant {
            let w = s.idw_weight(idx, x);
            if w > best.1 {
                best = (idx, w);
            }
        }
        let nearest = relevant
            .iter()
            .copied()
            .min_by(|&a, &b| {
                s.blocks()[a]
                    .relevance_distance(x)
                    .partial_cmp(&s.blocks()[b].relevance_distance(x))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("nearest");
        assert_eq!(best.0, nearest, "largest weight should be nearest block");
    }

    #[test]
    fn merged_query_is_convex_blend_in_range() {
        let s = scene(5);
        // Query several points; merged rgb must lie within [0,1] (convex blend
        // of per-block sigmoids), sigma must be non-negative.
        let pts = [
            [0.0, 0.0, 0.0],
            [0.4, -0.3, 0.2],
            [-0.6, 0.6, -0.1],
            [0.5, 0.5, 0.5],
        ];
        for &p in &pts {
            let (sigma, rgb) = s.query_point(p).expect("query");
            assert!(sigma.is_finite() && sigma >= 0.0, "sigma={sigma}");
            for c in rgb {
                assert!(c.is_finite() && (0.0..=1.0).contains(&c), "rgb={rgb:?}");
            }
        }
    }

    #[test]
    fn merged_query_matches_single_block_in_exclusive_region() {
        let s = scene(6);
        // A point covered by exactly one block must reproduce that block's
        // prediction exactly (IDW with a single contributor = identity).
        // Find such a point by probing near a corner far enough that only the
        // corner block covers it.
        let mut found = None;
        for &probe in &[[0.95, 0.95, 0.95], [0.9, 0.9, 0.9], [0.98, 0.0, 0.0]] {
            let relevant = s.relevant_blocks(probe);
            if relevant.len() == 1 {
                found = Some((probe, relevant[0]));
                break;
            }
        }
        if let Some((p, idx)) = found {
            let merged = s.query_point(p).expect("merged");
            let direct = s.eval_block(idx, p).expect("direct");
            assert!((merged.0 - direct.0).abs() < 1e-5, "sigma mismatch");
            for k in 0..3 {
                assert!(
                    (merged.1[k] - direct.1[k]).abs() < 1e-5,
                    "rgb[{k}] mismatch"
                );
            }
        }
        // If no exclusive point found in this config, the test is vacuous but
        // the rest of the suite covers the multi-block path.
    }

    #[test]
    fn outside_scene_returns_empty_space() {
        let s = scene(7);
        // Far outside the bounds: no block covers it.
        let (sigma, rgb) = s.query_point([100.0, 100.0, 100.0]).expect("query");
        assert_eq!(sigma, 0.0);
        assert_eq!(rgb, [0.0, 0.0, 0.0]);
        assert!(s.relevant_blocks([100.0, 100.0, 100.0]).is_empty());
    }

    #[test]
    fn render_ray_through_scene_valid() {
        let s = scene(8);
        let t_vals: Vec<f32> = (0..32).map(|i| -1.0 + i as f32 * (2.0 / 31.0)).collect();
        let res = s
            .render_ray([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], &t_vals)
            .expect("render");
        assert!(res.opacity.is_finite() && (0.0..=1.0).contains(&res.opacity));
        for c in res.rgb {
            assert!(c.is_finite() && (0.0..=1.0).contains(&c));
        }
    }

    #[test]
    fn render_ray_deterministic() {
        let s = scene(9);
        let t_vals: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.8).collect();
        let a = s
            .render_ray([0.1, 0.0, -0.9], [0.0, 0.0, 1.0], &t_vals)
            .expect("a");
        let b = s
            .render_ray([0.1, 0.0, -0.9], [0.0, 0.0, 1.0], &t_vals)
            .expect("b");
        assert_eq!(a.rgb, b.rgb);
        assert_eq!(a.opacity, b.opacity);
    }

    #[test]
    fn validation_errors() {
        let mut rng = LcgRng::new(10);
        let bad_grid = BlockNerfConfig {
            grid: 0,
            ..Default::default()
        };
        assert!(BlockNerfScene::new([-1.0; 3], [1.0; 3], bad_grid, &mut rng).is_err());
        // degenerate bounds
        assert!(
            BlockNerfScene::new([0.0; 3], [0.0; 3], BlockNerfConfig::default(), &mut rng).is_err()
        );
        let bad_power = BlockNerfConfig {
            idw_power: 0.0,
            ..Default::default()
        };
        assert!(BlockNerfScene::new([-1.0; 3], [1.0; 3], bad_power, &mut rng).is_err());
    }

    #[test]
    fn render_ray_rejects_zero_direction() {
        let s = scene(11);
        let t_vals = [0.1_f32, 0.2, 0.3];
        assert!(s.render_ray([0.0; 3], [0.0; 3], &t_vals).is_err());
    }
}
