//! Tile-based Gaussian splatting rasterizer (Inria 3DGS layout).
//!
//! Reference: Kerbl, Kopanas, Leimkühler, Drettakis, *"3D Gaussian Splatting
//! for Real-Time Radiance Field Rendering"*, SIGGRAPH 2023 — the tiled forward
//! renderer.
//!
//! The image plane is partitioned into a regular grid of `TILE × TILE`-pixel
//! tiles. Each projected Gaussian is bucketed into every tile that its `3σ`
//! axis-aligned bounding box overlaps. Within a tile the Gaussian list is
//! depth-sorted (ascending, nearest first) and alpha-composited front-to-back
//! independently of every other tile, on top of a constant background base
//! layer. The per-tile transmittance early-out (`T < T_min`) terminates a
//! pixel's blend once it is effectively opaque.
//!
//! Because compositing is a *per-pixel* operation and each pixel belongs to
//! exactly one tile, the tiled output is numerically identical (up to
//! floating-point summation order, which is preserved here) to the single
//! global sweep in [`crate::gaussian::rasterize::rasterize_gaussians`]: both
//! visit the Gaussians touching a pixel in the same global depth order. This
//! module exists to expose the data layout (per-tile Gaussian lists) that a GPU
//! kernel consumes, with a faithful CPU reference.

use crate::error::{Geom3dError, Geom3dResult};
use crate::gaussian::gaussian::Gaussian3d;
use crate::gaussian::project::ProjectedGaussian;

/// Edge length (in pixels) of a square rasterization tile.
pub const TILE: usize = 16;

/// Per-pixel transmittance below which compositing stops.
const T_MIN: f32 = 1e-4;

/// Minimum per-Gaussian alpha that is worth blending.
const ALPHA_MIN: f32 = 1.0 / 255.0;

/// Configuration for the tiled rasterizer.
#[derive(Debug, Clone)]
pub struct TileRasterConfig {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Background color blended under the final transmittance.
    pub bg_color: [f32; 3],
}

/// The per-tile binning of Gaussian indices.
///
/// `tile_lists[t]` holds the indices (into the input Gaussian array) of every
/// Gaussian whose `3σ` bounding box overlaps tile `t`, already sorted by
/// ascending depth. Tiles are stored row-major: `t = ty * tiles_x + tx`.
#[derive(Debug, Clone)]
pub struct TileBinning {
    /// Number of tile columns: `ceil(width / TILE)`.
    pub tiles_x: usize,
    /// Number of tile rows: `ceil(height / TILE)`.
    pub tiles_y: usize,
    /// Depth-sorted Gaussian index lists, one per tile.
    pub tile_lists: Vec<Vec<usize>>,
}

impl TileBinning {
    /// Total number of tiles.
    #[must_use]
    pub fn num_tiles(&self) -> usize {
        self.tiles_x * self.tiles_y
    }

    /// Total number of (Gaussian, tile) touch pairs across all tiles.
    #[must_use]
    pub fn total_touches(&self) -> usize {
        self.tile_lists.iter().map(Vec::len).sum()
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn invert2x2(m: &[f32; 4]) -> Option<[f32; 4]> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-12 {
        return None;
    }
    let inv = 1.0 / det;
    Some([m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv])
}

/// Bin the projected Gaussians into tiles and depth-sort each tile's list.
///
/// Only Gaussians with `valid == true` and a positive-definite 2D covariance
/// participate. Off-screen Gaussians whose AABB misses the framebuffer are
/// dropped.
///
/// # Errors
///
/// Returns [`Geom3dError::DimensionMismatch`] if the projected list length
/// differs from `n_gaussians`.
pub fn bin_gaussians_to_tiles(
    projected: &[ProjectedGaussian],
    n_gaussians: usize,
    width: usize,
    height: usize,
) -> Geom3dResult<TileBinning> {
    if projected.len() != n_gaussians {
        return Err(Geom3dError::DimensionMismatch {
            expected: n_gaussians,
            got: projected.len(),
        });
    }
    let tiles_x = width.div_ceil(TILE);
    let tiles_y = height.div_ceil(TILE);
    let mut tile_lists: Vec<Vec<usize>> = vec![Vec::new(); tiles_x * tiles_y];

    if width == 0 || height == 0 {
        return Ok(TileBinning {
            tiles_x,
            tiles_y,
            tile_lists,
        });
    }

    for (gi, pg) in projected.iter().enumerate() {
        if !pg.valid {
            continue;
        }
        // Reject non-PD covariances up front (their 3σ box is meaningless).
        if invert2x2(&pg.cov2d).is_none() {
            continue;
        }
        let sx = 3.0 * pg.cov2d[0].max(0.0).sqrt();
        let sy = 3.0 * pg.cov2d[3].max(0.0).sqrt();

        let x0 = (pg.xy[0] - sx).floor();
        let y0 = (pg.xy[1] - sy).floor();
        let x1 = (pg.xy[0] + sx).ceil();
        let y1 = (pg.xy[1] + sy).ceil();

        // Clamp to framebuffer, then convert to inclusive tile ranges.
        let px0 = x0.max(0.0) as usize;
        let py0 = y0.max(0.0) as usize;
        if x1 < 0.0 || y1 < 0.0 {
            continue;
        }
        let px1 = ((x1 as i64).clamp(0, width as i64 - 1)) as usize;
        let py1 = ((y1 as i64).clamp(0, height as i64 - 1)) as usize;
        if px0 >= width || py0 >= height {
            continue;
        }

        let tx0 = px0 / TILE;
        let ty0 = py0 / TILE;
        let tx1 = (px1 / TILE).min(tiles_x - 1);
        let ty1 = (py1 / TILE).min(tiles_y - 1);

        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                tile_lists[ty * tiles_x + tx].push(gi);
            }
        }
    }

    // Depth-sort each tile (stable so equal-depth keeps input order, matching
    // the global sweep's stable sort).
    for list in &mut tile_lists {
        list.sort_by(|&a, &b| {
            projected[a]
                .depth
                .partial_cmp(&projected[b].depth)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    Ok(TileBinning {
        tiles_x,
        tiles_y,
        tile_lists,
    })
}

/// Tile-based forward rasterizer.
///
/// Produces the same RGB image as the global sweep, but processes one tile at a
/// time using the per-tile Gaussian lists from [`bin_gaussians_to_tiles`].
/// Returns the image as an interleaved channels-last `Vec<f32>` of length
/// `width · height · 3`.
///
/// # Errors
///
/// Returns [`Geom3dError::BatchSizeMismatch`] if `gaussians` and `projected`
/// have different lengths.
pub fn rasterize_gaussians_tiled(
    gaussians: &[Gaussian3d],
    projected: &[ProjectedGaussian],
    cfg: &TileRasterConfig,
) -> Geom3dResult<Vec<f32>> {
    if gaussians.len() != projected.len() {
        return Err(Geom3dError::BatchSizeMismatch {
            lhs: gaussians.len(),
            rhs: projected.len(),
        });
    }
    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let mut image = vec![0.0_f32; w * h * 3];
    if w == 0 || h == 0 {
        return Ok(image);
    }
    // Initialise with the background as a base layer, matching the global sweep
    // in [`crate::gaussian::rasterize::rasterize_gaussians`].
    for pix in 0..w * h {
        let base = pix * 3;
        image[base] = cfg.bg_color[0];
        image[base + 1] = cfg.bg_color[1];
        image[base + 2] = cfg.bg_color[2];
    }

    let binning = bin_gaussians_to_tiles(projected, gaussians.len(), w, h)?;
    let mut transmittance = vec![1.0_f32; w * h];

    // Pre-compute per-Gaussian inverse covariance, view-dir color, opacity.
    let view_dir = [0.0_f32, 0.0, 1.0];
    let mut inv_cov = vec![[0.0_f32; 4]; gaussians.len()];
    let mut colors = vec![[0.0_f32; 3]; gaussians.len()];
    let mut opacities = vec![0.0_f32; gaussians.len()];
    for (gi, g) in gaussians.iter().enumerate() {
        if let Some(ic) = invert2x2(&projected[gi].cov2d) {
            inv_cov[gi] = ic;
        }
        colors[gi] = g.sh_color(view_dir).unwrap_or([0.5, 0.5, 0.5]);
        opacities[gi] = sigmoid(g.opacity);
    }

    for ty in 0..binning.tiles_y {
        for tx in 0..binning.tiles_x {
            let list = &binning.tile_lists[ty * binning.tiles_x + tx];
            if list.is_empty() {
                continue;
            }
            let tile_x_lo = tx * TILE;
            let tile_y_lo = ty * TILE;
            let tile_x_hi = (tile_x_lo + TILE).min(w);
            let tile_y_hi = (tile_y_lo + TILE).min(h);

            for &gi in list {
                let ic = inv_cov[gi];
                let center = projected[gi].xy;
                let alpha0 = opacities[gi];
                let color = colors[gi];

                // Intersect the tile with the Gaussian's own 3σ pixel AABB so a
                // splat is evaluated on exactly the pixels the global sweep
                // visits (tile membership only governs *which* tiles, not which
                // pixels within them).
                let sx = 3.0 * projected[gi].cov2d[0].max(0.0).sqrt();
                let sy = 3.0 * projected[gi].cov2d[3].max(0.0).sqrt();
                let g_x_lo = ((center[0] - sx) as i32).max(0) as usize;
                let g_y_lo = ((center[1] - sy) as i32).max(0) as usize;
                let g_x_hi = (((center[0] + sx) as i32 + 1).max(0) as usize).min(w);
                let g_y_hi = (((center[1] + sy) as i32 + 1).max(0) as usize).min(h);
                let x_lo = tile_x_lo.max(g_x_lo);
                let y_lo = tile_y_lo.max(g_y_lo);
                let x_hi = tile_x_hi.min(g_x_hi);
                let y_hi = tile_y_hi.min(g_y_hi);

                for py in y_lo..y_hi {
                    let dy = py as f32 + 0.5 - center[1];
                    for px in x_lo..x_hi {
                        let pix = py * w + px;
                        let t = transmittance[pix];
                        if t < T_MIN {
                            continue;
                        }
                        let dx = px as f32 + 0.5 - center[0];
                        let mah = ic[0] * dx * dx + (ic[1] + ic[2]) * dx * dy + ic[3] * dy * dy;
                        let alpha = alpha0 * (-0.5 * mah).exp();
                        if alpha < ALPHA_MIN {
                            continue;
                        }
                        let weight = t * alpha;
                        let base = pix * 3;
                        image[base] += weight * color[0];
                        image[base + 1] += weight * color[1];
                        image[base + 2] += weight * color[2];
                        transmittance[pix] = t * (1.0 - alpha);
                    }
                }
            }
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::project::{CameraIntrinsics, project_gaussian};
    use crate::handle::LcgRng;

    fn default_cam() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 32.0,
            near: 0.1,
        }
    }

    fn identity_view() -> [f32; 12] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    fn random_scene(n: usize, seed: u64) -> (Vec<Gaussian3d>, Vec<ProjectedGaussian>) {
        let mut rng = LcgRng::new(seed);
        let view = identity_view();
        let cam = default_cam();
        let mut gs = Vec::with_capacity(n);
        let mut pg = Vec::with_capacity(n);
        for _ in 0..n {
            let x = (rng.next_u32() as f32 / 4_294_967_296.0) * 2.0 - 1.0;
            let y = (rng.next_u32() as f32 / 4_294_967_296.0) * 2.0 - 1.0;
            let z = 3.0 + (rng.next_u32() as f32 / 4_294_967_296.0) * 4.0;
            let mut g = Gaussian3d::new_unit([x, y, z]);
            // Mild scale and non-zero color so blending is non-trivial.
            g.scale = [-1.2, -1.2, -1.2];
            g.opacity = 1.0;
            for c in g.sh.iter_mut().step_by(9) {
                *c = 1.5;
            }
            let projected =
                project_gaussian(&g, &view, &cam).expect("project_gaussian should succeed");
            gs.push(g);
            pg.push(projected);
        }
        (gs, pg)
    }

    #[test]
    fn tiles_cover_framebuffer() {
        let (_, pg) = random_scene(0, 1);
        let binning = bin_gaussians_to_tiles(&pg, 0, 64, 48).expect("bin should succeed");
        assert_eq!(binning.tiles_x, 4); // 64/16
        assert_eq!(binning.tiles_y, 3); // 48/16
        assert_eq!(binning.num_tiles(), 12);
    }

    #[test]
    fn non_multiple_image_dims_round_up() {
        let (_, pg) = random_scene(0, 2);
        let binning = bin_gaussians_to_tiles(&pg, 0, 33, 17).expect("bin should succeed");
        assert_eq!(binning.tiles_x, 3); // ceil(33/16)
        assert_eq!(binning.tiles_y, 2); // ceil(17/16)
    }

    #[test]
    fn gaussian_at_center_touches_center_tile() {
        let g = Gaussian3d::new_unit([0.0, 0.0, 5.0]);
        let pg = project_gaussian(&g, &identity_view(), &default_cam())
            .expect("project_gaussian should succeed");
        // Image 64×64 → 4×4 tiles; center pixel (32,32) is in tile (2,2).
        let binning = bin_gaussians_to_tiles(&[pg], 1, 64, 64).expect("bin should succeed");
        let center_tile = 2 * binning.tiles_x + 2;
        assert!(
            binning.tile_lists[center_tile].contains(&0),
            "center Gaussian must touch center tile"
        );
        assert!(binning.total_touches() >= 1);
    }

    #[test]
    fn tiled_matches_global_sweep() {
        use crate::gaussian::rasterize::{RasterConfig, rasterize_gaussians};
        let (gs, pg) = random_scene(40, 7);
        let cam = default_cam();
        let global_cfg = RasterConfig {
            width: 64,
            height: 64,
            bg_color: [0.05, 0.1, 0.15],
        };
        let tile_cfg = TileRasterConfig {
            width: 64,
            height: 64,
            bg_color: [0.05, 0.1, 0.15],
        };
        let global = rasterize_gaussians(&gs, &pg, &cam, &global_cfg)
            .expect("rasterize_gaussians should succeed");
        let tiled =
            rasterize_gaussians_tiled(&gs, &pg, &tile_cfg).expect("tiled raster should succeed");
        assert_eq!(global.len(), tiled.len());
        let mut max_diff = 0.0_f32;
        for (a, b) in global.iter().zip(tiled.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        // The two renderers visit identical Gaussians per pixel in identical
        // depth order; only the background-compositing arithmetic differs by
        // associativity, which stays well under 1e-4.
        assert!(
            max_diff < 1e-4,
            "tiled output must match global sweep, max diff {max_diff}"
        );
    }

    #[test]
    fn empty_scene_is_background() {
        let cfg = TileRasterConfig {
            width: 16,
            height: 16,
            bg_color: [0.2, 0.3, 0.4],
        };
        let img = rasterize_gaussians_tiled(&[], &[], &cfg).expect("tiled raster should succeed");
        assert_eq!(img.len(), 16 * 16 * 3);
        assert!((img[0] - 0.2).abs() < 1e-6);
        assert!((img[1] - 0.3).abs() < 1e-6);
        assert!((img[2] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn output_is_finite() {
        let (gs, pg) = random_scene(20, 99);
        let cfg = TileRasterConfig {
            width: 48,
            height: 32,
            bg_color: [0.0, 0.0, 0.0],
        };
        let img = rasterize_gaussians_tiled(&gs, &pg, &cfg).expect("tiled raster should succeed");
        assert!(img.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn binning_length_mismatch_errors() {
        let (_, pg) = random_scene(3, 1);
        assert!(bin_gaussians_to_tiles(&pg, 5, 32, 32).is_err());
    }
}
