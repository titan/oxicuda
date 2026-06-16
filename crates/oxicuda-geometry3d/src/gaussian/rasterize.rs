//! Single-threaded alpha-compositing Gaussian splatting rasterizer.

use crate::error::{Geom3dError, Geom3dResult};
use crate::gaussian::gaussian::Gaussian3d;
use crate::gaussian::project::{CameraIntrinsics, ProjectedGaussian};

/// Rasterizer configuration.
#[derive(Debug, Clone)]
pub struct RasterConfig {
    pub width: u32,
    pub height: u32,
    pub bg_color: [f32; 3],
}

/// Compute sigmoid function.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Invert a 2×2 matrix.
fn invert2x2(m: &[f32; 4]) -> Option<[f32; 4]> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-8 {
        return None;
    }
    Some([m[3] / det, -m[1] / det, -m[2] / det, m[0] / det])
}

/// Simple single-threaded alpha-compositing Gaussian splatting rasterizer.
///
/// Returns RGB image `[height×width×3]`.
pub fn rasterize_gaussians(
    gaussians: &[Gaussian3d],
    projected: &[ProjectedGaussian],
    _cam: &CameraIntrinsics,
    cfg: &RasterConfig,
) -> Geom3dResult<Vec<f32>> {
    if gaussians.len() != projected.len() {
        return Err(Geom3dError::BatchSizeMismatch {
            lhs: gaussians.len(),
            rhs: projected.len(),
        });
    }

    let w = cfg.width as usize;
    let h = cfg.height as usize;
    let mut image = vec![0.0_f32; h * w * 3];

    // Initialize with background color
    for py in 0..h {
        for px in 0..w {
            let idx = (py * w + px) * 3;
            image[idx] = cfg.bg_color[0];
            image[idx + 1] = cfg.bg_color[1];
            image[idx + 2] = cfg.bg_color[2];
        }
    }

    // Depth-sort Gaussians (front to back = smallest depth first)
    let mut order: Vec<usize> = projected
        .iter()
        .enumerate()
        .filter(|(_, pg)| pg.valid)
        .map(|(i, _)| i)
        .collect();
    order.sort_unstable_by(|&a, &b| {
        projected[a]
            .depth
            .partial_cmp(&projected[b].depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Transmittance buffer (per pixel)
    let mut transmittance = vec![1.0_f32; h * w];

    for &gi in &order {
        let pg = &projected[gi];
        let g = &gaussians[gi];

        // Invert 2D covariance
        let inv_cov = match invert2x2(&pg.cov2d) {
            Some(m) => m,
            None => continue,
        };

        // 3σ AABB in pixels
        let sigma_x = (pg.cov2d[0]).sqrt() * 3.0;
        let sigma_y = (pg.cov2d[3]).sqrt() * 3.0;

        let x_center = pg.xy[0];
        let y_center = pg.xy[1];

        let px_min = ((x_center - sigma_x) as i32).max(0) as usize;
        let px_max = ((x_center + sigma_x) as i32 + 1).min(w as i32) as usize;
        let py_min = ((y_center - sigma_y) as i32).max(0) as usize;
        let py_max = ((y_center + sigma_y) as i32 + 1).min(h as i32) as usize;

        let opacity_actual = sigmoid(g.opacity);

        // Evaluate SH color in view direction (approximation: use -z as view dir)
        // View dir toward screen = -z in camera space; approximate with camera forward
        let view_dir = [0.0_f32, 0.0, 1.0];
        let color = g.sh_color(view_dir).unwrap_or([0.5, 0.5, 0.5]);

        for py in py_min..py_max {
            for px in px_min..px_max {
                let dx = px as f32 + 0.5 - x_center;
                let dy = py as f32 + 0.5 - y_center;

                // Mahalanobis: d^T Σ^{-1} d
                let mah = inv_cov[0] * dx * dx
                    + (inv_cov[1] + inv_cov[2]) * dx * dy
                    + inv_cov[3] * dy * dy;

                let alpha = opacity_actual * (-0.5 * mah).exp();
                if alpha < 1.0 / 255.0 {
                    continue;
                }

                let pix_idx = py * w + px;
                let t = transmittance[pix_idx];
                if t < 1e-4 {
                    continue;
                }

                let weight = t * alpha;
                let img_idx = pix_idx * 3;
                image[img_idx] += weight * color[0];
                image[img_idx + 1] += weight * color[1];
                image[img_idx + 2] += weight * color[2];

                transmittance[pix_idx] *= 1.0 - alpha;
            }
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::project::project_gaussian;

    fn make_gaussian_at(z: f32) -> Gaussian3d {
        Gaussian3d {
            pos: [0.0, 0.0, z],
            rot: [1.0, 0.0, 0.0, 0.0],
            scale: [0.0, 0.0, 0.0],
            opacity: 2.0, // high opacity
            sh: vec![1.0; 27],
        }
    }

    fn default_cam() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 100.0,
            fy: 100.0,
            cx: 50.0,
            cy: 50.0,
            near: 0.1,
        }
    }

    fn identity_view() -> [f32; 12] {
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    }

    #[test]
    fn rasterize_output_shape() {
        let gaussians: Vec<Gaussian3d> = Vec::new();
        let projected: Vec<ProjectedGaussian> = Vec::new();
        let cam = default_cam();
        let cfg = RasterConfig {
            width: 100,
            height: 100,
            bg_color: [0.0, 0.0, 0.0],
        };
        let img = rasterize_gaussians(&gaussians, &projected, &cam, &cfg)
            .expect("rasterize_gaussians should succeed");
        assert_eq!(img.len(), 100 * 100 * 3);
    }

    #[test]
    fn rasterize_background_color() {
        let cfg = RasterConfig {
            width: 4,
            height: 4,
            bg_color: [0.2, 0.4, 0.6],
        };
        let cam = default_cam();
        let img =
            rasterize_gaussians(&[], &[], &cam, &cfg).expect("rasterize_gaussians should succeed");
        assert!((img[0] - 0.2).abs() < 1e-5);
        assert!((img[1] - 0.4).abs() < 1e-5);
        assert!((img[2] - 0.6).abs() < 1e-5);
    }

    #[test]
    fn rasterize_gaussian_contributes() {
        let g = make_gaussian_at(5.0);
        let view = identity_view();
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).expect("project_gaussian should succeed");
        let cfg = RasterConfig {
            width: 100,
            height: 100,
            bg_color: [0.0, 0.0, 0.0],
        };
        let img = rasterize_gaussians(&[g], &[pg], &cam, &cfg)
            .expect("rasterize_gaussians should succeed");
        // Center pixel should have some contribution
        let cx = 50usize;
        let cy = 50usize;
        let idx = (cy * 100 + cx) * 3;
        let total: f32 = img[idx] + img[idx + 1] + img[idx + 2];
        assert!(
            total > 0.0,
            "Center pixel should have Gaussian contribution"
        );
    }

    #[test]
    fn rasterize_batch_size_mismatch() {
        let g = make_gaussian_at(5.0);
        let cam = default_cam();
        let cfg = RasterConfig {
            width: 10,
            height: 10,
            bg_color: [0.0; 3],
        };
        assert!(rasterize_gaussians(&[g], &[], &cam, &cfg).is_err());
    }

    #[test]
    fn rasterize_finite_output() {
        let g = make_gaussian_at(3.0);
        let view = identity_view();
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).expect("project_gaussian should succeed");
        let cfg = RasterConfig {
            width: 50,
            height: 50,
            bg_color: [0.1, 0.1, 0.1],
        };
        let img = rasterize_gaussians(&[g], &[pg], &cam, &cfg)
            .expect("rasterize_gaussians should succeed");
        assert!(
            img.iter().all(|v| v.is_finite()),
            "All pixels must be finite"
        );
    }
}
