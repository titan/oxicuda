//! 3D Gaussian projection to 2D screen space.

use crate::error::Geom3dResult;
use crate::gaussian::gaussian::Gaussian3d;

/// Pinhole camera intrinsics.
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub near: f32,
}

/// A projected 2D Gaussian on screen.
#[derive(Debug, Clone)]
pub struct ProjectedGaussian {
    pub xy: [f32; 2],    // screen-space center
    pub cov2d: [f32; 4], // 2×2 covariance (row-major)
    pub depth: f32,
    pub valid: bool,
}

/// Project a single 3D Gaussian to 2D screen space.
///
/// `view`: 3×4 `[R|t]` matrix (row-major 12 floats).
/// `J = [[fx/Z, 0, -fx*X/Z²], [0, fy/Z, -fy*Y/Z²]]`
/// `W = J * R_3x3`
/// `Σ_2d = W * Σ_3d * Wᵀ + 0.3*I`
/// `valid = Z > near`
pub fn project_gaussian(
    g: &Gaussian3d,
    view: &[f32; 12],
    cam: &CameraIntrinsics,
) -> Geom3dResult<ProjectedGaussian> {
    // Apply view transform: p_cam = R * pos + t
    let r = [
        [view[0], view[1], view[2]],
        [view[4], view[5], view[6]],
        [view[8], view[9], view[10]],
    ];
    let t = [view[3], view[7], view[11]];

    let pos = g.pos;
    let x_cam = r[0][0] * pos[0] + r[0][1] * pos[1] + r[0][2] * pos[2] + t[0];
    let y_cam = r[1][0] * pos[0] + r[1][1] * pos[1] + r[1][2] * pos[2] + t[1];
    let z_cam = r[2][0] * pos[0] + r[2][1] * pos[1] + r[2][2] * pos[2] + t[2];

    let valid = z_cam > cam.near;
    if !valid {
        return Ok(ProjectedGaussian {
            xy: [0.0, 0.0],
            cov2d: [1.0, 0.0, 0.0, 1.0],
            depth: z_cam,
            valid: false,
        });
    }

    // Perspective projection
    let x_screen = cam.fx * x_cam / z_cam + cam.cx;
    let y_screen = cam.fy * y_cam / z_cam + cam.cy;

    // Get 3D covariance
    let cov3d = g.covariance3d()?;

    // Jacobian J: 2×3 matrix
    let jac = [
        cam.fx / z_cam,
        0.0,
        -cam.fx * x_cam / (z_cam * z_cam),
        0.0,
        cam.fy / z_cam,
        -cam.fy * y_cam / (z_cam * z_cam),
    ];

    // W = J * R_3x3 (2×3 × 3×3 = 2×3)
    let mut w = [0.0_f32; 6]; // 2×3
    for i in 0..2 {
        for j in 0..3 {
            let mut s = 0.0_f32;
            for k in 0..3 {
                s += jac[i * 3 + k] * r[k][j];
            }
            w[i * 3 + j] = s;
        }
    }

    // Σ_2d = W * Σ_3d * Wᵀ + 0.3*I
    // WΣ = W [2×3] * Σ_3d [3×3] = [2×3]
    let mut w_sigma = [0.0_f32; 6]; // 2×3
    for i in 0..2 {
        for j in 0..3 {
            let mut s = 0.0_f32;
            for k in 0..3 {
                s += w[i * 3 + k] * cov3d[k * 3 + j];
            }
            w_sigma[i * 3 + j] = s;
        }
    }

    // WΣWᵀ: [2×3] * [3×2] = [2×2]
    let mut cov2d = [0.0_f32; 4]; // 2×2
    for i in 0..2 {
        for j in 0..2 {
            let mut s = 0.0_f32;
            for k in 0..3 {
                s += w_sigma[i * 3 + k] * w[j * 3 + k]; // w[j] = Wᵀ column j = W row j
            }
            cov2d[i * 2 + j] = s;
        }
    }

    // Add regularization 0.3*I
    cov2d[0] += 0.3;
    cov2d[3] += 0.3;

    Ok(ProjectedGaussian {
        xy: [x_screen, y_screen],
        cov2d,
        depth: z_cam,
        valid: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_view() -> [f32; 12] {
        [
            1.0, 0.0, 0.0, 0.0, // row 0: R[0] | t[0]
            0.0, 1.0, 0.0, 0.0, // row 1: R[1] | t[1]
            0.0, 0.0, 1.0, 0.0, // row 2: R[2] | t[2]
        ]
    }

    fn default_cam() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
            near: 0.1,
        }
    }

    #[test]
    fn project_valid_depth() {
        let g = Gaussian3d::new_unit([0.0, 0.0, 5.0]); // in front of camera
        let view = identity_view();
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).unwrap();
        assert!(pg.valid, "Gaussian in front should be valid");
        assert!(pg.depth > 0.0, "Depth must be positive");
    }

    #[test]
    fn project_behind_camera_invalid() {
        let g = Gaussian3d::new_unit([0.0, 0.0, -5.0]); // behind camera
        let view = identity_view();
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).unwrap();
        assert!(!pg.valid, "Gaussian behind camera should be invalid");
    }

    #[test]
    fn project_screen_coords() {
        // At exactly optical center direction
        let g = Gaussian3d::new_unit([0.0, 0.0, 10.0]);
        let view = identity_view();
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).unwrap();
        // x_screen = fx * 0/10 + cx = cx = 320
        assert!((pg.xy[0] - 320.0).abs() < 1e-3, "x={}", pg.xy[0]);
        assert!((pg.xy[1] - 240.0).abs() < 1e-3, "y={}", pg.xy[1]);
    }

    #[test]
    fn project_cov2d_positive_diagonal() {
        let g = Gaussian3d::new_unit([1.0, 0.0, 5.0]);
        let view = identity_view();
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).unwrap();
        assert!(pg.cov2d[0] > 0.0, "cov2d[0] must be positive");
        assert!(pg.cov2d[3] > 0.0, "cov2d[3] must be positive");
    }

    #[test]
    fn project_translation_in_view() {
        // Camera at offset: t = [0,0,-5] brings origin to z=5
        let g = Gaussian3d::new_unit([0.0, 0.0, 0.0]);
        let view = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 5.0, // t_z = 5
        ];
        let cam = default_cam();
        let pg = project_gaussian(&g, &view, &cam).unwrap();
        assert!(pg.valid, "Should be valid with t_z=5");
        assert!((pg.depth - 5.0).abs() < 1e-3);
    }
}
