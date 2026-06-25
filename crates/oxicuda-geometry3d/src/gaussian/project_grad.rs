//! Differentiable 3D→2D Gaussian projection (analytic backward).
//!
//! This complements [`crate::gaussian::project::project_gaussian`] with the
//! Jacobians required for differentiable rendering (back-propagating image-space
//! gradients into world-space Gaussian parameters).
//!
//! The projection is, with `p_cam = R·μ + t`, `Z = p_cam.z`:
//!
//! ```text
//! u = fx·X/Z + cx ,  v = fy·Y/Z + cy
//! J = [[fx/Z, 0, -fx·X/Z²], [0, fy/Z, -fy·Y/Z²]]   (2×3)
//! W = J·R                                          (2×3)
//! Σ_2d = W · Σ_3d · Wᵀ   (the +s·I low-pass is gradient-constant)
//! ```
//!
//! [`project_backward`] takes upstream gradients `dL/d(u,v)` and `dL/dΣ_2d` and
//! returns `dL/dμ` (world-space 3D mean) and `dL/dΣ_3d` (row-major 3×3). The
//! mean gradient threads through both the screen position *and* the
//! covariance's dependence on `μ` through the Jacobian `J(Z, X, Y)`.

use crate::error::{Geom3dError, Geom3dResult};
use crate::gaussian::project::CameraIntrinsics;

/// Per-projection gradients w.r.t. the 3D Gaussian parameters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProjectGrad {
    /// dL/dμ — gradient w.r.t. the 3D mean (world space).
    pub d_mean3d: [f32; 3],
    /// dL/dΣ_3d — gradient w.r.t. the 3D covariance, row-major 3×3.
    pub d_cov3d: [f32; 9],
}

/// Row-major 3×3 matrix-vector product.
fn mat3_vec(m: &[f32; 9], v: &[f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

/// Extract the 3×3 rotation block and translation from a 3×4 row-major view.
fn split_view(view: &[f32; 12]) -> ([f32; 9], [f32; 3]) {
    let r = [
        view[0], view[1], view[2], view[4], view[5], view[6], view[8], view[9], view[10],
    ];
    let t = [view[3], view[7], view[11]];
    (r, t)
}

/// Compute the 2×3 projection Jacobian `J` for a camera-space point.
fn jacobian(cam: &CameraIntrinsics, p_cam: &[f32; 3]) -> [f32; 6] {
    let (x, y, z) = (p_cam[0], p_cam[1], p_cam[2]);
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    [
        cam.fx * inv_z,
        0.0,
        -cam.fx * x * inv_z2,
        0.0,
        cam.fy * inv_z,
        -cam.fy * y * inv_z2,
    ]
}

/// `W = J · R`, `J` is 2×3, `R` is row-major 3×3, result 2×3.
fn w_matrix(jac: &[f32; 6], r: &[f32; 9]) -> [f32; 6] {
    let mut w = [0.0_f32; 6];
    for i in 0..2 {
        for j in 0..3 {
            let mut s = 0.0_f32;
            for k in 0..3 {
                s += jac[i * 3 + k] * r[k * 3 + j];
            }
            w[i * 3 + j] = s;
        }
    }
    w
}

/// Screen-position gradient: returns dL/dμ from `dL/d(u,v)` alone.
///
/// `du/dμ = (∂u/∂p_cam)·R`, with `∂u/∂p_cam = (fx/Z, 0, -fx·X/Z²)` (the first
/// row of `J`) and likewise for `v`.
fn mean_grad_from_screen(d_uv: &[f32; 2], jac: &[f32; 6], r: &[f32; 9]) -> [f32; 3] {
    // dL/dp_cam = J^T · d_uv  (3-vector)
    let dp_cam = [
        jac[0] * d_uv[0] + jac[3] * d_uv[1],
        jac[1] * d_uv[0] + jac[4] * d_uv[1],
        jac[2] * d_uv[0] + jac[5] * d_uv[1],
    ];
    // p_cam = R·μ + t  ⇒  dL/dμ = R^T · dL/dp_cam
    [
        r[0] * dp_cam[0] + r[3] * dp_cam[1] + r[6] * dp_cam[2],
        r[1] * dp_cam[0] + r[4] * dp_cam[1] + r[7] * dp_cam[2],
        r[2] * dp_cam[0] + r[5] * dp_cam[1] + r[8] * dp_cam[2],
    ]
}

/// Back-propagate image-space gradients through the projection.
///
/// * `view` — 3×4 row-major `[R|t]`.
/// * `cam` — pinhole intrinsics.
/// * `mean3d` — the 3D Gaussian mean (world space).
/// * `cov3d` — the 3D covariance, row-major 3×3.
/// * `d_xy` — upstream `dL/d(u, v)` in screen pixels.
/// * `d_cov2d` — upstream `dL/dΣ_2d`, row-major 2×2 (symmetric).
///
/// Returns the gradients packed in [`ProjectGrad`].
///
/// # Errors
///
/// Returns [`Geom3dError::NanEncountered`] if the point is at or behind the
/// camera (`Z <= near`), where the projection Jacobian is undefined.
pub fn project_backward(
    view: &[f32; 12],
    cam: &CameraIntrinsics,
    mean3d: &[f32; 3],
    cov3d: &[f32; 9],
    d_xy: &[f32; 2],
    d_cov2d: &[f32; 4],
) -> Geom3dResult<ProjectGrad> {
    let (r, t) = split_view(view);
    let p_cam = [
        mat3_vec(&r, mean3d)[0] + t[0],
        mat3_vec(&r, mean3d)[1] + t[1],
        mat3_vec(&r, mean3d)[2] + t[2],
    ];
    if p_cam[2] <= cam.near {
        return Err(Geom3dError::NanEncountered {
            location: "project_backward: point at or behind camera",
        });
    }

    let jac = jacobian(cam, &p_cam);
    let w = w_matrix(&jac, &r);

    // --- Gradient w.r.t. Σ_3d ---------------------------------------------
    // Σ_2d = W·Σ_3d·Wᵀ ⇒ dL/dΣ_3d = Wᵀ·(dL/dΣ_2d)·W (3×3), then symmetrize.
    // Build symmetric upstream Ḡ = dL/dΣ_2d.
    let g = *d_cov2d; // [g00, g01, g10, g11]
    // tmp = Ḡ · W  (2×2 · 2×3 = 2×3)
    let mut tmp = [0.0_f32; 6];
    for i in 0..2 {
        for j in 0..3 {
            tmp[i * 3 + j] = g[i * 2] * w[j] + g[i * 2 + 1] * w[3 + j];
        }
    }
    // dcov3 = Wᵀ · tmp  (3×2 · 2×3 = 3×3)
    let mut d_cov3 = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            d_cov3[i * 3 + j] = w[i] * tmp[j] + w[3 + i] * tmp[3 + j];
        }
    }

    // --- Gradient w.r.t. the mean -----------------------------------------
    // (a) through the screen position (u, v).
    let mut d_mean = mean_grad_from_screen(d_xy, &jac, &r);

    // (b) through Σ_2d's dependence on μ via J(X, Y, Z). We obtain this with a
    // tight analytic directional derivative: Σ_2d depends on μ only through the
    // camera-space coordinates, so dΣ_2d/dμ = (dΣ_2d/dp_cam)·R. We compute
    // dL/dp_cam by differentiating Σ_2d = (J R) Σ (J R)ᵀ w.r.t. each component
    // of p_cam through J, contract with Ḡ, then map to μ with Rᵀ.
    let (x, y, z) = (p_cam[0], p_cam[1], p_cam[2]);
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let inv_z3 = inv_z2 * inv_z;

    // dJ/dX, dJ/dY, dJ/dZ as 2×3 matrices (only entries that depend on each).
    // J = [[fx/Z,0,-fx X/Z²],[0,fy/Z,-fy Y/Z²]]
    let dj_dx = [0.0, 0.0, -cam.fx * inv_z2, 0.0, 0.0, 0.0];
    let dj_dy = [0.0, 0.0, 0.0, 0.0, 0.0, -cam.fy * inv_z2];
    let dj_dz = [
        -cam.fx * inv_z2,
        0.0,
        2.0 * cam.fx * x * inv_z3,
        0.0,
        -cam.fy * inv_z2,
        2.0 * cam.fy * y * inv_z3,
    ];

    // For a given dJ, dW = dJ·R, and dΣ_2d = dW·Σ_3d·Wᵀ + W·Σ_3d·dWᵀ.
    // ⟨Ḡ, dΣ_2d⟩ = 2·⟨Ḡ, dW·(Σ_3d·Wᵀ)⟩  (Ḡ symmetric).
    // Precompute M = Σ_3d · Wᵀ  (3×2).
    let mut sw_t = [0.0_f32; 6]; // M, row-major 3×2
    for i in 0..3 {
        for j in 0..2 {
            let mut s = 0.0_f32;
            for k in 0..3 {
                s += cov3d[i * 3 + k] * w[j * 3 + k];
            }
            sw_t[i * 2 + j] = s;
        }
    }
    let contract = |dj: &[f32; 6]| -> f32 {
        // dW = dJ · R (2×3)
        let dw = w_matrix(dj, &r);
        // dΣ contribution ⟨Ḡ, 2·dW·M⟩ where M=sw_t (3×2). dW·M is 2×2.
        let mut acc = 0.0_f32;
        for i in 0..2 {
            for j in 0..2 {
                let mut e = 0.0_f32;
                for k in 0..3 {
                    e += dw[i * 3 + k] * sw_t[k * 2 + j];
                }
                acc += g[i * 2 + j] * 2.0 * e;
            }
        }
        acc
    };
    let dcov_dp_cam = [contract(&dj_dx), contract(&dj_dy), contract(&dj_dz)];
    // Map dL/dp_cam (from the covariance path) into μ via Rᵀ and add.
    d_mean[0] += r[0] * dcov_dp_cam[0] + r[3] * dcov_dp_cam[1] + r[6] * dcov_dp_cam[2];
    d_mean[1] += r[1] * dcov_dp_cam[0] + r[4] * dcov_dp_cam[1] + r[7] * dcov_dp_cam[2];
    d_mean[2] += r[2] * dcov_dp_cam[0] + r[5] * dcov_dp_cam[1] + r[8] * dcov_dp_cam[2];

    Ok(ProjectGrad {
        d_mean3d: d_mean,
        d_cov3d: d_cov3,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::gaussian::Gaussian3d;
    use crate::gaussian::project::project_gaussian;

    fn default_cam() -> CameraIntrinsics {
        CameraIntrinsics {
            fx: 400.0,
            fy: 420.0,
            cx: 64.0,
            cy: 64.0,
            near: 0.1,
        }
    }

    // A non-axis-aligned view (small yaw) so R is a genuine rotation.
    fn yaw_view(angle: f32, tz: f32) -> [f32; 12] {
        let (s, c) = angle.sin_cos();
        [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, tz]
    }

    fn make_gaussian(pos: [f32; 3], scale: [f32; 3], rot: [f32; 4]) -> Gaussian3d {
        Gaussian3d {
            pos,
            rot,
            scale,
            opacity: 0.0,
            sh: vec![0.0; 27],
        }
    }

    // Project and return (u, v, cov2d) directly from a 3D mean+cov.
    fn project_raw(
        view: &[f32; 12],
        cam: &CameraIntrinsics,
        mean: &[f32; 3],
        cov3d: &[f32; 9],
    ) -> ([f32; 2], [f32; 4]) {
        let (r, t) = split_view(view);
        let p = [
            mat3_vec(&r, mean)[0] + t[0],
            mat3_vec(&r, mean)[1] + t[1],
            mat3_vec(&r, mean)[2] + t[2],
        ];
        let u = cam.fx * p[0] / p[2] + cam.cx;
        let v = cam.fy * p[1] / p[2] + cam.cy;
        let jac = jacobian(cam, &p);
        let w = w_matrix(&jac, &r);
        // Σ_2d = W Σ Wᵀ
        let mut wsig = [0.0_f32; 6];
        for i in 0..2 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += w[i * 3 + k] * cov3d[k * 3 + j];
                }
                wsig[i * 3 + j] = s;
            }
        }
        let mut cov2 = [0.0_f32; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += wsig[i * 3 + k] * w[j * 3 + k];
                }
                cov2[i * 2 + j] = s;
            }
        }
        ([u, v], cov2)
    }

    #[test]
    fn matches_project_gaussian_position() {
        // Sanity: our raw projection agrees with the library projection's xy.
        let cam = default_cam();
        let view = yaw_view(0.2, 6.0);
        let g = make_gaussian([0.3, -0.4, 0.0], [-1.0, -1.2, -0.8], [1.0, 0.0, 0.0, 0.0]);
        let cov3d = g.covariance3d().expect("cov should succeed");
        let (uv, _) = project_raw(&view, &cam, &g.pos, &cov3d);
        let pg = project_gaussian(&g, &view, &cam).expect("project should succeed");
        assert!((uv[0] - pg.xy[0]).abs() < 1e-2);
        assert!((uv[1] - pg.xy[1]).abs() < 1e-2);
    }

    #[test]
    fn grad_mean_matches_numeric() {
        let cam = default_cam();
        let view = yaw_view(0.25, 6.0);
        let mean = [0.2, -0.3, 0.1];
        let g = make_gaussian(mean, [-0.9, -1.1, -1.0], [1.0, 0.0, 0.0, 0.0]);
        let cov3d = g.covariance3d().expect("cov should succeed");

        // Upstream gradients: arbitrary fixed weights.
        let d_xy = [0.7_f32, -0.4];
        let d_cov2d = [0.3_f32, 0.1, 0.1, -0.2];

        let grad = project_backward(&view, &cam, &mean, &cov3d, &d_xy, &d_cov2d)
            .expect("backward should succeed");

        let loss = |m: &[f32; 3]| -> f32 {
            let (uv, c2) = project_raw(&view, &cam, m, &cov3d);
            d_xy[0] * uv[0]
                + d_xy[1] * uv[1]
                + d_cov2d[0] * c2[0]
                + d_cov2d[1] * c2[1]
                + d_cov2d[2] * c2[2]
                + d_cov2d[3] * c2[3]
        };
        let eps = 1e-4_f32;
        for axis in 0..3 {
            let mut mp = mean;
            mp[axis] += eps;
            let mut mm = mean;
            mm[axis] -= eps;
            let num = (loss(&mp) - loss(&mm)) / (2.0 * eps);
            let ana = grad.d_mean3d[axis];
            assert!(
                (num - ana).abs() < 1e-2 * (1.0 + num.abs()),
                "d_mean[{axis}]: num {num} vs ana {ana}"
            );
        }
    }

    #[test]
    fn grad_cov3d_matches_numeric() {
        let cam = default_cam();
        let view = yaw_view(0.15, 7.0);
        let mean = [-0.1, 0.25, -0.2];
        let g = make_gaussian(mean, [-0.8, -1.0, -0.9], [1.0, 0.0, 0.0, 0.0]);
        let cov3d = g.covariance3d().expect("cov should succeed");

        let d_xy = [0.0_f32, 0.0]; // isolate the covariance path
        let d_cov2d = [0.5_f32, -0.2, -0.2, 0.4];
        let grad = project_backward(&view, &cam, &mean, &cov3d, &d_xy, &d_cov2d)
            .expect("backward should succeed");

        let loss = |c: &[f32; 9]| -> f32 {
            let (_, c2) = project_raw(&view, &cam, &mean, c);
            d_cov2d[0] * c2[0] + d_cov2d[1] * c2[1] + d_cov2d[2] * c2[2] + d_cov2d[3] * c2[3]
        };
        let eps = 1e-4_f32;
        // Check every entry of the 3×3 covariance gradient.
        for idx in 0..9 {
            let mut cp = cov3d;
            cp[idx] += eps;
            let mut cm = cov3d;
            cm[idx] -= eps;
            let num = (loss(&cp) - loss(&cm)) / (2.0 * eps);
            let ana = grad.d_cov3d[idx];
            assert!(
                (num - ana).abs() < 1e-2 * (1.0 + num.abs()),
                "d_cov3d[{idx}]: num {num} vs ana {ana}"
            );
        }
    }

    #[test]
    fn behind_camera_errors() {
        let cam = default_cam();
        let view = yaw_view(0.0, 0.0); // no translation → z stays at mean.z
        let mean = [0.0, 0.0, -1.0]; // behind camera
        let cov3d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(project_backward(&view, &cam, &mean, &cov3d, &[0.0, 0.0], &[0.0; 4]).is_err());
    }

    #[test]
    fn zero_upstream_zero_grad() {
        let cam = default_cam();
        let view = yaw_view(0.1, 6.0);
        let mean = [0.1, 0.1, 0.0];
        let cov3d = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let grad = project_backward(&view, &cam, &mean, &cov3d, &[0.0, 0.0], &[0.0; 4])
            .expect("backward should succeed");
        assert!(grad.d_mean3d.iter().all(|v| v.abs() < 1e-9));
        assert!(grad.d_cov3d.iter().all(|v| v.abs() < 1e-9));
    }
}
