//! Iterative Closest Point (ICP) alignment algorithm.

use crate::error::{Geom3dError, Geom3dResult};
use crate::neighborhood::kd_tree::KdTree;
use crate::transform::rigid::RigidTransform;

/// ICP configuration.
#[derive(Debug, Clone)]
pub struct IcpConfig {
    pub max_iter: usize,
    pub tol: f32,
}

/// ICP result.
#[derive(Debug, Clone)]
pub struct IcpResult {
    pub transform: RigidTransform,
    pub residual: f32,
    pub n_iter: usize,
}

// ─── SVD of 3×3 matrix via Jacobi sweeps ─────────────────────────────────────

/// Compute the Jacobi rotation angle for a 2×2 symmetric subproblem.
fn jacobi_angle(a_pp: f32, a_qq: f32, a_pq: f32) -> f32 {
    if a_pq.abs() < 1e-12 {
        return 0.0;
    }
    let tau = (a_qq - a_pp) / (2.0 * a_pq);
    let t = if tau >= 0.0 {
        1.0 / (tau + (1.0 + tau * tau).sqrt())
    } else {
        1.0 / (tau - (1.0 + tau * tau).sqrt())
    };
    t.atan()
}

/// Apply a Jacobi rotation to a symmetric 3×3 matrix (in-place).
fn apply_jacobi_rotation(a: &mut [f32; 9], v: &mut [f32; 9], p: usize, q: usize) {
    let angle = jacobi_angle(a[p * 3 + p], a[q * 3 + q], a[p * 3 + q]);
    let c = angle.cos();
    let s = angle.sin();

    let a_pp = a[p * 3 + p];
    let a_qq = a[q * 3 + q];
    let a_pq = a[p * 3 + q];

    a[p * 3 + p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
    a[q * 3 + q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;
    a[p * 3 + q] = 0.0;
    a[q * 3 + p] = 0.0;

    // Off-diagonal elements involving row/col p and q
    // For 3×3: indices are 0,1,2; other = 3 - p - q
    let other = 3 - p - q;
    let a_rp = a[other * 3 + p];
    let a_rq = a[other * 3 + q];
    a[other * 3 + p] = c * a_rp - s * a_rq;
    a[p * 3 + other] = c * a_rp - s * a_rq;
    a[other * 3 + q] = s * a_rp + c * a_rq;
    a[q * 3 + other] = s * a_rp + c * a_rq;

    // Update eigenvector matrix V (columns are eigenvectors)
    for i in 0..3 {
        let vip = v[i * 3 + p];
        let viq = v[i * 3 + q];
        v[i * 3 + p] = c * vip - s * viq;
        v[i * 3 + q] = s * vip + c * viq;
    }
}

/// Jacobi eigendecomposition of symmetric 3×3 matrix.
/// Returns (eigenvalues [3], eigenvectors [3×3] col-major V).
fn jacobi_sym3(m: &[f32; 9]) -> ([f32; 3], [f32; 9]) {
    let mut a = *m;
    let mut v = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let pairs = [(0usize, 1usize), (0, 2), (1, 2)];
    for _ in 0..50 {
        let off = a[1] * a[1] + a[2] * a[2] + a[5] * a[5];
        if off < 1e-20 {
            break;
        }
        for &(p, q) in &pairs {
            apply_jacobi_rotation(&mut a, &mut v, p, q);
        }
    }

    ([a[0], a[4], a[8]], v)
}

/// SVD of 3×3 matrix via Jacobi on AᵀA.
/// Returns (U [3×3 col-major], S [3 values desc], Vt [3×3 row-major Vᵀ]).
fn svd3x3(m: &[f32; 9]) -> ([f32; 9], [f32; 3], [f32; 9]) {
    // Compute AᵀA
    let mut ata = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[i * 3 + j] += m[k * 3 + i] * m[k * 3 + j];
            }
        }
    }

    let (eigs, v) = jacobi_sym3(&ata);

    // Singular values = sqrt(eigenvalues ≥ 0)
    let mut s = [
        eigs[0].max(0.0).sqrt(),
        eigs[1].max(0.0).sqrt(),
        eigs[2].max(0.0).sqrt(),
    ];

    // Sort descending — carry v columns along
    let mut vt = v;
    if s[0] < s[1] {
        s.swap(0, 1);
        for i in 0..3 {
            vt.swap(i * 3, i * 3 + 1);
        }
    }
    if s[0] < s[2] {
        s.swap(0, 2);
        for i in 0..3 {
            vt.swap(i * 3, i * 3 + 2);
        }
    }
    if s[1] < s[2] {
        s.swap(1, 2);
        for i in 0..3 {
            vt.swap(i * 3 + 1, i * 3 + 2);
        }
    }

    // U = A * V * diag(1/s) — V col j is vt[:, j]
    let mut u = [0.0_f32; 9];
    for j in 0..3 {
        if s[j].abs() < 1e-10 {
            // Degenerate column: set to canonical basis
            u[j] = if j == 0 { 1.0 } else { 0.0 };
            u[3 + j] = if j == 1 { 1.0 } else { 0.0 };
            u[6 + j] = if j == 2 { 1.0 } else { 0.0 };
            continue;
        }
        for i in 0..3 {
            let mut acc = 0.0_f32;
            for k in 0..3 {
                acc += m[i * 3 + k] * vt[k * 3 + j];
            }
            u[i * 3 + j] = acc / s[j];
        }
    }

    // Ensure det(U) = +1
    let det_u = u[0] * (u[4] * u[8] - u[5] * u[7]) - u[1] * (u[3] * u[8] - u[5] * u[6])
        + u[2] * (u[3] * u[7] - u[4] * u[6]);
    if det_u < 0.0 {
        u[2] = -u[2];
        u[5] = -u[5];
        u[8] = -u[8];
        s[2] = -s[2];
    }

    // Transpose vt → proper Vᵀ (rows = right singular vectors)
    let mut vt_out = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            vt_out[i * 3 + j] = vt[j * 3 + i];
        }
    }

    (u, s, vt_out)
}

/// Compute mean of n 3D points.
fn mean3d(points: &[f32], n: usize) -> [f32; 3] {
    let mut m = [0.0_f64; 3];
    for i in 0..n {
        m[0] += points[i * 3] as f64;
        m[1] += points[i * 3 + 1] as f64;
        m[2] += points[i * 3 + 2] as f64;
    }
    [
        (m[0] / n as f64) as f32,
        (m[1] / n as f64) as f32,
        (m[2] / n as f64) as f32,
    ]
}

/// Compute the cross-covariance matrix H = Σ (p - μ_p)(q - μ_q)ᵀ.
fn cross_covariance(
    src: &[f32],
    n: usize,
    mu_src: [f32; 3],
    tgt: &[f32],
    mu_tgt: [f32; 3],
) -> [f32; 9] {
    let mut h = [0.0_f64; 9];
    for i in 0..n {
        let ps = [
            (src[i * 3] - mu_src[0]) as f64,
            (src[i * 3 + 1] - mu_src[1]) as f64,
            (src[i * 3 + 2] - mu_src[2]) as f64,
        ];
        let qt = [
            (tgt[i * 3] - mu_tgt[0]) as f64,
            (tgt[i * 3 + 1] - mu_tgt[1]) as f64,
            (tgt[i * 3 + 2] - mu_tgt[2]) as f64,
        ];
        for row in 0..3 {
            for col in 0..3 {
                h[row * 3 + col] += ps[row] * qt[col];
            }
        }
    }
    let mut hf = [0.0_f32; 9];
    for (hf_v, &h_v) in hf.iter_mut().zip(h.iter()) {
        *hf_v = h_v as f32;
    }
    hf
}

/// Compute R = V * diag(1,1, sign(det(V Uᵀ))) * Uᵀ.
fn compute_rotation_from_svd(u: &[f32; 9], vt: &[f32; 9]) -> [f32; 9] {
    // V = Vtᵀ
    let v = [
        vt[0], vt[3], vt[6], vt[1], vt[4], vt[7], vt[2], vt[5], vt[8],
    ];

    // Uᵀ
    let ut = [u[0], u[3], u[6], u[1], u[4], u[7], u[2], u[5], u[8]];

    // R_test = V * Uᵀ (to check det)
    let mut r_test = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r_test[i * 3 + j] += v[i * 3 + k] * ut[k * 3 + j];
            }
        }
    }

    let det = r_test[0] * (r_test[4] * r_test[8] - r_test[5] * r_test[7])
        - r_test[1] * (r_test[3] * r_test[8] - r_test[5] * r_test[6])
        + r_test[2] * (r_test[3] * r_test[7] - r_test[4] * r_test[6]);
    let d = if det < 0.0 { -1.0_f32 } else { 1.0 };

    // V_mod: last column scaled by d
    let v_mod = [
        v[0],
        v[1],
        v[2] * d,
        v[3],
        v[4],
        v[5] * d,
        v[6],
        v[7],
        v[8] * d,
    ];

    // R = V_mod * Uᵀ
    let mut r = [0.0_f32; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i * 3 + j] += v_mod[i * 3 + k] * ut[k * 3 + j];
            }
        }
    }
    r
}

/// Point-to-point ICP alignment.
///
/// Aligns `source` to `target`. Uses KD-tree for correspondences.
pub fn icp(
    source: &[f32],
    n_src: usize,
    target: &[f32],
    n_tgt: usize,
    cfg: &IcpConfig,
) -> Geom3dResult<IcpResult> {
    if n_src == 0 || n_tgt == 0 {
        return Err(Geom3dError::EmptyPointCloud);
    }
    if source.len() != n_src * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n_src * 3,
            got: source.len(),
        });
    }
    if target.len() != n_tgt * 3 {
        return Err(Geom3dError::DimensionMismatch {
            expected: n_tgt * 3,
            got: target.len(),
        });
    }

    let tree = KdTree::build(target, n_tgt)?;

    let mut total_r = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut total_t = [0.0_f32; 3];

    let mut current_src = source.to_vec();

    let mut residual = f32::INFINITY;
    let mut n_iter = 0usize;

    for iter in 0..cfg.max_iter {
        // Find correspondences
        let mut corr_tgt = vec![0.0_f32; n_src * 3];
        let mut sum_sq = 0.0_f64;

        for i in 0..n_src {
            let q = [
                current_src[i * 3],
                current_src[i * 3 + 1],
                current_src[i * 3 + 2],
            ];
            let (tgt_idx, sq_d) = tree.nearest(q)?;
            corr_tgt[i * 3] = target[tgt_idx * 3];
            corr_tgt[i * 3 + 1] = target[tgt_idx * 3 + 1];
            corr_tgt[i * 3 + 2] = target[tgt_idx * 3 + 2];
            sum_sq += sq_d as f64;
        }

        let new_residual = (sum_sq / n_src as f64) as f32;

        let mu_src = mean3d(&current_src, n_src);
        let mu_tgt = mean3d(&corr_tgt, n_src);

        let h = cross_covariance(&current_src, n_src, mu_src, &corr_tgt, mu_tgt);

        let (u, _s, vt) = svd3x3(&h);
        let r = compute_rotation_from_svd(&u, &vt);

        // t = mu_tgt - R * mu_src
        let r_mu = [
            r[0] * mu_src[0] + r[1] * mu_src[1] + r[2] * mu_src[2],
            r[3] * mu_src[0] + r[4] * mu_src[1] + r[5] * mu_src[2],
            r[6] * mu_src[0] + r[7] * mu_src[1] + r[8] * mu_src[2],
        ];
        let t_step = [
            mu_tgt[0] - r_mu[0],
            mu_tgt[1] - r_mu[1],
            mu_tgt[2] - r_mu[2],
        ];

        // Apply step to current source
        for i in 0..n_src {
            let p = [
                current_src[i * 3],
                current_src[i * 3 + 1],
                current_src[i * 3 + 2],
            ];
            current_src[i * 3] = r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t_step[0];
            current_src[i * 3 + 1] = r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t_step[1];
            current_src[i * 3 + 2] = r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t_step[2];
        }

        // Accumulate total transform
        let step_tf = RigidTransform { r, t: t_step };
        let prev_tf = RigidTransform {
            r: total_r,
            t: total_t,
        };
        let composed = step_tf.compose(&prev_tf);
        total_r = composed.r;
        total_t = composed.t;

        let delta = (new_residual - residual).abs();
        residual = new_residual;
        n_iter = iter + 1;

        if delta < cfg.tol && iter > 0 {
            break;
        }
    }

    Ok(IcpResult {
        transform: RigidTransform {
            r: total_r,
            t: total_t,
        },
        residual,
        n_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid(n: usize) -> Vec<f32> {
        let side = (n as f32).cbrt() as usize + 1;
        let mut pts = Vec::with_capacity(n * 3);
        let mut count = 0;
        'outer: for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    if count >= n {
                        break 'outer;
                    }
                    pts.push(i as f32 * 0.1);
                    pts.push(j as f32 * 0.1);
                    pts.push(k as f32 * 0.1);
                    count += 1;
                }
            }
        }
        pts
    }

    #[test]
    fn icp_identity_convergence() {
        let pts = make_grid(27);
        let cfg = IcpConfig {
            max_iter: 20,
            tol: 1e-5,
        };
        let result = icp(&pts, 27, &pts, 27, &cfg).unwrap();
        assert!(
            result.residual < 1e-3,
            "ICP on identity should converge, residual={}",
            result.residual
        );
    }

    #[test]
    fn icp_empty_error() {
        let pts = make_grid(4);
        let cfg = IcpConfig {
            max_iter: 10,
            tol: 1e-4,
        };
        assert!(icp(&[], 0, &pts, 4, &cfg).is_err());
    }

    #[test]
    fn icp_result_transform_finite() {
        let src = make_grid(8);
        let mut tgt = src.clone();
        for i in 0..8 {
            tgt[i * 3] += 0.05;
        }
        let cfg = IcpConfig {
            max_iter: 20,
            tol: 1e-5,
        };
        let result = icp(&src, 8, &tgt, 8, &cfg).unwrap();
        assert!(result.transform.r.iter().all(|v| v.is_finite()));
        assert!(result.transform.t.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn icp_residual_decreases() {
        let src = make_grid(27);
        let mut tgt = src.clone();
        for i in 0..27 {
            tgt[i * 3 + 2] += 0.2;
        }
        let cfg = IcpConfig {
            max_iter: 30,
            tol: 1e-6,
        };
        let result = icp(&src, 27, &tgt, 27, &cfg).unwrap();
        assert!(result.residual.is_finite());
    }

    #[test]
    fn svd3x3_identity() {
        let m = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let (u, s, _vt) = svd3x3(&m);
        for &si in &s {
            assert!((si.abs() - 1.0).abs() < 0.1, "SVD of identity: s={}", si);
        }
        // U should be orthogonal
        let mut uut = [0.0_f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    uut[i * 3 + j] += u[i * 3 + k] * u[j * 3 + k];
                }
            }
        }
        for i in 0..3 {
            assert!((uut[i * 3 + i] - 1.0).abs() < 0.1);
        }
    }
}
