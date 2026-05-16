//! Simple SDP interior-point solver (educational).
//!
//! Solves `min tr(C X)  s.t. tr(A_k X) = b_k, X ⪰ 0` with X symmetric n × n.
//!
//! Uses a basic primal scaling Newton step on `-log det X` with regularised system.
//! This is a self-contained pure-Rust implementation good for tiny problems.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::norm2;
use crate::projection::psd_cone::project_psd_cone;
use crate::sdp::log_det_barrier::{log_det, log_det_gradient};

/// SDP result.
#[derive(Debug, Clone)]
pub struct SdpResult {
    pub x: Vec<f64>,
    pub iter: usize,
    pub objective: f64,
}

/// SDP via projected (sub)gradient with log-det barrier.
///
/// `c` is the cost matrix (`n × n`), `a_list` is m matrices each `n × n`, `b` length m.
/// We minimise `tr(C X) − t · log det X` with shrinking `t`, projecting onto affine subspace.
#[allow(clippy::too_many_arguments)]
pub fn sdp_interior_point(
    c: &[f64],
    n: usize,
    a_list: &[Vec<f64>],
    b: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<SdpResult> {
    if c.len() != n * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![n, n],
            got: vec![c.len()],
        });
    }
    let m = a_list.len();
    if b.len() != m {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    for a in a_list {
        if a.len() != n * n {
            return Err(CvxError::ShapeMismatch {
                expected: vec![n, n],
                got: vec![a.len()],
            });
        }
    }
    // Initialise X = I (assume problem is feasible with X = I as starting point).
    let mut x = vec![0.0_f64; n * n];
    for i in 0..n {
        x[i * n + i] = 1.0;
    }
    let mut t = 1.0_f64;
    let beta = 0.5_f64;
    for it in 0..max_iter {
        // Gradient of objective: tr(C X) + t · (-log det X), so grad = C - t X⁻¹.
        // X⁻¹ via solve_dense (already orchestrated by log_det_gradient: gives -X⁻¹).
        let neg_xinv = log_det_gradient(&x, n)?;
        let mut grad = vec![0.0_f64; n * n];
        for i in 0..(n * n) {
            grad[i] = c[i] + t * neg_xinv[i];
        }
        // Project gradient onto null space of constraints: g_proj = g - Σ a_k (a_k · g / a_k·a_k).
        // Simplified Gram-Schmidt.
        let mut g_proj = grad.clone();
        for a in a_list {
            let dot_ag: f64 = (0..(n * n)).map(|j| a[j] * g_proj[j]).sum();
            let dot_aa: f64 = (0..(n * n)).map(|j| a[j] * a[j]).sum();
            if dot_aa > 1.0e-300 {
                let f = dot_ag / dot_aa;
                for j in 0..(n * n) {
                    g_proj[j] -= f * a[j];
                }
            }
        }
        // Step.
        let step = 0.05_f64;
        let mut x_new = vec![0.0_f64; n * n];
        for j in 0..(n * n) {
            x_new[j] = x[j] - step * g_proj[j];
        }
        // Project onto PSD cone.
        let x_psd = project_psd_cone(&x_new, n)?;
        // Reproject onto constraint set via single-pass correction.
        let mut x_final = x_psd.clone();
        for (k, a) in a_list.iter().enumerate() {
            let dot_ax: f64 = (0..(n * n)).map(|j| a[j] * x_final[j]).sum();
            let dot_aa: f64 = (0..(n * n)).map(|j| a[j] * a[j]).sum();
            if dot_aa > 1.0e-300 {
                let f = (dot_ax - b[k]) / dot_aa;
                for j in 0..(n * n) {
                    x_final[j] -= f * a[j];
                }
            }
        }
        let diff: Vec<f64> = x_final.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let d_nrm = norm2(&diff);
        x = x_final;
        if d_nrm < tol {
            // Compute final objective.
            let mut obj = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    obj += c[i * n + j] * x[j * n + i];
                }
            }
            return Ok(SdpResult {
                x,
                iter: it + 1,
                objective: obj,
            });
        }
        t *= beta;
        let _ = log_det(&x, n).unwrap_or(0.0);
    }
    let mut obj = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            obj += c[i * n + j] * x[j * n + i];
        }
    }
    Ok(SdpResult {
        x,
        iter: max_iter,
        objective: obj,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdp_identity_constraint() {
        // Minimise tr(C X) where C = I; constraint tr(X) = 1, X ⪰ 0.
        // Optimum X = e_k e_k^T for any single eigvec — tr(C X) = 1 for any feasible X.
        let n = 2;
        let c = vec![1.0_f64, 0.0, 0.0, 1.0];
        let a1 = vec![1.0_f64, 0.0, 0.0, 1.0];
        let b = vec![1.0_f64];
        let res = sdp_interior_point(&c, n, &[a1], &b, 200, 1.0e-7).expect("ok");
        let tr_x = res.x[0] + res.x[3];
        assert!((tr_x - 1.0).abs() < 1.0e-3);
        // tr(C X) = tr(X) = 1.
        assert!((res.objective - 1.0).abs() < 1.0e-3);
    }
}
