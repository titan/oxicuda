//! Solver for a single-cone SOCP `min c^T x  s.t. A x = b,  x ∈ K_soc`.
//!
//! Implementation strategy: alternating projection + dual ascent.
//!
//! 1. `dy` is a Lagrange multiplier for the equality constraint.
//! 2. Each outer iter:
//!    - Compute gradient of Lagrangian: g = c + A^T y.
//!    - Move x in -g direction; project onto SOC; project onto affine `Ax=b`.
//!    - Update y via gradient ascent on dual: y ← y + ρ (Ax − b).
//!
//! This is correct for small problems but not as fast as a full primal-dual IPM.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_t_vec, mat_vec, norm2};
use crate::linalg::solve::solve_dense;
use crate::projection::project_soc;

/// SOCP result.
#[derive(Debug, Clone)]
pub struct SocpResult {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub iter: usize,
    pub residual: f64,
}

/// Solve `min c^T x s.t. A x = b, x ∈ SOC`.
///
/// `x = (t, w)`, t ≥ ||w||.
pub fn primal_dual_socp(
    a: &[f64],
    m: usize,
    n: usize,
    b: &[f64],
    c: &[f64],
    max_iter: usize,
    tol: f64,
) -> CvxResult<SocpResult> {
    if n < 1 {
        return Err(CvxError::InvalidParameter(
            "SOCP requires n≥1 (t coord)".into(),
        ));
    }
    if a.len() != m * n {
        return Err(CvxError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }
    if b.len() != m || c.len() != n {
        return Err(CvxError::DimensionMismatch { a: b.len(), b: m });
    }
    // Form AA^T for projection onto affine subspace.
    let mut aat = vec![0.0_f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += a[i * n + k] * a[j * n + k];
            }
            aat[i * m + j] = acc;
        }
    }
    for i in 0..m {
        aat[i * m + i] += 1.0e-12;
    }
    // Initial x: project 0 onto Ax=b.
    let dy0 = solve_dense(&aat, m, b)?;
    let mut x = mat_t_vec(a, m, n, &dy0)?;
    // Force into cone.
    let (t0, w0) = (x[0], x[1..].to_vec());
    let (t_new, w_new) = project_soc(t0, &w0)?;
    x[0] = t_new;
    for (j, wj) in w_new.into_iter().enumerate() {
        x[j + 1] = wj;
    }
    let mut y = vec![0.0_f64; m];
    let mut residual = f64::INFINITY;
    for it in 0..max_iter {
        // Lagrangian gradient: g = c + A^T y.
        let aty = mat_t_vec(a, m, n, &y)?;
        let mut g = vec![0.0_f64; n];
        for j in 0..n {
            g[j] = c[j] + aty[j];
        }
        // Gradient step.
        let step = 0.05_f64;
        let mut x_new = vec![0.0_f64; n];
        for j in 0..n {
            x_new[j] = x[j] - step * g[j];
        }
        // Project onto SOC.
        let (tt, ww) = project_soc(x_new[0], &x_new[1..])?;
        x_new[0] = tt;
        for (j, wj) in ww.into_iter().enumerate() {
            x_new[j + 1] = wj;
        }
        // Project onto affine subspace Ax = b: x ← x - A^T (AA^T)^{-1} (Ax - b).
        let ax = mat_vec(a, m, n, &x_new)?;
        let mut diff = vec![0.0_f64; m];
        for i in 0..m {
            diff[i] = ax[i] - b[i];
        }
        let corr = solve_dense(&aat, m, &diff)?;
        let at_corr = mat_t_vec(a, m, n, &corr)?;
        for j in 0..n {
            x_new[j] -= at_corr[j];
        }
        // Re-project onto SOC (alternating projection).
        let (tt2, ww2) = project_soc(x_new[0], &x_new[1..])?;
        x_new[0] = tt2;
        for (j, wj) in ww2.into_iter().enumerate() {
            x_new[j + 1] = wj;
        }
        // Update dual via ascent.
        let ax2 = mat_vec(a, m, n, &x_new)?;
        let mut feas = vec![0.0_f64; m];
        for i in 0..m {
            feas[i] = ax2[i] - b[i];
        }
        let rho = 1.0_f64;
        for i in 0..m {
            y[i] += rho * feas[i];
        }
        let diff_x: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        let dn = norm2(&diff_x);
        residual = norm2(&feas).max(dn);
        x = x_new;
        if residual < tol {
            return Ok(SocpResult {
                x,
                y,
                iter: it + 1,
                residual,
            });
        }
    }
    Ok(SocpResult {
        x,
        y,
        iter: max_iter,
        residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn socp_unit_minimisation() {
        // min t s.t. t = 1.  Optimum t=1, w=0.
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![1.0_f64];
        let c = vec![1.0_f64, 0.0, 0.0];
        let res = primal_dual_socp(&a, 1, 3, &b, &c, 500, 1.0e-7).expect("ok");
        assert!((res.x[0] - 1.0).abs() < 1.0e-3);
    }
}
