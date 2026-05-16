//! Vanilla ADMM in scaled form for `min f(x) + g(z)  s.t. Ax + Bz = c`.
//!
//! Updates (scaled dual `u = y/ρ`):
//!   x ← argmin_x  f(x) + ρ/2 ||A x + B z − c + u||²
//!   z ← argmin_z  g(z) + ρ/2 ||A x + B z − c + u||²
//!   u ← u + (A x + B z − c)
//!
//! `x_update(z, u)` produces the new `x` (problem-specific solver — typically uses prox of f).
//! Similarly `z_update(x, u)`.

use crate::error::{CvxError, CvxResult};
use crate::linalg::matvec::{mat_vec, norm2};

/// ADMM solver. `a`, `b` are constraint matrices (row-major), `c` constraint RHS.
#[allow(clippy::too_many_arguments)]
pub fn admm_solve<X, Z>(
    a: &[f64],
    am: usize,
    an: usize,
    b: &[f64],
    bn: usize,
    c: &[f64],
    rho: f64,
    x_update: X,
    z_update: Z,
    max_iter: usize,
    tol_pri: f64,
    tol_dual: f64,
) -> CvxResult<AdmmResult>
where
    X: Fn(&[f64], &[f64]) -> CvxResult<Vec<f64>>,
    Z: Fn(&[f64], &[f64]) -> CvxResult<Vec<f64>>,
{
    if rho <= 0.0 || !rho.is_finite() {
        return Err(CvxError::InvalidParameter(format!(
            "ADMM rho > 0 required, got {rho}"
        )));
    }
    if a.len() != am * an {
        return Err(CvxError::ShapeMismatch {
            expected: vec![am, an],
            got: vec![a.len()],
        });
    }
    if b.len() != am * bn {
        return Err(CvxError::ShapeMismatch {
            expected: vec![am, bn],
            got: vec![b.len()],
        });
    }
    if c.len() != am {
        return Err(CvxError::DimensionMismatch { a: c.len(), b: am });
    }
    let mut x = vec![0.0_f64; an];
    let mut z = vec![0.0_f64; bn];
    let mut u = vec![0.0_f64; am];
    let mut iters = 0usize;
    let mut pri_norm = 0.0_f64;
    let mut dual_norm = 0.0_f64;
    for it in 0..max_iter {
        // x update.
        let x_new = x_update(&z, &u)?;
        if x_new.len() != an {
            return Err(CvxError::DimensionMismatch {
                a: x_new.len(),
                b: an,
            });
        }
        // z update (uses new x).
        let z_new = z_update(&x_new, &u)?;
        if z_new.len() != bn {
            return Err(CvxError::DimensionMismatch {
                a: z_new.len(),
                b: bn,
            });
        }
        // residual = A x + B z - c
        let ax = mat_vec(a, am, an, &x_new)?;
        let bz = mat_vec(b, am, bn, &z_new)?;
        let mut r = vec![0.0_f64; am];
        for i in 0..am {
            r[i] = ax[i] + bz[i] - c[i];
        }
        // u update.
        for i in 0..am {
            u[i] += r[i];
        }
        // Primal residual norm.
        pri_norm = norm2(&r);
        // Dual residual: ρ · A^T B (z_new − z).
        let mut dz = vec![0.0_f64; bn];
        for i in 0..bn {
            dz[i] = z_new[i] - z[i];
        }
        let b_dz = mat_vec(b, am, bn, &dz)?;
        // ρ * A^T * B Δz.
        let mut s = vec![0.0_f64; an];
        for (i, &bdz_i) in b_dz.iter().enumerate().take(am) {
            let row = i * an;
            for j in 0..an {
                s[j] += rho * a[row + j] * bdz_i;
            }
        }
        dual_norm = norm2(&s);
        x = x_new;
        z = z_new;
        iters = it + 1;
        if pri_norm < tol_pri && dual_norm < tol_dual {
            break;
        }
    }
    Ok(AdmmResult {
        x,
        z,
        u,
        iter: iters,
        pri_residual: pri_norm,
        dual_residual: dual_norm,
    })
}

/// ADMM result bundle.
#[derive(Debug, Clone)]
pub struct AdmmResult {
    pub x: Vec<f64>,
    pub z: Vec<f64>,
    pub u: Vec<f64>,
    pub iter: usize,
    pub pri_residual: f64,
    pub dual_residual: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prox_ops::l1::soft_threshold;

    #[test]
    fn admm_basic_lasso() {
        // min 0.5 ||x - b||² + lambda ||z||₁ s.t. x - z = 0.
        // x update: prox_{(1/ρ)·0.5||·-b||²}(z - u) = (b + ρ(z - u)) / (1 + ρ).
        // z update: prox_{(λ/ρ)||·||₁}(x + u).
        let b = vec![3.0_f64, -2.0, 0.5];
        let lambda = 1.0_f64;
        let rho = 1.0_f64;
        let an = b.len();
        let bn = an;
        let am = an;
        // A = I, B = -I, c = 0.
        let mut a_mat = vec![0.0_f64; am * an];
        let mut b_mat = vec![0.0_f64; am * bn];
        for i in 0..am {
            a_mat[i * an + i] = 1.0;
            b_mat[i * bn + i] = -1.0;
        }
        let c = vec![0.0_f64; am];
        let b_clone = b.clone();
        let xu = |z: &[f64], u: &[f64]| -> CvxResult<Vec<f64>> {
            // x_i = (b_i + rho (z_i - u_i)) / (1 + rho)
            Ok((0..an)
                .map(|i| (b_clone[i] + rho * (z[i] - u[i])) / (1.0 + rho))
                .collect())
        };
        let zu = |x: &[f64], u: &[f64]| -> CvxResult<Vec<f64>> {
            // soft-threshold (x + u) by lambda/rho.
            Ok((0..bn)
                .map(|i| soft_threshold(x[i] + u[i], lambda / rho))
                .collect())
        };
        let res = admm_solve(
            &a_mat, am, an, &b_mat, bn, &c, rho, xu, zu, 500, 1.0e-8, 1.0e-8,
        )
        .expect("ok");
        // True solution: prox_{||·||₁}(b) = [2, -1, 0].
        assert!((res.z[0] - 2.0).abs() < 1.0e-4);
        assert!((res.z[1] + 1.0).abs() < 1.0e-4);
        assert!(res.z[2].abs() < 1.0e-4);
    }
}
