//! QMR (Quasi-Minimal Residual) iterative solver.
//!
//! Solves a **non-symmetric** linear system `A * x = b` using the Quasi-Minimal
//! Residual method of Freund & Nachtigal (1991), "QMR: a quasi-minimal residual
//! method for non-Hermitian linear systems", Numer. Math. 60, 315–339.
//!
//! QMR is built on the **two-sided (non-symmetric) Lanczos** process, which
//! generates bi-orthogonal Krylov bases for `A` and `Aᵀ` with a short three-term
//! recurrence. Compared with BiCGSTAB, QMR produces a much smoother residual
//! curve (it "quasi-minimises" the residual via a least-squares update of the
//! tridiagonal factor) and is less prone to the erratic stagnation that can
//! afflict BiCG-type methods. Compared with GMRES it uses constant `O(n)` memory
//! — no growing Krylov basis — at the cost of requiring a transpose-product.
//!
//! Because the two-sided Lanczos needs the action of both `A` and `Aᵀ`, the
//! matrix-free interface here takes **two** closures: `spmv` for `y = A·x` and
//! `spmv_t` for `y = Aᵀ·x`.
//!
//! This is the *unpreconditioned, look-ahead-free* variant. A serious Lanczos
//! breakdown (`v·w ≈ 0` with neither factor zero) is reported as an error rather
//! than silently producing a wrong answer.

#![allow(dead_code)]

use oxicuda_blas::GpuFloat;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;

// ---------------------------------------------------------------------------
// GpuFloat <-> f64 conversion helpers
// ---------------------------------------------------------------------------

/// Converts a `GpuFloat` value to `f64` via bit reinterpretation.
fn to_f64<T: GpuFloat>(val: T) -> f64 {
    if T::SIZE == 4 {
        f32::from_bits(val.to_bits_u64() as u32) as f64
    } else {
        f64::from_bits(val.to_bits_u64())
    }
}

/// Converts an `f64` value to `T: GpuFloat` via bit reinterpretation.
fn from_f64<T: GpuFloat>(val: f64) -> T {
    if T::SIZE == 4 {
        T::from_bits_u64(u64::from((val as f32).to_bits()))
    } else {
        T::from_bits_u64(val.to_bits())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`qmr_solve`] solver.
#[derive(Debug, Clone)]
pub struct QmrConfig {
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Convergence tolerance (relative to `‖b‖₂`).
    pub tol: f64,
}

impl Default for QmrConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves `A · x = b` for a general (non-symmetric) matrix using QMR.
///
/// Two matrix-free closures must be supplied:
/// * `spmv`  — computes `y = A · x`,
/// * `spmv_t` — computes `y = Aᵀ · x`.
///
/// On entry `x` holds the initial guess; on exit it holds the approximate
/// solution. Returns the number of iterations performed.
///
/// # Errors
///
/// * [`SolverError::DimensionMismatch`] if `b` or `x` are shorter than `n`.
/// * [`SolverError::InternalError`] on a serious Lanczos / QMR breakdown.
/// * [`SolverError::ConvergenceFailure`] if the residual tolerance is not reached
///   within `config.max_iter` iterations.
pub fn qmr_solve<T, FA, FT>(
    _handle: &SolverHandle,
    spmv: FA,
    spmv_t: FT,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &QmrConfig,
) -> SolverResult<u32>
where
    T: GpuFloat,
    FA: Fn(&[T], &mut [T]) -> SolverResult<()>,
    FT: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "qmr_solve: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "qmr_solve: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(0);
    }

    let b64: Vec<f64> = (0..n_usize).map(|i| to_f64(b[i])).collect();
    let mut x64: Vec<f64> = (0..n_usize).map(|i| to_f64(x[i])).collect();

    let spmv64 = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
        let vt: Vec<T> = v.iter().map(|&c| from_f64::<T>(c)).collect();
        let mut ot = vec![T::gpu_zero(); n_usize];
        spmv(&vt, &mut ot)?;
        for (o, ot_i) in out.iter_mut().zip(ot.iter()) {
            *o = to_f64(*ot_i);
        }
        Ok(())
    };
    let spmv_t64 = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
        let vt: Vec<T> = v.iter().map(|&c| from_f64::<T>(c)).collect();
        let mut ot = vec![T::gpu_zero(); n_usize];
        spmv_t(&vt, &mut ot)?;
        for (o, ot_i) in out.iter_mut().zip(ot.iter()) {
            *o = to_f64(*ot_i);
        }
        Ok(())
    };

    let iters = qmr_f64(&spmv64, &spmv_t64, &b64, &mut x64, n_usize, config)?;

    for (xi, &v) in x.iter_mut().zip(x64.iter()) {
        *xi = from_f64::<T>(v);
    }
    Ok(iters)
}

/// Core QMR iteration in `f64` (Freund–Nachtigal 1991, unpreconditioned).
///
/// Uses the two-sided Lanczos recurrence with the standard QMR scalar names
/// (`rho`, `xi`, `gamma`, `eta`, `delta`, `epsilon`, `beta`, `theta`) so the
/// update can be checked against the published algorithm. No look-ahead.
fn qmr_f64<FA, FT>(
    spmv: &FA,
    spmv_t: &FT,
    b: &[f64],
    x: &mut [f64],
    n: usize,
    config: &QmrConfig,
) -> SolverResult<u32>
where
    FA: Fn(&[f64], &mut [f64]) -> SolverResult<()>,
    FT: Fn(&[f64], &mut [f64]) -> SolverResult<()>,
{
    const TINY: f64 = 1e-300;

    let b_norm = dot(b, b, n).sqrt();
    if b_norm == 0.0 {
        for xi in x.iter_mut().take(n) {
            *xi = 0.0;
        }
        return Ok(0);
    }
    let abs_tol = config.tol * b_norm;

    // r = b - A x0
    let mut ax = vec![0.0f64; n];
    spmv(x, &mut ax)?;
    let mut r: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();

    let mut res_norm = dot(&r, &r, n).sqrt();
    if res_norm < abs_tol {
        return Ok(0);
    }

    // Two-sided Lanczos starting vectors: v_tilde = r, w_tilde = r.
    let mut v_tilde = r.clone();
    let mut w_tilde = r.clone();

    let mut rho = dot(&v_tilde, &v_tilde, n).sqrt();
    let mut xi = dot(&w_tilde, &w_tilde, n).sqrt();

    // QMR recurrence scalars.
    let mut gamma = 1.0f64;
    let mut eta = -1.0f64;
    let mut epsilon = 1.0f64;
    let mut theta;
    let mut theta_old = 0.0f64;

    // Lanczos vectors (current/previous) and search directions.
    let mut v = vec![0.0f64; n];
    let mut w = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];
    let mut d = vec![0.0f64; n];
    let mut s = vec![0.0f64; n];

    for itn in 1..=config.max_iter {
        if rho < TINY || xi < TINY {
            return Err(SolverError::InternalError(
                "qmr: Lanczos breakdown (rho or xi ~ 0)".into(),
            ));
        }

        // Normalise Lanczos vectors.
        let inv_rho = 1.0 / rho;
        let inv_xi = 1.0 / xi;
        for i in 0..n {
            v[i] = v_tilde[i] * inv_rho;
            w[i] = w_tilde[i] * inv_xi;
        }

        let delta = dot(&w, &v, n);
        if delta.abs() < TINY {
            return Err(SolverError::InternalError(
                "qmr: breakdown (w·v ~ 0); look-ahead required".into(),
            ));
        }

        // p, q update.
        if itn == 1 {
            p.copy_from_slice(&v);
            q.copy_from_slice(&w);
        } else {
            let pde = xi * delta / epsilon;
            let qde = rho * delta / epsilon;
            for i in 0..n {
                p[i] = v[i] - pde * p[i];
                q[i] = w[i] - qde * q[i];
            }
        }

        // A p  and  Aᵀ q
        let mut ap = vec![0.0f64; n];
        spmv(&p, &mut ap)?;
        epsilon = dot(&q, &ap, n);
        if epsilon.abs() < TINY {
            return Err(SolverError::InternalError(
                "qmr: breakdown (q·Ap ~ 0)".into(),
            ));
        }
        let beta = epsilon / delta;
        if beta.abs() < TINY {
            return Err(SolverError::InternalError(
                "qmr: breakdown (beta ~ 0)".into(),
            ));
        }

        // v_tilde = A p - beta * v ;  w_tilde = Aᵀ q - beta * w
        let mut atq = vec![0.0f64; n];
        spmv_t(&q, &mut atq)?;
        for i in 0..n {
            v_tilde[i] = ap[i] - beta * v[i];
            w_tilde[i] = atq[i] - beta * w[i];
        }

        let rho_new = dot(&v_tilde, &v_tilde, n).sqrt();
        let xi_new = dot(&w_tilde, &w_tilde, n).sqrt();

        // QMR least-squares update (rotation on the tridiagonal factor).
        theta = rho_new / (gamma * beta.abs());
        let gamma_new = 1.0 / (1.0 + theta * theta).sqrt();
        if gamma_new < TINY {
            return Err(SolverError::InternalError(
                "qmr: breakdown (gamma ~ 0)".into(),
            ));
        }
        let eta_new = -eta * rho * gamma_new * gamma_new / (beta * gamma * gamma);

        // d, s direction update and solution increment.
        if itn == 1 {
            for i in 0..n {
                d[i] = eta_new * p[i];
                s[i] = eta_new * ap[i];
            }
        } else {
            let scal = theta_old * theta_old * gamma_new * gamma_new;
            for i in 0..n {
                d[i] = eta_new * p[i] + scal * d[i];
                s[i] = eta_new * ap[i] + scal * s[i];
            }
        }

        for i in 0..n {
            x[i] += d[i];
            r[i] -= s[i];
        }

        // Advance scalars for the next iteration.
        gamma = gamma_new;
        eta = eta_new;
        theta_old = theta;
        rho = rho_new;
        xi = xi_new;

        // Convergence: use the (cheaply recurred) residual r.
        res_norm = dot(&r, &r, n).sqrt();
        if res_norm <= abs_tol {
            return Ok(itn);
        }
        if rho < TINY {
            // Krylov space exhausted; r should already be tiny.
            return Ok(itn);
        }
    }

    Err(SolverError::ConvergenceFailure {
        iterations: config.max_iter,
        residual: res_norm,
    })
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

/// Dot product of two length-`n` slices.
fn dot(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n {
        acc += a[i] * b[i];
    }
    acc
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spmv(a: Vec<Vec<f64>>) -> impl Fn(&[f64], &mut [f64]) -> SolverResult<()> {
        move |x: &[f64], out: &mut [f64]| {
            let n = a.len();
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += a[i][j] * x[j];
                }
                out[i] = acc;
            }
            Ok(())
        }
    }

    fn make_spmv_t(a: Vec<Vec<f64>>) -> impl Fn(&[f64], &mut [f64]) -> SolverResult<()> {
        // y = Aᵀ x  ->  y[j] = sum_i a[i][j] x[i]
        move |x: &[f64], out: &mut [f64]| {
            let n = a.len();
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += a[i][j] * x[i];
                }
                out[j] = acc;
            }
            Ok(())
        }
    }

    fn residual_norm(a: &[Vec<f64>], x: &[f64], b: &[f64]) -> f64 {
        let n = b.len();
        let mut s = 0.0;
        for i in 0..n {
            let mut ax = 0.0;
            for j in 0..n {
                ax += a[i][j] * x[j];
            }
            s += (ax - b[i]).powi(2);
        }
        s.sqrt()
    }

    #[test]
    fn config_default() {
        let c = QmrConfig::default();
        assert_eq!(c.max_iter, 1000);
        assert!((c.tol - 1e-8).abs() < 1e-18);
    }

    #[test]
    fn qmr_nonsymmetric_3x3() {
        // A non-symmetric, well-conditioned 3x3 system.
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![2.0, 0.0, 5.0],
        ];
        // x_exact = [1, 2, 3]  ->  b = A x_exact
        let x_exact = [1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        for i in 0..3 {
            let mut acc = 0.0;
            for j in 0..3 {
                acc += a[i][j] * x_exact[j];
            }
            b[i] = acc;
        }
        let mut x = vec![0.0; 3];
        let cfg = QmrConfig {
            max_iter: 200,
            tol: 1e-12,
        };
        let iters = qmr_f64(
            &make_spmv(a.clone()),
            &make_spmv_t(a.clone()),
            &b,
            &mut x,
            3,
            &cfg,
        )
        .expect("qmr converges");
        assert!(iters <= 10, "qmr 3x3 should be quick, got {iters}");
        for i in 0..3 {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-8,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-8);
    }

    #[test]
    fn qmr_diagonal_5x5() {
        let diag = [2.0, 4.0, 6.0, 8.0, 10.0];
        let mut a = vec![vec![0.0; 5]; 5];
        for i in 0..5 {
            a[i][i] = diag[i];
        }
        let b: Vec<f64> = diag.iter().map(|&d| d * 1.0).collect(); // x_exact = ones
        let mut x = vec![0.0; 5];
        let cfg = QmrConfig {
            max_iter: 100,
            tol: 1e-12,
        };
        let iters = qmr_f64(
            &make_spmv(a.clone()),
            &make_spmv_t(a.clone()),
            &b,
            &mut x,
            5,
            &cfg,
        )
        .expect("qmr converges");
        assert!(iters <= 6, "diag qmr should finish in <= n+1, got {iters}");
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-8, "x[{i}] = {xi}");
        }
    }

    #[test]
    fn qmr_nonsymmetric_convection_diffusion_8x8() {
        // 1D convection-diffusion: tridiag with asymmetric off-diagonals.
        let n = 8;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 2.0;
            if i > 0 {
                a[i][i - 1] = -1.0 - 0.3; // lower
            }
            if i + 1 < n {
                a[i][i + 1] = -1.0 + 0.3; // upper (asymmetric -> non-symmetric)
            }
        }
        let x_exact: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += a[i][j] * x_exact[j];
            }
            b[i] = acc;
        }
        let mut x = vec![0.0; n];
        let cfg = QmrConfig {
            max_iter: 300,
            tol: 1e-11,
        };
        let iters = qmr_f64(
            &make_spmv(a.clone()),
            &make_spmv_t(a.clone()),
            &b,
            &mut x,
            n,
            &cfg,
        )
        .expect("qmr converges");
        assert!(iters <= n as u32 + 4, "convdiff qmr near n it, got {iters}");
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-6,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-6);
    }

    #[test]
    fn qmr_zero_rhs_is_zero() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![0.0, 0.0];
        let mut x = vec![5.0, 7.0];
        let cfg = QmrConfig::default();
        let iters = qmr_f64(&make_spmv(a.clone()), &make_spmv_t(a), &b, &mut x, 2, &cfg).unwrap();
        assert_eq!(iters, 0);
        assert_eq!(x, vec![0.0, 0.0]);
    }
}
