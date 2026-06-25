//! LSQR — iterative least-squares solver for `min ‖A·x − b‖₂`.
//!
//! LSQR (Paige & Saunders 1982, "LSQR: An Algorithm for Sparse Linear Equations
//! and Sparse Least Squares", ACM TOMS 8(1), 43–71) solves, for a general
//! rectangular matrix `A ∈ ℝ^{m×n}`:
//!
//! * the **least-squares** problem `min_x ‖A·x − b‖₂` (overdetermined, `m ≥ n`),
//! * the **minimum-norm** solution of a consistent system (underdetermined),
//! * optionally a **damped** (Tikhonov-regularised) problem
//!   `min_x ‖A·x − b‖₂² + λ²·‖x‖₂²`.
//!
//! Analytically LSQR is equivalent to applying Conjugate Gradients to the normal
//! equations `(AᵀA)·x = Aᵀ·b`, but it works directly with the **Golub–Kahan
//! bidiagonalization** of `A`, which is far more numerically stable: it never
//! forms `AᵀA` (whose condition number is the *square* of `cond(A)`).
//!
//! The solver is matrix-free and rectangular-aware: it takes two closures,
//! `aprod` for `y = A·x` (length-`m` output) and `atprod` for `y = Aᵀ·x`
//! (length-`n` output).

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

/// Configuration for the [`lsqr_solve`] solver.
#[derive(Debug, Clone)]
pub struct LsqrConfig {
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Stopping tolerance on the relative residual / normal-equation residual.
    ///
    /// LSQR stops when `‖Aᵀ·r‖ / (‖A‖·‖r‖) ≤ atol` (least-squares solution found)
    /// or `‖r‖ / ‖b‖ ≤ btol` (compatible system solved). Both use this value.
    pub tol: f64,
    /// Tikhonov damping parameter `λ` (set `0.0` for the undamped problem).
    pub damp: f64,
}

impl Default for LsqrConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            damp: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves `min ‖A·x − b‖₂` (optionally damped) using LSQR.
///
/// * `aprod`  — closure computing `y = A · x`  (input length `n`, output length `m`).
/// * `atprod` — closure computing `y = Aᵀ · x` (input length `m`, output length `n`).
/// * `b`      — right-hand side, length `m`.
/// * `x`      — solution, length `n`; **overwritten** (initial contents ignored —
///   LSQR starts from `x = 0`, as in the reference implementation).
/// * `m`, `n` — matrix row / column counts.
///
/// Returns the number of iterations performed.
///
/// # Errors
///
/// * [`SolverError::DimensionMismatch`] if `b.len() < m` or `x.len() < n`.
/// * [`SolverError::ConvergenceFailure`] if neither stopping criterion is met
///   within `config.max_iter` iterations.
// Both `m` and `n` are intrinsic to a rectangular least-squares solve, and the
// two matrix-free closures plus the handle round out the (irreducible) 8-arg API.
#[allow(clippy::too_many_arguments)]
pub fn lsqr_solve<T, FA, FT>(
    _handle: &SolverHandle,
    aprod: FA,
    atprod: FT,
    b: &[T],
    x: &mut [T],
    m: u32,
    n: u32,
    config: &LsqrConfig,
) -> SolverResult<u32>
where
    T: GpuFloat,
    FA: Fn(&[T], &mut [T]) -> SolverResult<()>,
    FT: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let m_usize = m as usize;
    let n_usize = n as usize;
    if b.len() < m_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "lsqr_solve: b length ({}) < m ({m})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "lsqr_solve: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if m == 0 || n == 0 {
        for xi in x.iter_mut().take(n_usize) {
            *xi = T::gpu_zero();
        }
        return Ok(0);
    }

    let b64: Vec<f64> = (0..m_usize).map(|i| to_f64(b[i])).collect();
    let mut x64 = vec![0.0f64; n_usize];

    let aprod64 = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
        let vt: Vec<T> = v.iter().map(|&c| from_f64::<T>(c)).collect();
        let mut ot = vec![T::gpu_zero(); m_usize];
        aprod(&vt, &mut ot)?;
        for (o, ot_i) in out.iter_mut().zip(ot.iter()) {
            *o = to_f64(*ot_i);
        }
        Ok(())
    };
    let atprod64 = |v: &[f64], out: &mut [f64]| -> SolverResult<()> {
        let vt: Vec<T> = v.iter().map(|&c| from_f64::<T>(c)).collect();
        let mut ot = vec![T::gpu_zero(); n_usize];
        atprod(&vt, &mut ot)?;
        for (o, ot_i) in out.iter_mut().zip(ot.iter()) {
            *o = to_f64(*ot_i);
        }
        Ok(())
    };

    let iters = lsqr_f64(
        &aprod64, &atprod64, &b64, &mut x64, m_usize, n_usize, config,
    )?;

    for (xi, &v) in x.iter_mut().zip(x64.iter()) {
        *xi = from_f64::<T>(v);
    }
    Ok(iters)
}

/// Core LSQR iteration in `f64` (Paige–Saunders 1982).
///
/// Variable names follow the published algorithm: the Golub–Kahan scalars
/// `alpha`, `beta`, the QR rotation `(cs, sn)` and the recurred quantities
/// `rho`, `phi`, `phibar`, `rhobar`, `theta`. Damping `λ` is folded into `rho`
/// via the extra rotation described in §5 of the paper.
fn lsqr_f64<FA, FT>(
    aprod: &FA,
    atprod: &FT,
    b: &[f64],
    x: &mut [f64],
    m: usize,
    n: usize,
    config: &LsqrConfig,
) -> SolverResult<u32>
where
    FA: Fn(&[f64], &mut [f64]) -> SolverResult<()>,
    FT: Fn(&[f64], &mut [f64]) -> SolverResult<()>,
{
    const TINY: f64 = 1e-300;
    let damp = config.damp;

    // ---- Initialise the bidiagonalization (Bidiag(1)). ----
    // beta * u = b   (u length m)
    let mut u = b.to_vec();
    let mut beta = nrm2(&u);
    let b_norm = beta;
    if b_norm == 0.0 {
        for xi in x.iter_mut().take(n) {
            *xi = 0.0;
        }
        return Ok(0);
    }
    for ui in u.iter_mut() {
        *ui /= beta;
    }

    // alpha * v = Aᵀ u   (v length n)
    let mut v = vec![0.0f64; n];
    atprod(&u, &mut v)?;
    let mut alpha = nrm2(&v);
    if alpha > TINY {
        for vi in v.iter_mut() {
            *vi /= alpha;
        }
    }

    let mut w = v.clone();
    for xi in x.iter_mut().take(n) {
        *xi = 0.0;
    }

    // ---- Initialise the QR factorization of the lower-bidiagonal matrix. ----
    let mut rhobar = alpha;
    let mut phibar = beta;

    // Running estimate of ‖A‖_F, used in the least-squares stopping test.
    let mut anorm = 0.0f64;
    let mut au = vec![0.0f64; m];

    if alpha <= TINY {
        // Aᵀ b = 0  ->  x = 0 is already the least-squares solution.
        return Ok(0);
    }

    for itn in 1..=config.max_iter {
        // ---- Continue the bidiagonalization. ----
        // beta * u = A v - alpha * u
        aprod(&v, &mut au)?;
        for i in 0..m {
            u[i] = au[i] - alpha * u[i];
        }
        beta = nrm2(&u);
        if beta > TINY {
            for ui in u.iter_mut() {
                *ui /= beta;
            }
        }

        // Accumulate an estimate of ‖A‖_F from the bidiagonal entries.
        anorm = (anorm * anorm + alpha * alpha + beta * beta + damp * damp).sqrt();

        // alpha * v = Aᵀ u - beta * v
        if beta > TINY {
            let mut atu = vec![0.0f64; n];
            atprod(&u, &mut atu)?;
            for i in 0..n {
                v[i] = atu[i] - beta * v[i];
            }
            alpha = nrm2(&v);
            if alpha > TINY {
                for vi in v.iter_mut() {
                    *vi /= alpha;
                }
            }
        }

        // ---- Apply Tikhonov damping (extra plane rotation on rhobar). ----
        // Paige–Saunders §5: rotate (rhobar, damp) so the damped least-squares
        // problem is reduced to the same recurrence. The `sn1 * phibar` part
        // (psi) feeds the damping residual and is not needed for the solution.
        if damp > 0.0 {
            let rhobar_damp = (rhobar * rhobar + damp * damp).sqrt();
            let cs1 = rhobar / rhobar_damp;
            rhobar = rhobar_damp;
            phibar *= cs1;
        }

        // ---- Construct and apply the next plane rotation Q_k. ----
        let rho = (rhobar * rhobar + beta * beta).sqrt().max(TINY);
        let cs = rhobar / rho;
        let sn = beta / rho;
        let theta = sn * alpha;
        let rhobar_next = -cs * alpha;
        let phi = cs * phibar;
        let phibar_next = sn * phibar;

        // ---- Update x and the search direction w. ----
        let t1 = phi / rho;
        let t2 = -theta / rho;
        for i in 0..n {
            x[i] += t1 * w[i];
            w[i] = v[i] + t2 * w[i];
        }

        rhobar = rhobar_next;
        phibar = phibar_next;

        // ---- Stopping criteria (Paige–Saunders, simplified but faithful). ----
        // ‖r‖ ≈ |phibar|.   ‖Aᵀ r‖ ≈ |phibar * alpha * cs|.
        let r_norm = phibar.abs();
        let arnorm = (phibar * alpha * cs).abs();

        let test1 = r_norm / b_norm; // compatible-system residual ratio
        let test2 = if anorm * r_norm > TINY {
            arnorm / (anorm * r_norm) // least-squares (normal-eqn) residual ratio
        } else {
            0.0
        };

        if test1 <= config.tol || test2 <= config.tol {
            return Ok(itn);
        }
    }

    Err(SolverError::ConvergenceFailure {
        iterations: config.max_iter,
        residual: phibar,
    })
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

/// Euclidean 2-norm of a slice.
fn nrm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_aprod(a: Vec<Vec<f64>>) -> impl Fn(&[f64], &mut [f64]) -> SolverResult<()> {
        // y = A x  (m x n) * (n) = (m)
        move |x: &[f64], out: &mut [f64]| {
            let m = a.len();
            let n = if m > 0 { a[0].len() } else { 0 };
            for i in 0..m {
                let mut acc = 0.0;
                for j in 0..n {
                    acc += a[i][j] * x[j];
                }
                out[i] = acc;
            }
            Ok(())
        }
    }

    fn make_atprod(a: Vec<Vec<f64>>) -> impl Fn(&[f64], &mut [f64]) -> SolverResult<()> {
        // y = Aᵀ x  (n x m) * (m) = (n)
        move |x: &[f64], out: &mut [f64]| {
            let m = a.len();
            let n = if m > 0 { a[0].len() } else { 0 };
            for j in 0..n {
                let mut acc = 0.0;
                for i in 0..m {
                    acc += a[i][j] * x[i];
                }
                out[j] = acc;
            }
            Ok(())
        }
    }

    #[test]
    fn config_default() {
        let c = LsqrConfig::default();
        assert_eq!(c.max_iter, 1000);
        assert!((c.tol - 1e-8).abs() < 1e-18);
        assert_eq!(c.damp, 0.0);
    }

    #[test]
    fn lsqr_square_consistent_system() {
        // Square, non-symmetric, consistent: solve A x = b exactly.
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![2.0, 0.0, 5.0],
        ];
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
        let cfg = LsqrConfig {
            max_iter: 100,
            tol: 1e-12,
            damp: 0.0,
        };
        let iters = lsqr_f64(
            &make_aprod(a.clone()),
            &make_atprod(a.clone()),
            &b,
            &mut x,
            3,
            3,
            &cfg,
        )
        .expect("lsqr converges");
        assert!(
            iters <= 10,
            "square consistent should be quick, got {iters}"
        );
        for i in 0..3 {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-7,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
    }

    #[test]
    fn lsqr_overdetermined_least_squares() {
        // Overdetermined 4x2 system. Fit y = c0 + c1*t to data exactly on a line.
        // Rows: [1, t_i]; data t = 0,1,2,3, true [c0,c1] = [2, 3] -> y = 2,5,8,11.
        let a = vec![
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
        ];
        let b = vec![2.0, 5.0, 8.0, 11.0]; // exactly on the line
        let mut x = vec![0.0; 2];
        let cfg = LsqrConfig {
            max_iter: 100,
            tol: 1e-12,
            damp: 0.0,
        };
        let iters = lsqr_f64(
            &make_aprod(a.clone()),
            &make_atprod(a.clone()),
            &b,
            &mut x,
            4,
            2,
            &cfg,
        )
        .expect("lsqr converges");
        assert!(iters <= 10, "least squares should be quick, got {iters}");
        assert!((x[0] - 2.0).abs() < 1e-6, "intercept = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-6, "slope = {}", x[1]);
    }

    #[test]
    fn lsqr_overdetermined_normal_equation_residual() {
        // Inconsistent overdetermined system: LSQR must return the least-squares
        // solution, i.e. one where Aᵀ(A x - b) ≈ 0.
        let a = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
        ];
        let b = vec![6.0, 5.0, 7.0, 10.0]; // noisy
        let mut x = vec![0.0; 2];
        let cfg = LsqrConfig {
            max_iter: 200,
            tol: 1e-13,
            damp: 0.0,
        };
        lsqr_f64(
            &make_aprod(a.clone()),
            &make_atprod(a.clone()),
            &b,
            &mut x,
            4,
            2,
            &cfg,
        )
        .expect("lsqr converges");
        // Compute Aᵀ (A x - b) and check it is ~0 (normal equations satisfied).
        let m = 4;
        let n = 2;
        let mut ax = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                ax[i] += a[i][j] * x[j];
            }
        }
        let r: Vec<f64> = (0..m).map(|i| ax[i] - b[i]).collect();
        let mut atr = vec![0.0; n];
        for j in 0..n {
            for i in 0..m {
                atr[j] += a[i][j] * r[i];
            }
        }
        let atr_norm = nrm2(&atr);
        assert!(
            atr_norm < 1e-6,
            "normal-equation residual ‖Aᵀr‖ = {atr_norm}"
        );
    }

    #[test]
    fn lsqr_damped_regularization_shrinks_solution() {
        // With strong damping, ‖x‖ should be smaller than the undamped solution.
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let b = vec![3.0, 4.0, 6.0];

        let mut x0 = vec![0.0; 2];
        let cfg0 = LsqrConfig {
            max_iter: 200,
            tol: 1e-12,
            damp: 0.0,
        };
        lsqr_f64(
            &make_aprod(a.clone()),
            &make_atprod(a.clone()),
            &b,
            &mut x0,
            3,
            2,
            &cfg0,
        )
        .expect("undamped converges");

        let mut x1 = vec![0.0; 2];
        let cfg1 = LsqrConfig {
            max_iter: 200,
            tol: 1e-12,
            damp: 5.0,
        };
        lsqr_f64(
            &make_aprod(a.clone()),
            &make_atprod(a.clone()),
            &b,
            &mut x1,
            3,
            2,
            &cfg1,
        )
        .expect("damped converges");

        let nrm0 = nrm2(&x0);
        let nrm1 = nrm2(&x1);
        assert!(
            nrm1 < nrm0,
            "damping should shrink solution: ‖x_damped‖={nrm1} should be < ‖x‖={nrm0}"
        );
    }

    #[test]
    fn lsqr_zero_rhs_is_zero() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![0.0, 0.0];
        let mut x = vec![5.0, 7.0];
        let cfg = LsqrConfig::default();
        let iters = lsqr_f64(
            &make_aprod(a.clone()),
            &make_atprod(a),
            &b,
            &mut x,
            2,
            2,
            &cfg,
        )
        .unwrap();
        assert_eq!(iters, 0);
        assert_eq!(x, vec![0.0, 0.0]);
    }
}
