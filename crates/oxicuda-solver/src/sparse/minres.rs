//! MINRES (Minimal Residual) iterative solver.
//!
//! Solves the linear system `A * x = b` where `A` is **symmetric** (it may be
//! indefinite, unlike Conjugate Gradient which requires positive definiteness).
//! MINRES minimizes the residual 2-norm `‖b − A·xₖ‖₂` over the Krylov subspace
//! `Kₖ(A, b) = span{b, A·b, …, Aᵏ⁻¹·b}` and is therefore the method of choice for
//! symmetric but possibly indefinite systems (saddle-point problems, shifted
//! Helmholtz operators, etc.).
//!
//! The solver is matrix-free: it only requires a closure that computes the
//! matrix-vector product `y = A · x`.
//!
//! # Algorithm
//!
//! MINRES is built on the **Lanczos** three-term recurrence, which generates an
//! orthonormal basis `v₁, v₂, …` of the Krylov subspace together with a
//! symmetric tridiagonal matrix `Tₖ`. The least-squares problem
//! `min_y ‖β₁·e₁ − T̄ₖ·y‖₂` is solved incrementally by applying plane (Givens)
//! rotations to `T̄ₖ`, which keeps the cost per iteration at a handful of axpy
//! operations and three vectors of storage. The implementation follows the
//! classic formulation of Paige & Saunders (1975), "Solution of Sparse
//! Indefinite Systems of Linear Equations", SIAM J. Numer. Anal. 12(4), 617–629,
//! using the variable names of the reference Fortran/`minres.m`.
//!
//! Unlike GMRES, MINRES does **not** store the full Krylov basis, so its memory
//! footprint is constant (`O(n)`) regardless of the iteration count — this is the
//! exploit that symmetry buys us.

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

/// Configuration for the [`minres_solve`] solver.
#[derive(Debug, Clone)]
pub struct MinresConfig {
    /// Maximum number of iterations.
    pub max_iter: u32,
    /// Convergence tolerance (relative to `‖b‖₂`).
    pub tol: f64,
}

impl Default for MinresConfig {
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

/// Solves `A · x = b` for a **symmetric** (possibly indefinite) matrix `A`
/// using MINRES.
///
/// The matrix is supplied through the matrix-free closure `spmv`, which must
/// compute `y = A · x`. `A` is assumed symmetric; the algorithm is undefined for
/// non-symmetric operators (use BiCGSTAB / GMRES / [`crate::sparse::qmr`] there).
///
/// On entry `x` holds the initial guess (typically zeros); on exit it holds the
/// approximate solution. Returns the number of iterations performed.
///
/// # Errors
///
/// * [`SolverError::DimensionMismatch`] if `b` or `x` are shorter than `n`.
/// * [`SolverError::ConvergenceFailure`] if the residual tolerance is not reached
///   within `config.max_iter` iterations.
pub fn minres_solve<T, F>(
    _handle: &SolverHandle,
    spmv: F,
    b: &[T],
    x: &mut [T],
    n: u32,
    config: &MinresConfig,
) -> SolverResult<u32>
where
    T: GpuFloat,
    F: Fn(&[T], &mut [T]) -> SolverResult<()>,
{
    let n_usize = n as usize;
    if b.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "minres_solve: b length ({}) < n ({n})",
            b.len()
        )));
    }
    if x.len() < n_usize {
        return Err(SolverError::DimensionMismatch(format!(
            "minres_solve: x length ({}) < n ({n})",
            x.len()
        )));
    }
    if n == 0 {
        return Ok(0);
    }

    // Run the algorithm in f64 for numerical robustness, then cast the converged
    // solution back into the requested precision.
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

    let iters = minres_f64(&spmv64, &b64, &mut x64, n_usize, config)?;

    for (xi, &v) in x.iter_mut().zip(x64.iter()) {
        *xi = from_f64::<T>(v);
    }
    Ok(iters)
}

/// Core MINRES iteration in `f64` (Paige & Saunders 1975).
///
/// This is a faithful transcription of the reference `minres` algorithm. The
/// scalar variable names (`alfa`, `beta`, `cs`, `sn`, `gamma`, `delta`, `epsln`,
/// `dbar`, `phibar`, …) match the published pseudocode so the recurrence can be
/// checked line-by-line against the source.
fn minres_f64<F>(
    spmv: &F,
    b: &[f64],
    x: &mut [f64],
    n: usize,
    config: &MinresConfig,
) -> SolverResult<u32>
where
    F: Fn(&[f64], &mut [f64]) -> SolverResult<()>,
{
    let b_norm = dot(b, b, n).sqrt();
    if b_norm == 0.0 {
        for xi in x.iter_mut().take(n) {
            *xi = 0.0;
        }
        return Ok(0);
    }
    let abs_tol = config.tol * b_norm;

    // y = r1 = b - A x0  (initial residual; unnormalised first Lanczos vector).
    let mut ax = vec![0.0f64; n];
    spmv(x, &mut ax)?;
    let mut r1: Vec<f64> = (0..n).map(|i| b[i] - ax[i]).collect();
    let mut y: Vec<f64> = r1.clone();

    let beta1 = dot(&r1, &y, n).sqrt(); // = ‖r1‖ (no preconditioner).
    if beta1 < abs_tol {
        return Ok(0);
    }

    // Initialise Lanczos / Givens recurrence state.
    let mut old_beta = 0.0f64;
    let mut beta = beta1;

    // `epsln` is produced at iteration k and consumed (as `oldeps`) at k+1; it
    // multiplies the direction vector from two steps back in the x-update.
    let mut epsln = 0.0f64;
    let mut cs = -1.0f64;
    let mut sn = 0.0f64;
    let mut dbar = 0.0f64;
    let mut phibar = beta1;

    let mut r2 = r1.clone();

    // Direction vectors (current and two previous).
    let mut w = vec![0.0f64; n];
    let mut w1 = vec![0.0f64; n];
    let mut w2;

    for itn in 1..=config.max_iter {
        // ---- Lanczos step (generates v = next basis vector, alfa, beta). ----
        let s = 1.0 / beta;
        let v: Vec<f64> = y.iter().map(|&yi| s * yi).collect();

        // y = A v
        spmv(&v, &mut y)?;

        if itn >= 2 {
            let scal = beta / old_beta;
            for i in 0..n {
                y[i] -= scal * r1[i];
            }
        }

        let alfa = dot(&v, &y, n);
        let scal2 = alfa / beta;
        for i in 0..n {
            y[i] -= scal2 * r2[i];
        }

        // Shift residual vectors: r1 <- r2, r2 <- y.
        r1.copy_from_slice(&r2);
        r2.copy_from_slice(&y);

        old_beta = beta;
        beta = dot(&r2, &y, n).sqrt();

        // ---- Apply previous rotation Q_{k-1} to the new column. ----
        // `oldeps` is the epsilon computed in the previous iteration; it scales
        // the direction vector two steps back (w2).
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa; // delta_k
        let gbar = sn * dbar - cs * alfa; // gbar_k
        epsln = sn * beta; // epsln_{k+1} (used next iteration as oldeps)
        dbar = -cs * beta; // dbar_{k+1}

        // ---- New plane rotation Q_k to annihilate beta_{k+1} from gbar. ----
        let gamma = (gbar * gbar + beta * beta).sqrt().max(1e-300);
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar; // last term of the solution increment
        phibar *= sn; // residual norm estimate ‖r_k‖

        // ---- Update of x via the direction vectors. ----
        // w2 <- w1, w1 <- w, w <- (v - oldeps*w2 - delta*w1) / gamma
        w2 = w1;
        w1 = w.clone();
        let denom = 1.0 / gamma;
        for i in 0..n {
            w[i] = (v[i] - oldeps * w2[i] - delta * w1[i]) * denom;
            x[i] += phi * w[i];
        }

        // ---- Convergence test on the recurred residual estimate. ----
        if phibar <= abs_tol {
            return Ok(itn);
        }
        // Lanczos breakdown: beta ~ 0 means the Krylov space is exhausted and
        // the current x already minimises the residual.
        if beta <= 1e-300 {
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

/// Dot product of two length-`n` slices.
fn dot(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn config_default() {
        let c = MinresConfig::default();
        assert_eq!(c.max_iter, 1000);
        assert!((c.tol - 1e-8).abs() < 1e-18);
    }

    #[test]
    fn minres_spd_2x2_matches_exact() {
        // A = [[4, 1], [1, 3]] (SPD), b = [1, 2]; exact x = [1/11, 7/11].
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![1.0, 2.0];
        let mut x = vec![0.0; 2];
        let cfg = MinresConfig {
            max_iter: 100,
            tol: 1e-12,
        };
        let iters = minres_f64(&make_spmv(a.clone()), &b, &mut x, 2, &cfg).unwrap();
        assert!(
            iters <= 2,
            "MINRES on 2x2 SPD should finish in <= 2 it, got {iters}"
        );
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-9, "x0 = {}", x[0]);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-9, "x1 = {}", x[1]);
        assert!(residual_norm(&a, &x, &b) < 1e-9);
    }

    #[test]
    fn minres_symmetric_indefinite_converges() {
        // A symmetric INDEFINITE matrix (eigenvalues of opposite sign): CG would
        // break down here, but MINRES must converge.
        // A = [[0, 1], [1, 0]]  -> eigenvalues +1, -1.  b = [1, 1] -> x = [1, 1].
        let a = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let b = vec![1.0, 1.0];
        let mut x = vec![0.0; 2];
        let cfg = MinresConfig {
            max_iter: 100,
            tol: 1e-12,
        };
        let iters = minres_f64(&make_spmv(a.clone()), &b, &mut x, 2, &cfg).unwrap();
        assert!(
            iters <= 2,
            "indefinite 2x2 should finish quickly, got {iters}"
        );
        assert!((x[0] - 1.0).abs() < 1e-9, "x0 = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-9, "x1 = {}", x[1]);
    }

    #[test]
    fn minres_indefinite_diagonal_5x5() {
        // D = diag(1, -2, 3, -4, 5), b = D * [1,1,1,1,1] = [1,-2,3,-4,5].
        let diag = [1.0, -2.0, 3.0, -4.0, 5.0];
        let mut a = vec![vec![0.0; 5]; 5];
        for i in 0..5 {
            a[i][i] = diag[i];
        }
        let b: Vec<f64> = diag.to_vec(); // x_exact = ones
        let mut x = vec![0.0; 5];
        let cfg = MinresConfig {
            max_iter: 100,
            tol: 1e-12,
        };
        let iters = minres_f64(&make_spmv(a.clone()), &b, &mut x, 5, &cfg).unwrap();
        assert!(iters <= 5, "diag 5x5 should finish in <= n it, got {iters}");
        for (i, &xi) in x.iter().enumerate() {
            assert!((xi - 1.0).abs() < 1e-8, "x[{i}] = {xi}");
        }
        assert!(residual_norm(&a, &x, &b) < 1e-8);
    }

    #[test]
    fn minres_spd_tridiagonal_10x10() {
        // 1D Laplacian: tridiag(-1, 2, -1) — SPD. Solve A x = b for random-ish b.
        let n = 10;
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 2.0;
            if i > 0 {
                a[i][i - 1] = -1.0;
            }
            if i + 1 < n {
                a[i][i + 1] = -1.0;
            }
        }
        // Pick x_exact, form b = A x_exact, then recover.
        let x_exact: Vec<f64> = (0..n).map(|i| ((i as f64) - 4.5).sin()).collect();
        let mut b = vec![0.0; n];
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += a[i][j] * x_exact[j];
            }
            b[i] = acc;
        }
        let mut x = vec![0.0; n];
        let cfg = MinresConfig {
            max_iter: 200,
            tol: 1e-11,
        };
        let iters = minres_f64(&make_spmv(a.clone()), &b, &mut x, n, &cfg).unwrap();
        assert!(
            iters <= n as u32 + 2,
            "tridiag should finish near n it, got {iters}"
        );
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-7,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-7);
    }

    #[test]
    fn minres_zero_rhs_is_zero() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let b = vec![0.0, 0.0];
        let mut x = vec![5.0, 7.0];
        let cfg = MinresConfig::default();
        let iters = minres_f64(&make_spmv(a), &b, &mut x, 2, &cfg).unwrap();
        assert_eq!(iters, 0);
        assert_eq!(x, vec![0.0, 0.0]);
    }
}
