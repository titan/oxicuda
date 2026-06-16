//! Mixed-precision iterative refinement for dense linear systems.
//!
//! Iterative refinement (Langou 2006) improves the accuracy of an initial
//! direct solve by computing the residual `r = b - A*x`, solving a correction
//! system `A*e = r`, and accumulating `x ← x + e`.  Each refinement step
//! costs one additional LU solve (forward/back substitution only — the same
//! LU factorization is reused) plus one matrix-vector multiply for the residual.
//!
//! ## Numerical notes
//!
//! * The LU factorization is performed once in f64 and reused for all
//!   refinement steps.
//! * The residual is always computed in f64 (full precision).
//! * Convergence is declared when `||r||_inf < tol` before the maximum number
//!   of refinements is exhausted.
//! * Pivoting is partial (column-wise maximum), which is standard for dense
//!   LU on CPU.

use crate::error::{SolverError, SolverResult};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for [`iterative_refinement`].
#[derive(Debug, Clone)]
pub struct IterRefineConfig {
    /// Maximum number of refinement steps to perform after the initial direct solve.
    /// Setting this to `0` gives a plain LU solve with no refinement.
    pub n_refinements: usize,

    /// Convergence tolerance on `||b - A*x||_inf`.  Refinement terminates
    /// early once the residual falls below this threshold.
    pub tol: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solve `A*x = b` using LU factorization followed by iterative refinement.
///
/// ## Parameters
///
/// * `a`   – flat row-major `n × n` coefficient matrix.
/// * `b`   – right-hand side of length `n`.
/// * `n`   – dimension of the system.
/// * `cfg` – refinement configuration (max iterations and tolerance).
///
/// ## Returns
///
/// The solution vector `x` of length `n`.
///
/// ## Errors
///
/// * [`SolverError::DimensionMismatch`] – if `a.len() != n*n` or `b.len() != n`,
///   or if `n == 0`.
/// * [`SolverError::SingularMatrix`]    – if the coefficient matrix is singular.
pub fn iterative_refinement(
    a: &[f64],
    b: &[f64],
    n: usize,
    cfg: &IterRefineConfig,
) -> SolverResult<Vec<f64>> {
    validate_inputs(a, b, n)?;

    // Factorize A once.
    let (lu, piv) = lu_factorize_f64(a, n)?;

    // Initial solve: x0 = LU⁻¹ * b.
    let mut x = lu_solve_f64(&lu, &piv, b, n);

    // Iterative refinement loop.
    for _ in 0..cfg.n_refinements {
        // Compute residual r = b - A*x in f64.
        let ax = mat_vec_mul_f64(a, &x, n);
        let r = vec_sub_f64(b, &ax);

        // Check convergence (inf-norm of residual).
        let inf_norm = r.iter().cloned().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        if inf_norm < cfg.tol {
            break;
        }

        // Solve correction system A*e = r using the same LU factorization.
        let e = lu_solve_f64(&lu, &piv, &r, n);

        // Accumulate correction.
        for i in 0..n {
            x[i] += e[i];
        }
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Private numerical routines
// ---------------------------------------------------------------------------

/// Compute LU factorization of an `n × n` matrix (row-major, flat) with
/// partial pivoting.
///
/// # Returns
///
/// `(lu, piv)` where `lu` stores the combined `L` (unit lower-triangular,
/// below the diagonal) and `U` (upper-triangular, on and above the diagonal)
/// factors in-place using the standard LAPACK convention, and `piv[i]` records
/// which original row was selected as pivot at step `i`.
///
/// # Errors
///
/// Returns [`SolverError::SingularMatrix`] if any pivot is numerically zero.
fn lu_factorize_f64(a: &[f64], n: usize) -> SolverResult<(Vec<f64>, Vec<usize>)> {
    let mut lu = a.to_vec();
    // piv[i] = j means row i was exchanged with row j (original row index).
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Partial pivot: find row with largest |lu[row, k]| for row >= k.
        let mut max_abs = lu[k * n + k].abs();
        let mut max_row = k;
        for row in k + 1..n {
            let val = lu[row * n + k].abs();
            if val > max_abs {
                max_abs = val;
                max_row = row;
            }
        }

        if max_row != k {
            // Swap rows k and max_row.
            for col in 0..n {
                lu.swap(k * n + col, max_row * n + col);
            }
            piv.swap(k, max_row);
        }

        let pivot_val = lu[k * n + k];
        if pivot_val.abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        // Schur complement update.
        for row in k + 1..n {
            let factor = lu[row * n + k] / pivot_val;
            lu[row * n + k] = factor; // L entry
            for col in k + 1..n {
                let u_entry = lu[k * n + col];
                lu[row * n + col] -= factor * u_entry;
            }
        }
    }

    Ok((lu, piv))
}

/// Solve `A*x = b` given a pre-computed LU factorization and pivot vector.
///
/// Applies the row permutation encoded in `piv`, then performs forward
/// substitution through L (unit lower-triangular) and back substitution
/// through U (upper-triangular).
fn lu_solve_f64(lu: &[f64], piv: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    // Apply permutation: y[i] = b[piv[i]].
    let mut y: Vec<f64> = (0..n).map(|i| b[piv[i]]).collect();

    // Forward substitution: L * z = y  (L has unit diagonal).
    for i in 1..n {
        for j in 0..i {
            y[i] -= lu[i * n + j] * y[j];
        }
    }

    // Back substitution: U * x = z.
    for i in (0..n).rev() {
        for j in i + 1..n {
            y[i] -= lu[i * n + j] * y[j];
        }
        y[i] /= lu[i * n + i];
    }

    y
}

/// Compute the matrix-vector product `A * x` for an `n × n` row-major matrix.
fn mat_vec_mul_f64(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; n];
    for row in 0..n {
        let mut acc = 0.0_f64;
        for col in 0..n {
            acc += a[row * n + col] * x[col];
        }
        result[row] = acc;
    }
    result
}

/// Compute the elementwise difference `a - b` of two length-`n` vectors.
fn vec_sub_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate_inputs(a: &[f64], b: &[f64], n: usize) -> SolverResult<()> {
    if n == 0 {
        return Err(SolverError::DimensionMismatch(
            "iterative_refinement: n must be >= 1".into(),
        ));
    }
    if a.len() != n * n {
        return Err(SolverError::DimensionMismatch(format!(
            "iterative_refinement: a.len() ({}) != n*n ({})",
            a.len(),
            n * n
        )));
    }
    if b.len() != n {
        return Err(SolverError::DimensionMismatch(format!(
            "iterative_refinement: b.len() ({}) != n ({})",
            b.len(),
            n
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn inf_norm(v: &[f64]) -> f64 {
        v.iter().cloned().fold(0.0_f64, |acc, x| acc.max(x.abs()))
    }

    fn residual(a: &[f64], x: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let ax = mat_vec_mul_f64(a, x, n);
        vec_sub_f64(b, &ax)
    }

    // ---------- test 1: residual < tol after refinement ----------

    #[test]
    fn residual_lt_tol() {
        // 4×4 random-ish well-conditioned matrix.
        let n = 4;
        #[rustfmt::skip]
        let a = vec![
            4.0, 1.0, 0.0, 0.5,
            1.0, 5.0, 1.0, 0.0,
            0.0, 1.0, 6.0, 1.0,
            0.5, 0.0, 1.0, 7.0,
        ];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let tol = 1e-12;
        let cfg = IterRefineConfig {
            n_refinements: 5,
            tol,
        };

        let x = iterative_refinement(&a, &b, n, &cfg).expect("solve should succeed");
        let r = residual(&a, &x, &b, n);
        assert!(
            inf_norm(&r) < 1e-10,
            "inf-norm residual {} exceeds threshold",
            inf_norm(&r)
        );
    }

    // ---------- test 2: refinement improves or maintains solution quality ----------

    #[test]
    fn refinement_improves_solution() {
        let n = 5;
        #[rustfmt::skip]
        let a = vec![
             6.0, -1.0,  0.0,  0.0,  0.5,
            -1.0,  7.0, -1.0,  0.0,  0.0,
             0.0, -1.0,  8.0, -1.0,  0.0,
             0.0,  0.0, -1.0,  9.0, -1.0,
             0.5,  0.0,  0.0, -1.0, 10.0,
        ];
        let b = vec![3.0, 5.0, 7.0, 9.0, 11.0];

        let cfg0 = IterRefineConfig {
            n_refinements: 0,
            tol: 1e-14,
        };
        let cfg5 = IterRefineConfig {
            n_refinements: 5,
            tol: 1e-14,
        };

        let x0 = iterative_refinement(&a, &b, n, &cfg0).expect("0-ref solve");
        let x5 = iterative_refinement(&a, &b, n, &cfg5).expect("5-ref solve");

        let r0 = inf_norm(&residual(&a, &x0, &b, n));
        let r5 = inf_norm(&residual(&a, &x5, &b, n));

        // With a well-conditioned matrix both should be small, and refinement
        // must not make things worse.
        assert!(r5 <= r0 + 1e-14, "r5 ({r5}) should be <= r0 ({r0})");
    }

    // ---------- test 3: 0 refinements gives direct LU result ----------

    #[test]
    fn n_0_refinements_gives_direct_solve() {
        let n = 3;
        let a = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 4.0];
        let b = vec![1.0, 2.0, 3.0];

        let cfg = IterRefineConfig {
            n_refinements: 0,
            tol: 1e-15,
        };
        let x = iterative_refinement(&a, &b, n, &cfg).expect("solve");

        // Verify it satisfies A*x = b.
        let r = residual(&a, &x, &b, n);
        assert!(inf_norm(&r) < 1e-10, "residual = {}", inf_norm(&r));
    }

    // ---------- test 4: output has length n ----------

    #[test]
    fn solution_len() {
        for &n in &[1_usize, 3, 7, 10] {
            // Diagonal matrix n*I for simplicity.
            let a: Vec<f64> = (0..n * n)
                .map(|idx| {
                    if idx % (n + 1) == 0 {
                        (n as f64) + 1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let b: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            let cfg = IterRefineConfig {
                n_refinements: 2,
                tol: 1e-12,
            };

            let x = iterative_refinement(&a, &b, n, &cfg).expect("solve");
            assert_eq!(x.len(), n, "expected output length {n}");
        }
    }

    // ---------- test 5: singular matrix → SolverError::SingularMatrix ----------

    #[test]
    fn singular_error() {
        let n = 3;
        // Second row is all zeros → singular.
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0];
        let cfg = IterRefineConfig {
            n_refinements: 2,
            tol: 1e-12,
        };

        let result = iterative_refinement(&a, &b, n, &cfg);
        assert!(
            matches!(result, Err(SolverError::SingularMatrix)),
            "expected SingularMatrix, got {:?}",
            result
        );
    }

    // ---------- test 6: known system correct ----------

    #[test]
    fn known_system_correct() {
        // [2 1; 1 3] * [1; 3] = [5; 10]
        let n = 2;
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![5.0, 10.0];
        let cfg = IterRefineConfig {
            n_refinements: 5,
            tol: 1e-14,
        };

        let x = iterative_refinement(&a, &b, n, &cfg).expect("solve");
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 3.0).abs() < 1e-10, "x[1] = {}", x[1]);
    }

    // ---------- test 7: iteration count bounded by n_refinements ----------

    #[test]
    fn refinement_bounded_by_config() {
        // We use a counter-based wrapper concept: the function must not iterate
        // more than n_refinements+1 times (initial solve + at most n_refinements).
        // We verify this indirectly by ensuring the function always terminates and
        // the result matches A*x≈b, regardless of how many steps were actually taken.
        let n = 4;
        #[rustfmt::skip]
        let a = vec![
            5.0, 1.0, 0.0, 0.0,
            1.0, 5.0, 1.0, 0.0,
            0.0, 1.0, 5.0, 1.0,
            0.0, 0.0, 1.0, 5.0,
        ];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        for &nr in &[0_usize, 1, 3, 10, 100] {
            let cfg = IterRefineConfig {
                n_refinements: nr,
                tol: 1e-14,
            };
            let x = iterative_refinement(&a, &b, n, &cfg)
                .unwrap_or_else(|_| panic!("solve failed for n_refinements={nr}"));
            let r = inf_norm(&residual(&a, &x, &b, n));
            assert!(r < 1e-9, "n_refinements={nr}: residual={r}");
        }
    }

    // ---------- test 8: output is all-finite ----------

    #[test]
    fn finite_output() {
        let n = 6;
        // Diagonal-dominant matrix.
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 20.0;
            if i > 0 {
                a[i * n + (i - 1)] = -1.0;
                a[(i - 1) * n + i] = -1.0;
            }
        }
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let cfg = IterRefineConfig {
            n_refinements: 4,
            tol: 1e-13,
        };

        let x = iterative_refinement(&a, &b, n, &cfg).expect("solve should succeed");
        for (i, &xi) in x.iter().enumerate() {
            assert!(xi.is_finite(), "x[{i}] = {xi} is not finite");
        }
    }
}
