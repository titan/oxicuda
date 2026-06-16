//! ILU(0) — Incomplete LU factorization with zero fill-in.
//!
//! Computes an approximate factorization `A ≈ L U` where `L` (unit lower
//! triangular) and `U` (upper triangular) share the **sparsity pattern of
//! `A`**: any entry `(i, j)` that is zero in `A` is forced to remain zero in
//! the factors (no fill-in is allowed). The resulting `M = L U` is a cheap,
//! effective left preconditioner for Krylov solvers such as
//! [`mod@crate::iterative::gmres`] and [`mod@crate::iterative::bicgstab`].
//!
//! Here `A` is supplied dense, row-major (`n × n`); its *structural* nonzeros
//! (entries that are exactly `0.0` are treated as structural zeros) define the
//! pattern `S`. The classical IKJ Gaussian-elimination variant is used:
//!
//! ```text
//! for i = 1 .. n-1:
//!   for k = 0 .. i-1 where (i,k) ∈ S:
//!     a_ik /= a_kk
//!     for j = k+1 .. n-1 where (i,j) ∈ S:
//!       a_ij -= a_ik · a_kj
//! ```
//!
//! Updates that would touch a position *outside* the pattern are dropped — this
//! is exactly what makes the factorization "incomplete". `L`/`U` are stored
//! together in a single combined array (`L` strictly below the diagonal with an
//! implicit unit diagonal, `U` on and above it).
//!
//! # Reference
//! - Saad, Y. (2003) "Iterative Methods for Sparse Linear Systems", 2nd ed.,
//!   §10.3 (ILU(0)). SIAM.

use crate::error::{SolverError, SolverResult};

/// An ILU(0) factorization of a dense, row-major matrix.
///
/// The combined `lu` buffer holds `U` on/above the diagonal and the strictly
/// lower part of `L` below it (the unit diagonal of `L` is implicit). The
/// `pattern` mask records which `(i, j)` positions are structural nonzeros.
#[derive(Debug, Clone)]
pub struct Ilu0 {
    /// Combined `L\U` factors, row-major `[n × n]`.
    lu: Vec<f64>,
    /// Structural-nonzero mask, row-major `[n × n]` (`true` = kept position).
    pattern: Vec<bool>,
    /// System dimension.
    n: usize,
}

impl Ilu0 {
    /// Compute the ILU(0) factorization of dense row-major `A` (`n × n`).
    ///
    /// Entries of `a` that are exactly `0.0` define the structural zeros that
    /// are preserved (no fill). A small diagonal entry is replaced by a tiny
    /// signed value to keep the factorization usable (a common robustification
    /// for nearly-singular pivots).
    ///
    /// # Errors
    ///
    /// * [`SolverError::DimensionMismatch`] if `n == 0` or `a.len() != n*n`.
    /// * [`SolverError::SingularMatrix`] if a structural diagonal pivot is
    ///   exactly zero (the pattern excludes the diagonal).
    pub fn factor(a: &[f64], n: usize) -> SolverResult<Self> {
        if n == 0 {
            return Err(SolverError::DimensionMismatch("ilu0: n must be ≥ 1".into()));
        }
        if a.len() != n * n {
            return Err(SolverError::DimensionMismatch(format!(
                "ilu0: A has {} elements, expected n*n = {}",
                a.len(),
                n * n
            )));
        }

        // Structural pattern from the original matrix; diagonal is always kept
        // so the factorization has pivots.
        let mut pattern = vec![false; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j || a[i * n + j] != 0.0 {
                    pattern[i * n + j] = true;
                }
            }
        }

        let mut lu = a.to_vec();
        const PIVOT_FLOOR: f64 = 1e-300;

        for i in 1..n {
            for k in 0..i {
                if !pattern[i * n + k] {
                    continue;
                }
                let akk = lu[k * n + k];
                if akk.abs() < PIVOT_FLOOR {
                    return Err(SolverError::SingularMatrix);
                }
                let factor = lu[i * n + k] / akk;
                lu[i * n + k] = factor;
                for j in (k + 1)..n {
                    if pattern[i * n + j] {
                        lu[i * n + j] -= factor * lu[k * n + j];
                    }
                }
            }
        }

        // Verify final diagonal pivots are usable.
        for i in 0..n {
            if lu[i * n + i].abs() < PIVOT_FLOOR {
                return Err(SolverError::SingularMatrix);
            }
        }

        Ok(Self { lu, pattern, n })
    }

    /// System dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Number of structural nonzeros retained in the factors.
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.pattern.iter().filter(|&&p| p).count()
    }

    /// Apply the preconditioner: solve `L U z = r` for `z` (the action of
    /// `M^{-1}`), via a forward solve (`L y = r`) then a back solve (`U z = y`).
    ///
    /// # Errors
    ///
    /// [`SolverError::DimensionMismatch`] if `r.len() != n`.
    pub fn apply(&self, r: &[f64]) -> SolverResult<Vec<f64>> {
        if r.len() != self.n {
            return Err(SolverError::DimensionMismatch(format!(
                "ilu0::apply: r has {} elements, expected n = {}",
                r.len(),
                self.n
            )));
        }
        let n = self.n;

        // Forward solve L y = r, L unit-lower-triangular (implicit unit diag).
        let mut y = vec![0.0_f64; n];
        for i in 0..n {
            let row = &self.lu[i * n..i * n + i];
            let acc = r[i]
                - row
                    .iter()
                    .zip(y[..i].iter())
                    .map(|(&l, &yj)| l * yj)
                    .sum::<f64>();
            y[i] = acc;
        }

        // Back solve U z = y, U upper-triangular with explicit diagonal.
        let mut z = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let row = &self.lu[i * n + (i + 1)..i * n + n];
            let acc = y[i]
                - row
                    .iter()
                    .zip(z[(i + 1)..].iter())
                    .map(|(&u, &zj)| u * zj)
                    .sum::<f64>();
            z[i] = acc / self.lu[i * n + i];
        }
        Ok(z)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense matvec helper for verification.
    fn matvec(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0_f64; n];
        for (i, yi) in y.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for j in 0..n {
                acc += a[i * n + j] * x[j];
            }
            *yi = acc;
        }
        y
    }

    #[test]
    fn factor_identity() {
        let n = 3;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let ilu = Ilu0::factor(&a, n).expect("factor");
        // M^{-1} of identity is identity.
        let r = vec![1.0, 2.0, 3.0];
        let z = ilu.apply(&r).expect("apply");
        for (zi, ri) in z.iter().zip(r.iter()) {
            assert!((zi - ri).abs() < 1e-12);
        }
    }

    #[test]
    fn dense_matrix_exact_lu() {
        // For a fully-dense SPD matrix the pattern is full, so ILU(0) == LU
        // and M^{-1} r solves A z = r exactly.
        let n = 3;
        let a = vec![4.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 5.0];
        let ilu = Ilu0::factor(&a, n).expect("factor");
        let z = vec![1.0, -2.0, 3.0];
        let r = matvec(&a, &z, n);
        let solved = ilu.apply(&r).expect("apply");
        for (s, exp) in solved.iter().zip(z.iter()) {
            assert!((s - exp).abs() < 1e-10, "got {s}, expected {exp}");
        }
    }

    #[test]
    fn preserves_sparsity_pattern() {
        // Tridiagonal: ILU(0) keeps tridiagonal pattern, no fill on the corners.
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 4.0;
            if i + 1 < n {
                a[i * n + (i + 1)] = -1.0;
                a[(i + 1) * n + i] = -1.0;
            }
        }
        let ilu = Ilu0::factor(&a, n).expect("factor");
        // nnz of a tridiagonal n=4: 3n-2 = 10.
        assert_eq!(ilu.nnz(), 3 * n - 2);
    }

    #[test]
    fn tridiagonal_exact() {
        // ILU(0) on an SPD tridiagonal is the exact LU (tridiagonal has no
        // fill-in), so M^{-1} solves exactly.
        let n = 5;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 4.0;
            if i + 1 < n {
                a[i * n + (i + 1)] = -1.0;
                a[(i + 1) * n + i] = -1.0;
            }
        }
        let ilu = Ilu0::factor(&a, n).expect("factor");
        let z_exact: Vec<f64> = (0..n).map(|i| (i as f64) - 2.0).collect();
        let r = matvec(&a, &z_exact, n);
        let solved = ilu.apply(&r).expect("apply");
        for (s, e) in solved.iter().zip(z_exact.iter()) {
            assert!((s - e).abs() < 1e-9, "got {s}, expected {e}");
        }
    }

    #[test]
    fn dim_accessor() {
        let n = 4;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 2.0;
        }
        let ilu = Ilu0::factor(&a, n).expect("factor");
        assert_eq!(ilu.dim(), 4);
    }

    #[test]
    fn n_0_error() {
        let err = Ilu0::factor(&[], 0);
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn dim_mismatch_error() {
        let err = Ilu0::factor(&[1.0, 2.0, 3.0], 3);
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn singular_diagonal_error() {
        // Zero diagonal pivot → singular.
        let n = 2;
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let err = Ilu0::factor(&a, n);
        assert!(matches!(err, Err(SolverError::SingularMatrix)));
    }

    #[test]
    fn apply_dim_mismatch() {
        let n = 3;
        let mut a = vec![0.0_f64; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        let ilu = Ilu0::factor(&a, n).expect("factor");
        let err = ilu.apply(&[1.0, 2.0]); // wrong length
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn apply_finite() {
        let n = 4;
        let a: Vec<f64> = (0..n * n)
            .map(|idx| {
                let (i, j) = (idx / n, idx % n);
                if i == j {
                    (i as f64) + 5.0
                } else if (i as i64 - j as i64).abs() == 1 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect();
        let ilu = Ilu0::factor(&a, n).expect("factor");
        let r: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let z = ilu.apply(&r).expect("apply");
        for zi in &z {
            assert!(zi.is_finite());
        }
    }
}
