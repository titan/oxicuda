//! Block tridiagonal system solver using the block Thomas algorithm.
//!
//! Solves a block-tridiagonal linear system of the form:
//!
//! ```text
//! B[0]*x[0] + C[0]*x[1]                              = rhs[0]
//! A[i-1]*x[i-1] + B[i]*x[i] + C[i]*x[i+1]          = rhs[i]   (1 ≤ i ≤ n-2)
//! A[n-2]*x[n-2] + B[n-1]*x[n-1]                      = rhs[n-1]
//! ```
//!
//! Each diagonal block `B[i]`, sub-diagonal block `A[i]`, and super-diagonal
//! block `C[i]` is a `block_size × block_size` matrix stored in row-major order
//! as a flat `Vec<f64>`.  The right-hand side vectors `rhs[i]` and solution
//! vectors `x[i]` have length `block_size`.
//!
//! ## Algorithm — block Thomas (LU-based)
//!
//! The standard scalar Thomas algorithm is lifted to the block setting by
//! replacing scalar divisions with small-block LU solves.  The complexity is
//! O(n · block_size³) — optimal for this structure.

use crate::error::{SolverError, SolverResult};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solve a block-tridiagonal linear system using the block Thomas algorithm.
///
/// ## Parameters
///
/// * `a`          – sub-diagonal blocks (length `n-1`); each entry is a flat
///   row-major `block_size × block_size` matrix.
/// * `b`          – diagonal blocks (length `n`).
/// * `c`          – super-diagonal blocks (length `n-1`).
/// * `rhs`        – right-hand sides (length `n`); each entry has length
///   `block_size`.
/// * `n`          – number of block rows.
/// * `block_size` – dimension of each square block.
///
/// ## Returns
///
/// Solution vectors `x` (length `n`, each of length `block_size`).
///
/// ## Errors
///
/// * [`SolverError::DimensionMismatch`] – if any slice length or block size is wrong.
/// * [`SolverError::SingularMatrix`]    – if a pivot block is numerically singular.
pub fn block_tridiagonal_solve(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    rhs: &[Vec<f64>],
    n: usize,
    block_size: usize,
) -> SolverResult<Vec<Vec<f64>>> {
    validate_inputs(a, b, c, rhs, n, block_size)?;

    if n == 1 {
        // Trivial: single block solve B[0] * x[0] = rhs[0]
        let x0 = block_lu_solve(&b[0], &rhs[0], block_size)?;
        return Ok(vec![x0]);
    }

    let bs = block_size;
    let bs2 = bs * bs;

    // -----------------------------------------------------------------
    // Storage for modified quantities accumulated during forward sweep.
    //
    //   b_prime[i]   – modified diagonal block after elimination
    //   w[i]         – upper factor W[i] = inv(B'[i]) * C[i]   (i = 0..n-2)
    //   d_prime[i]   – modified RHS after elimination
    //
    // We do not allocate separate arrays for b_prime; instead we keep
    // the most recently updated block in `b_cur`.
    // -----------------------------------------------------------------

    // w[i] is a bs×bs matrix, stored flat (i = 0..n-2).
    let mut w: Vec<Vec<f64>> = vec![vec![0.0_f64; bs2]; n - 1];
    // d_prime[i] is a bs-vector (i = 0..n).
    let mut d_prime: Vec<Vec<f64>> = rhs.to_vec();
    // b_prime[i]: we store the full sequence so that back-substitution can use W[i].
    // However W[i] already encodes inv(B'[i]) * C[i], and d_prime[i] = inv(B'[i]) * rhs'[i].
    // We need no extra storage for b_prime beyond what we use for the W computation.

    // Current modified diagonal block (updated in-place row by row).
    let mut b_cur: Vec<f64> = b[0].clone();

    // Forward sweep: i = 0 .. n-2
    for i in 0..n - 1 {
        // W[i] = inv(B'[i]) * C[i]
        w[i] = block_lu_invert_mul(&b_cur, &c[i], bs)?;

        // d_prime[i] = inv(B'[i]) * d_prime[i]
        d_prime[i] = block_lu_solve(&b_cur, &d_prime[i], bs)?;

        // B'[i+1] = B[i+1] - A[i] * W[i]
        let aw = block_mat_mat_mul(&a[i], &w[i], bs);
        b_cur = block_mat_sub(&b[i + 1], &aw, bs);

        // d_prime[i+1] = d_prime[i+1] - A[i] * d_prime[i]
        let ad = block_mat_vec_mul(&a[i], &d_prime[i], bs);
        d_prime[i + 1] = block_vec_sub(&d_prime[i + 1], &ad);
    }

    // Final solve for x[n-1]: inv(B'[n-1]) * d_prime[n-1]
    let mut x: Vec<Vec<f64>> = vec![vec![0.0_f64; bs]; n];
    x[n - 1] = block_lu_solve(&b_cur, &d_prime[n - 1], bs)?;

    // Back substitution: i = n-2 down to 0
    // x[i] = d_prime[i] - W[i] * x[i+1]
    for i in (0..n - 1).rev() {
        let wx = block_mat_vec_mul(&w[i], &x[i + 1], bs);
        x[i] = block_vec_sub(&d_prime[i], &wx);
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Block arithmetic helpers (private)
// ---------------------------------------------------------------------------

/// Multiply a `bs × bs` matrix `m` (row-major) by a length-`bs` vector `v`.
fn block_mat_vec_mul(m: &[f64], v: &[f64], bs: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; bs];
    for row in 0..bs {
        let mut acc = 0.0_f64;
        for col in 0..bs {
            acc += m[row * bs + col] * v[col];
        }
        result[row] = acc;
    }
    result
}

/// Multiply two `bs × bs` matrices `a` and `b` (both row-major).
/// Returns a flat row-major `bs × bs` result.
fn block_mat_mat_mul(a: &[f64], b: &[f64], bs: usize) -> Vec<f64> {
    let mut result = vec![0.0_f64; bs * bs];
    for row in 0..bs {
        for col in 0..bs {
            let mut acc = 0.0_f64;
            for k in 0..bs {
                acc += a[row * bs + k] * b[k * bs + col];
            }
            result[row * bs + col] = acc;
        }
    }
    result
}

/// Elementwise subtract two `bs × bs` matrices: returns `a - b`.
fn block_mat_sub(a: &[f64], b: &[f64], bs: usize) -> Vec<f64> {
    let len = bs * bs;
    let mut result = vec![0.0_f64; len];
    for i in 0..len {
        result[i] = a[i] - b[i];
    }
    result
}

/// Elementwise subtract two length-`bs` vectors: returns `a - b`.
fn block_vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Solve `mat * x = rhs` for a `bs × bs` matrix and length-`bs` RHS using
/// LU factorization with partial pivoting.
///
/// For small blocks (the expected use-case) this is very fast and exact.
fn block_lu_solve(mat: &[f64], rhs: &[f64], bs: usize) -> SolverResult<Vec<f64>> {
    let (lu, piv) = lu_factorize(mat, bs)?;
    Ok(lu_forward_back(&lu, &piv, rhs, bs))
}

/// Solve `mat * X = other` where both `mat` and `other` are `bs × bs` matrices.
///
/// Equivalent to computing `mat⁻¹ · other` column by column.
/// Returns a flat row-major `bs × bs` result.
fn block_lu_invert_mul(mat: &[f64], other: &[f64], bs: usize) -> SolverResult<Vec<f64>> {
    let (lu, piv) = lu_factorize(mat, bs)?;
    let mut result = vec![0.0_f64; bs * bs];

    // Solve for each column of `other` independently.
    for col in 0..bs {
        // Extract column `col` from `other` (row-major).
        let rhs_col: Vec<f64> = (0..bs).map(|row| other[row * bs + col]).collect();
        let sol = lu_forward_back(&lu, &piv, &rhs_col, bs);
        // Store solution back into column `col` of result (row-major).
        for row in 0..bs {
            result[row * bs + col] = sol[row];
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// LU factorization with partial pivoting (private, for small blocks)
// ---------------------------------------------------------------------------

/// Compute the LU factorization of an `n × n` matrix (row-major, flat).
///
/// Returns `(lu, piv)` where `lu` stores `L` (unit lower-triangular) and `U`
/// (upper-triangular) in the same flat array (LAPACK convention), and `piv`
/// is the pivot index vector.
///
/// # Errors
///
/// Returns [`SolverError::SingularMatrix`] if any pivot is numerically zero.
fn lu_factorize(a: &[f64], n: usize) -> SolverResult<(Vec<f64>, Vec<usize>)> {
    let mut lu = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // Find pivot: row with maximum absolute value in column k, rows k..n.
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for row in k + 1..n {
            let val = lu[row * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows k and max_row.
        if max_row != k {
            for col in 0..n {
                lu.swap(k * n + col, max_row * n + col);
            }
            piv.swap(k, max_row);
        }

        let pivot = lu[k * n + k];
        if pivot.abs() < 1e-300 {
            return Err(SolverError::SingularMatrix);
        }

        // Eliminate below the pivot.
        for row in k + 1..n {
            let factor = lu[row * n + k] / pivot;
            lu[row * n + k] = factor; // store L below diagonal
            for col in k + 1..n {
                let u_val = lu[k * n + col];
                lu[row * n + col] -= factor * u_val;
            }
        }
    }

    Ok((lu, piv))
}

/// Forward and backward substitution using an LU factorization produced by
/// [`lu_factorize`].
///
/// Solves `L * U * x = P * b` (where `P` is encoded in `piv`).
fn lu_forward_back(lu: &[f64], piv: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    // Apply row permutation to b.
    let mut x: Vec<f64> = vec![0.0_f64; n];
    // Build inverse permutation mapping: position[piv[i]] = i
    // Actually piv[i] records which original row ended up at row i, so
    // the permuted RHS is just b[piv[i]] for row i.
    for i in 0..n {
        x[i] = b[piv[i]];
    }

    // Forward substitution: solve L * y = P*b  (L is unit lower-triangular)
    for i in 1..n {
        for j in 0..i {
            x[i] -= lu[i * n + j] * x[j];
        }
    }

    // Back substitution: solve U * x = y
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lu[i * n + j] * x[j];
        }
        x[i] /= lu[i * n + i];
    }

    x
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate_inputs(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    rhs: &[Vec<f64>],
    n: usize,
    block_size: usize,
) -> SolverResult<()> {
    if n == 0 {
        return Err(SolverError::DimensionMismatch(
            "block_tridiagonal_solve: n must be >= 1".into(),
        ));
    }

    if b.len() != n {
        return Err(SolverError::DimensionMismatch(format!(
            "block_tridiagonal_solve: b.len() ({}) != n ({})",
            b.len(),
            n
        )));
    }
    if rhs.len() != n {
        return Err(SolverError::DimensionMismatch(format!(
            "block_tridiagonal_solve: rhs.len() ({}) != n ({})",
            rhs.len(),
            n
        )));
    }
    if a.len() != n - 1 {
        return Err(SolverError::DimensionMismatch(format!(
            "block_tridiagonal_solve: a.len() ({}) != n-1 ({})",
            a.len(),
            n - 1
        )));
    }
    if c.len() != n - 1 {
        return Err(SolverError::DimensionMismatch(format!(
            "block_tridiagonal_solve: c.len() ({}) != n-1 ({})",
            c.len(),
            n - 1
        )));
    }

    let bs2 = block_size * block_size;

    for (i, bi) in b.iter().enumerate() {
        if bi.len() != bs2 {
            return Err(SolverError::DimensionMismatch(format!(
                "block_tridiagonal_solve: b[{}].len() ({}) != block_size^2 ({})",
                i,
                bi.len(),
                bs2
            )));
        }
    }
    for (i, ai) in a.iter().enumerate() {
        if ai.len() != bs2 {
            return Err(SolverError::DimensionMismatch(format!(
                "block_tridiagonal_solve: a[{}].len() ({}) != block_size^2 ({})",
                i,
                ai.len(),
                bs2
            )));
        }
    }
    for (i, ci) in c.iter().enumerate() {
        if ci.len() != bs2 {
            return Err(SolverError::DimensionMismatch(format!(
                "block_tridiagonal_solve: c[{}].len() ({}) != block_size^2 ({})",
                i,
                ci.len(),
                bs2
            )));
        }
    }
    for (i, ri) in rhs.iter().enumerate() {
        if ri.len() != block_size {
            return Err(SolverError::DimensionMismatch(format!(
                "block_tridiagonal_solve: rhs[{}].len() ({}) != block_size ({})",
                i,
                ri.len(),
                block_size
            )));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build identity block of size `bs`.
    fn identity(bs: usize) -> Vec<f64> {
        let mut m = vec![0.0_f64; bs * bs];
        for i in 0..bs {
            m[i * bs + i] = 1.0;
        }
        m
    }

    // Helper: build a zero block of size `bs`.
    fn zeros_block(bs: usize) -> Vec<f64> {
        vec![0.0_f64; bs * bs]
    }

    // Helper: compute A*x (block-tridiagonal matrix-vector product) for
    // verification of solution quality.
    fn matvec_block_tridiag(
        a_blocks: &[Vec<f64>],
        b_blocks: &[Vec<f64>],
        c_blocks: &[Vec<f64>],
        x: &[Vec<f64>],
        n: usize,
        bs: usize,
    ) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0_f64; bs]; n];
        for i in 0..n {
            // Contribution of B[i] * x[i]
            let bx = block_mat_vec_mul(&b_blocks[i], &x[i], bs);
            for k in 0..bs {
                result[i][k] += bx[k];
            }
            // Contribution of C[i] * x[i+1]
            if i < n - 1 {
                let cx = block_mat_vec_mul(&c_blocks[i], &x[i + 1], bs);
                for k in 0..bs {
                    result[i][k] += cx[k];
                }
            }
            // Contribution of A[i-1] * x[i-1]
            if i > 0 {
                let ax = block_mat_vec_mul(&a_blocks[i - 1], &x[i - 1], bs);
                for k in 0..bs {
                    result[i][k] += ax[k];
                }
            }
        }
        result
    }

    // ---------- test 1: known scalar tridiagonal [-1, 2, -1] with bs=1 ----------

    #[test]
    fn known_scalar_tridiagonal() {
        // 3-equation system:  [-1, 2, -1] tridiagonal with rhs = [1, 0, 1]
        // Solution: [1, 1, 1]
        let n = 3;
        let bs = 1;
        let b = vec![vec![2.0], vec![2.0], vec![2.0]];
        let a = vec![vec![-1.0], vec![-1.0]];
        let c = vec![vec![-1.0], vec![-1.0]];
        let rhs = vec![vec![1.0], vec![0.0], vec![1.0]];

        let x = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs).expect("solve should succeed");

        assert_eq!(x.len(), n);
        assert!((x[0][0] - 1.0).abs() < 1e-10, "x[0] = {}", x[0][0]);
        assert!((x[1][0] - 1.0).abs() < 1e-10, "x[1] = {}", x[1][0]);
        assert!((x[2][0] - 1.0).abs() < 1e-10, "x[2] = {}", x[2][0]);
    }

    // ---------- test 2: solution satisfies A*x ≈ b ----------

    #[test]
    fn solution_satisfies_ax_b() {
        // 4-block system with bs=2.
        let n = 4;
        let bs = 2;
        // Diagonal blocks: 4*I
        let b_block = vec![4.0, 0.0, 0.0, 4.0];
        let b = vec![b_block.clone(); n];
        // Sub/super-diagonal: -I
        let off_block = vec![-1.0, 0.0, 0.0, -1.0];
        let a = vec![off_block.clone(); n - 1];
        let c = vec![off_block.clone(); n - 1];
        // RHS
        let rhs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64, (i + 1) as f64]).collect();

        let x = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs).expect("solve should succeed");

        // Verify A*x ≈ rhs
        let ax = matvec_block_tridiag(&a, &b, &c, &x, n, bs);
        for i in 0..n {
            for k in 0..bs {
                assert!(
                    (ax[i][k] - rhs[i][k]).abs() < 1e-9,
                    "residual[{i}][{k}] = {}",
                    ax[i][k] - rhs[i][k]
                );
            }
        }
    }

    // ---------- test 3: n=1 trivial single-block solve ----------

    #[test]
    fn n_1_trivial_solve() {
        // Single block: [3 1; 2 4] * x = [5; 6]
        // det = 12-2 = 10, x = [1/10*(4*5-1*6), 1/10*(3*6-2*5)] = [14/10, 8/10]
        let n = 1;
        let bs = 2;
        let b = vec![vec![3.0, 1.0, 2.0, 4.0]];
        let a: Vec<Vec<f64>> = vec![];
        let c: Vec<Vec<f64>> = vec![];
        let rhs = vec![vec![5.0, 6.0]];

        let x = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs)
            .expect("single-block solve should succeed");

        assert_eq!(x.len(), 1);
        assert!((x[0][0] - 1.4).abs() < 1e-10, "x[0][0] = {}", x[0][0]);
        assert!((x[0][1] - 0.8).abs() < 1e-10, "x[0][1] = {}", x[0][1]);
    }

    // ---------- test 4: identity diagonal → solution = rhs ----------

    #[test]
    fn identity_diagonal_solves_trivially() {
        let n = 5;
        let bs = 3;
        let b = vec![identity(bs); n];
        let a = vec![zeros_block(bs); n - 1];
        let c = vec![zeros_block(bs); n - 1];
        let rhs: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![i as f64, (2 * i) as f64, (3 * i) as f64])
            .collect();

        let x = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs)
            .expect("identity diagonal solve should succeed");

        for i in 0..n {
            for k in 0..bs {
                assert!(
                    (x[i][k] - rhs[i][k]).abs() < 1e-12,
                    "x[{i}][{k}] = {} vs rhs = {}",
                    x[i][k],
                    rhs[i][k]
                );
            }
        }
    }

    // ---------- test 5: block_size=1 matches scalar Thomas ----------

    #[test]
    fn block_size_1_matches_scalar() {
        // System:  [2 -1 0; -1 2 -1; 0 -1 2] * x = [1; 1; 1]
        // Solution: [1.5, 2.0, 1.5]
        let n = 3;
        let bs = 1;
        let b = vec![vec![2.0], vec![2.0], vec![2.0]];
        let a = vec![vec![-1.0], vec![-1.0]];
        let c = vec![vec![-1.0], vec![-1.0]];
        let rhs = vec![vec![1.0], vec![1.0], vec![1.0]];

        let x = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs).expect("solve should succeed");

        assert!((x[0][0] - 1.5).abs() < 1e-10, "x[0] = {}", x[0][0]);
        assert!((x[1][0] - 2.0).abs() < 1e-10, "x[1] = {}", x[1][0]);
        assert!((x[2][0] - 1.5).abs() < 1e-10, "x[2] = {}", x[2][0]);
    }

    // ---------- test 6: singular diagonal block → SolverError::SingularMatrix ----------

    #[test]
    fn singular_block_error() {
        let n = 2;
        let bs = 2;
        // Singular diagonal block: zero matrix
        let b_singular = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![b_singular, identity(bs)];
        let a = vec![identity(bs)];
        let c = vec![identity(bs)];
        let rhs = vec![vec![1.0, 1.0], vec![1.0, 1.0]];

        let result = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs);
        assert!(
            matches!(result, Err(SolverError::SingularMatrix)),
            "expected SingularMatrix, got {:?}",
            result
        );
    }

    // ---------- test 7: dimension mismatch errors ----------

    #[test]
    fn dimension_mismatch_error() {
        let n = 3;
        let bs = 2;
        let b = vec![identity(bs); n];
        let a = vec![identity(bs); n - 1];
        let c = vec![identity(bs); n - 1];
        let rhs = vec![vec![1.0, 1.0]; n];

        // n = 0
        let err = block_tridiagonal_solve(&a, &b, &c, &rhs, 0, bs);
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));

        // Wrong number of b blocks
        let b_wrong = vec![identity(bs); n + 1];
        let err = block_tridiagonal_solve(&a, &b_wrong, &c, &rhs, n, bs);
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));

        // Wrong block size in rhs
        let rhs_bad: Vec<Vec<f64>> = vec![vec![1.0]; n]; // wrong length per entry
        let err = block_tridiagonal_solve(&a, &b, &c, &rhs_bad, n, bs);
        assert!(matches!(err, Err(SolverError::DimensionMismatch(_))));
    }

    // ---------- test 8: solution is all-finite ----------

    #[test]
    fn solution_finite() {
        let n = 6;
        let bs = 2;
        // Well-conditioned diagonal dominant blocks.
        let b_block = vec![10.0, 1.0, 1.0, 10.0];
        let a_block = vec![1.0, 0.0, 0.0, 1.0];
        let c_block = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![b_block; n];
        let a = vec![a_block; n - 1];
        let c = vec![c_block; n - 1];
        let rhs: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 + 1.0, 2.0]).collect();

        let x = block_tridiagonal_solve(&a, &b, &c, &rhs, n, bs).expect("solve should succeed");

        for (i, xi) in x.iter().enumerate() {
            for (k, xik) in xi.iter().enumerate() {
                assert!(xik.is_finite(), "x[{i}][{k}] = {} is not finite", xik);
            }
        }
    }
}
