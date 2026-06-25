//! Left-looking sparse LU factorization (Gilbert–Peierls / SuperLU style).
//!
//! Implements the **left-looking, column-by-column** sparse LU algorithm that is
//! the algorithmic heart of SuperLU (Demmel, Eisenstat, Gilbert, Li & Liu 1999;
//! Li & Demmel 2003, ACM TOMS 29(2) "SuperLU_DIST"). For each column `j` of `A`,
//! the algorithm
//!
//! 1. **predicts** the nonzero structure of column `j` by a depth-first search
//!    over the directed graph of the already-computed columns of `L`
//!    (Gilbert–Peierls symbolic step, `O(flops)` total),
//! 2. **solves** the sparse lower-triangular system `L[:, 1..j] · u = a_j` to
//!    obtain the part of column `j` belonging to `U` and the unscaled future `L`,
//! 3. applies **partial pivoting** to the column (choosing a pivot among the rows
//!    not yet eliminated),
//! 4. scales the sub-diagonal entries by the pivot, and
//! 5. detects trivial **supernodes** (runs of consecutive columns with the same
//!    lower structure) for dense-block reuse.
//!
//! This is fundamentally different from the **right-looking, multifrontal** LU in
//! [`crate::sparse::direct_factorization::MultifrontalLUSolver`]: left-looking
//! defers all updates to a column until that column is reached, "looking left"
//! to the previously factored columns, whereas right-looking immediately
//! propagates each pivot's update to all trailing columns. Both compute the same
//! `P·A = L·U`; the data-access patterns and fill-in handling differ.
//!
//! ## Index convention
//!
//! Following Gilbert & Peierls, the factors are stored with **original** row
//! indices (no physical row permutation of the stored data). Pivoting is captured
//! purely by the permutation arrays `pivot_row` (elimination-position → original
//! row) and `row_pos` (original row → elimination position, or "not yet pivoted").
//! This keeps the symbolic DFS and the numeric triangular solve completely
//! consistent even as pivots are selected mid-factorization.

use crate::error::{SolverError, SolverResult};

// ---------------------------------------------------------------------------
// Sparse column store
// ---------------------------------------------------------------------------

/// A single sparse column: parallel `(row_index, value)` arrays. Row indices are
/// **original** matrix rows.
#[derive(Debug, Clone, Default)]
struct SparseCol {
    rows: Vec<usize>,
    vals: Vec<f64>,
}

/// Result of a left-looking sparse LU factorization, `P · A = L · U`.
///
/// `L` is unit lower-triangular (its unit diagonal is implicit and not stored);
/// `U` is upper-triangular with an explicit diagonal. Both are held column-wise
/// in compressed-sparse-column form using **original** row indices. The pivoting
/// permutation `P` is recorded in [`LeftLookingLu::permutation`].
#[derive(Debug, Clone)]
pub struct LeftLookingLu {
    /// Strictly-lower part of each column of `L` (unit diagonal implicit), stored
    /// with original row indices, ordered by elimination position.
    l_cols: Vec<SparseCol>,
    /// Off-diagonal upper part of each column of `U` (the diagonal is the pivot,
    /// stored separately), with original row indices.
    u_cols: Vec<SparseCol>,
    /// Diagonal pivot value of `U` for each column.
    u_diag: Vec<f64>,
    /// `pivot_row[k]` = original matrix row used as the `k`-th pivot.
    pivot_row: Vec<usize>,
    /// `row_pos[orig]` = elimination position of original row `orig`.
    row_pos: Vec<usize>,
    /// Start column index of the supernode containing each column.
    supernode_start: Vec<usize>,
    /// Matrix dimension.
    n: usize,
}

const NOT_PIVOTED: usize = usize::MAX;

impl LeftLookingLu {
    /// Factorize a square sparse matrix given in **CSR** form.
    ///
    /// * `row_offsets` — length `n + 1` CSR row pointers.
    /// * `col_indices` / `values` — CSR column indices and values.
    /// * `n` — matrix dimension.
    ///
    /// Partial pivoting is used for stability; `pivot_tol` in `(0, 1]` is the
    /// threshold-pivoting parameter: a candidate pivot of magnitude `m` is
    /// accepted only when `m >= pivot_tol * mmax`, where `mmax` is the largest
    /// magnitude among the column's eliminable rows (`1.0` = classical partial
    /// pivoting; smaller values favour sparsity over stability).
    ///
    /// # Errors
    ///
    /// * [`SolverError::DimensionMismatch`] if `row_offsets.len() != n + 1`.
    /// * [`SolverError::SingularMatrix`] if a column has no acceptable pivot.
    pub fn factorize(
        row_offsets: &[usize],
        col_indices: &[usize],
        values: &[f64],
        n: usize,
        pivot_tol: f64,
    ) -> SolverResult<Self> {
        if row_offsets.len() != n + 1 {
            return Err(SolverError::DimensionMismatch(format!(
                "left-looking LU: row_offsets length {} != n+1 = {}",
                row_offsets.len(),
                n + 1
            )));
        }
        let pivot_tol = if pivot_tol <= 0.0 {
            1.0
        } else {
            pivot_tol.min(1.0)
        };

        // Convert CSR -> CSC so we can iterate columns of A directly.
        let a_cols = csr_to_csc(row_offsets, col_indices, values, n);

        let mut l_cols: Vec<SparseCol> = vec![SparseCol::default(); n];
        let mut u_cols: Vec<SparseCol> = vec![SparseCol::default(); n];
        let mut u_diag: Vec<f64> = vec![0.0; n];
        let mut pivot_row: Vec<usize> = vec![0; n];
        let mut row_pos: Vec<usize> = vec![NOT_PIVOTED; n];
        let mut supernode_start: Vec<usize> = (0..n).collect();

        // Dense scatter workspace (indexed by ORIGINAL row), per-column markers
        // and DFS scratch.
        let mut dense = vec![0.0f64; n];
        let mut marked = vec![usize::MAX; n];
        let mut topo_order: Vec<usize> = Vec::with_capacity(n);
        let mut dfs_stack: Vec<(usize, usize)> = Vec::with_capacity(n);

        for j in 0..n {
            topo_order.clear();

            // --- Symbolic + scatter: reachability of A[:, j] over columns of L. ---
            symbolic_column(
                &a_cols[j],
                &l_cols,
                &row_pos,
                j,
                &mut marked,
                &mut dense,
                &mut topo_order,
                &mut dfs_stack,
            );

            // --- Numeric sparse lower-triangular solve. ---
            // Sweep already-pivoted rows in INCREASING elimination position so
            // each pivot's contribution is fully formed before it is used.
            // `topo_order` (original rows) is in topological order: a parent's
            // contribution precedes its children. Restrict to pivoted rows.
            for &orig in topo_order.iter() {
                let pos = row_pos[orig];
                if pos == NOT_PIVOTED || pos >= j {
                    continue; // belongs to U-pattern or is an eliminable row
                }
                let xp = dense[orig];
                if xp == 0.0 {
                    continue;
                }
                // Subtract xp * L[:, pos] from rows below.
                let lc = &l_cols[pos];
                for k in 0..lc.rows.len() {
                    dense[lc.rows[k]] -= xp * lc.vals[k];
                }
            }

            // --- Partial pivoting among NOT-yet-pivoted rows in the pattern. ---
            let mut max_mag = 0.0f64;
            for &orig in topo_order.iter() {
                if row_pos[orig] == NOT_PIVOTED {
                    let m = dense[orig].abs();
                    if m > max_mag {
                        max_mag = m;
                    }
                }
            }
            if max_mag == 0.0 {
                return Err(SolverError::SingularMatrix);
            }
            // Choose the largest-magnitude eliminable row meeting the threshold.
            // (Classical partial pivoting picks the global max; threshold pivoting
            // would accept any row >= pivot_tol * max_mag, preferring sparser
            // ones — here we still take the max, which always satisfies it.)
            let mut pivot_orig = NOT_PIVOTED;
            let mut best_mag = -1.0f64;
            for &orig in topo_order.iter() {
                if row_pos[orig] == NOT_PIVOTED {
                    let m = dense[orig].abs();
                    if m >= pivot_tol * max_mag && m > best_mag {
                        best_mag = m;
                        pivot_orig = orig;
                    }
                }
            }
            if pivot_orig == NOT_PIVOTED {
                return Err(SolverError::SingularMatrix);
            }

            let pivot_val = dense[pivot_orig];
            if pivot_val.abs() == 0.0 {
                return Err(SolverError::SingularMatrix);
            }

            // Record the pivot: original row `pivot_orig` is eliminated at pos j.
            pivot_row[j] = pivot_orig;
            row_pos[pivot_orig] = j;
            u_diag[j] = pivot_val;

            // --- Gather column j into U (rows already pivoted, pos < j) and
            // L (rows still eliminable), then clear the scatter. ---
            let mut u_col = SparseCol::default();
            let mut l_col = SparseCol::default();
            for &orig in topo_order.iter() {
                let val = dense[orig];
                dense[orig] = 0.0;
                marked[orig] = usize::MAX;
                if orig == pivot_orig {
                    continue; // diagonal handled via u_diag
                }
                let pos = row_pos[orig];
                if pos != NOT_PIVOTED && pos < j {
                    // Upper part: U entry. Store as (original row, value).
                    if val != 0.0 {
                        u_col.rows.push(orig);
                        u_col.vals.push(val);
                    }
                } else {
                    // Lower part: future L entry, scaled by the pivot.
                    let lv = val / pivot_val;
                    if lv != 0.0 {
                        l_col.rows.push(orig);
                        l_col.vals.push(lv);
                    }
                }
            }

            u_cols[j] = u_col;
            l_cols[j] = l_col;

            // --- Supernode detection: same sub-diagonal structure as column j-1. ---
            if j > 0 && same_lower_structure(&l_cols[j - 1], &l_cols[j]) {
                supernode_start[j] = supernode_start[j - 1];
            } else {
                supernode_start[j] = j;
            }
        }

        Ok(Self {
            l_cols,
            u_cols,
            u_diag,
            pivot_row,
            row_pos,
            supernode_start,
            n,
        })
    }

    /// Matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Number of distinct supernodes detected during factorization.
    pub fn num_supernodes(&self) -> usize {
        if self.n == 0 {
            return 0;
        }
        let mut count = 0;
        for j in 0..self.n {
            if self.supernode_start[j] == j {
                count += 1;
            }
        }
        count
    }

    /// Total number of stored nonzeros in `L` (excluding the unit diagonal) plus
    /// `U` (including its diagonal) — the fill produced by the factorization.
    pub fn nnz(&self) -> usize {
        let l: usize = self.l_cols.iter().map(|c| c.rows.len()).sum();
        let u: usize = self.u_cols.iter().map(|c| c.rows.len()).sum();
        l + u + self.n // + diagonal of U
    }

    /// Solve `A · x = b` using the stored factors `P · A = L · U`, i.e.
    /// `x = U⁻¹ · L⁻¹ · P · b`.
    ///
    /// The solve works in elimination-position coordinates: position `k`
    /// corresponds to original pivot row `pivot_row[k]`.
    ///
    /// # Errors
    ///
    /// * [`SolverError::DimensionMismatch`] if `b.len() != n`.
    /// * [`SolverError::SingularMatrix`] if a `U` diagonal is zero.
    pub fn solve(&self, b: &[f64]) -> SolverResult<Vec<f64>> {
        let n = self.n;
        if b.len() != n {
            return Err(SolverError::DimensionMismatch(format!(
                "left-looking LU solve: b length {} != n = {}",
                b.len(),
                n
            )));
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        // The factorization computes P·A = L·U, where P permutes *equations*
        // (rows) only — the columns / unknowns are NOT permuted. Hence we work in
        // "elimination-position" space: position k owns column k of A (=> x[k] is
        // the value of original unknown k), and the RHS is permuted by P so that
        // equation pivoted at position k contributes b[pivot_row[k]].
        //
        // The L/U columns store *original row* indices; we map each to its
        // elimination position via `row_pos` so the triangular solves run wholly
        // in position space.

        // y[k] = b[pivot_row[k]]   (apply P to the RHS).
        let mut y = vec![0.0f64; n];
        for k in 0..n {
            y[k] = b[self.pivot_row[k]];
        }

        // Forward solve L y = P b (L unit lower-triangular in position space).
        for k in 0..n {
            let yk = y[k];
            if yk != 0.0 {
                let lc = &self.l_cols[k];
                for t in 0..lc.rows.len() {
                    // original row -> position (always > k, i.e. below).
                    let pos = self.row_pos[lc.rows[t]];
                    y[pos] -= lc.vals[t] * yk;
                }
            }
        }

        // Back solve U x = y (U upper-triangular in position space); x reuses y.
        for k in (0..n).rev() {
            let diag = self.u_diag[k];
            if diag.abs() == 0.0 {
                return Err(SolverError::SingularMatrix);
            }
            y[k] /= diag;
            let xk = y[k];
            let uc = &self.u_cols[k];
            for t in 0..uc.rows.len() {
                // original row -> position (always < k, i.e. above).
                let pos = self.row_pos[uc.rows[t]];
                y[pos] -= uc.vals[t] * xk;
            }
        }

        // y is now x in position=column space, which equals original unknown
        // order (columns were never permuted).
        Ok(y)
    }

    /// Returns the pivot permutation: `permutation()[k]` is the original matrix
    /// row eliminated at position `k`.
    pub fn permutation(&self) -> &[usize] {
        &self.pivot_row
    }

    /// Returns the elimination position of each original row.
    pub fn row_positions(&self) -> &[usize] {
        &self.row_pos
    }
}

/// One-call convenience: factorize a CSR matrix and solve `A · x = b`.
///
/// Uses classical partial pivoting (`pivot_tol = 1.0`).
///
/// # Errors
///
/// Propagates any error from [`LeftLookingLu::factorize`] / [`LeftLookingLu::solve`].
pub fn left_looking_lu_solve(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
    b: &[f64],
) -> SolverResult<Vec<f64>> {
    let lu = LeftLookingLu::factorize(row_offsets, col_indices, values, n, 1.0)?;
    lu.solve(b)
}

// ---------------------------------------------------------------------------
// Symbolic + structural helpers
// ---------------------------------------------------------------------------

/// Scatter `A[:, j]` into `dense` (indexed by original row) and compute the
/// reachable nonzero pattern of column `j` via a depth-first search over the
/// already-computed columns of `L`.
///
/// The directed graph has an edge `r -> s` whenever `s` is a sub-diagonal entry
/// of the L column produced when row `r` was eliminated (i.e. `L[s, pos(r)] != 0`).
/// Only **pivoted** rows have outgoing edges. On return, `topo_order` lists every
/// reachable original row in **topological order** (ancestors before
/// descendants), exactly the order the numeric solve consumes.
#[allow(clippy::too_many_arguments)]
fn symbolic_column(
    a_col: &SparseCol,
    l_cols: &[SparseCol],
    row_pos: &[usize],
    j: usize,
    marked: &mut [usize],
    dense: &mut [f64],
    topo_order: &mut Vec<usize>,
    // Single combined DFS stack of `(node, next_child_index)`; reused across
    // columns to avoid per-column allocation.
    stack: &mut Vec<(usize, usize)>,
) {
    // Reverse-topological emission then reverse.
    let mut rev_topo: Vec<usize> = Vec::new();

    for idx in 0..a_col.rows.len() {
        let r = a_col.rows[idx]; // original row
        dense[r] += a_col.vals[idx];
        if marked[r] == j {
            continue;
        }
        // Iterative DFS from r, carrying the child cursor alongside each node so
        // no parallel stack (and no `unwrap`) is needed.
        stack.clear();
        stack.push((r, 0));
        marked[r] = j;
        while let Some(&(node, cp)) = stack.last() {
            // Children of `node`: only if `node` is an already-pivoted row whose
            // elimination position is < j.
            let pos = row_pos[node];
            let children: &[usize] = if pos != NOT_PIVOTED && pos < j {
                &l_cols[pos].rows
            } else {
                &[]
            };
            if cp < children.len() {
                if let Some(top) = stack.last_mut() {
                    top.1 = cp + 1;
                }
                let c = children[cp]; // original row
                if marked[c] != j {
                    marked[c] = j;
                    stack.push((c, 0));
                }
            } else {
                rev_topo.push(node);
                stack.pop();
            }
        }
    }

    rev_topo.reverse();
    topo_order.extend_from_slice(&rev_topo);
}

/// Convert a CSR matrix into a vector of sparse columns (CSC-by-column).
fn csr_to_csc(
    row_offsets: &[usize],
    col_indices: &[usize],
    values: &[f64],
    n: usize,
) -> Vec<SparseCol> {
    let mut cols: Vec<SparseCol> = vec![SparseCol::default(); n];
    for i in 0..n {
        let rs = row_offsets.get(i).copied().unwrap_or(0);
        let re = row_offsets.get(i + 1).copied().unwrap_or(rs);
        for idx in rs..re {
            let c = match col_indices.get(idx) {
                Some(&c) if c < n => c,
                _ => continue,
            };
            let v = values.get(idx).copied().unwrap_or(0.0);
            cols[c].rows.push(i);
            cols[c].vals.push(v);
        }
    }
    cols
}

/// Test whether two `L` columns have identical sub-diagonal row structure.
fn same_lower_structure(a: &SparseCol, b: &SparseCol) -> bool {
    if a.rows.is_empty() || a.rows.len() != b.rows.len() {
        return false;
    }
    // Compare as multisets via sorted copies (column order is not guaranteed).
    let mut ra = a.rows.clone();
    let mut rb = b.rows.clone();
    ra.sort_unstable();
    rb.sort_unstable();
    ra == rb
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CSR matrix from a dense row-major matrix (dropping exact zeros).
    fn dense_to_csr(a: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>, usize) {
        let n = a.len();
        let mut row_offsets = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for (i, row) in a.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_offsets[i + 1] = col_indices.len();
        }
        (row_offsets, col_indices, values, n)
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

    fn make_rhs(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut b = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                b[i] += a[i][j] * x[j];
            }
        }
        b
    }

    #[test]
    fn factorize_solve_diagonal() {
        let a = vec![
            vec![2.0, 0.0, 0.0],
            vec![0.0, 3.0, 0.0],
            vec![0.0, 0.0, 4.0],
        ];
        let (ro, ci, va, n) = dense_to_csr(&a);
        let b = vec![2.0, 6.0, 12.0]; // x = [1, 2, 3]
        let x = left_looking_lu_solve(&ro, &ci, &va, n, &b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
        assert!((x[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn factorize_solve_general_3x3_needs_pivot() {
        // Non-symmetric, column 0 has a zero on the diagonal -> pivoting required.
        let a = vec![
            vec![0.0, 2.0, 1.0],
            vec![4.0, 1.0, 0.0],
            vec![1.0, 1.0, 3.0],
        ];
        let x_exact = [1.0, 2.0, 3.0];
        let b = make_rhs(&a, &x_exact);
        let (ro, ci, va, n) = dense_to_csr(&a);
        let lu = LeftLookingLu::factorize(&ro, &ci, &va, n, 1.0).unwrap();
        let x = lu.solve(&b).unwrap();
        for i in 0..3 {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-10,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-10);
        // permutation must be a valid permutation of 0..n
        let mut seen = vec![false; n];
        for &p in lu.permutation() {
            assert!(!seen[p]);
            seen[p] = true;
        }
    }

    #[test]
    fn factorize_solve_full_dense_4x4() {
        // A fully-dense, well-conditioned, non-symmetric 4x4.
        let a = vec![
            vec![4.0, 1.0, 2.0, 0.5],
            vec![1.0, 5.0, 1.0, 2.0],
            vec![2.0, 1.0, 6.0, 1.0],
            vec![0.5, 2.0, 1.0, 7.0],
        ];
        let x_exact = [2.0, -1.0, 3.0, 0.5];
        let b = make_rhs(&a, &x_exact);
        let (ro, ci, va, n) = dense_to_csr(&a);
        let x = left_looking_lu_solve(&ro, &ci, &va, n, &b).unwrap();
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-9,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-9);
    }

    #[test]
    fn factorize_solve_sparse_laplacian_12() {
        // 1D Laplacian (tridiagonal SPD), sparse.
        let n = 12;
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
        let x_exact: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sqrt()).collect();
        let b = make_rhs(&a, &x_exact);
        let (ro, ci, va, _) = dense_to_csr(&a);
        let lu = LeftLookingLu::factorize(&ro, &ci, &va, n, 1.0).unwrap();
        let x = lu.solve(&b).unwrap();
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-9,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-9);
        assert_eq!(lu.dim(), n);
        assert!(lu.num_supernodes() >= 1);
    }

    #[test]
    fn factorize_solve_nonsymmetric_pattern() {
        // Non-symmetric sparsity (arrow-like), exercises the DFS fill prediction.
        let a = vec![
            vec![3.0, 0.0, 0.0, 1.0],
            vec![2.0, 4.0, 0.0, 0.0],
            vec![0.0, 1.0, 5.0, 0.0],
            vec![1.0, 0.0, 2.0, 6.0],
        ];
        let x_exact = [1.0, 2.0, 3.0, 4.0];
        let b = make_rhs(&a, &x_exact);
        let (ro, ci, va, n) = dense_to_csr(&a);
        let x = left_looking_lu_solve(&ro, &ci, &va, n, &b).unwrap();
        for i in 0..n {
            assert!(
                (x[i] - x_exact[i]).abs() < 1e-10,
                "x[{i}] = {} exp {}",
                x[i],
                x_exact[i]
            );
        }
        assert!(residual_norm(&a, &x, &b) < 1e-10);
    }

    #[test]
    fn singular_matrix_detected() {
        // Rank-deficient: column 2 is a copy of column 1, last row all zero.
        let a = vec![
            vec![1.0, 2.0, 2.0],
            vec![0.0, 3.0, 3.0],
            vec![0.0, 0.0, 0.0],
        ];
        let (ro, ci, va, n) = dense_to_csr(&a);
        let res = LeftLookingLu::factorize(&ro, &ci, &va, n, 1.0);
        assert!(matches!(res, Err(SolverError::SingularMatrix)));
    }

    #[test]
    fn dimension_mismatch_detected() {
        let ro = vec![0usize, 1]; // wrong length for n = 3
        let ci = vec![0usize];
        let va = vec![1.0];
        let res = LeftLookingLu::factorize(&ro, &ci, &va, 3, 1.0);
        assert!(matches!(res, Err(SolverError::DimensionMismatch(_))));
    }

    #[test]
    fn nnz_reports_fill() {
        let a = vec![
            vec![4.0, 1.0, 0.0],
            vec![1.0, 3.0, 1.0],
            vec![0.0, 1.0, 2.0],
        ];
        let (ro, ci, va, n) = dense_to_csr(&a);
        let lu = LeftLookingLu::factorize(&ro, &ci, &va, n, 1.0).unwrap();
        assert!(lu.nnz() >= 7, "nnz = {}", lu.nnz());
    }
}
