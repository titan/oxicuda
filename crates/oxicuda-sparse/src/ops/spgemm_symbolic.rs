//! Symbolic SpGEMM (Gustavson) -- the sparsity pattern of `C = A * B`.
//!
//! This module computes **only** the output sparsity structure (the `row_ptr`
//! array and the sorted column indices of each row) of the sparse-sparse
//! product `C = A * B`, without evaluating any numeric values. Knowing the
//! pattern ahead of time lets a caller pre-allocate the result and run the
//! numeric phase without reallocation -- the standard two-phase SpGEMM strategy
//! and the foundation of fixed-pattern iterative schemes (repeated Galerkin
//! products, polynomial preconditioners, graph powers).
//!
//! ## Algorithm
//!
//! The implementation follows Gustavson's 1978 row-wise formulation. For each
//! row `i` of `A`, every non-zero `A[i, p]` selects row `p` of `B`; the union
//! of the column indices over all such `B` rows is exactly the set of
//! structurally non-zero columns of `C[i, :]`. Uniqueness within the row is
//! tracked by a single integer **mask** array indexed by column: `mask[c]`
//! holds the index of the row that most recently inserted column `c`, so a
//! membership test and insertion are both `O(1)` with no hashing and no
//! per-row reinitialisation (the row index itself is the generation marker).
//! Each completed row is sorted ascending to match the crate's CSR convention.
//!
//! The result is the **structural** pattern. It equals the numeric non-zero
//! pattern of `A * B` exactly whenever no numeric cancellation occurs (e.g. for
//! sign-definite operands); with cancellation the structural pattern is a
//! superset, as is true of every symbolic SpGEMM.

use crate::error::{SparseError, SparseResult};
use crate::host_csr::HostCsr;

/// The sparsity pattern (structure) of a sparse matrix in CSR layout.
///
/// Carries no numeric values -- only the row pointers and the column indices.
/// Column indices are sorted ascending within each row, matching [`HostCsr`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolicPattern {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointer array of length `nrows + 1`, monotone non-decreasing, with
    /// `row_ptr[0] == 0` and `row_ptr[nrows] == nnz`.
    pub row_ptr: Vec<usize>,
    /// Column indices of length `nnz`, sorted ascending within each row and all
    /// in the range `[0, ncols)`.
    pub col_indices: Vec<usize>,
}

impl SymbolicPattern {
    /// Number of structural non-zeros (`row_ptr[nrows]`).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// Returns the sorted column indices of structural row `row`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= nrows` (the caller is responsible for the bound, as
    /// with ordinary slice indexing).
    #[inline]
    pub fn row(&self, row: usize) -> &[usize] {
        &self.col_indices[self.row_ptr[row]..self.row_ptr[row + 1]]
    }
}

/// Computes the structural sparsity pattern of `C = A * B` via Gustavson's
/// row-wise symbolic algorithm.
///
/// Only the structure of `A` and `B` is inspected; their stored values are
/// never read. The returned [`SymbolicPattern`] has `A.nrows` rows and
/// `B.ncols` columns, with each row's column indices sorted ascending.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.ncols != B.nrows`.
pub fn spgemm_symbolic_pattern(a: &HostCsr, b: &HostCsr) -> SparseResult<SymbolicPattern> {
    if a.ncols != b.nrows {
        return Err(SparseError::DimensionMismatch(format!(
            "A.ncols ({}) != B.nrows ({})",
            a.ncols, b.nrows
        )));
    }

    let out_cols = b.ncols;
    let mut row_ptr = vec![0usize; a.nrows + 1];
    let mut col_indices: Vec<usize> = Vec::new();

    // Gustavson generation mask: `mask[c]` records the row that last inserted
    // column `c`. The sentinel `usize::MAX` marks a column never yet touched,
    // which cannot collide with any real row index in practice.
    let mut mask = vec![usize::MAX; out_cols];
    let mut row_cols: Vec<usize> = Vec::new();

    for i in 0..a.nrows {
        row_cols.clear();
        let a_start = a.row_ptr[i];
        let a_end = a.row_ptr[i + 1];
        for &a_col in &a.col_indices[a_start..a_end] {
            let b_start = b.row_ptr[a_col];
            let b_end = b.row_ptr[a_col + 1];
            for &b_col in &b.col_indices[b_start..b_end] {
                if mask[b_col] != i {
                    mask[b_col] = i;
                    row_cols.push(b_col);
                }
            }
        }
        row_cols.sort_unstable();
        col_indices.extend_from_slice(&row_cols);
        row_ptr[i + 1] = col_indices.len();
    }

    Ok(SymbolicPattern {
        nrows: a.nrows,
        ncols: out_cols,
        row_ptr,
        col_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG for building reproducible random test matrices.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_add(0x9e37_79b9_7f4a_7c15),
            }
        }
        fn next_u32(&mut self) -> u32 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.state >> 32) as u32
        }
        /// Strictly positive value in `(0, 4]` so products never cancel.
        fn next_pos(&mut self) -> f64 {
            1.0 + (self.next_u32() as f64 / u32::MAX as f64) * 3.0
        }
        fn next_bool(&mut self, num: u32, den: u32) -> bool {
            self.next_u32() % den < num
        }
    }

    /// Builds a random `nrows x ncols` matrix with strictly positive entries
    /// (so the numeric product has no cancellation) at the given fill density.
    fn random_positive(nrows: usize, ncols: usize, num: u32, den: u32, seed: u64) -> HostCsr {
        let mut rng = Lcg::new(seed);
        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..nrows {
            for j in 0..ncols {
                if rng.next_bool(num, den) {
                    col_indices.push(j);
                    values.push(rng.next_pos());
                }
            }
            row_ptr[i + 1] = col_indices.len();
        }
        HostCsr::new(nrows, ncols, row_ptr, col_indices, values).expect("valid random csr")
    }

    /// Extracts the numeric non-zero pattern of a host matrix as per-row sets.
    fn numeric_pattern(m: &HostCsr) -> Vec<Vec<usize>> {
        (0..m.nrows)
            .map(|i| m.col_indices[m.row_ptr[i]..m.row_ptr[i + 1]].to_vec())
            .collect()
    }

    /// Asserts the symbolic pattern matches the numeric product exactly.
    fn assert_matches_numeric(a: &HostCsr, b: &HostCsr) {
        let sym = spgemm_symbolic_pattern(a, b).expect("symbolic");
        let numeric = a.matmul(b).expect("numeric");

        // row_ptr must agree exactly (so nnz per row and total nnz agree).
        assert_eq!(sym.row_ptr, numeric.row_ptr, "row_ptr mismatch");
        assert_eq!(sym.nnz(), numeric.nnz(), "nnz mismatch");

        // Per-row column sets must agree (both are sorted ascending).
        let num_sets = numeric_pattern(&numeric);
        for (i, num_row) in num_sets.iter().enumerate() {
            assert_eq!(sym.row(i), num_row.as_slice(), "row {i} columns differ");
        }

        // Structural invariants.
        assert_eq!(sym.row_ptr[0], 0);
        assert_eq!(*sym.row_ptr.last().expect("nonempty"), sym.nnz());
        for w in sym.row_ptr.windows(2) {
            assert!(w[1] >= w[0], "row_ptr not monotone");
        }
        for &c in &sym.col_indices {
            assert!(c < sym.ncols, "column {c} out of range");
        }
    }

    #[test]
    fn matches_dense_handbuilt() {
        // A = [[1,2,0],[0,3,4],[5,0,6]] ; B = [[7,0],[0,8],[9,0]]  (all positive)
        let a = HostCsr::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("a");
        let b =
            HostCsr::new(3, 2, vec![0, 1, 2, 3], vec![0, 1, 0], vec![7.0, 8.0, 9.0]).expect("b");
        assert_matches_numeric(&a, &b);

        // Spot-check the exact expected pattern.
        let sym = spgemm_symbolic_pattern(&a, &b).expect("sym");
        assert_eq!(sym.row_ptr, vec![0, 2, 4, 5]);
        assert_eq!(sym.row(0), &[0, 1]);
        assert_eq!(sym.row(1), &[0, 1]);
        assert_eq!(sym.row(2), &[0]); // row 2 -> only column 0 is structurally hit
    }

    #[test]
    fn identity_preserves_pattern() {
        let a = random_positive(6, 6, 1, 3, 0xabc);
        let eye = HostCsr::new(
            6,
            6,
            vec![0, 1, 2, 3, 4, 5, 6],
            vec![0, 1, 2, 3, 4, 5],
            vec![1.0; 6],
        )
        .expect("eye");
        // A * I and I * A both reproduce A's structure.
        assert_matches_numeric(&a, &eye);
        assert_matches_numeric(&eye, &a);
    }

    #[test]
    fn handles_empty_rows() {
        // Row 1 of A is empty -> row 1 of C must be empty.
        let a = HostCsr::new(3, 3, vec![0, 1, 1, 2], vec![2, 0], vec![1.0, 1.0]).expect("a");
        let b = random_positive(3, 4, 1, 2, 0x55);
        let sym = spgemm_symbolic_pattern(&a, &b).expect("sym");
        assert_eq!(sym.row_ptr[1], sym.row_ptr[2], "empty A row -> empty C row");
        assert_matches_numeric(&a, &b);
    }

    #[test]
    fn empty_b_column_block() {
        // B has a fully empty row 0; columns reached only via A[*,0] vanish.
        let a = random_positive(4, 3, 2, 3, 0x99);
        // B rows: 0 empty, 1 -> col 2, 2 -> cols 0,1.
        let b =
            HostCsr::new(3, 3, vec![0, 0, 1, 3], vec![2, 0, 1], vec![1.0, 1.0, 1.0]).expect("b");
        assert_matches_numeric(&a, &b);
    }

    #[test]
    fn random_small_matrices() {
        // Several independent random pairs of compatible dimensions.
        let cases = [
            (5usize, 4usize, 6usize, 1u32, 2u32, 0x1u64),
            (7, 7, 7, 1, 3, 0x2),
            (3, 8, 5, 2, 3, 0x3),
            (10, 6, 9, 1, 4, 0x4),
            (6, 6, 6, 1, 1, 0x5), // dense-ish
        ];
        for (m, k, n, num, den, seed) in cases {
            let a = random_positive(m, k, num, den, seed);
            let b = random_positive(k, n, num, den, seed ^ 0xdead);
            assert_matches_numeric(&a, &b);
        }
    }

    #[test]
    fn dimension_mismatch_errors() {
        let a = random_positive(3, 4, 1, 2, 1);
        let b = random_positive(5, 2, 1, 2, 2); // 5 != 4
        assert!(spgemm_symbolic_pattern(&a, &b).is_err());
    }

    #[test]
    fn nnz_and_row_accessor() {
        let a = random_positive(4, 4, 1, 2, 7);
        let b = random_positive(4, 4, 1, 2, 8);
        let sym = spgemm_symbolic_pattern(&a, &b).expect("sym");
        let total: usize = (0..sym.nrows).map(|i| sym.row(i).len()).sum();
        assert_eq!(total, sym.nnz());
    }
}
