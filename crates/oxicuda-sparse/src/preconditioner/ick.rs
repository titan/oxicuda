//! Incomplete Cholesky factorization with level-of-fill -- IC(k).
//!
//! For a symmetric positive-definite (SPD) matrix `A` this computes a sparse
//! lower-triangular factor `L` such that `L Lᵀ ≈ A`, where `L` is allowed fill
//! entries up to level `k` away from an original non-zero. IC(0) keeps only the
//! original lower-triangle positions; larger `k` admits more fill, producing a
//! more accurate (and, when `k ≥ n`, an exact) factorization.
//!
//! ## Symbolic phase
//!
//! The fill pattern is determined with the same level-of-fill recurrence as the
//! crate's ILU(k) ([`iluk`](crate::preconditioner::iluk)): an original entry has
//! level 0, and a fill entry `(i, j)` receives level
//! `min over m ( level(i, m) + level(m, j) + 1 )`. Because `A` is symmetric the
//! filled graph is symmetric, so the pattern of `L` is exactly the lower
//! triangle of the symbolic ILU(k) pattern of `A`. With `k ≥ n` this reproduces
//! the *complete* Cholesky fill and the factorization is exact.
//!
//! ## Numeric phase
//!
//! A left-looking Cholesky restricted to the symbolic pattern: for each column
//! `j`, `L[j, j] = sqrt(A[j, j] − Σ_m L[j, m]²)` and
//! `L[i, j] = (A[i, j] − Σ_m L[i, m] L[j, m]) / L[j, j]` for `i > j` in the
//! pattern, where the sums run over columns `m < j` common to rows `i` and `j`.
//! A non-positive pivot signals that the (modified) matrix is not SPD.

use crate::error::{SparseError, SparseResult};
use crate::host_csr::HostCsr;

/// An IC(k) factorization: the lower-triangular Cholesky factor `L` (including
/// its diagonal) stored in host CSR layout, together with the fill level used.
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyK {
    /// Lower-triangular factor `L` (with explicit diagonal), CSR layout.
    l: HostCsr,
    /// The level of fill `k` that produced this factor.
    level: usize,
}

impl IncompleteCholeskyK {
    /// Returns a reference to the lower-triangular factor `L`.
    #[inline]
    pub fn l_factor(&self) -> &HostCsr {
        &self.l
    }

    /// Returns the level of fill `k`.
    #[inline]
    pub fn level(&self) -> usize {
        self.level
    }

    /// Applies the preconditioner: solves `L Lᵀ z = r` and returns `z`.
    ///
    /// This performs a forward substitution `L y = r` followed by a backward
    /// substitution `Lᵀ z = y`. For the *complete* factor this yields the exact
    /// solution `z = A⁻¹ r`.
    ///
    /// # Panics
    ///
    /// Never panics; if `r` has the wrong length the surplus/missing entries
    /// are treated as zero through the bounds of `L`.
    pub fn apply(&self, r: &[f64]) -> Vec<f64> {
        let n = self.l.nrows;
        let mut y = vec![0.0f64; n];

        // Forward solve L y = r (L lower-triangular with explicit diagonal).
        for i in 0..n {
            let start = self.l.row_ptr[i];
            let end = self.l.row_ptr[i + 1];
            let mut sum = if i < r.len() { r[i] } else { 0.0 };
            let mut diag = 1.0;
            for kk in start..end {
                let j = self.l.col_indices[kk];
                if j < i {
                    sum -= self.l.values[kk] * y[j];
                } else if j == i {
                    diag = self.l.values[kk];
                }
            }
            y[i] = sum / diag;
        }

        // Backward solve Lᵀ z = y. Iterate rows from the bottom; row i of L
        // contributes its off-diagonal entries to the already-solved unknowns.
        let mut z = vec![0.0f64; n];
        for i in (0..n).rev() {
            let start = self.l.row_ptr[i];
            let end = self.l.row_ptr[i + 1];
            let mut diag = 1.0;
            for kk in start..end {
                if self.l.col_indices[kk] == i {
                    diag = self.l.values[kk];
                }
            }
            z[i] = y[i] / diag;
            // Subtract this unknown's contribution from earlier rows.
            for kk in start..end {
                let j = self.l.col_indices[kk];
                if j < i {
                    y[j] -= self.l.values[kk] * z[i];
                }
            }
        }
        z
    }
}

/// Computes the IC(k) factorization of an SPD host CSR matrix.
///
/// # Arguments
///
/// * `a` -- a square SPD matrix in host CSR layout. Both triangles may be
///   stored; only the symmetric structure is used for the pattern and the
///   stored entries are read for the numeric values.
/// * `k` -- the level of fill (`0` = IC(0); large `k` gives complete Cholesky).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square,
/// [`SparseError::InvalidArgument`] if `a` is empty, or
/// [`SparseError::SingularMatrix`] if a non-positive pivot is encountered
/// (i.e. the matrix is not SPD on the retained pattern).
pub fn ic_k(a: &HostCsr, k: usize) -> SparseResult<IncompleteCholeskyK> {
    if a.nrows != a.ncols {
        return Err(SparseError::DimensionMismatch(format!(
            "IC(k) requires a square matrix, got {}x{}",
            a.nrows, a.ncols
        )));
    }
    let n = a.nrows;
    if n == 0 {
        return Err(SparseError::InvalidArgument(
            "cannot factor an empty matrix".to_string(),
        ));
    }

    // Symbolic phase: full symmetric level-of-fill pattern, then keep lower tri.
    let lower_pattern = ic_k_symbolic(a, k);

    // Numeric phase: left-looking Cholesky on the lower-triangular pattern.
    let values = ic_k_numeric(a, &lower_pattern, n)?;

    let mut row_ptr = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut out_values = Vec::new();
    for i in 0..n {
        for &(col, val) in &lower_pattern[i] {
            col_indices.push(col);
            out_values.push(values[&(i, col)]);
            let _ = val;
        }
        row_ptr[i + 1] = col_indices.len();
    }

    let l = HostCsr::new(n, n, row_ptr, col_indices, out_values)?;
    Ok(IncompleteCholeskyK { l, level: k })
}

/// Sparse-row entry carrying a level-of-fill annotation.
#[derive(Clone, Copy)]
struct LevEntry {
    col: usize,
    level: usize,
}

/// Computes the lower-triangular fill pattern of `L` with level-of-fill `k`.
///
/// Returns, for each row `i`, a sorted list of `(col, level)` pairs with
/// `col ≤ i` (the explicit diagonal is always present). The recurrence mirrors
/// [`iluk`](crate::preconditioner::iluk) but is applied to the symmetric graph,
/// so that with `k ≥ n` the result is the exact Cholesky fill.
fn ic_k_symbolic(a: &HostCsr, k: usize) -> Vec<Vec<(usize, usize)>> {
    let n = a.nrows;

    // Build the symmetric adjacency with level 0 for every structural entry of
    // A and its transpose (so the pattern is symmetric even if A stored only
    // one triangle). The diagonal is forced present.
    let mut rows: Vec<Vec<LevEntry>> = Vec::with_capacity(n);
    {
        let mut sets: Vec<std::collections::BTreeMap<usize, usize>> =
            vec![std::collections::BTreeMap::new(); n];
        for i in 0..n {
            let start = a.row_ptr[i];
            let end = a.row_ptr[i + 1];
            sets[i].insert(i, 0);
            for kk in start..end {
                let j = a.col_indices[kk];
                sets[i].insert(j, 0);
                sets[j].insert(i, 0);
            }
        }
        for set in sets {
            rows.push(
                set.into_iter()
                    .map(|(col, level)| LevEntry { col, level })
                    .collect(),
            );
        }
    }

    // Symmetric level-of-fill elimination, processing rows in order. For row i,
    // for each lower neighbour m < i with level lev_im, pull in the upper part
    // of row m (cols j with m < j, i.e. the transpose entries) to generate
    // fill (i, j).
    for i in 0..n {
        let mut idx = 0;
        loop {
            if idx >= rows[i].len() {
                break;
            }
            let m = rows[i][idx].col;
            if m >= i {
                break;
            }
            let lev_im = rows[i][idx].level;

            // Entries of row m with column j > m (the "upper" structure of the
            // factor at row m). By symmetry these are exactly the lower
            // neighbours of those j through m.
            let upper_m: Vec<(usize, usize)> = rows[m]
                .iter()
                .filter(|e| e.col > m)
                .map(|e| (e.col, e.level))
                .collect();

            for (j, lev_mj) in upper_m {
                if j >= i {
                    // Only fill the lower triangle of row i (cols < i) plus the
                    // diagonal; columns >= i are handled when their own row is
                    // processed via symmetry.
                    if j != i {
                        continue;
                    }
                }
                let new_level = lev_im + lev_mj + 1;
                if new_level > k {
                    continue;
                }
                match rows[i].iter().position(|e| e.col == j) {
                    Some(pos) => {
                        if new_level < rows[i][pos].level {
                            rows[i][pos].level = new_level;
                        }
                    }
                    None => {
                        let insert_pos = rows[i]
                            .iter()
                            .position(|e| e.col > j)
                            .unwrap_or(rows[i].len());
                        rows[i].insert(
                            insert_pos,
                            LevEntry {
                                col: j,
                                level: new_level,
                            },
                        );
                    }
                }
            }
            idx += 1;
        }
    }

    // Keep only the lower triangle (cols <= i) for L.
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| {
            row.into_iter()
                .filter(|e| e.col <= i)
                .map(|e| (e.col, e.level))
                .collect()
        })
        .collect()
}

/// Left-looking numeric Cholesky on the given lower-triangular pattern.
///
/// Returns a map from `(row, col)` to the computed value of `L[row, col]`.
fn ic_k_numeric(
    a: &HostCsr,
    pattern: &[Vec<(usize, usize)>],
    n: usize,
) -> SparseResult<std::collections::HashMap<(usize, usize), f64>> {
    // Dense column views of L are unnecessary; we store the computed entries in
    // a map and, for each row, a sorted column list to intersect rows i and j.
    let mut l: std::collections::HashMap<(usize, usize), f64> = std::collections::HashMap::new();

    // Pre-extract sorted column lists per row for fast intersection.
    let row_cols: Vec<Vec<usize>> = pattern
        .iter()
        .map(|row| row.iter().map(|&(c, _)| c).collect())
        .collect();

    for i in 0..n {
        for &(j, _lev) in &pattern[i] {
            if j > i {
                continue;
            }
            // Dot product Σ_{m < j} L[i,m] L[j,m] over common columns m.
            let mut sum = a.get(i, j).unwrap_or(0.0);
            let ci = &row_cols[i];
            let cj = &row_cols[j];
            let (mut pi, mut pj) = (0usize, 0usize);
            while pi < ci.len() && pj < cj.len() {
                let mi = ci[pi];
                let mj = cj[pj];
                if mi >= j || mj >= j {
                    break;
                }
                match mi.cmp(&mj) {
                    std::cmp::Ordering::Less => pi += 1,
                    std::cmp::Ordering::Greater => pj += 1,
                    std::cmp::Ordering::Equal => {
                        let lim = l.get(&(i, mi)).copied().unwrap_or(0.0);
                        let ljm = l.get(&(j, mj)).copied().unwrap_or(0.0);
                        sum -= lim * ljm;
                        pi += 1;
                        pj += 1;
                    }
                }
            }

            if i == j {
                if sum <= 0.0 {
                    return Err(SparseError::SingularMatrix);
                }
                l.insert((i, j), sum.sqrt());
            } else {
                let ljj = l.get(&(j, j)).copied().unwrap_or(0.0);
                if ljj == 0.0 {
                    return Err(SparseError::SingularMatrix);
                }
                l.insert((i, j), sum / ljj);
            }
        }
    }

    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_csr::{HostCsr, laplacian_1d, laplacian_2d};

    /// A small dense-ish SPD matrix (Gram matrix of a random-ish basis) used to
    /// exercise complete-fill exactness.
    fn spd_dense_like(n: usize) -> HostCsr {
        // Build B (n x n) deterministically, then A = Bᵀ B + n*I (SPD, dense).
        let mut b = vec![0.0f64; n * n];
        let mut state: u64 = 12345;
        for v in b.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((state >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
        }
        let mut dense = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for m in 0..n {
                    acc += b[m * n + i] * b[m * n + j];
                }
                if i == j {
                    acc += n as f64;
                }
                dense[i * n + j] = acc;
            }
        }
        dense_to_csr(&dense, n)
    }

    fn dense_to_csr(dense: &[f64], n: usize) -> HostCsr {
        let mut row_ptr = vec![0usize; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let v = dense[i * n + j];
                if v != 0.0 {
                    col_indices.push(j);
                    values.push(v);
                }
            }
            row_ptr[i + 1] = col_indices.len();
        }
        HostCsr::new(n, n, row_ptr, col_indices, values).expect("valid")
    }

    /// Reconstruct (L Lᵀ)[i,j] for a host-CSR lower factor.
    fn llt(l: &HostCsr, i: usize, j: usize) -> f64 {
        // Row i and row j of L; dot over common columns.
        let ci_s = l.row_ptr[i];
        let ci_e = l.row_ptr[i + 1];
        let cj_s = l.row_ptr[j];
        let cj_e = l.row_ptr[j + 1];
        let mut acc = 0.0;
        let (mut pi, mut pj) = (ci_s, cj_s);
        while pi < ci_e && pj < cj_e {
            let a = l.col_indices[pi];
            let b = l.col_indices[pj];
            match a.cmp(&b) {
                std::cmp::Ordering::Less => pi += 1,
                std::cmp::Ordering::Greater => pj += 1,
                std::cmp::Ordering::Equal => {
                    acc += l.values[pi] * l.values[pj];
                    pi += 1;
                    pj += 1;
                }
            }
        }
        acc
    }

    #[test]
    fn llt_matches_a_on_pattern_ic0() {
        // For IC(0) on the 1D Laplacian, (LLᵀ)[i,j] == A[i,j] for (i,j) in the
        // pattern -- the defining property of incomplete Cholesky.
        let a = laplacian_1d(8);
        let fac = ic_k(&a, 0).expect("ic0");
        let l = fac.l_factor();
        for i in 0..l.nrows {
            let s = l.row_ptr[i];
            let e = l.row_ptr[i + 1];
            for kk in s..e {
                let j = l.col_indices[kk];
                let recon = llt(l, i, j);
                let aij = a.get(i, j).unwrap_or(0.0);
                assert!(
                    (recon - aij).abs() < 1e-12,
                    "IC(0) pattern mismatch at ({i},{j}): {recon} vs {aij}"
                );
            }
        }
    }

    #[test]
    fn fill_pattern_monotone() {
        // IC(0) ⊆ IC(1) ⊆ IC(2) on a 2D Laplacian (non-trivial fill).
        let a = laplacian_2d(5, 5);
        let p0 = ic_k_symbolic(&a, 0);
        let p1 = ic_k_symbolic(&a, 1);
        let p2 = ic_k_symbolic(&a, 2);
        for i in 0..a.nrows {
            let s0: std::collections::HashSet<usize> = p0[i].iter().map(|&(c, _)| c).collect();
            let s1: std::collections::HashSet<usize> = p1[i].iter().map(|&(c, _)| c).collect();
            let s2: std::collections::HashSet<usize> = p2[i].iter().map(|&(c, _)| c).collect();
            assert!(s0.is_subset(&s1), "IC(0) not subset of IC(1) at row {i}");
            assert!(s1.is_subset(&s2), "IC(1) not subset of IC(2) at row {i}");
        }
    }

    #[test]
    fn complete_fill_reconstructs_exactly() {
        // Large k => complete Cholesky => L Lᵀ == A exactly (dense compare).
        let n = 10;
        let a = spd_dense_like(n);
        let fac = ic_k(&a, n + 5).expect("complete cholesky");
        let l = fac.l_factor();
        for i in 0..n {
            for j in 0..n {
                let recon = llt(l, i, j);
                let aij = a.get(i, j).unwrap_or(0.0);
                assert!(
                    (recon - aij).abs() < 1e-9,
                    "complete reconstruction mismatch at ({i},{j}): {recon} vs {aij}"
                );
            }
        }
    }

    #[test]
    fn apply_is_exact_inverse_for_complete_factor() {
        // For the complete factor, apply(r) == A⁻¹ r, hence A·apply(r) ≈ r.
        let n = 9;
        let a = spd_dense_like(n);
        let fac = ic_k(&a, n + 5).expect("complete");
        let r: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.3).collect();
        let z = fac.apply(&r);
        let az = a.matvec(&z);
        for i in 0..n {
            assert!(
                (az[i] - r[i]).abs() < 1e-8,
                "A·apply(r) != r at {i}: {} vs {}",
                az[i],
                r[i]
            );
        }
    }

    #[test]
    fn apply_solves_laplacian_with_complete_fill() {
        // Complete factor of the 1D Laplacian solves L Lᵀ z = r exactly.
        let n = 12;
        let a = laplacian_1d(n);
        let fac = ic_k(&a, n + 1).expect("complete");
        let r = vec![1.0f64; n];
        let z = fac.apply(&r);
        let az = a.matvec(&z);
        for i in 0..n {
            assert!((az[i] - r[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn non_spd_errors() {
        // Indefinite symmetric matrix: [[1, 2], [2, 1]] has eigenvalues 3, -1.
        let a = HostCsr::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 2.0, 1.0],
        )
        .expect("valid");
        assert!(matches!(ic_k(&a, 5), Err(SparseError::SingularMatrix)));
    }

    #[test]
    fn rejects_non_square() {
        let a = HostCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("valid");
        assert!(matches!(
            ic_k(&a, 0),
            Err(SparseError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn ic0_pattern_equals_lower_triangle() {
        // IC(0) keeps exactly the lower triangle of A's structure.
        let a = laplacian_2d(4, 4);
        let p0 = ic_k_symbolic(&a, 0);
        for (i, prow) in p0.iter().enumerate() {
            let cols: Vec<usize> = prow.iter().map(|&(c, _)| c).collect();
            // Every original lower entry must be present, and no extra fill.
            let mut expected: Vec<usize> = Vec::new();
            let s = a.row_ptr[i];
            let e = a.row_ptr[i + 1];
            for kk in s..e {
                let j = a.col_indices[kk];
                if j <= i {
                    expected.push(j);
                }
            }
            if !expected.contains(&i) {
                expected.push(i);
            }
            expected.sort_unstable();
            assert_eq!(cols, expected, "row {i}");
        }
    }
}
