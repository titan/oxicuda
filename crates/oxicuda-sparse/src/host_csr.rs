//! Host-resident CSR matrix and shared CPU kernels for advanced solvers.
//!
//! The public [`CsrMatrix`] stores its arrays in GPU device memory and
//! therefore cannot be constructed or manipulated without a live CUDA context.
//! The classical sparse linear-algebra algorithms in this crate -- incomplete
//! Cholesky with level-of-fill ([`ick`](crate::preconditioner::ick)), the
//! LOBPCG eigensolver ([`lobpcg`](crate::eig::lobpcg::lobpcg)), and
//! smoothed-aggregation algebraic multigrid ([`amg`](crate::preconditioner::amg))
//! -- are inherently iterative host-driven procedures: they repeatedly form
//! Galerkin products, solve small dense subproblems, and run sweeps that
//! interleave CPU control flow with linear-algebra primitives.
//!
//! This module provides [`HostCsr`], a lightweight CPU-side CSR container
//! holding plain `Vec<i32>`/`Vec<f64>` arrays, together with the shared
//! building blocks (SpMV, transpose, sparse-sparse product, triangular solves,
//! and a dense Gaussian-elimination solver) that the three algorithms reuse.
//! A [`HostCsr`] can be lifted from a GPU [`CsrMatrix`] via
//! [`HostCsr::from_gpu`] (which downloads the arrays through `to_host`) and
//! materialised back onto the device through [`HostCsr::to_gpu`].

use oxicuda_blas::GpuFloat;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;

/// A square or rectangular sparse matrix stored on the host in CSR layout.
///
/// Column indices within each row are kept sorted in ascending order, which the
/// numeric kernels rely on for efficient merging. All values are `f64`; the
/// advanced solvers accumulate in double precision regardless of the storage
/// precision of any originating device matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct HostCsr {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointer array of length `nrows + 1`.
    pub row_ptr: Vec<usize>,
    /// Column indices of length `nnz`, sorted ascending within each row.
    pub col_indices: Vec<usize>,
    /// Non-zero values of length `nnz`.
    pub values: Vec<f64>,
}

impl HostCsr {
    /// Builds a host CSR matrix from raw arrays, validating the structure.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::InvalidFormat`] if the array lengths are
    /// inconsistent, `row_ptr` is not monotone, or a column index is out of
    /// range.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> SparseResult<Self> {
        if row_ptr.len() != nrows + 1 {
            return Err(SparseError::InvalidFormat(format!(
                "row_ptr length ({}) must be nrows + 1 ({})",
                row_ptr.len(),
                nrows + 1
            )));
        }
        if col_indices.len() != values.len() {
            return Err(SparseError::InvalidFormat(format!(
                "col_indices length ({}) must equal values length ({})",
                col_indices.len(),
                values.len()
            )));
        }
        if !row_ptr.is_empty() && row_ptr[0] != 0 {
            return Err(SparseError::InvalidFormat(
                "row_ptr[0] must be 0".to_string(),
            ));
        }
        if let Some(&last) = row_ptr.last() {
            if last != values.len() {
                return Err(SparseError::InvalidFormat(format!(
                    "row_ptr[nrows] ({}) must equal nnz ({})",
                    last,
                    values.len()
                )));
            }
        }
        for i in 0..nrows {
            if row_ptr[i] > row_ptr[i + 1] {
                return Err(SparseError::InvalidFormat(
                    "row_ptr must be non-decreasing".to_string(),
                ));
            }
        }
        for &c in &col_indices {
            if c >= ncols {
                return Err(SparseError::InvalidFormat(format!(
                    "column index {c} out of range (ncols = {ncols})"
                )));
            }
        }
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_indices,
            values,
        })
    }

    /// Downloads a GPU CSR matrix into a host CSR matrix.
    ///
    /// The values are converted to `f64`. Column indices within each row are
    /// sorted ascending so the host kernels can merge rows efficiently.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::Cuda`] on a device-to-host transfer failure.
    pub fn from_gpu<T: GpuFloat>(matrix: &CsrMatrix<T>) -> SparseResult<Self> {
        let (h_row_ptr, h_col_idx, h_values) = matrix.to_host()?;
        let nrows = matrix.rows() as usize;
        let ncols = matrix.cols() as usize;

        let mut row_ptr = vec![0usize; nrows + 1];
        let mut col_indices = Vec::with_capacity(h_col_idx.len());
        let mut values = Vec::with_capacity(h_values.len());

        for i in 0..nrows {
            let start = h_row_ptr[i] as usize;
            let end = h_row_ptr[i + 1] as usize;
            // Sort this row's entries by column index.
            let mut entries: Vec<(usize, f64)> = (start..end)
                .map(|k| (h_col_idx[k] as usize, gpu_to_f64(h_values[k])))
                .collect();
            entries.sort_by_key(|&(c, _)| c);
            for (c, v) in entries {
                col_indices.push(c);
                values.push(v);
            }
            row_ptr[i + 1] = col_indices.len();
        }

        Self::new(nrows, ncols, row_ptr, col_indices, values)
    }

    /// Materialises this host matrix onto the device as a GPU CSR matrix.
    ///
    /// Empty matrices are rejected because the device format forbids zero nnz.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::ZeroNnz`] if the matrix has no stored entries, or
    /// [`SparseError::Cuda`] on an allocation/transfer failure.
    pub fn to_gpu<T: GpuFloat>(&self) -> SparseResult<CsrMatrix<T>> {
        if self.values.is_empty() {
            return Err(SparseError::ZeroNnz);
        }
        let row_ptr: Vec<i32> = self.row_ptr.iter().map(|&x| x as i32).collect();
        let col_idx: Vec<i32> = self.col_indices.iter().map(|&x| x as i32).collect();
        let values: Vec<T> = self.values.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        CsrMatrix::from_host(
            self.nrows as u32,
            self.ncols as u32,
            &row_ptr,
            &col_idx,
            &values,
        )
    }

    /// Number of stored non-zeros.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Returns the value at `(row, col)` if present, else `None`.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        // Column indices are sorted; binary search.
        match self.col_indices[start..end].binary_search(&col) {
            Ok(pos) => Some(self.values[start + pos]),
            Err(_) => None,
        }
    }

    /// Extracts the main diagonal as a dense vector of length `min(nrows, ncols)`.
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.nrows.min(self.ncols);
        let mut diag = vec![0.0f64; n];
        for (i, slot) in diag.iter_mut().enumerate() {
            if let Some(v) = self.get(i, i) {
                *slot = v;
            }
        }
        diag
    }

    /// Computes `y = A * x` for a dense vector `x` of length `ncols`.
    ///
    /// The output has length `nrows`. This is the host SpMV reference reused by
    /// every iterative method in the module.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0f64; self.nrows];
        for (i, yi) in y.iter_mut().enumerate() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut acc = 0.0f64;
            for k in start..end {
                acc += self.values[k] * x[self.col_indices[k]];
            }
            *yi = acc;
        }
        y
    }

    /// Returns the transpose `Aᵀ` as a new host CSR matrix.
    ///
    /// Used to form the restriction operator `R = Pᵀ` in algebraic multigrid.
    pub fn transpose(&self) -> HostCsr {
        let mut col_counts = vec![0usize; self.ncols];
        for &c in &self.col_indices {
            col_counts[c] += 1;
        }
        let mut t_row_ptr = vec![0usize; self.ncols + 1];
        for j in 0..self.ncols {
            t_row_ptr[j + 1] = t_row_ptr[j] + col_counts[j];
        }
        let nnz = self.values.len();
        let mut t_col_indices = vec![0usize; nnz];
        let mut t_values = vec![0.0f64; nnz];
        let mut write_pos = t_row_ptr.clone();
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                let c = self.col_indices[k];
                let dest = write_pos[c];
                t_col_indices[dest] = i;
                t_values[dest] = self.values[k];
                write_pos[c] += 1;
            }
        }
        HostCsr {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr: t_row_ptr,
            col_indices: t_col_indices,
            values: t_values,
        }
    }

    /// Computes the sparse-sparse product `C = A * B`.
    ///
    /// This is the host counterpart of the crate's GPU SpGEMM (symbolic +
    /// numeric phases) and is the workhorse of the algebraic-multigrid Galerkin
    /// triple product `Pᵀ A P`. The implementation uses a per-row dense
    /// accumulator (sparse accumulator / SPA) with a touched-column list, which
    /// is the standard Gustavson formulation.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `self.ncols != rhs.nrows`.
    pub fn matmul(&self, rhs: &HostCsr) -> SparseResult<HostCsr> {
        if self.ncols != rhs.nrows {
            return Err(SparseError::DimensionMismatch(format!(
                "A.ncols ({}) != B.nrows ({})",
                self.ncols, rhs.nrows
            )));
        }
        let out_cols = rhs.ncols;
        let mut c_row_ptr = vec![0usize; self.nrows + 1];
        let mut c_col_indices: Vec<usize> = Vec::new();
        let mut c_values: Vec<f64> = Vec::new();

        // Dense accumulator across columns of the result, plus a marker array
        // recording which columns are currently active in the row being built.
        let mut accum = vec![0.0f64; out_cols];
        let mut touched = vec![false; out_cols];
        let mut touched_cols: Vec<usize> = Vec::new();

        for i in 0..self.nrows {
            let a_start = self.row_ptr[i];
            let a_end = self.row_ptr[i + 1];
            for ak in a_start..a_end {
                let a_col = self.col_indices[ak];
                let a_val = self.values[ak];
                let b_start = rhs.row_ptr[a_col];
                let b_end = rhs.row_ptr[a_col + 1];
                for bk in b_start..b_end {
                    let b_col = rhs.col_indices[bk];
                    if !touched[b_col] {
                        touched[b_col] = true;
                        touched_cols.push(b_col);
                    }
                    accum[b_col] += a_val * rhs.values[bk];
                }
            }
            // Emit the accumulated row in sorted column order, dropping exact
            // zeros that arose from cancellation.
            touched_cols.sort_unstable();
            for &col in &touched_cols {
                let v = accum[col];
                if v != 0.0 {
                    c_col_indices.push(col);
                    c_values.push(v);
                }
                accum[col] = 0.0;
                touched[col] = false;
            }
            touched_cols.clear();
            c_row_ptr[i + 1] = c_col_indices.len();
        }

        Ok(HostCsr {
            nrows: self.nrows,
            ncols: out_cols,
            row_ptr: c_row_ptr,
            col_indices: c_col_indices,
            values: c_values,
        })
    }

    /// Converts the matrix to a dense row-major `nrows × ncols` array.
    ///
    /// Intended for small matrices (coarse-level operators, test assertions).
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0f64; self.nrows * self.ncols];
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for k in start..end {
                dense[i * self.ncols + self.col_indices[k]] = self.values[k];
            }
        }
        dense
    }
}

/// Solves the dense linear system `A x = b` via Gaussian elimination with
/// partial pivoting. `a` is row-major `n × n`; `b` has length `n`.
///
/// Used as the direct coarse-grid solver in algebraic multigrid.
///
/// # Errors
///
/// Returns [`SparseError::SingularMatrix`] if a pivot is numerically zero.
pub fn dense_solve(a: &[f64], b: &[f64], n: usize) -> SparseResult<Vec<f64>> {
    let mut m = a.to_vec();
    let mut rhs = b.to_vec();

    for col in 0..n {
        // Partial pivot: find the largest magnitude entry in this column.
        let mut pivot_row = col;
        let mut pivot_mag = m[col * n + col].abs();
        for r in (col + 1)..n {
            let mag = m[r * n + col].abs();
            if mag > pivot_mag {
                pivot_mag = mag;
                pivot_row = r;
            }
        }
        if pivot_mag < 1e-300 {
            return Err(SparseError::SingularMatrix);
        }
        if pivot_row != col {
            for c in 0..n {
                m.swap(col * n + c, pivot_row * n + c);
            }
            rhs.swap(col, pivot_row);
        }
        let pivot = m[col * n + col];
        for r in (col + 1)..n {
            let factor = m[r * n + col] / pivot;
            if factor != 0.0 {
                for c in col..n {
                    m[r * n + c] -= factor * m[col * n + c];
                }
                rhs[r] -= factor * rhs[col];
            }
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for col in (0..n).rev() {
        let mut acc = rhs[col];
        for c in (col + 1)..n {
            acc -= m[col * n + c] * x[c];
        }
        x[col] = acc / m[col * n + col];
    }
    Ok(x)
}

/// Converts a [`GpuFloat`] storage value to `f64` using the crate's bit-cast
/// convention (matching the conversion helpers in the IC(0)/ILU(k) modules).
#[inline]
pub fn gpu_to_f64<T: GpuFloat>(v: T) -> f64 {
    if T::SIZE == 4 {
        f64::from(f32::from_bits(v.to_bits_u64() as u32))
    } else {
        f64::from_bits(v.to_bits_u64())
    }
}

/// Converts an `f64` to a [`GpuFloat`] storage value.
#[inline]
pub fn f64_to_gpu<T: GpuFloat>(v: f64) -> T {
    if T::SIZE == 4 {
        T::from_bits_u64(u64::from((v as f32).to_bits()))
    } else {
        T::from_bits_u64(v.to_bits())
    }
}

/// Builds the 1-D Laplacian `tridiag(-1, 2, -1)` of order `n` as a host CSR
/// matrix. Shared test helper for the SPD-matrix algorithms.
#[cfg(test)]
pub(crate) fn laplacian_1d(n: usize) -> HostCsr {
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    for i in 0..n {
        if i > 0 {
            col_indices.push(i - 1);
            values.push(-1.0);
        }
        col_indices.push(i);
        values.push(2.0);
        if i + 1 < n {
            col_indices.push(i + 1);
            values.push(-1.0);
        }
        row_ptr[i + 1] = col_indices.len();
    }
    HostCsr {
        nrows: n,
        ncols: n,
        row_ptr,
        col_indices,
        values,
    }
}

/// Builds the 2-D 5-point Laplacian on a `gx × gy` grid as a host CSR matrix
/// of order `gx * gy`. Shared test helper.
#[cfg(test)]
pub(crate) fn laplacian_2d(gx: usize, gy: usize) -> HostCsr {
    let n = gx * gy;
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();
    let idx = |x: usize, y: usize| -> usize { y * gx + x };
    for y in 0..gy {
        for x in 0..gx {
            // Collect neighbour entries, then sort by column index.
            let mut entries: Vec<(usize, f64)> = Vec::new();
            entries.push((idx(x, y), 4.0));
            if x > 0 {
                entries.push((idx(x - 1, y), -1.0));
            }
            if x + 1 < gx {
                entries.push((idx(x + 1, y), -1.0));
            }
            if y > 0 {
                entries.push((idx(x, y - 1), -1.0));
            }
            if y + 1 < gy {
                entries.push((idx(x, y + 1), -1.0));
            }
            entries.sort_by_key(|&(c, _)| c);
            for (c, v) in entries {
                col_indices.push(c);
                values.push(v);
            }
            row_ptr[idx(x, y) + 1] = col_indices.len();
        }
    }
    HostCsr {
        nrows: n,
        ncols: n,
        row_ptr,
        col_indices,
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_validates_row_ptr_length() {
        let r = HostCsr::new(2, 2, vec![0, 1], vec![0], vec![1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn new_validates_col_range() {
        let r = HostCsr::new(2, 2, vec![0, 1, 2], vec![0, 5], vec![1.0, 2.0]);
        assert!(r.is_err());
    }

    #[test]
    fn diagonal_extraction() {
        let a = laplacian_1d(4);
        assert_eq!(a.diagonal(), vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn get_returns_entries() {
        let a = laplacian_1d(3);
        assert_eq!(a.get(0, 0), Some(2.0));
        assert_eq!(a.get(0, 1), Some(-1.0));
        assert_eq!(a.get(0, 2), None);
        assert_eq!(a.get(1, 0), Some(-1.0));
    }

    #[test]
    fn matvec_laplacian() {
        let a = laplacian_1d(4);
        // A * [1,1,1,1] = [1, 0, 0, 1]
        let y = a.matvec(&[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(y, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn transpose_of_symmetric_is_self() {
        let a = laplacian_1d(5);
        let at = a.transpose();
        assert_eq!(at.nrows, a.nrows);
        assert_eq!(at.ncols, a.ncols);
        // Symmetric: A == Aᵀ entrywise.
        for i in 0..a.nrows {
            for j in 0..a.ncols {
                assert_eq!(a.get(i, j), at.get(i, j));
            }
        }
    }

    #[test]
    fn transpose_rectangular() {
        // 2x3 matrix:
        // [1 0 2]
        // [0 3 0]
        let a = HostCsr::new(2, 3, vec![0, 2, 3], vec![0, 2, 1], vec![1.0, 2.0, 3.0])
            .expect("valid csr");
        let at = a.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        assert_eq!(at.get(0, 0), Some(1.0));
        assert_eq!(at.get(2, 0), Some(2.0));
        assert_eq!(at.get(1, 1), Some(3.0));
    }

    #[test]
    fn matmul_identity() {
        // I * A == A
        let a = laplacian_1d(4);
        let eye = HostCsr::new(
            4,
            4,
            vec![0, 1, 2, 3, 4],
            vec![0, 1, 2, 3],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid csr");
        let c = eye.matmul(&a).expect("matmul");
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(c.get(i, j), a.get(i, j));
            }
        }
    }

    #[test]
    fn matmul_matches_dense() {
        // A = [[1,2,0],[0,3,4],[5,0,6]] ; B = [[7,0],[0,8],[9,0]]
        let a = HostCsr::new(
            3,
            3,
            vec![0, 2, 4, 6],
            vec![0, 1, 1, 2, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("valid");
        let b = HostCsr::new(3, 2, vec![0, 1, 2, 3], vec![0, 1, 0], vec![7.0, 8.0, 9.0])
            .expect("valid");
        let c = a.matmul(&b).expect("matmul");
        // C = A*B = [[7,16],[36,24],[89,0]]
        assert_eq!(c.get(0, 0), Some(7.0));
        assert_eq!(c.get(0, 1), Some(16.0));
        assert_eq!(c.get(1, 0), Some(36.0));
        assert_eq!(c.get(1, 1), Some(24.0));
        assert_eq!(c.get(2, 0), Some(89.0));
        assert_eq!(c.get(2, 1), None);
    }

    #[test]
    fn dense_solve_small() {
        // [[2,1],[1,3]] x = [3,5] -> x = [0.8, 1.4]
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![3.0, 5.0];
        let x = dense_solve(&a, &b, 2).expect("solve");
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.4).abs() < 1e-12);
    }

    #[test]
    fn dense_solve_singular_errors() {
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![1.0, 2.0];
        assert!(dense_solve(&a, &b, 2).is_err());
    }

    #[test]
    fn gpu_f64_roundtrip() {
        let v = 3.5_f64;
        let g = f64_to_gpu::<f64>(v);
        assert!((gpu_to_f64(g) - v).abs() < 1e-15);
        let gf = f64_to_gpu::<f32>(v);
        assert!((gpu_to_f64(gf) - v).abs() < 1e-6);
    }

    #[test]
    fn laplacian_2d_structure() {
        let a = laplacian_2d(3, 3);
        assert_eq!(a.nrows, 9);
        // Center node (1,1) -> index 4 has diagonal 4 and four -1 neighbours.
        assert_eq!(a.get(4, 4), Some(4.0));
        let start = a.row_ptr[4];
        let end = a.row_ptr[4 + 1];
        assert_eq!(end - start, 5);
    }
}
