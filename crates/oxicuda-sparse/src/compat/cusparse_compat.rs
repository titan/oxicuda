//! cuSPARSE-compatible API shim over the crate's host sparse kernels.
//!
//! This module mirrors the shape of the core cuSPARSE generic-API entry points
//! so that code written against cuSPARSE reads almost unchanged when ported to
//! OxiCUDA. The naming is cuSPARSE-flavoured (a library *handle* with a
//! create/destroy lifecycle and a scalar *pointer mode*, plus `cusparse_spmv` /
//! `cusparse_spgemm` / `cusparse_spsv` operations) but the surface is fully
//! idiomatic Rust: operations take `&` references, return
//! [`SparseResult`], validate their arguments, and expose no raw pointers or
//! `unsafe`.
//!
//! All operations act on the crate's host CSR type [`HostCsr`] and delegate to
//! its existing kernels:
//!
//! - [`cusparse_spmv`] computes `y = alpha * A * x + beta * y` by reusing
//!   [`HostCsr::matvec`];
//! - [`cusparse_spgemm`] computes `C = A * B` by reusing [`HostCsr::matmul`]
//!   (the crate's Gustavson sparse-sparse product);
//! - [`cusparse_spsv`] solves the triangular system `A * x = alpha * b` for a
//!   lower- or upper-triangular `A` by sparse forward/backward substitution.
//!
//! The triangular descriptor is expressed with the crate's existing
//! [`FillMode`] and [`DiagType`] enums (re-exported from `oxicuda-blas`), the
//! same descriptor types the GPU [`sptrsv`](mod@crate::ops::sptrsv) uses.

use std::cell::Cell;

use oxicuda_blas::{DiagType, FillMode};

use crate::error::{SparseError, SparseResult};
use crate::host_csr::HostCsr;

/// How scalar arguments (`alpha`, `beta`) are interpreted, mirroring
/// `cusparsePointerMode_t`.
///
/// In this host shim both modes pass scalars by value; the distinction is
/// retained for API faithfulness and round-trips through the handle so that
/// porting code which sets a pointer mode keeps compiling and behaving sanely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CusparsePointerMode {
    /// Scalars are taken from host values (the default).
    #[default]
    Host,
    /// Scalars are nominally device-resident; treated identically here.
    Device,
}

/// A cuSPARSE-compatible library handle, analogous to `cusparseHandle_t`.
///
/// Created with [`CusparseCompatHandle::create`] and released with
/// [`CusparseCompatHandle::destroy`] (which consumes the handle, matching
/// `cusparseDestroy`). Every operation in this module takes a handle reference
/// as its first argument and records itself in the handle's operation counter,
/// so a caller can observe activity over the handle's lifetime.
#[derive(Debug)]
pub struct CusparseCompatHandle {
    /// Scalar pointer mode for `alpha`/`beta` arguments.
    pointer_mode: CusparsePointerMode,
    /// Count of operations dispatched through this handle.
    op_count: Cell<u64>,
}

impl CusparseCompatHandle {
    /// Creates a new handle with the default ([`CusparsePointerMode::Host`])
    /// pointer mode, analogous to `cusparseCreate`.
    ///
    /// # Errors
    ///
    /// This host shim never fails to create a handle; the [`SparseResult`]
    /// return type preserves the cuSPARSE signature shape.
    pub fn create() -> SparseResult<Self> {
        Ok(Self {
            pointer_mode: CusparsePointerMode::default(),
            op_count: Cell::new(0),
        })
    }

    /// Releases the handle, analogous to `cusparseDestroy`.
    ///
    /// Consuming `self` makes use-after-destroy a compile-time error, the
    /// idiomatic-Rust equivalent of invalidating the opaque cuSPARSE handle.
    ///
    /// # Errors
    ///
    /// Never returns an error in this host shim.
    pub fn destroy(self) -> SparseResult<()> {
        Ok(())
    }

    /// Returns the current scalar pointer mode.
    #[inline]
    pub fn pointer_mode(&self) -> CusparsePointerMode {
        self.pointer_mode
    }

    /// Sets the scalar pointer mode, analogous to `cusparseSetPointerMode`.
    #[inline]
    pub fn set_pointer_mode(&mut self, mode: CusparsePointerMode) {
        self.pointer_mode = mode;
    }

    /// Number of operations dispatched through this handle so far.
    #[inline]
    pub fn op_count(&self) -> u64 {
        self.op_count.get()
    }

    /// Records one dispatched operation (internal bookkeeping).
    #[inline]
    fn record_op(&self) {
        self.op_count.set(self.op_count.get() + 1);
    }
}

/// Computes the sparse matrix-vector product `y = alpha * A * x + beta * y`,
/// mirroring `cusparseSpMV` for a CSR operand.
///
/// `x` must have length `A.ncols` and `y` length `A.nrows`. When `beta == 0`
/// the prior contents of `y` are ignored (overwritten), matching cuSPARSE.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `x.len() != A.ncols` or
/// `y.len() != A.nrows`.
pub fn cusparse_spmv(
    handle: &CusparseCompatHandle,
    alpha: f64,
    a: &HostCsr,
    x: &[f64],
    beta: f64,
    y: &mut [f64],
) -> SparseResult<()> {
    handle.record_op();
    if x.len() != a.ncols {
        return Err(SparseError::DimensionMismatch(format!(
            "SpMV: x length ({}) must equal A.ncols ({})",
            x.len(),
            a.ncols
        )));
    }
    if y.len() != a.nrows {
        return Err(SparseError::DimensionMismatch(format!(
            "SpMV: y length ({}) must equal A.nrows ({})",
            y.len(),
            a.nrows
        )));
    }

    // Reuse the crate's host SpMV kernel for `A * x`, then apply the BLAS-style
    // alpha/beta accumulation. `beta == 0` overwrites `y` (and is robust to a
    // pre-existing non-finite value in `y`, as cuSPARSE requires).
    let ax = a.matvec(x);
    for (yi, &ax_i) in y.iter_mut().zip(ax.iter()) {
        let scaled = if beta == 0.0 { 0.0 } else { beta * *yi };
        *yi = alpha * ax_i + scaled;
    }
    Ok(())
}

/// Computes the sparse-sparse product `C = A * B`, mirroring `cusparseSpGEMM`.
///
/// Delegates to [`HostCsr::matmul`], the crate's Gustavson SpGEMM, returning a
/// freshly allocated [`HostCsr`] with sorted column indices per row.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.ncols != B.nrows`.
pub fn cusparse_spgemm(
    handle: &CusparseCompatHandle,
    a: &HostCsr,
    b: &HostCsr,
) -> SparseResult<HostCsr> {
    handle.record_op();
    a.matmul(b)
}

/// Solves the triangular system `A * x = alpha * b`, mirroring `cusparseSpSV`.
///
/// `A` must be square and is interpreted as lower-triangular when
/// `fill_mode == FillMode::Lower` (forward substitution) or upper-triangular
/// when `fill_mode == FillMode::Upper` (backward substitution). Entries on the
/// "wrong" side of the diagonal for the chosen `fill_mode` are ignored, as in
/// cuSPARSE. With `diag_type == DiagType::Unit` the diagonal is treated as an
/// implicit all-ones diagonal and any stored diagonal entries are not read.
///
/// The returned vector `x` has length `A.nrows` and satisfies `A * x ≈ alpha * b`
/// (taking the configured triangle and diagonal mode into account).
///
/// # Errors
///
/// Returns [`SparseError::InvalidArgument`] if `fill_mode == FillMode::Full`
/// (the solve requires a triangle), [`SparseError::DimensionMismatch`] if `A`
/// is not square or `b.len() != A.nrows`, and [`SparseError::SingularMatrix`]
/// if a needed diagonal entry is missing or numerically zero under
/// [`DiagType::NonUnit`].
pub fn cusparse_spsv(
    handle: &CusparseCompatHandle,
    fill_mode: FillMode,
    diag_type: DiagType,
    alpha: f64,
    a: &HostCsr,
    b: &[f64],
) -> SparseResult<Vec<f64>> {
    handle.record_op();
    if a.nrows != a.ncols {
        return Err(SparseError::DimensionMismatch(format!(
            "SpSV requires a square matrix, got {}x{}",
            a.nrows, a.ncols
        )));
    }
    if b.len() != a.nrows {
        return Err(SparseError::DimensionMismatch(format!(
            "SpSV: b length ({}) must equal A.nrows ({})",
            b.len(),
            a.nrows
        )));
    }
    let n = a.nrows;
    let mut x = vec![0.0f64; n];

    match fill_mode {
        FillMode::Lower => {
            // Forward substitution: rows top-to-bottom. Each row reads the
            // already-solved entries `x[c]` for `c < i`.
            for i in 0..n {
                let (acc, diag) = substitution_row(a, i, alpha * b[i], &x, c_below(i));
                x[i] = finish_row(acc, diag, diag_type)?;
            }
        }
        FillMode::Upper => {
            // Backward substitution: rows bottom-to-top. Each row reads the
            // already-solved entries `x[c]` for `c > i`.
            for i in (0..n).rev() {
                let (acc, diag) = substitution_row(a, i, alpha * b[i], &x, c_above(i));
                x[i] = finish_row(acc, diag, diag_type)?;
            }
        }
        FillMode::Full => {
            return Err(SparseError::InvalidArgument(
                "SpSV requires a triangular fill mode (Lower or Upper), got Full".to_string(),
            ));
        }
    }

    Ok(x)
}

/// Predicate selecting strictly-lower columns `c < i` (forward substitution).
#[inline]
fn c_below(i: usize) -> impl Fn(usize) -> bool {
    move |c| c < i
}

/// Predicate selecting strictly-upper columns `c > i` (backward substitution).
#[inline]
fn c_above(i: usize) -> impl Fn(usize) -> bool {
    move |c| c > i
}

/// Accumulates one substitution step for row `i`: starts from `rhs`, subtracts
/// `A[i, c] * x[c]` for every already-solved column accepted by `is_solved`,
/// and returns `(numerator, diagonal_entry)`.
#[inline]
fn substitution_row(
    a: &HostCsr,
    i: usize,
    rhs: f64,
    x: &[f64],
    is_solved: impl Fn(usize) -> bool,
) -> (f64, Option<f64>) {
    let mut acc = rhs;
    let mut diag: Option<f64> = None;
    let start = a.row_ptr[i];
    let end = a.row_ptr[i + 1];
    for k in start..end {
        let c = a.col_indices[k];
        if c == i {
            diag = Some(a.values[k]);
        } else if is_solved(c) {
            acc -= a.values[k] * x[c];
        }
    }
    (acc, diag)
}

/// Completes a substitution step: divides the accumulated numerator by the
/// diagonal (or treats it as unit), validating non-singularity.
#[inline]
fn finish_row(acc: f64, diag: Option<f64>, diag_type: DiagType) -> SparseResult<f64> {
    match diag_type {
        DiagType::Unit => Ok(acc),
        DiagType::NonUnit => match diag {
            Some(d) if d.abs() > 1e-300 => Ok(acc / d),
            _ => Err(SparseError::SingularMatrix),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense `A * x` reference (row-major `nrows x ncols`).
    fn dense_matvec(dense: &[f64], nrows: usize, ncols: usize, x: &[f64]) -> Vec<f64> {
        (0..nrows)
            .map(|i| {
                dense[i * ncols..(i + 1) * ncols]
                    .iter()
                    .zip(x.iter())
                    .map(|(&a, &xj)| a * xj)
                    .sum()
            })
            .collect()
    }

    fn sample_matrix() -> HostCsr {
        // [[1, 0, 2],
        //  [0, 3, 0],
        //  [4, 0, 5]]
        HostCsr::new(
            3,
            3,
            vec![0, 2, 3, 5],
            vec![0, 2, 1, 0, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        )
        .expect("sample")
    }

    #[test]
    fn spmv_alpha1_beta0_matches_dense() {
        let h = CusparseCompatHandle::create().expect("handle");
        let a = sample_matrix();
        let x = [1.0, 2.0, 3.0];
        let mut y = vec![0.0; 3];
        cusparse_spmv(&h, 1.0, &a, &x, 0.0, &mut y).expect("spmv");
        let reference = dense_matvec(&a.to_dense(), 3, 3, &x);
        for (got, want) in y.iter().zip(reference.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        }
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn spmv_general_alpha_beta() {
        let h = CusparseCompatHandle::create().expect("handle");
        let a = sample_matrix();
        let x = [2.0, -1.0, 0.5];
        let y0 = [10.0, 20.0, 30.0];
        let alpha = 1.5;
        let beta = -2.0;
        let mut y = y0.to_vec();
        cusparse_spmv(&h, alpha, &a, &x, beta, &mut y).expect("spmv");
        let ax = dense_matvec(&a.to_dense(), 3, 3, &x);
        for i in 0..3 {
            let want = alpha * ax[i] + beta * y0[i];
            assert!((y[i] - want).abs() < 1e-12, "row {i}: {} vs {}", y[i], want);
        }
    }

    #[test]
    fn spmv_beta0_ignores_prior_nonfinite_y() {
        let h = CusparseCompatHandle::create().expect("handle");
        let a = sample_matrix();
        let x = [1.0, 1.0, 1.0];
        let mut y = vec![f64::NAN, f64::INFINITY, -f64::INFINITY];
        cusparse_spmv(&h, 1.0, &a, &x, 0.0, &mut y).expect("spmv");
        assert!(y.iter().all(|v| v.is_finite()), "beta=0 must overwrite y");
    }

    #[test]
    fn spgemm_matches_dense() {
        let h = CusparseCompatHandle::create().expect("handle");
        let a = sample_matrix();
        // B is 3x2.
        let b = HostCsr::new(
            3,
            2,
            vec![0, 1, 2, 4],
            vec![0, 1, 0, 1],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .expect("b");
        let c = cusparse_spgemm(&h, &a, &b).expect("spgemm");

        // Dense reference C = A*B.
        let da = a.to_dense();
        let db = b.to_dense();
        let mut dc = [0.0f64; 3 * 2];
        for i in 0..3 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += da[i * 3 + k] * db[k * 2 + j];
                }
                dc[i * 2 + j] = acc;
            }
        }
        for i in 0..3 {
            for j in 0..2 {
                let got = c.get(i, j).unwrap_or(0.0);
                assert!((got - dc[i * 2 + j]).abs() < 1e-12, "C[{i},{j}]");
            }
        }
    }

    #[test]
    fn spsv_lower_solves_system() {
        let h = CusparseCompatHandle::create().expect("handle");
        // Lower-triangular L (non-unit diagonal):
        // [[2, 0, 0],
        //  [1, 3, 0],
        //  [4, 5, 6]]
        let l = HostCsr::new(
            3,
            3,
            vec![0, 1, 3, 6],
            vec![0, 0, 1, 0, 1, 2],
            vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0],
        )
        .expect("l");
        let x_true = [1.0, 2.0, 3.0];
        let b = l.matvec(&x_true);
        let x = cusparse_spsv(&h, FillMode::Lower, DiagType::NonUnit, 1.0, &l, &b).expect("spsv");
        for (got, want) in x.iter().zip(x_true.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        }
        // Verify A * x ≈ b.
        let check = l.matvec(&x);
        for (c, bi) in check.iter().zip(b.iter()) {
            assert!((c - bi).abs() < 1e-12);
        }
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn spsv_upper_solves_system() {
        let h = CusparseCompatHandle::create().expect("handle");
        // Upper-triangular U:
        // [[2, 1, 1],
        //  [0, 3, 2],
        //  [0, 0, 4]]
        let u = HostCsr::new(
            3,
            3,
            vec![0, 3, 5, 6],
            vec![0, 1, 2, 1, 2, 2],
            vec![2.0, 1.0, 1.0, 3.0, 2.0, 4.0],
        )
        .expect("u");
        let x_true = [-1.0, 2.0, 0.5];
        let b = u.matvec(&x_true);
        let x = cusparse_spsv(&h, FillMode::Upper, DiagType::NonUnit, 1.0, &u, &b).expect("spsv");
        for (got, want) in x.iter().zip(x_true.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn spsv_unit_diagonal() {
        let h = CusparseCompatHandle::create().expect("handle");
        // Unit lower-triangular: implicit 1 on the diagonal (stored diagonal,
        // if present, is ignored). Use a matrix WITHOUT stored diagonal.
        // L = [[1,0,0],[2,1,0],[3,4,1]] with the unit diagonal implicit.
        let l =
            HostCsr::new(3, 3, vec![0, 0, 1, 3], vec![0, 0, 1], vec![2.0, 3.0, 4.0]).expect("l");
        let x_true = [5.0, -1.0, 2.0];
        // b = L_full * x_true where L_full has unit diagonal.
        let b = [
            x_true[0],
            2.0 * x_true[0] + x_true[1],
            3.0 * x_true[0] + 4.0 * x_true[1] + x_true[2],
        ];
        let x = cusparse_spsv(&h, FillMode::Lower, DiagType::Unit, 1.0, &l, &b).expect("spsv");
        for (got, want) in x.iter().zip(x_true.iter()) {
            assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        }
    }

    #[test]
    fn spsv_alpha_scales_rhs() {
        let h = CusparseCompatHandle::create().expect("handle");
        let l = HostCsr::new(2, 2, vec![0, 1, 3], vec![0, 0, 1], vec![2.0, 1.0, 4.0]).expect("l");
        let b = [3.0, 7.0];
        let alpha = 2.0;
        let x = cusparse_spsv(&h, FillMode::Lower, DiagType::NonUnit, alpha, &l, &b).expect("spsv");
        // A * x should equal alpha * b.
        let ax = l.matvec(&x);
        for (axi, bi) in ax.iter().zip(b.iter()) {
            assert!((axi - alpha * bi).abs() < 1e-12);
        }
    }

    #[test]
    fn handle_lifecycle_and_op_count() {
        let mut h = CusparseCompatHandle::create().expect("create");
        assert_eq!(h.pointer_mode(), CusparsePointerMode::Host);
        assert_eq!(h.op_count(), 0);

        let a = sample_matrix();
        let x = [1.0, 1.0, 1.0];
        let mut y = vec![0.0; 3];
        cusparse_spmv(&h, 1.0, &a, &x, 0.0, &mut y).expect("spmv");
        assert_eq!(h.op_count(), 1);

        h.set_pointer_mode(CusparsePointerMode::Device);
        assert_eq!(h.pointer_mode(), CusparsePointerMode::Device);

        let _ = cusparse_spgemm(&h, &a, &a).expect("spgemm");
        assert_eq!(h.op_count(), 2);

        h.destroy().expect("destroy");
    }

    #[test]
    fn spmv_dimension_mismatch_errors() {
        let h = CusparseCompatHandle::create().expect("handle");
        let a = sample_matrix();
        let bad_x = [1.0, 2.0]; // len 2 != ncols 3
        let mut y = vec![0.0; 3];
        assert!(cusparse_spmv(&h, 1.0, &a, &bad_x, 0.0, &mut y).is_err());
        let x = [1.0, 2.0, 3.0];
        let mut bad_y = vec![0.0; 2]; // len 2 != nrows 3
        assert!(cusparse_spmv(&h, 1.0, &a, &x, 0.0, &mut bad_y).is_err());
    }

    #[test]
    fn spgemm_dimension_mismatch_errors() {
        let h = CusparseCompatHandle::create().expect("handle");
        let a = sample_matrix(); // 3x3
        let b = HostCsr::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("b"); // 2x2
        assert!(cusparse_spgemm(&h, &a, &b).is_err());
    }

    #[test]
    fn spsv_rejects_non_square_and_full() {
        let h = CusparseCompatHandle::create().expect("handle");
        let rect = HostCsr::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).expect("rect");
        let b = [1.0, 2.0];
        assert!(cusparse_spsv(&h, FillMode::Lower, DiagType::NonUnit, 1.0, &rect, &b).is_err());

        let sq = sample_matrix();
        let b3 = [1.0, 2.0, 3.0];
        assert!(
            cusparse_spsv(&h, FillMode::Full, DiagType::NonUnit, 1.0, &sq, &b3).is_err(),
            "Full fill mode must be rejected"
        );
    }

    #[test]
    fn spsv_singular_diagonal_errors() {
        let h = CusparseCompatHandle::create().expect("handle");
        // Lower-triangular with a missing diagonal on row 1 -> singular under
        // NonUnit.
        let l = HostCsr::new(2, 2, vec![0, 1, 2], vec![0, 0], vec![2.0, 5.0]).expect("l");
        let b = [2.0, 5.0];
        assert!(cusparse_spsv(&h, FillMode::Lower, DiagType::NonUnit, 1.0, &l, &b).is_err());
    }
}
