//! Triangular solve with multiple right-hand sides (TRSM).
//!
//! Solves `op(A) * X = alpha * B` (side = Left) or
//! `X * op(A) = alpha * B` (side = Right), where A is triangular
//! and the solution X overwrites B.
//!
//! # Block algorithm
//!
//! For large matrices, TRSM is decomposed into blocks:
//!
//! 1. Scale the whole right-hand-side matrix B by `alpha` exactly once.
//! 2. Solve a small triangular system on the diagonal block with the
//!    [`trsm_kernel`](super::trsm_kernel) PTX kernel.
//! 3. Update the remaining columns/rows with a trailing GEMM.
//! 4. Repeat for the next diagonal block.
//!
//! Steps 2–3 leverage the optimised [`gemm`](super::gemm_api::gemm)
//! dispatcher for the bulk of the work, so throughput stays high on large
//! problems while the diagonal kernel keeps the substitution exact.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{
    DiagType, FillMode, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Side, Transpose,
};

use super::trsm_kernel::{
    TrsmKernelConfig, generate_trsm_diag_ptx, generate_trsm_scale_ptx, generate_trsm_update_ptx,
};

// ---------------------------------------------------------------------------
// Block size for the blocked TRSM algorithm
// ---------------------------------------------------------------------------

/// Block size for the blocked TRSM decomposition.
///
/// Each diagonal block is solved with the small TRSM kernel, then the
/// trailing matrix is updated with a GEMM call. The block size trades off
/// between the overhead of small diagonal solves and the efficiency of large
/// GEMM updates.
const TRSM_BLOCK_SIZE: u32 = 64;

/// Threads per block for the diagonal-solve and scale kernel launches.
const TRSM_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Solves a triangular linear system with multiple right-hand sides.
///
/// Depending on `side`:
/// - **Left**: `op(A) * X = alpha * B`, A is M x M, B/X is M x N.
/// - **Right**: `X * op(A) = alpha * B`, A is N x N, B/X is M x N.
///
/// The solution X overwrites B in-place.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `side` — whether A appears on the left or right.
/// * `fill_mode` — which triangle of A is stored (upper or lower).
/// * `trans_a` — transpose mode for A.
/// * `diag` — whether A has an implicit unit diagonal.
/// * `alpha` — scalar multiplier for B.
/// * `a` — descriptor for the triangular matrix A.
/// * `b` — descriptor for the right-hand side / solution matrix B (in-place).
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if A is not square or dimensions
/// are zero. Returns [`BlasError::DimensionMismatch`] if the sizes are
/// incompatible. Returns [`BlasError::UnsupportedOperation`] for element
/// types other than `f32` / `f64`. Returns [`BlasError::PtxGeneration`] or
/// [`BlasError::LaunchFailed`] on kernel build / launch failure.
#[allow(clippy::too_many_arguments)]
pub fn trsm<T: GpuFloat>(
    handle: &BlasHandle,
    side: Side,
    fill_mode: FillMode,
    trans_a: Transpose,
    diag: DiagType,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    // Validate A is square.
    if a.rows != a.cols {
        return Err(BlasError::InvalidDimension(format!(
            "TRSM: triangular matrix A must be square, got {}x{}",
            a.rows, a.cols
        )));
    }

    let tri_n = a.rows;
    let m = b.rows;
    let n = b.cols;

    if m == 0 || n == 0 {
        return Err(BlasError::InvalidDimension(
            "TRSM: B dimensions must be non-zero".into(),
        ));
    }

    // Validate side-dependent dimension agreement.
    match side {
        Side::Left => {
            if tri_n != m {
                return Err(BlasError::DimensionMismatch(format!(
                    "TRSM left: A is {t}x{t} but B has {m} rows",
                    t = tri_n
                )));
            }
        }
        Side::Right => {
            if tri_n != n {
                return Err(BlasError::DimensionMismatch(format!(
                    "TRSM right: A is {t}x{t} but B has {n} cols",
                    t = tri_n
                )));
            }
        }
    }

    // The hand-written diagonal kernel covers the real floating-point types.
    if T::PTX_TYPE != PtxType::F32 && T::PTX_TYPE != PtxType::F64 {
        return Err(BlasError::UnsupportedOperation(
            "TRSM: only f32 and f64 element types are supported".into(),
        ));
    }

    blocked_trsm(handle, side, fill_mode, trans_a, diag, alpha, a, b)
}

// ---------------------------------------------------------------------------
// Blocked TRSM implementation
// ---------------------------------------------------------------------------

/// Blocked TRSM: scales B by `alpha`, then iterates diagonal-block solves
/// interleaved with trailing GEMM updates.
#[allow(clippy::too_many_arguments)]
fn blocked_trsm<T: GpuFloat>(
    handle: &BlasHandle,
    side: Side,
    fill_mode: FillMode,
    trans_a: Transpose,
    diag: DiagType,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    let tri_n = a.rows;
    let nb = TRSM_BLOCK_SIZE.min(tri_n);
    let num_blocks = tri_n.div_ceil(nb);

    // --- Step 1: apply alpha to the whole of B exactly once -------------
    //
    // After this, every diagonal solve and trailing GEMM runs with an
    // implicit unit scalar — pre-scaling is the only consistent way to
    // apply `alpha` once when the trailing updates accumulate `A * X`.
    if alpha != T::gpu_one() {
        scale_matrix(handle, alpha, b)?;
    }

    // --- Diagonal-solve kernel (compiled once, reused per block) --------
    let kernel_config = TrsmKernelConfig {
        sm: handle.sm_version(),
        elem: T::PTX_TYPE,
        side,
        fill_mode,
        trans: trans_a,
        diag,
    };
    let (ptx, kernel_name) = generate_trsm_diag_ptx(&kernel_config)?;
    let module = Arc::new(
        Module::from_ptx(&ptx)
            .map_err(|e| BlasError::LaunchFailed(format!("TRSM: module load failed: {e}")))?,
    );
    let diag_kernel = Kernel::from_module(Arc::clone(&module), &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("TRSM: kernel lookup failed: {e}")))?;

    // --- Trailing-update kernel -----------------------------------------
    //
    // Each diagonal block's contribution is subtracted from the unsolved
    // part of B with a strided matmul-accumulate kernel. It operates
    // directly on strided sub-blocks of A and B, so no scratch packing is
    // required. The kernel is only needed when there is more than one block.
    let update_kernel = if num_blocks > 1 {
        let (upd_ptx, upd_name) = generate_trsm_update_ptx(handle.sm_version(), T::PTX_TYPE)?;
        let upd_module = Arc::new(Module::from_ptx(&upd_ptx).map_err(|e| {
            BlasError::LaunchFailed(format!("TRSM update: module load failed: {e}"))
        })?);
        Some(
            Kernel::from_module(Arc::clone(&upd_module), &upd_name).map_err(|e| {
                BlasError::LaunchFailed(format!("TRSM update: kernel lookup failed: {e}"))
            })?,
        )
    } else {
        None
    };

    // Iteration order: forward (first block first) or backward.
    let forward = kernel_config.forward();
    let a_transposed = matches!(trans_a, Transpose::Trans | Transpose::ConjTrans);

    for block_idx in 0..num_blocks {
        let idx = if forward {
            block_idx
        } else {
            num_blocks - 1 - block_idx
        };

        let block_start = idx * nb;
        let block_end = (block_start + nb).min(tri_n);
        let block_size = block_end - block_start;

        // --- Step 2: solve the diagonal block --------------------------
        solve_diagonal_block(handle, &diag_kernel, side, a, b, block_start, block_size)?;

        // --- Step 3: trailing GEMM update ------------------------------
        //
        // Subtract the just-solved block's contribution from the rows /
        // columns of B that have not been solved yet.
        if let Some(update) = update_kernel.as_ref() {
            trailing_gemm_update(
                handle,
                update,
                side,
                a_transposed,
                a,
                b,
                block_start,
                block_end,
                tri_n,
                forward,
            )?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Step 2 — diagonal-block solve
// ---------------------------------------------------------------------------

/// Launches the diagonal-block solve kernel for one `block_size x block_size`
/// triangular block.
fn solve_diagonal_block<T: GpuFloat>(
    handle: &BlasHandle,
    kernel: &Kernel,
    side: Side,
    a: &MatrixDesc<T>,
    b: &MatrixDescMut<T>,
    block_start: u32,
    block_size: u32,
) -> BlasResult<()> {
    // Element strides of A and B for the current layout.
    let (a_row_stride, a_col_stride) = elem_strides(a.layout, a.ld);
    let (b_row_stride, b_col_stride) = elem_strides(b.layout, b.ld);

    // Diagonal block of A starts at (block_start, block_start).
    let a_off = u64::from(block_start) * u64::from(a_row_stride)
        + u64::from(block_start) * u64::from(a_col_stride);
    let a_ptr = a.ptr + a_off * T::SIZE as u64;

    // The B panel and the number of independent right-hand sides depend on
    // the side: Left owns full columns, Right owns full rows.
    let (b_ptr, vec_count) = match side {
        Side::Left => {
            // Rows [block_start, block_start + block_size); all columns.
            let off = u64::from(block_start) * u64::from(b_row_stride);
            (b.ptr + off * T::SIZE as u64, b.cols)
        }
        Side::Right => {
            // Columns [block_start, block_start + block_size); all rows.
            let off = u64::from(block_start) * u64::from(b_col_stride);
            (b.ptr + off * T::SIZE as u64, b.rows)
        }
    };

    if vec_count == 0 {
        return Ok(());
    }

    let grid_x = vec_count.div_ceil(TRSM_THREADS);
    let params = LaunchParams::new(Dim3::new(grid_x, 1, 1), Dim3::new(TRSM_THREADS, 1, 1));

    // Kernel args: ptr_a, ptr_b, bs, vec_count, lda, ldb,
    //              a_row_stride, a_col_stride, b_row_stride, b_col_stride.
    let args = (
        a_ptr,
        b_ptr,
        block_size,
        vec_count,
        a.ld,
        b.ld,
        a_row_stride,
        a_col_stride,
        b_row_stride,
        b_col_stride,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("TRSM diagonal launch failed: {e}")))
}

// ---------------------------------------------------------------------------
// Step 3 — trailing GEMM update
// ---------------------------------------------------------------------------

/// Issues the trailing GEMM `B_trailing -= op(A_offdiag) * X_block` (Left) or
/// `B_trailing -= X_block * op(A_offdiag)` (Right) via the crate's GEMM
/// dispatcher.
///
/// The GEMM kernels require tightly-packed operands, so every strided
/// sub-block is gathered into a contiguous scratch buffer first; the GEMM
/// result is computed in a contiguous buffer and scattered back over the
/// trailing region of B.
#[allow(clippy::too_many_arguments)]
fn trailing_gemm_update<T: GpuFloat>(
    handle: &BlasHandle,
    update_kernel: &Kernel,
    side: Side,
    a_transposed: bool,
    a: &MatrixDesc<T>,
    b: &mut MatrixDescMut<T>,
    block_start: u32,
    block_end: u32,
    tri_n: u32,
    forward: bool,
) -> BlasResult<()> {
    // Range of not-yet-solved indices along the triangular dimension.
    let (rem_start, rem_len) = if forward {
        (block_end, tri_n.saturating_sub(block_end))
    } else {
        (0u32, block_start)
    };
    if rem_len == 0 {
        return Ok(());
    }

    let neg_one = neg_one::<T>();
    let block_size = block_end - block_start;
    let (b_row, b_col) = elem_strides(b.layout, b.ld);
    let (a_row, a_col) = elem_strides(a.layout, a.ld);
    let esz = T::SIZE as u64;

    // Every operand of the trailing update `C += (-1) * LHS * RHS` is a
    // strided sub-block of A or B. A transpose of `op(A_offdiag)` is applied
    // simply by swapping that operand's `(row_stride, col_stride)` pair, so
    // the matmul-accumulate kernel needs no transpose flag of its own.
    match side {
        Side::Left => {
            // C = B rows [rem_start, rem_start+rem_len), all columns.
            // LHS = op(A_offdiag) (rem_len x block_size).
            // RHS = X_block = B rows [block_start, block_end), all columns.
            let c = block_view(b.ptr, rem_start, 0, b_row, b_col, b_row, b_col, esz);
            let rhs = block_view(b.ptr, block_start, 0, b_row, b_col, b_row, b_col, esz);

            // op(A_offdiag) must be (rem_len x block_size).
            //   NoTrans: it is the stored A panel A[rem_rows, block_cols].
            //   Trans:   it is A[block_rows, rem_cols] read transposed — the
            //            base corner is parent element (block_start, rem_start)
            //            but the view strides are A's strides swapped.
            let lhs = if a_transposed {
                block_view(
                    a.ptr,
                    block_start,
                    rem_start,
                    a_row,
                    a_col,
                    a_col,
                    a_row,
                    esz,
                )
            } else {
                block_view(
                    a.ptr,
                    rem_start,
                    block_start,
                    a_row,
                    a_col,
                    a_row,
                    a_col,
                    esz,
                )
            };

            launch_update::<T>(
                handle,
                update_kernel,
                rem_len,
                b.cols,
                block_size,
                neg_one,
                &c,
                &lhs,
                &rhs,
            )
        }
        Side::Right => {
            // C = B columns [rem_start, rem_start+rem_len), all rows.
            // LHS = X_block = B columns [block_start, block_end), all rows.
            // RHS = op(A_offdiag) (block_size x rem_len).
            let c = block_view(b.ptr, 0, rem_start, b_row, b_col, b_row, b_col, esz);
            let lhs = block_view(b.ptr, 0, block_start, b_row, b_col, b_row, b_col, esz);

            // op(A_offdiag) must be (block_size x rem_len).
            //   NoTrans: stored A panel A[block_rows, rem_cols].
            //   Trans:   A[rem_rows, block_cols] read transposed.
            let rhs = if a_transposed {
                block_view(
                    a.ptr,
                    rem_start,
                    block_start,
                    a_row,
                    a_col,
                    a_col,
                    a_row,
                    esz,
                )
            } else {
                block_view(
                    a.ptr,
                    block_start,
                    rem_start,
                    a_row,
                    a_col,
                    a_row,
                    a_col,
                    esz,
                )
            };

            launch_update::<T>(
                handle,
                update_kernel,
                b.rows,
                rem_len,
                block_size,
                neg_one,
                &c,
                &lhs,
                &rhs,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Trailing-update kernel launch
// ---------------------------------------------------------------------------

/// A strided matrix sub-block: a base device pointer plus the element
/// strides along its row and column axes.
#[derive(Debug, Clone, Copy)]
struct BlockView {
    /// Device pointer to element `(0, 0)` of the sub-block.
    ptr: u64,
    /// Element stride between consecutive rows.
    row_stride: u32,
    /// Element stride between consecutive columns.
    col_stride: u32,
}

/// Builds a [`BlockView`] for a sub-block of a parent matrix.
///
/// The parent matrix has element strides `parent_row` / `parent_col`. The
/// sub-block's top-left corner is the parent element `(r0, c0)`, addressed
/// with those parent strides. The returned view, however, advertises
/// `view_row` / `view_col` strides — which may be the parent strides
/// *swapped*, so the sub-block is seen transposed by the consuming kernel
/// without any data movement.
///
/// `elem_size` is the size of one element in bytes.
#[allow(clippy::too_many_arguments)]
fn block_view(
    base_ptr: u64,
    r0: u32,
    c0: u32,
    parent_row: u32,
    parent_col: u32,
    view_row: u32,
    view_col: u32,
    elem_size: u64,
) -> BlockView {
    // The base pointer is always located with the *parent* strides.
    let off = u64::from(r0) * u64::from(parent_row) + u64::from(c0) * u64::from(parent_col);
    BlockView {
        ptr: base_ptr + off * elem_size,
        row_stride: view_row,
        col_stride: view_col,
    }
}

/// Launches the trailing-update kernel `C[m x n] += alpha * LHS[m x kc] *
/// RHS[kc x n]`, passing the element-typed scalar in its native width.
#[allow(clippy::too_many_arguments)]
fn launch_update<T: GpuFloat>(
    handle: &BlasHandle,
    kernel: &Kernel,
    m: u32,
    n: u32,
    kc: u32,
    alpha: T,
    c: &BlockView,
    lhs: &BlockView,
    rhs: &BlockView,
) -> BlasResult<()> {
    let total = m * n;
    if total == 0 {
        return Ok(());
    }
    let grid_x = total.div_ceil(TRSM_THREADS);
    let params = LaunchParams::new(Dim3::new(grid_x, 1, 1), Dim3::new(TRSM_THREADS, 1, 1));

    match T::PTX_TYPE {
        PtxType::F64 => {
            let alpha_f64 = f64::from_bits(alpha.to_bits_u64());
            let args = (
                c.ptr,
                lhs.ptr,
                rhs.ptr,
                m,
                n,
                kc,
                alpha_f64,
                c.row_stride,
                c.col_stride,
                lhs.row_stride,
                lhs.col_stride,
                rhs.row_stride,
                rhs.col_stride,
            );
            kernel
                .launch(&params, handle.stream(), &args)
                .map_err(|e| BlasError::LaunchFailed(format!("TRSM update launch failed: {e}")))
        }
        _ => {
            let alpha_f32 = f32::from_bits(alpha.to_bits_u64() as u32);
            let args = (
                c.ptr,
                lhs.ptr,
                rhs.ptr,
                m,
                n,
                kc,
                alpha_f32,
                c.row_stride,
                c.col_stride,
                lhs.row_stride,
                lhs.col_stride,
                rhs.row_stride,
                rhs.col_stride,
            );
            kernel
                .launch(&params, handle.stream(), &args)
                .map_err(|e| BlasError::LaunchFailed(format!("TRSM update launch failed: {e}")))
        }
    }
}

// ---------------------------------------------------------------------------
// alpha-scale of B
// ---------------------------------------------------------------------------

/// Scales every element of the matrix `b` in place by `alpha`.
fn scale_matrix<T: GpuFloat>(
    handle: &BlasHandle,
    alpha: T,
    b: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    let (ptx, kernel_name) = generate_trsm_scale_ptx(handle.sm_version(), T::PTX_TYPE)?;
    let module =
        Arc::new(Module::from_ptx(&ptx).map_err(|e| {
            BlasError::LaunchFailed(format!("TRSM scale: module load failed: {e}"))
        })?);
    let kernel = Kernel::from_module(Arc::clone(&module), &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("TRSM scale: kernel lookup failed: {e}")))?;

    let (row_stride, col_stride) = elem_strides(b.layout, b.ld);
    let total = b.rows * b.cols;
    let grid_x = total.div_ceil(TRSM_THREADS);
    let params = LaunchParams::new(Dim3::new(grid_x, 1, 1), Dim3::new(TRSM_THREADS, 1, 1));

    // The scalar must be passed in the element type's own width.
    match T::PTX_TYPE {
        PtxType::F64 => {
            let alpha_f64 = f64::from_bits(alpha.to_bits_u64());
            let args = (b.ptr, b.rows, b.cols, row_stride, col_stride, alpha_f64);
            kernel
                .launch(&params, handle.stream(), &args)
                .map_err(|e| BlasError::LaunchFailed(format!("TRSM scale launch failed: {e}")))
        }
        _ => {
            let alpha_f32 = f32::from_bits(alpha.to_bits_u64() as u32);
            let args = (b.ptr, b.rows, b.cols, row_stride, col_stride, alpha_f32);
            kernel
                .launch(&params, handle.stream(), &args)
                .map_err(|e| BlasError::LaunchFailed(format!("TRSM scale launch failed: {e}")))
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor helpers
// ---------------------------------------------------------------------------

/// Returns the `(row_stride, col_stride)` element strides for a layout.
fn elem_strides(layout: Layout, ld: u32) -> (u32, u32) {
    match layout {
        Layout::RowMajor => (ld, 1),
        Layout::ColMajor => (1, ld),
    }
}

/// Returns `-1` in the element type `T`.
fn neg_one<T: GpuFloat>() -> T {
    match T::PTX_TYPE {
        PtxType::F64 => T::from_bits_u64((-1.0f64).to_bits()),
        _ => T::from_bits_u64(u64::from((-1.0f32).to_bits())),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trsm_block_size_positive() {
        const { assert!(TRSM_BLOCK_SIZE > 0) };
    }

    #[test]
    fn validate_non_square_error_message() {
        let err = BlasError::InvalidDimension("TRSM: triangular matrix A must be square".into());
        assert!(err.to_string().contains("square"));
    }

    #[test]
    fn blocked_iteration_count() {
        // 256 / 64 = 4 blocks.
        let tri_n = 256u32;
        let nb = TRSM_BLOCK_SIZE.min(tri_n);
        let num_blocks = tri_n.div_ceil(nb);
        assert_eq!(num_blocks, 4);
    }

    #[test]
    fn blocked_iteration_count_non_divisible() {
        // 300 / 64 = 5 blocks (last block is 44).
        let tri_n = 300u32;
        let nb = TRSM_BLOCK_SIZE.min(tri_n);
        let num_blocks = tri_n.div_ceil(nb);
        assert_eq!(num_blocks, 5);
    }

    #[test]
    fn diag_type_values() {
        assert_ne!(DiagType::Unit, DiagType::NonUnit);
    }

    #[test]
    fn elem_strides_row_major() {
        assert_eq!(elem_strides(Layout::RowMajor, 16), (16, 1));
    }

    #[test]
    fn elem_strides_col_major() {
        assert_eq!(elem_strides(Layout::ColMajor, 16), (1, 16));
    }

    #[test]
    fn neg_one_f32_is_minus_one() {
        assert_eq!(neg_one::<f32>(), -1.0f32);
    }

    #[test]
    fn neg_one_f64_is_minus_one() {
        assert_eq!(neg_one::<f64>(), -1.0f64);
    }

    #[test]
    fn block_view_offsets_row_major() {
        // Row-major 8x8 (row_stride 8, col_stride 1): element (2,3) is at
        // 2*8 + 3 = 19; with f32 elements that is byte offset 19*4.
        let (row_stride, col_stride) = elem_strides(Layout::RowMajor, 8);
        let view = block_view(0, 2, 3, row_stride, col_stride, row_stride, col_stride, 4);
        assert_eq!(view.ptr, 19 * 4);
        assert_eq!((view.row_stride, view.col_stride), (8, 1));
    }

    #[test]
    fn block_view_offsets_col_major() {
        // Column-major 8x8 (row_stride 1, col_stride 8): element (2,3) is at
        // 3*8 + 2 = 26; with f64 elements that is byte offset 26*8.
        let (row_stride, col_stride) = elem_strides(Layout::ColMajor, 8);
        let view = block_view(0, 2, 3, row_stride, col_stride, row_stride, col_stride, 8);
        assert_eq!(view.ptr, 26 * 8);
        assert_eq!((view.row_stride, view.col_stride), (1, 8));
    }

    #[test]
    fn block_view_transposed_keeps_true_base_swaps_view() {
        // Row-major 16-wide parent. The transposed view of element (2, 3)
        // must still locate the base at the *true* element 2*16 + 3 = 35,
        // but advertise swapped view strides.
        let (row_stride, col_stride) = elem_strides(Layout::RowMajor, 16);
        let view = block_view(0, 2, 3, row_stride, col_stride, col_stride, row_stride, 4);
        assert_eq!(view.ptr, 35 * 4);
        assert_eq!((view.row_stride, view.col_stride), (1, 16));
    }
}
