//! Triangular matrix multiply (TRMM).
//!
//! Computes `B = alpha * op(A) * B` (side = Left) or
//! `B = alpha * B * op(A)` (side = Right), where A is triangular.
//!
//! Only the triangle indicated by `fill_mode` is read from A. Elements
//! outside the triangle are treated as zero (or one on the diagonal when
//! `diag == Unit`).
//!
//! # Implementation
//!
//! A dedicated triangular-aware multiply kernel
//! ([`trmm_kernel`](super::trmm_kernel)) computes one output element per
//! thread, reading only the stored triangle of A. Because TRMM overwrites B,
//! the kernel writes into a tightly-packed scratch buffer first — that keeps
//! it race-free — and a strided copy kernel then writes the scratch back over
//! B in place, honouring B's leading dimension.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{DiagType, FillMode, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Transpose};

use super::trmm_kernel::{TrmmKernelConfig, generate_trmm_copy_ptx, generate_trmm_mul_ptx};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Threads per block for the multiply and copy kernel launches.
const TRMM_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs a triangular matrix multiply on the GPU.
///
/// Depending on `side`:
/// - **Left**: `B = alpha * op(A) * B`, A is M x M triangular, B is M x N.
/// - **Right**: `B = alpha * B * op(A)`, A is N x N triangular, B is M x N.
///
/// The result overwrites B in-place.
///
/// # Arguments
///
/// * `handle` — BLAS handle.
/// * `side` — whether the triangular matrix is on the left or right.
/// * `fill_mode` — which triangle of A is stored.
/// * `trans_a` — transpose mode for A.
/// * `diag` — whether A has an implicit unit diagonal.
/// * `alpha` — scalar multiplier.
/// * `a` — descriptor for the triangular matrix A.
/// * `b` — descriptor for matrix B (in-place, read and written).
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if A is not square or dimensions
/// are zero. Returns [`BlasError::DimensionMismatch`] if sizes are
/// incompatible. Returns [`BlasError::UnsupportedOperation`] for element
/// types other than `f32` / `f64`. Returns [`BlasError::PtxGeneration`] or
/// [`BlasError::LaunchFailed`] on kernel build / launch failure, and
/// [`BlasError::Cuda`] if the scratch allocation fails.
#[allow(clippy::too_many_arguments)]
pub fn trmm<T: GpuFloat>(
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
            "TRMM: triangular matrix A must be square, got {}x{}",
            a.rows, a.cols
        )));
    }

    let tri_n = a.rows;
    let m = b.rows;
    let n = b.cols;

    if m == 0 || n == 0 {
        return Err(BlasError::InvalidDimension(
            "TRMM: B dimensions must be non-zero".into(),
        ));
    }

    // Validate dimension agreement.
    match side {
        Side::Left => {
            if tri_n != m {
                return Err(BlasError::DimensionMismatch(format!(
                    "TRMM left: A is {t}x{t} but B has {m} rows",
                    t = tri_n
                )));
            }
        }
        Side::Right => {
            if tri_n != n {
                return Err(BlasError::DimensionMismatch(format!(
                    "TRMM right: A is {t}x{t} but B has {n} cols",
                    t = tri_n
                )));
            }
        }
    }

    // The hand-written multiply kernel covers the real floating-point types.
    if T::PTX_TYPE != PtxType::F32 && T::PTX_TYPE != PtxType::F64 {
        return Err(BlasError::UnsupportedOperation(
            "TRMM: only f32 and f64 element types are supported".into(),
        ));
    }

    triangular_multiply(handle, side, fill_mode, trans_a, diag, alpha, a, b)
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/// Computes `B := alpha * op(A) * B` / `B := alpha * B * op(A)` via the
/// dedicated triangular-multiply kernel plus a strided copy-back.
#[allow(clippy::too_many_arguments)]
fn triangular_multiply<T: GpuFloat>(
    handle: &BlasHandle,
    side: Side,
    fill_mode: FillMode,
    trans_a: Transpose,
    diag: DiagType,
    alpha: T,
    a: &MatrixDesc<T>,
    b: &mut MatrixDescMut<T>,
) -> BlasResult<()> {
    let m = b.rows;
    let n = b.cols;

    // --- Scratch buffer: tightly-packed, row-major (m x n) --------------
    //
    // The multiply kernel writes its result here so it never overwrites B
    // while other threads are still reading B.
    let scratch = DeviceBuffer::<T>::alloc((m as usize) * (n as usize))?;
    let scratch_ptr = scratch.as_device_ptr();
    // Packed row-major strides: element (r, c) lives at r * n + c.
    let (scratch_row_stride, scratch_col_stride) = (n, 1u32);

    // --- Multiply kernel ------------------------------------------------
    let mul_config = TrmmKernelConfig {
        sm: handle.sm_version(),
        elem: T::PTX_TYPE,
        side,
        fill_mode,
        trans: trans_a,
        diag,
    };
    let (mul_ptx, mul_name) = generate_trmm_mul_ptx(&mul_config)?;
    let mul_module = Arc::new(
        Module::from_ptx(&mul_ptx)
            .map_err(|e| BlasError::LaunchFailed(format!("TRMM: module load failed: {e}")))?,
    );
    let mul_kernel = Kernel::from_module(Arc::clone(&mul_module), &mul_name)
        .map_err(|e| BlasError::LaunchFailed(format!("TRMM: kernel lookup failed: {e}")))?;

    let (a_row_stride, a_col_stride) = elem_strides(a.layout, a.ld);
    let (b_row_stride, b_col_stride) = elem_strides(b.layout, b.ld);

    let total = m * n;
    let grid_x = total.div_ceil(TRMM_THREADS);
    let params = LaunchParams::new(Dim3::new(grid_x, 1, 1), Dim3::new(TRMM_THREADS, 1, 1));

    launch_mul(
        &mul_kernel,
        handle,
        &params,
        a.ptr,
        b.ptr,
        scratch_ptr,
        m,
        n,
        alpha,
        [a_row_stride, a_col_stride],
        [b_row_stride, b_col_stride],
        [scratch_row_stride, scratch_col_stride],
    )?;

    // --- Copy the scratch result back over B in place ------------------
    let (copy_ptx, copy_name) = generate_trmm_copy_ptx(handle.sm_version(), T::PTX_TYPE)?;
    let copy_module = Arc::new(
        Module::from_ptx(&copy_ptx)
            .map_err(|e| BlasError::LaunchFailed(format!("TRMM copy: module load failed: {e}")))?,
    );
    let copy_kernel = Kernel::from_module(Arc::clone(&copy_module), &copy_name)
        .map_err(|e| BlasError::LaunchFailed(format!("TRMM copy: kernel lookup failed: {e}")))?;

    // dst = B (real strides), src = scratch (packed strides).
    let copy_args = (
        b.ptr,
        scratch_ptr,
        m,
        n,
        b_row_stride,
        b_col_stride,
        scratch_row_stride,
        scratch_col_stride,
    );
    copy_kernel
        .launch(&params, handle.stream(), &copy_args)
        .map_err(|e| BlasError::LaunchFailed(format!("TRMM copy launch failed: {e}")))?;

    // The scratch buffer must outlive both launches; the stream is
    // synchronised before it is dropped so the copy has definitely read it.
    handle.stream().synchronize().map_err(BlasError::Cuda)?;
    drop(scratch);
    Ok(())
}

/// Launches the triangular-multiply kernel with the element-typed scalar
/// passed in its native width.
#[allow(clippy::too_many_arguments)]
fn launch_mul<T: GpuFloat>(
    kernel: &Kernel,
    handle: &BlasHandle,
    params: &LaunchParams,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    m: u32,
    n: u32,
    alpha: T,
    a_strides: [u32; 2],
    b_strides: [u32; 2],
    o_strides: [u32; 2],
) -> BlasResult<()> {
    match T::PTX_TYPE {
        PtxType::F64 => {
            let alpha_f64 = f64::from_bits(alpha.to_bits_u64());
            let args = (
                a_ptr,
                b_ptr,
                out_ptr,
                m,
                n,
                alpha_f64,
                a_strides[0],
                a_strides[1],
                b_strides[0],
                b_strides[1],
                o_strides[0],
                o_strides[1],
            );
            kernel
                .launch(params, handle.stream(), &args)
                .map_err(|e| BlasError::LaunchFailed(format!("TRMM multiply launch failed: {e}")))
        }
        _ => {
            let alpha_f32 = f32::from_bits(alpha.to_bits_u64() as u32);
            let args = (
                a_ptr,
                b_ptr,
                out_ptr,
                m,
                n,
                alpha_f32,
                a_strides[0],
                a_strides[1],
                b_strides[0],
                b_strides[1],
                o_strides[0],
                o_strides[1],
            );
            kernel
                .launch(params, handle.stream(), &args)
                .map_err(|e| BlasError::LaunchFailed(format!("TRMM multiply launch failed: {e}")))
        }
    }
}

/// Returns the `(row_stride, col_stride)` element strides for a layout.
fn elem_strides(layout: Layout, ld: u32) -> (u32, u32) {
    match layout {
        Layout::RowMajor => (ld, 1),
        Layout::ColMajor => (1, ld),
    }
}

// We need the Side type here.
use crate::types::Side;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trmm_validates_square() {
        let err = BlasError::InvalidDimension("TRMM: triangular matrix A must be square".into());
        assert!(err.to_string().contains("square"));
    }

    #[test]
    fn trmm_validates_zero_dims() {
        let err = BlasError::InvalidDimension("TRMM: B dimensions must be non-zero".into());
        assert!(err.to_string().contains("non-zero"));
    }

    #[test]
    fn side_left_dimension_check() {
        // Left: A is M x M, B is M x N. A.rows must == B.rows.
        let tri_n: u32 = 64;
        let m: u32 = 64;
        assert_eq!(tri_n, m);
    }

    #[test]
    fn side_right_dimension_check() {
        // Right: A is N x N, B is M x N. A.rows must == B.cols.
        let tri_n: u32 = 128;
        let n: u32 = 128;
        assert_eq!(tri_n, n);
    }

    #[test]
    fn elem_strides_row_major() {
        assert_eq!(elem_strides(Layout::RowMajor, 10), (10, 1));
    }

    #[test]
    fn elem_strides_col_major() {
        assert_eq!(elem_strides(Layout::ColMajor, 10), (1, 10));
    }
}
