//! GEMV -- General matrix-vector multiplication.
//!
//! Computes `y = alpha * op(A) * x + beta * y` where `op(A)` is either `A`,
//! `A^T`, or `A^H` depending on the [`Transpose`] parameter.
//!
//! # GPU Strategy
//!
//! The kernel addresses `A` as **row-major** (`A[i][j] = ptr[i * lda + j]`)
//! for both the `NoTrans` and `Trans` cases:
//!
//! - **NoTrans**: each thread computes one element of `y` by performing a dot
//!   product of one row of `A` with `x`.
//! - **Trans**: each thread walks one column of `A` (stride `lda`), so the
//!   result is `A^T * x`.
//!
//! Column-major descriptors are **not** supported by this kernel and are
//! rejected up-front with [`BlasError::InvalidArgument`]; a caller with a
//! column-major matrix can pass its transpose as a row-major view instead.
//! All index/offset arithmetic is performed in 64-bit so matrices larger than
//! 4 GiB are addressed correctly.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{GpuFloat, Layout, MatrixDesc, Transpose};

/// Default block size for GEMV kernels.
const GEMV_BLOCK_SIZE: u32 = 256;

/// Computes `y = alpha * op(A) * x + beta * y`.
///
/// This is the general matrix-vector multiply operation. The matrix `A` can
/// optionally be transposed via the `trans` parameter.
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `trans` -- Whether to transpose `A`.
/// * `m` -- Number of rows of matrix `A`.
/// * `n` -- Number of columns of matrix `A`.
/// * `alpha` -- Scalar multiplier for `op(A) * x`.
/// * `a` -- Matrix descriptor for `A` (device memory).
/// * `x` -- Input vector (device memory).
/// * `incx` -- Stride between consecutive elements of `x`. Must be positive.
/// * `beta` -- Scalar multiplier for the existing `y`.
/// * `y` -- Input/output vector (device memory, modified in-place).
/// * `incy` -- Stride between consecutive elements of `y`. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `m` or `n` is zero.
/// Returns [`BlasError::InvalidArgument`] if `incx` or `incy` is not positive.
/// Returns [`BlasError::BufferTooSmall`] if any buffer is undersized.
/// Returns [`BlasError::PtxGeneration`] if kernel generation fails.
/// Returns [`BlasError::Cuda`] on kernel launch or driver failure.
#[allow(clippy::too_many_arguments)]
pub fn gemv<T: GpuFloat>(
    handle: &BlasHandle,
    trans: Transpose,
    m: u32,
    n: u32,
    alpha: T,
    a: &MatrixDesc<T>,
    x: &DeviceBuffer<T>,
    incx: i32,
    beta: T,
    y: &mut DeviceBuffer<T>,
    incy: i32,
) -> BlasResult<()> {
    // -- Early exit for empty dimensions --
    if m == 0 || n == 0 {
        return Ok(());
    }

    // -- Validate increments --
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(
            "incx must be positive".to_string(),
        ));
    }
    if incy <= 0 {
        return Err(BlasError::InvalidArgument(
            "incy must be positive".to_string(),
        ));
    }

    // -- Reject unsupported layout: the kernel hard-codes row-major A --
    if a.layout == Layout::ColMajor {
        return Err(BlasError::InvalidArgument(
            "gemv: ColMajor layout is not supported by this kernel; pass a RowMajor descriptor \
             (a column-major m x n matrix equals a row-major n x m view with the transpose flag \
             flipped)"
                .to_string(),
        ));
    }

    // -- Validate matrix dimensions --
    validate_gemv_dimensions(trans, m, n, a, x, incx, y, incy)?;

    // -- Determine output vector length --
    let (output_len, inner_len) = match trans {
        Transpose::NoTrans => (m, n),
        Transpose::Trans | Transpose::ConjTrans => (n, m),
    };

    // -- Generate PTX kernel --
    let ptx = generate_gemv_ptx::<T>(handle.sm_version(), trans)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "gemv")?;

    // -- Configure launch dimensions --
    let block_size = GEMV_BLOCK_SIZE;
    let grid_size = grid_size_for(output_len, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    // -- Launch --
    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.ptr,
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            m,
            n,
            a.ld,
            incx as u32,
            incy as u32,
            output_len,
            inner_len,
        ),
    )?;

    Ok(())
}

/// Validates dimensions and buffer sizes for GEMV.
#[allow(clippy::too_many_arguments)]
fn validate_gemv_dimensions<T: GpuFloat>(
    trans: Transpose,
    m: u32,
    n: u32,
    a: &MatrixDesc<T>,
    x: &DeviceBuffer<T>,
    incx: i32,
    y: &DeviceBuffer<T>,
    incy: i32,
) -> BlasResult<()> {
    // Matrix must have at least m rows and n cols
    if a.rows < m {
        return Err(BlasError::InvalidDimension(format!(
            "A.rows ({}) < m ({})",
            a.rows, m
        )));
    }
    if a.cols < n {
        return Err(BlasError::InvalidDimension(format!(
            "A.cols ({}) < n ({})",
            a.cols, n
        )));
    }
    // Row-major leading dimension must span at least a full row. `with_ld`
    // performs no validation, so guard the extent the kernel will address here.
    if a.ld < a.cols {
        return Err(BlasError::InvalidDimension(format!(
            "A.ld ({}) < A.cols ({}) for RowMajor layout",
            a.ld, a.cols
        )));
    }

    // Determine x and y lengths based on transpose
    let (x_len, y_len) = match trans {
        Transpose::NoTrans => (n, m),
        Transpose::Trans | Transpose::ConjTrans => (m, n),
    };

    // Validate x buffer size
    let x_required = required_elements(x_len, incx);
    if x.len() < x_required {
        return Err(BlasError::BufferTooSmall {
            expected: x_required,
            actual: x.len(),
        });
    }

    // Validate y buffer size
    let y_required = required_elements(y_len, incy);
    if y.len() < y_required {
        return Err(BlasError::BufferTooSmall {
            expected: y_required,
            actual: y.len(),
        });
    }

    Ok(())
}

/// Generates PTX source for the GEMV kernel.
///
/// Each thread computes one element of the output vector by performing
/// a dot product of the corresponding row (or column, for transposed case)
/// of A with x, scaled by alpha, plus beta * y.
fn generate_gemv_ptx<T: GpuFloat>(sm: SmVersion, trans: Transpose) -> BlasResult<String> {
    let suffix = T::NAME;
    let ptx_ty = T::PTX_TYPE;
    let elem_bytes = T::size_u32();
    let is_f64 = elem_bytes == 8;

    // Choose the load/store helpers based on precision
    let _kernel_name = format!("gemv_{suffix}_{}", trans_label(trans));

    KernelBuilder::new("gemv")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("m", PtxType::U32)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("incy", PtxType::U32)
        .param("output_len", PtxType::U32)
        .param("inner_len", PtxType::U32)
        .body(move |b| {
            // Compute global thread ID -- each thread handles one output element
            let gid = b.global_thread_id_x();
            let output_len = b.load_param_u32("output_len");

            // Bounds check: skip threads beyond output vector
            let gid_inner = gid.clone();
            b.if_lt_u32(gid, output_len, move |b| {
                let gid = gid_inner;
                let a_ptr = b.load_param_u64("a_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let inner_len = b.load_param_u32("inner_len");
                let lda = b.load_param_u32("lda");
                let incx = b.load_param_u32("incx");
                let incy = b.load_param_u32("incy");

                // Load alpha and beta from their bit representations
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");

                // Reinterpret bits as float
                let alpha = if is_f64 {
                    let r = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("mov.b64 {r}, {alpha_bits};"));
                    r
                } else {
                    let lo32 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("cvt.u32.u64 {lo32}, {alpha_bits};"));
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {r}, {lo32};"));
                    r
                };

                let beta = if is_f64 {
                    let r = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("mov.b64 {r}, {beta_bits};"));
                    r
                } else {
                    let lo32 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("cvt.u32.u64 {lo32}, {beta_bits};"));
                    let r = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mov.b32 {r}, {lo32};"));
                    r
                };

                // Initialize accumulator to zero
                let acc = b.alloc_reg(ptx_ty);
                if is_f64 {
                    b.raw_ptx(&format!("mov.b64 {acc}, 0d0000000000000000;"));
                } else {
                    b.raw_ptx(&format!("mov.b32 {acc}, 0f00000000;"));
                }

                // Compute base address for this row/col in A
                // For NoTrans (row-major): row gid, iterate over columns
                // For Trans (row-major): col gid, iterate over rows
                // Address of A[gid, k] = a_ptr + (gid * lda + k) * elem_bytes
                // (row-major, NoTrans)
                // Address of A[k, gid] = a_ptr + (k * lda + gid) * elem_bytes
                // (row-major, Trans)

                let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);

                // Row base offset: gid * lda * elem_bytes (NoTrans) or
                //                  gid * elem_bytes (Trans).
                //
                // The NoTrans product `gid * lda` can exceed 2^32 for a matrix
                // wider than 4 GiB, so it is formed with `mul.wide.u32` (→ u64)
                // and byte-scaled in 64-bit; a 32-bit `mul.lo` here would wrap
                // and read the wrong row.
                let row_base = if !use_trans {
                    let row_elems64 = b.mul_wide_u32_to_u64(gid.clone(), lda.clone());
                    let row_off = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!(
                        "mul.lo.u64 {row_off}, {row_elems64}, {elem_bytes};"
                    ));
                    b.add_u64(a_ptr, row_off)
                } else {
                    // a_ptr + gid * elem_bytes (gid < output_len fits a u32;
                    // byte_offset_addr scales the offset in 64-bit).
                    b.byte_offset_addr(a_ptr, gid.clone(), elem_bytes)
                };

                // Loop over inner dimension
                let loop_label = b.fresh_label("gemv_loop");
                let done_label = b.fresh_label("gemv_done");
                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k}, 0;"));

                b.label(&loop_label);

                // Check k < inner_len
                let pred_loop = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred_loop}, {k}, {inner_len};"));
                b.raw_ptx(&format!("@!{pred_loop} bra ${done_label};"));

                // Compute address of A element:
                // NoTrans: A[gid][k] => row_base + k * elem_bytes
                // Trans:   A[k][gid] => row_base + k * lda * elem_bytes
                let a_addr = if !use_trans {
                    b.byte_offset_addr(row_base.clone(), k.clone(), elem_bytes)
                } else {
                    // For trans, stride between elements is lda * elem_bytes;
                    // form it in 64-bit so a large leading dimension cannot
                    // wrap, then scale by the 64-bit element index k.
                    let lda64 = b.cvt_u32_to_u64(lda.clone());
                    let stride64 = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("mul.lo.u64 {stride64}, {lda64}, {elem_bytes};"));
                    let k64 = b.cvt_u32_to_u64(k.clone());
                    let off = b.alloc_reg(PtxType::U64);
                    b.raw_ptx(&format!("mul.lo.u64 {off}, {k64}, {stride64};"));
                    b.add_u64(row_base.clone(), off)
                };

                // Compute address of x[k * incx] in 64-bit (k * incx can exceed
                // 2^32 for a long strided vector).
                let x_elem64 = b.mul_wide_u32_to_u64(k.clone(), incx.clone());
                let x_off = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("mul.lo.u64 {x_off}, {x_elem64}, {elem_bytes};"));
                let x_addr = b.add_u64(x_ptr.clone(), x_off);

                // Load A[..][..] and x[k*incx]
                let a_val = if is_f64 {
                    b.load_global_f64(a_addr)
                } else {
                    b.load_global_f32(a_addr)
                };
                let x_val = if is_f64 {
                    b.load_global_f64(x_addr)
                } else {
                    b.load_global_f32(x_addr)
                };

                // acc += a_val * x_val  (FMA: acc = a_val * x_val + acc)
                let new_acc = if is_f64 {
                    b.fma_f64(a_val, x_val, acc.clone())
                } else {
                    b.fma_f32(a_val, x_val, acc.clone())
                };
                b.raw_ptx(&format!(
                    "mov.{} {acc}, {new_acc};",
                    if is_f64 { "f64" } else { "f32" }
                ));

                // k++
                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);

                b.label(&done_label);

                // Compute y address: y_ptr + gid * incy * elem_bytes, in 64-bit.
                let y_elem64 = b.mul_wide_u32_to_u64(gid.clone(), incy.clone());
                let y_off = b.alloc_reg(PtxType::U64);
                b.raw_ptx(&format!("mul.lo.u64 {y_off}, {y_elem64}, {elem_bytes};"));
                let y_addr = b.add_u64(y_ptr, y_off);

                // Load current y value
                let y_cur = if is_f64 {
                    b.load_global_f64(y_addr.clone())
                } else {
                    b.load_global_f32(y_addr.clone())
                };

                // result = alpha * acc + beta * y_cur
                let alpha_acc = if is_f64 {
                    let tmp = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("mul.rn.f64 {tmp}, {alpha}, {acc};"));
                    tmp
                } else {
                    // alpha * acc
                    let tmp = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {tmp}, {alpha}, {acc};"));
                    tmp
                };
                let beta_y = if is_f64 {
                    let tmp = b.alloc_reg(PtxType::F64);
                    b.raw_ptx(&format!("mul.rn.f64 {tmp}, {beta}, {y_cur};"));
                    tmp
                } else {
                    let tmp = b.alloc_reg(PtxType::F32);
                    b.raw_ptx(&format!("mul.rn.f32 {tmp}, {beta}, {y_cur};"));
                    tmp
                };
                let result = if is_f64 {
                    b.add_f64(alpha_acc, beta_y)
                } else {
                    b.add_f32(alpha_acc, beta_y)
                };

                // Store result to y
                if is_f64 {
                    b.store_global_f64(y_addr, result);
                } else {
                    b.store_global_f32(y_addr, result);
                }
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

/// Returns a short label for the transpose mode.
fn trans_label(t: Transpose) -> &'static str {
    match t {
        Transpose::NoTrans => "n",
        Transpose::Trans => "t",
        Transpose::ConjTrans => "c",
    }
}

/// Computes the minimum buffer elements required for a strided vector.
fn required_elements(n: u32, inc: i32) -> usize {
    if n == 0 {
        return 0;
    }
    1 + (n as usize - 1) * inc.unsigned_abs() as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trans_label_values() {
        assert_eq!(trans_label(Transpose::NoTrans), "n");
        assert_eq!(trans_label(Transpose::Trans), "t");
        assert_eq!(trans_label(Transpose::ConjTrans), "c");
    }

    #[test]
    fn required_elements_basic() {
        assert_eq!(required_elements(0, 1), 0);
        assert_eq!(required_elements(1, 1), 1);
        assert_eq!(required_elements(5, 1), 5);
        assert_eq!(required_elements(5, 2), 9);
    }

    #[test]
    fn gemv_ptx_generation_f32() {
        let ptx = generate_gemv_ptx::<f32>(SmVersion::Sm80, Transpose::NoTrans);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry gemv"));
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn gemv_ptx_generation_f64() {
        let ptx = generate_gemv_ptx::<f64>(SmVersion::Sm80, Transpose::Trans);
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry gemv"));
    }
}
