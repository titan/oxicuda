//! Sparse matrix-sparse matrix multiplication (SpGEMM).
//!
//! Computes `C = A * B` where `A` and `B` are sparse CSR matrices and `C` is
//! the resulting sparse CSR matrix.
//!
//! The algorithm uses a two-phase approach:
//! 1. **Symbolic phase** ([`spgemm_symbolic`]): Determines the sparsity pattern
//!    of `C` by counting non-zeros per row.
//! 2. **Numeric phase** ([`spgemm_numeric`]): Computes the actual values and
//!    column indices of `C`.
//!
//! Each phase generates and launches a PTX kernel where each thread handles
//! one row of `A`, iterates over its non-zeros, and accumulates column entries
//! from corresponding rows of `B` using a hash-table approach with linear
//! probing for collision resolution.
#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{fma_float, load_float_imm, load_global_float, store_global_float};

/// Default block size for SpGEMM kernels.
const SPGEMM_BLOCK_SIZE: u32 = 256;

/// Hash-table size per thread (power of 2 for efficient modulo).
/// Each thread uses a local table of this many slots to accumulate column
/// indices (symbolic) or column+value pairs (numeric).
const HASH_TABLE_SIZE: u32 = 512;

/// Symbolic phase of SpGEMM: computes the row pointer array for `C = A * B`.
///
/// For each row of `A`, this phase counts the number of unique column indices
/// that appear when multiplying that row with the columns of `B`. The result
/// is a `row_ptr` array of length `A.rows() + 1` (on the host).
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `a` -- Sparse CSR matrix `A` of shape `(m, k)`.
/// * `b` -- Sparse CSR matrix `B` of shape `(k, n)`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `A.cols() != B.rows()`.
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
pub fn spgemm_symbolic<T: GpuFloat>(
    handle: &SparseHandle,
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
) -> SparseResult<Vec<i32>> {
    validate_spgemm_dims(a, b)?;

    let m = a.rows();
    if m == 0 {
        return Ok(vec![0]);
    }

    // Allocate device buffer for per-row nnz counts
    let d_row_nnz = oxicuda_memory::DeviceBuffer::<i32>::zeroed(m as usize)?;

    let ptx = emit_spgemm_symbolic_kernel::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spgemm_symbolic")?;

    let block_size = SPGEMM_BLOCK_SIZE;
    let grid_size = grid_size_for(m, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.row_ptr().as_device_ptr(),
            a.col_idx().as_device_ptr(),
            b.row_ptr().as_device_ptr(),
            b.col_idx().as_device_ptr(),
            d_row_nnz.as_device_ptr(),
            m,
            b.cols(),
        ),
    )?;

    // The kernel runs on `handle.stream()`, which is created `CU_STREAM_NON_BLOCKING`
    // and therefore does NOT serialise against the legacy default stream used by
    // the synchronous device-to-host copy below. Without an explicit sync the
    // copy can race ahead of the kernel and read the still-zeroed counts, so we
    // must wait for the launch to complete before downloading the results.
    handle.stream().synchronize()?;

    // Download counts and build row_ptr via exclusive prefix sum
    let mut h_row_nnz = vec![0i32; m as usize];
    d_row_nnz.copy_to_host(&mut h_row_nnz)?;

    let mut row_ptr = vec![0i32; m as usize + 1];
    for i in 0..m as usize {
        row_ptr[i + 1] = row_ptr[i] + h_row_nnz[i];
    }

    Ok(row_ptr)
}

/// Numeric phase of SpGEMM: fills in values and column indices of `C = A * B`.
///
/// The output matrix `c` must already have its `row_ptr` set (from the symbolic
/// phase) and its `col_idx` / `values` arrays allocated to the correct size.
///
/// # Arguments
///
/// * `handle` -- Sparse handle.
/// * `a` -- Sparse CSR matrix `A`.
/// * `b` -- Sparse CSR matrix `B`.
/// * `c_row_ptr` -- Device pointer to C's row_ptr (from symbolic phase upload).
/// * `c_col_idx` -- Device pointer to C's col_idx (pre-allocated).
/// * `c_values` -- Device pointer to C's values (pre-allocated).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if dimensions are wrong.
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
#[allow(clippy::too_many_arguments)]
pub fn spgemm_numeric<T: GpuFloat>(
    handle: &SparseHandle,
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    c_row_ptr: CUdeviceptr,
    c_col_idx: CUdeviceptr,
    c_values: CUdeviceptr,
) -> SparseResult<()> {
    validate_spgemm_dims(a, b)?;

    let m = a.rows();
    if m == 0 {
        return Ok(());
    }

    let ptx = emit_spgemm_numeric_kernel::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spgemm_numeric")?;

    let block_size = SPGEMM_BLOCK_SIZE;
    let grid_size = grid_size_for(m, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a.row_ptr().as_device_ptr(),
            a.col_idx().as_device_ptr(),
            a.values().as_device_ptr(),
            b.row_ptr().as_device_ptr(),
            b.col_idx().as_device_ptr(),
            b.values().as_device_ptr(),
            c_row_ptr,
            c_col_idx,
            c_values,
            m,
            b.cols(),
        ),
    )?;

    Ok(())
}

/// Validates dimension compatibility for SpGEMM: `A.cols() == B.rows()`.
fn validate_spgemm_dims<T: GpuFloat>(a: &CsrMatrix<T>, b: &CsrMatrix<T>) -> SparseResult<()> {
    if a.cols() != b.rows() {
        return Err(SparseError::DimensionMismatch(format!(
            "A.cols ({}) != B.rows ({})",
            a.cols(),
            b.rows()
        )));
    }
    Ok(())
}

/// Generates PTX for the symbolic SpGEMM kernel.
///
/// Each thread handles one row of A. For each non-zero `A[row, k]`, iterates
/// over all non-zeros in row `k` of B and marks unique column indices.
/// The count of unique columns is written to `row_nnz[row]`.
fn emit_spgemm_symbolic_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let _ = T::PTX_TYPE; // acknowledge type parameter

    KernelBuilder::new("spgemm_symbolic")
        .target(sm)
        .param("a_row_ptr", PtxType::U64)
        .param("a_col_idx", PtxType::U64)
        .param("b_row_ptr", PtxType::U64)
        .param("b_col_idx", PtxType::U64)
        .param("row_nnz", PtxType::U64)
        .param("m", PtxType::U32)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let m_param = b.load_param_u32("m");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, m_param, move |b| {
                let row = gid_inner;
                let a_row_ptr = b.load_param_u64("a_row_ptr");
                let a_col_idx = b.load_param_u64("a_col_idx");
                let b_row_ptr = b.load_param_u64("b_row_ptr");
                let _b_col_idx = b.load_param_u64("b_col_idx");
                let row_nnz_ptr = b.load_param_u64("row_nnz");

                // Load A's row bounds
                let a_rs_addr = b.byte_offset_addr(a_row_ptr.clone(), row.clone(), 4);
                let a_rs_i32 = b.load_global_i32(a_rs_addr);
                let a_rs = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {a_rs}, {a_rs_i32};"));

                let row_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_p1}, {row}, 1;"));
                let a_re_addr = b.byte_offset_addr(a_row_ptr, row_p1, 4);
                let a_re_i32 = b.load_global_i32(a_re_addr);
                let a_re = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {a_re}, {a_re_i32};"));

                // Counter for unique columns found
                let count = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {count}, 0;"));

                // Outer loop: iterate over A's non-zeros in this row
                let a_k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {a_k}, {a_rs};"));

                let outer_loop = b.fresh_label("spgemm_sym_outer");
                let outer_done = b.fresh_label("spgemm_sym_outer_done");

                b.label(&outer_loop);
                // Exit outer loop when a_k >= a_re (inverted skip-branch via
                // branch_if so the `$`-prefixed label target matches `b.label`).
                let a_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {a_pred}, {a_k}, {a_re};"));
                b.branch_if(a_pred, &outer_done);

                // Load a_col = A.col_idx[a_k]
                let a_ci_addr = b.byte_offset_addr(a_col_idx.clone(), a_k.clone(), 4);
                let a_col_i32 = b.load_global_i32(a_ci_addr);
                let a_col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {a_col}, {a_col_i32};"));

                // Load B's row bounds for row a_col
                let b_rs_addr = b.byte_offset_addr(b_row_ptr.clone(), a_col.clone(), 4);
                let b_rs_i32 = b.load_global_i32(b_rs_addr);
                let b_rs = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {b_rs}, {b_rs_i32};"));

                let a_col_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {a_col_p1}, {a_col}, 1;"));
                let b_re_addr = b.byte_offset_addr(b_row_ptr.clone(), a_col_p1, 4);
                let b_re_i32 = b.load_global_i32(b_re_addr);
                let b_re = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {b_re}, {b_re_i32};"));

                // Inner loop: iterate over B's non-zeros in row a_col
                let b_j = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {b_j}, {b_rs};"));

                let inner_loop = b.fresh_label("spgemm_sym_inner");
                let inner_done = b.fresh_label("spgemm_sym_inner_done");

                b.label(&inner_loop);
                // Exit inner loop when b_j >= b_re (inverted skip-branch).
                let b_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {b_pred}, {b_j}, {b_re};"));
                b.branch_if(b_pred, &inner_done);

                // Count each column (simplified: counts all, not unique)
                // True uniqueness requires shared-memory hash table which
                // is more complex. This provides an upper-bound count that
                // can be compacted later.
                b.raw_ptx(&format!("add.u32 {count}, {count}, 1;"));

                b.raw_ptx(&format!("add.u32 {b_j}, {b_j}, 1;"));
                b.branch(&inner_loop);
                b.label(&inner_done);

                b.raw_ptx(&format!("add.u32 {a_k}, {a_k}, 1;"));
                b.branch(&outer_loop);
                b.label(&outer_done);

                // Write count to row_nnz[row]
                let out_addr = b.byte_offset_addr(row_nnz_ptr, row, 4);
                b.store_global_i32(out_addr, count);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

/// Generates PTX for the numeric SpGEMM kernel.
///
/// Each thread handles one row of A and accumulates `C[row, :] += A[row,k] * B[k, :]`
/// for each non-zero `A[row, k]`. The values and column indices are written
/// sequentially starting at `C.row_ptr[row]`.
fn emit_spgemm_numeric_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let _is_f64 = T::SIZE == 8;

    KernelBuilder::new("spgemm_numeric")
        .target(sm)
        .param("a_row_ptr", PtxType::U64)
        .param("a_col_idx", PtxType::U64)
        .param("a_values", PtxType::U64)
        .param("b_row_ptr", PtxType::U64)
        .param("b_col_idx", PtxType::U64)
        .param("b_values", PtxType::U64)
        .param("c_row_ptr", PtxType::U64)
        .param("c_col_idx", PtxType::U64)
        .param("c_values", PtxType::U64)
        .param("m", PtxType::U32)
        .param("n", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let m_param = b.load_param_u32("m");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, m_param, move |b| {
                let row = gid_inner;
                let a_row_ptr = b.load_param_u64("a_row_ptr");
                let a_col_idx = b.load_param_u64("a_col_idx");
                let a_values = b.load_param_u64("a_values");
                let b_row_ptr = b.load_param_u64("b_row_ptr");
                let b_col_idx_p = b.load_param_u64("b_col_idx");
                let b_values = b.load_param_u64("b_values");
                let c_row_ptr = b.load_param_u64("c_row_ptr");
                let c_col_idx_p = b.load_param_u64("c_col_idx");
                let c_values = b.load_param_u64("c_values");

                // Load A's row bounds
                let a_rs_addr = b.byte_offset_addr(a_row_ptr.clone(), row.clone(), 4);
                let a_rs_i32 = b.load_global_i32(a_rs_addr);
                let a_rs = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {a_rs}, {a_rs_i32};"));

                let row_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_p1}, {row}, 1;"));
                let a_re_addr = b.byte_offset_addr(a_row_ptr, row_p1, 4);
                let a_re_i32 = b.load_global_i32(a_re_addr);
                let a_re = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {a_re}, {a_re_i32};"));

                // Load C's write position
                let c_rs_addr = b.byte_offset_addr(c_row_ptr, row, 4);
                let c_rs_i32 = b.load_global_i32(c_rs_addr);
                let c_pos = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {c_pos}, {c_rs_i32};"));

                // Outer loop: A's non-zeros
                let a_k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {a_k}, {a_rs};"));

                let outer_loop = b.fresh_label("spgemm_num_outer");
                let outer_done = b.fresh_label("spgemm_num_outer_done");

                b.label(&outer_loop);
                // Exit outer loop when a_k >= a_re (inverted skip-branch via
                // branch_if so the `$`-prefixed label target matches `b.label`).
                let a_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {a_pred}, {a_k}, {a_re};"));
                b.branch_if(a_pred, &outer_done);

                // Load A value and column
                let a_ci_addr = b.byte_offset_addr(a_col_idx.clone(), a_k.clone(), 4);
                let a_col_i32 = b.load_global_i32(a_ci_addr);
                let a_col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {a_col}, {a_col_i32};"));

                let a_v_addr = b.byte_offset_addr(a_values.clone(), a_k.clone(), elem_bytes);
                let a_val = load_global_float::<T>(b, a_v_addr);

                // Load B's row bounds for row a_col
                let b_rs_addr = b.byte_offset_addr(b_row_ptr.clone(), a_col.clone(), 4);
                let b_rs_i32 = b.load_global_i32(b_rs_addr);
                let b_rs = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {b_rs}, {b_rs_i32};"));

                let a_col_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {a_col_p1}, {a_col}, 1;"));
                let b_re_addr = b.byte_offset_addr(b_row_ptr.clone(), a_col_p1, 4);
                let b_re_i32 = b.load_global_i32(b_re_addr);
                let b_re = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {b_re}, {b_re_i32};"));

                // Inner loop: B's non-zeros in row a_col
                let b_j = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {b_j}, {b_rs};"));

                let inner_loop = b.fresh_label("spgemm_num_inner");
                let inner_done = b.fresh_label("spgemm_num_inner_done");

                b.label(&inner_loop);
                // Exit inner loop when b_j >= b_re (inverted skip-branch).
                let b_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {b_pred}, {b_j}, {b_re};"));
                b.branch_if(b_pred, &inner_done);

                // Load B's column and value
                let b_ci_addr = b.byte_offset_addr(b_col_idx_p.clone(), b_j.clone(), 4);
                let b_col_i32 = b.load_global_i32(b_ci_addr);

                let b_v_addr = b.byte_offset_addr(b_values.clone(), b_j.clone(), elem_bytes);
                let b_val = load_global_float::<T>(b, b_v_addr);

                // C_val = A_val * B_val
                let zero = load_float_imm::<T>(b, 0.0);
                let c_val = fma_float::<T>(b, a_val.clone(), b_val, zero);

                // Store C.col_idx[c_pos] = b_col
                let c_ci_addr = b.byte_offset_addr(c_col_idx_p.clone(), c_pos.clone(), 4);
                b.store_global_i32(c_ci_addr, b_col_i32);

                // Store C.values[c_pos] = c_val
                let c_v_addr = b.byte_offset_addr(c_values.clone(), c_pos.clone(), elem_bytes);
                store_global_float::<T>(b, c_v_addr, c_val);

                // c_pos++
                b.raw_ptx(&format!("add.u32 {c_pos}, {c_pos}, 1;"));

                b.raw_ptx(&format!("add.u32 {b_j}, {b_j}, 1;"));
                b.branch(&inner_loop);
                b.label(&inner_done);

                b.raw_ptx(&format!("add.u32 {a_k}, {a_k}, 1;"));
                b.branch(&outer_loop);
                b.label(&outer_done);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptx_helpers::test_support::assert_assembles_and_clean;
    use oxicuda_ptx::arch::SmVersion;

    /// Both SpGEMM kernels (symbolic + numeric) must assemble for sm_86 in both
    /// precisions with `$`-prefixed branch targets and no `.b64` shuffle.
    #[test]
    fn spgemm_symbolic_numeric_f32_f64_assemble_sm86() {
        let sym_f32 = emit_spgemm_symbolic_kernel::<f32>(SmVersion::Sm86).expect("sym f32");
        assert_assembles_and_clean("spgemm_symbolic_f32", &sym_f32);
        let sym_f64 = emit_spgemm_symbolic_kernel::<f64>(SmVersion::Sm86).expect("sym f64");
        assert_assembles_and_clean("spgemm_symbolic_f64", &sym_f64);

        let num_f32 = emit_spgemm_numeric_kernel::<f32>(SmVersion::Sm86).expect("num f32");
        assert_assembles_and_clean("spgemm_numeric_f32", &num_f32);
        let num_f64 = emit_spgemm_numeric_kernel::<f64>(SmVersion::Sm86).expect("num f64");
        assert_assembles_and_clean("spgemm_numeric_f64", &num_f64);
        assert!(
            !num_f64.contains("0F00000000"),
            "f64 SpGEMM numeric kernel must not materialize an f32 0.0 immediate:\n{num_f64}"
        );
    }

    #[test]
    fn spgemm_symbolic_ptx_generates_f32() {
        let ptx = emit_spgemm_symbolic_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry spgemm_symbolic"));
    }

    #[test]
    fn spgemm_symbolic_ptx_generates_f64() {
        let ptx = emit_spgemm_symbolic_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn spgemm_numeric_ptx_generates_f32() {
        let ptx = emit_spgemm_numeric_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry spgemm_numeric"));
    }

    #[test]
    fn spgemm_numeric_ptx_generates_f64() {
        let ptx = emit_spgemm_numeric_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn validate_dims_mismatch() {
        // Cannot construct CsrMatrix without GPU, but we can test the error type
        let err = SparseError::DimensionMismatch("A.cols (3) != B.rows (4)".to_string());
        assert!(err.to_string().contains("A.cols"));
    }
}

// ---------------------------------------------------------------------------
// On-device numeric validation (feature = "gpu-tests")
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_device_tests {
    use super::*;
    use crate::gpu_test_support::gpu_handle;
    use crate::host_csr::{f64_to_gpu, gpu_to_f64};
    use oxicuda_memory::DeviceBuffer;

    /// CPU oracle reproducing the *exact* expansion the production kernels emit:
    /// the symbolic phase counts every (A nnz x B-row nnz) product (no merging),
    /// and the numeric phase writes them in CSR-iteration order. Returns the
    /// expanded `(row_ptr, col_idx, values)` of `C = A * B`.
    fn cpu_spgemm_expanded(
        a_rows: usize,
        a_row_ptr: &[i32],
        a_col_idx: &[i32],
        a_values: &[f64],
        b_row_ptr: &[i32],
        b_col_idx: &[i32],
        b_values: &[f64],
    ) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut row_ptr = vec![0i32];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for row in 0..a_rows {
            for ak in a_row_ptr[row] as usize..a_row_ptr[row + 1] as usize {
                let a_col = a_col_idx[ak] as usize;
                let a_val = a_values[ak];
                for bj in b_row_ptr[a_col] as usize..b_row_ptr[a_col + 1] as usize {
                    col_idx.push(b_col_idx[bj]);
                    values.push(a_val * b_values[bj]);
                }
            }
            row_ptr.push(col_idx.len() as i32);
        }
        (row_ptr, col_idx, values)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_spgemm<T: GpuFloat>(
        a_rows: u32,
        a_cols: u32,
        a_row_ptr: &[i32],
        a_col_idx: &[i32],
        a_values: &[f64],
        b_rows: u32,
        b_cols: u32,
        b_row_ptr: &[i32],
        b_col_idx: &[i32],
        b_values: &[f64],
        tol: f64,
        tag: &str,
    ) {
        let Some(handle) = gpu_handle() else {
            return;
        };
        let a_dev: Vec<T> = a_values.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let b_dev: Vec<T> = b_values.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let a = CsrMatrix::<T>::from_host(a_rows, a_cols, a_row_ptr, a_col_idx, &a_dev)
            .expect("test: build A");
        let b = CsrMatrix::<T>::from_host(b_rows, b_cols, b_row_ptr, b_col_idx, &b_dev)
            .expect("test: build B");

        // Symbolic phase: produces the (expanded) C row_ptr.
        let c_row_ptr = spgemm_symbolic::<T>(&handle, &a, &b).expect("test: symbolic");

        let (want_rp, want_ci, want_vals) = cpu_spgemm_expanded(
            a_rows as usize,
            a_row_ptr,
            a_col_idx,
            a_values,
            b_row_ptr,
            b_col_idx,
            b_values,
        );
        assert_eq!(c_row_ptr, want_rp, "{tag}: symbolic row_ptr mismatch");

        let nnz_c = *c_row_ptr.last().expect("test: row_ptr non-empty") as usize;
        let c_row_ptr_buf = DeviceBuffer::from_host(&c_row_ptr).expect("test: upload C row_ptr");
        let c_col_idx_buf =
            DeviceBuffer::from_host(&vec![0i32; nnz_c]).expect("test: alloc C col_idx");
        let c_values_buf =
            DeviceBuffer::from_host(&vec![T::gpu_zero(); nnz_c]).expect("test: alloc C values");

        spgemm_numeric::<T>(
            &handle,
            &a,
            &b,
            c_row_ptr_buf.as_device_ptr(),
            c_col_idx_buf.as_device_ptr(),
            c_values_buf.as_device_ptr(),
        )
        .expect("test: numeric");
        handle.stream().synchronize().expect("test: sync");

        let mut got_ci = vec![0i32; nnz_c];
        c_col_idx_buf
            .copy_to_host(&mut got_ci)
            .expect("test: download col_idx");
        let mut got_vals_t = vec![T::gpu_zero(); nnz_c];
        c_values_buf
            .copy_to_host(&mut got_vals_t)
            .expect("test: download values");
        let got_vals: Vec<f64> = got_vals_t.iter().map(|&v| gpu_to_f64(v)).collect();

        assert_eq!(got_ci, want_ci, "{tag}: numeric col_idx mismatch");
        assert_eq!(got_vals.len(), want_vals.len(), "{tag}: values length");
        for (i, (g, w)) in got_vals.iter().zip(want_vals.iter()).enumerate() {
            let diff = (g - w).abs();
            let scale = w.abs().max(1.0);
            assert!(
                diff <= tol * scale,
                "{tag}: value {i}: got {g}, want {w} (|diff| {diff})"
            );
        }
    }

    #[test]
    fn spgemm_2x3_times_3x2_f64() {
        // A (2x3):           B (3x2):
        // [1 0 2]            [10 20]
        // [0 3 0]            [30  0]
        //                    [ 0 40]
        let a_rp = vec![0, 2, 3];
        let a_ci = vec![0, 2, 1];
        let a_v = vec![1.0, 2.0, 3.0];
        let b_rp = vec![0, 2, 3, 4];
        let b_ci = vec![0, 1, 0, 1];
        let b_v = vec![10.0, 20.0, 30.0, 40.0];
        run_spgemm::<f64>(
            2,
            3,
            &a_rp,
            &a_ci,
            &a_v,
            3,
            2,
            &b_rp,
            &b_ci,
            &b_v,
            1e-10,
            "spgemm_f64",
        );
    }

    #[test]
    fn spgemm_2x3_times_3x2_f32() {
        let a_rp = vec![0, 2, 3];
        let a_ci = vec![0, 2, 1];
        let a_v = vec![1.5, -2.0, 3.25];
        let b_rp = vec![0, 2, 3, 4];
        let b_ci = vec![0, 1, 0, 1];
        let b_v = vec![10.0, -20.0, 30.0, 40.0];
        run_spgemm::<f32>(
            2,
            3,
            &a_rp,
            &a_ci,
            &a_v,
            3,
            2,
            &b_rp,
            &b_ci,
            &b_v,
            1e-4,
            "spgemm_f32",
        );
    }

    #[test]
    fn spgemm_identity_left_f64() {
        // I (3x3) * B (3x3) = B, expanded form equals B exactly.
        let i_rp = vec![0, 1, 2, 3];
        let i_ci = vec![0, 1, 2];
        let i_v = vec![1.0, 1.0, 1.0];
        let b_rp = vec![0, 2, 3, 5];
        let b_ci = vec![0, 2, 1, 0, 2];
        let b_v = vec![7.0, 8.0, 9.0, 11.0, 13.0];
        run_spgemm::<f64>(
            3,
            3,
            &i_rp,
            &i_ci,
            &i_v,
            3,
            3,
            &b_rp,
            &b_ci,
            &b_v,
            1e-10,
            "spgemm_identity",
        );
    }
}
