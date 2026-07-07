//! ELL-optimized SpMV kernel.
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is in ELLPACK (ELL) format.
//!
//! The ELL format stores data in column-major order for coalesced GPU memory
//! access: `indices[row + col*num_rows]`, `values[row + col*num_rows]`.
//! Each thread handles exactly one row, iterating over at most
//! `max_nnz_per_row` elements. Rows shorter than `max_nnz_per_row` are padded
//! with sentinel column indices (-1) and zero values.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::format::EllMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, fma_float, load_float_imm, load_global_float, mul_float, reinterpret_bits_to_float,
    store_global_float,
};

/// Default block size for ELL SpMV.
const SPMV_ELL_BLOCK: u32 = 256;

/// ELL SpMV: `y = alpha * A * x + beta * y`.
///
/// Each thread processes one row of the ELL matrix, iterating over the
/// `max_nnz_per_row` padded entries. Consecutive threads access consecutive
/// memory locations (coalesced access) due to the column-major ELL layout.
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `ell` -- Sparse ELL matrix `A`.
/// * `x` -- Dense input vector of length `A.cols()`.
/// * `y` -- Dense output vector of length `A.rows()`.
/// * `alpha` -- Scalar multiplier for `A * x`.
/// * `beta` -- Scalar multiplier for existing `y`.
///
/// # Errors
///
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
/// Returns [`SparseError::Cuda`] on kernel launch failure.
/// Returns [`SparseError::DimensionMismatch`] if vector lengths are wrong.
pub fn spmv_ell<T: GpuFloat>(
    handle: &SparseHandle,
    ell: &EllMatrix<T>,
    x: &DeviceBuffer<T>,
    y: &mut DeviceBuffer<T>,
    alpha: T,
    beta: T,
) -> SparseResult<()> {
    if ell.rows() == 0 || ell.cols() == 0 {
        return Ok(());
    }

    if x.len() < ell.cols() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "x length ({}) must be >= cols ({})",
            x.len(),
            ell.cols()
        )));
    }
    if y.len() < ell.rows() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "y length ({}) must be >= rows ({})",
            y.len(),
            ell.rows()
        )));
    }

    let ptx = emit_spmv_ell::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spmv_ell")?;

    let block_size = SPMV_ELL_BLOCK;
    let grid_size = grid_size_for(ell.rows(), block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            ell.col_idx().as_device_ptr(),
            ell.values().as_device_ptr(),
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            ell.rows(),
            ell.max_nnz_per_row(),
        ),
    )?;

    Ok(())
}

/// Generates PTX for ELL SpMV (one thread per row, coalesced access).
///
/// The kernel iterates `k = 0..max_nnz_per_row`, loading from column-major
/// layout: `col_idx[k * num_rows + row]` and `values[k * num_rows + row]`.
/// If `col_idx` is the sentinel (-1), the entry is skipped (padded).
fn emit_spmv_ell<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("spmv_ell")
        .target(sm)
        .param("col_idx_ptr", PtxType::U64)
        .param("values_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .param("max_nnz_per_row", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_rows = b.load_param_u32("num_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_rows, move |b| {
                let row = gid_inner;
                let col_idx_base = b.load_param_u64("col_idx_ptr");
                let values_base = b.load_param_u64("values_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");
                let num_rows_reg = b.load_param_u32("num_rows");
                let max_nnz = b.load_param_u32("max_nnz_per_row");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Initialize accumulator
                let acc = load_float_imm::<T>(b, 0.0);

                // Loop: k = 0 .. max_nnz_per_row
                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k}, 0;"));

                let loop_label = b.fresh_label("ell_loop");
                let done_label = b.fresh_label("ell_done");
                let skip_label = b.fresh_label("ell_skip");

                b.label(&loop_label);
                // Exit when k >= max_nnz. Inverted skip-branch (`setp.lo` ->
                // `setp.hs`) routed through the structured `branch_if` so the
                // target carries the `$`-prefix matching the `b.label` definition.
                let pred_k = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {pred_k}, {k}, {max_nnz};"));
                b.branch_if(pred_k, &done_label);

                // Compute index = k * num_rows + row (column-major ELL layout)
                let ell_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {ell_idx}, {k}, {num_rows_reg}, {row};"
                ));

                // Load col_idx[ell_idx] (i32 = 4 bytes)
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), ell_idx.clone(), 4);
                let col = b.load_global_i32(ci_addr);

                // Check sentinel: if col < 0 (i.e. col == -1), skip
                // Skip sentinel columns (col < 0). Inverted skip-branch
                // (`setp.ge` -> `setp.lt`) via the structured `branch_if`.
                let is_invalid = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lt.s32 {is_invalid}, {col}, 0;"));
                b.branch_if(is_invalid, &skip_label);

                let col_u32 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col_u32}, {col};"));

                // Load values[ell_idx]
                let v_addr = b.byte_offset_addr(values_base.clone(), ell_idx, elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                // Load x[col]
                let x_addr = b.byte_offset_addr(x_ptr.clone(), col_u32, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                // acc += val * x_val
                let new_acc = fma_float::<T>(b, val, x_val, acc.clone());
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {new_acc};"));

                b.label(&skip_label);

                // k++
                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);
                b.label(&done_label);

                // Compute y = alpha * acc + beta * y_old
                let y_addr = b.byte_offset_addr(y_ptr, row, elem_bytes);
                let y_old = load_global_float::<T>(b, y_addr.clone());

                let alpha_acc = mul_float::<T>(b, alpha, acc);
                let beta_y = mul_float::<T>(b, beta, y_old);
                let result = add_float::<T>(b, alpha_acc, beta_y);

                store_global_float::<T>(b, y_addr, result);
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

    /// The ELL SpMV kernel must assemble for sm_86 in both precisions with
    /// `$`-prefixed branch targets and no `.b64` shuffle.
    #[test]
    fn spmv_ell_f32_f64_assemble_sm86() {
        let f32_ptx = emit_spmv_ell::<f32>(SmVersion::Sm86).expect("f32 ELL PTX");
        assert_assembles_and_clean("spmv_ell_f32", &f32_ptx);

        let f64_ptx = emit_spmv_ell::<f64>(SmVersion::Sm86).expect("f64 ELL PTX");
        assert_assembles_and_clean("spmv_ell_f64", &f64_ptx);
        assert!(
            !f64_ptx.contains("0F00000000"),
            "f64 ELL kernel must not materialize an f32 0.0 immediate:\n{f64_ptx}"
        );
    }

    #[test]
    fn spmv_ell_ptx_generates_f32() {
        let ptx = emit_spmv_ell::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_ell"));
        assert!(ptx_text.contains(".target sm_80"));
    }

    #[test]
    fn spmv_ell_ptx_generates_f64() {
        let ptx = emit_spmv_ell::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_ell"));
    }

    #[test]
    fn spmv_ell_ptx_contains_sentinel_check() {
        let ptx = emit_spmv_ell::<f32>(SmVersion::Sm80);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        // The kernel should contain a signed comparison to detect the -1
        // sentinel. The structured skip-branch tests `col < 0` (`setp.lt.s32`)
        // and branches to the skip label when true.
        assert!(ptx_text.contains("setp.lt.s32"));
    }

    #[test]
    fn spmv_ell_block_size_is_reasonable() {
        let block = SPMV_ELL_BLOCK;
        assert!(block >= 128);
        assert!(block <= 1024);
        assert_eq!(SPMV_ELL_BLOCK % 32, 0);
    }

    #[test]
    fn spmv_ell_ptx_has_coalesced_pattern() {
        // The kernel should use mad.lo.u32 for computing k*num_rows+row
        let ptx = emit_spmv_ell::<f32>(SmVersion::Sm80);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains("mad.lo.u32"));
    }
}

// ---------------------------------------------------------------------------
// On-device numeric validation (feature = "gpu-tests")
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_device_tests {
    use super::*;
    use crate::format::CsrMatrix;
    use crate::gpu_test_support::{assert_close, gpu_handle};
    use crate::host_csr::{f64_to_gpu, gpu_to_f64};

    /// CPU oracle for `y = alpha * A * x + beta * y0` over a CSR matrix
    /// (row count is derived from `row_ptr`).
    fn cpu_csr_spmv(
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        x: &[f64],
        y0: &[f64],
        alpha: f64,
        beta: f64,
    ) -> Vec<f64> {
        let rows = row_ptr.len() - 1;
        let mut y = vec![0.0_f64; rows];
        for (i, slot) in y.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for k in row_ptr[i] as usize..row_ptr[i + 1] as usize {
                acc += values[k] * x[col_idx[k] as usize];
            }
            *slot = alpha * acc + beta * y0[i];
        }
        y
    }

    /// Drive the production `spmv_ell` op (building ELL from CSR via the real
    /// conversion path) and compare to the CPU oracle.
    #[allow(clippy::too_many_arguments)]
    fn run_ell<T: GpuFloat>(
        rows: u32,
        cols: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        x: &[f64],
        y0: &[f64],
        alpha: f64,
        beta: f64,
        tol: f64,
        tag: &str,
    ) {
        let Some(handle) = gpu_handle() else {
            return;
        };
        let dev_values: Vec<T> = values.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let csr = CsrMatrix::<T>::from_host(rows, cols, row_ptr, col_idx, &dev_values)
            .expect("test: build CSR");
        let ell = EllMatrix::<T>::from_csr(&csr).expect("test: build ELL");

        let dev_x: Vec<T> = x.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let dev_y: Vec<T> = y0.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let x_buf = DeviceBuffer::from_host(&dev_x).expect("test: upload x");
        let mut y_buf = DeviceBuffer::from_host(&dev_y).expect("test: upload y");

        spmv_ell::<T>(
            &handle,
            &ell,
            &x_buf,
            &mut y_buf,
            f64_to_gpu::<T>(alpha),
            f64_to_gpu::<T>(beta),
        )
        .expect("test: spmv_ell launch");
        handle.stream().synchronize().expect("test: sync");

        let mut out = vec![T::gpu_zero(); rows as usize];
        y_buf.copy_to_host(&mut out).expect("test: download y");
        let got: Vec<f64> = out.iter().map(|&v| gpu_to_f64(v)).collect();
        let want = cpu_csr_spmv(row_ptr, col_idx, values, x, y0, alpha, beta);
        assert_close(&got, &want, tol, tag);
    }

    /// Irregular row lengths force genuine ELL padding (sentinel column -1).
    fn irregular_4x5() -> (u32, u32, Vec<i32>, Vec<i32>, Vec<f64>) {
        // row0: 3 nnz, row1: 1 nnz, row2: 4 nnz, row3: 2 nnz
        let row_ptr = vec![0, 3, 4, 8, 10];
        let col_idx = vec![0, 2, 4, 1, 0, 1, 3, 4, 2, 3];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        (4, 5, row_ptr, col_idx, values)
    }

    #[test]
    fn ell_f64_alpha_beta() {
        let (r, c, rp, ci, v) = irregular_4x5();
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y0 = vec![10.0, 20.0, 30.0, 40.0];
        run_ell::<f64>(r, c, &rp, &ci, &v, &x, &y0, 1.5, -0.5, 1e-10, "ell_f64");
    }

    #[test]
    fn ell_f32_alpha_beta() {
        let (r, c, rp, ci, v) = irregular_4x5();
        let x = vec![0.5, 1.5, -2.0, 3.0, 0.25];
        let y0 = vec![1.0, 2.0, 3.0, 4.0];
        run_ell::<f32>(r, c, &rp, &ci, &v, &x, &y0, 2.0, 0.5, 1e-4, "ell_f32");
    }

    #[test]
    fn ell_f64_beta_zero() {
        let (r, c, rp, ci, v) = irregular_4x5();
        let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let y0 = vec![1e9, -1e9, 1e8, -1e8];
        run_ell::<f64>(r, c, &rp, &ci, &v, &x, &y0, 1.0, 0.0, 1e-10, "ell_beta0");
    }
}
