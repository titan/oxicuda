//! BSR SpMV kernel.
//!
//! Computes `y = alpha * A * x + beta * y` where `A` is in Block Sparse Row
//! (BSR) format. The kernel exploits dense sub-blocks for higher arithmetic
//! intensity compared to scalar CSR SpMV.
//!
//! Each thread block handles one block-row. Within a block-row, threads
//! cooperate to multiply the dense `block_dim x block_dim` sub-blocks by
//! the corresponding segments of `x`, accumulating into a shared-memory
//! partial result vector of length `block_dim`.

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SparseError, SparseResult};
use crate::format::BsrMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{
    add_float, fma_float, load_float_imm, load_global_float, mul_float, reinterpret_bits_to_float,
    store_global_float,
};

/// Maximum threads per block for BSR SpMV.
/// Each thread block handles one block-row, with threads distributed
/// across the dense block elements.
const SPMV_BSR_MAX_BLOCK: u32 = 256;

/// BSR SpMV: `y = alpha * A * x + beta * y`.
///
/// The kernel launches one thread block per block-row. Within each thread
/// block, threads iterate over the non-zero blocks in that block-row,
/// performing dense block-vector multiplication.
///
/// # Arguments
///
/// * `handle` -- Sparse handle providing stream and device context.
/// * `bsr` -- Sparse BSR matrix `A`.
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
pub fn spmv_bsr<T: GpuFloat>(
    handle: &SparseHandle,
    bsr: &BsrMatrix<T>,
    x: &DeviceBuffer<T>,
    y: &mut DeviceBuffer<T>,
    alpha: T,
    beta: T,
) -> SparseResult<()> {
    if bsr.rows() == 0 || bsr.cols() == 0 {
        return Ok(());
    }

    if x.len() < bsr.cols() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "x length ({}) must be >= cols ({})",
            x.len(),
            bsr.cols()
        )));
    }
    if y.len() < bsr.rows() as usize {
        return Err(SparseError::DimensionMismatch(format!(
            "y length ({}) must be >= rows ({})",
            y.len(),
            bsr.rows()
        )));
    }

    let block_dim = bsr.block_dim();
    let block_rows = bsr.block_rows();

    // Threads per block: at least block_dim threads (one per row in the block),
    // rounded up to a multiple of 32 (warp size), capped at SPMV_BSR_MAX_BLOCK.
    let threads_per_block = (block_dim.div_ceil(32) * 32).min(SPMV_BSR_MAX_BLOCK);

    let ptx = emit_spmv_bsr::<T>(handle.sm_version(), block_dim)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "spmv_bsr")?;

    // One thread block per block-row
    let params = LaunchParams::new(block_rows, threads_per_block);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            bsr.row_ptr().as_device_ptr(),
            bsr.col_idx().as_device_ptr(),
            bsr.values().as_device_ptr(),
            x.as_device_ptr(),
            y.as_device_ptr(),
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            bsr.rows(),
            block_dim,
        ),
    )?;

    Ok(())
}

/// Generates PTX for BSR SpMV.
///
/// Each thread block handles one block-row. Thread `tid` within the block is
/// responsible for the `tid`-th row within the dense block (if `tid < block_dim`).
/// For each non-zero block in the block-row, the thread computes a dot product
/// of its row of the dense block with the corresponding segment of `x`.
fn emit_spmv_bsr<T: GpuFloat>(sm: SmVersion, _block_dim: u32) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("spmv_bsr")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("num_rows", PtxType::U32)
        .param("block_dim", PtxType::U32)
        .body(move |b| {
            // Each thread block handles one block-row.
            // blockIdx.x = block-row index.
            // threadIdx.x = local row within the block (if < block_dim).
            let block_row = b.block_id_x();
            let tid = b.thread_id_x();
            let block_dim_reg = b.load_param_u32("block_dim");

            // Only threads with tid < block_dim participate
            let tid_inner = tid.clone();
            let block_row_inner = block_row.clone();
            b.if_lt_u32(tid, block_dim_reg, move |b| {
                let tid = tid_inner;
                let block_row = block_row_inner;
                let block_dim_reg = b.load_param_u32("block_dim");

                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load row_ptr[block_row] and row_ptr[block_row+1]
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), block_row.clone(), 4);
                let blk_start_i32 = b.load_global_i32(rp_addr);
                let blk_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {blk_start}, {blk_start_i32};"));

                let block_row_plus_1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {block_row_plus_1}, {block_row}, 1;"));
                let rp_addr_next = b.byte_offset_addr(row_ptr_base, block_row_plus_1, 4);
                let blk_end_i32 = b.load_global_i32(rp_addr_next);
                let blk_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {blk_end}, {blk_end_i32};"));

                // Initialize accumulator for this thread's row
                let acc = load_float_imm::<T>(b, 0.0);

                // block_dim^2 = elements per dense block
                let blk_sq = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {blk_sq}, {block_dim_reg}, {block_dim_reg};"
                ));

                // Loop over non-zero blocks: blk_idx = blk_start .. blk_end
                let blk_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {blk_idx}, {blk_start};"));

                let blk_loop = b.fresh_label("bsr_blk_loop");
                let blk_done = b.fresh_label("bsr_blk_done");

                b.label(&blk_loop);
                // Exit the loop when blk_idx >= blk_end. The original skip-branch
                // (`@!pred bra`) used `setp.lo`; invert to `setp.hs` and use the
                // structured `branch_if` so the target matches the `$`-prefixed
                // `b.label` definition (a bare `bra blk_done` is rejected by ptxas
                // as an unknown symbol).
                let pred_blk = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {pred_blk}, {blk_idx}, {blk_end};"));
                b.branch_if(pred_blk, &blk_done);

                // Load block column index
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), blk_idx.clone(), 4);
                let blk_col_i32 = b.load_global_i32(ci_addr);
                let blk_col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {blk_col}, {blk_col_i32};"));

                // Base offset of this block in values array:
                // values_offset = blk_idx * block_dim^2
                let val_block_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {val_block_offset}, {blk_idx}, {blk_sq};"
                ));

                // Row offset within block: tid * block_dim
                let row_in_block_offset = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {row_in_block_offset}, {tid}, {block_dim_reg};"
                ));

                // x column base: blk_col * block_dim
                let x_col_base = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mul.lo.u32 {x_col_base}, {blk_col}, {block_dim_reg};"
                ));

                // Inner loop over block columns: j = 0 .. block_dim
                let j = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {j}, 0;"));

                let inner_loop = b.fresh_label("bsr_inner");
                let inner_done = b.fresh_label("bsr_inner_done");

                b.label(&inner_loop);
                // Exit the inner loop when j >= block_dim (inverted skip-branch).
                let pred_j = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {pred_j}, {j}, {block_dim_reg};"));
                b.branch_if(pred_j, &inner_done);

                // values index = val_block_offset + row_in_block_offset + j
                let val_flat = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "add.u32 {val_flat}, {val_block_offset}, {row_in_block_offset};"
                ));
                let val_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {val_idx}, {val_flat}, {j};"));

                let v_addr = b.byte_offset_addr(values_base.clone(), val_idx, elem_bytes);
                let val = load_global_float::<T>(b, v_addr);

                // x index = x_col_base + j
                let x_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {x_idx}, {x_col_base}, {j};"));

                let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx, elem_bytes);
                let x_val = load_global_float::<T>(b, x_addr);

                // acc += val * x_val
                let new_acc = fma_float::<T>(b, val, x_val, acc.clone());
                let mov_suffix = if is_f64 { "f64" } else { "f32" };
                b.raw_ptx(&format!("mov.{mov_suffix} {acc}, {new_acc};"));

                // j++
                b.raw_ptx(&format!("add.u32 {j}, {j}, 1;"));
                b.branch(&inner_loop);
                b.label(&inner_done);

                // blk_idx++
                b.raw_ptx(&format!("add.u32 {blk_idx}, {blk_idx}, 1;"));
                b.branch(&blk_loop);
                b.label(&blk_done);

                // Compute global row index = block_row * block_dim + tid
                let global_row = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!(
                    "mad.lo.u32 {global_row}, {block_row}, {block_dim_reg}, {tid};"
                ));

                // Write y = alpha * acc + beta * y_old
                let y_addr = b.byte_offset_addr(y_ptr, global_row, elem_bytes);
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

    /// The BSR SpMV kernel must assemble for sm_86 in both precisions, with the
    /// branch labels carrying the `$`-prefix (so `bra`/`@pred bra` resolve) and
    /// no illegal `.b64` warp shuffle. Regression guard for the bare-label
    /// "Unknown symbol 'L__…'" ptxas failure.
    #[test]
    fn spmv_bsr_f32_f64_assemble_sm86() {
        let f32_ptx = emit_spmv_bsr::<f32>(SmVersion::Sm86, 4).expect("f32 BSR PTX");
        assert_assembles_and_clean("spmv_bsr_f32", &f32_ptx);

        let f64_ptx = emit_spmv_bsr::<f64>(SmVersion::Sm86, 4).expect("f64 BSR PTX");
        assert_assembles_and_clean("spmv_bsr_f64", &f64_ptx);
        assert!(
            !f64_ptx.contains("0F00000000"),
            "f64 BSR kernel must not materialize an f32 0.0 immediate:\n{f64_ptx}"
        );
    }

    #[test]
    fn spmv_bsr_ptx_generates_f32() {
        let ptx = emit_spmv_bsr::<f32>(SmVersion::Sm80, 2);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_bsr"));
        assert!(ptx_text.contains(".target sm_80"));
    }

    #[test]
    fn spmv_bsr_ptx_generates_f64() {
        let ptx = emit_spmv_bsr::<f64>(SmVersion::Sm80, 4);
        assert!(ptx.is_ok());
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_text.contains(".entry spmv_bsr"));
    }

    #[test]
    fn spmv_bsr_ptx_block_sizes() {
        // Various common block sizes should generate valid PTX
        for bd in [2, 4, 8] {
            let ptx = emit_spmv_bsr::<f32>(SmVersion::Sm80, bd);
            assert!(ptx.is_ok(), "BSR PTX generation failed for block_dim={bd}");
        }
    }

    #[test]
    fn spmv_bsr_threads_per_block() {
        // Verify threads_per_block calculation
        for block_dim in [2u32, 4, 8, 16, 32, 64] {
            let threads = (block_dim.div_ceil(32) * 32).min(SPMV_BSR_MAX_BLOCK);
            assert!(threads >= block_dim);
            assert_eq!(threads % 32, 0);
            assert!(threads <= SPMV_BSR_MAX_BLOCK);
        }
    }

    #[test]
    fn spmv_bsr_ptx_contains_block_multiply() {
        let ptx = emit_spmv_bsr::<f32>(SmVersion::Sm80, 4);
        let ptx_text = ptx.expect("test: PTX gen should succeed");
        // Should contain nested loop structure (block multiply)
        assert!(ptx_text.contains("bsr_blk_loop"));
        assert!(ptx_text.contains("bsr_inner"));
    }
}

// ---------------------------------------------------------------------------
// On-device numeric validation (feature = "gpu-tests")
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_device_tests {
    use super::*;
    use crate::gpu_test_support::{assert_close, gpu_handle};
    use crate::host_csr::{f64_to_gpu, gpu_to_f64};

    /// CPU oracle for blocked SpMV `y = alpha * A * x + beta * y0`, where the
    /// dense blocks are stored row-major: `value[blk*bd^2 + r*bd + c]` is the
    /// `(r, c)` entry of the `blk`-th non-zero block.
    #[allow(clippy::too_many_arguments)]
    fn cpu_bsr_spmv(
        rows: usize,
        block_dim: usize,
        row_ptr: &[i32],
        col_idx: &[i32],
        values: &[f64],
        x: &[f64],
        y0: &[f64],
        alpha: f64,
        beta: f64,
    ) -> Vec<f64> {
        let block_rows = rows / block_dim;
        let bd2 = block_dim * block_dim;
        let mut acc = vec![0.0_f64; rows];
        for block_row in 0..block_rows {
            for blk in row_ptr[block_row] as usize..row_ptr[block_row + 1] as usize {
                let blk_col = col_idx[blk] as usize;
                for r in 0..block_dim {
                    for c in 0..block_dim {
                        let v = values[blk * bd2 + r * block_dim + c];
                        acc[block_row * block_dim + r] += v * x[blk_col * block_dim + c];
                    }
                }
            }
        }
        (0..rows).map(|g| alpha * acc[g] + beta * y0[g]).collect()
    }

    /// Drive the production `spmv_bsr` op and compare to the CPU oracle.
    #[allow(clippy::too_many_arguments)]
    fn run_bsr<T: GpuFloat>(
        rows: u32,
        cols: u32,
        block_dim: u32,
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
        let bsr = BsrMatrix::<T>::from_host(rows, cols, block_dim, row_ptr, col_idx, &dev_values)
            .expect("test: build BSR");

        let dev_x: Vec<T> = x.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let dev_y: Vec<T> = y0.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let x_buf = DeviceBuffer::from_host(&dev_x).expect("test: upload x");
        let mut y_buf = DeviceBuffer::from_host(&dev_y).expect("test: upload y");

        spmv_bsr::<T>(
            &handle,
            &bsr,
            &x_buf,
            &mut y_buf,
            f64_to_gpu::<T>(alpha),
            f64_to_gpu::<T>(beta),
        )
        .expect("test: spmv_bsr launch");
        handle.stream().synchronize().expect("test: sync");

        let mut out = vec![T::gpu_zero(); rows as usize];
        y_buf.copy_to_host(&mut out).expect("test: download y");
        let got: Vec<f64> = out.iter().map(|&v| gpu_to_f64(v)).collect();
        let want = cpu_bsr_spmv(
            rows as usize,
            block_dim as usize,
            row_ptr,
            col_idx,
            values,
            x,
            y0,
            alpha,
            beta,
        );
        assert_close(&got, &want, tol, tag);
    }

    /// 4x4 matrix, block_dim 2 (2x2 grid of 2x2 blocks), 3 non-zero blocks.
    fn bsr_4x4_b2() -> (u32, u32, u32, Vec<i32>, Vec<i32>, Vec<f64>) {
        // block_row 0: blocks at block_col 0 and 1
        // block_row 1: block at block_col 1
        let row_ptr = vec![0, 2, 3];
        let col_idx = vec![0, 1, 1];
        let values = vec![
            1.0, 2.0, 3.0, 4.0, // block (0,0)
            5.0, 6.0, 7.0, 8.0, // block (0,1)
            9.0, 10.0, 11.0, 12.0, // block (1,1)
        ];
        (4, 4, 2, row_ptr, col_idx, values)
    }

    /// 6x6 matrix, block_dim 3 (2x2 grid of 3x3 blocks), 3 non-zero blocks.
    fn bsr_6x6_b3() -> (u32, u32, u32, Vec<i32>, Vec<i32>, Vec<f64>) {
        let row_ptr = vec![0, 2, 3];
        let col_idx = vec![0, 1, 0];
        let mut values = Vec::new();
        for blk in 0..3 {
            for r in 0..3 {
                for c in 0..3 {
                    values.push(1.0 + (blk * 9 + r * 3 + c) as f64 * 0.5);
                }
            }
        }
        (6, 6, 3, row_ptr, col_idx, values)
    }

    #[test]
    fn bsr_b2_f64_alpha_beta() {
        let (r, c, bd, rp, ci, v) = bsr_4x4_b2();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y0 = vec![10.0, 20.0, 30.0, 40.0];
        run_bsr::<f64>(
            r,
            c,
            bd,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            1.5,
            -0.25,
            1e-10,
            "bsr_b2_f64",
        );
    }

    #[test]
    fn bsr_b2_f32_alpha_beta() {
        let (r, c, bd, rp, ci, v) = bsr_4x4_b2();
        let x = vec![0.5, -1.0, 2.0, 1.5];
        let y0 = vec![1.0, 2.0, 3.0, 4.0];
        run_bsr::<f32>(
            r,
            c,
            bd,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            2.0,
            0.5,
            1e-4,
            "bsr_b2_f32",
        );
    }

    #[test]
    fn bsr_b3_f64_alpha_beta() {
        let (r, c, bd, rp, ci, v) = bsr_6x6_b3();
        let x: Vec<f64> = (0..r as usize).map(|i| 1.0 + i as f64).collect();
        let y0: Vec<f64> = (0..r as usize).map(|i| 100.0 - 2.0 * i as f64).collect();
        run_bsr::<f64>(
            r,
            c,
            bd,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            1.25,
            0.75,
            1e-10,
            "bsr_b3_f64",
        );
    }

    #[test]
    fn bsr_b2_f64_beta_zero() {
        let (r, c, bd, rp, ci, v) = bsr_4x4_b2();
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y0 = vec![1e9, -1e9, 5e8, -5e8];
        run_bsr::<f64>(
            r,
            c,
            bd,
            &rp,
            &ci,
            &v,
            &x,
            &y0,
            1.0,
            0.0,
            1e-10,
            "bsr_b2_beta0",
        );
    }
}
