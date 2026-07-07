//! Sampled Dense-Dense Matrix Multiply (SDDMM).
//!
//! Computes `C_ij = alpha * (A @ B)_ij * spy(S)_ij + beta * S_ij`
//! where the result is only computed at positions where the sparse matrix `S`
//! has non-zero entries. This is a key primitive in graph neural networks and
//! sparse attention mechanisms.
//!
//! ## Strategy
//!
//! Each thread handles one non-zero entry of `S`. For that entry at position
//! `(row, col)`, the thread computes the dot product `A[row, :] . B[:, col]`
//! (inner dimension `K`), scales by `alpha`, and blends with the existing
//! value via `beta`.
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
use crate::ptx_helpers::{
    add_float, fma_float, load_float_imm, load_global_float, mul_float, reinterpret_bits_to_float,
    store_global_float,
};

/// Default block size for SDDMM kernel.
const SDDMM_BLOCK_SIZE: u32 = 256;

/// Sampled Dense-Dense Matrix Multiply.
///
/// Computes `S_ij = alpha * dot(A[i,:], B[:,j]) + beta * S_ij` for each
/// non-zero position `(i, j)` in the sparse matrix `S`.
///
/// # Arguments
///
/// * `handle` -- Sparse handle.
/// * `alpha` -- Scalar multiplier for the dense product.
/// * `a_ptr` -- Device pointer to dense matrix `A` (row-major, shape `m x k`).
/// * `a_rows` -- Number of rows of `A`.
/// * `a_cols` -- Number of columns of `A` (= inner dimension `K`).
/// * `a_ld` -- Leading dimension (row stride) of `A`.
/// * `b_ptr` -- Device pointer to dense matrix `B` (row-major, shape `k x n`).
/// * `b_cols` -- Number of columns of `B`.
/// * `b_ld` -- Leading dimension (row stride) of `B`.
/// * `beta` -- Scalar multiplier for existing `S` values.
/// * `s` -- Sparse CSR matrix `S`, updated in place.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if dimensions are incompatible.
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
#[allow(clippy::too_many_arguments)]
pub fn sddmm<T: GpuFloat>(
    handle: &SparseHandle,
    alpha: T,
    a_ptr: CUdeviceptr,
    a_rows: u32,
    a_cols: u32,
    a_ld: u32,
    b_ptr: CUdeviceptr,
    b_cols: u32,
    b_ld: u32,
    beta: T,
    s: &mut CsrMatrix<T>,
) -> SparseResult<()> {
    // Validate dimensions
    if s.rows() != a_rows {
        return Err(SparseError::DimensionMismatch(format!(
            "S.rows ({}) != A.rows ({})",
            s.rows(),
            a_rows
        )));
    }
    if s.cols() != b_cols {
        return Err(SparseError::DimensionMismatch(format!(
            "S.cols ({}) != B.cols ({})",
            s.cols(),
            b_cols
        )));
    }

    if s.nnz() == 0 || a_cols == 0 {
        return Ok(());
    }

    let ptx = emit_sddmm_kernel::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "sddmm")?;

    let block_size = SDDMM_BLOCK_SIZE;
    let grid_size = grid_size_for(s.nnz(), block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            s.row_ptr().as_device_ptr(),
            s.col_idx().as_device_ptr(),
            s.values().as_device_ptr(),
            a_ptr,
            b_ptr,
            alpha.to_bits_u64(),
            beta.to_bits_u64(),
            s.rows(),
            a_cols,
            a_ld,
            b_ld,
        ),
    )?;

    Ok(())
}

/// Generates PTX for the SDDMM kernel.
///
/// Each thread handles one non-zero of `S`. It identifies the row and column
/// from the CSR structure, then computes the dot product of `A[row, :]` and
/// `B[:, col]` over the inner dimension `K`.
fn emit_sddmm_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let is_f64 = T::SIZE == 8;

    KernelBuilder::new("sddmm")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("a_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("alpha_bits", PtxType::U64)
        .param("beta_bits", PtxType::U64)
        .param("m", PtxType::U32)
        .param("k", PtxType::U32)
        .param("a_ld", PtxType::U32)
        .param("b_ld", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let m_param = b.load_param_u32("m");
            let mov_suffix = if is_f64 { "f64" } else { "f32" };

            // We need to find which row this non-zero belongs to.
            // Simple approach: binary search in row_ptr.
            // But for PTX simplicity, we use a linear scan.
            //
            // Actually, we launch one thread per row and iterate over that row's nnz.
            // This is simpler and avoids the row-finding problem.

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, m_param, move |b| {
                let row = gid_inner;
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let a_ptr = b.load_param_u64("a_ptr");
                let b_ptr = b.load_param_u64("b_ptr");
                let alpha_bits = b.load_param_u64("alpha_bits");
                let beta_bits = b.load_param_u64("beta_bits");
                let k_param = b.load_param_u32("k");
                let a_ld = b.load_param_u32("a_ld");
                let b_ld = b.load_param_u32("b_ld");

                let alpha = reinterpret_bits_to_float::<T>(b, alpha_bits);
                let beta = reinterpret_bits_to_float::<T>(b, beta_bits);

                // Load row bounds
                let rp_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let rs_i32 = b.load_global_i32(rp_addr);
                let rs = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {rs}, {rs_i32};"));

                let row_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_p1}, {row}, 1;"));
                let re_addr = b.byte_offset_addr(row_ptr_base, row_p1, 4);
                let re_i32 = b.load_global_i32(re_addr);
                let re = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {re}, {re_i32};"));

                // For each non-zero in this row
                let nz_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {nz_idx}, {rs};"));

                let nz_loop = b.fresh_label("sddmm_nz_loop");
                let nz_done = b.fresh_label("sddmm_nz_done");

                b.label(&nz_loop);
                // Exit when nz_idx >= row_end (inverted skip-branch via branch_if
                // so the `$`-prefixed label target matches the `b.label` def).
                let nz_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {nz_pred}, {nz_idx}, {re};"));
                b.branch_if(nz_pred, &nz_done);

                // Load column index
                let ci_addr = b.byte_offset_addr(col_idx_base.clone(), nz_idx.clone(), 4);
                let col_i32 = b.load_global_i32(ci_addr);
                let col = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {col}, {col_i32};"));

                // Compute dot product: A[row, :] . B[:, col]
                let dot = load_float_imm::<T>(b, 0.0);

                let kk = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {kk}, 0;"));

                let k_loop = b.fresh_label("sddmm_k_loop");
                let k_done = b.fresh_label("sddmm_k_done");

                b.label(&k_loop);
                // Exit when kk >= k (inverted skip-branch via branch_if).
                let k_pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.hs.u32 {k_pred}, {kk}, {k_param};"));
                b.branch_if(k_pred, &k_done);

                // A[row, kk] = a_ptr + (row * a_ld + kk) * elem_bytes
                let a_row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {a_row_off}, {row}, {a_ld};"));
                let a_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {a_idx}, {a_row_off}, {kk};"));
                let a_addr = b.byte_offset_addr(a_ptr.clone(), a_idx, elem_bytes);
                let a_val = load_global_float::<T>(b, a_addr);

                // B[kk, col] = b_ptr + (kk * b_ld + col) * elem_bytes
                let b_row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {b_row_off}, {kk}, {b_ld};"));
                let b_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {b_idx}, {b_row_off}, {col};"));
                let b_addr = b.byte_offset_addr(b_ptr.clone(), b_idx, elem_bytes);
                let b_val = load_global_float::<T>(b, b_addr);

                // dot += a_val * b_val
                let new_dot = fma_float::<T>(b, a_val, b_val, dot.clone());
                b.raw_ptx(&format!("mov.{mov_suffix} {dot}, {new_dot};"));

                b.raw_ptx(&format!("add.u32 {kk}, {kk}, 1;"));
                b.branch(&k_loop);
                b.label(&k_done);

                // Load old S value
                let s_v_addr = b.byte_offset_addr(values_base.clone(), nz_idx.clone(), elem_bytes);
                let s_old = load_global_float::<T>(b, s_v_addr.clone());

                // result = alpha * dot + beta * s_old
                let alpha_dot = mul_float::<T>(b, alpha.clone(), dot);
                let beta_s = mul_float::<T>(b, beta.clone(), s_old);
                let result = add_float::<T>(b, alpha_dot, beta_s);

                store_global_float::<T>(b, s_v_addr, result);

                b.raw_ptx(&format!("add.u32 {nz_idx}, {nz_idx}, 1;"));
                b.branch(&nz_loop);
                b.label(&nz_done);
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

    /// The SDDMM kernel must assemble for sm_86 in both precisions with
    /// `$`-prefixed branch targets and no `.b64` shuffle.
    #[test]
    fn sddmm_f32_f64_assemble_sm86() {
        let f32_ptx = emit_sddmm_kernel::<f32>(SmVersion::Sm86).expect("f32 SDDMM PTX");
        assert_assembles_and_clean("sddmm_f32", &f32_ptx);

        let f64_ptx = emit_sddmm_kernel::<f64>(SmVersion::Sm86).expect("f64 SDDMM PTX");
        assert_assembles_and_clean("sddmm_f64", &f64_ptx);
        assert!(
            !f64_ptx.contains("0F00000000"),
            "f64 SDDMM kernel must not materialize an f32 0.0 immediate:\n{f64_ptx}"
        );
    }
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn sddmm_ptx_generates_f32() {
        let ptx = emit_sddmm_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry sddmm"));
    }

    #[test]
    fn sddmm_ptx_generates_f64() {
        let ptx = emit_sddmm_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn sddmm_ptx_has_correct_target() {
        let ptx = emit_sddmm_kernel::<f32>(SmVersion::Sm75);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".target sm_75"));
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
    use oxicuda_memory::DeviceBuffer;

    /// CPU oracle for SDDMM: for each non-zero `(row, col)` of `S`,
    /// `S = alpha * sum_k A[row,k]*B[k,col] + beta * S_old`.
    /// `A` is m x k row-major, `B` is k x n row-major (contiguous, no padding).
    #[allow(clippy::too_many_arguments)]
    fn cpu_sddmm(
        m: usize,
        k: usize,
        n: usize,
        row_ptr: &[i32],
        col_idx: &[i32],
        s_old: &[f64],
        a: &[f64],
        b: &[f64],
        alpha: f64,
        beta: f64,
    ) -> Vec<f64> {
        let mut s = s_old.to_vec();
        for row in 0..m {
            for nz in row_ptr[row] as usize..row_ptr[row + 1] as usize {
                let col = col_idx[nz] as usize;
                let mut dot = 0.0_f64;
                for kk in 0..k {
                    // A is m x k row-major; B is k x n row-major.
                    dot += a[row * k + kk] * b[kk * n + col];
                }
                s[nz] = alpha * dot + beta * s_old[nz];
            }
        }
        s
    }

    /// Drive the production `sddmm` op and compare to the CPU oracle.
    #[allow(clippy::too_many_arguments)]
    fn run_sddmm<T: GpuFloat>(
        m: u32,
        k: u32,
        n: u32,
        row_ptr: &[i32],
        col_idx: &[i32],
        s_old: &[f64],
        a: &[f64],
        b: &[f64],
        alpha: f64,
        beta: f64,
        tol: f64,
        tag: &str,
    ) {
        let Some(handle) = gpu_handle() else {
            return;
        };
        let dev_s: Vec<T> = s_old.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let mut s =
            CsrMatrix::<T>::from_host(m, n, row_ptr, col_idx, &dev_s).expect("test: build CSR S");

        let dev_a: Vec<T> = a.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let dev_b: Vec<T> = b.iter().map(|&v| f64_to_gpu::<T>(v)).collect();
        let a_buf = DeviceBuffer::from_host(&dev_a).expect("test: upload A");
        let b_buf = DeviceBuffer::from_host(&dev_b).expect("test: upload B");

        sddmm::<T>(
            &handle,
            f64_to_gpu::<T>(alpha),
            a_buf.as_device_ptr(),
            m,
            k,
            k, // a_ld (row-major, contiguous)
            b_buf.as_device_ptr(),
            n,
            n, // b_ld (row-major, contiguous)
            f64_to_gpu::<T>(beta),
            &mut s,
        )
        .expect("test: sddmm launch");
        handle.stream().synchronize().expect("test: sync");

        let (_rp, _ci, out_vals) = s.to_host().expect("test: download S");
        let got: Vec<f64> = out_vals.iter().map(|&v| gpu_to_f64(v)).collect();
        let want = cpu_sddmm(
            m as usize, k as usize, n as usize, row_ptr, col_idx, s_old, a, b, alpha, beta,
        );
        assert_close(&got, &want, tol, tag);
    }

    /// Sparse mask S (3x4):
    /// nonzeros at (0,0),(0,2),(1,1),(1,3),(2,0),(2,3)
    fn mask_3x4() -> (u32, Vec<i32>, Vec<i32>, Vec<f64>) {
        let row_ptr = vec![0, 2, 4, 6];
        let col_idx = vec![0, 2, 1, 3, 0, 3];
        // pre-existing S values (used by the beta term)
        let s_old = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        (4, row_ptr, col_idx, s_old)
    }

    fn dense(rows: usize, cols: usize, base: f64) -> Vec<f64> {
        (0..rows * cols)
            .map(|idx| base + 0.25 * (idx as f64) - 0.05 * ((idx % 5) as f64))
            .collect()
    }

    #[test]
    fn sddmm_f64_alpha_beta() {
        let m = 3usize;
        let k = 5usize;
        let (n, rp, ci, s_old) = mask_3x4();
        let a = dense(m, k, 1.0);
        let b = dense(k, n as usize, -0.5);
        run_sddmm::<f64>(
            m as u32,
            k as u32,
            n,
            &rp,
            &ci,
            &s_old,
            &a,
            &b,
            1.5,
            -0.75,
            1e-10,
            "sddmm_f64",
        );
    }

    #[test]
    fn sddmm_f32_alpha_beta() {
        let m = 3usize;
        let k = 4usize;
        let (n, rp, ci, s_old) = mask_3x4();
        let a = dense(m, k, 0.5);
        let b = dense(k, n as usize, 1.0);
        run_sddmm::<f32>(
            m as u32,
            k as u32,
            n,
            &rp,
            &ci,
            &s_old,
            &a,
            &b,
            2.0,
            0.5,
            1e-4,
            "sddmm_f32",
        );
    }

    #[test]
    fn sddmm_f64_beta_zero() {
        // beta = 0: result depends only on the dense product, not S_old.
        let m = 3usize;
        let k = 6usize;
        let (n, rp, ci, _s_old) = mask_3x4();
        let s_old = vec![1e7; 6];
        let a = dense(m, k, 2.0);
        let b = dense(k, n as usize, 0.3);
        run_sddmm::<f64>(
            m as u32,
            k as u32,
            n,
            &rp,
            &ci,
            &s_old,
            &a,
            &b,
            1.0,
            0.0,
            1e-10,
            "sddmm_beta0",
        );
    }
}
