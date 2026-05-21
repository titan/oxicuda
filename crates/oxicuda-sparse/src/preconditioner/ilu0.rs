//! Incomplete LU factorization with zero fill-in — ILU(0).
//!
//! Computes an approximate factorization `A ~ L * U` where `L` is unit lower
//! triangular and `U` is upper triangular. The sparsity pattern of `L + U` is
//! identical to that of `A` (zero fill-in).
//!
//! ## Algorithm
//!
//! Uses a level-set parallel approach:
//! 1. Rows are grouped into dependency levels based on the lower-triangular
//!    part of the sparsity pattern (same analysis as SpTRSV).
//! 2. For each level, all rows in the level are updated in parallel — one GPU
//!    thread per level-row.
//! 3. For each row `i` in the level and each lower-triangular column `k`:
//!    `a_ij -= (a_ik * a_kj) / a_kk` for all `j` in the intersection of
//!    row `i` and row `k`.
//! 4. After processing all levels, the result is split into `L` (lower + unit
//!    diagonal) and `U` (diagonal + upper).
//!
//! The GPU kernel (`emit_ilu0_kernel`) mirrors the CPU reference
//! [`ilu0`] exactly: identical CSR traversal, identical pivot handling and
//! identical `div`/`mul`/`sub` arithmetic, so the GPU factorization agrees
//! numerically with the CPU one.
#![allow(dead_code)]

use std::sync::Arc;

use oxicuda_blas::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::KernelBuilder;
use oxicuda_ptx::ir::PtxType;

use crate::error::{SparseError, SparseResult};
use crate::format::CsrMatrix;
use crate::handle::SparseHandle;
use crate::ptx_helpers::{load_global_float, ptx_suffix, store_global_float};

/// Default block size for ILU0 kernels.
const ILU0_BLOCK_SIZE: u32 = 256;

/// Status sentinel written by the GPU kernel when a zero pivot is hit.
const ILU0_STATUS_SINGULAR: u32 = 1;

/// Incomplete LU(0) factorization: `A ~ L * U`.
///
/// Returns `(L, U)` where `L` is unit lower triangular and `U` is upper
/// triangular (including diagonal). Both have the same sparsity pattern
/// as the corresponding triangle of `A`.
///
/// The factorization is performed entirely on the host. It is the numerical
/// reference for the GPU kernel; [`ilu0_gpu`] runs the equivalent computation
/// on the device and must produce identical results.
///
/// # Arguments
///
/// * `handle` -- Sparse handle.
/// * `a` -- Sparse CSR matrix to factor (must be square).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square.
/// Returns [`SparseError::SingularMatrix`] if a zero pivot is encountered.
pub fn ilu0<T: GpuFloat>(
    handle: &SparseHandle,
    a: &CsrMatrix<T>,
) -> SparseResult<(CsrMatrix<T>, CsrMatrix<T>)> {
    let _ = handle;
    if a.rows() != a.cols() {
        return Err(SparseError::DimensionMismatch(format!(
            "ILU(0) requires square matrix, got {}x{}",
            a.rows(),
            a.cols()
        )));
    }

    let n = a.rows();
    if n == 0 {
        return Err(SparseError::InvalidArgument(
            "cannot factor an empty matrix".to_string(),
        ));
    }

    // Download structure and values for analysis.
    let (h_row_ptr, h_col_idx, h_values) = a.to_host()?;

    // Build dependency levels from the lower-triangular structure.
    let levels = analyze_ilu0_levels(&h_row_ptr, &h_col_idx, n)?;

    // Work on a mutable copy of the values: the factorization is in-place on
    // the original sparsity pattern.
    let mut work_values = h_values;

    // Execute the factorization level by level on the host.
    for level_rows in &levels {
        for &row_u32 in level_rows {
            factor_row_host(row_u32 as usize, &h_row_ptr, &h_col_idx, &mut work_values)?;
        }
    }

    // Split into L and U.
    split_lu(&h_row_ptr, &h_col_idx, &work_values, n)
}

/// GPU implementation of [`ilu0`].
///
/// Performs the ILU(0) factorization on the device: the working values are
/// uploaded once, the level-update kernel (`emit_ilu0_kernel`) is launched
/// once per dependency level, and the factored values are downloaded and
/// split into `(L, U)`.
///
/// This produces the same `(L, U)` as the host [`ilu0`]; the host path is the
/// numerical reference.
///
/// # Arguments
///
/// * `handle` -- Sparse handle (provides the CUDA stream and SM version).
/// * `a` -- Sparse CSR matrix to factor (must be square).
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square.
/// Returns [`SparseError::SingularMatrix`] if a zero pivot is encountered.
/// Returns [`SparseError::PtxGeneration`] if kernel generation fails.
/// Returns [`SparseError::Cuda`] on any device-side failure.
pub fn ilu0_gpu<T: GpuFloat>(
    handle: &SparseHandle,
    a: &CsrMatrix<T>,
) -> SparseResult<(CsrMatrix<T>, CsrMatrix<T>)> {
    if a.rows() != a.cols() {
        return Err(SparseError::DimensionMismatch(format!(
            "ILU(0) requires square matrix, got {}x{}",
            a.rows(),
            a.cols()
        )));
    }

    let n = a.rows();
    if n == 0 {
        return Err(SparseError::InvalidArgument(
            "cannot factor an empty matrix".to_string(),
        ));
    }

    // Download the structure (for level analysis) and the values (the kernel
    // factors a copy of them in device memory).
    let (h_row_ptr, h_col_idx, h_values) = a.to_host()?;
    let levels = analyze_ilu0_levels(&h_row_ptr, &h_col_idx, n)?;

    // Compile the level-update kernel once.
    let ptx = emit_ilu0_kernel::<T>(handle.sm_version())?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "ilu0_level")?;

    // Upload the working state. `row_ptr` and `col_idx` are read-only and can
    // be reused directly from the input matrix; `values` is mutated in place,
    // so a private device copy is used.
    let d_values = DeviceBuffer::<T>::from_host(&h_values)?;
    let row_ptr_dev = a.row_ptr().as_device_ptr();
    let col_idx_dev = a.col_idx().as_device_ptr();

    // One u32 status flag, checked after every level. The kernel ORs in
    // `ILU0_STATUS_SINGULAR` if a zero pivot is encountered.
    let d_status = DeviceBuffer::<u32>::from_host(&[0u32])?;

    // Run the elimination one level at a time. A full stream synchronization
    // between levels guarantees that every dependency from earlier levels is
    // committed to memory before the next level reads it.
    for level_rows in &levels {
        if level_rows.is_empty() {
            continue;
        }

        let d_level_rows = DeviceBuffer::<u32>::from_host(level_rows)?;
        let num_rows_in_level = level_rows.len() as u32;

        let block_size = ILU0_BLOCK_SIZE;
        let grid_size = grid_size_for(num_rows_in_level, block_size);
        let params = LaunchParams::new(grid_size, block_size);

        kernel.launch(
            &params,
            handle.stream(),
            &(
                row_ptr_dev,
                col_idx_dev,
                d_values.as_device_ptr(),
                d_level_rows.as_device_ptr(),
                num_rows_in_level,
                d_status.as_device_ptr(),
            ),
        )?;

        handle.stream().synchronize()?;

        // Check the singular-pivot flag for this level.
        let mut status = [0u32];
        d_status.copy_to_host(&mut status)?;
        if status[0] & ILU0_STATUS_SINGULAR != 0 {
            return Err(SparseError::SingularMatrix);
        }
    }

    // Download the factored values and split into L and U.
    let mut factored = vec![T::gpu_zero(); h_values.len()];
    d_values.copy_to_host(&mut factored)?;

    split_lu(&h_row_ptr, &h_col_idx, &factored, n)
}

/// Performs the ILU(0) elimination for a single row on the host.
///
/// This is the per-row body of the host factorization and the exact numerical
/// reference the GPU kernel mirrors. `work_values` is updated in place.
fn factor_row_host<T: GpuFloat>(
    row: usize,
    h_row_ptr: &[i32],
    h_col_idx: &[i32],
    work_values: &mut [T],
) -> SparseResult<()> {
    let row_start = h_row_ptr[row] as usize;
    let row_end = h_row_ptr[row + 1] as usize;

    // Process the lower-triangular columns of this row.
    for nz in row_start..row_end {
        let k = h_col_idx[nz] as usize;
        if k >= row {
            break; // past the lower triangle (columns are sorted)
        }

        // Find the diagonal of pivot row k: a_kk.
        let k_start = h_row_ptr[k] as usize;
        let k_end = h_row_ptr[k + 1] as usize;
        let diag_pos = match find_col_in_row(&h_col_idx[k_start..k_end], k as i32) {
            Some(pos) => k_start + pos,
            None => return Err(SparseError::SingularMatrix),
        };

        let a_kk = work_values[diag_pos];
        if a_kk == T::gpu_zero() {
            return Err(SparseError::SingularMatrix);
        }

        // a_ik /= a_kk.
        let a_ik = work_values[nz];
        let ratio = div_gpu_float(a_ik, a_kk);
        work_values[nz] = ratio;

        // For each j > k in row k, update a_ij -= ratio * a_kj.
        for k_nz in (diag_pos + 1)..k_end {
            let j = h_col_idx[k_nz];
            if let Some(ij_off) = find_col_in_row(&h_col_idx[row_start..row_end], j) {
                let ij_pos = row_start + ij_off;
                let a_kj = work_values[k_nz];
                let update = mul_gpu_float(ratio, a_kj);
                work_values[ij_pos] = sub_gpu_float(work_values[ij_pos], update);
            }
        }
    }

    Ok(())
}

/// Finds the position of `target_col` within a sorted slice of column indices.
fn find_col_in_row(col_slice: &[i32], target_col: i32) -> Option<usize> {
    col_slice.iter().position(|&c| c == target_col)
}

/// Analyzes dependency levels for ILU(0).
///
/// Row `i` depends on row `k` if there exists a lower-triangular entry
/// `A[i, k]` with `k < i`. The level of row `i` is one plus the maximum
/// level of any row it depends on.
fn analyze_ilu0_levels(row_ptr: &[i32], col_idx: &[i32], n: u32) -> SparseResult<Vec<Vec<u32>>> {
    let n_usize = n as usize;
    let mut depth = vec![0u32; n_usize];
    let mut max_depth: u32 = 0;

    for i in 0..n_usize {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        let mut max_dep = 0u32;
        for &cj in &col_idx[start..end] {
            let j = cj as usize;
            if j < i {
                let d = depth[j] + 1;
                if d > max_dep {
                    max_dep = d;
                }
            }
        }
        depth[i] = max_dep;
        if max_dep > max_depth {
            max_depth = max_dep;
        }
    }

    let num_levels = max_depth as usize + 1;
    let mut levels: Vec<Vec<u32>> = vec![Vec::new(); num_levels];
    for (i, &d) in depth.iter().enumerate() {
        levels[d as usize].push(i as u32);
    }

    Ok(levels)
}

/// Splits the factored matrix into L (unit lower triangular) and U (upper
/// triangular including diagonal).
fn split_lu<T: GpuFloat>(
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
    n: u32,
) -> SparseResult<(CsrMatrix<T>, CsrMatrix<T>)> {
    let n_usize = n as usize;

    // Count nnz for L and U.
    let mut l_nnz = 0usize;
    let mut u_nnz = 0usize;
    for i in 0..n_usize {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        for &cj in &col_idx[start..end] {
            let j = cj as usize;
            if j < i {
                l_nnz += 1;
            } else {
                u_nnz += 1;
            }
        }
        l_nnz += 1; // unit diagonal for L
    }

    // Build L arrays.
    let mut l_row_ptr = vec![0i32; n_usize + 1];
    let mut l_col_idx = Vec::with_capacity(l_nnz);
    let mut l_values = Vec::with_capacity(l_nnz);

    // Build U arrays.
    let mut u_row_ptr = vec![0i32; n_usize + 1];
    let mut u_col_idx = Vec::with_capacity(u_nnz);
    let mut u_values = Vec::with_capacity(u_nnz);

    for i in 0..n_usize {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;

        // Lower triangle entries + unit diagonal.
        for idx in start..end {
            let j = col_idx[idx] as usize;
            if j < i {
                l_col_idx.push(col_idx[idx]);
                l_values.push(values[idx]);
            }
        }
        // Unit diagonal for L.
        l_col_idx.push(i as i32);
        l_values.push(T::gpu_one());
        l_row_ptr[i + 1] = l_col_idx.len() as i32;

        // Upper triangle entries (including diagonal).
        for idx in start..end {
            let j = col_idx[idx] as usize;
            if j >= i {
                u_col_idx.push(col_idx[idx]);
                u_values.push(values[idx]);
            }
        }
        u_row_ptr[i + 1] = u_col_idx.len() as i32;
    }

    let l_mat = CsrMatrix::from_host(n, n, &l_row_ptr, &l_col_idx, &l_values)?;
    let u_mat = CsrMatrix::from_host(n, n, &u_row_ptr, &u_col_idx, &u_values)?;

    Ok((l_mat, u_mat))
}

/// Generates PTX for the ILU(0) level-update kernel (GPU path).
///
/// Each thread processes one row from the current level, performing the
/// elimination updates for all lower-triangular columns in that row. The body
/// is a faithful translation of [`factor_row_host`]:
///
/// 1. `row = level_rows[tid]`; load `row_ptr[row]` / `row_ptr[row + 1]`.
/// 2. For every `nz` in the row with `col_idx[nz] = k < row`, scan pivot row
///    `k` for its diagonal `a_kk` (a missing diagonal or a zero `a_kk` sets
///    `ILU0_STATUS_SINGULAR` in `status` and aborts), compute the multiplier
///    `ratio = values[nz] / a_kk` and write it back to `values[nz]`.
/// 3. For every `k_nz` in pivot row `k` past the diagonal, look up the column
///    in row `i`'s pattern and, if present, apply
///    `values[ij] -= ratio * values[k_nz]`.
///
/// All loads and stores go through device global memory; the float helpers
/// `div`/`mul`/`sub` use the standard rounded PTX instructions.
fn emit_ilu0_kernel<T: GpuFloat>(sm: SmVersion) -> SparseResult<String> {
    let elem_bytes = T::size_u32();
    let suffix = ptx_suffix::<T>();

    KernelBuilder::new("ilu0_level")
        .target(sm)
        .param("row_ptr", PtxType::U64)
        .param("col_idx", PtxType::U64)
        .param("values", PtxType::U64)
        .param("level_rows", PtxType::U64)
        .param("num_level_rows", PtxType::U32)
        .param("status", PtxType::U64)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let num_level_rows = b.load_param_u32("num_level_rows");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, num_level_rows, move |b| {
                let tid = gid_inner;
                let level_rows_ptr = b.load_param_u64("level_rows");
                let row_ptr_base = b.load_param_u64("row_ptr");
                let col_idx_base = b.load_param_u64("col_idx");
                let values_base = b.load_param_u64("values");
                let status_ptr = b.load_param_u64("status");

                // row = level_rows[tid]
                let row_addr = b.byte_offset_addr(level_rows_ptr, tid, 4);
                let row = b.load_global_u32(row_addr);

                // row_start = row_ptr[row]
                let rs_addr = b.byte_offset_addr(row_ptr_base.clone(), row.clone(), 4);
                let rs_i32 = b.load_global_i32(rs_addr);
                let row_start = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_start}, {rs_i32};"));

                // row_end = row_ptr[row + 1]
                let row_p1 = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {row_p1}, {row}, 1;"));
                let re_addr = b.byte_offset_addr(row_ptr_base.clone(), row_p1, 4);
                let re_i32 = b.load_global_i32(re_addr);
                let row_end = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.b32 {row_end}, {re_i32};"));

                // ---- outer loop over the lower-triangular columns of row i ----
                // for (nz = row_start; nz < row_end; ++nz)
                let nz = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {nz}, {row_start};"));

                let outer_top = b.fresh_label("ilu0_outer");
                let outer_end = b.fresh_label("ilu0_outer_end");

                b.label(&outer_top);
                {
                    let cont = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.lo.u32 {cont}, {nz}, {row_end};"));
                    b.raw_ptx(&format!("@!{cont} bra ${outer_end};"));

                    // k = col_idx[nz]
                    let ci_addr = b.byte_offset_addr(col_idx_base.clone(), nz.clone(), 4);
                    let k_i32 = b.load_global_i32(ci_addr);
                    let k = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.b32 {k}, {k_i32};"));

                    // if (k >= row) break;  -- columns are sorted ascending
                    let is_lower = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.lo.u32 {is_lower}, {k}, {row};"));
                    b.raw_ptx(&format!("@!{is_lower} bra ${outer_end};"));

                    // k_start = row_ptr[k], k_end = row_ptr[k + 1]
                    let ks_addr = b.byte_offset_addr(row_ptr_base.clone(), k.clone(), 4);
                    let ks_i32 = b.load_global_i32(ks_addr);
                    let k_start = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.b32 {k_start}, {ks_i32};"));

                    let k_p1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {k_p1}, {k}, 1;"));
                    let ke_addr = b.byte_offset_addr(row_ptr_base.clone(), k_p1, 4);
                    let ke_i32 = b.load_global_i32(ke_addr);
                    let k_end = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.b32 {k_end}, {ke_i32};"));

                    // ---- find the diagonal a_kk: scan row k for col == k ----
                    // diag_pos starts as k_end (sentinel "not found").
                    let diag_pos = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {diag_pos}, {k_end};"));

                    let scan = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {scan}, {k_start};"));

                    let diag_top = b.fresh_label("ilu0_diag");
                    let diag_end = b.fresh_label("ilu0_diag_end");

                    b.label(&diag_top);
                    {
                        let scan_ok = b.alloc_reg(PtxType::Pred);
                        b.raw_ptx(&format!("setp.lo.u32 {scan_ok}, {scan}, {k_end};"));
                        b.raw_ptx(&format!("@!{scan_ok} bra ${diag_end};"));

                        let sc_addr = b.byte_offset_addr(col_idx_base.clone(), scan.clone(), 4);
                        let sc_i32 = b.load_global_i32(sc_addr);
                        let sc_col = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.b32 {sc_col}, {sc_i32};"));

                        let hit = b.alloc_reg(PtxType::Pred);
                        b.raw_ptx(&format!("setp.eq.u32 {hit}, {sc_col}, {k};"));
                        // On a hit, record scan into diag_pos and stop scanning.
                        b.raw_ptx(&format!("@{hit} mov.u32 {diag_pos}, {scan};"));
                        b.raw_ptx(&format!("@{hit} bra ${diag_end};"));

                        b.raw_ptx(&format!("add.u32 {scan}, {scan}, 1;"));
                        b.branch(&diag_top);
                    }
                    b.label(&diag_end);

                    // if (diag not found) -> singular, abort this thread.
                    let no_diag = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {no_diag}, {diag_pos}, {k_end};"));
                    let after_singular = b.fresh_label("ilu0_after_sing");
                    b.raw_ptx(&format!("@!{no_diag} bra ${after_singular};"));
                    {
                        let one = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {one}, {ILU0_STATUS_SINGULAR};"));
                        let old = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("atom.global.or.b32 {old}, [{status_ptr}], {one};"));
                        b.branch(&outer_end);
                    }
                    b.label(&after_singular);

                    // a_kk = values[diag_pos]
                    let akk_addr =
                        b.byte_offset_addr(values_base.clone(), diag_pos.clone(), elem_bytes);
                    let a_kk = load_global_float::<T>(b, akk_addr);

                    // if (a_kk == 0) -> singular, abort this thread.
                    let zero = b.alloc_reg(T::PTX_TYPE);
                    if elem_bytes == 4 {
                        b.raw_ptx(&format!("mov.b32 {zero}, 0F00000000;"));
                    } else {
                        b.raw_ptx(&format!("mov.b64 {zero}, 0D0000000000000000;"));
                    }
                    let pivot_zero = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.eq.{suffix} {pivot_zero}, {a_kk}, {zero};"));
                    let after_zero = b.fresh_label("ilu0_after_zero");
                    b.raw_ptx(&format!("@!{pivot_zero} bra ${after_zero};"));
                    {
                        let one = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {one}, {ILU0_STATUS_SINGULAR};"));
                        let old = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("atom.global.or.b32 {old}, [{status_ptr}], {one};"));
                        b.branch(&outer_end);
                    }
                    b.label(&after_zero);

                    // ratio = values[nz] / a_kk; values[nz] = ratio
                    let aik_addr = b.byte_offset_addr(values_base.clone(), nz.clone(), elem_bytes);
                    let a_ik = load_global_float::<T>(b, aik_addr.clone());
                    let ratio = b.alloc_reg(T::PTX_TYPE);
                    b.raw_ptx(&format!("div.rn.{suffix} {ratio}, {a_ik}, {a_kk};"));
                    store_global_float::<T>(b, aik_addr, ratio.clone());

                    // ---- inner loop: for k_nz in (diag_pos + 1 .. k_end) ----
                    let k_nz = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {k_nz}, {diag_pos}, 1;"));

                    let inner_top = b.fresh_label("ilu0_inner");
                    let inner_end = b.fresh_label("ilu0_inner_end");

                    b.label(&inner_top);
                    {
                        let inner_ok = b.alloc_reg(PtxType::Pred);
                        b.raw_ptx(&format!("setp.lo.u32 {inner_ok}, {k_nz}, {k_end};"));
                        b.raw_ptx(&format!("@!{inner_ok} bra ${inner_end};"));

                        // j = col_idx[k_nz]
                        let j_addr = b.byte_offset_addr(col_idx_base.clone(), k_nz.clone(), 4);
                        let j_i32 = b.load_global_i32(j_addr);
                        let j_col = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.b32 {j_col}, {j_i32};"));

                        // Look up column j in row i's pattern [row_start, row_end).
                        // ij_pos starts as row_end (sentinel "not found").
                        let ij_pos = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {ij_pos}, {row_end};"));

                        let pscan = b.alloc_reg(PtxType::U32);
                        b.raw_ptx(&format!("mov.u32 {pscan}, {row_start};"));

                        let psc_top = b.fresh_label("ilu0_pat");
                        let psc_end = b.fresh_label("ilu0_pat_end");

                        b.label(&psc_top);
                        {
                            let psc_ok = b.alloc_reg(PtxType::Pred);
                            b.raw_ptx(&format!("setp.lo.u32 {psc_ok}, {pscan}, {row_end};"));
                            b.raw_ptx(&format!("@!{psc_ok} bra ${psc_end};"));

                            let pc_addr =
                                b.byte_offset_addr(col_idx_base.clone(), pscan.clone(), 4);
                            let pc_i32 = b.load_global_i32(pc_addr);
                            let pc_col = b.alloc_reg(PtxType::U32);
                            b.raw_ptx(&format!("mov.b32 {pc_col}, {pc_i32};"));

                            let pat_hit = b.alloc_reg(PtxType::Pred);
                            b.raw_ptx(&format!("setp.eq.u32 {pat_hit}, {pc_col}, {j_col};"));
                            b.raw_ptx(&format!("@{pat_hit} mov.u32 {ij_pos}, {pscan};"));
                            b.raw_ptx(&format!("@{pat_hit} bra ${psc_end};"));

                            b.raw_ptx(&format!("add.u32 {pscan}, {pscan}, 1;"));
                            b.branch(&psc_top);
                        }
                        b.label(&psc_end);

                        // if (j is in row i's pattern): values[ij] -= ratio * a_kj
                        let found = b.alloc_reg(PtxType::Pred);
                        b.raw_ptx(&format!("setp.lo.u32 {found}, {ij_pos}, {row_end};"));
                        let skip_update = b.fresh_label("ilu0_skip_upd");
                        b.raw_ptx(&format!("@!{found} bra ${skip_update};"));
                        {
                            // a_kj = values[k_nz]
                            let akj_addr =
                                b.byte_offset_addr(values_base.clone(), k_nz.clone(), elem_bytes);
                            let a_kj = load_global_float::<T>(b, akj_addr);

                            // a_ij = values[ij_pos]
                            let ij_addr =
                                b.byte_offset_addr(values_base.clone(), ij_pos.clone(), elem_bytes);
                            let a_ij = load_global_float::<T>(b, ij_addr.clone());

                            // update = ratio * a_kj
                            let update = b.alloc_reg(T::PTX_TYPE);
                            b.raw_ptx(&format!("mul.rn.{suffix} {update}, {ratio}, {a_kj};"));
                            // a_ij = a_ij - update
                            let new_ij = b.alloc_reg(T::PTX_TYPE);
                            b.raw_ptx(&format!("sub.rn.{suffix} {new_ij}, {a_ij}, {update};"));
                            store_global_float::<T>(b, ij_addr, new_ij);
                        }
                        b.label(&skip_update);

                        b.raw_ptx(&format!("add.u32 {k_nz}, {k_nz}, 1;"));
                        b.branch(&inner_top);
                    }
                    b.label(&inner_end);

                    b.raw_ptx(&format!("add.u32 {nz}, {nz}, 1;"));
                    b.branch(&outer_top);
                }
                b.label(&outer_end);
            });

            b.ret();
        })
        .build()
        .map_err(|e| SparseError::PtxGeneration(e.to_string()))
}

// -- Arithmetic helpers for GpuFloat on host ----------------------------------

/// Divides two `GpuFloat` values via their bit representation.
fn div_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    let a_bits = a.to_bits_u64();
    let b_bits = b.to_bits_u64();
    if T::SIZE == 4 {
        let fa = f32::from_bits(a_bits as u32);
        let fb = f32::from_bits(b_bits as u32);
        T::from_bits_u64(u64::from((fa / fb).to_bits()))
    } else {
        let fa = f64::from_bits(a_bits);
        let fb = f64::from_bits(b_bits);
        T::from_bits_u64((fa / fb).to_bits())
    }
}

/// Multiplies two `GpuFloat` values via their bit representation.
fn mul_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    let a_bits = a.to_bits_u64();
    let b_bits = b.to_bits_u64();
    if T::SIZE == 4 {
        let fa = f32::from_bits(a_bits as u32);
        let fb = f32::from_bits(b_bits as u32);
        T::from_bits_u64(u64::from((fa * fb).to_bits()))
    } else {
        let fa = f64::from_bits(a_bits);
        let fb = f64::from_bits(b_bits);
        T::from_bits_u64((fa * fb).to_bits())
    }
}

/// Subtracts two `GpuFloat` values via their bit representation.
fn sub_gpu_float<T: GpuFloat>(a: T, b: T) -> T {
    let a_bits = a.to_bits_u64();
    let b_bits = b.to_bits_u64();
    if T::SIZE == 4 {
        let fa = f32::from_bits(a_bits as u32);
        let fb = f32::from_bits(b_bits as u32);
        T::from_bits_u64(u64::from((fa - fb).to_bits()))
    } else {
        let fa = f64::from_bits(a_bits);
        let fb = f64::from_bits(b_bits);
        T::from_bits_u64((fa - fb).to_bits())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    #[test]
    fn ilu0_kernel_ptx_generates_f32() {
        let ptx = emit_ilu0_kernel::<f32>(SmVersion::Sm80);
        assert!(ptx.is_ok());
        let ptx_str = ptx.expect("test: PTX gen should succeed");
        assert!(ptx_str.contains(".entry ilu0_level"));
    }

    #[test]
    fn ilu0_kernel_ptx_generates_f64() {
        let ptx = emit_ilu0_kernel::<f64>(SmVersion::Sm80);
        assert!(ptx.is_ok());
    }

    #[test]
    fn ilu0_kernel_body_performs_elimination() {
        // The kernel body must now actually run the elimination: it has to
        // load values, divide for the multiplier, and store updated entries.
        let ptx = emit_ilu0_kernel::<f32>(SmVersion::Sm80).expect("test: PTX gen should succeed");

        // The multiplier `ratio = a_ik / a_kk` requires a float division.
        assert!(
            ptx.contains("div.rn.f32"),
            "kernel must compute the elimination multiplier"
        );
        // The fill update `a_ij -= ratio * a_kj` requires a multiply and a sub.
        assert!(
            ptx.contains("mul.rn.f32"),
            "kernel must compute the fill update product"
        );
        assert!(
            ptx.contains("sub.rn.f32"),
            "kernel must apply the fill update"
        );
        // The factored values are written back to global memory.
        assert!(
            ptx.contains("st.global.f32"),
            "kernel must store the factored values"
        );
        // Singular pivots are reported through the status flag.
        assert!(
            ptx.contains("atom.global.or.b32"),
            "kernel must report singular pivots"
        );
    }

    #[test]
    fn ilu0_kernel_body_has_no_dead_loads() {
        // Regression guard: the previous stub loaded the parameter pointers
        // and the row address into `_`-prefixed registers and did nothing.
        // A real body issues many loads and stores; require a non-trivial
        // instruction count so an accidental revert to the stub is caught.
        let ptx = emit_ilu0_kernel::<f64>(SmVersion::Sm80).expect("test: PTX gen should succeed");
        let global_loads = ptx.matches("ld.global").count();
        let global_stores = ptx.matches("st.global").count();
        assert!(
            global_loads >= 8,
            "elimination kernel must issue many global loads, got {global_loads}"
        );
        assert!(
            global_stores >= 2,
            "elimination kernel must write factored values, got {global_stores}"
        );
    }

    #[test]
    fn ilu0_levels_identity() {
        // Identity matrix: all rows independent.
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let levels = analyze_ilu0_levels(&row_ptr, &col_idx, 3);
        assert!(levels.is_ok());
        let levels = levels.expect("test: levels should succeed");
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].len(), 3);
    }

    #[test]
    fn host_float_arithmetic() {
        let a = 6.0_f32;
        let b = 2.0_f32;
        let result = div_gpu_float(a, b);
        assert!((result - 3.0_f32).abs() < 1e-6);

        let result = mul_gpu_float(a, b);
        assert!((result - 12.0_f32).abs() < 1e-6);

        let result = sub_gpu_float(a, b);
        assert!((result - 4.0_f32).abs() < 1e-6);
    }

    #[test]
    fn host_float_arithmetic_f64() {
        let a = 6.0_f64;
        let b = 2.0_f64;
        let result = div_gpu_float(a, b);
        assert!((result - 3.0_f64).abs() < 1e-12);
    }

    #[test]
    fn find_col_works() {
        let cols = [0, 2, 5, 7];
        assert_eq!(find_col_in_row(&cols, 2), Some(1));
        assert_eq!(find_col_in_row(&cols, 3), None);
        assert_eq!(find_col_in_row(&cols, 7), Some(3));
    }

    #[test]
    fn factor_row_host_lower_update() {
        // 2x2 dense matrix:
        //   [4 2]
        //   [1 3]
        // row_ptr/col_idx are sorted ascending.
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let mut values = vec![4.0_f64, 2.0, 1.0, 3.0];

        // Row 0 has no lower-triangular entries: unchanged.
        factor_row_host(0, &row_ptr, &col_idx, &mut values)
            .expect("test: row 0 factorization should succeed");
        assert_eq!(values, vec![4.0, 2.0, 1.0, 3.0]);

        // Row 1: k = 0, ratio = a_10 / a_00 = 1/4 = 0.25;
        //        a_11 -= ratio * a_01 = 3 - 0.25 * 2 = 2.5.
        factor_row_host(1, &row_ptr, &col_idx, &mut values)
            .expect("test: row 1 factorization should succeed");
        assert!((values[2] - 0.25).abs() < 1e-12, "multiplier l_10");
        assert!((values[3] - 2.5).abs() < 1e-12, "updated u_11");
    }

    #[test]
    fn factor_row_host_singular_pivot() {
        // 2x2 matrix with a zero diagonal in the pivot row:
        //   [0 2]
        //   [1 3]
        let row_ptr = vec![0, 2, 4];
        let col_idx = vec![0, 1, 0, 1];
        let mut values = vec![0.0_f64, 2.0, 1.0, 3.0];

        // Row 1 needs pivot row 0's diagonal, which is zero -> singular.
        let result = factor_row_host(1, &row_ptr, &col_idx, &mut values);
        assert!(matches!(result, Err(SparseError::SingularMatrix)));
    }

    // ------------------------------------------------------------------
    // GPU integration tests (behind feature gate)
    // ------------------------------------------------------------------

    #[cfg(feature = "gpu-tests")]
    mod gpu {
        use super::*;

        /// Creates a live CUDA context for the calling test thread, or returns
        /// `None` if no GPU is available.
        ///
        /// The CUDA driver API requires an *active* context on the **calling
        /// thread**; `Context::new` calls `cuCtxCreate`, which both creates the
        /// context and sets it current. Callers must keep the returned
        /// `Context` alive for the duration of the test.
        fn gpu_context() -> Option<oxicuda_driver::Context> {
            oxicuda_driver::init().ok()?;
            if oxicuda_driver::Device::count().ok()? == 0 {
                return None;
            }
            let dev = oxicuda_driver::Device::get(0).ok()?;
            oxicuda_driver::Context::new(&dev).ok()
        }

        /// Maximum absolute difference between two CSR value vectors.
        fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max)
        }

        /// Asserts that the GPU ILU(0) factorization of `a` matches the CPU
        /// reference `ilu0` exactly (within tight tolerance).
        fn assert_gpu_matches_cpu(handle: &SparseHandle, a: &CsrMatrix<f64>) {
            let (cpu_l, cpu_u) = ilu0(handle, a).expect("test: CPU ILU(0) should succeed");
            let (gpu_l, gpu_u) = ilu0_gpu(handle, a).expect("test: GPU ILU(0) should succeed");

            let (cpu_l_rp, cpu_l_ci, cpu_l_v) = cpu_l.to_host().expect("test: download CPU L");
            let (gpu_l_rp, gpu_l_ci, gpu_l_v) = gpu_l.to_host().expect("test: download GPU L");
            assert_eq!(cpu_l_rp, gpu_l_rp, "L row_ptr must match");
            assert_eq!(cpu_l_ci, gpu_l_ci, "L col_idx must match");
            assert!(
                max_abs_diff(&cpu_l_v, &gpu_l_v) < 1e-9,
                "L values diverge: cpu={cpu_l_v:?} gpu={gpu_l_v:?}"
            );

            let (cpu_u_rp, cpu_u_ci, cpu_u_v) = cpu_u.to_host().expect("test: download CPU U");
            let (gpu_u_rp, gpu_u_ci, gpu_u_v) = gpu_u.to_host().expect("test: download GPU U");
            assert_eq!(cpu_u_rp, gpu_u_rp, "U row_ptr must match");
            assert_eq!(cpu_u_ci, gpu_u_ci, "U col_idx must match");
            assert!(
                max_abs_diff(&cpu_u_v, &gpu_u_v) < 1e-9,
                "U values diverge: cpu={cpu_u_v:?} gpu={gpu_u_v:?}"
            );
        }

        #[test]
        fn gpu_ilu0_dense_2x2() {
            let Some(ctx) = gpu_context() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let handle = SparseHandle::new(&ctx).expect("test: handle creation");
            // [4 2]
            // [1 3]
            let a =
                CsrMatrix::<f64>::from_host(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &[4.0, 2.0, 1.0, 3.0])
                    .expect("test: matrix creation");
            assert_gpu_matches_cpu(&handle, &a);
        }

        #[test]
        fn gpu_ilu0_dense_3x3() {
            let Some(ctx) = gpu_context() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let handle = SparseHandle::new(&ctx).expect("test: handle creation");
            // A well-conditioned dense 3x3:
            // [ 4  1  2]
            // [ 1  5  1]
            // [ 2  1  6]
            let a = CsrMatrix::<f64>::from_host(
                3,
                3,
                &[0, 3, 6, 9],
                &[0, 1, 2, 0, 1, 2, 0, 1, 2],
                &[4.0, 1.0, 2.0, 1.0, 5.0, 1.0, 2.0, 1.0, 6.0],
            )
            .expect("test: matrix creation");
            assert_gpu_matches_cpu(&handle, &a);
        }

        #[test]
        fn gpu_ilu0_tridiagonal_5x5() {
            let Some(ctx) = gpu_context() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let handle = SparseHandle::new(&ctx).expect("test: handle creation");
            // 5x5 tridiagonal with diagonal 4, off-diagonals -1.
            // This exercises multiple dependency levels and a true sparse
            // pattern (the inner pattern lookup must skip absent columns).
            let mut row_ptr = vec![0i32];
            let mut col_idx = Vec::new();
            let mut values = Vec::new();
            for i in 0..5i32 {
                if i > 0 {
                    col_idx.push(i - 1);
                    values.push(-1.0_f64);
                }
                col_idx.push(i);
                values.push(4.0_f64);
                if i < 4 {
                    col_idx.push(i + 1);
                    values.push(-1.0_f64);
                }
                row_ptr.push(col_idx.len() as i32);
            }
            let a = CsrMatrix::<f64>::from_host(5, 5, &row_ptr, &col_idx, &values)
                .expect("test: matrix creation");
            assert_gpu_matches_cpu(&handle, &a);
        }

        #[test]
        fn gpu_ilu0_identity() {
            let Some(ctx) = gpu_context() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let handle = SparseHandle::new(&ctx).expect("test: handle creation");
            let a = CsrMatrix::<f64>::from_host(
                4,
                4,
                &[0, 1, 2, 3, 4],
                &[0, 1, 2, 3],
                &[1.0, 1.0, 1.0, 1.0],
            )
            .expect("test: matrix creation");
            assert_gpu_matches_cpu(&handle, &a);
        }

        #[test]
        fn gpu_ilu0_singular_pivot() {
            let Some(ctx) = gpu_context() else {
                return;
            };
            let ctx = Arc::new(ctx);
            let handle = SparseHandle::new(&ctx).expect("test: handle creation");
            // Zero diagonal in row 0 -> row 1 hits a zero pivot.
            let a =
                CsrMatrix::<f64>::from_host(2, 2, &[0, 2, 4], &[0, 1, 0, 1], &[0.0, 2.0, 1.0, 3.0])
                    .expect("test: matrix creation");
            let result = ilu0_gpu(&handle, &a);
            assert!(matches!(result, Err(SparseError::SingularMatrix)));
        }
    }
}
