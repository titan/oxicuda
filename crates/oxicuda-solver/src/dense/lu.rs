//! LU Factorization with partial pivoting.
//!
//! Computes `P * A = L * U` where:
//! - P is a permutation matrix (represented by pivot indices)
//! - L is unit lower triangular
//! - U is upper triangular
//!
//! Uses a blocked right-looking algorithm:
//! 1. Panel factorization: factor a narrow column panel using a dedicated GPU kernel
//! 2. Apply pivots: swap rows in the trailing portion
//! 3. TRSM: solve for the upper triangle block
//! 4. GEMM: update the trailing submatrix
//!
//! The L and U factors overwrite the input matrix A in-place (LAPACK-style packed
//! storage with unit diagonal for L implicitly assumed).

use std::sync::Arc;

use oxicuda_blas::types::{
    DiagType, FillMode, GpuFloat, Layout, MatrixDesc, MatrixDescMut, Side, Transpose,
};
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;
use crate::ptx_helpers::{
    SOLVER_BLOCK_SIZE, abs_float, div_float, fma_float, load_global_float, mul_float,
    store_global_float, sub_float,
};

/// Block size for the panel factorization step.
const LU_BLOCK_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of an LU factorization.
///
/// Contains diagnostic information about the factorization.
#[derive(Debug, Clone)]
pub struct LuResult {
    /// Status info:
    /// - 0: successful factorization
    /// - i > 0: U(i,i) is exactly zero, matrix is singular at column i
    pub info: i32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs LU factorization with partial pivoting in-place.
///
/// On exit, the lower triangle of `a` (with implicit unit diagonal) contains L,
/// and the upper triangle contains U. The `pivots` array records the row
/// permutations: row `i` was interchanged with row `pivots[i]`.
///
/// The matrix is stored in column-major order with leading dimension `lda`.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `a` — matrix buffer (n x n, column-major, lda stride), modified in-place.
/// * `n` — matrix dimension.
/// * `lda` — leading dimension (>= n).
/// * `pivots` — output pivot indices buffer (length >= n).
///
/// # Returns
///
/// [`LuResult`] with `info == 0` on success, `info > 0` if singular.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid or a kernel launch fails.
pub fn lu_factorize<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    pivots: &mut DeviceBuffer<i32>,
) -> SolverResult<LuResult> {
    // Validate dimensions.
    if n == 0 {
        return Ok(LuResult { info: 0 });
    }
    if lda < n {
        return Err(SolverError::DimensionMismatch(format!(
            "lu_factorize: lda ({lda}) must be >= n ({n})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "lu_factorize: buffer too small ({} < {required})",
            a.len()
        )));
    }
    if pivots.len() < n as usize {
        return Err(SolverError::DimensionMismatch(format!(
            "lu_factorize: pivots buffer too small ({} < {n})",
            pivots.len()
        )));
    }

    blocked_lu::<T>(handle, a, n, lda, pivots)
}

/// Solves `A * X = B` given an LU-factored matrix.
///
/// The LU factors must have been computed by [`lu_factorize`]. The solution
/// overwrites `b` in-place.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `lu` — LU-factored matrix (output of `lu_factorize`).
/// * `pivots` — pivot indices from `lu_factorize`.
/// * `b` — right-hand side matrix (n x nrhs), overwritten with solution.
/// * `n` — matrix dimension.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid or BLAS operations fail.
pub fn lu_solve<T: GpuFloat>(
    handle: &SolverHandle,
    lu: &DeviceBuffer<T>,
    pivots: &DeviceBuffer<i32>,
    b: &mut DeviceBuffer<T>,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    if lu.len() < (n as usize * n as usize) {
        return Err(SolverError::DimensionMismatch(
            "lu_solve: LU buffer too small".into(),
        ));
    }
    if pivots.len() < n as usize {
        return Err(SolverError::DimensionMismatch(
            "lu_solve: pivots buffer too small".into(),
        ));
    }
    if b.len() < (n as usize * nrhs as usize) {
        return Err(SolverError::DimensionMismatch(
            "lu_solve: B buffer too small".into(),
        ));
    }

    // Step 1: Apply row permutations to B.
    // Each pivot[i] says row i was swapped with row pivot[i] during
    // factorization, so we replay the swaps in forward order.
    apply_pivots_to_rhs::<T>(handle, b, pivots, n, nrhs)?;

    // Step 2: Solve L * Y = P * B (forward substitution) via TRSM.
    let l_desc = MatrixDesc::<T>::from_raw(lu.as_device_ptr(), n, n, n, Layout::ColMajor);
    let mut b_desc = MatrixDescMut::<T>::from_raw(b.as_device_ptr(), n, nrhs, n, Layout::ColMajor);

    oxicuda_blas::level3::trsm(
        handle.blas(),
        Side::Left,
        FillMode::Lower,
        Transpose::NoTrans,
        DiagType::Unit,
        T::gpu_one(),
        &l_desc,
        &mut b_desc,
    )?;

    // Step 3: Solve U * X = Y (backward substitution) via TRSM.
    let u_desc = MatrixDesc::<T>::from_raw(lu.as_device_ptr(), n, n, n, Layout::ColMajor);

    oxicuda_blas::level3::trsm(
        handle.blas(),
        Side::Left,
        FillMode::Upper,
        Transpose::NoTrans,
        DiagType::NonUnit,
        T::gpu_one(),
        &u_desc,
        &mut b_desc,
    )?;

    Ok(())
}

/// Solves the transposed system `Aᵀ * X = B` given an LU-factored matrix.
///
/// The LU factors must have been produced by [`lu_factorize`], which computes
/// `P * A = L * U`. Hence `A = Pᵀ * L * U` and the transpose factorizes as
///
/// ```text
/// Aᵀ = Uᵀ * Lᵀ * P
/// ```
///
/// so `Aᵀ * X = B` is solved by the three stages
///
/// 1. `Uᵀ * Y = B` — forward substitution (`Uᵀ` is lower triangular, non-unit
///    diagonal),
/// 2. `Lᵀ * W = Y` — backward substitution (`Lᵀ` is upper triangular, unit
///    diagonal),
/// 3. `X = Pᵀ * W` — apply the row permutation transposed, i.e. replay the
///    pivot transpositions in reverse order.
///
/// The solution overwrites `b` in-place. The LU buffer is interpreted in
/// column-major order with leading dimension `n`, matching [`lu_solve`].
///
/// This is required by algorithms — such as Hager's 1-norm condition
/// estimator — that alternate solves with `A` and `Aᵀ`.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `lu` — LU-factored matrix (output of `lu_factorize`).
/// * `pivots` — pivot indices from `lu_factorize`.
/// * `b` — right-hand side matrix (n x nrhs, column-major), overwritten with
///   the solution.
/// * `n` — matrix dimension.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid, a pivot index is out of
/// range, or a host transfer fails.
pub fn lu_solve_transposed<T: GpuFloat>(
    handle: &SolverHandle,
    lu: &DeviceBuffer<T>,
    pivots: &DeviceBuffer<i32>,
    b: &mut DeviceBuffer<T>,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    lu_solve_with_transpose::<T>(handle, lu, pivots, b, n, nrhs, true)
}

/// Solves `A * X = B` or `Aᵀ * X = B` depending on `transpose`.
///
/// When `transpose` is `false` this is equivalent to [`lu_solve`]; when it is
/// `true` it is equivalent to [`lu_solve_transposed`]. Both directions are
/// computed with an exact host-side triangular solve so that the result is
/// correct independently of the device TRSM path.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid, a pivot index is out of
/// range, or a host transfer fails.
pub fn lu_solve_with_transpose<T: GpuFloat>(
    _handle: &SolverHandle,
    lu: &DeviceBuffer<T>,
    pivots: &DeviceBuffer<i32>,
    b: &mut DeviceBuffer<T>,
    n: u32,
    nrhs: u32,
    transpose: bool,
) -> SolverResult<()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    let n_usize = n as usize;
    let nrhs_usize = nrhs as usize;
    if lu.len() < n_usize * n_usize {
        return Err(SolverError::DimensionMismatch(
            "lu_solve_with_transpose: LU buffer too small".into(),
        ));
    }
    if pivots.len() < n_usize {
        return Err(SolverError::DimensionMismatch(
            "lu_solve_with_transpose: pivots buffer too small".into(),
        ));
    }
    if b.len() < n_usize * nrhs_usize {
        return Err(SolverError::DimensionMismatch(
            "lu_solve_with_transpose: B buffer too small".into(),
        ));
    }

    // Copy the LU factors, pivots, and right-hand side to the host. The LU
    // buffer is column-major with leading dimension n.
    let mut lu_host = vec![T::gpu_zero(); lu.len()];
    lu.copy_to_host(&mut lu_host)?;
    let mut piv_host = vec![0_i32; pivots.len()];
    pivots.copy_to_host(&mut piv_host)?;
    let mut b_host = vec![T::gpu_zero(); b.len()];
    b.copy_to_host(&mut b_host)?;

    // Validate pivot indices up front so substitution can ignore the issue.
    for &p in piv_host.iter().take(n_usize) {
        let p_usize = p.max(0) as usize;
        if p_usize >= n_usize {
            return Err(SolverError::DimensionMismatch(format!(
                "lu_solve_with_transpose: pivot index out of range ({p_usize} >= n {n_usize})"
            )));
        }
    }

    // `lu_at(row, col)` reads the column-major LU buffer (lda = n).
    let lu_at =
        |row: usize, col: usize| -> f64 { t_value_to_f64::<T>(lu_host[col * n_usize + row]) };

    for col in 0..nrhs_usize {
        // Working column of the right-hand side / solution.
        let base = col * n_usize;
        let mut rhs: Vec<f64> = (0..n_usize)
            .map(|i| t_value_to_f64::<T>(b_host[base + i]))
            .collect();

        if transpose {
            // Aᵀ x = b  ⇒  Uᵀ y = b, Lᵀ w = y, x = Pᵀ w.
            //
            // Stage 1: Uᵀ y = b. Uᵀ is lower triangular with the (non-unit)
            // diagonal U[k,k]; forward substitution over k = 0..n.
            for k in 0..n_usize {
                let acc: f64 = rhs[..k]
                    .iter()
                    .enumerate()
                    .map(|(i, &rhs_i)| lu_at(i, k) * rhs_i)
                    .sum();
                let diag = lu_at(k, k);
                if diag.abs() <= f64::MIN_POSITIVE {
                    return Err(SolverError::InternalError(format!(
                        "lu_solve_with_transpose: zero pivot U[{k},{k}] (singular)"
                    )));
                }
                rhs[k] = (rhs[k] - acc) / diag;
            }

            // Stage 2: Lᵀ w = y. Lᵀ is upper triangular with unit diagonal;
            // backward substitution over k = n-1..0. L[i,k] (i > k) lives in
            // the strict lower triangle of the LU buffer.
            for k in (0..n_usize).rev() {
                let acc: f64 = rhs[(k + 1)..]
                    .iter()
                    .enumerate()
                    .map(|(offset, &rhs_i)| lu_at(k + 1 + offset, k) * rhs_i)
                    .sum();
                rhs[k] -= acc;
            }

            // Stage 3: x = Pᵀ w. Pᵀ is the inverse permutation: replay the
            // pivot transpositions in reverse order.
            for row in (0..n_usize).rev() {
                let piv = piv_host[row].max(0) as usize;
                if piv != row {
                    rhs.swap(row, piv);
                }
            }
        } else {
            // A x = b  ⇒  L y = P b, U x = y.
            //
            // Stage 1: apply P (pivot transpositions in forward order).
            for (row, &piv_entry) in piv_host.iter().enumerate().take(n_usize) {
                let piv = piv_entry.max(0) as usize;
                if piv != row {
                    rhs.swap(row, piv);
                }
            }

            // Stage 2: L y = P b. L is unit lower triangular; forward
            // substitution over k = 0..n.
            for k in 0..n_usize {
                let acc: f64 = rhs[..k]
                    .iter()
                    .enumerate()
                    .map(|(i, &rhs_i)| lu_at(k, i) * rhs_i)
                    .sum();
                rhs[k] -= acc;
            }

            // Stage 3: U x = y. U is upper triangular, non-unit diagonal;
            // backward substitution over k = n-1..0.
            for k in (0..n_usize).rev() {
                let acc: f64 = rhs[(k + 1)..]
                    .iter()
                    .enumerate()
                    .map(|(offset, &rhs_i)| lu_at(k, k + 1 + offset) * rhs_i)
                    .sum();
                let diag = lu_at(k, k);
                if diag.abs() <= f64::MIN_POSITIVE {
                    return Err(SolverError::InternalError(format!(
                        "lu_solve_with_transpose: zero pivot U[{k},{k}] (singular)"
                    )));
                }
                rhs[k] = (rhs[k] - acc) / diag;
            }
        }

        for (i, &value) in rhs.iter().enumerate() {
            b_host[base + i] = f64_to_t_value::<T>(value);
        }
    }

    b.copy_from_host(&b_host)?;

    Ok(())
}

/// Reinterprets a `T: GpuFloat` value as `f64`.
///
/// 8-byte types are read directly; narrower types are read as `f32` then
/// widened, matching the bit-reinterpretation convention used throughout the
/// solver crate's host-fallback code.
fn t_value_to_f64<T: GpuFloat>(value: T) -> f64 {
    if T::SIZE == 8 {
        f64::from_bits(value.to_bits_u64())
    } else {
        f64::from(f32::from_bits(value.to_bits_u64() as u32))
    }
}

/// Converts an `f64` value back to a `T: GpuFloat` via bit reinterpretation.
///
/// Inverse of [`t_value_to_f64`]: narrower types narrow through `f32` first.
fn f64_to_t_value<T: GpuFloat>(value: f64) -> T {
    if T::SIZE == 8 {
        T::from_bits_u64(value.to_bits())
    } else {
        T::from_bits_u64(u64::from((value as f32).to_bits()))
    }
}

// ---------------------------------------------------------------------------
// Blocked LU implementation
// ---------------------------------------------------------------------------

/// Blocked right-looking LU factorization.
///
/// Processes the matrix in column panels of width `LU_BLOCK_SIZE`:
/// 1. Factor the panel (find pivots, compute L column, compute U row).
/// 2. Swap rows in the trailing matrix according to pivots.
/// 3. TRSM: compute U block for the panel's upper triangle.
/// 4. GEMM: update the trailing submatrix.
fn blocked_lu<T: GpuFloat>(
    handle: &mut SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    pivots: &mut DeviceBuffer<i32>,
) -> SolverResult<LuResult> {
    let nb = LU_BLOCK_SIZE.min(n);
    let num_blocks = n.div_ceil(nb);
    let mut info: i32 = 0;

    for block_idx in 0..num_blocks {
        let j = block_idx * nb;
        let jb = nb.min(n - j); // Actual panel width (may be smaller for last block).

        // Step 1: Panel factorization — factorize columns j..j+jb of the
        // submatrix A[j:n, j:j+jb].
        let panel_info = panel_lu::<T>(handle, a, n, lda, j, jb, pivots)?;
        if panel_info > 0 && info == 0 {
            info = panel_info + j as i32;
        }

        // Step 2: Apply pivots to columns outside the panel.
        // Left side (columns 0..j): swap rows according to pivots.
        if j > 0 {
            apply_panel_pivots::<T>(handle, a, lda, j, jb, pivots, 0, j)?;
        }
        // Right side (columns j+jb..n): swap rows according to pivots.
        let right_start = j + jb;
        if right_start < n {
            apply_panel_pivots::<T>(handle, a, lda, j, jb, pivots, right_start, n - right_start)?;
        }

        // Step 3: Panel triangular solve — compute the U row block
        // U[j:j+jb, j+jb:n] = L[j:j+jb, j:j+jb]^{-1} * A[j:j+jb, j+jb:n] in place.
        //
        // Implemented with a dedicated strided device kernel (`trsm_unit_lower`)
        // rather than the BLAS TRSM: the matrices here are sub-blocks with
        // leading dimension `lda > rows`, and a correct strided solve is required.
        let right_start = j + jb;
        if right_start < n {
            let l_ptr = a.as_device_ptr() + (j as u64 + j as u64 * lda as u64) * T::SIZE as u64;
            let u_ptr =
                a.as_device_ptr() + (j as u64 + right_start as u64 * lda as u64) * T::SIZE as u64;
            launch_trsm_unit_lower::<T>(handle, l_ptr, u_ptr, jb, n - right_start, lda, lda)?;
        }

        // Step 4: Trailing rank-`jb` update:
        // A[j+jb:n, j+jb:n] -= A[j+jb:n, j:j+jb] * A[j:j+jb, j+jb:n]
        //
        // Implemented with a dedicated strided device kernel (`gemm_update`) for
        // the same leading-dimension reason as Step 3.
        let remaining = n.saturating_sub(j + jb);
        if remaining > 0 {
            let a21_ptr =
                a.as_device_ptr() + ((j + jb) as u64 + j as u64 * lda as u64) * T::SIZE as u64;
            let u12_ptr =
                a.as_device_ptr() + (j as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64;
            let a22_ptr = a.as_device_ptr()
                + ((j + jb) as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64;
            launch_gemm_update::<T>(
                handle, a21_ptr, u12_ptr, a22_ptr, remaining, remaining, jb, lda, lda, lda,
            )?;
        }
    }

    Ok(LuResult { info })
}

/// Panel factorization: factorizes columns `j..j+jb` of `A[j:n, j:j+jb]`
/// **on the device**.
///
/// Launches a single-CTA, right-looking unblocked LU kernel (Doolittle with
/// partial pivoting) that runs entirely in global memory — the same in-place
/// structure used by the dense Cholesky panel. The panel is `m = n - j` rows by
/// `jb` columns; thread `t` owns panel column `t` (`jb <= LU_BLOCK_SIZE <=
/// SOLVER_BLOCK_SIZE`). For each pivot column `k`:
///
/// 1. every owner redundantly scans column `k` (rows `k..m`) for the
///    largest-magnitude entry, and the owner of `k` records the absolute pivot
///    index into `pivots[j + k]`;
/// 2. each owner swaps its column's rows `k`/`pivot` (partial pivoting across
///    the whole panel);
/// 3. the owner of `k` scales the sub-diagonal of column `k` by the pivot;
/// 4. owners `k < t < jb` apply the rank-1 trailing update to their column.
///
/// Block-wide barriers separate the four phases so the only cross-thread
/// dependency — reading the freshly published pivot column — is ordered.
///
/// Returns the panel-local info (`0` on success, `> 0` = 1-based panel column of
/// the first (near-)zero pivot).
fn panel_lu<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
    j: u32,
    jb: u32,
    pivots: &mut DeviceBuffer<i32>,
) -> SolverResult<i32> {
    if jb == 0 {
        return Ok(0);
    }

    let sm = handle.sm_version();
    let ptx = emit_panel_lu::<T>(sm, jb)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &panel_lu_name::<T>(jb))?;

    // Per-panel singular-column flag. Initialised to a positive sentinel so a
    // `min`-reduction over (k + 1) records the first (smallest) singular column.
    const INFO_SENTINEL: i32 = i32::MAX;
    let info_buf = DeviceBuffer::<i32>::from_host(&[INFO_SENTINEL])?;

    // Panel rows: the trailing height m = n - j.
    let m = n.saturating_sub(j);

    // One CTA; one thread per panel column. No dynamic shared memory.
    let params = LaunchParams::new(1u32, SOLVER_BLOCK_SIZE);

    let panel_offset = (j as u64 + j as u64 * lda as u64) * T::SIZE as u64;
    let panel_ptr = a.as_device_ptr() + panel_offset;
    let pivots_ptr = pivots.as_device_ptr();
    let info_ptr = info_buf.as_device_ptr();

    let args = (panel_ptr, pivots_ptr, info_ptr, m, jb, j, lda);
    kernel.launch(&params, handle.stream(), &args)?;

    let mut info_host = [INFO_SENTINEL];
    info_buf.copy_to_host(&mut info_host)?;
    let info = if info_host[0] == INFO_SENTINEL {
        0
    } else {
        info_host[0]
    };

    Ok(info)
}

/// Applies pivot swaps from panel factorization to columns outside the panel,
/// **on the device**.
///
/// For each `t` in `0..jb` the row `row = j + t` is swapped with `pivots[row]`
/// across the column range `[col_start, col_start + col_count)`. The swaps must
/// be replayed in increasing `t` because partial-pivoting transpositions
/// compose, so the kernel parallelises across columns (one thread per column)
/// while each thread walks the `jb` transpositions sequentially.
#[allow(clippy::too_many_arguments)]
fn apply_panel_pivots<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    lda: u32,
    j: u32,
    jb: u32,
    pivots: &DeviceBuffer<i32>,
    col_start: u32,
    col_count: u32,
) -> SolverResult<()> {
    if col_count == 0 || jb == 0 {
        return Ok(());
    }
    launch_pivot_swap::<T>(handle, a, pivots, j, jb, col_start, col_count, lda)
}

/// Applies pivot permutations to the right-hand side `B` **on the device**.
///
/// Equivalent to applying every transposition `pivots[row]` (`row = 0..n`) in
/// forward order to each of the `nrhs` columns of `B` (column-major, leading
/// dimension `n`). Implemented with the same device row-swap kernel as
/// [`apply_panel_pivots`] using `j = 0`, `jb = n`, `col_start = 0`.
fn apply_pivots_to_rhs<T: GpuFloat>(
    handle: &SolverHandle,
    b: &mut DeviceBuffer<T>,
    pivots: &DeviceBuffer<i32>,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    launch_pivot_swap::<T>(handle, b, pivots, 0, n, 0, nrhs, n)
}

/// Launches the device row-swap kernel that replays the `jb` transpositions
/// `pivots[j..j+jb]` over the column slab `[col_start, col_start + col_count)`.
#[allow(clippy::too_many_arguments)]
fn launch_pivot_swap<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    pivots: &DeviceBuffer<i32>,
    j: u32,
    jb: u32,
    col_start: u32,
    col_count: u32,
    lda: u32,
) -> SolverResult<()> {
    let sm = handle.sm_version();
    let ptx = emit_pivot_swap::<T>(sm)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &pivot_swap_name::<T>())?;

    let num_blocks = col_count.div_ceil(SOLVER_BLOCK_SIZE).max(1);
    let params = LaunchParams::new(num_blocks, SOLVER_BLOCK_SIZE);

    let a_ptr = a.as_device_ptr();
    let pivots_ptr = pivots.as_device_ptr();
    let args = (a_ptr, pivots_ptr, j, jb, col_start, col_count, lda);
    kernel.launch(&params, handle.stream(), &args)?;

    Ok(())
}

/// Launches the panel triangular solve `B := L^{-1} B`, where `L` is the
/// `jb x jb` unit-lower-triangular block (leading dimension `ldl`) and `B` is
/// the `jb x rcols` block (leading dimension `ldb`), both column-major.
#[allow(clippy::too_many_arguments)]
fn launch_trsm_unit_lower<T: GpuFloat>(
    handle: &SolverHandle,
    l_ptr: u64,
    b_ptr: u64,
    jb: u32,
    rcols: u32,
    ldl: u32,
    ldb: u32,
) -> SolverResult<()> {
    if rcols == 0 || jb == 0 {
        return Ok(());
    }
    let sm = handle.sm_version();
    let ptx = emit_trsm_unit_lower::<T>(sm)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &trsm_unit_lower_name::<T>())?;

    let num_blocks = rcols.div_ceil(SOLVER_BLOCK_SIZE).max(1);
    let params = LaunchParams::new(num_blocks, SOLVER_BLOCK_SIZE);
    let args = (l_ptr, b_ptr, jb, rcols, ldl, ldb);
    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Launches the trailing rank-`kk` update `C := C - A * B`, where `A` is
/// `rrows x kk` (leading dimension `lda`), `B` is `kk x rcols` (leading
/// dimension `ldb`) and `C` is `rrows x rcols` (leading dimension `ldc`), all
/// column-major.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_gemm_update<T: GpuFloat>(
    handle: &SolverHandle,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    rrows: u32,
    rcols: u32,
    kk: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
) -> SolverResult<()> {
    if rrows == 0 || rcols == 0 || kk == 0 {
        return Ok(());
    }
    let sm = handle.sm_version();
    let ptx = emit_gemm_update::<T>(sm)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &gemm_update_name::<T>())?;

    // 2-D grid: the kernel maps `col` to the X dimension (bounded by `rcols`)
    // and `row` to the Y dimension (bounded by `rrows`) via
    // `global_thread_id_2d`. The grid must therefore size X by `rcols` and Y by
    // `rrows`; swapping them silently drops part of `C` for non-square trailing
    // blocks larger than one tile.
    const TILE: u32 = 16;
    let grid_x = rcols.div_ceil(TILE).max(1);
    let grid_y = rrows.div_ceil(TILE).max(1);
    let params = LaunchParams::new((grid_x, grid_y), (TILE, TILE));
    let args = (a_ptr, b_ptr, c_ptr, rrows, rcols, kk, lda, ldb, ldc);
    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

fn panel_lu_name<T: GpuFloat>(block_size: u32) -> String {
    format!("solver_panel_lu_{}_{}", T::NAME, block_size)
}

fn pivot_swap_name<T: GpuFloat>() -> String {
    format!("solver_pivot_swap_{}", T::NAME)
}

fn trsm_unit_lower_name<T: GpuFloat>() -> String {
    format!("solver_lu_trsm_ll_{}", T::NAME)
}

fn gemm_update_name<T: GpuFloat>() -> String {
    format!("solver_lu_gemm_update_{}", T::NAME)
}

/// Emits PTX for an in-place panel triangular solve `B := L^{-1} B` with `L`
/// unit-lower-triangular (`jb x jb`, leading dimension `ldl`) and `B`
/// (`jb x rcols`, leading dimension `ldb`), column-major.
///
/// Thread `gid` (for `gid < rcols`) owns column `gid` of `B` and performs the
/// forward substitution `B[i,c] -= sum_{k<i} L[i,k] * B[k,c]` for `i = 0..jb`
/// (the unit diagonal means no division). Column ownership keeps every thread
/// independent of the others, so no barriers are needed; the sequential inner
/// loop honours the data dependence on already-solved rows.
pub(crate) fn emit_trsm_unit_lower<T: GpuFloat>(sm: SmVersion) -> SolverResult<String> {
    let name = trsm_unit_lower_name::<T>();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("l_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("jb", PtxType::U32)
        .param("rcols", PtxType::U32)
        .param("ldl", PtxType::U32)
        .param("ldb", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let rcols = b.load_param_u32("rcols");
            let inactive = b.alloc_reg(PtxType::Pred);
            let done = b.fresh_label("trsm_done");
            b.raw_ptx(&format!("setp.ge.u32 {inactive}, {gid}, {rcols};"));
            b.raw_ptx(&format!("@{inactive} bra {done};"));

            let l_ptr = b.load_param_u64("l_ptr");
            let b_ptr = b.load_param_u64("b_ptr");
            let jb = b.load_param_u32("jb");
            let ldl = b.load_param_u32("ldl");
            let ldb = b.load_param_u32("ldb");

            // Column offset of this thread's B column: bcol = gid * ldb.
            let bcol = b.mul_lo_u32(gid, ldb);

            // i = 0..jb
            let i = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {i}, 0;"));
            let i_loop = b.fresh_label("trsm_i");
            let i_exit = b.fresh_label("trsm_ix");
            b.raw_ptx(&format!("{i_loop}:"));
            let i_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {i_done}, {i}, {jb};"));
            b.raw_ptx(&format!("@{i_done} bra {i_exit};"));

            // acc = B[i, c]
            let bi_idx = b.add_u32(i.clone(), bcol.clone());
            let bi_addr = b.byte_offset_addr(b_ptr.clone(), bi_idx, T::size_u32());
            let acc = b.alloc_reg(T::PTX_TYPE);
            let suffix = T::PTX_TYPE.as_ptx_str();
            let bi_val = load_global_float::<T>(b, bi_addr.clone());
            b.raw_ptx(&format!("mov{suffix} {acc}, {bi_val};"));

            // k = 0..i: acc -= L[i,k] * B[k,c]
            let k = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k}, 0;"));
            let k_loop = b.fresh_label("trsm_k");
            let k_exit = b.fresh_label("trsm_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {i};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));
            // L[i,k] = l_ptr + (i + k*ldl)
            let kldl = b.mul_lo_u32(k.clone(), ldl.clone());
            let lik_idx = b.add_u32(i.clone(), kldl);
            let lik_addr = b.byte_offset_addr(l_ptr.clone(), lik_idx, T::size_u32());
            let lik = load_global_float::<T>(b, lik_addr);
            // B[k,c] = b_ptr + (k + bcol)
            let bk_idx = b.add_u32(k.clone(), bcol.clone());
            let bk_addr = b.byte_offset_addr(b_ptr.clone(), bk_idx, T::size_u32());
            let bk = load_global_float::<T>(b, bk_addr);
            let prod = mul_float::<T>(b, lik, bk);
            let new_acc = sub_float::<T>(b, acc.clone(), prod);
            b.raw_ptx(&format!("mov{suffix} {acc}, {new_acc};"));
            b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            // B[i,c] = acc  (unit diagonal: no division)
            store_global_float::<T>(b, bi_addr, acc);
            b.raw_ptx(&format!("add.u32 {i}, {i}, 1;"));
            b.raw_ptx(&format!("bra {i_loop};"));
            b.raw_ptx(&format!("{i_exit}:"));
            b.raw_ptx(&format!("{done}:"));

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits PTX for the trailing rank-`kk` update `C := C - A * B`, with `A`
/// (`rrows x kk`, leading dimension `lda`), `B` (`kk x rcols`, leading dimension
/// `ldb`) and `C` (`rrows x rcols`, leading dimension `ldc`), all column-major.
///
/// A 2-D grid maps thread `(x, y)` to output element `C[x, y]`; each thread
/// accumulates the dot product over the shared `kk` dimension and subtracts it
/// from `C`. Leading dimensions are honoured explicitly so the kernel is correct
/// for sub-matrices embedded in a larger buffer (`ld* > rows`).
pub(crate) fn emit_gemm_update<T: GpuFloat>(sm: SmVersion) -> SolverResult<String> {
    let name = gemm_update_name::<T>();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("c_ptr", PtxType::U64)
        .param("rrows", PtxType::U32)
        .param("rcols", PtxType::U32)
        .param("kk", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("ldb", PtxType::U32)
        .param("ldc", PtxType::U32)
        .body(move |b| {
            let (row, col) = b.global_thread_id_2d();
            let rrows = b.load_param_u32("rrows");
            let rcols = b.load_param_u32("rcols");

            let oob_r = b.alloc_reg(PtxType::Pred);
            let oob_c = b.alloc_reg(PtxType::Pred);
            let done = b.fresh_label("gemm_done");
            b.raw_ptx(&format!("setp.ge.u32 {oob_r}, {row}, {rrows};"));
            b.raw_ptx(&format!("@{oob_r} bra {done};"));
            b.raw_ptx(&format!("setp.ge.u32 {oob_c}, {col}, {rcols};"));
            b.raw_ptx(&format!("@{oob_c} bra {done};"));

            let a_ptr = b.load_param_u64("a_ptr");
            let b_ptr = b.load_param_u64("b_ptr");
            let c_ptr = b.load_param_u64("c_ptr");
            let kk = b.load_param_u32("kk");
            let lda = b.load_param_u32("lda");
            let ldb = b.load_param_u32("ldb");
            let ldc = b.load_param_u32("ldc");
            let suffix = T::PTX_TYPE.as_ptx_str();

            // acc = 0
            let acc = b.alloc_reg(T::PTX_TYPE);
            let zero_lit = if T::SIZE == 8 {
                "0d0000000000000000"
            } else {
                "0f00000000"
            };
            b.raw_ptx(&format!("mov{suffix} {acc}, {zero_lit};"));

            // B column offset: col * ldb.
            let bcol = b.mul_lo_u32(col.clone(), ldb);

            // k = 0..kk: acc += A[row,k] * B[k,col]
            let k = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k}, 0;"));
            let k_loop = b.fresh_label("gemm_k");
            let k_exit = b.fresh_label("gemm_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {kk};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));
            // A[row,k] = a_ptr + (row + k*lda)
            let klda = b.mul_lo_u32(k.clone(), lda.clone());
            let ark_idx = b.add_u32(row.clone(), klda);
            let ark_addr = b.byte_offset_addr(a_ptr.clone(), ark_idx, T::size_u32());
            let ark = load_global_float::<T>(b, ark_addr);
            // B[k,col] = b_ptr + (k + bcol)
            let bkc_idx = b.add_u32(k.clone(), bcol.clone());
            let bkc_addr = b.byte_offset_addr(b_ptr.clone(), bkc_idx, T::size_u32());
            let bkc = load_global_float::<T>(b, bkc_addr);
            let new_acc = fma_float::<T>(b, ark, bkc, acc.clone());
            b.raw_ptx(&format!("mov{suffix} {acc}, {new_acc};"));
            b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            // C[row,col] = C[row,col] - acc
            let ccol = b.mul_lo_u32(col.clone(), ldc);
            let c_idx = b.add_u32(row.clone(), ccol);
            let c_addr = b.byte_offset_addr(c_ptr.clone(), c_idx, T::size_u32());
            let c_val = load_global_float::<T>(b, c_addr.clone());
            let updated = sub_float::<T>(b, c_val, acc);
            store_global_float::<T>(b, c_addr, updated);
            b.raw_ptx(&format!("{done}:"));

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits a float immediate of type `T` into a fresh register.
///
/// Uses PTX hexadecimal float literals (`0d…` for `f64`, `0f…` for `f32`) so the
/// exact bit pattern is preserved across the assembler.
fn emit_float_const<T: GpuFloat>(b: &mut BodyBuilder<'_>, value: f64) -> Register {
    let dst = b.alloc_reg(T::PTX_TYPE);
    if T::SIZE == 8 {
        let bits = value.to_bits();
        b.raw_ptx(&format!("mov.f64 {dst}, 0d{bits:016X};"));
    } else {
        let bits = (value as f32).to_bits();
        b.raw_ptx(&format!("mov.f32 {dst}, 0f{bits:08X};"));
    }
    dst
}

/// Computes the device address of a panel element given the precomputed column
/// offset `col_off = col * lda` (column-major): `base + (row + col_off) *
/// sizeof(T)`.
fn panel_elem_addr<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    base: &Register,
    row: &Register,
    col_off: &Register,
) -> Register {
    let idx = b.add_u32(row.clone(), col_off.clone());
    b.byte_offset_addr(base.clone(), idx, T::size_u32())
}

/// Emits PTX for a single-CTA, in-place panel LU factorization kernel
/// (right-looking Doolittle with partial pivoting).
///
/// Operates directly on the `m x panel_cols` panel held in global memory
/// (column-major, leading dimension `lda`). Thread `t` owns panel column `t`
/// (`panel_cols <= LU_BLOCK_SIZE <= SOLVER_BLOCK_SIZE`). For each pivot column
/// `k = 0..panel_cols` the four phases — pivot search, row swap, column scaling
/// and the rank-1 trailing update — are separated by block-wide barriers; the
/// only cross-thread dependency (reading the published pivot column `k`) is
/// therefore correctly ordered. The owner of `k` records the absolute pivot
/// index `j + pivot_row` into `pivots[j + k]` and, on a (near-)zero pivot,
/// `min`-reduces `k + 1` into `info` so the host can report the first singular
/// column.
pub(crate) fn emit_panel_lu<T: GpuFloat>(sm: SmVersion, panel_cols: u32) -> SolverResult<String> {
    let name = panel_lu_name::<T>(panel_cols);
    let suffix = T::PTX_TYPE.as_ptx_str();
    // Threshold below which a pivot is treated as numerically zero (singular).
    let tiny: f64 = 1e-30;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("panel_ptr", PtxType::U64)
        .param("pivots_ptr", PtxType::U64)
        .param("info_ptr", PtxType::U64)
        .param("panel_rows", PtxType::U32)
        .param("panel_cols", PtxType::U32)
        .param("j", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let m = b.load_param_u32("panel_rows");
            let nc = b.load_param_u32("panel_cols");
            let j_reg = b.load_param_u32("j");
            let lda = b.load_param_u32("lda");
            let base = b.load_param_u64("panel_ptr");
            let piv_base = b.load_param_u64("pivots_ptr");
            let info_base = b.load_param_u64("info_ptr");

            // Persistent loop registers.
            let k = b.alloc_reg(PtxType::U32);
            let prow = b.alloc_reg(PtxType::U32);
            let rr = b.alloc_reg(PtxType::U32);
            let colk = b.alloc_reg(PtxType::U32);
            let colt = b.alloc_reg(PtxType::U32);
            let maxabs = b.alloc_reg(T::PTX_TYPE);

            b.raw_ptx(&format!("mov.u32 {k}, 0;"));

            let k_loop = b.fresh_label("lu_k");
            let k_exit = b.fresh_label("lu_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {nc};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));

            // col offset of pivot column k: colk = k * lda.
            b.raw_ptx(&format!("mul.lo.u32 {colk}, {k}, {lda};"));

            // ---- Phase 1: pivot search (rows k..m of column k) ----------------
            b.raw_ptx(&format!("mov.u32 {prow}, {k};"));
            let p1_end = b.fresh_label("lu_p1e");
            let not_active1 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {not_active1}, {tid}, {nc};"));
            b.raw_ptx(&format!("@{not_active1} bra {p1_end};"));
            {
                let zero_lit = if T::SIZE == 8 {
                    "0d0000000000000000"
                } else {
                    "0f00000000"
                };
                b.raw_ptx(&format!("mov{suffix} {maxabs}, {zero_lit};"));
                b.raw_ptx(&format!("mov.u32 {rr}, {k};"));
                let s_loop = b.fresh_label("lu_s");
                let s_exit = b.fresh_label("lu_sx");
                b.raw_ptx(&format!("{s_loop}:"));
                let s_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {s_done}, {rr}, {m};"));
                b.raw_ptx(&format!("@{s_done} bra {s_exit};"));
                let addr = panel_elem_addr::<T>(b, &base, &rr, &colk);
                let v = load_global_float::<T>(b, addr);
                let av = abs_float::<T>(b, v);
                let gt = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.gt{suffix} {gt}, {av}, {maxabs};"));
                b.raw_ptx(&format!("@{gt} mov{suffix} {maxabs}, {av};"));
                b.raw_ptx(&format!("@{gt} mov.u32 {prow}, {rr};"));
                b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                b.raw_ptx(&format!("bra {s_loop};"));
                b.raw_ptx(&format!("{s_exit}:"));

                // Owner of column k records the absolute pivot index.
                let not_owner = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {not_owner}, {tid}, {k};"));
                let skip_write = b.fresh_label("lu_pw");
                b.raw_ptx(&format!("@{not_owner} bra {skip_write};"));
                let pglobal = b.add_u32(j_reg.clone(), prow.clone());
                let pidx = b.add_u32(j_reg.clone(), k.clone());
                let paddr = b.byte_offset_addr(piv_base.clone(), pidx, 4);
                b.raw_ptx(&format!("st.global.u32 [{paddr}], {pglobal};"));
                b.raw_ptx(&format!("{skip_write}:"));
            }
            b.raw_ptx(&format!("{p1_end}:"));
            b.bar_sync(0);

            // ---- Phase 2: swap rows k / prow in this thread's column ----------
            b.raw_ptx(&format!("mul.lo.u32 {colt}, {tid}, {lda};"));
            let p2_end = b.fresh_label("lu_p2e");
            let not_active2 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {not_active2}, {tid}, {nc};"));
            b.raw_ptx(&format!("@{not_active2} bra {p2_end};"));
            let no_swap = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {no_swap}, {prow}, {k};"));
            b.raw_ptx(&format!("@{no_swap} bra {p2_end};"));
            {
                let ak_addr = panel_elem_addr::<T>(b, &base, &k, &colt);
                let ap_addr = panel_elem_addr::<T>(b, &base, &prow, &colt);
                let vk = load_global_float::<T>(b, ak_addr.clone());
                let vp = load_global_float::<T>(b, ap_addr.clone());
                store_global_float::<T>(b, ak_addr, vp);
                store_global_float::<T>(b, ap_addr, vk);
            }
            b.raw_ptx(&format!("{p2_end}:"));
            b.bar_sync(0);

            // ---- Phase 3: scale sub-diagonal of column k (owner of k) ---------
            let p3_end = b.fresh_label("lu_p3e");
            let not_owner3 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ne.u32 {not_owner3}, {tid}, {k};"));
            b.raw_ptx(&format!("@{not_owner3} bra {p3_end};"));
            {
                let akk_addr = panel_elem_addr::<T>(b, &base, &k, &colk);
                let pivot = load_global_float::<T>(b, akk_addr);
                let apv = abs_float::<T>(b, pivot.clone());
                let tiny_reg = emit_float_const::<T>(b, tiny);
                let singular = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.le{suffix} {singular}, {apv}, {tiny_reg};"));
                let do_scale = b.fresh_label("lu_sc");
                let after_scale = b.fresh_label("lu_sce");
                b.raw_ptx(&format!("@!{singular} bra {do_scale};"));
                // Singular pivot: record first singular column via min-reduction.
                let one = one_u32(b);
                let kp1 = b.add_u32(k.clone(), one);
                b.raw_ptx(&format!("red.global.min.u32 [{info_base}], {kp1};"));
                b.raw_ptx(&format!("bra {after_scale};"));
                b.raw_ptx(&format!("{do_scale}:"));
                b.raw_ptx(&format!("add.u32 {rr}, {k}, 1;"));
                let c_loop = b.fresh_label("lu_c");
                let c_exit = b.fresh_label("lu_cx");
                b.raw_ptx(&format!("{c_loop}:"));
                let c_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {c_done}, {rr}, {m};"));
                b.raw_ptx(&format!("@{c_done} bra {c_exit};"));
                let addr = panel_elem_addr::<T>(b, &base, &rr, &colk);
                let v = load_global_float::<T>(b, addr.clone());
                let scaled = div_float::<T>(b, v, pivot.clone());
                store_global_float::<T>(b, addr, scaled);
                b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                b.raw_ptx(&format!("bra {c_loop};"));
                b.raw_ptx(&format!("{c_exit}:"));
                b.raw_ptx(&format!("{after_scale}:"));
            }
            b.raw_ptx(&format!("{p3_end}:"));
            b.bar_sync(0);

            // ---- Phase 4: rank-1 trailing update (owners k < tid < nc) --------
            let p4_end = b.fresh_label("lu_p4e");
            let le_k = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.le.u32 {le_k}, {tid}, {k};"));
            b.raw_ptx(&format!("@{le_k} bra {p4_end};"));
            let ge_nc = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {ge_nc}, {tid}, {nc};"));
            b.raw_ptx(&format!("@{ge_nc} bra {p4_end};"));
            {
                // U[k, tid] lives at panel row k of this thread's column.
                let akc_addr = panel_elem_addr::<T>(b, &base, &k, &colt);
                let akc = load_global_float::<T>(b, akc_addr);
                b.raw_ptx(&format!("add.u32 {rr}, {k}, 1;"));
                let t_loop = b.fresh_label("lu_t");
                let t_exit = b.fresh_label("lu_tx");
                b.raw_ptx(&format!("{t_loop}:"));
                let t_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {t_done}, {rr}, {m};"));
                b.raw_ptx(&format!("@{t_done} bra {t_exit};"));
                let ark_addr = panel_elem_addr::<T>(b, &base, &rr, &colk);
                let ark = load_global_float::<T>(b, ark_addr);
                let arc_addr = panel_elem_addr::<T>(b, &base, &rr, &colt);
                let arc = load_global_float::<T>(b, arc_addr.clone());
                let prod = mul_float::<T>(b, ark, akc.clone());
                let upd = sub_float::<T>(b, arc, prod);
                store_global_float::<T>(b, arc_addr, upd);
                b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                b.raw_ptx(&format!("bra {t_loop};"));
                b.raw_ptx(&format!("{t_exit}:"));
            }
            b.raw_ptx(&format!("{p4_end}:"));
            b.bar_sync(0);

            b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits PTX for a device row-permutation (pivot replay) kernel.
///
/// Thread `gid` (for `gid < col_count`) owns column `col_start + gid` and
/// replays the `jb` transpositions `pivots[j + t]` (`t = 0..jb`) in increasing
/// order, swapping rows `j + t` and `pivots[j + t]` within its column. Column
/// ownership makes every thread independent (no barriers needed); the sequential
/// inner loop preserves the composition order of partial-pivoting swaps.
pub(crate) fn emit_pivot_swap<T: GpuFloat>(sm: SmVersion) -> SolverResult<String> {
    let name = pivot_swap_name::<T>();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("a_ptr", PtxType::U64)
        .param("pivots_ptr", PtxType::U64)
        .param("j", PtxType::U32)
        .param("jb", PtxType::U32)
        .param("col_start", PtxType::U32)
        .param("col_count", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let col_count = b.load_param_u32("col_count");

            let inactive = b.alloc_reg(PtxType::Pred);
            let done = b.fresh_label("sw_done");
            b.raw_ptx(&format!("setp.ge.u32 {inactive}, {gid}, {col_count};"));
            b.raw_ptx(&format!("@{inactive} bra {done};"));

            let a_ptr = b.load_param_u64("a_ptr");
            let piv_base = b.load_param_u64("pivots_ptr");
            let j_reg = b.load_param_u32("j");
            let jb = b.load_param_u32("jb");
            let col_start = b.load_param_u32("col_start");
            let lda = b.load_param_u32("lda");

            // Column base address: a_ptr + (col_start + gid) * lda * sizeof(T).
            let col_idx = b.add_u32(gid, col_start);
            let col_off = b.mul_lo_u32(col_idx, lda);
            let col_base = b.byte_offset_addr(a_ptr, col_off, T::size_u32());

            // t = 0..jb: swap rows (j + t) and pivots[j + t].
            let t = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {t}, 0;"));
            let t_loop = b.fresh_label("sw_t");
            let t_exit = b.fresh_label("sw_tx");
            b.raw_ptx(&format!("{t_loop}:"));
            let t_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {t_done}, {t}, {jb};"));
            b.raw_ptx(&format!("@{t_done} bra {t_exit};"));

            let row = b.add_u32(j_reg.clone(), t.clone());
            let piv_addr = b.byte_offset_addr(piv_base.clone(), row.clone(), 4);
            let piv = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("ld.global.u32 {piv}, [{piv_addr}];"));

            let same = b.alloc_reg(PtxType::Pred);
            let skip = b.fresh_label("sw_skip");
            b.raw_ptx(&format!("setp.eq.u32 {same}, {piv}, {row};"));
            b.raw_ptx(&format!("@{same} bra {skip};"));

            let row_addr = b.byte_offset_addr(col_base.clone(), row.clone(), T::size_u32());
            let piv_row_addr = b.byte_offset_addr(col_base.clone(), piv.clone(), T::size_u32());
            let vr = load_global_float::<T>(b, row_addr.clone());
            let vp = load_global_float::<T>(b, piv_row_addr.clone());
            store_global_float::<T>(b, row_addr, vp);
            store_global_float::<T>(b, piv_row_addr, vr);

            b.raw_ptx(&format!("{skip}:"));
            b.raw_ptx(&format!("add.u32 {t}, {t}, 1;"));
            b.raw_ptx(&format!("bra {t_loop};"));
            b.raw_ptx(&format!("{t_exit}:"));
            b.raw_ptx(&format!("{done}:"));

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Allocates a `u32` register holding the constant `1`.
fn one_u32(b: &mut BodyBuilder<'_>) -> Register {
    let r = b.alloc_reg(PtxType::U32);
    b.raw_ptx(&format!("mov.u32 {r}, 1;"));
    r
}

// ---------------------------------------------------------------------------
// Tests (in a separate file to stay under the 2000-line refactoring limit)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "lu_tests.rs"]
mod tests;
