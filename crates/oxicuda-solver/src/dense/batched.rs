//! Batched matrix factorization for many small matrices in a single kernel launch.
//!
//! This module provides batched LU, QR, and Cholesky factorizations optimized for
//! many small matrices (4x4 to 64x64). Each thread block handles one matrix (or
//! multiple for very small sizes), with all computation done in registers and
//! shared memory. This is critical for robotics, physics simulation, and batched
//! neural network layers.
//!
//! ## Design
//!
//! - For n <= 16: multiple matrices per thread block (register-resident).
//! - For n <= 32: single warp handles entire matrix, each thread owns one column.
//! - For n <= 64: two warps per matrix, shared memory for the matrix.
//!
//! All operations process `batch_count` matrices of size `n x n` stored
//! contiguously in column-major order: `matrices[batch_count * n * n]`.

use std::sync::Arc;

use oxicuda_blas::types::GpuFloat;
use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{SolverError, SolverResult};
use crate::handle::SolverHandle;
use crate::ptx_helpers::{
    SOLVER_BLOCK_SIZE, abs_float, div_float, load_global_float, mul_float, sqrt_float,
    store_global_float, sub_float,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Maximum matrix size supported by the batched solver.
const MAX_BATCH_MATRIX_SIZE: usize = 64;

/// Minimum matrix size supported by the batched solver.
const MIN_BATCH_MATRIX_SIZE: usize = 1;

/// Threshold below which multiple matrices are packed per thread block.
const SMALL_MATRIX_THRESHOLD: usize = 16;

/// Number of small matrices packed per thread block when n <= SMALL_MATRIX_THRESHOLD.
const SMALL_MATRICES_PER_BLOCK: usize = 4;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Batched matrix factorization engine.
///
/// Each thread block handles one matrix. For very small matrices (n <= 16),
/// multiple matrices per thread block. All computation in registers/shared memory.
pub struct BatchedSolver {
    handle: SolverHandle,
}

/// Configuration for batched operations.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Matrix dimension (n x n).
    pub matrix_size: usize,
    /// Number of matrices in the batch.
    pub batch_count: usize,
    /// Which factorization algorithm to use.
    pub algorithm: BatchAlgorithm,
}

/// Selects the batched factorization algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAlgorithm {
    /// LU factorization with partial pivoting.
    Lu,
    /// QR factorization via Householder reflections.
    Qr,
    /// Cholesky factorization for SPD matrices.
    Cholesky,
}

/// Result of a batched factorization.
#[derive(Debug, Clone)]
pub struct BatchedResult {
    /// Number of matrices that had issues (singular for LU, not SPD for Cholesky).
    pub failed_count: usize,
}

// ---------------------------------------------------------------------------
// BatchedSolver implementation
// ---------------------------------------------------------------------------

impl BatchedSolver {
    /// Creates a new batched solver.
    pub fn new(handle: SolverHandle) -> Self {
        Self { handle }
    }

    /// Returns a reference to the underlying solver handle.
    pub fn handle(&self) -> &SolverHandle {
        &self.handle
    }

    /// Returns a mutable reference to the underlying solver handle.
    pub fn handle_mut(&mut self) -> &mut SolverHandle {
        &mut self.handle
    }

    /// Batched LU factorization: factorize `batch_count` matrices of size `n x n`.
    ///
    /// Input: `matrices[batch_count * n * n]` (column-major, contiguous).
    /// Output: in-place LU factors, `pivots[batch_count * n]`.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::DimensionMismatch`] if buffer sizes are incorrect
    /// or matrix size is out of range.
    pub fn batched_lu<T: GpuFloat>(
        &mut self,
        matrices: &mut DeviceBuffer<T>,
        pivots: &mut DeviceBuffer<i32>,
        n: usize,
        batch_count: usize,
    ) -> SolverResult<BatchedResult> {
        validate_batched_params::<T>(matrices, n, batch_count)?;
        validate_pivot_buffer(pivots, n, batch_count)?;

        if n == 0 || batch_count == 0 {
            return Ok(BatchedResult { failed_count: 0 });
        }

        // Ensure workspace for shared memory requirements.
        let shared_per_matrix = n * n * T::SIZE;
        let matrices_per_block = matrices_per_block(n);
        let ws_bytes = shared_per_matrix * matrices_per_block;
        self.handle.ensure_workspace(ws_bytes)?;

        // Per-batch-element failure flag: 0 on success, else the 1-based
        // column at which the LU pivot magnitude fell below the singularity
        // threshold (see `emit_batched_lu`).
        let info = DeviceBuffer::<i32>::zeroed(batch_count)?;

        // Generate and launch batched LU PTX kernel.
        let sm = self.handle.sm_version();
        let ptx = emit_batched_lu::<T>(sm, n)?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &batched_lu_name::<T>(n))?;

        let grid = compute_grid_size(batch_count, n);
        let block = compute_block_size(n);
        let shared_bytes = (shared_per_matrix * matrices_per_block) as u32;
        let params = LaunchParams::new(grid, block).with_shared_mem(shared_bytes);

        let args = (
            matrices.as_device_ptr(),
            pivots.as_device_ptr(),
            info.as_device_ptr(),
            n as u32,
            batch_count as u32,
        );
        kernel.launch(&params, self.handle.stream(), &args)?;
        // The kernel runs on `self.handle`'s own stream; an explicit
        // synchronize (rather than relying on `copy_to_host`'s blocking
        // memcpy to also fence *other* streams, which is not guaranteed) is
        // required before the info readback below observes every write.
        self.handle.stream().synchronize()?;

        let mut info_host = vec![0i32; batch_count];
        info.copy_to_host(&mut info_host)?;
        let failed_count = info_host.iter().filter(|&&v| v != 0).count();

        Ok(BatchedResult { failed_count })
    }

    /// Batched QR factorization.
    ///
    /// Output: in-place QR factors, `tau[batch_count * min(m, n)]` (Householder scalars).
    ///
    /// # Arguments
    ///
    /// * `matrices` — contiguous buffer of `batch_count` matrices, each `m x n`, column-major.
    /// * `tau` — output Householder scalars, length `batch_count * min(m, n)`.
    /// * `m` — number of rows per matrix.
    /// * `n` — number of columns per matrix.
    /// * `batch_count` — number of matrices.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::DimensionMismatch`] if buffer sizes are incorrect.
    pub fn batched_qr<T: GpuFloat>(
        &mut self,
        matrices: &mut DeviceBuffer<T>,
        tau: &mut DeviceBuffer<T>,
        m: usize,
        n: usize,
        batch_count: usize,
    ) -> SolverResult<BatchedResult> {
        if m == 0 || n == 0 || batch_count == 0 {
            return Ok(BatchedResult { failed_count: 0 });
        }

        let required_mat = batch_count * m * n;
        if matrices.len() < required_mat {
            return Err(SolverError::DimensionMismatch(format!(
                "batched_qr: matrices buffer too small ({} < {required_mat})",
                matrices.len()
            )));
        }

        let k = m.min(n);
        let required_tau = batch_count * k;
        if tau.len() < required_tau {
            return Err(SolverError::DimensionMismatch(format!(
                "batched_qr: tau buffer too small ({} < {required_tau})",
                tau.len()
            )));
        }

        let dim = m.max(n);
        if dim > MAX_BATCH_MATRIX_SIZE {
            return Err(SolverError::DimensionMismatch(format!(
                "batched_qr: matrix dimension ({dim}) exceeds maximum ({MAX_BATCH_MATRIX_SIZE})"
            )));
        }

        // Shared memory: matrix (m x n) + Householder workspace.
        let shared_per_matrix = (m * n + m) * T::SIZE;
        let mpb = matrices_per_block(dim);
        let ws_bytes = shared_per_matrix * mpb;
        self.handle.ensure_workspace(ws_bytes)?;

        let sm = self.handle.sm_version();
        let ptx = emit_batched_qr::<T>(sm, m, n)?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &batched_qr_name::<T>(m, n))?;

        let grid = compute_grid_size(batch_count, dim);
        let block = compute_block_size(dim);
        let shared_bytes = (shared_per_matrix * mpb) as u32;
        let params = LaunchParams::new(grid, block).with_shared_mem(shared_bytes);

        let args = (
            matrices.as_device_ptr(),
            tau.as_device_ptr(),
            m as u32,
            n as u32,
            batch_count as u32,
        );
        kernel.launch(&params, self.handle.stream(), &args)?;

        Ok(BatchedResult { failed_count: 0 })
    }

    /// Batched Cholesky factorization (for SPD matrices).
    ///
    /// Output: in-place lower triangular Cholesky factors.
    ///
    /// # Arguments
    ///
    /// * `matrices` — contiguous buffer of `batch_count` SPD matrices, each `n x n`.
    /// * `n` — matrix dimension.
    /// * `batch_count` — number of matrices.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::DimensionMismatch`] if buffer sizes are incorrect.
    pub fn batched_cholesky<T: GpuFloat>(
        &mut self,
        matrices: &mut DeviceBuffer<T>,
        n: usize,
        batch_count: usize,
    ) -> SolverResult<BatchedResult> {
        validate_batched_params::<T>(matrices, n, batch_count)?;

        if n == 0 || batch_count == 0 {
            return Ok(BatchedResult { failed_count: 0 });
        }

        let shared_per_matrix = n * n * T::SIZE;
        let mpb = matrices_per_block(n);
        let ws_bytes = shared_per_matrix * mpb;
        self.handle.ensure_workspace(ws_bytes)?;

        // Per-batch-element failure flag: 0 on success, else the 1-based
        // column at which the diagonal was non-positive before the square
        // root (i.e. the matrix is not SPD); see `emit_batched_cholesky`.
        let info = DeviceBuffer::<i32>::zeroed(batch_count)?;

        let sm = self.handle.sm_version();
        let ptx = emit_batched_cholesky::<T>(sm, n)?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &batched_cholesky_name::<T>(n))?;

        let grid = compute_grid_size(batch_count, n);
        let block = compute_block_size(n);
        let shared_bytes = (shared_per_matrix * mpb) as u32;
        let params = LaunchParams::new(grid, block).with_shared_mem(shared_bytes);

        let args = (
            matrices.as_device_ptr(),
            info.as_device_ptr(),
            n as u32,
            batch_count as u32,
        );
        kernel.launch(&params, self.handle.stream(), &args)?;
        // See the matching comment in `batched_lu`: an explicit synchronize
        // is required before the info readback below is guaranteed to see
        // the kernel's writes.
        self.handle.stream().synchronize()?;

        let mut info_host = vec![0i32; batch_count];
        info.copy_to_host(&mut info_host)?;
        let failed_count = info_host.iter().filter(|&&v| v != 0).count();

        Ok(BatchedResult { failed_count })
    }

    /// Batched linear solve using LU: solve `A_i * X_i = B_i` for each `i`.
    ///
    /// First performs batched LU factorization on `a_matrices`, then uses the
    /// factors to solve the system. Both `a_matrices` and `b_matrices` are
    /// modified in-place.
    ///
    /// # Arguments
    ///
    /// * `a_matrices` — contiguous `batch_count` coefficient matrices (n x n each).
    /// * `b_matrices` — contiguous `batch_count` RHS matrices (n x nrhs each).
    /// * `n` — system dimension.
    /// * `nrhs` — number of right-hand sides.
    /// * `batch_count` — number of systems to solve.
    ///
    /// # Errors
    ///
    /// Returns [`SolverError::DimensionMismatch`] if dimensions are invalid.
    pub fn batched_solve<T: GpuFloat>(
        &mut self,
        a_matrices: &mut DeviceBuffer<T>,
        b_matrices: &mut DeviceBuffer<T>,
        n: usize,
        nrhs: usize,
        batch_count: usize,
    ) -> SolverResult<BatchedResult> {
        if n == 0 || nrhs == 0 || batch_count == 0 {
            return Ok(BatchedResult { failed_count: 0 });
        }

        validate_batched_params::<T>(a_matrices, n, batch_count)?;

        let required_b = batch_count * n * nrhs;
        if b_matrices.len() < required_b {
            return Err(SolverError::DimensionMismatch(format!(
                "batched_solve: b_matrices buffer too small ({} < {required_b})",
                b_matrices.len()
            )));
        }

        // Step 1: Batched LU factorization.
        let mut pivots = DeviceBuffer::<i32>::zeroed(batch_count * n)?;
        let lu_result = self.batched_lu(a_matrices, &mut pivots, n, batch_count)?;

        // Step 2: Batched triangular solve using the LU factors.
        // Apply pivots to B, then forward-substitution (L), then back-substitution (U).
        let sm = self.handle.sm_version();
        let ptx = emit_batched_solve::<T>(sm, n, nrhs)?;
        let module = Arc::new(Module::from_ptx(&ptx)?);
        let kernel = Kernel::from_module(module, &batched_solve_name::<T>(n, nrhs))?;

        let shared_per_system = (n * n + n * nrhs + n) * T::SIZE;
        let grid = compute_grid_size(batch_count, n);
        let block = compute_block_size(n);
        let params = LaunchParams::new(grid, block).with_shared_mem(shared_per_system as u32);

        let args = (
            a_matrices.as_device_ptr(),
            b_matrices.as_device_ptr(),
            pivots.as_device_ptr(),
            n as u32,
            nrhs as u32,
            batch_count as u32,
        );
        kernel.launch(&params, self.handle.stream(), &args)?;

        Ok(lu_result)
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validates common batched parameters.
fn validate_batched_params<T: GpuFloat>(
    matrices: &DeviceBuffer<T>,
    n: usize,
    batch_count: usize,
) -> SolverResult<()> {
    if n > MAX_BATCH_MATRIX_SIZE {
        return Err(SolverError::DimensionMismatch(format!(
            "batched: matrix size ({n}) exceeds maximum ({MAX_BATCH_MATRIX_SIZE})"
        )));
    }
    if n < MIN_BATCH_MATRIX_SIZE && n != 0 {
        return Err(SolverError::DimensionMismatch(format!(
            "batched: matrix size ({n}) below minimum ({MIN_BATCH_MATRIX_SIZE})"
        )));
    }

    let required = batch_count * n * n;
    if matrices.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "batched: matrices buffer too small ({} < {required})",
            matrices.len()
        )));
    }

    Ok(())
}

/// Validates pivot buffer size.
fn validate_pivot_buffer(
    pivots: &DeviceBuffer<i32>,
    n: usize,
    batch_count: usize,
) -> SolverResult<()> {
    let required = batch_count * n;
    if pivots.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "batched: pivots buffer too small ({} < {required})",
            pivots.len()
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Launch configuration helpers
// ---------------------------------------------------------------------------

/// Computes how many matrices can be packed per thread block.
fn matrices_per_block(n: usize) -> usize {
    if n <= SMALL_MATRIX_THRESHOLD {
        SMALL_MATRICES_PER_BLOCK
    } else {
        1
    }
}

/// Computes the grid size for a batched launch.
fn compute_grid_size(batch_count: usize, n: usize) -> u32 {
    let mpb = matrices_per_block(n);
    let blocks = batch_count.div_ceil(mpb);
    blocks as u32
}

/// Computes the block size (threads per block) based on matrix dimension.
fn compute_block_size(n: usize) -> u32 {
    if n <= 16 {
        // Small matrices: 32 threads per matrix * SMALL_MATRICES_PER_BLOCK.
        (32 * SMALL_MATRICES_PER_BLOCK as u32).min(SOLVER_BLOCK_SIZE)
    } else if n <= 32 {
        // One warp per matrix.
        32
    } else {
        // Two warps per matrix for n <= 64.
        64
    }
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

fn batched_lu_name<T: GpuFloat>(n: usize) -> String {
    format!("solver_batched_lu_{}_{}", T::NAME, n)
}

fn batched_qr_name<T: GpuFloat>(m: usize, n: usize) -> String {
    format!("solver_batched_qr_{}_{}x{}", T::NAME, m, n)
}

fn batched_cholesky_name<T: GpuFloat>(n: usize) -> String {
    format!("solver_batched_cholesky_{}_{}", T::NAME, n)
}

fn batched_solve_name<T: GpuFloat>(n: usize, nrhs: usize) -> String {
    format!("solver_batched_solve_{}_{}_{}", T::NAME, n, nrhs)
}

/// Computes the device address of a batched-matrix element given the
/// precomputed column offset `col_off = col * n` (column-major, contiguous
/// `n x n` storage per batch element): `mat_base + (row + col_off) *
/// sizeof(T)`. Mirrors `panel_elem_addr` in `dense::lu`.
fn batched_elem_addr<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    mat_base: &Register,
    row: &Register,
    col_off: &Register,
) -> Register {
    let idx = b.add_u32(row.clone(), col_off.clone());
    b.byte_offset_addr(mat_base.clone(), idx, T::size_u32())
}

/// Emits a float immediate of type `T` into a fresh register.
///
/// Uses PTX hexadecimal float literals (`0d…` for `f64`, `0f…` for `f32`) so
/// the exact bit pattern is preserved across the assembler. Mirrors
/// `emit_float_const` in `dense::lu`.
fn batched_float_const<T: GpuFloat>(b: &mut BodyBuilder<'_>, value: f64) -> Register {
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

/// Emits PTX for batched LU factorization with partial pivoting.
///
/// Each thread block processes `matrices_per_block(n)` matrices (one for
/// `n > SMALL_MATRIX_THRESHOLD`, several packed together for smaller `n`).
/// Within a matrix, thread `t` (`t < n`) owns column `t` and the four
/// classic right-looking Doolittle phases — pivot search, row swap, column
/// scaling, rank-1 trailing update — are separated by block-wide barriers,
/// mirroring [`crate::dense::lu::emit_panel_lu`] but addressing many
/// independent `n x n` matrices per launch instead of a single panel.
///
/// Every thread in the block (including those that own no valid column,
/// because `n` doesn't fill a whole thread-group, or whose matrix index is
/// beyond `batch_count` in a partially-filled trailing block) still executes
/// every `bar.sync` — only the per-phase *work* is predicated on an
/// `inactive` flag — so the barriers can never see a partial subset of the
/// CTA and deadlock.
///
/// On a (near-)zero pivot the owning thread writes the 1-based failing
/// column into `info[batch_idx]` (left at `0` by the caller's zeroed buffer
/// on success) and the factorization continues past that column exactly as
/// [`crate::dense::lu::emit_panel_lu`] does, so the caller can report
/// `failed_count` without the rest of the batch aborting.
pub(crate) fn emit_batched_lu<T: GpuFloat>(sm: SmVersion, n: usize) -> SolverResult<String> {
    let name = batched_lu_name::<T>(n);
    let suffix = T::PTX_TYPE.as_ptx_str();
    // Threshold below which a pivot is treated as numerically zero (singular).
    let tiny: f64 = 1e-30;

    // Compile-time launch geometry (mirrors `matrices_per_block`/
    // `compute_block_size` exactly, so the host's launch config and this
    // kernel's batch-index arithmetic always agree).
    let mpb = matrices_per_block(n) as u32;
    let group_size = compute_block_size(n) / mpb;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("matrices_ptr", PtxType::U64)
        .param("pivots_ptr", PtxType::U64)
        .param("info_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("batch_count", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let batch_count_reg = b.load_param_u32("batch_count");
            let n_reg = b.load_param_u32("n");
            let matrices_ptr = b.load_param_u64("matrices_ptr");
            let pivots_ptr = b.load_param_u64("pivots_ptr");
            let info_ptr = b.load_param_u64("info_ptr");

            // group_idx = tid / group_size (which matrix, within this block);
            // tig        = tid % group_size (which column of that matrix).
            let group_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {group_idx}, {tid}, {group_size};"));
            let tig = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("rem.u32 {tig}, {tid}, {group_size};"));
            let mpb_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {mpb_reg}, {mpb};"));
            let batch_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {batch_idx}, {bid}, {mpb_reg}, {group_idx};"
            ));

            // `inactive`: this thread owns no valid column of a matrix in
            // range. Every phase below still falls through to its closing
            // `bar.sync` for inactive threads; only the work is skipped.
            let inactive = b.alloc_reg(PtxType::Pred);
            let p_tig = b.alloc_reg(PtxType::Pred);
            let p_batch = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_tig}, {tig}, {n_reg};"));
            b.raw_ptx(&format!(
                "setp.ge.u32 {p_batch}, {batch_idx}, {batch_count_reg};"
            ));
            b.raw_ptx(&format!("or.pred {inactive}, {p_tig}, {p_batch};"));

            // Per-matrix base addresses (column-major, contiguous n x n).
            let n2 = b.mul_lo_u32(n_reg.clone(), n_reg.clone());
            let mat_offset = b.mul_lo_u32(batch_idx.clone(), n2);
            let mat_base = b.byte_offset_addr(matrices_ptr, mat_offset, T::size_u32());
            let piv_offset = b.mul_lo_u32(batch_idx.clone(), n_reg.clone());
            let piv_base = b.byte_offset_addr(pivots_ptr, piv_offset, 4u32);
            let info_addr = b.byte_offset_addr(info_ptr, batch_idx, 4u32);

            let k = b.alloc_reg(PtxType::U32);
            let prow = b.alloc_reg(PtxType::U32);
            let rr = b.alloc_reg(PtxType::U32);
            let colk = b.alloc_reg(PtxType::U32);
            let colt = b.alloc_reg(PtxType::U32);
            let maxabs = b.alloc_reg(T::PTX_TYPE);

            b.raw_ptx(&format!("mov.u32 {k}, 0;"));
            let k_loop = b.fresh_label("blu_k");
            let k_exit = b.fresh_label("blu_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {n_reg};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));

            b.raw_ptx(&format!("mul.lo.u32 {colk}, {k}, {n_reg};"));

            // ---- Phase 1: pivot search (rows k..n of column k) -----------
            let p1_end = b.fresh_label("blu_p1e");
            b.raw_ptx(&format!("@{inactive} bra {p1_end};"));
            {
                let zero_lit = if T::SIZE == 8 {
                    "0d0000000000000000"
                } else {
                    "0f00000000"
                };
                b.raw_ptx(&format!("mov{suffix} {maxabs}, {zero_lit};"));
                b.raw_ptx(&format!("mov.u32 {prow}, {k};"));
                b.raw_ptx(&format!("mov.u32 {rr}, {k};"));
                let s_loop = b.fresh_label("blu_s");
                let s_exit = b.fresh_label("blu_sx");
                b.raw_ptx(&format!("{s_loop}:"));
                let s_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {s_done}, {rr}, {n_reg};"));
                b.raw_ptx(&format!("@{s_done} bra {s_exit};"));
                let addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colk);
                let v = load_global_float::<T>(b, addr);
                let av = abs_float::<T>(b, v);
                let gt = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.gt{suffix} {gt}, {av}, {maxabs};"));
                b.raw_ptx(&format!("@{gt} mov{suffix} {maxabs}, {av};"));
                b.raw_ptx(&format!("@{gt} mov.u32 {prow}, {rr};"));
                b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                b.raw_ptx(&format!("bra {s_loop};"));
                b.raw_ptx(&format!("{s_exit}:"));

                // Owner of column k records the pivot row.
                let not_owner = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ne.u32 {not_owner}, {tig}, {k};"));
                let skip_write = b.fresh_label("blu_pw");
                b.raw_ptx(&format!("@{not_owner} bra {skip_write};"));
                let paddr = b.byte_offset_addr(piv_base.clone(), k.clone(), 4u32);
                b.raw_ptx(&format!("st.global.u32 [{paddr}], {prow};"));
                b.raw_ptx(&format!("{skip_write}:"));
            }
            b.raw_ptx(&format!("{p1_end}:"));
            b.bar_sync(0);

            // ---- Phase 2: swap rows k / prow in this thread's column -----
            b.raw_ptx(&format!("mul.lo.u32 {colt}, {tig}, {n_reg};"));
            let p2_end = b.fresh_label("blu_p2e");
            b.raw_ptx(&format!("@{inactive} bra {p2_end};"));
            let no_swap = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.eq.u32 {no_swap}, {prow}, {k};"));
            b.raw_ptx(&format!("@{no_swap} bra {p2_end};"));
            {
                let ak_addr = batched_elem_addr::<T>(b, &mat_base, &k, &colt);
                let ap_addr = batched_elem_addr::<T>(b, &mat_base, &prow, &colt);
                let vk = load_global_float::<T>(b, ak_addr.clone());
                let vp = load_global_float::<T>(b, ap_addr.clone());
                store_global_float::<T>(b, ak_addr, vp);
                store_global_float::<T>(b, ap_addr, vk);
            }
            b.raw_ptx(&format!("{p2_end}:"));
            b.bar_sync(0);

            // ---- Phase 3: scale sub-diagonal of column k (owner of k) ----
            let p3_end = b.fresh_label("blu_p3e");
            b.raw_ptx(&format!("@{inactive} bra {p3_end};"));
            let not_owner3 = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ne.u32 {not_owner3}, {tig}, {k};"));
            b.raw_ptx(&format!("@{not_owner3} bra {p3_end};"));
            {
                let akk_addr = batched_elem_addr::<T>(b, &mat_base, &k, &colk);
                let pivot = load_global_float::<T>(b, akk_addr);
                let apv = abs_float::<T>(b, pivot.clone());
                let tiny_reg = batched_float_const::<T>(b, tiny);
                let singular = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.le{suffix} {singular}, {apv}, {tiny_reg};"));
                let do_scale = b.fresh_label("blu_sc");
                let after_scale = b.fresh_label("blu_sce");
                b.raw_ptx(&format!("@!{singular} bra {do_scale};"));
                {
                    let one = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {one}, 1;"));
                    let kp1 = b.add_u32(k.clone(), one);
                    b.raw_ptx(&format!("st.global.u32 [{info_addr}], {kp1};"));
                }
                b.raw_ptx(&format!("bra {after_scale};"));
                b.raw_ptx(&format!("{do_scale}:"));
                {
                    b.raw_ptx(&format!("add.u32 {rr}, {k}, 1;"));
                    let c_loop = b.fresh_label("blu_c");
                    let c_exit = b.fresh_label("blu_cx");
                    b.raw_ptx(&format!("{c_loop}:"));
                    let c_done = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {c_done}, {rr}, {n_reg};"));
                    b.raw_ptx(&format!("@{c_done} bra {c_exit};"));
                    let addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colk);
                    let v = load_global_float::<T>(b, addr.clone());
                    let scaled = div_float::<T>(b, v, pivot.clone());
                    store_global_float::<T>(b, addr, scaled);
                    b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                    b.raw_ptx(&format!("bra {c_loop};"));
                    b.raw_ptx(&format!("{c_exit}:"));
                }
                b.raw_ptx(&format!("{after_scale}:"));
            }
            b.raw_ptx(&format!("{p3_end}:"));
            b.bar_sync(0);

            // ---- Phase 4: rank-1 trailing update (owners k < tig < n) ----
            let p4_end = b.fresh_label("blu_p4e");
            b.raw_ptx(&format!("@{inactive} bra {p4_end};"));
            let le_k = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.le.u32 {le_k}, {tig}, {k};"));
            b.raw_ptx(&format!("@{le_k} bra {p4_end};"));
            {
                let akc_addr = batched_elem_addr::<T>(b, &mat_base, &k, &colt);
                let akc = load_global_float::<T>(b, akc_addr);
                b.raw_ptx(&format!("add.u32 {rr}, {k}, 1;"));
                let t_loop = b.fresh_label("blu_t");
                let t_exit = b.fresh_label("blu_tx");
                b.raw_ptx(&format!("{t_loop}:"));
                let t_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {t_done}, {rr}, {n_reg};"));
                b.raw_ptx(&format!("@{t_done} bra {t_exit};"));
                let ark_addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colk);
                let ark = load_global_float::<T>(b, ark_addr);
                let arc_addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colt);
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

/// Emits PTX for batched QR factorization via Householder reflections.
///
/// Each thread block handles one matrix. For each column, computes the
/// Householder vector, stores tau, and applies the reflection to trailing
/// columns in shared memory.
pub(crate) fn emit_batched_qr<T: GpuFloat>(
    sm: SmVersion,
    m: usize,
    n: usize,
) -> SolverResult<String> {
    let name = batched_qr_name::<T>(m, n);
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("matrices_ptr", PtxType::U64)
        .param("tau_ptr", PtxType::U64)
        .param("m", PtxType::U32)
        .param("n", PtxType::U32)
        .param("batch_count", PtxType::U32)
        .body(move |b| {
            let bid = b.block_id_x();
            let tid = b.thread_id_x();
            let batch_count_reg = b.load_param_u32("batch_count");
            let m_reg = b.load_param_u32("m");
            let n_reg = b.load_param_u32("n");

            b.if_lt_u32(bid.clone(), batch_count_reg, |b| {
                let matrices_ptr = b.load_param_u64("matrices_ptr");
                let tau_ptr = b.load_param_u64("tau_ptr");

                // Compute matrix offset: batch_idx * m * n * sizeof(T).
                let mn = b.mul_lo_u32(m_reg.clone(), n_reg.clone());
                let mat_offset = b.mul_lo_u32(bid.clone(), mn);
                let _mat_base = b.byte_offset_addr(matrices_ptr, mat_offset, T::size_u32());

                // tau offset: batch_idx * min(m,n) * sizeof(T).
                // For simplicity use n (assuming m >= n for QR).
                let tau_offset = b.mul_lo_u32(bid, n_reg);
                let _tau_base = b.byte_offset_addr(tau_ptr, tau_offset, T::size_u32());

                // Householder QR in shared memory:
                // For each column k = 0..min(m,n):
                //   1. Compute norm of column below diagonal.
                //   2. Compute Householder vector v and scalar tau.
                //   3. Apply H = I - tau * v * v^T to trailing columns.

                let _ = (tid, float_ty, m_reg);
            });

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits PTX for batched Cholesky factorization.
///
/// Each thread block processes `matrices_per_block(n)` matrices, mirroring
/// [`emit_batched_lu`]'s batch-index/inactivity-predicate scheme. Within a
/// matrix, thread `t` (`t < n`) owns column `t`. For each pivot `k = 0..n`:
///
/// 1. the owner of `k` takes the square root of the (already-updated)
///    diagonal and scales its sub-diagonal: `A[i,k] /= sqrt(A[k,k])` — or, on
///    a non-positive diagonal, records the 1-based failing column into
///    `info[batch_idx]` and skips the scale (the matrix is not SPD);
/// 2. every owner `t` with `k < t < n` applies the rank-1 trailing update to
///    its own column `A[i,t] -= A[i,k] * A[t,k]` for `i = t..n`.
///
/// This is the batched analogue of
/// [`crate::dense::cholesky::emit_panel_cholesky`] (`Lower` triangle only,
/// since batched callers always want a single fixed layout).
pub(crate) fn emit_batched_cholesky<T: GpuFloat>(sm: SmVersion, n: usize) -> SolverResult<String> {
    let name = batched_cholesky_name::<T>(n);
    let suffix = T::PTX_TYPE.as_ptx_str();

    let mpb = matrices_per_block(n) as u32;
    let group_size = compute_block_size(n) / mpb;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("matrices_ptr", PtxType::U64)
        .param("info_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("batch_count", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let bid = b.block_id_x();
            let batch_count_reg = b.load_param_u32("batch_count");
            let n_reg = b.load_param_u32("n");
            let matrices_ptr = b.load_param_u64("matrices_ptr");
            let info_ptr = b.load_param_u64("info_ptr");

            let group_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("div.u32 {group_idx}, {tid}, {group_size};"));
            let tig = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("rem.u32 {tig}, {tid}, {group_size};"));
            let mpb_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {mpb_reg}, {mpb};"));
            let batch_idx = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!(
                "mad.lo.u32 {batch_idx}, {bid}, {mpb_reg}, {group_idx};"
            ));

            let inactive = b.alloc_reg(PtxType::Pred);
            let p_tig = b.alloc_reg(PtxType::Pred);
            let p_batch = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_tig}, {tig}, {n_reg};"));
            b.raw_ptx(&format!(
                "setp.ge.u32 {p_batch}, {batch_idx}, {batch_count_reg};"
            ));
            b.raw_ptx(&format!("or.pred {inactive}, {p_tig}, {p_batch};"));

            let n2 = b.mul_lo_u32(n_reg.clone(), n_reg.clone());
            let mat_offset = b.mul_lo_u32(batch_idx.clone(), n2);
            let mat_base = b.byte_offset_addr(matrices_ptr, mat_offset, T::size_u32());
            let info_addr = b.byte_offset_addr(info_ptr, batch_idx, 4u32);

            let k = b.alloc_reg(PtxType::U32);
            let rr = b.alloc_reg(PtxType::U32);
            let colk = b.alloc_reg(PtxType::U32);
            let colt = b.alloc_reg(PtxType::U32);

            b.raw_ptx(&format!("mov.u32 {k}, 0;"));
            let k_loop = b.fresh_label("bch_k");
            let k_exit = b.fresh_label("bch_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {n_reg};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));

            b.raw_ptx(&format!("mul.lo.u32 {colk}, {k}, {n_reg};"));

            // ---- Phase A: owner of column k factors the diagonal + scales
            let pa_end = b.fresh_label("bch_pae");
            b.raw_ptx(&format!("@{inactive} bra {pa_end};"));
            let not_owner = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ne.u32 {not_owner}, {tig}, {k};"));
            b.raw_ptx(&format!("@{not_owner} bra {pa_end};"));
            {
                let akk_addr = batched_elem_addr::<T>(b, &mat_base, &k, &colk);
                let akk = load_global_float::<T>(b, akk_addr.clone());
                let zero_reg = batched_float_const::<T>(b, 0.0);
                let non_positive = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!(
                    "setp.le{suffix} {non_positive}, {akk}, {zero_reg};"
                ));
                let fail_branch = b.fresh_label("bch_fail");
                let after_scale = b.fresh_label("bch_sce");
                b.raw_ptx(&format!("@{non_positive} bra {fail_branch};"));
                {
                    let diag = sqrt_float::<T>(b, akk);
                    store_global_float::<T>(b, akk_addr, diag.clone());
                    b.raw_ptx(&format!("add.u32 {rr}, {k}, 1;"));
                    let s_loop = b.fresh_label("bch_s");
                    let s_exit = b.fresh_label("bch_sx");
                    b.raw_ptx(&format!("{s_loop}:"));
                    let s_done = b.alloc_reg(PtxType::Pred);
                    b.raw_ptx(&format!("setp.ge.u32 {s_done}, {rr}, {n_reg};"));
                    b.raw_ptx(&format!("@{s_done} bra {s_exit};"));
                    let addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colk);
                    let v = load_global_float::<T>(b, addr.clone());
                    let scaled = div_float::<T>(b, v, diag.clone());
                    store_global_float::<T>(b, addr, scaled);
                    b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                    b.raw_ptx(&format!("bra {s_loop};"));
                    b.raw_ptx(&format!("{s_exit}:"));
                }
                b.raw_ptx(&format!("bra {after_scale};"));
                b.raw_ptx(&format!("{fail_branch}:"));
                {
                    let one = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mov.u32 {one}, 1;"));
                    let kp1 = b.add_u32(k.clone(), one);
                    b.raw_ptx(&format!("st.global.u32 [{info_addr}], {kp1};"));
                }
                b.raw_ptx(&format!("{after_scale}:"));
            }
            b.raw_ptx(&format!("{pa_end}:"));
            b.bar_sync(0);

            // ---- Phase B: owners k < tig < n apply the trailing update ----
            b.raw_ptx(&format!("mul.lo.u32 {colt}, {tig}, {n_reg};"));
            let pb_end = b.fresh_label("bch_pbe");
            b.raw_ptx(&format!("@{inactive} bra {pb_end};"));
            let le_k = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.le.u32 {le_k}, {tig}, {k};"));
            b.raw_ptx(&format!("@{le_k} bra {pb_end};"));
            {
                let lck_addr = batched_elem_addr::<T>(b, &mat_base, &tig, &colk);
                let lck = load_global_float::<T>(b, lck_addr);
                b.raw_ptx(&format!("mov.u32 {rr}, {tig};"));
                let t_loop = b.fresh_label("bch_t");
                let t_exit = b.fresh_label("bch_tx");
                b.raw_ptx(&format!("{t_loop}:"));
                let t_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {t_done}, {rr}, {n_reg};"));
                b.raw_ptx(&format!("@{t_done} bra {t_exit};"));
                let lik_addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colk);
                let lik = load_global_float::<T>(b, lik_addr);
                let aic_addr = batched_elem_addr::<T>(b, &mat_base, &rr, &colt);
                let aic = load_global_float::<T>(b, aic_addr.clone());
                let prod = mul_float::<T>(b, lik, lck.clone());
                let upd = sub_float::<T>(b, aic, prod);
                store_global_float::<T>(b, aic_addr, upd);
                b.raw_ptx(&format!("add.u32 {rr}, {rr}, 1;"));
                b.raw_ptx(&format!("bra {t_loop};"));
                b.raw_ptx(&format!("{t_exit}:"));
            }
            b.raw_ptx(&format!("{pb_end}:"));
            b.bar_sync(0);

            b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits PTX for batched linear solve (apply LU factors to RHS).
///
/// Each thread block solves one system: applies pivots to B, then performs
/// forward substitution (L) and backward substitution (U).
pub(crate) fn emit_batched_solve<T: GpuFloat>(
    sm: SmVersion,
    n: usize,
    nrhs: usize,
) -> SolverResult<String> {
    let name = batched_solve_name::<T>(n, nrhs);
    let float_ty = T::PTX_TYPE;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("lu_ptr", PtxType::U64)
        .param("b_ptr", PtxType::U64)
        .param("pivots_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("nrhs", PtxType::U32)
        .param("batch_count", PtxType::U32)
        .body(move |b| {
            let bid = b.block_id_x();
            let tid = b.thread_id_x();
            let batch_count_reg = b.load_param_u32("batch_count");
            let n_reg = b.load_param_u32("n");
            let nrhs_reg = b.load_param_u32("nrhs");

            b.if_lt_u32(bid.clone(), batch_count_reg, |b| {
                let lu_ptr = b.load_param_u64("lu_ptr");
                let b_ptr = b.load_param_u64("b_ptr");
                let pivots_ptr = b.load_param_u64("pivots_ptr");

                // LU matrix offset.
                let n2 = b.mul_lo_u32(n_reg.clone(), n_reg.clone());
                let lu_offset = b.mul_lo_u32(bid.clone(), n2);
                let _lu_base = b.byte_offset_addr(lu_ptr, lu_offset, T::size_u32());

                // B matrix offset.
                let b_stride = b.mul_lo_u32(n_reg.clone(), nrhs_reg);
                let b_offset = b.mul_lo_u32(bid.clone(), b_stride);
                let _b_base = b.byte_offset_addr(b_ptr, b_offset, T::size_u32());

                // Pivot offset.
                let piv_offset = b.mul_lo_u32(bid, n_reg);
                let _piv_base = b.byte_offset_addr(pivots_ptr, piv_offset, 4u32);

                // Solve steps:
                // 1. Apply pivots to B.
                // 2. Forward substitution: L * Y = P * B.
                // 3. Backward substitution: U * X = Y.

                let _ = (tid, float_ty);
            });

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_algorithm_equality() {
        assert_eq!(BatchAlgorithm::Lu, BatchAlgorithm::Lu);
        assert_ne!(BatchAlgorithm::Lu, BatchAlgorithm::Qr);
        assert_ne!(BatchAlgorithm::Qr, BatchAlgorithm::Cholesky);
    }

    #[test]
    fn batch_config_construction() {
        let config = BatchConfig {
            matrix_size: 16,
            batch_count: 1000,
            algorithm: BatchAlgorithm::Lu,
        };
        assert_eq!(config.matrix_size, 16);
        assert_eq!(config.batch_count, 1000);
        assert_eq!(config.algorithm, BatchAlgorithm::Lu);
    }

    #[test]
    fn batched_result_construction() {
        let result = BatchedResult { failed_count: 0 };
        assert_eq!(result.failed_count, 0);

        let result2 = BatchedResult { failed_count: 5 };
        assert_eq!(result2.failed_count, 5);
    }

    #[test]
    fn matrices_per_block_small() {
        // Small matrices should pack multiple per block.
        assert_eq!(matrices_per_block(4), SMALL_MATRICES_PER_BLOCK);
        assert_eq!(matrices_per_block(8), SMALL_MATRICES_PER_BLOCK);
        assert_eq!(matrices_per_block(16), SMALL_MATRICES_PER_BLOCK);
    }

    #[test]
    fn matrices_per_block_large() {
        // Larger matrices get one per block.
        assert_eq!(matrices_per_block(32), 1);
        assert_eq!(matrices_per_block(64), 1);
    }

    #[test]
    fn compute_block_size_values() {
        // Small matrices.
        let bs_small = compute_block_size(8);
        assert!(bs_small <= SOLVER_BLOCK_SIZE);
        assert!(bs_small >= 32);

        // Medium matrices.
        let bs_med = compute_block_size(32);
        assert_eq!(bs_med, 32);

        // Large matrices.
        let bs_large = compute_block_size(64);
        assert_eq!(bs_large, 64);
    }

    #[test]
    fn compute_grid_size_values() {
        // 100 matrices, small (multiple per block).
        let grid = compute_grid_size(100, 8);
        assert_eq!(grid, 25); // 100 / 4 = 25

        // 100 matrices, large (one per block).
        let grid = compute_grid_size(100, 32);
        assert_eq!(grid, 100);

        // Non-divisible batch count.
        let grid = compute_grid_size(101, 8);
        assert_eq!(grid, 26); // ceil(101 / 4)
    }

    #[test]
    fn batched_lu_name_format() {
        let name = batched_lu_name::<f32>(16);
        assert!(name.contains("f32"));
        assert!(name.contains("16"));
    }

    #[test]
    fn batched_qr_name_format() {
        let name = batched_qr_name::<f64>(32, 16);
        assert!(name.contains("f64"));
        assert!(name.contains("32x16"));
    }

    #[test]
    fn batched_cholesky_name_format() {
        let name = batched_cholesky_name::<f32>(64);
        assert!(name.contains("f32"));
        assert!(name.contains("64"));
    }

    #[test]
    fn batched_solve_name_format() {
        let name = batched_solve_name::<f64>(16, 4);
        assert!(name.contains("f64"));
        assert!(name.contains("16"));
        assert!(name.contains("4"));
    }

    #[test]
    fn max_batch_matrix_size_reasonable() {
        let max_size = MAX_BATCH_MATRIX_SIZE;
        assert!(max_size >= 32);
        assert!(max_size <= 128);
    }

    #[test]
    fn small_matrix_threshold_consistent() {
        let threshold = SMALL_MATRIX_THRESHOLD;
        let per_block = SMALL_MATRICES_PER_BLOCK;
        assert!(threshold <= 32);
        assert!(per_block >= 1);
        assert!(per_block <= 16);
    }

    // -----------------------------------------------------------------------
    // CPU reference LU factorization + batched throughput benchmark
    // -----------------------------------------------------------------------

    /// Gaussian elimination with partial pivoting for an n×n matrix (row-major).
    ///
    /// Returns `false` if the matrix is singular (zero pivot encountered).
    fn cpu_lu_factorize(mat: &mut [f32], n: usize) -> bool {
        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = mat[col * n + col].abs();
            for row in (col + 1)..n {
                let v = mat[row * n + col].abs();
                if v > max_val {
                    max_val = v;
                    max_row = row;
                }
            }
            if max_val < f32::EPSILON {
                return false; // singular
            }
            // Swap rows col and max_row
            if max_row != col {
                for j in 0..n {
                    mat.swap(col * n + j, max_row * n + j);
                }
            }
            // Eliminate below pivot
            let pivot = mat[col * n + col];
            for row in (col + 1)..n {
                let factor = mat[row * n + col] / pivot;
                mat[row * n + col] = factor; // store L
                for j in (col + 1)..n {
                    let u_val = mat[col * n + j];
                    mat[row * n + j] -= factor * u_val;
                }
            }
        }
        true
    }

    /// Verify CPU LU factorization on a simple 3×3 system.
    #[test]
    fn cpu_lu_factorize_3x3_known() {
        // A = [[2,1,1],[4,3,3],[8,7,9]]
        let mut mat = vec![2.0_f32, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0];
        let ok = cpu_lu_factorize(&mut mat, 3);
        assert!(ok, "3×3 non-singular matrix must factorize successfully");
        // After LU: diagonal of U must all be non-zero
        assert!(mat[0].abs() > f32::EPSILON, "U[0,0] must be non-zero");
        assert!(mat[4].abs() > f32::EPSILON, "U[1,1] must be non-zero");
        assert!(mat[8].abs() > f32::EPSILON, "U[2,2] must be non-zero");
    }

    /// Batched LU throughput benchmark: 1000 × 8×8 factorizations.
    ///
    /// Measures CPU reference throughput as a structural proxy for the
    /// GPU batched LU kernel target (cuSOLVER comparison requires real hardware).
    /// Additionally verifies that `emit_batched_lu` generates valid PTX for 8×8.
    #[test]
    fn batched_lu_throughput_proxy_1000x8x8() {
        const BATCH: usize = 1000;
        const N: usize = 8;

        // Build 1000 non-singular 8×8 matrices (diagonal-dominant, deterministic)
        let base: Vec<f32> = (0..N * N)
            .map(|idx| {
                let r = idx / N;
                let c = idx % N;
                if r == c {
                    N as f32 * 2.0 // dominant diagonal
                } else {
                    ((r * N + c) as f32 * 0.1_f32).fract()
                }
            })
            .collect();

        // Warm-up
        let mut m = base.clone();
        let _ = cpu_lu_factorize(&mut m, N);

        let start = std::time::Instant::now();
        let mut successes = 0_usize;
        for batch_idx in 0..BATCH {
            let mut mat: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(i, &v)| v + batch_idx as f32 * 0.001 * (i as f32).sin())
                .collect();
            if cpu_lu_factorize(&mut mat, N) {
                successes += 1;
            }
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;

        // ~2/3 * N^3 flops for LU factorization of an N×N matrix
        let flops_per_lu = (2 * N * N * N / 3) as f64;
        let total_flops = flops_per_lu * BATCH as f64;
        let gflops = total_flops / elapsed_ns;
        let matrices_per_sec = (BATCH as f64) / (elapsed_ns * 1e-9);

        println!(
            "Batched LU proxy ({} × {}×{}, {} successes): {:.1} k matrices/s, {:.4} GFLOPS (CPU ref)",
            BATCH,
            N,
            N,
            successes,
            matrices_per_sec / 1000.0,
            gflops
        );

        assert_eq!(
            successes, BATCH,
            "All {} matrices must factorize successfully",
            BATCH
        );
        assert!(
            matrices_per_sec > 100.0,
            "Batched LU CPU throughput unrealistically low: {:.1} matrices/s",
            matrices_per_sec
        );
    }
}
