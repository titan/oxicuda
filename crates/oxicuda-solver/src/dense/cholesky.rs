//! Cholesky Decomposition for symmetric positive definite matrices.
//!
//! Computes `A = L * L^T` (lower) or `A = U^T * U` (upper) where A is
//! symmetric positive definite.
//!
//! Uses a blocked algorithm:
//! 1. Diagonal block: compute Cholesky of the small diagonal block
//! 2. Column panel: TRSM for the off-diagonal block
//! 3. Trailing update: SYRK for the symmetric rank-k update

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
    SOLVER_BLOCK_SIZE, div_float, fma_float, load_global_float, mul_float, sqrt_float,
    store_global_float, sub_float,
};

/// Block size for the blocked Cholesky algorithm.
const CHOL_BLOCK_SIZE: u32 = 64;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Performs Cholesky decomposition in-place.
///
/// On exit, the specified triangle of `a` is overwritten with the factor:
/// - `FillMode::Lower`: `A = L * L^T`, lower triangle contains L.
/// - `FillMode::Upper`: `A = U^T * U`, upper triangle contains U.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `uplo` — which triangle to read/write (Lower or Upper).
/// * `a` — symmetric positive definite matrix (n x n, column-major, lda stride).
/// * `n` — matrix dimension.
/// * `lda` — leading dimension (>= n).
///
/// # Errors
///
/// Returns [`SolverError::NotPositiveDefinite`] if the matrix is not SPD.
/// Returns [`SolverError::DimensionMismatch`] for invalid dimensions.
pub fn cholesky<T: GpuFloat>(
    handle: &mut SolverHandle,
    uplo: FillMode,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
) -> SolverResult<()> {
    if n == 0 {
        return Ok(());
    }
    if lda < n {
        return Err(SolverError::DimensionMismatch(format!(
            "cholesky: lda ({lda}) must be >= n ({n})"
        )));
    }
    let required = n as usize * lda as usize;
    if a.len() < required {
        return Err(SolverError::DimensionMismatch(format!(
            "cholesky: buffer too small ({} < {required})",
            a.len()
        )));
    }

    if uplo == FillMode::Full {
        return Err(SolverError::DimensionMismatch(
            "cholesky: uplo must be Upper or Lower, not Full".into(),
        ));
    }

    blocked_cholesky::<T>(handle, uplo, a, n, lda)
}

/// Solves `A * X = B` given a Cholesky-factored matrix.
///
/// The factor must have been computed by [`cholesky`].
///
/// For `uplo == Lower`: solves `L * L^T * X = B` via forward then backward TRSM.
/// For `uplo == Upper`: solves `U^T * U * X = B` via forward then backward TRSM.
///
/// # Arguments
///
/// * `handle` — solver handle.
/// * `uplo` — which triangle contains the factor.
/// * `a` — Cholesky factor (output of `cholesky`).
/// * `b` — right-hand side (n x nrhs), overwritten with solution.
/// * `n` — matrix dimension.
/// * `nrhs` — number of right-hand side columns.
///
/// # Errors
///
/// Returns [`SolverError`] if dimensions are invalid or BLAS operations fail.
pub fn cholesky_solve<T: GpuFloat>(
    handle: &SolverHandle,
    uplo: FillMode,
    a: &DeviceBuffer<T>,
    b: &mut DeviceBuffer<T>,
    n: u32,
    nrhs: u32,
) -> SolverResult<()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    if a.len() < (n as usize * n as usize) {
        return Err(SolverError::DimensionMismatch(
            "cholesky_solve: factor buffer too small".into(),
        ));
    }
    if b.len() < (n as usize * nrhs as usize) {
        return Err(SolverError::DimensionMismatch(
            "cholesky_solve: B buffer too small".into(),
        ));
    }

    let a_desc = MatrixDesc::<T>::from_raw(a.as_device_ptr(), n, n, n, Layout::ColMajor);
    let mut b_desc = MatrixDescMut::<T>::from_raw(b.as_device_ptr(), n, nrhs, n, Layout::ColMajor);

    match uplo {
        FillMode::Lower => {
            // Solve L * Y = B (forward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Lower,
                Transpose::NoTrans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
            // Solve L^T * X = Y (backward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Lower,
                Transpose::Trans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
        }
        FillMode::Upper => {
            // Solve U^T * Y = B (forward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Upper,
                Transpose::Trans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
            // Solve U * X = Y (backward substitution).
            oxicuda_blas::level3::trsm(
                handle.blas(),
                Side::Left,
                FillMode::Upper,
                Transpose::NoTrans,
                DiagType::NonUnit,
                T::gpu_one(),
                &a_desc,
                &mut b_desc,
            )?;
        }
        FillMode::Full => {
            return Err(SolverError::DimensionMismatch(
                "cholesky_solve: uplo must be Upper or Lower, not Full".into(),
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Blocked Cholesky implementation
// ---------------------------------------------------------------------------

/// Blocked Cholesky factorization (lower triangular).
///
/// Processes the matrix in blocks of size `CHOL_BLOCK_SIZE`:
/// 1. Factor the diagonal block using a small Cholesky kernel.
/// 2. Solve for the off-diagonal panel via TRSM.
/// 3. Update the trailing submatrix via SYRK.
fn blocked_cholesky<T: GpuFloat>(
    handle: &mut SolverHandle,
    uplo: FillMode,
    a: &mut DeviceBuffer<T>,
    n: u32,
    lda: u32,
) -> SolverResult<()> {
    let nb = CHOL_BLOCK_SIZE.min(n);
    let num_blocks = n.div_ceil(nb);

    for block_idx in 0..num_blocks {
        let j = block_idx * nb;
        let jb = nb.min(n - j);

        // Step 1: Factor the diagonal block A[j:j+jb, j:j+jb].
        panel_cholesky::<T>(handle, a, lda, j, jb, uplo)?;

        let remaining = n.saturating_sub(j + jb);
        if remaining > 0 {
            let is_lower = uplo != FillMode::Upper;
            // Pointers into the larger column-major buffer (all leading dim = lda).
            let diag_ptr = a.as_device_ptr() + (j as u64 + j as u64 * lda as u64) * T::SIZE as u64;
            // Off-diagonal panel: A21 (lower, at [j+jb, j]) or A12 (upper, at [j, j+jb]).
            let panel_ptr = if is_lower {
                a.as_device_ptr() + ((j + jb) as u64 + j as u64 * lda as u64) * T::SIZE as u64
            } else {
                a.as_device_ptr() + (j as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64
            };
            let a22_ptr = a.as_device_ptr()
                + ((j + jb) as u64 + (j + jb) as u64 * lda as u64) * T::SIZE as u64;

            // Step 2: strided panel TRSM (honours leading dimensions).
            //  Lower: solve L21 · L11ᵀ = A21  (A21 is `remaining x jb`).
            //  Upper: solve U11ᵀ · U12 = A12  (A12 is `jb x remaining`).
            // `free` is the count of independent panel lines (rows for lower,
            // columns for upper), one device thread each.
            launch_chol_panel_trsm::<T>(
                handle, is_lower, diag_ptr, panel_ptr, jb, remaining, lda, lda,
            )?;

            // Step 3: strided symmetric rank-`jb` update (honours leading dims).
            //  Lower: A22 -= L21 · L21ᵀ  (lower triangle).
            //  Upper: A22 -= U12ᵀ · U12  (upper triangle).
            launch_chol_syrk::<T>(
                handle, is_lower, panel_ptr, a22_ptr, remaining, jb, lda, lda,
            )?;
        }
    }

    Ok(())
}

/// Panel Cholesky: factorizes a small diagonal block on the GPU.
///
/// Launches a single-CTA kernel that performs an in-place, right-looking
/// Cholesky factorization of the `jb x jb` diagonal block located at
/// `A[j:j+jb, j:j+jb]` (column-major, leading dimension `lda`). The block fits
/// within one thread block: column/row `t` of the block is owned by thread `t`
/// (`jb <= CHOL_BLOCK_SIZE <= SOLVER_BLOCK_SIZE`), so no inter-thread write
/// races occur and a single barrier per pivot column suffices for correctness.
fn panel_cholesky<T: GpuFloat>(
    handle: &SolverHandle,
    a: &mut DeviceBuffer<T>,
    lda: u32,
    j: u32,
    jb: u32,
    uplo: FillMode,
) -> SolverResult<()> {
    let sm = handle.sm_version();
    let ptx = emit_panel_cholesky::<T>(sm, jb, uplo)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &panel_cholesky_name::<T>(uplo, jb))?;

    // One thread block; one thread per block column/row. No dynamic shared mem.
    let params = LaunchParams::new(1u32, SOLVER_BLOCK_SIZE);

    let diag_offset = (j as u64 + j as u64 * lda as u64) * T::SIZE as u64;
    let diag_ptr = a.as_device_ptr() + diag_offset;

    let args = (diag_ptr, jb, lda);
    kernel.launch(&params, handle.stream(), &args)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Strided off-diagonal panel update (BLAS-free; honours leading dimensions)
// ---------------------------------------------------------------------------

/// Launches the strided off-diagonal panel TRSM that computes the off-diagonal
/// Cholesky factor in place.
///
/// * `is_lower == true`: the panel is `A21` (`free x jb`, leading dimension
///   `lda`) and the kernel solves `L21 · L11ᵀ = A21` so that the panel is
///   overwritten with `L21`. `free` is the number of trailing rows; one device
///   thread owns each row.
/// * `is_lower == false`: the panel is `A12` (`jb x free`, leading dimension
///   `lda`) and the kernel solves `U11ᵀ · U12 = A12` so that the panel is
///   overwritten with `U12`. `free` is the number of trailing columns; one
///   device thread owns each column.
///
/// `diag_ptr` addresses the already-factored `jb x jb` diagonal block (`L11` or
/// `U11`) with leading dimension `ldt`; `panel_ptr` addresses the panel with
/// leading dimension `lda`. All matrices are column-major sub-blocks of a larger
/// buffer, so the leading dimensions are honoured explicitly.
#[allow(clippy::too_many_arguments)]
fn launch_chol_panel_trsm<T: GpuFloat>(
    handle: &SolverHandle,
    is_lower: bool,
    diag_ptr: u64,
    panel_ptr: u64,
    jb: u32,
    free: u32,
    ldt: u32,
    lda: u32,
) -> SolverResult<()> {
    if jb == 0 || free == 0 {
        return Ok(());
    }
    let sm = handle.sm_version();
    let ptx = emit_chol_panel_trsm::<T>(sm, is_lower)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &chol_trsm_name::<T>(is_lower))?;

    let num_blocks = free.div_ceil(SOLVER_BLOCK_SIZE).max(1);
    let params = LaunchParams::new(num_blocks, SOLVER_BLOCK_SIZE);
    let args = (diag_ptr, panel_ptr, jb, free, ldt, lda);
    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

/// Launches the strided symmetric rank-`jb` trailing update.
///
/// * `is_lower == true`: computes `A22 -= L21 · L21ᵀ`, updating the *lower*
///   triangle (`row >= col`). The operand `panel_ptr` is `L21` (`rem x jb`).
/// * `is_lower == false`: computes `A22 -= U12ᵀ · U12`, updating the *upper*
///   triangle (`row <= col`). The operand `panel_ptr` is `U12` (`jb x rem`).
///
/// `c_ptr` addresses the `rem x rem` trailing block `A22` (leading dimension
/// `ldc`); the operand has leading dimension `ldp`. Both are column-major
/// sub-blocks, so leading dimensions are honoured explicitly.
#[allow(clippy::too_many_arguments)]
fn launch_chol_syrk<T: GpuFloat>(
    handle: &SolverHandle,
    is_lower: bool,
    panel_ptr: u64,
    c_ptr: u64,
    rem: u32,
    jb: u32,
    ldp: u32,
    ldc: u32,
) -> SolverResult<()> {
    if rem == 0 || jb == 0 {
        return Ok(());
    }
    let sm = handle.sm_version();
    let ptx = emit_chol_syrk::<T>(sm, is_lower)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, &chol_syrk_name::<T>(is_lower))?;

    const TILE: u32 = 16;
    let grid = rem.div_ceil(TILE).max(1);
    let params = LaunchParams::new((grid, grid), (TILE, TILE));
    let args = (panel_ptr, c_ptr, rem, jb, ldp, ldc);
    kernel.launch(&params, handle.stream(), &args)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernel generation
// ---------------------------------------------------------------------------

fn chol_trsm_name<T: GpuFloat>(is_lower: bool) -> String {
    let tri = if is_lower { "lower" } else { "upper" };
    format!("solver_chol_trsm_{tri}_{}", T::NAME)
}

fn chol_syrk_name<T: GpuFloat>(is_lower: bool) -> String {
    let tri = if is_lower { "lower" } else { "upper" };
    format!("solver_chol_syrk_{tri}_{}", T::NAME)
}

/// Computes the column-major index `add + mul * stride` in a fresh `u32`
/// register: `mul.lo.u32 t, mul, stride; add.u32 idx, add, t`.
fn col_index(
    b: &mut BodyBuilder<'_>,
    add: &Register,
    mul: &Register,
    stride: &Register,
) -> Register {
    let off = b.mul_lo_u32(mul.clone(), stride.clone());
    b.add_u32(add.clone(), off)
}

/// Emits PTX for the strided off-diagonal Cholesky panel TRSM.
///
/// One thread owns one panel line (`g = gid < free`) and solves the `jb`
/// unknowns of that line sequentially. With the triangular diagonal block `D`
/// (`L11` or `U11`, leading dimension `ldt`) and the panel `P` (leading
/// dimension `lda`):
///
/// * Lower (`L21 · L11ᵀ = A21`, thread owns row `g`): for `p = 0..jb`,
///   `P[g,p] = (A21[g,p] - Σ_{k<p} L11[p,k]·P[g,k]) / L11[p,p]`.
/// * Upper (`U11ᵀ · U12 = A12`, thread owns column `g`): for `p = 0..jb`,
///   `P[p,g] = (A12[p,g] - Σ_{k<p} U11[k,p]·P[k,g]) / U11[p,p]`.
///
/// Column ownership makes the threads independent (no barriers); the sequential
/// `p` loop honours the dependence on already-solved unknowns. Leading
/// dimensions are honoured explicitly so the kernel is correct for sub-blocks
/// embedded in a larger buffer.
pub(crate) fn emit_chol_panel_trsm<T: GpuFloat>(
    sm: SmVersion,
    is_lower: bool,
) -> SolverResult<String> {
    let name = chol_trsm_name::<T>(is_lower);
    let suffix = T::PTX_TYPE.as_ptx_str();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("diag_ptr", PtxType::U64)
        .param("panel_ptr", PtxType::U64)
        .param("jb", PtxType::U32)
        .param("free", PtxType::U32)
        .param("ldt", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let free = b.load_param_u32("free");
            let inactive = b.alloc_reg(PtxType::Pred);
            let done = b.fresh_label("ctrsm_done");
            b.raw_ptx(&format!("setp.ge.u32 {inactive}, {gid}, {free};"));
            b.raw_ptx(&format!("@{inactive} bra {done};"));

            let diag = b.load_param_u64("diag_ptr");
            let panel = b.load_param_u64("panel_ptr");
            let jb = b.load_param_u32("jb");
            let ldt = b.load_param_u32("ldt");
            let lda = b.load_param_u32("lda");

            // p = 0..jb (unknown index along the pivot direction).
            let p = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {p}, 0;"));
            let p_loop = b.fresh_label("ctrsm_p");
            let p_exit = b.fresh_label("ctrsm_px");
            b.raw_ptx(&format!("{p_loop}:"));
            let p_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {p_done}, {p}, {jb};"));
            b.raw_ptx(&format!("@{p_done} bra {p_exit};"));

            // Address of this line's unknown P at pivot position p.
            //  Lower: P[g,p] -> idx = g + p*lda.   Upper: P[p,g] -> idx = p + g*lda.
            let solve_idx = if is_lower {
                col_index(b, &gid, &p, &lda)
            } else {
                col_index(b, &p, &gid, &lda)
            };
            let solve_addr = b.byte_offset_addr(panel.clone(), solve_idx, T::size_u32());

            // acc = P[solve]  (the original A entry, still in place).
            let acc = b.alloc_reg(T::PTX_TYPE);
            let a_val = load_global_float::<T>(b, solve_addr.clone());
            b.raw_ptx(&format!("mov{suffix} {acc}, {a_val};"));

            // k = 0..p: acc -= D[tri_off] * P[x].
            let k = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k}, 0;"));
            let k_loop = b.fresh_label("ctrsm_k");
            let k_exit = b.fresh_label("ctrsm_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {p};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));

            //  Lower: D=L11[p,k] -> p + k*ldt;  X=P[g,k] -> g + k*lda.
            //  Upper: D=U11[k,p] -> k + p*ldt;  X=P[k,g] -> k + g*lda.
            let (tri_idx, x_idx) = if is_lower {
                (col_index(b, &p, &k, &ldt), col_index(b, &gid, &k, &lda))
            } else {
                (col_index(b, &k, &p, &ldt), col_index(b, &k, &gid, &lda))
            };
            let tri_addr = b.byte_offset_addr(diag.clone(), tri_idx, T::size_u32());
            let x_addr = b.byte_offset_addr(panel.clone(), x_idx, T::size_u32());
            let tri_val = load_global_float::<T>(b, tri_addr);
            let x_val = load_global_float::<T>(b, x_addr);
            let prod = mul_float::<T>(b, tri_val, x_val);
            let new_acc = sub_float::<T>(b, acc.clone(), prod);
            b.raw_ptx(&format!("mov{suffix} {acc}, {new_acc};"));
            b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            // Divide by the diagonal D[p,p] -> p + p*ldt (non-unit).
            let diag_idx = col_index(b, &p, &p, &ldt);
            let diag_addr = b.byte_offset_addr(diag.clone(), diag_idx, T::size_u32());
            let dpp = load_global_float::<T>(b, diag_addr);
            let solved = div_float::<T>(b, acc, dpp);
            store_global_float::<T>(b, solve_addr, solved);

            b.raw_ptx(&format!("add.u32 {p}, {p}, 1;"));
            b.raw_ptx(&format!("bra {p_loop};"));
            b.raw_ptx(&format!("{p_exit}:"));
            b.raw_ptx(&format!("{done}:"));
            b.ret();
        })
        .build()?;

    Ok(ptx)
}

/// Emits PTX for the strided symmetric rank-`jb` trailing Cholesky update.
///
/// A 2-D grid maps thread `(row, col)` to output element `A22[row, col]`. With
/// the panel operand `P` (leading dimension `ldp`) and trailing block `C = A22`
/// (leading dimension `ldc`):
///
/// * Lower (`A22 -= L21 · L21ᵀ`, active for `row >= col`):
///   `C[row,col] -= Σ_k L21[row,k]·L21[col,k]`, with `L21[r,k] -> r + k*ldp`.
/// * Upper (`A22 -= U12ᵀ · U12`, active for `row <= col`):
///   `C[row,col] -= Σ_k U12[k,row]·U12[k,col]`, with `U12[k,r] -> k + r*ldp`.
///
/// Only the active triangle is written, matching SYRK semantics and leaving the
/// opposite triangle untouched for the next block sweep. Leading dimensions are
/// honoured explicitly for correctness on embedded sub-blocks.
pub(crate) fn emit_chol_syrk<T: GpuFloat>(sm: SmVersion, is_lower: bool) -> SolverResult<String> {
    let name = chol_syrk_name::<T>(is_lower);
    let suffix = T::PTX_TYPE.as_ptx_str();

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("panel_ptr", PtxType::U64)
        .param("c_ptr", PtxType::U64)
        .param("rem", PtxType::U32)
        .param("jb", PtxType::U32)
        .param("ldp", PtxType::U32)
        .param("ldc", PtxType::U32)
        .body(move |b| {
            let (row, col) = b.global_thread_id_2d();
            let rem = b.load_param_u32("rem");
            let done = b.fresh_label("csyrk_done");
            let oob_r = b.alloc_reg(PtxType::Pred);
            let oob_c = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {oob_r}, {row}, {rem};"));
            b.raw_ptx(&format!("@{oob_r} bra {done};"));
            b.raw_ptx(&format!("setp.ge.u32 {oob_c}, {col}, {rem};"));
            b.raw_ptx(&format!("@{oob_c} bra {done};"));

            // Triangle guard: lower keeps row >= col, upper keeps row <= col.
            let off_tri = b.alloc_reg(PtxType::Pred);
            if is_lower {
                b.raw_ptx(&format!("setp.lt.u32 {off_tri}, {row}, {col};"));
            } else {
                b.raw_ptx(&format!("setp.gt.u32 {off_tri}, {row}, {col};"));
            }
            b.raw_ptx(&format!("@{off_tri} bra {done};"));

            let panel = b.load_param_u64("panel_ptr");
            let c_ptr = b.load_param_u64("c_ptr");
            let jb = b.load_param_u32("jb");
            let ldp = b.load_param_u32("ldp");
            let ldc = b.load_param_u32("ldc");

            // acc = 0.
            let acc = b.alloc_reg(T::PTX_TYPE);
            let zero_lit = if T::SIZE == 8 {
                "0d0000000000000000"
            } else {
                "0f00000000"
            };
            b.raw_ptx(&format!("mov{suffix} {acc}, {zero_lit};"));

            // k = 0..jb: acc += op(row,k) * op(col,k).
            let k = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k}, 0;"));
            let k_loop = b.fresh_label("csyrk_k");
            let k_exit = b.fresh_label("csyrk_kx");
            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k}, {jb};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));

            //  Lower: L21[row,k] -> row + k*ldp;  L21[col,k] -> col + k*ldp.
            //  Upper: U12[k,row] -> k + row*ldp;  U12[k,col] -> k + col*ldp.
            let (lidx, ridx) = if is_lower {
                (col_index(b, &row, &k, &ldp), col_index(b, &col, &k, &ldp))
            } else {
                (col_index(b, &k, &row, &ldp), col_index(b, &k, &col, &ldp))
            };
            let l_addr = b.byte_offset_addr(panel.clone(), lidx, T::size_u32());
            let r_addr = b.byte_offset_addr(panel.clone(), ridx, T::size_u32());
            let lval = load_global_float::<T>(b, l_addr);
            let rval = load_global_float::<T>(b, r_addr);
            let new_acc = fma_float::<T>(b, lval, rval, acc.clone());
            b.raw_ptx(&format!("mov{suffix} {acc}, {new_acc};"));
            b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            // C[row,col] -= acc.
            let c_idx = col_index(b, &row, &col, &ldc);
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

fn panel_cholesky_name<T: GpuFloat>(uplo: FillMode, block_size: u32) -> String {
    let tri = match uplo {
        FillMode::Upper => "upper",
        _ => "lower",
    };
    format!("solver_panel_cholesky_{tri}_{}_{}", T::NAME, block_size)
}

/// Computes the device address of block element `A[row, col]` (column-major,
/// leading dimension `lda`): `base + (row + col * lda) * sizeof(T)`.
fn block_elem_addr<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    base: &Register,
    row: &Register,
    col: &Register,
    lda: &Register,
) -> Register {
    let col_off = b.mul_lo_u32(col.clone(), lda.clone());
    let idx = b.add_u32(col_off, row.clone());
    b.byte_offset_addr(base.clone(), idx, T::size_u32())
}

/// Triangle-aware address of the block element addressed by a `(major, minor)`
/// coordinate.
///
/// For `Lower` the pivot direction (`major`) is the row index, so the element
/// is `A[major, minor]`. For `Upper` the factor is the transpose, so the
/// element is `A[minor, major]`. Expressing every access through this mapping
/// lets a single code path serve both triangles: a right-looking column
/// elimination for `Lower` becomes the identical row elimination for `Upper`.
fn tri_elem_addr<T: GpuFloat>(
    b: &mut BodyBuilder<'_>,
    is_lower: bool,
    base: &Register,
    major: &Register,
    minor: &Register,
    lda: &Register,
) -> Register {
    if is_lower {
        block_elem_addr::<T>(b, base, major, minor, lda)
    } else {
        block_elem_addr::<T>(b, base, minor, major, lda)
    }
}

/// Emits PTX for an in-place, single-CTA right-looking Cholesky factorization
/// of a `jb x jb` diagonal block held in global memory (column-major, stride
/// `lda`).
///
/// Ownership model (with `blockDim >= jb`): thread `t` owns block column `t`
/// (`Lower`) or block row `t` (`Upper`). For each pivot index `k = 0..jb`:
///
/// 1. The owner of `k` takes the square root of the pivot and scales its
///    sub-/super-diagonal entries: `A[r,k] /= sqrt(A[k,k])`.
/// 2. A block barrier publishes the finished pivot column/row.
/// 3. Every owner `t` with `k < t < jb` applies the rank-1 trailing update to
///    its own column/row: `A[r,t] -= A[r,k] * A[t,k]`.
///
/// Because each column/row is written by exactly one thread across all pivot
/// steps, the only cross-thread dependency is reading the freshly published
/// pivot column/row, which the per-step barrier covers. The `Upper`
/// (`A = Uᵀ U`) case is the transpose of the `Lower` (`A = L Lᵀ`) case and is
/// generated by the same body via [`tri_elem_addr`].
pub(crate) fn emit_panel_cholesky<T: GpuFloat>(
    sm: SmVersion,
    block_size: u32,
    uplo: FillMode,
) -> SolverResult<String> {
    let name = panel_cholesky_name::<T>(uplo, block_size);
    let is_lower = uplo != FillMode::Upper;

    let ptx = KernelBuilder::new(&name)
        .target(sm)
        .max_threads_per_block(SOLVER_BLOCK_SIZE)
        .param("diag_ptr", PtxType::U64)
        .param("jb", PtxType::U32)
        .param("lda", PtxType::U32)
        .body(move |b| {
            let tid = b.thread_id_x();
            let jb_reg = b.load_param_u32("jb");
            let lda_reg = b.load_param_u32("lda");
            let base = b.load_param_u64("diag_ptr");

            // Pivot loop counter `k`, iterated 0..jb at runtime.
            let k_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {k_reg}, 0;"));

            let k_loop = b.fresh_label("chol_k");
            let k_exit = b.fresh_label("chol_kx");

            b.raw_ptx(&format!("{k_loop}:"));
            let k_done = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {k_done}, {k_reg}, {jb_reg};"));
            b.raw_ptx(&format!("@{k_done} bra {k_exit};"));

            // --- Step 1: owner of pivot `k` factors the pivot column/row. ---
            let skip_scale = b.fresh_label("chol_skipscale");
            let not_owner = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ne.u32 {not_owner}, {tid}, {k_reg};"));
            b.raw_ptx(&format!("@{not_owner} bra {skip_scale};"));
            {
                // pivot = sqrt(A[k,k]).
                let diag_addr = tri_elem_addr::<T>(b, is_lower, &base, &k_reg, &k_reg, &lda_reg);
                let akk = load_global_float::<T>(b, diag_addr.clone());
                let pivot = sqrt_float::<T>(b, akk);
                store_global_float::<T>(b, diag_addr, pivot.clone());

                // A[r,k] /= pivot for r = k+1 .. jb.
                let r_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("add.u32 {r_reg}, {k_reg}, 1;"));
                let s_loop = b.fresh_label("chol_scale");
                let s_exit = b.fresh_label("chol_scalex");
                b.raw_ptx(&format!("{s_loop}:"));
                let s_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {s_done}, {r_reg}, {jb_reg};"));
                b.raw_ptx(&format!("@{s_done} bra {s_exit};"));
                let rk_addr = tri_elem_addr::<T>(b, is_lower, &base, &r_reg, &k_reg, &lda_reg);
                let v = load_global_float::<T>(b, rk_addr.clone());
                let scaled = div_float::<T>(b, v, pivot.clone());
                store_global_float::<T>(b, rk_addr, scaled);
                b.raw_ptx(&format!("add.u32 {r_reg}, {r_reg}, 1;"));
                b.raw_ptx(&format!("bra {s_loop};"));
                b.raw_ptx(&format!("{s_exit}:"));
            }
            b.raw_ptx(&format!("{skip_scale}:"));

            // Publish the finished pivot column/row to the whole block.
            b.bar_sync(0);

            // --- Step 2: owners `k < tid < jb` apply the trailing update. ---
            let skip_trail = b.fresh_label("chol_skiptrail");
            let le_k = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.le.u32 {le_k}, {tid}, {k_reg};"));
            b.raw_ptx(&format!("@{le_k} bra {skip_trail};"));
            let ge_jb = b.alloc_reg(PtxType::Pred);
            b.raw_ptx(&format!("setp.ge.u32 {ge_jb}, {tid}, {jb_reg};"));
            b.raw_ptx(&format!("@{ge_jb} bra {skip_trail};"));
            {
                // pivot factor for this column/row: A[tid,k].
                let tk_addr = tri_elem_addr::<T>(b, is_lower, &base, &tid, &k_reg, &lda_reg);
                let a_tk = load_global_float::<T>(b, tk_addr);

                // A[r,tid] -= A[r,k] * A[tid,k] for r = tid .. jb.
                let r_reg = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {r_reg}, {tid};"));
                let t_loop = b.fresh_label("chol_trail");
                let t_exit = b.fresh_label("chol_trailx");
                b.raw_ptx(&format!("{t_loop}:"));
                let t_done = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.ge.u32 {t_done}, {r_reg}, {jb_reg};"));
                b.raw_ptx(&format!("@{t_done} bra {t_exit};"));
                let rk_addr = tri_elem_addr::<T>(b, is_lower, &base, &r_reg, &k_reg, &lda_reg);
                let a_rk = load_global_float::<T>(b, rk_addr);
                let rt_addr = tri_elem_addr::<T>(b, is_lower, &base, &r_reg, &tid, &lda_reg);
                let a_rt = load_global_float::<T>(b, rt_addr.clone());
                let prod = mul_float::<T>(b, a_rk, a_tk.clone());
                let updated = sub_float::<T>(b, a_rt, prod);
                store_global_float::<T>(b, rt_addr, updated);
                b.raw_ptx(&format!("add.u32 {r_reg}, {r_reg}, 1;"));
                b.raw_ptx(&format!("bra {t_loop};"));
                b.raw_ptx(&format!("{t_exit}:"));
            }
            b.raw_ptx(&format!("{skip_trail}:"));

            // Trailing writes complete before the next pivot is factored.
            b.bar_sync(0);

            b.raw_ptx(&format!("add.u32 {k_reg}, {k_reg}, 1;"));
            b.raw_ptx(&format!("bra {k_loop};"));
            b.raw_ptx(&format!("{k_exit}:"));

            b.ret();
        })
        .build()?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chol_block_size_positive() {
        let block_size = CHOL_BLOCK_SIZE;
        assert!(block_size > 0);
        assert!(block_size <= 256);
    }

    #[test]
    fn panel_cholesky_name_format() {
        let name = panel_cholesky_name::<f32>(FillMode::Lower, 64);
        assert!(name.contains("f32"));
        assert!(name.contains("64"));
        assert!(name.contains("lower"));
        let upper = panel_cholesky_name::<f64>(FillMode::Upper, 32);
        assert!(upper.contains("f64"));
        assert!(upper.contains("upper"));
    }

    #[test]
    fn emit_panel_cholesky_is_not_empty() {
        // The factorization kernel must emit a real body (sqrt + division +
        // FMA-style trailing update), not a bare `ret`. A no-op body was the
        // historical correctness bug, so guard against its return.
        let sm = SmVersion::Sm86;
        let lower = emit_panel_cholesky::<f64>(sm, 3, FillMode::Lower)
            .expect("lower panel PTX must generate");
        assert!(lower.contains("sqrt"), "kernel must compute a square root");
        assert!(lower.contains("div"), "kernel must scale the pivot column");
        assert!(lower.contains("bar.sync"), "kernel must synchronize");
        let upper = emit_panel_cholesky::<f64>(sm, 3, FillMode::Upper)
            .expect("upper panel PTX must generate");
        assert!(upper.contains("sqrt"));
        assert!(upper.contains("bar.sync"));
    }

    /// Assembles `ptx` with `ptxas -arch=sm_86` to a throwaway object. Returns
    /// `Ok(())` on success (or when `ptxas` is absent) and the captured stderr on
    /// assembler failure.
    fn ptxas_assembles(ptx: &str, tag: &str) -> Result<(), String> {
        use std::process::Command;
        let dir = std::env::temp_dir();
        let src = dir.join(format!("oxicuda_chol_{tag}.ptx"));
        std::fs::write(&src, ptx).map_err(|e| format!("write ptx: {e}"))?;
        let out = Command::new("ptxas")
            .arg("-arch=sm_86")
            .arg(&src)
            .arg("-o")
            .arg("/dev/null")
            .output();
        let _ = std::fs::remove_file(&src);
        match out {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => Err(String::from_utf8_lossy(&o.stderr).into_owned()),
            Err(e) => {
                eprintln!("skipping ptxas validation ({tag}): {e}");
                Ok(())
            }
        }
    }

    #[test]
    fn chol_strided_trsm_ptx_assembles() {
        for is_lower in [true, false] {
            let tag = if is_lower { "lower" } else { "upper" };
            let ptx64 =
                emit_chol_panel_trsm::<f64>(SmVersion::Sm86, is_lower).expect("emit chol trsm f64");
            assert!(
                ptx64.contains("div.rn.f64"),
                "trsm must divide by the diagonal"
            );
            ptxas_assembles(&ptx64, &format!("trsm_{tag}_f64"))
                .expect("chol trsm f64 PTX must assemble");
            let ptx32 =
                emit_chol_panel_trsm::<f32>(SmVersion::Sm86, is_lower).expect("emit chol trsm f32");
            assert!(ptx32.contains("div.rn.f32"));
            ptxas_assembles(&ptx32, &format!("trsm_{tag}_f32"))
                .expect("chol trsm f32 PTX must assemble");
        }
    }

    #[test]
    fn chol_strided_syrk_ptx_assembles() {
        for is_lower in [true, false] {
            let tag = if is_lower { "lower" } else { "upper" };
            let ptx64 =
                emit_chol_syrk::<f64>(SmVersion::Sm86, is_lower).expect("emit chol syrk f64");
            assert!(ptx64.contains("fma.rn.f64"), "syrk must accumulate via FMA");
            assert!(ptx64.contains("sub.f64"), "syrk must subtract from C");
            ptxas_assembles(&ptx64, &format!("syrk_{tag}_f64"))
                .expect("chol syrk f64 PTX must assemble");
            let ptx32 =
                emit_chol_syrk::<f32>(SmVersion::Sm86, is_lower).expect("emit chol syrk f32");
            assert!(ptx32.contains("fma.rn.f32"));
            ptxas_assembles(&ptx32, &format!("syrk_{tag}_f32"))
                .expect("chol syrk f32 PTX must assemble");
        }
    }

    // ---------------------------------------------------------------------------
    // Cholesky + SYRK integration tests (CPU reference)
    // ---------------------------------------------------------------------------

    /// Compute Cholesky factorization L of a 2×2 SPD matrix A = [[a, b],[b, c]].
    /// Returns L such that L * L^T = A.
    fn cholesky_2x2(a: f64, b: f64, c: f64) -> [[f64; 2]; 2] {
        // L[0][0] = sqrt(a)
        // L[1][0] = b / L[0][0]
        // L[1][1] = sqrt(c - L[1][0]^2)
        let l00 = a.sqrt();
        let l10 = b / l00;
        let l11 = (c - l10 * l10).sqrt();
        [[l00, 0.0], [l10, l11]]
    }

    #[test]
    fn cholesky_syrk_trailing_update() {
        // For A = [[4, 2], [2, 3]], the Cholesky factor should be
        // L = [[2, 0], [1, sqrt(2)]].
        // Verify L * L^T = A to tolerance 1e-14.
        let l = cholesky_2x2(4.0, 2.0, 3.0);

        // L[0][0] = 2
        assert!((l[0][0] - 2.0).abs() < 1e-14, "L[0,0] = {}", l[0][0]);
        // L[1][0] = 1
        assert!((l[1][0] - 1.0).abs() < 1e-14, "L[1,0] = {}", l[1][0]);
        // L[1][1] = sqrt(2)
        assert!(
            (l[1][1] - 2.0_f64.sqrt()).abs() < 1e-14,
            "L[1,1] = {}",
            l[1][1]
        );
        // L[0][1] = 0 (strict lower triangular)
        assert!(l[0][1].abs() < 1e-15, "L[0,1] must be 0.0");

        // Reconstruct A = L * L^T.
        let a_rec = [
            [
                l[0][0] * l[0][0] + l[0][1] * l[0][1],
                l[0][0] * l[1][0] + l[0][1] * l[1][1],
            ],
            [
                l[1][0] * l[0][0] + l[1][1] * l[0][1],
                l[1][0] * l[1][0] + l[1][1] * l[1][1],
            ],
        ];

        let a_orig = [[4.0_f64, 2.0], [2.0, 3.0]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a_rec[i][j] - a_orig[i][j]).abs() < 1e-14,
                    "(L*L^T)[{i},{j}] = {} ≠ A[{i},{j}] = {}",
                    a_rec[i][j],
                    a_orig[i][j]
                );
            }
        }
    }

    #[test]
    fn cholesky_diagonal_is_positive() {
        // For any SPD matrix A, all diagonal entries of L are strictly positive.
        // Test several SPD matrices.
        let test_cases: &[(f64, f64, f64)] = &[
            (4.0, 2.0, 3.0),     // [[4,2],[2,3]]
            (9.0, 3.0, 5.0),     // [[9,3],[3,5]]
            (1.0, 0.0, 1.0),     // [[1,0],[0,1]] (identity)
            (16.0, 4.0, 4.0),    // [[16,4],[4,4+eps]] but diag must stay positive
            (100.0, 50.0, 50.0), // nearly singular SPD
        ];

        // For A = [[a, b],[b, c]] to be SPD: a > 0 and a*c - b^2 > 0.
        // Let's only test truly SPD cases.
        let spd_cases: &[(f64, f64, f64)] = &[(4.0, 2.0, 3.0), (9.0, 3.0, 5.0), (1.0, 0.0, 1.0)];
        let _ = test_cases; // suppress unused warning

        for &(a, b, c) in spd_cases {
            // Check SPD: a > 0 and det = a*c - b^2 > 0
            assert!(
                a > 0.0 && a * c - b * b > 0.0,
                "Test case [{a},{b},{b},{c}] must be SPD"
            );
            let l = cholesky_2x2(a, b, c);
            assert!(
                l[0][0] > 0.0,
                "L[0,0] = {} must be positive for a={a}",
                l[0][0]
            );
            assert!(
                l[1][1] > 0.0,
                "L[1,1] = {} must be positive for a={a}, b={b}, c={c}",
                l[1][1]
            );
        }
    }

    #[test]
    fn cholesky_backward_error_4x4_spd() {
        // A = D^T D where D is upper triangular, so A is SPD by construction:
        //   A = [[4, 2, 0, 0],
        //        [2, 4, 1, 0],
        //        [0, 1, 3, 1],
        //        [0, 0, 1, 2]]
        // Verify ||A - L*L^T||_F < n * eps * ||A||_F (backward error bound)
        let a = [
            [4.0_f64, 2.0, 0.0, 0.0],
            [2.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 3.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ];
        let norm_a = a
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let tol = 4.0 * 2.22e-16 * norm_a;

        // Compute L step by step (standard Cholesky):
        // L[0][0] = sqrt(4) = 2
        let l00 = a[0][0].sqrt();
        assert!(l00 > 0.0, "L[0,0] must be positive");
        // L[1][0] = a[1][0] / l00 = 2/2 = 1
        let l10 = a[1][0] / l00;
        // L[1][1] = sqrt(a[1][1] - l10^2) = sqrt(4 - 1) = sqrt(3)
        let l11 = (a[1][1] - l10 * l10).sqrt();
        assert!(l11 > 0.0, "L[1,1] must be positive");
        // L[2][0] = a[2][0] / l00 = 0
        let l20 = a[2][0] / l00;
        // L[2][1] = (a[2][1] - l20*l10) / l11 = (1 - 0) / sqrt(3)
        let l21 = (a[2][1] - l20 * l10) / l11;
        // L[2][2] = sqrt(a[2][2] - l20^2 - l21^2)
        let l22 = (a[2][2] - l20 * l20 - l21 * l21).sqrt();
        assert!(l22 > 0.0, "L[2,2] must be positive");
        // L[3][0] = a[3][0] / l00 = 0
        let l30 = a[3][0] / l00;
        // L[3][1] = (a[3][1] - l30*l10) / l11 = 0
        let l31 = (a[3][1] - l30 * l10) / l11;
        // L[3][2] = (a[3][2] - l30*l20 - l31*l21) / l22
        let l32 = (a[3][2] - l30 * l20 - l31 * l21) / l22;
        // L[3][3] = sqrt(a[3][3] - l30^2 - l31^2 - l32^2)
        let l33 = (a[3][3] - l30 * l30 - l31 * l31 - l32 * l32).sqrt();
        assert!(l33 > 0.0, "L[3,3] must be positive");

        // Verify reconstruction error for the 2×2 top-left sub-block (analytic check)
        let a00_recon = l00 * l00;
        let a10_recon = l10 * l00;
        let a11_recon = l10 * l10 + l11 * l11;
        assert!((a00_recon - a[0][0]).abs() < tol, "a[0][0] backward error");
        assert!((a10_recon - a[1][0]).abs() < tol, "a[1][0] backward error");
        assert!((a11_recon - a[1][1]).abs() < tol, "a[1][1] backward error");

        // Verify full reconstruction for all entries via L*L^T
        let l = [
            [l00, 0.0, 0.0, 0.0],
            [l10, l11, 0.0, 0.0],
            [l20, l21, l22, 0.0],
            [l30, l31, l32, l33],
        ];
        for i in 0..4 {
            for j in 0..=i {
                // (L * L^T)[i][j] = sum_k L[i][k] * L[j][k]
                let recon: f64 = (0..=j).map(|k| l[i][k] * l[j][k]).sum();
                assert!(
                    (recon - a[i][j]).abs() < tol,
                    "L*L^T[{i},{j}] = {recon} vs A[{i},{j}] = {}, err = {}",
                    a[i][j],
                    (recon - a[i][j]).abs()
                );
            }
        }
    }

    // ---------------------------------------------------------------------------
    // On-device factorization + solve tests (real GPU; skip when no NVIDIA card)
    // ---------------------------------------------------------------------------

    /// Builds a solver handle bound to device 0, or `None` (with a printed skip
    /// notice) when CUDA is unavailable so CPU-only hosts degrade gracefully.
    fn try_solver_handle() -> Option<(std::sync::Arc<oxicuda_driver::Context>, SolverHandle)> {
        if oxicuda_driver::init().is_err() {
            eprintln!("skipping device test: CUDA driver unavailable");
            return None;
        }
        let has_device = oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false);
        if !has_device {
            eprintln!("skipping device test: no NVIDIA CUDA device");
            return None;
        }
        let dev = oxicuda_driver::device::Device::get(0).expect("device 0 must be retrievable");
        let ctx = std::sync::Arc::new(
            oxicuda_driver::Context::new(&dev).expect("CUDA context must be creatable"),
        );
        let handle = SolverHandle::new(&ctx).expect("solver handle must be creatable");
        Some((ctx, handle))
    }

    /// Host reference: lower Cholesky `A = L Lᵀ` for a column-major SPD matrix
    /// (leading dimension `n`). Returns `L` column-major with the strict upper
    /// triangle zeroed. `l[col * n + row]` is `L[row, col]`.
    fn host_cholesky_lower(a: &[f64], n: usize) -> Vec<f64> {
        let mut l = vec![0.0_f64; n * n];
        for j in 0..n {
            let mut diag = a[j * n + j];
            for k in 0..j {
                diag -= l[k * n + j] * l[k * n + j];
            }
            let ljj = diag.sqrt();
            l[j * n + j] = ljj;
            for i in (j + 1)..n {
                let mut s = a[j * n + i];
                for k in 0..j {
                    s -= l[k * n + i] * l[k * n + j];
                }
                l[j * n + i] = s / ljj;
            }
        }
        l
    }

    /// `y = A · x` for a column-major matrix `a` (`a[col * n + row] == A[row, col]`).
    fn matvec_colmajor(a: &[f64], x: &[f64], n: usize) -> Vec<f64> {
        let mut y = vec![0.0_f64; n];
        for (j, &xj) in x.iter().enumerate().take(n) {
            for (i, yi) in y.iter_mut().enumerate().take(n) {
                *yi += a[j * n + i] * xj;
            }
        }
        y
    }

    /// Factors `a` (column-major SPD, lower) on the device and asserts the
    /// recovered factor matches the host reference `L` within `tol`.
    fn assert_device_factor_matches(handle: &mut SolverHandle, a: &[f64], n: usize, tol: f64) {
        let expected = host_cholesky_lower(a, n);
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(a).expect("upload A");
        cholesky::<f64>(handle, FillMode::Lower, &mut d_a, n as u32, n as u32)
            .expect("device Cholesky factorization");
        let mut factored = vec![0.0_f64; n * n];
        d_a.copy_to_host(&mut factored).expect("download factor");
        for j in 0..n {
            for i in j..n {
                let got = factored[j * n + i];
                let want = expected[j * n + i];
                assert!(
                    (got - want).abs() < tol,
                    "L[{i},{j}] device={got} expected={want} (|diff|={})",
                    (got - want).abs()
                );
            }
        }
    }

    /// Full factor + solve check: solves `A · X = B` on the device and asserts
    /// `A · X ≈ B` column-by-column within 1e-9.
    fn assert_device_solve(handle: &mut SolverHandle, a: &[f64], b: &[f64], n: usize, nrhs: usize) {
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(a).expect("upload A");
        cholesky::<f64>(handle, FillMode::Lower, &mut d_a, n as u32, n as u32)
            .expect("device Cholesky factorization");
        let mut d_b = oxicuda_memory::DeviceBuffer::from_host(b).expect("upload B");
        cholesky_solve::<f64>(
            handle,
            FillMode::Lower,
            &d_a,
            &mut d_b,
            n as u32,
            nrhs as u32,
        )
        .expect("device Cholesky solve");
        let mut x = vec![0.0_f64; n * nrhs];
        d_b.copy_to_host(&mut x).expect("download solution");
        for col in 0..nrhs {
            let xc = &x[col * n..(col + 1) * n];
            let bc = &b[col * n..(col + 1) * n];
            let ax = matvec_colmajor(a, xc, n);
            for i in 0..n {
                assert!(
                    (ax[i] - bc[i]).abs() < 1e-9,
                    "rhs {col}: (A·x)[{i}]={} != b[{i}]={} (|diff|={})",
                    ax[i],
                    bc[i],
                    (ax[i] - bc[i]).abs()
                );
            }
        }
    }

    /// Builds a well-conditioned column-major SPD matrix `A = M·Mᵀ + n·I` from a
    /// deterministic pseudo-random `M` (fixed LCG seed), so reproductions are
    /// bit-stable across runs.
    fn build_spd(n: usize) -> Vec<f64> {
        // Deterministic LCG (Numerical Recipes constants) → M in [-1, 1).
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
        };
        let mut m = vec![0.0_f64; n * n];
        for v in m.iter_mut() {
            *v = next();
        }
        // A = M·Mᵀ + n·I, column-major. A[i,j] = sum_k M[i,k]·M[j,k] (+ n if i==j).
        let mut a = vec![0.0_f64; n * n];
        for j in 0..n {
            for i in 0..n {
                let mut s = 0.0_f64;
                for k in 0..n {
                    s += m[k * n + i] * m[k * n + j];
                }
                if i == j {
                    s += n as f64;
                }
                a[j * n + i] = s;
            }
        }
        a
    }

    /// Maximum absolute reconstruction error `max |A - L·Lᵀ|` over the lower
    /// triangle, given the device factor `l` (column-major) and original `a`.
    fn max_recon_error_lower(a: &[f64], l: &[f64], n: usize) -> f64 {
        let mut worst = 0.0_f64;
        for j in 0..n {
            for i in j..n {
                // (L·Lᵀ)[i,j] = sum_{k<=min(i,j)} L[i,k]·L[j,k].
                let recon: f64 = (0..=j).map(|k| l[k * n + i] * l[k * n + j]).sum();
                let err = (recon - a[j * n + i]).abs();
                if err > worst {
                    worst = err;
                }
            }
        }
        worst
    }

    #[test]
    fn device_cholesky_factor_and_solve_3x3() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // SPD tridiagonal A = [[4,1,0],[1,3,1],[0,1,2]] (symmetric: column-major
        // storage equals row-major). Known factor:
        //   L = [[2, 0, 0], [0.5, 1.658312…, 0], [0, 0.603022…, 1.279204…]].
        let n = 3;
        let a = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];

        // Cross-check our host reference against the closed-form factor.
        let expected = host_cholesky_lower(&a, n);
        assert!((expected[0] - 2.0).abs() < 1e-12);
        assert!((expected[1] - 0.5).abs() < 1e-12);
        assert!((expected[2] - 0.0).abs() < 1e-12);
        assert!((expected[n + 1] - 2.75_f64.sqrt()).abs() < 1e-12);

        assert_device_factor_matches(&mut handle, &a, n, 1e-9);

        let b = vec![1.0, 2.0, 3.0];
        assert_device_solve(&mut handle, &a, &b, n, 1);
    }

    #[test]
    fn device_cholesky_solve_4x4_dense() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // Dense SPD A = [[4,2,0,0],[2,4,1,0],[0,1,3,1],[0,0,1,2]] (symmetric).
        let n = 4;
        let a = vec![
            4.0, 2.0, 0.0, 0.0, // column 0
            2.0, 4.0, 1.0, 0.0, // column 1
            0.0, 1.0, 3.0, 1.0, // column 2
            0.0, 0.0, 1.0, 2.0, // column 3
        ];
        assert_device_factor_matches(&mut handle, &a, n, 1e-9);
        let b = vec![1.0, 1.0, 1.0, 1.0];
        assert_device_solve(&mut handle, &a, &b, n, 1);
    }

    #[test]
    fn device_cholesky_solve_multi_rhs() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // nrhs = 2: B holds two right-hand sides column-major (n x nrhs).
        let n = 3;
        let a = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let b = vec![
            1.0, 2.0, 3.0, // rhs 0
            3.0, 1.0, 2.0, // rhs 1
        ];
        assert_device_solve(&mut handle, &a, &b, n, 2);
    }

    #[test]
    fn device_cholesky_upper_reconstructs_spd() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        // Factor with FillMode::Upper (A = Uᵀ U) and verify the reconstruction.
        let n = 3;
        let a = vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
        let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("upload A");
        cholesky::<f64>(&mut handle, FillMode::Upper, &mut d_a, n as u32, n as u32)
            .expect("device upper Cholesky");
        let mut factored = vec![0.0_f64; n * n];
        d_a.copy_to_host(&mut factored).expect("download factor");
        // U is upper triangular: U[r, c] = factored[c * n + r] for r <= c.
        let u_at = |r: usize, c: usize| if r <= c { factored[c * n + r] } else { 0.0 };
        for i in 0..n {
            for j in 0..n {
                // (Uᵀ U)[i,j] = sum_k U[k,i] * U[k,j].
                let recon: f64 = (0..n).map(|k| u_at(k, i) * u_at(k, j)).sum();
                let want = a[j * n + i];
                assert!(
                    (recon - want).abs() < 1e-9,
                    "(UᵀU)[{i},{j}]={recon} != A[{i},{j}]={want}"
                );
            }
        }
    }

    /// Blocked-path reproduction (n > CHOL_BLOCK_SIZE): factor a well-conditioned
    /// SPD matrix and verify `A = L·Lᵀ`. Sizes straddle the 64-wide block
    /// boundary so multiple block sweeps with the strided off-diagonal TRSM/SYRK
    /// kernels are exercised.
    #[test]
    fn device_cholesky_blocked_reconstructs_spd() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        for &n in &[3usize, 64, 65, 80, 100, 127, 200] {
            let a = build_spd(n);
            let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("upload A");
            cholesky::<f64>(&mut handle, FillMode::Lower, &mut d_a, n as u32, n as u32)
                .expect("device blocked Cholesky");
            let mut factored = vec![0.0_f64; n * n];
            d_a.copy_to_host(&mut factored).expect("download factor");
            let err = max_recon_error_lower(&a, &factored, n);
            // Tolerance scales mildly with n; ||A|| ~ n so a per-entry bound of
            // n·1e-11 stays far tighter than 1e-9 for the tested sizes.
            let tol = (n as f64) * 1e-11;
            assert!(
                err < tol,
                "n={n}: blocked Cholesky reconstruction error {err:e} exceeds tol {tol:e}"
            );
        }
    }

    /// Blocked-path factor + multi-RHS solve for sizes spanning the block
    /// boundary, lower fill mode.
    #[test]
    fn device_cholesky_blocked_solve_multi_rhs() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        for &n in &[3usize, 64, 65, 80, 127, 200] {
            let a = build_spd(n);
            // Two deterministic right-hand sides.
            let mut b = vec![0.0_f64; n * 2];
            for i in 0..n {
                b[i] = 1.0 + (i as f64) * 0.5;
                b[n + i] = 2.0 - (i as f64) * 0.25;
            }
            assert_device_solve(&mut handle, &a, &b, n, 2);
        }
    }

    /// Blocked-path upper-triangular factorization (`A = Uᵀ U`) reconstruction
    /// for sizes spanning the block boundary.
    #[test]
    fn device_cholesky_blocked_upper_reconstructs_spd() {
        let Some((_ctx, mut handle)) = try_solver_handle() else {
            return;
        };
        for &n in &[3usize, 64, 65, 80, 127, 200] {
            let a = build_spd(n);
            let mut d_a = oxicuda_memory::DeviceBuffer::from_host(&a).expect("upload A");
            cholesky::<f64>(&mut handle, FillMode::Upper, &mut d_a, n as u32, n as u32)
                .expect("device blocked upper Cholesky");
            let mut factored = vec![0.0_f64; n * n];
            d_a.copy_to_host(&mut factored).expect("download factor");
            // U[r,c] = factored[c*n + r] for r <= c; reconstruct (UᵀU)[i,j].
            let u_at = |r: usize, c: usize| if r <= c { factored[c * n + r] } else { 0.0 };
            let tol = (n as f64) * 1e-11;
            let mut worst = 0.0_f64;
            for i in 0..n {
                for j in i..n {
                    let recon: f64 = (0..n).map(|k| u_at(k, i) * u_at(k, j)).sum();
                    let err = (recon - a[j * n + i]).abs();
                    if err > worst {
                        worst = err;
                    }
                }
            }
            assert!(
                worst < tol,
                "n={n}: blocked upper Cholesky reconstruction error {worst:e} exceeds tol {tol:e}"
            );
        }
    }
}
