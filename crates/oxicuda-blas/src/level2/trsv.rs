//! TRSV -- Triangular solve: `op(A) * x = b`.
//!
//! Solves a triangular system of equations where `A` is upper or lower
//! triangular, with unit or non-unit diagonal. The solution `x` overwrites
//! the input `b` in-place.
//!
//! # GPU Strategy
//!
//! Triangular solve is inherently sequential along the substitution chain.
//!
//! - **Small N (`<= TRSV_SINGLE_BLOCK_MAX`)**: A single thread block performs
//!   forward/back substitution sequentially. Thread 0 computes each element in
//!   order, with the matrix and vector in global memory.
//!
//! - **Large N (`> TRSV_SINGLE_BLOCK_MAX`)**: A block-based level-set
//!   algorithm. The matrix is partitioned into `TRSV_BLOCK_SIZE`-sized
//!   diagonal blocks. For each diagonal block we issue the single-block
//!   solver kernel against an offset sub-pointer; for each panel of
//!   off-diagonal rows or columns we issue a GEMV-style update with
//!   `alpha = -1, beta = 1` (`b -= L_off * x_block`). The dependency chain
//!   between blocks is preserved by serialising the launches on the
//!   handle's stream.
//!
//! The blocked path is always correctness-equivalent to the single-block
//! path; it exists to support `n` larger than what one block can
//! reasonably handle in one launch, and to expose intra-block parallelism
//! for the off-diagonal updates.

use std::sync::Arc;

use oxicuda_driver::{CUdeviceptr, Module};
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::prelude::*;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::{DiagType, FillMode, GpuFloat, Layout, MatrixDesc, Transpose};

/// Maximum N for the single-block sequential kernel. Larger systems are
/// solved via [`trsv_blocked`] (the multi-block level-set path).
const TRSV_SINGLE_BLOCK_MAX: u32 = 4096;

/// Diagonal-block size used by the blocked TRSV path. Each diagonal block
/// is solved in one launch of the existing single-block kernel; each
/// off-diagonal panel becomes one GEMV launch with `alpha = -1, beta = 1`.
///
/// The block size is bounded above by [`TRSV_SINGLE_BLOCK_MAX`] so the
/// inner solver can always run in one block.
const TRSV_BLOCK_SIZE: u32 = 1024;

/// Threads per block for the GEMV update kernel used by [`trsv_blocked`].
const TRSV_GEMV_BLOCK: u32 = 256;

/// Solves `op(A) * x = b`, overwriting `b` with the solution `x`.
///
/// The triangular matrix `A` is specified by `uplo` (upper/lower), `trans`
/// (transpose mode), and `diag` (unit/non-unit diagonal).
///
/// # Arguments
///
/// * `handle` -- BLAS handle providing stream and device context.
/// * `uplo` -- Whether `A` is upper or lower triangular.
/// * `trans` -- Whether to use `A`, `A^T`, or `A^H`.
/// * `diag` -- Whether the diagonal is unit or non-unit.
/// * `n` -- Order of the triangular matrix `A`.
/// * `a` -- Matrix descriptor for `A`.
/// * `x` -- Input vector `b`, overwritten with the solution `x`.
/// * `incx` -- Stride between consecutive elements of `x`. Must be positive.
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `n` exceeds single-block limit.
/// Returns [`BlasError::InvalidArgument`] if `incx` is not positive.
/// Returns [`BlasError::BufferTooSmall`] if any buffer is undersized.
#[allow(clippy::too_many_arguments)]
pub fn trsv<T: GpuFloat>(
    handle: &BlasHandle,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: u32,
    a: &MatrixDesc<T>,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> BlasResult<()> {
    if n == 0 {
        return Ok(());
    }

    // The kernel addresses A as row-major; reject column-major descriptors.
    if a.layout == Layout::ColMajor {
        return Err(BlasError::InvalidArgument(
            "trsv: ColMajor layout is not supported by this kernel; pass a RowMajor descriptor"
                .to_string(),
        ));
    }

    validate_trsv_args(n, a, x, incx)?;

    if n > TRSV_SINGLE_BLOCK_MAX {
        return trsv_blocked(handle, uplo, trans, diag, n, a, x, incx);
    }

    let ptx = generate_trsv_ptx::<T>(handle.sm_version(), uplo, trans, diag, n)?;
    let module = Arc::new(Module::from_ptx(&ptx)?);
    let kernel = Kernel::from_module(module, "trsv")?;

    // Single block with enough threads to hold the vector in shared memory.
    // Thread 0 does the sequential solve; other threads are idle but present
    // for future warp-collaborative optimization.
    let block_size = n.min(256);
    let params = LaunchParams::new(1u32, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(a.ptr, x.as_device_ptr(), n, a.ld, incx as u32),
    )?;

    Ok(())
}

/// Blocked (level-set) TRSV for `n > TRSV_SINGLE_BLOCK_MAX`.
///
/// Solves `op(L) * x = b` for lower triangular `L` (or the analogous
/// upper-triangular system) by partitioning the matrix into
/// [`TRSV_BLOCK_SIZE`]-sized diagonal blocks. For each diagonal block the
/// existing single-block kernel solves the sub-system; the corresponding
/// solution segment is then used to update the still-unsolved entries of
/// `x` (which still hold the right-hand side `b`) via a GEMV-like kernel.
///
/// # Algorithm
///
/// **Forward substitution (Lower + NoTrans, Upper + Trans):**
///
/// ```text
/// for i in (0..n).step_by(B):
///     k = min(B, n - i)
///     // Diagonal solve: L[i:i+k, i:i+k] * x[i:i+k] = x[i:i+k]   (in place)
///     trsv_kernel(A_block_ii, x_segment_i, k)
///     if i + k < n:
///         // Off-diagonal update:
///         //   x[i+k:n] -= L[i+k:n, i:i+k] * x[i:i+k]
///         gemv_minus_one(A_panel_below, x_segment_i, x_remaining)
/// ```
///
/// **Backward substitution (Upper + NoTrans, Lower + Trans):** identical
/// loop run backwards; the off-diagonal update writes to `x[0..i]` using
/// the panel `A[0..i, i..i+k]` (or the transposed equivalent). The
/// per-block solver is the same single-block TRSV kernel that handles the
/// boundary cases, since `k <= TRSV_BLOCK_SIZE <= TRSV_SINGLE_BLOCK_MAX`.
///
/// The launches are issued in dependency order on the same stream;
/// successive kernels see the prior writes through stream-ordered
/// execution, preserving the substitution chain.
#[allow(clippy::too_many_arguments)]
fn trsv_blocked<T: GpuFloat>(
    handle: &BlasHandle,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    n: u32,
    a: &MatrixDesc<T>,
    x: &mut DeviceBuffer<T>,
    incx: i32,
) -> BlasResult<()> {
    // Compile both kernels once (the inner block solver and the GEMV
    // updater) and reuse the resulting `Kernel` objects across iterations.
    let trsv_ptx = generate_trsv_ptx::<T>(handle.sm_version(), uplo, trans, diag, n)?;
    let trsv_module = Arc::new(Module::from_ptx(&trsv_ptx)?);
    let trsv_kernel = Kernel::from_module(trsv_module, "trsv")?;

    let gemv_ptx = generate_trsv_update_gemv_ptx::<T>(handle.sm_version(), trans)?;
    let gemv_module = Arc::new(Module::from_ptx(&gemv_ptx)?);
    let gemv_kernel = Kernel::from_module(gemv_module, "trsv_update_gemv")?;

    let elem_bytes = u64::from(T::size_u32());
    let lda64 = u64::from(a.ld);
    let incx_u = incx as u32;
    let incx64 = u64::from(incx_u);

    let is_upper = matches!(uplo, FillMode::Upper);
    let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);
    // Same direction logic as the single-block path: forward iteration
    // covers Lower + NoTrans and Upper + Trans; backward covers the rest.
    let forward = is_upper == use_trans;

    let block = TRSV_BLOCK_SIZE;
    let inner_block_size = block.min(256);

    if forward {
        let mut i: u32 = 0;
        while i < n {
            let k = block.min(n - i);

            // ---- Diagonal solve: L[i:i+k, i:i+k] * x[i:i+k] = x[i:i+k] ----
            // The diagonal block lives at A[i, i] which, in row-major
            // (lda = #cols), is at byte offset (i * lda + i) * elem_bytes.
            let a_block_ptr = offset_ptr(a.ptr, u64::from(i) * lda64 + u64::from(i), elem_bytes);
            let x_segment_ptr = offset_ptr(x.as_device_ptr(), u64::from(i) * incx64, elem_bytes);

            let inner_threads = k.min(inner_block_size);
            let inner_params = LaunchParams::new(1u32, inner_threads.max(1));
            trsv_kernel.launch(
                &inner_params,
                handle.stream(),
                &(a_block_ptr, x_segment_ptr, k, a.ld, incx_u),
            )?;

            // ---- Off-diagonal panel update for everything below this block ----
            let remaining = n - (i + k);
            if remaining > 0 {
                // Update target: x[i+k .. n] -= L_panel * x[i .. i+k]
                let x_remaining_ptr =
                    offset_ptr(x.as_device_ptr(), u64::from(i + k) * incx64, elem_bytes);
                let panel_ptr = trsv_panel_below_ptr(a, i, k, use_trans, elem_bytes);

                launch_trsv_gemv_update(
                    &gemv_kernel,
                    handle,
                    panel_ptr,
                    x_segment_ptr,
                    x_remaining_ptr,
                    remaining,
                    k,
                    a.ld,
                    incx_u,
                )?;
            }

            i += k;
        }
    } else {
        // Backward substitution. We loop with `i` representing the start of
        // the current diagonal block (so the loop body is identical to the
        // forward case but iterated from high to low).
        let mut i_end = n;
        while i_end > 0 {
            let k = block.min(i_end);
            let i = i_end - k;

            // Diagonal solve at A[i, i].
            let a_block_ptr = offset_ptr(a.ptr, u64::from(i) * lda64 + u64::from(i), elem_bytes);
            let x_segment_ptr = offset_ptr(x.as_device_ptr(), u64::from(i) * incx64, elem_bytes);

            let inner_threads = k.min(inner_block_size);
            let inner_params = LaunchParams::new(1u32, inner_threads.max(1));
            trsv_kernel.launch(
                &inner_params,
                handle.stream(),
                &(a_block_ptr, x_segment_ptr, k, a.ld, incx_u),
            )?;

            // Off-diagonal panel update for everything above this block.
            let remaining = i;
            if remaining > 0 {
                let x_remaining_ptr = offset_ptr(x.as_device_ptr(), 0u64, elem_bytes);
                let panel_ptr = trsv_panel_above_ptr(a, i, k, use_trans, elem_bytes);

                launch_trsv_gemv_update(
                    &gemv_kernel,
                    handle,
                    panel_ptr,
                    x_segment_ptr,
                    x_remaining_ptr,
                    remaining,
                    k,
                    a.ld,
                    incx_u,
                )?;
            }

            i_end = i;
        }
    }

    Ok(())
}

/// Offsets a CUDA device pointer by `elements * elem_bytes`. The offset is
/// computed in `u64` to avoid 32-bit overflow on large matrices.
#[inline]
fn offset_ptr(base: CUdeviceptr, elements: u64, elem_bytes: u64) -> CUdeviceptr {
    base.wrapping_add(elements * elem_bytes)
}

/// Computes the device pointer to the row-major panel of `A` that lies
/// directly below the current diagonal block `[i..i+k, i..i+k]`. In the
/// transposed case (where the panel is read as `A[i..i+k, i+k..n]`) the
/// pointer points to `A[i, i+k]` instead.
#[inline]
fn trsv_panel_below_ptr<T: GpuFloat>(
    a: &MatrixDesc<T>,
    i: u32,
    k: u32,
    use_trans: bool,
    elem_bytes: u64,
) -> CUdeviceptr {
    let lda = u64::from(a.ld);
    let i_u64 = u64::from(i);
    let k_u64 = u64::from(k);
    if !use_trans {
        // Panel is L[i+k:n, i:i+k] -- panel base is A[i+k, i].
        offset_ptr(a.ptr, (i_u64 + k_u64) * lda + i_u64, elem_bytes)
    } else {
        // Trans case: the relevant panel reads `A[i:i+k, i+k:n]`. Panel
        // base is A[i, i+k] in row-major storage.
        offset_ptr(a.ptr, i_u64 * lda + i_u64 + k_u64, elem_bytes)
    }
}

/// Computes the device pointer to the row-major panel of `A` used for the
/// "above" update during the backward substitution (Upper + NoTrans,
/// Lower + Trans).
#[inline]
fn trsv_panel_above_ptr<T: GpuFloat>(
    a: &MatrixDesc<T>,
    i: u32,
    k: u32,
    use_trans: bool,
    elem_bytes: u64,
) -> CUdeviceptr {
    let lda = u64::from(a.ld);
    let i_u64 = u64::from(i);
    let k_u64 = u64::from(k);
    if !use_trans {
        // Panel is U[0:i, i:i+k] -- panel base is A[0, i].
        offset_ptr(a.ptr, i_u64, elem_bytes)
    } else {
        // Trans of an upper triangle (or NoTrans of a lower triangle)
        // pulls from L[i:i+k, 0:i] -- panel base is A[i, 0].
        let _ = k_u64;
        offset_ptr(a.ptr, i_u64 * lda, elem_bytes)
    }
}

/// Launches the GEMV update kernel used by [`trsv_blocked`] with the
/// hard-coded coefficients `alpha = -1, beta = 1`. The kernel name is
/// `"trsv_update_gemv"` and is generated by [`generate_trsv_update_gemv_ptx`].
#[allow(clippy::too_many_arguments)]
fn launch_trsv_gemv_update(
    kernel: &Kernel,
    handle: &BlasHandle,
    a_panel_ptr: CUdeviceptr,
    x_panel_ptr: CUdeviceptr,
    y_target_ptr: CUdeviceptr,
    output_len: u32,
    inner_len: u32,
    lda: u32,
    incx: u32,
) -> BlasResult<()> {
    let block_size = TRSV_GEMV_BLOCK;
    let grid_size = grid_size_for(output_len, block_size);
    let params = LaunchParams::new(grid_size, block_size);

    kernel.launch(
        &params,
        handle.stream(),
        &(
            a_panel_ptr,
            x_panel_ptr,
            y_target_ptr,
            lda,
            incx,
            output_len,
            inner_len,
        ),
    )?;
    Ok(())
}

/// Validates arguments for TRSV. Sizes above [`TRSV_SINGLE_BLOCK_MAX`]
/// are routed through [`trsv_blocked`]; the validator no longer rejects
/// them (the previous stub error has been replaced by the multi-block
/// implementation).
fn validate_trsv_args<T: GpuFloat>(
    n: u32,
    a: &MatrixDesc<T>,
    x: &DeviceBuffer<T>,
    incx: i32,
) -> BlasResult<()> {
    if incx <= 0 {
        return Err(BlasError::InvalidArgument(
            "incx must be positive".to_string(),
        ));
    }
    if a.rows < n || a.cols < n {
        return Err(BlasError::InvalidDimension(format!(
            "A must be at least {n}x{n}, got {}x{}",
            a.rows, a.cols
        )));
    }
    let x_req = required_elements(n, incx);
    if x.len() < x_req {
        return Err(BlasError::BufferTooSmall {
            expected: x_req,
            actual: x.len(),
        });
    }
    Ok(())
}

/// Generates PTX for the TRSV kernel (sequential single-block approach).
///
/// Thread 0 performs forward substitution (lower triangle) or back
/// substitution (upper triangle). The algorithm processes elements in order,
/// using global memory reads for both A and x.
fn generate_trsv_ptx<T: GpuFloat>(
    sm: SmVersion,
    uplo: FillMode,
    trans: Transpose,
    diag: DiagType,
    _n: u32,
) -> BlasResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let ptx_ty = T::PTX_TYPE;
    let is_upper = matches!(uplo, FillMode::Upper);
    let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);
    let is_unit = matches!(diag, DiagType::Unit);

    // Determine iteration direction:
    // Lower + NoTrans => forward (i = 0..n)
    // Upper + NoTrans => backward (i = n-1..0)
    // Lower + Trans   => backward
    // Upper + Trans   => forward
    let forward = is_upper == use_trans;

    KernelBuilder::new("trsv")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .body(move |b| {
            // Only thread 0 executes the sequential solve
            let tid = b.thread_id_x();
            let one_reg = b.alloc_reg(PtxType::U32);
            b.raw_ptx(&format!("mov.u32 {one_reg}, 1;"));

            b.if_lt_u32(tid, one_reg, |b| {
                let a_ptr = b.load_param_u64("a_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let n_reg = b.load_param_u32("n");
                let lda = b.load_param_u32("lda");
                let incx = b.load_param_u32("incx");

                // Outer loop: i iterates over each element to solve
                let outer_label = b.fresh_label("trsv_outer");
                let outer_done = b.fresh_label("trsv_outer_done");
                let i = b.alloc_reg(PtxType::U32);

                if forward {
                    b.raw_ptx(&format!("mov.u32 {i}, 0;"));
                } else {
                    // i = n - 1
                    b.raw_ptx(&format!("sub.u32 {i}, {n_reg}, 1;"));
                }

                b.label(&outer_label);

                // Check bounds
                let outer_pred = b.alloc_reg(PtxType::Pred);
                if forward {
                    b.raw_ptx(&format!("setp.lo.u32 {outer_pred}, {i}, {n_reg};"));
                } else {
                    // i >= 0 is always true for u32; check i < n (handles wrap-around)
                    b.raw_ptx(&format!("setp.lo.u32 {outer_pred}, {i}, {n_reg};"));
                }
                b.raw_ptx(&format!("@!{outer_pred} bra ${outer_done};"));

                // Load x[i * incx] (the current right-hand side value)
                let xi_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {xi_idx}, {i}, {incx};"));
                let xi_addr = b.byte_offset_addr(x_ptr.clone(), xi_idx, elem_bytes);
                let xi_val = load_float(b, xi_addr.clone(), is_f64);

                // Subtract contributions from previously solved elements
                // For forward: j = 0..i, for backward: j = i+1..n
                let inner_label = b.fresh_label("trsv_inner");
                let inner_done = b.fresh_label("trsv_inner_done");
                let j = b.alloc_reg(PtxType::U32);
                let sum = b.alloc_reg(ptx_ty);
                emit_zero(b, sum.clone(), is_f64);

                if forward {
                    b.raw_ptx(&format!("mov.u32 {j}, 0;"));
                } else {
                    let i_plus1 = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("add.u32 {i_plus1}, {i}, 1;"));
                    b.raw_ptx(&format!("mov.u32 {j}, {i_plus1};"));
                }

                b.label(&inner_label);
                let inner_pred = b.alloc_reg(PtxType::Pred);
                if forward {
                    b.raw_ptx(&format!("setp.lo.u32 {inner_pred}, {j}, {i};"));
                } else {
                    b.raw_ptx(&format!("setp.lo.u32 {inner_pred}, {j}, {n_reg};"));
                }
                b.raw_ptx(&format!("@!{inner_pred} bra ${inner_done};"));

                // A[i][j] or A[j][i] depending on transpose
                let (row, col) = if !use_trans {
                    (i.clone(), j.clone())
                } else {
                    (j.clone(), i.clone())
                };
                let row_off = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {row_off}, {row}, {lda};"));
                let a_idx = b.add_u32(row_off, col);
                let a_addr = b.byte_offset_addr(a_ptr.clone(), a_idx, elem_bytes);
                let a_val = load_float(b, a_addr, is_f64);

                // Load x[j * incx]
                let xj_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {xj_idx}, {j}, {incx};"));
                let xj_addr = b.byte_offset_addr(x_ptr.clone(), xj_idx, elem_bytes);
                let xj_val = load_float(b, xj_addr, is_f64);

                // sum += A[..][..] * x[j]
                let new_sum = if is_f64 {
                    b.fma_f64(a_val, xj_val, sum.clone())
                } else {
                    b.fma_f32(a_val, xj_val, sum.clone())
                };
                emit_mov_float(b, sum.clone(), new_sum, is_f64);

                b.raw_ptx(&format!("add.u32 {j}, {j}, 1;"));
                b.branch(&inner_label);

                b.label(&inner_done);

                // x[i] = (x[i] - sum) / A[i][i]
                let diff = if is_f64 {
                    b.sub_f64(xi_val, sum)
                } else {
                    b.sub_f32(xi_val, sum)
                };

                let result = if is_unit {
                    // Unit diagonal: x[i] = x[i] - sum (no division)
                    diff
                } else {
                    // Non-unit: divide by diagonal element A[i][i]
                    let diag_off = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {diag_off}, {i}, {lda};"));
                    let diag_idx = b.add_u32(diag_off, i.clone());
                    let diag_addr = b.byte_offset_addr(a_ptr.clone(), diag_idx, elem_bytes);
                    let diag_val = load_float(b, diag_addr, is_f64);

                    let r = b.alloc_reg(ptx_ty);
                    if is_f64 {
                        b.raw_ptx(&format!("div.rn.f64 {r}, {diff}, {diag_val};"));
                    } else {
                        b.raw_ptx(&format!("div.rn.f32 {r}, {diff}, {diag_val};"));
                    }
                    r
                };

                // Store x[i * incx] = result
                store_float(b, xi_addr, result, is_f64);

                // Advance i
                if forward {
                    b.raw_ptx(&format!("add.u32 {i}, {i}, 1;"));
                } else {
                    // i-- (subtract 1; will wrap to large value if i was 0)
                    b.raw_ptx(&format!("sub.u32 {i}, {i}, 1;"));
                }
                b.branch(&outer_label);

                b.label(&outer_done);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

// -- Shared helpers --

fn emit_zero(b: &mut BodyBuilder<'_>, reg: Register, is_f64: bool) {
    if is_f64 {
        b.raw_ptx(&format!("mov.b64 {reg}, 0d0000000000000000;"));
    } else {
        b.raw_ptx(&format!("mov.b32 {reg}, 0f00000000;"));
    }
}

fn emit_mov_float(b: &mut BodyBuilder<'_>, dst: Register, src: Register, is_f64: bool) {
    let ty = if is_f64 { "f64" } else { "f32" };
    b.raw_ptx(&format!("mov.{ty} {dst}, {src};"));
}

fn load_float(b: &mut BodyBuilder<'_>, addr: Register, is_f64: bool) -> Register {
    if is_f64 {
        b.load_global_f64(addr)
    } else {
        b.load_global_f32(addr)
    }
}

fn store_float(b: &mut BodyBuilder<'_>, addr: Register, val: Register, is_f64: bool) {
    if is_f64 {
        b.store_global_f64(addr, val);
    } else {
        b.store_global_f32(addr, val);
    }
}

fn required_elements(n: u32, inc: i32) -> usize {
    if n == 0 {
        return 0;
    }
    1 + (n as usize - 1) * inc.unsigned_abs() as usize
}

// ---------------------------------------------------------------------------
// PTX for the off-diagonal GEMV update used by `trsv_blocked`
// ---------------------------------------------------------------------------

/// Generates PTX for the dedicated GEMV-update kernel used by the blocked
/// TRSV path. The kernel computes `y -= A_panel * x_panel` element-wise:
///
/// - **NoTrans panel** (Lower + NoTrans / Upper + NoTrans):
///   `y[r] -= sum_c A[r, c] * x[c]`, where `A` is row-major with leading
///   dimension `lda`, and `r ∈ [0, output_len)`, `c ∈ [0, inner_len)`.
/// - **Trans panel** (Upper + Trans / Lower + Trans):
///   `y[r] -= sum_c A[c, r] * x[c]` -- the panel is read transposed.
///
/// The kernel is intentionally minimal: one thread per output element,
/// straight dot product, hard-coded `alpha = -1, beta = 1`. This keeps the
/// arg list short (no alpha/beta bits to pack) and avoids a dependency
/// on the higher-level `gemv()` wrapper.
fn generate_trsv_update_gemv_ptx<T: GpuFloat>(
    sm: SmVersion,
    trans: Transpose,
) -> BlasResult<String> {
    let is_f64 = T::SIZE == 8;
    let elem_bytes = T::size_u32();
    let ptx_ty = T::PTX_TYPE;
    let use_trans = matches!(trans, Transpose::Trans | Transpose::ConjTrans);

    KernelBuilder::new("trsv_update_gemv")
        .target(sm)
        .param("a_ptr", PtxType::U64)
        .param("x_ptr", PtxType::U64)
        .param("y_ptr", PtxType::U64)
        .param("lda", PtxType::U32)
        .param("incx", PtxType::U32)
        .param("output_len", PtxType::U32)
        .param("inner_len", PtxType::U32)
        .body(move |b| {
            let gid = b.global_thread_id_x();
            let output_len = b.load_param_u32("output_len");

            let gid_inner = gid.clone();
            b.if_lt_u32(gid, output_len, move |b| {
                let gid = gid_inner;
                let a_ptr = b.load_param_u64("a_ptr");
                let x_ptr = b.load_param_u64("x_ptr");
                let y_ptr = b.load_param_u64("y_ptr");
                let inner_len = b.load_param_u32("inner_len");
                let lda = b.load_param_u32("lda");
                let incx = b.load_param_u32("incx");

                // Accumulate the dot product: acc = sum_k A[..][..] * x[k * incx]
                let acc = b.alloc_reg(ptx_ty);
                emit_zero(b, acc.clone(), is_f64);

                let loop_label = b.fresh_label("trsv_gemv_loop");
                let done_label = b.fresh_label("trsv_gemv_done");
                let k = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mov.u32 {k}, 0;"));

                b.label(&loop_label);
                let pred = b.alloc_reg(PtxType::Pred);
                b.raw_ptx(&format!("setp.lo.u32 {pred}, {k}, {inner_len};"));
                b.raw_ptx(&format!("@!{pred} bra ${done_label};"));

                // Compute A address. row-major A with lda = stride between rows.
                // NoTrans: A[gid][k] => (gid * lda + k) * elem_bytes
                // Trans:   A[k][gid] => (k * lda + gid) * elem_bytes
                let row_off = b.alloc_reg(PtxType::U32);
                if !use_trans {
                    b.raw_ptx(&format!("mul.lo.u32 {row_off}, {gid}, {lda};"));
                    let idx = b.add_u32(row_off.clone(), k.clone());
                    let a_addr = b.byte_offset_addr(a_ptr.clone(), idx, elem_bytes);
                    let a_val = load_float(b, a_addr, is_f64);

                    // x[k * incx]
                    let x_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {x_idx}, {k}, {incx};"));
                    let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx, elem_bytes);
                    let x_val = load_float(b, x_addr, is_f64);

                    let new_acc = if is_f64 {
                        b.fma_f64(a_val, x_val, acc.clone())
                    } else {
                        b.fma_f32(a_val, x_val, acc.clone())
                    };
                    emit_mov_float(b, acc.clone(), new_acc, is_f64);
                } else {
                    b.raw_ptx(&format!("mul.lo.u32 {row_off}, {k}, {lda};"));
                    let idx = b.add_u32(row_off.clone(), gid.clone());
                    let a_addr = b.byte_offset_addr(a_ptr.clone(), idx, elem_bytes);
                    let a_val = load_float(b, a_addr, is_f64);

                    let x_idx = b.alloc_reg(PtxType::U32);
                    b.raw_ptx(&format!("mul.lo.u32 {x_idx}, {k}, {incx};"));
                    let x_addr = b.byte_offset_addr(x_ptr.clone(), x_idx, elem_bytes);
                    let x_val = load_float(b, x_addr, is_f64);

                    let new_acc = if is_f64 {
                        b.fma_f64(a_val, x_val, acc.clone())
                    } else {
                        b.fma_f32(a_val, x_val, acc.clone())
                    };
                    emit_mov_float(b, acc.clone(), new_acc, is_f64);
                }

                b.raw_ptx(&format!("add.u32 {k}, {k}, 1;"));
                b.branch(&loop_label);

                b.label(&done_label);

                // y[gid * incx] -= acc.  (incy == incx in the TRSV blocked
                // path; both views read/write the same stride along x.)
                let y_idx = b.alloc_reg(PtxType::U32);
                b.raw_ptx(&format!("mul.lo.u32 {y_idx}, {gid}, {incx};"));
                let y_addr = b.byte_offset_addr(y_ptr, y_idx, elem_bytes);
                let y_cur = load_float(b, y_addr.clone(), is_f64);

                let updated = if is_f64 {
                    b.sub_f64(y_cur, acc)
                } else {
                    b.sub_f32(y_cur, acc)
                };
                store_float(b, y_addr, updated, is_f64);
            });

            b.ret();
        })
        .build()
        .map_err(|e| BlasError::PtxGeneration(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trsv_ptx_generation_lower_notrans_nonunit() {
        let ptx = generate_trsv_ptx::<f32>(
            SmVersion::Sm80,
            FillMode::Lower,
            Transpose::NoTrans,
            DiagType::NonUnit,
            64,
        );
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry trsv"));
    }

    #[test]
    fn trsv_ptx_generation_upper_trans_unit() {
        let ptx = generate_trsv_ptx::<f64>(
            SmVersion::Sm80,
            FillMode::Upper,
            Transpose::Trans,
            DiagType::Unit,
            128,
        );
        assert!(ptx.is_ok());
        let ptx = ptx.expect("test: PTX generation should succeed");
        assert!(ptx.contains(".entry trsv"));
    }

    #[test]
    fn trsv_ptx_generation_various_sizes() {
        // Verify PTX generation succeeds for different sizes.
        for &sz in &[1, 32, 256, 512] {
            let ptx = generate_trsv_ptx::<f32>(
                SmVersion::Sm80,
                FillMode::Lower,
                Transpose::NoTrans,
                DiagType::NonUnit,
                sz,
            );
            assert!(ptx.is_ok(), "failed for n={sz}");
        }
    }

    // ---- blocked TRSV (n > TRSV_SINGLE_BLOCK_MAX) -----------------------

    /// The blocked path's GEMV update PTX must compile cleanly for both
    /// trans modes and both supported precisions.
    #[test]
    fn trsv_update_gemv_ptx_compiles_for_all_modes() {
        for &trans in &[Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            let f32_ptx = generate_trsv_update_gemv_ptx::<f32>(SmVersion::Sm80, trans);
            assert!(f32_ptx.is_ok(), "f32 TRSV update GEMV failed for {trans:?}");
            let f32_ptx = f32_ptx.expect("test: PTX generation should succeed");
            assert!(f32_ptx.contains(".entry trsv_update_gemv"));
            assert!(f32_ptx.contains("ld.global.f32"));
            assert!(f32_ptx.contains("st.global.f32"));
            assert!(f32_ptx.contains("fma.rn.f32"));
            assert!(f32_ptx.contains("sub.f32"));

            let f64_ptx = generate_trsv_update_gemv_ptx::<f64>(SmVersion::Sm80, trans);
            assert!(f64_ptx.is_ok(), "f64 TRSV update GEMV failed for {trans:?}");
            let f64_ptx = f64_ptx.expect("test: PTX generation should succeed");
            assert!(f64_ptx.contains(".entry trsv_update_gemv"));
            assert!(f64_ptx.contains("fma.rn.f64"));
            assert!(f64_ptx.contains("sub.f64"));
        }
    }

    /// Pointer-arithmetic spot-check for the panel-below helper. For
    /// `n = 8192`, `B = TRSV_BLOCK_SIZE`, the second iteration's panel for
    /// Lower + NoTrans starts at row `i + B = 2048` and column `i = 1024`,
    /// which is byte offset `(2048 * lda + 1024) * sizeof(T)` from the
    /// matrix base. The helper must agree with that algebra exactly --
    /// any drift here yields off-diagonal updates that read random data.
    #[test]
    fn trsv_panel_below_ptr_layout_lower_notrans() {
        let lda = 8192u32;
        let n = 8192u32;
        let elem_bytes = u64::from(<f32 as GpuFloat>::size_u32());
        // Synthesise a MatrixDesc with raw pointer so we don't need a
        // device allocation. The pointer is the offset we care about.
        let base: CUdeviceptr = 0x1000_0000;
        let a = MatrixDesc::<f32>::from_raw(base, n, n, lda, crate::types::Layout::RowMajor);
        let i = TRSV_BLOCK_SIZE; // 1024
        let k = TRSV_BLOCK_SIZE;
        let p = trsv_panel_below_ptr(&a, i, k, false, elem_bytes);
        let expected_offset = (u64::from(i + k) * u64::from(lda) + u64::from(i)) * elem_bytes;
        assert_eq!(p - base, expected_offset);
    }

    /// Same shape check for the trans variant: in the trans case the
    /// "below" panel is actually `A[i:i+k, i+k:n]`, so the base address
    /// is `A[i, i+k]`.
    #[test]
    fn trsv_panel_below_ptr_layout_upper_trans() {
        let lda = 8192u32;
        let n = 8192u32;
        let elem_bytes = u64::from(<f32 as GpuFloat>::size_u32());
        let base: CUdeviceptr = 0x2000_0000;
        let a = MatrixDesc::<f32>::from_raw(base, n, n, lda, crate::types::Layout::RowMajor);
        let i = TRSV_BLOCK_SIZE * 2;
        let k = TRSV_BLOCK_SIZE;
        let p = trsv_panel_below_ptr(&a, i, k, true, elem_bytes);
        let expected_offset =
            (u64::from(i) * u64::from(lda) + u64::from(i) + u64::from(k)) * elem_bytes;
        assert_eq!(p - base, expected_offset);
    }

    /// Blocked-loop iteration count: for `n = 8192` and
    /// `TRSV_BLOCK_SIZE = 1024` the forward path makes `n / B = 8`
    /// diagonal-block solves and `7` off-diagonal updates. This verifies
    /// the loop math doesn't drift with partial-last-block edge cases.
    #[test]
    fn trsv_blocked_iteration_count_8192() {
        let n: u32 = 8192;
        let block = TRSV_BLOCK_SIZE;
        let mut diag_solves = 0u32;
        let mut off_updates = 0u32;
        let mut i: u32 = 0;
        while i < n {
            let k = block.min(n - i);
            diag_solves += 1;
            if i + k < n {
                off_updates += 1;
            }
            i += k;
        }
        assert_eq!(diag_solves, 8);
        assert_eq!(off_updates, 7);
    }

    /// Same check for the partial-last-block case: `n = 8200` => 8 full
    /// blocks plus a 1-row tail, totalling 9 diagonal solves and 8 off-
    /// diagonal updates.
    #[test]
    fn trsv_blocked_iteration_count_partial_last() {
        let n: u32 = 8200;
        let block = TRSV_BLOCK_SIZE;
        let mut diag_solves = 0u32;
        let mut off_updates = 0u32;
        let mut i: u32 = 0;
        while i < n {
            let k = block.min(n - i);
            diag_solves += 1;
            if i + k < n {
                off_updates += 1;
            }
            i += k;
        }
        assert_eq!(diag_solves, 9);
        assert_eq!(off_updates, 8);
    }

    /// CPU reference for the blocked algorithm: this validates that the
    /// pure-host loop structure produces the correct triangular solve, so
    /// any later drift in the GPU dispatch math can be caught by reading
    /// the corresponding blocked-loop math here. The test is a host-side
    /// simulation only; it exercises *the algorithm*, not the device.
    #[test]
    fn cpu_reference_blocked_lower_notrans_matches_dense_solve() {
        // Generate a deterministic, well-conditioned lower-triangular L of
        // moderate size (the algorithm is the same at any n; the test
        // works on n = 256 to keep it fast in CI).
        let n: usize = 256;
        let lda = n;
        let block: usize = 32;

        let mut l = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..=i {
                // Diagonal slightly larger than 1 to keep things
                // numerically tame; off-diagonals small.
                l[i * lda + j] = if i == j {
                    1.0 + (i as f64) * 0.001
                } else {
                    -0.01 + 0.001 * ((i + j) as f64).cos()
                };
            }
        }

        // RHS b -- arbitrary deterministic vector.
        let b: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64).sin()).collect();

        // Reference: straight forward substitution, one element at a time.
        let mut x_ref = b.clone();
        for i in 0..n {
            let mut s = 0.0f64;
            for j in 0..i {
                s += l[i * lda + j] * x_ref[j];
            }
            x_ref[i] = (x_ref[i] - s) / l[i * lda + i];
        }

        // Blocked: same algorithm, partitioned into `block`-sized chunks.
        // This mirrors the GPU dispatch structure exactly; if it disagrees
        // with the reference, the dispatch math is wrong.
        let mut x_blk = b.clone();
        let mut i = 0usize;
        while i < n {
            let k = block.min(n - i);
            // Diagonal block solve in [i, i+k)
            for ii in i..(i + k) {
                let mut s = 0.0f64;
                for jj in i..ii {
                    s += l[ii * lda + jj] * x_blk[jj];
                }
                x_blk[ii] = (x_blk[ii] - s) / l[ii * lda + ii];
            }
            // Off-diagonal update for rows [i+k, n)
            if i + k < n {
                for r in (i + k)..n {
                    let mut s = 0.0f64;
                    for c in i..(i + k) {
                        s += l[r * lda + c] * x_blk[c];
                    }
                    x_blk[r] -= s;
                }
            }
            i += k;
        }

        for idx in 0..n {
            let diff = (x_blk[idx] - x_ref[idx]).abs();
            assert!(
                diff < 1e-10,
                "blocked algorithm diverged at index {idx}: diff={diff}"
            );
        }
    }

    /// CPU reference for the upper-triangular backward path. The same
    /// blocked algorithm should match a straight back-substitution
    /// reference within tight tolerance.
    #[test]
    fn cpu_reference_blocked_upper_notrans_matches_dense_solve() {
        let n: usize = 256;
        let lda = n;
        let block: usize = 32;

        let mut u = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                u[i * lda + j] = if i == j {
                    1.0 + (i as f64) * 0.001
                } else {
                    -0.01 + 0.001 * ((i + j) as f64).sin()
                };
            }
        }

        let b: Vec<f64> = (0..n).map(|i| 0.5 + (i as f64).cos()).collect();

        // Reference: back-substitution.
        let mut x_ref = b.clone();
        for i in (0..n).rev() {
            let mut s = 0.0f64;
            for j in (i + 1)..n {
                s += u[i * lda + j] * x_ref[j];
            }
            x_ref[i] = (x_ref[i] - s) / u[i * lda + i];
        }

        // Blocked back-substitution: matches the GPU dispatch loop
        // (Upper + NoTrans => backward iteration).
        let mut x_blk = b.clone();
        let mut i_end = n;
        while i_end > 0 {
            let k = block.min(i_end);
            let i = i_end - k;
            // Diagonal block solve in [i, i+k) using back-substitution.
            for ii in (i..(i + k)).rev() {
                let mut s = 0.0f64;
                for jj in (ii + 1)..(i + k) {
                    s += u[ii * lda + jj] * x_blk[jj];
                }
                x_blk[ii] = (x_blk[ii] - s) / u[ii * lda + ii];
            }
            // Off-diagonal update for rows [0, i)
            if i > 0 {
                for r in 0..i {
                    let mut s = 0.0f64;
                    for c in i..(i + k) {
                        s += u[r * lda + c] * x_blk[c];
                    }
                    x_blk[r] -= s;
                }
            }
            i_end = i;
        }

        for idx in 0..n {
            let diff = (x_blk[idx] - x_ref[idx]).abs();
            assert!(
                diff < 1e-10,
                "blocked upper-triangular algorithm diverged at index {idx}: diff={diff}"
            );
        }
    }

    /// `TRSV_BLOCK_SIZE` must remain `<= TRSV_SINGLE_BLOCK_MAX` so the
    /// inner solver can always fit one diagonal block in a single launch.
    /// Future tuning that grows the block size needs to update both
    /// constants in lockstep -- this guards against drift at compile
    /// time via a `const` assertion.
    const _: () = assert!(TRSV_BLOCK_SIZE <= TRSV_SINGLE_BLOCK_MAX);
}
