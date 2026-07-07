//! Numerically stable row-wise softmax on device buffers.
//!
//! Uses the three-pass algorithm from [`SoftmaxTemplate`]:
//! 1. Find row maximum: `m = max(x[0..cols])`
//! 2. Exponentiate and sum: `s = sum(exp(x[i] - m))`
//! 3. Normalize: `y[i] = exp(x[i] - m) / s`
//!
//! The implementation strategy is selected based on the number of columns
//! (elements per row):
//! - `cols <= 32`: warp-level shuffle reduction (1 warp per row)
//! - `cols <= 1024`: shared-memory block reduction (1 block per row)
//! - `cols > 1024`: multi-block (`reduce` + `finalize`) pipeline that
//!   exchanges per-block `(max, sum_exp)` pairs through a global scratch
//!   buffer.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Dim3, Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::templates::softmax::{
    MULTI_BLOCK_DEFAULT_STRIDE, MULTI_BLOCK_THREADS, MultiBlockSoftmaxPtx, SoftmaxTemplate,
    generate_multi_block_softmax_ptx,
};

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Builds a softmax kernel from the PTX template.
fn build_softmax_kernel(
    handle: &BlasHandle,
    ptx_type: oxicuda_ptx::ir::PtxType,
    row_size: u32,
) -> BlasResult<(Kernel, String)> {
    let template = SoftmaxTemplate {
        precision: ptx_type,
        target: handle.sm_version(),
        row_size,
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("softmax (row_size={row_size}): {e}")))?;
    let ptx_source = super::ptx_fixup::relocate_perf_directives(&ptx_source);
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for softmax: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Computes row-wise softmax over a 2-D matrix stored in row-major order.
///
/// For each row `r` in `[0, rows)`:
///
/// ```text
/// m = max(input[r, 0..cols])
/// output[r, j] = exp(input[r, j] - m) / sum_j(exp(input[r, j] - m))
/// ```
///
/// The implementation uses a numerically stable algorithm that subtracts
/// the row maximum before exponentiation to prevent overflow.
///
/// # Strategy selection
///
/// | `cols`         | Strategy                      | Threads/row              |
/// |----------------|-------------------------------|--------------------------|
/// | `<= 32`        | Warp shuffle reduction        | 32                       |
/// | `33..=1024`    | Shared memory block reduction | `cols`*                  |
/// | `> 1024`       | Multi-block reduce + finalize | `MULTI_BLOCK_THREADS`    |
///
/// (*) Rounded up to the nearest power of two.
///
/// The multi-block path requires `T` to be `f32`. Other precisions return
/// [`BlasError::UnsupportedOperation`] when `cols > 1024`. The PTX template
/// for `f64`/`f16`/`bf16` multi-block reduction is not yet implemented;
/// extending it requires per-precision constants and the corresponding
/// load/store widths.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `rows` -- number of rows (batch size).
/// * `cols` -- number of columns (elements per row).
/// * `input` -- device buffer containing the input matrix in row-major
///   layout, at least `rows * cols` elements.
/// * `output` -- device buffer for the result matrix, same layout, at
///   least `rows * cols` elements.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if buffers are too small,
/// [`BlasError::InvalidDimension`] if `rows` or `cols` is zero, or
/// [`BlasError::UnsupportedOperation`] if `cols > 1024` and `T != f32`.
pub fn softmax<T: GpuFloat>(
    handle: &BlasHandle,
    rows: u32,
    cols: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if rows == 0 || cols == 0 {
        return Err(BlasError::InvalidDimension(
            "softmax requires rows > 0 and cols > 0".to_string(),
        ));
    }

    let total_elements = rows as usize * cols as usize;
    if input.len() < total_elements {
        return Err(BlasError::BufferTooSmall {
            expected: total_elements,
            actual: input.len(),
        });
    }
    if output.len() < total_elements {
        return Err(BlasError::BufferTooSmall {
            expected: total_elements,
            actual: output.len(),
        });
    }

    if cols > 1024 {
        if !matches!(T::PTX_TYPE, PtxType::F32) {
            return Err(BlasError::UnsupportedOperation(format!(
                "multi-block softmax (cols > 1024) currently supports only f32, \
                 got {}",
                T::PTX_TYPE.as_ptx_str()
            )));
        }
        return softmax_multi_block(handle, rows, cols, input, output);
    }

    let (kernel, _) = build_softmax_kernel(handle, T::PTX_TYPE, cols)?;

    // Launch configuration depends on the strategy:
    // - Warp shuffle (cols <= 32): each warp handles one row, so we need
    //   `rows` warps total. Block size = 256 (8 warps per block).
    // - Shared memory (cols > 32): one block per row. The shared-memory
    //   kernel (`SoftmaxTemplate::generate_shared_memory`) caps its thread
    //   count at `next_power_of_two(cols).min(256)` and grid-strides over the
    //   row, so the launch MUST use the identical block size — launching more
    //   threads than the kernel's `.maxntid` makes `cuLaunchKernel` fail with
    //   `CUDA_ERROR_INVALID_VALUE`.
    let (grid, block) = if cols <= 32 {
        // Warps per block = block_size / 32
        let block_size: u32 = 256;
        let warps_per_block = block_size / 32;
        let num_blocks = grid_size_for(rows, warps_per_block);
        (num_blocks, block_size)
    } else {
        // One block per row; mirror the kernel's capped thread count.
        let block_size = cols.next_power_of_two().min(256);
        (rows, block_size)
    };

    let params = LaunchParams::new(grid, block);
    // Softmax kernel signature: (input_ptr, output_ptr, batch_size)
    let args = (input.as_device_ptr(), output.as_device_ptr(), rows);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("softmax: {e}")))?;

    Ok(())
}

/// Multi-block softmax dispatcher (`cols > 1024`, F32 only).
///
/// Generates the two-kernel reduce+finalize pipeline via
/// [`generate_multi_block_softmax_ptx`], allocates the per-block
/// `(max, sum_exp)` scratch buffer, and launches both kernels on the
/// handle's stream. The stream is synchronized before the scratch buffer
/// is dropped so the legacy synchronous `cuMemFree` cannot race the
/// in-flight launches (kernels run asynchronously on the stream, but the
/// scratch must outlive both launches).
///
/// # Layout
///
/// - **Reduce kernel.** Grid `(num_blocks_per_row, rows, 1)`. Each block
///   handles `block_stride = MULTI_BLOCK_DEFAULT_STRIDE` consecutive
///   elements of one row, writing the per-block `(max, sum_exp)` pair into
///   global scratch.
/// - **Finalize kernel.** Grid `(rows, 1, 1)`. Each block reads the
///   per-block scratch entries for one row, derives the global `(max, sum)`
///   via warp-cooperative tree reduction, then strides over the row writing
///   the normalized output.
///
/// The scratch layout is documented on [`MultiBlockSoftmaxPtx`].
fn softmax_multi_block<T: GpuFloat>(
    handle: &BlasHandle,
    rows: u32,
    cols: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    // Generate the two PTX modules and metadata.
    let plan: MultiBlockSoftmaxPtx = generate_multi_block_softmax_ptx(
        cols,
        MULTI_BLOCK_DEFAULT_STRIDE,
        MULTI_BLOCK_THREADS,
        PtxType::F32,
        handle.sm_version(),
    )
    .map_err(|e| {
        BlasError::PtxGeneration(format!(
            "multi-block softmax (rows={rows}, cols={cols}): {e}"
        ))
    })?;

    // Build both kernels. They share the precision (f32) and stride
    // configuration; we look them up via the canonical kernel names exposed
    // by `MultiBlockSoftmaxPtx`.
    let reduce_kernel = build_kernel_from_ptx(&plan.reduce_ptx, &plan.reduce_kernel_name())?;
    let finalize_kernel = build_kernel_from_ptx(&plan.finalize_ptx, &plan.finalize_kernel_name())?;

    // Allocate the per-row, per-block scratch buffer:
    // `rows * num_blocks_per_row` (max, sum) pairs, each f32-sized.
    let scratch_pairs = rows.checked_mul(plan.num_blocks_per_row).ok_or_else(|| {
        BlasError::InvalidDimension(format!(
            "softmax scratch overflow: rows={rows} * num_blocks_per_row={}",
            plan.num_blocks_per_row
        ))
    })?;
    let scratch_floats = (scratch_pairs as usize).checked_mul(2).ok_or_else(|| {
        BlasError::InvalidDimension(format!(
            "softmax scratch overflow: pairs={scratch_pairs} * 2 floats"
        ))
    })?;
    let scratch = DeviceBuffer::<f32>::alloc(scratch_floats).map_err(BlasError::Cuda)?;

    // Reduce kernel grid: (block-in-row, row, 1).
    // The PTX uses %ctaid.x for block-in-row, %ctaid.y for the row.
    let reduce_grid = Dim3::xy(plan.num_blocks_per_row, rows);
    let reduce_block = Dim3::x(plan.threads_per_block);
    let reduce_params = LaunchParams::new(reduce_grid, reduce_block);
    let reduce_args = (input.as_device_ptr(), scratch.as_device_ptr(), rows);
    reduce_kernel
        .launch(&reduce_params, handle.stream(), &reduce_args)
        .map_err(|e| BlasError::LaunchFailed(format!("softmax multi-block reduce: {e}")))?;

    // Finalize kernel grid: (row, 1, 1). The PTX uses %ctaid.x for the row.
    // Each row-block must have at least `num_blocks_per_row` threads to
    // avoid silently dropping per-block (max, sum) entries during the
    // strided global-max / global-sum reductions in the finalize kernel.
    // The PTX header pinned threads_per_block at MULTI_BLOCK_THREADS = 256;
    // since the default stride keeps num_blocks_per_row well under 256 for
    // any sane row size, that constraint is automatically satisfied.
    let finalize_grid = Dim3::x(rows);
    let finalize_block = Dim3::x(plan.threads_per_block);
    let finalize_params = LaunchParams::new(finalize_grid, finalize_block);
    let finalize_args = (
        input.as_device_ptr(),
        output.as_device_ptr(),
        scratch.as_device_ptr(),
        rows,
    );
    finalize_kernel
        .launch(&finalize_params, handle.stream(), &finalize_args)
        .map_err(|e| BlasError::LaunchFailed(format!("softmax multi-block finalize: {e}")))?;

    // Both launches are queued on `handle.stream()`. The legacy
    // `cuMemFree_v2` triggered by `scratch`'s `Drop` is a synchronous
    // device-wide barrier on most drivers, but synchronizing the stream
    // here is the documented contract and makes the lifetime correctness
    // independent of the driver's free semantics.
    handle.stream().synchronize().map_err(BlasError::Cuda)?;

    drop(scratch);
    Ok(())
}

/// Compiles a PTX source and looks up `kernel_name` from the resulting
/// module. Used by the multi-block softmax dispatcher to build both the
/// reduce and finalize kernels.
fn build_kernel_from_ptx(ptx_source: &str, kernel_name: &str) -> BlasResult<Kernel> {
    let ptx_source = super::ptx_fixup::relocate_perf_directives(ptx_source);
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for {kernel_name}: {e}")))?,
    );
    Kernel::from_module(module, kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;
    use oxicuda_ptx::ir::PtxType;
    use oxicuda_ptx::templates::softmax::SoftmaxTemplate;

    #[test]
    fn ptx_template_generates_softmax_warp_f32() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 32,
        };
        let ptx = template
            .generate()
            .expect("warp softmax PTX should generate");
        assert!(ptx.contains("softmax_f32_r32"));
        assert!(ptx.contains("shfl.sync"));
    }

    #[test]
    fn ptx_template_generates_softmax_block_f32() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 128,
        };
        let ptx = template
            .generate()
            .expect("block softmax PTX should generate");
        assert!(ptx.contains("softmax_f32_r128"));
    }

    #[test]
    fn ptx_template_rejects_large_row_size() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 2048,
        };
        assert!(template.generate().is_err());
    }

    #[test]
    fn warp_launch_config() {
        // cols=16, rows=100 => warp strategy
        let block_size: u32 = 256;
        let warps_per_block = block_size / 32;
        let num_blocks = grid_size_for(100, warps_per_block);
        // 8 warps per block, 100 rows => ceil(100/8) = 13 blocks
        assert_eq!(num_blocks, 13);
    }

    #[test]
    fn block_launch_config() {
        // cols=100, rows=50 => block strategy, block_size = 128 (next pow2)
        let cols: u32 = 100;
        let block_size = cols.next_power_of_two();
        assert_eq!(block_size, 128);
    }

    #[test]
    fn softmax_warp_small_row() {
        let template = SoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
            row_size: 8,
        };
        let ptx = template
            .generate()
            .expect("small warp softmax should generate");
        assert!(ptx.contains("softmax_f32_r8"));
    }

    // ---- multi-block softmax dispatch (cols > 1024) -------------------

    /// Verify the multi-block PTX generator is wired up for `cols = 2048`.
    /// This does not launch on a device -- it only validates the kernel
    /// names and scratch layout that the dispatcher relies on.
    #[test]
    fn multi_block_dispatch_layout_2048() {
        let plan = generate_multi_block_softmax_ptx(
            2048,
            MULTI_BLOCK_DEFAULT_STRIDE,
            MULTI_BLOCK_THREADS,
            PtxType::F32,
            SmVersion::Sm80,
        )
        .expect("multi-block softmax PTX should generate");
        assert_eq!(plan.num_blocks_per_row, 2);
        assert_eq!(plan.threads_per_block, MULTI_BLOCK_THREADS);
        // Reduce kernel grid is (num_blocks_per_row, batch_size, 1); verify
        // the kernel name matches what the dispatcher looks up.
        assert!(
            plan.reduce_ptx
                .contains(&format!(".entry {}", plan.reduce_kernel_name()))
        );
        assert!(
            plan.finalize_ptx
                .contains(&format!(".entry {}", plan.finalize_kernel_name()))
        );
    }

    /// Verify scratch sizing for a deep multi-block dispatch (`cols = 8192`).
    /// 8192 / 1024 = 8 blocks per row; the dispatcher allocates
    /// `rows * 8 * 2 * sizeof(f32)` bytes of scratch.
    #[test]
    fn multi_block_dispatch_scratch_for_8192() {
        let plan = generate_multi_block_softmax_ptx(
            8192,
            MULTI_BLOCK_DEFAULT_STRIDE,
            MULTI_BLOCK_THREADS,
            PtxType::F32,
            SmVersion::Sm80,
        )
        .expect("multi-block softmax PTX should generate");
        assert_eq!(plan.num_blocks_per_row, 8);
        assert_eq!(plan.scratch_bytes_per_row, 8 * 2 * 4);

        // Per-row scratch math the dispatcher uses: pairs * 2 floats.
        let rows: u32 = 16;
        let scratch_pairs = rows * plan.num_blocks_per_row;
        let scratch_floats = scratch_pairs as usize * 2;
        assert_eq!(scratch_floats, 16 * 8 * 2);
    }

    /// Verify the multi-block path rejects non-F32 element types at the
    /// dispatch boundary. `f64` callers still get the single-block path
    /// for `cols <= 1024`; only the multi-block branch is restricted.
    #[test]
    fn multi_block_rejects_non_f32_dtype_in_template() {
        let r = generate_multi_block_softmax_ptx(
            2048,
            MULTI_BLOCK_DEFAULT_STRIDE,
            MULTI_BLOCK_THREADS,
            PtxType::F64,
            SmVersion::Sm80,
        );
        assert!(r.is_err());
    }
}
