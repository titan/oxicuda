//! Broadcast **bias-add** over device buffers: `out[i, j] = in[i, j] + bias[j]`.
//!
//! Adds a length-`n` bias vector to every row of a row-major `m x n` matrix.
//! This is the standard post-GEMM bias broadcast in a transformer projection /
//! feed-forward: the matmul yields an `[m, n]` activation, and each column `j`
//! receives `bias[j]` added across all `m` rows.
//!
//! PTX is generated via [`BiasAddTemplate`], loaded into the driver, and
//! launched on the handle's stream as a flat one-thread-per-element map over
//! the `m * n` elements (the kernel recovers the column as `tid % n` to index
//! the bias).

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::templates::bias_add::BiasAddTemplate;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Standard block size for the flat bias-add kernel.
///
/// 256 is the occupancy-friendly default shared with the other elementwise
/// kernels; the grid is sized to cover `m * n` threads.
const BIAS_ADD_BLOCK: u32 = 256;

/// Builds the bias-add kernel from the PTX template.
fn build_bias_add_kernel(handle: &BlasHandle, ptx_type: PtxType) -> BlasResult<(Kernel, String)> {
    let template = BiasAddTemplate {
        precision: ptx_type,
        target: handle.sm_version(),
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("bias_add: {e}")))?;
    let module = Arc::new(
        Module::from_ptx(&ptx_source)
            .map_err(|e| BlasError::LaunchFailed(format!("module load for bias_add: {e}")))?,
    );
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Adds a broadcast bias vector to every row of a row-major matrix:
/// `output[i, j] = input[i, j] + bias[j]`.
///
/// `input` and `output` are `m x n` matrices stored row-major; `bias` is a
/// length-`n` vector added to each of the `m` rows. The result is written to
/// `output` (a separate buffer — this is an out-of-place op).
///
/// This is the device-buffer equivalent of the post-GEMM bias broadcast used
/// by transformer linear layers, and a 1:1 match for the `trustformers`
/// `add_bias_gpu_to_gpu` CUDA op.
///
/// # Parallelization
///
/// One thread per output element, `ceil(m * n / 256)` blocks of 256 threads.
/// Each thread recovers its column as `tid % n` to fetch the matching bias
/// entry.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `m` -- number of rows.
/// * `n` -- number of columns (and the bias length).
/// * `input` -- device buffer holding the `m x n` input matrix (row-major), at
///   least `m * n` elements.
/// * `bias` -- device buffer holding the length-`n` bias vector, at least `n`
///   elements.
/// * `output` -- device buffer for the `m x n` result (row-major), at least
///   `m * n` elements.
///
/// # Type support
///
/// Supports `f32` and `f64`. Half precisions (`f16`/`bf16`) are rejected with
/// [`BlasError::UnsupportedOperation`] to match the [`BiasAddTemplate`]
/// restriction (and the `causal_softmax` precedent).
///
/// # Errors
///
/// Returns [`BlasError::InvalidDimension`] if `m` or `n` is zero (or `m * n`
/// overflows), [`BlasError::UnsupportedOperation`] if `T` is not `f32`/`f64`,
/// [`BlasError::BufferTooSmall`] if any buffer is undersized, or
/// [`BlasError::PtxGeneration`] / [`BlasError::LaunchFailed`] if kernel
/// construction or launch fails.
pub fn bias_add<T: GpuFloat>(
    handle: &BlasHandle,
    m: u32,
    n: u32,
    input: &DeviceBuffer<T>,
    bias: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if m == 0 || n == 0 {
        return Err(BlasError::InvalidDimension(
            "bias_add requires m > 0 and n > 0".to_string(),
        ));
    }

    if !matches!(T::PTX_TYPE, PtxType::F32 | PtxType::F64) {
        return Err(BlasError::UnsupportedOperation(format!(
            "bias_add supports only f32/f64, got {}",
            T::PTX_TYPE.as_ptx_str()
        )));
    }

    let total_elements = (m as usize).checked_mul(n as usize).ok_or_else(|| {
        BlasError::InvalidDimension(format!("bias_add dimension overflow: m={m} * n={n}"))
    })?;

    if input.len() < total_elements {
        return Err(BlasError::BufferTooSmall {
            expected: total_elements,
            actual: input.len(),
        });
    }
    if bias.len() < n as usize {
        return Err(BlasError::BufferTooSmall {
            expected: n as usize,
            actual: bias.len(),
        });
    }
    if output.len() < total_elements {
        return Err(BlasError::BufferTooSmall {
            expected: total_elements,
            actual: output.len(),
        });
    }

    let (kernel, _name) = build_bias_add_kernel(handle, T::PTX_TYPE)?;

    // One thread per element, 256 threads per block, sized to cover m * n.
    let total_u32 = u32::try_from(total_elements).map_err(|_| {
        BlasError::InvalidDimension(format!(
            "bias_add element count {total_elements} exceeds u32 grid range"
        ))
    })?;
    let grid = grid_size_for(total_u32, BIAS_ADD_BLOCK);
    let params = LaunchParams::new(grid, BIAS_ADD_BLOCK);

    // Kernel signature: (input_ptr, bias_ptr, output_ptr, m, n).
    let args = (
        input.as_device_ptr(),
        bias.as_device_ptr(),
        output.as_device_ptr(),
        m,
        n,
    );

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("bias_add: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    // ---- PTX generation (host-only, no GPU) -----------------------------

    #[test]
    fn ptx_template_generates_bias_add_f32() {
        let template = BiasAddTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = template.generate().expect("bias_add PTX should generate");
        assert!(ptx.contains("bias_add_f32"));
        assert!(ptx.contains("add.f32 %f2, %f0, %f1;"));
    }

    #[test]
    fn block_size_is_power_of_two() {
        assert!(BIAS_ADD_BLOCK.is_power_of_two());
        const { assert!(BIAS_ADD_BLOCK >= 32) };
    }

    #[test]
    fn launch_grid_covers_all_elements() {
        // 300 elements, 256 threads/block => 2 blocks.
        assert_eq!(grid_size_for(300, BIAS_ADD_BLOCK), 2);
        // Exactly one full block.
        assert_eq!(grid_size_for(256, BIAS_ADD_BLOCK), 1);
        // A single element still needs one block.
        assert_eq!(grid_size_for(1, BIAS_ADD_BLOCK), 1);
    }

    // ---- CPU reference for the intended math ----------------------------

    /// Naive double-loop bias broadcast over a row-major `m x n` matrix. This is
    /// the ground truth the device kernel must reproduce: every row gets the same
    /// length-`n` bias added.
    fn bias_add_reference(input: &[f32], bias: &[f32], m: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = input[i * n + j] + bias[j];
            }
        }
        out
    }

    #[test]
    fn reference_adds_bias_per_column() {
        // 3x4 matrix, bias broadcast down each column.
        let m = 3;
        let n = 4;
        let input: Vec<f32> = (0..(m * n)).map(|i| i as f32).collect();
        let bias = vec![10.0f32, 20.0, 30.0, 40.0];
        let out = bias_add_reference(&input, &bias, m, n);

        for i in 0..m {
            for j in 0..n {
                let expect = input[i * n + j] + bias[j];
                assert_eq!(out[i * n + j], expect, "mismatch at ({i},{j})");
            }
        }
        // Spot check a couple of explicit values.
        assert_eq!(out[0], 0.0 + 10.0);
        assert_eq!(out[5], 5.0 + 20.0);
        assert_eq!(out[11], 11.0 + 40.0);
    }

    #[test]
    fn reference_single_row_is_plain_vector_add() {
        // m = 1 reduces to a plain elementwise vector add.
        let m = 1;
        let n = 5;
        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bias = vec![0.5f32, -1.0, 2.0, 0.0, 10.0];
        let out = bias_add_reference(&input, &bias, m, n);
        assert_eq!(out, vec![1.5, 1.0, 5.0, 4.0, 15.0]);
    }

    // ---- Device test (skipped unless a GPU is present) ------------------
    //
    // The device kernel runs on a GPU box later; on this CI host oxicuda
    // runtime-loads libcuda and there is no device, so we only attempt the
    // launch when a context can actually be created. The CPU reference above
    // is what pins the intended math; this guards the wiring end-to-end when
    // hardware is available.

    #[test]
    fn device_matches_reference_when_gpu_present() {
        use oxicuda_driver::{Context, Device};

        let device = match Device::get(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("device_matches_reference_when_gpu_present: no GPU, skipping device run");
                return;
            }
        };
        let ctx = match Context::new(&device) {
            Ok(c) => Arc::new(c),
            Err(_) => {
                eprintln!("device_matches_reference_when_gpu_present: no context, skipping");
                return;
            }
        };
        let handle = match BlasHandle::new(&ctx) {
            Ok(h) => h,
            Err(_) => {
                eprintln!("device_matches_reference_when_gpu_present: no BLAS handle, skipping");
                return;
            }
        };

        let m = 3usize;
        let n = 4usize;
        let host_in: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let host_bias: Vec<f32> = (0..n).map(|j| (j as f32) - 1.5).collect();
        let expected = bias_add_reference(&host_in, &host_bias, m, n);

        let input = DeviceBuffer::<f32>::from_host(&host_in).expect("upload input");
        let bias = DeviceBuffer::<f32>::from_host(&host_bias).expect("upload bias");
        let mut output = DeviceBuffer::<f32>::zeroed(m * n).expect("alloc output");

        bias_add(&handle, m as u32, n as u32, &input, &bias, &mut output).expect("bias_add launch");
        handle.stream().synchronize().expect("sync");

        let mut got = vec![0.0f32; m * n];
        output.copy_to_host(&mut got).expect("download output");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "mismatch at {i}: device={g} reference={e}"
            );
        }
    }
}
