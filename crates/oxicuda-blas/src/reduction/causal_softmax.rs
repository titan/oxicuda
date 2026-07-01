//! Numerically stable *causal* (masked) row-wise softmax on device buffers.
//!
//! This is the masked counterpart of [`crate::reduction::softmax`]. For an
//! autoregressive attention score matrix stored row-major as `rows x cols`
//! (row = query position, column = key position), it computes, for each
//! output row `i`:
//!
//! ```text
//! live = { j : 0 <= j <= i  and  j < cols }      // causal mask
//! m    = max_{j in live} input[i, j]
//! out[i, j] = exp(input[i, j] - m) / sum_{k in live} exp(input[i, k] - m)   (j in live)
//! out[i, j] = 0                                                              (j not in live)
//! ```
//!
//! i.e. query position `i` only attends to key positions `j <= i`; all
//! "future" positions `j > i` are excluded from the row max and the
//! exponential sum and are written as `0`.
//!
//! No scaling (e.g. `1/sqrt(d_k)`) is applied here — the caller is expected
//! to have already scaled the scores, exactly as in the reference
//! `trustformers` `softmax_causal_f32` kernel this mirrors.
//!
//! ## Algorithm and parallelization
//!
//! The kernel assigns **one thread per row** and walks the row sequentially
//! through the standard numerically-stable three-pass softmax (row max,
//! exponential sum, normalize), restricted to the unmasked prefix. This is a
//! 1:1 port of the reference CUDA kernel, chosen over a warp/block reduction
//! because the triangular mask would otherwise leave most lanes idle.
//!
//! Two degenerate-row guards match the reference kernel:
//! - if the masked max is `< -1e38`, the row is written as `[1, 0, 0, ...]`;
//! - if the masked exponential sum is `< 1e-10`, likewise `[1, 0, 0, ...]`.

use std::sync::Arc;

use oxicuda_driver::Module;
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;
use oxicuda_ptx::templates::causal_softmax::CausalSoftmaxTemplate;

use crate::error::{BlasError, BlasResult};
use crate::handle::BlasHandle;
use crate::types::GpuFloat;

/// Threads per block for the one-thread-per-row causal softmax launch.
///
/// 256 is a standard occupancy-friendly block size; the grid is sized to
/// cover `rows` threads.
const CAUSAL_SOFTMAX_BLOCK: u32 = 256;

/// Builds the causal-softmax kernel from the PTX template.
fn build_causal_softmax_kernel(
    handle: &BlasHandle,
    ptx_type: PtxType,
) -> BlasResult<(Kernel, String)> {
    let template = CausalSoftmaxTemplate {
        precision: ptx_type,
        target: handle.sm_version(),
    };
    let kernel_name = template.kernel_name();
    let ptx_source = template
        .generate()
        .map_err(|e| BlasError::PtxGeneration(format!("causal softmax: {e}")))?;
    let module =
        Arc::new(Module::from_ptx(&ptx_source).map_err(|e| {
            BlasError::LaunchFailed(format!("module load for causal softmax: {e}"))
        })?);
    let kernel = Kernel::from_module(module, &kernel_name)
        .map_err(|e| BlasError::LaunchFailed(format!("kernel lookup for {kernel_name}: {e}")))?;
    Ok((kernel, kernel_name))
}

/// Computes a row-wise **causal (masked) softmax** over a 2-D matrix stored
/// in row-major order.
///
/// This is the masked sibling of [`softmax`](crate::reduction::softmax) and a
/// 1:1 equivalent of the `trustformers` `softmax_causal_f32` GPU op. For each
/// row `r` (a query position), only columns `j <= r` (key positions up to and
/// including `r`) contribute; columns `j > r` are masked:
///
/// ```text
/// live    = min(r + 1, cols)                      // number of unmasked columns
/// m       = max(input[r, 0..live])
/// out[r, j] = exp(input[r, j] - m) / sum_k(exp(input[r, k] - m))   for j < live
/// out[r, j] = 0                                                    for j >= live
/// ```
///
/// The masked entries are excluded from both the row maximum and the
/// exponential sum (not merely zeroed afterwards), and the output is
/// numerically stable (the per-row max over unmasked entries is subtracted
/// before exponentiation).
///
/// For the square attention case the reference op targets, pass
/// `rows == cols == seq_len`; the `j <= r` rule then reproduces the standard
/// lower-triangular (causal) attention mask exactly. Rectangular shapes are
/// also accepted: when `r >= cols` every column is unmasked (the whole row is
/// "live"), which is the natural generalization.
///
/// No score scaling is applied; scale the input beforehand if required.
///
/// # Parallelization
///
/// One thread processes one row sequentially. The launch uses
/// `ceil(rows / 256)` blocks of 256 threads.
///
/// # Arguments
///
/// * `handle` -- BLAS handle bound to a CUDA context and stream.
/// * `rows` -- number of rows (query positions).
/// * `cols` -- number of columns (key positions).
/// * `input` -- device buffer containing the input matrix in row-major
///   layout, at least `rows * cols` elements.
/// * `output` -- device buffer for the result matrix, same layout, at least
///   `rows * cols` elements.
///
/// # Type support
///
/// Supports `f32` and `f64`. Half precisions (`f16`/`bf16`) are rejected with
/// [`BlasError::UnsupportedOperation`] because the stable softmax relies on
/// `ex2.approx`/`rcp.approx`, which are not defined for half types.
///
/// # Errors
///
/// Returns [`BlasError::BufferTooSmall`] if buffers are too small,
/// [`BlasError::InvalidDimension`] if `rows` or `cols` is zero,
/// [`BlasError::UnsupportedOperation`] if `T` is not `f32`/`f64`, or
/// [`BlasError::LaunchFailed`] / [`BlasError::PtxGeneration`] if kernel
/// construction or launch fails.
pub fn causal_softmax<T: GpuFloat>(
    handle: &BlasHandle,
    rows: u32,
    cols: u32,
    input: &DeviceBuffer<T>,
    output: &mut DeviceBuffer<T>,
) -> BlasResult<()> {
    if rows == 0 || cols == 0 {
        return Err(BlasError::InvalidDimension(
            "causal_softmax requires rows > 0 and cols > 0".to_string(),
        ));
    }

    if !matches!(T::PTX_TYPE, PtxType::F32 | PtxType::F64) {
        return Err(BlasError::UnsupportedOperation(format!(
            "causal_softmax supports only f32/f64, got {}",
            T::PTX_TYPE.as_ptx_str()
        )));
    }

    let total_elements = (rows as usize).checked_mul(cols as usize).ok_or_else(|| {
        BlasError::InvalidDimension(format!(
            "causal_softmax dimension overflow: rows={rows} * cols={cols}"
        ))
    })?;
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

    let (kernel, _) = build_causal_softmax_kernel(handle, T::PTX_TYPE)?;

    // One thread per row, 256 threads per block.
    let grid = grid_size_for(rows, CAUSAL_SOFTMAX_BLOCK);
    let params = LaunchParams::new(grid, CAUSAL_SOFTMAX_BLOCK);

    // Kernel signature: (input_ptr, output_ptr, rows, cols).
    let args = (input.as_device_ptr(), output.as_device_ptr(), rows, cols);

    kernel
        .launch(&params, handle.stream(), &args)
        .map_err(|e| BlasError::LaunchFailed(format!("causal_softmax: {e}")))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_ptx::arch::SmVersion;

    // ---- PTX generation (host-only, no GPU) -----------------------------

    #[test]
    fn ptx_template_generates_causal_softmax_f32() {
        let template = CausalSoftmaxTemplate {
            precision: PtxType::F32,
            target: SmVersion::Sm80,
        };
        let ptx = template
            .generate()
            .expect("causal softmax PTX should generate");
        assert!(ptx.contains("causal_softmax_f32"));
        assert!(ptx.contains("ex2.approx.f32"));
    }

    #[test]
    fn launch_grid_covers_all_rows() {
        // 300 rows, 256 threads/block => 2 blocks.
        assert_eq!(grid_size_for(300, CAUSAL_SOFTMAX_BLOCK), 2);
        // Exactly one full block.
        assert_eq!(grid_size_for(256, CAUSAL_SOFTMAX_BLOCK), 1);
        // A single row still needs one block.
        assert_eq!(grid_size_for(1, CAUSAL_SOFTMAX_BLOCK), 1);
    }

    // ---- CPU reference for the intended math ----------------------------

    /// Naive triple-loop causal softmax over a row-major `rows x cols`
    /// matrix. This is the ground truth the device kernel must reproduce; it
    /// encodes the exact masking + stability rule the GPU kernel implements.
    fn causal_softmax_reference(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let base = r * cols;
            // Unmasked columns: j <= r and j < cols.
            let live = (r + 1).min(cols);

            // Pass 1: masked max.
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..live {
                max_val = max_val.max(input[base + j]);
            }

            // Degenerate: all -inf => one-hot.
            if max_val < -1e38 {
                for j in 0..cols {
                    out[base + j] = if j == 0 { 1.0 } else { 0.0 };
                }
                continue;
            }

            // Pass 2: masked exp sum.
            let mut sum = 0.0f32;
            for j in 0..live {
                sum += (input[base + j] - max_val).exp();
            }

            // Degenerate: underflowed sum => one-hot.
            if sum < 1e-10 {
                for j in 0..cols {
                    out[base + j] = if j == 0 { 1.0 } else { 0.0 };
                }
                continue;
            }

            // Pass 3: normalize live columns, zero masked columns.
            for j in 0..cols {
                if j < live {
                    out[base + j] = (input[base + j] - max_val).exp() / sum;
                } else {
                    out[base + j] = 0.0;
                }
            }
        }
        out
    }

    #[test]
    fn reference_masks_strictly_upper_triangle() {
        // 4x4 square attention scores. Future columns must be exactly zero.
        let rows = 4;
        let cols = 4;
        let input: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let out = causal_softmax_reference(&input, rows, cols);

        for r in 0..rows {
            for j in 0..cols {
                let v = out[r * cols + j];
                if j > r {
                    assert_eq!(v, 0.0, "masked entry ({r},{j}) must be 0, got {v}");
                } else {
                    assert!(v > 0.0, "live entry ({r},{j}) must be > 0, got {v}");
                }
            }
        }
    }

    #[test]
    fn reference_rows_sum_to_one() {
        let rows = 5;
        let cols = 5;
        // Mix of magnitudes to exercise the stable subtraction.
        let input: Vec<f32> = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, //
            -2.0, 5.0, -1.0, 0.5, 9.0, //
            10.0, 10.0, 10.0, 10.0, 10.0, //
            -5.0, -4.0, -3.0, -2.0, -1.0, //
            100.0, 99.0, 98.0, 97.0, 96.0, // large values: stability matters
        ];
        let out = causal_softmax_reference(&input, rows, cols);
        for r in 0..rows {
            let row_sum: f32 = (0..cols).map(|j| out[r * cols + j]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "row {r} should sum to 1, got {row_sum}"
            );
        }
    }

    #[test]
    fn reference_first_row_is_one_hot() {
        // Row 0 has exactly one live column (j == 0), so it must be [1, 0, ...].
        let rows = 3;
        let cols = 3;
        let input = vec![7.0, 1.0, 2.0, 0.0, 3.0, 4.0, -1.0, -2.0, 5.0];
        let out = causal_softmax_reference(&input, rows, cols);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert_eq!(out[1], 0.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn reference_matches_hand_computed_diagonal_row() {
        // Row 1 of a 3x3: live columns {0, 1}. With scores [a, b], the result
        // is the plain 2-element softmax of those two, and column 2 is masked.
        let rows = 3;
        let cols = 3;
        let a = 1.5f32;
        let b = 0.5f32;
        let input = vec![0.0, 0.0, 0.0, a, b, 99.0, 0.0, 0.0, 0.0];
        let out = causal_softmax_reference(&input, rows, cols);

        let m = a.max(b);
        let ea = (a - m).exp();
        let eb = (b - m).exp();
        let s = ea + eb;
        let expect0 = ea / s;
        let expect1 = eb / s;

        assert!((out[3] - expect0).abs() < 1e-6, "got {}", out[3]);
        assert!((out[4] - expect1).abs() < 1e-6, "got {}", out[4]);
        // Column 2 is masked even though its score (99.0) is the largest.
        assert_eq!(out[5], 0.0);
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

        // Try to bring up a CUDA device + context. If there is no GPU/driver
        // (the expected case on the macOS build host), skip the device
        // assertion -- the CPU reference tests above already pin the math.
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

        let seq_len: u32 = 6;
        let rows = seq_len as usize;
        let cols = seq_len as usize;
        let host_in: Vec<f32> = (0..(rows * cols)).map(|i| ((i % 7) as f32) - 3.0).collect();
        let expected = causal_softmax_reference(&host_in, rows, cols);

        let input = DeviceBuffer::<f32>::from_host(&host_in).expect("upload input");
        let mut output = DeviceBuffer::<f32>::zeroed(rows * cols).expect("alloc output");

        causal_softmax(&handle, seq_len, seq_len, &input, &mut output)
            .expect("causal_softmax launch");
        handle.stream().synchronize().expect("sync");

        let mut got = vec![0.0f32; rows * cols];
        output.copy_to_host(&mut got).expect("download output");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "mismatch at {i}: device={g} reference={e}"
            );
        }
    }
}
