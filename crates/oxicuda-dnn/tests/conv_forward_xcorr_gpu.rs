//! End-to-end correctness tests for `conv_forward` 2-D valid cross-correlation.
//!
//! These tests pin down the numerical contract that `scirs2-vision`'s
//! `cuda_convolve_2d` relies on: a single-channel (`N=C=1`, `groups=1`) forward
//! convolution in the cuDNN **cross-correlation** convention (no 180° kernel
//! flip), with `pad=0, stride=1, dilation=1`, must reproduce a plain CPU
//! cross-correlation exactly (f64) / to f32 tolerance.
//!
//! Such a problem routes through the depthwise engine (because
//! `groups == in_channels == out_channels == 1`), so this file is the device
//! regression guard for that kernel. A non-square image and a non-symmetric
//! kernel are included to catch index transpose / kernel-flip regressions that
//! a square symmetric kernel would hide.
//!
//! Every device-requiring test acquires the GPU through `try_handle`, which
//! returns `None` on any host without a usable CUDA driver; the test then
//! skips (prints a notice) instead of failing. The PTX-assembly test runs
//! everywhere — it only generates and inspects code.

use std::sync::Arc;

use oxicuda_dnn::conv::conv_forward;
use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::direct::DepthwiseConv;
use oxicuda_dnn::handle::DnnHandle;
use oxicuda_dnn::types::{ConvolutionDescriptor, TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_dnn::{DnnError, DnnResult};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// CPU reference: plain 2-D valid cross-correlation (no kernel flip).
// ---------------------------------------------------------------------------

/// Computes the `(H-kh+1) x (W-kw+1)` valid cross-correlation of a single
/// `H x W` image with a `kh x kw` kernel, row-major:
///
/// `out[oh, ow] = Σ_kr Σ_ks image[oh+kr, ow+ks] * kernel[kr, ks]`.
fn cpu_xcorr(image: &[f64], h: usize, w: usize, kernel: &[f64], kh: usize, kw: usize) -> Vec<f64> {
    let out_h = h - kh + 1;
    let out_w = w - kw + 1;
    let mut out = vec![0.0f64; out_h * out_w];
    for oh in 0..out_h {
        for ow in 0..out_w {
            let mut acc = 0.0f64;
            for kr in 0..kh {
                for ks in 0..kw {
                    acc += image[(oh + kr) * w + (ow + ks)] * kernel[kr * kw + ks];
                }
            }
            out[oh * out_w + ow] = acc;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GPU acquisition helper (skips gracefully when no device is present).
// ---------------------------------------------------------------------------

/// Builds an `(Arc<Context>, DnnHandle)` or returns `None` when CUDA is
/// unavailable. The `Arc<Context>` must outlive every device buffer in use.
fn try_handle() -> Option<(Arc<Context>, DnnHandle)> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = DnnHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

/// Runs a single-channel `conv_forward::<f64>` valid cross-correlation on the
/// device and returns the `out_h * out_w` result, retrying with a workspace if
/// the engine requests one (mirroring the `scirs2-vision` call site).
fn gpu_xcorr(
    handle: &DnnHandle,
    image: &[f64],
    h: usize,
    w: usize,
    kernel: &[f64],
    kh: usize,
    kw: usize,
) -> DnnResult<Vec<f64>> {
    let out_h = h - kh + 1;
    let out_w = w - kw + 1;

    let map_cuda = |e: oxicuda_driver::CudaError| DnnError::LaunchFailed(e.to_string());

    let d_input = DeviceBuffer::from_host(image).map_err(map_cuda)?;
    let d_filter = DeviceBuffer::from_host(kernel).map_err(map_cuda)?;
    let mut d_output = DeviceBuffer::<f64>::alloc(out_h * out_w).map_err(map_cuda)?;

    let input = TensorDesc::<f64>::nchw(&d_input, 1, 1, h as u32, w as u32)?;
    let filter = TensorDesc::<f64>::nchw(&d_filter, 1, 1, kh as u32, kw as u32)?;
    let mut output = TensorDescMut::<f64>::nchw(&mut d_output, 1, 1, out_h as u32, out_w as u32)?;

    let conv = ConvolutionDescriptor::conv2d(0, 0, 1, 1, 1, 1, 1)?;

    match conv_forward::<f64>(handle, &input, &filter, &mut output, &conv, None) {
        Ok(()) => {}
        Err(DnnError::WorkspaceRequired(bytes)) => {
            let mut ws = DeviceBuffer::<u8>::alloc(bytes).map_err(map_cuda)?;
            conv_forward::<f64>(handle, &input, &filter, &mut output, &conv, Some(&mut ws))?;
        }
        Err(e) => return Err(e),
    }

    handle.stream().synchronize().map_err(map_cuda)?;

    let mut host = vec![0.0f64; out_h * out_w];
    d_output.copy_to_host(&mut host).map_err(map_cuda)?;
    Ok(host)
}

/// Shared driver: run the GPU cross-correlation and assert element-wise
/// agreement with the CPU reference within `1e-9`.
fn assert_matches_cpu(image: &[f64], h: usize, w: usize, kernel: &[f64], kh: usize, kw: usize) {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping conv_forward xcorr GPU test: no CUDA device");
        return;
    };

    let got = match gpu_xcorr(&handle, image, h, w, kernel, kh, kw) {
        Ok(v) => v,
        Err(e) => panic!("conv_forward::<f64> failed: {e}"),
    };
    let want = cpu_xcorr(image, h, w, kernel, kh, kw);
    assert_eq!(got.len(), want.len(), "output element count mismatch");

    let mut max_diff = 0.0f64;
    let mut worst = 0usize;
    for (i, (g, c)) in got.iter().zip(want.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_diff {
            max_diff = d;
            worst = i;
        }
    }
    assert!(
        max_diff < 1e-9,
        "max abs diff {max_diff} at index {worst} (gpu={}, cpu={}) exceeds 1e-9",
        got[worst],
        want[worst]
    );
}

// ---------------------------------------------------------------------------
// PTX assembly / shape test (no device required).
// ---------------------------------------------------------------------------

/// The depthwise f64 kernel must emit *real* double-precision compute (an fma
/// and a global store), not a comment-only stub, and the generated PTX is
/// written to `temp_dir()` so it can be assembled offline with
/// `ptxas -arch=sm_86`.
#[test]
fn depthwise_f64_ptx_has_real_compute() {
    let problem = ConvProblem {
        batch: 1,
        in_channels: 1,
        in_dims: vec![5, 5],
        out_channels: 1,
        filter_dims: vec![3, 3],
        padding: vec![0, 0],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F64,
        output_type: PtxType::F64,
        layout: TensorLayout::Nchw,
    };
    assert!(
        problem.is_depthwise(),
        "1x1-channel conv routes to depthwise"
    );

    let engine = DepthwiseConv::new(problem, SmVersion::Sm86).expect("depthwise engine");
    let ptx = engine.generate_ptx().expect("ptx generation");

    assert!(
        ptx.contains("fma.rn.f64"),
        "f64 depthwise kernel must accumulate with fma.rn.f64"
    );
    assert!(
        ptx.contains("st.global.f64"),
        "f64 depthwise kernel must store an f64 result"
    );
    assert!(
        ptx.contains("ld.global.f64"),
        "f64 depthwise kernel must load f64 inputs/weights"
    );

    let path = std::env::temp_dir().join("oxicuda_depthwise_3x3_f64.ptx");
    std::fs::write(&path, &ptx).expect("write ptx to temp_dir");
    eprintln!("wrote depthwise f64 PTX to {}", path.display());
}

// ---------------------------------------------------------------------------
// Device correctness tests.
// ---------------------------------------------------------------------------

/// The exact `scirs2-vision::cuda_convolve_2d` smoke case: a 5x5 image of
/// values 1..=25 and a 3x3 plus-shaped kernel. The valid interior must match
/// the CPU cross-correlation (e.g. the bottom-right output is 95).
#[test]
fn conv_forward_f64_5x5_plus_kernel() {
    let image: Vec<f64> = (1..=25).map(|v| v as f64).collect();
    let kernel = vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    assert_matches_cpu(&image, 5, 5, &kernel, 3, 3);
}

/// Non-square image with a non-symmetric 3x3 kernel: a kernel flip
/// (true convolution) or an H/W index transpose would change the result, so
/// this catches those structural bugs that a symmetric kernel hides.
#[test]
fn conv_forward_f64_nonsquare_asymmetric_kernel() {
    // 6 rows x 4 cols, distinct values.
    let h = 6usize;
    let w = 4usize;
    let image: Vec<f64> = (0..h * w).map(|i| (i as f64) * 0.5 - 3.0).collect();
    // Non-symmetric 3x3 kernel (not equal to its 180° rotation, nor its
    // transpose): pins down both flip and transpose orientation.
    let kernel = vec![1.0, 2.0, 3.0, 0.0, -1.0, 4.0, 5.0, 0.0, -2.0];
    assert_matches_cpu(&image, h, w, &kernel, 3, 3);
}

/// Non-square image with a non-square, non-symmetric 3x5 kernel: exercises
/// `filter_h != filter_w` index arithmetic on top of the asymmetry checks.
#[test]
fn conv_forward_f64_nonsquare_3x5_kernel() {
    let h = 7usize;
    let w = 9usize;
    let image: Vec<f64> = (0..h * w).map(|i| ((i * 7) % 13) as f64 - 6.0).collect();
    let kernel: Vec<f64> = (0..15).map(|i| (i as f64) * 0.25 - 1.0).collect(); // 3x5
    assert_matches_cpu(&image, h, w, &kernel, 3, 5);
}
