//! GPU numerical tests for the S2 stub-check implementations.
//!
//! Covers:
//! * **im2col expansion kernel** — the kernel emitted by
//!   [`Im2colGemmConv::generate_im2col_ptx`] must fill the column-matrix
//!   workspace exactly as a CPU im2col reference does. (The downstream
//!   `launch_gemm` BLAS dispatch is verified for orientation by the
//!   `im2col_gemm` unit tests; an end-to-end GPU conv additionally depends
//!   on `oxicuda_blas::gemm`.)
//! * **FP8 E4M3 quantize / dequantize** — a real E4M3 round-trip must stay
//!   within E4M3 precision, including subnormals and saturation.
//!
//! Following the crate convention for device-requiring code (see
//! `oxicuda-blas/tests/trsm_trmm_gpu.rs`), every test acquires the GPU
//! through a `try_*` helper that returns `None` on any platform without a
//! usable CUDA driver — the test then skips instead of failing.

use std::sync::Arc;

use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::im2col_gemm::Im2colGemmConv;
use oxicuda_dnn::handle::DnnHandle;
use oxicuda_dnn::quantize::fp8_quantize::{dequantize_from_fp8, quantize_to_fp8};
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_driver::{Context, Device, Module};
use oxicuda_launch::{Kernel, LaunchParams, grid_size_for};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// GPU acquisition helper
// ---------------------------------------------------------------------------

/// Attempts to initialise the driver and build a [`DnnHandle`].
///
/// Returns `None` on any platform without a working CUDA driver so the
/// caller can skip the test gracefully. The returned [`Arc<Context>`] must
/// outlive every device buffer in use.
fn try_handle() -> Option<(Arc<Context>, DnnHandle)> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = DnnHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

/// Deterministic pseudo-random value in roughly `[-1, 1)`.
fn pseudo(i: usize, salt: u64) -> f32 {
    let mut x = (i as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(salt);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    ((x & 0xff_ffff) as f32 / 0x80_0000 as f32) - 1.0
}

// ---------------------------------------------------------------------------
// CPU reference: direct 2-D convolution (NCHW, batch 1, zero padding)
// ---------------------------------------------------------------------------

struct ConvSpec {
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    r: usize,
    s: usize,
    pad: usize,
    stride: usize,
}

impl ConvSpec {
    fn out_h(&self) -> usize {
        (self.in_h + 2 * self.pad - self.r) / self.stride + 1
    }
    fn out_w(&self) -> usize {
        (self.in_w + 2 * self.pad - self.s) / self.stride + 1
    }
}

/// Builds a [`ConvProblem`] (NCHW, batch 1) from a [`ConvSpec`].
fn problem_from_spec(spec: &ConvSpec) -> ConvProblem {
    ConvProblem {
        batch: 1,
        in_channels: spec.in_c as u32,
        in_dims: vec![spec.in_h as u32, spec.in_w as u32],
        out_channels: spec.out_c as u32,
        filter_dims: vec![spec.r as u32, spec.s as u32],
        padding: vec![spec.pad as u32, spec.pad as u32],
        stride: vec![spec.stride as u32, spec.stride as u32],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    }
}

/// CPU im2col reference producing the `(C*R*S) x M` column matrix in
/// row-major order — exactly the layout the GPU im2col kernel writes and
/// the GEMM phase consumes as operand `B`. `batch == 1`.
fn cpu_im2col(input: &[f32], spec: &ConvSpec) -> Vec<f32> {
    let (out_h, out_w) = (spec.out_h(), spec.out_w());
    let k_dim = spec.in_c * spec.r * spec.s;
    let m = out_h * out_w;
    let mut col = vec![0.0f32; k_dim * m];
    for c in 0..spec.in_c {
        for kr in 0..spec.r {
            for ks in 0..spec.s {
                let k_idx = (c * spec.r + kr) * spec.s + ks;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let ih = (oh * spec.stride + kr) as isize - spec.pad as isize;
                        let iw = (ow * spec.stride + ks) as isize - spec.pad as isize;
                        let m_idx = oh * out_w + ow;
                        if ih >= 0
                            && iw >= 0
                            && (ih as usize) < spec.in_h
                            && (iw as usize) < spec.in_w
                        {
                            let in_idx = (c * spec.in_h + ih as usize) * spec.in_w + iw as usize;
                            col[k_idx * m + m_idx] = input[in_idx];
                        }
                    }
                }
            }
        }
    }
    col
}

/// Launches the im2col expansion kernel on the GPU and checks the resulting
/// column-matrix workspace against the CPU `cpu_im2col` reference. This
/// directly exercises the kernel emitted by `generate_im2col_ptx` — the
/// first half of im2col convolution.
fn run_im2col_expand_case(spec: ConvSpec) {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping im2col GPU test: no CUDA device");
        return;
    };

    let (out_h, out_w) = (spec.out_h(), spec.out_w());
    let in_len = spec.in_c * spec.in_h * spec.in_w;
    let m = out_h * out_w; // batch == 1
    let k_dim = spec.in_c * spec.r * spec.s;
    let col_len = k_dim * m;

    let host_input: Vec<f32> = (0..in_len).map(|i| pseudo(i, 0x1111)).collect();

    let mut d_input = DeviceBuffer::<f32>::alloc(in_len).expect("alloc input");
    // The workspace is a byte buffer reinterpreted as f32 by the kernel.
    let d_col = DeviceBuffer::<f32>::alloc(col_len).expect("alloc col");
    d_input.copy_from_host(&host_input).expect("h2d input");

    let problem = problem_from_spec(&spec);
    let engine = Im2colGemmConv::new(problem, handle.sm_version());

    // Build and launch the im2col expansion kernel directly.
    let ptx = engine.generate_im2col_ptx().expect("im2col ptx");
    let module = Arc::new(Module::from_ptx(&ptx).expect("module from ptx"));
    let kernel = Kernel::from_module(module, &engine.im2col_kernel_name()).expect("kernel");

    let total_elements = (k_dim * m) as u32;
    let block = 256u32;
    let grid = grid_size_for(total_elements, block);
    let params = LaunchParams::new(grid, block);

    let args = (
        d_input.as_device_ptr(),
        d_col.as_device_ptr(),
        1u32,               // batch_size
        spec.in_c as u32,   // in_channels
        spec.in_h as u32,   // in_h
        spec.in_w as u32,   // in_w
        spec.r as u32,      // filter_h
        spec.s as u32,      // filter_w
        out_h as u32,       // out_h
        out_w as u32,       // out_w
        spec.pad as u32,    // pad_h
        spec.pad as u32,    // pad_w
        spec.stride as u32, // stride_h
        spec.stride as u32, // stride_w
        1u32,               // dilation_h
        1u32,               // dilation_w
        total_elements,     // total_elements
    );
    kernel
        .launch(&params, handle.stream(), &args)
        .expect("im2col kernel launch");
    handle.stream().synchronize().expect("stream sync");

    let mut gpu_col = vec![0.0f32; col_len];
    d_col.copy_to_host(&mut gpu_col).expect("d2h col");

    let cpu_col = cpu_im2col(&host_input, &spec);
    assert_eq!(gpu_col.len(), cpu_col.len());

    let mut max_err = 0.0f32;
    for (g, c) in gpu_col.iter().zip(cpu_col.iter()) {
        max_err = max_err.max((g - c).abs());
    }
    assert!(
        max_err < 1e-5,
        "im2col expansion differs from CPU reference: max_err = {max_err}"
    );
}

#[test]
fn im2col_expand_3x3_same_padding() {
    run_im2col_expand_case(ConvSpec {
        in_c: 3,
        in_h: 12,
        in_w: 12,
        out_c: 8,
        r: 3,
        s: 3,
        pad: 1,
        stride: 1,
    });
}

#[test]
fn im2col_expand_1x1() {
    run_im2col_expand_case(ConvSpec {
        in_c: 6,
        in_h: 10,
        in_w: 10,
        out_c: 5,
        r: 1,
        s: 1,
        pad: 0,
        stride: 1,
    });
}

#[test]
fn im2col_expand_5x5_no_padding() {
    run_im2col_expand_case(ConvSpec {
        in_c: 4,
        in_h: 16,
        in_w: 14,
        out_c: 7,
        r: 5,
        s: 5,
        pad: 0,
        stride: 1,
    });
}

#[test]
fn im2col_expand_strided() {
    run_im2col_expand_case(ConvSpec {
        in_c: 3,
        in_h: 15,
        in_w: 15,
        out_c: 6,
        r: 3,
        s: 3,
        pad: 1,
        stride: 2,
    });
}

// ---------------------------------------------------------------------------
// FP8 E4M3 quantize / dequantize round-trip
// ---------------------------------------------------------------------------

/// Quantizes a host vector to FP8 E4M3 on the GPU, dequantizes it back, and
/// returns the recovered values.
fn fp8_round_trip(handle: &DnnHandle, host: &[f32]) -> Vec<f32> {
    let n = host.len();
    let mut d_in = DeviceBuffer::<f32>::alloc(n).expect("alloc d_in");
    let mut d_q = DeviceBuffer::<u8>::alloc(n).expect("alloc d_q");
    let mut d_scale = DeviceBuffer::<f32>::alloc(1).expect("alloc d_scale");
    let d_out = DeviceBuffer::<f32>::alloc(n).expect("alloc d_out");
    d_in.copy_from_host(host).expect("h2d d_in");

    let input = TensorDesc::<f32>::from_raw(
        d_in.as_device_ptr(),
        vec![n as u32],
        vec![1],
        TensorLayout::Nchw,
    )
    .expect("input desc");
    quantize_to_fp8(handle, &input, &mut d_q, &mut d_scale).expect("quantize_to_fp8");

    let mut output = TensorDescMut::<f32>::from_raw(
        d_out.as_device_ptr(),
        vec![n as u32],
        vec![1],
        TensorLayout::Nchw,
    )
    .expect("output desc");
    dequantize_from_fp8(handle, &d_q, &d_scale, &mut output, n as u32)
        .expect("dequantize_from_fp8");
    handle.stream().synchronize().expect("stream sync");

    let mut recovered = vec![0.0f32; n];
    d_out.copy_to_host(&mut recovered).expect("d2h d_out");
    recovered
}

/// A real E4M3 round-trip must keep every value within E4M3 precision. The
/// per-tensor scale is `absmax / 448`, so the quantization step for a value
/// near `absmax` is bounded by `scale * (largest E4M3 ulp / 448)`.
#[test]
fn fp8_e4m3_round_trip_within_precision() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping FP8 GPU test: no CUDA device");
        return;
    };

    let host: Vec<f32> = (0..256).map(|i| pseudo(i, 0xF8F8) * 10.0).collect();
    let recovered = fp8_round_trip(&handle, &host);

    let absmax = host.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = (absmax / 448.0).max(1e-12);

    for (orig, deq) in host.iter().zip(recovered.iter()) {
        // E4M3 has a 3-bit mantissa: the relative step is at most 1/8. The
        // absolute error is bounded by half a mantissa step at the value's
        // magnitude, plus the scale granularity.
        let mag = orig.abs().max(scale);
        let tol = mag / 8.0 + scale + 1e-3;
        assert!(
            (orig - deq).abs() <= tol,
            "E4M3 GPU round-trip {orig} -> {deq}, error {} > tol {tol}",
            (orig - deq).abs()
        );
    }
}

/// E4M3 round-trip must preserve sign across positive and negative inputs.
#[test]
fn fp8_e4m3_round_trip_preserves_sign() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping FP8 GPU test: no CUDA device");
        return;
    };

    let host: Vec<f32> = vec![
        5.0, -5.0, 1.0, -1.0, 0.5, -0.5, 12.0, -12.0, 0.125, -0.125, 30.0, -30.0,
    ];
    let recovered = fp8_round_trip(&handle, &host);

    for (orig, deq) in host.iter().zip(recovered.iter()) {
        if *orig > 0.0 {
            assert!(*deq > 0.0, "positive {orig} became {deq}");
        } else {
            assert!(*deq < 0.0, "negative {orig} became {deq}");
        }
    }
}

/// E4M3 saturation: magnitudes far above the format range clamp to ±448
/// (scaled). With every value already near `absmax`, the largest entries
/// must dequantize close to the original absmax, not overflow.
#[test]
fn fp8_e4m3_saturation_round_trip() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping FP8 GPU test: no CUDA device");
        return;
    };

    // A spike far above the rest: the scale is set by the spike, so the
    // spike maps to E4M3 max (0x7E) and recovers near its own magnitude.
    let mut host: Vec<f32> = (0..64).map(|i| pseudo(i, 0x5A7) * 0.5).collect();
    host[10] = 1000.0;
    host[40] = -1000.0;
    let recovered = fp8_round_trip(&handle, &host);

    // The spike recovers within one E4M3 step of 1000 (scale = 1000/448).
    let step = (1000.0f32 / 448.0) * (448.0 / 8.0);
    assert!(
        (recovered[10] - 1000.0).abs() <= step + 1.0,
        "positive saturation spike: {} vs 1000",
        recovered[10]
    );
    assert!(
        (recovered[40] + 1000.0).abs() <= step + 1.0,
        "negative saturation spike: {} vs -1000",
        recovered[40]
    );
}

/// E4M3 subnormals: very small magnitudes relative to `absmax` exercise the
/// subnormal grid (`exponent field 0`). They must round-trip to a small
/// value (subnormal or flushed to zero), never to garbage.
#[test]
fn fp8_e4m3_subnormal_round_trip() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping FP8 GPU test: no CUDA device");
        return;
    };

    // absmax = 1.0, so scale = 1/448. Values around scale * 2^-6 land in the
    // E4M3 subnormal range; values far below scale * 2^-10 flush to zero.
    let scale = 1.0f32 / 448.0;
    let mut host: Vec<f32> = vec![1.0, -1.0];
    for k in 1..=8 {
        host.push(scale * (k as f32) * 2.0f32.powi(-6));
    }
    host.push(scale * 2.0f32.powi(-12)); // below the subnormal grid -> ~0

    let recovered = fp8_round_trip(&handle, &host);

    // The tiny entry must round-trip to (near) zero.
    let tiny = *recovered.last().expect("non-empty");
    assert!(
        tiny.abs() < scale * 2.0f32.powi(-8),
        "sub-grid value should flush toward zero, got {tiny}"
    );
    // The subnormal-range entries must be small and finite.
    for &v in &recovered[2..recovered.len() - 1] {
        assert!(
            v.is_finite(),
            "subnormal round-trip produced non-finite {v}"
        );
        assert!(
            v.abs() <= scale * 2.0f32.powi(-3),
            "subnormal-range value too large after round-trip: {v}"
        );
    }
}

/// Sanity guard for the SM version used by the handle: the FP8 kernels are
/// generated for the device's architecture, which must be a known SM.
#[test]
fn handle_reports_known_sm_version() {
    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping SM version test: no CUDA device");
        return;
    };
    let sm = handle.sm_version();
    assert!(
        matches!(
            sm,
            SmVersion::Sm75
                | SmVersion::Sm80
                | SmVersion::Sm86
                | SmVersion::Sm89
                | SmVersion::Sm90
                | SmVersion::Sm90a
                | SmVersion::Sm100
                | SmVersion::Sm120
        ),
        "unexpected SM version {sm:?}"
    );
}
