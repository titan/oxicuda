//! End-to-end correctness tests for the **standard multi-channel** forward
//! convolution kernels: the general implicit-GEMM engine
//! ([`ImplicitGemmConv`]) and the 1x1 engine ([`Conv1x1`]).
//!
//! These pin down the numerical contract for `C_in > 1, groups = 1`
//! convolutions in the cuDNN **cross-correlation** convention (no 180° kernel
//! flip). Each device test compares against a hand-written CPU
//! cross-correlation reference to `< 1e-9` (f64). A non-square image with a
//! non-square, non-symmetric kernel catches kernel-flip / H-W-transpose /
//! filter-transpose regressions that a square symmetric kernel would hide.
//!
//! The engines are exercised directly (not via the `conv_forward` heuristic)
//! so the test deterministically covers the two kernels that were previously
//! comment-only stubs, regardless of algorithm-selection routing.
//!
//! Device tests acquire the GPU via `try_handle` and skip (print a notice) when
//! no usable CUDA driver is present. The PTX-assembly tests run everywhere
//! `ptxas` is on `PATH` and otherwise skip gracefully.

use std::process::Command;
use std::sync::Arc;

use oxicuda_dnn::conv::descriptor::ConvProblem;
use oxicuda_dnn::conv::fprop::direct::Conv1x1;
use oxicuda_dnn::conv::fprop::implicit_gemm::ImplicitGemmConv;
use oxicuda_dnn::handle::DnnHandle;
use oxicuda_dnn::types::{TensorDesc, TensorDescMut, TensorLayout};
use oxicuda_dnn::{DnnError, DnnResult};
use oxicuda_driver::{Context, Device};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

// ---------------------------------------------------------------------------
// Problem geometry.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct ConvGeom {
    n: usize,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    pad_h: usize,
    pad_w: usize,
    stride_h: usize,
    stride_w: usize,
    dil_h: usize,
    dil_w: usize,
    groups: usize,
}

impl ConvGeom {
    fn out_h(&self) -> usize {
        (self.h + 2 * self.pad_h - self.dil_h * (self.kh - 1) - 1) / self.stride_h + 1
    }
    fn out_w(&self) -> usize {
        (self.w + 2 * self.pad_w - self.dil_w * (self.kw - 1) - 1) / self.stride_w + 1
    }
    fn icpg(&self) -> usize {
        self.c_in / self.groups
    }
    fn ocpg(&self) -> usize {
        self.c_out / self.groups
    }
}

// ---------------------------------------------------------------------------
// CPU references (cross-correlation, no kernel flip).
// ---------------------------------------------------------------------------

/// NCHW reference: input `[N,C,H,W]`, filter `[K, C/g, R, S]`, output `[N,K,P,Q]`.
fn cpu_conv_nchw(g: &ConvGeom, input: &[f64], filter: &[f64]) -> Vec<f64> {
    let (ph, pw) = (g.out_h(), g.out_w());
    let icpg = g.icpg();
    let mut out = vec![0.0f64; g.n * g.c_out * ph * pw];
    for nn in 0..g.n {
        for k in 0..g.c_out {
            let group = k / g.ocpg();
            let cg_start = group * icpg;
            for oh in 0..ph {
                for ow in 0..pw {
                    let mut acc = 0.0f64;
                    for cg in 0..icpg {
                        let c_in = cg_start + cg;
                        for r in 0..g.kh {
                            let ih = oh * g.stride_h + r * g.dil_h;
                            if ih < g.pad_h || ih - g.pad_h >= g.h {
                                continue;
                            }
                            let ih = ih - g.pad_h;
                            for s in 0..g.kw {
                                let iw = ow * g.stride_w + s * g.dil_w;
                                if iw < g.pad_w || iw - g.pad_w >= g.w {
                                    continue;
                                }
                                let iw = iw - g.pad_w;
                                let in_idx = ((nn * g.c_in + c_in) * g.h + ih) * g.w + iw;
                                let f_idx = ((k * icpg + cg) * g.kh + r) * g.kw + s;
                                acc += input[in_idx] * filter[f_idx];
                            }
                        }
                    }
                    out[((nn * g.c_out + k) * ph + oh) * pw + ow] = acc;
                }
            }
        }
    }
    out
}

/// NHWC reference: input `[N,H,W,C]`, filter `[K, R, S, C/g]`, output `[N,P,Q,K]`.
fn cpu_conv_nhwc(g: &ConvGeom, input: &[f64], filter: &[f64]) -> Vec<f64> {
    let (ph, pw) = (g.out_h(), g.out_w());
    let icpg = g.icpg();
    let mut out = vec![0.0f64; g.n * ph * pw * g.c_out];
    for nn in 0..g.n {
        for oh in 0..ph {
            for ow in 0..pw {
                for k in 0..g.c_out {
                    let group = k / g.ocpg();
                    let cg_start = group * icpg;
                    let mut acc = 0.0f64;
                    for cg in 0..icpg {
                        let c_in = cg_start + cg;
                        for r in 0..g.kh {
                            let ih = oh * g.stride_h + r * g.dil_h;
                            if ih < g.pad_h || ih - g.pad_h >= g.h {
                                continue;
                            }
                            let ih = ih - g.pad_h;
                            for s in 0..g.kw {
                                let iw = ow * g.stride_w + s * g.dil_w;
                                if iw < g.pad_w || iw - g.pad_w >= g.w {
                                    continue;
                                }
                                let iw = iw - g.pad_w;
                                let in_idx = ((nn * g.h + ih) * g.w + iw) * g.c_in + c_in;
                                let f_idx = ((k * g.kh + r) * g.kw + s) * icpg + cg;
                                acc += input[in_idx] * filter[f_idx];
                            }
                        }
                    }
                    out[((nn * ph + oh) * pw + ow) * g.c_out + k] = acc;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Synthetic data.
// ---------------------------------------------------------------------------

fn ramp(len: usize, scale: f64, bias: f64) -> Vec<f64> {
    (0..len).map(|i| (i as f64) * scale + bias).collect()
}

// ---------------------------------------------------------------------------
// GPU acquisition.
// ---------------------------------------------------------------------------

fn try_handle() -> Option<(Arc<Context>, DnnHandle)> {
    oxicuda_driver::init().ok()?;
    let device = Device::get(0).ok()?;
    let ctx = Arc::new(Context::new(&device).ok()?);
    let handle = DnnHandle::new(&ctx).ok()?;
    Some((ctx, handle))
}

fn problem(g: &ConvGeom, layout: TensorLayout) -> ConvProblem {
    ConvProblem {
        batch: g.n as u32,
        in_channels: g.c_in as u32,
        in_dims: vec![g.h as u32, g.w as u32],
        out_channels: g.c_out as u32,
        filter_dims: vec![g.kh as u32, g.kw as u32],
        padding: vec![g.pad_h as u32, g.pad_w as u32],
        stride: vec![g.stride_h as u32, g.stride_w as u32],
        dilation: vec![g.dil_h as u32, g.dil_w as u32],
        groups: g.groups as u32,
        input_type: PtxType::F64,
        output_type: PtxType::F64,
        layout,
    }
}

/// Engine selector for the shared GPU driver.
enum Engine {
    ImplicitGemm,
    Conv1x1,
}

/// Runs the requested engine on device and returns the output buffer (in the
/// engine's native memory order for the chosen layout).
fn run_gpu(
    handle: &DnnHandle,
    g: &ConvGeom,
    layout: TensorLayout,
    engine: Engine,
    input: &[f64],
    filter: &[f64],
) -> DnnResult<Vec<f64>> {
    let (ph, pw) = (g.out_h(), g.out_w());
    let map_cuda = |e: oxicuda_driver::CudaError| DnnError::LaunchFailed(e.to_string());

    let d_input = DeviceBuffer::from_host(input).map_err(map_cuda)?;
    let d_filter = DeviceBuffer::from_host(filter).map_err(map_cuda)?;
    let out_len = g.n * g.c_out * ph * pw;
    let mut d_output = DeviceBuffer::<f64>::alloc(out_len).map_err(map_cuda)?;

    let make_in = |buf: &DeviceBuffer<f64>, c: usize, hh: usize, ww: usize| match layout {
        TensorLayout::Nhwc => {
            TensorDesc::<f64>::nhwc(buf, g.n as u32, c as u32, hh as u32, ww as u32)
        }
        _ => TensorDesc::<f64>::nchw(buf, g.n as u32, c as u32, hh as u32, ww as u32),
    };

    // Filter descriptor: [K, C/g, R, S] (the descriptor stores K and the
    // spatial extent; its layout flag drives the in-kernel filter addressing).
    let filter_desc = match layout {
        TensorLayout::Nhwc => TensorDesc::<f64>::nhwc(
            &d_filter,
            g.c_out as u32,
            g.icpg() as u32,
            g.kh as u32,
            g.kw as u32,
        )?,
        _ => TensorDesc::<f64>::nchw(
            &d_filter,
            g.c_out as u32,
            g.icpg() as u32,
            g.kh as u32,
            g.kw as u32,
        )?,
    };

    let input_desc = make_in(&d_input, g.c_in, g.h, g.w)?;
    let mut output_desc = match layout {
        TensorLayout::Nhwc => TensorDescMut::<f64>::nhwc(
            &mut d_output,
            g.n as u32,
            g.c_out as u32,
            ph as u32,
            pw as u32,
        )?,
        _ => TensorDescMut::<f64>::nchw(
            &mut d_output,
            g.n as u32,
            g.c_out as u32,
            ph as u32,
            pw as u32,
        )?,
    };

    let prob = problem(g, layout);
    match engine {
        Engine::ImplicitGemm => {
            let eng = ImplicitGemmConv::new(prob, handle.sm_version());
            eng.execute(handle, &input_desc, &filter_desc, None, &mut output_desc)?;
        }
        Engine::Conv1x1 => {
            let eng = Conv1x1::new(prob, handle.sm_version())?;
            eng.execute(handle, &input_desc, &filter_desc, &mut output_desc)?;
        }
    }

    handle.stream().synchronize().map_err(map_cuda)?;
    let mut host = vec![0.0f64; out_len];
    d_output.copy_to_host(&mut host).map_err(map_cuda)?;
    Ok(host)
}

fn assert_close(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "output element count mismatch");
    let mut max_diff = 0.0f64;
    let mut worst = 0usize;
    for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let d = (a - b).abs();
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
// Device correctness tests.
// ---------------------------------------------------------------------------

/// Multi-channel 3x3 (N=1, C_in=3, C_out=4, 5x5, pad=1): the canonical
/// standard-conv case via the general implicit-GEMM engine.
#[test]
fn implicit_gemm_multichannel_3x3_nchw() {
    let g = ConvGeom {
        n: 1,
        c_in: 3,
        c_out: 4,
        h: 5,
        w: 5,
        kh: 3,
        kw: 3,
        pad_h: 1,
        pad_w: 1,
        stride_h: 1,
        stride_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
    };
    let input = ramp(g.n * g.c_in * g.h * g.w, 0.37, -2.0);
    let filter = ramp(g.c_out * g.icpg() * g.kh * g.kw, 0.11, -0.5);

    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping implicit_gemm_multichannel_3x3_nchw: no CUDA device");
        return;
    };
    let got = run_gpu(
        &handle,
        &g,
        TensorLayout::Nchw,
        Engine::ImplicitGemm,
        &input,
        &filter,
    )
    .expect("implicit-gemm execute");
    let want = cpu_conv_nchw(&g, &input, &filter);
    assert_close(&got, &want);
}

/// 1x1 convolution (C_in=4, C_out=2): per-spatial channel matmul via `Conv1x1`.
#[test]
fn conv1x1_channel_matmul_nchw() {
    let g = ConvGeom {
        n: 2,
        c_in: 4,
        c_out: 2,
        h: 4,
        w: 5,
        kh: 1,
        kw: 1,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
    };
    let input = ramp(g.n * g.c_in * g.h * g.w, 0.21, -1.3);
    let filter = ramp(g.c_out * g.icpg() * g.kh * g.kw, 0.5, 0.25);

    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping conv1x1_channel_matmul_nchw: no CUDA device");
        return;
    };
    let got = run_gpu(
        &handle,
        &g,
        TensorLayout::Nchw,
        Engine::Conv1x1,
        &input,
        &filter,
    )
    .expect("conv1x1 execute");
    let want = cpu_conv_nchw(&g, &input, &filter);
    assert_close(&got, &want);
}

/// Non-square image (7x9) with a non-square, non-symmetric 3x5 kernel and
/// multiple channels: catches kernel-flip, H/W transpose, and filter-transpose
/// regressions that square/symmetric cases hide.
#[test]
fn implicit_gemm_nonsquare_asymmetric_3x5_nchw() {
    let g = ConvGeom {
        n: 1,
        c_in: 2,
        c_out: 3,
        h: 7,
        w: 9,
        kh: 3,
        kw: 5,
        pad_h: 0,
        pad_w: 0,
        stride_h: 1,
        stride_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
    };
    // Deliberately non-symmetric, irregular values.
    let input: Vec<f64> = (0..g.n * g.c_in * g.h * g.w)
        .map(|i| ((i * 7) % 13) as f64 - 6.0)
        .collect();
    let filter: Vec<f64> = (0..g.c_out * g.icpg() * g.kh * g.kw)
        .map(|i| (i as f64) * 0.25 - 1.0)
        .collect();

    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping implicit_gemm_nonsquare_asymmetric_3x5_nchw: no CUDA device");
        return;
    };
    let got = run_gpu(
        &handle,
        &g,
        TensorLayout::Nchw,
        Engine::ImplicitGemm,
        &input,
        &filter,
    )
    .expect("implicit-gemm execute");
    let want = cpu_conv_nchw(&g, &input, &filter);
    assert_close(&got, &want);
}

/// Strided + dilated multi-channel case to exercise the geometry parameters.
#[test]
fn implicit_gemm_strided_dilated_nchw() {
    let g = ConvGeom {
        n: 2,
        c_in: 3,
        c_out: 5,
        h: 9,
        w: 8,
        kh: 3,
        kw: 3,
        pad_h: 2,
        pad_w: 1,
        stride_h: 2,
        stride_w: 2,
        dil_h: 2,
        dil_w: 1,
        groups: 1,
    };
    let input = ramp(g.n * g.c_in * g.h * g.w, 0.05, -1.0);
    let filter = ramp(g.c_out * g.icpg() * g.kh * g.kw, 0.13, -0.7);

    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping implicit_gemm_strided_dilated_nchw: no CUDA device");
        return;
    };
    let got = run_gpu(
        &handle,
        &g,
        TensorLayout::Nchw,
        Engine::ImplicitGemm,
        &input,
        &filter,
    )
    .expect("implicit-gemm execute");
    let want = cpu_conv_nchw(&g, &input, &filter);
    assert_close(&got, &want);
}

/// Grouped convolution (groups=2) to exercise the channel-group routing.
#[test]
fn implicit_gemm_grouped_nchw() {
    let g = ConvGeom {
        n: 1,
        c_in: 4,
        c_out: 6,
        h: 6,
        w: 6,
        kh: 3,
        kw: 3,
        pad_h: 1,
        pad_w: 1,
        stride_h: 1,
        stride_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 2,
    };
    let input = ramp(g.n * g.c_in * g.h * g.w, 0.09, -0.4);
    let filter = ramp(g.c_out * g.icpg() * g.kh * g.kw, 0.17, 0.2);

    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping implicit_gemm_grouped_nchw: no CUDA device");
        return;
    };
    let got = run_gpu(
        &handle,
        &g,
        TensorLayout::Nchw,
        Engine::ImplicitGemm,
        &input,
        &filter,
    )
    .expect("implicit-gemm execute");
    let want = cpu_conv_nchw(&g, &input, &filter);
    assert_close(&got, &want);
}

/// NHWC (channels-last) multi-channel 3x3: validates the layout-specific
/// addressing path against an NHWC CPU reference.
#[test]
fn implicit_gemm_multichannel_3x3_nhwc() {
    let g = ConvGeom {
        n: 1,
        c_in: 3,
        c_out: 4,
        h: 5,
        w: 5,
        kh: 3,
        kw: 3,
        pad_h: 1,
        pad_w: 1,
        stride_h: 1,
        stride_w: 1,
        dil_h: 1,
        dil_w: 1,
        groups: 1,
    };
    let input = ramp(g.n * g.h * g.w * g.c_in, 0.37, -2.0);
    let filter = ramp(g.c_out * g.kh * g.kw * g.icpg(), 0.11, -0.5);

    let Some((_ctx, handle)) = try_handle() else {
        eprintln!("skipping implicit_gemm_multichannel_3x3_nhwc: no CUDA device");
        return;
    };
    let got = run_gpu(
        &handle,
        &g,
        TensorLayout::Nhwc,
        Engine::ImplicitGemm,
        &input,
        &filter,
    )
    .expect("implicit-gemm execute (nhwc)");
    let want = cpu_conv_nhwc(&g, &input, &filter);
    assert_close(&got, &want);
}

// ---------------------------------------------------------------------------
// PTX assembly tests (ptxas -arch=sm_86).
// ---------------------------------------------------------------------------

/// Assembles a PTX string for sm_86, returning `Ok(())` on success or the
/// ptxas stderr on failure. Skips (returns `Ok`) when ptxas is unavailable.
fn ptxas_assembles(ptx: &str, tag: &str) -> Result<(), String> {
    let path = std::env::temp_dir().join(format!("oxicuda_{tag}.ptx"));
    std::fs::write(&path, ptx).map_err(|e| e.to_string())?;
    let out = match Command::new("ptxas")
        .arg("-arch=sm_86")
        .arg(&path)
        .arg("-o")
        .arg("/dev/null")
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            eprintln!("skipping ptxas check for {tag}: ptxas not on PATH");
            return Ok(());
        }
    };
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

fn ptx_problem(prec: PtxType, kh: u32, kw: u32, layout: TensorLayout) -> ConvProblem {
    ConvProblem {
        batch: 1,
        in_channels: 8,
        in_dims: vec![7, 7],
        out_channels: 6,
        filter_dims: vec![kh, kw],
        padding: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: prec,
        output_type: prec,
        layout,
    }
}

#[test]
fn implicit_gemm_ptx_assembles_f32_f64() {
    for prec in [PtxType::F32, PtxType::F64] {
        for layout in [TensorLayout::Nchw, TensorLayout::Nhwc] {
            let eng = ImplicitGemmConv::new(ptx_problem(prec, 3, 5, layout), SmVersion::Sm86);
            let ptx = eng.generate_ptx().expect("implicit-gemm ptx");
            // Real compute markers (not a comment-only stub).
            let suffix = if prec == PtxType::F64 { "f64" } else { "f32" };
            assert!(
                ptx.contains(&format!("fma.rn.{suffix}")),
                "kernel must accumulate with fma.rn.{suffix}"
            );
            assert!(
                ptx.contains(&format!("st.global.{suffix}")),
                "kernel must store an {suffix} result"
            );
            let tag = format!(
                "implicit_gemm_{suffix}_{}",
                if layout.is_channels_last() {
                    "nhwc"
                } else {
                    "nchw"
                }
            );
            ptxas_assembles(&ptx, &tag).expect("implicit-gemm ptx assembles under sm_86");
        }
    }
}

#[test]
fn conv1x1_ptx_assembles_f32_f64() {
    for prec in [PtxType::F32, PtxType::F64] {
        for layout in [TensorLayout::Nchw, TensorLayout::Nhwc] {
            let mut p = ptx_problem(prec, 1, 1, layout);
            p.padding = vec![0, 0];
            let eng = Conv1x1::new(p, SmVersion::Sm86).expect("conv1x1 engine");
            let ptx = eng.generate_ptx().expect("conv1x1 ptx");
            let suffix = if prec == PtxType::F64 { "f64" } else { "f32" };
            assert!(
                ptx.contains(&format!("fma.rn.{suffix}")),
                "1x1 kernel must accumulate with fma.rn.{suffix}"
            );
            let tag = format!(
                "conv1x1_{suffix}_{}",
                if layout.is_channels_last() {
                    "nhwc"
                } else {
                    "nchw"
                }
            );
            ptxas_assembles(&ptx, &tag).expect("conv1x1 ptx assembles under sm_86");
        }
    }
}
