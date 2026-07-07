//! On-device GPU validation for the `conv_other` subsystem of `oxicuda-dnn`.
//!
//! Coverage (RTX A4000, sm_86):
//!
//! * **Numeric (CPU-oracle)** — the complete kernels of this cluster, each
//!   driven through its public `generate_*_ptx` (or `DeformableConvPlan` /
//!   `FftConv2dPlan`) entry, launched directly on the device, and checked
//!   against an independent CPU re-derivation:
//!   - conv3d: `im2col3d`, `col2im3d`, `direct3d_3x3x3`, `wgrad3d`
//!   - depthwise-separable: `depthwise`, `pointwise`, fully-fused `dw+pw`
//!   - deformable (DCNv2): forward (with/without modulation), backward-input
//!   - transposed conv: `col2im`, `weight_reshape`
//!   - FFT conv: `pointwise_mul` (real complex MAC)
//! * **Fragment (load / launch + canary or exact partial check)** — kernels
//!   whose body is a structural skeleton or computes only part of the intended
//!   op. For the gradient skeletons (all `dgrad`/`wgrad` implicit-GEMM +
//!   Winograd transforms, deformable backward-offset/weight) the body emits no
//!   real math, so we assert they assemble (`ptxas`), JIT-load, launch and
//!   synchronise fault-free and leave the output untouched / write the
//!   documented zeros (a canary that fails the moment a real body is added).
//!   For the two FFT-conv fragments (`pad_and_fft`, `ifft_and_crop`) we
//!   additionally assert the exact value of the partial operation they DO
//!   perform (zero-pad copy; strided gather + 1/N scale) — they do NOT perform
//!   the actual FFT butterflies.
//!
//! Every test skips cleanly when no CUDA device is present.

use super::*;

use oxicuda_launch::LaunchParams;
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::ir::PtxType;

use crate::conv::conv3d::{
    Conv3dConfig, generate_col2im3d_ptx, generate_direct3d_ptx, generate_im2col3d_ptx,
    generate_wgrad3d_ptx,
};
use crate::conv::deformable::{DeformableConvConfig, DeformableConvPlan};
use crate::conv::depthwise_separable::{
    ActivationType, DepthwiseSeparableConfig, generate_depthwise_conv_ptx,
    generate_fused_dw_pw_ptx, generate_pointwise_conv_ptx,
};
use crate::conv::descriptor::ConvProblem;
use crate::conv::dgrad::implicit_gemm::DgradImplicitGemm;
use crate::conv::dgrad::winograd::WinogradDgrad;
use crate::conv::fft_conv::FftConv2dPlan;
use crate::conv::transpose_conv::{
    TransposeConvConfig, generate_col2im_ptx, generate_weight_reshape_ptx,
};
use crate::conv::wgrad::implicit_gemm::WgradImplicitGemm;
use crate::conv::wgrad::winograd::WinogradWgrad;
use crate::types::TensorLayout;

// ---------------------------------------------------------------------------
// Small shared helpers
// ---------------------------------------------------------------------------

/// Deterministic random `f32` vector in `[lo, hi)`.
fn rand_f32(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut g = Lcg::new(seed);
    (0..n).map(|_| g.range_f32(lo, hi)).collect()
}

/// Deterministic random `f64` vector in `[lo, hi)`.
fn rand_f64(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f64> {
    let mut g = Lcg::new(seed);
    (0..n).map(|_| g.range_f64(lo, hi)).collect()
}

/// `f32` slice -> `f64` (for feeding the device data into a CPU oracle).
fn up64(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// `f64` slice -> `f32` (for comparing an oracle result to a device `f32`).
fn down32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

/// Maps the device SM version to the numeric form `FftConv2dPlan::new` wants.
fn sm_to_numeric(sm: SmVersion) -> u32 {
    match sm {
        SmVersion::Sm75 => 75,
        SmVersion::Sm80 => 80,
        SmVersion::Sm86 => 86,
        SmVersion::Sm89 => 89,
        SmVersion::Sm90 => 90,
        _ => 86,
    }
}

/// Asserts a fragment kernel performed no store (output left at `sentinel`).
fn assert_untouched_f32(buf: &DeviceBuffer<f32>, sentinel: f32, n: usize, tag: &str) {
    let mut host = vec![0.0f32; n];
    buf.copy_to_host(&mut host)
        .expect("copy back fragment buffer");
    for (i, &v) in host.iter().enumerate() {
        assert!(
            v == sentinel,
            "{tag}: element {i} = {v} != sentinel {sentinel}; \
             fragment kernel must not store (it has a comment-only body)"
        );
    }
}

/// Signed `o*stride + kdil - pad`, returned as `Some(coord)` when in `[0,size)`.
fn axis_in(o: usize, stride: usize, kdil: usize, pad: usize, size: usize) -> Option<usize> {
    let v = o as isize * stride as isize + kdil as isize - pad as isize;
    if v >= 0 && (v as usize) < size {
        Some(v as usize)
    } else {
        None
    }
}

// ===========================================================================
// conv3d — im2col3d / col2im3d / direct3d / wgrad3d  (numeric)
// ===========================================================================

/// CPU im2col for a single-sample 3-D volume (`groups == 1`).
fn im2col3d_ref(
    cfg: &Conv3dConfig,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    input: &[f64],
) -> Vec<f64> {
    let (od_n, oh_n, ow_n) = cfg.output_size(in_d, in_h, in_w);
    let c = cfg.in_channels;
    let (kd, kh, kw) = (cfg.kernel_d, cfg.kernel_h, cfg.kernel_w);
    let total_columns = od_n * oh_n * ow_n;
    let rows = c * kd * kh * kw;
    let in_hw = in_h * in_w;
    let in_dhw = in_d * in_hw;
    let mut out = vec![0.0f64; rows * total_columns];
    for col in 0..total_columns {
        let od = col / (oh_n * ow_n);
        let rem = col % (oh_n * ow_n);
        let oh = rem / ow_n;
        let ow = rem % ow_n;
        for ci in 0..c {
            let row_base = ci * kd * kh * kw;
            for kd_v in 0..kd {
                for kh_v in 0..kh {
                    for kw_v in 0..kw {
                        let val = match (
                            axis_in(od, cfg.stride_d, kd_v * cfg.dilation_d, cfg.pad_d, in_d),
                            axis_in(oh, cfg.stride_h, kh_v * cfg.dilation_h, cfg.pad_h, in_h),
                            axis_in(ow, cfg.stride_w, kw_v * cfg.dilation_w, cfg.pad_w, in_w),
                        ) {
                            (Some(id), Some(ih), Some(iw)) => {
                                input[ci * in_dhw + id * in_hw + ih * in_w + iw]
                            }
                            _ => 0.0,
                        };
                        let row = row_base + kd_v * kh * kw + kh_v * kw + kw_v;
                        out[row * total_columns + col] = val;
                    }
                }
            }
        }
    }
    out
}

#[test]
fn conv3d_im2col3d_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = Conv3dConfig {
        in_channels: 3,
        out_channels: 1,
        kernel_d: 2,
        kernel_h: 3,
        kernel_w: 3,
        stride_d: 1,
        stride_h: 1,
        stride_w: 1,
        pad_d: 0,
        pad_h: 1,
        pad_w: 1,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    let (in_d, in_h, in_w) = (4usize, 5, 5);
    let (od, oh, ow) = cfg.output_size(in_d, in_h, in_w);
    let total_columns = (od * oh * ow) as u32;
    let rows = cfg.in_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w;
    let cols = total_columns as usize;

    let input = rand_f32(cfg.in_channels * in_d * in_h * in_w, 0x3d_01, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let init = vec![-987.0f32; rows * cols];
    let col_buf = DeviceBuffer::from_host(&init).expect("alloc col");

    let ptx = generate_im2col3d_ptx(&cfg, 1, in_d, in_h, in_w, "f32", fx.sm).expect("im2col3d ptx");
    ptxas_assembles(&ptx, "im2col3d_f32").expect("ptxas im2col3d");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total_columns, 64), 64u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                in_buf.as_device_ptr(),
                col_buf.as_device_ptr(),
                total_columns,
            ),
        )
        .expect("launch im2col3d");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; rows * cols];
    col_buf.copy_to_host(&mut gpu).expect("copy col");
    let exp = down32(&im2col3d_ref(&cfg, in_d, in_h, in_w, &up64(&input)));
    assert_close_f32(&gpu, &exp, 1e-6, 1e-6, "im2col3d_f32");
}

#[test]
fn conv3d_im2col3d_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = Conv3dConfig {
        in_channels: 2,
        out_channels: 1,
        kernel_d: 3,
        kernel_h: 2,
        kernel_w: 2,
        stride_d: 2,
        stride_h: 1,
        stride_w: 1,
        pad_d: 1,
        pad_h: 0,
        pad_w: 1,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    let (in_d, in_h, in_w) = (5usize, 4, 5);
    let (od, oh, ow) = cfg.output_size(in_d, in_h, in_w);
    let total_columns = (od * oh * ow) as u32;
    let rows = cfg.in_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w;
    let cols = total_columns as usize;

    let input = rand_f64(cfg.in_channels * in_d * in_h * in_w, 0x3d_02, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let init = vec![-987.0f64; rows * cols];
    let col_buf = DeviceBuffer::from_host(&init).expect("alloc col");

    let ptx = generate_im2col3d_ptx(&cfg, 1, in_d, in_h, in_w, "f64", fx.sm).expect("im2col3d ptx");
    ptxas_assembles(&ptx, "im2col3d_f64").expect("ptxas im2col3d f64");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total_columns, 64), 64u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                in_buf.as_device_ptr(),
                col_buf.as_device_ptr(),
                total_columns,
            ),
        )
        .expect("launch im2col3d f64");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f64; rows * cols];
    col_buf.copy_to_host(&mut gpu).expect("copy col");
    let exp = im2col3d_ref(&cfg, in_d, in_h, in_w, &input);
    assert_close_f64(&gpu, &exp, 1e-12, 1e-12, "im2col3d_f64");
}

/// CPU col2im (gather) mirroring the kernel's stride/pad divisibility logic.
fn col2im3d_ref(
    cfg: &Conv3dConfig,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    col: &[f64],
) -> Vec<f64> {
    let (od_n, oh_n, ow_n) = cfg.output_size(in_d, in_h, in_w);
    let c = cfg.in_channels;
    let (kd, kh, kw) = (cfg.kernel_d, cfg.kernel_h, cfg.kernel_w);
    let kd_kh_kw = kd * kh * kw;
    let out_hw = oh_n * ow_n;
    let out_dhw = od_n * out_hw;
    let in_hw = in_h * in_w;
    let total = c * in_d * in_hw;
    let mut out = vec![0.0f64; total];
    let map = |idx: usize, pad: usize, kdil: usize, stride: usize, lim: usize| -> Option<usize> {
        let p = idx + pad;
        if p < kdil {
            return None;
        }
        let off = p - kdil;
        if off % stride != 0 {
            return None;
        }
        let o = off / stride;
        if o < lim { Some(o) } else { None }
    };
    for gid in 0..total {
        let ci = gid / (in_d * in_hw);
        let r1 = gid % (in_d * in_hw);
        let id = r1 / in_hw;
        let r2 = r1 % in_hw;
        let ih = r2 / in_w;
        let iw = r2 % in_w;
        let mut acc = 0.0;
        for kd_v in 0..kd {
            for kh_v in 0..kh {
                for kw_v in 0..kw {
                    let od = map(id, cfg.pad_d, kd_v * cfg.dilation_d, cfg.stride_d, od_n);
                    let oh = map(ih, cfg.pad_h, kh_v * cfg.dilation_h, cfg.stride_h, oh_n);
                    let ow = map(iw, cfg.pad_w, kw_v * cfg.dilation_w, cfg.stride_w, ow_n);
                    if let (Some(od), Some(oh), Some(ow)) = (od, oh, ow) {
                        let k_flat = kd_v * kh * kw + kh_v * kw + kw_v;
                        let col_row = ci * kd_kh_kw + k_flat;
                        let spatial = od * out_hw + oh * ow_n + ow;
                        acc += col[col_row * out_dhw + spatial];
                    }
                }
            }
        }
        out[gid] = acc;
    }
    out
}

#[test]
fn conv3d_col2im3d_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = Conv3dConfig {
        in_channels: 2,
        out_channels: 1,
        kernel_d: 2,
        kernel_h: 3,
        kernel_w: 3,
        stride_d: 1,
        stride_h: 1,
        stride_w: 1,
        pad_d: 0,
        pad_h: 1,
        pad_w: 1,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    let (in_d, in_h, in_w) = (4usize, 5, 5);
    let (od, oh, ow) = cfg.output_size(in_d, in_h, in_w);
    let rows = cfg.in_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w;
    let col_n = rows * (od * oh * ow);
    let total = (cfg.in_channels * in_d * in_h * in_w) as u32;

    let col = rand_f32(col_n, 0x3d_03, -1.0, 1.0);
    let col_buf = DeviceBuffer::from_host(&col).expect("upload col");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc out");

    let ptx = generate_col2im3d_ptx(&cfg, "f32", fx.sm).expect("col2im3d ptx");
    ptxas_assembles(&ptx, "col2im3d_f32").expect("ptxas col2im3d");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        col_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        cfg.in_channels as u32, // channels_per_group (unused by body)
        in_d as u32,
        in_h as u32,
        in_w as u32,
        od as u32,
        oh as u32,
        ow as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch col2im3d");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&col2im3d_ref(&cfg, in_d, in_h, in_w, &up64(&col)));
    assert_close_f32(&gpu, &exp, 1e-6, 1e-6, "col2im3d_f32");
}

/// CPU direct 3x3x3 conv (single sample, single group).
fn direct3d_ref(
    cfg: &Conv3dConfig,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    input: &[f64],
    weight: &[f64],
) -> Vec<f64> {
    let (od_n, oh_n, ow_n) = cfg.output_size(in_d, in_h, in_w);
    let in_cpg = cfg.in_channels;
    let out_cpg = cfg.out_channels;
    let out_dhw = od_n * oh_n * ow_n;
    let in_hw = in_h * in_w;
    let in_dhw = in_d * in_hw;
    let total = out_cpg * out_dhw;
    let mut out = vec![0.0f64; total];
    for gid in 0..total {
        let k = gid / out_dhw;
        let r1 = gid % out_dhw;
        let od = r1 / (oh_n * ow_n);
        let r2 = r1 % (oh_n * ow_n);
        let oh = r2 / ow_n;
        let ow = r2 % ow_n;
        let mut acc = 0.0;
        for c in 0..in_cpg {
            for kd_v in 0..3 {
                for kh_v in 0..3 {
                    for kw_v in 0..3 {
                        if let (Some(id), Some(ih), Some(iw)) = (
                            axis_in(od, cfg.stride_d, kd_v * cfg.dilation_d, cfg.pad_d, in_d),
                            axis_in(oh, cfg.stride_h, kh_v * cfg.dilation_h, cfg.pad_h, in_h),
                            axis_in(ow, cfg.stride_w, kw_v * cfg.dilation_w, cfg.pad_w, in_w),
                        ) {
                            let iv = input[c * in_dhw + id * in_hw + ih * in_w + iw];
                            let wv =
                                weight[k * in_cpg * 27 + c * 27 + (kd_v * 9 + kh_v * 3 + kw_v)];
                            acc += iv * wv;
                        }
                    }
                }
            }
        }
        out[gid] = acc;
    }
    out
}

#[test]
fn conv3d_direct3d_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = Conv3dConfig {
        in_channels: 3,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        stride_d: 1,
        stride_h: 1,
        stride_w: 1,
        pad_d: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    let (in_d, in_h, in_w) = (4usize, 5, 5);
    let (od, oh, ow) = cfg.output_size(in_d, in_h, in_w);
    let total = (cfg.out_channels * od * oh * ow) as u32;

    let input = rand_f32(cfg.in_channels * in_d * in_h * in_w, 0x3d_04, -1.0, 1.0);
    let weight = rand_f32(cfg.out_channels * cfg.in_channels * 27, 0x3d_05, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let w_buf = DeviceBuffer::from_host(&weight).expect("upload weight");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc out");

    let ptx = generate_direct3d_ptx(&cfg, "f32", fx.sm).expect("direct3d ptx");
    ptxas_assembles(&ptx, "direct3d_f32").expect("ptxas direct3d");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        in_d as u32,
        in_h as u32,
        in_w as u32,
        od as u32,
        oh as u32,
        ow as u32,
        cfg.out_channels as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch direct3d");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&direct3d_ref(
        &cfg,
        in_d,
        in_h,
        in_w,
        &up64(&input),
        &up64(&weight),
    ));
    assert_close_f32(&gpu, &exp, 1e-5, 1e-5, "direct3d_f32");
}

#[test]
fn conv3d_direct3d_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = Conv3dConfig {
        in_channels: 2,
        out_channels: 3,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        stride_d: 1,
        stride_h: 1,
        stride_w: 1,
        pad_d: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    let (in_d, in_h, in_w) = (4usize, 4, 4);
    let (od, oh, ow) = cfg.output_size(in_d, in_h, in_w);
    let total = (cfg.out_channels * od * oh * ow) as u32;

    let input = rand_f64(cfg.in_channels * in_d * in_h * in_w, 0x3d_06, -1.0, 1.0);
    let weight = rand_f64(cfg.out_channels * cfg.in_channels * 27, 0x3d_07, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let w_buf = DeviceBuffer::from_host(&weight).expect("upload weight");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f64; total as usize]).expect("alloc out");

    let ptx = generate_direct3d_ptx(&cfg, "f64", fx.sm).expect("direct3d ptx");
    ptxas_assembles(&ptx, "direct3d_f64").expect("ptxas direct3d f64");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        in_d as u32,
        in_h as u32,
        in_w as u32,
        od as u32,
        oh as u32,
        ow as u32,
        cfg.out_channels as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch direct3d f64");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f64; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = direct3d_ref(&cfg, in_d, in_h, in_w, &input, &weight);
    assert_close_f64(&gpu, &exp, 1e-10, 1e-12, "direct3d_f64");
}

/// CPU weight-gradient (wgrad) for a 3-D conv (single group, batched).
fn wgrad3d_ref(
    cfg: &Conv3dConfig,
    batch: usize,
    in_d: usize,
    in_h: usize,
    in_w: usize,
    input: &[f64],
    grad_out: &[f64],
) -> Vec<f64> {
    let (od_n, oh_n, ow_n) = cfg.output_size(in_d, in_h, in_w);
    let in_cpg = cfg.in_channels;
    let out_cpg = cfg.out_channels;
    let (kd, kh, kw) = (cfg.kernel_d, cfg.kernel_h, cfg.kernel_w);
    let kd_kh_kw = kd * kh * kw;
    let in_hw = in_h * in_w;
    let in_dhw = in_d * in_hw;
    let out_hw = oh_n * ow_n;
    let out_dhw = od_n * out_hw;
    let total = out_cpg * in_cpg * kd_kh_kw;
    let mut gw = vec![0.0f64; total];
    for gid in 0..total {
        let c_kd_kh_kw = in_cpg * kd_kh_kw;
        let k = gid / c_kd_kh_kw;
        let r1 = gid % c_kd_kh_kw;
        let c = r1 / kd_kh_kw;
        let r2 = r1 % kd_kh_kw;
        let kd_v = r2 / (kh * kw);
        let r3 = r2 % (kh * kw);
        let kh_v = r3 / kw;
        let kw_v = r3 % kw;
        let mut acc = 0.0;
        for n in 0..batch {
            for od in 0..od_n {
                let Some(id) = axis_in(od, cfg.stride_d, kd_v * cfg.dilation_d, cfg.pad_d, in_d)
                else {
                    continue;
                };
                for oh in 0..oh_n {
                    let Some(ih) =
                        axis_in(oh, cfg.stride_h, kh_v * cfg.dilation_h, cfg.pad_h, in_h)
                    else {
                        continue;
                    };
                    for ow in 0..ow_n {
                        let Some(iw) =
                            axis_in(ow, cfg.stride_w, kw_v * cfg.dilation_w, cfg.pad_w, in_w)
                        else {
                            continue;
                        };
                        let in_idx = n * in_cpg * in_dhw + c * in_dhw + id * in_hw + ih * in_w + iw;
                        let go_idx =
                            n * out_cpg * out_dhw + k * out_dhw + od * out_hw + oh * ow_n + ow;
                        acc += input[in_idx] * grad_out[go_idx];
                    }
                }
            }
        }
        gw[gid] = acc;
    }
    gw
}

#[test]
fn conv3d_wgrad3d_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = Conv3dConfig {
        in_channels: 2,
        out_channels: 3,
        kernel_d: 2,
        kernel_h: 3,
        kernel_w: 3,
        stride_d: 1,
        stride_h: 1,
        stride_w: 1,
        pad_d: 0,
        pad_h: 1,
        pad_w: 1,
        dilation_d: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    };
    let batch = 2usize;
    let (in_d, in_h, in_w) = (4usize, 5, 5);
    let (od, oh, ow) = cfg.output_size(in_d, in_h, in_w);
    let total =
        (cfg.out_channels * cfg.in_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w) as u32;

    let input = rand_f32(
        batch * cfg.in_channels * in_d * in_h * in_w,
        0x3d_08,
        -1.0,
        1.0,
    );
    let grad_out = rand_f32(batch * cfg.out_channels * od * oh * ow, 0x3d_09, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let go_buf = DeviceBuffer::from_host(&grad_out).expect("upload grad_out");
    let gw_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc gw");

    let ptx =
        generate_wgrad3d_ptx(&cfg, batch, in_d, in_h, in_w, "f32", fx.sm).expect("wgrad3d ptx");
    ptxas_assembles(&ptx, "wgrad3d_f32").expect("ptxas wgrad3d");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        go_buf.as_device_ptr(),
        gw_buf.as_device_ptr(),
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch wgrad3d");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    gw_buf.copy_to_host(&mut gpu).expect("copy gw");
    let exp = down32(&wgrad3d_ref(
        &cfg,
        batch,
        in_d,
        in_h,
        in_w,
        &up64(&input),
        &up64(&grad_out),
    ));
    assert_close_f32(&gpu, &exp, 1e-5, 1e-5, "wgrad3d_f32");
}

// ===========================================================================
// depthwise-separable — depthwise / pointwise / fused  (numeric)
// ===========================================================================

fn apply_act(act: ActivationType, x: f64) -> f64 {
    match act {
        ActivationType::None => x,
        ActivationType::Relu => x.max(0.0),
        ActivationType::Relu6 => x.clamp(0.0, 6.0),
        // Silu/HardSwish use the base-2 sigmoid quirk / extra ops; these tests
        // never select them (an oracle would diverge). Map defensively.
        _ => x,
    }
}

/// CPU depthwise conv (NCHW in/out, filter `[C,kh,kw]`, bias ignored by kernel).
fn dw_ref(
    cfg: &DepthwiseSeparableConfig,
    batch: usize,
    in_h: usize,
    in_w: usize,
    input: &[f64],
    filter: &[f64],
) -> Vec<f64> {
    let (oh_n, ow_n) = cfg.output_size(in_h, in_w);
    let c = cfg.channels;
    let in_hw = in_h * in_w;
    let mut out = vec![0.0f64; batch * c * oh_n * ow_n];
    for n in 0..batch {
        for ch in 0..c {
            for oh in 0..oh_n {
                for ow in 0..ow_n {
                    let mut acc = 0.0;
                    for kh_v in 0..cfg.kernel_h {
                        for kw_v in 0..cfg.kernel_w {
                            if let (Some(ih), Some(iw)) = (
                                axis_in(oh, cfg.stride_h, kh_v * cfg.dilation_h, cfg.pad_h, in_h),
                                axis_in(ow, cfg.stride_w, kw_v * cfg.dilation_w, cfg.pad_w, in_w),
                            ) {
                                let iv = input[(n * c + ch) * in_hw + ih * in_w + iw];
                                let fv = filter
                                    [ch * cfg.kernel_h * cfg.kernel_w + kh_v * cfg.kernel_w + kw_v];
                                acc += iv * fv;
                            }
                        }
                    }
                    let o = ((n * c + ch) * oh_n + oh) * ow_n + ow;
                    out[o] = apply_act(cfg.depthwise_activation, acc);
                }
            }
        }
    }
    out
}

fn run_depthwise(fx: &GpuFixture, act: ActivationType, seed: u64, tag: &str) {
    let cfg = DepthwiseSeparableConfig {
        channels: 4,
        out_channels: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        depth_multiplier: 1,
        depthwise_activation: act,
        pointwise_activation: ActivationType::None,
        depthwise_bn: false,
        pointwise_bn: false,
    };
    let batch = 2usize;
    let (in_h, in_w) = (6usize, 7);
    let (oh, ow) = cfg.output_size(in_h, in_w);
    let total = (batch * cfg.channels * oh * ow) as u32;

    let input = rand_f32(batch * cfg.channels * in_h * in_w, seed, -1.0, 1.0);
    let filter = rand_f32(
        cfg.channels * cfg.kernel_h * cfg.kernel_w,
        seed ^ 0xabc,
        -1.0,
        1.0,
    );
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let f_buf = DeviceBuffer::from_host(&filter).expect("upload filter");
    let bias = DeviceBuffer::from_host(&vec![0.0f32; cfg.channels]).expect("alloc bias");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc out");

    let ptx = generate_depthwise_conv_ptx(&cfg, "f32", fx.sm).expect("depthwise ptx");
    ptxas_assembles(&ptx, tag).expect("ptxas depthwise");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        f_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        bias.as_device_ptr(),
        batch as u32,
        cfg.channels as u32,
        in_h as u32,
        in_w as u32,
        oh as u32,
        ow as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch depthwise");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&dw_ref(
        &cfg,
        batch,
        in_h,
        in_w,
        &up64(&input),
        &up64(&filter),
    ));
    assert_close_f32(&gpu, &exp, 1e-5, 1e-5, tag);
}

#[test]
fn depthwise_identity_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_depthwise(&fx, ActivationType::None, 0xd_a1, "depthwise_identity_f32");
}

#[test]
fn depthwise_relu_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_depthwise(&fx, ActivationType::Relu, 0xd_a2, "depthwise_relu_f32");
}

/// CPU pointwise (1x1) conv: `out[s,oc] = sum_ic input[s,ic]*weight[oc,ic]`.
fn pw_ref(
    cfg: &DepthwiseSeparableConfig,
    spatial: usize,
    input: &[f64],
    weight: &[f64],
) -> Vec<f64> {
    let in_ch = cfg.depthwise_out_channels();
    let out_ch = cfg.out_channels;
    let mut out = vec![0.0f64; spatial * out_ch];
    for s in 0..spatial {
        for oc in 0..out_ch {
            let mut acc = 0.0;
            for ic in 0..in_ch {
                acc += input[s * in_ch + ic] * weight[oc * in_ch + ic];
            }
            out[s * out_ch + oc] = apply_act(cfg.pointwise_activation, acc);
        }
    }
    out
}

fn run_pointwise(fx: &GpuFixture, act: ActivationType, seed: u64, tag: &str) {
    let cfg = DepthwiseSeparableConfig {
        channels: 6,
        out_channels: 5,
        kernel_h: 1,
        kernel_w: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        dilation_h: 1,
        dilation_w: 1,
        depth_multiplier: 1,
        depthwise_activation: ActivationType::None,
        pointwise_activation: act,
        depthwise_bn: false,
        pointwise_bn: false,
    };
    let spatial = 11usize;
    let in_ch = cfg.depthwise_out_channels();
    let out_ch = cfg.out_channels;
    let total = (spatial * out_ch) as u32;

    let input = rand_f32(spatial * in_ch, seed, -1.0, 1.0);
    let weight = rand_f32(out_ch * in_ch, seed ^ 0x55, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let w_buf = DeviceBuffer::from_host(&weight).expect("upload weight");
    let bias = DeviceBuffer::from_host(&vec![0.0f32; out_ch]).expect("alloc bias");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc out");

    let ptx = generate_pointwise_conv_ptx(&cfg, "f32", fx.sm).expect("pointwise ptx");
    ptxas_assembles(&ptx, tag).expect("ptxas pointwise");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        bias.as_device_ptr(),
        out_buf.as_device_ptr(),
        in_ch as u32,
        out_ch as u32,
        spatial as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch pointwise");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&pw_ref(&cfg, spatial, &up64(&input), &up64(&weight)));
    assert_close_f32(&gpu, &exp, 1e-5, 1e-5, tag);
}

#[test]
fn pointwise_identity_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_pointwise(&fx, ActivationType::None, 0xd_b1, "pointwise_identity_f32");
}

#[test]
fn pointwise_relu_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_pointwise(&fx, ActivationType::Relu, 0xd_b2, "pointwise_relu_f32");
}

/// CPU fully-fused depthwise+pointwise. Output is NHWC `[N, oh, ow, oc]`.
fn fused_ref(
    cfg: &DepthwiseSeparableConfig,
    batch: usize,
    in_h: usize,
    in_w: usize,
    input: &[f64],
    dw_filter: &[f64],
    pw_weight: &[f64],
) -> Vec<f64> {
    let (oh_n, ow_n) = cfg.output_size(in_h, in_w);
    let c = cfg.channels;
    let dw_out = cfg.depthwise_out_channels(); // == c for depth_multiplier 1
    let out_ch = cfg.out_channels;
    let in_hw = in_h * in_w;
    let mut out = vec![0.0f64; batch * oh_n * ow_n * out_ch];
    for n in 0..batch {
        for oh in 0..oh_n {
            for ow in 0..ow_n {
                for oc in 0..out_ch {
                    let mut pw_acc = 0.0;
                    for ch in 0..dw_out {
                        let mut dw_acc = 0.0;
                        for kh_v in 0..cfg.kernel_h {
                            for kw_v in 0..cfg.kernel_w {
                                if let (Some(ih), Some(iw)) = (
                                    axis_in(
                                        oh,
                                        cfg.stride_h,
                                        kh_v * cfg.dilation_h,
                                        cfg.pad_h,
                                        in_h,
                                    ),
                                    axis_in(
                                        ow,
                                        cfg.stride_w,
                                        kw_v * cfg.dilation_w,
                                        cfg.pad_w,
                                        in_w,
                                    ),
                                ) {
                                    let iv = input[(n * c + ch) * in_hw + ih * in_w + iw];
                                    let fv = dw_filter[ch * cfg.kernel_h * cfg.kernel_w
                                        + kh_v * cfg.kernel_w
                                        + kw_v];
                                    dw_acc += iv * fv;
                                }
                            }
                        }
                        let dw_act = apply_act(cfg.depthwise_activation, dw_acc);
                        pw_acc += pw_weight[oc * dw_out + ch] * dw_act;
                    }
                    let o = ((n * oh_n + oh) * ow_n + ow) * out_ch + oc;
                    out[o] = apply_act(cfg.pointwise_activation, pw_acc);
                }
            }
        }
    }
    out
}

#[test]
fn fused_dw_pw_identity_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = DepthwiseSeparableConfig {
        channels: 4,
        out_channels: 8,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        depth_multiplier: 1,
        depthwise_activation: ActivationType::None,
        pointwise_activation: ActivationType::None,
        depthwise_bn: false,
        pointwise_bn: false,
    };
    let batch = 1usize;
    let (in_h, in_w) = (6usize, 6);
    let (oh, ow) = cfg.output_size(in_h, in_w);
    let dw_out = cfg.depthwise_out_channels();
    let total = (batch * oh * ow * cfg.out_channels) as u32;
    // The kernel's `bar.sync 0` follows the divergent bounds guard, so launch a
    // single block whose size exactly equals total_outputs (no thread takes the
    // early-exit branch -> the barrier is reached uniformly, never hangs).
    assert!(total <= 1024, "fused test must fit one block");

    let input = rand_f32(batch * cfg.channels * in_h * in_w, 0xd_c1, -1.0, 1.0);
    let dw_filter = rand_f32(
        cfg.channels * cfg.kernel_h * cfg.kernel_w,
        0xd_c2,
        -1.0,
        1.0,
    );
    let pw_weight = rand_f32(cfg.out_channels * dw_out, 0xd_c3, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let dwf_buf = DeviceBuffer::from_host(&dw_filter).expect("upload dw_filter");
    let pww_buf = DeviceBuffer::from_host(&pw_weight).expect("upload pw_weight");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc out");

    let ptx = generate_fused_dw_pw_ptx(&cfg, "f32", fx.sm).expect("fused ptx");
    ptxas_assembles(&ptx, "fused_dw_pw_identity_f32").expect("ptxas fused");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(1u32, total);
    let args = (
        in_buf.as_device_ptr(),
        dwf_buf.as_device_ptr(),
        pww_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        batch as u32,
        cfg.channels as u32,
        dw_out as u32,
        cfg.out_channels as u32,
        in_h as u32,
        in_w as u32,
        oh as u32,
        ow as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch fused");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&fused_ref(
        &cfg,
        batch,
        in_h,
        in_w,
        &up64(&input),
        &up64(&dw_filter),
        &up64(&pw_weight),
    ));
    assert_close_f32(&gpu, &exp, 1e-5, 1e-5, "fused_dw_pw_identity_f32");
}

// ===========================================================================
// deformable conv (DCNv2) — forward / backward-input (numeric); offset/weight (stub)
// ===========================================================================

/// Scalar geometry for the deformable-conv CPU oracles.
#[derive(Clone, Copy)]
struct Dcn {
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    dh: usize,
    dw: usize,
    offset_groups: usize,
    use_modulation: bool,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
}

impl Dcn {
    fn base(&self, o: usize, stride: usize, kdil: usize, pad: usize) -> f64 {
        (o * stride + kdil) as f64 - pad as f64
    }
}

/// CPU deformable-conv forward (DCNv2). Output is NCHW `[N, out_ch, oh, ow]`.
fn dcn_forward_ref(
    d: &Dcn,
    input: &[f64],
    offset: &[f64],
    mask: &[f64],
    weight: &[f64],
    bias: &[f64],
) -> Vec<f64> {
    let kh_kw = d.kh * d.kw;
    let cpog = d.in_channels / d.offset_groups;
    let out_hw = d.out_h * d.out_w;
    let off_stride = 2 * kh_kw * d.offset_groups;
    let mask_stride = kh_kw * d.offset_groups;
    let mut out = vec![0.0f64; d.batch * d.out_channels * out_hw];
    for n in 0..d.batch {
        for c_out in 0..d.out_channels {
            for oh in 0..d.out_h {
                for ow in 0..d.out_w {
                    let spatial = oh * d.out_w + ow;
                    let mut acc = 0.0;
                    for c_in in 0..d.in_channels {
                        let og = c_in / cpog;
                        for kh_v in 0..d.kh {
                            for kw_v in 0..d.kw {
                                let kpos = kh_v * d.kw + kw_v;
                                let h_base = d.base(oh, d.sh, kh_v * d.dh, d.ph);
                                let w_base = d.base(ow, d.sw, kw_v * d.dw, d.pw);
                                let off_base =
                                    (n * off_stride + (og * kh_kw + kpos) * 2) * out_hw + spatial;
                                let h_s = h_base + offset[off_base];
                                let w_s = w_base + offset[off_base + out_hw];
                                let h_fl = h_s.floor();
                                let w_fl = w_s.floor();
                                let hf = h_s - h_fl;
                                let wf = w_s - w_fl;
                                let h0 = h_fl as isize;
                                let w0 = w_fl as isize;
                                let mut interp = 0.0;
                                for (hh, ww, bw) in [
                                    (h0, w0, (1.0 - hf) * (1.0 - wf)),
                                    (h0, w0 + 1, (1.0 - hf) * wf),
                                    (h0 + 1, w0, hf * (1.0 - wf)),
                                    (h0 + 1, w0 + 1, hf * wf),
                                ] {
                                    if hh >= 0
                                        && ww >= 0
                                        && (hh as usize) < d.in_h
                                        && (ww as usize) < d.in_w
                                    {
                                        let pix = ((n * d.in_channels + c_in) * d.in_h
                                            + hh as usize)
                                            * d.in_w
                                            + ww as usize;
                                        interp += bw * input[pix];
                                    }
                                }
                                let w_idx = (c_out * d.in_channels + c_in) * kh_kw + kpos;
                                let mut contrib = interp * weight[w_idx];
                                if d.use_modulation {
                                    let m_base =
                                        (n * mask_stride + og * kh_kw + kpos) * out_hw + spatial;
                                    contrib *= mask[m_base];
                                }
                                acc += contrib;
                            }
                        }
                    }
                    acc += bias[c_out];
                    let o = ((n * d.out_channels + c_out) * d.out_h + oh) * d.out_w + ow;
                    out[o] = acc;
                }
            }
        }
    }
    out
}

fn run_dcn_forward(fx: &GpuFixture, use_modulation: bool, tag: &str) {
    let cfg = DeformableConvConfig {
        in_channels: 3,
        out_channels: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        offset_groups: 1,
        use_modulation,
        sm_version: fx.sm,
        float_type: PtxType::F32,
    };
    let (batch, in_h, in_w) = (2usize, 5, 5);
    let (out_h, out_w) = cfg.output_size(in_h as u32, in_w as u32);
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let d = Dcn {
        batch,
        in_channels: cfg.in_channels as usize,
        out_channels: cfg.out_channels as usize,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
        dh: 1,
        dw: 1,
        offset_groups: 1,
        use_modulation,
        in_h,
        in_w,
        out_h,
        out_w,
    };
    let kh_kw = d.kh * d.kw;
    let in_n = batch * d.in_channels * in_h * in_w;
    let off_n = batch * 2 * kh_kw * d.offset_groups * out_h * out_w;
    let mask_n = batch * kh_kw * d.offset_groups * out_h * out_w;
    let w_n = d.out_channels * d.in_channels * kh_kw;
    let out_n = batch * d.out_channels * out_h * out_w;

    let input = rand_f32(in_n, 0xdc_01, -1.0, 1.0);
    // Offsets in [0.2, 0.7): integer base + this frac keeps floor() stable (away
    // from integer boundaries) so the f32 kernel and f64 oracle round identically.
    let offset = rand_f32(off_n, 0xdc_02, 0.2, 0.7);
    let mask = rand_f32(mask_n, 0xdc_03, 0.1, 1.0);
    let weight = rand_f32(w_n, 0xdc_04, -1.0, 1.0);
    let bias = rand_f32(d.out_channels, 0xdc_05, -0.5, 0.5);

    let in_buf = DeviceBuffer::from_host(&input).expect("upload input");
    let off_buf = DeviceBuffer::from_host(&offset).expect("upload offset");
    let mask_buf = DeviceBuffer::from_host(&mask).expect("upload mask");
    let w_buf = DeviceBuffer::from_host(&weight).expect("upload weight");
    let bias_buf = DeviceBuffer::from_host(&bias).expect("upload bias");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; out_n]).expect("alloc out");

    let plan = DeformableConvPlan::new(cfg.clone()).expect("plan");
    let ptx = plan.generate_forward().expect("dcn forward ptx");
    ptxas_assembles(&ptx, tag).expect("ptxas dcn forward");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let total = out_n as u32;
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        off_buf.as_device_ptr(),
        mask_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        bias_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        batch as u32,
        cfg.in_channels,
        in_h as u32,
        in_w as u32,
        cfg.out_channels,
        out_h as u32,
        out_w as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch dcn forward");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; out_n];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&dcn_forward_ref(
        &d,
        &up64(&input),
        &up64(&offset),
        &up64(&mask),
        &up64(&weight),
        &up64(&bias),
    ));
    assert_close_f32(&gpu, &exp, 2e-4, 2e-4, tag);
}

#[test]
fn deformable_forward_dcnv2_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_dcn_forward(&fx, true, "deformable_forward_dcnv2_f32");
}

#[test]
fn deformable_forward_dcnv1_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_dcn_forward(&fx, false, "deformable_forward_dcnv1_f32");
}

/// CPU deformable backward-input scatter. Returns grad_input `[N, C, in_h, in_w]`.
fn dcn_bwd_input_ref(
    d: &Dcn,
    grad_output: &[f64],
    offset: &[f64],
    mask: &[f64],
    weight: &[f64],
) -> Vec<f64> {
    let kh_kw = d.kh * d.kw;
    let cpog = d.in_channels / d.offset_groups;
    let out_hw = d.out_h * d.out_w;
    let in_hw = d.in_h * d.in_w;
    let off_stride = 2 * kh_kw * d.offset_groups;
    let mask_stride = kh_kw * d.offset_groups;
    let mut gi = vec![0.0f64; d.batch * d.in_channels * in_hw];
    for n in 0..d.batch {
        for c_out in 0..d.out_channels {
            for oh in 0..d.out_h {
                for ow in 0..d.out_w {
                    let spatial = oh * d.out_w + ow;
                    let go =
                        grad_output[((n * d.out_channels + c_out) * d.out_h + oh) * d.out_w + ow];
                    for c_in in 0..d.in_channels {
                        let og = c_in / cpog;
                        for kh_v in 0..d.kh {
                            for kw_v in 0..d.kw {
                                let kpos = kh_v * d.kw + kw_v;
                                let h_base = d.base(oh, d.sh, kh_v * d.dh, d.ph);
                                let w_base = d.base(ow, d.sw, kw_v * d.dw, d.pw);
                                let off_base =
                                    (n * off_stride + (og * kh_kw + kpos) * 2) * out_hw + spatial;
                                let h_s = h_base + offset[off_base];
                                let w_s = w_base + offset[off_base + out_hw];
                                let h_fl = h_s.floor();
                                let w_fl = w_s.floor();
                                let hf = h_s - h_fl;
                                let wf = w_s - w_fl;
                                let h0 = h_fl as isize;
                                let w0 = w_fl as isize;
                                let w_idx = (c_out * d.in_channels + c_in) * kh_kw + kpos;
                                let mut grad_scaled = go * weight[w_idx];
                                if d.use_modulation {
                                    let m_base =
                                        (n * mask_stride + og * kh_kw + kpos) * out_hw + spatial;
                                    grad_scaled *= mask[m_base];
                                }
                                for (hh, ww, bw) in [
                                    (h0, w0, (1.0 - hf) * (1.0 - wf)),
                                    (h0, w0 + 1, (1.0 - hf) * wf),
                                    (h0 + 1, w0, hf * (1.0 - wf)),
                                    (h0 + 1, w0 + 1, hf * wf),
                                ] {
                                    if hh >= 0
                                        && ww >= 0
                                        && (hh as usize) < d.in_h
                                        && (ww as usize) < d.in_w
                                    {
                                        let pix = ((n * d.in_channels + c_in) * d.in_h
                                            + hh as usize)
                                            * d.in_w
                                            + ww as usize;
                                        gi[pix] += grad_scaled * bw;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    gi
}

#[test]
fn deformable_backward_input_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = DeformableConvConfig {
        in_channels: 3,
        out_channels: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        offset_groups: 1,
        use_modulation: true,
        sm_version: fx.sm,
        float_type: PtxType::F32,
    };
    let (batch, in_h, in_w) = (1usize, 5, 5);
    let (out_h, out_w) = cfg.output_size(in_h as u32, in_w as u32);
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let d = Dcn {
        batch,
        in_channels: cfg.in_channels as usize,
        out_channels: cfg.out_channels as usize,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
        dh: 1,
        dw: 1,
        offset_groups: 1,
        use_modulation: true,
        in_h,
        in_w,
        out_h,
        out_w,
    };
    let kh_kw = d.kh * d.kw;
    let go_n = batch * d.out_channels * out_h * out_w;
    let off_n = batch * 2 * kh_kw * out_h * out_w;
    let mask_n = batch * kh_kw * out_h * out_w;
    let w_n = d.out_channels * d.in_channels * kh_kw;
    let gi_n = batch * d.in_channels * in_h * in_w;

    let grad_out = rand_f32(go_n, 0xdc_11, -1.0, 1.0);
    let offset = rand_f32(off_n, 0xdc_12, 0.2, 0.7);
    let mask = rand_f32(mask_n, 0xdc_13, 0.1, 1.0);
    let weight = rand_f32(w_n, 0xdc_14, -1.0, 1.0);

    let go_buf = DeviceBuffer::from_host(&grad_out).expect("upload grad_out");
    let off_buf = DeviceBuffer::from_host(&offset).expect("upload offset");
    let mask_buf = DeviceBuffer::from_host(&mask).expect("upload mask");
    let w_buf = DeviceBuffer::from_host(&weight).expect("upload weight");
    let gi_buf = DeviceBuffer::from_host(&vec![0.0f32; gi_n]).expect("alloc grad_input");

    let plan = DeformableConvPlan::new(cfg.clone()).expect("plan");
    let ptx = plan.generate_backward_input().expect("dcn bwd input ptx");
    ptxas_assembles(&ptx, "deformable_backward_input_f32").expect("ptxas dcn bwd input");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let total = go_n as u32;
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        go_buf.as_device_ptr(),
        off_buf.as_device_ptr(),
        mask_buf.as_device_ptr(),
        w_buf.as_device_ptr(),
        gi_buf.as_device_ptr(),
        batch as u32,
        cfg.in_channels,
        in_h as u32,
        in_w as u32,
        cfg.out_channels,
        out_h as u32,
        out_w as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch dcn bwd input");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; gi_n];
    gi_buf.copy_to_host(&mut gpu).expect("copy grad_input");
    let exp = down32(&dcn_bwd_input_ref(
        &d,
        &up64(&grad_out),
        &up64(&offset),
        &up64(&mask),
        &up64(&weight),
    ));
    // f32 atomic-add accumulation in nondeterministic order vs an f64 oracle.
    assert_close_f32(&gpu, &exp, 2e-3, 2e-3, "deformable_backward_input_f32");
}

#[test]
fn deformable_backward_offset_stub_runs_clean() {
    // STUB body: loads params, inits grad_acc = 0, stores 0 to grad_offset[gid].
    // No gradient math. Assert it assembles + launches + writes the documented
    // zeros (never a fabricated gradient).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = DeformableConvConfig {
        in_channels: 3,
        out_channels: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        offset_groups: 1,
        use_modulation: true,
        sm_version: fx.sm,
        float_type: PtxType::F32,
    };
    let (batch, in_h, in_w) = (1u32, 5u32, 5u32);
    let (out_h, out_w) = cfg.output_size(in_h, in_w);
    let total = batch * 2 * cfg.kernel_h * cfg.kernel_w * cfg.offset_groups * out_h * out_w;

    let dummy = DeviceBuffer::from_host(&vec![0.5f32; 64]).expect("dummy");
    let go_buf = DeviceBuffer::from_host(&vec![
        0.3f32;
        (batch * cfg.out_channels * out_h * out_w) as usize
    ])
    .expect("go");
    let grad_off = DeviceBuffer::from_host(&vec![-1.0f32; total as usize]).expect("grad_off");
    let grad_mask = DeviceBuffer::from_host(&vec![-2.0f32; total as usize]).expect("grad_mask");

    let plan = DeformableConvPlan::new(cfg.clone()).expect("plan");
    let ptx = plan.generate_backward_offset().expect("dcn bwd offset ptx");
    ptxas_assembles(&ptx, "deformable_backward_offset").expect("ptxas dcn bwd offset");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        go_buf.as_device_ptr(),
        dummy.as_device_ptr(),
        dummy.as_device_ptr(),
        dummy.as_device_ptr(),
        dummy.as_device_ptr(),
        grad_off.as_device_ptr(),
        grad_mask.as_device_ptr(),
        batch,
        cfg.in_channels,
        in_h,
        in_w,
        cfg.out_channels,
        out_h,
        out_w,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch dcn bwd offset");
    fx.stream().synchronize().expect("sync");

    let mut go = vec![0.0f32; total as usize];
    grad_off.copy_to_host(&mut go).expect("copy grad_off");
    for (i, &v) in go.iter().enumerate() {
        assert!(
            v == 0.0,
            "backward_offset stub: grad_offset[{i}] = {v}, expected 0"
        );
    }
}

#[test]
fn deformable_backward_weight_stub_runs_clean() {
    // STUB body: decomposes gid, inits acc = 0, stores 0 to grad_weight[gid].
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = DeformableConvConfig {
        in_channels: 3,
        out_channels: 4,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        offset_groups: 1,
        use_modulation: true,
        sm_version: fx.sm,
        float_type: PtxType::F32,
    };
    let (batch, in_h, in_w) = (1u32, 5u32, 5u32);
    let (out_h, out_w) = cfg.output_size(in_h, in_w);
    let total = cfg.out_channels * cfg.in_channels * cfg.kernel_h * cfg.kernel_w;

    let dummy = DeviceBuffer::from_host(&vec![0.5f32; 64]).expect("dummy");
    let go_buf = DeviceBuffer::from_host(&vec![
        0.3f32;
        (batch * cfg.out_channels * out_h * out_w) as usize
    ])
    .expect("go");
    let grad_w = DeviceBuffer::from_host(&vec![-1.0f32; total as usize]).expect("grad_w");

    let plan = DeformableConvPlan::new(cfg.clone()).expect("plan");
    let ptx = plan.generate_backward_weight().expect("dcn bwd weight ptx");
    ptxas_assembles(&ptx, "deformable_backward_weight").expect("ptxas dcn bwd weight");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        go_buf.as_device_ptr(),
        dummy.as_device_ptr(),
        dummy.as_device_ptr(),
        dummy.as_device_ptr(),
        grad_w.as_device_ptr(),
        batch,
        cfg.in_channels,
        in_h,
        in_w,
        cfg.out_channels,
        out_h,
        out_w,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch dcn bwd weight");
    fx.stream().synchronize().expect("sync");

    let mut gw = vec![0.0f32; total as usize];
    grad_w.copy_to_host(&mut gw).expect("copy grad_w");
    for (i, &v) in gw.iter().enumerate() {
        assert!(
            v == 0.0,
            "backward_weight stub: grad_weight[{i}] = {v}, expected 0"
        );
    }
}

// ===========================================================================
// transposed conv — col2im / weight_reshape  (numeric)
// ===========================================================================

fn tconv_cfg() -> TransposeConvConfig {
    TransposeConvConfig {
        in_channels: 2,
        out_channels: 3,
        kernel_h: 3,
        kernel_w: 3,
        stride_h: 2,
        stride_w: 2,
        pad_h: 1,
        pad_w: 1,
        output_pad_h: 1,
        output_pad_w: 1,
        dilation_h: 1,
        dilation_w: 1,
        groups: 1,
    }
}

/// CPU col2im scatter for transposed conv (single sample, single group).
fn tconv_col2im_ref(
    cfg: &TransposeConvConfig,
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    col: &[f64],
) -> Vec<f64> {
    let out_cpg = cfg.out_channels_per_group();
    let (kh, kw) = (cfg.kernel_h, cfg.kernel_w);
    let kh_kw = kh * kw;
    let in_hw = in_h * in_w;
    let mut out = vec![0.0f64; out_cpg * out_h * out_w];
    let map = |o_plus_pad: usize, kdil: usize, stride: usize, lim: usize| -> Option<usize> {
        if o_plus_pad < kdil {
            return None;
        }
        let off = o_plus_pad - kdil;
        if off % stride != 0 {
            return None;
        }
        let i = off / stride;
        if i < lim { Some(i) } else { None }
    };
    for gid in 0..(out_cpg * out_h * out_w) {
        let c_out = gid / (out_h * out_w);
        let rem = gid % (out_h * out_w);
        let oh = rem / out_w;
        let ow = rem % out_w;
        let mut acc = 0.0;
        for kh_v in 0..kh {
            for kw_v in 0..kw {
                let ih = map(oh + cfg.pad_h, kh_v * cfg.dilation_h, cfg.stride_h, in_h);
                let iw = map(ow + cfg.pad_w, kw_v * cfg.dilation_w, cfg.stride_w, in_w);
                if let (Some(ih), Some(iw)) = (ih, iw) {
                    let col_row = c_out * kh_kw + (kh_v * kw + kw_v);
                    acc += col[col_row * in_hw + ih * in_w + iw];
                }
            }
        }
        out[gid] = acc;
    }
    out
}

#[test]
fn transpose_conv_col2im_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = tconv_cfg();
    let (in_h, in_w) = (4usize, 4);
    let (out_h, out_w) = cfg.output_size(in_h, in_w);
    let out_cpg = cfg.out_channels_per_group();
    let rows = out_cpg * cfg.kernel_h * cfg.kernel_w;
    let col_n = rows * in_h * in_w;
    let total = (out_cpg * out_h * out_w) as u32;

    let col64 = rand_f64(col_n, 0x7c_01, -1.0, 1.0);
    let col_t = down32(&col64);
    let col_buf = DeviceBuffer::from_host(&col_t).expect("upload col");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc out");

    let ptx = generate_col2im_ptx(&cfg, "f32", fx.sm).expect("tconv col2im ptx");
    ptxas_assembles(&ptx, "tconv_col2im_f32").expect("ptxas tconv col2im");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        col_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        out_cpg as u32,
        out_h as u32,
        out_w as u32,
        in_h as u32,
        in_w as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch tconv col2im");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = down32(&tconv_col2im_ref(&cfg, in_h, in_w, out_h, out_w, &col64));
    assert_close_f32(&gpu, &exp, 1e-5, 1e-6, "tconv_col2im_f32");
}

#[test]
fn transpose_conv_col2im_f64_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = tconv_cfg();
    let (in_h, in_w) = (4usize, 4);
    let (out_h, out_w) = cfg.output_size(in_h, in_w);
    let out_cpg = cfg.out_channels_per_group();
    let rows = out_cpg * cfg.kernel_h * cfg.kernel_w;
    let col_n = rows * in_h * in_w;
    let total = (out_cpg * out_h * out_w) as u32;

    let col = rand_f64(col_n, 0x7c_02, -1.0, 1.0);
    let col_buf = DeviceBuffer::from_host(&col).expect("upload col");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f64; total as usize]).expect("alloc out");

    let ptx = generate_col2im_ptx(&cfg, "f64", fx.sm).expect("tconv col2im ptx");
    ptxas_assembles(&ptx, "tconv_col2im_f64").expect("ptxas tconv col2im f64");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        col_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
        out_cpg as u32,
        out_h as u32,
        out_w as u32,
        in_h as u32,
        in_w as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch tconv col2im f64");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f64; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");
    let exp = tconv_col2im_ref(&cfg, in_h, in_w, out_h, out_w, &col);
    assert_close_f64(&gpu, &exp, 1e-12, 1e-12, "tconv_col2im_f64");
}

/// CPU weight-reshape permutation: `dst[g,oc,ic,kh,kw] = src[g*in_cpg+ic, oc, kh, kw]`.
fn wreshape_ref(cfg: &TransposeConvConfig, src: &[f64]) -> Vec<f64> {
    let groups = cfg.groups;
    let in_cpg = cfg.in_channels_per_group();
    let out_cpg = cfg.out_channels_per_group();
    let (kh, kw) = (cfg.kernel_h, cfg.kernel_w);
    let kh_kw = kh * kw;
    let mut dst = vec![0.0f64; groups * out_cpg * in_cpg * kh_kw];
    for g in 0..groups {
        for oc in 0..out_cpg {
            for ic in 0..in_cpg {
                for k in 0..kh_kw {
                    let in_ch = g * in_cpg + ic;
                    let src_idx = in_ch * (out_cpg * kh_kw) + oc * kh_kw + k;
                    let dst_idx = ((g * out_cpg + oc) * in_cpg + ic) * kh_kw + k;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
    dst
}

#[test]
fn transpose_conv_weight_reshape_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = TransposeConvConfig {
        in_channels: 4,
        out_channels: 6,
        kernel_h: 3,
        kernel_w: 2,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
        output_pad_h: 0,
        output_pad_w: 0,
        dilation_h: 1,
        dilation_w: 1,
        groups: 2,
    };
    let in_cpg = cfg.in_channels_per_group();
    let out_cpg = cfg.out_channels_per_group();
    let kh_kw = cfg.kernel_h * cfg.kernel_w;
    let src_n = cfg.in_channels * out_cpg * kh_kw;
    let total = (cfg.groups * out_cpg * in_cpg * kh_kw) as u32;

    let src = rand_f32(src_n, 0x7c_03, -1.0, 1.0);
    let src_buf = DeviceBuffer::from_host(&src).expect("upload src");
    let dst_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("alloc dst");

    let ptx = generate_weight_reshape_ptx(&cfg, "f32", fx.sm).expect("weight reshape ptx");
    ptxas_assembles(&ptx, "tconv_weight_reshape_f32").expect("ptxas weight reshape");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (src_buf.as_device_ptr(), dst_buf.as_device_ptr(), total);
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch weight reshape");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    dst_buf.copy_to_host(&mut gpu).expect("copy dst");
    let exp = down32(&wreshape_ref(&cfg, &up64(&src)));
    // Pure bit-copy permutation -> exact.
    assert_close_f32(&gpu, &exp, 0.0, 0.0, "tconv_weight_reshape_f32");
}

// ===========================================================================
// FFT conv — pointwise_mul (numeric) ; pad_and_fft / ifft_and_crop (fragments)
// ===========================================================================

/// CPU frequency-domain complex MAC: `Y[b,oc,p] = sum_ic X[b,ic,p]*W[oc,ic,p]`.
fn fft_pmul_ref(
    dims: (usize, usize, usize, usize),
    x_re: &[f64],
    x_im: &[f64],
    w_re: &[f64],
    w_im: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (batch, in_ch, out_ch, area) = dims;
    let mut out_re = vec![0.0f64; batch * out_ch * area];
    let mut out_im = vec![0.0f64; batch * out_ch * area];
    for b in 0..batch {
        for oc in 0..out_ch {
            for p in 0..area {
                let mut acc_re = 0.0;
                let mut acc_im = 0.0;
                for ic in 0..in_ch {
                    let xi = (b * in_ch + ic) * area + p;
                    let wi = (oc * in_ch + ic) * area + p;
                    acc_re += x_re[xi] * w_re[wi] - x_im[xi] * w_im[wi];
                    acc_im += x_re[xi] * w_im[wi] + x_im[xi] * w_re[wi];
                }
                let o = (b * out_ch + oc) * area + p;
                out_re[o] = acc_re;
                out_im[o] = acc_im;
            }
        }
    }
    (out_re, out_im)
}

#[test]
fn fft_pointwise_multiply_f32_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = FftConv2dPlan::new(3, 4, 7, 7, 1, 1, 0, 0, sm_to_numeric(fx.sm), PtxType::F32)
        .expect("fft plan");
    let (batch, in_ch, out_ch, area) = (2usize, 3usize, 4usize, 5usize);
    let total = (batch * out_ch * area) as u32;

    let x_re = rand_f32(batch * in_ch * area, 0xff_01, -1.0, 1.0);
    let x_im = rand_f32(batch * in_ch * area, 0xff_02, -1.0, 1.0);
    let w_re = rand_f32(out_ch * in_ch * area, 0xff_03, -1.0, 1.0);
    let w_im = rand_f32(out_ch * in_ch * area, 0xff_04, -1.0, 1.0);
    let xr = DeviceBuffer::from_host(&x_re).expect("xr");
    let xi = DeviceBuffer::from_host(&x_im).expect("xi");
    let wr = DeviceBuffer::from_host(&w_re).expect("wr");
    let wi = DeviceBuffer::from_host(&w_im).expect("wi");
    let or = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("or");
    let oi = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("oi");

    // pointwise_multiply's body only depends on float_type; in/out channels and
    // fft_area are runtime params, so any (in_h, in_w) producing a valid FFT
    // size works for codegen.
    let ptx = plan.generate_pointwise_multiply(8, 8).expect("pmul ptx");
    ptxas_assembles(&ptx, "fft_pointwise_multiply_f32").expect("ptxas pmul");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        xr.as_device_ptr(),
        xi.as_device_ptr(),
        wr.as_device_ptr(),
        wi.as_device_ptr(),
        or.as_device_ptr(),
        oi.as_device_ptr(),
        batch as u32,
        in_ch as u32,
        out_ch as u32,
        area as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch pmul");
    fx.stream().synchronize().expect("sync");

    let mut gpu_re = vec![0.0f32; total as usize];
    let mut gpu_im = vec![0.0f32; total as usize];
    or.copy_to_host(&mut gpu_re).expect("copy re");
    oi.copy_to_host(&mut gpu_im).expect("copy im");
    let (exp_re, exp_im) = fft_pmul_ref(
        (batch, in_ch, out_ch, area),
        &up64(&x_re),
        &up64(&x_im),
        &up64(&w_re),
        &up64(&w_im),
    );
    assert_close_f32(&gpu_re, &down32(&exp_re), 1e-5, 1e-5, "fft_pmul_re");
    assert_close_f32(&gpu_im, &down32(&exp_im), 1e-5, 1e-5, "fft_pmul_im");
}

#[test]
fn fft_pad_and_fft_zero_pad_copy_matches_cpu() {
    // FRAGMENT: this kernel only performs the zero-padded copy into padded_re
    // and zeroes padded_im — the FFT butterflies are delegated/unwritten and
    // the twiddle outputs are never touched. We assert exactly that partial
    // operation (it is NOT a real forward FFT).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = FftConv2dPlan::new(2, 1, 3, 3, 1, 1, 1, 1, sm_to_numeric(fx.sm), PtxType::F32)
        .expect("fft plan");
    let (batch, in_ch, in_h, in_w) = (1usize, 2usize, 4usize, 4usize);
    let (fft_h, fft_w) = plan.fft_size(in_h as u32, in_w as u32).expect("fft size");
    let (fft_h, fft_w) = (fft_h as usize, fft_w as usize);
    let fft_area = fft_h * fft_w;
    let total = (batch * in_ch * fft_area) as u32;
    let (pad_h, pad_w) = (1usize, 1usize);

    let input = rand_f32(batch * in_ch * in_h * in_w, 0xfa_01, -1.0, 1.0);
    let in_buf = DeviceBuffer::from_host(&input).expect("input");
    let pr = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("padded_re");
    let pi = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("padded_im");
    let tw = DeviceBuffer::from_host(&vec![0.0f32; fft_h + fft_w]).expect("twiddle");

    let ptx = plan
        .generate_pad_and_fft_kernel(in_h as u32, in_w as u32)
        .expect("pad_fft ptx");
    ptxas_assembles(&ptx, "fft_pad_and_fft_f32").expect("ptxas pad_fft");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        in_buf.as_device_ptr(),
        pr.as_device_ptr(),
        pi.as_device_ptr(),
        tw.as_device_ptr(),
        tw.as_device_ptr(),
        batch as u32,
        in_h as u32,
        in_w as u32,
        fft_h as u32,
        fft_w as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch pad_fft");
    fx.stream().synchronize().expect("sync");

    let mut gpu_re = vec![0.0f32; total as usize];
    let mut gpu_im = vec![0.0f32; total as usize];
    pr.copy_to_host(&mut gpu_re).expect("copy re");
    pi.copy_to_host(&mut gpu_im).expect("copy im");

    let mut exp_re = vec![0.0f32; total as usize];
    for gid in 0..(total as usize) {
        let batch_ch = gid / fft_area;
        let spatial = gid % fft_area;
        let row = spatial / fft_w;
        let col = spatial % fft_w;
        if row >= pad_h && row < pad_h + in_h && col >= pad_w && col < pad_w + in_w {
            let in_row = row - pad_h;
            let in_col = col - pad_w;
            exp_re[gid] = input[batch_ch * in_h * in_w + in_row * in_w + in_col];
        }
    }
    assert_close_f32(&gpu_re, &exp_re, 0.0, 0.0, "fft_pad_re");
    assert_close_f32(
        &gpu_im,
        &vec![0.0f32; total as usize],
        0.0,
        0.0,
        "fft_pad_im",
    );
}

#[test]
fn fft_ifft_and_crop_gather_scale_matches_cpu() {
    // FRAGMENT: no inverse FFT is emitted. The kernel gathers
    // freq_re[oh*stride_h, ow*stride_w] and applies the 1/N scale. We assert
    // exactly that partial operation (it is NOT a real IFFT).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = FftConv2dPlan::new(1, 3, 3, 3, 1, 1, 0, 0, sm_to_numeric(fx.sm), PtxType::F32)
        .expect("fft plan");
    let (batch, out_ch, in_h, in_w) = (2usize, 3usize, 4usize, 4usize);
    let (fft_h, fft_w) = plan.fft_size(in_h as u32, in_w as u32).expect("fft size");
    let (fft_h, fft_w) = (fft_h as usize, fft_w as usize);
    let (out_h, out_w) = plan.output_size(in_h as u32, in_w as u32);
    let (out_h, out_w) = (out_h as usize, out_w as usize);
    let (stride_h, stride_w) = (1usize, 1usize);
    let fft_area = fft_h * fft_w;
    let total = (batch * out_ch * out_h * out_w) as u32;
    let inv_n = 1.0f32 / (fft_h * fft_w) as f32;

    let freq_re = rand_f32(batch * out_ch * fft_area, 0xfc_01, -1.0, 1.0);
    let fr = DeviceBuffer::from_host(&freq_re).expect("freq_re");
    let fi = DeviceBuffer::from_host(&vec![0.0f32; batch * out_ch * fft_area]).expect("freq_im");
    let tw = DeviceBuffer::from_host(&vec![0.0f32; fft_h + fft_w]).expect("twiddle");
    let out_buf = DeviceBuffer::from_host(&vec![-987.0f32; total as usize]).expect("output");

    let ptx = plan
        .generate_ifft_and_crop(in_h as u32, in_w as u32)
        .expect("ifft_crop ptx");
    ptxas_assembles(&ptx, "fft_ifft_and_crop_f32").expect("ptxas ifft_crop");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        fr.as_device_ptr(),
        fi.as_device_ptr(),
        out_buf.as_device_ptr(),
        tw.as_device_ptr(),
        tw.as_device_ptr(),
        batch as u32,
        out_ch as u32,
        fft_h as u32,
        fft_w as u32,
        out_h as u32,
        out_w as u32,
        total,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch ifft_crop");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; total as usize];
    out_buf.copy_to_host(&mut gpu).expect("copy out");

    let mut exp = vec![0.0f32; total as usize];
    let out_area = out_h * out_w;
    for gid in 0..(total as usize) {
        let batch_idx = gid / (out_ch * out_area);
        let r1 = gid % (out_ch * out_area);
        let oc = r1 / out_area;
        let r2 = r1 % out_area;
        let oh = r2 / out_w;
        let ow = r2 % out_w;
        let fft_row = oh * stride_h;
        let fft_col = ow * stride_w;
        let freq_idx = (batch_idx * out_ch + oc) * fft_area + fft_row * fft_w + fft_col;
        exp[gid] = freq_re[freq_idx] * inv_n;
    }
    assert_close_f32(&gpu, &exp, 1e-6, 1e-7, "fft_ifft_crop_gather");
}

// ===========================================================================
// dgrad / wgrad fragments — implicit-GEMM + Winograd transforms (load + canary)
// ===========================================================================

#[test]
fn dgrad_implicit_gemm_fragment_runs_clean() {
    // FRAGMENT: body is comment-only + ret (no stores). Assert it assembles,
    // launches and leaves grad_input untouched (a canary for a future body).
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let problem = ConvProblem {
        batch: 1,
        in_channels: 4,
        in_dims: vec![6, 6],
        out_channels: 4,
        filter_dims: vec![3, 3],
        padding: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    };
    let (oh, ow) = (6u32, 6u32);
    let gi_n = (problem.batch * problem.in_channels * 6 * 6) as usize;
    let go_n = (problem.batch * problem.out_channels * oh * ow) as usize;
    let f_n = (problem.out_channels * problem.in_channels * 9) as usize;

    let go = DeviceBuffer::from_host(&vec![0.5f32; go_n]).expect("grad_out");
    let filter = DeviceBuffer::from_host(&vec![0.25f32; f_n]).expect("filter");
    let gi = DeviceBuffer::from_host(&vec![-987.0f32; gi_n]).expect("grad_input");

    let eng = DgradImplicitGemm::new(problem.clone(), fx.sm);
    let ptx = eng.generate_ptx().expect("dgrad ptx");
    ptxas_assembles(&ptx, "dgrad_implicit_gemm").expect("ptxas dgrad");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let total = gi_n as u32;
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        go.as_device_ptr(),
        filter.as_device_ptr(),
        gi.as_device_ptr(),
        problem.batch,
        problem.in_channels,
        6u32,
        6u32,
        problem.out_channels,
        3u32,
        3u32,
        oh,
        ow,
        1u32,
        1u32,
        1u32,
        1u32,
        1u32,
        1u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch dgrad");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&gi, -987.0, gi_n, "dgrad_implicit_gemm");
}

#[test]
fn wgrad_implicit_gemm_fragment_runs_clean() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let problem = ConvProblem {
        batch: 1,
        in_channels: 4,
        in_dims: vec![6, 6],
        out_channels: 4,
        filter_dims: vec![3, 3],
        padding: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    };
    let (oh, ow) = (6u32, 6u32);
    let in_n = (problem.batch * problem.in_channels * 6 * 6) as usize;
    let go_n = (problem.batch * problem.out_channels * oh * ow) as usize;
    let gf_n = (problem.out_channels * problem.in_channels * 9) as usize;

    let inp = DeviceBuffer::from_host(&vec![0.5f32; in_n]).expect("input");
    let go = DeviceBuffer::from_host(&vec![0.25f32; go_n]).expect("grad_out");
    let gf = DeviceBuffer::from_host(&vec![-987.0f32; gf_n]).expect("grad_filter");

    let eng = WgradImplicitGemm::new(problem.clone(), fx.sm);
    let ptx = eng.generate_ptx().expect("wgrad ptx");
    ptxas_assembles(&ptx, "wgrad_implicit_gemm").expect("ptxas wgrad");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let total = gf_n as u32;
    let params = LaunchParams::new(ceil_div(total, 64), 64u32);
    let args = (
        inp.as_device_ptr(),
        go.as_device_ptr(),
        gf.as_device_ptr(),
        problem.batch,
        problem.in_channels,
        6u32,
        6u32,
        problem.out_channels,
        3u32,
        3u32,
        oh,
        ow,
        1u32,
        1u32,
        1u32,
        1u32,
        1u32,
        1u32,
    );
    kernel
        .launch(&params, fx.stream(), &args)
        .expect("launch wgrad");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&gf, -987.0, gf_n, "wgrad_implicit_gemm");
}

fn frag_problem() -> ConvProblem {
    ConvProblem {
        batch: 1,
        in_channels: 4,
        in_dims: vec![8, 8],
        out_channels: 4,
        filter_dims: vec![3, 3],
        padding: vec![1, 1],
        stride: vec![1, 1],
        dilation: vec![1, 1],
        groups: 1,
        input_type: PtxType::F32,
        output_type: PtxType::F32,
        layout: TensorLayout::Nchw,
    }
}

#[test]
fn winograd_dgrad_output_transform_fragment_runs_clean() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let eng = WinogradDgrad::new(frag_problem(), fx.sm).expect("winograd dgrad");
    let ptx = eng.generate_grad_output_transform_ptx().expect("ptx");
    ptxas_assembles(&ptx, "winograd_dgrad_output_transform").expect("ptxas");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let num_tiles = 16u32;
    let n = 64usize;
    let go = DeviceBuffer::from_host(&vec![0.5f32; n]).expect("grad_out");
    let tr = DeviceBuffer::from_host(&vec![-987.0f32; n]).expect("transformed");
    let params = LaunchParams::new(ceil_div(num_tiles, 64), 64u32);
    let args = (
        go.as_device_ptr(),
        tr.as_device_ptr(),
        1u32,
        4u32,
        8u32,
        8u32,
        8u32,
        8u32,
        1u32,
        1u32,
        num_tiles,
    );
    kernel.launch(&params, fx.stream(), &args).expect("launch");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&tr, -987.0, n, "winograd_dgrad_output_transform");
}

#[test]
fn winograd_dgrad_input_transform_fragment_runs_clean() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let eng = WinogradDgrad::new(frag_problem(), fx.sm).expect("winograd dgrad");
    let ptx = eng.generate_grad_input_transform_ptx().expect("ptx");
    ptxas_assembles(&ptx, "winograd_dgrad_input_transform").expect("ptxas");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let num_tiles = 16u32;
    let n = 64usize;
    let tr = DeviceBuffer::from_host(&vec![0.5f32; n]).expect("transformed");
    let gi = DeviceBuffer::from_host(&vec![-987.0f32; n]).expect("grad_input");
    let params = LaunchParams::new(ceil_div(num_tiles, 64), 64u32);
    let args = (
        tr.as_device_ptr(),
        gi.as_device_ptr(),
        1u32,
        4u32,
        8u32,
        8u32,
        num_tiles,
    );
    kernel.launch(&params, fx.stream(), &args).expect("launch");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&gi, -987.0, n, "winograd_dgrad_input_transform");
}

#[test]
fn winograd_wgrad_input_transform_fragment_runs_clean() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let eng = WinogradWgrad::new(frag_problem(), fx.sm).expect("winograd wgrad");
    let ptx = eng.generate_input_transform_ptx().expect("ptx");
    ptxas_assembles(&ptx, "winograd_wgrad_input_transform").expect("ptxas");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let num_tiles = 16u32;
    let n = 64usize;
    let inp = DeviceBuffer::from_host(&vec![0.5f32; n]).expect("input");
    let tr = DeviceBuffer::from_host(&vec![-987.0f32; n]).expect("transformed");
    let params = LaunchParams::new(ceil_div(num_tiles, 64), 64u32);
    let args = (
        inp.as_device_ptr(),
        tr.as_device_ptr(),
        1u32,
        4u32,
        8u32,
        8u32,
        8u32,
        8u32,
        1u32,
        1u32,
        num_tiles,
    );
    kernel.launch(&params, fx.stream(), &args).expect("launch");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&tr, -987.0, n, "winograd_wgrad_input_transform");
}

#[test]
fn winograd_wgrad_grad_output_transform_fragment_runs_clean() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let eng = WinogradWgrad::new(frag_problem(), fx.sm).expect("winograd wgrad");
    let ptx = eng.generate_grad_output_transform_ptx().expect("ptx");
    ptxas_assembles(&ptx, "winograd_wgrad_grad_output_transform").expect("ptxas");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let num_tiles = 16u32;
    let n = 64usize;
    let go = DeviceBuffer::from_host(&vec![0.5f32; n]).expect("grad_out");
    let tr = DeviceBuffer::from_host(&vec![-987.0f32; n]).expect("transformed");
    let params = LaunchParams::new(ceil_div(num_tiles, 64), 64u32);
    let args = (
        go.as_device_ptr(),
        tr.as_device_ptr(),
        1u32,
        4u32,
        8u32,
        8u32,
        1u32,
        1u32,
        num_tiles,
    );
    kernel.launch(&params, fx.stream(), &args).expect("launch");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&tr, -987.0, n, "winograd_wgrad_grad_output_transform");
}

#[test]
fn winograd_wgrad_filter_transform_fragment_runs_clean() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let eng = WinogradWgrad::new(frag_problem(), fx.sm).expect("winograd wgrad");
    let ptx = eng.generate_filter_grad_transform_ptx().expect("ptx");
    ptxas_assembles(&ptx, "winograd_wgrad_filter_transform").expect("ptxas");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);
    let num_filters = 16u32;
    let n = 64usize;
    let gfw = DeviceBuffer::from_host(&vec![0.5f32; n]).expect("grad_filter_wino");
    let gf = DeviceBuffer::from_host(&vec![-987.0f32; n]).expect("grad_filter");
    let params = LaunchParams::new(ceil_div(num_filters, 64), 64u32);
    let args = (
        gfw.as_device_ptr(),
        gf.as_device_ptr(),
        4u32,
        4u32,
        num_filters,
    );
    kernel.launch(&params, fx.stream(), &args).expect("launch");
    fx.stream().synchronize().expect("sync");
    assert_untouched_f32(&gf, -987.0, n, "winograd_wgrad_filter_transform");
}
