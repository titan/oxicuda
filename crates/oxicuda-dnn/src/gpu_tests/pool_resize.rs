//! On-device GPU validation for the `pool_resize` subsystem of `oxicuda-dnn`.
//!
//! Each test drives a public pooling / resize operation (`avg_pool2d`,
//! `max_pool2d`, `global_*`, `adaptive_*`, `resize_*`) on the live CUDA
//! device and compares the result against an independent CPU re-derivation.
//!
//! Four of the ten generate functions in this cluster emitted PTX that
//! `ptxas` rejected before this pass (the `{true}` predicate literal in
//! avg/max-pool, and `[smem + reg*imm]` shared-memory operands in the global
//! reductions). Those bugs are fixed in the owned source, so every kernel
//! here now assembles and is validated numerically end-to-end.

use super::*;

use oxicuda_blas::GpuFloat;
use oxicuda_memory::DeviceBuffer;

use crate::types::{TensorDesc, TensorDescMut};

/// `(N, C, H, W)` shape tuple.
type Dims4 = (u32, u32, u32, u32);
/// `(H, W)` spatial-extent tuple.
type Dims2 = (u32, u32);

/// Full geometry of a windowed (avg/max) pooling op.
#[derive(Clone, Copy)]
struct PoolGeom {
    n: u32,
    c: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    ph: u32,
    pw: u32,
}

// ---------------------------------------------------------------------------
// Small numeric helper trait so the device runners stay generic over T.
// ---------------------------------------------------------------------------

/// A float usable in these tests: device-storable plus host conversions.
trait TestFloat: GpuFloat {
    fn zero() -> Self;
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl TestFloat for f32 {
    fn zero() -> Self {
        0.0
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl TestFloat for f64 {
    fn zero() -> Self {
        0.0
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Builds an input vector of `n` deterministic pseudo-random values in
/// `[lo, hi)` for the given float type.
fn make_input<T: TestFloat>(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<T> {
    let mut rng = Lcg::new(seed);
    (0..n).map(|_| T::from_f64(rng.range_f64(lo, hi))).collect()
}

/// Widens an `f32` slice to `f64` (lossless).
fn widen(data: &[f32]) -> Vec<f64> {
    data.iter().map(|&x| f64::from(x)).collect()
}

// ---------------------------------------------------------------------------
// Generic device runner for the "one input desc -> one output desc" ops.
// ---------------------------------------------------------------------------

/// Runs an NCHW op that maps a single input descriptor to a single output
/// descriptor, returning the device result widened to `f64`.
fn gpu_run<T, F>(fx: &GpuFixture, in_data: &[T], in_dims: Dims4, out_dims: Dims2, op: F) -> Vec<f64>
where
    T: TestFloat,
    F: FnOnce(&DnnHandle, &TensorDesc<T>, &mut TensorDescMut<T>),
{
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let n_out = (n * c * oh * ow) as usize;

    let in_buf = DeviceBuffer::<T>::from_host(in_data).expect("input alloc");
    let mut out_buf = DeviceBuffer::<T>::zeroed(n_out).expect("output alloc");

    let in_desc = TensorDesc::<T>::nchw(&in_buf, n, c, ih, iw).expect("input desc");
    let mut out_desc = TensorDescMut::<T>::nchw(&mut out_buf, n, c, oh, ow).expect("output desc");

    op(&fx.handle, &in_desc, &mut out_desc);
    fx.stream().synchronize().expect("stream sync");

    let mut host = vec![T::zero(); n_out];
    out_buf.copy_to_host(&mut host).expect("copy back");
    host.into_iter().map(TestFloat::to_f64).collect()
}

// ===========================================================================
// CPU oracles (independent re-derivations of each kernel's math).
// ===========================================================================

fn avg_pool_oracle(inp: &[f64], g: PoolGeom, count_include_pad: bool) -> Vec<f64> {
    let plane = (g.ih * g.iw) as usize;
    let at = |nn: u32, cc: u32, hh: u32, ww: u32| -> usize {
        ((nn * g.c + cc) as usize) * plane + (hh * g.iw + ww) as usize
    };
    let mut out = vec![0.0f64; (g.n * g.c * g.oh * g.ow) as usize];
    for nn in 0..g.n {
        for cc in 0..g.c {
            for oh_i in 0..g.oh {
                for ow_i in 0..g.ow {
                    let h0 = oh_i as i32 * g.sh as i32 - g.ph as i32;
                    let w0 = ow_i as i32 * g.sw as i32 - g.pw as i32;
                    let mut sum = 0.0f64;
                    let mut cnt = 0u32;
                    for hh in h0..h0 + g.kh as i32 {
                        for ww in w0..w0 + g.kw as i32 {
                            let in_bounds =
                                hh >= 0 && hh < g.ih as i32 && ww >= 0 && ww < g.iw as i32;
                            if count_include_pad {
                                cnt += 1;
                            }
                            if in_bounds {
                                sum += inp[at(nn, cc, hh as u32, ww as u32)];
                                if !count_include_pad {
                                    cnt += 1;
                                }
                            }
                        }
                    }
                    let o = ((nn * g.c + cc) * g.oh * g.ow + oh_i * g.ow + ow_i) as usize;
                    out[o] = sum / cnt as f64;
                }
            }
        }
    }
    out
}

fn max_pool_oracle(inp: &[f64], g: PoolGeom) -> (Vec<f64>, Vec<i32>) {
    let plane = (g.ih * g.iw) as usize;
    let at = |nn: u32, cc: u32, hh: u32, ww: u32| -> usize {
        ((nn * g.c + cc) as usize) * plane + (hh * g.iw + ww) as usize
    };
    let n_out = (g.n * g.c * g.oh * g.ow) as usize;
    let mut vals = vec![0.0f64; n_out];
    let mut idxs = vec![-1i32; n_out];
    for nn in 0..g.n {
        for cc in 0..g.c {
            for oh_i in 0..g.oh {
                for ow_i in 0..g.ow {
                    let h0 = oh_i as i32 * g.sh as i32 - g.ph as i32;
                    let w0 = ow_i as i32 * g.sw as i32 - g.pw as i32;
                    let mut best = f64::NEG_INFINITY;
                    let mut best_idx = -1i32;
                    for hh in h0..h0 + g.kh as i32 {
                        for ww in w0..w0 + g.kw as i32 {
                            if hh >= 0 && hh < g.ih as i32 && ww >= 0 && ww < g.iw as i32 {
                                let v = inp[at(nn, cc, hh as u32, ww as u32)];
                                if v > best {
                                    best = v;
                                    best_idx = (hh as u32 * g.iw + ww as u32) as i32;
                                }
                            }
                        }
                    }
                    let o = ((nn * g.c + cc) * g.oh * g.ow + oh_i * g.ow + ow_i) as usize;
                    vals[o] = best;
                    idxs[o] = best_idx;
                }
            }
        }
    }
    (vals, idxs)
}

fn max_backward_oracle(grad_out: &[f64], idx: &[i32], in_dims: Dims4, out_dims: Dims2) -> Vec<f64> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let in_hw = (ih * iw) as usize;
    let out_hw = (oh * ow) as usize;
    let mut grad_in = vec![0.0f64; (n * c) as usize * in_hw];
    for (gid, &g) in grad_out.iter().enumerate() {
        let hw_idx = idx[gid];
        if hw_idx >= 0 {
            let nc = gid / out_hw;
            grad_in[nc * in_hw + hw_idx as usize] += g;
        }
    }
    grad_in
}

fn global_avg_oracle(inp: &[f64], nc: u32, hw: u32) -> Vec<f64> {
    (0..nc as usize)
        .map(|p| {
            let s: f64 = inp[p * hw as usize..(p + 1) * hw as usize].iter().sum();
            s / hw as f64
        })
        .collect()
}

fn global_max_oracle(inp: &[f64], nc: u32, hw: u32) -> Vec<f64> {
    (0..nc as usize)
        .map(|p| {
            inp[p * hw as usize..(p + 1) * hw as usize]
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

fn adaptive_avg_oracle(inp: &[f64], in_dims: Dims4, out_dims: Dims2) -> Vec<f64> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let plane = (ih * iw) as usize;
    let at = |p: u32, hh: u32, ww: u32| -> usize { p as usize * plane + (hh * iw + ww) as usize };
    let mut out = vec![0.0f64; (n * c * oh * ow) as usize];
    for p in 0..n * c {
        for oh_i in 0..oh {
            let h_start = oh_i * ih / oh;
            let h_end = (oh_i + 1) * ih / oh;
            for ow_i in 0..ow {
                let w_start = ow_i * iw / ow;
                let w_end = (ow_i + 1) * iw / ow;
                let mut sum = 0.0f64;
                let mut cnt = 0u32;
                for hh in h_start..h_end {
                    for ww in w_start..w_end {
                        sum += inp[at(p, hh, ww)];
                        cnt += 1;
                    }
                }
                out[(p * oh * ow + oh_i * ow + ow_i) as usize] = sum / cnt as f64;
            }
        }
    }
    out
}

fn adaptive_max_oracle(inp: &[f64], in_dims: Dims4, out_dims: Dims2) -> Vec<f64> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let plane = (ih * iw) as usize;
    let at = |p: u32, hh: u32, ww: u32| -> usize { p as usize * plane + (hh * iw + ww) as usize };
    let mut out = vec![0.0f64; (n * c * oh * ow) as usize];
    for p in 0..n * c {
        for oh_i in 0..oh {
            let h_start = oh_i * ih / oh;
            let h_end = (oh_i + 1) * ih / oh;
            for ow_i in 0..ow {
                let w_start = ow_i * iw / ow;
                let w_end = (ow_i + 1) * iw / ow;
                let mut best = f64::NEG_INFINITY;
                for hh in h_start..h_end {
                    for ww in w_start..w_end {
                        best = best.max(inp[at(p, hh, ww)]);
                    }
                }
                out[(p * oh * ow + oh_i * ow + ow_i) as usize] = best;
            }
        }
    }
    out
}

fn nearest_oracle(inp: &[f64], in_dims: Dims4, out_dims: Dims2) -> Vec<f64> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let plane = (ih * iw) as usize;
    let at = |p: u32, hh: u32, ww: u32| -> usize { p as usize * plane + (hh * iw + ww) as usize };
    let mut out = vec![0.0f64; (n * c * oh * ow) as usize];
    for p in 0..n * c {
        for oh_i in 0..oh {
            let src_h = oh_i * ih / oh;
            for ow_i in 0..ow {
                let src_w = ow_i * iw / ow;
                out[(p * oh * ow + oh_i * ow + ow_i) as usize] = inp[at(p, src_h, src_w)];
            }
        }
    }
    out
}

fn bilinear_oracle_f32(
    inp: &[f32],
    in_dims: Dims4,
    out_dims: Dims2,
    align_corners: bool,
) -> Vec<f32> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let plane = (ih * iw) as usize;
    let at = |p: u32, hh: u32, ww: u32| -> usize { p as usize * plane + (hh * iw + ww) as usize };
    let mut out = vec![0.0f32; (n * c * oh * ow) as usize];
    for p in 0..n * c {
        for oh_i in 0..oh {
            for ow_i in 0..ow {
                let (src_h, src_w) = if align_corners {
                    let sh = (ih - 1) as f32 / (oh - 1) as f32;
                    let sw = (iw - 1) as f32 / (ow - 1) as f32;
                    (oh_i as f32 * sh, ow_i as f32 * sw)
                } else {
                    let h = (oh_i as f32 + 0.5) * ih as f32 / oh as f32 - 0.5;
                    let w = (ow_i as f32 + 0.5) * iw as f32 / ow as f32 - 0.5;
                    (h, w)
                };
                let src_h = src_h.max(0.0);
                let src_w = src_w.max(0.0);
                let h0f = src_h.floor();
                let w0f = src_w.floor();
                let h0 = h0f as u32;
                let w0 = w0f as u32;
                let fh = src_h - h0f;
                let fw = src_w - w0f;
                let h1 = (h0 + 1).min(ih - 1);
                let h0c = h0.min(ih - 1);
                let w1 = (w0 + 1).min(iw - 1);
                let w0c = w0.min(iw - 1);

                let v00 = inp[at(p, h0c, w0c)];
                let v01 = inp[at(p, h0c, w1)];
                let v10 = inp[at(p, h1, w0c)];
                let v11 = inp[at(p, h1, w1)];

                let one_mfh = 1.0f32 - fh;
                let one_mfw = 1.0f32 - fw;
                let w00 = one_mfh * one_mfw;
                let w01 = one_mfh * fw;
                let w10 = fh * one_mfw;
                let w11 = fh * fw;

                // Mirror the kernel's fused-multiply-add accumulation order.
                let t0 = w00 * v00;
                let t1 = w01.mul_add(v01, t0);
                let t2 = w10.mul_add(v10, t1);
                let res = w11.mul_add(v11, t2);
                out[(p * oh * ow + oh_i * ow + ow_i) as usize] = res;
            }
        }
    }
    out
}

/// Keys cubic convolution weight for `|x| = t`, `a = -0.75`, mirroring the
/// kernel's evaluation order.
fn cubic_weight(t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let wn = (-2.25f32).mul_add(t2, 1.25f32 * t3) + 1.0;
    let wf = {
        let a = -0.75f32 * t3;
        let a = 3.75f32.mul_add(t2, a);
        let a = (-6.0f32).mul_add(t, a);
        a + 3.0
    };
    let w_far = if t < 2.0 { wf } else { 0.0 };
    if t <= 1.0 { wn } else { w_far }
}

fn bicubic_oracle_f32(
    inp: &[f32],
    in_dims: Dims4,
    out_dims: Dims2,
    align_corners: bool,
) -> Vec<f32> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let plane = (ih * iw) as usize;
    let at = |p: u32, hh: u32, ww: u32| -> usize { p as usize * plane + (hh * iw + ww) as usize };
    let mut out = vec![0.0f32; (n * c * oh * ow) as usize];
    for p in 0..n * c {
        for oh_i in 0..oh {
            for ow_i in 0..ow {
                let (src_h, src_w) = if align_corners {
                    let sh = (ih - 1) as f32 / (oh - 1) as f32;
                    let sw = (iw - 1) as f32 / (ow - 1) as f32;
                    (oh_i as f32 * sh, ow_i as f32 * sw)
                } else {
                    let h = (oh_i as f32 + 0.5) * ih as f32 / oh as f32 - 0.5;
                    let w = (ow_i as f32 + 0.5) * iw as f32 / ow as f32 - 0.5;
                    (h, w)
                };
                let h_floor = src_h.floor();
                let w_floor = src_w.floor();
                let frac_h = src_h - h_floor;
                let frac_w = src_w - w_floor;
                let h_center = h_floor as i32;
                let w_center = w_floor as i32;

                let mut res = 0.0f32;
                for dy in -1i32..=2 {
                    let w_h = cubic_weight((frac_h - dy as f32).abs());
                    let sy = (h_center + dy).clamp(0, ih as i32 - 1) as u32;
                    for dx in -1i32..=2 {
                        let w_w = cubic_weight((frac_w - dx as f32).abs());
                        let sx = (w_center + dx).clamp(0, iw as i32 - 1) as u32;
                        let weight = w_h * w_w;
                        res += weight * inp[at(p, sy, sx)];
                    }
                }
                out[(p * oh * ow + oh_i * ow + ow_i) as usize] = res;
            }
        }
    }
    out
}

// ===========================================================================
// Tests: average pooling
// ===========================================================================

#[test]
fn avg_pool2d_cip_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 2,
        c: 3,
        ih: 5,
        iw: 4,
        oh: 3,
        ow: 3,
        kh: 2,
        kw: 2,
        sh: 2,
        sw: 2,
        ph: 1,
        pw: 1,
    };
    let input = make_input::<f32>((g.n * g.c * g.ih * g.iw) as usize, 0x51, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(
        &fx,
        &input,
        (g.n, g.c, g.ih, g.iw),
        (g.oh, g.ow),
        |h, i, o| {
            crate::pool::avg_pool::avg_pool2d(
                h,
                i,
                o,
                (g.kh, g.kw),
                (g.sh, g.sw),
                (g.ph, g.pw),
                true,
            )
            .expect("avg_pool2d cip");
        },
    );

    let cpu = avg_pool_oracle(&widen(&input), g, true);
    assert_close_f64(&gpu, &cpu, 1e-4, 1e-5, "avg_pool2d_cip_f32");
}

#[test]
fn avg_pool2d_nocip_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 2,
        c: 3,
        ih: 5,
        iw: 4,
        oh: 5,
        ow: 4,
        kh: 3,
        kw: 3,
        sh: 1,
        sw: 1,
        ph: 1,
        pw: 1,
    };
    let input = make_input::<f32>((g.n * g.c * g.ih * g.iw) as usize, 0x52, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(
        &fx,
        &input,
        (g.n, g.c, g.ih, g.iw),
        (g.oh, g.ow),
        |h, i, o| {
            crate::pool::avg_pool::avg_pool2d(
                h,
                i,
                o,
                (g.kh, g.kw),
                (g.sh, g.sw),
                (g.ph, g.pw),
                false,
            )
            .expect("avg_pool2d nocip");
        },
    );

    let cpu = avg_pool_oracle(&widen(&input), g, false);
    assert_close_f64(&gpu, &cpu, 1e-4, 1e-5, "avg_pool2d_nocip_f32");
}

#[test]
fn avg_pool2d_cip_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 1,
        c: 2,
        ih: 6,
        iw: 5,
        oh: 3,
        ow: 2,
        kh: 2,
        kw: 2,
        sh: 2,
        sw: 2,
        ph: 0,
        pw: 0,
    };
    let input = make_input::<f64>((g.n * g.c * g.ih * g.iw) as usize, 0x53, -3.0, 3.0);

    let gpu = gpu_run::<f64, _>(
        &fx,
        &input,
        (g.n, g.c, g.ih, g.iw),
        (g.oh, g.ow),
        |h, i, o| {
            crate::pool::avg_pool::avg_pool2d(
                h,
                i,
                o,
                (g.kh, g.kw),
                (g.sh, g.sw),
                (g.ph, g.pw),
                true,
            )
            .expect("avg_pool2d cip f64");
        },
    );

    let cpu = avg_pool_oracle(&input, g, true);
    assert_close_f64(&gpu, &cpu, 1e-12, 1e-12, "avg_pool2d_cip_f64");
}

// ===========================================================================
// Tests: max pooling (forward, indices) + backward
// ===========================================================================

/// Forward max-pool on the device; returns (values widened to f64, indices).
fn gpu_max_pool_f32(
    fx: &GpuFixture,
    input: &[f32],
    g: PoolGeom,
    want_idx: bool,
) -> (Vec<f64>, Vec<i32>) {
    let n_out = (g.n * g.c * g.oh * g.ow) as usize;
    let in_buf = DeviceBuffer::<f32>::from_host(input).expect("input alloc");
    let mut out_buf = DeviceBuffer::<f32>::zeroed(n_out).expect("output alloc");
    let mut idx_buf = if want_idx {
        Some(DeviceBuffer::<i32>::zeroed(n_out).expect("idx alloc"))
    } else {
        None
    };

    let in_desc = TensorDesc::<f32>::nchw(&in_buf, g.n, g.c, g.ih, g.iw).expect("input desc");
    let mut out_desc =
        TensorDescMut::<f32>::nchw(&mut out_buf, g.n, g.c, g.oh, g.ow).expect("output desc");

    crate::pool::max_pool::max_pool2d(
        &fx.handle,
        &in_desc,
        &mut out_desc,
        idx_buf.as_mut(),
        (g.kh, g.kw),
        (g.sh, g.sw),
        (g.ph, g.pw),
    )
    .expect("max_pool2d");
    fx.stream().synchronize().expect("stream sync");

    let mut vals = vec![0.0f32; n_out];
    out_buf.copy_to_host(&mut vals).expect("copy values");
    let idxs = if let Some(ref ib) = idx_buf {
        let mut v = vec![0i32; n_out];
        ib.copy_to_host(&mut v).expect("copy indices");
        v
    } else {
        Vec::new()
    };
    (vals.into_iter().map(f64::from).collect(), idxs)
}

#[test]
fn max_pool2d_values_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 2,
        c: 3,
        ih: 5,
        iw: 4,
        oh: 3,
        ow: 3,
        kh: 2,
        kw: 2,
        sh: 2,
        sw: 2,
        ph: 1,
        pw: 1,
    };
    let input = make_input::<f32>((g.n * g.c * g.ih * g.iw) as usize, 0x61, -5.0, 5.0);

    let (gpu, _idx) = gpu_max_pool_f32(&fx, &input, g, false);

    let (cpu, _) = max_pool_oracle(&widen(&input), g);
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "max_pool2d_values_f32");
}

#[test]
fn max_pool2d_indices_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 2,
        c: 2,
        ih: 6,
        iw: 6,
        oh: 3,
        ow: 3,
        kh: 3,
        kw: 3,
        sh: 2,
        sw: 2,
        ph: 1,
        pw: 1,
    };
    let input = make_input::<f32>((g.n * g.c * g.ih * g.iw) as usize, 0x62, -5.0, 5.0);

    let (gpu, idx) = gpu_max_pool_f32(&fx, &input, g, true);

    let (cpu, cpu_idx) = max_pool_oracle(&widen(&input), g);
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "max_pool2d_indices_values");
    assert_eq!(
        idx, cpu_idx,
        "max_pool2d_indices_f32: argmax index mismatch"
    );
}

/// Backward max-pool on the device; returns grad_input widened to f64.
fn gpu_max_pool_backward<T: TestFloat>(
    fx: &GpuFixture,
    grad_out: &[T],
    idx: &[i32],
    in_dims: Dims4,
    out_dims: Dims2,
) -> Vec<f64> {
    let (n, c, ih, iw) = in_dims;
    let (oh, ow) = out_dims;
    let go_buf = DeviceBuffer::<T>::from_host(grad_out).expect("grad_out alloc");
    let idx_buf = DeviceBuffer::<i32>::from_host(idx).expect("idx alloc");
    let mut gi_buf = DeviceBuffer::<T>::zeroed((n * c * ih * iw) as usize).expect("grad_in alloc");

    let go_desc = TensorDesc::<T>::nchw(&go_buf, n, c, oh, ow).expect("grad_out desc");
    let mut gi_desc = TensorDescMut::<T>::nchw(&mut gi_buf, n, c, ih, iw).expect("grad_in desc");

    crate::pool::max_pool::max_pool2d_backward(&fx.handle, &go_desc, &idx_buf, &mut gi_desc)
        .expect("max_pool2d_backward");
    fx.stream().synchronize().expect("stream sync");

    let mut host = vec![T::zero(); (n * c * ih * iw) as usize];
    gi_buf.copy_to_host(&mut host).expect("copy grad_in");
    host.into_iter().map(TestFloat::to_f64).collect()
}

#[test]
fn max_pool2d_backward_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 2,
        c: 2,
        ih: 6,
        iw: 6,
        oh: 3,
        ow: 3,
        kh: 3,
        kw: 3,
        sh: 2,
        sw: 2,
        ph: 1,
        pw: 1,
    };
    let input = make_input::<f32>((g.n * g.c * g.ih * g.iw) as usize, 0x71, -5.0, 5.0);

    // Forward to obtain the device-produced argmax index buffer.
    let (_vals, idx) = gpu_max_pool_f32(&fx, &input, g, true);

    let grad_out = make_input::<f32>((g.n * g.c * g.oh * g.ow) as usize, 0x72, -1.0, 1.0);
    let gpu = gpu_max_pool_backward(&fx, &grad_out, &idx, (g.n, g.c, g.ih, g.iw), (g.oh, g.ow));

    let cpu = max_backward_oracle(
        &widen(&grad_out),
        &idx,
        (g.n, g.c, g.ih, g.iw),
        (g.oh, g.ow),
    );
    assert_close_f64(&gpu, &cpu, 1e-5, 1e-6, "max_pool2d_backward_f32");
}

#[test]
fn max_pool2d_backward_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let g = PoolGeom {
        n: 1,
        c: 3,
        ih: 5,
        iw: 5,
        oh: 2,
        ow: 2,
        kh: 2,
        kw: 2,
        sh: 2,
        sw: 2,
        ph: 0,
        pw: 0,
    };
    let input = make_input::<f32>((g.n * g.c * g.ih * g.iw) as usize, 0x73, -5.0, 5.0);

    // Indices come from an f32 forward pass; the values they reference are
    // independent of the gradient dtype.
    let (_vals, idx) = gpu_max_pool_f32(&fx, &input, g, true);

    let grad_out = make_input::<f64>((g.n * g.c * g.oh * g.ow) as usize, 0x74, -2.0, 2.0);
    let gpu = gpu_max_pool_backward(&fx, &grad_out, &idx, (g.n, g.c, g.ih, g.iw), (g.oh, g.ow));

    let cpu = max_backward_oracle(&grad_out, &idx, (g.n, g.c, g.ih, g.iw), (g.oh, g.ow));
    assert_close_f64(&gpu, &cpu, 1e-12, 1e-12, "max_pool2d_backward_f64");
}

// ===========================================================================
// Tests: global pooling (shared-memory tree reduction)
// ===========================================================================

#[test]
fn global_avg_pool2d_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (2u32, 3u32, 4u32, 5u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0x81, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (1, 1), |h, i, o| {
        crate::pool::global_pool::global_avg_pool2d(h, i, o).expect("global_avg_pool2d");
    });

    let cpu = global_avg_oracle(&widen(&input), n * c, ih * iw);
    assert_close_f64(&gpu, &cpu, 1e-4, 1e-5, "global_avg_pool2d_f32");
}

#[test]
fn global_avg_pool2d_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (2u32, 2u32, 7u32, 3u32);
    let input = make_input::<f64>((n * c * ih * iw) as usize, 0x82, -3.0, 3.0);

    let gpu = gpu_run::<f64, _>(&fx, &input, (n, c, ih, iw), (1, 1), |h, i, o| {
        crate::pool::global_pool::global_avg_pool2d(h, i, o).expect("global_avg_pool2d f64");
    });

    let cpu = global_avg_oracle(&input, n * c, ih * iw);
    assert_close_f64(&gpu, &cpu, 1e-12, 1e-12, "global_avg_pool2d_f64");
}

#[test]
fn global_max_pool2d_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (2u32, 3u32, 4u32, 5u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0x83, -5.0, 5.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (1, 1), |h, i, o| {
        crate::pool::global_pool::global_max_pool2d(h, i, o).expect("global_max_pool2d");
    });

    let cpu = global_max_oracle(&widen(&input), n * c, ih * iw);
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "global_max_pool2d_f32");
}

#[test]
fn global_max_pool2d_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (3u32, 2u32, 5u32, 6u32);
    let input = make_input::<f64>((n * c * ih * iw) as usize, 0x84, -4.0, 4.0);

    let gpu = gpu_run::<f64, _>(&fx, &input, (n, c, ih, iw), (1, 1), |h, i, o| {
        crate::pool::global_pool::global_max_pool2d(h, i, o).expect("global_max_pool2d f64");
    });

    let cpu = global_max_oracle(&input, n * c, ih * iw);
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "global_max_pool2d_f64");
}

// ===========================================================================
// Tests: adaptive pooling
// ===========================================================================

#[test]
fn adaptive_avg_pool2d_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 2u32, 8u32, 6u32);
    let (oh, ow) = (3u32, 4u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0x91, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::pool::adaptive_pool::adaptive_avg_pool2d(h, i, o, (oh, ow))
            .expect("adaptive_avg_pool2d");
    });

    let cpu = adaptive_avg_oracle(&widen(&input), (n, c, ih, iw), (oh, ow));
    assert_close_f64(&gpu, &cpu, 1e-4, 1e-5, "adaptive_avg_pool2d_f32");
}

#[test]
fn adaptive_avg_pool2d_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (2u32, 1u32, 7u32, 5u32);
    let (oh, ow) = (2u32, 3u32);
    let input = make_input::<f64>((n * c * ih * iw) as usize, 0x92, -3.0, 3.0);

    let gpu = gpu_run::<f64, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::pool::adaptive_pool::adaptive_avg_pool2d(h, i, o, (oh, ow))
            .expect("adaptive_avg_pool2d f64");
    });

    let cpu = adaptive_avg_oracle(&input, (n, c, ih, iw), (oh, ow));
    assert_close_f64(&gpu, &cpu, 1e-12, 1e-12, "adaptive_avg_pool2d_f64");
}

#[test]
fn adaptive_max_pool2d_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 2u32, 8u32, 6u32);
    let (oh, ow) = (3u32, 4u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0x93, -5.0, 5.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::pool::adaptive_pool::adaptive_max_pool2d(h, i, o, (oh, ow))
            .expect("adaptive_max_pool2d");
    });

    let cpu = adaptive_max_oracle(&widen(&input), (n, c, ih, iw), (oh, ow));
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "adaptive_max_pool2d_f32");
}

// ===========================================================================
// Tests: nearest-neighbour resize
// ===========================================================================

#[test]
fn resize_nearest_upsample_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 2u32, 3u32, 4u32);
    let (oh, ow) = (6u32, 5u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0xA1, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::resize::nearest::resize_nearest(h, i, o).expect("resize_nearest");
    });

    let cpu = nearest_oracle(&widen(&input), (n, c, ih, iw), (oh, ow));
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "resize_nearest_upsample_f32");
}

#[test]
fn resize_nearest_downsample_f64() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (2u32, 1u32, 8u32, 6u32);
    let (oh, ow) = (3u32, 2u32);
    let input = make_input::<f64>((n * c * ih * iw) as usize, 0xA2, -3.0, 3.0);

    let gpu = gpu_run::<f64, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::resize::nearest::resize_nearest(h, i, o).expect("resize_nearest f64");
    });

    let cpu = nearest_oracle(&input, (n, c, ih, iw), (oh, ow));
    assert_close_f64(&gpu, &cpu, 0.0, 0.0, "resize_nearest_downsample_f64");
}

// ===========================================================================
// Tests: bilinear resize (align_corners on / off)
// ===========================================================================

#[test]
fn resize_bilinear_align_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 1u32, 4u32, 4u32);
    let (oh, ow) = (8u32, 8u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0xB1, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::resize::bilinear::resize_bilinear(h, i, o, true).expect("resize_bilinear ac");
    });

    let cpu = widen(&bilinear_oracle_f32(&input, (n, c, ih, iw), (oh, ow), true));
    assert_close_f64(&gpu, &cpu, 1e-4, 1e-4, "resize_bilinear_align_f32");
}

#[test]
fn resize_bilinear_noalign_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 2u32, 5u32, 4u32);
    let (oh, ow) = (7u32, 9u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0xB2, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::resize::bilinear::resize_bilinear(h, i, o, false).expect("resize_bilinear noac");
    });

    let cpu = widen(&bilinear_oracle_f32(
        &input,
        (n, c, ih, iw),
        (oh, ow),
        false,
    ));
    assert_close_f64(&gpu, &cpu, 1e-4, 1e-4, "resize_bilinear_noalign_f32");
}

// ===========================================================================
// Tests: bicubic resize (align_corners on / off)
// ===========================================================================

#[test]
fn resize_bicubic_align_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 1u32, 5u32, 5u32);
    let (oh, ow) = (7u32, 7u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0xC1, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::resize::bicubic::resize_bicubic(h, i, o, true).expect("resize_bicubic ac");
    });

    let cpu = widen(&bicubic_oracle_f32(&input, (n, c, ih, iw), (oh, ow), true));
    assert_close_f64(&gpu, &cpu, 2e-4, 2e-4, "resize_bicubic_align_f32");
}

#[test]
fn resize_bicubic_noalign_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (n, c, ih, iw) = (1u32, 2u32, 6u32, 5u32);
    let (oh, ow) = (9u32, 8u32);
    let input = make_input::<f32>((n * c * ih * iw) as usize, 0xC2, -2.0, 2.0);

    let gpu = gpu_run::<f32, _>(&fx, &input, (n, c, ih, iw), (oh, ow), |h, i, o| {
        crate::resize::bicubic::resize_bicubic(h, i, o, false).expect("resize_bicubic noac");
    });

    let cpu = widen(&bicubic_oracle_f32(&input, (n, c, ih, iw), (oh, ow), false));
    assert_close_f64(&gpu, &cpu, 2e-4, 2e-4, "resize_bicubic_noalign_f32");
}
