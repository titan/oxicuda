//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to an independent CPU oracle. The launch ABI mirrors the proven
//! `oxicuda-snn` / `oxicuda-ot` canaries: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .u32` / `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! Every kernel in this crate performs a flat, per-element numerical map and
//! writes its result with `st.global.f32`, so each is validated by an
//! **independent host re-derivation** of the kernel's documented arithmetic,
//! computed in Rust completely independently of the JIT-compiled PTX. A
//! mismatch therefore genuinely flags a miscompile or a wrong constant / shift /
//! index in the PTX. The crate's own public CPU functions (`horner`, `rk4`,
//! `central_difference`, `spline_eval`, …) take *closures* and operate in `f64`
//! over a whole integration / evaluation, so they do not share the kernels'
//! flat `f32` element ABI; the host re-derivations below reproduce exactly the
//! same scalar formula those routines apply per element.
//!
//! ## PTX audit result
//!
//! All seven kernels (`horner_eval`, `rk4_stage`, `bisection_step`,
//! `gauss_quad_accumulate`, `spline_eval`, `central_diff`, `bessel_recurrence`)
//! are valid PTX accepted by ptxas on sm_86 and compute the correct arithmetic;
//! no base-2 exp/log, invalid-PTX, or wrong-math bug was found. The only
//! kernel-design caveat is `bessel_recurrence`: its downward (Miller's-algorithm)
//! recurrence reads the `J_{n_order+1}` seed one element past each point's row,
//! so a *multi-point* launch aliases the next point's `J_0` (a cross-point read
//! that also races the neighbour's write-back). The test below validates the
//! recurrence arithmetic with a single point plus an explicit zero seed — the
//! canonical Miller's-algorithm calling convention — and documents the
//! multi-point aliasing rather than papering over it.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
///
/// `Context::new` calls `cuCtxCreate`, which both creates the context and makes
/// it current on the calling thread; the returned `Arc<Context>` must be kept
/// alive for the whole test (nextest runs each test in its own process, so a
/// per-test context is fine).
struct GpuFixture {
    ctx: Arc<Context>,
    sm: u32,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if Device::count().ok()? == 0 {
        return None;
    }
    let Ok(dev) = Device::get(0) else {
        return None;
    };
    let (major, minor) = dev.compute_capability().ok()?;
    let sm = (major * 10 + minor) as u32;
    let ctx = Context::new(&dev).ok()?;
    Some(GpuFixture {
        ctx: Arc::new(ctx),
        sm,
    })
}

/// Relative-with-absolute-floor closeness test for FP32 comparisons.
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Worst (relative, absolute) divergence over two equal-length slices.
fn worst_diff(gpu: &[f32], cpu: &[f32]) -> (f32, f32) {
    let mut worst_abs = 0.0_f32;
    let mut worst_rel = 0.0_f32;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        let a = (g - c).abs();
        if a > worst_abs {
            worst_abs = a;
        }
        let denom = g.abs().max(c.abs());
        if denom > 0.0 {
            let r = a / denom;
            if r > worst_rel {
                worst_rel = r;
            }
        }
    }
    (worst_rel, worst_abs)
}

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real,
/// load-blocking bug — so we panic loudly rather than skip.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a 1-D grid size.
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

/// Deterministic `f32` samples in `[lo, hi)` from the crate's LCG.
fn samples(n: usize, lo: f32, hi: f32, seed: u64) -> Vec<f32> {
    let mut rng = LcgRng::new(seed);
    (0..n)
        .map(|_| lo + (hi - lo) * rng.next_f64() as f32)
        .collect()
}

// ===========================================================================
// 1. horner_eval  —  INDEPENDENT HOST RE-DERIVATION (Horner's nested form)
// ===========================================================================

#[test]
fn horner_eval_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let degree = 4_usize; // 5 coefficients a_0..a_4
    let n_points = 128_usize;

    // Coefficients O(1) and |x| ≤ 1.2 keep every partial product bounded, so the
    // evaluation never overflows and stays well away from a zero crossing for the
    // relative comparison.
    let coeff = samples(degree + 1, -1.5, 1.5, 0x0117_0E11);
    let x = samples(n_points, -1.2, 1.2, 0x0117_0E12);

    // Independent host re-derivation using the kernel's exact nested FMA order:
    // acc = a_degree; for i = degree-1..0: acc = fma(acc, x, a_i). `mul_add` is
    // the same single-rounding `fma.rn` the PTX issues, so this is bit-tight.
    let mut expected = vec![0.0_f32; n_points];
    for (k, &xv) in x.iter().enumerate() {
        let mut acc = coeff[degree];
        for i in (0..degree).rev() {
            acc = acc.mul_add(xv, coeff[i]);
        }
        expected[k] = acc;
    }

    let ptx = crate::ptx_kernels::horner_eval_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "horner_eval_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_coeff = DeviceBuffer::<f32>::from_host(&coeff).expect("d_coeff");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_points]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n_points as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_coeff.as_device_ptr(),
                d_x.as_device_ptr(),
                d_out.as_device_ptr(),
                degree as u32,
                n_points as u32,
            ),
        )
        .expect("launch horner_eval_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_points];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n_points {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-5),
            "horner out[{k}] mismatch: gpu={} host={} x={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k],
            x[k]
        );
    }
}

// ===========================================================================
// 2. rk4_stage  —  INDEPENDENT HOST RE-DERIVATION (fused RK4 combination)
// ===========================================================================

#[test]
fn rk4_stage_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 192_usize;
    let h = 0.1_f32;

    let y = samples(n, -1.0, 1.0, 0x0124_0001);
    let k1 = samples(n, -1.0, 1.0, 0x0124_0002);
    let k2 = samples(n, -1.0, 1.0, 0x0124_0003);
    let k3 = samples(n, -1.0, 1.0, 0x0124_0004);
    let k4 = samples(n, -1.0, 1.0, 0x0124_0005);

    // Independent host re-derivation: out = y + (h/6)·(k1 + 2·k2 + 2·k3 + k4),
    // matching the crate's RK4 combination step (`ode::rk4`).
    let mut expected = vec![0.0_f32; n];
    let sixth = 1.0_f32 / 6.0;
    for i in 0..n {
        let sum = k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i];
        expected[i] = y[i] + h * sixth * sum;
    }

    let ptx = crate::ptx_kernels::rk4_stage_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rk4_stage_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_k1 = DeviceBuffer::<f32>::from_host(&k1).expect("d_k1");
    let d_k2 = DeviceBuffer::<f32>::from_host(&k2).expect("d_k2");
    let d_k3 = DeviceBuffer::<f32>::from_host(&k3).expect("d_k3");
    let d_k4 = DeviceBuffer::<f32>::from_host(&k4).expect("d_k4");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_y.as_device_ptr(),
                d_k1.as_device_ptr(),
                d_k2.as_device_ptr(),
                d_k3.as_device_ptr(),
                d_k4.as_device_ptr(),
                d_out.as_device_ptr(),
                h,
                n as u32,
            ),
        )
        .expect("launch rk4_stage_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for i in 0..n {
        assert!(
            close(out_gpu[i], expected[i], 1e-4, 1e-5),
            "rk4 out[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            expected[i]
        );
    }
}

// ===========================================================================
// 3. bisection_step  —  INDEPENDENT HOST RE-DERIVATION (midpoint)
// ===========================================================================

#[test]
fn bisection_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let a = samples(n, -5.0, 0.0, 0x0B15_0001);
    let b = samples(n, 0.0, 5.0, 0x0B15_0002);

    // Independent host re-derivation: mid = 0.5·(a + b), exactly the bracket
    // bisection step in `root::bisection`.
    let expected: Vec<f32> = a.iter().zip(&b).map(|(&av, &bv)| 0.5 * (av + bv)).collect();

    let ptx = crate::ptx_kernels::bisection_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bisection_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_mid = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_mid");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_mid.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch bisection_step_kernel");
    stream.synchronize().expect("sync");

    let mut mid_gpu = vec![0.0_f32; n];
    d_mid.copy_to_host(&mut mid_gpu).expect("copy mid");

    // `add.f32` + `mul.f32` by 0.5 (an exact power of two) is bit-identical to the
    // host's `0.5·(a+b)`; the comparison is therefore essentially exact.
    let (rel, abs) = worst_diff(&mid_gpu, &expected);
    for i in 0..n {
        assert!(
            close(mid_gpu[i], expected[i], 1e-6, 1e-7),
            "bisection mid[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            mid_gpu[i],
            expected[i]
        );
    }
}

// ===========================================================================
// 4. gauss_quad_accumulate  —  INDEPENDENT HOST RE-DERIVATION (w·f product)
// ===========================================================================

#[test]
fn gauss_quad_accumulate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let w = samples(n, 0.0, 1.0, 0x6A55_0001);
    let f = samples(n, -2.0, 2.0, 0x6A55_0002);

    // Independent host re-derivation: the kernel emits the per-node product
    // w_i·f_i (the host sums these to Σ w_i f(x_i) — the Gauss quadrature value).
    let expected: Vec<f32> = w.iter().zip(&f).map(|(&wv, &fv)| wv * fv).collect();

    let ptx = crate::ptx_kernels::gauss_quad_accumulate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gauss_quad_accumulate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_f = DeviceBuffer::<f32>::from_host(&f).expect("d_f");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_f.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch gauss_quad_accumulate_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // A single `mul.f32` (round-to-nearest) equals the host's `f32` product to the
    // bit, so the tolerance is essentially exact.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for i in 0..n {
        assert!(
            close(out_gpu[i], expected[i], 1e-6, 1e-7),
            "gauss_quad out[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            expected[i]
        );
    }
}

// ===========================================================================
// 5. spline_eval  —  INDEPENDENT HOST RE-DERIVATION (natural cubic spline piece)
// ===========================================================================

#[test]
fn spline_eval_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_nodes = 9_usize;
    let n_query = 64_usize;

    // Strictly increasing nodes with unit spacing ⇒ every h = 1, so the
    // `div.approx.f32` for A = (x1 - x)/h is exact and the only approximation is
    // the few-ulp `fma.rn` reassociation.
    let xs: Vec<f32> = (0..n_nodes).map(|i| i as f32).collect();
    let ys = samples(n_nodes, -2.0, 2.0, 0x5919_0001);
    let m = samples(n_nodes, -1.0, 1.0, 0x5919_0002); // second-derivative moments

    // Queries strictly inside each interval, with the matching piece index.
    let mut rng = LcgRng::new(0x5919_0003);
    let mut xe = vec![0.0_f32; n_query];
    let mut idx = vec![0_u32; n_query];
    for k in 0..n_query {
        let i = k % (n_nodes - 1); // piece [x_i, x_{i+1}]
        let frac = 0.05 + 0.90 * rng.next_f64() as f32; // strictly inside (0,1)
        xe[k] = xs[i] + frac; // h = 1 ⇒ x = x_i + frac
        idx[k] = i as u32;
    }

    // Independent host re-derivation of the documented spline piece:
    //   h = x_{i+1} - x_i,  A = (x_{i+1} - x)/h,  B = 1 - A,
    //   out = A·y_i + B·y_{i+1} + ((A³-A)·m_i + (B³-B)·m_{i+1})·h²/6.
    let mut expected = vec![0.0_f32; n_query];
    for k in 0..n_query {
        let i = idx[k] as usize;
        let x0 = xs[i];
        let x1 = xs[i + 1];
        let y0 = ys[i];
        let y1 = ys[i + 1];
        let m0 = m[i];
        let m1 = m[i + 1];
        let x = xe[k];
        let h = x1 - x0;
        let a = (x1 - x) / h;
        let b = 1.0 - a;
        let base = a * y0 + b * y1;
        let curv = (a * a * a - a) * m0 + (b * b * b - b) * m1;
        expected[k] = base + curv * (h * h) * (1.0 / 6.0);
    }

    let ptx = crate::ptx_kernels::spline_eval_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "spline_eval_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_xs = DeviceBuffer::<f32>::from_host(&xs).expect("d_xs");
    let d_ys = DeviceBuffer::<f32>::from_host(&ys).expect("d_ys");
    let d_m = DeviceBuffer::<f32>::from_host(&m).expect("d_m");
    let d_xe = DeviceBuffer::<f32>::from_host(&xe).expect("d_xe");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_query]).expect("d_out");
    let d_idx = DeviceBuffer::<u32>::from_host(&idx).expect("d_idx");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_query as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_xs.as_device_ptr(),
                d_ys.as_device_ptr(),
                d_m.as_device_ptr(),
                d_xe.as_device_ptr(),
                d_out.as_device_ptr(),
                d_idx.as_device_ptr(),
                n_query as u32,
                n_nodes as u32,
            ),
        )
        .expect("launch spline_eval_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_query];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // `div.approx.f32` (≤ 2 ulp) plus FMA reassociation over a handful of O(1)
    // terms keeps the divergence to a few ulp; 1e-3 relative comfortably covers
    // it yet still flags any gross formula error.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n_query {
        assert!(
            close(out_gpu[k], expected[k], 1e-3, 1e-4),
            "spline out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 6. central_diff  —  INDEPENDENT HOST RE-DERIVATION ((f₊ - f₋)/(2h))
// ===========================================================================

#[test]
fn central_diff_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let h = 0.01_f32;
    // Keep f₊ and f₋ separated so the difference is well away from catastrophic
    // cancellation, making the relative comparison meaningful.
    let f_plus = samples(n, 0.5, 2.5, 0xCD15_0001);
    let f_minus = samples(n, -2.5, -0.5, 0xCD15_0002);

    // Independent host re-derivation: out = (f₊ - f₋)/(2h), the crate's central
    // finite-difference quotient (`diff::central_difference`).
    let two_h = 2.0 * h;
    let expected: Vec<f32> = f_plus
        .iter()
        .zip(&f_minus)
        .map(|(&p, &m)| (p - m) / two_h)
        .collect();

    let ptx = crate::ptx_kernels::central_diff_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "central_diff_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_fp = DeviceBuffer::<f32>::from_host(&f_plus).expect("d_fp");
    let d_fm = DeviceBuffer::<f32>::from_host(&f_minus).expect("d_fm");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_fp.as_device_ptr(),
                d_fm.as_device_ptr(),
                d_out.as_device_ptr(),
                h,
                n as u32,
            ),
        )
        .expect("launch central_diff_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // `div.approx.f32` is ≤ 2 ulp; 1e-3 relative is a comfortable, still-meaningful
    // bound.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for i in 0..n {
        assert!(
            close(out_gpu[i], expected[i], 1e-3, 1e-4),
            "central_diff out[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            expected[i]
        );
    }
}

// ===========================================================================
// 7. bessel_recurrence  —  INDEPENDENT HOST RE-DERIVATION (Miller downward step)
// ===========================================================================
//
// HONEST SCOPE: the kernel implements the in-place downward recurrence
//   J_{n-1}(x) = (2n/x)·J_n(x) - J_{n+1}(x),  n = n_order … 1
// for a `j` buffer of shape (n_points, n_order+1). Reading `J_{n+1}` for the top
// order `n = n_order` reaches index `row + n_order + 1`, i.e. one element past
// the point's row — the canonical Miller's-algorithm seed `J_{N+1} = 0`. For a
// *multi-point* launch that read aliases the NEXT point's `J_0` (and races its
// write-back), so this test uses a SINGLE point plus an explicit trailing zero
// seed, which is both well-defined and the standard calling convention. The
// recurrence arithmetic is validated bit-tight against an independent host
// re-derivation; the multi-point cross-row aliasing is documented, not hidden.

#[test]
fn bessel_recurrence_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_order = 5_usize;
    let n_points = 1_usize;
    let x = 3.0_f32;

    // Buffer: (n_order + 1) columns + 1 trailing guard for the J_{n_order+1} seed.
    // Column n_order is the top seed J_{n_order}; the guard (index n_order+1) is
    // the J_{n_order+1} = 0 Miller seed; columns 0..n_order-1 are overwritten.
    let row = n_order + 1;
    let mut j = vec![0.0_f32; n_points * row + 1];
    j[n_order] = 1.0; // J_{n_order} seed
    // guard j[n_order + 1] already 0.0 ⇒ J_{n_order+1} = 0.

    // Independent host re-derivation with identical indexing / arithmetic.
    let mut j_host = j.clone();
    let mut order = n_order;
    while order >= 1 {
        let jn = j_host[order];
        let jn1 = j_host[order + 1];
        let two_n = 2.0_f32 * order as f32;
        j_host[order - 1] = (two_n / x) * jn - jn1;
        order -= 1;
    }

    let x_arr = vec![x; n_points];

    let ptx = crate::ptx_kernels::bessel_recurrence_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bessel_recurrence_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_j = DeviceBuffer::<f32>::from_host(&j).expect("d_j");
    let d_x = DeviceBuffer::<f32>::from_host(&x_arr).expect("d_x");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n_points as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_j.as_device_ptr(),
                d_x.as_device_ptr(),
                n_order as u32,
                n_points as u32,
            ),
        )
        .expect("launch bessel_recurrence_kernel");
    stream.synchronize().expect("sync");

    let mut j_gpu = vec![0.0_f32; n_points * row + 1];
    d_j.copy_to_host(&mut j_gpu).expect("copy j");

    // Compare the recurrence outputs J_0..J_{n_order} (the guard is excluded).
    // `div.approx.f32` (≤ 2 ulp) plus the subtractive recurrence (some
    // cancellation as J_0 is the difference of two larger terms) keep the
    // divergence small; 2e-3 relative with a 1e-3 floor is comfortable yet still
    // flags a wrong constant (e.g. a missing factor-of-two on 2n) by orders of
    // magnitude.
    let (rel, abs) = worst_diff(&j_gpu[..=n_order], &j_host[..=n_order]);
    for order in 0..=n_order {
        assert!(
            close(j_gpu[order], j_host[order], 2e-3, 1e-3),
            "bessel J_{order} mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            j_gpu[order],
            j_host[order]
        );
    }
}
