//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to the crate's CPU
//! reference. The launch ABI mirrors the working `oxicuda-snn` / `oxicuda-ot`
//! canaries: device buffers are passed as their `CUdeviceptr` (a `.param .u64`),
//! scalars are passed as the matching Rust scalar (`.param .u32` / `.param .f32`)
//! in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! All seven kernels are validated by a real CPU-vs-GPU numerical (or bit-exact)
//! equivalence check — none are stubs.
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel mirrors:
//!   `dp_clip_gradient_kernel` ↔ [`crate::privacy::gaussian::GaussianMechanism::clip_gradient`],
//!   `qsgd_quantize_kernel` ↔ the documented per-element QSGD rule shared with
//!   [`crate::compression::quantize::stochastic_quantize`] (the kernel takes the
//!   uniform noise as a buffer rather than drawing it from an RNG, so the test
//!   supplies the noise and re-derives the identical floor/clamp pipeline),
//!   `pairwise_mask_kernel` ↔ [`crate::secure_agg::masking::apply_mask`].
//! * **Independent host re-derivation** — the op has no standalone `pub fn`
//!   (it is fused into a larger routine on the CPU), so the oracle is an
//!   independent Rust re-implementation of the kernel's documented arithmetic:
//!   `fedavg_weighted_sum_kernel` (`out += weight·param`),
//!   `gaussian_noise_kernel` (Box–Muller `z = sqrt(-2·ln u1)·cos(2π·u2)`,
//!   computed with BASE-E `ln`/`exp` so a base-2 error would be caught),
//!   `topk_mask_kernel` (threshold mask, bit-exact),
//!   `aggregate_mean_kernel` (online mean `m·(n-1)/n + x/n`). These still
//!   genuinely fail if ptxas miscompiles or the PTX has a wrong constant /
//!   shift / index, because the host code is independent of the JIT-compiled PTX.
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
/// A failure here means ptxas rejected the hand-written PTX — a real bug in
/// `ptx_kernels.rs`, not a reason to skip.
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

// ===========================================================================
// 1. fedavg_weighted_sum  —  HOST RE-DERIVATION (out += weight·param)
// ===========================================================================

#[test]
fn fedavg_weighted_sum_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let weight = 0.37_f32;

    let mut rng = LcgRng::new(0xFED_A7E5);
    let param: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let out0: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference: out[i] += weight * param[i].
    let expected: Vec<f32> = out0
        .iter()
        .zip(&param)
        .map(|(&o, &p)| o + weight * p)
        .collect();

    let ptx = crate::ptx_kernels::fedavg_weighted_sum_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fedavg_weighted_sum_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&out0).expect("d_out");
    let d_param = DeviceBuffer::<f32>::from_host(&param).expect("d_param");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_param.as_device_ptr(),
                weight,
                n as u32,
            ),
        )
        .expect("launch fedavg_weighted_sum_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // GPU fuses `fma.rn(weight, param, out)` (one rounding) vs the host's
    // mul+add (two roundings): ~1 ulp divergence, well within 1e-5 relative.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for (k, (&g, &c)) in out_gpu.iter().zip(&expected).enumerate() {
        assert!(
            close(g, c, 1e-5, 1e-6),
            "fedavg out[{k}] mismatch: gpu={g} host={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 2. dp_clip_gradient  —  CRATE ORACLE (GaussianMechanism::clip_gradient)
// ===========================================================================

#[test]
fn dp_clip_gradient_matches_cpu() {
    use crate::privacy::gaussian::GaussianMechanism;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // The kernel reduces the sum-of-squares with a single grid-stride loop and a
    // per-thread accumulator (no cross-thread reduction), so a CORRECT full-vector
    // clip requires a SINGLE thread. We launch grid=1, block=1 accordingly; the
    // stride then equals 1 and the one thread visits every element.
    let n = 64_usize;
    let clip_norm = 1.0_f32;

    let mut rng = LcgRng::new(0xC11_9009);
    // Scale so the L2 norm is comfortably > clip_norm and the clip path is
    // exercised (scale = clip_norm/norm < 1, a non-trivial transform).
    let grad0: Vec<f32> = (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.8).collect();

    // ---- CPU reference ----
    let mut grad_cpu = grad0.clone();
    GaussianMechanism::clip_gradient(&mut grad_cpu, clip_norm).expect("cpu clip_gradient");
    // Precondition: clipping actually happened (else the test would be vacuous).
    let norm0: f32 = grad0.iter().map(|&g| g * g).sum::<f32>().sqrt();
    assert!(
        norm0 > clip_norm,
        "test setup: gradient norm {norm0} must exceed clip_norm {clip_norm}"
    );

    // ---- GPU ----
    let ptx = crate::ptx_kernels::dp_clip_gradient_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dp_clip_gradient_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grad = DeviceBuffer::<f32>::from_host(&grad0).expect("d_grad");

    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_grad.as_device_ptr(), clip_norm, n as u32),
        )
        .expect("launch dp_clip_gradient_kernel");
    stream.synchronize().expect("sync");

    let mut grad_gpu = vec![0.0_f32; n];
    d_grad.copy_to_host(&mut grad_gpu).expect("copy grad");

    // The kernel uses `sqrt.approx.f32` and `div.approx.f32` (~1 ulp each), the
    // CPU uses correctly-rounded libm; the shared `scale` is ~1e-6 relative off,
    // which propagates linearly. 1e-4 is comfortable yet still flags any gross
    // error (e.g. a missing sqrt, ~scale²).
    let (rel, abs) = worst_diff(&grad_gpu, &grad_cpu);
    for (k, (&g, &c)) in grad_gpu.iter().zip(&grad_cpu).enumerate() {
        assert!(
            close(g, c, 1e-4, 1e-6),
            "dp_clip grad[{k}] mismatch: gpu={g} cpu={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}

// ===========================================================================
// 3. gaussian_noise  —  HOST RE-DERIVATION (Box–Muller, BASE-E oracle)
// ===========================================================================

#[test]
fn gaussian_noise_matches_host_box_muller() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let sigma = 0.5_f32;

    let mut rng = LcgRng::new(0x6A0_5512);
    // u1 ∈ [0.1, 0.95): the kernel floors u1 at 1e-6 to avoid ln(0); staying well
    // inside (0,1) keeps `sqrt(-2 ln u1)` in a moderate, well-conditioned range.
    // u2 ∈ [0, 1): full phase range, exercising `cos.approx` across [0, 2π).
    let u1: Vec<f32> = (0..n).map(|_| 0.1 + 0.85 * rng.next_f32()).collect();
    let u2: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let grad0: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference using BASE-E natural log (so a base-2 omission in the PTX
    // would diverge by ~30%, far outside tolerance). Mirrors the kernel:
    //   u1c = max(u1, 1e-6); z = sqrt(-2 ln u1c) * cos(2π u2); grad += sigma·z.
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let expected: Vec<f32> = grad0
        .iter()
        .zip(u1.iter().zip(u2.iter()))
        .map(|(&g, (&a, &b))| {
            let a_c = a.max(1e-6_f32);
            let z = (-2.0_f32 * a_c.ln()).sqrt() * (two_pi * b).cos();
            g + sigma * z
        })
        .collect();

    let ptx = crate::ptx_kernels::gaussian_noise_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gaussian_noise_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grad = DeviceBuffer::<f32>::from_host(&grad0).expect("d_grad");
    let d_u1 = DeviceBuffer::<f32>::from_host(&u1).expect("d_u1");
    let d_u2 = DeviceBuffer::<f32>::from_host(&u2).expect("d_u2");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_grad.as_device_ptr(),
                d_u1.as_device_ptr(),
                d_u2.as_device_ptr(),
                sigma,
                n as u32,
            ),
        )
        .expect("launch gaussian_noise_kernel");
    stream.synchronize().expect("sync");

    let mut grad_gpu = vec![0.0_f32; n];
    d_grad.copy_to_host(&mut grad_gpu).expect("copy grad");

    // The kernel chains `lg2.approx` (with the correct `·(-2 ln2)` base
    // conversion), `sqrt.approx`, and `cos.approx`. `cos.approx.f32` carries the
    // largest error (~2^-20 in the reduced argument), and it is multiplied by a
    // magnitude up to ~2.1, so the absolute floor must accommodate a few-e-3
    // worst case; a base-2 error (no conversion) would be ~30% — far outside.
    let (rel, abs) = worst_diff(&grad_gpu, &expected);
    for (k, (&g, &c)) in grad_gpu.iter().zip(&expected).enumerate() {
        assert!(
            close(g, c, 3e-3, 3e-3),
            "gaussian_noise grad[{k}] mismatch: gpu={g} host={c} \
             (u1={} u2={} worst rel={rel:e} abs={abs:e})",
            u1[k],
            u2[k]
        );
    }
}

// ===========================================================================
// 4. topk_mask  —  HOST RE-DERIVATION (threshold mask, BIT-EXACT)
// ===========================================================================

#[test]
fn topk_mask_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let thresh = 0.4_f32;

    let mut rng = LcgRng::new(0x70F_4A5C);
    // Values in [-1, 1); none are exactly ±thresh, so the `>=` decision is
    // unambiguous and the masked output is bit-identical to the host.
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference: out[i] = |x[i]| >= thresh ? x[i] : 0.0.
    let expected: Vec<f32> = x
        .iter()
        .map(|&v| if v.abs() >= thresh { v } else { 0.0 })
        .collect();

    let ptx = crate::ptx_kernels::topk_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "topk_mask_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), d_x.as_device_ptr(), thresh, n as u32),
        )
        .expect("launch topk_mask_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // No arithmetic approximation — a passthrough or zero — so compare bit-exact.
    for (k, (&g, &c)) in out_gpu.iter().zip(&expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            c.to_bits(),
            "topk_mask out[{k}] mismatch: gpu={g} host={c} (x={})",
            x[k]
        );
    }
}

// ===========================================================================
// 5. qsgd_quantize  —  CRATE ORACLE (QSGD floor/clamp rule)
// ===========================================================================

#[test]
fn qsgd_quantize_matches_cpu_rule() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 200_usize;
    let norm = 5.0_f32;
    let s_levels = 8.0_f32;

    // Deterministic inputs chosen so that `|x|/norm·s + u` lands comfortably away
    // from any integer boundary (fractional part ≈ 0.55–0.75). The GPU's
    // `div.approx.f32` is ~1e-7 relative, so floor() is stable and the integer
    // result is identical to the host's exact-division derivation.
    let mut x = vec![0.0_f32; n];
    let mut u = vec![0.0_f32; n];
    for (k, (xs, us)) in x.iter_mut().zip(u.iter_mut()).enumerate() {
        let level = (k % (s_levels as usize)) as f32; // target floor level 0..7
        let sign = if k % 2 == 0 { 1.0_f32 } else { -1.0_f32 };
        // |x|/norm·s = level + 0.45  ⇒  |x| = norm·(level + 0.45)/s.
        *xs = sign * norm * (level + 0.45) / s_levels;
        *us = 0.20; // sum fractional part ≈ 0.65, floor = level.
    }

    // Host reference (exact f32 division), matching the kernel's documented
    // `q = sign(x) · clamp(floor(|x|/max(norm,1e-6)·s + u), 0, s)`.
    let norm_safe = norm.max(1e-6_f32);
    let expected: Vec<f32> = x
        .iter()
        .zip(&u)
        .map(|(&xv, &uv)| {
            let sign = if xv >= 0.0 { 1.0_f32 } else { -1.0_f32 };
            let level = (xv.abs() / norm_safe * s_levels + uv).floor();
            let clamped = level.clamp(0.0, s_levels);
            sign * clamped
        })
        .collect();

    let ptx = crate::ptx_kernels::qsgd_quantize_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "qsgd_quantize_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_q");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_u = DeviceBuffer::<f32>::from_host(&u).expect("d_u");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_x.as_device_ptr(),
                d_u.as_device_ptr(),
                norm,
                s_levels,
                n as u32,
            ),
        )
        .expect("launch qsgd_quantize_kernel");
    stream.synchronize().expect("sync");

    let mut q_gpu = vec![0.0_f32; n];
    d_q.copy_to_host(&mut q_gpu).expect("copy q");

    // Outputs are signed integers (levels times ±1). Any real discrepancy is an
    // off-by-one (≥ 1.0) or sign flip; the < 0.5 bound catches both while
    // tolerating the inputs being kept off the floor boundary.
    for (k, (&g, &c)) in q_gpu.iter().zip(&expected).enumerate() {
        assert!(
            (g - c).abs() < 0.5,
            "qsgd q[{k}] mismatch: gpu={g} cpu={c} (x={} u={})",
            x[k],
            u[k]
        );
    }
}

// ===========================================================================
// 6. pairwise_mask  —  CRATE ORACLE (masking::apply_mask, BIT-EXACT u32)
// ===========================================================================

#[test]
fn pairwise_mask_matches_cpu() {
    use crate::secure_agg::masking::{apply_mask, generate_mask};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;

    let mut rng = LcgRng::new(0x9A5_4ED0);
    // The kernel adds raw u32 words mod 2^32. The CPU oracle `apply_mask`
    // interprets the gradient as f32 bit patterns; to make the two paths
    // bit-identical we drive the GPU with the same f32 bit patterns reinterpreted
    // as u32, exactly as `apply_mask` does internally.
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let mask: Vec<u32> = generate_mask(1, 3, 0xDEAD_BEEF_u64, n);

    // ---- CPU reference: (grad_bits + mask) mod 2^32 ----
    let expected: Vec<u32> = apply_mask(&grad, &mask).expect("cpu apply_mask");

    // GPU input as raw u32 words (the f32 bit patterns).
    let x_bits: Vec<u32> = grad.iter().map(|&g| g.to_bits()).collect();

    let ptx = crate::ptx_kernels::pairwise_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pairwise_mask_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_out");
    let d_x = DeviceBuffer::<u32>::from_host(&x_bits).expect("d_x");
    let d_mask = DeviceBuffer::<u32>::from_host(&mask).expect("d_mask");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_x.as_device_ptr(),
                d_mask.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch pairwise_mask_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Integer arithmetic mod 2^32 is exact and reproducible — compare bit-exact.
    for (k, (&g, &c)) in out_gpu.iter().zip(&expected).enumerate() {
        assert_eq!(
            g, c,
            "pairwise_mask out[{k}] mismatch: gpu={g} cpu={c} \
             (x_bits={:#010x} mask={:#010x})",
            x_bits[k], mask[k]
        );
    }
}

// ===========================================================================
// 7. aggregate_mean  —  HOST RE-DERIVATION (online mean m·(n-1)/n + x/n)
// ===========================================================================

#[test]
fn aggregate_mean_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let round_n = 5_u32; // current sample count for the online update

    let mut rng = LcgRng::new(0xA66_3E47);
    let mean0: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference: mean[i] = mean[i]·(n-1)/n + x[i]/n.
    let nf = round_n as f32;
    let factor = (nf - 1.0) / nf;
    let inv = 1.0 / nf;
    let expected: Vec<f32> = mean0
        .iter()
        .zip(&x)
        .map(|(&m, &xv)| m * factor + xv * inv)
        .collect();

    let ptx = crate::ptx_kernels::aggregate_mean_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "aggregate_mean_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mean = DeviceBuffer::<f32>::from_host(&mean0).expect("d_mean");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mean.as_device_ptr(),
                d_x.as_device_ptr(),
                round_n,
                n as u32,
            ),
        )
        .expect("launch aggregate_mean_kernel");
    stream.synchronize().expect("sync");

    let mut mean_gpu = vec![0.0_f32; n];
    d_mean.copy_to_host(&mut mean_gpu).expect("copy mean");

    // The kernel forms (n-1)/n with `div.approx.f32` and 1/n with
    // `rcp.approx.f32` (~1 ulp each), then a fused multiply-add; the host uses
    // correctly-rounded division. ~1e-6 relative divergence; 1e-4 is comfortable.
    let (rel, abs) = worst_diff(&mean_gpu, &expected);
    for (k, (&g, &c)) in mean_gpu.iter().zip(&expected).enumerate() {
        assert!(
            close(g, c, 1e-4, 1e-6),
            "aggregate_mean mean[{k}] mismatch: gpu={g} host={c} (worst rel={rel:e} abs={abs:e})"
        );
    }
}
