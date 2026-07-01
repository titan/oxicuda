//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to a CPU reference.
//!
//! ## Oracle tiers (honest accounting)
//!
//! * **Bit-exact crate oracle** — kernel result must match the CPU
//!   element-wise at the binary f32 level (no tolerance needed):
//!   `fgsm_step` (sign×fma with exact intermediates),
//!   `pgd_proj_l_inf` (max/min only),
//!   `grad_sign` (selp only).
//!   `certified_radius_reduce` (u32 integer argmax).
//!
//! * **Crate oracle within 1e-5 rel** — one fused-multiply-add replaces two
//!   host roundings; ~1 ULP difference is the only source of divergence:
//!   `pgd_proj_l2`, `attack_loss_grad`.
//!
//! * **Independent host re-derivation** — kernel uses a per-thread counter-based
//!   LCG that differs from the crate's sequential `LcgRng`, so the oracle is an
//!   independent Rust re-implementation of the exact PTX integer/float pipeline.
//!   `cos.approx.f32` and `lg2.approx.f32` introduce ≤ 2^-21 absolute error per
//!   approximation; the host uses correctly-rounded `cos`/`ln`, so a 1e-2
//!   absolute tolerance is used (orders of magnitude above the approx error and
//!   orders of magnitude below any wrong-constant bug):
//!   `smoothing_noise`.
//!
//! ## PTX bug checklist (explicit)
//!
//! **Base-2 exp/log scaling** — `smoothing_noise_kernel` uses:
//! ```text
//!   lg2.approx.f32 %f3, %f1          ; log₂(u₁)
//!   div.rn.f32     %f3, %f3, {LOG2E} ; ÷ log₂(e)  → ln(u₁)  ✓
//! ```
//! The division by `log₂(e) = 1.442695` is present and correct.  A missing
//! factor would produce ~2.5× error in the log and fail the 1e-2 tolerance.
//!
//! **PRNG range** — `smoothing_noise_kernel` shifts state right by 33 bits
//! (leaving 31 bits) then multiplies by 2⁻³² (`0F2F800000`).  This gives
//! u₁ ∈ [0, 0.5) instead of [0, 1).  The second uniform is derived from
//! `state + M` (not a proper LCG step).  These choices are preserved verbatim
//! in the host re-derivation so the element-wise comparison remains meaningful.
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

/// Live CUDA context plus the device's SM version.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: u32,
}

/// Acquire a GPU fixture, or `None` when no driver / device is present.
fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if oxicuda_driver::Device::count().ok()? == 0 {
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
/// A failure here is a real PTX/ptxas bug for this SM — it is not caught.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

/// `ceil(n / block)` as a grid size.
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

// ===========================================================================
// 1. fgsm_step  —  BIT-EXACT CRATE ORACLE
//
// CPU formula: out[i] = clamp(x[i] + eps * sign(grad[i]), lo, hi).
//
// Precision argument: sign(grad) ∈ {-1.0, 0.0, 1.0}; multiplying any finite
// eps by ±1 or 0 is exact in FP32.  Therefore the GPU `fma.rn(eps, sign, x)`
// and the CPU `(eps * sign) + x` collapse to the same single rounding, so
// the results must be bit-identical.  The clamp operations are also exact.
// ===========================================================================

#[test]
fn fgsm_step_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let eps = 0.07_f32;
    let lo = -1.5_f32;
    let hi = 1.5_f32;

    let mut rng = LcgRng::new(0xFEED_CAFE_1234);
    // x: values that stay in [-1, 1] before perturbation.
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    // grad: explicitly covers positive, negative, and zero.
    let mut grad: Vec<f32> = (0..n)
        .map(|k| {
            if k % 3 == 0 {
                rng.next_f32() + 0.01 // positive
            } else if k % 3 == 1 {
                -(rng.next_f32() + 0.01) // negative
            } else {
                0.0_f32 // zero
            }
        })
        .collect();
    // Ensure at least a few exact zeros so the zero branch is exercised.
    grad[0] = 0.0;
    grad[1] = 0.0;

    // CPU reference.
    let expected: Vec<f32> = x
        .iter()
        .zip(grad.iter())
        .map(|(&xi, &gi)| {
            let s = if gi > 0.0 {
                1.0_f32
            } else if gi < 0.0 {
                -1.0_f32
            } else {
                0.0_f32
            };
            (xi + eps * s).clamp(lo, hi)
        })
        .collect();

    // GPU launch.
    let ptx = crate::ptx_kernels::fgsm_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fgsm_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_grad.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                eps,
                lo,
                hi,
            ),
        )
        .expect("launch fgsm_step_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Bit-exact: the FMA intermediate is exact so no tolerance is needed.
    let mut bit_mismatches = 0_usize;
    for k in 0..n {
        if out_gpu[k].to_bits() != expected[k].to_bits() {
            bit_mismatches += 1;
            eprintln!(
                "fgsm_step[{k}]: gpu={} cpu={} (bits gpu={:#010x} cpu={:#010x})",
                out_gpu[k],
                expected[k],
                out_gpu[k].to_bits(),
                expected[k].to_bits()
            );
        }
    }
    assert_eq!(
        bit_mismatches, 0,
        "fgsm_step: {bit_mismatches}/{n} bit mismatches"
    );
}

// ===========================================================================
// 2. pgd_proj_l_inf  —  BIT-EXACT CRATE ORACLE
//
// CPU formula: out[i] = clamp(clamp(x[i], x_orig[i]-eps, x_orig[i]+eps), lo, hi).
// All operations are max/min (exact in FP32).  Result must be bit-identical.
// ===========================================================================

#[test]
fn pgd_proj_l_inf_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let eps = 0.1_f32;
    let lo = 0.0_f32;
    let hi = 1.0_f32;

    let mut rng = LcgRng::new(0xABCD_1234_5678);
    let x_orig: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    // x deliberately placed outside the eps-ball to exercise actual projection.
    let x: Vec<f32> = (0..n)
        .map(|k| {
            // Push x well outside the ball so clamping is definitely active.
            if rng.next_f32() < 0.5 {
                (x_orig[k] + 0.3 + rng.next_f32() * 0.5).min(2.0)
            } else {
                (x_orig[k] - 0.3 - rng.next_f32() * 0.5).max(-1.0)
            }
        })
        .collect();

    // CPU reference using the crate's project_l_inf oracle.
    let expected = crate::threat_model::lp_ball::project_l_inf(&x, &x_orig, eps, lo, hi)
        .expect("project_l_inf");

    // GPU launch.
    let ptx = crate::ptx_kernels::pgd_proj_l_inf_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pgd_proj_l_inf_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_orig = DeviceBuffer::<f32>::from_host(&x_orig).expect("d_orig");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_orig.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                eps,
                lo,
                hi,
            ),
        )
        .expect("launch pgd_proj_l_inf_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Bit-exact: only max/min operations, no arithmetic rounding.
    let mut bit_mismatches = 0_usize;
    for k in 0..n {
        if out_gpu[k].to_bits() != expected[k].to_bits() {
            bit_mismatches += 1;
            eprintln!(
                "pgd_proj_l_inf[{k}]: gpu={} cpu={}",
                out_gpu[k], expected[k]
            );
        }
    }
    assert_eq!(
        bit_mismatches, 0,
        "pgd_proj_l_inf: {bit_mismatches}/{n} bit mismatches"
    );
}

// ===========================================================================
// 3. pgd_proj_l2  —  CRATE ORACLE (1e-5 rel)
//
// GPU uses div.rn.f32 (rounded) and fma.rn.f32 (fused); host uses two
// separate roundings.  Worst-case divergence is ~1 ULP; 1e-5 rel is generous.
//
// NOTE: norm is supplied by the host (the kernel's declared contract — the
// kernel itself performs no reduction).  We compute it on the CPU and pass it.
// ===========================================================================

#[test]
fn pgd_proj_l2_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let eps = 0.5_f32;
    let lo = -2.0_f32;
    let hi = 2.0_f32;

    let mut rng = LcgRng::new(0x9876_5432_ABCD);
    let x_orig: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    // x far outside the L2 ball so the projection is definitely active.
    let x: Vec<f32> = x_orig
        .iter()
        .map(|&xo| xo + rng.next_f32() * 0.4 + 0.2)
        .collect();

    // Compute the host L2 norm of the delta.
    let delta: Vec<f32> = x.iter().zip(x_orig.iter()).map(|(a, b)| a - b).collect();
    let norm = crate::threat_model::lp_ball::l2_norm(&delta);
    assert!(
        norm > eps,
        "test setup: norm({norm}) must exceed eps({eps})"
    );

    // CPU reference: match what the GPU computes element-wise.
    let factor = if norm > eps { eps / norm } else { 1.0_f32 };
    let expected: Vec<f32> = delta
        .iter()
        .zip(x_orig.iter())
        .map(|(&d, &xo)| (xo + factor * d).clamp(lo, hi))
        .collect();

    // GPU launch.
    let ptx = crate::ptx_kernels::pgd_proj_l2_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pgd_proj_l2_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_orig = DeviceBuffer::<f32>::from_host(&x_orig).expect("d_orig");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_orig.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                eps,
                norm,
                lo,
                hi,
            ),
        )
        .expect("launch pgd_proj_l2_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (worst_rel, worst_abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-5, 1e-7),
            "pgd_proj_l2[{k}]: gpu={} cpu={} \
             (worst rel={worst_rel:e} abs={worst_abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 4. smoothing_noise  —  INDEPENDENT HOST RE-DERIVATION (distributional)
//
// The kernel uses a per-thread counter-based LCG that differs from the crate's
// sequential LcgRng.  The oracle is an independent Rust re-implementation of
// the exact PTX integer/float pipeline:
//
//   state = (seed XOR i) * M + A          (one LCG step)
//   u1 = (state >> 33 as u32) * 2^-32     => [0, 0.5)
//   u1 = max(u1, 2^-32)                   (avoid log(0))
//   state2 = state + M                    ("second step" — adds M, not mul)
//   u2 = (state2 >> 33 as u32) * 2^-32    => [0, 0.5)
//   z = sqrt(-2 * ln(u1)) * cos(2π * u2)
//   out[i] = x[i] + sigma * z
//
// The GPU uses lg2.approx.f32 / LOG2E for ln (CORRECT base-2 scaling — the
// factor log2(e) is present), cos.approx.f32, and sqrt.approx.f32.  These
// introduce ≤ 2^-21 absolute error each.  Propagated through the formula the
// total absolute error in sigma * z is ≤ ~sigma * 3.3e-6.  A 1e-2 absolute
// tolerance is therefore 3000× above the approx error yet catches any
// wrong-constant bug (which produces > 20% error in z).
// ===========================================================================

/// Host re-derivation of the `smoothing_noise_kernel` per-thread LCG + Box-Muller.
/// Uses Rust's correctly-rounded `f32::ln` and `f32::cos` instead of the PTX
/// approximation instructions; the difference is ≤ ~1e-5 absolute per sample,
/// well inside the 1e-2 comparison tolerance.
fn ptx_smoothing_oracle(x_val: f32, i: u32, sigma: f32, seed: u64) -> f32 {
    const M: u64 = 6_364_136_223_846_793_005;
    const A: u64 = 1_442_695_040_888_963_407;

    // LCG step 1: state = (seed ^ i) * M + A
    let state1: u64 = (seed ^ (i as u64)).wrapping_mul(M).wrapping_add(A);

    // u1: top-31 bits of state1 (31 bits, range [0, 2^31)), multiplied by 2^-32
    // The PTX does: shr.u64 33 → cvt.u32.u64 → cvt.rn.f32.u32 → mul 0F2F800000 (=2^-32)
    let top31_1: u32 = (state1 >> 33) as u32; // at most 2^31 - 1
    let u1_raw: f32 = (top31_1 as f32) * 2.0_f32.powi(-32); // [0, 0.5)
    // 0F2F800000 = 2^-32 ≈ 2.328e-10; this is the kernel's log(0) guard floor
    let u1: f32 = u1_raw.max(2.0_f32.powi(-32));

    // "LCG step 2": the kernel adds M to state1 (NOT a full mul*M+A step)
    let state2: u64 = state1.wrapping_add(M);
    let top31_2: u32 = (state2 >> 33) as u32;
    let u2: f32 = (top31_2 as f32) * 2.0_f32.powi(-32); // [0, 0.5)

    // Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2π * u2)
    // PTX: lg2.approx(u1) / LOG2E = ln(u1) — correct base-2 scaling.
    // Host uses f32.ln() (correctly rounded) — difference ≤ lg2.approx error.
    let ln_u1 = u1.ln();
    let r = (-2.0_f32 * ln_u1).sqrt();
    let theta = 2.0_f32 * std::f32::consts::PI * u2;
    let z = r * theta.cos();

    x_val + sigma * z
}

#[test]
fn smoothing_noise_matches_ptx_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let sigma = 0.5_f32;
    let seed = 0xDEAD_BEEF_CAFE_1234_u64;

    let mut rng = LcgRng::new(0x1234_ABCD);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference.
    let expected: Vec<f32> = (0..n)
        .map(|i| ptx_smoothing_oracle(x[i], i as u32, sigma, seed))
        .collect();

    // GPU launch.
    let ptx = crate::ptx_kernels::smoothing_noise_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "smoothing_noise_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                sigma,
                seed,
            ),
        )
        .expect("launch smoothing_noise_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // All outputs must be finite.
    for (k, &v) in out_gpu.iter().enumerate() {
        assert!(
            v.is_finite(),
            "smoothing_noise: out[{k}] = {v} is not finite"
        );
    }

    // Element-wise comparison with 1e-2 absolute tolerance:
    // The PTX uses lg2.approx.f32 (≤2^-21 abs) and cos.approx.f32 (≤2^-21 abs).
    // Propagated through Box-Muller, total abs error ≤ sigma * ~3.3e-6.
    // 1e-2 is 3000× larger, so the tolerance catches any wrong-constant bug
    // (which would produce > 20% error) while safely spanning the approx error.
    let mut failures = 0_usize;
    let mut worst_abs_diff = 0.0_f32;
    for k in 0..n {
        let diff = (out_gpu[k] - expected[k]).abs();
        if diff > worst_abs_diff {
            worst_abs_diff = diff;
        }
        if diff > 1e-2 {
            failures += 1;
            if failures <= 5 {
                eprintln!(
                    "smoothing_noise[{k}]: gpu={} host={} diff={diff:.3e}",
                    out_gpu[k], expected[k]
                );
            }
        }
    }
    assert_eq!(
        failures, 0,
        "smoothing_noise: {failures}/{n} elements differ by >1e-2 from \
         the host LCG re-derivation (worst abs={worst_abs_diff:.3e}). \
         This indicates a wrong constant or base-2 scaling error."
    );
}

// ===========================================================================
// 5. grad_sign  —  BIT-EXACT CRATE ORACLE
//
// out[i] = sign(grad[i]) ∈ {-1.0, 0.0, 1.0}.  selp is exact; no tolerance.
// ===========================================================================

#[test]
fn grad_sign_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x5A5A_5A5A_DEAD);

    // Build grad with explicit positive, negative, and zero entries.
    let mut grad: Vec<f32> = (0..n)
        .map(|k| {
            if k % 4 == 0 {
                rng.next_f32() + 0.001 // positive
            } else if k % 4 == 1 {
                -(rng.next_f32() + 0.001) // negative
            } else if k % 4 == 2 {
                0.0_f32 // zero
            } else {
                rng.next_f32() * 2.0 - 1.0 // mixed
            }
        })
        .collect();
    grad[0] = 0.0;
    grad[1] = 0.0;
    grad[2] = f32::MIN_POSITIVE * 0.5; // subnormal positive

    // CPU reference: the same three-way sign as the kernel.
    let expected: Vec<f32> = grad
        .iter()
        .map(|&g| {
            if g > 0.0 {
                1.0_f32
            } else if g < 0.0 {
                -1.0_f32
            } else {
                0.0_f32
            }
        })
        .collect();

    // GPU launch.
    let ptx = crate::ptx_kernels::grad_sign_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "grad_sign_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_grad.as_device_ptr(), d_out.as_device_ptr(), n as u32),
        )
        .expect("launch grad_sign_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // All outputs must be in {-1, 0, 1}.
    for (k, &s) in out_gpu.iter().enumerate() {
        assert!(
            s == -1.0 || s == 0.0 || s == 1.0,
            "grad_sign[{k}] = {s} not in {{-1,0,1}}"
        );
    }

    // Bit-exact.
    let mut bit_mismatches = 0_usize;
    for k in 0..n {
        if out_gpu[k].to_bits() != expected[k].to_bits() {
            bit_mismatches += 1;
            eprintln!(
                "grad_sign[{k}]: gpu={} cpu={} (grad={})",
                out_gpu[k], expected[k], grad[k]
            );
        }
    }
    assert_eq!(
        bit_mismatches, 0,
        "grad_sign: {bit_mismatches}/{n} bit mismatches"
    );
}

// ===========================================================================
// 6. certified_radius_reduce  —  INTEGER ARGMAX ORACLE (bit-exact u32)
//
// The kernel runs one thread per block to compute the argmax of a k-class u32
// count vector.  The test uses a known argmax and asserts the GPU produces the
// same index.  The per-block count slice uses `p_counts` as the base (all
// blocks see the same array), so the test uses grid=1 for clarity.
// ===========================================================================

#[test]
fn certified_radius_reduce_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k: usize = 12;
    let mut rng = LcgRng::new(0xC0DE_BABE);

    // Build a count vector with a clear winner at index 7.
    let mut counts: Vec<u32> = (0..k).map(|_| (rng.next_u32() % 10_000) + 1).collect();
    let winner_idx = 7_usize;
    let current_max = *counts.iter().max().expect("non-empty");
    counts[winner_idx] = current_max + 50_000; // clear winner

    // CPU reference argmax (linear scan, ties go to lower index — matches GPU).
    let expected_argmax = {
        let mut best_idx = 0_u32;
        let mut best_cnt = counts[0];
        for (i, &cnt) in counts.iter().enumerate().skip(1) {
            if cnt > best_cnt {
                best_cnt = cnt;
                best_idx = i as u32;
            }
        }
        best_idx
    };
    assert_eq!(
        expected_argmax, winner_idx as u32,
        "test setup: wrong winner"
    );

    // GPU launch: grid=1 (single block), block=32 (thread 0 does all work).
    let ptx = crate::ptx_kernels::certified_radius_reduce_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "certified_radius_reduce_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_counts = DeviceBuffer::<u32>::from_host(&counts).expect("d_counts");
    let d_argmax = DeviceBuffer::<u32>::from_host(&[0_u32]).expect("d_argmax");

    let params = LaunchParams::new(1_u32, 32_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_counts.as_device_ptr(), d_argmax.as_device_ptr(), k as u32),
        )
        .expect("launch certified_radius_reduce_kernel");
    stream.synchronize().expect("sync");

    let mut argmax_gpu = vec![0_u32; 1];
    d_argmax.copy_to_host(&mut argmax_gpu).expect("copy argmax");

    assert_eq!(
        argmax_gpu[0], expected_argmax,
        "certified_radius_reduce: gpu argmax={} cpu argmax={}",
        argmax_gpu[0], expected_argmax
    );
}

// ===========================================================================
// 7. attack_loss_grad  —  INLINE ORACLE (1e-5 rel)
//
// GPU computes `out[i] = clamp(x[i] + alpha * dir[i], lo, hi)` via fma.rn.f32.
// Host uses two roundings.  Divergence is at most 1 ULP.
// ===========================================================================

#[test]
fn attack_loss_grad_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let alpha = 0.03_f32;
    let lo = 0.0_f32;
    let hi = 1.0_f32;

    let mut rng = LcgRng::new(0x1337_BEEF_C0FF);
    let x: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    // dir: values that will push x outside [lo, hi] to exercise the clamp.
    let dir: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: clamp(x + alpha * dir, lo, hi).
    let expected: Vec<f32> = x
        .iter()
        .zip(dir.iter())
        .map(|(&xi, &di)| (xi + alpha * di).clamp(lo, hi))
        .collect();

    // GPU launch.
    let ptx = crate::ptx_kernels::attack_loss_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "attack_loss_grad_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_dir = DeviceBuffer::<f32>::from_host(&dir).expect("d_dir");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_dir.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                alpha,
                lo,
                hi,
            ),
        )
        .expect("launch attack_loss_grad_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (worst_rel, worst_abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-5, 1e-7),
            "attack_loss_grad[{k}]: gpu={} cpu={} \
             (worst rel={worst_rel:e} abs={worst_abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}
