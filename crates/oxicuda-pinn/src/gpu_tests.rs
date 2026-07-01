//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` canaries: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .u32` /
//! `.param .f32` / `.param .u64`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU routine the kernel mirrors:
//!   `dual_mul_kernel` ↔ [`crate::autodiff::dual::Dual`] `Mul` (product rule),
//!   `pinn_residual_kernel` ↔ `n ·`[`crate::pinn_loss::residual::pde_residual_loss`]
//!   (the CPU helper returns the *mean* of the squares; the kernel accumulates
//!   the *sum*, so `sum == n · mse`).
//! * **Independent host re-derivation** — the op is fused into a larger CPU
//!   routine with no standalone `pub fn`, so the oracle is an independent Rust
//!   re-implementation of the kernel's *documented* arithmetic:
//!   `spectral_conv_kernel` (complex multiply), `adjoint_step_kernel`
//!   (`a += h·dadt`), `branch_trunk_dot_kernel` (inner product),
//!   `siren_forward_kernel` (`sin(ω₀·(Wx+b))`), and `lhs_sample_kernel`
//!   (the inline counter-based LCG + cell scaling). These still genuinely fail
//!   if ptxas miscompiles or the PTX has a wrong constant / shift / index,
//!   because the host code is independent of the JIT-compiled PTX.
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
/// A failure here means ptxas rejected the hand-written PTX (a real bug), so we
/// panic with the compiler diagnostic rather than skipping.
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
// 1. pinn_residual  —  CRATE ORACLE (n · pde_residual_loss = Σ r_i²)
// ===========================================================================

#[test]
fn pinn_residual_matches_cpu() {
    use crate::pinn_loss::residual::pde_residual_loss;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let mut rng = LcgRng::new(0x0B1E_5500);
    // Residuals in [-1, 1): Σ r_i² is O(n/3) ≈ 170, well away from zero so the
    // relative comparison is meaningful.
    let residuals: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: the crate computes the MEAN of the squares; the kernel
    // accumulates the SUM, so the oracle is n · mse.
    let mse = pde_residual_loss(&residuals).expect("cpu pde_residual_loss");
    let sum_cpu = mse * n as f32;

    let ptx = crate::ptx_kernels::pinn_residual_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pinn_residual_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_res = DeviceBuffer::<f32>::from_host(&residuals).expect("d_res");
    let d_sum = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_sum");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_res.as_device_ptr(), d_sum.as_device_ptr(), n as u32),
        )
        .expect("launch pinn_residual_kernel");
    stream.synchronize().expect("sync");

    let mut sum_gpu = [0.0_f32];
    d_sum.copy_to_host(&mut sum_gpu).expect("copy sum");

    // The kernel issues one `atom.global.add.f32` per element, so the summation
    // order is non-deterministic; over 512 O(1) terms the accumulated FP32
    // rounding is bounded by ~n ulp (~6e-5 relative). 1e-3 relative is generous
    // yet still flags any gross error (e.g. a missing square or wrong stride).
    assert!(
        close(sum_gpu[0], sum_cpu, 1e-3, 1e-3),
        "pinn_residual sum mismatch: gpu={} cpu={}",
        sum_gpu[0],
        sum_cpu
    );
}

// ===========================================================================
// 2. spectral_conv  —  INDEPENDENT HOST RE-DERIVATION (complex multiply)
// ===========================================================================

#[test]
fn spectral_conv_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x5BEC_C047);
    let a_real: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let a_imag: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let w_real: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let w_imag: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Independent host complex multiply:
    //   out_real = a_real*w_real - a_imag*w_imag
    //   out_imag = a_real*w_imag + a_imag*w_real
    let mut out_real_host = vec![0.0_f32; n];
    let mut out_imag_host = vec![0.0_f32; n];
    for i in 0..n {
        out_real_host[i] = a_real[i] * w_real[i] - a_imag[i] * w_imag[i];
        out_imag_host[i] = a_real[i] * w_imag[i] + a_imag[i] * w_real[i];
    }

    let ptx = crate::ptx_kernels::spectral_conv_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "spectral_conv_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ar = DeviceBuffer::<f32>::from_host(&a_real).expect("d_ar");
    let d_ai = DeviceBuffer::<f32>::from_host(&a_imag).expect("d_ai");
    let d_wr = DeviceBuffer::<f32>::from_host(&w_real).expect("d_wr");
    let d_wi = DeviceBuffer::<f32>::from_host(&w_imag).expect("d_wi");
    let d_or = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_or");
    let d_oi = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_oi");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ar.as_device_ptr(),
                d_ai.as_device_ptr(),
                d_wr.as_device_ptr(),
                d_wi.as_device_ptr(),
                d_or.as_device_ptr(),
                d_oi.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch spectral_conv_kernel");
    stream.synchronize().expect("sync");

    let mut out_real_gpu = vec![0.0_f32; n];
    let mut out_imag_gpu = vec![0.0_f32; n];
    d_or.copy_to_host(&mut out_real_gpu).expect("copy out_real");
    d_oi.copy_to_host(&mut out_imag_gpu).expect("copy out_imag");

    // The kernel computes out_real via `a*c − b*d` (sub of two products) and
    // out_imag via a fused `fma(a_imag, w_real, a_real*w_imag)`; the host uses
    // plain mul/add. Divergence is a few ulp (~1e-6 relative); 1e-4 is a
    // comfortable, still-meaningful bound.
    let (rel_r, abs_r) = worst_diff(&out_real_gpu, &out_real_host);
    for i in 0..n {
        assert!(
            close(out_real_gpu[i], out_real_host[i], 1e-4, 1e-6),
            "spectral_conv out_real[{i}] mismatch: gpu={} host={} (worst rel={rel_r:e} abs={abs_r:e})",
            out_real_gpu[i],
            out_real_host[i]
        );
    }
    let (rel_i, abs_i) = worst_diff(&out_imag_gpu, &out_imag_host);
    for i in 0..n {
        assert!(
            close(out_imag_gpu[i], out_imag_host[i], 1e-4, 1e-6),
            "spectral_conv out_imag[{i}] mismatch: gpu={} host={} (worst rel={rel_i:e} abs={abs_i:e})",
            out_imag_gpu[i],
            out_imag_host[i]
        );
    }
}

// ===========================================================================
// 3. dual_op  —  CRATE ORACLE (autodiff::dual::Dual Mul, product rule)
// ===========================================================================

#[test]
fn dual_mul_matches_crate() {
    use crate::autodiff::dual::Dual;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0xD0A1_900D);
    let a_val: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let a_dval: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let b_val: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let b_dval: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: the crate's own dual-number multiply (product rule).
    let mut out_val_cpu = vec![0.0_f32; n];
    let mut out_dval_cpu = vec![0.0_f32; n];
    for i in 0..n {
        let a = Dual {
            value: a_val[i],
            dvalue: a_dval[i],
        };
        let b = Dual {
            value: b_val[i],
            dvalue: b_dval[i],
        };
        let c = a * b;
        out_val_cpu[i] = c.value;
        out_dval_cpu[i] = c.dvalue;
    }

    let ptx = crate::ptx_kernels::dual_op_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dual_mul_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_av = DeviceBuffer::<f32>::from_host(&a_val).expect("d_av");
    let d_ad = DeviceBuffer::<f32>::from_host(&a_dval).expect("d_ad");
    let d_bv = DeviceBuffer::<f32>::from_host(&b_val).expect("d_bv");
    let d_bd = DeviceBuffer::<f32>::from_host(&b_dval).expect("d_bd");
    let d_ov = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_ov");
    let d_od = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_od");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_av.as_device_ptr(),
                d_ad.as_device_ptr(),
                d_bv.as_device_ptr(),
                d_bd.as_device_ptr(),
                d_ov.as_device_ptr(),
                d_od.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch dual_mul_kernel");
    stream.synchronize().expect("sync");

    let mut out_val_gpu = vec![0.0_f32; n];
    let mut out_dval_gpu = vec![0.0_f32; n];
    d_ov.copy_to_host(&mut out_val_gpu).expect("copy out_val");
    d_od.copy_to_host(&mut out_dval_gpu).expect("copy out_dval");

    // out_val is a single product (bit-exact vs CPU). out_dval is
    // `fma(a_dval, b_val, a_val*b_dval)` vs the CPU's two-rounding sum of
    // products — a few ulp. 1e-4 relative is comfortable and meaningful.
    let (rel_v, abs_v) = worst_diff(&out_val_gpu, &out_val_cpu);
    for i in 0..n {
        assert!(
            close(out_val_gpu[i], out_val_cpu[i], 1e-4, 1e-6),
            "dual_mul out_val[{i}] mismatch: gpu={} cpu={} (worst rel={rel_v:e} abs={abs_v:e})",
            out_val_gpu[i],
            out_val_cpu[i]
        );
    }
    let (rel_d, abs_d) = worst_diff(&out_dval_gpu, &out_dval_cpu);
    for i in 0..n {
        assert!(
            close(out_dval_gpu[i], out_dval_cpu[i], 1e-4, 1e-6),
            "dual_mul out_dval[{i}] mismatch: gpu={} cpu={} (worst rel={rel_d:e} abs={abs_d:e})",
            out_dval_gpu[i],
            out_dval_cpu[i]
        );
    }
}

// ===========================================================================
// 4. adjoint_ode  —  INDEPENDENT HOST RE-DERIVATION (Euler step a += h·dadt)
// ===========================================================================

#[test]
fn adjoint_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let h = 0.05_f32;
    let mut rng = LcgRng::new(0xAD01_07E5);
    let a0: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let dadt: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Independent host re-derivation: a[i] += h * dadt[i].
    let mut a_host = a0.clone();
    for i in 0..n {
        a_host[i] += h * dadt[i];
    }

    let ptx = crate::ptx_kernels::adjoint_ode_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "adjoint_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a0).expect("d_a");
    let d_dadt = DeviceBuffer::<f32>::from_host(&dadt).expect("d_dadt");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_a.as_device_ptr(), d_dadt.as_device_ptr(), h, n as u32),
        )
        .expect("launch adjoint_step_kernel");
    stream.synchronize().expect("sync");

    let mut a_gpu = vec![0.0_f32; n];
    d_a.copy_to_host(&mut a_gpu).expect("copy a");

    // GPU `fma.rn(h, dadt, a)` is single-rounding vs the host two-rounding
    // `a + h*dadt` — ~1 ulp. 1e-5 relative is tight yet honest.
    let (rel, abs) = worst_diff(&a_gpu, &a_host);
    for i in 0..n {
        assert!(
            close(a_gpu[i], a_host[i], 1e-5, 1e-6),
            "adjoint a[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            a_gpu[i],
            a_host[i]
        );
    }
}

// ===========================================================================
// 5. branch_trunk_dot  —  INDEPENDENT HOST RE-DERIVATION (warp-reduced dot)
// ===========================================================================

#[test]
fn branch_trunk_dot_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // One full warp (block = 32, grid = 1) so every lane participates in the
    // `shfl.sync.bfly.b32` butterfly (membermask 0xFFFFFFFF). p < 32 exercises
    // the inactive-lane (partial = 0) path while keeping the warp converged.
    let p = 24_usize;
    let mut rng = LcgRng::new(0x08A1_4D07);
    let branch: Vec<f32> = (0..p).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let trunk: Vec<f32> = (0..p).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Independent host re-derivation: out = Σ_k branch[k] * trunk[k].
    let mut dot_host = 0.0_f32;
    for k in 0..p {
        dot_host += branch[k] * trunk[k];
    }

    let ptx = crate::ptx_kernels::branch_trunk_dot_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "branch_trunk_dot_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_branch = DeviceBuffer::<f32>::from_host(&branch).expect("d_branch");
    let d_trunk = DeviceBuffer::<f32>::from_host(&trunk).expect("d_trunk");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let params = LaunchParams::new(1_u32, 32_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_branch.as_device_ptr(),
                d_trunk.as_device_ptr(),
                d_out.as_device_ptr(),
                p as u32,
            ),
        )
        .expect("launch branch_trunk_dot_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = [0.0_f32];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Tree reduction (GPU) vs sequential sum (host) over 24 O(1) terms: a few
    // ulp. 1e-4 relative with a small absolute floor is comfortable and still
    // catches a wrong shuffle direction / missing lane.
    assert!(
        close(out_gpu[0], dot_host, 1e-4, 1e-4),
        "branch_trunk_dot mismatch: gpu={} host={}",
        out_gpu[0],
        dot_host
    );
}

// ===========================================================================
// 6. siren_forward  —  INDEPENDENT HOST RE-DERIVATION (sin(ω₀·(Wx+b)))
// ===========================================================================

#[test]
fn siren_forward_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let din = 6_usize;
    let dout = 32_usize;
    let omega0 = 3.0_f32;

    // Keep the pre-activation ω₀·(Wx+b) within a modest range so the kernel's
    // `sin.approx.f32` (accurate to ~2 ulp near the origin) stays in its tight
    // accuracy domain; small weights/inputs give |arg| ≲ 3.
    let mut rng = LcgRng::new(0x5132_E000);
    let w: Vec<f32> = (0..dout * din)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.3)
        .collect();
    let x: Vec<f32> = (0..din)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.5)
        .collect();
    let b: Vec<f32> = (0..dout)
        .map(|_| (rng.next_f32() * 2.0 - 1.0) * 0.2)
        .collect();

    // Independent host re-derivation:
    //   out[i] = sin(ω₀ · (Σ_j w[i,j]·x[j] + b[i])).
    let mut out_host = vec![0.0_f32; dout];
    for i in 0..dout {
        let mut acc = 0.0_f32;
        for j in 0..din {
            acc += w[i * din + j] * x[j];
        }
        acc += b[i];
        out_host[i] = (omega0 * acc).sin();
    }

    let ptx = crate::ptx_kernels::siren_forward_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "siren_forward_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; dout]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(dout as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_x.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                din as u32,
                dout as u32,
                omega0,
            ),
        )
        .expect("launch siren_forward_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; dout];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // `sin.approx.f32` carries a few-ulp absolute error near the origin; over the
    // modest argument range here the divergence stays under ~1e-4 absolute.
    // 1e-3 absolute is generous yet still flags a missing ω₀ scale, a wrong
    // activation, or a dropped bias.
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for i in 0..dout {
        assert!(
            close(out_gpu[i], out_host[i], 1e-4, 1e-3),
            "siren out[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_host[i]
        );
    }
}

// ===========================================================================
// 7. lhs_sample  —  INDEPENDENT HOST RE-DERIVATION (inline counter LCG)
// ===========================================================================

/// Host re-derivation of `lhs_sample_kernel`'s exact integer/float pipeline:
///   state = (seed + tid) · M + A   (wrapping u64)
///   rand  = (state >> 33) as u32   (a 31-bit value)
///   out   = (cell + rand · 2⁻³²) / n.
/// The `cvt.rn.f32.u32` rounds-to-nearest-even (matching Rust `as f32`), the
/// `· 2⁻³²` is an exact power-of-two scale, and `div.rn.f32` matches Rust's `/`,
/// so this is bit-exact with the GPU.
fn lhs_host_value(tid: u64, seed: u64, cell: u32, n: u32) -> f32 {
    const M: u64 = 6_364_136_223_846_793_005;
    const A: u64 = 1_442_695_040_888_963_407;
    let state = seed.wrapping_add(tid).wrapping_mul(M).wrapping_add(A);
    let rand = (state >> 33) as u32;
    let frac = (rand as f32) * (1.0_f32 / 4_294_967_296.0_f32);
    (cell as f32 + frac) / n as f32
}

#[test]
fn lhs_sample_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 16_usize; // number of samples
    let dim = 4_usize; // dimensions
    let seed = 0x1357_9BDF_0246_8ACE_u64;

    // perm[j*n + i] is the integer cell for (sample i, dim j); a valid LHS uses a
    // per-dimension permutation of 0..n. Build deterministic permutations.
    let mut perm = vec![0_u32; dim * n];
    for j in 0..dim {
        for i in 0..n {
            // A simple deterministic permutation of 0..n per dimension.
            perm[j * n + i] = (((i + 3 * j) % n) as u32).min(n as u32 - 1);
        }
    }

    // Host reference over the kernel's flat layout: tid = i*dim + j, out[tid].
    let mut out_host = vec![0.0_f32; n * dim];
    for i in 0..n {
        for j in 0..dim {
            let tid = (i * dim + j) as u64;
            let cell = perm[j * n + i];
            out_host[i * dim + j] = lhs_host_value(tid, seed, cell, n as u32);
        }
    }

    let ptx = crate::ptx_kernels::lhs_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lhs_sample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_perm = DeviceBuffer::<u32>::from_host(&perm).expect("d_perm");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * dim]).expect("d_out");

    let total = (n * dim) as u32;
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_perm.as_device_ptr(),
                seed,
                n as u32,
                dim as u32,
                d_out.as_device_ptr(),
            ),
        )
        .expect("launch lhs_sample_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n * dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Every sample must land in its cell's [cell/n, (cell+1)/n) interval (and the
    // host re-derivation is bit-exact, so the tolerance is essentially ulp-level).
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_host[k], 1e-5, 1e-6),
            "lhs_sample out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
        assert!(
            (0.0..=1.0).contains(&out_gpu[k]),
            "lhs_sample out[{k}] = {} outside [0, 1]",
            out_gpu[k]
        );
    }
}
