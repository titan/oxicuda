//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`] and [`crate::ptx_advanced`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the `oxicuda-snn` / `oxicuda-ot`
//! canaries: device buffers are passed as their `CUdeviceptr` (a `.param .u64`),
//! scalars as the matching Rust scalar (`.param .u32` / `.param .f32`), in the
//! kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance / bit-exactly
//!   to a `pub` CPU function the kernel mirrors:
//!   `correlate_kernel` ↔ [`crate::linalg::mat_t_vec`] (Φᵀr) and
//!   `soft_threshold_kernel` ↔ [`crate::thresholding::soft_threshold`].
//! * **Independent host re-derivation** — the op has no single dedicated crate
//!   `pub fn` with a matching signature (it is fused into a larger routine, or
//!   the only same-named CPU helper computes a *different* op — e.g. the crate's
//!   `hard_threshold_k` is a top-K selector while the kernel is an element-wise
//!   magnitude gate, and `svt` is a full SVD shrink while the kernel is the
//!   per-singular-value shrink). The oracle is an independent Rust re-derivation
//!   of the kernel's documented arithmetic: `hard_threshold_kernel`,
//!   `iht_step_kernel`, `amp_onsager_kernel`, `svt_threshold_kernel`,
//!   `tv_grad_kernel`, `iht_step_cp_async_kernel`, `svt_threshold_ws_kernel`.
//!   These still genuinely fail if ptxas miscompiles or the PTX has a wrong
//!   constant / shift / index, because the host code is independent of the
//!   JIT-compiled PTX.
//!
//! ## PTX bugs found and fixed (on a real RTX A4000, sm_86)
//!
//! ### `iht_step_cp_async_kernel` — two independent bugs
//!
//! 1. **Invalid PTX (ptxas rejected):** the staging copy was emitted as
//!    `cp.async.cg.shared.global [dst], [src], 4`. The `.cg` qualifier only
//!    accepts a **16-byte** copy; for a 4-byte element ptxas aborts with
//!    *"Argument 2 of instruction 'cp.async': unexpected value '4', expected to
//!    be 16"*, so the kernel had never loaded on any GPU. Fixed by switching to
//!    `cp.async.ca.shared.global` (the `.ca` qualifier accepts 4/8/16 bytes).
//! 2. **Divergent-barrier deadlock:** out-of-range threads (`gid >= n`) branched
//!    *past* `bar.sync 0`, so any ragged launch (`n % blockDim.x != 0`) would
//!    hang forever — the active threads in the last block wait at the barrier for
//!    lanes that already exited. Fixed by skipping only the async copy for
//!    out-of-range threads while still letting every launched thread reach the
//!    barrier. `iht_step_cp_async_ragged_matches_host` exercises a ragged launch
//!    (n = 300, block = 256) that would previously deadlock.
//!
//! ## Hopper / Ada paths unreachable on this device (documented, not validated)
//!
//! `correlate_tma_ptx` (TMA `cp.async.bulk`, needs sm ≥ 90) and
//! `correlate_fp8_ptx` (e4m3, needs sm ≥ 89) gate their special bodies behind the
//! SM version and on sm_86 return the **portable** `correlate_kernel`. The
//! fallback path IS validated on-device here (it loads and matches the Φᵀr
//! oracle), but the TMA / FP8 bodies themselves cannot be produced for, or run
//! on, an Ampere RTX A4000 and are therefore not numerically validated.
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug in the
/// kernel string, surfaced as a test failure rather than silently skipped.
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

/// Deterministic FP32 vector of length `n` with values in `[lo, hi)`.
fn fill(rng: &mut LcgRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|_| (f64::from(lo) + f64::from(hi - lo) * rng.next_f64()) as f32)
        .collect()
}

// ===========================================================================
// 1. correlate  —  CRATE ORACLE (linalg::mat_t_vec, c = Φᵀr)
// ===========================================================================

#[test]
fn correlate_matches_crate() {
    use crate::linalg::mat_t_vec;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m = 12_usize;
    let n = 20_usize;
    let mut rng = LcgRng::new(0x0C0_1A7E);
    let phi = fill(&mut rng, m * n, -1.0, 1.0);
    let r = fill(&mut rng, m, -1.0, 1.0);

    // Crate oracle in f64 (f32 -> f64 widening is exact).
    let phi64: Vec<f64> = phi.iter().map(|&v| f64::from(v)).collect();
    let r64: Vec<f64> = r.iter().map(|&v| f64::from(v)).collect();
    let c_cpu_f64 = mat_t_vec(&phi64, m, n, &r64).expect("mat_t_vec");
    let c_cpu: Vec<f32> = c_cpu_f64.iter().map(|&v| v as f32).collect();

    let ptx = crate::ptx_kernels::correlate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "correlate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_c = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_c");
    let d_phi = DeviceBuffer::<f32>::from_host(&phi).expect("d_phi");
    let d_r = DeviceBuffer::<f32>::from_host(&r).expect("d_r");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_c.as_device_ptr(),
                d_phi.as_device_ptr(),
                d_r.as_device_ptr(),
                m as u32,
                n as u32,
            ),
        )
        .expect("launch correlate_kernel");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; n];
    d_c.copy_to_host(&mut c_gpu).expect("copy c");

    let (rel, abs) = worst_diff(&c_gpu, &c_cpu);
    for j in 0..n {
        assert!(
            close(c_gpu[j], c_cpu[j], 1e-4, 1e-5),
            "correlate c[{j}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            c_gpu[j],
            c_cpu[j]
        );
    }
}

// ===========================================================================
// 2. hard_threshold  —  HOST RE-DERIVATION (element-wise magnitude gate)
// ===========================================================================
//
// The crate's same-named `hard_threshold_k` is a *top-K* selector, a different
// op; the kernel is the element-wise gate `out[i] = |x[i]| > t ? x[i] : 0`. The
// kernel passes the value through unchanged (a `selp`, no arithmetic) and the
// FP32 magnitude compare is identical on host and device, so the result is
// **bit-exact**.

#[test]
fn hard_threshold_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let threshold = 0.5_f32;
    let mut rng = LcgRng::new(0x4A2D_7011);
    // Values in [-2, 2); none land near ±threshold so the gate is unambiguous.
    let x = fill(&mut rng, n, -2.0, 2.0);

    let expected: Vec<f32> = x
        .iter()
        .map(|&v| if v.abs() > threshold { v } else { 0.0 })
        .collect();

    let ptx = crate::ptx_kernels::hard_threshold_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hard_threshold_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_x.as_device_ptr(),
                threshold,
                n as u32,
            ),
        )
        .expect("launch hard_threshold_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for (k, (&g, &e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "hard_threshold out[{k}] mismatch: gpu={g} host={e} (x={})",
            x[k]
        );
    }
}

// ===========================================================================
// 3. soft_threshold  —  CRATE ORACLE (thresholding::soft_threshold)
// ===========================================================================

#[test]
fn soft_threshold_matches_crate() {
    use crate::thresholding::soft_threshold;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let lambda = 0.3_f32;
    let mut rng = LcgRng::new(0x50F7_0FED);
    let x = fill(&mut rng, n, -2.0, 2.0);

    // Crate oracle (f64); compared within FP32 tolerance.
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let y_cpu_f64 = soft_threshold(&x64, f64::from(lambda));
    let y_cpu: Vec<f32> = y_cpu_f64.iter().map(|&v| v as f32).collect();

    let ptx = crate::ptx_kernels::soft_threshold_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "soft_threshold_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_y.as_device_ptr(), d_x.as_device_ptr(), lambda, n as u32),
        )
        .expect("launch soft_threshold_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    let (rel, abs) = worst_diff(&y_gpu, &y_cpu);
    for k in 0..n {
        assert!(
            close(y_gpu[k], y_cpu[k], 1e-5, 1e-6),
            "soft_threshold y[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[k],
            y_cpu[k]
        );
    }
}

// ===========================================================================
// 4. iht_step  —  HOST RE-DERIVATION (x[i] += mu * grad[i])
// ===========================================================================

#[test]
fn iht_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let mu = 0.25_f32;
    let mut rng = LcgRng::new(0x1147_5742);
    let x0 = fill(&mut rng, n, -1.0, 1.0);
    let grad = fill(&mut rng, n, -1.0, 1.0);

    let expected: Vec<f32> = x0
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| x + mu * g)
        .collect();

    let ptx = crate::ptx_kernels::iht_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "iht_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x0).expect("d_x");
    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), d_grad.as_device_ptr(), mu, n as u32),
        )
        .expect("launch iht_step_kernel");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    let (rel, abs) = worst_diff(&x_gpu, &expected);
    for k in 0..n {
        // GPU `fma.rn(mu, grad, x)` is single-rounding vs the host's two-rounding
        // `x + mu*grad`; ~1 ulp divergence, well inside 1e-5 relative.
        assert!(
            close(x_gpu[k], expected[k], 1e-5, 1e-6),
            "iht_step x[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            x_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 5. amp_onsager  —  HOST RE-DERIVATION (z_new = residual + (b/m)*z_prev)
// ===========================================================================

#[test]
fn amp_onsager_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m = 100_usize;
    let b_over_m = 0.7_f32;
    let mut rng = LcgRng::new(0xA3B0_05A6);
    let residual = fill(&mut rng, m, -1.5, 1.5);
    let z_prev = fill(&mut rng, m, -1.5, 1.5);

    let expected: Vec<f32> = residual
        .iter()
        .zip(z_prev.iter())
        .map(|(&r, &z)| r + b_over_m * z)
        .collect();

    let ptx = crate::ptx_kernels::amp_onsager_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "amp_onsager_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_new = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m]).expect("d_new");
    let d_res = DeviceBuffer::<f32>::from_host(&residual).expect("d_res");
    let d_prev = DeviceBuffer::<f32>::from_host(&z_prev).expect("d_prev");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(m as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_new.as_device_ptr(),
                d_res.as_device_ptr(),
                d_prev.as_device_ptr(),
                b_over_m,
                m as u32,
            ),
        )
        .expect("launch amp_onsager_kernel");
    stream.synchronize().expect("sync");

    let mut z_gpu = vec![0.0_f32; m];
    d_new.copy_to_host(&mut z_gpu).expect("copy z_new");

    let (rel, abs) = worst_diff(&z_gpu, &expected);
    for k in 0..m {
        assert!(
            close(z_gpu[k], expected[k], 1e-5, 1e-6),
            "amp_onsager z_new[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            z_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 6. svt_threshold  —  HOST RE-DERIVATION (sigma' = max(sigma - tau, 0))
// ===========================================================================
//
// The crate's `svt` is a full SVD-based matrix-completion shrink; this kernel is
// only the per-singular-value shrink. The op is `max(sigma - tau, 0)` with one
// FP32 subtract and one FP32 max, identical on host and device, so the result is
// **bit-exact**.

#[test]
fn svt_threshold_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let tau = 0.4_f32;
    let mut rng = LcgRng::new(0x5174_7A11);
    let sigma = fill(&mut rng, n, 0.0, 2.0);

    let expected: Vec<f32> = sigma.iter().map(|&s| (s - tau).max(0.0)).collect();

    let ptx = crate::ptx_kernels::svt_threshold_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "svt_threshold_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    let d_in = DeviceBuffer::<f32>::from_host(&sigma).expect("d_in");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), d_in.as_device_ptr(), tau, n as u32),
        )
        .expect("launch svt_threshold_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for (k, (&g, &e)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "svt_threshold out[{k}] mismatch: gpu={g} host={e} (sigma={})",
            sigma[k]
        );
    }
}

// ===========================================================================
// 7. tv_grad  —  HOST RE-DERIVATION (forward difference, grad[n-1] = 0)
// ===========================================================================
//
// `grad[i] = x[i+1] - x[i]` for `i < n-1`, `grad[n-1] = 0`. A single FP32
// subtract, identical on host and device, so the result is **bit-exact**.

#[test]
fn tv_grad_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let mut rng = LcgRng::new(0x7174_6D11);
    let x = fill(&mut rng, n, -3.0, 3.0);

    let mut expected = vec![0.0_f32; n];
    for i in 0..n - 1 {
        expected[i] = x[i + 1] - x[i];
    }
    // expected[n-1] stays 0.

    let ptx = crate::ptx_kernels::tv_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tv_grad_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grad = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_grad");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_grad.as_device_ptr(), d_x.as_device_ptr(), n as u32),
        )
        .expect("launch tv_grad_kernel");
    stream.synchronize().expect("sync");

    let mut grad_gpu = vec![0.0_f32; n];
    d_grad.copy_to_host(&mut grad_gpu).expect("copy grad");

    for (k, (&g, &e)) in grad_gpu.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "tv_grad grad[{k}] mismatch: gpu={g} host={e}"
        );
    }
}

// ===========================================================================
// 8. iht_step_cp_async  —  HOST RE-DERIVATION (Ampere cp.async staged x += mu*grad)
// ===========================================================================
//
// Same arithmetic as `iht_step` but the gradient element is staged
// global -> shared via `cp.async.ca.shared.global` before the FMA. Two PTX bugs
// were found and fixed here (see the module docs): the `.cg`/4-byte ptxas
// rejection, and the divergent-barrier deadlock on ragged launches.

fn run_iht_cp_async(fx: &GpuFixture, n: usize, block: u32) {
    let mu = 0.3125_f32; // exact in FP32
    let mut rng = LcgRng::new(0xCA50_0000 ^ n as u64);
    let x0 = fill(&mut rng, n, -1.0, 1.0);
    let grad = fill(&mut rng, n, -1.0, 1.0);

    let expected: Vec<f32> = x0
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| x + mu * g)
        .collect();

    let ptx = crate::ptx_advanced::iht_step_cp_async_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "iht_step_cp_async_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x0).expect("d_x");
    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");

    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), d_grad.as_device_ptr(), mu, n as u32),
        )
        .expect("launch iht_step_cp_async_kernel");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    let (rel, abs) = worst_diff(&x_gpu, &expected);
    for k in 0..n {
        assert!(
            close(x_gpu[k], expected[k], 1e-5, 1e-6),
            "iht_cp_async x[{k}] mismatch (n={n}, block={block}): gpu={} host={} \
             (worst rel={rel:e} abs={abs:e})",
            x_gpu[k],
            expected[k]
        );
    }
}

#[test]
fn iht_step_cp_async_aligned_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Block-aligned launch (n == blockDim.x): every thread is in range.
    run_iht_cp_async(&fx, 256, 256);
}

#[test]
fn iht_step_cp_async_ragged_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Ragged launch: n is NOT a multiple of the block. Before the divergent-
    // barrier fix this DEADLOCKED (out-of-range threads skipped `bar.sync`).
    run_iht_cp_async(&fx, 300, 256);
}

// ===========================================================================
// 9. svt_threshold_ws  —  HOST RE-DERIVATION (shrink + warp-shuffle nuclear norm)
// ===========================================================================
//
// Element-wise `sigma' = max(sigma - tau, 0)` (bit-exact, as in test 6) PLUS a
// full-warp `shfl.sync.down` butterfly that sums `sigma'` and writes one partial
// per warp (lane 0). With a single 32-thread block, lane 0 holds the total
// thresholded nuclear-norm contribution `Σ sigma'`.

#[test]
fn svt_threshold_ws_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // n < 32 so the block has out-of-range lanes (they must contribute 0 to the
    // warp sum); a single full warp keeps the shuffle mask 0xFFFFFFFF valid.
    let n = 24_usize;
    let tau = 0.7_f32;
    let block = 32_u32;
    let mut rng = LcgRng::new(0x5374_7757);
    let sigma = fill(&mut rng, n, 0.0, 2.0);

    let thresholded: Vec<f32> = sigma.iter().map(|&s| (s - tau).max(0.0)).collect();
    let nucnorm_expected: f32 = thresholded.iter().copied().sum();

    let ptx = crate::ptx_advanced::svt_threshold_warpshuffle_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "svt_threshold_ws_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");
    // One warp -> one partial.
    let d_partial = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; 1]).expect("d_partial");
    let d_in = DeviceBuffer::<f32>::from_host(&sigma).expect("d_in");

    let params = LaunchParams::new(1_u32, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_out.as_device_ptr(),
                d_partial.as_device_ptr(),
                d_in.as_device_ptr(),
                tau,
                n as u32,
            ),
        )
        .expect("launch svt_threshold_ws_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    let mut partial_gpu = vec![0.0_f32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");
    d_partial
        .copy_to_host(&mut partial_gpu)
        .expect("copy partial");

    // Element-wise shrink is bit-exact (single sub + max, as in svt_threshold).
    for (k, (&g, &e)) in out_gpu.iter().zip(thresholded.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "svt_ws out[{k}] mismatch: gpu={g} host={e} (sigma={})",
            sigma[k]
        );
    }
    // Warp-shuffle reduced nuclear norm: the GPU sums in lane order via a
    // butterfly, the host sums sequentially, so a few-ulp reorder is expected.
    assert!(
        close(partial_gpu[0], nucnorm_expected, 1e-5, 1e-6),
        "svt_ws nuclear-norm partial mismatch: gpu={} host={}",
        partial_gpu[0],
        nucnorm_expected
    );
}

// ===========================================================================
// 10. correlate_tma / correlate_fp8 fallbacks  —  LOAD + Φᵀr ORACLE on sm_86
// ===========================================================================
//
// On an Ampere RTX A4000 (sm_86 < 89/90) both architecture-specialised
// correlate variants gate their special bodies behind the SM version and return
// the PORTABLE `correlate_kernel`. We validate that fallback runs on-device and
// matches the Φᵀr oracle. The TMA (`cp.async.bulk`, sm ≥ 90) and FP8 (e4m3,
// sm ≥ 89) bodies cannot be produced for, or executed on, this device and are
// therefore NOT numerically validated here (documented honestly).

fn run_correlate_fallback(fx: &GpuFixture, ptx: &str) {
    use crate::linalg::mat_t_vec;

    let m = 10_usize;
    let n = 16_usize;
    let mut rng = LcgRng::new(0xFA11_BACC);
    let phi = fill(&mut rng, m * n, -1.0, 1.0);
    let r = fill(&mut rng, m, -1.0, 1.0);

    let phi64: Vec<f64> = phi.iter().map(|&v| f64::from(v)).collect();
    let r64: Vec<f64> = r.iter().map(|&v| f64::from(v)).collect();
    let c_cpu_f64 = mat_t_vec(&phi64, m, n, &r64).expect("mat_t_vec");
    let c_cpu: Vec<f32> = c_cpu_f64.iter().map(|&v| v as f32).collect();

    // The fallback keeps the portable entry name `correlate_kernel`.
    let kernel = load_kernel(ptx, "correlate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_c = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_c");
    let d_phi = DeviceBuffer::<f32>::from_host(&phi).expect("d_phi");
    let d_r = DeviceBuffer::<f32>::from_host(&r).expect("d_r");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_c.as_device_ptr(),
                d_phi.as_device_ptr(),
                d_r.as_device_ptr(),
                m as u32,
                n as u32,
            ),
        )
        .expect("launch correlate fallback");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; n];
    d_c.copy_to_host(&mut c_gpu).expect("copy c");

    for j in 0..n {
        assert!(
            close(c_gpu[j], c_cpu[j], 1e-4, 1e-5),
            "correlate fallback c[{j}] mismatch: gpu={} cpu={}",
            c_gpu[j],
            c_cpu[j]
        );
    }
}

#[test]
fn correlate_tma_fallback_matches_crate_on_sm86() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // sm_86 < 90: the TMA path is unavailable, so this must be the portable
    // correlate; assert the string identity then validate it on-device.
    let ptx = crate::ptx_advanced::correlate_tma_ptx(fx.sm);
    if fx.sm >= 90 {
        return; // would exercise the real TMA body — out of scope for Ampere.
    }
    assert_eq!(
        ptx,
        crate::ptx_kernels::correlate_ptx(fx.sm),
        "sm {} TMA fallback must equal portable correlate",
        fx.sm
    );
    run_correlate_fallback(&fx, &ptx);
}

#[test]
fn correlate_fp8_fallback_matches_crate_on_sm86() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = crate::ptx_advanced::correlate_fp8_ptx(fx.sm);
    if fx.sm >= 89 {
        return; // would exercise the real e4m3 body — out of scope for Ampere.
    }
    assert_eq!(
        ptx,
        crate::ptx_kernels::correlate_ptx(fx.sm),
        "sm {} FP8 fallback must equal portable correlate",
        fx.sm
    );
    run_correlate_fallback(&fx, &ptx);
}
