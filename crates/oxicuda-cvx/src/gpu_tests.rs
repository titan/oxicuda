//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` harnesses: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .f32` /
//! `.param .u32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! All seven kernels are simple, one-thread-per-element maps (no shared memory,
//! no reductions, no shuffles, no `ex2`/`lg2`), so each has a single, exact CPU
//! oracle:
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to the `pub`
//!   CPU function the kernel mirrors:
//!   `soft_threshold_kernel` ↔ [`crate::prox_ops::soft_threshold`] (the prox of
//!   `λ|·|`), and `proj_l2_ball_kernel` ↔ [`crate::projection::project_l2_ball`]
//!   (the kernel applies the precomputed `min(1, r/‖x‖)` scale that the CPU
//!   projection computes internally; the test passes that exact scale and checks
//!   the scaled vector matches the crate projection).
//! * **Independent host re-derivation** — the op is a fused inner step of a
//!   larger CPU routine (AXPY, the gradient step, FISTA momentum extrapolation,
//!   the ADMM dual update, and the element-wise shrink of the simplex
//!   projection), so the oracle is an independent Rust re-implementation of the
//!   kernel's documented per-element arithmetic. These still genuinely fail if
//!   ptxas miscompiles or the PTX has a wrong constant / shift / index, because
//!   the host code is independent of the JIT-compiled PTX.
//!
//! ## PTX audit result
//!
//! All seven kernels are valid PTX (ptxas accepts them on sm_86) and compute the
//! correct value — no base-2 exp/log bug (none use `ex2`/`lg2`), no invalid-PTX
//! rejection, no race (every kernel is an independent per-element map), and no
//! hollow stub (every kernel performs a real `st.global.f32`). No bug was found
//! or fixed in this crate's PTX.
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
/// A failure here means ptxas rejected the PTX (an invalid-PTX bug) or the entry
/// is missing — both are real defects, so the test panics rather than skips.
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

/// Deterministic spread in `[lo, hi)` from the crate's `LcgRng` (`÷2^53` uniform
/// `next_f64`, cast to `f32`).  Fixed seeds keep every run reproducible.
fn fill(rng: &mut LcgRng, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    (0..n)
        .map(|_| lo + (hi - lo) * rng.next_f64() as f32)
        .collect()
}

const BLOCK: u32 = 256;

// ===========================================================================
// 1. axpy  —  INDEPENDENT HOST RE-DERIVATION (y = alpha*x + y)
// ===========================================================================

#[test]
fn axpy_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let alpha = 1.75_f32;
    let mut rng = LcgRng::new(0x0A_C0FF_EE01);
    let x = fill(&mut rng, n, -2.0, 2.0);
    let y0 = fill(&mut rng, n, -3.0, 3.0);

    // Host: single-rounding fma to match the GPU's `fma.rn.f32`.
    let y_host: Vec<f32> = (0..n).map(|i| alpha.mul_add(x[i], y0[i])).collect();

    let ptx = crate::ptx_kernels::axpy_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "axpy_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y0).expect("d_y");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
    kernel
        .launch(
            &params,
            &stream,
            &(d_y.as_device_ptr(), d_x.as_device_ptr(), alpha, n as u32),
        )
        .expect("launch axpy_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    let (rel, abs) = worst_diff(&y_gpu, &y_host);
    for i in 0..n {
        assert!(
            close(y_gpu[i], y_host[i], 1e-6, 1e-6),
            "axpy y[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[i],
            y_host[i]
        );
    }
}

// ===========================================================================
// 2. soft_threshold  —  CRATE ORACLE (prox_ops::soft_threshold, the L1 prox)
// ===========================================================================

#[test]
fn soft_threshold_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let lambda = 0.5_f32;
    // x spread across [-2, 2] so ~half the entries fall inside the [-λ, λ] band
    // (shrunk to exactly 0) and half outside (shrunk toward 0 by λ). Random f32
    // never land exactly on |x| = λ, so the comparison is never knife-edge.
    let mut rng = LcgRng::new(0x50F7_0001);
    let x = fill(&mut rng, n, -2.0, 2.0);

    // CRATE oracle: the scalar prox of λ|·|, computed in f64 then cast.
    let c_host: Vec<f32> = x
        .iter()
        .map(|&xi| crate::prox_ops::soft_threshold(f64::from(xi), f64::from(lambda)) as f32)
        .collect();

    let ptx = crate::ptx_kernels::soft_threshold_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "soft_threshold_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
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

    let (rel, abs) = worst_diff(&y_gpu, &c_host);
    for i in 0..n {
        // The band entries must be bit-exact zero, the outside entries within
        // one f32 ulp of the (sub then conditional-negate) host arithmetic.
        assert!(
            close(y_gpu[i], c_host[i], 1e-6, 1e-7),
            "soft_threshold y[{i}] mismatch: gpu={} crate={} x={} \
             (worst rel={rel:e} abs={abs:e})",
            y_gpu[i],
            c_host[i],
            x[i]
        );
    }
}

// ===========================================================================
// 3. simplex_proj  —  INDEPENDENT HOST RE-DERIVATION (y = max(x - tau, 0))
// ===========================================================================

#[test]
fn simplex_proj_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let tau = 0.3_f32;
    // x in [-1, 1] so the shrink-and-clamp produces a mix of clamped-to-zero and
    // positive entries — exactly the element-wise step the simplex projection
    // applies once the host has solved for `tau`.
    let mut rng = LcgRng::new(0x5119_7E00);
    let x = fill(&mut rng, n, -1.0, 1.0);

    let host: Vec<f32> = x.iter().map(|&xi| (xi - tau).max(0.0)).collect();

    let ptx = crate::ptx_kernels::simplex_proj_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "simplex_proj_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
    kernel
        .launch(
            &params,
            &stream,
            &(d_y.as_device_ptr(), d_x.as_device_ptr(), tau, n as u32),
        )
        .expect("launch simplex_proj_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    let (rel, abs) = worst_diff(&y_gpu, &host);
    for i in 0..n {
        assert!(
            close(y_gpu[i], host[i], 1e-6, 1e-7),
            "simplex_proj y[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[i],
            host[i]
        );
    }
}

// ===========================================================================
// 4. gradient_step  —  INDEPENDENT HOST RE-DERIVATION (x = x - alpha*g)
// ===========================================================================

#[test]
fn gradient_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let alpha = 0.05_f32;
    let mut rng = LcgRng::new(0x64AD_57E9);
    let x0 = fill(&mut rng, n, -2.0, 2.0);
    let g = fill(&mut rng, n, -1.0, 1.0);

    // Host: `x - alpha*g` via single-rounding fma(-alpha, g, x) to match the GPU.
    let host: Vec<f32> = (0..n).map(|i| (-alpha).mul_add(g[i], x0[i])).collect();

    let ptx = crate::ptx_kernels::gradient_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gradient_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // In-place: the kernel overwrites x.
    let d_x = DeviceBuffer::<f32>::from_host(&x0).expect("d_x");
    let d_g = DeviceBuffer::<f32>::from_host(&g).expect("d_g");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), d_g.as_device_ptr(), alpha, n as u32),
        )
        .expect("launch gradient_step_kernel");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    let (rel, abs) = worst_diff(&x_gpu, &host);
    for i in 0..n {
        assert!(
            close(x_gpu[i], host[i], 1e-6, 1e-7),
            "gradient_step x[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            x_gpu[i],
            host[i]
        );
    }
}

// ===========================================================================
// 5. fista_extrapolate  —  INDEPENDENT HOST RE-DERIVATION
//    (y = x_new + beta*(x_new - x_old))
// ===========================================================================

#[test]
fn fista_extrapolate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let beta = 0.732_f32; // a representative (t_k-1)/t_{k+1} momentum
    let mut rng = LcgRng::new(0xF157_A001);
    let x_new = fill(&mut rng, n, -2.0, 2.0);
    let x_old = fill(&mut rng, n, -2.0, 2.0);

    // Host: delta = x_new - x_old; y = fma(beta, delta, x_new) — matches the
    // kernel's `sub` then `fma.rn`.
    let host: Vec<f32> = (0..n)
        .map(|i| {
            let delta = x_new[i] - x_old[i];
            beta.mul_add(delta, x_new[i])
        })
        .collect();

    let ptx = crate::ptx_kernels::fista_extrapolate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fista_extrapolate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_y");
    let d_new = DeviceBuffer::<f32>::from_host(&x_new).expect("d_new");
    let d_old = DeviceBuffer::<f32>::from_host(&x_old).expect("d_old");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_y.as_device_ptr(),
                d_new.as_device_ptr(),
                d_old.as_device_ptr(),
                beta,
                n as u32,
            ),
        )
        .expect("launch fista_extrapolate_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    let (rel, abs) = worst_diff(&y_gpu, &host);
    for i in 0..n {
        assert!(
            close(y_gpu[i], host[i], 1e-6, 1e-6),
            "fista_extrapolate y[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[i],
            host[i]
        );
    }
}

// ===========================================================================
// 6. admm_dual_update  —  INDEPENDENT HOST RE-DERIVATION (u = u + residual)
// ===========================================================================

#[test]
fn admm_dual_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1000_usize;
    let mut rng = LcgRng::new(0xAD77_0001);
    let u0 = fill(&mut rng, n, -3.0, 3.0);
    let residual = fill(&mut rng, n, -1.0, 1.0);

    let host: Vec<f32> = (0..n).map(|i| u0[i] + residual[i]).collect();

    let ptx = crate::ptx_kernels::admm_dual_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "admm_dual_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // In-place: the kernel overwrites u.
    let d_u = DeviceBuffer::<f32>::from_host(&u0).expect("d_u");
    let d_res = DeviceBuffer::<f32>::from_host(&residual).expect("d_res");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
    kernel
        .launch(
            &params,
            &stream,
            &(d_u.as_device_ptr(), d_res.as_device_ptr(), n as u32),
        )
        .expect("launch admm_dual_update_kernel");
    stream.synchronize().expect("sync");

    let mut u_gpu = vec![0.0_f32; n];
    d_u.copy_to_host(&mut u_gpu).expect("copy u");

    let (rel, abs) = worst_diff(&u_gpu, &host);
    for i in 0..n {
        assert!(
            close(u_gpu[i], host[i], 1e-6, 1e-7),
            "admm_dual_update u[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            u_gpu[i],
            host[i]
        );
    }
}

// ===========================================================================
// 7. proj_l2_ball  —  CRATE ORACLE (projection::project_l2_ball)
// ===========================================================================

#[test]
fn proj_l2_ball_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let mut rng = LcgRng::new(0x12BA_1107);
    // Vector with ‖x‖ well above the chosen radius so the projection actually
    // shrinks (scale < 1); otherwise the op is a trivial scale-by-one.
    let x = fill(&mut rng, n, -1.0, 1.0);

    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let norm = x64.iter().map(|v| v * v).sum::<f64>().sqrt();
    let r = 0.25 * norm; // strictly inside ‖x‖, so the ball clips
    assert!(r < norm, "test setup error: radius must be below the norm");

    // CRATE oracle: the full L2-ball projection. The kernel receives the same
    // `scale = r/‖x‖` the projection computes internally and applies it per
    // element, so the GPU result must equal the crate projection.
    let crate_proj = crate::projection::project_l2_ball(&x64, r).expect("project_l2_ball");
    let host: Vec<f32> = crate_proj.iter().map(|&v| v as f32).collect();
    let scale = (r / norm) as f32;

    let ptx = crate::ptx_kernels::proj_l2_ball_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "proj_l2_ball_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // In-place: the kernel scales x.
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");

    let params = LaunchParams::new(grid_1d(n as u32, BLOCK), BLOCK);
    kernel
        .launch(&params, &stream, &(d_x.as_device_ptr(), scale, n as u32))
        .expect("launch proj_l2_ball_kernel");
    stream.synchronize().expect("sync");

    let mut x_gpu = vec![0.0_f32; n];
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    // Tolerance: the kernel multiplies in f32 (`scale` rounded once from f64),
    // the crate scales in f64; the divergence is a couple of f32 ulp. 1e-5
    // relative comfortably covers it yet flags any gross error (wrong factor).
    let (rel, abs) = worst_diff(&x_gpu, &host);
    for i in 0..n {
        assert!(
            close(x_gpu[i], host[i], 1e-5, 1e-6),
            "proj_l2_ball x[{i}] mismatch: gpu={} crate={} (worst rel={rel:e} abs={abs:e})",
            x_gpu[i],
            host[i]
        );
    }
}
