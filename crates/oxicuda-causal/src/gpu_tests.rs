//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` harnesses: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .u32` /
//! `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a crate
//!   CPU function the kernel mirrors:
//!   `expm_pade_kernel` ↔ [`crate::discovery::notears::expm_pade`].
//! * **Independent host re-derivation** — the kernel's op is fused into a larger
//!   CPU routine (no callable standalone `pub fn`), so the oracle is an
//!   independent Rust re-implementation of the kernel's *documented* arithmetic:
//!   `notears_loss_kernel` (matches the private `NotearsSem::compute_gradient`
//!   formula `grad = (1/n)·Xᵀ(XW − X)`), `propensity_logit_kernel`
//!   (clamped logistic prediction), `ipw_estimator_kernel` (the IPW ATE *sum*,
//!   i.e. `n ·`[`crate::effect::ipw::ipw_ate`]), `dml_residual_kernel`
//!   (elementwise `Y − g`, `T − m`), and `causal_split_score_kernel`
//!   (heterogeneous-effect split criterion). These still genuinely fail if
//!   ptxas miscompiles or the PTX has a wrong constant / shift / index, because
//!   the host code is independent of the JIT-compiled PTX.
//! * **Documented-divergence host re-derivation** — `partial_corr_kernel` does
//!   NOT compute the crate's mean-centered, residualised Fisher-Z partial
//!   correlation (`discovery::pc::partial_corr`). It computes the *uncentered*
//!   correlation `Σ xᵢxⱼ / √(Σ xᵢ² · Σ xⱼ²)` (a cosine similarity). The test
//!   validates that actual arithmetic and reports the name/semantics mismatch.
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
/// A `Module::from_ptx` failure means ptxas rejected the hand-written PTX — a
/// real bug, surfaced as a panic rather than a skipped test.
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
// 1. partial_corr  —  DOCUMENTED-DIVERGENCE host re-derivation (uncentered corr)
// ===========================================================================

#[test]
fn partial_corr_matches_uncentered_correlation() {
    // HONEST SCOPE: `partial_corr_kernel` is mis-named. The crate's CPU partial
    // correlation (`discovery::pc::partial_corr`) mean-centers the columns and
    // residualises on the conditioning set; the kernel does NEITHER. It computes
    // the uncentered correlation (cosine similarity)
    //     corr[i,j] = Σ_s x[s,i]·x[s,j] / √(Σ_s x[s,i]² · Σ_s x[s,j]²),
    // with a denominator floor of 1e-6. We validate that actual arithmetic; the
    // divergence from a true partial correlation is reported, not papered over.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 16_usize; // samples
    let d = 5_usize; // variables

    // Strictly positive data ⇒ every column norm ≫ 1e-6, so the kernel's
    // denominator floor never triggers and the comparison is unambiguous.
    let mut rng = LcgRng::new(0x9A1_C033);
    let x: Vec<f32> = (0..n * d).map(|_| 0.25 + rng.next_f32()).collect();

    // Independent host re-derivation of the documented per-(i,j) arithmetic.
    let mut corr_host = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut sxy = 0.0_f32;
            let mut sxx = 0.0_f32;
            let mut syy = 0.0_f32;
            for s in 0..n {
                let xi = x[s * d + i];
                let xj = x[s * d + j];
                sxy += xi * xj;
                sxx += xi * xi;
                syy += xj * xj;
            }
            let mut denom = (sxx * syy).sqrt();
            if denom < 1e-6 {
                denom = 1.0;
            }
            corr_host[i * d + j] = sxy / denom;
        }
    }

    let ptx = crate::ptx_kernels::partial_corr_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "partial_corr_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_corr = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; d * d]).expect("d_corr");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d((d * d) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_corr.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch partial_corr_kernel");
    stream.synchronize().expect("sync");

    let mut corr_gpu = vec![0.0_f32; d * d];
    d_corr.copy_to_host(&mut corr_gpu).expect("copy corr");

    // Diagonal of an uncentered correlation is exactly 1.
    for i in 0..d {
        assert!(
            close(corr_gpu[i * d + i], 1.0, 1e-4, 1e-5),
            "partial_corr diagonal[{i}] = {} (expected ~1.0)",
            corr_gpu[i * d + i]
        );
    }
    // GPU uses `fma` for the three running sums (single rounding); the host uses
    // mul+add (two roundings). Over n = 16 positive terms the divergence is a few
    // ulp; 2e-4 relative comfortably covers it yet flags any wrong index/formula.
    let (rel, abs) = worst_diff(&corr_gpu, &corr_host);
    for k in 0..corr_gpu.len() {
        assert!(
            close(corr_gpu[k], corr_host[k], 2e-4, 1e-5),
            "partial_corr[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            corr_gpu[k],
            corr_host[k]
        );
    }
}

// ===========================================================================
// 2. notears_loss  —  INDEPENDENT HOST RE-DERIVATION (matches compute_gradient)
// ===========================================================================

#[test]
fn notears_loss_matches_host_gradient() {
    // The crate computes this gradient inside the private
    // `NotearsSem::compute_gradient` (`grad = (1/n)·Xᵀ(XW − X)`). The kernel
    // computes the same quantity per output cell:
    //   grad[j,k] = (1/n)·Σ_i X[i,j]·(Σ_l X[i,l]·W[l,k] − X[i,k]).
    // The oracle is an independent host re-derivation of that formula.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 12_usize; // samples
    let d = 4_usize; // variables

    let mut rng = LcgRng::new(0x0007_EA05);
    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..d * d).map(|_| rng.next_f32() * 0.6 - 0.3).collect();

    // Host re-derivation.
    let inv_n = 1.0_f32 / n as f32;
    let mut grad_host = vec![0.0_f32; d * d];
    for j in 0..d {
        for k in 0..d {
            let mut acc = 0.0_f32;
            for i in 0..n {
                let mut xw = 0.0_f32;
                for l in 0..d {
                    xw += x[i * d + l] * w[l * d + k];
                }
                let resid = xw - x[i * d + k];
                acc += x[i * d + j] * resid;
            }
            grad_host[j * d + k] = acc * inv_n;
        }
    }

    let ptx = crate::ptx_kernels::notears_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "notears_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_grad = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; d * d]).expect("d_grad");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d((d * d) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_w.as_device_ptr(),
                d_grad.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch notears_loss_kernel");
    stream.synchronize().expect("sync");

    let mut grad_gpu = vec![0.0_f32; d * d];
    d_grad.copy_to_host(&mut grad_gpu).expect("copy grad");

    let (rel, abs) = worst_diff(&grad_gpu, &grad_host);
    for k in 0..grad_gpu.len() {
        assert!(
            close(grad_gpu[k], grad_host[k], 1e-4, 1e-5),
            "notears grad[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            grad_gpu[k],
            grad_host[k]
        );
    }
}

// ===========================================================================
// 3. expm_pade  —  CRATE ORACLE (discovery::notears::expm_pade)
// ===========================================================================
//
// PTX BUG FOUND AND FIXED here. The original `expm_pade_kernel` shipped INVALID
// PTX that ptxas rejects on the RTX A4000 (sm_86): every shared-memory access
// used a scaled-symbol address `[sh_cur + %r5*4]` (44 sites), which PTX forbids,
// plus two `st.shared.f32 [...], <immediate>` stores (PTX `st` cannot take an
// immediate data operand). Fix applied in `ptx_kernels.rs`: each shared base is
// materialised once with `mov.u32 %rN, sh_*`, and every access computes its byte
// address with `mad.lo.u32 %r37, idx, 4, base` before `ld.shared`/`st.shared`;
// the identity right-half stores the already-computed `I[i,j]` register. With
// this fix the kernel JIT-loads and its `expm(A)` matches the CPU reference.

#[test]
fn expm_pade_matches_crate_cpu() {
    use crate::discovery::notears::expm_pade;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let d = 4_usize;
    // A modest-norm matrix that exercises the scaling-and-squaring path
    // (‖A‖∞ > θ = 0.5 ⇒ s ≥ 1) while staying well-conditioned for Padé(1,1) and
    // the in-block Gauss-Jordan inversion of V.
    let mut rng = LcgRng::new(0x000E_0FA1);
    let a: Vec<f32> = (0..d * d).map(|_| rng.next_f32() * 0.6 - 0.3).collect();

    // ---- CPU reference (crate oracle) ----
    let e_cpu = expm_pade(&a, d).expect("cpu expm_pade");

    let ptx = crate::ptx_kernels::expm_pade_ptx(fx.sm);
    // LOAD: a ptxas rejection would mean the invalid-PTX fix regressed.
    let kernel = load_kernel(&ptx, "expm_pade_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; d * d]).expect("d_out");

    // Single cooperative block of d*d threads (one thread per matrix element).
    let params = LaunchParams::new(1_u32, (d * d) as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_a.as_device_ptr(), d_out.as_device_ptr(), d as u32),
        )
        .expect("launch expm_pade_kernel");
    stream.synchronize().expect("sync");

    let mut e_gpu = vec![0.0_f32; d * d];
    d_out.copy_to_host(&mut e_gpu).expect("copy out");

    // Both paths are FP32 Padé(1,1) with scaling-and-squaring; the GPU uses
    // `ex2.approx`/`lg2.approx` only to derive the *integer* scaling exponent s
    // (the matrix arithmetic is exact-rounding `fma`/`div.rn`). A handful of ulp
    // accumulate across the inversion + s squarings; 3e-3 absolute / 5e-3
    // relative is comfortable yet flags any wrong address / transpose / index in
    // the rewritten shared addressing by orders of magnitude.
    let (rel, abs) = worst_diff(&e_gpu, &e_cpu);
    for k in 0..e_gpu.len() {
        assert!(
            close(e_gpu[k], e_cpu[k], 5e-3, 3e-3),
            "expm[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            e_gpu[k],
            e_cpu[k]
        );
    }
}

// ===========================================================================
// 4. propensity_logit  —  INDEPENDENT HOST RE-DERIVATION (clamped logistic)
// ===========================================================================

#[test]
fn propensity_logit_matches_host_sigmoid() {
    // out[i] = clamp(sigmoid(b + Σ_j X[i,j]·w[j]), 0.05, 0.95),
    // sigmoid(z) = 1/(1+e^{-z}). The kernel forms e^{-z} as `ex2(-z·log2 e)`,
    // i.e. a correct base-e exponential (the base-2→base-e conversion factor
    // log2(e)=0x3FB8AA3B IS applied), so a base-e host oracle must match.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 96_usize;
    let d = 5_usize;

    let mut rng = LcgRng::new(0x0920_9517);
    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.6 - 0.3).collect();
    let bias = 0.1_f32;

    // Host reference. With |x|<1, |w|<0.3, d=5 the logit stays within ~[-1.6,1.6]
    // so sigmoid ∈ (0.17, 0.83): the [0.05,0.95] clamp is inactive and the test
    // exercises the raw logistic, but we apply the same clamp for exactness.
    let mut out_host = vec![0.0_f32; n];
    for i in 0..n {
        let mut z = bias;
        for j in 0..d {
            z += x[i * d + j] * w[j];
        }
        let s = 1.0_f32 / (1.0_f32 + (-z).exp());
        out_host[i] = s.clamp(0.05, 0.95);
    }

    let ptx = crate::ptx_kernels::propensity_logit_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "propensity_logit_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_b = DeviceBuffer::<f32>::from_host(&[bias]).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_w.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch propensity_logit_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // `ex2.approx.f32` carries ~2 ulp; 5e-4 relative covers it and still flags a
    // missing base-2→base-e conversion (which would skew the sigmoid by ~tens of
    // percent) by orders of magnitude.
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for i in 0..n {
        assert!(
            close(out_gpu[i], out_host[i], 5e-4, 1e-5),
            "propensity[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_host[i]
        );
    }
}

// ===========================================================================
// 5. ipw_estimator  —  INDEPENDENT HOST RE-DERIVATION (IPW ATE sum)
// ===========================================================================

#[test]
fn ipw_estimator_matches_host_sum() {
    // The kernel atomically accumulates the IPW ATE *sum* (it does NOT divide by
    // n): Σ_i (Y_i·T_i/π_i − Y_i·(1−T_i)/(1−π_i)) with π clamped to [0.05,0.95].
    // That is exactly `n · ipw_ate(...)`. The oracle is an independent host sum.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;

    let mut rng = LcgRng::new(0x01B0_7A7E);
    let y: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let t: Vec<f32> = (0..n)
        .map(|_| if rng.next_f32() < 0.5 { 1.0 } else { 0.0 })
        .collect();
    // Propensities in [0.2, 0.8]: the [0.05,0.95] clamp is inactive, matching the
    // host clamp exactly, and both denominators π and 1−π stay ≥ 0.2.
    let pi: Vec<f32> = (0..n).map(|_| 0.2 + 0.6 * rng.next_f32()).collect();

    // Host reference sum (same clamp as the kernel).
    let mut sum_host = 0.0_f32;
    for i in 0..n {
        let pic = pi[i].clamp(0.05, 0.95);
        sum_host += y[i] * t[i] / pic - y[i] * (1.0 - t[i]) / (1.0 - pic);
    }
    // Cross-check against the crate's averaged estimator: kernel sum == n·ATE.
    let ate_crate = crate::effect::ipw::ipw_ate(&y, &t, &pi).expect("ipw_ate");
    assert!(
        (sum_host - ate_crate * n as f32).abs() <= 1e-3 * sum_host.abs().max(1.0),
        "host IPW sum {sum_host} disagrees with n·ipw_ate {}",
        ate_crate * n as f32
    );

    let ptx = crate::ptx_kernels::ipw_estimator_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ipw_estimator_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_pi = DeviceBuffer::<f32>::from_host(&pi).expect("d_pi");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_y.as_device_ptr(),
                d_t.as_device_ptr(),
                d_pi.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch ipw_estimator_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Atomic adds commit in a nondeterministic order, so the GPU sum differs from
    // the sequential host sum only by accumulated FP32 rounding (~n·ulp); 1e-3
    // relative / 1e-2 absolute is a comfortable, still-meaningful bound.
    assert!(
        close(out_gpu[0], sum_host, 1e-3, 1e-2),
        "ipw sum mismatch: gpu={} host={}",
        out_gpu[0],
        sum_host
    );
}

// ===========================================================================
// 6. dml_residual  —  INDEPENDENT HOST RE-DERIVATION (elementwise residuals)
// ===========================================================================

#[test]
fn dml_residual_matches_host() {
    // ytilde[i] = Y[i] − g(X)[i],  ttilde[i] = T[i] − m(X)[i].  Pure elementwise
    // subtraction — the host oracle is exact (no FP reassociation).
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;

    let mut rng = LcgRng::new(0x0D_47_1E);
    let y: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let t: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let gy: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let mt: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    let ytilde_host: Vec<f32> = (0..n).map(|i| y[i] - gy[i]).collect();
    let ttilde_host: Vec<f32> = (0..n).map(|i| t[i] - mt[i]).collect();

    let ptx = crate::ptx_kernels::dml_residual_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dml_residual_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_gy = DeviceBuffer::<f32>::from_host(&gy).expect("d_gy");
    let d_mt = DeviceBuffer::<f32>::from_host(&mt).expect("d_mt");
    let d_yt = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_yt");
    let d_tt = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_tt");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_y.as_device_ptr(),
                d_t.as_device_ptr(),
                d_gy.as_device_ptr(),
                d_mt.as_device_ptr(),
                d_yt.as_device_ptr(),
                d_tt.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch dml_residual_kernel");
    stream.synchronize().expect("sync");

    let mut yt_gpu = vec![0.0_f32; n];
    let mut tt_gpu = vec![0.0_f32; n];
    d_yt.copy_to_host(&mut yt_gpu).expect("copy yt");
    d_tt.copy_to_host(&mut tt_gpu).expect("copy tt");

    for i in 0..n {
        assert_eq!(
            yt_gpu[i].to_bits(),
            ytilde_host[i].to_bits(),
            "dml ytilde[{i}] mismatch: gpu={} host={}",
            yt_gpu[i],
            ytilde_host[i]
        );
        assert_eq!(
            tt_gpu[i].to_bits(),
            ttilde_host[i].to_bits(),
            "dml ttilde[{i}] mismatch: gpu={} host={}",
            tt_gpu[i],
            ttilde_host[i]
        );
    }
}

// ===========================================================================
// 7. causal_split_score  —  INDEPENDENT HOST RE-DERIVATION (split criterion)
// ===========================================================================

#[test]
fn causal_split_score_matches_host() {
    // Per (feature f, threshold sample t) the kernel computes the heterogeneous
    // treatment-effect split score
    //   Δ = (τ_L − τ_R)² · n_L · n_R / n,
    // where a sample i goes LEFT iff feature[i,f] < feature[t,f] (strict), and
    //   τ_side = Σ Y·[T>0.5] / n_side_t1 − Σ Y·[T≤0.5] / n_side_t0
    // with each per-arm mean defaulting to 0 when its count is 0. The output is
    // row-major [d × n] indexed by (f·n + t).
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 24_usize; // samples (= number of candidate thresholds)
    let d = 3_usize; // features

    let mut rng = LcgRng::new(0x000C_5911);
    // Distinct random feature values ⇒ the strict `<` partition is unambiguous
    // (only the threshold sample itself can equal the threshold).
    let features: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();
    let y: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let t: Vec<f32> = (0..n)
        .map(|_| if rng.next_f32() < 0.5 { 1.0 } else { 0.0 })
        .collect();

    // Host re-derivation matching the kernel's exact branch / guard structure.
    let mut scores_host = vec![0.0_f32; d * n];
    for f in 0..d {
        for tidx in 0..n {
            let thr = features[tidx * d + f];
            let (mut sy_l1, mut sy_l0, mut sy_r1, mut sy_r0) = (0.0_f32, 0.0, 0.0, 0.0);
            let (mut nl1, mut nl0, mut nr1, mut nr0) = (0u32, 0u32, 0u32, 0u32);
            for i in 0..n {
                let fv = features[i * d + f];
                let yi = y[i];
                let treated = t[i] > 0.5;
                if fv < thr {
                    if treated {
                        sy_l1 += yi;
                        nl1 += 1;
                    } else {
                        sy_l0 += yi;
                        nl0 += 1;
                    }
                } else if treated {
                    sy_r1 += yi;
                    nr1 += 1;
                } else {
                    sy_r0 += yi;
                    nr0 += 1;
                }
            }
            let mean = |s: f32, c: u32| if c > 0 { s / c as f32 } else { 0.0 };
            let tau_l = mean(sy_l1, nl1) - mean(sy_l0, nl0);
            let tau_r = mean(sy_r1, nr1) - mean(sy_r0, nr0);
            let n_l = nl1 + nl0;
            let n_r = nr1 + nr0;
            let diff = tau_l - tau_r;
            let weight = (n_l as f32 * n_r as f32) / n as f32;
            scores_host[f * n + tidx] = diff * diff * weight;
        }
    }

    let ptx = crate::ptx_kernels::causal_split_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "causal_split_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_feat = DeviceBuffer::<f32>::from_host(&features).expect("d_feat");
    let d_scores = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; d * n]).expect("d_scores");

    // One thread per (feature, threshold) candidate; the kernel guards thread id
    // >= d*n, so any block size covering d*n works.
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d((d * n) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_y.as_device_ptr(),
                d_t.as_device_ptr(),
                d_feat.as_device_ptr(),
                d_scores.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch causal_split_score_kernel");
    stream.synchronize().expect("sync");

    let mut scores_gpu = vec![0.0_f32; d * n];
    d_scores.copy_to_host(&mut scores_gpu).expect("copy scores");

    // Same accumulation order (i = 0..n) and the same per-arm guards on the GPU
    // and host; the only divergence is `div.rn`/`mul` rounding. 1e-3 relative /
    // 1e-4 absolute is comfortable yet flags any wrong branch / index / count.
    let (rel, abs) = worst_diff(&scores_gpu, &scores_host);
    for k in 0..scores_gpu.len() {
        assert!(
            close(scores_gpu[k], scores_host[k], 1e-3, 1e-4),
            "causal_split[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            scores_gpu[k],
            scores_host[k]
        );
    }
}
