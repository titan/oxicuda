//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to an independent CPU re-derivation. The launch ABI mirrors the proven
//! `oxicuda-snn` / `oxicuda-ot` harnesses: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .u32` / `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Kernel inventory
//!
//! All five `*_ptx` functions in [`crate::ptx_kernels`] emit a 1-D grid-stride
//! loop over `n` and are launchable on sm_86 (none use Hopper-only `wgmma` /
//! TMA / `cp.async.bulk` / FP8). Every kernel here is validated with a real
//! CPU-vs-GPU numerical-equivalence assertion (no stubs):
//!
//! * `td_error`             — δ = r + γ·next_v·(1−done) − v
//! * `normalize_advantages` — adv = (adv − mean) / (std + eps)
//! * `ppo_ratio`            — ratio = exp(lpₙ − lpₒ); obj = min(ratio·A, clip·A)
//! * `sac_target`           — y = r + γ·(1−done)·(min_q − α·log_π)
//! * `per_is_weight`        — w = (1 / (N·P))^β
//!
//! ## Oracle strength tier
//!
//! Each kernel's arithmetic is fused into a larger CPU routine on the host
//! (`sac_critic_loss`, `ppo_loss`, the PER buffer, …) rather than exposed as a
//! standalone "compute exactly the kernel output" function, so the oracle is an
//! independent Rust re-implementation of the kernel's *documented* per-element
//! math, matching the formula used by the corresponding crate routine. These
//! oracles still genuinely fail if ptxas miscompiles, if the PTX has a wrong
//! constant / shift / index, or — critically for `ppo_ratio` and
//! `per_is_weight` — if the `ex2.approx` / `lg2.approx` base-2 ↔ base-e
//! conversion is wrong, because the host code uses libm `exp` / `ln` / `powf`
//! and is independent of the JIT-compiled PTX.
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug in
/// `ptx_kernels.rs`, surfaced loudly rather than skipped.
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
// 1. td_error  —  INDEPENDENT HOST RE-DERIVATION (δ = r + γ·next_v·(1−done) − v)
// ===========================================================================

#[test]
fn td_error_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let gamma = 0.99_f32;
    let mut rng = LcgRng::new(0x7DE7_7001);

    // Moderate magnitudes keep every δ comfortably away from zero so the
    // relative comparison is never evaluated at a knife-edge cancellation.
    let reward: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let next_v: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let v: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let done: Vec<f32> = (0..n)
        .map(|_| if rng.next_f32() < 0.5 { 1.0 } else { 0.0 })
        .collect();

    // Host re-derivation of the documented per-element formula.
    let mut delta_host = vec![0.0_f32; n];
    for i in 0..n {
        delta_host[i] = reward[i] + gamma * next_v[i] * (1.0 - done[i]) - v[i];
    }

    let ptx = crate::ptx_kernels::td_error_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "td_error");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_reward = DeviceBuffer::<f32>::from_host(&reward).expect("d_reward");
    let d_next_v = DeviceBuffer::<f32>::from_host(&next_v).expect("d_next_v");
    let d_done = DeviceBuffer::<f32>::from_host(&done).expect("d_done");
    let d_v = DeviceBuffer::<f32>::from_host(&v).expect("d_v");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_reward.as_device_ptr(),
                d_next_v.as_device_ptr(),
                d_done.as_device_ptr(),
                d_v.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                gamma,
            ),
        )
        .expect("launch td_error");
    stream.synchronize().expect("sync");

    let mut delta_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut delta_gpu).expect("copy out");

    // GPU fuses γ·nv with `fma.rn`; host uses two-rounding mul/add. ~1 ulp.
    let (rel, abs) = worst_diff(&delta_gpu, &delta_host);
    for i in 0..n {
        assert!(
            close(delta_gpu[i], delta_host[i], 1e-4, 1e-5),
            "td_error δ[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            delta_gpu[i],
            delta_host[i]
        );
    }
}

// ===========================================================================
// 2. normalize_advantages  —  INDEPENDENT HOST RE-DERIVATION ((a−μ)/(σ+ε))
// ===========================================================================

#[test]
fn normalize_advantages_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mean = 0.37_f32;
    // The kernel receives `std + eps` already summed on the host.
    let std_eps = 1.25_f32;
    let mut rng = LcgRng::new(0x00AD_4055);

    let adv: Vec<f32> = (0..n).map(|_| rng.next_f32() * 6.0 - 3.0).collect();

    // Host re-derivation: in-place normalisation, exactly as the kernel does it.
    let mut adv_host = adv.clone();
    for x in &mut adv_host {
        *x = (*x - mean) / std_eps;
    }

    let ptx = crate::ptx_kernels::normalize_advantages_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "normalize_advantages");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_adv = DeviceBuffer::<f32>::from_host(&adv).expect("d_adv");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_adv.as_device_ptr(), n as u32, mean, std_eps),
        )
        .expect("launch normalize_advantages");
    stream.synchronize().expect("sync");

    let mut adv_gpu = vec![0.0_f32; n];
    d_adv.copy_to_host(&mut adv_gpu).expect("copy adv");

    // `sub.rn` + `div.rn` are correctly rounded; host uses the same ops. <1 ulp.
    let (rel, abs) = worst_diff(&adv_gpu, &adv_host);
    for i in 0..n {
        assert!(
            close(adv_gpu[i], adv_host[i], 1e-5, 1e-6),
            "normalize_advantages adv[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            adv_gpu[i],
            adv_host[i]
        );
    }
}

// ===========================================================================
// 3. ppo_ratio  —  INDEPENDENT HOST RE-DERIVATION (matches loss::ppo_loss)
//    Exercises the base-2 ↔ base-e `ex2.approx` exp conversion.
// ===========================================================================

#[test]
fn ppo_ratio_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let clip_eps = 0.2_f32;
    let lo = 1.0 - clip_eps;
    let hi = 1.0 + clip_eps;
    let mut rng = LcgRng::new(0x0099_0001);

    // log-prob differences in [-0.7, 0.7] ⇒ ratio = exp(d) ∈ [0.50, 2.01], well
    // inside `ex2.approx`'s accurate domain. We keep every ratio at least 0.02
    // away from the clip bounds {0.8, 1.2} so the `min(ratio·A, clip·A)` branch
    // selection is identical on host and device (the ~2-ulp `ex2` error can
    // never flip which surrogate is smaller).
    let mut lp_new = vec![0.0_f32; n];
    let lp_old = vec![0.0_f32; n];
    let mut adv = vec![0.0_f32; n];
    for i in 0..n {
        let mut d = rng.next_f32() * 1.4 - 0.7;
        let mut ratio = d.exp();
        while (ratio - lo).abs() < 0.02 || (ratio - hi).abs() < 0.02 {
            d = rng.next_f32() * 1.4 - 0.7;
            ratio = d.exp();
        }
        lp_new[i] = d; // lp_old = 0 ⇒ ratio = exp(lp_new)
        adv[i] = rng.next_f32() * 4.0 - 2.0; // spans negative & positive advantage
    }

    // Host re-derivation matching `loss::ppo_loss`:
    //   ratio = exp(lpₙ − lpₒ)
    //   obj   = min(ratio·A, clamp(ratio, 1−ε, 1+ε)·A)
    let mut ratio_host = vec![0.0_f32; n];
    let mut obj_host = vec![0.0_f32; n];
    for i in 0..n {
        let ratio = (lp_new[i] - lp_old[i]).exp();
        let clip = ratio.clamp(lo, hi);
        ratio_host[i] = ratio;
        obj_host[i] = (ratio * adv[i]).min(clip * adv[i]);
    }

    let ptx = crate::ptx_kernels::ppo_ratio_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ppo_ratio");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_lp_new = DeviceBuffer::<f32>::from_host(&lp_new).expect("d_lp_new");
    let d_lp_old = DeviceBuffer::<f32>::from_host(&lp_old).expect("d_lp_old");
    let d_adv = DeviceBuffer::<f32>::from_host(&adv).expect("d_adv");
    let d_ratio = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_ratio");
    let d_obj = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_obj");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_lp_new.as_device_ptr(),
                d_lp_old.as_device_ptr(),
                d_adv.as_device_ptr(),
                d_ratio.as_device_ptr(),
                d_obj.as_device_ptr(),
                n as u32,
                clip_eps,
            ),
        )
        .expect("launch ppo_ratio");
    stream.synchronize().expect("sync");

    let mut ratio_gpu = vec![0.0_f32; n];
    let mut obj_gpu = vec![0.0_f32; n];
    d_ratio.copy_to_host(&mut ratio_gpu).expect("copy ratio");
    d_obj.copy_to_host(&mut obj_gpu).expect("copy obj");

    // `ex2.approx.f32` carries ~2 ulp; a missing `* log2(e)` base conversion
    // would yield 2^d instead of e^d — a 20–60 % error this 5e-4 bound flags.
    let (rrel, rabs) = worst_diff(&ratio_gpu, &ratio_host);
    for i in 0..n {
        assert!(
            close(ratio_gpu[i], ratio_host[i], 5e-4, 1e-6),
            "ppo ratio[{i}] mismatch: gpu={} host={} (worst rel={rrel:e} abs={rabs:e})",
            ratio_gpu[i],
            ratio_host[i]
        );
    }
    let (orel, oabs) = worst_diff(&obj_gpu, &obj_host);
    for i in 0..n {
        assert!(
            close(obj_gpu[i], obj_host[i], 5e-4, 1e-5),
            "ppo obj[{i}] mismatch: gpu={} host={} ratio={} adv={} (worst rel={orel:e} abs={oabs:e})",
            obj_gpu[i],
            obj_host[i],
            ratio_host[i],
            adv[i]
        );
    }
}

// ===========================================================================
// 4. sac_target  —  INDEPENDENT HOST RE-DERIVATION (matches loss::sac_critic_loss)
// ===========================================================================

#[test]
fn sac_target_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let gamma = 0.99_f32;
    let alpha = 0.2_f32;
    let mut rng = LcgRng::new(0x05AC_7A19);

    let reward: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let min_q: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    // Entropy term log_π is typically negative; sample in [-2, 0).
    let log_pi: Vec<f32> = (0..n).map(|_| -2.0 * rng.next_f32()).collect();
    let done: Vec<f32> = (0..n)
        .map(|_| if rng.next_f32() < 0.5 { 1.0 } else { 0.0 })
        .collect();

    // Host re-derivation, matching the target inside `sac_critic_loss`:
    //   soft_value = min_q − α·log_π
    //   y          = r + γ·(1 − done)·soft_value
    let mut y_host = vec![0.0_f32; n];
    for i in 0..n {
        let soft_value = min_q[i] - alpha * log_pi[i];
        y_host[i] = reward[i] + gamma * (1.0 - done[i]) * soft_value;
    }

    let ptx = crate::ptx_kernels::sac_target_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sac_target");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_reward = DeviceBuffer::<f32>::from_host(&reward).expect("d_reward");
    let d_done = DeviceBuffer::<f32>::from_host(&done).expect("d_done");
    let d_min_q = DeviceBuffer::<f32>::from_host(&min_q).expect("d_min_q");
    let d_log_pi = DeviceBuffer::<f32>::from_host(&log_pi).expect("d_log_pi");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_reward.as_device_ptr(),
                d_done.as_device_ptr(),
                d_min_q.as_device_ptr(),
                d_log_pi.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                gamma,
                alpha,
            ),
        )
        .expect("launch sac_target");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut y_gpu).expect("copy out");

    // Chain of `fma.rn` / `mul.rn` vs host two-rounding ops: a few ulp.
    let (rel, abs) = worst_diff(&y_gpu, &y_host);
    for i in 0..n {
        assert!(
            close(y_gpu[i], y_host[i], 1e-4, 1e-5),
            "sac_target y[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[i],
            y_host[i]
        );
    }
}

// ===========================================================================
// 5. per_is_weight  —  INDEPENDENT HOST RE-DERIVATION (w = (1/(N·P))^β)
//    Exercises the base-2 ↔ base-e `lg2.approx` / `ex2.approx` conversion.
// ===========================================================================

#[test]
fn per_is_weight_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let n_f = n as f32;
    let beta = 0.5_f32;
    let mut rng = LcgRng::new(0x009E_1607);

    // Sampling probabilities in [0.01, 0.5]: bounded away from 0 (no overflow in
    // 1/(N·P)) and ≤ 0.5, so N·P ≥ 2.56 and the weights stay in a moderate
    // range where `lg2.approx` / `ex2.approx` are accurate.
    let probs: Vec<f32> = (0..n).map(|_| 0.01 + 0.49 * rng.next_f32()).collect();

    // Host re-derivation of the documented IS-weight formula (unnormalised):
    //   w = (1 / (N·P))^β
    // computed with libm `powf`, independent of the kernel's lg2/ex2 path. A
    // missing base conversion in either log or exp would change the result by a
    // large factor, which the 2e-3 bound flags decisively.
    let mut w_host = vec![0.0_f32; n];
    for i in 0..n {
        w_host[i] = (1.0_f32 / (n_f * probs[i])).powf(beta);
    }

    let ptx = crate::ptx_kernels::per_is_weight_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "per_is_weight");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_probs = DeviceBuffer::<f32>::from_host(&probs).expect("d_probs");
    let d_w = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_w");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_probs.as_device_ptr(),
                d_w.as_device_ptr(),
                n as u32,
                n_f,
                beta,
            ),
        )
        .expect("launch per_is_weight");
    stream.synchronize().expect("sync");

    let mut w_gpu = vec![0.0_f32; n];
    d_w.copy_to_host(&mut w_gpu).expect("copy w");

    // `lg2.approx` (~2–3 ulp) + `ex2.approx` (~2 ulp) compounded ⇒ ~1e-3
    // relative; 2e-3 is comfortable yet flags any base-2/base-e error.
    let (rel, abs) = worst_diff(&w_gpu, &w_host);
    for i in 0..n {
        assert!(
            close(w_gpu[i], w_host[i], 2e-3, 1e-6),
            "per_is_weight w[{i}] mismatch: gpu={} host={} prob={} (worst rel={rel:e} abs={abs:e})",
            w_gpu[i],
            w_host[i],
            probs[i]
        );
    }
}
