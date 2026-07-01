//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies the results
//! back, and asserts numerical equivalence to a CPU oracle. The launch ABI
//! mirrors the proven `oxicuda-snn` / `oxicuda-ot` canaries: device buffers are
//! passed as their `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust
//! scalar (`.param .u32` / `.param .f32`), in the kernel's declared param order.
//!
//! ## Oracle strength (honest accounting)
//!
//! Every kernel here is a self-contained numerical primitive whose result is
//! compared against an **independent hand-derivation in base-e arithmetic**.
//! The crate's public CPU functions all operate on a higher-level `Dataset`
//! rather than these element-wise primitives, so the oracle is recomputed from
//! the kernel's documented formula. These checks genuinely fail if ptxas
//! miscompiles, the PTX has a wrong constant / shift / index, or — critically —
//! an exponential is left in base-2 (`ex2` without the `log2(e)` scale), because
//! the host oracle uses true `exp`/`ln`.
//!
//! ## PTX bugs found and fixed (see `ptx_kernels.rs`)
//!
//! * `cox_risk_sum_kernel`, `cox_score_kernel`, `cox_info_kernel` — each applied
//!   `ex2.approx.f32` directly to `eta`, computing `2^eta` instead of the Cox
//!   risk `exp(eta) = e^eta` (`CoxFitResult::predict_risk` uses `.exp()`). The
//!   summed risk still looked plausible (a positive number) but was ~18-30 %
//!   wrong; only a base-e oracle catches it. Fixed by scaling `eta` by
//!   `log2(e) = 0f3FB8AA3B` before `ex2`.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

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

/// JIT-compile `ptx` and look up `entry`, returning a launchable kernel.
///
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug, so we
/// panic loudly rather than skip.
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
// 1. km_step  —  HAND-DERIVED ORACLE: S(t_i) = s_init · Π_{k≤i} (1 − d_k/n_k)
// ===========================================================================

#[test]
fn km_step_matches_host_product() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Deterministic, well-conditioned inputs: every cumulative survival stays in
    // [0.3, 1.0], comfortably away from zero, so the relative comparison is honest.
    let d: Vec<f32> = vec![1.0, 0.0, 2.0, 1.0, 0.0, 3.0];
    let n_at_risk: Vec<f32> = vec![10.0, 9.0, 9.0, 7.0, 6.0, 6.0];
    let n_steps = d.len();
    let s_init = 1.0_f32;

    // Host oracle: prefix product of (1 − d_k/n_k) up to and including index i.
    let mut s_host = vec![0.0_f32; n_steps];
    let mut acc = s_init;
    for i in 0..n_steps {
        acc *= 1.0 - d[i] / n_at_risk[i];
        s_host[i] = acc;
    }

    let ptx = crate::ptx_kernels::km_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "km_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_d = DeviceBuffer::<f32>::from_host(&d).expect("d_d");
    let d_n = DeviceBuffer::<f32>::from_host(&n_at_risk).expect("d_n");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_steps]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_steps as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_d.as_device_ptr(),
                d_n.as_device_ptr(),
                d_out.as_device_ptr(),
                n_steps as u32,
                s_init,
            ),
        )
        .expect("launch km_step_kernel");
    stream.synchronize().expect("sync");

    let mut s_gpu = vec![0.0_f32; n_steps];
    d_out.copy_to_host(&mut s_gpu).expect("copy s");

    // `div.approx.f32` (~2 ulp) chained over 6 factors stays well inside 1e-3.
    for i in 0..n_steps {
        assert!(
            close(s_gpu[i], s_host[i], 1e-3, 1e-5),
            "km_step S[{i}] mismatch: gpu={} host={}",
            s_gpu[i],
            s_host[i]
        );
    }
}

// ===========================================================================
// 2. cox_risk_sum  —  HAND-DERIVED ORACLE: Σ_{mask=1} exp(eta_j)  (base-e!)
// ===========================================================================

#[test]
fn cox_risk_sum_matches_host_exp() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let eta: Vec<f32> = vec![0.5, -0.3, 1.0, 0.0, 0.7];
    let mask: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let n = eta.len();

    // Host oracle in true base-e. With the base-2 bug this would be Σ 2^eta,
    // ~18 % smaller — far outside the 1e-3 tolerance.
    let mut expected = 0.0_f32;
    for j in 0..n {
        if mask[j] != 0.0 {
            expected += eta[j].exp();
        }
    }

    let ptx = crate::ptx_kernels::cox_risk_sum_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cox_risk_sum_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_eta = DeviceBuffer::<f32>::from_host(&eta).expect("d_eta");
    let d_mask = DeviceBuffer::<f32>::from_host(&mask).expect("d_mask");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_eta.as_device_ptr(),
                d_mask.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch cox_risk_sum_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert!(
        close(out_gpu[0], expected, 1e-3, 1e-5),
        "cox_risk_sum mismatch: gpu={} host(base-e)={} \
         (a base-2 ex2 would give ~{})",
        out_gpu[0],
        expected,
        {
            let mut b2 = 0.0_f32;
            for j in 0..n {
                if mask[j] != 0.0 {
                    b2 += 2.0_f32.powf(eta[j]);
                }
            }
            b2
        }
    );
}

// ===========================================================================
// 3. cox_score  —  HAND-DERIVED ORACLE: score[k] = Σ_{mask=1} exp(eta_j)·x[j,k]
// ===========================================================================

#[test]
fn cox_score_matches_host_exp() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let eta: Vec<f32> = vec![0.5, -0.3, 1.0, 0.0, 0.7];
    let mask: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let n = 5_usize;
    let p = 2_usize;
    // Row-major n×p covariate matrix.
    let x: Vec<f32> = vec![
        0.2, -0.5, // j0
        1.0, 0.3, // j1 (masked out)
        -0.4, 0.8, // j2
        0.6, -0.1, // j3
        0.9, 0.2, // j4 (masked out)
    ];

    // Host oracle (base-e).
    let mut score_host = vec![0.0_f32; p];
    for j in 0..n {
        if mask[j] != 0.0 {
            let w = eta[j].exp();
            for k in 0..p {
                score_host[k] += w * x[j * p + k];
            }
        }
    }

    let ptx = crate::ptx_kernels::cox_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cox_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_eta = DeviceBuffer::<f32>::from_host(&eta).expect("d_eta");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_mask = DeviceBuffer::<f32>::from_host(&mask).expect("d_mask");
    let d_score = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; p]).expect("d_score");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_eta.as_device_ptr(),
                d_x.as_device_ptr(),
                d_mask.as_device_ptr(),
                d_score.as_device_ptr(),
                n as u32,
                p as u32,
            ),
        )
        .expect("launch cox_score_kernel");
    stream.synchronize().expect("sync");

    let mut score_gpu = vec![0.0_f32; p];
    d_score.copy_to_host(&mut score_gpu).expect("copy score");

    for k in 0..p {
        assert!(
            close(score_gpu[k], score_host[k], 1e-3, 1e-5),
            "cox_score score[{k}] mismatch: gpu={} host={}",
            score_gpu[k],
            score_host[k]
        );
    }
}

// ===========================================================================
// 4. cox_info  —  HAND-DERIVED ORACLE: info[k,l] = Σ_{mask=1} exp(eta_j)·x_jk·x_jl
// ===========================================================================

#[test]
fn cox_info_matches_host_exp() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let eta: Vec<f32> = vec![0.5, -0.3, 1.0, 0.0, 0.7];
    let mask: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let n = 5_usize;
    let p = 2_usize;
    let x: Vec<f32> = vec![
        0.2, -0.5, // j0
        1.0, 0.3, // j1 (masked out)
        -0.4, 0.8, // j2
        0.6, -0.1, // j3
        0.9, 0.2, // j4 (masked out)
    ];

    // Host oracle (base-e), the p×p Fisher information accumulation.
    let mut info_host = vec![0.0_f32; p * p];
    for j in 0..n {
        if mask[j] != 0.0 {
            let w = eta[j].exp();
            for k in 0..p {
                for l in 0..p {
                    info_host[k * p + l] += w * x[j * p + k] * x[j * p + l];
                }
            }
        }
    }

    let ptx = crate::ptx_kernels::cox_info_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cox_info_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_eta = DeviceBuffer::<f32>::from_host(&eta).expect("d_eta");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_mask = DeviceBuffer::<f32>::from_host(&mask).expect("d_mask");
    let d_info = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; p * p]).expect("d_info");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_eta.as_device_ptr(),
                d_x.as_device_ptr(),
                d_mask.as_device_ptr(),
                d_info.as_device_ptr(),
                n as u32,
                p as u32,
            ),
        )
        .expect("launch cox_info_kernel");
    stream.synchronize().expect("sync");

    let mut info_gpu = vec![0.0_f32; p * p];
    d_info.copy_to_host(&mut info_gpu).expect("copy info");

    for idx in 0..p * p {
        assert!(
            close(info_gpu[idx], info_host[idx], 1e-3, 1e-5),
            "cox_info info[{idx}] mismatch: gpu={} host={}",
            info_gpu[idx],
            info_host[idx]
        );
    }
}

// ===========================================================================
// 5. logrank_oe  —  HAND-DERIVED ORACLE: oe[t] = d_g − n_g·(d_t/n_t)
// ===========================================================================

#[test]
fn logrank_oe_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let d_g: Vec<f32> = vec![2.0, 1.0, 0.0, 3.0];
    let n_g: Vec<f32> = vec![10.0, 8.0, 6.0, 5.0];
    let d_t: Vec<f32> = vec![5.0, 3.0, 1.0, 4.0];
    let n_t: Vec<f32> = vec![20.0, 16.0, 10.0, 9.0];
    let n = d_g.len();

    let mut oe_host = vec![0.0_f32; n];
    for t in 0..n {
        oe_host[t] = d_g[t] - n_g[t] * (d_t[t] / n_t[t]);
    }

    let ptx = crate::ptx_kernels::logrank_oe_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "logrank_oe_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_dg = DeviceBuffer::<f32>::from_host(&d_g).expect("d_dg");
    let d_ng = DeviceBuffer::<f32>::from_host(&n_g).expect("d_ng");
    let d_dt = DeviceBuffer::<f32>::from_host(&d_t).expect("d_dt");
    let d_nt = DeviceBuffer::<f32>::from_host(&n_t).expect("d_nt");
    let d_oe = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_oe");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_dg.as_device_ptr(),
                d_ng.as_device_ptr(),
                d_dt.as_device_ptr(),
                d_nt.as_device_ptr(),
                d_oe.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch logrank_oe_kernel");
    stream.synchronize().expect("sync");

    let mut oe_gpu = vec![0.0_f32; n];
    d_oe.copy_to_host(&mut oe_gpu).expect("copy oe");

    for t in 0..n {
        assert!(
            close(oe_gpu[t], oe_host[t], 1e-3, 1e-4),
            "logrank_oe oe[{t}] mismatch: gpu={} host={}",
            oe_gpu[t],
            oe_host[t]
        );
    }
}

// ===========================================================================
// 6. brier_score  —  HAND-DERIVED ORACLE: w·(1{t≤t*,δ>0} − s_pred)²
// ===========================================================================

#[test]
fn brier_score_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let t_star = 3.0_f32;
    let t: Vec<f32> = vec![1.0, 2.0, 4.0, 3.0, 5.0];
    let delta: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0, 1.0];
    let s_pred: Vec<f32> = vec![0.8, 0.6, 0.5, 0.7, 0.4];
    let w: Vec<f32> = vec![1.0, 1.2, 0.9, 1.1, 0.8];
    let n = t.len();

    let mut out_host = vec![0.0_f32; n];
    for i in 0..n {
        let indicator = if t[i] <= t_star && delta[i] > 0.0 {
            1.0_f32
        } else {
            0.0_f32
        };
        let diff = indicator - s_pred[i];
        out_host[i] = w[i] * diff * diff;
    }

    let ptx = crate::ptx_kernels::brier_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "brier_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_delta = DeviceBuffer::<f32>::from_host(&delta).expect("d_delta");
    let d_s = DeviceBuffer::<f32>::from_host(&s_pred).expect("d_s");
    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_t.as_device_ptr(),
                d_delta.as_device_ptr(),
                d_s.as_device_ptr(),
                d_w.as_device_ptr(),
                t_star,
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch brier_score_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for i in 0..n {
        assert!(
            close(out_gpu[i], out_host[i], 1e-5, 1e-6),
            "brier_score out[{i}] mismatch: gpu={} host={}",
            out_gpu[i],
            out_host[i]
        );
    }
}

// ===========================================================================
// 7. rmst_integrate  —  HAND-DERIVED ORACLE: max(0, min(t[i+1],τ)−t[i])·s[i]
// ===========================================================================

#[test]
fn rmst_integrate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let tau = 4.0_f32;
    let t: Vec<f32> = vec![0.0, 1.0, 2.5, 3.5, 6.0];
    let s: Vec<f32> = vec![1.0, 0.8, 0.6, 0.5, 0.3];
    let n = t.len();

    // The kernel writes only indices 0..n-1 (each rectangle spans [t_i, t_{i+1}]);
    // index n-1 is never written and stays at its zero-init value.
    let mut out_host = vec![0.0_f32; n];
    for i in 0..n - 1 {
        let upper = t[i + 1].min(tau);
        let width = (upper - t[i]).max(0.0);
        out_host[i] = s[i] * width;
    }

    let ptx = crate::ptx_kernels::rmst_integrate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rmst_integrate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_t = DeviceBuffer::<f32>::from_host(&t).expect("d_t");
    let d_s = DeviceBuffer::<f32>::from_host(&s).expect("d_s");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_t.as_device_ptr(),
                d_s.as_device_ptr(),
                tau,
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch rmst_integrate_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for i in 0..n {
        assert!(
            close(out_gpu[i], out_host[i], 1e-5, 1e-6),
            "rmst_integrate out[{i}] mismatch: gpu={} host={}",
            out_gpu[i],
            out_host[i]
        );
    }
    // The final rectangle index is never written by the kernel.
    assert_eq!(
        out_gpu[n - 1].to_bits(),
        0_u32,
        "rmst_integrate: trailing out[{}] should stay zero-init, got {}",
        n - 1,
        out_gpu[n - 1]
    );
}
