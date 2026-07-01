//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to the crate's CPU
//! reference. The launch ABI mirrors the working `oxicuda-sparse` ILU(0) and
//! `oxicuda-driver` `vector_add` paths: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars are passed as the matching Rust
//! scalar (`.param .u32` / `.param .f32`), in declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared bit-for-bit / within FP32 tol to a
//!   `pub` CPU function the kernel is meant to mirror:
//!   `lif_step`, `surrogate_grad` (all 5 modes), `stdp_update`.
//! * **Independent host re-derivation** — the kernel has no single dedicated
//!   crate function (the op is fused into a larger routine on the CPU), so the
//!   oracle is an independent Rust re-implementation of the kernel's *documented*
//!   arithmetic: `bptt_accum` (outer-product accumulate), `rate_encode` /
//!   `poisson_sample` (the kernels' inline counter-based LCG). These still
//!   genuinely fail if ptxas miscompiles or the PTX has a wrong constant / shift
//!   / index, because the host code is independent of the JIT-compiled PTX.
//! * **Load + structural only** — `spike_conv` does *not* have a clean crate
//!   oracle (its input-dimension inference and neuron model diverge from
//!   `SpikingConv2d::forward_step`; see that test). It is validated for LOAD on
//!   the device plus structural well-formedness of the output, and the
//!   divergence is reported rather than papered over.
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
// 1. lif_step  —  CRATE ORACLE (crate::neuron::lif::lif_step), both reset modes
// ===========================================================================

/// Build deterministic LIF inputs whose post-integration membrane `v_new` is
/// guaranteed to sit at least `0.2` away from `v_th` for every neuron, so the
/// single-rounding `fma` on the GPU can never flip a spike decision relative to
/// the CPU's two-rounding `beta*v + I`. This keeps the spike assertion strictly
/// bit-exact while the membrane is compared within FP32 tolerance.
fn lif_inputs(n: usize, beta: f32, v_th: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = LcgRng::new(seed);
    let mut v0 = vec![0.0_f32; n];
    let mut current = vec![0.0_f32; n];
    for k in 0..n {
        let v = rng.next_f32() * 2.0 - 1.0; // [-1, 1)
        // Target membrane at least 0.2 from threshold, spanning sub/supra.
        let target = if rng.next_f32() < 0.5 {
            v_th - 0.2 - rng.next_f32() // (v_th - 1.2, v_th - 0.2]
        } else {
            v_th + 0.2 + rng.next_f32() // [v_th + 0.2, v_th + 1.2)
        };
        v0[k] = v;
        current[k] = target - beta * v;
    }
    (v0, current)
}

fn run_lif_case(fx: &GpuFixture, reset: crate::neuron::lif::ResetMode, reset_mode_u32: u32) {
    use crate::neuron::lif::{LifConfig, LifState, beta, lif_step};

    let n = 256_usize;
    let cfg = LifConfig {
        tau_m: 20.0,
        v_th: 1.0,
        v_rest: -0.1,
        dt: 1.0,
        reset,
    };
    let b = beta(&cfg);
    let (v0, current) = lif_inputs(n, b, cfg.v_th, 0xC0FFEE);

    // ---- CPU reference ----
    let mut state = LifState { v: v0.clone() };
    let mut spikes_cpu = vec![0.0_f32; n];
    lif_step(&mut state, &current, &cfg, &mut spikes_cpu).expect("cpu lif_step");
    let v_cpu = state.v;

    // Precondition: every neuron is comfortably away from threshold so the
    // spike comparison is an honest bit-exact check (not a knife-edge).
    for k in 0..n {
        let v_new = b * v0[k] + current[k];
        assert!(
            (v_new - cfg.v_th).abs() > 0.1,
            "test setup error: neuron {k} too close to threshold ({v_new})"
        );
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::lif_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lif_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_v = DeviceBuffer::<f32>::from_host(&v0).expect("d_v");
    let d_i = DeviceBuffer::<f32>::from_host(&current).expect("d_i");
    let d_s = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_s");

    let block = 256_u32;
    let grid = grid_1d(n as u32, block);
    let params = LaunchParams::new(grid, block);

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_v.as_device_ptr(),
                d_i.as_device_ptr(),
                d_s.as_device_ptr(),
                n as u32,
                b,
                cfg.v_th,
                cfg.v_rest,
                reset_mode_u32,
            ),
        )
        .expect("launch lif_step_kernel");
    stream.synchronize().expect("sync");

    let mut v_gpu = vec![0.0_f32; n];
    let mut spikes_gpu = vec![0.0_f32; n];
    d_v.copy_to_host(&mut v_gpu).expect("copy v");
    d_s.copy_to_host(&mut spikes_gpu).expect("copy s");

    // Spikes must be bit-exact.
    for k in 0..n {
        assert_eq!(
            spikes_gpu[k].to_bits(),
            spikes_cpu[k].to_bits(),
            "spike mismatch at {k}: gpu={} cpu={}",
            spikes_gpu[k],
            spikes_cpu[k]
        );
    }
    // Membrane within 1e-4 relative (FP32): the only source of divergence is the
    // GPU's single-rounding `fma.rn(beta, v, I)` vs the CPU's two-rounding
    // `beta*v + I`, bounded by ~1 ulp (~1.2e-7 relative); 1e-4 is generous.
    let (rel, abs) = worst_diff(&v_gpu, &v_cpu);
    for k in 0..n {
        assert!(
            close(v_gpu[k], v_cpu[k], 1e-4, 1e-6),
            "membrane mismatch at {k}: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            v_gpu[k],
            v_cpu[k]
        );
    }
}

#[test]
fn lif_step_hard_reset_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_lif_case(&fx, crate::neuron::lif::ResetMode::Hard, 0);
}

#[test]
fn lif_step_soft_reset_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_lif_case(&fx, crate::neuron::lif::ResetMode::Soft, 1);
}

// ===========================================================================
// 2. surrogate_grad  —  CRATE ORACLE (crate::surrogate::*), all 5 modes
// ===========================================================================

/// CPU reference for surrogate mode `m`, matching the PTX branch table:
/// 0=sigmoid, 1=atan, 2=triangle, 3=super_spike, 4=fast_sigmoid.
fn surrogate_cpu(mode: u32, v: &[f32], v_th: f32, alpha: f32) -> Vec<f32> {
    use crate::surrogate::{
        atan::atan_grad, fast_sigmoid::fast_sigmoid_grad, sigmoid::sigmoid_grad,
        super_spike::super_spike_grad, triangle::triangle_grad,
    };
    let mut g = vec![0.0_f32; v.len()];
    match mode {
        0 => sigmoid_grad(v, v_th, alpha, &mut g),
        1 => atan_grad(v, v_th, alpha, &mut g),
        2 => triangle_grad(v, v_th, alpha, &mut g),
        3 => super_spike_grad(v, v_th, alpha, &mut g),
        4 => fast_sigmoid_grad(v, v_th, alpha, &mut g),
        _ => unreachable!("invalid surrogate mode"),
    }
    .expect("cpu surrogate");
    g
}

fn run_surrogate_case(fx: &GpuFixture, mode: u32) {
    let n = 256_usize;
    let v_th = 0.3_f32;
    let alpha = 2.0_f32;

    // v spread across [v_th-3, v_th+3]; moderate range keeps the sigmoid's
    // `ex2.approx` exponential well inside its accurate domain.
    let mut rng = LcgRng::new(0x5EED ^ u64::from(mode));
    let v: Vec<f32> = (0..n).map(|_| v_th - 3.0 + 6.0 * rng.next_f32()).collect();

    let g_cpu = surrogate_cpu(mode, &v, v_th, alpha);

    let ptx = crate::ptx_kernels::surrogate_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "surrogate_grad_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_v = DeviceBuffer::<f32>::from_host(&v).expect("d_v");
    let d_g = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_g");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_v.as_device_ptr(),
                d_g.as_device_ptr(),
                n as u32,
                v_th,
                alpha,
                mode,
            ),
        )
        .expect("launch surrogate_grad_kernel");
    stream.synchronize().expect("sync");

    let mut g_gpu = vec![0.0_f32; n];
    d_g.copy_to_host(&mut g_gpu).expect("copy g");

    let (rel, abs) = worst_diff(&g_gpu, &g_cpu);
    // Tolerance: sigmoid uses `ex2.approx.f32` (~2 ulp); the others use
    // correctly-rounded `div.rn`. 5e-4 relative comfortably covers the
    // approximation yet still catches a gross formula error (e.g. an off-by-pi^2
    // scale) by orders of magnitude.
    for k in 0..n {
        assert!(
            close(g_gpu[k], g_cpu[k], 5e-4, 1e-6),
            "surrogate mode {mode} mismatch at {k}: gpu={} cpu={} v={} \
             (worst rel={rel:e} abs={abs:e})",
            g_gpu[k],
            g_cpu[k],
            v[k]
        );
    }
}

#[test]
fn surrogate_grad_sigmoid_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_surrogate_case(&fx, 0);
}

#[test]
fn surrogate_grad_atan_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_surrogate_case(&fx, 1);
}

#[test]
fn surrogate_grad_triangle_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_surrogate_case(&fx, 2);
}

#[test]
fn surrogate_grad_super_spike_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_surrogate_case(&fx, 3);
}

#[test]
fn surrogate_grad_fast_sigmoid_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_surrogate_case(&fx, 4);
}

// ===========================================================================
// 3. stdp_update  —  CRATE ORACLE (crate::plasticity::stdp::pair_delta)
// ===========================================================================

#[test]
fn stdp_update_matches_cpu() {
    use crate::plasticity::stdp::{StdpConfig, pair_delta};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_pre = 8_usize;
    let n_post = 6_usize;
    let mut rng = LcgRng::new(0x57DB);

    let x_pre: Vec<f32> = (0..n_pre).map(|_| rng.next_f32()).collect();
    let y_post: Vec<f32> = (0..n_post).map(|_| rng.next_f32()).collect();
    let pre_spike: Vec<f32> = (0..n_pre)
        .map(|_| if rng.next_f32() < 0.5 { 1.0 } else { 0.0 })
        .collect();
    let post_spike: Vec<f32> = (0..n_post)
        .map(|_| if rng.next_f32() < 0.5 { 1.0 } else { 0.0 })
        .collect();
    let w0: Vec<f32> = (0..n_pre * n_post).map(|_| rng.next_f32()).collect();

    let cfg = StdpConfig {
        a_plus: 0.01,
        a_minus: 0.012,
        ..StdpConfig::default()
    };

    // CPU reference: w += pair_delta(...). The kernel computes exactly this
    // per (i,j) cell (no clamp / no trace decay — those live in stdp_step).
    let dw = pair_delta(
        &pre_spike,
        &post_spike,
        &x_pre,
        &y_post,
        n_pre,
        n_post,
        &cfg,
    );
    let w_expected: Vec<f32> = w0.iter().zip(&dw).map(|(w, d)| w + d).collect();

    let ptx = crate::ptx_kernels::stdp_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "stdp_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&w0).expect("d_w");
    let d_x = DeviceBuffer::<f32>::from_host(&x_pre).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&y_post).expect("d_y");
    let d_pre = DeviceBuffer::<f32>::from_host(&pre_spike).expect("d_pre");
    let d_post = DeviceBuffer::<f32>::from_host(&post_spike).expect("d_post");

    // One thread per (i,j): grid = (n_pre, n_post), block = 1.
    let params = LaunchParams::new((n_pre as u32, n_post as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                d_pre.as_device_ptr(),
                d_post.as_device_ptr(),
                n_pre as u32,
                n_post as u32,
                cfg.a_plus,
                cfg.a_minus,
            ),
        )
        .expect("launch stdp_update_kernel");
    stream.synchronize().expect("sync");

    let mut w_gpu = vec![0.0_f32; n_pre * n_post];
    d_w.copy_to_host(&mut w_gpu).expect("copy w");

    let (rel, abs) = worst_diff(&w_gpu, &w_expected);
    for k in 0..w_gpu.len() {
        assert!(
            close(w_gpu[k], w_expected[k], 1e-4, 1e-7),
            "stdp w[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            w_gpu[k],
            w_expected[k]
        );
    }
}

// ===========================================================================
// 4. bptt_accum  —  INDEPENDENT HOST RE-DERIVATION (outer-product accumulate)
// ===========================================================================

#[test]
fn bptt_accum_matches_host_outer_product() {
    // The CPU `bptt_unroll` fuses `dW += dv * x^T` inside its time loop, so
    // there is no standalone crate function. The oracle is an independent host
    // outer-product accumulate — the kernel's documented `dW[i,j] += dv[i]*I[j]`.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let out_dim = 8_usize;
    let in_dim = 6_usize;
    let mut rng = LcgRng::new(0xB977);

    let dv: Vec<f32> = (0..out_dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let input: Vec<f32> = (0..in_dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let dw0: Vec<f32> = (0..out_dim * in_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    let mut expected = dw0.clone();
    for i in 0..out_dim {
        for j in 0..in_dim {
            expected[i * in_dim + j] += dv[i] * input[j];
        }
    }

    let ptx = crate::ptx_kernels::bptt_accum_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bptt_accum_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_dv = DeviceBuffer::<f32>::from_host(&dv).expect("d_dv");
    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_dw = DeviceBuffer::<f32>::from_host(&dw0).expect("d_dw");

    let params = LaunchParams::new((out_dim as u32, in_dim as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_dv.as_device_ptr(),
                d_in.as_device_ptr(),
                d_dw.as_device_ptr(),
                out_dim as u32,
                in_dim as u32,
            ),
        )
        .expect("launch bptt_accum_kernel");
    stream.synchronize().expect("sync");

    let mut dw_gpu = vec![0.0_f32; out_dim * in_dim];
    d_dw.copy_to_host(&mut dw_gpu).expect("copy dw");

    let (rel, abs) = worst_diff(&dw_gpu, &expected);
    for k in 0..dw_gpu.len() {
        // GPU `fma.rn(dv, in, dw)` is single-rounding vs host two-rounding add;
        // ~1 ulp divergence, well within 1e-5 relative.
        assert!(
            close(dw_gpu[k], expected[k], 1e-5, 1e-6),
            "bptt dW[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            dw_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 5. rate_encode  —  INDEPENDENT HOST RE-DERIVATION of the inline counter LCG
// ===========================================================================

/// Host re-derivation of the `rate_encode_kernel`'s inline counter-based LCG.
///
/// This is an *independent* Rust implementation of the exact integer/float
/// pipeline the PTX describes (not the crate's `encoding::rate::rate_encode`,
/// which draws from a sequential `LcgRng` with a different schedule). The
/// 24-bit mantissa fits an integer `< 2^24` exactly and the `* 2^-24` is an
/// exact power-of-two scale, so the uniform `u` is bit-identical to the GPU's
/// `cvt.rn.f32.u32` + `mul.f32`; the Bernoulli decision is therefore bit-exact.
fn rate_encode_uniform(t: u32, i: u32, seed: u64) -> f32 {
    const M: u64 = 6_364_136_223_846_793_005;
    const A: u64 = 1_442_695_040_888_963_407;
    let mut state = (t as u64)
        .wrapping_mul(M)
        .wrapping_add((i as u64).wrapping_mul(A));
    state ^= seed;
    state = state.wrapping_mul(M).wrapping_add(A);
    let mix = (state >> 33) ^ state;
    let r = (mix as u32) >> 8;
    (r as f32) * (1.0_f32 / 16_777_216.0_f32)
}

#[test]
fn rate_encode_matches_host_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let t_steps = 48_usize;
    let seed = 0x1234_5678_9ABC_DEF0_u64;

    let mut vrng = LcgRng::new(0x7A7E);
    // Probabilities in [0.05, 0.95]; random f32 are never exactly k/2^24, so the
    // `u < value` comparison is unambiguous.
    let values: Vec<f32> = (0..n).map(|_| 0.05 + 0.90 * vrng.next_f32()).collect();

    // Host reference (bit-exact spikes).
    let mut out_host = vec![0.0_f32; t_steps * n];
    for t in 0..t_steps {
        for i in 0..n {
            let u = rate_encode_uniform(t as u32, i as u32, seed);
            out_host[t * n + i] = if u < values[i] { 1.0 } else { 0.0 };
        }
    }

    let ptx = crate::ptx_kernels::rate_encode_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rate_encode_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_val = DeviceBuffer::<f32>::from_host(&values).expect("d_val");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; t_steps * n]).expect("d_out");

    // grid = (t_steps, n), block = 1 (kernel uses ctaid.x = t, ctaid.y = i).
    let params = LaunchParams::new((t_steps as u32, n as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_val.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                t_steps as u32,
                seed,
            ),
        )
        .expect("launch rate_encode_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; t_steps * n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Structural: every entry is a binary spike.
    for (k, &s) in out_gpu.iter().enumerate() {
        assert!(
            s == 0.0 || s == 1.0,
            "rate_encode out[{k}] = {s} not binary"
        );
    }
    // Bit-exact spike pattern vs the independent host LCG.
    let mut mismatches = 0usize;
    for k in 0..out_gpu.len() {
        if out_gpu[k].to_bits() != out_host[k].to_bits() {
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches,
        0,
        "rate_encode: {mismatches}/{} spikes differ from host LCG",
        out_gpu.len()
    );
}

// ===========================================================================
// 6. poisson_sample  —  INDEPENDENT HOST RE-DERIVATION (stateful per-neuron LCG)
// ===========================================================================

#[test]
fn poisson_sample_matches_host_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    const M: u64 = 6_364_136_223_846_793_005;
    const A: u64 = 1_442_695_040_888_963_407;

    let n = 256_usize;
    let dt = 0.5_f32;
    let mut rng = LcgRng::new(0x9015_0014);

    let rate: Vec<f32> = (0..n).map(|_| 0.90 * rng.next_f32()).collect();
    let state0: Vec<u64> = (0..n).map(|_| rng.next_u64()).collect();

    // Host reference: advance state once, derive u, threshold against rate*dt.
    let mut out_host = vec![0.0_f32; n];
    let mut state_host = vec![0_u64; n];
    for i in 0..n {
        let s = state0[i].wrapping_mul(M).wrapping_add(A);
        state_host[i] = s;
        let mix = (s >> 33) ^ s;
        let r = (mix as u32) >> 8;
        let u = (r as f32) * (1.0_f32 / 16_777_216.0_f32);
        let p = rate[i] * dt;
        out_host[i] = if u < p { 1.0 } else { 0.0 };
    }

    let ptx = crate::ptx_kernels::poisson_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "poisson_sample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_rate = DeviceBuffer::<f32>::from_host(&rate).expect("d_rate");
    let d_state = DeviceBuffer::<u64>::from_host(&state0).expect("d_state");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_rate.as_device_ptr(),
                d_state.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                dt,
            ),
        )
        .expect("launch poisson_sample_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    let mut state_gpu = vec![0_u64; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");
    d_state.copy_to_host(&mut state_gpu).expect("copy state");

    // The advanced RNG state written back must match exactly (u64 integer math).
    for i in 0..n {
        assert_eq!(
            state_gpu[i], state_host[i],
            "poisson state[{i}] mismatch: gpu={} host={}",
            state_gpu[i], state_host[i]
        );
    }
    // Binary + bit-exact spike decision.
    for (i, &s) in out_gpu.iter().enumerate() {
        assert!(s == 0.0 || s == 1.0, "poisson out[{i}] = {s} not binary");
        assert_eq!(
            s.to_bits(),
            out_host[i].to_bits(),
            "poisson spike[{i}] mismatch: gpu={s} host={}",
            out_host[i]
        );
    }
}

// ===========================================================================
// 7. spike_conv  —  LOAD + STRUCTURAL ONLY (no clean crate oracle; see below)
// ===========================================================================

#[test]
fn spike_conv_loads_and_runs_wellformed() {
    // HONEST SCOPE: `spike_conv_kernel` does NOT cleanly mirror the crate's
    // `layer::spiking_conv::SpikingConv2d::forward_step`, so this is a LOAD +
    // structural test, not a numerical-equivalence test. Two divergences make a
    // crate-equivalence oracle impossible without rewriting the kernel:
    //
    //   (a) Input-dimension inference. forward_step uses no-padding stride-1
    //       conv: input height IH = OH + KH - 1. The kernel instead indexes the
    //       input with IH = OH + KH (and IW = OW + KW) — an off-by-one in each
    //       spatial dimension. (The kernel's own comment claims IH = OH+KH+1,
    //       disagreeing with both the code and the correct value — internally
    //       inconsistent.)
    //   (b) Neuron model. forward_step feeds the conv pre-activation through the
    //       general `lif_step` (membrane decay beta = exp(-dt/tau), configurable
    //       hard/soft reset, v_rest). The kernel uses beta = 1 (plain running
    //       sum) with an always-subtractive `v -= v_th` reset and no v_rest.
    //
    // We therefore validate only that the hand-written PTX LOADS on this SM and
    // that a launch over a self-consistent (kernel-sized) input produces a
    // well-formed result: binary spikes and finite membrane. These assertions
    // CAN fail (e.g. invalid PTX, a 3-D grid launch ABI error, or garbage/NaN
    // output), so the test is meaningful even though it is not an oracle match.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let oc = 2_usize;
    let oh = 3_usize;
    let ow = 3_usize;
    let ic = 1_usize;
    let kh = 2_usize;
    let kw = 2_usize;
    let v_th = 0.1_f32;

    // Size the input to the kernel's own layout: IH = oh+kh, IW = ow+kw.
    let in_h = oh + kh;
    let in_w = ow + kw;
    let in_len = ic * in_h * in_w;
    let w_len = oc * ic * kh * kw;
    let out_len = oc * oh * ow;

    let mut rng = LcgRng::new(0x5C09);
    let input: Vec<f32> = (0..in_len).map(|_| rng.next_f32()).collect();
    let weights: Vec<f32> = (0..w_len).map(|_| rng.next_f32()).collect();
    let v0 = vec![0.0_f32; out_len];

    // LOAD: a failure here is a real PTX/ptxas bug for this SM.
    let ptx = crate::ptx_kernels::spike_conv_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "spike_conv_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<f32>::from_host(&input).expect("d_in");
    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_v = DeviceBuffer::<f32>::from_host(&v0).expect("d_v");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_len]).expect("d_out");

    // grid = (oc, oh, ow), block = 1.
    let params = LaunchParams::new((oc as u32, oh as u32, ow as u32), (1u32, 1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_in.as_device_ptr(),
                d_w.as_device_ptr(),
                d_v.as_device_ptr(),
                d_out.as_device_ptr(),
                oc as u32,
                oh as u32,
                ow as u32,
                ic as u32,
                kh as u32,
                kw as u32,
                v_th,
            ),
        )
        .expect("launch spike_conv_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; out_len];
    let mut v_gpu = vec![0.0_f32; out_len];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");
    d_v.copy_to_host(&mut v_gpu).expect("copy v");

    for (k, &s) in out_gpu.iter().enumerate() {
        assert!(s == 0.0 || s == 1.0, "spike_conv out[{k}] = {s} not binary");
    }
    for (k, &v) in v_gpu.iter().enumerate() {
        assert!(v.is_finite(), "spike_conv membrane[{k}] = {v} not finite");
    }
}
