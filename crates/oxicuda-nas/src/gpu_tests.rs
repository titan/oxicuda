//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's CPU reference (or an independent host re-derivation of the
//! kernel's documented arithmetic). The launch ABI mirrors the working
//! `oxicuda-snn` / `oxicuda-ot` canaries: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars are passed as the matching Rust
//! scalar (`.param .u32` / `.param .f32`), in the kernel's declared parameter
//! order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel mirrors:
//!   `arch_softmax_kernel` ↔ [`crate::ops::mixed_op::softmax`];
//!   `arch_grad_kernel` ↔ [`crate::ops::mixed_op::MixedOp::arch_gradient`].
//! * **Independent host re-derivation** — the op is fused into a larger CPU
//!   routine with no standalone `pub fn`, so the oracle is an independent Rust
//!   re-implementation of the kernel's documented arithmetic:
//!   `mixed_op_blend_kernel`, `gumbel_softmax_kernel` (BASE-E reference — see
//!   the bug note below), `flops_accumulate_kernel`, `pareto_dominate_kernel`,
//!   `crossover_uniform_kernel`. These still genuinely fail if ptxas
//!   miscompiles or the PTX has a wrong constant / shift / index, because the
//!   host code is independent of the JIT-compiled PTX.
//!
//! ## PTX bug found and fixed
//!
//! ### `gumbel_softmax_kernel` — base-2 vs base-e log-conversion error
//!
//! The Gumbel noise `g = -ln(-ln(u+ε)+ε)` is built from `lg2.approx.f32`
//! (base-2 log). To recover a natural log, the base-2 result must be scaled by
//! `ln(2) = 0.6931471806` (hex `0x3F317218`). The original PTX instead scaled
//! it by `1/ln(2) = log2(e) = 1.442695` (hex `0x3FB8AA3B`) — the *inverse* of
//! the correct factor — making every `ln` ≈ 2.08× too large. The resulting
//! Gumbel-softmax still sums to 1 (a shape/sum check misses it), but the
//! distribution is grossly wrong; only the BASE-E host oracle catches it.
//! Fixed in `ptx_kernels.rs` by scaling each `lg2` result by `ln(2)` instead.
//!
//! Every test skips (returns early) when no CUDA device is present, so the
//! suite stays green on CPU-only machines.

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
/// the kernel source, surfaced loudly here rather than skipped.
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
// 1. arch_softmax  —  CRATE ORACLE (ops::mixed_op::softmax)
// ===========================================================================

#[test]
fn arch_softmax_matches_cpu() {
    use crate::ops::mixed_op::softmax;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // One independent softmax over `n` architecture logits. The kernel assumes a
    // single block per softmax (each thread scans all elements for max/sum then
    // writes its own element), so we launch grid = 1, block = n.
    let n = 16_usize;
    let mut rng = LcgRng::new(0x50F_7A0C);
    // Logits spread across [-3, 3): keeps exp(x - max) well inside ex2's accurate
    // domain while still exercising a non-uniform distribution.
    let logits: Vec<f32> = (0..n).map(|_| rng.next_f32() * 6.0 - 3.0).collect();

    // ---- CPU reference (base-e exp) ----
    let w_cpu = softmax(&logits);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::arch_softmax_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "arch_softmax_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let params = LaunchParams::new(1_u32, n as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_logits.as_device_ptr(), d_out.as_device_ptr(), n as u32),
        )
        .expect("launch arch_softmax_kernel");
    stream.synchronize().expect("sync");

    let mut w_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut w_gpu).expect("copy out");

    // Output must be a valid probability distribution.
    let sum: f32 = w_gpu.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "arch_softmax output does not sum to 1: {sum}"
    );

    // GPU uses `ex2.approx.f32` with the correct `* log2(e)` base conversion
    // (~2 ulp); the CPU uses libm `exp` (<1 ulp). 1e-4 relative comfortably
    // covers the approximation yet flags any gross formula / base error.
    let (rel, abs) = worst_diff(&w_gpu, &w_cpu);
    for k in 0..n {
        assert!(
            close(w_gpu[k], w_cpu[k], 1e-4, 1e-6),
            "arch_softmax w[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            w_gpu[k],
            w_cpu[k]
        );
    }
}

// ===========================================================================
// 2. mixed_op_blend  —  INDEPENDENT HOST RE-DERIVATION (weighted op sum)
// ===========================================================================

#[test]
fn mixed_op_blend_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // out[i] = Σ_k w[k] · ops_out[k * n_elems + i]. One thread per element.
    let n_elems = 64_usize;
    let n_ops = 5_usize;
    let mut rng = LcgRng::new(0xB13D_0005);

    let weights: Vec<f32> = (0..n_ops).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let ops_out: Vec<f32> = (0..n_ops * n_elems)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // Independent host re-derivation in the kernel's accumulation order (k ascending).
    let mut out_host = vec![0.0_f32; n_elems];
    for (i, slot) in out_host.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for k in 0..n_ops {
            acc += weights[k] * ops_out[k * n_elems + i];
        }
        *slot = acc;
    }

    let ptx = crate::ptx_kernels::mixed_op_blend_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mixed_op_blend_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_ops = DeviceBuffer::<f32>::from_host(&ops_out).expect("d_ops");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_elems]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_elems as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_ops.as_device_ptr(),
                d_out.as_device_ptr(),
                n_elems as u32,
                n_ops as u32,
            ),
        )
        .expect("launch mixed_op_blend_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_elems];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // GPU fuses with `fma.rn` (one rounding/term) vs host mul+add (two); over
    // n_ops = 5 terms the divergence is a few ulp (~1e-6 relative).
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for k in 0..n_elems {
        assert!(
            close(out_gpu[k], out_host[k], 1e-5, 1e-6),
            "mixed_op_blend out[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_host[k]
        );
    }
}

// ===========================================================================
// 3. gumbel_softmax  —  INDEPENDENT HOST RE-DERIVATION (BASE-E; bug fixed)
// ===========================================================================

/// Base-e host reference for the Gumbel-softmax the kernel documents:
/// `g_i = -ln(-ln(u_i+ε)+ε)`, `perturbed_i = (logit_i + g_i)/τ`, then a
/// max-stabilised softmax over `perturbed`. Independent of the JIT-compiled PTX.
fn gumbel_softmax_host(logits: &[f32], uniform: &[f32], temperature: f32) -> Vec<f32> {
    const EPS: f32 = 1e-10;
    let n = logits.len();
    let mut perturbed = vec![0.0_f32; n];
    let mut max_val = f32::NEG_INFINITY;
    for i in 0..n {
        let g = -((-((uniform[i] + EPS).ln()) + EPS).ln());
        let p = (logits[i] + g) / temperature;
        perturbed[i] = p;
        if p > max_val {
            max_val = p;
        }
    }
    let mut sum = 0.0_f32;
    for &p in &perturbed {
        sum += (p - max_val).exp();
    }
    perturbed
        .iter()
        .map(|&p| (p - max_val).exp() / sum)
        .collect()
}

#[test]
fn gumbel_softmax_matches_host_base_e() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Single Gumbel-softmax over `n` logits; one block, block = n.
    let n = 16_usize;
    let temperature = 1.0_f32;
    let mut rng = LcgRng::new(0x9E_4B_12);
    let logits: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    // Uniforms in [0.05, 0.95): away from 0/1 so the nested logs stay finite and
    // well-conditioned (the kernel's ε floor never dominates).
    let uniform: Vec<f32> = (0..n).map(|_| 0.05 + 0.90 * rng.next_f32()).collect();

    // ---- CPU reference (BASE-E) ----
    let w_cpu = gumbel_softmax_host(&logits, &uniform, temperature);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::gumbel_softmax_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gumbel_softmax_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_uniform = DeviceBuffer::<f32>::from_host(&uniform).expect("d_uniform");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let params = LaunchParams::new(1_u32, n as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_uniform.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                temperature,
            ),
        )
        .expect("launch gumbel_softmax_kernel");
    stream.synchronize().expect("sync");

    let mut w_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut w_gpu).expect("copy out");

    // Output must be a valid probability distribution.
    let sum: f32 = w_gpu.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-3,
        "gumbel_softmax output does not sum to 1: {sum}"
    );

    // After the base-conversion fix, the GPU uses `lg2.approx`/`ex2.approx`
    // (~few ulp) vs the CPU's libm `ln`/`exp`. Over the nested-log Gumbel
    // transform the relative error stays ~1e-5; 1e-3 is a comfortable bound that
    // still catches the (≈2.08×) base-2-vs-base-e error by orders of magnitude.
    let (rel, abs) = worst_diff(&w_gpu, &w_cpu);
    for k in 0..n {
        assert!(
            close(w_gpu[k], w_cpu[k], 1e-3, 1e-5),
            "gumbel_softmax w[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            w_gpu[k],
            w_cpu[k]
        );
    }
}

// ===========================================================================
// 4. flops_accumulate  —  INDEPENDENT HOST RE-DERIVATION (Σ flops·weight)
// ===========================================================================

#[test]
fn flops_accumulate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // total = Σ_i flops[i] · weight[i], accumulated via atom.global.add.f32.
    let n = 100_usize;
    let mut rng = LcgRng::new(0x000F_10B5);
    // Positive operands keep the atomic running sum free of catastrophic
    // cancellation, so its (order-dependent) value stays close to the host sum.
    let flops: Vec<f32> = (0..n).map(|_| 1.0 + rng.next_f32()).collect();
    let weights: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

    let total_host: f32 = (0..n).map(|i| flops[i] * weights[i]).sum();

    let ptx = crate::ptx_kernels::flops_accumulate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "flops_accumulate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_flops = DeviceBuffer::<f32>::from_host(&flops).expect("d_flops");
    let d_weights = DeviceBuffer::<f32>::from_host(&weights).expect("d_weights");
    let d_total = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_total");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_flops.as_device_ptr(),
                d_weights.as_device_ptr(),
                d_total.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch flops_accumulate_kernel");
    stream.synchronize().expect("sync");

    let mut total_gpu = vec![0.0_f32; 1];
    d_total.copy_to_host(&mut total_gpu).expect("copy total");

    // Atomic accumulation order is nondeterministic, so the FP sum differs from
    // the sequential host sum by a handful of ulp over 100 positive terms.
    assert!(
        close(total_gpu[0], total_host, 1e-4, 1e-3),
        "flops_accumulate total mismatch: gpu={} host={}",
        total_gpu[0],
        total_host
    );
}

// ===========================================================================
// 5. pareto_dominate  —  INDEPENDENT HOST RE-DERIVATION (NSGA-II dominance)
// ===========================================================================

#[test]
fn pareto_dominate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // dom[i*n + j] = 1 iff solution i dominates j: all obj_i[k] ≤ obj_j[k] and
    // at least one strictly <. Diagonal (i == j) is 0.
    let n = 6_usize;
    let m = 3_usize; // objectives
    let mut rng = LcgRng::new(0x9A_2E_70);
    let obj: Vec<f32> = (0..n * m).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    let mut dom_host = vec![0_u32; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut all_leq = true;
            let mut any_lt = false;
            for k in 0..m {
                let a = obj[i * m + k];
                let b = obj[j * m + k];
                if a > b {
                    all_leq = false;
                }
                if a < b {
                    any_lt = true;
                }
            }
            dom_host[i * n + j] = u32::from(all_leq && any_lt);
        }
    }

    let ptx = crate::ptx_kernels::pareto_dominate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pareto_dominate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_obj = DeviceBuffer::<f32>::from_host(&obj).expect("d_obj");
    let d_dom = DeviceBuffer::<u32>::from_host(&vec![0_u32; n * n]).expect("d_dom");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d((n * n) as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_obj.as_device_ptr(),
                d_dom.as_device_ptr(),
                n as u32,
                m as u32,
            ),
        )
        .expect("launch pareto_dominate_kernel");
    stream.synchronize().expect("sync");

    let mut dom_gpu = vec![0_u32; n * n];
    d_dom.copy_to_host(&mut dom_gpu).expect("copy dom");

    // Integer dominance relation must match bit-exactly.
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            assert_eq!(
                dom_gpu[idx], dom_host[idx],
                "pareto_dominate dom[{i},{j}] mismatch: gpu={} host={}",
                dom_gpu[idx], dom_host[idx]
            );
        }
    }
}

// ===========================================================================
// 6. arch_grad  —  CRATE ORACLE (ops::mixed_op::MixedOp::arch_gradient)
// ===========================================================================

#[test]
fn arch_grad_matches_cpu() {
    use crate::ops::mixed_op::MixedOp;
    use crate::ops::primitives::OpKind;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_elems = 16_usize;
    let mut rng = LcgRng::new(0x0A64_AD01);

    // Build a MixedOp over the full op set and overwrite its logits with a
    // deterministic spread so the softmax weights are non-uniform (w_k·(1−w_k)
    // is comfortably non-zero). `weights()` is the exact softmax the kernel is
    // fed as `p_weights`.
    let mut op = MixedOp::new(OpKind::all().to_vec(), &mut rng);
    let n_ops = op.n_ops();
    op.arch_params = (0..n_ops).map(|k| (k as f32) * 0.5 - 1.0).collect();
    let w = op.weights();

    let out_grad: Vec<f32> = (0..n_elems).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let op_outputs: Vec<Vec<f32>> = (0..n_ops)
        .map(|_| (0..n_elems).map(|_| rng.next_f32() * 2.0 - 1.0).collect())
        .collect();

    // ---- CPU reference (crate) ----
    let grad_cpu = op.arch_gradient(&out_grad, &op_outputs);

    // Flatten op outputs to the kernel's [k * n_elems + i] layout.
    let mut op_flat = vec![0.0_f32; n_ops * n_elems];
    for k in 0..n_ops {
        for i in 0..n_elems {
            op_flat[k * n_elems + i] = op_outputs[k][i];
        }
    }

    let ptx = crate::ptx_kernels::arch_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "arch_grad_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_grad = DeviceBuffer::<f32>::from_host(&out_grad).expect("d_grad");
    let d_ops = DeviceBuffer::<f32>::from_host(&op_flat).expect("d_ops");
    let d_galpha = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_ops]).expect("d_galpha");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_ops as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_grad.as_device_ptr(),
                d_ops.as_device_ptr(),
                d_galpha.as_device_ptr(),
                n_ops as u32,
                n_elems as u32,
            ),
        )
        .expect("launch arch_grad_kernel");
    stream.synchronize().expect("sync");

    let mut grad_gpu = vec![0.0_f32; n_ops];
    d_galpha.copy_to_host(&mut grad_gpu).expect("copy grad");

    // GPU fuses the dot product with `fma.rn`; the CPU sums plain products.
    // Over n_elems = 16 terms the divergence is ~1e-6 relative.
    let (rel, abs) = worst_diff(&grad_gpu, &grad_cpu);
    for k in 0..n_ops {
        assert!(
            close(grad_gpu[k], grad_cpu[k], 1e-4, 1e-6),
            "arch_grad grad_alpha[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            grad_gpu[k],
            grad_cpu[k]
        );
    }
}

// ===========================================================================
// 7. crossover_uniform  —  INDEPENDENT HOST RE-DERIVATION (masked select)
// ===========================================================================

#[test]
fn crossover_uniform_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // child[i] = (mask[i] != 0) ? parent_a[i] : parent_b[i]; all u32 genes.
    let n = 70_usize;
    let mut rng = LcgRng::new(0xC0_5A_07);
    let parent_a: Vec<u32> = (0..n).map(|_| rng.next_u32() % 11).collect();
    let parent_b: Vec<u32> = (0..n).map(|_| rng.next_u32() % 11).collect();
    let mask: Vec<u32> = (0..n).map(|_| u32::from(rng.next_f32() < 0.5)).collect();

    let child_host: Vec<u32> = (0..n)
        .map(|i| {
            if mask[i] != 0 {
                parent_a[i]
            } else {
                parent_b[i]
            }
        })
        .collect();

    let ptx = crate::ptx_kernels::crossover_uniform_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "crossover_uniform_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<u32>::from_host(&parent_a).expect("d_a");
    let d_b = DeviceBuffer::<u32>::from_host(&parent_b).expect("d_b");
    let d_mask = DeviceBuffer::<u32>::from_host(&mask).expect("d_mask");
    let d_child = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_child");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_mask.as_device_ptr(),
                d_child.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch crossover_uniform_kernel");
    stream.synchronize().expect("sync");

    let mut child_gpu = vec![0_u32; n];
    d_child.copy_to_host(&mut child_gpu).expect("copy child");

    for i in 0..n {
        assert_eq!(
            child_gpu[i], child_host[i],
            "crossover child[{i}] mismatch: gpu={} host={} (mask={}, a={}, b={})",
            child_gpu[i], child_host[i], mask[i], parent_a[i], parent_b[i]
        );
    }
}
