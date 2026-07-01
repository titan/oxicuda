//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies the results
//! back, and asserts numerical equivalence to the crate's CPU reference. The
//! launch ABI mirrors the proven `oxicuda-snn` canary: device buffers are passed
//! as their `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .u32` / `.param .f32` / `.param .u64`), in declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   `inner_sgd_kernel` ↔ [`crate::gradient::inner_loop::inner_sgd_step`].
//! * **Independent host re-derivation** — the op has no standalone `pub fn` (it
//!   is fused into a larger routine on the CPU), so the oracle is an independent
//!   Rust re-implementation of the kernel's *documented* arithmetic:
//!   `reptile_update_kernel` (per-element interpolation),
//!   `proto_distance_kernel` (squared Euclidean distance),
//!   `cosine_sim_kernel` (dot / (‖a‖·‖b‖ + ε)),
//!   `relation_score_kernel` (two-layer ReLU MLP → sigmoid),
//!   `meta_grad_accum_kernel` (mean over tasks), and
//!   `episode_sample_kernel` (Fisher–Yates with the kernel's inline LCG). These
//!   still genuinely fail if ptxas miscompiles or the PTX has a wrong constant /
//!   shift / index, because the host code is independent of the JIT-compiled PTX.
//!
//! All seven kernels are real, store-bearing implementations (none is a stub):
//! every test below asserts a non-trivial computed result, not a no-op.
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real,
/// must-fix bug, so we panic loudly rather than skip.
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
// 1. inner_sgd  —  CRATE ORACLE (gradient::inner_loop::inner_sgd_step)
// ===========================================================================

#[test]
fn inner_sgd_matches_cpu() {
    use crate::gradient::inner_loop::inner_sgd_step;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let alpha = 0.1_f32;

    let mut rng = LcgRng::new(0x1_5E5D);
    let theta: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // ---- CPU reference: theta' = theta - alpha * grad ----
    let out_cpu = inner_sgd_step(&theta, &grad, alpha).expect("inner_sgd_step");

    let ptx = crate::ptx_kernels::inner_sgd_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "inner_sgd_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_theta = DeviceBuffer::<f32>::from_host(&theta).expect("d_theta");
    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_theta.as_device_ptr(),
                d_grad.as_device_ptr(),
                d_out.as_device_ptr(),
                alpha,
                n as u32,
            ),
        )
        .expect("launch inner_sgd_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // GPU `mul`/`sub` and the CPU `p - lr*g` are both two-rounding; ~1 ulp.
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..n {
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-5, 1e-7),
            "inner_sgd out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k]
        );
    }
}

// ===========================================================================
// 2. reptile_update  —  INDEPENDENT HOST RE-DERIVATION (in-place interpolation)
// ===========================================================================

#[test]
fn reptile_update_matches_host() {
    // The crate's `reptile_update` is a full multi-task meta-update; the kernel
    // computes only the per-element interpolation `theta += eps*(theta' - theta)`,
    // so the oracle is an independent host re-derivation of that exact formula.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let eps = 0.3_f32;

    let mut rng = LcgRng::new(0x2_7E97);
    let theta0: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let theta_prime: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // ---- Host reference: theta_new[i] = theta[i] + eps*(theta'[i] - theta[i]) ----
    let expected: Vec<f32> = theta0
        .iter()
        .zip(theta_prime.iter())
        .map(|(&t, &tp)| t + eps * (tp - t))
        .collect();

    let ptx = crate::ptx_kernels::reptile_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "reptile_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // The kernel updates theta in place (writes back to p_theta).
    let d_theta = DeviceBuffer::<f32>::from_host(&theta0).expect("d_theta");
    let d_prime = DeviceBuffer::<f32>::from_host(&theta_prime).expect("d_prime");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_theta.as_device_ptr(),
                d_prime.as_device_ptr(),
                eps,
                n as u32,
            ),
        )
        .expect("launch reptile_update_kernel");
    stream.synchronize().expect("sync");

    let mut theta_gpu = vec![0.0_f32; n];
    d_theta.copy_to_host(&mut theta_gpu).expect("copy theta");

    let (rel, abs) = worst_diff(&theta_gpu, &expected);
    for k in 0..n {
        assert!(
            close(theta_gpu[k], expected[k], 1e-5, 1e-7),
            "reptile theta[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            theta_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 3. proto_distance  —  INDEPENDENT HOST RE-DERIVATION (squared Euclidean)
// ===========================================================================

#[test]
fn proto_distance_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_query = 4_usize;
    let n_way = 3_usize;
    let feat_dim = 5_usize;

    let mut rng = LcgRng::new(0x3_0157);
    let query: Vec<f32> = (0..n_query * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let proto: Vec<f32> = (0..n_way * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // ---- Host reference: d[q*n_way + k] = Σ_j (query[q,j] - proto[k,j])² ----
    let mut d_host = vec![0.0_f32; n_query * n_way];
    for q in 0..n_query {
        for k in 0..n_way {
            let mut s = 0.0_f32;
            for j in 0..feat_dim {
                let diff = query[q * feat_dim + j] - proto[k * feat_dim + j];
                s += diff * diff;
            }
            d_host[q * n_way + k] = s;
        }
    }

    let ptx = crate::ptx_kernels::proto_distance_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "proto_distance_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_query = DeviceBuffer::<f32>::from_host(&query).expect("d_query");
    let d_proto = DeviceBuffer::<f32>::from_host(&proto).expect("d_proto");
    let d_dist = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_query * n_way]).expect("d_dist");

    // One thread per (query, way) cell; flat global thread id = q*n_way + k.
    let total = (n_query * n_way) as u32;
    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_query.as_device_ptr(),
                d_proto.as_device_ptr(),
                d_dist.as_device_ptr(),
                n_query as u32,
                n_way as u32,
                feat_dim as u32,
            ),
        )
        .expect("launch proto_distance_kernel");
    stream.synchronize().expect("sync");

    let mut d_gpu = vec![0.0_f32; n_query * n_way];
    d_dist.copy_to_host(&mut d_gpu).expect("copy dist");

    // GPU fuses each (diff*diff + acc) with `fma.rn`; host uses mul+add. A few ulp.
    let (rel, abs) = worst_diff(&d_gpu, &d_host);
    for k in 0..d_gpu.len() {
        assert!(
            close(d_gpu[k], d_host[k], 1e-4, 1e-5),
            "proto_distance d[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            d_gpu[k],
            d_host[k]
        );
    }
}

// ===========================================================================
// 4. cosine_sim  —  INDEPENDENT HOST RE-DERIVATION (dot / (‖a‖·‖b‖ + ε))
// ===========================================================================

#[test]
fn cosine_sim_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 8_usize;
    let feat_dim = 6_usize;
    let eps = 1e-8_f32;

    let mut rng = LcgRng::new(0x4_C051);
    // Random vectors of norm ≈ √(feat_dim/3) ≈ 1.4 — comfortably away from the
    // ε-floored denominator, so the comparison is never near a zero crossing.
    let a: Vec<f32> = (0..n * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let b: Vec<f32> = (0..n * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // ---- Host reference ----
    let mut sim_host = vec![0.0_f32; n];
    for i in 0..n {
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for j in 0..feat_dim {
            let av = a[i * feat_dim + j];
            let bv = b[i * feat_dim + j];
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        sim_host[i] = dot / (na.sqrt() * nb.sqrt() + eps);
    }

    let ptx = crate::ptx_kernels::cosine_sim_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cosine_sim_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                feat_dim as u32,
            ),
        )
        .expect("launch cosine_sim_kernel");
    stream.synchronize().expect("sync");

    let mut sim_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut sim_gpu).expect("copy sim");

    // fma accumulation + sqrt.rn + div.rn vs host mul/add + sqrt + div: a few ulp.
    let (rel, abs) = worst_diff(&sim_gpu, &sim_host);
    for i in 0..n {
        assert!(
            close(sim_gpu[i], sim_host[i], 1e-4, 1e-6),
            "cosine_sim sim[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            sim_gpu[i],
            sim_host[i]
        );
    }
}

// ===========================================================================
// 5. relation_score  —  INDEPENDENT HOST RE-DERIVATION (ReLU MLP → sigmoid)
// ===========================================================================

#[test]
fn relation_score_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let feat_dim = 3_usize;
    let hidden_dim = 4_usize;
    let n_pairs = 5_usize;
    let in_dim = 2 * feat_dim;

    let mut rng = LcgRng::new(0x5_5C03);
    let query: Vec<f32> = (0..n_pairs * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let support: Vec<f32> = (0..n_pairs * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    // Small weights keep the pre-sigmoid argument O(1), so `ex2.approx`'s
    // exponential stays well inside its accurate range (no overflow).
    let w1: Vec<f32> = (0..hidden_dim * in_dim)
        .map(|_| rng.next_f32() - 0.5)
        .collect();
    let b1: Vec<f32> = (0..hidden_dim).map(|_| rng.next_f32() - 0.5).collect();
    let w2: Vec<f32> = (0..hidden_dim).map(|_| rng.next_f32() - 0.5).collect();
    let b2: Vec<f32> = vec![0.1_f32];

    // ---- Host reference: ReLU MLP forward then sigmoid (base-e). ----
    let mut score_host = vec![0.0_f32; n_pairs];
    for p in 0..n_pairs {
        let mut pre_sig = b2[0];
        for j in 0..hidden_dim {
            let mut acc = b1[j];
            for i in 0..feat_dim {
                acc += w1[j * in_dim + i] * query[p * feat_dim + i];
                acc += w1[j * in_dim + feat_dim + i] * support[p * feat_dim + i];
            }
            pre_sig += w2[j] * acc.max(0.0_f32);
        }
        score_host[p] = 1.0_f32 / (1.0_f32 + (-pre_sig).exp());
    }

    let ptx = crate::ptx_kernels::relation_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "relation_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_query = DeviceBuffer::<f32>::from_host(&query).expect("d_query");
    let d_support = DeviceBuffer::<f32>::from_host(&support).expect("d_support");
    let d_w1 = DeviceBuffer::<f32>::from_host(&w1).expect("d_w1");
    let d_b1 = DeviceBuffer::<f32>::from_host(&b1).expect("d_b1");
    let d_w2 = DeviceBuffer::<f32>::from_host(&w2).expect("d_w2");
    let d_b2 = DeviceBuffer::<f32>::from_host(&b2).expect("d_b2");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_pairs]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_pairs as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_query.as_device_ptr(),
                d_support.as_device_ptr(),
                d_w1.as_device_ptr(),
                d_b1.as_device_ptr(),
                d_w2.as_device_ptr(),
                d_b2.as_device_ptr(),
                d_out.as_device_ptr(),
                feat_dim as u32,
                hidden_dim as u32,
                n_pairs as u32,
            ),
        )
        .expect("launch relation_score_kernel");
    stream.synchronize().expect("sync");

    let mut score_gpu = vec![0.0_f32; n_pairs];
    d_out.copy_to_host(&mut score_gpu).expect("copy score");

    // The sigmoid uses `ex2.approx.f32` (~2 ulp) with a `* log2(e)` base
    // conversion; 5e-4 relative covers that yet flags any gross formula error
    // (e.g. a missing base conversion would be ~20–40 % off).
    let (rel, abs) = worst_diff(&score_gpu, &score_host);
    for p in 0..n_pairs {
        assert!(
            (0.0_f32..=1.0_f32).contains(&score_gpu[p]),
            "relation_score[{p}] = {} not a valid probability",
            score_gpu[p]
        );
        assert!(
            close(score_gpu[p], score_host[p], 5e-4, 1e-6),
            "relation_score[{p}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            score_gpu[p],
            score_host[p]
        );
    }
}

// ===========================================================================
// 6. meta_grad_accum  —  INDEPENDENT HOST RE-DERIVATION (mean over tasks)
// ===========================================================================

#[test]
fn meta_grad_accum_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_params = 16_usize;
    let n_tasks = 4_usize;

    let mut rng = LcgRng::new(0x6_6ACC);
    // Row-major [n_tasks * n_params]: grads[t*n_params + i].
    let grads: Vec<f32> = (0..n_tasks * n_params)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // ---- Host reference: out[i] = (Σ_t grads[t, i]) / n_tasks ----
    let mut out_host = vec![0.0_f32; n_params];
    for (i, slot) in out_host.iter_mut().enumerate() {
        let mut s = 0.0_f32;
        for t in 0..n_tasks {
            s += grads[t * n_params + i];
        }
        *slot = s / n_tasks as f32;
    }

    let ptx = crate::ptx_kernels::meta_grad_accum_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "meta_grad_accum_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grads = DeviceBuffer::<f32>::from_host(&grads).expect("d_grads");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_params]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_params as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_grads.as_device_ptr(),
                d_out.as_device_ptr(),
                n_params as u32,
                n_tasks as u32,
            ),
        )
        .expect("launch meta_grad_accum_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_params];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for i in 0..n_params {
        assert!(
            close(out_gpu[i], out_host[i], 1e-5, 1e-6),
            "meta_grad_accum out[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_host[i]
        );
    }
}

// ===========================================================================
// 7. episode_sample  —  INDEPENDENT HOST RE-DERIVATION (Fisher–Yates + LCG)
// ===========================================================================

/// Host re-derivation of the `episode_sample_kernel`'s in-place Fisher–Yates
/// shuffle. The kernel seeds an identity permutation `0..n_classes`, then for
/// `i = n_classes-1 .. 1` advances a 64-bit MMIX LCG once, takes `j = (state >>
/// 33) mod (i+1)`, and swaps `indices[i] <-> indices[j]`. u64 integer arithmetic
/// is perfectly reproducible, so this is bit-exact against the GPU.
fn episode_sample_host(n_classes: usize, seed: u64) -> Vec<u32> {
    const M: u64 = 6_364_136_223_846_793_005;
    const A: u64 = 1_442_695_040_888_963_407;
    let mut indices: Vec<u32> = (0..n_classes as u32).collect();
    let mut state = seed;
    for i in (1..n_classes).rev() {
        state = state.wrapping_mul(M).wrapping_add(A);
        let r = (state >> 33) as u32;
        let j = (r % (i as u32 + 1)) as usize;
        indices.swap(i, j);
    }
    indices
}

#[test]
fn episode_sample_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_classes = 8_usize;
    let n_way = 5_u32; // loaded but unused by the kernel (it shuffles the full array)
    let seed = 0x1234_5678_9ABC_DEF0_u64;

    // ---- Host reference (bit-exact permutation) ----
    let expected = episode_sample_host(n_classes, seed);

    let ptx = crate::ptx_kernels::episode_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "episode_sample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_classes]).expect("d_out");

    // Single thread (the kernel guards on tid.x == 0 and does serial Fisher–Yates).
    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_out.as_device_ptr(), n_classes as u32, n_way, seed),
        )
        .expect("launch episode_sample_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; n_classes];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // The shuffled array must be a permutation of 0..n_classes (each index once).
    let mut seen = vec![false; n_classes];
    for &v in &out_gpu {
        assert!(
            (v as usize) < n_classes,
            "episode_sample produced out-of-range index {v} (n_classes={n_classes})"
        );
        assert!(
            !seen[v as usize],
            "episode_sample produced duplicate index {v} (not a permutation)"
        );
        seen[v as usize] = true;
    }

    // Bit-exact against the independent host Fisher–Yates re-derivation.
    assert_eq!(
        out_gpu, expected,
        "episode_sample permutation mismatch: gpu={out_gpu:?} host={expected:?}"
    );
}
