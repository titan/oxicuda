//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it through `oxicuda-launch`, copies the results back, and asserts
//! numerical equivalence to the crate's CPU reference (or a documented
//! independent re-derivation). The launch ABI mirrors the working `oxicuda-snn`
//! canary path: device buffers are passed as their `CUdeviceptr` (`.param .u64`),
//! scalars as the matching Rust type (`.param .u32` / `.param .f32`), in declared
//! PTX order.
//!
//! Every test skips (returns early) when no CUDA device is present.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tol to a `pub` CPU
//!   function the kernel is meant to mirror:
//!   `mse_distill_kernel`, `at_pool_kernel`, `dml_loss_kernel`,
//!   `crd_score_kernel`, `gram_matrix_kernel`.
//! * **Independent host re-derivation** — the kernel has no single dedicated
//!   crate function for the exact per-head raw-sum form, so the oracle is an
//!   independent Rust re-implementation of the kernel's *documented* arithmetic:
//!   `attn_distill_kernel`.
//! * **Arithmetic oracle + design discrepancy flagged** — `kd_loss_kernel` is
//!   compared against an independent CPU re-derivation of the kernel's *actual*
//!   computation (unnormalized KL proxy), which validates the arithmetic.
//!   HOWEVER the kernel does NOT implement the documented Hinton KD loss
//!   (`logit::hinton_kd::kd_loss`) because it never computes softmax partition
//!   functions. This is a known design discrepancy reported below.

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
/// alive for the whole test.
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
    let dev = Device::get(0).ok()?;
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
// 1. kd_loss_kernel — ARITHMETIC ORACLE + DESIGN DISCREPANCY FLAGGED
// ===========================================================================
//
// ORACLE TIER: The kernel is compared against an independent CPU re-derivation
// of the kernel's *actual* arithmetic, not against `logit::hinton_kd::kd_loss`.
//
// DESIGN DISCREPANCY: `kd_loss_kernel` accumulates, per (batch, class) thread:
//
//   exp(t_i / T) · ln(exp(t_i / T) / exp(s_i / T))
//   = exp(t_i / T) · (t_i − s_i) / T
//
// The per-batch total therefore equals  Z_t · [KL(p_t ‖ p_s) + ln(Z_t / Z_s)],
// where Z_t = Σ exp(t_i/T). This diverges from the CPU `kd_loss` function
// (which correctly divides by Z_t to obtain proper KL on normalised
// distributions). The kernel would require a two-pass design or shared-memory
// reduction to compute proper Hinton KD loss; no PTX fix is attempted here.
//
// What the tests DO catch:
//   - PTX load failure on this SM (genuine ptxas bug)
//   - Wrong log constant, wrong sign, wrong scaling (arithmetic bugs)
//   - Case s=t: GPU output must be exactly 0 (no fudge: lg2(1.0)=0 exactly)

/// CPU re-derivation of the kernel's actual unnormalized per-class computation:
/// Σ_{b,i}  exp(t_{b,i}/T) · (t_{b,i} − s_{b,i}) / T
/// (skipping classes where the ratio is ≤ 0, matching the kernel's `setp.gt` guards).
fn kd_loss_unnorm_oracle(
    s_logits: &[f32],
    t_logits: &[f32],
    batch: usize,
    n: usize,
    temp: f32,
) -> f32 {
    let mut total = 0.0_f32;
    for b in 0..batch {
        for i in 0..n {
            let idx = b * n + i;
            let s = s_logits[idx];
            let t = t_logits[idx];
            let t_u = (t / temp).exp();
            let s_u = (s / temp).exp();
            if t_u > 0.0 {
                let ratio = t_u / s_u;
                // mirror kernel guard: ratio > FLT_MIN (normal float)
                if ratio > f32::MIN_POSITIVE {
                    let ln_ratio = ratio.ln();
                    total += t_u * ln_ratio;
                }
            }
        }
    }
    total
}

#[test]
fn kd_loss_loads_and_arithmetic_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let batch = 4_usize;
    let n_classes = 8_usize;
    let temp = 4.0_f32;

    // --- Sub-test A: s == t → GPU output must be exactly 0 ---
    let logits_eq: Vec<f32> = (0..batch * n_classes)
        .map(|k| (k as f32 * 0.25) - 1.0) // values in [-1, 1)
        .collect();

    let ptx = crate::ptx_kernels::kd_loss_ptx(fx.sm);
    // LOAD: a failure here is a real PTX/ptxas bug for this SM.
    let kernel = load_kernel(&ptx, "kd_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    {
        let d_s = DeviceBuffer::<f32>::from_host(&logits_eq).expect("d_s_eq");
        let d_t = DeviceBuffer::<f32>::from_host(&logits_eq).expect("d_t_eq");
        let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out_eq");

        // Grid=(batch, 1, 1), Block=(n_classes, 1, 1): one thread per (batch, class).
        let params = LaunchParams::new((batch as u32, 1u32, 1u32), (n_classes as u32, 1u32, 1u32));
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_s.as_device_ptr(),
                    d_t.as_device_ptr(),
                    d_out.as_device_ptr(),
                    n_classes as u32,
                    batch as u32,
                    temp,
                ),
            )
            .expect("launch kd_loss_kernel s==t");
        stream.synchronize().expect("sync");

        let mut out = vec![0.0_f32; 1];
        d_out.copy_to_host(&mut out).expect("copy");
        // lg2.approx(1.0) == 0 exactly; t_u * 0 = 0 for every element.
        assert_eq!(
            out[0], 0.0_f32,
            "kd_loss s==t: expected exactly 0.0, got {}",
            out[0]
        );
    }

    // --- Sub-test B: s ≠ t → arithmetic oracle match (1e-3 rel, accounts for approx ops) ---
    let mut rng = LcgRng::new(0x4B44_5F11);
    let s_logits: Vec<f32> = (0..batch * n_classes)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();
    let t_logits: Vec<f32> = (0..batch * n_classes)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();

    let oracle = kd_loss_unnorm_oracle(&s_logits, &t_logits, batch, n_classes, temp);
    // Oracle must be positive (different logits → nonzero unnormalized KL).
    assert!(
        oracle > 0.0 && oracle.is_finite(),
        "kd_loss oracle not positive-finite: {oracle}"
    );

    let d_s = DeviceBuffer::<f32>::from_host(&s_logits).expect("d_s");
    let d_t = DeviceBuffer::<f32>::from_host(&t_logits).expect("d_t");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let params = LaunchParams::new((batch as u32, 1u32, 1u32), (n_classes as u32, 1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_s.as_device_ptr(),
                d_t.as_device_ptr(),
                d_out.as_device_ptr(),
                n_classes as u32,
                batch as u32,
                temp,
            ),
        )
        .expect("launch kd_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; 1];
    d_out.copy_to_host(&mut out).expect("copy");

    let gpu_val = out[0];
    assert!(
        gpu_val.is_finite(),
        "kd_loss GPU output not finite: {gpu_val}"
    );
    // NOTE: the oracle matches the kernel's unnormalized arithmetic, NOT the
    // documented Hinton KD loss. See module-level doc for the discrepancy.
    assert!(
        close(gpu_val, oracle, 1e-3, 1e-5),
        "kd_loss arithmetic oracle mismatch: gpu={gpu_val} oracle={oracle} (rel diff {})",
        (gpu_val - oracle).abs() / oracle.abs().max(1e-9)
    );
}

// ===========================================================================
// 2. mse_distill_kernel — CRATE ORACLE (feature::fitnets::mse)
// ===========================================================================
//
// The kernel accumulates Σ(s_i − t_i)² into out_mse[0] via atomic add.
// The comment says "divide by n on host to obtain the mean", making
// gpu_result / n ≈ mse(s, t) from `crate::feature::fitnets`.

#[test]
fn mse_distill_matches_fitnets_mse() {
    use crate::feature::fitnets::mse;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let mut rng = LcgRng::new(0xA3E1_5C07);
    let s_feat: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let t_feat: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: mean squared error.
    let cpu_mse = mse(&s_feat, &t_feat);
    assert!(cpu_mse > 0.0 && cpu_mse.is_finite(), "cpu_mse={cpu_mse}");

    let ptx = crate::ptx_kernels::mse_distill_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mse_distill_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_s = DeviceBuffer::<f32>::from_host(&s_feat).expect("d_s");
    let d_t = DeviceBuffer::<f32>::from_host(&t_feat).expect("d_t");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 128_u32;
    let grid = grid_1d(n as u32, block);
    let params = LaunchParams::new(grid, block);

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_s.as_device_ptr(),
                d_t.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch mse_distill_kernel");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; 1];
    d_out.copy_to_host(&mut out).expect("copy");

    let gpu_sum = out[0];
    assert!(
        gpu_sum.is_finite() && gpu_sum >= 0.0,
        "mse_distill GPU output not finite/non-negative: {gpu_sum}"
    );

    let gpu_mse = gpu_sum / n as f32;
    let (rel, abs) = worst_diff(&[gpu_mse], &[cpu_mse]);
    assert!(
        close(gpu_mse, cpu_mse, 1e-4, 1e-7),
        "mse_distill mismatch: gpu={gpu_mse} cpu={cpu_mse} (rel={rel:e} abs={abs:e})"
    );
}

// ===========================================================================
// 3. attn_distill_kernel — INDEPENDENT HOST RE-DERIVATION (per-head raw sum)
// ===========================================================================
//
// The kernel accumulates Σ_{k in head h} (s_attn[k] − t_attn[k])² into
// out_loss[h] for each head. Dividing by seq_sq recovers the per-head MSE,
// which matches `attention::attn_distill::attn_loss(s_head, t_head)`.
//
// There is no standalone crate function that computes the *raw sum* form, so
// the oracle is an independent host re-derivation of the kernel's documented
// arithmetic, then divided by seq_sq and compared to `attn_loss`.

#[test]
fn attn_distill_matches_per_head_mse() {
    use crate::attention::attn_distill::attn_loss;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_heads = 4_usize;
    let seq_len = 8_usize;
    let seq_sq = seq_len * seq_len; // 64 per head
    let total = n_heads * seq_sq;

    let mut rng = LcgRng::new(0x22FB_C903);
    let s_attn: Vec<f32> = (0..total).map(|_| rng.next_f32()).collect();
    let t_attn: Vec<f32> = (0..total).map(|_| rng.next_f32()).collect();

    // CPU reference: per-head attn_loss (MSE over seq_sq elements per head).
    let cpu_per_head: Vec<f32> = (0..n_heads)
        .map(|h| {
            let lo = h * seq_sq;
            let hi = lo + seq_sq;
            attn_loss(&s_attn[lo..hi], &t_attn[lo..hi])
        })
        .collect();

    let ptx = crate::ptx_kernels::attn_distill_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "attn_distill_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_s = DeviceBuffer::<f32>::from_host(&s_attn).expect("d_s");
    let d_t = DeviceBuffer::<f32>::from_host(&t_attn).expect("d_t");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_heads]).expect("d_out");

    // 1D grid-stride over n_heads * seq_sq total elements.
    let block = 128_u32;
    let grid = grid_1d(total as u32, block);
    let params = LaunchParams::new(grid, block);

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_s.as_device_ptr(),
                d_t.as_device_ptr(),
                d_out.as_device_ptr(),
                n_heads as u32,
                seq_sq as u32,
            ),
        )
        .expect("launch attn_distill_kernel");
    stream.synchronize().expect("sync");

    let mut gpu_raw = vec![0.0_f32; n_heads];
    d_out.copy_to_host(&mut gpu_raw).expect("copy");

    let (worst_rel, worst_abs) = worst_diff(
        &gpu_raw
            .iter()
            .map(|&v| v / seq_sq as f32)
            .collect::<Vec<_>>(),
        &cpu_per_head,
    );

    for h in 0..n_heads {
        let gpu_mse_h = gpu_raw[h] / seq_sq as f32;
        let cpu_mse_h = cpu_per_head[h];
        assert!(
            close(gpu_mse_h, cpu_mse_h, 1e-4, 1e-7),
            "attn_distill head {h} mismatch: gpu={gpu_mse_h} cpu={cpu_mse_h} \
             (worst rel={worst_rel:e} abs={worst_abs:e})"
        );
    }
}

// ===========================================================================
// 4. at_pool_kernel — CRATE ORACLE (feature::at::at_map)
// ===========================================================================
//
// The kernel computes out[hw] = Σ_c |feat[c*hw + hw_idx]|^p using the
// log/exp approximation path (lg2.approx + ex2.approx). The crate's `at_map`
// uses Rust's `powf`, which is a correctly-rounded implementation for p=2.0.
//
// Tolerance: 2e-3 relative — the combined lg2.approx + ex2.approx path
// incurs ~4 ulp ≈ 5e-7 relative per element, but with channel summation and
// the worst-case interaction the bound is generous at 2e-3. This is still
// orders of magnitude tighter than a gross formula error (wrong sign, wrong
// channel count, wrong p-value) would produce.
//
// Input constraint: |feat| > 0.05 to stay well above the kernel's FLT_MIN
// guard (|x| ≤ FLT_MIN ≈ 1.18e-38 is clamped to 0).

#[test]
fn at_pool_matches_at_map() {
    use crate::feature::at::at_map;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let channels = 8_usize;
    let h = 4_usize;
    let w = 4_usize;
    let hw = h * w; // 16
    let p_exp = 2.0_f32;

    // Keep |feat| in [0.1, 1.6] — well above FLT_MIN, so the kernel's lg2 path
    // is always taken and doesn't clip to 0 prematurely.
    let mut rng = LcgRng::new(0x7CE0_3A5D);
    let feat: Vec<f32> = (0..channels * hw)
        .map(|_| 0.1 + rng.next_f32() * 1.5)
        .collect();

    // CPU reference: at_map(feat, channels, h, w, p).
    let cpu_out = at_map(&feat, channels, h, w, p_exp);
    assert_eq!(cpu_out.len(), hw);
    for (i, &v) in cpu_out.iter().enumerate() {
        assert!(
            v.is_finite() && v >= 0.0,
            "cpu at_map[{i}]={v} not finite/non-neg"
        );
    }

    let ptx = crate::ptx_kernels::at_pool_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "at_pool_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_feat = DeviceBuffer::<f32>::from_host(&feat).expect("d_feat");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; hw]).expect("d_out");

    // 1D grid-stride over hw spatial locations.
    let block = 64_u32;
    let grid = grid_1d(hw as u32, block);
    let params = LaunchParams::new(grid, block);

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_feat.as_device_ptr(),
                d_out.as_device_ptr(),
                channels as u32,
                hw as u32,
                p_exp,
            ),
        )
        .expect("launch at_pool_kernel");
    stream.synchronize().expect("sync");

    let mut gpu_out = vec![0.0_f32; hw];
    d_out.copy_to_host(&mut gpu_out).expect("copy");

    let (worst_rel, worst_abs) = worst_diff(&gpu_out, &cpu_out);
    for k in 0..hw {
        assert!(
            close(gpu_out[k], cpu_out[k], 2e-3, 1e-6),
            "at_pool hw[{k}] mismatch: gpu={} cpu={} (worst rel={worst_rel:e} abs={worst_abs:e})",
            gpu_out[k],
            cpu_out[k]
        );
    }
}

// ===========================================================================
// 5. dml_loss_kernel — CRATE ORACLE (online::dml::kl_divergence, block = 1)
// ===========================================================================
//
// The kernel accumulates KL(self_probs ‖ peer_probs[peer]) per peer block.
// It takes already-normalised probability vectors, not raw logits.
//
// LAUNCH CONSTRAINT: block must be (1, 1, 1). Each thread accumulates a local
// register sum over classes and then only thread 0 atomically writes to
// out_kl[peer]. With block_x > 1 the contributions from threads 1, 2, …
// are silently discarded (no shared-memory reduction). This is a kernel design
// limitation; launching with block = (1,1,1) gives correct results.
//
// Tolerance: 5e-3 relative — the lg2.approx path introduces ~2 ulp per log
// evaluation; summed over n_classes this is well within 5e-3 for the value
// ranges used here. Truly wrong formula (wrong sign, missing class) would
// differ by orders of magnitude.

#[test]
fn dml_loss_matches_kl_divergence() {
    use crate::online::dml::kl_divergence;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_peers = 4_usize;
    let n_classes = 6_usize;
    let mut rng = LcgRng::new(0x9D7C_44A2);

    // Build probability distributions with no near-zero class (min ≈ 0.08)
    // so the lg2.approx and EPS differences don't matter.
    let make_probs = |rng: &mut LcgRng| -> Vec<f32> {
        // logits in [-0.5, 0.5] → softmax values roughly uniform, min > 0.05
        let raw: Vec<f32> = (0..n_classes).map(|_| rng.next_f32() - 0.5).collect();
        let max = raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = raw.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    };

    let self_probs = make_probs(&mut rng);
    let peer_probs_flat: Vec<f32> = (0..n_peers).flat_map(|_| make_probs(&mut rng)).collect();

    // CPU reference: KL(self_probs ‖ peer_probs[p]) for each peer p.
    let cpu_kl: Vec<f32> = (0..n_peers)
        .map(|p| {
            let peer = &peer_probs_flat[p * n_classes..(p + 1) * n_classes];
            kl_divergence(&self_probs, peer)
        })
        .collect();

    for (i, &v) in cpu_kl.iter().enumerate() {
        assert!(
            v.is_finite() && v >= 0.0,
            "cpu kl_divergence[{i}] not finite/non-neg: {v}"
        );
    }

    let ptx = crate::ptx_kernels::dml_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dml_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_self = DeviceBuffer::<f32>::from_host(&self_probs).expect("d_self");
    let d_peers = DeviceBuffer::<f32>::from_host(&peer_probs_flat).expect("d_peers");
    // out_kl initialised to 0; kernel atomically adds each peer's result.
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_peers]).expect("d_out");

    // IMPORTANT: block = (1,1,1) — see LAUNCH CONSTRAINT above.
    let params = LaunchParams::new((n_peers as u32, 1u32, 1u32), (1u32, 1u32, 1u32));

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_self.as_device_ptr(),
                d_peers.as_device_ptr(),
                d_out.as_device_ptr(),
                n_classes as u32,
                n_peers as u32,
            ),
        )
        .expect("launch dml_loss_kernel");
    stream.synchronize().expect("sync");

    let mut gpu_kl = vec![0.0_f32; n_peers];
    d_out.copy_to_host(&mut gpu_kl).expect("copy");

    let (worst_rel, worst_abs) = worst_diff(&gpu_kl, &cpu_kl);
    for p in 0..n_peers {
        assert!(
            gpu_kl[p].is_finite() && gpu_kl[p] >= 0.0,
            "dml GPU kl[{p}] not finite/non-neg: {}",
            gpu_kl[p]
        );
        assert!(
            close(gpu_kl[p], cpu_kl[p], 5e-3, 1e-6),
            "dml kl[{p}] mismatch: gpu={} cpu={} (worst rel={worst_rel:e} abs={worst_abs:e})",
            gpu_kl[p],
            cpu_kl[p]
        );
    }
}

// ===========================================================================
// 6. crd_score_kernel — CRATE ORACLE (relation::crd cosine_sim, block = 1)
// ===========================================================================
//
// The kernel computes cosine_sim(anchor[b], keys[b]) per batch element and
// writes (not atomically adds) out_scores[b].
//
// LAUNCH CONSTRAINT: block must be (1, 1, 1) so thread 0 accumulates all
// feat_dim dot/norm² products and writes the result. With block_x > 1 each
// thread accumulates its own private register partial, but only thread 0 writes
// — the other threads' contributions are silently lost (no reduction). This is
// a kernel design limitation identical to `dml_loss_kernel`.
//
// The kernel uses ε ≈ 6e-8 in the denominator; the CPU uses EPS = 1e-8.
// For non-degenerate (non-zero) unit-scale vectors the difference is negligible.
//
// `cosine_sim` in `relation::crd` is private; the oracle below re-derives the
// formula directly.

/// CPU cosine similarity, re-derived from crd.rs (mirrors its formula exactly).
fn cosine_sim_oracle(a: &[f32], b: &[f32]) -> f32 {
    const EPS: f32 = 1e-8;
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    dot / (na * nb + EPS)
}

#[test]
fn crd_score_matches_cosine_similarity() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let batch = 8_usize;
    let feat_dim = 16_usize;

    let mut rng = LcgRng::new(0x1D39_78EE);
    // Vectors in [-1, 1]; clearly non-degenerate so the ε difference doesn't matter.
    let anchor: Vec<f32> = (0..batch * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let keys: Vec<f32> = (0..batch * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // CPU reference: cosine_sim per batch element.
    let cpu_scores: Vec<f32> = (0..batch)
        .map(|b| {
            let a = &anchor[b * feat_dim..(b + 1) * feat_dim];
            let k = &keys[b * feat_dim..(b + 1) * feat_dim];
            cosine_sim_oracle(a, k)
        })
        .collect();

    for (i, &s) in cpu_scores.iter().enumerate() {
        assert!(
            (-1.0 - 1e-5..=1.0 + 1e-5).contains(&s),
            "cpu cosine_score[{i}]={s} outside [-1,1]"
        );
    }

    let ptx = crate::ptx_kernels::crd_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "crd_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_anchor = DeviceBuffer::<f32>::from_host(&anchor).expect("d_anchor");
    let d_keys = DeviceBuffer::<f32>::from_host(&keys).expect("d_keys");
    // out_scores: kernel uses st.global (overwrite), not atomic add.
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; batch]).expect("d_out");

    // IMPORTANT: block = (1,1,1) — see LAUNCH CONSTRAINT above.
    let params = LaunchParams::new((batch as u32, 1u32, 1u32), (1u32, 1u32, 1u32));

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_anchor.as_device_ptr(),
                d_keys.as_device_ptr(),
                d_out.as_device_ptr(),
                batch as u32,
                feat_dim as u32,
            ),
        )
        .expect("launch crd_score_kernel");
    stream.synchronize().expect("sync");

    let mut gpu_scores = vec![0.0_f32; batch];
    d_out.copy_to_host(&mut gpu_scores).expect("copy");

    let (worst_rel, worst_abs) = worst_diff(&gpu_scores, &cpu_scores);
    for b in 0..batch {
        // Structural: cosine similarity must lie in [-1, 1].
        assert!(
            (-1.0 - 1e-3..=1.0 + 1e-3).contains(&gpu_scores[b]),
            "crd GPU score[{b}]={} outside [-1,1]",
            gpu_scores[b]
        );
        // Numerical: within 1e-4 relative of CPU oracle.
        assert!(
            close(gpu_scores[b], cpu_scores[b], 1e-4, 1e-6),
            "crd score[{b}] mismatch: gpu={} cpu={} \
             (worst rel={worst_rel:e} abs={worst_abs:e})",
            gpu_scores[b],
            cpu_scores[b]
        );
    }
}

// ===========================================================================
// 7. gram_matrix_kernel — CRATE ORACLE (relation::cc::gram_matrix)
// ===========================================================================
//
// The kernel computes G[j, i] = Σ_k F[k, i] · F[k, j] and writes it to
// gram[j * d + i]. The crate's `gram_matrix` writes G[i, j] = Σ_k F[k,i]·F[k,j]
// to g[i * d + j]. Since G is symmetric (G[i,j] = G[j,i]), the flat arrays
// are element-wise equal: at any flat index `m = a * d + b`,
// GPU stores Σ_k F[k,b]·F[k,a] == CPU stores Σ_k F[k,a]·F[k,b].
//
// The kernel uses a 2-D thread grid, one thread per (i, j) output cell.
// Threads with i ≥ d or j ≥ d are guarded and exit early.

#[test]
fn gram_matrix_matches_cpu_gram() {
    use crate::relation::cc::gram_matrix;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 6_usize; // samples
    let d = 8_usize; // feature dimension

    let mut rng = LcgRng::new(0xC0DE_F00D);
    let feat_flat: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU reference: gram_matrix takes &[Vec<f32>] (rows of F).
    let feat_rows: Vec<Vec<f32>> = (0..n)
        .map(|r| feat_flat[r * d..(r + 1) * d].to_vec())
        .collect();
    let cpu_gram = gram_matrix(&feat_rows);
    assert_eq!(cpu_gram.len(), d * d);

    let ptx = crate::ptx_kernels::gram_matrix_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gram_matrix_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_feat = DeviceBuffer::<f32>::from_host(&feat_flat).expect("d_feat");
    let d_gram = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; d * d]).expect("d_gram");

    // 2-D grid: ((d + 15) / 16, (d + 15) / 16), block (16, 16).
    // Threads with global i ≥ d or j ≥ d are masked by the PTX guards.
    let tile = 16_u32;
    let grid_d = d.div_ceil(tile as usize) as u32;
    let params = LaunchParams::new((grid_d, grid_d), (tile, tile));

    kernel
        .launch(
            &params,
            &stream,
            &(
                d_feat.as_device_ptr(),
                d_gram.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch gram_matrix_kernel");
    stream.synchronize().expect("sync");

    let mut gpu_gram = vec![0.0_f32; d * d];
    d_gram.copy_to_host(&mut gpu_gram).expect("copy");

    let (worst_rel, worst_abs) = worst_diff(&gpu_gram, &cpu_gram);
    for k in 0..d * d {
        assert!(
            close(gpu_gram[k], cpu_gram[k], 1e-4, 1e-6),
            "gram_matrix[{k}] mismatch: gpu={} cpu={} \
             (worst rel={worst_rel:e} abs={worst_abs:e})",
            gpu_gram[k],
            cpu_gram[k]
        );
    }
}
