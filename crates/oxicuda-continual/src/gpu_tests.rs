//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies the results
//! back, and asserts numerical equivalence to a CPU reference. The launch ABI
//! mirrors the proven `oxicuda-snn` / `oxicuda-ot` canaries: device buffers are
//! passed as their `CUdeviceptr` (a `.param .u64`), scalars are passed as the
//! matching Rust scalar (`.param .u32` / `.param .f32` / `.param .u64`), in the
//! kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared against a `pub` CPU function the
//!   kernel mirrors:
//!   `mask_apply_kernel` ↔ [`crate::architecture::packnet::apply_mask`].
//! * **Independent host re-derivation** — the op is fused into a larger CPU
//!   routine (or accumulates atomically into a scalar that no single `pub fn`
//!   returns), so the oracle is an independent Rust re-implementation of the
//!   kernel's *documented* arithmetic, computed independently of the JIT PTX:
//!   `ewc_penalty_kernel` (Σ Fᵢ·(θᵢ−θ*ᵢ)², the un-scaled core of
//!   [`crate::regularization::ewc::ewc_loss`]),
//!   `fisher_diag_kernel` (Fᵢ += gᵢ², the per-element core of
//!   [`crate::regularization::ewc::compute_fisher_empirical`]),
//!   `gradient_project_kernel` (gᵢ −= (g·m / m·m)·mᵢ),
//!   `si_omega_update_kernel` (Ωᵢ += |Δθᵢ·gᵢ|, the per-element core of
//!   [`crate::regularization::si::si_importance_update`]),
//!   `logit_distill_kernel` (Σ exp(z_sᵢ)·(z_sᵢ − z_cᵢ), the kernel's documented
//!   base-2-emulated KL contribution), and `replay_sample_kernel` (the inline
//!   counter-based LCG reservoir-swap index).
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.
//!
//! ## PTX bug-class audit (this crate)
//!
//! * **Base-2 exp/log** — only `logit_distill_kernel` uses `ex2.approx.f32` /
//!   `lg2.approx.f32`. It correctly multiplies the argument by `log2(e)` before
//!   `ex2` and the `lg2` output by `ln(2)`, so it is a true base-e exp/log. The
//!   base-e CPU oracle confirms this on device (a missing conversion would be
//!   ~30–45 % wrong and trip the 2e-3 tolerance).
//! * **Invalid PTX** — all seven kernels JIT-compile on the A4000 (sm_86); the
//!   `load_kernel` step is a hard ptxas gate.
//! * **Wrong math / races** — caught by the element-wise / scalar oracle
//!   comparisons below.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

/// Knuth MMIX 64-bit LCG multiplier, matching the PTX immediates.
const LCG_MUL: u64 = 6_364_136_223_846_793_005;
/// Knuth MMIX 64-bit LCG increment, matching the PTX immediates.
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

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
/// A failure here means ptxas rejected the PTX — a real, hard bug.
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
// 1. ewc_penalty  —  HOST RE-DERIVATION (Σ Fᵢ·(θᵢ − θ*ᵢ)², un-scaled ewc_loss core)
// ===========================================================================

#[test]
fn ewc_penalty_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // All-positive terms (Fᵢ > 0, δᵢ ≠ 0) ⇒ the atomic-accumulated scalar is a
    // sum of positives with no cancellation, so the only divergence from the
    // ordered host sum is FP reordering across the 256 atomic adds (~few×1e-5
    // relative on a sum of order 10²). The 1e-3 relative bound is comfortable yet
    // still flags any gross formula error (e.g. dropping Fᵢ, or wrong δ).
    let n = 256_usize;
    let mut rng = LcgRng::new(0x0EC0_1234);
    let theta: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let theta_star: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let fisher: Vec<f32> = (0..n).map(|_| 0.1 + rng.next_f32()).collect(); // [0.1, 1.1)

    // Host re-derivation: out = Σ Fᵢ·(θᵢ − θ*ᵢ)² (the kernel omits the 0.5·λ that
    // `ewc_loss` applies, so this is its un-scaled core).
    let mut expected = 0.0_f32;
    for i in 0..n {
        let d = theta[i] - theta_star[i];
        expected += fisher[i] * d * d;
    }

    let ptx = crate::ptx_kernels::ewc_penalty_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ewc_penalty_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_theta = DeviceBuffer::<f32>::from_host(&theta).expect("d_theta");
    let d_star = DeviceBuffer::<f32>::from_host(&theta_star).expect("d_star");
    let d_fisher = DeviceBuffer::<f32>::from_host(&fisher).expect("d_fisher");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_theta.as_device_ptr(),
                d_star.as_device_ptr(),
                d_fisher.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch ewc_penalty_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = [0.0_f32];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert!(
        close(out_gpu[0], expected, 1e-3, 1e-3),
        "ewc_penalty mismatch: gpu={} host={}",
        out_gpu[0],
        expected
    );
}

// ===========================================================================
// 2. fisher_diag  —  HOST RE-DERIVATION (Fᵢ += gᵢ², compute_fisher_empirical core)
// ===========================================================================

#[test]
fn fisher_diag_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x0F15_4E12);
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let fisher_init: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

    // Host: each Fᵢ gains exactly gᵢ² (one thread per element, no contention).
    let expected: Vec<f32> = fisher_init
        .iter()
        .zip(&grad)
        .map(|(&f, &g)| f + g * g)
        .collect();

    let ptx = crate::ptx_kernels::fisher_diag_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fisher_diag_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");
    let d_fisher = DeviceBuffer::<f32>::from_host(&fisher_init).expect("d_fisher");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_grad.as_device_ptr(), d_fisher.as_device_ptr(), n as u32),
        )
        .expect("launch fisher_diag_kernel");
    stream.synchronize().expect("sync");

    let mut fisher_gpu = vec![0.0_f32; n];
    d_fisher.copy_to_host(&mut fisher_gpu).expect("copy fisher");

    let (rel, abs) = worst_diff(&fisher_gpu, &expected);
    for k in 0..n {
        assert!(
            close(fisher_gpu[k], expected[k], 1e-5, 1e-6),
            "fisher_diag F[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            fisher_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 3. gradient_project  —  HOST RE-DERIVATION (gᵢ −= (dot_gm/dot_mm)·mᵢ)
// ===========================================================================

#[test]
fn gradient_project_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x006E_9011);
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let mem_grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // The kernel is the projection step *given* pre-computed dot products, so we
    // supply them and re-derive the projection from the identical scalars. The
    // only divergence is the GPU `div.rn` + `mul`/`sub` vs the host `/` + `*`/`-`
    // (≈1 ulp/element).
    let dot_gm: f32 = grad.iter().zip(&mem_grad).map(|(&g, &m)| g * m).sum();
    let dot_mm: f32 = mem_grad.iter().map(|&m| m * m).sum();
    let eps = 1e-12_f32;
    let scale = dot_gm / (dot_mm + eps);
    let expected: Vec<f32> = grad
        .iter()
        .zip(&mem_grad)
        .map(|(&g, &m)| g - scale * m)
        .collect();

    let ptx = crate::ptx_kernels::gradient_project_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gradient_project_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");
    let d_mem = DeviceBuffer::<f32>::from_host(&mem_grad).expect("d_mem");
    let d_dot_gm = DeviceBuffer::<f32>::from_host(&[dot_gm]).expect("d_dot_gm");
    let d_dot_mm = DeviceBuffer::<f32>::from_host(&[dot_mm]).expect("d_dot_mm");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_grad.as_device_ptr(),
                d_mem.as_device_ptr(),
                d_dot_gm.as_device_ptr(),
                d_dot_mm.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch gradient_project_kernel");
    stream.synchronize().expect("sync");

    let mut grad_gpu = vec![0.0_f32; n];
    d_grad.copy_to_host(&mut grad_gpu).expect("copy grad");

    let (rel, abs) = worst_diff(&grad_gpu, &expected);
    for k in 0..n {
        assert!(
            close(grad_gpu[k], expected[k], 1e-4, 1e-5),
            "gradient_project g[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            grad_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 4. mask_apply  —  CRATE ORACLE (architecture::packnet::apply_mask)
// ===========================================================================

#[test]
fn mask_apply_matches_cpu() {
    use crate::architecture::packnet::{PackNetMask, apply_mask};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x00A5_C0DE);
    let weights: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    // Mask bytes: 0 (drop) or 1 (keep), roughly half each.
    let mask: Vec<u8> = (0..n)
        .map(|_| if rng.next_f32() < 0.5 { 0 } else { 1 })
        .collect();

    // ---- CPU reference ----
    let mut w_cpu = weights.clone();
    let packnet_mask = PackNetMask {
        mask: mask.clone(),
        task_id: 0,
        sparsity: 0.5,
    };
    apply_mask(&mut w_cpu, &packnet_mask).expect("apply_mask");

    let ptx = crate::ptx_kernels::mask_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mask_apply_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_mask = DeviceBuffer::<u8>::from_host(&mask).expect("d_mask");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_w.as_device_ptr(), d_mask.as_device_ptr(), n as u32),
        )
        .expect("launch mask_apply_kernel");
    stream.synchronize().expect("sync");

    let mut w_gpu = vec![0.0_f32; n];
    d_w.copy_to_host(&mut w_gpu).expect("copy w");

    // The kernel multiplies by exactly 1.0 (kept) or 0.0 (dropped). For kept
    // weights `w·1.0` is bit-identical to the input. For dropped weights `w·0.0`
    // is numerically zero but IEEE yields `-0.0` for a negative `w` (vs the CPU's
    // `*w = 0.0` which is `+0.0`); both represent zero, so we compare by value
    // there (`-0.0 == 0.0` is true). `w_cpu` (the crate oracle) equals the input
    // for kept entries and `+0.0` for dropped entries, matching both branches.
    for k in 0..n {
        if mask[k] != 0 {
            assert_eq!(
                w_gpu[k].to_bits(),
                weights[k].to_bits(),
                "mask_apply kept w[{k}] altered: gpu={} input={} cpu={}",
                w_gpu[k],
                weights[k],
                w_cpu[k]
            );
        } else {
            assert_eq!(
                w_gpu[k], 0.0_f32,
                "mask_apply dropped w[{k}] not zero: gpu={} (cpu={})",
                w_gpu[k], w_cpu[k]
            );
        }
    }
}

// ===========================================================================
// 5. si_omega_update  —  HOST RE-DERIVATION (Ωᵢ += |Δθᵢ·gᵢ|, si_importance_update core)
// ===========================================================================

#[test]
fn si_omega_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0x0510_4E6A);
    let delta_theta: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let grad: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let omega_init: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

    // Host: exactly the per-element accumulation inside `si_importance_update`.
    let expected: Vec<f32> = (0..n)
        .map(|i| omega_init[i] + (delta_theta[i] * grad[i]).abs())
        .collect();

    let ptx = crate::ptx_kernels::si_omega_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "si_omega_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_delta = DeviceBuffer::<f32>::from_host(&delta_theta).expect("d_delta");
    let d_grad = DeviceBuffer::<f32>::from_host(&grad).expect("d_grad");
    let d_omega = DeviceBuffer::<f32>::from_host(&omega_init).expect("d_omega");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_delta.as_device_ptr(),
                d_grad.as_device_ptr(),
                d_omega.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch si_omega_update_kernel");
    stream.synchronize().expect("sync");

    let mut omega_gpu = vec![0.0_f32; n];
    d_omega.copy_to_host(&mut omega_gpu).expect("copy omega");

    let (rel, abs) = worst_diff(&omega_gpu, &expected);
    for k in 0..n {
        assert!(
            close(omega_gpu[k], expected[k], 1e-5, 1e-6),
            "si_omega Ω[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            omega_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 6. logit_distill  —  HOST RE-DERIVATION (Σ exp(z_sᵢ)·(z_sᵢ − z_cᵢ), base-e)
// ===========================================================================

#[test]
fn logit_distill_matches_host_base_e() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // The kernel computes, per class, exp(z_s)·(ln exp(z_s) − ln exp(z_c)) using
    // `ex2.approx`/`lg2.approx` *with* the log2(e) / ln(2) base conversions, i.e.
    // base-e exp/log. Mathematically that is exp(z_sᵢ)·(z_sᵢ − z_cᵢ), summed
    // atomically into the scalar `p_kl_out`.
    //
    // Conditioning: we force z_sᵢ > z_cᵢ (positive gap) so every term is
    // positive — the scalar is a clean sum of positives, and a *missing* base
    // conversion (the classic base-2 bug) would make exp/log ~1.44× off and
    // shift the sum by tens of percent, far outside the 2e-3 tolerance.
    let n = 64_usize;
    let mut rng = LcgRng::new(0x0D15_7111_u64);
    let z_current: Vec<f32> = (0..n).map(|_| rng.next_f32() * 1.5 - 0.75).collect();
    // z_stored = z_current + gap, gap ∈ [0.2, 1.2): keeps each (z_s − z_c) > 0.
    let z_stored: Vec<f32> = z_current
        .iter()
        .map(|&zc| zc + 0.2 + rng.next_f32())
        .collect();

    // Host re-derivation (base-e), in index order.
    let mut expected = 0.0_f32;
    for i in 0..n {
        expected += z_stored[i].exp() * (z_stored[i] - z_current[i]);
    }

    let ptx = crate::ptx_kernels::logit_distill_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "logit_distill_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_stored = DeviceBuffer::<f32>::from_host(&z_stored).expect("d_stored");
    let d_current = DeviceBuffer::<f32>::from_host(&z_current).expect("d_current");
    let d_kl = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_kl");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_stored.as_device_ptr(),
                d_current.as_device_ptr(),
                d_kl.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch logit_distill_kernel");
    stream.synchronize().expect("sync");

    let mut kl_gpu = [0.0_f32];
    d_kl.copy_to_host(&mut kl_gpu).expect("copy kl");

    assert!(
        close(kl_gpu[0], expected, 2e-3, 1e-3),
        "logit_distill (base-e) mismatch: gpu={} host={} \
         (a missing log2(e)/ln(2) conversion would be ~30–45% off)",
        kl_gpu[0],
        expected
    );
}

// ===========================================================================
// 7. replay_sample  —  HOST RE-DERIVATION (inline counter LCG reservoir swap)
// ===========================================================================

/// Independent re-derivation of `replay_sample_kernel`'s documented arithmetic:
/// `state = (seed ⊕ n_seen)·M + A`, `rand = (state >> 33) as u32`,
/// `r = rand % (n_seen + 1)`, swap index `= r if r < capacity else 0xFFFF_FFFF`.
fn replay_swap_index(seed: u64, n_seen: u32, capacity: u32) -> u32 {
    let state0 = seed ^ u64::from(n_seen);
    let state = state0.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    let rand = (state >> 33) as u32;
    let r = rand % (n_seen + 1);
    if r < capacity { r } else { 0xFFFF_FFFF }
}

#[test]
fn replay_sample_matches_host_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let ptx = crate::ptx_kernels::replay_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "replay_sample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // Several (n_seen, capacity, seed) triples: one within capacity, one likely
    // beyond it, one tiny. All integer math ⇒ bit-exact against the host.
    let cases: &[(u32, u32, u64)] = &[
        (1000, 256, 0x1234_5678_9ABC_DEF0),
        (5, 256, 0xDEAD_BEEF_CAFE_F00D),
        (0, 1, 0x0000_0000_0000_0001),
        (100_000, 64, 0xA5A5_5A5A_F0F0_0F0F),
    ];

    for &(n_seen, capacity, seed) in cases {
        let expected = replay_swap_index(seed, n_seen, capacity);

        // Sentinel != any plausible output so a no-write would be caught.
        let d_idx = DeviceBuffer::<u32>::from_host(&[0x7777_7777_u32]).expect("d_idx");

        // Single thread (kernel guards tid.x == 0).
        let params = LaunchParams::new(1_u32, 1_u32);
        kernel
            .launch(
                &params,
                &stream,
                &(d_idx.as_device_ptr(), n_seen, capacity, seed),
            )
            .expect("launch replay_sample_kernel");
        stream.synchronize().expect("sync");

        let mut idx_gpu = [0_u32];
        d_idx.copy_to_host(&mut idx_gpu).expect("copy idx");

        assert_eq!(
            idx_gpu[0], expected,
            "replay_sample swap idx mismatch (n_seen={n_seen}, cap={capacity}, \
             seed={seed:#x}): gpu={} host={}",
            idx_gpu[0], expected
        );
    }
}
