//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's CPU reference. The launch ABI mirrors the `oxicuda-snn` /
//! `oxicuda-recsys` harnesses: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .u32` /
//! `.param .f32`), in declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to the `pub`
//!   CPU function the kernel mirrors:
//!   - `bt_reward_loss_kernel`  → `preference::bradley_terry::bt_reward_loss`
//!     (× n, since the kernel atomically *sums* while the CPU fn averages).
//!   - `dpo_loss_kernel`        → `dpo::dpo::dpo_loss`     (× n).
//!   - `ipo_loss_kernel`        → `dpo::ipo::ipo_loss`     (× n).
//!   - `kto_loss_kernel`        → `dpo::kto::kto_loss`     (λ = 1, × n; one-sided).
//!   - `orpo_odds_kernel`       → `orpo::orpo::log_odds`   (element-wise).
//! * **Independent host re-derivation** — the kernel has no single dedicated
//!   crate function:
//!   - `rlhf_kl_kernel`  computes the per-token forward-KL contribution
//!     `exp(lp)·(lp − ref_lp)`, which differs from `kl_divergence_from_logps`
//!     (a mean of `lp − ref_lp`). Oracle is an independent host re-implementation.
//!   - `sft_mask_kernel` writes only the negated label-logit (the host adds the
//!     log-sum-exp denominator). Oracle re-derives that exact partial.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.
//!
//! ## PTX bugs found and fixed (see `ptx_kernels.rs`)
//!
//! 1. `bt_reward_loss_kernel` — BASE-2 exp: `ex2.approx.f32` was applied to the
//!    raw `-diff` (computing `2^-diff`, i.e. `ln(1 + 2^-diff)`) instead of the
//!    natural `exp(-diff)`. Fixed by scaling the argument by `log2(e)` first.
//! 2. `dpo_loss_kernel` — (a) the same BASE-2 exp error on `-logit`; (b) the
//!    `atom.global.add.f32` wrote its returned old value into `%f0`, which holds
//!    `beta` and is reused every grid-stride iteration — corrupting the loss for
//!    any thread that processes more than one element. Fixed both.
//! 3. `kto_loss_kernel` — the same BASE-2 exp error on `-arg`. Fixed.
//!
//! `ipo_loss_kernel`, `orpo_odds_kernel`, `rlhf_kl_kernel` and `sft_mask_kernel`
//! were already correct (the last three already scale by `log2(e)` / `ln(2)`),
//! and pass their oracle comparisons unchanged.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
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
///
/// Non-finite operands always fail: an `inf`/`NaN` GPU result must never slip
/// through because `inf <= rel*inf + abs` is vacuously true. (The DPO
/// beta-register clobber bug, for instance, drives the accumulator to `inf`.)
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug.
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
// 1. bt_reward_loss  —  CRATE ORACLE (preference::bradley_terry::bt_reward_loss)
// ===========================================================================
//
// Kernel: out[0] += Σ_i softplus(-(chosen_i − rejected_i))  (atomic sum).
// The CPU `bt_reward_loss` returns the MEAN of `-ln σ(rw − rl)`, so the oracle
// for the summed device output is `bt_reward_loss(..) × n`.
//
// A deliberate grid-stride config (block 64, grid 2 over n = 200) makes each of
// the first 72 threads process two elements, exercising the loop body more than
// once. The base-2 bug (now fixed) skewed every term by ~20-50%, far beyond the
// tolerance below.

#[test]
fn bt_reward_loss_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 200_usize;
    let mut rng = LcgRng::new(0x00B7_1055);
    // Rewards in [-2, 2): the diff spans [-4, 4), so softplus(-diff) covers the
    // whole curve (including the ln(2) crossing where base-2 and base-e agree).
    let chosen: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let rejected: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // ---- CPU reference (crate oracle, summed) ----
    let mean = crate::preference::bradley_terry::bt_reward_loss(&chosen, &rejected)
        .expect("cpu bt_reward_loss");
    let expected_sum = mean * n as f32;

    // ---- GPU ----
    let ptx = crate::ptx_kernels::bt_reward_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bt_reward_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_chosen = DeviceBuffer::<f32>::from_host(&chosen).expect("d_chosen");
    let d_rejected = DeviceBuffer::<f32>::from_host(&rejected).expect("d_rejected");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let params = LaunchParams::new(2_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_chosen.as_device_ptr(),
                d_rejected.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch bt_reward_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out = [0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert!(
        close(out[0], expected_sum, 2e-3, 1e-2),
        "bt_reward_loss sum mismatch: gpu={} cpu={} (base-2 exp bug would show ~20-50% error)",
        out[0],
        expected_sum
    );
}

// ===========================================================================
// 2. dpo_loss  —  CRATE ORACLE (dpo::dpo::dpo_loss), grid-stride multi-iter
// ===========================================================================
//
// Kernel: out[0] += Σ_i -log σ(β·Δ_i), Δ = (clp−rclp) − (rlp−rrlp)  (atomic sum).
// CPU `dpo_loss` returns the mean → oracle is `dpo_loss × n`.
//
// This test is the one that would catch BOTH dpo bugs: the base-2 exp error and
// the `%f0`/beta clobber by the atomic's returned old value (which only bites
// when a thread iterates more than once — hence the grid-stride config).

#[test]
fn dpo_loss_matches_cpu() {
    use crate::dpo::dpo::{DpoConfig, dpo_loss};
    use crate::preference::pair::PairBatch;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 200_usize;
    // beta = 1.0 widens the logit range to ~[-4, 4] so the per-term base-2 vs
    // base-e gap is ~20-50% — large enough that this test independently catches
    // a base-2 exp regression as well as the beta-clobber (a small beta would
    // hide the former by squashing every logit toward 0).
    let beta = 1.0_f32;
    let mut rng = LcgRng::new(0x0D90_10AA);
    let clp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 2.0).collect();
    let rclp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 2.0).collect();
    let rlp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 2.0).collect();
    let rrlp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 2.0).collect();

    let batch =
        PairBatch::new(clp.clone(), rlp.clone(), rclp.clone(), rrlp.clone()).expect("pair batch");
    let mean = dpo_loss(&batch, &DpoConfig { beta }).expect("cpu dpo_loss");
    let expected_sum = mean * n as f32;

    let ptx = crate::ptx_kernels::dpo_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dpo_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_clp = DeviceBuffer::<f32>::from_host(&clp).expect("d_clp");
    let d_rclp = DeviceBuffer::<f32>::from_host(&rclp).expect("d_rclp");
    let d_rlp = DeviceBuffer::<f32>::from_host(&rlp).expect("d_rlp");
    let d_rrlp = DeviceBuffer::<f32>::from_host(&rrlp).expect("d_rrlp");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    // Grid-stride: 128 threads over 200 elements → 72 threads do 2 iterations,
    // exercising the (now-fixed) beta-register clobber.
    let params = LaunchParams::new(2_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_clp.as_device_ptr(),
                d_rclp.as_device_ptr(),
                d_rlp.as_device_ptr(),
                d_rrlp.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                beta,
            ),
        )
        .expect("launch dpo_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out = [0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert!(
        close(out[0], expected_sum, 2e-3, 1e-2),
        "dpo_loss sum mismatch: gpu={} cpu={} (base-2 exp or beta-clobber bug)",
        out[0],
        expected_sum
    );
}

// ===========================================================================
// 3. ipo_loss  —  CRATE ORACLE (dpo::ipo::ipo_loss)
// ===========================================================================
//
// Kernel: out[0] += Σ_i (h_i − τ)², h = (clp−rclp) − (rlp−rrlp), τ = 1/(2β).
// CPU `ipo_loss` returns the mean → oracle is `ipo_loss × n`. No exp/log: this
// kernel is a pure squared-error and was already correct.

#[test]
fn ipo_loss_matches_cpu() {
    use crate::dpo::ipo::{IpoConfig, ipo_loss};
    use crate::preference::pair::PairBatch;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 200_usize;
    let beta = 0.1_f32;
    let mut rng = LcgRng::new(0x01B0_5151);
    let clp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let rclp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let rlp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let rrlp: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    let batch =
        PairBatch::new(clp.clone(), rlp.clone(), rclp.clone(), rrlp.clone()).expect("pair batch");
    let mean = ipo_loss(&batch, &IpoConfig { beta }).expect("cpu ipo_loss");
    let expected_sum = mean * n as f32;

    let ptx = crate::ptx_kernels::ipo_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ipo_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_clp = DeviceBuffer::<f32>::from_host(&clp).expect("d_clp");
    let d_rclp = DeviceBuffer::<f32>::from_host(&rclp).expect("d_rclp");
    let d_rlp = DeviceBuffer::<f32>::from_host(&rlp).expect("d_rlp");
    let d_rrlp = DeviceBuffer::<f32>::from_host(&rrlp).expect("d_rrlp");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let params = LaunchParams::new(2_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_clp.as_device_ptr(),
                d_rclp.as_device_ptr(),
                d_rlp.as_device_ptr(),
                d_rrlp.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                beta,
            ),
        )
        .expect("launch ipo_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out = [0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert!(
        close(out[0], expected_sum, 2e-3, 1e-2),
        "ipo_loss sum mismatch: gpu={} cpu={}",
        out[0],
        expected_sum
    );
}

// ===========================================================================
// 4. kto_loss  —  CRATE ORACLE (dpo::kto::kto_loss), both desirable polarities
// ===========================================================================
//
// Kernel: out[0] += Σ_i (1 − σ(β·arg_i)), arg = (r − z0) if desirable else
// (z0 − r), z0 = ln 2.  The CPU `kto_loss` with one side empty and λ = 1 returns
// exactly the mean of those per-element terms → oracle is `kto_loss × n`.

fn run_kto_case(fx: &GpuFixture, desirable: bool) {
    use crate::dpo::kto::{KtoConfig, kto_loss};

    let n = 200_usize;
    let beta = 0.5_f32;
    let mut rng = LcgRng::new(0x0470_u64 ^ u64::from(desirable));
    let rewards: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Crate oracle: route all rewards through the matching side with λ = 1.
    let (desirable_rewards, undesirable_rewards, cfg) = if desirable {
        (
            rewards.clone(),
            Vec::new(),
            KtoConfig {
                beta,
                lambda_d: 1.0,
                lambda_u: 0.0,
            },
        )
    } else {
        (
            Vec::new(),
            rewards.clone(),
            KtoConfig {
                beta,
                lambda_d: 0.0,
                lambda_u: 1.0,
            },
        )
    };
    let mean = kto_loss(&desirable_rewards, &undesirable_rewards, &cfg).expect("cpu kto_loss");
    let expected_sum = mean * n as f32;

    let ptx = crate::ptx_kernels::kto_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "kto_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_rewards = DeviceBuffer::<f32>::from_host(&rewards).expect("d_rewards");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let params = LaunchParams::new(2_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_rewards.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                beta,
                u32::from(desirable),
            ),
        )
        .expect("launch kto_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out = [0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert!(
        close(out[0], expected_sum, 2e-3, 1e-2),
        "kto_loss (desirable={desirable}) sum mismatch: gpu={} cpu={} (base-2 exp bug)",
        out[0],
        expected_sum
    );
}

#[test]
fn kto_loss_desirable_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_kto_case(&fx, true);
}

#[test]
fn kto_loss_undesirable_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_kto_case(&fx, false);
}

// ===========================================================================
// 5. orpo_odds  —  CRATE ORACLE (orpo::orpo::log_odds), element-wise
// ===========================================================================
//
// Kernel: out[i] = ln( p / (1 − p + 1e-7) ), p = exp(lp).  This already scales
// by log2(e) before `ex2` and by ln 2 after `lg2`, so it was correct.  Inputs
// are chosen in (-4, -0.3] so the CPU `log_odds` clamp/`max(1e-7)` guards never
// activate, making the comparison exact (modulo `ex2`/`lg2.approx` ~2 ulp).

#[test]
fn orpo_odds_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let mut rng = LcgRng::new(0x0000_0DD5);
    // lp in (-4, -0.3): p in (0.018, 0.74), odds well above 1e-7, no clamp.
    let logps: Vec<f32> = (0..n).map(|_| -0.3 - 3.7 * rng.next_f32()).collect();

    let expected: Vec<f32> = logps
        .iter()
        .map(|&lp| crate::orpo::orpo::log_odds(lp))
        .collect();

    let ptx = crate::ptx_kernels::orpo_odds_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "orpo_odds_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logps = DeviceBuffer::<f32>::from_host(&logps).expect("d_logps");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let params = LaunchParams::new(2_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_logps.as_device_ptr(), d_out.as_device_ptr(), n as u32),
        )
        .expect("launch orpo_odds_kernel");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy out");

    let (rel, abs) = worst_diff(&out, &expected);
    for k in 0..n {
        assert!(
            close(out[k], expected[k], 2e-3, 2e-3),
            "orpo_odds[{k}] mismatch: gpu={} cpu={} lp={} (worst rel={rel:e} abs={abs:e})",
            out[k],
            expected[k],
            logps[k]
        );
    }
}

// ===========================================================================
// 6. rlhf_kl  —  INDEPENDENT HOST RE-DERIVATION (forward-KL contribution)
// ===========================================================================
//
// Kernel: out[i] = exp(lp_i) · (lp_i − ref_lp_i).  This already scales by
// log2(e) before `ex2`, so it was correct.  The crate's
// `kl_divergence_from_logps` computes a DIFFERENT quantity (mean of lp − ref_lp),
// so the oracle here is an independent host re-implementation of the kernel's
// documented per-token arithmetic.

#[test]
fn rlhf_kl_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let mut rng = LcgRng::new(0x4C10);
    let logps: Vec<f32> = (0..n).map(|_| -3.0 * rng.next_f32()).collect();
    let ref_logps: Vec<f32> = (0..n).map(|_| -3.0 * rng.next_f32()).collect();

    let expected: Vec<f32> = logps
        .iter()
        .zip(ref_logps.iter())
        .map(|(&lp, &rlp)| lp.exp() * (lp - rlp))
        .collect();

    let ptx = crate::ptx_kernels::rlhf_kl_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rlhf_kl_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_lp = DeviceBuffer::<f32>::from_host(&logps).expect("d_lp");
    let d_rlp = DeviceBuffer::<f32>::from_host(&ref_logps).expect("d_rlp");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let params = LaunchParams::new(2_u32, 64_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_lp.as_device_ptr(),
                d_rlp.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch rlhf_kl_kernel");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out).expect("copy out");

    let (rel, abs) = worst_diff(&out, &expected);
    for k in 0..n {
        assert!(
            close(out[k], expected[k], 1e-3, 1e-4),
            "rlhf_kl[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 7. sft_mask  —  INDEPENDENT HOST RE-DERIVATION (masked negated label-logit)
// ===========================================================================
//
// Kernel: for each token with mask != 0, out[token] = -logits[token·V + label];
// masked-out tokens (mask == 0) leave out[token] at its initial value.  The host
// then adds the log-sum-exp denominator (`sft::loss::masked_token_ce`); here we
// validate the device half exactly (negation is bit-exact).

#[test]
fn sft_mask_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 12_usize;
    let n_vocab = 7_usize;
    let mut rng = LcgRng::new(0x05F7_1A5C);

    let logits: Vec<f32> = (0..n_tokens * n_vocab)
        .map(|_| rng.next_f32() * 6.0 - 3.0)
        .collect();
    let labels: Vec<u32> = (0..n_tokens)
        .map(|_| (rng.next_f32() * n_vocab as f32) as u32 % n_vocab as u32)
        .collect();
    // Deterministic mix of masked / unmasked tokens (and at least one of each).
    let mask: Vec<u8> = (0..n_tokens)
        .map(|t| {
            if t == 0 {
                1
            } else {
                u8::from(rng.next_f32() < 0.5)
            }
        })
        .collect();

    // Host reference: masked tokens get -label_logit, others stay zero.
    let mut expected = vec![0.0_f32; n_tokens];
    for t in 0..n_tokens {
        if mask[t] != 0 {
            let off = t * n_vocab + labels[t] as usize;
            expected[t] = -logits[off];
        }
    }

    let ptx = crate::ptx_kernels::sft_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "sft_mask_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_labels = DeviceBuffer::<u32>::from_host(&labels).expect("d_labels");
    let d_mask = DeviceBuffer::<u8>::from_host(&mask).expect("d_mask");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_labels.as_device_ptr(),
                d_mask.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                n_vocab as u32,
            ),
        )
        .expect("launch sft_mask_kernel");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n_tokens];
    d_out.copy_to_host(&mut out).expect("copy out");

    // Negation of a loaded float is exact, so this is a bit-exact comparison.
    for t in 0..n_tokens {
        assert_eq!(
            out[t].to_bits(),
            expected[t].to_bits(),
            "sft_mask out[{t}] mismatch: gpu={} host={} (mask={}, label={})",
            out[t],
            expected[t],
            mask[t],
            labels[t]
        );
    }
}
