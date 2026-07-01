//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to an independent CPU oracle. The launch ABI mirrors the working
//! `oxicuda-snn` / `oxicuda-ot` canaries: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .u32` / `.param .f32` / `.param .u64`), in the kernel's declared
//! parameter order.
//!
//! ## Kernels covered (all 7 are launchable on sm_86, all numerically validated)
//!
//! | Kernel | Oracle |
//! |--------|--------|
//! | `kl_gaussian_kernel`        | independent host KL(N(μ,σ²)‖N(0,1)) summed |
//! | `mc_dropout_mask_kernel`    | bit-exact host re-derivation of the inline LCG |
//! | `local_reparam_kernel`      | host Box-Muller (base-e) over the inline LCG |
//! | `ece_bucket_kernel`         | independent host histogram binning |
//! | `ensemble_aggregate_kernel` | host mean + Bessel-corrected variance |
//! | `flipout_perturb_kernel`    | host sign-perturbation with LCG `s_j` |
//! | `temp_scale_logits_kernel`  | host `logit / T` |
//!
//! ## Base-2 exp/log audit (honest)
//!
//! Two kernels use `ex2.approx.f32` / `lg2.approx.f32`: `kl_gaussian` (exp) and
//! `local_reparam` (exp + log). Both already apply the correct base conversion
//! (multiply the exponent by `log2(e)` before `ex2`; multiply the `lg2` result
//! by `ln(2)`), so the independent **base-e** CPU oracles below confirm the
//! conversions are present — a missing factor would show up as a ~20–45 %
//! discrepancy, orders of magnitude outside the tolerances used here.
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a **real bug**
/// in `ptx_kernels.rs`, surfaced here as a panic rather than a skip.
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

/// Knuth MMIX 64-bit LCG constants used by every inline-RNG kernel in this
/// crate (`mc_dropout_mask`, `local_reparam`, `flipout_perturb`).
const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

/// Reproduce the kernels' `u32 → f32 in [0,1)` step exactly: take the high 31
/// bits of the advanced LCG state (`>> 33`) and divide by `2^31`. Both the
/// `cvt.rn.f32.u32` and the `div.rn.f32` are IEEE round-to-nearest-even, so the
/// host result is bit-identical to the device's.
fn lcg_unit_from_state(state: u64) -> f32 {
    let r = (state >> 33) as u32;
    (r as f32) / 2_147_483_648.0_f32 // 2^31 == 0F4F000000
}

// ===========================================================================
// 1. kl_gaussian  —  INDEPENDENT HOST RE-DERIVATION (atomic-summed KL)
// ===========================================================================

#[test]
fn kl_gaussian_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Moderate ranges keep σ² = exp(2·log_σ) ∈ ~[0.37, 2.7] and every per-element
    // KL contribution O(1), so the `ex2.approx.f32` exponential stays well inside
    // its accurate domain and the atomic-summed total is O(n).
    let n = 256_usize;
    let mut rng = LcgRng::new(0x0B_A4E5);
    let mu: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let log_sigma: Vec<f32> = (0..n).map(|_| rng.next_f32() - 0.5).collect();

    // Host oracle: contrib_i = 0.5·(μ² + exp(2·log_σ) − 1 − 2·log_σ), summed.
    // (σ² = exp(2·log_σ); ln(σ²) = 2·log_σ.) Per-element math in f32 to match the
    // kernel; accumulation in f64 for an accurate reference total.
    let mut kl_host = 0.0_f64;
    for i in 0..n {
        let sigma_sq = (2.0_f32 * log_sigma[i]).exp();
        let contrib = 0.5_f32 * (mu[i] * mu[i] + sigma_sq - 1.0_f32 - 2.0_f32 * log_sigma[i]);
        kl_host += f64::from(contrib);
    }
    let kl_host = kl_host as f32;

    let ptx = crate::ptx_kernels::kl_gaussian_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "kl_gaussian_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mu = DeviceBuffer::<f32>::from_host(&mu).expect("d_mu");
    let d_log_sigma = DeviceBuffer::<f32>::from_host(&log_sigma).expect("d_log_sigma");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mu.as_device_ptr(),
                d_log_sigma.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch kl_gaussian_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Tolerance: ex2.approx (~2 ulp) per element plus nondeterministic atomic-add
    // ordering over 256 terms — a few-ulp relative drift (~1e-5). 1e-3 relative
    // is comfortable yet still flags a gross formula / base-2 error (which would
    // be tens of percent).
    assert!(
        close(out_gpu[0], kl_host, 1e-3, 1e-3),
        "kl_gaussian total mismatch: gpu={} host={}",
        out_gpu[0],
        kl_host
    );
}

// ===========================================================================
// 2. mc_dropout_mask  —  BIT-EXACT HOST RE-DERIVATION of the inline LCG
// ===========================================================================

#[test]
fn mc_dropout_mask_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let drop_rate = 0.3_f32;
    let keep_rate = 1.0_f32 - drop_rate;
    let scale = 1.0_f32 / keep_rate;
    let seed = 0x1357_9BDF_2468_ACE0_u64;

    // Host re-derivation of the kernel's per-element pipeline:
    //   state = (seed ^ i)·M + A ;  u = (state>>33)/2^31 ;
    //   mask = (u > drop_rate) ? 1/keep : 0.
    // Every op is integer or IEEE round-to-nearest f32, so the result is
    // bit-identical to the device's.
    let mut mask_host = vec![0.0_f32; n];
    for (i, slot) in mask_host.iter_mut().enumerate() {
        let state = (seed ^ (i as u64))
            .wrapping_mul(LCG_MUL)
            .wrapping_add(LCG_ADD);
        let u = lcg_unit_from_state(state);
        *slot = if u > drop_rate { scale } else { 0.0 };
    }

    let ptx = crate::ptx_kernels::mc_dropout_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mc_dropout_mask_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mask = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_mask");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_mask.as_device_ptr(), n as u32, drop_rate, seed),
        )
        .expect("launch mc_dropout_mask_kernel");
    stream.synchronize().expect("sync");

    let mut mask_gpu = vec![0.0_f32; n];
    d_mask.copy_to_host(&mut mask_gpu).expect("copy mask");

    for k in 0..n {
        // Structural: each entry is exactly 0 or the keep-scale.
        assert!(
            mask_gpu[k] == 0.0 || mask_gpu[k].to_bits() == scale.to_bits(),
            "mc_dropout mask[{k}] = {} is neither 0 nor 1/keep ({scale})",
            mask_gpu[k]
        );
        // Bit-exact vs the independent host LCG.
        assert_eq!(
            mask_gpu[k].to_bits(),
            mask_host[k].to_bits(),
            "mc_dropout mask[{k}] mismatch: gpu={} host={}",
            mask_gpu[k],
            mask_host[k]
        );
    }
}

// ===========================================================================
// 3. local_reparam  —  HOST BOX-MULLER (base-e) over the inline LCG
// ===========================================================================

#[test]
fn local_reparam_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let seed1 = 0x2468_ACE0_u32;
    let seed2 = 0x1357_9BDF_u32;
    let eps_floor = 1e-6_f32;

    let mut rng = LcgRng::new(0x10C0_DEAD);
    // W_log_var ∈ [-1, 1] ⇒ exp(W_log_var) ∈ [0.37, 2.7]; x ∈ [0.5, 1.5];
    // W_mu ∈ [-1, 1]. Keeps act_var moderate and z well-scaled.
    let w_mu: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let w_log_var: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let x: Vec<f32> = (0..n).map(|_| 0.5 + rng.next_f32()).collect();

    // Host re-derivation. u1, u2 are bit-exact (integer LCG + exact f32 scale);
    // eps uses base-e libm ln/sqrt/cos. The kernel uses lg2.approx·ln2, sqrt.approx
    // and cos.approx (all ~1–2 ulp), and ex2.approx·log2(e) for exp — so this is
    // an INDEPENDENT base-e oracle that would diverge tens of percent if any base
    // conversion were missing.
    let mut z_host = vec![0.0_f32; n];
    for i in 0..n {
        let s1 = ((seed1 as u64) ^ (i as u64))
            .wrapping_mul(LCG_MUL)
            .wrapping_add(LCG_ADD);
        let s2 = ((seed2 as u64) ^ (i as u64))
            .wrapping_mul(LCG_MUL)
            .wrapping_add(LCG_ADD);
        let mut u1 = lcg_unit_from_state(s1);
        u1 = u1.max(eps_floor).min(1.0_f32 - eps_floor);
        let u2 = lcg_unit_from_state(s2);

        let eps = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();

        let act_mu = w_mu[i] * x[i];
        let act_var = w_log_var[i].exp() * x[i] * x[i];
        z_host[i] = act_mu + act_var.sqrt() * eps;
    }

    let ptx = crate::ptx_kernels::local_reparam_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "local_reparam_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w_mu = DeviceBuffer::<f32>::from_host(&w_mu).expect("d_w_mu");
    let d_w_log_var = DeviceBuffer::<f32>::from_host(&w_log_var).expect("d_w_log_var");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_z = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_z");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w_mu.as_device_ptr(),
                d_w_log_var.as_device_ptr(),
                d_x.as_device_ptr(),
                d_z.as_device_ptr(),
                n as u32,
                seed1,
                seed2,
            ),
        )
        .expect("launch local_reparam_kernel");
    stream.synchronize().expect("sync");

    let mut z_gpu = vec![0.0_f32; n];
    d_z.copy_to_host(&mut z_gpu).expect("copy z");

    // Tolerance: the SFU approximations (ex2/lg2/sqrt/cos) carry ~1–2 ulp each, so
    // z diverges by only a few ulp (~1e-5) from the libm oracle; 1e-3 relative
    // with a 5e-4 absolute floor (for the cos≈0 nodes) is comfortable yet still
    // catches any missing base conversion (≥20 % on the sampled term).
    let (rel, abs) = worst_diff(&z_gpu, &z_host);
    for k in 0..n {
        assert!(
            close(z_gpu[k], z_host[k], 1e-3, 5e-4),
            "local_reparam z[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            z_gpu[k],
            z_host[k]
        );
    }
}

// ===========================================================================
// 4. ece_bucket  —  INDEPENDENT HOST HISTOGRAM BINNING (atomic counters)
// ===========================================================================

#[test]
fn ece_bucket_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let n_bins = 4_usize;

    let mut rng = LcgRng::new(0xECE0_B17C);
    // Confidences in [0, 1); random f32 are essentially never exactly on a bin
    // edge (k/n_bins), so the floor binning is unambiguous and matches the host.
    let confidence: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    let correct: Vec<u32> = (0..n)
        .map(|_| if rng.next_f32() < 0.5 { 1 } else { 0 })
        .collect();

    // Host oracle: bin = min(floor(conf·n_bins), n_bins-1). f32·f32 then truncate
    // toward zero matches the kernel's `cvt.rzi.u32.f32` exactly for conf ≥ 0.
    let mut count_host = vec![0_u32; n_bins];
    let mut sum_conf_host = vec![0.0_f32; n_bins];
    let mut sum_correct_host = vec![0.0_f32; n_bins];
    for i in 0..n {
        let raw = (confidence[i] * n_bins as f32) as u32;
        let bin = (raw as usize).min(n_bins - 1);
        count_host[bin] += 1;
        sum_conf_host[bin] += confidence[i];
        sum_correct_host[bin] += if correct[i] != 0 { 1.0 } else { 0.0 };
    }

    let ptx = crate::ptx_kernels::ece_bucket_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ece_bucket_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_conf = DeviceBuffer::<f32>::from_host(&confidence).expect("d_conf");
    let d_correct = DeviceBuffer::<u32>::from_host(&correct).expect("d_correct");
    let d_count = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_bins]).expect("d_count");
    let d_sum_conf = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_bins]).expect("d_sum_conf");
    let d_sum_correct =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_bins]).expect("d_sum_correct");

    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_conf.as_device_ptr(),
                d_correct.as_device_ptr(),
                d_count.as_device_ptr(),
                d_sum_conf.as_device_ptr(),
                d_sum_correct.as_device_ptr(),
                n as u32,
                n_bins as u32,
            ),
        )
        .expect("launch ece_bucket_kernel");
    stream.synchronize().expect("sync");

    let mut count_gpu = vec![0_u32; n_bins];
    let mut sum_conf_gpu = vec![0.0_f32; n_bins];
    let mut sum_correct_gpu = vec![0.0_f32; n_bins];
    d_count.copy_to_host(&mut count_gpu).expect("copy count");
    d_sum_conf
        .copy_to_host(&mut sum_conf_gpu)
        .expect("copy sum_conf");
    d_sum_correct
        .copy_to_host(&mut sum_correct_gpu)
        .expect("copy sum_correct");

    for b in 0..n_bins {
        // Counts are exact integers.
        assert_eq!(
            count_gpu[b], count_host[b],
            "ece count[{b}] mismatch: gpu={} host={}",
            count_gpu[b], count_host[b]
        );
        // sum_correct is a sum of exact-in-f32 1.0s ⇒ exact.
        assert_eq!(
            sum_correct_gpu[b].to_bits(),
            sum_correct_host[b].to_bits(),
            "ece sum_correct[{b}] mismatch: gpu={} host={}",
            sum_correct_gpu[b],
            sum_correct_host[b]
        );
        // sum_conf accumulates via atomics (nondeterministic order) ⇒ a few ulp.
        assert!(
            close(sum_conf_gpu[b], sum_conf_host[b], 1e-4, 1e-4),
            "ece sum_conf[{b}] mismatch: gpu={} host={}",
            sum_conf_gpu[b],
            sum_conf_host[b]
        );
    }
}

// ===========================================================================
// 5. ensemble_aggregate  —  HOST MEAN + BESSEL-CORRECTED VARIANCE
// ===========================================================================

#[test]
fn ensemble_aggregate_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m_members = 5_usize;
    let c_classes = 64_usize;

    let mut rng = LcgRng::new(0xE45E_3B1E);
    // logits[m*C + c] row-major.
    let logits: Vec<f32> = (0..m_members * c_classes)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();

    // Host oracle: mean[c] = (1/M)·Σ_m logits; var[c] = Σ_m (x-mean)²/(M-1),
    // accumulated in the kernel's m = 0..M order.
    let mut mean_host = vec![0.0_f32; c_classes];
    let mut var_host = vec![0.0_f32; c_classes];
    for c in 0..c_classes {
        let mut sum = 0.0_f32;
        for m in 0..m_members {
            sum += logits[m * c_classes + c];
        }
        let mean = sum / m_members as f32;
        mean_host[c] = mean;
        let mut acc = 0.0_f32;
        for m in 0..m_members {
            let d = logits[m * c_classes + c] - mean;
            acc += d * d;
        }
        var_host[c] = acc / (m_members as f32 - 1.0);
    }

    let ptx = crate::ptx_kernels::ensemble_aggregate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ensemble_aggregate_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_mean = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; c_classes]).expect("d_mean");
    let d_var = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; c_classes]).expect("d_var");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(c_classes as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_mean.as_device_ptr(),
                d_var.as_device_ptr(),
                m_members as u32,
                c_classes as u32,
            ),
        )
        .expect("launch ensemble_aggregate_kernel");
    stream.synchronize().expect("sync");

    let mut mean_gpu = vec![0.0_f32; c_classes];
    let mut var_gpu = vec![0.0_f32; c_classes];
    d_mean.copy_to_host(&mut mean_gpu).expect("copy mean");
    d_var.copy_to_host(&mut var_gpu).expect("copy var");

    let (rm, am) = worst_diff(&mean_gpu, &mean_host);
    for c in 0..c_classes {
        assert!(
            close(mean_gpu[c], mean_host[c], 1e-5, 1e-6),
            "ensemble mean[{c}] mismatch: gpu={} host={} (worst rel={rm:e} abs={am:e})",
            mean_gpu[c],
            mean_host[c]
        );
    }
    // The kernel fuses (x-mean)² with fma.rn; the host uses mul+add — a few ulp.
    let (rv, av) = worst_diff(&var_gpu, &var_host);
    for c in 0..c_classes {
        assert!(
            close(var_gpu[c], var_host[c], 1e-4, 1e-6),
            "ensemble var[{c}] mismatch: gpu={} host={} (worst rel={rv:e} abs={av:e})",
            var_gpu[c],
            var_host[c]
        );
    }
}

// ===========================================================================
// 6. flipout_perturb  —  HOST SIGN-PERTURBATION (LCG s_j + Σ W·r·x)
// ===========================================================================

#[test]
fn flipout_perturb_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let out_features = 16_usize;
    let in_features = 12_usize;
    let seed_s = 0x0F11_0007_C0DE_5161_u64;

    let mut rng = LcgRng::new(0xF11D_0017);
    let w_delta: Vec<f32> = (0..out_features * in_features)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let x: Vec<f32> = (0..in_features)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    // r_signs[i] ∈ {-1, +1}.
    let r_signs: Vec<f32> = (0..in_features)
        .map(|_| if rng.next_f32() < 0.5 { -1.0 } else { 1.0 })
        .collect();

    // Host oracle: s_j = LCG-bit sign; delta_out[j] = s_j·Σ_i W[j,i]·r_i·x[i].
    let mut out_host = vec![0.0_f32; out_features];
    for (j, slot) in out_host.iter_mut().enumerate() {
        let state = (seed_s ^ (j as u64))
            .wrapping_mul(LCG_MUL)
            .wrapping_add(LCG_ADD);
        let bit0 = ((state >> 33) as u32) & 1;
        let s_j = if bit0 == 1 { 1.0_f32 } else { -1.0_f32 };
        let mut acc = 0.0_f32;
        for i in 0..in_features {
            acc += w_delta[j * in_features + i] * r_signs[i] * x[i];
        }
        *slot = s_j * acc;
    }

    let ptx = crate::ptx_kernels::flipout_perturb_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "flipout_perturb_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_w = DeviceBuffer::<f32>::from_host(&w_delta).expect("d_w");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_r = DeviceBuffer::<f32>::from_host(&r_signs).expect("d_r");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; out_features]).expect("d_out");

    let block = 16_u32;
    let params = LaunchParams::new(grid_1d(out_features as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_w.as_device_ptr(),
                d_x.as_device_ptr(),
                d_r.as_device_ptr(),
                d_out.as_device_ptr(),
                out_features as u32,
                in_features as u32,
                seed_s,
            ),
        )
        .expect("launch flipout_perturb_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; out_features];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // s_j is exactly ±1; the inner product uses fma.rn vs the host's mul+add ⇒ a
    // few ulp over 12 terms (~1e-6 relative).
    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for j in 0..out_features {
        assert!(
            close(out_gpu[j], out_host[j], 1e-4, 1e-6),
            "flipout delta_out[{j}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[j],
            out_host[j]
        );
    }
}

// ===========================================================================
// 7. temp_scale_logits  —  HOST DIVISION (logit / T)
// ===========================================================================

#[test]
fn temp_scale_logits_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let temperature = 1.7_f32;

    let mut rng = LcgRng::new(0x7E_4953);
    let logits: Vec<f32> = (0..n).map(|_| rng.next_f32() * 8.0 - 4.0).collect();

    // Host oracle: out[i] = logit[i] / T. Both sides use IEEE div.rn ⇒ bit-exact.
    let out_host: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    let ptx = crate::ptx_kernels::temp_scale_logits_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "temp_scale_logits_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                temperature,
            ),
        )
        .expect("launch temp_scale_logits_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    for k in 0..n {
        assert_eq!(
            out_gpu[k].to_bits(),
            out_host[k].to_bits(),
            "temp_scale out[{k}] mismatch: gpu={} host={}",
            out_gpu[k],
            out_host[k]
        );
    }
}
