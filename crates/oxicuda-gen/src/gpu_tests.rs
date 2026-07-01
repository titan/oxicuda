//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's CPU reference. The launch ABI mirrors the working
//! `oxicuda-snn` / `oxicuda-ot` paths: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars are passed as the matching Rust
//! scalar (`.param .u32` / `.param .f32`), in the kernel's declared parameter
//! order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel is meant to mirror:
//!   `cfg_combine` ↔ [`crate::guidance::cfg::CfgGuidance::apply`],
//!   `lora_apply` ↔ [`crate::lora::merge::merge_lora`],
//!   `flow_velocity` ↔ [`crate::scheduler::flow_matching::FlowMatchingScheduler::euler_step`],
//!   `vae_kl_loss` ↔ [`crate::vae::kl::GaussianLatent::kl_loss_elementwise`].
//! * **Independent host re-derivation + crate cross-check** —
//!   `ddpm_step`: the kernel implements the *Algorithm-2* posterior form
//!   `(x_t − β/√(1−ᾱ)·ε)/√α + σ·z`, which is algebraically identical to (but a
//!   different FP arrangement from) the crate's posterior-mean
//!   [`crate::scheduler::ddpm::DdpmScheduler::step`]. The primary oracle is an
//!   independent host re-derivation of the kernel's exact documented formula;
//!   a secondary cross-check confirms it also matches the crate's DDPM step
//!   (with sample-clipping disabled, since the kernel never clips).
//!   `timestep_embed`: the kernel emits a **concatenated** sinusoidal layout
//!   `[sin_0..sin_{h−1}, cos_0..cos_{h−1}]`, whereas the crate's
//!   [`crate::score::timestep::SinusoidalEmbedding`] emits the **interleaved**
//!   layout `[sin_0, cos_0, sin_1, cos_1, …]`. Same frequencies, different
//!   positions. The primary oracle is an independent host re-derivation of the
//!   kernel's documented concatenated layout; a secondary cross-check confirms
//!   the de-permuted kernel output equals the crate embedding element-for-element.
//!
//! ## Known PTX bug-class coverage
//!
//! * **base-2 vs base-e exp** — `vae_kl_loss` computes `exp(logvar)` with
//!   `ex2.approx.f32` and so *must* scale the argument by `log2(e)` first. The
//!   kernel does (`mul.f32 %f3, logvar, 0F3FB8AA3B`). The oracle
//!   `kl_loss_elementwise` uses libm `exp` (true base-e), so a missing/incorrect
//!   `log2(e)` scale would turn `e^x` into `2^x` and fail this test by a wide
//!   margin (≈20–60 % at the logvars used here), not merely a few ulp.
//! * **base-2 power** — `timestep_embed` forms `max_period^(−freq)` as
//!   `ex2.approx(−freq·lg2(max_period))`; the base-e host re-derivation
//!   (`exp(−freq·ln(max_period))`) catches any base confusion there too.
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

/// JIT-compile `ptx` for the live device and look up `entry`.
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

/// `n` uniform `[-1, 1)` samples from the crate LCG.
fn uniform_pm1(rng: &mut LcgRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
}

// ===========================================================================
// 1. ddpm_step  —  INDEPENDENT HOST RE-DERIVATION + CRATE CROSS-CHECK
// ===========================================================================
//
// Kernel formula (Ho et al. 2020 Algorithm 2 posterior form):
//   x_prev = (x_t − β/√(1−ᾱ)·ε) / √α + σ·z
//
// The kernel takes α, ᾱ, β, σ as scalars and uses `sqrt.approx`/`rcp.approx`/
// `div.approx` (~2 ulp each). We feed it the *same* schedule scalars that the
// crate `DdpmScheduler::step` derives internally, so a comparison against both
// (a) an independent host re-derivation of the documented formula and
// (b) the crate DDPM step with clipping disabled (algebraically identical) is
// meaningful and genuinely fails on any miscompiled constant / shift / op.

#[test]
fn ddpm_step_matches_host_and_crate() {
    use crate::scheduler::ddpm::DdpmScheduler;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Linear 1000-step schedule; pick a well-conditioned timestep where
    // 1−ᾱ is moderate (so β/√(1−ᾱ) and the approx reciprocals are far from
    // their ill-conditioned regimes). Sample clipping is DISABLED so the crate
    // step matches the kernel, which never clips.
    let sched = DdpmScheduler::new(1000)
        .expect("DDPM scheduler")
        .with_clip_sample(false, 1.0);

    let alphas_bar = sched.schedule().alphas_bar();
    let t = (1..alphas_bar.len())
        .find(|&i| {
            let omab = 1.0 - alphas_bar[i];
            (0.2..=0.95).contains(&omab)
        })
        .expect("a timestep with moderate 1-alpha_bar must exist on a 1000-step schedule");

    let beta = sched.schedule().betas()[t];
    let alpha = sched.schedule().alphas()[t];
    let alpha_bar = sched.schedule().alphas_bar()[t];
    let alpha_bar_prev = sched.schedule().alphas_bar()[t - 1];
    let one_minus_ab = (1.0_f32 - alpha_bar).max(1e-10);
    // σ computed exactly as the crate `step` does for t > 0.
    let sigma = (beta * (1.0 - alpha_bar_prev) / one_minus_ab)
        .max(0.0)
        .sqrt();

    let n = 512_usize;
    let mut rng = LcgRng::new(0x0DD9_0000);
    let x_t = uniform_pm1(&mut rng, n);
    let eps = uniform_pm1(&mut rng, n);
    let z = uniform_pm1(&mut rng, n);

    // ---- (a) independent host re-derivation of the kernel's exact formula ----
    let coeff = beta / (1.0_f32 - alpha_bar).sqrt();
    let inv_sqrt_alpha = 1.0_f32 / alpha.sqrt();
    let host: Vec<f32> = (0..n)
        .map(|i| (x_t[i] - coeff * eps[i]) * inv_sqrt_alpha + sigma * z[i])
        .collect();

    // ---- (b) crate DDPM step (clip disabled) — algebraically identical ----
    let crate_out = sched.step(&eps, &x_t, t, &z).expect("ddpm step");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::ddpm_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ddpm_step");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x_t = DeviceBuffer::<f32>::from_host(&x_t).expect("d_x_t");
    let d_eps = DeviceBuffer::<f32>::from_host(&eps).expect("d_eps");
    let d_z = DeviceBuffer::<f32>::from_host(&z).expect("d_z");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x_t.as_device_ptr(),
                d_eps.as_device_ptr(),
                d_z.as_device_ptr(),
                d_out.as_device_ptr(),
                alpha,
                alpha_bar,
                beta,
                sigma,
                n as u32,
            ),
        )
        .expect("launch ddpm_step");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // (a) Primary: vs the independent host re-derivation of the documented
    // formula. The only divergence is the kernel's approx sqrt/rcp/div (~few
    // ulp ≈ 1e-6 relative); 1e-4 is comfortable yet flags any gross error.
    let (rel_h, abs_h) = worst_diff(&gpu, &host);
    for i in 0..n {
        assert!(
            close(gpu[i], host[i], 1e-4, 1e-5),
            "ddpm_step[{i}] vs host: gpu={} host={} (worst rel={rel_h:e} abs={abs_h:e}, t={t})",
            gpu[i],
            host[i]
        );
    }

    // (b) Secondary: vs the crate DDPM posterior-mean step. Same value computed
    // via a different FP arrangement (x0_pred → mean) plus the approx ops; a
    // looser 1e-3 relative bound is justified by the rearrangement and still
    // catches any semantic mismatch with the crate's DDPM math.
    let (rel_c, abs_c) = worst_diff(&gpu, &crate_out);
    for i in 0..n {
        assert!(
            close(gpu[i], crate_out[i], 1e-3, 1e-4),
            "ddpm_step[{i}] vs crate: gpu={} crate={} (worst rel={rel_c:e} abs={abs_c:e}, t={t})",
            gpu[i],
            crate_out[i]
        );
    }
}

// ===========================================================================
// 2. cfg_combine  —  CRATE ORACLE (guidance::cfg::CfgGuidance::apply)
// ===========================================================================

#[test]
fn cfg_combine_matches_crate() {
    use crate::guidance::cfg::{CfgConfig, CfgGuidance};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let scale = 7.5_f32;
    let mut rng = LcgRng::new(0x00C0_FFE1);
    let cond = uniform_pm1(&mut rng, n);
    let uncond = uniform_pm1(&mut rng, n);

    // ---- CPU oracle: out = uncond + scale*(cond - uncond) ----
    let guide = CfgGuidance::new(CfgConfig::new(scale).expect("cfg config"));
    let cpu = guide.apply(&cond, &uncond).expect("cfg apply");

    let ptx = crate::ptx_kernels::cfg_combine_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cfg_combine");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_cond = DeviceBuffer::<f32>::from_host(&cond).expect("d_cond");
    let d_uncond = DeviceBuffer::<f32>::from_host(&uncond).expect("d_uncond");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_cond.as_device_ptr(),
                d_uncond.as_device_ptr(),
                d_out.as_device_ptr(),
                scale,
                n as u32,
            ),
        )
        .expect("launch cfg_combine");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // GPU `fma.rn(scale, diff, uncond)` is single-rounding vs the CPU's
    // `uncond + scale*diff` (two roundings); ~1 ulp divergence.
    let (rel, abs) = worst_diff(&gpu, &cpu);
    for i in 0..n {
        assert!(
            close(gpu[i], cpu[i], 1e-4, 1e-6),
            "cfg_combine[{i}]: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[i],
            cpu[i]
        );
    }
}

// ===========================================================================
// 3. lora_apply  —  CRATE ORACLE (lora::merge::merge_lora)
// ===========================================================================

#[test]
fn lora_apply_matches_crate_merge() {
    use crate::lora::adapter::{LoraConfig, LoraLinear};
    use crate::lora::merge::merge_lora;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let in_features = 16_usize;
    let out_features = 32_usize;
    let rank = 4_usize;
    let n = out_features * in_features;

    let mut rng = LcgRng::new(0x10A0_0001);
    // Build a LoRA adapter and give it a non-trivial B so the delta is non-zero.
    let cfg = LoraConfig::new(rank, rank as f32 * 2.0).expect("lora config");
    let mut lora = LoraLinear::new(in_features, out_features, &cfg, &mut rng).expect("lora new");
    for v in lora.matrix_b_mut() {
        *v = rng.next_f32() * 2.0 - 1.0;
    }

    let delta = lora.delta_weight(); // [out × in] flat
    let scale = lora.scaling(); // alpha / rank
    let base = uniform_pm1(&mut rng, n);

    // ---- CPU oracle: W_merged = base + scale * delta ----
    let cpu = merge_lora(&base, &lora).expect("merge_lora");

    let ptx = crate::ptx_kernels::lora_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lora_apply");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_base = DeviceBuffer::<f32>::from_host(&base).expect("d_base");
    let d_delta = DeviceBuffer::<f32>::from_host(&delta).expect("d_delta");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_base.as_device_ptr(),
                d_delta.as_device_ptr(),
                d_out.as_device_ptr(),
                scale,
                n as u32,
            ),
        )
        .expect("launch lora_apply");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    let (rel, abs) = worst_diff(&gpu, &cpu);
    for i in 0..n {
        assert!(
            close(gpu[i], cpu[i], 1e-4, 1e-6),
            "lora_apply[{i}]: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[i],
            cpu[i]
        );
    }
}

// ===========================================================================
// 4. flow_velocity  —  CRATE ORACLE (scheduler::flow_matching::euler_step)
// ===========================================================================

#[test]
fn flow_velocity_matches_crate_euler() {
    use crate::scheduler::flow_matching::FlowMatchingScheduler;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let dt = 0.137_f32;
    let mut rng = LcgRng::new(0xF10C_0042);
    let x_t = uniform_pm1(&mut rng, n);
    let v = uniform_pm1(&mut rng, n);

    // ---- CPU oracle: x_next = x_t + dt*v ----
    let sched = FlowMatchingScheduler::new(50);
    let cpu = sched.euler_step(&x_t, &v, dt).expect("euler_step");

    let ptx = crate::ptx_kernels::flow_velocity_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "flow_velocity");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x_t).expect("d_x");
    let d_v = DeviceBuffer::<f32>::from_host(&v).expect("d_v");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_v.as_device_ptr(),
                d_out.as_device_ptr(),
                dt,
                n as u32,
            ),
        )
        .expect("launch flow_velocity");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // GPU `fma.rn(dt, v, x)` (single rounding) vs CPU `x + dt*v` (two): ~1 ulp.
    let (rel, abs) = worst_diff(&gpu, &cpu);
    for i in 0..n {
        assert!(
            close(gpu[i], cpu[i], 1e-4, 1e-6),
            "flow_velocity[{i}]: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[i],
            cpu[i]
        );
    }
}

// ===========================================================================
// 5. vae_kl_loss  —  CRATE ORACLE (vae::kl::GaussianLatent::kl_loss_elementwise)
// ===========================================================================
//
// BASE-2 BUG-CLASS CHECK: the kernel forms exp(logvar) via `ex2.approx` and so
// must pre-scale by log2(e). The oracle `kl_loss_elementwise` uses true base-e
// `exp`, so a missing/incorrect log2(e) scale would make exp(logvar) → 2^logvar
// and blow this test far past tolerance — not a few-ulp slip.

#[test]
fn vae_kl_loss_matches_crate() {
    use crate::vae::kl::GaussianLatent;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let mut rng = LcgRng::new(0x7AE0_C0DE);
    // mu in [-1, 1); logvar in [-2, 2) so the crate's [-30, 20] clamp is a
    // no-op and exp(logvar) ∈ [0.135, 7.39] stays in `ex2.approx`'s accurate band.
    let mu = uniform_pm1(&mut rng, n);
    let logvar: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // ---- CPU oracle: 0.5*(mu^2 + exp(logvar) - 1 - logvar) per element ----
    let latent = GaussianLatent::new(mu.clone(), logvar.clone()).expect("gaussian latent");
    let cpu = latent.kl_loss_elementwise();

    let ptx = crate::ptx_kernels::vae_kl_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "vae_kl_loss");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mu = DeviceBuffer::<f32>::from_host(&mu).expect("d_mu");
    let d_logvar = DeviceBuffer::<f32>::from_host(&logvar).expect("d_logvar");
    let d_loss = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_loss");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mu.as_device_ptr(),
                d_logvar.as_device_ptr(),
                d_loss.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch vae_kl_loss");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_loss.copy_to_host(&mut gpu).expect("copy loss");

    // `ex2.approx.f32` carries ~2 ulp; scaled by exp(logvar) ≤ 7.4 the absolute
    // error is ~1e-6. 5e-4 relative comfortably covers it while still catching a
    // base-2/base-e confusion (which would be tens of percent).
    let (rel, abs) = worst_diff(&gpu, &cpu);
    for i in 0..n {
        assert!(
            close(gpu[i], cpu[i], 5e-4, 1e-5),
            "vae_kl_loss[{i}]: gpu={} cpu={} mu={} logvar={} (worst rel={rel:e} abs={abs:e})",
            gpu[i],
            cpu[i],
            mu[i],
            logvar[i]
        );
    }
}

// ===========================================================================
// 6. timestep_embed  —  HOST RE-DERIVATION (concatenated) + CRATE CROSS-CHECK
// ===========================================================================
//
// LAYOUT NOTE: the kernel writes the CONCATENATED order
//   out[t·dim + k]        = sin(t / max_period^(2k/dim))     for k ∈ [0, half)
//   out[t·dim + half + k] = cos(t / max_period^(2k/dim))     for k ∈ [0, half)
// while the crate `SinusoidalEmbedding::embed_timestep` writes the INTERLEAVED
//   e[2k] = sin(...),  e[2k+1] = cos(...).
// Same frequencies, different positions. We assert the kernel against an
// independent host re-derivation of its documented concatenated layout, AND
// against the de-permuted crate embedding (kernel[k]==crate[2k],
// kernel[half+k]==crate[2k+1]) as a crate-oracle cross-check.

#[test]
fn timestep_embed_matches_host_and_crate() {
    use crate::score::timestep::SinusoidalEmbedding;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 16_usize;
    let half = dim / 2;
    let max_period = 10000.0_f32;
    // Small timesteps keep every angle ≤ t ≤ 3 < π, where `sin.approx`/
    // `cos.approx` are accurate to ~1e-6 (their argument-reduction error grows
    // only for large |x|). The k=0 frequency has divisor 1, so its angle equals
    // t exactly — the worst case, still inside the accurate band.
    let timesteps = vec![0.5_f32, 1.0, 2.0, 3.0];
    let n_t = timesteps.len();
    let total = n_t * dim;

    // ---- (a) independent host re-derivation of the concatenated layout ----
    let ln_mp = max_period.ln();
    let mut host = vec![0.0_f32; total];
    for (t_idx, &t) in timesteps.iter().enumerate() {
        for dim_idx in 0..dim {
            let (freq_idx, is_sin) = if dim_idx < half {
                (dim_idx, true)
            } else {
                (dim_idx - half, false)
            };
            let expo = 2.0_f32 * freq_idx as f32 / dim as f32;
            let inv_freq = (-expo * ln_mp).exp(); // = max_period^(-expo)
            let angle = t * inv_freq;
            host[t_idx * dim + dim_idx] = if is_sin { angle.sin() } else { angle.cos() };
        }
    }

    // ---- (b) crate embedding (interleaved) for the de-permutation check ----
    let embed = SinusoidalEmbedding::with_params(dim, max_period, 1.0).expect("embedding");
    let crate_rows: Vec<Vec<f32>> = timesteps.iter().map(|&t| embed.embed_timestep(t)).collect();

    let ptx = crate::ptx_kernels::timestep_embed_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "timestep_embed");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ts = DeviceBuffer::<f32>::from_host(&timesteps).expect("d_ts");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ts.as_device_ptr(),
                d_out.as_device_ptr(),
                dim as u32,
                n_t as u32,
                max_period,
            ),
        )
        .expect("launch timestep_embed");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; total];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    // (a) Primary: vs the independent host re-derivation of the documented
    // concatenated layout. sin/cos/ex2 approx ⇒ ~1e-6 absolute; 1e-4 absolute
    // (sin/cos ∈ [-1,1]) is comfortable yet catches a layout/exponent bug
    // (which would be O(1)).
    let (rel_h, abs_h) = worst_diff(&gpu, &host);
    for i in 0..total {
        assert!(
            close(gpu[i], host[i], 1e-3, 1e-4),
            "timestep_embed[{i}] vs host: gpu={} host={} (worst rel={rel_h:e} abs={abs_h:e})",
            gpu[i],
            host[i]
        );
    }

    // (b) Secondary crate cross-check: de-permute concatenated → interleaved.
    for (t_idx, crate_row) in crate_rows.iter().enumerate() {
        let base = t_idx * dim;
        for k in 0..half {
            let g_sin = gpu[base + k]; // kernel sin block
            let c_sin = crate_row[2 * k]; // crate interleaved sin
            assert!(
                close(g_sin, c_sin, 1e-3, 1e-4),
                "timestep_embed sin[t={t_idx},k={k}]: gpu={g_sin} crate={c_sin}"
            );
            let g_cos = gpu[base + half + k]; // kernel cos block
            let c_cos = crate_row[2 * k + 1]; // crate interleaved cos
            assert!(
                close(g_cos, c_cos, 1e-3, 1e-4),
                "timestep_embed cos[t={t_idx},k={k}]: gpu={g_cos} crate={c_cos}"
            );
        }
    }
}
