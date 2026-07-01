//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts equivalence to a CPU oracle. The launch ABI
//! mirrors the working `oxicuda-snn` `gpu_tests` path: device buffers are
//! passed as their `CUdeviceptr` (`.param .u64`), scalars as the matching Rust
//! scalar (`.param .u32` / `.param .u64` / `.param .f64`), in declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate / deterministic oracle** (strongest, exact up to FP rounding) —
//!   `clip_gradient` (vs `optimizer::clip_gradients`), `prv_convolve`
//!   (vs `accounting::prv::convolve_pmfs`), `exponential_sample` (vs an
//!   independent host inverse-CDF cumulative scan — the selection step the CPU
//!   `mechanism::exponential::exponential_sample` performs). These match within
//!   a tight tolerance bounded by a single fused-vs-separate rounding.
//! * **Bit-exact RNG** — the kernels draw a uniform from an inline counter-based
//!   Knuth-MMIX LCG whose integer pipeline is *exactly* reproducible in f64
//!   (a 53-bit integer scaled by an exact power of two). For `oue_encode` the
//!   whole output is a `u < threshold` decision on that exact uniform, so the
//!   bit vector is asserted bit-for-bit. For `svt_threshold` the uniform is
//!   exact but the Laplace transform is approximate (see below); inputs are
//!   engineered so each comparison sits a wide margin from the threshold, making
//!   the binary decision bit-exact regardless.
//! * **Bit-exact uniform + SFU-bounded transform + distributional** —
//!   `laplace_noise` / `gaussian_noise` build their noise from the exact uniform
//!   via `ln`/`sin`/`cos`, which PTX only provides as `.approx.f32` SFU
//!   instructions. The host re-derives each sample through the *same* f32 path,
//!   so the only divergence is hardware-SFU vs libm rounding of one transcendental
//!   (bounded by the documented SFU error). Each sample is checked against that
//!   re-derivation, and — independently — a large aggregate sample's mean and
//!   standard deviation are checked against the analytic moments within a
//!   tolerance derived from the standard error (a genuinely-failable
//!   distributional test, clearly labelled distributional, not exact).
//!
//! Every test skips (returns early) when no CUDA device is present.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

// ---------------------------------------------------------------------------
// Inline-LCG constants — must match the PTX kernels' RNG exactly.
// ---------------------------------------------------------------------------

/// Knuth MMIX multiplier (ptx_kernels.rs: `rd_mul`).
const LCG_M: u64 = 6_364_136_223_846_793_005;
/// Knuth MMIX increment (ptx_kernels.rs: `rd_add`).
const LCG_A: u64 = 1_442_695_040_888_963_407;
/// Fibonacci-hashing (golden-ratio) per-thread seed mixer (ptx_kernels.rs).
const GOLDEN: u64 = 11_400_714_819_323_198_485;
/// `2^53` as f64 — the exact divisor for the 53-bit → uniform conversion.
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
/// PTX `ln 2` constant `0D3FE62E42FEFA39EF`.
const LN2_BITS: u64 = 0x3FE6_2E42_FEFA_39EF;
/// PTX `2*pi` constant `0D401921FB54442D18`.
const TWO_PI_BITS: u64 = 0x4019_21FB_5444_2D18;
/// PTX gaussian `u1` clamp `0D36A0000000000000` (= 2^-149).
const U1_CLAMP_BITS: u64 = 0x36A0_0000_0000_0000;

/// The kernels' inline one-step uniform draw, re-derived **bit-exactly**:
/// `state = (seed ^ (tid * GOLDEN))` advanced one LCG step; the top 53 bits are
/// converted to f64 (exact for values `< 2^53`) and scaled by `1/2^53` (an exact
/// power of two), so the result is bit-identical to the GPU's
/// `cvt.rn.f64.u64` + `div.rn.f64`.
fn lcg_uniform_onestep(seed: u64, tid: u32) -> f64 {
    let mixed = seed ^ u64::from(tid).wrapping_mul(GOLDEN);
    let state = mixed.wrapping_mul(LCG_M).wrapping_add(LCG_A);
    ((state >> 11) as f64) / TWO_POW_53
}

/// Host re-derivation of the fixed `laplace_noise` transform for a given exact
/// uniform `u`, mirroring the kernel's `lg2.approx.f32` path (`ln(arg)` is taken
/// in f32, everything else in f64).
fn laplace_noise_host(u: f64, scale: f64) -> f64 {
    let up = u - 0.5;
    let arg = 1.0 - 2.0 * up.abs(); // in (0, 1]
    let log2 = (arg as f32).log2(); // f32 SFU mirror
    let ln_arg = f64::from(log2) * f64::from_bits(LN2_BITS);
    let neg_ln = -ln_arg;
    let s = if up >= 0.0 { scale } else { -scale };
    neg_ln * s
}

/// Host re-derivation of the fixed `gaussian_noise` Box-Muller transform for a
/// thread index `tid`, mirroring the kernel's f32 `lg2`/`sin`/`cos` SFU path and
/// IEEE `sqrt.rn.f64`.
fn gaussian_noise_host(seed: u64, tid: u32, sigma: f64) -> f64 {
    let pair = u64::from(tid >> 1);
    let odd = (tid & 1) == 1;

    let s1 = (seed ^ pair.wrapping_mul(GOLDEN))
        .wrapping_mul(LCG_M)
        .wrapping_add(LCG_A);
    let u1_raw = ((s1 >> 11) as f64) / TWO_POW_53;
    let u1 = u1_raw.max(f64::from_bits(U1_CLAMP_BITS));

    let s2 = s1.wrapping_mul(LCG_M).wrapping_add(LCG_A);
    let u2 = ((s2 >> 11) as f64) / TWO_POW_53;

    let log2 = (u1 as f32).log2();
    let ln_u1 = f64::from(log2) * f64::from_bits(LN2_BITS);
    let r = (-2.0 * ln_u1).sqrt(); // sqrt.rn.f64 == IEEE f64 sqrt

    let theta = u2 * f64::from_bits(TWO_PI_BITS);
    let cs = if odd {
        f64::from((theta as f32).sin())
    } else {
        f64::from((theta as f32).cos())
    };
    r * cs * sigma
}

// ---------------------------------------------------------------------------
// Shared GPU helpers (mirrors the snn template).
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
struct GpuFixture {
    ctx: Arc<Context>,
    sm: u32,
}

fn gpu_fixture() -> Option<GpuFixture> {
    oxicuda_driver::init().ok()?;
    if oxicuda_driver::Device::count().ok()? == 0 {
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

/// Relative-with-absolute-floor closeness test for FP comparisons.
fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Worst (relative, absolute) divergence over two equal-length slices.
fn worst_diff(gpu: &[f64], cpu: &[f64]) -> (f64, f64) {
    let mut worst_abs = 0.0_f64;
    let mut worst_rel = 0.0_f64;
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

/// Sample mean and (population) standard deviation of a slice.
fn mean_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    (mean, var.sqrt())
}

// ===========================================================================
// 1. clip_gradient  —  CRATE ORACLE (optimizer::clip_gradients), deterministic
// ===========================================================================

#[test]
fn clip_gradient_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_params = 100_usize;
    let batch = 8_usize;
    let clip = 1.0_f64;

    // Per-sample gradients whose L2 norms straddle `clip`: even samples are
    // scaled to ~2x the bound (must be clipped), odd ones to ~0.5x (unchanged).
    let mut rng = LcgRng::new(0xC11D_6244);
    let mut grads = vec![0.0_f64; batch * n_params];
    for s in 0..batch {
        let mut raw: Vec<f64> = (0..n_params).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
        let raw_norm = raw.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let target = if s % 2 == 0 { 2.0 } else { 0.5 } * clip;
        let k = target / raw_norm;
        for (j, v) in raw.iter_mut().enumerate() {
            *v *= k;
            grads[s * n_params + j] = *v;
        }
    }

    // ---- CPU oracle: crate clip_gradients on per-sample rows ----
    let rows: Vec<Vec<f64>> = (0..batch)
        .map(|s| grads[s * n_params..(s + 1) * n_params].to_vec())
        .collect();
    let clipped = crate::optimizer::clip_gradients(&rows, clip).expect("clip_gradients");
    let expected: Vec<f64> = clipped.into_iter().flatten().collect();

    // ---- GPU ----
    let ptx = crate::ptx_kernels::clip_gradient_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "clip_gradient");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_grads = DeviceBuffer::<f64>::from_host(&grads).expect("d_grads");

    // One block per sample; one thread per parameter (n_params <= 256).
    let params = LaunchParams::new(batch as u32, n_params as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_grads.as_device_ptr(), n_params as u32, batch as u32, clip),
        )
        .expect("launch clip_gradient");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f64; batch * n_params];
    d_grads.copy_to_host(&mut gpu).expect("copy grads");

    let (rel, abs) = worst_diff(&gpu, &expected);
    // sqrt.rn.f64 + div.rn.f64 + same summation order ⇒ essentially bit-exact;
    // 1e-9 relative is far tighter than any formula error yet covers fused-vs-
    // separate rounding.
    for k in 0..gpu.len() {
        assert!(
            close(gpu[k], expected[k], 1e-9, 1e-12),
            "clip_gradient[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            expected[k]
        );
    }
    // Sanity: clipped (even) samples must now have norm ≈ clip; unclipped
    // (odd) samples must be unchanged.
    for s in 0..batch {
        let norm = gpu[s * n_params..(s + 1) * n_params]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            .sqrt();
        if s % 2 == 0 {
            assert!(
                (norm - clip).abs() < 1e-6,
                "sample {s} should be clipped to {clip}, got norm {norm}"
            );
        } else {
            assert!(
                norm < clip,
                "sample {s} (norm {norm}) was within bound, must be unchanged"
            );
        }
    }
}

// ===========================================================================
// 2. prv_convolve  —  CRATE ORACLE (accounting::prv::convolve_pmfs), determ.
// ===========================================================================

#[test]
fn prv_convolve_matches_crate() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let grid = 32_usize;
    let out_len = 2 * grid - 1;
    let mut rng = LcgRng::new(0x9C04_0FFE);

    // Two normalised PMFs (positive, sum to 1).
    let mut a: Vec<f64> = (0..grid).map(|_| rng.next_f64() + 0.01).collect();
    let mut b: Vec<f64> = (0..grid).map(|_| rng.next_f64() + 0.01).collect();
    let sa: f64 = a.iter().sum();
    let sb: f64 = b.iter().sum();
    for v in &mut a {
        *v /= sa;
    }
    for v in &mut b {
        *v /= sb;
    }

    let expected = crate::accounting::prv::convolve_pmfs(&a, &b);
    assert_eq!(expected.len(), out_len);

    let ptx = crate::ptx_kernels::prv_convolve_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "prv_convolve");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f64>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f64>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; out_len]).expect("d_out");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(out_len as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                grid as u32,
            ),
        )
        .expect("launch prv_convolve");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f64; out_len];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    let (rel, abs) = worst_diff(&gpu, &expected);
    for k in 0..out_len {
        // fma.rn.f64 accumulation vs separate mul+add ⇒ ~1 ulp/term.
        assert!(
            close(gpu[k], expected[k], 1e-10, 1e-12),
            "prv_convolve[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 3. exponential_sample  —  INDEPENDENT HOST inverse-CDF scan, deterministic
// ===========================================================================

/// Host oracle: the kernel's serial cumulative scan — first index whose running
/// prefix sum reaches `threshold`, else `n-1` (the CPU
/// `mechanism::exponential::exponential_sample` selection step). The GPU and
/// host accumulate in the same order with f64 adds, so the chosen index is exact.
fn exponential_select_host(weights: &[f64], threshold: f64) -> u32 {
    let mut cum = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        cum += w;
        if cum >= threshold {
            return i as u32;
        }
    }
    (weights.len() - 1) as u32
}

#[test]
fn exponential_sample_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 16_usize;
    let mut rng = LcgRng::new(0xE9F0_1234);
    let weights: Vec<f64> = (0..n).map(|_| 0.05 + rng.next_f64()).collect();
    let total: f64 = weights.iter().sum();

    let ptx = crate::ptx_kernels::exponential_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "exponential_sample");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // Exercise a spread of thresholds: early, middle, late indices, and the
    // numerical fallback (threshold just above the total weight).
    let fractions = [0.0_f64, 0.1, 0.27, 0.5, 0.73, 0.99, 1.0001];
    for &frac in &fractions {
        let threshold = frac * total;
        let expected = exponential_select_host(&weights, threshold);

        let d_w = DeviceBuffer::<f64>::from_host(&weights).expect("d_w");
        // Initialise output to a sentinel that the kernel must overwrite.
        let d_out = DeviceBuffer::<u32>::from_host(&[0xDEAD_BEEF_u32]).expect("d_out");

        let params = LaunchParams::new(1u32, 1u32);
        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_w.as_device_ptr(),
                    n as u32,
                    threshold,
                    d_out.as_device_ptr(),
                ),
            )
            .expect("launch exponential_sample");
        stream.synchronize().expect("sync");

        let mut got = [0u32; 1];
        d_out.copy_to_host(&mut got).expect("copy out");
        assert_eq!(
            got[0], expected,
            "exponential_sample(frac={frac}): gpu={} host={expected}",
            got[0]
        );
    }
}

// ===========================================================================
// 4. oue_encode  —  BIT-EXACT RNG (inline LCG uniform + threshold decision)
// ===========================================================================

#[test]
fn oue_encode_matches_host_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k = 64_usize;
    let true_bit = 19_u32;
    let epsilon = 1.5_f64;
    let p_half = 0.5_f64;
    let p_flip = 1.0 / (epsilon.exp() + 1.0); // 1/(e^eps + 1)
    let seed = 0x0DE0_1234_5678_9ABC_u64;

    // Host: bit-exact. The whole output is `u < threshold` on the exact uniform.
    let mut expected = vec![0u8; k];
    for (i, slot) in expected.iter_mut().enumerate() {
        let u = lcg_uniform_onestep(seed, i as u32);
        let thresh = if i as u32 == true_bit { p_half } else { p_flip };
        *slot = u8::from(u < thresh);
    }

    let ptx = crate::ptx_kernels::oue_encode_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "oue_encode");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_out = DeviceBuffer::<u8>::from_host(&vec![0xFF_u8; k]).expect("d_out");

    let params = LaunchParams::new(1u32, k as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                true_bit,
                d_out.as_device_ptr(),
                k as u32,
                p_half,
                p_flip,
                seed,
            ),
        )
        .expect("launch oue_encode");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0u8; k];
    d_out.copy_to_host(&mut gpu).expect("copy out");

    for i in 0..k {
        assert!(
            gpu[i] == 0 || gpu[i] == 1,
            "oue bit[{i}] = {} not binary",
            gpu[i]
        );
        assert_eq!(
            gpu[i], expected[i],
            "oue_encode bit[{i}] mismatch: gpu={} host={}",
            gpu[i], expected[i]
        );
    }
}

// ===========================================================================
// 5. svt_threshold  —  BIT-EXACT decision (exact RNG + engineered margins)
// ===========================================================================

#[test]
fn svt_threshold_decisions_match_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let noise_scale = 1.0_f64;
    let noisy_threshold = 3.0_f64;
    let seed = 0x5471_0FED_CBA9_8765_u64;
    // Margin ≫ the worst possible GPU-vs-host noise gap (SFU-bounded, ~1e-4·|noise|).
    let margin = 0.05_f64;

    // Engineer queries so that query[i] + (exact host noise) lands `margin` above
    // the threshold for even i (expected result 1) and `margin` below for odd i
    // (expected result 0). Because the GPU noise matches the host re-derivation
    // to far better than `margin`, the binary decision is bit-exact — yet a real
    // RNG/transform bug would move the sum by O(noise) and flip the result.
    let mut queries = vec![0.0_f64; n];
    let mut expected = vec![0u8; n];
    for i in 0..n {
        let u = lcg_uniform_onestep(seed, i as u32);
        let noise = laplace_noise_host(u, noise_scale);
        let delta = if i % 2 == 0 { margin } else { -margin };
        queries[i] = noisy_threshold - noise + delta;
        expected[i] = u8::from(delta >= 0.0);
    }

    let ptx = crate::ptx_kernels::svt_threshold_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "svt_threshold");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f64>::from_host(&queries).expect("d_q");
    let d_res = DeviceBuffer::<u8>::from_host(&vec![0xFF_u8; n]).expect("d_res");

    let params = LaunchParams::new(1u32, n as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                n as u32,
                noisy_threshold,
                noise_scale,
                seed,
                d_res.as_device_ptr(),
            ),
        )
        .expect("launch svt_threshold");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0u8; n];
    d_res.copy_to_host(&mut gpu).expect("copy res");

    for i in 0..n {
        assert!(
            gpu[i] == 0 || gpu[i] == 1,
            "svt res[{i}] = {} not binary",
            gpu[i]
        );
        assert_eq!(
            gpu[i], expected[i],
            "svt_threshold[{i}] mismatch: gpu={} host={} (q={}, expected_side={})",
            gpu[i], expected[i], queries[i], expected[i]
        );
    }
}

// ===========================================================================
// 6. laplace_noise  —  bit-exact uniform + SFU transform + distributional
// ===========================================================================

/// Run `laplace_noise` over a zeroed buffer of length `n`, returning the noise.
fn run_laplace(kernel: &Kernel, stream: &Stream, n: usize, scale: f64, seed: u64) -> Vec<f64> {
    let d_data = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_data");
    let block = n as u32;
    let params = LaunchParams::new(1u32, block);
    kernel
        .launch(
            &params,
            stream,
            &(d_data.as_device_ptr(), n as u32, scale, seed),
        )
        .expect("launch laplace_noise");
    stream.synchronize().expect("sync");
    let mut out = vec![0.0_f64; n];
    d_data.copy_to_host(&mut out).expect("copy data");
    out
}

#[test]
fn laplace_noise_matches_host_and_moments() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let scale = 1.3_f64;
    let n = 1024_usize;
    let seed0 = 0x1A2B_3C4D_5E6F_7081_u64;

    let ptx = crate::ptx_kernels::laplace_noise_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "laplace_noise");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // ---- (a) Per-element: GPU noise vs exact-uniform + SFU-mirror transform ----
    let gpu = run_laplace(&kernel, &stream, n, scale, seed0);
    let host: Vec<f64> = (0..n)
        .map(|i| laplace_noise_host(lcg_uniform_onestep(seed0, i as u32), scale))
        .collect();
    for &g in &gpu {
        assert!(g.is_finite(), "laplace noise produced non-finite {g}");
    }
    let (rel, abs) = worst_diff(&gpu, &host);
    for i in 0..n {
        // Only the single `ln` differs (f32 SFU vs libm). Measured on this
        // device/seed: worst |Δ| = 8.6e-7 absolute (rel reaches 1.4e-4 only at
        // near-zero noise, where Δ is sub-µ). The bound below (3e-4 rel, 5e-6
        // abs) clears those by 2–6× yet is ~10^3–10^4× tighter than any
        // sign/scale/formula error (which moves the value by O(noise)).
        assert!(
            close(gpu[i], host[i], 3e-4, 5e-6),
            "laplace[{i}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[i],
            host[i]
        );
    }

    // ---- (b) Distributional (genuinely failable): mean ≈ 0, std ≈ scale·√2 ----
    let launches = 32_usize;
    let mut all = Vec::with_capacity(launches * n);
    for l in 0..launches {
        let seed = seed0 ^ (0x9E37_79B9_7F4A_7C15_u64.wrapping_mul(l as u64 + 1));
        all.extend(run_laplace(&kernel, &stream, n, scale, seed));
    }
    let total = all.len() as f64;
    let (mean, std) = mean_std(&all);
    let true_std = scale * std::f64::consts::SQRT_2;
    // SE(mean) = true_std/√N; 6σ band. A sign error pins the mean far outside.
    let mean_tol = 6.0 * true_std / total.sqrt();
    assert!(
        mean.abs() < mean_tol,
        "laplace mean {mean:e} exceeds {mean_tol:e} (N={total})"
    );
    // SE(std) for Laplace (kurtosis 6) ≈ std·√(5/(4N)) ≈ 0.62% here, so the 5%
    // band is ~8·SE (negligible flake) yet a missing-√2 factor (≈29% low) is
    // ~47·SE out — decisively caught.
    assert!(
        (std - true_std).abs() < 0.05 * true_std,
        "laplace std {std} vs analytic {true_std} (N={total})"
    );
}

// ===========================================================================
// 7. gaussian_noise  —  bit-exact uniform + SFU transform + distributional
// ===========================================================================

fn run_gaussian(kernel: &Kernel, stream: &Stream, n: usize, sigma: f64, seed: u64) -> Vec<f64> {
    let d_data = DeviceBuffer::<f64>::from_host(&vec![0.0_f64; n]).expect("d_data");
    let block = n as u32;
    let params = LaunchParams::new(1u32, block);
    kernel
        .launch(
            &params,
            stream,
            &(d_data.as_device_ptr(), n as u32, sigma, seed),
        )
        .expect("launch gaussian_noise");
    stream.synchronize().expect("sync");
    let mut out = vec![0.0_f64; n];
    d_data.copy_to_host(&mut out).expect("copy data");
    out
}

#[test]
fn gaussian_noise_matches_host_and_moments() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let sigma = 0.75_f64;
    let n = 1024_usize;
    let seed0 = 0x0F1E_2D3C_4B5A_6978_u64;

    let ptx = crate::ptx_kernels::gaussian_noise_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gaussian_noise");
    let stream = Stream::new(&fx.ctx).expect("stream");

    // ---- (a) Per-element: GPU noise vs exact-uniform + SFU-mirror Box-Muller --
    let gpu = run_gaussian(&kernel, &stream, n, sigma, seed0);
    let host: Vec<f64> = (0..n)
        .map(|i| gaussian_noise_host(seed0, i as u32, sigma))
        .collect();
    for &g in &gpu {
        assert!(g.is_finite(), "gaussian noise produced non-finite {g}");
    }
    let (rel, abs) = worst_diff(&gpu, &host);
    for i in 0..n {
        // r = sqrt(-2 ln u1) shares `ln` (f32 SFU), and z scales an SFU sin/cos;
        // near sin/cos zeros the *relative* error is unbounded, so use an
        // absolute floor sized to the SFU absolute error (~r·sigma·2^-21).
        // Measured worst |Δ| = 1.45e-6 absolute over all samples; the bound
        // below (2e-4 rel, 1e-5 abs) clears it ~7× yet is orders of magnitude
        // tighter than any formula error.
        assert!(
            close(gpu[i], host[i], 2e-4, 1e-5),
            "gaussian[{i}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[i],
            host[i]
        );
    }

    // ---- (b) Distributional (genuinely failable): mean ≈ 0, std ≈ sigma -------
    let launches = 32_usize;
    let mut all = Vec::with_capacity(launches * n);
    for l in 0..launches {
        let seed = seed0 ^ (0xD1B5_4A32_D192_ED03_u64.wrapping_mul(l as u64 + 1));
        all.extend(run_gaussian(&kernel, &stream, n, sigma, seed));
    }
    let total = all.len() as f64;
    let (mean, std) = mean_std(&all);
    let mean_tol = 6.0 * sigma / total.sqrt();
    assert!(
        mean.abs() < mean_tol,
        "gaussian mean {mean:e} exceeds {mean_tol:e} (N={total})"
    );
    // SE(std) for a normal ≈ sigma/√(2N); allow 5% > 6·SE at this N.
    assert!(
        (std - sigma).abs() < 0.05 * sigma,
        "gaussian std {std} vs sigma {sigma} (N={total})"
    );
}
