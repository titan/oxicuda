//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to an independent CPU oracle. The launch ABI mirrors the proven
//! `oxicuda-snn` / `oxicuda-ot` harnesses: device buffers are passed as their
//! `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .u32` / `.param .f32` / `.param .u64`) in the kernel's declared
//! parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Numerical equivalence** (strongest) — compared within FP32 tolerance (or
//!   bit-exact where the math is integer / power-of-two exact) to an independent
//!   host re-derivation of the kernel's documented arithmetic:
//!   `nt_xent_softmax` (masked scale — pass 1 of a host-finished softmax),
//!   `momentum_update`, `byol_cosine_loss`, `barlow_cross_corr`, `random_mask`
//!   (bit-exact Bernoulli LCG mask), `cosine_similarity`, `gather_features`
//!   (bit-exact gather), `momentum_update_f16` (bit-exact f16 EMA), and
//!   `byol_cosine_loss_bf16` (bit-exact bf16 cosine accumulate).
//! * **Load + structural (architecture fallback)** — the three architecture-
//!   deepening kernels (`barlow_cross_corr_wgmma`, `nt_xent_softmax_warp`,
//!   `gather_features_bulk`) emit Hopper-`wgmma` / `redux.sync` / Blackwell-TMA
//!   instructions that the A4000 (Ampere, sm_86) cannot execute. On sm_86 each
//!   dispatches to its portable scalar fallback (byte-identical to the already-
//!   validated `barlow_cross_corr` / `nt_xent_softmax` / `gather_features`
//!   kernel). These tests JIT-load the sm_86 output on the device, assert the
//!   fallback path was taken (no advanced ISA token), and launch it once to
//!   confirm on-device execution. The advanced ISA fast-path itself is NOT
//!   runnable here and is reported as scope-excluded — never green-washed.
//!
//! ## PTX bug found and fixed
//!
//! ### `random_mask` — Bernoulli uniform squashed to `[0, 0.5)`
//!
//! The inline LCG produced `u = (state >> 33) / 2^31`, which is already a
//! correct uniform in `[0, 1)` (a 31-bit value divided by `2^31`). The kernel
//! then multiplied by `0.5` (`mul.f32 %f3, %f3, 0F3F000000`), squashing every
//! draw into `[0, 0.5)` and roughly **doubling** the effective drop ratio
//! (`P(u < r) = r / 0.5 = 2r` for `r <= 0.5`). Fixed in `ptx_kernels.rs` by
//! deleting the spurious `*0.5`; the mask is now validated bit-exact against an
//! independent host LCG (see `random_mask_matches_host_lcg`).
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

// IEEE-754 half / bfloat16 codecs for the mixed-precision kernel oracles. Only
// the exact (lossless) paths are exercised: the tests feed values that are
// exactly representable so the GPU's `cvt.rn` rounding never fires and the
// comparison is bit-exact.

/// Decode an IEEE-754 binary16 (`f16`) bit pattern to `f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = if (h >> 15) & 1 == 1 {
        -1.0_f32
    } else {
        1.0_f32
    };
    let exp = i32::from((h >> 10) & 0x1f);
    let mant = f32::from(h & 0x3ff);
    let mag = if exp == 0 {
        mant * 2.0_f32.powi(-24)
    } else if exp == 0x1f {
        if mant == 0.0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + mant / 1024.0) * 2.0_f32.powi(exp - 15)
    };
    sign * mag
}

/// Encode an `f32` that is *exactly* representable in `f16` (its low 13 mantissa
/// bits are zero) to its binary16 bit pattern, losslessly.
fn f16_from_exact(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    if exp32 == 0 {
        return sign; // ±0
    }
    let exp16 = exp32 - 127 + 15;
    let mant16 = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exp16 as u16) << 10) | mant16
}

/// Decode a bfloat16 bit pattern to `f32` (zero-extend the mantissa), matching
/// the GPU's `cvt.f32.bf16`.
fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Encode an `f32` that is *exactly* representable in `bf16` (its low 16
/// mantissa bits are zero) by truncating to the high 16 bits, losslessly.
fn bf16_from_exact(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

// ===========================================================================
// 1. nt_xent_softmax  —  NUMERICAL (masked, scaled similarity; pass 1 of 3)
// ===========================================================================

#[test]
fn nt_xent_softmax_matches_host() {
    // The kernel performs pass 1 of a three-pass NT-Xent softmax: it scales each
    // similarity by `inv_temp` and masks the diagonal (i == j) to -INF, writing
    // back in place. Passes 2/3 (the per-row exp-normalise) are documented as
    // host-side, so the kernel's *contract* is exactly this masked scale — which
    // is a real, fully checkable on-device computation, not a stub.
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 4_usize; // N samples
    let n2 = 2 * n; // 2N rows/cols of the similarity matrix
    let inv_temp = 2.0_f32; // 1 / temperature, temperature = 0.5

    let mut rng = LcgRng::new(0x6E_7C_E0_7A);
    let sim: Vec<f32> = (0..n2 * n2).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    let ptx = crate::ptx_kernels::nt_xent_softmax_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "nt_xent_softmax_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sim = DeviceBuffer::<f32>::from_host(&sim).expect("d_sim");

    // Grid = (2N) blocks (ctaid.x = row i), block = (2N) threads (tid.x = col j).
    let params = LaunchParams::new(n2 as u32, n2 as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_sim.as_device_ptr(), n2 as u32, inv_temp),
        )
        .expect("launch nt_xent_softmax_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n2 * n2];
    d_sim.copy_to_host(&mut out_gpu).expect("copy sim");

    for i in 0..n2 {
        for j in 0..n2 {
            let g = out_gpu[i * n2 + j];
            if i == j {
                // Diagonal self-mask must be -INF (a real, observable write).
                assert!(
                    g.is_infinite() && g < 0.0,
                    "nt_xent diagonal [{i},{j}] = {g}, expected -INF"
                );
            } else {
                let expected = sim[i * n2 + j] * inv_temp;
                assert!(
                    close(g, expected, 1e-5, 1e-6),
                    "nt_xent [{i},{j}] mismatch: gpu={g} host={expected}"
                );
            }
        }
    }
}

// ===========================================================================
// 2. momentum_update  —  NUMERICAL (EMA blend theta = m*target + (1-m)*online)
// ===========================================================================

#[test]
fn momentum_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 1024_usize;
    let momentum = 0.9_f32;

    let mut rng = LcgRng::new(0x0000_EA07);
    let target0: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let online: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference: theta = m*target + (1-m)*online.
    let expected: Vec<f32> = target0
        .iter()
        .zip(&online)
        .map(|(&t, &o)| momentum * t + (1.0 - momentum) * o)
        .collect();

    let ptx = crate::ptx_kernels::momentum_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "momentum_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_target = DeviceBuffer::<f32>::from_host(&target0).expect("d_target");
    let d_online = DeviceBuffer::<f32>::from_host(&online).expect("d_online");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_target.as_device_ptr(),
                d_online.as_device_ptr(),
                n as u32,
                momentum,
            ),
        )
        .expect("launch momentum_update_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n];
    d_target.copy_to_host(&mut gpu).expect("copy target");

    // GPU uses single-rounding `fma.rn`; host uses two-rounding mul+add (~1 ulp).
    let (rel, abs) = worst_diff(&gpu, &expected);
    for k in 0..n {
        assert!(
            close(gpu[k], expected[k], 1e-5, 1e-6),
            "momentum [{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 3. byol_cosine_loss  —  NUMERICAL (scalar accumulate of 2 - 2*p*z)
// ===========================================================================

#[test]
fn byol_cosine_loss_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;
    let mut rng = LcgRng::new(0xB7_01_C0_5E);
    let p: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let z: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // Host reference: sum_i (2 - 2 * p_i * z_i), accumulated in f64 to bound the
    // oracle error; the GPU uses one f32 accumulator via atom.global.add.f32.
    let expected: f32 = p
        .iter()
        .zip(&z)
        .map(|(&pi, &zi)| f64::from(2.0 - 2.0 * pi * zi))
        .sum::<f64>() as f32;

    let ptx = crate::ptx_kernels::byol_cosine_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "byol_cosine_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<f32>::from_host(&p).expect("d_p");
    let d_z = DeviceBuffer::<f32>::from_host(&z).expect("d_z");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_z.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch byol_cosine_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out = [0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    // Atomic-add reorders the n f32 terms; the worst-case rounding divergence
    // over |sum| ~ O(n) is a few hundred ulp — far inside this absolute bound,
    // yet 1e-2 still flags any gross formula error (e.g. a missing 2 - or sign).
    assert!(
        close(out[0], expected, 1e-4, 1e-2),
        "byol_cosine_loss sum mismatch: gpu={} host={}",
        out[0],
        expected
    );
}

// ===========================================================================
// 4. barlow_cross_corr  —  NUMERICAL (C[i,j] = sum_n Z_A[n,i]*Z_B[n,j])
// ===========================================================================

#[test]
fn barlow_cross_corr_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let batch_n = 16_usize;
    let dim_d = 6_usize;

    let mut rng = LcgRng::new(0xBA_71_0C_05);
    let za: Vec<f32> = (0..batch_n * dim_d)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let zb: Vec<f32> = (0..batch_n * dim_d)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // Host reference: C[i,j] = sum_n Z_A[n,i] * Z_B[n,j].
    let mut expected = vec![0.0_f32; dim_d * dim_d];
    for i in 0..dim_d {
        for j in 0..dim_d {
            let mut acc = 0.0_f32;
            for n in 0..batch_n {
                acc += za[n * dim_d + i] * zb[n * dim_d + j];
            }
            expected[i * dim_d + j] = acc;
        }
    }

    let ptx = crate::ptx_kernels::barlow_cross_corr_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "barlow_cross_corr_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_za = DeviceBuffer::<f32>::from_host(&za).expect("d_za");
    let d_zb = DeviceBuffer::<f32>::from_host(&zb).expect("d_zb");
    let d_c = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; dim_d * dim_d]).expect("d_c");

    // Grid = (D, D) blocks (ctaid.x = i, ctaid.y = j); block.x = batch_n threads
    // each owning one batch element n (strides if block.x < batch_n).
    let params = LaunchParams::new((dim_d as u32, dim_d as u32), (batch_n as u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_za.as_device_ptr(),
                d_zb.as_device_ptr(),
                d_c.as_device_ptr(),
                batch_n as u32,
                dim_d as u32,
            ),
        )
        .expect("launch barlow_cross_corr_kernel");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; dim_d * dim_d];
    d_c.copy_to_host(&mut c_gpu).expect("copy c");

    // Atomic accumulation over batch_n terms reorders the f32 adds; a few ulp.
    let (rel, abs) = worst_diff(&c_gpu, &expected);
    for k in 0..c_gpu.len() {
        assert!(
            close(c_gpu[k], expected[k], 1e-4, 1e-4),
            "barlow C[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            c_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 5. random_mask  —  NUMERICAL bit-exact (Bernoulli LCG mask; BUG FIXED)
// ===========================================================================

/// Independent host re-derivation of the (fixed) `random_mask` inline LCG.
///
/// Mirrors the PTX exactly: `state = (seed ^ i) * M + A`, take the top 31 bits
/// (`state >> 33`), convert to f32 and divide by `2^31` to get a uniform in
/// `[0, 1)`. The integer < 2^31 may not fit f32's 24-bit mantissa, so the
/// `as f32` rounds round-to-nearest — exactly as the GPU's `cvt.rn.f32.u32`
/// does — and the `/2^31` is an exact power-of-two division, so `u` (and hence
/// the `u < drop_ratio` decision) is bit-identical to the device.
fn random_mask_uniform(i: u32, seed: u64) -> f32 {
    const M: u64 = 6_364_136_223_846_793_005;
    const A: u64 = 1_442_695_040_888_963_407;
    let state = (seed ^ u64::from(i)).wrapping_mul(M).wrapping_add(A);
    let r = (state >> 33) as u32;
    (r as f32) / 2_147_483_648.0_f32
}

#[test]
fn random_mask_matches_host_lcg() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 512_usize;
    let drop_ratio = 0.6_f32;
    let seed = 0x0123_4567_89AB_CDEF_u64;

    // Host reference (bit-exact). With the *0.5 bug present, every u < 0.5 so the
    // fraction of dropped patches would be ~min(1, 2*drop_ratio); after the fix
    // it is ~drop_ratio. We check the exact per-element mask, not just the rate.
    let mask_host: Vec<f32> = (0..n)
        .map(|i| {
            let u = random_mask_uniform(i as u32, seed);
            if u < drop_ratio { 0.0 } else { 1.0 }
        })
        .collect();

    let ptx = crate::ptx_kernels::random_mask_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "random_mask_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mask = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_mask");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_mask.as_device_ptr(), n as u32, drop_ratio, seed),
        )
        .expect("launch random_mask_kernel");
    stream.synchronize().expect("sync");

    let mut mask_gpu = vec![0.0_f32; n];
    d_mask.copy_to_host(&mut mask_gpu).expect("copy mask");

    // Every entry must be a binary mask, and bit-exact vs the host LCG.
    let mut dropped = 0usize;
    for (k, &m) in mask_gpu.iter().enumerate() {
        assert!(m == 0.0 || m == 1.0, "random_mask[{k}] = {m} not binary");
        if m == 0.0 {
            dropped += 1;
        }
        assert_eq!(
            m.to_bits(),
            mask_host[k].to_bits(),
            "random_mask[{k}] mismatch: gpu={m} host={}",
            mask_host[k]
        );
    }

    // Regression guard for the fixed *0.5 bug: with a correct [0,1) uniform the
    // drop fraction must track drop_ratio (0.6). The buggy [0,0.5) uniform would
    // have driven this to ~1.0 (every draw < 0.6). 512 samples keep the binomial
    // spread tight enough that a 0.45..0.75 window cannot be reached by the bug.
    let frac = dropped as f32 / n as f32;
    assert!(
        (0.45..0.75).contains(&frac),
        "random_mask drop fraction {frac} off expected ~{drop_ratio} (bug regression?)"
    );
}

// ===========================================================================
// 6. cosine_similarity  —  NUMERICAL (sim[k] = dot(a[k,*], b[k,*]))
// ===========================================================================

#[test]
fn cosine_similarity_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k_pairs = 8_usize;
    let dim_d = 16_usize;

    let mut rng = LcgRng::new(0xC0_51_4E_05);
    let a: Vec<f32> = (0..k_pairs * dim_d)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let b: Vec<f32> = (0..k_pairs * dim_d)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // Host reference: sim[k] = sum_d a[k,d] * b[k,d] (kernel assumes pre-normed
    // inputs; the dot product is the exact op it performs).
    let expected: Vec<f32> = (0..k_pairs)
        .map(|k| {
            (0..dim_d)
                .map(|d| a[k * dim_d + d] * b[k * dim_d + d])
                .sum()
        })
        .collect();

    let ptx = crate::ptx_kernels::cosine_similarity_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cosine_similarity_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<f32>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; k_pairs]).expect("d_out");

    // Grid = (K) blocks (ctaid.x = pair k), block = (D) threads (tid.x = dim d).
    // The kernel has no K param, so the grid must be exactly K blocks.
    let params = LaunchParams::new(k_pairs as u32, dim_d as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_out.as_device_ptr(),
                dim_d as u32,
            ),
        )
        .expect("launch cosine_similarity_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; k_pairs];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Atomic accumulation over dim_d terms reorders the f32 adds; a few ulp.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..k_pairs {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-4),
            "cosine_similarity sim[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 7. gather_features  —  NUMERICAL bit-exact (out[k,d] = queue[idx[k], d])
// ===========================================================================

#[test]
fn gather_features_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let vocab = 10_usize;
    let k_pairs = 6_usize;
    let dim_d = 8_usize;

    let mut rng = LcgRng::new(0x006A_7E05);
    let queue: Vec<f32> = (0..vocab * dim_d)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let idx: Vec<u32> = (0..k_pairs).map(|_| rng.next_usize(vocab) as u32).collect();

    // Host reference: out[k,d] = queue[idx[k]*D + d] (a pure gather, bit-exact).
    let mut expected = vec![0.0_f32; k_pairs * dim_d];
    for k in 0..k_pairs {
        for d in 0..dim_d {
            expected[k * dim_d + d] = queue[idx[k] as usize * dim_d + d];
        }
    }

    let ptx = crate::ptx_kernels::gather_features_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gather_features_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_queue = DeviceBuffer::<f32>::from_host(&queue).expect("d_queue");
    let d_idx = DeviceBuffer::<u32>::from_host(&idx).expect("d_idx");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; k_pairs * dim_d]).expect("d_out");

    // Grid = (K) blocks (ctaid.x = k), block = (D) threads (tid.x = d).
    let params = LaunchParams::new(k_pairs as u32, dim_d as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_queue.as_device_ptr(),
                d_idx.as_device_ptr(),
                d_out.as_device_ptr(),
                k_pairs as u32,
                dim_d as u32,
            ),
        )
        .expect("launch gather_features_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; k_pairs * dim_d];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // A pure memory gather: every element is bit-exact.
    for k in 0..out_gpu.len() {
        assert_eq!(
            out_gpu[k].to_bits(),
            expected[k].to_bits(),
            "gather_features out[{k}] mismatch: gpu={} host={}",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 8. momentum_update_f16  —  NUMERICAL bit-exact (f16 storage, f32 EMA blend)
// ===========================================================================

#[test]
fn momentum_update_f16_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Conditioning: momentum = 0.5 and both buffers hold even integers, so the
    // f32 blend `0.5*target + 0.5*online = (target+online)/2` is an integer that
    // is exactly representable in f16. The GPU's `cvt.rn.f16.f32` therefore never
    // rounds and the stored half is bit-identical to the host reference.
    let n = 96_usize;
    let momentum = 0.5_f32;

    let mut rng = LcgRng::new(0x00F1_6E05);
    let target_f: Vec<f32> = (0..n)
        .map(|_| f32::from(2 * (rng.next_usize(60) as u16)))
        .collect();
    let online_f: Vec<f32> = (0..n)
        .map(|_| f32::from(2 * (rng.next_usize(60) as u16)))
        .collect();

    // Guard: every value (and the blended result) is exactly f16-representable.
    for k in 0..n {
        assert_eq!(f16_to_f32(f16_from_exact(target_f[k])), target_f[k]);
        assert_eq!(f16_to_f32(f16_from_exact(online_f[k])), online_f[k]);
    }

    let target_h: Vec<u16> = target_f.iter().map(|&v| f16_from_exact(v)).collect();
    let online_h: Vec<u16> = online_f.iter().map(|&v| f16_from_exact(v)).collect();

    // Host reference: result = m*target + (1-m)*online (an exact integer here).
    let expected: Vec<f32> = target_f
        .iter()
        .zip(&online_f)
        .map(|(&t, &o)| momentum * t + (1.0 - momentum) * o)
        .collect();

    let ptx = crate::ptx_kernels::momentum_update_f16_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "momentum_update_f16_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_target = DeviceBuffer::<u16>::from_host(&target_h).expect("d_target");
    let d_online = DeviceBuffer::<u16>::from_host(&online_h).expect("d_online");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_target.as_device_ptr(),
                d_online.as_device_ptr(),
                n as u32,
                momentum,
            ),
        )
        .expect("launch momentum_update_f16_kernel");
    stream.synchronize().expect("sync");

    let mut target_gpu = vec![0_u16; n];
    d_target.copy_to_host(&mut target_gpu).expect("copy target");

    for k in 0..n {
        let g = f16_to_f32(target_gpu[k]);
        assert_eq!(
            g.to_bits(),
            expected[k].to_bits(),
            "momentum_f16 [{k}] mismatch: gpu={g} host={} (t={} o={})",
            expected[k],
            target_f[k],
            online_f[k]
        );
    }
}

// ===========================================================================
// 9. byol_cosine_loss_bf16  —  NUMERICAL bit-exact (bf16 storage, f32 accum)
// ===========================================================================

#[test]
fn byol_cosine_loss_bf16_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Conditioning: p, z drawn from {0, 1, 2} (all bf16-exact integers). Each
    // term `2 - 2*p*z` is then an integer in {2, 0, -2, -6}, and the running sum
    // of n such integers stays well below 2^24, so it is exactly representable in
    // f32 regardless of atomic-add ordering — making the comparison bit-exact.
    let n = 64_usize;
    let mut rng = LcgRng::new(0xB7_BF_16_05);
    let levels = [0.0_f32, 1.0, 2.0];
    let p: Vec<f32> = (0..n).map(|_| levels[rng.next_usize(3)]).collect();
    let z: Vec<f32> = (0..n).map(|_| levels[rng.next_usize(3)]).collect();

    // Guard: bf16 round-trip is lossless for these values.
    for k in 0..n {
        assert_eq!(bf16_to_f32(bf16_from_exact(p[k])), p[k]);
        assert_eq!(bf16_to_f32(bf16_from_exact(z[k])), z[k]);
    }

    let p_h: Vec<u16> = p.iter().map(|&v| bf16_from_exact(v)).collect();
    let z_h: Vec<u16> = z.iter().map(|&v| bf16_from_exact(v)).collect();

    // Host reference: exact integer sum of (2 - 2*p*z).
    let expected: f32 = p.iter().zip(&z).map(|(&pi, &zi)| 2.0 - 2.0 * pi * zi).sum();

    let ptx = crate::ptx_kernels::byol_cosine_loss_bf16_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "byol_cosine_loss_bf16_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<u16>::from_host(&p_h).expect("d_p");
    let d_z = DeviceBuffer::<u16>::from_host(&z_h).expect("d_z");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_z.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch byol_cosine_loss_bf16_kernel");
    stream.synchronize().expect("sync");

    let mut out = [0.0_f32];
    d_out.copy_to_host(&mut out).expect("copy out");

    assert_eq!(
        out[0].to_bits(),
        expected.to_bits(),
        "byol_cosine_loss_bf16 sum mismatch: gpu={} host={}",
        out[0],
        expected
    );
}

// ===========================================================================
// 10-12. Architecture-deepening kernels  —  LOAD + STRUCTURAL (sm_86 fallback)
// ===========================================================================
//
// The wgmma / redux.sync / TMA fast paths require Hopper (sm_90) or Blackwell
// (sm_100). On the A4000 (sm_86) each `*_ptx` dispatcher emits its portable
// scalar fallback, byte-identical to the already-validated base kernel. These
// tests confirm the dispatch fell back correctly, JIT-load the sm_86 output on
// the real device, and launch it once to prove on-device execution. The
// advanced ISA path is reported as scope-excluded (not runnable here), never
// claimed as validated.

#[test]
fn barlow_cross_corr_wgmma_falls_back_and_runs_on_sm86() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = crate::ptx_kernels::barlow_cross_corr_wgmma_ptx(fx.sm);
    // sm_86 must NOT emit the Hopper warp-group MMA path.
    assert!(
        !ptx.contains("wgmma"),
        "sm_{} unexpectedly emitted wgmma (not runnable on Ampere)",
        fx.sm
    );
    let kernel = load_kernel(&ptx, "barlow_cross_corr_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let batch_n = 8_usize;
    let dim_d = 4_usize;
    let za = vec![0.5_f32; batch_n * dim_d];
    let zb = vec![0.25_f32; batch_n * dim_d];
    let d_za = DeviceBuffer::<f32>::from_host(&za).expect("d_za");
    let d_zb = DeviceBuffer::<f32>::from_host(&zb).expect("d_zb");
    let d_c = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; dim_d * dim_d]).expect("d_c");

    let params = LaunchParams::new((dim_d as u32, dim_d as u32), (batch_n as u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_za.as_device_ptr(),
                d_zb.as_device_ptr(),
                d_c.as_device_ptr(),
                batch_n as u32,
                dim_d as u32,
            ),
        )
        .expect("launch barlow_cross_corr_kernel (fallback)");
    stream.synchronize().expect("sync");

    let mut c_gpu = vec![0.0_f32; dim_d * dim_d];
    d_c.copy_to_host(&mut c_gpu).expect("copy c");
    // The scalar fallback is the validated kernel: C[i,j] = N * 0.5 * 0.25.
    let expected = batch_n as f32 * 0.5 * 0.25;
    for (k, &v) in c_gpu.iter().enumerate() {
        assert!(
            close(v, expected, 1e-5, 1e-5),
            "barlow fallback C[{k}] = {v}, expected {expected}"
        );
    }
}

#[test]
fn nt_xent_softmax_warp_falls_back_and_runs_on_sm86() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = crate::ptx_kernels::nt_xent_softmax_warp_ptx(fx.sm);
    // sm_86 must NOT emit the Hopper redux.sync reductions.
    assert!(
        !ptx.contains("redux.sync"),
        "sm_{} unexpectedly emitted redux.sync (not runnable on Ampere)",
        fx.sm
    );
    let kernel = load_kernel(&ptx, "nt_xent_softmax_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let n2 = 8_usize;
    let inv_temp = 2.0_f32;
    let sim: Vec<f32> = (0..n2 * n2).map(|i| (i as f32) * 0.01 - 0.3).collect();
    let d_sim = DeviceBuffer::<f32>::from_host(&sim).expect("d_sim");

    let params = LaunchParams::new(n2 as u32, n2 as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_sim.as_device_ptr(), n2 as u32, inv_temp),
        )
        .expect("launch nt_xent_softmax_kernel (fallback)");
    stream.synchronize().expect("sync");

    let mut out = vec![0.0_f32; n2 * n2];
    d_sim.copy_to_host(&mut out).expect("copy sim");
    // Fallback masked-scale: diagonal -INF, off-diagonal = sim * inv_temp.
    for i in 0..n2 {
        for j in 0..n2 {
            let g = out[i * n2 + j];
            if i == j {
                assert!(g.is_infinite() && g < 0.0, "fallback diag [{i},{j}] = {g}");
            } else {
                assert!(
                    close(g, sim[i * n2 + j] * inv_temp, 1e-5, 1e-6),
                    "fallback [{i},{j}] = {g}"
                );
            }
        }
    }
}

#[test]
fn gather_features_bulk_falls_back_and_runs_on_sm86() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx = crate::ptx_kernels::gather_features_bulk_ptx(fx.sm);
    // sm_86 must NOT emit the Blackwell TMA bulk-tensor copy.
    assert!(
        !ptx.contains("cp.async.bulk.tensor"),
        "sm_{} unexpectedly emitted cp.async.bulk.tensor (not runnable on Ampere)",
        fx.sm
    );
    let kernel = load_kernel(&ptx, "gather_features_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let vocab = 5_usize;
    let k_pairs = 3_usize;
    let dim_d = 4_usize;
    let queue: Vec<f32> = (0..vocab * dim_d).map(|i| i as f32).collect();
    let idx: Vec<u32> = vec![4, 1, 2];
    let d_queue = DeviceBuffer::<f32>::from_host(&queue).expect("d_queue");
    let d_idx = DeviceBuffer::<u32>::from_host(&idx).expect("d_idx");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; k_pairs * dim_d]).expect("d_out");

    let params = LaunchParams::new(k_pairs as u32, dim_d as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_queue.as_device_ptr(),
                d_idx.as_device_ptr(),
                d_out.as_device_ptr(),
                k_pairs as u32,
                dim_d as u32,
            ),
        )
        .expect("launch gather_features_kernel (fallback)");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; k_pairs * dim_d];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");
    for k in 0..k_pairs {
        for d in 0..dim_d {
            let expected = queue[idx[k] as usize * dim_d + d];
            assert_eq!(
                out_gpu[k * dim_d + d].to_bits(),
                expected.to_bits(),
                "gather fallback out[{k},{d}] mismatch"
            );
        }
    }
}
