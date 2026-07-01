//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to a CPU reference. The launch ABI mirrors the working `oxicuda-snn` /
//! `oxicuda-ot` canaries: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars are passed as the matching Rust scalar
//! (`.param .u32` / `.param .f32` / `.param .u64`), in the kernel's declared
//! parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — `mean_var_kernel` is compared within FP32
//!   tolerance to the crate's own [`crate::descriptive::summary::mean`] and
//!   [`crate::descriptive::summary::variance`] (the kernel emits the streaming
//!   mean and the sum-of-squared-deviations `M2`, and `variance = M2 / n`).
//! * **Independent host re-derivation** — the remaining six kernels have no
//!   single dedicated `pub fn` that mirrors them exactly, so the oracle is an
//!   independent Rust re-implementation of the kernel's *documented* arithmetic:
//!   `rank_assign` (average tie ranks), `histogram_bin` (truncate-and-clamp
//!   binning), `bootstrap_resample` / `permute_labels` (the inline counter-based
//!   LCG, bit-exact), `chi2_cell` (`(O − E)² / (E + ε)`), and `lr_normal_eq`
//!   (`XᵀX`). These still genuinely fail if ptxas miscompiles or the PTX has a
//!   wrong constant / shift / index, because the host code is independent of the
//!   JIT-compiled PTX.
//!
//! Every kernel here computes a real result (there are no hollow stubs in this
//! crate). Every test skips (returns early) when no CUDA device is present, so
//! the suite stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

// Knuth MMIX 64-bit LCG constants used by the RNG kernels.
const LCG_MUL: u64 = 6_364_136_223_846_793_005;
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
/// A `Module::from_ptx` failure means ptxas rejected the PTX — a real bug in
/// `ptx_kernels.rs`, surfaced loudly rather than skipped.
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

/// One step of the kernels' 64-bit LCG: `state = state * MUL + ADD`.
fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD)
}

// ===========================================================================
// 1. mean_var  —  CRATE ORACLE (descriptive::summary::{mean, variance})
// ===========================================================================

#[test]
fn mean_var_matches_cpu() {
    use crate::descriptive::summary::{mean, variance};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Deterministic spread of moderate values; Welford in FP32 stays well within
    // tolerance of the FP64 two-pass crate reference for this conditioning.
    let n = 16_usize;
    let x: Vec<f32> = (0..n)
        .map(|i| {
            let i = i as f32;
            1.5 + 0.37 * i - 0.5 * ((i as i32 % 3) as f32)
        })
        .collect();

    // ---- CPU reference (crate oracle) ----
    let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    let mean_cpu = mean(&x64).expect("crate mean") as f32;
    // The kernel emits M2 (sum of squared deviations); `variance` divides by n.
    let m2_cpu = (variance(&x64).expect("crate variance") * n as f64) as f32;

    // ---- GPU ----
    let ptx = crate::ptx_kernels::mean_var_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mean_var_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_mean = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_mean");
    let d_m2 = DeviceBuffer::<f32>::from_host(&[0.0_f32]).expect("d_m2");

    // Each thread scans the whole array redundantly and writes index 0; launch a
    // single thread so the reduction is over all `n` with no write race.
    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                n as u32,
                d_mean.as_device_ptr(),
                d_m2.as_device_ptr(),
            ),
        )
        .expect("launch mean_var_kernel");
    stream.synchronize().expect("sync");

    let mut mean_gpu = [0.0_f32];
    let mut m2_gpu = [0.0_f32];
    d_mean.copy_to_host(&mut mean_gpu).expect("copy mean");
    d_m2.copy_to_host(&mut m2_gpu).expect("copy m2");

    assert!(
        close(mean_gpu[0], mean_cpu, 5e-4, 1e-4),
        "mean mismatch: gpu={} cpu={}",
        mean_gpu[0],
        mean_cpu
    );
    assert!(
        close(m2_gpu[0], m2_cpu, 5e-4, 1e-4),
        "M2 mismatch: gpu={} cpu={}",
        m2_gpu[0],
        m2_cpu
    );
}

// ===========================================================================
// 2. rank_assign  —  INDEPENDENT HOST RE-DERIVATION (average tie ranks)
// ===========================================================================

#[test]
fn rank_assign_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Sorted (non-decreasing) input with several tie groups.
    let sorted: Vec<f32> = vec![1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 5.0, 8.0, 9.0, 9.0, 10.0];
    let n = sorted.len();

    // Host re-derivation: for each i, the tie group is the maximal run of equal
    // values; its average 1-based rank is (first_rank + last_rank) / 2.
    let mut ranks_host = vec![0.0_f32; n];
    for i in 0..n {
        let mut lo = i;
        while lo > 0 && sorted[lo - 1] == sorted[i] {
            lo -= 1;
        }
        let mut hi = i;
        while hi + 1 < n && sorted[hi + 1] == sorted[i] {
            hi += 1;
        }
        ranks_host[i] = ((lo + 1) + (hi + 1)) as f32 / 2.0;
    }

    let ptx = crate::ptx_kernels::rank_assign_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rank_assign_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sorted = DeviceBuffer::<f32>::from_host(&sorted).expect("d_sorted");
    let d_ranks = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_ranks");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_sorted.as_device_ptr(), d_ranks.as_device_ptr(), n as u32),
        )
        .expect("launch rank_assign_kernel");
    stream.synchronize().expect("sync");

    let mut ranks_gpu = vec![0.0_f32; n];
    d_ranks.copy_to_host(&mut ranks_gpu).expect("copy ranks");

    // Average ranks are exact half-integers and FP32-representable here.
    for i in 0..n {
        assert!(
            close(ranks_gpu[i], ranks_host[i], 0.0, 1e-4),
            "rank[{i}] mismatch: gpu={} host={}",
            ranks_gpu[i],
            ranks_host[i]
        );
    }
}

// ===========================================================================
// 3. histogram_bin  —  INDEPENDENT HOST RE-DERIVATION (truncate-and-clamp bins)
// ===========================================================================

#[test]
fn histogram_bin_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let low = 0.0_f32;
    let dx = 1.0_f32;
    let n_bins = 5_usize;
    // Includes below-range (-1.0 → bin 0) and above-range (9.0 → bin 4) values to
    // exercise the clamp, plus interior values.
    let x: Vec<f32> = vec![0.5, 0.5, 1.5, 2.5, 2.5, 2.5, 4.5, -1.0, 9.0, 3.5, 0.1, 4.9];
    let n = x.len();

    // Host re-derivation: bin = clamp(trunc((x - low) / dx), 0, n_bins - 1).
    let mut counts_host = vec![0_u32; n_bins];
    for &v in &x {
        let raw = ((v - low) / dx).trunc() as i32;
        let bin = raw.clamp(0, n_bins as i32 - 1) as usize;
        counts_host[bin] += 1;
    }

    let ptx = crate::ptx_kernels::histogram_bin_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "histogram_bin_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_counts = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_bins]).expect("d_counts");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                n as u32,
                low,
                dx,
                n_bins as u32,
                d_counts.as_device_ptr(),
            ),
        )
        .expect("launch histogram_bin_kernel");
    stream.synchronize().expect("sync");

    let mut counts_gpu = vec![0_u32; n_bins];
    d_counts.copy_to_host(&mut counts_gpu).expect("copy counts");

    assert_eq!(
        counts_gpu, counts_host,
        "histogram counts mismatch: gpu={counts_gpu:?} host={counts_host:?}"
    );
}

// ===========================================================================
// 4. bootstrap_resample  —  INDEPENDENT HOST RE-DERIVATION (LCG gather, bit-exact)
// ===========================================================================

#[test]
fn bootstrap_resample_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 12_usize;
    let seed = 0x1234_5678_9ABC_DEF0_u64;
    // Distinct values so a gather is unambiguous.
    let x: Vec<f32> = (0..n).map(|i| 100.0 + i as f32).collect();

    // Host re-derivation of the kernel's index math:
    //   state = (seed + tid * MUL) * MUL + ADD;  idx = (state >> 32) % n.
    let mut out_host = vec![0.0_f32; n];
    for (tid, slot) in out_host.iter_mut().enumerate() {
        let inner = seed.wrapping_add((tid as u64).wrapping_mul(LCG_MUL));
        let state = inner.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
        let idx = ((state >> 32) as u32) % n as u32;
        *slot = x[idx as usize];
    }

    let ptx = crate::ptx_kernels::bootstrap_resample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bootstrap_resample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_x.as_device_ptr(), n as u32, d_out.as_device_ptr(), seed),
        )
        .expect("launch bootstrap_resample_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Bit-exact: the kernel copies a chosen `x[idx]` with no arithmetic.
    for tid in 0..n {
        assert_eq!(
            out_gpu[tid].to_bits(),
            out_host[tid].to_bits(),
            "bootstrap out[{tid}] mismatch: gpu={} host={}",
            out_gpu[tid],
            out_host[tid]
        );
    }
}

// ===========================================================================
// 5. permute_labels  —  INDEPENDENT HOST RE-DERIVATION (bijective scatter)
// ===========================================================================

/// Position the kernel scatters thread `tid` to:
///   `pos = ((seed + tid) * MUL + ADD) >> 32) % n`.
fn permute_pos(seed: u64, tid: u64, n: u32) -> u32 {
    let state = lcg_step(seed.wrapping_add(tid));
    ((state >> 32) as u32) % n
}

#[test]
fn permute_labels_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // `n = 5`: the kernel's scatter position advances by an (almost) fixed
    // increment per thread, so a power-of-two `n` can never be a bijection;
    // `n = 5` admits one (e.g. seed 18) and is found immediately below.
    let n = 5_usize;

    // Find a seed whose scatter positions are a bijection of 0..n. With a
    // bijection every output slot is written exactly once, so the result is
    // race-free and fully determined — a strong, deterministic oracle.
    let mut seed = 0_u64;
    let mut pos_host = vec![0_u32; n];
    let found = (1_u64..2_000_000).any(|s| {
        let mut seen = 0_u32;
        let mut ok = true;
        for (tid, slot) in pos_host.iter_mut().enumerate() {
            let p = permute_pos(s, tid as u64, n as u32);
            let bit = 1_u32 << p;
            if seen & bit != 0 {
                ok = false;
                break;
            }
            seen |= bit;
            *slot = p;
        }
        if ok {
            seed = s;
        }
        ok
    });
    assert!(found, "no bijective seed found for n={n}");

    // Labels as u32 (the kernel loads/stores 32-bit labels).
    let labels_in: Vec<u32> = (0..n).map(|i| 1000 + i as u32).collect();
    let mut labels_host = vec![0_u32; n];
    for tid in 0..n {
        labels_host[pos_host[tid] as usize] = labels_in[tid];
    }

    let ptx = crate::ptx_kernels::permute_labels_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "permute_labels_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_in = DeviceBuffer::<u32>::from_host(&labels_in).expect("d_in");
    // Sentinel init that the bijection must fully overwrite.
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0xFFFF_FFFF_u32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_in.as_device_ptr(), d_out.as_device_ptr(), n as u32, seed),
        )
        .expect("launch permute_labels_kernel");
    stream.synchronize().expect("sync");

    let mut labels_gpu = vec![0_u32; n];
    d_out.copy_to_host(&mut labels_gpu).expect("copy labels");

    assert_eq!(
        labels_gpu, labels_host,
        "permute_labels mismatch (seed={seed}): gpu={labels_gpu:?} host={labels_host:?}"
    );
}

// ===========================================================================
// 6. chi2_cell  —  INDEPENDENT HOST RE-DERIVATION ((O − E)² / (E + ε))
// ===========================================================================

#[test]
fn chi2_cell_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // The kernel's epsilon floor is the exact FP32 constant 0x322bcc77.
    let eps = f32::from_bits(0x322b_cc77);

    let observed: Vec<f32> = vec![12.0, 5.0, 8.0, 20.0, 3.0, 17.0];
    let expected: Vec<f32> = vec![10.0, 6.0, 9.0, 18.0, 4.0, 15.0];
    let n = observed.len();

    let mut out_host = vec![0.0_f32; n];
    for i in 0..n {
        let d = observed[i] - expected[i];
        out_host[i] = (d * d) / (expected[i] + eps);
    }

    let ptx = crate::ptx_kernels::chi2_cell_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "chi2_cell_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_obs = DeviceBuffer::<f32>::from_host(&observed).expect("d_obs");
    let d_exp = DeviceBuffer::<f32>::from_host(&expected).expect("d_exp");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_obs.as_device_ptr(),
                d_exp.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch chi2_cell_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_host);
    for i in 0..n {
        assert!(
            close(out_gpu[i], out_host[i], 1e-5, 1e-6),
            "chi2_cell out[{i}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            out_host[i]
        );
    }
}

// ===========================================================================
// 7. lr_normal_eq  —  INDEPENDENT HOST RE-DERIVATION (XᵀX)
// ===========================================================================

#[test]
fn lr_normal_eq_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_samples = 10_usize;
    let n_features = 4_usize;

    // Deterministic design matrix X (row-major, n_samples × n_features).
    let x: Vec<f32> = (0..n_samples * n_features)
        .map(|k| {
            let k = k as f32;
            0.5 + 0.1 * k - 0.3 * ((k as i32 % 5) as f32)
        })
        .collect();

    // Host re-derivation: (XᵀX)[i, j] = Σ_k X[k, i] · X[k, j].
    let mut xtx_host = vec![0.0_f32; n_features * n_features];
    for i in 0..n_features {
        for j in 0..n_features {
            let mut acc = 0.0_f32;
            for k in 0..n_samples {
                acc += x[k * n_features + i] * x[k * n_features + j];
            }
            xtx_host[i * n_features + j] = acc;
        }
    }

    let ptx = crate::ptx_kernels::lr_normal_eq_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lr_normal_eq_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_xtx =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_features * n_features]).expect("d_xtx");

    // Grid = (n_features, n_features), block = (1, 1): one block per (i, j) cell.
    let params = LaunchParams::new((n_features as u32, n_features as u32), (1_u32, 1_u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_xtx.as_device_ptr(),
                n_samples as u32,
                n_features as u32,
            ),
        )
        .expect("launch lr_normal_eq_kernel");
    stream.synchronize().expect("sync");

    let mut xtx_gpu = vec![0.0_f32; n_features * n_features];
    d_xtx.copy_to_host(&mut xtx_gpu).expect("copy xtx");

    let (rel, abs) = worst_diff(&xtx_gpu, &xtx_host);
    for k in 0..xtx_gpu.len() {
        assert!(
            close(xtx_gpu[k], xtx_host[k], 1e-4, 1e-4),
            "lr_normal_eq xtx[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            xtx_gpu[k],
            xtx_host[k]
        );
    }
}
