//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to a CPU reference.  The
//! launch ABI mirrors `oxicuda-snn`: device buffers are passed as their
//! `CUdeviceptr` (`.param .u64`), scalars as the matching Rust scalar.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate / direct oracle** — compared within FP32 tolerance to an
//!   independent host re-computation of the *exact same formula* the kernel
//!   is documented to compute.  Applies to: `svdd_loss`, `recon_score`,
//!   `iforest_score`, `ensemble_normalize`, `abod_batch`, `fast_mcd_cstep`.
//! * **LOF reach-distance oracle** — `lof_reach_dist_kernel` had a
//!   **register-clobber bug** (see §Bug report below); after the fix the
//!   oracle is `max(kd_j, dist(x_i, data_j))` computed on the host.
//! * **Independent host re-derivation** — `copod_ecdf` (`−ln(ecdf+ε)` via
//!   `lg2.approx × ln 2`), `mahal_dist` (double-sum quadratic form).
//! * **Load + structural** — `fused_knn_lof` uses `shfl.sync.down.b32`
//!   (sm_80+) shared-memory warp reduction; the test verifies PTX loads and
//!   that `out[0] ≥ 0`, `out[1] ≥ out[0]`, and a brute-force host 1-NN
//!   matches.
//!
//! ## Bug report: `lof_reach_dist_kernel` register-clobber (fixed)
//!
//! **Root cause**: After loading `j` (neighbour index) into `%r13`, the
//! kernel immediately clobbered it with `mov.u32 %r13, 0` (dim-loop init).
//! The inner distance loop then used `i` (not `j`) for *both* address
//! computations, computing `dist(x_i, x_i) = 0` and returning
//! `reach_dist = kd_j` for every pair regardless of actual distance.
//!
//! **Fix**: Added `mov.u32 %r14, %r13` to save `j`, expanded the register
//! file from `%r<14>` to `%r<15>`, and changed the data_j address
//! calculation to use `%r14` (j) instead of `%r11` (i).
//!
//! Every test returns early (not fails) when no CUDA device is present, so
//! the suite stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::handle::LcgRng;

// ──────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ──────────────────────────────────────────────────────────────────────────────

/// A live CUDA context plus the device's SM version.
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

/// Relative-with-absolute-floor closeness for FP32 comparisons.
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

// ──────────────────────────────────────────────────────────────────────────────
// CPU oracle helpers (independent re-derivations; no crate internals exposed)
// ──────────────────────────────────────────────────────────────────────────────

/// `||phi_i − c||²` for every sample; mirrors `svdd_loss_kernel` exactly.
fn svdd_loss_host(phi: &[f32], center: &[f32], n: usize, rep_dim: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            (0..rep_dim)
                .map(|j| {
                    let d = phi[i * rep_dim + j] - center[j];
                    d * d
                })
                .sum::<f32>()
        })
        .collect()
}

/// `(1/d) · Σ_j (x_j − x̂_j)²` per sample; mirrors `recon_score_kernel`.
fn recon_score_host(x: &[f32], xhat: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let s: f32 = (0..d)
                .map(|j| {
                    let diff = x[i * d + j] - xhat[i * d + j];
                    diff * diff
                })
                .sum();
            s / d as f32
        })
        .collect()
}

/// `max(kd_j, dist(x_i, data_j))` per (i, ki) pair; post-fix oracle for
/// `lof_reach_dist_kernel`.
fn lof_reach_dist_host(
    x: &[f32],
    data: &[f32],
    knn_idx: &[u32],
    knn_dist: &[f32],
    n: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * k];
    for i in 0..n {
        for ki in 0..k {
            let j = knn_idx[i * k + ki] as usize;
            let kd_j = knn_dist[j];
            let sq: f32 = (0..d)
                .map(|dim| {
                    let diff = x[i * d + dim] - data[j * d + dim];
                    diff * diff
                })
                .sum();
            let dist = sq.sqrt();
            out[i * k + ki] = kd_j.max(dist);
        }
    }
    out
}

/// `−ln(ecdf + ε)` for left tail and `−ln(1 − ecdf + ε)` for right tail;
/// mirrors `copod_ecdf_kernel`.
fn copod_ecdf_host(ecdf: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let eps = 1.0e-10_f32;
    let mut left = vec![0.0_f32; ecdf.len()];
    let mut right = vec![0.0_f32; ecdf.len()];
    for (k, &e) in ecdf.iter().enumerate() {
        left[k] = -(e + eps).ln();
        right[k] = -(1.0_f32 - e + eps).ln();
    }
    (left, right)
}

/// `(x − μ)ᵀ Σ⁻¹ (x − μ)` per sample; independent re-derivation for both
/// `mahal_dist_kernel` and `fast_mcd_cstep_kernel`.
fn mahal_sq_host(x: &[f32], mean: &[f32], inv_cov: &[f32], n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let xi = &x[i * d..(i + 1) * d];
            let mut acc = 0.0_f32;
            for r in 0..d {
                let dr = xi[r] - mean[r];
                for c in 0..d {
                    let dc = xi[c] - mean[c];
                    acc += dr * inv_cov[r * d + c] * dc;
                }
            }
            acc
        })
        .collect()
}

/// `2^{−avg_path / c_n}` per sample; mirrors `iforest_score_kernel`.
fn iforest_score_host(avg_path: &[f32], c_n: f32) -> Vec<f32> {
    avg_path
        .iter()
        .map(|&p| {
            if c_n < 1.0e-8 {
                0.5_f32
            } else {
                (2.0_f32).powf(-p / c_n)
            }
        })
        .collect()
}

/// Min-max normalise per detector then average; mirrors `ensemble_normalize_kernel`.
fn ensemble_normalize_host(
    scores: &[f32],
    mins: &[f32],
    maxs: &[f32],
    n: usize,
    n_det: usize,
) -> Vec<f32> {
    let eps = 1.0e-8_f32;
    (0..n)
        .map(|i| {
            let s: f32 = (0..n_det)
                .map(|d| {
                    let sc = scores[i * n_det + d];
                    let mn = mins[d];
                    let mx = maxs[d];
                    let norm = (sc - mn) / (mx - mn + eps);
                    norm.clamp(0.0, 1.0)
                })
                .sum();
            s / n_det as f32
        })
        .collect()
}

/// Brute-force `1-NN` distance and reach-distance for a single query;
/// used as the `fused_knn_lof_kernel` structural oracle.
fn brute_force_1nn(
    query: &[f32],
    data: &[f32],
    knn_dist: &[f32],
    m: usize,
    d: usize,
) -> (f32, f32) {
    let mut best_sq = f32::MAX;
    let mut best_j = 0_usize;
    for j in 0..m {
        let sq: f32 = (0..d)
            .map(|dim| {
                let diff = query[dim] - data[j * d + dim];
                diff * diff
            })
            .sum();
        if sq < best_sq {
            best_sq = sq;
            best_j = j;
        }
    }
    let dist = best_sq.sqrt();
    let reach = knn_dist[best_j].max(dist);
    (dist, reach)
}

/// Streaming ABOF (angle-based outlier factor) per query; independent
/// re-derivation of `abod_batch_kernel`.
///
/// `out[q] = 1 / (Var_{pairs}[f(a,b)] + ε)` where
/// `f(a,b) = ⟨p−a, p−b⟩ / (‖p−a‖²·‖p−b‖² + ε)`.
fn abod_host(query: &[f32], data: &[f32], nq: usize, m: usize, d: usize) -> Vec<f32> {
    let eps = 1.0e-10_f32;
    (0..nq)
        .map(|q| {
            let p = &query[q * d..(q + 1) * d];
            let mut sum_f = 0.0_f32;
            let mut sum_f2 = 0.0_f32;
            let mut count = 0_u32;

            for a in 0..m {
                let da = &data[a * d..(a + 1) * d];
                for b in (a + 1)..m {
                    let db = &data[b * d..(b + 1) * d];

                    let mut dot = 0.0_f32;
                    let mut na2 = 0.0_f32;
                    let mut nb2 = 0.0_f32;
                    for dim in 0..d {
                        let pa = p[dim] - da[dim];
                        let pb = p[dim] - db[dim];
                        dot += pa * pb;
                        na2 += pa * pa;
                        nb2 += pb * pb;
                    }
                    let denom = na2 * nb2 + eps;
                    let f = dot / denom;
                    sum_f += f;
                    sum_f2 += f * f;
                    count += 1;
                }
            }

            if count == 0 {
                return 0.0_f32;
            }
            let cnt = count as f32;
            let mean_f = sum_f / cnt;
            let mean_f2 = sum_f2 / cnt;
            let variance = (mean_f2 - mean_f * mean_f).max(0.0);
            1.0_f32 / (variance + eps)
        })
        .collect()
}

// ===========================================================================
// 1. svdd_loss_kernel  —  DIRECT ORACLE (‖phi_i − c‖² per sample)
// ===========================================================================

#[test]
fn svdd_loss_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let rep_dim = 8_usize;
    let mut rng = LcgRng::new(0xDEAD_BEEF);
    let phi: Vec<f32> = (0..n * rep_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let center: Vec<f32> = (0..rep_dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    let expected = svdd_loss_host(&phi, &center, n, rep_dim);

    let ptx = crate::ptx_kernels::svdd_loss_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "svdd_loss_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_phi = DeviceBuffer::<f32>::from_host(&phi).expect("d_phi");
    let d_center = DeviceBuffer::<f32>::from_host(&center).expect("d_center");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_phi.as_device_ptr(),
                d_center.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                rep_dim as u32,
            ),
        )
        .expect("launch svdd_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-6),
            "svdd_loss[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 2. recon_score_kernel  —  DIRECT ORACLE ((1/d)·Σ(x−x̂)² per sample)
// ===========================================================================

#[test]
fn recon_score_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let d = 16_usize;
    let mut rng = LcgRng::new(0xC0DE_CAFE);
    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let xhat: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    let expected = recon_score_host(&x, &xhat, n, d);

    let ptx = crate::ptx_kernels::recon_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "recon_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_xhat = DeviceBuffer::<f32>::from_host(&xhat).expect("d_xhat");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_xhat.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch recon_score_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-6),
            "recon_score[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 3. lof_reach_dist_kernel  —  POST-FIX ORACLE (max(kd_j, dist(x_i, data_j)))
//
// BUG FOUND AND FIXED (see module doc and ptx_kernels.rs):
//   Original code: `%r13` held j then was overwritten by `mov.u32 %r13, 0`
//   (dim-loop init).  Inner loop used `i` for both x_i and data_j address,
//   computing dist(x_i, x_i) = 0 always → reach_dist = kd_j always.
//   Fix: save j to %r14, expand .reg .u32 %r<14> → %r<15>.
// ===========================================================================

#[test]
fn lof_reach_dist_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Small but representative: 8 queries, 12 training points, k=3, d=4.
    let n = 8_usize; // queries
    let m = 12_usize; // training
    let k = 3_usize; // neighbours
    let d = 4_usize;

    let mut rng = LcgRng::new(0xABCD_1234);
    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let data: Vec<f32> = (0..m * d).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // Build knn_idx: for each query i, pick k distinct training indices in [0,m).
    // For simplicity use a deterministic cycling scheme.
    let mut knn_idx = vec![0_u32; n * k];
    for i in 0..n {
        for ki in 0..k {
            knn_idx[i * k + ki] = ((i * 3 + ki * 4 + 1) % m) as u32;
        }
    }

    // knn_dist[j] = k-distance of training point j.  Use a fixed small positive value
    // that creates interesting reachability (some distances < kd_j, some > kd_j).
    let knn_dist: Vec<f32> = (0..m).map(|j| 0.3_f32 + 0.05 * j as f32).collect();

    let expected = lof_reach_dist_host(&x, &data, &knn_idx, &knn_dist, n, k, d);

    let ptx = crate::ptx_kernels::lof_reach_dist_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lof_reach_dist_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");
    let d_knn_idx = DeviceBuffer::<u32>::from_host(&knn_idx).expect("d_knn_idx");
    let d_knn_dist = DeviceBuffer::<f32>::from_host(&knn_dist).expect("d_knn_dist");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * k]).expect("d_out");

    // One thread per (i, ki) pair; total = n*k.
    let total = (n * k) as u32;
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_data.as_device_ptr(),
                d_knn_idx.as_device_ptr(),
                d_knn_dist.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                k as u32,
                d as u32,
            ),
        )
        .expect("launch lof_reach_dist_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n * k];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Sanity: reach_dist must be >= kd_j (by definition of reachability distance).
    for (pair_idx, &rd) in out_gpu.iter().enumerate() {
        let i = pair_idx / k;
        let ki = pair_idx % k;
        let j = knn_idx[i * k + ki] as usize;
        let kd_j = knn_dist[j];
        assert!(
            rd >= kd_j - 1e-5,
            "reach_dist[{pair_idx}] = {rd} < kd_j = {kd_j}"
        );
    }

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for pair_idx in 0..out_gpu.len() {
        assert!(
            close(out_gpu[pair_idx], expected[pair_idx], 1e-4, 1e-6),
            "lof_reach_dist[{pair_idx}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            out_gpu[pair_idx],
            expected[pair_idx]
        );
    }
}

// ===========================================================================
// 4. copod_ecdf_kernel  —  INDEPENDENT HOST RE-DERIVATION (−ln(ecdf+ε))
//
// The kernel uses `lg2.approx.f32` × LN_2 to compute natural log.
// Scaling: log2(x) × ln(2) = ln(x).  No base-2 bug: LN_2 = 0.693… is the
// correct factor (not a missing one like the ex2/log2(e) pitfall).
// ===========================================================================

#[test]
fn copod_ecdf_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let d = 8_usize;
    let total = n * d;

    let mut rng = LcgRng::new(0x5ECF_ECDF);
    // ecdf values in (0.05, 0.95) to avoid log(≈0) extremes that stress
    // `lg2.approx.f32` accuracy.
    let ecdf: Vec<f32> = (0..total).map(|_| 0.05 + 0.90 * rng.next_f32()).collect();

    let (left_cpu, right_cpu) = copod_ecdf_host(&ecdf);

    let ptx = crate::ptx_kernels::copod_ecdf_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "copod_ecdf_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ecdf = DeviceBuffer::<f32>::from_host(&ecdf).expect("d_ecdf");
    let d_left = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_left");
    let d_right = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; total]).expect("d_right");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ecdf.as_device_ptr(),
                d_left.as_device_ptr(),
                d_right.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch copod_ecdf_kernel");
    stream.synchronize().expect("sync");

    let mut left_gpu = vec![0.0_f32; total];
    let mut right_gpu = vec![0.0_f32; total];
    d_left.copy_to_host(&mut left_gpu).expect("copy left");
    d_right.copy_to_host(&mut right_gpu).expect("copy right");

    // `lg2.approx.f32` has ~2 ulp error in [0.5, 2]; for ecdf in (0.05, 0.95)
    // the argument `ecdf + eps ≈ ecdf` stays in a moderate range.  1e-3 relative
    // covers the approximation error with comfortable margin.
    let (rel_l, abs_l) = worst_diff(&left_gpu, &left_cpu);
    let (rel_r, abs_r) = worst_diff(&right_gpu, &right_cpu);
    for k in 0..total {
        assert!(
            close(left_gpu[k], left_cpu[k], 1e-3, 1e-5),
            "copod_ecdf left[{k}] mismatch: gpu={} cpu={} (worst rel={rel_l:e} abs={abs_l:e})",
            left_gpu[k],
            left_cpu[k]
        );
        assert!(
            close(right_gpu[k], right_cpu[k], 1e-3, 1e-5),
            "copod_ecdf right[{k}] mismatch: gpu={} cpu={} (worst rel={rel_r:e} abs={abs_r:e})",
            right_gpu[k],
            right_cpu[k]
        );
    }
}

// ===========================================================================
// 5. mahal_dist_kernel  —  INDEPENDENT HOST RE-DERIVATION
//    ((x−μ)ᵀ Σ⁻¹ (x−μ) double-sum quadratic form)
// ===========================================================================

#[test]
fn mahal_dist_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 32_usize;
    let d = 3_usize; // small d keeps the quadratic form host oracle fast
    let mut rng = LcgRng::new(0xB1A5_E77E);

    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let mean: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.5).collect();

    // Build a symmetric positive-definite inv_cov: D + small off-diagonal.
    let mut inv_cov = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            inv_cov[i * d + j] = if i == j {
                2.0 + rng.next_f32()
            } else {
                0.05 * rng.next_f32()
            };
        }
        // Symmetrise row i vs col i.
        for j in 0..i {
            let avg = (inv_cov[i * d + j] + inv_cov[j * d + i]) * 0.5;
            inv_cov[i * d + j] = avg;
            inv_cov[j * d + i] = avg;
        }
    }

    let expected = mahal_sq_host(&x, &mean, &inv_cov, n, d);

    let ptx = crate::ptx_kernels::mahal_dist_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mahal_dist_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_mean = DeviceBuffer::<f32>::from_host(&mean).expect("d_mean");
    let d_inv_cov = DeviceBuffer::<f32>::from_host(&inv_cov).expect("d_inv_cov");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_mean.as_device_ptr(),
                d_inv_cov.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch mahal_dist_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // The double-loop accumulates d² FMAs; with d=3 that is 9 terms.
    // The maximum relative FP32 rounding across 9 fused adds is ~4e-6;
    // 1e-3 is generous while still catching a wrong kernel formula.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-3, 1e-5),
            "mahal_dist[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 6. iforest_score_kernel  —  DIRECT ORACLE (2^{−avg_path/c_n})
//
// `ex2.approx.f32` computes 2^x directly (base-2), which is the CORRECT
// instruction here — no ln(2) scaling is needed because the Isolation Forest
// formula is base-2 by definition.  The base-2 bug class applies to kernels
// that want e^x and wrongly use ex2 without × log2(e); this kernel is clean.
// ===========================================================================

#[test]
fn iforest_score_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 128_usize;
    let c_n = crate::isolation::iforest_score::c_factor(200); // representative

    let mut rng = LcgRng::new(0x1F0_CAFE);
    // avg_path in [0.5, 2·c_n]; random but bounded so 2^(−p/c_n) ∈ (0, 1).
    let avg_path: Vec<f32> = (0..n).map(|_| 0.5 + rng.next_f32() * 2.0 * c_n).collect();

    let expected = iforest_score_host(&avg_path, c_n);

    let ptx = crate::ptx_kernels::iforest_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "iforest_score_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_path = DeviceBuffer::<f32>::from_host(&avg_path).expect("d_path");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_path.as_device_ptr(), c_n, d_out.as_device_ptr(), n as u32),
        )
        .expect("launch iforest_score_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // `ex2.approx.f32` has ~2 ulp relative error; 2e-4 covers that.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 2e-4, 1e-7),
            "iforest_score[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 7. ensemble_normalize_kernel  —  DIRECT ORACLE (min-max → clamp → mean)
// ===========================================================================

#[test]
fn ensemble_normalize_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize;
    let n_det = 4_usize;
    let mut rng = LcgRng::new(0xE55E_AB1E);

    let scores: Vec<f32> = (0..n * n_det).map(|_| rng.next_f32() * 5.0).collect();
    // mins and maxs: ensure max > min for all detectors.
    let mins: Vec<f32> = (0..n_det).map(|_| rng.next_f32()).collect();
    let maxs: Vec<f32> = mins
        .iter()
        .map(|&mn| mn + 0.5 + rng.next_f32() * 3.0)
        .collect();

    let expected = ensemble_normalize_host(&scores, &mins, &maxs, n, n_det);

    let ptx = crate::ptx_kernels::ensemble_normalize_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ensemble_normalize_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_scores = DeviceBuffer::<f32>::from_host(&scores).expect("d_scores");
    let d_mins = DeviceBuffer::<f32>::from_host(&mins).expect("d_mins");
    let d_maxs = DeviceBuffer::<f32>::from_host(&maxs).expect("d_maxs");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_scores.as_device_ptr(),
                d_mins.as_device_ptr(),
                d_maxs.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                n_det as u32,
            ),
        )
        .expect("launch ensemble_normalize_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Output must be in [0, 1] by construction.
    for (k, &v) in out_gpu.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&v),
            "ensemble_normalize[{k}] = {v} not in [0,1]"
        );
    }

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-6),
            "ensemble_normalize[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 8. fused_knn_lof_kernel  —  LOAD + BRUTE-FORCE HOST ORACLE (sm_80+)
//
// Block-cooperative 1-NN reduction with `shfl.sync.down.b32`.  We provide an
// independent brute-force host 1-NN as oracle for `out[0]` and
// `max(knn_dist[nn], out[0])` for `out[1]`.
// ===========================================================================

#[test]
fn fused_knn_lof_brute_force_oracle() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // shfl.sync.down.b32 is sm_70+; the kernel docs say sm_80+ for the full
    // block-warp staging.  Skip on older hardware.
    if fx.sm < 80 {
        return;
    }

    let m = 16_usize; // reference points
    let d = 4_usize; // features
    let mut rng = LcgRng::new(0xF4EE_D500);

    let query: Vec<f32> = (0..d).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let data: Vec<f32> = (0..m * d).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let knn_dist: Vec<f32> = (0..m).map(|j| 0.1 + 0.05 * j as f32).collect();

    let (expected_dist, expected_reach) = brute_force_1nn(&query, &data, &knn_dist, m, d);

    // LOAD: a failure here is a real PTX/ptxas bug for this SM.
    let ptx = crate::ptx_kernels::fused_knn_lof_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fused_knn_lof_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_query = DeviceBuffer::<f32>::from_host(&query).expect("d_query");
    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");
    let d_knn_dist_buf = DeviceBuffer::<f32>::from_host(&knn_dist).expect("d_knn_dist");
    // out[0] = d(x, nn₁), out[1] = reach-dist
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32, 0.0_f32]).expect("d_out");

    // Block-cooperative: 1 block of 32 threads (1 warp) over m reference points.
    let block = 32_u32;
    let params = LaunchParams::new(1_u32, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_query.as_device_ptr(),
                d_data.as_device_ptr(),
                d_knn_dist_buf.as_device_ptr(),
                d_out.as_device_ptr(),
                m as u32,
                d as u32,
            ),
        )
        .expect("launch fused_knn_lof_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = [0.0_f32; 2];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Structural checks: both values must be non-negative and finite.
    assert!(
        out_gpu[0].is_finite() && out_gpu[0] >= 0.0,
        "fused_knn_lof out[0] = {} not finite/non-negative",
        out_gpu[0]
    );
    assert!(
        out_gpu[1].is_finite() && out_gpu[1] >= 0.0,
        "fused_knn_lof out[1] = {} not finite/non-negative",
        out_gpu[1]
    );
    // Reachability invariant: reach_dist >= actual_dist.
    assert!(
        out_gpu[1] >= out_gpu[0] - 1e-5,
        "fused_knn_lof reach ({}) < dist ({})",
        out_gpu[1],
        out_gpu[0]
    );

    // Oracle comparison: brute-force 1-NN.
    // shfl.sync.down.b32 is a bitwise integer shuffle then FP reinterpret;
    // the minimum comparison uses setp.lt.f32 — same FP32 precision as the host.
    assert!(
        close(out_gpu[0], expected_dist, 1e-4, 1e-6),
        "fused_knn_lof dist mismatch: gpu={} host={expected_dist}",
        out_gpu[0]
    );
    assert!(
        close(out_gpu[1], expected_reach, 1e-4, 1e-6),
        "fused_knn_lof reach mismatch: gpu={} host={expected_reach}",
        out_gpu[1]
    );
}

// ===========================================================================
// 9. abod_batch_kernel  —  INDEPENDENT HOST RE-DERIVATION (ABOF variance)
//
// Streaming `E[f²] − E[f]²` variance of reciprocal-distance-weighted
// inner products, per query batch.  The tolerance is wider (1e-2 relative)
// because the variance involves subtraction of nearly-equal quantities that
// amplifies FP rounding; the test still catches a wrong formula (the error
// from a missing `/denom` or the wrong eps placement is orders of magnitude
// larger than 1e-2).
// ===========================================================================

#[test]
fn abod_batch_matches_host_formula() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let nq = 8_usize; // queries
    let m = 6_usize; // reference points (pairs = 15 per query — tractable)
    let d = 3_usize;

    let mut rng = LcgRng::new(0xAB0D_CAFE);
    // Spread queries and data in [−2, 2]^d; avoid coincident points by
    // using a deterministic offset so na2 or nb2 never hits machine zero.
    let query: Vec<f32> = (0..nq * d).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let data: Vec<f32> = (0..m * d).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    let expected = abod_host(&query, &data, nq, m, d);

    let ptx = crate::ptx_kernels::abod_batch_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "abod_batch_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_query = DeviceBuffer::<f32>::from_host(&query).expect("d_query");
    let d_data = DeviceBuffer::<f32>::from_host(&data).expect("d_data");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; nq]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(nq as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_query.as_device_ptr(),
                d_data.as_device_ptr(),
                d_out.as_device_ptr(),
                nq as u32,
                m as u32,
                d as u32,
            ),
        )
        .expect("launch abod_batch_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; nq];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Scores must be finite and positive (1 / (variance + eps)).
    for (k, &s) in out_gpu.iter().enumerate() {
        assert!(
            s.is_finite() && s > 0.0,
            "abod_batch[{k}] = {s} not finite/positive"
        );
    }

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..nq {
        // 1e-2 relative: variance subtraction amplifies FP32 rounding but a
        // wrong formula produces O(1) relative error — orders of magnitude larger.
        assert!(
            close(out_gpu[k], expected[k], 1e-2, 1e-4),
            "abod_batch[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 10. fast_mcd_cstep_kernel  —  INDEPENDENT HOST RE-DERIVATION
//     (same double-sum quadratic form as mahal_dist, different kernel)
// ===========================================================================

#[test]
fn fast_mcd_cstep_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 32_usize;
    let d = 4_usize;
    let mut rng = LcgRng::new(0xFA57_C07E);

    let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32() * 6.0 - 3.0).collect();
    let mean: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.5 - 0.25).collect();

    // SPD inv_cov: diagonal-dominant with small off-diagonal.
    let mut inv_cov = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            inv_cov[i * d + j] = if i == j {
                1.5 + rng.next_f32()
            } else {
                0.03 * (rng.next_f32() - 0.5)
            };
        }
        // Symmetrise.
        for j in 0..i {
            let avg = (inv_cov[i * d + j] + inv_cov[j * d + i]) * 0.5;
            inv_cov[i * d + j] = avg;
            inv_cov[j * d + i] = avg;
        }
    }

    let expected = mahal_sq_host(&x, &mean, &inv_cov, n, d);

    let ptx = crate::ptx_kernels::fast_mcd_cstep_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fast_mcd_cstep_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_mean = DeviceBuffer::<f32>::from_host(&mean).expect("d_mean");
    let d_inv_cov = DeviceBuffer::<f32>::from_host(&inv_cov).expect("d_inv_cov");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_mean.as_device_ptr(),
                d_inv_cov.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                d as u32,
            ),
        )
        .expect("launch fast_mcd_cstep_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // d=4: double-sum has 16 FMA terms; FP32 rounding < ~1e-5 relative.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..n {
        assert!(
            close(out_gpu[k], expected[k], 1e-3, 1e-5),
            "fast_mcd_cstep[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}
