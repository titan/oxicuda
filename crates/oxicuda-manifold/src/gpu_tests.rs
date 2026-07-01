//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch` on the real CUDA
//! device, copies the results back, and asserts numerical equivalence to an
//! independent host re-derivation of the kernel's documented arithmetic. The
//! launch ABI mirrors the `oxicuda-snn` / `oxicuda-recsys` harnesses: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars as the
//! matching Rust scalar (`.param .u32` / `.param .f32`), in declared order.
//!
//! ## Oracle strength (honest accounting)
//!
//! Every kernel here is validated by an **independent host re-derivation** of
//! the exact arithmetic its PTX documents (pairwise squared distance, double
//! centering, JL random projection, t-SNE / UMAP gradient steps, top-k
//! insertion). These oracles are written from first principles and are
//! independent of the JIT-compiled PTX, so they genuinely fail if ptxas
//! miscompiles, a constant is wrong, or an index/sign is off.
//!
//! ## PTX bug found and fixed
//!
//! ### `knn_topk_kernel` — degenerate top-k (only the last slot was used)
//!
//! The shipped kernel compared each candidate distance only against the *last*
//! (worst) slot and, on success, overwrote *only* that last slot — it never
//! inserted into / shifted the sorted buffer. For `k > 1` this leaves slots
//! `0..k-2` at `+inf` and slot `k-1` holding the single global minimum, i.e. it
//! is NOT a top-k at all (it is correct only for `k == 1`). Fixed in
//! `ptx_kernels::knn_topk_ptx` by adding a proper ascending insertion (bubble
//! the freshly written last slot up while it is smaller than its predecessor),
//! validated below against a host sort-and-truncate oracle for `k = 3`.
//!
//! Every test skips (returns early) when no CUDA device is present.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

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
/// A failure here means ptxas rejected the PTX — a real bug to fix in
/// `ptx_kernels.rs`, not something to skip.
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
// 1. pairwise_dist_sq — d[i,j] = sum_k (x[i,k] - x[j,k])^2
// ===========================================================================

#[test]
fn pairwise_dist_sq_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 5_usize;
    let dim = 3_usize;
    // Deterministic, distinct rows.
    let x: Vec<f32> = (0..n * dim)
        .map(|t| 0.3 + 0.7 * (t as f32) - 0.05 * (t * t) as f32)
        .collect();

    // Host oracle.
    let mut expected = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for k in 0..dim {
                let diff = x[i * dim + k] - x[j * dim + k];
                acc += diff * diff;
            }
            expected[i * n + j] = acc;
        }
    }

    let ptx = crate::ptx_kernels::pairwise_dist_sq_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pairwise_dist_sq_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_d = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * n]).expect("d_d");

    // Block (16,16); grid covers n in both dims. ctaid.y -> row i, ctaid.x -> col j.
    let block = (16u32, 16u32);
    let grid = (grid_1d(n as u32, 16), grid_1d(n as u32, 16));
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_d.as_device_ptr(),
                n as u32,
                dim as u32,
            ),
        )
        .expect("launch pairwise_dist_sq_kernel");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * n];
    d_d.copy_to_host(&mut got).expect("copy d");

    let (rel, abs) = worst_diff(&got, &expected);
    for k in 0..got.len() {
        assert!(
            close(got[k], expected[k], 1e-5, 1e-4),
            "pairwise_dist[{k}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            got[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 2. knn_topk — per-row k smallest distances of a distance matrix (excl. self)
//    (kernel fixed: real ascending insertion-sort top-k)
// ===========================================================================

#[test]
fn knn_topk_matches_host_topk() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 6_usize;
    let k = 3_usize;
    // Build a distance matrix with DISTINCT off-diagonal values so the top-k is
    // unambiguous (no ties to break). d[i,j] = ((i*7 + j*13 + 1) mod 23) + 0.5,
    // diagonal set to 0 (self, skipped by the kernel).
    let mut d = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            d[i * n + j] = if i == j {
                0.0
            } else {
                (((i * 7 + j * 13 + 1) % 23) as f32) + 0.5
            };
        }
    }

    // Host oracle: per row, sort (dist, idx) for j != i ascending, take first k.
    let mut exp_dist = vec![0.0_f32; n * k];
    let mut exp_idx = vec![0u32; n * k];
    for i in 0..n {
        let mut cand: Vec<(f32, u32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (d[i * n + j], j as u32))
            .collect();
        // Distinct distances guaranteed, but sort by (dist, idx) for determinism.
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("no NaN").then(a.1.cmp(&b.1)));
        for s in 0..k {
            exp_dist[i * k + s] = cand[s].0;
            exp_idx[i * k + s] = cand[s].1;
        }
    }

    let ptx = crate::ptx_kernels::knn_topk_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "knn_topk_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_d = DeviceBuffer::<f32>::from_host(&d).expect("d_d");
    let d_idx = DeviceBuffer::<u32>::from_host(&vec![0u32; n * k]).expect("d_idx");
    let d_dist = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * k]).expect("d_dist");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_d.as_device_ptr(),
                d_idx.as_device_ptr(),
                d_dist.as_device_ptr(),
                n as u32,
                k as u32,
            ),
        )
        .expect("launch knn_topk_kernel");
    stream.synchronize().expect("sync");

    let mut got_dist = vec![0.0_f32; n * k];
    let mut got_idx = vec![0u32; n * k];
    d_dist.copy_to_host(&mut got_dist).expect("copy dist");
    d_idx.copy_to_host(&mut got_idx).expect("copy idx");

    for i in 0..n {
        for s in 0..k {
            let p = i * k + s;
            assert!(
                close(got_dist[p], exp_dist[p], 1e-6, 1e-5),
                "knn dist[row {i} slot {s}] gpu={} host={}",
                got_dist[p],
                exp_dist[p]
            );
            assert_eq!(
                got_idx[p], exp_idx[p],
                "knn idx[row {i} slot {s}] gpu={} host={}",
                got_idx[p], exp_idx[p]
            );
        }
    }
}

// ===========================================================================
// 3. tsne_grad — grad[i,d] = 4 * sum_{j!=i} (p_ij - q_ij) * q_ij * (y_id - y_jd)
// ===========================================================================

#[test]
fn tsne_grad_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 4_usize;
    let dim = 2_usize;
    let p: Vec<f32> = (0..n * n).map(|t| 0.02 + 0.011 * (t as f32)).collect();
    let q: Vec<f32> = (0..n * n).map(|t| 0.015 + 0.007 * (t as f32)).collect();
    let y: Vec<f32> = (0..n * dim).map(|t| -0.4 + 0.3 * (t as f32)).collect();

    // Host oracle.
    let mut expected = vec![0.0_f32; n * dim];
    for i in 0..n {
        for dd in 0..dim {
            let mut acc = 0.0_f32;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let pij = p[i * n + j];
                let qij = q[i * n + j];
                let yi = y[i * dim + dd];
                let yj = y[j * dim + dd];
                acc += (pij - qij) * qij * (yi - yj);
            }
            expected[i * dim + dd] = 4.0 * acc;
        }
    }

    let ptx = crate::ptx_kernels::tsne_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tsne_grad_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_p = DeviceBuffer::<f32>::from_host(&p).expect("d_p");
    let d_q = DeviceBuffer::<f32>::from_host(&q).expect("d_q");
    let d_y = DeviceBuffer::<f32>::from_host(&y).expect("d_y");
    let d_grad = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * dim]).expect("d_grad");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_p.as_device_ptr(),
                d_q.as_device_ptr(),
                d_y.as_device_ptr(),
                d_grad.as_device_ptr(),
                n as u32,
                dim as u32,
            ),
        )
        .expect("launch tsne_grad_kernel");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * dim];
    d_grad.copy_to_host(&mut got).expect("copy grad");

    let (rel, abs) = worst_diff(&got, &expected);
    for k in 0..got.len() {
        assert!(
            close(got[k], expected[k], 1e-4, 1e-5),
            "tsne_grad[{k}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            got[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 4. umap_step — attractive edge SGD step (disjoint edges => race-free)
// ===========================================================================

#[test]
fn umap_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 8_usize;
    let dim = 2_usize;
    let alpha = 0.5_f32;
    // Disjoint edges: each vertex appears in at most one edge => no write races.
    let edges_i: Vec<u32> = vec![0, 2, 4, 6];
    let edges_j: Vec<u32> = vec![1, 3, 5, 7];
    let n_edges = edges_i.len();

    let y0: Vec<f32> = (0..n * dim).map(|t| -0.6 + 0.2 * (t as f32)).collect();

    // Host oracle: replicate the kernel's exact formula.
    let mut expected = y0.clone();
    for e in 0..n_edges {
        let vi = edges_i[e] as usize;
        let vj = edges_j[e] as usize;
        let mut dist2 = 0.0_f32;
        for dd in 0..dim {
            let diff = expected[vi * dim + dd] - expected[vj * dim + dd];
            dist2 += diff * diff;
        }
        let coef = -(2.0 * alpha) / (1.0 + dist2);
        for dd in 0..dim {
            let yi = expected[vi * dim + dd];
            let yj = expected[vj * dim + dd];
            let g = coef * (yi - yj);
            expected[vi * dim + dd] = yi + g;
            expected[vj * dim + dd] = yj - g;
        }
    }

    let ptx = crate::ptx_kernels::umap_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "umap_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_y = DeviceBuffer::<f32>::from_host(&y0).expect("d_y");
    let d_ei = DeviceBuffer::<u32>::from_host(&edges_i).expect("d_ei");
    let d_ej = DeviceBuffer::<u32>::from_host(&edges_j).expect("d_ej");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n_edges as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_y.as_device_ptr(),
                d_ei.as_device_ptr(),
                d_ej.as_device_ptr(),
                n_edges as u32,
                dim as u32,
                alpha,
            ),
        )
        .expect("launch umap_step_kernel");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * dim];
    d_y.copy_to_host(&mut got).expect("copy y");

    let (rel, abs) = worst_diff(&got, &expected);
    for k in 0..got.len() {
        assert!(
            close(got[k], expected[k], 1e-4, 1e-5),
            "umap_step y[{k}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            got[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 5. pca_center — x[i,d] -= mean[d]
// ===========================================================================

#[test]
fn pca_center_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 4_usize;
    let dim = 5_usize;
    let x0: Vec<f32> = (0..n * dim).map(|t| 1.0 + 0.5 * (t as f32)).collect();
    let mean: Vec<f32> = (0..dim).map(|d| 0.25 + 0.4 * (d as f32)).collect();

    let mut expected = x0.clone();
    for i in 0..n {
        for d in 0..dim {
            expected[i * dim + d] -= mean[d];
        }
    }

    let ptx = crate::ptx_kernels::pca_center_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pca_center_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x0).expect("d_x");
    let d_mean = DeviceBuffer::<f32>::from_host(&mean).expect("d_mean");

    let total = (n * dim) as u32;
    let block = 128_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_mean.as_device_ptr(),
                n as u32,
                dim as u32,
            ),
        )
        .expect("launch pca_center_kernel");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * dim];
    d_x.copy_to_host(&mut got).expect("copy x");

    for k in 0..got.len() {
        assert!(
            close(got[k], expected[k], 1e-6, 1e-5),
            "pca_center x[{k}] gpu={} host={}",
            got[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 6. mds_double_center — b[i,j] = -0.5*(d2 - row_mean[i] - col_mean[j] + tot)
// ===========================================================================

#[test]
fn mds_double_center_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 4_usize;
    let d2: Vec<f32> = (0..n * n).map(|t| 0.1 + 0.3 * (t as f32)).collect();
    let row_mean: Vec<f32> = (0..n).map(|i| 0.5 + 0.2 * (i as f32)).collect();
    let col_mean: Vec<f32> = (0..n).map(|j| 0.3 + 0.15 * (j as f32)).collect();
    let total_mean = 0.42_f32;

    let mut expected = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let v = d2[i * n + j] - row_mean[i] - col_mean[j] + total_mean;
            expected[i * n + j] = -0.5 * v;
        }
    }

    let ptx = crate::ptx_kernels::mds_double_center_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mds_double_center_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_d2 = DeviceBuffer::<f32>::from_host(&d2).expect("d_d2");
    let d_rm = DeviceBuffer::<f32>::from_host(&row_mean).expect("d_rm");
    let d_cm = DeviceBuffer::<f32>::from_host(&col_mean).expect("d_cm");
    let d_b = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * n]).expect("d_b");

    let block = (16u32, 16u32);
    let grid = (grid_1d(n as u32, 16), grid_1d(n as u32, 16));
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_d2.as_device_ptr(),
                d_rm.as_device_ptr(),
                d_cm.as_device_ptr(),
                total_mean,
                d_b.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch mds_double_center_kernel");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * n];
    d_b.copy_to_host(&mut got).expect("copy b");

    let (rel, abs) = worst_diff(&got, &expected);
    for k in 0..got.len() {
        assert!(
            close(got[k], expected[k], 1e-5, 1e-5),
            "mds b[{k}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            got[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 7. random_proj — out[i,c] = sum_d x[i,d] * R[d,c]
// ===========================================================================

#[test]
fn random_proj_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 3_usize;
    let d_dim = 4_usize;
    let k = 2_usize;
    let x: Vec<f32> = (0..n * d_dim).map(|t| 0.2 + 0.3 * (t as f32)).collect();
    let r: Vec<f32> = (0..d_dim * k).map(|t| -0.5 + 0.25 * (t as f32)).collect();

    let mut expected = vec![0.0_f32; n * k];
    for i in 0..n {
        for c in 0..k {
            let mut acc = 0.0_f32;
            for dd in 0..d_dim {
                acc += x[i * d_dim + dd] * r[dd * k + c];
            }
            expected[i * k + c] = acc;
        }
    }

    let ptx = crate::ptx_kernels::random_proj_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "random_proj_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_r = DeviceBuffer::<f32>::from_host(&r).expect("d_r");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * k]).expect("d_out");

    // ctaid.y -> row i (n), ctaid.x -> col c (k). Block (16,16).
    let block = (16u32, 16u32);
    let grid = (grid_1d(k as u32, 16), grid_1d(n as u32, 16));
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_r.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                d_dim as u32,
                k as u32,
            ),
        )
        .expect("launch random_proj_kernel");
    stream.synchronize().expect("sync");

    let mut got = vec![0.0_f32; n * k];
    d_out.copy_to_host(&mut got).expect("copy out");

    let (rel, abs) = worst_diff(&got, &expected);
    for kk in 0..got.len() {
        assert!(
            close(got[kk], expected[kk], 1e-5, 1e-5),
            "random_proj out[{kk}] gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            got[kk],
            expected[kk]
        );
    }
}
