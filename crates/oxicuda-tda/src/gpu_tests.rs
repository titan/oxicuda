//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical /
//! structural equivalence to an independent CPU oracle. The launch ABI mirrors
//! the proven `oxicuda-snn` / `oxicuda-ot` harnesses: device buffers are passed
//! as their `CUdeviceptr` (a `.param .u64`), scalars as the matching Rust scalar
//! (`.param .u32` / `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Grid/block mapping note
//!
//! `LaunchParams::new(grid, block)` takes `(u32, u32)` tuples as `(x, y)`. The
//! 2-D kernels here map the matrix **row** to `ctaid.y` and the **column** to
//! `ctaid.x`, so a kernel whose output is `out[row * n_cols + col]` is launched
//! with `grid = (n_cols, n_rows)` (x = columns, y = rows).
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Independent host re-derivation** (numerical equivalence) —
//!   `pairwise_dist`, `diagram_match`, `witness_dist`: the kernel's documented
//!   per-element arithmetic is recomputed in independent host f32 and compared
//!   element-wise within an FP32 tolerance. These genuinely fail if ptxas
//!   miscompiles or the PTX has a wrong index / constant / op, because the host
//!   code never touches the JIT-compiled PTX.
//! * **Exact host re-derivation** (bit / integer equality) —
//!   `filtration_sort` (argsort permutation), `betti_count` (atomic essential
//!   count), `mapper_cluster` (single-linkage mark): integer / exact-FP
//!   semantics, asserted exactly.
//! * **Deterministic race-free re-derivation** — `boundary_reduce`: the kernel's
//!   per-thread inverse-pivot write is racy for general inputs, so it is
//!   validated on a *provably order-independent* input whose final array is the
//!   same under every thread interleaving (see the test for the proof).
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
///
/// A failure here means ptxas rejected the PTX (a real bug to be fixed in
/// `ptx_kernels.rs`), or the entry name is wrong.
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

/// Deterministic uniform `f32` in `[0, 1)` from the crate's MMIX LCG.
///
/// `LcgRng::next_f64` already returns `[0, 1)` via the high 53 bits scaled by
/// `2^-53`; casting to `f32` keeps it in `[0, 1)`.
fn unit_f32(rng: &mut LcgRng) -> f32 {
    rng.next_f64() as f32
}

// ===========================================================================
// 1. pairwise_dist  —  INDEPENDENT HOST RE-DERIVATION (squared Euclidean)
// ===========================================================================
//
// dist[i * n + j] = Σ_d (points[i,d] − points[j,d])^2  (squared, no sqrt).
// Block = (16, 16); one thread per (i, j) output cell.

#[test]
fn pairwise_dist_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 6_usize;
    let dim = 4_usize;

    let mut rng = LcgRng::new(0x7DA_0001);
    let points: Vec<f32> = (0..n * dim)
        .map(|_| unit_f32(&mut rng) * 2.0 - 1.0)
        .collect();

    // Independent host re-derivation in the kernel's accumulation order.
    let mut host = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f32;
            for d in 0..dim {
                let diff = points[i * dim + d] - points[j * dim + d];
                s += diff * diff;
            }
            host[i * n + j] = s;
        }
    }

    let ptx = crate::ptx_kernels::pairwise_dist_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pairwise_dist_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_dist = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * n]).expect("d_dist");

    // Grid = (ceil(n/16), ceil(n/16)), block = (16, 16). Square matrix ⇒ x and y
    // ranges are both n, so the row/col tuple ordering is immaterial here.
    let gx = grid_1d(n as u32, 16);
    let params = LaunchParams::new((gx, gx), (16u32, 16u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_points.as_device_ptr(),
                d_dist.as_device_ptr(),
                n as u32,
                dim as u32,
            ),
        )
        .expect("launch pairwise_dist_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n * n];
    d_dist.copy_to_host(&mut gpu).expect("copy dist");

    // GPU uses `fma.rn` (single rounding/term); host uses mul+add. Over dim = 4
    // terms the divergence is a few ulp (~1e-6 relative); 1e-5 is comfortable.
    let (rel, abs) = worst_diff(&gpu, &host);
    for k in 0..gpu.len() {
        assert!(
            close(gpu[k], host[k], 1e-5, 1e-5),
            "pairwise_dist[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            host[k]
        );
    }
    // Diagonal must be exactly zero (i == j ⇒ all diffs zero).
    for i in 0..n {
        assert_eq!(
            gpu[i * n + i].to_bits(),
            0_u32,
            "pairwise_dist diagonal [{i}] = {} (expected 0.0)",
            gpu[i * n + i]
        );
    }
}

// ===========================================================================
// 2. filtration_sort  —  EXACT HOST RE-DERIVATION (count-rank argsort)
// ===========================================================================
//
// For DISTINCT keys, each thread i writes its own index into
// indices[ rank(i) ] where rank(i) = #{k : filt[k] < filt[i]}. The result is
// the ascending argsort permutation. Bit-exact integer comparison.

#[test]
fn filtration_sort_matches_host_argsort() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Distinct keys (no ties ⇒ collision-free ranks, as the kernel documents).
    let filt: Vec<f32> = vec![0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4];
    let n = filt.len();

    // Sanity: keys are pairwise distinct so the rank map is a permutation.
    for a in 0..n {
        for b in (a + 1)..n {
            assert_ne!(filt[a], filt[b], "test setup: duplicate filtration key");
        }
    }

    // Exact host re-derivation: expected[rank(i)] = i.
    let mut expected = vec![0_u32; n];
    for i in 0..n {
        let rank = (0..n).filter(|&k| filt[k] < filt[i]).count();
        expected[rank] = i as u32;
    }

    let ptx = crate::ptx_kernels::filtration_sort_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "filtration_sort_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_filt = DeviceBuffer::<f32>::from_host(&filt).expect("d_filt");
    let d_idx = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_idx");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(d_filt.as_device_ptr(), d_idx.as_device_ptr(), n as u32),
        )
        .expect("launch filtration_sort_kernel");
    stream.synchronize().expect("sync");

    let mut idx_gpu = vec![0_u32; n];
    d_idx.copy_to_host(&mut idx_gpu).expect("copy idx");

    for r in 0..n {
        assert_eq!(
            idx_gpu[r], expected[r],
            "filtration_sort: indices[{r}] = {} (expected {} — argsort by ascending key)",
            idx_gpu[r], expected[r]
        );
    }
    // The output must itself be a permutation of 0..n.
    let mut seen = vec![false; n];
    for &v in &idx_gpu {
        assert!((v as usize) < n, "filtration_sort: index {v} out of range");
        assert!(!seen[v as usize], "filtration_sort: duplicate index {v}");
        seen[v as usize] = true;
    }
}

// ===========================================================================
// 3. boundary_reduce  —  DETERMINISTIC RACE-FREE RE-DERIVATION (inverse pivot)
// ===========================================================================
//
// Per thread t: read v = pivot_col[t]; if v == -1 skip; else atomically write
// t into pivot_col[v] (build the inverse low^{-1} map for boundary reduction).
//
// This is racy for general inputs (a written slot is also read by its own
// thread). We therefore use a PROVABLY order-independent input:
//
//   pivot_col = [3, -1, -1, -1, 7, -1, -1, -1]   (n = 8)
//
//   * thread 0 reads 3   ⇒ writes pivot_col[3] = 0.
//   * thread 4 reads 7   ⇒ writes pivot_col[7] = 4.
//   * thread 3 reads slot 3: either the initial -1 (⇒ skip) or the value 0
//     written by thread 0. If it reads 0 it writes pivot_col[0] = 3 — but
//     slot 0 ALREADY holds 3, so that write is a no-op. Hence slot 0 is 3 under
//     every interleaving.
//   * thread 7 mirrors thread 3: any write it makes is pivot_col[4] = 7, and
//     slot 4 already holds 7 — a no-op.
//   * threads 1,2,5,6 read -1 and skip; their slots are never written.
//
//   ⇒ Final array is [3, -1, -1, 0, 7, -1, -1, 4] under EVERY thread ordering.

#[test]
fn boundary_reduce_inverse_pivot_race_free() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let pivot_init: Vec<i32> = vec![3, -1, -1, -1, 7, -1, -1, -1];
    let n_cols = pivot_init.len();
    let expected: Vec<i32> = vec![3, -1, -1, 0, 7, -1, -1, 4];

    let ptx = crate::ptx_kernels::boundary_reduce_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "boundary_reduce_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_pivot = DeviceBuffer::<i32>::from_host(&pivot_init).expect("d_pivot");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_cols as u32, block), block);
    kernel
        .launch(&params, &stream, &(d_pivot.as_device_ptr(), n_cols as u32))
        .expect("launch boundary_reduce_kernel");
    stream.synchronize().expect("sync");

    let mut pivot_gpu = vec![0_i32; n_cols];
    d_pivot.copy_to_host(&mut pivot_gpu).expect("copy pivot");

    for k in 0..n_cols {
        assert_eq!(
            pivot_gpu[k], expected[k],
            "boundary_reduce: pivot_col[{k}] = {} (expected {} — inverse-pivot write)",
            pivot_gpu[k], expected[k]
        );
    }
}

// ===========================================================================
// 4. diagram_match  —  INDEPENDENT HOST RE-DERIVATION (L∞ ground cost)
// ===========================================================================
//
// cost[i * n_b + j] = max(|birth_a[i] − birth_b[j]|, |death_a[i] − death_b[j]|)
// (the Chebyshev / bottleneck ground distance). One thread per (i, j) cell;
// i = ctaid.y (over n_a rows), j = ctaid.x (over n_b cols).

#[test]
fn diagram_match_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_a = 4_usize;
    let n_b = 5_usize;

    let mut rng = LcgRng::new(0xD1A_6044);
    let birth_a: Vec<f32> = (0..n_a).map(|_| unit_f32(&mut rng)).collect();
    let death_a: Vec<f32> = (0..n_a).map(|_| 1.0 + unit_f32(&mut rng)).collect();
    let birth_b: Vec<f32> = (0..n_b).map(|_| unit_f32(&mut rng)).collect();
    let death_b: Vec<f32> = (0..n_b).map(|_| 1.0 + unit_f32(&mut rng)).collect();

    // Independent host re-derivation (sub/abs/max are exact in f32).
    let mut host = vec![0.0_f32; n_a * n_b];
    for i in 0..n_a {
        for j in 0..n_b {
            let db = (birth_a[i] - birth_b[j]).abs();
            let dd = (death_a[i] - death_b[j]).abs();
            host[i * n_b + j] = db.max(dd);
        }
    }

    let ptx = crate::ptx_kernels::diagram_match_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "diagram_match_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ba = DeviceBuffer::<f32>::from_host(&birth_a).expect("d_ba");
    let d_da = DeviceBuffer::<f32>::from_host(&death_a).expect("d_da");
    let d_bb = DeviceBuffer::<f32>::from_host(&birth_b).expect("d_bb");
    let d_db = DeviceBuffer::<f32>::from_host(&death_b).expect("d_db");
    let d_cost = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_a * n_b]).expect("d_cost");

    // Grid = (x = n_b cols, y = n_a rows), block = (1, 1).
    let params = LaunchParams::new((n_b as u32, n_a as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ba.as_device_ptr(),
                d_da.as_device_ptr(),
                d_bb.as_device_ptr(),
                d_db.as_device_ptr(),
                d_cost.as_device_ptr(),
                n_a as u32,
                n_b as u32,
            ),
        )
        .expect("launch diagram_match_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n_a * n_b];
    d_cost.copy_to_host(&mut gpu).expect("copy cost");

    // sub/abs/max are exact ⇒ effectively bit-exact; 1e-6 is a safe floor.
    let (rel, abs) = worst_diff(&gpu, &host);
    for k in 0..gpu.len() {
        assert!(
            close(gpu[k], host[k], 1e-6, 1e-6),
            "diagram_match cost[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            host[k]
        );
    }
}

// ===========================================================================
// 5. witness_dist  —  INDEPENDENT HOST RE-DERIVATION (Euclidean, with sqrt)
// ===========================================================================
//
// dist[l * n_pts + w] = sqrt( Σ_d (landmarks[l,d] − points[w,d])^2 ).
// l = ctaid.y (over n_land), w = ctaid.x (over n_pts).

#[test]
fn witness_dist_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_pts = 5_usize;
    let n_land = 3_usize;
    let dim = 4_usize;

    let mut rng = LcgRng::new(0x317E_5500);
    let points: Vec<f32> = (0..n_pts * dim)
        .map(|_| unit_f32(&mut rng) * 2.0 - 1.0)
        .collect();
    let landmarks: Vec<f32> = (0..n_land * dim)
        .map(|_| unit_f32(&mut rng) * 2.0 - 1.0)
        .collect();

    let mut host = vec![0.0_f32; n_land * n_pts];
    for l in 0..n_land {
        for w in 0..n_pts {
            let mut s = 0.0_f32;
            for d in 0..dim {
                let diff = landmarks[l * dim + d] - points[w * dim + d];
                s += diff * diff;
            }
            host[l * n_pts + w] = s.sqrt();
        }
    }

    let ptx = crate::ptx_kernels::witness_dist_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "witness_dist_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_land = DeviceBuffer::<f32>::from_host(&landmarks).expect("d_land");
    let d_dist = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_land * n_pts]).expect("d_dist");

    // Grid = (x = n_pts witnesses, y = n_land landmarks), block = (1, 1).
    let params = LaunchParams::new((n_pts as u32, n_land as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_points.as_device_ptr(),
                d_land.as_device_ptr(),
                d_dist.as_device_ptr(),
                n_pts as u32,
                n_land as u32,
                dim as u32,
            ),
        )
        .expect("launch witness_dist_kernel");
    stream.synchronize().expect("sync");

    let mut gpu = vec![0.0_f32; n_land * n_pts];
    d_dist.copy_to_host(&mut gpu).expect("copy dist");

    // fma accumulation + correctly-rounded sqrt vs host mul/add + sqrt: a few
    // ulp (~1e-6 relative). 1e-5 is comfortable and still catches any real bug.
    let (rel, abs) = worst_diff(&gpu, &host);
    for k in 0..gpu.len() {
        assert!(
            close(gpu[k], host[k], 1e-5, 1e-5),
            "witness_dist[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            gpu[k],
            host[k]
        );
    }
}

// ===========================================================================
// 6. betti_count  —  EXACT HOST RE-DERIVATION (atomic essential-pair count)
// ===========================================================================
//
// For each pair: if dims[tid] == query_dim AND deaths[tid] == +inf (essential),
// atomically increment betti[0]. The result is the count of essential pairs in
// the queried dimension. (The kernel compares against a hardcoded +inf and
// ignores the `max_death` param, which we pass as +inf anyway.)

#[test]
fn betti_count_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let inf = f32::INFINITY;
    let query_dim = 1_u32;

    // dims and deaths chosen so several dim==1 pairs are essential (death==inf)
    // and others (dim==1 but finite death, or wrong dim with inf death) are not.
    let dims: Vec<i32> = vec![0, 1, 1, 2, 1, 0, 1, 2, 1, 1];
    let deaths: Vec<f32> = vec![inf, inf, 0.7, inf, inf, inf, 0.3, inf, inf, 1.5];
    let n_pairs = dims.len();

    // Exact host count: dim == query_dim AND death is +inf.
    let expected: u32 = (0..n_pairs)
        .filter(|&k| dims[k] as u32 == query_dim && deaths[k] == inf)
        .count() as u32;
    // (indices 1, 4, 8 satisfy both ⇒ expected == 3)
    assert_eq!(expected, 3, "test setup: expected essential count");

    let ptx = crate::ptx_kernels::betti_count_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "betti_count_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_dims = DeviceBuffer::<i32>::from_host(&dims).expect("d_dims");
    let d_deaths = DeviceBuffer::<f32>::from_host(&deaths).expect("d_deaths");
    let d_betti = DeviceBuffer::<u32>::from_host(&[0_u32; 1]).expect("d_betti");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_pairs as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_dims.as_device_ptr(),
                d_deaths.as_device_ptr(),
                d_betti.as_device_ptr(),
                n_pairs as u32,
                query_dim,
                inf,
            ),
        )
        .expect("launch betti_count_kernel");
    stream.synchronize().expect("sync");

    let mut betti_gpu = vec![0_u32; 1];
    d_betti.copy_to_host(&mut betti_gpu).expect("copy betti");

    assert_eq!(
        betti_gpu[0], expected,
        "betti_count: betti[0] = {} (expected {} essential dim-{query_dim} pairs)",
        betti_gpu[0], expected
    );
}

// ===========================================================================
// 7. mapper_cluster  —  EXACT HOST RE-DERIVATION (single-linkage mark)
// ===========================================================================
//
// For the upper triangle i < j: if Euclidean(points[i], points[j]) <= threshold,
// write cluster_id[j] = cluster_id[i]. Racy for general inputs (chained / many
// writers to the same j), so we use points with EXACTLY ONE qualifying pair
// (0, 1) and all other pairwise distances ≫ threshold ⇒ a single deterministic
// write: cluster_id[1] := cluster_id[0].

#[test]
fn mapper_cluster_single_pair_mark() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 2_usize;
    // Only (0,1) are within threshold; the rest are >= 9.9 apart.
    let points: Vec<f32> = vec![
        0.0, 0.0, // p0
        0.05, 0.0, // p1  (dist to p0 = 0.05)
        10.0, 0.0, // p2
        20.0, 0.0, // p3
        30.0, 0.0, // p4
    ];
    let n_pts = points.len() / dim;
    let threshold = 0.1_f32;
    let cluster_init: Vec<i32> = (0..n_pts as i32).collect();

    // Exact host re-derivation: exactly one qualifying pair (0,1).
    let mut expected = cluster_init.clone();
    let mut qualifying = 0_usize;
    for i in 0..n_pts {
        for j in (i + 1)..n_pts {
            let mut s = 0.0_f32;
            for d in 0..dim {
                let diff = points[i * dim + d] - points[j * dim + d];
                s += diff * diff;
            }
            if s.sqrt() <= threshold {
                expected[j] = expected[i];
                qualifying += 1;
            }
        }
    }
    assert_eq!(qualifying, 1, "test setup: expected exactly one close pair");

    let ptx = crate::ptx_kernels::mapper_cluster_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "mapper_cluster_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_cluster = DeviceBuffer::<i32>::from_host(&cluster_init).expect("d_cluster");

    // Grid = (n_pts, n_pts), block = (1, 1); i = ctaid.y, j = ctaid.x.
    let params = LaunchParams::new((n_pts as u32, n_pts as u32), (1u32, 1u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_points.as_device_ptr(),
                d_cluster.as_device_ptr(),
                n_pts as u32,
                dim as u32,
                threshold,
            ),
        )
        .expect("launch mapper_cluster_kernel");
    stream.synchronize().expect("sync");

    let mut cluster_gpu = vec![0_i32; n_pts];
    d_cluster
        .copy_to_host(&mut cluster_gpu)
        .expect("copy cluster");

    for k in 0..n_pts {
        assert_eq!(
            cluster_gpu[k], expected[k],
            "mapper_cluster: cluster_id[{k}] = {} (expected {} — single-linkage mark)",
            cluster_gpu[k], expected[k]
        );
    }
}

// ===========================================================================
// 8. batched_column_reduce  —  EXACT HOST RE-DERIVATION (integer pivot lookup)
// ===========================================================================
//
// One thread per column j (kernel in `homology::gpu_reduction`):
//   add_src[j] = owner[low[j]]   iff  low[j] >= 0  AND  0 <= owner[low[j]] < j
//   add_src[j] = -1              otherwise.
//
// Each thread writes only its own `add_src[j]`, so the result is fully
// deterministic (no races) and asserted bit-exact. On sm_80+ the kernel also
// commits/awaits a (decorative, empty) `cp.async` group; this test exercises
// that path on the live sm_86 device. Inputs are crafted to hit every branch:
// zero-column skip, owner == -1 skip, owner >= j skip, and a real column-add.

#[test]
fn batched_column_reduce_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // low[j] = pivot row of column j (-1 ⇒ zero column).
    let low: Vec<i32> = vec![-1, 0, 3, 1, 2, -1, 5, 4];
    // owner[r] = leftmost column currently claiming pivot row r (-1 ⇒ none).
    let owner: Vec<i32> = vec![-1, 3, 1, 0, 7, 2, 0, 0];
    let n_cols = low.len();
    assert_eq!(owner.len(), n_cols, "test setup: low/owner length mismatch");

    // Exact host re-derivation mirroring the kernel branch-for-branch.
    let expected: Vec<i32> = low
        .iter()
        .enumerate()
        .map(|(j, &lj)| {
            if lj < 0 {
                return -1;
            }
            let owner_of_low = owner[lj as usize];
            if owner_of_low >= 0 && owner_of_low < j as i32 {
                owner_of_low
            } else {
                -1
            }
        })
        .collect();
    // Pin the crafted expectation so the test self-documents every branch:
    //   j=0 zero col ⇒ -1; j=1 owner==-1 ⇒ -1; j=2 owner 0 < 2 ⇒ 0;
    //   j=3 owner 3 >= 3 ⇒ -1; j=4 owner 1 < 4 ⇒ 1; j=5 zero col ⇒ -1;
    //   j=6 owner 2 < 6 ⇒ 2; j=7 owner 7 >= 7 ⇒ -1.
    assert_eq!(
        expected,
        vec![-1, -1, 0, -1, 1, -1, 2, -1],
        "test setup: crafted oracle drifted"
    );

    let ptx = crate::homology::gpu_reduction::batched_column_reduce_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "batched_column_reduce_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_low = DeviceBuffer::<i32>::from_host(&low).expect("d_low");
    let d_owner = DeviceBuffer::<i32>::from_host(&owner).expect("d_owner");
    // Pre-fill add_src with a sentinel the kernel must overwrite for every column.
    let d_add_src = DeviceBuffer::<i32>::from_host(&vec![99_i32; n_cols]).expect("d_add_src");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_cols as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_low.as_device_ptr(),
                d_owner.as_device_ptr(),
                d_add_src.as_device_ptr(),
                n_cols as u32,
            ),
        )
        .expect("launch batched_column_reduce_kernel");
    stream.synchronize().expect("sync");

    let mut add_src_gpu = vec![0_i32; n_cols];
    d_add_src
        .copy_to_host(&mut add_src_gpu)
        .expect("copy add_src");

    for j in 0..n_cols {
        assert_eq!(
            add_src_gpu[j], expected[j],
            "batched_column_reduce: add_src[{j}] = {} (expected {} — owner[low[j]] pivot lookup)",
            add_src_gpu[j], expected[j]
        );
    }
}

// ===========================================================================
// 9. vietoris_rips_edges  —  INDEPENDENT HOST RE-DERIVATION (Euclidean + flag)
// ===========================================================================
//
// Thread (i, j); i = ctaid.y·ntid.y + tid.y, j = ctaid.x·ntid.x + tid.x. For
// the STRICT upper triangle i < j only:
//   edge_len[i·n + j]  = sqrt( Σ_d (points[i,d] − points[j,d])^2 )
//   edge_flag[i·n + j] = 1 if edge_len <= threshold else 0.
// Cells with i >= j (lower triangle + diagonal) and out-of-range threads are
// never written, so they must retain their pre-fill sentinel — asserted here to
// catch any spurious out-of-triangle store.

#[test]
fn vietoris_rips_edges_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_pts = 6_usize;
    let dim = 3_usize;

    let mut rng = LcgRng::new(0x71E5_0042);
    let points: Vec<f32> = (0..n_pts * dim)
        .map(|_| unit_f32(&mut rng) * 2.0 - 1.0)
        .collect();
    let threshold = 1.5_f32;

    // Independent host oracle, upper triangle only.
    let len_sentinel = -7.0_f32;
    let flag_sentinel = 9_u32;
    let mut host_len = vec![len_sentinel; n_pts * n_pts];
    let mut host_flag = vec![flag_sentinel; n_pts * n_pts];
    let mut n_pass = 0_usize;
    let mut n_fail = 0_usize;
    for i in 0..n_pts {
        for j in (i + 1)..n_pts {
            let mut s = 0.0_f32;
            for d in 0..dim {
                let diff = points[i * dim + d] - points[j * dim + d];
                s += diff * diff;
            }
            let len = s.sqrt();
            host_len[i * n_pts + j] = len;
            if len <= threshold {
                host_flag[i * n_pts + j] = 1;
                n_pass += 1;
            } else {
                host_flag[i * n_pts + j] = 0;
                n_fail += 1;
            }
        }
    }
    // Make sure the chosen threshold exercises BOTH the accept and reject paths.
    assert!(
        n_pass > 0 && n_fail > 0,
        "test setup: threshold must split edges (pass={n_pass}, fail={n_fail})"
    );

    let ptx = crate::homology::gpu_reduction::vietoris_rips_edges_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "vietoris_rips_edges_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_points = DeviceBuffer::<f32>::from_host(&points).expect("d_points");
    let d_flag =
        DeviceBuffer::<u32>::from_host(&vec![flag_sentinel; n_pts * n_pts]).expect("d_flag");
    let d_len = DeviceBuffer::<f32>::from_host(&vec![len_sentinel; n_pts * n_pts]).expect("d_len");

    // Grid = (x = cols, y = rows) = (ceil(n/16), ceil(n/16)); block = (16, 16).
    let g = grid_1d(n_pts as u32, 16);
    let params = LaunchParams::new((g, g), (16u32, 16u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_points.as_device_ptr(),
                d_flag.as_device_ptr(),
                d_len.as_device_ptr(),
                n_pts as u32,
                dim as u32,
                threshold,
            ),
        )
        .expect("launch vietoris_rips_edges_kernel");
    stream.synchronize().expect("sync");

    let mut len_gpu = vec![0.0_f32; n_pts * n_pts];
    let mut flag_gpu = vec![0_u32; n_pts * n_pts];
    d_len.copy_to_host(&mut len_gpu).expect("copy edge_len");
    d_flag.copy_to_host(&mut flag_gpu).expect("copy edge_flag");

    for i in 0..n_pts {
        for j in 0..n_pts {
            let e = i * n_pts + j;
            if i < j {
                // fma accumulation + correctly-rounded sqrt vs host mul/add+sqrt.
                assert!(
                    close(len_gpu[e], host_len[e], 1e-4, 1e-4),
                    "vietoris edge_len[{i},{j}] mismatch: gpu={} host={}",
                    len_gpu[e],
                    host_len[e]
                );
                assert_eq!(
                    flag_gpu[e], host_flag[e],
                    "vietoris edge_flag[{i},{j}] = {} (expected {}; len={}, thr={threshold})",
                    flag_gpu[e], host_flag[e], len_gpu[e]
                );
            } else {
                // Lower triangle + diagonal must be untouched (sentinels intact).
                assert_eq!(
                    len_gpu[e].to_bits(),
                    len_sentinel.to_bits(),
                    "vietoris edge_len[{i},{j}] = {} but should be untouched",
                    len_gpu[e]
                );
                assert_eq!(
                    flag_gpu[e], flag_sentinel,
                    "vietoris edge_flag[{i},{j}] = {} but should be untouched",
                    flag_gpu[e]
                );
            }
        }
    }
}

// ===========================================================================
// 10. wasserstein_auction  —  INDEPENDENT HOST RE-DERIVATION (Bertsekas bid)
// ===========================================================================
//
// One thread per UNASSIGNED person i. Scanning j = 0..n_b with net value
// v(j) = −cost[i,j] − price[j], it tracks the best and second-best net values
// (strict ">") and emits:
//   bid_obj[i] = argmax_j v(j)
//   bid_val[i] = (best − second) + epsilon.
// Degenerate cases: n_b == 0 ⇒ bid_obj = −1, bid_val = 0; n_b == 1 ⇒ second is
// −inf ⇒ bid_val = +inf (the "must take it" bid). The non-zero epsilon and a
// genuine second-best make this catch a dropped-epsilon or wrong-second bug.

#[test]
fn wasserstein_auction_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_a = 4_usize;
    let n_b = 5_usize;
    let epsilon = 0.05_f32;

    let mut rng = LcgRng::new(0x4A3C_77E1);
    let cost: Vec<f32> = (0..n_a * n_b).map(|_| unit_f32(&mut rng)).collect();
    let price: Vec<f32> = (0..n_b).map(|_| unit_f32(&mut rng)).collect();

    // Independent host oracle, branch-for-branch identical to the kernel.
    let mut host_obj = vec![0_i32; n_a];
    let mut host_val = vec![0.0_f32; n_a];
    let mut any_nonzero_obj = false;
    for i in 0..n_a {
        let mut best = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        let mut best_j = 0_i32;
        // Net values within a row must be distinct so argmax / second are
        // unambiguous (matches the kernel's strict-greater comparisons).
        let mut nets = Vec::with_capacity(n_b);
        for j in 0..n_b {
            let v = -cost[i * n_b + j] - price[j];
            nets.push(v);
            if v > best {
                second = best;
                best = v;
                best_j = j as i32;
            } else if v > second {
                second = v;
            }
        }
        for a in 0..n_b {
            for b in (a + 1)..n_b {
                assert!(
                    (nets[a] - nets[b]).abs() > 0.0,
                    "test setup: tied net values in row {i} (j={a}, j={b})"
                );
            }
        }
        host_obj[i] = best_j;
        host_val[i] = (best - second) + epsilon;
        if best_j != 0 {
            any_nonzero_obj = true;
        }
    }
    // Ensure at least one row's winner is not object 0 (exercises real argmax).
    assert!(
        any_nonzero_obj,
        "test setup: every winner is object 0 — argmax path under-exercised"
    );

    let ptx = crate::homology::gpu_reduction::wasserstein_auction_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "wasserstein_auction_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_cost = DeviceBuffer::<f32>::from_host(&cost).expect("d_cost");
    let d_price = DeviceBuffer::<f32>::from_host(&price).expect("d_price");
    let d_obj = DeviceBuffer::<i32>::from_host(&vec![77_i32; n_a]).expect("d_obj");
    let d_val = DeviceBuffer::<f32>::from_host(&vec![-7.0_f32; n_a]).expect("d_val");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_a as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_cost.as_device_ptr(),
                d_price.as_device_ptr(),
                d_obj.as_device_ptr(),
                d_val.as_device_ptr(),
                n_a as u32,
                n_b as u32,
                epsilon,
            ),
        )
        .expect("launch wasserstein_auction_kernel");
    stream.synchronize().expect("sync");

    let mut obj_gpu = vec![0_i32; n_a];
    let mut val_gpu = vec![0.0_f32; n_a];
    d_obj.copy_to_host(&mut obj_gpu).expect("copy bid_obj");
    d_val.copy_to_host(&mut val_gpu).expect("copy bid_val");

    for i in 0..n_a {
        assert_eq!(
            obj_gpu[i], host_obj[i],
            "wasserstein bid_obj[{i}] = {} (expected {} — argmax net value)",
            obj_gpu[i], host_obj[i]
        );
        assert!(
            close(val_gpu[i], host_val[i], 1e-4, 1e-4),
            "wasserstein bid_val[{i}] mismatch: gpu={} host={} (best−second+ε)",
            val_gpu[i],
            host_val[i]
        );
    }
}

// ---------------------------------------------------------------------------
// 11. wasserstein_auction degenerate object counts (n_b == 0 and n_b == 1)
// ---------------------------------------------------------------------------
//
// n_b == 0 ⇒ no object: bid_obj = −1, bid_val = 0.
// n_b == 1 ⇒ only object 0; second stays −inf ⇒ bid_val = +inf, bid_obj = 0.

#[test]
fn wasserstein_auction_degenerate_object_counts() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let ptx = crate::homology::gpu_reduction::wasserstein_auction_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "wasserstein_auction_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let n_a = 3_usize;
    let epsilon = 0.1_f32;
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_a as u32, block), block);

    // --- n_b == 0: empty object set ---------------------------------------
    {
        // `n_b == 0` ⇒ the kernel reads no cost/price element; `from_host` still
        // needs a non-null pointer, so we pass length-1 dummy buffers.
        let d_cost = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_cost0");
        let d_price = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_price0");
        let d_obj = DeviceBuffer::<i32>::from_host(&vec![77_i32; n_a]).expect("d_obj0");
        let d_val = DeviceBuffer::<f32>::from_host(&vec![-7.0_f32; n_a]).expect("d_val0");

        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_cost.as_device_ptr(),
                    d_price.as_device_ptr(),
                    d_obj.as_device_ptr(),
                    d_val.as_device_ptr(),
                    n_a as u32,
                    0u32,
                    epsilon,
                ),
            )
            .expect("launch wasserstein_auction_kernel (n_b=0)");
        stream.synchronize().expect("sync n_b=0");

        let mut obj_gpu = vec![0_i32; n_a];
        let mut val_gpu = vec![0.0_f32; n_a];
        d_obj.copy_to_host(&mut obj_gpu).expect("copy obj0");
        d_val.copy_to_host(&mut val_gpu).expect("copy val0");
        for i in 0..n_a {
            assert_eq!(obj_gpu[i], -1, "n_b=0: bid_obj[{i}] should be -1");
            assert_eq!(
                val_gpu[i].to_bits(),
                0.0_f32.to_bits(),
                "n_b=0: bid_val[{i}] should be 0.0"
            );
        }
    }

    // --- n_b == 1: single object ⇒ +inf "must take it" bid -----------------
    {
        let n_b = 1_usize;
        let cost: Vec<f32> = vec![0.3, 0.7, 0.1]; // one cost per person to object 0
        let price: Vec<f32> = vec![0.2];
        let d_cost = DeviceBuffer::<f32>::from_host(&cost).expect("d_cost1");
        let d_price = DeviceBuffer::<f32>::from_host(&price).expect("d_price1");
        let d_obj = DeviceBuffer::<i32>::from_host(&vec![77_i32; n_a]).expect("d_obj1");
        let d_val = DeviceBuffer::<f32>::from_host(&vec![-7.0_f32; n_a]).expect("d_val1");

        kernel
            .launch(
                &params,
                &stream,
                &(
                    d_cost.as_device_ptr(),
                    d_price.as_device_ptr(),
                    d_obj.as_device_ptr(),
                    d_val.as_device_ptr(),
                    n_a as u32,
                    n_b as u32,
                    epsilon,
                ),
            )
            .expect("launch wasserstein_auction_kernel (n_b=1)");
        stream.synchronize().expect("sync n_b=1");

        let mut obj_gpu = vec![0_i32; n_a];
        let mut val_gpu = vec![0.0_f32; n_a];
        d_obj.copy_to_host(&mut obj_gpu).expect("copy obj1");
        d_val.copy_to_host(&mut val_gpu).expect("copy val1");
        for i in 0..n_a {
            assert_eq!(obj_gpu[i], 0, "n_b=1: bid_obj[{i}] should be object 0");
            assert!(
                val_gpu[i].is_infinite() && val_gpu[i] > 0.0,
                "n_b=1: bid_val[{i}] = {} should be +inf (best − (−inf) + ε)",
                val_gpu[i]
            );
        }
    }
}
