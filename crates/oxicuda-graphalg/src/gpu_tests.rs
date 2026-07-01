//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts equivalence to an
//! independent CPU re-derivation of the kernel's documented arithmetic. The
//! launch ABI mirrors the working `oxicuda-snn` / `oxicuda-ot` canaries: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars as the
//! matching Rust scalar (`.param .u32` / `.param .f32`), in declared order.
//!
//! ## Oracle strength (honest accounting)
//!
//! All seven kernels are single-step CSR primitives (one BFS frontier hop, one
//! Dijkstra edge relaxation, one PageRank power-iteration sweep, one
//! Floyd-Warshall `k`-update, one triangle-count pass, one boolean SpMV, one
//! min-label propagation step). The crate's *public* CPU algorithms operate on
//! `Graph`/adjacency-list structures and run to completion, so none exposes a
//! single-step-on-raw-CSR function to call directly. The oracle is therefore an
//! **independent host re-derivation** of each kernel's documented per-element
//! arithmetic, evaluated on a tiny deterministic graph whose expected output is
//! also hand-verified in the comments. Because the host code is written
//! independently of the JIT-compiled PTX, a mismatch genuinely indicates a wrong
//! constant / shift / index / miscompile in the kernel — not a tautology.
//!
//! Integer-valued kernels (BFS levels, frontier flags, triangle counts, boolean
//! SpMV, labels) are compared **bit-exactly**; the float kernels (Dijkstra
//! distances, PageRank ranks, Floyd-Warshall distances) within FP32 tolerance.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

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
/// A failure here means ptxas rejected the hand-written PTX — a real bug that
/// must be fixed in `ptx_kernels.rs`, not skipped.
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

/// Sentinel `-1` (unvisited) stored in a `u32` `level` array, matching the
/// kernel's `0xFFFFFFFF` comparison.
const UNVISITED: u32 = 0xFFFF_FFFF;

/// A small fixed **undirected** graph in CSR (sorted adjacency), shared by the
/// BFS / triangle / SpMV / label tests.
///
/// Vertices 0..6, edges {0-1,0-2,1-2,1-3,2-3,3-4,4-5}:
/// ```text
///   0: [1,2]      3: [1,2,4]
///   1: [0,2,3]    4: [3,5]
///   2: [0,1,3]    5: [4]
/// ```
/// It contains exactly two triangles: {0,1,2} and {1,2,3}.
fn shared_undirected_csr() -> (Vec<u32>, Vec<u32>, usize) {
    let row_ptr: Vec<u32> = vec![0, 2, 5, 8, 11, 13, 14];
    let col_idx: Vec<u32> = vec![1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 4, 3, 5, 4];
    (row_ptr, col_idx, 6)
}

// ===========================================================================
// 1. bfs_level  —  INDEPENDENT HOST RE-DERIVATION (one frontier hop)
// ===========================================================================

#[test]
fn bfs_level_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (row_ptr, col_idx, n) = shared_undirected_csr();

    // Initial state: vertex 0 already visited at level 0, in the frontier.
    let depth = 0_u32;
    let mut level_init = vec![UNVISITED; n];
    level_init[0] = 0;
    let mut frontier_in = vec![0_u32; n];
    frontier_in[0] = 1;
    let frontier_out_init = vec![0_u32; n];

    // ---- Host re-derivation: for each u in the frontier, set every unvisited
    // neighbour v to depth+1 and flag it in frontier_out. (Equal writes from
    // multiple predecessors are idempotent, so the parallel kernel is race-free
    // for this single-source frontier.) Hand-verified expected for this graph:
    //   level    = [0, 1, 1, -1, -1, -1]
    //   frontier = [0, 1, 1,  0,  0,  0]
    let mut level_host = level_init.clone();
    let mut frontier_out_host = frontier_out_init.clone();
    for u in 0..n {
        if frontier_in[u] == 0 {
            continue;
        }
        let s = row_ptr[u] as usize;
        let e = row_ptr[u + 1] as usize;
        for &v in &col_idx[s..e] {
            let v = v as usize;
            if level_host[v] == UNVISITED {
                level_host[v] = depth + 1;
                frontier_out_host[v] = 1;
            }
        }
    }
    assert_eq!(level_host, vec![0, 1, 1, UNVISITED, UNVISITED, UNVISITED]);
    assert_eq!(frontier_out_host, vec![0, 1, 1, 0, 0, 0]);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::bfs_level_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bfs_level_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_level = DeviceBuffer::<u32>::from_host(&level_init).expect("d_level");
    let d_fin = DeviceBuffer::<u32>::from_host(&frontier_in).expect("d_fin");
    let d_fout = DeviceBuffer::<u32>::from_host(&frontier_out_init).expect("d_fout");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_level.as_device_ptr(),
                d_fin.as_device_ptr(),
                d_fout.as_device_ptr(),
                n as u32,
                depth,
            ),
        )
        .expect("launch bfs_level_kernel");
    stream.synchronize().expect("sync");

    let mut level_gpu = vec![0_u32; n];
    let mut frontier_gpu = vec![0_u32; n];
    d_level.copy_to_host(&mut level_gpu).expect("copy level");
    d_fout
        .copy_to_host(&mut frontier_gpu)
        .expect("copy frontier");

    assert_eq!(level_gpu, level_host, "bfs_level: level array mismatch");
    assert_eq!(
        frontier_gpu, frontier_out_host,
        "bfs_level: frontier_out mismatch"
    );
}

// ===========================================================================
// 2. dijkstra_relax  —  INDEPENDENT HOST RE-DERIVATION (one edge relaxation)
// ===========================================================================

#[test]
fn dijkstra_relax_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (row_ptr, col_idx, n) = shared_undirected_csr();
    let n_edges = col_idx.len();

    // Per-directed-edge weights (parallel to col_idx). Only the edges leaving
    // the source vertex `u = 0` (col_idx[0]=1 w=2.0, col_idx[1]=5.0) are read.
    let mut weights = vec![1.0_f32; n_edges];
    weights[0] = 2.0; // edge 0->1
    weights[1] = 5.0; // edge 0->2
    let source = 0_u32;

    // dist[source]=0, others a finite ceiling so relaxation strictly improves.
    let dist_init: Vec<f32> = vec![0.0, 10.0, 10.0, 10.0, 10.0, 10.0];
    let frontier_init = vec![0_u32; n];

    // ---- Host re-derivation: for each neighbour v of source, candidate =
    // dist[source] + w; if candidate < dist[v] then dist[v]=candidate,
    // frontier[v]=1. Hand-verified:
    //   dist     = [0, 2, 5, 10, 10, 10]
    //   frontier = [0, 1, 1,  0,  0,  0]
    let mut dist_host = dist_init.clone();
    let mut frontier_host = frontier_init.clone();
    {
        let u = source as usize;
        let s = row_ptr[u] as usize;
        let e = row_ptr[u + 1] as usize;
        for j in s..e {
            let v = col_idx[j] as usize;
            let cand = dist_host[u] + weights[j];
            if cand < dist_host[v] {
                dist_host[v] = cand;
                frontier_host[v] = 1;
            }
        }
    }
    assert_eq!(dist_host, vec![0.0, 2.0, 5.0, 10.0, 10.0, 10.0]);
    assert_eq!(frontier_host, vec![0, 1, 1, 0, 0, 0]);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::dijkstra_relax_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dijkstra_relax_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_w = DeviceBuffer::<f32>::from_host(&weights).expect("d_w");
    let d_dist = DeviceBuffer::<f32>::from_host(&dist_init).expect("d_dist");
    let d_front = DeviceBuffer::<u32>::from_host(&frontier_init).expect("d_front");

    // One thread per edge of the source row; a 32-wide block covers the 2 edges,
    // extra lanes are masked by the kernel's `j >= row_ptr[u+1]` guard.
    let params = LaunchParams::new(1_u32, 32_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_w.as_device_ptr(),
                d_dist.as_device_ptr(),
                d_front.as_device_ptr(),
                n as u32,
                source,
            ),
        )
        .expect("launch dijkstra_relax_kernel");
    stream.synchronize().expect("sync");

    let mut dist_gpu = vec![0.0_f32; n];
    let mut frontier_gpu = vec![0_u32; n];
    d_dist.copy_to_host(&mut dist_gpu).expect("copy dist");
    d_front
        .copy_to_host(&mut frontier_gpu)
        .expect("copy frontier");

    let (rel, abs) = worst_diff(&dist_gpu, &dist_host);
    for k in 0..n {
        assert!(
            close(dist_gpu[k], dist_host[k], 1e-5, 1e-6),
            "dijkstra dist[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            dist_gpu[k],
            dist_host[k]
        );
    }
    assert_eq!(frontier_gpu, frontier_host, "dijkstra: frontier mismatch");
}

// ===========================================================================
// 3. pagerank_step  —  INDEPENDENT HOST RE-DERIVATION (one power-iteration)
// ===========================================================================

#[test]
fn pagerank_step_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // Directed graph 0->1, 0->2, 1->2, 2->0.
    //   out_degree = [2, 1, 1]
    // Transposed CSR (in-neighbours of each v):
    //   v=0: [2]   v=1: [0]   v=2: [0,1]
    let n = 3_usize;
    let damping = 0.85_f32;
    let row_ptr_t: Vec<u32> = vec![0, 1, 2, 4];
    let col_idx_t: Vec<u32> = vec![2, 0, 0, 1];
    let out_degree: Vec<u32> = vec![2, 1, 1];
    let rank_in: Vec<f32> = vec![1.0 / 3.0; n];

    // ---- Host re-derivation: teleport = (1-d)/n; rank_out[v] = teleport +
    // d * sum_{u in in(v)} rank_in[u]/out_degree[u]. Hand-verified:
    //   rank_out ≈ [0.333333, 0.191667, 0.475000]
    let teleport = (1.0 - damping) / n as f32;
    let mut rank_host = vec![0.0_f32; n];
    for v in 0..n {
        let s = row_ptr_t[v] as usize;
        let e = row_ptr_t[v + 1] as usize;
        let mut sum = 0.0_f32;
        for j in s..e {
            let u = col_idx_t[j] as usize;
            sum += rank_in[u] / out_degree[u] as f32;
        }
        rank_host[v] = damping * sum + teleport;
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::pagerank_step_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pagerank_step_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr_t).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx_t).expect("d_col");
    let d_deg = DeviceBuffer::<u32>::from_host(&out_degree).expect("d_deg");
    let d_in = DeviceBuffer::<f32>::from_host(&rank_in).expect("d_in");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 32_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_deg.as_device_ptr(),
                d_in.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
                damping,
            ),
        )
        .expect("launch pagerank_step_kernel");
    stream.synchronize().expect("sync");

    let mut rank_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut rank_gpu).expect("copy rank");

    let (rel, abs) = worst_diff(&rank_gpu, &rank_host);
    for v in 0..n {
        assert!(
            close(rank_gpu[v], rank_host[v], 1e-5, 1e-6),
            "pagerank rank[{v}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            rank_gpu[v],
            rank_host[v]
        );
    }
}

// ===========================================================================
// 4. fw_inner  —  INDEPENDENT HOST RE-DERIVATION (one Floyd-Warshall k-update)
// ===========================================================================

#[test]
fn fw_inner_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 4_usize;
    let k = 1_u32;
    // INF = 1e9 is exactly representable in f32 (1e9 = 15_625_000 * 64).
    let inf = 1.0e9_f32;
    #[rustfmt::skip]
    let dist_init: Vec<f32> = vec![
        0.0, 3.0, inf, 7.0,
        8.0, 0.0, 2.0, inf,
        5.0, inf, 0.0, 1.0,
        2.0, inf, inf, 0.0,
    ];

    // ---- Host re-derivation: dist[i*n+j] = min(dist[i*n+j],
    // dist[i*n+k] + dist[k*n+j]) for the single fixed k. The only entry that
    // improves for k=1 is dist[0][2] = dist[0][1]+dist[1][2] = 3+2 = 5.
    let mut dist_host = dist_init.clone();
    let kk = k as usize;
    for i in 0..n {
        let dik = dist_init[i * n + kk];
        for j in 0..n {
            let cand = dik + dist_init[kk * n + j];
            if cand < dist_host[i * n + j] {
                dist_host[i * n + j] = cand;
            }
        }
    }
    assert_eq!(dist_host[0 * n + 2], 5.0, "host fw expected dist[0][2]=5");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::fw_inner_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "fw_inner_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_dist = DeviceBuffer::<f32>::from_host(&dist_init).expect("d_dist");

    // Grid = (n, n), block = (1, 1): ctaid.x = j, ctaid.y = i.
    let params = LaunchParams::new((n as u32, n as u32), (1u32, 1u32));
    kernel
        .launch(&params, &stream, &(d_dist.as_device_ptr(), n as u32, k))
        .expect("launch fw_inner_kernel");
    stream.synchronize().expect("sync");

    let mut dist_gpu = vec![0.0_f32; n * n];
    d_dist.copy_to_host(&mut dist_gpu).expect("copy dist");

    let (rel, abs) = worst_diff(&dist_gpu, &dist_host);
    for idx in 0..n * n {
        assert!(
            close(dist_gpu[idx], dist_host[idx], 1e-5, 1e-3),
            "fw dist[{idx}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            dist_gpu[idx],
            dist_host[idx]
        );
    }
}

// ===========================================================================
// 5. triangle_count  —  INDEPENDENT HOST RE-DERIVATION (per-vertex triangles)
// ===========================================================================

#[test]
fn triangle_count_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (row_ptr, col_idx, n) = shared_undirected_csr();

    // ---- Host re-derivation mirroring the kernel exactly: for u, for each
    // neighbour v>u, for each later neighbour w>v, count when edge (v,w) exists
    // (scan row v for w). Hand-verified per-vertex counts: [1,1,0,0,0,0]
    // (triangles {0,1,2} counted at u=0, {1,2,3} counted at u=1); total = 2.
    let edge_exists = |a: u32, w: u32| -> bool {
        let s = row_ptr[a as usize] as usize;
        let e = row_ptr[a as usize + 1] as usize;
        col_idx[s..e].contains(&w)
    };
    let mut count_host = vec![0_u32; n];
    for u in 0..n {
        let s = row_ptr[u] as usize;
        let e = row_ptr[u + 1] as usize;
        let mut c = 0_u32;
        for j in s..e {
            let v = col_idx[j];
            if v <= u as u32 {
                continue;
            }
            for &w in &col_idx[j + 1..e] {
                if w <= v {
                    continue;
                }
                if edge_exists(v, w) {
                    c += 1;
                }
            }
        }
        count_host[u] = c;
    }
    assert_eq!(count_host, vec![1, 1, 0, 0, 0, 0]);
    assert_eq!(count_host.iter().sum::<u32>(), 2);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::triangle_count_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "triangle_count_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_count = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_count");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_count.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch triangle_count_kernel");
    stream.synchronize().expect("sync");

    let mut count_gpu = vec![0_u32; n];
    d_count.copy_to_host(&mut count_gpu).expect("copy count");

    assert_eq!(
        count_gpu,
        count_host,
        "triangle_count: per-vertex counts mismatch (gpu sum={}, host sum={})",
        count_gpu.iter().sum::<u32>(),
        count_host.iter().sum::<u32>()
    );
}

// ===========================================================================
// 6. csr_spmv_bool  —  INDEPENDENT HOST RE-DERIVATION (boolean OR matvec)
// ===========================================================================

#[test]
fn csr_spmv_bool_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (row_ptr, col_idx, n) = shared_undirected_csr();

    // x active only at vertex 0; one boolean SpMV = one BFS expansion hop.
    let mut x = vec![0_u32; n];
    x[0] = 1;

    // ---- Host re-derivation: y[i] = OR over neighbours j of x[col_idx[j]].
    // Hand-verified: y = [0,1,1,0,0,0] (neighbours of 0 are {1,2}).
    let mut y_host = vec![0_u32; n];
    for i in 0..n {
        let s = row_ptr[i] as usize;
        let e = row_ptr[i + 1] as usize;
        let mut acc = 0_u32;
        for &c in &col_idx[s..e] {
            acc |= x[c as usize];
        }
        y_host[i] = acc;
    }
    assert_eq!(y_host, vec![0, 1, 1, 0, 0, 0]);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::csr_spmv_bool_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "csr_spmv_bool_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_x = DeviceBuffer::<u32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_y");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch csr_spmv_bool_kernel");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0_u32; n];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    assert_eq!(y_gpu, y_host, "csr_spmv_bool: y mismatch");
}

// ===========================================================================
// 7. community_label  —  INDEPENDENT HOST RE-DERIVATION (min-label propagation)
// ===========================================================================
//
// NOTE on intent: the kernel's *header* doc-comment describes the overall
// label-propagation algorithm ("most-frequent label among neighbours"), but the
// kernel's *inline* comment and actual instructions implement a single
// min-label propagation step — `label_out[u] = min(label_in[u], min over
// neighbours label_in)` — with frequency aggregation explicitly deferred to the
// CPU. We validate the kernel's actual, internally-consistent min-label
// behaviour (a legitimate connected-component-style propagation primitive).

#[test]
fn community_label_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (row_ptr, col_idx, n) = shared_undirected_csr();

    // Distinct labels so the min is unambiguous.
    let label_in: Vec<u32> = vec![5, 3, 4, 2, 1, 0];

    // ---- Host re-derivation: label_out[u] = min(own, min over neighbours).
    // Hand-verified: [3,2,2,1,0,0].
    let mut label_host = vec![0_u32; n];
    for u in 0..n {
        let s = row_ptr[u] as usize;
        let e = row_ptr[u + 1] as usize;
        let mut best = label_in[u];
        for &c in &col_idx[s..e] {
            let nl = label_in[c as usize];
            if nl < best {
                best = nl;
            }
        }
        label_host[u] = best;
    }
    assert_eq!(label_host, vec![3, 2, 2, 1, 0, 0]);

    // ---- GPU ----
    let ptx = crate::ptx_kernels::community_label_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "community_label_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_lin = DeviceBuffer::<u32>::from_host(&label_in).expect("d_lin");
    let d_lout = DeviceBuffer::<u32>::from_host(&vec![0_u32; n]).expect("d_lout");

    let block = 64_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_lin.as_device_ptr(),
                d_lout.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch community_label_kernel");
    stream.synchronize().expect("sync");

    let mut label_gpu = vec![0_u32; n];
    d_lout.copy_to_host(&mut label_gpu).expect("copy labels");

    assert_eq!(label_gpu, label_host, "community_label: labels mismatch");
}
