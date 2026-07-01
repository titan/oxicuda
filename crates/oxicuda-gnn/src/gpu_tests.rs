//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies the
//! results back, and asserts numerical equivalence to the crate's CPU
//! reference. The launch ABI mirrors the working `oxicuda-snn` path: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars are
//! passed as the matching Rust scalar (`.param .u32` / `.param .f32`), in
//! declared order. CSR arrays (`row_ptr`, `col_idx`) are `u32` device buffers,
//! values / features are `f32` device buffers.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel mirrors:
//!   - `csr_spmv`       → [`crate::graph::csr::CsrGraph::spmv`] (explicit edge
//!     weights, `feat_dim = 1`) and an independent direct CSR loop.
//!   - `scatter_add`    → [`crate::message_passing::scatter::scatter_add`]
//!     (`feat_dim = 1`).
//!   - `softmax_edge`   → [`crate::message_passing::scatter::segment_softmax`]
//!     with per-source segment ids derived from `row_ptr`.
//!   - `aggregate_mean` → [`crate::message_passing::aggregate::aggregate_mean`]
//!     fed neighbour-feature messages keyed by source node.
//! * **Independent host re-derivation** — the kernel has no single dedicated
//!   crate function (the op is fused into a larger routine on the CPU), so the
//!   oracle is an independent Rust re-implementation of the kernel's *documented*
//!   arithmetic, cross-checked against a `pub` crate primitive where one exists:
//!   - `gat_attention` → host `aᵀ[src‖dst]` dot + [`update::leaky_relu`].
//!   - `gin_combine`   → host `(1+ε)·self + aggr` (GIN's documented combine).
//!   - `topk_score`    → host `tanh(dot(x,p)/‖p‖)`, matching
//!     [`crate::pooling::topk_pool::TopKPool`]'s documented scoring formula.
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

/// `ceil(n / block)` as a grid size.
fn grid_1d(n: u32, block: u32) -> u32 {
    n.div_ceil(block)
}

/// A small hand-built CSR test graph shared by the SpMV / softmax / aggregate
/// tests. Node 3 is intentionally isolated (zero out-edges) to exercise the
/// degenerate path on the device.
///
/// Returns `(n_nodes, row_ptr [n_nodes+1], col_idx [n_edges])`.
fn test_csr() -> (usize, Vec<u32>, Vec<u32>) {
    // node 0 → {1,2,3}, 1 → {0,2}, 2 → {4}, 3 → {}, 4 → {0,1,5}, 5 → {3}
    let row_ptr = vec![0u32, 3, 5, 6, 6, 9, 10];
    let col_idx = vec![1u32, 2, 3, 0, 2, 4, 0, 1, 5, 3];
    (6, row_ptr, col_idx)
}

// ===========================================================================
// 1. csr_spmv  —  CRATE ORACLE (CsrGraph::spmv) + independent direct CSR loop
// ===========================================================================

#[test]
fn csr_spmv_matches_cpu() {
    use crate::graph::csr::CsrGraph;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let (n_nodes, row_ptr, col_idx) = test_csr();
    let n_edges = col_idx.len();

    let mut rng = LcgRng::new(0x00C5_5AF0);
    // Non-trivial values and input vector so a wrong column index or a dropped
    // term cannot accidentally still match.
    let values: Vec<f32> = (0..n_edges).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let x: Vec<f32> = (0..n_nodes).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // ---- CPU oracle A: CsrGraph::spmv with explicit edge weights (feat_dim=1).
    let row_ptr_usize: Vec<usize> = row_ptr.iter().map(|&v| v as usize).collect();
    let col_idx_usize: Vec<usize> = col_idx.iter().map(|&v| v as usize).collect();
    let g = CsrGraph::with_weights(
        n_nodes,
        row_ptr_usize.clone(),
        col_idx_usize.clone(),
        values.clone(),
    )
    .expect("build weighted CSR");
    let y_crate = g.spmv(&x, 1).expect("cpu spmv");

    // ---- CPU oracle B: independent direct loop (crate-internals independent).
    let mut y_direct = vec![0.0_f32; n_nodes];
    for i in 0..n_nodes {
        let start = row_ptr_usize[i];
        let end = row_ptr_usize[i + 1];
        let mut acc = 0.0_f32;
        for e in start..end {
            acc += values[e] * x[col_idx_usize[e]];
        }
        y_direct[i] = acc;
    }
    for i in 0..n_nodes {
        assert!(
            close(y_crate[i], y_direct[i], 1e-6, 1e-7),
            "the two CPU oracles disagree at {i}: {} vs {}",
            y_crate[i],
            y_direct[i]
        );
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::csr_spmv_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "csr_spmv");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_val = DeviceBuffer::<f32>::from_host(&values).expect("d_val");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_y = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_nodes]).expect("d_y");

    // warp-per-row: 32 threads per row.
    let block = 256_u32;
    let grid = grid_1d(n_nodes as u32 * 32, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_val.as_device_ptr(),
                d_x.as_device_ptr(),
                d_y.as_device_ptr(),
                n_nodes as u32,
            ),
        )
        .expect("launch csr_spmv");
    stream.synchronize().expect("sync");

    let mut y_gpu = vec![0.0_f32; n_nodes];
    d_y.copy_to_host(&mut y_gpu).expect("copy y");

    let (rel, abs) = worst_diff(&y_gpu, &y_crate);
    for i in 0..n_nodes {
        // GPU warp shfl reduction sums in tree order vs the CPU's sequential
        // order: ~1 ulp per term, far inside 1e-4 relative.
        assert!(
            close(y_gpu[i], y_crate[i], 1e-4, 1e-6),
            "csr_spmv y[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            y_gpu[i],
            y_crate[i]
        );
    }
    // Isolated node 3 must be exactly zero.
    assert_eq!(y_gpu[3], 0.0, "isolated row 3 must be 0, got {}", y_gpu[3]);
}

// ===========================================================================
// 2. scatter_add  —  CRATE ORACLE (message_passing::scatter::scatter_add)
// ===========================================================================

#[test]
fn scatter_add_matches_cpu() {
    use crate::message_passing::scatter::scatter_add;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 64_usize; // number of scattered elements (feat_dim = 1)
    let n_out = 8_usize;
    let mut rng = LcgRng::new(0x05CA_77E5);

    let idx: Vec<u32> = (0..n).map(|_| rng.next_u32() % n_out as u32).collect();
    let src: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // ---- CPU oracle: scatter_add with feat_dim = 1.
    let idx_usize: Vec<usize> = idx.iter().map(|&v| v as usize).collect();
    let out_cpu = scatter_add(&src, &idx_usize, n_out, 1).expect("cpu scatter_add");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::scatter_add_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "scatter_add");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_idx = DeviceBuffer::<u32>::from_host(&idx).expect("d_idx");
    let d_src = DeviceBuffer::<f32>::from_host(&src).expect("d_src");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_out]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_idx.as_device_ptr(),
                d_src.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch scatter_add");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_out];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for o in 0..n_out {
        // Atomic float adds run in nondeterministic order; with ~8 small terms
        // per bucket the order-dependent rounding stays well inside 1e-4.
        assert!(
            close(out_gpu[o], out_cpu[o], 1e-4, 1e-5),
            "scatter_add out[{o}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[o],
            out_cpu[o]
        );
    }
}

// ===========================================================================
// 3. gat_attention  —  INDEPENDENT HOST RE-DERIVATION + crate leaky_relu
// ===========================================================================

#[test]
fn gat_attention_matches_cpu() {
    use crate::message_passing::update::leaky_relu;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_edges = 40_usize;
    let feat_dim = 6_usize;
    let slope = 0.2_f32; // hard-wired in the kernel (f32_hex(0.2))
    let mut rng = LcgRng::new(0x006A_7A11);

    // Per-edge projected source / destination features and the [2*fd] attention
    // vector a = [a_src ‖ a_dst].
    let src_feat: Vec<f32> = (0..n_edges * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let dst_feat: Vec<f32> = (0..n_edges * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let a: Vec<f32> = (0..2 * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // ---- CPU oracle: raw = aᵀ[src‖dst], then crate leaky_relu(0.2). ----
    let mut score_cpu = vec![0.0_f32; n_edges];
    for e in 0..n_edges {
        let mut raw = 0.0_f32;
        for k in 0..feat_dim {
            raw += src_feat[e * feat_dim + k] * a[k];
            raw += dst_feat[e * feat_dim + k] * a[feat_dim + k];
        }
        score_cpu[e] = leaky_relu(&[raw], slope)[0];
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::gat_attention_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gat_attention");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_src = DeviceBuffer::<f32>::from_host(&src_feat).expect("d_src");
    let d_dst = DeviceBuffer::<f32>::from_host(&dst_feat).expect("d_dst");
    let d_a = DeviceBuffer::<f32>::from_host(&a).expect("d_a");
    let d_score = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_edges]).expect("d_score");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_edges as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_src.as_device_ptr(),
                d_dst.as_device_ptr(),
                d_a.as_device_ptr(),
                d_score.as_device_ptr(),
                feat_dim as u32,
                n_edges as u32,
            ),
        )
        .expect("launch gat_attention");
    stream.synchronize().expect("sync");

    let mut score_gpu = vec![0.0_f32; n_edges];
    d_score.copy_to_host(&mut score_gpu).expect("copy score");

    let (rel, abs) = worst_diff(&score_gpu, &score_cpu);
    for e in 0..n_edges {
        // Pure fma + select; ~1 ulp accumulation divergence.
        assert!(
            close(score_gpu[e], score_cpu[e], 1e-4, 1e-6),
            "gat_attention score[{e}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            score_gpu[e],
            score_cpu[e]
        );
    }
}

// ===========================================================================
// 4. softmax_edge  —  CRATE ORACLE (segment_softmax); guards the base-e fix
// ===========================================================================

#[test]
fn softmax_edge_matches_cpu() {
    use crate::message_passing::scatter::segment_softmax;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let (n_nodes, row_ptr, col_idx) = test_csr();
    let n_edges = col_idx.len();
    let mut rng = LcgRng::new(0x0005_0F7A);

    // Per-edge raw attention scores in a moderate range.
    let scores: Vec<f32> = (0..n_edges).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // ---- CPU oracle: per-source softmax via segment_softmax. The segment id of
    // edge e is the source node whose CSR row contains e. ----
    let mut segment_ids = vec![0usize; n_edges];
    for i in 0..n_nodes {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        segment_ids[start..end].fill(i);
    }
    let alpha_cpu = segment_softmax(&scores, &segment_ids, n_nodes).expect("cpu segment_softmax");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::softmax_edge_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "softmax_edge");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_score = DeviceBuffer::<f32>::from_host(&scores).expect("d_score");
    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_alpha = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_edges]).expect("d_alpha");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_nodes as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_score.as_device_ptr(),
                d_row.as_device_ptr(),
                d_alpha.as_device_ptr(),
                n_nodes as u32,
            ),
        )
        .expect("launch softmax_edge");
    stream.synchronize().expect("sync");

    let mut alpha_gpu = vec![0.0_f32; n_edges];
    d_alpha.copy_to_host(&mut alpha_gpu).expect("copy alpha");

    // Per-source neighbour softmax must sum to 1 on the device.
    for i in 0..n_nodes {
        let start = row_ptr[i] as usize;
        let end = row_ptr[i + 1] as usize;
        if start == end {
            continue; // isolated source has no out-edges
        }
        let sum: f32 = alpha_gpu[start..end].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "softmax_edge source {i} weights sum to {sum}, not 1"
        );
    }

    // Equivalence to the base-e CPU softmax. ex2.approx (~2 ulp) keeps this well
    // inside 5e-4; a base-2 softmax (the pre-fix bug) would be ~20% off here.
    let (rel, abs) = worst_diff(&alpha_gpu, &alpha_cpu);
    for e in 0..n_edges {
        assert!(
            close(alpha_gpu[e], alpha_cpu[e], 5e-4, 1e-6),
            "softmax_edge alpha[{e}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            alpha_gpu[e],
            alpha_cpu[e]
        );
    }
}

// ===========================================================================
// 5. aggregate_mean  —  CRATE ORACLE (aggregate_mean over neighbour messages)
// ===========================================================================

#[test]
fn aggregate_mean_matches_cpu() {
    use crate::message_passing::aggregate::aggregate_mean;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let (n_nodes, row_ptr, col_idx) = test_csr();
    let n_edges = col_idx.len();
    let feat_dim = 4_usize;
    let mut rng = LcgRng::new(0x00A6_63EA);

    let feat: Vec<f32> = (0..n_nodes * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // ---- CPU oracle: build per-edge messages = neighbour features, keyed by
    // the source node, then mean-aggregate. The kernel divides each node's
    // neighbour-feature sum by its degree, which is exactly aggregate_mean's
    // in-degree normalisation over these messages. ----
    let mut messages = vec![0.0_f32; n_edges * feat_dim];
    let mut target_idx = vec![0usize; n_edges];
    for i in 0..n_nodes {
        for e in row_ptr[i] as usize..row_ptr[i + 1] as usize {
            let nbr = col_idx[e] as usize;
            target_idx[e] = i;
            for k in 0..feat_dim {
                messages[e * feat_dim + k] = feat[nbr * feat_dim + k];
            }
        }
    }
    let out_cpu = aggregate_mean(&messages, &target_idx, n_nodes, feat_dim).expect("cpu agg mean");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::aggregate_mean_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "aggregate_mean");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_feat = DeviceBuffer::<f32>::from_host(&feat).expect("d_feat");
    let d_row = DeviceBuffer::<u32>::from_host(&row_ptr).expect("d_row");
    let d_col = DeviceBuffer::<u32>::from_host(&col_idx).expect("d_col");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_nodes * feat_dim]).expect("d_out");

    let block = 256_u32;
    let grid = grid_1d((n_nodes * feat_dim) as u32, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_feat.as_device_ptr(),
                d_row.as_device_ptr(),
                d_col.as_device_ptr(),
                d_out.as_device_ptr(),
                feat_dim as u32,
                n_nodes as u32,
            ),
        )
        .expect("launch aggregate_mean");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_nodes * feat_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-4, 1e-6),
            "aggregate_mean out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k]
        );
    }
    // Isolated node 3 contributes no neighbours → its whole feature row is 0.
    for k in 0..feat_dim {
        assert_eq!(
            out_gpu[3 * feat_dim + k],
            0.0,
            "isolated node 3 feature {k} must be 0"
        );
    }
}

// ===========================================================================
// 6. gin_combine  —  INDEPENDENT HOST RE-DERIVATION ((1+ε)·self + aggr)
// ===========================================================================

#[test]
fn gin_combine_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 12_usize;
    let feat_dim = 5_usize;
    let eps = 0.37_f32;
    let mut rng = LcgRng::new(0x0006_1EC0);

    let self_feat: Vec<f32> = (0..n * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let aggr_feat: Vec<f32> = (0..n * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // ---- CPU oracle: the GIN combine, out = (1+ε)·self + aggr. ----
    let mut out_cpu = vec![0.0_f32; n * feat_dim];
    for k in 0..n * feat_dim {
        out_cpu[k] = (1.0 + eps) * self_feat[k] + aggr_feat[k];
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::gin_combine_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "gin_combine");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_self = DeviceBuffer::<f32>::from_host(&self_feat).expect("d_self");
    let d_aggr = DeviceBuffer::<f32>::from_host(&aggr_feat).expect("d_aggr");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n * feat_dim]).expect("d_out");

    let block = 256_u32;
    let grid = grid_1d((n * feat_dim) as u32, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_self.as_device_ptr(),
                d_aggr.as_device_ptr(),
                d_out.as_device_ptr(),
                eps,
                n as u32,
                feat_dim as u32,
            ),
        )
        .expect("launch gin_combine");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n * feat_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..out_gpu.len() {
        // GPU fma.rn((1+eps), self, aggr) is single-rounding vs host two-rounding.
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-5, 1e-6),
            "gin_combine out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k]
        );
    }
}

// ===========================================================================
// 7. topk_score  —  INDEPENDENT RE-DERIVATION of TopKPool's scoring formula
// ===========================================================================

#[test]
fn topk_score_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_nodes = 32_usize;
    let feat_dim = 6_usize;
    let mut rng = LcgRng::new(0x0007_0FC5);

    let feat: Vec<f32> = (0..n_nodes * feat_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    // Non-degenerate projection vector (‖proj‖ comfortably away from 0).
    let proj: Vec<f32> = (0..feat_dim).map(|_| rng.next_f32() + 0.5).collect();

    // ---- CPU oracle: TopKPool's documented score[i] = tanh(dot(x[i],p)/‖p‖). --
    let norm_sq: f32 = proj.iter().map(|&v| v * v).sum();
    let norm = norm_sq.sqrt().max(1e-12);
    let mut score_cpu = vec![0.0_f32; n_nodes];
    for i in 0..n_nodes {
        let dot: f32 = (0..feat_dim)
            .map(|k| feat[i * feat_dim + k] * proj[k])
            .sum();
        score_cpu[i] = (dot / norm).tanh();
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::topk_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "topk_score");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_feat = DeviceBuffer::<f32>::from_host(&feat).expect("d_feat");
    let d_proj = DeviceBuffer::<f32>::from_host(&proj).expect("d_proj");
    let d_score = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_nodes]).expect("d_score");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_nodes as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_feat.as_device_ptr(),
                d_proj.as_device_ptr(),
                d_score.as_device_ptr(),
                feat_dim as u32,
                n_nodes as u32,
            ),
        )
        .expect("launch topk_score");
    stream.synchronize().expect("sync");

    let mut score_gpu = vec![0.0_f32; n_nodes];
    d_score.copy_to_host(&mut score_gpu).expect("copy score");

    // tanh is bounded in (-1, 1).
    for (i, &s) in score_gpu.iter().enumerate() {
        assert!(
            s.abs() < 1.0 + 1e-4,
            "topk_score[{i}] = {s} out of tanh range"
        );
    }
    let (rel, abs) = worst_diff(&score_gpu, &score_cpu);
    for i in 0..n_nodes {
        // tanh via ex2.approx + div.approx (~few ulp); 5e-4 relative is generous
        // yet catches a wrong norm / missing tanh by orders of magnitude.
        assert!(
            close(score_gpu[i], score_cpu[i], 5e-4, 1e-6),
            "topk_score[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            score_gpu[i],
            score_cpu[i]
        );
    }
}
