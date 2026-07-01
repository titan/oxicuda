//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts numerical equivalence
//! to the crate's CPU reference. The launch ABI mirrors the working
//! `oxicuda-snn` canary: device buffers are passed as their `CUdeviceptr`
//! (a `.param .u64`), scalars as the matching Rust scalar (`.param .u32` /
//! `.param .f32`), in the kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance / bit-exact to
//!   a `pub` CPU function the kernel mirrors:
//!   `l2_distance_batch` ↔ [`crate::distance::l2::l2_sq_all`],
//!   `ip_distance_batch` ↔ [`crate::distance::inner_product::ip`],
//!   `hnsw_neighbor_eval` ↔ [`crate::distance::l2::l2_sq`],
//!   `topk_select` ↔ [`crate::topk::select::select_topk`].
//! * **Independent host re-derivation** — the op is fused into a larger CPU
//!   routine (or the inputs the kernel needs are not exposed by the crate type),
//!   so the oracle is an independent Rust re-implementation of the kernel's
//!   documented arithmetic: `pq_adc_table` (sub-vector L2², matching
//!   [`crate::pq::adc::build_adc_table`]), `ivf_assign` (nearest-centroid
//!   argmin), and `lsh_random_proj` (sign-bit packing, matching the body of
//!   `crate::lsh::random_proj::RandomProjLsh::hash`, whose projection matrix is a
//!   private field and therefore cannot be injected into the kernel). These
//!   still genuinely fail if ptxas miscompiles or the PTX has a wrong constant /
//!   shift / index, because the host code is independent of the JIT-compiled PTX.
//!
//! ## PTX bug found and fixed
//!
//! ### `topk_select` — invalid PTX (never loaded) + wrong algorithm
//!
//! The shipped kernel was rejected by ptxas on every GPU and, even if it had
//! loaded, did not compute a top-K:
//!
//! 1. **Invalid PTX — scaled-register shared-memory operands.** Every shared
//!    access used `[sh_dists + %tid * 4]` / `[sh_indices + %partner * 4]`. PTX
//!    memory operands do not support a `symbol + reg*imm` form; ptxas rejected
//!    the module (`Module::from_ptx` failed). Fixed by materialising the byte
//!    address in a register first (`mov.u32 base, sh_dists; mad.lo.u32 addr,
//!    tid, 4, base; [addr]`), mirroring the `oxicuda-infer` softmax fix.
//! 2. **`.reg` declarations after instructions.** `%partner`, `%even`, `%p_gt`
//!    were declared mid-body; all register declarations now sit at the function
//!    top.
//! 3. **Wrong algorithm.** The body performed a single odd-even compare-exchange
//!    pass (`tid ^ 1`), which does not sort, so slots `0..K` were not the K
//!    minima. Replaced with a full ascending **bitonic sort** over the shared
//!    arrays (carrying indices), after which slots `0..K` hold the K smallest
//!    distances in ascending order — exactly `select_topk`'s contract.
//!
//! The other six kernels (`l2_distance_batch`, `ip_distance_batch`,
//! `pq_adc_table`, `hnsw_neighbor_eval`, `ivf_assign`, `lsh_random_proj`) loaded
//! and matched their oracles on the first launch — no bugs found.
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
/// A failure here means ptxas rejected the hand-written PTX — a real bug, so we
/// panic with the compiler diagnostic rather than skipping.
fn load_kernel(ptx: &str, entry: &str) -> Kernel {
    let module = Module::from_ptx(ptx)
        .unwrap_or_else(|e| panic!("PTX JIT compile failed for `{entry}`: {e}"));
    Kernel::from_module(Arc::new(module), entry)
        .unwrap_or_else(|e| panic!("kernel `{entry}` not found in module: {e}"))
}

// ===========================================================================
// 1. l2_distance_batch  —  CRATE ORACLE (crate::distance::l2::l2_sq_all)
// ===========================================================================

#[test]
fn l2_distance_batch_matches_cpu() {
    use crate::distance::l2::l2_sq_all;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_q = 4_usize; // B
    let n_db = 6_usize; // N
    let dim = 5_usize; // D

    let mut rng = LcgRng::new(0x102D15);
    let q: Vec<f32> = (0..n_q * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let x: Vec<f32> = (0..n_db * dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    let expected = l2_sq_all(&q, &x, n_q, n_db, dim).expect("cpu l2_sq_all");

    let ptx = crate::ptx_kernels::l2_distance_batch_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "l2_distance_batch");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f32>::from_host(&q).expect("d_q");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_q * n_db]).expect("d_out");

    // n -> x dim, b -> y dim; one block covers the whole (N, B) tile.
    let params = LaunchParams::new((1_u32, 1_u32), (n_db as u32, n_q as u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_x.as_device_ptr(),
                d_out.as_device_ptr(),
                n_q as u32,
                n_db as u32,
                dim as u32,
            ),
        )
        .expect("launch l2_distance_batch");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_q * n_db];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-4),
            "l2[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 2. ip_distance_batch  —  CRATE ORACLE (crate::distance::inner_product::ip)
// ===========================================================================

#[test]
fn ip_distance_batch_matches_cpu() {
    use crate::distance::inner_product::ip;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_q = 4_usize; // B
    let n_db = 6_usize; // N
    let dim = 5_usize; // D

    let mut rng = LcgRng::new(0x0019_2D15);
    let q: Vec<f32> = (0..n_q * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let x: Vec<f32> = (0..n_db * dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // CPU oracle: out[b, n] = <q[b], x[n]>.
    let mut expected = vec![0.0_f32; n_q * n_db];
    for b in 0..n_q {
        let qb = &q[b * dim..(b + 1) * dim];
        for n in 0..n_db {
            let xn = &x[n * dim..(n + 1) * dim];
            expected[b * n_db + n] = ip(qb, xn).expect("cpu ip");
        }
    }

    let ptx = crate::ptx_kernels::ip_distance_batch_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ip_distance_batch");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f32>::from_host(&q).expect("d_q");
    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_q * n_db]).expect("d_out");

    let params = LaunchParams::new((1_u32, 1_u32), (n_db as u32, n_q as u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_x.as_device_ptr(),
                d_out.as_device_ptr(),
                n_q as u32,
                n_db as u32,
                dim as u32,
            ),
        )
        .expect("launch ip_distance_batch");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_q * n_db];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], expected[k], 1e-4, 1e-4),
            "ip[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 3. pq_adc_table  —  INDEPENDENT HOST RE-DERIVATION (matches build_adc_table)
// ===========================================================================

#[test]
fn pq_adc_table_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m = 3_usize; // subspaces
    let ksub = 4_usize; // codebook entries per subspace
    let dsub = 2_usize; // dims per subspace

    let mut rng = LcgRng::new(0x9DC0DE);
    let query: Vec<f32> = (0..m * dsub).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let centroids: Vec<f32> = (0..m * ksub * dsub)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // Host re-derivation: table[s*ksub + c] = Σ_d (query[s*dsub+d] - cent[(s*ksub+c)*dsub+d])^2
    let mut expected = vec![0.0_f32; m * ksub];
    for s in 0..m {
        for c in 0..ksub {
            let mut acc = 0.0_f32;
            for d in 0..dsub {
                let qd = query[s * dsub + d];
                let cd = centroids[(s * ksub + c) * dsub + d];
                let diff = qd - cd;
                acc += diff * diff;
            }
            expected[s * ksub + c] = acc;
        }
    }

    let ptx = crate::ptx_kernels::pq_adc_table_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "pq_adc_table");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f32>::from_host(&query).expect("d_q");
    let d_c = DeviceBuffer::<f32>::from_host(&centroids).expect("d_c");
    let d_t = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; m * ksub]).expect("d_t");

    // grid.x = m (one block per subspace); block.x = ksub (one thread per entry).
    let params = LaunchParams::new((m as u32, 1_u32), (ksub as u32, 1_u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_c.as_device_ptr(),
                d_t.as_device_ptr(),
                m as u32,
                ksub as u32,
                dsub as u32,
            ),
        )
        .expect("launch pq_adc_table");
    stream.synchronize().expect("sync");

    let mut table_gpu = vec![0.0_f32; m * ksub];
    d_t.copy_to_host(&mut table_gpu).expect("copy table");

    let (rel, abs) = worst_diff(&table_gpu, &expected);
    for k in 0..table_gpu.len() {
        assert!(
            close(table_gpu[k], expected[k], 1e-4, 1e-5),
            "pq_adc[{k}] mismatch: gpu={} host={} (worst rel={rel:e} abs={abs:e})",
            table_gpu[k],
            expected[k]
        );
    }
}

// ===========================================================================
// 4. hnsw_neighbor_eval  —  CRATE ORACLE (crate::distance::l2::l2_sq)
// ===========================================================================

#[test]
fn hnsw_neighbor_eval_matches_cpu() {
    use crate::distance::l2::l2_sq;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 5_usize;
    let n_db = 7_usize;
    let k = 4_usize; // candidates

    let mut rng = LcgRng::new(0x4B57);
    let query: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let dataset: Vec<f32> = (0..n_db * dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let candidates: Vec<u32> = vec![6, 0, 3, 5]; // valid indices < n_db, in arbitrary order

    // CPU oracle: out[i] = L2²(query, dataset[candidates[i]]).
    let mut expected = vec![0.0_f32; k];
    for (i, &cand) in candidates.iter().enumerate() {
        let row = &dataset[cand as usize * dim..(cand as usize + 1) * dim];
        expected[i] = l2_sq(&query, row).expect("cpu l2_sq");
    }

    let ptx = crate::ptx_kernels::hnsw_neighbor_eval_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hnsw_neighbor_eval");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f32>::from_host(&query).expect("d_q");
    let d_d = DeviceBuffer::<f32>::from_host(&dataset).expect("d_d");
    let d_c = DeviceBuffer::<u32>::from_host(&candidates).expect("d_c");
    let d_o = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; k]).expect("d_o");

    let params = LaunchParams::new((1_u32, 1_u32), (k as u32, 1_u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_d.as_device_ptr(),
                d_c.as_device_ptr(),
                d_o.as_device_ptr(),
                dim as u32,
                k as u32,
            ),
        )
        .expect("launch hnsw_neighbor_eval");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; k];
    d_o.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &expected);
    for i in 0..k {
        assert!(
            close(out_gpu[i], expected[i], 1e-4, 1e-5),
            "hnsw_eval[{i}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[i],
            expected[i]
        );
    }
}

// ===========================================================================
// 5. ivf_assign  —  INDEPENDENT HOST RE-DERIVATION (nearest-centroid argmin)
// ===========================================================================

#[test]
fn ivf_assign_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 3_usize;
    let n_c = 4_usize;
    let b = 5_usize;

    // Well-separated centroids so the nearest-centroid argmin is unambiguous
    // (the GPU's fma accumulation vs the host's sequential sum can only flip a
    // near-tie; a large margin keeps the integer assignment bit-exact).
    let centroids: Vec<f32> = vec![
        0.0, 0.0, 0.0, // c0
        10.0, 0.0, 0.0, // c1
        0.0, 10.0, 0.0, // c2
        0.0, 0.0, 10.0, // c3
    ];
    let truth = [0_usize, 1, 2, 3, 1];

    let mut rng = LcgRng::new(0x1FA551);
    let mut vectors = vec![0.0_f32; b * dim];
    for (i, &c) in truth.iter().enumerate() {
        for d in 0..dim {
            vectors[i * dim + d] = centroids[c * dim + d] + (rng.next_f32() - 0.5);
        }
    }

    // Host argmin oracle (first-wins on strict <, matching the kernel's
    // `setp.lt.f32`), plus a margin guard so the comparison is honest.
    let mut expected = vec![0_u32; b];
    for i in 0..b {
        let v = &vectors[i * dim..(i + 1) * dim];
        let mut best = 0_usize;
        let mut best_d = f32::INFINITY;
        let mut second_d = f32::INFINITY;
        for c in 0..n_c {
            let cc = &centroids[c * dim..(c + 1) * dim];
            let d: f32 = v
                .iter()
                .zip(cc.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            if d < best_d {
                second_d = best_d;
                best_d = d;
                best = c;
            } else if d < second_d {
                second_d = d;
            }
        }
        assert!(
            second_d - best_d > 1.0,
            "test setup: vector {i} argmin margin too small ({best_d} vs {second_d})"
        );
        assert_eq!(best, truth[i], "test setup: vector {i} argmin != truth");
        expected[i] = best as u32;
    }

    let ptx = crate::ptx_kernels::ivf_assign_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "ivf_assign");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_v = DeviceBuffer::<f32>::from_host(&vectors).expect("d_v");
    let d_c = DeviceBuffer::<f32>::from_host(&centroids).expect("d_c");
    let d_a = DeviceBuffer::<u32>::from_host(&vec![0_u32; b]).expect("d_a");

    let params = LaunchParams::new((1_u32, 1_u32), (b as u32, 1_u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_v.as_device_ptr(),
                d_c.as_device_ptr(),
                d_a.as_device_ptr(),
                b as u32,
                n_c as u32,
                dim as u32,
            ),
        )
        .expect("launch ivf_assign");
    stream.synchronize().expect("sync");

    let mut assign_gpu = vec![0_u32; b];
    d_a.copy_to_host(&mut assign_gpu).expect("copy assign");

    for i in 0..b {
        assert_eq!(
            assign_gpu[i], expected[i],
            "ivf_assign[{i}] mismatch: gpu={} host={}",
            assign_gpu[i], expected[i]
        );
    }
}

// ===========================================================================
// 6. lsh_random_proj  —  INDEPENDENT HOST RE-DERIVATION (sign-bit packing)
// ===========================================================================

#[test]
fn lsh_random_proj_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_hashes = 8_usize; // K
    let b = 3_usize; // B rows
    let dim = 4_usize; // D
    let stride = n_hashes.div_ceil(32); // u32 words per row

    let mut rng = LcgRng::new(0x154321);
    let x: Vec<f32> = (0..b * dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let w: Vec<f32> = (0..n_hashes * dim).map(|_| rng.next_normal()).collect();

    // Host re-derivation of `RandomProjLsh::hash`: bit (b, j) set iff <W[j], x[b]> >= 0,
    // packed little-endian into out[b*stride + j/32]. (W is a private field of the
    // crate type, so the projection cannot be injected — we replicate the body.)
    let mut expected = vec![0_u32; b * stride];
    for row in 0..b {
        let xb = &x[row * dim..(row + 1) * dim];
        for j in 0..n_hashes {
            let wj = &w[j * dim..(j + 1) * dim];
            let dot: f32 = wj.iter().zip(xb.iter()).map(|(a, c)| a * c).sum();
            // Keep every projection comfortably away from zero so the sign bit is
            // not a single-ulp knife-edge between fma and sequential accumulation.
            assert!(
                dot.abs() > 1e-2,
                "test setup: projection ({row},{j}) too close to zero ({dot})"
            );
            if dot >= 0.0 {
                expected[row * stride + j / 32] |= 1_u32 << (j % 32);
            }
        }
    }

    let ptx = crate::ptx_kernels::lsh_random_proj_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lsh_random_proj");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_w = DeviceBuffer::<f32>::from_host(&w).expect("d_w");
    let d_o = DeviceBuffer::<u32>::from_host(&vec![0_u32; b * stride]).expect("d_o");

    // j -> x dim (hash index), row -> y dim.
    let params = LaunchParams::new((1_u32, 1_u32), (n_hashes as u32, b as u32));
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_w.as_device_ptr(),
                d_o.as_device_ptr(),
                b as u32,
                n_hashes as u32,
                dim as u32,
            ),
        )
        .expect("launch lsh_random_proj");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; b * stride];
    d_o.copy_to_host(&mut out_gpu).expect("copy out");

    for k in 0..out_gpu.len() {
        assert_eq!(
            out_gpu[k], expected[k],
            "lsh_random_proj word[{k}] mismatch: gpu={:#010x} host={:#010x}",
            out_gpu[k], expected[k]
        );
    }
}

// ===========================================================================
// 7. topk_select  —  CRATE ORACLE (crate::topk::select::select_topk)
// ===========================================================================

#[test]
fn topk_select_matches_cpu() {
    use crate::topk::select::select_topk;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 50_usize;
    let block = 64_u32; // power of two >= n, <= shared capacity (64)
    let k = 8_usize;

    // Distinct distances (a shuffled permutation of 1.0..=N) so the K-smallest
    // ordering — and therefore the index assignment — is unambiguous.
    let mut dists: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let mut rng = LcgRng::new(0x709CA1);
    for i in (1..n).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        dists.swap(i, j);
    }

    // CPU oracle: K smallest (id, dist), ascending.
    let pairs: Vec<(usize, f32)> = dists.iter().copied().enumerate().collect();
    let expected = select_topk(&pairs, k);
    assert_eq!(expected.len(), k, "oracle returned wrong count");

    let ptx = crate::ptx_kernels::topk_select_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "topk_select");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_d = DeviceBuffer::<f32>::from_host(&dists).expect("d_d");
    let d_od = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; k]).expect("d_od");
    let d_oi = DeviceBuffer::<u32>::from_host(&vec![0_u32; k]).expect("d_oi");

    // One block of `block` threads sorts this query's N distances.
    let params = LaunchParams::new(1_u32, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_d.as_device_ptr(),
                d_od.as_device_ptr(),
                d_oi.as_device_ptr(),
                n as u32,
                k as u32,
            ),
        )
        .expect("launch topk_select");
    stream.synchronize().expect("sync");

    let mut od_gpu = vec![0.0_f32; k];
    let mut oi_gpu = vec![0_u32; k];
    d_od.copy_to_host(&mut od_gpu).expect("copy out_dists");
    d_oi.copy_to_host(&mut oi_gpu).expect("copy out_indices");

    for i in 0..k {
        let (exp_id, exp_d) = expected[i];
        assert!(
            close(od_gpu[i], exp_d, 1e-6, 1e-6),
            "topk dist[{i}] mismatch: gpu={} cpu={}",
            od_gpu[i],
            exp_d
        );
        assert_eq!(
            oi_gpu[i] as usize, exp_id,
            "topk idx[{i}] mismatch: gpu={} cpu={} (dist gpu={} cpu={})",
            oi_gpu[i], exp_id, od_gpu[i], exp_d
        );
    }
}
