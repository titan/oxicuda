//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it on the real CUDA device through
//! `oxicuda-launch`, copies the results back, and asserts equivalence to an
//! independent host re-derivation of the kernel's documented arithmetic. The
//! launch ABI mirrors the proven `oxicuda-snn` / `oxicuda-ot` / `oxicuda-recsys`
//! harnesses: device buffers are passed as their `CUdeviceptr` (`.param .u64`),
//! scalars as the matching Rust scalar (`.param .u32` / `.param .u64`), in the
//! kernel's declared parameter order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! Every kernel in this crate hashes its key with a simple multiply-shift /
//! multiply-add scheme that is *internal to the PTX* and does **not** mirror the
//! crate's higher-level hash families (Murmur3, tabulation, …) used by the CPU
//! sketch structs. There is therefore no `pub` CPU function that computes the
//! exact same value, so each oracle is an **independent host re-derivation** of
//! the kernel's documented integer/float arithmetic (the same tier the `ot` and
//! `snn` harnesses use for fused ops). These oracles still genuinely fail if
//! ptxas miscompiles, if the PTX has a wrong constant / shift / index, or if a
//! reduction races, because the host code is written independently of the
//! JIT-compiled PTX:
//!
//! * `cm_update_kernel`    — Count-Min increment: `table[row*w + col] += 1`,
//!   `col = lo32(a[row]*x + b[row]) mod w` (one disjoint cell per row).
//! * `cm_query_kernel`     — Count-Min point query: atomic-min reduction of the
//!   `d` per-row cells into `out[0]`.
//! * `hll_register_kernel` — HyperLogLog register max-update: `idx` from the low
//!   `p_bits`, `rho = clz32(lo32(hash >> p_bits)) + 1`, `reg[idx] = max(…)`.
//! * `bloom_insert_kernel` — Bloom set-bit: `bit = lo32(seed[i]*x) mod m`, OR a
//!   single bit into `bits[bit/32]` for each of `k` hashes.
//! * `minhash_sketch_kernel` — MinHash signature min-update over `k` hashes.
//! * `tdigest_merge_kernel`  — t-Digest centroid merge (f64), weighted mean.
//! * `reservoir_sample_kernel` — Vitter replacement step (u64 payload).
//!
//! None of the seven kernels is a hollow stub: each contains a real
//! `st.global` / `atom.global.*` write whose effect is asserted below. Every
//! test skips (returns early) when no CUDA device is present, so the suite stays
//! green on CPU-only machines.

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

/// JIT-compile `ptx` for the live device and look up `entry`.
///
/// A failure here means ptxas rejected the PTX — a real bug in the kernel
/// source, not a skip condition.
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

/// The exact 32-bit multiply-add hash the CM / MinHash kernels compute:
/// `lo32(a * x + b)` using wrapping 64-bit arithmetic (matches `mul.lo.u64`
/// + `add.u64` + `cvt.u32.u64`).
fn hash_mul_add_lo32(a: u64, b: u64, x: u64) -> u32 {
    a.wrapping_mul(x).wrapping_add(b) as u32
}

/// Relative-with-absolute-floor closeness for f64 comparisons.
fn close_f64(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

// ===========================================================================
// 1. cm_update  —  INDEPENDENT HOST RE-DERIVATION (Count-Min increment)
// ===========================================================================
//
// Each thread owns one row `row` of the `d x w` table. It computes
// `col = lo32(a[row]*x + b[row]) mod w` and does `atom.global.add.u32 +1` into
// `table[row*w + col]`. Distinct rows touch distinct cells, so the result is
// deterministic: every row's chosen cell becomes 1, all others stay 0.

#[test]
fn cm_update_increments_expected_cell() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let d = 6_usize;
    let w = 11_usize;
    let mut rng = LcgRng::new(0xC0FF_EE01);
    let a: Vec<u64> = (0..d).map(|_| rng.next_u64()).collect();
    let b: Vec<u64> = (0..d).map(|_| rng.next_u64()).collect();
    let x: u64 = rng.next_u64();

    // Host reference table.
    let mut table_expected = vec![0_u32; d * w];
    for row in 0..d {
        let col = (hash_mul_add_lo32(a[row], b[row], x) as usize) % w;
        table_expected[row * w + col] += 1;
    }

    let ptx = crate::ptx_kernels::cm_update_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cm_update_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_table = DeviceBuffer::<u32>::from_host(&vec![0_u32; d * w]).expect("d_table");
    let d_a = DeviceBuffer::<u64>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<u64>::from_host(&b).expect("d_b");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(d as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_table.as_device_ptr(),
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                w as u32,
                d as u32,
                x,
            ),
        )
        .expect("launch cm_update_kernel");
    stream.synchronize().expect("sync");

    let mut table_gpu = vec![0_u32; d * w];
    d_table.copy_to_host(&mut table_gpu).expect("copy table");

    // Exactly `d` cells must hold 1 (one per row); the rest must stay 0.
    assert_eq!(
        table_gpu.iter().filter(|&&c| c == 1).count(),
        d,
        "cm_update: expected exactly {d} incremented cells"
    );
    for k in 0..d * w {
        assert_eq!(
            table_gpu[k], table_expected[k],
            "cm_update: table[{k}] gpu={} cpu={}",
            table_gpu[k], table_expected[k]
        );
    }
}

// ===========================================================================
// 2. cm_query  —  INDEPENDENT HOST RE-DERIVATION (atomic-min reduction)
// ===========================================================================
//
// Each of the `d` row-threads reads `table[row*w + col(row)]` and reduces it
// into `out[0]` via `atom.global.min.u32`. The host oracle is the minimum of
// those `d` cell values. `out[0]` is seeded with `u32::MAX` so the reduction is
// well-defined.

#[test]
fn cm_query_reduces_min_over_rows() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let d = 6_usize;
    let w = 11_usize;
    let mut rng = LcgRng::new(0xC0FF_EE02);
    let a: Vec<u64> = (0..d).map(|_| rng.next_u64()).collect();
    let b: Vec<u64> = (0..d).map(|_| rng.next_u64()).collect();
    let x: u64 = rng.next_u64();

    // Non-trivial table so the per-row cells differ and the min is meaningful.
    let table: Vec<u32> = (0..d * w).map(|k| (k as u32 * 7 + 3) % 97).collect();

    let mut min_expected = u32::MAX;
    for row in 0..d {
        let col = (hash_mul_add_lo32(a[row], b[row], x) as usize) % w;
        min_expected = min_expected.min(table[row * w + col]);
    }

    let ptx = crate::ptx_kernels::cm_query_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "cm_query_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_table = DeviceBuffer::<u32>::from_host(&table).expect("d_table");
    let d_a = DeviceBuffer::<u64>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<u64>::from_host(&b).expect("d_b");
    let d_out = DeviceBuffer::<u32>::from_host(&[u32::MAX]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(d as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_table.as_device_ptr(),
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                w as u32,
                d as u32,
                x,
                d_out.as_device_ptr(),
            ),
        )
        .expect("launch cm_query_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = [0_u32];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert_eq!(
        out_gpu[0], min_expected,
        "cm_query: gpu min={} cpu min={}",
        out_gpu[0], min_expected
    );
    assert_ne!(
        out_gpu[0],
        u32::MAX,
        "cm_query: reduction never ran (out still u32::MAX) — kernel wrote nothing"
    );
}

// ===========================================================================
// 3. hll_register  —  INDEPENDENT HOST RE-DERIVATION (register max-update)
// ===========================================================================
//
// Single-thread kernel: `idx = lo32(hash) & (m-1)` (low `p_bits`),
// `rho = clz32(lo32(hash >> p_bits)) + 1`, then
// `atom.global.max.u32 reg[idx], rho`. The index bits `[0, p)` and the rho bits
// `[p, p+32)` are disjoint, so this is a self-consistent low-bit-indexed HLL.
// (The in-source comment claiming `idx = hash >> 32` disagrees with the code;
// the code is the source of truth and is what we re-derive.)

#[test]
fn hll_register_max_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let p_bits = 4_u32;
    let m = 1_u32 << p_bits; // 16 registers
    let hash: u64 = 0x0123_4567_89AB_CDEF;

    let idx = (hash as u32 & (m - 1)) as usize;
    let rho = ((hash >> p_bits) as u32).leading_zeros() + 1;

    // Pre-seed registers with small values so the `max` is exercised (and we can
    // confirm only `idx` changes and only upward).
    let regs_init: Vec<u32> = (0..m).map(|k| k % 3).collect();
    let mut regs_expected = regs_init.clone();
    regs_expected[idx] = regs_expected[idx].max(rho);

    let ptx = crate::ptx_kernels::hll_register_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "hll_register_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_regs = DeviceBuffer::<u32>::from_host(&regs_init).expect("d_regs");

    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(&params, &stream, &(d_regs.as_device_ptr(), m, p_bits, hash))
        .expect("launch hll_register_kernel");
    stream.synchronize().expect("sync");

    let mut regs_gpu = vec![0_u32; m as usize];
    d_regs.copy_to_host(&mut regs_gpu).expect("copy regs");

    for k in 0..m as usize {
        assert_eq!(
            regs_gpu[k], regs_expected[k],
            "hll: reg[{k}] gpu={} cpu={} (idx={idx} rho={rho})",
            regs_gpu[k], regs_expected[k]
        );
    }
    assert!(
        regs_gpu[idx] >= regs_init[idx],
        "hll: register max-update must be monotone non-decreasing"
    );
}

// ===========================================================================
// 4. bloom_insert  —  INDEPENDENT HOST RE-DERIVATION (set k bits)
// ===========================================================================
//
// For each of the `k` hash-threads: `bit = lo32(seed[i]*x) mod m`, then
// `atom.global.or.b32 bits[bit/32], 1 << (bit & 31)`. The host OR-accumulates
// the same `k` bits (collisions are absorbed by OR on both sides).

#[test]
fn bloom_insert_sets_expected_bits() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let m = 256_u32; // bit count (multiple of 32)
    let n_words = (m / 32) as usize;
    let k = 5_usize;
    let mut rng = LcgRng::new(0xB100_0FAA);
    let seeds: Vec<u64> = (0..k).map(|_| rng.next_u64()).collect();
    let x: u64 = rng.next_u64();

    let mut bits_expected = vec![0_u32; n_words];
    for &seed in &seeds {
        let bit = (seed.wrapping_mul(x) as u32) % m;
        let word = (bit >> 5) as usize;
        let off = bit & 31;
        bits_expected[word] |= 1_u32 << off;
    }

    let ptx = crate::ptx_kernels::bloom_insert_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bloom_insert_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_bits = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_words]).expect("d_bits");
    let d_seeds = DeviceBuffer::<u64>::from_host(&seeds).expect("d_seeds");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(k as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_bits.as_device_ptr(),
                m,
                k as u32,
                d_seeds.as_device_ptr(),
                x,
            ),
        )
        .expect("launch bloom_insert_kernel");
    stream.synchronize().expect("sync");

    let mut bits_gpu = vec![0_u32; n_words];
    d_bits.copy_to_host(&mut bits_gpu).expect("copy bits");

    for w in 0..n_words {
        assert_eq!(
            bits_gpu[w], bits_expected[w],
            "bloom: bits[{w}] gpu={:#034b} cpu={:#034b}",
            bits_gpu[w], bits_expected[w]
        );
    }
    let popcount: u32 = bits_gpu.iter().map(|w| w.count_ones()).sum();
    assert!(
        (1..=k as u32).contains(&popcount),
        "bloom: set-bit count {popcount} out of range 1..={k}"
    );
}

// ===========================================================================
// 5. minhash_sketch  —  INDEPENDENT HOST RE-DERIVATION (signature min-update)
// ===========================================================================
//
// For each of the `k` hash-threads: `h = lo32(a[i]*x + b[i])`, then
// `atom.global.min.u32 sig[i], h`. The signature is pre-seeded with a mix of
// large and small values so the `min` semantics are genuinely exercised.

#[test]
fn minhash_sketch_min_update_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k = 7_usize;
    let mut rng = LcgRng::new(0x31A5_4001);
    let a: Vec<u64> = (0..k).map(|_| rng.next_u64()).collect();
    let b: Vec<u64> = (0..k).map(|_| rng.next_u64()).collect();
    let x: u64 = rng.next_u64();

    // Mix of MAX (always replaced) and small (sometimes kept) initial slots.
    let sig_init: Vec<u32> = (0..k)
        .map(|i| if i % 2 == 0 { u32::MAX } else { 1_000 })
        .collect();

    let mut sig_expected = sig_init.clone();
    for i in 0..k {
        let h = hash_mul_add_lo32(a[i], b[i], x);
        sig_expected[i] = sig_expected[i].min(h);
    }

    let ptx = crate::ptx_kernels::minhash_sketch_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "minhash_sketch_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_sig = DeviceBuffer::<u32>::from_host(&sig_init).expect("d_sig");
    let d_a = DeviceBuffer::<u64>::from_host(&a).expect("d_a");
    let d_b = DeviceBuffer::<u64>::from_host(&b).expect("d_b");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(k as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_sig.as_device_ptr(),
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                k as u32,
                x,
            ),
        )
        .expect("launch minhash_sketch_kernel");
    stream.synchronize().expect("sync");

    let mut sig_gpu = vec![0_u32; k];
    d_sig.copy_to_host(&mut sig_gpu).expect("copy sig");

    for i in 0..k {
        assert_eq!(
            sig_gpu[i], sig_expected[i],
            "minhash: sig[{i}] gpu={} cpu={}",
            sig_gpu[i], sig_expected[i]
        );
    }
}

// ===========================================================================
// 6. tdigest_merge  —  INDEPENDENT HOST RE-DERIVATION (weighted centroid merge)
// ===========================================================================
//
// Single-thread f64 kernel merging centroid `i` into centroid `j`:
//   new_w = w_i + w_j
//   new_m = (m_i*w_i + m_j*w_j) / new_w   (GPU uses one fused `fma.rn.f64`)
// then writes `means[j]=new_m`, `weights[j]=new_w`, and zeros `means[i]` and
// `weights[i]`. f64 integer-clean inputs make the comparison essentially exact.

#[test]
fn tdigest_merge_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let means: Vec<f64> = vec![-2.0, 0.5, 3.0, 7.25, -1.5];
    let weights: Vec<f64> = vec![1.0, 4.0, 2.0, 3.0, 5.0];
    let i = 1_usize;
    let j = 3_usize;

    let new_w = weights[i] + weights[j];
    let new_m = (means[i] * weights[i] + means[j] * weights[j]) / new_w;

    let mut means_expected = means.clone();
    let mut weights_expected = weights.clone();
    means_expected[j] = new_m;
    weights_expected[j] = new_w;
    means_expected[i] = 0.0;
    weights_expected[i] = 0.0;

    let ptx = crate::ptx_kernels::tdigest_merge_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "tdigest_merge_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_means = DeviceBuffer::<f64>::from_host(&means).expect("d_means");
    let d_weights = DeviceBuffer::<f64>::from_host(&weights).expect("d_weights");

    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_means.as_device_ptr(),
                d_weights.as_device_ptr(),
                i as u32,
                j as u32,
            ),
        )
        .expect("launch tdigest_merge_kernel");
    stream.synchronize().expect("sync");

    let mut means_gpu = vec![0.0_f64; means.len()];
    let mut weights_gpu = vec![0.0_f64; weights.len()];
    d_means.copy_to_host(&mut means_gpu).expect("copy means");
    d_weights
        .copy_to_host(&mut weights_gpu)
        .expect("copy weights");

    for idx in 0..means.len() {
        assert!(
            close_f64(means_gpu[idx], means_expected[idx], 1e-12, 1e-12),
            "tdigest: means[{idx}] gpu={} cpu={}",
            means_gpu[idx],
            means_expected[idx]
        );
        assert!(
            close_f64(weights_gpu[idx], weights_expected[idx], 1e-12, 1e-12),
            "tdigest: weights[{idx}] gpu={} cpu={}",
            weights_gpu[idx],
            weights_expected[idx]
        );
    }
    // The merged centroid weight must be the exact integer sum (no rounding).
    assert_eq!(
        weights_gpu[j], new_w,
        "tdigest: merged weight must equal w_i + w_j exactly"
    );
}

// ===========================================================================
// 7. reservoir_sample  —  INDEPENDENT HOST RE-DERIVATION (Vitter step)
// ===========================================================================
//
// Single-thread kernel: `j = rand % i`; if `j < k` then `reservoir[j] = item`.
// We exercise BOTH branches: a `j < k` case that writes, and a `j >= k` case
// that leaves the reservoir byte-for-byte unchanged.

#[test]
fn reservoir_sample_replaces_when_in_range() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let k = 3_u32;
    let i = 10_u32;
    let item: u64 = 0xDEAD_BEEF_CAFE_F00D;

    // --- Case A: rand=2 => j = 2 < k => reservoir[2] := item ---
    let rand_hit = 2_u32;
    let j_hit = (rand_hit % i) as usize;
    assert!(j_hit < k as usize, "test setup: case A must hit");
    let res_init: Vec<u64> = vec![10, 20, 30];
    let mut res_expected = res_init.clone();
    res_expected[j_hit] = item;

    let ptx = crate::ptx_kernels::reservoir_sample_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "reservoir_sample_kernel");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_res = DeviceBuffer::<u64>::from_host(&res_init).expect("d_res");
    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_res.as_device_ptr(), k, i, rand_hit, item),
        )
        .expect("launch reservoir_sample_kernel (hit)");
    stream.synchronize().expect("sync");

    let mut res_gpu = vec![0_u64; k as usize];
    d_res.copy_to_host(&mut res_gpu).expect("copy res hit");
    for idx in 0..k as usize {
        assert_eq!(
            res_gpu[idx], res_expected[idx],
            "reservoir(hit): res[{idx}] gpu={} cpu={}",
            res_gpu[idx], res_expected[idx]
        );
    }

    // --- Case B: rand=7 => j = 7 >= k => reservoir unchanged ---
    let rand_miss = 7_u32;
    assert!(
        (rand_miss % i) as usize >= k as usize,
        "test setup: case B must miss"
    );
    let d_res2 = DeviceBuffer::<u64>::from_host(&res_init).expect("d_res2");
    kernel
        .launch(
            &params,
            &stream,
            &(d_res2.as_device_ptr(), k, i, rand_miss, item),
        )
        .expect("launch reservoir_sample_kernel (miss)");
    stream.synchronize().expect("sync");

    let mut res_gpu2 = vec![0_u64; k as usize];
    d_res2.copy_to_host(&mut res_gpu2).expect("copy res miss");
    for idx in 0..k as usize {
        assert_eq!(
            res_gpu2[idx], res_init[idx],
            "reservoir(miss): res[{idx}] mutated despite j >= k (gpu={})",
            res_gpu2[idx]
        );
    }
}
