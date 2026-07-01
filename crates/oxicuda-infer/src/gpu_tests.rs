//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies the results
//! back, and asserts numerical equivalence to the crate's CPU reference. The
//! launch ABI mirrors the working `oxicuda-snn` / `oxicuda-ot` canaries: device
//! buffers are passed as their `CUdeviceptr` (a `.param .u64`), scalars as the
//! matching Rust scalar (`.param .u32` / `.param .f32`), in declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to the `pub`
//!   CPU function the kernel mirrors:
//!   `paged_attention` ↔ [`crate::executor::paged_attention_cpu`] (base-e
//!   softmax attention), `logits_softmax` ↔ [`crate::sampling::softmax`]
//!   (base-e, max-stabilised).
//! * **Independent host re-derivation** — the kernel has no single dedicated
//!   crate function, so the oracle is an independent Rust re-implementation of
//!   the kernel's *documented* arithmetic: `rope_apply` (f64 cos/sin RoPE),
//!   `top_k_filter` (index-window mask), `kv_append` (paged-cache copy). These
//!   still genuinely fail if ptxas miscompiles or the PTX has a wrong constant /
//!   shift / index, because the host code is independent of the JIT PTX.
//!
//! ## PTX bugs found on-device and fixed (see `ptx_kernels.rs` for details)
//!
//! * `paged_attention` — invalid PTX (`mul.wide.u32` with a 64-bit source, never
//!   loaded); per-element score instead of full dot product; value read from the
//!   key buffer; base-2 softmax (missing `· log2(e)`); wrong GQA head map.
//! * `logits_softmax` — invalid PTX (`[smem + reg*4]` scaled address, never
//!   loaded); base-2 softmax; a shared-memory read-after-overwrite race.
//! * `rope_apply` — the `Q[2i]` output lost its `−Q[2i+1]·sin` term (two
//!   cancelling fmas); imprecise `log2(10000)` constant.
//!
//! Every test skips (returns early) when no CUDA device is present, so the suite
//! stays green on CPU-only machines.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

use crate::cache::kv_cache::{BlockId, PagedKvCache};
use crate::executor::paged_attention_cpu;
use crate::sampling::{Rng, softmax};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
///
/// `Context::new` calls `cuCtxCreate`, which both creates the context and makes
/// it current on the calling thread; the returned `Arc<Context>` must be kept
/// alive for the whole test (nextest runs each test in its own process).
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

/// Uniform `f32` in `[-1, 1)` from the crate's sampling RNG.
fn signed_unit(rng: &mut Rng) -> f32 {
    rng.next_f32() * 2.0 - 1.0
}

// ===========================================================================
// 1. paged_attention  —  CRATE ORACLE (executor::paged_attention_cpu), MHA + GQA
// ===========================================================================

/// Build a paged KV cache + a contiguous device-side K/V buffer that are
/// byte-for-byte consistent, then assert the GPU kernel matches
/// `paged_attention_cpu` (base-e softmax attention).
///
/// A non-identity `block_table = [phys 2, phys 0]` exercises the block-table
/// indirection (logical block 0 → physical block 2, logical 1 → physical 0).
fn run_attention_case(fx: &GpuFixture, n_heads: usize, n_kv_heads: usize) {
    let head_dim = 8_usize;
    let block_size = 3_usize;
    let n_logical = 2_usize;
    let seq_len = n_logical * block_size; // full blocks (= 6)
    let n_phys = 4_usize; // allocate extras so the block table is non-identity
    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    let mut rng = Rng::new(0x0A77 ^ ((n_heads as u64) << 8) ^ n_kv_heads as u64);

    let mut cache = PagedKvCache::new(1, n_kv_heads, head_dim, block_size, n_phys);
    // Allocate all physical blocks (ids 0..n_phys) so we can pick a permuted table.
    for _ in 0..n_phys {
        cache.alloc_block().expect("phys block");
    }
    let block_table = vec![BlockId(2), BlockId(0)];

    // Append `block_size` tokens to each logical block's physical target.
    let kv_per_tok = n_kv_heads * head_dim;
    for &phys in &block_table {
        for _ in 0..block_size {
            let k_tok: Vec<f32> = (0..kv_per_tok).map(|_| signed_unit(&mut rng)).collect();
            let v_tok: Vec<f32> = (0..kv_per_tok).map(|_| signed_unit(&mut rng)).collect();
            cache.append_token(phys, 0, &k_tok, &v_tok).expect("append");
        }
    }

    // Contiguous device buffers laid out as [n_phys, block_size, n_kv_heads, head_dim],
    // read straight out of the cache so they are perfectly consistent with the oracle.
    let block_elems = block_size * n_kv_heads * head_dim;
    let mut flat_k = vec![0.0_f32; n_phys * block_elems];
    let mut flat_v = vec![0.0_f32; n_phys * block_elems];
    for p in 0..n_phys {
        let blk = cache.block(BlockId(p as u32), 0).expect("block");
        flat_k[p * block_elems..(p + 1) * block_elems].copy_from_slice(&blk.keys);
        flat_v[p * block_elems..(p + 1) * block_elems].copy_from_slice(&blk.values);
    }

    let q: Vec<f32> = (0..n_heads * head_dim)
        .map(|_| signed_unit(&mut rng))
        .collect();

    // ---- CPU reference (crate oracle) ----
    let out_cpu = paged_attention_cpu(
        &q,
        &cache,
        &block_table,
        seq_len,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        block_size,
        scale,
    )
    .expect("paged_attention_cpu");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::paged_attn_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "paged_attention");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let btbl: Vec<u32> = block_table.iter().map(|b| b.0).collect();
    let d_q = DeviceBuffer::<f32>::from_host(&q).expect("d_q");
    let d_k = DeviceBuffer::<f32>::from_host(&flat_k).expect("d_k");
    let d_v = DeviceBuffer::<f32>::from_host(&flat_v).expect("d_v");
    let d_bt = DeviceBuffer::<u32>::from_host(&btbl).expect("d_bt");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_heads * head_dim]).expect("d_out");

    // Grid = (n_heads), block = (head_dim): block must equal head_dim exactly.
    let params = LaunchParams::new(n_heads as u32, head_dim as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_k.as_device_ptr(),
                d_v.as_device_ptr(),
                d_bt.as_device_ptr(),
                d_out.as_device_ptr(),
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
                block_size as u32,
                n_logical as u32,
                seq_len as u32,
                scale,
            ),
        )
        .expect("launch paged_attention");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_heads * head_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    // Observed on RTX A4000 (sm_86): worst rel ≈ 4e-6 (MHA), ≈ 2e-6 (GQA).
    // Tolerance: the GPU online softmax uses `ex2.approx.f32` (~2 ulp) plus a
    // `· log2(e)` scale, and rescales the accumulator each step, while the CPU
    // computes all `exp` (libm, <1 ulp) then a single normalised weighted sum.
    // Over seq_len = 6 well-conditioned tokens the divergence is a few ulp; the
    // bound below comfortably covers it yet still catches a base-2 softmax
    // (~20–50 %), a wrong dot product, or a V/K pointer swap by many orders.
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_cpu[k], 5e-4, 1e-5),
            "paged_attention out[{k}] (n_heads={n_heads}, n_kv_heads={n_kv_heads}) mismatch: \
             gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k]
        );
    }
}

#[test]
fn paged_attention_mha_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_attention_case(&fx, 4, 4); // multi-head attention
}

#[test]
fn paged_attention_gqa_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_attention_case(&fx, 4, 2); // grouped-query attention (gqa_ratio = 2)
}

// ===========================================================================
// 2. rope_apply  —  INDEPENDENT HOST RE-DERIVATION (f64 cos/sin RoPE)
// ===========================================================================

#[test]
fn rope_apply_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let seq_len = 4_usize;
    let n_heads = 2_usize;
    let head_dim = 8_usize; // even; head_dim/2 = 4 pairs per token-head
    let n = seq_len * n_heads * head_dim;

    let mut rng = Rng::new(0x60_9E);
    let q0: Vec<f32> = (0..n).map(|_| signed_unit(&mut rng)).collect();
    let k0: Vec<f32> = (0..n).map(|_| signed_unit(&mut rng)).collect();
    // Positions in [0, 3]: keeps θ = pos / 10000^freq ≤ pos < π, where
    // `cos.approx.f32` / `sin.approx.f32` are accurate.
    let positions: Vec<u32> = (0..seq_len as u32).collect();

    // Host re-derivation in f64 of the documented rotation.
    let mut q_host = q0.clone();
    let mut k_host = k0.clone();
    for (s, &pos_u) in positions.iter().enumerate() {
        let pos = pos_u as f64;
        for h in 0..n_heads {
            let base = (s * n_heads + h) * head_dim;
            for pair in 0..head_dim / 2 {
                let d0 = base + 2 * pair;
                let d1 = d0 + 1;
                let freq = (2 * pair) as f64 / head_dim as f64;
                let theta = pos / 10000.0_f64.powf(freq);
                let (sin, cos) = theta.sin_cos();
                let rot = |a: f64, b: f64| (a * cos - b * sin, b * cos + a * sin);
                let (q0n, q1n) = rot(q0[d0] as f64, q0[d1] as f64);
                let (k0n, k1n) = rot(k0[d0] as f64, k0[d1] as f64);
                q_host[d0] = q0n as f32;
                q_host[d1] = q1n as f32;
                k_host[d0] = k0n as f32;
                k_host[d1] = k1n as f32;
            }
        }
    }

    let ptx = crate::ptx_kernels::rope_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rope_apply");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_q = DeviceBuffer::<f32>::from_host(&q0).expect("d_q");
    let d_k = DeviceBuffer::<f32>::from_host(&k0).expect("d_k");
    let d_pos = DeviceBuffer::<u32>::from_host(&positions).expect("d_pos");

    // Grid = (seq_len * n_heads), block = (head_dim / 2).
    let params = LaunchParams::new((seq_len * n_heads) as u32, (head_dim / 2) as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_q.as_device_ptr(),
                d_k.as_device_ptr(),
                d_pos.as_device_ptr(),
                n_heads as u32,
                head_dim as u32,
                seq_len as u32,
            ),
        )
        .expect("launch rope_apply");
    stream.synchronize().expect("sync");

    let mut q_gpu = vec![0.0_f32; n];
    let mut k_gpu = vec![0.0_f32; n];
    d_q.copy_to_host(&mut q_gpu).expect("copy q");
    d_k.copy_to_host(&mut k_gpu).expect("copy k");

    let (relq, absq) = worst_diff(&q_gpu, &q_host);
    let (relk, absk) = worst_diff(&k_gpu, &k_host);
    // Observed on RTX A4000 (sm_86): worst rel ≈ 1.4e-6, abs ≈ 1.8e-7.
    // Tolerance: `cos.approx`/`sin.approx` (~2 ulp), `ex2.approx`/`rcp.approx`
    // for 10000^freq, and `div.approx` for the freq index each add a few ulp; the
    // absolute floor handles components that rotate near zero. The bound below is
    // ~70× the observed divergence yet still catches the (fixed) missing-sine
    // term, which produced O(0.1–1) errors on rotated components (≈10⁴× larger).
    for i in 0..n {
        assert!(
            close(q_gpu[i], q_host[i], 3e-4, 1e-4),
            "rope q[{i}] mismatch: gpu={} host={} (worst rel={relq:e} abs={absq:e})",
            q_gpu[i],
            q_host[i]
        );
        assert!(
            close(k_gpu[i], k_host[i], 3e-4, 1e-4),
            "rope k[{i}] mismatch: gpu={} host={} (worst rel={relk:e} abs={absk:e})",
            k_gpu[i],
            k_host[i]
        );
    }
}

// ===========================================================================
// 3. top_k_filter  —  INDEPENDENT HOST RE-DERIVATION (index-window mask)
// ===========================================================================

/// The kernel masks positions with `idx >= min(k, vocab)` to −∞ and leaves the
/// rest untouched (documented as a reference *index-window* mask, not value
/// top-k). Both branches are bit-exact, so we compare raw bit patterns.
fn run_top_k_case(fx: &GpuFixture, k: usize) {
    let batch = 2_usize;
    let vocab = 8_usize;
    let k_eff = k.min(vocab);

    let mut rng = Rng::new(0x70_9C ^ k as u64);
    let logits: Vec<f32> = (0..batch * vocab)
        .map(|_| signed_unit(&mut rng) * 4.0)
        .collect();

    let mut expected = logits.clone();
    for b in 0..batch {
        for i in 0..vocab {
            if i >= k_eff {
                expected[b * vocab + i] = f32::NEG_INFINITY;
            }
        }
    }

    let ptx = crate::ptx_kernels::top_k_filter_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "top_k_filter");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");

    // Grid = (batch), block = (vocab).
    let params = LaunchParams::new(batch as u32, vocab as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                batch as u32,
                vocab as u32,
                k as u32,
            ),
        )
        .expect("launch top_k_filter");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; batch * vocab];
    d_logits.copy_to_host(&mut out_gpu).expect("copy logits");

    for idx in 0..batch * vocab {
        assert_eq!(
            out_gpu[idx].to_bits(),
            expected[idx].to_bits(),
            "top_k_filter (k={k}) out[{idx}] mismatch: gpu={} expected={}",
            out_gpu[idx],
            expected[idx]
        );
    }
}

#[test]
fn top_k_filter_masks_window() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_top_k_case(&fx, 3); // mask positions 3..8
}

#[test]
fn top_k_filter_clamps_k_above_vocab() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_top_k_case(&fx, 20); // k > vocab → k_eff = vocab → nothing masked
}

// ===========================================================================
// 4. logits_softmax  —  CRATE ORACLE (sampling::softmax), base-e
// ===========================================================================

#[test]
fn logits_softmax_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let batch = 3_usize;
    let vocab = 64_usize; // two warps: exercises the (fixed) cross-warp race
    let mut rng = Rng::new(0x50_F7);
    let logits: Vec<f32> = (0..batch * vocab)
        .map(|_| signed_unit(&mut rng) * 4.0)
        .collect();

    // CPU reference: the crate's stable base-e softmax, applied per row.
    let mut expected = vec![0.0_f32; batch * vocab];
    for b in 0..batch {
        let probs = softmax(&logits[b * vocab..(b + 1) * vocab]);
        expected[b * vocab..(b + 1) * vocab].copy_from_slice(&probs);
    }

    let ptx = crate::ptx_kernels::logits_softmax_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "logits_softmax");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");

    // Grid = (batch), block = (vocab): block must equal vocab so all threads
    // reach every `bar.sync`.
    let params = LaunchParams::new(batch as u32, vocab as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(d_logits.as_device_ptr(), batch as u32, vocab as u32),
        )
        .expect("launch logits_softmax");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; batch * vocab];
    d_logits.copy_to_host(&mut out_gpu).expect("copy logits");

    // Probabilities are positive and sum to 1, so a relative test with a small
    // absolute floor is meaningful everywhere.
    let (rel, abs) = worst_diff(&out_gpu, &expected);
    // Observed on RTX A4000 (sm_86): worst rel ≈ 4.3e-7. A base-2 softmax (the
    // bug that was fixed) would diverge by ~20–50 % here.
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], expected[k], 5e-4, 1e-6),
            "logits_softmax out[{k}] mismatch: gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            expected[k]
        );
    }
    // Each GPU row must also sum to 1 (the normalisation actually happened).
    for b in 0..batch {
        let s: f32 = out_gpu[b * vocab..(b + 1) * vocab].iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "row {b} sums to {s}, not 1");
    }
}

// ===========================================================================
// 5. kv_append  —  INDEPENDENT HOST RE-DERIVATION (paged-cache copy)
// ===========================================================================

#[test]
fn kv_append_matches_host() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_kv_heads = 3_usize;
    let head_dim = 8_usize;
    let block_size = 4_usize;
    let n_blocks = 5_usize;
    let block_id = 2_usize;
    let slot = 1_usize;

    let mut rng = Rng::new(0x6B_A9);
    let k_new: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|_| signed_unit(&mut rng))
        .collect();
    let v_new: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|_| signed_unit(&mut rng))
        .collect();
    let cache_len = n_blocks * block_size * n_kv_heads * head_dim;

    // Host re-derivation: write k_new/v_new into the single target slot, zeros
    // everywhere else.
    let mut k_exp = vec![0.0_f32; cache_len];
    let mut v_exp = vec![0.0_f32; cache_len];
    for head in 0..n_kv_heads {
        for d in 0..head_dim {
            let off = (block_id * block_size + slot) * n_kv_heads * head_dim + head * head_dim + d;
            k_exp[off] = k_new[head * head_dim + d];
            v_exp[off] = v_new[head * head_dim + d];
        }
    }

    let ptx = crate::ptx_kernels::kv_append_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "kv_append");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_knew = DeviceBuffer::<f32>::from_host(&k_new).expect("d_knew");
    let d_vnew = DeviceBuffer::<f32>::from_host(&v_new).expect("d_vnew");
    let d_kc = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; cache_len]).expect("d_kc");
    let d_vc = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; cache_len]).expect("d_vc");

    // Grid = (n_kv_heads), block = (head_dim).
    let params = LaunchParams::new(n_kv_heads as u32, head_dim as u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_knew.as_device_ptr(),
                d_vnew.as_device_ptr(),
                d_kc.as_device_ptr(),
                d_vc.as_device_ptr(),
                block_id as u32,
                slot as u32,
                n_kv_heads as u32,
                head_dim as u32,
                block_size as u32,
            ),
        )
        .expect("launch kv_append");
    stream.synchronize().expect("sync");

    let mut k_gpu = vec![0.0_f32; cache_len];
    let mut v_gpu = vec![0.0_f32; cache_len];
    d_kc.copy_to_host(&mut k_gpu).expect("copy kc");
    d_vc.copy_to_host(&mut v_gpu).expect("copy vc");

    // Pure copy → bit-exact, including the all-zero untouched region.
    for idx in 0..cache_len {
        assert_eq!(
            k_gpu[idx].to_bits(),
            k_exp[idx].to_bits(),
            "kv_append k_cache[{idx}] mismatch: gpu={} host={}",
            k_gpu[idx],
            k_exp[idx]
        );
        assert_eq!(
            v_gpu[idx].to_bits(),
            v_exp[idx].to_bits(),
            "kv_append v_cache[{idx}] mismatch: gpu={} host={}",
            v_gpu[idx],
            v_exp[idx]
        );
    }
}
