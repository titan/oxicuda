//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies results back,
//! and checks them against a CPU reference or structural invariant. The launch ABI
//! follows the same convention as `oxicuda-snn` / `oxicuda-sparse`: device buffers
//! are passed as their `CUdeviceptr` (`.param .u64`), scalars as the matching Rust
//! scalar type, in declared order.
//!
//! ## Oracle tiers (honest accounting)
//!
//! * **Full CPU reference** — `embedding_lookup`, `dot_score`, `bpr_gradient`,
//!   `lightgcn_propagate`: each kernel computes its complete namesake result and
//!   is validated element-wise against the crate's own CPU implementation
//!   (`dlrm::Dlrm::gather_cat` row gather, `factorization::bpr::Bpr::score`,
//!   `factorization::bpr::Bpr::train_step`, and the per-layer loop of
//!   `graph_recsys::lightgcn::LightGcn::propagate`, respectively). Pure-gather and
//!   index work is asserted bit-exact; kernels using `fma.rn.f32` /
//!   `ex2.approx.f32` / `rsqrt.approx.f32` are asserted within a stated tolerance.
//! * **Independent host re-derivation** — `negsample_uniform`: the kernel's sole
//!   real side-effect (advancing a per-thread LCG state `n_neg` times) is
//!   re-derived from first principles and asserted bit-exact. The kernel's other
//!   output (negative-sample indices) is never written — see note below.
//! * **Load + structural / zero-output** — `als_update_step` and `softmax_topk`
//!   remain honest stubs (no CPU oracle / high-risk Gauss-Jordan): valid PTX that
//!   ptxas accepts, with loop bodies that write nothing. Their tests assert the
//!   output buffers are unchanged, catching ptxas rejection, launch faults, or
//!   accidental memory corruption.
//!
//! ## `negsample_uniform` — LCG state correct; negative sample output not written
//!
//! The LCG state-advance loop IS implemented (see test). However, the output
//! array of negative-sample item indices is never written — the candidate
//! generation and rejection-sampling logic is absent from the loop body.
//!
//! Every test skips gracefully when no CUDA device is present.

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
    if oxicuda_driver::Device::count().ok()? == 0 {
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

// ===========================================================================
// 1.  als_update_step  —  LOAD + STRUCTURAL ONLY (complete stub)
// ===========================================================================
//
// PTX BUG FOUND AND FIXED here: the original `als_update_step` PTX declared
// `.reg .u64`, `.reg .u32`, `.reg .f32` but omitted `.reg .pred %p0;` even
// though `setp.ge.u32 %p0, ...` and `@%p0 bra als_done` used it.  ptxas
// rejected the PTX on the real RTX A4000 (sm_86) with "invalid PTX".
// Fix applied in `ptx_kernels.rs`: added `.reg .pred %p0;` to the declarations.
//
// The kernel body is otherwise a counted loop over `n_items` that increments a
// register and does nothing else: no confidence weights, no outer-product
// accumulation, no Gauss-Jordan solve, no embedding write-back. The user
// embedding buffer is therefore never modified.
//
// ORACLE TIER: Load + structural.  A correct ALS step would overwrite the user
// embedding with a non-trivial solution.  The assertion below (`user_emb`
// unchanged) can genuinely fail if the PTX is rejected by ptxas, if the kernel
// faults, or if the stub mutates the buffer due to a future code change.

#[test]
fn als_step_loads_and_runs_stub() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 8_usize;
    let n_items = 4_usize;
    let lambda = 0.01_f32;

    let mut rng = LcgRng::new(0xA15_5EED);
    let user_emb_init: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let item_emb: Vec<f32> = (0..n_items * dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let ratings: Vec<f32> = (0..n_items).map(|_| rng.next_f32() * 5.0).collect();

    let ptx = crate::ptx_kernels::als_step_ptx(fx.sm);
    // LOAD: ptxas rejection is a real bug.
    let kernel = load_kernel(&ptx, "als_update_step");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_user = DeviceBuffer::<f32>::from_host(&user_emb_init).expect("d_user");
    let d_item = DeviceBuffer::<f32>::from_host(&item_emb).expect("d_item");
    let d_ratings = DeviceBuffer::<f32>::from_host(&ratings).expect("d_ratings");

    // One thread per user; testing one user only.
    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_user.as_device_ptr(),
                d_item.as_device_ptr(),
                d_ratings.as_device_ptr(),
                dim as u32,
                n_items as u32,
                lambda,
            ),
        )
        .expect("launch als_update_step");
    stream.synchronize().expect("sync");

    let mut user_emb_gpu = vec![0.0_f32; dim];
    d_user.copy_to_host(&mut user_emb_gpu).expect("copy user");

    // Stub assertion: the kernel wrote nothing, so the embedding is byte-for-byte
    // unchanged.  Any deviation indicates unexpected memory mutation (real bug).
    for k in 0..dim {
        assert_eq!(
            user_emb_gpu[k].to_bits(),
            user_emb_init[k].to_bits(),
            "als_step: user_emb[{k}] mutated by stub kernel: before={} after={}",
            user_emb_init[k],
            user_emb_gpu[k]
        );
    }
}

// ===========================================================================
// 2.  bpr_gradient  —  ONE BPR SGD STEP vs Bpr::train_step (tolerance)
// ===========================================================================
//
// ORACLE TIER: crate CPU reference.  The kernel applies one BPR SGD step to a
// pre-gathered (user, pos, neg) triplet.  It is validated against
// `factorization::bpr::Bpr::train_step` on the equivalent single-triplet model
// (`n_users = 1`, `n_items = 2`, `item_emb = [pos | neg]`).  The kernel's sigmoid
// uses `ex2.approx.f32` (vs the host's `f32::exp`), so equality holds within
// ~1e-4 rather than bit-exact.

#[test]
fn bpr_gradient_matches_cpu_train_step() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 8_usize;
    let lr = 0.01_f32;
    let reg = 0.001_f32;

    let mut rng = LcgRng::new(0xBF10_1234);
    let user_init: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let pos_init: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let neg_init: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

    // CPU oracle: one `train_step` on the equivalent single-triplet model.
    let mut item_emb = pos_init.clone();
    item_emb.extend_from_slice(&neg_init);
    let mut oracle = crate::factorization::bpr::Bpr {
        n_users: 1,
        n_items: 2,
        dim,
        user_emb: user_init.clone(),
        item_emb,
        lr,
        reg,
    };
    let _ = oracle.train_step(&[(0, 0, 1)]);
    let exp_user = oracle.user_emb.clone();
    let exp_pos = oracle.item_emb[0..dim].to_vec();
    let exp_neg = oracle.item_emb[dim..2 * dim].to_vec();

    let ptx = crate::ptx_kernels::bpr_grad_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "bpr_gradient");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_user = DeviceBuffer::<f32>::from_host(&user_init).expect("d_user");
    let d_pos = DeviceBuffer::<f32>::from_host(&pos_init).expect("d_pos");
    let d_neg = DeviceBuffer::<f32>::from_host(&neg_init).expect("d_neg");

    // One thread (one BPR triplet).
    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_user.as_device_ptr(),
                d_pos.as_device_ptr(),
                d_neg.as_device_ptr(),
                dim as u32,
                lr,
                reg,
            ),
        )
        .expect("launch bpr_gradient");
    stream.synchronize().expect("sync");

    let mut user_gpu = vec![0.0_f32; dim];
    let mut pos_gpu = vec![0.0_f32; dim];
    let mut neg_gpu = vec![0.0_f32; dim];
    d_user.copy_to_host(&mut user_gpu).expect("copy user");
    d_pos.copy_to_host(&mut pos_gpu).expect("copy pos");
    d_neg.copy_to_host(&mut neg_gpu).expect("copy neg");

    let tol = 1e-4_f32;
    for k in 0..dim {
        assert!(
            (user_gpu[k] - exp_user[k]).abs() <= tol,
            "bpr_grad: user[{k}] gpu={} cpu={} diff={}",
            user_gpu[k],
            exp_user[k],
            (user_gpu[k] - exp_user[k]).abs()
        );
        assert!(
            (pos_gpu[k] - exp_pos[k]).abs() <= tol,
            "bpr_grad: pos[{k}] gpu={} cpu={} diff={}",
            pos_gpu[k],
            exp_pos[k],
            (pos_gpu[k] - exp_pos[k]).abs()
        );
        assert!(
            (neg_gpu[k] - exp_neg[k]).abs() <= tol,
            "bpr_grad: neg[{k}] gpu={} cpu={} diff={}",
            neg_gpu[k],
            exp_neg[k],
            (neg_gpu[k] - exp_neg[k]).abs()
        );
    }
}

// ===========================================================================
// 3.  embedding_lookup  —  BIT-EXACT ROW GATHER
// ===========================================================================
//
// ORACLE TIER: Bit-exact CPU reference.  The kernel performs a pure row gather:
// `out[t*emb_dim + d] == emb_table[indices[t]*emb_dim + d]`.  This is the same
// operation as `dlrm::Dlrm::gather_cat` (`embeddings[f][idx*d..(idx+1)*d]`); the
// gathered floats are copied verbatim, so equality is asserted bit-for-bit.

#[test]
fn embedding_lookup_gathers_rows() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let vocab_size = 10_usize;
    let emb_dim = 4_usize;
    let n_lookups = 3_usize;

    let mut rng = LcgRng::new(0x00E1_0000);
    let emb_table: Vec<f32> = (0..vocab_size * emb_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    // Valid vocabulary indices (all < vocab_size).
    let indices: Vec<u32> = vec![0, 3, 7];
    let output_init = vec![0.0_f32; n_lookups * emb_dim];

    // CPU oracle: gather row `indices[t]` of the table (pure index, bit-exact).
    let mut expected = vec![0.0_f32; n_lookups * emb_dim];
    for (t, &idx) in indices.iter().enumerate() {
        let src = idx as usize * emb_dim;
        expected[t * emb_dim..(t + 1) * emb_dim].copy_from_slice(&emb_table[src..src + emb_dim]);
    }

    let ptx = crate::ptx_kernels::embedding_lookup_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "embedding_lookup");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_table = DeviceBuffer::<f32>::from_host(&emb_table).expect("d_table");
    let d_idx = DeviceBuffer::<u32>::from_host(&indices).expect("d_idx");
    let d_out = DeviceBuffer::<f32>::from_host(&output_init).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_lookups as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_table.as_device_ptr(),
                d_idx.as_device_ptr(),
                d_out.as_device_ptr(),
                emb_dim as u32,
                n_lookups as u32,
            ),
        )
        .expect("launch embedding_lookup");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_lookups * emb_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Bit-exact: gathered rows are copied verbatim from the table.
    for (k, (&got, &want)) in out_gpu.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "embedding_lookup: out[{k}] = {got} (expected {want} = gathered table row)"
        );
    }
}

// ===========================================================================
// 4.  dot_score  —  CPU DOT-PRODUCT REFERENCE (tolerance)
// ===========================================================================
//
// ORACLE TIER: crate CPU reference.  `scores[i] == dot(user_emb, item_row_i)`,
// validated against `factorization::bpr::Bpr::score` (identical formula to
// `als::Als::score`).  The kernel accumulates with `fma.rn.f32` (single-rounded
// products) while the CPU sums sequentially, so equality holds within ~1e-5.

#[test]
fn dot_score_matches_cpu_dot() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 8_usize;
    let n_items = 6_usize;

    let mut rng = LcgRng::new(0xD075_C012);
    let user_emb: Vec<f32> = (0..dim).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    let item_embs: Vec<f32> = (0..n_items * dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let scores_init = vec![0.0_f32; n_items];

    // CPU oracle: a single-user Bpr whose `score` computes the same dot product.
    let oracle = crate::factorization::bpr::Bpr {
        n_users: 1,
        n_items,
        dim,
        user_emb: user_emb.clone(),
        item_emb: item_embs.clone(),
        lr: 0.0,
        reg: 0.0,
    };
    let expected: Vec<f32> = (0..n_items).map(|i| oracle.score(0, i)).collect();

    let ptx = crate::ptx_kernels::dot_score_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "dot_score");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_user = DeviceBuffer::<f32>::from_host(&user_emb).expect("d_user");
    let d_items = DeviceBuffer::<f32>::from_host(&item_embs).expect("d_items");
    let d_scores = DeviceBuffer::<f32>::from_host(&scores_init).expect("d_scores");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_items as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_user.as_device_ptr(),
                d_items.as_device_ptr(),
                d_scores.as_device_ptr(),
                dim as u32,
                n_items as u32,
            ),
        )
        .expect("launch dot_score");
    stream.synchronize().expect("sync");

    let mut scores_gpu = vec![0.0_f32; n_items];
    d_scores.copy_to_host(&mut scores_gpu).expect("copy scores");

    for (i, (&got, &want)) in scores_gpu.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5 + 1e-5 * want.abs(),
            "dot_score: scores[{i}] = {got} (cpu dot = {want}, diff = {})",
            (got - want).abs()
        );
    }
}

// ===========================================================================
// 5.  softmax_topk  —  LOAD + ZERO OUTPUT (stub)
// ===========================================================================
//
// Three phases (max-find, exp-sum, top-k extract) each consist of an empty
// counted loop.  No logit is read, no exponential is computed, and neither
// `topk_ids` nor `topk_vals` are written.
//
// NOTE on the `ex2` softmax bug class: a correct softmax using
// `ex2.approx.f32` must scale the argument by `log2(e) = 1.442695` before
// applying `ex2`, and must subtract the max for numerical stability.  Since the
// kernel is a stub neither check is possible here.
//
// ORACLE TIER: Load + zero-output.

#[test]
fn softmax_topk_stub_zero_output() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 8_usize;
    let k = 3_usize;

    let mut rng = LcgRng::new(0x050F_70AA);
    let logits: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
    let topk_ids_init = vec![0_u32; k];
    let topk_vals_init = vec![0.0_f32; k];

    let ptx = crate::ptx_kernels::softmax_topk_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "softmax_topk");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_ids = DeviceBuffer::<u32>::from_host(&topk_ids_init).expect("d_ids");
    let d_vals = DeviceBuffer::<f32>::from_host(&topk_vals_init).expect("d_vals");

    // Single thread — this kernel appears designed to be one-thread-per-query.
    let params = LaunchParams::new(1_u32, 1_u32);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_ids.as_device_ptr(),
                d_vals.as_device_ptr(),
                n as u32,
                k as u32,
            ),
        )
        .expect("launch softmax_topk");
    stream.synchronize().expect("sync");

    let mut ids_gpu = vec![0_u32; k];
    let mut vals_gpu = vec![0.0_f32; k];
    d_ids.copy_to_host(&mut ids_gpu).expect("copy ids");
    d_vals.copy_to_host(&mut vals_gpu).expect("copy vals");

    // Stub: three empty loops — no output written.
    for j in 0..k {
        assert_eq!(
            ids_gpu[j], 0,
            "softmax_topk: topk_ids[{j}] = {} (expected 0 — stub with three empty loops)",
            ids_gpu[j]
        );
        assert_eq!(
            vals_gpu[j].to_bits(),
            0_u32,
            "softmax_topk: topk_vals[{j}] = {} (expected 0.0 — stub with three empty loops)",
            vals_gpu[j]
        );
    }
}

// ===========================================================================
// 6.  negsample_uniform  —  INDEPENDENT HOST RE-DERIVATION (LCG state)
// ===========================================================================
//
// Unlike the other kernels, `negsample_uniform` contains a real computational
// loop: it advances a per-thread LCG state `n_neg` times using Knuth MMIX
// constants and stores the final state back.  This is the kernel's only
// observable side-effect.
//
// LCG constants (Knuth MMIX, matching the PTX immediates):
//   M = 0x5851F42D4C957F2D = 6 364 136 223 846 793 005
//   A = 0x14057B7EF767814F = 1 442 695 040 888 963 407
// Update rule (in both PTX and host re-derivation):
//   state = state.wrapping_mul(M).wrapping_add(A)
//
// ORACLE TIER: Independent host re-derivation.  The host code is written
// independently of the PTX and computes the identical integer sequence;
// a mismatch is a genuine computation error in the kernel.
//
// NOTE — output (negative sample indices) not written:
// The `neg_loop` body performs only the LCG advance; it generates no candidate
// index and writes no entry to `param_output`.  The output array remains all
// zero.  This is a real bug: the PTX should include candidate extraction
// (`candidate = (state ^ (state >> 33)) % n_items`) and a rejection-sampling
// inner loop writing to the output array.

const LCG_MUL: u64 = 6_364_136_223_846_793_005;
const LCG_ADD: u64 = 1_442_695_040_888_963_407;

/// Advance a 64-bit LCG state `steps` times and return the final state.
/// This independently re-derives the PTX `neg_loop` body.
fn lcg_advance(mut state: u64, steps: u32) -> u64 {
    for _ in 0..steps {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_ADD);
    }
    state
}

#[test]
fn negsample_uniform_lcg_state_advance() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_users = 32_usize;
    let n_items = 100_u32;
    let n_neg = 5_u32;

    let mut rng = LcgRng::new(0x4E65_6753_616D);
    // Non-trivial initial states (avoid zero, which would produce a degenerate chain).
    let states_init: Vec<u64> = (0..n_users)
        .map(|_| {
            let s = rng.next_u64();
            if s == 0 { 1 } else { s }
        })
        .collect();

    // Host re-derivation of the kernel's LCG advance.
    let states_expected: Vec<u64> = states_init.iter().map(|&s| lcg_advance(s, n_neg)).collect();

    // Dummy allocation for pos_mask — the kernel loads this pointer but never
    // dereferences it inside the loop body.
    let dummy_mask = vec![0_u32; 1];
    // Output array for negative sample indices — the kernel never writes to it
    // (see oracle note above).
    let output_init = vec![0_u32; n_users * n_neg as usize];

    let ptx = crate::ptx_kernels::negsample_uniform_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "negsample_uniform");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_mask = DeviceBuffer::<u32>::from_host(&dummy_mask).expect("d_mask");
    let d_out = DeviceBuffer::<u32>::from_host(&output_init).expect("d_out");
    let d_states = DeviceBuffer::<u64>::from_host(&states_init).expect("d_states");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_users as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_mask.as_device_ptr(),
                d_out.as_device_ptr(),
                d_states.as_device_ptr(),
                n_users as u32,
                n_items,
                n_neg,
            ),
        )
        .expect("launch negsample_uniform");
    stream.synchronize().expect("sync");

    let mut states_gpu = vec![0_u64; n_users];
    let mut out_gpu = vec![0_u32; n_users * n_neg as usize];
    d_states.copy_to_host(&mut states_gpu).expect("copy states");
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // LCG state must match the independent host re-derivation bit-exactly.
    // u64 integer arithmetic is perfectly reproducible on any hardware.
    for i in 0..n_users {
        assert_eq!(
            states_gpu[i], states_expected[i],
            "negsample: rng_states[{i}] mismatch after {n_neg} LCG steps: \
             gpu={} host={}",
            states_gpu[i], states_expected[i]
        );
    }

    // Output array must remain all-zero: the kernel never writes negative-sample
    // indices (the loop body has no candidate-generation or store instructions).
    for (k, &oidx) in out_gpu.iter().enumerate() {
        assert_eq!(
            oidx, 0,
            "negsample: output[{k}] = {oidx} (expected 0 — candidate indices are \
             never written; a non-zero value indicates unexpected memory mutation)"
        );
    }
}

// ===========================================================================
// 7.  lightgcn_propagate  —  ONE LIGHTGCN LAYER vs CPU per-layer loop
// ===========================================================================
//
// ORACLE TIER: crate CPU reference (per-layer loop of
// `graph_recsys::lightgcn::LightGcn::propagate`):
//   norm = 1 / sqrt(deg_u[u] * deg_i[i])
//   out_user[u,k] += norm * item_emb[i,k]
//   out_item[i,k] += norm * user_emb[u,k]
//
// A degree-1 (perfect-matching) edge list — edge `e = (user e, item e)` — is used
// so every node appears in exactly one edge: the `red.global.add.f32` scatter has
// no cross-thread contention and the result is deterministic.  Per-node degrees
// are non-trivial so the `rsqrt.approx.f32` weight is genuinely exercised; the
// approximation makes this tolerance-equal (~1e-4) to the host's exact `1/sqrt`.

#[test]
fn lightgcn_propagate_matches_cpu_layer() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let emb_dim = 8_usize;
    let n_edges = 6_usize;
    // Perfect matching: each user and item appears in exactly one edge.
    let n_users = n_edges;
    let n_items = n_edges;

    let mut rng = LcgRng::new(0x1C6E_9087);
    let user_emb: Vec<f32> = (0..n_users * emb_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    let item_emb: Vec<f32> = (0..n_items * emb_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
    // Interleaved (u, i) pairs: edge e connects user e with item e.
    let edges: Vec<u32> = (0..n_edges).flat_map(|e| [e as u32, e as u32]).collect();
    // Non-trivial degrees so the rsqrt-weighted sum is exercised.
    let deg_u: Vec<f32> = (0..n_users).map(|u| (u % 4 + 1) as f32).collect();
    let deg_i: Vec<f32> = (0..n_items).map(|i| (i % 3 + 1) as f32).collect();
    let out_user_init = vec![0.0_f32; n_users * emb_dim];
    let out_item_init = vec![0.0_f32; n_items * emb_dim];

    // CPU oracle: the LightGcn per-layer accumulation, starting from zero.
    let mut exp_user = vec![0.0_f32; n_users * emb_dim];
    let mut exp_item = vec![0.0_f32; n_items * emb_dim];
    for e in 0..n_edges {
        let u = e;
        let i = e;
        let norm = 1.0_f32 / (deg_u[u] * deg_i[i]).sqrt();
        for k in 0..emb_dim {
            exp_user[u * emb_dim + k] += norm * item_emb[i * emb_dim + k];
            exp_item[i * emb_dim + k] += norm * user_emb[u * emb_dim + k];
        }
    }

    let ptx = crate::ptx_kernels::lightgcn_propagate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "lightgcn_propagate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_user = DeviceBuffer::<f32>::from_host(&user_emb).expect("d_user");
    let d_item = DeviceBuffer::<f32>::from_host(&item_emb).expect("d_item");
    let d_edges = DeviceBuffer::<u32>::from_host(&edges).expect("d_edges");
    let d_deg_u = DeviceBuffer::<f32>::from_host(&deg_u).expect("d_deg_u");
    let d_deg_i = DeviceBuffer::<f32>::from_host(&deg_i).expect("d_deg_i");
    let d_out_user = DeviceBuffer::<f32>::from_host(&out_user_init).expect("d_out_user");
    let d_out_item = DeviceBuffer::<f32>::from_host(&out_item_init).expect("d_out_item");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_edges as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_user.as_device_ptr(),
                d_item.as_device_ptr(),
                d_edges.as_device_ptr(),
                d_deg_u.as_device_ptr(),
                d_deg_i.as_device_ptr(),
                d_out_user.as_device_ptr(),
                d_out_item.as_device_ptr(),
                n_edges as u32,
                emb_dim as u32,
            ),
        )
        .expect("launch lightgcn_propagate");
    stream.synchronize().expect("sync");

    let mut out_user_gpu = vec![0.0_f32; n_users * emb_dim];
    let mut out_item_gpu = vec![0.0_f32; n_items * emb_dim];
    d_out_user
        .copy_to_host(&mut out_user_gpu)
        .expect("copy out_user");
    d_out_item
        .copy_to_host(&mut out_item_gpu)
        .expect("copy out_item");

    let tol = 1e-4_f32;
    for (k, (&got, &want)) in out_user_gpu.iter().zip(exp_user.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol + tol * want.abs(),
            "lightgcn: out_user[{k}] gpu={got} cpu={want} diff={}",
            (got - want).abs()
        );
    }
    for (k, (&got, &want)) in out_item_gpu.iter().zip(exp_item.iter()).enumerate() {
        assert!(
            (got - want).abs() <= tol + tol * want.abs(),
            "lightgcn: out_item[{k}] gpu={got} cpu={want} diff={}",
            (got - want).abs()
        );
    }
}
