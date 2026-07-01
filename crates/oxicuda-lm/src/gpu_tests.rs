//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version,
//! launches it on the real CUDA device through `oxicuda-launch`, copies results
//! back, and asserts numerical equivalence to the crate's CPU reference. The
//! launch ABI mirrors the working `oxicuda-snn` canary: device buffers are
//! passed as their `CUdeviceptr` (`.param .u64`), scalars as the matching Rust
//! scalar type, in declared order.
//!
//! ## Oracle strength tiers (honest accounting)
//!
//! * **Crate oracle** (strongest):
//!   - `embedding_forward_matches_cpu` ↔ [`crate::layer::TokenEmbedding::forward`]
//!     (bit-exact; pure table lookup, no FP arithmetic).
//!   - `rope_apply_matches_cpu` ↔ [`crate::layer::RotaryEmbedding::apply`]
//!     (within 1e-5 relative; FP32 multiply-subtract rotation).
//!   - `silu_gate_matches_cpu` ↔ [`crate::layer::ffn::silu`]
//!     (within 5e-4 relative; `ex2.approx` used for sigmoid with log2(e) scaling).
//!   - `rms_norm_matches_cpu` ↔ [`crate::layer::RmsNorm::forward`]
//!     (within 1e-4 relative; `sqrt.approx` + shared-memory butterfly reduction).
//! * **Independent host re-derivation**:
//!   - `causal_attn_softmax_matches_cpu`: numerically stable base-e causal
//!     softmax. The `ex2.approx` kernel scales by `log2(e)` (correctly), so the
//!     base-e oracle detects any missing scale factor (which would cause ~20–50%
//!     error on individual probabilities even though they still sum to 1).
//!
//! ## PTX bugs found and fixed
//!
//! ### `causal_attn_softmax` — two SIMT race conditions (fixed in `ptx_kernels.rs`)
//!
//! **Bug 1 — MAX broadcast race**: All threads reconverge at `$MAX_DONE` and
//! write `%f0` to `smem[0]`. Thread 0 has the correct total-max; threads 1..N
//! have their per-thread partial-max. Because the writes are unguarded, the last
//! writer wins and the broadcast max is indeterminate (any thread's partial-max).
//! On a SIMT warp the highest-numbered active thread typically wins, corrupting
//! `smem[0]` with a value smaller than the true maximum. This makes some
//! exponentials overflow (`exp(score − wrong_max) = exp(large positive) → inf`),
//! producing NaN outputs.
//!
//! **Bug 2 — SUM broadcast race**: Identical structure at `$SUM_DONE`. Without a
//! guard, a thread's partial exp-sum overwrites the total exp-sum that thread 0
//! computed, producing a wrong normaliser and hence wrong softmax probabilities.
//!
//! **Fix**: `setp.eq.u32 %p1, %r5, 0` before each divergence, then
//! `@%p1 st.shared.f32 [smem], %fN` at the merge label. Only thread 0 (the one
//! that ran the sequential accumulation loop) writes the correct total
//! max/sum to `smem[0]`; the broadcast `bar.sync 0` then makes it visible to all.
//!
//! ## Notes
//!
//! * `rms_norm` requires `block = 256` regardless of `dim`. When `block < 256`
//!   the butterfly reduction reads uninitialised `smem` slots (threads above `dim`
//!   must write 0.0 to their slots for the reduction to be correct; this only
//!   happens when the full 256-thread block is present). The kernel's documentation
//!   says `block = min(dim, 256)` but this is INCORRECT for `dim < 256`.
//!   All tests launch with `block = 256`.
//! * Every test returns early (pass) when no CUDA device is present.

use std::sync::Arc;

use oxicuda_driver::{Context, Device, Module, Stream};
use oxicuda_launch::{Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// A live CUDA context plus the device's SM version, or `None` if unavailable.
///
/// `Context::new` calls `cuCtxCreate`, making the context current on the
/// calling thread. The `Arc` must be kept alive for the test duration (nextest
/// runs each test in its own process, so a per-test context is safe).
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

/// Relative-with-absolute-floor FP32 closeness test.
fn close(a: f32, b: f32, rel: f32, abs: f32) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
}

/// Worst (relative, absolute) divergence over two equal-length f32 slices.
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

/// JIT-compile `ptx` and look up `entry`; panics with a descriptive message on
/// any error so failures are immediately actionable.
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

// ---------------------------------------------------------------------------
// Minimal LCG random number generator (no `rand` dependency).
//
// Uses Knuth MMIX constants. `next_f32` returns a value in [0, 1) by
// dividing the upper 32 bits of the 64-bit state by 2^32 (never 2^31).
// ---------------------------------------------------------------------------

struct LcgRng {
    state: u64,
}

impl LcgRng {
    const MUL: u64 = 6_364_136_223_846_793_005;
    const ADD: u64 = 1_442_695_040_888_963_407;

    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0xDEAD_BEEF_CAFE_1234,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(Self::MUL).wrapping_add(Self::ADD);
        self.state
    }

    /// Uniform f32 in [0, 1), using upper 32 bits ÷ 2^32.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 32) as u32 as f32 / 4_294_967_296.0_f32
    }
}

// ===========================================================================
// 1. embedding_forward  —  CRATE ORACLE (TokenEmbedding::forward, bit-exact)
// ===========================================================================
//
// The kernel is a pure table-lookup: for each (token, dim) pair it reads
// `embed_table[token_id * embed_dim + dim_idx]` and writes it to the output.
// No floating-point arithmetic is involved, so the comparison is bit-exact.
//
// PTX analysis: the kernel correctly computes
//   tok_idx = tid / embed_dim,  dim_idx = tid % embed_dim,
//   token_id = p_token_ids[tok_idx],
//   out[tid] = p_embed[token_id * embed_dim + dim_idx].
// This matches `TokenEmbedding::forward` exactly.

#[test]
fn embedding_forward_matches_cpu() {
    use crate::layer::TokenEmbedding;
    use crate::weights::WeightTensor;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let vocab_size = 16_usize;
    let embed_dim = 8_usize;
    let n_tokens = 6_usize;

    let mut rng = LcgRng::new(0x000E_1BED_0001_u64);

    // Non-trivial random embedding table; if any value were zero the test
    // would still pass but we want to verify real data movement.
    let table_data: Vec<f32> = (0..vocab_size * embed_dim)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();

    // Valid token ids (all < vocab_size).
    let mut token_ids: Vec<u32> = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        token_ids.push((rng.next_u64() % vocab_size as u64) as u32);
    }

    // ---- CPU reference ----
    let mut emb = TokenEmbedding::new(vocab_size, embed_dim).expect("TokenEmbedding::new");
    emb.weight = WeightTensor::from_data(table_data.clone(), vec![vocab_size, embed_dim])
        .expect("WeightTensor::from_data");
    let out_cpu = emb.forward(&token_ids).expect("TokenEmbedding::forward");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::embedding_forward_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "embedding_forward");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ids = DeviceBuffer::<u32>::from_host(&token_ids).expect("d_ids");
    let d_embed = DeviceBuffer::<f32>::from_host(&table_data).expect("d_embed");
    let d_out =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens * embed_dim]).expect("d_out");

    let total = (n_tokens * embed_dim) as u32;
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ids.as_device_ptr(),
                d_embed.as_device_ptr(),
                d_out.as_device_ptr(),
                embed_dim as u32,
                n_tokens as u32,
            ),
        )
        .expect("launch embedding_forward");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_tokens * embed_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Pure table lookup → bit-exact comparison.
    for k in 0..out_gpu.len() {
        assert_eq!(
            out_gpu[k].to_bits(),
            out_cpu[k].to_bits(),
            "embedding_forward: out[{k}] gpu={} cpu={} (token_ids={:?})",
            out_gpu[k],
            out_cpu[k],
            token_ids
        );
    }
}

// ===========================================================================
// 2. rope_apply  —  CRATE ORACLE (RotaryEmbedding::apply, 1e-5 rel)
// ===========================================================================
//
// The kernel applies the rotation
//   x[2i]   = x[2i]·cos − x[2i+1]·sin
//   x[2i+1] = x[2i]·sin + x[2i+1]·cos
// in-place, with (cos, sin) from pre-computed tables.  We supply the tables
// extracted from the crate's `RotaryEmbedding` (via `cos_at` / `sin_at`) so
// the GPU and CPU see identical floating-point inputs.  The only source of
// divergence is the GPU's FP32 multiply-subtract vs the CPU's multiply-add.

#[test]
fn rope_apply_matches_cpu() {
    use crate::layer::RotaryEmbedding;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let head_dim = 8_usize;
    let n_heads = 2_usize;
    let n_tokens = 3_usize;
    let pos_offset = 2_usize;
    let max_positions = 16_usize;
    let theta = 10_000.0_f32;
    let half_dim = head_dim / 2;

    let rope = RotaryEmbedding::new(head_dim, max_positions, theta).expect("RotaryEmbedding::new");

    let mut rng = LcgRng::new(0xBEEF_A15E_0001_u64);
    let x_init: Vec<f32> = (0..n_tokens * n_heads * head_dim)
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    // CPU reference: apply in-place on a clone.
    let mut x_cpu = x_init.clone();
    rope.apply(&mut x_cpu, n_heads, n_tokens, pos_offset)
        .expect("RotaryEmbedding::apply");

    // Extract cos/sin tables into flat host buffers for the GPU.
    let table_len = max_positions * half_dim;
    let mut cos_table = vec![0.0_f32; table_len];
    let mut sin_table = vec![0.0_f32; table_len];
    for pos in 0..max_positions {
        for i in 0..half_dim {
            cos_table[pos * half_dim + i] = rope.cos_at(pos, i);
            sin_table[pos * half_dim + i] = rope.sin_at(pos, i);
        }
    }

    // ---- GPU (modifies x in-place) ----
    let ptx = crate::ptx_kernels::rope_apply_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rope_apply");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x_init).expect("d_x");
    let d_cos = DeviceBuffer::<f32>::from_host(&cos_table).expect("d_cos");
    let d_sin = DeviceBuffer::<f32>::from_host(&sin_table).expect("d_sin");

    // Launch: each thread handles one (head, token, pair) combo.
    // total threads = n_tokens * n_heads * half_dim
    let total = (n_tokens * n_heads * half_dim) as u32;
    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(total, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_cos.as_device_ptr(),
                d_sin.as_device_ptr(),
                n_heads as u32,
                head_dim as u32,
                n_tokens as u32,
                pos_offset as u32,
            ),
        )
        .expect("launch rope_apply");
    stream.synchronize().expect("sync");

    let mut x_gpu = x_init.clone();
    d_x.copy_to_host(&mut x_gpu).expect("copy x");

    // FP32 multiply-subtract: one rounding per operation.
    // The CPU uses two roundings (mul + sub separately). Over a head_dim/2 = 4
    // dimensional rotation the divergence is a few ulp, well inside 1e-5 rel.
    let (rel, abs) = worst_diff(&x_gpu, &x_cpu);
    for k in 0..x_gpu.len() {
        assert!(
            close(x_gpu[k], x_cpu[k], 1e-5, 1e-6),
            "rope_apply out[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            x_gpu[k],
            x_cpu[k]
        );
    }
}

// ===========================================================================
// 3. silu_gate  —  CRATE ORACLE (layer::ffn::silu, 5e-4 rel)
// ===========================================================================
//
// The kernel computes `out[i] = silu(gate[i]) * up[i]` where
// `silu(x) = x · sigmoid(x)` and sigmoid uses the base-2 approximation
//   `sigmoid(x) = 1 / (1 + ex2(-x · log2(e)))`
//
// PTX analysis (correct):
//   f2 = f0 * log2e  (0F3FB8AA3B = 1.44269504...)
//   f2 = -f2          → -gate · log2e
//   f3 = ex2(f2)       → exp(-gate)
//   f3 = 1.0 + f3      → 1 + exp(-gate)
//   f3 = 1/f3          → sigmoid(gate)
//   f2 = f0 * f3       → silu(gate)
//   out = f2 * f1      → silu(gate) * up
//
// The log2(e) scale factor (0x3FB8AA3B) IS present and correct. No base-2
// error. Tolerance 5e-4 covers `ex2.approx.f32` (~2 ulp), comfortably above
// the ~1e-6 CPU libm error while catching any gross formula mistake.

#[test]
fn silu_gate_matches_cpu() {
    use crate::layer::ffn::silu;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n = 256_usize;

    let mut rng = LcgRng::new(0x5110_61AE);

    // Gate in [-3, 3]: sigmoid is well-conditioned in this range and ex2.approx
    // is accurate. Up in [-2, 2].
    let gate: Vec<f32> = (0..n).map(|_| rng.next_f32() * 6.0 - 3.0).collect();
    let up: Vec<f32> = (0..n).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // ---- CPU reference ----
    let out_cpu: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect();

    // ---- GPU ----
    let ptx = crate::ptx_kernels::silu_gate_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "silu_gate");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_gate = DeviceBuffer::<f32>::from_host(&gate).expect("d_gate");
    let d_up = DeviceBuffer::<f32>::from_host(&up).expect("d_up");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_gate.as_device_ptr(),
                d_up.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("launch silu_gate");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Tolerance: `rcp.approx.f32` + `ex2.approx.f32` each contribute ~2 ulp.
    // The product `silu(gate) * up` adds one more rounding. 5e-4 relative is
    // generous for approx instructions yet still catches a missing log2(e) scale
    // factor (which would give silu_2(gate) = gate/(1+2^(-gate)) ≠ silu(gate)
    // by ~5-30% depending on gate magnitude).
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..n {
        assert!(
            close(out_gpu[k], out_cpu[k], 5e-4, 1e-6),
            "silu_gate out[{k}] mismatch: gpu={} cpu={} gate={} up={} \
             (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k],
            gate[k],
            up[k]
        );
    }
}

// ===========================================================================
// 4. rms_norm  —  CRATE ORACLE (RmsNorm::forward, 1e-4 rel)
// ===========================================================================
//
// The kernel normalises each token row using a shared-memory butterfly
// reduction to accumulate `Σ x²`. IMPORTANT: the butterfly assumes a
// 256-thread block. If launched with `block < 256`, threads above `dim` are
// absent and their `smem` slots are never initialised to 0, causing the
// reduction to accumulate garbage.
//
// **Always launch with block = 256** regardless of `dim`.  The LOOP body
// checks `tid >= dim` and skips for out-of-range threads, storing 0.0 into
// their `smem` slot — making the reduction correct for any `dim ≤ 256`.
//
// Observed potential (non-correctness) issue: `%p1` is set to `(tid != 0)`
// but NEVER used to guard the inv_rms store. In practice all 256 threads
// compute the SAME inv_rms (from the same smem[0] after the butterfly), so
// the unguarded multi-write is harmless (same value, indeterminate writer,
// but identical outcome).

#[test]
fn rms_norm_matches_cpu() {
    use crate::layer::RmsNorm;

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let dim = 64_usize;
    let n_tokens = 2_usize;
    let eps = 1e-5_f32;

    let mut rng = LcgRng::new(0x1234_5678_9ABC_DEF0_u64);

    let input: Vec<f32> = (0..n_tokens * dim)
        .map(|_| rng.next_f32() * 4.0 - 2.0)
        .collect();

    // Non-trivial weight in (0.5, 1.5) to distinguish scale from identity.
    let weight: Vec<f32> = (0..dim).map(|_| 0.5 + rng.next_f32()).collect();

    // ---- CPU reference ----
    let norm = RmsNorm::from_weight(weight.clone(), eps).expect("RmsNorm::from_weight");
    let out_cpu = norm.forward(&input, n_tokens).expect("RmsNorm::forward");

    // ---- GPU ----
    let ptx = crate::ptx_kernels::rms_norm_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "rms_norm");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&input).expect("d_x");
    let d_weight = DeviceBuffer::<f32>::from_host(&weight).expect("d_weight");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens * dim]).expect("d_out");

    // MUST use block = 256; see module doc for why block < 256 is incorrect.
    let block = 256_u32;
    let params = LaunchParams::new(n_tokens as u32, block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_weight.as_device_ptr(),
                d_out.as_device_ptr(),
                dim as u32,
                n_tokens as u32,
                eps,
            ),
        )
        .expect("launch rms_norm");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_tokens * dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Tolerance: `sqrt.approx.f32` is ~1 ulp, `rcp.approx.f32` ~2 ulp;
    // `div.approx.f32` in the mean_sq divide is ~1 ulp. The butterfly
    // reduction reorders additions which adds a few ulp per accumulation step.
    // Over 64 elements and 8 butterfly steps, 1e-4 relative is comfortable yet
    // still catches a completely wrong normalisation (e.g. missing the eps,
    // dividing by sum instead of mean, or a wrong butterfly stride).
    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for k in 0..out_gpu.len() {
        assert!(
            close(out_gpu[k], out_cpu[k], 1e-4, 1e-6),
            "rms_norm out[{k}] mismatch: gpu={} cpu={} (tok={}, dim={}) \
             (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            out_cpu[k],
            k / dim,
            k % dim
        );
    }
}

// ===========================================================================
// 5. causal_attn_softmax  —  INDEPENDENT HOST RE-DERIVATION (base-e oracle)
// ===========================================================================
//
// The kernel applies a causal mask (future kv positions → −∞) and then
// computes a numerically stable in-place softmax using:
//   exp(score − max) ≈ ex2((score − max) · log2(e))
//
// Oracle tier: INDEPENDENT HOST RE-DERIVATION. The CPU oracle uses
// base-e `f32::exp` in the same max-subtraction form. Any bug where the
// kernel uses `ex2(score − max)` without the `log2(e)` scale factor would
// produce `2^(score−max)` instead of `e^(score−max)`, causing ~20–50% error
// on individual probabilities (they'd still sum to 1 under normalisation, so
// a "sum-to-1" check alone would miss the bug — the base-e oracle catches it).
//
// PTX analysis (correct after fixing the two race conditions):
//   scale = 0F3FB8AA3B = 1.44269504 = log2(e)   ← present and correct
//   ex2((score − max) · log2(e)) = e^(score−max) ← correct
//
// The two SIMT race conditions (MAX broadcast and SUM broadcast) described in
// the module doc were found through this test and fixed in `ptx_kernels.rs`.
//
// Launch geometry: `grid = (n_q, n_heads)`, `block = kv_len` (≤ 256).
// The kernel's %ctaid.x = q_pos, %ctaid.y = head_idx, %tid.x = kv_pos.

/// Numerically stable causal softmax applied in-place on a flat score buffer
/// of shape `[n_q × n_heads × kv_len]`.
///
/// This is the INDEPENDENT host oracle: it does NOT call any crate function.
/// It re-derives the kernel's documented arithmetic from scratch.
fn cpu_causal_softmax(
    scores: &mut [f32],
    n_q: usize,
    n_heads: usize,
    kv_len: usize,
    past_len: u32,
) {
    for q_pos in 0..n_q {
        let abs_q_pos = past_len + q_pos as u32;
        for head_idx in 0..n_heads {
            let row_idx = q_pos * n_heads + head_idx;
            let start = row_idx * kv_len;
            let row = &mut scores[start..start + kv_len];

            // Apply causal mask: kv_pos > abs_q_pos → −∞
            for (kv_pos, s) in row.iter_mut().enumerate() {
                if kv_pos as u32 > abs_q_pos {
                    *s = f32::NEG_INFINITY;
                }
            }

            // Numerically stable softmax.
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for s in row.iter_mut() {
                // Use base-e exp (matches the kernel's ex2((x-max)*log2(e))).
                *s = (*s - max_v).exp();
                sum += *s;
            }
            let inv_sum = 1.0_f32 / sum;
            for s in row.iter_mut() {
                *s *= inv_sum;
            }
        }
    }
}

#[test]
fn causal_attn_softmax_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_q = 2_usize;
    let n_heads = 2_usize;
    let kv_len = 6_usize;
    let past_len = 2_u32;

    // kv_len ≤ 256 so the block fits within the smem[256] reduction array.
    let block = kv_len as u32;
    assert!(
        block <= 256,
        "kv_len must be ≤ 256 for this kernel's smem reduction to be correct"
    );

    let mut rng = LcgRng::new(0xCA50_F75E_0001_u64);

    // Score values in [-2, 2]: a realistic pre-softmax range.  With kv_len=6
    // and typical values all O(1) the max-subtraction keeps all exponentials
    // in (0, 1] so ex2.approx is accurate and no overflow/underflow occurs.
    let n_scores = n_q * n_heads * kv_len;
    let mut scores_init: Vec<f32> = (0..n_scores).map(|_| rng.next_f32() * 4.0 - 2.0).collect();

    // ---- CPU reference (independent re-derivation, base-e) ----
    let mut scores_cpu = scores_init.clone();
    cpu_causal_softmax(&mut scores_cpu, n_q, n_heads, kv_len, past_len);

    // Structural invariants on CPU oracle (sanity-check the oracle itself).
    for q_pos in 0..n_q {
        let abs_q_pos = past_len + q_pos as u32;
        for head_idx in 0..n_heads {
            let row_idx = q_pos * n_heads + head_idx;
            let start = row_idx * kv_len;
            let row = &scores_cpu[start..start + kv_len];

            // Non-negative probabilities.
            for (kv_pos, &p) in row.iter().enumerate() {
                assert!(
                    p >= 0.0,
                    "cpu oracle: score[q={q_pos},h={head_idx},kv={kv_pos}] = {p} < 0"
                );
                // Masked positions must be 0 (exp(-inf) = 0).
                if kv_pos as u32 > abs_q_pos {
                    assert!(
                        p < 1e-30,
                        "cpu oracle: masked position kv={kv_pos} has p={p} ≠ 0"
                    );
                }
            }

            // Row sums to 1.
            let row_sum: f32 = row.iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "cpu oracle: row sum = {row_sum} ≠ 1 for q={q_pos} h={head_idx}"
            );
        }
    }

    // ---- GPU ----
    let ptx = crate::ptx_kernels::causal_attn_softmax_ptx(fx.sm);
    let kernel = load_kernel(&ptx, "causal_attn_softmax");
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_scores = DeviceBuffer::<f32>::from_host(&scores_init).expect("d_scores");

    // 2-D grid: (n_q, n_heads) blocks, 1-D block of kv_len threads.
    let params = LaunchParams::new((n_q as u32, n_heads as u32), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_scores.as_device_ptr(),
                kv_len as u32,
                n_heads as u32,
                past_len,
            ),
        )
        .expect("launch causal_attn_softmax");
    stream.synchronize().expect("sync");

    d_scores
        .copy_to_host(&mut scores_init)
        .expect("copy scores");
    let out_gpu = scores_init;

    // GPU structural invariants (non-negative, masked → 0, row sums to 1).
    for q_pos in 0..n_q {
        let abs_q_pos = past_len + q_pos as u32;
        for head_idx in 0..n_heads {
            let row_idx = q_pos * n_heads + head_idx;
            let start = row_idx * kv_len;
            let row = &out_gpu[start..start + kv_len];

            for (kv_pos, &p) in row.iter().enumerate() {
                assert!(
                    p.is_finite() && p >= 0.0,
                    "causal_attn_softmax: out[q={q_pos},h={head_idx},kv={kv_pos}] = {p} \
                     not a finite non-negative probability"
                );
                if kv_pos as u32 > abs_q_pos {
                    assert!(
                        p < 1e-30,
                        "causal_attn_softmax: masked kv={kv_pos} has p={p} ≠ 0"
                    );
                }
            }
            let row_sum: f32 = row.iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "causal_attn_softmax: row sum = {row_sum} ≠ 1 for q={q_pos} h={head_idx}"
            );
        }
    }

    // Numerical comparison against base-e host oracle.
    //
    // Tolerance: `ex2.approx.f32` is ~2 ulp; the division (via rcp.approx)
    // adds ~2 ulp. For values in the masked-softmax range (all probs between 0
    // and 1 after normalisation), 1e-4 relative is comfortable and still detects
    // a missing log2(e) scale factor by three orders of magnitude (that error is
    // ~20–50% in relative terms for typical scores of O(1)).
    let (rel, abs) = worst_diff(&out_gpu, &scores_cpu);
    for k in 0..out_gpu.len() {
        // Skip masked positions (both GPU and CPU should be ≈ 0).
        if scores_cpu[k] < 1e-30 && out_gpu[k] < 1e-30 {
            continue;
        }
        assert!(
            close(out_gpu[k], scores_cpu[k], 1e-4, 1e-6),
            "causal_attn_softmax out[{k}] mismatch: gpu={} cpu={} \
             (worst rel={rel:e} abs={abs:e})",
            out_gpu[k],
            scores_cpu[k]
        );
    }
}
