//! On-device GPU validation for the hand-written PTX kernels in
//! [`crate::ptx_kernels`].
//!
//! Each test JIT-compiles a kernel's PTX for the live device's SM version via
//! `Module::from_ptx`, launches it through `oxicuda-launch`, copies results back,
//! and checks them against a CPU reference or an independent re-derivation. The
//! launch ABI follows the same convention as `oxicuda-snn` / `oxicuda-recsys`:
//! device buffers are passed as their `CUdeviceptr` (`.param .u64`), scalars as
//! the matching Rust scalar type, in declared order.
//!
//! ## Oracle tiers (honest accounting)
//!
//! * **Crate oracle** (strongest) — compared within FP32 tolerance to a `pub`
//!   CPU function the kernel mirrors:
//!   - `top_k_gate`     → [`crate::routing::top_k::stable_softmax`] + arg-max.
//!   - `router_z_loss`  → [`crate::loss::router_z::router_z_loss`].
//!   - `expert_dispatch`→ capacity-bounded assignment re-implemented from the
//!     same semantics (placed iff old_count < capacity).
//!   - `soft_moe_dispatch` → [`crate::routing::soft_moe::SoftMoeRouter::dispatch_weights`]
//!     (full slot-softmax dispatch matrix; each output row sums to 1).
//! * **Independent host re-derivation** — the kernel computes a *documented,
//!   intentionally simplified* fragment that has no standalone crate function, so
//!   the oracle re-implements exactly the arithmetic the PTX performs:
//!   - `expert_ffn`     → first-element GELU FFN cell (see test header).
//!   - `expert_combine` → score-weighted scatter of the first feature.
//! * **Load + re-derivation of a PROXY (documented divergence)** — `load_balance_loss`
//!   does NOT compute its namesake quantity; its single-pass arithmetic is a
//!   proxy. The test re-derives exactly what the PTX does (catching a ptxas
//!   miscompile, a wrong constant, or a race) and the header documents how it
//!   diverges from the real algorithm. This is NOT counted as crate-validated.
//!
//! ## PTX bugs found and FIXED in `ptx_kernels.rs`
//!
//! 1. `expert_dispatch_kernel` — **invalid PTX (ptxas reject)**: the epilogue used
//!    `mov.f32 %f0, ...` but the kernel declared no `.reg .f32`. ptxas (sm_86)
//!    aborted with "Unknown symbol '%f0'". Fixed by adding `.reg .f32 %f<1>;`.
//! 2. `expert_ffn_kernel` — **wrong math (base-2 / tanh)**: the GELU tanh was
//!    computed as `(e^z - 1)/(e^z + 1)`, which equals `tanh(z/2)`, not `tanh(z)`.
//!    Fixed by doubling the exponent (`e^{2z}`) so the identity
//!    `tanh(z) = (e^{2z} - 1)/(e^{2z} + 1)` holds. The `* log2e` base conversion
//!    before `ex2.approx` was already present and correct.
//!
//! Every test skips gracefully (returns) when no CUDA device is present.

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

/// JIT-compile `ptx` for the live device and look up `entry`.
///
/// A failure here means ptxas rejected the PTX — a real, must-fix bug.
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

/// CPU reference for numerically-stable softmax (mirrors `stable_softmax`).
fn cpu_softmax(logits: &[f32]) -> Vec<f32> {
    crate::routing::top_k::stable_softmax(logits)
}

// ===========================================================================
// 1. top_k_gate  —  CRATE ORACLE (stable_softmax + arg-max index)
// ===========================================================================

#[test]
fn top_k_gate_softmax_and_argmax_match_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 8_usize;
    let n_experts = 4_usize;
    let k = 1_u32; // kernel writes full softmax + top-1 index; k is unused by it.

    // Deterministic logits with an unambiguous max per token (no ties).
    let logits: Vec<f32> = (0..n_tokens * n_experts)
        .map(|i| {
            let t = (i / n_experts) as f32;
            let e = (i % n_experts) as f32;
            // Distinct, spread values; the +e*0.37 term breaks ties cleanly.
            (t * 0.13).sin() * 2.0 + e * 0.37 - 1.0
        })
        .collect();

    // ---- CPU reference ----
    let mut scores_cpu = vec![0.0_f32; n_tokens * n_experts];
    let mut idx_cpu = vec![0_u32; n_tokens];
    for tok in 0..n_tokens {
        let row = &logits[tok * n_experts..(tok + 1) * n_experts];
        let probs = cpu_softmax(row);
        scores_cpu[tok * n_experts..(tok + 1) * n_experts].copy_from_slice(&probs);
        // First strict-max index (matches the kernel's `setp.gt.f32` tie-break).
        let mut best = f32::NEG_INFINITY;
        let mut best_idx = 0_u32;
        for (e, &p) in probs.iter().enumerate() {
            if p > best {
                best = p;
                best_idx = e as u32;
            }
        }
        idx_cpu[tok] = best_idx;
    }

    // ---- GPU ----
    let kernel = load_kernel(
        &crate::ptx_kernels::top_k_gate_ptx(fx.sm),
        "top_k_gate_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_scores =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens * n_experts]).expect("d_scores");
    let d_idx = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_tokens]).expect("d_idx");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_scores.as_device_ptr(),
                d_idx.as_device_ptr(),
                n_tokens as u32,
                n_experts as u32,
                k,
            ),
        )
        .expect("launch top_k_gate_kernel");
    stream.synchronize().expect("sync");

    let mut scores_gpu = vec![0.0_f32; n_tokens * n_experts];
    let mut idx_gpu = vec![0_u32; n_tokens];
    d_scores.copy_to_host(&mut scores_gpu).expect("copy scores");
    d_idx.copy_to_host(&mut idx_gpu).expect("copy idx");

    // Softmax probabilities: GPU uses `ex2.approx` (a few ulp); 5e-4 rel covers it.
    let (rel, abs) = worst_diff(&scores_gpu, &scores_cpu);
    for k in 0..scores_gpu.len() {
        assert!(
            close(scores_gpu[k], scores_cpu[k], 5e-4, 1e-6),
            "top_k_gate scores[{k}] gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            scores_gpu[k],
            scores_cpu[k]
        );
    }
    // Each token's softmax row must sum to ~1.
    for tok in 0..n_tokens {
        let s: f32 = scores_gpu[tok * n_experts..(tok + 1) * n_experts]
            .iter()
            .sum();
        assert!((s - 1.0).abs() < 1e-3, "token {tok} softmax sum {s} != 1");
    }
    // Top-1 index must match exactly.
    for tok in 0..n_tokens {
        assert_eq!(
            idx_gpu[tok], idx_cpu[tok],
            "top_k_gate argmax token {tok}: gpu={} cpu={}",
            idx_gpu[tok], idx_cpu[tok]
        );
    }
}

// ===========================================================================
// 2. expert_dispatch  —  CRATE-LOGIC ORACLE (capacity-bounded assignment)
// ===========================================================================
//
// PTX BUG FOUND AND FIXED: the kernel declared no `.reg .f32` yet the epilogue
// did `mov.f32 %f0, 0.0`. ptxas (sm_86) rejected the module with
// "Unknown symbol '%f0'". Fixed by adding `.reg .f32 %f<1>;`.
//
// Semantics: for each token, atomically post-increment slot_counts[expert] and
// read the OLD count; place (write expert id) iff old_count < capacity, else
// write the sentinel 0xFFFFFFFF. The slot_counts histogram is order-independent;
// the placement is order-independent ONLY when no expert exceeds capacity.

#[test]
fn expert_dispatch_no_overflow_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 8_usize;
    let n_experts = 4_usize;
    let capacity = 100_u32; // large → every token is placed regardless of order.
    let expert_ids: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3];

    // CPU oracle: all placed → output == input; counts == histogram.
    let mut counts_cpu = vec![0_u32; n_experts];
    for &e in &expert_ids {
        counts_cpu[e as usize] += 1;
    }

    let kernel = load_kernel(
        &crate::ptx_kernels::expert_dispatch_ptx(fx.sm),
        "expert_dispatch_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ids = DeviceBuffer::<u32>::from_host(&expert_ids).expect("d_ids");
    let d_counts = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_experts]).expect("d_counts");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_tokens]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ids.as_device_ptr(),
                d_counts.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                capacity,
            ),
        )
        .expect("launch expert_dispatch_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; n_tokens];
    let mut counts_gpu = vec![0_u32; n_experts];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");
    d_counts.copy_to_host(&mut counts_gpu).expect("copy counts");

    for t in 0..n_tokens {
        assert_eq!(
            out_gpu[t], expert_ids[t],
            "dispatch (no overflow) token {t}: gpu={} expected expert {}",
            out_gpu[t], expert_ids[t]
        );
    }
    for e in 0..n_experts {
        assert_eq!(
            counts_gpu[e], counts_cpu[e],
            "dispatch slot_counts[{e}]: gpu={} cpu={}",
            counts_gpu[e], counts_cpu[e]
        );
    }
}

#[test]
fn expert_dispatch_overflow_aggregate_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    // All 8 tokens to expert 0, capacity 3 → exactly 3 placed, 5 overflow.
    // WHICH tokens win is order-dependent (atomics race), but the aggregate is
    // deterministic: 3 placed (id==0), 5 sentinel, and counts[0]==8.
    let n_tokens = 8_usize;
    let n_experts = 2_usize;
    let capacity = 3_u32;
    let expert_ids: Vec<u32> = vec![0; n_tokens];
    let sentinel = u32::MAX;

    let kernel = load_kernel(
        &crate::ptx_kernels::expert_dispatch_ptx(fx.sm),
        "expert_dispatch_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_ids = DeviceBuffer::<u32>::from_host(&expert_ids).expect("d_ids");
    let d_counts = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_experts]).expect("d_counts");
    let d_out = DeviceBuffer::<u32>::from_host(&vec![0_u32; n_tokens]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_ids.as_device_ptr(),
                d_counts.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                capacity,
            ),
        )
        .expect("launch expert_dispatch_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0_u32; n_tokens];
    let mut counts_gpu = vec![0_u32; n_experts];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");
    d_counts.copy_to_host(&mut counts_gpu).expect("copy counts");

    let placed = out_gpu.iter().filter(|&&v| v == 0).count();
    let overflow = out_gpu.iter().filter(|&&v| v == sentinel).count();
    assert_eq!(
        placed, capacity as usize,
        "expected {capacity} placed, got {placed}"
    );
    assert_eq!(
        overflow,
        n_tokens - capacity as usize,
        "expected {} overflow, got {overflow}",
        n_tokens - capacity as usize
    );
    assert_eq!(
        counts_gpu[0], n_tokens as u32,
        "counts[0] should be {n_tokens}"
    );
    assert_eq!(counts_gpu[1], 0, "counts[1] should be 0");
}

// ===========================================================================
// 3. expert_ffn  —  INDEPENDENT HOST RE-DERIVATION (first-element GELU cell)
// ===========================================================================
//
// HONEST SCOPE: `expert_ffn_kernel` is documented as a *simplified* cell that
// only computes the FIRST output feature using w1[0], b1[0], w2[0], b2[0] and
// x[token*input_dim + 0] (it does NOT sum over input_dim or loop over ffn_dim).
// It therefore has no equivalence to `ExpertFfn::forward`. The oracle re-derives
// EXACTLY the documented arithmetic with the CORRECT GELU tanh formula.
//
// PTX BUG FOUND AND FIXED: the GELU tanh used `(e^z-1)/(e^z+1) = tanh(z/2)`
// instead of `tanh(z)`. Fixed to `(e^{2z}-1)/(e^{2z}+1)`. Before the fix this
// re-derivation FAILS (gpu uses tanh(z/2)); after the fix it matches.

/// GELU (tanh approximation, OpenAI variant) — mirrors `ffn::gelu_approx`.
fn gelu_tanh(v: f32) -> f32 {
    const GELU_COEFF: f32 = 0.797_884_6_f32;
    const GELU_CUBIC: f32 = 0.044_715_f32;
    v * 0.5 * (1.0 + (GELU_COEFF * (v + GELU_CUBIC * v * v * v)).tanh())
}

#[test]
fn expert_ffn_first_cell_matches_rederivation() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 4_usize;
    let input_dim = 3_usize;
    let ffn_dim = 2_usize;

    // Deterministic, modest-magnitude inputs (keeps the tanh argument well within
    // ex2.approx's accurate range).
    let x: Vec<f32> = (0..n_tokens * input_dim)
        .map(|i| 0.1 * (i as f32) - 0.4)
        .collect();
    let w1: Vec<f32> = (0..ffn_dim * input_dim)
        .map(|i| 0.2 + 0.05 * i as f32)
        .collect();
    let b1: Vec<f32> = (0..ffn_dim).map(|i| 0.01 * i as f32 - 0.05).collect();
    let w2: Vec<f32> = (0..input_dim * ffn_dim)
        .map(|i| 0.15 - 0.03 * i as f32)
        .collect();
    let b2: Vec<f32> = (0..input_dim).map(|i| 0.02 * i as f32 + 0.1).collect();

    // Oracle: only out[token*input_dim + 0] is written by the kernel; the rest
    // remain at their initial value (0).
    let mut out_cpu = vec![0.0_f32; n_tokens * input_dim];
    for t in 0..n_tokens {
        let pre = w1[0] * x[t * input_dim] + b1[0];
        let g = gelu_tanh(pre);
        out_cpu[t * input_dim] = w2[0] * g + b2[0];
    }

    let kernel = load_kernel(
        &crate::ptx_kernels::expert_ffn_ptx(fx.sm),
        "expert_ffn_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_w1 = DeviceBuffer::<f32>::from_host(&w1).expect("d_w1");
    let d_b1 = DeviceBuffer::<f32>::from_host(&b1).expect("d_b1");
    let d_w2 = DeviceBuffer::<f32>::from_host(&w2).expect("d_w2");
    let d_b2 = DeviceBuffer::<f32>::from_host(&b2).expect("d_b2");
    let d_out =
        DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens * input_dim]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_w1.as_device_ptr(),
                d_b1.as_device_ptr(),
                d_w2.as_device_ptr(),
                d_b2.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                input_dim as u32,
                ffn_dim as u32,
            ),
        )
        .expect("launch expert_ffn_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_tokens * input_dim];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for t in 0..n_tokens {
        let idx = t * input_dim;
        assert!(
            close(out_gpu[idx], out_cpu[idx], 1e-3, 1e-5),
            "expert_ffn out[{idx}] gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[idx],
            out_cpu[idx]
        );
        // Untouched features must stay zero.
        for d in 1..input_dim {
            assert_eq!(
                out_gpu[idx + d].to_bits(),
                0_u32,
                "expert_ffn out[{}] should be untouched (0), got {}",
                idx + d,
                out_gpu[idx + d]
            );
        }
    }
}

// ===========================================================================
// 4. expert_combine  —  INDEPENDENT HOST RE-DERIVATION (weighted scatter)
// ===========================================================================
//
// HONEST SCOPE: the kernel combines only the FIRST feature of each slot:
//   combined_out[token_id*d_model + 0] += score[slot] * expert_out[slot*d_model + 0]
// (no loop over d_model). With distinct token ids each output cell receives a
// single atomic add, so the result is exact (no accumulation-order ambiguity).

#[test]
fn expert_combine_first_feature_matches_rederivation() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_slots = 6_usize;
    let d_model = 4_usize;
    let n_tokens = n_slots; // distinct token per slot → one add per output cell.

    let expert_out: Vec<f32> = (0..n_slots * d_model)
        .map(|i| 0.3 * (i as f32) - 1.0)
        .collect();
    let scores: Vec<f32> = (0..n_slots).map(|i| 0.1 + 0.15 * i as f32).collect();
    let token_ids: Vec<u32> = (0..n_slots as u32).collect();

    // Oracle: each combined_out[t*d_model] gets exactly one weighted add.
    let mut out_cpu = vec![0.0_f32; n_tokens * d_model];
    for s in 0..n_slots {
        let t = token_ids[s] as usize;
        out_cpu[t * d_model] = scores[s] * expert_out[s * d_model];
    }

    let kernel = load_kernel(
        &crate::ptx_kernels::expert_combine_ptx(fx.sm),
        "expert_combine_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_expert = DeviceBuffer::<f32>::from_host(&expert_out).expect("d_expert");
    let d_scores = DeviceBuffer::<f32>::from_host(&scores).expect("d_scores");
    let d_tokens = DeviceBuffer::<u32>::from_host(&token_ids).expect("d_tokens");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens * d_model]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_slots as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_expert.as_device_ptr(),
                d_scores.as_device_ptr(),
                d_tokens.as_device_ptr(),
                d_out.as_device_ptr(),
                n_slots as u32,
                d_model as u32,
            ),
        )
        .expect("launch expert_combine_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_tokens * d_model];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    let (rel, abs) = worst_diff(&out_gpu, &out_cpu);
    for t in 0..n_tokens {
        let idx = t * d_model;
        assert!(
            close(out_gpu[idx], out_cpu[idx], 1e-5, 1e-6),
            "expert_combine out[{idx}] gpu={} cpu={} (worst rel={rel:e} abs={abs:e})",
            out_gpu[idx],
            out_cpu[idx]
        );
        for d in 1..d_model {
            assert_eq!(
                out_gpu[idx + d].to_bits(),
                0_u32,
                "expert_combine out[{}] should be untouched (0), got {}",
                idx + d,
                out_gpu[idx + d]
            );
        }
    }
}

// ===========================================================================
// 5. router_z_loss  —  CRATE ORACLE (crate::loss::router_z::router_z_loss)
// ===========================================================================
//
// Per token the kernel computes lse = max + ln(Σ exp(logit-max) + eps) and
// atomically adds lse² / n_tokens. The grand total therefore equals
// mean_t(lse²) = router_z_loss. The base conversions are CORRECT: `ex2` is fed
// `(logit-max) * log2e` and `lg2` is scaled by `ln2` to recover the natural log.

#[test]
fn router_z_loss_matches_cpu() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 8_usize;
    let n_experts = 4_usize;
    let logits: Vec<f32> = (0..n_tokens * n_experts)
        .map(|i| {
            let t = (i / n_experts) as f32;
            let e = (i % n_experts) as f32;
            (t * 0.21).cos() * 1.5 + e * 0.4 - 0.7
        })
        .collect();

    let loss_cpu = crate::loss::router_z::router_z_loss(&logits, n_tokens, n_experts)
        .expect("cpu router_z_loss");

    let kernel = load_kernel(
        &crate::ptx_kernels::router_z_loss_ptx(fx.sm),
        "router_z_loss_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                n_experts as u32,
            ),
        )
        .expect("launch router_z_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = [0.0_f32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert!(
        out_gpu[0].is_finite() && out_gpu[0] >= 0.0,
        "router_z_loss gpu must be finite & non-negative, got {}",
        out_gpu[0]
    );
    assert!(
        close(out_gpu[0], loss_cpu, 2e-3, 1e-5),
        "router_z_loss gpu={} cpu={}",
        out_gpu[0],
        loss_cpu
    );
}

// ===========================================================================
// 6. load_balance_loss  —  PROXY (documented divergence); re-derive PTX exactly
// ===========================================================================
//
// HONEST SCOPE: this kernel does NOT compute the Switch load-balance loss
// `n_experts * Σ_i f_i P_i` (which is not separable into a per-token sum and so
// cannot be produced by a single-pass atomic kernel with this signature). The
// kernel instead atomically accumulates, per non-overflow token,
//   contribution = exp2( (1/n_tokens) * logit[token, assignment] ) * (1/n_tokens)
// using `ex2.approx` (base-2) and `rcp.approx`. We re-derive EXACTLY that (a
// faithful proxy oracle) so the test still catches a ptxas miscompile, a wrong
// constant, a wrong index, or a race — while NOT pretending it equals the loss.

#[test]
fn load_balance_loss_proxy_matches_rederivation() {
    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_tokens = 8_usize;
    let n_experts = 4_usize;
    let logits: Vec<f32> = (0..n_tokens * n_experts)
        .map(|i| 0.05 * (i as f32) - 0.3)
        .collect();
    // All-valid assignments (no 0xFFFFFFFF sentinel).
    let assignments: Vec<u32> = (0..n_tokens as u32).map(|t| t % n_experts as u32).collect();

    // Faithful re-derivation of the PTX arithmetic (base-2 exp, 1/T weight).
    let inv_t = 1.0_f32 / n_tokens as f32;
    let mut proxy_cpu = 0.0_f32;
    for t in 0..n_tokens {
        let a = assignments[t] as usize;
        let logit = logits[t * n_experts + a];
        proxy_cpu += (inv_t * logit).exp2() * inv_t;
    }

    let kernel = load_kernel(
        &crate::ptx_kernels::load_balance_loss_ptx(fx.sm),
        "load_balance_loss_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_logits = DeviceBuffer::<f32>::from_host(&logits).expect("d_logits");
    let d_assign = DeviceBuffer::<u32>::from_host(&assignments).expect("d_assign");
    let d_out = DeviceBuffer::<f32>::from_host(&[0.0_f32; 1]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_logits.as_device_ptr(),
                d_assign.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                n_experts as u32,
            ),
        )
        .expect("launch load_balance_loss_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = [0.0_f32; 1];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    assert!(out_gpu[0].is_finite(), "load_balance proxy must be finite");
    assert!(
        close(out_gpu[0], proxy_cpu, 2e-3, 1e-5),
        "load_balance proxy gpu={} cpu(re-derived)={}",
        out_gpu[0],
        proxy_cpu
    );
}

// ===========================================================================
// 7. soft_moe_dispatch  —  CRATE ORACLE: full slot-softmax dispatch
// ===========================================================================
//
// The kernel now computes the real Soft-MoE dispatch matrix
// `D = softmax(X · Φ / sqrt(d), dim=slots)`, validated element-wise against
// [`crate::routing::soft_moe::SoftMoeRouter::dispatch_weights`] (the same Φ that
// the router holds is fed to the kernel). Φ is laid out `[n_slots, input_dim]`
// with `Φ[s,d] = phi[s*input_dim + d]`. Each output row must additionally be a
// probability distribution (sums to 1) over the slot dimension.

#[test]
fn soft_moe_dispatch_matches_router_dispatch_weights() {
    use crate::handle::LcgRng;
    use crate::routing::soft_moe::{SoftMoeConfig, SoftMoeRouter};

    let Some(fx) = gpu_fixture() else {
        return;
    };

    let n_experts = 3_usize;
    let n_slots_per_expert = 2_usize;
    let input_dim = 8_usize;
    let n_slots = n_experts * n_slots_per_expert;
    let n_tokens = 5_usize;
    let scale = 1.0_f32 / (input_dim as f32).sqrt();

    let cfg = SoftMoeConfig {
        n_experts,
        n_slots_per_expert,
        input_dim,
    };
    let mut rng = LcgRng::new(0x50f7_3110_u64);
    let router = SoftMoeRouter::new(cfg, &mut rng).expect("build SoftMoeRouter");

    // Deterministic, modest-magnitude token features.
    let x: Vec<f32> = (0..n_tokens * input_dim)
        .map(|i| 0.13 * (i as f32) - 0.4 + 0.05 * ((i % 5) as f32))
        .collect();

    // CRATE ORACLE: the full slot-softmax dispatch matrix.
    let out_cpu = router
        .dispatch_weights(&x, n_tokens)
        .expect("dispatch_weights oracle");

    let kernel = load_kernel(
        &crate::ptx_kernels::soft_moe_dispatch_ptx(fx.sm),
        "soft_moe_dispatch_kernel",
    );
    let stream = Stream::new(&fx.ctx).expect("stream");

    let d_x = DeviceBuffer::<f32>::from_host(&x).expect("d_x");
    let d_phi = DeviceBuffer::<f32>::from_host(&router.phi).expect("d_phi");
    let d_out = DeviceBuffer::<f32>::from_host(&vec![0.0_f32; n_tokens * n_slots]).expect("d_out");

    let block = 256_u32;
    let params = LaunchParams::new(grid_1d(n_tokens as u32, block), block);
    kernel
        .launch(
            &params,
            &stream,
            &(
                d_x.as_device_ptr(),
                d_phi.as_device_ptr(),
                d_out.as_device_ptr(),
                n_tokens as u32,
                n_slots as u32,
                input_dim as u32,
                scale,
            ),
        )
        .expect("launch soft_moe_dispatch_kernel");
    stream.synchronize().expect("sync");

    let mut out_gpu = vec![0.0_f32; n_tokens * n_slots];
    d_out.copy_to_host(&mut out_gpu).expect("copy out");

    // Element-wise agreement with the crate oracle (full softmax over all slots).
    let (worst_rel, worst_abs) = worst_diff(&out_gpu, &out_cpu);
    for (idx, (&g, &c)) in out_gpu.iter().zip(out_cpu.iter()).enumerate() {
        assert!(
            close(g, c, 5e-4, 5e-4),
            "soft_moe_dispatch[{idx}]: gpu={g} cpu={c} (worst_rel={worst_rel}, worst_abs={worst_abs})"
        );
    }

    // Each token's output row is a probability distribution over slots.
    for t in 0..n_tokens {
        let row_sum: f32 = out_gpu[t * n_slots..(t + 1) * n_slots].iter().sum();
        assert!(
            (row_sum - 1.0).abs() <= 1e-4,
            "soft_moe_dispatch token {t} row sum = {row_sum}, expected 1.0"
        );
    }
}
