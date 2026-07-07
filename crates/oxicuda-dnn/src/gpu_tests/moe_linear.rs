//! On-device GPU validation for the `moe_linear` subsystem of `oxicuda-dnn`.
//!
//! Covers the Mixture-of-Experts kernels (`moe/{routing,permute,capacity,
//! aux_loss,fused_moe,monitoring}`) plus `linear/fused_linear`. Each test
//! JITs the kernel PTX (directly, or via the crate's public op API), launches
//! on the live device, copies the result back, and compares against an
//! independent CPU re-derivation.
//!
//! ## Per-kernel classification (verified against source)
//!
//! * `moe_permute_tokens` / `moe_unpermute_tokens` — complete → numeric oracle.
//! * `moe_dynamic_capacity` — complete → numeric oracle (exact integer output).
//! * `moe_overflow_mask` — token→mask assignment is atomic-order dependent, but
//!   per-expert overflow *count* is deterministic → aggregate numeric oracle.
//! * `moe_utilization_count` — atomic histogram, order-independent → byte-exact.
//! * `moe_load_balance_loss` — complete; reduced quantity is linear. The warp
//!   `shfl.sync` membermask is `0xFFFFFFFF`, so we launch a full 32-lane warp
//!   (`num_experts = 32`) to keep the shuffle well-defined → numeric oracle.
//! * `moe_imbalance_score` — correct only for a single full warp
//!   (`num_experts <= 32`); launched with a full 32-lane warp → numeric oracle.
//! * `moe_z_loss` — complete log-sum-exp (base-2 `ex2`/`lg2`) → numeric oracle
//!   with an approximation tolerance.
//! * `moe_fused_token_parallel` — the genuinely complete end-to-end fused FFN
//!   path → numeric oracle (identity + ReLU).
//! * `moe_expand_tokens` / `moe_activation` — complete → numeric oracle.
//! * `moe_topk_softmax` — computes the argmax (top-1) per token; validated as a
//!   numeric argmax oracle for `top_k = 1`. `moe_sort_by_expert` is exercised
//!   for fault-free launch only (its permutation is a known-incomplete
//!   fragment: `moe_routing` zero-inits the offsets it scatters into).
//! * `fused_linear` — `generate_fused_linear_ptx` is a comment-only skeleton
//!   (loads params, never stores) → load/launch-only: assert it runs
//!   fault-free and leaves the output untouched.
//! * `generate_epilogue_ptx` (FP8) — comment-only string, sm_89+ → skipped.

use super::*;

use oxicuda_launch::{Dim3, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use crate::linear::fused_linear::{FusedLinearConfig, fused_linear};
use crate::moe::aux_loss::{AuxLossConfig, AuxLossPlan};
use crate::moe::capacity::{CapacityConfig, CapacityPlan};
use crate::moe::monitoring::{MoeMonitor, MoeUtilizationReport};
use crate::moe::permute::{permute_tokens, unpermute_tokens};
use crate::moe::routing::{MoeConfig, moe_routing};
use crate::types::{Activation, TensorDesc, TensorDescMut, TensorLayout};

// ---------------------------------------------------------------------------
// Small local helpers
// ---------------------------------------------------------------------------

/// Builds a row-major `TensorDesc` over an already-populated device buffer.
fn desc_rowmajor(buf: &DeviceBuffer<f32>, dims: Vec<u32>) -> TensorDesc<f32> {
    let strides = row_major_strides(&dims);
    TensorDesc::<f32>::from_raw(buf.as_device_ptr(), dims, strides, TensorLayout::RowMajor)
        .expect("valid tensor desc")
}

/// Builds a mutable row-major `TensorDescMut` over a device buffer.
fn desc_rowmajor_mut(buf: &DeviceBuffer<f32>, dims: Vec<u32>) -> TensorDescMut<f32> {
    let strides = row_major_strides(&dims);
    TensorDescMut::<f32>::from_raw(buf.as_device_ptr(), dims, strides, TensorLayout::RowMajor)
        .expect("valid tensor desc")
}

/// Computes contiguous row-major strides (in elements) for `dims`.
fn row_major_strides(dims: &[u32]) -> Vec<u32> {
    let mut strides = vec![1u32; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

// ---------------------------------------------------------------------------
// permute / unpermute
// ---------------------------------------------------------------------------

#[test]
fn permute_tokens_scatter_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_rows: u32 = 6;
    let hidden: u32 = 5;
    let mut rng = Lcg::new(0x9e37_79b9);

    let host_in: Vec<f32> = (0..num_rows * hidden)
        .map(|_| rng.range_f32(-2.0, 2.0))
        .collect();
    // A valid permutation of [0, num_rows): output[perm[row]] = input[row].
    let perm: Vec<i32> = vec![3, 0, 5, 1, 4, 2];

    let in_buf = DeviceBuffer::<f32>::from_host(&host_in).expect("upload input");
    let perm_buf = DeviceBuffer::<i32>::from_host(&perm).expect("upload perm");
    let out_buf =
        DeviceBuffer::<f32>::from_host(&vec![0.0f32; (num_rows * hidden) as usize]).expect("out");

    let input = desc_rowmajor(&in_buf, vec![num_rows, hidden]);
    let mut output = desc_rowmajor_mut(&out_buf, vec![num_rows, hidden]);

    permute_tokens::<f32>(&fx.handle, &input, &perm_buf, &mut output).expect("permute launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; (num_rows * hidden) as usize];
    out_buf.copy_to_host(&mut gpu).expect("download");

    let mut cpu = vec![0.0f32; (num_rows * hidden) as usize];
    for row in 0..num_rows as usize {
        let dest = perm[row] as usize;
        for c in 0..hidden as usize {
            cpu[dest * hidden as usize + c] = host_in[row * hidden as usize + c];
        }
    }
    assert_close_f32(&gpu, &cpu, 0.0, 0.0, "permute_tokens");
}

#[test]
fn unpermute_tokens_weighted_gather_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_tokens: u32 = 4;
    let hidden: u32 = 5;
    let top_k: u32 = 2;
    let total = (num_tokens * top_k) as usize;
    let mut rng = Lcg::new(0x1234_5678);

    let expert_out: Vec<f32> = (0..total * hidden as usize)
        .map(|_| rng.range_f32(-1.5, 1.5))
        .collect();
    // perm[t*top_k+j] indexes a row of expert_out (any in-range row).
    let perm: Vec<i32> = vec![0, 7, 2, 5, 4, 1, 6, 3];
    let weights: Vec<f32> = (0..total).map(|_| rng.range_f32(0.1, 0.9)).collect();

    let eo_buf = DeviceBuffer::<f32>::from_host(&expert_out).expect("eo");
    let perm_buf = DeviceBuffer::<i32>::from_host(&perm).expect("perm");
    let wt_buf = DeviceBuffer::<f32>::from_host(&weights).expect("wt");
    let out_buf =
        DeviceBuffer::<f32>::from_host(&vec![0.0f32; (num_tokens * hidden) as usize]).expect("out");

    let eo = desc_rowmajor(&eo_buf, vec![num_tokens * top_k, hidden]);
    let mut output = desc_rowmajor_mut(&out_buf, vec![num_tokens, hidden]);

    unpermute_tokens::<f32>(&fx.handle, &eo, &perm_buf, &wt_buf, &mut output, top_k)
        .expect("unpermute launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; (num_tokens * hidden) as usize];
    out_buf.copy_to_host(&mut gpu).expect("download");

    let mut cpu = vec![0.0f32; (num_tokens * hidden) as usize];
    for t in 0..num_tokens as usize {
        for c in 0..hidden as usize {
            let mut acc = 0.0f32;
            for j in 0..top_k as usize {
                let slot = t * top_k as usize + j;
                let src = perm[slot] as usize;
                acc += weights[slot] * expert_out[src * hidden as usize + c];
            }
            cpu[t * hidden as usize + c] = acc;
        }
    }
    assert_close_f32(&gpu, &cpu, 1e-5, 1e-5, "unpermute_tokens");
}

// ---------------------------------------------------------------------------
// capacity: dynamic_capacity (numeric) + overflow_mask (aggregate)
// ---------------------------------------------------------------------------

/// CPU mirror of the dynamic-capacity kernel, computed in `f32` to match the
/// device's `cvt.rpi`(ceil)/`cvt.rzi`(truncate) rounding exactly.
fn dynamic_capacity_oracle(
    observed: &[u32],
    base: u32,
    min_cap: u32,
    max_cap: u32,
    tokens_per_batch: u32,
    num_experts: u32,
) -> Vec<u32> {
    let expected = tokens_per_batch as f32 / num_experts as f32;
    observed
        .iter()
        .map(|&obs| {
            let ratio = obs as f32 / expected;
            let new_cap = (base as f32 * ratio).ceil() as u32;
            let clamped = new_cap.max(min_cap);
            if max_cap > 0 {
                clamped.min(max_cap)
            } else {
                clamped
            }
        })
        .collect()
}

fn run_dynamic_capacity(
    fx: &GpuFixture,
    observed: &[u32],
    base: u32,
    min_cap: u32,
    max_cap: u32,
    tokens_per_batch: u32,
) -> Vec<u32> {
    let num_experts = observed.len() as u32;
    let cfg = CapacityConfig {
        num_experts,
        capacity_factor: 1.25,
        min_capacity: min_cap,
        max_capacity: max_cap,
        tokens_per_batch,
        sm_version: fx.sm,
    };
    let plan = CapacityPlan::new(cfg).expect("plan");
    let ptx = plan.generate_dynamic_capacity_ptx().expect("ptx");
    let kernel = load_kernel(&ptx, "moe_dynamic_capacity");

    let obs_buf = DeviceBuffer::<u32>::from_host(observed).expect("obs");
    let adj_buf = DeviceBuffer::<u32>::from_host(&vec![0u32; num_experts as usize]).expect("adj");

    let block = 256u32;
    let grid = ceil_div(num_experts, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                obs_buf.as_device_ptr(),
                adj_buf.as_device_ptr(),
                num_experts,
                base,
                min_cap,
                max_cap,
                tokens_per_batch,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut out = vec![0u32; num_experts as usize];
    adj_buf.copy_to_host(&mut out).expect("download");
    out
}

#[test]
fn dynamic_capacity_no_max_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let observed = [256u32, 128, 64, 0, 200, 130, 128, 512];
    let (base, min_cap, max_cap, tpb) = (160u32, 4u32, 0u32, 1024u32);
    let gpu = run_dynamic_capacity(&fx, &observed, base, min_cap, max_cap, tpb);
    let cpu = dynamic_capacity_oracle(
        &observed,
        base,
        min_cap,
        max_cap,
        tpb,
        observed.len() as u32,
    );
    assert_eq!(gpu, cpu, "dynamic_capacity (no max)");
}

#[test]
fn dynamic_capacity_with_max_clamp_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let observed = [256u32, 128, 64, 0, 200, 130, 128, 512];
    let (base, min_cap, max_cap, tpb) = (160u32, 8u32, 300u32, 1024u32);
    let gpu = run_dynamic_capacity(&fx, &observed, base, min_cap, max_cap, tpb);
    let cpu = dynamic_capacity_oracle(
        &observed,
        base,
        min_cap,
        max_cap,
        tpb,
        observed.len() as u32,
    );
    assert_eq!(gpu, cpu, "dynamic_capacity (max clamp)");
    // Sanity: at least one expert is clamped at the ceiling and one at the floor.
    assert!(gpu.contains(&max_cap), "expected a ceiling clamp");
    assert!(gpu.contains(&min_cap), "expected a floor clamp");
}

#[test]
fn overflow_mask_per_expert_count_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_experts: u32 = 4;
    let capacity: u32 = 3;
    // Designed per-expert counts: e0=5, e1=2, e2=3, e3=0 (interleaved order).
    let assignments: Vec<u32> = vec![0, 1, 2, 0, 2, 0, 1, 2, 0, 0];
    let num_tokens = assignments.len() as u32;

    let cfg = CapacityConfig {
        num_experts,
        capacity_factor: 1.25,
        min_capacity: 1,
        max_capacity: 0,
        tokens_per_batch: num_tokens,
        sm_version: fx.sm,
    };
    let plan = CapacityPlan::new(cfg).expect("plan");
    let ptx = plan.generate_overflow_mask_ptx().expect("ptx");
    let kernel = load_kernel(&ptx, "moe_overflow_mask");

    let assign_buf = DeviceBuffer::<u32>::from_host(&assignments).expect("assign");
    let counts_buf = DeviceBuffer::<u32>::zeroed(num_experts as usize).expect("counts");
    let mask_buf = DeviceBuffer::<u32>::from_host(&vec![9u32; num_tokens as usize]).expect("mask");

    let block = 256u32;
    let grid = ceil_div(num_tokens, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                assign_buf.as_device_ptr(),
                counts_buf.as_device_ptr(),
                mask_buf.as_device_ptr(),
                capacity,
                num_tokens,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu_counts = vec![0u32; num_experts as usize];
    counts_buf.copy_to_host(&mut gpu_counts).expect("counts dl");
    let mut gpu_mask = vec![0u32; num_tokens as usize];
    mask_buf.copy_to_host(&mut gpu_mask).expect("mask dl");

    // Exact histogram check.
    let mut cpu_counts = vec![0u32; num_experts as usize];
    for &e in &assignments {
        cpu_counts[e as usize] += 1;
    }
    assert_eq!(gpu_counts, cpu_counts, "overflow_mask expert counts");

    // The *which* token is masked is atomic-order dependent, but the per-expert
    // masked total is deterministic: max(0, count_e - capacity).
    let mut masked_per_expert = vec![0u32; num_experts as usize];
    for (tok, &m) in gpu_mask.iter().enumerate() {
        assert!(m == 0 || m == 1, "mask must be 0/1, got {m}");
        if m == 1 {
            masked_per_expert[assignments[tok] as usize] += 1;
        }
    }
    for e in 0..num_experts as usize {
        let expect = cpu_counts[e].saturating_sub(capacity);
        assert_eq!(
            masked_per_expert[e], expect,
            "expert {e}: masked count {} != max(0, {}-{})",
            masked_per_expert[e], cpu_counts[e], capacity
        );
    }
}

// ---------------------------------------------------------------------------
// aux_loss: load_balance_loss + z_loss
// ---------------------------------------------------------------------------

#[test]
fn load_balance_loss_full_warp_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Launch a full 32-lane warp so the 0xFFFFFFFF shfl membermask is sound.
    let num_experts: u32 = 32;
    let num_tokens: u32 = 256;
    let alpha: f32 = 0.01;
    let mut rng = Lcg::new(0xa5a5_1234);

    let counts: Vec<u32> = (0..num_experts).map(|_| 1 + rng.next_u32() % 16).collect();
    let probs: Vec<f32> = (0..num_experts).map(|_| rng.range_f32(0.05, 4.0)).collect();

    let cfg = AuxLossConfig {
        num_experts,
        num_tokens,
        alpha,
        sm_version: fx.sm,
    };
    let plan = AuxLossPlan::new(cfg).expect("plan");
    let ptx = plan.generate_load_balance_loss_ptx().expect("ptx");
    let kernel = load_kernel(&ptx, "moe_load_balance_loss");

    let counts_buf = DeviceBuffer::<u32>::from_host(&counts).expect("counts");
    let probs_buf = DeviceBuffer::<f32>::from_host(&probs).expect("probs");
    let loss_buf = DeviceBuffer::<f32>::zeroed(1).expect("loss");

    let params = LaunchParams::new(1u32, 32u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                counts_buf.as_device_ptr(),
                probs_buf.as_device_ptr(),
                loss_buf.as_device_ptr(),
                num_experts,
                num_tokens,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = [0.0f32];
    loss_buf.copy_to_host(&mut gpu).expect("dl");

    // loss = alpha * num_experts * sum_i (count_i/N) * (prob_i/N)
    let mut sum = 0.0f32;
    for i in 0..num_experts as usize {
        let f_i = counts[i] as f32 / num_tokens as f32;
        let p_i = probs[i] / num_tokens as f32;
        sum += f_i * p_i;
    }
    let cpu = alpha * num_experts as f32 * sum;
    assert_close_f32(&gpu, &[cpu], 2e-4, 1e-8, "load_balance_loss");
}

#[test]
fn z_loss_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_tokens: u32 = 64;
    let num_experts: u32 = 8;
    let mut rng = Lcg::new(0x0bad_c0de);

    let logits: Vec<f32> = (0..num_tokens * num_experts)
        .map(|_| rng.range_f32(-2.0, 2.0))
        .collect();

    let cfg = AuxLossConfig {
        num_experts,
        num_tokens,
        alpha: 0.01,
        sm_version: fx.sm,
    };
    let plan = AuxLossPlan::new(cfg).expect("plan");
    let ptx = plan.generate_z_loss_ptx().expect("ptx");
    let kernel = load_kernel(&ptx, "moe_z_loss");

    let logits_buf = DeviceBuffer::<f32>::from_host(&logits).expect("logits");
    let loss_buf = DeviceBuffer::<f32>::zeroed(1).expect("loss");

    let block = 256u32;
    let grid = ceil_div(num_tokens, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                logits_buf.as_device_ptr(),
                loss_buf.as_device_ptr(),
                num_tokens,
                num_experts,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = [0.0f32];
    loss_buf.copy_to_host(&mut gpu).expect("dl");

    // z_loss = (1/N) * sum_t logsumexp(logits[t, :])^2
    let mut acc = 0.0f32;
    for t in 0..num_tokens as usize {
        let row = &logits[t * num_experts as usize..(t + 1) * num_experts as usize];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&v| (v - max_v).exp()).sum();
        let lse = sum_exp.ln() + max_v;
        acc += lse * lse;
    }
    let cpu = acc / num_tokens as f32;
    assert_close_f32(&gpu, &[cpu], 3e-3, 1e-4, "z_loss");
}

// ---------------------------------------------------------------------------
// monitoring: utilization_count (byte-exact) + imbalance_score (full warp)
// ---------------------------------------------------------------------------

#[test]
fn utilization_count_histogram_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_experts: u32 = 8;
    let mut rng = Lcg::new(0xfeed_face);
    // 200 tokens with a couple of deliberately out-of-range ids (ignored).
    let mut assignments: Vec<u32> = (0..200).map(|_| rng.next_u32() % num_experts).collect();
    assignments[10] = num_experts; // OOB, must be skipped
    assignments[100] = num_experts + 5; // OOB, must be skipped
    let num_tokens = assignments.len() as u32;

    let monitor = MoeMonitor::new(num_experts, fx.sm).expect("monitor");
    let ptx = monitor.generate_utilization_ptx().expect("ptx");
    let kernel = load_kernel(&ptx, "moe_utilization_count");

    let assign_buf = DeviceBuffer::<u32>::from_host(&assignments).expect("assign");
    let counts_buf = DeviceBuffer::<u32>::zeroed(num_experts as usize).expect("counts");

    let block = 256u32;
    let grid = ceil_div(num_tokens, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                assign_buf.as_device_ptr(),
                counts_buf.as_device_ptr(),
                num_tokens,
                num_experts,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0u32; num_experts as usize];
    counts_buf.copy_to_host(&mut gpu).expect("dl");

    let mut cpu = vec![0u32; num_experts as usize];
    for &e in &assignments {
        if e < num_experts {
            cpu[e as usize] += 1;
        }
    }
    assert_eq!(gpu, cpu, "utilization_count histogram");
}

#[test]
fn imbalance_score_full_warp_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Single full 32-lane warp keeps the shfl + per-warp CV correct.
    let num_experts: u32 = 32;
    let mut rng = Lcg::new(0xc0ff_ee01);
    let counts: Vec<u32> = (0..num_experts).map(|_| 20 + rng.next_u32() % 40).collect();
    let total_tokens: u32 = counts.iter().sum();

    let monitor = MoeMonitor::new(num_experts, fx.sm).expect("monitor");
    let ptx = monitor.generate_imbalance_score_ptx().expect("ptx");
    let kernel = load_kernel(&ptx, "moe_imbalance_score");

    let counts_buf = DeviceBuffer::<u32>::from_host(&counts).expect("counts");
    let imb_buf = DeviceBuffer::<f32>::zeroed(1).expect("imb");

    let params = LaunchParams::new(1u32, 32u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                counts_buf.as_device_ptr(),
                imb_buf.as_device_ptr(),
                num_experts,
                total_tokens,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = [0.0f32];
    imb_buf.copy_to_host(&mut gpu).expect("dl");

    // Independent CPU coefficient of variation (the in-file reference).
    let cpu = MoeUtilizationReport::from_counts(&counts)
        .expect("report")
        .imbalance_score;
    assert_close_f32(&gpu, &[cpu], 2e-3, 1e-4, "imbalance_score");
}

// ---------------------------------------------------------------------------
// fused_moe: token-parallel end-to-end (numeric) + expand + activation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn fused_moe_token_parallel_oracle(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    indices: &[i32],
    weights: &[f32],
    num_tokens: usize,
    hidden: usize,
    inter: usize,
    top_k: usize,
    activation: Activation,
) -> Vec<f32> {
    let act = |x: f32| match activation {
        Activation::None => x,
        Activation::Relu => x.max(0.0),
        _ => unreachable!("oracle only used for None/Relu"),
    };
    let mut out = vec![0.0f32; num_tokens * hidden];
    for t in 0..num_tokens {
        for kk in 0..top_k {
            let e = indices[t * top_k + kk] as usize;
            let w = weights[t * top_k + kk];
            for j in 0..inter {
                let mut acc = 0.0f32;
                for i in 0..hidden {
                    acc += input[t * hidden + i] * w1[e * hidden * inter + i * inter + j];
                }
                let a = act(acc);
                for h in 0..hidden {
                    out[t * hidden + h] += w * a * w2[e * inter * hidden + j * hidden + h];
                }
            }
        }
    }
    out
}

fn run_fused_moe_token_parallel(fx: &GpuFixture, activation: Activation) {
    let num_experts: u32 = 4;
    let hidden: u32 = 6;
    let inter: u32 = 8;
    let top_k: u32 = 2;
    let num_tokens: u32 = 4; // 4 < 4*2 => TokenParallel path
    let mut rng = Lcg::new(0x5151_2727);

    let input: Vec<f32> = (0..num_tokens * hidden)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let w1: Vec<f32> = (0..num_experts * hidden * inter)
        .map(|_| rng.range_f32(-0.5, 0.5))
        .collect();
    let w2: Vec<f32> = (0..num_experts * inter * hidden)
        .map(|_| rng.range_f32(-0.5, 0.5))
        .collect();
    let indices: Vec<i32> = (0..num_tokens * top_k)
        .map(|_| (rng.next_u32() % num_experts) as i32)
        .collect();
    let weights: Vec<f32> = (0..num_tokens * top_k)
        .map(|_| rng.range_f32(0.2, 0.8))
        .collect();

    let in_buf = DeviceBuffer::<f32>::from_host(&input).expect("in");
    let w1_buf = DeviceBuffer::<f32>::from_host(&w1).expect("w1");
    let w2_buf = DeviceBuffer::<f32>::from_host(&w2).expect("w2");
    let idx_buf = DeviceBuffer::<i32>::from_host(&indices).expect("idx");
    let wt_buf = DeviceBuffer::<f32>::from_host(&weights).expect("wt");
    let out_buf =
        DeviceBuffer::<f32>::from_host(&vec![7.0f32; (num_tokens * hidden) as usize]).expect("out");

    let input_desc = desc_rowmajor(&in_buf, vec![num_tokens, hidden]);
    let w1_desc = desc_rowmajor(&w1_buf, vec![num_experts, hidden, inter]);
    let w2_desc = desc_rowmajor(&w2_buf, vec![num_experts, inter, hidden]);
    let mut out_desc = desc_rowmajor_mut(&out_buf, vec![num_tokens, hidden]);

    let cfg = MoeConfig {
        num_experts,
        top_k,
        hidden_dim: hidden,
        intermediate_dim: inter,
        activation,
        precision: PtxType::F32,
        sm_version: fx.sm,
    };

    crate::moe::fused_moe::fused_moe::<f32>(
        &fx.handle,
        &input_desc,
        &w1_desc,
        &w2_desc,
        &idx_buf,
        &wt_buf,
        &mut out_desc,
        &cfg,
    )
    .expect("fused_moe launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; (num_tokens * hidden) as usize];
    out_buf.copy_to_host(&mut gpu).expect("dl");

    let cpu = fused_moe_token_parallel_oracle(
        &input,
        &w1,
        &w2,
        &indices,
        &weights,
        num_tokens as usize,
        hidden as usize,
        inter as usize,
        top_k as usize,
        activation,
    );
    assert_close_f32(&gpu, &cpu, 2e-3, 2e-3, "fused_moe_token_parallel");
}

#[test]
fn fused_moe_token_parallel_identity_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_fused_moe_token_parallel(&fx, Activation::None);
}

#[test]
fn fused_moe_token_parallel_relu_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    run_fused_moe_token_parallel(&fx, Activation::Relu);
}

#[test]
fn expand_tokens_replicate_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_tokens: u32 = 5;
    let hidden: u32 = 7;
    let top_k: u32 = 3;
    let total = num_tokens * top_k;
    let mut rng = Lcg::new(0x2718_2818);

    let input: Vec<f32> = (0..num_tokens * hidden)
        .map(|_| rng.range_f32(-3.0, 3.0))
        .collect();

    let ptx = crate::moe::fused_moe::generate_expand_ptx::<f32>(fx.sm, top_k).expect("ptx");
    let kernel = load_kernel(&ptx, "moe_expand_tokens_f32");

    let in_buf = DeviceBuffer::<f32>::from_host(&input).expect("in");
    let out_buf =
        DeviceBuffer::<f32>::from_host(&vec![0.0f32; (total * hidden) as usize]).expect("out");

    let grid = Dim3::new(ceil_div(hidden, 256), total, 1);
    let block = Dim3::new(256, 1, 1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                in_buf.as_device_ptr(),
                out_buf.as_device_ptr(),
                num_tokens,
                hidden,
                top_k,
            ),
        )
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; (total * hidden) as usize];
    out_buf.copy_to_host(&mut gpu).expect("dl");

    let mut cpu = vec![0.0f32; (total * hidden) as usize];
    for slot in 0..total as usize {
        let src = slot / top_k as usize;
        for c in 0..hidden as usize {
            cpu[slot * hidden as usize + c] = input[src * hidden as usize + c];
        }
    }
    assert_close_f32(&gpu, &cpu, 0.0, 0.0, "expand_tokens");
}

/// Drives the element-wise activation kernel for `act` and returns (gpu, cpu).
fn run_activation_kernel(fx: &GpuFixture, act: Activation, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let n = input.len() as u32;
    let ptx = crate::moe::fused_moe::generate_activation_ptx::<f32>(act, fx.sm).expect("ptx");
    let kernel = load_kernel(&ptx, "moe_activation_f32");

    let data_buf = DeviceBuffer::<f32>::from_host(input).expect("data");
    let block = 256u32;
    let grid = ceil_div(n, block);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(&params, fx.stream(), &(data_buf.as_device_ptr(), n))
        .expect("launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; input.len()];
    data_buf.copy_to_host(&mut gpu).expect("dl");

    let cpu: Vec<f32> = input.iter().map(|&x| activation_oracle(act, x)).collect();
    (gpu, cpu)
}

/// CPU reference matching the kernel's activation math (tanh-approx GELU).
fn activation_oracle(act: Activation, x: f32) -> f32 {
    match act {
        Activation::None => x,
        Activation::Relu => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Silu => x / (1.0 + (-x).exp()),
        Activation::Tanh => x.tanh(),
        Activation::Gelu | Activation::GeluTanh => {
            // sqrt(2/pi) baked as f64 then narrowed, matching the kernel's
            // load_float_imm path (avoids an f32 excessive-precision literal).
            let sqrt_2_over_pi = 0.797_884_560_802_865_4_f64 as f32;
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }
    }
}

#[test]
fn activation_relu_exact_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x9090_3030);
    let input: Vec<f32> = (0..128).map(|_| rng.range_f32(-3.0, 3.0)).collect();
    let (gpu, cpu) = run_activation_kernel(&fx, Activation::Relu, &input);
    assert_close_f32(&gpu, &cpu, 0.0, 0.0, "activation_relu");
}

#[test]
fn activation_transcendental_approx_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let mut rng = Lcg::new(0x7c7c_1212);
    let input: Vec<f32> = (0..128).map(|_| rng.range_f32(-3.0, 3.0)).collect();
    // ex2.approx-based; meaningful tolerance (a base-2 scaling bug would be
    // off by an exp-factor, far outside 1%).
    for act in [
        Activation::Sigmoid,
        Activation::Silu,
        Activation::Tanh,
        Activation::Gelu,
    ] {
        let (gpu, cpu) = run_activation_kernel(&fx, act, &input);
        assert_close_f32(&gpu, &cpu, 1e-2, 1e-2, "activation_transcendental");
    }
}

// ---------------------------------------------------------------------------
// routing: top-1 argmax (numeric) + sort fault-free launch
// ---------------------------------------------------------------------------

#[test]
fn routing_top1_argmax_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_tokens: u32 = 8;
    let num_experts: u32 = 8;
    let top_k: u32 = 1;
    let total = (num_tokens * top_k) as usize;

    // Distinct logits per row so the argmax is unambiguous.
    let mut logits = vec![0.0f32; (num_tokens * num_experts) as usize];
    let mut rng = Lcg::new(0x4242_8888);
    for t in 0..num_tokens as usize {
        for e in 0..num_experts as usize {
            logits[t * num_experts as usize + e] =
                rng.range_f32(-5.0, 5.0) + (e as f32) * 1e-3 + (t as f32) * 1e-4;
        }
    }

    let logits_buf = DeviceBuffer::<f32>::from_host(&logits).expect("logits");
    let mut idx_buf = DeviceBuffer::<i32>::from_host(&vec![-1i32; total]).expect("idx");
    let mut wt_buf = DeviceBuffer::<f32>::from_host(&vec![0.0f32; total]).expect("wt");
    let mut perm_buf = DeviceBuffer::<i32>::from_host(&vec![-1i32; total]).expect("perm");
    let mut off_buf = DeviceBuffer::<i32>::zeroed(num_experts as usize + 1).expect("off");

    let router = desc_rowmajor(&logits_buf, vec![num_tokens, num_experts]);
    let cfg = MoeConfig {
        num_experts,
        top_k,
        hidden_dim: 16,
        intermediate_dim: 32,
        activation: Activation::Silu,
        precision: PtxType::F32,
        sm_version: fx.sm,
    };

    // Drives both moe_topk_softmax (argmax) and moe_sort_by_expert (launch).
    moe_routing::<f32>(
        &fx.handle,
        &router,
        &mut idx_buf,
        &mut wt_buf,
        &mut perm_buf,
        &mut off_buf,
        &cfg,
    )
    .expect("routing launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu_idx = vec![0i32; total];
    idx_buf.copy_to_host(&mut gpu_idx).expect("idx dl");
    let mut gpu_wt = vec![0.0f32; total];
    wt_buf.copy_to_host(&mut gpu_wt).expect("wt dl");

    for t in 0..num_tokens as usize {
        let row = &logits[t * num_experts as usize..(t + 1) * num_experts as usize];
        let mut argmax = 0usize;
        for e in 1..num_experts as usize {
            if row[e] > row[argmax] {
                argmax = e;
            }
        }
        assert_eq!(
            gpu_idx[t], argmax as i32,
            "token {t}: argmax mismatch (gpu {} cpu {})",
            gpu_idx[t], argmax
        );
        // Top-1 softmax weight degenerates to 1.0.
        assert_close_f32(&[gpu_wt[t]], &[1.0], 0.0, 1e-6, "routing weight");
    }
}

// ---------------------------------------------------------------------------
// linear: fused_linear is a comment-only skeleton — load/launch only
// ---------------------------------------------------------------------------

#[test]
fn fused_linear_skeleton_no_write_f32() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // generate_fused_linear_ptx loads its params then discards them and never
    // stores to output. We assert it assembles + launches fault-free and leaves
    // the (sentinel-filled) output untouched — an honest load/launch check that
    // documents the fragment without asserting a fabricated numeric result.
    let batch = 2usize;
    let in_features = 3usize;
    let out_features = 4usize;
    let mut rng = Lcg::new(0x1357_9bdf);

    let input: Vec<f32> = (0..batch * in_features)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let weight: Vec<f32> = (0..out_features * in_features)
        .map(|_| rng.range_f32(-1.0, 1.0))
        .collect();
    let sentinel = 12345.0f32;

    let in_buf = DeviceBuffer::<f32>::from_host(&input).expect("in");
    let w_buf = DeviceBuffer::<f32>::from_host(&weight).expect("w");
    let mut out_buf =
        DeviceBuffer::<f32>::from_host(&vec![sentinel; batch * out_features]).expect("out");

    let cfg = FusedLinearConfig::identity();
    fused_linear::<f32>(
        &fx.handle,
        &cfg,
        &in_buf,
        &w_buf,
        None,
        &mut out_buf,
        batch,
        in_features,
        out_features,
    )
    .expect("fused_linear launch");
    fx.stream().synchronize().expect("sync");

    let mut gpu = vec![0.0f32; batch * out_features];
    out_buf.copy_to_host(&mut gpu).expect("dl");
    // Skeleton writes nothing: the sentinel survives.
    for (i, &v) in gpu.iter().enumerate() {
        assert_eq!(
            v, sentinel,
            "fused_linear skeleton must not write output[{i}] (got {v})"
        );
    }
}
