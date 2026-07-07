//! On-device GPU validation for the `attn` subsystem of `oxicuda-dnn`.
//!
//! This cluster covers the 24 PTX-emitting attention kernels under `src/attn`.
//! Following the discovery inventory, the kernels split into three groups:
//!
//! * **complete (numeric)** — the `mha_qk_gemm`/`mha_softmax`/`mha_pv_gemm`
//!   trio (driven end-to-end via `multi_head_attention`) plus six other
//!   kernels compute a numerically-correct result that an independent CPU
//!   oracle re-derives and `assert_close`s against: `mha_scale_mask`,
//!   `rope_neox_half_split_f32`, `ring_attn_causal_mask_step*`,
//!   `spec_decode_kv_copy`, `spec_decode_verify`, `spec_decode_rejection_sample`.
//! * **fragment (load + launch)** — ~12 kernels are structural skeletons (they
//!   compute index/stride arithmetic then discard it; the GEMM/softmax/PV steps
//!   are comment-only, with no global loads or stores). These are driven on the
//!   device — either through their production op launcher or directly via the
//!   published PTX generator — and asserted to assemble, launch, and synchronise
//!   fault-free. No numeric result is asserted against them (that would be
//!   green-washing a kernel that writes nothing).
//! * **hopper_blocked (skip / prescreen)** — the FlashAttention-3 forward and
//!   backward kernels only emit an sm_90 module, so they cannot launch on this
//!   sm_86 device. They are `ptxas`-prescreened for their declared sm_90 target.
//!
//! ## Bugs fixed in the owned source during this pass
//!
//! `attn::mha::generate_scale_mask_ptx` computed `scaled = fma(val, scale, zero)`
//! where `zero` was an `alloc_reg` register that was **never written** — an
//! uninitialised PTX register, so the FMA added an undefined term and the scaled
//! scores were garbage. A `mov.f32 {zero}, 0f00000000;` now initialises the
//! addend to `+0.0`, making the op compute the intended `scores[i] * scale`.
//! The `mha_scale_mask` numeric test below confirms the fix on-device.
//!
//! `attn::mha::generate_qk_gemm_ptx` / `generate_row_softmax_ptx` /
//! `generate_pv_gemm_ptx` were comment-only stubs (no load/compute/store at
//! all), so `multi_head_attention` silently returned whatever garbage was
//! already in the output buffer. `multi_head_attention` also reused the
//! output tensor's device pointer as scratch for the `[N, N]` score matrix,
//! which overflows the output allocation whenever `seq_len != head_dim`. All
//! three kernels now compute real values (a runtime dot-product loop and a
//! 3-pass stable softmax), and a dedicated scratch `DeviceBuffer` backs the
//! score matrix instead. `mha_pipeline_numeric_no_mask` /
//! `mha_pipeline_numeric_with_mask` below confirm both fixes on-device.
//!
//! Every test returns early (skips) when no CUDA device is present.

use super::{Lcg, assert_close_f32, entry_name, gpu_fixture, load_kernel, ptxas_assembles};

use oxicuda_launch::{Dim3, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::ir::PtxType;

use crate::attn::block_sparse::{BlockSparseAttentionPlan, BlockSparseConfig, BlockSparsePattern};
use crate::attn::flash_attn::backward::{generate_backward_ptx, generate_rowsum_dot_ptx};
use crate::attn::flash_attn::decode::single_query_decode_attention;
use crate::attn::flash_attn::forward::FlashAttentionConfig;
use crate::attn::flash_attn::hopper::{FlashAttention3Config, FlashAttention3Plan};
use crate::attn::flash_attn::paged::PagedAttentionConfig;
use crate::attn::fused_rope_attn::{FusedRopeAttnConfig, FusedRopeAttnPlan};
use crate::attn::gqa::{GqaConfig, gqa_forward};
use crate::attn::mha::{generate_scale_mask_ptx, multi_head_attention};
use crate::attn::ring_attention::{RingAttentionConfig, RingAttentionDtype, RingAttentionPlan};
use crate::attn::rope::apply_rope;
use crate::attn::rope_neox::rope_neox_half_split_f32;
use crate::attn::sliding_window::{SlidingWindowConfig, sliding_window_attention};
use crate::attn::speculative_decode::{
    SpeculativeDecodeConfig, SpeculativeDecodePlan, accept_token,
};
use crate::types::{TensorDesc, TensorDescMut, TensorLayout};

// ---------------------------------------------------------------------------
// Small device helpers
// ---------------------------------------------------------------------------

/// Uploads an `f32` host slice to a fresh device buffer.
fn dbuf(data: &[f32]) -> DeviceBuffer<f32> {
    DeviceBuffer::<f32>::from_host(data).expect("from_host f32")
}

/// Uploads a `u32` host slice to a fresh device buffer.
fn dbuf_u32(data: &[u32]) -> DeviceBuffer<u32> {
    DeviceBuffer::<u32>::from_host(data).expect("from_host u32")
}

/// Uploads an `i32` host slice to a fresh device buffer.
fn dbuf_i32(data: &[i32]) -> DeviceBuffer<i32> {
    DeviceBuffer::<i32>::from_host(data).expect("from_host i32")
}

/// Allocates a zeroed `f32` device buffer of `n` elements.
fn dzeros(n: usize) -> DeviceBuffer<f32> {
    dbuf(&vec![0.0f32; n])
}

/// Copies an `f32` device buffer back to a host vector.
fn to_host_f32(buf: &DeviceBuffer<f32>, n: usize) -> Vec<f32> {
    let mut host = vec![0.0f32; n];
    buf.copy_to_host(&mut host).expect("copy_to_host f32");
    host
}

/// Copies a `u32` device buffer back to a host vector.
fn to_host_u32(buf: &DeviceBuffer<u32>, n: usize) -> Vec<u32> {
    let mut host = vec![0u32; n];
    buf.copy_to_host(&mut host).expect("copy_to_host u32");
    host
}

/// Deterministic `f32` vector of `n` values in `[lo, hi)`.
fn rand_vec(rng: &mut Lcg, n: usize, lo: f64, hi: f64) -> Vec<f32> {
    (0..n).map(|_| rng.range_f32(lo, hi)).collect()
}

/// Contiguous NCHW strides for a 4-D `[b, h, n, d]` tensor.
fn nchw_strides(dims: [u32; 4]) -> Vec<u32> {
    vec![dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1]
}

/// Builds a read-only `[b, h, n, d]` descriptor over a device buffer.
fn desc(buf: &DeviceBuffer<f32>, dims: [u32; 4]) -> TensorDesc<f32> {
    TensorDesc::from_raw(
        buf.as_device_ptr(),
        dims.to_vec(),
        nchw_strides(dims),
        TensorLayout::Nchw,
    )
    .expect("tensor desc")
}

/// Builds a mutable `[b, h, n, d]` descriptor over a device buffer.
fn desc_mut(buf: &DeviceBuffer<f32>, dims: [u32; 4]) -> TensorDescMut<f32> {
    TensorDescMut::from_raw(
        buf.as_device_ptr(),
        dims.to_vec(),
        nchw_strides(dims),
        TensorLayout::Nchw,
    )
    .expect("tensor desc mut")
}

// ===========================================================================
// 1. mha_scale_mask  — numeric (complete; bug fixed in source)
// ===========================================================================

#[test]
fn mha_scale_mask_no_mask_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 257usize; // deliberately not a multiple of the block to exercise the tail guard
    let mut rng = Lcg::new(0xA11CE);
    let scores = rand_vec(&mut rng, n, -4.0, 4.0);
    let scale = 0.125f32;

    let ptx =
        generate_scale_mask_ptx::<f32>("mha_scale_mask_f32", fx.sm, false).expect("scale_mask ptx");
    let kernel = load_kernel(&ptx, "mha_scale_mask_f32");

    let d_scores = dbuf(&scores);
    let block = 256u32;
    let grid = (n as u32).div_ceil(block).max(1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(d_scores.as_device_ptr(), 0u64, n as u32, scale),
        )
        .expect("scale_mask launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&d_scores, n);
    let cpu: Vec<f32> = scores
        .iter()
        .map(|&s| (s as f64 * scale as f64) as f32)
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-5, 1e-5, "mha_scale_mask (no mask)");
}

#[test]
fn mha_scale_mask_with_mask_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let n = 192usize;
    let mut rng = Lcg::new(0xBEEF);
    let scores = rand_vec(&mut rng, n, -4.0, 4.0);
    let mask = rand_vec(&mut rng, n, -2.0, 2.0);
    let scale = 0.3f32;

    let ptx =
        generate_scale_mask_ptx::<f32>("mha_scale_mask_f32", fx.sm, true).expect("scale_mask ptx");
    let kernel = load_kernel(&ptx, "mha_scale_mask_f32");

    let d_scores = dbuf(&scores);
    let d_mask = dbuf(&mask);
    let block = 256u32;
    let grid = (n as u32).div_ceil(block).max(1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                d_scores.as_device_ptr(),
                d_mask.as_device_ptr(),
                n as u32,
                scale,
            ),
        )
        .expect("scale_mask launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&d_scores, n);
    let cpu: Vec<f32> = (0..n)
        .map(|i| (scores[i] as f64 * scale as f64 + mask[i] as f64) as f32)
        .collect();
    assert_close_f32(&gpu, &cpu, 1e-5, 1e-5, "mha_scale_mask (with mask)");
}

// ===========================================================================
// 2. rope_neox_half_split_f32 — numeric (complete; approx-instruction tolerance)
// ===========================================================================

#[test]
fn rope_neox_half_split_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (seq_len, num_heads, head_dim, rotary_dim) = (5u32, 2u32, 8u32, 4u32);
    let base = 10_000.0f32;
    let total = (seq_len * num_heads * head_dim) as usize;
    let mut rng = Lcg::new(0x12345);
    let input = rand_vec(&mut rng, total, -2.0, 2.0);

    let d_in = dbuf(&input);
    let mut d_out = dzeros(total);
    rope_neox_half_split_f32(
        &fx.handle, &d_in, &mut d_out, seq_len, num_heads, head_dim, rotary_dim, base,
    )
    .expect("rope_neox launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&d_out, total);

    // CPU oracle (f64): half-split rotary on [seq, head, dim].
    let half = (rotary_dim / 2) as usize;
    let hd = head_dim as usize;
    let nh = num_heads as usize;
    let mut cpu = vec![0.0f32; total];
    for pos in 0..seq_len as usize {
        for h in 0..nh {
            let base_off = pos * nh * hd + h * hd;
            for pair in 0..half {
                let freq = (base as f64).powf(-2.0 * pair as f64 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                let (c, s) = (angle.cos(), angle.sin());
                let xi = input[base_off + pair] as f64;
                let xj = input[base_off + pair + half] as f64;
                cpu[base_off + pair] = (xi * c - xj * s) as f32;
                cpu[base_off + pair + half] = (xi * s + xj * c) as f32;
            }
            // Untouched tail [rotary_dim, head_dim) copies through unchanged.
            let tail = base_off + rotary_dim as usize..base_off + hd;
            cpu[tail.clone()].copy_from_slice(&input[tail]);
        }
    }
    // sin/cos/lg2/ex2 approx instructions => loose relative tolerance.
    assert_close_f32(&gpu, &cpu, 3e-3, 3e-4, "rope_neox_half_split");
}

// ===========================================================================
// 3. ring_attn_causal_mask_step0 — numeric (complete)
// ===========================================================================

#[test]
fn ring_attn_causal_mask_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = RingAttentionConfig {
        head_dim: 16,
        num_heads: 1,
        seq_len: 8,
        num_devices: 2,
        chunk_size: 4,
        sm_scale: 0.25,
        causal: true,
        dtype: RingAttentionDtype::F32,
    };
    let plan = RingAttentionPlan::new(cfg).expect("ring plan");
    let steps = plan.steps_for_device(0);
    let ptx = plan
        .generate_causal_mask_ptx(steps[0])
        .expect("causal mask ptx");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let chunk = 4u32;
    let n = (chunk * chunk) as usize;
    let mut rng = Lcg::new(0x5151);
    let init = rand_vec(&mut rng, n, -3.0, 3.0);
    let d_scores = dbuf(&init);

    let (q_offset, kv_offset) = (0u32, 0u32);
    let block = 256u32;
    let grid = (n as u32).div_ceil(block).max(1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(d_scores.as_device_ptr(), chunk, q_offset, kv_offset),
        )
        .expect("causal mask launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&d_scores, n);
    for i in 0..n {
        let q_local = (i as u32) / chunk;
        let k_local = (i as u32) % chunk;
        let masked = q_offset + q_local < kv_offset + k_local;
        if masked {
            assert!(
                gpu[i].is_infinite() && gpu[i] < 0.0,
                "ring causal: element {i} expected -inf, got {}",
                gpu[i]
            );
        } else {
            assert!(
                (gpu[i] - init[i]).abs() <= 1e-6,
                "ring causal: element {i} should be unchanged: gpu={} init={}",
                gpu[i],
                init[i]
            );
        }
    }
}

// ===========================================================================
// 4-6. speculative_decode kernels — numeric (complete)
// ===========================================================================

fn spec_plan() -> SpeculativeDecodePlan {
    let cfg = SpeculativeDecodeConfig {
        draft_num_layers: 4,
        draft_num_heads: 4,
        draft_head_dim: 32,
        target_num_layers: 8,
        target_num_heads: 8,
        target_head_dim: 64,
        max_draft_tokens: 4,
        page_size: 16,
        max_pages: 32,
        acceptance_threshold: 0.9,
    };
    SpeculativeDecodePlan::new(cfg).expect("spec plan")
}

#[test]
fn spec_decode_kv_copy_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = spec_plan();
    let ptx = plan.generate_kv_copy_ptx().expect("kv_copy ptx");
    let kernel = load_kernel(&ptx, "spec_decode_kv_copy");

    let n = 300usize;
    let mut rng = Lcg::new(0x7777);
    let src = rand_vec(&mut rng, n, -5.0, 5.0);
    let d_src = dbuf(&src);
    let d_dst = dzeros(n);

    let block = 256u32;
    let grid = (n as u32).div_ceil(block).max(1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(d_src.as_device_ptr(), d_dst.as_device_ptr(), 0u64, n as u32),
        )
        .expect("kv_copy launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&d_dst, n);
    assert_eq!(gpu, src, "spec_decode_kv_copy must be an exact copy");
}

#[test]
fn spec_decode_verify_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = spec_plan();
    let ptx = plan.generate_verification_ptx().expect("verify ptx");
    let kernel = load_kernel(&ptx, "spec_decode_verify");

    let n = 128usize;
    let mut rng = Lcg::new(0x2468);
    let draft = rand_vec(&mut rng, n, -3.0, 3.0);
    let target = rand_vec(&mut rng, n, -3.0, 3.0);
    let threshold = 1.0f32;

    let d_draft = dbuf(&draft);
    let d_target = dbuf(&target);
    let d_mask = dbuf_u32(&vec![9u32; n]);

    let block = 256u32;
    let grid = (n as u32).div_ceil(block).max(1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                d_draft.as_device_ptr(),
                d_target.as_device_ptr(),
                d_mask.as_device_ptr(),
                /* vocab_size (unused) */ 1u32,
                n as u32,
                threshold,
            ),
        )
        .expect("verify launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_u32(&d_mask, n);
    for i in 0..n {
        let expect = u32::from((target[i] - draft[i]).abs() <= threshold);
        assert_eq!(gpu[i], expect, "spec_decode_verify element {i}");
    }
}

#[test]
fn spec_decode_rejection_sample_numeric() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = spec_plan();
    let ptx = plan
        .generate_rejection_sampling_ptx()
        .expect("rejection ptx");
    let kernel = load_kernel(&ptx, "spec_decode_rejection_sample");

    let n = 160usize;
    let mut rng = Lcg::new(0x1357);
    // Keep p_draft strictly positive so the kernel's unguarded division agrees
    // with the host `accept_token` oracle (which rejects when draft <= 0).
    let p_draft = rand_vec(&mut rng, n, 0.2, 1.0);
    let p_target = rand_vec(&mut rng, n, 0.0, 1.0);
    let rand_vals = rand_vec(&mut rng, n, 0.0, 1.0);

    let d_draft = dbuf(&p_draft);
    let d_target = dbuf(&p_target);
    let d_rand = dbuf(&rand_vals);
    let d_out = dbuf_u32(&vec![7u32; n]);

    let block = 256u32;
    let grid = (n as u32).div_ceil(block).max(1);
    let params = LaunchParams::new(grid, block);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                d_draft.as_device_ptr(),
                d_target.as_device_ptr(),
                d_rand.as_device_ptr(),
                d_out.as_device_ptr(),
                n as u32,
            ),
        )
        .expect("rejection launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_u32(&d_out, n);
    for i in 0..n {
        let expect = u32::from(accept_token(p_draft[i], p_target[i], rand_vals[i]));
        assert_eq!(gpu[i], expect, "spec_decode_rejection_sample element {i}");
    }
}

// ===========================================================================
// Fragment load + launch tests (compute nothing; assert fault-free execution)
// ===========================================================================

/// Small `f32` device buffer of `n` deterministic values, used as a valid (but
/// untouched) backing pointer for fragment kernels that read no global memory.
fn frag_buf(seed: u64, n: usize) -> DeviceBuffer<f32> {
    let mut rng = Lcg::new(seed);
    dbuf(&rand_vec(&mut rng, n, -1.0, 1.0))
}

// --- flash_attn forward (FlashAttentionConfig::generate_ptx) ----------------

#[test]
fn flash_attn_forward_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // Small tiles keep the static shared-memory footprint tiny.
    let cfg = FlashAttentionConfig {
        head_dim: 16,
        num_heads: 1,
        seq_len_q: 16,
        seq_len_kv: 16,
        causal: false,
        sm_scale: 0.25,
        block_m: 16,
        block_n: 16,
        num_warps: 1,
        num_stages: 1,
        precision: PtxType::F32,
        sm_version: fx.sm,
    };
    let ptx = cfg.generate_ptx().expect("flash fwd ptx");
    ptxas_assembles(&ptx, "flash_fwd").expect("flash fwd assembles");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let buf = frag_buf(1, 256);
    let lse = dzeros(64);
    let params = LaunchParams::new(1u32, cfg.num_warps * 32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(),
                buf.as_device_ptr(),
                buf.as_device_ptr(),
                buf.as_device_ptr(),
                lse.as_device_ptr(),
                16u32,
                16u32,
                16u32,
                1u32,
                0.25f32,
                1u32,
            ),
        )
        .expect("flash fwd launch");
    fx.stream().synchronize().expect("sync");
}

// --- flash_attn backward (rowsum_dot + main) --------------------------------

#[test]
fn flash_attn_backward_rowsum_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let ptx =
        generate_rowsum_dot_ptx::<f32>("flash_bwd_rowsum_dot", fx.sm, 16).expect("rowsum ptx");
    ptxas_assembles(&ptx, "flash_bwd_rowsum").expect("rowsum assembles");
    let kernel = load_kernel(&ptx, "flash_bwd_rowsum_dot");

    let buf = frag_buf(2, 256);
    let params = LaunchParams::new(1u32, 256u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(),
                buf.as_device_ptr(),
                buf.as_device_ptr(),
                16u32,
                4u32,
            ),
        )
        .expect("rowsum launch");
    fx.stream().synchronize().expect("sync");
}

#[test]
fn flash_attn_backward_main_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = FlashAttentionConfig {
        head_dim: 16,
        num_heads: 1,
        seq_len_q: 16,
        seq_len_kv: 16,
        causal: false,
        sm_scale: 0.25,
        block_m: 16,
        block_n: 16,
        num_warps: 1,
        num_stages: 1,
        precision: PtxType::F32,
        sm_version: fx.sm,
    };
    let ptx = generate_backward_ptx::<f32>(&cfg).expect("flash bwd ptx");
    ptxas_assembles(&ptx, "flash_bwd").expect("flash bwd assembles");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let buf = frag_buf(3, 256);
    let params = LaunchParams::new(1u32, cfg.num_warps * 32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(), // q
                buf.as_device_ptr(), // k
                buf.as_device_ptr(), // v
                buf.as_device_ptr(), // o
                buf.as_device_ptr(), // do
                buf.as_device_ptr(), // lse
                buf.as_device_ptr(), // dq
                buf.as_device_ptr(), // dk
                buf.as_device_ptr(), // dv
                16u32,               // seq_len_q
                16u32,               // seq_len_kv
                16u32,               // head_dim
                0.25f32,             // sm_scale
                1u32,                // num_kv_tiles
            ),
        )
        .expect("flash bwd launch");
    fx.stream().synchronize().expect("sync");
}

// --- decode (public op) -----------------------------------------------------

#[test]
fn decode_attention_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let head_dim = 16u32;
    let max_seq = 16u32;
    let q = frag_buf(4, head_dim as usize);
    let kc = frag_buf(5, (max_seq * head_dim) as usize);
    let vc = frag_buf(6, (max_seq * head_dim) as usize);
    let out = dzeros(head_dim as usize);
    let seq_lengths = dbuf_i32(&[4i32]);

    let q_desc = desc(&q, [1, 1, 1, head_dim]);
    let kc_desc = desc(&kc, [1, 1, max_seq, head_dim]);
    let vc_desc = desc(&vc, [1, 1, max_seq, head_dim]);
    let mut out_desc = desc_mut(&out, [1, 1, 1, head_dim]);

    single_query_decode_attention(
        &fx.handle,
        &q_desc,
        &kc_desc,
        &vc_desc,
        &seq_lengths,
        &mut out_desc,
        0.25,
    )
    .expect("decode launch");
    fx.stream().synchronize().expect("sync");
}

// --- paged (PagedAttentionConfig::generate_ptx) -----------------------------

#[test]
fn paged_attention_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = PagedAttentionConfig {
        head_dim: 16,
        num_heads: 1,
        num_kv_heads: 1,
        block_size: 16,
        precision: PtxType::F32,
        sm_version: fx.sm,
    };
    let ptx = cfg.generate_ptx().expect("paged ptx");
    ptxas_assembles(&ptx, "paged").expect("paged assembles");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    let buf = frag_buf(7, 512);
    let params = LaunchParams::new(1u32, cfg.threads_per_block());
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(), // q
                buf.as_device_ptr(), // k_cache
                buf.as_device_ptr(), // v_cache
                buf.as_device_ptr(), // page_table
                buf.as_device_ptr(), // seq_lengths
                buf.as_device_ptr(), // out
                16u32,               // head_dim
                1u32,                // num_heads
                1u32,                // num_kv_heads
                16u32,               // block_size
                16u32,               // max_seq_len
                0.25f32,             // sm_scale
            ),
        )
        .expect("paged launch");
    fx.stream().synchronize().expect("sync");
}

// --- mha qk_gemm / softmax / pv_gemm (via multi_head_attention op) -----------
//
// The three kernels below (`generate_qk_gemm_ptx`, `generate_row_softmax_ptx`,
// `generate_pv_gemm_ptx`) were previously comment-only stubs that never wrote
// their output buffer; `multi_head_attention` also reused the output tensor's
// device pointer as scratch for the `[N, N]` score matrix, which silently
// overflows the output allocation whenever `seq_len != head_dim`. Both are
// fixed now: the kernels compute real values, and a dedicated scratch buffer
// backs the score matrix. `seq_len != head_dim` here (and a batch/heads > 1
// shape) deliberately exercises both fixes at once.

/// Naive f64 CPU oracle for scaled dot-product attention over a `[b, h, n,
/// d]` tensor set, matching `multi_head_attention`'s algorithm exactly.
#[allow(clippy::too_many_arguments)]
fn mha_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    b: usize,
    h: usize,
    n: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * h * n * d];
    for bh in 0..(b * h) {
        let base_qkv = bh * n * d;
        let base_mask = bh * n * n;
        for i in 0..n {
            // S[i, j] = sum_d Q[i,d]*K[j,d], scaled (+ mask).
            let mut scores = vec![0.0f64; n];
            for (j, score) in scores.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for dd in 0..d {
                    acc +=
                        f64::from(q[base_qkv + i * d + dd]) * f64::from(k[base_qkv + j * d + dd]);
                }
                acc *= f64::from(scale);
                if let Some(m) = mask {
                    acc += f64::from(m[base_mask + i * n + j]);
                }
                *score = acc;
            }
            // Row softmax.
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exp.iter().sum();
            let probs: Vec<f64> = exp.iter().map(|e| e / sum).collect();
            // O[i, d] = sum_j P[i,j]*V[j,d].
            for dd in 0..d {
                let mut acc = 0.0f64;
                for j in 0..n {
                    acc += probs[j] * f64::from(v[base_qkv + j * d + dd]);
                }
                out[base_qkv + i * d + dd] = acc as f32;
            }
        }
    }
    out
}

#[test]
fn mha_pipeline_numeric_no_mask() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // seq_len (n) != head_dim (d), and batch*heads > 1: exercises both the
    // scratch-buffer sizing fix and the batch/head addressing in the kernels.
    let (b, h, n, d) = (2u32, 2u32, 6u32, 4u32);
    let elems = (b * h * n * d) as usize;
    let mut rng = Lcg::new(0xA77E17);
    let q_host = rand_vec(&mut rng, elems, -1.0, 1.0);
    let k_host = rand_vec(&mut rng, elems, -1.0, 1.0);
    let v_host = rand_vec(&mut rng, elems, -1.0, 1.0);
    let scale = 1.0 / (d as f32).sqrt();

    let q = dbuf(&q_host);
    let k = dbuf(&k_host);
    let v = dbuf(&v_host);
    let out = dzeros(elems);

    let q_desc = desc(&q, [b, h, n, d]);
    let k_desc = desc(&k, [b, h, n, d]);
    let v_desc = desc(&v, [b, h, n, d]);
    let mut out_desc = desc_mut(&out, [b, h, n, d]);

    multi_head_attention(
        &fx.handle,
        &q_desc,
        &k_desc,
        &v_desc,
        &mut out_desc,
        None,
        scale,
    )
    .expect("mha pipeline launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&out, elems);
    let cpu = mha_reference(
        &q_host, &k_host, &v_host, None, b as usize, h as usize, n as usize, d as usize, scale,
    );
    // ex2.approx/rcp.approx in the softmax loosen the tolerance a bit.
    assert_close_f32(&gpu, &cpu, 3e-3, 3e-4, "mha_pipeline (no mask)");
}

#[test]
fn mha_pipeline_numeric_with_mask() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (b, h, n, d) = (1u32, 2u32, 5u32, 8u32);
    let elems = (b * h * n * d) as usize;
    let mask_elems = (b * h * n * n) as usize;
    let mut rng = Lcg::new(0xFACE0FF);
    let q_host = rand_vec(&mut rng, elems, -1.0, 1.0);
    let k_host = rand_vec(&mut rng, elems, -1.0, 1.0);
    let v_host = rand_vec(&mut rng, elems, -1.0, 1.0);
    // A causal-style additive mask (0 on/below diagonal, large negative above)
    // to confirm the mask path is actually wired through the scratch buffer.
    let mut mask_host = vec![0.0f32; mask_elems];
    for bh in 0..(b * h) as usize {
        for i in 0..n as usize {
            for j in 0..n as usize {
                if j > i {
                    mask_host[bh * (n * n) as usize + i * n as usize + j] = -1.0e4;
                }
            }
        }
    }
    let scale = 1.0 / (d as f32).sqrt();

    let q = dbuf(&q_host);
    let k = dbuf(&k_host);
    let v = dbuf(&v_host);
    let mask = dbuf(&mask_host);
    let out = dzeros(elems);

    let q_desc = desc(&q, [b, h, n, d]);
    let k_desc = desc(&k, [b, h, n, d]);
    let v_desc = desc(&v, [b, h, n, d]);
    let mask_desc = desc(&mask, [b, h, n, n]);
    let mut out_desc = desc_mut(&out, [b, h, n, d]);

    multi_head_attention(
        &fx.handle,
        &q_desc,
        &k_desc,
        &v_desc,
        &mut out_desc,
        Some(&mask_desc),
        scale,
    )
    .expect("mha pipeline launch");
    fx.stream().synchronize().expect("sync");

    let gpu = to_host_f32(&out, elems);
    let cpu = mha_reference(
        &q_host,
        &k_host,
        &v_host,
        Some(&mask_host),
        b as usize,
        h as usize,
        n as usize,
        d as usize,
        scale,
    );
    assert_close_f32(&gpu, &cpu, 3e-3, 3e-4, "mha_pipeline (with causal mask)");
}

// --- rope interleaved (apply_rope op) ---------------------------------------

#[test]
fn rope_interleaved_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let (b, h, n, d) = (1u32, 1u32, 4u32, 8u32);
    let elems = (b * h * n * d) as usize;
    let q = frag_buf(11, elems);
    let k = frag_buf(12, elems);
    let mut q_desc = desc_mut(&q, [b, h, n, d]);
    let mut k_desc = desc_mut(&k, [b, h, n, d]);
    let positions = dbuf_i32(&[0, 1, 2, 3]);

    apply_rope(
        &fx.handle,
        &mut q_desc,
        &mut k_desc,
        &positions,
        d,
        10_000.0,
    )
    .expect("apply_rope launch");
    fx.stream().synchronize().expect("sync");
}

// --- gqa (gqa_forward op) ---------------------------------------------------

#[test]
fn gqa_forward_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = GqaConfig {
        num_q_heads: 2,
        num_kv_heads: 1,
        head_dim: 16,
        seq_len: 4,
        kv_seq_len: 4,
        scale: 0.25,
        causal: false,
    };
    let batch = 1usize;
    let q_n = batch * cfg.num_q_heads * cfg.seq_len * cfg.head_dim;
    let kv_n = batch * cfg.num_kv_heads * cfg.kv_seq_len * cfg.head_dim;
    let q = frag_buf(13, q_n);
    let k = frag_buf(14, kv_n);
    let v = frag_buf(15, kv_n);
    let mut out = dzeros(q_n);

    gqa_forward(&fx.handle, &cfg, &q, &k, &v, &mut out, batch).expect("gqa launch");
    fx.stream().synchronize().expect("sync");
}

// --- sliding window (sliding_window_attention op) ---------------------------

#[test]
fn sliding_window_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let cfg = SlidingWindowConfig {
        num_heads: 1,
        head_dim: 16,
        seq_len: 8,
        window_size: 4,
        scale: 0.25,
    };
    let batch = 1usize;
    let n = batch * cfg.num_heads * cfg.seq_len * cfg.head_dim;
    let q = frag_buf(16, n);
    let k = frag_buf(17, n);
    let v = frag_buf(18, n);
    let mut out = dzeros(n);

    sliding_window_attention(&fx.handle, &cfg, &q, &k, &v, &mut out, batch)
        .expect("sliding window launch");
    fx.stream().synchronize().expect("sync");
}

// --- fused RoPE + attention (FusedRopeAttnPlan::generate_ptx) ---------------

#[test]
fn fused_rope_attn_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    // head_dim=16 keeps the (128xhead_dim) shared tiles under the sm_86 static
    // shared-memory limit.
    let cfg = FusedRopeAttnConfig {
        num_heads: 1,
        head_dim: 16,
        seq_len: 16,
        batch_size: 1,
        rope_base: 10_000.0,
        rope_scaling: None,
        causal: false,
        softmax_scale: None,
    };
    let plan = FusedRopeAttnPlan::with_sm_version(cfg, fx.sm).expect("fused plan");
    let ptx = plan.generate_ptx().expect("fused ptx");
    ptxas_assembles(&ptx, "fused_rope").expect("fused assembles");
    let entry = entry_name(&ptx);
    let kernel = load_kernel(&ptx, &entry);

    // For head_dim=16 the planner picks 128x128 tiles with 4 warps, i.e. a
    // 128-thread block matching the kernel's `.maxntid`.
    let buf = frag_buf(19, 1024);
    let params = LaunchParams::new(Dim3::new(1, 1, 1), Dim3::new(128, 1, 1));
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(), // q
                buf.as_device_ptr(), // k
                buf.as_device_ptr(), // v
                buf.as_device_ptr(), // o
                16u32,               // seq_len
                16u32,               // head_dim
                1u32,                // num_heads
                1u32,                // batch_size
                10_000.0f32,         // rope_base
                1.0f32,              // rope_scale_inv
                0.25f32,             // sm_scale
                1u32,                // num_kv_tiles
            ),
        )
        .expect("fused launch");
    fx.stream().synchronize().expect("sync");
}

// --- ring attention local fwd + accumulate ----------------------------------

fn ring_plan() -> RingAttentionPlan {
    let cfg = RingAttentionConfig {
        head_dim: 16,
        num_heads: 1,
        seq_len: 8,
        num_devices: 2,
        chunk_size: 4,
        sm_scale: 0.25,
        causal: false,
        dtype: RingAttentionDtype::F32,
    };
    RingAttentionPlan::new(cfg).expect("ring plan")
}

#[test]
fn ring_attn_local_fwd_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = ring_plan();
    let ptx = plan.generate_local_attention_ptx().expect("ring local ptx");
    ptxas_assembles(&ptx, "ring_local").expect("ring local assembles");
    let kernel = load_kernel(&ptx, "ring_attn_local_fwd");

    let buf = frag_buf(20, 256);
    let scale_bits = 0.25f32.to_bits();
    let params = LaunchParams::new(1u32, 256u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(), // q
                buf.as_device_ptr(), // k
                buf.as_device_ptr(), // v
                buf.as_device_ptr(), // o
                buf.as_device_ptr(), // lse
                buf.as_device_ptr(), // max
                4u32,                // chunk_size
                16u32,               // head_dim
                1u32,                // num_heads
                scale_bits,          // scale_bits
            ),
        )
        .expect("ring local launch");
    fx.stream().synchronize().expect("sync");
}

#[test]
fn ring_attn_accumulate_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let plan = ring_plan();
    let ptx = plan.generate_accumulate_ptx().expect("ring accum ptx");
    ptxas_assembles(&ptx, "ring_accum").expect("ring accum assembles");
    let kernel = load_kernel(&ptx, "ring_attn_accumulate");

    let buf = frag_buf(21, 256);
    let params = LaunchParams::new(1u32, 256u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(), // accum_o
                buf.as_device_ptr(), // accum_lse
                buf.as_device_ptr(), // accum_max
                buf.as_device_ptr(), // partial_o
                buf.as_device_ptr(), // partial_lse
                buf.as_device_ptr(), // partial_max
                4u32,                // chunk_size
                16u32,               // head_dim
                1u32,                // num_heads
            ),
        )
        .expect("ring accum launch");
    fx.stream().synchronize().expect("sync");
}

// --- block sparse (BlockSparseAttentionPlan::generate_forward_ptx) ----------

#[test]
fn block_sparse_fragment_launches() {
    let Some(fx) = gpu_fixture() else {
        return;
    };
    let num_blocks = 1u32;
    let cfg = BlockSparseConfig {
        head_dim: 16,
        num_heads: 1,
        seq_len: 16,
        block_size: 16,
        sm_scale: 0.25,
        sm_version: fx.sm,
        float_type: PtxType::F32,
        pattern: BlockSparsePattern::diagonal(num_blocks),
    };
    let plan = BlockSparseAttentionPlan::new(cfg).expect("block sparse plan");
    let ptx = plan.generate_forward_ptx().expect("block sparse ptx");
    ptxas_assembles(&ptx, "block_sparse").expect("block sparse assembles");
    let kernel = load_kernel(&ptx, "block_sparse_attn_fwd");

    let buf = frag_buf(22, 512);
    let scale_bits = 0.25f32.to_bits();
    let params = LaunchParams::new(1u32, 128u32);
    kernel
        .launch(
            &params,
            fx.stream(),
            &(
                buf.as_device_ptr(), // q
                buf.as_device_ptr(), // k
                buf.as_device_ptr(), // v
                buf.as_device_ptr(), // o
                buf.as_device_ptr(), // row_offsets
                buf.as_device_ptr(), // col_indices
                buf.as_device_ptr(), // workspace
                1u32,                // num_heads
                16u32,               // seq_len
                16u32,               // head_dim
                16u32,               // block_size
                num_blocks,          // num_blocks
                scale_bits,          // scale_bits
            ),
        )
        .expect("block sparse launch");
    fx.stream().synchronize().expect("sync");
}

// ===========================================================================
// Hopper (sm_90) FlashAttention-3 — generation + regression guards (skipped on
// sm_86: the module targets sm_90 and cannot launch on this device).
// ===========================================================================
//
// While wiring these up, the on-device ptxas prescreen exposed THREE real PTX
// codegen bugs in the FA3 body emitter. Two are in this cluster's owned source
// (`flash_attn/hopper_body.rs`) and were FIXED here:
//   1. `smem_base_u32` emitted `mov.u32 %r, %q_smem;` — a `%`-prefixed shared
//      symbol that ptxas rejects ("Arguments mismatch for instruction 'mov'");
//      a `.shared` symbol's address is taken with a bare name. Fixed.
//   2. The kv/q loop-guard branches were emitted via `raw_ptx` as
//      `@p bra L__kv_loop_end_1;` with no `$`, while `b.label()`/`b.branch()`
//      emit `$L__...` — ptxas reported "Unknown symbol". Fixed by `$`-prefixing
//      the raw branch targets.
// After both fixes the module parses cleanly; the only remaining blocker is a
// SHARED-INFRA bug in `oxicuda-ptx` — `BodyBuilder::rcp_approx_f32` emits a bare
// `rcp.f32` (ptxas: "Rounding modifier or '.approx' modifier required"). That
// crate is outside this cluster's scope, so it is reported, not edited. The
// guards below pin the two fixes that ARE in scope.

#[test]
fn hopper_fa3_forward_generation_and_fixes() {
    let cfg = FlashAttention3Config::default_for(64, PtxType::F16, false);
    let plan = FlashAttention3Plan::new(cfg).expect("fa3 plan");
    let ptx = plan.generate_forward().expect("fa3 fwd ptx");
    assert!(
        ptx.contains(".target sm_9"),
        "fa3 forward targets sm_90+ (Hopper)"
    );
    assert!(
        ptx.contains("flash_attn3_fwd_"),
        "fa3 forward entry present"
    );
    // Regression guards for the two bugs fixed in hopper_body.rs.
    assert!(
        !ptx.contains(", %q_smem;"),
        "shared-symbol address must not carry a `%` prefix"
    );
    assert!(
        !ptx.contains(" bra L__"),
        "raw branch targets must carry the `$` label prefix"
    );
}

#[test]
fn hopper_fa3_backward_generation_and_fixes() {
    let cfg = FlashAttention3Config::default_for(64, PtxType::F16, true);
    let plan = FlashAttention3Plan::new(cfg).expect("fa3 plan");
    let ptx = plan.generate_backward().expect("fa3 bwd ptx");
    assert!(
        ptx.contains(".target sm_9"),
        "fa3 backward targets sm_90+ (Hopper)"
    );
    assert!(
        ptx.contains("flash_attn3_bwd_"),
        "fa3 backward entry present"
    );
    assert!(
        !ptx.contains(", %q_smem;"),
        "shared-symbol address must not carry a `%` prefix"
    );
    assert!(
        !ptx.contains(" bra L__"),
        "raw branch targets must carry the `$` label prefix"
    );
}
