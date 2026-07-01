//! # oxicuda-mamba
//!
//! State Space Model (SSM) primitives for OxiCUDA: S4 (HiPPO-LegS / DPLR),
//! Mamba selective scan (S6), Mamba-2 (SSD), and RWKV time-mixing —
//! pure Rust, zero CUDA SDK dependency.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-mamba
//! ├── error       — MambaError / MambaResult
//! ├── handle      — MambaHandle (SmVersion + LcgRng)
//! ├── ssm         — Discretization, prefix scan, SSM forward kernel,
//! │                 Liquid-S4 (input-modulated Δ) and selective-scan backward
//! ├── s4          — S4: HiPPO-LegS init, DPLR parameterization, S4 layer
//! │   ├── hippo   — HiPPO-LegS A/B matrices and NPLR decomposition
//! │   ├── dplr    — Diagonal Plus Low Rank SSM representation and kernel
//! │   └── s4_layer — Full S4 sequence layer (multi-channel, bidirectional)
//! ├── mamba       — Mamba (S6): selective scan, block, full LM
//! │   ├── selective_scan — Pure-Rust S6 selective scan reference
//! │   ├── mamba_block    — Mamba residual block (conv + gating + SSM)
//! │   └── mamba_model    — Full Mamba language model
//! ├── mamba2      — Mamba-2 (SSD): State Space Duality framework
//! │   ├── ssd         — Core SSD naive and recurrent forms
//! │   ├── chunk_scan  — Chunk-wise scan for efficient SSD computation
//! │   └── mamba2_block — Full Mamba-2 block with multi-head SSD
//! ├── rwkv        — RWKV-4: WKV time-mixing, channel-mixing, full block
//! │   ├── time_mixing    — WKV recurrence and receptance gating
//! │   ├── channel_mixing — Gated FFN with Square-ReLU
//! │   └── rwkv_block     — Complete RWKV residual block with pre-norm
//! ├── ptx_kernels — GPU PTX kernel strings for SSM operations
//! └── quant       — Q-Mamba symmetric INT8 post-training quantization
//! ```

pub mod bidirectional_ssm;
pub mod error;
pub mod handle;
pub mod hybrid;
pub mod hyena;
pub mod linear_attn;
pub mod mamba;
pub mod mamba2;
pub mod mamba_moe;
pub mod mega;
pub mod ptx_kernels;
pub mod quant;
pub mod rwkv;
pub mod s4;
pub mod s5;
pub mod ssm;
pub mod xlstm;

/// On-device GPU validation tests (feature-gated): JIT-compile each hand-written
/// PTX kernel, launch it on the real CUDA device, and assert numerical
/// equivalence to the matching CPU reference. Compiled only under
/// `--features gpu-tests` and only in test builds; every test skips gracefully
/// when no GPU is available.
#[cfg(all(test, feature = "gpu-tests"))]
mod gpu_tests;

/// Convenience re-exports for common Mamba types.
pub mod prelude {
    pub use crate::bidirectional_ssm::{BiDirMode, BiDirSsm, BiDirSsmConfig};
    pub use crate::error::{MambaError, MambaResult};
    pub use crate::handle::{LcgRng, MambaHandle, SmVersion};
    pub use crate::hybrid::mamba_attn::{HybridBlock, HybridConfig};
    pub use crate::hyena::{HyenaConfig, HyenaOperator};
    pub use crate::linear_attn::linear_attention::{
        FeatureMap, LinearAttentionConfig, gated_linear_attention, linear_attention_parallel,
        linear_attention_recurrent,
    };
    pub use crate::linear_attn::retnet::{
        RetentionConfig, RetentionState, msr_decays, retention_chunkwise, retention_parallel,
        retention_recurrent,
    };
    pub use crate::mamba::mamba_block::{
        MambaBlock, MambaBlockConfig, MambaBlockWeights, causal_depthwise_conv1d, linear, rms_norm,
        silu,
    };
    pub use crate::mamba::mamba_model::{MambaConfig, MambaModel, MambaModelWeights};
    pub use crate::mamba::selective_scan::{SelectiveScanConfig, selective_scan, softplus};
    pub use crate::mamba::selective_scan_mixed::{
        MixedPrecision, bf16_round, f16_round, mixed_precision_max_error, selective_scan_mixed,
    };
    pub use crate::mamba::selective_scan_parallel::{
        selective_scan_parallel, verify_selective_scan_equivalence,
    };
    pub use crate::mamba_moe::{MambaMoe, MambaMoeConfig};
    pub use crate::mamba2::chunk_scan::{ChunkConfig, chunk_scan, verify_chunk_equivalence};
    pub use crate::mamba2::mamba2_block::{Mamba2Block, Mamba2BlockConfig, Mamba2BlockWeights};
    pub use crate::mamba2::ssd::{ssd_naive, ssd_recurrent, verify_ssd_equivalence};
    pub use crate::mamba2::ssd_chunk_layer::{SsdChunk, SsdChunkConfig};
    pub use crate::mega::{MegaBlock, MegaConfig};
    pub use crate::ptx_kernels::{
        depthwise_conv1d_ptx, f32_hex, hippo_legendre_ptx, parallel_scan_ptx, rms_norm_silu_ptx,
        selective_scan_ptx, ssd_chunk_ptx, wkv_forward_ptx,
    };
    pub use crate::quant::qmamba::{QMambaQuantizer, QuantScheme, QuantizedTensor};
    pub use crate::rwkv::channel_mixing::{
        ChannelMixingConfig, ChannelMixingLayer, ChannelMixingWeights, square_relu,
    };
    pub use crate::rwkv::rwkv_block::{RwkvBlock, RwkvBlockConfig, RwkvBlockWeights};
    pub use crate::rwkv::rwkv5::{Rwkv5TimeMixLayer, Rwkv5TimeMixWeights, Rwkv5WkvState};
    pub use crate::rwkv::time_mixing::{
        TimeMixingConfig, TimeMixingLayer, TimeMixingWeights, WkvState, layer_norm, sigmoid,
    };
    pub use crate::s4::dplr::Dplr;
    pub use crate::s4::hippo::{hippo_legs, hippo_legs_diag, hippo_nplr};
    pub use crate::s4::s4_fft::{fft, fft_conv1d, s4_fft_conv};
    pub use crate::s4::s4_layer::{S4Config, S4Layer, S4Weights, naive_conv1d};
    pub use crate::s4::s4d::{S4D, S4DConfig, S4DInit};
    pub use crate::s5::{S5Config, S5Layer, S5Weights};
    pub use crate::ssm::discretize::{Discretization, discretize};
    pub use crate::ssm::hippo_variants::{
        HippoFou, HippoFouConfig, HippoLegT, HippoLegTConfig, HippoMatrix, compare_hippo_variants,
        hippo_legs_matrix,
    };
    pub use crate::ssm::liquid::{LiquidS4Config, LiquidS4Layer};
    pub use crate::ssm::parallel_scan::{
        ScanPair, blelloch_inclusive_scan, exclusive_scan, inclusive_scan, ssm_state_scan,
        ssm_state_scan_blelloch,
    };
    pub use crate::ssm::selective_scan_backward::{
        BatchedScanGrads, ScanGrads, scan_backward, scan_backward_batched, scan_forward,
    };
    pub use crate::ssm::ssm_kernel::{SsmConfig, SsmKernel};
    pub use crate::ssm::state_cache::SsmStateCache;
    pub use crate::xlstm::{MLstm, MLstmConfig, MLstmState, SLstm, SLstmConfig, SLstmState};
}

// ─── Integration tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::handle::{LcgRng, MambaHandle, SmVersion};
    use crate::mamba::mamba_block::{MambaBlock, MambaBlockConfig, MambaBlockWeights};
    use crate::mamba::mamba_model::{MambaConfig, MambaModel, MambaModelWeights};
    use crate::mamba::selective_scan::{SelectiveScanConfig, selective_scan, softplus};
    use crate::mamba2::chunk_scan::{ChunkConfig, verify_chunk_equivalence};
    use crate::mamba2::mamba2_block::{Mamba2Block, Mamba2BlockConfig, Mamba2BlockWeights};
    use crate::mamba2::ssd::verify_ssd_equivalence;
    use crate::ptx_kernels::{
        depthwise_conv1d_ptx, hippo_legendre_ptx, parallel_scan_ptx, rms_norm_silu_ptx,
        selective_scan_ptx, ssd_chunk_ptx, wkv_forward_ptx,
    };
    use crate::rwkv::rwkv_block::{RwkvBlock, RwkvBlockConfig, RwkvBlockWeights};
    use crate::rwkv::time_mixing::sigmoid;
    use crate::s4::hippo::{hippo_legs, hippo_nplr};
    use crate::s4::s4_layer::{S4Config, S4Layer};
    use crate::ssm::discretize::{Discretization, discretize};
    use crate::ssm::parallel_scan::{ScanPair, ssm_state_scan};
    use crate::ssm::ssm_kernel::{SsmConfig, SsmKernel};

    fn make_rng() -> LcgRng {
        LcgRng::new(42)
    }

    fn randn(rng: &mut LcgRng, n: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; n];
        rng.fill_normal(&mut v);
        v
    }

    // ── Handle ────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_handle_default() {
        let h = MambaHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn e2e_lcg_rng_reproducibility() {
        let mut a = LcgRng::new(999);
        let mut b = LcgRng::new(999);
        for _ in 0..50 {
            assert_eq!(a.next_f32(), b.next_f32());
        }
    }

    // ── SSM Discretization ────────────────────────────────────────────────────

    #[test]
    fn e2e_ssm_discretize_zoh() {
        let a_diag = vec![-1.0_f32, -2.0, -0.5];
        let b = vec![1.0_f32; 3];
        let (a_bar, b_bar) =
            discretize(&a_diag, &b, 0.1, Discretization::Zoh).expect("ZOH discretize must succeed");
        // For stable A (negative diagonal), A_bar should be in (0, 1)
        for &v in &a_bar {
            assert!(
                v > 0.0 && v < 1.0,
                "A_bar element {v} out of (0,1) for stable A"
            );
        }
        assert!(b_bar.iter().all(|v| v.is_finite()), "B_bar must be finite");
    }

    #[test]
    fn e2e_parallel_scan_associativity() {
        // Verify the (A, b) associative operator property
        let x = ScanPair { a: 2.0, b: 3.0 };
        let y = ScanPair { a: 0.5, b: 1.0 };
        let z = ScanPair { a: 0.8, b: 0.2 };
        let lhs = ScanPair::combine(ScanPair::combine(x, y), z);
        let rhs = ScanPair::combine(x, ScanPair::combine(y, z));
        assert!((lhs.a - rhs.a).abs() < 1e-5, "a: {lhs:?} vs {rhs:?}");
        assert!((lhs.b - rhs.b).abs() < 1e-5, "b: {lhs:?} vs {rhs:?}");
    }

    #[test]
    fn e2e_ssm_state_scan_cumsum_mode() {
        // With a_bar = 1.0 (no decay), ssm scan becomes cumulative sum
        let l = 8;
        let a_bar = vec![1.0_f32; l];
        let b_bar_u = vec![1.0_f32; l]; // unit inputs
        let states = ssm_state_scan(&a_bar, &b_bar_u).expect("scan must succeed");
        // h[t] = 1*h[t-1] + 1 → h[t] = t+1
        for (t, &h) in states.iter().enumerate() {
            assert!(
                (h - (t + 1) as f32).abs() < 1e-4,
                "t={t}: expected {}, got {h}",
                t + 1
            );
        }
    }

    // ── HiPPO + S4 ────────────────────────────────────────────────────────────

    #[test]
    fn e2e_s4_hippo_legs_forward() {
        let (a_mat, b_vec) = hippo_legs(4).expect("HiPPO-LegS n=4 must succeed");
        assert_eq!(a_mat.len(), 16); // 4×4
        assert_eq!(b_vec.len(), 4);
        // B entries should be positive
        assert!(b_vec.iter().all(|&v| v > 0.0), "B entries must be positive");
        // A diagonal: A[n,n] = -(n+1)
        for n in 0..4 {
            let diag = a_mat[n * 4 + n];
            let expected = -((n + 1) as f32);
            assert!(
                (diag - expected).abs() < 1e-4,
                "A[{n},{n}]={diag}, expected {expected}"
            );
        }
    }

    #[test]
    fn e2e_s4_nplr_stability() {
        let (lambda, p, q) = hippo_nplr(8).expect("HiPPO NPLR n=8 must succeed");
        assert_eq!(lambda.len(), 8);
        assert_eq!(p.len(), 8);
        assert_eq!(q.len(), 8);
        // All lambda should be negative (stable eigenvalues)
        assert!(
            lambda.iter().all(|&v| v < 0.0),
            "All lambda must be negative for stability"
        );
        // p == q for HiPPO-LegS
        for (pi, qi) in p.iter().zip(q.iter()) {
            assert!((pi - qi).abs() < 1e-5, "p must equal q for HiPPO-LegS");
        }
    }

    #[test]
    fn e2e_s4_layer_forward_shape() {
        let config = S4Config::new(4, 4, 8).expect("S4Config valid");
        let layer = S4Layer::new(config).expect("S4Layer must construct");
        let mut rng = make_rng();
        let u = randn(&mut rng, 8 * 4); // [L=8, D=4]
        let y = layer.forward(&u).expect("S4 forward must succeed");
        assert_eq!(y.len(), 8 * 4, "output shape [L*D] = {}", 8 * 4);
        assert!(y.iter().all(|v| v.is_finite()), "S4 output must be finite");
    }

    // ── Mamba ─────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_mamba_selective_scan_shape() {
        let config = SelectiveScanConfig::new(2, 8, 4, 4).expect("SelectiveScanConfig valid");
        let mut rng = make_rng();
        let u = randn(&mut rng, 2 * 8 * 4);
        let delta = randn(&mut rng, 2 * 8 * 4);
        let a_log = randn(&mut rng, 4 * 4);
        let b_proj = randn(&mut rng, 2 * 8 * 4);
        let c_proj = randn(&mut rng, 2 * 8 * 4);
        let y = selective_scan(&u, &delta, &a_log, &b_proj, &c_proj, &config)
            .expect("selective_scan must succeed");
        assert_eq!(y.len(), 2 * 8 * 4);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_mamba_block_forward_finite() {
        let config = MambaBlockConfig::new(16).expect("MambaBlockConfig must be valid for D=16");
        let mut rng = make_rng();
        let weights = MambaBlockWeights::random(&config, &mut rng);
        let block = MambaBlock::new(config, weights).expect("MambaBlock must construct");
        let u = randn(&mut rng, 8 * 16); // [L=8, D=16]
        let y = block
            .forward(&u, 8)
            .expect("MambaBlock forward must succeed");
        assert_eq!(y.len(), 8 * 16);
        assert!(
            y.iter().all(|v| v.is_finite()),
            "MambaBlock output must be finite"
        );
    }

    #[test]
    fn e2e_mamba_model_decode() {
        let config = MambaConfig::tiny(); // vocab=256, D=32, 2 layers
        let mut rng = make_rng();
        let weights = MambaModelWeights::random(&config, &mut rng);
        let model = MambaModel::new(config.clone(), weights).expect("MambaModel must construct");
        // 5-step greedy decode starting from token 0
        let mut context = vec![0usize];
        for _ in 0..5 {
            let next = model.next_token(&context).expect("next_token must succeed");
            assert!(
                next < config.vocab_size,
                "token {next} out of vocab {}",
                config.vocab_size
            );
            context.push(next);
        }
        assert_eq!(context.len(), 6);
    }

    #[test]
    fn e2e_mamba_parallel_scan_equals_sequential() {
        // The work-efficient Blelloch parallel selective scan must agree with
        // the sequential reference (the algorithm the fused GPU kernel realises).
        use crate::mamba::selective_scan_parallel::verify_selective_scan_equivalence;
        let cfg = SelectiveScanConfig::new(2, 16, 4, 8).expect("config");
        let mut rng = make_rng();
        let u = randn(&mut rng, 2 * 16 * 4);
        let delta = randn(&mut rng, 2 * 16 * 4);
        let a_log = randn(&mut rng, 4 * 8);
        let b_proj = randn(&mut rng, 2 * 16 * 8);
        let c_proj = randn(&mut rng, 2 * 16 * 8);
        let ok =
            verify_selective_scan_equivalence(&u, &delta, &a_log, &b_proj, &c_proj, &cfg, 1e-3)
                .expect("verify");
        assert!(ok, "parallel selective scan must match sequential");
    }

    #[test]
    fn e2e_mamba_mixed_precision_close_to_fp32() {
        // FP16 / BF16 mixed-precision scan with FP32 accumulation stays close to
        // the full-FP32 reference.
        use crate::mamba::selective_scan_mixed::{MixedPrecision, mixed_precision_max_error};
        let cfg = SelectiveScanConfig::new(1, 16, 4, 8).expect("config");
        let mut rng = make_rng();
        let u: Vec<f32> = randn(&mut rng, 16 * 4).iter().map(|v| v * 0.4).collect();
        let delta = vec![0.0_f32; 16 * 4];
        let a_log = vec![0.0_f32; 4 * 8];
        let b_proj: Vec<f32> = randn(&mut rng, 16 * 8).iter().map(|v| v * 0.3).collect();
        let c_proj: Vec<f32> = randn(&mut rng, 16 * 8).iter().map(|v| v * 0.3).collect();
        for prec in [MixedPrecision::Fp16, MixedPrecision::Bf16] {
            let err = mixed_precision_max_error(&u, &delta, &a_log, &b_proj, &c_proj, &cfg, prec)
                .expect("err");
            assert!(err < 0.5, "{prec:?} max error {err} too large");
        }
    }

    #[test]
    fn e2e_mamba_softplus_values() {
        assert!((softplus(0.0) - 2.0_f32.ln()).abs() < 1e-5);
        assert!((softplus(100.0) - 100.0).abs() < 1e-4);
        assert!(softplus(-100.0) < 1e-5);
    }

    // ── Mamba-2 ───────────────────────────────────────────────────────────────

    #[test]
    fn e2e_mamba2_ssd_naive_vs_recurrent() {
        let l = 8;
        let n = 2;
        let mut rng = make_rng();
        let a_seq: Vec<f32> = (0..l).map(|_| rng.next_f32() * 0.5 + 0.3).collect(); // (0.3,0.8)
        let b_seq = randn(&mut rng, l * n);
        let c_seq = randn(&mut rng, l * n);
        let x = randn(&mut rng, l);
        let ok = verify_ssd_equivalence(&a_seq, &b_seq, &c_seq, &x, l, n, 1e-4)
            .expect("verify must succeed");
        assert!(ok, "ssd_naive and ssd_recurrent must agree");
    }

    #[test]
    fn e2e_mamba2_chunk_scan_vs_recurrent() {
        let l = 16;
        let n = 2;
        let mut rng = make_rng();
        let a_seq: Vec<f32> = (0..l).map(|_| rng.next_f32() * 0.5 + 0.3).collect();
        let b_seq = randn(&mut rng, l * n);
        let c_seq = randn(&mut rng, l * n);
        let x = randn(&mut rng, l);
        let config = ChunkConfig::new(l, 4, n).expect("ChunkConfig valid");
        let ok = verify_chunk_equivalence(&a_seq, &b_seq, &c_seq, &x, &config, 1e-3)
            .expect("chunk verify must succeed");
        assert!(ok, "chunk_scan must match ssd_recurrent");
    }

    #[test]
    fn e2e_mamba2_block_forward_shape() {
        let config = Mamba2BlockConfig::new(8, 2).expect("Mamba2BlockConfig valid");
        let mut rng = make_rng();
        let weights = Mamba2BlockWeights::random(&config, &mut rng);
        let block = Mamba2Block::new(config, weights).expect("Mamba2Block must construct");
        let u = randn(&mut rng, 4 * 8); // [L=4, D=8]
        let y = block
            .forward(&u, 4)
            .expect("Mamba2Block forward must succeed");
        assert_eq!(y.len(), 4 * 8);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── RWKV ─────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_rwkv_wkv_numerical_stability() {
        // Large k values should remain finite due to running-max trick
        use crate::rwkv::time_mixing::{TimeMixingConfig, TimeMixingLayer, TimeMixingWeights};
        let config = TimeMixingConfig::new(4, 32).expect("TimeMixingConfig valid");
        let mut rng = make_rng();
        let weights = TimeMixingWeights::random(&config, &mut rng);
        let layer = TimeMixingLayer::new(config, weights).expect("TimeMixingLayer must construct");
        // Input with large values that would cause naive exp to overflow
        let x: Vec<f32> = (0..32 * 4).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let y = layer
            .forward(&x)
            .expect("time_mixing forward must not overflow");
        assert!(
            y.iter().all(|v| v.is_finite()),
            "WKV output must remain finite"
        );
    }

    #[test]
    fn e2e_rwkv_block_forward_shape() {
        let config = RwkvBlockConfig::new(8, 4).expect("RwkvBlockConfig valid");
        let mut rng = make_rng();
        let weights = RwkvBlockWeights::random(&config, &mut rng);
        let block = RwkvBlock::new(config.clone(), weights).expect("RwkvBlock must construct");
        let u = randn(&mut rng, 4 * 8); // [L=4, D=8]
        let y = block.forward(&u).expect("RwkvBlock forward must succeed");
        assert_eq!(y.len(), 4 * 8);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn e2e_rwkv_sigmoid_range() {
        for x in [-100.0_f32, -1.0, 0.0, 1.0, 100.0] {
            let s = sigmoid(x);
            assert!((0.0..=1.0).contains(&s), "sigmoid({x})={s} out of [0,1]");
        }
    }

    // ── SSM State Cache (streaming / checkpoint) ──────────────────────────────

    #[test]
    fn e2e_ssm_state_cache_streaming_and_checkpoint() {
        // Streaming in two chunks (with a checkpoint/restore across the split)
        // must equal a single full-sequence scan.
        use crate::ssm::state_cache::SsmStateCache;
        let (d, n, l) = (2_usize, 3_usize, 12_usize);
        let mut rng = make_rng();
        let u = randn(&mut rng, l * d);
        let a_bar: Vec<f32> = (0..l * d * n)
            .map(|_| rng.next_f32() * 0.9 + 0.05)
            .collect();
        let b_bar = randn(&mut rng, l * d * n);
        let c = randn(&mut rng, l * d * n);

        let mut full = SsmStateCache::new(d, n).expect("cache");
        let reference = full.advance_chunk(&u, &a_bar, &b_bar, &c, l).expect("full");

        let split = 5_usize;
        let mut cache = SsmStateCache::new(d, n).expect("cache");
        let _ = cache
            .advance_chunk(
                &u[..split * d],
                &a_bar[..split * d * n],
                &b_bar[..split * d * n],
                &c[..split * d * n],
                split,
            )
            .expect("first");
        let snap = cache.checkpoint();
        let mut resumed = SsmStateCache::from_checkpoint(d, n, &snap).expect("resume");
        let rest = l - split;
        let tail = resumed
            .advance_chunk(
                &u[split * d..],
                &a_bar[split * d * n..],
                &b_bar[split * d * n..],
                &c[split * d * n..],
                rest,
            )
            .expect("second");
        for (i, &v) in tail.iter().enumerate() {
            let r = reference[split * d + i];
            assert!((v - r).abs() < 1e-5, "stream/checkpoint mismatch at {i}");
        }
    }

    // ── SSM Kernel ────────────────────────────────────────────────────────────

    #[test]
    fn e2e_ssm_kernel_forward() {
        let config = SsmConfig::new(1, 4, 4, 4).expect("SsmConfig valid");
        let kernel = SsmKernel::new(config, None).expect("SsmKernel must construct");
        let mut rng = make_rng();
        let b = 1;
        let l = 4;
        let d = 4;
        let n = 4;
        let u = randn(&mut rng, b * l * d);
        let b_proj = randn(&mut rng, b * l * d * n);
        let c_proj = randn(&mut rng, b * l * d * n);
        let y = kernel
            .forward(&u, &b_proj, &c_proj, 0.01)
            .expect("SSM forward must succeed");
        assert_eq!(y.len(), b * l * d);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    // ── PTX Kernels × All SM Versions ─────────────────────────────────────────

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions: &[u32] = &[75, 80, 86, 90, 100, 120];
        let generators: &[(&str, fn(u32) -> String)] = &[
            ("selective_scan", selective_scan_ptx as fn(u32) -> String),
            ("parallel_scan", parallel_scan_ptx),
            ("depthwise_conv1d", depthwise_conv1d_ptx),
            ("wkv_forward", wkv_forward_ptx),
            ("ssd_chunk", ssd_chunk_ptx),
            ("hippo_legendre", hippo_legendre_ptx),
            ("rms_norm_silu", rms_norm_silu_ptx),
        ];
        for &sm in sm_versions {
            for (name, kernel_gen) in generators {
                let ptx = kernel_gen(sm);
                assert!(
                    ptx.contains(&format!(".target sm_{sm}")),
                    "kernel '{name}' sm={sm} missing .target directive"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "kernel '{name}' sm={sm} missing .visible .entry"
                );
                let _ = name; // suppress potential unused warning
            }
        }
        // Verify version strings per SM range
        assert!(selective_scan_ptx(120).contains(".version 8.7"));
        assert!(selective_scan_ptx(90).contains(".version 8.4"));
        assert!(selective_scan_ptx(80).contains(".version 8.0"));
        assert!(selective_scan_ptx(75).contains(".version 7.5"));
    }
}
