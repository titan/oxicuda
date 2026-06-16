//! `oxicuda-peft` — Parameter-Efficient Fine-Tuning primitives for OxiCUDA.
//!
//! Pure-Rust CPU simulation of PEFT methods covering the full spectrum from low-rank adapters
//! to prompt-based methods, adapter modules, sparse fine-tuning, and model merging utilities.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-peft
//! ├── lora/           — LoRA, QLoRA, AdaLoRA, DoRA
//! ├── ia3/            — IA³ element-wise scaling
//! ├── prefix/         — Prefix-Tuning, P-Tuning v2, Prompt-Tuning
//! ├── adapter/        — Houlsby, Pfeiffer, Parallel, Compacter adapters
//! ├── bitfit/         — BitFit bias-only fine-tuning
//! ├── diff_pruning/   — Diff-Pruning with Hard Concrete L0 regularisation
//! ├── merge/          — Linear merge, TIES, DARE arithmetic
//! ├── metrics/        — Efficiency metrics and merge quality tests
//! ├── handle          — SmVersion, LcgRng, PeftHandle
//! ├── error           — PeftError / PeftResult
//! └── ptx_kernels     — 7 GPU PTX kernel strings × 6 SM versions
//! ```

pub mod adapter;
pub mod bitfit;
pub mod diff_pruning;
pub mod error;
pub mod handle;
pub mod ia3;
pub mod lora;
pub mod memory;
pub mod merge;
pub mod metrics;
pub mod prefix;
pub mod ptx_kernels;
pub mod quant;

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use adapter::houlsby::HoulsbyAdapter;
    use bitfit::bitfit::BitFitMask;
    use handle::LcgRng;
    use ia3::ia3::{Ia3Placement, Ia3Vector};
    use lora::adalora::{AdaloraConfig, AdaloraLinear};
    use lora::lora::{LoraConfig, LoraLinear};
    use lora::qlora::{dequantize_block, quantize_block};
    use prefix::prefix_tuning::{PrefixConfig, PrefixModule};
    use prefix::prompt_tuning::SoftPrompt;
    use ptx_kernels::*;

    #[test]
    fn lora_forward_no_change_with_zero_b() {
        // B is zero-initialised → LoRA delta = 0 → output = W·x
        let mut rng = LcgRng::new(1);
        let cfg = LoraConfig {
            r: 4,
            alpha: 8.0,
            init_scale: 0.01,
        };
        let lora = LoraLinear::new(8, 8, &cfg, &mut rng);
        // W is also zero-initialised → output should be all zeros
        let x: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let out = lora.forward(&x);
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!(
                v.abs() < 1e-6,
                "expected zero output with zero W and B, got {v}"
            );
        }
    }

    #[test]
    fn lora_scale_equals_alpha_over_r() {
        let cfg = LoraConfig {
            r: 8,
            alpha: 16.0,
            init_scale: 0.01,
        };
        let mut rng = LcgRng::new(2);
        let lora = LoraLinear::new(32, 32, &cfg, &mut rng);
        let expected = cfg.alpha / cfg.r as f32;
        assert!(
            (lora.scale - expected).abs() < 1e-7,
            "scale={} expected={}",
            lora.scale,
            expected
        );
    }

    #[test]
    fn lora_merge_unmerge_roundtrip() {
        let mut rng = LcgRng::new(3);
        let cfg = LoraConfig {
            r: 4,
            alpha: 4.0,
            init_scale: 0.1,
        };
        let mut lora = LoraLinear::new(8, 8, &cfg, &mut rng);
        // Set W to something non-trivial
        for (i, v) in lora.w.iter_mut().enumerate() {
            *v = i as f32 * 0.01;
        }
        let w_before = lora.w.clone();
        lora.merge_into_w();
        lora.unmerge_from_w();
        for (before, after) in w_before.iter().zip(lora.w.iter()) {
            assert!(
                (before - after).abs() < 1e-4,
                "merge-unmerge roundtrip failed: {before} vs {after}"
            );
        }
    }

    #[test]
    fn qlora_dequant_reasonable_range() {
        let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let (codes, absmax) = quantize_block(&original);
        let dequant = dequantize_block(&codes, absmax);
        assert_eq!(dequant.len(), original.len());
        for &v in &dequant {
            assert!(
                v >= -absmax - 1e-6 && v <= absmax + 1e-6,
                "dequantized value {v} outside [-{absmax}, {absmax}]"
            );
        }
    }

    #[test]
    fn adalora_importance_scores_nonneg() {
        let mut rng = LcgRng::new(5);
        let cfg = AdaloraConfig {
            r: 6,
            alpha: 12.0,
            target_r: 3,
        };
        let adalora = AdaloraLinear::new(16, 16, &cfg, &mut rng);
        let scores = adalora.importance_scores();
        assert_eq!(scores.len(), 6);
        for &s in &scores {
            assert!(s >= 0.0, "importance score must be non-negative, got {s}");
        }
    }

    #[test]
    fn adalora_prune_reduces_rank() {
        let mut rng = LcgRng::new(6);
        let cfg = AdaloraConfig {
            r: 8,
            alpha: 16.0,
            target_r: 3,
        };
        let mut adalora = AdaloraLinear::new(16, 16, &cfg, &mut rng);
        adalora.prune_to_target();
        let nonzero = adalora.lambda.iter().filter(|&&v| v != 0.0).count();
        assert!(
            nonzero <= cfg.target_r,
            "after pruning, {nonzero} non-zero lambdas but target_r={}",
            cfg.target_r
        );
    }

    #[test]
    fn ia3_identity_scale() {
        let vec = Ia3Vector::new(16, Ia3Placement::Key);
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let out = vec.apply(&x);
        for (xi, oi) in x.iter().zip(out.iter()) {
            assert!(
                (xi - oi).abs() < 1e-7,
                "IA³ with ones-scale should be identity"
            );
        }
    }

    #[test]
    fn prefix_shape_correct() {
        let cfg = PrefixConfig {
            num_virtual_tokens: 10,
            prefix_dim: 64,
            num_layers: 4,
            num_heads: 8,
            head_dim: 64,
        };
        let mut rng = LcgRng::new(7);
        let module = PrefixModule::new(cfg.clone(), &mut rng);
        let (vt, nh, hd) = module.prefix_shape();
        assert_eq!(vt, cfg.num_virtual_tokens);
        assert_eq!(nh, cfg.num_heads);
        assert_eq!(hd, cfg.head_dim);
    }

    #[test]
    fn soft_prompt_prepend_length() {
        let mut rng = LcgRng::new(8);
        let num_tokens = 5;
        let embed_dim = 32;
        let seq_len = 20;
        let prompt = SoftPrompt::new(num_tokens, embed_dim, &mut rng);
        let seq: Vec<f32> = vec![0.1_f32; seq_len * embed_dim];
        let out = prompt.prepend_to_sequence(&seq, seq_len);
        assert_eq!(
            out.len(),
            (num_tokens + seq_len) * embed_dim,
            "prepend_to_sequence output length mismatch"
        );
    }

    #[test]
    fn houlsby_adapter_residual_init() {
        // With up_w=zeros, adapter output should equal the residual (original input)
        // plus the zero up-projection output, so output ≈ input.
        let mut rng = LcgRng::new(9);
        let in_dim = 16;
        let bottleneck_dim = 4;
        let seq_len = 3;
        let adapter = HoulsbyAdapter::new(in_dim, bottleneck_dim, &mut rng);
        let x: Vec<f32> = (0..seq_len * in_dim).map(|i| i as f32 * 0.01).collect();
        let out = adapter.forward(&x, seq_len);
        assert_eq!(out.len(), seq_len * in_dim);
        // Since up_w is zero-init, adapter branch contributes nothing; output = residual = x
        for (xi, oi) in x.iter().zip(out.iter()) {
            assert!(
                (xi - oi).abs() < 1e-5,
                "Houlsby zero-init: expected output≈input, got {xi} vs {oi}"
            );
        }
    }

    #[test]
    fn bitfit_trainable_param_count() {
        let mask = BitFitMask::for_transformer(12, 768, 3072, 12);
        let total = mask.total_trainable_params();
        assert!(total > 0, "BitFit should have > 0 trainable params");
        // For 12 layers: 8 biases per layer, sizes: 768*6 + 3072 + 768 = 4608+3072 = 7680 per layer
        // Rough lower bound check
        assert!(
            total >= 12 * (6 * 768 + 3072),
            "BitFit param count too low: {total}"
        );
    }

    #[test]
    fn ptx_kernels_non_empty_all_sm() {
        for sm in [75u32, 80, 86, 89, 90, 100] {
            assert!(!lora_matmul_ptx(sm).is_empty(), "lora_matmul sm={sm}");
            assert!(!ia3_scale_ptx(sm).is_empty(), "ia3_scale sm={sm}");
            assert!(!prefix_expand_ptx(sm).is_empty(), "prefix_expand sm={sm}");
            assert!(
                !adapter_forward_ptx(sm).is_empty(),
                "adapter_forward sm={sm}"
            );
            assert!(!nf4_dequant_ptx(sm).is_empty(), "nf4_dequant sm={sm}");
            assert!(!lora_merge_ptx(sm).is_empty(), "lora_merge sm={sm}");
            assert!(!prompt_concat_ptx(sm).is_empty(), "prompt_concat sm={sm}");
        }
    }
}
