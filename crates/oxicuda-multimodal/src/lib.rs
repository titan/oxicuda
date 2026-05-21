//! `oxicuda-multimodal` — Multi-modal learning primitives for OxiCUDA.
#![allow(clippy::needless_range_loop)]
//!
//! Pure-Rust implementation of cross-modal attention, alignment, fusion,
//! and encoder/decoder architectures for vision-language models.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-multimodal
//! ├── cross_attn/      — Multi-head cross-attention, self-cross blocks
//! ├── fusion/          — Concatenation, bilinear (MLB/MFB), attention-gated fusion
//! ├── alignment/       — CLIP InfoNCE, ImageBind triple, ITM head
//! ├── encoder/         — BERT text, ViT image, Conformer audio, Temporal ViT video
//! ├── caption/         — Prefix-LM generation, VQA classification head
//! ├── error            — MultiModalError / MmResult
//! ├── handle           — MultiModalHandle (SmVersion + LcgRng)
//! └── ptx_kernels      — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

pub mod alignment;
pub mod caption;
pub mod cross_attn;
pub mod encoder;
pub mod error;
pub mod fusion;
pub mod handle;
pub mod ptx_kernels;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common multi-modal types.
pub mod prelude {
    pub use crate::alignment::contrastive::{clip_loss, imagebind_loss, l2_normalise};
    pub use crate::alignment::llava_projector::{LlavaProjector, LlavaProjectorConfig};
    pub use crate::alignment::matching::{ItmHead, itm_loss};
    pub use crate::alignment::whisper_log_mel::{WhisperLogMel, WhisperLogMelConfig};
    pub use crate::caption::prefix_lm::{PrefixLm, PrefixLmConfig, PrefixLmWeights};
    pub use crate::caption::vqa_head::{VqaHead, softmax, vqa_loss};
    pub use crate::cross_attn::cross_attention::{
        CrossAttention, CrossAttnConfig, CrossAttnWeights,
    };
    pub use crate::cross_attn::flamingo::{
        FlamingoGatedConfig, FlamingoGatedLayer, FlamingoGatedWeights,
    };
    pub use crate::cross_attn::self_cross_block::{
        FeedForward, LayerNorm, SelfCrossBlock, SelfCrossBlockWeights,
    };
    pub use crate::encoder::audio_encoder::{
        AudioEncoder, AudioEncoderConfig, AudioEncoderWeights,
    };
    pub use crate::encoder::coca::{CoCa, CoCaConfig, CoCaWeights};
    pub use crate::encoder::image_encoder::{ViTEncoder, ViTEncoderConfig, ViTEncoderWeights};
    pub use crate::encoder::perceiver_io::{
        PerceiverIo, PerceiverIoConfig, PerceiverIoWeights, PerceiverSelfLayer,
    };
    pub use crate::encoder::qformer::{QFormer, QFormerConfig, QFormerWeights};
    pub use crate::encoder::text_encoder::{BertConfig, BertEncoder, BertWeights};
    pub use crate::encoder::video_encoder::{
        VideoEncoder, VideoEncoderConfig, VideoEncoderWeights,
    };
    pub use crate::error::{MmResult, MultiModalError};
    pub use crate::fusion::attention_fusion::AttentionFusion;
    pub use crate::fusion::bilinear_fusion::{MfbFusion, MlbFusion};
    pub use crate::fusion::concat_fusion::ConcatFusion;
    pub use crate::handle::{LcgRng, MultiModalHandle, SmVersion};
    pub use crate::ptx_kernels::{
        bilinear_pool_ptx, cross_attn_score_ptx, f32_hex, gate_fusion_ptx, itm_bce_ptx,
        modal_align_loss_ptx, temporal_pool_ptx, token_merge_ptx,
    };
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    // ── E2E 1: Cross-attention shape ─────────────────────────────────────────
    #[test]
    fn e2e_cross_attention_shape() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let weights = CrossAttnWeights::identity(&cfg);
        let attn = CrossAttention::with_weights(cfg, weights);

        let q_len = 5;
        let kv_len = 7;
        let query = vec![0.4_f32; q_len * d];
        let kv = vec![0.3_f32; kv_len * d];
        let out = attn.forward(&query, &kv, &kv, q_len, kv_len).unwrap();
        assert_eq!(
            out.len(),
            q_len * d,
            "output shape must be [q_len * d_model]"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 2: SelfCrossBlock residual + shape ────────────────────────────────
    #[test]
    fn e2e_self_cross_block_residual() {
        let cfg = CrossAttnConfig::tiny();
        let d = cfg.d_model;
        let block = SelfCrossBlock::new(cfg);

        let q_len = 4;
        let ctx_len = 6;
        let q = vec![0.2_f32; q_len * d];
        let ctx = vec![0.5_f32; ctx_len * d];
        let out = block.forward(&q, &ctx, q_len, ctx_len).unwrap();
        assert_eq!(out.len(), q_len * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 3: MLB fusion shape ───────────────────────────────────────────────
    #[test]
    fn e2e_mlb_fusion_shape() {
        let fuser = MlbFusion::zeros(16, 16, 32, 8);
        let v = vec![0.5_f32; 4 * 16];
        let q = vec![0.3_f32; 4 * 16];
        let out = fuser.forward(&v, &q, 4).unwrap();
        assert_eq!(out.len(), 4 * 8, "MLB output shape must be [batch * d_out]");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 4: AttentionFusion weights sum ───────────────────────────────────
    #[test]
    fn e2e_attention_fusion_weights_sum() {
        let af = AttentionFusion::zeros(3, 8).unwrap();
        let m0 = vec![1.0_f32; 8];
        let m1 = vec![2.0_f32; 8];
        let m2 = vec![0.5_f32; 8];
        let (weights, fused) = af.forward(&[&m0, &m1, &m2]).unwrap();
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "attention weights must sum to 1.0, got {sum}"
        );
        assert_eq!(fused.len(), 8);
        assert!(fused.iter().all(|v| v.is_finite()));
    }

    // ── E2E 5: CLIP loss with identical features ──────────────────────────────
    #[test]
    fn e2e_clip_loss_identical_gives_ln_n() {
        let n = 8;
        let dim = 16;
        // Identical unit vectors: all pairs have cosine sim = 1
        // → softmax is uniform → loss = ln(N)
        let mut feats = vec![0.0_f32; n * dim];
        for i in 0..n {
            feats[i * dim] = 1.0; // all identical (same direction)
        }
        let loss = clip_loss(&feats, &feats, n, dim, 1.0).unwrap();
        let ln_n = (n as f32).ln();
        assert!(
            loss.is_finite() && loss >= 0.0,
            "loss={loss} should be finite and non-negative"
        );
        // With identical features at T=1: sim matrix all-ones → uniform softmax → loss = ln(N)
        assert!(
            (loss - ln_n).abs() < 0.01,
            "loss={loss} should equal ln({n})={ln_n}"
        );
    }

    // ── E2E 6: ITM loss perfect predictions near zero ─────────────────────────
    #[test]
    fn e2e_itm_loss_perfect_prediction_near_zero() {
        let logits = vec![100.0_f32; 8]; // very high logits → σ(100) ≈ 1
        let labels = vec![1.0_f32; 8]; // all matched
        let loss = itm_loss(&logits, &labels).unwrap();
        assert!(
            loss < 0.01,
            "perfect prediction loss should be near zero: {loss}"
        );
        assert!(loss >= 0.0, "BCE loss must be non-negative");
    }

    // ── E2E 7: BERT encoder shape ─────────────────────────────────────────────
    #[test]
    fn e2e_bert_encoder_shape() {
        let cfg = BertConfig::tiny();
        let weights = BertWeights::zeros(&cfg);
        let token_ids = [0_u32, 1, 2, 3, 4];
        let out = BertEncoder::forward(&token_ids, &weights, &cfg).unwrap();
        assert_eq!(out.len(), cfg.d_model, "BERT output must be [d_model]");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 8: ViT encoder shape ──────────────────────────────────────────────
    #[test]
    fn e2e_vit_encoder_shape() {
        let cfg = ViTEncoderConfig::tiny();
        let weights = ViTEncoderWeights::zeros(&cfg);
        let image = vec![0.5_f32; 3 * 32 * 32];
        let out = ViTEncoder::forward(&image, &cfg, &weights).unwrap();
        assert_eq!(out.len(), cfg.d_model, "ViT CLS output must be [d_model]");
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 9: Audio encoder shape ────────────────────────────────────────────
    #[test]
    fn e2e_audio_encoder_shape() {
        let cfg = AudioEncoderConfig::tiny();
        let weights = AudioEncoderWeights::zeros(&cfg);
        let n_frames = 20;
        let mel = vec![0.1_f32; n_frames * cfg.n_mels];
        let out = AudioEncoder::forward(&mel, n_frames, &cfg, &weights).unwrap();
        assert_eq!(
            out.len(),
            2 * cfg.d_model,
            "audio encoder output must be [2 * d_model]"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 10: Video encoder shape ───────────────────────────────────────────
    #[test]
    fn e2e_video_encoder_shape() {
        let cfg = VideoEncoderConfig::tiny();
        let weights = VideoEncoderWeights::zeros(&cfg, 16);
        let frame_size = 3 * 32 * 32;
        let n_frames = 4;
        let frames = vec![0.2_f32; n_frames * frame_size];
        let out = VideoEncoder::forward(&frames, n_frames, &cfg, &weights).unwrap();
        assert_eq!(
            out.len(),
            cfg.d_model(),
            "video encoder output must be [d_model]"
        );
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── E2E 11: VQA head shape and loss ──────────────────────────────────────
    #[test]
    fn e2e_vqa_head_shape_and_loss() {
        let head = VqaHead::zeros(16, 32, 50).unwrap();
        let fused = vec![0.5_f32; 16];
        let logits = head.forward(&fused).unwrap();
        assert_eq!(logits.len(), 50, "VQA logits must have n_answers elements");

        let probs = softmax(&logits).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax of logits must sum to 1");

        let loss = vqa_loss(&logits, 5).unwrap();
        assert!(loss.is_finite(), "VQA CE loss must be finite");
        assert!(loss >= 0.0, "VQA CE loss must be non-negative");
    }

    // ── E2E 12: PTX kernels all SM versions ──────────────────────────────────
    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sms = [75_u32, 80, 86, 90, 100, 120];
        type KernelEntry = (&'static str, fn(u32) -> String);
        let kernels: &[KernelEntry] = &[
            ("cross_attn_score_kernel", cross_attn_score_ptx),
            ("modal_align_loss_kernel", modal_align_loss_ptx),
            ("bilinear_pool_kernel", bilinear_pool_ptx),
            ("temporal_pool_kernel", temporal_pool_ptx),
            ("token_merge_kernel", token_merge_ptx),
            ("gate_fusion_kernel", gate_fusion_ptx),
            ("itm_bce_kernel", itm_bce_ptx),
        ];

        for &sm in &sms {
            for (kernel_name, gen_fn) in kernels.iter() {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} SM={sm} missing target"
                );
                assert!(
                    ptx.contains(".version"),
                    "PTX for {kernel_name} SM={sm} missing .version"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "PTX for {kernel_name} SM={sm} missing .visible .entry"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "PTX for {kernel_name} SM={sm} missing kernel name"
                );
            }
        }

        // Verify f32_hex helper via prelude
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
        assert_eq!(f32_hex(0.0_f32), "0F00000000");
    }
}
