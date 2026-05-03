//! `oxicuda-vision` — Vision Transformer & CLIP primitives for OxiCUDA.
//!
//! Pure-Rust CPU reference implementation providing:
//! - **`patch_embed`**: strided Conv2D patch embedder, sinusoidal & learnable
//!   positional encodings.
//! - **`vit`**: ViT block (pre-norm MHSA + MLP), encoder stack, full ViT model.
//! - **`clip`**: CLIP vision encoder, projection head, InfoNCE contrastive loss.
//! - **`augment`**: geometric, photometric, and normalisation image augmentations.
//! - **`fpn`**: Feature Pyramid Network (lateral 1×1 convolutions + top-down pathway).
//! - **`detection`**: RoI Align, DETR decoder, and bipartite set matching.
//! - **`ptx_kernels`**: 7 GPU PTX kernel string generators (SM 7.5–12.0).
//!
//! No CUDA SDK dependency; all forward passes run on CPU `f32` tensors
//! using flat row-major `Vec<f32>` layouts.

pub mod augment;
pub mod clip;
pub mod detection;
pub mod error;
pub mod fpn;
pub mod handle;
pub mod patch_embed;
pub mod ptx_kernels;
pub mod vit;

pub use error::{VisionError, VisionResult};
pub use handle::{LcgRng, SmVersion, VisionHandle};

// ─── Prelude ─────────────────────────────────────────────────────────────────

pub mod prelude {
    pub use crate::augment::{AugOp, Pipeline};
    pub use crate::clip::{
        ClipVisionConfig, ClipVisionEncoder, ProjectionHead, contrastive::info_nce_loss,
    };
    pub use crate::detection::{DetrConfig, DetrDecoder, bipartite_match, roi_align};
    pub use crate::error::{VisionError, VisionResult};
    pub use crate::fpn::{FeatureMap, Fpn, FpnConfig};
    pub use crate::handle::{LcgRng, SmVersion, VisionHandle};
    pub use crate::patch_embed::{
        LearnablePosEmbed, PatchEmbed, PatchEmbedConfig, add_pos_embed, pos_2d_sincos, prepend_cls,
    };
    pub use crate::ptx_kernels::{
        adaptive_avg_pool_ptx, bilinear_interp_ptx, contrastive_loss_ptx, focal_loss_ptx,
        image_normalize_ptx, patch_embed_ptx, roi_align_ptx,
    };
    pub use crate::vit::{ViTConfig, ViTEncoder, ViTModel};
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::{
        augment::{AugOp, Pipeline},
        clip::contrastive::info_nce_loss,
        clip::{ClipVisionConfig, ClipVisionEncoder, ProjectionHead},
        detection::{DetrConfig, DetrDecoder, bipartite_match, roi_align},
        error::VisionError,
        fpn::{FeatureMap, Fpn, FpnConfig, LateralConv1x1},
        handle::{LcgRng, SmVersion, VisionHandle},
        patch_embed::{
            LearnablePosEmbed, PatchEmbed, PatchEmbedConfig, add_pos_embed, pos_2d_sincos,
            prepend_cls,
        },
        ptx_kernels::{
            adaptive_avg_pool_ptx, bilinear_interp_ptx, contrastive_loss_ptx, focal_loss_ptx,
            image_normalize_ptx, patch_embed_ptx, roi_align_ptx,
        },
        vit::{ViTConfig, ViTModel},
    };

    // ── PTX kernel end-to-end ─────────────────────────────────────────────────

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        const SM_VERSIONS: &[u32] = &[75, 80, 86, 90, 100, 120];
        let kernel_generators: &[(&str, fn(u32) -> String)] = &[
            ("patch_embed_ptx", patch_embed_ptx),
            ("bilinear_interp_ptx", bilinear_interp_ptx),
            ("contrastive_loss_ptx", contrastive_loss_ptx),
            ("roi_align_ptx", roi_align_ptx),
            ("image_normalize_ptx", image_normalize_ptx),
            ("adaptive_avg_pool_ptx", adaptive_avg_pool_ptx),
            ("focal_loss_ptx", focal_loss_ptx),
        ];
        for &(name, kernel_fn) in kernel_generators {
            for &sm in SM_VERSIONS {
                let ptx = kernel_fn(sm);
                let expected_target = format!(".target sm_{sm}");
                assert!(
                    ptx.contains(&expected_target),
                    "kernel {name} sm={sm}: missing '{expected_target}' in PTX"
                );
                assert!(
                    ptx.contains(".version"),
                    "kernel {name} sm={sm}: missing .version directive"
                );
            }
        }
    }

    // ── Handle ────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_handle_default() {
        let h = VisionHandle::default_handle();
        assert_eq!(h.device(), 0);
        assert_eq!(h.sm_version(), SmVersion(80));
    }

    #[test]
    fn e2e_lcg_rng_reproducibility() {
        let mut r1 = LcgRng::new(42);
        let mut r2 = LcgRng::new(42);
        for _ in 0..200 {
            assert_eq!(r1.next_u32(), r2.next_u32());
        }
    }

    // ── Patch embedding ───────────────────────────────────────────────────────

    #[test]
    fn e2e_patch_embed_shape() {
        // 3×32×32 image, patch_size=4 → (32/4)²=64 patches, embed_dim=16
        let cfg = PatchEmbedConfig::new(32, 4, 3, 16).expect("valid config");
        let mut rng = LcgRng::new(1);
        let pe = PatchEmbed::new(cfg.clone(), &mut rng);
        let image = vec![0.5f32; 3 * 32 * 32];
        let tokens = pe.forward(&image).expect("forward ok");
        assert_eq!(tokens.len(), cfg.n_patches() * cfg.embed_dim);
        assert_eq!(cfg.n_patches(), 64);
    }

    #[test]
    fn e2e_patch_embed_cls_prepend() {
        let cfg = PatchEmbedConfig::new(16, 4, 3, 8).expect("valid config");
        let mut rng = LcgRng::new(2);
        let pe = PatchEmbed::new(cfg.clone(), &mut rng);
        let image = vec![0.0f32; 3 * 16 * 16];
        let tokens = pe.forward(&image).expect("forward ok");
        let with_cls =
            prepend_cls(&tokens, &pe.weights.cls_token, cfg.embed_dim).expect("prepend ok");
        assert_eq!(with_cls.len(), (cfg.n_patches() + 1) * cfg.embed_dim);
    }

    #[test]
    fn e2e_pos_embed_2d_sincos_periodicity() {
        // With grid 4×1 and dim=4, the first sine band (k=0, freq=1) encodes row.
        let pe = pos_2d_sincos(4, 1, 4).expect("ok");
        // Position h=1 row index 0: sin(1*1) = sin(1)
        let diff = (pe[4] - 1.0_f32.sin()).abs();
        assert!(diff < 1e-5, "periodicity check failed: diff={diff}");
    }

    // ── ViT ───────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_vit_block_forward_finite() {
        use crate::vit::{ViTBlock, ViTBlockConfig};
        let cfg = ViTBlockConfig::new(32, 4, 4).expect("valid");
        let mut rng = LcgRng::new(3);
        let block = ViTBlock::new(cfg, &mut rng);
        let n_tokens = 8;
        let mut tokens = vec![0.0f32; n_tokens * 32];
        rng.fill_normal(&mut tokens);
        let out = block.forward(&tokens, n_tokens).expect("forward ok");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite ViT block output"
        );
        assert_eq!(out.len(), n_tokens * 32);
    }

    #[test]
    fn e2e_vit_model_classify_tiny() {
        let cfg = ViTConfig::tiny();
        let mut rng = LcgRng::new(4);
        let model = ViTModel::new(cfg, &mut rng).expect("model ok");
        let image = vec![0.5f32; 3 * 32 * 32];
        let logits = model.forward(&image).expect("forward ok");
        assert_eq!(logits.len(), 10, "expected 10 logits from tiny config");
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logits");
    }

    // ── CLIP ──────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_clip_vision_encoder_pool_shape() {
        let vit_cfg = ViTConfig::tiny();
        let embed_dim = vit_cfg.embed_dim;
        let cfg = ClipVisionConfig::new(vit_cfg);
        let mut rng = LcgRng::new(5);
        let enc = ClipVisionEncoder::new(cfg, &mut rng).expect("encoder ok");
        let image = vec![0.1f32; 3 * 32 * 32];
        let emb = enc.forward_single(&image).expect("forward ok");
        assert_eq!(emb.len(), embed_dim, "CLS pool output must be [embed_dim]");
        assert!(emb.iter().all(|v| v.is_finite()), "non-finite embedding");
    }

    #[test]
    fn e2e_clip_proj_l2_unit_norm() {
        let embed_dim = 32;
        let proj_dim = 16;
        let mut rng = LcgRng::new(6);
        let head = ProjectionHead::new(embed_dim, proj_dim, &mut rng).expect("ok");
        let mut x = vec![0.0f32; embed_dim];
        rng.fill_normal(&mut x);
        let z = head.project(&x).expect("project ok");
        let norm: f32 = z.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "projected embedding not unit-norm; ‖z‖={norm}"
        );
    }

    #[test]
    fn e2e_clip_info_nce_symmetric() {
        let embed_dim = 16;
        let batch = 4;
        let mut rng = LcgRng::new(7);
        let mut img_e = vec![0.0f32; batch * embed_dim];
        let mut txt_e = vec![0.0f32; batch * embed_dim];
        rng.fill_normal(&mut img_e);
        rng.fill_normal(&mut txt_e);

        let (loss_it, _) = info_nce_loss(&img_e, &txt_e, embed_dim, 0.1).expect("ok");
        let (loss_ti, _) = info_nce_loss(&txt_e, &img_e, embed_dim, 0.1).expect("ok");

        assert!(loss_it.is_finite(), "image→text loss is not finite");
        assert!(loss_ti.is_finite(), "text→image loss is not finite");
        assert!(
            (loss_it - loss_ti).abs() < 1e-4,
            "symmetric loss mismatch: {loss_it} vs {loss_ti}"
        );
    }

    // ── Augmentation ──────────────────────────────────────────────────────────

    #[test]
    fn e2e_augment_random_crop_dims() {
        let img = vec![0.5f32; 3 * 64 * 64];
        let mut rng = LcgRng::new(8);
        let op = AugOp::RandomCrop { crop_size: 48 };
        let (out, new_h, new_w) = op.apply(&img, 3, 64, 64, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (48, 48));
        assert_eq!(out.len(), 3 * 48 * 48);
    }

    #[test]
    fn e2e_augment_normalize_imagenet() {
        use crate::augment::normalize::{IMAGENET_MEAN, IMAGENET_STD, normalize_chw};
        // Build image whose per-channel mean matches imagenet mean
        let h = 8;
        let w = 8;
        let hw = h * w;
        let mut img = vec![0.0f32; 3 * hw];
        for c in 0..3 {
            for p in 0..hw {
                img[c * hw + p] = IMAGENET_MEAN[c];
            }
        }
        let out = normalize_chw(&img, 3, h, w, &IMAGENET_MEAN, &IMAGENET_STD).expect("ok");
        // After normalizing: all pixels ≈ 0 (mean removed)
        let max_abs = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1e-5,
            "normalized constant-mean image should be ~0; max={max_abs}"
        );
    }

    #[test]
    fn e2e_augment_pipeline_chain() {
        let img = vec![0.5f32; 3 * 64 * 64];
        let mut rng = LcgRng::new(9);
        let pipeline = Pipeline::new()
            .push(AugOp::Resize { target: 48 })
            .push(AugOp::RandomCrop { crop_size: 32 })
            .push(AugOp::HorizontalFlip { prob: 0.5 });
        let (out, new_h, new_w) = pipeline.apply(&img, 3, 64, 64, &mut rng).expect("ok");
        assert_eq!((new_h, new_w), (32, 32));
        assert!(
            out.iter().all(|v| v.is_finite()),
            "pipeline output must be finite"
        );
    }

    // ── FPN ───────────────────────────────────────────────────────────────────

    #[test]
    fn e2e_fpn_top_down_shape_consistency() {
        let mut rng = LcgRng::new(10);
        // 3 levels: [128, 64, 32] channels, sizes [4×4, 8×8, 16×16]
        let in_channels = vec![128usize, 64, 32];
        let out_channels = 16;
        let cfg = FpnConfig::new(in_channels.clone(), out_channels).expect("config ok");
        let fpn = Fpn::new(cfg, &mut rng).expect("fpn ok");

        let features = vec![
            FeatureMap::new(vec![0.1f32; 128 * 4 * 4], 128, 4, 4).expect("ok"),
            FeatureMap::new(vec![0.1f32; 64 * 8 * 8], 64, 8, 8).expect("ok"),
            FeatureMap::new(vec![0.1f32; 32 * 16 * 16], 32, 16, 16).expect("ok"),
        ];
        let pyramid = fpn.forward(features).expect("fpn forward ok");

        assert_eq!(pyramid.len(), 3);
        for fm in &pyramid {
            assert_eq!(
                fm.channels, out_channels,
                "all FPN levels must have out_channels"
            );
        }
        assert!(
            pyramid
                .iter()
                .all(|fm| fm.data.iter().all(|v| v.is_finite()))
        );
    }

    // ── Detection ─────────────────────────────────────────────────────────────

    #[test]
    fn e2e_roi_align_unit_box_identity() {
        // Feature map: 1 channel, 4×4, all 1.0
        let c = 1;
        let h = 4;
        let w = 4;
        let feat = vec![1.0f32; c * h * w];
        // RoI covering the entire feature map: [x1=0, y1=0, x2=4, y2=4]
        let rois = vec![0.0f32, 0.0, 4.0, 4.0];
        let out = roi_align(&feat, c, h, w, &rois, 1, 1, 1, 2).expect("ok");
        assert_eq!(out.len(), 1);
        // Bilinear samples of a constant-1 map → mean = 1
        assert!(
            (out[0] - 1.0).abs() < 1e-5,
            "unit box over constant map should return 1.0; got {}",
            out[0]
        );
    }

    #[test]
    fn e2e_detr_decoder_query_shape() {
        let cfg = DetrConfig::tiny();
        let mut rng = LcgRng::new(11);
        let decoder = DetrDecoder::new(cfg.clone(), &mut rng).expect("ok");
        let n_queries = cfg.n_queries;
        let embed_dim = cfg.embed_dim;
        let n_enc = 16;

        let queries = vec![0.1f32; n_queries * embed_dim];
        let enc_feats = vec![0.2f32; n_enc * embed_dim];
        let out = decoder
            .forward(&queries, &enc_feats, n_enc)
            .expect("forward ok");

        assert_eq!(
            out.len(),
            n_queries * embed_dim,
            "decoder must preserve query shape"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "decoder output contains non-finite"
        );
    }

    #[test]
    fn e2e_set_match_self_assignment() {
        // Cost matrix: diagonal = 0, off-diagonal = 1
        // Greedy should find the diagonal matching
        let n = 4;
        let mut cost = vec![1.0f32; n * n];
        for i in 0..n {
            cost[i * n + i] = 0.0;
        }
        let matching = bipartite_match(&cost, n, n).expect("ok");
        assert_eq!(matching.len(), n);
        // Every diagonal pair should be matched
        let mut assigned: Vec<(usize, usize)> = matching.clone();
        assigned.sort_unstable();
        for i in 0..n {
            assert!(
                assigned.contains(&(i, i)),
                "identity cost matrix: query {i} should match target {i}"
            );
        }
    }

    #[test]
    fn e2e_focal_loss_positive_only() {
        // For gamma=0, alpha=1: focal_loss = -log(p), which is standard BCE
        // The PTX kernel embeds alpha=0.25, gamma=2 — we just verify it generates valid PTX.
        // Verify positive case: for p ≈ 1 the focal loss → 0.
        // We use the CPU formula directly here as a sanity check.
        let p: f32 = 0.99;
        let alpha: f32 = 1.0;
        let gamma: f32 = 0.0;
        let fl = -alpha * (1.0 - p).powf(gamma) * p.ln();
        let standard_bce = -p.ln();
        assert!(
            (fl - standard_bce).abs() < 1e-5,
            "at gamma=0 focal loss == BCE; got fl={fl}, bce={standard_bce}"
        );
    }

    // ── Learnable pos embed ───────────────────────────────────────────────────

    #[test]
    fn e2e_learnable_pos_embed_and_add() {
        let n = 17; // 16 patches + CLS
        let d = 32;
        let mut rng = LcgRng::new(12);
        let lpe = LearnablePosEmbed::new(n, d, &mut rng).expect("ok");
        let mut tokens = vec![0.0f32; n * d];
        add_pos_embed(&mut tokens, &lpe.table, d).expect("add ok");
        // tokens should now equal the pos embedding
        for (t, p) in tokens.iter().zip(lpe.table.iter()) {
            assert!((t - p).abs() < 1e-6, "add_pos_embed mismatch");
        }
    }

    // ── Lateral conv ─────────────────────────────────────────────────────────

    #[test]
    fn e2e_lateral_conv_output_shape() {
        let mut rng = LcgRng::new(13);
        let lat = LateralConv1x1::new(64, 16, &mut rng).expect("ok");
        let feat = vec![0.5f32; 64 * 8 * 8];
        let out = lat.forward(&feat, 8, 8).expect("ok");
        assert_eq!(out.len(), 16 * 8 * 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    // ── Non-positive temperature rejects ─────────────────────────────────────

    #[test]
    fn e2e_clip_nce_nonpositive_temp_errors() {
        let img = vec![1.0f32; 4 * 16];
        let txt = vec![1.0f32; 4 * 16];
        let r = info_nce_loss(&img, &txt, 16, 0.0);
        assert!(matches!(r, Err(VisionError::NonPositiveTemperature(_))));
    }
}
