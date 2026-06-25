//! `oxicuda-ssl` — Self-supervised learning primitives for OxiCUDA.
//!
//! Pure-Rust implementation of the four canonical SSL families, suitable for
//! CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-ssl
//! ├── contrastive/      — SimCLR (NT-Xent), MoCo (memory-bank InfoNCE)
//! ├── non_contrastive/  — BYOL (cosine), Barlow Twins, VICReg
//! ├── masked/           — MAE (random patch mask + reconstruction MSE)
//! ├── clustering/       — SwAV (Sinkhorn-Knopp), DINO (centred + sharpened CE)
//! ├── augment/          — Color jitter, multi-crop helpers
//! ├── metrics/          — Uniformity, alignment, effective rank, collapse score
//! ├── momentum/         — EmaUpdater for momentum-encoder schemes
//! ├── head/             — MlpProjector, PredictorHead
//! ├── error             — SslError / SslResult
//! ├── handle            — SslHandle (SmVersion + LcgRng)
//! └── ptx_kernels       — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod augment;
pub mod clustering;
pub mod contrastive;
pub mod error;
pub mod handle;
pub mod head;
pub mod masked;
pub mod metrics;
pub mod momentum;
pub mod non_contrastive;
pub mod ptx_kernels;
pub mod ssl;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common SSL types.
pub mod prelude {
    pub use crate::augment::color::{color_jitter, random_grayscale_chw};
    pub use crate::augment::multi_crop::{MultiCropConfig, multi_crop};
    pub use crate::augment::rand_augment::{
        AugOp, AutoAugPolicy, AutoAugmentConfig, RandAugmentConfig, SubPolicy, all_aug_ops,
        apply_aug_op, auto_augment, rand_augment,
    };
    pub use crate::augment::solarize_blur::{
        SimClrBlurSolarConfig, add_gaussian_noise, gaussian_blur_chw, random_gaussian_blur_chw,
        random_solarize, simclr_blur_solar, solarize,
    };
    pub use crate::clustering::deep_cluster::{
        DeepClusterConfig, DeepClusterResult, DeeperClusterConfig, DeeperClusterResult,
        deep_cluster, deep_cluster_loss, deeper_cluster, pca_whiten,
    };
    pub use crate::clustering::dino::{DinoConfig, dino_loss};
    pub use crate::clustering::dino_v2::{DinoV2, DinoV2Config};
    pub use crate::clustering::ibot::{
        IBotCenters, IBotConfig, IBotResult, ibot_centers_init, ibot_cls_loss, ibot_loss,
        ibot_mim_loss, ibot_random_patch_mask, ibot_update_centers,
    };
    pub use crate::clustering::swav::{SwavConfig, sinkhorn_knopp, swav_loss};
    pub use crate::contrastive::info_nce::info_nce_loss;
    pub use crate::contrastive::moco::{MocoQueue, moco_loss};
    pub use crate::contrastive::moco_v3::{
        MocoV3Config, MocoV3State, moco_v3_loss, moco_v3_symmetric_loss,
    };
    pub use crate::contrastive::simclr::{SimClrConfig, simclr_loss};
    pub use crate::error::{SslError, SslResult};
    pub use crate::handle::{LcgRng, SmVersion, SslHandle};
    pub use crate::head::linear_probe::{
        FittedLinearProbe, LinearProbeConfig, LinearProbeResult, linear_probe_eval,
        linear_probe_fit, linear_probe_predict,
    };
    pub use crate::head::predictor::PredictorHead;
    pub use crate::head::projector::MlpProjector;
    pub use crate::masked::beit::{
        BeitConfig, BeitResult, VqCodebook, beit_block_mask, beit_loss, vq_codebook_init,
        vq_encode, vq_update_codebook,
    };
    pub use crate::masked::data2vec::{
        Data2VecConfig, Data2VecResult, Data2VecState, data2vec_batch_loss, data2vec_loss,
        data2vec_mask, huber_loss, normalize_teacher_targets,
    };
    pub use crate::masked::i_jepa::{IJepa, IJepaConfig};
    pub use crate::masked::mae::{MaeConfig, mae_reconstruction_loss, random_patch_mask};
    pub use crate::masked::simmim::{
        SimMimConfig, simmim_block_mask, simmim_l1_loss, simmim_l2_loss, simmim_random_mask,
        simmim_reconstruction_loss,
    };
    pub use crate::metrics::feature_metrics::{
        alignment_loss, collapse_score, effective_rank, pairwise_cosine_stats, uniformity_loss,
    };
    pub use crate::metrics::knn_eval::{KnnEvalConfig, KnnEvalResult, knn_eval};
    pub use crate::momentum::ema::{EmaUpdater, cosine_momentum};
    pub use crate::non_contrastive::barlow::{BarlowTwinsConfig, barlow_twins_loss};
    pub use crate::non_contrastive::byol::{ByolPredictor, byol_loss};
    pub use crate::non_contrastive::dense_cl::{
        DenseCLConfig, DenseCLResult, PixProConfig, dense_cl_loss, dense_correspondence,
        dense_infonce, pixpro_loss,
    };
    pub use crate::non_contrastive::msn::{
        MsnConfig, MsnPrototypes, MsnResult, msn_loss, msn_prototype_init, msn_random_mask,
        msn_update_prototypes,
    };
    pub use crate::non_contrastive::simsiam::{
        SimSiamConfig, SimSiamPredictor, is_collapsed, simsiam_loss, simsiam_loss_batch,
    };
    pub use crate::non_contrastive::vicreg::{VicRegConfig, vicreg_loss};
    pub use crate::ptx_kernels::{
        barlow_cross_corr_ptx, barlow_cross_corr_wgmma_ptx, byol_cosine_loss_bf16_ptx,
        byol_cosine_loss_ptx, cosine_similarity_ptx, f32_hex, gather_features_bulk_ptx,
        gather_features_ptx, momentum_update_f16_ptx, momentum_update_ptx, nt_xent_softmax_ptx,
        nt_xent_softmax_warp_ptx, random_mask_ptx,
    };
    pub use crate::ssl::data2vec_v2::{Data2VecModel, Data2VecModelConfig};
    pub use crate::ssl::jem::{Jem, JemConfig};
    pub use crate::ssl::sim_siam::{SimSiam, SimSiamConfig as SimSiamStructConfig};
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    /// Build a "perfectly aligned" projection batch where each row is a
    /// distinct one-hot basis vector — diagonal cosine = 1, off-diagonal = 0.
    fn aligned_projections(n: usize, d: usize) -> Vec<f32> {
        let mut z = vec![0.0_f32; n * d];
        for i in 0..n {
            z[i * d + i % d] = 1.0;
        }
        z
    }

    #[test]
    fn e2e_simclr_loss_drops_with_aligned_pairs() {
        let n = 8;
        let d = 16;
        let z = aligned_projections(n, d);
        let cfg = SimClrConfig::default();
        let (loss, acc) = simclr_loss(&z, &z, n, d, &cfg).expect("simclr_loss should succeed");
        assert!(loss.is_finite() && loss < 1.0, "loss = {loss}");
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn e2e_moco_queue_lifecycle_fifo() {
        let mut q = MocoQueue::new(8, 4).expect("new should succeed");
        for batch_id in 0..6 {
            let mut batch = vec![0.0_f32; 4];
            batch[batch_id % 4] = 1.0;
            q.enqueue(&batch).expect("enqueue should succeed");
        }
        assert_eq!(q.len(), 6);
        // Run MoCo loss with a meaningful query/key pair.
        let q_vec = vec![1.0_f32, 0.0, 0.0, 0.0];
        let k_vec = q_vec.clone();
        let l = moco_loss(&q_vec, &k_vec, 1, 4, &q, 0.1).expect("moco_loss should succeed");
        assert!(l.is_finite());
    }

    #[test]
    fn e2e_byol_loss_zero_for_identical_inputs() {
        let z = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let l = byol_loss(&z, &z, 2, 3).expect("byol_loss should succeed");
        assert!(l.abs() < 1e-4);
    }

    #[test]
    fn e2e_barlow_twins_low_for_identical_inputs() {
        let n = 16;
        let d = 4;
        // Each column has distinct mean → standardisation makes columns
        // independent. Identical Z_A = Z_B → diag(C) ≈ 1.
        let mut z = vec![0.0_f32; n * d];
        for i in 0..n {
            for j in 0..d {
                z[i * d + j] = (i as f32) * 0.1 + (j as f32) * 0.7;
            }
        }
        let cfg = BarlowTwinsConfig::default();
        let l = barlow_twins_loss(&z, &z, n, d, &cfg).expect("barlow_twins_loss should succeed");
        assert!(l.is_finite());
    }

    #[test]
    fn e2e_vicreg_three_terms_combine() {
        let n = 16;
        let d = 4;
        let z_a: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.013).sin()).collect();
        let z_b: Vec<f32> = (0..n * d)
            .map(|i| (i as f32 * 0.013).sin() + 0.01)
            .collect();
        let cfg = VicRegConfig::default();
        let l = vicreg_loss(&z_a, &z_b, n, d, &cfg).expect("vicreg_loss should succeed");
        assert!(l.is_finite() && l > 0.0);
    }

    #[test]
    fn e2e_mae_mask_ratio_respected() {
        let mut handle = SslHandle::default_handle();
        let mask = random_patch_mask(196, 0.75, handle.rng_mut()).expect("value should be present");
        let n_masked = mask.iter().filter(|&&v| v == 0.0).count();
        assert_eq!(n_masked, 147); // floor(196 * 0.75)
        // Reconstruction MSE on a perfect predictor is zero.
        let target = vec![1.5_f32; 196 * 4];
        let pred = target.clone();
        let l = mae_reconstruction_loss(&target, &pred, &mask, 196, 4)
            .expect("mae_reconstruction_loss should succeed");
        assert!(l.abs() < 1e-7);
    }

    #[test]
    fn e2e_swav_sinkhorn_normalises_uniform() {
        let n = 8;
        let k = 4;
        let mut q = vec![1.0_f32; n * k];
        sinkhorn_knopp(&mut q, n, k, 5).expect("sinkhorn_knopp should succeed");
        // After Sinkhorn, each row sums to 1 and is uniform.
        for i in 0..n {
            let s: f32 = q[i * k..(i + 1) * k].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row sum = {s}");
        }
    }

    #[test]
    fn e2e_dino_centred_softmax_returns_finite() {
        let n = 4;
        let k = 8;
        let mut handle = SslHandle::default_handle();
        let mut s = vec![0.0_f32; n * k];
        let mut t = vec![0.0_f32; n * k];
        handle.rng_mut().fill_normal(&mut s);
        handle.rng_mut().fill_normal(&mut t);
        let centre = vec![0.0_f32; k];
        let cfg = DinoConfig::default();
        let l = dino_loss(&s, &t, &centre, n, k, &cfg).expect("dino_loss should succeed");
        assert!(l.is_finite() && l > 0.0);
    }

    #[test]
    fn e2e_ema_converges_to_online_when_momentum_zero() {
        let mut updater = EmaUpdater::new();
        let mut target = vec![5.0_f32; 8];
        let online = vec![10.0_f32; 8];
        updater
            .update(&mut target, &online, 0.0)
            .expect("update should succeed");
        for &v in &target {
            assert!((v - 10.0).abs() < 1e-6);
        }
        // cosine_momentum is monotone increasing.
        let m1 = cosine_momentum(0, 100, 0.5, 1.0).expect("cosine_momentum should succeed");
        let m2 = cosine_momentum(100, 100, 0.5, 1.0).expect("cosine_momentum should succeed");
        assert!(m1 < m2);
    }

    #[test]
    fn e2e_mlp_projector_forward_correct_shape() {
        let mut handle = SslHandle::default_handle();
        let p = MlpProjector::new(64, 32, 16, handle.rng_mut()).expect("value should be present");
        let x = vec![0.1_f32; 64];
        let y = p.forward(&x).expect("forward should succeed");
        assert_eq!(y.len(), 16);
        // Predictor head similar interface
        let pred =
            PredictorHead::new(16, 32, 16, handle.rng_mut()).expect("value should be present");
        let y2 = pred.forward(&y).expect("forward should succeed");
        assert_eq!(y2.len(), 16);
    }

    #[test]
    fn e2e_multi_crop_returns_n_crops() {
        let cfg = MultiCropConfig::default();
        let crops = multi_crop(&cfg).expect("multi_crop should succeed");
        assert_eq!(crops.len(), cfg.n_crops());
        // First two are global.
        assert!(crops[0].is_global);
        assert!(crops[1].is_global);
        // Color jitter on a sample image runs without error.
        let mut handle = SslHandle::default_handle();
        let h = 8;
        let w = 8;
        let mut img = vec![0.5_f32; 3 * h * w];
        color_jitter(&mut img, h, w, 0.5, handle.rng_mut()).expect("value should be present");
        let _converted = random_grayscale_chw(&mut img, h, w, 0.5, handle.rng_mut())
            .expect("value should be present");
        for v in &img {
            assert!((0.0..=1.0).contains(v));
        }
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            for prog in [
                nt_xent_softmax_ptx(sm),
                momentum_update_ptx(sm),
                byol_cosine_loss_ptx(sm),
                barlow_cross_corr_ptx(sm),
                random_mask_ptx(sm),
                cosine_similarity_ptx(sm),
                gather_features_ptx(sm),
            ] {
                assert!(prog.contains(&format!("sm_{sm}")));
                assert!(prog.contains(".visible .entry"));
            }
        }
        // Smoke-test f32_hex to keep the prelude path live.
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
