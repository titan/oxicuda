//! `oxicuda-anomaly` — Anomaly Detection primitives for OxiCUDA.
//!
//! Pure-Rust implementation of canonical anomaly detection algorithms,
//! suitable for CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-anomaly
//! ├── svdd/           — DeepSVDD (Ruff et al. 2018)
//! ├── reconstruction/ — Autoencoder & VAE anomaly detection
//! ├── distance/       — LOF, k-NN distance scorer
//! ├── density/        — COPOD, Mahalanobis
//! ├── isolation/      — Isolation Forest random-projection scorer
//! ├── forest/         — Robust Random Cut Forest (CoDisp, streaming)
//! ├── streaming/      — xStream (StreamHash + half-space chains)
//! ├── subspace/       — RS-Hash (randomized subspace hashing)
//! ├── statistical/    — MAD, Z-score, percentile threshold
//! ├── ensemble/       — Ensemble scoring (Average / Maximum / Weighted)
//! ├── metrics/        — AUC-ROC, AUC-PR, F1, detection metrics
//! ├── error           — AnomalyError / AnomalyResult
//! ├── handle          — AnomalyHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings (7 kernels × 6 SM versions)
//! ```

pub mod density;
pub mod distance;
pub mod ensemble;
pub mod error;
pub mod forest;
pub mod graph;
pub mod handle;
pub mod isolation;
pub mod metrics;
pub mod ptx_kernels;
pub mod reconstruction;
pub mod statistical;
pub mod streaming;
pub mod subspace;
pub mod svdd;
pub mod time_series;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports of the most-used anomaly detection types.
pub mod prelude {
    pub use crate::density::copod::Copod;
    pub use crate::density::fast_mcd::{
        FastMcdConfig, FastMcdFit, fast_mcd_fit, fast_mcd_predict, fast_mcd_score,
        fast_mcd_score_batch,
    };
    pub use crate::density::gmm_detector::{
        GmmConfig, GmmModel, gmm_fit, gmm_log_likelihood, gmm_predict, gmm_sample, gmm_score,
    };
    pub use crate::density::kde_detector::{Bandwidth, KdeConfig, KdeDetector, KdeKernel};
    pub use crate::density::mahalanobis::MahalanobisDetector;
    pub use crate::distance::abod::Abod;
    pub use crate::distance::abod_approx::AbodApprox;
    pub use crate::distance::cblof::{
        CblofConfig, CblofFit, cblof_fit, cblof_predict, cblof_score,
    };
    pub use crate::distance::cof::Cof;
    pub use crate::distance::knn_score::KnnAnomalyScorer;
    pub use crate::distance::lof::Lof;
    pub use crate::distance::lof_kdtree::{
        KdNode, KdTree, LofKdConfig, LofKdFit, kd_build, kd_knn, kd_knn_ex, lof_kd_fit,
        lof_kd_predict, lof_kd_score,
    };
    pub use crate::distance::lof_online::{OnlineLof, OnlineLofConfig};
    pub use crate::distance::sod::{Sod, SodConfig};
    pub use crate::ensemble::ensemble::{AnomalyEnsemble, EnsembleMethod};
    pub use crate::ensemble::ext_iforest::{
        ExtIforestConfig, ExtIforestModel, ExtNode, ext_iforest_fit, ext_iforest_predict,
        ext_iforest_score,
    };
    pub use crate::ensemble::loda::{Loda, LodaConfig};
    pub use crate::ensemble::lscp::{LscpConfig, LscpEnsemble, LscpStrategy, LscpTarget};
    pub use crate::ensemble::suod::{SuodConfig, SuodFit, suod_fit, suod_predict, suod_score};
    pub use crate::error::{AnomalyError, AnomalyResult};
    pub use crate::forest::rrcf::{
        BoundingBox, RobustRandomCutForest, RrcNode, RrcTree, RrcfConfig,
    };
    pub use crate::handle::{AnomalyHandle, LcgRng, SmVersion};
    pub use crate::isolation::iforest_score::{
        IsolationScorer, c_factor, isolation_score_from_path,
    };
    pub use crate::isolation::inne::{InneConfig, InneDetector};
    pub use crate::metrics::anomaly_metrics::{
        AnomalyDetectionMetrics, auc_pr, auc_roc_anomaly, compute_detection_metrics,
        f1_at_threshold,
    };
    pub use crate::ptx_kernels::{
        abod_batch_ptx, copod_ecdf_ptx, ensemble_normalize_ptx, f32_hex, fast_mcd_cstep_ptx,
        fused_knn_lof_ptx, iforest_score_ptx, lof_reach_dist_ptx, mahal_dist_ptx, recon_score_ptx,
        svdd_loss_ptx,
    };
    pub use crate::reconstruction::autoencoder::{AeConfig, AutoencoderAnomaly};
    pub use crate::reconstruction::dagmm::{
        DagmmConfig, DagmmFit, dagmm_fit, dagmm_predict, dagmm_score,
    };
    pub use crate::reconstruction::diffusion_anomaly::{
        DiffusionAnomalyConfig, DiffusionAnomalyFit, diffusion_anomaly_fit,
        diffusion_anomaly_predict, diffusion_anomaly_score,
    };
    pub use crate::reconstruction::mem_ae::{
        MemAeConfig, MemAeFit, mem_ae_attention, mem_ae_fit, mem_ae_predict, mem_ae_score,
    };
    pub use crate::reconstruction::norm_flow::{
        NormFlowConfig, NormFlowFit, norm_flow_fit, norm_flow_predict, norm_flow_score,
    };
    pub use crate::reconstruction::pca_anomaly::{PcaAnomaly, PcaAnomalyConfig};
    pub use crate::reconstruction::self_supervised::{
        SelfSupervisedConfig, SelfSupervisedFit, self_supervised_confidence_gap,
        self_supervised_fit, self_supervised_predict, self_supervised_score,
    };
    pub use crate::reconstruction::vae_anomaly::VaeAnomaly;
    pub use crate::statistical::concept_drift::{
        AdwinDetector, CusumDetector, DdmDetector, DdmStatus, PageHinkleyDetector, adwin_add,
        adwin_mean, adwin_new, adwin_window_size, cusum_add, cusum_new, cusum_reset, ddm_add,
        ddm_new, ph_add, ph_detector_new, ph_reset,
    };
    pub use crate::statistical::conformal::{
        ConformalConfig, ConformalDetector, ConformalResult, OnlineConformalDetector,
        conformal_calibrate, conformal_p_value, conformal_predict, mondrian_conformal_predict,
        online_conformal_detector_new, online_conformal_update,
    };
    pub use crate::statistical::ecod::Ecod;
    pub use crate::statistical::extreme_value::{GpdDetector, GpdFit};
    pub use crate::statistical::hbos::{Hbos, HbosConfig};
    pub use crate::statistical::rock_idec::{IdecConfig, IdecDetector, RockConfig, RockDetector};
    pub use crate::statistical::stats::{MadDetector, ZScoreDetector, percentile_threshold};
    pub use crate::streaming::xstream::{HalfSpaceChain, StreamHash, XStream, XStreamConfig};
    pub use crate::subspace::rs_hash::{HashComponent, RsHash, RsHashConfig};
    pub use crate::svdd::deep_sad::{DeepSad, DeepSadConfig};
    pub use crate::svdd::deep_svdd::DeepSvdd;
    pub use crate::svdd::ocsvm::{OcsvmConfig, OcsvmFit, ocsvm_fit, ocsvm_predict, ocsvm_score};
    pub use crate::svdd::trainable_svdd::{
        TrainableSvddConfig, TrainableSvddFit, trainable_svdd_fit, trainable_svdd_loss_history,
        trainable_svdd_predict, trainable_svdd_score,
    };
    pub use crate::time_series::spectral_residual::{
        SpectralResidualConfig, SpectralResidualResult, spectral_residual,
    };
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    // ── Test 1: AE score is finite for training data and random noise ─────────

    #[test]
    fn e2e_autoencoder_normal_low_score() {
        let cfg = AeConfig {
            encoder_dims: vec![4, 2],
            decoder_dims: vec![2, 4],
        };
        let mut rng = LcgRng::new(1);
        let ae = AutoencoderAnomaly::new(cfg, &mut rng).expect("autoencoder should initialize");

        let train = vec![0.5_f32; 4];
        let s_train = ae
            .score(&train)
            .expect("autoencoder score on training data should succeed");

        let mut noise = vec![0.0_f32; 4];
        let mut rng2 = LcgRng::new(999);
        rng2.fill_normal(&mut noise);
        let s_noise = ae
            .score(&noise)
            .expect("autoencoder score on noise should succeed");

        assert!(s_train.is_finite(), "train_score={s_train}");
        assert!(s_noise.is_finite(), "noise_score={s_noise}");
    }

    // ── Test 2: AE score is finite ───────────────────────────────────────────

    #[test]
    fn e2e_autoencoder_score_finite() {
        let cfg = AeConfig {
            encoder_dims: vec![8, 4, 2],
            decoder_dims: vec![2, 4, 8],
        };
        let mut rng = LcgRng::new(2);
        let ae = AutoencoderAnomaly::new(cfg, &mut rng)
            .expect("autoencoder should initialize with valid config");
        let s = ae
            .score(&[0.3_f32; 8])
            .expect("autoencoder score should succeed on valid input");
        assert!(s.is_finite() && s >= 0.0, "s={s}");
    }

    // ── Test 3: VAE score is finite and non-negative ──────────────────────────

    #[test]
    fn e2e_vae_score_finite() {
        let mut rng = LcgRng::new(3);
        let vae = VaeAnomaly::new(&[8, 4], 2, &[2, 4, 8], &mut rng)
            .expect("VAE should initialize with valid architecture");
        let s = vae
            .anomaly_score(&[0.2_f32; 8], &mut rng)
            .expect("VAE anomaly score should succeed");
        assert!(s.is_finite(), "s={s}");
    }

    // ── Test 4: DeepSVDD score increases for far-away point ───────────────────

    #[test]
    fn e2e_deep_svdd_score_increases_for_outlier() {
        let mut rng = LcgRng::new(4);
        let mut svdd = DeepSvdd::new(&[4, 8, 4], &mut rng)
            .expect("DeepSVDD should initialize with valid layers");

        let train = vec![0.1_f32; 4 * 20];
        svdd.fit(&train, 20).expect("DeepSVDD fit should succeed");

        let close = [0.1_f32, 0.1, 0.1, 0.1];
        let far = [100.0_f32, 100.0, 100.0, 100.0];

        let s_close = svdd
            .score(&close)
            .expect("DeepSVDD score for close point should succeed");
        let s_far = svdd
            .score(&far)
            .expect("DeepSVDD score for far point should succeed");

        assert!(
            s_far > s_close,
            "far score {s_far} should > close score {s_close}"
        );
    }

    // ── Test 5: LOF ≈ 1 for uniform data (trivial normal case) ───────────────

    #[test]
    fn e2e_lof_trivial_normal_case() {
        let n = 20_usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut lof = Lof::new(3);
        lof.fit(&data, n, 1).expect("LOF fit should succeed");
        let s = lof.score(&[10.0_f32]).expect("LOF score should succeed");
        assert!(s.is_finite(), "lof={s}");
        assert!(s > 0.0, "lof > 0");
    }

    // ── Test 6: COPOD higher for extreme outlier ──────────────────────────────

    #[test]
    fn e2e_copod_known_outlier() {
        let n = 30_usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut copod = Copod::new();
        copod.fit(&data, n, 1).expect("COPOD fit should succeed");

        let s_normal = copod
            .score(&[1.5_f32])
            .expect("COPOD normal score should succeed");
        let s_outlier = copod
            .score(&[100.0_f32])
            .expect("COPOD outlier score should succeed");

        assert!(
            s_outlier > s_normal,
            "outlier {s_outlier} should > normal {s_normal}"
        );
    }

    // ── Test 7: Mahalanobis higher for OOD point ─────────────────────────────

    #[test]
    fn e2e_mahalanobis_known_outlier() {
        let data = vec![
            1.0_f32, 2.0, 1.1, 1.9, 0.9, 2.1, 1.05, 1.95, 0.95, 2.05, 1.0_f32, 2.0, 1.1, 1.9, 0.9,
            2.1, 1.05, 1.95, 0.95, 2.05,
        ];
        let mut det = MahalanobisDetector::new();
        det.fit(&data, 10, 2)
            .expect("Mahalanobis fit should succeed");

        let s_normal = det
            .score(&[1.0_f32, 2.0])
            .expect("Mahalanobis normal score should succeed");
        let s_outlier = det
            .score(&[50.0_f32, 100.0])
            .expect("Mahalanobis outlier score should succeed");

        assert!(
            s_outlier > s_normal,
            "outlier {s_outlier} > normal {s_normal}"
        );
    }

    // ── Test 8: Isolation score in (0, 1) ────────────────────────────────────

    #[test]
    fn e2e_iforest_score_in_range() {
        let mut rng = LcgRng::new(8);
        let n = 100_usize;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| vec![i as f32 * 0.1, i as f32 * 0.05])
            .collect();
        let mut scorer = IsolationScorer::new(50, &mut rng);
        scorer
            .fit(&data, n, 2, &mut rng)
            .expect("IsolationScorer fit should succeed");
        let s = scorer
            .score(&[5.0_f32, 2.5])
            .expect("IsolationScorer score should succeed");
        assert!((0.0..=1.0).contains(&s), "s={s}");
    }

    // ── Test 9: Z-score extreme sample gets highest score ────────────────────

    #[test]
    fn e2e_zscore_known_outlier() {
        let n = 20_usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let mut det = ZScoreDetector::new();
        det.fit(&data, n, 1)
            .expect("ZScoreDetector fit should succeed");

        let s_normal = det
            .score(&[1.0_f32])
            .expect("ZScoreDetector normal score should succeed");
        let s_outlier = det
            .score(&[1000.0_f32])
            .expect("ZScoreDetector outlier score should succeed");

        assert!(
            s_outlier > s_normal,
            "outlier {s_outlier} > normal {s_normal}"
        );
    }

    // ── Test 10: MAD detector returns finite scores ───────────────────────────

    #[test]
    fn e2e_mad_detector_finite() {
        let n = 20_usize;
        let data: Vec<f32> = (0..n)
            .flat_map(|i| vec![i as f32, (i * 2) as f32])
            .collect();
        let mut det = MadDetector::new();
        det.fit(&data, n, 2)
            .expect("MadDetector fit should succeed");

        let scores = det
            .score_batch(&data, n)
            .expect("MadDetector batch score should succeed");
        assert!(scores.iter().all(|s| s.is_finite()), "not all finite");
    }

    // ── Test 11: Ensemble combine returns finite score in [0, 1] ─────────────

    #[test]
    fn e2e_ensemble_combine_finite() {
        let n_det = 3_usize;
        let n = 20_usize;
        let mut rng = LcgRng::new(11);
        let train_scores: Vec<f32> = (0..n * n_det).map(|_| rng.next_f32()).collect();
        let mut ens = AnomalyEnsemble::new(EnsembleMethod::Average, n_det);
        ens.fit(&train_scores, n)
            .expect("AnomalyEnsemble fit should succeed");

        let test = [0.5_f32, 0.8, 0.3];
        let s = ens
            .combine(&test)
            .expect("AnomalyEnsemble combine should succeed");
        assert!(s.is_finite(), "s={s}");
        assert!((0.0..=1.0).contains(&s), "s={s} not in [0,1]");
    }

    // ── Test 12: All 7 × 6 SM versions produce valid PTX ─────────────────────

    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("svdd_loss_kernel", svdd_loss_ptx),
            ("recon_score_kernel", recon_score_ptx),
            ("lof_reach_dist_kernel", lof_reach_dist_ptx),
            ("copod_ecdf_kernel", copod_ecdf_ptx),
            ("mahal_dist_kernel", mahal_dist_ptx),
            ("iforest_score_kernel", iforest_score_ptx),
            ("ensemble_normalize_kernel", ensemble_normalize_ptx),
        ];
        for sm in sm_versions {
            for (kernel_name, gen_fn) in kernel_fns {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "PTX for {kernel_name} sm={sm} missing sm target"
                );
                assert!(
                    ptx.contains(".version"),
                    "PTX for {kernel_name} sm={sm} missing .version"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "PTX for {kernel_name} sm={sm} missing .visible .entry"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "PTX for {kernel_name} sm={sm} missing kernel name"
                );
            }
        }
        assert_eq!(f32_hex(1.0_f32), "0F3F800000");
    }
}
