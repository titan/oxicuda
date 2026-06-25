//! # oxicuda-tabular
//!
//! Tabular deep learning primitives for OxiCUDA.
//!
//! Covers TabNet, FT-Transformer, SAINT, NODE, GBDT, Random Forest, Extra
//! Trees, sparsemax, entmax-1.5, quantile / standard / min-max normalisation,
//! Weight-of-Evidence encoding, and evaluation metrics.

pub mod attention;
pub mod calibration;
pub mod conformal;
pub mod danet;
pub mod deepgbm;
pub mod diffusion;
pub mod error;
pub mod feature_select;
pub mod federated;
pub mod gan;
pub mod handle;
pub mod metrics;
mod nn;
pub mod preprocess;
pub mod ptx_kernels;
pub mod transformer;
pub mod tree;
pub mod vae;

/// Convenience re-exports for common types and functions.
pub mod prelude {
    pub use crate::attention::saint::{SaintConfig, SaintLayer};
    pub use crate::attention::sparsemax::{entmax15, sparsemax, sparsemax_batch};
    pub use crate::attention::tabnet::{BatchNorm1d, TabNetConfig, TabNetLayer, glu};
    pub use crate::calibration::isotonic::IsotonicCalibrator;
    pub use crate::conformal::aps_conformal::{ApsConformal, ApsConformalConfig};
    pub use crate::conformal::split_conformal::{
        ClassifierScore, ConformalConfig, ConformalizedQuantileRegressor, SplitConformalClassifier,
        SplitConformalRegressor, empirical_quantile,
    };
    pub use crate::danet::{AbstractLayer, Danet, DanetConfig};
    pub use crate::deepgbm::{DeepGbm, DeepGbmConfig};
    pub use crate::diffusion::tabddpm::{TabDdpm, TabDdpmConfig};
    pub use crate::error::{TabularError, TabularResult};
    pub use crate::feature_select::stg::{StgConfig, StgModel};
    pub use crate::federated::{
        SecureAggregator, Shard, fed_avg, fed_avg_uniform, fedprox_gradient, fedprox_proximal,
        horizontal_split, vertical_split,
    };
    pub use crate::gan::ctgan::{
        ColumnModes, ConditionalSampler, CtGan, CtganConfig, ModeNormalizer,
    };
    pub use crate::handle::{LcgRng, SmVersion, TabularHandle};
    pub use crate::metrics::tabular_metrics::{
        ClassificationMetrics, auc_roc, binary_accuracy, compute_binary_metrics, mae,
        multiclass_accuracy, rmse,
    };
    pub use crate::preprocess::augment::{
        CutMixConfig, MixupConfig, cutmix_batch, cutmix_pair, mixup_batch, mixup_pair,
    };
    pub use crate::preprocess::concept_drift::{
        AdwinTabular, AdwinTabularConfig, DriftStatus, KsDriftDetector, PageHinkleyTabular,
        PageHinkleyTabularConfig,
    };
    pub use crate::preprocess::embed::FeatureEmbedder;
    pub use crate::preprocess::normalize::{
        MinMaxNormalizer, QuantileNormalizer, StandardNormalizer,
    };
    pub use crate::preprocess::quantile_feat::{
        QuantileDist, QuantileTransformer, probit, std_normal_cdf,
    };
    pub use crate::preprocess::ssl_pretrain::{
        ScarfConfig, ScarfModel, VimeConfig, VimeModel, VimeSample,
    };
    pub use crate::preprocess::target_encode::{TargetEncoder, TargetEncoderConfig};
    pub use crate::preprocess::woe::{WoeEncoder, information_value};
    pub use crate::ptx_kernels::{
        auc_roc_ptx, f32_hex, feature_tokenize_ptx, intersample_attn_ptx, node_tree_eval_ptx,
        quantile_norm_ptx, sparsemax_ptx, tabnet_step_attn_ptx,
    };
    pub use crate::transformer::ft_rope::{FtRopeConfig, FtRopeTransformer};
    pub use crate::transformer::ft_transformer::{FeatureTokenizer, FtConfig, FtTransformer};
    pub use crate::transformer::ft_transformer_v2::{
        FtTransformer as FtTransformerV2, FtTransformerConfig as FtTransformerV2Config,
    };
    pub use crate::transformer::tabpfn::{TabPfn, TabPfnConfig};
    pub use crate::tree::extra_trees::{ExtraTrees, ExtraTreesConfig};
    pub use crate::tree::node::{NodeConfig, NodeEnsemble, NodeTree};
    pub use crate::tree::node_oblivious::{
        EnsembleReduction, NodeObliviousConfig, NodeObliviousLayer, ObliviousTree,
        entmax_alpha_f64, entmoid_alpha_f64, sparsemax_f64,
    };
    pub use crate::tree::random_forest::{ForestTask, RandomForest, RandomForestConfig};
    pub use crate::vae::tvae::{Tvae, TvaeConfig, kl_divergence_standard};
}

// ─── End-to-end tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    // ── 1. sparsemax sums to 1.0 ──────────────────────────────────────────────
    #[test]
    fn e2e_sparsemax_sums_to_one() {
        let inputs: &[&[f32]] = &[
            &[0.1, 0.5, 0.3, 0.1],
            &[-1.0, 2.0, 3.0, -0.5],
            &[0.0, 0.0, 0.0, 0.0],
            &[100.0, 0.0, 0.0, 0.0],
        ];
        for &z in inputs {
            let out = sparsemax(z).expect("sparsemax should succeed");
            let s: f32 = out.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "sum={s} for input {z:?}");
        }
    }

    // ── 2. sparsemax one-hot for dominated input ──────────────────────────────
    #[test]
    fn e2e_sparsemax_sparse_for_dominated() {
        // Large first element should produce near-one-hot output
        let z = [50.0_f32, 0.0, 0.0, 0.0];
        let out = sparsemax(&z).expect("sparsemax should succeed");
        assert!((out[0] - 1.0).abs() < 1e-5, "expected one-hot, got {out:?}");
        assert!(out[1..].iter().all(|&v| v < 1e-5));
    }

    // ── 3. entmax15 sums to 1.0 ───────────────────────────────────────────────
    #[test]
    fn e2e_entmax15_sums_to_one() {
        let inputs: &[&[f32]] = &[
            &[0.1, 0.5, 0.3, 0.1],
            &[-1.0, 2.0, 3.0, -0.5],
            &[1.0, 2.0, 3.0, 4.0],
        ];
        for &z in inputs {
            let out = entmax15(z).expect("entmax15 should succeed");
            let s: f32 = out.iter().sum();
            assert!((s - 1.0).abs() < 1e-2, "sum={s} for input {z:?}");
        }
    }

    // ── 4. GLU halves dimension ───────────────────────────────────────────────
    #[test]
    fn e2e_glu_halves_dim() {
        for input_dim in [4, 8, 16, 32] {
            let x = vec![0.5_f32; input_dim];
            let out = glu(&x).expect("glu should succeed");
            assert_eq!(out.len(), input_dim / 2, "input_dim={input_dim}");
        }
    }

    // ── 5. TabNet forward output shape ────────────────────────────────────────
    #[test]
    fn e2e_tabnet_forward_shape() {
        let mut rng = LcgRng::new(42);
        let n_classes = 3;
        let cfg = TabNetConfig {
            n_features: 16,
            n_d: 8,
            n_a: 8,
            n_steps: 4,
            gamma: 1.5,
            n_classes,
        };
        let layer = TabNetLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.3_f32; 16];
        let (logits, _masks) = layer.forward(&x).expect("forward should succeed");
        assert_eq!(logits.len(), n_classes);
    }

    // ── 6. TabNet attention masks non-negative ────────────────────────────────
    #[test]
    fn e2e_tabnet_attention_valid() {
        let mut rng = LcgRng::new(77);
        let n_steps = 5;
        let n_features = 12;
        let cfg = TabNetConfig {
            n_features,
            n_d: 4,
            n_a: 4,
            n_steps,
            gamma: 1.3,
            n_classes: 2,
        };
        let layer = TabNetLayer::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.1_f32; n_features];
        let (_, masks) = layer.forward(&x).expect("forward should succeed");
        assert_eq!(masks.len(), n_steps * n_features);
        assert!(
            masks.iter().all(|&v| v >= -1e-6),
            "attention masks must be non-negative"
        );
    }

    // ── 7. FT-Transformer forward returns finite logits ───────────────────────
    #[test]
    fn e2e_ft_transformer_forward_finite() {
        let mut rng = LcgRng::new(13);
        let cfg = FtConfig {
            n_cont_features: 4,
            cat_n_categories: vec![5, 3],
            embed_dim: 8,
            n_heads: 2,
            n_layers: 2,
            ffn_hidden: 16,
            dropout_rate: 0.0,
            n_classes: 3,
        };
        let model = FtTransformer::new(cfg, &mut rng).expect("new should succeed");
        let logits = model
            .forward(&[0.1, 0.2, 0.3, 0.4], &[1, 0])
            .expect("forward should succeed");
        assert_eq!(logits.len(), 3);
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "logits must be finite"
        );
    }

    // ── 8. FeatureTokenizer shape ─────────────────────────────────────────────
    #[test]
    fn e2e_feature_tokenizer_shape() {
        let n_cont = 4;
        let cat_sizes = vec![5usize, 3, 8];
        let embed_dim = 16;
        let mut rng = LcgRng::new(99);
        let tok = FeatureTokenizer::new(n_cont, &cat_sizes, embed_dim, &mut rng);
        let tokens = tok
            .tokenize(&[0.1, 0.2, 0.3, 0.4], &[2, 1, 7])
            .expect("tokenize should succeed");
        assert_eq!(
            tokens.len(),
            (n_cont + cat_sizes.len()) * embed_dim,
            "expected {} tokens × {embed_dim} dims",
            n_cont + cat_sizes.len()
        );
    }

    // ── 9. NODE forward output shape ──────────────────────────────────────────
    #[test]
    fn e2e_node_forward_shape() {
        let mut rng = LcgRng::new(31);
        let output_dim = 5;
        let cfg = NodeConfig {
            n_trees: 10,
            depth: 4,
            input_dim: 16,
            output_dim,
        };
        let ensemble = NodeEnsemble::new(cfg, &mut rng).expect("new should succeed");
        let x = vec![0.5_f32; 16];
        let out = ensemble.forward(&x).expect("forward should succeed");
        assert_eq!(out.len(), output_dim);
    }

    // ── 10. QuantileNormalizer: output in [0, 1] ──────────────────────────────
    #[test]
    fn e2e_quantile_normalizer_range() {
        let mut rng = LcgRng::new(55);
        let n_samples = 64;
        let n_features = 4;
        let mut data = vec![0.0_f32; n_samples * n_features];
        rng.fill_normal_scaled(&mut data, 2.0);

        let (norm, transformed) = QuantileNormalizer::fit_transform(&data, n_samples, n_features)
            .expect("fit_transform should succeed");
        assert!(
            transformed.iter().all(|&v| (0.0_f32..=1.0).contains(&v)),
            "transformed values must lie in [0, 1]"
        );

        // New sample should also be in range when within training distribution
        let row: Vec<f32> = data[0..n_features].to_vec();
        let t = norm.transform(&row).expect("transform should succeed");
        assert!(t.iter().all(|&v| (0.0_f32..=1.0).contains(&v)));
    }

    // ── 11. AUC-ROC perfect predictor = 1.0 ──────────────────────────────────
    #[test]
    fn e2e_auc_roc_perfect() {
        let scores = vec![0.95_f32, 0.87, 0.80, 0.30, 0.20, 0.10];
        let labels = vec![1u32, 1, 1, 0, 0, 0];
        let auc = auc_roc(&scores, &labels).expect("auc_roc should succeed");
        assert!(
            (auc - 1.0).abs() < 1e-5,
            "perfect predictor should have AUC=1, got {auc}"
        );
    }

    // ── 12. e2e target encoder binary task ───────────────────────────────────
    #[test]
    fn e2e_target_encoder_binary_task() {
        let x_cat = vec![0_usize, 0, 1, 1, 0, 1]; // 6 samples, 1 categorical feature
        let y = vec![0.0_f32, 0.0, 1.0, 1.0, 0.0, 1.0];
        let cfg = TargetEncoderConfig {
            k: 0.0,
            min_count: 1,
        };
        let enc = TargetEncoder::fit(&x_cat, &y, 6, 1, cfg).expect("fit should succeed");
        let encoded = enc.transform(&x_cat, 6).expect("transform should succeed");
        assert!((encoded[0] - 0.0_f32).abs() < 1e-5, "cat=0 → mean(y)=0");
        assert!((encoded[2] - 1.0_f32).abs() < 1e-5, "cat=1 → mean(y)=1");
    }

    // ── 13. e2e APS conformal coverage ───────────────────────────────────────
    #[test]
    fn e2e_aps_conformal_coverage() {
        // 3-class problem, 50 cal samples with near-perfect predictor
        let n_cal = 50_usize;
        let n_classes = 3_usize;
        // Perfect predictor: probs[true_class] = 0.9, others = 0.05
        let mut cal_probs = vec![0.0_f32; n_cal * n_classes];
        let mut cal_labels = vec![0_usize; n_cal];
        for i in 0..n_cal {
            let label = i % n_classes;
            cal_labels[i] = label;
            for c in 0..n_classes {
                cal_probs[i * n_classes + c] = if c == label { 0.9 } else { 0.05 };
            }
        }
        let cfg = ApsConformalConfig { alpha: 0.1 };
        let aps = ApsConformal::calibrate(&cal_probs, &cal_labels, n_cal, n_classes, cfg)
            .expect("calibrate should succeed");
        let cov = aps
            .coverage_rate(&cal_probs, &cal_labels, n_cal, n_classes)
            .expect("value should be present");
        assert!(cov >= 0.89, "coverage ≥ 1-alpha on cal set, got {cov}");
    }

    // ── 14. All 7 × 6 PTX kernel × SM-version combinations valid ─────────────
    #[test]
    #[allow(clippy::type_complexity)]
    fn e2e_ptx_kernels_all_sm_versions() {
        let sm_versions = [75_u32, 80, 86, 90, 100, 120];
        let kernel_fns: &[(&str, fn(u32) -> String)] = &[
            ("sparsemax_kernel", sparsemax_ptx),
            ("feature_tokenize_kernel", feature_tokenize_ptx),
            ("tabnet_step_attn_kernel", tabnet_step_attn_ptx),
            ("intersample_attn_kernel", intersample_attn_ptx),
            ("node_tree_eval_kernel", node_tree_eval_ptx),
            ("quantile_norm_kernel", quantile_norm_ptx),
            ("auc_roc_kernel", auc_roc_ptx),
        ];
        for &(kernel_name, gen_fn) in kernel_fns {
            for &sm in &sm_versions {
                let ptx = gen_fn(sm);
                assert!(
                    ptx.contains(&format!("sm_{sm}")),
                    "kernel {kernel_name} missing sm_{sm}"
                );
                assert!(
                    ptx.contains(".version"),
                    "kernel {kernel_name} missing .version"
                );
                assert!(
                    ptx.contains(".address_size 64"),
                    "kernel {kernel_name} missing .address_size 64"
                );
                assert!(
                    ptx.contains(".visible .entry"),
                    "kernel {kernel_name} missing .visible .entry"
                );
                assert!(
                    ptx.contains(kernel_name),
                    "kernel {kernel_name} name not found in PTX for sm_{sm}"
                );
            }
        }
    }
}
