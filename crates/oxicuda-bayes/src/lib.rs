//! `oxicuda-bayes` — Bayesian deep learning primitives for OxiCUDA.
//!
//! Pure-Rust implementation of variational inference and Bayesian neural network
//! building blocks suitable for CPU simulation and PTX kernel generation for GPU
//! execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-bayes
//! ├── layers/         — BayesLinear, BayesConv2d, Flipout layers
//! ├── variational/    — ELBO, normalizing flows, mean-field, reparameterization
//! ├── calibration/    — Temperature scaling, ECE/MCE/ACE, isotonic, Platt, Brier, NLL
//! ├── uncertainty/    — MC Dropout, Deep Ensembles, SWAG, last-layer Laplace, BALD
//! ├── error           — BayesError / BayesResult
//! ├── handle          — BayesHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod bayesopt;
pub mod calibration;
pub mod error;
pub mod flow;
pub mod gp;
pub mod handle;
pub mod layers;
pub mod mc;
pub mod mcmc;
pub mod ptx_kernels;
pub mod sparse;
pub mod uncertainty;
pub mod variational;
pub mod vi;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common Bayesian deep learning types.
pub mod prelude {
    pub use crate::bayesopt::bo::{
        AcquisitionFn, BayesOptConfig, BayesOptResult, acquisition_value, bayesopt,
    };
    pub use crate::calibration::beta::{BetaCalibConfig, BetaCalibrator};
    pub use crate::calibration::conformal::{
        ConformalClassifier, ConformalRegressor, RapsClassifier, conformal_quantile,
    };
    pub use crate::calibration::histogram::{
        BinStrategy, HistogramBinCalibrator, HistogramBinConfig,
    };
    pub use crate::calibration::isotonic::IsotonicRegressor;
    pub use crate::calibration::metrics::{
        ReliabilityBin, ReliabilityDiagram, adaptive_calibration_error, brier_score,
        expected_calibration_error, maximum_calibration_error, negative_log_likelihood,
        reliability_diagram, top1_confidences,
    };
    pub use crate::calibration::platt::{PlattFitConfig, PlattScaler};
    pub use crate::calibration::temperature::{TemperatureFitConfig, TemperatureScaler};
    pub use crate::calibration::vector_scaling::{ScalingMode, VectorScaler, VectorScalingConfig};
    pub use crate::error::{BayesError, BayesResult};
    pub use crate::flow::maf::{MafFlow, MafLayer, standard_normal_log_prob_vec};
    pub use crate::gp::deep_gp::{DeepGp, DeepGpConfig, DeepGpLayer, DeepGpLayerConfig};
    pub use crate::gp::gpr::{
        GprConfig, GprFit, GprKernel, gpr_fit, gpr_kernel_matrix, gpr_log_marginal_likelihood,
        gpr_predict,
    };
    pub use crate::gp::sparse_gp::{
        InducingInit, SparseGpConfig, SparseGpFit, sparse_gp_elbo, sparse_gp_fit, sparse_gp_predict,
    };
    pub use crate::handle::{BayesHandle, LcgRng, SmVersion};
    pub use crate::layers::bayes_conv::BayesConv2d;
    pub use crate::layers::bayes_gru::{BayesGru, BayesGruConfig, BayesGruState, BayesGruWeights};
    pub use crate::layers::bayes_linear::{BayesLinear, softplus};
    pub use crate::layers::bayes_lstm::{
        BayesLstm, BayesLstmConfig, BayesLstmSampledWeights, BayesLstmWeights,
    };
    pub use crate::layers::flipout::{FlipoutConv2d, FlipoutLinear};
    pub use crate::mc::convergence_diagnostics::{
        ConvergenceSummary, GewekeConfig, diagnose as mcmc_diagnose, geweke_z, multi_chain_ess,
        r_hat,
    };
    pub use crate::mc::smc::{
        SmcConfig, SmcState, effective_sample_size, smc_filter, smc_init, smc_mean, smc_step,
        smc_variance, systematic_resample,
    };
    pub use crate::mcmc::BayesRng;
    pub use crate::mcmc::hmc::{HmcConfig as McmcHmcConfig, HmcSampler};
    pub use crate::mcmc::sgld::{SgldConfig, SgldSampler};
    pub use crate::ptx_kernels::{
        ece_bucket_ptx, ensemble_aggregate_ptx, f32_hex, flipout_perturb_ptx, kl_gaussian_ptx,
        local_reparam_ptx, mc_dropout_mask_ptx, temp_scale_logits_ptx,
    };
    pub use crate::sparse::horseshoe::{
        HorseshoeConfig, HorseshoeFit, HorseshoeRegression, half_cauchy_log_pdf, horseshoe_log_pdf,
        ridge_regression, shrinkage_factor,
    };
    pub use crate::uncertainty::deep_ensemble::{DeepEnsemble, EnsembleStats};
    pub use crate::uncertainty::entropy::{
        aleatoric_entropy, epistemic_entropy, mutual_information, predictive_entropy,
    };
    pub use crate::uncertainty::evidential::{DirichletEvidence, NigEvidence, digamma, lgamma};
    pub use crate::uncertainty::functional_laplace::{FunctionalLaplace, FunctionalLaplaceConfig};
    pub use crate::uncertainty::laplace::LastLayerLaplace;
    pub use crate::uncertainty::mc_dropout::{McDropoutPredictor, mc_dropout_predict};
    pub use crate::uncertainty::swag::SwagPosterior;
    pub use crate::variational::elbo::{ElboConfig, elbo, iwae, kl_gaussian, kl_gaussian_vec};
    pub use crate::variational::flows::{PlanarFlow, RadialFlow};
    pub use crate::variational::hmc::{Hmc, HmcConfig, HmcResult, Nuts, NutsConfig, NutsResult};
    pub use crate::variational::iaf_flow::{IafFlow, IafStep, MadeNet, standard_normal_log_prob};
    pub use crate::variational::mean_field::MeanFieldDist;
    pub use crate::variational::nvae::{
        NVae, NVaeConfig, NVaeOutput, apply_free_bits, kl_gaussian_diag,
    };
    pub use crate::variational::reparam::{
        gaussian_log_prob, gaussian_sample, laplacian_log_prob, laplacian_sample,
        log_prob_gaussian_vec, sample_gaussian_vec, straight_through,
    };
    pub use crate::variational::vcl::{VclConfig, VclState};
    pub use crate::vi::advi::{Advi, AdviConfig, AdviModel, AdviResult, Transform};
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    /// Generate a synthetic over-confident classifier output: argmax is always
    /// class 0 (with margin 6) but only `acc_ratio` of true labels match.
    fn synthetic_overconfident(
        n: usize,
        n_classes: usize,
        acc_ratio: f32,
    ) -> (Vec<f32>, Vec<usize>) {
        let mut logits = Vec::with_capacity(n * n_classes);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            for k in 0..n_classes {
                logits.push(if k == 0 { 6.0 } else { 0.0 });
            }
            let frac = i as f32 / n as f32;
            labels.push(if frac < acc_ratio {
                0
            } else {
                1 + (i % n_classes.saturating_sub(1)).min(n_classes - 2)
            });
        }
        (logits, labels)
    }

    /// Apply softmax row-wise to a `[N, K]` buffer (returns fresh allocation).
    fn softmax_rows(logits: &[f32], k: usize) -> Vec<f32> {
        let n = logits.len() / k;
        let mut out = Vec::with_capacity(logits.len());
        for i in 0..n {
            let row = &logits[i * k..(i + 1) * k];
            let mut m = f32::NEG_INFINITY;
            for &v in row {
                if v > m {
                    m = v;
                }
            }
            let mut s = 0.0_f32;
            let mut tmp = Vec::with_capacity(k);
            for &v in row {
                let e = (v - m).exp();
                tmp.push(e);
                s += e;
            }
            let inv = 1.0_f32 / s;
            for v in tmp.iter_mut() {
                *v *= inv;
            }
            out.extend_from_slice(&tmp);
        }
        out
    }

    #[test]
    fn e2e_temperature_scaling_recalibrates_overconfident_classifier() {
        let n_classes = 3;
        let (logits, labels) = synthetic_overconfident(300, n_classes, 0.7);
        let scaler = TemperatureScaler::fit_default(&logits, &labels, n_classes)
            .expect("fit_default should succeed");
        let probs_before = softmax_rows(&logits, n_classes);
        let probs_after = scaler
            .apply(&logits, n_classes)
            .expect("apply should succeed");
        let (c_before, ok_before) = top1_confidences(&probs_before, &labels, n_classes)
            .expect("top1_confidences should succeed");
        let (c_after, ok_after) = top1_confidences(&probs_after, &labels, n_classes)
            .expect("top1_confidences should succeed");
        let ece_before = expected_calibration_error(&c_before, &ok_before, 10)
            .expect("expected_calibration_error should succeed");
        let ece_after = expected_calibration_error(&c_after, &ok_after, 10)
            .expect("expected_calibration_error should succeed");
        assert!(
            ece_after <= ece_before + 1e-4,
            "Temperature scaling should not worsen ECE (before={ece_before}, after={ece_after})"
        );
        // Argmax preserved → accuracy unchanged.
        let acc_before = ok_before.iter().filter(|&&x| x).count();
        let acc_after = ok_after.iter().filter(|&&x| x).count();
        assert_eq!(acc_before, acc_after);
    }

    #[test]
    fn e2e_isotonic_recalibrates_binary_scores() {
        let n = 100;
        let mut scores = Vec::new();
        let mut targets = Vec::new();
        for i in 0..n {
            let s = i as f32 / n as f32;
            scores.push(s);
            // True calibration: P(y=1|s) = s² (under-confident at mid range).
            targets.push((s * s).clamp(0.0, 1.0));
        }
        let labels: Vec<f32> = targets
            .iter()
            .map(|&p| if p > 0.5 { 1.0 } else { 0.0 })
            .collect();
        let r = IsotonicRegressor::fit(&scores, &labels).expect("fit should succeed");
        let preds = r.predict(&scores);
        for w in preds.windows(2) {
            assert!(w[0] <= w[1] + 1e-6, "isotonic must be non-decreasing");
        }
    }

    #[test]
    fn e2e_platt_scales_binary_logits() {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for i in 0..100 {
            let s = (i as f32 - 50.0) * 0.1;
            scores.push(s);
            labels.push(if s > 0.0 { 1_u8 } else { 0_u8 });
        }
        let p = PlattScaler::fit_default(&scores, &labels).expect("fit_default should succeed");
        assert!(p.predict_one(5.0) > p.predict_one(-5.0));
    }

    #[test]
    fn e2e_mc_dropout_quantifies_uncertainty() {
        let mut handle = BayesHandle::default_handle();
        // Simulate a stochastic model: y = base + ε (ε ~ N(0, σ²)). Each
        // closure call burns four LCG draws to decorrelate the components
        // (the Knuth LCG has visible correlation between consecutive normals).
        let base = [0.3_f32, 0.7];
        let stats = mc_dropout_predict(2048, handle.rng_mut(), |r| {
            let (e0, _) = r.next_normal_pair();
            let (e1, _) = r.next_normal_pair();
            Ok(vec![base[0] + 0.1 * e0, base[1] + 0.1 * e1])
        })
        .expect("value should be present");
        eprintln!(
            "MC Dropout mean=({:.3}, {:.3}), var=({:.4}, {:.4})",
            stats.mean[0], stats.mean[1], stats.variance[0], stats.variance[1]
        );
        // Variance is the principal outcome — should be on the order of 0.01.
        for v in &stats.variance {
            assert!(
                *v > 0.001 && *v < 0.05,
                "variance out of expected range: {v}"
            );
        }
        // Predictive mean should be in the [0, 1] band (sanity).
        assert!((0.0..=1.0).contains(&stats.mean[0]));
        assert!((0.0..=1.0).contains(&stats.mean[1]));
    }

    #[test]
    fn e2e_deep_ensemble_aggregates_disagreement() {
        let preds = vec![
            vec![0.9_f32, 0.05, 0.05],
            vec![0.05_f32, 0.9, 0.05],
            vec![0.05_f32, 0.05, 0.9],
        ];
        let ensemble = DeepEnsemble::new(preds).expect("new should succeed");
        let stats = ensemble
            .aggregate_probabilities()
            .expect("aggregate_probabilities should succeed");
        // Mean is ~ uniform — very high disagreement.
        for v in &stats.mean {
            assert!((v - 1.0 / 3.0).abs() < 0.01);
        }
        // Variance per class should be substantial.
        for v in &stats.variance {
            assert!(*v > 0.1);
        }
    }

    #[test]
    fn e2e_swag_posterior_sampling_round_trip() {
        let mut handle = BayesHandle::default_handle();
        let mut posterior = SwagPosterior::new(4, 3).expect("new should succeed");
        // Inject a few SGD-like iterates around mean (1, 2, 3, 4).
        for offset in [-0.1_f32, -0.05, 0.0, 0.05, 0.1] {
            let iterate: Vec<f32> = (0..4).map(|i| (i + 1) as f32 + offset).collect();
            posterior.update(&iterate).expect("update should succeed");
        }
        // Mean should be approximately (1, 2, 3, 4).
        for (i, &m) in posterior.mean.iter().enumerate() {
            assert!((m - (i + 1) as f32).abs() < 1e-4);
        }
        let theta = posterior
            .sample(handle.rng_mut())
            .expect("value should be present");
        assert_eq!(theta.len(), 4);
        for v in theta {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn e2e_laplace_widens_predictions_with_low_precision() {
        let map = vec![1.0_f32, -1.0];
        let phi: Vec<f32> = (0..30)
            .flat_map(|i| {
                let x = (i as f32 - 15.0) * 0.2;
                vec![x, x * 0.5]
            })
            .collect();
        let labels: Vec<u8> = (0..30)
            .map(|i| if (i as f32 - 15.0) * 0.2 > 0.0 { 1 } else { 0 })
            .collect();
        let laplace = LastLayerLaplace::fit_binary_logistic(&map, &phi, &labels, 1.0)
            .expect("fit_binary_logistic should succeed");
        let p = laplace
            .predictive_probability(&[2.0_f32, 1.0])
            .expect("predictive_probability should succeed");
        assert!((0.0..=1.0).contains(&p));
    }

    #[test]
    fn e2e_bald_finds_disagreement() {
        // 3 ensemble members on a 2-class problem disagreeing strongly.
        let samples = vec![
            0.95_f32, 0.05, // member 1
            0.05_f32, 0.95, // member 2
            0.5_f32, 0.5, // member 3
        ];
        let mi = mutual_information(&samples, 2, 3).expect("mutual_information should succeed");
        let ent = predictive_entropy(&samples, 2, 3).expect("predictive_entropy should succeed");
        let aleatoric =
            aleatoric_entropy(&samples, 2, 3).expect("aleatoric_entropy should succeed");
        assert!((ent - aleatoric - mi).abs() < 1e-5);
        assert!(mi > 0.0);
    }

    #[test]
    fn e2e_brier_and_nll_agree_on_perfect_predictor() {
        let probs = vec![1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let labels = vec![0_usize, 1, 2];
        let bs = brier_score(&probs, &labels, 3).expect("brier_score should succeed");
        let nll = negative_log_likelihood(&probs, &labels, 3)
            .expect("negative_log_likelihood should succeed");
        assert!(bs < 1e-5);
        assert!(nll < 1e-3);
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            for prog in [
                kl_gaussian_ptx(sm),
                mc_dropout_mask_ptx(sm),
                local_reparam_ptx(sm),
                ece_bucket_ptx(sm),
                ensemble_aggregate_ptx(sm),
                flipout_perturb_ptx(sm),
                temp_scale_logits_ptx(sm),
            ] {
                assert!(prog.contains(&format!("sm_{sm}")));
                assert!(prog.contains(".visible .entry"));
            }
        }
    }

    #[test]
    fn e2e_reliability_diagram_serialises_to_json() {
        let c = vec![0.1_f32, 0.5, 0.9];
        let ok = vec![false, true, true];
        let rd = reliability_diagram(&c, &ok, 5).expect("reliability_diagram should succeed");
        // Spot-check counts.
        assert_eq!(rd.bins.len(), 5);
        assert_eq!(rd.n_samples, 3);
        let total_count: usize = rd.bins.iter().map(|b| b.count).sum();
        assert_eq!(total_count, 3);
        // Use the diagram for ECE.
        let ece1 = rd.ece();
        let ece2 = expected_calibration_error(&c, &ok, 5)
            .expect("expected_calibration_error should succeed");
        assert!((ece1 - ece2).abs() < 1e-6);
    }
}
