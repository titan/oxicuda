//! `oxicuda-federated` — Federated learning primitives for OxiCUDA.
//!
//! Pure-Rust implementation of federated learning algorithms, communication
//! compression, differential-privacy mechanisms, secure aggregation, and
//! client-selection strategies suitable for CPU simulation and PTX kernel
//! generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-federated
//! ├── algorithm/      — FedAvg, FedProx, SCAFFOLD, FedAdam
//! ├── compression/    — PowerSGD, QSGD quantize, Random-K, Top-K sparsifiers
//! ├── privacy/        — Gaussian/Laplacian mechanisms, Moments/RDP accountants, PATE
//! ├── secure_agg/     — Shamir secret sharing, pairwise masking, secure aggregator
//! ├── selection/      — Client selection (random, stratified)
//! ├── error           — FedError / FedResult
//! ├── handle          — FedHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod algorithm;
pub mod compression;
pub mod error;
pub mod handle;
pub mod privacy;
pub mod ptx_kernels;
pub mod secure_agg;
pub mod selection;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common federated learning types.
pub mod prelude {
    pub use crate::algorithm::fedadam::{FedAdamState, ServerOptimizerKind};
    pub use crate::algorithm::fedavg::{FedAvgConfig, FedAvgState};
    pub use crate::algorithm::fedbuff::{BufferedUpdate, FedBuffConfig, FedBuffState};
    pub use crate::algorithm::fedprox::{
        FedProxConfig, fedprox_client_loss_correction, proximal_gradient, proximal_loss,
    };
    pub use crate::algorithm::scaffold::{
        ScaffoldClientState, ScaffoldState, scaffold_client_update, scaffold_server_aggregate,
    };
    pub use crate::compression::powersgd::{PowerSgdCompressor, frobenius_norm, residual};
    pub use crate::compression::quantize::{
        dequantize, gradient_norm, max_quantization_error, stochastic_quantize,
    };
    pub use crate::compression::randomk::{compression_ratio, random_sparsify};
    pub use crate::compression::sketch::{CountSketch, CountSketchConfig, RandomHadamard};
    pub use crate::compression::ternary::{
        TernaryCompressor, TernaryConfig, TernaryEncoded, TernaryMode,
    };
    pub use crate::compression::topk::{error_feedback, topk_sparsify};
    pub use crate::error::{FedError, FedResult};
    pub use crate::handle::{FedHandle, LcgRng, SmVersion};
    pub use crate::privacy::dp_ftrl::{DpFtrl, DpFtrlConfig, DpFtrlResult, DpFtrlState};
    pub use crate::privacy::gaussian::GaussianMechanism;
    pub use crate::privacy::laplacian::{LaplacianMechanism, add_laplacian_noise};
    pub use crate::privacy::moments::MomentsAccountant;
    pub use crate::privacy::pate::{PateConfig, data_dependent_epsilon, noisy_voting};
    pub use crate::privacy::randomized_response::{RandomizedResponse, RandomizedResponseConfig};
    pub use crate::privacy::rdp::{compose_rdp, optimal_epsilon, rdp_gaussian, rdp_to_dp};
    pub use crate::ptx_kernels::{
        aggregate_mean_ptx, dp_clip_gradient_ptx, fedavg_weighted_sum_ptx, gaussian_noise_ptx,
        pairwise_mask_ptx, qsgd_quantize_ptx, topk_mask_ptx,
    };
    pub use crate::secure_agg::aggregator::SecureAggregator;
    pub use crate::secure_agg::masking::{apply_mask, apply_pairwise_masks, generate_mask, unmask};
    pub use crate::secure_agg::shamir::{
        PRIME, ShamirConfig, reconstruct_gradient, reconstruct_scalar, share_gradient, share_scalar,
    };
    pub use crate::selection::power_of_choice::{
        PowerOfChoice, PowerOfChoiceConfig, SelectionStrategy,
    };
    pub use crate::selection::random::{random_select, stratified_select};
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    #[test]
    fn e2e_fedavg_aggregation_recovers_mean() {
        // 4 clients, 3 params each, equal weights → global = sample mean.
        let updates = vec![
            (vec![1.0_f32, 2.0, 3.0], 1.0_f32),
            (vec![3.0_f32, 4.0, 5.0], 1.0_f32),
            (vec![5.0_f32, 6.0, 7.0], 1.0_f32),
            (vec![7.0_f32, 8.0, 9.0], 1.0_f32),
        ];
        let mut state = FedAvgState::new(3);
        state.aggregate(&updates).unwrap();
        // Mean of (1,3,5,7), (2,4,6,8), (3,5,7,9) = 4, 5, 6
        assert!((state.global_params[0] - 4.0).abs() < 1e-5);
        assert!((state.global_params[1] - 5.0).abs() < 1e-5);
        assert!((state.global_params[2] - 6.0).abs() < 1e-5);
        assert_eq!(state.round, 1);
    }

    #[test]
    fn e2e_fedprox_proximal_term_decreases_distance() {
        let global = vec![0.0_f32, 0.0, 0.0];
        let local = vec![1.0_f32, 1.0, 1.0];
        let mu = 0.1;
        let loss = proximal_loss(&local, &global, mu).unwrap();
        // 0.5 · μ · ‖local − global‖² = 0.5 · 0.1 · 3 = 0.15
        assert!((loss - 0.15).abs() < 1e-5);
        let grad = proximal_gradient(&local, &global, mu).unwrap();
        for g in &grad {
            assert!((g - 0.1).abs() < 1e-6);
        }
    }

    #[test]
    fn e2e_topk_sparsify_with_error_feedback_compensates_loss() {
        let grad = vec![0.5_f32, -0.4, 0.3, -0.2, 0.1, -0.05, 0.02];
        let mut residual = vec![0.0_f32; grad.len()];
        let (sparse, _norm_lost) = topk_sparsify(&grad, 3).unwrap();
        // Update error-feedback residual.
        error_feedback(&mut residual, &grad, &sparse).unwrap();
        // Residual should hold the dropped entries → restoring them would recover grad.
        let mut combined = sparse.clone();
        for (c, &r) in combined.iter_mut().zip(residual.iter()) {
            *c += r;
        }
        for (a, &b) in combined.iter().zip(grad.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn e2e_qsgd_quantize_dequantize_unbiased() {
        use crate::compression::quantize::dequantize;
        let mut handle = FedHandle::default_handle();
        let grad = vec![0.1_f32, -0.2, 0.3, -0.4, 0.5];
        let s = 64_u32;
        let norm = gradient_norm(&grad);
        let mut sum = vec![0.0_f64; grad.len()];
        let trials = 1024;
        for _ in 0..trials {
            let q = stochastic_quantize(&grad, s, handle.rng_mut()).unwrap();
            let dq = dequantize(&q, norm, s).unwrap();
            for (acc, &v) in sum.iter_mut().zip(dq.iter()) {
                *acc += v as f64;
            }
        }
        // QSGD is unbiased in expectation; with 1024 trials, sample mean should
        // be within ‖g‖/s ≈ 0.012 with high probability — allow generous slack.
        for (acc, &g) in sum.iter().zip(grad.iter()) {
            let mean = (acc / trials as f64) as f32;
            assert!((mean - g).abs() < 0.1, "mean={mean}, g={g}");
        }
    }

    #[test]
    fn e2e_gaussian_dp_noise_adds_calibrated_variance() {
        let mut handle = FedHandle::default_handle();
        let mech = GaussianMechanism::new(1.0, 1.0, 1e-5).unwrap();
        let mut data = vec![0.0_f32; 1024];
        mech.add_noise(&mut data, handle.rng_mut()).unwrap();
        // Empirical std should be > 0.5 with such a budget.
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let var: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std = var.sqrt();
        assert!(std > 0.5, "DP noise std too small: {std}");
    }

    #[test]
    fn e2e_rdp_gaussian_composition_grows_linearly_in_steps() {
        let alpha = 2.0_f32;
        let sigma = 1.0_f32;
        let r1 = compose_rdp(alpha, sigma, 1).unwrap();
        let r10 = compose_rdp(alpha, sigma, 10).unwrap();
        assert!((r10 - 10.0 * r1).abs() < 1e-3);
        // Conversion to (ε, δ)-DP yields finite ε.
        let eps = rdp_to_dp(alpha, r10, 1e-5).unwrap();
        assert!(eps.is_finite() && eps > 0.0);
    }

    #[test]
    fn e2e_shamir_secure_aggregation_round_trip() {
        let mut handle = FedHandle::default_handle();
        let cfg = ShamirConfig::new(3, 5).unwrap();
        let secret = 12345_u64;
        let shares = share_scalar(secret, &cfg, handle.rng_mut()).unwrap();
        // Reconstruct from any 3 of 5 shares.
        let recovered = reconstruct_scalar(&shares[0..3], 3).unwrap();
        assert_eq!(recovered, secret);
        let recovered2 = reconstruct_scalar(&shares[2..5], 3).unwrap();
        assert_eq!(recovered2, secret);
    }

    #[test]
    fn e2e_shamir_gradient_round_trip() {
        let mut handle = FedHandle::default_handle();
        let cfg = ShamirConfig::new(3, 5).unwrap();
        let grad = vec![1.5_f32, -2.5, 0.0, 0.001, -100.0];
        let shares = share_gradient(&grad, &cfg, handle.rng_mut()).unwrap();
        // Reconstruct from a 3-of-5 subset of shares per element.
        let subset: Vec<Vec<(usize, u64)>> = shares.iter().map(|s| s[..3].to_vec()).collect();
        let recovered = reconstruct_gradient(&subset, 3).unwrap();
        for (a, &b) in recovered.iter().zip(grad.iter()) {
            assert!((a - b).abs() < 1e-2, "recovered {a} != {b}");
        }
    }

    #[test]
    fn e2e_random_select_returns_unique_indices() {
        let mut handle = FedHandle::default_handle();
        let selected = random_select(20, 5, handle.rng_mut()).unwrap();
        assert_eq!(selected.len(), 5);
        let mut copy = selected.clone();
        copy.sort_unstable();
        copy.dedup();
        assert_eq!(copy.len(), 5);
    }

    #[test]
    fn e2e_ptx_kernels_all_sm_versions() {
        for sm in [75_u32, 80, 86, 90, 100, 120] {
            for prog in [
                aggregate_mean_ptx(sm),
                dp_clip_gradient_ptx(sm),
                fedavg_weighted_sum_ptx(sm),
                gaussian_noise_ptx(sm),
                pairwise_mask_ptx(sm),
                qsgd_quantize_ptx(sm),
                topk_mask_ptx(sm),
            ] {
                assert!(prog.contains(&format!("sm_{sm}")));
                assert!(prog.contains(".visible .entry"));
            }
        }
    }
}
