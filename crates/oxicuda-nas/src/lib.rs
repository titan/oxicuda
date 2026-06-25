//! `oxicuda-nas` — Neural Architecture Search primitives for OxiCUDA.
//!
//! Pure-Rust implementation of differentiable (DARTS), evolutionary (NSGA-II),
//! and one-shot (supernet) neural architecture search building blocks suitable
//! for CPU simulation and PTX kernel generation for GPU execution.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-nas
//! ├── darts/          — DartsCell/Network, BilevelOptimizer, derive discrete arch
//! ├── evolution/      — ArchEncoding, NSGA-II selection, Population
//! ├── ops/            — Primitives (8 DARTS ops), MixedOp, SearchSpace
//! ├── supernet/       — Weight-shared Supernet, PathSampler, SlimmableNet
//! ├── predictor/      — FLOP/param accountant, latency LUT/MLP, k-NN/RBF/GP/GNN accuracy
//! ├── proxy/          — Zero-cost proxies (NASWOT, SNIP, GraSP, SynFlow)
//! ├── controller/     — ENAS LSTM RL controller (REINFORCE + EMA baseline)
//! ├── error           — NasError / NasResult
//! ├── handle          — NasHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod controller;
pub mod darts;
pub mod error;
pub mod evolution;
pub mod handle;
pub mod ops;
pub mod predictor;
pub mod proxy;
pub mod ptx_kernels;
pub mod search;
pub mod supernet;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common neural architecture search types.
pub mod prelude {
    pub use crate::controller::enas::{EnasConfig, EnasController};
    pub use crate::darts::bilevel::{BilevelConfig, BilevelOptimizer};
    pub use crate::darts::cell::DartsCell;
    pub use crate::darts::darts_plus::{DartsPlusConfig, DartsPlusState};
    pub use crate::darts::derive::{
        DiscretizedCell, DiscretizedNetwork, derive_discrete_cell, derive_network,
    };
    pub use crate::darts::network::DartsNetwork;
    pub use crate::darts::pc_darts::{PcDarts, PcDartsConfig};
    pub use crate::error::{NasError, NasResult};
    pub use crate::evolution::encoding::ArchEncoding;
    pub use crate::evolution::nas_bench::{
        NasBenchCache, TrialResult, arch_key, arch_rng, derive_arch_seed,
    };
    pub use crate::evolution::nsga2::{
        Individual, crowding_distance, fast_non_dominated_sort, nsga2_select, tournament_select,
    };
    pub use crate::evolution::population::Population;
    pub use crate::evolution::regularized_evolution::{
        RegEvoConfig, RegEvoResult, RegularizedEvolution,
    };
    pub use crate::handle::{LcgRng, NasHandle, SmVersion};
    pub use crate::ops::mixed_op::MixedOp;
    pub use crate::ops::primitives::{OpKind, OpWeights};
    pub use crate::ops::search_space::{CellSpace, NetworkSpace, SearchSpace};
    pub use crate::ops::transformer_nas::{BlockSpec, TransformerArch, TransformerSearchSpace};
    pub use crate::predictor::accuracy::{KnnAccuracyPredictor, RbfAccuracyPredictor};
    pub use crate::predictor::bayesian_gp::{Acquisition, GaussianProcess, Kernel};
    pub use crate::predictor::flops::{OpCost, op_cost, total_cost};
    pub use crate::predictor::gnn_predictor::{
        CellTopology, GnnPredictor, PathEncodedPredictor, PathEncoder,
    };
    pub use crate::predictor::latency::{LatencyLut, LatencyMlp};
    pub use crate::predictor::predictor_io::{ArchFeatures, LayerSpec};
    pub use crate::proxy::jacobian_covariance::{
        JACOV_EPSILON, jacobian_covariance_score, pearson_correlation_matrix, symmetric_eigenvalues,
    };
    pub use crate::proxy::zero_cost::{
        NASWOT_RIDGE, ZeroCostProxy, grasp_score, naswot_score, rank_architectures, snip_score,
        synflow_score,
    };
    pub use crate::ptx_kernels::{
        arch_grad_ptx, arch_softmax_ptx, crossover_uniform_ptx, f32_hex, flops_accumulate_ptx,
        gumbel_softmax_ptx, mixed_op_blend_ptx, pareto_dominate_ptx,
    };
    pub use crate::search::darts_ops::{DartsConfig, DartsMixedOp};
    pub use crate::search::hat::{
        BlockLatencyLut, Candidate, HatConfig, HatResult, HatSearcher, pareto_front,
    };
    pub use crate::search::latency_predictor::{
        LatencyPredictor, latency_features, train_latency_predictor,
    };
    pub use crate::search::local_search::{
        ArchSpace, LocalSearchConfig, LocalSearchNas, SearchResult, single_op_neighbors,
    };
    pub use crate::search::successive_halving::{
        BracketResult, Hyperband, HyperbandConfig, HyperbandResult, RoundInfo, ShaConfig,
        ShaResult, SuccessiveHalving,
    };
    pub use crate::supernet::bignas::{BigNasConfig, BigNasSampler};
    pub use crate::supernet::once_for_all::{
        OfaBlockConfig, OfaSpace, OfaSubnet, OfaUnit, ShrinkPhase, ShrinkSchedule,
    };
    pub use crate::supernet::path_sample::{PathSampler, SamplingStrategy};
    pub use crate::supernet::slimmable::{BnStats, SlimmableNet, WIDTH_MULTIPLIERS};
    pub use crate::supernet::weight_share::Supernet;
}

// ─── End-to-end integration tests ────────────────────────────────────────────

#[cfg(test)]
mod e2e_tests {
    use crate::prelude::*;

    fn sample_arch() -> Vec<LayerSpec> {
        vec![
            LayerSpec::new(OpKind::SepConv3x3, 3, 16, 32, 32),
            LayerSpec::new(OpKind::SepConv3x3, 16, 32, 32, 32),
            LayerSpec::new(OpKind::AvgPool3x3, 32, 32, 32, 32),
            LayerSpec::new(OpKind::SepConv5x5, 32, 64, 16, 16),
        ]
    }

    #[test]
    fn e2e_flop_accountant_produces_finite_cost() {
        let arch = sample_arch();
        let cost = total_cost(&arch).expect("total_cost should succeed");
        assert!(cost.flops > 0);
        assert!(cost.params > 0);
    }

    #[test]
    fn e2e_latency_lut_calibrated_predict() {
        let arch = sample_arch();
        let mut lut = LatencyLut::new();
        // Per-layer measurement: deeper layers get higher latency.
        for (idx, layer) in arch.iter().enumerate() {
            lut.insert(layer, 1e-4 * (idx + 1) as f32);
        }
        let total = lut.predict(&arch).expect("predict should succeed");
        let expected = (1.0_f32 + 2.0 + 3.0 + 4.0) * 1e-4;
        assert!((total - expected).abs() < 1e-6);
    }

    #[test]
    fn e2e_latency_mlp_train_and_predict() {
        let mut handle = NasHandle::default_handle();
        let arch = sample_arch();
        let f = ArchFeatures::from_layers(&arch).expect("from_layers should succeed");
        let in_dim = f.dim();
        let mut mlp = LatencyMlp::new(in_dim, 16, handle.rng_mut());
        // Synthetic single-target dataset.
        let samples: Vec<(Vec<f32>, f32)> = (0..32).map(|_| (f.data.clone(), 0.001_f32)).collect();
        let loss = mlp.fit(&samples, 200, 1e-5).expect("fit should succeed");
        assert!(loss.is_finite());
        let pred = mlp.predict(&arch).expect("predict should succeed");
        assert!(pred.is_finite());
    }

    #[test]
    fn e2e_knn_accuracy_predictor_round_trip() {
        let arch = sample_arch();
        let mut p = KnnAccuracyPredictor::new(3).expect("new should succeed");
        let f = ArchFeatures::from_layers(&arch).expect("from_layers should succeed");
        for _ in 0..6 {
            p.add(f.data.clone(), 0.85)
                .expect("value should be present");
        }
        let q = p.predict(&arch).expect("predict should succeed");
        assert!((q - 0.85).abs() < 1e-3);
    }

    #[test]
    fn e2e_rbf_accuracy_predictor_constant_target() {
        let arch = sample_arch();
        let f = ArchFeatures::from_layers(&arch).expect("from_layers should succeed");
        let samples = vec![(f.data, 0.7_f32); 4];
        let p = RbfAccuracyPredictor::fit(&samples, 1.0, 1e-3).expect("fit should succeed");
        let q = p.predict(&arch).expect("predict should succeed");
        assert!((q - 0.7).abs() < 1e-2, "q = {q}");
    }
}
