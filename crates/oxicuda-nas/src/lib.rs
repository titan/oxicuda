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
//! ├── predictor/      — FLOP/param accountant, latency LUT/MLP, k-NN/RBF accuracy
//! ├── error           — NasError / NasResult
//! ├── handle          — NasHandle (SmVersion + LcgRng)
//! └── ptx_kernels     — GPU PTX kernel strings
//! ```

// ─── Module declarations ─────────────────────────────────────────────────────

pub mod darts;
pub mod error;
pub mod evolution;
pub mod handle;
pub mod ops;
pub mod predictor;
pub mod ptx_kernels;
pub mod supernet;

// ─── Prelude ─────────────────────────────────────────────────────────────────

/// Convenience re-exports for common neural architecture search types.
pub mod prelude {
    pub use crate::darts::bilevel::{BilevelConfig, BilevelOptimizer};
    pub use crate::darts::cell::DartsCell;
    pub use crate::darts::derive::{
        DiscretizedCell, DiscretizedNetwork, derive_discrete_cell, derive_network,
    };
    pub use crate::darts::network::DartsNetwork;
    pub use crate::error::{NasError, NasResult};
    pub use crate::evolution::encoding::ArchEncoding;
    pub use crate::evolution::nsga2::{
        Individual, crowding_distance, fast_non_dominated_sort, nsga2_select, tournament_select,
    };
    pub use crate::evolution::population::Population;
    pub use crate::handle::{LcgRng, NasHandle, SmVersion};
    pub use crate::ops::mixed_op::MixedOp;
    pub use crate::ops::primitives::{OpKind, OpWeights};
    pub use crate::ops::search_space::{CellSpace, NetworkSpace, SearchSpace};
    pub use crate::predictor::accuracy::{KnnAccuracyPredictor, RbfAccuracyPredictor};
    pub use crate::predictor::flops::{OpCost, op_cost, total_cost};
    pub use crate::predictor::latency::{LatencyLut, LatencyMlp};
    pub use crate::predictor::predictor_io::{ArchFeatures, LayerSpec};
    pub use crate::ptx_kernels::{
        arch_grad_ptx, arch_softmax_ptx, crossover_uniform_ptx, f32_hex, flops_accumulate_ptx,
        gumbel_softmax_ptx, mixed_op_blend_ptx, pareto_dominate_ptx,
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
        let cost = total_cost(&arch).unwrap();
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
        let total = lut.predict(&arch).unwrap();
        let expected = (1.0_f32 + 2.0 + 3.0 + 4.0) * 1e-4;
        assert!((total - expected).abs() < 1e-6);
    }

    #[test]
    fn e2e_latency_mlp_train_and_predict() {
        let mut handle = NasHandle::default_handle();
        let arch = sample_arch();
        let f = ArchFeatures::from_layers(&arch).unwrap();
        let in_dim = f.dim();
        let mut mlp = LatencyMlp::new(in_dim, 16, handle.rng_mut());
        // Synthetic single-target dataset.
        let samples: Vec<(Vec<f32>, f32)> = (0..32).map(|_| (f.data.clone(), 0.001_f32)).collect();
        let loss = mlp.fit(&samples, 200, 1e-5).unwrap();
        assert!(loss.is_finite());
        let pred = mlp.predict(&arch).unwrap();
        assert!(pred.is_finite());
    }

    #[test]
    fn e2e_knn_accuracy_predictor_round_trip() {
        let arch = sample_arch();
        let mut p = KnnAccuracyPredictor::new(3).unwrap();
        let f = ArchFeatures::from_layers(&arch).unwrap();
        for _ in 0..6 {
            p.add(f.data.clone(), 0.85).unwrap();
        }
        let q = p.predict(&arch).unwrap();
        assert!((q - 0.85).abs() < 1e-3);
    }

    #[test]
    fn e2e_rbf_accuracy_predictor_constant_target() {
        let arch = sample_arch();
        let f = ArchFeatures::from_layers(&arch).unwrap();
        let samples = vec![(f.data, 0.7_f32); 4];
        let p = RbfAccuracyPredictor::fit(&samples, 1.0, 1e-3).unwrap();
        let q = p.predict(&arch).unwrap();
        assert!((q - 0.7).abs() < 1e-2, "q = {q}");
    }
}
