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
    pub use crate::compression::topk::{error_feedback, topk_sparsify};
    pub use crate::error::{FedError, FedResult};
    pub use crate::handle::{FedHandle, LcgRng, SmVersion};
    pub use crate::privacy::gaussian::GaussianMechanism;
    pub use crate::privacy::laplacian::{LaplacianMechanism, add_laplacian_noise};
    pub use crate::privacy::moments::MomentsAccountant;
    pub use crate::privacy::pate::{PateConfig, data_dependent_epsilon, noisy_voting};
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
    pub use crate::selection::random::{random_select, stratified_select};
}
