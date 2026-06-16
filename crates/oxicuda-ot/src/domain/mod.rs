//! OT-based domain adaptation via barycentric mapping.
//!
//! Given a source dataset `X ∈ ℝ^{m × dim}` and a target dataset
//! `Y ∈ ℝ^{n × dim}`, the entropic OT plan `P` between empirical distributions
//! defines a soft assignment that we collapse into a barycentric map
//!
//! ```text
//! T(x_i) = Σ_j (P_ij / Σ_k P_ik) · y_j
//! ```
//!
//! mapping each source sample to a convex combination of target samples
//! (Courty et al., *Optimal Transport for Domain Adaptation*, IEEE TPAMI
//! 2017). The mapping is differentiable and preserves cluster structure.

/// OT-based feature flow for domain generalisation.
pub mod feature_flow;
/// Barycentric mapping for OT-based domain adaptation.
pub mod mapping;

pub use feature_flow::{FeatureFlowConfig, FeatureFlowResult, domain_discrepancy, ot_feature_flow};

/// Distributionally Robust Optimisation with Wasserstein uncertainty sets (Esfahani-Kuhn 2018).
pub mod dro_wasserstein;
pub use dro_wasserstein::{
    DroConfig, DroResult, DroSolver, dro_lipschitz_bound, dro_quadratic_loss, robustness_gap,
};

/// Entropic domain adaptation with a group-lasso label prior (Courty 2017).
pub mod entropic_da;
pub use entropic_da::{EntropicDaConfig, EntropicDaResult, lpl1_barycentric_map, sinkhorn_lpl1_mm};

/// Laplacian-regularised OT for domain adaptation (Courty 2014).
pub mod laplacian_ot;
pub use laplacian_ot::{
    LaplacianOtConfig, LaplacianOtResult, laplacian_barycentric_map, laplacian_ot,
};
