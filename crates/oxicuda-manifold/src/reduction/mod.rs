//! Modern dimensionality-reduction methods.
//!
//! - [`mod@clustermap`]         ClusterMap (Damrich, Böhm, Hamprecht & Kobak 2022) — unified
//!   attraction/repulsion neighbour embedding that interpolates t-SNE / UMAP / ForceAtlas2
//!   via a generalised Cauchy kernel, tunable exponents and an annealed temperature.
//! - [`mod@trimap`]             Trimap (Wang et al., 2021) — global-structure-preserving embedding via
//!   weighted random triplets and logistic loss optimisation.
//! - [`mod@pacmap`]             PaCMAP (Wang et al., 2021) — pairwise controlled manifold approximation
//!   projection with three-phase optimisation (near, mid-near, far pairs).
//! - [`mod@poincare_embedding`] Poincaré Embeddings (Nickel & Kiela 2017) for hierarchical data.
//! - [`mod@parametric_tsne`]    Parametric t-SNE (van der Maaten 2009) — MLP encoder trained by
//!   KL(P‖Q) minimisation enabling out-of-sample embedding via the learned network.

pub mod clustermap;
pub mod pacmap;
pub mod parametric_tsne;
pub mod poincare_embedding;
pub mod trimap;

pub use clustermap::{ClusterMap, ClusterMapConfig, ClusterMapInit, ClusterMapPreset};
pub use pacmap::{PaCMapConfig, PaCMapInit, PaCMapResult, pacmap};
pub use parametric_tsne::{
    ParametricTsneConfig, ParametricTsneModel, parametric_tsne_fit, parametric_tsne_forward,
    parametric_tsne_transform,
};
pub use poincare_embedding::{
    PoincareConfig, PoincareModel, poincare_distance as poincare_embedding_distance,
    poincare_distances_all, poincare_fit, poincare_rank_relations,
};
pub use trimap::{TrimapConfig, TrimapInit, TrimapResult, trimap};
