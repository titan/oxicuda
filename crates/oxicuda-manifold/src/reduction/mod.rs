//! Modern dimensionality-reduction methods.
//!
//! - [`mod@trimap`]             Trimap (Wang et al., 2021) — global-structure-preserving embedding via
//!   weighted random triplets and logistic loss optimisation.
//! - [`mod@pacmap`]             PaCMAP (Wang et al., 2021) — pairwise controlled manifold approximation
//!   projection with three-phase optimisation (near, mid-near, far pairs).
//! - [`mod@poincare_embedding`] Poincaré Embeddings (Nickel & Kiela 2017) for hierarchical data.

pub mod pacmap;
pub mod poincare_embedding;
pub mod trimap;

pub use pacmap::{PaCMapConfig, PaCMapInit, PaCMapResult, pacmap};
pub use poincare_embedding::{
    PoincareConfig, PoincareModel, poincare_distance as poincare_embedding_distance,
    poincare_distances_all, poincare_fit, poincare_rank_relations,
};
pub use trimap::{TrimapConfig, TrimapInit, TrimapResult, trimap};
