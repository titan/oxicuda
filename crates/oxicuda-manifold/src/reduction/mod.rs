//! Modern dimensionality-reduction methods.
//!
//! - [`mod@trimap`] Trimap (Wang et al., 2021) — global-structure-preserving embedding via
//!   weighted random triplets and logistic loss optimisation.
//! - [`mod@pacmap`] PaCMAP (Wang et al., 2021) — pairwise controlled manifold approximation
//!   projection with three-phase optimisation (near, mid-near, far pairs).

pub mod pacmap;
pub mod trimap;

pub use pacmap::{PaCMapConfig, PaCMapInit, PaCMapResult, pacmap};
pub use trimap::{TrimapConfig, TrimapInit, TrimapResult, trimap};
