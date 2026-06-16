//! Embedding models: parametric encoders and additional projection methods.
//!
//! - [`mod@parametric_umap`]    Parametric UMAP (Sainburg 2021) — neural encoder
//!   approximating a UMAP embedding.
//! - [`mod@sammon`]             Sammon mapping (Sammon 1969) — nonlinear MDS that
//!   emphasises preservation of short pairwise distances.
//! - [`mod@random_projection`]  Random projection (Johnson-Lindenstrauss;
//!   Achlioptas 2003) — Gaussian and sparse distance-preserving projections.
//! - [`mod@landmark_mds`]       Landmark MDS (de Silva & Tenenbaum 2004) — linear-time
//!   MDS via landmark triangulation.

pub mod landmark_mds;
pub mod parametric_umap;
pub mod random_projection;
pub mod sammon;

pub use landmark_mds::{LandmarkMdsConfig, LandmarkMdsResult, landmark_mds};
pub use random_projection::{
    RandomProjectionConfig, RandomProjectionKind, johnson_lindenstrauss_min_dim, random_projection,
};
pub use sammon::{SammonConfig, SammonResult, sammon};
