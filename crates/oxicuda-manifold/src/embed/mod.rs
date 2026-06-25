//! Embedding models: parametric encoders and additional projection methods.
//!
//! - [`mod@isomap`]             Isomap (Tenenbaum, de Silva & Langford 2000) — config-struct
//!   ([`IsomapConfig`]) API wrapping the geodesic-distance + classical-MDS pipeline implemented
//!   in [`crate::local::isomap`].
//! - [`mod@parametric_umap`]    Parametric UMAP (Sainburg 2021) — neural encoder
//!   approximating a UMAP embedding.
//! - [`mod@sammon`]             Sammon mapping (Sammon 1969) — nonlinear MDS that
//!   emphasises preservation of short pairwise distances.
//! - [`mod@random_projection`]  Random projection (Johnson-Lindenstrauss;
//!   Achlioptas 2003) — Gaussian and sparse distance-preserving projections.
//! - [`mod@landmark_mds`]       Landmark MDS (de Silva & Tenenbaum 2004) — linear-time
//!   MDS via landmark triangulation.

pub mod isomap;
pub mod landmark_mds;
pub mod parametric_umap;
pub mod random_projection;
pub mod sammon;

// NOTE on the `isomap` name: a separate, already-wired `crate::local::isomap` module exists.
// That module exports `IsomapResult` / `isomap_fit`; this one exports `IsomapConfig` / `isomap`.
// The names are disjoint, the module (`embed::isomap`, type namespace) and the re-exported `isomap`
// free function (value namespace) coexist exactly like the sibling `sammon` module/function pair,
// and `local::isomap` is never re-exported at the crate root — so no ambiguity or glob clash arises.
pub use isomap::{IsomapConfig, isomap};
pub use landmark_mds::{LandmarkMdsConfig, LandmarkMdsResult, landmark_mds};
pub use random_projection::{
    RandomProjectionConfig, RandomProjectionKind, johnson_lindenstrauss_min_dim, random_projection,
};
pub use sammon::{SammonConfig, SammonResult, sammon};
