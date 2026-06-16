//! `oxicuda-manifold` — Manifold Learning, Dimensionality Reduction, and Riemannian Geometry.
//!
//! # Architecture
//!
//! ```text
//! oxicuda-manifold
//! ├── linear/      — PCA, Kernel PCA, FastICA
//! ├── tsne/        — t-SNE (perplexity, Barnes-Hut, gradient descent)
//! ├── umap/        — UMAP (kNN graph, fuzzy simplicial set, SGD embedding)
//! ├── local/       — LLE, MLLE, Isomap, Laplacian Eigenmaps
//! ├── diffusion/   — Diffusion Maps (Coifman-Lafon)
//! ├── mds/         — Classical MDS and SMACOF stress majorisation
//! ├── neighbor/    — Brute-force kNN, KD-tree, ball tree
//! ├── linalg/      — Jacobi eigendecomp, power iteration, Lanczos, Householder QR
//! ├── riemannian/  — Stiefel, Grassmann, SPD, Poincaré ball
//! ├── optim/       — Riemannian SGD with retractions
//! └── metrics/     — Trustworthiness, continuity, KL, neighbourhood preservation
//! ```
//!
//! All algorithms are implemented in pure Rust with no external linear-algebra dependencies.
//! Random sampling uses the workspace `LcgRng` (MMIX LCG with bit-32 boolean trick).

#![forbid(unsafe_code)]

pub mod autoencoder;
pub mod clustering;
pub mod diffusion;
pub mod embed;
pub mod error;
pub mod geometry;
pub mod handle;
pub mod linalg;
pub mod linear;
pub mod local;
pub mod mds;
pub mod metrics;
pub mod neighbor;
pub mod optim;
pub mod ptx_kernels;
pub mod reduction;
pub mod riemannian;
pub mod topology;
pub mod tsne;
pub mod umap;

pub use autoencoder::{
    EmbeddingExport, ManifoldHook, PcaManifoldHook, TsneRegHook, manifold_encode_and_export,
};
pub use clustering::kohonen_som::{
    KohonenSomConfig, KohonenSomResult, SomInit, kohonen_som_fit, som_grid_pos, som_predict,
    som_weight_at,
};
pub use diffusion::phate::{PhateConfig, PhateResult, phate_fit};
pub use embed::landmark_mds::{LandmarkMdsConfig, LandmarkMdsResult, landmark_mds};
pub use embed::random_projection::{
    RandomProjectionConfig, RandomProjectionKind, johnson_lindenstrauss_min_dim, random_projection,
};
pub use embed::sammon::{SammonConfig, SammonResult, sammon};
pub use error::{ManifoldError, ManifoldResult};
pub use handle::{LcgRng, ManifoldHandle, SmVersion};
pub use linear::cca_pls::{
    CcaConfig, CcaFit, PlsConfig, PlsFit, PlsSvdFit, cca_fit, cca_transform, pls_fit, pls_predict,
    pls_svd_fit, pls_transform,
};
pub use local::hessian_lle::hessian_lle;
pub use local::ltsa::ltsa;
pub use mds::nonmetric_mds::{NonmetricMdsResult, nonmetric_mds, pava};
pub use neighbor::hnsw::{
    HnswConfig, HnswDistance, HnswIndex, HnswSearchResult, hnsw_add, hnsw_build, hnsw_search,
};
pub use reduction::pacmap::{PaCMapConfig, PaCMapInit, PaCMapResult, pacmap};
pub use reduction::poincare_embedding::{
    PoincareConfig, PoincareModel, poincare_distances_all, poincare_fit, poincare_rank_relations,
};
pub use reduction::trimap::{TrimapConfig, TrimapInit, TrimapResult, trimap};
pub use riemannian::riemannian_median::{
    RiemannianMedianConfig, RiemannianMedianResult, riemannian_median, riemannian_median_objective,
    riemannian_trimmed_mean,
};
pub use riemannian::so_n::{
    so_2_rotation, so_n_check, so_n_distance, so_n_geodesic, so_n_identity, so_n_inner, so_n_log,
    so_n_norm, so_n_project_tangent, so_n_random, so_n_retract_cayley, so_n_retract_expm,
    so_n_retract_qr, so_n_riemannian_gradient,
};
pub use riemannian::spd_bures::{
    bures_distance, bures_exp, bures_frechet_mean, bures_geodesic, bures_geometric_mean, bures_log,
    spd_inv, spd_inv_sqrt, spd_sqrt,
};
pub use riemannian::spd_kmeans::{
    FrechetMeanConfig, FrechetMeanResult, SpdKmeansConfig, SpdKmeansResult, spd_frechet_mean,
    spd_kmeans,
};
pub use topology::persistent_homology::{
    MapperConfig, MapperGraph, MapperNode, PersistenceDiagram, PersistencePair, VietorisRipsConfig,
    bottleneck_distance, mapper, persistence_betti, vietoris_rips_persistence,
};
pub use tsne::heavy_tsne::{
    AlphaTsneConfig, HeavyTsneConfig, HeavyTsneResult, SsneConfig, SsneResult, alpha_tsne_fit,
    cauchy_tsne_fit, heavy_tsne_fit, ssne_fit,
};
pub use tsne::nerv_jse::{JseConfig, JseResult, NervConfig, NervResult, jse_fit, nerv_fit};
pub use umap::multiscale::{
    MultiScaleUmapConfig, MultiScaleUmapResult, combine_fuzzy_sets, multiscale_umap_fit,
};
pub use umap::supervised::{
    SupervisedUmapConfig, SupervisedUmapResult, UNLABELED, supervised_umap,
};

#[cfg(test)]
mod e2e_tests;
