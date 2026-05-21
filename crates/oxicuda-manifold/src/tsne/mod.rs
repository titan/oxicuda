//! t-SNE: t-distributed Stochastic Neighbor Embedding.

pub mod barnes_hut;
pub mod heavy_tsne;
pub mod nerv_jse;
pub mod perplexity;
pub mod tsne;

pub use barnes_hut::{Quad, QuadTree};
pub use heavy_tsne::{
    AlphaTsneConfig, HeavyTsneConfig, HeavyTsneResult, SsneConfig, SsneResult, alpha_tsne_fit,
    cauchy_tsne_fit, heavy_tsne_fit, ssne_fit,
};
pub use nerv_jse::{JseConfig, JseResult, NervConfig, NervResult, jse_fit, nerv_fit};
pub use perplexity::{compute_perplexity_p_matrix, p_row_from_distances};
pub use tsne::{TsneConfig, TsneResult, tsne_fit};
