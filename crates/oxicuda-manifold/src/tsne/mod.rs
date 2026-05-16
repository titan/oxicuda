//! t-SNE: t-distributed Stochastic Neighbor Embedding.

pub mod barnes_hut;
pub mod perplexity;
pub mod tsne;

pub use barnes_hut::{Quad, QuadTree};
pub use perplexity::{compute_perplexity_p_matrix, p_row_from_distances};
pub use tsne::{TsneConfig, TsneResult, tsne_fit};
