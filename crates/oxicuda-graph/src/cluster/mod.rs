//! Clustering algorithms operating on graph structures.
//!
//! Currently provides spectral clustering via the normalised Laplacian.

pub mod spectral;

pub use spectral::{SpectralClustering, SpectralConfig};
