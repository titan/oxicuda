//! iTransformer: inverted attention over the variate axis.

pub mod inverted_block;
#[allow(clippy::module_inception)]
pub mod itransformer;

pub use inverted_block::InvertedBlock;
pub use itransformer::{ITransformer, ITransformerConfig};
