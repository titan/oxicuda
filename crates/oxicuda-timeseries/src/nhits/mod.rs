//! NHiTS — Neural Hierarchical Interpolation for Time Series.
//!
//! Implements the multi-rate stack architecture from Challu et al. 2022.
//! Each stack pools at a different rate so the model learns to decompose the
//! signal into frequency bands, producing both backcast (history explanation)
//! and forecast (future prediction) components.

pub mod multi_rate_sampler;
#[allow(clippy::module_inception)]
pub mod nhits;
pub mod nhits_block;

pub use multi_rate_sampler::MultiRateSampler;
pub use nhits::{NHits, NHitsConfig};
pub use nhits_block::NHitsBlock;
