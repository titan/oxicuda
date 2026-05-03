//! Time-series decomposition into trend and seasonal components.

pub mod moving_avg;
pub mod series_decomp;

pub use moving_avg::MovingAvg;
pub use series_decomp::{DecompResult, SeriesDecomp};
